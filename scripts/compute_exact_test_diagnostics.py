#!/usr/bin/env python3
"""Experiment 5 re-run — recover per-instance exact-solver runtimes on the test split.

The main run persisted per-instance exact runtimes for only some targets. For the slow-exact
targets that the reviewer asked about, four are missing them (motif_bridge, geometric_cluster_cover,
decoy_complement, latent_class). This re-evaluates the reference time-limited exact backend on
each target's 500 held-out TEST instances, capped at the main run's 10 s budget, and writes the
per-instance diagnostics jsonl (`report/<backend>_test_diagnostics.jsonl`, fields
instance_id / native_runtime_ms / wall_clock_ms) that Exp 5's distribution analysis reads.

Safe: writes only the diagnostics jsonl; does NOT touch baseline_test.json or synthesis_summary.
Resumable per (target, backend): skips any whose diagnostics jsonl already has >= test_size lines.
Config (10 s native/gurobi/external limits) is reconstructed from each target's run_manifest.json;
external solver_config_path is nulled so package backends are used (no missing-file crash).

Detached, disconnect-proof launch:
  tmux new -d -s exp5 \
    'uv run python -m scripts.compute_exact_test_diagnostics 2>&1 | tee -a results/exp5_runtime_dist/exact_diag.log'
Watch:  tail -f results/exp5_runtime_dist/exact_diag.log   |   Resume: re-run the same command.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import os
from pathlib import Path

MAIN_RUN = "artifacts/second_scale_benchmark_v2/20260427_230552"
CONDITION = "seconds_scale_v2"
# target -> the exact backend the paper's Table 10 reports (fastest-at-best-quality PER CLASS):
#   MIS = kamis_vc_exact, MDS = scip_mip_exact, MDKP = cbc_mdkp_exact.
# (cpsat_exact was the WRONG backend — a deliberately-weak OR-Tools solver that times out.)
# kamis has no binary on this server; mis/clique_path already has kamis diagnostics from the main
# run, and mis/motif_bridge cannot be reproduced here (reported from the aggregate instead).
DEFAULT_TASKS = {
    "mds/geometric_cluster_cover_v1": ["scip_mip_exact"],
    "mdkp/decoy_complement_mixture_v1": ["cbc_mdkp_exact"],
    "mdkp/latent_class_knapsack_v1": ["cbc_mdkp_exact"],  # already have this from the prior run
}


def _worker(target: str, backend: str, main_run_str: str, timeout: float) -> tuple:
    import dataclasses
    import json

    os.environ["DASBENCH_SOLVER_TIMEOUT_SECONDS"] = str(timeout)  # 10 s cap, per process
    from dasbench.cli import _resolve_single_baseline
    from dasbench.data import load_manifest, load_split
    from dasbench.eval import evaluate_solver
    from dasbench.integrations import ExternalExactConfig, GurobiBaselineConfig, NativeExactConfig

    main_run = Path(main_run_str)
    tdir = main_run / "targets" / CONDITION / target
    ar, ds, rep = tdir / "agent_run", tdir / "dataset", tdir / "report"
    diag = rep / f"{backend}_test_diagnostics.jsonl"
    # Completion marker: diagnostics only hold SOLVED instances (timeouts get no row, so a
    # timeout-heavy target legitimately has <num_test / even 0 lines) -- can't key resume on
    # line count. The .done marker records that the eval finished (solved / total).
    marker = rep / f"{backend}_test_diagnostics.done"
    if marker.is_file():
        return (target, backend, f"skip ({marker.read_text().strip()})", None)

    man = json.loads((ar / "run_manifest.json").read_text())
    g = GurobiBaselineConfig.from_record(man.get("gurobi_baseline"))
    nx = NativeExactConfig.from_record(man.get("native_exact_baselines"))
    ex = dataclasses.replace(ExternalExactConfig.from_record(man.get("external_exact_baselines")), solver_config_path=None)
    prob = str(load_manifest(ds)["problem"])
    try:
        solver = _resolve_single_baseline(prob, backend, output_dir=ar, gurobi_config=g, native_exact_config=nx, external_config=ex)
        instances = load_split(ds, "test")
        rep.mkdir(parents=True, exist_ok=True)
        summary = evaluate_solver(prob, backend, solver, instances, split="test", diagnostics_path=diag)
    except Exception as exc:  # noqa: BLE001
        return (target, backend, f"ERROR: {type(exc).__name__}: {exc}", None)
    solved = sum(1 for _ in diag.open()) if diag.is_file() else 0
    marker.write_text(f"solved {solved}/{len(instances)}  mean_rt_ms={summary.get('average_runtime_ms')}")
    return (target, backend, f"done ({solved}/{len(instances)} solved)", summary.get("average_runtime_ms"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-run", default=MAIN_RUN)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--solver-timeout-seconds", type=float, default=10.0)
    ap.add_argument("--targets", nargs="*", default=None, help="Subset of targets (default: all 4 missing).")
    args = ap.parse_args()

    tasks = [(t, b) for t, bs in DEFAULT_TASKS.items()
             if (args.targets is None or t in args.targets) for b in bs]
    os.environ["DASBENCH_SOLVER_TIMEOUT_SECONDS"] = str(args.solver_timeout_seconds)
    print(f"Exp 5 exact re-run: {len(tasks)} (target,backend) tasks, max_workers={args.max_workers}, "
          f"cap={args.solver_timeout_seconds}s", flush=True)
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers, mp_context=ctx) as ex:
        futs = {ex.submit(_worker, t, b, args.main_run, args.solver_timeout_seconds): (t, b) for t, b in tasks}
        for fut in concurrent.futures.as_completed(futs):
            t, b = futs[fut]
            try:
                target, backend, status, rt = fut.result()
                print(f"  [{status.split()[0].upper():5s}] {target} / {backend}"
                      + (f"  mean_rt={rt:.1f}ms" if rt else f"  {status}"), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  [FATAL] {t} / {b}: {exc}", flush=True)
    print("All tasks processed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
