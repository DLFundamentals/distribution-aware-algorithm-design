#!/usr/bin/env python3
"""Fill the missing VALIDATION-split baseline metrics for Experiment 1.

Seven main-run targets (mdkp x3, mis/core_fringe_trap, tsp x3) have an empty
`baseline_validation.json` -- baselines were scored on test but not validation. The Exp 1
portfolio selector must fit on validation, so this evaluates the SAME catalog baselines on
each target's 32 validation instances, writing `baseline_validation.json` in place.

Design (safe + resumable, for running detached across an SSH disconnect):
  - Evaluates ONLY the baseline names already present in that target's `baseline_test.json`
    (the fixed classical/exact/Gurobi catalog) -- it does NOT call the full `resolve_baselines`
    (which also builds/optimizes ML baselines and does external discovery), so no `ml_` entries
    and no surprise slowness. Each baseline is built individually via `_resolve_single_baseline`.
  - NEVER touches `baseline_test.json` or `synthesis_summary.json`; it only writes validation.
  - Checkpoints after EVERY baseline (writes `baseline_validation.json` each time) and skips
    baselines already recorded, so an interrupted run resumes exactly where it stopped.
  - Config (Gurobi/native/external time limits, threads) is reconstructed from each target's
    own `run_manifest.json`, so validation matches how test was computed (10 s limits).

Detached, disconnect-proof launch (tmux keeps running server-side after you disconnect):
  tmux new -d -s exp1val \
    'cd "$REPO" && uv run python -m scripts.compute_missing_val_baselines \
       2>&1 | tee -a results/exp1_portfolio/val_baselines.log'
Watch it:   tail -f results/exp1_portfolio/val_baselines.log     (or: tmux attach -t exp1val)
Resume:     re-run the exact same command -- finished baselines/targets are skipped.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from dasbench.cli import _resolve_single_baseline
from dasbench.data import load_manifest, load_split
from dasbench.eval import evaluate_solver, write_summary
from dasbench.eval.baselines import external_diagnostics_path, gurobi_diagnostics_path
from dasbench.integrations import ExternalExactConfig, GurobiBaselineConfig, NativeExactConfig

DEFAULT_MAIN_RUN = "artifacts/second_scale_benchmark_v2/20260427_230552"
CONDITION = "seconds_scale_v2"
DEFAULT_TARGETS = [
    "mdkp/decoy_complement_mixture_v1",
    "mdkp/latent_class_knapsack_v1",
    "mdkp/single_resource_density_v1",
    "mis/core_fringe_trap_v1",
    "tsp/clustered_euclidean_v1",
    "tsp/latent_metric_mixture_v1",
    "tsp/paired_ribbon_zigzag_v1",
]


def _load_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size < 2:
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _diagnostics_path(output_dir: Path, baseline_name: str, gurobi_name: str) -> Path | None:
    if baseline_name == gurobi_name:
        return gurobi_diagnostics_path(output_dir, split="validation", baseline_name=baseline_name)
    if baseline_name.endswith("_exact"):
        return external_diagnostics_path(output_dir, split="validation", baseline_name=baseline_name)
    return None


def process_target(main_run: Path, target: str) -> None:
    ar = main_run / "targets" / CONDITION / target / "agent_run"
    ds = main_run / "targets" / CONDITION / target / "dataset"
    val_path = ar / "baseline_validation.json"

    test_summaries = _load_json(ar / "baseline_test.json")
    catalog = [b for b in test_summaries if not b.startswith("ml_")]  # fixed classical catalog
    if not catalog:
        print(f"[WARN] {target}: no baseline_test.json entries; skipping", flush=True)
        return

    val_summaries = _load_json(val_path)
    pending = [b for b in catalog if b not in val_summaries]
    if not pending:
        print(f"[SKIP] {target}: validation already complete ({len(val_summaries)}/{len(catalog)})", flush=True)
        return

    manifest = _load_json(ar / "run_manifest.json")
    gcfg = GurobiBaselineConfig.from_record(manifest.get("gurobi_baseline"))
    ncfg = NativeExactConfig.from_record(manifest.get("native_exact_baselines"))
    ecfg = ExternalExactConfig.from_record(manifest.get("external_exact_baselines"))
    # The main run's `baselines/external_solvers.json` (local binary paths) is not in this tree;
    # null the path so external exacts fall back to auto/package backends (pyscipopt, highspy)
    # instead of crashing on the missing file. CLI-only externals without binaries here
    # (KaMIS/Concorde/LKH/CBC) simply won't build and are skipped + logged.
    ecfg = dataclasses.replace(ecfg, solver_config_path=None)
    problem = str(load_manifest(ds)["problem"])
    instances = load_split(ds, "validation")  # full instances (with stored optima) for scoring

    print(f"[RUN ] {target}: {len(val_summaries)}/{len(catalog)} done, {len(pending)} pending "
          f"({len(instances)} val instances)", flush=True)
    for name in pending:
        t0 = time.perf_counter()
        try:
            solver = _resolve_single_baseline(
                problem, name, output_dir=ar,
                gurobi_config=gcfg, native_exact_config=ncfg, external_config=ecfg,
            )
            summary = evaluate_solver(
                problem, name, solver, instances,
                split="validation",
                diagnostics_path=_diagnostics_path(ar, name, gcfg.baseline_name),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    [!] {name}: FAILED to build/evaluate ({type(exc).__name__}: {exc}); skipping", flush=True)
            continue
        val_summaries[name] = summary
        write_summary(val_path, val_summaries)  # checkpoint after each baseline
        dt = time.perf_counter() - t0
        print(f"    [+] {name}: q={summary.get('average_normalized_quality'):.4f} "
              f"rt_ms={summary.get('average_runtime_ms'):.1f} ({dt:.1f}s)", flush=True)
    print(f"[DONE] {target}: validation {len(val_summaries)}/{len(catalog)}", flush=True)


def main() -> int:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--main-run", default=DEFAULT_MAIN_RUN)
    ap.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    ap.add_argument(
        "--solver-timeout-seconds",
        type=float,
        default=10.0,
        help="Per-instance evaluation cap, matching the main run's 10 s baseline budget. This is "
             "REQUIRED for faithfulness: anytime heuristics (e.g. multi_start_two_opt) otherwise "
             "run to their natural budget and score inflated validation quality vs the capped test.",
    )
    args = ap.parse_args()
    # Cap every baseline at the same 10 s the main-run test used (evaluator SIGALRM timeout),
    # so validation metrics are comparable to test. Overridable via --solver-timeout-seconds.
    os.environ["DASBENCH_SOLVER_TIMEOUT_SECONDS"] = str(args.solver_timeout_seconds)
    print(f"DASBENCH_SOLVER_TIMEOUT_SECONDS={os.environ['DASBENCH_SOLVER_TIMEOUT_SECONDS']}", flush=True)
    main_run = Path(args.main_run)
    print(f"Filling validation baselines for {len(args.targets)} targets under {main_run}", flush=True)
    for target in args.targets:
        process_target(main_run, target)
    print("All targets processed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
