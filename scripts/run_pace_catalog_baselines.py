#!/usr/bin/env python3
"""Evaluate the registry catalog baselines on a PACE competition dataset.

Only PACE 2025 Dominating Set currently carries a full baseline comparison; the
other competitions have released-solver numbers but no catalog or optimization
baseline. This fills that in for a competition whose dataset is already built,
without running synthesis.

Gurobi is off by default. `build_gurobi_solver` dispatches on seven problem
names and raises for hitting_set, ocm and dfvs, so asking for it on those
competitions fails the whole resolve step rather than skipping one baseline.
Pass --gurobi once a formulation exists for the problem.

Checkpoints after every baseline and skips ones already recorded, so an
interrupted run resumes where it stopped. Writes only `baseline_<split>.json`
in the run directory; nothing else is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dasbench.data import load_manifest, load_split  # noqa: E402
from dasbench.eval import evaluate_solver, write_summary  # noqa: E402
from dasbench.eval.baselines import resolve_baselines  # noqa: E402
from dasbench.integrations import (  # noqa: E402
    ExternalExactConfig,
    GurobiBaselineConfig,
    NativeExactConfig,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--run-output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--gurobi", action="store_true", help="Enable the Gurobi baseline (problem must support it).")
    parser.add_argument("--time-limit-seconds", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--solver-timeout-seconds", type=float, default=360.0,
                        help="Per-instance cap. PACE graphs are large; a baseline that cannot "
                             "finish one instance should not stall the whole run.")
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.run_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"baseline_{args.split}.json"

    manifest = load_manifest(dataset_dir)
    problem = str(manifest["problem"])
    instances = load_split(dataset_dir, args.split)
    print(f"problem={problem}  split={args.split}  instances={len(instances)}", flush=True)

    gurobi_config = GurobiBaselineConfig(
        enabled=bool(args.gurobi), time_limit_seconds=args.time_limit_seconds, threads=args.threads
    )
    try:
        baselines, discovery = resolve_baselines(
            problem,
            gurobi_config=gurobi_config,
            native_exact_config=NativeExactConfig(time_limit_seconds=args.time_limit_seconds),
            external_config=ExternalExactConfig(),
        )
    except ValueError as exc:
        print(f"ERROR resolving baselines for {problem}: {exc}", flush=True)
        if args.gurobi:
            print("  (Gurobi does not support this problem -- drop --gurobi or add a formulation.)", flush=True)
        return 2

    existing: dict[str, object] = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    print(f"baselines to run: {sorted(baselines)}  (already recorded: {sorted(existing)})", flush=True)

    failed = 0
    for name in sorted(baselines):
        if name in existing:
            print(f"  skip {name} (already recorded)", flush=True)
            continue
        started = time.perf_counter()
        try:
            summary = evaluate_solver(
                problem, name, baselines[name], instances,
                split=args.split, timeout_seconds=args.solver_timeout_seconds,
            )
            existing[name] = summary
            print(f"  ok   {name:24s} feas={summary.get('feasibility_rate')} "
                  f"q={summary.get('average_normalized_quality')} "
                  f"obj={summary.get('average_objective_value')} "
                  f"rt={summary.get('average_runtime_ms'):.1f}ms "
                  f"[{time.perf_counter()-started:.0f}s]", flush=True)
        except Exception as exc:  # noqa: BLE001 - one bad baseline must not lose the others
            failed += 1
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        write_summary(out_path, existing)  # checkpoint after every baseline

    write_summary(out_dir / "external_exact_discovery.json", discovery)
    print(f"wrote {out_path} with {len(existing)} baselines; {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
