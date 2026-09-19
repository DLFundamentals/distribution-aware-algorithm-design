"""Exp 3 — PACE hitting-set quality-time frontier.

Sweeps a shared wall-clock budget across the released PACE-2025 HS solvers (root, greeduce,
shadoks) on the SAME private test instances, so each solver's quality-vs-time curve is visible
and our synthesized solver's point can be placed on it.

Budget is enforced exactly as in the PACE competition: the solver reads the instance on stdin
and runs until SIGTERM at the deadline, then flushes its best-so-far solution to stdout (a short
grace window allows the flush; too-slow solvers get SIGKILLed and count as no-output). Reported
runtime is the FULL wall clock measured by the harness (startup + solve + flush), matching how we
timed Gurobi/KaMIS elsewhere -- we do NOT trust any solver-reported time.

Reuses an already-built dataset dir (no re-annotation); only re-runs the baseline solvers per
budget. Optionally restricts to the first --test-count instances via a truncated dataset copy.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

from benchmarks.pace_competitions import (
    COMPETITIONS,
    DEFAULT_SOLVER_ROOT,
    _resolve_solvers,
    run_baselines,
)


def _subset_dataset(src: Path, dst: Path, test_count: int) -> Path:
    """Copy a built dataset, truncating test.jsonl to the first `test_count` instances."""
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "benchmark_spec.json", "reproducibility.json",
                 "train.jsonl", "validation.jsonl"):
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)
    lines = (src / "test.jsonl").read_text(encoding="utf-8").splitlines()
    kept = lines[:test_count]
    (dst / "test.jsonl").write_text("\n".join(kept) + "\n", encoding="utf-8")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", default="pace2025_hs")
    ap.add_argument("--dataset-dir", type=Path, required=True,
                    help="Existing built dataset dir (reused, not re-annotated).")
    ap.add_argument("--output-base", type=Path, required=True)
    ap.add_argument("--solvers", default="root,greeduce,shadoks")
    ap.add_argument("--budgets", default="3,10,30",
                    help="Comma-separated wall-clock budgets in seconds.")
    ap.add_argument("--grace-seconds", type=float, default=15.0)
    ap.add_argument("--test-count", type=int, default=0,
                    help="If >0, restrict to the first N test instances (subset copy).")
    ap.add_argument("--solver-root", type=Path, default=DEFAULT_SOLVER_ROOT)
    ap.add_argument("--arg-time-limit", action="store_true",
                    help="Solvers self-terminate: pass the budget as argv[1] (not via SIGTERM) and "
                         "give the harness a generous timeout = budget + --timeout-margin. Use for "
                         "solvers like fontanf that ignore SIGTERM and write only on completion.")
    ap.add_argument("--timeout-margin", type=float, default=180.0,
                    help="With --arg-time-limit, extra wall-clock the harness allows beyond the "
                         "budget (for instance read + solution write on large instances).")
    args = ap.parse_args()

    config = COMPETITIONS[args.competition]
    budgets = [float(x) for x in args.budgets.split(",") if x.strip()]

    dataset_dir = args.dataset_dir
    if args.test_count and args.test_count > 0:
        dataset_dir = _subset_dataset(
            args.dataset_dir, args.output_base / f"dataset_first{args.test_count}", args.test_count)

    resolve_args = argparse.Namespace(
        solver=[], solvers=args.solvers, install_solvers=False,
        force_build=False, solver_root=args.solver_root)
    solvers = _resolve_solvers(config, resolve_args)
    print(f"solvers: {[s.name for s in solvers]}  budgets: {budgets}s  grace: {args.grace_seconds}s")
    print(f"dataset: {dataset_dir}")

    frontier: dict[str, list[dict[str, object]]] = {s.name: [] for s in solvers}
    for budget in budgets:
        out = args.output_base / f"budget_{int(budget)}s"
        print(f"\n=== budget {budget}s -> {out} ===", flush=True)
        if args.arg_time_limit:
            # Solver self-terminates at its own time limit passed as argv[1]; harness only guards
            # against a runaway with a generous timeout (budget + margin for large-instance I/O).
            budget_solvers = [
                dataclasses.replace(s, command=[*s.command, str(budget)]) for s in solvers]
            timeout = budget + args.timeout_margin
        else:
            budget_solvers = solvers
            timeout = budget
        summary = run_baselines(
            config=config, dataset_dir=dataset_dir, solvers=budget_solvers,
            output_dir=out, timeout_seconds=timeout, grace_seconds=args.grace_seconds)
        for name, payload in summary["solvers"].items():
            row = {
                "budget_seconds": budget,
                "valid_count": payload["valid_count"],
                "num_instances": payload["num_instances"],
                "avg_objective_value": payload["average_objective_value"],
                "avg_normalized_quality": payload["average_normalized_quality"],
                "avg_runtime_ms": payload["average_runtime_ms"],
                "timeout_count": payload["timeout_count"],
            }
            frontier.setdefault(name, []).append(row)
            print(f"  {name}: valid={row['valid_count']}/{row['num_instances']} "
                  f"obj={row['avg_objective_value']:.1f} q={row['avg_normalized_quality']:.4f} "
                  f"rt={row['avg_runtime_ms']:.0f}ms", flush=True)

    frontier_path = args.output_base / "frontier_summary.json"
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    frontier_path.write_text(json.dumps(frontier, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nfrontier summary: {frontier_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
