"""Seed-variance benchmark (rebuttal Experiment 6a).

Runs the full synthesis pipeline N independent times per target to quantify run-to-run
variance of the whole agent. Each "run" is one seed condition (`seed_01`, `seed_02`, ...):
identical config and identical (reused) datasets, so runs differ ONLY in the model's
sampling stochasticity -- there is no pinned decode seed, which is the point.

By default this reproduces the local-model sub-part: `gemma-4-31b-it` via the `agent`
generator (env in `.env.gemma4`), the paper's search config (beam / iterations /
candidate-width = 3/3/3, train=64 / val=32 / test=500), datasets reused from the frozen
main run, and baselines/report skipped (baselines are already cached; only the selected
solver's held-out test metrics are needed). Aggregate the finished sweep into a
mean +/- std table with `python -m scripts.collect_seed_variance_results --sweep-root <sweep dir>`.

Idiomatic to the other `benchmarks/*` sweeps: builds `SweepJob`s via `jobs_from_conditions`
and executes them with `run_sweep`, so it inherits resumability (a run is "done" once its
`synthesis_summary.json` exists), `--max-workers` parallelism, and `--env-file` handling.

Examples:
  # Seeds 1-5 across all 21 targets on the local Gemma4 server:
  python -m benchmarks.seed_variance_benchmark --num-runs 5

  # Extend the same experiment with seeds 6-10 later (resumable, additive):
  python -m benchmarks.seed_variance_benchmark --num-runs 5 --start-run 6 \
      --sweep-id <existing_sweep_id>

  # Narrow to one target, quick smoke:
  python -m benchmarks.seed_variance_benchmark --problem tsp --family paired_ribbon_zigzag_v1 \
      --num-runs 1

  # Preview jobs without running:
  python -m benchmarks.seed_variance_benchmark --num-runs 5 --dry-run
"""
from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.common import (
    add_common_arguments,
    jobs_from_conditions,
    resolve_sweep_artifact_root,
    run_sweep,
    selected_targets,
)
from dasbench.artifacts import main_sweep_root
from dasbench.utils import timestamp_token

SWEEP_KIND = "seed_variance_benchmark"

# Frozen main-run whose datasets are reused so seeds share identical held-out splits.
DEFAULT_SOURCE_RUN_ROOT = str(main_sweep_root())
DEFAULT_SOURCE_CONDITION_ID = "seconds_scale_v2"
DEFAULT_ENV_FILE = ".env.gemma4"


def build_conditions(
    *,
    num_runs: int,
    start_run: int,
    train_size: int,
    validation_size: int,
    test_size: int,
    iterations: int,
    beam_width: int,
    candidate_width: int | None,
) -> list[dict[str, object]]:
    if num_runs < 1:
        raise ValueError("--num-runs must be >= 1.")
    if start_run < 1:
        raise ValueError("--start-run must be >= 1.")
    conditions: list[dict[str, object]] = []
    for seed_index in range(start_run, start_run + num_runs):
        conditions.append(
            {
                "condition_id": f"seed_{seed_index:02d}",
                "train_size": train_size,
                "validation_size": validation_size,
                "test_size": test_size,
                "iterations": iterations,
                "beam_width": beam_width,
                "candidate_width": candidate_width,
                # Baselines are cached in the reused main run; the report stage is not needed
                # (we read the selected candidate's test metrics straight from synthesis).
                "skip_baselines": True,
                "skip_report": True,
                "seed_index": seed_index,
            }
        )
    return conditions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Seed-variance sweep: run the synthesis pipeline N times per target (one seed "
            "condition each) to quantify run-to-run variance. Defaults reproduce the local "
            "Gemma4 (agent generator) Experiment 6a setup."
        )
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of independent runs (seed conditions) per target. Default 5.",
    )
    parser.add_argument(
        "--start-run",
        type=int,
        default=1,
        help="First seed index, for additively extending an existing sweep (e.g. --start-run 6).",
    )
    parser.add_argument(
        "--candidate-width",
        type=int,
        default=3,
        help="Candidates generated per beam slot per iteration. Default 3 (matches the paper run).",
    )
    parser.add_argument("--train-size", type=int, default=64)
    parser.add_argument("--validation-size", type=int, default=32)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument(
        "--representative-only",
        action="store_true",
        help="Use one representative family per problem (7 targets) instead of all 21 — e.g. the "
             "GPT-5.2 seed-variance sub-part (7 targets x seeds).",
    )
    parser.set_defaults(
        # Local-model Experiment 6a defaults; override on the CLI for the GPT-5.2 sub-part.
        generator="llm",
        source_run_root=DEFAULT_SOURCE_RUN_ROOT,
        source_condition_id=DEFAULT_SOURCE_CONDITION_ID,
        env_file=DEFAULT_ENV_FILE,
        iterations=3,
        beam_width=3,
        repeats=1,
    )
    return parser


def build_jobs(args: argparse.Namespace, *, sweep_id: str | None = None) -> list:
    sweep_id = sweep_id or args.sweep_id or timestamp_token()
    targets = selected_targets(args.problem, args.family, representative_only=args.representative_only)
    conditions = build_conditions(
        num_runs=args.num_runs,
        start_run=args.start_run,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        iterations=args.iterations,
        beam_width=args.beam_width,
        candidate_width=args.candidate_width,
    )
    return jobs_from_conditions(
        sweep_id=sweep_id,
        artifact_root=resolve_sweep_artifact_root(args.output_root, SWEEP_KIND, sweep_id),
        targets=targets,
        conditions=conditions,
        args=args,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sweep_id = args.sweep_id or timestamp_token()
    jobs = build_jobs(args, sweep_id=sweep_id)
    summary = run_sweep(
        sweep_id=sweep_id,
        sweep_kind=SWEEP_KIND,
        jobs=jobs,
        output_root=Path(args.output_root),
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
