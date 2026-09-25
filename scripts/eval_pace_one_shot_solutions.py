#!/usr/bin/env python3
"""Evaluate manually-produced one-shot agent solvers on the PACE test splits.

Reads `solutions/<generator>/<track>/solution.py` from a capsule sweep built by
`scripts/build_pace_one_shot_capsules.py`, evaluates each on the held-out 100
private instances of its track, and writes per-target summaries plus an
aggregate in the same shape as the 2026-06 one-shot results, so the numbers drop
straight into the comparison tables.

  python scripts/eval_pace_one_shot_solutions.py \
      --sweep-root artifacts/pace_one_shot_capsules/agents_rerun_20260924

Targets with no `solution.py` yet are reported as pending and skipped, so this
can be run repeatedly as the manual runs land. Completed targets are skipped
unless --force.

Two cost guards, because a bad solver on a PACE instance can burn hours:

  Preflight -- the first few instances are scored first. A solver that returns
  nothing valid on all of them would score zero over the full split too, so the
  target is recorded as zero without running the remaining instances. This is
  how the 2026-06 Claude Code Dominating Set row became a bound rather than a
  measurement, and the record says so explicitly.

  Target budget -- once a target has spent --max-target-seconds, later instances
  fail immediately and the record is flagged as partial. Partial records are
  reported but must not be quoted as the target's runtime.

Evaluation is serial by default. These runtimes are speedup denominators in the
paper, and running targets concurrently on a shared machine inflates them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dasbench.agents.candidate import build_solver  # noqa: E402
from dasbench.data import load_manifest, load_split  # noqa: E402
from dasbench.eval.evaluator import evaluate_solver, write_summary  # noqa: E402
from dasbench.utils import timestamp_token, write_json  # noqa: E402

DEFAULT_PER_INSTANCE_TIMEOUT_SECONDS = 360.0
DEFAULT_MAX_TARGET_SECONDS = 3600.0
DEFAULT_PREFLIGHT_INSTANCES = 3
# Matches scripts/remeasure_agent_runtime.py: a solver that cannot produce a
# valid answer is not "instant", it is unusable.
FAILURE_RUNTIME_MS = 1_000_000.0


class TargetBudgetExceeded(RuntimeError):
    pass


class _BudgetedSolver:
    """Fails fast once the target has spent its wall budget."""

    def __init__(self, solver, budget_seconds: float | None) -> None:
        self._solver = solver
        self._budget = budget_seconds
        self._started: float | None = None
        self.attempted = 0
        self.exceeded = False

    def __call__(self, instance):
        if self._started is None:
            self._started = time.perf_counter()
        if self._budget and time.perf_counter() - self._started > self._budget:
            self.exceeded = True
            raise TargetBudgetExceeded(f"target exceeded its {self._budget:.0f}s evaluation budget")
        self.attempted += 1
        return self._solver(instance)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _looks_like_timeout(summary: dict[str, object]) -> bool:
    text = str(summary.get("error") or "")
    if not text:
        cases = summary.get("failure_cases")
        if isinstance(cases, list):
            text = " ".join(str(case.get("error", "")) for case in cases if isinstance(case, dict))
    return "Timeout" in text


def _aborted_record(
    preflight: dict[str, object],
    *,
    num_instances: int,
    per_instance_timeout: float,
    preflight_count: int,
) -> dict[str, object]:
    timed_out = _looks_like_timeout(preflight)
    runtime_ms = per_instance_timeout * 1000.0 if timed_out else FAILURE_RUNTIME_MS
    return {
        "num_instances": num_instances,
        "average_normalized_quality": 0.0,
        "average_objective_value": 0.0,
        "optimality_rate": 0.0,
        "feasibility_rate": 0.0,
        "average_runtime_ms": runtime_ms,
        "failure_cases": preflight.get("failure_cases", []),
        "error": preflight.get("error") or "preflight produced no valid solution",
        "error_count": num_instances,
        "aborted": True,
        "abort_reason": (
            f"preflight: no valid solution on the first {preflight_count} instances"
            + (" (timeouts)" if timed_out else "")
        ),
        "runtime_is_lower_bound": bool(timed_out),
        "instances_attempted": preflight_count,
        "preflight_summary": {
            k: preflight.get(k)
            for k in ("average_normalized_quality", "feasibility_rate", "average_runtime_ms", "error")
        },
    }


def evaluate_target(
    *,
    sweep_root: Path,
    generator: str,
    track: dict[str, object],
    split: str,
    per_instance_timeout: float,
    max_target_seconds: float | None,
    preflight_instances: int,
    agent_metadata: dict[str, object] | None,
) -> dict[str, object] | None:
    track_id = str(track["track_id"])
    solution_dir = sweep_root / "solutions" / generator / track_id
    solution_path = solution_dir / "solution.py"
    if not solution_path.is_file():
        print(f"  pending  {generator}/{track_id}: no solution.py yet", flush=True)
        return None

    dataset_dir = REPO / str(track["source_dataset_dir"])
    manifest = load_manifest(dataset_dir)
    problem_name = str(manifest["problem"])
    instances = load_split(dataset_dir, split)
    name = f"{generator}_one_shot_{track_id}"

    started = time.perf_counter()
    solver = build_solver(solution_dir, analysis=None, manifest=manifest)

    preflight_count = min(max(0, preflight_instances), len(instances))
    aborted_fields: dict[str, object] = {}
    if preflight_count:
        preflight = evaluate_solver(
            problem_name,
            f"{name}_preflight",
            solver,
            instances[:preflight_count],
            split=split,
            timeout_seconds=per_instance_timeout,
        )
        if float(preflight.get("average_normalized_quality") or 0.0) <= 0.0:
            print(
                f"  ABORT    {generator}/{track_id}: preflight scored 0 over {preflight_count} "
                f"instances -- recording zero without the remaining {len(instances) - preflight_count}",
                flush=True,
            )
            summary = _aborted_record(
                preflight,
                num_instances=len(instances),
                per_instance_timeout=per_instance_timeout,
                preflight_count=preflight_count,
            )
            aborted_fields = summary
        else:
            print(
                f"  preflight {generator}/{track_id}: q="
                f"{preflight.get('average_normalized_quality'):.4f} over {preflight_count} instances",
                flush=True,
            )

    if not aborted_fields:
        budgeted = _BudgetedSolver(solver, max_target_seconds)
        summary = evaluate_solver(
            problem_name,
            name,
            budgeted,
            instances,
            split=split,
            timeout_seconds=per_instance_timeout,
        )
        summary["instances_attempted"] = budgeted.attempted
        summary["aborted"] = False
        if budgeted.exceeded:
            summary["target_budget_exceeded"] = True
            summary["abort_reason"] = (
                f"target budget of {max_target_seconds:.0f}s spent after "
                f"{budgeted.attempted} of {len(instances)} instances"
            )
            print(
                f"  PARTIAL  {generator}/{track_id}: budget spent after "
                f"{budgeted.attempted}/{len(instances)} instances",
                flush=True,
            )

    record = {
        **summary,
        "name": name,
        "generator": generator,
        "track_id": track_id,
        "title": track.get("title"),
        "problem": problem_name,
        "pace_problem": track.get("problem"),
        "split": split,
        "dataset_dir": str(track["source_dataset_dir"]),
        "source_family": track.get("source_family"),
        "solution_path": str(solution_path.relative_to(REPO)),
        "solution_sha256": _sha256(solution_path),
        "per_instance_timeout_seconds": per_instance_timeout,
        "max_target_seconds": max_target_seconds,
        "evaluation_wall_seconds": time.perf_counter() - started,
        "evaluated_at": timestamp_token(),
        "agent_metadata": agent_metadata,
    }
    label = "zero" if record.get("aborted") else ("partial" if record.get("target_budget_exceeded") else "ok")
    print(
        f"  {label:8s} {generator}/{track_id}: q={record.get('average_normalized_quality'):.4f} "
        f"feas={record.get('feasibility_rate')} obj={record.get('average_objective_value'):.1f} "
        f"rt={record.get('average_runtime_ms'):.1f}ms "
        f"[{record['evaluation_wall_seconds']:.0f}s wall]",
        flush=True,
    )
    return record


def _write_aggregate(out_dir: Path, generator: str, records: list[dict[str, object]]) -> None:
    if not records:
        return
    write_json(out_dir / "aggregate_results.json", {"generator": generator, "results": records})
    scalar_keys = sorted(
        {k for r in records for k, v in r.items() if not isinstance(v, (list, dict))}
    )
    lines = [",".join(scalar_keys)]
    for record in records:
        row = []
        for key in scalar_keys:
            value = record.get(key, "")
            text = "" if value is None else str(value)
            row.append(f'"{text}"' if "," in text else text)
        lines.append(",".join(row))
    (out_dir / "aggregate_results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--generators", nargs="+", default=None, help="Defaults to the sweep manifest's list.")
    parser.add_argument("--tracks", nargs="+", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--per-instance-timeout", type=float, default=DEFAULT_PER_INSTANCE_TIMEOUT_SECONDS,
                        help="Matches the budget our synthesized solvers were given.")
    parser.add_argument("--max-target-seconds", type=float, default=DEFAULT_MAX_TARGET_SECONDS,
                        help="0 disables the per-target wall cap.")
    parser.add_argument("--preflight-instances", type=int, default=DEFAULT_PREFLIGHT_INSTANCES,
                        help="0 disables the preflight abort.")
    parser.add_argument("--force", action="store_true", help="Re-evaluate targets that already have a summary.")
    args = parser.parse_args(argv)

    sweep_root = Path(args.sweep_root)
    if not sweep_root.is_absolute():
        sweep_root = REPO / sweep_root
    manifest_path = sweep_root / "sweep_manifest.json"
    if not manifest_path.is_file():
        print(f"ERROR: no sweep_manifest.json under {sweep_root}", flush=True)
        return 2
    sweep = json.loads(manifest_path.read_text(encoding="utf-8"))

    tracks = list(sweep.get("tracks") or [])
    if args.tracks:
        tracks = [t for t in tracks if str(t.get("track_id")) in set(args.tracks)]
    generators = args.generators or list(sweep.get("generators") or [])
    max_target_seconds = args.max_target_seconds if args.max_target_seconds > 0 else None

    print(f"sweep={sweep_root.name}  split={args.split}")
    print(f"generators={generators}  tracks={[t['track_id'] for t in tracks]}")
    print(f"per-instance timeout={args.per_instance_timeout}s  target budget={max_target_seconds}s  "
          f"preflight={args.preflight_instances}")

    failed = 0
    pending = 0
    for generator in generators:
        out_dir = sweep_root / "results" / f"{generator}_{args.split}"
        meta_path = sweep_root / "solutions" / generator / "run_metadata.json"
        agent_metadata = None
        if meta_path.is_file():
            agent_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            print(f"{generator}: NOTE no run_metadata.json -- agent version will not be recorded", flush=True)

        print(f"{generator}:", flush=True)
        records: list[dict[str, object]] = []
        for track in tracks:
            track_id = str(track["track_id"])
            target_out = out_dir / "per_target" / track_id / f"{args.split}_summary.json"
            if target_out.is_file() and not args.force:
                print(f"  skip     {generator}/{track_id} (already evaluated)", flush=True)
                records.append(json.loads(target_out.read_text(encoding="utf-8")))
                continue
            try:
                record = evaluate_target(
                    sweep_root=sweep_root,
                    generator=generator,
                    track=track,
                    split=args.split,
                    per_instance_timeout=args.per_instance_timeout,
                    max_target_seconds=max_target_seconds,
                    preflight_instances=args.preflight_instances,
                    agent_metadata=agent_metadata,
                )
            except Exception as exc:  # noqa: BLE001 - one bad target must not lose the others
                failed += 1
                print(f"  FAIL     {generator}/{track_id}: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                continue
            if record is None:
                pending += 1
                continue
            write_summary(target_out, record)
            records.append(record)
            _write_aggregate(out_dir, generator, records)  # checkpoint after every target
        _write_aggregate(out_dir, generator, records)

    print(f"\ndone: {failed} failed, {pending} pending")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
