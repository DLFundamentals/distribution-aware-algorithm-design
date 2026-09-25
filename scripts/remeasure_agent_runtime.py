#!/usr/bin/env python3
"""Re-measure a synthesized solver's held-out runtime on the reporting basis.

Only the GPT-5.2 main run ever ran the report stage. Every other sweep's
runtime therefore comes from the evaluation inside the synthesis loop, which
ran while 21 targets were synthesizing and their baselines were being scored --
so those numbers carry CPU contention the cached baselines never paid. Comparing
them is apples-to-oranges, and it is what currently reverses the generator
ablation's runtime claim.

This re-runs only the piece that matters: the selected candidate's analysis, its
solver, and a repeated evaluation on the held-out split. It deliberately does
NOT go through `generate_benchmark_report`, because the sweeps that need fixing
were run with `--skip-baselines` and hold an empty `baseline_test.json` -- with
no cache to reuse, the report path recomputes the whole baseline catalog
(Gurobi at 10 s/instance, times `repeats`).

Baselines do not need re-measuring at all: every sweep reuses the main run's
datasets byte-for-byte, so the main run's `baseline_test.json` already describes
the same instances.

Output goes to `<target>/remeasure/agent_<split>_repeats<N>.json`, a fresh
directory, so no existing artifact is touched.

The GPT-5.2 sweep is included as a CONTROL: its published report numbers are
known, so if the re-measured values track them, the worker count used here is
not inflating the timings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dasbench.agents.candidate import build_solver, run_analysis  # noqa: E402
from dasbench.data import load_manifest, load_split  # noqa: E402
from dasbench.eval.evaluator import evaluate_solver_repeated  # noqa: E402

SWEEP_ROOT = REPO / "artifacts" / "second_scale_benchmark_v2"
CONTROL_SWEEP = "20260427_230552"
DEFAULT_SWEEPS = (
    "gemma4_second_scale_v2_001",
    "deepseek32_second_scale_v2_002",
    "glm47_second_scale_v2_002",
    CONTROL_SWEEP,
)
# A candidate that failed during synthesis is stored with zero quality and this
# sentinel runtime. There is no solver to re-time, so it is recorded, not run.
FAILURE_RUNTIME_MS = 1_000_000.0


def discover(sweeps: list[str]) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for sweep in sweeps:
        for summary_path in sorted((SWEEP_ROOT / sweep).glob("targets/*/*/*/agent_run/synthesis_summary.json")):
            agent_run = summary_path.parent
            target_dir = agent_run.parent
            jobs.append(
                {
                    "sweep": sweep,
                    "problem": target_dir.parts[-2],
                    "family": target_dir.parts[-1],
                    "target_dir": str(target_dir),
                    "agent_run": str(agent_run),
                }
            )
    return jobs


def remeasure(job: dict[str, str], *, split: str, repeats: int, force: bool) -> dict[str, object]:
    target_dir = Path(job["target_dir"])
    agent_run = Path(job["agent_run"])
    out_dir = target_dir / "remeasure"
    out_path = out_dir / f"agent_{split}_repeats{repeats}.json"
    label = f"{job['sweep']}:{job['problem']}/{job['family']}"

    if out_path.exists() and not force:
        return {"label": label, "status": "skipped", "path": str(out_path)}

    started = time.perf_counter()
    summary = json.loads((agent_run / "synthesis_summary.json").read_text(encoding="utf-8"))
    best = summary["best_candidate"]
    stored = best.get(split) or {}
    stored_runtime = float(stored.get("average_runtime_ms") or 0.0)

    record: dict[str, object] = {
        "sweep": job["sweep"],
        "problem": job["problem"],
        "family": job["family"],
        "split": split,
        "repeats": repeats,
        "candidate_slug": best.get("slug"),
        "stored_quality": stored.get("average_normalized_quality"),
        "stored_runtime_ms": stored.get("average_runtime_ms"),
    }

    # Nothing to re-time for a candidate that never produced a working solver.
    if stored_runtime >= FAILURE_RUNTIME_MS:
        record.update({"status": "failed_candidate", "note": "stored as a synthesis failure; not re-measured"})
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"label": label, "status": "failed_candidate", "path": str(out_path)}

    dataset_dir = target_dir / "dataset"
    manifest = load_manifest(dataset_dir)
    problem_name = str(manifest["problem"])
    train_public = load_split(dataset_dir, "train", public=True)
    instances = load_split(dataset_dir, split)

    candidate_dir = Path(best["candidate_dir"])
    if not candidate_dir.is_absolute():
        candidate_dir = REPO / candidate_dir

    # Same three steps the report stage performs, and nothing else.
    analysis = run_analysis(candidate_dir, train_public, manifest=manifest, artifact_dir=out_dir / "analysis")
    solver = build_solver(candidate_dir, analysis=analysis, manifest=manifest)
    result = evaluate_solver_repeated(
        problem_name, str(best["slug"]), solver, instances, split=split, repeats=repeats
    )

    record.update(
        {
            "status": "ok",
            "num_instances": result.get("num_instances"),
            "quality_mean": result.get("average_normalized_quality_mean"),
            "quality_std": result.get("average_normalized_quality_std"),
            "optimality_mean": result.get("optimality_rate_mean"),
            "feasibility_mean": result.get("feasibility_rate_mean"),
            "runtime_ms_mean": result.get("average_runtime_ms_mean"),
            "runtime_ms_std": result.get("average_runtime_ms_std"),
            "wall_seconds": round(time.perf_counter() - started, 1),
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "label": label,
        "status": "ok",
        "path": str(out_path),
        "runtime_ms_mean": record["runtime_ms_mean"],
        "runtime_ms_std": record["runtime_ms_std"],
        "stored_runtime_ms": stored_runtime,
        "wall_seconds": record["wall_seconds"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps", nargs="*", default=list(DEFAULT_SWEEPS))
    parser.add_argument("--problem")
    parser.add_argument("--family")
    parser.add_argument("--split", default="test")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="Re-measure targets that already have output.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    jobs = discover(args.sweeps)
    if args.problem:
        jobs = [j for j in jobs if j["problem"] == args.problem]
    if args.family:
        jobs = [j for j in jobs if j["family"] == args.family]

    print(f"{len(jobs)} targets to re-measure (split={args.split}, repeats={args.repeats}, "
          f"workers={args.max_workers})", flush=True)
    for sweep in args.sweeps:
        print(f"  {sweep}: {sum(1 for j in jobs if j['sweep'] == sweep)}", flush=True)
    if args.dry_run or not jobs:
        return 0

    started = time.perf_counter()
    done = failed = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(remeasure, job, split=args.split, repeats=args.repeats, force=args.force): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            done += 1
            try:
                out = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad target must not kill the sweep
                failed += 1
                print(f"[{done}/{len(jobs)}] FAIL {job['sweep']}:{job['problem']}/{job['family']}"
                      f" :: {type(exc).__name__}: {exc}", flush=True)
                continue
            if out["status"] == "ok":
                print(f"[{done}/{len(jobs)}] ok   {out['label']}  "
                      f"{out['runtime_ms_mean']:.3f} ± {out['runtime_ms_std']:.3f} ms  "
                      f"(stored {out['stored_runtime_ms']:.3f})  [{out['wall_seconds']:.0f}s]", flush=True)
            else:
                print(f"[{done}/{len(jobs)}] {out['status']} {out['label']}", flush=True)
    print(f"finished in {(time.perf_counter()-started)/60:.1f} min; {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
