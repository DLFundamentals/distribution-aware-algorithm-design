from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

from benchmarks.llm_pv_benchmark import (
    DEFAULT_API_TIMEOUT_SECONDS,
    DEFAULT_ATTEMPTS,
    DEFAULT_PROMPT_JSON_CHAR_LIMIT,
    DEFAULT_PROMPT_TRAIN_EXAMPLES,
    LLMPVConfig,
    LLMPVJob,
    _llm_pv_default_model,
    _llm_pv_default_reasoning_effort,
    _run_target,
)
from benchmarks.pace2025_dominating_set import export_pace_evaluation
from dasbench.data import load_manifest
from dasbench.utils import timestamp_token, write_json
from scripts.pace2025_collect_heuristic_report import (
    BASELINE_FIELDNAMES,
    DEFAULT_BASELINE_ROOT,
    DEFAULT_CACHE_DIR,
    DEFAULT_EXPANDED_DIR,
    DEFAULT_PACE_RUN_DIR,
    collect as collect_heuristic_report,
)

DEFAULT_SOLVER_NAME = "llm_pv_custom"
DEFAULT_OUTPUT_DIR = DEFAULT_BASELINE_ROOT / DEFAULT_SOLVER_NAME


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def _canonical_instance_id(instance_id: str) -> str:
    match = re.search(r"(private_heuristic_\d{3}|heuristic_\d{3}|private_exact_\d{3}|exact_\d{3})", instance_id)
    if match:
        return match.group(1)
    return instance_id


def _reference_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    return {_canonical_instance_id(row.get("instance_id", "")): row for row in _read_csv(path)}


def _baseline_rows_from_pace_evaluation(
    *,
    pace_rows: list[dict[str, str]],
    reference_by_instance: dict[str, dict[str, str]],
    output_dir: Path,
    solver_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    solution_root = output_dir / "solutions" / solver_name
    for row in pace_rows:
        instance_id = _canonical_instance_id(row.get("instance_id", ""))
        feasible = str(row.get("feasible", "")).strip().lower() in {"true", "1", "yes"}
        source_solution_text = row.get("solution_file", "")
        source_solution = Path(source_solution_text) if source_solution_text else None
        target_solution = solution_root / f"{instance_id}.sol"
        solution_file = ""
        if feasible and source_solution is not None and source_solution.exists():
            target_solution.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_solution, target_solution)
            solution_file = str(target_solution)

        reference = reference_by_instance.get(instance_id, {})
        rows.append(
            {
                "solver": solver_name,
                "instance_id": instance_id,
                "pace_source_path": row.get("pace_source_path", ""),
                "num_vertices": row.get("num_vertices", ""),
                "num_edges": row.get("num_edges", ""),
                "exit_code": 0 if feasible else 1,
                "timed_out": False,
                "valid": feasible,
                "valid_status": "verified_valid" if feasible else "invalid",
                "solution_size": row.get("solution_size", "") if feasible else "",
                "runtime_ms": row.get("runtime_ms", ""),
                "synth_solution_size": reference.get("solution_size", ""),
                "adapter_reference_objective": row.get("reference_objective", ""),
                "solution_file": solution_file,
                "stderr_file": "",
                "error": row.get("error", "") if not feasible else "",
                "source_dir": str(output_dir),
                "row_source": "baseline_results_csv",
            }
        )
    return rows


def _build_config(args: argparse.Namespace) -> LLMPVConfig:
    return LLMPVConfig(
        attempts=max(1, int(args.attempts)),
        model=str(args.model),
        reasoning_effort=None if args.reasoning_effort is None else str(args.reasoning_effort),
        max_output_tokens=None if args.max_output_tokens is None else max(1, int(args.max_output_tokens)),
        api_timeout_seconds=max(1.0, float(args.api_timeout_seconds)),
        enable_code_interpreter=False,
        tool_choice="auto",
        verbosity=str(args.verbosity),
        early_stop_score=None if args.no_early_stop else float(args.early_stop_score),
        prompt_train_examples=max(0, int(args.prompt_train_examples)),
        prompt_json_char_limit=max(1_000, int(args.prompt_json_char_limit)),
    )


def _require_exportable_llm_pv_solver(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_candidate = summary.get("best_candidate")
    if not isinstance(best_candidate, dict):
        raise RuntimeError(f"LLM-PV summary has no selected candidate: {summary_path}")

    candidate_dir_text = best_candidate.get("candidate_dir")
    candidate_dir = Path(str(candidate_dir_text)) if candidate_dir_text else None
    solution_path = None if candidate_dir is None else candidate_dir / "solution.py"
    if solution_path is not None and solution_path.exists() and not best_candidate.get("error"):
        return

    reason = best_candidate.get("error")
    test_summary = best_candidate.get("test")
    if not reason and isinstance(test_summary, dict):
        reason = test_summary.get("error")
    detail = f" Selected candidate error: {str(reason)[:1000]}" if reason else ""
    raise RuntimeError(
        "LLM-PV did not produce an exportable selected solver; PACE export requires a successful "
        f"`solution.py`. Summary: {summary_path}.{detail} "
        "Re-run the helper with `--force` after adjusting the LLM request or endpoint."
    )


def run_llm_pv_baseline(args: argparse.Namespace) -> dict[str, Any]:
    pace_run_dir = args.pace_run_dir
    dataset_dir = args.dataset_dir or pace_run_dir / "dataset"
    output_dir = args.output_dir
    run_artifact_root = args.run_artifact_root or output_dir / "llm_pv_artifacts"
    pace_eval_dir = args.pace_evaluation_dir or output_dir / "pace_evaluation"
    manifest = load_manifest(dataset_dir)
    problem = str(manifest["problem"])
    family = str(manifest["family"])
    sweep_id = args.sweep_id or f"{args.solver_name}_{timestamp_token()}"

    job = LLMPVJob(
        sweep_id=sweep_id,
        artifact_root=run_artifact_root,
        problem=problem,
        family=family,
        source_dataset_dir=dataset_dir,
        force=bool(args.force),
        config=_build_config(args),
    )
    if args.dry_run:
        payload = {
            "status": "dry_run",
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "staged_dataset_dir": str(job.dataset_dir),
            "llm_pv_run_dir": str(job.run_dir),
            "llm_pv_summary_path": str(job.summary_path),
            "solver": args.solver_name,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload

    llm_pv_result = _run_target(job, dry_run=False)
    _require_exportable_llm_pv_solver(job.summary_path)
    pace_summary = export_pace_evaluation(
        dataset_dir=job.dataset_dir,
        agent_run_dir=job.run_dir,
        output_dir=pace_eval_dir,
    )
    pace_rows_path = Path(str(pace_summary["results_csv"]))
    reference_csv = args.reference_results_csv or pace_run_dir / "pace_evaluation" / "pace_private_results.csv"
    baseline_rows = _baseline_rows_from_pace_evaluation(
        pace_rows=_read_csv(pace_rows_path),
        reference_by_instance=_reference_rows(reference_csv),
        output_dir=output_dir,
        solver_name=args.solver_name,
    )
    baseline_csv = output_dir / "baseline_results.csv"
    _write_csv(baseline_csv, baseline_rows, fieldnames=BASELINE_FIELDNAMES)

    summary = {
        "schema_version": "pace2025_llm_pv_baseline.v1",
        "solver": args.solver_name,
        "dataset_dir": str(dataset_dir),
        "staged_dataset_dir": str(job.dataset_dir),
        "llm_pv_run_dir": str(job.run_dir),
        "llm_pv_summary_path": str(job.summary_path),
        "pace_evaluation_dir": str(pace_eval_dir),
        "pace_evaluation_summary_path": str(pace_eval_dir / "pace_evaluation_summary.json"),
        "baseline_results_csv": str(baseline_csv),
        "baseline_solution_dir": str(output_dir / "solutions" / args.solver_name),
        "reference_results_csv": str(reference_csv) if reference_csv is not None else None,
        "llm_pv_result": llm_pv_result,
        "pace_evaluation": pace_summary,
    }
    write_json(output_dir / "llm_pv_baseline_summary.json", summary)

    if args.refresh_report:
        report_args = argparse.Namespace(
            pace_run_dir=pace_run_dir,
            baseline_root=args.baseline_root,
            baseline_dir=None,
            output_dir=args.report_output_dir or pace_run_dir / "heuristic_comparison_report",
            include_smoke=args.include_smoke,
            verify_reconstructed=args.verify_reconstructed,
            cache_dir=args.cache_dir,
            expanded_dir=args.expanded_dir,
            github_ref=args.github_ref,
        )
        summary["refreshed_report"] = collect_heuristic_report(report_args)["output_paths"]
        write_json(output_dir / "llm_pv_baseline_summary.json", summary)

    print(f"LLM-PV baseline CSV: {baseline_csv}")
    print(f"LLM-PV run summary: {job.summary_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an LLM-PV solver on an existing PACE 2025 dataset and export it as a baseline comparison."
    )
    parser.add_argument("--pace-run-dir", type=Path, default=DEFAULT_PACE_RUN_DIR)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-artifact-root", type=Path)
    parser.add_argument("--pace-evaluation-dir", type=Path)
    parser.add_argument("--reference-results-csv", type=Path)
    parser.add_argument("--solver-name", default=DEFAULT_SOLVER_NAME)
    parser.add_argument("--sweep-id")
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--model", default=_llm_pv_default_model())
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high", "xhigh"],
        default=_llm_pv_default_reasoning_effort(),
    )
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--api-timeout-seconds", type=float, default=DEFAULT_API_TIMEOUT_SECONDS)
    parser.add_argument("--verbosity", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--early-stop-score", type=float, default=1.0)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--prompt-train-examples", type=int, default=DEFAULT_PROMPT_TRAIN_EXAMPLES)
    parser.add_argument("--prompt-json-char-limit", type=int, default=DEFAULT_PROMPT_JSON_CHAR_LIMIT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh-report", action="store_true")
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--report-output-dir", type=Path)
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--verify-reconstructed", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--expanded-dir", type=Path, default=DEFAULT_EXPANDED_DIR)
    parser.add_argument("--github-ref", default="master")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_llm_pv_baseline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
