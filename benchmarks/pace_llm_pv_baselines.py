from __future__ import annotations

import argparse
import json
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
from benchmarks.pace_competitions import COMPETITIONS, CompetitionConfig, export_agent_solutions
from dasbench.data import load_manifest
from dasbench.integrations import load_chat_api_config
from dasbench.utils import timestamp_token, write_json

DEFAULT_OUTPUT_ROOT = Path("artifacts/pace_competitions/llm_pv")
DEFAULT_DATASETS = {
    "pace2025_hs": Path("artifacts/pace_competitions/gemma4_pace2025_hs_full/dataset"),
    "pace2024_ocm_exact": Path("artifacts/pace_competitions/gemma4_pace2024_ocm_exact_full/dataset"),
    "pace2024_ocm_cutwidth": Path("artifacts/pace_competitions/gemma4_pace2024_ocm_cutwidth_full/dataset"),
    "pace2022_dfvs_heuristic": Path(
        "artifacts/pace_competitions/gemma4_pace2022_dfvs_heuristic_100test_reuse_10test_solver/dataset"
    ),
}
DEFAULT_COMPETITIONS = tuple(DEFAULT_DATASETS)


def _build_config(args: argparse.Namespace) -> LLMPVConfig:
    reasoning_effort = None if args.reasoning_effort is None else str(args.reasoning_effort).strip()
    if reasoning_effort == "none":
        reasoning_effort = None
    return LLMPVConfig(
        attempts=max(1, int(args.attempts)),
        model=str(args.model),
        reasoning_effort=reasoning_effort or None,
        max_output_tokens=None if args.max_output_tokens is None else max(1, int(args.max_output_tokens)),
        api_timeout_seconds=max(1.0, float(args.api_timeout_seconds)),
        enable_code_interpreter=False,
        tool_choice="auto",
        verbosity=str(args.verbosity),
        early_stop_score=None if args.no_early_stop else float(args.early_stop_score),
        prompt_train_examples=max(0, int(args.prompt_train_examples)),
        prompt_json_char_limit=max(1_000, int(args.prompt_json_char_limit)),
    )


def _dataset_for_competition(args: argparse.Namespace, competition: str) -> Path:
    override = dict(item.split("=", 1) for item in args.dataset if "=" in item)
    if competition in override:
        return Path(override[competition])
    try:
        return DEFAULT_DATASETS[competition]
    except KeyError as exc:
        raise ValueError(f"No default dataset is configured for `{competition}`; pass --dataset {competition}=PATH.") from exc


def _require_exportable_solver(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_candidate = summary.get("best_candidate")
    if not isinstance(best_candidate, dict):
        raise RuntimeError(f"LLM-PV summary has no selected candidate: {summary_path}")
    candidate_dir = Path(str(best_candidate.get("candidate_dir", "")))
    solution_path = candidate_dir / "solution.py"
    if solution_path.exists() and not best_candidate.get("error"):
        return
    reason = best_candidate.get("error")
    test_summary = best_candidate.get("test")
    if not reason and isinstance(test_summary, dict):
        reason = test_summary.get("error")
    detail = f" Selected candidate error: {str(reason)[:1000]}" if reason else ""
    raise RuntimeError(f"LLM-PV did not produce an exportable selected solver for {summary_path}.{detail}")


def run_one(
    *,
    competition: str,
    config: CompetitionConfig,
    dataset_dir: Path,
    output_dir: Path,
    sweep_id: str,
    llm_pv_config: LLMPVConfig,
    force: bool,
    dry_run: bool,
    export_solutions: bool,
) -> dict[str, Any]:
    manifest = load_manifest(dataset_dir)
    job = LLMPVJob(
        sweep_id=sweep_id,
        artifact_root=output_dir / "llm_pv_artifacts",
        problem=str(manifest["problem"]),
        family=str(manifest["family"]),
        source_dataset_dir=dataset_dir,
        force=force,
        config=llm_pv_config,
    )
    result = _run_target(job, dry_run=dry_run)
    summary: dict[str, Any] = {
        "schema_version": "pace_llm_pv_baseline.v1",
        "competition": competition,
        "dataset_dir": str(dataset_dir),
        "staged_dataset_dir": str(job.dataset_dir),
        "llm_pv_run_dir": str(job.run_dir),
        "llm_pv_summary_path": str(job.summary_path),
        "llm_pv_result": result,
    }
    if dry_run:
        write_json(output_dir / "pace_llm_pv_summary.json", summary)
        return summary
    if export_solutions:
        _require_exportable_solver(job.summary_path)
        pace_summary = export_agent_solutions(
            config=config,
            dataset_dir=job.dataset_dir,
            agent_run_dir=job.run_dir,
            output_dir=output_dir / "pace_evaluation",
        )
        summary["pace_evaluation"] = pace_summary
    write_json(output_dir / "pace_llm_pv_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LLM-PV baselines on the imported PACE competition datasets.")
    parser.add_argument("--sweep-id", default=f"pace_llm_pv_{timestamp_token()}")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--competition",
        action="append",
        choices=sorted(DEFAULT_COMPETITIONS),
        default=[],
        help="Competition to run. Defaults to all main imported PACE competitions.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Override one dataset path as COMPETITION=PATH.",
    )
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--model", default=_llm_pv_default_model())
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
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
    parser.add_argument("--no-export-solutions", dest="export_solutions", action="store_false")
    parser.set_defaults(export_solutions=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    competitions = args.competition or list(DEFAULT_COMPETITIONS)
    if not args.dry_run:
        load_chat_api_config(required=True)
    output_root = args.output_root / args.sweep_id
    output_root.mkdir(parents=True, exist_ok=True)
    llm_pv_config = _build_config(args)
    results: list[dict[str, Any]] = []
    failed = False
    for competition in competitions:
        config = COMPETITIONS[competition]
        output_dir = output_root / competition
        try:
            result = run_one(
                competition=competition,
                config=config,
                dataset_dir=_dataset_for_competition(args, competition),
                output_dir=output_dir,
                sweep_id=args.sweep_id,
                llm_pv_config=llm_pv_config,
                force=bool(args.force),
                dry_run=bool(args.dry_run),
                export_solutions=bool(args.export_solutions),
            )
        except Exception as exc:
            failed = True
            result = {
                "schema_version": "pace_llm_pv_baseline.v1",
                "competition": competition,
                "dataset_dir": str(_dataset_for_competition(args, competition)),
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_json(output_dir / "pace_llm_pv_summary.json", result)
        results.append(result)
        print(json.dumps({"competition": competition, "summary": str(output_dir / "pace_llm_pv_summary.json")}))
    aggregate = {
        "schema_version": "pace_llm_pv_baselines.v1",
        "sweep_id": args.sweep_id,
        "output_root": str(output_root),
        "competitions": competitions,
        "results": results,
    }
    write_json(output_root / "pace_llm_pv_aggregate.json", aggregate)
    print(f"Aggregate JSON: {output_root / 'pace_llm_pv_aggregate.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
