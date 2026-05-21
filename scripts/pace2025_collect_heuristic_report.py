from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.pace2025_dominating_set import _private_ds_path
from dasbench.utils import timestamp_token, write_json
from scripts.pace2025_run_heuristic_baselines import (
    DEFAULT_CACHE_DIR,
    DEFAULT_EXPANDED_DIR,
    materialize_gr,
    parse_pace_header,
    parse_solution,
    verify_dominating_set,
)

DEFAULT_PACE_RUN_DIR = Path("artifacts/pace2025_dominating_set/pace2025_ds_heuristic_llm_01")
DEFAULT_BASELINE_ROOT = Path("artifacts/pace2025_dominating_set/baseline_comparisons")
DEFAULT_OUTPUT_DIR = DEFAULT_PACE_RUN_DIR / "heuristic_comparison_report"
BASELINE_FIELDNAMES = [
    "solver",
    "instance_id",
    "pace_source_path",
    "num_vertices",
    "num_edges",
    "exit_code",
    "timed_out",
    "valid",
    "valid_status",
    "solution_size",
    "runtime_ms",
    "synth_solution_size",
    "adapter_reference_objective",
    "solution_file",
    "stderr_file",
    "error",
    "source_dir",
    "row_source",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return value


def _display_solver_name(solver: object) -> str:
    text = str(solver)
    return text.removeprefix("dasbench_")


def _canonical_instance_id(instance_id: str) -> str:
    match = re.search(r"(private_heuristic_\d{3})", instance_id)
    if match:
        return match.group(1)
    return instance_id


def _instance_index(instance_id: str) -> int | None:
    match = re.search(r"private_heuristic_(\d{3})", instance_id)
    return int(match.group(1)) if match else None


def _bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _int_or_none(value: object) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _baseline_dirs(args: argparse.Namespace) -> list[Path]:
    if args.baseline_dir:
        return [Path(path) for path in args.baseline_dir]
    if not args.baseline_root.exists():
        return []
    dirs = [
        path
        for path in sorted(args.baseline_root.iterdir())
        if path.is_dir() and ((path / "baseline_results.csv").exists() or (path / "solutions").exists())
    ]
    if not args.include_smoke:
        dirs = [path for path in dirs if not path.name.startswith("smoke")]
    return dirs


def _agent_result_paths(pace_run_dir: Path) -> dict[str, Path]:
    return {
        "synthesis_summary": pace_run_dir / "agent_run" / "synthesis_summary.json",
        "pace_evaluation_summary": pace_run_dir / "pace_evaluation" / "pace_evaluation_summary.json",
        "pace_private_results": pace_run_dir / "pace_evaluation" / "pace_private_results.csv",
    }


def _agent_rows(pace_results_csv: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    rows = _read_csv(pace_results_csv)
    by_instance = {_canonical_instance_id(row["instance_id"]): row for row in rows}
    return rows, by_instance


def _row_from_baseline_csv(row: dict[str, str], *, source_dir: Path) -> dict[str, Any]:
    payload = {key: row.get(key, "") for key in BASELINE_FIELDNAMES}
    payload["solver"] = _display_solver_name(payload["solver"])
    payload["instance_id"] = _canonical_instance_id(str(payload["instance_id"]))
    payload["source_dir"] = str(source_dir)
    payload["row_source"] = "baseline_results_csv"
    valid = _bool_or_none(payload.get("valid"))
    payload["valid_status"] = "verified_valid" if valid is True else "invalid" if valid is False else "unknown"
    return payload


def _pace_source_for_instance(instance_id: str, agent_by_instance: dict[str, dict[str, str]]) -> str:
    agent_row = agent_by_instance.get(instance_id)
    if agent_row and agent_row.get("pace_source_path"):
        return agent_row["pace_source_path"]
    index = _instance_index(instance_id)
    if index is None:
        return ""
    return _private_ds_path("heuristic", index)


def _graph_header(
    relative_path: str,
    *,
    cache_dir: Path,
    expanded_dir: Path,
    github_ref: str,
) -> tuple[int | None, int | None, Path | None]:
    if not relative_path:
        return None, None, None
    input_path = materialize_gr(
        relative_path,
        cache_dir=cache_dir,
        expanded_dir=expanded_dir,
        github_ref=github_ref,
    )
    num_vertices, num_edges = parse_pace_header(input_path)
    return num_vertices, num_edges, input_path


def _reconstruct_solution_row(
    solution_file: Path,
    *,
    baseline_dir: Path,
    agent_by_instance: dict[str, dict[str, str]],
    verify: bool,
    cache_dir: Path,
    expanded_dir: Path,
    github_ref: str,
) -> dict[str, Any]:
    solver = solution_file.parent.name
    instance_id = _canonical_instance_id(solution_file.stem)
    agent_row = agent_by_instance.get(instance_id, {})
    relative_path = _pace_source_for_instance(instance_id, agent_by_instance)
    num_vertices = _int_or_none(agent_row.get("num_vertices"))
    num_edges = _int_or_none(agent_row.get("num_edges"))
    input_path: Path | None = None
    if num_vertices is None or num_edges is None or verify:
        header_vertices, header_edges, input_path = _graph_header(
            relative_path,
            cache_dir=cache_dir,
            expanded_dir=expanded_dir,
            github_ref=github_ref,
        )
        num_vertices = num_vertices if num_vertices is not None else header_vertices
        num_edges = num_edges if num_edges is not None else header_edges
    stdout = solution_file.read_text(encoding="utf-8", errors="replace")
    solution, parse_error = parse_solution(stdout, num_vertices=int(num_vertices or 0))
    valid: bool | None = None
    error = parse_error or ""
    valid_status = "parse_error" if parse_error else "parsed_unverified"
    if parse_error is None and verify:
        if input_path is None:
            _, _, input_path = _graph_header(
                relative_path,
                cache_dir=cache_dir,
                expanded_dir=expanded_dir,
                github_ref=github_ref,
            )
        assert input_path is not None
        valid, verify_error = verify_dominating_set(input_path, solution, num_vertices=int(num_vertices or 0))
        error = verify_error or ""
        valid_status = "verified_valid" if valid else "invalid"
    stderr_file = baseline_dir / "stderr" / solver / f"{instance_id}.stderr.txt"
    return {
        "solver": _display_solver_name(solver),
        "instance_id": instance_id,
        "pace_source_path": relative_path,
        "num_vertices": "" if num_vertices is None else num_vertices,
        "num_edges": "" if num_edges is None else num_edges,
        "exit_code": "",
        "timed_out": "",
        "valid": "" if valid is None else valid,
        "valid_status": valid_status,
        "solution_size": "" if parse_error else len(solution),
        "runtime_ms": "",
        "synth_solution_size": agent_row.get("solution_size", ""),
        "adapter_reference_objective": agent_row.get("reference_objective", ""),
        "solution_file": str(solution_file),
        "stderr_file": str(stderr_file) if stderr_file.exists() else "",
        "error": error,
        "source_dir": str(baseline_dir),
        "row_source": "reconstructed_solution_file",
    }


def _load_baseline_rows(
    baseline_dirs: list[Path],
    *,
    agent_by_instance: dict[str, dict[str, str]],
    verify_reconstructed: bool,
    cache_dir: Path,
    expanded_dir: Path,
    github_ref: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    csv_solution_files: set[str] = set()
    for baseline_dir in baseline_dirs:
        csv_path = baseline_dir / "baseline_results.csv"
        if csv_path.exists():
            for row in _read_csv(csv_path):
                payload = _row_from_baseline_csv(row, source_dir=baseline_dir)
                rows.append(payload)
                solution_file = str(payload.get("solution_file", ""))
                if solution_file:
                    csv_solution_files.add(solution_file)
    for baseline_dir in baseline_dirs:
        solutions_root = baseline_dir / "solutions"
        if not solutions_root.exists():
            continue
        for solution_file in sorted(solutions_root.glob("*/*.sol")):
            if str(solution_file) in csv_solution_files:
                continue
            rows.append(
                _reconstruct_solution_row(
                    solution_file,
                    baseline_dir=baseline_dir,
                    agent_by_instance=agent_by_instance,
                    verify=verify_reconstructed,
                    cache_dir=cache_dir,
                    expanded_dir=expanded_dir,
                    github_ref=github_ref,
                )
            )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["solver"]), str(row["instance_id"]))].append(row)
    deduped: list[dict[str, Any]] = []
    for key, key_rows in sorted(grouped.items()):
        best = sorted(key_rows, key=_baseline_row_sort_key, reverse=True)[0]
        deduped.append(best)
        for duplicate in key_rows:
            if duplicate is not best:
                duplicates.append(
                    {
                        "solver": key[0],
                        "instance_id": key[1],
                        "kept_solution_file": best.get("solution_file", ""),
                        "dropped_solution_file": duplicate.get("solution_file", ""),
                        "kept_row_source": best.get("row_source", ""),
                        "dropped_row_source": duplicate.get("row_source", ""),
                    }
                )
    return deduped, duplicates


def _baseline_row_sort_key(row: dict[str, Any]) -> tuple[int, int, int, float]:
    valid = _bool_or_none(row.get("valid"))
    solution_size = _int_or_none(row.get("solution_size"))
    has_solution = solution_size is not None
    if valid is True:
        validity_rank = 3
    elif valid is None and has_solution:
        validity_rank = 2
    elif valid is False:
        validity_rank = 1
    else:
        validity_rank = 0
    source_rank = 1 if row.get("row_source") == "baseline_results_csv" else 0
    size_rank = -float(solution_size) if solution_size is not None else float("-inf")
    return (validity_rank, int(has_solution), size_rank, source_rank)


def _agent_summary(
    *,
    synthesis_summary: dict[str, Any],
    pace_evaluation_summary: dict[str, Any],
    agent_rows: list[dict[str, str]],
) -> dict[str, Any]:
    best_candidate = synthesis_summary.get("best_candidate", {})
    solution_sizes = [_int_or_none(row.get("solution_size")) for row in agent_rows]
    solution_sizes = [value for value in solution_sizes if value is not None]
    reference_comparisons = []
    for row in agent_rows:
        solution_size = _int_or_none(row.get("solution_size"))
        reference = _int_or_none(row.get("reference_objective"))
        if solution_size is None or reference is None:
            continue
        reference_comparisons.append(solution_size - reference)
    return {
        "best_candidate_slug": best_candidate.get("slug"),
        "hypothesis": best_candidate.get("hypothesis"),
        "selection": best_candidate.get("selection"),
        "train": best_candidate.get("train"),
        "validation": best_candidate.get("validation"),
        "test": best_candidate.get("test"),
        "pace_evaluation": pace_evaluation_summary,
        "median_solution_size": statistics.median(solution_sizes) if solution_sizes else None,
        "reference_comparison": {
            "count": len(reference_comparisons),
            "agent_better_count": sum(1 for delta in reference_comparisons if delta < 0),
            "agent_worse_count": sum(1 for delta in reference_comparisons if delta > 0),
            "tie_count": sum(1 for delta in reference_comparisons if delta == 0),
            "mean_solution_minus_reference": (
                statistics.mean(reference_comparisons) if reference_comparisons else None
            ),
        },
    }


def _solver_summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_solver: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_solver[str(row["solver"])].append(row)
    summaries: dict[str, dict[str, Any]] = {}
    for solver, solver_rows in sorted(by_solver.items()):
        solution_rows = [row for row in solver_rows if _int_or_none(row.get("solution_size")) is not None]
        solution_sizes = [_int_or_none(row.get("solution_size")) for row in solution_rows]
        solution_sizes = [value for value in solution_sizes if value is not None]
        runtime_values = [_float_or_none(row.get("runtime_ms")) for row in solver_rows]
        runtime_values = [value for value in runtime_values if value is not None]
        valid_values = [_bool_or_none(row.get("valid")) for row in solver_rows]
        comparisons = _comparison_counts(solution_rows)
        summaries[solver] = {
            "row_count": len(solver_rows),
            "covered_instance_count": len({row["instance_id"] for row in solver_rows}),
            "solution_count": len(solution_rows),
            "verified_valid_count": sum(1 for value in valid_values if value is True),
            "parse_only_count": sum(1 for row in solver_rows if row.get("valid_status") == "parsed_unverified"),
            "invalid_count": sum(1 for value in valid_values if value is False),
            "timeout_count_known": sum(1 for row in solver_rows if _bool_or_none(row.get("timed_out")) is True),
            "runtime_known_count": len(runtime_values),
            "total_solution_size": sum(solution_sizes),
            "average_solution_size": statistics.mean(solution_sizes) if solution_sizes else None,
            "median_solution_size": statistics.median(solution_sizes) if solution_sizes else None,
            "average_runtime_ms": statistics.mean(runtime_values) if runtime_values else None,
            **comparisons,
        }
    return summaries


def _average_solution_runtime_table(
    *,
    agent: dict[str, Any],
    solver_summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    pace_eval = agent["pace_evaluation"]
    agent_instances = _int_or_none(pace_eval.get("num_instances"))
    rows = [
        {
            "solver": "agent",
            "role": "agent",
            "coverage": agent_instances,
            "solution_count": _int_or_none(pace_eval.get("feasible_count")) or agent_instances,
            "average_solution_length": _float_or_none(pace_eval.get("average_solution_size")),
            "median_solution_length": agent.get("median_solution_size"),
            "average_runtime_ms": _float_or_none(pace_eval.get("average_runtime_ms")),
            "runtime_known_count": agent_instances,
            "validity_note": "feasible PACE-format solutions",
        }
    ]
    for solver, summary in sorted(solver_summaries.items()):
        rows.append(
            {
                "solver": solver,
                "role": "baseline",
                "coverage": summary["covered_instance_count"],
                "solution_count": summary["solution_count"],
                "average_solution_length": summary["average_solution_size"],
                "median_solution_length": summary["median_solution_size"],
                "average_runtime_ms": summary["average_runtime_ms"],
                "runtime_known_count": summary["runtime_known_count"],
                "validity_note": (
                    f"{summary['verified_valid_count']} verified, "
                    f"{summary['parse_only_count']} parse-only"
                ),
            }
        )
    return rows


def _comparison_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    baseline_better = 0
    agent_better = 0
    tie = 0
    compared = 0
    for row in rows:
        baseline_size = _int_or_none(row.get("solution_size"))
        agent_size = _int_or_none(row.get("synth_solution_size"))
        if baseline_size is None or agent_size is None:
            continue
        compared += 1
        if baseline_size < agent_size:
            baseline_better += 1
        elif baseline_size > agent_size:
            agent_better += 1
        else:
            tie += 1
    return {
        "compared_to_agent_count": compared,
        "baseline_better_count": baseline_better,
        "agent_better_count": agent_better,
        "tie_count": tie,
    }


def _instance_comparisons(
    *,
    agent_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in baseline_rows:
        baseline_by_instance[str(row["instance_id"])].append(row)
    comparisons: list[dict[str, Any]] = []
    for agent_row in agent_rows:
        instance_id = _canonical_instance_id(agent_row["instance_id"])
        baseline_entries = []
        agent_size = _int_or_none(agent_row.get("solution_size"))
        for baseline_row in sorted(baseline_by_instance.get(instance_id, []), key=lambda item: str(item["solver"])):
            baseline_size = _int_or_none(baseline_row.get("solution_size"))
            baseline_entries.append(
                {
                    "solver": baseline_row.get("solver"),
                    "solution_size": baseline_size,
                    "valid": baseline_row.get("valid"),
                    "valid_status": baseline_row.get("valid_status"),
                    "runtime_ms": _float_or_none(baseline_row.get("runtime_ms")),
                    "baseline_minus_agent": (
                        baseline_size - agent_size
                        if baseline_size is not None and agent_size is not None
                        else None
                    ),
                    "solution_file": baseline_row.get("solution_file"),
                }
            )
        comparisons.append(
            {
                "instance_id": instance_id,
                "pace_source_path": agent_row.get("pace_source_path"),
                "num_vertices": _int_or_none(agent_row.get("num_vertices")),
                "num_edges": _int_or_none(agent_row.get("num_edges")),
                "agent_solution_size": agent_size,
                "agent_runtime_ms": _float_or_none(agent_row.get("runtime_ms")),
                "adapter_reference_objective": _int_or_none(agent_row.get("reference_objective")),
                "lower_bound": _int_or_none(agent_row.get("lower_bound")),
                "baselines": baseline_entries,
            }
        )
    return comparisons


def _top_baseline_improvements(instance_comparisons: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    rows = []
    for comparison in instance_comparisons:
        agent_size = comparison["agent_solution_size"]
        if agent_size is None:
            continue
        for baseline in comparison["baselines"]:
            baseline_size = baseline["solution_size"]
            if baseline_size is None:
                continue
            improvement = agent_size - baseline_size
            rows.append(
                {
                    "instance_id": comparison["instance_id"],
                    "solver": baseline["solver"],
                    "agent_solution_size": agent_size,
                    "baseline_solution_size": baseline_size,
                    "baseline_improvement": improvement,
                    "valid_status": baseline["valid_status"],
                }
            )
    return sorted(rows, key=lambda row: row["baseline_improvement"], reverse=True)[:limit]


def _format_number(value: object, *, digits: int = 2) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    agent = payload["agent"]
    pace_eval = agent["pace_evaluation"]
    solver_summaries = payload["baseline_solver_summaries"]
    source_paths = payload["source_paths"]
    average_rows = [
        [
            row["solver"],
            row["role"],
            row["coverage"],
            row["solution_count"],
            _format_number(row["average_solution_length"], digits=2),
            _format_number(row["median_solution_length"], digits=2),
            _format_number(
                None if row["average_runtime_ms"] is None else row["average_runtime_ms"] / 1000.0,
                digits=2,
            ),
            row["runtime_known_count"],
            row["validity_note"],
        ]
        for row in payload["average_solution_runtime_table"]
    ]
    rows = []
    for solver, summary in solver_summaries.items():
        rows.append(
            [
                solver,
                summary["covered_instance_count"],
                summary["solution_count"],
                summary["verified_valid_count"],
                summary["parse_only_count"],
                _format_number(summary["total_solution_size"], digits=0),
                _format_number(summary["average_solution_size"], digits=2),
                _format_number(summary["median_solution_size"], digits=2),
                summary["compared_to_agent_count"],
                summary["baseline_better_count"],
                summary["agent_better_count"],
                summary["tie_count"],
                _format_number(
                    None if summary["average_runtime_ms"] is None else summary["average_runtime_ms"] / 1000.0,
                    digits=2,
                ),
            ]
        )
    top_rows = [
        [
            row["instance_id"],
            row["solver"],
            row["agent_solution_size"],
            row["baseline_solution_size"],
            row["baseline_improvement"],
            row["valid_status"],
        ]
        for row in payload["top_baseline_improvements"]
    ]
    ref = agent["reference_comparison"]
    lines = [
        "# PACE 2025 Heuristic Dominating Set Report",
        "",
        "This is a local comparison artifact, not an official PACE score. The PACE private heuristic data in the public repository does not include best-known/optimal labels, so the report compares solution sizes, feasibility/proxy fields, and available baseline runs.",
        "",
        "## Inputs",
        "",
        f"- Agent run: `{source_paths['pace_run_dir']}`",
        f"- Agent synthesis summary: `{source_paths['synthesis_summary']}`",
        f"- Agent PACE results CSV: `{source_paths['pace_private_results']}`",
        f"- Baseline dirs: {', '.join(f'`{path}`' for path in source_paths['baseline_dirs']) or 'none'}",
        f"- Reconstructed baseline rows verified: `{payload['metadata']['verify_reconstructed']}`",
        "",
        "## Agent",
        "",
        f"- Best candidate: `{agent.get('best_candidate_slug')}`",
        f"- Private instances: `{pace_eval.get('num_instances')}`",
        f"- Feasible count: `{pace_eval.get('feasible_count')}`",
        f"- Total solution size: `{pace_eval.get('total_solution_size')}`",
        f"- Average solution size: `{_format_number(pace_eval.get('average_solution_size'), digits=2)}`",
        f"- Median solution size: `{_format_number(agent.get('median_solution_size'), digits=2)}`",
        f"- Average runtime seconds: `{_format_number(None if _float_or_none(pace_eval.get('average_runtime_ms')) is None else _float_or_none(pace_eval.get('average_runtime_ms')) / 1000.0, digits=2)}`",
        f"- Compared with adapter reference proxy: agent better `{ref['agent_better_count']}`, worse `{ref['agent_worse_count']}`, tied `{ref['tie_count']}` over `{ref['count']}` instances.",
        "",
        "## Average Solution Length And Runtime",
        "",
        _markdown_table(
            [
                "solver",
                "role",
                "coverage",
                "solutions",
                "avg solution length",
                "median solution length",
                "avg runtime seconds",
                "runtime known",
                "validity note",
            ],
            average_rows,
        ),
        "",
        "## Baselines",
        "",
        _markdown_table(
            [
                "solver",
                "coverage",
                "solutions",
                "verified",
                "parse-only",
                "total size",
                "avg size",
                "median size",
                "compared",
                "baseline better",
                "agent better",
                "ties",
                "avg runtime seconds",
            ],
            rows,
        ),
        "",
        "## Largest Baseline Improvements",
        "",
        _markdown_table(
            ["instance", "solver", "agent size", "baseline size", "improvement", "valid status"],
            top_rows,
        )
        if top_rows
        else "No comparable baseline solutions found.",
        "",
        "## Notes",
        "",
        "- Pairwise baseline/agent counts use every parsed baseline solution size. Rows marked `parse-only` came from interrupted baseline `.sol` files and are not graph-verified unless `--verify-reconstructed` is used.",
        f"- Baseline duplicate rows dropped: `{len(payload['duplicate_baseline_rows'])}`",
        f"- Combined baseline CSV: `{payload['output_paths']['combined_baseline_csv']}`",
        f"- Per-instance comparison CSV: `{payload['output_paths']['instance_comparison_csv']}`",
        f"- JSON payload: `{payload['output_paths']['json']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect(args: argparse.Namespace) -> dict[str, Any]:
    paths = _agent_result_paths(args.pace_run_dir)
    synthesis_summary = _read_json(paths["synthesis_summary"])
    pace_evaluation_summary = _read_json(paths["pace_evaluation_summary"])
    agent_rows, agent_by_instance = _agent_rows(paths["pace_private_results"])
    baseline_dirs = _baseline_dirs(args)
    baseline_rows, duplicate_rows = _load_baseline_rows(
        baseline_dirs,
        agent_by_instance=agent_by_instance,
        verify_reconstructed=args.verify_reconstructed,
        cache_dir=args.cache_dir,
        expanded_dir=args.expanded_dir,
        github_ref=args.github_ref,
    )
    instance_comparisons = _instance_comparisons(agent_rows=agent_rows, baseline_rows=baseline_rows)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "pace_heuristic_comparison.json"
    report_path = output_dir / "pace_heuristic_comparison.md"
    combined_csv_path = output_dir / "combined_baseline_results.csv"
    instance_csv_path = output_dir / "instance_comparisons.csv"
    _write_csv(combined_csv_path, baseline_rows, fieldnames=BASELINE_FIELDNAMES)
    _write_csv(
        instance_csv_path,
        _flatten_instance_comparisons(instance_comparisons),
        fieldnames=[
            "instance_id",
            "solver",
            "agent_solution_size",
            "baseline_solution_size",
            "baseline_minus_agent",
            "valid_status",
            "agent_runtime_ms",
            "baseline_runtime_ms",
            "num_vertices",
            "num_edges",
            "adapter_reference_objective",
            "lower_bound",
        ],
    )
    agent_summary = _agent_summary(
        synthesis_summary=synthesis_summary,
        pace_evaluation_summary=pace_evaluation_summary,
        agent_rows=agent_rows,
    )
    baseline_solver_summaries = _solver_summaries(baseline_rows)
    payload = {
        "schema_version": "pace2025_ds_heuristic_comparison_report.v1",
        "created_at": timestamp_token(),
        "metadata": {
            "verify_reconstructed": args.verify_reconstructed,
            "include_smoke": args.include_smoke,
            "github_ref": args.github_ref,
        },
        "source_paths": {
            "pace_run_dir": str(args.pace_run_dir),
            "synthesis_summary": str(paths["synthesis_summary"]),
            "pace_evaluation_summary": str(paths["pace_evaluation_summary"]),
            "pace_private_results": str(paths["pace_private_results"]),
            "baseline_dirs": [str(path) for path in baseline_dirs],
        },
        "output_paths": {
            "json": str(json_path),
            "report": str(report_path),
            "combined_baseline_csv": str(combined_csv_path),
            "instance_comparison_csv": str(instance_csv_path),
        },
        "agent": agent_summary,
        "agent_results": agent_rows,
        "baseline_rows": baseline_rows,
        "baseline_solver_summaries": baseline_solver_summaries,
        "average_solution_runtime_table": _average_solution_runtime_table(
            agent=agent_summary,
            solver_summaries=baseline_solver_summaries,
        ),
        "duplicate_baseline_rows": duplicate_rows,
        "instance_comparisons": instance_comparisons,
        "top_baseline_improvements": _top_baseline_improvements(instance_comparisons),
        "agent_synthesis_summary": synthesis_summary,
        "pace_evaluation_summary": pace_evaluation_summary,
    }
    write_json(json_path, payload)
    _write_report(report_path, payload)
    return payload


def _flatten_instance_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        baselines = row.get("baselines", [])
        if not baselines:
            flattened.append(
                {
                    "instance_id": row["instance_id"],
                    "solver": "",
                    "agent_solution_size": row["agent_solution_size"],
                    "baseline_solution_size": "",
                    "baseline_minus_agent": "",
                    "valid_status": "",
                    "agent_runtime_ms": row["agent_runtime_ms"],
                    "baseline_runtime_ms": "",
                    "num_vertices": row["num_vertices"],
                    "num_edges": row["num_edges"],
                    "adapter_reference_objective": row["adapter_reference_objective"],
                    "lower_bound": row["lower_bound"],
                }
            )
            continue
        for baseline in baselines:
            flattened.append(
                {
                    "instance_id": row["instance_id"],
                    "solver": baseline["solver"],
                    "agent_solution_size": row["agent_solution_size"],
                    "baseline_solution_size": baseline["solution_size"],
                    "baseline_minus_agent": baseline["baseline_minus_agent"],
                    "valid_status": baseline["valid_status"],
                    "agent_runtime_ms": row["agent_runtime_ms"],
                    "baseline_runtime_ms": baseline["runtime_ms"],
                    "num_vertices": row["num_vertices"],
                    "num_edges": row["num_edges"],
                    "adapter_reference_objective": row["adapter_reference_objective"],
                    "lower_bound": row["lower_bound"],
                }
            )
    return flattened


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect PACE 2025 heuristic agent and baseline artifacts into a JSON payload and Markdown report."
    )
    parser.add_argument("--pace-run-dir", type=Path, default=DEFAULT_PACE_RUN_DIR)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--baseline-dir", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument(
        "--verify-reconstructed",
        action="store_true",
        help="Verify reconstructed baseline .sol rows against the PACE graph files. This can be slow on large instances.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--expanded-dir", type=Path, default=DEFAULT_EXPANDED_DIR)
    parser.add_argument("--github-ref", default="master")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = collect(args)
    print(f"Report: {payload['output_paths']['report']}")
    print(f"JSON: {payload['output_paths']['json']}")
    print(f"Combined baseline CSV: {payload['output_paths']['combined_baseline_csv']}")
    print(f"Instance comparison CSV: {payload['output_paths']['instance_comparison_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
