from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from dasbench.artifacts import main_sweep_root


DEFAULT_OUTPUT_DIR = Path("results/collected_experiment_results_20260609")
DEFAULT_CANDIDATE_OUTPUT_DIR = Path("results/candidate_results")

RUN_SOURCES = {
    "second_scale_v2": main_sweep_root(),
    "sample_size_sweep": Path("artifacts/sample_size_sweep/20260502_211019"),
    "problem_size_sweep": Path("artifacts/problem_size_sweep/20260501_012438"),
    "iteration_count_sweep": Path("artifacts/iteration_count_sweep/20260501_012522"),
    "graph_relabel_invariance": Path("artifacts/graph_relabel_invariance_benchmark/20260505_115106"),
    "no_hint_recovery": Path("artifacts/no_hint_recovery_benchmark/20260506_020413"),
    "ml_baselines_gpu": Path("artifacts/ml_baseline_runs/ml_faithful_all_20260603_104930"),
    "ml_baselines_cpu_eval": Path("artifacts/ml_baseline_runs/ml_baseline_cpu_eval/ml_faithful_cpu_eval_20260606_115342"),
    "pace2025_dominating_set": Path("artifacts/pace2025_dominating_set"),
}

PAPER_ARXIV_URL = "https://arxiv.org/abs/2605.14141"

PAPER_HEADLINE_ROWS = {
    "coloring": {
        "q_llm": 0.868,
        "delta_q_avg": 0.217,
        "delta_q_best": 0.121,
        "t_llm_ms": 2.7,
        "speedup_best": 1326.3,
        "speedup_gurobi": 2285.6,
        "speedup_exact": 23.1,
    },
    "maxsat": {
        "q_llm": 1.000,
        "delta_q_avg": 0.122,
        "delta_q_best": 0.074,
        "t_llm_ms": 17.1,
        "speedup_best": 217.4,
        "speedup_gurobi": 328.8,
        "speedup_exact": 1.0,
    },
    "mis": {
        "q_llm": 0.992,
        "delta_q_avg": 0.218,
        "delta_q_best": 0.106,
        "t_llm_ms": 18.8,
        "speedup_best": 530.8,
        "speedup_gurobi": 155.8,
        "speedup_exact": 39.7,
    },
    "mds": {
        "q_llm": 0.973,
        "delta_q_avg": 0.148,
        "delta_q_best": 0.122,
        "t_llm_ms": 13.3,
        "speedup_best": 718.9,
        "speedup_gurobi": 443.0,
        "speedup_exact": 17.4,
    },
    "packing_lp": {
        "q_llm": 0.994,
        "delta_q_avg": 0.301,
        "delta_q_best": 0.259,
        "t_llm_ms": 3.3,
        "speedup_best": 3004.2,
        "speedup_gurobi": 2829.1,
        "speedup_exact": 37.2,
    },
    "mdkp": {
        "q_llm": 0.973,
        "delta_q_avg": 0.215,
        "delta_q_best": 0.009,
        "t_llm_ms": 94.0,
        "speedup_best": 46.4,
        "speedup_gurobi": 33.8,
        "speedup_exact": 12.4,
    },
    "tsp": {
        "q_llm": 0.993,
        "delta_q_avg": 0.348,
        "delta_q_best": -0.007,
        "t_llm_ms": 15.3,
        "speedup_best": 32.1,
        "speedup_gurobi": 112.1,
        "speedup_exact": 36.6,
    },
    "all": {
        "q_llm": 0.971,
        "delta_q_avg": 0.224,
        "delta_q_best": 0.098,
        "t_llm_ms": 12.8,
        "speedup_best": 336.9,
        "speedup_gurobi": 342.8,
        "speedup_exact": 16.1,
    },
}

PAPER_PACE_ROWS = {
    "agent": {
        "valid": 100,
        "total_size_m": 23.16,
        "avg_size": 231594.7,
        "agent_over_solver_size": 1.000,
        "avg_runtime_s": 2.89,
        "agent_speedup": 1.0,
    },
    "fontanf": {
        "valid": 100,
        "total_size_m": 22.41,
        "avg_size": 224103.9,
        "agent_over_solver_size": 1.033,
        "avg_runtime_s": 286.01,
        "agent_speedup": 98.8,
    },
    "root": {
        "valid": 100,
        "total_size_m": 22.41,
        "avg_size": 224123.7,
        "agent_over_solver_size": 1.033,
        "avg_runtime_s": 300.37,
        "agent_speedup": 103.8,
    },
    "shadoks": {
        "valid": 100,
        "total_size_m": 22.43,
        "avg_size": 224275.6,
        "agent_over_solver_size": 1.033,
        "avg_runtime_s": 315.89,
        "agent_speedup": 109.2,
    },
    "swats": {
        "valid": 75,
        "total_size_m": 15.77,
        "avg_size": 210235.7,
        "agent_over_solver_size": 1.028,
        "avg_runtime_s": 287.41,
        "agent_speedup": 130.2,
    },
}

PAPER_RELABEL_BY_PROBLEM = {
    "coloring": {
        "targets": 3,
        "original_quality": 0.868,
        "transformed_quality": 0.791,
        "delta_quality": -0.077,
        "quality_changed_fraction": 0.342,
        "optimality_changed_fraction": 0.342,
        "feasibility_changed_fraction": 0.000,
        "runtime_ratio": 1.38,
    },
    "mds": {
        "targets": 3,
        "original_quality": 0.973,
        "transformed_quality": 0.819,
        "delta_quality": -0.155,
        "quality_changed_fraction": 0.419,
        "optimality_changed_fraction": 0.333,
        "feasibility_changed_fraction": 0.000,
        "runtime_ratio": 1.39,
    },
    "mis": {
        "targets": 3,
        "original_quality": 0.992,
        "transformed_quality": 0.992,
        "delta_quality": -0.001,
        "quality_changed_fraction": 0.258,
        "optimality_changed_fraction": 0.191,
        "feasibility_changed_fraction": 0.000,
        "runtime_ratio": 1.31,
    },
    "all": {
        "targets": 9,
        "original_quality": 0.945,
        "transformed_quality": 0.867,
        "delta_quality": -0.077,
        "quality_changed_fraction": 0.340,
        "optimality_changed_fraction": 0.289,
        "feasibility_changed_fraction": 0.000,
        "runtime_ratio": 1.36,
    },
}

NATIVE_EXACT_SOLVER_NAMES = {
    "rc2_cadical195",
    "rc2_glucose4",
    "rc2_minisat22",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect compact experiment result exports under results/.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-output-dir", type=Path, default=DEFAULT_CANDIDATE_OUTPUT_DIR)
    parser.add_argument("--skip-pace-collector", action="store_true")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return value


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(parsed):
        return parsed
    return None


def int_or_none(value: Any) -> int | None:
    parsed = float_or_none(value)
    if parsed is None:
        return None
    return int(parsed)


def mean(values: Iterable[float]) -> float | None:
    data = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.mean(data) if data else None


def geomean(values: Iterable[float]) -> float | None:
    data = [value for value in values if value is not None and math.isfinite(value) and value > 0.0]
    if not data:
        return None
    return math.exp(statistics.mean(math.log(value) for value in data))


def metric(result: dict[str, Any], name: str) -> float | None:
    return float_or_none(result.get(f"{name}_mean", result.get(name)))


def is_exact_solver(name: str) -> bool:
    return name == "exact" or name.endswith("_exact") or name in NATIVE_EXACT_SOLVER_NAMES


def solver_role(name: str, agent_name: str | None) -> str:
    if agent_name and name == agent_name:
        return "agent"
    if name == "gurobi_timed":
        return "gurobi"
    if is_exact_solver(name):
        return "exact"
    return "heuristic"


def selection_key(row: dict[str, Any]) -> tuple[float, float, float]:
    quality = float_or_none(row.get("quality")) or 0.0
    optimality = float_or_none(row.get("optimality_rate")) or 0.0
    runtime = float_or_none(row.get("runtime_ms")) or 1e18
    return (round(quality, 9), round(optimality, 9), -runtime)


def target_parts_from_agent_run(agent_run_dir: Path) -> tuple[str, str, str]:
    parts = agent_run_dir.parts
    if "targets" in parts:
        index = parts.index("targets")
        if len(parts) > index + 3:
            return parts[index + 1], parts[index + 2], parts[index + 3]
    return "", "", ""


def benchmark_report_rows(root: Path, experiment: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_path in sorted(root.rglob("report/benchmark_report.json")):
        payload = read_json(report_path)
        manifest = payload.get("manifest", {})
        best_candidate = payload.get("best_candidate", {})
        agent_name = best_candidate.get("slug") if isinstance(best_candidate, dict) else None
        target_id, problem_from_path, family_from_path = target_parts_from_agent_run(
            Path(str(payload.get("agent_run_dir", "")))
        )
        problem = str(manifest.get("problem") or problem_from_path)
        family = str(manifest.get("family") or family_from_path)
        instance_params = manifest.get("instance_params", {})
        split_reports = payload.get("split_reports", {})
        test_results = split_reports.get("test", {}) if isinstance(split_reports, dict) else {}
        for solver, result in sorted(test_results.items()):
            if not isinstance(result, dict):
                continue
            rows.append(
                {
                    "experiment": experiment,
                    "source_root": str(root),
                    "target_id": target_id,
                    "problem": problem,
                    "family": family,
                    "solver": solver,
                    "role": solver_role(solver, str(agent_name) if agent_name else None),
                    "quality": metric(result, "average_normalized_quality"),
                    "optimality_rate": metric(result, "optimality_rate"),
                    "feasibility_rate": metric(result, "feasibility_rate"),
                    "runtime_ms": metric(result, "average_runtime_ms"),
                    "num_instances": result.get("num_instances"),
                    "report_json_path": str(report_path),
                    "agent_slug": agent_name or "",
                    "instance_params": instance_params,
                }
            )
    return rows


def agent_summary_rows(root: Path, experiment: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.rglob("agent_run/synthesis_summary.json")):
        payload = read_json(summary_path)
        best = payload.get("best_candidate", {})
        if not isinstance(best, dict):
            continue
        test = best.get("test", {})
        validation = best.get("validation", {})
        train = best.get("train", {})
        agent_run_dir = summary_path.parent
        target_id, problem_from_path, family_from_path = target_parts_from_agent_run(agent_run_dir)
        dataset_manifest = agent_run_dir.parent / "dataset" / "manifest.json"
        manifest = read_json(dataset_manifest) if dataset_manifest.exists() else {}
        rows.append(
            {
                "experiment": experiment,
                "source_root": str(root),
                "target_id": target_id,
                "problem": str(payload.get("problem") or manifest.get("problem") or problem_from_path),
                "family": str(payload.get("family") or manifest.get("family") or family_from_path),
                "agent_slug": best.get("slug"),
                "train_quality": metric(train, "average_normalized_quality"),
                "train_optimality_rate": metric(train, "optimality_rate"),
                "train_feasibility_rate": metric(train, "feasibility_rate"),
                "train_runtime_ms": metric(train, "average_runtime_ms"),
                "validation_quality": metric(validation, "average_normalized_quality"),
                "validation_optimality_rate": metric(validation, "optimality_rate"),
                "validation_feasibility_rate": metric(validation, "feasibility_rate"),
                "validation_runtime_ms": metric(validation, "average_runtime_ms"),
                "test_quality": metric(test, "average_normalized_quality"),
                "test_optimality_rate": metric(test, "optimality_rate"),
                "test_feasibility_rate": metric(test, "feasibility_rate"),
                "test_runtime_ms": metric(test, "average_runtime_ms"),
                "test_num_instances": test.get("num_instances") if isinstance(test, dict) else "",
                "synthesis_summary_path": str(summary_path),
                "instance_params": manifest.get("instance_params", {}),
            }
        )
    return rows


def parse_candidate_slug(slug: str) -> tuple[int | None, int | None]:
    match = re.search(r"iter(\d+)_slot(\d+)", slug)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_text_if_exists(path: Path, *, limit: int = 800) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text[:limit]


def summary_metric(summary: dict[str, Any], key: str) -> Any:
    return summary.get(key, "")


def add_split_metrics(row: dict[str, Any], prefix: str, summary: dict[str, Any]) -> None:
    row[f"{prefix}_evaluated"] = bool(summary)
    row[f"{prefix}_quality"] = summary_metric(summary, "average_normalized_quality")
    row[f"{prefix}_objective"] = summary_metric(summary, "average_objective_value")
    row[f"{prefix}_optimality_rate"] = summary_metric(summary, "optimality_rate")
    row[f"{prefix}_feasibility_rate"] = summary_metric(summary, "feasibility_rate")
    row[f"{prefix}_runtime_ms"] = summary_metric(summary, "average_runtime_ms")
    row[f"{prefix}_num_instances"] = summary_metric(summary, "num_instances")
    failures = summary.get("failure_cases", [])
    row[f"{prefix}_failure_case_count"] = len(failures) if isinstance(failures, list) else ""


def generation_metadata(path: Path) -> dict[str, Any]:
    payload = read_json_if_exists(path)
    parsed = payload.get("parsed_completion", {})
    choice = {}
    if isinstance(parsed, dict):
        choices = parsed.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
    usage = parsed.get("usage", {}) if isinstance(parsed, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    api_config = payload.get("api_config", {})
    if not isinstance(api_config, dict):
        api_config = {}
    return {
        "exists": path.exists(),
        "finish_reason": choice.get("finish_reason", ""),
        "deployment": api_config.get("deployment", ""),
        "reasoning_effort": api_config.get("reasoning_effort", ""),
        "prompt_tokens": usage.get("prompt_tokens", ""),
        "completion_tokens": usage.get("completion_tokens", ""),
        "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens", "")
        if isinstance(usage.get("completion_tokens_details"), dict)
        else "",
        "total_tokens": usage.get("total_tokens", ""),
    }


def second_scale_candidate_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent_run_dir in sorted(root.rglob("agent_run")):
        candidates_dir = agent_run_dir / "candidates"
        if not candidates_dir.exists():
            continue
        target_id, problem_from_path, family_from_path = target_parts_from_agent_run(agent_run_dir)
        dataset_manifest = agent_run_dir.parent / "dataset" / "manifest.json"
        manifest = read_json_if_exists(dataset_manifest)
        synthesis_summary = read_json_if_exists(agent_run_dir / "synthesis_summary.json")
        best_candidate = synthesis_summary.get("best_candidate", {})
        final_best_slug = best_candidate.get("slug", "") if isinstance(best_candidate, dict) else ""
        performance_history = []
        history_payload = json.loads((agent_run_dir / "performance_history.json").read_text(encoding="utf-8")) \
            if (agent_run_dir / "performance_history.json").exists() else []
        if isinstance(history_payload, list):
            performance_history = history_payload
        selected_iterations: dict[str, list[int]] = defaultdict(list)
        for item in performance_history:
            if isinstance(item, dict) and item.get("slug") is not None:
                iteration = int(item.get("iteration", -1))
                selected_iterations[str(item["slug"])].append(iteration)

        problem = str(synthesis_summary.get("problem") or manifest.get("problem") or problem_from_path)
        family = str(synthesis_summary.get("family") or manifest.get("family") or family_from_path)
        for candidate_dir in sorted(path for path in candidates_dir.iterdir() if path.is_dir()):
            slug = candidate_dir.name
            iteration, slot = parse_candidate_slug(slug)
            evaluation_dir = agent_run_dir / "evaluations" / slug
            hypothesis = read_json_if_exists(candidate_dir / "hypothesis.json")
            hyp_meta = generation_metadata(candidate_dir / "hypothesis_generation_metadata.json")
            analyze_meta = generation_metadata(candidate_dir / "analyze_generation_metadata.json")
            solution_meta = generation_metadata(candidate_dir / "solution_generation_metadata.json")
            train_summary = read_json_if_exists(evaluation_dir / "train_summary.json")
            validation_summary = read_json_if_exists(evaluation_dir / "validation_summary.json")
            test_summary = read_json_if_exists(evaluation_dir / "test_summary.json")
            has_analysis = (evaluation_dir / "analysis.json").exists()
            has_solution = (candidate_dir / "solution.py").exists()
            if test_summary:
                status = "test_evaluated"
            elif train_summary or validation_summary:
                status = "validation_evaluated"
            elif has_analysis:
                status = "analysis_only"
            elif has_solution:
                status = "solution_generated"
            else:
                status = "generated_only"
            selected = selected_iterations.get(slug, [])
            row: dict[str, Any] = {
                "experiment": "second_scale_v2",
                "source_root": str(root),
                "target_id": target_id,
                "problem": problem,
                "family": family,
                "candidate_slug": slug,
                "iteration": iteration,
                "slot": slot,
                "status": status,
                "is_final_best": slug == final_best_slug,
                "selected_iteration_count": len(selected),
                "selected_iterations": selected,
                "last_selected_iteration": max(selected) if selected else "",
                "candidate_dir": str(candidate_dir),
                "evaluation_dir": str(evaluation_dir) if evaluation_dir.exists() else "",
                "hypothesis_path": str(candidate_dir / "hypothesis.json") if (candidate_dir / "hypothesis.json").exists() else "",
                "analysis_path": str(evaluation_dir / "analysis.json") if has_analysis else "",
                "analyze_py_path": str(candidate_dir / "analyze.py") if (candidate_dir / "analyze.py").exists() else "",
                "solution_py_path": str(candidate_dir / "solution.py") if has_solution else "",
                "has_analyze_py": (candidate_dir / "analyze.py").exists(),
                "has_solution_py": has_solution,
                "has_analysis": has_analysis,
                "diversity_key": hypothesis.get("diversity_key", ""),
                "hypothesis_title": hypothesis.get("title", ""),
                "rule_summary": hypothesis.get("rule_summary", ""),
                "solver_strategy": hypothesis.get("solver_strategy", ""),
                "hypothesis_notes": read_text_if_exists(candidate_dir / "hypothesis_notes.txt"),
                "analyze_notes": read_text_if_exists(candidate_dir / "analyze_notes.txt"),
                "solution_notes": read_text_if_exists(candidate_dir / "solution_notes.txt"),
                "hypothesis_finish_reason": hyp_meta["finish_reason"],
                "analyze_finish_reason": analyze_meta["finish_reason"],
                "solution_finish_reason": solution_meta["finish_reason"],
                "model_deployment": hyp_meta["deployment"] or analyze_meta["deployment"] or solution_meta["deployment"],
                "reasoning_effort": hyp_meta["reasoning_effort"]
                or analyze_meta["reasoning_effort"]
                or solution_meta["reasoning_effort"],
                "hypothesis_total_tokens": hyp_meta["total_tokens"],
                "analyze_total_tokens": analyze_meta["total_tokens"],
                "solution_total_tokens": solution_meta["total_tokens"],
                "instance_params": manifest.get("instance_params", {}),
            }
            add_split_metrics(row, "train", train_summary)
            add_split_metrics(row, "validation", validation_summary)
            add_split_metrics(row, "test", test_summary)
            rows.append(row)
    return rows


def second_scale_candidate_summary_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_target: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_target[(str(row["problem"]), str(row["family"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for (problem, family), rows in sorted(by_target.items()):
        evaluated = [row for row in rows if row.get("validation_evaluated")]
        test_evaluated = [row for row in rows if row.get("test_evaluated")]
        best_validation = max(
            evaluated,
            key=lambda row: (
                float_or_none(row.get("validation_quality")) or 0.0,
                float_or_none(row.get("validation_optimality_rate")) or 0.0,
                -(float_or_none(row.get("validation_runtime_ms")) or 1e18),
            ),
        ) if evaluated else None
        final_best = next((row for row in rows if row.get("is_final_best")), None)
        summaries.append(
            {
                "problem": problem,
                "family": family,
                "candidate_count": len(rows),
                "validation_evaluated_count": len(evaluated),
                "test_evaluated_count": len(test_evaluated),
                "analysis_only_count": sum(1 for row in rows if row.get("status") == "analysis_only"),
                "solution_generated_count": sum(1 for row in rows if row.get("has_solution_py")),
                "final_best_slug": final_best.get("candidate_slug") if final_best else "",
                "final_best_validation_quality": final_best.get("validation_quality") if final_best else "",
                "final_best_validation_runtime_ms": final_best.get("validation_runtime_ms") if final_best else "",
                "final_best_test_quality": final_best.get("test_quality") if final_best else "",
                "final_best_test_runtime_ms": final_best.get("test_runtime_ms") if final_best else "",
                "best_validation_slug": best_validation.get("candidate_slug") if best_validation else "",
                "best_validation_quality": best_validation.get("validation_quality") if best_validation else "",
                "best_validation_runtime_ms": best_validation.get("validation_runtime_ms") if best_validation else "",
            }
        )
    return summaries


def safe_path_part(value: Any) -> str:
    text = str(value) if value is not None else "unknown"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or "unknown"


def copy_second_scale_candidate_code(candidate_rows: list[dict[str, Any]], candidate_output_dir: Path) -> list[dict[str, Any]]:
    code_root = candidate_output_dir / "code"
    if code_root.exists():
        shutil.rmtree(code_root)
    source_files = [
        "solution.py",
        "analyze.py",
        "hypothesis.json",
        "hypothesis_notes.txt",
        "analyze_notes.txt",
        "solution_notes.txt",
        "hypothesis_generation_metadata.json",
        "analyze_generation_metadata.json",
        "solution_generation_metadata.json",
    ]
    index_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        source_dir_text = str(row.get("candidate_dir") or "")
        source_dir = Path(source_dir_text)
        problem = safe_path_part(row.get("problem"))
        family = safe_path_part(row.get("family"))
        slug = safe_path_part(row.get("candidate_slug"))
        dst_dir = code_root / problem / family / slug
        copied: list[str] = []
        for file_name in source_files:
            src = source_dir / file_name
            if src.exists() and src.is_file():
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst_dir / file_name)
                copied.append(file_name)
        index_rows.append(
            {
                "problem": row.get("problem", ""),
                "family": row.get("family", ""),
                "candidate_slug": row.get("candidate_slug", ""),
                "iteration": row.get("iteration", ""),
                "slot": row.get("slot", ""),
                "status": row.get("status", ""),
                "is_final_best": row.get("is_final_best", ""),
                "validation_quality": row.get("validation_quality", ""),
                "validation_runtime_ms": row.get("validation_runtime_ms", ""),
                "test_quality": row.get("test_quality", ""),
                "test_runtime_ms": row.get("test_runtime_ms", ""),
                "source_candidate_dir": source_dir_text,
                "code_dir": str(dst_dir) if copied else "",
                "copied_file_count": len(copied),
                "copied_files": copied,
                "has_solution_py": "solution.py" in copied,
                "has_analyze_py": "analyze.py" in copied,
            }
        )
    return index_rows


def main_headline_rows(report_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in report_rows:
        if row["experiment"] == "second_scale_v2":
            targets[(str(row["problem"]), str(row["family"]))].append(row)
    target_rows: list[dict[str, Any]] = []
    for (problem, family), rows in sorted(targets.items()):
        agent = next((row for row in rows if row["role"] == "agent"), None)
        if agent is None:
            continue
        heuristics = [row for row in rows if row["role"] == "heuristic"]
        exacts = [row for row in rows if row["role"] == "exact"]
        gurobi = next((row for row in rows if row["role"] == "gurobi"), None)
        best_heuristic = max(heuristics, key=selection_key) if heuristics else None
        best_exact = max(exacts, key=selection_key) if exacts else None
        avg_heuristic_quality = mean([float_or_none(row.get("quality")) for row in heuristics])
        avg_heuristic_opt = mean([float_or_none(row.get("optimality_rate")) for row in heuristics])
        avg_heuristic_runtime = mean(
            [min(float_or_none(row.get("runtime_ms")) or 10000.0, 10000.0) for row in heuristics]
        )
        target_rows.append(
            {
                "problem": problem,
                "family": family,
                "agent_quality": agent["quality"],
                "agent_optimality_rate": agent["optimality_rate"],
                "agent_runtime_ms": agent["runtime_ms"],
                "best_heuristic_solver": best_heuristic["solver"] if best_heuristic else "",
                "best_heuristic_quality": best_heuristic["quality"] if best_heuristic else None,
                "best_heuristic_optimality_rate": best_heuristic["optimality_rate"] if best_heuristic else None,
                "best_heuristic_runtime_ms": (
                    min(float_or_none(best_heuristic["runtime_ms"]) or 10000.0, 10000.0)
                    if best_heuristic
                    else None
                ),
                "avg_heuristic_quality": avg_heuristic_quality,
                "avg_heuristic_optimality_rate": avg_heuristic_opt,
                "avg_heuristic_runtime_ms": avg_heuristic_runtime,
                "gurobi_quality": gurobi["quality"] if gurobi else None,
                "gurobi_optimality_rate": gurobi["optimality_rate"] if gurobi else None,
                "gurobi_runtime_ms": gurobi["runtime_ms"] if gurobi else None,
                "best_exact_solver": best_exact["solver"] if best_exact else "",
                "best_exact_quality": best_exact["quality"] if best_exact else None,
                "best_exact_optimality_rate": best_exact["optimality_rate"] if best_exact else None,
                "best_exact_runtime_ms": best_exact["runtime_ms"] if best_exact else None,
            }
        )

    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        by_problem[str(row["problem"])].append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for problem in sorted(by_problem):
        aggregate_rows.append(_headline_aggregate(problem, by_problem[problem]))
    aggregate_rows.append(_headline_aggregate("all", target_rows))
    return target_rows, aggregate_rows


def _headline_aggregate(problem: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    agent_q = [float_or_none(row["agent_quality"]) for row in rows]
    avg_heur_q = [float_or_none(row["avg_heuristic_quality"]) for row in rows]
    best_heur_q = [float_or_none(row["best_heuristic_quality"]) for row in rows]
    agent_t = [float_or_none(row["agent_runtime_ms"]) for row in rows]
    speed_best = []
    speed_gurobi = []
    speed_exact = []
    for row in rows:
        t_agent = float_or_none(row["agent_runtime_ms"])
        if not t_agent or t_agent <= 0:
            continue
        for output, key in [
            (speed_best, "best_heuristic_runtime_ms"),
            (speed_gurobi, "gurobi_runtime_ms"),
            (speed_exact, "best_exact_runtime_ms"),
        ]:
            value = float_or_none(row.get(key))
            if value is not None and value > 0:
                output.append(value / t_agent)
    q_llm = mean(agent_q)
    q_avg = mean(avg_heur_q)
    q_best = mean(best_heur_q)
    return {
        "problem": problem,
        "target_count": len(rows),
        "q_llm": q_llm,
        "delta_q_avg": None if q_llm is None or q_avg is None else q_llm - q_avg,
        "delta_q_best": None if q_llm is None or q_best is None else q_llm - q_best,
        "t_llm_ms": geomean([value for value in agent_t if value is not None]),
        "speedup_best": geomean(speed_best),
        "speedup_gurobi": geomean(speed_gurobi),
        "speedup_exact": geomean(speed_exact),
    }


def ml_rows(root: Path, experiment: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.rglob("run_summary.json")):
        payload = read_json(summary_path)
        if not payload.get("baseline"):
            continue
        row = {
            "experiment": experiment,
            "source_root": str(root),
            "summary_path": str(summary_path),
            "problem": payload.get("problem"),
            "family": payload.get("family"),
            "dataset_id": payload.get("dataset_id"),
            "target": payload.get("target"),
            "baseline": payload.get("baseline"),
            "device": payload.get("device"),
            "seed": payload.get("seed"),
            "num_train_instances": payload.get("num_train_instances"),
            "num_validation_instances": payload.get("num_validation_instances"),
            "num_test_instances": payload.get("num_test_instances"),
            "average_normalized_quality": payload.get("average_normalized_quality"),
            "average_objective_value": payload.get("average_objective_value"),
            "optimality_rate": payload.get("optimality_rate"),
            "feasibility_rate": payload.get("feasibility_rate"),
            "average_runtime_ms": payload.get("average_runtime_ms"),
            "error_count": payload.get("error_count"),
            "total_training_time_ms": payload.get("total_training_time_ms"),
            "trial_count": payload.get("trial_count"),
            "checkpoint_path": payload.get("checkpoint_path"),
            "run_dir": payload.get("run_dir"),
        }
        rows.append(row)
    return rows


def relabel_problem_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_problem: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_problem[row["problem"]].append(row)
    out = []
    for problem in sorted(by_problem):
        group = by_problem[problem]
        transformed = mean([float_or_none(row["transformed_quality"]) for row in group])
        original = mean([float_or_none(row["original_quality"]) for row in group])
        out.append(
            {
                "problem": problem,
                "targets": len(group),
                "original_quality": original,
                "transformed_quality": transformed,
                "delta_quality": None if original is None or transformed is None else transformed - original,
                "quality_changed_fraction": mean(
                    [float_or_none(row["quality_changed_fraction"]) for row in group]
                ),
                "optimality_changed_fraction": mean(
                    [float_or_none(row["optimality_changed_fraction"]) for row in group]
                ),
                "feasibility_changed_fraction": mean(
                    [float_or_none(row["feasibility_changed_fraction"]) for row in group]
                ),
                "runtime_ratio": geomean([float_or_none(row["runtime_ratio"]) for row in group]),
            }
        )
    out.append(
        {
            "problem": "all",
            "targets": len(rows),
            "original_quality": mean([float_or_none(row["original_quality"]) for row in rows]),
            "transformed_quality": mean([float_or_none(row["transformed_quality"]) for row in rows]),
            "delta_quality": (
                mean([float_or_none(row["transformed_quality"]) for row in rows])
                - mean([float_or_none(row["original_quality"]) for row in rows])
            ),
            "quality_changed_fraction": mean([float_or_none(row["quality_changed_fraction"]) for row in rows]),
            "optimality_changed_fraction": mean([float_or_none(row["optimality_changed_fraction"]) for row in rows]),
            "feasibility_changed_fraction": mean(
                [float_or_none(row["feasibility_changed_fraction"]) for row in rows]
            ),
            "runtime_ratio": geomean([float_or_none(row["runtime_ratio"]) for row in rows]),
        }
    )
    return out


def pace_table_rows(pace_json_path: Path) -> list[dict[str, Any]]:
    if not pace_json_path.exists():
        return []
    payload = read_json(pace_json_path)
    rows = []
    agent = payload.get("pace_evaluation_summary", {})
    agent_avg_runtime_ms = float_or_none(agent.get("average_runtime_ms"))
    agent_total_size = int_or_none(agent.get("total_solution_size"))
    agent_avg_size = float_or_none(agent.get("average_solution_size"))
    rows.append(
        {
            "solver": "agent",
            "valid": int_or_none(agent.get("feasible_count")),
            "total_size": agent_total_size,
            "total_size_m": None if agent_total_size is None else agent_total_size / 1_000_000.0,
            "avg_size": agent_avg_size,
            "agent_over_solver_size": 1.0,
            "avg_runtime_s": None if agent_avg_runtime_ms is None else agent_avg_runtime_ms / 1000.0,
            "agent_speedup": 1.0,
        }
    )
    for solver, summary in sorted((payload.get("baseline_solver_summaries") or {}).items()):
        total_size = int_or_none(summary.get("total_solution_size"))
        avg_size = float_or_none(summary.get("average_solution_size"))
        avg_runtime_ms = float_or_none(summary.get("average_runtime_ms"))
        matched_agent_sizes: list[int] = []
        matched_baseline_sizes: list[int] = []
        matched_agent_runtime_ms: list[float] = []
        matched_baseline_runtime_ms: list[float] = []
        for comparison in payload.get("instance_comparisons", []):
            if not isinstance(comparison, dict):
                continue
            agent_size = int_or_none(comparison.get("agent_solution_size"))
            agent_runtime = float_or_none(comparison.get("agent_runtime_ms"))
            for baseline in comparison.get("baselines", []):
                if not isinstance(baseline, dict) or str(baseline.get("solver")) != solver:
                    continue
                baseline_size = int_or_none(baseline.get("solution_size"))
                baseline_runtime = float_or_none(baseline.get("runtime_ms"))
                if agent_size is not None and baseline_size is not None:
                    matched_agent_sizes.append(agent_size)
                    matched_baseline_sizes.append(baseline_size)
                if agent_runtime is not None and baseline_runtime is not None:
                    matched_agent_runtime_ms.append(agent_runtime)
                    matched_baseline_runtime_ms.append(baseline_runtime)
        baseline_size_sum = sum(matched_baseline_sizes)
        agent_over_size = (
            sum(matched_agent_sizes) / baseline_size_sum
            if matched_agent_sizes and baseline_size_sum > 0
            else None if avg_size is None or agent_avg_size is None else agent_avg_size / avg_size
        )
        matched_agent_runtime = mean(matched_agent_runtime_ms)
        matched_baseline_runtime = mean(matched_baseline_runtime_ms)
        speedup = (
            matched_baseline_runtime / matched_agent_runtime
            if matched_agent_runtime is not None and matched_agent_runtime > 0 and matched_baseline_runtime is not None
            else None if avg_runtime_ms is None or agent_avg_runtime_ms is None or agent_avg_runtime_ms <= 0 else avg_runtime_ms / agent_avg_runtime_ms
        )
        rows.append(
            {
                "solver": solver,
                "valid": int_or_none(summary.get("verified_valid_count")),
                "total_size": total_size,
                "total_size_m": None if total_size is None else total_size / 1_000_000.0,
                "avg_size": avg_size,
                "agent_over_solver_size": agent_over_size,
                "avg_runtime_s": None if avg_runtime_ms is None else avg_runtime_ms / 1000.0,
                "agent_speedup": speedup,
            }
        )
    return rows


def compare_rows(
    observed: dict[str, float | int | None],
    expected: dict[str, float | int],
    *,
    label: str,
    tolerance_by_key: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for key, expected_value in expected.items():
        observed_value = observed.get(key)
        tolerance = tolerance_by_key.get(key, 0.001)
        if observed_value is None:
            status = "missing"
            delta = None
        else:
            delta = float(observed_value) - float(expected_value)
            status = "match" if abs(delta) <= tolerance else "diff"
        rows.append(
            {
                "label": label,
                "metric": key,
                "observed": observed_value,
                "paper": expected_value,
                "delta": delta,
                "tolerance": tolerance,
                "status": status,
            }
        )
    return rows


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_md_cell(value) for value in row) + " |")
    return "\n".join(lines)


def format_md_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def run_pace_collector(output_dir: Path) -> Path:
    pace_output = output_dir / "pace2025_dominating_set"
    command = [
        sys.executable,
        "-m",
        "scripts.pace2025_collect_heuristic_report",
        "--output-dir",
        str(pace_output),
    ]
    subprocess.run(command, check=True)
    return pace_output / "pace_heuristic_comparison.json"


def collect(args: argparse.Namespace) -> dict[str, Any]:
    output_dir: Path = args.output_dir
    candidate_output_dir: Path = args.candidate_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "paper": {
            "arxiv_url": PAPER_ARXIV_URL,
            "note": "Cross-reference constants were transcribed from arXiv source for tables in arXiv:2605.14141.",
        },
        "sources": {key: str(path) for key, path in RUN_SOURCES.items()},
        "outputs": {},
    }

    for name, root in RUN_SOURCES.items():
        if name == "pace2025_dominating_set":
            continue
        dst = output_dir / name
        copied = []
        for file_name in ("aggregate_results.csv", "aggregate_results.json", "benchmark_sweep_summary.json", "run_summary.json"):
            if copy_if_exists(root / file_name, dst / file_name):
                copied.append(file_name)
        manifest["outputs"][name] = {"dir": str(dst), "copied": copied}

    report_rows = []
    for name in ("second_scale_v2", "problem_size_sweep"):
        rows = benchmark_report_rows(RUN_SOURCES[name], name)
        report_rows.extend(rows)
        write_csv(output_dir / name / "solver_test_metrics.csv", rows)

    second_scale_rows = [row for row in report_rows if row["experiment"] == "second_scale_v2"]
    target_headline, headline = main_headline_rows(second_scale_rows)
    write_csv(output_dir / "second_scale_v2" / "per_target_group_metrics.csv", target_headline)
    write_csv(output_dir / "second_scale_v2" / "headline_by_problem.csv", headline)
    second_scale_candidates = second_scale_candidate_rows(RUN_SOURCES["second_scale_v2"])
    second_scale_candidate_summaries = second_scale_candidate_summary_rows(second_scale_candidates)
    second_scale_candidate_code = copy_second_scale_candidate_code(second_scale_candidates, candidate_output_dir)
    write_csv(candidate_output_dir / "candidate_metrics.csv", second_scale_candidates)
    write_json(candidate_output_dir / "candidate_metrics.json", second_scale_candidates)
    write_csv(candidate_output_dir / "candidate_summary_by_target.csv", second_scale_candidate_summaries)
    write_csv(candidate_output_dir / "candidate_code_index.csv", second_scale_candidate_code)
    write_json(candidate_output_dir / "candidate_code_index.json", second_scale_candidate_code)
    write_candidate_report(candidate_output_dir, second_scale_candidate_summaries, second_scale_candidate_code)
    write_json(candidate_output_dir / "collection_manifest.json", {
        "source": str(RUN_SOURCES["second_scale_v2"]),
        "candidate_count": len(second_scale_candidates),
        "candidate_summary_count": len(second_scale_candidate_summaries),
        "code_candidate_count": sum(1 for row in second_scale_candidate_code if row.get("copied_file_count")),
        "solution_py_count": sum(1 for row in second_scale_candidate_code if row.get("has_solution_py")),
        "analyze_py_count": sum(1 for row in second_scale_candidate_code if row.get("has_analyze_py")),
        "files": [
            "candidate_metrics.csv",
            "candidate_metrics.json",
            "candidate_summary_by_target.csv",
            "candidate_code_index.csv",
            "candidate_code_index.json",
            "candidate_results_report.md",
            "collection_manifest.json",
            "code/",
        ],
    })
    manifest["outputs"]["candidate_results"] = {
        "dir": str(candidate_output_dir),
        "source": str(RUN_SOURCES["second_scale_v2"]),
        "candidate_count": len(second_scale_candidates),
        "candidate_summary_count": len(second_scale_candidate_summaries),
        "code_candidate_count": sum(1 for row in second_scale_candidate_code if row.get("copied_file_count")),
        "solution_py_count": sum(1 for row in second_scale_candidate_code if row.get("has_solution_py")),
        "analyze_py_count": sum(1 for row in second_scale_candidate_code if row.get("has_analyze_py")),
    }

    for name in ("sample_size_sweep", "iteration_count_sweep", "no_hint_recovery"):
        rows = agent_summary_rows(RUN_SOURCES[name], name)
        write_csv(output_dir / name / "agent_test_metrics.csv", rows)
        manifest["outputs"].setdefault(name, {})["agent_metric_count"] = len(rows)

    problem_size_agent_rows = agent_summary_rows(RUN_SOURCES["problem_size_sweep"], "problem_size_sweep")
    write_csv(output_dir / "problem_size_sweep" / "agent_test_metrics.csv", problem_size_agent_rows)

    ml_gpu = ml_rows(RUN_SOURCES["ml_baselines_gpu"], "ml_baselines_gpu")
    ml_cpu = ml_rows(RUN_SOURCES["ml_baselines_cpu_eval"], "ml_baselines_cpu_eval")
    write_csv(output_dir / "ml_baselines_gpu" / "ml_baseline_results.csv", ml_gpu)
    write_csv(output_dir / "ml_baselines_cpu_eval" / "ml_baseline_results.csv", ml_cpu)
    write_csv(output_dir / "ml_baselines_combined.csv", ml_gpu + ml_cpu)
    manifest["outputs"]["ml_baselines_gpu"]["row_count"] = len(ml_gpu)
    manifest["outputs"]["ml_baselines_cpu_eval"]["row_count"] = len(ml_cpu)

    relabel_csv = RUN_SOURCES["graph_relabel_invariance"] / "aggregate_results.csv"
    relabel_rows = read_csv(relabel_csv)
    relabel_by_problem = relabel_problem_rows(relabel_rows)
    write_csv(output_dir / "graph_relabel_invariance" / "aggregate_by_problem.csv", relabel_by_problem)

    pace_json_path: Path | None = None
    if not args.skip_pace_collector:
        pace_json_path = run_pace_collector(output_dir)
    else:
        source = RUN_SOURCES["pace2025_dominating_set"] / "pace2025_ds_heuristic_llm_01" / "heuristic_comparison_report" / "pace_heuristic_comparison.json"
        dst = output_dir / "pace2025_dominating_set" / "pace_heuristic_comparison.json"
        copy_if_exists(source, dst)
        pace_json_path = dst if dst.exists() else None
    pace_rows = pace_table_rows(pace_json_path) if pace_json_path is not None else []
    write_csv(output_dir / "pace2025_dominating_set" / "pace_main_table_computed.csv", pace_rows)

    cross_reference_rows = []
    headline_by_problem = {row["problem"]: row for row in headline}
    for problem, expected in PAPER_HEADLINE_ROWS.items():
        observed = headline_by_problem.get(problem, {})
        cross_reference_rows.extend(
            compare_rows(
                observed,
                expected,
                label=f"headline:{problem}",
                tolerance_by_key={
                    "q_llm": 0.0006,
                    "delta_q_avg": 0.0008,
                    "delta_q_best": 0.0008,
                    "t_llm_ms": 0.15,
                    "speedup_best": 0.25,
                    "speedup_gurobi": 0.25,
                    "speedup_exact": 0.25,
                },
            )
        )
    pace_by_solver = {str(row["solver"]).lower(): row for row in pace_rows}
    for solver, expected in PAPER_PACE_ROWS.items():
        observed = pace_by_solver.get(solver, {})
        cross_reference_rows.extend(
            compare_rows(
                observed,
                expected,
                label=f"pace:{solver}",
                tolerance_by_key={
                    "valid": 0.0,
                    "total_size_m": 0.01,
                    "avg_size": 0.15,
                    "agent_over_solver_size": 0.001,
                    "avg_runtime_s": 0.02,
                    "agent_speedup": 0.05,
                },
            )
        )
    relabel_by_problem_map = {row["problem"]: row for row in relabel_by_problem}
    for problem, expected in PAPER_RELABEL_BY_PROBLEM.items():
        observed = relabel_by_problem_map.get(problem, {})
        cross_reference_rows.extend(
            compare_rows(
                observed,
                expected,
                label=f"relabel:{problem}",
                tolerance_by_key={
                    "targets": 0.0,
                    "original_quality": 0.0007,
                    "transformed_quality": 0.0007,
                    "delta_quality": 0.0007,
                    "quality_changed_fraction": 0.0007,
                    "optimality_changed_fraction": 0.0007,
                    "feasibility_changed_fraction": 0.0001,
                    "runtime_ratio": 0.006,
                },
            )
        )
    write_csv(output_dir / "paper_cross_reference.csv", cross_reference_rows)

    manifest["outputs"]["second_scale_v2"]["solver_metric_count"] = len(second_scale_rows)
    manifest["outputs"]["second_scale_v2"]["target_count"] = len(target_headline)
    manifest["outputs"]["paper_cross_reference"] = {
        "csv": str(output_dir / "paper_cross_reference.csv"),
        "match_count": sum(1 for row in cross_reference_rows if row["status"] == "match"),
        "diff_count": sum(1 for row in cross_reference_rows if row["status"] == "diff"),
        "missing_count": sum(1 for row in cross_reference_rows if row["status"] == "missing"),
    }
    write_json(output_dir / "collection_manifest.json", manifest)
    write_json(output_dir / "paper_reference_constants.json", {
        "headline": PAPER_HEADLINE_ROWS,
        "pace": PAPER_PACE_ROWS,
        "graph_relabel_invariance": PAPER_RELABEL_BY_PROBLEM,
        "source": PAPER_ARXIV_URL,
    })
    write_report(
        output_dir,
        manifest,
        headline,
        pace_rows,
        relabel_by_problem,
        cross_reference_rows,
    )
    return manifest


def write_candidate_report(
    candidate_output_dir: Path,
    candidate_summaries: list[dict[str, Any]],
    candidate_code_rows: list[dict[str, Any]],
) -> None:
    code_candidate_count = sum(1 for row in candidate_code_rows if row.get("copied_file_count"))
    solution_py_count = sum(1 for row in candidate_code_rows if row.get("has_solution_py"))
    analyze_py_count = sum(1 for row in candidate_code_rows if row.get("has_analyze_py"))
    lines = [
        "# Second-Scale v2 Candidate Results",
        "",
        f"- Source artifact: `{main_sweep_root()}`",
        "- `candidate_metrics.csv`: one row per synthesized candidate with generation/evaluation metrics.",
        "- `candidate_metrics.json`: JSON copy of the candidate table.",
        "- `candidate_summary_by_target.csv`: compact per-target candidate counts and selected-candidate metrics.",
        "- `candidate_code_index.csv`: one row per candidate with copied source file locations.",
        "- `code/`: copied synthesized code, notes, and generation metadata grouped by problem/family/candidate.",
        "",
        "## Code Export",
        "",
        f"- Candidates with copied text/source files: `{code_candidate_count}`",
        f"- Candidates with `solution.py`: `{solution_py_count}`",
        f"- Candidates with `analyze.py`: `{analyze_py_count}`",
        "",
        "## Summary By Target",
        "",
        markdown_table(
            [
                "problem",
                "family",
                "candidates",
                "val eval",
                "test eval",
                "final best",
                "final val Q",
                "final val ms",
                "final test Q",
                "final test ms",
            ],
            [
                [
                    row["problem"],
                    row["family"],
                    row["candidate_count"],
                    row["validation_evaluated_count"],
                    row["test_evaluated_count"],
                    row["final_best_slug"],
                    row["final_best_validation_quality"],
                    row["final_best_validation_runtime_ms"],
                    row["final_best_test_quality"],
                    row["final_best_test_runtime_ms"],
                ]
                for row in candidate_summaries
            ],
        ),
        "",
    ]
    (candidate_output_dir / "candidate_results_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    output_dir: Path,
    manifest: dict[str, Any],
    headline: list[dict[str, Any]],
    pace_rows: list[dict[str, Any]],
    relabel_by_problem: list[dict[str, Any]],
    cross_reference_rows: list[dict[str, Any]],
) -> None:
    mismatches = [row for row in cross_reference_rows if row["status"] != "match"]
    lines = [
        "# Collected Experiment Results",
        "",
        f"- Paper cross-reference source: [{PAPER_ARXIV_URL}]({PAPER_ARXIV_URL})",
        f"- Output directory: `{output_dir}`",
        "",
        "## Selected Artifact Runs",
        "",
        markdown_table(
            ["experiment", "artifact root"],
            [[name, str(path)] for name, path in RUN_SOURCES.items()],
        ),
        "",
        "## Main Benchmark Headline Recomputed",
        "",
        markdown_table(
            [
                "problem",
                "targets",
                "Q LLM",
                "dQ avg",
                "dQ best",
                "T LLM ms",
                "best speedup",
                "gurobi speedup",
                "exact speedup",
            ],
            [
                [
                    row["problem"],
                    row["target_count"],
                    row["q_llm"],
                    row["delta_q_avg"],
                    row["delta_q_best"],
                    row["t_llm_ms"],
                    row["speedup_best"],
                    row["speedup_gurobi"],
                    row["speedup_exact"],
                ]
                for row in headline
            ],
        ),
        "",
        "## PACE 2025 Dominating Set Summary",
        "",
        markdown_table(
            [
                "solver",
                "valid",
                "total size M",
                "avg size",
                "agent/solver size",
                "avg runtime s",
                "agent speedup",
            ],
            [
                [
                    row["solver"],
                    row["valid"],
                    row["total_size_m"],
                    row["avg_size"],
                    row["agent_over_solver_size"],
                    row["avg_runtime_s"],
                    row["agent_speedup"],
                ]
                for row in pace_rows
            ],
        ),
        "",
        "## Graph Relabel Invariance",
        "",
        markdown_table(
            [
                "problem",
                "targets",
                "Q orig",
                "Q relabeled",
                "dQ",
                "qual changed",
                "opt changed",
                "feas changed",
                "runtime ratio",
            ],
            [
                [
                    row["problem"],
                    row["targets"],
                    row["original_quality"],
                    row["transformed_quality"],
                    row["delta_quality"],
                    row["quality_changed_fraction"],
                    row["optimality_changed_fraction"],
                    row["feasibility_changed_fraction"],
                    row["runtime_ratio"],
                ]
                for row in relabel_by_problem
            ],
        ),
        "",
        "## Paper Cross-Reference",
        "",
        f"- Matches: `{manifest['outputs']['paper_cross_reference']['match_count']}`",
        f"- Differences: `{manifest['outputs']['paper_cross_reference']['diff_count']}`",
        f"- Missing: `{manifest['outputs']['paper_cross_reference']['missing_count']}`",
        "",
    ]
    if mismatches:
        lines.extend(
            [
                "The following values differ from the paper constants beyond the configured rounding tolerance:",
                "",
                markdown_table(
                    ["label", "metric", "observed", "paper", "delta", "tolerance", "status"],
                    [
                        [
                            row["label"],
                            row["metric"],
                            row["observed"],
                            row["paper"],
                            row["delta"],
                            row["tolerance"],
                            row["status"],
                        ]
                        for row in mismatches
                    ],
                ),
                "",
            ]
        )
    else:
        lines.append("All checked paper values match the collected artifacts within rounding tolerance.")
        lines.append("")
    lines.extend(
        [
            "## Files",
            "",
            "- `collection_manifest.json`: source run paths and output inventory.",
            "- `paper_cross_reference.csv`: observed versus paper table values.",
            "- `paper_reference_constants.json`: constants transcribed from the paper tables.",
            "- `second_scale_v2/`: main benchmark aggregate, recomputed per-target and headline CSVs.",
            "- `results/candidate_results/`: separate candidate-level second-scale v2 exports.",
            "- `sample_size_sweep/`, `problem_size_sweep/`, `iteration_count_sweep/`: sweep aggregate and agent metric CSVs.",
            "- `ml_baselines_gpu/`, `ml_baselines_cpu_eval/`, `ml_baselines_combined.csv`: ML baseline metrics.",
            "- `pace2025_dominating_set/`: PACE comparison Markdown, JSON, combined baseline CSV, and computed table.",
            "- `graph_relabel_invariance/`: aggregate and by-problem relabel metrics.",
            "- `no_hint_recovery/`: no-hint recovery aggregate and agent metric CSVs.",
        ]
    )
    (output_dir / "collected_results_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = collect(args)
    print(f"Report: {args.output_dir / 'collected_results_report.md'}")
    print(f"Manifest: {args.output_dir / 'collection_manifest.json'}")
    cross = manifest["outputs"]["paper_cross_reference"]
    print(
        "Paper checks: "
        f"{cross['match_count']} match, {cross['diff_count']} diff, {cross['missing_count']} missing"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
