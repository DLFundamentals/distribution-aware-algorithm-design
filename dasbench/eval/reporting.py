from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from dasbench.data import load_manifest, load_split
from dasbench.eval.baselines import (
    external_diagnostics_path,
    gurobi_diagnostics_path,
    resolve_baselines,
    write_external_discovery,
)
from dasbench.eval.evaluator import evaluate_solver_repeated, write_summary
from dasbench.integrations import ExternalExactConfig, GurobiBaselineConfig, NativeExactConfig
from dasbench.timing import BenchmarkTimingReporter
from dasbench.agents.candidate import build_solver, run_analysis


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return None


def _format_metric(value: float) -> str:
    return f"{value:.6f}"


def _format_optional_metric(value: object) -> str:
    if value is None:
        return "-"
    return _format_metric(float(value))


def _format_list(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _format_mapping(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    items = [f"`{key}={value[key]}`" for key in sorted(value)]
    return ", ".join(items)


def _hidden_rule_analysis(best_candidate: dict[str, object], manifest: dict[str, object]) -> dict[str, object]:
    hypothesis = best_candidate.get("hypothesis")
    if not isinstance(hypothesis, dict):
        hypothesis = {}
    ground_truth = manifest.get("ground_truth_hidden_rule")
    if not isinstance(ground_truth, dict):
        ground_truth = {}
    return {
        "agent_hypothesis": hypothesis,
        "ground_truth_hidden_rule": ground_truth,
    }


def _single_trial_to_repeated(summary: dict[str, object]) -> dict[str, object]:
    repeated = {
        "name": summary.get("name"),
        "problem": summary.get("problem"),
        "split": summary.get("split"),
        "num_instances": summary.get("num_instances"),
        "repeats": 1,
        "average_normalized_quality_mean": float(summary.get("average_normalized_quality", 0.0)),
        "average_normalized_quality_std": 0.0,
        "average_objective_value_mean": float(summary.get("average_objective_value", 0.0)),
        "average_objective_value_std": 0.0,
        "optimality_rate_mean": float(summary.get("optimality_rate", 0.0)),
        "optimality_rate_std": 0.0,
        "feasibility_rate_mean": float(summary.get("feasibility_rate", 0.0)),
        "feasibility_rate_std": 0.0,
        "average_runtime_ms_mean": float(summary.get("average_runtime_ms", 0.0)),
        "average_runtime_ms_std": 0.0,
        "representative_failure_cases": summary.get("failure_cases", []),
        "error_count": int(summary.get("error_count", 0)),
        "errors": [str(summary["error"])] if summary.get("error") else [],
        "trials": [summary],
    }
    if summary.get("average_gurobi_runtime_ms") is not None:
        repeated["average_gurobi_runtime_ms_mean"] = float(summary["average_gurobi_runtime_ms"])
        repeated["average_gurobi_runtime_ms_std"] = 0.0
        repeated["gurobi_runtime_trial_count"] = 1
    if summary.get("average_external_runtime_ms") is not None:
        repeated["average_external_runtime_ms_mean"] = float(summary["average_external_runtime_ms"])
        repeated["average_external_runtime_ms_std"] = 0.0
        repeated["external_runtime_trial_count"] = 1
    if summary.get("proved_optimal_rate") is not None:
        repeated["proved_optimal_rate_mean"] = float(summary["proved_optimal_rate"])
        repeated["proved_optimal_rate_std"] = 0.0
        repeated["proved_optimal_trial_count"] = 1
    if summary.get("average_mip_gap") is not None:
        repeated["average_mip_gap_mean"] = float(summary["average_mip_gap"])
        repeated["average_mip_gap_std"] = 0.0
    return repeated


def _load_cached_report_baselines(
    agent_run_dir: Path,
    *,
    split_name: str,
    repeats: int,
    gurobi_config: GurobiBaselineConfig,
    native_exact_config: NativeExactConfig,
    external_config: ExternalExactConfig,
) -> tuple[dict[str, dict[str, object]], dict[str, object]] | None:
    # `repeats` deliberately does not gate the cache. Repeating a report is for
    # measuring the *synthesized solver* several times; the baselines were
    # already evaluated once during the run and are reported as single
    # measurements either way. Re-running them per repeat would multiply a
    # ~417 CPU-hour baseline sweep by `repeats` to reproduce numbers that are
    # already on disk. Cached entries carry `repeats: 1` through
    # `_single_trial_to_repeated`, so the report states honestly that only the
    # agent's own row was repeated.
    run_manifest = _read_json_if_exists(agent_run_dir / "run_manifest.json")
    if run_manifest is None:
        return None
    if run_manifest.get("baseline_evaluation_skipped"):
        return None
    expected_configs = {
        "gurobi_baseline": gurobi_config.to_record(),
        "native_exact_baselines": native_exact_config.to_record(),
        "external_exact_baselines": external_config.to_record(),
    }
    for key, expected in expected_configs.items():
        observed = run_manifest.get(key)
        if observed != expected:
            return None

    baseline_path = agent_run_dir / f"baseline_{split_name}.json"
    discovery_path = agent_run_dir / "external_exact_discovery.json"
    cached_baselines = _read_json_if_exists(baseline_path)
    cached_discovery = _read_json_if_exists(discovery_path)
    if cached_baselines is None or cached_discovery is None:
        return None
    return (
        {
            str(name): _single_trial_to_repeated(summary)
            for name, summary in cached_baselines.items()
            if isinstance(summary, dict)
        },
        cached_discovery,
    )


def _copy_cached_diagnostics(
    *,
    source_dir: Path,
    output_dir: Path,
    split_name: str,
    baseline_name: str,
    gurobi_baseline_name: str,
) -> None:
    if baseline_name == gurobi_baseline_name:
        source = gurobi_diagnostics_path(source_dir, split=split_name, baseline_name=baseline_name)
        destination = gurobi_diagnostics_path(output_dir, split=split_name, baseline_name=baseline_name)
    elif baseline_name.endswith("_exact"):
        source = external_diagnostics_path(source_dir, split=split_name, baseline_name=baseline_name)
        destination = external_diagnostics_path(output_dir, split=split_name, baseline_name=baseline_name)
    else:
        return
    if source.exists() and source.resolve() != destination.resolve():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _table_rows(split_results: dict[str, dict[str, object]]) -> list[str]:
    rows = sorted(
        split_results.items(),
        key=lambda item: (
            float(item[1]["average_normalized_quality_mean"]),
            float(item[1]["optimality_rate_mean"]),
            -float(item[1]["average_runtime_ms_mean"]),
        ),
        reverse=True,
    )
    lines = [
        "| Solver | Quality Mean | Quality Std | Opt Rate | Feasibility | Proved Opt Rate | Wall Runtime Mean (ms) | Gurobi Runtime Mean (ms) | External Runtime Mean (ms) | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in rows:
        lines.append(
            "| "
            f"{name} | "
            f"{_format_metric(float(result['average_normalized_quality_mean']))} | "
            f"{_format_metric(float(result['average_normalized_quality_std']))} | "
            f"{_format_metric(float(result['optimality_rate_mean']))} | "
            f"{_format_metric(float(result['feasibility_rate_mean']))} | "
            f"{_format_optional_metric(result.get('proved_optimal_rate_mean'))} | "
            f"{_format_metric(float(result['average_runtime_ms_mean']))} | "
            f"{_format_optional_metric(result.get('average_gurobi_runtime_ms_mean'))} | "
            f"{_format_optional_metric(result.get('average_external_runtime_ms_mean'))} | "
            f"{int(result['error_count'])} |"
        )
    return lines


def build_markdown_report(
    *,
    synthesis_summary: dict[str, object],
    manifest: dict[str, object],
    repeats: int,
    split_reports: dict[str, dict[str, dict[str, object]]],
    gurobi_config: GurobiBaselineConfig,
    native_exact_config: NativeExactConfig,
    external_config: ExternalExactConfig,
    external_discovery: dict[str, object],
) -> str:
    best_candidate = synthesis_summary["best_candidate"]
    lines = [
        "# Benchmark Report",
        "",
        f"- Problem: `{manifest['problem']}`",
        f"- Family: `{manifest['family']}`",
        f"- Dataset: `{synthesis_summary['dataset_dir']}`",
        f"- Generator: `{synthesis_summary['generator']}`",
        f"- Best candidate: `{best_candidate['slug']}`",
        f"- Candidate directory: `{best_candidate['candidate_dir']}`",
        f"- Repeats per solver/split: `{repeats}`",
        "",
    ]
    split_sizes = manifest.get("split_sizes", {})
    instance_params = manifest.get("instance_params", {})
    family_params = manifest.get("family_params", {})
    lines.extend(
        [
            "## Dataset",
            "",
            f"- Train size: `{split_sizes.get('train', 'n/a')}`",
            f"- Validation size: `{split_sizes.get('validation', 'n/a')}`",
            f"- Test size: `{split_sizes.get('test', 'n/a')}`",
            f"- Instance parameters: {_format_mapping(instance_params)}",
        ]
    )
    if isinstance(family_params, dict) and family_params:
        lines.append(f"- Family parameters: {_format_mapping(family_params)}")
    lines.append("")
    if gurobi_config.enabled:
        lines.extend(
            [
                "## Industrial Baseline",
                "",
                f"- `gurobi_timed` enabled with `TimeLimit={gurobi_config.time_limit_seconds}` seconds, `Threads={gurobi_config.threads}`, `MIPGap={gurobi_config.mip_gap}`, `OutputFlag={gurobi_config.output_flag}`",
                "",
            ]
        )
    if native_exact_config.time_limit_seconds is not None:
        lines.extend(
            [
                "## Native Exact Baselines",
                "",
                f"- Problem-local exact baselines capped at `{native_exact_config.time_limit_seconds}` seconds",
                "",
            ]
        )
    if external_config.mode != "off":
        solvers = external_discovery.get("solvers", [])
        enabled_names = [
            str(item.get("baseline_name"))
            for item in solvers
            if isinstance(item, dict) and item.get("enabled")
        ]
        missing_required = [
            str(item.get("baseline_name"))
            for item in solvers
            if isinstance(item, dict) and item.get("missing_required")
        ]
        lines.extend(
            [
                "## External Exact Baselines",
                "",
                f"- Mode: `{external_config.mode}` with timeout `{external_config.time_limit_seconds}` seconds and threads `{external_config.threads}`",
                f"- Enabled: `{', '.join(enabled_names) if enabled_names else 'none'}`",
                f"- Missing required: `{', '.join(missing_required) if missing_required else 'none'}`",
                "",
            ]
        )
    hidden_rule = _hidden_rule_analysis(best_candidate, manifest)
    agent_hypothesis = hidden_rule["agent_hypothesis"]
    ground_truth = hidden_rule["ground_truth_hidden_rule"]
    if agent_hypothesis or ground_truth:
        lines.extend(["## Hidden Rule Analysis", ""])
        if agent_hypothesis:
            lines.extend(
                [
                    f"- Agent hypothesis: {agent_hypothesis.get('title', 'n/a')}",
                    f"- Agent rule summary: {agent_hypothesis.get('rule_summary', 'n/a')}",
                    f"- Agent diversity key: `{agent_hypothesis.get('diversity_key', 'n/a')}`",
                ]
            )
        if ground_truth:
            lines.extend(
                [
                    f"- Private ground truth: {ground_truth.get('summary', 'n/a')}",
                    f"- Ground-truth signals: {_format_list(ground_truth.get('signals', []))}",
                ]
            )
        lines.append("")
    lines.extend(
        [
        "## Metric",
        "",
        f"- {manifest['metric_definition']}",
        "",
        ]
    )
    for split_name, split_result in split_reports.items():
        lines.extend(
            [
                f"## {split_name.title()}",
                "",
                *_table_rows(split_result),
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def generate_benchmark_report(
    *,
    dataset_dir: Path,
    agent_run_dir: Path,
    output_dir: Path,
    repeats: int,
    include_train: bool = False,
    gurobi_config: GurobiBaselineConfig | None = None,
    native_exact_config: NativeExactConfig | None = None,
    external_config: ExternalExactConfig | None = None,
    timing_reporter: BenchmarkTimingReporter | None = None,
) -> dict[str, object]:
    synthesis_summary = _read_json(agent_run_dir / "synthesis_summary.json")
    manifest = load_manifest(dataset_dir)
    problem_name = str(manifest["problem"])
    resolved_gurobi_config = gurobi_config or GurobiBaselineConfig()
    resolved_native_exact_config = native_exact_config or NativeExactConfig()
    resolved_external_config = external_config or ExternalExactConfig()
    best_candidate = synthesis_summary["best_candidate"]
    candidate_dir = Path(best_candidate["candidate_dir"])
    train_instances = load_split(dataset_dir, "train", public=True)
    validation_instances_public = load_split(dataset_dir, "validation", public=True)
    candidate_analysis_start = time.perf_counter()
    analysis = run_analysis(
        candidate_dir,
        train_instances,
        manifest=manifest,
        artifact_dir=output_dir / "candidate_analysis",
    )
    if timing_reporter is not None:
        timing_reporter.record_stage_detail(
            "report",
            "candidate_analysis_wall_ms",
            (time.perf_counter() - candidate_analysis_start) * 1000.0,
        )
    solver = build_solver(candidate_dir, analysis=analysis, manifest=manifest)

    split_names = ["validation", "test"]
    if include_train:
        split_names.insert(0, "train")
    split_reports: dict[str, dict[str, dict[str, object]]] = {}
    baselines: dict[str, object] | None = None
    external_discovery: dict[str, object] | None = None
    for split_name in split_names:
        instances = load_split(dataset_dir, split_name)
        result: dict[str, dict[str, object]] = {}
        agent_eval_start = time.perf_counter()
        result[best_candidate["slug"]] = evaluate_solver_repeated(
            problem_name,
            best_candidate["slug"],
            solver,
            instances,
            split=split_name,
            repeats=repeats,
        )
        if timing_reporter is not None:
            timing_reporter.record_solver_evaluation(
                "report",
                split_name=split_name,
                solver_name=str(best_candidate["slug"]),
                role="agent",
                wall_ms=(time.perf_counter() - agent_eval_start) * 1000.0,
                summary=result[best_candidate["slug"]],
            )
        cached_report_baselines = _load_cached_report_baselines(
            agent_run_dir,
            split_name=split_name,
            repeats=repeats,
            gurobi_config=resolved_gurobi_config,
            native_exact_config=resolved_native_exact_config,
            external_config=resolved_external_config,
        )
        if cached_report_baselines is not None:
            cached_split_results, external_discovery = cached_report_baselines
            result.update(cached_split_results)
            if timing_reporter is not None:
                timing_reporter.record_stage_detail("report", f"reused_cached_{split_name}_baseline_results", True)
                for baseline_name, summary in cached_split_results.items():
                    _copy_cached_diagnostics(
                        source_dir=agent_run_dir,
                        output_dir=output_dir,
                        split_name=split_name,
                        baseline_name=baseline_name,
                        gurobi_baseline_name=resolved_gurobi_config.baseline_name,
                    )
                    timing_reporter.record_solver_evaluation(
                        "report",
                        split_name=split_name,
                        solver_name=baseline_name,
                        role="baseline_cached",
                        wall_ms=0.0,
                        summary=summary,
                    )
        else:
            if baselines is None:
                baselines, external_discovery = resolve_baselines(
                    problem_name,
                    gurobi_config=resolved_gurobi_config,
                    native_exact_config=resolved_native_exact_config,
                    external_config=resolved_external_config,
                    artifact_dir=output_dir,
                    train_instances=train_instances,
                    validation_instances=validation_instances_public,
                )
            for baseline_name, baseline_solver in baselines.items():
                baseline_start = time.perf_counter()
                result[baseline_name] = evaluate_solver_repeated(
                    problem_name,
                    baseline_name,
                    baseline_solver,
                    instances,
                    split=split_name,
                    repeats=repeats,
                    diagnostics_path=(
                        gurobi_diagnostics_path(output_dir, split=split_name, baseline_name=baseline_name)
                        if baseline_name == resolved_gurobi_config.baseline_name
                        else external_diagnostics_path(output_dir, split=split_name, baseline_name=baseline_name)
                        if baseline_name in baselines and baseline_name.endswith("_exact")
                        else None
                    ),
                )
                if timing_reporter is not None:
                    timing_reporter.record_solver_evaluation(
                        "report",
                        split_name=split_name,
                        solver_name=baseline_name,
                        role="baseline",
                        wall_ms=(time.perf_counter() - baseline_start) * 1000.0,
                        summary=result[baseline_name],
                    )
        split_reports[split_name] = result
    if external_discovery is None:
        _, external_discovery = resolve_baselines(
            problem_name,
            gurobi_config=resolved_gurobi_config,
            native_exact_config=resolved_native_exact_config,
            external_config=resolved_external_config,
            artifact_dir=output_dir,
        )
    write_external_discovery(output_dir, external_discovery)

    payload = {
        "dataset_dir": str(dataset_dir),
        "agent_run_dir": str(agent_run_dir),
        "repeats": repeats,
        "include_train": include_train,
        "reported_splits": split_names,
        "manifest": manifest,
        "best_candidate": best_candidate,
        "hidden_rule_analysis": _hidden_rule_analysis(best_candidate, manifest),
        "gurobi_baseline": resolved_gurobi_config.to_record(),
        "native_exact_baselines": resolved_native_exact_config.to_record(),
        "external_exact_baselines": resolved_external_config.to_record(),
        "timing_report_file": str((agent_run_dir / "timing_report.json")),
        "external_exact_discovery": external_discovery,
        "split_reports": split_reports,
    }
    json_path = output_dir / "benchmark_report.json"
    markdown_path = output_dir / "benchmark_report.md"
    write_summary(json_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        build_markdown_report(
            synthesis_summary=synthesis_summary,
            manifest=manifest,
            repeats=repeats,
            split_reports=split_reports,
            gurobi_config=resolved_gurobi_config,
            native_exact_config=resolved_native_exact_config,
            external_config=resolved_external_config,
            external_discovery=external_discovery,
        ),
        encoding="utf-8",
    )
    return {
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "split_reports": split_reports,
    }
