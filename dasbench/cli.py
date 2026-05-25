from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import time
from pathlib import Path

from dasbench.agents.llm import run_llm_no_hint_synthesis_loop, run_llm_synthesis_loop
from dasbench.agents.template import run_template_synthesis_loop
from dasbench.artifacts import default_agent_run_dir, default_dataset_dir, default_report_dir
from dasbench.data import BenchmarkSpec, generate_dataset, load_manifest, load_split
from dasbench.eval import evaluate_solver, generate_benchmark_report, write_summary
from dasbench.eval.baselines import external_diagnostics_path, gurobi_diagnostics_path, resolve_baselines
from dasbench.integrations import (
    ExternalExactConfig,
    GurobiBaselineConfig,
    NativeExactConfig,
    build_external_exact_solvers,
    build_gurobi_solver,
    load_chat_api_config,
    load_openai_dotenv,
    openai_api_is_configured,
)
from dasbench.integrations.native_exact import wrap_native_exact_baselines
from dasbench.integrations.external_exact import write_external_discovery
from dasbench.problems import available_problem_names, get_problem_definition
from dasbench.suites import benchmark_targets, run_parallel_benchmark_suite
from dasbench.timing import BenchmarkTimingReporter, timing_report_path
from dasbench.utils import parse_key_value_pairs


def _resolve_generator(generator: str) -> str:
    if generator != "auto":
        return generator
    return "llm" if openai_api_is_configured() else "template"


def _ensure_generator_ready(generator: str) -> None:
    if generator in {"llm", "llm_no_hint"}:
        load_chat_api_config(required=True)


def _build_spec_from_args(args: argparse.Namespace) -> BenchmarkSpec:
    return BenchmarkSpec(
        problem=args.problem,
        family=args.family,
        instance_params=parse_key_value_pairs(args.instance_param),
        family_params=parse_key_value_pairs(args.family_param),
        split_sizes={
            "train": args.train_size,
            "validation": args.validation_size,
            "test": args.test_size,
        },
        seeds={
            "family": args.family_seed,
            "train": args.train_seed,
            "validation": args.validation_seed,
            "test": args.test_seed,
        },
        compute_optima=args.compute_optima,
    )


def _validate_family(problem: str, family: str) -> None:
    from dasbench.families import get_family_definition

    get_family_definition(problem, family)


def _default_dataset_output(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) if args.output_dir else default_dataset_dir(args.problem, args.family, args.dataset_id)


def _default_run_output(dataset_dir: Path, run_id: str | None, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    manifest = load_manifest(dataset_dir)
    return default_agent_run_dir(str(manifest["problem"]), str(manifest["family"]), run_id)


def _default_report_output(dataset_dir: Path, agent_run_dir: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    manifest = load_manifest(dataset_dir)
    return default_report_dir(str(manifest["problem"]), str(manifest["family"]), agent_run_dir.name)


def _read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_manifest_exists(dataset_dir: Path) -> bool:
    return (dataset_dir / "manifest.json").exists()


def _load_cached_baseline_bundle(
    output_dir: Path,
    *,
    gurobi_config: GurobiBaselineConfig,
    native_exact_config: NativeExactConfig,
    external_config: ExternalExactConfig,
) -> tuple[dict[str, dict[str, dict[str, object]]], dict[str, object]] | None:
    validation_path = output_dir / "baseline_validation.json"
    test_path = output_dir / "baseline_test.json"
    discovery_path = output_dir / "external_exact_discovery.json"
    if not (validation_path.exists() and test_path.exists() and discovery_path.exists()):
        return None

    timing_payload = _read_json_if_exists(timing_report_path(output_dir)) or {}
    stages = timing_payload.get("stages", {})
    if isinstance(stages, dict):
        baseline_stage = stages.get("baseline_pre_synthesis", {})
        if isinstance(baseline_stage, dict):
            if baseline_stage.get("status") not in {"completed", "skipped"}:
                return None
            if baseline_stage.get("reason") == "baseline_evaluation_disabled":
                return None
            expected_configs = {
                "gurobi_baseline": gurobi_config.to_record(),
                "native_exact_baselines": native_exact_config.to_record(),
                "external_exact_baselines": external_config.to_record(),
            }
            for key, expected in expected_configs.items():
                observed = baseline_stage.get(key)
                if observed is not None and observed != expected:
                    return None

    validation = _read_json_if_exists(validation_path)
    test = _read_json_if_exists(test_path)
    discovery = _read_json_if_exists(discovery_path)
    if not (isinstance(validation, dict) and isinstance(test, dict) and isinstance(discovery, dict)):
        return None
    return {"validation": validation, "test": test}, discovery


def _baseline_stage_config_matches(
    output_dir: Path,
    *,
    gurobi_config: GurobiBaselineConfig,
    native_exact_config: NativeExactConfig,
    external_config: ExternalExactConfig,
) -> bool:
    timing_payload = _read_json_if_exists(timing_report_path(output_dir)) or {}
    stages = timing_payload.get("stages", {})
    if not isinstance(stages, dict):
        return True
    baseline_stage = stages.get("baseline_pre_synthesis", {})
    if not isinstance(baseline_stage, dict):
        return True
    expected_configs = {
        "gurobi_baseline": gurobi_config.to_record(),
        "native_exact_baselines": native_exact_config.to_record(),
        "external_exact_baselines": external_config.to_record(),
    }
    for key, expected in expected_configs.items():
        observed = baseline_stage.get(key)
        if observed is not None and observed != expected:
            return False
    return True


def _load_partial_baseline_summaries(
    output_dir: Path,
    *,
    split_name: str,
    gurobi_config: GurobiBaselineConfig,
    native_exact_config: NativeExactConfig,
    external_config: ExternalExactConfig,
) -> dict[str, dict[str, object]]:
    if not _baseline_stage_config_matches(
        output_dir,
        gurobi_config=gurobi_config,
        native_exact_config=native_exact_config,
        external_config=external_config,
    ):
        return {}

    summaries: dict[str, dict[str, object]] = {}
    persisted = _read_json_if_exists(output_dir / f"baseline_{split_name}.json")
    if isinstance(persisted, dict):
        for solver_name, summary in persisted.items():
            if isinstance(summary, dict):
                summaries[str(solver_name)] = summary

    timing_payload = _read_json_if_exists(timing_report_path(output_dir)) or {}
    stages = timing_payload.get("stages", {})
    if not isinstance(stages, dict):
        return summaries
    baseline_stage = stages.get("baseline_pre_synthesis", {})
    if not isinstance(baseline_stage, dict):
        return summaries
    splits = baseline_stage.get("splits", {})
    if not isinstance(splits, dict):
        return summaries
    split_payload = splits.get(split_name, {})
    if not isinstance(split_payload, dict):
        return summaries
    solvers = split_payload.get("solvers", {})
    if not isinstance(solvers, dict):
        return summaries
    for solver_name, payload in solvers.items():
        if solver_name in summaries or not isinstance(payload, dict):
            continue
        summary = payload.get("summary")
        if isinstance(summary, dict):
            summaries[str(solver_name)] = summary
    return summaries


def _gurobi_config_from_args(
    args: argparse.Namespace,
    *,
    fallback: dict[str, object] | None = None,
) -> GurobiBaselineConfig:
    fallback_config = GurobiBaselineConfig.from_record(fallback)
    enabled = getattr(args, "gurobi_baseline_enabled", None)
    time_limit = getattr(args, "gurobi_time_limit_seconds", None)
    threads = getattr(args, "gurobi_threads", None)
    return GurobiBaselineConfig(
        enabled=fallback_config.enabled if enabled is None else bool(enabled),
        time_limit_seconds=fallback_config.time_limit_seconds if time_limit is None else float(time_limit),
        threads=fallback_config.threads if threads is None else int(threads),
        output_flag=fallback_config.output_flag,
        mip_gap=fallback_config.mip_gap,
        baseline_name=fallback_config.baseline_name,
    )


def _external_config_from_args(
    args: argparse.Namespace,
    *,
    fallback: dict[str, object] | None = None,
) -> ExternalExactConfig:
    fallback_config = ExternalExactConfig.from_record(fallback)
    mode = getattr(args, "external_exact_baselines", None)
    time_limit = getattr(args, "external_time_limit_seconds", None)
    threads = getattr(args, "external_threads", None)
    solver_config = getattr(args, "external_solver_config", None)
    return ExternalExactConfig(
        mode=fallback_config.mode if mode is None else str(mode),
        time_limit_seconds=fallback_config.time_limit_seconds if time_limit is None else float(time_limit),
        threads=fallback_config.threads if threads is None else int(threads),
        solver_config_path=(
            fallback_config.solver_config_path
            if solver_config is None
            else str(solver_config)
        ),
        solver_paths=fallback_config.solver_paths,
    )


def _native_exact_config_from_args(
    args: argparse.Namespace,
    *,
    fallback: dict[str, object] | None = None,
) -> NativeExactConfig:
    fallback_config = NativeExactConfig.from_record(fallback)
    time_limit = getattr(args, "native_exact_time_limit_seconds", None)
    return NativeExactConfig(
        time_limit_seconds=(
            fallback_config.time_limit_seconds
            if time_limit is None
            else float(time_limit)
        ),
    )


def _print_result(label: str, summary: dict[str, object]) -> None:
    print(
        f"{label}: quality={summary['average_normalized_quality']:.4f} "
        f"opt_rate={summary['optimality_rate']:.4f} "
        f"feasibility={summary['feasibility_rate']:.4f} "
        f"runtime_ms={summary['average_runtime_ms']:.3f}"
    )


def _resolve_single_baseline(
    problem_name: str,
    baseline_name: str,
    *,
    output_dir: Path,
    gurobi_config: GurobiBaselineConfig,
    native_exact_config: NativeExactConfig,
    external_config: ExternalExactConfig,
):
    if baseline_name == gurobi_config.baseline_name and gurobi_config.enabled:
        return build_gurobi_solver(problem_name, gurobi_config)

    problem = get_problem_definition(problem_name)
    local_baselines = wrap_native_exact_baselines(dict(problem.baseline_registry()), native_exact_config)
    if baseline_name in local_baselines:
        return local_baselines[baseline_name]

    external_baselines, _ = build_external_exact_solvers(
        problem_name,
        external_config,
        artifact_dir=output_dir / "external_exact_logs",
    )
    if baseline_name in external_baselines:
        return external_baselines[baseline_name]

    raise KeyError(f"Unknown baseline `{baseline_name}` for problem `{problem_name}`.")


def _evaluate_single_baseline_task(
    *,
    dataset_dir: str,
    output_dir: str,
    problem_name: str,
    split_name: str,
    baseline_name: str,
    gurobi_record: dict[str, object],
    native_record: dict[str, object],
    external_record: dict[str, object],
) -> tuple[str, dict[str, object], float]:
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    gurobi_config = GurobiBaselineConfig.from_record(gurobi_record)
    native_exact_config = NativeExactConfig.from_record(native_record)
    external_config = ExternalExactConfig.from_record(external_record)
    baseline_solver = _resolve_single_baseline(
        problem_name,
        baseline_name,
        output_dir=output_path,
        gurobi_config=gurobi_config,
        native_exact_config=native_exact_config,
        external_config=external_config,
    )
    instances = load_split(dataset_path, split_name)
    start = time.perf_counter()
    summary = evaluate_solver(
        problem_name,
        baseline_name,
        baseline_solver,
        instances,
        split=split_name,
        diagnostics_path=(
            gurobi_diagnostics_path(output_path, split=split_name, baseline_name=baseline_name)
            if baseline_name == gurobi_config.baseline_name
            else external_diagnostics_path(output_path, split=split_name, baseline_name=baseline_name)
            if baseline_name.endswith("_exact")
            else None
        ),
    )
    wall_ms = (time.perf_counter() - start) * 1000.0
    return baseline_name, summary, wall_ms


def _run_baselines(
    dataset_dir: Path,
    output_dir: Path,
    *,
    gurobi_config: GurobiBaselineConfig,
    native_exact_config: NativeExactConfig,
    external_config: ExternalExactConfig,
    baseline_workers: int = 1,
    timing_reporter: BenchmarkTimingReporter | None = None,
) -> tuple[dict[str, dict[str, dict[str, object]]], dict[str, object]]:
    manifest = load_manifest(dataset_dir)
    train_public = load_split(dataset_dir, "train", public=True)
    validation_public = load_split(dataset_dir, "validation", public=True)
    baselines, external_discovery = resolve_baselines(
        str(manifest["problem"]),
        gurobi_config=gurobi_config,
        native_exact_config=native_exact_config,
        external_config=external_config,
        artifact_dir=output_dir,
        train_instances=train_public,
        validation_instances=validation_public,
    )
    write_external_discovery(output_dir, external_discovery)
    split_results: dict[str, dict[str, dict[str, object]]] = {}
    for split_name in ("validation", "test"):
        split_summaries = _load_partial_baseline_summaries(
            output_dir,
            split_name=split_name,
            gurobi_config=gurobi_config,
            native_exact_config=native_exact_config,
            external_config=external_config,
        )
        problem_name = str(manifest["problem"])
        baseline_names = list(baselines.keys())
        split_summaries = {
            baseline_name: split_summaries[baseline_name]
            for baseline_name in baseline_names
            if baseline_name in split_summaries
        }
        write_summary(output_dir / f"baseline_{split_name}.json", split_summaries)
        pending_baseline_names = [
            baseline_name
            for baseline_name in baseline_names
            if baseline_name not in split_summaries
        ]
        worker_count = max(1, int(baseline_workers))
        if any(name.startswith("ml_") for name in pending_baseline_names):
            worker_count = 1
        if not pending_baseline_names:
            split_results[split_name] = split_summaries
            continue
        if worker_count <= 1 or len(pending_baseline_names) <= 1:
            instances = load_split(dataset_dir, split_name)
            for baseline_name in pending_baseline_names:
                baseline_solver = baselines[baseline_name]
                baseline_start = time.perf_counter()
                split_summaries[baseline_name] = evaluate_solver(
                    problem_name,
                    baseline_name,
                    baseline_solver,
                    instances,
                    split=split_name,
                    diagnostics_path=(
                        gurobi_diagnostics_path(output_dir, split=split_name, baseline_name=baseline_name)
                        if baseline_name == gurobi_config.baseline_name
                        else external_diagnostics_path(output_dir, split=split_name, baseline_name=baseline_name)
                        if baseline_name.endswith("_exact")
                        else None
                    ),
                )
                if timing_reporter is not None:
                    timing_reporter.record_solver_evaluation(
                        "baseline_pre_synthesis",
                        split_name=split_name,
                        solver_name=baseline_name,
                        role="baseline",
                        wall_ms=(time.perf_counter() - baseline_start) * 1000.0,
                            summary=split_summaries[baseline_name],
                        )
                write_summary(output_dir / f"baseline_{split_name}.json", split_summaries)
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(worker_count, len(pending_baseline_names)),
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                future_to_baseline = {
                    executor.submit(
                        _evaluate_single_baseline_task,
                        dataset_dir=str(dataset_dir),
                        output_dir=str(output_dir),
                        problem_name=problem_name,
                        split_name=split_name,
                        baseline_name=baseline_name,
                        gurobi_record=gurobi_config.to_record(),
                        native_record=native_exact_config.to_record(),
                        external_record=external_config.to_record(),
                    ): baseline_name
                    for baseline_name in pending_baseline_names
                }
                for future in concurrent.futures.as_completed(future_to_baseline):
                    baseline_name, summary, wall_ms = future.result()
                    split_summaries[baseline_name] = summary
                    if timing_reporter is not None:
                        timing_reporter.record_solver_evaluation(
                            "baseline_pre_synthesis",
                            split_name=split_name,
                            solver_name=baseline_name,
                            role="baseline",
                            wall_ms=wall_ms,
                            summary=summary,
                        )
                    write_summary(output_dir / f"baseline_{split_name}.json", split_summaries)
        split_results[split_name] = split_summaries
    return split_results, external_discovery


def cmd_generate(args: argparse.Namespace) -> int:
    _validate_family(args.problem, args.family)
    spec = _build_spec_from_args(args)
    output_dir = _default_dataset_output(args)
    generate_dataset(output_dir, spec)
    print(f"Generated dataset at {output_dir}")
    print(json.dumps(load_manifest(output_dir), indent=2, sort_keys=True))
    return 0


def cmd_run_agent(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir)
    manifest = load_manifest(dataset_dir)
    output_dir = _default_run_output(dataset_dir, args.run_id, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_reporter = BenchmarkTimingReporter(
        timing_report_path(output_dir),
        metadata={
            "problem": manifest["problem"],
            "family": manifest["family"],
            "dataset_dir": str(dataset_dir),
            "agent_run_dir": str(output_dir),
        },
    )
    gurobi_config = _gurobi_config_from_args(args)
    native_exact_config = _native_exact_config_from_args(args)
    external_config = _external_config_from_args(args)
    existing_synthesis = _read_json_if_exists(output_dir / "synthesis_summary.json")

    baseline_results: dict[str, dict[str, dict[str, object]]] | None = None
    external_discovery: dict[str, object] | None = None
    baseline_future: concurrent.futures.Future[tuple[dict[str, dict[str, dict[str, object]]], dict[str, object]]] | None = None
    baseline_executor: concurrent.futures.ThreadPoolExecutor | None = None
    baseline_stage_start: float | None = None
    overlap_baselines_with_synthesis = bool(
        getattr(args, "overlap_baselines_with_synthesis", True)
        and not getattr(args, "skip_baselines", False)
    )
    cached_baselines = None
    if not getattr(args, "skip_baselines", False):
        cached_baselines = _load_cached_baseline_bundle(
            output_dir,
            gurobi_config=gurobi_config,
            native_exact_config=native_exact_config,
            external_config=external_config,
        )
    if (
        isinstance(existing_synthesis, dict)
        and (
            getattr(args, "skip_baselines", False)
            or cached_baselines is not None
        )
    ):
        timing_reporter.mark_status("agent_run_completed")
        print(f"Agent run directory: {output_dir}")
        print("Generator: reused_existing")
        return 0

    timing_reporter.mark_status("running")

    if getattr(args, "skip_baselines", False):
        baseline_results = {"validation": {}, "test": {}}
        external_discovery = {
            "mode": external_config.mode,
            "solver_config_path": external_config.solver_config_path,
            "solvers": [],
            "skipped": True,
        }
        write_summary(output_dir / "baseline_validation.json", baseline_results["validation"])
        write_summary(output_dir / "baseline_test.json", baseline_results["test"])
        timing_reporter.stage_skipped(
            "baseline_pre_synthesis",
            reason="baseline_evaluation_disabled",
            extra={
                "gurobi_baseline": gurobi_config.to_record(),
                "native_exact_baselines": native_exact_config.to_record(),
                "external_exact_baselines": external_config.to_record(),
            },
        )
    else:
        if cached_baselines is not None:
            baseline_results, external_discovery = cached_baselines
            timing_reporter.stage_skipped(
                "baseline_pre_synthesis",
                reason="existing_baselines_reused",
                extra={
                    "gurobi_baseline": gurobi_config.to_record(),
                    "native_exact_baselines": native_exact_config.to_record(),
                    "external_exact_baselines": external_config.to_record(),
                },
            )
        else:
            baseline_stage_start = time.perf_counter()
            baseline_stage_extra = {
                "gurobi_baseline": gurobi_config.to_record(),
                "native_exact_baselines": native_exact_config.to_record(),
                "external_exact_baselines": external_config.to_record(),
                "baseline_workers": int(getattr(args, "baseline_workers", 1)),
            }
            timing_reporter.stage_started(
                "baseline_pre_synthesis",
                extra=baseline_stage_extra,
            )

            if overlap_baselines_with_synthesis and not isinstance(existing_synthesis, dict):
                baseline_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                baseline_future = baseline_executor.submit(
                    _run_baselines,
                    dataset_dir,
                    output_dir,
                    gurobi_config=gurobi_config,
                    native_exact_config=native_exact_config,
                    external_config=external_config,
                    baseline_workers=int(getattr(args, "baseline_workers", 1)),
                    timing_reporter=timing_reporter,
                )
            else:
                try:
                    baseline_results, external_discovery = _run_baselines(
                        dataset_dir,
                        output_dir,
                        gurobi_config=gurobi_config,
                        native_exact_config=native_exact_config,
                        external_config=external_config,
                        baseline_workers=int(getattr(args, "baseline_workers", 1)),
                        timing_reporter=timing_reporter,
                    )
                    timing_reporter.stage_completed(
                        "baseline_pre_synthesis",
                        wall_ms=(time.perf_counter() - baseline_stage_start) * 1000.0,
                    )
                except Exception as exc:
                    timing_reporter.stage_failed(
                        "baseline_pre_synthesis",
                        error=f"{type(exc).__name__}: {exc}",
                        wall_ms=(time.perf_counter() - baseline_stage_start) * 1000.0,
                    )
                    timing_reporter.mark_status("failed", error=f"{type(exc).__name__}: {exc}")
                    raise
    generator = _resolve_generator(args.generator)
    _ensure_generator_ready(generator)
    if generator == "llm":
        runner = run_llm_synthesis_loop
    elif generator == "llm_no_hint":
        runner = run_llm_no_hint_synthesis_loop
    else:
        runner = run_template_synthesis_loop
    synthesis_extra = {
        "generator": generator,
        "mode": args.mode,
        "iterations": args.iterations,
        "beam_width": args.beam_width,
        "candidate_width": getattr(args, "candidate_width", None),
    }
    synthesis_error: Exception | None = None
    if isinstance(existing_synthesis, dict):
        synthesis_summary = existing_synthesis
        timing_reporter.stage_skipped(
            "synthesis",
            reason="existing_synthesis_reused",
            extra={
                **synthesis_extra,
                "best_candidate_slug": existing_synthesis.get("best_candidate", {}).get("slug")
                if isinstance(existing_synthesis.get("best_candidate"), dict)
                else None,
            },
        )
    else:
        synthesis_stage_start = time.perf_counter()
        timing_reporter.stage_started(
            "synthesis",
            extra=synthesis_extra,
        )
        try:
            synthesis_summary = runner(
                dataset_dir,
                output_dir,
                mode=args.mode,
                iterations=args.iterations,
                beam_width=args.beam_width,
                candidate_width=getattr(args, "candidate_width", None),
                timing_reporter=timing_reporter,
            )
            timing_reporter.stage_completed(
                "synthesis",
                wall_ms=(time.perf_counter() - synthesis_stage_start) * 1000.0,
                extra={
                    **synthesis_extra,
                    "best_candidate_slug": synthesis_summary["best_candidate"]["slug"],
                },
            )
        except Exception as exc:
            synthesis_error = exc
            timing_reporter.stage_failed(
                "synthesis",
                error=f"{type(exc).__name__}: {exc}",
                wall_ms=(time.perf_counter() - synthesis_stage_start) * 1000.0,
            )
            timing_reporter.mark_status("failed", error=f"{type(exc).__name__}: {exc}")
            synthesis_summary = None

    if baseline_future is not None:
        try:
            baseline_results, external_discovery = baseline_future.result()
            timing_reporter.stage_completed(
                "baseline_pre_synthesis",
                wall_ms=(time.perf_counter() - float(baseline_stage_start)) * 1000.0,
            )
        except Exception as exc:
            timing_reporter.stage_failed(
                "baseline_pre_synthesis",
                error=f"{type(exc).__name__}: {exc}",
                wall_ms=(time.perf_counter() - float(baseline_stage_start)) * 1000.0,
            )
            timing_reporter.mark_status("failed", error=f"{type(exc).__name__}: {exc}")
            if baseline_executor is not None:
                baseline_executor.shutdown(wait=False)
            raise
        finally:
            if baseline_executor is not None:
                baseline_executor.shutdown(wait=True)

    if synthesis_error is not None:
        raise synthesis_error

    assert baseline_results is not None
    assert external_discovery is not None
    assert synthesis_summary is not None
    external_baseline_names = [
        str(item["baseline_name"])
        for item in external_discovery.get("solvers", [])
        if isinstance(item, dict) and (item.get("enabled") or item.get("missing_required"))
    ]
    run_manifest = {
        "dataset_dir": str(dataset_dir),
        "agent_run_dir": str(output_dir),
        "problem": manifest["problem"],
        "family": manifest["family"],
        "generator": generator,
        "mode": args.mode,
        "iterations": args.iterations,
        "beam_width": args.beam_width,
        "candidate_width": synthesis_summary.get("candidate_width", getattr(args, "candidate_width", None)),
        "baseline_evaluation_skipped": bool(getattr(args, "skip_baselines", False)),
        "overlap_baselines_with_synthesis": bool(overlap_baselines_with_synthesis),
        "baseline_workers": int(getattr(args, "baseline_workers", 1)),
        "gurobi_baseline": gurobi_config.to_record(),
        "native_exact_baselines": native_exact_config.to_record(),
        "external_exact_baselines": external_config.to_record(),
        "timing_report_file": str(timing_report_path(output_dir)),
        "external_exact_discovery": external_discovery,
        "external_exact_discovery_file": str(output_dir / "external_exact_discovery.json"),
        "baseline_files": {
            "validation": str(output_dir / "baseline_validation.json"),
            "test": str(output_dir / "baseline_test.json"),
        },
        "gurobi_diagnostics_files": {
            split_name: str(gurobi_diagnostics_path(output_dir, split=split_name, baseline_name=gurobi_config.baseline_name))
            for split_name in ("test",)
            if gurobi_config.enabled
        },
        "external_exact_diagnostics_files": {
            baseline_name: {
                split_name: str(external_diagnostics_path(output_dir, split=split_name, baseline_name=baseline_name))
                for split_name in ("test",)
            }
            for baseline_name in external_baseline_names
        },
    }
    write_summary(output_dir / "run_manifest.json", run_manifest)

    print(f"Agent run directory: {output_dir}")
    print(f"Generator: {generator}")
    print("Test baselines")
    for name, summary in baseline_results["test"].items():
        _print_result(name, summary)
    print("Best candidate")
    _print_result("train", synthesis_summary["best_candidate"]["train"])
    _print_result("validation", synthesis_summary["best_candidate"]["validation"])
    _print_result("test", synthesis_summary["best_candidate"]["test"])
    timing_reporter.mark_status("agent_run_completed")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir)
    agent_run_dir = Path(args.agent_run_dir)
    output_dir = _default_report_output(dataset_dir, agent_run_dir, args.output_dir)
    manifest = load_manifest(dataset_dir)
    timing_reporter = BenchmarkTimingReporter(
        timing_report_path(agent_run_dir),
        metadata={
            "problem": manifest["problem"],
            "family": manifest["family"],
            "dataset_dir": str(dataset_dir),
            "agent_run_dir": str(agent_run_dir),
            "report_dir": str(output_dir),
        },
    )
    timing_reporter.mark_status("running")
    run_manifest = _read_json_if_exists(agent_run_dir / "run_manifest.json") or {}
    gurobi_config = _gurobi_config_from_args(args, fallback=run_manifest.get("gurobi_baseline"))
    native_exact_config = _native_exact_config_from_args(
        args,
        fallback=run_manifest.get("native_exact_baselines"),
    )
    external_config = _external_config_from_args(
        args,
        fallback=run_manifest.get("external_exact_baselines"),
    )
    report_stage_start = time.perf_counter()
    timing_reporter.stage_started(
        "report",
        extra={
            "repeats": args.repeats,
            "report_dir": str(output_dir),
        },
    )
    try:
        report = generate_benchmark_report(
            dataset_dir=dataset_dir,
            agent_run_dir=agent_run_dir,
            output_dir=output_dir,
            repeats=args.repeats,
            include_train=args.include_train,
            gurobi_config=gurobi_config,
            native_exact_config=native_exact_config,
            external_config=external_config,
            timing_reporter=timing_reporter,
        )
        timing_reporter.stage_completed(
            "report",
            wall_ms=(time.perf_counter() - report_stage_start) * 1000.0,
        )
        timing_reporter.mark_status("completed")
    except Exception as exc:
        timing_reporter.stage_failed(
            "report",
            error=f"{type(exc).__name__}: {exc}",
            wall_ms=(time.perf_counter() - report_stage_start) * 1000.0,
        )
        timing_reporter.mark_status("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    print(f"Report directory: {output_dir}")
    print(f"Markdown report: {report['markdown_path']}")
    print(f"JSON report: {report['json_path']}")
    return 0


def _run_benchmark_target(
    args: argparse.Namespace,
    *,
    problem: str,
    family: str,
    dataset_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, object]:
    target_args = argparse.Namespace(**vars(args))
    target_args.problem = problem
    target_args.family = family
    target_args.dataset_id = dataset_id if dataset_id is not None else args.dataset_id
    target_args.run_id = run_id if run_id is not None else args.run_id

    _validate_family(problem, family)
    dataset_dir = Path(target_args.dataset_dir) if target_args.dataset_dir else _default_dataset_output(target_args)
    run_dir = Path(target_args.run_output_dir) if target_args.run_output_dir else default_agent_run_dir(problem, family, target_args.run_id)
    report_dir = Path(target_args.report_output_dir) if target_args.report_output_dir else default_report_dir(problem, family, run_dir.name)
    timing_reporter = BenchmarkTimingReporter(
        timing_report_path(run_dir),
        metadata={
            "problem": problem,
            "family": family,
            "dataset_dir": str(dataset_dir),
            "agent_run_dir": str(run_dir),
            "report_dir": str(report_dir),
        },
    )
    timing_reporter.mark_status("running")
    if not _dataset_manifest_exists(dataset_dir) or args.force_regenerate:
        spec = _build_spec_from_args(target_args)
        dataset_stage_start = time.perf_counter()
        timing_reporter.stage_started(
            "dataset_generation",
            extra={
                "compute_optima": bool(spec.compute_optima),
                "split_sizes": dict(spec.split_sizes),
                "instance_params": dict(spec.instance_params),
                "family_params": dict(spec.family_params),
            },
        )
        try:
            dataset = generate_dataset(dataset_dir, spec)
            timing_reporter.record_dataset_summary(dataset, compute_optima=spec.compute_optima)
            timing_reporter.stage_completed(
                "dataset_generation",
                wall_ms=(time.perf_counter() - dataset_stage_start) * 1000.0,
            )
        except Exception as exc:
            timing_reporter.stage_failed(
                "dataset_generation",
                error=f"{type(exc).__name__}: {exc}",
                wall_ms=(time.perf_counter() - dataset_stage_start) * 1000.0,
            )
            timing_reporter.mark_status("failed", error=f"{type(exc).__name__}: {exc}")
            raise
    else:
        timing_reporter.stage_skipped(
            "dataset_generation",
            reason="existing_dataset_reused",
        )

    run_args = argparse.Namespace(
        dataset_dir=str(dataset_dir),
        run_id=target_args.run_id,
        output_dir=str(run_dir),
        generator=target_args.generator,
        mode=target_args.mode,
        iterations=target_args.iterations,
        beam_width=target_args.beam_width,
        candidate_width=getattr(target_args, "candidate_width", None),
        gurobi_baseline_enabled=target_args.gurobi_baseline_enabled,
        gurobi_time_limit_seconds=target_args.gurobi_time_limit_seconds,
        gurobi_threads=target_args.gurobi_threads,
        native_exact_time_limit_seconds=getattr(target_args, "native_exact_time_limit_seconds", None),
        external_exact_baselines=target_args.external_exact_baselines,
        external_time_limit_seconds=target_args.external_time_limit_seconds,
        external_threads=target_args.external_threads,
        external_solver_config=target_args.external_solver_config,
        skip_baselines=getattr(target_args, "skip_baselines", False),
        overlap_baselines_with_synthesis=getattr(target_args, "overlap_baselines_with_synthesis", True),
    )
    cmd_run_agent(run_args)

    if not getattr(target_args, "skip_report", False):
        report_args = argparse.Namespace(
            dataset_dir=str(dataset_dir),
            agent_run_dir=str(run_dir),
            output_dir=str(report_dir),
            repeats=target_args.repeats,
            include_train=target_args.include_train,
            gurobi_baseline_enabled=target_args.gurobi_baseline_enabled,
            gurobi_time_limit_seconds=target_args.gurobi_time_limit_seconds,
            gurobi_threads=target_args.gurobi_threads,
            native_exact_time_limit_seconds=getattr(target_args, "native_exact_time_limit_seconds", None),
            external_exact_baselines=target_args.external_exact_baselines,
            external_time_limit_seconds=target_args.external_time_limit_seconds,
            external_threads=target_args.external_threads,
            external_solver_config=target_args.external_solver_config,
        )
        cmd_report(report_args)

    return {
        "problem": problem,
        "family": family,
        "dataset_dir": str(dataset_dir),
        "agent_run_dir": str(run_dir),
        "report_dir": str(report_dir),
    }


def cmd_benchmark(args: argparse.Namespace) -> int:
    targets = benchmark_targets(args)
    if len(targets) == 1:
        problem, family = targets[0]
        summary = _run_benchmark_target(args, problem=problem, family=family)
        write_summary(Path(summary["report_dir"]) / "benchmark_summary.json", summary)
        return 0
    return run_parallel_benchmark_suite(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="dasbench: unified distribution-aware synthesis benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_gurobi_arguments(
        command: argparse.ArgumentParser,
        *,
        enabled_default: bool | None,
        time_limit_default: float | None,
        threads_default: int | None,
    ) -> None:
        command.add_argument("--no-gurobi-baseline", dest="gurobi_baseline_enabled", action="store_false")
        command.set_defaults(gurobi_baseline_enabled=enabled_default)
        command.add_argument("--gurobi-time-limit-seconds", type=float, default=time_limit_default)
        command.add_argument("--gurobi-threads", type=int, default=threads_default)

    def add_native_exact_arguments(
        command: argparse.ArgumentParser,
        *,
        time_limit_default: float | None,
    ) -> None:
        command.add_argument(
            "--native-exact-time-limit-seconds",
            type=float,
            default=time_limit_default,
            help=(
                "Optional wall-clock timeout for exact baselines defined directly in the problem registry "
                "(for example CP-SAT, Held-Karp, or local branch-and-bound implementations)."
            ),
        )

    def add_external_exact_arguments(
        command: argparse.ArgumentParser,
        *,
        mode_default: str | None,
        time_limit_default: float | None,
        threads_default: int | None,
    ) -> None:
        command.add_argument(
            "--external-exact-baselines",
            choices=["auto", "off", "required"],
            default=mode_default,
            help=(
                "Controls optional local binary exact baselines such as Concorde, KaMIS, and MaxHS. "
                "Package-backed exact baselines from pyproject.toml run automatically when available."
            ),
        )
        command.add_argument("--external-time-limit-seconds", type=float, default=time_limit_default)
        command.add_argument("--external-threads", type=int, default=threads_default)
        command.add_argument("--external-solver-config")

    def add_overlap_arguments(
        command: argparse.ArgumentParser,
        *,
        enabled_default: bool,
    ) -> None:
        command.add_argument(
            "--overlap-baselines-with-synthesis",
            dest="overlap_baselines_with_synthesis",
            action="store_true",
            help=(
                "Run the pre-synthesis baseline pass in the background while synthesis is running. "
                "This can reduce per-target wall time, but it increases per-target concurrency."
            ),
        )
        command.add_argument(
            "--no-overlap-baselines-with-synthesis",
            dest="overlap_baselines_with_synthesis",
            action="store_false",
            help="Keep baseline evaluation and synthesis fully sequential within each target.",
        )
        command.set_defaults(overlap_baselines_with_synthesis=enabled_default)

    def add_baseline_parallelism_arguments(
        command: argparse.ArgumentParser,
        *,
        default: int,
    ) -> None:
        command.add_argument(
            "--baseline-workers",
            type=int,
            default=default,
            help=(
                "Number of baselines to evaluate concurrently inside one benchmark target. "
                "This does not change per-solver thread settings."
            ),
        )

    def add_dataset_arguments(
        command: argparse.ArgumentParser,
        *,
        problem_required: bool = True,
        family_required: bool = True,
    ) -> None:
        command.add_argument("--problem", choices=available_problem_names(), required=problem_required)
        command.add_argument("--family", required=family_required)
        command.add_argument("--dataset-id")
        command.add_argument("--output-dir")
        command.add_argument("--instance-param", action="append", default=[], help="KEY=VALUE")
        command.add_argument("--family-param", action="append", default=[], help="KEY=VALUE")
        command.add_argument("--train-size", type=int, default=256)
        command.add_argument("--validation-size", type=int, default=128)
        command.add_argument("--test-size", type=int, default=10_000)
        command.add_argument("--family-seed", type=int, default=17)
        command.add_argument("--train-seed", type=int, default=101)
        command.add_argument("--validation-seed", type=int, default=202)
        command.add_argument("--test-seed", type=int, default=303)
        command.add_argument("--compute-optima", dest="compute_optima", action="store_true", default=True)
        command.add_argument("--no-compute-optima", dest="compute_optima", action="store_false")

    generate_parser = subparsers.add_parser("generate", help="Generate a benchmark dataset.")
    add_dataset_arguments(generate_parser)
    generate_parser.set_defaults(func=cmd_generate)

    run_parser = subparsers.add_parser("run-agent", help="Run baseline evaluation and synthesis on a dataset.")
    run_parser.add_argument("--dataset-dir", required=True)
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--output-dir")
    run_parser.add_argument("--generator", choices=["auto", "template", "llm", "llm_no_hint"], default="auto")
    run_parser.add_argument("--mode", choices=["single", "beam"], default="beam")
    run_parser.add_argument("--iterations", type=int, default=3)
    run_parser.add_argument("--beam-width", type=int, default=3)
    run_parser.add_argument("--candidate-width", type=int)
    run_parser.add_argument("--skip-baselines", action="store_true")
    add_overlap_arguments(run_parser, enabled_default=True)
    add_baseline_parallelism_arguments(run_parser, default=1)
    add_gurobi_arguments(run_parser, enabled_default=True, time_limit_default=60.0, threads_default=1)
    add_native_exact_arguments(run_parser, time_limit_default=None)
    add_external_exact_arguments(run_parser, mode_default="auto", time_limit_default=60.0, threads_default=1)
    run_parser.set_defaults(func=cmd_run_agent)

    report_parser = subparsers.add_parser("report", help="Benchmark a completed run and write a report.")
    report_parser.add_argument("--dataset-dir", required=True)
    report_parser.add_argument("--agent-run-dir", required=True)
    report_parser.add_argument("--output-dir")
    report_parser.add_argument("--repeats", type=int, default=10)
    report_parser.add_argument("--include-train", action="store_true")
    add_gurobi_arguments(report_parser, enabled_default=None, time_limit_default=None, threads_default=None)
    add_native_exact_arguments(report_parser, time_limit_default=None)
    add_external_exact_arguments(report_parser, mode_default=None, time_limit_default=None, threads_default=None)
    report_parser.set_defaults(func=cmd_report)

    benchmark_parser = subparsers.add_parser("benchmark", help="Generate data, run synthesis, and write a report.")
    add_dataset_arguments(benchmark_parser, problem_required=False, family_required=False)
    benchmark_parser.add_argument("--all-families", action="store_true")
    benchmark_parser.add_argument("--max-parallel", type=int)
    benchmark_parser.add_argument("--dataset-dir")
    benchmark_parser.add_argument("--force-regenerate", action="store_true")
    benchmark_parser.add_argument("--run-id")
    benchmark_parser.add_argument("--run-output-dir")
    benchmark_parser.add_argument("--report-output-dir")
    benchmark_parser.add_argument("--generator", choices=["auto", "template", "llm", "llm_no_hint"], default="auto")
    benchmark_parser.add_argument("--mode", choices=["single", "beam"], default="beam")
    benchmark_parser.add_argument("--iterations", type=int, default=3)
    benchmark_parser.add_argument("--beam-width", type=int, default=3)
    benchmark_parser.add_argument("--candidate-width", type=int)
    benchmark_parser.add_argument("--skip-baselines", action="store_true")
    benchmark_parser.add_argument("--skip-report", action="store_true")
    add_overlap_arguments(benchmark_parser, enabled_default=True)
    add_baseline_parallelism_arguments(benchmark_parser, default=1)
    benchmark_parser.add_argument("--repeats", type=int, default=10)
    benchmark_parser.add_argument("--include-train", action="store_true")
    add_gurobi_arguments(benchmark_parser, enabled_default=True, time_limit_default=60.0, threads_default=1)
    add_native_exact_arguments(benchmark_parser, time_limit_default=None)
    add_external_exact_arguments(benchmark_parser, mode_default="auto", time_limit_default=60.0, threads_default=1)
    benchmark_parser.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_openai_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
