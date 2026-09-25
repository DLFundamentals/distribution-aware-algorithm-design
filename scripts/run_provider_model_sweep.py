from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import dotenv_values

from dasbench.utils import write_json

DEFAULT_OUTPUT_ROOT = Path("artifacts/model_sweeps")
DEFAULT_PROVIDERS = {
    "kimi-k26": Path(".env.kimi-k26"),
    "deepseek-v4-pro": Path(".env.deepseek-v4-pro"),
    "glm-52": Path(".env.glm-52"),
}
MAIN_CONDITION_ID = "seconds_scale_v2"
PACE_COMPETITIONS = (
    "pace2025_hs",
    "pace2024_ocm_exact",
    "pace2024_ocm_cutwidth",
    "pace2022_dfvs_heuristic",
)


@dataclass(frozen=True)
class Provider:
    name: str
    env_file: Path


@dataclass(frozen=True)
class Stage:
    provider: Provider
    name: str
    command: list[str]
    completion_path: Path
    log_path: Path
    pid_path: Path
    status_path: Path
    cwd: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _python_module_command(module: str, *args: str | Path) -> list[str]:
    return ["uv", "run", "python", "-m", module, *(str(arg) for arg in args)]


def _provider_root(output_root: Path, provider: str) -> Path:
    return output_root / provider


def _provider_log_dir(output_root: Path, provider: str) -> Path:
    return _provider_root(output_root, provider) / "runner_logs"


def _provider_status_dir(output_root: Path, provider: str) -> Path:
    return _provider_root(output_root, provider) / "runner_status"


def _stage_paths(output_root: Path, provider: Provider, stage_name: str) -> tuple[Path, Path]:
    log_dir = _provider_log_dir(output_root, provider.name)
    status_dir = _provider_status_dir(output_root, provider.name)
    return log_dir / f"{stage_name}.log", status_dir / f"{stage_name}.json"


def _main_agent_stage(provider: Provider, output_root: Path, *, max_workers: int, force: bool) -> Stage:
    sweep_id = f"{provider.name}_main21_agent"
    provider_root = _provider_root(output_root, provider.name)
    command = _python_module_command(
        "benchmarks.main_paper_benchmark",
        "--sweep-id",
        sweep_id,
        "--output-root",
        provider_root,
        "--max-workers",
        str(max_workers),
    )
    if force:
        command.append("--force")
    completion_path = provider_root / "second_scale_benchmark_v2" / sweep_id / "benchmark_sweep_summary.json"
    return _stage(provider, output_root, "main21_agent", command, completion_path)


def _main_llm_pv_stage(provider: Provider, output_root: Path, *, attempts: int, max_workers: int, force: bool) -> Stage:
    sweep_id = f"{provider.name}_main21_llm_pv"
    source_sweep_id = f"{provider.name}_main21_agent"
    provider_root = _provider_root(output_root, provider.name)
    source_run_root = provider_root / "second_scale_benchmark_v2" / source_sweep_id
    command = _python_module_command(
        "benchmarks.llm_pv_benchmark",
        "--sweep-id",
        sweep_id,
        "--output-root",
        provider_root,
        "--source-run-root",
        source_run_root,
        "--source-condition-id",
        MAIN_CONDITION_ID,
        "--attempts",
        str(attempts),
        "--max-workers",
        str(max_workers),
        "--all-families",
    )
    if force:
        command.append("--force")
    completion_path = provider_root / "llm_pv_benchmark" / sweep_id / "benchmark_sweep_summary.json"
    return _stage(provider, output_root, "main21_llm_pv", command, completion_path)


def _pace_mds_agent_stage(provider: Provider, output_root: Path, *, force: bool) -> Stage:
    del force
    provider_root = _provider_root(output_root, provider.name)
    experiment_id = "pace2025_mds_heuristic_agent"
    command = _python_module_command(
        "benchmarks.pace",
        "--competition",
        "pace2025_ds_heuristic",
        "--test-source",
        "private",
        "--train-count",
        "5",
        "--validation-count",
        "5",
        "--test-count",
        "100",
        "--generator",
        "llm",
        "--mode",
        "beam",
        "--iterations",
        "2",
        "--beam-width",
        "3",
        "--candidate-width",
        "3",
        "--output-root",
        provider_root / "pace2025_dominating_set",
        "--experiment-id",
        experiment_id,
    )
    completion_path = (
        provider_root
        / "pace2025_dominating_set"
        / experiment_id
        / "pace_evaluation"
        / "pace_evaluation_summary.json"
    )
    return _stage(provider, output_root, "pace2025_mds_agent", command, completion_path)


def _pace_competition_agent_stage(
    provider: Provider,
    output_root: Path,
    competition: str,
    *,
    force: bool,
) -> Stage:
    del force
    provider_root = _provider_root(output_root, provider.name)
    experiment_id = f"{competition}_agent"
    command = _python_module_command(
        "benchmarks.pace",
        "--competition",
        competition,
        "--generator",
        "llm",
        "--mode",
        "beam",
        "--iterations",
        "2",
        "--beam-width",
        "3",
        "--candidate-width",
        "3",
        "--train-count",
        "5",
        "--validation-count",
        "5",
        "--test-count",
        "100",
        "--test-source",
        "private",
        "--output-root",
        provider_root / "pace_competitions",
        "--experiment-id",
        experiment_id,
    )
    completion_path = (
        provider_root / "pace_competitions" / experiment_id / "pace_evaluation" / "pace_evaluation_summary.json"
    )
    return _stage(provider, output_root, f"{competition}_agent", command, completion_path)


def _pace_mds_llm_pv_stage(provider: Provider, output_root: Path, *, attempts: int, force: bool) -> Stage:
    provider_root = _provider_root(output_root, provider.name)
    pace_run_dir = provider_root / "pace2025_dominating_set" / "pace2025_mds_heuristic_agent"
    output_dir = provider_root / "pace2025_dominating_set" / "pace2025_mds_heuristic_llm_pv"
    command = _python_module_command(
        "scripts.pace2025_run_llm_pv_baseline",
        "--pace-run-dir",
        pace_run_dir,
        "--output-dir",
        output_dir,
        "--run-artifact-root",
        output_dir / "llm_pv_artifacts",
        "--pace-evaluation-dir",
        output_dir / "pace_evaluation",
        "--sweep-id",
        f"{provider.name}_pace2025_mds_llm_pv",
        "--attempts",
        str(attempts),
    )
    if force:
        command.append("--force")
    completion_path = output_dir / "llm_pv_baseline_summary.json"
    return _stage(provider, output_root, "pace2025_mds_llm_pv", command, completion_path)


def _pace_imports_llm_pv_stage(provider: Provider, output_root: Path, *, attempts: int, force: bool) -> Stage:
    provider_root = _provider_root(output_root, provider.name)
    sweep_id = f"{provider.name}_pace_imports_llm_pv"
    output_dir = provider_root / "pace_competitions" / "llm_pv"
    command = _python_module_command(
        "benchmarks.pace_llm_pv_baselines",
        "--sweep-id",
        sweep_id,
        "--output-root",
        output_dir,
        "--attempts",
        str(attempts),
    )
    for competition in PACE_COMPETITIONS:
        command.extend(
            [
                "--dataset",
                f"{competition}={provider_root / 'pace_competitions' / f'{competition}_agent' / 'dataset'}",
            ]
        )
    if force:
        command.append("--force")
    completion_path = output_dir / sweep_id / "pace_llm_pv_aggregate.json"
    return _stage(provider, output_root, "pace_imports_llm_pv", command, completion_path)


def _stage(provider: Provider, output_root: Path, stage_name: str, command: list[str], completion_path: Path) -> Stage:
    log_path, status_path = _stage_paths(output_root, provider, stage_name)
    return Stage(
        provider=provider,
        name=stage_name,
        command=command,
        completion_path=completion_path,
        log_path=log_path,
        pid_path=status_path.with_suffix(".pid"),
        status_path=status_path,
        cwd=repo_root(),
    )


def build_stages(
    provider: Provider,
    output_root: Path,
    *,
    main_max_workers: int,
    llm_pv_max_workers: int,
    llm_pv_attempts: int,
    force: bool,
) -> list[Stage]:
    stages = [
        _main_agent_stage(provider, output_root, max_workers=main_max_workers, force=force),
        _main_llm_pv_stage(
            provider,
            output_root,
            attempts=llm_pv_attempts,
            max_workers=llm_pv_max_workers,
            force=force,
        ),
        _pace_mds_agent_stage(provider, output_root, force=force),
    ]
    stages.extend(
        _pace_competition_agent_stage(provider, output_root, competition, force=force)
        for competition in PACE_COMPETITIONS
    )
    stages.extend(
        [
            _pace_mds_llm_pv_stage(provider, output_root, attempts=llm_pv_attempts, force=force),
            _pace_imports_llm_pv_stage(provider, output_root, attempts=llm_pv_attempts, force=force),
        ]
    )
    return stages


def _status_payload(stage: Stage, status: str, **extra: object) -> dict[str, object]:
    payload = {
        "provider": stage.provider.name,
        "stage": stage.name,
        "status": status,
        "env_file": str(stage.provider.env_file),
        "command": stage.command,
        "command_text": shlex.join(stage.command),
        "completion_path": str(stage.completion_path),
        "log_path": str(stage.log_path),
        "pid_path": str(stage.pid_path),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    payload.update(extra)
    return payload


def _write_status(stage: Stage, status: str, **extra: object) -> None:
    stage.status_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(stage.status_path, _status_payload(stage, status, **extra))


def _load_provider_env(provider: Provider) -> dict[str, str]:
    env_file = provider.env_file
    if not env_file.exists():
        raise FileNotFoundError(f"Missing env file for provider `{provider.name}`: {env_file}")
    parsed = dotenv_values(env_file)
    env = os.environ.copy()
    for key, value in parsed.items():
        if value is not None:
            env[str(key)] = str(value)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_stage(stage: Stage, *, force: bool, dry_run: bool) -> dict[str, object]:
    if stage.completion_path.exists() and not force:
        _write_status(stage, "skipped", reason="completion_path_exists")
        return _status_payload(stage, "skipped", reason="completion_path_exists")
    if dry_run:
        _write_status(stage, "dry_run")
        return _status_payload(stage, "dry_run")

    env = _load_provider_env(stage.provider)
    stage.log_path.parent.mkdir(parents=True, exist_ok=True)
    stage.pid_path.parent.mkdir(parents=True, exist_ok=True)
    stage.pid_path.unlink(missing_ok=True)
    started = time.perf_counter()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_status(stage, "starting", started_at=started_at)
    with stage.log_path.open("ab") as log:
        log.write(f"\n\n===== {stage.provider.name}:{stage.name} started {started_at} =====\n".encode("utf-8"))
        log.write((shlex.join(stage.command) + "\n").encode("utf-8"))
        log.flush()
        process = subprocess.Popen(
            stage.command,
            cwd=stage.cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stage.pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
        _write_status(stage, "running", started_at=started_at, pid=process.pid)
        returncode = process.wait()
        finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        runtime_ms = (time.perf_counter() - started) * 1000.0
        log.write(
            f"===== {stage.provider.name}:{stage.name} finished {finished_at} returncode={returncode} =====\n".encode(
                "utf-8"
            )
        )
    stage.pid_path.unlink(missing_ok=True)
    status = "completed" if returncode == 0 and stage.completion_path.exists() else "failed"
    payload = {
        "pid": process.pid,
        "returncode": returncode,
        "started_at": started_at,
        "finished_at": finished_at,
        "runtime_ms": runtime_ms,
        "completion_path_exists": stage.completion_path.exists(),
    }
    _write_status(stage, status, **payload)
    return _status_payload(stage, status, **payload)


def _selected_providers(args: argparse.Namespace) -> list[Provider]:
    overrides = dict(item.split("=", 1) for item in args.env_file if "=" in item)
    names = args.provider or list(DEFAULT_PROVIDERS)
    providers: list[Provider] = []
    for name in names:
        default_env = DEFAULT_PROVIDERS.get(name)
        if default_env is None and name not in overrides:
            raise ValueError(f"Unknown provider `{name}`. Pass --env-file {name}=PATH to configure it.")
        providers.append(Provider(name=name, env_file=Path(overrides.get(name, default_env or ""))))
    return providers


def _filter_stages(stages: Iterable[Stage], requested: list[str]) -> list[Stage]:
    stage_list = list(stages)
    if not requested:
        return stage_list
    requested_set = set(requested)
    unknown = requested_set - {stage.name for stage in stage_list}
    if unknown:
        raise ValueError(f"Unknown stage(s): {sorted(unknown)}")
    return [stage for stage in stage_list if stage.name in requested_set]


def _write_manifest(output_root: Path, results: list[dict[str, object]], *, status: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "provider_sweep_manifest.json"
    write_json(
        path,
        {
            "schema_version": "provider_model_sweep.v1",
            "status": status,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": results,
        },
    )
    return path


def run(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    providers = _selected_providers(args)
    results: list[dict[str, object]] = []
    failed = False
    for provider in providers:
        stages = build_stages(
            provider,
            output_root,
            main_max_workers=max(1, int(args.main_max_workers)),
            llm_pv_max_workers=max(1, int(args.llm_pv_max_workers)),
            llm_pv_attempts=max(1, int(args.llm_pv_attempts)),
            force=bool(args.force),
        )
        for stage in _filter_stages(stages, args.stage):
            print(json.dumps({"provider": provider.name, "stage": stage.name, "command": shlex.join(stage.command)}))
            result = _run_stage(stage, force=bool(args.force), dry_run=bool(args.dry_run))
            results.append(result)
            _write_manifest(output_root, results, status="running" if not args.dry_run else "dry_run")
            if result["status"] == "failed":
                failed = True
                if not args.keep_going:
                    _write_manifest(output_root, results, status="failed")
                    return 1
    final_status = "failed" if failed else ("dry_run" if args.dry_run else "completed")
    manifest_path = _write_manifest(output_root, results, status=final_status)
    print(f"Provider sweep manifest: {manifest_path}")
    return 1 if failed else 0


def _detach(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "provider_sweep_runner.log"
    pid_path = output_root / "provider_sweep_runner.pid"
    command = [sys.executable, __file__]
    for raw in sys.argv[1:]:
        if raw != "--detach":
            command.append(raw)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=repo_root(),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    print(f"Detached provider sweep runner PID: {process.pid}")
    print(f"Runner log: {log_path}")
    print(f"Runner PID file: {pid_path}")
    return 0


def _stop(output_root: Path) -> int:
    pid_path = output_root / "provider_sweep_runner.pid"
    if not pid_path.exists():
        raise SystemExit(f"No runner PID file found: {pid_path}")
    pid = int(pid_path.read_text(encoding="utf-8").strip())
    os.killpg(pid, signal.SIGTERM)
    print(f"Sent SIGTERM to provider sweep runner process group {pid}.")
    return 0


def _init_env_templates(paths: dict[str, Path]) -> None:
    template = "\n".join(
        [
            "LLM_PROVIDER=custom_chat",
            "CUSTOM_CHAT_API_BASE_URL=",
            "CUSTOM_CHAT_API_KEY=",
            "CUSTOM_CHAT_MODEL=",
            "CUSTOM_CHAT_TIMEOUT_SECONDS=14400",
            "DASBENCH_SOLVER_TIMEOUT_SECONDS=360",
            "DASBENCH_SOLVER_MEMORY_LIMIT_MB=65536",
            "",
        ]
    )
    for provider, path in paths.items():
        if path.exists():
            print(f"exists: {path}")
            continue
        path.write_text(template, encoding="utf-8")
        print(f"created: {path} for {provider}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run resumable provider sweeps for DasBench and PACE experiments.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--provider", action="append", choices=sorted(DEFAULT_PROVIDERS), default=[])
    parser.add_argument("--env-file", action="append", default=[], help="Override provider env path as PROVIDER=PATH.")
    parser.add_argument("--stage", action="append", default=[], help="Run only a named stage; may be repeated.")
    parser.add_argument("--main-max-workers", type=int, default=1)
    parser.add_argument("--llm-pv-max-workers", type=int, default=1)
    parser.add_argument("--llm-pv-attempts", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="Continue to later stages after a stage failure.")
    parser.add_argument("--detach", action="store_true", help="Launch this runner in the background and return.")
    parser.add_argument("--stop", action="store_true", help="Send SIGTERM to the detached runner process group.")
    parser.add_argument("--init-env-templates", action="store_true", help="Create provider env-file templates if missing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.init_env_templates:
        _init_env_templates(DEFAULT_PROVIDERS)
        return 0
    if args.stop:
        return _stop(Path(args.output_root))
    if args.detach:
        return _detach(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
