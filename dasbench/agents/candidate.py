from __future__ import annotations

import inspect
import json
import os
import signal
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

from dasbench.utils import candidate_manifest

DEFAULT_ANALYSIS_TIMEOUT_SECONDS = 120.0
ANALYSIS_TIMEOUT_ENV_VAR = "DASBENCH_ANALYSIS_TIMEOUT_SECONDS"


class AnalysisTimeoutError(TimeoutError):
    pass


def _load_module(path: Path, *, prefix: str) -> ModuleType:
    module_name = f"{prefix}_{uuid.uuid4().hex}"
    source = path.read_text(encoding="utf-8")
    module = ModuleType(module_name)
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def _resolved_analysis_timeout(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is None:
        raw_value = os.environ.get(ANALYSIS_TIMEOUT_ENV_VAR)
        if raw_value is None or not raw_value.strip():
            timeout_seconds = DEFAULT_ANALYSIS_TIMEOUT_SECONDS
        else:
            try:
                timeout_seconds = float(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"{ANALYSIS_TIMEOUT_ENV_VAR} must be a number of seconds, got {raw_value!r}."
                ) from exc
    if timeout_seconds <= 0:
        return None
    return float(timeout_seconds)


@contextmanager
def _analysis_timeout(timeout_seconds: float | None, *, analyze_path: Path):
    resolved = _resolved_analysis_timeout(timeout_seconds)
    if resolved is None or threading.current_thread() is not threading.main_thread():
        yield
        return

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)

    def _handle_timeout(signum, frame):
        raise AnalysisTimeoutError(f"{analyze_path} exceeded analysis timeout of {resolved:.1f} seconds.")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, resolved)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])


def _write_analysis_error(
    artifact_dir: Path | None,
    *,
    analyze_path: Path,
    exc: Exception,
    timeout_seconds: float | None,
) -> None:
    if artifact_dir is None:
        return
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analyze_path": str(analyze_path),
        "error": f"{type(exc).__name__}: {exc}",
        "timeout_seconds": _resolved_analysis_timeout(timeout_seconds),
    }
    (artifact_dir / "analysis_error.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_analysis(
    candidate_dir: Path,
    train_instances: list[dict[str, object]],
    *,
    manifest: dict[str, object] | None = None,
    artifact_dir: Path | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any] | None:
    analyze_path = candidate_dir / "analyze.py"
    if not analyze_path.exists():
        return None
    if not train_instances:
        analysis: dict[str, Any] = {
            "empty_train": True,
            "num_instances": 0,
        }
        if manifest is not None:
            analysis["manifest_problem"] = manifest.get("problem")
            analysis["manifest_family"] = manifest.get("family")
            analysis["manifest_instance_params"] = manifest.get("instance_params", {})
            analysis["manifest_family_params"] = manifest.get("family_params", {})
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "analysis_error.json").unlink(missing_ok=True)
            (artifact_dir / "analysis.json").write_text(
                json.dumps(analysis, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return analysis
    try:
        with _analysis_timeout(timeout_seconds, analyze_path=analyze_path):
            module = _load_module(analyze_path, prefix="candidate_analyze")
            if not hasattr(module, "analyze"):
                raise AttributeError(f"{analyze_path} must define analyze(train_instances, manifest=None).")
            analyze_fn = module.analyze
            parameters = inspect.signature(analyze_fn).parameters
            exposed_manifest = candidate_manifest(manifest) if manifest is not None else None
            if len(parameters) >= 2:
                analysis = analyze_fn(train_instances, exposed_manifest)
            else:
                analysis = analyze_fn(train_instances)
    except Exception as exc:
        _write_analysis_error(
            artifact_dir,
            analyze_path=analyze_path,
            exc=exc,
            timeout_seconds=timeout_seconds,
        )
        raise
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "analysis_error.json").unlink(missing_ok=True)
        (artifact_dir / "analysis.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return analysis


def build_solver(
    candidate_dir: Path,
    *,
    analysis: dict[str, Any] | None = None,
    manifest: dict[str, object] | None = None,
):
    solution_path = candidate_dir / "solution.py"
    if not solution_path.exists():
        raise FileNotFoundError(f"Missing solver file: {solution_path}")
    module = _load_module(solution_path, prefix="candidate_solution")
    exposed_manifest = candidate_manifest(manifest) if manifest is not None else None
    if hasattr(module, "build_solver"):
        build_solver_fn = module.build_solver
        parameters = inspect.signature(build_solver_fn).parameters
        if len(parameters) >= 2:
            return build_solver_fn(analysis, exposed_manifest)
        if len(parameters) == 1:
            return build_solver_fn(analysis)
        return build_solver_fn()
    if not hasattr(module, "solve"):
        raise AttributeError(f"{solution_path} must define solve(instance, analysis=None, manifest=None).")
    solve_fn = module.solve
    parameters = inspect.signature(solve_fn).parameters
    if len(parameters) >= 3:
        return lambda instance: solve_fn(instance, analysis, exposed_manifest)
    if len(parameters) >= 2:
        return lambda instance: solve_fn(instance, analysis)
    return lambda instance: solve_fn(instance)
