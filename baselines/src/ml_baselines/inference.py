from __future__ import annotations

import multiprocessing
import queue
import time
from typing import Any, Callable


class InferenceTimeoutError(TimeoutError):
    pass


class InferenceWorkerError(RuntimeError):
    pass


def _mp_context() -> multiprocessing.context.BaseContext:
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return multiprocessing.get_context()


def _worker(
    output_queue: multiprocessing.Queue,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    start = time.perf_counter()
    try:
        output_queue.put(
            {
                "status": "ok",
                "result": fn(*args, **kwargs),
                "runtime_ms": (time.perf_counter() - start) * 1000.0,
            }
        )
    except Exception as exc:
        output_queue.put(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "runtime_ms": (time.perf_counter() - start) * 1000.0,
            }
        )


def run_with_timeout(
    fn: Callable[..., Any],
    *args: Any,
    timeout_seconds: float | None,
    **kwargs: Any,
) -> Any:
    """Run inference in a child process and terminate it on timeout."""

    if timeout_seconds is None or float(timeout_seconds) <= 0:
        return fn(*args, **kwargs)
    context = _mp_context()
    output_queue: multiprocessing.Queue = context.Queue()
    process = context.Process(target=_worker, args=(output_queue, fn, args, kwargs))
    process.start()
    process.join(float(timeout_seconds))
    if process.is_alive():
        process.terminate()
        process.join()
        raise InferenceTimeoutError(f"Inference timed out after {float(timeout_seconds):.3f}s.")
    try:
        payload = output_queue.get_nowait()
    except queue.Empty as exc:
        raise InferenceWorkerError("Inference worker exited without returning a result.") from exc
    if payload.get("status") != "ok":
        raise InferenceWorkerError(str(payload.get("error", "inference worker failed")))
    return payload["result"]


def solve_with_timeout(
    solve_fn: Callable[[dict[str, Any], Any, Any], Any],
    instance: dict[str, Any],
    trained_state: Any,
    config: Any,
) -> Any:
    timeout = getattr(config, "timeout_seconds", None)
    if isinstance(config, dict):
        timeout = config.get("timeout_seconds")
    return run_with_timeout(
        solve_fn,
        instance,
        trained_state,
        config,
        timeout_seconds=None if timeout is None else float(timeout),
    )
