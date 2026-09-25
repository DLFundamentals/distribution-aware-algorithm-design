from __future__ import annotations

import fcntl
import os
import threading
import time
from dataclasses import asdict, dataclass

from dotenv import load_dotenv
from openai import OpenAI

# Optional client-side pacing for low/rolling-quota accounts that return HTTP 429
# `insufficient_quota` when called too fast (the quota refills over ~minutes). Set
# `DASBENCH_OPENAI_MIN_INTERVAL_SECONDS` to space calls out, and `DASBENCH_OPENAI_MAX_RETRIES`
# so the SDK backs off and retries a throttled call instead of failing it. Both default to off.
# When work is spread across multiple worker PROCESSES (e.g. a ProcessPoolExecutor sweep), set
# `DASBENCH_OPENAI_PACE_FILE` to a shared path so the min-interval is enforced GLOBALLY across
# processes (via an flock + a stored wall-clock timestamp), preventing a startup burst that would
# trip the rolling quota. Without it, pacing is only per-process.
_PACE_LOCK = threading.Lock()
_LAST_CALL_MONOTONIC = [0.0]


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"`{name}` must be an integer.") from exc


def _pace_via_file(pace_file: str, min_interval_seconds: float) -> None:
    """Block until >= min_interval has elapsed since the last call across ALL processes.

    Holds an exclusive flock on `pace_file` while reading/updating the stored wall-clock
    timestamp, so concurrent workers serialize their call-starts by at least min_interval.
    """
    fd = os.open(pace_file, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        raw = os.read(fd, 64).decode("utf-8", "replace").strip()
        try:
            last = float(raw)
        except ValueError:
            last = 0.0
        wait = last + min_interval_seconds - time.time()
        if wait > 0:
            time.sleep(wait)
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, f"{time.time():.6f}".encode("utf-8"))
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _install_call_pacing(client: OpenAI, min_interval_seconds: float) -> None:
    """Wrap the client's create methods so calls are spaced >= min_interval apart."""

    pace_file = os.getenv("DASBENCH_OPENAI_PACE_FILE")
    pace_file = pace_file.strip() if pace_file and pace_file.strip() else None

    def _paced(create_fn):
        def wrapper(*args, **kwargs):
            if pace_file:
                _pace_via_file(pace_file, min_interval_seconds)
            else:
                with _PACE_LOCK:
                    wait = _LAST_CALL_MONOTONIC[0] + min_interval_seconds - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                    _LAST_CALL_MONOTONIC[0] = time.monotonic()
            return create_fn(*args, **kwargs)

        return wrapper

    # dasbench issues requests via `.with_raw_response.create` (see chat_api.py), so pace those;
    # the plain `.create` paths are wrapped too as a harmless belt-and-suspenders.
    for attr_path in (
        "chat.completions.with_raw_response",
        "chat.completions",
        "responses.with_raw_response",
        "responses",
    ):
        target = client
        try:
            for part in attr_path.split("."):
                target = getattr(target, part)
            target.create = _paced(target.create)  # type: ignore[attr-defined]
        except AttributeError:
            continue

DEFAULT_MODEL = "gpt-5.2"
DEFAULT_REASONING_EFFORT = "xhigh"
OPENAI_TIMEOUT_SECONDS_ENV_VAR = "OPENAI_TIMEOUT_SECONDS"


@dataclass(frozen=True)
class OpenAIAPIConfig:
    api_key: str
    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    base_url: str | None = None
    organization: str | None = None
    project: str | None = None
    timeout_seconds: float | None = None

    def public_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("api_key", None)
        return payload


def load_openai_dotenv() -> bool:
    load_dotenv()
    return True


def _optional_positive_float_env(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(f"`{name}` must be a positive number.") from exc
    if parsed <= 0:
        raise RuntimeError(f"`{name}` must be a positive number.")
    return parsed


def load_openai_api_config(*, required: bool = True) -> OpenAIAPIConfig | None:
    load_openai_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
    base_url = os.getenv("OPENAI_BASE_URL")
    organization = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")
    project = os.getenv("OPENAI_PROJECT_ID")
    timeout_seconds = _optional_positive_float_env(OPENAI_TIMEOUT_SECONDS_ENV_VAR)
    if not api_key:
        if not required:
            return None
        raise RuntimeError(
            "Missing OpenAI API configuration. Set `OPENAI_API_KEY` in `.env` "
            "or the environment. Set `OPENAI_MODEL` to override the default model."
        )
    return OpenAIAPIConfig(
        api_key=api_key,
        model=model,
        reasoning_effort=reasoning_effort,
        base_url=base_url,
        organization=organization,
        project=project,
        timeout_seconds=timeout_seconds,
    )


def openai_api_is_configured() -> bool:
    return load_openai_api_config(required=False) is not None


def build_openai_client(config: OpenAIAPIConfig) -> OpenAI:
    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        organization=config.organization,
        project=config.project,
        max_retries=_int_env("DASBENCH_OPENAI_MAX_RETRIES", 2),
    )
    min_interval = _optional_positive_float_env("DASBENCH_OPENAI_MIN_INTERVAL_SECONDS")
    if min_interval:
        _install_call_pacing(client, min_interval)
    return client
