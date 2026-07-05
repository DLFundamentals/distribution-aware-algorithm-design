from __future__ import annotations

import fcntl
import hashlib
import os
import sys
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from dasbench.integrations.openai_api import (
    OpenAIAPIConfig,
    build_openai_client,
    load_openai_api_config,
)

DEFAULT_PROVIDER = "openai"
CUSTOM_PROVIDER = "custom_chat"

PROVIDER_ENV_VAR = "LLM_PROVIDER"
CUSTOM_API_KEY_ENV_VAR = "CUSTOM_CHAT_API_KEY"
CUSTOM_API_BASE_URL_ENV_VAR = "CUSTOM_CHAT_API_BASE_URL"
CUSTOM_MODEL_ENV_VAR = "CUSTOM_CHAT_MODEL"
CUSTOM_REASONING_EFFORT_ENV_VAR = "CUSTOM_CHAT_REASONING_EFFORT"
CUSTOM_TIMEOUT_SECONDS_ENV_VAR = "CUSTOM_CHAT_TIMEOUT_SECONDS"
CUSTOM_MAX_TOKENS_ENV_VAR = "CUSTOM_CHAT_MAX_TOKENS"
LLM_WAIT_FOR_IDLE_ENV_VAR = "DASBENCH_LLM_WAIT_FOR_IDLE"
LLM_IDLE_POLL_SECONDS_ENV_VAR = "DASBENCH_LLM_IDLE_POLL_SECONDS"
LLM_IDLE_MAX_RUNNING_REQUESTS_ENV_VAR = "DASBENCH_LLM_IDLE_MAX_RUNNING_REQUESTS"
LLM_IDLE_MAX_WAITING_REQUESTS_ENV_VAR = "DASBENCH_LLM_IDLE_MAX_WAITING_REQUESTS"
LLM_IDLE_METRICS_URL_ENV_VAR = "DASBENCH_LLM_IDLE_METRICS_URL"
LLM_IDLE_LOCK_PATH_ENV_VAR = "DASBENCH_LLM_IDLE_LOCK_PATH"

DEFAULT_CUSTOM_TIMEOUT_SECONDS = 14_400.0
DEFAULT_LLM_IDLE_POLL_SECONDS = 5.0
DEFAULT_LLM_IDLE_MAX_RUNNING_REQUESTS = 0
DEFAULT_LLM_IDLE_MAX_WAITING_REQUESTS = 0
LLM_IDLE_METRICS_TIMEOUT_SECONDS = 2.0
CUSTOM_524_MAX_RETRIES = 3


@dataclass(frozen=True)
class CustomChatAPIConfig:
    api_key: str
    base_url: str
    model: str
    reasoning_effort: str | None = None
    timeout_seconds: float = DEFAULT_CUSTOM_TIMEOUT_SECONDS
    max_tokens: int | None = None
    wait_for_idle: bool = False
    idle_poll_seconds: float = DEFAULT_LLM_IDLE_POLL_SECONDS
    idle_max_running_requests: int = DEFAULT_LLM_IDLE_MAX_RUNNING_REQUESTS
    idle_max_waiting_requests: int = DEFAULT_LLM_IDLE_MAX_WAITING_REQUESTS
    idle_metrics_url: str | None = None
    idle_lock_path: str | None = None
    provider: str = CUSTOM_PROVIDER

    def public_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("api_key", None)
        return payload


ChatAPIConfig = OpenAIAPIConfig | CustomChatAPIConfig


def _env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_bool(name: str, *, default: bool = False) -> bool:
    value = _env_value(name)
    if value is None:
        return default
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"`{name}` must be a boolean value like 1/0, true/false, or yes/no.")


def _provider_name() -> str:
    return (_env_value(PROVIDER_ENV_VAR) or DEFAULT_PROVIDER).lower()


def _env_positive_float(name: str, *, default: float) -> float:
    value = _env_value(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(f"`{name}` must be a positive number.") from exc
    if parsed <= 0:
        raise RuntimeError(f"`{name}` must be a positive number.")
    return parsed


def _env_nonnegative_int(name: str, *, default: int) -> int:
    value = _env_value(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"`{name}` must be a nonnegative integer.") from exc
    if parsed < 0:
        raise RuntimeError(f"`{name}` must be a nonnegative integer.")
    return parsed


def _env_optional_positive_int(name: str) -> int | None:
    value = _env_value(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"`{name}` must be a positive integer.") from exc
    if parsed <= 0:
        raise RuntimeError(f"`{name}` must be a positive integer.")
    return parsed


def load_custom_chat_api_config(*, required: bool = True) -> CustomChatAPIConfig | None:
    load_dotenv()
    values = {
        CUSTOM_API_KEY_ENV_VAR: _env_value(CUSTOM_API_KEY_ENV_VAR),
        CUSTOM_API_BASE_URL_ENV_VAR: _env_value(CUSTOM_API_BASE_URL_ENV_VAR),
        CUSTOM_MODEL_ENV_VAR: _env_value(CUSTOM_MODEL_ENV_VAR),
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        if not required:
            return None
        raise RuntimeError(
            "Missing custom chat API configuration. Set "
            f"{', '.join(missing)} in `.env` or the environment."
        )
    return CustomChatAPIConfig(
        api_key=str(values[CUSTOM_API_KEY_ENV_VAR]),
        base_url=str(values[CUSTOM_API_BASE_URL_ENV_VAR]),
        model=str(values[CUSTOM_MODEL_ENV_VAR]),
        reasoning_effort=_env_value(CUSTOM_REASONING_EFFORT_ENV_VAR),
        timeout_seconds=_env_positive_float(
            CUSTOM_TIMEOUT_SECONDS_ENV_VAR,
            default=DEFAULT_CUSTOM_TIMEOUT_SECONDS,
        ),
        max_tokens=_env_optional_positive_int(CUSTOM_MAX_TOKENS_ENV_VAR),
        wait_for_idle=_env_bool(LLM_WAIT_FOR_IDLE_ENV_VAR),
        idle_poll_seconds=_env_positive_float(
            LLM_IDLE_POLL_SECONDS_ENV_VAR,
            default=DEFAULT_LLM_IDLE_POLL_SECONDS,
        ),
        idle_max_running_requests=_env_nonnegative_int(
            LLM_IDLE_MAX_RUNNING_REQUESTS_ENV_VAR,
            default=DEFAULT_LLM_IDLE_MAX_RUNNING_REQUESTS,
        ),
        idle_max_waiting_requests=_env_nonnegative_int(
            LLM_IDLE_MAX_WAITING_REQUESTS_ENV_VAR,
            default=DEFAULT_LLM_IDLE_MAX_WAITING_REQUESTS,
        ),
        idle_metrics_url=_env_value(LLM_IDLE_METRICS_URL_ENV_VAR),
        idle_lock_path=_env_value(LLM_IDLE_LOCK_PATH_ENV_VAR),
    )


def load_chat_api_config(*, required: bool = True) -> ChatAPIConfig | None:
    load_dotenv()
    provider = _provider_name()
    if provider == DEFAULT_PROVIDER:
        return load_openai_api_config(required=required)
    if provider == CUSTOM_PROVIDER:
        return load_custom_chat_api_config(required=required)
    if not required:
        return None
    raise RuntimeError(
        f"Unsupported {PROVIDER_ENV_VAR} `{provider}`. "
        f"Expected `{DEFAULT_PROVIDER}` or `{CUSTOM_PROVIDER}`."
    )


def build_chat_client(config: ChatAPIConfig) -> OpenAI:
    if isinstance(config, OpenAIAPIConfig):
        return build_openai_client(config)
    return OpenAI(api_key=config.api_key, base_url=config.base_url, max_retries=0)


def chat_api_is_configured() -> bool:
    return load_chat_api_config(required=False) is not None


def chat_config_with_overrides(
    config: ChatAPIConfig,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> ChatAPIConfig:
    resolved_model = model or config.model
    resolved_reasoning_effort = reasoning_effort if reasoning_effort is not None else config.reasoning_effort
    if isinstance(config, OpenAIAPIConfig):
        return OpenAIAPIConfig(
            api_key=config.api_key,
            model=resolved_model,
            reasoning_effort=resolved_reasoning_effort,
            base_url=config.base_url,
            organization=config.organization,
            project=config.project,
            timeout_seconds=config.timeout_seconds,
        )
    return CustomChatAPIConfig(
        api_key=config.api_key,
        base_url=config.base_url,
        model=resolved_model,
        reasoning_effort=resolved_reasoning_effort,
        timeout_seconds=config.timeout_seconds,
        max_tokens=config.max_tokens,
        wait_for_idle=config.wait_for_idle,
        idle_poll_seconds=config.idle_poll_seconds,
        idle_max_running_requests=config.idle_max_running_requests,
        idle_max_waiting_requests=config.idle_max_waiting_requests,
        idle_metrics_url=config.idle_metrics_url,
        idle_lock_path=config.idle_lock_path,
        provider=config.provider,
    )


def _status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status_code = getattr(response, "status_code", None)
    if isinstance(response_status_code, int):
        return response_status_code
    return None


def _create_custom_chat_completion_raw(
    client: OpenAI,
    *,
    config: CustomChatAPIConfig,
    request: dict[str, Any],
    timeout: float,
):
    with _custom_chat_idle_gate(config):
        retries = 0
        while True:
            try:
                return client.chat.completions.with_raw_response.create(**request, timeout=timeout)
            except Exception as exc:
                if _status_code(exc) != 524 or retries >= CUSTOM_524_MAX_RETRIES:
                    raise
                retries += 1


def _metrics_url_for_config(config: CustomChatAPIConfig) -> str:
    if config.idle_metrics_url is not None:
        return config.idle_metrics_url
    parsed = urllib.parse.urlparse(config.base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    metrics_path = f"{path}/metrics" if path else "/metrics"
    return urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, metrics_path, "", "", "")
    )


def _default_idle_lock_path(config: CustomChatAPIConfig) -> str:
    parsed = urllib.parse.urlparse(config.base_url)
    identity = parsed.netloc or config.base_url
    digest = hashlib.sha1(config.base_url.encode("utf-8")).hexdigest()[:10]
    safe_identity = "".join(ch if ch.isalnum() else "_" for ch in identity).strip("_") or "custom_chat"
    return f"/tmp/dasbench-llm-gate-{safe_identity}-{digest}.lock"


def _read_vllm_queue_metrics(metrics_url: str) -> dict[str, float]:
    with urllib.request.urlopen(metrics_url, timeout=LLM_IDLE_METRICS_TIMEOUT_SECONDS) as response:
        text = response.read().decode("utf-8", "replace")
    values = {
        "running": 0.0,
        "waiting": 0.0,
    }
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        metric_name = line.split("{", 1)[0].split(None, 1)[0]
        if metric_name not in {"vllm:num_requests_running", "vllm:num_requests_waiting"}:
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue
        if metric_name == "vllm:num_requests_running":
            values["running"] += value
        else:
            values["waiting"] += value
    return values


def _idle_status_message(prefix: str, *, elapsed_seconds: float, running: float | None, waiting: float | None) -> str:
    parts = [f"DASBench LLM gate: {prefix}", f"elapsed={elapsed_seconds:.0f}s"]
    if running is not None:
        parts.append(f"running={running:g}")
    if waiting is not None:
        parts.append(f"waiting={waiting:g}")
    return " ".join(parts)


def _emit_idle_status(message: str, *, last_emit: float) -> float:
    now = time.monotonic()
    if sys.stderr.isatty():
        print(f"\r{message[:180]:<180}", end="", file=sys.stderr, flush=True)
        return now
    if now - last_emit >= 60.0:
        print(message, file=sys.stderr, flush=True)
        return now
    return last_emit


def _clear_idle_status() -> None:
    if sys.stderr.isatty():
        print(f"\r{'':<180}\r", end="", file=sys.stderr, flush=True)


@contextmanager
def _custom_chat_idle_gate(config: CustomChatAPIConfig):
    if not config.wait_for_idle:
        yield
        return

    lock_path = Path(config.idle_lock_path or _default_idle_lock_path(config))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_url = _metrics_url_for_config(config)
    start = time.monotonic()
    last_emit = 0.0
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        while True:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                last_emit = _emit_idle_status(
                    _idle_status_message(
                        "waiting for another local generation request",
                        elapsed_seconds=time.monotonic() - start,
                        running=None,
                        waiting=None,
                    ),
                    last_emit=last_emit,
                )
                time.sleep(config.idle_poll_seconds)

        while True:
            try:
                metrics = _read_vllm_queue_metrics(metrics_url)
            except Exception as exc:
                last_emit = _emit_idle_status(
                    _idle_status_message(
                        f"metrics unavailable, proceeding under lock ({type(exc).__name__})",
                        elapsed_seconds=time.monotonic() - start,
                        running=None,
                        waiting=None,
                    ),
                    last_emit=last_emit,
                )
                break
            running = metrics["running"]
            waiting = metrics["waiting"]
            if running <= config.idle_max_running_requests and waiting <= config.idle_max_waiting_requests:
                break
            last_emit = _emit_idle_status(
                _idle_status_message(
                    "waiting for local vLLM idle",
                    elapsed_seconds=time.monotonic() - start,
                    running=running,
                    waiting=waiting,
                ),
                last_emit=last_emit,
            )
            time.sleep(config.idle_poll_seconds)
        _clear_idle_status()
        yield
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()
            _clear_idle_status()


def create_chat_completion_raw(
    config: ChatAPIConfig,
    *,
    messages: list[dict[str, str]],
    response_format: dict[str, Any] | None = None,
    timeout: float | None = None,
):
    client = build_chat_client(config)
    request: dict[str, Any] = {
        "messages": messages,
        "model": config.model,
    }
    if response_format is not None:
        request["response_format"] = response_format
    if isinstance(config, CustomChatAPIConfig):
        request["stream"] = False
        if config.reasoning_effort is not None:
            request["reasoning_effort"] = config.reasoning_effort
        if config.max_tokens is not None:
            request["max_tokens"] = config.max_tokens
        return _create_custom_chat_completion_raw(
            client,
            config=config,
            request=request,
            timeout=config.timeout_seconds if timeout is None else timeout,
        )
    else:
        request["reasoning_effort"] = config.reasoning_effort
    resolved_timeout = config.timeout_seconds if timeout is None else timeout
    if resolved_timeout is None:
        return client.chat.completions.with_raw_response.create(**request)
    return client.chat.completions.with_raw_response.create(**request, timeout=resolved_timeout)


def chat_completion_text(completion: object) -> str:
    choices = getattr(completion, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    chunks: list[str] = []
    for item in content or []:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            chunks.append(text)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            chunks.append(str(item["text"]))
    return "\n".join(chunks).strip()
