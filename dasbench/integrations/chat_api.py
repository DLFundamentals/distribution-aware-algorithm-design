from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import os
import re
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

from dasbench.integrations.anthropic_api import (
    ANTHROPIC_PROVIDER,
    AnthropicAPIConfig,
    anthropic_message_text,
    build_anthropic_client,
    create_anthropic_message_raw,
    load_anthropic_api_config,
)
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
CUSTOM_KEY_STATE_FILE_ENV_VAR = "DASBENCH_CHAT_KEY_STATE_FILE"
CUSTOM_KEY_WAIT_FOR_RESET_ENV_VAR = "DASBENCH_CHAT_KEY_WAIT_FOR_RESET"
CUSTOM_KEY_MAX_RESET_WAITS_ENV_VAR = "DASBENCH_CHAT_KEY_MAX_RESET_WAITS"
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
# Status codes that mean "this key is spent", not "this request was bad".
CUSTOM_QUOTA_STATUS_CODES = frozenset({402, 429})
CUSTOM_QUOTA_MESSAGE_MARKERS = (
    "insufficient_quota",
    "insufficient quota",
    "exceeded your current quota",
    "quota exceeded",
    "daily limit",
    "usage limit",
    "billing",
)


@dataclass(frozen=True)
class CustomChatAPIConfig:
    api_key: str
    base_url: str
    model: str
    api_keys: tuple[str, ...] = ()
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
        # Runs write this straight into candidate metadata, so the pooled keys
        # must not survive here either -- only how many were available.
        keys = payload.pop("api_keys", ()) or ()
        payload["api_key_count"] = len(keys)
        return payload


ChatAPIConfig = OpenAIAPIConfig | CustomChatAPIConfig | AnthropicAPIConfig


def _env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _split_api_keys(raw: str | None) -> tuple[str, ...]:
    """Parse one or several API keys out of a single env var.

    The gateway meters a fixed budget per key per day, so a lab can pool keys
    and let a long run roll onto the next one when the active key is spent.
    A single key parses unchanged; several are separated by commas or
    whitespace. Order is preserved and duplicates dropped, so rotation order is
    exactly the order the `.env` lists.
    """
    if raw is None:
        return ()
    seen: set[str] = set()
    keys: list[str] = []
    for chunk in re.split(r"[,\s]+", raw):
        key = chunk.strip().strip("\"'")
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return tuple(keys)


def _is_quota_exhausted(exc: Exception) -> bool:
    """True when the failure means this key is spent rather than the call is bad."""
    if _status_code(exc) in CUSTOM_QUOTA_STATUS_CODES:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in CUSTOM_QUOTA_MESSAGE_MARKERS)


def _spent_keys_path() -> Path | None:
    """Shared file recording which pooled keys are already spent.

    A sweep runs many worker processes; without shared state each would
    rediscover an exhausted key on its own and burn a failed call doing it.
    Unset, rotation is per-process, which still works but repeats that probe.
    """
    configured = _env_value(CUSTOM_KEY_STATE_FILE_ENV_VAR)
    return Path(configured) if configured else None


def _key_fingerprint(key: str) -> str:
    """Short stable id for a key, so state and logs never carry the key itself."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _budget_day() -> str:
    """The current budget day. Per-key daily budgets reset at 00:00 UTC."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def _seconds_until_budget_reset() -> float:
    """Seconds until the next 00:00 UTC, plus a small margin for clock skew."""
    now = dt.datetime.now(dt.timezone.utc)
    reset = (now + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max((reset - now).total_seconds(), 0.0) + 60.0


def _read_spent_keys() -> set[str]:
    """Fingerprints of keys already known spent *today*.

    The file's first line is the budget day it describes; a file from an earlier
    day is ignored, so the state expires by itself at the 00:00 UTC reset
    instead of stranding every key as permanently spent.
    """
    path = _spent_keys_path()
    if path is None:
        return set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                lines = [line.strip() for line in handle if line.strip()]
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (FileNotFoundError, OSError):
        return set()
    if not lines or lines[0] != _budget_day():
        return set()
    return set(lines[1:])


def _mark_key_spent(key: str) -> None:
    path = _spent_keys_path()
    if path is None:
        return
    fingerprint = _key_fingerprint(key)
    today = _budget_day()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                lines = [line.strip() for line in handle if line.strip()]
                if not lines or lines[0] != today:
                    # First spend of a new budget day: restart the file.
                    handle.seek(0)
                    handle.truncate()
                    handle.write(f"{today}\n")
                handle.write(f"{fingerprint}\n")
                handle.flush()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        return


def _rotation_order(config: CustomChatAPIConfig) -> list[str]:
    """Pooled keys to try, in order, unspent ones first."""
    keys = list(config.api_keys) or [config.api_key]
    spent = _read_spent_keys()
    fresh = [key for key in keys if _key_fingerprint(key) not in spent]
    # Everything is marked spent (a new day, or a stale state file): fall back to
    # the full list rather than refusing to make any call at all.
    return fresh or keys


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
    api_keys = _split_api_keys(str(values[CUSTOM_API_KEY_ENV_VAR]))
    return CustomChatAPIConfig(
        api_key=api_keys[0],
        api_keys=api_keys,
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
    if provider == ANTHROPIC_PROVIDER:
        return load_anthropic_api_config(required=required)
    if not required:
        return None
    raise RuntimeError(
        f"Unsupported {PROVIDER_ENV_VAR} `{provider}`. "
        f"Expected `{DEFAULT_PROVIDER}`, `{CUSTOM_PROVIDER}` or `{ANTHROPIC_PROVIDER}`."
    )


def build_chat_client(config: ChatAPIConfig):
    if isinstance(config, OpenAIAPIConfig):
        return build_openai_client(config)
    if isinstance(config, AnthropicAPIConfig):
        return build_anthropic_client(config)
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
    if isinstance(config, AnthropicAPIConfig):
        return AnthropicAPIConfig(
            api_key=config.api_key,
            model=resolved_model,
            reasoning_effort=resolved_reasoning_effort,
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
            max_tokens=config.max_tokens,
            provider=config.provider,
        )
    return CustomChatAPIConfig(
        api_key=config.api_key,
        api_keys=config.api_keys,
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
    """Issue one chat completion, rolling onto the next pooled key when one is spent.

    `client` is the one built for the active key. When the gateway reports the
    key's budget is gone, the call is retried on the next unspent key in the
    pool rather than failing the candidate -- a multi-hour sweep otherwise dies
    partway through the day on a per-key daily cap.
    """
    # Exhausting every pooled key must not fail the candidate. A failed
    # candidate is scored zero quality and kept in the beam, so a budget
    # outage would silently corrupt the run rather than pause it. When the
    # budgets reset on a daily clock, wait for the reset and carry on.
    waits_allowed = (
        _env_nonnegative_int(CUSTOM_KEY_MAX_RESET_WAITS_ENV_VAR, default=3)
        if _env_bool(CUSTOM_KEY_WAIT_FOR_RESET_ENV_VAR)
        else 0
    )
    waits_used = 0
    while True:
        try:
            return _attempt_custom_chat_completion(
                client, config=config, request=request, timeout=timeout
            )
        except _AllKeysExhausted as exhausted:
            if waits_used >= waits_allowed:
                raise RuntimeError(
                    f"All {exhausted.key_count} pooled chat API keys are out of budget. "
                    f"Last response: {exhausted.last_error}"
                ) from exhausted.last_error
            waits_used += 1
            delay = _seconds_until_budget_reset()
            print(
                f"DASBench chat: all {exhausted.key_count} pooled keys are out of budget; "
                f"sleeping {delay/3600:.1f} h until the 00:00 UTC reset "
                f"(wait {waits_used}/{waits_allowed}).",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


class _AllKeysExhausted(Exception):
    def __init__(self, key_count: int, last_error: Exception) -> None:
        super().__init__(f"all {key_count} keys exhausted")
        self.key_count = key_count
        self.last_error = last_error


def _attempt_custom_chat_completion(
    client: OpenAI,
    *,
    config: CustomChatAPIConfig,
    request: dict[str, Any],
    timeout: float,
):
    keys = _rotation_order(config)
    with _custom_chat_idle_gate(config):
        last_quota_error: Exception | None = None
        for index, key in enumerate(keys):
            active = client if key == config.api_key and index == 0 else OpenAI(
                api_key=key, base_url=config.base_url, max_retries=0
            )
            retries = 0
            while True:
                try:
                    return active.chat.completions.with_raw_response.create(**request, timeout=timeout)
                except Exception as exc:
                    if _status_code(exc) == 524 and retries < CUSTOM_524_MAX_RETRIES:
                        retries += 1
                        continue
                    if _is_quota_exhausted(exc) and len(keys) > 1:
                        _mark_key_spent(key)
                        last_quota_error = exc
                        print(
                            f"DASBench chat: key {_key_fingerprint(key)} is out of budget; "
                            f"trying {len(keys) - index - 1} more.",
                            file=sys.stderr,
                            flush=True,
                        )
                        break
                    raise
        assert last_quota_error is not None
        raise _AllKeysExhausted(len(keys), last_quota_error)


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
    if isinstance(config, AnthropicAPIConfig):
        # Anthropic takes a different request shape entirely: system prompt,
        # max_tokens and output_config rather than response_format.
        return create_anthropic_message_raw(
            config,
            messages=messages,
            response_format=response_format,
            timeout=timeout,
        )
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
        # An Anthropic Message has content blocks and no choices.
        if getattr(completion, "content", None) is not None:
            return anthropic_message_text(completion)
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
