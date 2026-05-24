from __future__ import annotations

import os
from dataclasses import asdict, dataclass
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

DEFAULT_CUSTOM_TIMEOUT_SECONDS = 14_400.0
CUSTOM_524_MAX_RETRIES = 3


@dataclass(frozen=True)
class CustomChatAPIConfig:
    api_key: str
    base_url: str
    model: str
    reasoning_effort: str | None = None
    timeout_seconds: float = DEFAULT_CUSTOM_TIMEOUT_SECONDS
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
        )
    return CustomChatAPIConfig(
        api_key=config.api_key,
        base_url=config.base_url,
        model=resolved_model,
        reasoning_effort=resolved_reasoning_effort,
        timeout_seconds=config.timeout_seconds,
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
    request: dict[str, Any],
    timeout: float,
):
    retries = 0
    while True:
        try:
            return client.chat.completions.with_raw_response.create(**request, timeout=timeout)
        except Exception as exc:
            if _status_code(exc) != 524 or retries >= CUSTOM_524_MAX_RETRIES:
                raise
            retries += 1


def create_chat_completion_raw(
    config: ChatAPIConfig,
    *,
    messages: list[dict[str, str]],
    response_format: dict[str, Any],
    timeout: float | None = None,
):
    client = build_chat_client(config)
    request: dict[str, Any] = {
        "messages": messages,
        "model": config.model,
        "response_format": response_format,
    }
    if isinstance(config, CustomChatAPIConfig):
        request["stream"] = False
        if config.reasoning_effort is not None:
            request["reasoning_effort"] = config.reasoning_effort
        return _create_custom_chat_completion_raw(
            client,
            request=request,
            timeout=config.timeout_seconds if timeout is None else timeout,
        )
    else:
        request["reasoning_effort"] = config.reasoning_effort
    if timeout is None:
        return client.chat.completions.with_raw_response.create(**request)
    return client.chat.completions.with_raw_response.create(**request, timeout=timeout)


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
