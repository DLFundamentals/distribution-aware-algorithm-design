"""Anthropic Messages API provider.

Selected with `LLM_PROVIDER=anthropic`, so synthesis does not require an
OpenAI-compatible endpoint.

Three differences from the OpenAI path are handled here rather than leaking into
the agents:

* The system prompt is a top-level `system` argument, not a message with
  `role="system"`, so the message list is split before the request.
* `max_tokens` is required. The agents never set it, so a configurable default
  is supplied; a truncated response would otherwise surface as a schema
  validation failure rather than a budget problem.
* Structured output is `output_config={"format": {...}}` carrying the bare JSON
  schema, where OpenAI takes `response_format` wrapping the same schema in a
  `json_schema` envelope. The repo's schema files are stored in the OpenAI shape,
  so the envelope is unwrapped here and the two providers stay on one set of
  schema files.

Reasoning effort maps onto `output_config.effort` alongside adaptive thinking.
`budget_tokens` is deliberately not used: current Claude models reject it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

ANTHROPIC_PROVIDER = "anthropic"

ANTHROPIC_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
ANTHROPIC_MODEL_ENV_VAR = "ANTHROPIC_MODEL"
ANTHROPIC_REASONING_EFFORT_ENV_VAR = "ANTHROPIC_REASONING_EFFORT"
ANTHROPIC_BASE_URL_ENV_VAR = "ANTHROPIC_BASE_URL"
ANTHROPIC_TIMEOUT_SECONDS_ENV_VAR = "ANTHROPIC_TIMEOUT_SECONDS"
ANTHROPIC_MAX_TOKENS_ENV_VAR = "ANTHROPIC_MAX_TOKENS"

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_ANTHROPIC_REASONING_EFFORT = "high"
# Solution bundles carry a whole Python module as a JSON string field, so the
# default has to be generous; too small truncates mid-module.
DEFAULT_ANTHROPIC_MAX_TOKENS = 32_000
DEFAULT_ANTHROPIC_TIMEOUT_SECONDS = 14_400.0

SUPPORTED_REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class AnthropicAPIConfig:
    api_key: str
    model: str = DEFAULT_ANTHROPIC_MODEL
    reasoning_effort: str | None = DEFAULT_ANTHROPIC_REASONING_EFFORT
    base_url: str | None = None
    timeout_seconds: float | None = DEFAULT_ANTHROPIC_TIMEOUT_SECONDS
    max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS
    provider: str = ANTHROPIC_PROVIDER

    def public_dict(self) -> dict[str, Any]:
        """Config for candidate metadata, with the key removed."""
        return {
            "provider": self.provider,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "max_tokens": self.max_tokens,
        }


def _env_value(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def load_anthropic_api_config(*, required: bool = True) -> AnthropicAPIConfig | None:
    api_key = _env_value(ANTHROPIC_API_KEY_ENV_VAR)
    if api_key is None:
        if not required:
            return None
        raise RuntimeError(
            f"{ANTHROPIC_API_KEY_ENV_VAR} is not set. "
            f"Required when {'LLM_PROVIDER'}=`{ANTHROPIC_PROVIDER}`."
        )

    effort = _env_value(ANTHROPIC_REASONING_EFFORT_ENV_VAR) or DEFAULT_ANTHROPIC_REASONING_EFFORT
    if effort.lower() not in SUPPORTED_REASONING_EFFORTS:
        raise RuntimeError(
            f"{ANTHROPIC_REASONING_EFFORT_ENV_VAR}=`{effort}` is not supported. "
            f"Expected one of {', '.join(SUPPORTED_REASONING_EFFORTS)}."
        )

    timeout_raw = _env_value(ANTHROPIC_TIMEOUT_SECONDS_ENV_VAR)
    max_tokens_raw = _env_value(ANTHROPIC_MAX_TOKENS_ENV_VAR)
    return AnthropicAPIConfig(
        api_key=api_key,
        model=_env_value(ANTHROPIC_MODEL_ENV_VAR) or DEFAULT_ANTHROPIC_MODEL,
        reasoning_effort=effort.lower(),
        base_url=_env_value(ANTHROPIC_BASE_URL_ENV_VAR),
        timeout_seconds=float(timeout_raw) if timeout_raw else DEFAULT_ANTHROPIC_TIMEOUT_SECONDS,
        max_tokens=int(max_tokens_raw) if max_tokens_raw else DEFAULT_ANTHROPIC_MAX_TOKENS,
    )


def build_anthropic_client(config: AnthropicAPIConfig):
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            "The anthropic package is required for LLM_PROVIDER=anthropic. Install it with `uv sync`."
        ) from exc
    kwargs: dict[str, Any] = {"api_key": config.api_key, "max_retries": 0}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return anthropic.Anthropic(**kwargs)


def split_system_messages(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
    """Lift `role="system"` entries into Anthropic's top-level system prompt.

    Several are joined rather than dropped so a caller that sends more than one
    does not silently lose instructions.
    """
    system_chunks = [str(m.get("content", "")) for m in messages if m.get("role") == "system"]
    conversation = [dict(m) for m in messages if m.get("role") != "system"]
    system = "\n\n".join(chunk for chunk in system_chunks if chunk) or None
    return system, conversation


def schema_from_response_format(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    """Unwrap an OpenAI `response_format` into the bare JSON schema.

    Accepts the stored `{"type": "json_schema", "json_schema": {"schema": ...}}`
    shape and a bare schema, so either can be handed to this provider.
    """
    if not isinstance(response_format, dict):
        return None
    envelope = response_format.get("json_schema")
    if isinstance(envelope, dict):
        schema = envelope.get("schema")
        return schema if isinstance(schema, dict) else None
    schema = response_format.get("schema")
    if isinstance(schema, dict):
        return schema
    if response_format.get("type") == "object" or "properties" in response_format:
        return response_format
    return None


def build_output_config(
    *,
    response_format: dict[str, Any] | None,
    reasoning_effort: str | None,
) -> dict[str, Any] | None:
    output_config: dict[str, Any] = {}
    if reasoning_effort:
        output_config["effort"] = reasoning_effort
    schema = schema_from_response_format(response_format)
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    return output_config or None


def create_anthropic_message_raw(
    config: AnthropicAPIConfig,
    *,
    messages: list[dict[str, str]],
    response_format: dict[str, Any] | None = None,
    timeout: float | None = None,
):
    client = build_anthropic_client(config)
    system, conversation = split_system_messages(messages)
    request: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "messages": conversation,
        "thinking": {"type": "adaptive"},
    }
    if system is not None:
        request["system"] = system
    output_config = build_output_config(
        response_format=response_format,
        reasoning_effort=config.reasoning_effort,
    )
    if output_config is not None:
        request["output_config"] = output_config

    resolved_timeout = config.timeout_seconds if timeout is None else timeout
    if resolved_timeout is not None:
        client = client.with_options(timeout=resolved_timeout)
    return client.messages.with_raw_response.create(**request)


def anthropic_message_text(message: object) -> str:
    """Join the text blocks of a Message, ignoring thinking and tool blocks."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    chunks: list[str] = []
    for block in content or []:
        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")
        if block_type != "text":
            continue
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return "\n".join(chunks).strip()


__all__ = [
    "ANTHROPIC_API_KEY_ENV_VAR",
    "ANTHROPIC_BASE_URL_ENV_VAR",
    "ANTHROPIC_MAX_TOKENS_ENV_VAR",
    "ANTHROPIC_MODEL_ENV_VAR",
    "ANTHROPIC_PROVIDER",
    "ANTHROPIC_REASONING_EFFORT_ENV_VAR",
    "ANTHROPIC_TIMEOUT_SECONDS_ENV_VAR",
    "AnthropicAPIConfig",
    "anthropic_message_text",
    "build_anthropic_client",
    "build_output_config",
    "create_anthropic_message_raw",
    "load_anthropic_api_config",
    "schema_from_response_format",
    "split_system_messages",
]
