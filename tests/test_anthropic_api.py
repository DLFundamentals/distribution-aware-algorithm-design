from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from dasbench.integrations.anthropic_api import (
    ANTHROPIC_API_KEY_ENV_VAR,
    ANTHROPIC_MAX_TOKENS_ENV_VAR,
    ANTHROPIC_MODEL_ENV_VAR,
    ANTHROPIC_PROVIDER,
    ANTHROPIC_REASONING_EFFORT_ENV_VAR,
    AnthropicAPIConfig,
    anthropic_message_text,
    build_output_config,
    create_anthropic_message_raw,
    load_anthropic_api_config,
    schema_from_response_format,
    split_system_messages,
)
from dasbench.integrations.chat_api import (
    PROVIDER_ENV_VAR,
    chat_completion_text,
    chat_config_with_overrides,
    load_chat_api_config,
)

# The stored schema files use OpenAI's response_format envelope.
OPENAI_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "solution_code_bundle",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"solution_py": {"type": "string"}},
            "required": ["solution_py"],
        },
    },
}


class _Block:
    def __init__(self, block_type: str, text: str | None = None, thinking: str | None = None) -> None:
        self.type = block_type
        if text is not None:
            self.text = text
        if thinking is not None:
            self.thinking = thinking


class _Message:
    def __init__(self, content: list[_Block]) -> None:
        self.content = content


class AnthropicConfigTests(unittest.TestCase):
    def test_config_is_loaded_from_the_environment(self) -> None:
        env = {
            PROVIDER_ENV_VAR: ANTHROPIC_PROVIDER,
            ANTHROPIC_API_KEY_ENV_VAR: "test-key",
            ANTHROPIC_MODEL_ENV_VAR: "claude-sonnet-5",
            ANTHROPIC_REASONING_EFFORT_ENV_VAR: "xhigh",
            ANTHROPIC_MAX_TOKENS_ENV_VAR: "4096",
        }
        with patch.dict("os.environ", env, clear=False):
            config = load_anthropic_api_config()
        assert config is not None
        self.assertEqual(config.model, "claude-sonnet-5")
        self.assertEqual(config.reasoning_effort, "xhigh")
        self.assertEqual(config.max_tokens, 4096)
        self.assertEqual(config.provider, ANTHROPIC_PROVIDER)

    def test_provider_selects_anthropic(self) -> None:
        env = {PROVIDER_ENV_VAR: ANTHROPIC_PROVIDER, ANTHROPIC_API_KEY_ENV_VAR: "test-key"}
        with patch.dict("os.environ", env, clear=False), patch(
            "dasbench.integrations.chat_api.load_dotenv"
        ):
            config = load_chat_api_config()
        self.assertIsInstance(config, AnthropicAPIConfig)

    def test_missing_key_is_optional_when_not_required(self) -> None:
        with patch.dict("os.environ", {ANTHROPIC_API_KEY_ENV_VAR: ""}, clear=False):
            self.assertIsNone(load_anthropic_api_config(required=False))
            with self.assertRaises(RuntimeError):
                load_anthropic_api_config(required=True)

    def test_unsupported_effort_is_rejected_at_load_time(self) -> None:
        """Better to fail on startup than to have every API call 400."""
        env = {
            ANTHROPIC_API_KEY_ENV_VAR: "test-key",
            ANTHROPIC_REASONING_EFFORT_ENV_VAR: "ludicrous",
        }
        with patch.dict("os.environ", env, clear=False):
            with self.assertRaises(RuntimeError):
                load_anthropic_api_config()

    def test_public_dict_drops_the_api_key(self) -> None:
        payload = AnthropicAPIConfig(api_key="super-secret").public_dict()
        self.assertNotIn("api_key", payload)
        self.assertNotIn("super-secret", str(payload))

    def test_overrides_preserve_provider_and_budget(self) -> None:
        config = AnthropicAPIConfig(api_key="k", max_tokens=1234)
        updated = chat_config_with_overrides(config, model="claude-opus-5", reasoning_effort="max")
        self.assertIsInstance(updated, AnthropicAPIConfig)
        self.assertEqual(updated.model, "claude-opus-5")
        self.assertEqual(updated.reasoning_effort, "max")
        self.assertEqual(updated.max_tokens, 1234)


class AnthropicRequestShapeTests(unittest.TestCase):
    def test_system_messages_are_lifted_out(self) -> None:
        system, conversation = split_system_messages(
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hello"},
            ]
        )
        self.assertEqual(system, "be terse")
        self.assertEqual(conversation, [{"role": "user", "content": "hello"}])

    def test_multiple_system_messages_are_joined_not_dropped(self) -> None:
        system, conversation = split_system_messages(
            [
                {"role": "system", "content": "first"},
                {"role": "system", "content": "second"},
                {"role": "user", "content": "hi"},
            ]
        )
        self.assertEqual(system, "first\n\nsecond")
        self.assertEqual(len(conversation), 1)

    def test_openai_envelope_is_unwrapped_to_a_bare_schema(self) -> None:
        schema = schema_from_response_format(OPENAI_RESPONSE_FORMAT)
        assert schema is not None
        self.assertEqual(schema["type"], "object")
        self.assertIn("solution_py", schema["properties"])
        self.assertNotIn("json_schema", schema)

    def test_bare_schema_passes_through(self) -> None:
        bare = {"type": "object", "properties": {"a": {"type": "string"}}}
        self.assertEqual(schema_from_response_format(bare), bare)

    def test_no_response_format_yields_effort_only(self) -> None:
        self.assertEqual(build_output_config(response_format=None, reasoning_effort="high"), {"effort": "high"})
        self.assertIsNone(build_output_config(response_format=None, reasoning_effort=None))

    def test_request_carries_max_tokens_adaptive_thinking_and_output_config(self) -> None:
        config = AnthropicAPIConfig(api_key="k", model="claude-opus-5", reasoning_effort="high", max_tokens=777)
        client = MagicMock()
        client.with_options.return_value = client
        with patch("dasbench.integrations.anthropic_api.build_anthropic_client", return_value=client):
            create_anthropic_message_raw(
                config,
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "go"},
                ],
                response_format=OPENAI_RESPONSE_FORMAT,
            )
        request = client.messages.with_raw_response.create.call_args.kwargs
        self.assertEqual(request["model"], "claude-opus-5")
        self.assertEqual(request["max_tokens"], 777)
        self.assertEqual(request["system"], "sys")
        self.assertEqual(request["messages"], [{"role": "user", "content": "go"}])
        self.assertEqual(request["thinking"], {"type": "adaptive"})
        self.assertEqual(request["output_config"]["effort"], "high")
        self.assertEqual(request["output_config"]["format"]["type"], "json_schema")
        # budget_tokens is rejected by current Claude models.
        self.assertNotIn("budget_tokens", str(request))
        self.assertNotIn("response_format", request)


class AnthropicResponseTests(unittest.TestCase):
    def test_text_blocks_are_joined_and_thinking_ignored(self) -> None:
        message = _Message(
            [
                _Block("thinking", thinking="internal reasoning"),
                _Block("text", text='{"solution_py": "x"}'),
            ]
        )
        self.assertEqual(anthropic_message_text(message), '{"solution_py": "x"}')
        self.assertNotIn("internal reasoning", anthropic_message_text(message))

    def test_chat_completion_text_handles_an_anthropic_message(self) -> None:
        """The agents call chat_completion_text regardless of provider."""
        message = _Message([_Block("text", text="hello")])
        self.assertEqual(chat_completion_text(message), "hello")

    def test_empty_content_is_empty_string(self) -> None:
        self.assertEqual(anthropic_message_text(_Message([])), "")


if __name__ == "__main__":
    unittest.main()
