from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

from dasbench.integrations.chat_api import (
    CUSTOM_API_BASE_URL_ENV_VAR,
    CUSTOM_API_KEY_ENV_VAR,
    CUSTOM_MODEL_ENV_VAR,
    CUSTOM_PROVIDER,
    CUSTOM_REASONING_EFFORT_ENV_VAR,
    CUSTOM_TIMEOUT_SECONDS_ENV_VAR,
    DEFAULT_CUSTOM_TIMEOUT_SECONDS,
    DEFAULT_PROVIDER,
    PROVIDER_ENV_VAR,
    CustomChatAPIConfig,
    create_chat_completion_raw,
    load_chat_api_config,
)
from dasbench.integrations.openai_api import OpenAIAPIConfig


class _FakeWithRawResponse:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return {"ok": True}


class _FakeClient:
    def __init__(self) -> None:
        self.raw_response = _FakeWithRawResponse()
        self.chat = type(
            "FakeChat",
            (),
            {
                "completions": type(
                    "FakeCompletions",
                    (),
                    {"with_raw_response": self.raw_response},
                )()
            },
        )()


class _FakeParsedCompletion:
    model = "custom-model"
    usage = None

    def __init__(self, content: str) -> None:
        message = type("FakeMessage", (), {"content": content})()
        self.choices = [type("FakeChoice", (), {"message": message})()]


class _FakeRawCompletion:
    status_code = 200

    def __init__(self, content: str) -> None:
        self.text = content
        self._parsed = _FakeParsedCompletion(content)

    def parse(self) -> _FakeParsedCompletion:
        return self._parsed


class _FakeStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def _without_dotenv():
    return patch.multiple(
        "dasbench.integrations.chat_api",
        load_dotenv=lambda: True,
    )


class ChatAPIConfigTests(unittest.TestCase):
    def test_default_provider_uses_openai_config(self) -> None:
        with _without_dotenv(), patch(
            "dasbench.integrations.openai_api.load_dotenv",
            return_value=True,
        ), patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True):
            config = load_chat_api_config(required=True)

        self.assertIsInstance(config, OpenAIAPIConfig)
        assert isinstance(config, OpenAIAPIConfig)
        self.assertEqual(config.model, "gpt-5.2")
        self.assertEqual(config.reasoning_effort, "xhigh")

    def test_custom_provider_loads_neutral_env_vars(self) -> None:
        env = {
            PROVIDER_ENV_VAR: CUSTOM_PROVIDER,
            CUSTOM_API_KEY_ENV_VAR: "custom-key",
            CUSTOM_API_BASE_URL_ENV_VAR: "https://example.test/api",
            CUSTOM_MODEL_ENV_VAR: "custom-model",
            CUSTOM_REASONING_EFFORT_ENV_VAR: "medium",
            CUSTOM_TIMEOUT_SECONDS_ENV_VAR: "3600",
        }
        with _without_dotenv(), patch.dict(os.environ, env, clear=True):
            config = load_chat_api_config(required=True)

        self.assertIsInstance(config, CustomChatAPIConfig)
        assert isinstance(config, CustomChatAPIConfig)
        self.assertEqual(config.provider, CUSTOM_PROVIDER)
        self.assertEqual(config.base_url, "https://example.test/api")
        self.assertEqual(config.model, "custom-model")
        self.assertEqual(config.reasoning_effort, "medium")
        self.assertEqual(config.timeout_seconds, 3600.0)
        self.assertNotIn("api_key", config.public_dict())

    def test_missing_custom_provider_config_fails_clearly(self) -> None:
        with _without_dotenv(), patch.dict(os.environ, {PROVIDER_ENV_VAR: CUSTOM_PROVIDER}, clear=True):
            self.assertIsNone(load_chat_api_config(required=False))
            with self.assertRaisesRegex(RuntimeError, CUSTOM_API_KEY_ENV_VAR):
                load_chat_api_config(required=True)

    def test_unknown_provider_fails_clearly(self) -> None:
        with _without_dotenv(), patch.dict(os.environ, {PROVIDER_ENV_VAR: "other"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, DEFAULT_PROVIDER):
                load_chat_api_config(required=True)


class ChatAPIRequestTests(unittest.TestCase):
    def test_custom_chat_request_uses_chat_completions_shape(self) -> None:
        fake_client = _FakeClient()
        config = CustomChatAPIConfig(
            api_key="custom-key",
            base_url="https://example.test/api",
            model="custom-model",
            reasoning_effort="medium",
        )
        with patch("dasbench.integrations.chat_api.build_chat_client", return_value=fake_client):
            result = create_chat_completion_raw(
                config,
                messages=[{"role": "user", "content": "hello"}],
                response_format={"type": "json_schema"},
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(
            fake_client.raw_response.kwargs,
            {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "custom-model",
                "response_format": {"type": "json_schema"},
                "stream": False,
                "reasoning_effort": "medium",
                "timeout": DEFAULT_CUSTOM_TIMEOUT_SECONDS,
            },
        )

    def test_custom_chat_retries_524_responses_three_times(self) -> None:
        fake_client = _FakeClient()
        fake_client.raw_response.create = Mock(
            side_effect=[
                _FakeStatusError(524),
                _FakeStatusError(524),
                _FakeStatusError(524),
                {"ok": True},
            ]
        )
        config = CustomChatAPIConfig(
            api_key="custom-key",
            base_url="https://example.test/api",
            model="custom-model",
        )
        with patch("dasbench.integrations.chat_api.build_chat_client", return_value=fake_client):
            result = create_chat_completion_raw(
                config,
                messages=[{"role": "user", "content": "hello"}],
                response_format={"type": "json_schema"},
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(fake_client.raw_response.create.call_count, 4)

    def test_custom_chat_stops_after_three_524_retries(self) -> None:
        fake_client = _FakeClient()
        fake_client.raw_response.create = Mock(side_effect=_FakeStatusError(524))
        config = CustomChatAPIConfig(
            api_key="custom-key",
            base_url="https://example.test/api",
            model="custom-model",
        )
        with patch("dasbench.integrations.chat_api.build_chat_client", return_value=fake_client):
            with self.assertRaises(_FakeStatusError):
                create_chat_completion_raw(
                    config,
                    messages=[{"role": "user", "content": "hello"}],
                    response_format={"type": "json_schema"},
                )

        self.assertEqual(fake_client.raw_response.create.call_count, 4)

    def test_openai_request_shape_remains_unchanged(self) -> None:
        fake_client = _FakeClient()
        config = OpenAIAPIConfig(api_key="openai-key")
        with patch("dasbench.integrations.chat_api.build_chat_client", return_value=fake_client):
            create_chat_completion_raw(
                config,
                messages=[{"role": "user", "content": "hello"}],
                response_format={"type": "json_schema"},
            )

        self.assertEqual(
            fake_client.raw_response.kwargs,
            {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-5.2",
                "response_format": {"type": "json_schema"},
                "reasoning_effort": "xhigh",
            },
        )

    def test_llm_pv_custom_provider_uses_chat_completions(self) -> None:
        from benchmarks.llm_pv_benchmark import LLMPVConfig, _call_llm_for_solution

        api_config = CustomChatAPIConfig(
            api_key="custom-key",
            base_url="https://example.test/api",
            model="custom-model",
            reasoning_effort=None,
        )
        create = Mock(
            return_value=_FakeRawCompletion(
                '{"solution_py": "def solve(instance, analysis=None, manifest=None):\\n    return []", "notes": ""}'
            )
        )
        config = LLMPVConfig(
            attempts=1,
            model="custom-model",
            reasoning_effort=None,
            max_output_tokens=None,
            api_timeout_seconds=10.0,
            enable_code_interpreter=False,
            tool_choice="auto",
            verbosity="low",
            early_stop_score=1.0,
            prompt_train_examples=1,
            prompt_json_char_limit=1000,
        )

        with patch("benchmarks.llm_pv_benchmark.load_chat_api_config", return_value=api_config), patch(
            "benchmarks.llm_pv_benchmark.create_chat_completion_raw",
            create,
        ):
            text, metadata = _call_llm_for_solution(
                messages=[{"role": "user", "content": "hello"}],
                config=config,
            )

        self.assertIn("solution_py", text)
        self.assertEqual(metadata["response_model"], "custom-model")
        create.assert_called_once()
        _, kwargs = create.call_args
        self.assertIs(kwargs["response_format"]["json_schema"]["strict"], True)
        self.assertEqual(kwargs["timeout"], 10.0)

    def test_llm_pv_custom_provider_rejects_code_interpreter(self) -> None:
        from benchmarks.llm_pv_benchmark import LLMPVConfig, _call_llm_for_solution

        api_config = CustomChatAPIConfig(
            api_key="custom-key",
            base_url="https://example.test/api",
            model="custom-model",
        )
        config = LLMPVConfig(
            attempts=1,
            model="custom-model",
            reasoning_effort=None,
            max_output_tokens=None,
            api_timeout_seconds=10.0,
            enable_code_interpreter=True,
            tool_choice="auto",
            verbosity="low",
            early_stop_score=1.0,
            prompt_train_examples=1,
            prompt_json_char_limit=1000,
        )

        with patch("benchmarks.llm_pv_benchmark.load_chat_api_config", return_value=api_config):
            with self.assertRaisesRegex(RuntimeError, "Code interpreter"):
                _call_llm_for_solution(messages=[{"role": "user", "content": "hello"}], config=config)


if __name__ == "__main__":
    unittest.main()
