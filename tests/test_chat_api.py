from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from dasbench.integrations.chat_api import (
    CUSTOM_API_BASE_URL_ENV_VAR,
    CUSTOM_API_KEY_ENV_VAR,
    CUSTOM_KEY_STATE_FILE_ENV_VAR,
    CUSTOM_KEY_WAIT_FOR_RESET_ENV_VAR,
    _mark_key_spent,
    _rotation_order,
    _split_api_keys,
    CUSTOM_MAX_TOKENS_ENV_VAR,
    CUSTOM_MODEL_ENV_VAR,
    CUSTOM_PROVIDER,
    CUSTOM_REASONING_EFFORT_ENV_VAR,
    CUSTOM_TIMEOUT_SECONDS_ENV_VAR,
    DEFAULT_CUSTOM_TIMEOUT_SECONDS,
    LLM_IDLE_LOCK_PATH_ENV_VAR,
    LLM_IDLE_MAX_RUNNING_REQUESTS_ENV_VAR,
    LLM_IDLE_MAX_WAITING_REQUESTS_ENV_VAR,
    LLM_IDLE_METRICS_URL_ENV_VAR,
    LLM_IDLE_POLL_SECONDS_ENV_VAR,
    LLM_WAIT_FOR_IDLE_ENV_VAR,
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
            CUSTOM_MAX_TOKENS_ENV_VAR: "8192",
            LLM_WAIT_FOR_IDLE_ENV_VAR: "1",
            LLM_IDLE_POLL_SECONDS_ENV_VAR: "7",
            LLM_IDLE_MAX_RUNNING_REQUESTS_ENV_VAR: "1",
            LLM_IDLE_MAX_WAITING_REQUESTS_ENV_VAR: "2",
            LLM_IDLE_METRICS_URL_ENV_VAR: "http://example.test/metrics",
            LLM_IDLE_LOCK_PATH_ENV_VAR: "/tmp/example-dasbench-llm.lock",
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
        self.assertEqual(config.max_tokens, 8192)
        self.assertTrue(config.wait_for_idle)
        self.assertEqual(config.idle_poll_seconds, 7.0)
        self.assertEqual(config.idle_max_running_requests, 1)
        self.assertEqual(config.idle_max_waiting_requests, 2)
        self.assertEqual(config.idle_metrics_url, "http://example.test/metrics")
        self.assertEqual(config.idle_lock_path, "/tmp/example-dasbench-llm.lock")
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
            max_tokens=8192,
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
                "max_tokens": 8192,
                "timeout": DEFAULT_CUSTOM_TIMEOUT_SECONDS,
            },
        )

    def test_custom_chat_request_can_omit_response_format(self) -> None:
        fake_client = _FakeClient()
        config = CustomChatAPIConfig(
            api_key="custom-key",
            base_url="https://example.test/api",
            model="custom-model",
        )
        with patch("dasbench.integrations.chat_api.build_chat_client", return_value=fake_client):
            result = create_chat_completion_raw(
                config,
                messages=[{"role": "user", "content": "hello"}],
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(
            fake_client.raw_response.kwargs,
            {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "custom-model",
                "stream": False,
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

    def test_custom_chat_waits_for_vllm_idle_before_request(self) -> None:
        fake_client = _FakeClient()
        config = CustomChatAPIConfig(
            api_key="custom-key",
            base_url="http://localhost:8001/v1",
            model="custom-model",
            wait_for_idle=True,
            idle_poll_seconds=0.01,
            idle_lock_path="/tmp/dasbench-test-chat-idle.lock",
        )

        with patch("dasbench.integrations.chat_api.build_chat_client", return_value=fake_client), patch(
            "dasbench.integrations.chat_api._read_vllm_queue_metrics",
            side_effect=[
                {"running": 1.0, "waiting": 0.0},
                {"running": 0.0, "waiting": 0.0},
            ],
        ) as read_metrics, patch("dasbench.integrations.chat_api.time.sleep", return_value=None):
            result = create_chat_completion_raw(
                config,
                messages=[{"role": "user", "content": "hello"}],
                response_format={"type": "json_schema"},
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(read_metrics.call_count, 2)
        self.assertEqual(fake_client.raw_response.kwargs["model"], "custom-model")

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

    def test_openai_timeout_config_is_passed_to_request(self) -> None:
        fake_client = _FakeClient()
        config = OpenAIAPIConfig(api_key="openai-key", timeout_seconds=14400.0)
        with patch("dasbench.integrations.chat_api.build_chat_client", return_value=fake_client):
            create_chat_completion_raw(
                config,
                messages=[{"role": "user", "content": "hello"}],
                response_format={"type": "json_schema"},
            )

        self.assertEqual(fake_client.raw_response.kwargs["timeout"], 14400.0)

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
                "def solve(instance, analysis=None, manifest=None):\n    return []"
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
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "constraints": ["Return JSON with solution_py."],
                                "response_format": {"solution_py": "full Python module"},
                            }
                        ),
                    }
                ],
                config=config,
            )

        self.assertIn("def solve", text)
        self.assertEqual(metadata["response_model"], "custom-model")
        create.assert_called_once()
        _, kwargs = create.call_args
        self.assertNotIn("response_format", kwargs)
        self.assertEqual(kwargs["timeout"], 10.0)
        prompt = json.loads(kwargs["messages"][0]["content"])
        self.assertEqual(prompt["response_format"], "raw Python module text defining solve(...) or build_solver(...)")

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


class PooledChatAPIKeyTests(unittest.TestCase):
    def test_single_key_parses_unchanged(self) -> None:
        self.assertEqual(_split_api_keys("only-key"), ("only-key",))

    def test_keys_split_on_commas_and_whitespace_and_dedupe(self) -> None:
        raw = 'key-a, key-b\nkey-c  key-a  "key-d"'
        self.assertEqual(_split_api_keys(raw), ("key-a", "key-b", "key-c", "key-d"))

    def test_config_load_keeps_every_pooled_key(self) -> None:
        env = {
            PROVIDER_ENV_VAR: CUSTOM_PROVIDER,
            CUSTOM_API_KEY_ENV_VAR: "key-a,key-b,key-c",
            CUSTOM_API_BASE_URL_ENV_VAR: "https://example.test/api",
            CUSTOM_MODEL_ENV_VAR: "custom-model",
        }
        with _without_dotenv(), patch.dict(os.environ, env, clear=True):
            config = load_chat_api_config()
        self.assertEqual(config.api_keys, ("key-a", "key-b", "key-c"))
        self.assertEqual(config.api_key, "key-a")

    def test_public_dict_never_carries_keys(self) -> None:
        config = CustomChatAPIConfig(
            api_key="key-a",
            api_keys=("key-a", "key-b"),
            base_url="https://example.test/api",
            model="custom-model",
        )
        payload = config.public_dict()
        self.assertNotIn("api_key", payload)
        self.assertNotIn("api_keys", payload)
        self.assertEqual(payload["api_key_count"], 2)
        self.assertNotIn("key-a", json.dumps(payload))

    def test_exhausted_key_rolls_onto_the_next_one(self) -> None:
        spent_client = _FakeClient()
        spent_client.raw_response.create = Mock(side_effect=_FakeStatusError(429))
        fresh_client = _FakeClient()
        fresh_client.raw_response.create = Mock(return_value={"ok": True})
        config = CustomChatAPIConfig(
            api_key="key-a",
            api_keys=("key-a", "key-b"),
            base_url="https://example.test/api",
            model="custom-model",
        )
        with (
            patch("dasbench.integrations.chat_api.build_chat_client", return_value=spent_client),
            patch("dasbench.integrations.chat_api.OpenAI", return_value=fresh_client) as make_client,
        ):
            result = create_chat_completion_raw(
                config,
                messages=[{"role": "user", "content": "hello"}],
                response_format={"type": "json_schema"},
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(make_client.call_args.kwargs["api_key"], "key-b")
        self.assertEqual(fresh_client.raw_response.create.call_count, 1)

    def test_all_keys_exhausted_reports_the_pool_size(self) -> None:
        spent_client = _FakeClient()
        spent_client.raw_response.create = Mock(side_effect=_FakeStatusError(429))
        config = CustomChatAPIConfig(
            api_key="key-a",
            api_keys=("key-a", "key-b"),
            base_url="https://example.test/api",
            model="custom-model",
        )
        with (
            patch("dasbench.integrations.chat_api.build_chat_client", return_value=spent_client),
            patch("dasbench.integrations.chat_api.OpenAI", return_value=spent_client),
        ):
            with self.assertRaisesRegex(RuntimeError, "All 2 pooled chat API keys"):
                create_chat_completion_raw(
                    config,
                    messages=[{"role": "user", "content": "hello"}],
                    response_format={"type": "json_schema"},
                )

    def test_spent_state_from_an_earlier_budget_day_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = os.path.join(tmp, "spent.txt")
            config = CustomChatAPIConfig(
                api_key="key-a",
                api_keys=("key-a", "key-b"),
                base_url="https://example.test/api",
                model="custom-model",
            )
            with patch.dict(os.environ, {CUSTOM_KEY_STATE_FILE_ENV_VAR: state}, clear=False):
                _mark_key_spent("key-a")
                self.assertEqual(_rotation_order(config), ["key-b"])
                # Rewrite the stamp as yesterday: the budget has since reset.
                lines = open(state, encoding="utf-8").read().splitlines()
                lines[0] = "1999-01-01"
                open(state, "w", encoding="utf-8").write("\n".join(lines) + "\n")
                self.assertEqual(_rotation_order(config), ["key-a", "key-b"])

    def test_exhausting_every_key_waits_for_the_reset_instead_of_failing(self) -> None:
        spent_client = _FakeClient()
        spent_client.raw_response.create = Mock(
            side_effect=[_FakeStatusError(429), _FakeStatusError(429), {"ok": True}]
        )
        config = CustomChatAPIConfig(
            api_key="key-a",
            api_keys=("key-a", "key-b"),
            base_url="https://example.test/api",
            model="custom-model",
        )
        env = {CUSTOM_KEY_WAIT_FOR_RESET_ENV_VAR: "1"}
        with (
            patch.dict(os.environ, env, clear=False),
            patch("dasbench.integrations.chat_api.build_chat_client", return_value=spent_client),
            patch("dasbench.integrations.chat_api.OpenAI", return_value=spent_client),
            patch("dasbench.integrations.chat_api.time.sleep") as sleep,
        ):
            result = create_chat_completion_raw(
                config,
                messages=[{"role": "user", "content": "hello"}],
                response_format={"type": "json_schema"},
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(sleep.call_count, 1)  # slept once, then succeeded

    def test_single_key_quota_error_still_surfaces_unchanged(self) -> None:
        fake_client = _FakeClient()
        fake_client.raw_response.create = Mock(side_effect=_FakeStatusError(429))
        config = CustomChatAPIConfig(
            api_key="only-key",
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


if __name__ == "__main__":
    unittest.main()
