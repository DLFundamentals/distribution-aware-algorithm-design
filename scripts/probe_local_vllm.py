from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from openai import OpenAI


def _schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "dasbench_local_vllm_probe",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ok": {"type": "boolean"},
                    "note": {"type": "string"},
                },
                "required": ["ok", "note"],
            },
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe a local vLLM OpenAI-compatible server.")
    parser.add_argument("--base-url", default=os.environ.get("CUSTOM_CHAT_API_BASE_URL", "http://localhost:8001/v1"))
    parser.add_argument("--api-key", default=os.environ.get("CUSTOM_CHAT_API_KEY", "local-vllm"))
    parser.add_argument("--model", default=os.environ.get("CUSTOM_CHAT_MODEL") or os.environ.get("VLLM_SERVED_MODEL_NAME"))
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)

    deadline = time.monotonic() + args.timeout_seconds
    last_error: Exception | None = None
    model = args.model

    while time.monotonic() < deadline:
        try:
            models = client.models.list()
            if model is None:
                model = models.data[0].id
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": 'Return JSON with ok=true and note="schema probe".',
                    }
                ],
                response_format=_schema(),
                temperature=0,
                max_tokens=64,
            )
            content = completion.choices[0].message.content or ""
            payload = json.loads(content)
            if payload.get("ok") is not True or not isinstance(payload.get("note"), str):
                raise RuntimeError(f"Unexpected structured payload: {payload!r}")
            print(json.dumps({"base_url": args.base_url, "model": model, "payload": payload}, indent=2))
            return 0
        except Exception as exc:  # pragma: no cover - operational retry loop
            last_error = exc
            time.sleep(args.interval_seconds)

    raise SystemExit(f"Local vLLM probe failed after {args.timeout_seconds:g}s: {last_error}")


if __name__ == "__main__":
    raise SystemExit(main())
