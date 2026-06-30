from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAIError
from tqdm import tqdm

from benchmarks.common import (
    DEFAULT_BENCHMARK_ARTIFACTS_ROOT,
    REPRESENTATIVE_FAMILY_BY_PROBLEM,
    resolve_sweep_artifact_root,
    selected_targets,
    write_aggregate_outputs,
)
from dasbench.agents.candidate import build_solver
from dasbench.agents.progress import selection_sort_key, summarize_selection
from dasbench.data import load_manifest, load_split
from dasbench.eval.evaluator import evaluate_solver, failed_summary, write_summary
from dasbench.integrations import (
    ChatAPIConfig,
    CustomChatAPIConfig,
    build_chat_client,
    chat_completion_text,
    chat_config_with_overrides,
    create_chat_completion_raw,
    load_chat_api_config,
    load_openai_dotenv,
)
from dasbench.integrations.chat_api import (
    CUSTOM_MODEL_ENV_VAR,
    CUSTOM_PROVIDER,
    CUSTOM_REASONING_EFFORT_ENV_VAR,
    PROVIDER_ENV_VAR,
)
from dasbench.problems import get_problem_definition
from dasbench.utils import candidate_manifest, public_instance, timestamp_token, write_json, write_jsonl

BENCHMARK_KIND = "llm_pv_benchmark"
CONDITION_ID = "llm_pv"
DEFAULT_SOURCE_RUN_ROOT = Path("artifacts/second_scale_benchmark_v2/20260427_230552")
DEFAULT_SOURCE_CONDITION_ID = "seconds_scale_v2"
OPENAI_MODEL_ENV_VAR = "OPENAI_MODEL"
OPENAI_REASONING_EFFORT_ENV_VAR = "OPENAI_REASONING_EFFORT"
DEFAULT_MODEL = "gpt-5"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_ATTEMPTS = 5
DEFAULT_API_TIMEOUT_SECONDS = 14_400
DEFAULT_PROMPT_TRAIN_EXAMPLES = 64
DEFAULT_PROMPT_JSON_CHAR_LIMIT = 60_000
SOLUTION_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "dasbench" / "schemas" / "solution_code_bundle.json"


@dataclass(frozen=True)
class LLMPVConfig:
    attempts: int
    model: str
    reasoning_effort: str | None
    max_output_tokens: int | None
    api_timeout_seconds: float
    enable_code_interpreter: bool
    tool_choice: str
    verbosity: str
    early_stop_score: float | None
    prompt_train_examples: int
    prompt_json_char_limit: int


@dataclass(frozen=True)
class LLMPVJob:
    sweep_id: str
    artifact_root: Path
    problem: str
    family: str
    source_dataset_dir: Path
    force: bool
    config: LLMPVConfig
    source_run_root: Path | None = None
    source_condition_id: str = DEFAULT_SOURCE_CONDITION_ID

    @property
    def target_root(self) -> Path:
        return self.artifact_root / "targets" / CONDITION_ID / self.problem / self.family

    @property
    def dataset_dir(self) -> Path:
        return self.target_root / "dataset"

    @property
    def run_dir(self) -> Path:
        return self.target_root / "llm_pv_run"

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "synthesis_summary.json"


def _source_dataset_dir(source_run_root: Path, source_condition_id: str, problem: str, family: str) -> Path:
    return source_run_root / "targets" / source_condition_id / problem / family / "dataset"


def _link_or_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _target_dataset_is_current(target_dir: Path, *, source_dir: Path) -> bool:
    required_files = (
        "manifest.json",
        "benchmark_spec.json",
        "reproducibility.json",
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
    )
    if not all((target_dir / name).exists() for name in required_files):
        return False
    try:
        target_spec = json.loads((target_dir / "benchmark_spec.json").read_text(encoding="utf-8"))
        source_spec = json.loads((source_dir / "benchmark_spec.json").read_text(encoding="utf-8"))
        target_manifest = json.loads((target_dir / "manifest.json").read_text(encoding="utf-8"))
        source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        target_spec == source_spec
        and target_manifest.get("problem") == source_manifest.get("problem")
        and target_manifest.get("family") == source_manifest.get("family")
        and target_manifest.get("split_sizes") == source_manifest.get("split_sizes")
        and target_manifest.get("reused_dataset_source") == str(source_dir)
    )


def _materialize_reused_dataset(target_dir: Path, *, source_dir: Path, force: bool) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source_dir}")
    if not force and _target_dataset_is_current(target_dir, source_dir=source_dir):
        return

    shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("benchmark_spec.json", "reproducibility.json", "train.jsonl", "validation.jsonl", "test.jsonl"):
        _link_or_copy_file(source_dir / filename, target_dir / filename)

    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifact_paths"] = {
        "dataset_dir": str(target_dir),
        "splits": {
            "train": str(target_dir / "train.jsonl"),
            "validation": str(target_dir / "validation.jsonl"),
            "test": str(target_dir / "test.jsonl"),
        },
        "manifest": str(target_dir / "manifest.json"),
        "benchmark_spec": str(target_dir / "benchmark_spec.json"),
        "reproducibility": str(target_dir / "reproducibility.json"),
    }
    manifest["reused_dataset_source"] = str(source_dir)
    write_json(target_dir / "manifest.json", manifest)


def _safe_model_dump(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    if isinstance(value, dict):
        return dict(value)
    return None


def _normalize_usage(usage_obj: object) -> dict[str, object]:
    if usage_obj is None:
        return {}
    if isinstance(usage_obj, dict):
        usage = dict(usage_obj)
    else:
        usage = {}
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
        ):
            value = getattr(usage_obj, key, None)
            if value is not None:
                usage[key] = value
        for details_key in (
            "prompt_tokens_details",
            "completion_tokens_details",
            "input_tokens_details",
            "output_tokens_details",
        ):
            details = _safe_model_dump(getattr(usage_obj, details_key, None))
            if details:
                usage[details_key] = details
    if "prompt_tokens" not in usage and "input_tokens" in usage:
        usage["prompt_tokens"] = usage["input_tokens"]
    if "completion_tokens" not in usage and "output_tokens" in usage:
        usage["completion_tokens"] = usage["output_tokens"]
    if usage.get("reasoning_tokens") is None:
        for details_key in ("output_tokens_details", "completion_tokens_details"):
            details = usage.get(details_key)
            if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
                usage["reasoning_tokens"] = details["reasoning_tokens"]
                break
    return usage


def _response_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def _load_solution_schema() -> dict[str, object]:
    return json.loads(SOLUTION_SCHEMA_PATH.read_text(encoding="utf-8"))


def _extract_code_field(obj: object) -> str | None:
    if not isinstance(obj, dict):
        return None
    for key in ("solution_py", "code", "solution"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def extract_solution_code(output_text: str) -> str | None:
    if not output_text:
        return None
    stripped = output_text.strip()
    try:
        code = _extract_code_field(json.loads(stripped))
        if code is not None:
            return code
    except Exception:
        pass

    decoder = json.JSONDecoder()
    best_code: str | None = None
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[index:])
        except Exception:
            continue
        code = _extract_code_field(obj)
        if code is not None:
            best_code = code
    if best_code is not None:
        return best_code

    for block in re.findall(r"```(?:json|python)?\s*([\s\S]*?)```", stripped, flags=re.IGNORECASE):
        nested = extract_solution_code(block)
        if nested is not None:
            return nested
        if "def solve" in block or "def build_solver" in block:
            return block.strip()

    if stripped.lstrip().startswith("{"):
        return None
    if "def solve" in stripped or "def build_solver" in stripped:
        return stripped
    return None


def _prepare_train_examples(
    train_instances: list[dict[str, object]],
    *,
    max_examples: int,
    max_chars: int,
) -> tuple[list[dict[str, object]], bool, int]:
    examples: list[dict[str, object]] = []
    used_chars = 0
    limit = max(0, max_examples)
    char_limit = max(1_000, max_chars)
    for instance in train_instances[:limit]:
        serialized = json.dumps(instance, sort_keys=True)
        if examples and used_chars + len(serialized) > char_limit:
            return examples, True, used_chars
        if not examples and len(serialized) > char_limit:
            clipped = serialized[:char_limit] + f"...[truncated {len(serialized) - char_limit} chars]"
            return [{"truncated_json_text": clipped, "id": instance.get("id")}], True, len(clipped)
        examples.append(instance)
        used_chars += len(serialized)
    return examples, len(train_instances) > len(examples), used_chars


def _prompt_train_summary(summary: dict[str, object]) -> dict[str, object]:
    sanitized = dict(summary)
    sanitized.pop("family", None)
    sanitized.pop("ground_truth_hidden_rule", None)
    return sanitized


def _solution_contract(problem_name: str, manifest: dict[str, object]) -> dict[str, object]:
    instance_params = manifest.get("instance_params")
    if not isinstance(instance_params, dict):
        instance_params = {}
    if problem_name == "tsp":
        num_cities = instance_params.get("num_cities", "instance['num_cities']")
        return {
            "return_type": "list[int]",
            "required_shape": (
                f"Return a plain Python list of exactly {num_cities} city ids. "
                "Each id must appear exactly once and ids are zero-based: 0..num_cities-1."
            ),
            "do_not_return": [
                "a dict such as {'tour': ...}",
                "a tuple such as (tour, length)",
                "a repeated start city at the end of the tour",
                "one-based city ids",
            ],
        }
    if problem_name == "mds":
        return {
            "return_type": "list[int]",
            "required_shape": "Return a plain Python list of selected zero-based vertex ids forming a dominating set.",
            "do_not_return": ["a dict", "a boolean vector", "metadata or objective values"],
        }
    if problem_name == "hitting_set":
        return {
            "return_type": "list[int]",
            "required_shape": (
                "Runtime instances provide instance['num_vertices'] and instance['sets'], where each set is a "
                "list of zero-based vertex ids. Return a plain Python list of selected zero-based vertex ids "
                "that hits every set."
            ),
            "do_not_return": ["a dict", "a boolean vector", "one-based vertex ids", "metadata or objective values"],
        }
    if problem_name == "ocm":
        return {
            "return_type": "list[int]",
            "required_shape": (
                "Runtime instances provide instance['num_fixed'], instance['num_free'], and instance['edges'] "
                "as zero-based [fixed_vertex, free_vertex] pairs. Return a permutation list containing each "
                "free-side vertex id 0..instance['num_free']-1 exactly once."
            ),
            "do_not_return": ["fixed-side vertex ids", "one-based vertex ids", "a dict", "metadata or crossing counts"],
        }
    if problem_name == "dfvs":
        return {
            "return_type": "list[int]",
            "required_shape": (
                "Runtime instances provide instance['num_vertices'] and directed arcs in instance['arcs'] as "
                "zero-based [tail, head] pairs; instance['edges'] is absent. Return a plain Python list of "
                "zero-based vertex ids to delete so the remaining directed graph is acyclic."
            ),
            "do_not_return": ["a dict", "a boolean vector", "one-based vertex ids", "metadata or objective values"],
        }
    if problem_name == "mis":
        return {
            "return_type": "list[int]",
            "required_shape": "Return a plain Python list of selected zero-based vertex ids forming an independent set.",
            "do_not_return": ["a dict", "a boolean vector", "metadata or objective values"],
        }
    if problem_name == "mdkp":
        return {
            "return_type": "list[int]",
            "required_shape": "Return a plain Python list of selected zero-based item ids. A 0/1 vector is accepted but item ids are preferred.",
            "do_not_return": ["a dict", "selected item objects", "metadata or objective values"],
        }
    if problem_name == "packing_lp":
        num_items = instance_params.get("num_items", "instance['num_items']")
        return {
            "return_type": "list[float]",
            "required_shape": f"Return a plain Python list of exactly {num_items} floats in [0, 1], one per item.",
            "do_not_return": ["a dict", "selected item ids", "metadata or objective values"],
        }
    if problem_name == "maxsat":
        num_variables = instance_params.get("num_variables", "instance['num_variables']")
        return {
            "return_type": "list[bool]",
            "required_shape": f"Return a plain Python list of exactly {num_variables} booleans. Index 0 is x1, index 1 is x2, and so on.",
            "do_not_return": ["a dict", "0/1 integers", "signed literals", "metadata or objective values"],
        }
    if problem_name == "coloring":
        num_vertices = instance_params.get("num_vertices", "instance['num_vertices']")
        return {
            "return_type": "list[int]",
            "required_shape": f"Return a plain Python list of exactly {num_vertices} color ids, where index i is vertex i.",
            "do_not_return": ["a dict unless it maps every vertex id to a color", "metadata or objective values"],
        }
    return {
        "return_type": "plain Python object accepted by the problem evaluator",
        "required_shape": "Return only the raw solution object, not a wrapper or metadata payload.",
        "do_not_return": ["a dict wrapper", "metadata or objective values"],
    }


def _build_prompt_messages(
    *,
    manifest: dict[str, object],
    train_summary: dict[str, object],
    train_public: list[dict[str, object]],
    attempt_index: int,
    config: LLMPVConfig,
    plain_python_response: bool = False,
) -> list[dict[str, str]]:
    problem = get_problem_definition(str(manifest["problem"]))
    examples, examples_truncated, example_chars = _prepare_train_examples(
        train_public,
        max_examples=config.prompt_train_examples,
        max_chars=config.prompt_json_char_limit,
    )
    prompt_payload = {
        "method": "LLM-PV propose-and-verify baseline",
        "attempt_index": attempt_index,
        "task": (
            "Write one complete Python solver program for this DasBench target. "
            "This attempt will be executed and scored on held-out validation instances; "
            "the best validation attempt will be selected and then evaluated on test."
        ),
        "problem": {
            "name": problem.name,
            "description": problem.description,
            "metric_definition": problem.metric_definition,
            "instance_schema_version": problem.instance_schema_version,
        },
        "manifest": candidate_manifest(manifest),
        "solution_contract": _solution_contract(problem.name, manifest),
        "train_summary": _prompt_train_summary(train_summary),
        "train_examples": examples,
        "train_examples_metadata": {
            "shown_count": len(examples),
            "available_count": len(train_public),
            "truncated": examples_truncated,
            "approx_json_chars": example_chars,
        },
        "interface": {
            "file": "solution.py",
            "required": "define solve(instance, analysis=None, manifest=None) -> object",
            "alternative": "or define build_solver(analysis=None, manifest=None) -> callable",
        },
        "constraints": [
            (
                "Return only the raw contents of solution.py as Python code. "
                "Do not wrap it in JSON, markdown fences, notes, or explanations."
                if plain_python_response
                else "Return only one JSON object with key solution_py containing the full contents of solution.py."
            ),
            (
                "Do not include text outside the Python module."
                if plain_python_response
                else "Do not include markdown outside the JSON object."
            ),
            "Use train examples only as empirical distribution examples or tuning data.",
            "Do not assume access to optimum_objective, optimum_solution, private fields, files, network, or API calls at solve time.",
            "The solver will receive one public instance at a time and must return a solution in the problem's expected format.",
            "Return the raw solution object from solve(...), not a dict wrapper, score, explanation, tuple, or metadata payload.",
            "Keep per-instance runtime low and deterministic.",
            "Do not import dasbench.integrations, gurobipy, pyscipopt, ortools, highspy, or external exact solvers.",
            "If using randomness, seed it deterministically from the public instance id.",
        ],
        "response_format": (
            "raw Python module text defining solve(...) or build_solver(...)"
            if plain_python_response
            else {
                "solution_py": "full Python module as a string; it must define solve(...) or build_solver(...)",
            }
        ),
    }
    return [
        {
            "role": "system",
            "content": (
                "You generate executable Python candidate solvers for an LLM-PV propose-and-verify loop. "
                "You do not receive validation feedback. Produce a self-contained candidate program."
            ),
        },
        {"role": "user", "content": json.dumps(prompt_payload, indent=2, sort_keys=True)},
    ]


def _api_config_for_run(config: LLMPVConfig) -> ChatAPIConfig:
    base = load_chat_api_config(required=True)
    assert base is not None
    return chat_config_with_overrides(
        base,
        model=config.model,
        reasoning_effort=config.reasoning_effort,
    )


def _call_llm_for_solution(
    *,
    messages: list[dict[str, str]],
    config: LLMPVConfig,
) -> tuple[str, dict[str, object]]:
    api_config = _api_config_for_run(config)
    if isinstance(api_config, CustomChatAPIConfig):
        plain_messages = _messages_for_plain_python_solution(messages)
        return _call_custom_chat_for_solution(messages=plain_messages, config=config, api_config=api_config)
    client = build_chat_client(api_config)
    request_body: dict[str, object] = {
        "model": api_config.model,
        "input": [
            {
                "role": message["role"],
                "content": [{"type": "input_text", "text": message["content"]}],
            }
            for message in messages
        ],
        "reasoning": {"effort": api_config.reasoning_effort},
        "text": {"verbosity": config.verbosity},
    }
    if config.max_output_tokens is not None:
        request_body["max_output_tokens"] = config.max_output_tokens
    if config.enable_code_interpreter:
        request_body["tools"] = [{"type": "code_interpreter", "container": {"type": "auto"}}]
        request_body["tool_choice"] = config.tool_choice
    started = time.perf_counter()
    metadata: dict[str, object] = {
        "api_config": {
            **api_config.public_dict(),
            "max_output_tokens": config.max_output_tokens,
            "api_timeout_seconds": config.api_timeout_seconds,
            "enable_code_interpreter": config.enable_code_interpreter,
            "tool_choice": config.tool_choice if config.enable_code_interpreter else None,
            "verbosity": config.verbosity,
        },
        "request_messages": messages,
    }
    try:
        response = client.responses.create(**request_body, timeout=config.api_timeout_seconds)
    except OpenAIError as exc:
        metadata["generation_wall_ms"] = (time.perf_counter() - started) * 1000.0
        metadata["generation_error"] = f"{type(exc).__name__}: {exc}"
        raise
    metadata["generation_wall_ms"] = (time.perf_counter() - started) * 1000.0
    metadata["response_id"] = getattr(response, "id", None)
    metadata["response_model"] = getattr(response, "model", None)
    metadata["usage"] = _normalize_usage(getattr(response, "usage", None))
    metadata["response_dump"] = _safe_model_dump(response)
    return _response_text(response), metadata


def _call_custom_chat_for_solution(
    *,
    messages: list[dict[str, str]],
    config: LLMPVConfig,
    api_config: CustomChatAPIConfig,
) -> tuple[str, dict[str, object]]:
    if config.enable_code_interpreter:
        raise RuntimeError("Code interpreter is only supported by the OpenAI Responses API path.")
    started = time.perf_counter()
    metadata: dict[str, object] = {
        "api_config": {
            **api_config.public_dict(),
            "max_output_tokens": config.max_output_tokens,
            "api_timeout_seconds": config.api_timeout_seconds,
            "enable_code_interpreter": False,
            "tool_choice": None,
            "verbosity": config.verbosity,
        },
        "request_messages": messages,
    }
    try:
        raw_response = create_chat_completion_raw(
            api_config,
            messages=messages,
            timeout=config.api_timeout_seconds,
        )
    except OpenAIError as exc:
        metadata["generation_wall_ms"] = (time.perf_counter() - started) * 1000.0
        metadata["generation_error"] = f"{type(exc).__name__}: {exc}"
        raise
    metadata["generation_wall_ms"] = (time.perf_counter() - started) * 1000.0
    metadata["status_code"] = raw_response.status_code
    metadata["raw_http_text"] = raw_response.text[:20_000]
    if raw_response.status_code != 200:
        raise RuntimeError(
            f"Custom chat API returned status {raw_response.status_code}: {raw_response.text[:1000]}"
        )
    completion = raw_response.parse()
    metadata["parsed_completion"] = _safe_model_dump(completion)
    metadata["response_model"] = getattr(completion, "model", None)
    metadata["usage"] = _normalize_usage(getattr(completion, "usage", None))
    return chat_completion_text(completion), metadata


def _messages_for_plain_python_solution(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    for message in messages:
        if message.get("role") != "user":
            converted.append(dict(message))
            continue
        try:
            payload = json.loads(message["content"])
        except Exception:
            converted.append(dict(message))
            continue
        payload["constraints"] = [
            (
                "Return only the raw contents of solution.py as Python code. "
                "Do not wrap it in JSON, markdown fences, notes, or explanations."
            ),
            "Use train examples only as empirical distribution examples or tuning data.",
            "Do not assume access to optimum_objective, optimum_solution, private fields, files, network, or API calls at solve time.",
            "The solver will receive one public instance at a time and must return a solution in the problem's expected format.",
            "Return the raw solution object from solve(...), not a dict wrapper, score, explanation, tuple, or metadata payload.",
            "Keep per-instance runtime low and deterministic.",
            "Do not import dasbench.integrations, gurobipy, pyscipopt, ortools, highspy, or external exact solvers.",
            "If using randomness, seed it deterministically from the public instance id.",
        ]
        payload["response_format"] = "raw Python module text defining solve(...) or build_solver(...)"
        converted.append({**message, "content": json.dumps(payload, indent=2, sort_keys=True)})
    return converted


def _write_attempt_failure(
    attempt_dir: Path,
    *,
    attempt_index: int,
    error: str,
    train_size: int,
    validation_size: int,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    if metadata is not None:
        write_summary(attempt_dir / "generation_metadata.json", metadata)
    train_eval = failed_summary(f"llm_pv_attempt_{attempt_index:03d}", "train", train_size, error)
    validation_eval = failed_summary(f"llm_pv_attempt_{attempt_index:03d}", "validation", validation_size, error)
    write_summary(attempt_dir / "train_summary.json", train_eval)
    write_summary(attempt_dir / "validation_summary.json", validation_eval)
    selection = summarize_selection(train_eval, validation_eval)
    record = {
        "slug": f"llm_pv_attempt_{attempt_index:03d}",
        "attempt": attempt_index,
        "candidate_dir": str(attempt_dir),
        "error": error,
        "train": train_eval,
        "validation": validation_eval,
        "selection": selection,
        "generation_metadata": metadata or {},
    }
    write_summary(attempt_dir / "attempt_record.json", record)
    return record


def _evaluate_attempt(
    *,
    attempt_index: int,
    attempt_dir: Path,
    manifest: dict[str, object],
    train_summary: dict[str, object],
    train_public: list[dict[str, object]],
    train_full: list[dict[str, object]],
    validation_full: list[dict[str, object]],
    config: LLMPVConfig,
    dry_run: bool,
) -> dict[str, object]:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    messages = _build_prompt_messages(
        manifest=manifest,
        train_summary=train_summary,
        train_public=train_public,
        attempt_index=attempt_index,
        config=config,
        plain_python_response=_llm_pv_uses_custom_provider(),
    )
    write_summary(attempt_dir / "prompt.json", {"messages": messages})
    if dry_run:
        return _write_attempt_failure(
            attempt_dir,
            attempt_index=attempt_index,
            error="dry_run: API call skipped",
            train_size=len(train_full),
            validation_size=len(validation_full),
            metadata={"dry_run": True, "request_messages": messages},
        )

    raw_text = ""
    metadata: dict[str, object] | None = None
    try:
        raw_text, metadata = _call_llm_for_solution(messages=messages, config=config)
        (attempt_dir / "raw_response.txt").write_text(raw_text + "\n", encoding="utf-8")
        write_summary(attempt_dir / "generation_metadata.json", metadata)
        solution_py = extract_solution_code(raw_text)
        if solution_py is None:
            raise ValueError("No solution_py code found in model response.")
        (attempt_dir / "solution.py").write_text(solution_py.strip() + "\n", encoding="utf-8")
        solver_build_start = time.perf_counter()
        solver = build_solver(attempt_dir, analysis=None, manifest=manifest)
        solver_build_wall_ms = (time.perf_counter() - solver_build_start) * 1000.0
        train_start = time.perf_counter()
        train_eval = evaluate_solver(
            str(manifest["problem"]),
            f"llm_pv_attempt_{attempt_index:03d}",
            solver,
            train_full,
            split="train",
            diagnostics_path=attempt_dir / "train_diagnostics.jsonl",
        )
        validation_start = time.perf_counter()
        validation_eval = evaluate_solver(
            str(manifest["problem"]),
            f"llm_pv_attempt_{attempt_index:03d}",
            solver,
            validation_full,
            split="validation",
            diagnostics_path=attempt_dir / "validation_diagnostics.jsonl",
        )
        timing = {
            "solver_build_wall_ms": solver_build_wall_ms,
            "train_eval_wall_ms": (validation_start - train_start) * 1000.0,
            "validation_eval_wall_ms": (time.perf_counter() - validation_start) * 1000.0,
        }
    except Exception as exc:
        if metadata is None:
            metadata = {"request_messages": messages}
        metadata["generation_or_evaluation_error"] = f"{type(exc).__name__}: {exc}"
        if raw_text:
            (attempt_dir / "raw_response.txt").write_text(raw_text + "\n", encoding="utf-8")
        return _write_attempt_failure(
            attempt_dir,
            attempt_index=attempt_index,
            error=f"{type(exc).__name__}: {exc}",
            train_size=len(train_full),
            validation_size=len(validation_full),
            metadata=metadata,
        )

    write_summary(attempt_dir / "train_summary.json", train_eval)
    write_summary(attempt_dir / "validation_summary.json", validation_eval)
    selection = summarize_selection(train_eval, validation_eval)
    record = {
        "slug": f"llm_pv_attempt_{attempt_index:03d}",
        "attempt": attempt_index,
        "candidate_dir": str(attempt_dir),
        "train": train_eval,
        "validation": validation_eval,
        "selection": selection,
        "timing": timing,
        "generation_metadata": metadata,
    }
    write_summary(attempt_dir / "attempt_record.json", record)
    return record


def _run_target(job: LLMPVJob, *, dry_run: bool) -> dict[str, object]:
    if job.summary_path.exists() and not job.force:
        return _result_from_summary(job, status="skipped", returncode=0)
    if dry_run:
        command = _command_for_job(job)
        return {
            "sweep_id": job.sweep_id,
            "condition_id": CONDITION_ID,
            "problem": job.problem,
            "family": job.family,
            "dataset_dir": str(job.dataset_dir),
            "run_dir": str(job.run_dir),
            "summary_path": str(job.summary_path),
            "source_dataset_dir": str(job.source_dataset_dir),
            "command": command,
            "status": "dry_run",
            "returncode": 0,
        }

    _materialize_reused_dataset(job.dataset_dir, source_dir=job.source_dataset_dir, force=job.force)
    manifest = load_manifest(job.dataset_dir)
    problem = get_problem_definition(str(manifest["problem"]))
    train_public = [public_instance(instance) for instance in load_split(job.dataset_dir, "train")]
    train_full = load_split(job.dataset_dir, "train")
    validation_full = load_split(job.dataset_dir, "validation")
    test_full = load_split(job.dataset_dir, "test")
    train_summary = problem.summarize_training_data(train_public, manifest) if train_public else {}

    job.run_dir.mkdir(parents=True, exist_ok=True)
    write_summary(
        job.run_dir / "run_manifest.json",
        {
            "benchmark_kind": BENCHMARK_KIND,
            "condition_id": CONDITION_ID,
            "sweep_id": job.sweep_id,
            "dataset_dir": str(job.dataset_dir),
            "source_dataset_dir": str(job.source_dataset_dir),
            "problem": job.problem,
            "family": job.family,
            "config": {
                "attempts": job.config.attempts,
                "model": job.config.model,
                "reasoning_effort": job.config.reasoning_effort,
                "max_output_tokens": job.config.max_output_tokens,
                "api_timeout_seconds": job.config.api_timeout_seconds,
                "enable_code_interpreter": job.config.enable_code_interpreter,
                "tool_choice": job.config.tool_choice,
                "verbosity": job.config.verbosity,
                "early_stop_score": job.config.early_stop_score,
                "prompt_train_examples": job.config.prompt_train_examples,
                "prompt_json_char_limit": job.config.prompt_json_char_limit,
            },
        },
    )

    attempts_dir = job.run_dir / "attempts"
    attempt_records: list[dict[str, object]] = []
    history: list[dict[str, object]] = []
    stopped_early = False
    for attempt_index in range(1, job.config.attempts + 1):
        record = _evaluate_attempt(
            attempt_index=attempt_index,
            attempt_dir=attempts_dir / f"attempt_{attempt_index:03d}",
            manifest=manifest,
            train_summary=train_summary,
            train_public=train_public,
            train_full=train_full,
            validation_full=validation_full,
            config=job.config,
            dry_run=False,
        )
        attempt_records.append(record)
        best_so_far = max(attempt_records, key=lambda item: selection_sort_key(item["selection"]))
        history.append(
            {
                "attempt": attempt_index,
                "slug": record["slug"],
                "selection": record["selection"],
                "best_slug": best_so_far["slug"],
                "best_selection": best_so_far["selection"],
                "train": record["train"],
                "validation": record["validation"],
            }
        )
        write_jsonl(job.run_dir / "attempts.jsonl", attempt_records)
        write_json(job.run_dir / "performance_history.json", history)
        validation_quality = float(record["validation"].get("average_normalized_quality", 0.0))
        if job.config.early_stop_score is not None and validation_quality >= job.config.early_stop_score:
            stopped_early = True
            break

    best_candidate = max(attempt_records, key=lambda item: selection_sort_key(item["selection"]))
    best_candidate = dict(best_candidate)
    solution_path = Path(str(best_candidate["candidate_dir"])) / "solution.py"
    if solution_path.exists() and not best_candidate.get("error"):
        selected_dir = job.run_dir / "selected_solver"
        selected_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(solution_path, selected_dir / "solution.py")
        solver = build_solver(Path(str(best_candidate["candidate_dir"])), analysis=None, manifest=manifest)
        test_start = time.perf_counter()
        test_eval = evaluate_solver(
            str(manifest["problem"]),
            str(best_candidate["slug"]),
            solver,
            test_full,
            split="test",
            diagnostics_path=job.run_dir / "selected_test_diagnostics.jsonl",
        )
        best_candidate["timing"] = dict(best_candidate.get("timing", {}))
        best_candidate["timing"]["best_candidate_test_wall_ms"] = (time.perf_counter() - test_start) * 1000.0
    else:
        test_eval = failed_summary(
            str(best_candidate["slug"]),
            "test",
            len(test_full),
            "No successful solution.py is available for the selected LLM-PV candidate.",
        )
    best_candidate["test"] = test_eval
    write_summary(Path(str(best_candidate["candidate_dir"])) / "test_summary.json", test_eval)

    summary = {
        "benchmark_kind": BENCHMARK_KIND,
        "condition_id": CONDITION_ID,
        "sweep_id": job.sweep_id,
        "problem": job.problem,
        "family": job.family,
        "generator": "llm_pv",
        "dataset_dir": str(job.dataset_dir),
        "source_dataset_dir": str(job.source_dataset_dir),
        "output_dir": str(job.run_dir),
        "attempts_requested": job.config.attempts,
        "attempts_evaluated": len(attempt_records),
        "model": job.config.model,
        "reasoning_effort": job.config.reasoning_effort,
        "max_output_tokens": job.config.max_output_tokens,
        "api_timeout_seconds": job.config.api_timeout_seconds,
        "enable_code_interpreter": job.config.enable_code_interpreter,
        "tool_choice": job.config.tool_choice,
        "verbosity": job.config.verbosity,
        "early_stop_score": job.config.early_stop_score,
        "stopped_early": stopped_early,
        "best_candidate": best_candidate,
        "attempts": attempt_records,
        "rounds": [
            {
                "iteration": 0,
                "evaluated_this_round": [str(record["slug"]) for record in attempt_records],
                "best_selected_slug": str(best_candidate["slug"]),
                "best_selected_train": best_candidate["train"],
                "best_selected_validation": best_candidate["validation"],
                "best_selected_selection": best_candidate["selection"],
            }
        ],
        "train_summary": train_summary,
        "performance_history_path": str(job.run_dir / "performance_history.json"),
        "attempts_jsonl_path": str(job.run_dir / "attempts.jsonl"),
    }
    write_summary(job.summary_path, summary)
    return _result_from_summary(job, status="completed", returncode=0)


def _result_from_summary(job: LLMPVJob, *, status: str, returncode: int) -> dict[str, object]:
    result = {
        "sweep_id": job.sweep_id,
        "condition_id": CONDITION_ID,
        "problem": job.problem,
        "family": job.family,
        "dataset_dir": str(job.dataset_dir),
        "run_dir": str(job.run_dir),
        "summary_path": str(job.summary_path),
        "source_dataset_dir": str(job.source_dataset_dir),
        "command": _command_for_job(job),
        "status": status,
        "returncode": returncode,
    }
    if job.summary_path.exists():
        summary = json.loads(job.summary_path.read_text(encoding="utf-8"))
        result.update(_metrics_from_summary(summary))
    return result


def _metrics_from_summary(summary: dict[str, object]) -> dict[str, object]:
    best_candidate = summary.get("best_candidate", {})
    if not isinstance(best_candidate, dict):
        best_candidate = {}
    train = best_candidate.get("train", {}) if isinstance(best_candidate.get("train"), dict) else {}
    validation = best_candidate.get("validation", {}) if isinstance(best_candidate.get("validation"), dict) else {}
    test = best_candidate.get("test", {}) if isinstance(best_candidate.get("test"), dict) else {}
    usage = _usage_totals(summary.get("attempts", []))
    return {
        "generator": "llm_pv",
        "agent_slug": best_candidate.get("slug"),
        "chosen_best_attempt": best_candidate.get("attempt"),
        "attempts_requested": summary.get("attempts_requested"),
        "attempts_evaluated": summary.get("attempts_evaluated"),
        "stopped_early": summary.get("stopped_early"),
        "model": summary.get("model"),
        "reasoning_effort": summary.get("reasoning_effort"),
        "max_output_tokens": summary.get("max_output_tokens"),
        "api_timeout_seconds": summary.get("api_timeout_seconds"),
        "enable_code_interpreter": summary.get("enable_code_interpreter"),
        "verbosity": summary.get("verbosity"),
        "agent_train_quality": _metric(train, "average_normalized_quality"),
        "agent_train_optimality": _metric(train, "optimality_rate"),
        "agent_train_feasibility": _metric(train, "feasibility_rate"),
        "agent_train_runtime_ms": _metric(train, "average_runtime_ms"),
        "agent_validation_quality": _metric(validation, "average_normalized_quality"),
        "agent_validation_optimality": _metric(validation, "optimality_rate"),
        "agent_validation_feasibility": _metric(validation, "feasibility_rate"),
        "agent_validation_runtime_ms": _metric(validation, "average_runtime_ms"),
        "agent_test_quality": _metric(test, "average_normalized_quality"),
        "agent_test_optimality": _metric(test, "optimality_rate"),
        "agent_test_feasibility": _metric(test, "feasibility_rate"),
        "agent_test_runtime_ms": _metric(test, "average_runtime_ms"),
        "evaluated_candidate_count": summary.get("attempts_evaluated"),
        **usage,
    }


def _usage_totals(attempts: object) -> dict[str, object]:
    if not isinstance(attempts, list):
        return {}
    totals = {
        "api_prompt_tokens": 0,
        "api_completion_tokens": 0,
        "api_total_tokens": 0,
        "api_reasoning_tokens": 0,
    }
    any_usage = False
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        metadata = attempt.get("generation_metadata")
        if not isinstance(metadata, dict):
            continue
        usage = metadata.get("usage")
        if not isinstance(usage, dict):
            continue
        any_usage = True
        totals["api_prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        totals["api_completion_tokens"] += int(usage.get("completion_tokens") or 0)
        totals["api_total_tokens"] += int(usage.get("total_tokens") or 0)
        totals["api_reasoning_tokens"] += int(usage.get("reasoning_tokens") or 0)
    return totals if any_usage else {}


def _metric(payload: dict[str, object], name: str) -> float | None:
    value = payload.get(name)
    if value is None:
        value = payload.get(f"{name}_mean")
    return float(value) if isinstance(value, (int, float)) else None


def _command_for_job(job: LLMPVJob) -> list[str] | None:
    if job.source_run_root is None:
        return None
    command = [
        "python",
        "-m",
        "benchmarks.llm_pv_benchmark",
        "--sweep-id",
        job.sweep_id,
        "--problem",
        job.problem,
        "--family",
        job.family,
        "--source-run-root",
        str(job.source_run_root),
        "--attempts",
        str(job.config.attempts),
        "--model",
        job.config.model,
        "--prompt-train-examples",
        str(job.config.prompt_train_examples),
        "--prompt-json-char-limit",
        str(job.config.prompt_json_char_limit),
    ]
    if job.source_condition_id != DEFAULT_SOURCE_CONDITION_ID:
        command.extend(["--source-condition-id", job.source_condition_id])
    if job.config.reasoning_effort is not None:
        command.extend(["--reasoning-effort", job.config.reasoning_effort])
    if job.config.max_output_tokens is not None:
        command.extend(["--max-output-tokens", str(job.config.max_output_tokens)])
    command.extend(["--api-timeout-seconds", str(job.config.api_timeout_seconds)])
    if job.config.enable_code_interpreter:
        command.append("--enable-code-interpreter")
        command.extend(["--tool-choice", job.config.tool_choice])
    command.extend(["--verbosity", job.config.verbosity])
    if job.config.early_stop_score is None:
        command.append("--no-early-stop")
    else:
        command.extend(["--early-stop-score", str(job.config.early_stop_score)])
    return command


def _build_config(args: argparse.Namespace) -> LLMPVConfig:
    reasoning_effort = None if args.reasoning_effort is None else str(args.reasoning_effort).strip()
    if reasoning_effort == "none":
        reasoning_effort = None
    return LLMPVConfig(
        attempts=max(1, int(args.attempts)),
        model=str(args.model),
        reasoning_effort=reasoning_effort or None,
        max_output_tokens=None if args.max_output_tokens is None else max(1, int(args.max_output_tokens)),
        api_timeout_seconds=max(1.0, float(args.api_timeout_seconds)),
        enable_code_interpreter=bool(args.enable_code_interpreter),
        tool_choice=str(args.tool_choice),
        verbosity=str(args.verbosity),
        early_stop_score=None if args.no_early_stop else float(args.early_stop_score),
        prompt_train_examples=max(0, int(args.prompt_train_examples)),
        prompt_json_char_limit=max(1_000, int(args.prompt_json_char_limit)),
    )


def _llm_pv_default_model() -> str:
    load_openai_dotenv()
    if _llm_pv_uses_custom_provider():
        return os.getenv(CUSTOM_MODEL_ENV_VAR, DEFAULT_MODEL)
    return os.getenv(OPENAI_MODEL_ENV_VAR, DEFAULT_MODEL)


def _llm_pv_default_reasoning_effort() -> str | None:
    load_openai_dotenv()
    if _llm_pv_uses_custom_provider():
        value = os.getenv(CUSTOM_REASONING_EFFORT_ENV_VAR)
        return value.strip() if value and value.strip() else None
    value = os.getenv(OPENAI_REASONING_EFFORT_ENV_VAR, DEFAULT_REASONING_EFFORT)
    return value.strip() if value and value.strip() else None


def _llm_pv_uses_custom_provider() -> bool:
    load_openai_dotenv()
    return os.getenv(PROVIDER_ENV_VAR, "").strip().lower() == CUSTOM_PROVIDER


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an LLM-PV propose-and-verify baseline on DasBench datasets. "
            "Each attempt generates one candidate solver, validation selects the best, and test evaluates the selected solver."
        )
    )
    parser.add_argument("--sweep-id")
    parser.add_argument("--output-root", default=str(DEFAULT_BENCHMARK_ARTIFACTS_ROOT))
    parser.add_argument("--problem")
    parser.add_argument("--family")
    parser.add_argument("--all-families", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--source-run-root", default=str(DEFAULT_SOURCE_RUN_ROOT))
    parser.add_argument("--source-condition-id", default=DEFAULT_SOURCE_CONDITION_ID)
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument(
        "--model",
        default=_llm_pv_default_model(),
        help=(
            f"LLM model. Defaults to ${CUSTOM_MODEL_ENV_VAR} for custom_chat, "
            f"otherwise ${OPENAI_MODEL_ENV_VAR} when set, otherwise {DEFAULT_MODEL}."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default=_llm_pv_default_reasoning_effort(),
        help=(
            f"LLM reasoning effort. Defaults to ${CUSTOM_REASONING_EFFORT_ENV_VAR} for custom_chat "
            f"when set, otherwise ${OPENAI_REASONING_EFFORT_ENV_VAR} when set, otherwise "
            f"{DEFAULT_REASONING_EFFORT} for OpenAI."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help=(
            "Optional cap on total generated output tokens, including reasoning tokens. "
            "Unset by default so the model can use its natural/model-level limit."
        ),
    )
    parser.add_argument(
        "--api-timeout-seconds",
        type=float,
        default=DEFAULT_API_TIMEOUT_SECONDS,
        help=(
            "Per-attempt LLM API request timeout in seconds. Defaults to "
            f"{DEFAULT_API_TIMEOUT_SECONDS} seconds so long reasoning calls can finish."
        ),
    )
    parser.add_argument("--enable-code-interpreter", action="store_true")
    parser.add_argument("--tool-choice", choices=["auto", "none"], default="auto")
    parser.add_argument("--verbosity", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--early-stop-score", type=float, default=1.0)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--prompt-train-examples", type=int, default=DEFAULT_PROMPT_TRAIN_EXAMPLES)
    parser.add_argument("--prompt-json-char-limit", type=int, default=DEFAULT_PROMPT_JSON_CHAR_LIMIT)
    return parser


def build_jobs(args: argparse.Namespace, *, sweep_id: str | None = None) -> list[LLMPVJob]:
    resolved_sweep_id = sweep_id or args.sweep_id or timestamp_token()
    targets = selected_targets(
        args.problem,
        args.family,
        representative_only=not bool(args.all_families),
    )
    source_run_root = Path(args.source_run_root)
    artifact_root = resolve_sweep_artifact_root(args.output_root, BENCHMARK_KIND, resolved_sweep_id)
    config = _build_config(args)
    return [
        LLMPVJob(
            sweep_id=resolved_sweep_id,
            artifact_root=artifact_root,
            problem=problem,
            family=family,
            source_dataset_dir=_source_dataset_dir(source_run_root, args.source_condition_id, problem, family),
            force=bool(args.force),
            config=config,
            source_run_root=source_run_root,
            source_condition_id=str(args.source_condition_id),
        )
        for problem, family in targets
    ]


def _aggregate_rows(results: list[dict[str, object]], jobs: list[LLMPVJob]) -> list[dict[str, object]]:
    job_by_key = {(job.problem, job.family): job for job in jobs}
    rows: list[dict[str, object]] = []
    for result in sorted(results, key=lambda item: (str(item["problem"]), str(item["family"]))):
        job = job_by_key[(str(result["problem"]), str(result["family"]))]
        manifest = load_manifest(job.dataset_dir) if (job.dataset_dir / "manifest.json").exists() else {}
        row = dict(result)
        row.update(
            {
                "train_size": manifest.get("split_sizes", {}).get("train") if isinstance(manifest.get("split_sizes"), dict) else None,
                "validation_size": (
                    manifest.get("split_sizes", {}).get("validation")
                    if isinstance(manifest.get("split_sizes"), dict)
                    else None
                ),
                "test_size": manifest.get("split_sizes", {}).get("test") if isinstance(manifest.get("split_sizes"), dict) else None,
                "representative_family": REPRESENTATIVE_FAMILY_BY_PROBLEM.get(job.problem),
                "attempts": job.config.attempts,
                "model": job.config.model,
                "reasoning_effort": job.config.reasoning_effort,
                "max_output_tokens": job.config.max_output_tokens,
                "api_timeout_seconds": job.config.api_timeout_seconds,
                "enable_code_interpreter": job.config.enable_code_interpreter,
                "verbosity": job.config.verbosity,
            }
        )
        rows.append(row)
    return rows


def run_sweep(args: argparse.Namespace, *, sweep_id: str | None = None) -> dict[str, object]:
    load_openai_dotenv()
    resolved_sweep_id = sweep_id or args.sweep_id or timestamp_token()
    jobs = build_jobs(args, sweep_id=resolved_sweep_id)
    if not args.dry_run:
        load_chat_api_config(required=True)
    output_dir = resolve_sweep_artifact_root(args.output_root, BENCHMARK_KIND, resolved_sweep_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_workers = max(1, min(int(args.max_workers), len(jobs) or 1))
    print(f"Running {BENCHMARK_KIND} `{resolved_sweep_id}` with {len(jobs)} targets and max_workers={max_workers}")

    results: list[dict[str, object]] = []
    status_counts = {"completed": 0, "failed": 0, "skipped": 0, "dry_run": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_target, job, dry_run=bool(args.dry_run)): job for job in jobs}
        with tqdm(total=len(jobs), desc=f"{BENCHMARK_KIND}:{resolved_sweep_id}", unit="target", dynamic_ncols=True) as progress:
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "sweep_id": resolved_sweep_id,
                        "condition_id": CONDITION_ID,
                        "problem": job.problem,
                        "family": job.family,
                        "dataset_dir": str(job.dataset_dir),
                        "run_dir": str(job.run_dir),
                        "summary_path": str(job.summary_path),
                        "source_dataset_dir": str(job.source_dataset_dir),
                        "command": _command_for_job(job),
                        "status": "failed",
                        "returncode": 1,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                results.append(result)
                status = str(result["status"])
                if status in status_counts:
                    status_counts[status] += 1
                progress.update(1)
                progress.set_postfix(status_counts, refresh=False)

    rows = _aggregate_rows(results, jobs)
    summary = write_aggregate_outputs(
        output_dir=output_dir,
        sweep_id=resolved_sweep_id,
        sweep_kind=BENCHMARK_KIND,
        rows=rows,
        results=sorted(results, key=lambda item: (str(item["problem"]), str(item["family"]))),
    )
    print(f"Aggregate JSON: {summary['aggregate_json_path']}")
    print(f"Aggregate CSV: {summary['aggregate_csv_path']}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_sweep(args, sweep_id=args.sweep_id)
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
