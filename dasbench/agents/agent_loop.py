"""Constrained-action agent generator for distribution-aware synthesis.

This is a Path-A scaffold: a minimal, model-agnostic ReAct-style loop that lets a
(typically local) model *drive its own* code generation with real execution
feedback instead of the harness-gated, structured-output pipeline in
``dasbench.agents.llm``.

Design goals:

* Produce the exact same candidate contract as every other generator -- a
  directory holding ``hypothesis.json``, ``analyze.py`` and ``solution.py`` -- so
  the whole downstream (evaluation, scoring, normalization, reporting, beam
  selection) is byte-for-byte identical and the ``agent`` condition is directly
  comparable to the ``llm`` condition.
* Reuse the outer beam/iteration/selection/test-evaluation machinery from
  ``dasbench.agents.llm`` unchanged (imported, not copied logic), so this module
  only replaces the *per-candidate generation step*.
* Stay provider-agnostic: no ``response_format`` is requested, so the loop works
  identically against a local vLLM OpenAI-compatible server and the OpenAI API.

The agent's only actions are ``write_file`` (hypothesis.json / analyze.py /
solution.py), ``run_check`` (a train-only feasibility + quality probe), ``peek_instance``
(inspect a sanitized train instance) and ``finish``. There is no arbitrary shell:
the harness executes a fixed action set, which keeps the leak boundary explicit
(the model only ever sees sanitized public train data plus the same aggregate
train feedback the ``llm`` generator already exposes during semantic repair).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAIError

from dasbench.agents.candidate import build_solver, run_analysis
from dasbench.agents.llm import (
    LLMPlan,
    _child_plans,
    _empty_train_summary,
    _evaluate_best_candidate_test,
    _hypothesis_diversity_key,
    _plan_payload,
    _seed_plans,
    _select_survivors,
)
from dasbench.agents.progress import (
    performance_plot_filename,
    progress_point,
    selection_sort_key,
    summarize_selection,
    write_history,
    write_performance_plot,
)
from dasbench.data import load_manifest, load_split
from dasbench.eval.evaluator import evaluate_solver, failed_summary, write_summary
from dasbench.integrations import (
    chat_completion_text,
    create_chat_completion_raw,
    load_chat_api_config,
)
from dasbench.problems import get_problem_definition
from dasbench.timing import BenchmarkTimingReporter

MODULE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = MODULE_DIR.parent
SYSTEM_PROMPT_PATH = PACKAGE_ROOT / "prompts" / "llm_system_prompt.txt"

AGENT_MAX_STEPS_ENV_VAR = "DASBENCH_AGENT_MAX_STEPS"
DEFAULT_AGENT_MAX_STEPS = 16
AGENT_CHECK_SAMPLE_ENV_VAR = "DASBENCH_AGENT_CHECK_SAMPLE"
DEFAULT_AGENT_CHECK_SAMPLE = 8
AGENT_FORMAT_RETRIES_ENV_VAR = "DASBENCH_AGENT_FORMAT_RETRIES"
DEFAULT_AGENT_FORMAT_RETRIES = 3

PROMPT_JSON_CHAR_LIMIT = 12_000
PEEK_CHAR_LIMIT = 4_000
CHECK_TRANSCRIPT_CHAR_LIMIT = 6_000

REQUIRED_FILES = ("hypothesis.json", "analyze.py", "solution.py")
_WRITABLE_FILES = frozenset(REQUIRED_FILES)

_ACTION_RE = re.compile(r"^\s*ACTION:\s*(?P<verb>[a-zA-Z_]+)\s*(?P<arg>\S+)?", re.MULTILINE)
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_]*\n(?P<body>.*?)```", re.DOTALL)


def _int_env(var: str, default: int) -> int:
    raw = os.environ.get(var)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{var} must be an integer, got {raw!r}.") from exc
    return value


def _agent_max_steps() -> int:
    return max(1, _int_env(AGENT_MAX_STEPS_ENV_VAR, DEFAULT_AGENT_MAX_STEPS))


def _agent_check_sample() -> int:
    return max(1, _int_env(AGENT_CHECK_SAMPLE_ENV_VAR, DEFAULT_AGENT_CHECK_SAMPLE))


def _agent_format_retries() -> int:
    return max(0, _int_env(AGENT_FORMAT_RETRIES_ENV_VAR, DEFAULT_AGENT_FORMAT_RETRIES))


def _load_base_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def _truncate(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def _compact_json(value: object, *, limit: int = PROMPT_JSON_CHAR_LIMIT) -> str:
    text = json.dumps(value, indent=2, sort_keys=True, default=str)
    return _truncate(text, limit=limit)


# --------------------------------------------------------------------------- #
# Action protocol
# --------------------------------------------------------------------------- #


@dataclass
class ParsedAction:
    verb: str
    arg: str | None
    payload: str | None


def _parse_action(text: str) -> ParsedAction | None:
    match = _ACTION_RE.search(text)
    if match is None:
        return None
    verb = match.group("verb").strip().lower()
    arg = match.group("arg")
    arg = arg.strip() if arg else None
    fence = _FENCE_RE.search(text, match.end())
    payload = fence.group("body") if fence else None
    return ParsedAction(verb=verb, arg=arg, payload=payload)


def _agent_protocol_prompt() -> str:
    return (
        "You work in a scratch workspace by emitting exactly ONE action per reply.\n"
        "Reply with a short reasoning line, then an action in this exact form:\n\n"
        "ACTION: <verb> [argument]\n"
        "```\n<optional payload>\n```\n\n"
        "Available actions:\n"
        "- ACTION: write_file <name>  (name is one of hypothesis.json, analyze.py, solution.py)\n"
        "    The fenced block is the COMPLETE new contents of that file.\n"
        "    hypothesis.json must be a JSON object stating your hidden-rule hypothesis;\n"
        "    include a short 'diversity_key' string field naming the structural idea.\n"
        "    analyze.py must define analyze(train_instances, manifest=None) -> dict.\n"
        "    solution.py must define solve(instance, analysis=None, manifest=None) -> object.\n"
        "- ACTION: run_check   Runs analyze.py then solution.py on a representative TRAIN sample\n"
        "    spanning the full training set (small to large instances) and reports feasibility,\n"
        "    normalized quality, runtime, and any tracebacks. Use this to debug and improve your\n"
        "    code before finishing; make sure it holds up on the largest instances too.\n"
        "- ACTION: peek_instance <index>   Prints one sanitized training instance.\n"
        "- ACTION: finish   Declare the candidate complete. Only succeeds once all three files\n"
        "    exist and run_check builds a solver without error.\n\n"
        "Rules: emit ONE action per reply. Write complete files, never diffs. Infer the hidden\n"
        "distributional rule from the training data; do not rely on named family metadata.\n"
        "Iterate with run_check until feasibility is 1.0 and quality is as high as you can get,\n"
        "then finish."
    )


def _build_task_message(
    *,
    manifest: dict[str, object],
    train_summary: dict[str, object],
    plan: LLMPlan,
    train_count: int,
) -> str:
    payload = {
        "problem_class": manifest.get("problem"),
        "note": "The specific family/distribution identity is hidden; infer exploitable structure from samples.",
        "search_focus": plan.focus,
        "hypothesis_directive": plan.hypothesis_directive,
        "train_instance_count": train_count,
        "train_summary": train_summary,
        "candidate_interface": {
            "analyze.py": "def analyze(train_instances, manifest=None) -> dict",
            "solution.py": "def solve(instance, analysis=None, manifest=None) -> object",
        },
        "objective": (
            "Maximize validation normalized quality with low per-instance runtime by exploiting the "
            "inferred distributional structure. Solver quality on train is your feedback signal."
        ),
    }
    return _compact_json(payload)


# --------------------------------------------------------------------------- #
# Sandboxed action handlers
# --------------------------------------------------------------------------- #


def _handle_write_file(candidate_dir: Path, action: ParsedAction) -> str:
    name = (action.arg or "").strip()
    if name not in _WRITABLE_FILES:
        return (
            f"write_file rejected: unknown target {name!r}. "
            f"Allowed files: {', '.join(sorted(_WRITABLE_FILES))}."
        )
    if action.payload is None:
        return f"write_file {name} rejected: no fenced code/JSON block was provided."
    body = action.payload
    if name == "hypothesis.json":
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            return f"hypothesis.json is not valid JSON: {exc}. Re-send a valid JSON object."
        if not isinstance(parsed, dict):
            return "hypothesis.json must be a JSON object (got a non-object)."
        body = json.dumps(parsed, indent=2, sort_keys=True) + "\n"
    else:
        try:
            compile(body, name, "exec")
        except SyntaxError as exc:
            return f"{name} has a Python syntax error: {exc}. Re-send the full corrected file."
    (candidate_dir / name).write_text(body, encoding="utf-8")
    return f"Wrote {name} ({len(body)} chars)."


def _handle_peek_instance(train_public: list[dict[str, object]], action: ParsedAction) -> str:
    if not train_public:
        return "No training instances are available to peek."
    index = 0
    if action.arg:
        try:
            index = int(action.arg)
        except ValueError:
            return f"peek_instance argument {action.arg!r} is not an integer index."
    if not (0 <= index < len(train_public)):
        return f"peek_instance index {index} out of range [0, {len(train_public) - 1}]."
    return f"train_instance[{index}]:\n" + _compact_json(train_public[index], limit=PEEK_CHAR_LIMIT)


def _spread_indices(total: int, count: int) -> list[int]:
    """Evenly spaced instance indices across ``[0, total)``.

    Used so ``run_check`` probes a representative slice of the whole train set
    (including the largest/hardest instances) rather than just the first N, which
    would let generalization failures that only appear on big instances slip past
    the agent's self-correction loop.
    """
    if count >= total:
        return list(range(total))
    if count <= 1:
        return [0]
    raw = (round(i * (total - 1) / (count - 1)) for i in range(count))
    seen: set[int] = set()
    ordered: list[int] = []
    for index in raw:
        if index not in seen:
            seen.add(index)
            ordered.append(index)
    return ordered


def _run_check(
    *,
    candidate_dir: Path,
    manifest: dict[str, object],
    problem_name: str,
    train_public: list[dict[str, object]],
    train_full: list[dict[str, object]],
    sample_size: int,
) -> tuple[bool, str]:
    """Run analyze+solve on a spread train sample. Returns (solver_built, feedback_text).

    The sample is evenly spaced across the full train set (not the first N) so the
    probe covers the instance-size range and exposes generalization failures in-loop.
    Only aggregate train feedback is surfaced (feasibility/quality/runtime plus the
    evaluator's per-instance messages) -- the same information the ``llm`` generator
    already exposes to the model during semantic repair. No stored optima leak.
    """
    missing = [name for name in ("analyze.py", "solution.py") if not (candidate_dir / name).exists()]
    if missing:
        return False, f"run_check skipped: missing files {missing}. Write them first."

    indices = _spread_indices(len(train_full), sample_size)
    sample_full = [train_full[i] for i in indices]
    sample_public = [train_public[i] for i in indices]
    try:
        analysis = run_analysis(candidate_dir, sample_public, manifest=manifest)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the model
        return False, f"analyze.py failed during execution:\n{type(exc).__name__}: {exc}"

    try:
        solver = build_solver(candidate_dir, analysis=analysis, manifest=manifest)
    except Exception as exc:  # noqa: BLE001
        return False, f"solution.py failed to build a solver:\n{type(exc).__name__}: {exc}"

    try:
        summary = evaluate_solver(problem_name, "agent_check", solver, sample_full, split="train")
    except Exception as exc:  # noqa: BLE001
        return True, f"solver built, but evaluation raised:\n{type(exc).__name__}: {exc}"

    report = {
        "sampled_train_instances": len(sample_full),
        "sample_spans_full_train_set": len(sample_full) == len(train_full) or len(indices) > 1,
        "total_train_instances": len(train_full),
        "feasibility_rate": summary.get("feasibility_rate"),
        "average_normalized_quality": summary.get("average_normalized_quality"),
        "optimality_rate": summary.get("optimality_rate"),
        "average_runtime_ms": summary.get("average_runtime_ms"),
        "error": summary.get("error"),
        "feedback": summary.get("feedback"),
    }
    return True, _truncate(_compact_json(report), limit=CHECK_TRANSCRIPT_CHAR_LIMIT)


# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #


@dataclass
class AgentRunResult:
    transcript: list[dict[str, object]] = field(default_factory=list)
    steps_used: int = 0
    finished: bool = False
    solver_built: bool = False
    api_error: str | None = None


def _chat(config, messages: list[dict[str, str]]) -> str:
    raw = create_chat_completion_raw(config, messages=messages, response_format=None)
    status = getattr(raw, "status_code", 200)
    if status != 200:
        raise OpenAIError(f"chat completion returned status {status}: {getattr(raw, 'text', '')[:500]}")
    completion = raw.parse()
    return chat_completion_text(completion)


def _run_constrained_agent(
    *,
    plan: LLMPlan,
    manifest: dict[str, object],
    train_summary: dict[str, object],
    train_public: list[dict[str, object]],
    train_full: list[dict[str, object]],
    candidate_dir: Path,
    config,
    problem_name: str,
) -> AgentRunResult:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = _load_base_system_prompt() + "\n\n" + _agent_protocol_prompt()
    task_message = _build_task_message(
        manifest=manifest,
        train_summary=train_summary,
        plan=plan,
        train_count=len(train_public),
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_message},
    ]

    result = AgentRunResult()
    max_steps = _agent_max_steps()
    sample_size = _agent_check_sample()
    format_retries_left = _agent_format_retries()

    step = 0
    while step < max_steps:
        try:
            reply = _chat(config, messages)
        except OpenAIError as exc:
            result.api_error = f"{type(exc).__name__}: {exc}"
            break
        messages.append({"role": "assistant", "content": reply})
        action = _parse_action(reply)

        if action is None:
            if format_retries_left <= 0:
                result.transcript.append({"step": step, "error": "unparseable_action", "reply": reply[:2000]})
                break
            format_retries_left -= 1
            observation = (
                "No valid ACTION was found. Reply with exactly one line 'ACTION: <verb> [arg]' "
                "optionally followed by a fenced ``` block. Do not include multiple actions."
            )
            messages.append({"role": "user", "content": observation})
            result.transcript.append({"step": step, "action": None, "observation": observation})
            continue

        step += 1
        result.steps_used = step
        observation: str

        if action.verb == "write_file":
            observation = _handle_write_file(candidate_dir, action)
        elif action.verb == "peek_instance":
            observation = _handle_peek_instance(train_public, action)
        elif action.verb == "run_check":
            built, observation = _run_check(
                candidate_dir=candidate_dir,
                manifest=manifest,
                problem_name=problem_name,
                train_public=train_public,
                train_full=train_full,
                sample_size=sample_size,
            )
            result.solver_built = result.solver_built or built
        elif action.verb == "finish":
            missing = [name for name in REQUIRED_FILES if not (candidate_dir / name).exists()]
            if missing:
                observation = f"Cannot finish: missing required files {missing}. Write them, then run_check."
            else:
                built, check_text = _run_check(
                    candidate_dir=candidate_dir,
                    manifest=manifest,
                    problem_name=problem_name,
                    train_public=train_public,
                    train_full=train_full,
                    sample_size=sample_size,
                )
                result.solver_built = result.solver_built or built
                if built:
                    result.finished = True
                    result.transcript.append(
                        {"step": step, "action": "finish", "observation": _truncate(check_text, limit=2000)}
                    )
                    break
                observation = "Cannot finish yet: run_check could not build a working solver:\n" + check_text
        else:
            observation = (
                f"Unknown action {action.verb!r}. Valid actions: write_file, run_check, peek_instance, finish."
            )

        messages.append({"role": "user", "content": observation})
        result.transcript.append(
            {
                "step": step,
                "action": action.verb,
                "arg": action.arg,
                "observation": _truncate(observation, limit=2000),
            }
        )

    return result


# --------------------------------------------------------------------------- #
# Per-candidate evaluation (mirrors llm._evaluate_plan record schema)
# --------------------------------------------------------------------------- #


def _read_json_if_exists(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _failure_record(
    *,
    plan: LLMPlan,
    candidate_dir: Path,
    evaluation_dir: Path,
    hypothesis: object | None,
    analyze_py: str,
    solution_py: str,
    error: str,
    train_count: int,
    validation_count: int,
    timing: dict[str, float],
) -> dict[str, object]:
    train_eval = failed_summary(plan.slug(), "train", train_count, error)
    validation_eval = failed_summary(plan.slug(), "validation", validation_count, error)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    write_summary(evaluation_dir / "train_summary.json", train_eval)
    write_summary(evaluation_dir / "validation_summary.json", validation_eval)
    return {
        "slug": plan.slug(),
        "plan": _plan_payload(plan),
        "hypothesis": hypothesis if isinstance(hypothesis, dict) else None,
        "candidate_dir": str(candidate_dir),
        "evaluation_dir": str(evaluation_dir),
        "code_bundle": {"analyze_py": analyze_py, "solution_py": solution_py},
        "stage_notes": {"hypothesis": "", "analyze": "", "solution": ""},
        "analysis_output": None,
        "train": train_eval,
        "validation": validation_eval,
        "selection": summarize_selection(train_eval, validation_eval),
        "timing": timing,
    }


def _evaluate_plan_via_agent(
    plan: LLMPlan,
    *,
    manifest: dict[str, object],
    train_summary: dict[str, object],
    train_instances_public: list[dict[str, object]],
    train_instances_full: list[dict[str, object]],
    validation_instances_full: list[dict[str, object]],
    candidates_dir: Path,
    evaluations_dir: Path,
    config,
) -> dict[str, object]:
    candidate_start = time.perf_counter()
    problem_name = str(manifest["problem"])
    candidate_dir = candidates_dir / plan.slug()
    evaluation_dir = evaluations_dir / plan.slug()

    # Idempotent resume: reuse a completed record if one was already written.
    saved_record = _read_json_if_exists(evaluation_dir / "agent_record.json")
    if isinstance(saved_record, dict):
        return saved_record

    timing: dict[str, float] = {}
    agent_start = time.perf_counter()
    run = _run_constrained_agent(
        plan=plan,
        manifest=manifest,
        train_summary=train_summary,
        train_public=train_instances_public,
        train_full=train_instances_full,
        candidate_dir=candidate_dir,
        config=config,
        problem_name=problem_name,
    )
    timing["agent_loop_wall_ms"] = (time.perf_counter() - agent_start) * 1000.0
    timing["agent_steps_used"] = float(run.steps_used)
    timing["agent_finished"] = 1.0 if run.finished else 0.0

    (candidate_dir / "agent_transcript.json").write_text(
        json.dumps(run.transcript, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (candidate_dir / "agent_metadata.json").write_text(
        json.dumps(
            {
                "slug": plan.slug(),
                "plan": _plan_payload(plan),
                "steps_used": run.steps_used,
                "finished": run.finished,
                "solver_built": run.solver_built,
                "api_error": run.api_error,
                "config": config.public_dict() if hasattr(config, "public_dict") else None,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    hypothesis = _read_json_if_exists(candidate_dir / "hypothesis.json")
    analyze_py = _read_text_if_exists(candidate_dir / "analyze.py")
    solution_py = _read_text_if_exists(candidate_dir / "solution.py")

    def _finalize(record: dict[str, object]) -> dict[str, object]:
        record["timing"] = {**timing, **record.get("timing", {})}
        record["timing"]["candidate_wall_ms"] = (time.perf_counter() - candidate_start) * 1000.0
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        (evaluation_dir / "agent_record.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        return record

    if run.api_error is not None and not analyze_py:
        return _finalize(
            _failure_record(
                plan=plan,
                candidate_dir=candidate_dir,
                evaluation_dir=evaluation_dir,
                hypothesis=hypothesis,
                analyze_py=analyze_py,
                solution_py=solution_py,
                error=f"Agent API error: {run.api_error}",
                train_count=len(train_instances_full),
                validation_count=len(validation_instances_full),
                timing=timing,
            )
        )

    if not analyze_py or not solution_py:
        missing = [name for name in ("analyze.py", "solution.py") if not (candidate_dir / name).exists()]
        return _finalize(
            _failure_record(
                plan=plan,
                candidate_dir=candidate_dir,
                evaluation_dir=evaluation_dir,
                hypothesis=hypothesis,
                analyze_py=analyze_py,
                solution_py=solution_py,
                error=f"Agent did not produce required files: missing {missing}.",
                train_count=len(train_instances_full),
                validation_count=len(validation_instances_full),
                timing=timing,
            )
        )

    # Final scoring: identical calls to the llm generator so results are comparable.
    try:
        analysis_output = run_analysis(
            candidate_dir, train_instances_public, manifest=manifest, artifact_dir=evaluation_dir
        )
    except Exception as exc:  # noqa: BLE001
        return _finalize(
            _failure_record(
                plan=plan,
                candidate_dir=candidate_dir,
                evaluation_dir=evaluation_dir,
                hypothesis=hypothesis,
                analyze_py=analyze_py,
                solution_py=solution_py,
                error=f"{type(exc).__name__}: {exc}",
                train_count=len(train_instances_full),
                validation_count=len(validation_instances_full),
                timing=timing,
            )
        )

    try:
        solver = build_solver(candidate_dir, analysis=analysis_output, manifest=manifest)
        train_eval = evaluate_solver(problem_name, plan.slug(), solver, train_instances_full, split="train")
    except Exception as exc:  # noqa: BLE001
        train_eval = failed_summary(plan.slug(), "train", len(train_instances_full), f"{type(exc).__name__}: {exc}")

    try:
        solver = build_solver(candidate_dir, analysis=analysis_output, manifest=manifest)
        validation_eval = evaluate_solver(
            problem_name, plan.slug(), solver, validation_instances_full, split="validation"
        )
    except Exception as exc:  # noqa: BLE001
        validation_eval = failed_summary(
            plan.slug(), "validation", len(validation_instances_full), f"{type(exc).__name__}: {exc}"
        )

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    write_summary(evaluation_dir / "train_summary.json", train_eval)
    write_summary(evaluation_dir / "validation_summary.json", validation_eval)

    return _finalize(
        {
            "slug": plan.slug(),
            "plan": _plan_payload(plan),
            "hypothesis": hypothesis if isinstance(hypothesis, dict) else None,
            "candidate_dir": str(candidate_dir),
            "evaluation_dir": str(evaluation_dir),
            "code_bundle": {"analyze_py": analyze_py, "solution_py": solution_py},
            "stage_notes": {"hypothesis": "", "analyze": "", "solution": ""},
            "analysis_output": analysis_output,
            "train": train_eval,
            "validation": validation_eval,
            "selection": summarize_selection(train_eval, validation_eval),
            "timing": timing,
        }
    )


# --------------------------------------------------------------------------- #
# Public entry point (mirrors llm.run_llm_synthesis_loop)
# --------------------------------------------------------------------------- #


def run_agent_synthesis_loop(
    dataset_dir: Path,
    output_dir: Path,
    *,
    mode: str = "single",
    iterations: int = 3,
    beam_width: int = 3,
    candidate_width: int | None = None,
    timing_reporter: BenchmarkTimingReporter | None = None,
) -> dict[str, object]:
    manifest = load_manifest(dataset_dir)
    problem_name = str(manifest["problem"])
    problem = get_problem_definition(problem_name)
    train_public = load_split(dataset_dir, "train", public=True)
    train_full = load_split(dataset_dir, "train")
    validation_full = load_split(dataset_dir, "validation")
    test_full = load_split(dataset_dir, "test")
    train_summary = (
        problem.summarize_training_data(train_public, manifest) if train_public else _empty_train_summary(manifest)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = output_dir / "candidates"
    evaluations_dir = output_dir / "evaluations"

    config = load_chat_api_config(required=True)

    from dasbench.agents.llm import _effective_candidate_width

    effective_candidate_width = _effective_candidate_width(mode, candidate_width, beam_width)
    frontier = _seed_plans(mode, candidate_width=effective_candidate_width, beam_width=beam_width)
    evaluated: dict[str, dict[str, object]] = {}
    rounds: list[dict[str, object]] = []
    history: list[dict[str, object]] = []

    for iteration in range(iterations):
        if not frontier:
            break
        current_round: list[dict[str, object]] = []
        for plan in frontier:
            if plan.slug() not in evaluated:
                evaluated[plan.slug()] = _evaluate_plan_via_agent(
                    plan,
                    manifest=manifest,
                    train_summary=train_summary,
                    train_instances_public=train_public,
                    train_instances_full=train_full,
                    validation_instances_full=validation_full,
                    candidates_dir=candidates_dir,
                    evaluations_dir=evaluations_dir,
                    config=config,
                )
                if timing_reporter is not None:
                    timing_reporter.record_synthesis_candidate(evaluated[plan.slug()])
            current_round.append(evaluated[plan.slug()])
        survivors = _select_survivors(list(evaluated.values()), mode=mode, beam_width=beam_width)
        history.append(progress_point(iteration, survivors[0]))
        rounds.append(
            {
                "iteration": iteration,
                "evaluated_this_round": [record["slug"] for record in current_round],
                "frontier_after_ranking": [record["slug"] for record in survivors],
                "frontier_diversity_keys": [_hypothesis_diversity_key(record) for record in survivors],
                "best_selected_slug": survivors[0]["slug"],
                "best_selected_hypothesis": survivors[0].get("hypothesis"),
                "best_selected_train": survivors[0]["train"],
                "best_selected_validation": survivors[0]["validation"],
                "best_selected_selection": survivors[0]["selection"],
            }
        )
        if timing_reporter is not None:
            timing_reporter.record_synthesis_round(rounds[-1])
        if iteration == iterations - 1:
            break
        frontier = _child_plans(
            iteration + 1,
            survivors,
            mode=mode,
            beam_width=beam_width,
            candidate_width=effective_candidate_width,
        )

    best_candidate = max(evaluated.values(), key=lambda record: selection_sort_key(record["selection"]))
    analysis = best_candidate.get("analysis_output")
    if analysis is not None:
        analysis_dir = output_dir / "best_candidate_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / "analysis.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
    _evaluate_best_candidate_test(
        problem_name=problem_name,
        best_candidate=best_candidate,
        test_full=test_full,
        analysis=analysis,
        manifest=manifest,
        timing_reporter=timing_reporter,
    )

    history_path = output_dir / "performance_history.json"
    write_history(history_path, history)
    plot_path = write_performance_plot(
        output_dir / performance_plot_filename(),
        history,
        title=f"{problem_name} agent search",
    )
    summary = {
        "problem": problem_name,
        "family": manifest["family"],
        "generator": "agent",
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "mode": mode,
        "iterations": iterations,
        "beam_width": beam_width,
        "candidate_width": effective_candidate_width,
        "ground_truth_hidden_rule": manifest.get("ground_truth_hidden_rule", {}),
        "best_candidate": best_candidate,
        "rounds": rounds,
        "train_summary": train_summary,
        "performance_history_path": str(history_path),
        "performance_plot_path": str(plot_path),
    }
    write_summary(output_dir / "synthesis_summary.json", summary)
    return summary
