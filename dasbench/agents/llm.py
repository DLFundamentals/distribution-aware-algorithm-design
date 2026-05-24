from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from openai import OpenAIError

from dasbench.agents.candidate import build_solver, run_analysis
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
from dasbench.integrations import create_chat_completion_raw, load_chat_api_config
from dasbench.problems import get_problem_definition
from dasbench.timing import BenchmarkTimingReporter
from dasbench.utils import candidate_manifest

MODULE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = MODULE_DIR.parent
SYSTEM_PROMPT_PATH = PACKAGE_ROOT / "prompts" / "llm_system_prompt.txt"
NO_HINT_SYSTEM_PROMPT_PATH = PACKAGE_ROOT / "prompts" / "llm_no_hint_system_prompt.txt"
HYPOTHESIS_RESPONSE_SCHEMA_PATH = PACKAGE_ROOT / "schemas" / "hypothesis_bundle.json"
ANALYZE_RESPONSE_SCHEMA_PATH = PACKAGE_ROOT / "schemas" / "analyze_code_bundle.json"
SOLUTION_RESPONSE_SCHEMA_PATH = PACKAGE_ROOT / "schemas" / "solution_code_bundle.json"
PROMPT_JSON_CHAR_LIMIT = 12_000
PROMPT_CODE_CHAR_LIMIT = 10_000
ANALYSIS_RETRY_ENV_VAR = "DASBENCH_ANALYSIS_RETRY_LIMIT"
DEFAULT_ANALYSIS_RETRY_LIMIT = 2


class GenerationDebugError(RuntimeError):
    def __init__(self, message: str, *, metadata: dict[str, object]) -> None:
        super().__init__(message)
        self.metadata = metadata


@dataclass(frozen=True)
class LLMPlan:
    iteration: int
    slot: int
    focus: str
    parent_slug: str | None = None
    hypothesis_directive: str = "Propose a concrete, testable hidden-rule hypothesis."

    def slug(self) -> str:
        return f"llm_iter{self.iteration:02d}_slot{self.slot:02d}"


@dataclass(frozen=True)
class LLMSolverOnlyPlan:
    iteration: int
    slot: int
    focus: str
    parent_slug: str | None = None
    solver_directive: str = "Write a fast, robust solver using training samples only as empirical tuning data."

    def slug(self) -> str:
        return f"llm_nohint_iter{self.iteration:02d}_slot{self.slot:02d}"


@lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def _load_no_hint_system_prompt() -> str:
    return NO_HINT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=None)
def _load_response_schema(path_text: str) -> dict[str, object]:
    return json.loads(Path(path_text).read_text(encoding="utf-8"))


def _safe_model_dump(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    return None


def _message_text_content(message: object) -> str | None:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            item_text = getattr(item, "text", None)
            if item_text:
                text_parts.append(item_text)
        return "\n".join(text_parts) or None
    return None


def _extract_json_object(text: str) -> dict[str, object]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Model response must decode to a JSON object.")
    return payload


def _read_json_value_if_exists(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_json_dict_if_exists(path: Path) -> dict[str, object] | None:
    payload = _read_json_value_if_exists(path)
    if isinstance(payload, dict):
        return payload
    return None


def _read_text_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _extract_stage_bundle(payload: dict[str, object], *, code_field: str) -> tuple[str, str]:
    code = payload.get(code_field)
    notes = payload.get("notes", "")
    if not isinstance(code, str):
        raise ValueError(f"Structured output field `{code_field}` must be a string.")
    if not isinstance(notes, str):
        raise ValueError("Structured output field `notes` must be a string.")
    return code, notes


def _clip_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"


def _prepare_json_for_prompt(value: object, *, max_chars: int = PROMPT_JSON_CHAR_LIMIT) -> object:
    serialized = json.dumps(value, indent=2, sort_keys=True)
    if len(serialized) <= max_chars:
        return value
    return {
        "truncated_json_text": _clip_text(serialized, max_chars=max_chars),
        "truncated": True,
        "original_char_count": len(serialized),
    }


def _prepare_text_for_prompt(text: str, *, max_chars: int = PROMPT_CODE_CHAR_LIMIT) -> str:
    return _clip_text(text, max_chars=max_chars)


def _compact_split_summary(summary: dict[str, object]) -> dict[str, object]:
    return {
        "average_normalized_quality": summary.get("average_normalized_quality"),
        "optimality_rate": summary.get("optimality_rate"),
        "feasibility_rate": summary.get("feasibility_rate"),
        "average_runtime_ms": summary.get("average_runtime_ms"),
        "failure_cases": summary.get("failure_cases", []),
        "error": summary.get("error"),
    }


def _plan_payload(plan: LLMPlan | LLMSolverOnlyPlan) -> dict[str, object]:
    payload = {
        "iteration": plan.iteration,
        "slot": plan.slot,
        "focus": plan.focus,
        "parent_slug": plan.parent_slug,
    }
    if isinstance(plan, LLMPlan):
        payload["search_style"] = "hint_recovery"
        payload["hypothesis_directive"] = plan.hypothesis_directive
    else:
        payload["search_style"] = "no_hint"
        payload["solver_directive"] = plan.solver_directive
    return payload


def _parent_context(parent_record: dict[str, object] | None) -> dict[str, object] | None:
    if parent_record is None:
        return None
    return {
        "parent_slug": parent_record["slug"],
        "parent_hypothesis": parent_record.get("hypothesis"),
        "parent_stage_notes": parent_record.get("stage_notes", {}),
        "parent_train": _compact_split_summary(parent_record["train"]),
        "parent_validation": _compact_split_summary(parent_record["validation"]),
        "parent_selection": parent_record.get("selection", {}),
        "parent_analyze_py": _prepare_text_for_prompt(parent_record["code_bundle"]["analyze_py"]),
        "parent_solution_py": _prepare_text_for_prompt(parent_record["code_bundle"]["solution_py"]),
        "parent_analysis_output": _prepare_json_for_prompt(parent_record.get("analysis_output")),
    }


def _solver_parent_context(parent_record: dict[str, object] | None) -> dict[str, object] | None:
    if parent_record is None:
        return None
    return {
        "parent_slug": parent_record["slug"],
        "parent_stage_notes": parent_record.get("stage_notes", {}),
        "parent_train": _compact_split_summary(parent_record["train"]),
        "parent_validation": _compact_split_summary(parent_record["validation"]),
        "parent_selection": parent_record.get("selection", {}),
        "parent_analyze_py": _prepare_text_for_prompt(parent_record["code_bundle"]["analyze_py"]),
        "parent_solution_py": _prepare_text_for_prompt(parent_record["code_bundle"]["solution_py"]),
        "parent_analysis_output": _prepare_json_for_prompt(parent_record.get("analysis_output")),
    }


def _prompt_train_summary(summary: dict[str, object]) -> dict[str, object]:
    sanitized = dict(summary)
    sanitized.pop("family", None)
    return sanitized


def _build_hypothesis_messages(
    *,
    manifest: dict[str, object],
    train_summary: dict[str, object],
    plan: LLMPlan,
    parent_record: dict[str, object] | None,
) -> list[dict[str, str]]:
    prompt_manifest = candidate_manifest(manifest)
    prompt_payload = {
        "stage": "hypothesis",
        "task": "Propose one concrete hidden-rule hypothesis for the unknown instance distribution.",
        "focus": plan.focus,
        "hypothesis_directive": plan.hypothesis_directive,
        "manifest": prompt_manifest,
        "train_summary": _prompt_train_summary(train_summary),
        "constraints": [
            "The exact family identity is hidden. Do not guess a named benchmark family.",
            "The hypothesis must be testable from candidate-facing training instances.",
            "Do not assume access to optimum labels in candidate-facing data.",
            "Prefer hypotheses about latent rules, motifs, communities, separators, overlaps, geometric regimes, or recurring role structure.",
            "The diversity_key should identify the broad hypothesis class, not the implementation details.",
        ],
        "iteration_context": _parent_context(parent_record),
        "response_format": {
            "title": "short hypothesis name",
            "rule_summary": "concrete explanation of the hidden distributional rule",
            "evidence_to_measure": "list of measurements analyze.py should compute",
            "solver_strategy": "how solution.py should exploit the rule",
            "expected_failure_modes": "list of ways the hypothesis could fail",
            "diversity_key": "stable lowercase key for preserving beam diversity",
            "notes": "brief rationale",
        },
    }
    return [
        {"role": "system", "content": _load_system_prompt()},
        {"role": "user", "content": json.dumps(prompt_payload, indent=2, sort_keys=True)},
    ]


def _effective_candidate_width(mode: str, candidate_width: int | None, beam_width: int) -> int:
    if mode == "single":
        return 1
    if candidate_width is None:
        return max(1, beam_width)
    return max(1, int(candidate_width))


def _seed_plans(mode: str, *, candidate_width: int | None = None, beam_width: int = 3) -> list[LLMPlan]:
    if mode == "single":
        return [
            LLMPlan(
                iteration=0,
                slot=0,
                focus="Start with a lightweight analysis-driven solver that tries to exploit recurrent structure without heavy online search.",
                hypothesis_directive=(
                    "Propose the single most plausible hidden-rule hypothesis from the problem type and training summary. "
                    "Make it concrete enough that analyze.py can test it."
                ),
            )
        ]
    templates = [
        (
            "Look for latent instance subtypes and write a fast solver that conditions on those subtypes.",
            "Propose a hidden-rule hypothesis centered on latent subtypes, regimes, motifs, communities, or geometric modes.",
        ),
        (
            "Prefer a very low-latency heuristic that uses compact training-set statistics.",
            "Propose a hidden-rule hypothesis centered on stable labels, reusable templates, persistent roles, or global summary statistics.",
        ),
        (
            "Try a hybrid approach with compact analysis output and a small amount of online repair.",
            "Propose a hidden-rule hypothesis centered on a structural rule plus bounded online repair for exceptions or noise.",
        ),
        (
            "Search for separator, bridge, bottleneck, or active-constraint roles that explain held-out structure.",
            "Propose a hidden-rule hypothesis centered on special structural roles such as separators, bridges, bottlenecks, gateways, or active constraints.",
        ),
        (
            "Look for interaction patterns that marginal statistics would miss.",
            "Propose a hidden-rule hypothesis centered on pairwise or higher-order interactions rather than single-feature heuristics.",
        ),
        (
            "Focus on decomposition: solve inferred components or blocks with different local rules.",
            "Propose a hidden-rule hypothesis centered on decomposing instances into communities, blocks, clusters, or motifs.",
        ),
        (
            "Try an objective-aware heuristic that estimates marginal gain or reduced cost from training samples.",
            "Propose a hidden-rule hypothesis centered on objective-aware marginal gain, density, reduced cost, or dual-price structure.",
        ),
        (
            "Prioritize robustness under noise and jitter in the distribution.",
            "Propose a hidden-rule hypothesis centered on stable structure that survives per-instance jitter and noisy exceptions.",
        ),
        (
            "Look for traps where the obvious greedy signal is anti-correlated with optimal decisions.",
            "Propose a hidden-rule hypothesis centered on decoys, traps, or misleading first-order statistics.",
        ),
        (
            "Try a deliberately simple distribution-specific shortcut, then bound its repair cost.",
            "Propose a hidden-rule hypothesis centered on a compact shortcut plus a minimal feasibility or quality repair step.",
        ),
    ]
    width = _effective_candidate_width(mode, candidate_width, beam_width)
    plans = []
    for slot in range(width):
        focus, directive = templates[slot % len(templates)]
        if slot >= len(templates):
            focus += f" Variant {slot // len(templates) + 1}: use a distinct evidence test and diversity_key."
            directive += " Make this variant meaningfully different from earlier slots."
        plans.append(
            LLMPlan(
                iteration=0,
                slot=slot,
                focus=focus,
                hypothesis_directive=directive,
            )
        )
    return plans


def _child_plans(
    iteration: int,
    survivors: list[dict[str, object]],
    *,
    mode: str,
    beam_width: int,
    candidate_width: int | None = None,
) -> list[LLMPlan]:
    plans: list[LLMPlan] = []
    slot = 0
    width = _effective_candidate_width(mode, candidate_width, beam_width)
    actions = [
        (
            "Refine the parent hidden-rule hypothesis using validation feedback. Keep the same broad diversity_key unless the evidence clearly contradicts it.",
            "Revise the hypothesis, analysis, and solver together. Keep only evidence that supports held-out generalization.",
        ),
        (
            "Fork the parent into a meaningfully different hidden-rule hypothesis that explains the data another way and should use a different diversity_key.",
            "Try a different structural explanation while reusing any useful implementation lessons from the parent.",
        ),
        (
            "Replace the parent hypothesis if validation feedback suggests it was solving by generic search or an accidental shortcut.",
            "Prioritize a more distribution-specific hypothesis over implementation polishing.",
        ),
        (
            "Create a more aggressive runtime-focused variant of the parent hypothesis while preserving feasibility and quality checks.",
            "Keep the inferred structure but simplify or precompute more so inference is faster.",
        ),
        (
            "Create a more quality-focused variant of the parent hypothesis with extra repair or fallback logic.",
            "Spend a little more inference time to improve solution quality on cases where the parent failed.",
        ),
    ]
    variant_round = 0
    while len(plans) < width:
        added_this_round = False
        for hypothesis_directive, default_focus in actions:
            for survivor in survivors:
                validation_quality = float(survivor["validation"]["average_normalized_quality"])
                validation_runtime = float(survivor["validation"]["average_runtime_ms"])
                gap = float(survivor["train"]["average_normalized_quality"]) - validation_quality
                focus = default_focus
                directive = hypothesis_directive
                if variant_round > 0:
                    focus += f" Variant round {variant_round + 1}: change the evidence test and diversity_key."
                    directive += " This variant must differ from earlier children for the same parent."
                if validation_quality < 0.99:
                    focus += " Improve normalized quality by identifying stronger distributional structure."
                if gap > 0.02:
                    focus += " Train quality is above validation, so prefer robust signals over memorized templates."
                if validation_runtime > 20.0:
                    focus += " Keep the useful structure but simplify online inference to reduce runtime."
                plans.append(
                    LLMPlan(
                        iteration=iteration,
                        slot=slot,
                        focus=focus,
                        parent_slug=survivor["slug"],
                        hypothesis_directive=directive,
                    )
                )
                slot += 1
                added_this_round = True
                if mode == "single":
                    return plans[:1]
                if len(plans) >= width:
                    return plans[:width]
        if not added_this_round:
            break
        variant_round += 1
    return plans[:width]


def _hypothesis_diversity_key(record: dict[str, object]) -> str:
    hypothesis = record.get("hypothesis")
    if isinstance(hypothesis, dict):
        diversity_key = str(hypothesis.get("diversity_key", "")).strip().lower()
        if diversity_key:
            return diversity_key
    return str(record["slug"])


def _select_survivors(
    records: list[dict[str, object]],
    *,
    mode: str,
    beam_width: int,
) -> list[dict[str, object]]:
    ranked = sorted(records, key=lambda record: selection_sort_key(record["selection"]), reverse=True)
    if mode == "single":
        return ranked[:1]
    selected: list[dict[str, object]] = []
    selected_slugs: set[str] = set()
    seen_keys: set[str] = set()
    for record in ranked:
        diversity_key = _hypothesis_diversity_key(record)
        if diversity_key in seen_keys:
            continue
        selected.append(record)
        selected_slugs.add(str(record["slug"]))
        seen_keys.add(diversity_key)
        if len(selected) >= beam_width:
            return selected
    for record in ranked:
        slug = str(record["slug"])
        if slug in selected_slugs:
            continue
        selected.append(record)
        selected_slugs.add(slug)
        if len(selected) >= beam_width:
            break
    return selected


def _seed_solver_only_plans(
    mode: str,
    *,
    candidate_width: int | None = None,
    beam_width: int = 3,
) -> list[LLMSolverOnlyPlan]:
    if mode == "single":
        return [
            LLMSolverOnlyPlan(
                iteration=0,
                slot=0,
                focus=(
                    "Start with a low-latency solver that uses training data only to tune parameters, "
                    "thresholds, or templates. Do not assume there is a single hidden rule to recover."
                ),
                solver_directive=(
                    "Write one strong generic solver that may use training-set statistics or exemplars, "
                    "but should not rely on a hidden-family explanation."
                ),
            )
        ]
    templates = [
        (
            "Try a runtime-first generic heuristic with lightweight training-set parameter tuning.",
            "Favor a simple greedy or constructive solver whose constants or scoring weights are tuned from training data.",
        ),
        (
            "Try a solver that builds a fast candidate ranking from training examples and then uses bounded repair.",
            "Favor a candidate-scoring heuristic plus a small local repair or cleanup step.",
        ),
        (
            "Try a sample-informed template or prototype solver if repeated structure appears in the training data.",
            "Use training examples to build reusable templates, motifs, exemplars, or cached orderings when helpful.",
        ),
        (
            "Try a decomposition-oriented solver that conditionally switches among a few generic subroutines.",
            "Use training data only to choose between generic solver behaviors, not to justify a hidden-rule story.",
        ),
        (
            "Try a quality-focused solver with a stronger fallback if the fast path is uncertain.",
            "Use analysis output to decide when to run the fast path and when to use a safer generic fallback.",
        ),
        (
            "Try a preprocessing-heavy solver that shrinks the online search space before a generic solve step.",
            "Use training data to tune preprocessing, pruning, ordering, or shortlist selection.",
        ),
        (
            "Try a local-search solver whose initialization and stopping rules are tuned from the training data.",
            "Favor a generic local-search or iterative-improvement solver with sample-informed initialization.",
        ),
        (
            "Try an instance-adaptive solver that routes between a few simple generic strategies.",
            "Use training summaries to define a small solver portfolio and a cheap routing rule per instance.",
        ),
        (
            "Try a conservative robust solver that avoids overfitting and minimizes pathological runtime spikes.",
            "Favor stability and broad generalization over aggressive specialization.",
        ),
        (
            "Try a solver that uses training data to learn a compact scoring rule but keeps the online algorithm generic.",
            "Use training examples as empirical tuning data only: thresholds, weights, candidate filters, or orderings.",
        ),
    ]
    width = _effective_candidate_width(mode, candidate_width, beam_width)
    plans: list[LLMSolverOnlyPlan] = []
    for slot in range(width):
        focus, directive = templates[slot % len(templates)]
        if slot >= len(templates):
            focus += f" Variant {slot // len(templates) + 1}: change the solver family or fallback behavior."
            directive += " Make this variant meaningfully different from earlier slots."
        plans.append(
            LLMSolverOnlyPlan(
                iteration=0,
                slot=slot,
                focus=focus,
                solver_directive=directive,
            )
        )
    return plans


def _child_solver_only_plans(
    iteration: int,
    survivors: list[dict[str, object]],
    *,
    mode: str,
    beam_width: int,
    candidate_width: int | None = None,
) -> list[LLMSolverOnlyPlan]:
    plans: list[LLMSolverOnlyPlan] = []
    slot = 0
    width = _effective_candidate_width(mode, candidate_width, beam_width)
    actions = [
        (
            "Refine the parent solver for lower runtime while preserving its validation quality.",
            "Keep the core algorithm but simplify online work, cache more, or reduce repair cost.",
        ),
        (
            "Refine the parent solver for higher validation quality even if it needs a slightly stronger fallback.",
            "Add safer fallback or bounded repair where the parent is fragile.",
        ),
        (
            "Fork the parent into a materially different generic solver family.",
            "Change the solver archetype rather than polishing the same one.",
        ),
        (
            "Use the training data more effectively as empirical tuning data, not as evidence for a hidden rule.",
            "Retune thresholds, weights, orderings, or portfolio routing from the training set.",
        ),
        (
            "Reduce overfitting: prefer solver behavior that should generalize beyond the seen samples.",
            "Keep only the sample-informed components that plausibly improve held-out validation.",
        ),
    ]
    variant_round = 0
    while len(plans) < width:
        added_this_round = False
        for solver_directive, default_focus in actions:
            for survivor in survivors:
                validation_quality = float(survivor["validation"]["average_normalized_quality"])
                validation_runtime = float(survivor["validation"]["average_runtime_ms"])
                gap = float(survivor["train"]["average_normalized_quality"]) - validation_quality
                focus = default_focus
                directive = solver_directive
                if variant_round > 0:
                    focus += f" Variant round {variant_round + 1}: make the solver family more distinct."
                    directive += " This child should differ meaningfully from earlier children of the same parent."
                if validation_quality < 0.99:
                    focus += " Improve solution quality on held-out instances."
                if gap > 0.02:
                    focus += " Validation lags train, so remove brittle sample-specific behavior."
                if validation_runtime > 20.0:
                    focus += " Simplify inference and reduce online search work."
                plans.append(
                    LLMSolverOnlyPlan(
                        iteration=iteration,
                        slot=slot,
                        focus=focus,
                        parent_slug=survivor["slug"],
                        solver_directive=directive,
                    )
                )
                slot += 1
                added_this_round = True
                if mode == "single":
                    return plans[:1]
                if len(plans) >= width:
                    return plans[:width]
        if not added_this_round:
            break
        variant_round += 1
    return plans[:width]


def _select_top_survivors(
    records: list[dict[str, object]],
    *,
    mode: str,
    beam_width: int,
) -> list[dict[str, object]]:
    ranked = sorted(records, key=lambda record: selection_sort_key(record["selection"]), reverse=True)
    return ranked[:1] if mode == "single" else ranked[:beam_width]


def _build_analyze_messages(
    *,
    manifest: dict[str, object],
    train_summary: dict[str, object],
    plan: LLMPlan,
    parent_record: dict[str, object] | None,
    hypothesis: dict[str, object] | None = None,
    execution_feedback: dict[str, object] | None = None,
) -> list[dict[str, str]]:
    prompt_manifest = candidate_manifest(manifest)
    prompt_payload = {
        "stage": "analyze",
        "task": "Write full replacement contents for analyze.py only. The analysis should test and parameterize the current hidden-rule hypothesis.",
        "focus": plan.focus,
        "current_hypothesis": hypothesis,
        "previous_execution_feedback": execution_feedback,
        "manifest": prompt_manifest,
        "train_summary": _prompt_train_summary(train_summary),
        "interfaces": {
            "analyze.py": "define analyze(train_instances, manifest=None) -> dict",
        },
        "constraints": [
            "Return runnable Python code only in the schema field analyze_py.",
            "Use only the Python standard library plus already-available dasbench modules.",
            "The analysis output must stay compact and JSON-serializable.",
            "Do not assume access to optimum labels in the candidate-facing dataset.",
            "Handle train_instances=[] gracefully. If there are no training instances, use manifest metadata only and return a compact fallback analysis instead of failing.",
            "The exact family identity is hidden. Infer exploitable structure from the samples and summaries.",
            "Measure evidence relevant to current_hypothesis and include enough parameters for solution.py to exploit it.",
            "Candidate analysis runs under a hard wall-clock timeout; avoid quadratic or exponential scans over training instances.",
        ],
        "iteration_context": _parent_context(parent_record),
        "response_format": {
            "analyze_py": "full contents of analyze.py as a string",
            "notes": "brief explanation of what the analysis computes and why",
        },
    }
    return [
        {"role": "system", "content": _load_system_prompt()},
        {"role": "user", "content": json.dumps(prompt_payload, indent=2, sort_keys=True)},
    ]


def _analysis_retry_limit() -> int:
    raw_value = os.environ.get(ANALYSIS_RETRY_ENV_VAR)
    if raw_value is None or not raw_value.strip():
        return DEFAULT_ANALYSIS_RETRY_LIMIT
    try:
        return max(0, int(raw_value))
    except ValueError as exc:
        raise ValueError(f"{ANALYSIS_RETRY_ENV_VAR} must be a nonnegative integer, got {raw_value!r}.") from exc


def _archive_failed_analysis_attempt(
    candidate_dir: Path,
    *,
    attempt: int,
    analyze_py: str | None,
    exc: Exception,
) -> None:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{attempt:02d}"
    if analyze_py is not None:
        (candidate_dir / f"analyze_failed_attempt_{suffix}.py").write_text(analyze_py, encoding="utf-8")
    (candidate_dir / f"analyze_failed_attempt_{suffix}.txt").write_text(
        f"{type(exc).__name__}: {exc}\n",
        encoding="utf-8",
    )


def _build_solution_messages(
    *,
    manifest: dict[str, object],
    train_summary: dict[str, object],
    plan: LLMPlan,
    parent_record: dict[str, object] | None,
    analyze_py: str,
    analysis_output: object,
    hypothesis: dict[str, object] | None = None,
) -> list[dict[str, str]]:
    prompt_manifest = candidate_manifest(manifest)
    prompt_payload = {
        "stage": "solution",
        "task": "Write full replacement contents for solution.py only. The solver should exploit the current hidden-rule hypothesis when the analysis supports it.",
        "focus": plan.focus,
        "current_hypothesis": hypothesis,
        "manifest": prompt_manifest,
        "train_summary_overview": {
            "problem": train_summary.get("problem"),
            "num_instances": train_summary.get("num_instances"),
        },
        "current_analyze_py": _prepare_text_for_prompt(analyze_py),
        "current_analysis_output": _prepare_json_for_prompt(analysis_output),
        "interfaces": {
            "solution.py": "define solve(instance, analysis=None, manifest=None) -> object",
        },
        "constraints": [
            "Return runnable Python code only in the schema field solution_py.",
            "Use the analysis output instead of redoing expensive training-time work online.",
            "Keep per-instance runtime low.",
            "The solver will be run unchanged on validation and test instances.",
            "analysis may come from an empty-train fallback summary, so include a robust path that works even when training evidence is minimal.",
            "The exact family identity is hidden. Treat the structure as unknown and infer it from the observed samples.",
            "If analysis contradicts the hypothesis, include a robust fallback that still respects runtime.",
        ],
        "iteration_context": _parent_context(parent_record),
        "response_format": {
            "solution_py": "full contents of solution.py as a string",
            "notes": "brief explanation of how the solver uses the analysis output",
        },
    }
    return [
        {"role": "system", "content": _load_system_prompt()},
        {"role": "user", "content": json.dumps(prompt_payload, indent=2, sort_keys=True)},
    ]


def _build_solver_only_analyze_messages(
    *,
    manifest: dict[str, object],
    train_summary: dict[str, object],
    plan: LLMSolverOnlyPlan,
    parent_record: dict[str, object] | None,
) -> list[dict[str, str]]:
    prompt_manifest = candidate_manifest(manifest)
    prompt_payload = {
        "stage": "analyze",
        "task": (
            "Write full replacement contents for analyze.py only. "
            "The analysis should summarize training examples into compact signals that help choose or tune a solver."
        ),
        "focus": plan.focus,
        "solver_directive": plan.solver_directive,
        "manifest": prompt_manifest,
        "train_summary": _prompt_train_summary(train_summary),
        "interfaces": {
            "analyze.py": "define analyze(train_instances, manifest=None) -> dict",
        },
        "constraints": [
            "Return runnable Python code only in the schema field analyze_py.",
            "Use only the Python standard library plus already-available dasbench modules.",
            "The analysis output must stay compact and JSON-serializable.",
            "Handle train_instances=[] gracefully. If there are no training instances, use manifest metadata only and return a compact fallback analysis instead of failing.",
            "Do not assume there is a single hidden rule to recover.",
            "Do not guess a named benchmark family.",
            "Use training samples only as empirical tuning data: summarize templates, thresholds, weights, motifs, candidate orderings, routing rules, or other reusable signals.",
            "Prefer compact aggregate statistics, exemplars, and tunable parameters over raw dumps.",
        ],
        "iteration_context": _solver_parent_context(parent_record),
        "response_format": {
            "analyze_py": "full contents of analyze.py as a string",
            "notes": "brief explanation of what the analysis computes and how it will help the solver",
        },
    }
    return [
        {"role": "system", "content": _load_no_hint_system_prompt()},
        {"role": "user", "content": json.dumps(prompt_payload, indent=2, sort_keys=True)},
    ]


def _build_solver_only_solution_messages(
    *,
    manifest: dict[str, object],
    train_summary: dict[str, object],
    plan: LLMSolverOnlyPlan,
    parent_record: dict[str, object] | None,
    analyze_py: str,
    analysis_output: object,
) -> list[dict[str, str]]:
    prompt_manifest = candidate_manifest(manifest)
    prompt_payload = {
        "stage": "solution",
        "task": (
            "Write full replacement contents for solution.py only. "
            "Use analysis as optional tuning data for a generic solver; do not assume a hidden family rule."
        ),
        "focus": plan.focus,
        "solver_directive": plan.solver_directive,
        "manifest": prompt_manifest,
        "train_summary_overview": {
            "problem": train_summary.get("problem"),
            "num_instances": train_summary.get("num_instances"),
        },
        "current_analyze_py": _prepare_text_for_prompt(analyze_py),
        "current_analysis_output": _prepare_json_for_prompt(analysis_output),
        "interfaces": {
            "solution.py": "define solve(instance, analysis=None, manifest=None) -> object",
        },
        "constraints": [
            "Return runnable Python code only in the schema field solution_py.",
            "Use the analysis output instead of redoing expensive training-time work online.",
            "Keep per-instance runtime low.",
            "The solver will be run unchanged on validation and test instances.",
            "analysis may come from an empty-train fallback summary, so include a robust path that works even when training evidence is minimal.",
            "Do not assume there is a single hidden rule to recover.",
            "Do not guess a named benchmark family.",
            "Favor generic algorithms, training-tuned parameters, shortlists, orderings, prototypes, and bounded repair over hidden-rule storytelling.",
            "If the analysis signal is weak, fall back to a solid generic solver rather than brittle specialization.",
        ],
        "iteration_context": _solver_parent_context(parent_record),
        "response_format": {
            "solution_py": "full contents of solution.py as a string",
            "notes": "brief explanation of how the solver uses the analysis output",
        },
    }
    return [
        {"role": "system", "content": _load_no_hint_system_prompt()},
        {"role": "user", "content": json.dumps(prompt_payload, indent=2, sort_keys=True)},
    ]


def _stage_metadata_path(candidate_dir: Path, stage_name: str) -> Path:
    return candidate_dir / f"{stage_name}_generation_metadata.json"


def _hypothesis_path(candidate_dir: Path) -> Path:
    return candidate_dir / "hypothesis.json"


def _analysis_path(evaluation_dir: Path) -> Path:
    return evaluation_dir / "analysis.json"


def _summary_path(evaluation_dir: Path, split: str) -> Path:
    return evaluation_dir / f"{split}_summary.json"


def _load_saved_llm_record(
    candidate_dir: Path,
    evaluation_dir: Path,
    *,
    saved_timing_record: dict[str, object] | None = None,
) -> dict[str, object] | None:
    train_summary = _read_json_dict_if_exists(_summary_path(evaluation_dir, "train"))
    validation_summary = _read_json_dict_if_exists(_summary_path(evaluation_dir, "validation"))
    if train_summary is None or validation_summary is None:
        return None

    hypothesis_path = _hypothesis_path(candidate_dir)
    hypothesis_metadata_path = _stage_metadata_path(candidate_dir, "hypothesis")
    hypothesis = (
        _read_json_dict_if_exists(hypothesis_path)
        if hypothesis_path.exists() and hypothesis_metadata_path.exists()
        else None
    )
    if hypothesis is None and isinstance(saved_timing_record, dict):
        title = saved_timing_record.get("hypothesis_title")
        diversity_key = saved_timing_record.get("hypothesis_diversity_key")
        if title is not None or diversity_key is not None:
            hypothesis = {
                "title": title,
                "diversity_key": diversity_key,
            }

    selection = summarize_selection(train_summary, validation_summary)
    slug = candidate_dir.name
    stage_notes = {
        "hypothesis": (_read_text_if_exists(candidate_dir / "hypothesis_notes.txt") or "").strip(),
        "analyze": (_read_text_if_exists(candidate_dir / "analyze_notes.txt") or "").strip(),
        "solution": (_read_text_if_exists(candidate_dir / "solution_notes.txt") or "").strip(),
    }
    analysis_output = _read_json_value_if_exists(_analysis_path(evaluation_dir))
    timing = {}
    plan = None
    spec = None
    if isinstance(saved_timing_record, dict):
        saved_timing = saved_timing_record.get("timing")
        if isinstance(saved_timing, dict):
            timing = dict(saved_timing)
        saved_plan = saved_timing_record.get("plan")
        if isinstance(saved_plan, dict):
            plan = dict(saved_plan)
        saved_spec = saved_timing_record.get("spec")
        if isinstance(saved_spec, dict):
            spec = dict(saved_spec)

    return {
        "slug": slug,
        "plan": plan,
        "spec": spec,
        "hypothesis": hypothesis,
        "candidate_dir": str(candidate_dir),
        "evaluation_dir": str(evaluation_dir),
        "code_bundle": {
            "analyze_py": _read_text_if_exists(candidate_dir / "analyze.py") or "",
            "solution_py": _read_text_if_exists(candidate_dir / "solution.py") or "",
        },
        "stage_notes": stage_notes,
        "analysis_output": analysis_output,
        "train": train_summary,
        "validation": validation_summary,
        "selection": selection,
        "timing": timing,
    }


def _load_saved_timing_candidates(output_dir: Path) -> dict[str, dict[str, object]]:
    payload = _read_json_dict_if_exists(output_dir / "timing_report.json")
    if payload is None:
        return {}
    stages = payload.get("stages", {})
    if not isinstance(stages, dict):
        return {}
    synthesis = stages.get("synthesis", {})
    if not isinstance(synthesis, dict):
        return {}
    candidates = synthesis.get("candidates", {})
    if not isinstance(candidates, dict):
        return {}
    return {
        str(slug): dict(record)
        for slug, record in candidates.items()
        if isinstance(record, dict)
    }


def _empty_train_summary(manifest: dict[str, object]) -> dict[str, object]:
    return {
        "problem": manifest["problem"],
        "family": manifest["family"],
        "num_instances": 0,
        "instance_params": manifest.get("instance_params", {}),
        "family_params": manifest.get("family_params", {}),
        "sample_instances": [],
        "notes": "No training instances were provided for this run.",
    }


def _load_saved_llm_records(
    candidates_dir: Path,
    evaluations_dir: Path,
    *,
    output_dir: Path,
) -> dict[str, dict[str, object]]:
    saved_timing_candidates = _load_saved_timing_candidates(output_dir)
    records: dict[str, dict[str, object]] = {}
    if not candidates_dir.exists() or not evaluations_dir.exists():
        return records
    for candidate_dir in sorted(path for path in candidates_dir.iterdir() if path.is_dir()):
        evaluation_dir = evaluations_dir / candidate_dir.name
        record = _load_saved_llm_record(
            candidate_dir,
            evaluation_dir,
            saved_timing_record=saved_timing_candidates.get(candidate_dir.name),
        )
        if record is not None:
            records[candidate_dir.name] = record
    return records


def _normalize_hypothesis_payload(payload: dict[str, object]) -> tuple[dict[str, object], str]:
    required_string_fields = ("title", "rule_summary", "solver_strategy", "diversity_key", "notes")
    required_list_fields = ("evidence_to_measure", "expected_failure_modes")
    for field in required_string_fields:
        if not isinstance(payload.get(field), str):
            raise ValueError(f"Hypothesis field `{field}` must be a string.")
    for field in required_list_fields:
        value = payload.get(field)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"Hypothesis field `{field}` must be a list of strings.")
    hypothesis = {
        "title": str(payload["title"]).strip(),
        "rule_summary": str(payload["rule_summary"]).strip(),
        "evidence_to_measure": [item.strip() for item in payload["evidence_to_measure"]],
        "solver_strategy": str(payload["solver_strategy"]).strip(),
        "expected_failure_modes": [item.strip() for item in payload["expected_failure_modes"]],
        "diversity_key": str(payload["diversity_key"]).strip().lower().replace(" ", "_"),
    }
    notes = str(payload["notes"]).strip()
    return hypothesis, notes


def _write_stage_artifacts(
    candidate_dir: Path,
    *,
    stage_name: str,
    filename: str,
    code: str,
    notes: str,
    metadata: dict[str, object],
) -> None:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / filename).write_text(code, encoding="utf-8")
    (candidate_dir / f"{stage_name}_notes.txt").write_text(notes.strip() + "\n", encoding="utf-8")
    _stage_metadata_path(candidate_dir, stage_name).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_hypothesis_artifacts(
    candidate_dir: Path,
    *,
    hypothesis: dict[str, object],
    notes: str,
    metadata: dict[str, object],
) -> None:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    _hypothesis_path(candidate_dir).write_text(
        json.dumps(hypothesis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (candidate_dir / "hypothesis_notes.txt").write_text(notes.strip() + "\n", encoding="utf-8")
    _stage_metadata_path(candidate_dir, "hypothesis").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_stage_failure_metadata(
    candidate_dir: Path,
    *,
    stage_name: str,
    plan: LLMPlan | LLMSolverOnlyPlan,
    messages: list[dict[str, str]],
    exc: Exception,
) -> None:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "plan": _plan_payload(plan),
        "stage": stage_name,
        "generation_error": f"{type(exc).__name__}: {exc}",
        "request_messages": messages,
    }
    if isinstance(exc, GenerationDebugError):
        metadata.update(exc.metadata)
    _stage_metadata_path(candidate_dir, stage_name).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _generate_response_payload(
    *,
    messages: list[dict[str, str]],
    candidate_dir: Path,
    stage_name: str,
    response_schema_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    config = load_chat_api_config(required=True)
    response_schema = _load_response_schema(str(response_schema_path))
    metadata: dict[str, object] = {
        "api_config": config.public_dict(),
        "candidate_dir": str(candidate_dir),
        "stage": stage_name,
        "request_messages": messages,
        "system_prompt_path": str(SYSTEM_PROMPT_PATH),
        "response_schema_path": str(response_schema_path),
    }
    try:
        raw_response = create_chat_completion_raw(
            config,
            messages=messages,
            response_format=response_schema,
        )
    except OpenAIError as exc:
        raise GenerationDebugError(
            f"OpenAI API request failed for {candidate_dir.name} during {stage_name}: {exc}",
            metadata=metadata,
        ) from exc
    metadata["status_code"] = raw_response.status_code
    metadata["raw_http_text"] = raw_response.text[:20_000]
    if raw_response.status_code != 200:
        raise GenerationDebugError(
            f"OpenAI API returned status {raw_response.status_code} during {stage_name}: {raw_response.text[:1000]}",
            metadata=metadata,
        )
    try:
        completion = raw_response.parse()
        metadata["parsed_completion"] = _safe_model_dump(completion)
        choice = completion.choices[0]
        message = getattr(choice, "message", None)
        refusal = None if message is None else getattr(message, "refusal", None)
        metadata["refusal"] = refusal
        if refusal:
            raise ValueError(f"Model refused the structured request: {refusal}")
        response_text = None if message is None else _message_text_content(message)
        if not response_text:
            raise ValueError("Structured Outputs returned empty assistant content.")
        payload = _extract_json_object(response_text)
        metadata["parsed_payload"] = payload
    except Exception as exc:
        raise GenerationDebugError(
            f"Could not decode structured model output for {candidate_dir.name} during {stage_name}: {exc}",
            metadata=metadata,
        ) from exc
    return payload, metadata


def _generate_hypothesis_output(
    *,
    messages: list[dict[str, str]],
    candidate_dir: Path,
) -> tuple[dict[str, object], str, dict[str, object]]:
    payload, metadata = _generate_response_payload(
        messages=messages,
        candidate_dir=candidate_dir,
        stage_name="hypothesis",
        response_schema_path=HYPOTHESIS_RESPONSE_SCHEMA_PATH,
    )
    try:
        hypothesis, notes = _normalize_hypothesis_payload(payload)
    except Exception as exc:
        raise GenerationDebugError(
            f"Could not decode hypothesis output for {candidate_dir.name}: {exc}",
            metadata=metadata,
        ) from exc
    return hypothesis, notes, metadata


def _generate_stage_output(
    *,
    messages: list[dict[str, str]],
    candidate_dir: Path,
    stage_name: str,
    response_schema_path: Path,
    code_field: str,
) -> tuple[str, str, dict[str, object]]:
    payload, metadata = _generate_response_payload(
        messages=messages,
        candidate_dir=candidate_dir,
        stage_name=stage_name,
        response_schema_path=response_schema_path,
    )
    try:
        code, notes = _extract_stage_bundle(payload, code_field=code_field)
    except Exception as exc:
        raise GenerationDebugError(
            f"Could not decode structured model output for {candidate_dir.name} during {stage_name}: {exc}",
            metadata=metadata,
        ) from exc
    return code, notes, metadata


def _evaluate_plan(
    plan: LLMPlan,
    *,
    manifest: dict[str, object],
    train_summary: dict[str, object],
    train_instances_public: list[dict[str, object]],
    train_instances_full: list[dict[str, object]],
    validation_instances_full: list[dict[str, object]],
    candidates_dir: Path,
    evaluations_dir: Path,
    evaluated_records: dict[str, dict[str, object]],
) -> dict[str, object]:
    candidate_start = time.perf_counter()
    problem_name = str(manifest["problem"])
    candidate_dir = candidates_dir / plan.slug()
    evaluation_dir = evaluations_dir / plan.slug()
    saved_record = _load_saved_llm_record(candidate_dir, evaluation_dir)
    if saved_record is not None:
        return saved_record

    timing: dict[str, float] = {}
    parent_record = evaluated_records.get(plan.parent_slug) if plan.parent_slug else None
    plan_payload = _plan_payload(plan)

    hypothesis = _read_json_dict_if_exists(_hypothesis_path(candidate_dir))
    hypothesis_notes = (_read_text_if_exists(candidate_dir / "hypothesis_notes.txt") or "").strip()
    if hypothesis is None:
        hypothesis_messages = _build_hypothesis_messages(
            manifest=manifest,
            train_summary=train_summary,
            plan=plan,
            parent_record=parent_record,
        )
        try:
            hypothesis_start = time.perf_counter()
            hypothesis, hypothesis_notes, hypothesis_metadata = _generate_hypothesis_output(
                messages=hypothesis_messages,
                candidate_dir=candidate_dir,
            )
            _write_hypothesis_artifacts(
                candidate_dir,
                hypothesis=hypothesis,
                notes=hypothesis_notes,
                metadata=hypothesis_metadata,
            )
            timing["hypothesis_generation_wall_ms"] = (time.perf_counter() - hypothesis_start) * 1000.0
        except Exception as exc:
            _write_stage_failure_metadata(
                candidate_dir,
                stage_name="hypothesis",
                plan=plan,
                messages=hypothesis_messages,
                exc=exc,
            )
            train_eval = failed_summary(plan.slug(), "train", len(train_instances_full), f"{type(exc).__name__}: {exc}")
            validation_eval = failed_summary(plan.slug(), "validation", len(validation_instances_full), f"{type(exc).__name__}: {exc}")
            selection = summarize_selection(train_eval, validation_eval)
            return {
                "slug": plan.slug(),
                "plan": plan_payload,
                "hypothesis": None,
                "candidate_dir": str(candidate_dir),
                "evaluation_dir": str(evaluation_dir),
                "code_bundle": {"analyze_py": "", "solution_py": ""},
                "stage_notes": {"hypothesis": "", "analyze": "", "solution": ""},
                "analysis_output": None,
                "train": train_eval,
                "validation": validation_eval,
                "selection": selection,
                "timing": {
                    **timing,
                    "candidate_wall_ms": (time.perf_counter() - candidate_start) * 1000.0,
                },
            }

    analyze_path = candidate_dir / "analyze.py"
    analyze_metadata_path = _stage_metadata_path(candidate_dir, "analyze")
    analyze_py = (
        _read_text_if_exists(analyze_path)
        if analyze_path.exists() and analyze_metadata_path.exists()
        else None
    )
    analyze_notes = (_read_text_if_exists(candidate_dir / "analyze_notes.txt") or "").strip()
    if analyze_py is None:
        analyze_messages = _build_analyze_messages(
            manifest=manifest,
            train_summary=train_summary,
            plan=plan,
            parent_record=parent_record,
            hypothesis=hypothesis,
        )
        try:
            analyze_generation_start = time.perf_counter()
            analyze_py, analyze_notes, analyze_metadata = _generate_stage_output(
                messages=analyze_messages,
                candidate_dir=candidate_dir,
                stage_name="analyze",
                response_schema_path=ANALYZE_RESPONSE_SCHEMA_PATH,
                code_field="analyze_py",
            )
            _write_stage_artifacts(
                candidate_dir,
                stage_name="analyze",
                filename="analyze.py",
                code=analyze_py,
                notes=analyze_notes,
                metadata=analyze_metadata,
            )
            timing["analyze_generation_wall_ms"] = (time.perf_counter() - analyze_generation_start) * 1000.0
        except Exception as exc:
            _write_stage_failure_metadata(
                candidate_dir,
                stage_name="analyze",
                plan=plan,
                messages=analyze_messages,
                exc=exc,
            )
            train_eval = failed_summary(plan.slug(), "train", len(train_instances_full), f"{type(exc).__name__}: {exc}")
            validation_eval = failed_summary(plan.slug(), "validation", len(validation_instances_full), f"{type(exc).__name__}: {exc}")
            selection = summarize_selection(train_eval, validation_eval)
            return {
                "slug": plan.slug(),
                "plan": plan_payload,
                "hypothesis": hypothesis,
                "candidate_dir": str(candidate_dir),
                "evaluation_dir": str(evaluation_dir),
                "code_bundle": {"analyze_py": "", "solution_py": ""},
                "stage_notes": {"hypothesis": hypothesis_notes, "analyze": "", "solution": ""},
                "analysis_output": None,
                "train": train_eval,
                "validation": validation_eval,
                "selection": selection,
                "timing": {
                    **timing,
                    "candidate_wall_ms": (time.perf_counter() - candidate_start) * 1000.0,
                },
            }

    analysis_output = _read_json_value_if_exists(_analysis_path(evaluation_dir))
    if analysis_output is None:
        analysis_execution_start = time.perf_counter()
        analysis_retries_used = 0
        max_analysis_retries = _analysis_retry_limit()
        analysis_error: Exception | None = None
        for attempt in range(max_analysis_retries + 1):
            try:
                attempt_start = time.perf_counter()
                analysis_output = run_analysis(
                    candidate_dir,
                    train_instances_public,
                    manifest=manifest,
                    artifact_dir=evaluation_dir,
                )
                timing[f"analysis_execution_attempt_{attempt:02d}_wall_ms"] = (
                    time.perf_counter() - attempt_start
                ) * 1000.0
                analysis_error = None
                break
            except Exception as exc:
                analysis_error = exc
                timing[f"analysis_execution_attempt_{attempt:02d}_wall_ms"] = (
                    time.perf_counter() - attempt_start
                ) * 1000.0
                _archive_failed_analysis_attempt(
                    candidate_dir,
                    attempt=attempt,
                    analyze_py=analyze_py,
                    exc=exc,
                )
                if attempt >= max_analysis_retries:
                    break
                analysis_retries_used += 1
                retry_feedback = {
                    "failed_attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "instruction": (
                        "Rewrite analyze.py to be much cheaper. Prefer linear-time summaries, sampling, "
                        "early exits, and compact aggregate statistics. Avoid dynamic programming, "
                        "all-pairs scans, subset enumeration, branch-and-bound, or repeated exact solves."
                    ),
                    "previous_analyze_py": _prepare_text_for_prompt(analyze_py or ""),
                }
                analyze_messages = _build_analyze_messages(
                    manifest=manifest,
                    train_summary=train_summary,
                    plan=plan,
                    parent_record=parent_record,
                    hypothesis=hypothesis,
                    execution_feedback=retry_feedback,
                )
                try:
                    analyze_generation_start = time.perf_counter()
                    analyze_py, analyze_notes, analyze_metadata = _generate_stage_output(
                        messages=analyze_messages,
                        candidate_dir=candidate_dir,
                        stage_name="analyze",
                        response_schema_path=ANALYZE_RESPONSE_SCHEMA_PATH,
                        code_field="analyze_py",
                    )
                    analyze_metadata = {
                        **analyze_metadata,
                        "analysis_retry_attempt": attempt + 1,
                        "previous_execution_error": f"{type(exc).__name__}: {exc}",
                    }
                    _write_stage_artifacts(
                        candidate_dir,
                        stage_name="analyze",
                        filename="analyze.py",
                        code=analyze_py,
                        notes=analyze_notes,
                        metadata=analyze_metadata,
                    )
                    timing[f"analyze_retry_generation_attempt_{attempt + 1:02d}_wall_ms"] = (
                        time.perf_counter() - analyze_generation_start
                    ) * 1000.0
                except Exception as generation_exc:
                    analysis_error = generation_exc
                    _write_stage_failure_metadata(
                        candidate_dir,
                        stage_name=f"analyze_retry_{attempt + 1:02d}",
                        plan=plan,
                        messages=analyze_messages,
                        exc=generation_exc,
                    )
                    break
        timing["analysis_execution_wall_ms"] = (time.perf_counter() - analysis_execution_start) * 1000.0
        timing["analysis_retries_used"] = analysis_retries_used
        if analysis_output is None:
            assert analysis_error is not None
            train_eval = failed_summary(
                plan.slug(),
                "train",
                len(train_instances_full),
                f"{type(analysis_error).__name__}: {analysis_error}",
            )
            validation_eval = failed_summary(
                plan.slug(),
                "validation",
                len(validation_instances_full),
                f"{type(analysis_error).__name__}: {analysis_error}",
            )
            write_summary(evaluation_dir / "train_summary.json", train_eval)
            write_summary(evaluation_dir / "validation_summary.json", validation_eval)
            selection = summarize_selection(train_eval, validation_eval)
            return {
                "slug": plan.slug(),
                "plan": plan_payload,
                "hypothesis": hypothesis,
                "candidate_dir": str(candidate_dir),
                "evaluation_dir": str(evaluation_dir),
                "code_bundle": {"analyze_py": analyze_py, "solution_py": ""},
                "stage_notes": {"hypothesis": hypothesis_notes, "analyze": analyze_notes, "solution": ""},
                "analysis_output": None,
                "train": train_eval,
                "validation": validation_eval,
                "selection": selection,
                "timing": {
                    **timing,
                    "candidate_wall_ms": (time.perf_counter() - candidate_start) * 1000.0,
                },
            }

    solution_path = candidate_dir / "solution.py"
    solution_metadata_path = _stage_metadata_path(candidate_dir, "solution")
    solution_py = (
        _read_text_if_exists(solution_path)
        if solution_path.exists() and solution_metadata_path.exists()
        else None
    )
    solution_notes = (_read_text_if_exists(candidate_dir / "solution_notes.txt") or "").strip()
    if solution_py is None:
        solution_messages = _build_solution_messages(
            manifest=manifest,
            train_summary=train_summary,
            plan=plan,
            parent_record=parent_record,
            analyze_py=analyze_py,
            analysis_output=analysis_output,
            hypothesis=hypothesis,
        )
        try:
            solution_generation_start = time.perf_counter()
            solution_py, solution_notes, solution_metadata = _generate_stage_output(
                messages=solution_messages,
                candidate_dir=candidate_dir,
                stage_name="solution",
                response_schema_path=SOLUTION_RESPONSE_SCHEMA_PATH,
                code_field="solution_py",
            )
            _write_stage_artifacts(
                candidate_dir,
                stage_name="solution",
                filename="solution.py",
                code=solution_py,
                notes=solution_notes,
                metadata=solution_metadata,
            )
            timing["solution_generation_wall_ms"] = (time.perf_counter() - solution_generation_start) * 1000.0
        except Exception as exc:
            _write_stage_failure_metadata(
                candidate_dir,
                stage_name="solution",
                plan=plan,
                messages=solution_messages,
                exc=exc,
            )
            train_eval = failed_summary(plan.slug(), "train", len(train_instances_full), f"{type(exc).__name__}: {exc}")
            validation_eval = failed_summary(plan.slug(), "validation", len(validation_instances_full), f"{type(exc).__name__}: {exc}")
            selection = summarize_selection(train_eval, validation_eval)
            return {
                "slug": plan.slug(),
                "plan": plan_payload,
                "hypothesis": hypothesis,
                "candidate_dir": str(candidate_dir),
                "evaluation_dir": str(evaluation_dir),
                "code_bundle": {"analyze_py": analyze_py, "solution_py": ""},
                "stage_notes": {"hypothesis": hypothesis_notes, "analyze": analyze_notes, "solution": ""},
                "analysis_output": analysis_output,
                "train": train_eval,
                "validation": validation_eval,
                "selection": selection,
                "timing": {
                    **timing,
                    "candidate_wall_ms": (time.perf_counter() - candidate_start) * 1000.0,
                },
            }

    train_eval = _read_json_dict_if_exists(_summary_path(evaluation_dir, "train"))
    validation_eval = _read_json_dict_if_exists(_summary_path(evaluation_dir, "validation"))
    if train_eval is None or validation_eval is None:
        try:
            solver_build_start = time.perf_counter()
            solver = build_solver(candidate_dir, analysis=analysis_output, manifest=manifest)
            timing["solver_build_wall_ms"] = (time.perf_counter() - solver_build_start) * 1000.0
            train_eval_start = time.perf_counter()
            train_eval = evaluate_solver(problem_name, plan.slug(), solver, train_instances_full, split="train")
            timing["train_eval_wall_ms"] = (time.perf_counter() - train_eval_start) * 1000.0
            validation_eval_start = time.perf_counter()
            validation_eval = evaluate_solver(problem_name, plan.slug(), solver, validation_instances_full, split="validation")
            timing["validation_eval_wall_ms"] = (time.perf_counter() - validation_eval_start) * 1000.0
        except Exception as exc:
            train_eval = failed_summary(plan.slug(), "train", len(train_instances_full), f"{type(exc).__name__}: {exc}")
            validation_eval = failed_summary(plan.slug(), "validation", len(validation_instances_full), f"{type(exc).__name__}: {exc}")

    timing["candidate_wall_ms"] = (time.perf_counter() - candidate_start) * 1000.0
    selection = summarize_selection(train_eval, validation_eval)
    write_summary(evaluation_dir / "train_summary.json", train_eval)
    write_summary(evaluation_dir / "validation_summary.json", validation_eval)
    return {
        "slug": plan.slug(),
        "plan": plan_payload,
        "hypothesis": hypothesis,
        "candidate_dir": str(candidate_dir),
        "evaluation_dir": str(evaluation_dir),
        "code_bundle": {"analyze_py": analyze_py, "solution_py": solution_py},
        "stage_notes": {"hypothesis": hypothesis_notes, "analyze": analyze_notes, "solution": solution_notes},
        "analysis_output": analysis_output,
        "train": train_eval,
        "validation": validation_eval,
        "selection": selection,
        "timing": timing,
    }


def _evaluate_solver_only_plan(
    plan: LLMSolverOnlyPlan,
    *,
    manifest: dict[str, object],
    train_summary: dict[str, object],
    train_instances_public: list[dict[str, object]],
    train_instances_full: list[dict[str, object]],
    validation_instances_full: list[dict[str, object]],
    candidates_dir: Path,
    evaluations_dir: Path,
    evaluated_records: dict[str, dict[str, object]],
) -> dict[str, object]:
    candidate_start = time.perf_counter()
    problem_name = str(manifest["problem"])
    candidate_dir = candidates_dir / plan.slug()
    evaluation_dir = evaluations_dir / plan.slug()
    saved_record = _load_saved_llm_record(candidate_dir, evaluation_dir)
    if saved_record is not None:
        return saved_record

    timing: dict[str, float] = {}
    parent_record = evaluated_records.get(plan.parent_slug) if plan.parent_slug else None
    plan_payload = _plan_payload(plan)

    analyze_path = candidate_dir / "analyze.py"
    analyze_metadata_path = _stage_metadata_path(candidate_dir, "analyze")
    analyze_py = (
        _read_text_if_exists(analyze_path)
        if analyze_path.exists() and analyze_metadata_path.exists()
        else None
    )
    analyze_notes = (_read_text_if_exists(candidate_dir / "analyze_notes.txt") or "").strip()
    if analyze_py is None:
        analyze_messages = _build_solver_only_analyze_messages(
            manifest=manifest,
            train_summary=train_summary,
            plan=plan,
            parent_record=parent_record,
        )
        try:
            analyze_generation_start = time.perf_counter()
            analyze_py, analyze_notes, analyze_metadata = _generate_stage_output(
                messages=analyze_messages,
                candidate_dir=candidate_dir,
                stage_name="analyze",
                response_schema_path=ANALYZE_RESPONSE_SCHEMA_PATH,
                code_field="analyze_py",
            )
            _write_stage_artifacts(
                candidate_dir,
                stage_name="analyze",
                filename="analyze.py",
                code=analyze_py,
                notes=analyze_notes,
                metadata=analyze_metadata,
            )
            timing["analyze_generation_wall_ms"] = (time.perf_counter() - analyze_generation_start) * 1000.0
        except Exception as exc:
            _write_stage_failure_metadata(
                candidate_dir,
                stage_name="analyze",
                plan=plan,
                messages=analyze_messages,
                exc=exc,
            )
            train_eval = failed_summary(plan.slug(), "train", len(train_instances_full), f"{type(exc).__name__}: {exc}")
            validation_eval = failed_summary(plan.slug(), "validation", len(validation_instances_full), f"{type(exc).__name__}: {exc}")
            selection = summarize_selection(train_eval, validation_eval)
            return {
                "slug": plan.slug(),
                "plan": plan_payload,
                "hypothesis": None,
                "candidate_dir": str(candidate_dir),
                "evaluation_dir": str(evaluation_dir),
                "code_bundle": {"analyze_py": "", "solution_py": ""},
                "stage_notes": {"hypothesis": "", "analyze": "", "solution": ""},
                "analysis_output": None,
                "train": train_eval,
                "validation": validation_eval,
                "selection": selection,
                "timing": {
                    **timing,
                    "candidate_wall_ms": (time.perf_counter() - candidate_start) * 1000.0,
                },
            }

    analysis_output = _read_json_value_if_exists(_analysis_path(evaluation_dir))
    if analysis_output is None:
        try:
            analysis_start = time.perf_counter()
            analysis_output = run_analysis(
                candidate_dir,
                train_instances_public,
                manifest=manifest,
                artifact_dir=evaluation_dir,
            )
            timing["analysis_execution_wall_ms"] = (time.perf_counter() - analysis_start) * 1000.0
        except Exception as exc:
            train_eval = failed_summary(plan.slug(), "train", len(train_instances_full), f"{type(exc).__name__}: {exc}")
            validation_eval = failed_summary(plan.slug(), "validation", len(validation_instances_full), f"{type(exc).__name__}: {exc}")
            selection = summarize_selection(train_eval, validation_eval)
            return {
                "slug": plan.slug(),
                "plan": plan_payload,
                "hypothesis": None,
                "candidate_dir": str(candidate_dir),
                "evaluation_dir": str(evaluation_dir),
                "code_bundle": {"analyze_py": analyze_py, "solution_py": ""},
                "stage_notes": {"hypothesis": "", "analyze": analyze_notes, "solution": ""},
                "analysis_output": None,
                "train": train_eval,
                "validation": validation_eval,
                "selection": selection,
                "timing": {
                    **timing,
                    "candidate_wall_ms": (time.perf_counter() - candidate_start) * 1000.0,
                },
            }

    solution_path = candidate_dir / "solution.py"
    solution_metadata_path = _stage_metadata_path(candidate_dir, "solution")
    solution_py = (
        _read_text_if_exists(solution_path)
        if solution_path.exists() and solution_metadata_path.exists()
        else None
    )
    solution_notes = (_read_text_if_exists(candidate_dir / "solution_notes.txt") or "").strip()
    if solution_py is None:
        solution_messages = _build_solver_only_solution_messages(
            manifest=manifest,
            train_summary=train_summary,
            plan=plan,
            parent_record=parent_record,
            analyze_py=analyze_py,
            analysis_output=analysis_output,
        )
        try:
            solution_generation_start = time.perf_counter()
            solution_py, solution_notes, solution_metadata = _generate_stage_output(
                messages=solution_messages,
                candidate_dir=candidate_dir,
                stage_name="solution",
                response_schema_path=SOLUTION_RESPONSE_SCHEMA_PATH,
                code_field="solution_py",
            )
            _write_stage_artifacts(
                candidate_dir,
                stage_name="solution",
                filename="solution.py",
                code=solution_py,
                notes=solution_notes,
                metadata=solution_metadata,
            )
            timing["solution_generation_wall_ms"] = (time.perf_counter() - solution_generation_start) * 1000.0
        except Exception as exc:
            _write_stage_failure_metadata(
                candidate_dir,
                stage_name="solution",
                plan=plan,
                messages=solution_messages,
                exc=exc,
            )
            train_eval = failed_summary(plan.slug(), "train", len(train_instances_full), f"{type(exc).__name__}: {exc}")
            validation_eval = failed_summary(plan.slug(), "validation", len(validation_instances_full), f"{type(exc).__name__}: {exc}")
            selection = summarize_selection(train_eval, validation_eval)
            return {
                "slug": plan.slug(),
                "plan": plan_payload,
                "hypothesis": None,
                "candidate_dir": str(candidate_dir),
                "evaluation_dir": str(evaluation_dir),
                "code_bundle": {"analyze_py": analyze_py, "solution_py": ""},
                "stage_notes": {"hypothesis": "", "analyze": analyze_notes, "solution": ""},
                "analysis_output": analysis_output,
                "train": train_eval,
                "validation": validation_eval,
                "selection": selection,
                "timing": {
                    **timing,
                    "candidate_wall_ms": (time.perf_counter() - candidate_start) * 1000.0,
                },
            }

    train_eval = _read_json_dict_if_exists(_summary_path(evaluation_dir, "train"))
    validation_eval = _read_json_dict_if_exists(_summary_path(evaluation_dir, "validation"))
    if train_eval is None or validation_eval is None:
        try:
            solver_build_start = time.perf_counter()
            solver = build_solver(candidate_dir, analysis=analysis_output, manifest=manifest)
            timing["solver_build_wall_ms"] = (time.perf_counter() - solver_build_start) * 1000.0
            train_eval_start = time.perf_counter()
            train_eval = evaluate_solver(problem_name, plan.slug(), solver, train_instances_full, split="train")
            timing["train_eval_wall_ms"] = (time.perf_counter() - train_eval_start) * 1000.0
            validation_eval_start = time.perf_counter()
            validation_eval = evaluate_solver(problem_name, plan.slug(), solver, validation_instances_full, split="validation")
            timing["validation_eval_wall_ms"] = (time.perf_counter() - validation_eval_start) * 1000.0
        except Exception as exc:
            train_eval = failed_summary(plan.slug(), "train", len(train_instances_full), f"{type(exc).__name__}: {exc}")
            validation_eval = failed_summary(plan.slug(), "validation", len(validation_instances_full), f"{type(exc).__name__}: {exc}")

    timing["candidate_wall_ms"] = (time.perf_counter() - candidate_start) * 1000.0
    selection = summarize_selection(train_eval, validation_eval)
    write_summary(evaluation_dir / "train_summary.json", train_eval)
    write_summary(evaluation_dir / "validation_summary.json", validation_eval)
    return {
        "slug": plan.slug(),
        "plan": plan_payload,
        "hypothesis": None,
        "candidate_dir": str(candidate_dir),
        "evaluation_dir": str(evaluation_dir),
        "code_bundle": {"analyze_py": analyze_py, "solution_py": solution_py},
        "stage_notes": {"hypothesis": "", "analyze": analyze_notes, "solution": solution_notes},
        "analysis_output": analysis_output,
        "train": train_eval,
        "validation": validation_eval,
        "selection": selection,
        "timing": timing,
    }


def run_llm_synthesis_loop(
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
    train_summary = problem.summarize_training_data(train_public, manifest) if train_public else _empty_train_summary(manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = output_dir / "candidates"
    evaluations_dir = output_dir / "evaluations"

    effective_candidate_width = _effective_candidate_width(mode, candidate_width, beam_width)
    frontier = _seed_plans(mode, candidate_width=effective_candidate_width, beam_width=beam_width)
    evaluated: dict[str, dict[str, object]] = _load_saved_llm_records(
        candidates_dir,
        evaluations_dir,
        output_dir=output_dir,
    )
    rounds: list[dict[str, object]] = []
    history: list[dict[str, object]] = []

    for iteration in range(iterations):
        if not frontier:
            break
        current_round: list[dict[str, object]] = []
        for plan in frontier:
            if plan.slug() not in evaluated:
                evaluated[plan.slug()] = _evaluate_plan(
                    plan,
                    manifest=manifest,
                    train_summary=train_summary,
                    train_instances_public=train_public,
                    train_instances_full=train_full,
                    validation_instances_full=validation_full,
                    candidates_dir=candidates_dir,
                    evaluations_dir=evaluations_dir,
                    evaluated_records=evaluated,
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
    if analysis is None:
        analysis = _read_json_value_if_exists(Path(best_candidate["evaluation_dir"]) / "analysis.json")
    best_candidate["timing"] = dict(best_candidate.get("timing", {}))
    solution_path = Path(best_candidate["candidate_dir"]) / "solution.py"
    final_test_error = None
    if analysis is None:
        final_test_error = "No successful analysis output is available for the selected candidate."
    elif not solution_path.exists():
        final_test_error = "No solution.py is available for the selected candidate."

    if final_test_error is not None:
        best_candidate["test"] = failed_summary(
            str(best_candidate["slug"]),
            "test",
            len(test_full),
            final_test_error,
        )
        best_candidate["timing"]["best_candidate_test_wall_ms"] = 0.0
        write_summary(Path(best_candidate["evaluation_dir"]) / "test_summary.json", best_candidate["test"])
        if timing_reporter is not None:
            timing_reporter.record_best_candidate_test(
                wall_ms=0.0,
                summary=best_candidate["test"],
                slug=str(best_candidate["slug"]),
            )
    else:
        analysis_dir = output_dir / "best_candidate_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / "analysis.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        solver = build_solver(Path(best_candidate["candidate_dir"]), analysis=analysis, manifest=manifest)
        best_test_start = time.perf_counter()
        best_candidate["test"] = evaluate_solver(problem_name, best_candidate["slug"], solver, test_full, split="test")
        best_candidate["timing"]["best_candidate_test_wall_ms"] = (time.perf_counter() - best_test_start) * 1000.0
        write_summary(Path(best_candidate["evaluation_dir"]) / "test_summary.json", best_candidate["test"])
        if timing_reporter is not None:
            timing_reporter.record_best_candidate_test(
                wall_ms=best_candidate["timing"]["best_candidate_test_wall_ms"],
                summary=best_candidate["test"],
                slug=str(best_candidate["slug"]),
            )

    history_path = output_dir / "performance_history.json"
    write_history(history_path, history)
    plot_path = write_performance_plot(
        output_dir / performance_plot_filename(),
        history,
        title=f"{problem_name} llm search",
    )
    summary = {
        "problem": problem_name,
        "family": manifest["family"],
        "generator": "llm",
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


def run_llm_no_hint_synthesis_loop(
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
    train_summary = problem.summarize_training_data(train_public, manifest) if train_public else _empty_train_summary(manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = output_dir / "candidates"
    evaluations_dir = output_dir / "evaluations"

    effective_candidate_width = _effective_candidate_width(mode, candidate_width, beam_width)
    frontier = _seed_solver_only_plans(mode, candidate_width=effective_candidate_width, beam_width=beam_width)
    evaluated: dict[str, dict[str, object]] = _load_saved_llm_records(
        candidates_dir,
        evaluations_dir,
        output_dir=output_dir,
    )
    rounds: list[dict[str, object]] = []
    history: list[dict[str, object]] = []

    for iteration in range(iterations):
        if not frontier:
            break
        current_round: list[dict[str, object]] = []
        for plan in frontier:
            if plan.slug() not in evaluated:
                evaluated[plan.slug()] = _evaluate_solver_only_plan(
                    plan,
                    manifest=manifest,
                    train_summary=train_summary,
                    train_instances_public=train_public,
                    train_instances_full=train_full,
                    validation_instances_full=validation_full,
                    candidates_dir=candidates_dir,
                    evaluations_dir=evaluations_dir,
                    evaluated_records=evaluated,
                )
                if timing_reporter is not None:
                    timing_reporter.record_synthesis_candidate(evaluated[plan.slug()])
            current_round.append(evaluated[plan.slug()])
        survivors = _select_top_survivors(list(evaluated.values()), mode=mode, beam_width=beam_width)
        history.append(progress_point(iteration, survivors[0]))
        rounds.append(
            {
                "iteration": iteration,
                "evaluated_this_round": [record["slug"] for record in current_round],
                "frontier_after_ranking": [record["slug"] for record in survivors],
                "best_selected_slug": survivors[0]["slug"],
                "best_selected_train": survivors[0]["train"],
                "best_selected_validation": survivors[0]["validation"],
                "best_selected_selection": survivors[0]["selection"],
            }
        )
        if timing_reporter is not None:
            timing_reporter.record_synthesis_round(rounds[-1])
        if iteration == iterations - 1:
            break
        frontier = _child_solver_only_plans(
            iteration + 1,
            survivors,
            mode=mode,
            beam_width=beam_width,
            candidate_width=effective_candidate_width,
        )

    best_candidate = max(evaluated.values(), key=lambda record: selection_sort_key(record["selection"]))
    analysis = run_analysis(
        Path(best_candidate["candidate_dir"]),
        train_public,
        manifest=manifest,
        artifact_dir=output_dir / "best_candidate_analysis",
    )
    solver = build_solver(Path(best_candidate["candidate_dir"]), analysis=analysis, manifest=manifest)
    best_candidate["timing"] = dict(best_candidate.get("timing", {}))
    best_test_start = time.perf_counter()
    best_candidate["test"] = evaluate_solver(problem_name, best_candidate["slug"], solver, test_full, split="test")
    best_candidate["timing"]["best_candidate_test_wall_ms"] = (time.perf_counter() - best_test_start) * 1000.0
    write_summary(Path(best_candidate["evaluation_dir"]) / "test_summary.json", best_candidate["test"])
    if timing_reporter is not None:
        timing_reporter.record_best_candidate_test(
            wall_ms=best_candidate["timing"]["best_candidate_test_wall_ms"],
            summary=best_candidate["test"],
            slug=str(best_candidate["slug"]),
        )

    history_path = output_dir / "performance_history.json"
    write_history(history_path, history)
    plot_path = write_performance_plot(
        output_dir / performance_plot_filename(),
        history,
        title=f"{problem_name} llm no-hint search",
    )
    summary = {
        "problem": problem_name,
        "family": manifest["family"],
        "generator": "llm_no_hint",
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
