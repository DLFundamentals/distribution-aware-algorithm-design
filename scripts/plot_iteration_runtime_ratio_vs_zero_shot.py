from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from dasbench.agents.candidate import build_solver, run_analysis
from dasbench.eval.evaluator import evaluate_solver
from dasbench.integrations import (
    chat_completion_text,
    chat_config_with_overrides,
    create_chat_completion_raw,
    load_chat_api_config,
)
from dasbench.utils import load_jsonl, public_instance


REPO_ROOT = Path(__file__).resolve().parents[1]
SOLUTION_SCHEMA_PATH = REPO_ROOT / "dasbench" / "schemas" / "solution_code_bundle.json"
OKABE_ITO = {
    "coloring": "#0072B2",
    "maxsat": "#D55E00",
    "mdkp": "#009E73",
    "mds": "#CC79A7",
    "mis": "#E69F00",
    "packing_lp": "#56B4E9",
    "tsp": "#111111",
}
MARKERS = {
    "coloring": "o",
    "maxsat": "s",
    "mdkp": "^",
    "mds": "D",
    "mis": "P",
    "packing_lp": "X",
    "tsp": "v",
}
CACHE_SCHEMA_VERSION = "iteration_runtime_ratio_vs_zero_shot.v4"
ZERO_SHOT_PROMPT_VERSION = "zero_shot_solution.v4"

PROBLEM_OUTPUT_CONTRACTS = {
    "coloring": {
        "required_return": "Return a raw Python list[int] of length num_vertices.",
        "meaning": "Entry i is the color id assigned to vertex i.",
        "constraints": [
            "Do not return a dict wrapper.",
            "Every vertex 0..num_vertices-1 must have exactly one color.",
            "Colors can be any integers, but a compact 0-based coloring is preferred.",
        ],
        "example": "[0, 1, 0, 2, 1]",
    },
    "maxsat": {
        "required_return": "Return a raw Python list[bool] of length num_variables.",
        "meaning": "Entry i is the truth value for variable x{i+1}.",
        "constraints": [
            "Do not return a dict wrapper.",
            "Use actual booleans True/False, not 0/1 strings.",
            "The returned list length must equal num_variables.",
        ],
        "example": "[True, False, True, True]",
    },
    "mdkp": {
        "required_return": "Return a raw Python list[int] of selected item indices.",
        "meaning": "Each integer is an item id in 0..num_items-1 that is included in the knapsack.",
        "constraints": [
            "Do not return a dict wrapper.",
            "Prefer selected item indices rather than a binary vector.",
            "Returned items must satisfy all resource capacities.",
        ],
        "example": "[0, 4, 7, 11]",
    },
    "mds": {
        "required_return": "Return a raw Python list[int] of selected vertex ids.",
        "meaning": "The listed vertices form the dominating set.",
        "constraints": [
            "Do not return a dict wrapper.",
            "Vertices must be integers in 0..num_vertices-1.",
            "The set must dominate every vertex in the graph.",
        ],
        "example": "[1, 5, 9]",
    },
    "mis": {
        "required_return": "Return a raw Python list[int] of selected vertex ids.",
        "meaning": "The listed vertices form the independent set.",
        "constraints": [
            "Do not return a dict wrapper.",
            "Vertices must be integers in 0..num_vertices-1.",
            "No returned edge endpoints may both be selected.",
        ],
        "example": "[0, 3, 6, 9]",
    },
    "packing_lp": {
        "required_return": "Return a raw Python list[float] of length num_items.",
        "meaning": "Entry i is the fractional amount x_i assigned to item i.",
        "constraints": [
            "Do not return a dict wrapper.",
            "Each value must be a float in [0.0, 1.0].",
            "The returned list length must equal num_items.",
        ],
        "example": "[1.0, 0.0, 0.35, 1.0, 0.0]",
    },
    "tsp": {
        "required_return": "Return a raw Python list[int] representing a Hamiltonian tour.",
        "meaning": "The list is an ordered permutation of all city ids 0..num_cities-1.",
        "constraints": [
            "Do not return a dict wrapper.",
            "The returned list length must equal num_cities.",
            "Every city id must appear exactly once.",
        ],
        "example": "[0, 4, 1, 3, 2]",
    },
}


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_cache(path: Path, *, reasoning_effort: str | None) -> dict[str, object]:
    if not path.exists():
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "zero_shot_reasoning_effort": reasoning_effort,
            "zero_shot_test_runtime_ms": {},
            "candidate_test_runtime_ms": {},
            "series": {},
        }
    payload = _load_json(path)
    if (
        payload.get("schema_version") != CACHE_SCHEMA_VERSION
        or payload.get("zero_shot_reasoning_effort") != reasoning_effort
    ):
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "zero_shot_reasoning_effort": reasoning_effort,
            "zero_shot_test_runtime_ms": {},
            "candidate_test_runtime_ms": {},
            "series": {},
        }
    payload.setdefault("zero_shot_test_runtime_ms", {})
    payload.setdefault("candidate_test_runtime_ms", {})
    payload.setdefault("series", {})
    payload.setdefault("zero_shot_reasoning_effort", reasoning_effort)
    return payload


def _cache_key(problem: str, family: str, slug: str) -> str:
    return f"{problem}/{family}:{slug}"


def _schema_of(value: object) -> object:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        if not value:
            return {"type": "list", "items": "unknown"}
        item_schemas = [_schema_of(item) for item in value[:3]]
        first = item_schemas[0]
        uniform = all(item == first for item in item_schemas[1:])
        return {
            "type": "list",
            "items": first if uniform else item_schemas,
            "example_length": len(value),
        }
    if isinstance(value, dict):
        return {key: _schema_of(item) for key, item in value.items()}
    return type(value).__name__


def _example_of(value: object, *, max_list_items: int = 4) -> object:
    if isinstance(value, list):
        return [_example_of(item, max_list_items=max_list_items) for item in value[:max_list_items]]
    if isinstance(value, tuple):
        return [_example_of(item, max_list_items=max_list_items) for item in value[:max_list_items]]
    if isinstance(value, dict):
        return {key: _example_of(item, max_list_items=max_list_items) for key, item in value.items()}
    return value


def _solution_messages(
    manifest: dict[str, object],
    instance_schema: object,
    instance_example: object,
) -> list[dict[str, str]]:
    problem = str(manifest["problem"])
    metric_definition = manifest.get("metric_definition", {})
    instance_params = manifest.get("instance_params", {})
    output_contract = PROBLEM_OUTPUT_CONTRACTS.get(problem)
    if output_contract is None:
        raise ValueError(f"Missing output contract for problem {problem!r}.")
    payload = {
        "problem": problem,
        "instance_params": instance_params,
        "metric_definition": metric_definition,
        "public_instance_schema": instance_schema,
        "public_instance_example": instance_example,
        "output_contract": output_contract,
        "task": "Write a generic zero-shot solver for this problem class.",
        "num_training_instances": 0,
        "constraints": [
            "Do not use or assume training examples.",
            "Return only valid Python code for solution.py plus brief notes.",
            "Prefer deterministic solvers.",
            "Use Python standard library only.",
            "Do not rely on hidden distributional structure.",
            "Favor feasible, reasonably good heuristics with low runtime over heavy exact search.",
            "The file must define `solve(instance, analysis=None, manifest=None)` or `build_solver(...)` exactly as dasbench expects.",
            "The solver input is the public instance dictionary described by `public_instance_schema`, not raw stdin.",
            "The solver must return the raw Python object described in `output_contract`, not a wrapper dict.",
        ],
        "response_format": {
            "solution_py": "Full contents of solution.py",
            "notes": "Brief explanation of the generic strategy",
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You write compact, correct Python solvers for combinatorial optimization tasks. "
                "When no training data is available, produce a generic heuristic tuned for speed. "
                "Return only structured JSON matching the requested schema."
            ),
        },
        {"role": "user", "content": json.dumps(payload, indent=2, sort_keys=True)},
    ]


def _generate_zero_shot_solution(
    candidate_dir: Path,
    manifest: dict[str, object],
    instance_schema: object,
    instance_example: object,
    *,
    reasoning_effort: str | None,
) -> None:
    solution_path = candidate_dir / "solution.py"
    metadata_path = candidate_dir / "solution_generation_metadata.json"
    notes_path = candidate_dir / "solution_notes.txt"
    if solution_path.exists() and metadata_path.exists():
        metadata = _load_json(metadata_path)
        if (
            metadata.get("prompt_version") == ZERO_SHOT_PROMPT_VERSION
            and metadata.get("reasoning_effort") == reasoning_effort
        ):
            return

    config = chat_config_with_overrides(load_chat_api_config(required=True), reasoning_effort=reasoning_effort)
    response_schema = _load_json(SOLUTION_SCHEMA_PATH)
    messages = _solution_messages(manifest, instance_schema, instance_example)
    raw_response = create_chat_completion_raw(
        config,
        messages=messages,
        response_format=response_schema,
    )
    if raw_response.status_code != 200:
        raise RuntimeError(
            f"Zero-shot generic solver generation failed with status {raw_response.status_code}: "
            f"{raw_response.text[:1000]}"
        )
    completion = raw_response.parse()
    choice = completion.choices[0]
    message = choice.message
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise RuntimeError(f"Model refused zero-shot solver generation: {refusal}")
    content = chat_completion_text(completion)
    if not content:
        raise RuntimeError("Zero-shot solver generation returned empty content.")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise RuntimeError("Zero-shot solver generation did not return a JSON object.")
    solution_py = payload.get("solution_py")
    notes = payload.get("notes", "")
    if not isinstance(solution_py, str) or not solution_py.strip():
        raise RuntimeError("Zero-shot solver generation returned invalid `solution_py`.")
    if not isinstance(notes, str):
        raise RuntimeError("Zero-shot solver generation returned invalid `notes`.")

    candidate_dir.mkdir(parents=True, exist_ok=True)
    solution_path.write_text(solution_py.rstrip() + "\n", encoding="utf-8")
    notes_path.write_text(notes.strip() + "\n", encoding="utf-8")
    _write_json(
        metadata_path,
        {
            "api_config": config.public_dict(),
            "candidate_dir": str(candidate_dir),
            "prompt_version": ZERO_SHOT_PROMPT_VERSION,
            "reasoning_effort": reasoning_effort,
            "stage": "zero_shot_solution",
            "request_messages": messages,
            "status_code": raw_response.status_code,
            "parsed_payload": payload,
            "raw_http_text": raw_response.text[:20_000],
        },
    )


def _selected_best_slugs(timing_report: dict[str, object]) -> list[tuple[int, str]]:
    stages = timing_report.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("Timing report is missing `stages`.")
    synthesis = stages.get("synthesis")
    if not isinstance(synthesis, dict):
        raise ValueError("Timing report is missing synthesis stage data.")
    rounds = synthesis.get("rounds")
    if not isinstance(rounds, list):
        raise ValueError("Timing report is missing synthesis rounds.")

    points: list[tuple[int, str]] = []
    for round_payload in sorted(
        (row for row in rounds if isinstance(row, dict)),
        key=lambda row: int(row.get("iteration", 0)),
    ):
        slug = round_payload.get("best_selected_slug")
        if not isinstance(slug, str) or not slug:
            continue
        points.append((int(round_payload.get("iteration", 0)) + 1, slug))
    if not points:
        raise ValueError("No selected best candidates found in timing report.")
    return points


def _evaluate_solver_test_runtime_ms(
    *,
    problem: str,
    candidate_slug: str,
    candidate_dir: Path,
    dataset_dir: Path,
) -> float:
    manifest = _load_json(dataset_dir / "manifest.json")
    train_instances = load_jsonl(dataset_dir / "train.jsonl")
    test_instances = load_jsonl(dataset_dir / "test.jsonl")
    analysis = run_analysis(candidate_dir, train_instances, manifest=manifest)
    solver = build_solver(candidate_dir, analysis=analysis, manifest=manifest)
    summary = evaluate_solver(problem, candidate_slug, solver, test_instances, split="test")
    runtime_ms = summary.get("average_runtime_ms")
    if not isinstance(runtime_ms, (int, float)) or float(runtime_ms) <= 0.0:
        raise ValueError(f"Invalid test runtime for {problem}/{candidate_slug}.")
    return float(runtime_ms)


def _zero_shot_test_runtime_ms(
    *,
    problem: str,
    family: str,
    dataset_dir: Path,
    zero_shot_root: Path,
    reasoning_effort: str | None,
    cache: dict[str, object],
    cache_path: Path,
    reuse_from_family: str | None = None,
) -> float:
    zero_shot_cache = cache["zero_shot_test_runtime_ms"]
    if not isinstance(zero_shot_cache, dict):
        raise ValueError("Zero-shot runtime cache has invalid structure.")
    key = f"{problem}/{family}"
    cached = zero_shot_cache.get(key)
    if isinstance(cached, (int, float)) and float(cached) > 0.0:
        return float(cached)

    manifest = _load_json(dataset_dir / "manifest.json")
    test_instances = load_jsonl(dataset_dir / "test.jsonl")
    if not test_instances:
        raise ValueError(f"Test split is empty for {problem}/{family}.")
    example_instance = public_instance(test_instances[0])
    instance_schema = _schema_of(example_instance)
    instance_example = _example_of(example_instance)
    candidate_dir = zero_shot_root / "candidates" / problem / family / "llm_iter00_slot00"
    if reuse_from_family and reuse_from_family != family and not candidate_dir.exists():
        source_dir = zero_shot_root / "candidates" / problem / reuse_from_family / "llm_iter00_slot00"
        if source_dir.exists():
            candidate_dir.mkdir(parents=True, exist_ok=True)
            for filename in ("solution.py", "solution_notes.txt", "solution_generation_metadata.json"):
                source_path = source_dir / filename
                if source_path.exists():
                    shutil.copy2(source_path, candidate_dir / filename)
            metadata_path = candidate_dir / "solution_generation_metadata.json"
            if metadata_path.exists():
                metadata = _load_json(metadata_path)
                metadata["candidate_dir"] = str(candidate_dir)
                metadata["reused_from_family"] = reuse_from_family
                _write_json(metadata_path, metadata)
    _generate_zero_shot_solution(
        candidate_dir,
        manifest,
        instance_schema,
        instance_example,
        reasoning_effort=reasoning_effort,
    )
    runtime_ms = _evaluate_solver_test_runtime_ms(
        problem=problem,
        candidate_slug="llm_iter00_slot00",
        candidate_dir=candidate_dir,
        dataset_dir=dataset_dir,
    )
    zero_shot_cache[key] = runtime_ms
    _write_json(cache_path, cache)
    return runtime_ms


def _runtime_ratio_series(
    *,
    problem: str,
    family: str,
    agent_run_dir: Path,
    dataset_dir: Path,
    timing_report: dict[str, object],
    zero_shot_runtime_ms: float,
    cache: dict[str, object],
    cache_path: Path,
) -> list[tuple[int, float]]:
    synthesis_summary = _load_json(agent_run_dir / "synthesis_summary.json")
    best_candidate = synthesis_summary.get("best_candidate")
    candidate_cache = cache["candidate_test_runtime_ms"]
    if not isinstance(candidate_cache, dict):
        raise ValueError("Candidate runtime cache has invalid structure.")

    round_points = _selected_best_slugs(timing_report)
    best_so_far_ms: float | None = None
    series: list[tuple[int, float]] = [(0, 1.0)]

    for iteration_value, slug in round_points:
        key = _cache_key(problem, family, slug)
        cached_runtime_ms = candidate_cache.get(key)
        if isinstance(cached_runtime_ms, (int, float)) and float(cached_runtime_ms) > 0.0:
            runtime_ms = float(cached_runtime_ms)
        elif (
            isinstance(best_candidate, dict)
            and best_candidate.get("slug") == slug
            and isinstance(best_candidate.get("test"), dict)
            and isinstance(best_candidate["test"].get("average_runtime_ms"), (int, float))
            and float(best_candidate["test"]["average_runtime_ms"]) > 0.0
        ):
            runtime_ms = float(best_candidate["test"]["average_runtime_ms"])
            candidate_cache[key] = runtime_ms
            _write_json(cache_path, cache)
        else:
            runtime_ms = _evaluate_solver_test_runtime_ms(
                problem=problem,
                candidate_slug=slug,
                candidate_dir=agent_run_dir / "candidates" / slug,
                dataset_dir=dataset_dir,
            )
            candidate_cache[key] = runtime_ms
            _write_json(cache_path, cache)
        best_so_far_ms = runtime_ms if best_so_far_ms is None else min(best_so_far_ms, runtime_ms)
        series.append((iteration_value, best_so_far_ms / zero_shot_runtime_ms))
    return series


def _paper_plot(
    *,
    path_png: Path,
    series: dict[str, list[tuple[int, float]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#9CA3AF",
            "axes.labelcolor": "#111827",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "figure.facecolor": "white",
            "font.size": 10,
            "legend.fontsize": 9,
            "xtick.color": "#374151",
            "xtick.labelsize": 10,
            "ytick.color": "#374151",
            "ytick.labelsize": 10,
        }
    )

    figure, axis = plt.subplots(figsize=(8.8, 4.9))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.axhline(1.0, color="#6B7280", linewidth=1.05, linestyle=(0, (4, 3)), alpha=0.95, zorder=1)

    all_y_values: list[float] = []
    problem_order = sorted(series, key=lambda problem: series[problem][-1][1])
    for problem in problem_order:
        points = series[problem]
        x_values = [x for x, _ in points]
        y_values = [y for _, y in points]
        all_y_values.extend(y_values)
        axis.plot(
            x_values,
            y_values,
            color=OKABE_ITO.get(problem, "#374151"),
            marker=MARKERS.get(problem, "o"),
            linewidth=2.1,
            markersize=5.6,
            markerfacecolor="white",
            markeredgewidth=1.2,
            solid_capstyle="round",
            zorder=3,
            label=problem,
        )

    y_min = min(all_y_values)
    y_max = max(all_y_values)
    if y_min > 0.0 and (y_max / y_min) >= 1.6:
        axis.set_yscale("log", base=2)
        candidate_ticks = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
        ticks = [tick for tick in candidate_ticks if y_min * 0.9 <= tick <= y_max * 1.12]
        if 1.0 not in ticks:
            ticks.append(1.0)
        axis.set_yticks(sorted(set(ticks)))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}x"))

    axis.set_xlim(-0.05, 5.05)
    axis.set_xticks([0, 1, 2, 3, 4, 5])
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Best Runtime So Far / Zero-Shot Runtime")
    axis.set_title("Runtime Improvement Across Iterations")
    axis.grid(True, axis="y", color="#D1D5DB", alpha=0.8, linewidth=0.8)
    axis.grid(False, axis="x")
    axis.tick_params(axis="both", which="major", length=0, pad=6)
    axis.margins(x=0.01)

    for spine_name in ("left", "bottom"):
        axis.spines[spine_name].set_color("#9CA3AF")
        axis.spines[spine_name].set_linewidth(0.8)

    handles, labels = axis.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            color="#6B7280",
            linewidth=1.05,
            linestyle=(0, (4, 3)),
        )
    )
    labels.append("zero-shot baseline")
    axis.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.5,
    )

    figure.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    path_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path_png, dpi=240, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a combined runtime-ratio iteration plot by comparing the completed "
            "`iterations_05` sweep against zero-shot generic LLM solvers."
        )
    )
    parser.add_argument("sweep_root", type=Path, help="Path to the completed iteration_count_sweep run root.")
    parser.add_argument("--condition-id", default="iterations_05")
    parser.add_argument("--problem", help="Filter plotting to one problem.")
    parser.add_argument("--family", help="Filter plotting to one family. Requires --problem.")
    parser.add_argument(
        "--zero-shot-reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default=None,
        help=(
            "Reasoning effort for the zero-shot solver generation calls. "
            "Defaults to the configured LLM API value."
        ),
    )
    parser.add_argument(
        "--zero-shot-root",
        type=Path,
        default=None,
        help=(
            "Artifact root for zero-shot candidate code and metadata. "
            "Defaults to <sweep_root>/zero_shot_generic_<reasoning_effort>."
        ),
    )
    parser.add_argument(
        "--reuse-zero-shot-from-family",
        default=None,
        help=(
            "Reuse the existing zero-shot solver code from another family of the same problem "
            "instead of generating a new one for the filtered target."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for the combined plot. Defaults to "
            "<sweep_root>/plots/iteration_runtime_ratio_vs_zero_shot_<reasoning_effort>."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.family and not args.problem:
        raise SystemExit("`--family` requires `--problem`.")

    sweep_root = args.sweep_root.resolve()
    condition_root = sweep_root / "targets" / str(args.condition_id)
    if not condition_root.exists():
        raise SystemExit(f"Condition directory not found: {condition_root}")

    config = load_chat_api_config(required=True)
    zero_shot_reasoning_effort = args.zero_shot_reasoning_effort or config.reasoning_effort
    zero_shot_reasoning_label = zero_shot_reasoning_effort or "default"
    zero_shot_root = (
        args.zero_shot_root.resolve()
        if args.zero_shot_root is not None
        else sweep_root / f"zero_shot_generic_{zero_shot_reasoning_label}"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else sweep_root / "plots" / f"iteration_runtime_ratio_vs_zero_shot_{zero_shot_reasoning_label}"
    )
    cache_path = output_dir / "iteration_runtime_ratio_vs_zero_shot_data.json"
    cache = _load_cache(cache_path, reasoning_effort=zero_shot_reasoning_effort)

    series: dict[str, list[tuple[int, float]]] = {}
    for timing_path in sorted(condition_root.glob("*/*/agent_run/timing_report.json")):
        family_dir = timing_path.parent.parent
        agent_run_dir = timing_path.parent
        dataset_dir = family_dir / "dataset"
        problem = family_dir.parent.name
        family = family_dir.name
        if args.problem and problem != args.problem:
            continue
        if args.family and family != args.family:
            continue
        timing_report = _load_json(timing_path)

        zero_shot_runtime_ms = _zero_shot_test_runtime_ms(
            problem=problem,
            family=family,
            dataset_dir=dataset_dir,
            zero_shot_root=zero_shot_root,
            reasoning_effort=zero_shot_reasoning_effort,
            cache=cache,
            cache_path=cache_path,
            reuse_from_family=args.reuse_zero_shot_from_family,
        )
        ratio_series = _runtime_ratio_series(
            problem=problem,
            family=family,
            agent_run_dir=agent_run_dir,
            dataset_dir=dataset_dir,
            timing_report=timing_report,
            zero_shot_runtime_ms=zero_shot_runtime_ms,
            cache=cache,
            cache_path=cache_path,
        )
        series[problem] = ratio_series
        cache["series"][f"{problem}/{family}"] = [
            {"iteration": int(iteration), "runtime_ratio": float(runtime_ratio)}
            for iteration, runtime_ratio in ratio_series
        ]
        _write_json(cache_path, cache)

    if not series:
        raise SystemExit(f"No timing reports found under {condition_root}.")

    path_png = output_dir / "iteration_runtime_ratio_vs_zero_shot.png"
    _paper_plot(path_png=path_png, series=series)
    print(path_png)
    print(cache_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
