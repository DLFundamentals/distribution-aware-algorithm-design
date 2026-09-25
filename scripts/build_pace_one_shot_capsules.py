#!/usr/bin/env python3
"""Build one-shot solver-generation capsules for every PACE competition track.

The 2026-06 capsules under `artifacts/llm_pv_one_shot_capsules/` covered the 21
controlled targets plus PACE 2025 Dominating Set, and were answered by Claude
Code and Codex releases that no longer exist. New runs are therefore not
comparable with the recorded ones, so every PACE track is rebuilt here and both
agents are re-run from scratch.

One capsule is the complete workspace for one agent's single generation run. It
carries the public training split only -- no validation split, no test split, no
hidden-rule metadata, no reference solutions, no DasBench source tree.

Unlike the 2026-06 layout, each generator gets its own workspace per track
rather than sharing one directory. Sharing required deleting the previous
agent's `solution.py` between runs; a missed deletion would let the second agent
read the first one's answer and silently void the comparison.

  python scripts/build_pace_one_shot_capsules.py                 # all 5 tracks
  python scripts/build_pace_one_shot_capsules.py --tracks pace2025_hs

Prompts are assembled from the same `_solution_contract` table the LLM-PV
baseline uses, so the contract text an agent sees matches what the API models
saw.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.llm_pv_benchmark import _prompt_train_summary, _solution_contract  # noqa: E402
from dasbench.data import load_manifest, load_split  # noqa: E402
from dasbench.problems import get_problem_definition  # noqa: E402
from dasbench.utils import public_instance, timestamp_token, write_json, write_jsonl  # noqa: E402

ARTIFACT_KIND = "pace_one_shot_capsules"
DEFAULT_OUTPUT_ROOT = Path("artifacts/pace_one_shot_capsules")
DEFAULT_GENERATORS = ("claude", "codex")

# Matches the 2026-06 capsules: 64 examples requested, clipped to 60k JSON chars.
# PACE graphs are large enough that this yields one truncated example per track,
# which is the point -- the agent gets the same partial view our pipeline had.
PROMPT_TRAIN_EXAMPLES = 64
PROMPT_JSON_CHAR_LIMIT = 60_000

# Only these manifest instance_params reach the workspace. The full params carry
# `reference_baselines` on the Dominating Set track, which names the released
# solvers we compare against.
PUBLIC_INSTANCE_PARAM_KEYS = ("source", "track")


@dataclass(frozen=True)
class Track:
    track_id: str
    title: str
    dataset_dir: str
    display_problem: str
    instance_noun: str
    distribution_note: str
    size_note: str


TRACKS: tuple[Track, ...] = (
    Track(
        track_id="pace2025_ds",
        title="PACE 2025 Dominating Set",
        dataset_dir="artifacts/pace2025_dominating_set/pace2025_ds_heuristic_llm_01/dataset",
        display_problem="pace2025_dominating_set_mds",
        instance_noun="MDS",
        distribution_note=(
            "Instances are PACE 2025 Dominating Set graphs. Infer reusable graph-heuristic "
            "structure from the public training examples."
        ),
        size_note="PACE graphs may be large.",
    ),
    Track(
        track_id="pace2025_hs",
        title="PACE 2025 Hitting Set",
        dataset_dir="artifacts/pace_competitions/baselines_pace2025_hs_private100/dataset",
        display_problem="pace2025_hitting_set",
        instance_noun="hitting set",
        distribution_note=(
            "Instances are PACE 2025 Hitting Set hypergraphs. Infer reusable set-cover-style "
            "structure from the public training examples."
        ),
        size_note="PACE hypergraphs declare far more vertices than actually occur in any set.",
    ),
    Track(
        track_id="pace2024_ocm_cutwidth",
        title="PACE 2024 One-Sided Crossing Minimization (cutwidth track)",
        dataset_dir="artifacts/pace_competitions/baselines_pace2024_ocm_cutwidth_private100/dataset",
        display_problem="pace2024_ocm_cutwidth",
        instance_noun="OCM",
        distribution_note=(
            "Instances are PACE 2024 One-Sided Crossing Minimization bipartite graphs from the "
            "cutwidth track. Infer reusable ordering-heuristic structure from the public training examples."
        ),
        size_note="crossing counting is quadratic in the free-side degree if done naively.",
    ),
    Track(
        track_id="pace2024_ocm_exact",
        title="PACE 2024 One-Sided Crossing Minimization (exact track)",
        dataset_dir="artifacts/pace_competitions/baselines_pace2024_ocm_exact_private100/dataset",
        display_problem="pace2024_ocm_exact",
        instance_noun="OCM",
        distribution_note=(
            "Instances are PACE 2024 One-Sided Crossing Minimization bipartite graphs from the "
            "exact track. Infer reusable ordering-heuristic structure from the public training examples."
        ),
        size_note="crossing counting is quadratic in the free-side degree if done naively.",
    ),
    Track(
        track_id="pace2022_dfvs",
        title="PACE 2022 Directed Feedback Vertex Set",
        dataset_dir="artifacts/pace_competitions/baselines_pace2022_dfvs_private100/dataset",
        display_problem="pace2022_dfvs",
        instance_noun="DFVS",
        distribution_note=(
            "Instances are PACE 2022 Directed Feedback Vertex Set digraphs. Infer reusable "
            "cycle-breaking structure from the public training examples."
        ),
        size_note="cycle enumeration does not scale on these digraphs.",
    ),
)

TRACKS_BY_ID = {track.track_id: track for track in TRACKS}


def _public_manifest(track: Track, manifest: dict[str, object]) -> dict[str, object]:
    raw_params = manifest.get("instance_params")
    params = raw_params if isinstance(raw_params, dict) else {}
    return {
        "problem": manifest["problem"],
        "instance_schema_version": manifest["instance_schema_version"],
        "metric_definition": manifest["metric_definition"],
        "instance_params": {k: params[k] for k in PUBLIC_INSTANCE_PARAM_KEYS if k in params},
        "distribution_note": track.distribution_note,
    }


def _prepare_train_examples(
    train_public: list[dict[str, object]],
) -> tuple[list[dict[str, object]], bool, int]:
    """Same clipping rule as the LLM-PV prompt builder, inlined to stay explicit.

    A single example larger than the whole budget is truncated rather than
    dropped, so the agent always sees at least the head of one real instance.
    """
    examples: list[dict[str, object]] = []
    used_chars = 0
    for instance in train_public[:PROMPT_TRAIN_EXAMPLES]:
        serialized = json.dumps(instance, sort_keys=True)
        if examples and used_chars + len(serialized) > PROMPT_JSON_CHAR_LIMIT:
            return examples, True, used_chars
        if not examples and len(serialized) > PROMPT_JSON_CHAR_LIMIT:
            clipped = (
                serialized[:PROMPT_JSON_CHAR_LIMIT]
                + f"...[truncated {len(serialized) - PROMPT_JSON_CHAR_LIMIT} chars]"
            )
            return [{"truncated_json_text": clipped, "id": instance.get("id")}], True, len(clipped)
        examples.append(instance)
        used_chars += len(serialized)
    return examples, len(train_public) > len(examples), used_chars


def _prompt_messages(
    track: Track,
    *,
    manifest_public: dict[str, object],
    train_summary: dict[str, object],
    train_public: list[dict[str, object]],
) -> list[dict[str, str]]:
    problem_name = str(manifest_public["problem"])
    problem = get_problem_definition(problem_name)
    examples, truncated, example_chars = _prepare_train_examples(train_public)
    contract = _solution_contract(problem_name, {"instance_params": manifest_public["instance_params"]})
    payload = {
        "method": "LLM-PV one-shot solver generation",
        "task": (
            f"Write one complete Python solver program for {track.title} instances exposed through "
            f"the DasBench {problem_name.upper()} interface. This is a one-shot generation run: you "
            "will not receive validation feedback, hidden rules, reference solutions, or a chance to "
            "revise the solution. Use the public training examples only as empirical examples or "
            "tuning data."
        ),
        "problem": {
            "name": track.display_problem,
            "description": f"{track.title} instances using the DasBench {problem_name} solution contract.",
            "metric_definition": manifest_public["metric_definition"],
            "instance_schema_version": manifest_public["instance_schema_version"],
        },
        "manifest": manifest_public,
        "solution_contract": contract,
        "train_summary": train_summary,
        "train_examples": examples,
        "train_examples_metadata": {
            "shown_count": len(examples),
            "available_count": len(train_public),
            "truncated": truncated,
            "approx_json_chars": example_chars,
        },
        "interface": {
            "file": "solution.py",
            "required": "define solve(instance, analysis=None, manifest=None) -> object",
            "alternative": "or define build_solver(analysis=None, manifest=None) -> callable",
        },
        "constraints": [
            "Write only solution.py in this workspace.",
            "Use train examples only as empirical distribution examples or tuning data.",
            "Do not assume access to optimum_objective, optimum_solution, private fields, reference "
            "solutions, files outside this workspace, network, or API calls at solve time.",
            f"The solver will receive one public {track.instance_noun} instance at a time. "
            f"{contract['required_shape']}",
            f"Keep per-instance runtime low and deterministic; {track.size_note}",
            "Return the raw solution object from solve(...), not a dict wrapper, score, explanation, "
            "tuple, or metadata payload.",
            "Do not import dasbench.integrations, gurobipy, pyscipopt, ortools, highspy, or external "
            "exact solvers.",
            "If using randomness, seed it deterministically from the public instance id.",
        ],
        "response_format": {
            "file_to_create": "solution.py",
            "content": "full Python module defining solve(...) or build_solver(...)",
        },
    }
    return [
        {
            "role": "system",
            "content": (
                f"You generate one executable Python solver for a {track.title} / DasBench "
                f"{problem_name.upper()} problem from public training examples only. Produce a "
                "self-contained solution.py program."
            ),
        },
        {"role": "user", "content": json.dumps(payload, indent=2, sort_keys=True)},
    ]


def _render_prompt_markdown(track: Track, messages: list[dict[str, str]]) -> str:
    system = next(m["content"] for m in messages if m["role"] == "system")
    user = next(m["content"] for m in messages if m["role"] == "user")
    return (
        f"# One-shot {track.title} solver generation\n\n"
        f"## System\n\n{system}\n\n"
        f"## User\n\n```json\n{user}\n```\n"
    )


def _workspace_readme(track: Track, contract: dict[str, object], generator: str) -> str:
    return f"""# One-shot {track.title} solver capsule

Generator slot: `{generator}`.

This directory is intended to be the complete workspace for one {generator} generation run.
It intentionally contains no validation split, no test split, no hidden-rule metadata, no reference solutions, and no DasBench source tree.

Task:
- Read `prompt.md` or `prompt_messages.json`.
- Write exactly one file named `solution.py` in this directory.
- `solution.py` must define `solve(instance, analysis=None, manifest=None) -> object` or `build_solver(analysis=None, manifest=None) -> callable`.
- {contract['required_shape']}
- Do not read files outside this directory.
- Do not use network calls or external APIs from the solver.

`train_public.jsonl` holds the {track.title} public training instances with private fields removed; `train_summary_public.json` is the aggregate view of them, and `manifest_public.json` is the public problem manifest.

The benchmark harness later copies `solution.py` out of this capsule and evaluates it on the held-out PACE instances.
"""


def build_track(
    track: Track,
    *,
    sweep_root: Path,
    sweep_id: str,
    generators: tuple[str, ...],
    force: bool,
) -> dict[str, object]:
    dataset_dir = REPO / track.dataset_dir
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"{track.track_id}: dataset dir missing: {track.dataset_dir}")

    manifest = load_manifest(dataset_dir)
    problem_name = str(manifest["problem"])
    problem = get_problem_definition(problem_name)

    train_full = load_split(dataset_dir, "train")
    train_public = [public_instance(instance) for instance in train_full]
    # Summaries are computed from the full records (the summarizer may use
    # reference objectives) and then sanitized, exactly as the prompt path does.
    train_summary = _prompt_train_summary(problem.summarize_training_data(train_full, manifest))
    manifest_public = _public_manifest(track, manifest)
    contract = _solution_contract(problem_name, {"instance_params": manifest_public["instance_params"]})

    leaked = sorted(set(train_summary) & {"family", "ground_truth_hidden_rule"})
    if leaked:
        raise AssertionError(f"{track.track_id}: train summary still carries {leaked}")

    messages = _prompt_messages(
        track,
        manifest_public=manifest_public,
        train_summary=train_summary,
        train_public=train_public,
    )
    prompt_markdown = _render_prompt_markdown(track, messages)

    target_dir = sweep_root / "targets" / track.track_id
    workspaces: dict[str, str] = {}
    for generator in generators:
        workspace = target_dir / f"workspace_{generator}"
        if workspace.exists():
            if not force:
                print(f"  {track.track_id}/{generator}: workspace exists, skipping (--force to rebuild)")
                workspaces[generator] = str(workspace.relative_to(REPO))
                continue
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "README.md").write_text(_workspace_readme(track, contract, generator), encoding="utf-8")
        (workspace / "prompt.md").write_text(prompt_markdown, encoding="utf-8")
        write_json(workspace / "prompt_messages.json", messages)
        write_json(workspace / "manifest_public.json", manifest_public)
        write_json(workspace / "train_summary_public.json", train_summary)
        write_jsonl(workspace / "train_public.jsonl", train_public)
        workspaces[generator] = str(workspace.relative_to(REPO))
        print(f"  {track.track_id}/{generator}: workspace ready ({workspace.relative_to(REPO)})")

        # The slot the operator copies solution.py into after the run.
        (sweep_root / "solutions" / generator / track.track_id).mkdir(parents=True, exist_ok=True)

    metadata = {
        "artifact_kind": ARTIFACT_KIND,
        "sweep_id": sweep_id,
        "track_id": track.track_id,
        "title": track.title,
        "problem": track.display_problem,
        "dasbench_problem": problem_name,
        "source_dataset_dir": track.dataset_dir,
        "source_family": manifest.get("family"),
        "prompt_train_examples": PROMPT_TRAIN_EXAMPLES,
        "prompt_json_char_limit": PROMPT_JSON_CHAR_LIMIT,
        "train_size": len(train_full),
        "validation_size": len(load_split(dataset_dir, "validation")),
        "test_size": (manifest.get("split_sizes") or {}).get("test"),
        "workspaces": workspaces,
        "workspace_policy": {
            "contains_test_split": False,
            "contains_validation_split": False,
            "contains_hidden_rule": False,
            "contains_reference_solutions": False,
            "contains_family_name_in_prompt": False,
            "intended_iterations": 1,
        },
    }
    write_json(target_dir / "capsule_metadata.json", metadata)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sweep-id", default=None, help="Defaults to a UTC timestamp token.")
    parser.add_argument("--generators", nargs="+", default=list(DEFAULT_GENERATORS))
    parser.add_argument("--tracks", nargs="+", default=None, help=f"Subset of {sorted(TRACKS_BY_ID)}")
    parser.add_argument("--force", action="store_true", help="Rebuild workspaces that already exist.")
    args = parser.parse_args(argv)

    if args.tracks:
        unknown = sorted(set(args.tracks) - set(TRACKS_BY_ID))
        if unknown:
            parser.error(f"unknown tracks: {unknown}; known: {sorted(TRACKS_BY_ID)}")
        tracks = tuple(TRACKS_BY_ID[t] for t in args.tracks)
    else:
        tracks = TRACKS

    sweep_id = args.sweep_id or timestamp_token()
    sweep_root = REPO / args.output_root / sweep_id
    generators = tuple(args.generators)

    print(f"sweep_id={sweep_id}")
    print(f"root={sweep_root.relative_to(REPO)}")
    print(f"generators={list(generators)}  tracks={[t.track_id for t in tracks]}")

    metadatas = []
    for track in tracks:
        print(f"{track.track_id} ({track.title})")
        metadatas.append(
            build_track(track, sweep_root=sweep_root, sweep_id=sweep_id, generators=generators, force=args.force)
        )

    write_json(
        sweep_root / "sweep_manifest.json",
        {
            "artifact_kind": ARTIFACT_KIND,
            "sweep_id": sweep_id,
            "created_at": timestamp_token(),
            "generators": list(generators),
            "track_count": len(metadatas),
            "notes": [
                "One-shot prompt capsules for external Claude Code / Codex solver generation.",
                "Every PACE track is rebuilt: the agent releases behind the 2026-06 numbers are "
                "retired, so new runs are not comparable with the recorded ones.",
                "Each generator has its own workspace per track; no workspace ever holds another "
                "generator's solution.py.",
                "Workspaces contain the public training split only; validation and test are excluded.",
            ],
            "tracks": metadatas,
        },
    )
    print(f"\nwrote {(sweep_root / 'sweep_manifest.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
