from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dasbench.agents.candidate import build_solver, run_analysis
from dasbench.cli import cmd_run_agent
from dasbench.data import load_manifest, load_split
from dasbench.eval.evaluator import SolverTimeoutError, _resolved_solver_timeout, _solver_timeout
from dasbench.integrations import load_openai_dotenv
from dasbench.problems import get_problem_definition
from dasbench.problems.graph_utils import adjacency_sets, normalized_edges
from dasbench.utils import load_jsonl, public_instance, timestamp_token, write_json, write_jsonl


PACE_REPO_URL = "https://github.com/MarioGrobler/PACE2025-instances"
PACE_RAW_BASE_URL = "https://raw.githubusercontent.com/MarioGrobler/PACE2025-instances"
DEFAULT_OUTPUT_ROOT = Path("artifacts/pace2025_dominating_set")
DEFAULT_CACHE_DIR = Path("artifacts/external/pace2025-instances")
BEST_GREEDY_BASELINES = ("high_degree_greedy", "marginal_gain_greedy", "redundancy_aware")
PACE_EVALUATION_FIELDNAMES = [
    "instance_id",
    "pace_source_path",
    "num_vertices",
    "num_edges",
    "feasible",
    "solution_size",
    "lower_bound",
    "reference_objective",
    "runtime_ms",
    "solution_file",
    "error",
]
PACE_OFFICIAL_COMPARISON_NOTE = (
    "PACE exact-track score requires proving optimality, and PACE heuristic score requires "
    "per-instance best-known/optimal solution values. The public instance repository does not "
    "bundle those labels, so this artifact reports feasibility, solution sizes, runtimes, and "
    "lower-bound/reference proxy columns."
)


@dataclass(frozen=True)
class SourceConfig:
    pace_root: Path | None
    cache_dir: Path
    github_ref: str


@dataclass(frozen=True)
class ReferenceResult:
    solution: list[int]
    objective_value: int
    source: str
    runtime_ms: float
    attempts: list[dict[str, object]]


def _public_ds_path(track: str, index: int) -> str:
    if track == "exact":
        return f"ds/exact/exact_{index:03d}.gr"
    if track == "heuristic":
        return f"ds/heuristic/heuristic_{index:03d}.gr.tar.xz"
    raise ValueError(f"Unsupported PACE track: {track!r}")


def _private_ds_path(track: str, index: int) -> str:
    if track == "exact":
        return f"private/ds/exact/private_exact_{index:03d}.gr"
    if track == "heuristic":
        return f"private/ds/heuristic/private_heuristic_{index:03d}.gr.tar.xz"
    raise ValueError(f"Unsupported PACE track: {track!r}")


def _instance_paths(
    *,
    source: str,
    track: str,
    start_index: int,
    count: int,
) -> list[str]:
    if count < 0:
        raise ValueError("Instance counts must be nonnegative.")
    if not (1 <= start_index <= 100):
        raise ValueError("PACE instance indices are one-based and must be in 1..100.")
    end_index = start_index + count - 1
    if count and end_index > 100:
        raise ValueError(f"Requested PACE instance range {start_index}..{end_index}, but only 1..100 exist.")
    builder = _private_ds_path if source == "private" else _public_ds_path
    return [builder(track, index) for index in range(start_index, end_index + 1)]


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _download_file(relative_path: str, source_config: SourceConfig, *, force: bool = False) -> Path:
    target = source_config.cache_dir / relative_path
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{PACE_RAW_BASE_URL}/{source_config.github_ref}/{relative_path}"
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    _atomic_write_bytes(target, payload)
    return target


def _source_file(relative_path: str, source_config: SourceConfig) -> Path:
    if source_config.pace_root is not None:
        path = source_config.pace_root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"PACE source file not found: {path}")
        return path
    return _download_file(relative_path, source_config)


def _read_pace_text_from_path(path: Path) -> str:
    if path.name.endswith(".tar.xz"):
        with tarfile.open(path, mode="r:xz") as archive:
            members = [member for member in archive.getmembers() if member.isfile() and member.name.endswith(".gr")]
            if not members:
                raise ValueError(f"No .gr file found inside {path}.")
            if len(members) > 1:
                members = sorted(members, key=lambda item: item.name)
            handle = archive.extractfile(members[0])
            if handle is None:
                raise ValueError(f"Could not read {members[0].name} from {path}.")
            return handle.read().decode("utf-8")
    return path.read_text(encoding="utf-8")


def _read_pace_text(relative_path: str, source_config: SourceConfig) -> str:
    path = _source_file(relative_path, source_config)
    try:
        return _read_pace_text_from_path(path)
    except (EOFError, tarfile.TarError):
        if source_config.pace_root is not None or not path.name.endswith(".tar.xz"):
            raise
        path.unlink(missing_ok=True)
        path = _download_file(relative_path, source_config, force=True)
        return _read_pace_text_from_path(path)


def parse_pace_gr_text(text: str, *, instance_id: str, source_path: str) -> dict[str, object]:
    num_vertices: int | None = None
    declared_edges: int | None = None
    edges: list[list[int]] = []
    comments: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("c"):
            comments.append(line[1:].strip())
            continue
        parts = line.split()
        if parts[0] == "p":
            if len(parts) != 4 or parts[1] != "ds":
                raise ValueError(f"Expected `p ds n m` at {source_path}:{line_number}, got {line!r}.")
            num_vertices = int(parts[2])
            declared_edges = int(parts[3])
            continue
        if num_vertices is None:
            raise ValueError(f"Edge appeared before the `p ds` header at {source_path}:{line_number}.")
        if len(parts) != 2:
            raise ValueError(f"Expected an edge line at {source_path}:{line_number}, got {line!r}.")
        u = int(parts[0]) - 1
        v = int(parts[1]) - 1
        if u == v:
            raise ValueError(f"Self-loop in PACE instance {source_path}:{line_number}.")
        if not (0 <= u < num_vertices and 0 <= v < num_vertices):
            raise ValueError(f"Vertex outside 1..{num_vertices} in {source_path}:{line_number}.")
        edges.append([u, v])
    if num_vertices is None or declared_edges is None:
        raise ValueError(f"Missing `p ds n m` header in {source_path}.")
    normalized = normalized_edges(edges)
    return {
        "id": instance_id,
        "num_vertices": num_vertices,
        "edges": normalized,
        "pace_source_path": source_path,
        "pace_declared_edges": declared_edges,
        "pace_actual_edges": len(normalized),
        "pace_comment_prefix": comments[:3],
    }


def load_pace_instance(relative_path: str, source_config: SourceConfig, *, split: str) -> dict[str, object]:
    text = _read_pace_text(relative_path, source_config)
    stem = Path(relative_path).name
    if stem.endswith(".tar.xz"):
        stem = stem.removesuffix(".tar.xz")
    instance_id = f"pace2025-ds-{split}-{stem.removesuffix('.gr')}"
    instance = parse_pace_gr_text(text, instance_id=instance_id, source_path=relative_path)
    instance["pace_split"] = split
    return instance


def domination_lower_bound(instance: dict[str, object]) -> int:
    num_vertices = int(instance["num_vertices"])
    if num_vertices <= 0:
        return 0
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    max_closed_neighborhood = max((len(neighbors) + 1 for neighbors in adjacency), default=1)
    return max(1, math.ceil(num_vertices / max_closed_neighborhood))


def compute_reference_solution(
    instance: dict[str, object],
    *,
    baseline_names: Iterable[str],
) -> ReferenceResult:
    problem = get_problem_definition("mds")
    registry = problem.baseline_registry()
    exposed = public_instance(instance)
    attempts: list[dict[str, object]] = []
    best_solution: list[int] | None = None
    best_name: str | None = None
    total_runtime_ms = 0.0
    for baseline_name in baseline_names:
        if baseline_name not in registry:
            raise ValueError(f"Unknown MDS baseline {baseline_name!r}. Available: {sorted(registry)}")
        start = time.perf_counter()
        try:
            raw_solution = registry[baseline_name](exposed)
            runtime_ms = (time.perf_counter() - start) * 1000.0
            solution = problem.canonicalize_solution(raw_solution, exposed)
            valid, error = problem.validate_solution(solution, exposed)
            objective = len(solution) if valid else None
        except Exception as exc:
            runtime_ms = (time.perf_counter() - start) * 1000.0
            valid = False
            error = f"{type(exc).__name__}: {exc}"
            solution = []
            objective = None
        total_runtime_ms += runtime_ms
        attempts.append(
            {
                "baseline": baseline_name,
                "valid": valid,
                "objective_value": objective,
                "runtime_ms": runtime_ms,
                "error": error if not valid else None,
            }
        )
        if valid and (best_solution is None or len(solution) < len(best_solution)):
            best_solution = solution
            best_name = baseline_name
    if best_solution is None or best_name is None:
        raise RuntimeError(f"No reference baseline produced a valid dominating set for {instance['id']}.")
    return ReferenceResult(
        solution=best_solution,
        objective_value=len(best_solution),
        source=f"best_of:{','.join(baseline_names)}",
        runtime_ms=total_runtime_ms,
        attempts=attempts,
    )


def _annotate_for_proxy_scoring(
    instance: dict[str, object],
    *,
    reference_baselines: Iterable[str],
) -> dict[str, object]:
    lower_bound = domination_lower_bound(instance)
    reference = compute_reference_solution(instance, baseline_names=reference_baselines)
    annotated = dict(instance)
    annotated["optimum_objective"] = lower_bound
    annotated["optimum_source"] = "pace2025_proxy_lower_bound:max_closed_neighborhood"
    annotated["_pace_reference_objective"] = reference.objective_value
    annotated["_pace_reference_solution"] = reference.solution
    annotated["_pace_reference_source"] = reference.source
    annotated["_pace_reference_runtime_ms"] = reference.runtime_ms
    annotated["_pace_reference_attempts"] = reference.attempts
    return annotated


def _dataset_manifest(
    *,
    dataset_dir: Path,
    output_root: Path,
    track: str,
    test_source: str,
    split_sizes: dict[str, int],
    reference_baselines: list[str],
    source_config: SourceConfig,
) -> dict[str, object]:
    return {
        "problem": "mds",
        "family": f"pace2025_ds_{track}_{test_source}",
        "description": (
            "PACE 2025 Dominating Set instances imported from the official instance repository. "
            "The repository does not include private optima, so DASBench normalized_quality uses a "
            "valid lower-bound proxy rather than the official PACE score."
        ),
        "ground_truth_hidden_rule": {
            "source": "PACE 2025 Dominating Set",
            "official_repo": PACE_REPO_URL,
            "track": track,
            "test_source": test_source,
        },
        "metric_definition": {
            "primary": "lower_bound_ratio",
            "secondary": "feasibility_rate",
            "tertiary": "average_runtime_ms",
            "notes": (
                "normalized_quality is lower_bound / returned_dominating_set_size. "
                "It is a monotone proxy for synthesis selection, not the official PACE score."
            ),
        },
        "instance_schema_version": "mds.v1",
        "compute_optima": False,
        "instance_params": {
            "source": "pace2025",
            "track": track,
            "test_source": test_source,
            "reference_baselines": reference_baselines,
            "github_ref": source_config.github_ref,
        },
        "family_params": {},
        "split_sizes": split_sizes,
        "seeds": {},
        "artifact_paths": {
            "dataset_dir": str(dataset_dir),
            "splits": {
                "train": str(dataset_dir / "train.jsonl"),
                "validation": str(dataset_dir / "validation.jsonl"),
                "test": str(dataset_dir / "test.jsonl"),
            },
            "manifest": str(dataset_dir / "manifest.json"),
            "benchmark_spec": str(dataset_dir / "benchmark_spec.json"),
            "reproducibility": str(dataset_dir / "reproducibility.json"),
            "output_root": str(output_root),
        },
    }


def build_pace_dataset(
    *,
    dataset_dir: Path,
    output_root: Path,
    source_config: SourceConfig,
    track: str,
    test_source: str,
    train_count: int,
    validation_count: int,
    test_count: int,
    public_start_index: int,
    test_start_index: int,
    reference_baselines: list[str],
) -> dict[str, object]:
    train_paths = _instance_paths(
        source="public",
        track=track,
        start_index=public_start_index,
        count=train_count,
    )
    validation_paths = _instance_paths(
        source="public",
        track=track,
        start_index=public_start_index + train_count,
        count=validation_count,
    )
    test_paths = _instance_paths(
        source=test_source,
        track=track,
        start_index=test_start_index,
        count=test_count,
    )
    split_paths = {
        "train": train_paths,
        "validation": validation_paths,
        "test": test_paths,
    }

    # Reuse an already-built dataset dir verbatim when its spec matches the requested splits:
    # re-annotating PACE's huge private graphs (running the greedy reference on up to 4.2M-vertex
    # instances) costs ~1 h, so skip it when the exact same instances are already annotated here.
    existing_spec = dataset_dir / "benchmark_spec.json"
    if existing_spec.is_file() and (dataset_dir / "manifest.json").is_file() and all(
        (dataset_dir / f"{split}.jsonl").is_file() for split in split_paths
    ):
        try:
            spec = json.loads(existing_spec.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            spec = {}
        spec_matches = (
            spec.get("track") == track
            and spec.get("test_source") == test_source
            and spec.get("train_paths") == train_paths
            and spec.get("validation_paths") == validation_paths
            and spec.get("test_paths") == test_paths
        )
        sizes_ok = all(
            sum(1 for _ in (dataset_dir / f"{split}.jsonl").open(encoding="utf-8")) == len(paths)
            for split, paths in split_paths.items()
        )
        if spec_matches and sizes_ok:
            print(f"Reusing existing PACE dataset at {dataset_dir} (spec matches; skipping re-annotation).")
            return json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))

    dataset_dir.mkdir(parents=True, exist_ok=True)
    problem = get_problem_definition("mds")
    split_sizes = {split: len(paths) for split, paths in split_paths.items()}
    for split, paths in split_paths.items():
        rows: list[dict[str, object]] = []
        for relative_path in paths:
            instance = load_pace_instance(relative_path, source_config, split=split)
            problem.validate_instance(instance)
            rows.append(_annotate_for_proxy_scoring(instance, reference_baselines=reference_baselines))
        write_jsonl(dataset_dir / f"{split}.jsonl", rows)

    manifest = _dataset_manifest(
        dataset_dir=dataset_dir,
        output_root=output_root,
        track=track,
        test_source=test_source,
        split_sizes=split_sizes,
        reference_baselines=reference_baselines,
        source_config=source_config,
    )
    write_json(dataset_dir / "manifest.json", manifest)
    repro = {
        "source": "pace2025_dominating_set",
        "track": track,
        "test_source": test_source,
        "train_paths": train_paths,
        "validation_paths": validation_paths,
        "test_paths": test_paths,
        "reference_baselines": reference_baselines,
        "pace_root": str(source_config.pace_root) if source_config.pace_root is not None else None,
        "cache_dir": str(source_config.cache_dir),
        "github_ref": source_config.github_ref,
    }
    write_json(dataset_dir / "benchmark_spec.json", repro)
    write_json(dataset_dir / "reproducibility.json", repro)
    return manifest


def write_pace_solution(path: Path, solution: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["c generated by dasbench PACE 2025 Dominating Set adapter", str(len(solution))]
    lines.extend(str(vertex + 1) for vertex in sorted(solution))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pace_evaluation_artifacts(
    *,
    dataset_dir: Path,
    agent_run_dir: Path,
    output_dir: Path,
    best_candidate_slug: str,
    rows: list[dict[str, object]],
    solution_dir: Path,
    error: str | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pace_private_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PACE_EVALUATION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    feasible_rows = [row for row in rows if row.get("feasible")]
    feasible_count = len(feasible_rows)
    total_solution_size = sum(int(row.get("solution_size") or 0) for row in feasible_rows)
    total_runtime_ms = sum(float(row.get("runtime_ms") or 0.0) for row in rows)
    timeout_count = sum(1 for row in rows if "SolverTimeoutError" in str(row.get("error", "")))
    summary = {
        "schema_version": "pace2025_ds_evaluation.v1",
        "dataset_dir": str(dataset_dir),
        "agent_run_dir": str(agent_run_dir),
        "best_candidate_slug": best_candidate_slug,
        "num_instances": len(rows),
        "feasible_count": feasible_count,
        "invalid_count": len(rows) - feasible_count,
        "timeout_count": timeout_count,
        "solver_timeout_seconds": _resolved_solver_timeout(None),
        "total_solution_size": total_solution_size,
        "average_solution_size": total_solution_size / feasible_count if feasible_count else 0.0,
        "average_runtime_ms": total_runtime_ms / len(rows) if rows else 0.0,
        "results_csv": str(csv_path),
        "solutions_dir": str(solution_dir),
        "official_comparison_note": PACE_OFFICIAL_COMPARISON_NOTE,
    }
    if error:
        summary["error"] = error
    write_json(output_dir / "pace_evaluation_summary.json", summary)
    return summary


def _failed_pace_evaluation_rows(test_full: list[dict[str, object]], *, error: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for instance in test_full:
        rows.append(
            {
                "instance_id": instance["id"],
                "pace_source_path": instance.get("pace_source_path"),
                "num_vertices": instance["num_vertices"],
                "num_edges": len(instance["edges"]),
                "feasible": False,
                "solution_size": 0,
                "lower_bound": instance.get("optimum_objective"),
                "reference_objective": instance.get("_pace_reference_objective"),
                "runtime_ms": 0.0,
                "solution_file": "",
                "error": error,
            }
        )
    return rows


def export_pace_evaluation(
    *,
    dataset_dir: Path,
    agent_run_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    manifest = load_manifest(dataset_dir)
    synthesis_summary = json.loads((agent_run_dir / "synthesis_summary.json").read_text(encoding="utf-8"))
    best_candidate = synthesis_summary["best_candidate"]
    candidate_dir = Path(best_candidate["candidate_dir"])
    solution_dir = output_dir / "solutions"
    if solution_dir.exists():
        shutil.rmtree(solution_dir)
    solution_path = candidate_dir / "solution.py"
    if not solution_path.exists():
        candidate_error = None
        for split_name in ("test", "validation", "train"):
            split_summary = best_candidate.get(split_name)
            if isinstance(split_summary, dict) and split_summary.get("error"):
                candidate_error = str(split_summary["error"])
                break
        error = (
            f"Selected candidate `{best_candidate['slug']}` is not exportable because "
            f"`{solution_path}` does not exist."
        )
        if candidate_error:
            error = f"{error} Candidate error: {candidate_error}"
        test_full = load_jsonl(dataset_dir / "test.jsonl")
        rows = _failed_pace_evaluation_rows(test_full, error=error)
        return _write_pace_evaluation_artifacts(
            dataset_dir=dataset_dir,
            agent_run_dir=agent_run_dir,
            output_dir=output_dir,
            best_candidate_slug=str(best_candidate["slug"]),
            rows=rows,
            solution_dir=solution_dir,
            error=error,
        )

    train_public = load_split(dataset_dir, "train", public=True)
    try:
        analysis = run_analysis(
            candidate_dir,
            train_public,
            manifest=manifest,
            artifact_dir=output_dir / "analysis",
        )
        solver = build_solver(candidate_dir, analysis=analysis, manifest=manifest)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        test_full = load_jsonl(dataset_dir / "test.jsonl")
        rows = _failed_pace_evaluation_rows(test_full, error=error)
        return _write_pace_evaluation_artifacts(
            dataset_dir=dataset_dir,
            agent_run_dir=agent_run_dir,
            output_dir=output_dir,
            best_candidate_slug=str(best_candidate["slug"]),
            rows=rows,
            solution_dir=solution_dir,
            error=error,
        )

    problem = get_problem_definition("mds")
    test_full = load_split(dataset_dir, "test")
    rows: list[dict[str, object]] = []
    for instance in test_full:
        exposed = public_instance(instance)
        start = time.perf_counter()
        error: str | None = None
        try:
            with _solver_timeout(
                None,
                name=str(best_candidate["slug"]),
                split="pace_export",
                instance_id=instance.get("id"),
            ):
                raw_solution = solver(exposed)
                solution = problem.canonicalize_solution(raw_solution, exposed)
                feasible, validation_error = problem.validate_solution(solution, exposed)
            runtime_ms = (time.perf_counter() - start) * 1000.0
            if not feasible:
                error = validation_error
        except SolverTimeoutError as exc:
            runtime_ms = (time.perf_counter() - start) * 1000.0
            solution = []
            feasible = False
            error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            runtime_ms = (time.perf_counter() - start) * 1000.0
            solution = []
            feasible = False
            error = f"{type(exc).__name__}: {exc}"
        solution_size = len(solution) if feasible else 0
        solution_file = solution_dir / f"{instance['id']}.sol"
        if feasible:
            write_pace_solution(solution_file, solution)
        rows.append(
            {
                "instance_id": instance["id"],
                "pace_source_path": instance.get("pace_source_path"),
                "num_vertices": instance["num_vertices"],
                "num_edges": len(instance["edges"]),
                "feasible": feasible,
                "solution_size": solution_size,
                "lower_bound": instance.get("optimum_objective"),
                "reference_objective": instance.get("_pace_reference_objective"),
                "runtime_ms": runtime_ms,
                "solution_file": str(solution_file) if feasible else "",
                "error": error or "",
            }
        )
    return _write_pace_evaluation_artifacts(
        dataset_dir=dataset_dir,
        agent_run_dir=agent_run_dir,
        output_dir=output_dir,
        best_candidate_slug=str(best_candidate["slug"]),
        rows=rows,
        solution_dir=solution_dir,
    )


def _reference_baselines_from_arg(value: str) -> list[str]:
    if value == "best-greedy":
        return list(BEST_GREEDY_BASELINES)
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one DASBench agent synthesis pass on PACE 2025 Dominating Set instances."
    )
    parser.add_argument("--pace-root", type=Path, help="Existing checkout of MarioGrobler/PACE2025-instances.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--github-ref", default="master")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--run-output-dir", type=Path)
    parser.add_argument("--evaluation-output-dir", type=Path)
    parser.add_argument("--track", choices=["exact", "heuristic"], default="exact")
    parser.add_argument("--test-source", choices=["private", "public"], default="private")
    parser.add_argument("--train-count", type=int, default=5)
    parser.add_argument("--validation-count", type=int, default=5)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--public-start-index", type=int, default=1)
    parser.add_argument("--test-start-index", type=int, default=1)
    parser.add_argument(
        "--reference-baselines",
        default="best-greedy",
        help="Comma-separated MDS baselines used for reference columns, or `best-greedy`.",
    )
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--generator", choices=["auto", "template", "llm"], default="auto")
    parser.add_argument("--mode", choices=["single", "beam"], default="beam")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--candidate-width", type=int, default=3)
    parser.add_argument("--run-id")
    parser.add_argument("--skip-baselines", action="store_true", default=True)
    parser.add_argument("--no-skip-baselines", dest="skip_baselines", action="store_false")
    parser.add_argument("--no-export-solutions", dest="export_solutions", action="store_false")
    parser.set_defaults(export_solutions=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_openai_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    experiment_id = args.experiment_id or f"pace2025_ds_{timestamp_token()}"
    output_root = args.output_root / experiment_id
    dataset_dir = args.dataset_dir or output_root / "dataset"
    run_output_dir = args.run_output_dir or output_root / "agent_run"
    evaluation_output_dir = args.evaluation_output_dir or output_root / "pace_evaluation"
    reference_baselines = _reference_baselines_from_arg(args.reference_baselines)
    source_config = SourceConfig(
        pace_root=args.pace_root,
        cache_dir=args.cache_dir,
        github_ref=args.github_ref,
    )

    manifest = build_pace_dataset(
        dataset_dir=dataset_dir,
        output_root=output_root,
        source_config=source_config,
        track=args.track,
        test_source=args.test_source,
        train_count=args.train_count,
        validation_count=args.validation_count,
        test_count=args.test_count,
        public_start_index=args.public_start_index,
        test_start_index=args.test_start_index,
        reference_baselines=reference_baselines,
    )
    print(f"PACE dataset: {dataset_dir}")
    print(json.dumps({"problem": manifest["problem"], "family": manifest["family"], "split_sizes": manifest["split_sizes"]}, indent=2, sort_keys=True))
    if args.build_only:
        return 0

    run_args = argparse.Namespace(
        dataset_dir=str(dataset_dir),
        run_id=args.run_id,
        output_dir=str(run_output_dir),
        generator=args.generator,
        mode=args.mode,
        iterations=args.iterations,
        beam_width=args.beam_width,
        candidate_width=args.candidate_width,
        gurobi_baseline_enabled=False,
        gurobi_time_limit_seconds=60.0,
        gurobi_threads=1,
        native_exact_time_limit_seconds=None,
        external_exact_baselines="off",
        external_time_limit_seconds=60.0,
        external_threads=1,
        external_solver_config=None,
        skip_baselines=args.skip_baselines,
        overlap_baselines_with_synthesis=False,
    )
    cmd_run_agent(run_args)
    if args.export_solutions:
        summary = export_pace_evaluation(
            dataset_dir=dataset_dir,
            agent_run_dir=run_output_dir,
            output_dir=evaluation_output_dir,
        )
        print(f"PACE evaluation summary: {evaluation_output_dir / 'pace_evaluation_summary.json'}")
        print(
            "PACE private eval: "
            f"feasible={summary['feasible_count']}/{summary['num_instances']} "
            f"avg_size={summary['average_solution_size']:.3f} "
            f"avg_runtime_ms={summary['average_runtime_ms']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
