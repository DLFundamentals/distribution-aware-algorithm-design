from __future__ import annotations

import argparse
import json
import math
import shutil
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from benchmarks.pace_common import (
    SolverRecipe,
    SolverSpec,
    clone_and_build_recipe,
    download_url,
    natural_key,
    parse_solver_argument,
    run_process,
    write_csv,
    write_json,
)
from dasbench.agents.candidate import build_solver, run_analysis
from dasbench.cli import cmd_run_agent
from dasbench.data import load_manifest, load_split
from dasbench.eval.evaluator import SolverTimeoutError, _resolved_solver_timeout, _solver_timeout
from dasbench.integrations import load_openai_dotenv
from dasbench.problems import get_problem_definition
from dasbench.problems.dfvs import greedy_dfvs, greedy_vertex_disjoint_cycle_lower_bound
from dasbench.problems.ocm import crossing_count
from dasbench.utils import load_jsonl, public_instance, timestamp_token, write_jsonl


PACE2025_INSTANCES_RAW = "https://raw.githubusercontent.com/MarioGrobler/PACE2025-instances"
PACE2025_INSTANCES_REPO = "https://github.com/MarioGrobler/PACE2025-instances"
PACE2024_BASE = "https://pacechallenge.org/2024"
DFVS_HEURISTIC_ALL_URL = "https://heibox.uni-heidelberg.de/f/97634323e3cb4aab8291/?dl=1"

DEFAULT_OUTPUT_ROOT = Path("artifacts/pace_competitions")
DEFAULT_CACHE_DIR = Path("artifacts/external/pace_competitions")
DEFAULT_SOLVER_ROOT = Path("artifacts/external/pace_solvers")
DFVS_LOWER_BOUND_CYCLE_CAP = 1024
DFVS_REFERENCE_MAX_VERTICES = 10_000
DFVS_REFERENCE_MAX_ARCS = 100_000


@dataclass(frozen=True)
class CompetitionConfig:
    name: str
    problem: str
    family: str
    track: str
    description: str
    instance_schema_version: str
    metric_notes: str


COMPETITIONS: dict[str, CompetitionConfig] = {
    "pace2025_hs": CompetitionConfig(
        name="pace2025_hs",
        problem="hitting_set",
        family="pace2025_hs_heuristic_private",
        track="heuristic",
        description="PACE 2025 Hitting Set heuristic-track instances.",
        instance_schema_version="hitting_set.v1",
        metric_notes="normalized_quality is lower_bound_proxy / returned_hitting_set_size.",
    ),
    "pace2024_ocm_exact": CompetitionConfig(
        name="pace2024_ocm_exact",
        problem="ocm",
        family="pace2024_ocm_exact_private",
        track="exact",
        description="PACE 2024 One-sided Crossing Minimization exact-track instances.",
        instance_schema_version="ocm.v1",
        metric_notes="normalized_quality is optimal_crossings / returned_crossings, with zero-crossing special handling.",
    ),
    "pace2024_ocm_cutwidth": CompetitionConfig(
        name="pace2024_ocm_cutwidth",
        problem="ocm",
        family="pace2024_ocm_cutwidth_private",
        track="cutwidth",
        description="PACE 2024 One-sided Crossing Minimization parameterized cutwidth-track instances.",
        instance_schema_version="ocm.v1",
        metric_notes="normalized_quality is optimal_crossings / returned_crossings, with zero-crossing special handling.",
    ),
    "pace2024_ocm_heuristic": CompetitionConfig(
        name="pace2024_ocm_heuristic",
        problem="ocm",
        family="pace2024_ocm_heuristic_private",
        track="heuristic",
        description="PACE 2024 One-sided Crossing Minimization heuristic-track instances.",
        instance_schema_version="ocm.v1",
        metric_notes="normalized_quality is reference_crossings / returned_crossings; this is not the official PACE score.",
    ),
    "pace2022_dfvs_heuristic": CompetitionConfig(
        name="pace2022_dfvs_heuristic",
        problem="dfvs",
        family="pace2022_dfvs_heuristic_private",
        track="heuristic",
        description="PACE 2022 Directed Feedback Vertex Set heuristic-track instances.",
        instance_schema_version="dfvs.v1",
        metric_notes="normalized_quality is vertex_disjoint_cycle_lower_bound / returned_dfvs_size.",
    ),
}


SOLVER_RECIPES: dict[str, SolverRecipe] = {
    "root": SolverRecipe(
        name="root",
        competition="pace2025_hs",
        repo_url="https://github.com/lxily/PACE2025.DS-HS",
        commit="a7574ce8beace02298ca75535458af1ab47019c1",
        build_commands=("g++ -o pace_solver -O2 -I . solver/lib/*.cpp solver/tools/*.cpp -std=c++2a",),
        executable="pace_solver",
    ),
    "greeduce": SolverRecipe(
        name="greeduce",
        competition="pace2025_hs",
        repo_url="https://github.com/adampolak/greeduce",
        commit="78d9d34039c3e0ca73e1375e337a792e2d2eb059",
        build_commands=("make",),
        executable="solver",
    ),
    "shadoks": SolverRecipe(
        name="shadoks",
        competition="pace2025_hs",
        repo_url="https://github.com/gfonsecabr/shadoks-PACE2025",
        commit="dd753c05f471fea9e6ac6b128de7f7caed01466e",
        build_commands=(),
        executable="heuristic",
    ),
    "fontanf": SolverRecipe(
        name="fontanf",
        competition="pace2025_hs",
        repo_url="https://github.com/fontanf/pace2025",
        commit="72b4fc3ed7ca9448167f4108061250d45b6ea45c",
        build_commands=(
            "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release",
            "cmake --build build --config Release --parallel",
            "cmake --install build --config Release --prefix install",
        ),
        executable="install/bin/pace2025_hs_heuristic",
    ),
    "cimat": SolverRecipe(
        name="cimat",
        competition="pace2024_ocm",
        repo_url="https://github.com/carlossegurag/PaceChallenge24",
        commit="c8c5dbf5cb1e1aa0942529963b1d54b2cdad9fa8",
        build_commands=("make -C src",),
        executable="src/solver",
        run_cwd="src",
    ),
    "pingpong": SolverRecipe(
        name="pingpong",
        competition="pace2024_ocm",
        repo_url="https://github.com/mwien/pingpong",
        commit="d64fbb2c77bc271a2018abd454cf9a95e4d863db",
        build_commands=("cargo build --release",),
        executable="target/release/pingpong",
    ),
    "ocmu64": SolverRecipe(
        name="ocmu64",
        competition="pace2024_ocm",
        repo_url="https://github.com/mjdv/ocmu64",
        commit="e8d3379a68fbc6852682077360ac0948e7f3e0f4",
        build_commands=('RUSTFLAGS="-Ctarget-cpu=native" cargo build --profile submit',),
        executable="target/submit/ocmu64",
    ),
    "diverses": SolverRecipe(
        name="diverses",
        competition="pace2022_dfvs_heuristic",
        repo_url="https://github.com/swacisko/pace-2022",
        commit="a38c1ee8d8645d194af6e5523b512416dd6f2e9d",
        build_commands=("cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G Ninja", "cmake --build build --parallel"),
        executable="build/DiVerSeS",
    ),
    "dreyfvs": SolverRecipe(
        name="dreyfvs",
        competition="pace2022_dfvs_heuristic",
        repo_url="https://github.com/Nanored4498/DreyFVS",
        commit="a7b6f0520facb869040610841daab9658dd5fd99",
        build_commands=("cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G Ninja", "cmake --build build --parallel"),
        executable="build/DFVS",
    ),
    "kenneth_dfvs": SolverRecipe(
        name="kenneth_dfvs",
        competition="pace2022_dfvs_heuristic",
        repo_url="https://github.com/KennethLangedal/DFVS",
        commit="1c226b8bd3cb1412cde55edbe81a22a2aa7bd8ea",
        build_commands=(
            "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -G Ninja",
            "cmake --build build --parallel",
        ),
        executable="build/dfvs_heuristic",
    ),
}


DEFAULT_SOLVERS = {
    "pace2025_hs": "root,greeduce,shadoks,fontanf",
    "pace2024_ocm_exact": "pingpong,ocmu64",
    "pace2024_ocm_cutwidth": "ocmu64,pingpong",
    "pace2024_ocm_heuristic": "cimat",
    "pace2022_dfvs_heuristic": "diverses,dreyfvs,kenneth_dfvs",
}


def _strip_commented_lines(text: str, *, comment_prefix: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith(comment_prefix)]


def parse_hs_hgr_text(text: str, *, instance_id: str, source_path: str) -> dict[str, object]:
    lines = _strip_commented_lines(text, comment_prefix="c")
    if not lines:
        raise ValueError(f"Missing Hitting Set header in {source_path}.")
    header = lines[0].split()
    if len(header) != 4 or header[:2] != ["p", "hs"]:
        raise ValueError(f"Expected `p hs n m` header in {source_path}, got {lines[0]!r}.")
    num_vertices = int(header[2])
    declared_sets = int(header[3])
    sets: list[list[int]] = []
    for line_number, line in enumerate(lines[1:], start=2):
        values = [int(value) - 1 for value in line.split()]
        if not values:
            raise ValueError(f"Empty set in {source_path}:{line_number}.")
        if any(value < 0 or value >= num_vertices for value in values):
            raise ValueError(f"Vertex outside 1..{num_vertices} in {source_path}:{line_number}.")
        normalized = sorted(set(values))
        if len(normalized) != len(values):
            raise ValueError(f"Repeated vertex in set at {source_path}:{line_number}.")
        sets.append(normalized)
    if len(sets) != declared_sets:
        raise ValueError(f"Expected {declared_sets} sets in {source_path}, parsed {len(sets)}.")
    return {
        "id": instance_id,
        "num_vertices": num_vertices,
        "sets": sets,
        "pace_source_path": source_path,
        "pace_declared_sets": declared_sets,
    }


def parse_ocm_gr_text(text: str, *, instance_id: str, source_path: str) -> dict[str, object]:
    lines = _strip_commented_lines(text, comment_prefix="c")
    if not lines:
        raise ValueError(f"Missing OCM header in {source_path}.")
    header = lines[0].split()
    if len(header) not in {5, 6} or header[:2] != ["p", "ocr"]:
        raise ValueError(f"Expected `p ocr n0 n1 m [cw]` in {source_path}, got {lines[0]!r}.")
    num_fixed = int(header[2])
    num_free = int(header[3])
    declared_edges = int(header[4])
    cutwidth = int(header[5]) if len(header) == 6 else None
    index = 1
    cutwidth_order: list[int] | None = None
    if cutwidth is not None:
        order_lines = lines[index : index + num_fixed + num_free]
        if len(order_lines) != num_fixed + num_free:
            raise ValueError(f"Missing cutwidth order lines in {source_path}.")
        cutwidth_order = [int(line) - 1 for line in order_lines]
        index += num_fixed + num_free
    raw_edges: list[tuple[int, int]] = []
    for line_number, line in enumerate(lines[index:], start=index + 1):
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Expected OCM edge in {source_path}:{line_number}, got {line!r}.")
        fixed = int(parts[0])
        free = int(parts[1])
        if not 1 <= fixed <= num_fixed:
            raise ValueError(f"Fixed vertex {fixed} outside 1..{num_fixed} in {source_path}:{line_number}.")
        if not num_fixed + 1 <= free <= num_fixed + num_free:
            raise ValueError(
                f"Free vertex {free} outside {num_fixed + 1}..{num_fixed + num_free} in {source_path}:{line_number}."
            )
        raw_edges.append((fixed - 1, free - num_fixed - 1))
    if len(raw_edges) != declared_edges:
        raise ValueError(f"Expected {declared_edges} OCM edges in {source_path}, parsed {len(raw_edges)}.")
    edges = [[fixed, free] for fixed, free in sorted(set(raw_edges))]
    instance: dict[str, object] = {
        "id": instance_id,
        "num_fixed": num_fixed,
        "num_free": num_free,
        "edges": edges,
        "pace_source_path": source_path,
        "pace_declared_edges": declared_edges,
        "pace_actual_edges": len(edges),
        "pace_duplicate_edges_removed": len(raw_edges) - len(edges),
    }
    if cutwidth is not None:
        instance["cutwidth"] = cutwidth
        instance["cutwidth_order"] = cutwidth_order
    return instance


def parse_dfvs_metis_text(text: str, *, instance_id: str, source_path: str) -> dict[str, object]:
    lines = [line.strip() for line in text.splitlines() if not line.strip().startswith("%")]
    while lines and not lines[0]:
        lines.pop(0)
    if not lines:
        raise ValueError(f"Missing DFVS header in {source_path}.")
    header = lines[0].split()
    if len(header) != 3 or header[2] != "0":
        raise ValueError(f"Expected `n m 0` DFVS header in {source_path}, got {lines[0]!r}.")
    num_vertices = int(header[0])
    declared_arcs = int(header[1])
    adjacency_lines = lines[1 : 1 + num_vertices]
    if len(adjacency_lines) != num_vertices:
        raise ValueError(f"Expected {num_vertices} adjacency lines in {source_path}, got {len(adjacency_lines)}.")
    arcs: list[list[int]] = []
    for tail, line in enumerate(adjacency_lines):
        for raw_head in line.split():
            head = int(raw_head) - 1
            if not 0 <= head < num_vertices:
                raise ValueError(f"Head vertex {head + 1} outside 1..{num_vertices} in {source_path}.")
            if head == tail:
                raise ValueError(f"Self-loop at vertex {tail + 1} in {source_path}.")
            arcs.append([tail, head])
    return {
        "id": instance_id,
        "num_vertices": num_vertices,
        "arcs": arcs,
        "pace_source_path": source_path,
        "pace_declared_arcs": declared_arcs,
        "pace_actual_arcs": len(arcs),
    }


def _read_tar_member_text(path: Path, *, suffix: str) -> str:
    mode = "r:xz" if path.name.endswith(".xz") else "r:gz"
    with tarfile.open(path, mode=mode) as archive:
        members = sorted(
            [member for member in archive.getmembers() if member.isfile() and member.name.endswith(suffix)],
            key=lambda item: natural_key(item.name),
        )
        if not members:
            raise ValueError(f"No {suffix} member found in {path}.")
        handle = archive.extractfile(members[0])
        if handle is None:
            raise ValueError(f"Could not read {members[0].name} from {path}.")
        return handle.read().decode("utf-8", errors="replace")


def _hs_relative_path(source: str, index: int) -> str:
    if source == "private":
        return f"private/hs/heuristic/private_heuristic_{index:03d}.hgr.tar.xz"
    return f"hs/heuristic/heuristic_{index:03d}.hgr.tar.xz"


def _download_pace2025_relative(relative_path: str, *, cache_dir: Path, github_ref: str) -> Path:
    url = f"{PACE2025_INSTANCES_RAW}/{github_ref}/{relative_path}"
    return download_url(url, cache_dir / "pace2025-instances" / relative_path)


def _load_hs_instance(relative_path: str, *, cache_dir: Path, github_ref: str, split: str) -> dict[str, object]:
    path = _download_pace2025_relative(relative_path, cache_dir=cache_dir, github_ref=github_ref)
    text = _read_tar_member_text(path, suffix=".hgr") if path.name.endswith(".tar.xz") else path.read_text(encoding="utf-8")
    stem = Path(relative_path).name.removesuffix(".tar.xz").removesuffix(".hgr")
    return parse_hs_hgr_text(text, instance_id=f"pace2025-hs-{split}-{stem}", source_path=relative_path)


def _hitting_set_lower_bound(instance: dict[str, object]) -> int:
    num_sets = len(instance["sets"])
    if num_sets == 0:
        return 0
    incidence = [0] * int(instance["num_vertices"])
    for hyperedge in instance["sets"]:
        for vertex in hyperedge:
            incidence[int(vertex)] += 1
    return max(1, math.ceil(num_sets / max(incidence, default=1)))


def _annotate_hs(instance: dict[str, object]) -> dict[str, object]:
    problem = get_problem_definition("hitting_set")
    reference = problem.baseline_registry()["frequency_greedy"](public_instance(instance))
    annotated = dict(instance)
    annotated["optimum_objective"] = _hitting_set_lower_bound(instance)
    annotated["optimum_source"] = "pace2025_proxy_lower_bound:max_incident_sets"
    annotated["_pace_reference_objective"] = len(reference)
    annotated["_pace_reference_solution"] = reference
    annotated["_pace_reference_source"] = "frequency_greedy"
    return annotated


def _ocm_archive_name(track: str, source: str, *, solutions: bool = False) -> str:
    prefix = "cutwidth" if track == "cutwidth" else track
    if solutions:
        return f"{prefix}-{source}-sol.zip"
    return f"{prefix}-{source}.zip"


def _download_ocm_archive(track: str, source: str, *, cache_dir: Path, solutions: bool = False) -> Path:
    name = _ocm_archive_name(track, source, solutions=solutions)
    return download_url(f"{PACE2024_BASE}/{name}", cache_dir / "pace2024-ocm" / name)


def _ocm_member_index(source: str, index: int) -> int:
    return index + 100 if source == "private" and index <= 100 else index


def _read_zip_text(path: Path, member_name: str) -> str:
    with zipfile.ZipFile(path) as archive:
        with archive.open(member_name) as handle:
            return handle.read().decode("utf-8", errors="replace")


def _zip_has_member(path: Path, member_name: str) -> bool:
    with zipfile.ZipFile(path) as archive:
        return member_name in set(archive.namelist())


def parse_ocm_native_solution(text: str, instance: dict[str, object]) -> list[int]:
    values = [int(line) for line in _strip_commented_lines(text, comment_prefix="c")]
    num_fixed = int(instance["num_fixed"])
    num_free = int(instance["num_free"])
    if len(values) != num_free:
        raise ValueError(f"Expected {num_free} OCM output vertices, got {len(values)}.")
    if all(num_fixed + 1 <= value <= num_fixed + num_free for value in values):
        return [value - num_fixed - 1 for value in values]
    if sorted(values) == list(range(num_free)):
        return values
    if sorted(values) == list(range(1, num_free + 1)):
        return [value - 1 for value in values]
    raise ValueError("OCM output is not a permutation of the free-side vertices.")


def _load_ocm_instance(
    *,
    track: str,
    source: str,
    index: int,
    cache_dir: Path,
    split: str,
) -> dict[str, object]:
    graph_archive = _download_ocm_archive(track, source, cache_dir=cache_dir, solutions=False)
    archive_index = _ocm_member_index(source, index)
    member = f"{archive_index}.gr"
    instance = parse_ocm_gr_text(
        _read_zip_text(graph_archive, member),
        instance_id=f"pace2024-ocm-{track}-{split}-{index:03d}",
        source_path=f"{graph_archive.name}:{member}",
    )
    if track in {"exact", "cutwidth"}:
        solution_archive = _download_ocm_archive(track, source, cache_dir=cache_dir, solutions=True)
        solution_member = f"{archive_index}.sol"
        if _zip_has_member(solution_archive, solution_member):
            solution = parse_ocm_native_solution(_read_zip_text(solution_archive, solution_member), instance)
            instance["optimum_objective"] = crossing_count(instance, solution)
            instance["optimum_solution"] = solution
            instance["optimum_source"] = f"pace2024_{track}_{source}_solutions"
        else:
            reference = get_problem_definition("ocm").baseline_registry()["barycenter"](public_instance(instance))
            instance["optimum_objective"] = crossing_count(instance, reference)
            instance["optimum_source"] = f"pace2024_{track}_{source}_missing_solution_proxy:barycenter"
            instance["_pace_reference_objective"] = instance["optimum_objective"]
            instance["_pace_reference_solution"] = reference
    else:
        reference = get_problem_definition("ocm").baseline_registry()["barycenter"](public_instance(instance))
        instance["optimum_objective"] = crossing_count(instance, reference)
        instance["optimum_source"] = "pace2024_proxy_reference:barycenter"
        instance["_pace_reference_objective"] = instance["optimum_objective"]
        instance["_pace_reference_solution"] = reference
    return instance


def _dfvs_archive(cache_dir: Path) -> Path:
    return download_url(DFVS_HEURISTIC_ALL_URL, cache_dir / "pace2022-dfvs" / "heuristic_track_final_instances_all.tar.gz")


def _dfvs_member_names(archive_path: Path, source: str) -> list[str]:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        all_members = sorted(
            [
                member.name
                for member in archive.getmembers()
                if member.isfile() and not Path(member.name).name.startswith(".")
            ],
            key=natural_key,
        )
    parseable = [
        name
        for name in all_members
        if Path(name).suffix.lower() in {"", ".graph", ".gr", ".txt", ".metis"}
        and "solution" not in name.lower()
        and "readme" not in name.lower()
    ]
    marked = [name for name in parseable if source in name.lower()]
    if marked:
        return marked
    if len(parseable) >= 200 and source in {"public", "private"}:
        return parseable[:100] if source == "public" else parseable[-100:]
    return parseable


def _read_tar_named_member(path: Path, member_name: str) -> str:
    with tarfile.open(path, mode="r:gz") as archive:
        handle = archive.extractfile(member_name)
        if handle is None:
            raise ValueError(f"Could not read {member_name} from {path}.")
        return handle.read().decode("utf-8", errors="replace")


def _read_tar_named_members(path: Path, member_names: list[str]) -> dict[str, str]:
    wanted = set(member_names)
    if not wanted:
        return {}
    texts: dict[str, str] = {}
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive:
            if member.name not in wanted:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Could not read {member.name} from {path}.")
            texts[member.name] = handle.read().decode("utf-8", errors="replace")
            if len(texts) == len(wanted):
                break
    missing = sorted(wanted - set(texts), key=natural_key)
    if missing:
        raise ValueError(f"Missing DFVS archive members in {path}: {missing[:5]}.")
    return texts


def _load_dfvs_instance(
    *,
    member_name: str,
    archive_path: Path,
    split: str,
    text: str | None = None,
) -> dict[str, object]:
    stem = Path(member_name).name
    instance = parse_dfvs_metis_text(
        text if text is not None else _read_tar_named_member(archive_path, member_name),
        instance_id=f"pace2022-dfvs-heuristic-{split}-{stem}",
        source_path=f"{archive_path.name}:{member_name}",
    )
    lower_bound = greedy_vertex_disjoint_cycle_lower_bound(instance, max_cycles=DFVS_LOWER_BOUND_CYCLE_CAP)
    instance["optimum_objective"] = lower_bound
    instance["optimum_source"] = f"pace2022_proxy_lower_bound:greedy_vertex_disjoint_cycles_cap_{DFVS_LOWER_BOUND_CYCLE_CAP}"
    if int(instance["num_vertices"]) <= DFVS_REFERENCE_MAX_VERTICES and len(instance["arcs"]) <= DFVS_REFERENCE_MAX_ARCS:
        reference = greedy_dfvs(public_instance(instance))
        instance["_pace_reference_objective"] = len(reference)
        instance["_pace_reference_solution"] = reference
        instance["_pace_reference_source"] = "cycle_degree_greedy"
    else:
        instance["_pace_reference_objective"] = int(instance["num_vertices"])
        instance["_pace_reference_source"] = "all_vertices_trivial_large_instance"
    return instance


def _split_indices(start_index: int, count: int) -> list[int]:
    if count < 0:
        raise ValueError("Split counts must be nonnegative.")
    return list(range(start_index, start_index + count))


def _load_competition_splits(args: argparse.Namespace, config: CompetitionConfig) -> dict[str, list[dict[str, object]]]:
    splits: dict[str, list[dict[str, object]]] = {"train": [], "validation": [], "test": []}
    if config.name == "pace2025_hs":
        public_indices = _split_indices(args.public_start_index, args.train_count + args.validation_count)
        train_indices = public_indices[: args.train_count]
        validation_indices = public_indices[args.train_count :]
        test_indices = _split_indices(args.test_start_index, args.test_count)
        for split, source, indices in [
            ("train", "public", train_indices),
            ("validation", "public", validation_indices),
            ("test", args.test_source, test_indices),
        ]:
            for index in indices:
                splits[split].append(
                    _annotate_hs(
                        _load_hs_instance(
                            _hs_relative_path(source, index),
                            cache_dir=args.cache_dir,
                            github_ref=args.github_ref,
                            split=split,
                        )
                    )
                )
        return splits
    if config.name.startswith("pace2024_ocm"):
        public_indices = _split_indices(args.public_start_index, args.train_count + args.validation_count)
        train_indices = public_indices[: args.train_count]
        validation_indices = public_indices[args.train_count :]
        test_indices = _split_indices(args.test_start_index, args.test_count)
        for split, source, indices in [
            ("train", "public", train_indices),
            ("validation", "public", validation_indices),
            ("test", args.test_source, test_indices),
        ]:
            for index in indices:
                splits[split].append(
                    _load_ocm_instance(
                        track=config.track,
                        source=source,
                        index=index,
                        cache_dir=args.cache_dir,
                        split=split,
                    )
                )
        return splits
    if config.name == "pace2022_dfvs_heuristic":
        archive_path = _dfvs_archive(args.cache_dir)
        public_members = _dfvs_member_names(archive_path, "public")
        private_members = _dfvs_member_names(archive_path, "private" if args.test_source == "private" else "public")
        public_selected = public_members[args.public_start_index - 1 : args.public_start_index - 1 + args.train_count + args.validation_count]
        test_selected = private_members[args.test_start_index - 1 : args.test_start_index - 1 + args.test_count]
        split_members = [
            ("train", public_selected[: args.train_count]),
            ("validation", public_selected[args.train_count :]),
            ("test", test_selected),
        ]
        selected_texts = _read_tar_named_members(
            archive_path,
            [member_name for _, members in split_members for member_name in members],
        )
        for split, members in split_members:
            for member_name in members:
                splits[split].append(
                    _load_dfvs_instance(
                        member_name=member_name,
                        archive_path=archive_path,
                        split=split,
                        text=selected_texts[member_name],
                    )
                )
        return splits
    raise ValueError(f"Unsupported competition {config.name!r}.")


def _manifest(
    *,
    dataset_dir: Path,
    output_root: Path,
    config: CompetitionConfig,
    split_sizes: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "problem": config.problem,
        "family": config.family if args.test_source == "private" else config.family.replace("_private", "_public"),
        "description": config.description,
        "ground_truth_hidden_rule": {
            "source": config.name,
            "track": config.track,
            "test_source": args.test_source,
        },
        "metric_definition": {
            "primary": "normalized_quality",
            "secondary": "feasibility_rate",
            "tertiary": "average_runtime_ms",
            "notes": config.metric_notes,
        },
        "instance_schema_version": config.instance_schema_version,
        "compute_optima": False,
        "instance_params": {
            "source": config.name,
            "track": config.track,
            "test_source": args.test_source,
            "github_ref": args.github_ref if config.name == "pace2025_hs" else None,
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
    config: CompetitionConfig,
    args: argparse.Namespace,
) -> dict[str, object]:
    problem = get_problem_definition(config.problem)
    splits = _load_competition_splits(args, config)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        for instance in rows:
            problem.validate_instance(instance)
        write_jsonl(dataset_dir / f"{split}.jsonl", rows)
    split_sizes = {split: len(rows) for split, rows in splits.items()}
    manifest = _manifest(dataset_dir=dataset_dir, output_root=output_root, config=config, split_sizes=split_sizes, args=args)
    write_json(dataset_dir / "manifest.json", manifest)
    repro = {
        "schema_version": "pace_competition_dataset.v1",
        "competition": config.name,
        "track": config.track,
        "test_source": args.test_source,
        "train_count": args.train_count,
        "validation_count": args.validation_count,
        "test_count": args.test_count,
        "public_start_index": args.public_start_index,
        "test_start_index": args.test_start_index,
        "cache_dir": str(args.cache_dir),
        "github_ref": args.github_ref,
    }
    write_json(dataset_dir / "benchmark_spec.json", repro)
    write_json(dataset_dir / "reproducibility.json", repro)
    return manifest


def native_input_text(config: CompetitionConfig, instance: dict[str, object]) -> str:
    if config.problem == "hitting_set":
        lines = [f"p hs {instance['num_vertices']} {len(instance['sets'])}"]
        lines.extend(" ".join(str(int(vertex) + 1) for vertex in hyperedge) for hyperedge in instance["sets"])
        return "\n".join(lines) + "\n"
    if config.problem == "ocm":
        num_fixed = int(instance["num_fixed"])
        num_free = int(instance["num_free"])
        if "cutwidth" in instance:
            lines = [f"p ocr {num_fixed} {num_free} {len(instance['edges'])} {instance['cutwidth']}"]
            lines.extend(str(int(vertex) + 1) for vertex in instance.get("cutwidth_order", []))
        else:
            lines = [f"p ocr {num_fixed} {num_free} {len(instance['edges'])}"]
        lines.extend(f"{int(fixed) + 1} {num_fixed + int(free) + 1}" for fixed, free in instance["edges"])
        return "\n".join(lines) + "\n"
    if config.problem == "dfvs":
        adjacency = [[] for _ in range(int(instance["num_vertices"]))]
        for tail, head in instance["arcs"]:
            adjacency[int(tail)].append(int(head) + 1)
        lines = [f"{instance['num_vertices']} {len(instance['arcs'])} 0"]
        lines.extend(" ".join(str(head) for head in sorted(heads)) for heads in adjacency)
        return "\n".join(lines) + "\n"
    raise ValueError(f"No native input writer for {config.problem!r}.")


def parse_native_solution(config: CompetitionConfig, text: str, instance: dict[str, object]) -> list[int]:
    if config.problem == "hitting_set":
        values = [int(line) for line in _strip_commented_lines(text, comment_prefix="c")]
        if not values:
            raise ValueError("Hitting Set solver produced no output.")
        declared = values[0]
        vertices = values[1:]
        if declared != len(vertices):
            raise ValueError(f"Declared solution size {declared}, but listed {len(vertices)} vertices.")
        return [vertex - 1 for vertex in vertices]
    if config.problem == "ocm":
        return parse_ocm_native_solution(text, instance)
    if config.problem == "dfvs":
        values: list[int] = []
        for line in _strip_commented_lines(text, comment_prefix="%"):
            try:
                values.append(int(line))
            except ValueError:
                continue
        if not values:
            return []
        if all(1 <= value <= int(instance["num_vertices"]) for value in values):
            return [value - 1 for value in values]
        if all(0 <= value < int(instance["num_vertices"]) for value in values):
            return values
        raise ValueError("DFVS output contains vertices outside the valid range.")
    raise ValueError(f"No native solution parser for {config.problem!r}.")


def write_native_solution(config: CompetitionConfig, solution: list[int], instance: dict[str, object]) -> str:
    if config.problem == "hitting_set":
        return "\n".join([str(len(solution)), *(str(vertex + 1) for vertex in solution)]) + "\n"
    if config.problem == "ocm":
        num_fixed = int(instance["num_fixed"])
        return "\n".join(str(num_fixed + vertex + 1) for vertex in solution) + "\n"
    if config.problem == "dfvs":
        return "\n".join(str(vertex + 1) for vertex in solution) + ("\n" if solution else "")
    raise ValueError(f"No native solution writer for {config.problem!r}.")


BASELINE_FIELDNAMES = [
    "solver",
    "instance_id",
    "pace_source_path",
    "exit_code",
    "timed_out",
    "valid",
    "objective_value",
    "normalized_quality",
    "runtime_ms",
    "solution_file",
    "stderr_file",
    "error",
]


def run_baselines(
    *,
    config: CompetitionConfig,
    dataset_dir: Path,
    solvers: list[SolverSpec],
    output_dir: Path,
    timeout_seconds: float,
    grace_seconds: float,
) -> dict[str, object]:
    problem = get_problem_definition(config.problem)
    instances = load_split(dataset_dir, "test")
    rows: list[dict[str, object]] = []
    metadata = {solver.name: solver.metadata or {} for solver in solvers}
    for instance in tqdm(instances, desc=f"{config.name} baselines", unit="instance", dynamic_ncols=True):
        input_path = output_dir / "inputs" / f"{instance['id']}.txt"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text(native_input_text(config, instance), encoding="utf-8")
        for solver in solvers:
            result = run_process(
                solver.command,
                input_path,
                timeout_seconds=timeout_seconds,
                grace_seconds=grace_seconds,
                cwd=solver.cwd,
            )
            solution_file = output_dir / "solutions" / solver.name / f"{instance['id']}.sol"
            stderr_file = output_dir / "stderr" / solver.name / f"{instance['id']}.stderr.txt"
            solution_file.parent.mkdir(parents=True, exist_ok=True)
            stderr_file.parent.mkdir(parents=True, exist_ok=True)
            solution_file.write_text(result.stdout, encoding="utf-8")
            stderr_file.write_text(result.stderr, encoding="utf-8")
            try:
                raw_solution = parse_native_solution(config, result.stdout, instance)
                solution = problem.canonicalize_solution(raw_solution, public_instance(instance))
                score = problem.score_solution(instance, solution)
                error = score.error or ""
            except Exception as exc:
                solution = []
                score = None
                error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "solver": solver.name,
                    "instance_id": instance["id"],
                    "pace_source_path": instance.get("pace_source_path", ""),
                    "exit_code": "" if result.exit_code is None else result.exit_code,
                    "timed_out": result.timed_out,
                    "valid": bool(score and score.is_feasible),
                    "objective_value": score.objective_value if score and score.is_feasible else "",
                    "normalized_quality": score.normalized_quality if score and score.is_feasible else 0.0,
                    "runtime_ms": result.runtime_ms,
                    "solution_file": str(solution_file),
                    "stderr_file": str(stderr_file),
                    "error": error,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "baseline_results.csv", rows, fieldnames=BASELINE_FIELDNAMES)
    summary = _baseline_summary(rows, config=config, dataset_dir=dataset_dir, metadata=metadata)
    write_json(output_dir / "baseline_summary.json", summary)
    _write_baseline_report(output_dir / "baseline_report.md", summary)
    return summary


def _baseline_summary(
    rows: list[dict[str, object]],
    *,
    config: CompetitionConfig,
    dataset_dir: Path,
    metadata: dict[str, object],
) -> dict[str, object]:
    by_solver: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_solver.setdefault(str(row["solver"]), []).append(row)
    solvers: dict[str, dict[str, object]] = {}
    for solver, solver_rows in sorted(by_solver.items()):
        valid_rows = [row for row in solver_rows if row["valid"]]
        solvers[solver] = {
            "num_instances": len(solver_rows),
            "valid_count": len(valid_rows),
            "invalid_count": len(solver_rows) - len(valid_rows),
            "timeout_count": sum(1 for row in solver_rows if row["timed_out"]),
            "average_objective_value": (
                sum(float(row["objective_value"]) for row in valid_rows) / len(valid_rows) if valid_rows else 0.0
            ),
            "average_normalized_quality": (
                sum(float(row["normalized_quality"]) for row in valid_rows) / len(valid_rows) if valid_rows else 0.0
            ),
            "average_runtime_ms": (
                sum(float(row["runtime_ms"]) for row in solver_rows) / len(solver_rows) if solver_rows else 0.0
            ),
        }
    return {
        "schema_version": "pace_competition_baselines.v1",
        "competition": config.name,
        "problem": config.problem,
        "dataset_dir": str(dataset_dir),
        "solvers": solvers,
        "solver_metadata": metadata,
    }


def _write_baseline_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        f"# PACE Baseline Report: {summary['competition']}",
        "",
        f"- Dataset: `{summary['dataset_dir']}`",
        "",
        "| Solver | Valid | Invalid | Timeouts | Avg Objective | Avg Quality | Avg Runtime ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for solver, payload in sorted(summary["solvers"].items()):
        lines.append(
            "| "
            f"{solver} | {payload['valid_count']} | {payload['invalid_count']} | {payload['timeout_count']} | "
            f"{float(payload['average_objective_value']):.6f} | "
            f"{float(payload['average_normalized_quality']):.6f} | "
            f"{float(payload['average_runtime_ms']):.3f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _installed_builtin_specs(config: CompetitionConfig, solver_root: Path) -> dict[str, SolverSpec]:
    bin_dir = solver_root / "bin"
    specs: dict[str, SolverSpec] = {}
    for name, recipe in SOLVER_RECIPES.items():
        if _recipe_matches(config, recipe):
            specs[name] = SolverSpec(name=name, command=[str(bin_dir / name)], metadata={"recipe": name})
    return specs


def _recipe_matches(config: CompetitionConfig, recipe: SolverRecipe) -> bool:
    if recipe.competition == config.name:
        return True
    return recipe.competition == "pace2024_ocm" and config.name.startswith("pace2024_ocm")


def _solver_names_from_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _install_solvers(config: CompetitionConfig, names: list[str], *, solver_root: Path, force_build: bool) -> list[SolverSpec]:
    specs: list[SolverSpec] = []
    for name in names:
        recipe = SOLVER_RECIPES[name]
        if not _recipe_matches(config, recipe):
            raise ValueError(f"Solver recipe `{name}` is for {recipe.competition}, not {config.name}.")
        specs.append(
            clone_and_build_recipe(
                recipe,
                source_root=solver_root / "src",
                bin_dir=solver_root / "bin",
                force_build=force_build,
            )
        )
    return specs


def _resolve_solvers(config: CompetitionConfig, args: argparse.Namespace) -> list[SolverSpec]:
    solver_texts = list(args.solver)
    if not solver_texts:
        solver_texts = _solver_names_from_arg(args.solvers or DEFAULT_SOLVERS[config.name])
    if args.install_solvers:
        builtin_names = [text for text in solver_texts if "=" not in text]
        custom_specs = [text for text in solver_texts if "=" in text]
        specs = _install_solvers(config, builtin_names, solver_root=args.solver_root, force_build=args.force_build)
        installed = {spec.name: spec for spec in specs}
        installed.update(_installed_builtin_specs(config, args.solver_root))
        specs.extend(parse_solver_argument(text, installed) for text in custom_specs)
        return specs
    installed = _installed_builtin_specs(config, args.solver_root)
    return [parse_solver_argument(text, installed) for text in solver_texts]


def export_agent_solutions(
    *,
    config: CompetitionConfig,
    dataset_dir: Path,
    agent_run_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    manifest = load_manifest(dataset_dir)
    synthesis_summary = json.loads((agent_run_dir / "synthesis_summary.json").read_text(encoding="utf-8"))
    best_candidate = synthesis_summary["best_candidate"]
    candidate_dir = Path(best_candidate["candidate_dir"])
    solution_path = candidate_dir / "solution.py"
    solution_dir = output_dir / "solutions"
    if solution_dir.exists():
        shutil.rmtree(solution_dir)
    rows: list[dict[str, object]] = []
    if not solution_path.exists():
        error = f"Selected candidate `{best_candidate['slug']}` has no solution.py."
        for instance in load_jsonl(dataset_dir / "test.jsonl"):
            rows.append(_agent_failure_row(instance, error))
        return _write_agent_eval(config=config, dataset_dir=dataset_dir, output_dir=output_dir, rows=rows, error=error)
    try:
        analysis = run_analysis(candidate_dir, load_split(dataset_dir, "train", public=True), manifest=manifest, artifact_dir=output_dir / "analysis")
        solver = build_solver(candidate_dir, analysis=analysis, manifest=manifest)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        for instance in load_jsonl(dataset_dir / "test.jsonl"):
            rows.append(_agent_failure_row(instance, error))
        return _write_agent_eval(config=config, dataset_dir=dataset_dir, output_dir=output_dir, rows=rows, error=error)
    problem = get_problem_definition(config.problem)
    for instance in load_split(dataset_dir, "test"):
        start = time.perf_counter()
        try:
            exposed_instance = public_instance(instance)
            with _solver_timeout(
                None,
                name=str(best_candidate["slug"]),
                split="pace_export",
                instance_id=instance.get("id"),
            ):
                raw_solution = solver(exposed_instance)
                solution = problem.canonicalize_solution(raw_solution, exposed_instance)
                score = problem.score_solution(instance, solution)
            runtime_ms = (time.perf_counter() - start) * 1000.0
            error = score.error or ""
        except SolverTimeoutError as exc:
            runtime_ms = (time.perf_counter() - start) * 1000.0
            solution = []
            score = None
            error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            runtime_ms = (time.perf_counter() - start) * 1000.0
            solution = []
            score = None
            error = f"{type(exc).__name__}: {exc}"
        solution_file = solution_dir / f"{instance['id']}.sol"
        if score and score.is_feasible:
            solution_file.parent.mkdir(parents=True, exist_ok=True)
            solution_file.write_text(write_native_solution(config, solution, instance), encoding="utf-8")
        rows.append(
            {
                "instance_id": instance["id"],
                "pace_source_path": instance.get("pace_source_path", ""),
                "feasible": bool(score and score.is_feasible),
                "objective_value": score.objective_value if score and score.is_feasible else "",
                "normalized_quality": score.normalized_quality if score and score.is_feasible else 0.0,
                "runtime_ms": runtime_ms,
                "solution_file": str(solution_file) if score and score.is_feasible else "",
                "error": error,
            }
        )
    return _write_agent_eval(config=config, dataset_dir=dataset_dir, output_dir=output_dir, rows=rows)


def _agent_failure_row(instance: dict[str, object], error: str) -> dict[str, object]:
    return {
        "instance_id": instance["id"],
        "pace_source_path": instance.get("pace_source_path", ""),
        "feasible": False,
        "objective_value": "",
        "normalized_quality": 0.0,
        "runtime_ms": 0.0,
        "solution_file": "",
        "error": error,
    }


AGENT_FIELDNAMES = [
    "instance_id",
    "pace_source_path",
    "feasible",
    "objective_value",
    "normalized_quality",
    "runtime_ms",
    "solution_file",
    "error",
]


def _write_agent_eval(
    *,
    config: CompetitionConfig,
    dataset_dir: Path,
    output_dir: Path,
    rows: list[dict[str, object]],
    error: str | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "pace_results.csv", rows, fieldnames=AGENT_FIELDNAMES)
    feasible = [row for row in rows if row["feasible"]]
    timeout_count = sum(1 for row in rows if "SolverTimeoutError" in str(row.get("error", "")))
    summary = {
        "schema_version": "pace_competition_agent_eval.v1",
        "competition": config.name,
        "problem": config.problem,
        "dataset_dir": str(dataset_dir),
        "num_instances": len(rows),
        "feasible_count": len(feasible),
        "invalid_count": len(rows) - len(feasible),
        "timeout_count": timeout_count,
        "solver_timeout_seconds": _resolved_solver_timeout(None),
        "average_objective_value": (
            sum(float(row["objective_value"]) for row in feasible) / len(feasible) if feasible else 0.0
        ),
        "average_normalized_quality": (
            sum(float(row["normalized_quality"]) for row in feasible) / len(feasible) if feasible else 0.0
        ),
        "average_runtime_ms": sum(float(row["runtime_ms"]) for row in rows) / len(rows) if rows else 0.0,
    }
    if error:
        summary["error"] = error
    write_json(output_dir / "pace_evaluation_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and run PACE competition benchmarks in DasBench format.")
    parser.add_argument("--competition", choices=sorted(COMPETITIONS), required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--github-ref", default="master")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-id")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--run-output-dir", type=Path)
    parser.add_argument("--evaluation-output-dir", type=Path)
    parser.add_argument("--test-source", choices=["private", "public"], default="private")
    parser.add_argument("--train-count", type=int, default=5)
    parser.add_argument("--validation-count", type=int, default=5)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--public-start-index", type=int, default=1)
    parser.add_argument("--test-start-index", type=int, default=1)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--generator", choices=["auto", "template", "llm", "agent"], default="auto")
    parser.add_argument("--mode", choices=["single", "beam"], default="beam")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--candidate-width", type=int, default=3)
    parser.add_argument("--run-id")
    parser.add_argument("--skip-baselines", action="store_true", default=True)
    parser.add_argument("--no-skip-baselines", dest="skip_baselines", action="store_false")
    parser.add_argument("--no-export-solutions", dest="export_solutions", action="store_false")
    parser.add_argument("--run-baselines", action="store_true")
    parser.add_argument("--install-solvers", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--solver-root", type=Path, default=DEFAULT_SOLVER_ROOT)
    parser.add_argument("--solver", action="append", default=[], help="Built-in solver name, or NAME='command args'.")
    parser.add_argument("--solvers", help="Comma-separated built-in solver names.")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--grace-seconds", type=float, default=30.0)
    parser.set_defaults(export_solutions=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_openai_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    config = COMPETITIONS[args.competition]
    experiment_id = args.experiment_id or f"{config.name}_{timestamp_token()}"
    output_root = args.output_root / experiment_id
    dataset_dir = args.dataset_dir or output_root / "dataset"
    run_output_dir = args.run_output_dir or output_root / "agent_run"
    evaluation_output_dir = args.evaluation_output_dir or output_root / "pace_evaluation"

    if args.install_solvers and not args.run_baselines:
        solvers = _resolve_solvers(config, args)
        print("Installed solver shims:")
        for solver in solvers:
            print(f"- {solver.name}: {' '.join(solver.command)}")
        return 0

    manifest = build_pace_dataset(dataset_dir=dataset_dir, output_root=output_root, config=config, args=args)
    print(f"PACE dataset: {dataset_dir}")
    print(json.dumps({"problem": manifest["problem"], "family": manifest["family"], "split_sizes": manifest["split_sizes"]}, indent=2, sort_keys=True))

    if args.run_baselines:
        solvers = _resolve_solvers(config, args)
        summary = run_baselines(
            config=config,
            dataset_dir=dataset_dir,
            solvers=solvers,
            output_dir=output_root / "baseline_comparisons",
            timeout_seconds=args.timeout_seconds,
            grace_seconds=args.grace_seconds,
        )
        print(f"Baseline summary: {output_root / 'baseline_comparisons' / 'baseline_summary.json'}")
        print(json.dumps(summary["solvers"], indent=2, sort_keys=True))

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
        summary = export_agent_solutions(
            config=config,
            dataset_dir=dataset_dir,
            agent_run_dir=run_output_dir,
            output_dir=evaluation_output_dir,
        )
        print(f"PACE evaluation summary: {evaluation_output_dir / 'pace_evaluation_summary.json'}")
        print(
            f"PACE eval: feasible={summary['feasible_count']}/{summary['num_instances']} "
            f"avg_objective={summary['average_objective_value']:.3f} "
            f"avg_quality={summary['average_normalized_quality']:.6f} "
            f"avg_runtime_ms={summary['average_runtime_ms']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
