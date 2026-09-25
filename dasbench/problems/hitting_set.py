from __future__ import annotations

import math
import statistics
import time
import heapq
from itertools import combinations
from typing import Iterable

from dasbench.problems.base import ExactSolveResult, ProblemDefinition, ScoreResult


def _sets(instance: dict[str, object]) -> list[list[int]]:
    return [[int(value) for value in hyperedge] for hyperedge in instance["sets"]]


def validate_instance(instance: dict[str, object]) -> None:
    num_vertices = int(instance["num_vertices"])
    if num_vertices < 0:
        raise ValueError("Hitting Set instances require a nonnegative number of vertices.")
    raw_sets = instance.get("sets")
    if not isinstance(raw_sets, list):
        raise ValueError("Hitting Set instances require a list of sets.")
    for set_index, raw_set in enumerate(raw_sets):
        if not isinstance(raw_set, list):
            raise ValueError(f"Hitting Set set {set_index} must be a list of vertices.")
        if not raw_set:
            raise ValueError(f"Hitting Set set {set_index} is empty.")
        seen: set[int] = set()
        for raw_vertex in raw_set:
            vertex = int(raw_vertex)
            if not 0 <= vertex < num_vertices:
                raise ValueError(f"Vertex {vertex} in set {set_index} is outside 0..{num_vertices - 1}.")
            if vertex in seen:
                raise ValueError(f"Vertex {vertex} is repeated in set {set_index}.")
            seen.add(vertex)


def canonicalize_solution(raw_solution, instance: dict[str, object]) -> list[int]:
    if not isinstance(raw_solution, (list, tuple, set)):
        raise TypeError("Hitting Set solver output must be a sequence of vertex ids.")
    return sorted(int(value) for value in raw_solution)


def validate_solution(solution: list[int], instance: dict[str, object]) -> tuple[bool, str | None]:
    num_vertices = int(instance["num_vertices"])
    selected: set[int] = set()
    for vertex in solution:
        if not 0 <= int(vertex) < num_vertices:
            return False, f"Vertex {vertex} is outside 0..{num_vertices - 1}."
        if int(vertex) in selected:
            return False, f"Vertex {vertex} is repeated."
        selected.add(int(vertex))
    for set_index, hyperedge in enumerate(_sets(instance)):
        if selected.isdisjoint(hyperedge):
            return False, f"Set {set_index} is not hit."
    return True, None


def _minimization_quality(optimum: float, objective: float) -> float:
    if objective <= 0:
        return 1.0 if optimum <= 0 else 0.0
    return min(1.0, max(0.0, optimum / objective))


def score_solution(instance: dict[str, object], solution: list[int]) -> ScoreResult:
    valid, error = validate_solution(solution, instance)
    if not valid:
        return ScoreResult(False, False, 0.0, 0.0, False, error)
    objective = float(len(solution))
    optimum = float(instance["optimum_objective"])
    return ScoreResult(
        is_valid=True,
        is_feasible=True,
        objective_value=objective,
        normalized_quality=_minimization_quality(optimum, objective),
        is_optimal=abs(objective - optimum) < 1e-9,
    )


def _incidence(instance: dict[str, object]) -> list[list[int]]:
    incidence: list[list[int]] = [[] for _ in range(int(instance["num_vertices"]))]
    for set_index, hyperedge in enumerate(_sets(instance)):
        for vertex in hyperedge:
            incidence[vertex].append(set_index)
    return incidence


def _greedy_solution(instance: dict[str, object]) -> list[int]:
    incidence = _incidence(instance)
    uncovered = set(range(len(instance["sets"])))
    heap = [(-len(incident), vertex) for vertex, incident in enumerate(incidence) if incident]
    heapq.heapify(heap)
    selected: set[int] = set()
    while uncovered and heap:
        neg_gain, vertex = heapq.heappop(heap)
        gain = sum(1 for set_index in incidence[vertex] if set_index in uncovered)
        if gain == 0:
            continue
        if gain != -neg_gain:
            heapq.heappush(heap, (-gain, vertex))
            continue
        selected.add(vertex)
        for set_index in incidence[vertex]:
            uncovered.discard(set_index)
    return _prune_redundant_fast(instance, selected, incidence)


def _prune_redundant(instance: dict[str, object], selected: Iterable[int]) -> list[int]:
    solution = sorted(set(int(vertex) for vertex in selected))
    for vertex in list(solution):
        candidate = [item for item in solution if item != vertex]
        valid, _ = validate_solution(candidate, instance)
        if valid:
            solution = candidate
    return solution


def _prune_redundant_fast(
    instance: dict[str, object],
    selected: Iterable[int],
    incidence: list[list[int]] | None = None,
) -> list[int]:
    selected_set = set(int(vertex) for vertex in selected)
    hit_count = [0] * len(instance["sets"])
    resolved_incidence = incidence if incidence is not None else _incidence(instance)
    for vertex in selected_set:
        for set_index in resolved_incidence[vertex]:
            hit_count[set_index] += 1
    for vertex in sorted(list(selected_set)):
        if all(hit_count[set_index] > 1 for set_index in resolved_incidence[vertex]):
            selected_set.remove(vertex)
            for set_index in resolved_incidence[vertex]:
                hit_count[set_index] -= 1
    return sorted(selected_set)


def solve_exact(instance: dict[str, object]) -> ExactSolveResult:
    start = time.perf_counter()
    num_vertices = int(instance["num_vertices"])
    if num_vertices > 26:
        raise RuntimeError("Built-in Hitting Set exact solver is limited to 26 vertices.")
    for size in range(num_vertices + 1):
        for candidate in combinations(range(num_vertices), size):
            solution = list(candidate)
            valid, _ = validate_solution(solution, instance)
            if valid:
                return ExactSolveResult(
                    solution=solution,
                    objective_value=float(size),
                    runtime_ms=(time.perf_counter() - start) * 1000.0,
                    source="hitting_set_bruteforce_exact",
                )
    raise RuntimeError("No hitting set found.")


def summarize_training_data(train_instances: list[dict[str, object]], manifest: dict[str, object]) -> dict[str, object]:
    set_counts = [len(instance["sets"]) for instance in train_instances]
    vertex_counts = [int(instance["num_vertices"]) for instance in train_instances]
    set_sizes = [len(edge) for instance in train_instances for edge in _sets(instance)]
    return {
        "problem": manifest["problem"],
        "family": manifest["family"],
        "num_instances": len(train_instances),
        "num_vertices_mean": round(statistics.mean(vertex_counts), 2) if vertex_counts else 0.0,
        "num_sets_mean": round(statistics.mean(set_counts), 2) if set_counts else 0.0,
        "set_size_mean": round(statistics.mean(set_sizes), 2) if set_sizes else 0.0,
        "set_size_max": max(set_sizes) if set_sizes else 0,
        "sample_instances": [
            {
                "id": instance["id"],
                "num_vertices": instance["num_vertices"],
                "num_sets": len(instance["sets"]),
                "set_prefix": instance["sets"][:6],
            }
            for instance in train_instances[:3]
        ],
    }


def failure_case(
    instance: dict[str, object],
    solution: list[int],
    score: ScoreResult,
    runtime_seconds: float,
) -> dict[str, object]:
    selected = set(solution)
    missed = [index for index, hyperedge in enumerate(_sets(instance)) if selected.isdisjoint(hyperedge)]
    return {
        "instance_id": instance["id"],
        "normalized_quality": score.normalized_quality,
        "objective_value": score.objective_value,
        "runtime_ms": runtime_seconds * 1000.0,
        "is_optimal": score.is_optimal,
        "solution_prefix": solution[:12],
        "missed_sets": missed[:6],
        "error": score.error,
    }


def baseline_registry() -> dict[str, object]:
    return {
        "frequency_greedy": _greedy_solution,
        "exact": lambda instance: solve_exact(instance).solution,
    }


PROBLEM = ProblemDefinition(
    name="hitting_set",
    description="PACE-style minimum hitting set over an explicit hypergraph/set system.",
    metric_definition={
        "primary": "normalized_quality",
        "secondary": "optimality_rate",
        "tertiary": "average_runtime_ms",
        "notes": "normalized_quality is optimum_or_proxy_hitting_set_size / returned_hitting_set_size",
    },
    instance_schema_version="hitting_set.v1",
    default_instance_params={"num_vertices": 24},
    validate_instance=validate_instance,
    canonicalize_solution=canonicalize_solution,
    validate_solution=validate_solution,
    score_solution=score_solution,
    summarize_training_data=summarize_training_data,
    failure_case=failure_case,
    baseline_registry=baseline_registry,
    exact_solver=solve_exact,
)
