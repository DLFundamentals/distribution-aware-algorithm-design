from __future__ import annotations

import statistics
import time
from itertools import permutations

from dasbench.problems.base import ExactSolveResult, ProblemDefinition, ScoreResult


def validate_instance(instance: dict[str, object]) -> None:
    num_fixed = int(instance["num_fixed"])
    num_free = int(instance["num_free"])
    if num_fixed < 0 or num_free < 0:
        raise ValueError("OCM instances require nonnegative partition sizes.")
    raw_edges = instance.get("edges")
    if not isinstance(raw_edges, list):
        raise ValueError("OCM instances require an edge list.")
    seen: set[tuple[int, int]] = set()
    for edge in raw_edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError(f"OCM edge must be [fixed_vertex, free_vertex], got {edge!r}.")
        fixed = int(edge[0])
        free = int(edge[1])
        if not 0 <= fixed < num_fixed:
            raise ValueError(f"Fixed vertex {fixed} is outside 0..{num_fixed - 1}.")
        if not 0 <= free < num_free:
            raise ValueError(f"Free vertex {free} is outside 0..{num_free - 1}.")
        key = (fixed, free)
        if key in seen:
            raise ValueError(f"OCM edge {key} is repeated.")
        seen.add(key)


def canonicalize_solution(raw_solution, instance: dict[str, object]) -> list[int]:
    if not isinstance(raw_solution, (list, tuple)):
        raise TypeError("OCM solver output must be a sequence of free-side vertex ids.")
    return [int(value) for value in raw_solution]


def validate_solution(solution: list[int], instance: dict[str, object]) -> tuple[bool, str | None]:
    num_free = int(instance["num_free"])
    if len(solution) != num_free:
        return False, f"Expected {num_free} free vertices, got {len(solution)}."
    seen: set[int] = set()
    for vertex in solution:
        if not 0 <= int(vertex) < num_free:
            return False, f"Free vertex {vertex} is outside 0..{num_free - 1}."
        if int(vertex) in seen:
            return False, f"Free vertex {vertex} is repeated."
        seen.add(int(vertex))
    return True, None


class _Fenwick:
    def __init__(self, size: int) -> None:
        self.values = [0] * (size + 1)

    def add(self, index: int, value: int) -> None:
        index += 1
        while index < len(self.values):
            self.values[index] += value
            index += index & -index

    def prefix_sum(self, index: int) -> int:
        total = 0
        index += 1
        while index > 0:
            total += self.values[index]
            index -= index & -index
        return total


def crossing_count(instance: dict[str, object], order: list[int]) -> int:
    valid, error = validate_solution(order, instance)
    if not valid:
        raise ValueError(error or "Invalid OCM order.")
    position = {vertex: index for index, vertex in enumerate(order)}
    grouped: dict[int, list[int]] = {}
    for fixed, free in instance["edges"]:
        grouped.setdefault(int(fixed), []).append(position[int(free)])
    tree = _Fenwick(int(instance["num_free"]))
    previous_edges = 0
    crossings = 0
    for fixed in sorted(grouped):
        positions = grouped[fixed]
        for pos in positions:
            crossings += previous_edges - tree.prefix_sum(pos)
        for pos in positions:
            tree.add(pos, 1)
            previous_edges += 1
    return crossings


def _minimization_quality(optimum: float, objective: float) -> float:
    if optimum <= 0:
        return 1.0 if objective <= 0 else 0.0
    if objective <= 0:
        return 1.0
    return min(1.0, max(0.0, optimum / objective))


def score_solution(instance: dict[str, object], solution: list[int]) -> ScoreResult:
    valid, error = validate_solution(solution, instance)
    if not valid:
        return ScoreResult(False, False, 0.0, 0.0, False, error)
    objective = float(crossing_count(instance, solution))
    optimum = float(instance["optimum_objective"])
    return ScoreResult(
        is_valid=True,
        is_feasible=True,
        objective_value=objective,
        normalized_quality=_minimization_quality(optimum, objective),
        is_optimal=abs(objective - optimum) < 1e-9,
    )


def _barycenter_order(instance: dict[str, object]) -> list[int]:
    incident: dict[int, list[int]] = {vertex: [] for vertex in range(int(instance["num_free"]))}
    for fixed, free in instance["edges"]:
        incident[int(free)].append(int(fixed))
    return sorted(
        range(int(instance["num_free"])),
        key=lambda free: (
            statistics.mean(incident[free]) if incident[free] else -1.0,
            free,
        ),
    )


def solve_exact(instance: dict[str, object]) -> ExactSolveResult:
    start = time.perf_counter()
    num_free = int(instance["num_free"])
    if num_free > 9:
        raise RuntimeError("Built-in OCM exact solver is limited to 9 free vertices.")
    best_order: list[int] | None = None
    best_crossings: int | None = None
    for order_tuple in permutations(range(num_free)):
        order = list(order_tuple)
        crossings = crossing_count(instance, order)
        if best_crossings is None or crossings < best_crossings:
            best_crossings = crossings
            best_order = order
    if best_order is None or best_crossings is None:
        best_order = []
        best_crossings = 0
    return ExactSolveResult(
        solution=best_order,
        objective_value=float(best_crossings),
        runtime_ms=(time.perf_counter() - start) * 1000.0,
        source="ocm_bruteforce_exact",
    )


def summarize_training_data(train_instances: list[dict[str, object]], manifest: dict[str, object]) -> dict[str, object]:
    fixed_counts = [int(instance["num_fixed"]) for instance in train_instances]
    free_counts = [int(instance["num_free"]) for instance in train_instances]
    edge_counts = [len(instance["edges"]) for instance in train_instances]
    return {
        "problem": manifest["problem"],
        "family": manifest["family"],
        "num_instances": len(train_instances),
        "num_fixed_mean": round(statistics.mean(fixed_counts), 2) if fixed_counts else 0.0,
        "num_free_mean": round(statistics.mean(free_counts), 2) if free_counts else 0.0,
        "num_edges_mean": round(statistics.mean(edge_counts), 2) if edge_counts else 0.0,
        "sample_instances": [
            {
                "id": instance["id"],
                "num_fixed": instance["num_fixed"],
                "num_free": instance["num_free"],
                "num_edges": len(instance["edges"]),
                "edge_prefix": instance["edges"][:10],
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
    return {
        "instance_id": instance["id"],
        "normalized_quality": score.normalized_quality,
        "objective_value": score.objective_value,
        "runtime_ms": runtime_seconds * 1000.0,
        "is_optimal": score.is_optimal,
        "order_prefix": solution[:16],
        "error": score.error,
    }


def baseline_registry() -> dict[str, object]:
    return {
        "barycenter": _barycenter_order,
        "exact": lambda instance: solve_exact(instance).solution,
    }


PROBLEM = ProblemDefinition(
    name="ocm",
    description="One-sided crossing minimization for a bipartite graph with one fixed side.",
    metric_definition={
        "primary": "normalized_quality",
        "secondary": "optimality_rate",
        "tertiary": "average_runtime_ms",
        "notes": "normalized_quality is optimum_or_reference_crossings / returned_crossings",
    },
    instance_schema_version="ocm.v1",
    default_instance_params={"num_fixed": 16, "num_free": 16},
    validate_instance=validate_instance,
    canonicalize_solution=canonicalize_solution,
    validate_solution=validate_solution,
    score_solution=score_solution,
    summarize_training_data=summarize_training_data,
    failure_case=failure_case,
    baseline_registry=baseline_registry,
    exact_solver=solve_exact,
)
