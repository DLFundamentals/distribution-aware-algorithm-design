from __future__ import annotations

import statistics
import time
from collections import deque
from itertools import combinations

from dasbench.problems.base import ExactSolveResult, ProblemDefinition, ScoreResult


def validate_instance(instance: dict[str, object]) -> None:
    num_vertices = int(instance["num_vertices"])
    if num_vertices < 0:
        raise ValueError("DFVS instances require a nonnegative number of vertices.")
    raw_arcs = instance.get("arcs")
    if not isinstance(raw_arcs, list):
        raise ValueError("DFVS instances require a directed arc list.")
    seen: set[tuple[int, int]] = set()
    for arc in raw_arcs:
        if not isinstance(arc, list) or len(arc) != 2:
            raise ValueError(f"DFVS arc must be [tail, head], got {arc!r}.")
        tail = int(arc[0])
        head = int(arc[1])
        if not 0 <= tail < num_vertices or not 0 <= head < num_vertices:
            raise ValueError(f"Arc {(tail, head)} contains a vertex outside 0..{num_vertices - 1}.")
        if tail == head:
            raise ValueError(f"Self-loop arc {(tail, head)} is not supported.")
        if (tail, head) in seen:
            raise ValueError(f"Arc {(tail, head)} is repeated.")
        seen.add((tail, head))


def canonicalize_solution(raw_solution, instance: dict[str, object]) -> list[int]:
    if not isinstance(raw_solution, (list, tuple, set)):
        raise TypeError("DFVS solver output must be a sequence of vertex ids.")
    return sorted(int(value) for value in raw_solution)


def _remaining_adjacency(instance: dict[str, object], removed: set[int]) -> list[list[int]]:
    adjacency = [[] for _ in range(int(instance["num_vertices"]))]
    for tail, head in instance["arcs"]:
        tail = int(tail)
        head = int(head)
        if tail not in removed and head not in removed:
            adjacency[tail].append(head)
    return adjacency


def _adjacency(instance: dict[str, object]) -> list[list[int]]:
    adjacency = [[] for _ in range(int(instance["num_vertices"]))]
    for tail, head in instance["arcs"]:
        adjacency[int(tail)].append(int(head))
    return adjacency


def _find_directed_cycle(adjacency: list[list[int]], blocked: list[bool]) -> list[int] | None:
    state = [0] * len(adjacency)
    stack: list[int] = []
    stack_index: dict[int, int] = {}

    for root in range(len(adjacency)):
        if blocked[root] or state[root] != 0:
            continue
        state[root] = 1
        stack_index[root] = len(stack)
        stack.append(root)
        frames = [(root, 0)]
        while frames:
            vertex, next_index = frames[-1]
            if next_index >= len(adjacency[vertex]):
                frames.pop()
                stack.pop()
                stack_index.pop(vertex, None)
                state[vertex] = 2
                continue
            head = adjacency[vertex][next_index]
            frames[-1] = (vertex, next_index + 1)
            if blocked[head]:
                continue
            if state[head] == 0:
                state[head] = 1
                stack_index[head] = len(stack)
                stack.append(head)
                frames.append((head, 0))
            elif state[head] == 1:
                return stack[stack_index[head] :]
    return None


def is_acyclic_after_removal(instance: dict[str, object], removed: set[int]) -> bool:
    num_vertices = int(instance["num_vertices"])
    adjacency = _remaining_adjacency(instance, removed)
    indegree = [0] * num_vertices
    active_count = 0
    for vertex in range(num_vertices):
        if vertex in removed:
            continue
        active_count += 1
        for head in adjacency[vertex]:
            indegree[head] += 1
    queue = deque(vertex for vertex in range(num_vertices) if vertex not in removed and indegree[vertex] == 0)
    visited = 0
    while queue:
        vertex = queue.popleft()
        visited += 1
        for head in adjacency[vertex]:
            indegree[head] -= 1
            if indegree[head] == 0:
                queue.append(head)
    return visited == active_count


def validate_solution(solution: list[int], instance: dict[str, object]) -> tuple[bool, str | None]:
    num_vertices = int(instance["num_vertices"])
    removed: set[int] = set()
    for vertex in solution:
        if not 0 <= int(vertex) < num_vertices:
            return False, f"Vertex {vertex} is outside 0..{num_vertices - 1}."
        if int(vertex) in removed:
            return False, f"Vertex {vertex} is repeated."
        removed.add(int(vertex))
    if not is_acyclic_after_removal(instance, removed):
        return False, "A directed cycle remains after deleting the proposed feedback vertex set."
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


def find_directed_cycle(instance: dict[str, object], removed: set[int] | None = None) -> list[int] | None:
    removed = removed or set()
    blocked = [vertex in removed for vertex in range(int(instance["num_vertices"]))]
    return _find_directed_cycle(_adjacency(instance), blocked)


def _degree_scores(instance: dict[str, object], removed: set[int]) -> list[int]:
    scores = [0] * int(instance["num_vertices"])
    for tail, head in instance["arcs"]:
        tail = int(tail)
        head = int(head)
        if tail in removed or head in removed:
            continue
        scores[tail] += 1
        scores[head] += 1
    return scores


def greedy_dfvs(instance: dict[str, object]) -> list[int]:
    adjacency = _adjacency(instance)
    blocked = [False] * len(adjacency)
    scores = _degree_scores(instance, set())
    removed: list[int] = []
    while True:
        cycle = _find_directed_cycle(adjacency, blocked)
        if cycle is None:
            break
        vertex = max(cycle, key=lambda item: (scores[item], -item))
        blocked[vertex] = True
        removed.append(vertex)
    return sorted(removed)


def prune_redundant(instance: dict[str, object], solution: list[int]) -> list[int]:
    selected = sorted(set(solution))
    for vertex in list(selected):
        candidate = [item for item in selected if item != vertex]
        valid, _ = validate_solution(candidate, instance)
        if valid:
            selected = candidate
    return selected


def greedy_vertex_disjoint_cycle_lower_bound(instance: dict[str, object], max_cycles: int | None = None) -> int:
    adjacency = _adjacency(instance)
    blocked = [False] * len(adjacency)
    count = 0
    while True:
        if max_cycles is not None and count >= max_cycles:
            return count
        cycle = _find_directed_cycle(adjacency, blocked)
        if cycle is None:
            return count
        for vertex in cycle:
            blocked[vertex] = True
        count += 1


def solve_exact(instance: dict[str, object]) -> ExactSolveResult:
    start = time.perf_counter()
    num_vertices = int(instance["num_vertices"])
    if num_vertices > 24:
        raise RuntimeError("Built-in DFVS exact solver is limited to 24 vertices.")
    for size in range(num_vertices + 1):
        for candidate in combinations(range(num_vertices), size):
            solution = list(candidate)
            valid, _ = validate_solution(solution, instance)
            if valid:
                return ExactSolveResult(
                    solution=solution,
                    objective_value=float(size),
                    runtime_ms=(time.perf_counter() - start) * 1000.0,
                    source="dfvs_bruteforce_exact",
                )
    raise RuntimeError("No directed feedback vertex set found.")


def summarize_training_data(train_instances: list[dict[str, object]], manifest: dict[str, object]) -> dict[str, object]:
    vertex_counts = [int(instance["num_vertices"]) for instance in train_instances]
    arc_counts = [len(instance["arcs"]) for instance in train_instances]
    return {
        "problem": manifest["problem"],
        "family": manifest["family"],
        "num_instances": len(train_instances),
        "runtime_instance_fields": {
            "num_vertices": "int number of vertices",
            "arcs": "list of directed 0-based [tail, head] pairs; use instance['arcs'], not instance['edges']",
        },
        "num_vertices_mean": round(statistics.mean(vertex_counts), 2) if vertex_counts else 0.0,
        "num_arcs_mean": round(statistics.mean(arc_counts), 2) if arc_counts else 0.0,
        "sample_instances": [
            {
                "id": instance["id"],
                "num_vertices": instance["num_vertices"],
                "num_arcs": len(instance["arcs"]),
                "arc_prefix": instance["arcs"][:10],
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
    cycle = find_directed_cycle(instance, set(solution))
    return {
        "instance_id": instance["id"],
        "normalized_quality": score.normalized_quality,
        "objective_value": score.objective_value,
        "runtime_ms": runtime_seconds * 1000.0,
        "is_optimal": score.is_optimal,
        "solution_prefix": solution[:12],
        "remaining_cycle": cycle[:12] if cycle is not None else [],
        "error": score.error,
        "schema_hint": "DFVS runtime instances use instance['arcs'] for directed arcs; instance['edges'] is absent.",
    }


def baseline_registry() -> dict[str, object]:
    return {
        "cycle_degree_greedy": greedy_dfvs,
        "exact": lambda instance: solve_exact(instance).solution,
    }


PROBLEM = ProblemDefinition(
    name="dfvs",
    description=(
        "Directed feedback vertex set: delete vertices so the remaining digraph is acyclic. "
        "Runtime instances provide directed arcs in instance['arcs'] as 0-based [tail, head] pairs."
    ),
    metric_definition={
        "primary": "normalized_quality",
        "secondary": "optimality_rate",
        "tertiary": "average_runtime_ms",
        "notes": "normalized_quality is optimum_or_proxy_dfvs_size / returned_dfvs_size",
    },
    instance_schema_version="dfvs.v1",
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
