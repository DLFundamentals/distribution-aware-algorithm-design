from __future__ import annotations

import itertools
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from dasbench.problems.base import SolveOutcome
from dasbench.problems.graph_utils import (
    adjacency_sets,
    dsatur_coloring,
    greedy_dominating_set_by_score,
    greedy_independent_set_with_local_improvement,
    prune_redundant_dominating_vertices,
)
from dasbench.problems.maxsat import (
    clause_is_satisfied,
    greedy_flip_improve,
    literal_majority_assignment,
)
from dasbench.problems.packing_utils import greedy_binary_solution, greedy_fractional_solution
from dasbench.problems.tsp_utils import (
    canonicalize_tour,
    distance_matrix,
    farthest_insertion_tour,
    two_opt_improve,
)


STATUS_NAMES = {
    GRB.LOADED: "LOADED",
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.CUTOFF: "CUTOFF",
    GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
    GRB.NODE_LIMIT: "NODE_LIMIT",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    GRB.INTERRUPTED: "INTERRUPTED",
    GRB.NUMERIC: "NUMERIC",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
}

_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class GurobiBaselineConfig:
    enabled: bool = True
    time_limit_seconds: float = 60.0
    threads: int = 1
    output_flag: int = 0
    mip_gap: float = 0.0
    baseline_name: str = "gurobi_timed"

    def to_record(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "time_limit_seconds": self.time_limit_seconds,
            "threads": self.threads,
            "output_flag": self.output_flag,
            "mip_gap": self.mip_gap,
            "baseline_name": self.baseline_name,
        }

    @classmethod
    def from_record(cls, payload: dict[str, object] | None) -> GurobiBaselineConfig:
        if not payload:
            return cls()
        return cls(
            enabled=bool(payload.get("enabled", True)),
            time_limit_seconds=float(payload.get("time_limit_seconds", 60.0)),
            threads=int(payload.get("threads", 1)),
            output_flag=int(payload.get("output_flag", 0)),
            mip_gap=float(payload.get("mip_gap", 0.0)),
            baseline_name=str(payload.get("baseline_name", "gurobi_timed")),
        )


class GurobiBaselineError(RuntimeError):
    def __init__(self, message: str, *, metadata: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.metadata = metadata or {}


def _status_name(status: int) -> str:
    return STATUS_NAMES.get(status, str(status))


def _thread_env(output_flag: int) -> gp.Env:
    cached = getattr(_THREAD_LOCAL, "gurobi_envs", None)
    if cached is None:
        cached = {}
        _THREAD_LOCAL.gurobi_envs = cached
    env = cached.get(output_flag)
    if env is None:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", int(output_flag))
        env.start()
        cached[output_flag] = env
    return env


@contextmanager
def _model_context(config: GurobiBaselineConfig, *, name: str):
    model = gp.Model(name=name, env=_thread_env(config.output_flag))
    try:
        model.Params.OutputFlag = int(config.output_flag)
        model.Params.Threads = int(config.threads)
        model.Params.TimeLimit = float(config.time_limit_seconds)
        model.Params.MIPGap = float(config.mip_gap)
        yield model
    finally:
        model.dispose()


def _safe_model_attr(model: gp.Model, attribute: str, default: Any = None) -> Any:
    try:
        return getattr(model, attribute)
    except Exception:
        return default


def _metadata(
    model: gp.Model,
    *,
    instance_id: str,
) -> dict[str, object]:
    status = int(model.Status)
    solution_count = int(_safe_model_attr(model, "SolCount", 0) or 0)
    objective_value = float(_safe_model_attr(model, "ObjVal")) if solution_count > 0 else None
    best_bound = float(_safe_model_attr(model, "ObjBound")) if solution_count > 0 else None
    raw_gap = _safe_model_attr(model, "MIPGap")
    gap = float(raw_gap) if solution_count > 0 and raw_gap is not None else None
    return {
        "instance_id": instance_id,
        "status": _status_name(status),
        "gurobi_runtime_ms": float(_safe_model_attr(model, "Runtime", 0.0) or 0.0) * 1000.0,
        "objective_value": objective_value,
        "best_bound": best_bound,
        "mip_gap": gap,
        "node_count": float(_safe_model_attr(model, "NodeCount", 0.0) or 0.0),
        "solution_count": solution_count,
        "time_limit_hit": status == GRB.TIME_LIMIT,
    }


def _raise_without_incumbent(model: gp.Model, *, instance_id: str) -> None:
    metadata = _metadata(model, instance_id=instance_id)
    raise GurobiBaselineError(
        f"Gurobi finished with status {metadata['status']} and no incumbent solution.",
        metadata=metadata,
    )


def _maxsat_warm_start(instance: dict[str, object]) -> list[bool]:
    initial = literal_majority_assignment(instance, include_last_clause=True)
    return greedy_flip_improve(
        instance,
        initial,
        max_flips=max(12, int(instance["num_variables"]) // 2),
    )


def _solve_maxsat(instance: dict[str, object], config: GurobiBaselineConfig) -> SolveOutcome:
    num_variables = int(instance["num_variables"])
    clauses = [[int(literal) for literal in clause] for clause in instance["clauses"]]
    warm_start = _maxsat_warm_start(instance)
    with _model_context(config, name="gurobi_maxsat") as model:
        variables = model.addVars(num_variables, vtype=GRB.BINARY, name="x")
        clause_sat = model.addVars(len(clauses), vtype=GRB.BINARY, name="sat")
        for clause_index, clause in enumerate(clauses):
            expression = gp.quicksum(
                variables[abs(literal) - 1] if literal > 0 else (1 - variables[abs(literal) - 1])
                for literal in clause
            )
            model.addConstr(clause_sat[clause_index] <= expression, name=f"clause_{clause_index}")
        model.setObjective(clause_sat.sum(), GRB.MAXIMIZE)

        for variable_index, value in enumerate(warm_start):
            variables[variable_index].Start = 1.0 if value else 0.0
        for clause_index, clause in enumerate(clauses):
            clause_sat[clause_index].Start = 1.0 if clause_is_satisfied(clause, warm_start) else 0.0

        model.optimize()
        if int(model.SolCount) <= 0:
            _raise_without_incumbent(model, instance_id=str(instance["id"]))
        solution = [bool(round(float(variables[variable_index].X))) for variable_index in range(num_variables)]
        return SolveOutcome(solution=solution, metadata=_metadata(model, instance_id=str(instance["id"])))


def _solve_mis(instance: dict[str, object], config: GurobiBaselineConfig) -> SolveOutcome:
    num_vertices = int(instance["num_vertices"])
    edges = [(int(left), int(right)) for left, right in instance["edges"]]
    warm_start = set(greedy_independent_set_with_local_improvement(instance))
    with _model_context(config, name="gurobi_mis") as model:
        chosen = model.addVars(num_vertices, vtype=GRB.BINARY, name="x")
        for left, right in edges:
            model.addConstr(chosen[left] + chosen[right] <= 1, name=f"edge_{left}_{right}")
        model.setObjective(chosen.sum(), GRB.MAXIMIZE)
        for vertex in range(num_vertices):
            chosen[vertex].Start = 1.0 if vertex in warm_start else 0.0
        model.optimize()
        if int(model.SolCount) <= 0:
            _raise_without_incumbent(model, instance_id=str(instance["id"]))
        solution = [vertex for vertex in range(num_vertices) if float(chosen[vertex].X) > 0.5]
        return SolveOutcome(solution=solution, metadata=_metadata(model, instance_id=str(instance["id"])))


def _mds_warm_start(instance: dict[str, object]) -> list[int]:
    return prune_redundant_dominating_vertices(
        instance,
        greedy_dominating_set_by_score(
            instance,
            score_fn=lambda vertex, closed, dominated, chosen: (
                len(closed[vertex] - dominated),
                -len(closed[vertex] & dominated),
            ),
        ),
    )


def _solve_mds(instance: dict[str, object], config: GurobiBaselineConfig) -> SolveOutcome:
    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    closed = [neighbors | {vertex} for vertex, neighbors in enumerate(adjacency)]
    warm_start = set(_mds_warm_start(instance))
    with _model_context(config, name="gurobi_mds") as model:
        chosen = model.addVars(num_vertices, vtype=GRB.BINARY, name="x")
        for vertex in range(num_vertices):
            model.addConstr(
                gp.quicksum(chosen[item] for item in closed[vertex]) >= 1,
                name=f"dominate_{vertex}",
            )
        model.setObjective(chosen.sum(), GRB.MINIMIZE)
        for vertex in range(num_vertices):
            chosen[vertex].Start = 1.0 if vertex in warm_start else 0.0
        model.optimize()
        if int(model.SolCount) <= 0:
            _raise_without_incumbent(model, instance_id=str(instance["id"]))
        solution = [vertex for vertex in range(num_vertices) if float(chosen[vertex].X) > 0.5]
        return SolveOutcome(solution=solution, metadata=_metadata(model, instance_id=str(instance["id"])))


def _hitting_set_warm_start(instance: dict[str, object]) -> list[int]:
    """Reuse the problem's own greedy so the incumbent matches the catalog baseline."""
    from dasbench.problems.hitting_set import _greedy_solution

    return _greedy_solution(instance)


def _solve_hitting_set(instance: dict[str, object], config: GurobiBaselineConfig) -> SolveOutcome:
    """Minimum hitting set: choose fewest vertices meeting every hyperedge.

    Only vertices that actually appear in some set can help, and PACE hypergraphs
    declare far more vertices than they use (millions declared, a fraction
    incident). Variables are therefore created per *incident* vertex rather than
    over the declared range, which keeps the model to the size of the instance
    that matters.
    """
    sets = [
        {int(vertex) for vertex in hyperedge}
        for hyperedge in instance["sets"]
        if hyperedge
    ]
    incident = sorted({vertex for hyperedge in sets for vertex in hyperedge})
    if not sets:
        return SolveOutcome(solution=[], metadata={"instance_id": str(instance["id"])})

    warm_start = set(_hitting_set_warm_start(instance))
    with _model_context(config, name="gurobi_hitting_set") as model:
        chosen = model.addVars(incident, vtype=GRB.BINARY, name="x")
        for index, hyperedge in enumerate(sets):
            model.addConstr(
                gp.quicksum(chosen[vertex] for vertex in hyperedge) >= 1,
                name=f"hit_{index}",
            )
        model.setObjective(chosen.sum(), GRB.MINIMIZE)
        for vertex in incident:
            chosen[vertex].Start = 1.0 if vertex in warm_start else 0.0
        model.optimize()
        if int(model.SolCount) <= 0:
            _raise_without_incumbent(model, instance_id=str(instance["id"]))
        solution = [vertex for vertex in incident if float(chosen[vertex].X) > 0.5]
        return SolveOutcome(solution=solution, metadata=_metadata(model, instance_id=str(instance["id"])))


def _solve_coloring(instance: dict[str, object], config: GurobiBaselineConfig) -> SolveOutcome:
    num_vertices = int(instance["num_vertices"])
    edges = [(int(left), int(right)) for left, right in instance["edges"]]
    warm_start = dsatur_coloring(instance)
    num_colors = len(set(warm_start))
    with _model_context(config, name="gurobi_coloring") as model:
        assignment = model.addVars(num_vertices, num_colors, vtype=GRB.BINARY, name="x")
        used = model.addVars(num_colors, vtype=GRB.BINARY, name="used")
        for vertex in range(num_vertices):
            model.addConstr(
                gp.quicksum(assignment[vertex, color] for color in range(num_colors)) == 1,
                name=f"assign_{vertex}",
            )
        for color in range(num_colors):
            for vertex in range(num_vertices):
                model.addConstr(assignment[vertex, color] <= used[color], name=f"use_{vertex}_{color}")
        for left, right in edges:
            for color in range(num_colors):
                model.addConstr(
                    assignment[left, color] + assignment[right, color] <= 1,
                    name=f"edge_{left}_{right}_{color}",
                )
        for color in range(num_colors - 1):
            model.addConstr(used[color] >= used[color + 1], name=f"order_{color}")
        model.setObjective(used.sum(), GRB.MINIMIZE)
        for vertex, color in enumerate(warm_start):
            assignment[vertex, color].Start = 1.0
        for color in range(num_colors):
            used[color].Start = 1.0 if color in set(warm_start) else 0.0
        model.optimize()
        if int(model.SolCount) <= 0:
            _raise_without_incumbent(model, instance_id=str(instance["id"]))
        solution = []
        for vertex in range(num_vertices):
            assigned_color = max(range(num_colors), key=lambda color: float(assignment[vertex, color].X))
            solution.append(assigned_color)
        return SolveOutcome(solution=solution, metadata=_metadata(model, instance_id=str(instance["id"])))


def _tsp_warm_start(instance: dict[str, object]) -> list[int]:
    return two_opt_improve(instance, farthest_insertion_tour(instance), max_rounds=6)


def _shortest_subtour(selected_edges: list[tuple[int, int]], num_cities: int) -> list[int]:
    neighbors = {city: [] for city in range(num_cities)}
    for left, right in selected_edges:
        neighbors[left].append(right)
        neighbors[right].append(left)
    remaining = set(range(num_cities))
    best_cycle = list(range(num_cities))
    while remaining:
        start = next(iter(remaining))
        cycle = []
        stack = [start]
        while stack:
            city = stack.pop()
            if city not in remaining:
                continue
            remaining.remove(city)
            cycle.append(city)
            for neighbor in neighbors[city]:
                if neighbor in remaining:
                    stack.append(neighbor)
        if len(cycle) < len(best_cycle):
            best_cycle = cycle
    return best_cycle


def _tour_edges(tour: list[int]) -> list[tuple[int, int]]:
    edges = []
    for left, right in zip(tour, tour[1:], strict=False):
        edges.append((min(left, right), max(left, right)))
    edges.append((min(tour[-1], tour[0]), max(tour[-1], tour[0])))
    return edges


def _extract_tour(selected_edges: list[tuple[int, int]], num_cities: int) -> list[int]:
    neighbors = {city: [] for city in range(num_cities)}
    for left, right in selected_edges:
        neighbors[left].append(right)
        neighbors[right].append(left)
    if any(len(items) != 2 for items in neighbors.values()):
        raise RuntimeError("Selected edges do not form a Hamiltonian cycle.")
    tour = [0]
    previous = None
    current = 0
    while len(tour) < num_cities:
        options = neighbors[current]
        next_city = options[0] if options[0] != previous else options[1]
        tour.append(next_city)
        previous, current = current, next_city
    return canonicalize_tour(tour, num_cities)


def _solve_tsp(instance: dict[str, object], config: GurobiBaselineConfig) -> SolveOutcome:
    num_cities = int(instance["num_cities"])
    matrix = distance_matrix(instance["points"])
    warm_start = _tsp_warm_start(instance)
    warm_start_edges = set(_tour_edges(warm_start))
    with _model_context(config, name="gurobi_tsp") as model:
        edge_vars = {
            (left, right): model.addVar(vtype=GRB.BINARY, obj=matrix[left][right], name=f"e_{left}_{right}")
            for left in range(num_cities)
            for right in range(left + 1, num_cities)
        }
        for city in range(num_cities):
            incident = [
                edge_vars[(min(city, other), max(city, other))]
                for other in range(num_cities)
                if other != city
            ]
            model.addConstr(gp.quicksum(incident) == 2, name=f"degree_{city}")
        model.ModelSense = GRB.MINIMIZE
        model.Params.LazyConstraints = 1
        for edge, variable in edge_vars.items():
            variable.Start = 1.0 if edge in warm_start_edges else 0.0
        model._edge_vars = edge_vars  # type: ignore[attr-defined]
        model._num_cities = num_cities  # type: ignore[attr-defined]

        def callback(callback_model, where):  # type: ignore[no-untyped-def]
            if where != GRB.Callback.MIPSOL:
                return
            values = callback_model.cbGetSolution(list(callback_model._edge_vars.values()))
            selected = [
                edge
                for edge, value in zip(callback_model._edge_vars.keys(), values, strict=True)
                if value > 0.5
            ]
            cycle = _shortest_subtour(selected, int(callback_model._num_cities))
            if len(cycle) == int(callback_model._num_cities):
                return
            callback_model.cbLazy(
                gp.quicksum(
                    callback_model._edge_vars[(min(left, right), max(left, right))]
                    for left, right in itertools.combinations(cycle, 2)
                )
                <= len(cycle) - 1
            )

        model.optimize(callback)
        if int(model.SolCount) <= 0:
            _raise_without_incumbent(model, instance_id=str(instance["id"]))
        selected_edges = [edge for edge, variable in edge_vars.items() if float(variable.X) > 0.5]
        solution = _extract_tour(selected_edges, num_cities)
        return SolveOutcome(solution=solution, metadata=_metadata(model, instance_id=str(instance["id"])))


def _solve_packing_lp(instance: dict[str, object], config: GurobiBaselineConfig) -> SolveOutcome:
    num_items = int(instance["num_items"])
    num_resources = int(instance["num_resources"])
    weights = [[float(value) for value in row] for row in instance["weights"]]
    values = [float(value) for value in instance["values"]]
    capacities = [float(value) for value in instance["capacities"]]
    warm_start = greedy_fractional_solution(instance)
    with _model_context(config, name="gurobi_packing_lp") as model:
        variables = model.addVars(num_items, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
        for resource in range(num_resources):
            model.addConstr(
                gp.quicksum(weights[item][resource] * variables[item] for item in range(num_items)) <= capacities[resource],
                name=f"capacity_{resource}",
            )
        model.setObjective(gp.quicksum(values[item] * variables[item] for item in range(num_items)), GRB.MAXIMIZE)
        for item in range(num_items):
            variables[item].Start = float(warm_start[item])
        model.optimize()
        if int(model.SolCount) <= 0:
            _raise_without_incumbent(model, instance_id=str(instance["id"]))
        solution = [max(0.0, min(1.0, float(variables[item].X))) for item in range(num_items)]
        return SolveOutcome(solution=solution, metadata=_metadata(model, instance_id=str(instance["id"])))


def _solve_mdkp(instance: dict[str, object], config: GurobiBaselineConfig) -> SolveOutcome:
    num_items = int(instance["num_items"])
    num_resources = int(instance["num_resources"])
    weights = [[float(value) for value in row] for row in instance["weights"]]
    values = [float(value) for value in instance["values"]]
    capacities = [float(value) for value in instance["capacities"]]
    warm_start = set(greedy_binary_solution(instance))
    with _model_context(config, name="gurobi_mdkp") as model:
        variables = model.addVars(num_items, vtype=GRB.BINARY, name="x")
        for resource in range(num_resources):
            model.addConstr(
                gp.quicksum(weights[item][resource] * variables[item] for item in range(num_items)) <= capacities[resource],
                name=f"capacity_{resource}",
            )
        model.setObjective(gp.quicksum(values[item] * variables[item] for item in range(num_items)), GRB.MAXIMIZE)
        for item in range(num_items):
            variables[item].Start = 1.0 if item in warm_start else 0.0
        model.optimize()
        if int(model.SolCount) <= 0:
            _raise_without_incumbent(model, instance_id=str(instance["id"]))
        solution = [item for item in range(num_items) if float(variables[item].X) > 0.5]
        return SolveOutcome(solution=solution, metadata=_metadata(model, instance_id=str(instance["id"])))


def _wrap_builder(builder):
    def solver(instance: dict[str, object]) -> SolveOutcome:
        return builder(instance)

    return solver


def build_gurobi_solver(problem_name: str, config: GurobiBaselineConfig):
    builders = {
        "maxsat": lambda instance: _solve_maxsat(instance, config),
        "mis": lambda instance: _solve_mis(instance, config),
        "mds": lambda instance: _solve_mds(instance, config),
        "hitting_set": lambda instance: _solve_hitting_set(instance, config),
        "coloring": lambda instance: _solve_coloring(instance, config),
        "tsp": lambda instance: _solve_tsp(instance, config),
        "packing_lp": lambda instance: _solve_packing_lp(instance, config),
        "mdkp": lambda instance: _solve_mdkp(instance, config),
    }
    try:
        return _wrap_builder(builders[problem_name])
    except KeyError as exc:
        raise ValueError(f"Unsupported Gurobi baseline problem: {problem_name}") from exc
