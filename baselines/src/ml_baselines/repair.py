from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

RepairHook = Callable[[Any, dict[str, Any], "RepairContext"], Any]


@dataclass
class RepairContext:
    problem: str | None = None
    budget: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def apply_repair_hooks(
    raw_solution: Any,
    instance: dict[str, Any],
    hooks: Sequence[RepairHook],
    *,
    context: RepairContext | None = None,
) -> Any:
    solution = raw_solution
    repair_context = context or RepairContext()
    for hook in hooks:
        solution = hook(solution, instance, repair_context)
    return solution


def deduplicate_sorted_indices(indices: Sequence[int], *, upper_bound: int | None = None) -> list[int]:
    values = []
    seen: set[int] = set()
    for raw_index in indices:
        index = int(raw_index)
        if upper_bound is not None and not (0 <= index < upper_bound):
            continue
        if index not in seen:
            seen.add(index)
            values.append(index)
    return sorted(values)


def order_from_scores(scores: Sequence[float], *, reverse: bool = True) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (float(scores[index]), -index if reverse else index), reverse=reverse)


def binary_capacity_repair(
    scores: Sequence[float],
    weights: Sequence[Sequence[float]],
    capacities: Sequence[float],
    *,
    values: Sequence[float] | None = None,
    budget: int | None = None,
) -> list[int]:
    num_items = len(scores)
    num_resources = len(capacities)
    objective_values = [1.0] * num_items if values is None else [float(value) for value in values]
    order = sorted(
        range(num_items),
        key=lambda item: (float(scores[item]), objective_values[item], -item),
        reverse=True,
    )
    selected: list[int] = []
    usage = [0.0] * num_resources
    steps = 0
    for item in order:
        if budget is not None and steps >= budget:
            break
        steps += 1
        if all(
            usage[resource] + float(weights[item][resource]) <= float(capacities[resource]) + 1e-9
            for resource in range(num_resources)
        ):
            selected.append(item)
            for resource in range(num_resources):
                usage[resource] += float(weights[item][resource])
    return sorted(selected)


def fractional_capacity_repair(
    fractions: Sequence[float],
    weights: Sequence[Sequence[float]],
    capacities: Sequence[float],
) -> list[float]:
    solution = [max(0.0, min(1.0, float(value))) for value in fractions]
    num_resources = len(capacities)
    usage = [
        sum(solution[item] * float(weights[item][resource]) for item in range(len(solution)))
        for resource in range(num_resources)
    ]
    scale = 1.0
    for resource in range(num_resources):
        if usage[resource] > float(capacities[resource]) + 1e-9:
            scale = min(scale, float(capacities[resource]) / max(usage[resource], 1e-9))
    if scale < 1.0:
        solution = [value * scale for value in solution]
    return solution
