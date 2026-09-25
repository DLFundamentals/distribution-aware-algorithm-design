from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ml_baselines.torch_utils import as_backend_tensor

PRIVATE_PREFIXES = ("optimum_", "_")
EPS = 1e-9


@dataclass(frozen=True)
class GraphTensor:
    instance_id: str
    num_nodes: int
    node_features: Any
    edge_index: Any
    edge_features: Any | None = None


@dataclass(frozen=True)
class PackingTensor:
    instance_id: str
    num_items: int
    num_resources: int
    item_features: Any
    resource_features: Any
    weights: Any
    capacities: Any
    values: Any


@dataclass(frozen=True)
class MaxSatTensor:
    instance_id: str
    num_variables: int
    num_clauses: int
    variable_features: Any
    clause_features: Any
    incidence_index: Any
    incidence_sign: Any


@dataclass(frozen=True)
class TspTensor:
    instance_id: str
    num_cities: int
    node_features: Any
    edge_index: Any
    edge_features: Any


def public_view(instance: dict[str, Any]) -> dict[str, Any]:
    """Drop evaluator-only and private fields before feature extraction."""

    return {
        key: value
        for key, value in instance.items()
        if not any(str(key).startswith(prefix) for prefix in PRIVATE_PREFIXES)
    }


def _edge_index_from_edges(num_nodes: int, edges: list[list[int]] | list[tuple[int, int]], *, include_reverse: bool) -> list[list[int]]:
    sources: list[int] = []
    targets: list[int] = []
    for raw_u, raw_v in edges:
        u = int(raw_u)
        v = int(raw_v)
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError(f"Edge {(u, v)!r} is outside 0..{num_nodes - 1}.")
        sources.append(u)
        targets.append(v)
        if include_reverse:
            sources.append(v)
            targets.append(u)
    return [sources, targets]


def tensorize_graph_instance(
    instance: dict[str, Any],
    *,
    include_reverse_edges: bool = True,
    backend: str = "auto",
    device: str | None = None,
) -> GraphTensor:
    public = public_view(instance)
    num_nodes = int(public["num_vertices"])
    edges = public.get("edges")
    if not isinstance(edges, list):
        raise ValueError("Graph instances require an `edges` list.")

    degrees = [0] * num_nodes
    adjacency = [set() for _ in range(num_nodes)]
    for raw_u, raw_v in edges:
        u = int(raw_u)
        v = int(raw_v)
        degrees[u] += 1
        degrees[v] += 1
        adjacency[u].add(v)
        adjacency[v].add(u)
    max_degree = max(degrees) if degrees else 1
    density = 0.0 if num_nodes <= 1 else (2.0 * len(edges)) / (num_nodes * (num_nodes - 1))
    edge_set = {
        (min(int(raw_u), int(raw_v)), max(int(raw_u), int(raw_v)))
        for raw_u, raw_v in edges
    }
    clustering_proxy: list[float] = []
    for vertex in range(num_nodes):
        neighbors = sorted(adjacency[vertex])
        possible = len(neighbors) * (len(neighbors) - 1) / 2.0
        if possible <= 0:
            clustering_proxy.append(0.0)
            continue
        triangles = 0
        for left_index, left in enumerate(neighbors):
            for right in neighbors[left_index + 1 :]:
                if (min(left, right), max(left, right)) in edge_set:
                    triangles += 1
        clustering_proxy.append(float(triangles) / possible)
    node_features = [
        [
            float(degree),
            float(degree) / max(float(max_degree), EPS),
            float(degree) / max(float(num_nodes - 1), 1.0),
            clustering_proxy[index],
            density,
            1.0,
        ]
        for index, degree in enumerate(degrees)
    ]
    edge_index = _edge_index_from_edges(num_nodes, edges, include_reverse=include_reverse_edges)
    return GraphTensor(
        instance_id=str(public.get("id", "")),
        num_nodes=num_nodes,
        node_features=as_backend_tensor(node_features, dtype="float", device=device, backend=backend),
        edge_index=as_backend_tensor(edge_index, dtype="long", device=device, backend=backend),
    )


def tensorize_graph_split(
    instances: list[dict[str, Any]],
    *,
    include_reverse_edges: bool = True,
    backend: str = "auto",
    device: str | None = None,
) -> list[GraphTensor]:
    return [
        tensorize_graph_instance(
            instance,
            include_reverse_edges=include_reverse_edges,
            backend=backend,
            device=device,
        )
        for instance in instances
    ]


def tensorize_packing_instance(
    instance: dict[str, Any],
    *,
    backend: str = "auto",
    device: str | None = None,
) -> PackingTensor:
    public = public_view(instance)
    num_items = int(public["num_items"])
    num_resources = int(public["num_resources"])
    values = [float(value) for value in public["values"]]
    weights = [[float(value) for value in row] for row in public["weights"]]
    capacities = [float(value) for value in public["capacities"]]
    if len(values) != num_items or len(weights) != num_items or len(capacities) != num_resources:
        raise ValueError("Packing instance dimensions do not match declared sizes.")

    max_value = max(values) if values else 1.0
    max_capacity = max(capacities) if capacities else 1.0
    resource_totals = [
        sum(weights[item][resource] for item in range(num_items))
        for resource in range(num_resources)
    ]
    resource_means = [
        resource_totals[resource] / max(float(num_items), 1.0)
        for resource in range(num_resources)
    ]
    resource_maxes = [
        max((weights[item][resource] for item in range(num_items)), default=0.0)
        for resource in range(num_resources)
    ]
    item_features: list[list[float]] = []
    for item in range(num_items):
        raw_weights = [weights[item][resource] for resource in range(num_resources)]
        normalized_weights = [
            weights[item][resource] / max(capacities[resource], EPS)
            for resource in range(num_resources)
        ]
        value_densities = [
            values[item] / max(weights[item][resource], EPS)
            for resource in range(num_resources)
        ]
        pressure = sum(normalized_weights)
        raw_sum = sum(raw_weights)
        raw_mean = raw_sum / max(float(num_resources), 1.0)
        raw_max = max(raw_weights) if raw_weights else 0.0
        item_features.append(
            [
                values[item],
                values[item] / max(max_value, EPS),
                *raw_weights,
                *value_densities,
                raw_sum,
                raw_mean,
                raw_max,
                pressure,
                values[item] / max(pressure, EPS),
                *normalized_weights,
                1.0,
            ]
        )

    resource_features = [
        [
            capacities[resource],
            capacities[resource] / max(max_capacity, EPS),
            capacities[resource] / max(resource_totals[resource], EPS),
            resource_totals[resource],
            resource_means[resource],
            resource_maxes[resource],
            resource_means[resource] / max(capacities[resource], EPS),
            resource_maxes[resource] / max(capacities[resource], EPS),
            1.0,
        ]
        for resource in range(num_resources)
    ]
    return PackingTensor(
        instance_id=str(public.get("id", "")),
        num_items=num_items,
        num_resources=num_resources,
        item_features=as_backend_tensor(item_features, dtype="float", device=device, backend=backend),
        resource_features=as_backend_tensor(resource_features, dtype="float", device=device, backend=backend),
        weights=as_backend_tensor(weights, dtype="float", device=device, backend=backend),
        capacities=as_backend_tensor(capacities, dtype="float", device=device, backend=backend),
        values=as_backend_tensor(values, dtype="float", device=device, backend=backend),
    )


def tensorize_packing_split(
    instances: list[dict[str, Any]],
    *,
    backend: str = "auto",
    device: str | None = None,
) -> list[PackingTensor]:
    return [tensorize_packing_instance(instance, backend=backend, device=device) for instance in instances]


def tensorize_maxsat_instance(
    instance: dict[str, Any],
    *,
    backend: str = "auto",
    device: str | None = None,
) -> MaxSatTensor:
    public = public_view(instance)
    num_variables = int(public["num_variables"])
    clauses = public["clauses"]
    num_clauses = len(clauses)

    positive = [0] * num_variables
    negative = [0] * num_variables
    sources: list[int] = []
    targets: list[int] = []
    signs: list[float] = []
    clause_features: list[list[float]] = []
    for clause_index, clause in enumerate(clauses):
        pos_count = 0
        for literal in clause:
            variable = abs(int(literal)) - 1
            sign = 1.0 if int(literal) > 0 else -1.0
            if sign > 0:
                positive[variable] += 1
                pos_count += 1
            else:
                negative[variable] += 1
            sources.append(variable)
            targets.append(clause_index)
            signs.append(sign)
        clause_features.append([float(pos_count), float(len(clause) - pos_count), float(len(clause)), 1.0])
    variable_features = [
        [float(positive[index]), float(negative[index]), float(positive[index] - negative[index]), 1.0]
        for index in range(num_variables)
    ]
    return MaxSatTensor(
        instance_id=str(public.get("id", "")),
        num_variables=num_variables,
        num_clauses=num_clauses,
        variable_features=as_backend_tensor(variable_features, dtype="float", device=device, backend=backend),
        clause_features=as_backend_tensor(clause_features, dtype="float", device=device, backend=backend),
        incidence_index=as_backend_tensor([sources, targets], dtype="long", device=device, backend=backend),
        incidence_sign=as_backend_tensor(signs, dtype="float", device=device, backend=backend),
    )


def tensorize_maxsat_split(
    instances: list[dict[str, Any]],
    *,
    backend: str = "auto",
    device: str | None = None,
) -> list[MaxSatTensor]:
    return [tensorize_maxsat_instance(instance, backend=backend, device=device) for instance in instances]


def tensorize_tsp_instance(
    instance: dict[str, Any],
    *,
    backend: str = "auto",
    device: str | None = None,
) -> TspTensor:
    public = public_view(instance)
    points = [[float(x), float(y)] for x, y in public["points"]]
    num_cities = int(public["num_cities"])
    if len(points) != num_cities:
        raise ValueError("TSP point count does not match num_cities.")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, EPS)
    height = max(max_y - min_y, EPS)
    centroid_x = sum(xs) / num_cities
    centroid_y = sum(ys) / num_cities
    node_features = [
        [
            (x - min_x) / width,
            (y - min_y) / height,
            x - centroid_x,
            y - centroid_y,
            1.0,
        ]
        for x, y in points
    ]
    sources: list[int] = []
    targets: list[int] = []
    edge_features: list[list[float]] = []
    distances: list[list[float]] = [[0.0 for _ in range(num_cities)] for _ in range(num_cities)]
    max_distance = EPS
    for left in range(num_cities):
        for right in range(left + 1, num_cities):
            dx = points[left][0] - points[right][0]
            dy = points[left][1] - points[right][1]
            distance = (dx * dx + dy * dy) ** 0.5
            distances[left][right] = distance
            distances[right][left] = distance
            max_distance = max(max_distance, distance)
    ranks = [[0 for _ in range(num_cities)] for _ in range(num_cities)]
    for city in range(num_cities):
        ordered = sorted(
            (other for other in range(num_cities) if other != city),
            key=lambda other: (distances[city][other], other),
        )
        for rank, other in enumerate(ordered, start=1):
            ranks[city][other] = rank
    for left in range(num_cities):
        for right in range(num_cities):
            if left == right:
                continue
            dx = points[left][0] - points[right][0]
            dy = points[left][1] - points[right][1]
            distance = distances[left][right]
            sources.append(left)
            targets.append(right)
            rank_scale = max(float(num_cities - 1), 1.0)
            edge_features.append(
                [
                    distance,
                    distance / max_distance,
                    dx,
                    dy,
                    abs(dx),
                    abs(dy),
                    float(ranks[left][right]) / rank_scale,
                    float(ranks[right][left]) / rank_scale,
                    1.0,
                ]
            )
    return TspTensor(
        instance_id=str(public.get("id", "")),
        num_cities=num_cities,
        node_features=as_backend_tensor(node_features, dtype="float", device=device, backend=backend),
        edge_index=as_backend_tensor([sources, targets], dtype="long", device=device, backend=backend),
        edge_features=as_backend_tensor(edge_features, dtype="float", device=device, backend=backend),
    )


def tensorize_tsp_split(
    instances: list[dict[str, Any]],
    *,
    backend: str = "auto",
    device: str | None = None,
) -> list[TspTensor]:
    return [tensorize_tsp_instance(instance, backend=backend, device=device) for instance in instances]
