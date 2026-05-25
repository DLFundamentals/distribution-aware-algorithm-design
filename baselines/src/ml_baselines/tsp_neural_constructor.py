from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_baselines.checkpoint import save_checkpoint
from ml_baselines.config import MLBaselineConfig
from ml_baselines.interface import TrainedState
from ml_baselines.metrics import MetricsLogger
from ml_baselines.seeding import seed_everything
from ml_baselines.tensorize import public_view, tensorize_tsp_instance
from ml_baselines.torch_utils import TorchUnavailableError, require_torch, resolve_device, torch_available

TSP_BASELINE_NAME = "ml_tsp_neural_constructor"
EPS = 1e-9


@dataclass(frozen=True)
class TspNeuralConstructorConfig:
    epochs: int = 50
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    batch_size: int = 1
    candidates: int = 4
    two_opt_budget: int = 256
    pseudo_two_opt_budget: int = 256
    device: str = "cpu"
    seed: int = 0

    @classmethod
    def from_config(cls, config: MLBaselineConfig | dict[str, Any] | None) -> "TspNeuralConstructorConfig":
        base = cls()
        if config is None:
            return base
        payload = config.to_dict() if isinstance(config, MLBaselineConfig) else dict(config)
        return cls(
            epochs=int(payload.get("epochs", base.epochs)),
            learning_rate=float(payload.get("learning_rate", payload.get("lr", base.learning_rate))),
            hidden_dim=int(payload.get("hidden_dim", base.hidden_dim)),
            batch_size=int(payload.get("batch_size", base.batch_size)),
            candidates=int(payload.get("candidates", payload.get("inference_candidates", base.candidates))),
            two_opt_budget=int(payload.get("two_opt_budget", payload.get("repair_budget", base.two_opt_budget))),
            pseudo_two_opt_budget=int(payload.get("pseudo_two_opt_budget", base.pseudo_two_opt_budget)),
            device=str(payload.get("device", base.device)),
            seed=int(payload.get("seed", base.seed)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "hidden_dim": self.hidden_dim,
            "batch_size": self.batch_size,
            "candidates": self.candidates,
            "two_opt_budget": self.two_opt_budget,
            "pseudo_two_opt_budget": self.pseudo_two_opt_budget,
            "device": self.device,
            "seed": self.seed,
        }


class EdgeHeatmapNet:
    """Small MLP over directed pair features for TSP edge desirability."""

    def __new__(cls, *args: Any, **kwargs: Any):
        torch = require_torch()
        nn = torch.nn

        class _EdgeHeatmapNet(nn.Module):
            def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int) -> None:
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(edge_dim + 2 * node_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, tensor: Any) -> Any:
                sources = tensor.edge_index[0].long()
                targets = tensor.edge_index[1].long()
                pair_features = require_torch().cat(
                    [
                        tensor.edge_features,
                        tensor.node_features[sources],
                        tensor.node_features[targets],
                    ],
                    dim=-1,
                )
                return self.mlp(pair_features).reshape(-1)

        return _EdgeHeatmapNet(*args, **kwargs)


def _points(instance: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in public_view(instance)["points"]]


def distance_matrix_from_points(points: list[tuple[float, float]]) -> list[list[float]]:
    matrix = [[0.0 for _ in points] for _ in points]
    for left, left_point in enumerate(points):
        for right in range(left + 1, len(points)):
            distance = math.dist(left_point, points[right])
            matrix[left][right] = distance
            matrix[right][left] = distance
    return matrix


def tour_length_from_matrix(matrix: list[list[float]], tour: list[int]) -> float:
    if not tour:
        return 0.0
    return sum(matrix[left][right] for left, right in zip(tour, tour[1:], strict=False)) + matrix[tour[-1]][tour[0]]


def canonicalize_tour(tour: list[int], num_cities: int) -> list[int]:
    normalized = [int(city) for city in tour]
    if len(normalized) == num_cities + 1 and normalized and normalized[0] == normalized[-1]:
        normalized = normalized[:-1]
    if len(normalized) != num_cities:
        raise ValueError(f"Expected {num_cities} cities, got {len(normalized)}.")
    if 0 not in normalized:
        return normalized
    pivot = normalized.index(0)
    rotated = normalized[pivot:] + normalized[:pivot]
    reversed_cycle = [rotated[0]] + list(reversed(rotated[1:]))
    return rotated if rotated <= reversed_cycle else reversed_cycle


def is_valid_tour(num_cities: int, tour: list[int]) -> bool:
    return len(tour) == num_cities and sorted(tour) == list(range(num_cities))


def nearest_neighbor_tour_from_matrix(matrix: list[list[float]], *, start_city: int = 0) -> list[int]:
    num_cities = len(matrix)
    remaining = set(range(num_cities))
    current = int(start_city) % max(num_cities, 1)
    tour = [current]
    remaining.remove(current)
    while remaining:
        current = min(remaining, key=lambda city: (matrix[current][city], city))
        remaining.remove(current)
        tour.append(current)
    return canonicalize_tour(tour, num_cities)


def farthest_insertion_tour_from_matrix(matrix: list[list[float]]) -> list[int]:
    num_cities = len(matrix)
    if num_cities <= 2:
        return list(range(num_cities))
    second = max(range(1, num_cities), key=lambda city: (matrix[0][city], -city))
    cycle = [0, second]
    remaining = set(range(num_cities)) - set(cycle)
    while remaining:
        city = max(
            remaining,
            key=lambda item: (
                min(matrix[item][anchor] for anchor in cycle),
                -item,
            ),
        )
        best_index = 0
        best_delta = math.inf
        for index in range(len(cycle)):
            left = cycle[index]
            right = cycle[(index + 1) % len(cycle)]
            delta = matrix[left][city] + matrix[city][right] - matrix[left][right]
            if delta < best_delta - EPS or (abs(delta - best_delta) <= EPS and index < best_index):
                best_delta = delta
                best_index = index + 1
        cycle.insert(best_index, city)
        remaining.remove(city)
    return canonicalize_tour(cycle, num_cities)


def nearest_insertion_tour_from_matrix(matrix: list[list[float]]) -> list[int]:
    num_cities = len(matrix)
    if num_cities <= 2:
        return list(range(num_cities))
    second = min(range(1, num_cities), key=lambda city: (matrix[0][city], city))
    cycle = [0, second]
    remaining = set(range(num_cities)) - set(cycle)
    while remaining:
        city = min(
            remaining,
            key=lambda item: (
                min(matrix[item][anchor] for anchor in cycle),
                item,
            ),
        )
        best_index = 0
        best_delta = math.inf
        for index in range(len(cycle)):
            left = cycle[index]
            right = cycle[(index + 1) % len(cycle)]
            delta = matrix[left][city] + matrix[city][right] - matrix[left][right]
            if delta < best_delta - EPS or (abs(delta - best_delta) <= EPS and index < best_index):
                best_delta = delta
                best_index = index + 1
        cycle.insert(best_index, city)
        remaining.remove(city)
    return canonicalize_tour(cycle, num_cities)


def bounded_two_opt_from_matrix(matrix: list[list[float]], tour: list[int], *, max_evaluations: int) -> list[int]:
    num_cities = len(matrix)
    if num_cities <= 3 or max_evaluations <= 0:
        return canonicalize_tour(tour, num_cities)
    best = canonicalize_tour(tour, num_cities)
    budget = int(max_evaluations)
    while budget > 0:
        improved = False
        for left_index in range(1, num_cities - 1):
            for right_index in range(left_index + 2, num_cities + 1):
                if left_index == 1 and right_index == num_cities:
                    continue
                budget -= 1
                left_prev = best[left_index - 1]
                left = best[left_index]
                right = best[right_index - 1]
                right_next = best[right_index % num_cities]
                delta = matrix[left_prev][right] + matrix[left][right_next] - matrix[left_prev][left] - matrix[right][right_next]
                if delta < -EPS:
                    best = best[:left_index] + list(reversed(best[left_index:right_index])) + best[right_index:]
                    best = canonicalize_tour(best, num_cities)
                    improved = True
                    break
                if budget <= 0:
                    break
            if improved or budget <= 0:
                break
        if not improved:
            break
    return canonicalize_tour(best, num_cities)


def pseudo_label_tour(instance: dict[str, Any], *, two_opt_budget: int = 256) -> list[int]:
    public = public_view(instance)
    points = _points(public)
    matrix = distance_matrix_from_points(points)
    candidates = [
        nearest_insertion_tour_from_matrix(matrix),
        farthest_insertion_tour_from_matrix(matrix),
        nearest_neighbor_tour_from_matrix(matrix, start_city=0),
    ]
    best = candidates[0]
    best_length = math.inf
    for candidate in candidates:
        improved = bounded_two_opt_from_matrix(matrix, candidate, max_evaluations=two_opt_budget)
        length = tour_length_from_matrix(matrix, improved)
        if length < best_length - EPS:
            best = improved
            best_length = length
    return canonicalize_tour(best, int(public["num_cities"]))


def _edge_targets(tensor: Any, tour: list[int]) -> Any:
    torch = require_torch()
    target_edges: set[tuple[int, int]] = set()
    for left, right in zip(tour, tour[1:], strict=False):
        target_edges.add((left, right))
        target_edges.add((right, left))
    target_edges.add((tour[-1], tour[0]))
    target_edges.add((tour[0], tour[-1]))
    sources = tensor.edge_index[0].detach().cpu().tolist()
    targets = tensor.edge_index[1].detach().cpu().tolist()
    labels = [1.0 if (int(source), int(target)) in target_edges else 0.0 for source, target in zip(sources, targets, strict=True)]
    return torch.tensor(labels, dtype=torch.float32, device=tensor.edge_features.device)


def _loss(model: Any, tensor: Any, targets: Any) -> Any:
    torch = require_torch()
    logits = model(tensor)
    positives = targets.sum().clamp_min(1.0)
    negatives = (float(targets.numel()) - targets.sum()).clamp_min(1.0)
    pos_weight = negatives / positives
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def fit(
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    config: TspNeuralConstructorConfig | MLBaselineConfig | dict[str, Any] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> TrainedState:
    if not torch_available():
        raise TorchUnavailableError("PyTorch is required for the TSP neural constructor baseline.")
    torch = require_torch()
    resolved_config = config if isinstance(config, TspNeuralConstructorConfig) else TspNeuralConstructorConfig.from_config(config)
    seed_everything(resolved_config.seed)
    device = resolve_device(resolved_config.device)
    if not train_instances:
        raise ValueError("The TSP neural constructor requires public training instances.")

    public_train = [public_view(instance) for instance in train_instances]
    public_val = [public_view(instance) for instance in (val_instances or [])]
    train_tensors = [tensorize_tsp_instance(instance, backend="torch", device=str(device)) for instance in public_train]
    val_tensors = [tensorize_tsp_instance(instance, backend="torch", device=str(device)) for instance in public_val]
    train_targets = [
        _edge_targets(tensor, pseudo_label_tour(instance, two_opt_budget=resolved_config.pseudo_two_opt_budget))
        for instance, tensor in zip(public_train, train_tensors, strict=True)
    ]
    val_targets = [
        _edge_targets(tensor, pseudo_label_tour(instance, two_opt_budget=resolved_config.pseudo_two_opt_budget))
        for instance, tensor in zip(public_val, val_tensors, strict=True)
    ]
    node_dim = int(train_tensors[0].node_features.shape[-1])
    edge_dim = int(train_tensors[0].edge_features.shape[-1])
    model = EdgeHeatmapNet(node_dim, edge_dim, resolved_config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=resolved_config.learning_rate)
    logger = MetricsLogger(Path(metrics_path) if metrics_path is not None else None)
    best_state: dict[str, Any] | None = None
    best_train_loss = math.inf

    for epoch in range(1, resolved_config.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for tensor, targets in zip(train_tensors, train_targets, strict=True):
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, tensor, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
        train_loss = sum(train_losses) / max(1, len(train_losses))
        val_loss = None
        if val_tensors:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for tensor, targets in zip(val_tensors, val_targets, strict=True):
                    loss = _loss(model, tensor, targets)
                    val_losses.append(float(loss.detach().cpu().item()))
            val_loss = sum(val_losses) / max(1, len(val_losses))
        logger.log(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "problem": "tsp",
        "baseline_name": TSP_BASELINE_NAME,
        "config": resolved_config.to_dict(),
        "train_instances": len(public_train),
        "validation_instances": len(public_val),
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "history": logger.rows,
        "training_mode": "edge_heatmap_self_training",
    }
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            {
                "model_state_dict": model.state_dict(),
                "node_dim": node_dim,
                "edge_dim": edge_dim,
            },
            metadata=metadata,
            use_torch=True,
        )
    return TrainedState(payload={"model": model, "node_dim": node_dim, "edge_dim": edge_dim}, metadata=metadata)


def _heatmap_from_model(model: Any, instance: dict[str, Any], config: TspNeuralConstructorConfig) -> list[list[float]]:
    torch = require_torch()
    device = resolve_device(config.device)
    public = public_view(instance)
    tensor = tensorize_tsp_instance(public, backend="torch", device=str(device))
    model.eval()
    with torch.no_grad():
        logits = model(tensor).detach().cpu().tolist()
    num_cities = int(public["num_cities"])
    heatmap = [[-math.inf for _ in range(num_cities)] for _ in range(num_cities)]
    sources = tensor.edge_index[0].detach().cpu().tolist()
    targets = tensor.edge_index[1].detach().cpu().tolist()
    for source, target, score in zip(sources, targets, logits, strict=True):
        heatmap[int(source)][int(target)] = float(score)
    return heatmap


def learned_nearest_neighbor_tour(heatmap: list[list[float]], matrix: list[list[float]], *, start_city: int = 0) -> list[int]:
    num_cities = len(heatmap)
    if num_cities <= 1:
        return [0] if num_cities == 1 else []
    current = int(start_city) % num_cities
    remaining = set(range(num_cities))
    remaining.remove(current)
    tour = [current]
    while remaining:
        current = max(
            remaining,
            key=lambda city: (
                heatmap[current][city],
                -matrix[current][city],
                -city,
            ),
        )
        tour.append(current)
        remaining.remove(current)
    return canonicalize_tour(tour, num_cities)


def learned_insertion_tour(heatmap: list[list[float]], matrix: list[list[float]]) -> list[int]:
    num_cities = len(heatmap)
    if num_cities <= 2:
        return list(range(num_cities))
    second = max(
        range(1, num_cities),
        key=lambda city: (
            heatmap[0][city] + heatmap[city][0] - matrix[0][city],
            -city,
        ),
    )
    cycle = [0, second]
    remaining = set(range(num_cities)) - set(cycle)
    while remaining:
        best_city: int | None = None
        best_index = 0
        best_key: tuple[float, float, int] | None = None
        for city in remaining:
            for index in range(len(cycle)):
                left = cycle[index]
                right = cycle[(index + 1) % len(cycle)]
                delta = matrix[left][city] + matrix[city][right] - matrix[left][right]
                score_delta = heatmap[left][city] + heatmap[city][right] - heatmap[left][right]
                key = (score_delta - delta, -delta, -city)
                if best_key is None or key > best_key:
                    best_key = key
                    best_city = city
                    best_index = index + 1
        assert best_city is not None
        cycle.insert(best_index, best_city)
        remaining.remove(best_city)
    return canonicalize_tour(cycle, num_cities)


def decode_tsp_heatmap(
    instance: dict[str, Any],
    heatmap: list[list[float]],
    *,
    candidates: int = 4,
    two_opt_budget: int = 256,
) -> list[int]:
    public = public_view(instance)
    num_cities = int(public["num_cities"])
    matrix = distance_matrix_from_points(_points(public))
    raw_candidates: list[list[int]] = [
        learned_insertion_tour(heatmap, matrix),
        nearest_insertion_tour_from_matrix(matrix),
        farthest_insertion_tour_from_matrix(matrix),
    ]
    for start in range(min(max(1, int(candidates)), num_cities)):
        raw_candidates.append(learned_nearest_neighbor_tour(heatmap, matrix, start_city=start))
    raw_candidates.append(nearest_neighbor_tour_from_matrix(matrix, start_city=0))

    best: list[int] | None = None
    best_length = math.inf
    seen: set[tuple[int, ...]] = set()
    for candidate in raw_candidates:
        if not is_valid_tour(num_cities, candidate):
            continue
        key = tuple(candidate)
        if key in seen:
            continue
        seen.add(key)
        improved = bounded_two_opt_from_matrix(matrix, candidate, max_evaluations=two_opt_budget)
        length = tour_length_from_matrix(matrix, improved)
        if best is None or length < best_length - EPS:
            best = improved
            best_length = length
    if best is None:
        best = list(range(num_cities))
    return canonicalize_tour(best, num_cities)


def solve(
    instance: dict[str, Any],
    trained_state: TrainedState,
    config: TspNeuralConstructorConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> list[int]:
    resolved_config = config if isinstance(config, TspNeuralConstructorConfig) else TspNeuralConstructorConfig.from_config(config)
    heatmap = _heatmap_from_model(trained_state.payload["model"], instance, resolved_config)
    return decode_tsp_heatmap(
        instance,
        heatmap,
        candidates=resolved_config.candidates,
        two_opt_budget=resolved_config.two_opt_budget,
    )


def build_tsp_neural_constructor_baselines(
    problem_name: str,
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    *,
    artifact_dir: str | Path | None = None,
    config: TspNeuralConstructorConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    if problem_name != "tsp" or not torch_available():
        return {}
    root = Path(artifact_dir) if artifact_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / f"{TSP_BASELINE_NAME}.pt" if root is not None else None
    metrics_path = root / f"{TSP_BASELINE_NAME}_metrics.jsonl" if root is not None else None
    resolved_config = config if isinstance(config, TspNeuralConstructorConfig) else TspNeuralConstructorConfig.from_config(config)
    trained_state = fit(
        train_instances,
        val_instances or [],
        resolved_config,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )

    def solver(instance: dict[str, Any]) -> list[int]:
        return solve(instance, trained_state, resolved_config)

    return {TSP_BASELINE_NAME: solver}
