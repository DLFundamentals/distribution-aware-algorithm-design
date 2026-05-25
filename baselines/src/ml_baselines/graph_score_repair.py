from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_baselines.checkpoint import save_checkpoint
from ml_baselines.config import MLBaselineConfig
from ml_baselines.interface import TrainedState
from ml_baselines.metrics import MetricsLogger
from ml_baselines.models import ManualMessagePassing
from ml_baselines.seeding import seed_everything
from ml_baselines.tensorize import tensorize_graph_instance
from ml_baselines.torch_utils import TorchUnavailableError, require_torch, resolve_device, torch_available

GRAPH_BASELINE_NAMES = {
    "mis": "ml_gnn_mis_score_repair",
    "mds": "ml_gnn_mds_score_repair",
    "coloring": "ml_gnn_coloring_priority",
}


@dataclass(frozen=True)
class GraphScoreRepairConfig:
    epochs: int = 50
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    layers: int = 3
    batch_size: int = 1
    inference_restarts: int = 4
    repair_budget: int = 64
    device: str = "cpu"
    seed: int = 0
    entropy_weight: float = 0.005
    edge_penalty_weight: float = 2.0
    domination_penalty_weight: float = 4.0
    coloring_pseudo_weight: float = 1.0

    @classmethod
    def from_config(cls, config: MLBaselineConfig | dict[str, Any] | None) -> "GraphScoreRepairConfig":
        base = cls()
        if config is None:
            return base
        payload = config.to_dict() if isinstance(config, MLBaselineConfig) else dict(config)
        return cls(
            epochs=int(payload.get("epochs", base.epochs)),
            learning_rate=float(payload.get("learning_rate", payload.get("lr", base.learning_rate))),
            hidden_dim=int(payload.get("hidden_dim", base.hidden_dim)),
            layers=int(payload.get("message_passing_layers", payload.get("layers", base.layers))),
            batch_size=int(payload.get("batch_size", base.batch_size)),
            inference_restarts=int(payload.get("inference_restarts", base.inference_restarts)),
            repair_budget=int(payload.get("repair_budget", base.repair_budget)),
            device=str(payload.get("device", base.device)),
            seed=int(payload.get("seed", base.seed)),
            entropy_weight=float(payload.get("entropy_weight", base.entropy_weight)),
            edge_penalty_weight=float(payload.get("edge_penalty_weight", base.edge_penalty_weight)),
            domination_penalty_weight=float(payload.get("domination_penalty_weight", base.domination_penalty_weight)),
            coloring_pseudo_weight=float(payload.get("coloring_pseudo_weight", base.coloring_pseudo_weight)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "batch_size": self.batch_size,
            "inference_restarts": self.inference_restarts,
            "repair_budget": self.repair_budget,
            "device": self.device,
            "seed": self.seed,
            "entropy_weight": self.entropy_weight,
            "edge_penalty_weight": self.edge_penalty_weight,
            "domination_penalty_weight": self.domination_penalty_weight,
            "coloring_pseudo_weight": self.coloring_pseudo_weight,
        }


def _adjacency(num_vertices: int, edges: list[list[int]] | list[tuple[int, int]]) -> list[set[int]]:
    adjacency = [set() for _ in range(num_vertices)]
    for raw_u, raw_v in edges:
        u = int(raw_u)
        v = int(raw_v)
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def decode_mis_scores(instance: dict[str, Any], scores: list[float], *, repair_budget: int = 64) -> list[int]:
    num_vertices = int(instance["num_vertices"])
    adjacency = _adjacency(num_vertices, instance["edges"])
    order = sorted(
        range(num_vertices),
        key=lambda vertex: (float(scores[vertex]), -len(adjacency[vertex]), -vertex),
        reverse=True,
    )
    selected: set[int] = set()
    for vertex in order:
        if adjacency[vertex].isdisjoint(selected):
            selected.add(vertex)
    fill_order = sorted(
        (vertex for vertex in range(num_vertices) if vertex not in selected),
        key=lambda vertex: (len(adjacency[vertex]), -float(scores[vertex]), vertex),
    )
    for vertex in fill_order[: max(0, int(repair_budget))]:
        if adjacency[vertex].isdisjoint(selected):
            selected.add(vertex)
    return sorted(selected)


def decode_mds_scores(instance: dict[str, Any], scores: list[float], *, repair_budget: int = 64) -> list[int]:
    num_vertices = int(instance["num_vertices"])
    adjacency = _adjacency(num_vertices, instance["edges"])
    closed = [neighbors | {vertex} for vertex, neighbors in enumerate(adjacency)]
    dominated: set[int] = set()
    selected: set[int] = set()
    budget = max(1, int(repair_budget))
    while len(dominated) < num_vertices and len(selected) < num_vertices:
        candidates = [vertex for vertex in range(num_vertices) if vertex not in selected]
        vertex = max(
            candidates,
            key=lambda item: (
                len(closed[item] - dominated),
                float(scores[item]),
                -item,
            ),
        )
        selected.add(vertex)
        dominated.update(closed[vertex])
        budget -= 1
        if budget <= 0 and len(dominated) < num_vertices:
            budget += num_vertices
    for vertex in sorted(selected, key=lambda item: (float(scores[item]), item)):
        candidate = selected - {vertex}
        candidate_dominated: set[int] = set()
        for item in candidate:
            candidate_dominated.update(closed[item])
        if len(candidate_dominated) == num_vertices:
            selected = candidate
    return sorted(selected)


def _greedy_coloring_from_order(instance: dict[str, Any], order: list[int]) -> list[int]:
    num_vertices = int(instance["num_vertices"])
    adjacency = _adjacency(num_vertices, instance["edges"])
    colors = [-1] * num_vertices
    for vertex in order:
        used = {colors[neighbor] for neighbor in adjacency[vertex] if colors[neighbor] >= 0}
        color = 0
        while color in used:
            color += 1
        colors[vertex] = color
    return colors


def _relabel_colors(colors: list[int]) -> list[int]:
    relabel = {color: index for index, color in enumerate(sorted(set(colors)))}
    return [relabel[color] for color in colors]


def _try_recolor_down(instance: dict[str, Any], colors: list[int], *, budget: int) -> list[int]:
    num_vertices = int(instance["num_vertices"])
    adjacency = _adjacency(num_vertices, instance["edges"])
    remaining_budget = max(0, int(budget))
    while remaining_budget > 0 and colors:
        improved = False
        max_color = max(colors)
        for vertex in [item for item in range(num_vertices) if colors[item] == max_color]:
            current = colors[vertex]
            for candidate_color in range(current):
                if all(colors[neighbor] != candidate_color for neighbor in adjacency[vertex]):
                    colors[vertex] = candidate_color
                    improved = True
                    break
            remaining_budget -= 1
            if remaining_budget <= 0:
                break
        colors = _relabel_colors(colors)
        if not improved:
            break
    return colors


def decode_coloring_scores(instance: dict[str, Any], scores: list[float], *, repair_budget: int = 64) -> list[int]:
    num_vertices = int(instance["num_vertices"])
    degrees = [0] * num_vertices
    for raw_u, raw_v in instance["edges"]:
        degrees[int(raw_u)] += 1
        degrees[int(raw_v)] += 1
    order = sorted(
        range(num_vertices),
        key=lambda vertex: (float(scores[vertex]), degrees[vertex], -vertex),
        reverse=True,
    )
    colors = _greedy_coloring_from_order(instance, order)
    return _try_recolor_down(instance, colors, budget=repair_budget)


def _entropy(probabilities: Any) -> Any:
    torch = require_torch()
    clipped = probabilities.clamp(1e-6, 1.0 - 1e-6)
    return -(clipped * torch.log(clipped) + (1.0 - clipped) * torch.log(1.0 - clipped)).mean()


def _mis_loss(model: Any, tensor: Any, config: GraphScoreRepairConfig) -> Any:
    torch = require_torch()
    logits = model(tensor.node_features, tensor.edge_index).reshape(-1)
    probabilities = torch.sigmoid(logits)
    edge_penalty = torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)
    if tensor.edge_index.numel() > 0:
        sources = tensor.edge_index[0]
        targets = tensor.edge_index[1]
        mask = sources < targets
        if bool(mask.any()):
            edge_penalty = (probabilities[sources[mask]] * probabilities[targets[mask]]).sum()
    return -probabilities.sum() + config.edge_penalty_weight * edge_penalty - config.entropy_weight * _entropy(probabilities)


def _mds_loss(model: Any, tensor: Any, instance: dict[str, Any], config: GraphScoreRepairConfig) -> Any:
    torch = require_torch()
    logits = model(tensor.node_features, tensor.edge_index).reshape(-1)
    probabilities = torch.sigmoid(logits)
    adjacency = _adjacency(int(instance["num_vertices"]), instance["edges"])
    uncovered_terms = []
    for vertex, neighbors in enumerate(adjacency):
        closed = [vertex, *sorted(neighbors)]
        uncovered_terms.append(torch.prod(1.0 - probabilities[closed]))
    uncovered = torch.stack(uncovered_terms).sum() if uncovered_terms else torch.zeros((), device=probabilities.device)
    return probabilities.sum() + config.domination_penalty_weight * uncovered - config.entropy_weight * _entropy(probabilities)


def _coloring_targets(instance: dict[str, Any], device: Any) -> Any:
    torch = require_torch()
    try:
        from dasbench.problems.graph_utils import dsatur_coloring

        colors = dsatur_coloring(instance)
    except Exception:
        colors = _greedy_coloring_from_order(instance, list(range(int(instance["num_vertices"]))))
    degrees = [0] * int(instance["num_vertices"])
    for raw_u, raw_v in instance["edges"]:
        degrees[int(raw_u)] += 1
        degrees[int(raw_v)] += 1
    max_color = max(colors) if colors else 0
    max_degree = max(degrees) if degrees else 1
    targets = [
        float(max_color - colors[vertex]) + 0.1 * float(degrees[vertex]) / max(float(max_degree), 1.0)
        for vertex in range(len(colors))
    ]
    if targets:
        minimum = min(targets)
        scale = max(max(targets) - minimum, 1e-9)
        targets = [(value - minimum) / scale for value in targets]
    return torch.tensor(targets, dtype=torch.float32, device=device)


def _coloring_loss(model: Any, tensor: Any, instance: dict[str, Any], config: GraphScoreRepairConfig) -> Any:
    torch = require_torch()
    logits = model(tensor.node_features, tensor.edge_index).reshape(-1)
    target = _coloring_targets(instance, logits.device)
    mse = torch.mean((logits - target) ** 2)
    if tensor.edge_index.numel() == 0:
        return mse
    sources = tensor.edge_index[0]
    targets = tensor.edge_index[1]
    mask = sources < targets
    if not bool(mask.any()):
        return mse
    edge_margin = torch.relu(0.05 - torch.abs(logits[sources[mask]] - logits[targets[mask]])).mean()
    return config.coloring_pseudo_weight * mse + edge_margin


def _loss_for_problem(
    problem_name: str,
    model: Any,
    tensor: Any,
    instance: dict[str, Any],
    config: GraphScoreRepairConfig,
) -> Any:
    if problem_name == "mis":
        return _mis_loss(model, tensor, config)
    if problem_name == "mds":
        return _mds_loss(model, tensor, instance, config)
    if problem_name == "coloring":
        return _coloring_loss(model, tensor, instance, config)
    raise ValueError(f"Unsupported graph ML problem: {problem_name}")


def fit(
    problem_name: str,
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    config: GraphScoreRepairConfig | MLBaselineConfig | dict[str, Any] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> TrainedState:
    if problem_name not in GRAPH_BASELINE_NAMES:
        raise ValueError(f"Unsupported graph ML problem: {problem_name}")
    if not torch_available():
        raise TorchUnavailableError("PyTorch is required for graph neural score-and-repair baselines.")
    torch = require_torch()
    resolved_config = config if isinstance(config, GraphScoreRepairConfig) else GraphScoreRepairConfig.from_config(config)
    seed_everything(resolved_config.seed)
    device = resolve_device(resolved_config.device)
    if not train_instances:
        raise ValueError("Graph ML baselines require at least one public training instance.")

    train_tensors = [tensorize_graph_instance(instance, backend="torch", device=str(device)) for instance in train_instances]
    val_tensors = [tensorize_graph_instance(instance, backend="torch", device=str(device)) for instance in (val_instances or [])]
    input_dim = int(train_tensors[0].node_features.shape[-1])
    model = ManualMessagePassing(input_dim, resolved_config.hidden_dim, 1, layers=resolved_config.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=resolved_config.learning_rate)
    logger = MetricsLogger(Path(metrics_path) if metrics_path is not None else None)
    best_state: dict[str, Any] | None = None
    best_train_loss = math.inf

    for epoch in range(1, resolved_config.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for instance, tensor in zip(train_instances, train_tensors, strict=True):
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_for_problem(problem_name, model, tensor, instance, resolved_config)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
        train_loss = sum(train_losses) / max(1, len(train_losses))
        val_loss = None
        if val_tensors:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for instance, tensor in zip(val_instances or [], val_tensors, strict=True):
                    loss = _loss_for_problem(problem_name, model, tensor, instance, resolved_config)
                    val_losses.append(float(loss.detach().cpu().item()))
            val_loss = sum(val_losses) / max(1, len(val_losses))
        logger.log(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "problem": problem_name,
        "baseline_name": GRAPH_BASELINE_NAMES[problem_name],
        "config": resolved_config.to_dict(),
        "train_instances": len(train_instances),
        "validation_instances": len(val_instances or []),
        "input_dim": input_dim,
        "history": logger.rows,
    }
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            {"model_state_dict": model.state_dict(), "input_dim": input_dim},
            metadata=metadata,
            use_torch=True,
        )
    return TrainedState(payload={"model": model, "input_dim": input_dim}, metadata=metadata)


def _scores_from_model(model: Any, instance: dict[str, Any], config: GraphScoreRepairConfig) -> list[float]:
    torch = require_torch()
    device = resolve_device(config.device)
    tensor = tensorize_graph_instance(instance, backend="torch", device=str(device))
    model.eval()
    with torch.no_grad():
        logits = model(tensor.node_features, tensor.edge_index).reshape(-1)
        return [float(value) for value in logits.detach().cpu().tolist()]


def solve(
    problem_name: str,
    instance: dict[str, Any],
    trained_state: TrainedState,
    config: GraphScoreRepairConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> list[int]:
    resolved_config = config if isinstance(config, GraphScoreRepairConfig) else GraphScoreRepairConfig.from_config(config)
    scores = _scores_from_model(trained_state.payload["model"], instance, resolved_config)
    if problem_name == "mis":
        return decode_mis_scores(instance, scores, repair_budget=resolved_config.repair_budget)
    if problem_name == "mds":
        return decode_mds_scores(instance, scores, repair_budget=resolved_config.repair_budget)
    if problem_name == "coloring":
        return decode_coloring_scores(instance, scores, repair_budget=resolved_config.repair_budget)
    raise ValueError(f"Unsupported graph ML problem: {problem_name}")


def build_graph_score_repair_baselines(
    problem_name: str,
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    *,
    artifact_dir: str | Path | None = None,
    config: GraphScoreRepairConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    if problem_name not in GRAPH_BASELINE_NAMES or not torch_available():
        return {}
    baseline_name = GRAPH_BASELINE_NAMES[problem_name]
    root = Path(artifact_dir) if artifact_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / f"{baseline_name}.pt" if root is not None else None
    metrics_path = root / f"{baseline_name}_metrics.jsonl" if root is not None else None
    resolved_config = config if isinstance(config, GraphScoreRepairConfig) else GraphScoreRepairConfig.from_config(config)
    trained_state = fit(
        problem_name,
        train_instances,
        val_instances or [],
        resolved_config,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )

    def solver(instance: dict[str, Any]) -> list[int]:
        return solve(problem_name, instance, trained_state, resolved_config)

    return {baseline_name: solver}
