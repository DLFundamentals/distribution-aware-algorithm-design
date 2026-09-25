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
from ml_baselines.tensorize import public_view, tensorize_graph_instance
from ml_baselines.torch_utils import TorchUnavailableError, require_torch, resolve_device, torch_available

PIGNN_MIS_BASELINE_NAME = "ml_pignn_mis"
EPS = 1e-9


@dataclass(frozen=True)
class PiGNNMISConfig:
    epochs: int = 200
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    layers: int = 3
    batch_size: int = 1
    qubo_penalty: float = 2.0
    temperature_start: float = 1.0
    temperature_end: float = 0.1
    inference_restarts: int = 8
    repair_budget: int = 128
    device: str = "cpu"
    seed: int = 0

    @classmethod
    def from_config(cls, config: MLBaselineConfig | dict[str, Any] | None) -> "PiGNNMISConfig":
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
            qubo_penalty=float(payload.get("qubo_penalty", payload.get("edge_penalty_weight", base.qubo_penalty))),
            temperature_start=float(payload.get("temperature_start", base.temperature_start)),
            temperature_end=float(payload.get("temperature_end", base.temperature_end)),
            inference_restarts=int(payload.get("inference_restarts", base.inference_restarts)),
            repair_budget=int(payload.get("repair_budget", base.repair_budget)),
            device=str(payload.get("device", base.device)),
            seed=int(payload.get("seed", base.seed)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "batch_size": self.batch_size,
            "qubo_penalty": self.qubo_penalty,
            "temperature_start": self.temperature_start,
            "temperature_end": self.temperature_end,
            "inference_restarts": self.inference_restarts,
            "repair_budget": self.repair_budget,
            "device": self.device,
            "seed": self.seed,
        }


def _adjacency(num_vertices: int, edges: list[list[int]] | list[tuple[int, int]]) -> list[set[int]]:
    adjacency = [set() for _ in range(num_vertices)]
    for raw_u, raw_v in edges:
        u = int(raw_u)
        v = int(raw_v)
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def _temperature(config: PiGNNMISConfig, epoch: int) -> float:
    if config.epochs <= 1:
        return max(float(config.temperature_end), EPS)
    fraction = min(1.0, max(0.0, float(epoch - 1) / float(config.epochs - 1)))
    value = config.temperature_start + fraction * (config.temperature_end - config.temperature_start)
    return max(float(value), EPS)


def mis_qubo_loss_from_probabilities(
    probabilities: Any,
    edge_index: Any,
    *,
    qubo_penalty: float,
) -> Any:
    torch = require_torch()
    node_term = probabilities.sum() / max(float(probabilities.shape[0]), 1.0)
    edge_term = torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)
    if edge_index.numel() > 0:
        sources = edge_index[0].long()
        targets = edge_index[1].long()
        mask = sources < targets
        if bool(mask.any()):
            edge_term = (probabilities[sources[mask]] * probabilities[targets[mask]]).sum()
            edge_term = edge_term / max(float(mask.sum().detach().cpu().item()), 1.0)
    return -node_term + float(qubo_penalty) * edge_term


def _loss(model: Any, tensor: Any, config: PiGNNMISConfig, *, epoch: int) -> Any:
    logits = model(tensor.node_features, tensor.edge_index).reshape(-1)
    probabilities = require_torch().sigmoid(logits / _temperature(config, epoch))
    return mis_qubo_loss_from_probabilities(probabilities, tensor.edge_index, qubo_penalty=config.qubo_penalty)


def _is_independent(adjacency: list[set[int]], selected: set[int]) -> bool:
    for vertex in selected:
        if not adjacency[vertex].isdisjoint(selected - {vertex}):
            return False
    return True


def _project_order_to_independent_set(
    adjacency: list[set[int]],
    order: list[int],
    *,
    initial_allowed: set[int] | None = None,
) -> set[int]:
    selected: set[int] = set()
    allowed = initial_allowed if initial_allowed is not None else set(range(len(adjacency)))
    for vertex in order:
        if vertex not in allowed:
            continue
        if adjacency[vertex].isdisjoint(selected):
            selected.add(vertex)
    return selected


def _local_improve(
    adjacency: list[set[int]],
    selected: set[int],
    scores: list[float],
    *,
    repair_budget: int,
) -> set[int]:
    budget = max(0, int(repair_budget))
    improved = True
    num_vertices = len(adjacency)
    while improved and budget > 0:
        improved = False
        excluded = sorted(
            (vertex for vertex in range(num_vertices) if vertex not in selected),
            key=lambda item: (float(scores[item]), -len(adjacency[item]), -item),
            reverse=True,
        )
        for pivot in sorted(selected, key=lambda item: (float(scores[item]), item)):
            candidates = [vertex for vertex in excluded if adjacency[vertex].isdisjoint(selected - {pivot})]
            for left_index, left in enumerate(candidates):
                if left in adjacency[pivot]:
                    continue
                for right in candidates[left_index + 1 :]:
                    budget -= 1
                    if right in adjacency[left] or right in adjacency[pivot]:
                        if budget <= 0:
                            break
                        continue
                    selected.remove(pivot)
                    selected.add(left)
                    selected.add(right)
                    improved = True
                    break
                if improved or budget <= 0:
                    break
            if improved or budget <= 0:
                break
    return selected


def decode_pignn_mis_scores(
    instance: dict[str, Any],
    scores: list[float],
    *,
    inference_restarts: int = 8,
    repair_budget: int = 128,
) -> list[int]:
    public = public_view(instance)
    num_vertices = int(public["num_vertices"])
    adjacency = _adjacency(num_vertices, public["edges"])
    if len(scores) < num_vertices:
        raise ValueError(f"Expected at least {num_vertices} MIS scores, got {len(scores)}.")
    degrees = [len(neighbors) for neighbors in adjacency]
    restarts = max(1, int(inference_restarts))
    orders: list[list[int]] = []
    orders.append(sorted(range(num_vertices), key=lambda vertex: (float(scores[vertex]), -degrees[vertex], -vertex), reverse=True))
    orders.append(sorted(range(num_vertices), key=lambda vertex: (float(scores[vertex]), degrees[vertex], -vertex), reverse=True))
    orders.append(sorted(range(num_vertices), key=lambda vertex: (float(scores[vertex]) / max(float(degrees[vertex] + 1), 1.0), -vertex), reverse=True))
    orders.append(sorted(range(num_vertices), key=lambda vertex: (degrees[vertex], float(scores[vertex]), -vertex)))
    while len(orders) < restarts:
        shift = len(orders)
        base = orders[shift % 4]
        orders.append(base[shift:] + base[:shift])

    best: set[int] = set()
    for order in orders[:restarts]:
        projected = _project_order_to_independent_set(adjacency, order)
        fill_order = sorted(
            (vertex for vertex in range(num_vertices) if vertex not in projected),
            key=lambda vertex: (float(scores[vertex]), -degrees[vertex], -vertex),
            reverse=True,
        )
        for vertex in fill_order:
            if adjacency[vertex].isdisjoint(projected):
                projected.add(vertex)
        improved = _local_improve(adjacency, set(projected), scores[:num_vertices], repair_budget=repair_budget)
        if len(improved) > len(best) or (
            len(improved) == len(best) and sorted(improved) < sorted(best)
        ):
            best = improved
    if not _is_independent(adjacency, best):
        best = _project_order_to_independent_set(adjacency, orders[0] if orders else list(range(num_vertices)))
    return sorted(best)


def fit(
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    config: PiGNNMISConfig | MLBaselineConfig | dict[str, Any] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> TrainedState:
    if not torch_available():
        raise TorchUnavailableError("PyTorch is required for the PI-GNN MIS baseline.")
    torch = require_torch()
    resolved_config = config if isinstance(config, PiGNNMISConfig) else PiGNNMISConfig.from_config(config)
    seed_everything(resolved_config.seed)
    device = resolve_device(resolved_config.device)
    if not train_instances:
        raise ValueError("The PI-GNN MIS baseline requires public training instances.")

    public_train = [public_view(instance) for instance in train_instances]
    public_val = [public_view(instance) for instance in (val_instances or [])]
    train_tensors = [tensorize_graph_instance(instance, backend="torch", device=str(device)) for instance in public_train]
    val_tensors = [tensorize_graph_instance(instance, backend="torch", device=str(device)) for instance in public_val]
    input_dim = int(train_tensors[0].node_features.shape[-1])
    model = ManualMessagePassing(input_dim, resolved_config.hidden_dim, 1, layers=resolved_config.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=resolved_config.learning_rate)
    logger = MetricsLogger(Path(metrics_path) if metrics_path is not None else None)
    best_state: dict[str, Any] | None = None
    best_loss = math.inf

    for epoch in range(1, resolved_config.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for tensor in train_tensors:
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, tensor, resolved_config, epoch=epoch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
        train_loss = sum(train_losses) / max(1, len(train_losses))
        val_loss = None
        if val_tensors:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for tensor in val_tensors:
                    loss = _loss(model, tensor, resolved_config, epoch=epoch)
                    val_losses.append(float(loss.detach().cpu().item()))
            val_loss = sum(val_losses) / max(1, len(val_losses))
        selection_loss = train_loss if val_loss is None else val_loss
        logger.log(epoch=epoch, train_loss=train_loss, val_loss=val_loss, temperature=_temperature(resolved_config, epoch))
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "problem": "mis",
        "baseline_name": PIGNN_MIS_BASELINE_NAME,
        "config": resolved_config.to_dict(),
        "train_instances": len(public_train),
        "validation_instances": len(public_val),
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


def _scores_from_model(model: Any, instance: dict[str, Any], config: PiGNNMISConfig) -> list[float]:
    torch = require_torch()
    device = resolve_device(config.device)
    tensor = tensorize_graph_instance(public_view(instance), backend="torch", device=str(device))
    model.eval()
    with torch.no_grad():
        logits = model(tensor.node_features, tensor.edge_index).reshape(-1)
        probabilities = torch.sigmoid(logits / max(float(config.temperature_end), EPS))
    return [float(value) for value in probabilities.detach().cpu().tolist()]


def solve(
    instance: dict[str, Any],
    trained_state: TrainedState,
    config: PiGNNMISConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> list[int]:
    resolved_config = config if isinstance(config, PiGNNMISConfig) else PiGNNMISConfig.from_config(config)
    scores = _scores_from_model(trained_state.payload["model"], instance, resolved_config)
    return decode_pignn_mis_scores(
        instance,
        scores,
        inference_restarts=resolved_config.inference_restarts,
        repair_budget=resolved_config.repair_budget,
    )


def build_pignn_mis_baselines(
    problem_name: str,
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    *,
    artifact_dir: str | Path | None = None,
    config: PiGNNMISConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    if problem_name != "mis" or not torch_available():
        return {}
    root = Path(artifact_dir) if artifact_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / f"{PIGNN_MIS_BASELINE_NAME}.pt" if root is not None else None
    metrics_path = root / f"{PIGNN_MIS_BASELINE_NAME}_metrics.jsonl" if root is not None else None
    resolved_config = config if isinstance(config, PiGNNMISConfig) else PiGNNMISConfig.from_config(config)
    trained_state = fit(
        train_instances,
        val_instances or [],
        resolved_config,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )

    def solver(instance: dict[str, Any]) -> list[int]:
        return solve(instance, trained_state, resolved_config)

    return {PIGNN_MIS_BASELINE_NAME: solver}
