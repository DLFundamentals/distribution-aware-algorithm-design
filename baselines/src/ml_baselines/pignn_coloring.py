from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dasbench.problems.graph_utils import canonicalize_coloring, dsatur_coloring

from ml_baselines.checkpoint import save_checkpoint
from ml_baselines.config import MLBaselineConfig
from ml_baselines.interface import TrainedState
from ml_baselines.metrics import MetricsLogger
from ml_baselines.models import ManualMessagePassing
from ml_baselines.seeding import seed_everything
from ml_baselines.tensorize import public_view, tensorize_graph_instance
from ml_baselines.torch_utils import TorchUnavailableError, require_torch, resolve_device, torch_available

PIGNN_COLORING_BASELINE_NAME = "ml_pignn_coloring"
EPS = 1e-9


@dataclass(frozen=True)
class PiGNNColoringConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    layers: int = 3
    batch_size: int = 1
    max_colors: int = 0
    color_slack: int = 0
    min_train_colors: int = 2
    min_decode_colors: int = 1
    inference_restarts: int = 8
    repair_budget: int = 256
    temperature_start: float = 1.0
    temperature_end: float = 0.2
    entropy_weight: float = 0.01
    balance_weight: float = 0.0
    train_color_count_mode: str = "cycle"
    device: str = "cpu"
    seed: int = 0

    @classmethod
    def from_config(cls, config: MLBaselineConfig | dict[str, Any] | None) -> "PiGNNColoringConfig":
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
            max_colors=int(payload.get("max_colors", base.max_colors)),
            color_slack=int(payload.get("color_slack", base.color_slack)),
            min_train_colors=int(payload.get("min_train_colors", base.min_train_colors)),
            min_decode_colors=int(payload.get("min_decode_colors", base.min_decode_colors)),
            inference_restarts=int(payload.get("inference_restarts", base.inference_restarts)),
            repair_budget=int(payload.get("repair_budget", base.repair_budget)),
            temperature_start=float(payload.get("temperature_start", base.temperature_start)),
            temperature_end=float(payload.get("temperature_end", base.temperature_end)),
            entropy_weight=float(payload.get("entropy_weight", base.entropy_weight)),
            balance_weight=float(payload.get("balance_weight", base.balance_weight)),
            train_color_count_mode=str(payload.get("train_color_count_mode", base.train_color_count_mode)),
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
            "max_colors": self.max_colors,
            "color_slack": self.color_slack,
            "min_train_colors": self.min_train_colors,
            "min_decode_colors": self.min_decode_colors,
            "inference_restarts": self.inference_restarts,
            "repair_budget": self.repair_budget,
            "temperature_start": self.temperature_start,
            "temperature_end": self.temperature_end,
            "entropy_weight": self.entropy_weight,
            "balance_weight": self.balance_weight,
            "train_color_count_mode": self.train_color_count_mode,
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


def _is_valid_coloring(
    num_vertices: int,
    edges: list[list[int]] | list[tuple[int, int]],
    colors: list[int],
) -> bool:
    if len(colors) != num_vertices or any(color < 0 for color in colors):
        return False
    return all(colors[int(u)] != colors[int(v)] for u, v in edges)


def _dsatur_color_count(instance: dict[str, Any]) -> int:
    return max(1, len(set(dsatur_coloring(public_view(instance)))))


def _resolve_max_colors(
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]],
    config: PiGNNColoringConfig,
) -> int:
    if config.max_colors > 0:
        return max(1, int(config.max_colors))
    counts = [_dsatur_color_count(instance) for instance in [*train_instances, *val_instances]]
    return max(1, max(counts, default=1) + max(0, int(config.color_slack)))


def _temperature(config: PiGNNColoringConfig, epoch: int) -> float:
    if config.epochs <= 1:
        return max(float(config.temperature_end), EPS)
    fraction = min(1.0, max(0.0, float(epoch - 1) / float(config.epochs - 1)))
    value = config.temperature_start + fraction * (config.temperature_end - config.temperature_start)
    return max(float(value), EPS)


def _training_color_count(max_colors: int, config: PiGNNColoringConfig, *, epoch: int, instance_index: int) -> int:
    if max_colors <= 1:
        return 1
    lower = min(max(2, int(config.min_train_colors)), max_colors)
    if config.train_color_count_mode == "max" or lower == max_colors:
        return max_colors
    span = max_colors - lower + 1
    return lower + ((epoch + instance_index) % span)


def _potts_loss(
    model: Any,
    tensor: Any,
    *,
    color_count: int,
    temperature: float,
    config: PiGNNColoringConfig,
) -> Any:
    torch = require_torch()
    logits = model(tensor.node_features, tensor.edge_index)[:, :color_count]
    probabilities = torch.softmax(logits / max(float(temperature), EPS), dim=-1)
    conflict = torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)
    if tensor.edge_index.numel() > 0:
        sources = tensor.edge_index[0].long()
        targets = tensor.edge_index[1].long()
        mask = sources < targets
        if bool(mask.any()):
            edge_conflicts = (probabilities[sources[mask]] * probabilities[targets[mask]]).sum(dim=-1)
            conflict = edge_conflicts.mean()
    clipped = probabilities.clamp(1e-8, 1.0)
    entropy = -(clipped * torch.log(clipped)).sum(dim=-1).mean()
    balance = torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)
    if config.balance_weight > 0.0 and color_count > 1:
        color_mass = probabilities.mean(dim=0)
        target = torch.full_like(color_mass, 1.0 / float(color_count))
        balance = ((color_mass - target) ** 2).mean()
    return conflict + config.entropy_weight * entropy + config.balance_weight * balance


def _mean_loss(
    model: Any,
    tensors: list[Any],
    *,
    max_colors: int,
    config: PiGNNColoringConfig,
    epoch: int,
) -> float:
    if not tensors:
        return math.inf
    torch = require_torch()
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for index, tensor in enumerate(tensors):
            color_count = _training_color_count(max_colors, config, epoch=epoch, instance_index=index)
            loss = _potts_loss(
                model,
                tensor,
                color_count=color_count,
                temperature=_temperature(config, epoch),
                config=config,
            )
            losses.append(float(loss.detach().cpu().item()))
    return sum(losses) / max(1, len(losses))


def _fixed_k_initial_coloring(
    instance: dict[str, Any],
    logits: list[list[float]],
    *,
    color_count: int,
    restart: int,
) -> list[int]:
    num_vertices = int(instance["num_vertices"])
    adjacency = _adjacency(num_vertices, instance["edges"])
    degrees = [len(neighbors) for neighbors in adjacency]

    def confidence(vertex: int) -> float:
        scores = list(logits[vertex][:color_count])
        if not scores:
            return 0.0
        ordered = sorted(scores, reverse=True)
        return ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)

    if restart % 3 == 0:
        order = sorted(range(num_vertices), key=lambda vertex: (confidence(vertex), degrees[vertex], -vertex), reverse=True)
    elif restart % 3 == 1:
        order = sorted(range(num_vertices), key=lambda vertex: (degrees[vertex], confidence(vertex), -vertex), reverse=True)
    else:
        order = sorted(range(num_vertices), key=lambda vertex: (-confidence(vertex), degrees[vertex], -vertex), reverse=True)

    colors = [-1] * num_vertices
    for vertex in order:
        neighbor_colors = {colors[neighbor] for neighbor in adjacency[vertex] if colors[neighbor] >= 0}
        feasible = [color for color in range(color_count) if color not in neighbor_colors]
        if feasible:
            colors[vertex] = max(feasible, key=lambda color: (float(logits[vertex][color]), -color))
            continue
        colors[vertex] = min(
            range(color_count),
            key=lambda color: (
                sum(1 for neighbor in adjacency[vertex] if colors[neighbor] == color),
                -float(logits[vertex][color]),
                color,
            ),
        )
    return colors


def _conflict_edges(instance: dict[str, Any], colors: list[int]) -> list[tuple[int, int]]:
    return [
        (int(u), int(v))
        for u, v in instance["edges"]
        if colors[int(u)] == colors[int(v)]
    ]


def _repair_fixed_k_coloring(
    instance: dict[str, Any],
    colors: list[int],
    logits: list[list[float]],
    *,
    color_count: int,
    repair_budget: int,
) -> list[int] | None:
    num_vertices = int(instance["num_vertices"])
    adjacency = _adjacency(num_vertices, instance["edges"])
    remaining_budget = max(0, int(repair_budget))
    while remaining_budget >= 0:
        conflicts = _conflict_edges(instance, colors)
        if not conflicts:
            return canonicalize_coloring(colors, num_vertices)
        if remaining_budget <= 0:
            break
        conflict_counts = [0] * num_vertices
        for u, v in conflicts:
            conflict_counts[u] += 1
            conflict_counts[v] += 1
        vertex = max(
            {vertex for edge in conflicts for vertex in edge},
            key=lambda item: (conflict_counts[item], len(adjacency[item]), -item),
        )
        current_color = colors[vertex]
        best_color = current_color
        best_key = (
            conflict_counts[vertex],
            -float(logits[vertex][current_color]) if current_color >= 0 else 0.0,
            current_color,
        )
        for candidate_color in range(color_count):
            candidate_conflicts = sum(1 for neighbor in adjacency[vertex] if colors[neighbor] == candidate_color)
            key = (candidate_conflicts, -float(logits[vertex][candidate_color]), candidate_color)
            if key < best_key:
                best_key = key
                best_color = candidate_color
        if best_color == current_color:
            break
        colors[vertex] = best_color
        remaining_budget -= 1
    return None


def decode_fixed_k_coloring(
    instance: dict[str, Any],
    logits: list[list[float]],
    *,
    color_count: int,
    repair_budget: int = 256,
    restart: int = 0,
) -> list[int] | None:
    public = public_view(instance)
    num_vertices = int(public["num_vertices"])
    if color_count <= 0:
        return None
    if num_vertices == 0:
        return []
    if color_count == 1 and public["edges"]:
        return None
    colors = _fixed_k_initial_coloring(public, logits, color_count=color_count, restart=restart)
    repaired = _repair_fixed_k_coloring(public, colors, logits, color_count=color_count, repair_budget=repair_budget)
    if repaired is not None and _is_valid_coloring(num_vertices, public["edges"], repaired):
        return repaired
    return None


def decode_pignn_coloring(
    instance: dict[str, Any],
    logits: list[list[float]],
    *,
    max_colors: int,
    config: PiGNNColoringConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> list[int]:
    resolved_config = config if isinstance(config, PiGNNColoringConfig) else PiGNNColoringConfig.from_config(config)
    public = public_view(instance)
    num_vertices = int(public["num_vertices"])
    fallback = dsatur_coloring(public)
    best = canonicalize_coloring(fallback, num_vertices)
    upper_bound = min(max(1, int(max_colors)), max(1, len(set(best))))
    lower_bound = max(1, min(int(resolved_config.min_decode_colors), upper_bound))
    for color_count in range(upper_bound, lower_bound - 1, -1):
        for restart in range(max(1, int(resolved_config.inference_restarts))):
            candidate = decode_fixed_k_coloring(
                public,
                logits,
                color_count=color_count,
                repair_budget=resolved_config.repair_budget,
                restart=restart,
            )
            if candidate is None:
                continue
            if len(set(candidate)) < len(set(best)):
                best = candidate
            break
    return canonicalize_coloring(best, num_vertices)


def fit(
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    config: PiGNNColoringConfig | MLBaselineConfig | dict[str, Any] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> TrainedState:
    if not torch_available():
        raise TorchUnavailableError("PyTorch is required for the PI-GNN coloring baseline.")
    torch = require_torch()
    resolved_config = config if isinstance(config, PiGNNColoringConfig) else PiGNNColoringConfig.from_config(config)
    seed_everything(resolved_config.seed)
    device = resolve_device(resolved_config.device)
    if not train_instances:
        raise ValueError("The PI-GNN coloring baseline requires public training instances.")

    public_train = [public_view(instance) for instance in train_instances]
    public_val = [public_view(instance) for instance in (val_instances or [])]
    max_colors = _resolve_max_colors(public_train, public_val, resolved_config)
    train_tensors = [tensorize_graph_instance(instance, backend="torch", device=str(device)) for instance in public_train]
    val_tensors = [tensorize_graph_instance(instance, backend="torch", device=str(device)) for instance in public_val]
    input_dim = int(train_tensors[0].node_features.shape[-1])
    model = ManualMessagePassing(input_dim, resolved_config.hidden_dim, max_colors, layers=resolved_config.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=resolved_config.learning_rate)
    logger = MetricsLogger(Path(metrics_path) if metrics_path is not None else None)
    best_state: dict[str, Any] | None = None
    best_loss = math.inf

    for epoch in range(1, resolved_config.epochs + 1):
        model.train()
        temperature = _temperature(resolved_config, epoch)
        train_losses: list[float] = []
        for index, tensor in enumerate(train_tensors):
            color_count = _training_color_count(max_colors, resolved_config, epoch=epoch, instance_index=index)
            optimizer.zero_grad(set_to_none=True)
            loss = _potts_loss(
                model,
                tensor,
                color_count=color_count,
                temperature=temperature,
                config=resolved_config,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
        train_loss = sum(train_losses) / max(1, len(train_losses))
        val_loss = _mean_loss(model, val_tensors, max_colors=max_colors, config=resolved_config, epoch=epoch) if val_tensors else None
        selection_loss = train_loss if val_loss is None else val_loss
        logger.log(epoch=epoch, train_loss=train_loss, val_loss=val_loss, temperature=temperature, max_colors=max_colors)
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "problem": "coloring",
        "baseline_name": PIGNN_COLORING_BASELINE_NAME,
        "config": resolved_config.to_dict(),
        "train_instances": len(train_instances),
        "validation_instances": len(val_instances or []),
        "input_dim": input_dim,
        "max_colors": max_colors,
        "history": logger.rows,
    }
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            {"model_state_dict": model.state_dict(), "input_dim": input_dim, "max_colors": max_colors},
            metadata=metadata,
            use_torch=True,
        )
    return TrainedState(payload={"model": model, "input_dim": input_dim, "max_colors": max_colors}, metadata=metadata)


def _logits_from_model(model: Any, instance: dict[str, Any], config: PiGNNColoringConfig) -> list[list[float]]:
    device = resolve_device(config.device)
    tensor = tensorize_graph_instance(public_view(instance), backend="torch", device=str(device))
    model.eval()
    with require_torch().no_grad():
        logits = model(tensor.node_features, tensor.edge_index)
        return [[float(value) for value in row] for row in logits.detach().cpu().tolist()]


def solve(
    instance: dict[str, Any],
    trained_state: TrainedState,
    config: PiGNNColoringConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> list[int]:
    resolved_config = config if isinstance(config, PiGNNColoringConfig) else PiGNNColoringConfig.from_config(config)
    logits = _logits_from_model(trained_state.payload["model"], instance, resolved_config)
    return decode_pignn_coloring(
        instance,
        logits,
        max_colors=int(trained_state.payload["max_colors"]),
        config=resolved_config,
    )


def build_pignn_coloring_baselines(
    problem_name: str,
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    *,
    artifact_dir: str | Path | None = None,
    config: PiGNNColoringConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    if problem_name != "coloring" or not torch_available():
        return {}
    root = Path(artifact_dir) if artifact_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / f"{PIGNN_COLORING_BASELINE_NAME}.pt" if root is not None else None
    metrics_path = root / f"{PIGNN_COLORING_BASELINE_NAME}_metrics.jsonl" if root is not None else None
    resolved_config = config if isinstance(config, PiGNNColoringConfig) else PiGNNColoringConfig.from_config(config)
    trained_state = fit(
        train_instances,
        val_instances or [],
        resolved_config,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )

    def solver(instance: dict[str, Any]) -> list[int]:
        return solve(instance, trained_state, resolved_config)

    return {PIGNN_COLORING_BASELINE_NAME: solver}
