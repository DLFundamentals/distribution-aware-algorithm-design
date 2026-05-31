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
from ml_baselines.tensorize import public_view, tensorize_packing_instance
from ml_baselines.torch_utils import TorchUnavailableError, require_torch, resolve_device, torch_available

PDL_PACKINGLP_BASELINE_NAME = "ml_pdl_packinglp"
EPS = 1e-9
FEASIBILITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PDLPackingLPConfig:
    epochs: int = 200
    learning_rate: float = 1e-3
    dual_learning_rate: float = 1e-3
    hidden_dim: int = 64
    layers: int = 3
    batch_size: int = 1
    rho: float = 10.0
    rho_growth: float = 1.05
    dual_feasibility_weight: float = 1.0
    dual_gap_weight: float = 0.1
    repair_budget: int = 128
    device: str = "cpu"
    seed: int = 0

    @classmethod
    def from_config(cls, config: MLBaselineConfig | dict[str, Any] | None) -> "PDLPackingLPConfig":
        base = cls()
        if config is None:
            return base
        payload = config.to_dict() if isinstance(config, MLBaselineConfig) else dict(config)
        return cls(
            epochs=int(payload.get("epochs", base.epochs)),
            learning_rate=float(payload.get("learning_rate", payload.get("lr", base.learning_rate))),
            dual_learning_rate=float(payload.get("dual_learning_rate", base.dual_learning_rate)),
            hidden_dim=int(payload.get("hidden_dim", base.hidden_dim)),
            layers=int(payload.get("message_passing_layers", payload.get("layers", base.layers))),
            batch_size=int(payload.get("batch_size", base.batch_size)),
            rho=float(payload.get("rho", payload.get("violation_penalty", base.rho))),
            rho_growth=float(payload.get("rho_growth", base.rho_growth)),
            dual_feasibility_weight=float(payload.get("dual_feasibility_weight", base.dual_feasibility_weight)),
            dual_gap_weight=float(payload.get("dual_gap_weight", base.dual_gap_weight)),
            repair_budget=int(payload.get("repair_budget", base.repair_budget)),
            device=str(payload.get("device", base.device)),
            seed=int(payload.get("seed", base.seed)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "dual_learning_rate": self.dual_learning_rate,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "batch_size": self.batch_size,
            "rho": self.rho,
            "rho_growth": self.rho_growth,
            "dual_feasibility_weight": self.dual_feasibility_weight,
            "dual_gap_weight": self.dual_gap_weight,
            "repair_budget": self.repair_budget,
            "device": self.device,
            "seed": self.seed,
        }


class PDLPackingLPNet:
    """Small primal-dual item/resource message passing network."""

    def __new__(cls, *args: Any, **kwargs: Any):
        torch = require_torch()
        nn = torch.nn

        class _PDLPackingLPNet(nn.Module):
            def __init__(self, item_dim: int, resource_dim: int, hidden_dim: int, layers: int) -> None:
                super().__init__()
                self.item_input = nn.Linear(item_dim, hidden_dim)
                self.resource_input = nn.Linear(resource_dim, hidden_dim)
                self.item_to_resource = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(max(1, int(layers))))
                self.resource_to_item = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(max(1, int(layers))))
                self.item_updates = nn.ModuleList(nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(max(1, int(layers))))
                self.resource_updates = nn.ModuleList(nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(max(1, int(layers))))
                self.primal_head = nn.Linear(hidden_dim, 1)
                self.dual_head = nn.Linear(hidden_dim, 1)
                self.activation = nn.ReLU()

            def forward(self, tensor: Any) -> tuple[Any, Any]:
                torch = require_torch()
                item_hidden = self.activation(self.item_input(tensor.item_features))
                resource_hidden = self.activation(self.resource_input(tensor.resource_features))
                normalized_weights = tensor.weights / tensor.capacities.clamp_min(EPS).reshape(1, -1)
                item_degree = normalized_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
                resource_degree = normalized_weights.sum(dim=0, keepdim=True).transpose(0, 1).clamp_min(1.0)
                for i2r, r2i, item_update, resource_update in zip(
                    self.item_to_resource,
                    self.resource_to_item,
                    self.item_updates,
                    self.resource_updates,
                    strict=True,
                ):
                    item_messages = self.activation(i2r(item_hidden))
                    resource_aggregate = normalized_weights.transpose(0, 1) @ item_messages / resource_degree
                    resource_hidden = self.activation(resource_update(torch.cat([resource_hidden, resource_aggregate], dim=-1)))
                    resource_messages = self.activation(r2i(resource_hidden))
                    item_aggregate = normalized_weights @ resource_messages / item_degree
                    item_hidden = self.activation(item_update(torch.cat([item_hidden, item_aggregate], dim=-1)))
                primal = torch.sigmoid(self.primal_head(item_hidden).reshape(-1))
                dual = torch.nn.functional.softplus(self.dual_head(resource_hidden).reshape(-1))
                return primal, dual

        return _PDLPackingLPNet(*args, **kwargs)


def _values_weights_capacities(instance: dict[str, Any]) -> tuple[list[float], list[list[float]], list[float]]:
    public = public_view(instance)
    return (
        [float(value) for value in public["values"]],
        [[float(value) for value in row] for row in public["weights"]],
        [float(value) for value in public["capacities"]],
    )


def _resource_usage(weights: list[list[float]], vector: list[float], num_resources: int) -> list[float]:
    usage = [0.0] * num_resources
    for amount, row in zip(vector, weights, strict=True):
        for resource in range(num_resources):
            usage[resource] += float(amount) * row[resource]
    return usage


def _density(values: list[float], weights: list[list[float]], capacities: list[float], item: int, dual_prices: list[float] | None) -> float:
    prices = dual_prices if dual_prices is not None else [1.0] * len(capacities)
    pressure = sum(prices[resource] * weights[item][resource] / max(capacities[resource], EPS) for resource in range(len(capacities)))
    return values[item] / max(pressure, FEASIBILITY_TOLERANCE)


def project_and_fill_packinglp(
    instance: dict[str, Any],
    fractions: list[float],
    *,
    dual_prices: list[float] | None = None,
    repair_budget: int = 128,
) -> list[float]:
    public = public_view(instance)
    num_items = int(public["num_items"])
    num_resources = int(public["num_resources"])
    values, weights, capacities = _values_weights_capacities(public)
    solution = [min(1.0, max(0.0, float(value))) for value in fractions[:num_items]]
    if len(solution) != num_items:
        raise ValueError(f"Expected {num_items} item fractions, got {len(fractions)}.")

    usage = _resource_usage(weights, solution, num_resources)
    scale = 1.0
    for resource in range(num_resources):
        if usage[resource] > capacities[resource] + FEASIBILITY_TOLERANCE:
            scale = min(scale, capacities[resource] / max(usage[resource], EPS))
    if scale < 1.0:
        solution = [value * max(0.0, scale) for value in solution]
        usage = _resource_usage(weights, solution, num_resources)

    remaining = [max(0.0, capacities[resource] - usage[resource]) for resource in range(num_resources)]
    order = sorted(
        range(num_items),
        key=lambda item: (_density(values, weights, capacities, item, dual_prices), values[item], -item),
        reverse=True,
    )
    budget = max(0, int(repair_budget))
    for item in order:
        if budget <= 0:
            break
        if solution[item] >= 1.0 - FEASIBILITY_TOLERANCE:
            continue
        max_add = 1.0 - solution[item]
        for resource in range(num_resources):
            weight = weights[item][resource]
            if weight > FEASIBILITY_TOLERANCE:
                max_add = min(max_add, remaining[resource] / weight)
        add = max(0.0, min(1.0 - solution[item], max_add))
        if add <= FEASIBILITY_TOLERANCE:
            continue
        solution[item] += add
        for resource in range(num_resources):
            remaining[resource] = max(0.0, remaining[resource] - weights[item][resource] * add)
        budget -= 1
    return [min(1.0, max(0.0, float(value))) for value in solution]


def _loss(model: Any, tensor: Any, config: PDLPackingLPConfig, *, rho: float) -> tuple[Any, dict[str, float]]:
    torch = require_torch()
    primal, dual = model(tensor)
    usage = tensor.weights.transpose(0, 1) @ primal
    violation = torch.relu((usage - tensor.capacities) / tensor.capacities.clamp_min(EPS))
    raw_violation = (usage - tensor.capacities) / tensor.capacities.clamp_min(EPS)
    value_scale = tensor.values.sum().clamp_min(1.0)
    primal_value = (tensor.values * primal).sum() / value_scale
    dual_weighted_violation = (dual * raw_violation).sum() / max(float(tensor.num_resources), 1.0)
    augmented = 0.5 * float(rho) * (violation**2).sum()
    dual_activity = tensor.weights @ dual
    dual_feasibility = torch.relu((tensor.values - dual_activity) / value_scale).pow(2).sum()
    dual_objective = (tensor.capacities * dual).sum() / value_scale
    dual_gap = torch.relu(dual_objective - primal_value)
    loss = (
        -primal_value
        + dual_weighted_violation
        + augmented
        + config.dual_feasibility_weight * dual_feasibility
        + config.dual_gap_weight * dual_gap
    )
    metrics = {
        "primal_value": float(primal_value.detach().cpu().item()),
        "violation": float(violation.sum().detach().cpu().item()),
        "dual_feasibility": float(dual_feasibility.detach().cpu().item()),
        "dual_gap": float(dual_gap.detach().cpu().item()),
    }
    return loss, metrics


def fit(
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    config: PDLPackingLPConfig | MLBaselineConfig | dict[str, Any] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> TrainedState:
    if not torch_available():
        raise TorchUnavailableError("PyTorch is required for the PDL Packing LP baseline.")
    torch = require_torch()
    resolved_config = config if isinstance(config, PDLPackingLPConfig) else PDLPackingLPConfig.from_config(config)
    seed_everything(resolved_config.seed)
    device = resolve_device(resolved_config.device)
    if not train_instances:
        raise ValueError("The PDL Packing LP baseline requires public training instances.")

    public_train = [public_view(instance) for instance in train_instances]
    public_val = [public_view(instance) for instance in (val_instances or [])]
    train_tensors = [tensorize_packing_instance(instance, backend="torch", device=str(device)) for instance in public_train]
    val_tensors = [tensorize_packing_instance(instance, backend="torch", device=str(device)) for instance in public_val]
    item_dim = int(train_tensors[0].item_features.shape[-1])
    resource_dim = int(train_tensors[0].resource_features.shape[-1])
    model = PDLPackingLPNet(item_dim, resource_dim, resolved_config.hidden_dim, resolved_config.layers).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": [param for name, param in model.named_parameters() if not name.startswith("dual_head")], "lr": resolved_config.learning_rate},
            {"params": [param for name, param in model.named_parameters() if name.startswith("dual_head")], "lr": resolved_config.dual_learning_rate},
        ]
    )
    logger = MetricsLogger(Path(metrics_path) if metrics_path is not None else None)
    best_state: dict[str, Any] | None = None
    best_loss = math.inf

    for epoch in range(1, resolved_config.epochs + 1):
        rho = resolved_config.rho * (resolved_config.rho_growth ** max(0, epoch - 1))
        model.train()
        train_losses: list[float] = []
        train_metric_rows: list[dict[str, float]] = []
        for tensor in train_tensors:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _loss(model, tensor, resolved_config, rho=rho)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
            train_metric_rows.append(metrics)
        train_loss = sum(train_losses) / max(1, len(train_losses))
        val_loss = None
        if val_tensors:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for tensor in val_tensors:
                    loss, _metrics = _loss(model, tensor, resolved_config, rho=rho)
                    val_losses.append(float(loss.detach().cpu().item()))
            val_loss = sum(val_losses) / max(1, len(val_losses))
        mean_metrics = {
            key: sum(row[key] for row in train_metric_rows) / max(1, len(train_metric_rows))
            for key in ("primal_value", "violation", "dual_feasibility", "dual_gap")
        }
        logger.log(epoch=epoch, train_loss=train_loss, val_loss=val_loss, rho=rho, **mean_metrics)
        selection_loss = train_loss if val_loss is None else val_loss
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "problem": "packing_lp",
        "baseline_name": PDL_PACKINGLP_BASELINE_NAME,
        "config": resolved_config.to_dict(),
        "train_instances": len(public_train),
        "validation_instances": len(public_val),
        "item_dim": item_dim,
        "resource_dim": resource_dim,
        "history": logger.rows,
    }
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            {"model_state_dict": model.state_dict(), "item_dim": item_dim, "resource_dim": resource_dim},
            metadata=metadata,
            use_torch=True,
        )
    return TrainedState(payload={"model": model, "item_dim": item_dim, "resource_dim": resource_dim}, metadata=metadata)


def _predict(model: Any, instance: dict[str, Any], config: PDLPackingLPConfig) -> tuple[list[float], list[float]]:
    torch = require_torch()
    device = resolve_device(config.device)
    tensor = tensorize_packing_instance(public_view(instance), backend="torch", device=str(device))
    model.eval()
    with torch.no_grad():
        primal, dual = model(tensor)
    return (
        [float(value) for value in primal.detach().cpu().tolist()],
        [float(value) for value in dual.detach().cpu().tolist()],
    )


def solve(
    instance: dict[str, Any],
    trained_state: TrainedState,
    config: PDLPackingLPConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> list[float]:
    resolved_config = config if isinstance(config, PDLPackingLPConfig) else PDLPackingLPConfig.from_config(config)
    fractions, dual_prices = _predict(trained_state.payload["model"], instance, resolved_config)
    return project_and_fill_packinglp(
        instance,
        fractions,
        dual_prices=dual_prices,
        repair_budget=resolved_config.repair_budget,
    )


def build_pdl_packinglp_baselines(
    problem_name: str,
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    *,
    artifact_dir: str | Path | None = None,
    config: PDLPackingLPConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    if problem_name != "packing_lp" or not torch_available():
        return {}
    root = Path(artifact_dir) if artifact_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / f"{PDL_PACKINGLP_BASELINE_NAME}.pt" if root is not None else None
    metrics_path = root / f"{PDL_PACKINGLP_BASELINE_NAME}_metrics.jsonl" if root is not None else None
    resolved_config = config if isinstance(config, PDLPackingLPConfig) else PDLPackingLPConfig.from_config(config)
    trained_state = fit(
        train_instances,
        val_instances or [],
        resolved_config,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )

    def solver(instance: dict[str, Any]) -> list[float]:
        return solve(instance, trained_state, resolved_config)

    return {PDL_PACKINGLP_BASELINE_NAME: solver}
