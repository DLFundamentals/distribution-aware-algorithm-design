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

MDKP_BASELINE_NAME = "ml_mdkp_item_scorer"
PACKINGLP_BASELINE_NAME = "ml_packinglp_item_fraction"
EPS = 1e-9
FEASIBILITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ItemResourceConfig:
    epochs: int = 50
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    batch_size: int = 1
    violation_penalty: float = 10.0
    repair_budget: int = 64
    device: str = "cpu"
    seed: int = 0
    binary_penalty: float = 0.01

    @classmethod
    def from_config(cls, config: MLBaselineConfig | dict[str, Any] | None) -> "ItemResourceConfig":
        base = cls()
        if config is None:
            return base
        payload = config.to_dict() if isinstance(config, MLBaselineConfig) else dict(config)
        return cls(
            epochs=int(payload.get("epochs", base.epochs)),
            learning_rate=float(payload.get("learning_rate", payload.get("lr", base.learning_rate))),
            hidden_dim=int(payload.get("hidden_dim", base.hidden_dim)),
            batch_size=int(payload.get("batch_size", base.batch_size)),
            violation_penalty=float(payload.get("violation_penalty", base.violation_penalty)),
            repair_budget=int(payload.get("repair_budget", base.repair_budget)),
            device=str(payload.get("device", base.device)),
            seed=int(payload.get("seed", base.seed)),
            binary_penalty=float(payload.get("binary_penalty", base.binary_penalty)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "hidden_dim": self.hidden_dim,
            "batch_size": self.batch_size,
            "violation_penalty": self.violation_penalty,
            "repair_budget": self.repair_budget,
            "device": self.device,
            "seed": self.seed,
            "binary_penalty": self.binary_penalty,
        }


class ItemResourceNet:
    """Small PyTorch-only item scorer with learned resource prices."""

    def __new__(cls, *args: Any, **kwargs: Any):
        torch = require_torch()
        nn = torch.nn

        class _ItemResourceNet(nn.Module):
            def __init__(self, item_dim: int, resource_dim: int, hidden_dim: int) -> None:
                super().__init__()
                self.resource_mlp = nn.Sequential(
                    nn.Linear(resource_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                self.item_mlp = nn.Sequential(
                    nn.Linear(item_dim + 1, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, tensor: Any) -> tuple[Any, Any]:
                torch = require_torch()
                resource_prices = torch.nn.functional.softplus(self.resource_mlp(tensor.resource_features)).reshape(-1)
                normalized_weights = tensor.weights / tensor.capacities.clamp_min(EPS).reshape(1, -1)
                pressure = normalized_weights @ resource_prices.reshape(-1, 1)
                logits = self.item_mlp(torch.cat([tensor.item_features, pressure], dim=-1)).reshape(-1)
                return logits, resource_prices

        return _ItemResourceNet(*args, **kwargs)


def _value_scale(tensor: Any) -> Any:
    return tensor.values.sum().clamp_min(1.0)


def _usage(tensor: Any, vector: Any) -> Any:
    return tensor.weights.transpose(0, 1) @ vector.reshape(-1)


def _violation_loss(tensor: Any, vector: Any) -> Any:
    torch = require_torch()
    violation = torch.relu((_usage(tensor, vector) - tensor.capacities) / tensor.capacities.clamp_min(EPS))
    return (violation**2).sum()


def _loss(problem_name: str, model: Any, tensor: Any, config: ItemResourceConfig) -> Any:
    torch = require_torch()
    logits, _prices = model(tensor)
    prediction = torch.sigmoid(logits)
    value_term = (tensor.values * prediction).sum() / _value_scale(tensor)
    violation = _violation_loss(tensor, prediction)
    if problem_name == "mdkp":
        binary = (prediction * (1.0 - prediction)).mean()
        return -value_term + config.violation_penalty * violation + config.binary_penalty * binary
    if problem_name == "packing_lp":
        return -value_term + config.violation_penalty * violation
    raise ValueError(f"Unsupported item/resource ML problem: {problem_name}")


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


def _selected_usage(weights: list[list[float]], selected: set[int], num_resources: int) -> list[float]:
    usage = [0.0] * num_resources
    for item in selected:
        for resource in range(num_resources):
            usage[resource] += weights[item][resource]
    return usage


def _feasible_add(usage: list[float], weights: list[list[float]], capacities: list[float], item: int) -> bool:
    return all(
        usage[resource] + weights[item][resource] <= capacities[resource] + FEASIBILITY_TOLERANCE
        for resource in range(len(capacities))
    )


def _density(values: list[float], weights: list[list[float]], capacities: list[float], item: int, prices: list[float] | None) -> float:
    resource_prices = prices if prices is not None else [1.0] * len(capacities)
    pressure = sum(
        resource_prices[resource] * weights[item][resource] / max(capacities[resource], EPS)
        for resource in range(len(capacities))
    )
    return values[item] / max(pressure, FEASIBILITY_TOLERANCE)


def decode_mdkp_scores(
    instance: dict[str, Any],
    probabilities: list[float],
    *,
    resource_prices: list[float] | None = None,
    repair_budget: int = 64,
) -> list[int]:
    public = public_view(instance)
    num_items = int(public["num_items"])
    num_resources = int(public["num_resources"])
    values, weights, capacities = _values_weights_capacities(public)
    clipped = [min(1.0, max(0.0, float(value))) for value in probabilities[:num_items]]
    if len(clipped) != num_items:
        raise ValueError(f"Expected {num_items} item probabilities, got {len(probabilities)}.")

    order = sorted(
        range(num_items),
        key=lambda item: (
            clipped[item] * _density(values, weights, capacities, item, resource_prices),
            clipped[item],
            values[item],
            -item,
        ),
        reverse=True,
    )
    selected: set[int] = set()
    usage = [0.0] * num_resources
    for item in order:
        if _feasible_add(usage, weights, capacities, item):
            selected.add(item)
            for resource in range(num_resources):
                usage[resource] += weights[item][resource]

    remaining_budget = max(0, int(repair_budget))
    improved = True
    while improved and remaining_budget > 0:
        improved = False
        usage = _selected_usage(weights, selected, num_resources)
        for item in order:
            if item in selected:
                continue
            remaining_budget -= 1
            if _feasible_add(usage, weights, capacities, item):
                selected.add(item)
                improved = True
                break
            if remaining_budget <= 0:
                break
        if improved:
            continue
        for remove_item in sorted(selected, key=lambda item: (values[item], item)):
            usage_without = [
                usage[resource] - weights[remove_item][resource]
                for resource in range(num_resources)
            ]
            for add_item in order:
                if add_item in selected or values[add_item] <= values[remove_item] + EPS:
                    continue
                remaining_budget -= 1
                if _feasible_add(usage_without, weights, capacities, add_item):
                    selected.remove(remove_item)
                    selected.add(add_item)
                    improved = True
                    break
                if remaining_budget <= 0:
                    break
            if improved or remaining_budget <= 0:
                break
    return sorted(selected)


def decode_packinglp_fractions(
    instance: dict[str, Any],
    fractions: list[float],
    *,
    resource_prices: list[float] | None = None,
    repair_budget: int = 64,
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
        key=lambda item: (
            _density(values, weights, capacities, item, resource_prices),
            values[item],
            -item,
        ),
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


def fit(
    problem_name: str,
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    config: ItemResourceConfig | MLBaselineConfig | dict[str, Any] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> TrainedState:
    if problem_name not in {"mdkp", "packing_lp"}:
        raise ValueError(f"Unsupported item/resource ML problem: {problem_name}")
    if not torch_available():
        raise TorchUnavailableError("PyTorch is required for item/resource neural baselines.")
    torch = require_torch()
    resolved_config = config if isinstance(config, ItemResourceConfig) else ItemResourceConfig.from_config(config)
    seed_everything(resolved_config.seed)
    device = resolve_device(resolved_config.device)
    if not train_instances:
        raise ValueError("Item/resource ML baselines require public training instances.")

    public_train = [public_view(instance) for instance in train_instances]
    public_val = [public_view(instance) for instance in (val_instances or [])]
    train_tensors = [tensorize_packing_instance(instance, backend="torch", device=str(device)) for instance in public_train]
    val_tensors = [tensorize_packing_instance(instance, backend="torch", device=str(device)) for instance in public_val]
    item_dim = int(train_tensors[0].item_features.shape[-1])
    resource_dim = int(train_tensors[0].resource_features.shape[-1])
    model = ItemResourceNet(item_dim, resource_dim, resolved_config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=resolved_config.learning_rate)
    logger = MetricsLogger(Path(metrics_path) if metrics_path is not None else None)
    best_state: dict[str, Any] | None = None
    best_train_loss = math.inf

    for epoch in range(1, resolved_config.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for tensor in train_tensors:
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(problem_name, model, tensor, resolved_config)
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
                    loss = _loss(problem_name, model, tensor, resolved_config)
                    val_losses.append(float(loss.detach().cpu().item()))
            val_loss = sum(val_losses) / max(1, len(val_losses))
        logger.log(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    baseline_name = MDKP_BASELINE_NAME if problem_name == "mdkp" else PACKINGLP_BASELINE_NAME
    metadata = {
        "problem": problem_name,
        "baseline_name": baseline_name,
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
            {
                "model_state_dict": model.state_dict(),
                "item_dim": item_dim,
                "resource_dim": resource_dim,
            },
            metadata=metadata,
            use_torch=True,
        )
    return TrainedState(
        payload={
            "model": model,
            "item_dim": item_dim,
            "resource_dim": resource_dim,
        },
        metadata=metadata,
    )


def _predict(model: Any, instance: dict[str, Any], config: ItemResourceConfig) -> tuple[list[float], list[float]]:
    torch = require_torch()
    device = resolve_device(config.device)
    tensor = tensorize_packing_instance(public_view(instance), backend="torch", device=str(device))
    model.eval()
    with torch.no_grad():
        logits, prices = model(tensor)
        probabilities = torch.sigmoid(logits)
    return (
        [float(value) for value in probabilities.detach().cpu().tolist()],
        [float(value) for value in prices.detach().cpu().tolist()],
    )


def solve(
    problem_name: str,
    instance: dict[str, Any],
    trained_state: TrainedState,
    config: ItemResourceConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> list[int] | list[float]:
    resolved_config = config if isinstance(config, ItemResourceConfig) else ItemResourceConfig.from_config(config)
    probabilities, prices = _predict(trained_state.payload["model"], instance, resolved_config)
    if problem_name == "mdkp":
        return decode_mdkp_scores(
            instance,
            probabilities,
            resource_prices=prices,
            repair_budget=resolved_config.repair_budget,
        )
    if problem_name == "packing_lp":
        return decode_packinglp_fractions(
            instance,
            probabilities,
            resource_prices=prices,
            repair_budget=resolved_config.repair_budget,
        )
    raise ValueError(f"Unsupported item/resource ML problem: {problem_name}")


def build_item_resource_baselines(
    problem_name: str,
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    *,
    artifact_dir: str | Path | None = None,
    config: ItemResourceConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    if problem_name not in {"mdkp", "packing_lp"} or not torch_available():
        return {}
    baseline_name = MDKP_BASELINE_NAME if problem_name == "mdkp" else PACKINGLP_BASELINE_NAME
    root = Path(artifact_dir) if artifact_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / f"{baseline_name}.pt" if root is not None else None
    metrics_path = root / f"{baseline_name}_metrics.jsonl" if root is not None else None
    resolved_config = config if isinstance(config, ItemResourceConfig) else ItemResourceConfig.from_config(config)
    trained_state = fit(
        problem_name,
        train_instances,
        val_instances or [],
        resolved_config,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )

    def solver(instance: dict[str, Any]) -> list[int] | list[float]:
        return solve(problem_name, instance, trained_state, resolved_config)

    return {baseline_name: solver}
