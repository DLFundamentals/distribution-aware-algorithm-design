from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_baselines.checkpoint import save_checkpoint
from ml_baselines.config import MLBaselineConfig
from ml_baselines.metrics import MetricsLogger, mean_metrics
from ml_baselines.seeding import seed_everything
from ml_baselines.torch_utils import move_to_device, require_torch, resolve_device


LossFn = Callable[[Any, Any], Any]


@dataclass
class TrainLoopResult:
    history: list[dict[str, Any]] = field(default_factory=list)
    best_validation_loss: float | None = None
    checkpoint_path: str | None = None


def _loss_and_metrics(raw: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(raw, tuple):
        if len(raw) == 2 and isinstance(raw[1], dict):
            return raw[0], raw[1]
        if len(raw) >= 1:
            return raw[0], {}
    return raw, {}


def _batch_iter(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    size = max(1, int(batch_size))
    for start in range(0, len(items), size):
        yield items[start : start + size]


def evaluate_loop(
    model: Any,
    batches: list[Any],
    loss_fn: LossFn,
    *,
    config: MLBaselineConfig,
) -> dict[str, float]:
    torch = require_torch()
    device = resolve_device(config.device)
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in _batch_iter(batches, config.batch_size):
            moved = move_to_device(batch, str(device))
            loss, metrics = _loss_and_metrics(loss_fn(model, moved))
            row = {"loss": float(loss.detach().cpu().item()), **metrics}
            rows.append(row)
    return mean_metrics(rows)


def train_loop(
    model: Any,
    train_batches: list[Any],
    val_batches: list[Any],
    loss_fn: LossFn,
    *,
    config: MLBaselineConfig | dict[str, Any] | None = None,
    optimizer: Any | None = None,
    checkpoint_path: str | Path | None = None,
    metrics_logger: MetricsLogger | None = None,
) -> TrainLoopResult:
    """Generic PyTorch train/validation loop for small baseline models."""

    torch = require_torch()
    resolved_config = config if isinstance(config, MLBaselineConfig) else MLBaselineConfig.from_mapping(config)
    seed_everything(resolved_config.seed)
    device = resolve_device(resolved_config.device)
    model.to(device)
    resolved_optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=resolved_config.learning_rate)
    logger = metrics_logger or MetricsLogger()
    result = TrainLoopResult()
    best_state = None
    best_validation_loss: float | None = None

    for epoch in range(1, int(resolved_config.epochs) + 1):
        model.train()
        train_rows: list[dict[str, Any]] = []
        for batch in _batch_iter(train_batches, resolved_config.batch_size):
            moved = move_to_device(batch, str(device))
            resolved_optimizer.zero_grad(set_to_none=True)
            loss, metrics = _loss_and_metrics(loss_fn(model, moved))
            loss.backward()
            resolved_optimizer.step()
            train_rows.append({"loss": float(loss.detach().cpu().item()), **metrics})

        train_metrics = {f"train_{key}": value for key, value in mean_metrics(train_rows).items()}
        val_metrics = {
            f"val_{key}": value
            for key, value in evaluate_loop(model, val_batches, loss_fn, config=resolved_config).items()
        } if val_batches else {}
        row = logger.log(epoch=epoch, **train_metrics, **val_metrics)
        result.history.append(row)

        validation_loss = val_metrics.get("val_loss")
        if validation_loss is not None and (best_validation_loss is None or validation_loss < best_validation_loss):
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    result.best_validation_loss = best_validation_loss
    if checkpoint_path is not None:
        saved = save_checkpoint(
            checkpoint_path,
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": resolved_optimizer.state_dict(),
            },
            metadata={"config": resolved_config.to_dict(), "history": result.history},
            use_torch=True,
        )
        result.checkpoint_path = str(saved)
    return result
