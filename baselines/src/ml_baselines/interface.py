from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ml_baselines.checkpoint import load_checkpoint, save_checkpoint
from ml_baselines.config import MLBaselineConfig
from ml_baselines.inference import solve_with_timeout


@dataclass
class TrainedState:
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainableBaselineProtocol(Protocol):
    def fit(
        self,
        train_instances: list[dict[str, Any]],
        val_instances: list[dict[str, Any]],
        config: MLBaselineConfig,
    ) -> TrainedState:
        ...

    def solve(self, instance: dict[str, Any], trained_state: TrainedState, config: MLBaselineConfig) -> Any:
        ...


FitFn = Callable[[list[dict[str, Any]], list[dict[str, Any]], MLBaselineConfig], TrainedState | Any]
SolveFn = Callable[[dict[str, Any], TrainedState, MLBaselineConfig], Any]


class BaselineAdapter:
    """Adapter from fit/solve functions to the current DasBench solver callable."""

    def __init__(
        self,
        *,
        fit: FitFn,
        solve: SolveFn,
        config: MLBaselineConfig | dict[str, Any] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self._fit = fit
        self._solve = solve
        self.config = config if isinstance(config, MLBaselineConfig) else MLBaselineConfig.from_mapping(config)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.trained_state: TrainedState | None = None

    def fit(
        self,
        train_instances: list[dict[str, Any]],
        val_instances: list[dict[str, Any]],
    ) -> TrainedState:
        result = self._fit(train_instances, val_instances, self.config)
        self.trained_state = result if isinstance(result, TrainedState) else TrainedState(payload=result)
        if self.checkpoint_path is not None:
            self.save_state(self.checkpoint_path)
        return self.trained_state

    def solve(self, instance: dict[str, Any]) -> Any:
        if self.trained_state is None:
            if self.checkpoint_path is None:
                raise RuntimeError("BaselineAdapter.solve called before fit/load_state.")
            self.load_state(self.checkpoint_path)
        assert self.trained_state is not None
        return solve_with_timeout(self._solve, instance, self.trained_state, self.config)

    def as_solver(self) -> Callable[[dict[str, Any]], Any]:
        return self.solve

    def save_state(self, path: str | Path) -> Path:
        if self.trained_state is None:
            raise RuntimeError("Cannot save before fitting or loading a trained state.")
        return save_checkpoint(
            path,
            self.trained_state.payload,
            metadata={
                **self.trained_state.metadata,
                "config": self.config.to_dict(),
            },
        )

    def load_state(self, path: str | Path) -> TrainedState:
        payload = load_checkpoint(path)
        self.trained_state = TrainedState(
            payload=payload["state"],
            metadata=dict(payload.get("metadata") or {}),
        )
        return self.trained_state
