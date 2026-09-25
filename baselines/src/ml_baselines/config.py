from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MLBaselineConfig:
    """Common training/inference knobs for lightweight neural baselines."""

    epochs: int = 20
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    message_passing_layers: int = 3
    batch_size: int = 8
    inference_samples: int = 1
    inference_restarts: int = 1
    repair_budget: int = 128
    device: str = "cpu"
    seed: int = 0
    timeout_seconds: float | None = None
    log_every: int = 1

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "MLBaselineConfig":
        if payload is None:
            return cls()
        allowed = {field for field in cls.__dataclass_fields__}
        values = {key: value for key, value in payload.items() if key in allowed}
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_updates(self, **updates: Any) -> "MLBaselineConfig":
        values = self.to_dict()
        values.update(updates)
        return type(self)(**values)
