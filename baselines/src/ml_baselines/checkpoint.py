from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from ml_baselines.torch_utils import require_torch, torch_available


def _contains_torch_tensor(value: Any) -> bool:
    if not torch_available():
        return False
    import torch

    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_torch_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_torch_tensor(item) for item in value)
    return False


def save_checkpoint(
    path: str | Path,
    state: Any,
    *,
    metadata: dict[str, Any] | None = None,
    use_torch: bool | None = None,
) -> Path:
    """Persist a trainable baseline state and lightweight metadata."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "metadata": dict(metadata or {}),
    }
    should_use_torch = _contains_torch_tensor(payload) if use_torch is None else bool(use_torch)
    if should_use_torch:
        torch = require_torch()
        torch.save(payload, destination)
    else:
        with destination.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | None = None,
    use_torch: bool | None = None,
) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`."""

    source = Path(path)
    should_use_torch = bool(use_torch)
    if use_torch is None and torch_available():
        try:
            torch = require_torch()
            return torch.load(source, map_location=map_location, weights_only=False)
        except Exception:
            should_use_torch = False
    if should_use_torch:
        torch = require_torch()
        return torch.load(source, map_location=map_location, weights_only=False)
    with source.open("rb") as handle:
        return pickle.load(handle)
