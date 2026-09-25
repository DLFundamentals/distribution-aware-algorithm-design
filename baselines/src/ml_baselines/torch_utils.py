from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from typing import Any

import numpy as np


class TorchUnavailableError(ImportError):
    """Raised when a PyTorch-only utility is called without PyTorch installed."""


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise TorchUnavailableError(
            "PyTorch is required for this operation. Install torch or use the NumPy-backed tensorization helpers only."
        ) from exc
    return torch


def resolve_device(device: str | None):
    torch = require_torch()
    requested = device or "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def as_backend_tensor(
    values: Any,
    *,
    dtype: str,
    device: str | None = None,
    backend: str = "auto",
):
    """Return a torch tensor when available/requested, otherwise a NumPy array."""

    if backend not in {"auto", "torch", "numpy"}:
        raise ValueError(f"Unknown tensor backend `{backend}`.")
    if backend == "torch" or (backend == "auto" and torch_available()):
        torch = require_torch()
        torch_dtype = {
            "float": torch.float32,
            "long": torch.long,
            "bool": torch.bool,
        }[dtype]
        return torch.tensor(values, dtype=torch_dtype, device=resolve_device(device))
    np_dtype = {
        "float": np.float32,
        "long": np.int64,
        "bool": np.bool_,
    }[dtype]
    return np.asarray(values, dtype=np_dtype)


def move_to_device(value: Any, device: str | None):
    """Recursively move torch tensors inside common containers/dataclasses."""

    torch = require_torch()
    resolved = resolve_device(device)
    if isinstance(value, torch.Tensor):
        return value.to(resolved)
    if is_dataclass(value) and not isinstance(value, type):
        updates = {field.name: move_to_device(getattr(value, field.name), device) for field in fields(value)}
        return replace(value, **updates)
    if isinstance(value, Mapping):
        return type(value)((key, move_to_device(item, device)) for key, item in value.items())
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(move_to_device(item, device) for item in value))
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return type(value)(move_to_device(item, device) for item in value)
    return value
