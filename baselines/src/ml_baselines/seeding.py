from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedState:
    seed: int
    torch_available: bool


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> SeedState:
    """Seed stdlib, NumPy, and PyTorch when PyTorch is installed."""

    resolved = int(seed)
    os.environ["PYTHONHASHSEED"] = str(resolved)
    random.seed(resolved)
    np.random.seed(resolved)

    torch_available = False
    try:
        import torch
    except ImportError:
        return SeedState(seed=resolved, torch_available=torch_available)

    torch_available = True
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    return SeedState(seed=resolved, torch_available=torch_available)
