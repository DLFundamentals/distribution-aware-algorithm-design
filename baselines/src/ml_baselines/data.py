from __future__ import annotations

from pathlib import Path
from typing import Any

from ml_baselines.tensorize import (
    tensorize_graph_split,
    tensorize_maxsat_split,
    tensorize_packing_split,
    tensorize_tsp_split,
)

GRAPH_PROBLEMS = {"coloring", "mis", "mds"}
PACKING_PROBLEMS = {"mdkp", "packing_lp"}


def load_public_split(dataset_dir: str | Path, split: str) -> list[dict[str, Any]]:
    """Load a DasBench split with evaluator-only fields removed."""

    from dasbench.data import load_split

    return load_split(Path(dataset_dir), split, public=True)


def load_public_train_validation(dataset_dir: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(dataset_dir)
    return load_public_split(root, "train"), load_public_split(root, "validation")


def tensorize_instances(
    problem_name: str,
    instances: list[dict[str, Any]],
    *,
    backend: str = "auto",
    device: str | None = None,
) -> list[Any]:
    if problem_name in GRAPH_PROBLEMS:
        return tensorize_graph_split(instances, backend=backend, device=device)
    if problem_name in PACKING_PROBLEMS:
        return tensorize_packing_split(instances, backend=backend, device=device)
    if problem_name == "maxsat":
        return tensorize_maxsat_split(instances, backend=backend, device=device)
    if problem_name == "tsp":
        return tensorize_tsp_split(instances, backend=backend, device=device)
    raise ValueError(f"Unsupported problem for ML tensorization: {problem_name}")


def tensorize_train_validation(
    problem_name: str,
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]],
    *,
    backend: str = "auto",
    device: str | None = None,
) -> tuple[list[Any], list[Any]]:
    return (
        tensorize_instances(problem_name, train_instances, backend=backend, device=device),
        tensorize_instances(problem_name, val_instances, backend=backend, device=device),
    )


def load_and_tensorize_train_validation(
    dataset_dir: str | Path,
    *,
    backend: str = "auto",
    device: str | None = None,
) -> tuple[list[Any], list[Any]]:
    from dasbench.data import load_manifest

    root = Path(dataset_dir)
    manifest = load_manifest(root)
    train_instances, val_instances = load_public_train_validation(root)
    return tensorize_train_validation(
        str(manifest["problem"]),
        train_instances,
        val_instances,
        backend=backend,
        device=device,
    )
