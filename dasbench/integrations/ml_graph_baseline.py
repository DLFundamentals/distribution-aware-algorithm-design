from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _ensure_baselines_src_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    source_dir = root / "baselines" / "src"
    if source_dir.exists() and str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))


def build_ml_graph_baselines(
    problem_name: str,
    *,
    train_instances: list[dict[str, Any]] | None,
    validation_instances: list[dict[str, Any]] | None,
    artifact_dir: Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Train public-data-only graph ML baselines when PyTorch is available."""

    if problem_name not in {"mis", "mds", "coloring"} or not train_instances:
        return {}
    _ensure_baselines_src_on_path()
    baselines: dict[str, object] = {}
    try:
        from ml_baselines.graph_score_repair import build_graph_score_repair_baselines
    except ImportError:
        build_graph_score_repair_baselines = None
    if build_graph_score_repair_baselines is not None:
        baselines.update(
            build_graph_score_repair_baselines(
                problem_name,
                train_instances,
                validation_instances or [],
                artifact_dir=artifact_dir,
                config=config,
            )
        )
    if problem_name == "coloring":
        try:
            from ml_baselines.pignn_coloring import build_pignn_coloring_baselines
        except ImportError:
            build_pignn_coloring_baselines = None
        if build_pignn_coloring_baselines is not None:
            baselines.update(
                build_pignn_coloring_baselines(
                    problem_name,
                    train_instances,
                    validation_instances or [],
                    artifact_dir=artifact_dir,
                    config=config,
                )
            )
    return baselines


def build_ml_maxsat_baselines(
    problem_name: str,
    *,
    train_instances: list[dict[str, Any]] | None,
    validation_instances: list[dict[str, Any]] | None,
    artifact_dir: Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Train public-data-only MaxSAT ML baselines when PyTorch is available."""

    if problem_name != "maxsat" or not train_instances:
        return {}
    _ensure_baselines_src_on_path()
    baselines: dict[str, object] = {}
    try:
        from ml_baselines.maxsat_assignment import build_maxsat_assignment_baselines
    except ImportError:
        build_maxsat_assignment_baselines = None
    if build_maxsat_assignment_baselines is not None:
        baselines.update(
            build_maxsat_assignment_baselines(
                problem_name,
                train_instances,
                validation_instances or [],
                artifact_dir=artifact_dir,
                config=config,
            )
        )
    try:
        from ml_baselines.runcsp_maxsat import build_runcsp_maxsat_baselines
    except ImportError:
        build_runcsp_maxsat_baselines = None
    if build_runcsp_maxsat_baselines is not None:
        baselines.update(
            build_runcsp_maxsat_baselines(
                problem_name,
                train_instances,
                validation_instances or [],
                artifact_dir=artifact_dir,
                config=config,
            )
        )
    return baselines


def build_ml_item_resource_baselines(
    problem_name: str,
    *,
    train_instances: list[dict[str, Any]] | None,
    validation_instances: list[dict[str, Any]] | None,
    artifact_dir: Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Train public-data-only MDKP/Packing LP item-resource baselines when PyTorch is available."""

    if problem_name not in {"mdkp", "packing_lp"} or not train_instances:
        return {}
    _ensure_baselines_src_on_path()
    baselines: dict[str, object] = {}
    try:
        from ml_baselines.item_resource_baselines import build_item_resource_baselines
    except ImportError:
        build_item_resource_baselines = None
    if build_item_resource_baselines is not None:
        baselines.update(
            build_item_resource_baselines(
                problem_name,
                train_instances,
                validation_instances or [],
                artifact_dir=artifact_dir,
                config=config,
            )
        )
    if problem_name == "mdkp":
        try:
            from ml_baselines.drl_mdkp import build_drl_mdkp_baselines
        except ImportError:
            build_drl_mdkp_baselines = None
        if build_drl_mdkp_baselines is not None:
            baselines.update(
                build_drl_mdkp_baselines(
                    problem_name,
                    train_instances,
                    validation_instances or [],
                    artifact_dir=artifact_dir,
                    config=config,
                )
            )
    return baselines


def build_ml_tsp_baselines(
    problem_name: str,
    *,
    train_instances: list[dict[str, Any]] | None,
    validation_instances: list[dict[str, Any]] | None,
    artifact_dir: Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Train public-data-only TSP neural construction baselines when PyTorch is available."""

    if problem_name != "tsp" or not train_instances:
        return {}
    _ensure_baselines_src_on_path()
    try:
        from ml_baselines.tsp_neural_constructor import build_tsp_neural_constructor_baselines
    except ImportError:
        return {}
    return build_tsp_neural_constructor_baselines(
        problem_name,
        train_instances,
        validation_instances or [],
        artifact_dir=artifact_dir,
        config=config,
    )
