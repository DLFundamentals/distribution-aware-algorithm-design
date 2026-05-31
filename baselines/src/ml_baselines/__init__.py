"""Shared utilities for trainable DasBench baseline implementations.

The package is intentionally independent of the benchmark evaluator.  A
trainable baseline can use these helpers to fit on public train/validation
instances, then expose a plain solver callable compatible with
``dasbench.eval.evaluate_solver``.
"""

from ml_baselines.checkpoint import load_checkpoint, save_checkpoint
from ml_baselines.config import MLBaselineConfig
from ml_baselines.data import (
    load_and_tensorize_train_validation,
    load_public_split,
    load_public_train_validation,
    tensorize_instances,
    tensorize_train_validation,
)
from ml_baselines.graph_score_repair import (
    GraphScoreRepairConfig,
    build_graph_score_repair_baselines,
    decode_mds_scores,
    decode_mis_scores,
)
from ml_baselines.interface import BaselineAdapter, TrainedState
from ml_baselines.item_resource_baselines import (
    MDKP_BASELINE_NAME,
    PACKINGLP_BASELINE_NAME,
    ItemResourceConfig,
    build_item_resource_baselines,
    decode_mdkp_scores,
    decode_packinglp_fractions,
)
from ml_baselines.maxsat_assignment import (
    MAXSAT_BASELINE_NAME,
    MaxSatAssignmentConfig,
    bounded_maxsat_local_search,
    build_maxsat_assignment_baselines,
    decode_assignment,
)
from ml_baselines.pignn_coloring import (
    PIGNN_COLORING_BASELINE_NAME,
    PiGNNColoringConfig,
    build_pignn_coloring_baselines,
    decode_fixed_k_coloring,
    decode_pignn_coloring,
)
from ml_baselines.runcsp_maxsat import (
    RUN_CSP_MAXSAT_BASELINE_NAME,
    RunCSPMaxSatConfig,
    bounded_walksat_local_search,
    build_runcsp_maxsat_baselines,
    decode_runcsp_assignment,
    tensorize_runcsp_maxsat_instance,
    weighted_satisfied_score,
)
from ml_baselines.seeding import seed_everything
from ml_baselines.tensorize import (
    GraphTensor,
    MaxSatTensor,
    PackingTensor,
    TspTensor,
    tensorize_graph_instance,
    tensorize_graph_split,
    tensorize_maxsat_instance,
    tensorize_maxsat_split,
    tensorize_packing_instance,
    tensorize_packing_split,
    tensorize_tsp_instance,
    tensorize_tsp_split,
)
from ml_baselines.tsp_neural_constructor import (
    TSP_BASELINE_NAME,
    TspNeuralConstructorConfig,
    bounded_two_opt_from_matrix,
    build_tsp_neural_constructor_baselines,
    decode_tsp_heatmap,
    tour_length_from_matrix,
)

__all__ = [
    "BaselineAdapter",
    "GraphScoreRepairConfig",
    "GraphTensor",
    "ItemResourceConfig",
    "MLBaselineConfig",
    "MAXSAT_BASELINE_NAME",
    "MDKP_BASELINE_NAME",
    "MaxSatTensor",
    "MaxSatAssignmentConfig",
    "PACKINGLP_BASELINE_NAME",
    "PIGNN_COLORING_BASELINE_NAME",
    "PackingTensor",
    "PiGNNColoringConfig",
    "RUN_CSP_MAXSAT_BASELINE_NAME",
    "RunCSPMaxSatConfig",
    "TrainedState",
    "TSP_BASELINE_NAME",
    "TspTensor",
    "TspNeuralConstructorConfig",
    "bounded_maxsat_local_search",
    "bounded_two_opt_from_matrix",
    "bounded_walksat_local_search",
    "build_graph_score_repair_baselines",
    "build_item_resource_baselines",
    "build_maxsat_assignment_baselines",
    "build_pignn_coloring_baselines",
    "build_runcsp_maxsat_baselines",
    "build_tsp_neural_constructor_baselines",
    "decode_assignment",
    "decode_mdkp_scores",
    "decode_mds_scores",
    "decode_mis_scores",
    "decode_packinglp_fractions",
    "decode_fixed_k_coloring",
    "decode_pignn_coloring",
    "decode_runcsp_assignment",
    "decode_tsp_heatmap",
    "load_checkpoint",
    "load_and_tensorize_train_validation",
    "load_public_split",
    "load_public_train_validation",
    "save_checkpoint",
    "seed_everything",
    "tensorize_graph_instance",
    "tensorize_graph_split",
    "tensorize_instances",
    "tensorize_maxsat_instance",
    "tensorize_maxsat_split",
    "tensorize_packing_instance",
    "tensorize_packing_split",
    "tensorize_runcsp_maxsat_instance",
    "tensorize_train_validation",
    "tensorize_tsp_instance",
    "tensorize_tsp_split",
    "tour_length_from_matrix",
    "weighted_satisfied_score",
]
