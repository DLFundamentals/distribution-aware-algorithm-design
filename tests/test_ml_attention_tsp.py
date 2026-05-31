from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

BASELINES_SRC = Path(__file__).resolve().parents[1] / "baselines" / "src"
if str(BASELINES_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINES_SRC))

from dasbench.integrations.ml_graph_baseline import build_ml_tsp_baselines
from dasbench.problems import get_problem_definition
from ml_baselines.attention_tsp import (
    ATTENTION_TSP_BASELINE_NAME,
    AttentionTSPConfig,
    AttentionTSPModel,
    build_attention_tsp_baselines,
    decode_attention_tsp,
    normalize_points,
    sample_tour,
)
from ml_baselines.checkpoint import load_checkpoint
from ml_baselines.torch_utils import torch_available
from ml_baselines.tsp_neural_constructor import distance_matrix_from_points, tour_length_from_matrix


def _square_instance() -> dict[str, object]:
    return {
        "id": "square-attention-tsp",
        "num_cities": 4,
        "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "optimum_objective": 4.0,
        "optimum_tour": [0, 1, 2, 3],
        "_hidden_hint": [0, 3, 2, 1],
    }


def _assert_valid_tsp(instance: dict[str, object], raw_solution: object) -> list[int]:
    problem = get_problem_definition("tsp")
    solution = problem.canonicalize_solution(raw_solution, instance)
    valid, error = problem.validate_solution(solution, instance)
    if not valid:
        raise AssertionError(error)
    return solution


class AttentionTSPUtilityTests(unittest.TestCase):
    def test_normalize_points_maps_bounding_box_to_unit_scale(self) -> None:
        normalized = normalize_points([(2.0, 4.0), (4.0, 8.0), (3.0, 6.0)])

        self.assertEqual(normalized[0], [0.0, 0.0])
        self.assertEqual(normalized[1], [0.5, 1.0])
        self.assertEqual(normalized[2], [0.25, 0.5])


@unittest.skipUnless(torch_available(), "PyTorch is not installed in this environment.")
class AttentionTSPModelTests(unittest.TestCase):
    def _tiny_config(self, **overrides: object) -> AttentionTSPConfig:
        payload = {
            "epochs": 1,
            "steps_per_epoch": 1,
            "batch_size": 1,
            "embedding_dim": 16,
            "hidden_dim": 16,
            "n_heads": 2,
            "n_encoder_layers": 1,
            "inference_samples": 0,
            "two_opt_budget": 8,
            "device": "cpu",
            "seed": 41,
        }
        payload.update(overrides)
        return AttentionTSPConfig.from_config(payload)

    def test_greedy_sample_returns_deterministic_permutation(self) -> None:
        from ml_baselines.seeding import seed_everything

        instance = _square_instance()
        config = self._tiny_config()
        seed_everything(config.seed)
        model = AttentionTSPModel(
            config.embedding_dim,
            config.hidden_dim,
            config.n_heads,
            config.n_encoder_layers,
        )

        first = sample_tour(model, instance, config, decode_type="greedy")
        second = sample_tour(model, instance, config, decode_type="greedy")

        self.assertEqual(first, second)
        self.assertEqual(sorted(_assert_valid_tsp(instance, first)), list(range(4)))

    def test_decode_returns_valid_tour_and_finite_length(self) -> None:
        from ml_baselines.seeding import seed_everything

        instance = _square_instance()
        config = self._tiny_config(two_opt_budget=32)
        seed_everything(config.seed)
        model = AttentionTSPModel(
            config.embedding_dim,
            config.hidden_dim,
            config.n_heads,
            config.n_encoder_layers,
        )

        tour = decode_attention_tsp(instance, model, config)

        solution = _assert_valid_tsp(instance, tour)
        matrix = distance_matrix_from_points([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        self.assertLessEqual(tour_length_from_matrix(matrix, solution), 4.8284272)

    def test_builder_trains_saves_checkpoint_and_solves_tiny_instance(self) -> None:
        instance = {
            "id": "tiny-attention-tsp-train",
            "num_cities": 5,
            "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]],
            "optimum_objective": 4.414213562373095,
        }
        config = self._tiny_config(seed=47)

        with tempfile.TemporaryDirectory(prefix="attention-tsp-") as root:
            baselines = build_attention_tsp_baselines(
                "tsp",
                train_instances=[instance],
                val_instances=[instance],
                artifact_dir=Path(root),
                config=config,
            )

            self.assertEqual(set(baselines), {ATTENTION_TSP_BASELINE_NAME})
            checkpoint_path = Path(root) / f"{ATTENTION_TSP_BASELINE_NAME}.pt"
            self.assertTrue(checkpoint_path.exists())
            checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", use_torch=True)
            self.assertEqual(checkpoint["metadata"]["baseline_name"], ATTENTION_TSP_BASELINE_NAME)
            self.assertEqual(checkpoint["metadata"]["training_mode"], "reinforce_with_greedy_rollout_baseline")
            solution = baselines[ATTENTION_TSP_BASELINE_NAME](instance)

        _assert_valid_tsp(instance, solution)

    def test_generic_tsp_integration_includes_attention_baseline(self) -> None:
        instance = {
            "id": "tiny-attention-tsp-integration",
            "num_cities": 4,
            "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "optimum_objective": 4.0,
        }
        config = {
            "epochs": 1,
            "steps_per_epoch": 1,
            "batch_size": 1,
            "hidden_dim": 16,
            "n_heads": 2,
            "layers": 1,
            "candidates": 1,
            "inference_samples": 0,
            "two_opt_budget": 4,
            "pseudo_two_opt_budget": 4,
            "seed": 53,
        }

        with tempfile.TemporaryDirectory(prefix="attention-tsp-integration-") as root:
            baselines = build_ml_tsp_baselines(
                "tsp",
                train_instances=[instance],
                validation_instances=[instance],
                artifact_dir=Path(root),
                config=config,
            )

        self.assertIn(ATTENTION_TSP_BASELINE_NAME, baselines)


if __name__ == "__main__":
    unittest.main()
