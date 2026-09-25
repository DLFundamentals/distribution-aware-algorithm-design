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
from ml_baselines.torch_utils import torch_available
from ml_baselines.tsp_neural_constructor import (
    TSP_BASELINE_NAME,
    TspNeuralConstructorConfig,
    decode_tsp_heatmap,
    distance_matrix_from_points,
    tour_length_from_matrix,
)


def _assert_valid_tsp(instance: dict[str, object], raw_solution: object) -> list[int]:
    problem = get_problem_definition("tsp")
    solution = problem.canonicalize_solution(raw_solution, instance)
    valid, error = problem.validate_solution(solution, instance)
    if not valid:
        raise AssertionError(error)
    return solution


class TspMLDecoderTests(unittest.TestCase):
    def test_square_decode_returns_valid_tour_and_length(self) -> None:
        instance = {
            "id": "square",
            "num_cities": 4,
            "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        }
        heatmap = [
            [-1.0, 2.0, 0.0, 2.0],
            [2.0, -1.0, 2.0, 0.0],
            [0.0, 2.0, -1.0, 2.0],
            [2.0, 0.0, 2.0, -1.0],
        ]

        tour = decode_tsp_heatmap(instance, heatmap, candidates=2, two_opt_budget=16)

        solution = _assert_valid_tsp(instance, tour)
        matrix = distance_matrix_from_points([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        self.assertAlmostEqual(tour_length_from_matrix(matrix, solution), 4.0, places=6)

    def test_small_random_euclidean_decode_returns_permutation(self) -> None:
        instance = {
            "id": "small-random",
            "num_cities": 5,
            "points": [[0.1, 0.2], [0.8, 0.1], [0.9, 0.7], [0.4, 0.9], [0.2, 0.5]],
        }
        heatmap = [
            [-1.0, 0.2, 0.1, 0.4, 0.9],
            [0.2, -1.0, 0.8, 0.1, 0.3],
            [0.1, 0.8, -1.0, 0.7, 0.2],
            [0.4, 0.1, 0.7, -1.0, 0.6],
            [0.9, 0.3, 0.2, 0.6, -1.0],
        ]

        tour = decode_tsp_heatmap(instance, heatmap, candidates=3, two_opt_budget=20)

        self.assertEqual(sorted(_assert_valid_tsp(instance, tour)), list(range(5)))


@unittest.skipUnless(torch_available(), "PyTorch is not installed in this environment.")
class TspMLEndToEndSmokeTests(unittest.TestCase):
    def test_builder_trains_and_solves_tiny_tsp(self) -> None:
        train_instance = {
            "id": "tiny-tsp-train",
            "num_cities": 5,
            "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]],
        }
        config = TspNeuralConstructorConfig(epochs=2, hidden_dim=16, candidates=2, two_opt_budget=20, seed=29)

        with tempfile.TemporaryDirectory(prefix="tsp-ml-baseline-") as root:
            baselines = build_ml_tsp_baselines(
                "tsp",
                train_instances=[train_instance],
                validation_instances=[train_instance],
                artifact_dir=Path(root),
                config=config.to_dict(),
            )

            self.assertIn(TSP_BASELINE_NAME, baselines)
            solution = baselines[TSP_BASELINE_NAME](train_instance)

        _assert_valid_tsp(train_instance, solution)


if __name__ == "__main__":
    unittest.main()
