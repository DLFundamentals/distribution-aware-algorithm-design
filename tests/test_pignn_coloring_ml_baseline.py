from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

BASELINES_SRC = Path(__file__).resolve().parents[1] / "baselines" / "src"
if str(BASELINES_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINES_SRC))

from dasbench.integrations.ml_graph_baseline import build_ml_graph_baselines
from dasbench.problems import get_problem_definition
from ml_baselines.pignn_coloring import (
    PIGNN_COLORING_BASELINE_NAME,
    PiGNNColoringConfig,
    build_pignn_coloring_baselines,
    decode_fixed_k_coloring,
    decode_pignn_coloring,
)
from ml_baselines.torch_utils import torch_available


def _assert_valid_coloring(instance: dict[str, object], raw_solution: object) -> list[int]:
    problem = get_problem_definition("coloring")
    solution = problem.canonicalize_solution(raw_solution, instance)
    valid, error = problem.validate_solution(solution, instance)
    if not valid:
        raise AssertionError(error)
    return solution


class PiGNNColoringDecoderTests(unittest.TestCase):
    def test_fixed_k_decoder_colors_triangle_and_clique(self) -> None:
        instances = [
            {
                "id": "triangle",
                "num_vertices": 3,
                "edges": [[0, 1], [1, 2], [2, 0]],
                "color_count": 3,
            },
            {
                "id": "k4",
                "num_vertices": 4,
                "edges": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
                "color_count": 4,
            },
        ]
        for instance in instances:
            with self.subTest(instance=instance["id"]):
                n = int(instance["num_vertices"])
                k = int(instance["color_count"])
                logits = [[0.0 for _ in range(k)] for _ in range(n)]
                solution = decode_fixed_k_coloring(instance, logits, color_count=k, repair_budget=16)
                self.assertIsNotNone(solution)
                assert solution is not None
                _assert_valid_coloring(instance, solution)

    def test_decreasing_k_decoder_falls_back_to_valid_coloring(self) -> None:
        instance = {
            "id": "odd-cycle",
            "num_vertices": 5,
            "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
        }
        logits = [[1.0, 0.0, -1.0] for _ in range(5)]
        config = PiGNNColoringConfig(min_decode_colors=2, inference_restarts=2, repair_budget=32)
        solution = decode_pignn_coloring(instance, logits, max_colors=3, config=config)
        canonical = _assert_valid_coloring(instance, solution)
        self.assertLessEqual(len(set(canonical)), 3)


@unittest.skipUnless(torch_available(), "PyTorch is not installed in this environment.")
class PiGNNColoringEndToEndTests(unittest.TestCase):
    def test_builder_fits_saves_checkpoint_and_solves_tiny_graph(self) -> None:
        instance = {
            "id": "tiny-coloring",
            "num_vertices": 5,
            "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
        }
        config = PiGNNColoringConfig(
            epochs=2,
            hidden_dim=8,
            layers=1,
            max_colors=3,
            inference_restarts=2,
            repair_budget=32,
            seed=7,
        )
        with tempfile.TemporaryDirectory(prefix="pignn-coloring-") as root:
            root_path = Path(root)
            baselines = build_pignn_coloring_baselines(
                "coloring",
                [instance],
                [instance],
                artifact_dir=root_path,
                config=config,
            )
            self.assertEqual(set(baselines), {PIGNN_COLORING_BASELINE_NAME})
            self.assertTrue((root_path / f"{PIGNN_COLORING_BASELINE_NAME}.pt").exists())
            solution = baselines[PIGNN_COLORING_BASELINE_NAME](instance)
            _assert_valid_coloring(instance, solution)

    def test_generic_graph_integration_includes_pignn_coloring(self) -> None:
        instance = {
            "id": "path-coloring",
            "num_vertices": 4,
            "edges": [[0, 1], [1, 2], [2, 3]],
        }
        config = {"epochs": 1, "hidden_dim": 8, "layers": 1, "max_colors": 2, "repair_budget": 16}
        with tempfile.TemporaryDirectory(prefix="pignn-coloring-integration-") as root:
            baselines = build_ml_graph_baselines(
                "coloring",
                train_instances=[instance],
                validation_instances=[instance],
                artifact_dir=Path(root),
                config=config,
            )
        self.assertIn(PIGNN_COLORING_BASELINE_NAME, baselines)


if __name__ == "__main__":
    unittest.main()
