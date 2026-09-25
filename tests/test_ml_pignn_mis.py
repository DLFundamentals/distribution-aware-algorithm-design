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
from ml_baselines.checkpoint import load_checkpoint
from ml_baselines.pignn_mis import (
    PIGNN_MIS_BASELINE_NAME,
    PiGNNMISConfig,
    build_pignn_mis_baselines,
    decode_pignn_mis_scores,
    mis_qubo_loss_from_probabilities,
)
from ml_baselines.torch_utils import require_torch, torch_available


def _assert_valid_mis(instance: dict[str, object], raw_solution: object) -> list[int]:
    problem = get_problem_definition("mis")
    solution = problem.canonicalize_solution(raw_solution, instance)
    valid, error = problem.validate_solution(solution, instance)
    if not valid:
        raise AssertionError(error)
    return solution


class PiGNNMISDecoderTests(unittest.TestCase):
    def test_decoder_repairs_path_cycle_and_clique(self) -> None:
        instances = [
            {"id": "path", "num_vertices": 4, "edges": [[0, 1], [1, 2], [2, 3]], "min_size": 2},
            {"id": "cycle", "num_vertices": 5, "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], "min_size": 2},
            {
                "id": "clique",
                "num_vertices": 4,
                "edges": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
                "min_size": 1,
            },
        ]
        for instance in instances:
            with self.subTest(instance=instance["id"]):
                scores = [1.0, 0.8, 0.6, 0.4, 0.2][: int(instance["num_vertices"])]
                solution = decode_pignn_mis_scores(instance, scores, inference_restarts=4, repair_budget=16)
                canonical = _assert_valid_mis(instance, solution)
                self.assertGreaterEqual(len(canonical), int(instance["min_size"]))

    @unittest.skipUnless(torch_available(), "PyTorch is not installed in this environment.")
    def test_qubo_loss_prefers_independent_assignment(self) -> None:
        torch = require_torch()
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        independent = torch.tensor([1.0, 0.0])
        conflicting = torch.tensor([1.0, 1.0])

        independent_loss = mis_qubo_loss_from_probabilities(independent, edge_index, qubo_penalty=2.0)
        conflicting_loss = mis_qubo_loss_from_probabilities(conflicting, edge_index, qubo_penalty=2.0)

        self.assertLess(float(independent_loss.item()), float(conflicting_loss.item()))


@unittest.skipUnless(torch_available(), "PyTorch is not installed in this environment.")
class PiGNNMISEndToEndTests(unittest.TestCase):
    def test_builder_trains_saves_checkpoint_and_solves_tiny_graph(self) -> None:
        instance = {
            "id": "tiny-mis-train",
            "num_vertices": 5,
            "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
        }
        config = PiGNNMISConfig(epochs=3, hidden_dim=8, layers=1, inference_restarts=4, repair_budget=16, seed=47)

        with tempfile.TemporaryDirectory(prefix="pignn-mis-") as root:
            baselines = build_pignn_mis_baselines(
                "mis",
                train_instances=[instance],
                val_instances=[instance],
                artifact_dir=Path(root),
                config=config,
            )

            self.assertEqual(set(baselines), {PIGNN_MIS_BASELINE_NAME})
            checkpoint_path = Path(root) / f"{PIGNN_MIS_BASELINE_NAME}.pt"
            self.assertTrue(checkpoint_path.exists())
            checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", use_torch=True)
            self.assertEqual(checkpoint["metadata"]["baseline_name"], PIGNN_MIS_BASELINE_NAME)
            solution = baselines[PIGNN_MIS_BASELINE_NAME](instance)

        _assert_valid_mis(instance, solution)

    def test_generic_graph_integration_includes_pignn_mis(self) -> None:
        instance = {
            "id": "tiny-mis-integration",
            "num_vertices": 4,
            "edges": [[0, 1], [1, 2], [2, 3]],
        }
        config = {"epochs": 2, "hidden_dim": 8, "layers": 1, "repair_budget": 8, "seed": 49}

        with tempfile.TemporaryDirectory(prefix="pignn-mis-integration-") as root:
            baselines = build_ml_graph_baselines(
                "mis",
                train_instances=[instance],
                validation_instances=[instance],
                artifact_dir=Path(root),
                config=config,
            )

        self.assertIn(PIGNN_MIS_BASELINE_NAME, baselines)


if __name__ == "__main__":
    unittest.main()
