from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

BASELINES_SRC = Path(__file__).resolve().parents[1] / "baselines" / "src"
if str(BASELINES_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINES_SRC))

from dasbench.integrations.ml_graph_baseline import build_ml_item_resource_baselines
from dasbench.problems import get_problem_definition
from ml_baselines.checkpoint import load_checkpoint
from ml_baselines.pdl_packinglp import (
    PDL_PACKINGLP_BASELINE_NAME,
    PDLPackingLPConfig,
    PDLPackingLPNet,
    build_pdl_packinglp_baselines,
    project_and_fill_packinglp,
)
from ml_baselines.tensorize import tensorize_packing_instance
from ml_baselines.torch_utils import torch_available


def _assert_valid_packinglp(instance: dict[str, object], raw_solution: object) -> list[float]:
    problem = get_problem_definition("packing_lp")
    solution = problem.canonicalize_solution(raw_solution, instance)
    valid, error = problem.validate_solution(solution, instance)
    if not valid:
        raise AssertionError(error)
    return solution


class PDLPackingLPDecoderTests(unittest.TestCase):
    def test_projection_makes_single_resource_solution_feasible(self) -> None:
        instance = {
            "id": "tiny-packing-lp",
            "num_items": 2,
            "num_resources": 1,
            "values": [10.0, 6.0],
            "weights": [[2.0], [1.0]],
            "capacities": [2.0],
        }

        solution = project_and_fill_packinglp(instance, [1.0, 1.0], repair_budget=4)

        _assert_valid_packinglp(instance, solution)
        self.assertGreater(sum(solution), 0.0)

    def test_projection_handles_two_resource_instance(self) -> None:
        instance = {
            "id": "tiny-two-resource",
            "num_items": 3,
            "num_resources": 2,
            "values": [9.0, 7.0, 4.0],
            "weights": [[3.0, 1.0], [1.0, 3.0], [1.0, 1.0]],
            "capacities": [3.0, 3.0],
        }

        solution = project_and_fill_packinglp(instance, [1.0, 1.0, 1.0], repair_budget=4)

        _assert_valid_packinglp(instance, solution)

    def test_tensorization_drops_hidden_metadata(self) -> None:
        instance = {
            "id": "tiny-public",
            "num_items": 2,
            "num_resources": 1,
            "values": [10.0, 6.0],
            "weights": [[2.0], [1.0]],
            "capacities": [2.0],
            "optimum_solution": [1.0, 0.0],
            "_hidden_hint": [0.0, 1.0],
        }

        tensor = tensorize_packing_instance(instance, backend="numpy")

        self.assertEqual(tensor.num_items, 2)
        self.assertEqual(tensor.num_resources, 1)


@unittest.skipUnless(torch_available(), "PyTorch is not installed in this environment.")
class PDLPackingLPEndToEndTests(unittest.TestCase):
    def test_dual_head_is_nonnegative(self) -> None:
        instance = {
            "id": "tiny-dual",
            "num_items": 2,
            "num_resources": 1,
            "values": [10.0, 6.0],
            "weights": [[2.0], [1.0]],
            "capacities": [2.0],
        }
        tensor = tensorize_packing_instance(instance, backend="torch", device="cpu")
        model = PDLPackingLPNet(
            int(tensor.item_features.shape[-1]),
            int(tensor.resource_features.shape[-1]),
            hidden_dim=8,
            layers=1,
        )

        _primal, dual = model(tensor)

        self.assertTrue(bool((dual >= 0.0).all()))

    def test_builder_trains_saves_checkpoint_and_solves_tiny_instance(self) -> None:
        instance = {
            "id": "tiny-packing-train",
            "num_items": 3,
            "num_resources": 1,
            "values": [10.0, 6.0, 3.0],
            "weights": [[2.0], [1.0], [1.0]],
            "capacities": [2.0],
        }
        config = PDLPackingLPConfig(epochs=3, hidden_dim=8, layers=1, repair_budget=8, seed=53)

        with tempfile.TemporaryDirectory(prefix="pdl-packinglp-") as root:
            baselines = build_pdl_packinglp_baselines(
                "packing_lp",
                train_instances=[instance],
                val_instances=[instance],
                artifact_dir=Path(root),
                config=config,
            )

            self.assertEqual(set(baselines), {PDL_PACKINGLP_BASELINE_NAME})
            checkpoint_path = Path(root) / f"{PDL_PACKINGLP_BASELINE_NAME}.pt"
            self.assertTrue(checkpoint_path.exists())
            checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", use_torch=True)
            self.assertEqual(checkpoint["metadata"]["baseline_name"], PDL_PACKINGLP_BASELINE_NAME)
            solution = baselines[PDL_PACKINGLP_BASELINE_NAME](instance)

        _assert_valid_packinglp(instance, solution)

    def test_generic_item_resource_integration_includes_pdl_packinglp(self) -> None:
        instance = {
            "id": "tiny-packing-integration",
            "num_items": 2,
            "num_resources": 1,
            "values": [10.0, 6.0],
            "weights": [[2.0], [1.0]],
            "capacities": [2.0],
        }
        config = {"epochs": 2, "hidden_dim": 8, "layers": 1, "repair_budget": 8, "seed": 59}

        with tempfile.TemporaryDirectory(prefix="pdl-packinglp-integration-") as root:
            baselines = build_ml_item_resource_baselines(
                "packing_lp",
                train_instances=[instance],
                validation_instances=[instance],
                artifact_dir=Path(root),
                config=config,
            )

        self.assertIn(PDL_PACKINGLP_BASELINE_NAME, baselines)


if __name__ == "__main__":
    unittest.main()
