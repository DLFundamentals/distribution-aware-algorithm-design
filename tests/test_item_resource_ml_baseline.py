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
from ml_baselines.item_resource_baselines import (
    MDKP_BASELINE_NAME,
    PACKINGLP_BASELINE_NAME,
    ItemResourceConfig,
    decode_mdkp_scores,
    decode_packinglp_fractions,
)
from ml_baselines.torch_utils import torch_available


def _assert_valid(problem_name: str, instance: dict[str, object], raw_solution: object) -> None:
    problem = get_problem_definition(problem_name)
    solution = problem.canonicalize_solution(raw_solution, instance)
    valid, error = problem.validate_solution(solution, instance)
    if not valid:
        raise AssertionError(error)


class ItemResourceDecoderTests(unittest.TestCase):
    def test_mdkp_decoding_repairs_single_resource_knapsack(self) -> None:
        instance = {
            "id": "tiny-knapsack",
            "num_items": 3,
            "num_resources": 1,
            "values": [6, 5, 4],
            "weights": [[4], [3], [2]],
            "capacities": [5],
        }

        solution = decode_mdkp_scores(instance, [0.9, 0.8, 0.7], repair_budget=8)

        _assert_valid("mdkp", instance, solution)
        self.assertEqual(solution, [1, 2])

    def test_mdkp_decoding_repairs_two_resource_instance(self) -> None:
        instance = {
            "id": "tiny-mdkp",
            "num_items": 4,
            "num_resources": 2,
            "values": [8, 7, 5, 4],
            "weights": [[4, 1], [3, 4], [2, 2], [1, 3]],
            "capacities": [5, 5],
        }

        solution = decode_mdkp_scores(instance, [0.95, 0.9, 0.7, 0.3], repair_budget=16)

        _assert_valid("mdkp", instance, solution)

    def test_packing_lp_decoding_projects_and_fills_fractional_solution(self) -> None:
        instance = {
            "id": "tiny-packing-lp",
            "num_items": 2,
            "num_resources": 1,
            "values": [10.0, 6.0],
            "weights": [[2.0], [1.0]],
            "capacities": [2.0],
        }

        solution = decode_packinglp_fractions(instance, [1.0, 1.0], repair_budget=4)

        _assert_valid("packing_lp", instance, solution)
        self.assertGreater(sum(solution), 0.0)


@unittest.skipUnless(torch_available(), "PyTorch is not installed in this environment.")
class ItemResourceEndToEndSmokeTests(unittest.TestCase):
    def test_builders_train_and_solve_tiny_instances(self) -> None:
        instances = {
            "mdkp": {
                "id": "tiny-mdkp-train",
                "num_items": 4,
                "num_resources": 2,
                "values": [8, 7, 5, 4],
                "weights": [[4, 1], [3, 4], [2, 2], [1, 3]],
                "capacities": [5, 5],
            },
            "packing_lp": {
                "id": "tiny-packing-train",
                "num_items": 3,
                "num_resources": 1,
                "values": [10.0, 6.0, 3.0],
                "weights": [[2.0], [1.0], [1.0]],
                "capacities": [2.0],
            },
        }
        names = {
            "mdkp": MDKP_BASELINE_NAME,
            "packing_lp": PACKINGLP_BASELINE_NAME,
        }
        config = ItemResourceConfig(epochs=2, hidden_dim=8, repair_budget=8, seed=23)

        with tempfile.TemporaryDirectory(prefix="item-resource-ml-baseline-") as root:
            root_path = Path(root)
            for problem_name, instance in instances.items():
                with self.subTest(problem=problem_name):
                    baselines = build_ml_item_resource_baselines(
                        problem_name,
                        train_instances=[instance],
                        validation_instances=[instance],
                        artifact_dir=root_path / problem_name,
                        config=config.to_dict(),
                    )
                    self.assertIn(names[problem_name], baselines)
                    _assert_valid(problem_name, instance, baselines[names[problem_name]](instance))


if __name__ == "__main__":
    unittest.main()
