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
from dasbench.problems.packing_utils import selection_objective_value
from ml_baselines.checkpoint import load_checkpoint
from ml_baselines.drl_mdkp import (
    DRL_MDKP_BASELINE_NAME,
    DRLMDKPConfig,
    MDKPEnvironment,
    build_drl_mdkp_baselines,
    heuristic_feasible_solution,
    item_worth_order,
    repair_mdkp_selection,
)
from ml_baselines.torch_utils import torch_available


def _assert_valid_mdkp(instance: dict[str, object], raw_solution: object) -> list[int]:
    problem = get_problem_definition("mdkp")
    solution = problem.canonicalize_solution(raw_solution, instance)
    valid, error = problem.validate_solution(solution, instance)
    if not valid:
        raise AssertionError(error)
    return solution


class DRLMDKPEnvironmentTests(unittest.TestCase):
    def test_heuristic_solution_is_feasible_and_uses_item_worth_order(self) -> None:
        instance = {
            "id": "tiny-knapsack",
            "num_items": 3,
            "num_resources": 1,
            "values": [6, 5, 4],
            "weights": [[4], [3], [2]],
            "capacities": [5],
        }

        self.assertEqual(item_worth_order(instance), [2, 1, 0])
        solution = heuristic_feasible_solution(instance)

        _assert_valid_mdkp(instance, solution)
        self.assertEqual(solution, [1, 2])

    def test_action_mask_prevents_infeasible_additions(self) -> None:
        instance = {
            "id": "tiny-mask",
            "num_items": 3,
            "num_resources": 1,
            "values": [6, 5, 4],
            "weights": [[5], [4], [1]],
            "capacities": [5],
        }
        env = MDKPEnvironment(instance, initial_solution=[0], max_episode_steps=4)

        self.assertEqual(env.feasible_actions(), [False, False, False])

    def test_reward_increases_on_feasible_value_improving_action(self) -> None:
        instance = {
            "id": "tiny-reward",
            "num_items": 2,
            "num_resources": 1,
            "values": [6, 3],
            "weights": [[3], [2]],
            "capacities": [5],
        }
        env = MDKPEnvironment(instance, initial_solution=[], max_episode_steps=4)

        reward, done = env.step(0)

        self.assertGreater(reward, 0.0)
        self.assertFalse(done)

    def test_repair_returns_feasible_solution_after_overfull_start(self) -> None:
        instance = {
            "id": "tiny-repair",
            "num_items": 4,
            "num_resources": 2,
            "values": [8, 7, 5, 4],
            "weights": [[4, 1], [3, 4], [2, 2], [1, 3]],
            "capacities": [5, 5],
        }

        solution = repair_mdkp_selection(instance, [0, 1, 2, 3], repair_budget=16)

        _assert_valid_mdkp(instance, solution)
        self.assertGreater(selection_objective_value(instance, solution), 0.0)


@unittest.skipUnless(torch_available(), "PyTorch is not installed in this environment.")
class DRLMDKPEndToEndSmokeTests(unittest.TestCase):
    def test_builder_trains_saves_checkpoint_and_solves_tiny_instance(self) -> None:
        instance = {
            "id": "tiny-mdkp-train",
            "num_items": 4,
            "num_resources": 2,
            "values": [8, 7, 5, 4],
            "weights": [[4, 1], [3, 4], [2, 2], [1, 3]],
            "capacities": [5, 5],
        }
        config = DRLMDKPConfig(episodes=3, hidden_dim=8, max_episode_steps=4, repair_budget=16, seed=29)

        with tempfile.TemporaryDirectory(prefix="drl-mdkp-") as root:
            baselines = build_drl_mdkp_baselines(
                "mdkp",
                train_instances=[instance],
                val_instances=[instance],
                artifact_dir=Path(root),
                config=config,
            )

            self.assertEqual(set(baselines), {DRL_MDKP_BASELINE_NAME})
            checkpoint_path = Path(root) / f"{DRL_MDKP_BASELINE_NAME}.pt"
            self.assertTrue(checkpoint_path.exists())
            checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", use_torch=True)
            self.assertEqual(checkpoint["metadata"]["baseline_name"], DRL_MDKP_BASELINE_NAME)
            solution = baselines[DRL_MDKP_BASELINE_NAME](instance)

        _assert_valid_mdkp(instance, solution)

    def test_generic_item_resource_integration_includes_drl_mdkp(self) -> None:
        instance = {
            "id": "tiny-mdkp-integration",
            "num_items": 3,
            "num_resources": 1,
            "values": [6, 5, 4],
            "weights": [[4], [3], [2]],
            "capacities": [5],
        }
        config = {"epochs": 2, "hidden_dim": 8, "max_episode_steps": 4, "repair_budget": 8, "seed": 31}

        with tempfile.TemporaryDirectory(prefix="drl-mdkp-integration-") as root:
            baselines = build_ml_item_resource_baselines(
                "mdkp",
                train_instances=[instance],
                validation_instances=[instance],
                artifact_dir=Path(root),
                config=config,
            )

        self.assertIn(DRL_MDKP_BASELINE_NAME, baselines)


if __name__ == "__main__":
    unittest.main()
