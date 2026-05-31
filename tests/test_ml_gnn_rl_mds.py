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
from ml_baselines.gnn_rl_mds import (
    GNN_RL_MDS_BASELINE_NAME,
    GNNRLMDSConfig,
    DominatingSetEnvironment,
    build_gnn_rl_mds_baselines,
    repair_dominating_set,
)
from ml_baselines.torch_utils import torch_available


def _assert_valid_mds(instance: dict[str, object], raw_solution: object) -> list[int]:
    problem = get_problem_definition("mds")
    solution = problem.canonicalize_solution(raw_solution, instance)
    valid, error = problem.validate_solution(solution, instance)
    if not valid:
        raise AssertionError(error)
    return solution


class GNNRLMDSRepairTests(unittest.TestCase):
    def test_environment_selecting_star_center_terminates(self) -> None:
        instance = {
            "id": "star",
            "num_vertices": 5,
            "edges": [[0, 1], [0, 2], [0, 3], [0, 4]],
        }
        env = DominatingSetEnvironment(instance, max_steps=5)

        reward, done = env.step(0)

        self.assertTrue(done)
        self.assertGreater(reward, 0.0)
        self.assertEqual(env.selected, {0})
        self.assertEqual(env.dominated, {0, 1, 2, 3, 4})

    def test_repair_empty_star_returns_center(self) -> None:
        instance = {
            "id": "star",
            "num_vertices": 5,
            "edges": [[0, 1], [0, 2], [0, 3], [0, 4]],
        }

        solution = repair_dominating_set(instance, [], repair_budget=8)

        self.assertEqual(solution, [0])
        _assert_valid_mds(instance, solution)

    def test_repair_path_and_single_node_are_valid(self) -> None:
        instances = [
            {"id": "path", "num_vertices": 5, "edges": [[0, 1], [1, 2], [2, 3], [3, 4]]},
            {"id": "single", "num_vertices": 1, "edges": []},
        ]
        for instance in instances:
            with self.subTest(instance=instance["id"]):
                solution = repair_dominating_set(instance, [], repair_budget=8)
                _assert_valid_mds(instance, solution)

    def test_pruning_preserves_domination(self) -> None:
        instance = {
            "id": "star",
            "num_vertices": 5,
            "edges": [[0, 1], [0, 2], [0, 3], [0, 4]],
        }

        solution = repair_dominating_set(instance, [0, 1, 2, 3, 4], repair_budget=8)

        self.assertEqual(solution, [0])
        _assert_valid_mds(instance, solution)


@unittest.skipUnless(torch_available(), "PyTorch is not installed in this environment.")
class GNNRLMDSIntegrationTests(unittest.TestCase):
    def test_builder_trains_saves_checkpoint_and_solves_tiny_graph(self) -> None:
        instance = {
            "id": "tiny-mds-train",
            "num_vertices": 5,
            "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
        }
        config = GNNRLMDSConfig(
            episodes=3,
            hidden_dim=8,
            layers=1,
            batch_size=2,
            target_update_interval=2,
            max_steps_factor=1.5,
            repair_budget=16,
            seed=41,
        )

        with tempfile.TemporaryDirectory(prefix="gnn-rl-mds-") as root:
            baselines = build_gnn_rl_mds_baselines(
                "mds",
                train_instances=[instance],
                val_instances=[instance],
                artifact_dir=Path(root),
                config=config,
            )

            self.assertEqual(set(baselines), {GNN_RL_MDS_BASELINE_NAME})
            checkpoint_path = Path(root) / f"{GNN_RL_MDS_BASELINE_NAME}.pt"
            self.assertTrue(checkpoint_path.exists())
            checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", use_torch=True)
            self.assertEqual(checkpoint["metadata"]["baseline_name"], GNN_RL_MDS_BASELINE_NAME)
            solution = baselines[GNN_RL_MDS_BASELINE_NAME](instance)

        _assert_valid_mds(instance, solution)

    def test_generic_graph_integration_includes_gnn_rl_mds(self) -> None:
        instance = {
            "id": "tiny-mds-integration",
            "num_vertices": 5,
            "edges": [[0, 1], [0, 2], [0, 3], [0, 4]],
        }
        config = {"epochs": 2, "hidden_dim": 8, "layers": 1, "batch_size": 2, "repair_budget": 8, "seed": 43}

        with tempfile.TemporaryDirectory(prefix="gnn-rl-mds-integration-") as root:
            baselines = build_ml_graph_baselines(
                "mds",
                train_instances=[instance],
                validation_instances=[instance],
                artifact_dir=Path(root),
                config=config,
            )

        self.assertIn(GNN_RL_MDS_BASELINE_NAME, baselines)


if __name__ == "__main__":
    unittest.main()
