from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

BASELINES_SRC = Path(__file__).resolve().parents[1] / "baselines" / "src"
if str(BASELINES_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINES_SRC))

from ml_baselines.checkpoint import load_checkpoint, save_checkpoint
from ml_baselines.config import MLBaselineConfig
from ml_baselines.interface import BaselineAdapter, TrainedState
from ml_baselines.seeding import seed_everything
from ml_baselines.tensorize import tensorize_graph_instance, tensorize_packing_instance
from ml_baselines.torch_utils import TorchUnavailableError, torch_available


class MLBaselineUtilityTests(unittest.TestCase):
    def test_graph_tensorization_uses_public_fields_only(self) -> None:
        instance = {
            "id": "toy-graph",
            "num_vertices": 3,
            "edges": [[0, 1], [1, 2]],
            "optimum_solution": [0, 2],
            "optimum_objective": 2.0,
            "_private_rule": {"do_not_use": True},
        }

        graph = tensorize_graph_instance(instance, backend="numpy")

        self.assertEqual(graph.instance_id, "toy-graph")
        self.assertEqual(graph.num_nodes, 3)
        self.assertEqual(graph.node_features.shape, (3, 6))
        self.assertEqual(graph.edge_index.shape, (2, 4))
        self.assertEqual(graph.edge_index.tolist(), [[0, 1, 1, 2], [1, 0, 2, 1]])
        self.assertTrue(np.all(graph.node_features[:, -1] == 1.0))

    def test_item_resource_tensorization_uses_public_fields_only(self) -> None:
        instance = {
            "id": "toy-pack",
            "num_items": 2,
            "num_resources": 2,
            "values": [6, 4],
            "weights": [[3, 1], [1, 2]],
            "capacities": [4, 3],
            "optimum_solution": [0, 1],
            "_latent_classes": [1, 0],
        }

        packed = tensorize_packing_instance(instance, backend="numpy")

        self.assertEqual(packed.instance_id, "toy-pack")
        self.assertEqual(packed.num_items, 2)
        self.assertEqual(packed.num_resources, 2)
        self.assertEqual(packed.item_features.shape, (2, 14))
        self.assertEqual(packed.resource_features.shape, (2, 9))
        np.testing.assert_allclose(packed.weights, np.asarray([[3, 1], [1, 2]], dtype=np.float32))
        np.testing.assert_allclose(packed.capacities, np.asarray([4, 3], dtype=np.float32))

    def test_checkpoint_round_trip_without_torch(self) -> None:
        state = {
            "weights": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            "bias": 0.25,
        }

        with tempfile.TemporaryDirectory(prefix="ml-baseline-checkpoint-") as root:
            path = Path(root) / "state.ckpt"
            save_checkpoint(path, state, metadata={"name": "roundtrip"}, use_torch=False)
            loaded = load_checkpoint(path, use_torch=False)

        self.assertEqual(loaded["metadata"], {"name": "roundtrip"})
        np.testing.assert_allclose(loaded["state"]["weights"], state["weights"])
        self.assertEqual(loaded["state"]["bias"], 0.25)

    def test_deterministic_seeding(self) -> None:
        first_state = seed_everything(123)
        first_random = [random.random() for _ in range(3)]
        first_numpy = np.random.rand(3)

        second_state = seed_everything(123)
        second_random = [random.random() for _ in range(3)]
        second_numpy = np.random.rand(3)

        self.assertEqual(first_state.seed, 123)
        self.assertEqual(second_state.seed, 123)
        self.assertEqual(first_random, second_random)
        np.testing.assert_allclose(first_numpy, second_numpy)

    def test_baseline_adapter_exposes_runner_callable(self) -> None:
        def fit(train_instances, val_instances, config):
            return TrainedState(
                payload={
                    "train_count": len(train_instances),
                    "val_count": len(val_instances),
                    "seed": config.seed,
                }
            )

        def solve(instance, trained_state, config):
            return [trained_state.payload["train_count"], trained_state.payload["val_count"], config.seed]

        adapter = BaselineAdapter(fit=fit, solve=solve, config=MLBaselineConfig(seed=7))
        adapter.fit([{"id": "train"}], [{"id": "val"}])

        self.assertEqual(adapter.as_solver()({"id": "test"}), [1, 1, 7])

    def test_manual_message_passing_reports_missing_torch_cleanly(self) -> None:
        if torch_available():
            self.skipTest("Current environment has PyTorch; missing-torch branch is not active.")
        from ml_baselines.models import ManualMessagePassing

        with self.assertRaises(TorchUnavailableError):
            ManualMessagePassing(2, 4, 1)


if __name__ == "__main__":
    unittest.main()
