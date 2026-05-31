from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

BASELINES_SRC = Path(__file__).resolve().parents[1] / "baselines" / "src"
if str(BASELINES_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINES_SRC))

from dasbench.integrations.ml_graph_baseline import build_ml_maxsat_baselines
from dasbench.problems import get_problem_definition
from dasbench.problems.maxsat import count_satisfied_clauses
from ml_baselines.checkpoint import load_checkpoint
from ml_baselines.runcsp_maxsat import (
    RUN_CSP_MAXSAT_BASELINE_NAME,
    RunCSPMaxSatConfig,
    bounded_walksat_local_search,
    decode_runcsp_assignment,
    tensorize_runcsp_maxsat_instance,
    weighted_satisfied_score,
)
from ml_baselines.torch_utils import torch_available


def _assert_valid_maxsat(instance: dict[str, object], raw_solution: object) -> list[bool]:
    problem = get_problem_definition("maxsat")
    solution = problem.canonicalize_solution(raw_solution, instance)
    valid, error = problem.validate_solution(solution, instance)
    if not valid:
        raise AssertionError(error)
    return solution


class RunCSPMaxSatDecoderTests(unittest.TestCase):
    def test_decode_tiny_satisfiable_cnf_returns_valid_assignment(self) -> None:
        instance = {
            "id": "tiny-sat",
            "num_variables": 3,
            "clauses": [
                [1, 2, 3],
                [1, -2, 3],
                [1, 2, -3],
            ],
        }

        solution = decode_runcsp_assignment(instance, [0.9, 0.9, 0.9], samples=2, walksat_flips=8, seed=5)

        assignment = _assert_valid_maxsat(instance, solution)
        self.assertEqual(count_satisfied_clauses(instance, assignment), 3)

    def test_decode_tiny_partial_cnf_preserves_assignment_format_and_objective(self) -> None:
        instance = {
            "id": "tiny-partial",
            "num_variables": 3,
            "clauses": [
                [1, 2, 3],
                [1, 2, -3],
                [1, -2, 3],
                [1, -2, -3],
                [-1, 2, 3],
                [-1, 2, -3],
                [-1, -2, 3],
                [-1, -2, -3],
            ],
        }

        solution = decode_runcsp_assignment(instance, [0.4, 0.6, 0.5], samples=4, walksat_flips=16, seed=13)

        assignment = _assert_valid_maxsat(instance, solution)
        self.assertEqual(len(assignment), 3)
        self.assertEqual(count_satisfied_clauses(instance, assignment), 7)

    def test_weighted_score_prefers_high_weight_clause(self) -> None:
        instance = {
            "id": "tiny-weighted",
            "num_variables": 3,
            "clauses": [
                [1, 2, 3],
                [-1, -2, -3],
            ],
            "weights": [10.0, 1.0],
        }

        assignment = decode_runcsp_assignment(instance, [0.9, 0.9, 0.9], samples=0, walksat_flips=0, seed=3)

        self.assertEqual(weighted_satisfied_score(instance, assignment), 10.0)

    def test_walksat_honors_zero_flip_budget(self) -> None:
        instance = {
            "id": "tiny-budget",
            "num_variables": 3,
            "clauses": [[1, 2, 3], [-1, -2, -3]],
        }
        start = [True, True, True]

        self.assertEqual(bounded_walksat_local_search(instance, start, max_flips=0, seed=2), start)

    def test_tensorization_uses_public_fields_only(self) -> None:
        instance = {
            "id": "tiny-public",
            "num_variables": 3,
            "clauses": [[1, 2, 3], [-1, -2, -3]],
            "weights": [2.0, 1.0],
            "optimum_assignment": [True, False, True],
            "_hidden_hint": [False, False, False],
        }

        tensor = tensorize_runcsp_maxsat_instance(instance, backend="numpy")

        self.assertEqual(tensor.num_variables, 3)
        self.assertEqual(tensor.num_clauses, 2)
        self.assertEqual(list(tensor.clause_weights), [2.0, 1.0])


@unittest.skipUnless(torch_available(), "PyTorch is not installed in this environment.")
class RunCSPMaxSatEndToEndSmokeTests(unittest.TestCase):
    def test_builder_trains_saves_checkpoint_and_solves_tiny_instance(self) -> None:
        train_instance = {
            "id": "tiny-train",
            "num_variables": 3,
            "clauses": [
                [1, 2, 3],
                [1, -2, 3],
                [1, 2, -3],
                [-1, -2, -3],
            ],
        }
        config = RunCSPMaxSatConfig(
            epochs=2,
            hidden_dim=8,
            message_passing_steps=2,
            samples=2,
            walksat_restarts=2,
            walksat_flips=8,
            seed=19,
        )

        with tempfile.TemporaryDirectory(prefix="runcsp-maxsat-") as root:
            baselines = build_ml_maxsat_baselines(
                "maxsat",
                train_instances=[train_instance],
                validation_instances=[train_instance],
                artifact_dir=Path(root),
                config=config.to_dict(),
            )

            self.assertIn(RUN_CSP_MAXSAT_BASELINE_NAME, baselines)
            checkpoint_path = Path(root) / f"{RUN_CSP_MAXSAT_BASELINE_NAME}.pt"
            self.assertTrue(checkpoint_path.exists())
            checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", use_torch=True)
            self.assertEqual(checkpoint["metadata"]["baseline_name"], RUN_CSP_MAXSAT_BASELINE_NAME)
            solution = baselines[RUN_CSP_MAXSAT_BASELINE_NAME](train_instance)

        assignment = _assert_valid_maxsat(train_instance, solution)
        self.assertGreaterEqual(count_satisfied_clauses(train_instance, assignment), 3)


if __name__ == "__main__":
    unittest.main()
