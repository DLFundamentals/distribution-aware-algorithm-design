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
from ml_baselines.maxsat_assignment import (
    MAXSAT_BASELINE_NAME,
    MaxSatAssignmentConfig,
    decode_assignment,
)
from ml_baselines.torch_utils import torch_available


def _assert_valid_maxsat(instance: dict[str, object], raw_solution: object) -> list[bool]:
    problem = get_problem_definition("maxsat")
    solution = problem.canonicalize_solution(raw_solution, instance)
    valid, error = problem.validate_solution(solution, instance)
    if not valid:
        raise AssertionError(error)
    return solution


class MaxSatMLDecoderTests(unittest.TestCase):
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

        solution = decode_assignment(instance, [0.9, 0.9, 0.9], samples=2, walksat_flips=8, seed=5)

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

        solution = decode_assignment(instance, [0.4, 0.6, 0.5], samples=4, walksat_flips=8, seed=13)

        assignment = _assert_valid_maxsat(instance, solution)
        self.assertEqual(len(assignment), 3)
        self.assertEqual(count_satisfied_clauses(instance, assignment), 7)


@unittest.skipUnless(torch_available(), "PyTorch is not installed in this environment.")
class MaxSatMLEndToEndSmokeTests(unittest.TestCase):
    def test_builder_trains_and_solves_tiny_instance(self) -> None:
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
        config = MaxSatAssignmentConfig(epochs=2, hidden_dim=8, layers=2, samples=2, walksat_flips=8, seed=19)

        with tempfile.TemporaryDirectory(prefix="maxsat-ml-baseline-") as root:
            baselines = build_ml_maxsat_baselines(
                "maxsat",
                train_instances=[train_instance],
                validation_instances=[train_instance],
                artifact_dir=Path(root),
                config=config.to_dict(),
            )

            self.assertIn(MAXSAT_BASELINE_NAME, baselines)
            solution = baselines[MAXSAT_BASELINE_NAME](train_instance)

        assignment = _assert_valid_maxsat(train_instance, solution)
        self.assertGreaterEqual(count_satisfied_clauses(train_instance, assignment), 3)


if __name__ == "__main__":
    unittest.main()
