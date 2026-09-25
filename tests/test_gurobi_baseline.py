from __future__ import annotations

import unittest
from unittest.mock import patch

from dasbench.eval.evaluator import evaluate_solver
from dasbench.integrations.gurobi_baseline import GurobiBaselineConfig, build_gurobi_solver
from dasbench.problems import get_problem_definition
import pytest

pytestmark = pytest.mark.gurobi


class GurobiBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GurobiBaselineConfig(time_limit_seconds=10.0, threads=1)

    def _assert_matches_exact(self, problem_name: str, instance: dict[str, object]) -> None:
        problem = get_problem_definition(problem_name)
        exact = problem.exact_solver(instance)
        solver = build_gurobi_solver(problem_name, self.config)
        outcome = solver(instance)
        solution = problem.canonicalize_solution(outcome.solution, instance)
        score = problem.score_solution({**instance, "optimum_objective": exact.objective_value}, solution)
        self.assertTrue(score.is_feasible)
        self.assertTrue(score.is_optimal)

    def test_maxsat_formulation_matches_exact(self) -> None:
        self._assert_matches_exact(
            "maxsat",
            {
                "id": "gurobi-maxsat",
                "num_variables": 4,
                "clauses": [[1, 2, 3], [-1, 2, 4], [1, -3, -4], [-2, 3, 4]],
            },
        )

    def test_mis_formulation_matches_exact(self) -> None:
        self._assert_matches_exact(
            "mis",
            {
                "id": "gurobi-mis",
                "num_vertices": 5,
                "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]],
            },
        )

    def test_mds_formulation_matches_exact(self) -> None:
        self._assert_matches_exact(
            "mds",
            {
                "id": "gurobi-mds",
                "num_vertices": 5,
                "edges": [[0, 1], [0, 2], [0, 3], [3, 4]],
            },
        )

    def test_coloring_formulation_matches_exact(self) -> None:
        self._assert_matches_exact(
            "coloring",
            {
                "id": "gurobi-coloring",
                "num_vertices": 5,
                "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 2]],
            },
        )

    def test_tsp_formulation_matches_exact(self) -> None:
        self._assert_matches_exact(
            "tsp",
            {
                "id": "gurobi-tsp",
                "num_cities": 6,
                "points": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.2], [2.1, 1.1], [0.9, 1.9], [0.0, 1.0]],
            },
        )

    def test_packing_lp_formulation_matches_exact(self) -> None:
        self._assert_matches_exact(
            "packing_lp",
            {
                "id": "gurobi-packing-lp",
                "num_items": 2,
                "num_resources": 1,
                "values": [2, 1],
                "weights": [[1], [1]],
                "capacities": [1],
            },
        )

    def test_mdkp_formulation_matches_exact(self) -> None:
        self._assert_matches_exact(
            "mdkp",
            {
                "id": "gurobi-mdkp",
                "num_items": 3,
                "num_resources": 2,
                "values": [6, 5, 4],
                "weights": [[4, 2], [3, 4], [2, 3]],
                "capacities": [5, 5],
            },
        )

    def test_gurobi_failure_becomes_failed_summary_not_exception(self) -> None:
        instance = {
            "id": "gurobi-failure",
            "num_vertices": 4,
            "edges": [[0, 1], [1, 2], [2, 3]],
            "optimum_objective": 2.0,
        }
        solver = build_gurobi_solver("mis", self.config)
        with patch("dasbench.integrations.gurobi_baseline._thread_env", side_effect=RuntimeError("license failure")):
            summary = evaluate_solver("mis", "gurobi_timed", solver, [instance], split="validation")
        self.assertEqual(summary["average_normalized_quality"], 0.0)
        self.assertIn("license failure", str(summary["error"]))


if __name__ == "__main__":
    unittest.main()
