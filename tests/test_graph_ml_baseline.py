from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

BASELINES_SRC = Path(__file__).resolve().parents[1] / "baselines" / "src"
if str(BASELINES_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINES_SRC))

from dasbench.problems import get_problem_definition
from ml_baselines.graph_score_repair import (
    GraphScoreRepairConfig,
    build_graph_score_repair_baselines,
    decode_coloring_scores,
    decode_mds_scores,
    decode_mis_scores,
)
from ml_baselines.torch_utils import torch_available


def _assert_valid(problem_name: str, instance: dict[str, object], raw_solution: object) -> None:
    problem = get_problem_definition(problem_name)
    solution = problem.canonicalize_solution(raw_solution, instance)
    valid, error = problem.validate_solution(solution, instance)
    if not valid:
        raise AssertionError(error)


class GraphMLDecoderTests(unittest.TestCase):
    def test_mis_decoding_repairs_path_cycle_and_clique(self) -> None:
        instances = [
            {"id": "path", "num_vertices": 4, "edges": [[0, 1], [1, 2], [2, 3]]},
            {"id": "cycle", "num_vertices": 5, "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]},
            {
                "id": "clique",
                "num_vertices": 4,
                "edges": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
            },
        ]
        for instance in instances:
            with self.subTest(instance=instance["id"]):
                solution = decode_mis_scores(instance, [1.0, 0.8, 0.6, 0.4, 0.2][: int(instance["num_vertices"])])
                _assert_valid("mis", instance, solution)

    def test_mds_decoding_repairs_star_and_path(self) -> None:
        instances = [
            {"id": "star", "num_vertices": 5, "edges": [[0, 1], [0, 2], [0, 3], [0, 4]]},
            {"id": "path", "num_vertices": 5, "edges": [[0, 1], [1, 2], [2, 3], [3, 4]]},
        ]
        for instance in instances:
            with self.subTest(instance=instance["id"]):
                solution = decode_mds_scores(instance, [1.0, 0.5, 0.4, 0.3, 0.2])
                _assert_valid("mds", instance, solution)

    def test_coloring_decoding_repairs_odd_cycle_and_clique(self) -> None:
        instances = [
            {"id": "odd-cycle", "num_vertices": 5, "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]},
            {
                "id": "clique",
                "num_vertices": 4,
                "edges": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
            },
        ]
        for instance in instances:
            with self.subTest(instance=instance["id"]):
                solution = decode_coloring_scores(instance, [0.2, 1.0, 0.4, 0.8, 0.6][: int(instance["num_vertices"])])
                _assert_valid("coloring", instance, solution)


@unittest.skipUnless(torch_available(), "PyTorch is not installed in this environment.")
class GraphMLEndToEndSmokeTests(unittest.TestCase):
    def test_builders_fit_and_solve_tiny_graphs(self) -> None:
        config = GraphScoreRepairConfig(epochs=2, hidden_dim=8, layers=2, repair_budget=16, seed=11)
        cases = {
            "mis": {"id": "path", "num_vertices": 4, "edges": [[0, 1], [1, 2], [2, 3]]},
            "mds": {"id": "star", "num_vertices": 5, "edges": [[0, 1], [0, 2], [0, 3], [0, 4]]},
            "coloring": {
                "id": "odd-cycle",
                "num_vertices": 5,
                "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
            },
        }
        with tempfile.TemporaryDirectory(prefix="graph-ml-baseline-") as root:
            root_path = Path(root)
            for problem_name, instance in cases.items():
                with self.subTest(problem=problem_name):
                    baselines = build_graph_score_repair_baselines(
                        problem_name,
                        [instance],
                        [instance],
                        artifact_dir=root_path / problem_name,
                        config=config,
                    )
                    self.assertEqual(len(baselines), 1)
                    solver = next(iter(baselines.values()))
                    _assert_valid(problem_name, instance, solver(instance))


if __name__ == "__main__":
    unittest.main()
