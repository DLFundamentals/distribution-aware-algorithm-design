from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dasbench.integrations.gurobi_baseline import GurobiBaselineConfig
from dasbench.problems.base import SolveOutcome
from scripts.pace2025_run_heuristic_baselines import (
    GUROBI_BASELINE_NAME,
    _default_gurobi_time_limit_seconds,
    _run_dasbench_baseline_child,
    compare_dasbench_mds_baselines,
)


class DummyQueue:
    def __init__(self) -> None:
        self.payload: dict[str, object] | None = None

    def put(self, payload: dict[str, object]) -> None:
        self.payload = payload


class PaceGurobiBaselineRunnerTests(unittest.TestCase):
    def test_default_gurobi_time_limit_leaves_process_timeout_slack(self) -> None:
        self.assertEqual(_default_gurobi_time_limit_seconds(8.0), 3.0)
        self.assertEqual(_default_gurobi_time_limit_seconds(360.0), 330.0)

    def test_gurobi_baseline_child_uses_gurobi_solver_and_preserves_metadata(self) -> None:
        exposed_instance = {
            "id": "toy-mds",
            "num_vertices": 3,
            "edges": [[0, 1], [1, 2]],
        }
        config = GurobiBaselineConfig(time_limit_seconds=7.0, threads=1)
        queue = DummyQueue()

        with patch("scripts.pace2025_run_heuristic_baselines.build_gurobi_solver") as build_solver:
            build_solver.return_value = lambda _: SolveOutcome(
                solution=[1],
                metadata={"status": "OPTIMAL", "gurobi_runtime_ms": 12.5},
            )
            _run_dasbench_baseline_child(GUROBI_BASELINE_NAME, exposed_instance, queue, config)

        self.assertIsNotNone(queue.payload)
        assert queue.payload is not None
        self.assertEqual(queue.payload["status"], "ok")
        self.assertEqual(queue.payload["solution"], [1])
        self.assertTrue(queue.payload["valid"])
        self.assertEqual(queue.payload["metadata"], {"status": "OPTIMAL", "gurobi_runtime_ms": 12.5})
        build_solver.assert_called_once_with("mds", config)

    def test_gurobi_baseline_name_is_accepted_by_empty_comparison(self) -> None:
        config = GurobiBaselineConfig(time_limit_seconds=7.0, threads=2, mip_gap=0.1)
        with tempfile.TemporaryDirectory(prefix="pace-gurobi-baseline-") as tmp:
            tmp_path = Path(tmp)
            summary = compare_dasbench_mds_baselines(
                baseline_names=[GUROBI_BASELINE_NAME],
                relative_paths=[],
                cache_dir=tmp_path / "cache",
                expanded_dir=tmp_path / "expanded",
                github_ref="master",
                output_dir=tmp_path / "out",
                reference_csv=None,
                timeout_seconds=9.0,
                max_workers=1,
                gurobi_config=config,
            )

            self.assertEqual(summary["gurobi_config"], config.to_record())
            self.assertTrue((tmp_path / "out" / "baseline_results.csv").exists())


if __name__ == "__main__":
    unittest.main()
