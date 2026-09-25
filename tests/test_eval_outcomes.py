from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from dasbench.eval.evaluator import evaluate_solver, evaluate_solver_repeated
from dasbench.problems.base import SolveOutcome


class SolveOutcomeEvaluatorTests(unittest.TestCase):
    def test_evaluate_solver_records_metadata_and_wall_clock(self) -> None:
        instance = {
            "id": "toy-mis-outcome",
            "num_vertices": 4,
            "edges": [[0, 1], [1, 2], [2, 3]],
            "optimum_objective": 2.0,
        }
        diagnostics_path = Path(tempfile.mkdtemp(prefix="dasbench-outcome-")) / "diagnostics.jsonl"

        def solver(_: dict[str, object]) -> SolveOutcome:
            time.sleep(0.01)
            return SolveOutcome(
                solution=[0, 2],
                metadata={
                    "status": "OPTIMAL",
                    "gurobi_runtime_ms": 3.5,
                    "objective_value": 2.0,
                    "best_bound": 2.0,
                    "mip_gap": 0.0,
                    "node_count": 0.0,
                    "solution_count": 1,
                    "time_limit_hit": False,
                },
            )

        summary = evaluate_solver(
            "mis",
            "dummy_outcome",
            solver,
            [instance],
            split="validation",
            diagnostics_path=diagnostics_path,
        )

        self.assertEqual(summary["feasibility_rate"], 1.0)
        self.assertGreaterEqual(float(summary["average_runtime_ms"]), 9.0)
        self.assertEqual(summary["average_gurobi_runtime_ms"], 3.5)
        self.assertEqual(summary["gurobi_runtime_instance_count"], 1)
        rows = [json.loads(line) for line in diagnostics_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "OPTIMAL")
        self.assertEqual(rows[0]["gurobi_runtime_ms"], 3.5)
        self.assertGreaterEqual(float(rows[0]["wall_clock_ms"]), 9.0)

    def test_repeated_evaluation_tracks_gurobi_runtime_summary(self) -> None:
        instance = {
            "id": "toy-mis-repeat",
            "num_vertices": 4,
            "edges": [[0, 1], [1, 2], [2, 3]],
            "optimum_objective": 2.0,
        }

        def solver(_: dict[str, object]) -> SolveOutcome:
            return SolveOutcome(solution=[0, 2], metadata={"gurobi_runtime_ms": 4.0})

        summary = evaluate_solver_repeated(
            "mis",
            "dummy_outcome",
            solver,
            [instance],
            split="validation",
            repeats=2,
        )

        self.assertEqual(summary["average_gurobi_runtime_ms_mean"], 4.0)
        self.assertEqual(summary["average_gurobi_runtime_ms_std"], 0.0)
        self.assertEqual(summary["gurobi_runtime_trial_count"], 2)

    def test_evaluate_solver_times_out_hanging_solver_per_instance(self) -> None:
        slow_instance = {
            "id": "toy-mis-timeout",
            "num_vertices": 2,
            "edges": [[0, 1]],
            "optimum_objective": 1.0,
        }
        fast_instance = {
            "id": "toy-mis-fast-after-timeout",
            "num_vertices": 2,
            "edges": [[0, 1]],
            "optimum_objective": 1.0,
        }

        def solver(instance: dict[str, object]) -> list[int]:
            if instance["id"] == "toy-mis-timeout":
                time.sleep(1.0)
            return [0]

        start = time.perf_counter()
        summary = evaluate_solver(
            "mis",
            "slow_solver",
            solver,
            [slow_instance, fast_instance],
            split="validation",
            timeout_seconds=0.05,
        )

        self.assertLess(time.perf_counter() - start, 0.5)
        self.assertEqual(summary["feasibility_rate"], 0.5)
        self.assertLess(float(summary["average_runtime_ms"]), 100.0)
        self.assertEqual(summary["error_count"], 1)
        self.assertNotIn("error", summary)
        self.assertEqual(summary["failure_cases"][0]["instance_id"], "toy-mis-timeout")
        self.assertIn("SolverTimeoutError", str(summary["failure_cases"][0]["error"]))

    def test_evaluate_solver_fails_memory_hungry_solver(self) -> None:
        instance = {
            "id": "toy-mis-memory",
            "num_vertices": 2,
            "edges": [[0, 1]],
            "optimum_objective": 1.0,
        }

        def solver(_: dict[str, object]) -> list[int]:
            _ = bytearray(2 * 1024 * 1024 * 1024)
            return [0]

        summary = evaluate_solver(
            "mis",
            "memory_hungry_solver",
            solver,
            [instance],
            split="validation",
            memory_limit_mb=1024,
        )

        self.assertEqual(summary["feasibility_rate"], 0.0)
        self.assertEqual(summary["average_runtime_ms"], 1_000_000.0)
        self.assertIn("MemoryError", str(summary["error"]))
        self.assertIn("1024.0 MiB", str(summary["error"]))
        self.assertEqual(summary["failure_cases"][0]["instance_id"], "__evaluation__")
        self.assertIn("MemoryError", str(summary["failure_cases"][0]["error"]))


if __name__ == "__main__":
    unittest.main()
