from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks.llm_pv_benchmark import LLMPVConfig, LLMPVJob, _result_from_summary
from dasbench.utils import write_json
from scripts.pace2025_run_llm_pv_baseline import (
    _baseline_rows_from_pace_evaluation,
    _canonical_instance_id,
    _require_exportable_llm_pv_solver,
)


class PaceLLMPVBaselineTests(unittest.TestCase):
    def test_canonical_instance_id_extracts_pace_private_name(self) -> None:
        self.assertEqual(
            _canonical_instance_id("pace2025-ds-test-private_heuristic_017"),
            "private_heuristic_017",
        )

    def test_baseline_rows_copy_solutions_and_join_reference_sizes(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="pace-llm-pv-baseline-"))
        source_solution = root / "pace_eval" / "solutions" / "pace2025-ds-test-private_heuristic_001.sol"
        source_solution.parent.mkdir(parents=True)
        source_solution.write_text("2\n1\n3\n", encoding="utf-8")
        output_dir = root / "baseline"

        rows = _baseline_rows_from_pace_evaluation(
            pace_rows=[
                {
                    "instance_id": "pace2025-ds-test-private_heuristic_001",
                    "pace_source_path": "private/ds/heuristic/private_heuristic_001.gr.tar.xz",
                    "num_vertices": "10",
                    "num_edges": "20",
                    "feasible": "True",
                    "solution_size": "2",
                    "reference_objective": "3",
                    "runtime_ms": "12.5",
                    "solution_file": str(source_solution),
                    "error": "",
                },
                {
                    "instance_id": "pace2025-ds-test-private_heuristic_002",
                    "pace_source_path": "private/ds/heuristic/private_heuristic_002.gr.tar.xz",
                    "num_vertices": "11",
                    "num_edges": "21",
                    "feasible": "False",
                    "solution_size": "0",
                    "reference_objective": "4",
                    "runtime_ms": "1.5",
                    "solution_file": "",
                    "error": "invalid",
                },
            ],
            reference_by_instance={
                "private_heuristic_001": {"solution_size": "5"},
                "private_heuristic_002": {"solution_size": "6"},
            },
            output_dir=output_dir,
            solver_name="llm_pv_custom",
        )

        self.assertEqual(rows[0]["solver"], "llm_pv_custom")
        self.assertEqual(rows[0]["instance_id"], "private_heuristic_001")
        self.assertEqual(rows[0]["valid"], True)
        self.assertEqual(rows[0]["valid_status"], "verified_valid")
        self.assertEqual(rows[0]["synth_solution_size"], "5")
        copied_solution = output_dir / "solutions" / "llm_pv_custom" / "private_heuristic_001.sol"
        self.assertEqual(rows[0]["solution_file"], str(copied_solution))
        self.assertEqual(copied_solution.read_text(encoding="utf-8"), "2\n1\n3\n")

        self.assertEqual(rows[1]["instance_id"], "private_heuristic_002")
        self.assertEqual(rows[1]["valid"], False)
        self.assertEqual(rows[1]["solution_size"], "")
        self.assertEqual(rows[1]["error"], "invalid")

    def test_direct_pace_dataset_result_does_not_require_source_sweep_layout(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="pace-llm-pv-result-"))
        job = LLMPVJob(
            sweep_id="pace",
            artifact_root=root / "llm_pv_artifacts",
            problem="mds",
            family="pace2025_private_heuristic_v1",
            source_dataset_dir=Path("artifacts/pace2025_dominating_set/pace_run/dataset"),
            force=False,
            config=LLMPVConfig(
                attempts=1,
                model="custom-model",
                reasoning_effort=None,
                max_output_tokens=None,
                api_timeout_seconds=10.0,
                enable_code_interpreter=False,
                tool_choice="auto",
                verbosity="low",
                early_stop_score=1.0,
                prompt_train_examples=1,
                prompt_json_char_limit=1000,
            ),
        )

        result = _result_from_summary(job, status="completed", returncode=0)

        self.assertIsNone(result["command"])

    def test_failed_llm_pv_summary_stops_before_pace_export(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="pace-llm-pv-failed-"))
        attempt_dir = root / "attempt_001"
        summary_path = root / "synthesis_summary.json"
        write_json(
            summary_path,
            {
                "best_candidate": {
                    "slug": "llm_pv_attempt_001",
                    "candidate_dir": str(attempt_dir),
                    "error": "InternalServerError: Error code: 524",
                }
            },
        )

        with self.assertRaisesRegex(RuntimeError, "524"):
            _require_exportable_llm_pv_solver(summary_path)

    def test_successful_llm_pv_summary_can_be_exported(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="pace-llm-pv-selected-"))
        attempt_dir = root / "attempt_001"
        attempt_dir.mkdir(parents=True)
        (attempt_dir / "solution.py").write_text("def solve(instance):\n    return []\n", encoding="utf-8")
        summary_path = root / "synthesis_summary.json"
        write_json(
            summary_path,
            {
                "best_candidate": {
                    "slug": "llm_pv_attempt_001",
                    "candidate_dir": str(attempt_dir),
                }
            },
        )

        _require_exportable_llm_pv_solver(summary_path)


if __name__ == "__main__":
    unittest.main()
