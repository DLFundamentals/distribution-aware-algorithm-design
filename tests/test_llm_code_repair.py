from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dasbench.agents.llm import (
    GenerationDebugError,
    _evaluate_solution_with_semantic_repairs,
    _summary_needs_solution_repair,
    _generate_validated_stage_output,
)


class LLMCodeRepairTests(unittest.TestCase):
    def test_invalid_python_is_repaired_before_saving(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_dir = Path(tmpdir)
            calls = [
                (
                    'def solve(instance, analysis=None, manifest=None):\n    """unterminated\n',
                    "bad notes",
                    {"status_code": 200},
                ),
                (
                    "def solve(instance, analysis=None, manifest=None):\n    return []\n",
                    "fixed notes",
                    {"status_code": 200},
                ),
            ]

            def fake_generate(**kwargs):
                return calls.pop(0)

            with patch("dasbench.agents.llm._generate_stage_output", side_effect=fake_generate):
                code, notes, metadata = _generate_validated_stage_output(
                    messages=[
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": '{"task": "write solution"}'},
                    ],
                    candidate_dir=candidate_dir,
                    stage_name="solution",
                    response_schema_path=Path("solution_schema.json"),
                    code_field="solution_py",
                    filename="solution.py",
                    required_functions=("solve", "build_solver"),
                )

            self.assertIn("def solve", code)
            self.assertEqual(notes, "fixed notes")
            self.assertEqual(metadata["code_validation"]["status"], "passed")
            self.assertEqual(metadata["code_validation"]["repair_attempts_used"], 1)
            self.assertTrue((candidate_dir / "solution_invalid_attempt_00.py").exists())
            self.assertTrue((candidate_dir / "solution_invalid_attempt_00.txt").exists())

    def test_repair_limit_failure_raises_debug_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_dir = Path(tmpdir)

            def fake_generate(**kwargs):
                return (
                    "def not_the_required_interface():\n    return None\n",
                    "bad notes",
                    {"status_code": 200},
                )

            with patch.dict("os.environ", {"DASBENCH_CODE_REPAIR_LIMIT": "1"}), patch(
                "dasbench.agents.llm._generate_stage_output",
                side_effect=fake_generate,
            ):
                with self.assertRaises(GenerationDebugError) as raised:
                    _generate_validated_stage_output(
                        messages=[
                            {"role": "system", "content": "system"},
                            {"role": "user", "content": '{"task": "write analyze"}'},
                        ],
                        candidate_dir=candidate_dir,
                        stage_name="analyze",
                        response_schema_path=Path("analyze_schema.json"),
                        code_field="analyze_py",
                        filename="analyze.py",
                        required_functions=("analyze",),
                    )

            self.assertIn("failed validation", str(raised.exception))
            self.assertTrue((candidate_dir / "analyze_invalid_attempt_00.py").exists())
            self.assertTrue((candidate_dir / "analyze_invalid_attempt_01.py").exists())

    def test_solution_semantic_failure_is_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate_dir = root / "candidate"
            evaluation_dir = root / "evaluation"
            candidate_dir.mkdir()
            evaluation_dir.mkdir()
            bad_code = "def solve(instance, analysis=None, manifest=None):\n    return [0, 1]\n"
            fixed_code = "def solve(instance, analysis=None, manifest=None):\n    return [0, 2]\n"
            (candidate_dir / "solution.py").write_text(bad_code, encoding="utf-8")
            train_instances = [
                {
                    "id": "train-0",
                    "num_vertices": 3,
                    "edges": [[0, 1]],
                    "optimum_objective": 2,
                }
            ]
            timing: dict[str, float] = {}

            with patch.dict("os.environ", {"DASBENCH_SOLUTION_REPAIR_LIMIT": "1"}), patch(
                "dasbench.agents.llm._generate_validated_stage_output",
                return_value=(fixed_code, "fixed", {"status_code": 200}),
            ):
                solution_py, notes, train_eval, validation_eval = _evaluate_solution_with_semantic_repairs(
                    problem_name="mis",
                    plan_slug="candidate",
                    candidate_dir=candidate_dir,
                    evaluation_dir=evaluation_dir,
                    manifest={
                        "problem": "mis",
                        "metric_definition": {},
                        "instance_schema_version": "mis.v1",
                        "instance_params": {"num_vertices": 3},
                    },
                    train_instances_full=train_instances,
                    validation_instances_full=train_instances,
                    analyze_py="def analyze(train_instances, manifest=None):\n    return {}\n",
                    analysis_output={},
                    solution_py=bad_code,
                    solution_notes="bad",
                    solution_messages=[
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": '{"task": "write solution"}'},
                    ],
                    timing=timing,
                )

            self.assertEqual(solution_py, fixed_code)
            self.assertEqual(notes, "fixed")
            self.assertTrue(_summary_needs_solution_repair({"feasibility_rate": 0.0}))
            self.assertEqual(train_eval["feasibility_rate"], 1.0)
            self.assertEqual(validation_eval["average_normalized_quality"], 1.0)
            self.assertEqual(timing["solution_semantic_repairs_used"], 1.0)
            self.assertTrue((candidate_dir / "solution_failed_eval_attempt_00.py").exists())
            self.assertEqual((candidate_dir / "solution.py").read_text(encoding="utf-8"), fixed_code)


if __name__ == "__main__":
    unittest.main()
