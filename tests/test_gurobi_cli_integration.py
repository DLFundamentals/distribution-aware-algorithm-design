from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from dasbench.cli import cmd_report, cmd_run_agent
from dasbench.data import BenchmarkSpec, generate_dataset
import pytest

pytestmark = pytest.mark.gurobi


class GurobiCliIntegrationTests(unittest.TestCase):
    def _dataset_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="dasbench-gurobi-cli-")) / "dataset"

    def _build_dataset(self, dataset_dir: Path) -> None:
        generate_dataset(
            dataset_dir,
            BenchmarkSpec(
                problem="mis",
                family="clique_path_mix_v1",
                instance_params={"num_vertices": 10},
                split_sizes={"train": 3, "validation": 2, "test": 2},
            ),
        )

    def test_default_on_gurobi_is_written_to_run_and_report(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dasbench-gurobi-enabled-"))
        dataset_dir = root / "dataset"
        run_dir = root / "run"
        report_dir = root / "report"
        self._build_dataset(dataset_dir)

        cmd_run_agent(
            argparse.Namespace(
                dataset_dir=str(dataset_dir),
                run_id="run_enabled",
                output_dir=str(run_dir),
                generator="template",
                mode="single",
                iterations=1,
                beam_width=1,
                gurobi_baseline_enabled=True,
                gurobi_time_limit_seconds=7.0,
                gurobi_threads=1,
            )
        )

        baseline_validation = json.loads((run_dir / "baseline_validation.json").read_text(encoding="utf-8"))
        self.assertIn("gurobi_timed", baseline_validation)
        run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(run_manifest["gurobi_baseline"]["time_limit_seconds"], 7.0)
        self.assertEqual(run_manifest["gurobi_baseline"]["threads"], 1)

        cmd_report(
            argparse.Namespace(
                dataset_dir=str(dataset_dir),
                agent_run_dir=str(run_dir),
                output_dir=str(report_dir),
                repeats=1,
                include_train=False,
                gurobi_baseline_enabled=None,
                gurobi_time_limit_seconds=None,
                gurobi_threads=None,
            )
        )

        report_payload = json.loads((report_dir / "benchmark_report.json").read_text(encoding="utf-8"))
        self.assertIn("gurobi_timed", report_payload["split_reports"]["validation"])
        self.assertEqual(report_payload["gurobi_baseline"]["time_limit_seconds"], 7.0)
        self.assertEqual(report_payload["gurobi_baseline"]["threads"], 1)
        self.assertTrue((report_dir / "gurobi_timed_validation_diagnostics.jsonl").exists())
        markdown = (report_dir / "benchmark_report.md").read_text(encoding="utf-8")
        self.assertIn("Industrial Baseline", markdown)

    def test_no_gurobi_baseline_disables_it_in_run_and_report(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dasbench-gurobi-disabled-"))
        dataset_dir = root / "dataset"
        run_dir = root / "run"
        report_dir = root / "report"
        self._build_dataset(dataset_dir)

        cmd_run_agent(
            argparse.Namespace(
                dataset_dir=str(dataset_dir),
                run_id="run_disabled",
                output_dir=str(run_dir),
                generator="template",
                mode="single",
                iterations=1,
                beam_width=1,
                gurobi_baseline_enabled=False,
                gurobi_time_limit_seconds=7.0,
                gurobi_threads=1,
            )
        )

        baseline_validation = json.loads((run_dir / "baseline_validation.json").read_text(encoding="utf-8"))
        self.assertNotIn("gurobi_timed", baseline_validation)
        run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(run_manifest["gurobi_baseline"]["enabled"])

        cmd_report(
            argparse.Namespace(
                dataset_dir=str(dataset_dir),
                agent_run_dir=str(run_dir),
                output_dir=str(report_dir),
                repeats=1,
                include_train=False,
                gurobi_baseline_enabled=None,
                gurobi_time_limit_seconds=None,
                gurobi_threads=None,
            )
        )

        report_payload = json.loads((report_dir / "benchmark_report.json").read_text(encoding="utf-8"))
        self.assertNotIn("gurobi_timed", report_payload["split_reports"]["validation"])
        self.assertFalse(report_payload["gurobi_baseline"]["enabled"])
        markdown = (report_dir / "benchmark_report.md").read_text(encoding="utf-8")
        self.assertNotIn("Industrial Baseline", markdown)


if __name__ == "__main__":
    unittest.main()
