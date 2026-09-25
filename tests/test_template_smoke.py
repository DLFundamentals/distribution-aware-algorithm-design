from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import pytest

from dasbench.agents.template import run_template_synthesis_loop
from dasbench.artifacts import default_agent_run_dir, default_dataset_dir, default_report_dir
from dasbench.data import BenchmarkSpec, generate_dataset
from dasbench.eval.reporting import generate_benchmark_report


class TemplateSmokeTests(unittest.TestCase):
    def _dataset_spec(self, problem: str, family: str) -> BenchmarkSpec:
        instance_params = {"num_vertices": 12}
        if problem == "maxsat":
            instance_params = {"num_variables": 12, "num_clauses": 22}
        elif problem == "tsp":
            instance_params = {"num_cities": 10}
        elif problem in {"packing_lp", "mdkp"}:
            instance_params = {"num_items": 10, "num_resources": 3}
        return BenchmarkSpec(
            problem=problem,
            family=family,
            instance_params=instance_params,
            split_sizes={"train": 4, "validation": 2, "test": 2},
        )

    # Synthesizes and reports on all seven problem classes, and the report runs
    # the Gurobi and exact baselines at their full time limits. Roughly 75 minutes,
    # which is the whole suite's runtime; everything else finishes in about one.
    @pytest.mark.slow
    @pytest.mark.gurobi
    def test_template_synthesis_smoke_runs_for_each_problem(self) -> None:
        cases = [
            ("coloring", "cluster_ring_mix_v1"),
            ("maxsat", "last_clause_signal_v1"),
            ("mis", "clique_path_mix_v1"),
            ("mds", "star_cluster_cover_v1"),
            ("packing_lp", "single_bottleneck_fractional_v1"),
            ("mdkp", "single_resource_density_v1"),
            ("tsp", "clustered_euclidean_v1"),
        ]
        for problem, family in cases:
            with self.subTest(problem=problem, family=family):
                root = Path(tempfile.mkdtemp(prefix=f"dasbench-smoke-{problem}-"))
                dataset_dir = root / "dataset"
                run_dir = root / "run"
                report_dir = root / "report"
                generate_dataset(dataset_dir, self._dataset_spec(problem, family))
                summary = run_template_synthesis_loop(
                    dataset_dir,
                    run_dir,
                    mode="single",
                    iterations=1,
                    beam_width=1,
                )
                self.assertEqual(summary["problem"], problem)
                self.assertIn("best_candidate", summary)
                self.assertIn("test", summary["best_candidate"])
                report = generate_benchmark_report(
                    dataset_dir=dataset_dir,
                    agent_run_dir=run_dir,
                    output_dir=report_dir,
                    repeats=1,
                    include_train=False,
                )
                self.assertTrue(Path(report["json_path"]).exists())
                self.assertTrue(Path(report["markdown_path"]).exists())
                payload = json.loads(Path(report["json_path"]).read_text(encoding="utf-8"))
                self.assertIn("gurobi_timed", payload["split_reports"]["validation"])
                self.assertIn("gurobi_timed", payload["split_reports"]["test"])
                self.assertIn("hidden_rule_analysis", payload)
                self.assertIn("ground_truth_hidden_rule", payload["hidden_rule_analysis"])
                markdown = Path(report["markdown_path"]).read_text(encoding="utf-8")
                self.assertIn("Hidden Rule Analysis", markdown)

    def test_artifact_helpers_match_required_layout(self) -> None:
        dataset_dir = default_dataset_dir("mis", "motif_bridge_mixture_v1", "dataset_demo")
        run_dir = default_agent_run_dir("mis", "motif_bridge_mixture_v1", "run_demo")
        report_dir = default_report_dir("mis", "motif_bridge_mixture_v1", "run_demo")
        self.assertEqual(dataset_dir.as_posix(), "artifacts/datasets/mis/motif_bridge_mixture_v1/dataset_demo")
        self.assertEqual(run_dir.as_posix(), "artifacts/agent_runs/mis/motif_bridge_mixture_v1/run_demo")
        self.assertEqual(report_dir.as_posix(), "artifacts/reports/mis/motif_bridge_mixture_v1/run_demo")


if __name__ == "__main__":
    unittest.main()
