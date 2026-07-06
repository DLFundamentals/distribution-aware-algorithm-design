from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.candidate_count_sweep import build_jobs as build_candidate_jobs
from benchmarks.candidate_count_sweep import main as candidate_main
from benchmarks.common import SweepJob, aggregate_rows, benchmark_command, run_job, run_sweep
from benchmarks.llm_pv_benchmark import build_jobs as build_llm_pv_jobs
from benchmarks.llm_pv_benchmark import _build_prompt_messages as build_llm_pv_prompt_messages
from benchmarks.llm_pv_benchmark import _messages_for_plain_python_solution, extract_solution_code
from benchmarks.llm_pv_benchmark import main as llm_pv_main
from benchmarks.problem_size_sweep import build_jobs as build_size_jobs
from benchmarks.problem_size_sweep import main as size_main
from benchmarks.sample_size_sweep import build_jobs as build_sample_jobs
from benchmarks.sample_size_sweep import main as sample_main


class BenchmarkSweepTests(unittest.TestCase):
    def test_sample_sweep_command_uses_deterministic_ids_and_sizes(self) -> None:
        parser_args = [
            "--dry-run",
            "--problem",
            "maxsat",
            "--family",
            "last_clause_signal_v1",
            "--train-sizes",
            "4,16",
            "--output-root",
            tempfile.mkdtemp(prefix="dasbench-sample-sweep-"),
        ]
        from benchmarks.sample_size_sweep import build_parser

        args = build_parser().parse_args(parser_args)
        jobs = build_sample_jobs(args)
        self.assertEqual([job.condition_id for job in jobs], ["sample_train4_val4", "sample_train16_val8"])
        command = benchmark_command(jobs[1])
        self.assertIn("--train-size", command)
        self.assertIn("16", command)
        self.assertIn("--validation-size", command)
        self.assertIn("8", command)

    def test_problem_size_sweep_applies_problem_specific_instance_params(self) -> None:
        from benchmarks.problem_size_sweep import build_parser

        args = build_parser().parse_args(
            [
                "--dry-run",
                "--problem",
                "maxsat",
                "--family",
                "last_clause_signal_v1",
                "--size-labels",
                "tiny",
                "--output-root",
                tempfile.mkdtemp(prefix="dasbench-size-sweep-"),
            ]
        )
        jobs = build_size_jobs(args)
        self.assertEqual(jobs[0].condition_id, "size_tiny")
        command = benchmark_command(jobs[0])
        self.assertIn("num_variables=10", command)
        self.assertIn("num_clauses=18", command)

    def test_candidate_count_sweep_separates_candidate_and_beam_width(self) -> None:
        from benchmarks.candidate_count_sweep import build_parser

        args = build_parser().parse_args(
            [
                "--dry-run",
                "--problem",
                "mds",
                "--family",
                "star_cluster_cover_v1",
                "--candidate-widths",
                "1,5",
                "--beam-width",
                "3",
                "--output-root",
                tempfile.mkdtemp(prefix="dasbench-candidate-sweep-"),
            ]
        )
        jobs = build_candidate_jobs(args)
        self.assertEqual([job.condition_id for job in jobs], ["candidates_gen1_beam1_iter3", "candidates_gen5_beam3_iter3"])
        self.assertEqual(jobs[0].beam_width, 1)
        self.assertEqual(jobs[1].beam_width, 3)
        self.assertEqual(jobs[1].candidate_width, 5)
        command = benchmark_command(jobs[1])
        self.assertIn("--candidate-width", command)
        self.assertIn("5", command)

    def test_llm_pv_defaults_to_representative_targets(self) -> None:
        from benchmarks.llm_pv_benchmark import build_parser

        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "openai", "OPENAI_MODEL": "gpt-5-env", "OPENAI_REASONING_EFFORT": "medium"},
        ):
            args = build_parser().parse_args(
                [
                    "--dry-run",
                    "--problem",
                    "tsp",
                    "--output-root",
                    tempfile.mkdtemp(prefix="dasbench-llm-pv-"),
                    "--sweep-id",
                    "llm-pv",
                ]
            )
        jobs = build_llm_pv_jobs(args)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].problem, "tsp")
        self.assertEqual(jobs[0].family, "clustered_euclidean_v1")
        self.assertEqual(jobs[0].config.attempts, 5)
        self.assertEqual(jobs[0].config.model, "gpt-5-env")
        self.assertEqual(jobs[0].config.reasoning_effort, "medium")
        self.assertIsNone(jobs[0].config.max_output_tokens)
        self.assertEqual(jobs[0].config.api_timeout_seconds, 14400)

    def test_llm_pv_rejects_truncated_json_as_raw_python(self) -> None:
        self.assertIsNone(extract_solution_code('{"solution_py": "def solve(instance):\\n    return ['))

    def test_llm_pv_prompt_hides_family_and_includes_output_contract(self) -> None:
        from benchmarks.llm_pv_benchmark import build_parser

        args = build_parser().parse_args(["--dry-run", "--problem", "tsp"])
        config = build_llm_pv_jobs(args, sweep_id="prompt")[0].config
        messages = build_llm_pv_prompt_messages(
            manifest={
                "problem": "tsp",
                "family": "clustered_euclidean_v1",
                "metric_definition": {"primary": "normalized_quality"},
                "instance_schema_version": "tsp.v1",
                "instance_params": {"num_cities": 64},
            },
            train_summary={"family": "clustered_euclidean_v1", "num_cities": 64},
            train_public=[],
            attempt_index=1,
            config=config,
        )
        prompt = json.loads(messages[1]["content"])
        self.assertNotIn("family", prompt["train_summary"])
        self.assertEqual(prompt["solution_contract"]["return_type"], "list[int]")
        self.assertIn("city ids", prompt["solution_contract"]["required_shape"])
        self.assertIn("dict", " ".join(prompt["solution_contract"]["do_not_return"]))

    def test_llm_pv_prompt_includes_pace_problem_contracts(self) -> None:
        from benchmarks.llm_pv_benchmark import build_parser

        args = build_parser().parse_args(["--dry-run", "--problem", "tsp"])
        config = build_llm_pv_jobs(args, sweep_id="pace-prompt")[0].config
        cases = {
            "hitting_set": ["instance['sets']", "hits every set"],
            "ocm": ["instance['edges']", "permutation"],
            "dfvs": ["instance['arcs']", "instance['edges'] is absent"],
        }
        for problem, expected_fragments in cases.items():
            with self.subTest(problem=problem):
                messages = build_llm_pv_prompt_messages(
                    manifest={
                        "problem": problem,
                        "family": f"{problem}_pace",
                        "metric_definition": {"primary": "normalized_quality"},
                        "instance_schema_version": f"{problem}.v1",
                    },
                    train_summary={"family": f"{problem}_pace"},
                    train_public=[],
                    attempt_index=1,
                    config=config,
                )
                prompt = json.loads(messages[1]["content"])
                contract_text = json.dumps(prompt["solution_contract"])
                for fragment in expected_fragments:
                    self.assertIn(fragment, contract_text)

    def test_llm_pv_plain_python_prompt_drops_json_wrapper_requirement(self) -> None:
        from benchmarks.llm_pv_benchmark import build_parser

        args = build_parser().parse_args(["--dry-run", "--problem", "tsp"])
        config = build_llm_pv_jobs(args, sweep_id="plain-python-prompt")[0].config
        messages = build_llm_pv_prompt_messages(
            manifest={
                "problem": "tsp",
                "family": "clustered_euclidean_v1",
                "metric_definition": {"primary": "normalized_quality"},
                "instance_schema_version": "tsp.v1",
                "instance_params": {"num_cities": 64},
            },
            train_summary={"family": "clustered_euclidean_v1", "num_cities": 64},
            train_public=[],
            attempt_index=1,
            config=config,
        )

        prompt = json.loads(_messages_for_plain_python_solution(messages)[1]["content"])

        self.assertIn("raw contents of solution.py", prompt["constraints"][0])
        self.assertNotIn("solution_py", prompt["constraints"][0])
        self.assertEqual(prompt["response_format"], "raw Python module text defining solve(...) or build_solver(...)")

    def test_completed_report_is_skipped_without_force(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dasbench-sweep-resume-"))
        job = SweepJob(
            sweep_id="resume",
            condition_id="sample_train4_val4",
            problem="maxsat",
            family="last_clause_signal_v1",
            train_size=4,
            validation_size=4,
            test_size=8,
        )
        with patch("benchmarks.common.default_report_dir", lambda problem, family, run_id: root / problem / family / run_id):
            job.report_json_path.parent.mkdir(parents=True, exist_ok=True)
            job.report_json_path.write_text("{}\n", encoding="utf-8")
            result = run_job(job, output_dir=root / "out", dry_run=False)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["returncode"], 0)

    def test_second_scale_exposes_skip_and_source_dataset_flags(self) -> None:
        from benchmarks.second_scale_benchmark_v2 import build_jobs as build_second_scale_jobs
        from benchmarks.second_scale_benchmark_v2 import build_parser as build_second_scale_parser

        source_root = Path("artifacts/second_scale_benchmark_v2/20260427_230552")
        args = build_second_scale_parser().parse_args(
            [
                "--dry-run",
                "--sweep-id",
                "glm47",
                "--problem",
                "tsp",
                "--family",
                "clustered_euclidean_v1",
                "--source-run-root",
                str(source_root),
                "--env-file",
                ".env-glm47",
                "--skip-baselines",
                "--skip-report",
                "--no-compute-optima",
                "--no-gurobi-baseline",
                "--external-exact-baselines",
                "off",
            ]
        )
        job = build_second_scale_jobs(args, sweep_id="glm47")[0]
        command = benchmark_command(job)
        self.assertEqual(
            job.source_dataset_dir,
            source_root / "targets" / "seconds_scale_v2" / "tsp" / "clustered_euclidean_v1" / "dataset",
        )
        self.assertEqual(job.env_file, Path(".env-glm47"))
        self.assertIn("--skip-baselines", command)
        self.assertIn("--skip-report", command)
        self.assertIn("--no-compute-optima", command)
        self.assertIn("--no-gurobi-baseline", command)
        self.assertNotIn("--force-regenerate", command)

    def test_run_job_materializes_reused_source_dataset(self) -> None:
        import subprocess

        root = Path(tempfile.mkdtemp(prefix="dasbench-reuse-dataset-"))
        source_dir = root / "source" / "targets" / "seconds_scale_v2" / "maxsat" / "last_clause_signal_v1" / "dataset"
        source_dir.mkdir(parents=True)
        manifest = {
            "problem": "maxsat",
            "family": "last_clause_signal_v1",
            "split_sizes": {"train": 4, "validation": 4, "test": 8},
            "artifact_paths": {"dataset_dir": str(source_dir)},
        }
        (source_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (source_dir / "benchmark_spec.json").write_text(json.dumps({"source": True}), encoding="utf-8")
        (source_dir / "reproducibility.json").write_text(json.dumps({"source": True}), encoding="utf-8")
        for split_name in ("train", "validation", "test"):
            (source_dir / f"{split_name}.jsonl").write_text("{}\n", encoding="utf-8")

        job = SweepJob(
            sweep_id="reuse",
            condition_id="seconds_scale_v2",
            problem="maxsat",
            family="last_clause_signal_v1",
            train_size=4,
            validation_size=4,
            test_size=8,
            artifact_root=root / "run",
            source_dataset_dir=source_dir,
            skip_baselines=True,
            skip_report=True,
        )

        def fake_run(*_args, **_kwargs):
            job.synthesis_summary_path.parent.mkdir(parents=True, exist_ok=True)
            job.synthesis_summary_path.write_text(
                json.dumps({"best_candidate": {"slug": "fake", "train": {}, "validation": {}, "test": {}}}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=[], returncode=0)

        with patch("benchmarks.common.subprocess.run", side_effect=fake_run):
            result = run_job(job, output_dir=root / "out", dry_run=False)

        reused_manifest = json.loads((job.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(reused_manifest["reused_dataset_source"], str(source_dir))
        self.assertEqual(reused_manifest["artifact_paths"]["dataset_dir"], str(job.dataset_dir))
        self.assertEqual((job.dataset_dir / "benchmark_spec.json").read_text(encoding="utf-8"), json.dumps({"source": True}))

    def test_run_sweep_env_file_overrides_process_and_dotenv_values(self) -> None:
        import subprocess

        root = Path(tempfile.mkdtemp(prefix="dasbench-env-file-"))
        env_file = root / ".env-glm47"
        env_file.write_text(
            "\n".join(
                [
                    "LLM_PROVIDER=custom_chat",
                    "CUSTOM_CHAT_API_BASE_URL=http://localhost:8001/v1",
                    "CUSTOM_CHAT_API_KEY=local-vllm",
                    "CUSTOM_CHAT_MODEL=glm-4.7-fp8",
                    "CUSTOM_CHAT_TIMEOUT_SECONDS=14400",
                    "CUSTOM_CHAT_MAX_TOKENS=8192",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        job = SweepJob(
            sweep_id="env-file",
            condition_id="seconds_scale_v2",
            problem="tsp",
            family="clustered_euclidean_v1",
            train_size=4,
            validation_size=4,
            test_size=8,
            artifact_root=root / "artifact_root",
            env_file=env_file,
            skip_report=True,
        )
        captured_env: dict[str, str] = {}

        def fake_run(*_args, **kwargs):
            captured_env.update(kwargs["env"])
            job.synthesis_summary_path.parent.mkdir(parents=True, exist_ok=True)
            job.synthesis_summary_path.write_text(
                json.dumps({"best_candidate": {"slug": "fake", "train": {}, "validation": {}, "test": {}}}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=[], returncode=0)

        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "custom_chat",
                "CUSTOM_CHAT_API_BASE_URL": "https://wrong.example/api",
                "CUSTOM_CHAT_API_KEY": "wrong-key",
                "CUSTOM_CHAT_MODEL": "wrong-model",
            },
            clear=False,
        ), patch("benchmarks.common.subprocess.run", side_effect=fake_run):
            summary = run_sweep(
                sweep_id="env-file",
                sweep_kind="second_scale_benchmark_v2",
                jobs=[job],
                output_root=root,
                max_workers=1,
                dry_run=False,
            )

        self.assertEqual(summary["failed_count"], 0)
        self.assertEqual(captured_env["CUSTOM_CHAT_API_BASE_URL"], "http://localhost:8001/v1")
        self.assertEqual(captured_env["CUSTOM_CHAT_API_KEY"], "local-vllm")
        self.assertEqual(captured_env["CUSTOM_CHAT_MODEL"], "glm-4.7-fp8")
        self.assertEqual(captured_env["CUSTOM_CHAT_MAX_TOKENS"], "8192")

    def test_aggregate_rows_extracts_agent_baseline_and_search_metrics(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dasbench-sweep-aggregate-"))
        job = SweepJob(
            sweep_id="aggregate",
            condition_id="sample_train4_val4",
            problem="maxsat",
            family="last_clause_signal_v1",
            train_size=4,
            validation_size=4,
            test_size=8,
            candidate_width=5,
        )
        with patch("benchmarks.common.default_report_dir", lambda problem, family, run_id: root / "reports" / problem / family / run_id), patch(
            "benchmarks.common.default_agent_run_dir", lambda problem, family, run_id: root / "runs" / problem / family / run_id
        ):
            run_dir = job.agent_run_dir
            report_path = job.report_json_path
            run_dir.mkdir(parents=True, exist_ok=True)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            (run_dir / "synthesis_summary.json").write_text(
                json.dumps(
                    {
                        "rounds": [
                            {
                                "evaluated_this_round": ["a", "b", "c", "d", "e"],
                                "frontier_diversity_keys": ["rule-a", "rule-b"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report_path.write_text(
                json.dumps(
                    {
                        "agent_run_dir": str(run_dir),
                        "best_candidate": {
                            "slug": "agent",
                            "hypothesis": {
                                "title": "Rule",
                                "diversity_key": "rule",
                                "rule_summary": "Hidden rule.",
                            },
                            "validation": {
                                "average_normalized_quality": 0.9,
                                "optimality_rate": 0.5,
                                "feasibility_rate": 1.0,
                                "average_runtime_ms": 2.0,
                            },
                            "test": {
                                "average_normalized_quality": 0.8,
                                "optimality_rate": 0.25,
                                "feasibility_rate": 1.0,
                                "average_runtime_ms": 3.0,
                            },
                        },
                        "split_reports": {
                            "test": {
                                "agent": {"average_normalized_quality_mean": 0.8, "average_runtime_ms_mean": 3.0},
                                "gurobi_timed": {
                                    "average_normalized_quality_mean": 1.0,
                                    "average_runtime_ms_mean": 50.0,
                                    "average_gurobi_runtime_ms_mean": 25.0,
                                },
                                "heuristic": {"average_normalized_quality_mean": 0.7, "average_runtime_ms_mean": 1.0},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = {
                "condition_id": job.condition_id,
                "problem": job.problem,
                "family": job.family,
                "status": "skipped",
                "returncode": 0,
            }
            rows = aggregate_rows([job], [result])
        self.assertEqual(rows[0]["agent_test_quality"], 0.8)
        self.assertEqual(rows[0]["gurobi_test_internal_runtime_ms"], 25.0)
        self.assertEqual(rows[0]["best_baseline_name"], "gurobi_timed")
        self.assertEqual(rows[0]["evaluated_candidate_count"], 5)

    def test_sweep_entrypoints_support_dry_run(self) -> None:
        root = tempfile.mkdtemp(prefix="dasbench-sweep-dry-run-")
        self.assertEqual(
            sample_main([
                "--dry-run",
                "--problem",
                "maxsat",
                "--family",
                "last_clause_signal_v1",
                "--train-sizes",
                "4",
                "--max-workers",
                "1",
                "--output-root",
                root,
                "--sweep-id",
                "sample",
            ]),
            0,
        )
        self.assertEqual(
            size_main([
                "--dry-run",
                "--problem",
                "mis",
                "--family",
                "clique_path_mix_v1",
                "--size-labels",
                "tiny",
                "--max-workers",
                "1",
                "--output-root",
                root,
                "--sweep-id",
                "size",
            ]),
            0,
        )
        self.assertEqual(
            candidate_main([
                "--dry-run",
                "--problem",
                "mds",
                "--family",
                "star_cluster_cover_v1",
                "--candidate-widths",
                "1",
                "--max-workers",
                "1",
                "--output-root",
                root,
                "--sweep-id",
                "candidate",
            ]),
            0,
        )
        self.assertEqual(
            llm_pv_main([
                "--dry-run",
                "--problem",
                "tsp",
                "--max-workers",
                "1",
                "--output-root",
                root,
                "--sweep-id",
                "llm_pv",
            ]),
            0,
        )
        self.assertTrue((Path(root) / "sample" / "aggregate_results.json").exists())
        self.assertTrue((Path(root) / "size" / "aggregate_results.csv").exists())
        self.assertTrue((Path(root) / "candidate" / "benchmark_sweep_summary.json").exists())
        self.assertTrue((Path(root) / "llm_pv_benchmark" / "llm_pv" / "benchmark_sweep_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
