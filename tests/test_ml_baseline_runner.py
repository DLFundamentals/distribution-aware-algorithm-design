from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class MLBaselineRunnerTests(unittest.TestCase):
    def test_runner_trains_and_writes_outputs_for_explicit_splits(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="ml-runner-test-") as root:
            root_path = Path(root)
            split_dir = root_path / "splits"
            split_dir.mkdir()
            instance = {
                "id": "tiny-maxsat",
                "num_variables": 3,
                "clauses": [
                    [1, 2, 3],
                    [1, -2, 3],
                    [1, 2, -3],
                    [-1, -2, -3],
                ],
                "optimum_objective": 4.0,
            }
            for split in ("train", "validation", "test"):
                (split_dir / f"{split}.jsonl").write_text(json.dumps(instance) + "\n", encoding="utf-8")
            config_path = root_path / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "baselines": {
                            "ml_gnn_maxsat_assignment": {
                                "epochs": 1,
                                "hidden_dim": 8,
                                "layers": 1,
                                "samples": 1,
                                "walksat_flips": 4,
                            }
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root_path / "outputs"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "baselines.src.ml_baselines.run_ml_baselines",
                    "--problem",
                    "maxsat",
                    "--baseline",
                    "ml_gnn_maxsat_assignment",
                    "--train-split",
                    str(split_dir / "train.jsonl"),
                    "--validation-split",
                    str(split_dir / "validation.jsonl"),
                    "--test-split",
                    str(split_dir / "test.jsonl"),
                    "--output-dir",
                    str(output_dir),
                    "--run-id",
                    "smoke",
                    "--config",
                    str(config_path),
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Aggregate CSV:", completed.stdout)
            aggregate_csv = output_dir / "smoke" / "aggregate_results.csv"
            self.assertTrue(aggregate_csv.exists())
            with aggregate_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["baseline"], "ml_gnn_maxsat_assignment")
            self.assertEqual(rows[0]["split"], "test")
            self.assertEqual(rows[0]["num_test_instances"], "1")
            self.assertTrue(Path(rows[0]["checkpoint_path"]).exists())
            self.assertTrue(Path(rows[0]["test_outputs_path"]).exists())

    def test_runner_eval_only_loads_checkpoint_and_skips_training(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="ml-runner-eval-only-test-") as root:
            root_path = Path(root)
            split_dir = root_path / "splits"
            split_dir.mkdir()
            instance = {
                "id": "tiny-runcsp-maxsat",
                "num_variables": 3,
                "clauses": [
                    [1, 2, 3],
                    [1, -2, 3],
                    [1, 2, -3],
                    [-1, -2, -3],
                ],
                "optimum_objective": 4.0,
            }
            for split in ("train", "validation", "test"):
                (split_dir / f"{split}.jsonl").write_text(json.dumps(instance) + "\n", encoding="utf-8")
            config_path = root_path / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "baselines": {
                            "ml_runcsp_maxsat": {
                                "epochs": 1,
                                "hidden_dim": 8,
                                "message_passing_steps": 2,
                                "samples": 1,
                                "walksat_restarts": 1,
                                "walksat_flips": 4,
                            }
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            train_output_dir = root_path / "train_outputs"
            eval_output_dir = root_path / "eval_outputs"
            base_command = [
                sys.executable,
                "-m",
                "baselines.src.ml_baselines.run_ml_baselines",
                "--problem",
                "maxsat",
                "--baseline",
                "ml_runcsp_maxsat",
                "--train-split",
                str(split_dir / "train.jsonl"),
                "--validation-split",
                str(split_dir / "validation.jsonl"),
                "--test-split",
                str(split_dir / "test.jsonl"),
                "--config",
                str(config_path),
                "--device",
                "cpu",
            ]

            subprocess.run(
                [
                    *base_command,
                    "--output-dir",
                    str(train_output_dir),
                    "--run-id",
                    "source",
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    *base_command,
                    "--output-dir",
                    str(eval_output_dir),
                    "--run-id",
                    "eval",
                    "--eval-only-from",
                    str(train_output_dir / "source"),
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )

            aggregate_csv = eval_output_dir / "eval" / "aggregate_results.csv"
            self.assertTrue(aggregate_csv.exists())
            with aggregate_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["baseline"], "ml_runcsp_maxsat")
            self.assertEqual(rows[0]["total_training_time_ms"], "0.0")
            self.assertEqual(rows[0]["trial_count"], "0")
            self.assertTrue(Path(rows[0]["checkpoint_path"]).exists())
            run_dir = Path(rows[0]["run_dir"])
            self.assertTrue((run_dir / "eval_only_source.json").exists())
            self.assertTrue((run_dir / "test_outputs.jsonl").exists())

    def test_runner_treats_pace_as_mds_shaped_problem(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="ml-runner-pace-test-") as root:
            root_path = Path(root)
            split_dir = root_path / "pace_dataset"
            split_dir.mkdir()
            manifest = {
                "problem": "mds",
                "family": "pace2025_tiny",
                "instance_schema_version": "mds.v1",
                "metric_definition": {"primary": "lower_bound_ratio"},
                "split_sizes": {"train": 1, "validation": 1, "test": 1},
            }
            (split_dir / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            instance = {
                "id": "pace-tiny",
                "num_vertices": 4,
                "edges": [[0, 1], [0, 2], [0, 3]],
                "optimum_objective": 1.0,
                "optimum_source": "tiny",
                "_pace_reference_solution": [0],
                "pace_source_path": "tiny.gr",
            }
            for split in ("train", "validation", "test"):
                (split_dir / f"{split}.jsonl").write_text(json.dumps(instance) + "\n", encoding="utf-8")
            config_path = root_path / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "baselines": {
                            "ml_gnn_mds_score_repair": {
                                "epochs": 1,
                                "hidden_dim": 8,
                                "layers": 1,
                                "repair_budget": 4,
                            }
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root_path / "outputs"

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "baselines.src.ml_baselines.run_ml_baselines",
                    "--problem",
                    "pace",
                    "--baseline",
                    "ml_gnn_mds_score_repair",
                    "--dataset-dir",
                    str(split_dir),
                    "--output-dir",
                    str(output_dir),
                    "--run-id",
                    "pace-smoke",
                    "--config",
                    str(config_path),
                    "--device",
                    "cpu",
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )

            aggregate_csv = output_dir / "pace-smoke" / "aggregate_results.csv"
            self.assertTrue(aggregate_csv.exists())
            with aggregate_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["problem"], "pace")
            self.assertEqual(rows[0]["family"], "pace2025_tiny")
            self.assertEqual(rows[0]["baseline"], "ml_gnn_mds_score_repair")
            self.assertEqual(rows[0]["num_test_instances"], "1")
            self.assertTrue(Path(rows[0]["test_outputs_path"]).exists())


if __name__ == "__main__":
    unittest.main()
