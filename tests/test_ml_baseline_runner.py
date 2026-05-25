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


if __name__ == "__main__":
    unittest.main()
