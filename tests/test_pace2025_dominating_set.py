from __future__ import annotations

import csv
from pathlib import Path

from benchmarks.pace2025_dominating_set import (
    SourceConfig,
    build_pace_dataset,
    domination_lower_bound,
    export_pace_evaluation,
    parse_pace_gr_text,
)
from dasbench.data import load_split
from dasbench.utils import load_json, public_instance, write_json, write_jsonl


def test_parse_pace_gr_text_converts_one_based_edges() -> None:
    text = """
c tiny graph
p ds 4 3
1 2
2 3
4 3
"""
    instance = parse_pace_gr_text(text, instance_id="tiny", source_path="tiny.gr")

    assert instance["num_vertices"] == 4
    assert instance["edges"] == [[0, 1], [1, 2], [2, 3]]
    assert instance["pace_declared_edges"] == 3


def test_build_pace_dataset_uses_private_proxy_fields(tmp_path: Path) -> None:
    pace_root = tmp_path / "pace"
    (pace_root / "ds" / "exact").mkdir(parents=True)
    (pace_root / "private" / "ds" / "exact").mkdir(parents=True)
    graph_one = "p ds 4 3\n1 2\n2 3\n3 4\n"
    graph_two = "p ds 5 4\n1 2\n1 3\n1 4\n1 5\n"
    graph_private = "p ds 3 2\n1 2\n2 3\n"
    (pace_root / "ds" / "exact" / "exact_001.gr").write_text(graph_one, encoding="utf-8")
    (pace_root / "ds" / "exact" / "exact_002.gr").write_text(graph_two, encoding="utf-8")
    (pace_root / "private" / "ds" / "exact" / "private_exact_001.gr").write_text(
        graph_private,
        encoding="utf-8",
    )

    dataset_dir = tmp_path / "dataset"
    build_pace_dataset(
        dataset_dir=dataset_dir,
        output_root=tmp_path / "out",
        source_config=SourceConfig(pace_root=pace_root, cache_dir=tmp_path / "cache", github_ref="master"),
        track="exact",
        test_source="private",
        train_count=1,
        validation_count=1,
        test_count=1,
        public_start_index=1,
        test_start_index=1,
        reference_baselines=["marginal_gain_greedy"],
    )

    manifest = load_json(dataset_dir / "manifest.json")
    train = load_split(dataset_dir, "train")
    public_train = public_instance(train[0])

    assert manifest["problem"] == "mds"
    assert manifest["split_sizes"] == {"train": 1, "validation": 1, "test": 1}
    assert train[0]["optimum_objective"] == domination_lower_bound(train[0])
    assert "_pace_reference_solution" in train[0]
    assert "_pace_reference_solution" not in public_train
    assert "optimum_objective" not in public_train


def test_export_pace_evaluation_records_missing_selected_solver(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    instance = {
        "id": "pace-tiny-001",
        "num_vertices": 3,
        "edges": [[0, 1], [1, 2]],
        "optimum_objective": 1,
        "pace_source_path": "private/ds/heuristic/private_heuristic_001.gr.tar.xz",
        "_pace_reference_objective": 1,
    }
    write_json(
        dataset_dir / "manifest.json",
        {
            "problem": "mds",
            "family": "pace2025_ds_heuristic_private",
            "metric_definition": {
                "primary": "normalized_quality",
                "secondary": "optimality_rate",
                "tertiary": "average_runtime_ms",
            },
            "instance_schema_version": "mds.v1",
            "instance_params": {},
            "split_sizes": {"train": 1, "validation": 0, "test": 1},
        },
    )
    write_jsonl(dataset_dir / "train.jsonl", [instance])
    write_jsonl(dataset_dir / "test.jsonl", [instance])

    agent_run_dir = tmp_path / "agent_run"
    candidate_dir = agent_run_dir / "candidates" / "llm_iter00_slot00"
    candidate_dir.mkdir(parents=True)
    write_json(
        agent_run_dir / "synthesis_summary.json",
        {
            "best_candidate": {
                "slug": "llm_iter00_slot00",
                "candidate_dir": str(candidate_dir),
                "train": {"error": "GenerationDebugError: malformed JSON"},
            }
        },
    )

    output_dir = tmp_path / "pace_evaluation"
    summary = export_pace_evaluation(
        dataset_dir=dataset_dir,
        agent_run_dir=agent_run_dir,
        output_dir=output_dir,
    )

    assert summary["feasible_count"] == 0
    assert summary["invalid_count"] == 1
    assert "solution.py" in str(summary["error"])
    assert "malformed JSON" in str(summary["error"])
    rows = list(csv.DictReader((output_dir / "pace_private_results.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["feasible"] == "False"
    assert "solution.py" in rows[0]["error"]


def test_export_pace_evaluation_enforces_solver_timeout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DASBENCH_SOLVER_TIMEOUT_SECONDS", "1")
    dataset_dir = tmp_path / "dataset"
    instance = {
        "id": "pace-timeout-001",
        "num_vertices": 3,
        "edges": [[0, 1], [1, 2]],
        "optimum_objective": 1,
        "pace_source_path": "private/ds/heuristic/private_heuristic_001.gr.tar.xz",
        "_pace_reference_objective": 1,
    }
    write_json(
        dataset_dir / "manifest.json",
        {
            "problem": "mds",
            "family": "pace2025_ds_heuristic_private",
            "metric_definition": {
                "primary": "normalized_quality",
                "secondary": "optimality_rate",
                "tertiary": "average_runtime_ms",
            },
            "instance_schema_version": "mds.v1",
            "instance_params": {},
            "split_sizes": {"train": 1, "validation": 0, "test": 1},
        },
    )
    write_jsonl(dataset_dir / "train.jsonl", [instance])
    write_jsonl(dataset_dir / "test.jsonl", [instance])

    agent_run_dir = tmp_path / "agent_run"
    candidate_dir = agent_run_dir / "candidates" / "llm_iter00_slot00"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "analyze.py").write_text(
        "def analyze(train_instances, manifest=None):\n    return {}\n",
        encoding="utf-8",
    )
    (candidate_dir / "solution.py").write_text(
        "import time\n"
        "def solve(instance, analysis=None, manifest=None):\n"
        "    time.sleep(2)\n"
        "    return [1]\n",
        encoding="utf-8",
    )
    write_json(
        agent_run_dir / "synthesis_summary.json",
        {
            "best_candidate": {
                "slug": "llm_iter00_slot00",
                "candidate_dir": str(candidate_dir),
            }
        },
    )

    output_dir = tmp_path / "pace_evaluation"
    summary = export_pace_evaluation(
        dataset_dir=dataset_dir,
        agent_run_dir=agent_run_dir,
        output_dir=output_dir,
    )

    assert summary["timeout_count"] == 1
    assert summary["solver_timeout_seconds"] == 1.0
    assert summary["feasible_count"] == 0
    rows = list(csv.DictReader((output_dir / "pace_private_results.csv").open(encoding="utf-8")))
    assert rows[0]["solution_file"] == ""
    assert "SolverTimeoutError" in rows[0]["error"]
