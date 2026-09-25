from __future__ import annotations

import argparse
import csv
import io
import sys
import tarfile
import zipfile
from pathlib import Path

from benchmarks.pace_common import SolverSpec
from benchmarks.pace_competitions import (
    COMPETITIONS,
    build_pace_dataset,
    export_agent_solutions,
    parse_dfvs_metis_text,
    parse_hs_hgr_text,
    parse_ocm_gr_text,
    run_baselines,
)
from dasbench.data import load_split
from dasbench.problems import get_problem_definition
from dasbench.utils import write_json, write_jsonl


def _args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values = {
        "cache_dir": tmp_path / "cache",
        "github_ref": "master",
        "test_source": "private",
        "train_count": 1,
        "validation_count": 1,
        "test_count": 1,
        "public_start_index": 1,
        "test_start_index": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_tar_xz(path: Path, member_name: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = text.encode("utf-8")
    info = tarfile.TarInfo(member_name)
    info.size = len(payload)
    with tarfile.open(path, "w:xz") as archive:
        archive.addfile(info, io.BytesIO(payload))


def _write_tar_gz(path: Path, members: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, text in members.items():
            payload = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _write_zip(path: Path, members: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)


def test_hitting_set_parser_and_validator() -> None:
    instance = parse_hs_hgr_text("c tiny\np hs 4 2\n1 2\n2 4\n", instance_id="hs", source_path="tiny.hgr")
    problem = get_problem_definition("hitting_set")

    assert instance["sets"] == [[0, 1], [1, 3]]
    assert problem.validate_solution([1], instance) == (True, None)
    valid, error = problem.validate_solution([0], instance)
    assert not valid
    assert "not hit" in str(error)


def test_hitting_set_parser_rejects_empty_set() -> None:
    try:
        parse_hs_hgr_text("p hs 3 1\n\n", instance_id="bad", source_path="bad.hgr")
    except ValueError as exc:
        assert "parsed 0" in str(exc)
    else:
        raise AssertionError("Expected empty/missing set to be rejected.")


def test_ocm_parser_crossing_count_and_validator() -> None:
    instance = parse_ocm_gr_text(
        "p ocr 2 2 2\n1 4\n2 3\n",
        instance_id="ocm",
        source_path="tiny.gr",
    )
    problem = get_problem_definition("ocm")

    assert problem.validate_solution([0, 1], instance) == (True, None)
    assert problem.score_solution({**instance, "optimum_objective": 0}, [1, 0]).objective_value == 0
    assert problem.score_solution({**instance, "optimum_objective": 0}, [0, 1]).objective_value == 1
    valid, error = problem.validate_solution([0, 0], instance)
    assert not valid
    assert "repeated" in str(error)


def test_ocm_cutwidth_parser() -> None:
    instance = parse_ocm_gr_text(
        "p ocr 2 2 1 3\n1\n3\n2\n4\n1 3\n",
        instance_id="cw",
        source_path="cw.gr",
    )

    assert instance["cutwidth"] == 3
    assert instance["cutwidth_order"] == [0, 2, 1, 3]


def test_ocm_parser_removes_duplicate_edges() -> None:
    instance = parse_ocm_gr_text(
        "p ocr 2 2 3\n1 3\n1 3\n2 4\n",
        instance_id="dupe",
        source_path="dupe.gr",
    )
    get_problem_definition("ocm").validate_instance(instance)

    assert instance["edges"] == [[0, 0], [1, 1]]
    assert instance["pace_declared_edges"] == 3
    assert instance["pace_actual_edges"] == 2
    assert instance["pace_duplicate_edges_removed"] == 1


def test_dfvs_parser_and_validator() -> None:
    instance = parse_dfvs_metis_text("3 3 0\n2\n3\n1\n", instance_id="dfvs", source_path="tiny.graph")
    problem = get_problem_definition("dfvs")

    assert instance["arcs"] == [[0, 1], [1, 2], [2, 0]]
    assert problem.validate_solution([0], instance) == (True, None)
    valid, error = problem.validate_solution([], instance)
    assert not valid
    assert "cycle remains" in str(error)
    summary = problem.summarize_training_data([instance], {"problem": "dfvs", "family": "test"})
    assert "instance['arcs']" in summary["runtime_instance_fields"]["arcs"]
    failure = problem.failure_case(instance, [], problem.score_solution(instance, []), 0.0)
    assert "instance['arcs']" in failure["schema_hint"]


def test_dfvs_parser_preserves_blank_adjacency_lines() -> None:
    instance = parse_dfvs_metis_text("3 1 0\n2\n\n\n", instance_id="dfvs", source_path="sparse.graph")

    assert instance["num_vertices"] == 3
    assert instance["arcs"] == [[0, 1]]


def test_build_hs_dataset_from_fake_github_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache" / "pace2025-instances"
    graph = "p hs 3 2\n1 2\n2 3\n"
    _write_tar_xz(cache / "hs/heuristic/heuristic_001.hgr.tar.xz", "./heuristic_001.hgr", graph)
    _write_tar_xz(cache / "hs/heuristic/heuristic_002.hgr.tar.xz", "./heuristic_002.hgr", graph)
    _write_tar_xz(cache / "private/hs/heuristic/private_heuristic_001.hgr.tar.xz", "./001.hgr", graph)

    dataset_dir = tmp_path / "dataset"
    manifest = build_pace_dataset(
        dataset_dir=dataset_dir,
        output_root=tmp_path / "out",
        config=COMPETITIONS["pace2025_hs"],
        args=_args(tmp_path),
    )

    train = load_split(dataset_dir, "train")
    assert manifest["problem"] == "hitting_set"
    assert train[0]["optimum_objective"] == 1
    assert "_pace_reference_objective" in train[0]


def test_build_ocm_dataset_from_fake_zips(tmp_path: Path) -> None:
    cache = tmp_path / "cache" / "pace2024-ocm"
    public_graph = "p ocr 2 2 2\n1 4\n2 3\n"
    private_graph = "p ocr 2 2 2\n1 3\n2 4\n"
    _write_zip(cache / "exact-public.zip", {"1.gr": public_graph, "2.gr": public_graph})
    _write_zip(cache / "exact-public-sol.zip", {"1.sol": "4\n3\n", "2.sol": "4\n3\n"})
    _write_zip(cache / "exact-private.zip", {"101.gr": private_graph})
    _write_zip(cache / "exact-private-sol.zip", {"101.sol": "3\n4\n"})

    dataset_dir = tmp_path / "dataset"
    manifest = build_pace_dataset(
        dataset_dir=dataset_dir,
        output_root=tmp_path / "out",
        config=COMPETITIONS["pace2024_ocm_exact"],
        args=_args(tmp_path),
    )
    test = load_split(dataset_dir, "test")

    assert manifest["problem"] == "ocm"
    assert test[0]["optimum_objective"] == 0
    assert test[0]["optimum_source"] == "pace2024_exact_private_solutions"


def test_build_dfvs_dataset_from_fake_tarball(tmp_path: Path) -> None:
    graph = "3 3 0\n2\n3\n1\n"
    _write_tar_gz(
        tmp_path / "cache" / "pace2022-dfvs" / "heuristic_track_final_instances_all.tar.gz",
        {
            "public/001.graph": graph,
            "public/002.graph": graph,
            "private/001.graph": graph,
        },
    )
    dataset_dir = tmp_path / "dataset"
    manifest = build_pace_dataset(
        dataset_dir=dataset_dir,
        output_root=tmp_path / "out",
        config=COMPETITIONS["pace2022_dfvs_heuristic"],
        args=_args(tmp_path),
    )
    test = load_split(dataset_dir, "test")

    assert manifest["problem"] == "dfvs"
    assert test[0]["optimum_objective"] == 1
    assert "_pace_reference_objective" in test[0]


def test_baseline_runner_records_valid_solution(tmp_path: Path) -> None:
    cache = tmp_path / "cache" / "pace2025-instances"
    graph = "p hs 3 2\n1 2\n2 3\n"
    _write_tar_xz(cache / "hs/heuristic/heuristic_001.hgr.tar.xz", "./heuristic_001.hgr", graph)
    _write_tar_xz(cache / "hs/heuristic/heuristic_002.hgr.tar.xz", "./heuristic_002.hgr", graph)
    _write_tar_xz(cache / "private/hs/heuristic/private_heuristic_001.hgr.tar.xz", "./001.hgr", graph)
    dataset_dir = tmp_path / "dataset"
    build_pace_dataset(
        dataset_dir=dataset_dir,
        output_root=tmp_path / "out",
        config=COMPETITIONS["pace2025_hs"],
        args=_args(tmp_path),
    )
    solver_script = tmp_path / "solver.py"
    solver_script.write_text("print('1')\nprint('2')\n", encoding="utf-8")

    summary = run_baselines(
        config=COMPETITIONS["pace2025_hs"],
        dataset_dir=dataset_dir,
        solvers=[SolverSpec(name="tiny", command=[sys.executable, str(solver_script)])],
        output_dir=tmp_path / "baselines",
        timeout_seconds=5,
        grace_seconds=1,
    )

    assert summary["solvers"]["tiny"]["valid_count"] == 1
    assert summary["solvers"]["tiny"]["average_objective_value"] == 1


def test_agent_solution_export_enforces_solver_timeout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DASBENCH_SOLVER_TIMEOUT_SECONDS", "1")
    dataset_dir = tmp_path / "dataset"
    instance = {
        "id": "hs-timeout",
        "num_vertices": 2,
        "sets": [[0], [1]],
        "optimum_objective": 2,
        "pace_source_path": "private/hs/heuristic/private_heuristic_001.hgr.tar.xz",
    }
    write_json(
        dataset_dir / "manifest.json",
        {
            "problem": "hitting_set",
            "family": "pace2025_hs_heuristic_private",
            "metric_definition": {"primary": "normalized_quality"},
            "instance_schema_version": "hitting_set.v1",
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
        "    return [0, 1]\n",
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
    summary = export_agent_solutions(
        config=COMPETITIONS["pace2025_hs"],
        dataset_dir=dataset_dir,
        agent_run_dir=agent_run_dir,
        output_dir=output_dir,
    )

    assert summary["timeout_count"] == 1
    assert summary["solver_timeout_seconds"] == 1.0
    assert summary["feasible_count"] == 0
    rows = list(csv.DictReader((output_dir / "pace_results.csv").open(encoding="utf-8")))
    assert "SolverTimeoutError" in rows[0]["error"]
    assert rows[0]["solution_file"] == ""
