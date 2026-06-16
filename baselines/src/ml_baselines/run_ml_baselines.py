from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml

from dasbench.data import load_manifest, load_split
from dasbench.problems import get_problem_definition
from dasbench.problems.base import ScoreResult
from dasbench.utils import load_jsonl, public_instance, write_json, write_jsonl
from ml_baselines.attention_tsp import ATTENTION_TSP_BASELINE_NAME
from ml_baselines.attention_tsp import AttentionTSPConfig, AttentionTSPModel
from ml_baselines.attention_tsp import fit as fit_attention_tsp
from ml_baselines.attention_tsp import solve as solve_attention_tsp
from ml_baselines.checkpoint import load_checkpoint
from ml_baselines.drl_mdkp import DRL_MDKP_BASELINE_NAME
from ml_baselines.drl_mdkp import DRLMDKPConfig, MDKPActorCriticNet
from ml_baselines.drl_mdkp import fit as fit_drl_mdkp
from ml_baselines.drl_mdkp import solve as solve_drl_mdkp
from ml_baselines.gnn_rl_mds import GNN_RL_MDS_BASELINE_NAME
from ml_baselines.gnn_rl_mds import GNNRLMDSConfig
from ml_baselines.gnn_rl_mds import fit as fit_gnn_rl_mds
from ml_baselines.gnn_rl_mds import solve as solve_gnn_rl_mds
from ml_baselines.graph_score_repair import GRAPH_BASELINE_NAMES
from ml_baselines.graph_score_repair import fit as fit_graph
from ml_baselines.graph_score_repair import solve as solve_graph
from ml_baselines.interface import TrainedState
from ml_baselines.item_resource_baselines import MDKP_BASELINE_NAME, PACKINGLP_BASELINE_NAME
from ml_baselines.item_resource_baselines import fit as fit_item_resource
from ml_baselines.item_resource_baselines import solve as solve_item_resource
from ml_baselines.maxsat_assignment import MAXSAT_BASELINE_NAME
from ml_baselines.maxsat_assignment import fit as fit_maxsat
from ml_baselines.maxsat_assignment import solve as solve_maxsat
from ml_baselines.models import ManualMessagePassing
from ml_baselines.pignn_coloring import PIGNN_COLORING_BASELINE_NAME
from ml_baselines.pignn_coloring import PiGNNColoringConfig
from ml_baselines.pignn_coloring import fit as fit_pignn_coloring
from ml_baselines.pignn_coloring import solve as solve_pignn_coloring
from ml_baselines.pignn_mis import PIGNN_MIS_BASELINE_NAME
from ml_baselines.pignn_mis import PiGNNMISConfig
from ml_baselines.pignn_mis import fit as fit_pignn_mis
from ml_baselines.pignn_mis import solve as solve_pignn_mis
from ml_baselines.pdl_packinglp import PDL_PACKINGLP_BASELINE_NAME
from ml_baselines.pdl_packinglp import PDLPackingLPConfig, PDLPackingLPNet
from ml_baselines.pdl_packinglp import fit as fit_pdl_packinglp
from ml_baselines.pdl_packinglp import solve as solve_pdl_packinglp
from ml_baselines.runcsp_maxsat import RUN_CSP_MAXSAT_BASELINE_NAME
from ml_baselines.runcsp_maxsat import RunCSPMaxSatConfig, RunCSPMaxSatNet
from ml_baselines.runcsp_maxsat import fit as fit_runcsp_maxsat
from ml_baselines.runcsp_maxsat import solve as solve_runcsp_maxsat
from ml_baselines.seeding import seed_everything
from ml_baselines.torch_utils import require_torch, resolve_device
from ml_baselines.tsp_neural_constructor import TSP_BASELINE_NAME
from ml_baselines.tsp_neural_constructor import fit as fit_tsp
from ml_baselines.tsp_neural_constructor import solve as solve_tsp

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    problem: str
    fit: Callable[..., Any]
    solve: Callable[..., Any]


@dataclass
class LoadedTarget:
    problem: str
    family: str
    dataset_id: str
    target: str
    dataset_dir: Path | None
    manifest: dict[str, object]
    train_public: list[dict[str, object]]
    validation_public: list[dict[str, object]]
    validation_full: list[dict[str, object]]
    test_full: list[dict[str, object]]


RESERVED_CONFIG_KEYS = {
    "defaults",
    "problems",
    "baselines",
    "search",
    "hyperparameter_grid",
    "baseline_search",
}

PACE_PROBLEM_NAME = "pace"
PACE_BACKING_PROBLEM_NAME = "mds"
DEFAULT_PACE_DATASET_DIR = Path("artifacts/pace2025_dominating_set/pace2025_ds_heuristic_llm_01/dataset")


def _implementation_problem(problem: str) -> str:
    return PACE_BACKING_PROBLEM_NAME if problem == PACE_PROBLEM_NAME else problem


def _cli_problem_choices() -> list[str]:
    return sorted({spec.problem for spec in BASELINE_SPECS.values()} | {PACE_PROBLEM_NAME})


BASELINE_SPECS: dict[str, BaselineSpec] = {
    GRAPH_BASELINE_NAMES["mis"]: BaselineSpec(
        name=GRAPH_BASELINE_NAMES["mis"],
        problem="mis",
        fit=lambda problem, train, val, config, checkpoint_path, metrics_path: fit_graph(
            problem,
            train,
            val,
            config,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
        ),
        solve=lambda problem, instance, state, config: solve_graph(problem, instance, state, config),
    ),
    PIGNN_MIS_BASELINE_NAME: BaselineSpec(
        name=PIGNN_MIS_BASELINE_NAME,
        problem="mis",
        fit=lambda problem, train, val, config, checkpoint_path, metrics_path: fit_pignn_mis(
            train,
            val,
            config,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
        ),
        solve=lambda problem, instance, state, config: solve_pignn_mis(instance, state, config),
    ),
    GRAPH_BASELINE_NAMES["mds"]: BaselineSpec(
        name=GRAPH_BASELINE_NAMES["mds"],
        problem="mds",
        fit=lambda problem, train, val, config, checkpoint_path, metrics_path: fit_graph(
            problem,
            train,
            val,
            config,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
        ),
        solve=lambda problem, instance, state, config: solve_graph(problem, instance, state, config),
    ),
    GNN_RL_MDS_BASELINE_NAME: BaselineSpec(
        name=GNN_RL_MDS_BASELINE_NAME,
        problem="mds",
        fit=lambda problem, train, val, config, checkpoint_path, metrics_path: fit_gnn_rl_mds(
            train,
            val,
            config,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
        ),
        solve=lambda problem, instance, state, config: solve_gnn_rl_mds(instance, state, config),
    ),
    PIGNN_COLORING_BASELINE_NAME: BaselineSpec(
        name=PIGNN_COLORING_BASELINE_NAME,
        problem="coloring",
        fit=lambda problem, train, val, config, checkpoint_path, metrics_path: fit_pignn_coloring(
            train,
            val,
            config,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
        ),
        solve=lambda problem, instance, state, config: solve_pignn_coloring(instance, state, config),
    ),
    MAXSAT_BASELINE_NAME: BaselineSpec(
        name=MAXSAT_BASELINE_NAME,
        problem="maxsat",
        fit=lambda problem, train, val, config, checkpoint_path, metrics_path: fit_maxsat(
            train,
            val,
            config,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
        ),
        solve=lambda problem, instance, state, config: solve_maxsat(instance, state, config),
    ),
    RUN_CSP_MAXSAT_BASELINE_NAME: BaselineSpec(
        name=RUN_CSP_MAXSAT_BASELINE_NAME,
        problem="maxsat",
        fit=lambda problem, train, val, config, checkpoint_path, metrics_path: fit_runcsp_maxsat(
            train,
            val,
            config,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
        ),
        solve=lambda problem, instance, state, config: solve_runcsp_maxsat(instance, state, config),
    ),
    MDKP_BASELINE_NAME: BaselineSpec(
        name=MDKP_BASELINE_NAME,
        problem="mdkp",
        fit=lambda problem, train, val, config, checkpoint_path, metrics_path: fit_item_resource(
            problem,
            train,
            val,
            config,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
        ),
        solve=lambda problem, instance, state, config: solve_item_resource(problem, instance, state, config),
    ),
    DRL_MDKP_BASELINE_NAME: BaselineSpec(
        name=DRL_MDKP_BASELINE_NAME,
        problem="mdkp",
        fit=lambda problem, train, val, config, checkpoint_path, metrics_path: fit_drl_mdkp(
            train,
            val,
            config,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
        ),
        solve=lambda problem, instance, state, config: solve_drl_mdkp(instance, state, config),
    ),
    PACKINGLP_BASELINE_NAME: BaselineSpec(
        name=PACKINGLP_BASELINE_NAME,
        problem="packing_lp",
        fit=lambda problem, train, val, config, checkpoint_path, metrics_path: fit_item_resource(
            problem,
            train,
            val,
            config,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
        ),
        solve=lambda problem, instance, state, config: solve_item_resource(problem, instance, state, config),
    ),
    PDL_PACKINGLP_BASELINE_NAME: BaselineSpec(
        name=PDL_PACKINGLP_BASELINE_NAME,
        problem="packing_lp",
        fit=lambda problem, train, val, config, checkpoint_path, metrics_path: fit_pdl_packinglp(
            train,
            val,
            config,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
        ),
        solve=lambda problem, instance, state, config: solve_pdl_packinglp(instance, state, config),
    ),
    TSP_BASELINE_NAME: BaselineSpec(
        name=TSP_BASELINE_NAME,
        problem="tsp",
        fit=lambda problem, train, val, config, checkpoint_path, metrics_path: fit_tsp(
            train,
            val,
            config,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
        ),
        solve=lambda problem, instance, state, config: solve_tsp(instance, state, config),
    ),
    ATTENTION_TSP_BASELINE_NAME: BaselineSpec(
        name=ATTENTION_TSP_BASELINE_NAME,
        problem="tsp",
        fit=lambda problem, train, val, config, checkpoint_path, metrics_path: fit_attention_tsp(
            train,
            val,
            config,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
        ),
        solve=lambda problem, instance, state, config: solve_attention_tsp(instance, state, config),
    ),
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config override file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Config override file must contain a JSON/YAML object.")
    return payload


def _merge_dicts(*payloads: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for payload in payloads:
        result.update({key: value for key, value in payload.items() if value is not None})
    return result


def _base_config(
    raw_config: dict[str, Any],
    *,
    problem: str,
    baseline_name: str,
    seed: int,
    device: str,
) -> dict[str, Any]:
    plain = {key: value for key, value in raw_config.items() if key not in RESERVED_CONFIG_KEYS}
    defaults = raw_config.get("defaults") if isinstance(raw_config.get("defaults"), dict) else {}
    problems = raw_config.get("problems") if isinstance(raw_config.get("problems"), dict) else {}
    baselines = raw_config.get("baselines") if isinstance(raw_config.get("baselines"), dict) else {}
    backing_problem = _implementation_problem(problem)
    backing_problem_config = (
        problems.get(backing_problem, {})
        if backing_problem != problem and isinstance(problems.get(backing_problem, {}), dict)
        else {}
    )
    problem_config = problems.get(problem, {}) if isinstance(problems.get(problem, {}), dict) else {}
    baseline_config = baselines.get(baseline_name, {}) if isinstance(baselines.get(baseline_name, {}), dict) else {}
    return _merge_dicts(plain, defaults, backing_problem_config, problem_config, baseline_config, {"seed": seed, "device": device})


def _search_configs(
    raw_config: dict[str, Any],
    *,
    baseline_name: str,
    base_config: dict[str, Any],
    select_on_validation: bool,
) -> list[dict[str, Any]]:
    if not select_on_validation:
        return [base_config]
    baseline_search = raw_config.get("baseline_search") if isinstance(raw_config.get("baseline_search"), dict) else {}
    search = baseline_search.get(baseline_name) if isinstance(baseline_search.get(baseline_name), list) else None
    if search is None:
        raw_search = raw_config.get("search", raw_config.get("hyperparameter_grid", []))
        search = raw_search if isinstance(raw_search, list) else []
    if not search:
        return [base_config]
    configs = []
    for entry in search:
        if not isinstance(entry, dict):
            raise ValueError("Each search/hyperparameter_grid entry must be an object.")
        configs.append(_merge_dicts(base_config, entry))
    return configs


def _read_split_file(path: Path, *, problem_name: str, public: bool) -> list[dict[str, object]]:
    problem = get_problem_definition(problem_name)
    rows = load_jsonl(path)
    for row in rows:
        problem.validate_instance(row)
    return [public_instance(row) for row in rows] if public else rows


def _infer_manifest_from_splits(problem: str, train_split: Path) -> dict[str, object]:
    manifest_path = train_split.parent / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "problem": problem,
        "family": "custom",
        "split_sizes": {},
        "artifact_paths": {
            "splits": {
                "train": str(train_split),
            }
        },
    }


def _load_target_from_splits(
    *,
    problem: str,
    train_split: Path,
    validation_split: Path,
    test_split: Path,
) -> LoadedTarget:
    manifest = _infer_manifest_from_splits(problem, train_split)
    validation_full = _read_split_file(validation_split, problem_name=problem, public=False)
    test_full = _read_split_file(test_split, problem_name=problem, public=False)
    family = str(manifest.get("family", "custom"))
    dataset_id = str(train_split.parent.name or "custom")
    return LoadedTarget(
        problem=problem,
        family=family,
        dataset_id=dataset_id,
        target=f"{family}/{dataset_id}",
        dataset_dir=train_split.parent,
        manifest=manifest,
        train_public=_read_split_file(train_split, problem_name=problem, public=True),
        validation_public=[public_instance(instance) for instance in validation_full],
        validation_full=validation_full,
        test_full=test_full,
    )


def _load_target_from_dataset_dir(dataset_dir: Path, *, requested_problem: str | None = None) -> LoadedTarget:
    manifest = load_manifest(dataset_dir)
    manifest_problem = str(manifest["problem"])
    if requested_problem == PACE_PROBLEM_NAME and manifest_problem != PACE_BACKING_PROBLEM_NAME:
        raise ValueError(f"PACE datasets must use `{PACE_BACKING_PROBLEM_NAME}` schema, got `{manifest_problem}`.")
    problem = PACE_PROBLEM_NAME if requested_problem == PACE_PROBLEM_NAME else manifest_problem
    family = str(manifest.get("family", dataset_dir.parent.name))
    dataset_id = str(dataset_dir.name)
    return LoadedTarget(
        problem=problem,
        family=family,
        dataset_id=dataset_id,
        target=f"{family}/{dataset_id}",
        dataset_dir=dataset_dir,
        manifest=manifest,
        train_public=load_split(dataset_dir, "train", public=True),
        validation_public=load_split(dataset_dir, "validation", public=True),
        validation_full=load_split(dataset_dir, "validation", public=False),
        test_full=load_split(dataset_dir, "test", public=False),
    )


def _discover_dataset_dirs(problem: str, *, target: str, dataset_root: Path) -> list[Path]:
    if problem == PACE_PROBLEM_NAME:
        if target == "all":
            if not DEFAULT_PACE_DATASET_DIR.exists():
                raise FileNotFoundError(f"Default PACE dataset not found: {DEFAULT_PACE_DATASET_DIR}")
            return [DEFAULT_PACE_DATASET_DIR]
        raw_target = Path(target)
        if raw_target.exists():
            return [raw_target]
        direct = dataset_root / problem / target
        if (direct / "manifest.json").exists():
            return [direct]
        raise FileNotFoundError(
            f"No PACE dataset target `{target}` found. Use --dataset-dir or target the default dataset with --target all."
        )
    if target == "all":
        return sorted(path.parent for path in (dataset_root / problem).glob("*/*/manifest.json"))
    raw_target = Path(target)
    if raw_target.exists():
        return [raw_target]
    direct = dataset_root / problem / target
    if (direct / "manifest.json").exists():
        return [direct]
    if direct.exists():
        found = sorted(path.parent for path in direct.glob("*/manifest.json"))
        if found:
            return found
    by_dataset_id = sorted(path.parent for path in (dataset_root / problem).glob(f"*/{target}/manifest.json"))
    if by_dataset_id:
        return by_dataset_id
    raise FileNotFoundError(f"No dataset target `{target}` found for problem `{problem}` under {dataset_root}.")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return str(value)
        return value
    if isinstance(value, ScoreResult):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _evaluate_per_instance(
    *,
    problem_name: str,
    baseline_name: str,
    solver: Callable[[dict[str, object]], Any],
    instances: list[dict[str, object]],
    split: str,
    output_jsonl: Path,
    output_csv: Path | None = None,
) -> dict[str, object]:
    problem = get_problem_definition(problem_name)
    rows: list[dict[str, object]] = []
    for instance in instances:
        exposed = public_instance(instance)
        start = time.perf_counter()
        try:
            raw_solution = solver(exposed)
            runtime_ms = (time.perf_counter() - start) * 1000.0
            solution = problem.canonicalize_solution(raw_solution, exposed)
            score = problem.score_solution(instance, solution)
        except Exception as exc:
            runtime_ms = (time.perf_counter() - start) * 1000.0
            solution = []
            score = ScoreResult(
                is_valid=False,
                is_feasible=False,
                objective_value=0.0,
                normalized_quality=0.0,
                is_optimal=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        rows.append(
            {
                "problem": problem_name,
                "baseline": baseline_name,
                "split": split,
                "instance_id": str(instance.get("id", "")),
                "solution": _json_safe(solution),
                "objective_value": score.objective_value,
                "normalized_quality": score.normalized_quality,
                "is_valid": score.is_valid,
                "is_feasible": score.is_feasible,
                "is_optimal": score.is_optimal,
                "runtime_ms": runtime_ms,
                "error": score.error,
            }
        )
    write_jsonl(output_jsonl, [_json_safe(row) for row in rows])
    if output_csv is not None:
        _write_csv(output_csv, rows, PER_INSTANCE_CSV_FIELDS)
    return _summary_from_rows(problem_name, baseline_name, split, rows)


def _summary_from_rows(
    problem_name: str,
    baseline_name: str,
    split: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    if not rows:
        return {
            "name": baseline_name,
            "problem": problem_name,
            "split": split,
            "num_instances": 0,
            "average_normalized_quality": 0.0,
            "average_objective_value": 0.0,
            "optimality_rate": 0.0,
            "feasibility_rate": 0.0,
            "average_runtime_ms": 0.0,
            "error_count": 0,
        }
    return {
        "name": baseline_name,
        "problem": problem_name,
        "split": split,
        "num_instances": len(rows),
        "average_normalized_quality": sum(float(row["normalized_quality"]) for row in rows) / len(rows),
        "average_objective_value": sum(float(row["objective_value"]) for row in rows) / len(rows),
        "optimality_rate": sum(1.0 for row in rows if bool(row["is_optimal"])) / len(rows),
        "feasibility_rate": sum(1.0 for row in rows if bool(row["is_feasible"])) / len(rows),
        "average_runtime_ms": sum(float(row["runtime_ms"]) for row in rows) / len(rows),
        "error_count": sum(1 for row in rows if row.get("error")),
    }


def _selection_key(summary: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        float(summary.get("average_normalized_quality", 0.0)),
        float(summary.get("feasibility_rate", 0.0)),
        float(summary.get("optimality_rate", 0.0)),
        -float(summary.get("average_runtime_ms", 0.0)),
    )


def _solver_for_state(spec: BaselineSpec, state: Any, config: dict[str, Any]) -> Callable[[dict[str, object]], Any]:
    return lambda instance: spec.solve(spec.problem, instance, state, config)


def _train_one_trial(
    *,
    spec: BaselineSpec,
    target: LoadedTarget,
    config: dict[str, Any],
    trial_dir: Path,
) -> tuple[Any, float, Path, Path]:
    checkpoint_path = trial_dir / f"{spec.name}.pt"
    metrics_path = trial_dir / f"{spec.name}_train_metrics.jsonl"
    write_json(trial_dir / "config.json", config)
    start = time.perf_counter()
    state = spec.fit(
        spec.problem,
        target.train_public,
        target.validation_public,
        config,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )
    training_ms = (time.perf_counter() - start) * 1000.0
    return state, training_ms, checkpoint_path, metrics_path


def run_baseline_on_target(
    *,
    spec: BaselineSpec,
    target: LoadedTarget,
    output_dir: Path,
    raw_config: dict[str, Any],
    seed: int,
    device: str,
    select_on_validation: bool,
) -> dict[str, object]:
    baseline_dir = output_dir / target.problem / target.family / target.dataset_id / spec.name
    trials_dir = baseline_dir / "trials"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    base_config = _base_config(raw_config, problem=target.problem, baseline_name=spec.name, seed=seed, device=device)
    trial_configs = _search_configs(
        raw_config,
        baseline_name=spec.name,
        base_config=base_config,
        select_on_validation=select_on_validation,
    )
    trial_records: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    selected_state: Any | None = None
    total_training_ms = 0.0
    for trial_index, config in enumerate(trial_configs):
        trial_dir = trials_dir / f"trial_{trial_index:03d}"
        state, training_ms, checkpoint_path, metrics_path = _train_one_trial(
            spec=spec,
            target=target,
            config=config,
            trial_dir=trial_dir,
        )
        total_training_ms += training_ms
        solver = _solver_for_state(spec, state, config)
        validation_summary = _evaluate_per_instance(
            problem_name=target.problem,
            baseline_name=spec.name,
            solver=solver,
            instances=target.validation_full,
            split="validation",
            output_jsonl=trial_dir / "validation_outputs.jsonl",
            output_csv=trial_dir / "validation_outputs.csv",
        )
        record = {
            "trial_index": trial_index,
            "config": config,
            "training_time_ms": training_ms,
            "checkpoint_path": str(checkpoint_path),
            "metrics_path": str(metrics_path),
            "validation_summary": validation_summary,
        }
        trial_records.append(record)
        if selected is None or _selection_key(validation_summary) > _selection_key(selected["validation_summary"]):  # type: ignore[index]
            selected = record
            selected_state = state
    assert selected is not None and selected_state is not None

    selected_config = dict(selected["config"])  # type: ignore[arg-type]
    selected_solver = _solver_for_state(spec, selected_state, selected_config)
    test_summary = _evaluate_per_instance(
        problem_name=target.problem,
        baseline_name=spec.name,
        solver=selected_solver,
        instances=target.test_full,
        split="test",
        output_jsonl=baseline_dir / "test_outputs.jsonl",
        output_csv=baseline_dir / "test_outputs.csv",
    )
    checkpoint_source = Path(str(selected["checkpoint_path"]))
    selected_checkpoint_path = baseline_dir / f"{spec.name}.pt"
    if checkpoint_source.exists() and checkpoint_source.resolve() != selected_checkpoint_path.resolve():
        shutil.copy2(checkpoint_source, selected_checkpoint_path)
    selected_metrics_source = Path(str(selected["metrics_path"]))
    selected_metrics_path = baseline_dir / f"{spec.name}_train_metrics.jsonl"
    if selected_metrics_source.exists() and selected_metrics_source.resolve() != selected_metrics_path.resolve():
        shutil.copy2(selected_metrics_source, selected_metrics_path)

    write_json(baseline_dir / "selected_config.json", selected_config)
    write_json(baseline_dir / "validation_selection.json", {"selected": selected, "trials": trial_records})
    write_json(baseline_dir / "test_summary.json", test_summary)
    row = _aggregate_row(
        target=target,
        spec=spec,
        selected=selected,
        test_summary=test_summary,
        baseline_dir=baseline_dir,
        selected_checkpoint_path=selected_checkpoint_path,
        selected_metrics_path=selected_metrics_path,
        total_training_ms=total_training_ms,
        trial_count=len(trial_records),
        seed=seed,
        device=device,
        selected_config=selected_config,
    )
    write_json(baseline_dir / "run_summary.json", row)
    return row


def _load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _checkpoint_config(
    *,
    checkpoint: dict[str, Any],
    target: LoadedTarget,
    spec: BaselineSpec,
    raw_config: dict[str, Any],
    seed: int,
    device: str,
) -> dict[str, Any]:
    metadata = checkpoint.get("metadata") or {}
    config = dict(metadata.get("config") or {})
    config.update(_base_config(raw_config, problem=target.problem, baseline_name=spec.name, seed=seed, device=device))
    return config


def _load_trained_state_from_checkpoint(
    *,
    spec: BaselineSpec,
    checkpoint_path: Path,
    config: dict[str, Any],
    device: str,
) -> TrainedState:
    torch = require_torch()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", use_torch=True)
    state = checkpoint.get("state")
    metadata = dict(checkpoint.get("metadata") or {})
    if not isinstance(state, dict) or "model_state_dict" not in state:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a model_state_dict.")
    resolved_device = resolve_device(device)

    if spec.name == PIGNN_MIS_BASELINE_NAME:
        resolved = PiGNNMISConfig.from_config(config)
        model = ManualMessagePassing(int(state["input_dim"]), resolved.hidden_dim, 1, layers=resolved.layers)
        payload = {"model": model, "input_dim": int(state["input_dim"])}
    elif spec.name == GNN_RL_MDS_BASELINE_NAME:
        resolved = GNNRLMDSConfig.from_config(config)
        model = ManualMessagePassing(int(state["input_dim"]), resolved.hidden_dim, 1, layers=resolved.layers)
        payload = {"model": model, "input_dim": int(state["input_dim"])}
    elif spec.name == PIGNN_COLORING_BASELINE_NAME:
        resolved = PiGNNColoringConfig.from_config(config)
        max_colors = int(state["max_colors"])
        model = ManualMessagePassing(int(state["input_dim"]), resolved.hidden_dim, max_colors, layers=resolved.layers)
        payload = {"model": model, "input_dim": int(state["input_dim"]), "max_colors": max_colors}
    elif spec.name == RUN_CSP_MAXSAT_BASELINE_NAME:
        resolved = RunCSPMaxSatConfig.from_config(config)
        model = RunCSPMaxSatNet(
            int(state["variable_dim"]),
            int(state["clause_dim"]),
            resolved.hidden_dim,
            resolved.message_passing_steps,
            resolved.recurrent_layers,
        )
        payload = {"model": model, "variable_dim": int(state["variable_dim"]), "clause_dim": int(state["clause_dim"])}
    elif spec.name == DRL_MDKP_BASELINE_NAME:
        resolved = DRLMDKPConfig.from_config(config)
        model = MDKPActorCriticNet(int(state["item_dim"]), resolved.hidden_dim)
        payload = {"model": model, "item_dim": int(state["item_dim"])}
    elif spec.name == PDL_PACKINGLP_BASELINE_NAME:
        resolved = PDLPackingLPConfig.from_config(config)
        model = PDLPackingLPNet(
            int(state["item_dim"]),
            int(state["resource_dim"]),
            resolved.hidden_dim,
            resolved.layers,
        )
        payload = {"model": model, "item_dim": int(state["item_dim"]), "resource_dim": int(state["resource_dim"])}
    elif spec.name == ATTENTION_TSP_BASELINE_NAME:
        resolved = AttentionTSPConfig.from_config(config)
        model = AttentionTSPModel(
            int(state.get("embedding_dim", resolved.embedding_dim)),
            int(state.get("hidden_dim", resolved.hidden_dim)),
            int(state.get("n_heads", resolved.n_heads)),
            int(state.get("n_encoder_layers", resolved.n_encoder_layers)),
        )
        payload = {"model": model, "embedding_dim": int(state.get("embedding_dim", resolved.embedding_dim))}
    else:
        raise ValueError(f"Eval-only checkpoint loading is not implemented for baseline `{spec.name}`.")

    model.load_state_dict(state["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return TrainedState(payload=payload, metadata=metadata)


def evaluate_checkpoint_on_target(
    *,
    spec: BaselineSpec,
    target: LoadedTarget,
    output_dir: Path,
    eval_only_from: Path,
    raw_config: dict[str, Any],
    seed: int,
    device: str,
) -> dict[str, object]:
    source_baseline_dir = eval_only_from / target.problem / target.family / target.dataset_id / spec.name
    checkpoint_path = source_baseline_dir / f"{spec.name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing eval-only checkpoint: {checkpoint_path}")

    baseline_dir = output_dir / target.problem / target.family / target.dataset_id / spec.name
    baseline_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", use_torch=True)
    selected_config = _checkpoint_config(
        checkpoint=checkpoint,
        target=target,
        spec=spec,
        raw_config=raw_config,
        seed=seed,
        device=device,
    )
    seed_everything(int(selected_config.get("seed", seed)))
    trained_state = _load_trained_state_from_checkpoint(
        spec=spec,
        checkpoint_path=checkpoint_path,
        config=selected_config,
        device=device,
    )
    solver = _solver_for_state(spec, trained_state, selected_config)
    test_summary = _evaluate_per_instance(
        problem_name=target.problem,
        baseline_name=spec.name,
        solver=solver,
        instances=target.test_full,
        split="test",
        output_jsonl=baseline_dir / "test_outputs.jsonl",
        output_csv=baseline_dir / "test_outputs.csv",
    )

    source_summary = _load_json_if_present(source_baseline_dir / "run_summary.json")
    source_selection = _load_json_if_present(source_baseline_dir / "validation_selection.json")
    selected_source = source_selection.get("selected") if isinstance(source_selection.get("selected"), dict) else {}
    validation_summary = selected_source.get("validation_summary") if isinstance(selected_source.get("validation_summary"), dict) else {}
    if not validation_summary:
        validation_summary = {
            "average_normalized_quality": float(source_summary.get("selected_validation_quality", 0.0) or 0.0),
            "feasibility_rate": float(source_summary.get("selected_validation_feasibility_rate", 0.0) or 0.0),
        }
    source_metrics_path = Path(
        str(source_summary.get("train_metrics_path") or source_baseline_dir / f"{spec.name}_train_metrics.jsonl")
    )
    selected = {
        "trial_index": int(source_summary.get("selected_trial_index", selected_source.get("trial_index", 0)) or 0),
        "config": selected_config,
        "training_time_ms": 0.0,
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(source_metrics_path),
        "validation_summary": validation_summary,
    }

    write_json(baseline_dir / "selected_config.json", selected_config)
    write_json(
        baseline_dir / "eval_only_source.json",
        {
            "eval_only_from": str(eval_only_from),
            "source_run_dir": str(source_baseline_dir),
            "source_checkpoint_path": str(checkpoint_path),
            "source_training_time_ms": float(source_summary.get("total_training_time_ms", 0.0) or 0.0),
        },
    )
    write_json(baseline_dir / "test_summary.json", test_summary)
    row = _aggregate_row(
        target=target,
        spec=spec,
        selected=selected,
        test_summary=test_summary,
        baseline_dir=baseline_dir,
        selected_checkpoint_path=checkpoint_path,
        selected_metrics_path=source_metrics_path,
        total_training_ms=0.0,
        trial_count=0,
        seed=seed,
        device=device,
        selected_config=selected_config,
    )
    row["eval_only_from"] = str(eval_only_from)
    row["source_run_dir"] = str(source_baseline_dir)
    write_json(baseline_dir / "run_summary.json", row)
    return row


def _aggregate_row(
    *,
    target: LoadedTarget,
    spec: BaselineSpec,
    selected: dict[str, object],
    test_summary: dict[str, object],
    baseline_dir: Path,
    selected_checkpoint_path: Path,
    selected_metrics_path: Path,
    total_training_ms: float,
    trial_count: int,
    seed: int,
    device: str,
    selected_config: dict[str, Any],
) -> dict[str, object]:
    return {
        "problem": target.problem,
        "family": target.family,
        "dataset_id": target.dataset_id,
        "target": target.target,
        "baseline": spec.name,
        "split": "test",
        "seed": seed,
        "device": device,
        "num_train_instances": len(target.train_public),
        "num_validation_instances": len(target.validation_full),
        "num_test_instances": int(test_summary.get("num_instances", 0)),
        "average_normalized_quality": float(test_summary.get("average_normalized_quality", 0.0)),
        "average_objective_value": float(test_summary.get("average_objective_value", 0.0)),
        "optimality_rate": float(test_summary.get("optimality_rate", 0.0)),
        "feasibility_rate": float(test_summary.get("feasibility_rate", 0.0)),
        "average_runtime_ms": float(test_summary.get("average_runtime_ms", 0.0)),
        "error_count": int(test_summary.get("error_count", 0)),
        "selected_trial_index": int(selected["trial_index"]),
        "selected_validation_quality": float(selected["validation_summary"]["average_normalized_quality"]),  # type: ignore[index]
        "selected_validation_feasibility_rate": float(selected["validation_summary"]["feasibility_rate"]),  # type: ignore[index]
        "selected_training_time_ms": float(selected["training_time_ms"]),
        "total_training_time_ms": float(total_training_ms),
        "trial_count": int(trial_count),
        "checkpoint_path": str(selected_checkpoint_path),
        "train_metrics_path": str(selected_metrics_path),
        "test_outputs_path": str(baseline_dir / "test_outputs.jsonl"),
        "test_outputs_csv_path": str(baseline_dir / "test_outputs.csv"),
        "run_dir": str(baseline_dir),
        "config_json": json.dumps(selected_config, sort_keys=True),
    }


AGGREGATE_CSV_FIELDS = [
    "problem",
    "family",
    "dataset_id",
    "target",
    "baseline",
    "split",
    "seed",
    "device",
    "num_train_instances",
    "num_validation_instances",
    "num_test_instances",
    "average_normalized_quality",
    "average_objective_value",
    "optimality_rate",
    "feasibility_rate",
    "average_runtime_ms",
    "error_count",
    "selected_trial_index",
    "selected_validation_quality",
    "selected_validation_feasibility_rate",
    "selected_training_time_ms",
    "total_training_time_ms",
    "trial_count",
    "checkpoint_path",
    "train_metrics_path",
    "test_outputs_path",
    "test_outputs_csv_path",
    "run_dir",
    "config_json",
]

PER_INSTANCE_CSV_FIELDS = [
    "problem",
    "baseline",
    "split",
    "instance_id",
    "objective_value",
    "normalized_quality",
    "is_valid",
    "is_feasible",
    "is_optimal",
    "runtime_ms",
    "error",
    "solution",
]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _baseline_specs_for_problem(problem: str, requested: list[str] | None) -> list[BaselineSpec]:
    implementation_problem = _implementation_problem(problem)
    specs = [spec for spec in BASELINE_SPECS.values() if spec.problem == implementation_problem]
    if requested:
        requested_set = set(requested)
        unknown = sorted(requested_set - set(BASELINE_SPECS))
        if unknown:
            raise ValueError(f"Unknown ML baseline(s): {', '.join(unknown)}")
        specs = [BASELINE_SPECS[name] for name in requested if BASELINE_SPECS[name].problem == implementation_problem]
    if not specs:
        raise ValueError(f"No implemented ML baselines selected for problem `{problem}`.")
    return specs


def run_from_args(args: argparse.Namespace) -> dict[str, object]:
    raw_config = _load_config(args.config)
    run_id = args.run_id or _timestamp()
    output_root = args.output_dir / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    if args.train_split or args.validation_split or args.test_split:
        if not (args.train_split and args.validation_split and args.test_split):
            raise ValueError("--train-split, --validation-split, and --test-split must be provided together.")
        targets = [
            _load_target_from_splits(
                problem=args.problem,
                train_split=args.train_split,
                validation_split=args.validation_split,
                test_split=args.test_split,
            )
        ]
    else:
        dataset_dirs = args.dataset_dir or _discover_dataset_dirs(
            args.problem,
            target=args.target,
            dataset_root=args.dataset_root,
        )
        targets = [_load_target_from_dataset_dir(path, requested_problem=args.problem) for path in dataset_dirs]
    for target in targets:
        if target.problem != args.problem:
            raise ValueError(f"Target {target.target} is for problem `{target.problem}`, not `{args.problem}`.")
    specs = _baseline_specs_for_problem(args.problem, args.baseline)
    rows: list[dict[str, object]] = []

    if args.eval_only_from is not None:
        requested_baselines = bool(args.baseline)
        for target in targets:
            for spec in specs:
                try:
                    rows.append(
                        evaluate_checkpoint_on_target(
                            spec=spec,
                            target=target,
                            output_dir=output_root,
                            eval_only_from=args.eval_only_from,
                            raw_config=raw_config,
                            seed=args.seed,
                            device=args.device,
                        )
                    )
                except FileNotFoundError:
                    if requested_baselines:
                        raise
                    continue
                _write_csv(output_root / "aggregate_results.csv", rows, AGGREGATE_CSV_FIELDS)
                write_json(output_root / "aggregate_results.json", rows)
        if not rows:
            raise FileNotFoundError(
                f"No matching eval-only checkpoints found for problem `{args.problem}` under {args.eval_only_from}."
            )
        summary = {
            "run_id": run_id,
            "problem": args.problem,
            "target": args.target,
            "seed": args.seed,
            "device": args.device,
            "select_on_validation": False,
            "eval_only_from": str(args.eval_only_from),
            "target_count": len(targets),
            "result_count": len(rows),
            "aggregate_csv_path": str(output_root / "aggregate_results.csv"),
            "aggregate_json_path": str(output_root / "aggregate_results.json"),
            "output_dir": str(output_root),
        }
        write_json(output_root / "run_summary.json", summary)
        return summary

    for target in targets:
        for spec in specs:
            rows.append(
                run_baseline_on_target(
                    spec=spec,
                    target=target,
                    output_dir=output_root,
                    raw_config=raw_config,
                    seed=args.seed,
                    device=args.device,
                    select_on_validation=args.select_on_validation,
                )
            )
            _write_csv(output_root / "aggregate_results.csv", rows, AGGREGATE_CSV_FIELDS)
            write_json(output_root / "aggregate_results.json", rows)
    summary = {
        "run_id": run_id,
        "problem": args.problem,
        "target": args.target,
        "seed": args.seed,
        "device": args.device,
        "select_on_validation": bool(args.select_on_validation),
        "target_count": len(targets),
        "result_count": len(rows),
        "aggregate_csv_path": str(output_root / "aggregate_results.csv"),
        "aggregate_json_path": str(output_root / "aggregate_results.json"),
        "output_dir": str(output_root),
    }
    write_json(output_root / "run_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate implemented DasBench ML baselines.")
    parser.add_argument("--problem", required=True, choices=_cli_problem_choices())
    parser.add_argument("--target", default="all", help="Dataset target under --dataset-root/<problem>, or `all`.")
    parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/datasets"))
    parser.add_argument("--dataset-dir", type=Path, action="append", help="Dataset directory containing manifest/train/validation/test.")
    parser.add_argument("--train-split", type=Path, help="Explicit train JSONL path.")
    parser.add_argument("--validation-split", type=Path, help="Explicit validation JSONL path.")
    parser.add_argument("--test-split", type=Path, help="Explicit test JSONL path.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ml_baseline_runs"))
    parser.add_argument("--run-id", help="Output run id. Defaults to a timestamp.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", type=Path, help="JSON or YAML config override.")
    parser.add_argument("--baseline", action="append", help="Baseline name to run. May be repeated.")
    parser.add_argument(
        "--eval-only-from",
        type=Path,
        help="Existing ML baseline run root to load checkpoints from; skips training and evaluates test split only.",
    )
    parser.add_argument(
        "--select-on-validation",
        action="store_true",
        help="Train configs from `search`/`hyperparameter_grid` and select using validation score before test.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_from_args(args)
    print(f"Aggregate CSV: {summary['aggregate_csv_path']}")
    print(f"Output dir: {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
