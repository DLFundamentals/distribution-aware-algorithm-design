from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from dasbench.artifacts import default_agent_run_dir, default_dataset_dir, default_report_dir
from dasbench.data import BenchmarkSpec, generate_dataset, load_manifest, load_spec
from dasbench.families import available_family_names
from dasbench.integrations import load_chat_api_config, load_openai_dotenv
from dasbench.problems import available_problem_names
from dasbench.utils import load_jsonl, timestamp_token, write_json, write_jsonl
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY_PATH = REPO_ROOT / "main.py"
DEFAULT_BENCHMARK_ARTIFACTS_ROOT = Path("artifacts")
REPRESENTATIVE_FAMILY_BY_PROBLEM: dict[str, str] = {
    "coloring": "cluster_ring_mix_v1",
    "maxsat": "latent_backdoor_mixture_v1",
    "mdkp": "latent_class_knapsack_v1",
    "mds": "geometric_cluster_cover_v1",
    "mis": "clique_path_mix_v1",
    "packing_lp": "block_coupled_resource_v1",
    "tsp": "clustered_euclidean_v1",
}
PRIMARY_SIZE_PARAM_BY_PROBLEM: dict[str, str] = {
    "coloring": "num_vertices",
    "maxsat": "num_variables",
    "mdkp": "num_items",
    "mds": "num_vertices",
    "mis": "num_vertices",
    "packing_lp": "num_items",
    "tsp": "num_cities",
}
SIZE_PARAM_KEYS_BY_PROBLEM: dict[str, tuple[str, ...]] = {
    "coloring": ("num_vertices",),
    "maxsat": ("num_variables", "num_clauses"),
    "mdkp": ("num_items", "num_resources"),
    "mds": ("num_vertices",),
    "mis": ("num_vertices",),
    "packing_lp": ("num_items", "num_resources"),
    "tsp": ("num_cities",),
}
_DATASET_PREP_LOCKS: dict[str, threading.Lock] = {}
_DATASET_PREP_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class SweepJob:
    sweep_id: str
    condition_id: str
    problem: str
    family: str
    train_size: int
    validation_size: int
    test_size: int
    artifact_root: Path | None = None
    generator: str = "llm"
    mode: str = "beam"
    iterations: int = 3
    beam_width: int = 3
    candidate_width: int | None = None
    repeats: int = 10
    instance_params: dict[str, object] = field(default_factory=dict)
    family_params: dict[str, object] = field(default_factory=dict)
    family_seed: int = 17
    train_seed: int = 101
    validation_seed: int = 202
    test_seed: int = 303
    compute_optima: bool = True
    force: bool = False
    include_train: bool = False
    gurobi_baseline_enabled: bool = True
    gurobi_time_limit_seconds: float = 60.0
    gurobi_threads: int = 1
    native_exact_time_limit_seconds: float | None = None
    external_exact_baselines: str = "auto"
    external_time_limit_seconds: float = 60.0
    external_threads: int = 1
    external_solver_config: str | None = None
    baseline_workers: int = 1
    shared_dataset_train_size: int | None = None
    shared_dataset_validation_size: int | None = None
    source_dataset_dir: Path | None = None
    env_file: Path | None = None
    skip_baselines: bool = False
    skip_report: bool = False
    condition_metadata: dict[str, object] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return self.condition_id

    @property
    def target_root(self) -> Path:
        artifact_root = self.artifact_root or DEFAULT_BENCHMARK_ARTIFACTS_ROOT
        return artifact_root / "targets" / self.condition_id / self.problem / self.family

    @property
    def dataset_dir(self) -> Path:
        if self.artifact_root is None:
            return default_dataset_dir(self.problem, self.family, self.run_id)
        return self.target_root / "dataset"

    @property
    def agent_run_dir(self) -> Path:
        if self.artifact_root is None:
            return default_agent_run_dir(self.problem, self.family, self.run_id)
        return self.target_root / "agent_run"

    @property
    def report_dir(self) -> Path:
        if self.artifact_root is None:
            return default_report_dir(self.problem, self.family, self.run_id)
        return self.target_root / "report"

    @property
    def report_json_path(self) -> Path:
        return self.report_dir / "benchmark_report.json"

    @property
    def synthesis_summary_path(self) -> Path:
        return self.agent_run_dir / "synthesis_summary.json"

    @property
    def completion_path(self) -> Path:
        return self.synthesis_summary_path if self.skip_report else self.report_json_path

    @property
    def uses_shared_dataset(self) -> bool:
        return self.shared_dataset_train_size is not None or self.shared_dataset_validation_size is not None

    @property
    def uses_reused_dataset(self) -> bool:
        return self.source_dataset_dir is not None

    @property
    def shared_dataset_dir(self) -> Path:
        artifact_root = self.artifact_root or DEFAULT_BENCHMARK_ARTIFACTS_ROOT
        return artifact_root / "shared_datasets" / self.problem / self.family


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("All values must be positive integers.")
    return values


def parse_nonnegative_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("All values must be non-negative integers.")
    return values


def selected_targets(
    problem: str | None,
    family: str | None,
    *,
    representative_only: bool = False,
) -> list[tuple[str, str]]:
    if family and not problem:
        raise ValueError("`--family` requires `--problem`.")
    if problem:
        if problem not in available_problem_names():
            raise ValueError(f"Unknown problem `{problem}`.")
        families = available_family_names(problem)
        if family:
            if family not in families:
                raise ValueError(f"Unknown family `{family}` for problem `{problem}`.")
            return [(problem, family)]
        if representative_only:
            return [(problem, REPRESENTATIVE_FAMILY_BY_PROBLEM[problem])]
        return [(problem, family_name) for family_name in families]
    families_by_problem = available_family_names()
    assert isinstance(families_by_problem, dict)
    if representative_only:
        return [
            (problem_name, REPRESENTATIVE_FAMILY_BY_PROBLEM[problem_name])
            for problem_name in sorted(families_by_problem)
        ]
    return [
        (problem_name, family_name)
        for problem_name in sorted(families_by_problem)
        for family_name in families_by_problem[problem_name]
    ]


def primary_size_value(problem: str, instance_params: dict[str, object]) -> int | None:
    key = PRIMARY_SIZE_PARAM_BY_PROBLEM.get(problem)
    if key is None:
        return None
    value = instance_params.get(key)
    return int(value) if isinstance(value, (int, float)) else None


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sweep-id")
    parser.add_argument("--output-root", default=str(DEFAULT_BENCHMARK_ARTIFACTS_ROOT))
    parser.add_argument("--problem", choices=available_problem_names())
    parser.add_argument("--family")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun completed targets and regenerate datasets.")
    parser.add_argument(
        "--env-file",
        help=(
            "Dotenv file to apply to this sweep and every per-target subprocess. "
            "Values in this file override the current environment and prevent overlapping `.env` values from winning."
        ),
    )
    parser.add_argument(
        "--source-run-root",
        help=(
            "Existing benchmark sweep root whose datasets should be reused, for example "
            "artifacts/second_scale_benchmark_v2/20260427_230552."
        ),
    )
    parser.add_argument(
        "--source-condition-id",
        help="Condition id under --source-run-root/targets to reuse. Defaults to each target condition id.",
    )
    parser.add_argument("--generator", choices=["llm", "llm_no_hint", "template", "auto", "agent"], default="llm")
    parser.add_argument("--mode", choices=["single", "beam"], default="beam")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--include-train", action="store_true")
    parser.add_argument("--family-seed", type=int, default=17)
    parser.add_argument("--train-seed", type=int, default=101)
    parser.add_argument("--validation-seed", type=int, default=202)
    parser.add_argument("--test-seed", type=int, default=303)
    parser.add_argument("--no-compute-optima", dest="compute_optima", action="store_false", default=True)
    parser.add_argument("--no-gurobi-baseline", dest="gurobi_baseline_enabled", action="store_false", default=True)
    parser.add_argument("--gurobi-time-limit-seconds", type=float, default=60.0)
    parser.add_argument("--gurobi-threads", type=int, default=1)
    parser.add_argument("--native-exact-time-limit-seconds", type=float, default=None)
    parser.add_argument("--external-exact-baselines", choices=["auto", "off", "required"], default="auto")
    parser.add_argument("--external-time-limit-seconds", type=float, default=60.0)
    parser.add_argument("--external-threads", type=int, default=1)
    parser.add_argument("--external-solver-config")
    parser.add_argument("--baseline-workers", type=int, default=1)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-report", action="store_true")


def resolve_sweep_artifact_root(output_root: str | Path, sweep_kind: str, sweep_id: str) -> Path:
    return Path(output_root) / sweep_kind / sweep_id


def benchmark_command(job: SweepJob) -> list[str]:
    command = [
        sys.executable,
        str(MAIN_PY_PATH),
        "benchmark",
        "--problem",
        job.problem,
        "--family",
        job.family,
        "--dataset-id",
        job.run_id,
        "--run-id",
        job.run_id,
        "--dataset-dir",
        str(job.dataset_dir),
        "--run-output-dir",
        str(job.agent_run_dir),
        "--report-output-dir",
        str(job.report_dir),
        "--generator",
        job.generator,
        "--mode",
        job.mode,
        "--iterations",
        str(job.iterations),
        "--beam-width",
        str(job.beam_width),
        "--repeats",
        str(job.repeats),
        "--train-size",
        str(job.train_size),
        "--validation-size",
        str(job.validation_size),
        "--test-size",
        str(job.test_size),
        "--family-seed",
        str(job.family_seed),
        "--train-seed",
        str(job.train_seed),
        "--validation-seed",
        str(job.validation_seed),
        "--test-seed",
        str(job.test_seed),
    ]
    if job.candidate_width is not None:
        command.extend(["--candidate-width", str(job.candidate_width)])
    if job.force and not job.uses_shared_dataset and not job.uses_reused_dataset:
        command.append("--force-regenerate")
    if job.include_train:
        command.append("--include-train")
    if not job.compute_optima:
        command.append("--no-compute-optima")
    if not job.gurobi_baseline_enabled:
        command.append("--no-gurobi-baseline")
    command.extend(["--gurobi-time-limit-seconds", str(job.gurobi_time_limit_seconds)])
    command.extend(["--gurobi-threads", str(job.gurobi_threads)])
    if job.native_exact_time_limit_seconds is not None:
        command.extend(["--native-exact-time-limit-seconds", str(job.native_exact_time_limit_seconds)])
    command.extend(["--external-exact-baselines", job.external_exact_baselines])
    command.extend(["--external-time-limit-seconds", str(job.external_time_limit_seconds)])
    command.extend(["--external-threads", str(job.external_threads)])
    command.extend(["--baseline-workers", str(job.baseline_workers)])
    if job.external_solver_config:
        command.extend(["--external-solver-config", job.external_solver_config])
    if job.skip_baselines:
        command.append("--skip-baselines")
    if job.skip_report:
        command.append("--skip-report")
    for key, value in sorted(job.instance_params.items()):
        command.extend(["--instance-param", f"{key}={json.dumps(value)}"])
    for key, value in sorted(job.family_params.items()):
        command.extend(["--family-param", f"{key}={json.dumps(value)}"])
    return command


def log_path_for_job(output_dir: Path, job: SweepJob) -> Path:
    return output_dir / "logs" / job.condition_id / f"{job.problem}__{job.family}.log"


def _dataset_prep_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _DATASET_PREP_LOCKS_GUARD:
        lock = _DATASET_PREP_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _DATASET_PREP_LOCKS[key] = lock
        return lock


def _spec_for_job(job: SweepJob, *, train_size: int, validation_size: int | None = None) -> BenchmarkSpec:
    return BenchmarkSpec(
        problem=job.problem,
        family=job.family,
        instance_params=dict(job.instance_params),
        family_params=dict(job.family_params),
        split_sizes={
            "train": train_size,
            "validation": job.validation_size if validation_size is None else validation_size,
            "test": job.test_size,
        },
        seeds={
            "family": job.family_seed,
            "train": job.train_seed,
            "validation": job.validation_seed,
            "test": job.test_seed,
        },
        compute_optima=job.compute_optima,
    )


def _link_or_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _source_dataset_dir(source_run_root: Path, source_condition_id: str, problem: str, family: str) -> Path:
    return source_run_root / "targets" / source_condition_id / problem / family / "dataset"


def _target_reused_dataset_is_current(target_dir: Path, *, source_dir: Path) -> bool:
    required_files = (
        "manifest.json",
        "benchmark_spec.json",
        "reproducibility.json",
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
    )
    if not all((target_dir / name).exists() for name in required_files):
        return False
    try:
        target_spec = json.loads((target_dir / "benchmark_spec.json").read_text(encoding="utf-8"))
        source_spec = json.loads((source_dir / "benchmark_spec.json").read_text(encoding="utf-8"))
        target_manifest = json.loads((target_dir / "manifest.json").read_text(encoding="utf-8"))
        source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        target_spec == source_spec
        and target_manifest.get("problem") == source_manifest.get("problem")
        and target_manifest.get("family") == source_manifest.get("family")
        and target_manifest.get("split_sizes") == source_manifest.get("split_sizes")
        and target_manifest.get("reused_dataset_source") == str(source_dir)
    )


def _validate_reused_source_dataset(job: SweepJob, source_dir: Path) -> dict[str, object]:
    required_files = (
        "manifest.json",
        "benchmark_spec.json",
        "reproducibility.json",
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
    )
    missing = [name for name in required_files if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Source dataset {source_dir} is missing required files: {', '.join(missing)}")

    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("problem") != job.problem or manifest.get("family") != job.family:
        raise RuntimeError(
            f"Source dataset {source_dir} is for {manifest.get('problem')}/{manifest.get('family')}, "
            f"but job expects {job.problem}/{job.family}."
        )
    split_sizes = manifest.get("split_sizes", {})
    expected_split_sizes = {
        "train": job.train_size,
        "validation": job.validation_size,
        "test": job.test_size,
    }
    if split_sizes != expected_split_sizes:
        raise RuntimeError(
            f"Source dataset {source_dir} has split sizes {split_sizes}, "
            f"but job expects {expected_split_sizes}."
        )
    return manifest


def _materialize_reused_dataset(job: SweepJob, *, source_dir: Path) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source_dir}")
    if not job.force and _target_reused_dataset_is_current(job.dataset_dir, source_dir=source_dir):
        return

    manifest = _validate_reused_source_dataset(job, source_dir)
    shutil.rmtree(job.dataset_dir, ignore_errors=True)
    job.dataset_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("benchmark_spec.json", "reproducibility.json", "train.jsonl", "validation.jsonl", "test.jsonl"):
        _link_or_copy_file(source_dir / filename, job.dataset_dir / filename)

    reused_manifest = json.loads(json.dumps(manifest))
    reused_manifest["artifact_paths"] = {
        "dataset_dir": str(job.dataset_dir),
        "splits": {
            "train": str(job.dataset_dir / "train.jsonl"),
            "validation": str(job.dataset_dir / "validation.jsonl"),
            "test": str(job.dataset_dir / "test.jsonl"),
        },
        "manifest": str(job.dataset_dir / "manifest.json"),
        "benchmark_spec": str(job.dataset_dir / "benchmark_spec.json"),
        "reproducibility": str(job.dataset_dir / "reproducibility.json"),
    }
    reused_manifest["reused_dataset_source"] = str(source_dir)
    write_json(job.dataset_dir / "manifest.json", reused_manifest)


def _materialize_dataset_view(job: SweepJob, *, shared_dataset_dir: Path) -> None:
    target_dir = job.dataset_dir
    expected_spec = _spec_for_job(job, train_size=job.train_size, validation_size=job.validation_size)
    expected_record = expected_spec.to_reproducibility_record()
    if (
        not job.force
        and (target_dir / "manifest.json").exists()
        and (target_dir / "benchmark_spec.json").exists()
        and (target_dir / "train.jsonl").exists()
        and (target_dir / "validation.jsonl").exists()
        and (target_dir / "test.jsonl").exists()
    ):
        observed_record = json.loads((target_dir / "benchmark_spec.json").read_text(encoding="utf-8"))
        if observed_record == expected_record:
            return

    shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    shared_manifest = load_manifest(shared_dataset_dir)
    shared_train_rows = load_jsonl(shared_dataset_dir / "train.jsonl")
    shared_validation_rows = load_jsonl(shared_dataset_dir / "validation.jsonl")
    if len(shared_train_rows) < job.train_size:
        raise RuntimeError(
            f"Shared dataset {shared_dataset_dir} has only {len(shared_train_rows)} train rows, "
            f"but {job.train_size} are required."
        )
    if len(shared_validation_rows) < job.validation_size:
        raise RuntimeError(
            f"Shared dataset {shared_dataset_dir} has only {len(shared_validation_rows)} validation rows, "
            f"but {job.validation_size} are required."
        )

    if job.train_size == len(shared_train_rows):
        _link_or_copy_file(shared_dataset_dir / "train.jsonl", target_dir / "train.jsonl")
    else:
        write_jsonl(target_dir / "train.jsonl", shared_train_rows[: job.train_size])
    if job.validation_size == len(shared_validation_rows):
        _link_or_copy_file(shared_dataset_dir / "validation.jsonl", target_dir / "validation.jsonl")
    else:
        write_jsonl(target_dir / "validation.jsonl", shared_validation_rows[: job.validation_size])
    _link_or_copy_file(shared_dataset_dir / "test.jsonl", target_dir / "test.jsonl")

    manifest = json.loads(json.dumps(shared_manifest))
    split_sizes = dict(manifest.get("split_sizes", {}))
    split_sizes["train"] = job.train_size
    split_sizes["validation"] = job.validation_size
    split_sizes["test"] = job.test_size
    manifest["split_sizes"] = split_sizes
    manifest["artifact_paths"] = {
        "dataset_dir": str(target_dir),
        "splits": {
            "train": str(target_dir / "train.jsonl"),
            "validation": str(target_dir / "validation.jsonl"),
            "test": str(target_dir / "test.jsonl"),
        },
        "manifest": str(target_dir / "manifest.json"),
        "benchmark_spec": str(target_dir / "benchmark_spec.json"),
        "reproducibility": str(target_dir / "reproducibility.json"),
    }
    manifest["shared_dataset_source"] = str(shared_dataset_dir)
    write_json(target_dir / "manifest.json", manifest)
    write_json(target_dir / "benchmark_spec.json", expected_record)
    write_json(target_dir / "reproducibility.json", expected_record)


def _prepare_shared_dataset(job: SweepJob) -> None:
    if not job.uses_shared_dataset:
        return
    if job.uses_reused_dataset:
        raise RuntimeError("A sweep job cannot use both a source dataset and a generated shared dataset.")
    assert job.shared_dataset_train_size is not None
    resolved_shared_validation_size = (
        job.shared_dataset_validation_size if job.shared_dataset_validation_size is not None else job.validation_size
    )
    if job.shared_dataset_train_size < job.train_size:
        raise RuntimeError(
            f"Shared dataset train size {job.shared_dataset_train_size} is smaller than requested train size {job.train_size}."
        )
    if resolved_shared_validation_size < job.validation_size:
        raise RuntimeError(
            f"Shared dataset validation size {resolved_shared_validation_size} is smaller than requested "
            f"validation size {job.validation_size}."
        )

    shared_dataset_dir = job.shared_dataset_dir
    expected_spec = _spec_for_job(
        job,
        train_size=job.shared_dataset_train_size,
        validation_size=resolved_shared_validation_size,
    )
    expected_record = expected_spec.to_reproducibility_record()
    lock = _dataset_prep_lock(shared_dataset_dir)
    with lock:
        spec_path = shared_dataset_dir / "benchmark_spec.json"
        manifest_path = shared_dataset_dir / "manifest.json"
        if job.force:
            shutil.rmtree(shared_dataset_dir, ignore_errors=True)
            generate_dataset(shared_dataset_dir, expected_spec)
        else:
            spec_mismatch = False
            if spec_path.exists():
                spec_mismatch = load_spec(shared_dataset_dir) != expected_record
            elif shared_dataset_dir.exists() and any(shared_dataset_dir.iterdir()):
                spec_mismatch = True
            if spec_mismatch:
                shutil.rmtree(shared_dataset_dir, ignore_errors=True)
            if spec_mismatch or not manifest_path.exists():
                generate_dataset(shared_dataset_dir, expected_spec)
        _materialize_dataset_view(job, shared_dataset_dir=shared_dataset_dir)


def _prepare_reused_dataset(job: SweepJob) -> None:
    if job.source_dataset_dir is None:
        return
    lock = _dataset_prep_lock(job.dataset_dir)
    with lock:
        _materialize_reused_dataset(job, source_dir=job.source_dataset_dir)


@contextmanager
def _temporary_environ(env: dict[str, str]):
    previous = os.environ.copy()
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def _env_file_for_jobs(jobs: list[SweepJob]) -> Path | None:
    env_files = {job.env_file for job in jobs if job.env_file is not None}
    if len(env_files) > 1:
        raise RuntimeError(f"Expected one env file per sweep, got: {sorted(str(path) for path in env_files)}")
    return next(iter(env_files), None)


def _env_with_file(env_file: Path) -> dict[str, str]:
    if not env_file.exists():
        raise FileNotFoundError(f"Env file does not exist: {env_file}")
    parsed = dotenv_values(env_file)
    env = os.environ.copy()
    for key, value in parsed.items():
        if value is not None:
            env[str(key)] = str(value)
    return env


def _prepare_sweep_env(jobs: list[SweepJob], *, require_chat_config: bool) -> dict[str, str]:
    env_file = _env_file_for_jobs(jobs)
    if env_file is None:
        load_openai_dotenv()
        if require_chat_config:
            load_chat_api_config(required=True)
        return os.environ.copy()

    env = _env_with_file(env_file)
    with _temporary_environ(env):
        load_openai_dotenv()
        if require_chat_config:
            load_chat_api_config(required=True)
        return os.environ.copy()


def run_job(job: SweepJob, *, output_dir: Path, dry_run: bool, env: dict[str, str] | None = None) -> dict[str, object]:
    command = benchmark_command(job)
    log_path = log_path_for_job(output_dir, job)
    if job.completion_path.exists() and not job.force:
        return _job_result(job, command, log_path, returncode=0, status="skipped")
    if dry_run:
        return _job_result(job, command, log_path, returncode=0, status="dry_run")

    _prepare_reused_dataset(job)
    _prepare_shared_dataset(job)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Command: {' '.join(command)}\n\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env or os.environ.copy(),
        )
    status = "completed" if completed.returncode == 0 and job.completion_path.exists() else "failed"
    return _job_result(job, command, log_path, returncode=completed.returncode, status=status)


def _job_result(
    job: SweepJob,
    command: list[str],
    log_path: Path,
    *,
    returncode: int,
    status: str,
) -> dict[str, object]:
    return {
        "sweep_id": job.sweep_id,
        "condition_id": job.condition_id,
        "problem": job.problem,
        "family": job.family,
        "dataset_dir": str(job.dataset_dir),
        "agent_run_dir": str(job.agent_run_dir),
        "report_dir": str(job.report_dir),
        "report_json_path": str(job.report_json_path),
        "source_dataset_dir": str(job.source_dataset_dir) if job.source_dataset_dir is not None else None,
        "env_file": str(job.env_file) if job.env_file is not None else None,
        "log_path": str(log_path),
        "command": command,
        "returncode": returncode,
        "status": status,
        **{f"condition_{key}": value for key, value in job.condition_metadata.items()},
    }


def aggregate_rows(jobs: list[SweepJob], results: list[dict[str, object]]) -> list[dict[str, object]]:
    job_by_key = {(job.condition_id, job.problem, job.family): job for job in jobs}
    rows: list[dict[str, object]] = []
    for result in sorted(results, key=lambda item: (str(item["condition_id"]), str(item["problem"]), str(item["family"]))):
        job = job_by_key[(str(result["condition_id"]), str(result["problem"]), str(result["family"]))]
        row = dict(result)
        row.update({
            "train_size": job.train_size,
            "validation_size": job.validation_size,
            "test_size": job.test_size,
            "generator": job.generator,
            "mode": job.mode,
            "iterations": job.iterations,
            "beam_width": job.beam_width,
            "candidate_width": job.candidate_width,
            "repeats": job.repeats,
            "instance_params": dict(job.instance_params),
            "primary_size_param": PRIMARY_SIZE_PARAM_BY_PROBLEM.get(job.problem),
            "primary_size_value": primary_size_value(job.problem, job.instance_params),
            "skip_baselines": job.skip_baselines,
            "skip_report": job.skip_report,
        })
        if job.report_json_path.exists():
            row.update(_metrics_from_report(job.report_json_path, job))
        elif job.synthesis_summary_path.exists():
            row.update(_metrics_from_synthesis(job.synthesis_summary_path))
        rows.append(row)
    return rows


def _metrics_from_report(report_path: Path, job: SweepJob) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    best_candidate = report.get("best_candidate", {})
    test = best_candidate.get("test", {}) if isinstance(best_candidate, dict) else {}
    validation = best_candidate.get("validation", {}) if isinstance(best_candidate, dict) else {}
    hypothesis = best_candidate.get("hypothesis", {}) if isinstance(best_candidate, dict) else {}
    split_reports = report.get("split_reports", {})
    test_split = split_reports.get("test", {}) if isinstance(split_reports, dict) else {}
    gurobi_row = test_split.get("gurobi_timed", {}) if isinstance(test_split, dict) else {}
    best_baseline_name, best_baseline = _best_baseline(test_split, str(best_candidate.get("slug", "")))
    synthesis = _read_json(Path(report.get("agent_run_dir", job.agent_run_dir)) / "synthesis_summary.json")
    search_stats = _search_stats(synthesis)
    return {
        "agent_slug": best_candidate.get("slug"),
        "hypothesis_title": hypothesis.get("title") if isinstance(hypothesis, dict) else None,
        "hypothesis_diversity_key": hypothesis.get("diversity_key") if isinstance(hypothesis, dict) else None,
        "hypothesis_rule_summary": hypothesis.get("rule_summary") if isinstance(hypothesis, dict) else None,
        "agent_test_quality": _metric(test, "average_normalized_quality"),
        "agent_test_optimality": _metric(test, "optimality_rate"),
        "agent_test_feasibility": _metric(test, "feasibility_rate"),
        "agent_test_runtime_ms": _metric(test, "average_runtime_ms"),
        "agent_validation_quality": _metric(validation, "average_normalized_quality"),
        "agent_validation_optimality": _metric(validation, "optimality_rate"),
        "agent_validation_feasibility": _metric(validation, "feasibility_rate"),
        "agent_validation_runtime_ms": _metric(validation, "average_runtime_ms"),
        "gurobi_test_quality": _metric(gurobi_row, "average_normalized_quality"),
        "gurobi_test_runtime_ms": _metric(gurobi_row, "average_runtime_ms"),
        "gurobi_test_internal_runtime_ms": _metric(gurobi_row, "average_gurobi_runtime_ms"),
        "best_baseline_name": best_baseline_name,
        "best_baseline_quality": _metric(best_baseline, "average_normalized_quality"),
        "best_baseline_runtime_ms": _metric(best_baseline, "average_runtime_ms"),
        **search_stats,
    }


def _metrics_from_synthesis(synthesis_path: Path) -> dict[str, object]:
    synthesis = _read_json(synthesis_path)
    best_candidate = synthesis.get("best_candidate", {})
    train = best_candidate.get("train", {}) if isinstance(best_candidate, dict) else {}
    validation = best_candidate.get("validation", {}) if isinstance(best_candidate, dict) else {}
    test = best_candidate.get("test", {}) if isinstance(best_candidate, dict) else {}
    hypothesis = best_candidate.get("hypothesis", {}) if isinstance(best_candidate, dict) else {}
    return {
        "agent_slug": best_candidate.get("slug"),
        "hypothesis_title": hypothesis.get("title") if isinstance(hypothesis, dict) else None,
        "hypothesis_diversity_key": hypothesis.get("diversity_key") if isinstance(hypothesis, dict) else None,
        "hypothesis_rule_summary": hypothesis.get("rule_summary") if isinstance(hypothesis, dict) else None,
        "agent_train_quality": _metric(train, "average_normalized_quality"),
        "agent_train_optimality": _metric(train, "optimality_rate"),
        "agent_train_feasibility": _metric(train, "feasibility_rate"),
        "agent_train_runtime_ms": _metric(train, "average_runtime_ms"),
        "agent_validation_quality": _metric(validation, "average_normalized_quality"),
        "agent_validation_optimality": _metric(validation, "optimality_rate"),
        "agent_validation_feasibility": _metric(validation, "feasibility_rate"),
        "agent_validation_runtime_ms": _metric(validation, "average_runtime_ms"),
        "agent_test_quality": _metric(test, "average_normalized_quality"),
        "agent_test_optimality": _metric(test, "optimality_rate"),
        "agent_test_feasibility": _metric(test, "feasibility_rate"),
        "agent_test_runtime_ms": _metric(test, "average_runtime_ms"),
        "gurobi_test_quality": None,
        "gurobi_test_runtime_ms": None,
        "gurobi_test_internal_runtime_ms": None,
        "best_baseline_name": None,
        "best_baseline_quality": None,
        "best_baseline_runtime_ms": None,
        **_search_stats(synthesis),
    }


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: object, name: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(name)
    if value is None:
        value = payload.get(f"{name}_mean")
    return float(value) if isinstance(value, (int, float)) else None


def _best_baseline(split_payload: object, agent_slug: str) -> tuple[str | None, dict[str, object]]:
    if not isinstance(split_payload, dict):
        return None, {}
    candidates = [
        (name, payload)
        for name, payload in split_payload.items()
        if name != agent_slug and isinstance(payload, dict)
    ]
    if not candidates:
        return None, {}
    name, payload = max(
        candidates,
        key=lambda item: (
            _metric(item[1], "average_normalized_quality") or 0.0,
            _metric(item[1], "optimality_rate") or 0.0,
            -(_metric(item[1], "average_runtime_ms") or float("inf")),
        ),
    )
    return name, payload


def _search_stats(synthesis: dict[str, object]) -> dict[str, object]:
    rounds = synthesis.get("rounds", [])
    if not isinstance(rounds, list):
        rounds = []
    evaluated: set[str] = set()
    diversity_keys: list[str] = []
    for round_record in rounds:
        if not isinstance(round_record, dict):
            continue
        evaluated.update(str(slug) for slug in round_record.get("evaluated_this_round", []))
        diversity_keys.extend(str(key) for key in round_record.get("frontier_diversity_keys", []))
    return {
        "evaluated_candidate_count": len(evaluated),
        "rounds_completed": len(rounds),
        "frontier_diversity_keys": sorted(set(diversity_keys)),
    }


def write_aggregate_outputs(
    *,
    output_dir: Path,
    sweep_id: str,
    sweep_kind: str,
    rows: list[dict[str, object]],
    results: list[dict[str, object]],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "aggregate_results.json"
    csv_path = output_dir / "aggregate_results.csv"
    summary_path = output_dir / "benchmark_sweep_summary.json"
    write_json(json_path, rows)
    _write_csv(csv_path, rows)
    summary = {
        "sweep_id": sweep_id,
        "sweep_kind": sweep_kind,
        "created_at": timestamp_token(),
        "target_count": len(results),
        "completed_count": sum(1 for result in results if result["status"] in {"completed", "skipped"}),
        "failed_count": sum(1 for result in results if result["returncode"] != 0 or result["status"] == "failed"),
        "dry_run_count": sum(1 for result in results if result["status"] == "dry_run"),
        "aggregate_json_path": str(json_path),
        "aggregate_csv_path": str(csv_path),
        "targets": results,
    }
    write_json(summary_path, summary)
    return summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def run_sweep(
    *,
    sweep_id: str,
    sweep_kind: str,
    jobs: list[SweepJob],
    output_root: Path,
    max_workers: int,
    dry_run: bool,
) -> dict[str, object]:
    subprocess_env = _prepare_sweep_env(
        jobs,
        require_chat_config=any(job.generator in {"llm", "llm_no_hint", "agent"} for job in jobs) and not dry_run,
    )

    output_dir = resolve_sweep_artifact_root(output_root, sweep_kind, sweep_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_workers = max(1, min(max_workers, len(jobs) or 1))
    print(f"Running {sweep_kind} `{sweep_id}` with {len(jobs)} targets and max_workers={max_workers}")

    if dry_run:
        precomputed_results: list[dict[str, object]] = []
        pending_jobs = jobs
    else:
        precomputed_results = [
            _job_result(
                job,
                benchmark_command(job),
                log_path_for_job(output_dir, job),
                returncode=0,
                status="skipped",
            )
            for job in jobs
            if job.completion_path.exists() and not job.force
        ]
        pending_jobs = [
            job
            for job in jobs
            if not (job.completion_path.exists() and not job.force)
        ]

    results: list[dict[str, object]] = list(precomputed_results)
    status_counts = {"completed": 0, "failed": 0, "skipped": 0, "dry_run": 0}
    for result in precomputed_results:
        status = str(result["status"])
        if status in status_counts:
            status_counts[status] += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_job, job, output_dir=output_dir, dry_run=dry_run, env=subprocess_env): job
            for job in pending_jobs
        }
        with tqdm(
            total=len(jobs),
            desc=f"{sweep_kind}:{sweep_id}",
            unit="target",
            dynamic_ncols=True,
            initial=len(precomputed_results),
        ) as progress:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                status = str(result["status"])
                if status in status_counts:
                    status_counts[status] += 1
                progress.update(1)
                progress.set_postfix(status_counts, refresh=False)

    rows = aggregate_rows(jobs, results)
    summary = write_aggregate_outputs(
        output_dir=output_dir,
        sweep_id=sweep_id,
        sweep_kind=sweep_kind,
        rows=rows,
        results=sorted(results, key=lambda item: (str(item["condition_id"]), str(item["problem"]), str(item["family"]))),
    )
    legacy_output_dir = output_root / sweep_id
    if dry_run and output_root != DEFAULT_BENCHMARK_ARTIFACTS_ROOT and legacy_output_dir != output_dir:
        write_aggregate_outputs(
            output_dir=legacy_output_dir,
            sweep_id=sweep_id,
            sweep_kind=sweep_kind,
            rows=rows,
            results=sorted(
                results,
                key=lambda item: (str(item["condition_id"]), str(item["problem"]), str(item["family"])),
            ),
        )
    print(f"Aggregate JSON: {summary['aggregate_json_path']}")
    print(f"Aggregate CSV: {summary['aggregate_csv_path']}")
    return summary


def jobs_from_conditions(
    *,
    sweep_id: str,
    artifact_root: Path,
    targets: list[tuple[str, str]],
    conditions: list[dict[str, object]],
    args: argparse.Namespace,
) -> list[SweepJob]:
    jobs: list[SweepJob] = []
    for condition in conditions:
        condition_id = str(condition["condition_id"])
        for problem, family in targets:
            instance_params_by_problem = condition.get("instance_params_by_problem", {})
            instance_params_by_target = condition.get("instance_params_by_target", {})
            target_key = f"{problem}/{family}"
            instance_params = {}
            if isinstance(instance_params_by_target, dict):
                instance_params = dict(instance_params_by_target.get(target_key, {}))
            if isinstance(instance_params_by_problem, dict):
                merged_instance_params = dict(instance_params_by_problem.get(problem, {}))
                merged_instance_params.update(instance_params)
                instance_params = merged_instance_params
            source_dataset_dir = None
            source_run_root = getattr(args, "source_run_root", None)
            if source_run_root:
                source_condition_id = getattr(args, "source_condition_id", None) or condition_id
                source_dataset_dir = _source_dataset_dir(Path(str(source_run_root)), str(source_condition_id), problem, family)
            env_file = getattr(args, "env_file", None)
            jobs.append(
                SweepJob(
                    artifact_root=artifact_root,
                    sweep_id=sweep_id,
                    condition_id=condition_id,
                    problem=problem,
                    family=family,
                    train_size=int(condition["train_size"]),
                    validation_size=int(condition["validation_size"]),
                    test_size=int(condition["test_size"]),
                    generator=args.generator,
                    mode=args.mode,
                    iterations=int(condition.get("iterations", args.iterations)),
                    beam_width=int(condition.get("beam_width", args.beam_width)),
                    candidate_width=(
                        int(condition["candidate_width"])
                        if condition.get("candidate_width") is not None
                        else None
                    ),
                    repeats=args.repeats,
                    instance_params=instance_params,
                    family_seed=args.family_seed,
                    train_seed=args.train_seed,
                    validation_seed=args.validation_seed,
                    test_seed=args.test_seed,
                    compute_optima=args.compute_optima,
                    force=args.force,
                    include_train=args.include_train,
                    gurobi_baseline_enabled=args.gurobi_baseline_enabled,
                    gurobi_time_limit_seconds=args.gurobi_time_limit_seconds,
                    gurobi_threads=args.gurobi_threads,
                    native_exact_time_limit_seconds=args.native_exact_time_limit_seconds,
                    external_exact_baselines=args.external_exact_baselines,
                    external_time_limit_seconds=args.external_time_limit_seconds,
                    external_threads=args.external_threads,
                    external_solver_config=args.external_solver_config,
                    baseline_workers=args.baseline_workers,
                    shared_dataset_train_size=(
                        int(condition["shared_dataset_train_size"])
                        if condition.get("shared_dataset_train_size") is not None
                        else None
                    ),
                    shared_dataset_validation_size=(
                        int(condition["shared_dataset_validation_size"])
                        if condition.get("shared_dataset_validation_size") is not None
                        else None
                    ),
                    source_dataset_dir=source_dataset_dir,
                    env_file=Path(str(env_file)) if env_file else None,
                    skip_baselines=bool(getattr(args, "skip_baselines", False) or condition.get("skip_baselines", False)),
                    skip_report=bool(getattr(args, "skip_report", False) or condition.get("skip_report", False)),
                    condition_metadata={
                        key: value
                        for key, value in condition.items()
                        if key
                        not in {
                            "instance_params_by_problem",
                            "instance_params_by_target",
                            "shared_dataset_train_size",
                            "shared_dataset_validation_size",
                        }
                    },
                )
            )
    return jobs


def load_aggregate_rows(summary: dict[str, object]) -> list[dict[str, object]]:
    aggregate_json_path = Path(str(summary["aggregate_json_path"]))
    payload = json.loads(aggregate_json_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _runtime_plot_axes(
    *,
    axis: Any,
    x_values: list[float],
    x_label: str,
    y_label: str,
    x_scale: str,
) -> None:
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.25)
    if x_scale == "sample_symlog":
        axis.set_xscale("symlog", base=2, linthresh=1.0)
        axis.set_xticks(x_values)
        axis.set_xticklabels([str(int(value)) for value in x_values])
    else:
        axis.set_xticks(x_values)
        axis.set_xticklabels([str(int(value)) if float(value).is_integer() else f"{value:g}" for value in x_values])


def write_agent_runtime_plots(
    *,
    output_dir: Path,
    rows: list[dict[str, object]],
    sweep_name: str,
    x_field: str,
    x_label: str,
    x_scale: str = "linear",
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row.get("status")) not in {"completed", "skipped"}:
            continue
        if row.get("agent_test_runtime_ms") is None:
            continue
        grouped[(str(row["problem"]), str(row["family"]))].append(row)

    written: list[Path] = []
    for (problem, family), target_rows in sorted(grouped.items()):
        points = sorted(
            (
                float(row[x_field]),
                float(row["agent_test_runtime_ms"]) / 1000.0,
            )
            for row in target_rows
            if isinstance(row.get(x_field), (int, float)) and isinstance(row.get("agent_test_runtime_ms"), (int, float))
        )
        if not points:
            continue
        figure, axis = plt.subplots(figsize=(7.2, 4.5))
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        axis.plot(x_values, y_values, marker="o", linewidth=2.4, color="#2563eb", label="agent")
        axis.legend()
        axis.set_title(f"{sweep_name}: {problem} / {family}")
        _runtime_plot_axes(
            axis=axis,
            x_values=x_values,
            x_label=x_label,
            y_label="Runtime (s)",
            x_scale=x_scale,
        )
        path = plots_dir / f"{problem}__{family}__runtime.png"
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)
        written.append(path)
    return written


def write_problem_size_runtime_plots(
    *,
    output_dir: Path,
    rows: list[dict[str, object]],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row.get("status")) not in {"completed", "skipped"}:
            continue
        report_path = row.get("report_json_path")
        if not isinstance(report_path, str) or not Path(report_path).exists():
            continue
        grouped[(str(row["problem"]), str(row["family"]))].append(row)

    written: list[Path] = []
    for (problem, family), target_rows in sorted(grouped.items()):
        series: dict[str, list[tuple[float, float]]] = defaultdict(list)
        x_label_param = str(target_rows[0].get("primary_size_param") or "problem_size")
        for row in target_rows:
            report = _read_json(Path(str(row["report_json_path"])))
            test_split = report.get("split_reports", {}).get("test", {})
            if not isinstance(test_split, dict):
                continue
            x_value = row.get("primary_size_value")
            if not isinstance(x_value, (int, float)):
                continue
            agent_slug = report.get("best_candidate", {}).get("slug")
            for solver_name, payload in test_split.items():
                runtime_ms = _metric(payload, "average_runtime_ms")
                if runtime_ms is None:
                    continue
                label = "agent" if solver_name == agent_slug else str(solver_name)
                series[label].append((float(x_value), float(runtime_ms) / 1000.0))
        if not series:
            continue
        figure, axis = plt.subplots(figsize=(8.0, 4.8))
        ordered_labels = sorted(series, key=lambda label: (label != "agent", label))
        all_x_values = sorted({point[0] for points in series.values() for point in points})
        for label in ordered_labels:
            points = sorted(series[label], key=lambda item: item[0])
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                linewidth=2.8 if label == "agent" else 1.8,
                alpha=1.0 if label == "agent" else 0.9,
                label=label,
            )
        axis.legend(fontsize=8, ncol=2)
        axis.set_title(f"problem_size_sweep: {problem} / {family}")
        _runtime_plot_axes(
            axis=axis,
            x_values=all_x_values,
            x_label=f"Problem Size ({x_label_param})",
            y_label="Runtime (s)",
            x_scale="linear",
        )
        path = plots_dir / f"{problem}__{family}__runtime.png"
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)
        written.append(path)
    return written
