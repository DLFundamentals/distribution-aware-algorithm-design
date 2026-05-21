from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import Queue, get_context
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from benchmarks.pace2025_dominating_set import (
    DEFAULT_CACHE_DIR,
    PACE_RAW_BASE_URL,
    _private_ds_path,
    _public_ds_path,
    parse_pace_gr_text,
    write_pace_solution,
)
from dasbench.problems import get_problem_definition
from dasbench.utils import public_instance
from dasbench.utils import write_json


DEFAULT_OUTPUT_ROOT = Path("artifacts/pace2025_dominating_set/baseline_comparisons")
DEFAULT_EXPANDED_DIR = Path("artifacts/external/pace2025-instances-expanded")
DEFAULT_DASBENCH_MDS_BASELINES = (
    "high_degree_greedy",
    "fast_marginal_gain_greedy",
    "fast_redundancy_aware",
    "marginal_gain_greedy",
    "redundancy_aware",
)


@dataclass(frozen=True)
class SolverSpec:
    name: str
    command: list[str]
    cwd: Path | None = None


@dataclass(frozen=True)
class PreparedInstance:
    relative_path: str
    input_path: Path
    instance_id: str
    num_vertices: int
    num_edges: int


def _builtin_solver_specs() -> dict[str, SolverSpec]:
    bin_dir = Path("baselines/bin").resolve()
    return {
        "root": SolverSpec("root", [str(bin_dir / "pace2025_root_ds")]),
        "shadoks": SolverSpec("shadoks", [str(bin_dir / "pace2025_shadoks_ds")]),
        "fontanf": SolverSpec("fontanf", [str(bin_dir / "pace2025_fontanf_ds")]),
        "swats": SolverSpec("swats", [str(bin_dir / "pace2025_swats_ds")]),
        "aeg": SolverSpec("aeg", [str(bin_dir / "pace2025_aeg_ds")]),
        "greeduce": SolverSpec("greeduce", [str(bin_dir / "pace2025_greeduce_ds")]),
    }


def _parse_solver_spec(text: str) -> SolverSpec:
    if "=" not in text:
        builtins = _builtin_solver_specs()
        if text not in builtins:
            raise ValueError(f"Unknown built-in solver {text!r}. Available: {sorted(builtins)}")
        return builtins[text]
    name, command_text = text.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Missing solver name in {text!r}.")
    command = shlex.split(command_text)
    if not command:
        raise ValueError(f"Missing solver command in {text!r}.")
    return SolverSpec(name, command)


def _instance_relative_path(*, track: str, source: str, index: int) -> str:
    if source == "private":
        return _private_ds_path(track, index)
    if source == "public":
        return _public_ds_path(track, index)
    raise ValueError(f"Unsupported source {source!r}.")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _download_file(relative_path: str, *, cache_dir: Path, github_ref: str, force: bool = False) -> Path:
    import urllib.request

    target = cache_dir / relative_path
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{PACE_RAW_BASE_URL}/{github_ref}/{relative_path}"
    with urllib.request.urlopen(url) as response:
        _atomic_write_bytes(target, response.read())
    return target


def _extract_gr_from_tar(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source, mode="r:xz") as archive:
        members = [member for member in archive.getmembers() if member.isfile() and member.name.endswith(".gr")]
        if not members:
            raise RuntimeError(f"No .gr member found in {source}.")
        member = sorted(members, key=lambda item: item.name)[0]
        handle = archive.extractfile(member)
        if handle is None:
            raise RuntimeError(f"Could not extract {member.name} from {source}.")
        _atomic_write_bytes(target, handle.read())
    return target


def materialize_gr(relative_path: str, *, cache_dir: Path, expanded_dir: Path, github_ref: str) -> Path:
    source = _download_file(relative_path, cache_dir=cache_dir, github_ref=github_ref)
    if not source.name.endswith(".tar.xz"):
        return source
    target = expanded_dir / relative_path.removesuffix(".tar.xz")
    if target.exists():
        return target
    try:
        return _extract_gr_from_tar(source, target)
    except (EOFError, tarfile.TarError):
        source.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        source = _download_file(relative_path, cache_dir=cache_dir, github_ref=github_ref, force=True)
        return _extract_gr_from_tar(source, target)


def parse_pace_header(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("c"):
                continue
            parts = line.split()
            if len(parts) != 4 or parts[0] != "p" or parts[1] != "ds":
                raise ValueError(f"Expected `p ds n m` header in {path}, got {line!r}.")
            return int(parts[2]), int(parts[3])
    raise ValueError(f"Missing PACE header in {path}.")


def parse_solution(text: str, *, num_vertices: int) -> tuple[list[int], str | None]:
    values: list[int] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("c"):
            continue
        parts = line.split()
        if len(parts) != 1:
            return [], f"Expected one integer per non-comment output line, got {line!r}."
        try:
            values.append(int(parts[0]))
        except ValueError:
            return [], f"Expected integer output line, got {line!r}."
    if not values:
        return [], "Solver produced no solution size line."
    declared_size = values[0]
    raw_vertices = values[1:]
    if declared_size != len(raw_vertices):
        return [], f"Declared solution size {declared_size}, but output listed {len(raw_vertices)} vertices."
    vertices = [vertex - 1 for vertex in raw_vertices]
    seen: set[int] = set()
    for vertex in vertices:
        if not 0 <= vertex < num_vertices:
            return [], f"Vertex {vertex + 1} is outside 1..{num_vertices}."
        if vertex in seen:
            return [], f"Vertex {vertex + 1} is repeated."
        seen.add(vertex)
    return sorted(vertices), None


def verify_dominating_set(path: Path, solution: list[int], *, num_vertices: int) -> tuple[bool, str | None]:
    selected = bytearray(num_vertices)
    dominated = bytearray(num_vertices)
    dominated_count = 0
    for vertex in solution:
        if not selected[vertex]:
            selected[vertex] = 1
        if not dominated[vertex]:
            dominated[vertex] = 1
            dominated_count += 1
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("c") or line.startswith("p"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            u = int(parts[0]) - 1
            v = int(parts[1]) - 1
            if selected[u] and not dominated[v]:
                dominated[v] = 1
                dominated_count += 1
            if selected[v] and not dominated[u]:
                dominated[u] = 1
                dominated_count += 1
    if dominated_count == num_vertices:
        return True, None
    missing = [str(index + 1) for index, value in enumerate(dominated) if not value][:5]
    return False, f"Undominated vertices remain: {', '.join(missing)}"


def run_solver(
    solver: SolverSpec,
    input_path: Path,
    *,
    timeout_seconds: float,
    grace_seconds: float,
) -> tuple[str, str, float, bool, int | None]:
    start = time.perf_counter()
    with input_path.open("rb") as stdin:
        process = subprocess.Popen(
            solver.command,
            cwd=str(solver.cwd) if solver.cwd is not None else None,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        timed_out = False
        exit_code: int | None
        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout_bytes, stderr_bytes = process.communicate(timeout=grace_seconds)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout_bytes, stderr_bytes = process.communicate()
                exit_code = process.returncode
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return (
        stdout_bytes.decode("utf-8", errors="replace"),
        stderr_bytes.decode("utf-8", errors="replace"),
        runtime_ms,
        timed_out,
        exit_code,
    )


def _load_reference_csv(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["pace_source_path"]: row
            for row in csv.DictReader(handle)
            if row.get("pace_source_path")
        }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_solver: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_solver.setdefault(str(row["solver"]), []).append(row)
    summaries: dict[str, dict[str, object]] = {}
    for solver, solver_rows in sorted(by_solver.items()):
        valid_rows = [row for row in solver_rows if row["valid"]]
        solution_sizes = [int(row["solution_size"]) for row in valid_rows]
        runtime_values = [float(row["runtime_ms"]) for row in solver_rows]
        synth_better = sum(
            1
            for row in valid_rows
            if row.get("synth_solution_size") not in ("", None)
            and int(row["synth_solution_size"]) < int(row["solution_size"])
        )
        synth_worse = sum(
            1
            for row in valid_rows
            if row.get("synth_solution_size") not in ("", None)
            and int(row["synth_solution_size"]) > int(row["solution_size"])
        )
        synth_tie = sum(
            1
            for row in valid_rows
            if row.get("synth_solution_size") not in ("", None)
            and int(row["synth_solution_size"]) == int(row["solution_size"])
        )
        summaries[solver] = {
            "num_instances": len(solver_rows),
            "valid_count": len(valid_rows),
            "invalid_count": len(solver_rows) - len(valid_rows),
            "timeout_count": sum(1 for row in solver_rows if row["timed_out"]),
            "total_solution_size": sum(solution_sizes),
            "average_solution_size": sum(solution_sizes) / len(solution_sizes) if solution_sizes else 0.0,
            "average_runtime_ms": sum(runtime_values) / len(runtime_values) if runtime_values else 0.0,
            "synth_better_count": synth_better,
            "synth_worse_count": synth_worse,
            "synth_tie_count": synth_tie,
        }
    return {"solvers": summaries}


def _prepare_instances(
    *,
    relative_paths: list[str],
    cache_dir: Path,
    expanded_dir: Path,
    github_ref: str,
) -> list[PreparedInstance]:
    prepared: list[PreparedInstance] = []
    for relative_path in relative_paths:
        input_path = materialize_gr(
            relative_path,
            cache_dir=cache_dir,
            expanded_dir=expanded_dir,
            github_ref=github_ref,
        )
        num_vertices, num_edges = parse_pace_header(input_path)
        instance_id = Path(relative_path).name.removesuffix(".tar.xz").removesuffix(".gr")
        prepared.append(
            PreparedInstance(
                relative_path=relative_path,
                input_path=input_path,
                instance_id=instance_id,
                num_vertices=num_vertices,
                num_edges=num_edges,
            )
        )
    return prepared


def _run_external_solver_job(
    *,
    solver: SolverSpec,
    prepared: PreparedInstance,
    output_dir: Path,
    timeout_seconds: float,
    grace_seconds: float,
    reference: dict[str, str],
) -> dict[str, object]:
    stdout, stderr, runtime_ms, timed_out, exit_code = run_solver(
        solver,
        prepared.input_path,
        timeout_seconds=timeout_seconds,
        grace_seconds=grace_seconds,
    )
    solution, parse_error = parse_solution(stdout, num_vertices=prepared.num_vertices)
    valid = False
    verify_error = None
    if parse_error is None:
        valid, verify_error = verify_dominating_set(
            prepared.input_path,
            solution,
            num_vertices=prepared.num_vertices,
        )
    error = parse_error or verify_error or ""
    solution_file = output_dir / "solutions" / solver.name / f"{prepared.instance_id}.sol"
    stderr_file = output_dir / "stderr" / solver.name / f"{prepared.instance_id}.stderr.txt"
    solution_file.parent.mkdir(parents=True, exist_ok=True)
    stderr_file.parent.mkdir(parents=True, exist_ok=True)
    solution_file.write_text(stdout, encoding="utf-8")
    stderr_file.write_text(stderr, encoding="utf-8")
    return {
        "solver": solver.name,
        "instance_id": prepared.instance_id,
        "pace_source_path": prepared.relative_path,
        "num_vertices": prepared.num_vertices,
        "num_edges": prepared.num_edges,
        "exit_code": "" if exit_code is None else exit_code,
        "timed_out": timed_out,
        "valid": valid,
        "solution_size": len(solution) if valid else "",
        "runtime_ms": runtime_ms,
        "synth_solution_size": reference.get("solution_size", ""),
        "adapter_reference_objective": reference.get("reference_objective", ""),
        "solution_file": str(solution_file),
        "stderr_file": str(stderr_file),
        "error": error,
    }


def compare_solvers(
    *,
    solvers: list[SolverSpec],
    relative_paths: list[str],
    cache_dir: Path,
    expanded_dir: Path,
    github_ref: str,
    output_dir: Path,
    timeout_seconds: float,
    grace_seconds: float,
    reference_csv: Path | None,
    max_workers: int,
) -> dict[str, object]:
    references = _load_reference_csv(reference_csv)
    prepared_instances = _prepare_instances(
        relative_paths=relative_paths,
        cache_dir=cache_dir,
        expanded_dir=expanded_dir,
        github_ref=github_ref,
    )
    jobs: list[tuple[SolverSpec, PreparedInstance, dict[str, str]]] = []
    for prepared in prepared_instances:
        for solver in solvers:
            jobs.append((solver, prepared, references.get(prepared.relative_path, {})))
    progress_desc = f"PACE external baselines ({len(solvers)} solvers x {len(prepared_instances)} instances)"
    if max_workers == 1:
        rows = [
            _run_external_solver_job(
                solver=solver,
                prepared=prepared,
                output_dir=output_dir,
                timeout_seconds=timeout_seconds,
                grace_seconds=grace_seconds,
                reference=reference,
            )
            for solver, prepared, reference in tqdm(
                jobs,
                desc=progress_desc,
                unit="job",
                dynamic_ncols=True,
            )
        ]
    else:
        rows: list[dict[str, object] | None] = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_external_solver_job,
                    solver=solver,
                    prepared=prepared,
                    output_dir=output_dir,
                    timeout_seconds=timeout_seconds,
                    grace_seconds=grace_seconds,
                    reference=reference,
                ): index
                for index, (solver, prepared, reference) in enumerate(jobs)
            }
            with tqdm(
                total=len(futures),
                desc=progress_desc,
                unit="job",
                dynamic_ncols=True,
            ) as progress:
                for future in as_completed(futures):
                    rows[futures[future]] = future.result()
                    progress.update(1)
        rows = [row for row in rows if row is not None]
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "baseline_results.csv"
    _write_csv(results_csv, rows)
    summary = {
        "schema_version": "pace2025_ds_baseline_comparison.v1",
        "results_csv": str(results_csv),
        "reference_csv": str(reference_csv) if reference_csv is not None else None,
        "timeout_seconds": timeout_seconds,
        "grace_seconds": grace_seconds,
        "max_workers": max_workers,
        "instances": relative_paths,
        **_summarize(rows),
    }
    write_json(output_dir / "baseline_summary.json", summary)
    return summary


def _dasbench_baselines_from_arg(value: str) -> list[str]:
    if value == "heuristic":
        return list(DEFAULT_DASBENCH_MDS_BASELINES)
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_dasbench_baseline(
    *,
    baseline_name: str,
    exposed_instance: dict[str, object],
    timeout_seconds: float,
) -> tuple[list[int], bool, str, float, bool]:
    context = get_context("fork")
    queue: Queue = context.Queue(maxsize=1)
    process = context.Process(target=_run_dasbench_baseline_child, args=(baseline_name, exposed_instance, queue))
    start = time.perf_counter()
    process.start()
    process.join(timeout_seconds)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    if process.is_alive():
        process.terminate()
        process.join(2.0)
        if process.is_alive():
            process.kill()
            process.join()
        return [], False, f"DasBench baseline `{baseline_name}` exceeded {timeout_seconds:.3f}s.", runtime_ms, True
    if queue.empty():
        return [], False, f"DasBench baseline `{baseline_name}` exited without returning a result.", runtime_ms, False
    payload = queue.get()
    if payload["status"] == "ok":
        return payload["solution"], payload["valid"], payload["error"], runtime_ms, False
    return [], False, payload["error"], runtime_ms, False


def _run_dasbench_baseline_child(
    baseline_name: str,
    exposed_instance: dict[str, object],
    queue: Queue,
) -> None:
    problem = get_problem_definition("mds")
    registry = problem.baseline_registry()
    try:
        raw_solution = registry[baseline_name](exposed_instance)
        solution = problem.canonicalize_solution(raw_solution, exposed_instance)
        valid, validation_error = problem.validate_solution(solution, exposed_instance)
        queue.put({"status": "ok", "solution": solution, "valid": valid, "error": validation_error or ""})
    except Exception as exc:
        queue.put({"status": "error", "error": f"{type(exc).__name__}: {exc}"})


def _run_dasbench_baseline_job(
    *,
    baseline_name: str,
    prepared: PreparedInstance,
    output_dir: Path,
    timeout_seconds: float,
    reference: dict[str, str],
) -> dict[str, object]:
    instance = parse_pace_gr_text(
        prepared.input_path.read_text(encoding="utf-8", errors="replace"),
        instance_id=f"pace2025-ds-baseline-{prepared.instance_id}",
        source_path=prepared.relative_path,
    )
    exposed = public_instance(instance)
    solver_name = f"dasbench_{baseline_name}"
    solution, valid, error, runtime_ms, timed_out = _run_dasbench_baseline(
        baseline_name=baseline_name,
        exposed_instance=exposed,
        timeout_seconds=timeout_seconds,
    )

    solution_file = output_dir / "solutions" / solver_name / f"{prepared.instance_id}.sol"
    if valid:
        write_pace_solution(solution_file, solution)
    else:
        solution_file.parent.mkdir(parents=True, exist_ok=True)
        solution_file.write_text("", encoding="utf-8")
    return {
        "solver": solver_name,
        "instance_id": prepared.instance_id,
        "pace_source_path": prepared.relative_path,
        "num_vertices": prepared.num_vertices,
        "num_edges": prepared.num_edges,
        "exit_code": 0 if valid else 124 if timed_out else 1,
        "timed_out": timed_out,
        "valid": valid,
        "solution_size": len(solution) if valid else "",
        "runtime_ms": runtime_ms,
        "synth_solution_size": reference.get("solution_size", ""),
        "adapter_reference_objective": reference.get("reference_objective", ""),
        "solution_file": str(solution_file),
        "stderr_file": "",
        "error": "" if valid else error,
    }


def compare_dasbench_mds_baselines(
    *,
    baseline_names: list[str],
    relative_paths: list[str],
    cache_dir: Path,
    expanded_dir: Path,
    github_ref: str,
    output_dir: Path,
    reference_csv: Path | None,
    timeout_seconds: float,
    max_workers: int,
) -> dict[str, object]:
    references = _load_reference_csv(reference_csv)
    problem = get_problem_definition("mds")
    registry = problem.baseline_registry()
    unknown = [name for name in baseline_names if name not in registry]
    if unknown:
        raise ValueError(f"Unknown DasBench MDS baseline(s): {unknown}. Available: {sorted(registry)}")

    prepared_instances = _prepare_instances(
        relative_paths=relative_paths,
        cache_dir=cache_dir,
        expanded_dir=expanded_dir,
        github_ref=github_ref,
    )
    jobs: list[tuple[str, PreparedInstance, dict[str, str]]] = []
    for prepared in prepared_instances:
        for baseline_name in baseline_names:
            jobs.append((baseline_name, prepared, references.get(prepared.relative_path, {})))

    progress_desc = f"PACE DasBench MDS baselines ({len(baseline_names)} baselines x {len(prepared_instances)} instances)"
    if max_workers == 1:
        rows = [
            _run_dasbench_baseline_job(
                baseline_name=baseline_name,
                prepared=prepared,
                output_dir=output_dir,
                timeout_seconds=timeout_seconds,
                reference=reference,
            )
            for baseline_name, prepared, reference in tqdm(
                jobs,
                desc=progress_desc,
                unit="job",
                dynamic_ncols=True,
            )
        ]
    else:
        rows: list[dict[str, object] | None] = [None] * len(jobs)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_dasbench_baseline_job,
                    baseline_name=baseline_name,
                    prepared=prepared,
                    output_dir=output_dir,
                    timeout_seconds=timeout_seconds,
                    reference=reference,
                ): index
                for index, (baseline_name, prepared, reference) in enumerate(jobs)
            }
            with tqdm(
                total=len(futures),
                desc=progress_desc,
                unit="job",
                dynamic_ncols=True,
            ) as progress:
                for future in as_completed(futures):
                    rows[futures[future]] = future.result()
                    progress.update(1)
        rows = [row for row in rows if row is not None]

    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "baseline_results.csv"
    _write_csv(results_csv, rows)
    summary = {
        "schema_version": "pace2025_ds_dasbench_mds_baseline_comparison.v1",
        "results_csv": str(results_csv),
        "reference_csv": str(reference_csv) if reference_csv is not None else None,
        "timeout_seconds": timeout_seconds,
        "max_workers": max_workers,
        "instances": relative_paths,
        **_summarize(rows),
    }
    write_json(output_dir / "baseline_summary.json", summary)
    return summary


def _instance_paths_from_args(args: argparse.Namespace) -> list[str]:
    if args.instance:
        return list(args.instance)
    end = args.start_index + args.count
    return [
        _instance_relative_path(track=args.track, source=args.source, index=index)
        for index in range(args.start_index, end)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PACE 2025 DS heuristic baseline solvers on selected instances.")
    parser.add_argument("--solver", action="append", default=[], help="Built-in name, or name='command args'.")
    parser.add_argument("--solvers", default="root,shadoks", help="Comma-separated built-in solver names.")
    parser.add_argument(
        "--dasbench-mds-baselines",
        action="store_true",
        help="Run DasBench's in-process MDS baselines instead of external PACE solver binaries.",
    )
    parser.add_argument(
        "--dasbench-baselines",
        default="heuristic",
        help=(
            "Comma-separated DasBench MDS baseline names, or `heuristic` for the local greedy/redundancy baselines. "
            "Used only with --dasbench-mds-baselines."
        ),
    )
    parser.add_argument(
        "--dasbench-timeout-seconds",
        type=float,
        default=300.0,
        help="Per-instance timeout for each in-process DasBench MDS baseline.",
    )
    parser.add_argument("--track", choices=["heuristic", "exact"], default="heuristic")
    parser.add_argument("--source", choices=["private", "public"], default="private")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--instance", action="append", help="Explicit repo-relative instance path.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--expanded-dir", type=Path, default=DEFAULT_EXPANDED_DIR)
    parser.add_argument("--github-ref", default="master")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "latest")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--grace-seconds", type=float, default=20.0)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help=(
            "Maximum concurrent solver/instance jobs. External solvers use subprocess threads; "
            "--dasbench-mds-baselines uses worker processes."
        ),
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("artifacts/pace2025_dominating_set/pace2025_ds_heuristic_llm_01/pace_evaluation/pace_private_results.csv"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1.")
    relative_paths = _instance_paths_from_args(args)
    if args.dasbench_mds_baselines:
        summary = compare_dasbench_mds_baselines(
            baseline_names=_dasbench_baselines_from_arg(args.dasbench_baselines),
            relative_paths=relative_paths,
            cache_dir=args.cache_dir,
            expanded_dir=args.expanded_dir,
            github_ref=args.github_ref,
            output_dir=args.output_dir,
            reference_csv=args.reference_csv,
            timeout_seconds=args.dasbench_timeout_seconds,
            max_workers=args.max_workers,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    solver_texts = list(args.solver)
    if not solver_texts:
        solver_texts = [item.strip() for item in args.solvers.split(",") if item.strip()]
    solvers = [_parse_solver_spec(text) for text in solver_texts]
    missing = []
    for solver in solvers:
        executable = Path(solver.command[0])
        if not executable.is_absolute() and solver.cwd is not None:
            executable = solver.cwd / executable
        if "/" in solver.command[0] and not executable.exists():
            missing.append(solver)
    if missing:
        raise FileNotFoundError(f"Missing solver executable(s): {', '.join(solver.command[0] for solver in missing)}")
    summary = compare_solvers(
        solvers=solvers,
        relative_paths=relative_paths,
        cache_dir=args.cache_dir,
        expanded_dir=args.expanded_dir,
        github_ref=args.github_ref,
        output_dir=args.output_dir,
        timeout_seconds=args.timeout_seconds,
        grace_seconds=args.grace_seconds,
        reference_csv=args.reference_csv,
        max_workers=args.max_workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
