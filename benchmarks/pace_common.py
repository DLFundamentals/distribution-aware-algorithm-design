from __future__ import annotations

import csv
import json
import os
import shlex
import signal
import stat
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ProcessResult:
    stdout: str
    stderr: str
    runtime_ms: float
    timed_out: bool
    exit_code: int | None


@dataclass(frozen=True)
class SolverSpec:
    name: str
    command: list[str]
    cwd: Path | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class SolverRecipe:
    name: str
    competition: str
    repo_url: str
    commit: str
    build_commands: tuple[str, ...]
    executable: str
    run_cwd: str = "."


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def download_url(url: str, target: Path, *, force: bool = False) -> Path:
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        atomic_write_bytes(target, response.read())
    return target


def run_process(
    command: list[str],
    input_path: Path,
    *,
    timeout_seconds: float,
    grace_seconds: float,
    cwd: Path | None = None,
) -> ProcessResult:
    start = time.perf_counter()
    with input_path.open("rb") as stdin:
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd is not None else None,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        timed_out = False
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
    return ProcessResult(
        stdout=stdout_bytes.decode("utf-8", errors="replace"),
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
        runtime_ms=(time.perf_counter() - start) * 1000.0,
        timed_out=timed_out,
        exit_code=exit_code,
    )


def write_csv(path: Path, rows: list[dict[str, object]], *, fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or (rows[0].keys() if rows else []))
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def natural_key(text: str) -> tuple[object, ...]:
    parts: list[object] = []
    current = ""
    in_digits = False
    for char in text:
        char_is_digit = char.isdigit()
        if current and char_is_digit != in_digits:
            parts.append(int(current) if in_digits else current)
            current = char
            in_digits = char_is_digit
        else:
            current += char
            in_digits = char_is_digit
    if current:
        parts.append(int(current) if in_digits else current)
    return tuple(parts)


def clone_and_build_recipe(
    recipe: SolverRecipe,
    *,
    source_root: Path,
    bin_dir: Path,
    force_build: bool = False,
) -> SolverSpec:
    source_root.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    checkout = source_root / recipe.name
    if not checkout.exists():
        subprocess.run(["git", "clone", recipe.repo_url, str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "fetch", "--all", "--tags"], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", recipe.commit], check=True)
    executable = checkout / recipe.executable
    if force_build or not executable.exists():
        for command in recipe.build_commands:
            subprocess.run(command, cwd=checkout, shell=True, check=True, executable="/usr/bin/bash")
    if not executable.exists():
        raise FileNotFoundError(f"Recipe `{recipe.name}` did not produce expected executable: {executable}")
    shim = bin_dir / recipe.name
    run_cwd = checkout / recipe.run_cwd
    shim.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from __future__ import annotations",
                "import os, subprocess, sys",
                f"binary = {str(executable.resolve())!r}",
                f"cwd = {str(run_cwd.resolve())!r}",
                "raise SystemExit(subprocess.run([binary, *sys.argv[1:]], cwd=cwd, env=os.environ, check=False).returncode)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return SolverSpec(
        name=recipe.name,
        command=[str(shim)],
        metadata={
            "repo_url": recipe.repo_url,
            "commit": recipe.commit,
            "source_dir": str(checkout),
            "executable": str(executable),
            "build_commands": list(recipe.build_commands),
        },
    )


def parse_solver_argument(text: str, builtins: dict[str, SolverSpec]) -> SolverSpec:
    if "=" not in text:
        try:
            return builtins[text]
        except KeyError as exc:
            raise ValueError(f"Unknown solver `{text}`. Available: {sorted(builtins)}") from exc
    name, command_text = text.split("=", 1)
    command = shlex.split(command_text)
    if not name.strip() or not command:
        raise ValueError(f"Expected solver spec NAME='command args', got {text!r}.")
    return SolverSpec(name=name.strip(), command=command)
