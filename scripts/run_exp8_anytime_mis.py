#!/usr/bin/env python3
"""Experiment 8 - anytime quality-time curves for KaMIS heuristic modes on MIS.

Addresses the reviewer point that KaMIS is used only as the *exact* backend, while its
heuristic / evolutionary modes (ReduMIS, OnlineMIS) should be compared under shared time
budgets.  For each of the 3 MIS targets we run `redumis` and `online_mis` on all 500 held-out
test instances across a sweep of `--time_limit` budgets, record the achieved independent-set
quality (|IS| / optimum) and the *actual* wall-clock per call, and overlay our synthesized
solver's operating point.

KaMIS reads a METIS graph (reuse `serialize_metis_graph`) and writes the IS as a 0/1-per-line
indicator to `--output`.  Each call runs single-threaded (OMP_NUM_THREADS=1) so the process pool
does not oversubscribe.  Pure evaluation of an external solver -- no training, no API.

Usage:
  python -m scripts.run_exp8_anytime_mis [--max-workers 32] [--test-size 500]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from dasbench.integrations.external_exact import serialize_metis_graph

MAIN = "artifacts/second_scale_benchmark_v2/20260427_230552/targets/seconds_scale_v2"
# KaMIS is built out of tree; point DASBENCH_KAMIS_BUILD_DIR at its build directory.
KAMIS_BUILD_DIR = os.environ.get("DASBENCH_KAMIS_BUILD_DIR", "~/kamis/build")
REDUMIS = os.path.expanduser(f"{KAMIS_BUILD_DIR}/redumis")
ONLINE_MIS = os.path.expanduser(f"{KAMIS_BUILD_DIR}/online_mis")
BINARIES = {"redumis": REDUMIS, "online_mis": ONLINE_MIS}
TARGETS = ["clique_path_mix_v1", "core_fringe_trap_v1", "motif_bridge_mixture_v1"]
BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]  # --time_limit seconds


def _run_one(binary: str, graph_path: str, overhead_ms: float, budget: float, optimum: float) -> tuple[float, float, int]:
    """Return (quality, full_wall_ms, is_size) for one KaMIS invocation.

    ``full_wall_ms`` is the *complete* wall-clock cost of using KaMIS on one instance:
    ``overhead_ms`` (our instance -> METIS serialization + graph file write, measured upstream)
    plus the timed region here -- process startup, graph load, kernelization, solve, output write
    (all inside the subprocess), and reading/parsing the result. This is the full cost our lean
    in-process synthesized solver avoids, timed the same way Gurobi was (wall-clock, not the
    solver's self-reported solve time).
    """
    sol_path = f"{graph_path}.{os.path.basename(binary)}.{budget}.sol"
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    t0 = time.perf_counter()
    try:
        subprocess.run(
            [binary, graph_path, f"--output={sol_path}", "--seed=0", f"--time_limit={budget}", "--disable_checks"],
            capture_output=True, text=True, timeout=budget + 60, env=env, check=False,
        )
    except subprocess.TimeoutExpired:
        return (0.0, overhead_ms + (time.perf_counter() - t0) * 1000.0, 0)
    size = 0
    if os.path.exists(sol_path):
        with open(sol_path) as fh:
            size = sum(1 for line in fh if line.strip() == "1")
        os.unlink(sol_path)
    full_wall_ms = overhead_ms + (time.perf_counter() - t0) * 1000.0
    quality = (size / optimum) if optimum else float("nan")
    return (quality, full_wall_ms, size)


def _task(args):
    method, binary, graph_path, overhead_ms, budget, optimum = args
    quality, wall_ms, size = _run_one(binary, graph_path, overhead_ms, budget, optimum)
    return (method, budget, quality, wall_ms, size)


def _our_operating_points() -> dict[str, dict[str, float]]:
    out = {}
    for fam in TARGETS:
        report = json.loads(Path(f"{MAIN}/mis/{fam}/report/benchmark_report.json").read_text())
        rt = ((report.get("best_candidate_diagnostics") or {}).get("runtime_summary") or {}).get("average_runtime_ms")
        q = (report.get("best_candidate") or {}).get("test", {}).get("average_normalized_quality")
        out[fam] = {"quality": q, "runtime_ms": rt}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-workers", type=int, default=32)
    ap.add_argument("--test-size", type=int, default=500)
    ap.add_argument("--out-root", default="results/exp8_anytime_mis")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.out_root) / ts
    outdir.mkdir(parents=True, exist_ok=True)
    ours = _our_operating_points()
    per_target_rows: list[dict] = []

    for fam in TARGETS:
        instances = [json.loads(line) for line in Path(f"{MAIN}/mis/{fam}/dataset/test.jsonl").read_text().splitlines()]
        instances = instances[: args.test_size]
        print(f"[{fam}] {len(instances)} instances x {len(BUDGETS)} budgets x {len(BINARIES)} methods", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"exp8_{fam}_") as workdir:
            # Serialize + write each instance's graph once, and MEASURE that overhead (our
            # instance -> METIS conversion + file write) so it is charged to KaMIS's full
            # wall-clock. The file is reused across budgets/methods; the identical, deterministic
            # overhead is added to every invocation.
            graphs = []  # (graph_path, overhead_ms, optimum)
            for idx, inst in enumerate(instances):
                gp = os.path.join(workdir, f"inst_{idx:04d}.graph")
                t = time.perf_counter()
                Path(gp).write_text(serialize_metis_graph(inst), encoding="utf-8")
                overhead_ms = (time.perf_counter() - t) * 1000.0
                graphs.append((gp, overhead_ms, float(inst.get("optimum_objective") or 0.0)))
            tasks = [
                (method, binary, gp, oh, budget, opt)
                for method, binary in BINARIES.items()
                for (gp, oh, opt) in graphs
                for budget in BUDGETS
            ]
            # accumulate per (method, budget)
            agg: dict[tuple, list] = {(m, b): [] for m in BINARIES for b in BUDGETS}
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as ex:
                for method, budget, quality, wall_ms, size in ex.map(_task, tasks, chunksize=8):
                    agg[(method, budget)].append((quality, wall_ms))
            for (method, budget), vals in agg.items():
                qs = [q for q, _ in vals if not math.isnan(q)]
                ws = [w for _, w in vals]
                per_target_rows.append({
                    "target": fam, "method": method, "budget_s": budget,
                    "mean_quality": statistics.fmean(qs) if qs else float("nan"),
                    "mean_wall_ms": statistics.fmean(ws) if ws else float("nan"),
                    "n": len(vals),
                })
        print(f"[{fam}] done", flush=True)

    # ---- write outputs ----
    with (outdir / "anytime_curves.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["target", "method", "budget_s", "mean_quality", "mean_wall_ms", "n"])
        w.writeheader()
        for r in per_target_rows:
            w.writerow(r)

    lines = ["# Experiment 8 - KaMIS ReduMIS / OnlineMIS anytime curves (MIS)", "",
             f"Generated {ts} UTC. Each cell: mean quality (|IS|/optimum) over 500 test instances, "
             "with mean **full wall-clock** per call. `time_limit` is the requested budget; the wall-clock "
             "is the complete cost of using KaMIS on one instance -- instance -> METIS serialization + file "
             "write + process startup + graph load + kernelization + solve + output write + parse -- i.e. "
             "everything our lean in-process synthesized solver avoids, timed the same way Gurobi was "
             "(wall-clock, not the solver's self-reported solve time).", ""]
    for fam in TARGETS:
        o = ours[fam]
        lines += [f"## mis/{fam}", "",
                  f"**Ours:** quality **{o['quality']:.4f}** @ **{o['runtime_ms']:.1f} ms**.", "",
                  "| method | time_limit (s) | mean quality | mean full wall-clock (ms) |",
                  "| --- | --- | --- | --- |"]
        for method in BINARIES:
            for b in BUDGETS:
                r = next((x for x in per_target_rows if x["target"] == fam and x["method"] == method and x["budget_s"] == b), None)
                if r:
                    lines.append(f"| {method} | {b} | {r['mean_quality']:.4f} | {r['mean_wall_ms']:.1f} |")
        lines.append("")
    (outdir / "anytime_curves.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote Experiment 8 results to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
