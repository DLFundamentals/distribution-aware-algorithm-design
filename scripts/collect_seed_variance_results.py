#!/usr/bin/env python3
"""Aggregate Experiment 6a seed-variance runs into a per-target mean +/- std table.

Primary input is a `benchmarks.seed_variance_benchmark` sweep root, whose runs live at
  <sweep_root>/targets/seed_NN/<problem>/<family>/agent_run/synthesis_summary.json
Optionally, legacy ad-hoc runs at
  <legacy_root>/<prefix><problem>_<family>/run_NN/synthesis_summary.json
can be folded in as additional seeds (e.g. the preliminary MIS motif_bridge runs).

For each run it pulls the selected (best) candidate's held-out TEST metrics and its
hypothesis, then writes:
  - variance_table.csv / .md  : per target, mean +/- std (and median/IQR) of
                                quality, runtime_ms, optimality across seeds
  - run_index.csv             : raw per-seed records (auditable)
  - hint_stability.md         : per-target tally of hypothesis diversity_key across seeds
  - README.md                 : commit hash, command, timestamp, notes

Aggregation convention: this is a WITHIN-target, ACROSS-seed statistic (arithmetic
mean +/- sample std of each seed's per-instance mean). It is deliberately distinct from
the paper's cross-target geometric-mean runtime RATIO -- do not conflate the two.

Usage:
  python -m scripts.collect_seed_variance_results \
      --sweep-root artifacts/seed_variance_benchmark/<sweep_id> \
      [--legacy-root artifacts/ablations/variance --legacy-prefix gemma4_] \
      [--out-root results/exp6_variance]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from dasbench.artifacts import main_sweep_root

# Canonical 21-target order (problem/family), to mirror the paper table layout.
TARGET_ORDER = [
    "coloring/cluster_ring_mix_v1",
    "coloring/planted_palette_overlap_v1",
    "coloring/separator_palette_trap_v1",
    "maxsat/community_parity_overlay_v1",
    "maxsat/last_clause_signal_v1",
    "maxsat/latent_backdoor_mixture_v1",
    "mdkp/decoy_complement_mixture_v1",
    "mdkp/latent_class_knapsack_v1",
    "mdkp/single_resource_density_v1",
    "mds/gateway_overlap_cover_v1",
    "mds/geometric_cluster_cover_v1",
    "mds/star_cluster_cover_v1",
    "mis/clique_path_mix_v1",
    "mis/core_fringe_trap_v1",
    "mis/motif_bridge_mixture_v1",
    "packing_lp/block_coupled_resource_v1",
    "packing_lp/latent_active_basis_v1",
    "packing_lp/single_bottleneck_fractional_v1",
    "tsp/clustered_euclidean_v1",
    "tsp/latent_metric_mixture_v1",
    "tsp/paired_ribbon_zigzag_v1",
]


def slug_for(problem: str, family: str, prefix: str) -> str:
    return f"{prefix}{problem}_{family}"


def load_run(summ_path: Path, label: str) -> dict | None:
    """Extract the selected candidate's test metrics + hypothesis from one run."""
    if not summ_path.is_file():
        return None
    try:
        data = json.loads(summ_path.read_text())
    except Exception as exc:  # noqa: BLE001
        return {"run": label, "error": f"parse: {exc}"}
    bc = data.get("best_candidate") or {}
    test = bc.get("test") or {}
    hyp = bc.get("hypothesis") or {}
    # First genuine per-instance error (non-optimal cases carry error=None), for the footnote.
    fail_note = None
    for fc in test.get("failure_cases") or []:
        if isinstance(fc, dict) and fc.get("error"):
            fail_note = str(fc["error"]).split("\n")[0][:160]
            break
    return {
        "run": label,
        "quality": test.get("average_normalized_quality"),
        "runtime_ms": test.get("average_runtime_ms"),
        "optimality": test.get("optimality_rate"),
        "feasibility": test.get("feasibility_rate"),
        "objective": test.get("average_objective_value"),
        "num_instances": test.get("num_instances"),
        "diversity_key": hyp.get("diversity_key"),
        "hypothesis": (hyp.get("hypothesis") or "").strip(),
        "slug": bc.get("slug"),
        "failure_note": fail_note,
        "error": None,
    }


def discover_sweep(sweep_root: Path) -> dict[str, list[dict]]:
    """Runs from a seed_variance_benchmark sweep: targets/seed_NN/<problem>/<family>/agent_run."""
    out: dict[str, list[dict]] = {t: [] for t in TARGET_ORDER}
    troot = sweep_root / "targets"
    if not troot.is_dir():
        return out
    for tgt in TARGET_ORDER:
        problem, family = tgt.split("/", 1)
        for seed_dir in sorted(troot.glob("seed_*")):
            summ = seed_dir / problem / family / "agent_run" / "synthesis_summary.json"
            rec = load_run(summ, seed_dir.name)
            if rec is not None:
                rec["target"] = tgt
                rec["source"] = "sweep"
                out[tgt].append(rec)
    return out


def discover_legacy(legacy_root: Path, prefix: str) -> dict[str, list[dict]]:
    """Legacy ad-hoc runs: <prefix><problem>_<family>/run_NN/synthesis_summary.json."""
    out: dict[str, list[dict]] = {t: [] for t in TARGET_ORDER}
    for tgt in TARGET_ORDER:
        problem, family = tgt.split("/", 1)
        tdir = legacy_root / slug_for(problem, family, prefix)
        if not tdir.is_dir():
            continue
        for run_dir in sorted(tdir.glob("run_*")):
            rec = load_run(run_dir / "synthesis_summary.json", f"legacy_{run_dir.name}")
            if rec is not None:
                rec["target"] = tgt
                rec["source"] = "legacy"
                out[tgt].append(rec)
    return out


# Where the cached, single-run baseline metrics live (same frozen main run the sweep reuses).
BASELINE_SOURCE_RUN_ROOT = str(main_sweep_root())
BASELINE_SOURCE_CONDITION_ID = "seconds_scale_v2"


def load_baselines(source_run_root: Path, condition_id: str) -> dict[str, dict[str, dict]]:
    """Per-target aggregate baseline metrics from the main run's baseline_test.json.

    These are single-run, mostly-deterministic values (no across-seed std): classical
    heuristics and exact/certifying solvers are deterministic; only randomized heuristics
    would vary, and they were run once. So we report baseline MEANS as reference points.
    """
    out: dict[str, dict[str, dict]] = {}
    for tgt in TARGET_ORDER:
        problem, family = tgt.split("/", 1)
        path = source_run_root / "targets" / condition_id / problem / family / "agent_run" / "baseline_test.json"
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        entries: dict[str, dict] = {}
        for name, e in (data.items() if isinstance(data, dict) else []):
            if not isinstance(e, dict):
                continue
            entries[name] = {
                "quality": e.get("average_normalized_quality"),
                "runtime_ms": e.get("average_runtime_ms"),
                "optimality": e.get("optimality_rate"),
                "feasibility": e.get("feasibility_rate"),
                "num_instances": e.get("num_instances"),
            }
        out[tgt] = entries
    return out


# Penalty runtime the harness records when a solver fails on every instance (not a measurement).
SENTINEL_RUNTIME_MS = 1_000_000.0


def is_failed_run(rec: dict) -> bool:
    """True if this run's selected solver failed at deployment and must be excluded from
    the metric mean/std (it errored, was infeasible on all test instances, or recorded the
    sentinel penalty runtime). A feasible-but-suboptimal solver (e.g. a DSATUR fallback with
    quality 0.80, feasibility 1.0) is NOT a failure and stays in the stats."""
    if rec.get("error") is not None:
        return True
    feas = rec.get("feasibility")
    if feas is not None and feas <= 0.0:
        return True
    rt = rec.get("runtime_ms")
    if rt is not None and rt >= SENTINEL_RUNTIME_MS:
        return True
    q = rec.get("quality")
    if q is not None and q <= 0.0:
        return True
    return False


def failure_trigger(rec: dict) -> str:
    """Short reason a run was classified as failed, for the footnote."""
    if rec.get("error") is not None:
        return f"load-error: {rec['error']}"
    reasons = []
    feas = rec.get("feasibility")
    if feas is not None and feas <= 0.0:
        reasons.append("feasibility=0")
    rt = rec.get("runtime_ms")
    if rt is not None and rt >= SENTINEL_RUNTIME_MS:
        reasons.append("sentinel-runtime")
    q = rec.get("quality")
    if q is not None and q <= 0.0:
        reasons.append("quality=0")
    note = rec.get("failure_note")
    tag = ", ".join(reasons) or "unknown"
    return f"{tag}{(' — ' + note) if note else ''}"


def stats(values: list[float]) -> dict:
    vals = [v for v in values if isinstance(v, (int, float))]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "q1": None, "q3": None, "min": None, "max": None}
    out = {
        "n": n,
        "mean": statistics.fmean(vals),
        "std": statistics.stdev(vals) if n >= 2 else 0.0,
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }
    if n >= 4:
        q = statistics.quantiles(vals, n=4, method="inclusive")
        out["q1"], out["q3"] = q[0], q[2]
    else:
        out["q1"], out["q3"] = out["min"], out["max"]
    return out


def fmt(x, nd=4):
    return "" if x is None else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sweep-root",
        help="seed_variance_benchmark sweep dir, e.g. artifacts/seed_variance_benchmark/<sweep_id>.",
    )
    ap.add_argument(
        "--legacy-root",
        help="Optional legacy ad-hoc variance dir to fold in (e.g. artifacts/ablations/variance).",
    )
    ap.add_argument("--legacy-prefix", default="gemma4_")
    ap.add_argument(
        "--source-run-root",
        default=BASELINE_SOURCE_RUN_ROOT,
        help="Main run holding cached baseline_test.json for the reference table. "
             "Pass '' / a missing path to skip baseline reference outputs.",
    )
    ap.add_argument("--source-condition-id", default=BASELINE_SOURCE_CONDITION_ID)
    ap.add_argument("--out-root", default="results/exp6_variance")
    ap.add_argument(
        "--run-note",
        default="",
        help="Free-text provenance recorded in the generated table, e.g. the model, generator and widths.",
    )
    args = ap.parse_args()

    if not args.sweep_root and not args.legacy_root:
        ap.error("provide --sweep-root and/or --legacy-root")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.out_root) / ts
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"

    per_target: dict[str, list[dict]] = {t: [] for t in TARGET_ORDER}
    if args.sweep_root:
        sweep = discover_sweep(Path(args.sweep_root))
        for t in TARGET_ORDER:
            per_target[t].extend(sweep[t])
    if args.legacy_root:
        legacy = discover_legacy(Path(args.legacy_root), args.legacy_prefix)
        for t in TARGET_ORDER:
            per_target[t].extend(legacy[t])

    # Split each target's runs into successful (in stats) vs failed (reported separately).
    ok_by_target: dict[str, list[dict]] = {}
    failed_by_target: dict[str, list[dict]] = {}
    for tgt in TARGET_ORDER:
        ok_by_target[tgt] = [r for r in per_target[tgt] if not is_failed_run(r)]
        failed_by_target[tgt] = [r for r in per_target[tgt] if is_failed_run(r)]

    # ---- run_index.csv (raw per-seed) ----
    with (outdir / "run_index.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "run", "failed", "quality", "runtime_ms", "optimality",
                    "feasibility", "objective", "num_instances", "diversity_key", "failure_trigger"])
        for tgt in TARGET_ORDER:
            for r in per_target[tgt]:
                failed = is_failed_run(r)
                w.writerow([tgt, r.get("run"), int(failed), r.get("quality"), r.get("runtime_ms"),
                            r.get("optimality"), r.get("feasibility"), r.get("objective"),
                            r.get("num_instances"), r.get("diversity_key"),
                            failure_trigger(r) if failed else ""])

    # ---- variance_table.csv ---- (stats over successful runs only)
    with (outdir / "variance_table.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "target", "n_seeds", "n_ok", "n_failed",
            "quality_mean", "quality_std", "quality_min", "quality_max",
            "runtime_ms_mean", "runtime_ms_std", "runtime_ms_median", "runtime_ms_q1", "runtime_ms_q3",
            "optimality_mean", "optimality_std",
        ])
        for tgt in TARGET_ORDER:
            runs = ok_by_target[tgt]
            n_total = len(per_target[tgt])
            n_failed = len(failed_by_target[tgt])
            q = stats([r["quality"] for r in runs])
            rt = stats([r["runtime_ms"] for r in runs])
            op = stats([r["optimality"] for r in runs])
            w.writerow([
                tgt, n_total, q["n"], n_failed,
                fmt(q["mean"]), fmt(q["std"]), fmt(q["min"]), fmt(q["max"]),
                fmt(rt["mean"], 1), fmt(rt["std"], 1), fmt(rt["median"], 1), fmt(rt["q1"], 1), fmt(rt["q3"], 1),
                fmt(op["mean"], 3), fmt(op["std"], 3),
            ])

    # ---- variance_table.md ---- (stats over successful runs only)
    lines = [
        "# Experiment 6a - Seed variance",
        "",
        f"Generated {ts} UTC - commit `{commit}`",
        "",
        "Per target: mean +/- sample std across seeds of the selected solver's held-out test",
        "metrics (test=500 instances). Within-target, across-seed statistic - NOT the paper's",
        "cross-target geometric-mean runtime ratio.",
        "",
        "Statistics are over **successful** runs. A run is excluded when its selected solver",
        "failed at deployment (infeasible on all test instances / errored / sentinel penalty",
        "runtime); such failures are counted in the `seeds` column and detailed below, not",
        "folded into mean/std (a sentinel penalty runtime would otherwise dominate the mean).",
        "",
        "| Target | seeds (ok/excl) | Quality (mean +/- std) | Runtime ms (mean +/- std) | Runtime ms (median [Q1,Q3]) | Optimality (mean +/- std) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    excluded: list[tuple[str, dict]] = []
    for tgt in TARGET_ORDER:
        runs = ok_by_target[tgt]
        n_failed = len(failed_by_target[tgt])
        for r in failed_by_target[tgt]:
            excluded.append((tgt, r))
        seeds_cell = f"{len(runs)}/{n_failed}" if n_failed else f"{len(runs)}/0"
        if not runs:
            note = " (all excluded)" if n_failed else " (no runs)"
            lines.append(f"| {tgt} | {seeds_cell}{note} | | | | |")
            continue
        q = stats([r["quality"] for r in runs])
        rt = stats([r["runtime_ms"] for r in runs])
        op = stats([r["optimality"] for r in runs])
        lines.append(
            f"| {tgt} | {seeds_cell} "
            f"| {fmt(q['mean'])} +/- {fmt(q['std'])} "
            f"| {fmt(rt['mean'],1)} +/- {fmt(rt['std'],1)} "
            f"| {fmt(rt['median'],1)} [{fmt(rt['q1'],1)}, {fmt(rt['q3'],1)}] "
            f"| {fmt(op['mean'],3)} +/- {fmt(op['std'],3)} |"
        )
    if excluded:
        lines += [
            "",
            "## Excluded runs (failed-solver seeds, not in the stats above)",
            "",
            "| Target | run | trigger |",
            "| --- | --- | --- |",
        ]
        for tgt, r in excluded:
            lines.append(f"| {tgt} | {r.get('run')} | {failure_trigger(r)} |")
    (outdir / "variance_table.md").write_text("\n".join(lines) + "\n")

    # ---- hint_stability.md ----
    hlines = [
        "# Experiment 6a - Hint stability across seeds",
        "",
        f"Generated {ts} UTC - commit `{commit}`",
        "",
        "For each target, the selected hypothesis `diversity_key` per seed. Recurrence = the",
        "modal key's share of seeds. Divergent targets are the interesting qualitative cases.",
        "",
        "| Target | seeds | distinct keys | modal key (share) | keys |",
        "| --- | --- | --- | --- | --- |",
    ]
    for tgt in TARGET_ORDER:
        runs = [r for r in per_target[tgt] if r.get("error") is None]
        keys = [r.get("diversity_key") or "(none)" for r in runs]
        if not keys:
            hlines.append(f"| {tgt} | 0 | 0 | - | |")
            continue
        counts: dict[str, int] = {}
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
        modal, modal_n = max(counts.items(), key=lambda kv: kv[1])
        keylist = "; ".join(f"{k}x{c}" for k, c in sorted(counts.items(), key=lambda kv: -kv[1]))
        hlines.append(
            f"| {tgt} | {len(keys)} | {len(counts)} | `{modal}` ({modal_n}/{len(keys)}) | {keylist} |"
        )
    hlines += ["", "## Per-seed hypothesis text (for qualitative writeup)", ""]
    for tgt in TARGET_ORDER:
        runs = [r for r in per_target[tgt] if r.get("error") is None]
        if not runs:
            continue
        hlines.append(f"### {tgt}")
        for r in runs:
            txt = r.get("hypothesis") or ""
            if len(txt) > 400:
                txt = txt[:400] + " ..."
            mark = " **[FAILED at deploy]**" if is_failed_run(r) else ""
            hlines.append(f"- **{r.get('run')}**{mark} [`{r.get('diversity_key')}`]: {txt}")
        hlines.append("")
    (outdir / "hint_stability.md").write_text("\n".join(hlines) + "\n")

    # ---- baseline reference (cached single-run baseline means) ----
    src_root = Path(args.source_run_root) if args.source_run_root else None
    baselines = load_baselines(src_root, args.source_condition_id) if src_root else {}
    n_baseline_targets = sum(1 for t in TARGET_ORDER if baselines.get(t))
    if baselines:
        with (outdir / "baseline_reference.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["target", "baseline", "quality", "runtime_ms", "optimality", "feasibility", "num_instances"])
            for tgt in TARGET_ORDER:
                for name, m in sorted(baselines.get(tgt, {}).items(),
                                      key=lambda kv: -(kv[1]["quality"] if kv[1]["quality"] is not None else -1)):
                    w.writerow([tgt, name, fmt(m["quality"]), fmt(m["runtime_ms"], 1),
                                fmt(m["optimality"], 3), fmt(m["feasibility"], 3), m["num_instances"]])

        blines = [
            "# Experiment 6a - Baseline reference (cached main run, single deterministic run)",
            "",
            f"Generated {ts} UTC - commit `{commit}`",
            "",
            f"Baseline metrics from `{args.source_run_root}` (condition `{args.source_condition_id}`),",
            "test split (500 instances). These are single-run values: classical heuristics and exact",
            "solvers are deterministic (no across-seed std); only randomized heuristics would vary and",
            "were run once. Reported as reference MEANS (std n/a). Compare against the synthesized",
            "solver's across-seed mean +/- std from `variance_table.md`.",
            "",
            "## Synthesized solver vs. baseline reference",
            "",
            "| Target | Ours Q (mean +/- std) | Ours RT ms (mean +/- std) | Best baseline Q | best baseline | Best baseline RT ms | Gurobi Q | Gurobi RT ms |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for tgt in TARGET_ORDER:
            runs = ok_by_target[tgt]
            q = stats([r["quality"] for r in runs])
            rt = stats([r["runtime_ms"] for r in runs])
            bl = baselines.get(tgt, {})
            best_name, best_m = None, None
            for name, m in bl.items():
                if m["quality"] is None:
                    continue
                if best_m is None or m["quality"] > best_m["quality"]:
                    best_name, best_m = name, m
            g = bl.get("gurobi_timed")
            ours_q = f"{fmt(q['mean'])} +/- {fmt(q['std'])}" if q["n"] else "(no runs)"
            ours_rt = f"{fmt(rt['mean'], 1)} +/- {fmt(rt['std'], 1)}" if q["n"] else ""
            bits = [
                tgt, ours_q, ours_rt,
                fmt(best_m["quality"]) if best_m else "", best_name or "",
                fmt(best_m["runtime_ms"], 1) if best_m else "",
                fmt(g["quality"]) if g else "", fmt(g["runtime_ms"], 1) if g else "",
            ]
            blines.append("| " + " | ".join(bits) + " |")

        blines += ["", "## Full baseline catalog per target (quality desc)", ""]
        for tgt in TARGET_ORDER:
            bl = baselines.get(tgt, {})
            if not bl:
                continue
            blines.append(f"### {tgt}")
            blines.append("| Baseline | Q | Runtime ms | Optimality | Feasibility |")
            blines.append("| --- | --- | --- | --- | --- |")
            for name, m in sorted(bl.items(), key=lambda kv: -(kv[1]["quality"] if kv[1]["quality"] is not None else -1)):
                blines.append(f"| {name} | {fmt(m['quality'])} | {fmt(m['runtime_ms'], 1)} | {fmt(m['optimality'], 3)} | {fmt(m['feasibility'], 3)} |")
            blines.append("")
        (outdir / "baseline_reference.md").write_text("\n".join(blines) + "\n")

    # ---- README.md ----
    total_runs = sum(len(per_target[t]) for t in TARGET_ORDER)
    n_failed_total = sum(len(failed_by_target[t]) for t in TARGET_ORDER)
    ok_total = total_runs - n_failed_total
    (outdir / "README.md").write_text(
        f"""# Experiment 6a aggregation - {ts} UTC

- commit: `{commit}`
- sweep root: `{args.sweep_root or "(none)"}`
- legacy root: `{args.legacy_root or "(none)"}`
- targets: {len(TARGET_ORDER)}
- runs found: {total_runs} (successful: {ok_total}, excluded-failed: {n_failed_total})

Run settings: {args.run_note or "(pass --run-note to record model, generator and widths)"}
Datasets reused from `{args.source_run_root or "(none)"}` (train=64/val=32/test=500), baselines skipped.

Metric mean/std are over **successful** runs. A run is excluded when its selected solver failed
at deployment (feasibility=0 on test / errored / sentinel penalty runtime {int(SENTINEL_RUNTIME_MS)} ms);
excluded runs are counted in the table's `seeds` column and listed under "Excluded runs". A
feasible-but-suboptimal solver is NOT a failure and stays in the stats.

Baseline reference: {n_baseline_targets}/{len(TARGET_ORDER)} targets loaded from
`{args.source_run_root or "(none)"}` (single deterministic run; std n/a for baselines).

Outputs:
- `variance_table.{{csv,md}}` - per-target mean +/- std of quality/runtime/optimality (successful runs)
- `run_index.csv` - raw per-seed records with a `failed` flag and `failure_trigger`
- `hint_stability.md` - hypothesis diversity_key tally + per-seed hypothesis text
- `baseline_reference.{{csv,md}}` - cached baseline means per target + ours-vs-baselines comparison
""",
    )

    print(f"Wrote aggregation to {outdir}")
    print(f"  targets={len(TARGET_ORDER)} runs_found={total_runs} successful={ok_total} excluded_failed={n_failed_total}")
    print(f"  baseline reference targets: {n_baseline_targets}/{len(TARGET_ORDER)}")


if __name__ == "__main__":
    main()
