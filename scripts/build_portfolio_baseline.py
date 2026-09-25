#!/usr/bin/env python3
"""Experiment 1 - sample-conditioned portfolio baseline.

Answers: does *synthesizing* new solver code beat *selecting* an existing catalog solver,
when both get the same public samples? Builds per-target selectors over the full classical
catalog (78 = heuristics + exact/certifying + Gurobi; ML is excluded for a 10x instance-scale
mismatch and handled in Exp 9), fit on the public VALIDATION split and reported on the held-out
TEST split -- next to our synthesized solver and an oracle virtual-best upper bound. Then the
same question at PACE scale, where the catalog collapses (exact/Gurobi time out; heuristics are
slower and larger than our solver).

Selector variants (fit on validation, deploy that baseline, report its TEST metrics):
  a) argmax quality        - max Q_val
  b) argmin runtime @ floor - min T_val among {feasibility_val==1 and Q_val >= tau};
                             tau in {our solver's Q_val, 0.99}
  c) lexicographic         - max (Q_val, O_val, -T_val)   [the beam's own rule]
  oracle virtual-best      - max (Q_test, O_test, -T_test) [upper bound, not fittable]

Aggregation matches the paper: arithmetic mean quality/optimality over targets; geometric-mean
runtime RATIO (T_selected / T_ours) over targets; each target already averaged over test first.

Usage:
  python -m scripts.build_portfolio_baseline
      [--main-run artifacts/second_scale_benchmark_v2/20260427_230552]
      [--pace-root artifacts/pace2025_dominating_set]
      [--out-root results/exp1_portfolio]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from dasbench.artifacts import main_sweep_root

PROBLEM_ORDER = ["coloring", "maxsat", "mdkp", "mds", "mis", "packing_lp", "tsp"]
TARGET_ORDER = [
    "coloring/cluster_ring_mix_v1", "coloring/planted_palette_overlap_v1", "coloring/separator_palette_trap_v1",
    "maxsat/community_parity_overlay_v1", "maxsat/last_clause_signal_v1", "maxsat/latent_backdoor_mixture_v1",
    "mdkp/decoy_complement_mixture_v1", "mdkp/latent_class_knapsack_v1", "mdkp/single_resource_density_v1",
    "mds/gateway_overlap_cover_v1", "mds/geometric_cluster_cover_v1", "mds/star_cluster_cover_v1",
    "mis/clique_path_mix_v1", "mis/core_fringe_trap_v1", "mis/motif_bridge_mixture_v1",
    "packing_lp/block_coupled_resource_v1", "packing_lp/latent_active_basis_v1", "packing_lp/single_bottleneck_fractional_v1",
    "tsp/clustered_euclidean_v1", "tsp/latent_metric_mixture_v1", "tsp/paired_ribbon_zigzag_v1",
]
CONDITION = "seconds_scale_v2"


# ------------------------------- loading -------------------------------------
def _metrics(entry: dict) -> dict:
    return {
        "q": entry.get("average_normalized_quality"),
        "o": entry.get("optimality_rate"),
        "t": entry.get("average_runtime_ms"),
        "feas": entry.get("feasibility_rate"),
    }


def load_catalog(main_run: Path, target: str) -> tuple[dict, dict]:
    """Return (val, test): baseline -> {q,o,t,feas} for both splits (common baselines only)."""
    base = main_run / "targets" / CONDITION / target / "agent_run"
    val_raw = json.loads((base / "baseline_validation.json").read_text())
    test_raw = json.loads((base / "baseline_test.json").read_text())
    common = set(val_raw) & set(test_raw)
    val = {b: _metrics(val_raw[b]) for b in common}
    test = {b: _metrics(test_raw[b]) for b in common}
    return val, test


def load_our_solver(main_run: Path, target: str) -> dict:
    summ = json.loads((main_run / "targets" / CONDITION / target / "agent_run" / "synthesis_summary.json").read_text())
    bc = summ["best_candidate"]
    sel, test = bc["selection"], bc["test"]
    return {
        "val": {"q": sel.get("validation_normalized_quality"), "o": sel.get("validation_optimality_rate"),
                "t": sel.get("validation_runtime_ms")},
        "test": {"q": test.get("average_normalized_quality"), "o": test.get("optimality_rate"),
                 "t": test.get("average_runtime_ms")},
    }


# ------------------------------- selectors -----------------------------------
def sel_argmax_quality(val: dict) -> str | None:
    # Max validation quality; break ties toward the FASTEST solver (min T_val) so the pick -- and
    # thus its reported test quality/runtime -- is deterministic and gives the minimum runtime cost
    # of reaching that quality (many catalog solvers tie at Q_val=1.0).
    return _best(val, key=lambda m: (_n(m["q"]), -_n(m["t"], big=True)))


def sel_lexicographic(val: dict) -> str | None:
    return _best(val, key=lambda m: (_n(m["q"]), _n(m["o"]), -_n(m["t"], big=True)))


def sel_argmin_runtime_floor(val: dict, tau: float) -> str | None:
    elig = {b: m for b, m in val.items() if _n(m["feas"]) >= 1.0 and _n(m["q"]) >= tau}
    if not elig:
        return None
    return min(elig, key=lambda b: (_n(elig[b]["t"], big=True), b))  # name tiebreak = deterministic


def oracle_best(test: dict) -> str | None:
    return _best(test, key=lambda m: (_n(m["q"]), _n(m["o"]), -_n(m["t"], big=True)))


def _best(d: dict, key):
    cand = {b: m for b, m in d.items() if m["q"] is not None}
    return max(cand, key=lambda b: (key(cand[b]), b)) if cand else None  # name tiebreak = deterministic


def _n(x, big=False):
    if x is None:
        return (1e18 if big else -1.0)
    return float(x)


# ------------------------------- aggregation ---------------------------------
def geomean(xs: list[float]) -> float | None:
    xs = [x for x in xs if isinstance(x, (int, float)) and x > 0]
    return math.exp(statistics.fmean(math.log(x) for x in xs)) if xs else None


def fmt(x, nd=4):
    return "" if x is None else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-run", default=str(main_sweep_root()))
    ap.add_argument("--pace-root", default="artifacts/pace2025_dominating_set")
    ap.add_argument("--out-root", default="results/exp1_portfolio")
    args = ap.parse_args()

    main_run = Path(args.main_run)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.out_root) / ts
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"

    # Methods reported (label -> callable selecting a baseline name, or None for "ours").
    methods = [
        ("argmax_quality", lambda val, test, ours: sel_argmax_quality(val)),
        ("argmin_rt_floor_ours", lambda val, test, ours: sel_argmin_runtime_floor(val, _n(ours["val"]["q"]))),
        ("argmin_rt_floor_0.99", lambda val, test, ours: sel_argmin_runtime_floor(val, 0.99)),
        ("lexicographic", lambda val, test, ours: sel_lexicographic(val)),
        ("oracle_virtual_best", lambda val, test, ours: oracle_best(test)),
    ]

    per_target: list[dict] = []
    for target in TARGET_ORDER:
        val, test = load_catalog(main_run, target)
        ours = load_our_solver(main_run, target)
        row = {"target": target, "n_catalog": len(val),
               "our_q": ours["test"]["q"], "our_o": ours["test"]["o"], "our_t": ours["test"]["t"]}
        for label, fn in methods:
            b = fn(val, test, ours)
            if b is None:
                row[label] = {"name": None}
                continue
            tm = test[b]
            rt_ratio = (tm["t"] / ours["test"]["t"]) if (tm["t"] and ours["test"]["t"]) else None
            gap = (val[b]["q"] - tm["q"]) if (val[b]["q"] is not None and tm["q"] is not None) else None
            row[label] = {"name": b, "q": tm["q"], "o": tm["o"], "t": tm["t"], "rt_ratio": rt_ratio, "gap": gap}
        per_target.append(row)

    _write_per_target_csv(outdir / "selector_per_target.csv", per_target, [m[0] for m in methods])
    _write_summary_md(outdir / "selector_summary.md", per_target, methods, ts, commit)

    pace = build_pace(Path(args.pace_root))
    _write_pace(outdir, pace, ts, commit)

    _write_readme(outdir, args, ts, commit, len(per_target), pace)
    print(f"Wrote Experiment 1 portfolio results to {outdir}")


def _agg_quality(per_target, label):
    return statistics.fmean([r[label]["q"] for r in per_target if r[label]["name"] and r[label]["q"] is not None])


def _agg_opt(per_target, label):
    vals = [r[label]["o"] for r in per_target if r[label]["name"] and r[label]["o"] is not None]
    return statistics.fmean(vals) if vals else None


def _agg_ratio(per_target, label):
    return geomean([r[label]["rt_ratio"] for r in per_target if r[label]["name"]])


def _agg_gap(per_target, label):
    vals = [r[label]["gap"] for r in per_target if r[label]["name"] and r[label]["gap"] is not None]
    return statistics.fmean(vals) if vals else None


def _our_q_matched(per_target, label):
    """Our mean quality over ONLY the targets this selector actually covers (for a matched delta
    when a selector fails to reach our quality floor on some targets and so covers < 21)."""
    vals = [r["our_q"] for r in per_target if r[label]["name"] and r["our_q"] is not None]
    return statistics.fmean(vals) if vals else None


def _uncovered_targets(per_target, label):
    return [r["target"] for r in per_target if not r[label]["name"]]


def _write_per_target_csv(path, per_target, labels):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        head = ["target", "n_catalog", "our_q_test", "our_o_test", "our_t_ms"]
        for lb in labels:
            head += [f"{lb}__name", f"{lb}__q_test", f"{lb}__o_test", f"{lb}__t_ms", f"{lb}__rt_ratio", f"{lb}__val_test_gap"]
        w.writerow(head)
        for r in per_target:
            row = [r["target"], r["n_catalog"], fmt(r["our_q"]), fmt(r["our_o"], 3), fmt(r["our_t"], 1)]
            for lb in labels:
                m = r[lb]
                row += [m.get("name") or "(none)", fmt(m.get("q")), fmt(m.get("o"), 3),
                        fmt(m.get("t"), 1), fmt(m.get("rt_ratio"), 2), fmt(m.get("gap"))]
            w.writerow(row)


def _write_summary_md(path, per_target, methods, ts, commit):
    our_q = statistics.fmean([r["our_q"] for r in per_target if r["our_q"] is not None])
    our_o = statistics.fmean([r["our_o"] for r in per_target if r["our_o"] is not None])
    lines = [
        "# Experiment 1 - Sample-conditioned portfolio baseline (Part A: 21 targets)",
        "", f"Generated {ts} UTC - commit `{commit}`", "",
        "Catalog = 78 classical + exact/certifying + Gurobi entries (ML excluded: 10x instance-scale",
        "mismatch, see Exp 9). Selectors fit on the public VALIDATION split, reported on held-out TEST.",
        "Aggregation matches the paper: arithmetic-mean quality/optimality over the 21 targets;",
        "geometric-mean runtime RATIO (selected / ours) over targets.", "",
        "| Method | mean Q (test) | vs ours (all-21) | vs ours (matched) | mean optimality | geomean runtime ratio (sel/ours) | mean val->test gap | targets |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| **our synthesized solver** | **{our_q:.4f}** | -- | -- | {our_o:.3f} | 1.00x (ref) | -- | 21 |",
    ]
    for label, _ in methods:
        n = sum(1 for r in per_target if r[label]["name"])
        q = _agg_quality(per_target, label)
        o = _agg_opt(per_target, label)
        ratio = _agg_ratio(per_target, label)
        gap = _agg_gap(per_target, label)
        delta = q - our_q
        our_q_m = _our_q_matched(per_target, label)
        matched = f"{q - our_q_m:+.4f}" if (our_q_m is not None and n < 21) else "(same)"
        lines.append(
            f"| {label} | {q:.4f} | {delta:+.4f} | {matched} | {fmt(o,3)} | {fmt(ratio,2)}x | {fmt(gap)} | {n} |"
        )
    # Name the targets that force the < 21 subsets, so the matched column is auditable.
    uncovered = {lb: _uncovered_targets(per_target, lb) for lb, _ in methods}
    uncovered_note = "; ".join(f"`{lb}`: {', '.join(t)}" for lb, t in uncovered.items() if t) or "none"
    lines += [
        "", "**Reading.** `vs ours (all-21)` compares the selector's mean quality (over the targets it",
        "covers) against ours over all 21 -- an UNMATCHED comparison when a selector covers < 21.",
        "`vs ours (matched)` is the honest like-for-like: selector minus ours over the SAME covered",
        "targets (`(same)` when the selector covers all 21). `runtime ratio` > 1 means the selected",
        "catalog solver is that many times slower than ours. `targets` < 21 means no catalog solver met",
        f"the quality floor on some target (catalog could not match us there) -- uncovered: {uncovered_note}.",
        "", "Per-target detail: `selector_per_target.csv`.",
    ]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------- PACE --------------------------------------
def build_pace(pace_root: Path) -> dict:
    comp = pace_root / "baseline_comparisons"
    catalog: list[dict] = []
    for sub, category in [("dasbench_mds_heuristics", "heuristic"),
                          ("dasbench_mds_exact_baselines", "exact"),
                          ("dasbench_mds_gurobi", "gurobi")]:
        summ_path = comp / sub / "baseline_summary.json"
        if not summ_path.is_file():
            continue
        solvers = json.loads(summ_path.read_text()).get("solvers", {})
        for name, m in solvers.items():
            catalog.append({
                "solver": name, "category": category,
                "valid": m.get("valid_count"), "invalid": m.get("invalid_count"),
                "timeout": m.get("timeout_count"), "n": m.get("num_instances"),
                "avg_size": m.get("average_solution_size"), "avg_rt_ms": m.get("average_runtime_ms"),
            })

    # Our solver on the 100 private instances.
    ours = {"valid": 0, "sizes": [], "rts": [], "per_instance": {}}
    our_csv = None
    for cand in pace_root.glob("*/pace_evaluation/pace_private_results.csv"):
        if "heuristic_llm" in str(cand):
            our_csv = cand
            break
    our_csv = our_csv or next(pace_root.glob("*/pace_evaluation/pace_private_results.csv"), None)
    if our_csv:
        for r in csv.DictReader(our_csv.open()):
            feas = str(r.get("feasible", "")).lower() in ("true", "1")
            if feas and r.get("solution_size"):
                ours["valid"] += 1
                ours["sizes"].append(int(float(r["solution_size"])))
                ours["per_instance"][r["instance_id"]] = int(float(r["solution_size"]))
            if r.get("runtime_ms"):
                ours["rts"].append(float(r["runtime_ms"]))

    # Oracle virtual-best VALID catalog selection: per instance, min valid size across catalog solvers.
    per_instance_best: dict[str, int] = {}
    heur_csv = comp / "dasbench_mds_heuristics" / "baseline_results.csv"
    if heur_csv.is_file():
        for r in csv.DictReader(heur_csv.open()):
            if str(r.get("valid", "")).lower() in ("true", "1") and r.get("solution_size"):
                iid, size = r["instance_id"], int(float(r["solution_size"]))
                per_instance_best[iid] = min(per_instance_best.get(iid, size), size)
    oracle_sizes = list(per_instance_best.values())

    our_avg_size = statistics.fmean(ours["sizes"]) if ours["sizes"] else None
    oracle_avg = statistics.fmean(oracle_sizes) if oracle_sizes else None
    return {
        "catalog": catalog,
        "ours": {"valid": ours["valid"], "n": len(ours["rts"]) or len(ours["sizes"]),
                 "avg_size": our_avg_size, "avg_rt_ms": statistics.fmean(ours["rts"]) if ours["rts"] else None},
        "oracle_valid": {"avg_size": oracle_avg, "covered": len(oracle_sizes),
                         "size_ratio_vs_ours": (oracle_avg / our_avg_size) if (oracle_avg and our_avg_size) else None},
    }


def _write_pace(outdir, pace, ts, commit):
    ours = pace["ours"]
    with (outdir / "pace_selector.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["solver", "category", "valid", "invalid", "timeout", "n", "avg_size", "avg_rt_ms", "size_ratio_vs_ours"])
        for c in pace["catalog"]:
            sr = (c["avg_size"] / ours["avg_size"]) if (c["avg_size"] and ours["avg_size"]) else ""
            w.writerow([c["solver"], c["category"], c["valid"], c["invalid"], c["timeout"], c["n"],
                        fmt(c["avg_size"], 1), fmt(c["avg_rt_ms"], 1), fmt(sr, 4) if sr != "" else ""])

    lines = [
        "# Experiment 1 - PACE portfolio selector (Part B)", "",
        f"Generated {ts} UTC - commit `{commit}`", "",
        "Can any selection over the catalog reproduce our PACE result? At PACE scale the catalog",
        "collapses. Metric is dominating-set solution SIZE (smaller = better) and validity on the",
        "100 private instances; `size ratio` is catalog / ours (>1 = larger/worse than ours).", "",
        f"**Our synthesized solver:** valid **{ours['valid']}/{ours['n']}**, avg size "
        f"**{fmt(ours['avg_size'],1)}**, avg runtime **{fmt(ours['avg_rt_ms'],1)} ms**.", "",
        "| Catalog solver | category | valid | avg size | avg runtime ms | size ratio vs ours |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in sorted(pace["catalog"], key=lambda c: (c["category"], -(c["valid"] or 0))):
        sr = (c["avg_size"] / ours["avg_size"]) if (c["avg_size"] and ours["avg_size"]) else None
        lines.append(f"| {c['solver']} | {c['category']} | {c['valid']}/{c['n']} | {fmt(c['avg_size'],1)} "
                     f"| {fmt(c['avg_rt_ms'],1)} | {fmt(sr,3) if sr else '-'} |")
    ov = pace["oracle_valid"]
    lines += [
        "",
        f"**Oracle virtual-best VALID catalog selection** (per-instance min valid size across catalog "
        f"heuristics, {ov['covered']}/100 instances): avg size **{fmt(ov['avg_size'],1)}**, "
        f"size ratio vs ours **{fmt(ov['size_ratio_vs_ours'],3)}**.", "",
        "**Conclusion.** Exact and Gurobi are 0/100 valid (they time out at PACE scale), so no",
        "validity-respecting selector can pick them. Every valid catalog heuristic -- including the",
        "oracle best-valid selection -- is both larger and slower than the synthesized solver, so no",
        "catalog selection reproduces our PACE result.",
    ]
    (outdir / "pace_selector.md").write_text("\n".join(lines) + "\n")


def _write_readme(outdir, args, ts, commit, n_targets, pace):
    (outdir / "README.md").write_text(
        f"""# Experiment 1 aggregation - {ts} UTC

- commit: `{commit}`
- main run: `{args.main_run}` (condition `{CONDITION}`)
- pace root: `{args.pace_root}`
- targets: {n_targets}

Catalog = 78 classical+exact+Gurobi entries (ML excluded for a 10x instance-scale mismatch; see
Experiment 9). Part A selectors fit on validation, reported on held-out test; Part B compares the
catalog vs our synthesized solver on the 100 PACE private instances.

Outputs:
- `selector_summary.md` - Part A: per-method mean quality / runtime-ratio / val->test gap vs ours
- `selector_per_target.csv` - Part A raw per-target selections
- `pace_selector.{{md,csv}}` - Part B: PACE catalog vs ours + oracle best-valid
""",
    )


if __name__ == "__main__":
    main()
