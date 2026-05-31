from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_INPUT_ROOT = Path("artifacts/ml_baseline_runs/second_scale_v2_20260427_230552")
DEFAULT_OUTPUT_DIR = Path("results/ml_baseline_second_scale_v2_20260427_230552")

AGGREGATE_FIELDS = [
    "source_run_id",
    "source_aggregate_csv",
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

BASELINE_DESCRIPTIONS = {
    "ml_gnn_mis_score_repair": {
        "problem": "mis",
        "source": "baselines/src/ml_baselines/graph_score_repair.py",
        "learns": "Node inclusion priorities from degree, normalized degree, clustering proxy, density, and constant features.",
        "training": "Unsupervised relaxation maximizes selected-node probabilities while penalizing selected edges.",
        "feasibility": "Scores are decoded by score-sorted greedy independent-set construction plus bounded fill repair.",
        "defaults": "epochs=50, hidden_dim=64, layers=3, lr=1e-3, repair_budget=64, seed=0",
        "papers": [
            "[Dai et al., 2017, Learning Combinatorial Optimization Algorithms over Graphs](https://papers.neurips.cc/paper/7214-learning-combinatorial-optimization-algorithms-over-graphs)",
            "[Li, Chen, and Koltun, 2018, Combinatorial Optimization with Graph Convolutional Networks and Guided Tree Search](https://papers.neurips.cc/paper/7335-combinatorial-optimization-with-graph-convolutional-networks-and-guided-tree-search)",
            "[Schuetz, Brubaker, and Katzgraber, 2022, Combinatorial optimization with physics-inspired graph neural networks](https://www.nature.com/articles/s42256-022-00468-6)",
        ],
    },
    "ml_pignn_mis": {
        "problem": "mis",
        "source": "baselines/src/ml_baselines/pignn_mis.py",
        "learns": "Relaxed binary node variables for maximum independent set using graph message passing.",
        "training": "Unsupervised PI-GNN/QUBO Hamiltonian maximizes node mass while penalizing selected edges with annealed sigmoid temperature.",
        "feasibility": "Probabilities are projected by score-ordered greedy independent-set construction, fill repair, and bounded 1-for-2 local improvement.",
        "defaults": "epochs=200, hidden_dim=64, layers=3, lr=1e-3, qubo_penalty=2.0, temperature_start=1.0, temperature_end=0.1, inference_restarts=8, repair_budget=128, seed=0",
        "papers": [
            "[Schuetz, Brubaker, and Katzgraber, 2022, Combinatorial optimization with physics-inspired graph neural networks](https://www.nature.com/articles/s42256-022-00468-6)",
            "[Schuetz, Brubaker, and Katzgraber, 2021, arXiv preprint](https://arxiv.org/abs/2107.01188)",
        ],
    },
    "ml_gnn_mds_score_repair": {
        "problem": "mds",
        "source": "baselines/src/ml_baselines/graph_score_repair.py",
        "learns": "Node domination priorities from the same public graph features as the MIS graph scorer.",
        "training": "Unsupervised relaxation minimizes selected-node mass while penalizing softly uncovered closed neighborhoods.",
        "feasibility": "Greedy coverage decoding adds vertices until all nodes are dominated, then prunes redundant choices.",
        "defaults": "epochs=50, hidden_dim=64, layers=3, lr=1e-3, repair_budget=64, seed=0",
        "papers": [
            "[Cappart et al., 2023, Combinatorial optimization and reasoning with graph neural networks](https://www.jmlr.org/papers/v24/21-0449.html)",
            "[Sato, Yamada, and Kashima, 2019, Approximation Ratios of Graph Neural Networks for Combinatorial Problems](https://arxiv.org/abs/1905.10261)",
            "[Learning-Based Heuristic for Combinatorial Optimization of the Minimum Dominating Set Problem using Graph Convolutional Networks, 2023](https://arxiv.org/abs/2306.03434)",
        ],
    },
    "ml_gnn_rl_mds": {
        "problem": "mds",
        "source": "baselines/src/ml_baselines/gnn_rl_mds.py",
        "learns": "Q-values for selecting graph vertices into a dominating set from dynamic selected/dominated/gain state features.",
        "training": "DDQN-style public graph rollouts with epsilon-greedy exploration, replay, and a target GNN; no optimum sets are used.",
        "feasibility": "Greedy Q-policy is bounded by max_steps_factor*n, then deterministic coverage repair and redundancy pruning produce a valid dominating set.",
        "defaults": "episodes=5000, hidden_dim=64, layers=3, lr=1e-3, gamma=0.99, batch_size=64, target_update_interval=200, repair_budget=128, seed=0",
        "papers": [
            "[Chen, Liu, and He, 2024, Learn to solve dominating set problem with GNN and reinforcement learning](https://www.sciencedirect.com/science/article/abs/pii/S0096300324001899)",
            "[Mnih et al., 2015, Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)",
        ],
    },
    "ml_pignn_coloring": {
        "problem": "coloring",
        "source": "baselines/src/ml_baselines/pignn_coloring.py",
        "learns": "Per-node color logits for a fixed color budget using graph message passing.",
        "training": "Unsupervised Potts-model conflict loss over adjacent same-color probabilities with entropy annealing.",
        "feasibility": "Tries decreasing color counts with deterministic fixed-k greedy/repair decoding and falls back to DSATUR for validity.",
        "defaults": "epochs=100, hidden_dim=64, layers=3, lr=1e-3, inference_restarts=8, repair_budget=256, seed=0",
        "papers": [
            "[Schuetz, Brubaker, Zhu, and Katzgraber, 2022, Graph Coloring with Physics-Inspired Graph Neural Networks](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.043131)",
            "[Schuetz et al., 2022, arXiv preprint](https://arxiv.org/abs/2202.01606)",
        ],
    },
    "ml_gnn_maxsat_assignment": {
        "problem": "maxsat",
        "source": "baselines/src/ml_baselines/maxsat_assignment.py",
        "learns": "Variable assignment probabilities using bipartite variable-clause message passing with literal-sign edges.",
        "training": "Unsupervised expected satisfied-clause objective from public clauses, with early entropy regularization.",
        "feasibility": "Threshold, Bernoulli, and polarity-majority seeds are scored exactly and improved with bounded variable flips.",
        "defaults": "epochs=50, hidden_dim=64, layers=3, lr=1e-3, samples=8, walksat_flips=1000, seed=0",
        "papers": [
            "[Selman, Kautz, and Cohen, 1993/1996, Local search strategies for satisfiability testing](https://dblp.org/rec/conf/dimacs/SelmanKC93)",
            "[Khalil et al., 2017, Learning Combinatorial Optimization Algorithms over Graphs](https://papers.neurips.cc/paper/7214-learning-combinatorial-optimization-algorithms-over-graphs)",
        ],
    },
    "ml_runcsp_maxsat": {
        "problem": "maxsat",
        "source": "baselines/src/ml_baselines/runcsp_maxsat.py",
        "learns": "Boolean variable assignment probabilities using shared recurrent variable/factor message passing over signed clause incidences.",
        "training": "Unsupervised RUN-CSP-style expected weighted satisfied-clause objective with entropy annealing and late binarization.",
        "feasibility": "Threshold, polarity, and Bernoulli candidates are scored exactly and improved by bounded stochastic WalkSAT flips.",
        "defaults": "epochs=100, hidden_dim=64, message_passing_steps=16, recurrent_layers=1, lr=1e-3, samples=16, walksat_restarts=4, walksat_flips=1000, noise=0.1, seed=0",
        "papers": [
            "[Toenshoff et al., 2020, Graph Neural Networks for Maximum Constraint Satisfaction](https://www.frontiersin.org/articles/10.3389/frai.2020.580607/full)",
            "[Selman, Kautz, and Cohen, 1993/1996, Local search strategies for satisfiability testing](https://dblp.org/rec/conf/dimacs/SelmanKC93)",
        ],
    },
    "ml_mdkp_item_scorer": {
        "problem": "mdkp",
        "source": "baselines/src/ml_baselines/item_resource_baselines.py",
        "learns": "Item scores and resource prices from public values, resource consumptions, capacities, and density features.",
        "training": "Unsupervised Lagrangian-style binary relaxation maximizes predicted value and penalizes capacity violation.",
        "feasibility": "Greedy feasible add plus bounded add/drop/swap repair returns item indices accepted by the MDKP verifier.",
        "defaults": "epochs=50, hidden_dim=64, lr=1e-3, violation_penalty=10.0, repair_budget=64, seed=0",
        "papers": [
            "[Bello et al., 2016, Neural Combinatorial Optimization with Reinforcement Learning](https://arxiv.org/abs/1611.09940)",
            "[Approximating Solutions to the Knapsack Problem using the Lagrangian Dual Framework, 2023](https://arxiv.org/abs/2312.03413)",
            "[A Deep Reinforcement Learning-Based Scheme for Solving Multiple Knapsack Problems, 2022](https://www.mdpi.com/2076-3417/12/6/3068)",
        ],
    },
    "ml_drl_mdkp": {
        "problem": "mdkp",
        "source": "baselines/src/ml_baselines/drl_mdkp.py",
        "learns": "A sequential feasible-item selection policy with item encodings, residual-capacity context, and an actor-critic value head.",
        "training": "Public-data actor-critic rollouts reward feasible value gains; no exact/solver labels or hidden optima are used.",
        "feasibility": "Invalid item actions are masked, inference starts from empty and public heuristic solutions, and bounded add/drop/swap repair enforces feasibility.",
        "defaults": "episodes=5000, hidden_dim=128, lr=3e-4, gamma=1.0, entropy_coef=0.01, value_coef=0.5, repair_budget=128, seed=0",
        "papers": [
            "[Bushaj and Buyuktahtakin, 2024, A K-means Supported Reinforcement Learning Framework to Multi-dimensional Knapsack](https://link.springer.com/article/10.1007/s10898-024-01364-6)",
            "[Bello et al., 2016, Neural Combinatorial Optimization with Reinforcement Learning](https://arxiv.org/abs/1611.09940)",
        ],
    },
    "ml_packinglp_item_fraction": {
        "problem": "packing_lp",
        "source": "baselines/src/ml_baselines/item_resource_baselines.py",
        "learns": "Fractional item values and resource prices from public LP packing features.",
        "training": "Unsupervised fractional relaxation maximizes value dot predicted fractions and penalizes capacity violation.",
        "feasibility": "Predicted fractions are scaled/projected to capacities and greedily filled with residual capacity.",
        "defaults": "epochs=50, hidden_dim=64, lr=1e-3, violation_penalty=10.0, repair_budget=64, seed=0",
        "papers": [
            "[Self-Supervised Primal-Dual Learning for Constrained Optimization, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25520)",
            "[A deep learning approach for solving linear programming problems, 2022](https://www.sciencedirect.com/science/article/pii/S0925231222014412)",
            "[Buchbinder and Naor, 2009, Online Primal-Dual Algorithms for Covering and Packing](https://pubsonline.informs.org/doi/10.1287/moor.1080.0363)",
        ],
    },
    "ml_pdl_packinglp": {
        "problem": "packing_lp",
        "source": "baselines/src/ml_baselines/pdl_packinglp.py",
        "learns": "Primal item fractions and nonnegative dual resource multipliers using item/resource message passing.",
        "training": "Self-supervised primal-dual loss combines primal objective, dual-weighted violation, augmented penalty, dual feasibility, and gap terms.",
        "feasibility": "Predicted fractions are clamped, capacity-scaled if needed, and greedily filled under a fixed repair budget.",
        "defaults": "epochs=200, hidden_dim=64, layers=3, lr=1e-3, dual_lr=1e-3, rho=10.0, rho_growth=1.05, repair_budget=128, seed=0",
        "papers": [
            "[Park and Van Hentenryck, 2023, Self-Supervised Primal-Dual Learning for Constrained Optimization](https://ojs.aaai.org/index.php/AAAI/article/view/25520)",
            "[Park and Van Hentenryck, 2022, arXiv preprint](https://arxiv.org/abs/2208.09046)",
        ],
    },
    "ml_tsp_neural_constructor": {
        "problem": "tsp",
        "source": "baselines/src/ml_baselines/tsp_neural_constructor.py",
        "learns": "Directed edge desirability heatmaps from public city coordinates, distances, dx/dy, and nearest-neighbor rank features.",
        "training": "Self-training against pseudo-label tours from nearest insertion, farthest insertion, nearest neighbor, and bounded 2-opt.",
        "feasibility": "Learned insertion and learned nearest-neighbor tours are combined with classical candidates and bounded 2-opt.",
        "defaults": "epochs=50, hidden_dim=128, lr=1e-3, candidates=4, two_opt_budget=256, seed=0",
        "papers": [
            "[Vinyals, Fortunato, and Jaitly, 2015, Pointer Networks](https://arxiv.org/abs/1506.03134)",
            "[Bello et al., 2016, Neural Combinatorial Optimization with Reinforcement Learning](https://arxiv.org/abs/1611.09940)",
            "[Kool, van Hoof, and Welling, 2019, Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475)",
            "[Joshi, Laurent, and Bresson, 2019, An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem](https://arxiv.org/abs/1906.01227)",
        ],
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect ML baseline aggregate metrics and write result reports.")
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        help=(
            "ML baseline run root containing aggregate_results.csv files. "
            f"Defaults to {DEFAULT_INPUT_ROOT}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory under results/ for combined metrics and markdown reports. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument("--title", default="Second-Scale v2 ML Baseline Results")
    return parser


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0.0)
    except ValueError:
        return 0.0


def _int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "0") or 0.0))
    except ValueError:
        return 0


def _format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _format_ms(value: float) -> str:
    if value >= 1000.0:
        return f"{value / 1000.0:.2f} s"
    return f"{value:.2f} ms"


def _format_minutes(value_ms: float) -> str:
    return f"{value_ms / 60_000.0:.2f}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_rows(input_roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for root in input_roots:
        if root.is_file() and root.name == "aggregate_results.csv":
            csv_paths = [root]
        else:
            csv_paths = sorted(root.rglob("aggregate_results.csv"))
        for csv_path in csv_paths:
            resolved = csv_path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            source_run_id = csv_path.parent.name
            for row in _read_csv(csv_path):
                enriched: dict[str, Any] = {field: row.get(field, "") for field in AGGREGATE_FIELDS}
                enriched["source_run_id"] = source_run_id
                enriched["source_aggregate_csv"] = str(csv_path)
                rows.append(enriched)
    rows.sort(key=lambda row: (str(row["problem"]), str(row["family"]), str(row["baseline"]), str(row["source_run_id"])))
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_baseline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_problem[str(row["problem"])].append(row)
        by_baseline[str(row["baseline"])].append(row)
    problem_summary = {
        problem: _summary_stats(problem_rows)
        for problem, problem_rows in sorted(by_problem.items())
    }
    baseline_summary = {
        baseline: _summary_stats(baseline_rows)
        for baseline, baseline_rows in sorted(by_baseline.items())
    }
    return {
        "row_count": len(rows),
        "problem_count": len(by_problem),
        "baseline_count": len(by_baseline),
        "overall": _summary_stats(rows),
        "by_problem": problem_summary,
        "by_baseline": baseline_summary,
    }


def _summary_stats(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {
            "target_count": 0,
            "mean_quality": 0.0,
            "mean_feasibility_rate": 0.0,
            "mean_optimality_rate": 0.0,
            "mean_runtime_ms": 0.0,
            "total_training_time_ms": 0.0,
            "total_test_instances": 0,
        }
    return {
        "target_count": len(rows),
        "mean_quality": statistics.mean(_float(row, "average_normalized_quality") for row in rows),
        "mean_feasibility_rate": statistics.mean(_float(row, "feasibility_rate") for row in rows),
        "mean_optimality_rate": statistics.mean(_float(row, "optimality_rate") for row in rows),
        "mean_runtime_ms": statistics.mean(_float(row, "average_runtime_ms") for row in rows),
        "total_training_time_ms": sum(_float(row, "total_training_time_ms") for row in rows),
        "total_test_instances": sum(_int(row, "num_test_instances") for row in rows),
    }


def build_results_markdown(rows: list[dict[str, Any]], summary: dict[str, Any], *, title: str, sources_filename: str) -> str:
    overall = summary["overall"]
    lines = [
        f"# {title}",
        "",
        "This report collects the final held-out test metrics from the trainable ML baseline runs.",
        "",
        f"- Aggregate rows: `{summary['row_count']}`",
        f"- Problems: `{summary['problem_count']}`",
        f"- Baselines: `{summary['baseline_count']}`",
        f"- Total test instances: `{overall['total_test_instances']}`",
        f"- Mean normalized quality: `{_format_float(float(overall['mean_quality']))}`",
        f"- Mean feasibility rate: `{_format_float(float(overall['mean_feasibility_rate']))}`",
        f"- Mean inference runtime: `{_format_ms(float(overall['mean_runtime_ms']))}`",
        f"- Total offline training time: `{_format_minutes(float(overall['total_training_time_ms']))} min`",
        f"- Baseline descriptions and source files: [{sources_filename}]({sources_filename})",
        "",
        "Training time is reported separately from the per-instance test runtime. The test runtime columns below are inference/scoring runtimes from the ML baseline runner.",
        "",
        "## Problem Summary",
        "",
        "| Problem | Targets | Mean Quality | Mean Feasibility | Mean Opt Rate | Mean Runtime | Training Time (min) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for problem, stats in summary["by_problem"].items():
        lines.append(
            "| "
            f"`{problem}` | "
            f"{stats['target_count']} | "
            f"{_format_float(float(stats['mean_quality']))} | "
            f"{_format_float(float(stats['mean_feasibility_rate']))} | "
            f"{_format_float(float(stats['mean_optimality_rate']))} | "
            f"{_format_ms(float(stats['mean_runtime_ms']))} | "
            f"{_format_minutes(float(stats['total_training_time_ms']))} |"
        )
    lines.extend(
        [
            "",
            "## Per-Family Results",
            "",
            "| Problem | Family | Baseline | Quality | Objective | Opt Rate | Feasibility | Runtime | Training Time (min) | Test n |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"`{row['problem']}` | "
            f"`{row['family']}` | "
            f"`{row['baseline']}` | "
            f"{_format_float(_float(row, 'average_normalized_quality'))} | "
            f"{_format_float(_float(row, 'average_objective_value'))} | "
            f"{_format_float(_float(row, 'optimality_rate'))} | "
            f"{_format_float(_float(row, 'feasibility_rate'))} | "
            f"{_format_ms(_float(row, 'average_runtime_ms'))} | "
            f"{_format_minutes(_float(row, 'total_training_time_ms'))} | "
            f"{_int(row, 'num_test_instances')} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `ml_baseline_results.csv`: combined aggregate metrics from all discovered ML baseline runs.",
            "- `ml_baseline_summary.json`: machine-readable summary grouped by problem and baseline.",
            "- `ml_baseline_report.md`: this report.",
            f"- `{sources_filename}`: baseline method descriptions and implementation source paths.",
            "",
        ]
    )
    return "\n".join(lines)


def build_sources_markdown(rows: list[dict[str, Any]], *, title: str) -> str:
    observed = sorted({str(row["baseline"]) for row in rows})
    lines = [
        f"# {title}: Baselines and Sources",
        "",
        "All ML baselines are implemented under `baselines/src/ml_baselines/`. They train on public train/validation data only; evaluator-only fields such as `optimum_*` and private `_...` metadata are stripped before tensorization. Held-out test data is used only for final evaluation with the existing benchmark scorer.",
        "",
        "The implementations here are lightweight rebuttal baselines, not exact reproductions of the cited systems. The paper links below identify the closest method lineage for each baseline: neural/graph embeddings for combinatorial optimization, SAT/MaxSAT message passing, Lagrangian/primal-dual neural relaxations, and neural TSP construction.",
        "",
        "## Baseline Methods",
        "",
        "| Baseline | Problem | Source File | Learns | Training Signal | Feasibility / Decoder | Defaults | Method lineage / paper sources |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for baseline in observed:
        info = BASELINE_DESCRIPTIONS.get(baseline, {})
        papers = "<br>".join(info.get("papers", []))
        lines.append(
            "| "
            f"`{baseline}` | "
            f"`{info.get('problem', '')}` | "
            f"`{info.get('source', '')}` | "
            f"{info.get('learns', '')} | "
            f"{info.get('training', '')} | "
            f"{info.get('feasibility', '')} | "
            f"{info.get('defaults', '')} | "
            f"{papers} |"
        )
    lines.extend(
        [
            "",
            "## Shared Infrastructure",
            "",
            "- `baselines/src/ml_baselines/tensorize.py`: public-instance tensorization for graph, MaxSAT, packing, and TSP instances.",
            "- `baselines/src/ml_baselines/models.py`: manual PyTorch message passing used by graph baselines.",
            "- `baselines/src/ml_baselines/checkpoint.py`: checkpoint save/load helpers.",
            "- `baselines/src/ml_baselines/seeding.py`: deterministic seed setup for Python, NumPy, and PyTorch.",
            "- `baselines/src/ml_baselines/run_ml_baselines.py`: train/validation/test experiment runner.",
            "",
            "## Benchmark Compatibility",
            "",
            "- Training uses public train split instances.",
            "- Validation may be used for hyperparameter selection.",
            "- Test metrics are computed once using the existing DasBench verifier/scorer.",
            "- Offline training time is reported separately from per-instance test runtime.",
            "- Search and repair are bounded by fixed config values such as `repair_budget`, `walksat_flips`, and `two_opt_budget`.",
            "",
        ]
    )
    return "\n".join(lines)


def collect(input_roots: list[Path], output_dir: Path, *, title: str) -> dict[str, Any]:
    rows = load_rows(input_roots)
    if not rows:
        raise FileNotFoundError(f"No aggregate_results.csv rows found under: {', '.join(str(path) for path in input_roots)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = output_dir / "ml_baseline_results.csv"
    summary_json = output_dir / "ml_baseline_summary.json"
    report_md = output_dir / "ml_baseline_report.md"
    sources_md = output_dir / "ml_baseline_sources.md"
    summary = summarize(rows)
    _write_csv(combined_csv, rows, AGGREGATE_FIELDS)
    _write_json(summary_json, summary)
    report_md.write_text(
        build_results_markdown(rows, summary, title=title, sources_filename=sources_md.name) + "\n",
        encoding="utf-8",
    )
    sources_md.write_text(
        build_sources_markdown(rows, title=title) + "\n",
        encoding="utf-8",
    )
    return {
        "input_roots": [str(path) for path in input_roots],
        "output_dir": str(output_dir),
        "row_count": len(rows),
        "combined_csv": str(combined_csv),
        "summary_json": str(summary_json),
        "report_markdown": str(report_md),
        "sources_markdown": str(sources_md),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_roots = args.input_root or [DEFAULT_INPUT_ROOT]
    result = collect(input_roots, args.output_dir, title=args.title)
    print(f"Collected {result['row_count']} rows")
    print(f"CSV: {result['combined_csv']}")
    print(f"Report: {result['report_markdown']}")
    print(f"Sources: {result['sources_markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
