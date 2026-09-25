# Reproducibility

This repository contains the source code needed to regenerate the paper experiments. Generated
datasets, candidate solvers, reports, and large result bundles are excluded from git and should be
regenerated locally or supplied through an anonymous artifact archive for review.

## Setup

```bash
uv sync --group dev
uv run python -m pytest -q
```

The benchmark runs on CPU. No local GPU path is used for generated solver or baseline evaluation.
LLM synthesis uses the OpenAI API configured by the environment variables documented in `README.md`.

Optional solver backends:

- Gurobi is enabled by default for benchmark runs. Use `--no-gurobi-baseline` to disable it.
- External exact baselines run in `auto` mode by default. Configure binaries with the
  `DASBENCH_*_BIN` environment variables listed in `README.md`, or rely on native Python backends
  when available.

## Main Paper Benchmark

Run the benchmark that produced the headline results:

```bash
python -m benchmarks.main_paper_benchmark --max-workers 21
```

The legacy equivalent is:

```bash
python -m benchmarks.second_scale_benchmark_v2 --max-workers 21
```

The public alias and the legacy module call the same implementation. The internal condition id is
`seconds_scale_v2` for compatibility with existing artifacts, so target outputs appear under:

```text
artifacts/second_scale_benchmark_v2/<sweep_id>/targets/seconds_scale_v2/<problem>/<family>/
```

Each target contains:

```text
dataset/
agent_run/
report/
```

Sweep-level outputs include:

```text
aggregate_results.json
aggregate_results.csv
benchmark_sweep_summary.json
```

## Ablations

From scratch:

```bash
python -m benchmarks.sample_size_sweep --validation-size 32 --max-workers 4
python -m benchmarks.problem_size_sweep --max-workers 4
python -m benchmarks.candidate_count_sweep --max-workers 4
python -m benchmarks.iteration_count_sweep --max-workers 4
```

Artifact-dependent ablations:

```bash
export MAIN_SWEEP_ROOT=artifacts/second_scale_benchmark_v2/<sweep_id>

python -m benchmarks.no_hint_recovery_benchmark \
  --source-run-root "$MAIN_SWEEP_ROOT" \
  --max-workers 4

python -m benchmarks.graph_relabel_invariance_benchmark \
  --source-run-root "$MAIN_SWEEP_ROOT" \
  --max-workers 4
```

The no-hint and graph-relabel ablations reuse datasets or selected solvers from a completed main
benchmark run. They therefore require `--source-run-root`.

## PACE 2025 Diagnostic

Run one DASBench synthesis pass on PACE 2025 Dominating Set instances:

```bash
python -m benchmarks.pace2025_dominating_set \
  --track heuristic \
  --test-source private
```

Run selected external PACE heuristic baselines and collect a local comparison report:

```bash
python -m scripts.pace2025_run_heuristic_baselines --count 5
python -m scripts.pace2025_collect_heuristic_report
```

PACE private heuristic instances are released by the competition repository, but private best-known
or optimal labels are not included. The report compares feasibility, solution sizes, proxy fields, and
available baseline runtimes; it is not an official PACE score.

## Additional PACE Competition Imports

Build the pragmatic first-pass PACE imports without running synthesis:

```bash
python -m benchmarks.pace_competitions --competition pace2025_hs --build-only
python -m benchmarks.pace_competitions --competition pace2024_ocm_exact --build-only
python -m benchmarks.pace_competitions --competition pace2024_ocm_cutwidth --build-only
python -m benchmarks.pace_competitions --competition pace2022_dfvs_heuristic --build-only
```

The OCM exact and cutwidth imports use released solution archives when a `.sol` member is present.
The HS and DFVS heuristic imports use lower-bound proxy objectives plus local reference-solver
metadata, so their normalized quality is not an official PACE score.

External competition solvers are installed from pinned Git commits into ignored artifact paths:

```bash
python -m benchmarks.pace_competitions \
  --competition pace2025_hs \
  --install-solvers \
  --solvers root,greeduce,shadoks,fontanf

python -m benchmarks.pace_competitions \
  --competition pace2025_hs \
  --run-baselines \
  --solvers root,greeduce \
  --build-only \
  --test-count 5
```

## Result Export

Collect completed target reports and selected candidate code into a compact export folder:

```bash
python -m scripts.collect_experiment_results \
  artifacts/second_scale_benchmark_v2/<sweep_id> \
  exports/main_paper
```

Plot/export helpers:

```bash
python -m scripts.export_baseline_catalog --output exports/baseline_catalog.json
python -m scripts.export_problem_size_runtimes artifacts/problem_size_sweep/<sweep_id>
python -m scripts.plot_iteration_best_runtime_so_far artifacts/iteration_count_sweep/<sweep_id>
python -m scripts.plot_iteration_runtime_ratio_vs_zero_shot artifacts/iteration_count_sweep/<sweep_id>
python -m scripts.plot_problem_size_solver_transfer artifacts/problem_size_sweep/<sweep_id>
```

## Compute Notes

The paper's historical main run was `artifacts/second_scale_benchmark_v2/20260427_230552`. The run
was resumed and patched multiple times, so the most defensible compute accounting is the sum of
recorded per-target stage timings rather than raw calendar elapsed time.

- Main benchmark synthesis stage total: 264,252,366 ms, approximately 73.40 hours.
- Baseline pre-synthesis stage total: 1,502,342,612 ms, approximately 417.32 hours.
- Report stage total: 900,769 ms, approximately 15.01 minutes.
- Additional train/validation evaluation inside synthesis: approximately 2,144,095 ms, or 35.73 minutes.
- Generated solvers and benchmark baselines were evaluated locally on CPU.
- The historical run used parallel execution across targets with `--max-workers 21`; later tail
  resumes used smaller worker counts.
- The sample-size sweep used up to 120 workers; the problem-size sweep used up to 70 workers.
- LLM calls used fixed reasoning effort and structured response schemas; temperature, top-p, and
  max-token settings were left at service defaults.

Recorded historical API usage for the main benchmark:

- successful calls: 463
- prompt tokens: 5,949,071
- completion tokens: 10,158,307
- reasoning tokens: 8,050,900
- total tokens: 16,107,378
