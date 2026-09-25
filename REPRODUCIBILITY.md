# Reproducibility

This repository contains the source code needed to regenerate the paper experiments. Generated
datasets, candidate solvers, reports, and large result bundles are excluded from git and should be
regenerated locally or supplied through an anonymous artifact archive for review.

## Setup

```bash
uv sync --group dev
uv run python -m pytest -q -m "not slow and not gurobi"
```

That is the install check: 191 tests in about 40 seconds, entirely offline.

Two groups are deselected by default and are opt-in:

- `slow` is one end-to-end test that synthesizes and reports on all seven problem classes with the
  Gurobi and exact baselines at their real time limits. It takes about 75 minutes on its own, which
  is essentially the whole suite's runtime.
- `gurobi` marks the tests that build a real Gurobi solver. A WLS license checks out over the
  network on every environment creation, so these are the only tests that contact an external
  service. They skip themselves when no license is usable, rather than failing.

```bash
uv run python -m pytest -q                       # everything
uv run python -m pytest -q -m "not slow"         # adds the Gurobi tests back
```

The benchmark runs on CPU. No local GPU path is used for generated solver or baseline evaluation.

LLM synthesis works through any of three providers, selected with `LLM_PROVIDER` and configured by
the environment variables documented in `README.md` and `.env.example`:

- `openai` (the default) for OpenAI or any OpenAI-compatible endpoint. The paper's runs used this
  with `gpt-5.2` at `xhigh` reasoning effort.
- `anthropic` for the Claude Messages API.
- `custom_chat` for a local OpenAI-compatible server such as vLLM.

All three drive the same pipeline and the same response schemas, so a reproduction may use whichever
is available; the model and provider are recorded in each run's manifest.

Optional solver backends:

- Gurobi is enabled by default for benchmark runs. Use `--no-gurobi-baseline` to disable it.
- External exact baselines run in `auto` mode by default. Configure binaries with the
  `DASBENCH_*_BIN` environment variables listed in `README.md`, or rely on native Python backends
  when available.

### Pointing the analyses at your own run

Every analysis and collection script reads one main-benchmark sweep. By default that is the sweep
the paper reports, so the published numbers reproduce unchanged from an artifact archive. After
running your own main benchmark, export its root once and every script follows:

```bash
export MAIN_SWEEP_ROOT=artifacts/second_scale_benchmark_v2/<your_sweep_id>
```

Scripts that take an explicit `--main-run` or `--source-run-root` still win over the variable.

## What produces which result

| Paper result | Command |
| :-- | :-- |
| Main table, 21 targets (quality, speedups) | `benchmarks.main_paper_benchmark`, then `scripts.collect_all_experiment_results` |
| Fast-heuristic speedup headline | `scripts.recompute_honest_speedups` |
| Learned (ML) baseline columns | `baselines.src.ml_baselines.run_ml_baselines`, then `scripts.collect_ml_baseline_results` |
| One-shot LLM baselines (API models) | `benchmarks.llm_pv_benchmark` |
| One-shot coding-agent baselines (Claude Code, Codex) | `scripts.build_pace_one_shot_capsules`, then `scripts.eval_pace_one_shot_solutions` |
| Open/local model runs | `scripts.run_provider_model_sweep` |
| Seed variance | `benchmarks.seed_variance_benchmark`, then `scripts.collect_seed_variance_results` |
| Portfolio comparison | `scripts.build_portfolio_baseline` |
| Sample/problem/candidate/iteration ablations | the four sweep modules under **Ablations** |
| PACE 2025 Dominating Set | `benchmarks.pace --competition pace2025_ds_heuristic`, then `scripts.pace2025_collect_heuristic_report` |
| Other PACE competitions | `benchmarks.pace --competition <name>`, then `scripts.run_pace_catalog_baselines` |
| PACE quality-time frontier | `scripts.run_exp3_pace_frontier` |
| Anytime MIS curves | `scripts.run_exp8_anytime_mis` |
| Baseline catalog appendix | `scripts.export_baseline_catalog` |

Run everything from the repository root as `python -m <module>`.

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

## PACE Competitions

Every competition runs through one entrypoint. List what is available:

```bash
python -m benchmarks.pace --list
```

`--competition` selects the competition and track; all other flags belong to that competition and
are shown by `python -m benchmarks.pace --competition <name> --help`.

Run one synthesis pass on the Dominating Set heuristic track, the track the paper reports:

```bash
python -m benchmarks.pace \
  --competition pace2025_ds_heuristic \
  --test-source private
```

Import the other competitions without running synthesis:

```bash
python -m benchmarks.pace --competition pace2025_hs --build-only
python -m benchmarks.pace --competition pace2024_ocm_exact --build-only
python -m benchmarks.pace --competition pace2024_ocm_cutwidth --build-only
python -m benchmarks.pace --competition pace2022_dfvs_heuristic --build-only
```

Dominating Set writes under `artifacts/pace2025_dominating_set/` and the rest under
`artifacts/pace_competitions/`. The two report different columns -- Dominating Set reports solution
sizes against a domination lower bound, the others report normalized quality -- and the paper's
tables read each as it is, so the entrypoint is shared while the artifact layouts are not.

The OCM exact and cutwidth imports use released solution archives when a `.sol` member is present.
The HS and DFVS heuristic imports use lower-bound proxy objectives plus local reference-solver
metadata, so their normalized quality is not an official PACE score. PACE private instances are
released by the competition repositories, but private best-known or optimal labels are not, so none
of these reports is an official PACE score.

### Released competition solvers

Install from pinned Git commits into ignored artifact paths, then run them as baselines:

```bash
python -m benchmarks.pace \
  --competition pace2025_hs \
  --install-solvers \
  --solvers root,greeduce,shadoks,fontanf

python -m benchmarks.pace \
  --competition pace2025_hs \
  --run-baselines \
  --solvers root,greeduce \
  --build-only \
  --test-count 5
```

Dominating Set keeps its own baseline runner and report collector, which read its artifact layout:

```bash
python -m scripts.pace2025_run_heuristic_baselines --count 5
python -m scripts.pace2025_collect_heuristic_report
```

Registry catalog baselines on any built PACE dataset:

```bash
python -m scripts.run_pace_catalog_baselines \
  --dataset-dir artifacts/pace_competitions/<run>/dataset \
  --run-output-dir artifacts/pace_competitions/<run>/agent_run
```

## Baselines in the comparison tables

Classical heuristics, exact backends and Gurobi run inside the main benchmark; the sections below
cover the baselines that are separate runs.

### Learned (ML) baselines

One architecture-matched model per problem class, trained and evaluated per target. These are the
only part of the pipeline that uses a GPU, and they are trained on reduced-size per-target datasets
rather than the 500-instance test splits used for the classical comparisons:

```bash
python -m baselines.src.ml_baselines.run_ml_baselines \
  --problem mis --select-on-validation \
  --output-dir artifacts/ml_baseline_runs/seconds_scale_v2_<sweep_id>

python -m scripts.collect_ml_baseline_results
```

### One-shot LLM baselines

Propose-and-verify solver generation from the same specification, without the refinement loop:

```bash
python -m benchmarks.llm_pv_benchmark --all-families --attempts 5 --max-workers 4
python -m benchmarks.pace_llm_pv_baselines --competition pace2025_hs --attempts 5
```

### One-shot coding-agent baselines

Claude Code and Codex are driven interactively, so their runs are staged as self-contained
workspaces and scored afterwards. Build the capsules, run each agent once in its own workspace
following the generated `RUNBOOK.md`, copy each `solution.py` into `solutions/<agent>/<track>/`,
then score:

```bash
python -m scripts.build_pace_one_shot_capsules --sweep-id <sweep_id>
python -m scripts.eval_pace_one_shot_solutions \
  --sweep-root artifacts/pace_one_shot_capsules/<sweep_id>
```

Each capsule carries the public training split only. Agent releases change over time, so record the
CLI and model version in `solutions/<agent>/run_metadata.json`; the scorer copies it into every
result record.

### Open and local models

```bash
bash scripts/start_local_vllm.sh
python -m scripts.run_provider_model_sweep --provider glm-52
```

The sweep is resumable and runs every stage by default; pass `--stage main21_agent` (repeatable) to
run one. `--provider` selects the env-file template, so a new model needs only a new template.

## Analyses

Re-analysis of a completed sweep. These make no API calls and re-run no synthesis:

```bash
python -m scripts.recompute_honest_speedups
python -m scripts.build_portfolio_baseline
python -m scripts.collect_all_experiment_results
```

`recompute_honest_speedups` forms the per-target runtime ratios behind the headline speedup. The
heuristic pool mixes fixed-cost constructors with anytime local-search methods that consume their
whole budget, so it reports the ratio with those members excluded as well as pooled.

Seed variance over independent synthesis runs of the same targets:

```bash
python -m benchmarks.seed_variance_benchmark --representative-only --num-runs 3 --generator llm
python -m scripts.collect_seed_variance_results \
  --sweep-root artifacts/seed_variance_benchmark/<sweep_id> \
  --run-note "gpt-5.2, llm generator, beam/iterations/candidate-width 3/3/3"
```

PACE quality-time frontier and anytime MIS curves:

```bash
python -m scripts.run_exp3_pace_frontier \
  --competition pace2025_hs \
  --dataset-dir artifacts/pace_competitions/<run>/dataset \
  --output-base results/pace_frontier

python -m scripts.run_exp8_anytime_mis --max-workers 8
```

`run_exp8_anytime_mis` shells out to KaMIS; set `DASBENCH_KAMIS_BUILD_DIR` to its build directory.

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
