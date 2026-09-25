# Benchmark Experiments

The paper-facing experiments are runnable from this package. The main paper benchmark is
`second_scale_benchmark_v2`; `benchmarks.main_paper_benchmark` is the public alias used in
submission instructions. Existing artifacts still use the historical condition id
`seconds_scale_v2` under `targets/seconds_scale_v2/...`.

Generated datasets, solver candidates, reports, and aggregate outputs are not committed. Regenerate
them locally, or provide them through an anonymous external artifact archive for review.

| Experiment | Supports | Command | Artifact dependency |
| --- | --- | --- | --- |
| Main Paper Benchmark | Headline quality/runtime results and per-target table | `python -m benchmarks.main_paper_benchmark --max-workers 21` | Starts from scratch |
| Sample Size | Sample-size ablation | `python -m benchmarks.sample_size_sweep --validation-size 32 --max-workers 4` | Starts from scratch |
| Problem Size | Problem-size and solver-transfer curves | `python -m benchmarks.problem_size_sweep --max-workers 4` | Starts from scratch |
| Candidate Count | Candidate-width ablation | `python -m benchmarks.candidate_count_sweep --max-workers 4` | Starts from scratch |
| Iteration Count | Iterative synthesis runtime figure | `python -m benchmarks.iteration_count_sweep --max-workers 4` | Starts from scratch |
| No-Hint Recovery | Hidden-rule framing ablation | `python -m benchmarks.no_hint_recovery_benchmark --source-run-root "$MAIN_SWEEP_ROOT" --max-workers 4` | Needs a completed main benchmark run |
| LLM-PV Baseline | Propose-and-verify baseline: sample solver programs, select by validation, evaluate on test | `python -m benchmarks.llm_pv_benchmark --source-run-root "$MAIN_SWEEP_ROOT" --attempts 5 --model gpt-5 --reasoning-effort high --max-workers 1` | Reuses datasets from a completed main benchmark run |
| Graph Relabel Invariance | Graph presentation perturbation ablation | `python -m benchmarks.graph_relabel_invariance_benchmark --source-run-root "$MAIN_SWEEP_ROOT" --max-workers 4` | Needs a completed main benchmark run |
| PACE competitions (all) | External PACE experiments, one entrypoint | `python -m benchmarks.pace --list`, then `--competition <name>` | Downloads or reads PACE instances; needs LLM API for synthesis; optional pinned solver builds |
| Provider Model Sweep | Run main 21-target, LLM-PV, and PACE experiments across custom-chat providers | `python -m scripts.run_provider_model_sweep --dry-run` | Reuses provider env files such as `.env.kimi-k26` |

`$MAIN_SWEEP_ROOT` should point to a completed main benchmark sweep root, for example
`artifacts/second_scale_benchmark_v2/<sweep_id>`.

## LLM-PV Baseline

`benchmarks.llm_pv_benchmark` adapts LLM-PV to DasBench as a solver-only propose-and-verify
baseline. For each target it stages `train.jsonl`, `validation.jsonl`, and `test.jsonl` from
the source sweep into `artifacts/llm_pv_benchmark/<sweep_id>/targets/llm_pv/...`, prompts the
LLM for `solution.py` candidate programs, evaluates every attempt on train and validation,
selects the best attempt by validation quality/optimality/runtime, and evaluates only that
selected solver on test.

Defaults match the reference LLM-PV search shape closely: `--attempts 5` and early stop at
validation quality `1.0`. `--model` defaults to the active provider's model environment
variable when set and otherwise `gpt-5`; `--reasoning-effort` defaults to the active provider's
reasoning-effort environment variable when set and otherwise `high` for OpenAI. Output is
uncapped by default; pass `--max-output-tokens` only when you want to cap the combined reasoning
and visible output budget. LLM API requests use a large per-attempt timeout by default
(`--api-timeout-seconds 14400`) so high-reasoning calls can finish. Code
interpreter is off by default and can be enabled with `--enable-code-interpreter`. Prompts include
problem-specific return-shape contracts, but evaluation still uses the normal DasBench solver
interfaces. The benchmark runs one representative family per problem unless `--all-families` is passed.
Calls through the alternate chat provider use `CUSTOM_CHAT_TIMEOUT_SECONDS=14400` by default when
the caller does not pass a timeout, and retry HTTP 524 responses up to three times.

Example:

```bash
python -m benchmarks.llm_pv_benchmark \
  --source-run-root artifacts/second_scale_benchmark_v2/20260427_230552 \
  --attempts 5 \
  --model gpt-5 \
  --reasoning-effort high \
  --max-workers 1
```

Useful lower-cost checks:

```bash
python -m benchmarks.llm_pv_benchmark --dry-run --problem tsp
python -m benchmarks.llm_pv_benchmark --problem maxsat --family latent_backdoor_mixture_v1 --attempts 1 --max-workers 1
```

## Defaults

- `--generator llm`
- Gurobi enabled with one thread
- external exact baselines in `auto`
- main paper benchmark defaults to all 21 target distributions
- ablation sweeps default to representative families unless their module documents otherwise
- each sweep writes to `artifacts/<sweep_kind>/<sweep_id>/`
- completed targets are resumable unless `--force` is passed

## Reusable Helpers

Reusable report/export helpers live in `scripts/`:

```bash
python -m scripts.collect_experiment_results artifacts/second_scale_benchmark_v2/<sweep_id> exports/main_paper
python -m scripts.export_baseline_catalog --output exports/baseline_catalog.json
python -m scripts.export_problem_size_runtimes artifacts/problem_size_sweep/<sweep_id>
python -m scripts.plot_iteration_best_runtime_so_far artifacts/iteration_count_sweep/<sweep_id>
python -m scripts.plot_iteration_runtime_ratio_vs_zero_shot artifacts/iteration_count_sweep/<sweep_id>
python -m scripts.plot_problem_size_solver_transfer artifacts/problem_size_sweep/<sweep_id>
```

PACE helper scripts are kept because they support the external Dominating Set diagnostic:

```bash
python -m scripts.pace2025_run_heuristic_baselines --count 5
python -m scripts.pace2025_run_llm_pv_baseline --attempts 5
python -m scripts.pace2025_collect_heuristic_report
```

To compare PACE 2025 Dominating Set instances against the in-repo MDS heuristics:

```bash
python -m scripts.pace2025_run_heuristic_baselines \
  --dasbench-mds-baselines \
  --source private \
  --track heuristic \
  --count 100 \
  --dasbench-timeout-seconds 300 \
  --max-workers 5 \
  --output-dir artifacts/pace2025_dominating_set/baseline_comparisons/dasbench_mds_heuristics
```

To compare against locally installed PACE heuristic solvers using the wrappers in `baselines/bin/`:

```bash
python -m scripts.pace2025_run_heuristic_baselines \
  --solvers fontanf,root,swats,shadoks,aeg,greeduce \
  --source private \
  --track heuristic \
  --count 100 \
  --timeout-seconds 360 \
  --grace-seconds 30 \
  --max-workers 6 \
  --output-dir artifacts/pace2025_dominating_set/baseline_comparisons/pace_top_heuristics
```

The solver checkouts live under `baselines/src/pace2025_*`. The longer external timeout gives
solvers with hardcoded near-competition runtimes enough room to exit normally and write their final
PACE-format solution.

Additional PACE imports share one CLI:

```bash
python -m benchmarks.pace --competition pace2025_hs --build-only
python -m benchmarks.pace --competition pace2024_ocm_exact --build-only
python -m benchmarks.pace --competition pace2024_ocm_cutwidth --build-only
python -m benchmarks.pace --competition pace2022_dfvs_heuristic --build-only
```

## Provider Model Sweep

`scripts.run_provider_model_sweep` coordinates the long model comparison runs across
provider-specific custom-chat env files. It runs sequentially by default, writes per-stage logs
and status JSON under `artifacts/model_sweeps/<provider>/runner_*`, and skips completed stages
unless `--force` is passed.

Create env-file templates:

```bash
python -m scripts.run_provider_model_sweep --init-env-templates
```

Dry-run a single provider:

```bash
python -m scripts.run_provider_model_sweep --dry-run --provider kimi-k26
```

Launch the full sweep detached:

```bash
python -m scripts.run_provider_model_sweep --detach
```

The default providers are `kimi-k26`, `deepseek-v4-pro`, and `glm-52`, backed by
`.env.kimi-k26`, `.env.deepseek-v4-pro`, and `.env.glm-52`.

To install and run pinned external solvers without committing third-party source trees:

```bash
python -m benchmarks.pace \
  --competition pace2024_ocm_heuristic \
  --install-solvers \
  --solvers cimat

python -m benchmarks.pace \
  --competition pace2024_ocm_heuristic \
  --run-baselines \
  --solvers cimat \
  --build-only \
  --test-count 5
```

Solver clones, shims, raw outputs, stderr logs, and baseline reports live under ignored
`artifacts/external/pace_solvers/` and per-run `artifacts/pace_competitions/` directories.

Local run-management utilities used during development, such as failed-run cleanup, candidate
removal, tail finishers, diagnostic patchers, and missing-runtime rerunners, are intentionally omitted
from the submission tree. Their provenance remains in git history.

## Smoke Checks

```bash
python -m benchmarks.main_paper_benchmark --dry-run --problem tsp
python -m benchmarks.second_scale_benchmark_v2 --dry-run --problem tsp
python -m benchmarks.sample_size_sweep --validation-size 32 --dry-run --problem maxsat --family last_clause_signal_v1
python -m benchmarks.problem_size_sweep --dry-run --problem tsp
python -m benchmarks.candidate_count_sweep --dry-run --problem tsp
python -m benchmarks.iteration_count_sweep --dry-run --problem tsp
python -m benchmarks.llm_pv_benchmark --dry-run --problem tsp
python -m benchmarks.pace --competition pace2025_ds_heuristic --help
python -m benchmarks.pace --competition pace2024_ocm_exact --build-only --train-count 1 --validation-count 1 --test-count 1
```
