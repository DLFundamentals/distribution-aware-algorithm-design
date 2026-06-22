# dasbench

`dasbench` is a unified benchmark framework for distribution-aware algorithm synthesis on hard combinatorial problems.

The current benchmark suite supports:

- `coloring`
- `maxsat`
- `mdkp`
- `mis`
- `mds`
- `packing_lp`
- `tsp`

The framework includes:

- problem-specific validation, scoring, baselines, and exact solvers
- a default-on timed Gurobi industrial baseline for all supported problems
- distribution family registries grouped by problem
- dataset generation with stored exact optima
- template and LLM synthesis loops using a common candidate interface
- organized artifacts under:
  - `artifacts/datasets/<problem>/<family>/<dataset_id>/`
  - `artifacts/agent_runs/<problem>/<family>/<run_id>/`
  - `artifacts/reports/<problem>/<family>/<run_id>/`

## Candidate Interface

A candidate directory contains:

- `analyze.py` with `analyze(train_instances, manifest=None) -> dict`
- `solution.py` with `solve(instance, analysis=None, manifest=None) -> object`

Candidate-facing instances are sanitized before analysis and inference, so stored optimum metadata is not exposed to synthesized code.

For LLM-generated candidates, the run also stores `hypothesis.json`. This is the agent's explicit guess about the hidden distributional rule, the evidence its analysis should measure, and the solver strategy implied by that rule. Beam-mode LLM runs preserve hypothesis diversity before filling remaining beam slots by performance.

## Install

```bash
uv sync
```

Optional OpenAI API environment variables for the LLM generator:

```dotenv
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
OPENAI_MODEL=gpt-5.2
OPENAI_REASONING_EFFORT=xhigh
# Optional for OpenAI-compatible endpoints:
# OPENAI_BASE_URL=https://api.openai.com/v1
```

### Local vLLM Server

DasBench can use a local vLLM OpenAI-compatible server through the alternate
chat provider path. The server environment is intentionally separate from the
main project environment because vLLM carries a large CUDA/PyTorch dependency
stack.

Start a local server on port `8001`:

```bash
cp .env.local-vllm.example .env.local-vllm
set -a
source .env.local-vllm
set +a
scripts/start_local_vllm.sh
```

In another shell, load the same client variables and verify that strict
JSON-schema output works:

```bash
set -a
source .env.local-vllm
set +a
uv run python scripts/probe_local_vllm.py
```

Then run LLM-backed experiments normally, for example:

```bash
uv run python main.py run-agent \
  --dataset-dir artifacts/datasets/mis/motif_bridge_mixture_v1/smoke_mis \
  --generator llm \
  --mode beam \
  --iterations 1 \
  --beam-width 1
```

Useful server overrides:

```bash
VLLM_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
VLLM_SERVED_MODEL_NAME=local-coder-32b \
CUSTOM_CHAT_MODEL=local-coder-32b \
CUSTOM_CHAT_MAX_TOKENS=8192 \
VLLM_TENSOR_PARALLEL_SIZE=2 \
DASBENCH_VLLM_PORT=8001 \
VLLM_MAX_MODEL_LEN=32768 \
scripts/start_local_vllm.sh
```

For vLLM, leave `CUSTOM_CHAT_REASONING_EFFORT` unset unless the specific server
version and model accept that OpenAI-specific field.
`CUSTOM_CHAT_MAX_TOKENS` caps each local structured response, and
`DASBENCH_LLM_WAIT_FOR_IDLE=1` serializes local vLLM generation requests across
concurrent DasBench runs and waits for `/metrics` to report no running or
queued requests before starting inference.
`DASBENCH_CODE_REPAIR_LIMIT` controls how many times invalid generated
`analyze.py` or `solution.py` code is repaired before the candidate is marked
failed. `DASBENCH_ANALYSIS_RETRY_LIMIT` controls repair attempts for generated
`analyze.py` code that passes validation but fails during execution.
`DASBENCH_SOLUTION_REPAIR_LIMIT` controls repair attempts for generated
`solution.py` code that passes validation but produces infeasible or erroring
train-set solutions.
`DASBENCH_SOLVER_TIMEOUT_SECONDS` caps each generated solver call per instance;
set it to `0` to disable the guardrail. `DASBENCH_SOLVER_MEMORY_LIMIT_MB` caps
solver address-space usage in MiB and records memory-limit hits as failed
candidates; set it to `0` to disable the guardrail.

## Quick Start

Generate a MAXSAT dataset with default artifact placement:

```bash
python main.py generate \
  --problem maxsat \
  --family latent_backdoor_mixture_v1
```

Generate a small MIS dataset explicitly:

```bash
python main.py generate \
  --problem mis \
  --family motif_bridge_mixture_v1 \
  --dataset-id smoke_mis \
  --instance-param num_vertices=18 \
  --train-size 32 \
  --validation-size 16 \
  --test-size 16
```

Run baselines plus synthesis on an existing dataset:

```bash
python main.py run-agent \
  --dataset-dir artifacts/datasets/mis/motif_bridge_mixture_v1/smoke_mis \
  --generator template \
  --mode beam \
  --iterations 3 \
  --beam-width 3
```

Disable the default Gurobi industrial baseline or tune its limits:

```bash
python main.py run-agent \
  --dataset-dir artifacts/datasets/mis/motif_bridge_mixture_v1/smoke_mis \
  --no-gurobi-baseline
```

```bash
python main.py run-agent \
  --dataset-dir artifacts/datasets/mis/motif_bridge_mixture_v1/smoke_mis \
  --gurobi-time-limit-seconds 30 \
  --gurobi-threads 1
```

Enable optional external exact solvers with `auto` discovery. MaxSAT rows use Hermax for Open-WBO, UWrMaxSAT, EvalMaxSAT, and WMaxCDCL when no executable is configured. SCIP-backed rows use PySCIPOpt when no SCIP executable is configured, and HiGHS-backed rows use `highspy` when no HiGHS executable is configured. Binary paths remain supported for solvers without native Python APIs or when you explicitly want the CLI backend:

```bash
export DASBENCH_OPEN_WBO_BIN=/path/to/open-wbo
export DASBENCH_UWRMAXSAT_BIN=/path/to/uwrmaxsat
export DASBENCH_EVALMAXSAT_BIN=/path/to/evalmaxsat
export DASBENCH_MAXHS_BIN=/path/to/maxhs
export DASBENCH_WMAXCDCL_BIN=/path/to/wmaxcdcl
export DASBENCH_KAMIS_EXACT_BIN=/path/to/kamis-exact
export DASBENCH_SCIP_BIN=/path/to/scip
export DASBENCH_CONCORDE_BIN=/path/to/concorde
export DASBENCH_HIGHS_BIN=/path/to/highs

python main.py run-agent \
  --dataset-dir artifacts/datasets/maxsat/last_clause_signal_v1/smoke_maxsat \
  --external-exact-baselines auto \
  --external-time-limit-seconds 60 \
  --external-threads 1
```

In `auto` mode, configured binaries take precedence. Without a configured binary, `open_wbo_exact`, `uwrmaxsat_exact`, `evalmaxsat_exact`, and `wmaxcdcl_exact` use `hermax`; `scip_*` baselines use `pyscipopt`; `highs_*` baselines use `highspy`. MaxHS, KaMIS, and Concorde remain CLI-based unless a future Python backend is available.

Write a repeated benchmark report for a completed run:

```bash
python main.py report \
  --dataset-dir artifacts/datasets/mis/motif_bridge_mixture_v1/smoke_mis \
  --agent-run-dir artifacts/agent_runs/mis/motif_bridge_mixture_v1/20260416_120000 \
  --repeats 10
```

Run the full benchmark flow end to end:

```bash
python main.py benchmark \
  --problem mds \
  --family gateway_overlap_cover_v1 \
  --generator template \
  --instance-param num_vertices=20 \
  --train-size 64 \
  --validation-size 32 \
  --test-size 32
```

Run a TSP benchmark with the new Euclidean families:

```bash
python main.py benchmark \
  --problem tsp \
  --family latent_metric_mixture_v1 \
  --generator template \
  --instance-param num_cities=12 \
  --train-size 64 \
  --validation-size 32 \
  --test-size 32
```

Run all families for one problem with default settings:

```bash
python main.py benchmark \
  --problem maxsat \
  --all-families
```

Run all registered families across all problems:

```bash
python main.py benchmark --all-families
```

Suite runs execute targets in parallel by default. You can cap concurrency if needed:

```bash
python main.py benchmark --all-families --max-parallel 4
```

This writes per-family datasets, agent runs, and reports under the normal artifact roots, and also writes a suite summary JSON plus per-target log files under `artifacts/reports/suites/<suite_id>/`.

Run paper-facing sweep suites from `benchmarks/`:

```bash
python -m benchmarks.main_paper_benchmark --max-workers 21
python -m benchmarks.sample_size_sweep --validation-size 32 --max-workers 4
python -m benchmarks.problem_size_sweep --max-workers 4
python -m benchmarks.candidate_count_sweep --max-workers 4
```

The headline paper results use the main paper benchmark alias above, which wraps
`benchmarks.second_scale_benchmark_v2`. Existing result artifacts use the historical internal
condition id `seconds_scale_v2`; this is preserved for compatibility with saved runs.

Additional ablations and the PACE 2025 diagnostic are documented in [benchmarks/README.md](benchmarks/README.md)
and [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Generated datasets, solver candidates, reports, and large
result bundles are intentionally not committed; regenerate them with the documented commands or
provide them through an anonymous external artifact archive.

## Supported Families

### Coloring

| Family | What It Is | Hidden Rule / Exploitable Signal |
| --- | --- | --- |
| `cluster_ring_mix_v1` | Smoke family with vertex blocks arranged as local multipartite clusters, plus ring-style bridges between neighboring blocks. | Each block has a planted 3- or 4-color palette permutation. Edges are mostly between different planted colors, and bridge edges preserve the same planted coloring. The useful rule is to infer a stable block palette/order rather than color greedily from local degree alone. |
| `planted_palette_overlap_v1` | Paper family with overlapping block palettes and a latent regime that changes which color-pair interactions are dense. | A planted 4-coloring exists across all blocks. Blocks use shifted, overlapping palettes, and each instance samples a hidden regime that changes color-pair edge probabilities and bridge-pair density. The useful rule is to recover the planted palette structure despite overlapping local statistics. |
| `separator_palette_trap_v1` | Paper family with block-local palette gaps and separator vertices between adjacent blocks. | Each block has a planted 4-color permutation, but one color pair is intentionally sparse inside the block. Boundary separator vertices connect across blocks while exempting specific colors, creating long-range color-reuse constraints. The useful rule is to infer the global palette/separator structure, not just local greedy choices. |

### MAXSAT

| Family | What It Is | Hidden Rule / Exploitable Signal |
| --- | --- | --- |
| `last_clause_signal_v1` | Smoke family where one anchor clause controls a planted assignment. | The final clause encodes a 3-bit anchor pattern on variables `1`, `2`, and `3`. Every other variable copies or negates one anchor bit according to a fixed hidden rule table. Clauses mostly agree with the induced assignment. |
| `latent_backdoor_mixture_v1` | Paper family with three hidden regimes and regime-specific variable subsets. | Each instance samples a latent regime. Within that regime, variables `4..n` are determined by hidden Boolean functions of the three anchor bits, including single-bit and parity functions, with optional negation and small noise. Early clauses emphasize regime-specific backdoor blocks, bridge clauses emphasize different blocks, and marginal literal frequencies overlap across regimes. |
| `community_parity_overlay_v1` | Paper family with variable communities, parity-style community rules, and sparse bridge clauses. | Variables `4..n` are partitioned into four communities. Each community shares a hidden value given by a Boolean function of the anchor bits, such as `x1`, `x2`, `x1 xor x2`, or a three-bit parity, optionally negated. Most clauses are intra-community, while bridge clauses couple specific community pairs. |

### MIS

| Family | What It Is | Hidden Rule / Exploitable Signal |
| --- | --- | --- |
| `clique_path_mix_v1` | Smoke family with alternating clique-heavy and path-heavy blocks plus sparse bridges and noise. | A hidden two-way regime flips which block parity is clique-like versus path-like. The useful rule is to identify each block type: clique blocks contribute at most one independent-set vertex, while path blocks allow alternating selections subject to bridge edges. |
| `motif_bridge_mixture_v1` | Paper family assembled from latent motif libraries: cliques, cycles, bicliques, crowns, sparse bridges, and light noise. | Each instance samples one of three regimes. The regime determines the motif sequence assigned to the vertex blocks. Each motif has a different MIS structure, and sparse bridge or skip edges couple adjacent motifs. The useful rule is motif-aware decomposition plus bridge handling. |
| `core_fringe_trap_v1` | Paper family with a dense core and low-degree fringe gadgets attached to core vertices. | The core is a clique-like trap: high-degree core vertices look important but usually cannot all be selected. Fringe groups use regime-dependent path, cycle, or trap gadgets and attach to core anchors. The useful rule is to favor compatible fringe selections while accounting for which core attachments block them. |

### MDS

| Family | What It Is | Hidden Rule / Exploitable Signal |
| --- | --- | --- |
| `star_cluster_cover_v1` | Smoke family with star-like clusters and sparse hub connectors. | Each cluster has a hidden hub at the first vertex of the cluster, and those hubs dominate most of their local cluster. The useful rule is to select the stable cluster hubs; connector edges between hubs add noise but do not remove the hub-cover structure. |
| `gateway_overlap_cover_v1` | Paper family where cluster hubs and gateway vertices create overlapping domination coverage. | Each cluster has a hub and a gateway. Gateways link to neighboring gateways and cover selected vertices in adjacent clusters, so raw degree can be misleading. The useful rule is overlap-aware coverage: combine hubs and gateways so one selected vertex can help dominate neighboring clusters. |
| `geometric_cluster_cover_v1` | Paper family with hidden random-geometric cluster layouts, heterogeneous density, and noisy connector edges. | Each instance samples one of several geometric center layouts. Edges come from distance thresholds plus periodic connector edges. The useful rule is to infer local geometric neighborhoods and connector roles, then choose dominators by marginal coverage and redundancy rather than degree alone. |

### Packing LP

| Family | What It Is | Hidden Rule / Exploitable Signal |
| --- | --- | --- |
| `single_bottleneck_fractional_v1` | Smoke bounded multidimensional packing LP where one hidden resource is much tighter than the others. | A single recurring resource usually determines the optimal fractional cutoff. The useful rule is to infer the binding resource from capacity tightness and sort items by value per unit of that resource. |
| `latent_active_basis_v1` | Paper LP family with hidden regimes that share similar coefficient marginals but bind different resource pairs. | Each instance samples a latent dual-price regime. Values are noisy functions of the hidden resource-price vector, and capacities make a regime-specific resource pair active. The useful rule is to infer the active basis/dual prices instead of relying on aggregate value density. |
| `block_coupled_resource_v1` | Paper LP family with item/resource blocks and a sparse coupling resource. | Items have block-local high coefficients, but a shared coupling resource quietly limits the mix of attractive block-local choices. The useful rule is to detect block membership and penalize coupling-resource pressure. |

### MDKP

| Family | What It Is | Hidden Rule / Exploitable Signal |
| --- | --- | --- |
| `single_resource_density_v1` | Smoke multidimensional knapsack family where one resource makes simple density heuristics strong. | One hidden resource is consistently tight, while secondary resources are usually slack. The useful rule is value per unit of the recurring bottleneck with feasibility repair. |
| `latent_class_knapsack_v1` | Paper MDKP family with hidden item classes and regime-dependent bottleneck resources. | Items come from latent resource-consumption classes, and each instance has a hidden bottleneck resource that changes which classes are valuable. The useful rule is to cluster items by weight profile, infer the current bottleneck, and prefer low-pressure classes. |
| `decoy_complement_mixture_v1` | Paper MDKP family with high-value decoys and complementary item classes. | High-value decoys look attractive locally but consume a hidden scarce resource. Complementary classes combine better across non-scarce resources. The useful rule is to penalize scarce-resource burn and select complementary bundles rather than scalar density winners. |

### TSP

| Family | What It Is | Hidden Rule / Exploitable Signal |
| --- | --- | --- |
| `clustered_euclidean_v1` | Smoke Euclidean TSP family with balanced city clusters arranged around a ring. | Cities are sampled around four hidden ring centers with a shared phase. The useful rule is to recover the ring/cluster geometry and build tours that respect the circular cluster order while handling short intra-cluster visits. |
| `paired_ribbon_zigzag_v1` | Paper family with two noisy parallel ribbons, shuffled city order, and a hidden offset regime. | Cities lie on two parallel lines with either a small or larger stagger offset, and the whole structure may be transposed. The useful rule is to use PCA or clustering to recover the two ribbons, split the cities evenly, and traverse one ribbon in one direction and the other in reverse. |
| `latent_metric_mixture_v1` | Paper family mixing ring-cluster, ribbon, and barrier-bridge geometric regimes with overlapping scale statistics. | Each instance samples one of three latent geometric regimes: ring clusters, paired ribbons, or two separated sides connected by central bridge points. The useful rule is to classify the geometry from higher-order structure and choose the corresponding tour strategy. |

## Metrics

All problems report:

- `average_normalized_quality`
- `optimality_rate`
- `feasibility_rate`
- `average_runtime_ms`
- External exact baselines also report `proved_optimal_rate` and `average_external_runtime_ms` when enabled.

Problem-specific normalization:

- `coloring`: optimum number of colors / returned number of colors
- `mdkp`: returned item value / optimum item value
- `maxsat`: satisfied clauses / optimum satisfied clauses
- `mis`: independent set size / optimum independent set size
- `mds`: optimum dominating set size / returned dominating set size
- `packing_lp`: returned objective / optimum objective
- `tsp`: optimum tour length / returned tour length

## Notes

- Exact optima are computed at dataset-generation time and stored with each instance.
- OR-Tools is the exact backend for graph problems and the LP/IP packing pair (`GLOP` for `packing_lp`, CP-SAT for `mdkp`).
- Gurobi is integrated as a timed industrial baseline only; it does not replace stored-optimum generation.
- Gurobi diagnostics are written per split when enabled, for example `gurobi_timed_validation_diagnostics.jsonl`.
- Optional external exact baselines are discovery-based and skipped in `auto` mode when binaries are missing, including HiGHS for `packing_lp` and `mdkp`.
- The template generator is fully local and is the recommended smoke-test path.
- The LLM generator uses structured outputs and the system prompt in [dasbench/prompts/llm_system_prompt.txt](dasbench/prompts/llm_system_prompt.txt).
