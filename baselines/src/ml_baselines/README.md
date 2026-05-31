# Trainable ML Baselines

This package contains lightweight trainable baselines for DasBench distributions.  They train only on public train/validation instances: fields with private prefixes such as `optimum_*` and `_...` are stripped before tensorization and model fitting.  Validation scores may be used for hyperparameter selection; test instances are evaluated once with the existing verifier/scorer.

## Baselines

- `ml_gnn_mis_score_repair`: manual PyTorch message passing over graph features, unsupervised independent-set loss, greedy feasible repair.
- `ml_gnn_mds_score_repair`: manual PyTorch message passing, unsupervised domination loss, greedy cover/prune repair.
- `ml_pignn_coloring`: physics-inspired graph coloring baseline with a Potts-model conflict loss over color logits, fixed-budget repair, and DSATUR feasibility fallback.
- `ml_gnn_maxsat_assignment`: variable-clause bipartite message passing, soft expected-satisfied-clause objective, threshold/sample/polarity seeds plus bounded flips.
- `ml_runcsp_maxsat`: RUN-CSP-style recurrent variable/factor message passing for MaxSAT, unsupervised expected satisfied-clause loss, and bounded stochastic WalkSAT decoding.
- `ml_mdkp_item_scorer`: item/resource MLP with learned resource prices, Lagrangian-style unsupervised binary relaxation, greedy add/drop/swap repair.
- `ml_drl_mdkp`: actor-critic sequential MDKP constructor trained with public-data rewards, public heuristic initialization at inference, masked feasible item actions, and bounded repair.
- `ml_packinglp_item_fraction`: item/resource MLP with learned resource prices, unsupervised fractional packing relaxation, scaling/projection and density fill.
- `ml_tsp_neural_constructor`: edge-heatmap MLP over pairwise city features, self-trained on cheap heuristic plus bounded 2-opt pseudo-label tours, learned/classical candidate construction plus bounded 2-opt.

## Runner

Run a generated dataset target:

```bash
python -m baselines.src.ml_baselines.run_ml_baselines \
  --problem mis \
  --target clique_path_mix_v1/20260426_173348 \
  --seed 0 \
  --device cpu \
  --output-dir artifacts/ml_baseline_runs
```

Run every generated target for one problem:

```bash
python -m baselines.src.ml_baselines.run_ml_baselines \
  --problem tsp \
  --target all \
  --seed 0
```

Run with explicit split paths:

```bash
python -m baselines.src.ml_baselines.run_ml_baselines \
  --problem maxsat \
  --train-split artifacts/datasets/maxsat/last_clause_signal_v1/smoke/train.jsonl \
  --validation-split artifacts/datasets/maxsat/last_clause_signal_v1/smoke/validation.jsonl \
  --test-split artifacts/datasets/maxsat/last_clause_signal_v1/smoke/test.jsonl \
  --output-dir artifacts/ml_baseline_runs \
  --seed 0
```

Equivalent invocation if you prefer putting `baselines/src` on `PYTHONPATH`:

```bash
PYTHONPATH=baselines/src python -m ml_baselines.run_ml_baselines --problem mdkp --target all
```

### Per-Problem Commands

Each command trains on public train, uses validation for logging/optional selection, and evaluates the selected model once on test:

```bash
python -m baselines.src.ml_baselines.run_ml_baselines --problem mis --target all --seed 0 --device cpu
python -m baselines.src.ml_baselines.run_ml_baselines --problem mds --target all --seed 0 --device cpu
python -m baselines.src.ml_baselines.run_ml_baselines --problem coloring --target all --seed 0 --device cpu
python -m baselines.src.ml_baselines.run_ml_baselines --problem maxsat --target all --seed 0 --device cpu
python -m baselines.src.ml_baselines.run_ml_baselines --problem mdkp --target all --seed 0 --device cpu
python -m baselines.src.ml_baselines.run_ml_baselines --problem packing_lp --target all --seed 0 --device cpu
python -m baselines.src.ml_baselines.run_ml_baselines --problem tsp --target all --seed 0 --device cpu
```

Restrict a run to one baseline when a problem has more than one registered ML baseline or when debugging:

```bash
python -m baselines.src.ml_baselines.run_ml_baselines \
  --problem maxsat \
  --target last_clause_signal_v1/20260426_173348 \
  --baseline ml_gnn_maxsat_assignment
```

## Config Overrides

Pass JSON or YAML with `--config`. Plain top-level keys apply to all baselines; `defaults`, `problems`, and `baselines` can scope overrides.

```yaml
defaults:
  epochs: 50
  learning_rate: 0.001
  hidden_dim: 64
baselines:
  ml_tsp_neural_constructor:
    hidden_dim: 128
    candidates: 4
    two_opt_budget: 256
```

Validation-based selection is enabled with `--select-on-validation`. Provide a `search`, `hyperparameter_grid`, or per-baseline `baseline_search` list:

```yaml
defaults:
  epochs: 30
baseline_search:
  ml_gnn_maxsat_assignment:
    - hidden_dim: 32
      lr: 0.001
    - hidden_dim: 64
      lr: 0.0005
```

Each trial trains on the public train split, evaluates on validation, selects by validation quality/feasibility/optimality/runtime, and evaluates the selected model once on test.

## Default Hyperparameters

| Baseline | Key Defaults |
| --- | --- |
| `ml_gnn_mis_score_repair` | `epochs=50`, `hidden_dim=64`, `layers=3`, `learning_rate=1e-3`, `inference_restarts=4`, `repair_budget=64`, `seed=0` |
| `ml_gnn_mds_score_repair` | `epochs=50`, `hidden_dim=64`, `layers=3`, `learning_rate=1e-3`, `inference_restarts=4`, `repair_budget=64`, `seed=0` |
| `ml_pignn_coloring` | `epochs=100`, `hidden_dim=64`, `layers=3`, `learning_rate=1e-3`, `inference_restarts=8`, `repair_budget=256`, `temperature_start=1.0`, `temperature_end=0.2`, `seed=0` |
| `ml_gnn_maxsat_assignment` | `epochs=50`, `hidden_dim=64`, `layers=3`, `learning_rate=1e-3`, `samples=8`, `walksat_flips=1000`, `seed=0` |
| `ml_runcsp_maxsat` | `epochs=100`, `hidden_dim=64`, `message_passing_steps=16`, `recurrent_layers=1`, `learning_rate=1e-3`, `samples=16`, `walksat_restarts=4`, `walksat_flips=1000`, `noise=0.1`, `seed=0` |
| `ml_mdkp_item_scorer` | `epochs=50`, `hidden_dim=64`, `learning_rate=1e-3`, `violation_penalty=10.0`, `repair_budget=64`, `seed=0` |
| `ml_drl_mdkp` | `episodes=5000`, `hidden_dim=128`, `learning_rate=3e-4`, `gamma=1.0`, `entropy_coef=0.01`, `value_coef=0.5`, `max_episode_steps=2*num_items`, `repair_budget=128`, `seed=0` |
| `ml_packinglp_item_fraction` | `epochs=50`, `hidden_dim=64`, `learning_rate=1e-3`, `violation_penalty=10.0`, `repair_budget=64`, `seed=0` |
| `ml_tsp_neural_constructor` | `epochs=50`, `hidden_dim=128`, `learning_rate=1e-3`, `candidates=4`, `two_opt_budget=256`, `seed=0` |

All baselines accept `device` and deterministic `seed` overrides.  Search/repair budgets are fixed by these config values and are not allowed to grow with solver progress.

## Outputs

For each run id, the runner writes:

- `aggregate_results.csv`: one row per target/baseline, intended for paper-table aggregation.
- `aggregate_results.json`: JSON equivalent of the aggregate rows.
- `run_summary.json`: run metadata and output paths.
- `<problem>/<family>/<dataset_id>/<baseline>/<baseline>.pt`: selected checkpoint.
- `<problem>/<family>/<dataset_id>/<baseline>/selected_config.json`: selected config.
- `<problem>/<family>/<dataset_id>/<baseline>/<baseline>_train_metrics.jsonl`: training metrics for selected trial.
- `<problem>/<family>/<dataset_id>/<baseline>/validation_selection.json`: validation summaries for all trials.
- `<problem>/<family>/<dataset_id>/<baseline>/test_outputs.jsonl`: per-instance test solutions/scores/runtimes.
- `<problem>/<family>/<dataset_id>/<baseline>/test_outputs.csv`: CSV version of per-instance outputs.
- `trials/trial_*/`: checkpoint, config, train metrics, and validation outputs for every trained trial.

Training time is recorded separately as `selected_training_time_ms` and `total_training_time_ms`. Test runtime metrics measure only per-instance inference/scoring time.

## Aggregate CSV Schema

`aggregate_results.csv` columns:

```text
problem,family,dataset_id,target,baseline,split,seed,device,
num_train_instances,num_validation_instances,num_test_instances,
average_normalized_quality,average_objective_value,optimality_rate,
feasibility_rate,average_runtime_ms,error_count,selected_trial_index,
selected_validation_quality,selected_validation_feasibility_rate,
selected_training_time_ms,total_training_time_ms,trial_count,
checkpoint_path,train_metrics_path,test_outputs_path,test_outputs_csv_path,
run_dir,config_json
```

`test_outputs.csv` columns:

```text
problem,baseline,split,instance_id,objective_value,normalized_quality,
is_valid,is_feasible,is_optimal,runtime_ms,error,solution
```

## Limitations

- Models are intentionally small and one-instance-at-a-time; `batch_size` is accepted in configs but not fully batched yet.
- Hyperparameter search is an explicit list, not a Cartesian product expander.
- TSP uses an edge-heatmap self-training fallback rather than an attention/pointer decoder.
- `ml_drl_mdkp` uses public heuristic initialization instead of the solver-assisted initialization described in the reference paper.
- Checkpoints are saved for reproducibility, but the runner retrains for each invocation rather than loading old checkpoints automatically.
- Validation/test scoring requires generated split files with the usual scorer metadata, including optima where the problem scorer expects them.
