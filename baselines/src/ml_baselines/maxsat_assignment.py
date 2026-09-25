from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_baselines.checkpoint import save_checkpoint
from ml_baselines.config import MLBaselineConfig
from ml_baselines.interface import TrainedState
from ml_baselines.metrics import MetricsLogger
from ml_baselines.seeding import seed_everything
from ml_baselines.tensorize import public_view, tensorize_maxsat_instance
from ml_baselines.torch_utils import TorchUnavailableError, require_torch, resolve_device, torch_available

MAXSAT_BASELINE_NAME = "ml_gnn_maxsat_assignment"
EPS = 1e-9


@dataclass(frozen=True)
class MaxSatAssignmentConfig:
    epochs: int = 50
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    layers: int = 3
    batch_size: int = 1
    samples: int = 8
    walksat_flips: int = 1000
    repair_budget: int = 1000
    device: str = "cpu"
    seed: int = 0
    entropy_weight: float = 0.01
    entropy_decay_epochs: int = 10

    @classmethod
    def from_config(cls, config: MLBaselineConfig | dict[str, Any] | None) -> "MaxSatAssignmentConfig":
        base = cls()
        if config is None:
            return base
        payload = config.to_dict() if isinstance(config, MLBaselineConfig) else dict(config)
        walksat_flips = int(payload.get("walksat_flips", payload.get("repair_budget", base.walksat_flips)))
        return cls(
            epochs=int(payload.get("epochs", base.epochs)),
            learning_rate=float(payload.get("learning_rate", payload.get("lr", base.learning_rate))),
            hidden_dim=int(payload.get("hidden_dim", base.hidden_dim)),
            layers=int(payload.get("message_passing_layers", payload.get("layers", base.layers))),
            batch_size=int(payload.get("batch_size", base.batch_size)),
            samples=int(payload.get("samples", payload.get("inference_samples", base.samples))),
            walksat_flips=walksat_flips,
            repair_budget=int(payload.get("repair_budget", walksat_flips)),
            device=str(payload.get("device", base.device)),
            seed=int(payload.get("seed", base.seed)),
            entropy_weight=float(payload.get("entropy_weight", base.entropy_weight)),
            entropy_decay_epochs=int(payload.get("entropy_decay_epochs", base.entropy_decay_epochs)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "batch_size": self.batch_size,
            "samples": self.samples,
            "walksat_flips": self.walksat_flips,
            "repair_budget": self.repair_budget,
            "device": self.device,
            "seed": self.seed,
            "entropy_weight": self.entropy_weight,
            "entropy_decay_epochs": self.entropy_decay_epochs,
        }


class BipartiteMaxSatNet:
    """Small PyTorch-only variable-clause message passing network."""

    def __new__(cls, *args: Any, **kwargs: Any):
        torch = require_torch()
        nn = torch.nn

        class _BipartiteMaxSatNet(nn.Module):
            def __init__(self, variable_dim: int, clause_dim: int, hidden_dim: int, layers: int) -> None:
                super().__init__()
                self.variable_input = nn.Linear(variable_dim, hidden_dim)
                self.clause_input = nn.Linear(clause_dim, hidden_dim)
                self.var_to_clause = nn.Linear(hidden_dim + 1, hidden_dim)
                self.clause_update = nn.Linear(hidden_dim * 2, hidden_dim)
                self.clause_to_var = nn.Linear(hidden_dim + 1, hidden_dim)
                self.variable_update = nn.Linear(hidden_dim * 2, hidden_dim)
                self.output = nn.Linear(hidden_dim, 1)
                self.layers = max(1, int(layers))

            def forward(self, variable_features: Any, clause_features: Any, incidence_index: Any, incidence_sign: Any) -> Any:
                torch = require_torch()
                variable_hidden = torch.relu(self.variable_input(variable_features))
                clause_hidden = torch.relu(self.clause_input(clause_features))
                if incidence_index.numel() == 0:
                    return self.output(variable_hidden).squeeze(-1)

                variable_index = incidence_index[0].long()
                clause_index = incidence_index[1].long()
                signs = incidence_sign.reshape(-1, 1).to(variable_hidden.dtype)
                num_variables = int(variable_features.shape[0])
                num_clauses = int(clause_features.shape[0])
                hidden_dim = int(variable_hidden.shape[-1])

                for _ in range(self.layers):
                    variable_messages = torch.relu(self.var_to_clause(torch.cat([variable_hidden[variable_index], signs], dim=-1)))
                    clause_aggregate = torch.zeros((num_clauses, hidden_dim), dtype=variable_hidden.dtype, device=variable_hidden.device)
                    clause_aggregate.index_add_(0, clause_index, variable_messages)
                    clause_degrees = torch.zeros((num_clauses, 1), dtype=variable_hidden.dtype, device=variable_hidden.device)
                    clause_degrees.index_add_(0, clause_index, torch.ones((len(clause_index), 1), dtype=variable_hidden.dtype, device=variable_hidden.device))
                    clause_aggregate = clause_aggregate / clause_degrees.clamp_min(1.0)
                    clause_hidden = torch.relu(self.clause_update(torch.cat([clause_hidden, clause_aggregate], dim=-1)))

                    clause_messages = torch.relu(self.clause_to_var(torch.cat([clause_hidden[clause_index], signs], dim=-1)))
                    variable_aggregate = torch.zeros((num_variables, hidden_dim), dtype=variable_hidden.dtype, device=variable_hidden.device)
                    variable_aggregate.index_add_(0, variable_index, clause_messages)
                    variable_degrees = torch.zeros((num_variables, 1), dtype=variable_hidden.dtype, device=variable_hidden.device)
                    variable_degrees.index_add_(0, variable_index, torch.ones((len(variable_index), 1), dtype=variable_hidden.dtype, device=variable_hidden.device))
                    variable_aggregate = variable_aggregate / variable_degrees.clamp_min(1.0)
                    variable_hidden = torch.relu(self.variable_update(torch.cat([variable_hidden, variable_aggregate], dim=-1)))

                return self.output(variable_hidden).squeeze(-1)

        return _BipartiteMaxSatNet(*args, **kwargs)


def _clause_weight_values(instance: dict[str, Any], *, device: Any | None = None, dtype: Any | None = None) -> Any | None:
    weights = instance.get("weights", instance.get("clause_weights"))
    if weights is None:
        return None
    if len(weights) != len(instance["clauses"]):
        raise ValueError("MaxSAT clause weights must match the number of clauses.")
    if device is None:
        return [float(value) for value in weights]
    torch = require_torch()
    return torch.tensor([float(value) for value in weights], dtype=dtype or torch.float32, device=device)


def _soft_clause_satisfaction(logits: Any, tensor: Any) -> Any:
    torch = require_torch()
    probabilities = torch.sigmoid(logits)
    variable_index = tensor.incidence_index[0].long()
    clause_index = tensor.incidence_index[1].long()
    signs = tensor.incidence_sign
    clause_terms: list[list[Any]] = [[] for _ in range(int(tensor.num_clauses))]
    for edge_index in range(int(variable_index.numel())):
        variable = int(variable_index[edge_index].detach().cpu().item())
        clause = int(clause_index[edge_index].detach().cpu().item())
        sign = float(signs[edge_index].detach().cpu().item())
        literal_probability = probabilities[variable] if sign > 0 else 1.0 - probabilities[variable]
        clause_terms[clause].append(1.0 - literal_probability)

    satisfied = []
    for terms in clause_terms:
        if not terms:
            satisfied.append(torch.zeros((), dtype=probabilities.dtype, device=probabilities.device))
            continue
        unsatisfied_probability = torch.stack(terms).prod()
        satisfied.append(1.0 - unsatisfied_probability)
    return torch.stack(satisfied) if satisfied else torch.zeros((0,), dtype=probabilities.dtype, device=probabilities.device)


def _entropy(probabilities: Any) -> Any:
    torch = require_torch()
    clipped = probabilities.clamp(1e-6, 1.0 - 1e-6)
    return -(clipped * torch.log(clipped) + (1.0 - clipped) * torch.log(1.0 - clipped)).mean()


def _loss(model: Any, tensor: Any, instance: dict[str, Any], config: MaxSatAssignmentConfig, *, epoch: int) -> Any:
    torch = require_torch()
    logits = model(tensor.variable_features, tensor.clause_features, tensor.incidence_index, tensor.incidence_sign)
    clause_satisfaction = _soft_clause_satisfaction(logits.reshape(-1), tensor)
    weights = _clause_weight_values(instance, device=logits.device, dtype=logits.dtype)
    expected_satisfied = clause_satisfaction.sum() if weights is None else (clause_satisfaction * weights).sum()
    probabilities = torch.sigmoid(logits)
    entropy_scale = 0.0
    if config.entropy_decay_epochs > 0:
        entropy_scale = config.entropy_weight * max(0.0, 1.0 - (epoch - 1) / float(config.entropy_decay_epochs))
    return -expected_satisfied - entropy_scale * _entropy(probabilities)


def _weighted_satisfied_score(instance: dict[str, Any], assignment: list[bool]) -> float:
    clauses = instance["clauses"]
    weights = _clause_weight_values(instance)
    score = 0.0
    for clause_index, clause in enumerate(clauses):
        satisfied = False
        for literal in clause:
            value = assignment[abs(int(literal)) - 1]
            if (int(literal) > 0 and value) or (int(literal) < 0 and not value):
                satisfied = True
                break
        if satisfied:
            score += 1.0 if weights is None else float(weights[clause_index])
    return score


def _polarity_majority_seed(instance: dict[str, Any]) -> list[bool]:
    positive = [0] * int(instance["num_variables"])
    negative = [0] * int(instance["num_variables"])
    for clause in instance["clauses"]:
        for literal in clause:
            if int(literal) > 0:
                positive[abs(int(literal)) - 1] += 1
            else:
                negative[abs(int(literal)) - 1] += 1
    return [positive[index] > negative[index] for index in range(int(instance["num_variables"]))]


def candidate_assignments_from_probabilities(
    instance: dict[str, Any],
    probabilities: list[float],
    *,
    samples: int,
    seed: int,
) -> list[list[bool]]:
    public = public_view(instance)
    num_variables = int(public["num_variables"])
    clipped = [min(1.0, max(0.0, float(value))) for value in probabilities[:num_variables]]
    if len(clipped) != num_variables:
        raise ValueError(f"Expected {num_variables} assignment probabilities, got {len(probabilities)}.")

    candidates: list[list[bool]] = []

    def add(candidate: list[bool]) -> None:
        if candidate not in candidates:
            candidates.append(candidate)

    add([value >= 0.5 for value in clipped])
    add(_polarity_majority_seed(public))
    rng = random.Random(f"maxsat-ml:{public.get('id', '')}:{seed}")
    for _ in range(max(0, int(samples))):
        add([rng.random() < value for value in clipped])
    return candidates


def bounded_maxsat_local_search(
    instance: dict[str, Any],
    start_assignment: list[bool],
    *,
    max_flips: int,
) -> list[bool]:
    public = public_view(instance)
    assignment = list(start_assignment)
    if max_flips <= 0:
        return assignment
    current_score = _weighted_satisfied_score(public, assignment)
    num_variables = int(public["num_variables"])
    for _ in range(int(max_flips)):
        best_variable: int | None = None
        best_score = current_score
        for variable_index in range(num_variables):
            assignment[variable_index] = not assignment[variable_index]
            candidate_score = _weighted_satisfied_score(public, assignment)
            assignment[variable_index] = not assignment[variable_index]
            if candidate_score > best_score + EPS:
                best_score = candidate_score
                best_variable = variable_index
        if best_variable is None:
            break
        assignment[best_variable] = not assignment[best_variable]
        current_score = best_score
    return assignment


def decode_assignment(
    instance: dict[str, Any],
    probabilities: list[float],
    *,
    samples: int = 8,
    walksat_flips: int = 1000,
    seed: int = 0,
) -> list[bool]:
    public = public_view(instance)
    candidates = candidate_assignments_from_probabilities(public, probabilities, samples=samples, seed=seed)
    best_assignment: list[bool] | None = None
    best_score = -math.inf
    for candidate in candidates:
        improved = bounded_maxsat_local_search(public, candidate, max_flips=walksat_flips)
        score = _weighted_satisfied_score(public, improved)
        if best_assignment is None or score > best_score + EPS:
            best_assignment = improved
            best_score = score
    if best_assignment is None:
        return [False] * int(public["num_variables"])
    return [bool(value) for value in best_assignment]


def fit(
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    config: MaxSatAssignmentConfig | MLBaselineConfig | dict[str, Any] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> TrainedState:
    if not torch_available():
        raise TorchUnavailableError("PyTorch is required for the MaxSAT neural assignment baseline.")
    torch = require_torch()
    resolved_config = config if isinstance(config, MaxSatAssignmentConfig) else MaxSatAssignmentConfig.from_config(config)
    seed_everything(resolved_config.seed)
    device = resolve_device(resolved_config.device)
    if not train_instances:
        raise ValueError("The MaxSAT neural assignment baseline requires public training instances.")

    public_train = [public_view(instance) for instance in train_instances]
    public_val = [public_view(instance) for instance in (val_instances or [])]
    train_tensors = [tensorize_maxsat_instance(instance, backend="torch", device=str(device)) for instance in public_train]
    val_tensors = [tensorize_maxsat_instance(instance, backend="torch", device=str(device)) for instance in public_val]
    variable_dim = int(train_tensors[0].variable_features.shape[-1])
    clause_dim = int(train_tensors[0].clause_features.shape[-1])
    model = BipartiteMaxSatNet(variable_dim, clause_dim, resolved_config.hidden_dim, resolved_config.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=resolved_config.learning_rate)
    logger = MetricsLogger(Path(metrics_path) if metrics_path is not None else None)
    best_state: dict[str, Any] | None = None
    best_train_loss = math.inf

    for epoch in range(1, resolved_config.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for instance, tensor in zip(public_train, train_tensors, strict=True):
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, tensor, instance, resolved_config, epoch=epoch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
        train_loss = sum(train_losses) / max(1, len(train_losses))
        val_loss = None
        if val_tensors:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for instance, tensor in zip(public_val, val_tensors, strict=True):
                    loss = _loss(model, tensor, instance, resolved_config, epoch=epoch)
                    val_losses.append(float(loss.detach().cpu().item()))
            val_loss = sum(val_losses) / max(1, len(val_losses))
        logger.log(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "problem": "maxsat",
        "baseline_name": MAXSAT_BASELINE_NAME,
        "config": resolved_config.to_dict(),
        "train_instances": len(public_train),
        "validation_instances": len(public_val),
        "variable_dim": variable_dim,
        "clause_dim": clause_dim,
        "history": logger.rows,
        "weighted": any("weights" in instance or "clause_weights" in instance for instance in public_train),
    }
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            {
                "model_state_dict": model.state_dict(),
                "variable_dim": variable_dim,
                "clause_dim": clause_dim,
            },
            metadata=metadata,
            use_torch=True,
        )
    return TrainedState(
        payload={
            "model": model,
            "variable_dim": variable_dim,
            "clause_dim": clause_dim,
        },
        metadata=metadata,
    )


def _probabilities_from_model(model: Any, instance: dict[str, Any], config: MaxSatAssignmentConfig) -> list[float]:
    torch = require_torch()
    device = resolve_device(config.device)
    public = public_view(instance)
    tensor = tensorize_maxsat_instance(public, backend="torch", device=str(device))
    model.eval()
    with torch.no_grad():
        logits = model(tensor.variable_features, tensor.clause_features, tensor.incidence_index, tensor.incidence_sign).reshape(-1)
        probabilities = torch.sigmoid(logits)
    return [float(value) for value in probabilities.detach().cpu().tolist()]


def solve(
    instance: dict[str, Any],
    trained_state: TrainedState,
    config: MaxSatAssignmentConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> list[bool]:
    resolved_config = config if isinstance(config, MaxSatAssignmentConfig) else MaxSatAssignmentConfig.from_config(config)
    probabilities = _probabilities_from_model(trained_state.payload["model"], instance, resolved_config)
    return decode_assignment(
        instance,
        probabilities,
        samples=resolved_config.samples,
        walksat_flips=resolved_config.walksat_flips,
        seed=resolved_config.seed,
    )


def build_maxsat_assignment_baselines(
    problem_name: str,
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    *,
    artifact_dir: str | Path | None = None,
    config: MaxSatAssignmentConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    if problem_name != "maxsat" or not torch_available():
        return {}
    root = Path(artifact_dir) if artifact_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / f"{MAXSAT_BASELINE_NAME}.pt" if root is not None else None
    metrics_path = root / f"{MAXSAT_BASELINE_NAME}_metrics.jsonl" if root is not None else None
    resolved_config = config if isinstance(config, MaxSatAssignmentConfig) else MaxSatAssignmentConfig.from_config(config)
    trained_state = fit(
        train_instances,
        val_instances or [],
        resolved_config,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )

    def solver(instance: dict[str, Any]) -> list[bool]:
        return solve(instance, trained_state, resolved_config)

    return {MAXSAT_BASELINE_NAME: solver}
