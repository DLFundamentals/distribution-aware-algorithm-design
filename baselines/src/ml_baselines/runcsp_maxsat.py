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
from ml_baselines.tensorize import public_view
from ml_baselines.torch_utils import TorchUnavailableError, as_backend_tensor, require_torch, resolve_device, torch_available

RUN_CSP_MAXSAT_BASELINE_NAME = "ml_runcsp_maxsat"
EPS = 1e-9


@dataclass(frozen=True)
class RunCSPMaxSatConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    message_passing_steps: int = 16
    recurrent_layers: int = 1
    batch_size: int = 1
    samples: int = 16
    walksat_restarts: int = 4
    walksat_flips: int = 1000
    noise: float = 0.1
    entropy_weight: float = 0.01
    entropy_decay_epochs: int = 20
    binarization_weight: float = 0.001
    binarization_start_epoch: int = 50
    device: str = "cpu"
    seed: int = 0

    @classmethod
    def from_config(cls, config: MLBaselineConfig | dict[str, Any] | None) -> "RunCSPMaxSatConfig":
        base = cls()
        if config is None:
            return base
        payload = config.to_dict() if isinstance(config, MLBaselineConfig) else dict(config)
        walksat_flips = int(
            payload.get(
                "walksat_flips",
                payload.get("local_search_flips", payload.get("repair_budget", base.walksat_flips)),
            )
        )
        return cls(
            epochs=int(payload.get("epochs", base.epochs)),
            learning_rate=float(payload.get("learning_rate", payload.get("lr", base.learning_rate))),
            hidden_dim=int(payload.get("hidden_dim", base.hidden_dim)),
            message_passing_steps=int(
                payload.get("message_passing_steps", payload.get("layers", base.message_passing_steps))
            ),
            recurrent_layers=int(payload.get("recurrent_layers", base.recurrent_layers)),
            batch_size=int(payload.get("batch_size", base.batch_size)),
            samples=int(payload.get("samples", payload.get("inference_samples", base.samples))),
            walksat_restarts=int(payload.get("walksat_restarts", payload.get("inference_restarts", base.walksat_restarts))),
            walksat_flips=walksat_flips,
            noise=float(payload.get("noise", base.noise)),
            entropy_weight=float(payload.get("entropy_weight", base.entropy_weight)),
            entropy_decay_epochs=int(payload.get("entropy_decay_epochs", base.entropy_decay_epochs)),
            binarization_weight=float(payload.get("binarization_weight", base.binarization_weight)),
            binarization_start_epoch=int(payload.get("binarization_start_epoch", base.binarization_start_epoch)),
            device=str(payload.get("device", base.device)),
            seed=int(payload.get("seed", base.seed)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "hidden_dim": self.hidden_dim,
            "message_passing_steps": self.message_passing_steps,
            "recurrent_layers": self.recurrent_layers,
            "batch_size": self.batch_size,
            "samples": self.samples,
            "walksat_restarts": self.walksat_restarts,
            "walksat_flips": self.walksat_flips,
            "noise": self.noise,
            "entropy_weight": self.entropy_weight,
            "entropy_decay_epochs": self.entropy_decay_epochs,
            "binarization_weight": self.binarization_weight,
            "binarization_start_epoch": self.binarization_start_epoch,
            "device": self.device,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class RunCSPMaxSatTensor:
    instance_id: str
    num_variables: int
    num_clauses: int
    variable_features: Any
    clause_features: Any
    incidence_index: Any
    incidence_sign: Any
    clause_weights: Any


def _clause_weights(instance: dict[str, Any]) -> list[float]:
    clauses = instance["clauses"]
    weights = instance.get("weights", instance.get("clause_weights"))
    if weights is None:
        return [1.0] * len(clauses)
    if len(weights) != len(clauses):
        raise ValueError("MaxSAT clause weights must match the number of clauses.")
    return [float(value) for value in weights]


def tensorize_runcsp_maxsat_instance(
    instance: dict[str, Any],
    *,
    backend: str = "auto",
    device: str | None = None,
) -> RunCSPMaxSatTensor:
    public = public_view(instance)
    num_variables = int(public["num_variables"])
    clauses = public["clauses"]
    num_clauses = len(clauses)
    weights = _clause_weights(public)
    max_weight = max(weights) if weights else 1.0
    mean_weight = sum(weights) / max(float(len(weights)), 1.0)

    positive = [0] * num_variables
    negative = [0] * num_variables
    sources: list[int] = []
    targets: list[int] = []
    signs: list[float] = []
    clause_features: list[list[float]] = []
    max_clause_length = max((len(clause) for clause in clauses), default=1)
    for clause_index, clause in enumerate(clauses):
        pos_count = 0
        for literal in clause:
            variable = abs(int(literal)) - 1
            sign = 1.0 if int(literal) > 0 else -1.0
            if sign > 0:
                positive[variable] += 1
                pos_count += 1
            else:
                negative[variable] += 1
            sources.append(variable)
            targets.append(clause_index)
            signs.append(sign)
        length = len(clause)
        weight = weights[clause_index]
        clause_features.append(
            [
                float(length),
                float(length) / max(float(max_clause_length), 1.0),
                float(pos_count) / max(float(length), 1.0),
                float(length - pos_count) / max(float(length), 1.0),
                weight / max(max_weight, EPS),
                weight / max(mean_weight, EPS),
                1.0,
            ]
        )
    max_occurrences = max((positive[index] + negative[index] for index in range(num_variables)), default=1)
    variable_features = [
        [
            float(positive[index]),
            float(negative[index]),
            float(positive[index] + negative[index]) / max(float(max_occurrences), 1.0),
            float(positive[index] - negative[index]) / max(float(positive[index] + negative[index]), 1.0),
            1.0,
        ]
        for index in range(num_variables)
    ]
    return RunCSPMaxSatTensor(
        instance_id=str(public.get("id", "")),
        num_variables=num_variables,
        num_clauses=num_clauses,
        variable_features=as_backend_tensor(variable_features, dtype="float", device=device, backend=backend),
        clause_features=as_backend_tensor(clause_features, dtype="float", device=device, backend=backend),
        incidence_index=as_backend_tensor([sources, targets], dtype="long", device=device, backend=backend),
        incidence_sign=as_backend_tensor(signs, dtype="float", device=device, backend=backend),
        clause_weights=as_backend_tensor(weights, dtype="float", device=device, backend=backend),
    )


class RunCSPMaxSatNet:
    """RUN-CSP-style recurrent factor graph network for MaxSAT clauses."""

    def __new__(cls, *args: Any, **kwargs: Any):
        torch = require_torch()
        nn = torch.nn

        class _RunCSPMaxSatNet(nn.Module):
            def __init__(
                self,
                variable_dim: int,
                clause_dim: int,
                hidden_dim: int,
                message_passing_steps: int,
                recurrent_layers: int,
            ) -> None:
                super().__init__()
                self.variable_input = nn.Linear(variable_dim, hidden_dim)
                self.clause_input = nn.Linear(clause_dim, hidden_dim)
                self.variable_to_clause = nn.Linear(hidden_dim * 2 + 1, hidden_dim)
                self.clause_to_variable = nn.Linear(hidden_dim * 2 + 1, hidden_dim)
                self.clause_grus = nn.ModuleList(nn.GRUCell(hidden_dim, hidden_dim) for _ in range(max(1, recurrent_layers)))
                self.variable_grus = nn.ModuleList(nn.GRUCell(hidden_dim, hidden_dim) for _ in range(max(1, recurrent_layers)))
                self.output = nn.Linear(hidden_dim, 1)
                self.message_passing_steps = max(1, int(message_passing_steps))
                self.activation = nn.ReLU()

            def _mean_aggregate(self, messages: Any, index: Any, size: int) -> Any:
                torch = require_torch()
                output = torch.zeros((size, messages.shape[-1]), dtype=messages.dtype, device=messages.device)
                output.index_add_(0, index, messages)
                degree = torch.zeros((size, 1), dtype=messages.dtype, device=messages.device)
                degree.index_add_(0, index, torch.ones((messages.shape[0], 1), dtype=messages.dtype, device=messages.device))
                return output / degree.clamp_min(1.0)

            def forward(self, variable_features: Any, clause_features: Any, incidence_index: Any, incidence_sign: Any) -> Any:
                variable_hidden = self.activation(self.variable_input(variable_features))
                clause_hidden = self.activation(self.clause_input(clause_features))
                if incidence_index.numel() == 0:
                    return self.output(variable_hidden).squeeze(-1)

                variable_index = incidence_index[0].long()
                clause_index = incidence_index[1].long()
                signs = incidence_sign.reshape(-1, 1).to(variable_hidden.dtype)
                num_variables = int(variable_features.shape[0])
                num_clauses = int(clause_features.shape[0])

                for _ in range(self.message_passing_steps):
                    v_to_c_input = torch.cat([variable_hidden[variable_index], clause_hidden[clause_index], signs], dim=-1)
                    v_to_c = self.activation(self.variable_to_clause(v_to_c_input))
                    clause_aggregate = self._mean_aggregate(v_to_c, clause_index, num_clauses)
                    for gru in self.clause_grus:
                        clause_hidden = gru(clause_aggregate, clause_hidden)

                    c_to_v_input = torch.cat([clause_hidden[clause_index], variable_hidden[variable_index], signs], dim=-1)
                    c_to_v = self.activation(self.clause_to_variable(c_to_v_input))
                    variable_aggregate = self._mean_aggregate(c_to_v, variable_index, num_variables)
                    for gru in self.variable_grus:
                        variable_hidden = gru(variable_aggregate, variable_hidden)

                return self.output(variable_hidden).squeeze(-1)

        return _RunCSPMaxSatNet(*args, **kwargs)


def _soft_clause_satisfaction(logits: Any, tensor: RunCSPMaxSatTensor) -> Any:
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


def _loss(model: Any, tensor: RunCSPMaxSatTensor, config: RunCSPMaxSatConfig, *, epoch: int) -> Any:
    logits = model(tensor.variable_features, tensor.clause_features, tensor.incidence_index, tensor.incidence_sign)
    probabilities = require_torch().sigmoid(logits)
    clause_satisfaction = _soft_clause_satisfaction(logits.reshape(-1), tensor)
    weighted_satisfied = (clause_satisfaction * tensor.clause_weights).sum()
    denominator = tensor.clause_weights.sum().clamp_min(1.0)
    expected_fraction = weighted_satisfied / denominator
    entropy_scale = 0.0
    if config.entropy_decay_epochs > 0:
        entropy_scale = config.entropy_weight * max(0.0, 1.0 - (epoch - 1) / float(config.entropy_decay_epochs))
    binarization_scale = config.binarization_weight if epoch >= config.binarization_start_epoch else 0.0
    binarization = (probabilities * (1.0 - probabilities)).mean()
    return -expected_fraction - entropy_scale * _entropy(probabilities) + binarization_scale * binarization


def weighted_satisfied_score(instance: dict[str, Any], assignment: list[bool]) -> float:
    public = public_view(instance)
    weights = _clause_weights(public)
    score = 0.0
    for clause_index, clause in enumerate(public["clauses"]):
        if any((int(literal) > 0 and assignment[abs(int(literal)) - 1]) or (int(literal) < 0 and not assignment[abs(int(literal)) - 1]) for literal in clause):
            score += weights[clause_index]
    return float(score)


def unsatisfied_clause_indices(instance: dict[str, Any], assignment: list[bool]) -> list[int]:
    public = public_view(instance)
    unsatisfied: list[int] = []
    for clause_index, clause in enumerate(public["clauses"]):
        if not any((int(literal) > 0 and assignment[abs(int(literal)) - 1]) or (int(literal) < 0 and not assignment[abs(int(literal)) - 1]) for literal in clause):
            unsatisfied.append(clause_index)
    return unsatisfied


def _polarity_majority_seed(instance: dict[str, Any]) -> list[bool]:
    public = public_view(instance)
    positive = [0] * int(public["num_variables"])
    negative = [0] * int(public["num_variables"])
    for clause in public["clauses"]:
        for literal in clause:
            if int(literal) > 0:
                positive[abs(int(literal)) - 1] += 1
            else:
                negative[abs(int(literal)) - 1] += 1
    return [positive[index] > negative[index] for index in range(int(public["num_variables"]))]


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
    rng = random.Random(f"runcsp-maxsat:{public.get('id', '')}:{seed}")
    for _ in range(max(0, int(samples))):
        add([rng.random() < value for value in clipped])
    return candidates


def bounded_walksat_local_search(
    instance: dict[str, Any],
    start_assignment: list[bool],
    *,
    max_flips: int,
    noise: float = 0.1,
    seed: int = 0,
) -> list[bool]:
    public = public_view(instance)
    assignment = [bool(value) for value in start_assignment]
    if max_flips <= 0:
        return assignment
    rng = random.Random(f"runcsp-walksat:{public.get('id', '')}:{seed}")
    best_assignment = list(assignment)
    best_score = weighted_satisfied_score(public, best_assignment)
    current_score = best_score
    for _ in range(int(max_flips)):
        unsatisfied = unsatisfied_clause_indices(public, assignment)
        if not unsatisfied:
            break
        clause_index = rng.choice(unsatisfied)
        variables = sorted({abs(int(literal)) - 1 for literal in public["clauses"][clause_index]})
        if not variables:
            break
        if rng.random() < max(0.0, min(1.0, float(noise))):
            flip_variable = rng.choice(variables)
        else:
            best_variable = variables[0]
            best_candidate_score = -math.inf
            for variable in variables:
                assignment[variable] = not assignment[variable]
                candidate_score = weighted_satisfied_score(public, assignment)
                assignment[variable] = not assignment[variable]
                if candidate_score > best_candidate_score + EPS or (
                    abs(candidate_score - best_candidate_score) <= EPS and variable < best_variable
                ):
                    best_candidate_score = candidate_score
                    best_variable = variable
            flip_variable = best_variable
        assignment[flip_variable] = not assignment[flip_variable]
        current_score = weighted_satisfied_score(public, assignment)
        if current_score > best_score + EPS:
            best_score = current_score
            best_assignment = list(assignment)
    return best_assignment


def decode_runcsp_assignment(
    instance: dict[str, Any],
    probabilities: list[float],
    *,
    samples: int = 16,
    walksat_restarts: int = 4,
    walksat_flips: int = 1000,
    noise: float = 0.1,
    seed: int = 0,
) -> list[bool]:
    public = public_view(instance)
    candidates = candidate_assignments_from_probabilities(public, probabilities, samples=samples, seed=seed)
    candidates = sorted(candidates, key=lambda candidate: weighted_satisfied_score(public, candidate), reverse=True)
    best_assignment = candidates[0] if candidates else [False] * int(public["num_variables"])
    best_score = weighted_satisfied_score(public, best_assignment)
    for restart_index, candidate in enumerate(candidates[: max(1, int(walksat_restarts))]):
        improved = bounded_walksat_local_search(
            public,
            candidate,
            max_flips=walksat_flips,
            noise=noise,
            seed=seed + restart_index,
        )
        score = weighted_satisfied_score(public, improved)
        if score > best_score + EPS:
            best_assignment = improved
            best_score = score
    return [bool(value) for value in best_assignment]


def fit(
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    config: RunCSPMaxSatConfig | MLBaselineConfig | dict[str, Any] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> TrainedState:
    if not torch_available():
        raise TorchUnavailableError("PyTorch is required for the RUN-CSP MaxSAT baseline.")
    torch = require_torch()
    resolved_config = config if isinstance(config, RunCSPMaxSatConfig) else RunCSPMaxSatConfig.from_config(config)
    seed_everything(resolved_config.seed)
    device = resolve_device(resolved_config.device)
    if not train_instances:
        raise ValueError("The RUN-CSP MaxSAT baseline requires public training instances.")

    public_train = [public_view(instance) for instance in train_instances]
    public_val = [public_view(instance) for instance in (val_instances or [])]
    train_tensors = [tensorize_runcsp_maxsat_instance(instance, backend="torch", device=str(device)) for instance in public_train]
    val_tensors = [tensorize_runcsp_maxsat_instance(instance, backend="torch", device=str(device)) for instance in public_val]
    variable_dim = int(train_tensors[0].variable_features.shape[-1])
    clause_dim = int(train_tensors[0].clause_features.shape[-1])
    model = RunCSPMaxSatNet(
        variable_dim,
        clause_dim,
        resolved_config.hidden_dim,
        resolved_config.message_passing_steps,
        resolved_config.recurrent_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=resolved_config.learning_rate)
    logger = MetricsLogger(Path(metrics_path) if metrics_path is not None else None)
    best_state: dict[str, Any] | None = None
    best_loss = math.inf

    for epoch in range(1, resolved_config.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for tensor in train_tensors:
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, tensor, resolved_config, epoch=epoch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
        train_loss = sum(train_losses) / max(1, len(train_losses))
        val_loss = None
        if val_tensors:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for tensor in val_tensors:
                    loss = _loss(model, tensor, resolved_config, epoch=epoch)
                    val_losses.append(float(loss.detach().cpu().item()))
            val_loss = sum(val_losses) / max(1, len(val_losses))
        selection_loss = train_loss if val_loss is None else val_loss
        logger.log(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "problem": "maxsat",
        "baseline_name": RUN_CSP_MAXSAT_BASELINE_NAME,
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


def _probabilities_from_model(model: Any, instance: dict[str, Any], config: RunCSPMaxSatConfig) -> list[float]:
    torch = require_torch()
    device = resolve_device(config.device)
    tensor = tensorize_runcsp_maxsat_instance(public_view(instance), backend="torch", device=str(device))
    model.eval()
    with torch.no_grad():
        logits = model(tensor.variable_features, tensor.clause_features, tensor.incidence_index, tensor.incidence_sign).reshape(-1)
        probabilities = torch.sigmoid(logits)
    return [float(value) for value in probabilities.detach().cpu().tolist()]


def solve(
    instance: dict[str, Any],
    trained_state: TrainedState,
    config: RunCSPMaxSatConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> list[bool]:
    resolved_config = config if isinstance(config, RunCSPMaxSatConfig) else RunCSPMaxSatConfig.from_config(config)
    probabilities = _probabilities_from_model(trained_state.payload["model"], instance, resolved_config)
    return decode_runcsp_assignment(
        instance,
        probabilities,
        samples=resolved_config.samples,
        walksat_restarts=resolved_config.walksat_restarts,
        walksat_flips=resolved_config.walksat_flips,
        noise=resolved_config.noise,
        seed=resolved_config.seed,
    )


def build_runcsp_maxsat_baselines(
    problem_name: str,
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    *,
    artifact_dir: str | Path | None = None,
    config: RunCSPMaxSatConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    if problem_name != "maxsat" or not torch_available():
        return {}
    root = Path(artifact_dir) if artifact_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / f"{RUN_CSP_MAXSAT_BASELINE_NAME}.pt" if root is not None else None
    metrics_path = root / f"{RUN_CSP_MAXSAT_BASELINE_NAME}_metrics.jsonl" if root is not None else None
    resolved_config = config if isinstance(config, RunCSPMaxSatConfig) else RunCSPMaxSatConfig.from_config(config)
    trained_state = fit(
        train_instances,
        val_instances or [],
        resolved_config,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )

    def solver(instance: dict[str, Any]) -> list[bool]:
        return solve(instance, trained_state, resolved_config)

    return {RUN_CSP_MAXSAT_BASELINE_NAME: solver}
