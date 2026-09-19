"""ANYCSP-style learned baseline ("One Model, Any CSP", Toenshoff et al., IJCAI 2023).

A faithful compact reproduction in the DasBench baseline style (cf. ``runcsp_maxsat`` /
``pignn_coloring``).  ANYCSP's signature is a *single* recurrent network over a **constraint
value graph** -- nodes are (variable, value) assignment candidates plus one node per
constraint -- so one architecture handles any finite-domain CSP.  Here the same
``ANYCSPNet`` is trained and deployed on both graph coloring (domain = k colours) and MaxSAT
(domain = {false, true}).

Departures from the paper, consistent with the other reproductions in this package: the model
is distribution-trained on the public train/validation split and optimised with a differentiable
expected-satisfied-constraint objective over the recurrent value-graph (an ANYCSP-style soft
variant rather than the paper's REINFORCE rollouts), then decoded with the same bounded repair
used by the existing coloring / MaxSAT baselines.  Private ``optimum_*`` / ``_`` fields are
stripped via ``public_view`` before tensorisation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_baselines.checkpoint import save_checkpoint
from ml_baselines.config import MLBaselineConfig
from ml_baselines.interface import TrainedState
from ml_baselines.metrics import MetricsLogger
from ml_baselines.pignn_coloring import _dsatur_color_count, decode_pignn_coloring
from ml_baselines.runcsp_maxsat import decode_runcsp_assignment
from ml_baselines.seeding import seed_everything
from ml_baselines.tensorize import public_view
from ml_baselines.torch_utils import (
    TorchUnavailableError,
    as_backend_tensor,
    require_torch,
    resolve_device,
    torch_available,
)

ANYCSP_BASELINE_NAME = "ml_anycsp"
ANYCSP_COLORING_BASELINE_NAME = "ml_anycsp_coloring"
ANYCSP_MAXSAT_BASELINE_NAME = "ml_anycsp_maxsat"
# One method, two registered names (the runner keys baselines by unique name, one problem each --
# cf. ml_pignn_mis / ml_pignn_coloring).
ANYCSP_BASELINE_NAMES = {"coloring": ANYCSP_COLORING_BASELINE_NAME, "maxsat": ANYCSP_MAXSAT_BASELINE_NAME}
SUPPORTED_PROBLEMS = ("coloring", "maxsat")
EPS = 1e-9

# Fixed feature widths so one model transfers across instances *and* problems.
ASSIGNMENT_FEATURE_DIM = 5
CONSTRAINT_FEATURE_DIM = 3
EDGE_FEATURE_DIM = 2


@dataclass(frozen=True)
class ANYCSPConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    message_passing_steps: int = 12
    recurrent_layers: int = 1
    samples: int = 16
    walksat_restarts: int = 4
    walksat_flips: int = 1000
    noise: float = 0.1
    repair_budget: int = 256
    inference_restarts: int = 8
    entropy_weight: float = 0.01
    entropy_decay_epochs: int = 20
    color_slack: int = 0
    max_colors: int = 0
    device: str = "cpu"
    seed: int = 0

    @classmethod
    def from_config(cls, config: MLBaselineConfig | dict[str, Any] | None) -> "ANYCSPConfig":
        base = cls()
        if config is None:
            return base
        payload = config.to_dict() if isinstance(config, MLBaselineConfig) else dict(config)
        return cls(
            epochs=int(payload.get("epochs", base.epochs)),
            learning_rate=float(payload.get("learning_rate", payload.get("lr", base.learning_rate))),
            hidden_dim=int(payload.get("hidden_dim", base.hidden_dim)),
            message_passing_steps=int(
                payload.get("message_passing_steps", payload.get("layers", base.message_passing_steps))
            ),
            recurrent_layers=int(payload.get("recurrent_layers", base.recurrent_layers)),
            samples=int(payload.get("samples", payload.get("inference_samples", base.samples))),
            walksat_restarts=int(payload.get("walksat_restarts", base.walksat_restarts)),
            walksat_flips=int(payload.get("walksat_flips", payload.get("repair_budget", base.walksat_flips))),
            noise=float(payload.get("noise", base.noise)),
            repair_budget=int(payload.get("repair_budget", base.repair_budget)),
            inference_restarts=int(payload.get("inference_restarts", base.inference_restarts)),
            entropy_weight=float(payload.get("entropy_weight", base.entropy_weight)),
            entropy_decay_epochs=int(payload.get("entropy_decay_epochs", base.entropy_decay_epochs)),
            color_slack=int(payload.get("color_slack", base.color_slack)),
            max_colors=int(payload.get("max_colors", base.max_colors)),
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
            "samples": self.samples,
            "walksat_restarts": self.walksat_restarts,
            "walksat_flips": self.walksat_flips,
            "noise": self.noise,
            "repair_budget": self.repair_budget,
            "inference_restarts": self.inference_restarts,
            "entropy_weight": self.entropy_weight,
            "entropy_decay_epochs": self.entropy_decay_epochs,
            "color_slack": self.color_slack,
            "max_colors": self.max_colors,
            "device": self.device,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class CSPValueTensor:
    """A constraint value graph: assignment nodes (variable, value) + constraint nodes.

    Assignment nodes are ordered variable-major: node (i, v) has index ``i * domain_size + v``,
    so per-variable value distributions are ``logits.reshape(num_variables, domain_size)``.
    """

    problem: str
    instance_id: str
    num_variables: int
    domain_size: int
    num_constraints: int
    assignment_features: Any      # [num_variables * domain_size, ASSIGNMENT_FEATURE_DIM]
    constraint_features: Any      # [num_constraints, CONSTRAINT_FEATURE_DIM]
    incidence_index: Any          # [2, E]  row 0 = assignment node, row 1 = constraint node
    incidence_feature: Any        # [E, EDGE_FEATURE_DIM]
    # MaxSAT soft-satisfaction structure (empty for coloring):
    lit_var: Any                  # [L] long   variable index of each literal
    lit_false_value: Any          # [L] long   the value (0/1) that makes the literal false
    lit_clause: Any               # [L] long   clause index of each literal
    clause_weights: Any           # [num_constraints] float
    # Coloring soft-conflict structure (empty for maxsat):
    edge_u: Any                   # [Eg] long  endpoint u of each graph edge
    edge_w: Any                   # [Eg] long  endpoint w of each graph edge


def _empty(dtype: str, device: str | None, backend: str) -> Any:
    return as_backend_tensor([], dtype=dtype, device=device, backend=backend)


def _tensorize_maxsat(instance: dict[str, Any], *, backend: str, device: str | None) -> CSPValueTensor:
    public = public_view(instance)
    num_variables = int(public["num_variables"])
    clauses = public["clauses"]
    num_clauses = len(clauses)
    weights = public.get("weights", public.get("clause_weights")) or [1.0] * num_clauses
    weights = [float(w) for w in weights]
    domain = 2

    occurrences = [0] * num_variables
    for clause in clauses:
        for literal in clause:
            occurrences[abs(int(literal)) - 1] += 1
    max_occ = max(occurrences, default=1) or 1
    max_len = max((len(clause) for clause in clauses), default=1) or 1

    sources: list[int] = []
    targets: list[int] = []
    edge_feature: list[list[float]] = []
    lit_var: list[int] = []
    lit_false_value: list[int] = []
    lit_clause: list[int] = []
    constraint_features: list[list[float]] = []
    for clause_index, clause in enumerate(clauses):
        for literal in clause:
            variable = abs(int(literal)) - 1
            positive = int(literal) > 0
            satisfying_value = 1 if positive else 0
            for value in (0, 1):
                sources.append(variable * domain + value)
                targets.append(clause_index)
                edge_feature.append([1.0 if value == satisfying_value else -1.0, 1.0 if positive else -1.0])
            lit_var.append(variable)
            lit_false_value.append(0 if positive else 1)  # literal false when var != satisfying value
            lit_clause.append(clause_index)
        constraint_features.append([float(len(clause)) / float(max_len), 0.0, 1.0])

    assignment_features: list[list[float]] = []
    for variable in range(num_variables):
        degree = float(occurrences[variable]) / float(max_occ)
        for value in range(domain):
            assignment_features.append([float(value), degree, 1.0, 0.0, 1.0])

    return CSPValueTensor(
        problem="maxsat",
        instance_id=str(public.get("id", "")),
        num_variables=num_variables,
        domain_size=domain,
        num_constraints=num_clauses,
        assignment_features=as_backend_tensor(assignment_features, dtype="float", device=device, backend=backend),
        constraint_features=as_backend_tensor(
            constraint_features or [[0.0, 0.0, 1.0]], dtype="float", device=device, backend=backend
        ),
        incidence_index=as_backend_tensor([sources, targets], dtype="long", device=device, backend=backend),
        incidence_feature=as_backend_tensor(edge_feature, dtype="float", device=device, backend=backend),
        lit_var=as_backend_tensor(lit_var, dtype="long", device=device, backend=backend),
        lit_false_value=as_backend_tensor(lit_false_value, dtype="long", device=device, backend=backend),
        lit_clause=as_backend_tensor(lit_clause, dtype="long", device=device, backend=backend),
        clause_weights=as_backend_tensor(weights or [1.0], dtype="float", device=device, backend=backend),
        edge_u=_empty("long", device, backend),
        edge_w=_empty("long", device, backend),
    )


def _tensorize_coloring(
    instance: dict[str, Any], color_count: int, *, backend: str, device: str | None
) -> CSPValueTensor:
    public = public_view(instance)
    num_vertices = int(public["num_vertices"])
    edges = [[int(u), int(v)] for u, v in public["edges"]]
    k = max(1, int(color_count))

    degree = [0] * num_vertices
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    max_degree = max(degree, default=1) or 1
    denom = float(k - 1) if k > 1 else 1.0

    sources: list[int] = []
    targets: list[int] = []
    edge_feature: list[list[float]] = []
    edge_u: list[int] = []
    edge_w: list[int] = []
    for edge_index, (u, v) in enumerate(edges):
        for color in range(k):
            sources.append(u * k + color)
            targets.append(edge_index)
            edge_feature.append([1.0, 0.0])
            sources.append(v * k + color)
            targets.append(edge_index)
            edge_feature.append([-1.0, 0.0])
        edge_u.append(u)
        edge_w.append(v)

    assignment_features: list[list[float]] = []
    for vertex in range(num_vertices):
        deg = float(degree[vertex]) / float(max_degree)
        for color in range(k):
            assignment_features.append([float(color) / denom, deg, 1.0, 1.0, 0.0])
    constraint_features = [[1.0, 1.0, 0.0] for _ in edges] or [[1.0, 1.0, 0.0]]

    return CSPValueTensor(
        problem="coloring",
        instance_id=str(public.get("id", "")),
        num_variables=num_vertices,
        domain_size=k,
        num_constraints=len(edges),
        assignment_features=as_backend_tensor(assignment_features, dtype="float", device=device, backend=backend),
        constraint_features=as_backend_tensor(constraint_features, dtype="float", device=device, backend=backend),
        incidence_index=as_backend_tensor([sources, targets], dtype="long", device=device, backend=backend),
        incidence_feature=as_backend_tensor(edge_feature, dtype="float", device=device, backend=backend),
        lit_var=_empty("long", device, backend),
        lit_false_value=_empty("long", device, backend),
        lit_clause=_empty("long", device, backend),
        clause_weights=_empty("float", device, backend),
        edge_u=as_backend_tensor(edge_u, dtype="long", device=device, backend=backend),
        edge_w=as_backend_tensor(edge_w, dtype="long", device=device, backend=backend),
    )


def tensorize_csp_instance(
    problem: str,
    instance: dict[str, Any],
    *,
    color_count: int | None = None,
    backend: str = "auto",
    device: str | None = None,
) -> CSPValueTensor:
    if problem == "maxsat":
        return _tensorize_maxsat(instance, backend=backend, device=device)
    if problem == "coloring":
        if color_count is None:
            raise ValueError("Coloring tensorization requires a fixed color_count.")
        return _tensorize_coloring(instance, color_count, backend=backend, device=device)
    raise ValueError(f"ANYCSP baseline supports {SUPPORTED_PROBLEMS}, got {problem!r}.")


class ANYCSPNet:
    """Recurrent message passing over the constraint value graph (assignment <-> constraint)."""

    def __new__(cls, *args: Any, **kwargs: Any):
        torch = require_torch()
        nn = torch.nn

        class _ANYCSPNet(nn.Module):
            def __init__(
                self,
                assignment_dim: int,
                constraint_dim: int,
                edge_dim: int,
                hidden_dim: int,
                message_passing_steps: int,
                recurrent_layers: int,
            ) -> None:
                super().__init__()
                self.assignment_input = nn.Linear(assignment_dim, hidden_dim)
                self.constraint_input = nn.Linear(constraint_dim, hidden_dim)
                self.assignment_to_constraint = nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim)
                self.constraint_to_assignment = nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim)
                self.constraint_grus = nn.ModuleList(
                    nn.GRUCell(hidden_dim, hidden_dim) for _ in range(max(1, recurrent_layers))
                )
                self.assignment_grus = nn.ModuleList(
                    nn.GRUCell(hidden_dim, hidden_dim) for _ in range(max(1, recurrent_layers))
                )
                self.output = nn.Linear(hidden_dim, 1)
                self.message_passing_steps = max(1, int(message_passing_steps))
                self.activation = nn.ReLU()

            def _mean_aggregate(self, messages: Any, index: Any, size: int) -> Any:
                output = torch.zeros((size, messages.shape[-1]), dtype=messages.dtype, device=messages.device)
                output.index_add_(0, index, messages)
                degree = torch.zeros((size, 1), dtype=messages.dtype, device=messages.device)
                degree.index_add_(0, index, torch.ones((messages.shape[0], 1), dtype=messages.dtype, device=messages.device))
                return output / degree.clamp_min(1.0)

            def forward(
                self,
                assignment_features: Any,
                constraint_features: Any,
                incidence_index: Any,
                incidence_feature: Any,
            ) -> Any:
                assignment_hidden = self.activation(self.assignment_input(assignment_features))
                constraint_hidden = self.activation(self.constraint_input(constraint_features))
                if incidence_index.numel() == 0:
                    return self.output(assignment_hidden).squeeze(-1)

                assignment_index = incidence_index[0].long()
                constraint_index = incidence_index[1].long()
                edge_feature = incidence_feature.to(assignment_hidden.dtype)
                num_assignments = int(assignment_features.shape[0])
                num_constraints = int(constraint_features.shape[0])

                for _ in range(self.message_passing_steps):
                    a_to_c = self.activation(
                        self.assignment_to_constraint(
                            torch.cat(
                                [assignment_hidden[assignment_index], constraint_hidden[constraint_index], edge_feature],
                                dim=-1,
                            )
                        )
                    )
                    constraint_aggregate = self._mean_aggregate(a_to_c, constraint_index, num_constraints)
                    for gru in self.constraint_grus:
                        constraint_hidden = gru(constraint_aggregate, constraint_hidden)

                    c_to_a = self.activation(
                        self.constraint_to_assignment(
                            torch.cat(
                                [constraint_hidden[constraint_index], assignment_hidden[assignment_index], edge_feature],
                                dim=-1,
                            )
                        )
                    )
                    assignment_aggregate = self._mean_aggregate(c_to_a, assignment_index, num_assignments)
                    for gru in self.assignment_grus:
                        assignment_hidden = gru(assignment_aggregate, assignment_hidden)

                return self.output(assignment_hidden).squeeze(-1)

        return _ANYCSPNet(*args, **kwargs)


def _assignment_probabilities(logits: Any, num_variables: int, domain_size: int) -> Any:
    torch = require_torch()
    return torch.softmax(logits.reshape(num_variables, domain_size), dim=-1)


def _maxsat_expected_fraction(probabilities: Any, tensor: CSPValueTensor) -> Any:
    """Differentiable expected weighted satisfied-clause fraction from per-variable P(value)."""
    torch = require_torch()
    if int(tensor.lit_clause.numel()) == 0:
        return torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)
    literal_false = probabilities[tensor.lit_var, tensor.lit_false_value].clamp(EPS, 1.0)
    clause_log_unsat = torch.zeros(
        int(tensor.num_constraints), dtype=probabilities.dtype, device=probabilities.device
    )
    clause_log_unsat.index_add_(0, tensor.lit_clause, torch.log(literal_false))
    clause_satisfied = 1.0 - torch.exp(clause_log_unsat)
    weighted = (clause_satisfied * tensor.clause_weights).sum()
    denominator = tensor.clause_weights.sum().clamp_min(1.0)
    return weighted / denominator


def _coloring_conflict(probabilities: Any, tensor: CSPValueTensor) -> Any:
    torch = require_torch()
    if int(tensor.edge_u.numel()) == 0:
        return torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)
    same_color_mass = (probabilities[tensor.edge_u] * probabilities[tensor.edge_w]).sum(dim=-1)
    return same_color_mass.mean()


def _loss(model: Any, tensor: CSPValueTensor, config: ANYCSPConfig, *, epoch: int) -> Any:
    torch = require_torch()
    logits = model(
        tensor.assignment_features, tensor.constraint_features, tensor.incidence_index, tensor.incidence_feature
    ).reshape(-1)
    probabilities = _assignment_probabilities(logits, tensor.num_variables, tensor.domain_size)
    entropy_scale = 0.0
    if config.entropy_decay_epochs > 0:
        entropy_scale = config.entropy_weight * max(0.0, 1.0 - (epoch - 1) / float(config.entropy_decay_epochs))
    clipped = probabilities.clamp(1e-8, 1.0)
    entropy = -(clipped * torch.log(clipped)).sum(dim=-1).mean()
    if tensor.problem == "maxsat":
        expected_fraction = _maxsat_expected_fraction(probabilities, tensor)
        return -expected_fraction - entropy_scale * entropy
    conflict = _coloring_conflict(probabilities, tensor)
    return conflict + entropy_scale * entropy


def _resolve_color_count(
    problem: str,
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]],
    config: ANYCSPConfig,
) -> int:
    if problem != "coloring":
        return 2
    if config.max_colors > 0:
        return max(1, int(config.max_colors))
    counts = [_dsatur_color_count(instance) for instance in [*train_instances, *val_instances]]
    return max(1, max(counts, default=1) + max(0, int(config.color_slack)))


def fit(
    problem: str,
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    config: ANYCSPConfig | MLBaselineConfig | dict[str, Any] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> TrainedState:
    if problem not in SUPPORTED_PROBLEMS:
        raise ValueError(f"ANYCSP baseline supports {SUPPORTED_PROBLEMS}, got {problem!r}.")
    if not torch_available():
        raise TorchUnavailableError("PyTorch is required for the ANYCSP baseline.")
    torch = require_torch()
    resolved_config = config if isinstance(config, ANYCSPConfig) else ANYCSPConfig.from_config(config)
    seed_everything(resolved_config.seed)
    device = resolve_device(resolved_config.device)
    if not train_instances:
        raise ValueError("The ANYCSP baseline requires public training instances.")

    public_train = [public_view(instance) for instance in train_instances]
    public_val = [public_view(instance) for instance in (val_instances or [])]
    color_count = _resolve_color_count(problem, public_train, public_val, resolved_config)

    def _tensor(instance: dict[str, Any]) -> CSPValueTensor:
        return tensorize_csp_instance(
            problem, instance, color_count=color_count, backend="torch", device=str(device)
        )

    train_tensors = [_tensor(instance) for instance in public_train]
    val_tensors = [_tensor(instance) for instance in public_val]

    model = ANYCSPNet(
        ASSIGNMENT_FEATURE_DIM,
        CONSTRAINT_FEATURE_DIM,
        EDGE_FEATURE_DIM,
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
            val_losses: list[float] = []
            with torch.no_grad():
                for tensor in val_tensors:
                    val_losses.append(float(_loss(model, tensor, resolved_config, epoch=epoch).detach().cpu().item()))
            val_loss = sum(val_losses) / max(1, len(val_losses))
        selection_loss = train_loss if val_loss is None else val_loss
        logger.log(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "problem": problem,
        "baseline_name": ANYCSP_BASELINE_NAMES.get(problem, ANYCSP_BASELINE_NAME),
        "config": resolved_config.to_dict(),
        "color_count": int(color_count),
        "train_instances": len(public_train),
        "validation_instances": len(public_val),
        "history": logger.rows,
    }
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            {"model_state_dict": model.state_dict(), "color_count": int(color_count)},
            metadata=metadata,
            use_torch=True,
        )
    return TrainedState(payload={"model": model, "color_count": int(color_count)}, metadata=metadata)


def _variable_value_logits(
    model: Any, problem: str, instance: dict[str, Any], config: ANYCSPConfig, color_count: int
) -> list[list[float]]:
    torch = require_torch()
    device = resolve_device(config.device)
    tensor = tensorize_csp_instance(problem, instance, color_count=color_count, backend="torch", device=str(device))
    model.eval()
    with torch.no_grad():
        logits = model(
            tensor.assignment_features, tensor.constraint_features, tensor.incidence_index, tensor.incidence_feature
        ).reshape(tensor.num_variables, tensor.domain_size)
    return [[float(value) for value in row] for row in logits.detach().cpu().tolist()]


def solve(
    problem: str,
    instance: dict[str, Any],
    trained_state: TrainedState,
    config: ANYCSPConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> Any:
    resolved_config = config if isinstance(config, ANYCSPConfig) else ANYCSPConfig.from_config(config)
    color_count = int(trained_state.payload.get("color_count", trained_state.metadata.get("color_count", 2)))
    model = trained_state.payload["model"]
    logits = _variable_value_logits(model, problem, instance, resolved_config, color_count)
    if problem == "maxsat":
        torch = require_torch()
        probabilities = torch.softmax(torch.tensor(logits), dim=-1)[:, 1]
        return decode_runcsp_assignment(
            instance,
            [float(value) for value in probabilities.tolist()],
            samples=resolved_config.samples,
            walksat_restarts=resolved_config.walksat_restarts,
            walksat_flips=resolved_config.walksat_flips,
            noise=resolved_config.noise,
            seed=resolved_config.seed,
        )
    return decode_pignn_coloring(
        instance,
        logits,
        max_colors=color_count,
        config={
            "repair_budget": resolved_config.repair_budget,
            "inference_restarts": resolved_config.inference_restarts,
            "seed": resolved_config.seed,
        },
    )


def build_anycsp_baselines(
    problem_name: str,
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    *,
    artifact_dir: str | Path | None = None,
    config: ANYCSPConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    if problem_name not in SUPPORTED_PROBLEMS or not torch_available():
        return {}
    root = Path(artifact_dir) if artifact_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    baseline_name = ANYCSP_BASELINE_NAMES[problem_name]
    checkpoint_path = root / f"{baseline_name}.pt" if root is not None else None
    metrics_path = root / f"{baseline_name}_metrics.jsonl" if root is not None else None
    resolved_config = config if isinstance(config, ANYCSPConfig) else ANYCSPConfig.from_config(config)
    trained_state = fit(
        problem_name,
        train_instances,
        val_instances or [],
        resolved_config,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )

    def solver(instance: dict[str, Any]) -> Any:
        return solve(problem_name, instance, trained_state, resolved_config)

    return {baseline_name: solver}
