"""GFlowNet learned baseline ("Let the Flows Tell", Zhang et al., NeurIPS 2023).

A faithful compact reproduction in the DasBench baseline style for Maximum Independent Set and
Minimum Dominating Set.  A Trajectory-Balance GFlowNet amortises a constructive policy that adds
one vertex at a time and is trained so terminal solutions are sampled in proportion to their
reward (larger independent sets / smaller dominating sets).  Distinctive GFlowNet elements kept:
the sequential forward policy, a learnable partition-function scalar ``log_z``, the
trajectory-balance objective with a uniform backward policy, and reward-proportional sampling at
inference (best of several sampled trajectories).

For tractability on the large MDS graphs the graph is encoded once by a GraphSAGE-style GNN and
the per-step policy is a state-conditioned head over that embedding (an amortised variant rather
than re-encoding every step).  Terminal solutions are projected to feasibility with the same
bounded repair used by the other MIS/MDS baselines.  Private ``optimum_*`` / ``_`` fields are
stripped via ``public_view`` before tensorisation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_baselines.checkpoint import save_checkpoint
from ml_baselines.config import MLBaselineConfig
from ml_baselines.graph_score_repair import decode_mds_scores, decode_mis_scores
from ml_baselines.interface import TrainedState
from ml_baselines.metrics import MetricsLogger
from ml_baselines.models import ManualMessagePassing, scatter_add_messages
from ml_baselines.seeding import seed_everything
from ml_baselines.tensorize import public_view
from ml_baselines.torch_utils import (
    TorchUnavailableError,
    as_backend_tensor,
    require_torch,
    resolve_device,
    torch_available,
)

GFLOWNET_MIS_BASELINE_NAME = "ml_gflownet_mis"
GFLOWNET_MDS_BASELINE_NAME = "ml_gflownet_mds"
GFLOWNET_BASELINE_NAMES = {"mis": GFLOWNET_MIS_BASELINE_NAME, "mds": GFLOWNET_MDS_BASELINE_NAME}
SUPPORTED_PROBLEMS = ("mis", "mds")

NODE_FEATURE_DIM = 2   # [degree / max_degree, 1.0]
STATE_FEATURE_DIM = 3  # [in_solution, blocked-or-covered, active-neighbour-fraction]
NEG_INF = -1e9


@dataclass(frozen=True)
class GFlowNetConfig:
    epochs: int = 30
    learning_rate: float = 1e-3
    logz_learning_rate: float = 1e-2
    hidden_dim: int = 64
    layers: int = 3
    reward_scale: float = 8.0
    trajectory_samples: int = 2
    inference_samples: int = 12
    max_steps_factor: float = 1.1
    train_instance_cap: int = 32
    repair_budget: int = 128
    device: str = "cpu"
    seed: int = 0

    @classmethod
    def from_config(cls, config: MLBaselineConfig | dict[str, Any] | None) -> "GFlowNetConfig":
        base = cls()
        if config is None:
            return base
        payload = config.to_dict() if isinstance(config, MLBaselineConfig) else dict(config)
        return cls(
            epochs=int(payload.get("epochs", base.epochs)),
            learning_rate=float(payload.get("learning_rate", payload.get("lr", base.learning_rate))),
            logz_learning_rate=float(payload.get("logz_learning_rate", base.logz_learning_rate)),
            hidden_dim=int(payload.get("hidden_dim", base.hidden_dim)),
            layers=int(payload.get("layers", base.layers)),
            reward_scale=float(payload.get("reward_scale", base.reward_scale)),
            trajectory_samples=int(payload.get("trajectory_samples", base.trajectory_samples)),
            inference_samples=int(payload.get("inference_samples", payload.get("inference_restarts", base.inference_samples))),
            max_steps_factor=float(payload.get("max_steps_factor", base.max_steps_factor)),
            train_instance_cap=int(payload.get("train_instance_cap", base.train_instance_cap)),
            repair_budget=int(payload.get("repair_budget", base.repair_budget)),
            device=str(payload.get("device", base.device)),
            seed=int(payload.get("seed", base.seed)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "logz_learning_rate": self.logz_learning_rate,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "reward_scale": self.reward_scale,
            "trajectory_samples": self.trajectory_samples,
            "inference_samples": self.inference_samples,
            "max_steps_factor": self.max_steps_factor,
            "train_instance_cap": self.train_instance_cap,
            "repair_budget": self.repair_budget,
            "device": self.device,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class GraphConstruction:
    instance_id: str
    num_vertices: int
    node_features: Any            # [nv, NODE_FEATURE_DIM]
    edge_index: Any               # [2, 2E] symmetric
    adjacency: list[list[int]]    # python adjacency for feasibility bookkeeping


def _build_construction(instance: dict[str, Any], *, device: str | None) -> GraphConstruction:
    public = public_view(instance)
    num_vertices = int(public["num_vertices"])
    edges = [(int(u), int(v)) for u, v in public["edges"]]
    adjacency: list[list[int]] = [[] for _ in range(num_vertices)]
    sources: list[int] = []
    targets: list[int] = []
    for u, v in edges:
        if u == v:
            continue
        adjacency[u].append(v)
        adjacency[v].append(u)
        sources.extend((u, v))
        targets.extend((v, u))
    degrees = [len(neighbours) for neighbours in adjacency]
    max_degree = max(degrees, default=1) or 1
    node_features = [[float(degrees[v]) / float(max_degree), 1.0] for v in range(num_vertices)]
    return GraphConstruction(
        instance_id=str(public.get("id", "")),
        num_vertices=num_vertices,
        node_features=as_backend_tensor(node_features, dtype="float", device=device, backend="torch"),
        edge_index=as_backend_tensor([sources, targets], dtype="long", device=device, backend="torch"),
        adjacency=adjacency,
    )


class GFlowNetPolicy:
    """GraphSAGE encoder (run once) + a state-conditioned per-vertex scoring head + log_z."""

    def __new__(cls, *args: Any, **kwargs: Any):
        torch = require_torch()
        nn = torch.nn

        class _GFlowNetPolicy(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, layers: int, state_dim: int) -> None:
                super().__init__()
                self.encoder = ManualMessagePassing(input_dim, hidden_dim, hidden_dim, layers=layers)
                self.score = nn.Sequential(
                    nn.Linear(hidden_dim + state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                self.log_z = nn.Parameter(torch.zeros(1))

            def encode(self, node_features: Any, edge_index: Any) -> Any:
                return self.encoder(node_features, edge_index)

            def vertex_logits(self, embedding: Any, state_features: Any) -> Any:
                return self.score(torch.cat([embedding, state_features], dim=-1)).reshape(-1)

        return _GFlowNetPolicy(*args, **kwargs)


def _state_features(
    problem: str,
    construction: GraphConstruction,
    selected_mask: Any,
    inactive_mask: Any,
) -> Any:
    """Per-vertex [in_solution, blocked/covered, active-neighbour fraction].

    ``inactive_mask`` is "blocked" for MIS (selected or adjacent to a selected vertex) and
    "covered" for MDS (selected or dominated).  The active-neighbour fraction counts neighbours
    that are still *available* (MIS) / *uncovered* (MDS) -- the vertices this one could still help.
    """
    torch = require_torch()
    active = 1.0 - inactive_mask
    active_neighbours = scatter_add_messages(active.reshape(-1, 1), construction.edge_index, num_nodes=construction.num_vertices).reshape(-1)
    degree = scatter_add_messages(
        torch.ones((construction.edge_index.shape[-1], 1), dtype=selected_mask.dtype, device=selected_mask.device),
        construction.edge_index,
        num_nodes=construction.num_vertices,
    ).reshape(-1)
    fraction = active_neighbours / degree.clamp_min(1.0)
    return torch.stack([selected_mask, inactive_mask, fraction], dim=-1)


def _rollout(
    problem: str,
    policy: Any,
    construction: GraphConstruction,
    embedding: Any,
    config: GFlowNetConfig,
    *,
    greedy: bool = False,
) -> tuple[list[int], Any, bool]:
    """One constructive trajectory. Returns (selected vertices, sum log P_F, feasible)."""
    torch = require_torch()
    nv = construction.num_vertices
    device = embedding.device
    selected_mask = torch.zeros(nv, dtype=embedding.dtype, device=device)
    inactive_mask = torch.zeros(nv, dtype=embedding.dtype, device=device)  # blocked (MIS) / covered (MDS)
    selected: list[int] = []
    log_forward = torch.zeros((), dtype=embedding.dtype, device=device)
    max_steps = max(1, int(config.max_steps_factor * nv))
    adjacency = construction.adjacency

    for _ in range(max_steps):
        if problem == "mis":
            available = (selected_mask + inactive_mask) < 0.5  # not selected and not blocked
        else:  # mds: any unselected vertex is a legal action; stop once everything is covered
            if bool((inactive_mask > 0.5).all()):
                break
            available = selected_mask < 0.5
        if not bool(available.any()):
            break
        state_features = _state_features(problem, construction, selected_mask, inactive_mask)
        logits = policy.vertex_logits(embedding, state_features)
        logits = torch.where(available, logits, torch.full_like(logits, NEG_INF))
        log_probs = torch.log_softmax(logits, dim=-1)
        if greedy:
            choice = int(torch.argmax(log_probs).item())
        else:
            choice = int(torch.multinomial(torch.exp(log_probs), 1).item())
        log_forward = log_forward + log_probs[choice]
        selected.append(choice)
        selected_mask[choice] = 1.0
        if problem == "mis":
            inactive_mask[choice] = 1.0
            for neighbour in adjacency[choice]:
                inactive_mask[neighbour] = 1.0
        else:
            inactive_mask[choice] = 1.0
            for neighbour in adjacency[choice]:
                inactive_mask[neighbour] = 1.0

    feasible = True
    if problem == "mds":
        feasible = bool((inactive_mask > 0.5).all())
    return selected, log_forward, feasible


def _log_reward(problem: str, selected_size: int, num_vertices: int, config: GFlowNetConfig, *, feasible: bool) -> float:
    if num_vertices <= 0:
        return 0.0
    fraction = float(selected_size) / float(num_vertices)
    if problem == "mis":
        return config.reward_scale * fraction
    if not feasible:
        return -config.reward_scale  # heavy penalty for an incomplete dominating set
    return config.reward_scale * (1.0 - fraction)


def _trajectory_balance_loss(policy: Any, log_forward: Any, log_reward: float, selected_size: int) -> Any:
    torch = require_torch()
    log_backward = -math.lgamma(selected_size + 1)  # uniform P_B over the set: sum_k log(1/k) = -log(n!)
    residual = policy.log_z.reshape(()) + log_forward - float(log_reward) - float(log_backward)
    return residual * residual


def fit(
    problem: str,
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    config: GFlowNetConfig | MLBaselineConfig | dict[str, Any] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> TrainedState:
    if problem not in SUPPORTED_PROBLEMS:
        raise ValueError(f"GFlowNet baseline supports {SUPPORTED_PROBLEMS}, got {problem!r}.")
    if not torch_available():
        raise TorchUnavailableError("PyTorch is required for the GFlowNet baseline.")
    torch = require_torch()
    resolved_config = config if isinstance(config, GFlowNetConfig) else GFlowNetConfig.from_config(config)
    seed_everything(resolved_config.seed)
    device = resolve_device(resolved_config.device)
    if not train_instances:
        raise ValueError("The GFlowNet baseline requires public training instances.")

    train_public = [public_view(instance) for instance in train_instances][: max(1, resolved_config.train_instance_cap)]
    constructions = [_build_construction(instance, device=str(device)) for instance in train_public]
    constructions = [c for c in constructions if c.num_vertices > 0]

    policy = GFlowNetPolicy(NODE_FEATURE_DIM, resolved_config.hidden_dim, resolved_config.layers, STATE_FEATURE_DIM).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": [p for name, p in policy.named_parameters() if name != "log_z"]},
            {"params": [policy.log_z], "lr": resolved_config.logz_learning_rate},
        ],
        lr=resolved_config.learning_rate,
    )
    logger = MetricsLogger(Path(metrics_path) if metrics_path is not None else None)
    best_state: dict[str, Any] | None = None
    best_score = -math.inf

    for epoch in range(1, resolved_config.epochs + 1):
        policy.train()
        epoch_losses: list[float] = []
        epoch_sizes: list[float] = []
        for construction in constructions:
            embedding = policy.encode(construction.node_features, construction.edge_index)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), dtype=embedding.dtype, device=device)
            sizes: list[int] = []
            for _ in range(max(1, resolved_config.trajectory_samples)):
                selected, log_forward, feasible = _rollout(problem, policy, construction, embedding, resolved_config)
                log_reward = _log_reward(problem, len(selected), construction.num_vertices, resolved_config, feasible=feasible)
                loss = loss + _trajectory_balance_loss(policy, log_forward, log_reward, len(selected))
                sizes.append(len(selected))
            loss = loss / float(max(1, resolved_config.trajectory_samples))
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
            epoch_sizes.append(sum(sizes) / max(1, len(sizes)))
        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        mean_size = sum(epoch_sizes) / max(1, len(epoch_sizes))
        # Selection signal: larger IS is better; smaller DS is better.
        selection_score = mean_size if problem == "mis" else -mean_size
        logger.log(epoch=epoch, train_loss=train_loss, mean_solution_size=mean_size, log_z=float(policy.log_z.detach().cpu().item()))
        if selection_score > best_score:
            best_score = selection_score
            best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}

    if best_state is not None:
        policy.load_state_dict(best_state)
    metadata = {
        "problem": problem,
        "baseline_name": GFLOWNET_BASELINE_NAMES[problem],
        "config": resolved_config.to_dict(),
        "train_instances": len(constructions),
        "history": logger.rows,
        "log_z": float(policy.log_z.detach().cpu().item()),
    }
    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, {"model_state_dict": policy.state_dict()}, metadata=metadata, use_torch=True)
    return TrainedState(payload={"policy": policy}, metadata=metadata)


def _decode(problem: str, instance: dict[str, Any], selected: list[int], repair_budget: int) -> list[int]:
    num_vertices = int(public_view(instance)["num_vertices"])
    scores = [0.0] * num_vertices
    for vertex in selected:
        if 0 <= vertex < num_vertices:
            scores[vertex] = 1.0
    if problem == "mis":
        return decode_mis_scores(instance, scores, repair_budget=repair_budget)
    return decode_mds_scores(instance, scores, repair_budget=repair_budget)


def solve(
    problem: str,
    instance: dict[str, Any],
    trained_state: TrainedState,
    config: GFlowNetConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> list[int]:
    resolved_config = config if isinstance(config, GFlowNetConfig) else GFlowNetConfig.from_config(config)
    torch = require_torch()
    device = resolve_device(resolved_config.device)
    policy = trained_state.payload["policy"]
    policy.eval()
    construction = _build_construction(instance, device=str(device))
    if construction.num_vertices == 0:
        return []

    def _quality(candidate: list[int]) -> int:
        return len(candidate) if problem == "mis" else -len(candidate)

    best: list[int] | None = None
    with torch.no_grad():
        embedding = policy.encode(construction.node_features, construction.edge_index)
        rollouts = [(True, )] + [(False, )] * max(1, resolved_config.inference_samples)
        for (greedy, ) in rollouts:
            selected, _log_forward, _feasible = _rollout(problem, policy, construction, embedding, resolved_config, greedy=greedy)
            candidate = _decode(problem, instance, selected, resolved_config.repair_budget)
            if candidate is None:
                continue
            if best is None or _quality(candidate) > _quality(best):
                best = candidate
    if best is None:
        best = _decode(problem, instance, [], resolved_config.repair_budget)
    return best


def build_gflownet_baselines(
    problem_name: str,
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    *,
    artifact_dir: str | Path | None = None,
    config: GFlowNetConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    if problem_name not in SUPPORTED_PROBLEMS or not torch_available():
        return {}
    root = Path(artifact_dir) if artifact_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    baseline_name = GFLOWNET_BASELINE_NAMES[problem_name]
    checkpoint_path = root / f"{baseline_name}.pt" if root is not None else None
    metrics_path = root / f"{baseline_name}_metrics.jsonl" if root is not None else None
    resolved_config = config if isinstance(config, GFlowNetConfig) else GFlowNetConfig.from_config(config)
    trained_state = fit(
        problem_name,
        train_instances,
        val_instances or [],
        resolved_config,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )

    def solver(instance: dict[str, Any]) -> list[int]:
        return solve(problem_name, instance, trained_state, resolved_config)

    return {baseline_name: solver}
