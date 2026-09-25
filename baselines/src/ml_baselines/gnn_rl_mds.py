from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_baselines.checkpoint import save_checkpoint
from ml_baselines.config import MLBaselineConfig
from ml_baselines.interface import TrainedState
from ml_baselines.metrics import MetricsLogger
from ml_baselines.models import ManualMessagePassing
from ml_baselines.seeding import seed_everything
from ml_baselines.tensorize import public_view, tensorize_graph_instance
from ml_baselines.torch_utils import TorchUnavailableError, require_torch, resolve_device, torch_available

GNN_RL_MDS_BASELINE_NAME = "ml_gnn_rl_mds"
EPS = 1e-9
STATE_FEATURE_DIM = 7


@dataclass(frozen=True)
class GNNRLMDSConfig:
    episodes: int = 5000
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    layers: int = 3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    target_update_interval: int = 200
    replay_size: int = 50000
    batch_size: int = 64
    max_steps_factor: float = 1.5
    repair_budget: int = 128
    device: str = "cpu"
    seed: int = 0

    @classmethod
    def from_config(cls, config: MLBaselineConfig | dict[str, Any] | None) -> "GNNRLMDSConfig":
        base = cls()
        if config is None:
            return base
        payload = config.to_dict() if isinstance(config, MLBaselineConfig) else dict(config)
        return cls(
            episodes=int(payload.get("episodes", payload.get("epochs", base.episodes))),
            learning_rate=float(payload.get("learning_rate", payload.get("lr", base.learning_rate))),
            hidden_dim=int(payload.get("hidden_dim", base.hidden_dim)),
            layers=int(payload.get("message_passing_layers", payload.get("layers", base.layers))),
            gamma=float(payload.get("gamma", base.gamma)),
            epsilon_start=float(payload.get("epsilon_start", base.epsilon_start)),
            epsilon_end=float(payload.get("epsilon_end", base.epsilon_end)),
            epsilon_decay_steps=int(payload.get("epsilon_decay_steps", base.epsilon_decay_steps)),
            target_update_interval=int(payload.get("target_update_interval", base.target_update_interval)),
            replay_size=int(payload.get("replay_size", base.replay_size)),
            batch_size=int(payload.get("batch_size", base.batch_size)),
            max_steps_factor=float(payload.get("max_steps_factor", base.max_steps_factor)),
            repair_budget=int(payload.get("repair_budget", base.repair_budget)),
            device=str(payload.get("device", base.device)),
            seed=int(payload.get("seed", base.seed)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "learning_rate": self.learning_rate,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "target_update_interval": self.target_update_interval,
            "replay_size": self.replay_size,
            "batch_size": self.batch_size,
            "max_steps_factor": self.max_steps_factor,
            "repair_budget": self.repair_budget,
            "device": self.device,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class MDSExperience:
    instance_index: int
    selected: tuple[int, ...]
    dominated: tuple[int, ...]
    step: int
    action: int
    reward: float
    next_selected: tuple[int, ...]
    next_dominated: tuple[int, ...]
    next_step: int
    done: bool


def _adjacency(num_vertices: int, edges: list[list[int]] | list[tuple[int, int]]) -> list[set[int]]:
    adjacency = [set() for _ in range(num_vertices)]
    for raw_u, raw_v in edges:
        u = int(raw_u)
        v = int(raw_v)
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def _closed_neighborhoods(adjacency: list[set[int]]) -> list[set[int]]:
    return [neighbors | {vertex} for vertex, neighbors in enumerate(adjacency)]


def _dominated_from_selected(closed: list[set[int]], selected: set[int]) -> set[int]:
    dominated: set[int] = set()
    for vertex in selected:
        dominated.update(closed[vertex])
    return dominated


def _is_dominating(num_vertices: int, closed: list[set[int]], selected: set[int]) -> bool:
    return len(_dominated_from_selected(closed, selected)) == num_vertices


class DominatingSetEnvironment:
    def __init__(self, instance: dict[str, Any], *, max_steps: int | None = None) -> None:
        self.instance = public_view(instance)
        self.num_vertices = int(self.instance["num_vertices"])
        self.adjacency = _adjacency(self.num_vertices, self.instance["edges"])
        self.closed = _closed_neighborhoods(self.adjacency)
        self.degrees = [len(neighbors) for neighbors in self.adjacency]
        self.max_degree = max(self.degrees) if self.degrees else 1
        self.max_steps = int(max_steps or self.num_vertices)
        self.selected: set[int] = set()
        self.dominated: set[int] = set()
        self.steps = 0

    def snapshot(self) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        return tuple(sorted(self.selected)), tuple(sorted(self.dominated)), int(self.steps)

    def feasible_actions(self) -> list[bool]:
        return [vertex not in self.selected for vertex in range(self.num_vertices)]

    def coverage_gain(self, vertex: int) -> int:
        return len(self.closed[vertex] - self.dominated)

    def is_done(self) -> bool:
        return len(self.dominated) == self.num_vertices or self.steps >= self.max_steps

    def step(self, action: int) -> tuple[float, bool]:
        if not 0 <= action < self.num_vertices or action in self.selected:
            self.steps += 1
            return -1.0, True
        previous_count = len(self.dominated)
        self.selected.add(action)
        self.dominated.update(self.closed[action])
        self.steps += 1
        newly_dominated = len(self.dominated) - previous_count
        reward = float(newly_dominated) / max(float(self.num_vertices), 1.0)
        reward -= 1.0 / max(float(self.num_vertices), 1.0)
        done = self.is_done()
        if len(self.dominated) == self.num_vertices:
            reward += 1.0
        elif done:
            reward -= 1.0
        return reward, done


def _state_features_from_sets(
    instance: dict[str, Any],
    *,
    selected: set[int],
    dominated: set[int],
    step: int,
    max_steps: int,
    device: Any,
) -> Any:
    public = public_view(instance)
    num_vertices = int(public["num_vertices"])
    adjacency = _adjacency(num_vertices, public["edges"])
    closed = _closed_neighborhoods(adjacency)
    degrees = [len(neighbors) for neighbors in adjacency]
    max_degree = max(degrees) if degrees else 1
    density = 0.0 if num_vertices <= 1 else (2.0 * len(public["edges"])) / (num_vertices * (num_vertices - 1))
    rows = []
    for vertex in range(num_vertices):
        gain = len(closed[vertex] - dominated)
        rows.append(
            [
                float(degrees[vertex]),
                float(degrees[vertex]) / max(float(max_degree), 1.0),
                1.0 if vertex in selected else 0.0,
                1.0 if vertex in dominated else 0.0,
                float(gain) / max(float(num_vertices), 1.0),
                float(step) / max(float(max_steps), 1.0),
                density,
            ]
        )
    torch = require_torch()
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _available_mask(num_vertices: int, selected: set[int], *, device: Any) -> Any:
    torch = require_torch()
    return torch.tensor([vertex not in selected for vertex in range(num_vertices)], dtype=torch.bool, device=device)


def _epsilon(config: GNNRLMDSConfig, step: int) -> float:
    if config.epsilon_decay_steps <= 0:
        return float(config.epsilon_end)
    fraction = min(1.0, max(0.0, float(step) / float(config.epsilon_decay_steps)))
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def _greedy_action_from_model(
    model: Any,
    instance: dict[str, Any],
    edge_index: Any,
    *,
    selected: set[int],
    dominated: set[int],
    step: int,
    max_steps: int,
    device: Any,
) -> tuple[int, list[float]]:
    torch = require_torch()
    features = _state_features_from_sets(
        instance,
        selected=selected,
        dominated=dominated,
        step=step,
        max_steps=max_steps,
        device=device,
    )
    with torch.no_grad():
        q_values = model(features, edge_index).reshape(-1)
        scores = [float(value) for value in q_values.detach().cpu().tolist()]
        mask = _available_mask(int(public_view(instance)["num_vertices"]), selected, device=device)
        q_values = q_values.masked_fill(~mask, -1.0e9)
        action = int(torch.argmax(q_values).detach().cpu().item())
    return action, scores


def _experience_loss(
    model: Any,
    target_model: Any,
    experience: MDSExperience,
    public_train: list[dict[str, Any]],
    edge_indices: list[Any],
    config: GNNRLMDSConfig,
    *,
    device: Any,
) -> Any:
    torch = require_torch()
    instance = public_train[experience.instance_index]
    num_vertices = int(instance["num_vertices"])
    max_steps = max(1, int(math.ceil(config.max_steps_factor * num_vertices)))
    selected = set(experience.selected)
    dominated = set(experience.dominated)
    features = _state_features_from_sets(
        instance,
        selected=selected,
        dominated=dominated,
        step=experience.step,
        max_steps=max_steps,
        device=device,
    )
    q_value = model(features, edge_indices[experience.instance_index]).reshape(-1)[experience.action]
    with torch.no_grad():
        target = torch.tensor(float(experience.reward), dtype=q_value.dtype, device=device)
        if not experience.done:
            next_selected = set(experience.next_selected)
            next_dominated = set(experience.next_dominated)
            next_features = _state_features_from_sets(
                instance,
                selected=next_selected,
                dominated=next_dominated,
                step=experience.next_step,
                max_steps=max_steps,
                device=device,
            )
            next_q = target_model(next_features, edge_indices[experience.instance_index]).reshape(-1)
            mask = _available_mask(max(num_vertices, 1), next_selected, device=device)
            if bool(mask.any()):
                next_q = next_q.masked_fill(~mask, -1.0e9)
                target = target + config.gamma * next_q.max()
    return (q_value - target) ** 2


def repair_dominating_set(
    instance: dict[str, Any],
    selected_vertices: list[int],
    *,
    repair_budget: int = 128,
    q_scores: list[float] | None = None,
) -> list[int]:
    public = public_view(instance)
    num_vertices = int(public["num_vertices"])
    adjacency = _adjacency(num_vertices, public["edges"])
    closed = _closed_neighborhoods(adjacency)
    selected = {int(vertex) for vertex in selected_vertices if 0 <= int(vertex) < num_vertices}
    dominated = _dominated_from_selected(closed, selected)
    scores = q_scores if q_scores is not None and len(q_scores) >= num_vertices else [0.0] * num_vertices
    max_additions = max(num_vertices, int(repair_budget))
    additions = 0
    while len(dominated) < num_vertices and len(selected) < num_vertices and additions < max_additions:
        vertex = max(
            (item for item in range(num_vertices) if item not in selected),
            key=lambda item: (len(closed[item] - dominated), float(scores[item]), len(adjacency[item]), -item),
        )
        selected.add(vertex)
        dominated.update(closed[vertex])
        additions += 1

    ordered_for_prune = sorted(selected, key=lambda item: (float(scores[item]), len(adjacency[item]), item))
    for vertex in ordered_for_prune:
        if vertex not in selected:
            continue
        candidate = selected - {vertex}
        if _is_dominating(num_vertices, closed, candidate):
            selected = candidate
    return sorted(selected)


def decode_gnn_rl_mds(
    instance: dict[str, Any],
    model: Any,
    config: GNNRLMDSConfig,
) -> list[int]:
    torch = require_torch()
    device = resolve_device(config.device)
    public = public_view(instance)
    num_vertices = int(public["num_vertices"])
    edge_index = tensorize_graph_instance(public, backend="torch", device=str(device)).edge_index
    max_steps = max(1, int(math.ceil(config.max_steps_factor * num_vertices)))
    env = DominatingSetEnvironment(public, max_steps=max_steps)
    last_scores = [0.0] * num_vertices
    model.eval()
    while not env.is_done() and env.steps < max_steps:
        action, last_scores = _greedy_action_from_model(
            model,
            public,
            edge_index,
            selected=set(env.selected),
            dominated=set(env.dominated),
            step=env.steps,
            max_steps=max_steps,
            device=device,
        )
        _reward, done = env.step(action)
        if done:
            break
    return repair_dominating_set(public, sorted(env.selected), repair_budget=config.repair_budget, q_scores=last_scores)


def fit(
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    config: GNNRLMDSConfig | MLBaselineConfig | dict[str, Any] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> TrainedState:
    if not torch_available():
        raise TorchUnavailableError("PyTorch is required for the GNN-RL MDS baseline.")
    torch = require_torch()
    resolved_config = config if isinstance(config, GNNRLMDSConfig) else GNNRLMDSConfig.from_config(config)
    seed_everything(resolved_config.seed)
    device = resolve_device(resolved_config.device)
    if not train_instances:
        raise ValueError("The GNN-RL MDS baseline requires public training instances.")

    public_train = [public_view(instance) for instance in train_instances]
    public_val = [public_view(instance) for instance in (val_instances or [])]
    train_tensors = [tensorize_graph_instance(instance, backend="torch", device=str(device)) for instance in public_train]
    edge_indices = [tensor.edge_index for tensor in train_tensors]
    model = ManualMessagePassing(STATE_FEATURE_DIM, resolved_config.hidden_dim, 1, layers=resolved_config.layers).to(device)
    target_model = ManualMessagePassing(STATE_FEATURE_DIM, resolved_config.hidden_dim, 1, layers=resolved_config.layers).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=resolved_config.learning_rate)
    replay: deque[MDSExperience] = deque(maxlen=max(1, int(resolved_config.replay_size)))
    logger = MetricsLogger(Path(metrics_path) if metrics_path is not None else None)
    rng = random.Random(f"gnn-rl-mds:{resolved_config.seed}")
    best_state: dict[str, Any] | None = None
    best_reward = -math.inf
    global_step = 0

    for episode in range(1, resolved_config.episodes + 1):
        instance_index = rng.randrange(len(public_train))
        instance = public_train[instance_index]
        num_vertices = int(instance["num_vertices"])
        max_steps = max(1, int(math.ceil(resolved_config.max_steps_factor * num_vertices)))
        env = DominatingSetEnvironment(instance, max_steps=max_steps)
        episode_reward = 0.0
        losses: list[float] = []
        while not env.is_done():
            selected_snapshot, dominated_snapshot, step_snapshot = env.snapshot()
            epsilon = _epsilon(resolved_config, global_step)
            feasible = [vertex for vertex, allowed in enumerate(env.feasible_actions()) if allowed]
            if not feasible:
                break
            if rng.random() < epsilon:
                action = rng.choice(feasible)
            else:
                action, _scores = _greedy_action_from_model(
                    model,
                    instance,
                    edge_indices[instance_index],
                    selected=set(env.selected),
                    dominated=set(env.dominated),
                    step=env.steps,
                    max_steps=max_steps,
                    device=device,
                )
            reward, done = env.step(action)
            episode_reward += reward
            next_selected, next_dominated, next_step = env.snapshot()
            replay.append(
                MDSExperience(
                    instance_index=instance_index,
                    selected=selected_snapshot,
                    dominated=dominated_snapshot,
                    step=step_snapshot,
                    action=action,
                    reward=reward,
                    next_selected=next_selected,
                    next_dominated=next_dominated,
                    next_step=next_step,
                    done=done,
                )
            )
            if len(replay) >= max(1, min(resolved_config.batch_size, len(replay))):
                batch = rng.sample(list(replay), k=min(resolved_config.batch_size, len(replay)))
                optimizer.zero_grad(set_to_none=True)
                batch_losses = [
                    _experience_loss(model, target_model, experience, public_train, edge_indices, resolved_config, device=device)
                    for experience in batch
                ]
                loss = torch.stack(batch_losses).mean()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
            global_step += 1
            if resolved_config.target_update_interval > 0 and global_step % resolved_config.target_update_interval == 0:
                target_model.load_state_dict(model.state_dict())
            if done:
                break
        val_size = None
        if public_val and (episode == resolved_config.episodes or episode % max(1, resolved_config.episodes // 10) == 0):
            sizes = []
            for val_instance in public_val:
                solution = decode_gnn_rl_mds(val_instance, model, resolved_config)
                sizes.append(len(solution))
            val_size = sum(sizes) / max(1, len(sizes))
        logger.log(
            episode=episode,
            train_reward=episode_reward,
            train_loss=(sum(losses) / max(1, len(losses))) if losses else None,
            epsilon=_epsilon(resolved_config, global_step),
            val_solution_size=val_size,
        )
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "problem": "mds",
        "baseline_name": GNN_RL_MDS_BASELINE_NAME,
        "config": resolved_config.to_dict(),
        "train_instances": len(public_train),
        "validation_instances": len(public_val),
        "input_dim": STATE_FEATURE_DIM,
        "history": logger.rows,
    }
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            {"model_state_dict": model.state_dict(), "input_dim": STATE_FEATURE_DIM},
            metadata=metadata,
            use_torch=True,
        )
    return TrainedState(payload={"model": model, "input_dim": STATE_FEATURE_DIM}, metadata=metadata)


def solve(
    instance: dict[str, Any],
    trained_state: TrainedState,
    config: GNNRLMDSConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> list[int]:
    resolved_config = config if isinstance(config, GNNRLMDSConfig) else GNNRLMDSConfig.from_config(config)
    return decode_gnn_rl_mds(instance, trained_state.payload["model"], resolved_config)


def build_gnn_rl_mds_baselines(
    problem_name: str,
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    *,
    artifact_dir: str | Path | None = None,
    config: GNNRLMDSConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    if problem_name != "mds" or not torch_available():
        return {}
    root = Path(artifact_dir) if artifact_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / f"{GNN_RL_MDS_BASELINE_NAME}.pt" if root is not None else None
    metrics_path = root / f"{GNN_RL_MDS_BASELINE_NAME}_metrics.jsonl" if root is not None else None
    resolved_config = config if isinstance(config, GNNRLMDSConfig) else GNNRLMDSConfig.from_config(config)
    trained_state = fit(
        train_instances,
        val_instances or [],
        resolved_config,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )

    def solver(instance: dict[str, Any]) -> list[int]:
        return solve(instance, trained_state, resolved_config)

    return {GNN_RL_MDS_BASELINE_NAME: solver}
