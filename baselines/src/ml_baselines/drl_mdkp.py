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
from ml_baselines.tensorize import public_view, tensorize_packing_instance
from ml_baselines.torch_utils import TorchUnavailableError, require_torch, resolve_device, torch_available

DRL_MDKP_BASELINE_NAME = "ml_drl_mdkp"
EPS = 1e-9
FEASIBILITY_TOLERANCE = 1e-6
CONTEXT_DIM = 6
DYNAMIC_ITEM_DIM = 4


@dataclass(frozen=True)
class DRLMDKPConfig:
    episodes: int = 5000
    learning_rate: float = 3e-4
    hidden_dim: int = 128
    rollout_steps: int = 0
    gamma: float = 1.0
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_episode_steps: int = 0
    repair_budget: int = 128
    initial_solution_source: str = "heuristic"
    device: str = "cpu"
    seed: int = 0

    @classmethod
    def from_config(cls, config: MLBaselineConfig | dict[str, Any] | None) -> "DRLMDKPConfig":
        base = cls()
        if config is None:
            return base
        payload = config.to_dict() if isinstance(config, MLBaselineConfig) else dict(config)
        episodes = int(payload.get("episodes", payload.get("epochs", base.episodes)))
        return cls(
            episodes=episodes,
            learning_rate=float(payload.get("learning_rate", payload.get("lr", base.learning_rate))),
            hidden_dim=int(payload.get("hidden_dim", base.hidden_dim)),
            rollout_steps=int(payload.get("rollout_steps", base.rollout_steps)),
            gamma=float(payload.get("gamma", base.gamma)),
            entropy_coef=float(payload.get("entropy_coef", base.entropy_coef)),
            value_coef=float(payload.get("value_coef", base.value_coef)),
            max_episode_steps=int(payload.get("max_episode_steps", base.max_episode_steps)),
            repair_budget=int(payload.get("repair_budget", base.repair_budget)),
            initial_solution_source=str(payload.get("initial_solution_source", base.initial_solution_source)),
            device=str(payload.get("device", base.device)),
            seed=int(payload.get("seed", base.seed)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "learning_rate": self.learning_rate,
            "hidden_dim": self.hidden_dim,
            "rollout_steps": self.rollout_steps,
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_episode_steps": self.max_episode_steps,
            "repair_budget": self.repair_budget,
            "initial_solution_source": self.initial_solution_source,
            "device": self.device,
            "seed": self.seed,
        }


class MDKPActorCriticNet:
    """Small actor-critic item policy for sequential MDKP construction."""

    def __new__(cls, *args: Any, **kwargs: Any):
        torch = require_torch()
        nn = torch.nn

        class _MDKPActorCriticNet(nn.Module):
            def __init__(self, item_dim: int, hidden_dim: int) -> None:
                super().__init__()
                policy_input = item_dim + DYNAMIC_ITEM_DIM + CONTEXT_DIM
                self.item_encoder = nn.Sequential(
                    nn.Linear(policy_input, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.policy_head = nn.Linear(hidden_dim, 1)
                self.value_head = nn.Sequential(
                    nn.Linear(hidden_dim + CONTEXT_DIM, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, item_features: Any, dynamic_item_features: Any, context: Any) -> tuple[Any, Any]:
                context_rows = context.reshape(1, -1).expand(item_features.shape[0], -1)
                encoded = self.item_encoder(torch.cat([item_features, dynamic_item_features, context_rows], dim=-1))
                logits = self.policy_head(encoded).reshape(-1)
                pooled = encoded.mean(dim=0)
                value = self.value_head(torch.cat([pooled, context.reshape(-1)], dim=-1)).reshape(())
                return logits, value

        return _MDKPActorCriticNet(*args, **kwargs)


def _values_weights_capacities(instance: dict[str, Any]) -> tuple[list[float], list[list[float]], list[float]]:
    public = public_view(instance)
    return (
        [float(value) for value in public["values"]],
        [[float(value) for value in row] for row in public["weights"]],
        [float(value) for value in public["capacities"]],
    )


def _usage(weights: list[list[float]], selected: set[int], num_resources: int) -> list[float]:
    usage = [0.0] * num_resources
    for item in selected:
        for resource in range(num_resources):
            usage[resource] += weights[item][resource]
    return usage


def _feasible_with_usage(
    usage: list[float],
    weights: list[list[float]],
    capacities: list[float],
    item: int,
) -> bool:
    return all(
        usage[resource] + weights[item][resource] <= capacities[resource] + FEASIBILITY_TOLERANCE
        for resource in range(len(capacities))
    )


def _selection_value(values: list[float], selected: set[int]) -> float:
    return sum(values[item] for item in selected)


def _density_score(values: list[float], weights: list[list[float]], capacities: list[float], item: int) -> float:
    pressure = sum(weights[item][resource] / max(capacities[resource], EPS) for resource in range(len(capacities)))
    return values[item] / max(pressure, FEASIBILITY_TOLERANCE)


def item_worth_order(instance: dict[str, Any]) -> list[int]:
    public = public_view(instance)
    values, weights, capacities = _values_weights_capacities(public)
    return sorted(
        range(int(public["num_items"])),
        key=lambda item: (_density_score(values, weights, capacities, item), values[item], -item),
        reverse=True,
    )


def heuristic_feasible_solution(instance: dict[str, Any]) -> list[int]:
    public = public_view(instance)
    values, weights, capacities = _values_weights_capacities(public)
    num_resources = int(public["num_resources"])
    selected: set[int] = set()
    usage = [0.0] * num_resources
    for item in item_worth_order(public):
        if _feasible_with_usage(usage, weights, capacities, item):
            selected.add(item)
            for resource in range(num_resources):
                usage[resource] += weights[item][resource]
    return sorted(selected)


class MDKPEnvironment:
    def __init__(
        self,
        instance: dict[str, Any],
        *,
        initial_solution: list[int] | None = None,
        max_episode_steps: int | None = None,
    ) -> None:
        self.instance = public_view(instance)
        self.num_items = int(self.instance["num_items"])
        self.num_resources = int(self.instance["num_resources"])
        self.values, self.weights, self.capacities = _values_weights_capacities(self.instance)
        self.total_value = max(sum(self.values), 1.0)
        self.selected = set(int(item) for item in (initial_solution or []))
        self.usage = _usage(self.weights, self.selected, self.num_resources)
        self.steps = 0
        self.max_episode_steps = int(max_episode_steps or max(1, 2 * self.num_items))
        self.best_selected = set(self.selected)
        self.best_value = _selection_value(self.values, self.best_selected)

    def feasible_actions(self) -> list[bool]:
        return [
            item not in self.selected and _feasible_with_usage(self.usage, self.weights, self.capacities, item)
            for item in range(self.num_items)
        ]

    def residuals(self) -> list[float]:
        return [max(0.0, self.capacities[resource] - self.usage[resource]) for resource in range(self.num_resources)]

    def context_features(self) -> list[float]:
        residual_ratios = [
            residual / max(self.capacities[resource], EPS)
            for resource, residual in enumerate(self.residuals())
        ]
        selected_value = _selection_value(self.values, self.selected)
        return [
            selected_value / self.total_value,
            len(self.selected) / max(float(self.num_items), 1.0),
            min(residual_ratios) if residual_ratios else 0.0,
            sum(residual_ratios) / max(float(len(residual_ratios)), 1.0),
            self.steps / max(float(self.max_episode_steps), 1.0),
            1.0,
        ]

    def dynamic_item_features(self) -> list[list[float]]:
        feasible = self.feasible_actions()
        residuals = self.residuals()
        features: list[list[float]] = []
        for item in range(self.num_items):
            min_ratio_after_add = 0.0
            if feasible[item]:
                ratios = [
                    (residuals[resource] - self.weights[item][resource]) / max(self.capacities[resource], EPS)
                    for resource in range(self.num_resources)
                ]
                min_ratio_after_add = min(ratios) if ratios else 0.0
            features.append(
                [
                    1.0 if item in self.selected else 0.0,
                    1.0 if feasible[item] else 0.0,
                    min_ratio_after_add,
                    _density_score(self.values, self.weights, self.capacities, item) / max(max(self.values), 1.0),
                ]
            )
        return features

    def step(self, action: int) -> tuple[float, bool]:
        previous_value = _selection_value(self.values, self.selected)
        feasible = self.feasible_actions()
        self.steps += 1
        if not (0 <= action < self.num_items) or not feasible[action]:
            return -0.05, True
        self.selected.add(action)
        for resource in range(self.num_resources):
            self.usage[resource] += self.weights[action][resource]
        current_value = _selection_value(self.values, self.selected)
        if current_value > self.best_value + EPS:
            self.best_value = current_value
            self.best_selected = set(self.selected)
        done = self.steps >= self.max_episode_steps or not any(self.feasible_actions())
        reward = (current_value - previous_value) / self.total_value
        if done:
            reward += current_value / self.total_value
        return reward, done


def _initial_selected(instance: dict[str, Any], source: str) -> list[int]:
    if source == "heuristic":
        return heuristic_feasible_solution(instance)
    if source == "empty":
        return []
    raise ValueError(f"Unsupported MDKP initial_solution_source `{source}`.")


def _state_tensors(env: MDKPEnvironment, item_features: Any, *, device: Any) -> tuple[Any, Any, Any]:
    torch = require_torch()
    dynamic = torch.tensor(env.dynamic_item_features(), dtype=item_features.dtype, device=device)
    context = torch.tensor(env.context_features(), dtype=item_features.dtype, device=device)
    mask = torch.tensor(env.feasible_actions(), dtype=torch.bool, device=device)
    return dynamic, context, mask


def _masked_logits(logits: Any, mask: Any) -> Any:
    return logits.masked_fill(~mask, -1.0e9)


def _episode_loss(
    model: Any,
    instance: dict[str, Any],
    item_features: Any,
    config: DRLMDKPConfig,
    *,
    device: Any,
) -> tuple[Any, float]:
    torch = require_torch()
    max_steps = config.max_episode_steps if config.max_episode_steps > 0 else 2 * int(instance["num_items"])
    env = MDKPEnvironment(instance, initial_solution=[], max_episode_steps=max_steps)
    log_probs: list[Any] = []
    values: list[Any] = []
    entropies: list[Any] = []
    rewards: list[float] = []
    done = False
    while not done:
        dynamic, context, mask = _state_tensors(env, item_features, device=device)
        if not bool(mask.any()):
            break
        logits, value = model(item_features, dynamic, context)
        distribution = torch.distributions.Categorical(logits=_masked_logits(logits, mask))
        action_tensor = distribution.sample()
        action = int(action_tensor.detach().cpu().item())
        reward, done = env.step(action)
        log_probs.append(distribution.log_prob(action_tensor))
        values.append(value)
        entropies.append(distribution.entropy())
        rewards.append(float(reward))
        if config.rollout_steps > 0 and len(rewards) >= config.rollout_steps:
            done = True
    if not rewards:
        return torch.zeros((), dtype=item_features.dtype, device=device, requires_grad=True), env.best_value
    returns: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + config.gamma * running
        returns.append(running)
    returns.reverse()
    return_tensor = torch.tensor(returns, dtype=item_features.dtype, device=device)
    value_tensor = torch.stack(values)
    log_prob_tensor = torch.stack(log_probs)
    advantage = return_tensor - value_tensor.detach()
    policy_loss = -(log_prob_tensor * advantage).mean()
    value_loss = ((value_tensor - return_tensor) ** 2).mean()
    entropy = torch.stack(entropies).mean()
    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
    return loss, env.best_value


def repair_mdkp_selection(
    instance: dict[str, Any],
    selected_items: list[int],
    *,
    repair_budget: int = 128,
    item_scores: list[float] | None = None,
) -> list[int]:
    public = public_view(instance)
    num_items = int(public["num_items"])
    num_resources = int(public["num_resources"])
    values, weights, capacities = _values_weights_capacities(public)
    selected = {item for item in selected_items if 0 <= int(item) < num_items}
    selected = {int(item) for item in selected}

    while True:
        usage = _usage(weights, selected, num_resources)
        violated = [resource for resource in range(num_resources) if usage[resource] > capacities[resource] + FEASIBILITY_TOLERANCE]
        if not violated:
            break
        remove_item = min(
            selected,
            key=lambda item: (
                values[item] / max(
                    sum(weights[item][resource] / max(capacities[resource], EPS) for resource in violated),
                    FEASIBILITY_TOLERANCE,
                ),
                item,
            ),
        )
        selected.remove(remove_item)

    scores = item_scores if item_scores is not None else [
        _density_score(values, weights, capacities, item)
        for item in range(num_items)
    ]
    order = sorted(range(num_items), key=lambda item: (float(scores[item]), values[item], -item), reverse=True)
    budget = max(0, int(repair_budget))
    improved = True
    while improved and budget > 0:
        improved = False
        usage = _usage(weights, selected, num_resources)
        for item in order:
            if item in selected:
                continue
            budget -= 1
            if _feasible_with_usage(usage, weights, capacities, item):
                selected.add(item)
                improved = True
                break
            if budget <= 0:
                break
        if improved:
            continue
        for remove_item in sorted(selected, key=lambda item: (values[item], item)):
            usage_without = [
                usage[resource] - weights[remove_item][resource]
                for resource in range(num_resources)
            ]
            for add_item in order:
                if add_item in selected or values[add_item] <= values[remove_item] + EPS:
                    continue
                budget -= 1
                if _feasible_with_usage(usage_without, weights, capacities, add_item):
                    selected.remove(remove_item)
                    selected.add(add_item)
                    improved = True
                    break
                if budget <= 0:
                    break
            if improved or budget <= 0:
                break
    return sorted(selected)


def _rollout_policy(
    model: Any,
    instance: dict[str, Any],
    item_features: Any,
    config: DRLMDKPConfig,
    *,
    device: Any,
    initial_solution: list[int],
) -> tuple[list[int], list[float]]:
    torch = require_torch()
    max_steps = config.max_episode_steps if config.max_episode_steps > 0 else 2 * int(public_view(instance)["num_items"])
    env = MDKPEnvironment(instance, initial_solution=initial_solution, max_episode_steps=max_steps)
    best = list(env.best_selected)
    last_scores = [0.0] * env.num_items
    model.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            dynamic, context, mask = _state_tensors(env, item_features, device=device)
            if not bool(mask.any()):
                break
            logits, _value = model(item_features, dynamic, context)
            last_scores = [float(value) for value in logits.detach().cpu().tolist()]
            masked = _masked_logits(logits, mask)
            action = int(torch.argmax(masked).detach().cpu().item())
            _reward, done = env.step(action)
            if env.best_value >= _selection_value(env.values, set(best)) - EPS:
                best = list(env.best_selected)
            if done:
                break
    return sorted(best), last_scores


def fit(
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    config: DRLMDKPConfig | MLBaselineConfig | dict[str, Any] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> TrainedState:
    if not torch_available():
        raise TorchUnavailableError("PyTorch is required for the DRL MDKP baseline.")
    torch = require_torch()
    resolved_config = config if isinstance(config, DRLMDKPConfig) else DRLMDKPConfig.from_config(config)
    seed_everything(resolved_config.seed)
    device = resolve_device(resolved_config.device)
    if not train_instances:
        raise ValueError("The DRL MDKP baseline requires public training instances.")

    public_train = [public_view(instance) for instance in train_instances]
    public_val = [public_view(instance) for instance in (val_instances or [])]
    train_tensors = [tensorize_packing_instance(instance, backend="torch", device=str(device)) for instance in public_train]
    val_tensors = [tensorize_packing_instance(instance, backend="torch", device=str(device)) for instance in public_val]
    item_dim = int(train_tensors[0].item_features.shape[-1])
    model = MDKPActorCriticNet(item_dim, resolved_config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=resolved_config.learning_rate)
    logger = MetricsLogger(Path(metrics_path) if metrics_path is not None else None)
    rng = random.Random(f"drl-mdkp:{resolved_config.seed}")
    best_state: dict[str, Any] | None = None
    best_loss = math.inf

    for episode in range(1, resolved_config.episodes + 1):
        index = rng.randrange(len(public_train))
        instance = public_train[index]
        tensor = train_tensors[index]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, episode_value = _episode_loss(
            model,
            instance,
            tensor.item_features,
            resolved_config,
            device=device,
        )
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        val_value_fraction = None
        if public_val and (episode == resolved_config.episodes or episode % max(1, resolved_config.episodes // 10) == 0):
            fractions = []
            for val_instance, val_tensor in zip(public_val, val_tensors, strict=True):
                selected, _scores = _rollout_policy(
                    model,
                    val_instance,
                    val_tensor.item_features,
                    resolved_config,
                    device=device,
                    initial_solution=[],
                )
                values, _weights, _capacities = _values_weights_capacities(val_instance)
                fractions.append(_selection_value(values, set(selected)) / max(sum(values), 1.0))
            val_value_fraction = sum(fractions) / max(1, len(fractions))
        logger.log(episode=episode, train_loss=loss_value, train_value=episode_value, val_value_fraction=val_value_fraction)
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "problem": "mdkp",
        "baseline_name": DRL_MDKP_BASELINE_NAME,
        "config": resolved_config.to_dict(),
        "train_instances": len(public_train),
        "validation_instances": len(public_val),
        "item_dim": item_dim,
        "history": logger.rows,
        "initialization": "public heuristic; no solver labels",
    }
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            {"model_state_dict": model.state_dict(), "item_dim": item_dim},
            metadata=metadata,
            use_torch=True,
        )
    return TrainedState(payload={"model": model, "item_dim": item_dim}, metadata=metadata)


def solve(
    instance: dict[str, Any],
    trained_state: TrainedState,
    config: DRLMDKPConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> list[int]:
    resolved_config = config if isinstance(config, DRLMDKPConfig) else DRLMDKPConfig.from_config(config)
    torch = require_torch()
    device = resolve_device(resolved_config.device)
    public = public_view(instance)
    tensor = tensorize_packing_instance(public, backend="torch", device=str(device))
    model = trained_state.payload["model"]
    candidates = [[], _initial_selected(public, resolved_config.initial_solution_source)]
    best_solution: list[int] = []
    best_value = -math.inf
    best_scores: list[float] | None = None
    for start in candidates:
        selected, scores = _rollout_policy(
            model,
            public,
            tensor.item_features,
            resolved_config,
            device=device,
            initial_solution=start,
        )
        repaired = repair_mdkp_selection(public, selected, repair_budget=resolved_config.repair_budget, item_scores=scores)
        values, _weights, _capacities = _values_weights_capacities(public)
        value = _selection_value(values, set(repaired))
        if value > best_value + EPS:
            best_solution = repaired
            best_value = value
            best_scores = scores
    return repair_mdkp_selection(public, best_solution, repair_budget=resolved_config.repair_budget, item_scores=best_scores)


def build_drl_mdkp_baselines(
    problem_name: str,
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    *,
    artifact_dir: str | Path | None = None,
    config: DRLMDKPConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    if problem_name != "mdkp" or not torch_available():
        return {}
    root = Path(artifact_dir) if artifact_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / f"{DRL_MDKP_BASELINE_NAME}.pt" if root is not None else None
    metrics_path = root / f"{DRL_MDKP_BASELINE_NAME}_metrics.jsonl" if root is not None else None
    resolved_config = config if isinstance(config, DRLMDKPConfig) else DRLMDKPConfig.from_config(config)
    trained_state = fit(
        train_instances,
        val_instances or [],
        resolved_config,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )

    def solver(instance: dict[str, Any]) -> list[int]:
        return solve(instance, trained_state, resolved_config)

    return {DRL_MDKP_BASELINE_NAME: solver}
