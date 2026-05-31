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
from ml_baselines.torch_utils import TorchUnavailableError, require_torch, resolve_device, torch_available
from ml_baselines.tsp_neural_constructor import (
    bounded_two_opt_from_matrix,
    canonicalize_tour,
    distance_matrix_from_points,
    is_valid_tour,
    tour_length_from_matrix,
)

ATTENTION_TSP_BASELINE_NAME = "ml_attention_tsp"
EPS = 1e-9


@dataclass(frozen=True)
class AttentionTSPConfig:
    epochs: int = 100
    steps_per_epoch: int = 10
    batch_size: int = 128
    embedding_dim: int = 128
    hidden_dim: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 3
    learning_rate: float = 1e-4
    baseline_update_threshold: float = 0.0
    max_grad_norm: float = 1.0
    inference_samples: int = 128
    two_opt_budget: int = 0
    device: str = "cpu"
    seed: int = 0

    @classmethod
    def from_config(cls, config: MLBaselineConfig | dict[str, Any] | None) -> "AttentionTSPConfig":
        base = cls()
        if config is None:
            return base
        payload = config.to_dict() if isinstance(config, MLBaselineConfig) else dict(config)
        return cls(
            epochs=int(payload.get("epochs", base.epochs)),
            steps_per_epoch=int(payload.get("steps_per_epoch", base.steps_per_epoch)),
            batch_size=int(payload.get("batch_size", base.batch_size)),
            embedding_dim=int(payload.get("embedding_dim", payload.get("hidden_dim", base.embedding_dim))),
            hidden_dim=int(payload.get("hidden_dim", base.hidden_dim)),
            n_heads=int(payload.get("n_heads", base.n_heads)),
            n_encoder_layers=int(payload.get("n_encoder_layers", payload.get("layers", base.n_encoder_layers))),
            learning_rate=float(payload.get("learning_rate", payload.get("lr", base.learning_rate))),
            baseline_update_threshold=float(payload.get("baseline_update_threshold", base.baseline_update_threshold)),
            max_grad_norm=float(payload.get("max_grad_norm", base.max_grad_norm)),
            inference_samples=int(
                payload.get(
                    "inference_samples",
                    payload.get("samples", payload.get("candidates", base.inference_samples)),
                )
            ),
            two_opt_budget=int(payload.get("two_opt_budget", payload.get("repair_budget", base.two_opt_budget))),
            device=str(payload.get("device", base.device)),
            seed=int(payload.get("seed", base.seed)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "batch_size": self.batch_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "n_heads": self.n_heads,
            "n_encoder_layers": self.n_encoder_layers,
            "learning_rate": self.learning_rate,
            "baseline_update_threshold": self.baseline_update_threshold,
            "max_grad_norm": self.max_grad_norm,
            "inference_samples": self.inference_samples,
            "two_opt_budget": self.two_opt_budget,
            "device": self.device,
            "seed": self.seed,
        }


class AttentionTSPModel:
    """Compact Attention Model style encoder/decoder for Euclidean TSP."""

    def __new__(cls, *args: Any, **kwargs: Any):
        torch = require_torch()
        nn = torch.nn

        class _AttentionTSPModel(nn.Module):
            def __init__(self, embedding_dim: int, hidden_dim: int, n_heads: int, n_encoder_layers: int) -> None:
                super().__init__()
                self.embedding_dim = int(embedding_dim)
                self.node_embed = nn.Linear(2, embedding_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=max(1, int(n_heads)),
                    dim_feedforward=max(int(hidden_dim), int(embedding_dim)),
                    dropout=0.0,
                    batch_first=True,
                    activation="relu",
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(n_encoder_layers)))
                self.context_proj = nn.Linear(embedding_dim * 3, embedding_dim)
                self.query_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
                self.key_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

            def encode(self, coords: Any) -> Any:
                embedded = self.node_embed(coords)
                return self.encoder(embedded)

            def decode(
                self,
                coords: Any,
                *,
                decode_type: str,
            ) -> tuple[Any, Any]:
                torch = require_torch()
                encoded = self.encode(coords)
                batch_size, num_cities, _ = encoded.shape
                graph_embedding = encoded.mean(dim=1)
                visited = torch.zeros((batch_size, num_cities), dtype=torch.bool, device=coords.device)
                first = torch.zeros(batch_size, dtype=torch.long, device=coords.device)
                current = first
                visited.scatter_(1, current.reshape(-1, 1), True)
                tours = [current]
                log_probs: list[Any] = []
                for step in range(1, num_cities):
                    first_embedding = encoded[torch.arange(batch_size, device=coords.device), first]
                    current_embedding = encoded[torch.arange(batch_size, device=coords.device), current]
                    context = self.context_proj(torch.cat([graph_embedding, first_embedding, current_embedding], dim=-1))
                    query = self.query_proj(context).unsqueeze(1)
                    keys = self.key_proj(encoded)
                    logits = (query * keys).sum(dim=-1) / math.sqrt(float(self.embedding_dim))
                    logits = logits.masked_fill(visited.clone(), -1.0e9)
                    if decode_type == "greedy":
                        next_city = torch.argmax(logits, dim=-1)
                        log_probability = torch.log_softmax(logits, dim=-1).gather(1, next_city.reshape(-1, 1)).squeeze(1)
                    elif decode_type == "sampling":
                        distribution = torch.distributions.Categorical(logits=logits)
                        next_city = distribution.sample()
                        log_probability = distribution.log_prob(next_city)
                    else:
                        raise ValueError(f"Unsupported TSP decode_type `{decode_type}`.")
                    tours.append(next_city)
                    log_probs.append(log_probability)
                    current = next_city
                    visited.scatter_(1, current.reshape(-1, 1), True)
                return torch.stack(tours, dim=1), torch.stack(log_probs, dim=1).sum(dim=1)

        return _AttentionTSPModel(*args, **kwargs)


def _points(instance: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in public_view(instance)["points"]]


def normalize_points(points: list[tuple[float, float]]) -> list[list[float]]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    scale = max(max_x - min_x, max_y - min_y, EPS)
    return [[(x - min_x) / scale, (y - min_y) / scale] for x, y in points]


def _coords_tensor(instances: list[dict[str, Any]], *, device: Any) -> Any:
    torch = require_torch()
    coords = [normalize_points(_points(instance)) for instance in instances]
    return torch.tensor(coords, dtype=torch.float32, device=device)


def _tour_lengths(coords: Any, tours: Any) -> Any:
    torch = require_torch()
    ordered = coords.gather(1, tours.unsqueeze(-1).expand(-1, -1, 2))
    shifted = torch.roll(ordered, shifts=-1, dims=1)
    return torch.linalg.vector_norm(ordered - shifted, dim=-1).sum(dim=1)


def _group_by_city_count(instances: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for instance in instances:
        groups.setdefault(int(instance["num_cities"]), []).append(instance)
    return groups


def _sample_batch(groups: dict[int, list[dict[str, Any]]], *, batch_size: int, rng: random.Random) -> list[dict[str, Any]]:
    num_cities = rng.choice(sorted(groups))
    bucket = groups[num_cities]
    if len(bucket) >= batch_size:
        return rng.sample(bucket, batch_size)
    return [rng.choice(bucket) for _ in range(max(1, int(batch_size)))]


def sample_tour(
    model: Any,
    instance: dict[str, Any],
    config: AttentionTSPConfig | MLBaselineConfig | dict[str, Any] | None = None,
    *,
    decode_type: str = "greedy",
) -> list[int]:
    resolved_config = config if isinstance(config, AttentionTSPConfig) else AttentionTSPConfig.from_config(config)
    torch = require_torch()
    device = resolve_device(resolved_config.device)
    coords = _coords_tensor([public_view(instance)], device=device)
    model.eval()
    with torch.no_grad():
        tour_tensor, _log_prob = model.decode(coords, decode_type=decode_type)
    return canonicalize_tour([int(value) for value in tour_tensor[0].detach().cpu().tolist()], int(public_view(instance)["num_cities"]))


def _candidate_tours(model: Any, instance: dict[str, Any], config: AttentionTSPConfig) -> list[list[int]]:
    torch = require_torch()
    device = resolve_device(config.device)
    public = public_view(instance)
    num_cities = int(public["num_cities"])
    coords = _coords_tensor([public], device=device)
    model.eval()
    candidates: list[list[int]] = []

    def add(candidate: list[int]) -> None:
        try:
            tour = canonicalize_tour(candidate, num_cities)
        except Exception:
            return
        if is_valid_tour(num_cities, tour) and tour not in candidates:
            candidates.append(tour)

    with torch.no_grad():
        greedy, _ = model.decode(coords, decode_type="greedy")
        add([int(value) for value in greedy[0].detach().cpu().tolist()])
        for _ in range(max(0, int(config.inference_samples))):
            sampled, _ = model.decode(coords, decode_type="sampling")
            add([int(value) for value in sampled[0].detach().cpu().tolist()])
    return candidates or [list(range(num_cities))]


def decode_attention_tsp(
    instance: dict[str, Any],
    model: Any,
    config: AttentionTSPConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> list[int]:
    resolved_config = config if isinstance(config, AttentionTSPConfig) else AttentionTSPConfig.from_config(config)
    public = public_view(instance)
    num_cities = int(public["num_cities"])
    matrix = distance_matrix_from_points(_points(public))
    best: list[int] | None = None
    best_length = math.inf
    for candidate in _candidate_tours(model, public, resolved_config):
        improved = bounded_two_opt_from_matrix(matrix, candidate, max_evaluations=resolved_config.two_opt_budget)
        length = tour_length_from_matrix(matrix, improved)
        if best is None or length < best_length - EPS:
            best = improved
            best_length = length
    if best is None:
        best = list(range(num_cities))
    return canonicalize_tour(best, num_cities)


def fit(
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    config: AttentionTSPConfig | MLBaselineConfig | dict[str, Any] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> TrainedState:
    if not torch_available():
        raise TorchUnavailableError("PyTorch is required for the Attention TSP baseline.")
    torch = require_torch()
    resolved_config = config if isinstance(config, AttentionTSPConfig) else AttentionTSPConfig.from_config(config)
    seed_everything(resolved_config.seed)
    device = resolve_device(resolved_config.device)
    if not train_instances:
        raise ValueError("The Attention TSP baseline requires public training instances.")

    public_train = [public_view(instance) for instance in train_instances]
    public_val = [public_view(instance) for instance in (val_instances or [])]
    groups = _group_by_city_count(public_train)
    model = AttentionTSPModel(
        resolved_config.embedding_dim,
        resolved_config.hidden_dim,
        resolved_config.n_heads,
        resolved_config.n_encoder_layers,
    ).to(device)
    baseline_model = AttentionTSPModel(
        resolved_config.embedding_dim,
        resolved_config.hidden_dim,
        resolved_config.n_heads,
        resolved_config.n_encoder_layers,
    ).to(device)
    baseline_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=resolved_config.learning_rate)
    logger = MetricsLogger(Path(metrics_path) if metrics_path is not None else None)
    rng = random.Random(f"attention-tsp:{resolved_config.seed}")
    best_state: dict[str, Any] | None = None
    best_validation_length = math.inf

    for epoch in range(1, resolved_config.epochs + 1):
        model.train()
        epoch_costs: list[float] = []
        epoch_losses: list[float] = []
        for _step in range(max(1, int(resolved_config.steps_per_epoch))):
            batch = _sample_batch(groups, batch_size=resolved_config.batch_size, rng=rng)
            coords = _coords_tensor(batch, device=device)
            sampled_tours, log_probs = model.decode(coords, decode_type="sampling")
            costs = _tour_lengths(coords, sampled_tours)
            with torch.no_grad():
                baseline_tours, _ = baseline_model.decode(coords, decode_type="greedy")
                baseline_costs = _tour_lengths(coords, baseline_tours)
            loss = ((costs - baseline_costs).detach() * log_probs).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if resolved_config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), resolved_config.max_grad_norm)
            optimizer.step()
            epoch_costs.append(float(costs.mean().detach().cpu().item()))
            epoch_losses.append(float(loss.detach().cpu().item()))
        train_length = sum(epoch_costs) / max(1, len(epoch_costs))
        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        val_length = None
        if public_val:
            model.eval()
            lengths = []
            with torch.no_grad():
                for num_cities, bucket in _group_by_city_count(public_val).items():
                    _ = num_cities
                    for start in range(0, len(bucket), max(1, int(resolved_config.batch_size))):
                        batch = bucket[start : start + max(1, int(resolved_config.batch_size))]
                        coords = _coords_tensor(batch, device=device)
                        tours, _ = model.decode(coords, decode_type="greedy")
                        lengths.extend(float(value) for value in _tour_lengths(coords, tours).detach().cpu().tolist())
            val_length = sum(lengths) / max(1, len(lengths))
            if val_length + resolved_config.baseline_update_threshold < best_validation_length:
                baseline_model.load_state_dict(model.state_dict())
                best_validation_length = val_length
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            if train_length < best_validation_length:
                baseline_model.load_state_dict(model.state_dict())
                best_validation_length = train_length
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        logger.log(epoch=epoch, train_loss=train_loss, train_length=train_length, val_length=val_length)

    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "problem": "tsp",
        "baseline_name": ATTENTION_TSP_BASELINE_NAME,
        "config": resolved_config.to_dict(),
        "train_instances": len(public_train),
        "validation_instances": len(public_val),
        "history": logger.rows,
        "training_mode": "reinforce_with_greedy_rollout_baseline",
    }
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            {
                "model_state_dict": model.state_dict(),
                "embedding_dim": resolved_config.embedding_dim,
                "hidden_dim": resolved_config.hidden_dim,
                "n_heads": resolved_config.n_heads,
                "n_encoder_layers": resolved_config.n_encoder_layers,
            },
            metadata=metadata,
            use_torch=True,
        )
    return TrainedState(payload={"model": model, "embedding_dim": resolved_config.embedding_dim}, metadata=metadata)


def solve(
    instance: dict[str, Any],
    trained_state: TrainedState,
    config: AttentionTSPConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> list[int]:
    resolved_config = config if isinstance(config, AttentionTSPConfig) else AttentionTSPConfig.from_config(config)
    return decode_attention_tsp(instance, trained_state.payload["model"], resolved_config)


def build_attention_tsp_baselines(
    problem_name: str,
    train_instances: list[dict[str, Any]],
    val_instances: list[dict[str, Any]] | None,
    *,
    artifact_dir: str | Path | None = None,
    config: AttentionTSPConfig | MLBaselineConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    if problem_name != "tsp" or not torch_available():
        return {}
    root = Path(artifact_dir) if artifact_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / f"{ATTENTION_TSP_BASELINE_NAME}.pt" if root is not None else None
    metrics_path = root / f"{ATTENTION_TSP_BASELINE_NAME}_metrics.jsonl" if root is not None else None
    resolved_config = config if isinstance(config, AttentionTSPConfig) else AttentionTSPConfig.from_config(config)
    trained_state = fit(
        train_instances,
        val_instances or [],
        resolved_config,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )

    def solver(instance: dict[str, Any]) -> list[int]:
        return solve(instance, trained_state, resolved_config)

    return {ATTENTION_TSP_BASELINE_NAME: solver}
