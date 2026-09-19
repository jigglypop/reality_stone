"""Strong learned-comparator and OOD evaluation for the confirmed V10 kernel."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Literal, Sequence

import numpy as np
import torch
from torch import nn

from .local_cloud_benchmark import (
    LocalCloudBenchmarkConfig,
    feature_matrix,
    fit_frozen_ridge,
    generate_episodes,
    kernel_for_arm,
)


ModelName = Literal["v10", "elman20", "gru20", "elman3"]
MODELS: tuple[ModelName, ...] = ("v10", "elman20", "gru20", "elman3")
PANELS: tuple[str, ...] = ("id", "noise", "horizon", "combined")


@dataclass(frozen=True)
class LearnedOODConfig:
    train_episodes: int = 256
    evaluation_episodes: int = 256
    epochs: int = 100
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0

    def __post_init__(self) -> None:
        for name, minimum in (
            ("train_episodes", 32),
            ("evaluation_episodes", 32),
            ("epochs", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an exact integer >= {minimum}")
        if self.train_episodes % 32 or self.evaluation_episodes % 32:
            raise ValueError("episode counts must be divisible by 32")
        for name in ("learning_rate", "weight_decay", "gradient_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite real number")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    brier: float


@dataclass(frozen=True)
class LearnedModelLedger:
    parameter_count: int
    recurrent_state_scalars: int
    multiply_estimate_per_tick: int
    train_seconds: float
    state_hash: str


@dataclass(frozen=True)
class OODSeedMetrics:
    seed: int
    panels: dict[str, dict[str, BinaryMetrics]]
    ledgers: dict[str, LearnedModelLedger]
    nonfinite_count: int


@dataclass(frozen=True)
class OODDevelopmentResult:
    seed_count: int
    panel_means: dict[str, dict[str, BinaryMetrics]]
    strong_contrasts: dict[str, tuple[float, float, float]]
    compute_matched_contrasts: dict[str, tuple[float, float, float]]
    mean_ledgers: dict[str, dict[str, float]]
    gates: dict[str, bool]
    integrity: dict[str, int]
    overall: Literal["GO", "STOP"]
    seed_metrics: tuple[OODSeedMetrics, ...]


class _SequenceClassifier(nn.Module):
    def __init__(self, kind: Literal["elman", "gru"], hidden: int) -> None:
        super().__init__()
        if kind == "elman":
            self.recurrent = nn.RNN(20, hidden, nonlinearity="tanh", batch_first=True)
        else:
            self.recurrent = nn.GRU(20, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        states, _ = self.recurrent(sequence)
        return self.readout(states[:, -1]).squeeze(-1)


def _panel_configs(config: LearnedOODConfig) -> dict[str, LocalCloudBenchmarkConfig]:
    return {
        "id": LocalCloudBenchmarkConfig(
            train_episodes=config.train_episodes,
            evaluation_episodes=config.evaluation_episodes,
            episode_steps=4,
            noise_sigma=0.04,
        ),
        "noise": LocalCloudBenchmarkConfig(
            train_episodes=config.train_episodes,
            evaluation_episodes=config.evaluation_episodes,
            episode_steps=4,
            noise_sigma=0.08,
        ),
        "horizon": LocalCloudBenchmarkConfig(
            train_episodes=config.train_episodes,
            evaluation_episodes=config.evaluation_episodes,
            episode_steps=8,
            noise_sigma=0.04,
        ),
        "combined": LocalCloudBenchmarkConfig(
            train_episodes=config.train_episodes,
            evaluation_episodes=config.evaluation_episodes,
            episode_steps=8,
            noise_sigma=0.08,
        ),
    }


def _raw_sequences(episodes) -> np.ndarray:
    rows = []
    for episode in episodes:
        steps = []
        for observation in episode.observations:
            steps.append(
                tuple(value for row in observation.local for value in row) + observation.shared
            )
        rows.append(tuple(steps))
    result = np.asarray(rows, dtype=np.float32)
    if result.ndim != 3 or result.shape[2] != 20 or not np.all(np.isfinite(result)):
        raise ValueError("raw recurrent input must have finite shape (episodes, ticks, 20)")
    return result


def _binary_metrics(scores: np.ndarray, targets: Sequence[int]) -> BinaryMetrics:
    labels = np.asarray(tuple(targets), dtype=np.float64)
    if scores.shape != labels.shape or not np.all(np.isfinite(scores)):
        raise ValueError("scores and targets must be matching finite vectors")
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -40.0, 40.0)))
    binary = (labels + 1.0) / 2.0
    return BinaryMetrics(
        accuracy=float(np.mean(np.where(scores >= 0.0, 1.0, -1.0) == labels)),
        brier=float(np.mean((probabilities - binary) ** 2)),
    )


def _state_hash(model: nn.Module) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest().upper()


def _train_model(
    kind: Literal["elman", "gru"],
    hidden: int,
    train_x: np.ndarray,
    train_y: Sequence[int],
    *,
    seed: int,
    config: LearnedOODConfig,
) -> tuple[_SequenceClassifier, LearnedModelLedger]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = _SequenceClassifier(kind, hidden)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = nn.BCEWithLogitsLoss()
    features = torch.from_numpy(train_x)
    labels = torch.as_tensor(
        (np.asarray(tuple(train_y), dtype=np.float32) + 1.0) / 2.0,
        dtype=torch.float32,
    )
    started = perf_counter()
    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(features), labels)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite learned-comparator loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
    elapsed = perf_counter() - started
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    multiplier = 1 if kind == "elman" else 3
    mac = multiplier * (20 * hidden + hidden * hidden + hidden)
    return model, LearnedModelLedger(
        parameter_count=parameter_count,
        recurrent_state_scalars=hidden,
        multiply_estimate_per_tick=mac,
        train_seconds=elapsed,
        state_hash=_state_hash(model),
    )


def _predict_model(model: nn.Module, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(features)).cpu().numpy().astype(np.float64)
    return scores


def evaluate_ood_seed(seed: int, config: LearnedOODConfig) -> OODSeedMetrics:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an exact nonnegative integer")
    panels = _panel_configs(config)
    train_config = panels["id"]
    train = generate_episodes(seed, config.train_episodes, train_config, split="train")
    train_targets = tuple(episode.target for episode in train)
    train_raw = _raw_sequences(train)

    learned = {}
    ledgers: dict[str, LearnedModelLedger] = {}
    for name, kind, hidden, tag in (
        ("elman20", "elman", 20, 11),
        ("gru20", "gru", 20, 12),
        ("elman3", "elman", 3, 13),
    ):
        model, ledger = _train_model(
            kind, hidden, train_raw, train_targets, seed=seed * 100 + tag, config=config
        )
        learned[name] = model
        ledgers[name] = ledger

    kernel = kernel_for_arm("full")
    train_features = feature_matrix(train, kernel)
    v10_readout = fit_frozen_ridge(
        train_features, train_targets, ridge_lambda=train_config.ridge_lambda
    )
    ledgers["v10"] = LearnedModelLedger(
        parameter_count=30,
        recurrent_state_scalars=20,
        multiply_estimate_per_tick=76,
        train_seconds=0.0,
        state_hash="DETERMINISTIC_KERNEL_PLUS_RIDGE",
    )

    result_panels: dict[str, dict[str, BinaryMetrics]] = {}
    nonfinite = 0
    for panel_name, panel_config in panels.items():
        evaluation = generate_episodes(
            seed, config.evaluation_episodes, panel_config, split="evaluation"
        )
        targets = tuple(episode.target for episode in evaluation)
        raw = _raw_sequences(evaluation)
        v10_scores = v10_readout.predict_scores(feature_matrix(evaluation, kernel))
        row = {"v10": _binary_metrics(v10_scores, targets)}
        for model_name, model in learned.items():
            scores = _predict_model(model, raw)
            nonfinite += int(scores.size - np.count_nonzero(np.isfinite(scores)))
            row[model_name] = _binary_metrics(scores, targets)
        result_panels[panel_name] = row
    return OODSeedMetrics(
        seed=seed,
        panels=result_panels,
        ledgers=ledgers,
        nonfinite_count=nonfinite,
    )


def _interval(values: Sequence[float], *, samples: int, seed: int) -> tuple[float, float, float]:
    raw = np.asarray(tuple(values), dtype=np.float64)
    if raw.size < 2 or not np.all(np.isfinite(raw)):
        raise ValueError("bootstrap requires at least two finite seed rows")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, raw.size, size=(samples, raw.size))
    means = np.mean(raw[indices], axis=1)
    return float(np.mean(raw)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_ood_development(
    seeds: Sequence[int],
    config: LearnedOODConfig,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> OODDevelopmentResult:
    raw_seeds = tuple(seeds)
    if len(raw_seeds) < 2 or any(type(seed) is not int or seed < 0 for seed in raw_seeds):
        raise ValueError("at least two exact nonnegative seeds are required")
    if len(set(raw_seeds)) != len(raw_seeds):
        raise ValueError("duplicate OOD seeds are forbidden")
    if type(bootstrap_samples) is not int or bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be an exact integer >= 100")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be an exact nonnegative integer")
    rows = tuple(evaluate_ood_seed(seed, config) for seed in raw_seeds)
    panel_means = {
        panel: {
            model: BinaryMetrics(
                accuracy=float(np.mean([row.panels[panel][model].accuracy for row in rows])),
                brier=float(np.mean([row.panels[panel][model].brier for row in rows])),
            )
            for model in MODELS
        }
        for panel in PANELS
    }
    strong = {
        panel: _interval(
            [
                row.panels[panel]["v10"].accuracy
                - max(
                    row.panels[panel]["elman20"].accuracy,
                    row.panels[panel]["gru20"].accuracy,
                )
                for row in rows
            ],
            samples=bootstrap_samples,
            seed=bootstrap_seed + index,
        )
        for index, panel in enumerate(PANELS)
    }
    matched = {
        panel: _interval(
            [
                row.panels[panel]["v10"].accuracy - row.panels[panel]["elman3"].accuracy
                for row in rows
            ],
            samples=bootstrap_samples,
            seed=bootstrap_seed + 10 + index,
        )
        for index, panel in enumerate(PANELS)
    }
    mean_ledgers = {
        model: {
            "parameter_count": float(np.mean([row.ledgers[model].parameter_count for row in rows])),
            "state_scalars": float(
                np.mean([row.ledgers[model].recurrent_state_scalars for row in rows])
            ),
            "mac_per_tick": float(
                np.mean([row.ledgers[model].multiply_estimate_per_tick for row in rows])
            ),
            "train_seconds": float(np.mean([row.ledgers[model].train_seconds for row in rows])),
        }
        for model in MODELS
    }
    integrity = {
        "duplicate_seed_count": len(raw_seeds) - len(set(raw_seeds)),
        "nonfinite_count": sum(row.nonfinite_count for row in rows),
        "panel_count_mismatch": int(set(panel_means) != set(PANELS)),
        "model_count_mismatch": int(any(set(panel_means[p]) != set(MODELS) for p in PANELS)),
    }
    gates = {}
    for panel in PANELS:
        gates[f"{panel}_strong_noninferiority"] = strong[panel][1] >= -0.02
        gates[f"{panel}_compute_matched_superiority"] = matched[panel][1] > 0.0
    gates["id_accuracy"] = panel_means["id"]["v10"].accuracy >= 0.60
    for panel in PANELS[1:]:
        gates[f"{panel}_accuracy"] = panel_means[panel]["v10"].accuracy >= 0.55
    gates["integrity"] = all(value == 0 for value in integrity.values())
    return OODDevelopmentResult(
        seed_count=len(raw_seeds),
        panel_means=panel_means,
        strong_contrasts=strong,
        compute_matched_contrasts=matched,
        mean_ledgers=mean_ledgers,
        gates=gates,
        integrity=integrity,
        overall="GO" if all(gates.values()) else "STOP",
        seed_metrics=rows,
    )
