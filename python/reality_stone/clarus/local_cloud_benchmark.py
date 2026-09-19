"""Train-only mechanism benchmark for the V10 local/cloud kernel.

The benchmark is import-safe and performs no registered seed run by itself.
It exposes a balanced synthetic task, equal-width recurrent arms, a frozen
ridge readout, and true transition lesions.  It is evidence infrastructure,
not an AGI or neuroscience result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np

from .local_cloud_kernel import (
    KernelLesion,
    LocalCloudKernelConfig,
    LocalCloudObservation,
    LocalCloudTransitionKernel,
)


Arm = Literal["full", "local_only", "cloud_only", "no_memory"]
ARMS: tuple[Arm, ...] = ("full", "local_only", "cloud_only", "no_memory")


def _exact_int(value: object, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}")
    return value


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real number")
    return result


@dataclass(frozen=True)
class LocalCloudBenchmarkConfig:
    train_episodes: int = 256
    evaluation_episodes: int = 256
    episode_steps: int = 4
    noise_sigma: float = 0.04
    ridge_lambda: float = 1e-3

    def __post_init__(self) -> None:
        for name, minimum in (
            ("train_episodes", 32),
            ("evaluation_episodes", 32),
            ("episode_steps", 4),
        ):
            object.__setattr__(self, name, _exact_int(getattr(self, name), name, minimum))
        if self.train_episodes % 32 or self.evaluation_episodes % 32:
            raise ValueError("episode counts must be divisible by 32 for the factorial panel")
        object.__setattr__(self, "noise_sigma", _finite_positive(self.noise_sigma, "noise_sigma"))
        object.__setattr__(
            self, "ridge_lambda", _finite_positive(self.ridge_lambda, "ridge_lambda")
        )


@dataclass(frozen=True)
class LocalCloudEpisode:
    observations: tuple[LocalCloudObservation, ...]
    target: int
    context_index: int
    local_bits: tuple[int, int, int, int]


@dataclass(frozen=True)
class FrozenRidgeReadout:
    coefficients: tuple[float, ...]
    intercept: float
    ridge_lambda: float
    effective_degrees_of_freedom: float

    def predict_scores(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        matrix = np.asarray(tuple(tuple(row) for row in features), dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.coefficients):
            raise ValueError("features do not match the frozen readout width")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("features must be finite")
        return matrix @ np.asarray(self.coefficients) + self.intercept


@dataclass(frozen=True)
class DevelopmentSeedMetrics:
    seed: int
    accuracies: dict[str, float]
    lesion_accuracies: dict[str, float]
    effective_degrees_of_freedom: dict[str, float]
    coefficient_l2: dict[str, float]
    nonfinite_count: int
    label_state_bypass_count: int


@dataclass(frozen=True)
class DevelopmentResult:
    seed_count: int
    train_episodes_per_seed: int
    evaluation_episodes_per_seed: int
    mean_accuracies: dict[str, float]
    mean_lesion_accuracies: dict[str, float]
    mean_effective_degrees_of_freedom: dict[str, float]
    mean_coefficient_l2: dict[str, float]
    strongest_factorial_control: str
    paired_full_minus_per_seed_strongest: tuple[float, float, float]
    paired_factorial_interaction: tuple[float, float, float]
    paired_lesion_losses: dict[str, tuple[float, float, float]]
    gates: dict[str, bool]
    integrity: dict[str, int]
    state_scalar_count: dict[str, int]
    declared_scalar_multiply_upper_bound_per_tick: dict[str, int]
    overall: Literal["GO", "STOP"]
    seed_metrics: tuple[DevelopmentSeedMetrics, ...]


def _rng(seed: int, tag: int) -> np.random.Generator:
    seed = _exact_int(seed, "seed", 0)
    return np.random.default_rng(np.random.SeedSequence((seed, tag)))


def generate_episodes(
    seed: int,
    count: int,
    config: LocalCloudBenchmarkConfig,
    *,
    split: Literal["train", "evaluation"],
) -> tuple[LocalCloudEpisode, ...]:
    count = _exact_int(count, "count", 32)
    if count % 32:
        raise ValueError("count must be divisible by 32")
    if type(split) is not str or split not in {"train", "evaluation"}:
        raise ValueError("split must be an exact registered string")
    rng = _rng(seed, 110 if split == "train" else 220)
    factorial = tuple(
        (context, first, second, third)
        for context in range(4)
        for first in (-1, 1)
        for second in (-1, 1)
        for third in (-1, 1)
    )
    conditions = list(factorial * (count // len(factorial)))
    rng.shuffle(conditions)
    episodes: list[LocalCloudEpisode] = []
    context_weights = np.asarray(((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)), dtype=np.int64)
    for context_index, first, second, third in conditions:
        local_bits = np.asarray((first, second, third, rng.choice((-1, 1))), dtype=np.int64)
        local_events = rng.normal(0.0, config.noise_sigma, size=(config.episode_steps, 4, 4))
        shared_events = rng.normal(0.0, config.noise_sigma, size=(config.episode_steps, 4))
        local_events[0] += local_bits[:, None]
        shared_events[1, context_index] += 1.0
        observations = tuple(
            LocalCloudObservation(
                local=tuple(tuple(float(value) for value in row) for row in local_events[tick]),
                shared=tuple(float(value) for value in shared_events[tick]),
            )
            for tick in range(config.episode_steps)
        )
        target = int(np.sign(context_weights[context_index] @ local_bits[:3]))
        episodes.append(
            LocalCloudEpisode(
                observations=observations,
                target=target,
                context_index=context_index,
                local_bits=tuple(int(value) for value in local_bits),
            )
        )
    return tuple(episodes)


def kernel_for_arm(arm: Arm) -> LocalCloudTransitionKernel:
    if type(arm) is not str or arm not in ARMS:
        raise ValueError("arm must be an exact registered string")
    if arm == "full":
        config = LocalCloudKernelConfig()
    elif arm == "local_only":
        config = LocalCloudKernelConfig(
            cloud_retention=0.0,
            local_to_cloud=0.0,
            cloud_to_local=0.0,
            interaction_gain=0.0,
        )
    elif arm == "cloud_only":
        config = LocalCloudKernelConfig(local_retentions=(0.0, 0.0, 0.0, 0.0))
    else:
        config = LocalCloudKernelConfig(
            local_retentions=(0.0, 0.0, 0.0, 0.0),
            cloud_retention=0.0,
            local_to_cloud=0.0,
            cloud_to_local=0.0,
            interaction_gain=0.0,
        )
    return LocalCloudTransitionKernel(config)


def episode_features(
    episode: LocalCloudEpisode,
    kernel: LocalCloudTransitionKernel,
    *,
    lesion: KernelLesion = "intact",
    lesion_at_decision_only: bool = False,
) -> tuple[float, ...]:
    if type(episode) is not LocalCloudEpisode:
        raise ValueError("episode must be an exact LocalCloudEpisode")
    state = kernel.zero_state()
    result = None
    final_tick = len(episode.observations) - 1
    for tick, observation in enumerate(episode.observations):
        applied = lesion if (not lesion_at_decision_only or tick == final_tick) else "intact"
        result = kernel.step(state, observation, lesion=applied)
        state = result.state
    if result is None:
        raise ValueError("episode observations must be nonempty")
    return result.features


def feature_matrix(
    episodes: Sequence[LocalCloudEpisode],
    kernel: LocalCloudTransitionKernel,
    *,
    lesion: KernelLesion = "intact",
    lesion_at_decision_only: bool = False,
) -> np.ndarray:
    rows = tuple(
        episode_features(
            episode,
            kernel,
            lesion=lesion,
            lesion_at_decision_only=lesion_at_decision_only,
        )
        for episode in tuple(episodes)
    )
    if not rows:
        raise ValueError("episodes must be nonempty")
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape[1] != 20 or not np.all(np.isfinite(matrix)):
        raise ValueError("kernel must expose exactly twenty finite state features")
    return matrix


def fit_frozen_ridge(
    features: Sequence[Sequence[float]],
    targets: Sequence[int],
    *,
    ridge_lambda: float,
) -> FrozenRidgeReadout:
    matrix = np.asarray(tuple(tuple(row) for row in features), dtype=np.float64)
    labels = np.asarray(tuple(targets), dtype=np.float64)
    penalty = _finite_positive(ridge_lambda, "ridge_lambda")
    if matrix.ndim != 2 or matrix.shape[0] != labels.shape[0] or matrix.shape[0] < 2:
        raise ValueError("features and targets must have matching nontrivial rows")
    if matrix.shape[1] != 20 or not np.all(np.isfinite(matrix)):
        raise ValueError("readout requires twenty finite state features")
    if not np.all(np.isin(labels, (-1.0, 1.0))):
        raise ValueError("targets must be binary signs")
    centered = matrix - np.mean(matrix, axis=0)
    target_mean = float(np.mean(labels))
    centered_targets = labels - target_mean
    gram = centered.T @ centered
    regularized = gram + penalty * np.eye(matrix.shape[1])
    coefficients = np.linalg.solve(regularized, centered.T @ centered_targets)
    intercept = target_mean - float(np.mean(matrix, axis=0) @ coefficients)
    effective_df = float(np.trace(np.linalg.solve(regularized, gram))) + 1.0
    return FrozenRidgeReadout(
        coefficients=tuple(float(value) for value in coefficients),
        intercept=intercept,
        ridge_lambda=penalty,
        effective_degrees_of_freedom=effective_df,
    )


def accuracy(readout: FrozenRidgeReadout, features: np.ndarray, targets: Sequence[int]) -> float:
    labels = np.asarray(tuple(targets), dtype=np.int64)
    scores = readout.predict_scores(features)
    predictions = np.where(scores >= 0.0, 1, -1)
    if predictions.shape != labels.shape:
        raise ValueError("targets must match prediction rows")
    return float(np.mean(predictions == labels))


def _accuracy_and_nonfinite(
    readout: FrozenRidgeReadout, features: np.ndarray, targets: Sequence[int]
) -> tuple[float, int]:
    scores = readout.predict_scores(features)
    invalid = int(np.size(scores) - np.count_nonzero(np.isfinite(scores)))
    if invalid:
        return 0.0, invalid
    return accuracy(readout, features, targets), 0


def evaluate_development_seed(
    seed: int, config: LocalCloudBenchmarkConfig
) -> DevelopmentSeedMetrics:
    train = generate_episodes(seed, config.train_episodes, config, split="train")
    evaluation = generate_episodes(seed, config.evaluation_episodes, config, split="evaluation")
    train_targets = tuple(episode.target for episode in train)
    evaluation_targets = tuple(episode.target for episode in evaluation)
    accuracies: dict[str, float] = {}
    effective_df: dict[str, float] = {}
    coefficient_l2: dict[str, float] = {}
    nonfinite = 0
    full_readout: FrozenRidgeReadout | None = None
    full_kernel: LocalCloudTransitionKernel | None = None
    for arm in ARMS:
        kernel = kernel_for_arm(arm)
        train_features = feature_matrix(train, kernel)
        evaluation_features = feature_matrix(evaluation, kernel)
        readout = fit_frozen_ridge(
            train_features,
            train_targets,
            ridge_lambda=config.ridge_lambda,
        )
        accuracies[arm], invalid = _accuracy_and_nonfinite(
            readout, evaluation_features, evaluation_targets
        )
        nonfinite += invalid
        effective_df[arm] = readout.effective_degrees_of_freedom
        coefficient_l2[arm] = float(np.linalg.norm(readout.coefficients))
        if arm == "full":
            full_readout = readout
            full_kernel = kernel
    if full_readout is None or full_kernel is None:
        raise RuntimeError("registered full arm was not evaluated")

    lesion_features = {
        "cross_cut": feature_matrix(evaluation, full_kernel, lesion="cross_cut"),
        "local_reset": feature_matrix(
            evaluation,
            full_kernel,
            lesion="local_reset",
            lesion_at_decision_only=True,
        ),
        "cloud_reset": feature_matrix(
            evaluation,
            full_kernel,
            lesion="cloud_reset",
            lesion_at_decision_only=True,
        ),
    }
    lesion_accuracies: dict[str, float] = {}
    for name, features in lesion_features.items():
        lesion_accuracies[name], invalid = _accuracy_and_nonfinite(
            full_readout, features, evaluation_targets
        )
        nonfinite += invalid

    probe = evaluation[0]
    forged = LocalCloudEpisode(
        observations=probe.observations,
        target=-probe.target,
        context_index=probe.context_index,
        local_bits=probe.local_bits,
    )
    bypass_count = int(
        episode_features(probe, full_kernel) != episode_features(forged, full_kernel)
    )
    return DevelopmentSeedMetrics(
        seed=seed,
        accuracies=accuracies,
        lesion_accuracies=lesion_accuracies,
        effective_degrees_of_freedom=effective_df,
        coefficient_l2=coefficient_l2,
        nonfinite_count=nonfinite,
        label_state_bypass_count=bypass_count,
    )


def _paired_interval(
    values: Sequence[float], *, samples: int, seed: int
) -> tuple[float, float, float]:
    raw = np.asarray(tuple(values), dtype=np.float64)
    if raw.ndim != 1 or raw.size < 2 or not np.all(np.isfinite(raw)):
        raise ValueError("paired bootstrap values must be a finite vector with at least two rows")
    samples = _exact_int(samples, "bootstrap_samples", 100)
    seed = _exact_int(seed, "bootstrap_seed", 0)
    rng = _rng(seed, 990)
    indices = rng.integers(0, raw.size, size=(samples, raw.size))
    sampled = np.mean(raw[indices], axis=1)
    return (
        float(np.mean(raw)),
        float(np.quantile(sampled, 0.025)),
        float(np.quantile(sampled, 0.975)),
    )


def evaluate_registered_development(
    seeds: Sequence[int],
    config: LocalCloudBenchmarkConfig,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> DevelopmentResult:
    raw_seeds = tuple(seeds)
    if not raw_seeds or any(type(seed) is not int or seed < 0 for seed in raw_seeds):
        raise ValueError("seeds must be exact nonnegative built-in integers")
    if len(set(raw_seeds)) != len(raw_seeds):
        raise ValueError("duplicate development seeds are forbidden")
    metrics = tuple(evaluate_development_seed(seed, config) for seed in raw_seeds)
    mean_accuracies = {
        arm: float(np.mean(tuple(row.accuracies[arm] for row in metrics))) for arm in ARMS
    }
    lesion_names = ("cross_cut", "local_reset", "cloud_reset")
    mean_lesions = {
        name: float(np.mean(tuple(row.lesion_accuracies[name] for row in metrics)))
        for name in lesion_names
    }
    mean_df = {
        arm: float(np.mean(tuple(row.effective_degrees_of_freedom[arm] for row in metrics)))
        for arm in ARMS
    }
    mean_l2 = {
        arm: float(np.mean(tuple(row.coefficient_l2[arm] for row in metrics))) for arm in ARMS
    }
    controls = ("local_only", "cloud_only", "no_memory")
    strongest = max(controls, key=lambda arm: (mean_accuracies[arm], arm))
    full_minus_strongest = tuple(
        row.accuracies["full"] - max(row.accuracies[arm] for arm in controls) for row in metrics
    )
    interaction = tuple(
        row.accuracies["full"]
        - row.accuracies["local_only"]
        - row.accuracies["cloud_only"]
        + row.accuracies["no_memory"]
        for row in metrics
    )
    improvement_interval = _paired_interval(
        full_minus_strongest, samples=bootstrap_samples, seed=bootstrap_seed
    )
    interaction_interval = _paired_interval(
        interaction, samples=bootstrap_samples, seed=bootstrap_seed + 1
    )
    lesion_intervals = {
        name: _paired_interval(
            tuple(row.accuracies["full"] - row.lesion_accuracies[name] for row in metrics),
            samples=bootstrap_samples,
            seed=bootstrap_seed + 2 + index,
        )
        for index, name in enumerate(lesion_names)
    }
    integrity = {
        "duplicate_seed_count": len(raw_seeds) - len(set(raw_seeds)),
        "nonfinite_count": sum(row.nonfinite_count for row in metrics),
        "label_state_bypass_count": sum(row.label_state_bypass_count for row in metrics),
        "arm_count_mismatch": int(set(mean_accuracies) != set(ARMS)),
    }
    gates = {
        "full_accuracy": mean_accuracies["full"] >= 0.60,
        "paired_improvement_lcb": improvement_interval[1] >= 0.05,
        "factorial_interaction_lcb": interaction_interval[1] >= 0.05,
        "cross_cut_loss_lcb_positive": lesion_intervals["cross_cut"][1] > 0.0,
        "local_reset_loss_lcb_positive": lesion_intervals["local_reset"][1] > 0.0,
        "cloud_reset_loss_lcb_positive": lesion_intervals["cloud_reset"][1] > 0.0,
        "integrity": all(value == 0 for value in integrity.values()),
    }
    state_counts = {arm: 20 for arm in ARMS}
    multiply_bound = {arm: 76 for arm in ARMS}
    return DevelopmentResult(
        seed_count=len(raw_seeds),
        train_episodes_per_seed=config.train_episodes,
        evaluation_episodes_per_seed=config.evaluation_episodes,
        mean_accuracies=mean_accuracies,
        mean_lesion_accuracies=mean_lesions,
        mean_effective_degrees_of_freedom=mean_df,
        mean_coefficient_l2=mean_l2,
        strongest_factorial_control=strongest,
        paired_full_minus_per_seed_strongest=improvement_interval,
        paired_factorial_interaction=interaction_interval,
        paired_lesion_losses=lesion_intervals,
        gates=gates,
        integrity=integrity,
        state_scalar_count=state_counts,
        declared_scalar_multiply_upper_bound_per_tick=multiply_bound,
        overall="GO" if all(gates.values()) else "STOP",
        seed_metrics=metrics,
    )
