"""Held-out recombination benchmark for shared option representations.

This deliberately removes hidden context, recurrence, hazard inference, and TD
credit. Its only question is whether reusable subaction identity, rather than a
hand-coded tree or atomic output, causes compositional generalization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SharedOptionBenchConfig:
    goal_count: int = 4
    subaction_count: int = 4
    train_samples_per_pair: int = 48
    test_samples_per_pair: int = 192
    noise_std: float = 0.18
    epochs: int = 240
    learning_rate: float = 0.45
    l2: float = 1e-4
    seeds: int = 40

    def __post_init__(self) -> None:
        if self.goal_count != self.subaction_count:
            raise ValueError("Latin-square holdout requires equal goal and subaction counts")
        if self.goal_count < 3:
            raise ValueError("at least three factors are required")
        for name, value in (
            ("train_samples_per_pair", self.train_samples_per_pair),
            ("test_samples_per_pair", self.test_samples_per_pair),
            ("epochs", self.epochs),
            ("seeds", self.seeds),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(self.noise_std) or self.noise_std < 0.0:
            raise ValueError("noise_std must be finite and nonnegative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.l2) or self.l2 < 0.0:
            raise ValueError("l2 must be finite and nonnegative")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=1, keepdims=True)


def _fit_linear_classifier(
    features: np.ndarray,
    targets: np.ndarray,
    class_count: int,
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    weights = rng.normal(0.0, 0.01, size=(class_count, features.shape[1]))
    bias = np.zeros(class_count, dtype=np.float64)
    target_matrix = np.eye(class_count, dtype=np.float64)[targets]
    for _ in range(epochs):
        probabilities = _softmax(features @ weights.T + bias)
        residual = (probabilities - target_matrix) / float(features.shape[0])
        weights -= learning_rate * (residual.T @ features + l2 * weights)
        bias -= learning_rate * np.sum(residual, axis=0)
    return weights, bias


def _predict(features: np.ndarray, model: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    weights, bias = model
    return _softmax(features @ weights.T + bias)


def _samples(
    pairs: list[tuple[int, int]],
    count_per_pair: int,
    config: SharedOptionBenchConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    goals: list[int] = []
    subactions: list[int] = []
    goal_features: list[np.ndarray] = []
    subaction_features: list[np.ndarray] = []
    for goal, subaction in pairs:
        for _ in range(count_per_pair):
            goal_vector = np.zeros(config.goal_count, dtype=np.float64)
            subaction_vector = np.zeros(config.subaction_count, dtype=np.float64)
            goal_vector[goal] = 1.0
            subaction_vector[subaction] = 1.0
            goal_features.append(goal_vector + rng.normal(0.0, config.noise_std, goal_vector.shape))
            subaction_features.append(
                subaction_vector + rng.normal(0.0, config.noise_std, subaction_vector.shape)
            )
            goals.append(goal)
            subactions.append(subaction)
    order = rng.permutation(len(goals))
    return (
        np.asarray(goal_features)[order],
        np.asarray(subaction_features)[order],
        np.asarray(goals, dtype=np.int64)[order],
        np.asarray(subactions, dtype=np.int64)[order],
    )


def _compound_probabilities(goal: np.ndarray, subaction: np.ndarray) -> np.ndarray:
    return (goal[:, :, None] * subaction[:, None, :]).reshape(goal.shape[0], -1)


def _metrics(probabilities: np.ndarray, targets: np.ndarray) -> tuple[float, float, float]:
    selected = probabilities[np.arange(targets.size), targets]
    accuracy = float(np.mean(np.argmax(probabilities, axis=1) == targets))
    nll = -float(np.mean(np.log(np.clip(selected, 1e-15, 1.0))))
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    return accuracy, nll, brier


def _seed_run(config: SharedOptionBenchConfig, seed: int) -> dict[str, tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    count = config.goal_count
    held_out = [(goal, goal) for goal in range(count)]
    training = [
        (goal, subaction)
        for goal in range(count)
        for subaction in range(count)
        if goal != subaction
    ]
    xg, xs, goal, subaction = _samples(
        training, config.train_samples_per_pair, config, rng
    )
    test_g, test_s, target_g, target_s = _samples(
        held_out, config.test_samples_per_pair, config, rng
    )
    compound = goal * count + subaction
    target = target_g * count + target_s
    fit_kwargs = {
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "l2": config.l2,
        "rng": rng,
    }

    atomic_model = _fit_linear_classifier(
        np.concatenate((xg, xs), axis=1), compound, count * count, **fit_kwargs
    )
    atomic = _predict(np.concatenate((test_g, test_s), axis=1), atomic_model)

    goal_model = _fit_linear_classifier(xg, goal, count, **fit_kwargs)
    goal_probability = _predict(test_g, goal_model)
    shared_model = _fit_linear_classifier(xs, subaction, count, **fit_kwargs)
    shared_subaction = _predict(test_s, shared_model)
    shared_dag = _compound_probabilities(goal_probability, shared_subaction)
    factorized_flat = shared_dag.copy()

    tree_subaction = np.zeros((target.size, count, count), dtype=np.float64)
    for branch in range(count):
        mask = goal == branch
        branch_model = _fit_linear_classifier(
            xs[mask], subaction[mask], count, **fit_kwargs
        )
        tree_subaction[:, branch, :] = _predict(test_s, branch_model)
    strict_tree = (goal_probability[:, :, None] * tree_subaction).reshape(target.size, -1)

    permuted_target = (subaction + goal) % count
    destroyed_model = _fit_linear_classifier(xs, permuted_target, count, **fit_kwargs)
    destroyed_raw = _predict(test_s, destroyed_model)
    destroyed_subaction = np.zeros((target.size, count, count), dtype=np.float64)
    for branch in range(count):
        for semantic in range(count):
            destroyed_subaction[:, branch, semantic] = destroyed_raw[
                :, (semantic + branch) % count
            ]
    destroyed_dag = (
        goal_probability[:, :, None] * destroyed_subaction
    ).reshape(target.size, -1)

    return {
        "atomic_flat": _metrics(atomic, target),
        "strict_tree": _metrics(strict_tree, target),
        "shared_dag": _metrics(shared_dag, target),
        "factorized_flat": _metrics(factorized_flat, target),
        "destroyed_dag": _metrics(destroyed_dag, target),
    }


def _mean_lcb(differences: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(differences))
    if differences.size < 2:
        return mean, mean
    standard_error = float(np.std(differences, ddof=1) / math.sqrt(differences.size))
    return mean, mean - 1.96 * standard_error


def evaluate_shared_options(
    config: SharedOptionBenchConfig = SharedOptionBenchConfig(),
) -> dict[str, object]:
    runs = [_seed_run(config, 2026081100 + seed) for seed in range(config.seeds)]
    arms = tuple(runs[0])
    summaries: dict[str, dict[str, float]] = {}
    for arm in arms:
        values = np.asarray([run[arm] for run in runs], dtype=np.float64)
        summaries[arm] = {
            "accuracy": float(np.mean(values[:, 0])),
            "nll": float(np.mean(values[:, 1])),
            "brier": float(np.mean(values[:, 2])),
        }

    def accuracy_difference(left: str, right: str) -> np.ndarray:
        return np.asarray(
            [run[left][0] - run[right][0] for run in runs], dtype=np.float64
        )

    shared_tree_mean, shared_tree_lcb = _mean_lcb(
        accuracy_difference("shared_dag", "strict_tree")
    )
    shared_atomic_mean, shared_atomic_lcb = _mean_lcb(
        accuracy_difference("shared_dag", "atomic_flat")
    )
    shared_destroyed_mean, shared_destroyed_lcb = _mean_lcb(
        accuracy_difference("shared_dag", "destroyed_dag")
    )
    shared_factorized_mean, shared_factorized_lcb = _mean_lcb(
        accuracy_difference("shared_dag", "factorized_flat")
    )
    sharing_pass = (
        shared_tree_lcb > 0.10
        and shared_atomic_lcb > 0.10
        and shared_destroyed_lcb > 0.10
    )
    dag_specific_pass = shared_factorized_lcb > 0.01
    return {
        "schema": "clarus.shared-option-topology.validation.v1",
        "config": config.__dict__,
        "summaries": summaries,
        "effects": {
            "shared_minus_tree_accuracy_mean": shared_tree_mean,
            "shared_minus_tree_accuracy_lcb": shared_tree_lcb,
            "shared_minus_atomic_accuracy_mean": shared_atomic_mean,
            "shared_minus_atomic_accuracy_lcb": shared_atomic_lcb,
            "shared_minus_destroyed_accuracy_mean": shared_destroyed_mean,
            "shared_minus_destroyed_accuracy_lcb": shared_destroyed_lcb,
            "shared_minus_factorized_accuracy_mean": shared_factorized_mean,
            "shared_minus_factorized_accuracy_lcb": shared_factorized_lcb,
        },
        "gates": {
            "sharing_identity": sharing_pass,
            "dag_specificity": dag_specific_pass,
        },
        "verdict": (
            "DAG_GO"
            if sharing_pass and dag_specific_pass
            else "FACTORIZATION_GO_DAG_UNRESOLVED"
            if sharing_pass
            else "STOP"
        ),
    }


__all__ = ["SharedOptionBenchConfig", "evaluate_shared_options"]
