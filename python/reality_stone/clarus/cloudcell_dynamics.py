"""Leakage-resistant neural dynamics gates for the CloudCell bridge.

The formal CloudCell type is tested elsewhere.  This module asks a different,
empirical question: does a target unit's next measured state require both its
own causal history and a time-aligned population context?

Only NumPy is required for the gate itself.  The optional PredictionCode
``.mat`` loader imports SciPy lazily so the base package remains lightweight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
EQUATION_VARIANTS = ("additive", "innovation", "interaction", "nonlinear")
AML310_EXCLUDED_INTERVALS: Mapping[str, tuple[tuple[float, float], ...]] = {
    "BrainScanner20200130_105254": ((65.0, 75.0),),
    "BrainScanner20200310_141211": ((200.0, 210.0), (240.0, 250.0)),
}


@dataclass(frozen=True)
class NeuralRecording:
    """One population recording in units-by-time orientation."""

    recording_id: str
    time: np.ndarray
    activity: np.ndarray


@dataclass(frozen=True)
class TargetDynamicsScore:
    """Held-out ablation scores for one target unit."""

    target_index: int
    n_train: int
    n_validation: int
    n_test: int
    r2_intercept: float
    r2_current: float
    r2_local: float
    r2_cloud: float
    r2_full: float
    r2_shifted_cloud: float
    ridge_current: float
    ridge_local: float
    ridge_cloud: float
    ridge_full: float

    @property
    def delta_memory(self) -> float:
        """Increment from history beyond the currently observed target state."""

        return self.r2_local - self.r2_current

    @property
    def delta_current_state(self) -> float:
        return self.r2_current - self.r2_intercept

    @property
    def delta_cloud_given_local(self) -> float:
        return self.r2_full - self.r2_local

    @property
    def delta_local_given_cloud(self) -> float:
        return self.r2_full - self.r2_cloud

    @property
    def delta_time_alignment(self) -> float:
        return self.r2_full - self.r2_shifted_cloud

    def to_dict(self) -> dict[str, int | float]:
        return {
            "target_index": self.target_index,
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "n_test": self.n_test,
            "r2_intercept": self.r2_intercept,
            "r2_current": self.r2_current,
            "r2_local": self.r2_local,
            "r2_cloud": self.r2_cloud,
            "r2_full": self.r2_full,
            "r2_shifted_cloud": self.r2_shifted_cloud,
            "delta_current_state": self.delta_current_state,
            "delta_memory": self.delta_memory,
            "delta_cloud_given_local": self.delta_cloud_given_local,
            "delta_local_given_cloud": self.delta_local_given_cloud,
            "delta_time_alignment": self.delta_time_alignment,
            "ridge_current": self.ridge_current,
            "ridge_local": self.ridge_local,
            "ridge_cloud": self.ridge_cloud,
            "ridge_full": self.ridge_full,
        }


@dataclass(frozen=True)
class RecordingDynamicsGate:
    """Aggregate gate for one independently recorded animal."""

    recording_id: str
    n_units: int
    n_timepoints: int
    horizon_steps: int
    equation_variant: str
    scores: tuple[TargetDynamicsScore, ...]
    min_cloud_delta: float = 0.001
    min_memory_delta: float = 0.001
    min_positive_fraction: float = 0.6

    def _values(self, name: str) -> np.ndarray:
        return np.asarray([getattr(score, name) for score in self.scores], dtype=float)

    def median(self, name: str) -> float:
        values = self._values(name)
        return float(np.median(values)) if values.size else float("nan")

    def positive_fraction(self, name: str) -> float:
        values = self._values(name)
        return float(np.mean(values > 0.0)) if values.size else 0.0

    @property
    def passed(self) -> bool:
        if len(self.scores) < 3:
            return False
        return (
            self.median("delta_memory") > self.min_memory_delta
            and self.median("delta_cloud_given_local") > self.min_cloud_delta
            and self.median("delta_local_given_cloud") > self.min_cloud_delta
            and self.median("delta_time_alignment") > self.min_cloud_delta
            and self.positive_fraction("delta_cloud_given_local")
            >= self.min_positive_fraction
        )

    def summary(self) -> dict[str, object]:
        return {
            "recording_id": self.recording_id,
            "n_units": self.n_units,
            "n_timepoints": self.n_timepoints,
            "horizon_steps": self.horizon_steps,
            "equation_variant": self.equation_variant,
            "n_targets_evaluated": len(self.scores),
            "median_r2_intercept": self.median("r2_intercept"),
            "median_r2_current": self.median("r2_current"),
            "median_r2_local": self.median("r2_local"),
            "median_r2_cloud": self.median("r2_cloud"),
            "median_r2_full": self.median("r2_full"),
            "median_delta_memory": self.median("delta_memory"),
            "median_delta_cloud_given_local": self.median("delta_cloud_given_local"),
            "median_delta_local_given_cloud": self.median("delta_local_given_cloud"),
            "median_delta_time_alignment": self.median("delta_time_alignment"),
            "positive_fraction_cloud_given_local": self.positive_fraction(
                "delta_cloud_given_local"
            ),
            "criteria": {
                "min_memory_delta": self.min_memory_delta,
                "min_cloud_delta": self.min_cloud_delta,
                "min_positive_fraction": self.min_positive_fraction,
            },
            "passed": self.passed,
        }

    def to_dict(self, *, include_targets: bool = True) -> dict[str, object]:
        result = self.summary()
        if include_targets:
            result["targets"] = [score.to_dict() for score in self.scores]
        return result


@dataclass(frozen=True)
class DynamicsPanelGate:
    """Panel result; each recording, rather than each neuron, is a replicate."""

    recordings: tuple[RecordingDynamicsGate, ...]
    min_recordings_passed: int = 3

    @property
    def pass_count(self) -> int:
        return sum(recording.passed for recording in self.recordings)

    @property
    def passed(self) -> bool:
        return self.pass_count >= self.min_recordings_passed

    def to_dict(self, *, include_targets: bool = True) -> dict[str, object]:
        return {
            "recording_count": len(self.recordings),
            "recordings_passed": self.pass_count,
            "min_recordings_passed": self.min_recordings_passed,
            "passed": self.passed,
            "replicate_unit": "independently recorded animal",
            "recordings": [
                recording.to_dict(include_targets=include_targets)
                for recording in self.recordings
            ],
        }


@dataclass(frozen=True)
class LatentClosureScore:
    """Held-out Markov/composition score for an ensemble latent state."""

    recording_id: str
    n_units: int
    n_timepoints: int
    n_components: int
    state_order: int
    horizon_steps: int
    n_train: int
    n_validation: int
    n_test: int
    r2_persistence: float
    r2_diagonal: float
    r2_direct: float
    r2_composed: float
    min_transition_delta: float = 0.01
    max_composition_gap: float = 0.05

    @property
    def delta_coupling(self) -> float:
        return self.r2_direct - self.r2_diagonal

    @property
    def delta_direct_over_persistence(self) -> float:
        return self.r2_direct - self.r2_persistence

    @property
    def delta_composed_over_persistence(self) -> float:
        return self.r2_composed - self.r2_persistence

    @property
    def composition_gap(self) -> float:
        return self.r2_direct - self.r2_composed

    @property
    def passed(self) -> bool:
        return (
            self.delta_direct_over_persistence > self.min_transition_delta
            and self.delta_composed_over_persistence > 0.0
            and self.composition_gap <= self.max_composition_gap
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "recording_id": self.recording_id,
            "n_units": self.n_units,
            "n_timepoints": self.n_timepoints,
            "n_components": self.n_components,
            "state_order": self.state_order,
            "horizon_steps": self.horizon_steps,
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "n_test": self.n_test,
            "r2_persistence": self.r2_persistence,
            "r2_diagonal": self.r2_diagonal,
            "r2_direct": self.r2_direct,
            "r2_composed": self.r2_composed,
            "delta_coupling": self.delta_coupling,
            "delta_direct_over_persistence": self.delta_direct_over_persistence,
            "delta_composed_over_persistence": self.delta_composed_over_persistence,
            "composition_gap": self.composition_gap,
            "criteria": {
                "min_transition_delta": self.min_transition_delta,
                "max_composition_gap": self.max_composition_gap,
            },
            "passed": self.passed,
        }


@dataclass(frozen=True)
class _RidgeModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    ridge: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = (features - self.feature_mean) / self.feature_scale
        design = np.column_stack([np.ones(len(standardized)), standardized])
        return design @ self.coefficients


def _fit_ridge(features: np.ndarray, target: np.ndarray, ridge: float) -> _RidgeModel:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (features - mean) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    penalty = np.eye(design.shape[1], dtype=float) * ridge
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ target
    try:
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(lhs) @ rhs
    return _RidgeModel(mean, scale, coefficients, float(ridge))


def _select_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    ridge_grid: Sequence[float],
) -> float:
    best_ridge = float(ridge_grid[0])
    best_error = float("inf")
    for ridge in ridge_grid:
        prediction = _fit_ridge(train_x, train_y, float(ridge)).predict(validation_x)
        error = float(np.mean(np.square(validation_y - prediction)))
        if error < best_error:
            best_error = error
            best_ridge = float(ridge)
    return best_ridge


def _fit_selected_model(
    features: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    ridge_grid: Sequence[float],
) -> _RidgeModel:
    ridge = _select_ridge(
        features[train_mask],
        target[train_mask],
        features[validation_mask],
        target[validation_mask],
        ridge_grid,
    )
    fit_mask = train_mask | validation_mask
    return _fit_ridge(features[fit_mask], target[fit_mask], ridge)


def _residual_cloud_features(
    local: np.ndarray,
    cloud: np.ndarray,
    equation_variant: str,
) -> np.ndarray:
    """Fixed feature maps for exploratory equation transformations."""

    if equation_variant == "innovation":
        return cloud
    current = local[:, [0]]
    local_delta = local[:, [0]] - local[:, [1]]
    interaction = np.column_stack(
        [
            cloud,
            cloud * current,
            cloud * local_delta,
        ]
    )
    if equation_variant == "interaction":
        return interaction
    if equation_variant == "nonlinear":
        return np.column_stack(
            [
                interaction,
                np.square(cloud),
                np.tanh(cloud),
            ]
        )
    raise ValueError(f"unsupported residual equation variant: {equation_variant}")


def _fit_residual_cloud_model(
    residual_features: np.ndarray,
    local: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    local_ridge: float,
    ridge_grid: Sequence[float],
) -> _RidgeModel:
    """Select a cloud model for the innovation left by a train-only local fit."""

    local_train_model = _fit_ridge(local[train_mask], target[train_mask], local_ridge)
    train_residual = target[train_mask] - local_train_model.predict(local[train_mask])
    best_ridge = float(ridge_grid[0])
    best_error = float("inf")
    for ridge in ridge_grid:
        residual_model = _fit_ridge(
            residual_features[train_mask],
            train_residual,
            float(ridge),
        )
        prediction = local_train_model.predict(local[validation_mask])
        prediction += residual_model.predict(residual_features[validation_mask])
        error = float(np.mean(np.square(target[validation_mask] - prediction)))
        if error < best_error:
            best_error = error
            best_ridge = float(ridge)

    fit_mask = train_mask | validation_mask
    local_final = _fit_ridge(local[fit_mask], target[fit_mask], local_ridge)
    fit_residual = target[fit_mask] - local_final.predict(local[fit_mask])
    return _fit_ridge(residual_features[fit_mask], fit_residual, best_ridge)


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum(np.square(target - np.mean(target))))
    if denominator <= 1e-15:
        return 0.0
    return 1.0 - float(np.sum(np.square(target - prediction))) / denominator


def _population_latent(
    activity: np.ndarray,
    train_end: int,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    frames = activity.T
    train = frames[:train_end]
    finite_fraction = np.mean(np.isfinite(train), axis=0)
    keep = finite_fraction >= 0.5
    if not np.any(keep):
        raise ValueError("no population units have sufficient finite training data")
    frames = frames[:, keep]
    train = train[:, keep]
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    valid_frames = np.mean(np.isfinite(frames), axis=1) >= 0.5
    frames = np.where(np.isfinite(frames), frames, medians)
    train = np.where(np.isfinite(train), train, medians)
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    varying = scale > 1e-12
    if not np.any(varying):
        raise ValueError("population training activity is constant")
    frames_z = (frames[:, varying] - mean[varying]) / scale[varying]
    train_z = (train[:, varying] - mean[varying]) / scale[varying]
    _, _, right = np.linalg.svd(train_z, full_matrices=False)
    count = min(n_components, right.shape[0])
    components = right[:count].T
    return frames_z @ components, valid_frames


def _lagged_latent_state(
    latent: np.ndarray,
    indices: np.ndarray,
    state_order: int,
) -> np.ndarray:
    return np.column_stack(
        [latent[indices - lag] for lag in range(state_order)]
    )


def _diagonal_latent_prediction(
    state: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    test_mask: np.ndarray,
    n_components: int,
    state_order: int,
    ridge_grid: Sequence[float],
) -> np.ndarray:
    predictions = np.empty((int(np.sum(test_mask)), n_components), dtype=float)
    for component in range(n_components):
        columns = [
            lag * n_components + component for lag in range(state_order)
        ]
        features = state[:, columns]
        model = _fit_selected_model(
            features,
            target[:, component],
            train_mask,
            validation_mask,
            ridge_grid,
        )
        predictions[:, component] = model.predict(features[test_mask])
    return predictions


def _compose_latent_transition(
    model: _RidgeModel,
    initial_state: np.ndarray,
    n_components: int,
    state_order: int,
    horizon_steps: int,
) -> np.ndarray:
    state = initial_state.copy()
    prediction = state[:, :n_components]
    for _ in range(horizon_steps):
        prediction = model.predict(state)
        if state_order == 1:
            state = prediction
        else:
            state = np.column_stack(
                [
                    prediction,
                    state[:, : n_components * (state_order - 1)],
                ]
            )
    return prediction


def evaluate_latent_closure(
    recording: NeuralRecording,
    *,
    n_components: int = 8,
    state_order: int = 1,
    horizon_steps: int = 6,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    embargo: int = 5,
    max_gap_factor: float = 2.0,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    min_transition_delta: float = 0.01,
    max_composition_gap: float = 0.05,
) -> LatentClosureScore:
    """Test whether an ensemble latent is closed under repeated transition."""

    activity = np.asarray(recording.activity, dtype=float)
    time = np.asarray(recording.time, dtype=float).reshape(-1)
    if activity.ndim != 2 or activity.shape[1] != time.size:
        raise ValueError("activity must have shape (units, time)")
    if state_order < 1 or horizon_steps < 2:
        raise ValueError("state_order >= 1 and horizon_steps >= 2 are required")

    train_end = int(np.floor(train_fraction * time.size))
    validation_end = int(np.floor((train_fraction + validation_fraction) * time.size))
    latent, valid_frames = _population_latent(activity, train_end, n_components)
    component_count = latent.shape[1]
    positive_steps = np.diff(time)
    positive_steps = positive_steps[np.isfinite(positive_steps) & (positive_steps > 0.0)]
    if not positive_steps.size:
        raise ValueError("time must contain positive increments")
    max_gap = float(np.median(positive_steps) * max_gap_factor)

    def valid_indices(horizon: int) -> tuple[np.ndarray, np.ndarray]:
        indices = np.arange(state_order - 1, time.size - horizon)
        valid = np.ones(indices.size, dtype=bool)
        for offset in range(-(state_order - 1), horizon + 1):
            valid &= valid_frames[indices + offset]
        for offset in range(-(state_order - 1), horizon):
            step = time[indices + offset + 1] - time[indices + offset]
            valid &= np.isfinite(step) & (step > 0.0) & (step <= max_gap)
        return indices, valid

    one_indices, one_valid = valid_indices(1)
    one_state = _lagged_latent_state(latent, one_indices, state_order)
    one_target = latent[one_indices + 1]
    one_train = one_valid & (one_indices < train_end - embargo)
    one_validation = one_valid & (one_indices >= train_end + embargo) & (
        one_indices < validation_end - embargo
    )
    if min(np.sum(one_train), np.sum(one_validation)) < 20:
        raise ValueError("insufficient one-step train/validation samples")
    one_model = _fit_selected_model(
        one_state,
        one_target,
        one_train,
        one_validation,
        ridge_grid,
    )

    indices, valid = valid_indices(horizon_steps)
    state = _lagged_latent_state(latent, indices, state_order)
    target = latent[indices + horizon_steps]
    train_mask = valid & (indices < train_end - embargo - horizon_steps)
    validation_mask = valid & (indices >= train_end + embargo) & (
        indices < validation_end - embargo - horizon_steps
    )
    test_mask = valid & (indices >= validation_end + embargo)
    if min(np.sum(train_mask), np.sum(validation_mask), np.sum(test_mask)) < 20:
        raise ValueError("insufficient direct-horizon train/validation/test samples")

    direct_model = _fit_selected_model(
        state,
        target,
        train_mask,
        validation_mask,
        ridge_grid,
    )
    direct_prediction = direct_model.predict(state[test_mask])
    diagonal_prediction = _diagonal_latent_prediction(
        state,
        target,
        train_mask,
        validation_mask,
        test_mask,
        component_count,
        state_order,
        ridge_grid,
    )
    composed_prediction = _compose_latent_transition(
        one_model,
        state[test_mask],
        component_count,
        state_order,
        horizon_steps,
    )
    persistence_prediction = state[test_mask, :component_count]
    test_target = target[test_mask]

    return LatentClosureScore(
        recording_id=recording.recording_id,
        n_units=int(activity.shape[0]),
        n_timepoints=int(activity.shape[1]),
        n_components=component_count,
        state_order=state_order,
        horizon_steps=horizon_steps,
        n_train=int(np.sum(train_mask)),
        n_validation=int(np.sum(validation_mask)),
        n_test=int(np.sum(test_mask)),
        r2_persistence=_r2(test_target, persistence_prediction),
        r2_diagonal=_r2(test_target, diagonal_prediction),
        r2_direct=_r2(test_target, direct_prediction),
        r2_composed=_r2(test_target, composed_prediction),
        min_transition_delta=min_transition_delta,
        max_composition_gap=max_composition_gap,
    )


def _cloud_features(
    activity: np.ndarray,
    target_index: int,
    sample_indices: np.ndarray,
    train_mask: np.ndarray,
    n_components: int,
) -> np.ndarray:
    other_indices = np.delete(np.arange(activity.shape[0]), target_index)
    current = activity[other_indices][:, sample_indices].T
    previous = activity[other_indices][:, sample_indices - 1].T

    train_stack = np.vstack([current[train_mask], previous[train_mask]])
    finite_fraction = np.mean(np.isfinite(train_stack), axis=0)
    keep = finite_fraction >= 0.5
    if not np.any(keep):
        raise ValueError("no population units have sufficient finite training data")

    current = current[:, keep]
    previous = previous[:, keep]
    train_stack = train_stack[:, keep]
    medians = np.nanmedian(train_stack, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)

    current = np.where(np.isfinite(current), current, medians)
    previous = np.where(np.isfinite(previous), previous, medians)
    train_stack = np.where(np.isfinite(train_stack), train_stack, medians)

    mean = np.mean(train_stack, axis=0)
    scale = np.std(train_stack, axis=0)
    varying = scale > 1e-12
    if not np.any(varying):
        raise ValueError("population training activity is constant")

    mean = mean[varying]
    scale = scale[varying]
    current_z = (current[:, varying] - mean) / scale
    previous_z = (previous[:, varying] - mean) / scale
    train_z = (train_stack[:, varying] - mean) / scale
    _, _, right = np.linalg.svd(train_z, full_matrices=False)
    count = min(n_components, right.shape[0])
    components = right[:count].T
    return np.column_stack([current_z @ components, previous_z @ components])


def evaluate_recording_dynamics(
    recording: NeuralRecording,
    *,
    n_components: int = 8,
    horizon_steps: int = 1,
    equation_variant: str = "additive",
    max_targets: int | None = None,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    embargo: int = 5,
    max_gap_factor: float = 2.0,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    min_cloud_delta: float = 0.001,
    min_memory_delta: float = 0.001,
    min_positive_fraction: float = 0.6,
) -> RecordingDynamicsGate:
    """Evaluate local-history and leave-target-out cloud ablations.

    Splits are chronological.  Imputation, standardization, PCA, and ridge
    selection see no test observations.  A circularly shifted test cloud is
    retained as an autocorrelation-preserving temporal-alignment control.
    """

    activity = np.asarray(recording.activity, dtype=float)
    time = np.asarray(recording.time, dtype=float).reshape(-1)
    if activity.ndim != 2:
        raise ValueError("activity must have shape (units, time)")
    if activity.shape[1] != time.size:
        raise ValueError("activity time dimension must match time")
    if activity.shape[0] < 3 or time.size < 80:
        raise ValueError("at least 3 units and 80 timepoints are required")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    if equation_variant not in EQUATION_VARIANTS:
        raise ValueError(f"equation_variant must be one of {EQUATION_VARIANTS}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 < validation_fraction < 1.0 - train_fraction:
        raise ValueError("validation_fraction leaves no test block")

    positive_steps = np.diff(time)
    positive_steps = positive_steps[np.isfinite(positive_steps) & (positive_steps > 0.0)]
    if not positive_steps.size:
        raise ValueError("time must contain positive increments")
    max_gap = float(np.median(positive_steps) * max_gap_factor)

    sample_indices = np.arange(2, time.size - horizon_steps)
    local_steps = np.column_stack(
        [
            time[sample_indices + offset + 1] - time[sample_indices + offset]
            for offset in range(-2, horizon_steps)
        ]
    )
    continuous = np.all(
        np.isfinite(local_steps) & (local_steps > 0.0) & (local_steps <= max_gap),
        axis=1,
    )

    train_end = int(np.floor(train_fraction * time.size))
    validation_end = int(np.floor((train_fraction + validation_fraction) * time.size))
    base_train = sample_indices < train_end - embargo
    base_validation = (sample_indices >= train_end + embargo) & (
        sample_indices < validation_end - embargo
    )
    base_test = sample_indices >= validation_end + embargo

    finite_fraction = np.mean(np.isfinite(activity), axis=1)
    variance = np.nanvar(activity[:, : max(train_end, 1)], axis=1)
    eligible = np.flatnonzero((finite_fraction >= 0.5) & (variance > 1e-12))
    if max_targets is not None and eligible.size > max_targets:
        positions = np.linspace(0, eligible.size - 1, max_targets, dtype=int)
        eligible = eligible[positions]

    scores: list[TargetDynamicsScore] = []
    for target_index in eligible:
        target_series = activity[target_index]
        local = np.column_stack(
            [
                target_series[sample_indices],
                target_series[sample_indices - 1],
                target_series[sample_indices - 2],
            ]
        )
        target = target_series[sample_indices + horizon_steps]
        finite_target = np.isfinite(target) & np.all(np.isfinite(local), axis=1)
        valid = continuous & finite_target
        train_mask = valid & base_train
        validation_mask = valid & base_validation
        test_mask = valid & base_test
        if min(np.sum(train_mask), np.sum(validation_mask), np.sum(test_mask)) < 20:
            continue

        try:
            cloud = _cloud_features(
                activity,
                int(target_index),
                sample_indices,
                train_mask,
                n_components,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        full = np.column_stack([local, cloud])

        current_model = _fit_selected_model(
            local[:, :1], target, train_mask, validation_mask, ridge_grid
        )
        local_model = _fit_selected_model(
            local, target, train_mask, validation_mask, ridge_grid
        )
        cloud_model = _fit_selected_model(
            cloud, target, train_mask, validation_mask, ridge_grid
        )
        if equation_variant == "additive":
            full_features = full
            full_model = _fit_selected_model(
                full_features, target, train_mask, validation_mask, ridge_grid
            )
            residual_features = None
        else:
            residual_features = _residual_cloud_features(
                local,
                cloud,
                equation_variant,
            )
            full_features = residual_features
            full_model = _fit_residual_cloud_model(
                residual_features,
                local,
                target,
                train_mask,
                validation_mask,
                local_model.ridge,
                ridge_grid,
            )

        test_target = target[test_mask]
        fit_mask = train_mask | validation_mask
        intercept_prediction = np.full(
            test_target.shape, float(np.mean(target[fit_mask])), dtype=float
        )
        current_prediction = current_model.predict(local[test_mask, :1])
        local_prediction = local_model.predict(local[test_mask])
        cloud_prediction = cloud_model.predict(cloud[test_mask])
        if equation_variant == "additive":
            full_prediction = full_model.predict(full[test_mask])
        else:
            full_prediction = local_model.predict(local[test_mask])
            full_prediction += full_model.predict(full_features[test_mask])

        test_local = local[test_mask]
        test_cloud = cloud[test_mask]
        shift = max(1, len(test_cloud) // 3)
        shifted_cloud = np.roll(test_cloud, shift, axis=0)
        if equation_variant == "additive":
            shifted_full = np.column_stack([test_local, shifted_cloud])
            shifted_prediction = full_model.predict(shifted_full)
        else:
            shifted_features = _residual_cloud_features(
                test_local,
                shifted_cloud,
                equation_variant,
            )
            shifted_prediction = local_model.predict(test_local)
            shifted_prediction += full_model.predict(shifted_features)

        scores.append(
            TargetDynamicsScore(
                target_index=int(target_index),
                n_train=int(np.sum(train_mask)),
                n_validation=int(np.sum(validation_mask)),
                n_test=int(np.sum(test_mask)),
                r2_intercept=_r2(test_target, intercept_prediction),
                r2_current=_r2(test_target, current_prediction),
                r2_local=_r2(test_target, local_prediction),
                r2_cloud=_r2(test_target, cloud_prediction),
                r2_full=_r2(test_target, full_prediction),
                r2_shifted_cloud=_r2(test_target, shifted_prediction),
                ridge_current=current_model.ridge,
                ridge_local=local_model.ridge,
                ridge_cloud=cloud_model.ridge,
                ridge_full=full_model.ridge,
            )
        )

    return RecordingDynamicsGate(
        recording_id=recording.recording_id,
        n_units=int(activity.shape[0]),
        n_timepoints=int(activity.shape[1]),
        horizon_steps=horizon_steps,
        equation_variant=equation_variant,
        scores=tuple(scores),
        min_cloud_delta=min_cloud_delta,
        min_memory_delta=min_memory_delta,
        min_positive_fraction=min_positive_fraction,
    )


def evaluate_dynamics_panel(
    recordings: Sequence[NeuralRecording],
    *,
    min_recordings_passed: int = 3,
    **gate_options: object,
) -> DynamicsPanelGate:
    gates = tuple(
        evaluate_recording_dynamics(recording, **gate_options) for recording in recordings
    )
    return DynamicsPanelGate(gates, min_recordings_passed=min_recordings_passed)


def load_predictioncode_recordings(
    root: str | Path,
    *,
    signal_field: str = "Ratio2",
) -> tuple[NeuralRecording, ...]:
    """Load the selectively extracted AML310/AKS297.51 PredictionCode archive."""

    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise RuntimeError("SciPy is required to load PredictionCode MATLAB files") from error

    root_path = Path(root)
    logs = sorted(root_path.glob("*_datasets.txt"))
    if len(logs) != 1:
        raise ValueError(f"expected exactly one *_datasets.txt under {root_path}")

    recordings: list[NeuralRecording] = []
    for line in logs[0].read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        recording_id = fields[0]
        cut_volume = int(fields[1]) if len(fields) > 1 else None
        mat_path = root_path / f"{recording_id}_MS" / "heatDataMS.mat"
        if not mat_path.is_file():
            raise FileNotFoundError(mat_path)
        data = loadmat(
            mat_path,
            variable_names=(signal_field, "hasPointsTime"),
            simplify_cells=True,
        )
        if signal_field not in data or "hasPointsTime" not in data:
            raise KeyError(f"{mat_path} lacks {signal_field!r} or hasPointsTime")
        activity = np.asarray(data[signal_field], dtype=float)
        time = np.asarray(data["hasPointsTime"], dtype=float).reshape(-1)
        stop = time.size if cut_volume is None else min(time.size, cut_volume + 1)
        activity = activity[:, :stop].copy()
        time = time[:stop].copy()
        for start, end in AML310_EXCLUDED_INTERVALS.get(recording_id, ()):
            activity[:, (time >= start) & (time <= end)] = np.nan
        recordings.append(NeuralRecording(recording_id, time, activity))
    return tuple(recordings)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dynamics_artifact(
    panel: DynamicsPanelGate,
    *,
    source_url: str,
    archive_path: str | Path,
    expected_sha256: str | None,
    include_targets: bool = True,
) -> dict[str, object]:
    archive = Path(archive_path)
    observed_sha256 = sha256_file(archive)
    integrity_passed = expected_sha256 is None or observed_sha256 == expected_sha256.lower()
    return {
        "artifact_type": "clarus_cloudcell_dynamics_gate",
        "artifact_version": 1,
        "source": {
            "url": source_url,
            "archive_name": archive.name,
            "archive_bytes": archive.stat().st_size,
            "sha256": observed_sha256,
            "expected_sha256": expected_sha256,
            "integrity_passed": integrity_passed,
        },
        "method": {
            "target": (
                f"x_i(t+{panel.recordings[0].horizon_steps})"
                if panel.recordings
                else "x_i(t+h)"
            ),
            "equation_variant": (
                panel.recordings[0].equation_variant if panel.recordings else "unknown"
            ),
            "local": "[x_i(t), x_i(t-1), x_i(t-2)]",
            "current_state_baseline": "x_i(t)",
            "cloud": "train-only PCA of all non-target units at t and t-1",
            "split": "chronological 60/20/20 with five-sample embargo",
            "selection": "ridge selected on validation; test untouched",
            "null": "circularly shifted held-out cloud",
            "replicate_unit": "recording/animal, never neuron",
        },
        "gate_passed": bool(integrity_passed and panel.passed),
        "claim_scope": (
            "observational support for a local-state plus population-context dynamics "
            "bridge; not proof that a biological neuron is a monad"
        ),
        "result": panel.to_dict(include_targets=include_targets),
    }


def build_latent_closure_artifact(
    scores: Sequence[LatentClosureScore],
    *,
    source_url: str,
    archive_path: str | Path,
    expected_sha256: str | None,
) -> dict[str, object]:
    archive = Path(archive_path)
    observed_sha256 = sha256_file(archive)
    integrity_passed = expected_sha256 is None or observed_sha256 == expected_sha256.lower()
    pass_count = sum(score.passed for score in scores)
    return {
        "artifact_type": "clarus_cloudcell_latent_closure_gate",
        "artifact_version": 1,
        "source": {
            "url": source_url,
            "archive_name": archive.name,
            "archive_bytes": archive.stat().st_size,
            "sha256": observed_sha256,
            "expected_sha256": expected_sha256,
            "integrity_passed": integrity_passed,
        },
        "method": {
            "state": "train-only population PCA latent with fixed lag order",
            "transition": "ridge affine map",
            "composition": "repeat the held-out one-step map h times",
            "coupling_control": "per-latent-axis diagonal autoregression",
            "status": "exploratory refinement after per-neuron cloud gate failure",
        },
        "gate_passed": bool(integrity_passed and pass_count >= 3),
        "result": {
            "recording_count": len(scores),
            "recordings_passed": pass_count,
            "min_recordings_passed": 3,
            "recordings": [score.to_dict() for score in scores],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Extracted AKS297.51_moving directory")
    parser.add_argument("--archive", required=True, help="Downloaded source archive")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output")
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--horizon-steps", type=int, default=1)
    parser.add_argument("--variant", choices=EQUATION_VARIANTS, default="additive")
    parser.add_argument("--latent-closure", action="store_true")
    parser.add_argument("--state-order", type=int, default=1)
    parser.add_argument("--min-recordings-passed", type=int, default=3)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    recordings = load_predictioncode_recordings(args.root)
    if args.latent_closure:
        scores = tuple(
            evaluate_latent_closure(
                recording,
                n_components=args.components,
                state_order=args.state_order,
                horizon_steps=args.horizon_steps,
            )
            for recording in recordings
        )
        artifact = build_latent_closure_artifact(
            scores,
            source_url="https://osf.io/download/evhrg/",
            archive_path=args.archive,
            expected_sha256=args.expected_sha256,
        )
    else:
        panel = evaluate_dynamics_panel(
            recordings,
            min_recordings_passed=args.min_recordings_passed,
            n_components=args.components,
            horizon_steps=args.horizon_steps,
            equation_variant=args.variant,
            max_targets=args.max_targets,
        )
        artifact = build_dynamics_artifact(
            panel,
            source_url="https://osf.io/download/evhrg/",
            archive_path=args.archive,
            expected_sha256=args.expected_sha256,
            include_targets=not args.summary_only,
        )
    payload = json.dumps(artifact, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    if not args.quiet:
        print(payload)
    return int(args.require_pass and not artifact["gate_passed"])


__all__ = [
    "AML310_EXCLUDED_INTERVALS",
    "DynamicsPanelGate",
    "EQUATION_VARIANTS",
    "LatentClosureScore",
    "NeuralRecording",
    "RecordingDynamicsGate",
    "TargetDynamicsScore",
    "build_dynamics_artifact",
    "build_latent_closure_artifact",
    "evaluate_dynamics_panel",
    "evaluate_latent_closure",
    "evaluate_recording_dynamics",
    "load_predictioncode_recordings",
    "main",
    "sha256_file",
]


if __name__ == "__main__":
    raise SystemExit(main())
