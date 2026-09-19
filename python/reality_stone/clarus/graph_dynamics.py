"""Leakage-resistant graph residual gates for population neural dynamics.

The first graph loop is deliberately linear and nested:

``G0``
    local target history only;
``G1``
    local history plus a symmetric non-negative diffusion message learned
    without the future target;
``G2``
    local history plus a sparse signed directed message learned from the
    one-step innovation left by the local model.

The directed graph is an *effective predictive graph*, not an anatomical
connectome.  It is fitted from the training block only.  Node-permuted graphs
preserve the learned weighted graph while breaking its alignment to recorded
units, providing a whole-pipeline graph null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .cloudcell_dynamics import (
    NeuralRecording,
    load_predictioncode_recordings,
    sha256_file,
)


RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class _RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    ridge: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = (features - self.mean) / self.scale
        design = np.column_stack((np.ones(len(standardized)), standardized))
        return design @ self.coefficients


@dataclass(frozen=True)
class GraphTargetScore:
    """Held-out comparison for one recorded target unit."""

    target_index: int
    n_train: int
    n_validation: int
    n_test: int
    r2_local: float
    r2_diffusion: float
    r2_directed: float
    r2_rewired: tuple[float, ...]
    ridge_local: float
    ridge_diffusion: float
    ridge_directed: float

    @property
    def delta_diffusion_over_local(self) -> float:
        return self.r2_diffusion - self.r2_local

    @property
    def delta_directed_over_local(self) -> float:
        return self.r2_directed - self.r2_local

    @property
    def delta_directed_over_diffusion(self) -> float:
        return self.r2_directed - self.r2_diffusion

    def to_dict(self, *, include_null: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "target_index": self.target_index,
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "n_test": self.n_test,
            "r2_local": self.r2_local,
            "r2_diffusion": self.r2_diffusion,
            "r2_directed": self.r2_directed,
            "delta_diffusion_over_local": self.delta_diffusion_over_local,
            "delta_directed_over_local": self.delta_directed_over_local,
            "delta_directed_over_diffusion": self.delta_directed_over_diffusion,
            "ridge_local": self.ridge_local,
            "ridge_diffusion": self.ridge_diffusion,
            "ridge_directed": self.ridge_directed,
        }
        if include_null:
            result["r2_rewired"] = list(self.r2_rewired)
        return result


@dataclass(frozen=True)
class GraphRecordingGate:
    """Recording-level graph gate; the animal is the replicate."""

    recording_id: str
    n_units: int
    n_timepoints: int
    horizon_steps: int
    graph_learn_horizon: int
    neighbor_count: int
    graph_feature_mode: str
    graph_regimes: int
    scores: tuple[GraphTargetScore, ...]
    rewired_median_deltas: tuple[float, ...]
    adjacency_density: float
    adjacency_spectral_radius: float
    adjacency_sha256: str
    min_graph_delta: float = 0.001
    min_graph_over_diffusion: float = 0.0
    min_positive_fraction: float = 0.6
    max_rewired_p: float = 0.05

    def _values(self, name: str) -> np.ndarray:
        return np.asarray([getattr(score, name) for score in self.scores], dtype=float)

    def median(self, name: str) -> float:
        values = self._values(name)
        return float(np.median(values)) if values.size else float("nan")

    def positive_fraction(self, name: str) -> float:
        values = self._values(name)
        return float(np.mean(values > 0.0)) if values.size else 0.0

    @property
    def rewired_p_value(self) -> float:
        null = np.asarray(self.rewired_median_deltas, dtype=float)
        if not null.size:
            return 1.0
        observed = self.median("delta_directed_over_local")
        return float((1 + np.sum(null >= observed)) / (null.size + 1))

    @property
    def passed(self) -> bool:
        return (
            len(self.scores) >= 3
            and self.median("delta_directed_over_local") > self.min_graph_delta
            and self.median("delta_directed_over_diffusion")
            > self.min_graph_over_diffusion
            and self.positive_fraction("delta_directed_over_local")
            >= self.min_positive_fraction
            and self.rewired_p_value <= self.max_rewired_p
            and self.adjacency_spectral_radius <= 1.0 + 1e-8
        )

    def to_dict(self, *, include_targets: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "recording_id": self.recording_id,
            "n_units": self.n_units,
            "n_timepoints": self.n_timepoints,
            "horizon_steps": self.horizon_steps,
            "graph_learn_horizon": self.graph_learn_horizon,
            "neighbor_count": self.neighbor_count,
            "graph_feature_mode": self.graph_feature_mode,
            "graph_regimes": self.graph_regimes,
            "n_targets_evaluated": len(self.scores),
            "median_r2_local": self.median("r2_local"),
            "median_r2_diffusion": self.median("r2_diffusion"),
            "median_r2_directed": self.median("r2_directed"),
            "median_delta_diffusion_over_local": self.median(
                "delta_diffusion_over_local"
            ),
            "median_delta_directed_over_local": self.median(
                "delta_directed_over_local"
            ),
            "median_delta_directed_over_diffusion": self.median(
                "delta_directed_over_diffusion"
            ),
            "positive_fraction_directed_over_local": self.positive_fraction(
                "delta_directed_over_local"
            ),
            "rewired_median_deltas": list(self.rewired_median_deltas),
            "rewired_p_value": self.rewired_p_value,
            "adjacency_density": self.adjacency_density,
            "adjacency_spectral_radius": self.adjacency_spectral_radius,
            "adjacency_sha256": self.adjacency_sha256,
            "criteria": {
                "min_graph_delta": self.min_graph_delta,
                "min_graph_over_diffusion": self.min_graph_over_diffusion,
                "min_positive_fraction": self.min_positive_fraction,
                "max_rewired_p": self.max_rewired_p,
                "spectral_radius_upper_bound": 1.0,
            },
            "passed": self.passed,
        }
        if include_targets:
            result["targets"] = [score.to_dict() for score in self.scores]
        return result


@dataclass(frozen=True)
class GraphPanelGate:
    recordings: tuple[GraphRecordingGate, ...]
    min_recordings_passed: int

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
            "replicate_unit": "independently recorded animal, never target unit",
            "recordings": [
                recording.to_dict(include_targets=include_targets)
                for recording in self.recordings
            ],
        }


def _fit_ridge(features: np.ndarray, target: np.ndarray, ridge: float) -> _RidgeModel:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (features - mean) / scale
    design = np.column_stack((np.ones(len(standardized)), standardized))
    penalty = np.eye(design.shape[1], dtype=float) * float(ridge)
    penalty[0, 0] = 0.0
    left = design.T @ design + penalty
    right = design.T @ target
    try:
        coefficients = np.linalg.solve(left, right)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(left) @ right
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
    for candidate in ridge_grid:
        model = _fit_ridge(train_x, train_y, float(candidate))
        error = float(np.mean(np.square(validation_y - model.predict(validation_x))))
        if error < best_error:
            best_error = error
            best_ridge = float(candidate)
    return best_ridge


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum(np.square(target - np.mean(target))))
    if denominator <= 1e-15:
        return 0.0
    return 1.0 - float(np.sum(np.square(target - prediction))) / denominator


def _continuity_mask(
    time: np.ndarray,
    sample_indices: np.ndarray,
    horizon: int,
    max_gap: float,
) -> np.ndarray:
    steps = np.column_stack(
        [
            time[sample_indices + offset + 1] - time[sample_indices + offset]
            for offset in range(-2, horizon)
        ]
    )
    return np.all(
        np.isfinite(steps) & (steps > 0.0) & (steps <= max_gap),
        axis=1,
    )


def _preprocess_activity(
    activity: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    finite_fraction = np.mean(np.isfinite(activity[:, :train_end]), axis=1)
    variance = np.nanvar(activity[:, :train_end], axis=1)
    eligible = np.flatnonzero((finite_fraction >= 0.5) & (variance > 1e-12))
    if eligible.size < 3:
        raise ValueError("fewer than three units have finite variable training activity")
    medians = np.nanmedian(activity[:, :train_end], axis=1)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(activity), activity, medians[:, None])
    mean = np.mean(filled[:, :train_end], axis=1)
    scale = np.std(filled[:, :train_end], axis=1)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (filled - mean[:, None]) / scale[:, None], eligible


def _top_k(values: np.ndarray, candidates: np.ndarray, count: int) -> np.ndarray:
    if candidates.size <= count:
        return candidates
    order = np.argsort(-np.abs(values[candidates]), kind="stable")
    return candidates[order[:count]]


def _directed_adjacency(
    standardized: np.ndarray,
    original: np.ndarray,
    eligible: np.ndarray,
    sample_indices: np.ndarray,
    base_mask: np.ndarray,
    continuous: np.ndarray,
    neighbor_count: int,
    regime_by_frame: np.ndarray | None = None,
    regime_value: int = 0,
) -> np.ndarray:
    """Learn signed source-to-target rows from one-step local innovations."""

    n_units = standardized.shape[0]
    adjacency = np.zeros((n_units, n_units), dtype=float)
    for target_index in eligible:
        local = np.column_stack(
            [
                standardized[target_index, sample_indices],
                standardized[target_index, sample_indices - 1],
                standardized[target_index, sample_indices - 2],
            ]
        )
        target = standardized[target_index, sample_indices + 1]
        original_target = original[target_index]
        finite = np.all(
            np.column_stack(
                [
                    np.isfinite(original_target[sample_indices]),
                    np.isfinite(original_target[sample_indices - 1]),
                    np.isfinite(original_target[sample_indices - 2]),
                    np.isfinite(original_target[sample_indices + 1]),
                ]
            ),
            axis=1,
        )
        mask = base_mask & continuous & finite
        if regime_by_frame is not None:
            mask &= regime_by_frame[sample_indices] == regime_value
        if np.sum(mask) < 20:
            continue
        local_model = _fit_ridge(local[mask], target[mask], ridge=1.0)
        residual = target[mask] - local_model.predict(local[mask])
        residual = residual - np.mean(residual)
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= 1e-12:
            continue
        source_values = standardized[:, sample_indices[mask]]
        source_values = source_values - np.mean(source_values, axis=1, keepdims=True)
        source_norm = np.linalg.norm(source_values, axis=1)
        denominator = np.maximum(source_norm * residual_norm, 1e-12)
        correlations = (source_values @ residual) / denominator
        correlations[target_index] = 0.0
        candidates = eligible[eligible != target_index]
        selected = _top_k(correlations, candidates, neighbor_count)
        weights = correlations[selected]
        norm = float(np.sum(np.abs(weights)))
        if norm > 1e-12:
            adjacency[target_index, selected] = weights / norm
    return adjacency


def _diffusion_adjacency(
    standardized: np.ndarray,
    eligible: np.ndarray,
    frame_mask: np.ndarray,
    neighbor_count: int,
) -> np.ndarray:
    """Learn a symmetric non-negative correlation graph without future targets."""

    n_units = standardized.shape[0]
    selected_activity = standardized[eligible][:, frame_mask]
    correlation = np.corrcoef(selected_activity)
    correlation = np.nan_to_num(np.abs(correlation), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 0.0)
    sparse = np.zeros_like(correlation)
    local_indices = np.arange(eligible.size)
    for row in local_indices:
        candidates = local_indices[local_indices != row]
        selected = _top_k(correlation[row], candidates, neighbor_count)
        sparse[row, selected] = correlation[row, selected]
    sparse = np.maximum(sparse, sparse.T)
    row_sum = np.sum(sparse, axis=1, keepdims=True)
    sparse = np.divide(sparse, row_sum, out=np.zeros_like(sparse), where=row_sum > 0.0)
    adjacency = np.zeros((n_units, n_units), dtype=float)
    adjacency[np.ix_(eligible, eligible)] = sparse
    return adjacency


def _population_regime(
    standardized: np.ndarray,
    eligible: np.ndarray,
    frame_mask: np.ndarray,
) -> np.ndarray:
    """Return a causal two-state label from a train-fitted population PC1."""

    frames = standardized[eligible].T
    training = frames[frame_mask]
    _, _, right = np.linalg.svd(training, full_matrices=False)
    component = right[0]
    anchor = int(np.argmax(np.abs(component)))
    if component[anchor] < 0.0:
        component = -component
    scores = frames @ component
    threshold = float(np.median(scores[frame_mask]))
    return (scores >= threshold).astype(np.int8)


def _message_features(
    standardized: np.ndarray,
    adjacency: np.ndarray,
    sample_indices: np.ndarray,
    target_index: int,
    *,
    diffusion: bool,
) -> np.ndarray:
    row = adjacency[target_index]
    degree = float(np.sum(row))
    columns = []
    for lag in range(3):
        values = standardized[:, sample_indices - lag].T
        message = values @ row
        if diffusion:
            message -= degree * standardized[target_index, sample_indices - lag]
        columns.append(message)
    return np.column_stack(columns)


def _sparse_source_features(
    standardized: np.ndarray,
    adjacency: np.ndarray,
    sample_indices: np.ndarray,
    target_index: int,
    neighbor_count: int,
) -> np.ndarray:
    """Keep top directed sources separate instead of summing their signals."""

    row = adjacency[target_index]
    selected = np.flatnonzero(np.abs(row) > 0.0)
    if selected.size:
        order = np.argsort(-np.abs(row[selected]), kind="stable")
        selected = selected[order[:neighbor_count]]
    features = np.zeros((sample_indices.size, neighbor_count), dtype=float)
    if selected.size:
        features[:, : selected.size] = standardized[selected][:, sample_indices].T
    return features


def _dynamic_graph_features(
    standardized: np.ndarray,
    adjacencies: tuple[np.ndarray, ...],
    regime_by_frame: np.ndarray,
    sample_indices: np.ndarray,
    target_index: int,
    neighbor_count: int,
    graph_feature_mode: str,
) -> np.ndarray:
    """Select the graph row associated with the currently observed regime."""

    per_regime = []
    for adjacency in adjacencies:
        if graph_feature_mode == "aggregate":
            features = _message_features(
                standardized,
                adjacency,
                sample_indices,
                target_index,
                diffusion=False,
            )
        else:
            features = _sparse_source_features(
                standardized,
                adjacency,
                sample_indices,
                target_index,
                neighbor_count,
            )
        per_regime.append(features)
    result = np.zeros_like(per_regime[0])
    current_regimes = regime_by_frame[sample_indices]
    for regime, features in enumerate(per_regime):
        mask = current_regimes == regime
        result[mask] = features[mask]
    return result


def _node_permutation(adjacency: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    return adjacency[np.ix_(permutation, permutation)]


def _fit_nested_message_model(
    local: np.ndarray,
    target: np.ndarray,
    train_message: np.ndarray,
    fit_message: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    test_mask: np.ndarray,
    ridge_grid: Sequence[float],
) -> tuple[float, float]:
    train_features = np.column_stack((local, train_message))
    ridge = _select_ridge(
        train_features[train_mask],
        target[train_mask],
        train_features[validation_mask],
        target[validation_mask],
        ridge_grid,
    )
    fit_features = np.column_stack((local, fit_message))
    fit_mask = train_mask | validation_mask
    model = _fit_ridge(fit_features[fit_mask], target[fit_mask], ridge)
    prediction = model.predict(fit_features[test_mask])
    return _r2(target[test_mask], prediction), ridge


def evaluate_graph_recording(
    recording: NeuralRecording,
    *,
    horizon_steps: int = 1,
    graph_learn_horizon: int = 1,
    neighbor_count: int = 4,
    graph_feature_mode: str = "aggregate",
    graph_regimes: int = 1,
    n_rewired: int = 19,
    random_seed: int = 1729,
    max_targets: int | None = None,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    embargo: int = 5,
    max_gap_factor: float = 2.0,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    min_graph_delta: float = 0.001,
    min_graph_over_diffusion: float = 0.0,
    min_positive_fraction: float = 0.6,
    max_rewired_p: float = 0.05,
) -> GraphRecordingGate:
    """Evaluate local, diffusion, directed, and node-permuted graph models."""

    activity = np.asarray(recording.activity, dtype=float)
    time = np.asarray(recording.time, dtype=float).reshape(-1)
    if activity.ndim != 2 or activity.shape[1] != time.size:
        raise ValueError("recording activity must be units x time and match time")
    if activity.shape[0] < 3 or time.size < 100:
        raise ValueError("at least three units and 100 timepoints are required")
    if horizon_steps < 1 or graph_learn_horizon != 1:
        raise ValueError("positive horizon and graph_learn_horizon=1 are required")
    if neighbor_count < 1 or n_rewired < 1:
        raise ValueError("neighbor_count and n_rewired must be positive")
    if graph_feature_mode not in {"aggregate", "sparse_var"}:
        raise ValueError("graph_feature_mode must be 'aggregate' or 'sparse_var'")
    if graph_regimes not in {1, 2}:
        raise ValueError("graph_regimes must be 1 or 2")

    train_end = int(np.floor(train_fraction * time.size))
    validation_end = int(
        np.floor((train_fraction + validation_fraction) * time.size)
    )
    if not 0 < train_end < validation_end < time.size:
        raise ValueError("chronological fractions leave an empty split")

    positive_steps = np.diff(time)
    positive_steps = positive_steps[np.isfinite(positive_steps) & (positive_steps > 0.0)]
    if not positive_steps.size:
        raise ValueError("time must contain positive increments")
    max_gap = float(np.median(positive_steps) * max_gap_factor)
    standardized, eligible = _preprocess_activity(activity, train_end)

    learn_indices = np.arange(2, time.size - graph_learn_horizon)
    learn_continuous = _continuity_mask(
        time,
        learn_indices,
        graph_learn_horizon,
        max_gap,
    )
    learn_train = learn_indices < train_end - embargo
    learn_fit = learn_indices < validation_end - embargo
    train_frames = np.arange(time.size) < train_end - embargo
    fit_frames = np.arange(time.size) < validation_end - embargo
    if graph_regimes == 1:
        regime_train = np.zeros(time.size, dtype=np.int8)
        regime_fit = np.zeros(time.size, dtype=np.int8)
    else:
        regime_train = _population_regime(standardized, eligible, train_frames)
        regime_fit = _population_regime(standardized, eligible, fit_frames)
    directed_train = tuple(
        _directed_adjacency(
            standardized,
            activity,
            eligible,
            learn_indices,
            learn_train,
            learn_continuous,
            neighbor_count,
            regime_by_frame=regime_train if graph_regimes > 1 else None,
            regime_value=regime,
        )
        for regime in range(graph_regimes)
    )
    directed_fit = tuple(
        _directed_adjacency(
            standardized,
            activity,
            eligible,
            learn_indices,
            learn_fit,
            learn_continuous,
            neighbor_count,
            regime_by_frame=regime_fit if graph_regimes > 1 else None,
            regime_value=regime,
        )
        for regime in range(graph_regimes)
    )
    diffusion_train = _diffusion_adjacency(
        standardized,
        eligible,
        train_frames,
        neighbor_count,
    )
    diffusion_fit = _diffusion_adjacency(
        standardized,
        eligible,
        fit_frames,
        neighbor_count,
    )

    rng = np.random.default_rng(random_seed)
    permutations = tuple(rng.permutation(activity.shape[0]) for _ in range(n_rewired))
    directed_train_null = tuple(
        tuple(_node_permutation(adjacency, permutation) for adjacency in directed_train)
        for permutation in permutations
    )
    directed_fit_null = tuple(
        tuple(_node_permutation(adjacency, permutation) for adjacency in directed_fit)
        for permutation in permutations
    )

    sample_indices = np.arange(2, time.size - horizon_steps)
    continuous = _continuity_mask(time, sample_indices, horizon_steps, max_gap)
    base_train = sample_indices < train_end - embargo
    base_validation = (sample_indices >= train_end + embargo) & (
        sample_indices < validation_end - embargo
    )
    base_test = sample_indices >= validation_end + embargo

    selected_targets = eligible
    if max_targets is not None and selected_targets.size > max_targets:
        positions = np.linspace(0, selected_targets.size - 1, max_targets, dtype=int)
        selected_targets = selected_targets[positions]

    scores = []
    for target_index in selected_targets:
        local = np.column_stack(
            [
                standardized[target_index, sample_indices],
                standardized[target_index, sample_indices - 1],
                standardized[target_index, sample_indices - 2],
            ]
        )
        target = standardized[target_index, sample_indices + horizon_steps]
        original_target = activity[target_index]
        finite = np.all(
            np.column_stack(
                [
                    np.isfinite(original_target[sample_indices]),
                    np.isfinite(original_target[sample_indices - 1]),
                    np.isfinite(original_target[sample_indices - 2]),
                    np.isfinite(original_target[sample_indices + horizon_steps]),
                ]
            ),
            axis=1,
        )
        valid = continuous & finite
        train_mask = valid & base_train
        validation_mask = valid & base_validation
        test_mask = valid & base_test
        if min(np.sum(train_mask), np.sum(validation_mask), np.sum(test_mask)) < 20:
            continue

        local_ridge = _select_ridge(
            local[train_mask],
            target[train_mask],
            local[validation_mask],
            target[validation_mask],
            ridge_grid,
        )
        fit_mask = train_mask | validation_mask
        local_model = _fit_ridge(local[fit_mask], target[fit_mask], local_ridge)
        r2_local = _r2(target[test_mask], local_model.predict(local[test_mask]))

        diffusion_message_train = _message_features(
            standardized,
            diffusion_train,
            sample_indices,
            int(target_index),
            diffusion=True,
        )
        diffusion_message_fit = _message_features(
            standardized,
            diffusion_fit,
            sample_indices,
            int(target_index),
            diffusion=True,
        )
        r2_diffusion, diffusion_ridge = _fit_nested_message_model(
            local,
            target,
            diffusion_message_train,
            diffusion_message_fit,
            train_mask,
            validation_mask,
            test_mask,
            ridge_grid,
        )

        directed_message_train = _dynamic_graph_features(
            standardized,
            directed_train,
            regime_train,
            sample_indices,
            int(target_index),
            neighbor_count,
            graph_feature_mode,
        )
        directed_message_fit = _dynamic_graph_features(
            standardized,
            directed_fit,
            regime_fit,
            sample_indices,
            int(target_index),
            neighbor_count,
            graph_feature_mode,
        )
        r2_directed, directed_ridge = _fit_nested_message_model(
            local,
            target,
            directed_message_train,
            directed_message_fit,
            train_mask,
            validation_mask,
            test_mask,
            ridge_grid,
        )

        null_scores = []
        for null_train, null_fit in zip(
            directed_train_null,
            directed_fit_null,
            strict=True,
        ):
            null_message_train = _dynamic_graph_features(
                standardized,
                null_train,
                regime_train,
                sample_indices,
                int(target_index),
                neighbor_count,
                graph_feature_mode,
            )
            null_message_fit = _dynamic_graph_features(
                standardized,
                null_fit,
                regime_fit,
                sample_indices,
                int(target_index),
                neighbor_count,
                graph_feature_mode,
            )
            null_r2, _ = _fit_nested_message_model(
                local,
                target,
                null_message_train,
                null_message_fit,
                train_mask,
                validation_mask,
                test_mask,
                ridge_grid,
            )
            null_scores.append(null_r2)

        scores.append(
            GraphTargetScore(
                target_index=int(target_index),
                n_train=int(np.sum(train_mask)),
                n_validation=int(np.sum(validation_mask)),
                n_test=int(np.sum(test_mask)),
                r2_local=r2_local,
                r2_diffusion=r2_diffusion,
                r2_directed=r2_directed,
                r2_rewired=tuple(float(value) for value in null_scores),
                ridge_local=local_ridge,
                ridge_diffusion=diffusion_ridge,
                ridge_directed=directed_ridge,
            )
        )

    rewired_medians = []
    for null_index in range(n_rewired):
        values = [
            score.r2_rewired[null_index] - score.r2_local for score in scores
        ]
        rewired_medians.append(float(np.median(values)) if values else float("nan"))

    directed_eligible = tuple(
        adjacency[np.ix_(eligible, eligible)] for adjacency in directed_fit
    )
    spectral_radius = max(
        (
            float(np.max(np.abs(np.linalg.eigvals(adjacency))))
            if adjacency.size
            else 0.0
        )
        for adjacency in directed_eligible
    )
    density = float(
        np.mean(
            [
                np.mean(np.abs(adjacency) > 0.0)
                for adjacency in directed_eligible
            ]
        )
    )
    digest_builder = hashlib.sha256()
    for adjacency in directed_fit:
        digest_builder.update(np.ascontiguousarray(adjacency, dtype="<f8").tobytes())
    digest = digest_builder.hexdigest()
    return GraphRecordingGate(
        recording_id=recording.recording_id,
        n_units=int(activity.shape[0]),
        n_timepoints=int(time.size),
        horizon_steps=horizon_steps,
        graph_learn_horizon=graph_learn_horizon,
        neighbor_count=neighbor_count,
        graph_feature_mode=graph_feature_mode,
        graph_regimes=graph_regimes,
        scores=tuple(scores),
        rewired_median_deltas=tuple(rewired_medians),
        adjacency_density=density,
        adjacency_spectral_radius=spectral_radius,
        adjacency_sha256=digest,
        min_graph_delta=min_graph_delta,
        min_graph_over_diffusion=min_graph_over_diffusion,
        min_positive_fraction=min_positive_fraction,
        max_rewired_p=max_rewired_p,
    )


def evaluate_graph_panel(
    recordings: Sequence[NeuralRecording],
    *,
    min_recordings_passed: int,
    **options: object,
) -> GraphPanelGate:
    gates = tuple(evaluate_graph_recording(recording, **options) for recording in recordings)
    return GraphPanelGate(gates, min_recordings_passed=min_recordings_passed)


def build_graph_artifact(
    panel: GraphPanelGate,
    *,
    phase: str,
    source_url: str,
    archive_path: str | Path,
    expected_sha256: str | None,
    include_targets: bool,
) -> dict[str, object]:
    archive = Path(archive_path)
    observed_sha256 = sha256_file(archive)
    verified = expected_sha256 is None or observed_sha256 == expected_sha256.lower()
    return {
        "artifact_type": "clarus_weighted_directed_graph_dynamics_gate",
        "artifact_version": 1,
        "phase": phase,
        "claim_tested": (
            "a sparse train-only directed effective graph adds held-out neural-state "
            "prediction beyond local history, symmetric diffusion, and node-permuted graphs"
        ),
        "claim_not_identified": (
            "anonymous activity rows do not identify the effective graph with the "
            "anatomical C. elegans connectome"
        ),
        "equations": {
            "local": "x_i[t+h] = f_i(x_i[t], x_i[t-1], x_i[t-2]) + error",
            "diffusion": (
                "x_i[t+h] = f_i(local) + b_i^T sum_j a_ij "
                "(x_j[t:t-2] - x_i[t:t-2]) + error"
            ),
            "directed": (
                "aggregate: x_i[t+h] = f_i(local) + c_i^T sum_j w_ij "
                "x_j[t:t-2] + error; sparse_var: x_i[t+h] = f_i(local) "
                "+ sum_{j in N_i} beta_ij x_j[t] + error"
            ),
        },
        "split_policy": (
            "chronological 60/20/20 with embargo; imputation, scaling, adjacency, "
            "and ridge selection never inspect the test block"
        ),
        "graph_policy": {
            "directed_edges": (
                "top-k signed correlations between source state and one-step local "
                "innovation in the training block"
            ),
            "diffusion_edges": (
                "symmetric non-negative top-k absolute activity correlations in "
                "the training block"
            ),
            "null": (
                "node permutation P A P^T preserves learned weights and degree "
                "distribution while breaking alignment to recorded units"
            ),
            "multi_horizon_rule": (
                "the directed adjacency is always learned at h=1 and reused for "
                "the requested evaluation horizon"
            ),
            "dynamic_graph": (
                "when graph_regimes=2, a train-fitted population PC1 median split "
                "selects one of two separately trained directed graphs"
            ),
        },
        "provenance": {
            "source_url": source_url,
            "archive": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": observed_sha256,
            "expected_sha256": expected_sha256,
            "sha256_verified": verified,
        },
        "gate_passed": verified and panel.passed,
        "result": panel.to_dict(include_targets=include_targets),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Extracted PredictionCode archive root")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--source-url", default="https://osf.io/dpr3h/")
    parser.add_argument("--phase", choices=("exploratory", "confirmatory"), required=True)
    parser.add_argument("--horizon-steps", type=int, default=1)
    parser.add_argument("--neighbor-count", type=int, default=4)
    parser.add_argument(
        "--graph-feature-mode",
        choices=("aggregate", "sparse_var"),
        default="aggregate",
    )
    parser.add_argument("--graph-regimes", type=int, choices=(1, 2), default=1)
    parser.add_argument("--n-rewired", type=int, default=19)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--min-recordings-passed", type=int, default=3)
    parser.add_argument("--output")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    recordings = load_predictioncode_recordings(args.root)
    panel = evaluate_graph_panel(
        recordings,
        min_recordings_passed=args.min_recordings_passed,
        horizon_steps=args.horizon_steps,
        neighbor_count=args.neighbor_count,
        graph_feature_mode=args.graph_feature_mode,
        graph_regimes=args.graph_regimes,
        n_rewired=args.n_rewired,
        max_targets=args.max_targets,
    )
    artifact = build_graph_artifact(
        panel,
        phase=args.phase,
        source_url=args.source_url,
        archive_path=args.archive,
        expected_sha256=args.expected_sha256,
        include_targets=not args.summary_only,
    )
    rendered = json.dumps(artifact, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if artifact["gate_passed"] else 2


__all__ = [
    "GraphPanelGate",
    "GraphRecordingGate",
    "GraphTargetScore",
    "build_graph_artifact",
    "evaluate_graph_panel",
    "evaluate_graph_recording",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
