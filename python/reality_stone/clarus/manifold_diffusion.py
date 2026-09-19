"""Population-manifold diffusion/analog forecasting gates.

This module tests a different hypothesis from a graph over recorded neurons.
It builds a train-only delay embedding of the population state, connects a
query to nearby training states with a Gaussian kernel, and transports their
observed future states through that kernel.

The comparison is nested and chronological:

* persistence: ``X[t+h] = X[t]``;
* linear state: ridge map from the same delay embedding;
* diffusion analog: kernel-weighted futures of neighboring training states;
* shift null: the same kernel graph with circularly shifted library futures.
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
NEIGHBOR_GRID = (4, 8, 16, 32, 64, 128, 256)


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
class DiffusionRecordingGate:
    recording_id: str
    n_observed_units: int
    n_output_dimensions: int
    n_timepoints: int
    horizon_steps: int
    n_components: int
    delay_count: int
    delay_stride: int
    forecast_target: str
    output_space: str
    selected_neighbors: int
    selected_ridge: float
    n_train: int
    n_validation: int
    n_test: int
    r2_persistence: float
    r2_linear: float
    r2_diffusion: float
    positive_unit_fraction_over_linear: float
    shifted_r2: tuple[float, ...]
    state_model_sha256: str
    min_diffusion_delta: float = 0.01
    min_positive_fraction: float = 0.6
    max_shift_p: float = 0.05

    @property
    def delta_diffusion_over_linear(self) -> float:
        return self.r2_diffusion - self.r2_linear

    @property
    def delta_diffusion_over_persistence(self) -> float:
        return self.r2_diffusion - self.r2_persistence

    @property
    def delta_diffusion_over_best_baseline(self) -> float:
        return self.r2_diffusion - max(self.r2_linear, self.r2_persistence)

    @property
    def shift_p_value(self) -> float:
        null = np.asarray(self.shifted_r2, dtype=float)
        return float((1 + np.sum(null >= self.r2_diffusion)) / (null.size + 1))

    @property
    def passed(self) -> bool:
        return (
            self.delta_diffusion_over_best_baseline > self.min_diffusion_delta
            and self.positive_unit_fraction_over_linear >= self.min_positive_fraction
            and self.shift_p_value <= self.max_shift_p
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "recording_id": self.recording_id,
            "n_observed_units": self.n_observed_units,
            "n_output_dimensions": self.n_output_dimensions,
            "n_timepoints": self.n_timepoints,
            "horizon_steps": self.horizon_steps,
            "n_components": self.n_components,
            "delay_count": self.delay_count,
            "delay_stride": self.delay_stride,
            "forecast_target": self.forecast_target,
            "output_space": self.output_space,
            "state_dimension": self.n_components * self.delay_count,
            "selected_neighbors": self.selected_neighbors,
            "selected_ridge": self.selected_ridge,
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "n_test": self.n_test,
            "r2_persistence": self.r2_persistence,
            "r2_linear": self.r2_linear,
            "r2_diffusion": self.r2_diffusion,
            "delta_diffusion_over_persistence": self.delta_diffusion_over_persistence,
            "delta_diffusion_over_linear": self.delta_diffusion_over_linear,
            "delta_diffusion_over_best_baseline": (
                self.delta_diffusion_over_best_baseline
            ),
            "positive_unit_fraction_over_linear": (
                self.positive_unit_fraction_over_linear
            ),
            "shifted_r2": list(self.shifted_r2),
            "shift_p_value": self.shift_p_value,
            "state_model_sha256": self.state_model_sha256,
            "criteria": {
                "min_diffusion_delta": self.min_diffusion_delta,
                "min_positive_fraction": self.min_positive_fraction,
                "max_shift_p": self.max_shift_p,
            },
            "passed": self.passed,
        }


@dataclass(frozen=True)
class DiffusionPanelGate:
    recordings: tuple[DiffusionRecordingGate, ...]
    min_recordings_passed: int

    @property
    def pass_count(self) -> int:
        return sum(recording.passed for recording in self.recordings)

    @property
    def passed(self) -> bool:
        return self.pass_count >= self.min_recordings_passed

    def to_dict(self) -> dict[str, object]:
        return {
            "recording_count": len(self.recordings),
            "recordings_passed": self.pass_count,
            "min_recordings_passed": self.min_recordings_passed,
            "passed": self.passed,
            "replicate_unit": "independently recorded animal",
            "recordings": [recording.to_dict() for recording in self.recordings],
        }


@dataclass(frozen=True)
class _PopulationState:
    standardized_activity: np.ndarray
    latent: np.ndarray
    component_bytes: bytes


def _fit_population_state(
    activity: np.ndarray,
    fit_frames: np.ndarray,
    n_components: int,
) -> _PopulationState:
    finite_fraction = np.mean(np.isfinite(activity[:, fit_frames]), axis=1)
    variance = np.nanvar(activity[:, fit_frames], axis=1)
    keep = (finite_fraction >= 0.5) & (variance > 1e-12)
    if np.sum(keep) < 3:
        raise ValueError("fewer than three population units survive preprocessing")
    selected = activity[keep]
    medians = np.nanmedian(selected[:, fit_frames], axis=1)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(selected), selected, medians[:, None])
    mean = np.mean(filled[:, fit_frames], axis=1)
    scale = np.std(filled[:, fit_frames], axis=1)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (filled - mean[:, None]) / scale[:, None]
    training = standardized[:, fit_frames].T
    _, _, right = np.linalg.svd(training, full_matrices=False)
    count = min(n_components, right.shape[0])
    components = right[:count].T
    for column in range(components.shape[1]):
        anchor = int(np.argmax(np.abs(components[:, column])))
        if components[anchor, column] < 0.0:
            components[:, column] *= -1.0
    latent = standardized.T @ components
    digest_parts = [
        np.ascontiguousarray(mean, dtype="<f8").tobytes(),
        np.ascontiguousarray(scale, dtype="<f8").tobytes(),
        np.ascontiguousarray(components, dtype="<f8").tobytes(),
    ]
    return _PopulationState(standardized, latent, b"".join(digest_parts))


def _delay_state(
    latent: np.ndarray,
    indices: np.ndarray,
    delay_count: int,
    delay_stride: int,
) -> np.ndarray:
    return np.column_stack(
        [
            latent[indices - delay * delay_stride]
            for delay in range(delay_count)
        ]
    )


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


def _multivariate_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    centered = target - np.mean(target, axis=0, keepdims=True)
    denominator = float(np.sum(np.square(centered)))
    if denominator <= 1e-15:
        return 0.0
    return 1.0 - float(np.sum(np.square(target - prediction))) / denominator


def _unit_r2(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    centered = target - np.mean(target, axis=0, keepdims=True)
    denominator = np.sum(np.square(centered), axis=0)
    numerator = np.sum(np.square(target - prediction), axis=0)
    return np.where(denominator > 1e-15, 1.0 - numerator / denominator, 0.0)


def _scaled_states(
    library: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(library, axis=0)
    scale = np.std(library, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (library - mean) / scale, (query - mean) / scale


def _analog_prediction(
    library_state: np.ndarray,
    library_target: np.ndarray,
    query_state: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    count = min(int(neighbors), library_state.shape[0])
    if count < 1:
        raise ValueError("analog library is empty")
    library, query = _scaled_states(library_state, query_state)
    distances = (
        np.sum(np.square(query), axis=1, keepdims=True)
        + np.sum(np.square(library), axis=1)[None, :]
        - 2.0 * query @ library.T
    )
    distances = np.maximum(distances, 0.0)
    nearest = np.argpartition(distances, count - 1, axis=1)[:, :count]
    nearest_distances = np.take_along_axis(distances, nearest, axis=1)
    bandwidth = np.median(nearest_distances, axis=1, keepdims=True)
    bandwidth = np.where(bandwidth > 1e-12, bandwidth, 1.0)
    weights = np.exp(-nearest_distances / bandwidth)
    weights /= np.sum(weights, axis=1, keepdims=True)
    futures = library_target[nearest]
    return np.sum(weights[:, :, None] * futures, axis=1)


def _select_neighbors(
    train_state: np.ndarray,
    train_target: np.ndarray,
    validation_state: np.ndarray,
    validation_target: np.ndarray,
    neighbor_grid: Sequence[int],
) -> int:
    best = int(neighbor_grid[0])
    best_error = float("inf")
    for neighbors in neighbor_grid:
        prediction = _analog_prediction(
            train_state,
            train_target,
            validation_state,
            int(neighbors),
        )
        error = float(np.mean(np.square(validation_target - prediction)))
        if error < best_error:
            best_error = error
            best = int(neighbors)
    return best


def _select_ridge(
    train_state: np.ndarray,
    train_target: np.ndarray,
    validation_state: np.ndarray,
    validation_target: np.ndarray,
    ridge_grid: Sequence[float],
) -> float:
    best = float(ridge_grid[0])
    best_error = float("inf")
    for ridge in ridge_grid:
        prediction = _fit_ridge(train_state, train_target, float(ridge)).predict(
            validation_state
        )
        error = float(np.mean(np.square(validation_target - prediction)))
        if error < best_error:
            best_error = error
            best = float(ridge)
    return best


def _valid_samples(
    activity: np.ndarray,
    time: np.ndarray,
    indices: np.ndarray,
    max_lag: int,
    horizon: int,
    max_gap: float,
) -> np.ndarray:
    steps = np.column_stack(
        [
            time[indices + offset + 1] - time[indices + offset]
            for offset in range(-max_lag, horizon)
        ]
    )
    continuous = np.all(
        np.isfinite(steps) & (steps > 0.0) & (steps <= max_gap),
        axis=1,
    )
    frame_finite = np.mean(np.isfinite(activity), axis=0) >= 0.5
    finite = np.ones(indices.size, dtype=bool)
    for offset in range(-max_lag, horizon + 1):
        finite &= frame_finite[indices + offset]
    return continuous & finite


def _shift_offsets(length: int, n_shifts: int) -> tuple[int, ...]:
    fractions = np.linspace(1.0 / (n_shifts + 1), n_shifts / (n_shifts + 1), n_shifts)
    offsets = np.unique(np.maximum(1, np.rint(fractions * length).astype(int)))
    return tuple(int(offset) for offset in offsets if offset < length)


def evaluate_diffusion_recording(
    recording: NeuralRecording,
    *,
    horizon_steps: int = 6,
    n_components: int = 8,
    delay_count: int = 3,
    delay_stride: int = 2,
    forecast_target: str = "absolute",
    output_space: str = "activity",
    neighbor_grid: Sequence[int] = NEIGHBOR_GRID,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    n_shifts: int = 19,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    embargo: int = 5,
    max_gap_factor: float = 2.0,
    min_diffusion_delta: float = 0.01,
    min_positive_fraction: float = 0.6,
    max_shift_p: float = 0.05,
) -> DiffusionRecordingGate:
    """Evaluate one recording without exposing its test block to model selection."""

    activity = np.asarray(recording.activity, dtype=float)
    time = np.asarray(recording.time, dtype=float).reshape(-1)
    if activity.ndim != 2 or activity.shape[1] != time.size:
        raise ValueError("activity must be units x time and match time")
    if min(horizon_steps, n_components, delay_count, delay_stride, n_shifts) < 1:
        raise ValueError("horizon, state sizes, and n_shifts must be positive")
    if forecast_target not in {"absolute", "increment"}:
        raise ValueError("forecast_target must be 'absolute' or 'increment'")
    if output_space not in {"activity", "latent"}:
        raise ValueError("output_space must be 'activity' or 'latent'")
    if not neighbor_grid or min(neighbor_grid) < 1:
        raise ValueError("neighbor_grid must contain positive values")

    train_end = int(np.floor(train_fraction * time.size))
    validation_end = int(
        np.floor((train_fraction + validation_fraction) * time.size)
    )
    train_frames = np.arange(time.size) < train_end - embargo
    fit_frames = np.arange(time.size) < validation_end - embargo
    train_state_model = _fit_population_state(activity, train_frames, n_components)
    fit_state_model = _fit_population_state(activity, fit_frames, n_components)

    positive_steps = np.diff(time)
    positive_steps = positive_steps[np.isfinite(positive_steps) & (positive_steps > 0.0)]
    if not positive_steps.size:
        raise ValueError("time must contain positive increments")
    max_gap = float(np.median(positive_steps) * max_gap_factor)
    max_lag = (delay_count - 1) * delay_stride
    indices = np.arange(max_lag, time.size - horizon_steps)
    valid = _valid_samples(
        activity,
        time,
        indices,
        max_lag,
        horizon_steps,
        max_gap,
    )
    train_mask = valid & (indices + horizon_steps < train_end - embargo)
    validation_mask = valid & (indices >= train_end + embargo) & (
        indices + horizon_steps < validation_end - embargo
    )
    test_mask = valid & (indices >= validation_end + embargo)
    fit_mask = train_mask | validation_mask
    if min(np.sum(train_mask), np.sum(validation_mask), np.sum(test_mask)) < 20:
        raise ValueError("chronological split has fewer than 20 valid samples")

    train_delay = _delay_state(
        train_state_model.latent,
        indices,
        delay_count,
        delay_stride,
    )
    fit_delay = _delay_state(
        fit_state_model.latent,
        indices,
        delay_count,
        delay_stride,
    )
    if output_space == "activity":
        train_future_all = train_state_model.standardized_activity[
            :, indices + horizon_steps
        ].T
        fit_future_all = fit_state_model.standardized_activity[
            :, indices + horizon_steps
        ].T
        train_current_all = train_state_model.standardized_activity[:, indices].T
        fit_current_all = fit_state_model.standardized_activity[:, indices].T
    else:
        train_future_all = train_state_model.latent[indices + horizon_steps]
        fit_future_all = fit_state_model.latent[indices + horizon_steps]
        train_current_all = train_state_model.latent[indices]
        fit_current_all = fit_state_model.latent[indices]
    if forecast_target == "increment":
        train_target_all = train_future_all - train_current_all
        fit_target_all = fit_future_all - fit_current_all
    else:
        train_target_all = train_future_all
        fit_target_all = fit_future_all

    selected_neighbors = _select_neighbors(
        train_delay[train_mask],
        train_target_all[train_mask],
        train_delay[validation_mask],
        train_target_all[validation_mask],
        neighbor_grid,
    )
    selected_ridge = _select_ridge(
        train_delay[train_mask],
        train_target_all[train_mask],
        train_delay[validation_mask],
        train_target_all[validation_mask],
        ridge_grid,
    )

    test_target = fit_future_all[test_mask]
    diffusion_forecast = _analog_prediction(
        fit_delay[fit_mask],
        fit_target_all[fit_mask],
        fit_delay[test_mask],
        selected_neighbors,
    )
    linear_model = _fit_ridge(
        fit_delay[fit_mask],
        fit_target_all[fit_mask],
        selected_ridge,
    )
    linear_forecast = linear_model.predict(fit_delay[test_mask])
    persistence_prediction = fit_current_all[test_mask]
    if forecast_target == "increment":
        diffusion_prediction = persistence_prediction + diffusion_forecast
        linear_prediction = persistence_prediction + linear_forecast
    else:
        diffusion_prediction = diffusion_forecast
        linear_prediction = linear_forecast

    shifted_scores = []
    for offset in _shift_offsets(int(np.sum(fit_mask)), n_shifts):
        shifted_target = np.roll(fit_target_all[fit_mask], offset, axis=0)
        shifted_forecast = _analog_prediction(
            fit_delay[fit_mask],
            shifted_target,
            fit_delay[test_mask],
            selected_neighbors,
        )
        prediction = (
            persistence_prediction + shifted_forecast
            if forecast_target == "increment"
            else shifted_forecast
        )
        shifted_scores.append(_multivariate_r2(test_target, prediction))

    diffusion_unit_r2 = _unit_r2(test_target, diffusion_prediction)
    linear_unit_r2 = _unit_r2(test_target, linear_prediction)
    digest = hashlib.sha256()
    digest.update(fit_state_model.component_bytes)
    digest.update(np.asarray([selected_neighbors, selected_ridge], dtype="<f8").tobytes())
    digest.update(forecast_target.encode("ascii"))
    digest.update(output_space.encode("ascii"))
    return DiffusionRecordingGate(
        recording_id=recording.recording_id,
        n_observed_units=int(fit_state_model.standardized_activity.shape[0]),
        n_output_dimensions=int(fit_future_all.shape[1]),
        n_timepoints=int(time.size),
        horizon_steps=horizon_steps,
        n_components=min(n_components, fit_state_model.latent.shape[1]),
        delay_count=delay_count,
        delay_stride=delay_stride,
        forecast_target=forecast_target,
        output_space=output_space,
        selected_neighbors=selected_neighbors,
        selected_ridge=selected_ridge,
        n_train=int(np.sum(train_mask)),
        n_validation=int(np.sum(validation_mask)),
        n_test=int(np.sum(test_mask)),
        r2_persistence=_multivariate_r2(test_target, persistence_prediction),
        r2_linear=_multivariate_r2(test_target, linear_prediction),
        r2_diffusion=_multivariate_r2(test_target, diffusion_prediction),
        positive_unit_fraction_over_linear=float(
            np.mean(diffusion_unit_r2 > linear_unit_r2)
        ),
        shifted_r2=tuple(float(value) for value in shifted_scores),
        state_model_sha256=digest.hexdigest(),
        min_diffusion_delta=min_diffusion_delta,
        min_positive_fraction=min_positive_fraction,
        max_shift_p=max_shift_p,
    )


def evaluate_diffusion_panel(
    recordings: Sequence[NeuralRecording],
    *,
    min_recordings_passed: int,
    **options: object,
) -> DiffusionPanelGate:
    gates = tuple(
        evaluate_diffusion_recording(recording, **options) for recording in recordings
    )
    return DiffusionPanelGate(gates, min_recordings_passed=min_recordings_passed)


def build_diffusion_artifact(
    panel: DiffusionPanelGate,
    *,
    phase: str,
    source_url: str,
    archive_path: str | Path,
    expected_sha256: str | None,
) -> dict[str, object]:
    archive = Path(archive_path)
    observed_sha256 = sha256_file(archive)
    verified = expected_sha256 is None or observed_sha256 == expected_sha256.lower()
    return {
        "artifact_type": "clarus_population_manifold_diffusion_gate",
        "artifact_version": 1,
        "phase": phase,
        "claim_tested": (
            "a train-only delay-state diffusion kernel transports future population "
            "activity or latent state better than persistence, a linear state map, "
            "and shifted futures"
        ),
        "claim_not_identified": (
            "kernel analog prediction is a predictive manifold result, not proof of "
            "an anatomical neural graph or biological monad"
        ),
        "equation": (
            "K_ts = exp(-||z_t-z_s||^2/epsilon_t); "
            "absolute: Xhat[t+h] = sum_s K_ts X[s+h] / sum_s K_ts; "
            "increment: Xhat[t+h] = X[t] + sum_s K_ts "
            "(X[s+h]-X[s]) / sum_s K_ts"
        ),
        "split_policy": (
            "chronological 60/20/20 with embargo; imputation, scaling, PCA, delay "
            "state, neighbor count, ridge, and kernel library never inspect test"
        ),
        "null_policy": (
            "circularly shift autocorrelated future states within the fit library "
            "while preserving query states and kernel geometry"
        ),
        "provenance": {
            "source_url": source_url,
            "archive": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": observed_sha256,
            "expected_sha256": expected_sha256,
            "sha256_verified": verified,
        },
        "gate_passed": verified and panel.passed,
        "result": panel.to_dict(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--source-url", default="https://osf.io/dpr3h/")
    parser.add_argument("--phase", choices=("exploratory", "confirmatory"), required=True)
    parser.add_argument("--horizon-steps", type=int, default=6)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--delay-count", type=int, default=3)
    parser.add_argument("--delay-stride", type=int, default=2)
    parser.add_argument(
        "--forecast-target",
        choices=("absolute", "increment"),
        default="absolute",
    )
    parser.add_argument(
        "--output-space",
        choices=("activity", "latent"),
        default="activity",
    )
    parser.add_argument("--n-shifts", type=int, default=19)
    parser.add_argument("--min-recordings-passed", type=int, default=3)
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    recordings = load_predictioncode_recordings(args.root)
    panel = evaluate_diffusion_panel(
        recordings,
        min_recordings_passed=args.min_recordings_passed,
        horizon_steps=args.horizon_steps,
        n_components=args.n_components,
        delay_count=args.delay_count,
        delay_stride=args.delay_stride,
        forecast_target=args.forecast_target,
        output_space=args.output_space,
        n_shifts=args.n_shifts,
    )
    artifact = build_diffusion_artifact(
        panel,
        phase=args.phase,
        source_url=args.source_url,
        archive_path=args.archive,
        expected_sha256=args.expected_sha256,
    )
    rendered = json.dumps(artifact, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if artifact["gate_passed"] else 2


__all__ = [
    "DiffusionPanelGate",
    "DiffusionRecordingGate",
    "build_diffusion_artifact",
    "evaluate_diffusion_panel",
    "evaluate_diffusion_recording",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
