"""Confirmatory local temporal-memory gate for neural population recordings.

The gate tests a deliberately narrow claim: past measurements from one unit
contain held-out predictive information about that same unit beyond a flexible
function of its current measurement.  It does not identify a population graph,
an anatomical mechanism, a monad, or a CloudCell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from reality_stone.clarus.cloudcell_dynamics import (
    RIDGE_GRID,
    NeuralRecording,
    _fit_selected_model,
    _r2,
    load_predictioncode_recordings,
    sha256_file,
)


@dataclass(frozen=True)
class LocalMemoryTargetScore:
    """Held-out local-memory ablation for one measured unit."""

    target_index: int
    n_train: int
    n_validation: int
    n_test: int
    r2_current_nonlinear: float
    r2_local_memory: float
    null_r2_local_memory: tuple[float, ...]
    ridge_current: float
    ridge_local: float

    @property
    def delta_memory(self) -> float:
        return self.r2_local_memory - self.r2_current_nonlinear

    def null_delta_memory(self, null_index: int) -> float:
        return self.null_r2_local_memory[null_index] - self.r2_current_nonlinear

    def to_dict(self) -> dict[str, object]:
        return {
            "target_index": self.target_index,
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "n_test": self.n_test,
            "r2_current_nonlinear": self.r2_current_nonlinear,
            "r2_local_memory": self.r2_local_memory,
            "delta_memory": self.delta_memory,
            "null_r2_local_memory": list(self.null_r2_local_memory),
            "ridge_current": self.ridge_current,
            "ridge_local": self.ridge_local,
        }


@dataclass(frozen=True)
class LocalMemoryRecordingGate:
    """One-animal gate; neurons are targets, not independent replicates."""

    recording_id: str
    n_units: int
    n_timepoints: int
    horizon_steps: int
    scores: tuple[LocalMemoryTargetScore, ...]
    model_sha256: str
    n_null_shifts: int = 19
    min_memory_delta: float = 0.01
    min_positive_fraction: float = 0.8
    max_null_p: float = 0.05
    min_targets: int = 20

    @property
    def median_delta_memory(self) -> float:
        if not self.scores:
            return float("nan")
        return float(np.median([score.delta_memory for score in self.scores]))

    @property
    def positive_fraction(self) -> float:
        if not self.scores:
            return 0.0
        return float(np.mean([score.delta_memory > 0.0 for score in self.scores]))

    @property
    def null_median_deltas(self) -> tuple[float, ...]:
        return tuple(
            float(
                np.median(
                    [score.null_delta_memory(null_index) for score in self.scores]
                )
            )
            for null_index in range(self.n_null_shifts)
        )

    @property
    def null_p_value(self) -> float:
        observed = self.median_delta_memory
        exceedances = sum(value >= observed for value in self.null_median_deltas)
        return float((1 + exceedances) / (self.n_null_shifts + 1))

    @property
    def passed(self) -> bool:
        return (
            len(self.scores) >= self.min_targets
            and self.median_delta_memory > self.min_memory_delta
            and self.positive_fraction >= self.min_positive_fraction
            and self.null_p_value <= self.max_null_p
        )

    def to_dict(self, *, include_targets: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "recording_id": self.recording_id,
            "n_units": self.n_units,
            "n_timepoints": self.n_timepoints,
            "horizon_steps": self.horizon_steps,
            "n_targets_evaluated": len(self.scores),
            "median_r2_current_nonlinear": (
                float(np.median([score.r2_current_nonlinear for score in self.scores]))
                if self.scores
                else float("nan")
            ),
            "median_r2_local_memory": (
                float(np.median([score.r2_local_memory for score in self.scores]))
                if self.scores
                else float("nan")
            ),
            "median_delta_memory": self.median_delta_memory,
            "positive_fraction_memory": self.positive_fraction,
            "null_median_deltas": list(self.null_median_deltas),
            "null_p_value": self.null_p_value,
            "model_sha256": self.model_sha256,
            "criteria": {
                "min_memory_delta": self.min_memory_delta,
                "min_positive_fraction": self.min_positive_fraction,
                "max_null_p": self.max_null_p,
                "min_targets": self.min_targets,
                "n_null_shifts": self.n_null_shifts,
            },
            "passed": self.passed,
        }
        if include_targets:
            result["targets"] = [score.to_dict() for score in self.scores]
        return result


@dataclass(frozen=True)
class LocalMemoryPanelGate:
    """Panel inference with independently recorded animals as replicates."""

    recordings: tuple[LocalMemoryRecordingGate, ...]
    min_recordings_passed: int

    @property
    def pass_count(self) -> int:
        return sum(recording.passed for recording in self.recordings)

    @property
    def passed(self) -> bool:
        return self.pass_count >= self.min_recordings_passed

    def to_dict(self, *, include_targets: bool = False) -> dict[str, object]:
        return {
            "recording_count": len(self.recordings),
            "recordings_passed": self.pass_count,
            "min_recordings_passed": self.min_recordings_passed,
            "replicate_unit": "independently recorded animal",
            "passed": self.passed,
            "recordings": [
                recording.to_dict(include_targets=include_targets)
                for recording in self.recordings
            ],
        }


def _current_nonlinear_features(current: np.ndarray) -> np.ndarray:
    """Fixed flexible current-only baseline with no fitted test transform."""

    current = np.asarray(current, dtype=float)
    return np.column_stack(
        [
            current,
            np.square(current),
            np.power(current, 3),
            np.tanh(current),
        ]
    )


def _shift_past_within_splits(
    past: np.ndarray,
    masks: Sequence[np.ndarray],
    null_index: int,
    n_null_shifts: int,
) -> np.ndarray:
    """Phase-shift history within each split, preserving its autocorrelation."""

    shifted = past.copy()
    for mask in masks:
        rows = np.flatnonzero(mask)
        if rows.size < 2:
            continue
        shift = max(1, int(round((null_index + 1) * rows.size / (n_null_shifts + 1))))
        shifted[rows] = np.roll(past[rows], shift, axis=0)
    return shifted


def _update_model_hash(digest: hashlib._Hash, *arrays: np.ndarray | float | int) -> None:
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())


def evaluate_local_memory_recording(
    recording: NeuralRecording,
    *,
    horizon_steps: int = 1,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    embargo: int = 5,
    max_gap_factor: float = 2.0,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    n_null_shifts: int = 19,
    min_memory_delta: float = 0.01,
    min_positive_fraction: float = 0.8,
    max_null_p: float = 0.05,
    min_targets: int = 20,
    max_targets: int | None = None,
) -> LocalMemoryRecordingGate:
    """Evaluate aligned two-lag history against strong current and null models."""

    activity = np.asarray(recording.activity, dtype=float)
    time = np.asarray(recording.time, dtype=float).reshape(-1)
    if activity.ndim != 2 or activity.shape[1] != time.size:
        raise ValueError("activity must have shape (units, time)")
    if time.size < 80:
        raise ValueError("at least 80 timepoints are required")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    if n_null_shifts < 1:
        raise ValueError("n_null_shifts must be positive")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 < validation_fraction < 1.0 - train_fraction:
        raise ValueError("validation_fraction leaves no test block")

    positive_steps = np.diff(time)
    positive_steps = positive_steps[np.isfinite(positive_steps) & (positive_steps > 0)]
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

    training_activity = activity[:, : max(train_end, 1)]
    finite_fraction = np.mean(np.isfinite(training_activity), axis=1)
    variance = np.nanvar(training_activity, axis=1)
    eligible = np.flatnonzero((finite_fraction >= 0.5) & (variance > 1e-12))
    if max_targets is not None and eligible.size > max_targets:
        positions = np.linspace(0, eligible.size - 1, max_targets, dtype=int)
        eligible = eligible[positions]

    digest = hashlib.sha256()
    scores: list[LocalMemoryTargetScore] = []
    for target_index in eligible:
        target_series = activity[target_index]
        local_raw = np.column_stack(
            [
                target_series[sample_indices],
                target_series[sample_indices - 1],
                target_series[sample_indices - 2],
            ]
        )
        target = target_series[sample_indices + horizon_steps]
        valid = continuous & np.isfinite(target) & np.all(np.isfinite(local_raw), axis=1)
        train_mask = valid & base_train
        validation_mask = valid & base_validation
        test_mask = valid & base_test
        if min(np.sum(train_mask), np.sum(validation_mask), np.sum(test_mask)) < 20:
            continue

        current_features = _current_nonlinear_features(local_raw[:, 0])
        local_features = np.column_stack([current_features, local_raw[:, 1:]])
        current_model = _fit_selected_model(
            current_features,
            target,
            train_mask,
            validation_mask,
            ridge_grid,
        )
        local_model = _fit_selected_model(
            local_features,
            target,
            train_mask,
            validation_mask,
            ridge_grid,
        )
        test_target = target[test_mask]
        r2_current = _r2(
            test_target,
            current_model.predict(current_features[test_mask]),
        )
        r2_local = _r2(
            test_target,
            local_model.predict(local_features[test_mask]),
        )

        null_r2: list[float] = []
        for null_index in range(n_null_shifts):
            shifted_past = _shift_past_within_splits(
                local_raw[:, 1:],
                (train_mask, validation_mask, test_mask),
                null_index,
                n_null_shifts,
            )
            null_features = np.column_stack([current_features, shifted_past])
            null_model = _fit_selected_model(
                null_features,
                target,
                train_mask,
                validation_mask,
                ridge_grid,
            )
            null_r2.append(
                _r2(
                    test_target,
                    null_model.predict(null_features[test_mask]),
                )
            )

        _update_model_hash(
            digest,
            int(target_index),
            current_model.ridge,
            current_model.feature_mean,
            current_model.feature_scale,
            current_model.coefficients,
            local_model.ridge,
            local_model.feature_mean,
            local_model.feature_scale,
            local_model.coefficients,
        )
        scores.append(
            LocalMemoryTargetScore(
                target_index=int(target_index),
                n_train=int(np.sum(train_mask)),
                n_validation=int(np.sum(validation_mask)),
                n_test=int(np.sum(test_mask)),
                r2_current_nonlinear=r2_current,
                r2_local_memory=r2_local,
                null_r2_local_memory=tuple(null_r2),
                ridge_current=current_model.ridge,
                ridge_local=local_model.ridge,
            )
        )

    return LocalMemoryRecordingGate(
        recording_id=recording.recording_id,
        n_units=int(activity.shape[0]),
        n_timepoints=int(activity.shape[1]),
        horizon_steps=horizon_steps,
        scores=tuple(scores),
        model_sha256=digest.hexdigest(),
        n_null_shifts=n_null_shifts,
        min_memory_delta=min_memory_delta,
        min_positive_fraction=min_positive_fraction,
        max_null_p=max_null_p,
        min_targets=min_targets,
    )


def evaluate_local_memory_panel(
    recordings: Sequence[NeuralRecording],
    *,
    min_recordings_passed: int,
    **gate_options: object,
) -> LocalMemoryPanelGate:
    gates = tuple(
        evaluate_local_memory_recording(recording, **gate_options)
        for recording in recordings
    )
    return LocalMemoryPanelGate(gates, min_recordings_passed)


def build_local_memory_artifact(
    panel: LocalMemoryPanelGate,
    *,
    phase: str,
    source_url: str,
    archive_path: str | Path,
    expected_sha256: str | None,
    include_targets: bool = False,
) -> dict[str, object]:
    archive = Path(archive_path)
    observed_sha256 = sha256_file(archive)
    integrity_passed = expected_sha256 is None or (
        observed_sha256 == expected_sha256.lower()
    )
    return {
        "artifact_type": "clarus_local_temporal_memory_gate",
        "artifact_version": 1,
        "phase": phase,
        "claim_tested": (
            "aligned same-unit past measurements add held-out prediction beyond "
            "a nonlinear current-only model and refitted phase-shifted histories"
        ),
        "claim_not_identified": (
            "the result does not identify a biological mechanism, anatomical graph, "
            "CloudCell, monad, consciousness, or AGI architecture"
        ),
        "equation": (
            "xhat_i[t+h] = beta0 + g(x_i[t]) + beta1*x_i[t-1] + "
            "beta2*x_i[t-2], g=[x,x^2,x^3,tanh(x)]"
        ),
        "split_policy": (
            "chronological 60/20/20 with embargo; model and ridge selection never "
            "inspect test"
        ),
        "null_policy": (
            "19 autocorrelation-preserving circular shifts of both past columns "
            "within train, validation, and test; every null model is refitted"
        ),
        "provenance": {
            "source_url": source_url,
            "archive": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": observed_sha256,
            "expected_sha256": expected_sha256,
            "sha256_verified": integrity_passed,
        },
        "gate_passed": bool(integrity_passed and panel.passed),
        "result": panel.to_dict(include_targets=include_targets),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--phase", choices=("exploratory", "confirmatory"), required=True)
    parser.add_argument("--horizon-steps", type=int, required=True)
    parser.add_argument("--min-recordings-passed", type=int, required=True)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--output")
    parser.add_argument("--include-targets", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    recordings = load_predictioncode_recordings(args.root)
    panel = evaluate_local_memory_panel(
        recordings,
        horizon_steps=args.horizon_steps,
        min_recordings_passed=args.min_recordings_passed,
        max_targets=args.max_targets,
    )
    artifact = build_local_memory_artifact(
        panel,
        phase=args.phase,
        source_url="https://osf.io/dpr3h/",
        archive_path=args.archive,
        expected_sha256=args.expected_sha256,
        include_targets=args.include_targets,
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
    "LocalMemoryPanelGate",
    "LocalMemoryRecordingGate",
    "LocalMemoryTargetScore",
    "build_local_memory_artifact",
    "evaluate_local_memory_panel",
    "evaluate_local_memory_recording",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
