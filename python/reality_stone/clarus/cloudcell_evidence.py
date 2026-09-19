"""Leakage-safe empirical gates for the CloudCell neuroscience hypothesis.

The core in this module is deliberately NumPy-only.  File-format handling
(including the optional :mod:`h5py` dependency) belongs in the command-line
loader.  The observational gate tests a narrower, falsifiable claim than
``a neuron is a monad``:

* a population code must beat a train-selected best single unit;
* that advantage must survive leave-one-unit-out ablation and block shifts;
* early maintenance activity must contain both local-state and population
  context needed to predict late maintenance activity.

Passing these tests is evidence for a stateful unit embedded in a distributed
population.  It is not an identification of a biological neuron with a
mathematical monad.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


DEFAULT_RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class TrialPopulation:
    """Causal per-trial spike-rate windows for one recording/subject."""

    subject_id: str
    encoding: np.ndarray
    maintenance_early: np.ndarray
    maintenance_late: np.ndarray
    probe: np.ndarray
    memory_load: np.ndarray
    probe_in_out: np.ndarray
    trial_ids: np.ndarray

    def __post_init__(self) -> None:
        windows = {
            "encoding": np.asarray(self.encoding, dtype=float),
            "maintenance_early": np.asarray(self.maintenance_early, dtype=float),
            "maintenance_late": np.asarray(self.maintenance_late, dtype=float),
            "probe": np.asarray(self.probe, dtype=float),
        }
        labels = {
            "memory_load": np.asarray(self.memory_load).reshape(-1),
            "probe_in_out": np.asarray(self.probe_in_out).reshape(-1),
            "trial_ids": np.asarray(self.trial_ids).reshape(-1),
        }
        n_trials = labels["trial_ids"].size
        for name, value in windows.items():
            if value.ndim != 2:
                raise ValueError(f"{name} must be trials x units")
            if value.shape[0] != n_trials:
                raise ValueError(f"{name} does not match trial_ids")
            if value.shape[1] < 2:
                raise ValueError("at least two simultaneously recorded units are required")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if len({value.shape for value in windows.values()}) != 1:
            raise ValueError("all causal windows must have the same trials x units shape")
        for name, value in labels.items():
            if value.size != n_trials:
                raise ValueError(f"{name} does not match trial_ids")
        if n_trials < 12:
            raise ValueError("at least 12 trials are required for nested chronological splits")
        if np.unique(labels["trial_ids"]).size != n_trials:
            raise ValueError("trial_ids must be unique")
        if np.any(np.diff(labels["trial_ids"].astype(float)) <= 0):
            raise ValueError("trial_ids must be in strictly increasing chronological order")
        if np.unique(labels["memory_load"]).size < 2:
            raise ValueError("memory_load must contain at least two classes")
        if np.unique(labels["probe_in_out"]).size < 2:
            raise ValueError("probe_in_out must contain at least two classes")

        object.__setattr__(self, "encoding", windows["encoding"])
        object.__setattr__(self, "maintenance_early", windows["maintenance_early"])
        object.__setattr__(self, "maintenance_late", windows["maintenance_late"])
        object.__setattr__(self, "probe", windows["probe"])
        object.__setattr__(self, "memory_load", labels["memory_load"])
        object.__setattr__(self, "probe_in_out", labels["probe_in_out"])
        object.__setattr__(self, "trial_ids", labels["trial_ids"])

    @property
    def n_trials(self) -> int:
        return int(self.encoding.shape[0])

    @property
    def n_units(self) -> int:
        return int(self.encoding.shape[1])


@dataclass(frozen=True)
class CloudCellGateConfig:
    """Pre-registered thresholds and model-selection settings."""

    train_fraction: float = 0.70
    inner_train_fraction: float = 0.75
    ridge_grid: tuple[float, ...] = DEFAULT_RIDGE_GRID
    n_shifts: int = 19
    block_size: int = 5
    min_population_gain: float = 0.02
    min_dropout_gain: float = 0.0
    min_local_gain: float = 0.01
    min_full_over_best_gain: float = 0.01
    max_null_p: float = 0.05
    min_subject_fraction: float = 2.0 / 3.0
    probe_feature_variant: str = "raw"
    persistence_baseline: str = "mean"

    def __post_init__(self) -> None:
        if not 0.5 <= self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in [0.5, 1)")
        if not 0.5 <= self.inner_train_fraction < 1.0:
            raise ValueError("inner_train_fraction must be in [0.5, 1)")
        if not self.ridge_grid or any(value < 0.0 for value in self.ridge_grid):
            raise ValueError("ridge_grid must contain nonnegative values")
        if self.n_shifts < 1:
            raise ValueError("n_shifts must be positive")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if not 0.0 < self.max_null_p <= 1.0:
            raise ValueError("max_null_p must be in (0, 1]")
        if not 0.0 < self.min_subject_fraction <= 1.0:
            raise ValueError("min_subject_fraction must be in (0, 1]")
        if self.probe_feature_variant not in {"raw", "innovation"}:
            raise ValueError("probe_feature_variant must be 'raw' or 'innovation'")
        if self.persistence_baseline not in {"mean", "task_load"}:
            raise ValueError("persistence_baseline must be 'mean' or 'task_load'")

    def to_dict(self) -> dict[str, object]:
        return {
            "train_fraction": self.train_fraction,
            "inner_train_fraction": self.inner_train_fraction,
            "ridge_grid": list(self.ridge_grid),
            "n_shifts": self.n_shifts,
            "block_size": self.block_size,
            "min_population_gain": self.min_population_gain,
            "min_dropout_gain": self.min_dropout_gain,
            "min_local_gain": self.min_local_gain,
            "min_full_over_best_gain": self.min_full_over_best_gain,
            "max_null_p": self.max_null_p,
            "min_subject_fraction": self.min_subject_fraction,
            "probe_feature_variant": self.probe_feature_variant,
            "persistence_baseline": self.persistence_baseline,
        }


@dataclass(frozen=True)
class CodingComparison:
    """Held-out population-versus-single-unit decoding comparison."""

    n_train: int
    n_test: int
    n_units: int
    population_balanced_accuracy: float
    best_single_balanced_accuracy: float
    baseline_balanced_accuracy: float
    population_gain_over_single: float
    population_gain_over_baseline: float
    best_single_unit: int
    population_ridge: float
    single_ridge: float

    def to_dict(self) -> dict[str, object]:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_units": self.n_units,
            "population_balanced_accuracy": self.population_balanced_accuracy,
            "best_single_balanced_accuracy": self.best_single_balanced_accuracy,
            "baseline_balanced_accuracy": self.baseline_balanced_accuracy,
            "population_gain_over_single": self.population_gain_over_single,
            "population_gain_over_baseline": self.population_gain_over_baseline,
            "best_single_unit": self.best_single_unit,
            "population_ridge": self.population_ridge,
            "single_ridge": self.single_ridge,
        }


@dataclass(frozen=True)
class DropoutResult:
    """Leave-one-unit-out robustness of population-over-single advantage."""

    minimum_population_gain: float
    mean_population_gain: float
    worst_dropped_unit: int
    per_dropped_unit_gain: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_population_gain": self.minimum_population_gain,
            "mean_population_gain": self.mean_population_gain,
            "worst_dropped_unit": self.worst_dropped_unit,
            "per_dropped_unit_gain": list(self.per_dropped_unit_gain),
        }


@dataclass(frozen=True)
class ShiftNullResult:
    """Circular block-shift null for the population-over-single statistic."""

    observed_gain: float
    null_mean_gain: float
    p_value: float
    effective_shifts: int
    offsets: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "observed_gain": self.observed_gain,
            "null_mean_gain": self.null_mean_gain,
            "p_value": self.p_value,
            "effective_shifts": self.effective_shifts,
            "offsets": list(self.offsets),
        }


@dataclass(frozen=True)
class PersistenceResult:
    """Held-out early-to-late maintenance prediction."""

    valid_units: int
    baseline_r2: float
    local_only_r2: float
    cloud_only_r2: float
    full_r2: float
    local_gain_over_baseline: float
    full_gain_over_local: float
    full_gain_over_cloud: float
    full_gain_over_best_partial: float

    def to_dict(self) -> dict[str, object]:
        return {
            "valid_units": self.valid_units,
            "baseline_r2": self.baseline_r2,
            "local_only_r2": self.local_only_r2,
            "cloud_only_r2": self.cloud_only_r2,
            "full_r2": self.full_r2,
            "local_gain_over_baseline": self.local_gain_over_baseline,
            "full_gain_over_local": self.full_gain_over_local,
            "full_gain_over_cloud": self.full_gain_over_cloud,
            "full_gain_over_best_partial": self.full_gain_over_best_partial,
        }


def coding_comparison(
    features: object,
    labels: object,
    *,
    train_fraction: float = 0.70,
    inner_train_fraction: float = 0.75,
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
) -> CodingComparison:
    """Decode labels with nested, chronological, train-only model selection."""

    x, y = _classification_arrays(features, labels)
    split = _split_index(x.shape[0], train_fraction)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    classes = np.unique(y)

    population_ridge, _ = _select_classifier_ridge(
        x_train,
        y_train,
        classes,
        inner_train_fraction,
        ridge_grid,
    )
    population_prediction = _fit_predict_classifier(
        x_train,
        y_train,
        x_test,
        classes,
        population_ridge,
    )

    best_unit = 0
    best_single_ridge = float(tuple(ridge_grid)[0])
    best_validation_score = -np.inf
    for unit in range(x.shape[1]):
        ridge, validation_score = _select_classifier_ridge(
            x_train[:, unit : unit + 1],
            y_train,
            classes,
            inner_train_fraction,
            ridge_grid,
        )
        if validation_score > best_validation_score + 1e-15:
            best_unit = unit
            best_single_ridge = ridge
            best_validation_score = validation_score
    single_prediction = _fit_predict_classifier(
        x_train[:, best_unit : best_unit + 1],
        y_train,
        x_test[:, best_unit : best_unit + 1],
        classes,
        best_single_ridge,
    )
    baseline_class = _majority_class(y_train, classes)
    baseline_prediction = np.full(y_test.shape, baseline_class, dtype=y.dtype)

    population_score = _balanced_accuracy(y_test, population_prediction, classes)
    single_score = _balanced_accuracy(y_test, single_prediction, classes)
    baseline_score = _balanced_accuracy(y_test, baseline_prediction, classes)
    return CodingComparison(
        n_train=split,
        n_test=x.shape[0] - split,
        n_units=x.shape[1],
        population_balanced_accuracy=population_score,
        best_single_balanced_accuracy=single_score,
        baseline_balanced_accuracy=baseline_score,
        population_gain_over_single=population_score - single_score,
        population_gain_over_baseline=population_score - baseline_score,
        best_single_unit=best_unit,
        population_ridge=population_ridge,
        single_ridge=best_single_ridge,
    )


def unit_dropout_gate(
    features: object,
    labels: object,
    *,
    train_fraction: float = 0.70,
    inner_train_fraction: float = 0.75,
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
) -> DropoutResult:
    """Repeat the nested decoder after dropping each recorded unit."""

    x, y = _classification_arrays(features, labels)
    if x.shape[1] < 3:
        raise ValueError("unit dropout requires at least three units")
    gains: list[float] = []
    for dropped in range(x.shape[1]):
        reduced = np.delete(x, dropped, axis=1)
        comparison = coding_comparison(
            reduced,
            y,
            train_fraction=train_fraction,
            inner_train_fraction=inner_train_fraction,
            ridge_grid=ridge_grid,
        )
        gains.append(comparison.population_gain_over_single)
    worst = int(np.argmin(gains))
    return DropoutResult(
        minimum_population_gain=float(gains[worst]),
        mean_population_gain=float(np.mean(gains)),
        worst_dropped_unit=worst,
        per_dropped_unit_gain=tuple(float(value) for value in gains),
    )


def block_shift_null(
    features: object,
    labels: object,
    *,
    train_fraction: float = 0.70,
    inner_train_fraction: float = 0.75,
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
    n_shifts: int = 19,
    block_size: int = 5,
) -> ShiftNullResult:
    """Compare the decoder gain to nonzero circular shifts by whole blocks."""

    x, y = _classification_arrays(features, labels)
    if n_shifts < 1 or block_size < 1:
        raise ValueError("n_shifts and block_size must be positive")
    observed = coding_comparison(
        x,
        y,
        train_fraction=train_fraction,
        inner_train_fraction=inner_train_fraction,
        ridge_grid=ridge_grid,
    ).population_gain_over_single
    possible = tuple(range(block_size, y.size, block_size))
    if not possible:
        raise ValueError("block_size leaves no nonzero circular-shift null")
    if len(possible) <= n_shifts:
        offsets = possible
    else:
        indices = np.linspace(0, len(possible) - 1, n_shifts, dtype=int)
        offsets = tuple(possible[int(index)] for index in np.unique(indices))
    null_gains = np.asarray(
        [
            coding_comparison(
                x,
                np.roll(y, offset),
                train_fraction=train_fraction,
                inner_train_fraction=inner_train_fraction,
                ridge_grid=ridge_grid,
            ).population_gain_over_single
            for offset in offsets
        ],
        dtype=float,
    )
    p_value = float((1 + np.sum(null_gains >= observed)) / (null_gains.size + 1))
    return ShiftNullResult(
        observed_gain=float(observed),
        null_mean_gain=float(np.mean(null_gains)),
        p_value=p_value,
        effective_shifts=int(null_gains.size),
        offsets=offsets,
    )


def maintenance_persistence_gate(
    early: object,
    late: object,
    *,
    covariates: object | None = None,
    train_fraction: float = 0.70,
    inner_train_fraction: float = 0.75,
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
) -> PersistenceResult:
    """Predict each unit's late rate from local, other-unit, and full early rates."""

    x = np.asarray(early, dtype=float)
    y = np.asarray(late, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError("early and late must share a trials x units shape")
    if x.shape[1] < 2:
        raise ValueError("persistence gate requires at least two units")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("persistence arrays must be finite")
    if covariates is None:
        baseline_features = None
    else:
        baseline_features = np.asarray(covariates, dtype=float)
        if (
            baseline_features.ndim != 2
            or baseline_features.shape[0] != x.shape[0]
            or baseline_features.shape[1] < 1
        ):
            raise ValueError("covariates must be trials x features")
        if not np.isfinite(baseline_features).all():
            raise ValueError("covariates must be finite")
    split = _split_index(x.shape[0], train_fraction)

    baseline_scores: list[float] = []
    local_scores: list[float] = []
    cloud_scores: list[float] = []
    full_scores: list[float] = []
    for unit in range(x.shape[1]):
        target_train = y[:split, unit]
        target_test = y[split:, unit]
        if float(np.var(target_test)) <= 1e-12:
            continue
        other_units = np.arange(x.shape[1]) != unit
        local = x[:, unit : unit + 1]
        cloud = x[:, other_units]
        full = x
        if baseline_features is not None:
            local = np.column_stack((baseline_features, local))
            cloud = np.column_stack((baseline_features, cloud))
            full = np.column_stack((baseline_features, full))
        model_scores = []
        for features in (local, cloud, full):
            ridge, _ = _select_regression_ridge(
                features[:split],
                target_train,
                inner_train_fraction,
                ridge_grid,
            )
            prediction = _fit_predict_regression(
                features[:split],
                target_train,
                features[split:],
                ridge,
            )
            model_scores.append(_r2(target_test, prediction))
        if baseline_features is None:
            baseline_prediction = np.full(
                target_test.shape,
                float(np.mean(target_train)),
            )
        else:
            baseline_ridge, _ = _select_regression_ridge(
                baseline_features[:split],
                target_train,
                inner_train_fraction,
                ridge_grid,
            )
            baseline_prediction = _fit_predict_regression(
                baseline_features[:split],
                target_train,
                baseline_features[split:],
                baseline_ridge,
            )
        baseline_scores.append(_r2(target_test, baseline_prediction))
        local_scores.append(model_scores[0])
        cloud_scores.append(model_scores[1])
        full_scores.append(model_scores[2])
    if not local_scores:
        raise ValueError("late maintenance has no variable held-out unit targets")

    baseline_r2 = float(np.mean(baseline_scores))
    local_r2 = float(np.mean(local_scores))
    cloud_r2 = float(np.mean(cloud_scores))
    full_r2 = float(np.mean(full_scores))
    return PersistenceResult(
        valid_units=len(local_scores),
        baseline_r2=baseline_r2,
        local_only_r2=local_r2,
        cloud_only_r2=cloud_r2,
        full_r2=full_r2,
        local_gain_over_baseline=local_r2 - baseline_r2,
        full_gain_over_local=full_r2 - local_r2,
        full_gain_over_cloud=full_r2 - cloud_r2,
        full_gain_over_best_partial=full_r2 - max(local_r2, cloud_r2),
    )


def evaluate_subject(
    data: TrialPopulation,
    config: CloudCellGateConfig = CloudCellGateConfig(),
) -> dict[str, object]:
    """Run all registered gates for one subject without crossing trial folds."""

    common = {
        "train_fraction": config.train_fraction,
        "inner_train_fraction": config.inner_train_fraction,
        "ridge_grid": config.ridge_grid,
    }
    probe_features = data.probe
    if config.probe_feature_variant == "innovation":
        probe_features = data.probe - data.maintenance_late
    coding_inputs = {
        "memory_load_from_encoding": (data.encoding, data.memory_load),
        "probe_membership_from_probe_epoch": (
            probe_features,
            data.probe_in_out,
        ),
    }
    coding: dict[str, object] = {}
    coding_passes: list[bool] = []
    for name, (features, labels) in coding_inputs.items():
        comparison = coding_comparison(features, labels, **common)
        dropout = unit_dropout_gate(features, labels, **common)
        shift_null = block_shift_null(
            features,
            labels,
            **common,
            n_shifts=config.n_shifts,
            block_size=config.block_size,
        )
        criteria = {
            "population_over_single": (
                comparison.population_gain_over_single >= config.min_population_gain
            ),
            "population_over_baseline": (
                comparison.population_gain_over_baseline >= config.min_population_gain
            ),
            "unit_dropout": dropout.minimum_population_gain >= config.min_dropout_gain,
            "block_shift_null": shift_null.p_value <= config.max_null_p,
        }
        passed = all(criteria.values())
        coding_passes.append(passed)
        coding[name] = {
            "comparison": comparison.to_dict(),
            "unit_dropout": dropout.to_dict(),
            "block_shift_null": shift_null.to_dict(),
            "criteria_passed": criteria,
            "passed": passed,
        }

    persistence_covariates = None
    if config.persistence_baseline == "task_load":
        load_classes = np.unique(data.memory_load)
        persistence_covariates = np.column_stack(
            [data.memory_load == label for label in load_classes[1:]]
        ).astype(float)
    persistence = maintenance_persistence_gate(
        data.maintenance_early,
        data.maintenance_late,
        covariates=persistence_covariates,
        **common,
    )
    persistence_criteria = {
        "local_state_over_mean": (
            persistence.local_only_r2 > 0.0
            and persistence.local_gain_over_baseline >= config.min_local_gain
        ),
        "local_and_cloud_complementarity": (
            persistence.full_r2 > 0.0
            and persistence.full_gain_over_best_partial >= config.min_full_over_best_gain
        ),
    }
    persistence_passed = all(persistence_criteria.values())
    operational_passed = all(coding_passes) and persistence_passed
    return {
        "subject_id": data.subject_id,
        "n_trials": data.n_trials,
        "n_units": data.n_units,
        "window_policy": {
            "encoding": "[Encoding1 onset, Maintenance onset)",
            "maintenance_early": "[Maintenance onset, maintenance midpoint)",
            "maintenance_late": "[maintenance midpoint, Probe onset)",
            "probe": "[Probe onset, Response onset)",
            "rates": "spike count divided by each causal window duration",
            "probe_feature_variant": config.probe_feature_variant,
            "persistence_baseline": config.persistence_baseline,
        },
        "coding": coding,
        "maintenance_persistence": {
            "comparison": persistence.to_dict(),
            "criteria_passed": persistence_criteria,
            "passed": persistence_passed,
        },
        "operational_gate_passed": operational_passed,
    }


def evaluate_panel(
    datasets: Sequence[TrialPopulation],
    config: CloudCellGateConfig = CloudCellGateConfig(),
) -> dict[str, object]:
    """Evaluate the downloaded panel without treating subjects as independent replications."""

    if not datasets:
        raise ValueError("at least one subject dataset is required")
    subject_ids = [dataset.subject_id for dataset in datasets]
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("subject_id values must be unique within the panel")
    subjects = [evaluate_subject(dataset, config) for dataset in datasets]
    pass_count = sum(bool(subject["operational_gate_passed"]) for subject in subjects)
    pass_fraction = pass_count / len(subjects)
    operational_passed = pass_fraction >= config.min_subject_fraction
    return {
        "subject_count": len(subjects),
        "subject_pass_count": pass_count,
        "subject_pass_fraction": pass_fraction,
        "operational_gate_passed": operational_passed,
        "literal_coded_monad_claim": {
            "decision": "withhold_identity_claim",
            "reason": (
                "The registered observational signatures passed, but observational "
                "prediction cannot identify a biological unit with a mathematical monad."
                if operational_passed
                else "The downloaded panel did not pass the registered operational signatures."
            ),
            "scope": (
                "This panel is a convenience sample of simultaneously recorded human "
                "single units; it is not a subject-independent population estimate."
            ),
        },
        "subjects": subjects,
    }


def build_cloudcell_artifact(
    panel: Mapping[str, object],
    *,
    config: CloudCellGateConfig,
    provenance: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build a self-describing artifact for a real-data or synthetic gate run."""

    return {
        "artifact_type": "clarus_cloudcell_human_mtl_gate",
        "artifact_version": 1,
        "source_id": "human_mtl_sternberg_nwb_panel",
        "claim_tested": (
            "stateful local units embedded in a population cloud show distributed coding "
            "and complementary maintenance dynamics"
        ),
        "claim_not_identified": "a biological neuron is literally a mathematical monad",
        "target_policy": {
            "memory_set_proxy": (
                "NWB loads (set cardinality); this does not test exact picture-set identity"
            ),
            "probe": "NWB probe_in_out decoded during [Probe onset, Response onset)",
        },
        "split_policy": (
            "outer chronological trial split; ridge and best-single selection use only "
            "an inner chronological split of the outer training trials"
        ),
        "criteria": {
            **config.to_dict(),
            "positive_heldout_r2_required_for_persistence": True,
        },
        "provenance": [dict(item) for item in provenance],
        "gate_passed": bool(panel["operational_gate_passed"]),
        "result": dict(panel),
    }


def _classification_arrays(features: object, labels: object) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("features must be trials x units and match labels")
    if x.shape[1] < 1:
        raise ValueError("at least one feature is required")
    if not np.isfinite(x).all():
        raise ValueError("features contain non-finite values")
    if np.unique(y).size < 2:
        raise ValueError("labels must contain at least two classes")
    return x, y


def _split_index(n_samples: int, fraction: float) -> int:
    if not 0.5 <= fraction < 1.0:
        raise ValueError("split fraction must be in [0.5, 1)")
    split = int(np.floor(n_samples * fraction))
    if split < 4 or n_samples - split < 2:
        raise ValueError("split requires at least four train and two test trials")
    return split


def _select_classifier_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    classes: np.ndarray,
    inner_train_fraction: float,
    ridge_grid: Sequence[float],
) -> tuple[float, float]:
    grid = tuple(float(value) for value in ridge_grid)
    if not grid:
        raise ValueError("ridge_grid cannot be empty")
    split = _split_index(x_train.shape[0], inner_train_fraction)
    best_ridge = grid[0]
    best_score = -np.inf
    for ridge in grid:
        prediction = _fit_predict_classifier(
            x_train[:split],
            y_train[:split],
            x_train[split:],
            classes,
            ridge,
        )
        score = _balanced_accuracy(y_train[split:], prediction, classes)
        if score > best_score + 1e-15:
            best_ridge = ridge
            best_score = score
    return best_ridge, float(best_score)


def _fit_predict_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    classes: np.ndarray,
    ridge: float,
) -> np.ndarray:
    train, test = _standardize_train_test(x_train, x_test)
    targets = np.column_stack([(y_train == label).astype(float) for label in classes])
    target_mean = np.mean(targets, axis=0)
    weights = _ridge_weights(train, targets - target_mean, ridge)
    scores = test @ weights + target_mean
    return classes[np.argmax(scores, axis=1)]


def _select_regression_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    inner_train_fraction: float,
    ridge_grid: Sequence[float],
) -> tuple[float, float]:
    grid = tuple(float(value) for value in ridge_grid)
    if not grid:
        raise ValueError("ridge_grid cannot be empty")
    split = _split_index(x_train.shape[0], inner_train_fraction)
    best_ridge = grid[0]
    best_score = -np.inf
    for ridge in grid:
        prediction = _fit_predict_regression(
            x_train[:split],
            y_train[:split],
            x_train[split:],
            ridge,
        )
        score = _r2(y_train[split:], prediction)
        if score > best_score + 1e-15:
            best_ridge = ridge
            best_score = score
    return best_ridge, float(best_score)


def _fit_predict_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    ridge: float,
) -> np.ndarray:
    train, test = _standardize_train_test(x_train, x_test)
    target_mean = float(np.mean(y_train))
    weights = _ridge_weights(train, y_train - target_mean, ridge)
    return test @ weights + target_mean


def _ridge_weights(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = float(ridge) * np.eye(x.shape[1], dtype=float)
    left = x.T @ x + penalty
    right = x.T @ y
    try:
        return np.linalg.solve(left, right)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(left) @ right


def _standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (x_train - mean) / scale, (x_test - mean) / scale


def _majority_class(y: np.ndarray, classes: np.ndarray) -> object:
    counts = np.asarray([np.sum(y == label) for label in classes])
    return classes[int(np.argmax(counts))]


def _balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
) -> float:
    recalls = [
        float(np.mean(y_pred[y_true == label] == label))
        for label in classes
        if np.any(y_true == label)
    ]
    if not recalls:
        raise ValueError("balanced accuracy requires at least one held-out class")
    return float(np.mean(recalls))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denominator <= 1e-15:
        return 0.0
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denominator


__all__ = [
    "CloudCellGateConfig",
    "CodingComparison",
    "DEFAULT_RIDGE_GRID",
    "DropoutResult",
    "PersistenceResult",
    "ShiftNullResult",
    "TrialPopulation",
    "block_shift_null",
    "build_cloudcell_artifact",
    "coding_comparison",
    "evaluate_panel",
    "evaluate_subject",
    "maintenance_persistence_gate",
    "unit_dropout_gate",
]
