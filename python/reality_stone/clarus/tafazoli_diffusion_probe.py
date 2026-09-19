"""Low-cost diffusion/noise proxy probe for the Tafazoli PFC snapshot.

All fits are session-local and label blind.  A common VAR(1) drift is followed
by a covariance ladder.  Covariances are estimated from trial-wise,
outer-training-only out-of-fold residuals and scored once on common outer-test
targets with a multivariate Gaussian/BIC codelength proxy.

This module compares restricted observational predictors.  It cannot identify
biological diffusion, a score function, a generative reverse process, a causal
mechanism, or spatial graph diffusion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import factorial, log2, pi
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .tafazoli_call_graph_probe import (
    PastOnlyGate,
    TrajectoryDesign,
    assign_past_only_states,
    build_trajectory_design,
    fit_past_only_gate,
    whiten_prepared_latent_fold,
)
from .tafazoli_session_operator_probe import (
    OFFICIAL_CLASSIFIER_MD5,
    PreparedLatentFold,
    SessionSpec,
    load_tafazoli_train_dimensions,
    make_whole_trial_folds,
    prepare_session_latent_fold,
    recovered_session_specs,
    verify_official_classifier_checksum,
)


SCHEMA_VERSION = "clarus-tafazoli-diffusion-probe/v1"
CHECKPOINT_SCHEMA_VERSION = "clarus-tafazoli-diffusion-checkpoint/v2"
IMPLEMENTATION_REVISION = "auditable-semigroup-scores/v5"
PROBE_SCOPE = "label_blind_session_local_diffusion_covariance_proxy"

YES = "YES"
NO = "NO"
PENDING = "PENDING"
TEST_UNAVAILABLE = "TEST_UNAVAILABLE"

OU_ISO = "OU_ISO"
OU_DIAG = "OU_DIAG"
OU_FULL = "OU_FULL"
TIME_SCALE = "TIME_SCALE"
STATE_SCALE = "STATE_SCALE_K2_CURRENT_ONLY"
QUADRATIC_DRIFT_FULL_Q = "QUADRATIC_DRIFT_FULL_Q"

PRIMARY_FAMILIES = (
    OU_ISO,
    OU_DIAG,
    OU_FULL,
    TIME_SCALE,
    STATE_SCALE,
    QUADRATIC_DRIFT_FULL_Q,
)


@dataclass(frozen=True)
class DiffusionProbeConfig:
    """Fixed low-cost protocol."""

    seed: int = 20260730
    rank_cap: int = 3
    time_bin_milliseconds: int = 10
    observation_window_bins: int = 10
    lag_bins: int = 10
    primary_stride_bins: int = 10
    global_anchor_depth: int = 3
    outer_fold_count: int = 6
    covariance_oof_fold_count: int = 3
    ridge_alpha: float = 1.0
    full_covariance_shrinkage: float = 0.1
    state_count: int = 2
    kmeans_restarts: int = 4
    kmeans_max_iterations: int = 100
    minimum_codelength_advantage_bits_per_scalar: float = 0.01
    minimum_session_unit_win_fraction: float = 0.5
    semigroup_max_excess_bits_per_scalar: float = 0.02
    markov_orders: tuple[int, ...] = (1, 2, 3)
    semigroup_horizons: tuple[int, ...] = (2, 3)
    run_event_mean_removed_sensitivity: bool = True
    run_reverse_classification: bool = True
    run_markov_order_sensitivity: bool = True
    run_semigroup_sensitivity: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("seed", self.seed),
            ("rank_cap", self.rank_cap),
            ("time_bin_milliseconds", self.time_bin_milliseconds),
            ("observation_window_bins", self.observation_window_bins),
            ("lag_bins", self.lag_bins),
            ("primary_stride_bins", self.primary_stride_bins),
            ("global_anchor_depth", self.global_anchor_depth),
            ("outer_fold_count", self.outer_fold_count),
            ("covariance_oof_fold_count", self.covariance_oof_fold_count),
            ("state_count", self.state_count),
            ("kmeans_restarts", self.kmeans_restarts),
            ("kmeans_max_iterations", self.kmeans_max_iterations),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.outer_fold_count < 2 or self.covariance_oof_fold_count < 2:
            raise ValueError("outer and covariance OOF fold counts must be at least two")
        if self.observation_window_bins != 10:
            raise ValueError("v1 fixes the observation window width to 10 bins")
        if self.time_bin_milliseconds != 10:
            raise ValueError("v1 fixes the processed time-bin width to 10 ms")
        if self.lag_bins < self.observation_window_bins:
            raise ValueError("prediction lag must cover the observation window")
        if self.primary_stride_bins < self.observation_window_bins:
            raise ValueError(
                "primary stride must not reuse overlapping observation windows"
            )
        if self.global_anchor_depth < max(self.markov_orders):
            raise ValueError("global anchor depth must cover all Markov orders")
        if self.state_count != 2:
            raise ValueError("v1 fixes the current-only state scale gate to K=2")
        if float(self.ridge_alpha) != 1.0:
            raise ValueError("v1 fixes drift ridge_alpha to 1")
        if float(self.full_covariance_shrinkage) != 0.1:
            raise ValueError("v1 fixes full covariance shrinkage to 0.1")
        if float(self.minimum_codelength_advantage_bits_per_scalar) != 0.01:
            raise ValueError("v1 fixes the codelength advantage gate to 0.01")
        if float(self.minimum_session_unit_win_fraction) != 0.5:
            raise ValueError("v1 fixes the session-unit win-fraction gate to 0.5")
        if float(self.semigroup_max_excess_bits_per_scalar) != 0.02:
            raise ValueError("v1 fixes the semigroup excess tolerance to 0.02")
        if self.markov_orders != (1, 2, 3):
            raise ValueError("v1 fixes the Markov sensitivity to orders 1, 2, and 3")
        if self.semigroup_horizons != (2, 3):
            raise ValueError("v1 fixes semigroup horizons to 200 and 300 ms")


@dataclass(frozen=True)
class DriftModel:
    """Frozen conditional-mean model."""

    family: str
    order: int
    latent_rank: int
    coefficients: np.ndarray
    parameter_count: int


@dataclass(frozen=True)
class OOFResiduals:
    """Outer-training, trial-wise cross-fitted residual vectors."""

    residuals: np.ndarray
    current: np.ndarray
    anchor_indices: np.ndarray
    trial_indices: np.ndarray
    fold_count: int
    every_trial_held_out_once: bool

    @property
    def vector_count(self) -> int:
        return int(self.residuals.shape[0])


@dataclass(frozen=True)
class NoiseModel:
    """Frozen covariance shape and optional time/state scales."""

    family: str
    base_covariance: np.ndarray
    scale_keys: tuple[int, ...]
    scales: tuple[float, ...]
    covariance_parameter_count: int
    gate_parameter_count: int
    state_gate: PastOnlyGate | None
    model_selection_bits: float


@dataclass(frozen=True)
class GaussianCodelengthResult:
    """Held-out multivariate Gaussian/BIC proxy."""

    family: str
    oof_train_vector_count: int
    bic_reference_vector_count: int
    test_vector_count: int
    latent_rank: int
    drift_parameter_count: int
    covariance_parameter_count: int
    gate_parameter_count: int
    model_selection_bits: float
    heldout_multivariate_gaussian_nll_bits: float
    bic_parameter_bits: float
    total_codelength_bits: float
    bits_per_test_vector: float
    bits_per_test_scalar: float
    test_sse: float


@dataclass(frozen=True)
class MarkovOrderSensitivity:
    """One order's common-anchor, full-covariance score."""

    order: int
    score: GaussianCodelengthResult
    common_anchor_depth: int
    used_in_primary_diffusion_gate: bool


@dataclass(frozen=True)
class SemigroupSensitivity:
    """Frozen 100 ms affine semigroup versus direct long-lag refit."""

    horizon_steps: int
    horizon_milliseconds: int
    test_vector_count: int
    direct_oof_train_vector_count: int
    frozen_oof_train_vector_count: int
    bic_reference_vector_count: int
    shared_parameter_count: int
    shared_bic_parameter_bits: float
    frozen_score: GaussianCodelengthResult
    direct_score: GaussianCodelengthResult
    frozen_semigroup_sse: float
    direct_refit_sse: float
    frozen_semigroup_bits_per_scalar: float
    direct_refit_bits_per_scalar: float
    frozen_advantage_over_direct_bits_per_scalar: float
    frozen_excess_over_direct_bits_per_scalar: float
    frozen_semigroup_within_tolerance: bool
    used_in_primary_diffusion_gate: bool


@dataclass(frozen=True)
class DirectionClassification:
    """Forward/reverse descriptive label; never a diffusion gate."""

    forward_full_bits_per_scalar: float
    reverse_full_bits_per_scalar: float
    forward_full_total_bits: float
    reverse_full_total_bits: float
    test_scalar_count: int
    lower_code_direction: str
    used_in_primary_diffusion_gate: bool


@dataclass(frozen=True)
class DiffusionFoldResult:
    """All low-cost comparisons for one outer fold."""

    fold_index_zero_based: int
    latent_rank: int
    active_neuron_count: int
    scores: tuple[GaussianCodelengthResult, ...]
    markov_order_sensitivity: tuple[MarkovOrderSensitivity, ...]
    semigroup_sensitivity: tuple[SemigroupSensitivity, ...]
    direction_classification: DirectionClassification | None
    common_outer_test_vector_count: int
    covariance_fit_from_outer_train_trial_oof: bool
    outer_test_used_for_covariance_or_gate: bool
    state_gate_uses_current_only: bool
    d1_d3_rows_treated_as_paired_trials: bool

    def score(self, family: str) -> GaussianCodelengthResult:
        matches = tuple(item for item in self.scores if item.family == family)
        if len(matches) != 1:
            raise KeyError(family)
        return matches[0]


@dataclass(frozen=True)
class DiffusionUnitResult:
    """One session x dimension x preprocessing-sensitivity unit."""

    analysis_key: str
    session_index_one_based: int
    animal: str
    neuron_count: int
    dimension: int
    event_mean_removed: bool
    fold_results: tuple[DiffusionFoldResult, ...]
    test_scalar_count: int
    state_advantage_over_full_bits_per_scalar: float
    state_advantage_over_time_bits_per_scalar: float
    state_advantage_over_quadratic_bits_per_scalar: float
    state_beats_all_three_controls: bool
    complete_outer_folds: bool
    markov_order_vote: int | None
    direction_vote: str | None


@dataclass(frozen=True)
class DiffusionAggregateResult:
    """Session-level aggregate; vectors, anchors, and neurons are not replicates."""

    event_mean_removed: bool
    animal: str
    unit_count: int
    all_units_complete: bool
    median_state_advantage_over_full_bits_per_scalar: float
    median_state_advantage_over_time_bits_per_scalar: float
    median_state_advantage_over_quadratic_bits_per_scalar: float
    joint_state_survivor_win_fraction: float


@dataclass(frozen=True)
class DiffusionSessionCheckpoint:
    """Serializable, independently computable physical-session checkpoint."""

    schema_version: str
    config_fingerprint: str
    source_file_md5: str | None
    session: SessionSpec
    results: tuple[DiffusionUnitResult, ...]
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimVerdict:
    key: str
    answer: str
    reason: str


@dataclass(frozen=True)
class DiffusionClaimLocks:
    """Claims unavailable from the processed observational snapshot."""

    labels_or_responses_used: bool = False
    all_factors_used: bool = False
    dimension_two_used: bool = False
    saved_classifier_test_set_used: bool = False
    full_pseudopopulation_fit: bool = False
    d1_d3_rows_treated_as_paired_trials: bool = False
    biological_diffusion_identified: bool = False
    generative_reverse_process_identified: bool = False
    score_function_identified: bool = False
    causal_mechanism_identified: bool = False
    spatial_graph_diffusion_identified: bool = False


@dataclass(frozen=True)
class TafazoliDiffusionProbeReport:
    """Serializable full report assembled from session checkpoints."""

    schema_version: str
    scope: str
    method_status: str
    source_file_md5: str | None
    official_checksum_verified: bool
    config: DiffusionProbeConfig
    session_specs: tuple[SessionSpec, ...]
    fields_used_for_fitting: tuple[str, ...]
    blind_fields_used: tuple[str, ...]
    saved_test_role: str
    train_only_preprocessing: bool
    primary_inference_unit: str
    codelength_name: str
    checkpoints: tuple[DiffusionSessionCheckpoint, ...]
    results: tuple[DiffusionUnitResult, ...]
    aggregates: tuple[DiffusionAggregateResult, ...]
    verdicts: tuple[ClaimVerdict, ...]
    claim_locks: DiffusionClaimLocks
    limitations: tuple[str, ...]
    conclusion: str

    def verdict(self, key: str) -> ClaimVerdict:
        matches = tuple(item for item in self.verdicts if item.key == key)
        if len(matches) != 1:
            raise KeyError(key)
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _derived_seed(base_seed: int, *tokens: Any) -> int:
    payload = json.dumps(
        (SCHEMA_VERSION, int(base_seed), *tokens),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:16],
        byteorder="little",
        signed=False,
    )


def config_fingerprint(config: DiffusionProbeConfig) -> str:
    payload = json.dumps(
        {
            "config": asdict(config),
            "implementation_revision": IMPLEMENTATION_REVISION,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ridge_coefficients(
    design: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("ridge design and target must be aligned matrices")
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.pinv(x.T @ x + penalty) @ x.T @ y


def _quadratic_terms(current: np.ndarray) -> np.ndarray:
    values = np.asarray(current, dtype=np.float64)
    blocks = [
        values[:, left] * values[:, right]
        for left in range(values.shape[1])
        for right in range(left, values.shape[1])
    ]
    return np.column_stack(blocks)


def _drift_design_matrix(
    design: TrajectoryDesign,
    *,
    order: int,
    quadratic: bool,
) -> np.ndarray:
    history = design.history.reshape(-1, design.history.shape[-1])
    rank = design.current.shape[-1]
    if history.shape[1] != order * rank:
        raise ValueError("history feature count does not match drift order")
    if quadratic:
        if order != 1:
            raise ValueError("quadratic drift is defined only for order one")
        current = history[:, :rank]
        features = np.column_stack((current, _quadratic_terms(current)))
    else:
        features = history
    return np.column_stack((np.ones(features.shape[0]), features))


def fit_drift_model(
    design: TrajectoryDesign,
    *,
    order: int,
    quadratic: bool,
    ridge_alpha: float = 1.0,
) -> DriftModel:
    """Fit a linear VAR or the fixed linear+quadratic order-one drift."""

    matrix = _drift_design_matrix(
        design,
        order=order,
        quadratic=quadratic,
    )
    target = design.successor.reshape(-1, design.successor.shape[-1])
    coefficients = _ridge_coefficients(matrix, target, alpha=ridge_alpha)
    rank = target.shape[1]
    return DriftModel(
        family=QUADRATIC_DRIFT_FULL_Q if quadratic else f"VAR_ORDER_{order}",
        order=order,
        latent_rank=rank,
        coefficients=np.asarray(coefficients, dtype=np.float64),
        parameter_count=int(coefficients.size),
    )


def apply_drift_model(
    model: DriftModel,
    design: TrajectoryDesign,
) -> np.ndarray:
    quadratic = model.family == QUADRATIC_DRIFT_FULL_Q
    matrix = _drift_design_matrix(
        design,
        order=model.order,
        quadratic=quadratic,
    )
    if matrix.shape[1] != model.coefficients.shape[0]:
        raise ValueError("drift design does not match frozen coefficients")
    return (matrix @ model.coefficients).reshape(design.successor.shape)


def _common_design(
    latent: np.ndarray,
    *,
    order: int,
    config: DiffusionProbeConfig,
    reverse: bool,
) -> TrajectoryDesign:
    return build_trajectory_design(
        latent,
        history_depth=order,
        anchor_history_depth=config.global_anchor_depth,
        lag_bins=config.lag_bins,
        stride_bins=config.primary_stride_bins,
        reverse=reverse,
    )


def _flatten_metadata(design: TrajectoryDesign) -> tuple[np.ndarray, np.ndarray]:
    trials, anchors = design.current.shape[:2]
    return (
        np.tile(design.anchor_indices, trials),
        np.repeat(np.arange(trials, dtype=np.int64), anchors),
    )


def crossfit_drift_residuals(
    outer_training_latent: np.ndarray,
    *,
    order: int,
    quadratic: bool,
    config: DiffusionProbeConfig,
    reverse: bool,
    seed_tokens: tuple[Any, ...] = (),
) -> OOFResiduals:
    """Generate trial-wise OOF residuals in one outer-training coordinate frame."""

    latent = np.asarray(outer_training_latent, dtype=np.float64)
    if latent.ndim != 3 or not np.all(np.isfinite(latent)):
        raise ValueError("outer_training_latent must be finite trial x time x latent")
    folds = make_whole_trial_folds(
        latent.shape[0],
        fold_count=config.covariance_oof_fold_count,
        seed=_derived_seed(config.seed, *seed_tokens, "covariance_oof"),
    )
    residual_blocks = []
    current_blocks = []
    anchor_blocks = []
    trial_blocks = []
    held_out_trials = []
    for fold in folds:
        train_indices = np.asarray(fold.train_indices, dtype=np.int64)
        test_indices = np.asarray(fold.test_indices, dtype=np.int64)
        train_design = _common_design(
            latent[train_indices],
            order=order,
            config=config,
            reverse=reverse,
        )
        test_design = _common_design(
            latent[test_indices],
            order=order,
            config=config,
            reverse=reverse,
        )
        drift = fit_drift_model(
            train_design,
            order=order,
            quadratic=quadratic,
            ridge_alpha=config.ridge_alpha,
        )
        prediction = apply_drift_model(drift, test_design)
        residual_blocks.append(
            (test_design.successor - prediction).reshape(
                -1,
                test_design.successor.shape[-1],
            )
        )
        current_blocks.append(
            test_design.current.reshape(-1, test_design.current.shape[-1])
        )
        anchor_blocks.append(
            np.tile(test_design.anchor_indices, test_design.current.shape[0])
        )
        trial_blocks.append(
            np.repeat(test_indices, test_design.current.shape[1])
        )
        held_out_trials.extend(int(value) for value in test_indices)
    every_once = tuple(sorted(held_out_trials)) == tuple(range(latent.shape[0]))
    if not every_once:
        raise RuntimeError("covariance OOF folds did not hold every trial out once")
    residuals = np.concatenate(residual_blocks, axis=0)
    current = np.concatenate(current_blocks, axis=0)
    anchors = np.concatenate(anchor_blocks, axis=0)
    trials = np.concatenate(trial_blocks, axis=0)
    if not (
        residuals.shape[0]
        == current.shape[0]
        == anchors.size
        == trials.size
    ):
        raise RuntimeError("OOF residual metadata does not align")
    return OOFResiduals(
        residuals=np.asarray(residuals, dtype=np.float64),
        current=np.asarray(current, dtype=np.float64),
        anchor_indices=np.asarray(anchors, dtype=np.int64),
        trial_indices=np.asarray(trials, dtype=np.int64),
        fold_count=len(folds),
        every_trial_held_out_once=every_once,
    )


def _second_moment(residuals: np.ndarray) -> np.ndarray:
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("residuals must be a non-empty matrix")
    return values.T @ values / values.shape[0]


def _stabilize_covariance(covariance: np.ndarray) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    mean_variance = max(
        float(np.trace(symmetric) / symmetric.shape[0]),
        np.finfo(np.float64).eps,
    )
    floor = max(mean_variance * 1e-8, 1e-10)
    stabilized = (
        eigenvectors
        @ np.diag(np.maximum(eigenvalues, floor))
        @ eigenvectors.T
    )
    return np.asarray(0.5 * (stabilized + stabilized.T), dtype=np.float64)


def _full_shrink_covariance(
    residuals: np.ndarray,
    *,
    shrinkage: float,
) -> np.ndarray:
    empirical = _second_moment(residuals)
    rank = empirical.shape[0]
    isotropic = np.eye(rank) * float(np.trace(empirical) / rank)
    return _stabilize_covariance(
        (1.0 - shrinkage) * empirical + shrinkage * isotropic
    )


def _mahalanobis_scale(
    residuals: np.ndarray,
    covariance: np.ndarray,
) -> float:
    values = np.asarray(residuals, dtype=np.float64)
    stabilized = _stabilize_covariance(covariance)
    cholesky = np.linalg.cholesky(stabilized)
    solved = np.linalg.solve(cholesky, values.T)
    quadratic = np.sum(np.square(solved), axis=0)
    return max(
        float(np.mean(quadratic) / values.shape[1]),
        1e-8,
    )


def _normalized_group_scales(
    residuals: np.ndarray,
    keys: np.ndarray,
    base_covariance: np.ndarray,
) -> tuple[tuple[int, ...], tuple[float, ...], np.ndarray]:
    labels = np.asarray(keys, dtype=np.int64).reshape(-1)
    unique = tuple(int(value) for value in sorted(np.unique(labels)))
    raw = []
    weights = []
    for key in unique:
        mask = labels == key
        if not np.any(mask):
            raise RuntimeError("scale group is empty")
        raw.append(_mahalanobis_scale(residuals[mask], base_covariance))
        weights.append(int(np.count_nonzero(mask)))
    log_normalizer = float(
        np.average(
            np.log(np.asarray(raw, dtype=np.float64)),
            weights=np.asarray(weights, dtype=np.float64),
        )
    )
    normalizer = float(np.exp(log_normalizer))
    normalized = tuple(float(value / normalizer) for value in raw)
    return unique, normalized, _stabilize_covariance(
        base_covariance * normalizer
    )


def _quadratic_and_logdet(
    residuals: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, float]:
    stabilized = _stabilize_covariance(covariance)
    cholesky = np.linalg.cholesky(stabilized)
    solved = np.linalg.solve(
        cholesky,
        np.asarray(residuals, dtype=np.float64).T,
    )
    quadratic = np.sum(np.square(solved), axis=0)
    logdet = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
    return np.asarray(quadratic, dtype=np.float64), logdet


def fit_noise_model(
    family: str,
    oof: OOFResiduals,
    *,
    config: DiffusionProbeConfig,
    full_training_current: np.ndarray | None = None,
    seed_tokens: tuple[Any, ...] = (),
) -> NoiseModel:
    """Fit one covariance family from outer-training OOF residuals only."""

    residuals = np.asarray(oof.residuals, dtype=np.float64)
    rank = residuals.shape[1]
    family_bits = log2(float(len(PRIMARY_FAMILIES)))
    if family == OU_ISO:
        variance = max(
            float(np.mean(np.square(residuals))),
            np.finfo(np.float64).eps,
        )
        return NoiseModel(
            family=family,
            base_covariance=np.eye(rank) * variance,
            scale_keys=(),
            scales=(),
            covariance_parameter_count=1,
            gate_parameter_count=0,
            state_gate=None,
            model_selection_bits=family_bits,
        )
    if family == OU_DIAG:
        variances = np.maximum(
            np.mean(np.square(residuals), axis=0),
            np.finfo(np.float64).eps,
        )
        return NoiseModel(
            family=family,
            base_covariance=np.diag(variances),
            scale_keys=(),
            scales=(),
            covariance_parameter_count=rank,
            gate_parameter_count=0,
            state_gate=None,
            model_selection_bits=family_bits,
        )
    base = _full_shrink_covariance(
        residuals,
        shrinkage=config.full_covariance_shrinkage,
    )
    full_parameters = rank * (rank + 1) // 2
    if family in (OU_FULL, QUADRATIC_DRIFT_FULL_Q):
        return NoiseModel(
            family=family,
            base_covariance=base,
            scale_keys=(),
            scales=(),
            covariance_parameter_count=full_parameters,
            gate_parameter_count=0,
            state_gate=None,
            model_selection_bits=family_bits,
        )
    if family == TIME_SCALE:
        keys, scales, adjusted = _normalized_group_scales(
            residuals,
            oof.anchor_indices,
            base,
        )
        return NoiseModel(
            family=family,
            base_covariance=adjusted,
            scale_keys=keys,
            scales=scales,
            covariance_parameter_count=full_parameters + max(len(keys) - 1, 0),
            gate_parameter_count=0,
            state_gate=None,
            model_selection_bits=family_bits,
        )
    if family != STATE_SCALE:
        raise ValueError("unknown covariance family")
    if full_training_current is None:
        raise ValueError("STATE_SCALE requires full outer-training current states")
    gate_values = np.asarray(full_training_current, dtype=np.float64)
    if gate_values.ndim != 3 or gate_values.shape[-1] != rank:
        raise ValueError("state gate training tensor must be trial x anchor x latent")
    gate = fit_past_only_gate(
        gate_values,
        state_count=config.state_count,
        history_depth=1,
        latent_rank=rank,
        seed=_derived_seed(config.seed, *seed_tokens, "state_scale_gate"),
        restarts=config.kmeans_restarts,
        max_iterations=config.kmeans_max_iterations,
    )
    oof_states = assign_past_only_states(
        gate,
        oof.current.reshape(1, oof.current.shape[0], rank),
    ).reshape(-1)
    keys, scales, adjusted = _normalized_group_scales(
        residuals,
        oof_states,
        base,
    )
    return NoiseModel(
        family=family,
        base_covariance=adjusted,
        scale_keys=keys,
        scales=scales,
        covariance_parameter_count=full_parameters + config.state_count - 1,
        gate_parameter_count=config.state_count * rank,
        state_gate=gate,
        model_selection_bits=family_bits
        + log2(float(factorial(config.state_count))),
    )


def _sample_scale_keys(
    noise: NoiseModel,
    *,
    anchor_indices: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    if noise.family == TIME_SCALE:
        return np.asarray(anchor_indices, dtype=np.int64).reshape(-1)
    if noise.family == STATE_SCALE:
        if noise.state_gate is None:
            raise ValueError("state scale model has no frozen gate")
        values = np.asarray(current, dtype=np.float64)
        return assign_past_only_states(
            noise.state_gate,
            values.reshape(1, values.shape[0], values.shape[1]),
        ).reshape(-1)
    return np.zeros(np.asarray(current).shape[0], dtype=np.int64)


def _multivariate_gaussian_nll_bits(
    residuals: np.ndarray,
    noise: NoiseModel,
    *,
    anchor_indices: np.ndarray,
    current: np.ndarray,
) -> float:
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("residuals must be a matrix")
    if not noise.scale_keys:
        quadratic, logdet = _quadratic_and_logdet(
            values,
            noise.base_covariance,
        )
        nll_nats = 0.5 * np.sum(
            values.shape[1] * np.log(2.0 * pi) + logdet + quadratic
        )
        return float(nll_nats / np.log(2.0))
    sample_keys = _sample_scale_keys(
        noise,
        anchor_indices=anchor_indices,
        current=current,
    )
    scale_lookup = dict(zip(noise.scale_keys, noise.scales))
    total_nats = 0.0
    for key in noise.scale_keys:
        mask = sample_keys == key
        if not np.any(mask):
            continue
        quadratic, logdet = _quadratic_and_logdet(
            values[mask],
            noise.base_covariance * scale_lookup[key],
        )
        total_nats += 0.5 * float(
            np.sum(
                values.shape[1] * np.log(2.0 * pi)
                + logdet
                + quadratic
            )
        )
    unknown = ~np.isin(sample_keys, np.asarray(noise.scale_keys))
    if np.any(unknown):
        raise ValueError("test scale key was absent from outer training")
    return float(total_nats / np.log(2.0))


def score_multivariate_gaussian(
    family: str,
    test_residuals: np.ndarray,
    noise: NoiseModel,
    *,
    oof_train_vector_count: int,
    bic_reference_vector_count: int | None = None,
    drift_parameter_count: int,
    anchor_indices: np.ndarray,
    current: np.ndarray,
) -> GaussianCodelengthResult:
    """Score fixed residual covariance on untouched outer-test vectors."""

    residuals = np.asarray(test_residuals, dtype=np.float64)
    if residuals.ndim != 2 or residuals.shape[0] < 1:
        raise ValueError("test residuals must be a non-empty matrix")
    if residuals.shape[1] != noise.base_covariance.shape[0]:
        raise ValueError("test residual rank does not match noise model")
    if np.asarray(anchor_indices).size != residuals.shape[0]:
        raise ValueError("anchor metadata does not align with test residuals")
    if np.asarray(current).shape != residuals.shape:
        raise ValueError("current states do not align with test residuals")
    nll_bits = _multivariate_gaussian_nll_bits(
        residuals,
        noise,
        anchor_indices=anchor_indices,
        current=current,
    )
    parameter_count = (
        drift_parameter_count
        + noise.covariance_parameter_count
        + noise.gate_parameter_count
    )
    reference_count = (
        int(oof_train_vector_count)
        if bic_reference_vector_count is None
        else int(bic_reference_vector_count)
    )
    if oof_train_vector_count < 1 or reference_count < 1:
        raise ValueError("training and BIC reference vector counts must be positive")
    bic_bits = 0.5 * parameter_count * log2(float(max(reference_count, 2)))
    total = nll_bits + bic_bits + noise.model_selection_bits
    return GaussianCodelengthResult(
        family=family,
        oof_train_vector_count=int(oof_train_vector_count),
        bic_reference_vector_count=reference_count,
        test_vector_count=int(residuals.shape[0]),
        latent_rank=int(residuals.shape[1]),
        drift_parameter_count=int(drift_parameter_count),
        covariance_parameter_count=noise.covariance_parameter_count,
        gate_parameter_count=noise.gate_parameter_count,
        model_selection_bits=noise.model_selection_bits,
        heldout_multivariate_gaussian_nll_bits=nll_bits,
        bic_parameter_bits=float(bic_bits),
        total_codelength_bits=float(total),
        bits_per_test_vector=float(total / residuals.shape[0]),
        bits_per_test_scalar=float(total / residuals.size),
        test_sse=float(np.sum(np.square(residuals), dtype=np.float64)),
    )


@dataclass(frozen=True)
class _PrimaryContext:
    train_design: TrajectoryDesign
    test_design: TrajectoryDesign
    linear_drift: DriftModel
    linear_oof: OOFResiduals
    full_noise: NoiseModel
    linear_test_residuals: np.ndarray
    test_anchor_indices: np.ndarray
    test_current: np.ndarray


def _test_residual_metadata(
    design: TrajectoryDesign,
    prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residuals = (design.successor - prediction).reshape(
        -1,
        design.successor.shape[-1],
    )
    anchors, _ = _flatten_metadata(design)
    current = design.current.reshape(-1, design.current.shape[-1])
    return residuals, anchors, current


def _evaluate_primary_ladder(
    train_latent: np.ndarray,
    test_latent: np.ndarray,
    *,
    config: DiffusionProbeConfig,
    seed_tokens: tuple[Any, ...],
) -> tuple[tuple[GaussianCodelengthResult, ...], _PrimaryContext]:
    train_design = _common_design(
        train_latent,
        order=1,
        config=config,
        reverse=False,
    )
    test_design = _common_design(
        test_latent,
        order=1,
        config=config,
        reverse=False,
    )
    linear_oof = crossfit_drift_residuals(
        train_latent,
        order=1,
        quadratic=False,
        config=config,
        reverse=False,
        seed_tokens=(*seed_tokens, "linear"),
    )
    linear_drift = fit_drift_model(
        train_design,
        order=1,
        quadratic=False,
        ridge_alpha=config.ridge_alpha,
    )
    linear_prediction = apply_drift_model(linear_drift, test_design)
    linear_residuals, anchors, current = _test_residual_metadata(
        test_design,
        linear_prediction,
    )
    scores = []
    noises: dict[str, NoiseModel] = {}
    for family in (OU_ISO, OU_DIAG, OU_FULL, TIME_SCALE, STATE_SCALE):
        noise = fit_noise_model(
            family,
            linear_oof,
            config=config,
            full_training_current=(
                train_design.current if family == STATE_SCALE else None
            ),
            seed_tokens=(*seed_tokens, family),
        )
        noises[family] = noise
        scores.append(
            score_multivariate_gaussian(
                family,
                linear_residuals,
                noise,
                oof_train_vector_count=linear_oof.vector_count,
                drift_parameter_count=linear_drift.parameter_count,
                anchor_indices=anchors,
                current=current,
            )
        )

    quadratic_oof = crossfit_drift_residuals(
        train_latent,
        order=1,
        quadratic=True,
        config=config,
        reverse=False,
        seed_tokens=(*seed_tokens, "linear"),
    )
    quadratic_drift = fit_drift_model(
        train_design,
        order=1,
        quadratic=True,
        ridge_alpha=config.ridge_alpha,
    )
    quadratic_prediction = apply_drift_model(quadratic_drift, test_design)
    quadratic_residuals, quadratic_anchors, quadratic_current = (
        _test_residual_metadata(test_design, quadratic_prediction)
    )
    quadratic_noise = fit_noise_model(
        QUADRATIC_DRIFT_FULL_Q,
        quadratic_oof,
        config=config,
    )
    scores.append(
        score_multivariate_gaussian(
            QUADRATIC_DRIFT_FULL_Q,
            quadratic_residuals,
            quadratic_noise,
            oof_train_vector_count=quadratic_oof.vector_count,
            drift_parameter_count=quadratic_drift.parameter_count,
            anchor_indices=quadratic_anchors,
            current=quadratic_current,
        )
    )
    vector_counts = {item.test_vector_count for item in scores}
    if len(vector_counts) != 1:
        raise RuntimeError("primary covariance ladder used different outer targets")
    context = _PrimaryContext(
        train_design=train_design,
        test_design=test_design,
        linear_drift=linear_drift,
        linear_oof=linear_oof,
        full_noise=noises[OU_FULL],
        linear_test_residuals=linear_residuals,
        test_anchor_indices=anchors,
        test_current=current,
    )
    return tuple(scores), context


def _evaluate_full_direction(
    train_latent: np.ndarray,
    test_latent: np.ndarray,
    *,
    config: DiffusionProbeConfig,
    reverse: bool,
    seed_tokens: tuple[Any, ...],
) -> GaussianCodelengthResult:
    train_design = _common_design(
        train_latent,
        order=1,
        config=config,
        reverse=reverse,
    )
    test_design = _common_design(
        test_latent,
        order=1,
        config=config,
        reverse=reverse,
    )
    oof = crossfit_drift_residuals(
        train_latent,
        order=1,
        quadratic=False,
        config=config,
        reverse=reverse,
        seed_tokens=seed_tokens,
    )
    drift = fit_drift_model(
        train_design,
        order=1,
        quadratic=False,
        ridge_alpha=config.ridge_alpha,
    )
    prediction = apply_drift_model(drift, test_design)
    residuals, anchors, current = _test_residual_metadata(
        test_design,
        prediction,
    )
    noise = fit_noise_model(OU_FULL, oof, config=config)
    return score_multivariate_gaussian(
        OU_FULL,
        residuals,
        noise,
        oof_train_vector_count=oof.vector_count,
        drift_parameter_count=drift.parameter_count,
        anchor_indices=anchors,
        current=current,
    )


def _markov_sensitivity(
    train_latent: np.ndarray,
    test_latent: np.ndarray,
    *,
    primary_full: GaussianCodelengthResult,
    config: DiffusionProbeConfig,
    seed_tokens: tuple[Any, ...],
) -> tuple[MarkovOrderSensitivity, ...]:
    if not config.run_markov_order_sensitivity:
        return ()
    results = []
    for order in config.markov_orders:
        if order == 1:
            score = primary_full
        else:
            train_design = _common_design(
                train_latent,
                order=order,
                config=config,
                reverse=False,
            )
            test_design = _common_design(
                test_latent,
                order=order,
                config=config,
                reverse=False,
            )
            oof = crossfit_drift_residuals(
                train_latent,
                order=order,
                quadratic=False,
                config=config,
                reverse=False,
                seed_tokens=(*seed_tokens, "linear"),
            )
            drift = fit_drift_model(
                train_design,
                order=order,
                quadratic=False,
                ridge_alpha=config.ridge_alpha,
            )
            prediction = apply_drift_model(drift, test_design)
            residuals, anchors, current = _test_residual_metadata(
                test_design,
                prediction,
            )
            noise = fit_noise_model(OU_FULL, oof, config=config)
            score = score_multivariate_gaussian(
                f"MARKOV_ORDER_{order}_FULL",
                residuals,
                noise,
                oof_train_vector_count=oof.vector_count,
                drift_parameter_count=drift.parameter_count,
                anchor_indices=anchors,
                current=current,
            )
        results.append(
            MarkovOrderSensitivity(
                order=order,
                score=score,
                common_anchor_depth=config.global_anchor_depth,
                used_in_primary_diffusion_gate=False,
            )
        )
    if len({item.score.test_vector_count for item in results}) != 1:
        raise RuntimeError("Markov orders did not use common outer targets")
    return tuple(results)


def _horizon_design(
    latent: np.ndarray,
    *,
    horizon_steps: int,
    config: DiffusionProbeConfig,
    reverse: bool,
) -> TrajectoryDesign:
    values = np.asarray(latent, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("latent must be trial x time x latent")
    if reverse:
        values = values[:, ::-1, :]
    first_anchor = (config.global_anchor_depth - 1) * config.lag_bins
    horizon_bins = horizon_steps * config.lag_bins
    stop_exclusive = values.shape[1] - horizon_bins
    if first_anchor >= stop_exclusive:
        raise ValueError("trajectory is too short for the requested horizon")
    anchors = np.arange(
        first_anchor,
        stop_exclusive,
        config.primary_stride_bins,
        dtype=np.int64,
    )
    current = values[:, anchors, :]
    return TrajectoryDesign(
        history=np.asarray(current, dtype=np.float64),
        current=np.asarray(current, dtype=np.float64),
        successor=np.asarray(values[:, anchors + horizon_bins, :], dtype=np.float64),
        anchor_indices=anchors,
    )


def _crossfit_horizon_residuals(
    train_latent: np.ndarray,
    *,
    horizon_steps: int,
    config: DiffusionProbeConfig,
    seed_tokens: tuple[Any, ...],
) -> OOFResiduals:
    latent = np.asarray(train_latent, dtype=np.float64)
    folds = make_whole_trial_folds(
        latent.shape[0],
        fold_count=config.covariance_oof_fold_count,
        seed=_derived_seed(
            config.seed,
            *seed_tokens,
            "horizon_oof",
            horizon_steps,
        ),
    )
    residual_blocks = []
    current_blocks = []
    anchor_blocks = []
    trial_blocks = []
    held_out = []
    for fold in folds:
        train_indices = np.asarray(fold.train_indices, dtype=np.int64)
        test_indices = np.asarray(fold.test_indices, dtype=np.int64)
        train_design = _horizon_design(
            latent[train_indices],
            horizon_steps=horizon_steps,
            config=config,
            reverse=False,
        )
        test_design = _horizon_design(
            latent[test_indices],
            horizon_steps=horizon_steps,
            config=config,
            reverse=False,
        )
        drift = fit_drift_model(
            train_design,
            order=1,
            quadratic=False,
            ridge_alpha=config.ridge_alpha,
        )
        residual_blocks.append(
            (test_design.successor - apply_drift_model(drift, test_design)).reshape(
                -1,
                test_design.successor.shape[-1],
            )
        )
        current_blocks.append(
            test_design.current.reshape(-1, test_design.current.shape[-1])
        )
        anchor_blocks.append(
            np.tile(test_design.anchor_indices, test_design.current.shape[0])
        )
        trial_blocks.append(
            np.repeat(test_indices, test_design.current.shape[1])
        )
        held_out.extend(int(value) for value in test_indices)
    every_once = tuple(sorted(held_out)) == tuple(range(latent.shape[0]))
    if not every_once:
        raise RuntimeError("horizon OOF folds did not cover every trial once")
    return OOFResiduals(
        residuals=np.concatenate(residual_blocks, axis=0),
        current=np.concatenate(current_blocks, axis=0),
        anchor_indices=np.concatenate(anchor_blocks, axis=0),
        trial_indices=np.concatenate(trial_blocks, axis=0),
        fold_count=len(folds),
        every_trial_held_out_once=True,
    )


def _semigroup_prediction(
    drift: DriftModel,
    current: np.ndarray,
    *,
    horizon_steps: int,
) -> np.ndarray:
    if drift.order != 1 or drift.family == QUADRATIC_DRIFT_FULL_Q:
        raise ValueError("semigroup requires the affine VAR(1) drift")
    values = np.asarray(current, dtype=np.float64)
    intercept = drift.coefficients[0]
    operator = drift.coefficients[1:]
    prediction = np.array(values, copy=True)
    for _ in range(horizon_steps):
        prediction = intercept + prediction @ operator
    return prediction


def _propagated_covariance(
    drift: DriftModel,
    one_step_covariance: np.ndarray,
    *,
    horizon_steps: int,
) -> np.ndarray:
    operator = drift.coefficients[1:]
    base = _stabilize_covariance(one_step_covariance)
    total = np.zeros_like(base)
    power = np.eye(base.shape[0])
    for _ in range(horizon_steps):
        total += power.T @ base @ power
        power = power @ operator
    return _stabilize_covariance(total)


def _semigroup_sensitivity(
    train_latent: np.ndarray,
    test_latent: np.ndarray,
    *,
    primary: _PrimaryContext,
    config: DiffusionProbeConfig,
    seed_tokens: tuple[Any, ...],
) -> tuple[SemigroupSensitivity, ...]:
    if not config.run_semigroup_sensitivity:
        return ()
    results = []
    for horizon in config.semigroup_horizons:
        train_design = _horizon_design(
            train_latent,
            horizon_steps=horizon,
            config=config,
            reverse=False,
        )
        test_design = _horizon_design(
            test_latent,
            horizon_steps=horizon,
            config=config,
            reverse=False,
        )
        direct_oof = _crossfit_horizon_residuals(
            train_latent,
            horizon_steps=horizon,
            config=config,
            seed_tokens=(*seed_tokens, horizon),
        )
        direct_drift = fit_drift_model(
            train_design,
            order=1,
            quadratic=False,
            ridge_alpha=config.ridge_alpha,
        )
        direct_prediction = apply_drift_model(direct_drift, test_design)
        direct_residuals, anchors, current = _test_residual_metadata(
            test_design,
            direct_prediction,
        )
        direct_noise = fit_noise_model(OU_FULL, direct_oof, config=config)
        direct_noise = NoiseModel(
            family=direct_noise.family,
            base_covariance=direct_noise.base_covariance,
            scale_keys=direct_noise.scale_keys,
            scales=direct_noise.scales,
            covariance_parameter_count=direct_noise.covariance_parameter_count,
            gate_parameter_count=direct_noise.gate_parameter_count,
            state_gate=direct_noise.state_gate,
            model_selection_bits=log2(2.0),
        )
        comparison_train_vector_count = direct_oof.vector_count
        direct_score = score_multivariate_gaussian(
            f"DIRECT_{horizon}_STEP_FULL",
            direct_residuals,
            direct_noise,
            oof_train_vector_count=direct_oof.vector_count,
            bic_reference_vector_count=comparison_train_vector_count,
            drift_parameter_count=direct_drift.parameter_count,
            anchor_indices=anchors,
            current=current,
        )

        flat_current = test_design.current.reshape(
            -1,
            test_design.current.shape[-1],
        )
        frozen_prediction = _semigroup_prediction(
            primary.linear_drift,
            flat_current,
            horizon_steps=horizon,
        )
        observed = test_design.successor.reshape(
            -1,
            test_design.successor.shape[-1],
        )
        frozen_residuals = observed - frozen_prediction
        propagated_noise = NoiseModel(
            family=f"FROZEN_SEMIGROUP_{horizon}",
            base_covariance=_propagated_covariance(
                primary.linear_drift,
                primary.full_noise.base_covariance,
                horizon_steps=horizon,
            ),
            scale_keys=(),
            scales=(),
            covariance_parameter_count=primary.full_noise.covariance_parameter_count,
            gate_parameter_count=0,
            state_gate=None,
            model_selection_bits=log2(2.0),
        )
        frozen_score = score_multivariate_gaussian(
            f"FROZEN_SEMIGROUP_{horizon}",
            frozen_residuals,
            propagated_noise,
            oof_train_vector_count=primary.linear_oof.vector_count,
            bic_reference_vector_count=comparison_train_vector_count,
            drift_parameter_count=primary.linear_drift.parameter_count,
            anchor_indices=anchors,
            current=current,
        )
        if frozen_score.test_vector_count != direct_score.test_vector_count:
            raise RuntimeError("semigroup and direct refit used different targets")
        if (
            frozen_score.drift_parameter_count
            != direct_score.drift_parameter_count
            or frozen_score.covariance_parameter_count
            != direct_score.covariance_parameter_count
            or frozen_score.gate_parameter_count
            != direct_score.gate_parameter_count
            or frozen_score.model_selection_bits
            != direct_score.model_selection_bits
            or frozen_score.bic_parameter_bits
            != direct_score.bic_parameter_bits
        ):
            raise RuntimeError(
                "semigroup comparison candidates must share one complexity basis"
            )
        scalar_count = (
            frozen_score.test_vector_count * frozen_score.latent_rank
        )
        frozen_excess = (
            frozen_score.total_codelength_bits
            - direct_score.total_codelength_bits
        ) / max(scalar_count, 1)
        results.append(
            SemigroupSensitivity(
                horizon_steps=horizon,
                horizon_milliseconds=(
                    horizon
                    * config.lag_bins
                    * config.time_bin_milliseconds
                ),
                test_vector_count=frozen_score.test_vector_count,
                direct_oof_train_vector_count=(
                    direct_score.oof_train_vector_count
                ),
                frozen_oof_train_vector_count=(
                    frozen_score.oof_train_vector_count
                ),
                bic_reference_vector_count=(
                    direct_score.bic_reference_vector_count
                ),
                shared_parameter_count=(
                    direct_score.drift_parameter_count
                    + direct_score.covariance_parameter_count
                    + direct_score.gate_parameter_count
                ),
                shared_bic_parameter_bits=direct_score.bic_parameter_bits,
                frozen_score=frozen_score,
                direct_score=direct_score,
                frozen_semigroup_sse=frozen_score.test_sse,
                direct_refit_sse=direct_score.test_sse,
                frozen_semigroup_bits_per_scalar=frozen_score.bits_per_test_scalar,
                direct_refit_bits_per_scalar=direct_score.bits_per_test_scalar,
                frozen_advantage_over_direct_bits_per_scalar=(
                    direct_score.total_codelength_bits
                    - frozen_score.total_codelength_bits
                )
                / max(scalar_count, 1),
                frozen_excess_over_direct_bits_per_scalar=float(
                    frozen_excess
                ),
                frozen_semigroup_within_tolerance=(
                    frozen_excess
                    <= config.semigroup_max_excess_bits_per_scalar
                ),
                used_in_primary_diffusion_gate=False,
            )
        )
    return tuple(results)


def evaluate_diffusion_fold(
    prepared: PreparedLatentFold,
    *,
    config: DiffusionProbeConfig,
    seed_tokens: tuple[Any, ...] = (),
) -> DiffusionFoldResult:
    """Evaluate one source-train-frozen outer fold."""

    whitened = whiten_prepared_latent_fold(prepared)
    train_latent = whitened.source_train
    test_latent = whitened.target_test
    scores, primary = _evaluate_primary_ladder(
        train_latent,
        test_latent,
        config=config,
        seed_tokens=(*seed_tokens, prepared.fold.index_zero_based),
    )
    score_lookup = {item.family: item for item in scores}
    markov = _markov_sensitivity(
        train_latent,
        test_latent,
        primary_full=score_lookup[OU_FULL],
        config=config,
        seed_tokens=(*seed_tokens, prepared.fold.index_zero_based),
    )
    semigroup = _semigroup_sensitivity(
        train_latent,
        test_latent,
        primary=primary,
        config=config,
        seed_tokens=(*seed_tokens, prepared.fold.index_zero_based),
    )
    if config.run_reverse_classification:
        forward_direction_full = _evaluate_full_direction(
            train_latent,
            test_latent,
            config=config,
            reverse=False,
            seed_tokens=(
                *seed_tokens,
                prepared.fold.index_zero_based,
                "direction",
            ),
        )
        reverse_full = _evaluate_full_direction(
            train_latent,
            test_latent,
            config=config,
            reverse=True,
            seed_tokens=(
                *seed_tokens,
                prepared.fold.index_zero_based,
                "direction",
            ),
        )
        forward_bits = forward_direction_full.bits_per_test_scalar
        reverse_bits = reverse_full.bits_per_test_scalar
        tolerance = 1e-12
        if forward_bits + tolerance < reverse_bits:
            direction_label = "FORWARD_LOWER_CODE"
        elif reverse_bits + tolerance < forward_bits:
            direction_label = "REVERSE_LOWER_CODE"
        else:
            direction_label = "TIE"
        direction = DirectionClassification(
            forward_full_bits_per_scalar=forward_bits,
            reverse_full_bits_per_scalar=reverse_bits,
            forward_full_total_bits=forward_direction_full.total_codelength_bits,
            reverse_full_total_bits=reverse_full.total_codelength_bits,
            test_scalar_count=(
                forward_direction_full.test_vector_count
                * forward_direction_full.latent_rank
            ),
            lower_code_direction=direction_label,
            used_in_primary_diffusion_gate=False,
        )
    else:
        direction = None
    vector_counts = {item.test_vector_count for item in scores}
    if len(vector_counts) != 1:
        raise RuntimeError("primary families used different outer-test targets")
    vector_count = next(iter(vector_counts))
    return DiffusionFoldResult(
        fold_index_zero_based=prepared.fold.index_zero_based,
        latent_rank=prepared.transform.rank,
        active_neuron_count=prepared.transform.active_neuron_count,
        scores=scores,
        markov_order_sensitivity=markov,
        semigroup_sensitivity=semigroup,
        direction_classification=direction,
        common_outer_test_vector_count=vector_count,
        covariance_fit_from_outer_train_trial_oof=True,
        outer_test_used_for_covariance_or_gate=False,
        state_gate_uses_current_only=True,
        d1_d3_rows_treated_as_paired_trials=False,
    )


def _sum_family_scores(
    folds: Sequence[DiffusionFoldResult],
    family: str,
) -> tuple[float, int]:
    scores = tuple(item.score(family) for item in folds)
    return (
        float(sum(item.total_codelength_bits for item in scores)),
        int(
            sum(
                item.test_vector_count * item.latent_rank
                for item in scores
            )
        ),
    )


def summarize_diffusion_unit(
    *,
    session: SessionSpec,
    dimension: int,
    event_mean_removed: bool,
    fold_results: Sequence[DiffusionFoldResult],
    config: DiffusionProbeConfig,
) -> DiffusionUnitResult:
    folds = tuple(fold_results)
    if not folds:
        raise ValueError("fold_results must not be empty")
    state_bits, scalar_count = _sum_family_scores(folds, STATE_SCALE)
    full_bits, full_scalars = _sum_family_scores(folds, OU_FULL)
    time_bits, time_scalars = _sum_family_scores(folds, TIME_SCALE)
    quadratic_bits, quadratic_scalars = _sum_family_scores(
        folds,
        QUADRATIC_DRIFT_FULL_Q,
    )
    if len(
        {scalar_count, full_scalars, time_scalars, quadratic_scalars}
    ) != 1:
        raise RuntimeError("state survivor controls used different test scalars")
    denominator = max(scalar_count, 1)
    advantage_full = (full_bits - state_bits) / denominator
    advantage_time = (time_bits - state_bits) / denominator
    advantage_quadratic = (quadratic_bits - state_bits) / denominator
    markov_totals: dict[int, float] = {}
    markov_scalars: dict[int, int] = {}
    for fold in folds:
        for item in fold.markov_order_sensitivity:
            markov_totals[item.order] = (
                markov_totals.get(item.order, 0.0)
                + item.score.total_codelength_bits
            )
            markov_scalars[item.order] = (
                markov_scalars.get(item.order, 0)
                + item.score.test_vector_count * item.score.latent_rank
            )
    if markov_totals:
        if len(set(markov_scalars.values())) != 1:
            raise RuntimeError("Markov orders used different session test scalars")
        ordered_markov = sorted(markov_totals.items(), key=lambda item: item[1])
        markov_vote = ordered_markov[0][0]
        if (
            len(ordered_markov) > 1
            and abs(ordered_markov[1][1] - ordered_markov[0][1]) <= 1e-12
        ):
            markov_vote = None
    else:
        markov_vote = None
    directions = tuple(
        fold.direction_classification
        for fold in folds
        if fold.direction_classification is not None
    )
    if len(directions) == len(folds):
        forward_total = sum(
            item.forward_full_total_bits for item in directions
        )
        reverse_total = sum(
            item.reverse_full_total_bits for item in directions
        )
        if forward_total + 1e-12 < reverse_total:
            direction_vote = "FORWARD_LOWER_CODE"
        elif reverse_total + 1e-12 < forward_total:
            direction_vote = "REVERSE_LOWER_CODE"
        else:
            direction_vote = "TIE"
    else:
        direction_vote = None
    complete = all(
        fold.covariance_fit_from_outer_train_trial_oof
        and not fold.outer_test_used_for_covariance_or_gate
        and fold.state_gate_uses_current_only
        and not fold.d1_d3_rows_treated_as_paired_trials
        and set(item.family for item in fold.scores) == set(PRIMARY_FAMILIES)
        for fold in folds
    )
    return DiffusionUnitResult(
        analysis_key=(
            f"session{session.index_one_based}:dim{dimension}:"
            f"eventmean={int(event_mean_removed)}"
        ),
        session_index_one_based=session.index_one_based,
        animal=session.animal,
        neuron_count=session.neuron_count,
        dimension=dimension,
        event_mean_removed=event_mean_removed,
        fold_results=folds,
        test_scalar_count=scalar_count,
        state_advantage_over_full_bits_per_scalar=float(advantage_full),
        state_advantage_over_time_bits_per_scalar=float(advantage_time),
        state_advantage_over_quadratic_bits_per_scalar=float(
            advantage_quadratic
        ),
        state_beats_all_three_controls=(
            advantage_full
            > config.minimum_codelength_advantage_bits_per_scalar
            and advantage_time
            > config.minimum_codelength_advantage_bits_per_scalar
            and advantage_quadratic
            > config.minimum_codelength_advantage_bits_per_scalar
        ),
        complete_outer_folds=complete,
        markov_order_vote=markov_vote,
        direction_vote=direction_vote,
    )


def _validate_session_population(
    population: np.ndarray,
    *,
    session: SessionSpec,
    label: str,
) -> np.ndarray:
    values = np.asarray(population, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"{label} must be trial x neuron x time")
    if min(values.shape) < 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must be finite and non-empty")
    if np.any(values < 0.0):
        raise ValueError(f"{label} contains negative count-like values")
    if values.shape[1] != session.neuron_count:
        raise ValueError(f"{label} does not match the session neuron count")
    return values


def run_diffusion_session_checkpoint(
    dimension_one_session: np.ndarray,
    dimension_three_session: np.ndarray,
    *,
    session: SessionSpec,
    config: DiffusionProbeConfig = DiffusionProbeConfig(),
    source_file_md5: str | None = None,
) -> DiffusionSessionCheckpoint:
    """Compute one independently serializable physical-session checkpoint."""

    if not isinstance(config, DiffusionProbeConfig):
        raise TypeError("config must be DiffusionProbeConfig")
    if not isinstance(session, SessionSpec):
        raise TypeError("session must be SessionSpec")
    dim1 = _validate_session_population(
        dimension_one_session,
        session=session,
        label="dimension_one_session",
    )
    dim3 = _validate_session_population(
        dimension_three_session,
        session=session,
        label="dimension_three_session",
    )
    if dim1.shape != dim3.shape:
        raise ValueError("session dimensions must have one shape")
    if dim1.shape[0] < config.outer_fold_count:
        raise ValueError("not enough trials for outer whole-trial CV")
    minimum_outer_train = dim1.shape[0] - int(
        np.ceil(dim1.shape[0] / config.outer_fold_count)
    )
    if minimum_outer_train < config.covariance_oof_fold_count:
        raise ValueError("not enough outer-training trials for covariance OOF")
    maximum_horizon = (
        max(config.semigroup_horizons)
        if config.run_semigroup_sensitivity
        else 1
    )
    required_last_target = (
        (config.global_anchor_depth - 1) * config.lag_bins
        + maximum_horizon * config.lag_bins
    )
    if dim1.shape[2] <= required_last_target:
        raise ValueError("session trajectory is too short for fixed sensitivities")
    event_modes = (
        (False, True)
        if config.run_event_mean_removed_sensitivity
        else (False,)
    )
    results = []
    for dimension, population in ((1, dim1), (3, dim3)):
        folds = make_whole_trial_folds(
            population.shape[0],
            fold_count=config.outer_fold_count,
            seed=_derived_seed(
                config.seed,
                "outer_folds",
                session.index_one_based,
                dimension,
            ),
        )
        for event_mean_removed in event_modes:
            fold_results = []
            for fold in folds:
                prepared = prepare_session_latent_fold(
                    population,
                    population,
                    fold,
                    rank_cap=config.rank_cap,
                    event_mean_removed=event_mean_removed,
                )
                fold_results.append(
                    evaluate_diffusion_fold(
                        prepared,
                        config=config,
                        seed_tokens=(
                            session.index_one_based,
                            dimension,
                            event_mean_removed,
                        ),
                    )
                )
            results.append(
                summarize_diffusion_unit(
                    session=session,
                    dimension=dimension,
                    event_mean_removed=event_mean_removed,
                    fold_results=tuple(fold_results),
                    config=config,
                )
            )
    checkpoint = DiffusionSessionCheckpoint(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        config_fingerprint=config_fingerprint(config),
        source_file_md5=source_file_md5,
        session=session,
        results=tuple(results),
        complete=all(item.complete_outer_folds for item in results),
    )
    validate_diffusion_session_checkpoint(checkpoint, config=config)
    return checkpoint


def _median(values: Sequence[float]) -> float:
    items = tuple(float(value) for value in values)
    if not items:
        raise ValueError("median requires at least one value")
    return float(np.median(np.asarray(items, dtype=np.float64)))


def aggregate_diffusion_results(
    results: Sequence[DiffusionUnitResult],
    *,
    config: DiffusionProbeConfig,
) -> tuple[DiffusionAggregateResult, ...]:
    items = tuple(results)
    event_modes = (
        (False, True)
        if config.run_event_mean_removed_sensitivity
        else (False,)
    )
    aggregates = []
    for event_mean_removed in event_modes:
        for animal in ("all", "Chico", "Silas"):
            group = tuple(
                item
                for item in items
                if item.event_mean_removed == event_mean_removed
                and (animal == "all" or item.animal == animal)
            )
            if not group:
                continue
            aggregates.append(
                DiffusionAggregateResult(
                    event_mean_removed=event_mean_removed,
                    animal=animal,
                    unit_count=len(group),
                    all_units_complete=all(
                        item.complete_outer_folds for item in group
                    ),
                    median_state_advantage_over_full_bits_per_scalar=_median(
                        tuple(
                            item.state_advantage_over_full_bits_per_scalar
                            for item in group
                        )
                    ),
                    median_state_advantage_over_time_bits_per_scalar=_median(
                        tuple(
                            item.state_advantage_over_time_bits_per_scalar
                            for item in group
                        )
                    ),
                    median_state_advantage_over_quadratic_bits_per_scalar=_median(
                        tuple(
                            item.state_advantage_over_quadratic_bits_per_scalar
                            for item in group
                        )
                    ),
                    joint_state_survivor_win_fraction=float(
                        np.mean(
                            np.asarray(
                                tuple(
                                    item.state_beats_all_three_controls
                                    for item in group
                                ),
                                dtype=np.float64,
                            )
                        )
                    ),
                )
            )
    return tuple(aggregates)


def _aggregate_lookup(
    aggregates: Sequence[DiffusionAggregateResult],
    *,
    event_mean_removed: bool,
    animal: str,
) -> DiffusionAggregateResult | None:
    matches = tuple(
        item
        for item in aggregates
        if item.event_mean_removed == event_mean_removed
        and item.animal == animal
    )
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError("diffusion aggregate key is not unique")
    return matches[0]


def _state_noise_survives(
    aggregates: Sequence[DiffusionAggregateResult],
    *,
    config: DiffusionProbeConfig,
) -> bool:
    if not config.run_event_mean_removed_sensitivity:
        return False
    groups = tuple(
        _aggregate_lookup(
            aggregates,
            event_mean_removed=event_mean_removed,
            animal=animal,
        )
        for event_mean_removed in (False, True)
        for animal in ("all", "Chico", "Silas")
    )
    if any(group is None for group in groups):
        return False
    minimum_advantage = config.minimum_codelength_advantage_bits_per_scalar
    return all(
        group.all_units_complete
        and group.median_state_advantage_over_full_bits_per_scalar
        > minimum_advantage
        and group.median_state_advantage_over_time_bits_per_scalar
        > minimum_advantage
        and group.median_state_advantage_over_quadratic_bits_per_scalar
        > minimum_advantage
        and group.joint_state_survivor_win_fraction
        > config.minimum_session_unit_win_fraction
        for group in groups
        if group is not None
    )


def _is_model_relative_iso_winner(
    results: Sequence[DiffusionUnitResult],
) -> bool:
    items = tuple(results)
    if not items:
        return False
    for item in items:
        totals = {
            family: sum(
                fold.score(family).total_codelength_bits
                for fold in item.fold_results
            )
            for family in PRIMARY_FAMILIES
        }
        if not all(
            totals[OU_ISO] < total
            for family, total in totals.items()
            if family != OU_ISO
        ):
            return False
    return True


def _build_verdicts(
    aggregates: Sequence[DiffusionAggregateResult],
    results: Sequence[DiffusionUnitResult],
    *,
    config: DiffusionProbeConfig,
) -> tuple[ClaimVerdict, ...]:
    state_pass = _state_noise_survives(aggregates, config=config)
    state_answer = (
        YES
        if state_pass
        else (
            NO
            if config.run_event_mean_removed_sensitivity
            else PENDING
        )
    )
    iso_winner = _is_model_relative_iso_winner(results)
    return (
        ClaimVerdict(
            "session_local_diffusion_covariance_ladder_completed",
            YES,
            "Six covariance/drift families share outer targets and OOF covariance fits.",
        ),
        ClaimVerdict(
            "model_relative_local_affine_isotropic_gaussian_proxy_winner",
            YES if iso_winner else NO,
            (
                "OU_ISO has the lowest aggregate codelength in every session x "
                "dimension x preprocessing analysis cell."
                if iso_winner
                else "OU_ISO is not the strict winner in every analysis cell."
            ),
        ),
        ClaimVerdict(
            "gaussian_innovation_law_identified",
            NO,
            "Only Gaussian residual families were compared; Gaussianity was not tested.",
        ),
        ClaimVerdict(
            "continuous_time_ou_process_identified",
            NO,
            (
                "No stability, stationary law, continuous-time generator, or "
                "embeddability test was performed."
            ),
        ),
        ClaimVerdict(
            "semigroup_sensitivity_completed",
            YES if config.run_semigroup_sensitivity else PENDING,
            (
                "Frozen 100 ms discrete affine-Gaussian composition was compared "
                "with direct 200/300 ms refits using a common complexity basis."
                if config.run_semigroup_sensitivity
                else "The optional semigroup sensitivity was not run."
            ),
        ),
        ClaimVerdict(
            "state_dependent_noise_proxy_survived_controls",
            state_answer,
            (
                "STATE_SCALE beats FULL, TIME_SCALE, and quadratic drift in raw "
                "and event-demeaned data, all/Chico/Silas, with majority unit wins."
                if state_pass
                else (
                    "STATE_SCALE does not pass every predeclared control."
                    if config.run_event_mean_removed_sensitivity
                    else "The event-mean-removed sensitivity was not completed."
                )
            ),
        ),
        ClaimVerdict(
            "biological_diffusion_identified",
            NO,
            "A Gaussian residual model is not a biological diffusion mechanism.",
        ),
        ClaimVerdict(
            "biological_diffusion_exists_or_is_absent",
            TEST_UNAVAILABLE,
            "This processed single-area snapshot cannot decide universal existence.",
        ),
        ClaimVerdict(
            "generative_reverse_process_identified",
            NO,
            "Reverse-time code is a descriptive classifier, not a learned reverse process.",
        ),
        ClaimVerdict(
            "score_function_identified",
            NO,
            "No density-gradient or denoising-score estimator is fitted.",
        ),
        ClaimVerdict(
            "causal_diffusion_mechanism_identified",
            NO,
            "No perturbation, rescue, or causal noise intervention is present.",
        ),
        ClaimVerdict(
            "spatial_graph_diffusion_identified",
            NO,
            "Session latent coordinates are not an anatomical spatial graph.",
        ),
    )


def validate_diffusion_claim_locks(locks: DiffusionClaimLocks) -> None:
    if not isinstance(locks, DiffusionClaimLocks):
        raise TypeError("locks must be DiffusionClaimLocks")
    unlocked = tuple(
        name for name, value in asdict(locks).items() if value is not False
    )
    if unlocked:
        raise ValueError(f"diffusion claim locks must remain false: {unlocked}")


def _close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=1e-12, atol=1e-9))


def _validate_gaussian_score(score: GaussianCodelengthResult) -> None:
    if not isinstance(score, GaussianCodelengthResult):
        raise TypeError("Gaussian score has an unexpected type")
    positive_counts = (
        score.oof_train_vector_count,
        score.bic_reference_vector_count,
        score.test_vector_count,
        score.latent_rank,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in positive_counts
    ):
        raise ValueError("Gaussian score vector counts and rank must be positive integers")
    parameter_counts = (
        score.drift_parameter_count,
        score.covariance_parameter_count,
        score.gate_parameter_count,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in parameter_counts
    ):
        raise ValueError("Gaussian score parameter counts must be non-negative integers")
    finite_values = (
        score.model_selection_bits,
        score.heldout_multivariate_gaussian_nll_bits,
        score.bic_parameter_bits,
        score.total_codelength_bits,
        score.bits_per_test_vector,
        score.bits_per_test_scalar,
        score.test_sse,
    )
    if not all(np.isfinite(float(value)) for value in finite_values):
        raise ValueError("Gaussian score contains a non-finite value")
    if score.model_selection_bits < 0.0 or score.test_sse < 0.0:
        raise ValueError("Gaussian model code and SSE must be non-negative")
    parameter_count = sum(parameter_counts)
    expected_bic = 0.5 * parameter_count * log2(
        float(max(score.bic_reference_vector_count, 2))
    )
    expected_total = (
        score.heldout_multivariate_gaussian_nll_bits
        + expected_bic
        + score.model_selection_bits
    )
    expected_vector_bits = expected_total / score.test_vector_count
    expected_scalar_bits = expected_total / (
        score.test_vector_count * score.latent_rank
    )
    if not (
        _close(score.bic_parameter_bits, expected_bic)
        and _close(score.total_codelength_bits, expected_total)
        and _close(score.bits_per_test_vector, expected_vector_bits)
        and _close(score.bits_per_test_scalar, expected_scalar_bits)
    ):
        raise ValueError("Gaussian score arithmetic drifted")


def _validate_semigroup_sensitivity(
    item: SemigroupSensitivity,
    *,
    config: DiffusionProbeConfig,
) -> None:
    if item.horizon_steps not in config.semigroup_horizons:
        raise ValueError("unexpected semigroup horizon")
    expected_milliseconds = (
        item.horizon_steps
        * config.lag_bins
        * config.time_bin_milliseconds
    )
    if item.horizon_milliseconds != expected_milliseconds:
        raise ValueError("semigroup horizon milliseconds drifted")
    positive_counts = (
        item.test_vector_count,
        item.direct_oof_train_vector_count,
        item.frozen_oof_train_vector_count,
        item.bic_reference_vector_count,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in positive_counts
    ):
        raise ValueError("semigroup vector counts must be positive integers")
    if (
        isinstance(item.shared_parameter_count, bool)
        or not isinstance(item.shared_parameter_count, int)
        or item.shared_parameter_count < 0
    ):
        raise ValueError("semigroup parameter count must be a non-negative integer")
    if item.bic_reference_vector_count != item.direct_oof_train_vector_count:
        raise ValueError("semigroup candidates do not share the direct-horizon BIC basis")
    expected_bic = 0.5 * item.shared_parameter_count * log2(
        float(max(item.bic_reference_vector_count, 2))
    )
    finite_values = (
        item.shared_bic_parameter_bits,
        item.frozen_semigroup_sse,
        item.direct_refit_sse,
        item.frozen_semigroup_bits_per_scalar,
        item.direct_refit_bits_per_scalar,
        item.frozen_advantage_over_direct_bits_per_scalar,
        item.frozen_excess_over_direct_bits_per_scalar,
    )
    if not all(np.isfinite(float(value)) for value in finite_values):
        raise ValueError("semigroup sensitivity contains a non-finite value")
    if item.frozen_semigroup_sse < 0.0 or item.direct_refit_sse < 0.0:
        raise ValueError("semigroup SSE must be non-negative")
    expected_excess = (
        item.frozen_semigroup_bits_per_scalar
        - item.direct_refit_bits_per_scalar
    )
    if not (
        _close(item.shared_bic_parameter_bits, expected_bic)
        and _close(item.frozen_excess_over_direct_bits_per_scalar, expected_excess)
        and _close(
            item.frozen_advantage_over_direct_bits_per_scalar,
            -expected_excess,
        )
    ):
        raise ValueError("semigroup comparison arithmetic drifted")
    expected_tolerance = (
        item.frozen_excess_over_direct_bits_per_scalar
        <= config.semigroup_max_excess_bits_per_scalar
    )
    if item.frozen_semigroup_within_tolerance != expected_tolerance:
        raise ValueError("semigroup tolerance flag drifted")
    if item.used_in_primary_diffusion_gate:
        raise ValueError("semigroup sensitivity entered the primary gate")


def _validate_direction_classification(item: DirectionClassification) -> None:
    if item.test_scalar_count < 1:
        raise ValueError("direction comparison must contain test scalars")
    values = (
        item.forward_full_bits_per_scalar,
        item.reverse_full_bits_per_scalar,
        item.forward_full_total_bits,
        item.reverse_full_total_bits,
    )
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("direction comparison contains a non-finite value")
    if not (
        _close(
            item.forward_full_bits_per_scalar,
            item.forward_full_total_bits / item.test_scalar_count,
        )
        and _close(
            item.reverse_full_bits_per_scalar,
            item.reverse_full_total_bits / item.test_scalar_count,
        )
    ):
        raise ValueError("direction comparison arithmetic drifted")
    expected_direction = (
        "FORWARD_LOWER_CODE"
        if item.forward_full_total_bits + 1e-12
        < item.reverse_full_total_bits
        else (
            "REVERSE_LOWER_CODE"
            if item.reverse_full_total_bits + 1e-12
            < item.forward_full_total_bits
            else "TIE"
        )
    )
    if item.lower_code_direction != expected_direction:
        raise ValueError("direction classification label drifted")
    if item.used_in_primary_diffusion_gate:
        raise ValueError("forward/reverse classification entered the gate")


def validate_diffusion_session_checkpoint(
    checkpoint: DiffusionSessionCheckpoint,
    *,
    config: DiffusionProbeConfig,
) -> None:
    if not isinstance(checkpoint, DiffusionSessionCheckpoint):
        raise TypeError("checkpoint must be DiffusionSessionCheckpoint")
    if checkpoint.schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema version drifted")
    if checkpoint.config_fingerprint != config_fingerprint(config):
        raise ValueError("checkpoint config fingerprint does not match")
    if checkpoint.source_file_md5 is not None and len(
        checkpoint.source_file_md5
    ) != 32:
        raise ValueError("checkpoint source MD5 must be a 32-character digest")
    if checkpoint.source_file_md5 is not None:
        try:
            int(checkpoint.source_file_md5, 16)
        except ValueError as error:
            raise ValueError("checkpoint source MD5 must be hexadecimal") from error
    expected_events = (
        2 if config.run_event_mean_removed_sensitivity else 1
    )
    if len(checkpoint.results) != 2 * expected_events:
        raise ValueError("checkpoint result count drifted")
    expected_keys = {
        (dimension, event_mean_removed)
        for dimension in (1, 3)
        for event_mean_removed in (
            (False, True)
            if config.run_event_mean_removed_sensitivity
            else (False,)
        )
    }
    observed_keys = {
        (item.dimension, item.event_mean_removed)
        for item in checkpoint.results
    }
    if observed_keys != expected_keys:
        raise ValueError("checkpoint dimension/sensitivity grid drifted")
    for item in checkpoint.results:
        if (
            item.session_index_one_based
            != checkpoint.session.index_one_based
            or item.animal != checkpoint.session.animal
            or item.neuron_count != checkpoint.session.neuron_count
        ):
            raise ValueError("checkpoint result crossed a session boundary")
        if len(item.fold_results) != config.outer_fold_count:
            raise ValueError("outer diffusion fold silently disappeared")
        if tuple(
            fold.fold_index_zero_based for fold in item.fold_results
        ) != tuple(range(config.outer_fold_count)):
            raise ValueError("outer diffusion fold indices drifted")
        if not item.complete_outer_folds:
            raise ValueError("checkpoint contains an incomplete outer fold")
        for fold in item.fold_results:
            score_families = tuple(score.family for score in fold.scores)
            if (
                len(score_families) != len(PRIMARY_FAMILIES)
                or set(score_families) != set(PRIMARY_FAMILIES)
            ):
                raise ValueError("primary diffusion family set drifted")
            if (
                not fold.covariance_fit_from_outer_train_trial_oof
                or fold.outer_test_used_for_covariance_or_gate
                or not fold.state_gate_uses_current_only
                or fold.d1_d3_rows_treated_as_paired_trials
            ):
                raise ValueError("OOF, gate, or D1/D3 pairing contract failed")
            if len(
                {score.test_vector_count for score in fold.scores}
            ) != 1:
                raise ValueError("primary scores used different outer targets")
            if (
                fold.common_outer_test_vector_count < 1
                or any(
                    score.test_vector_count
                    != fold.common_outer_test_vector_count
                    or score.latent_rank != fold.latent_rank
                    for score in fold.scores
                )
            ):
                raise ValueError("primary score metadata drifted")
            for score in fold.scores:
                _validate_gaussian_score(score)
            expected_orders = (
                config.markov_orders
                if config.run_markov_order_sensitivity
                else ()
            )
            if tuple(
                markov.order for markov in fold.markov_order_sensitivity
            ) != expected_orders:
                raise ValueError("Markov sensitivity order grid drifted")
            for markov in fold.markov_order_sensitivity:
                _validate_gaussian_score(markov.score)
                expected_family = (
                    OU_FULL
                    if markov.order == 1
                    else f"MARKOV_ORDER_{markov.order}_FULL"
                )
                if (
                    markov.common_anchor_depth
                    != config.global_anchor_depth
                    or markov.used_in_primary_diffusion_gate
                    or markov.score.family != expected_family
                    or markov.score.latent_rank != fold.latent_rank
                ):
                    raise ValueError("Markov sensitivity metadata drifted")
            if fold.markov_order_sensitivity and len(
                {
                    markov.score.test_vector_count
                    for markov in fold.markov_order_sensitivity
                }
            ) != 1:
                raise ValueError("Markov sensitivity used different outer targets")
            expected_horizons = (
                config.semigroup_horizons
                if config.run_semigroup_sensitivity
                else ()
            )
            if tuple(
                semigroup.horizon_steps
                for semigroup in fold.semigroup_sensitivity
            ) != expected_horizons:
                raise ValueError("semigroup horizon grid drifted")
            for semigroup in fold.semigroup_sensitivity:
                _validate_semigroup_sensitivity(semigroup, config=config)
            if config.run_reverse_classification:
                if fold.direction_classification is None:
                    raise ValueError("forward/reverse classification disappeared")
                _validate_direction_classification(fold.direction_classification)
            elif fold.direction_classification is not None:
                raise ValueError("disabled forward/reverse classification appeared")
        expected_item = summarize_diffusion_unit(
            session=checkpoint.session,
            dimension=item.dimension,
            event_mean_removed=item.event_mean_removed,
            fold_results=item.fold_results,
            config=config,
        )
        if item != expected_item:
            raise ValueError("diffusion unit summary does not match its fold scores")
    if checkpoint.complete != all(
        item.complete_outer_folds for item in checkpoint.results
    ):
        raise ValueError("checkpoint completion flag drifted")


def _validate_session_specs(
    specs: Sequence[SessionSpec],
    *,
    neuron_count: int,
) -> tuple[SessionSpec, ...]:
    items = tuple(specs)
    if not items:
        raise ValueError("at least one session spec is required")
    expected_start = 0
    for expected_index, item in enumerate(items, start=1):
        if not isinstance(item, SessionSpec):
            raise TypeError("session_specs must contain SessionSpec")
        if item.index_one_based != expected_index:
            raise ValueError("session indices must be consecutive and one-based")
        if item.column_start_zero_based != expected_start:
            raise ValueError("session column ranges must be contiguous")
        if item.column_stop_exclusive <= item.column_start_zero_based:
            raise ValueError("session column range must not be empty")
        expected_start = item.column_stop_exclusive
    if expected_start != neuron_count:
        raise ValueError("session column ranges do not cover the population")
    return items


def assemble_tafazoli_diffusion_report(
    checkpoints: Sequence[DiffusionSessionCheckpoint],
    *,
    config: DiffusionProbeConfig,
    session_specs: Sequence[SessionSpec],
    source_file_md5: str | None = None,
    verified_classifier_file: str | Path | None = None,
) -> TafazoliDiffusionProbeReport:
    """Assemble a deterministic report from independently saved checkpoints."""

    items = tuple(checkpoints)
    specs = tuple(session_specs)
    official_checksum_verified = False
    if verified_classifier_file is not None:
        verified_md5 = verify_official_classifier_checksum(
            verified_classifier_file
        )
        if source_file_md5 != verified_md5:
            raise ValueError("verified classifier file does not match checkpoint source")
        official_checksum_verified = True
    if not items:
        raise ValueError("at least one checkpoint is required")
    if tuple(item.session for item in items) != specs:
        raise ValueError("checkpoint order does not match session specs")
    if any(item.source_file_md5 != source_file_md5 for item in items):
        raise ValueError("checkpoint source MD5 does not match report source")
    for checkpoint in items:
        validate_diffusion_session_checkpoint(checkpoint, config=config)
    results = tuple(
        result
        for checkpoint in items
        for result in checkpoint.results
    )
    aggregates = aggregate_diffusion_results(results, config=config)
    locks = DiffusionClaimLocks()
    validate_diffusion_claim_locks(locks)
    report = TafazoliDiffusionProbeReport(
        schema_version=SCHEMA_VERSION,
        scope=PROBE_SCOPE,
        method_status="DIFFUSION_COVARIANCE_PROXY_COMPLETE",
        source_file_md5=source_file_md5,
        official_checksum_verified=official_checksum_verified,
        config=config,
        session_specs=specs,
        fields_used_for_fitting=(
            "ClassifierOpts.Dimpredictors[dimension=1].train",
            "ClassifierOpts.Dimpredictors[dimension=3].train",
        ),
        blind_fields_used=(),
        saved_test_role="not_used",
        train_only_preprocessing=True,
        primary_inference_unit="physical_recording_session",
        codelength_name="heldout multivariate Gaussian/BIC proxy",
        checkpoints=items,
        results=results,
        aggregates=aggregates,
        verdicts=_build_verdicts(aggregates, results, config=config),
        claim_locks=locks,
        limitations=(
            "The 36 rows are processed pseudotrials from one overwritten classifier fold.",
            "Covariances use three-fold OOF residuals inside each outer-training set.",
            "The outer-training latent coordinate frame is frozen before residual cross-fitting.",
            (
                "There are 27 independent recording sessions; D1 and D3 form 54 "
                "within-session analysis units per preprocessing mode."
            ),
            "D1 and D3 folds are independently seeded and rows are never paired.",
            "The 100 ms stride prevents overlapping observation windows becoming replicates.",
            "Fixed 0.1 shrinkage is a regularizer, not a learned biological prior.",
            "Markov order and semigroup sensitivities are not duplicate primary evidence.",
            "Forward/reverse is descriptive and cannot identify a generative reverse process.",
            "The codelength is a held-out Gaussian/BIC proxy, not strict prequential MDL.",
            (
                "The semigroup sensitivity is a discrete affine-Gaussian composition "
                "test, not an OU, stationarity, or continuous-time embedding test."
            ),
        ),
        conclusion=(
            "The probe can decide whether a restricted current-state noise scale "
            "survives constant full covariance, event-time scale, and quadratic "
            "drift controls. It cannot identify biological diffusion, a score "
            "function, causal noise, or spatial graph diffusion."
        ),
    )
    validate_tafazoli_diffusion_report(report)
    return report


def _validate_population(population: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(population, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"{label} must be trial x neuron x time")
    if min(values.shape) < 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must be finite and non-empty")
    if np.any(values < 0.0):
        raise ValueError(f"{label} contains negative count-like values")
    return values


def run_tafazoli_diffusion_probe_from_arrays(
    dimension_one_train: np.ndarray,
    dimension_three_train: np.ndarray,
    *,
    config: DiffusionProbeConfig = DiffusionProbeConfig(),
    session_specs: Sequence[SessionSpec] | None = None,
    source_file_md5: str | None = None,
) -> TafazoliDiffusionProbeReport:
    """Run every session checkpoint and assemble the NumPy-only report."""

    dim1 = _validate_population(
        dimension_one_train,
        label="dimension_one_train",
    )
    dim3 = _validate_population(
        dimension_three_train,
        label="dimension_three_train",
    )
    if dim1.shape != dim3.shape:
        raise ValueError("dimension 1 and 3 tensors must have one shape")
    specs = (
        recovered_session_specs()
        if session_specs is None
        else tuple(session_specs)
    )
    specs = _validate_session_specs(specs, neuron_count=dim1.shape[1])
    checkpoints = []
    for session in specs:
        column_slice = slice(
            session.column_start_zero_based,
            session.column_stop_exclusive,
        )
        checkpoints.append(
            run_diffusion_session_checkpoint(
                dim1[:, column_slice, :],
                dim3[:, column_slice, :],
                session=session,
                config=config,
                source_file_md5=source_file_md5,
            )
        )
    return assemble_tafazoli_diffusion_report(
        checkpoints,
        config=config,
        session_specs=specs,
        source_file_md5=source_file_md5,
    )


def run_tafazoli_diffusion_probe(
    classifier_file: str | Path,
    *,
    config: DiffusionProbeConfig = DiffusionProbeConfig(),
) -> TafazoliDiffusionProbeReport:
    """Checksum-lock the official MAT snapshot and run all checkpoints."""

    observed_md5 = verify_official_classifier_checksum(classifier_file)
    dim1, dim3 = load_tafazoli_train_dimensions(classifier_file)
    unverified_report = run_tafazoli_diffusion_probe_from_arrays(
        dim1,
        dim3,
        config=config,
        source_file_md5=observed_md5,
    )
    return assemble_tafazoli_diffusion_report(
        unverified_report.checkpoints,
        config=config,
        session_specs=unverified_report.session_specs,
        source_file_md5=observed_md5,
        verified_classifier_file=classifier_file,
    )


def validate_tafazoli_diffusion_report(
    report: TafazoliDiffusionProbeReport,
) -> None:
    """Reject leakage, unit drift, duplicated evidence, and overclaims."""

    if not isinstance(report, TafazoliDiffusionProbeReport):
        raise TypeError("report must be TafazoliDiffusionProbeReport")
    if report.schema_version != SCHEMA_VERSION or report.scope != PROBE_SCOPE:
        raise ValueError("unexpected diffusion report schema or scope")
    if report.method_status != "DIFFUSION_COVARIANCE_PROXY_COMPLETE":
        raise ValueError("unexpected diffusion method status")
    if report.codelength_name != "heldout multivariate Gaussian/BIC proxy":
        raise ValueError("codelength name drifted")
    if report.blind_fields_used or report.saved_test_role != "not_used":
        raise ValueError("blind fields or saved classifier test entered the probe")
    if report.official_checksum_verified and (
        report.source_file_md5 != OFFICIAL_CLASSIFIER_MD5
    ):
        raise ValueError("official checksum flag lacks the official source MD5")
    if not report.train_only_preprocessing:
        raise ValueError("preprocessing was not training-only")
    validate_diffusion_claim_locks(report.claim_locks)
    if tuple(item.session for item in report.checkpoints) != report.session_specs:
        raise ValueError("report checkpoint/session order drifted")
    if any(
        item.source_file_md5 != report.source_file_md5
        for item in report.checkpoints
    ):
        raise ValueError("report/checkpoint source MD5 drifted")
    for checkpoint in report.checkpoints:
        validate_diffusion_session_checkpoint(
            checkpoint,
            config=report.config,
        )
    flattened = tuple(
        item
        for checkpoint in report.checkpoints
        for item in checkpoint.results
    )
    if flattened != report.results:
        raise ValueError("report results do not match checkpoint contents")
    expected_events = (
        2 if report.config.run_event_mean_removed_sensitivity else 1
    )
    expected_result_count = len(report.session_specs) * 2 * expected_events
    if len(report.results) != expected_result_count:
        raise ValueError("report session x dimension unit count drifted")
    expected_aggregates = aggregate_diffusion_results(
        report.results,
        config=report.config,
    )
    if report.aggregates != expected_aggregates:
        raise ValueError("report aggregates do not match the unit results")
    for event_mean_removed in (
        (False, True)
        if report.config.run_event_mean_removed_sensitivity
        else (False,)
    ):
        all_group = _aggregate_lookup(
            report.aggregates,
            event_mean_removed=event_mean_removed,
            animal="all",
        )
        if all_group is None or all_group.unit_count != len(
            report.session_specs
        ) * 2:
            raise ValueError("all-animal aggregate unit count drifted")
    verdict_map = {item.key: item.answer for item in report.verdicts}
    required = {
        "session_local_diffusion_covariance_ladder_completed",
        "model_relative_local_affine_isotropic_gaussian_proxy_winner",
        "gaussian_innovation_law_identified",
        "continuous_time_ou_process_identified",
        "semigroup_sensitivity_completed",
        "state_dependent_noise_proxy_survived_controls",
        "biological_diffusion_identified",
        "biological_diffusion_exists_or_is_absent",
        "generative_reverse_process_identified",
        "score_function_identified",
        "causal_diffusion_mechanism_identified",
        "spatial_graph_diffusion_identified",
    }
    if set(verdict_map) != required:
        raise ValueError("diffusion verdict key set drifted")
    for key in (
        "gaussian_innovation_law_identified",
        "continuous_time_ou_process_identified",
        "biological_diffusion_identified",
        "generative_reverse_process_identified",
        "score_function_identified",
        "causal_diffusion_mechanism_identified",
        "spatial_graph_diffusion_identified",
    ):
        if verdict_map[key] != NO:
            raise ValueError(f"observational proxy overclaimed {key}")
    expected_verdicts = _build_verdicts(
        report.aggregates,
        report.results,
        config=report.config,
    )
    if report.verdicts != expected_verdicts:
        raise ValueError("report verdicts do not match the validated results")


__all__ = [
    "DiffusionAggregateResult",
    "DiffusionClaimLocks",
    "DiffusionFoldResult",
    "DiffusionProbeConfig",
    "DiffusionSessionCheckpoint",
    "DiffusionUnitResult",
    "DirectionClassification",
    "DriftModel",
    "GaussianCodelengthResult",
    "IMPLEMENTATION_REVISION",
    "MarkovOrderSensitivity",
    "NO",
    "NoiseModel",
    "OOFResiduals",
    "OU_DIAG",
    "OU_FULL",
    "OU_ISO",
    "PENDING",
    "PRIMARY_FAMILIES",
    "PROBE_SCOPE",
    "QUADRATIC_DRIFT_FULL_Q",
    "SCHEMA_VERSION",
    "STATE_SCALE",
    "SemigroupSensitivity",
    "TEST_UNAVAILABLE",
    "TIME_SCALE",
    "TafazoliDiffusionProbeReport",
    "YES",
    "aggregate_diffusion_results",
    "apply_drift_model",
    "assemble_tafazoli_diffusion_report",
    "config_fingerprint",
    "crossfit_drift_residuals",
    "evaluate_diffusion_fold",
    "fit_drift_model",
    "fit_noise_model",
    "run_diffusion_session_checkpoint",
    "run_tafazoli_diffusion_probe",
    "run_tafazoli_diffusion_probe_from_arrays",
    "score_multivariate_gaussian",
    "summarize_diffusion_unit",
    "validate_diffusion_claim_locks",
    "validate_diffusion_session_checkpoint",
    "validate_tafazoli_diffusion_report",
]
