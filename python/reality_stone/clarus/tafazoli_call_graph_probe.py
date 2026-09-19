"""Label-blind call-graph proxy probe for the Tafazoli PFC snapshot.

This module tests restricted observational proxies.  It does not identify a
biological callee, a task-inheritance tree, or a brain programming language.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from math import factorial, log2, pi
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .tafazoli_session_operator_probe import (
    OFFICIAL_CLASSIFIER_MD5,
    PreparedLatentFold,
    ProbeClaimLocks as StationaryClaimLocks,
    SessionSpec,
    WholeTrialFold,
    load_tafazoli_train_dimensions,
    make_whole_trial_folds,
    prepare_session_latent_fold,
    recovered_session_specs,
    verify_official_classifier_checksum,
)


SCHEMA_VERSION = "clarus-tafazoli-call-graph-probe/v1"
PROBE_SCOPE = "label_blind_session_local_call_graph_proxy_probe"

YES = "YES"
NO = "NO"
PENDING = "PENDING"
TEST_UNAVAILABLE = "TEST_UNAVAILABLE"


@dataclass(frozen=True)
class CallGraphProbeConfig:
    """Predeclared, deterministic model-comparison protocol."""

    seed: int = 20260730
    states: tuple[int, ...] = (2, 3)
    history_depths: tuple[int, ...] = (1, 2, 3)
    rank_cap: int = 3
    lag_bins: int = 10
    primary_stride_bins: int = 10
    fold_count: int = 6
    ridge_alpha: float = 1.0
    kmeans_restarts: int = 8
    kmeans_max_iterations: int = 100
    run_event_mean_removed_sensitivity: bool = True
    run_reverse_descriptive_control: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("seed", self.seed),
            ("rank_cap", self.rank_cap),
            ("lag_bins", self.lag_bins),
            ("primary_stride_bins", self.primary_stride_bins),
            ("fold_count", self.fold_count),
            ("kmeans_restarts", self.kmeans_restarts),
            ("kmeans_max_iterations", self.kmeans_max_iterations),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.rank_cap < 1:
            raise ValueError("rank_cap must be positive")
        if self.lag_bins < 1 or self.primary_stride_bins < 1:
            raise ValueError("lag and stride must be positive")
        if self.fold_count < 2:
            raise ValueError("fold_count must be at least two")
        if self.kmeans_restarts < 1 or self.kmeans_max_iterations < 1:
            raise ValueError("k-means controls must be positive")
        if not self.states or not self.history_depths:
            raise ValueError("states and history_depths must not be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2
            for value in self.states
        ):
            raise ValueError("state counts must be integers of at least two")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.history_depths
        ):
            raise ValueError("history depths must be positive integers")
        if len(set(self.states)) != len(self.states):
            raise ValueError("state counts must be unique")
        if len(set(self.history_depths)) != len(self.history_depths):
            raise ValueError("history depths must be unique")
        if isinstance(self.ridge_alpha, bool) or not np.isfinite(float(self.ridge_alpha)):
            raise ValueError("ridge_alpha must be finite")
        if self.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must not be negative")

    @property
    def state_counts(self) -> tuple[int, ...]:
        """Compatibility name used internally for the predeclared states."""

        return self.states


@dataclass(frozen=True)
class PastOnlyGate:
    """Train-fold centroids; assignment accepts histories, never targets."""

    state_count: int
    history_depth: int
    latent_rank: int
    centroids: np.ndarray
    training_inertia: float


@dataclass(frozen=True)
class TrajectoryDesign:
    """Past histories, current states, successors, and event-anchor indices."""

    history: np.ndarray
    current: np.ndarray
    successor: np.ndarray
    anchor_indices: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(np.prod(self.current.shape[:-1]))


@dataclass(frozen=True)
class WhitenedLatentFold:
    """Source-train-only whitening applied to all source/target fold tensors."""

    source_train: np.ndarray
    source_test: np.ndarray
    target_train: np.ndarray
    target_test: np.ndarray
    latent_mean: np.ndarray
    latent_scale: np.ndarray


@dataclass(frozen=True)
class LinearPredictor:
    """A fitted deterministic predictor plus its declared parameter count."""

    family: str
    coefficients: np.ndarray
    parameter_count: int
    state_count: int | None = None
    history_order: int | None = None


@dataclass(frozen=True)
class CodelengthResult:
    """Held-out Gaussian codelength/BIC proxy; this is not prequential MDL."""

    family: str
    train_scalar_count: int
    test_scalar_count: int
    dynamic_parameter_count: int
    gate_parameter_count: int
    variance_parameter_count: int
    model_selection_bits: float
    residual_variance: float
    train_sse: float
    test_sse: float
    heldout_gaussian_nll_bits: float
    bic_parameter_bits: float
    total_codelength_bits: float
    bits_per_test_scalar: float
    grand_mean_r2: float


@dataclass(frozen=True)
class HubFoldResult:
    """One fold's non-overlap observational convergence proxy."""

    available: bool
    hub_state: int | None
    train_hub_samples: int
    test_hub_samples: int
    distinct_train_callers: int
    predecessor_entropy: float
    successor_entropy: float
    occupancy_fraction: float
    hub_score: float
    shared: CodelengthResult | None
    caller_specific: CodelengthResult | None
    time_locked: CodelengthResult | None


@dataclass(frozen=True)
class FoldComparison:
    """One outer-fold comparison with train-only representation and gate."""

    fold_index_zero_based: int
    latent_rank: int
    active_neuron_count: int
    switching: CodelengthResult
    matched_var: CodelengthResult
    state_parent_rank1: CodelengthResult
    reverse_switching: CodelengthResult | None
    reverse_var: CodelengthResult | None
    hub: HubFoldResult
    reverse_hub: HubFoldResult | None
    gate_uses_current_and_past_only: bool
    test_target_passed_to_gate: bool


@dataclass(frozen=True)
class SessionModelResult:
    """One session, dimension, state-count, and history-depth result."""

    analysis_key: str
    session_index_one_based: int
    animal: str
    neuron_count: int
    dimension: int
    state_count: int
    history_depth: int
    var_order: int
    parameter_matched_dynamic_block: bool
    event_mean_removed: bool
    stride_bins: int
    fold_results: tuple[FoldComparison, ...]
    switching_vs_var_skill: float
    switching_codelength_advantage_bits_per_scalar: float
    forward_vs_reverse_skill: float | None
    forward_codelength_advantage_over_reverse_bits_per_scalar: float | None
    state_parent_rank1_vs_var_skill: float
    state_parent_rank1_codelength_advantage_bits_per_scalar: float
    hub_available_fold_count: int
    hub_total_fold_count: int
    hub_all_folds_available: bool
    hub_shared_vs_time_skill: float | None
    hub_shared_codelength_advantage_over_time_bits_per_scalar: float | None
    hub_shared_codelength_advantage_over_caller_bits_per_scalar: float | None
    hub_forward_vs_reverse_skill: float | None
    hub_forward_codelength_advantage_over_reverse_bits_per_scalar: float | None


@dataclass(frozen=True)
class FrozenTransferResult:
    """Source-frozen D1↔D3 switching transfer in one physical session."""

    analysis_key: str
    session_index_one_based: int
    animal: str
    neuron_count: int
    source_dimension: int
    target_dimension: int
    state_count: int
    history_depth: int
    event_mean_removed: bool
    frozen_test_sse: float
    target_refit_test_sse: float
    frozen_vs_target_refit_skill: float
    frozen_codelength_advantage_over_target_refit_bits_per_scalar: float
    source_representation_and_gate_frozen: bool
    target_rows_paired_to_source_rows: bool


@dataclass(frozen=True)
class AggregateResult:
    """Session×dimension aggregation; time bins and neurons are not replicates."""

    state_count: int
    history_depth: int
    event_mean_removed: bool
    animal: str
    unit_count: int
    median_switching_vs_var_skill: float
    median_switching_codelength_advantage_bits_per_scalar: float
    median_forward_vs_reverse_skill: float | None
    median_forward_codelength_advantage_over_reverse_bits_per_scalar: float | None
    median_state_parent_rank1_vs_var_skill: float
    median_state_parent_rank1_codelength_advantage_bits_per_scalar: float
    all_units_have_complete_hub_folds: bool
    median_hub_shared_vs_time_skill: float | None
    median_hub_shared_codelength_advantage_over_time_bits_per_scalar: float | None
    median_hub_shared_codelength_advantage_over_caller_bits_per_scalar: float | None
    median_hub_forward_vs_reverse_skill: float | None


@dataclass(frozen=True)
class ClaimVerdict:
    """Claim-local result whose wording fixes the inference boundary."""

    key: str
    answer: str
    reason: str


@dataclass(frozen=True)
class CallGraphClaimLocks:
    """Scientific claims that an observational processed snapshot cannot unlock."""

    labels_or_responses_used: bool = False
    all_factors_used: bool = False
    dimension_two_used: bool = False
    saved_classifier_test_set_used: bool = False
    full_pseudopopulation_fit: bool = False
    test_future_used_for_gate: bool = False
    d1_d3_rows_treated_as_paired_trials: bool = False
    biological_common_callee_identified: bool = False
    task_inheritance_identified: bool = False
    causal_call_return_identified: bool = False
    brain_programming_language_identified: bool = False


@dataclass(frozen=True)
class TafazoliCallGraphProbeReport:
    """Serializable output of the session-local observational proxy probe."""

    schema_version: str
    scope: str
    method_status: str
    source_file_md5: str | None
    official_checksum_verified: bool
    config: CallGraphProbeConfig
    session_specs: tuple[SessionSpec, ...]
    fields_used_for_fitting: tuple[str, ...]
    blind_fields_used: tuple[str, ...]
    saved_test_role: str
    train_only_preprocessing: bool
    primary_inference_unit: str
    codelength_name: str
    model_results: tuple[SessionModelResult, ...]
    event_mean_removed_results: tuple[SessionModelResult, ...]
    frozen_transfer_results: tuple[FrozenTransferResult, ...]
    event_mean_removed_frozen_transfer_results: tuple[FrozenTransferResult, ...]
    aggregates: tuple[AggregateResult, ...]
    verdicts: tuple[ClaimVerdict, ...]
    claim_locks: CallGraphClaimLocks
    inherited_stationary_claim_locks: StationaryClaimLocks
    limitations: tuple[str, ...]
    conclusion: str

    def verdict(self, key: str) -> ClaimVerdict:
        """Return one verdict by its stable key."""

        matches = tuple(item for item in self.verdicts if item.key == key)
        if len(matches) != 1:
            raise KeyError(key)
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return asdict(self)


def _validate_latent(latent: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(latent, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"{label} must be trial x time x latent")
    if min(values.shape) < 1:
        raise ValueError(f"{label} axes must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains NaN or infinity")
    return values


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


def build_trajectory_design(
    latent: np.ndarray,
    *,
    history_depth: int,
    anchor_history_depth: int,
    lag_bins: int,
    stride_bins: int,
    reverse: bool = False,
) -> TrajectoryDesign:
    """Build a forecast design without crossing a trial boundary.

    ``history`` contains ``z_t, z_(t-L), ...`` only.  ``successor`` is kept
    separate so it cannot enter the gate API.  ``anchor_history_depth`` lets
    competing families use exactly the same forecast anchors even when their
    history depths differ.
    """

    values = _validate_latent(latent, label="latent")
    for name, value in (
        ("history_depth", history_depth),
        ("anchor_history_depth", anchor_history_depth),
        ("lag_bins", lag_bins),
        ("stride_bins", stride_bins),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if history_depth > anchor_history_depth:
        raise ValueError("history_depth cannot exceed anchor_history_depth")
    if reverse:
        values = values[:, ::-1, :]
    first_anchor = (anchor_history_depth - 1) * lag_bins
    stop_exclusive = values.shape[1] - lag_bins
    if first_anchor >= stop_exclusive:
        raise ValueError("trajectory is too short for the requested history")
    anchors = np.arange(
        first_anchor,
        stop_exclusive,
        stride_bins,
        dtype=np.int64,
    )
    blocks = tuple(values[:, anchors - offset * lag_bins, :] for offset in range(history_depth))
    history = np.concatenate(blocks, axis=2)
    return TrajectoryDesign(
        history=np.asarray(history, dtype=np.float64),
        current=np.asarray(blocks[0], dtype=np.float64),
        successor=np.asarray(values[:, anchors + lag_bins, :], dtype=np.float64),
        anchor_indices=anchors,
    )


def _squared_distances(samples: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    delta = samples[:, None, :] - centroids[None, :, :]
    return np.einsum("nkd,nkd->nk", delta, delta, optimize=True)


def _initial_centroids(
    samples: np.ndarray,
    state_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    first = int(rng.integers(samples.shape[0]))
    selected = [first]
    minimum_distance = np.sum(
        np.square(samples - samples[first]),
        axis=1,
    )
    for _ in range(1, state_count):
        total = float(np.sum(minimum_distance))
        if total <= np.finfo(np.float64).eps:
            remaining = np.setdiff1d(
                np.arange(samples.shape[0]),
                np.asarray(selected),
                assume_unique=False,
            )
            selected.append(int(remaining[0]))
        else:
            selected.append(int(rng.choice(samples.shape[0], p=minimum_distance / total)))
        candidate_distance = np.sum(
            np.square(samples - samples[selected[-1]]),
            axis=1,
        )
        minimum_distance = np.minimum(minimum_distance, candidate_distance)
    return np.array(samples[selected], copy=True)


def _canonicalize_centroids(centroids: np.ndarray) -> np.ndarray:
    order = sorted(
        range(centroids.shape[0]),
        key=lambda index: tuple(float(value) for value in centroids[index]),
    )
    return np.asarray(centroids[order], dtype=np.float64)


def fit_past_only_gate(
    history: np.ndarray,
    *,
    state_count: int,
    history_depth: int,
    latent_rank: int,
    seed: int,
    restarts: int = 8,
    max_iterations: int = 100,
) -> PastOnlyGate:
    """Fit deterministic k-means using current/past history and no target."""

    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("history must be trial x anchor x feature")
    if not np.all(np.isfinite(values)):
        raise ValueError("history contains NaN or infinity")
    if state_count < 2 or state_count > int(np.prod(values.shape[:2])):
        raise ValueError("state_count is incompatible with sample count")
    if values.shape[2] != history_depth * latent_rank:
        raise ValueError("history feature count does not match depth and rank")
    if restarts < 1 or max_iterations < 1:
        raise ValueError("k-means controls must be positive")
    samples = values.reshape(-1, values.shape[-1])
    best_centroids: np.ndarray | None = None
    best_inertia = np.inf
    for restart in range(restarts):
        rng = np.random.Generator(np.random.PCG64(_derived_seed(seed, "kmeans", restart)))
        centroids = _initial_centroids(samples, state_count, rng)
        previous_labels: np.ndarray | None = None
        for _ in range(max_iterations):
            distances = _squared_distances(samples, centroids)
            labels = np.argmin(distances, axis=1)
            if previous_labels is not None and np.array_equal(
                labels,
                previous_labels,
            ):
                break
            previous_labels = labels
            updated = np.empty_like(centroids)
            nearest_distance = distances[
                np.arange(samples.shape[0]),
                labels,
            ]
            for state in range(state_count):
                members = samples[labels == state]
                if members.size:
                    updated[state] = members.mean(axis=0)
                else:
                    replacement = int(np.argmax(nearest_distance))
                    updated[state] = samples[replacement]
                    nearest_distance[replacement] = -np.inf
            centroids = updated
        centroids = _canonicalize_centroids(centroids)
        final_distances = _squared_distances(samples, centroids)
        inertia = float(np.sum(np.min(final_distances, axis=1), dtype=np.float64))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = np.array(centroids, copy=True)
    if best_centroids is None:
        raise RuntimeError("k-means did not produce centroids")
    return PastOnlyGate(
        state_count=state_count,
        history_depth=history_depth,
        latent_rank=latent_rank,
        centroids=best_centroids,
        training_inertia=best_inertia,
    )


def assign_past_only_states(
    gate: PastOnlyGate,
    history: np.ndarray,
) -> np.ndarray:
    """Assign states from current/past history; no future argument exists."""

    if not isinstance(gate, PastOnlyGate):
        raise TypeError("gate must be PastOnlyGate")
    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("history must be trial x anchor x feature")
    if values.shape[2] != gate.centroids.shape[1]:
        raise ValueError("history feature count does not match frozen gate")
    if not np.all(np.isfinite(values)):
        raise ValueError("history contains NaN or infinity")
    distances = _squared_distances(
        values.reshape(-1, values.shape[-1]),
        gate.centroids,
    )
    return np.argmin(distances, axis=1).reshape(values.shape[:2])


def switching_parameter_count(state_count: int, latent_rank: int) -> int:
    """Common intercept plus one current-state linear map per gate state."""

    return latent_rank + state_count * latent_rank * latent_rank


def var_parameter_count(history_order: int, latent_rank: int) -> int:
    """Common intercept plus a dense VAR history block."""

    return latent_rank + history_order * latent_rank * latent_rank


def centroid_parameter_count(
    state_count: int,
    history_depth: int,
    latent_rank: int,
) -> int:
    """Parameters transmitted for the deterministic nearest-centroid gate."""

    return state_count * history_depth * latent_rank


def state_parent_rank1_parameter_count(
    state_count: int,
    latent_rank: int,
) -> int:
    """Common affine parent plus state-level rank-one residual maps."""

    parent = latent_rank * (latent_rank + 1)
    child_residuals = state_count * (2 * latent_rank - 1)
    return parent + child_residuals


def model_selection_bits(
    config: CallGraphProbeConfig,
    *,
    family_count: int,
    includes_history_choice: bool,
    includes_hub_choice: bool,
    state_count: int,
    includes_state_label_permutation: bool = True,
) -> float:
    """Code predeclared family/grid/hub choices without calling it strict MDL."""

    if family_count < 1:
        raise ValueError("family_count must be positive")
    choices = family_count * len(config.state_counts)
    if includes_history_choice:
        choices *= len(config.history_depths)
    bits = log2(float(max(choices, 1)))
    if includes_hub_choice:
        bits += log2(float(state_count))
    if includes_state_label_permutation:
        bits += log2(float(factorial(state_count)))
    return float(bits)


def _ridge_coefficients(
    design: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float,
    penalize_intercept: bool,
) -> np.ndarray:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("ridge design and target must be aligned matrices")
    gram = x.T @ x
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    if not penalize_intercept:
        penalty[0, 0] = 0.0
    return np.linalg.pinv(gram + penalty) @ x.T @ y


def _switching_design(
    current: np.ndarray,
    states: np.ndarray,
    *,
    state_count: int,
) -> np.ndarray:
    x = np.asarray(current, dtype=np.float64).reshape(
        -1,
        current.shape[-1],
    )
    labels = np.asarray(states, dtype=np.int64).reshape(-1)
    if x.shape[0] != labels.size:
        raise ValueError("current states and gate labels do not align")
    if np.any(labels < 0) or np.any(labels >= state_count):
        raise ValueError("gate label lies outside the declared state count")
    blocks = np.zeros(
        (x.shape[0], state_count * x.shape[1]),
        dtype=np.float64,
    )
    rows = np.arange(x.shape[0])
    starts = labels * x.shape[1]
    for offset in range(x.shape[1]):
        blocks[rows, starts + offset] = x[:, offset]
    return np.column_stack((np.ones(x.shape[0]), blocks))


def fit_switching_predictor(
    current: np.ndarray,
    successor: np.ndarray,
    states: np.ndarray,
    *,
    state_count: int,
    ridge_alpha: float,
) -> LinearPredictor:
    """Fit common intercept plus state-specific current-state maps."""

    x = _validate_latent(current, label="current")
    y = _validate_latent(successor, label="successor")
    if x.shape != y.shape or np.asarray(states).shape != x.shape[:2]:
        raise ValueError("switching tensors and state labels must align")
    design = _switching_design(x, states, state_count=state_count)
    coefficients = _ridge_coefficients(
        design,
        y.reshape(-1, y.shape[-1]),
        alpha=ridge_alpha,
        penalize_intercept=False,
    )
    return LinearPredictor(
        family="past_gated_switching_current_map",
        coefficients=coefficients,
        parameter_count=switching_parameter_count(
            state_count,
            x.shape[-1],
        ),
        state_count=state_count,
    )


def apply_switching_predictor(
    predictor: LinearPredictor,
    current: np.ndarray,
    states: np.ndarray,
) -> np.ndarray:
    """Apply a frozen switching predictor to current state and past-only labels."""

    if predictor.family != "past_gated_switching_current_map":
        raise ValueError("predictor is not a switching current-map model")
    if predictor.state_count is None:
        raise ValueError("switching predictor has no state count")
    x = _validate_latent(current, label="current")
    design = _switching_design(
        x,
        states,
        state_count=predictor.state_count,
    )
    return (design @ predictor.coefficients).reshape(x.shape)


def fit_var_predictor(
    history: np.ndarray,
    successor: np.ndarray,
    *,
    history_order: int,
    latent_rank: int,
    ridge_alpha: float,
) -> LinearPredictor:
    """Fit a stationary VAR on the same held-out forecast anchors."""

    h = np.asarray(history, dtype=np.float64)
    y = _validate_latent(successor, label="successor")
    if h.ndim != 3 or h.shape[:2] != y.shape[:2]:
        raise ValueError("VAR history and successor must align")
    if h.shape[2] != history_order * latent_rank:
        raise ValueError("VAR history feature count is inconsistent")
    flat = h.reshape(-1, h.shape[-1])
    design = np.column_stack((np.ones(flat.shape[0]), flat))
    coefficients = _ridge_coefficients(
        design,
        y.reshape(-1, y.shape[-1]),
        alpha=ridge_alpha,
        penalize_intercept=False,
    )
    return LinearPredictor(
        family="stationary_var",
        coefficients=coefficients,
        parameter_count=var_parameter_count(history_order, latent_rank),
        history_order=history_order,
    )


def apply_var_predictor(
    predictor: LinearPredictor,
    history: np.ndarray,
) -> np.ndarray:
    """Apply a frozen stationary VAR predictor."""

    if predictor.family != "stationary_var":
        raise ValueError("predictor is not a stationary VAR")
    h = np.asarray(history, dtype=np.float64)
    if h.ndim != 3:
        raise ValueError("history must be trial x anchor x feature")
    flat = h.reshape(-1, h.shape[-1])
    design = np.column_stack((np.ones(flat.shape[0]), flat))
    if design.shape[1] != predictor.coefficients.shape[0]:
        raise ValueError("history feature count does not match frozen VAR")
    predicted = design @ predictor.coefficients
    return predicted.reshape(h.shape[0], h.shape[1], -1)


def fit_state_parent_rank1_predictor(
    current: np.ndarray,
    successor: np.ndarray,
    states: np.ndarray,
    *,
    state_count: int,
    ridge_alpha: float,
) -> LinearPredictor:
    """Fit an affine state parent plus rank-one state residuals.

    This is a state-level low-rank switching proxy.  It is deliberately not
    named or interpreted as task inheritance.
    """

    x = _validate_latent(current, label="current")
    y = _validate_latent(successor, label="successor")
    labels = np.asarray(states, dtype=np.int64).reshape(-1)
    if x.shape != y.shape or labels.size != int(np.prod(x.shape[:2])):
        raise ValueError("parent-rank1 tensors and states must align")
    flat_x = x.reshape(-1, x.shape[-1])
    flat_y = y.reshape(-1, y.shape[-1])
    parent_design = np.column_stack((np.ones(flat_x.shape[0]), flat_x))
    parent = _ridge_coefficients(
        parent_design,
        flat_y,
        alpha=ridge_alpha,
        penalize_intercept=False,
    )
    residual = flat_y - parent_design @ parent
    rank_one_maps = np.zeros(
        (state_count, x.shape[-1], x.shape[-1]),
        dtype=np.float64,
    )
    for state in range(state_count):
        member_mask = labels == state
        if np.count_nonzero(member_mask) < 2:
            continue
        full_map = _ridge_coefficients(
            flat_x[member_mask],
            residual[member_mask],
            alpha=ridge_alpha,
            penalize_intercept=True,
        )
        left, singular, right = np.linalg.svd(full_map, full_matrices=False)
        rank_one_maps[state] = (left[:, :1] * singular[:1]) @ right[:1, :]
    packed = np.concatenate(
        (
            parent,
            rank_one_maps.reshape(state_count * x.shape[-1], x.shape[-1]),
        ),
        axis=0,
    )
    return LinearPredictor(
        family="state_parent_plus_rank1_residual",
        coefficients=packed,
        parameter_count=state_parent_rank1_parameter_count(
            state_count,
            x.shape[-1],
        ),
        state_count=state_count,
    )


def apply_state_parent_rank1_predictor(
    predictor: LinearPredictor,
    current: np.ndarray,
    states: np.ndarray,
) -> np.ndarray:
    """Apply the frozen state-level parent-plus-rank-one proxy."""

    if predictor.family != "state_parent_plus_rank1_residual":
        raise ValueError("predictor is not a state parent-rank1 proxy")
    if predictor.state_count is None:
        raise ValueError("state parent-rank1 predictor has no state count")
    x = _validate_latent(current, label="current")
    labels = np.asarray(states, dtype=np.int64).reshape(-1)
    flat_x = x.reshape(-1, x.shape[-1])
    if labels.size != flat_x.shape[0]:
        raise ValueError("current states and labels do not align")
    parent_rows = x.shape[-1] + 1
    parent = predictor.coefficients[:parent_rows]
    residual_maps = predictor.coefficients[parent_rows:].reshape(
        predictor.state_count,
        x.shape[-1],
        x.shape[-1],
    )
    parent_prediction = np.column_stack((np.ones(flat_x.shape[0]), flat_x)) @ parent
    residual_prediction = np.einsum(
        "ni,nij->nj",
        flat_x,
        residual_maps[labels],
        optimize=True,
    )
    return (parent_prediction + residual_prediction).reshape(x.shape)


def _sse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(
        np.sum(
            np.square(
                np.asarray(observed, dtype=np.float64) - np.asarray(predicted, dtype=np.float64)
            ),
            dtype=np.float64,
        )
    )


def _skill(model_sse: float, baseline_sse: float) -> float:
    epsilon = np.finfo(np.float64).eps
    denominator = max(float(baseline_sse), epsilon)
    return float(1.0 - float(model_sse) / denominator)


def score_heldout_gaussian_codelength(
    *,
    family: str,
    train_observed: np.ndarray,
    train_predicted: np.ndarray,
    test_observed: np.ndarray,
    test_predicted: np.ndarray,
    dynamic_parameter_count: int,
    gate_parameter_count: int,
    model_selection_cost_bits: float,
) -> CodelengthResult:
    """Score a held-out Gaussian codelength/BIC proxy.

    The scalar residual variance is estimated on the outer training fold.
    The BIC-style parameter transmission cost uses the declared parameter
    count.  This intentionally is not labelled strict or prequential MDL.
    """

    train_y = np.asarray(train_observed, dtype=np.float64)
    train_hat = np.asarray(train_predicted, dtype=np.float64)
    test_y = np.asarray(test_observed, dtype=np.float64)
    test_hat = np.asarray(test_predicted, dtype=np.float64)
    if train_y.shape != train_hat.shape or test_y.shape != test_hat.shape:
        raise ValueError("observed and predicted tensors must align")
    if min(train_y.size, test_y.size) < 1:
        raise ValueError("codelength inputs must not be empty")
    train_sse = _sse(train_y, train_hat)
    test_sse = _sse(test_y, test_hat)
    train_scalar_count = int(train_y.size)
    test_scalar_count = int(test_y.size)
    variance_parameter_count = 1
    degrees = max(train_scalar_count - dynamic_parameter_count, 1)
    residual_variance = max(
        train_sse / degrees,
        np.finfo(np.float64).eps,
    )
    heldout_nll = 0.5 * test_scalar_count * log2(2.0 * pi * residual_variance) + test_sse / (
        2.0 * residual_variance * np.log(2.0)
    )
    total_parameter_count = (
        dynamic_parameter_count + gate_parameter_count + variance_parameter_count
    )
    parameter_bits = 0.5 * total_parameter_count * log2(float(max(train_scalar_count, 2)))
    total_bits = heldout_nll + parameter_bits + model_selection_cost_bits
    grand_mean = train_y.reshape(-1, train_y.shape[-1]).mean(axis=0)
    grand_prediction = np.broadcast_to(grand_mean, test_y.shape)
    return CodelengthResult(
        family=family,
        train_scalar_count=train_scalar_count,
        test_scalar_count=test_scalar_count,
        dynamic_parameter_count=dynamic_parameter_count,
        gate_parameter_count=gate_parameter_count,
        variance_parameter_count=variance_parameter_count,
        model_selection_bits=float(model_selection_cost_bits),
        residual_variance=float(residual_variance),
        train_sse=train_sse,
        test_sse=test_sse,
        heldout_gaussian_nll_bits=float(heldout_nll),
        bic_parameter_bits=float(parameter_bits),
        total_codelength_bits=float(total_bits),
        bits_per_test_scalar=float(total_bits / test_scalar_count),
        grand_mean_r2=_skill(test_sse, _sse(test_y, grand_prediction)),
    )


def _normalized_entropy(labels: np.ndarray, state_count: int) -> float:
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not values.size or state_count < 2:
        return 0.0
    counts = np.bincount(values, minlength=state_count).astype(np.float64)
    probabilities = counts[counts > 0.0] / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / np.log(float(state_count))


def _select_train_hub(
    states: np.ndarray,
    *,
    state_count: int,
) -> tuple[int | None, float, float, float, float]:
    labels = np.asarray(states, dtype=np.int64)
    if labels.ndim != 2 or labels.shape[1] < 3:
        return None, 0.0, 0.0, 0.0, 0.0
    middle = labels[:, 1:-1]
    predecessor = labels[:, :-2]
    successor = labels[:, 2:]
    total = int(middle.size)
    best: tuple[float, int, float, float, float] | None = None
    for state in range(state_count):
        mask = middle == state
        occupancy = float(np.count_nonzero(mask) / max(total, 1))
        if not np.any(mask):
            score = 0.0
            previous_entropy = 0.0
            next_entropy = 0.0
        else:
            previous_entropy = _normalized_entropy(
                predecessor[mask],
                state_count,
            )
            next_entropy = _normalized_entropy(successor[mask], state_count)
            score = previous_entropy * (1.0 - next_entropy) * np.sqrt(occupancy)
        candidate = (
            float(score),
            -state,
            float(previous_entropy),
            float(next_entropy),
            occupancy,
        )
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        return None, 0.0, 0.0, 0.0, 0.0
    return -best[1], best[0], best[2], best[3], best[4]


def _hub_sample_tensors(
    design: TrajectoryDesign,
    states: np.ndarray,
    *,
    hub_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(states, dtype=np.int64)
    if labels.shape != design.current.shape[:2]:
        raise ValueError("hub states and trajectory design do not align")
    middle_mask = labels[:, 1:-1] == hub_state
    current = design.current[:, 1:-1, :][middle_mask]
    successor = design.successor[:, 1:-1, :][middle_mask]
    callers = labels[:, :-2][middle_mask]
    anchor_grid = np.broadcast_to(
        design.anchor_indices[None, 1:-1],
        middle_mask.shape,
    )
    anchor_indices = anchor_grid[middle_mask]
    return (
        current[:, None, :],
        successor[:, None, :],
        callers[:, None],
        np.asarray(anchor_indices, dtype=np.int64),
    )


def _fit_affine_predictions(
    train_current: np.ndarray,
    train_successor: np.ndarray,
    test_current: np.ndarray,
    *,
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_x = train_current.reshape(-1, train_current.shape[-1])
    train_y = train_successor.reshape(-1, train_successor.shape[-1])
    coefficients = _ridge_coefficients(
        np.column_stack((np.ones(train_x.shape[0]), train_x)),
        train_y,
        alpha=ridge_alpha,
        penalize_intercept=False,
    )

    def apply(values: np.ndarray) -> np.ndarray:
        flat = values.reshape(-1, values.shape[-1])
        design = np.column_stack((np.ones(flat.shape[0]), flat))
        return (design @ coefficients).reshape(values.shape)

    return apply(train_current), apply(test_current)


def _time_locked_predictions(
    train_successor: np.ndarray,
    train_anchor_indices: np.ndarray,
    test_successor: np.ndarray,
    test_anchor_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    train_y = train_successor.reshape(-1, train_successor.shape[-1])
    test_y = test_successor.reshape(-1, test_successor.shape[-1])
    train_positions = np.asarray(train_anchor_indices, dtype=np.int64)
    test_positions = np.asarray(test_anchor_indices, dtype=np.int64)
    if train_positions.size != train_y.shape[0]:
        raise ValueError("train time indices and hub samples do not align")
    if test_positions.size != test_y.shape[0]:
        raise ValueError("test time indices and hub samples do not align")
    grand_mean = train_y.mean(axis=0)
    means: dict[int, np.ndarray] = {}
    for position in np.unique(train_positions):
        means[int(position)] = train_y[train_positions == position].mean(axis=0)

    def predict(positions: np.ndarray) -> np.ndarray:
        rows = tuple(means.get(int(position), grand_mean) for position in positions)
        return np.asarray(rows, dtype=np.float64)[:, None, :]

    return (
        predict(train_positions),
        predict(test_positions),
        len(means) * train_y.shape[-1],
    )


def evaluate_hub_fold(
    *,
    train_design: TrajectoryDesign,
    test_design: TrajectoryDesign,
    train_states: np.ndarray,
    test_states: np.ndarray,
    gate: PastOnlyGate,
    config: CallGraphProbeConfig,
) -> HubFoldResult:
    """Evaluate a non-overlap common-successor proxy on one outer fold."""

    hub_state, score, previous_entropy, next_entropy, occupancy = _select_train_hub(
        train_states,
        state_count=gate.state_count,
    )
    if hub_state is None:
        return HubFoldResult(
            available=False,
            hub_state=None,
            train_hub_samples=0,
            test_hub_samples=0,
            distinct_train_callers=0,
            predecessor_entropy=0.0,
            successor_entropy=0.0,
            occupancy_fraction=0.0,
            hub_score=0.0,
            shared=None,
            caller_specific=None,
            time_locked=None,
        )
    train_x, train_y, train_callers, train_time = _hub_sample_tensors(
        train_design,
        train_states,
        hub_state=hub_state,
    )
    test_x, test_y, test_callers, test_time = _hub_sample_tensors(
        test_design,
        test_states,
        hub_state=hub_state,
    )
    distinct_callers = int(np.unique(train_callers).size)
    minimum_train = max(
        2 * gate.state_count,
        train_x.shape[-1] + 2,
    )
    if train_x.shape[0] < minimum_train or test_x.shape[0] < 1 or distinct_callers < 2:
        return HubFoldResult(
            available=False,
            hub_state=hub_state,
            train_hub_samples=int(train_x.shape[0]),
            test_hub_samples=int(test_x.shape[0]),
            distinct_train_callers=distinct_callers,
            predecessor_entropy=previous_entropy,
            successor_entropy=next_entropy,
            occupancy_fraction=occupancy,
            hub_score=score,
            shared=None,
            caller_specific=None,
            time_locked=None,
        )

    shared_train, shared_test = _fit_affine_predictions(
        train_x,
        train_y,
        test_x,
        ridge_alpha=config.ridge_alpha,
    )
    caller_predictor = fit_switching_predictor(
        train_x,
        train_y,
        train_callers,
        state_count=gate.state_count,
        ridge_alpha=config.ridge_alpha,
    )
    caller_train = apply_switching_predictor(
        caller_predictor,
        train_x,
        train_callers,
    )
    caller_test = apply_switching_predictor(
        caller_predictor,
        test_x,
        test_callers,
    )
    time_train, time_test, time_parameters = _time_locked_predictions(
        train_y,
        train_time,
        test_y,
        test_time,
    )
    gate_parameters = centroid_parameter_count(
        gate.state_count,
        gate.history_depth,
        gate.latent_rank,
    )
    selection_cost = model_selection_bits(
        config,
        family_count=3,
        includes_history_choice=True,
        includes_hub_choice=True,
        state_count=gate.state_count,
    )
    shared = score_heldout_gaussian_codelength(
        family="hub_shared_successor",
        train_observed=train_y,
        train_predicted=shared_train,
        test_observed=test_y,
        test_predicted=shared_test,
        dynamic_parameter_count=gate.latent_rank * (gate.latent_rank + 1),
        gate_parameter_count=gate_parameters,
        model_selection_cost_bits=selection_cost,
    )
    caller_specific = score_heldout_gaussian_codelength(
        family="hub_caller_specific_successor",
        train_observed=train_y,
        train_predicted=caller_train,
        test_observed=test_y,
        test_predicted=caller_test,
        dynamic_parameter_count=caller_predictor.parameter_count,
        gate_parameter_count=gate_parameters,
        model_selection_cost_bits=selection_cost,
    )
    time_locked = score_heldout_gaussian_codelength(
        family="hub_time_locked_successor",
        train_observed=train_y,
        train_predicted=time_train,
        test_observed=test_y,
        test_predicted=time_test,
        dynamic_parameter_count=time_parameters,
        gate_parameter_count=gate_parameters,
        model_selection_cost_bits=selection_cost,
    )
    return HubFoldResult(
        available=True,
        hub_state=hub_state,
        train_hub_samples=int(train_x.shape[0]),
        test_hub_samples=int(test_x.shape[0]),
        distinct_train_callers=distinct_callers,
        predecessor_entropy=previous_entropy,
        successor_entropy=next_entropy,
        occupancy_fraction=occupancy,
        hub_score=score,
        shared=shared,
        caller_specific=caller_specific,
        time_locked=time_locked,
    )


def whiten_prepared_latent_fold(
    prepared: PreparedLatentFold,
) -> WhitenedLatentFold:
    """Whiten latent coordinates using source training samples only."""

    if not isinstance(prepared, PreparedLatentFold):
        raise TypeError("prepared must be PreparedLatentFold")
    source_train = _validate_latent(
        prepared.source_train,
        label="source_train",
    )
    flat = source_train.reshape(-1, source_train.shape[-1])
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)

    def apply(values: np.ndarray) -> np.ndarray:
        latent = _validate_latent(values, label="latent whitened by source")
        if latent.shape[-1] != mean.size:
            raise ValueError("latent rank differs from source whitening")
        return (latent - mean[None, None, :]) / scale[None, None, :]

    return WhitenedLatentFold(
        source_train=apply(prepared.source_train),
        source_test=apply(prepared.source_test),
        target_train=apply(prepared.target_train_in_source_coordinates),
        target_test=apply(prepared.target_test_in_source_coordinates),
        latent_mean=np.asarray(mean, dtype=np.float64),
        latent_scale=np.asarray(scale, dtype=np.float64),
    )


def _unavailable_hub() -> HubFoldResult:
    return HubFoldResult(
        available=False,
        hub_state=None,
        train_hub_samples=0,
        test_hub_samples=0,
        distinct_train_callers=0,
        predecessor_entropy=0.0,
        successor_entropy=0.0,
        occupancy_fraction=0.0,
        hub_score=0.0,
        shared=None,
        caller_specific=None,
        time_locked=None,
    )


def _fit_and_score_direction(
    train_latent: np.ndarray,
    test_latent: np.ndarray,
    *,
    state_count: int,
    history_depth: int,
    reverse: bool,
    config: CallGraphProbeConfig,
    seed_tokens: tuple[Any, ...],
) -> tuple[
    CodelengthResult,
    CodelengthResult,
    CodelengthResult,
    HubFoldResult,
]:
    anchor_depth = max(state_count, history_depth)
    switching_train = build_trajectory_design(
        train_latent,
        history_depth=history_depth,
        anchor_history_depth=anchor_depth,
        lag_bins=config.lag_bins,
        stride_bins=config.primary_stride_bins,
        reverse=reverse,
    )
    switching_test = build_trajectory_design(
        test_latent,
        history_depth=history_depth,
        anchor_history_depth=anchor_depth,
        lag_bins=config.lag_bins,
        stride_bins=config.primary_stride_bins,
        reverse=reverse,
    )
    var_train = build_trajectory_design(
        train_latent,
        history_depth=state_count,
        anchor_history_depth=anchor_depth,
        lag_bins=config.lag_bins,
        stride_bins=config.primary_stride_bins,
        reverse=reverse,
    )
    var_test = build_trajectory_design(
        test_latent,
        history_depth=state_count,
        anchor_history_depth=anchor_depth,
        lag_bins=config.lag_bins,
        stride_bins=config.primary_stride_bins,
        reverse=reverse,
    )
    if not np.array_equal(
        switching_train.anchor_indices,
        var_train.anchor_indices,
    ) or not np.array_equal(
        switching_test.anchor_indices,
        var_test.anchor_indices,
    ):
        raise RuntimeError("switching and VAR anchors diverged")
    if not np.array_equal(switching_train.current, var_train.current):
        raise RuntimeError("switching and VAR train currents diverged")
    if not np.array_equal(switching_test.successor, var_test.successor):
        raise RuntimeError("switching and VAR test targets diverged")

    rank = switching_train.current.shape[-1]
    gate = fit_past_only_gate(
        switching_train.history,
        state_count=state_count,
        history_depth=history_depth,
        latent_rank=rank,
        seed=_derived_seed(
            config.seed,
            *seed_tokens,
            "reverse" if reverse else "forward",
            state_count,
            history_depth,
        ),
        restarts=config.kmeans_restarts,
        max_iterations=config.kmeans_max_iterations,
    )
    train_states = assign_past_only_states(gate, switching_train.history)
    test_states = assign_past_only_states(gate, switching_test.history)
    switching = fit_switching_predictor(
        switching_train.current,
        switching_train.successor,
        train_states,
        state_count=state_count,
        ridge_alpha=config.ridge_alpha,
    )
    switching_train_prediction = apply_switching_predictor(
        switching,
        switching_train.current,
        train_states,
    )
    switching_test_prediction = apply_switching_predictor(
        switching,
        switching_test.current,
        test_states,
    )
    var = fit_var_predictor(
        var_train.history,
        var_train.successor,
        history_order=state_count,
        latent_rank=rank,
        ridge_alpha=config.ridge_alpha,
    )
    var_train_prediction = apply_var_predictor(var, var_train.history)
    var_test_prediction = apply_var_predictor(var, var_test.history)
    parent_rank1 = fit_state_parent_rank1_predictor(
        switching_train.current,
        switching_train.successor,
        train_states,
        state_count=state_count,
        ridge_alpha=config.ridge_alpha,
    )
    parent_train_prediction = apply_state_parent_rank1_predictor(
        parent_rank1,
        switching_train.current,
        train_states,
    )
    parent_test_prediction = apply_state_parent_rank1_predictor(
        parent_rank1,
        switching_test.current,
        test_states,
    )
    gate_parameters = centroid_parameter_count(
        state_count,
        history_depth,
        rank,
    )
    switching_selection = model_selection_bits(
        config,
        family_count=3,
        includes_history_choice=True,
        includes_hub_choice=False,
        state_count=state_count,
    )
    var_selection = model_selection_bits(
        config,
        family_count=3,
        includes_history_choice=False,
        includes_hub_choice=False,
        state_count=state_count,
        includes_state_label_permutation=False,
    )
    switching_score = score_heldout_gaussian_codelength(
        family="past_gated_switching_current_map",
        train_observed=switching_train.successor,
        train_predicted=switching_train_prediction,
        test_observed=switching_test.successor,
        test_predicted=switching_test_prediction,
        dynamic_parameter_count=switching.parameter_count,
        gate_parameter_count=gate_parameters,
        model_selection_cost_bits=switching_selection,
    )
    var_score = score_heldout_gaussian_codelength(
        family=f"stationary_var_order_{state_count}",
        train_observed=var_train.successor,
        train_predicted=var_train_prediction,
        test_observed=var_test.successor,
        test_predicted=var_test_prediction,
        dynamic_parameter_count=var.parameter_count,
        gate_parameter_count=0,
        model_selection_cost_bits=var_selection,
    )
    parent_score = score_heldout_gaussian_codelength(
        family="state_parent_plus_rank1_residual",
        train_observed=switching_train.successor,
        train_predicted=parent_train_prediction,
        test_observed=switching_test.successor,
        test_predicted=parent_test_prediction,
        dynamic_parameter_count=parent_rank1.parameter_count,
        gate_parameter_count=gate_parameters,
        model_selection_cost_bits=switching_selection,
    )
    if config.primary_stride_bins == config.lag_bins:
        hub = evaluate_hub_fold(
            train_design=switching_train,
            test_design=switching_test,
            train_states=train_states,
            test_states=test_states,
            gate=gate,
            config=config,
        )
    else:
        hub = _unavailable_hub()
    return switching_score, var_score, parent_score, hub


def evaluate_fold_comparison(
    prepared: PreparedLatentFold,
    *,
    state_count: int,
    history_depth: int,
    config: CallGraphProbeConfig,
    seed_tokens: tuple[Any, ...] = (),
) -> FoldComparison:
    """Evaluate one source-train-frozen outer fold."""

    whitened = whiten_prepared_latent_fold(prepared)
    switching, var, parent, hub = _fit_and_score_direction(
        whitened.source_train,
        whitened.target_test,
        state_count=state_count,
        history_depth=history_depth,
        reverse=False,
        config=config,
        seed_tokens=(*seed_tokens, prepared.fold.index_zero_based),
    )
    if config.run_reverse_descriptive_control:
        reverse_switching, reverse_var, _, reverse_hub = _fit_and_score_direction(
            whitened.source_train,
            whitened.target_test,
            state_count=state_count,
            history_depth=history_depth,
            reverse=True,
            config=config,
            seed_tokens=(*seed_tokens, prepared.fold.index_zero_based),
        )
    else:
        reverse_switching = None
        reverse_var = None
        reverse_hub = None
    return FoldComparison(
        fold_index_zero_based=prepared.fold.index_zero_based,
        latent_rank=prepared.transform.rank,
        active_neuron_count=prepared.transform.active_neuron_count,
        switching=switching,
        matched_var=var,
        state_parent_rank1=parent,
        reverse_switching=reverse_switching,
        reverse_var=reverse_var,
        hub=hub,
        reverse_hub=reverse_hub,
        gate_uses_current_and_past_only=True,
        test_target_passed_to_gate=False,
    )


def _sum_codelength(
    results: Sequence[CodelengthResult],
) -> tuple[float, float, int]:
    items = tuple(results)
    if not items:
        raise ValueError("at least one codelength result is required")
    return (
        float(sum(item.test_sse for item in items)),
        float(sum(item.total_codelength_bits for item in items)),
        int(sum(item.test_scalar_count for item in items)),
    )


def summarize_session_model(
    *,
    session: SessionSpec,
    dimension: int,
    state_count: int,
    history_depth: int,
    event_mean_removed: bool,
    stride_bins: int,
    fold_results: Sequence[FoldComparison],
) -> SessionModelResult:
    """Aggregate outer folds without treating anchors as inference units."""

    folds = tuple(fold_results)
    if not folds:
        raise ValueError("fold_results must not be empty")
    switching_sse, switching_bits, scalar_count = _sum_codelength(
        tuple(item.switching for item in folds)
    )
    var_sse, var_bits, var_scalar_count = _sum_codelength(tuple(item.matched_var for item in folds))
    parent_sse, parent_bits, parent_scalar_count = _sum_codelength(
        tuple(item.state_parent_rank1 for item in folds)
    )
    if scalar_count != var_scalar_count or scalar_count != parent_scalar_count:
        raise RuntimeError("competing models used different held-out scalars")
    reverse_items = tuple(
        item.reverse_switching for item in folds if item.reverse_switching is not None
    )
    if len(reverse_items) == len(folds):
        reverse_sse, reverse_bits, reverse_scalars = _sum_codelength(reverse_items)
        forward_vs_reverse = _skill(switching_sse, reverse_sse)
        forward_code_advantage = (reverse_bits - switching_bits) / max(reverse_scalars, 1)
    else:
        forward_vs_reverse = None
        forward_code_advantage = None

    available_hubs = tuple(item.hub for item in folds if item.hub.available)
    hub_available_fold_count = len(available_hubs)
    hub_all_folds_available = hub_available_fold_count == len(folds)
    if available_hubs and all(
        item.shared is not None
        and item.caller_specific is not None
        and item.time_locked is not None
        for item in available_hubs
    ):
        shared_items = tuple(item.shared for item in available_hubs if item.shared is not None)
        caller_items = tuple(
            item.caller_specific for item in available_hubs if item.caller_specific is not None
        )
        time_items = tuple(
            item.time_locked for item in available_hubs if item.time_locked is not None
        )
        shared_sse, shared_bits, shared_scalars = _sum_codelength(shared_items)
        caller_sse, caller_bits, caller_scalars = _sum_codelength(caller_items)
        time_sse, time_bits, time_scalars = _sum_codelength(time_items)
        if shared_scalars != caller_scalars or shared_scalars != time_scalars:
            raise RuntimeError("hub competitors used different held-out scalars")
        hub_shared_vs_time = _skill(shared_sse, time_sse)
        hub_code_vs_time = (time_bits - shared_bits) / max(time_scalars, 1)
        hub_code_vs_caller = (caller_bits - shared_bits) / max(caller_scalars, 1)
        _ = caller_sse
    else:
        hub_shared_vs_time = None
        hub_code_vs_time = None
        hub_code_vs_caller = None
    paired_hubs = tuple(
        (item.hub, item.reverse_hub)
        for item in folds
        if item.hub.available
        and item.hub.shared is not None
        and item.reverse_hub is not None
        and item.reverse_hub.available
        and item.reverse_hub.shared is not None
    )
    if paired_hubs:
        forward_hub_items = tuple(
            forward.shared for forward, _ in paired_hubs if forward.shared is not None
        )
        reverse_hub_items = tuple(
            reverse.shared
            for _, reverse in paired_hubs
            if reverse is not None and reverse.shared is not None
        )
        forward_hub_sse, forward_hub_bits, forward_hub_scalars = _sum_codelength(forward_hub_items)
        reverse_hub_sse, reverse_hub_bits, reverse_hub_scalars = _sum_codelength(reverse_hub_items)
        hub_forward_vs_reverse = _skill(
            forward_hub_sse / max(forward_hub_scalars, 1),
            reverse_hub_sse / max(reverse_hub_scalars, 1),
        )
        hub_forward_code_vs_reverse = reverse_hub_bits / max(
            reverse_hub_scalars, 1
        ) - forward_hub_bits / max(forward_hub_scalars, 1)
    else:
        hub_forward_vs_reverse = None
        hub_forward_code_vs_reverse = None
    return SessionModelResult(
        analysis_key=(f"within_dim{dimension}_states{state_count}_history{history_depth}"),
        session_index_one_based=session.index_one_based,
        animal=session.animal,
        neuron_count=session.neuron_count,
        dimension=dimension,
        state_count=state_count,
        history_depth=history_depth,
        var_order=state_count,
        parameter_matched_dynamic_block=(
            switching_parameter_count(
                state_count,
                folds[0].latent_rank,
            )
            == var_parameter_count(state_count, folds[0].latent_rank)
        ),
        event_mean_removed=event_mean_removed,
        stride_bins=stride_bins,
        fold_results=folds,
        switching_vs_var_skill=_skill(switching_sse, var_sse),
        switching_codelength_advantage_bits_per_scalar=(
            (var_bits - switching_bits) / max(scalar_count, 1)
        ),
        forward_vs_reverse_skill=forward_vs_reverse,
        forward_codelength_advantage_over_reverse_bits_per_scalar=(forward_code_advantage),
        state_parent_rank1_vs_var_skill=_skill(parent_sse, var_sse),
        state_parent_rank1_codelength_advantage_bits_per_scalar=(
            (var_bits - parent_bits) / max(scalar_count, 1)
        ),
        hub_available_fold_count=hub_available_fold_count,
        hub_total_fold_count=len(folds),
        hub_all_folds_available=hub_all_folds_available,
        hub_shared_vs_time_skill=hub_shared_vs_time,
        hub_shared_codelength_advantage_over_time_bits_per_scalar=(hub_code_vs_time),
        hub_shared_codelength_advantage_over_caller_bits_per_scalar=(hub_code_vs_caller),
        hub_forward_vs_reverse_skill=hub_forward_vs_reverse,
        hub_forward_codelength_advantage_over_reverse_bits_per_scalar=(hub_forward_code_vs_reverse),
    )


def _validate_population(
    population: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    values = np.asarray(population, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"{label} must be trial x neuron x time")
    if min(values.shape) < 1:
        raise ValueError(f"{label} axes must not be empty")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{label} must contain finite nonnegative counts")
    return values


def _run_session_model_grid(
    dimensions: Mapping[int, np.ndarray],
    session_specs: tuple[SessionSpec, ...],
    folds: tuple[WholeTrialFold, ...],
    *,
    config: CallGraphProbeConfig,
    event_mean_removed: bool,
) -> tuple[SessionModelResult, ...]:
    results = []
    for session in session_specs:
        column_slice = slice(
            session.column_start_zero_based,
            session.column_stop_exclusive,
        )
        for dimension in (1, 3):
            population = dimensions[dimension][:, column_slice, :]
            for state_count in config.state_counts:
                for history_depth in config.history_depths:
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
                            evaluate_fold_comparison(
                                prepared,
                                state_count=state_count,
                                history_depth=history_depth,
                                config=config,
                                seed_tokens=(
                                    "within",
                                    session.index_one_based,
                                    dimension,
                                    int(event_mean_removed),
                                ),
                            )
                        )
                    results.append(
                        summarize_session_model(
                            session=session,
                            dimension=dimension,
                            state_count=state_count,
                            history_depth=history_depth,
                            event_mean_removed=event_mean_removed,
                            stride_bins=config.primary_stride_bins,
                            fold_results=fold_results,
                        )
                    )
    return tuple(results)


def _evaluate_frozen_transfer_fold(
    prepared: PreparedLatentFold,
    *,
    state_count: int,
    history_depth: int,
    config: CallGraphProbeConfig,
    seed_tokens: tuple[Any, ...],
) -> tuple[CodelengthResult, CodelengthResult]:
    whitened = whiten_prepared_latent_fold(prepared)
    frozen, _, _, _ = _fit_and_score_direction(
        whitened.source_train,
        whitened.target_test,
        state_count=state_count,
        history_depth=history_depth,
        reverse=False,
        config=config,
        seed_tokens=(*seed_tokens, "source_frozen"),
    )
    target_refit, _, _, _ = _fit_and_score_direction(
        whitened.target_train,
        whitened.target_test,
        state_count=state_count,
        history_depth=history_depth,
        reverse=False,
        config=config,
        seed_tokens=(*seed_tokens, "target_refit"),
    )
    return frozen, target_refit


def _run_frozen_transfer_grid(
    dimensions: Mapping[int, np.ndarray],
    session_specs: tuple[SessionSpec, ...],
    folds: tuple[WholeTrialFold, ...],
    *,
    config: CallGraphProbeConfig,
    event_mean_removed: bool,
) -> tuple[FrozenTransferResult, ...]:
    results = []
    for session in session_specs:
        column_slice = slice(
            session.column_start_zero_based,
            session.column_stop_exclusive,
        )
        for source_dimension, target_dimension in ((1, 3), (3, 1)):
            source = dimensions[source_dimension][:, column_slice, :]
            target = dimensions[target_dimension][:, column_slice, :]
            for state_count in config.state_counts:
                history_depth = state_count
                frozen_folds = []
                refit_folds = []
                for fold in folds:
                    prepared = prepare_session_latent_fold(
                        source,
                        target,
                        fold,
                        rank_cap=config.rank_cap,
                        event_mean_removed=event_mean_removed,
                    )
                    frozen, target_refit = _evaluate_frozen_transfer_fold(
                        prepared,
                        state_count=state_count,
                        history_depth=history_depth,
                        config=config,
                        seed_tokens=(
                            "transfer",
                            session.index_one_based,
                            source_dimension,
                            target_dimension,
                            state_count,
                            int(event_mean_removed),
                            fold.index_zero_based,
                        ),
                    )
                    frozen_folds.append(frozen)
                    refit_folds.append(target_refit)
                frozen_sse, frozen_bits, frozen_scalars = _sum_codelength(frozen_folds)
                refit_sse, refit_bits, refit_scalars = _sum_codelength(refit_folds)
                if frozen_scalars != refit_scalars:
                    raise RuntimeError("frozen and target-refit transfer scalars diverged")
                results.append(
                    FrozenTransferResult(
                        analysis_key=(
                            f"frozen_dim{source_dimension}_to_dim"
                            f"{target_dimension}_states{state_count}"
                        ),
                        session_index_one_based=session.index_one_based,
                        animal=session.animal,
                        neuron_count=session.neuron_count,
                        source_dimension=source_dimension,
                        target_dimension=target_dimension,
                        state_count=state_count,
                        history_depth=history_depth,
                        event_mean_removed=event_mean_removed,
                        frozen_test_sse=frozen_sse,
                        target_refit_test_sse=refit_sse,
                        frozen_vs_target_refit_skill=_skill(
                            frozen_sse,
                            refit_sse,
                        ),
                        frozen_codelength_advantage_over_target_refit_bits_per_scalar=(
                            (refit_bits - frozen_bits) / max(frozen_scalars, 1)
                        ),
                        source_representation_and_gate_frozen=True,
                        target_rows_paired_to_source_rows=False,
                    )
                )
    return tuple(results)


def _median(values: Sequence[float]) -> float:
    items = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if not items.size:
        raise ValueError("median requires at least one value")
    return float(np.median(items))


def _optional_median(values: Sequence[float | None]) -> float | None:
    finite = tuple(float(value) for value in values if value is not None)
    if not finite:
        return None
    return _median(finite)


def aggregate_session_models(
    results: Sequence[SessionModelResult],
) -> tuple[AggregateResult, ...]:
    """Aggregate first over session×dimension units, never over anchors."""

    items = tuple(results)
    keys = sorted(
        {
            (
                item.state_count,
                item.history_depth,
                item.event_mean_removed,
            )
            for item in items
        }
    )
    animals = tuple(sorted({item.animal for item in items}))
    aggregates = []
    for state_count, history_depth, event_mean_removed in keys:
        base = tuple(
            item
            for item in items
            if item.state_count == state_count
            and item.history_depth == history_depth
            and item.event_mean_removed == event_mean_removed
        )
        for animal in ("all", *animals):
            group = (
                base if animal == "all" else tuple(item for item in base if item.animal == animal)
            )
            if not group:
                continue
            aggregates.append(
                AggregateResult(
                    state_count=state_count,
                    history_depth=history_depth,
                    event_mean_removed=event_mean_removed,
                    animal=animal,
                    unit_count=len(group),
                    median_switching_vs_var_skill=_median(
                        tuple(item.switching_vs_var_skill for item in group)
                    ),
                    median_switching_codelength_advantage_bits_per_scalar=(
                        _median(
                            tuple(
                                item.switching_codelength_advantage_bits_per_scalar
                                for item in group
                            )
                        )
                    ),
                    median_forward_vs_reverse_skill=_optional_median(
                        tuple(item.forward_vs_reverse_skill for item in group)
                    ),
                    median_forward_codelength_advantage_over_reverse_bits_per_scalar=(
                        _optional_median(
                            tuple(
                                item.forward_codelength_advantage_over_reverse_bits_per_scalar
                                for item in group
                            )
                        )
                    ),
                    median_state_parent_rank1_vs_var_skill=_median(
                        tuple(item.state_parent_rank1_vs_var_skill for item in group)
                    ),
                    median_state_parent_rank1_codelength_advantage_bits_per_scalar=(
                        _median(
                            tuple(
                                item.state_parent_rank1_codelength_advantage_bits_per_scalar
                                for item in group
                            )
                        )
                    ),
                    all_units_have_complete_hub_folds=all(
                        item.hub_all_folds_available for item in group
                    ),
                    median_hub_shared_vs_time_skill=_optional_median(
                        tuple(item.hub_shared_vs_time_skill for item in group)
                    ),
                    median_hub_shared_codelength_advantage_over_time_bits_per_scalar=(
                        _optional_median(
                            tuple(
                                item.hub_shared_codelength_advantage_over_time_bits_per_scalar
                                for item in group
                            )
                        )
                    ),
                    median_hub_shared_codelength_advantage_over_caller_bits_per_scalar=(
                        _optional_median(
                            tuple(
                                item.hub_shared_codelength_advantage_over_caller_bits_per_scalar
                                for item in group
                            )
                        )
                    ),
                    median_hub_forward_vs_reverse_skill=_optional_median(
                        tuple(item.hub_forward_vs_reverse_skill for item in group)
                    ),
                )
            )
    return tuple(aggregates)


def _aggregate_lookup(
    aggregates: Sequence[AggregateResult],
    *,
    state_count: int,
    event_mean_removed: bool,
    animal: str,
) -> AggregateResult | None:
    matches = tuple(
        item
        for item in aggregates
        if item.state_count == state_count
        and item.history_depth == state_count
        and item.event_mean_removed == event_mean_removed
        and item.animal == animal
    )
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError("aggregate key is not unique")
    return matches[0]


def _positive(value: float | None) -> bool:
    return value is not None and np.isfinite(value) and value > 0.0


def _switching_proxy_passes(
    aggregates: Sequence[AggregateResult],
    *,
    state_count: int,
) -> bool:
    groups = tuple(
        _aggregate_lookup(
            aggregates,
            state_count=state_count,
            event_mean_removed=event_mean_removed,
            animal=animal,
        )
        for event_mean_removed in (False, True)
        for animal in ("all", "Chico", "Silas")
    )
    if any(group is None for group in groups):
        return False
    return all(
        _positive(group.median_switching_vs_var_skill)
        and _positive(group.median_switching_codelength_advantage_bits_per_scalar)
        and _positive(group.median_forward_vs_reverse_skill)
        for group in groups
        if group is not None
    )


def _hub_proxy_passes(
    aggregates: Sequence[AggregateResult],
    *,
    state_count: int,
) -> bool:
    groups = tuple(
        _aggregate_lookup(
            aggregates,
            state_count=state_count,
            event_mean_removed=event_mean_removed,
            animal=animal,
        )
        for event_mean_removed in (False, True)
        for animal in ("all", "Chico", "Silas")
    )
    if any(group is None for group in groups):
        return False
    return all(
        group.all_units_have_complete_hub_folds
        and _positive(group.median_hub_shared_vs_time_skill)
        and _positive(group.median_hub_shared_codelength_advantage_over_time_bits_per_scalar)
        and _positive(group.median_hub_shared_codelength_advantage_over_caller_bits_per_scalar)
        and _positive(group.median_hub_forward_vs_reverse_skill)
        for group in groups
        if group is not None
    )


def _state_parent_proxy_passes(
    aggregates: Sequence[AggregateResult],
    *,
    state_count: int,
) -> bool:
    groups = tuple(
        _aggregate_lookup(
            aggregates,
            state_count=state_count,
            event_mean_removed=event_mean_removed,
            animal=animal,
        )
        for event_mean_removed in (False, True)
        for animal in ("all", "Chico", "Silas")
    )
    if any(group is None for group in groups):
        return False
    return all(
        _positive(group.median_state_parent_rank1_vs_var_skill)
        and _positive(group.median_state_parent_rank1_codelength_advantage_bits_per_scalar)
        for group in groups
        if group is not None
    )


def _frozen_transfer_passes(
    results: Sequence[FrozenTransferResult],
    *,
    state_count: int,
) -> bool:
    items = tuple(item for item in results if item.state_count == state_count)
    for event_mean_removed in (False, True):
        for animal in ("Chico", "Silas"):
            for direction in ((1, 3), (3, 1)):
                group = tuple(
                    item
                    for item in items
                    if item.event_mean_removed == event_mean_removed
                    and item.animal == animal
                    and (
                        item.source_dimension,
                        item.target_dimension,
                    )
                    == direction
                )
                if not group:
                    return False
                if not _positive(
                    _median(tuple(item.frozen_vs_target_refit_skill for item in group))
                ):
                    return False
                if not _positive(
                    _median(
                        tuple(
                            item.frozen_codelength_advantage_over_target_refit_bits_per_scalar
                            for item in group
                        )
                    )
                ):
                    return False
    return True


def _build_verdicts(
    aggregates: Sequence[AggregateResult],
    transfer_results: Sequence[FrozenTransferResult],
    *,
    config: CallGraphProbeConfig,
) -> tuple[ClaimVerdict, ...]:
    sensitivity_complete = config.run_event_mean_removed_sensitivity
    switching_pass = sensitivity_complete and any(
        _switching_proxy_passes(aggregates, state_count=state_count)
        for state_count in config.state_counts
    )
    hub_pass = sensitivity_complete and any(
        _hub_proxy_passes(aggregates, state_count=state_count)
        for state_count in config.state_counts
    )
    state_parent_pass = sensitivity_complete and any(
        _state_parent_proxy_passes(aggregates, state_count=state_count)
        for state_count in config.state_counts
    )
    transfer_pass = sensitivity_complete and any(
        _frozen_transfer_passes(
            transfer_results,
            state_count=state_count,
        )
        for state_count in config.state_counts
    )
    frontend_callee_candidate_pass = sensitivity_complete and any(
        _hub_proxy_passes(aggregates, state_count=state_count)
        and _frozen_transfer_passes(
            transfer_results,
            state_count=state_count,
        )
        for state_count in config.state_counts
    )
    incomplete_answer = PENDING if not sensitivity_complete else NO
    return (
        ClaimVerdict(
            key="past_only_gate_contract_verified",
            answer=YES,
            reason=(
                "The frozen gate API accepts current/past history only; "
                "the held-out successor is not an argument."
            ),
        ),
        ClaimVerdict(
            key="nonoverlap_session_local_proxy_completed",
            answer=YES,
            reason=(
                "Models use whole-trial outer folds, recovered session slices, "
                "and the primary 100 ms stride."
            ),
        ),
        ClaimVerdict(
            key="state_dependent_switching_proxy_passed",
            answer=YES if switching_pass else incomplete_answer,
            reason=(
                "A predeclared observational switching proxy beats matched VAR, "
                "reverse, and event-mean-removed controls in both animals."
                if switching_pass
                else "The complete predeclared numerical switching gates do not pass."
            ),
        ),
        ClaimVerdict(
            key="state_dependent_switching_operator_identified",
            answer=PENDING if switching_pass else NO,
            reason=(
                "The observational proxy passes, but two animals and a processed "
                "snapshot do not identify a biological operator."
                if switching_pass
                else "No state-dependent switching operator is identified by "
                "this snapshot and model class."
            ),
        ),
        ClaimVerdict(
            key="latent_common_successor_proxy_passed",
            answer=YES if hub_pass else incomplete_answer,
            reason=(
                "A train-selected hub has caller-diverse entry and a shared "
                "successor model that beats caller, time, reverse, and "
                "event-mean controls in both animals."
                if hub_pass
                else "The complete observational common-successor gates do not pass."
            ),
        ),
        ClaimVerdict(
            key="frontend_to_common_callee_observational_candidate_supported",
            answer=YES if frontend_callee_candidate_pass else incomplete_answer,
            reason=(
                "Both the within-dimension common-successor proxy and frozen "
                "bidirectional D1↔D3 transfer pass every declared control."
                if frontend_callee_candidate_pass
                else "A within-dimension hub alone is not frontend-to-callee "
                "evidence; hub and frozen cross-dimension transfer do not both pass."
            ),
        ),
        ClaimVerdict(
            key="biological_common_callee_assembly_identified",
            answer=NO,
            reason=(
                "A latent bottleneck is not a biological call boundary, and no "
                "selective perturbation or rescue is present."
            ),
        ),
        ClaimVerdict(
            key="common_callee_architecture_exists_or_is_absent",
            answer=TEST_UNAVAILABLE,
            reason=(
                "This single-area processed snapshot cannot decide universal "
                "existence or absence of a common callee architecture."
            ),
        ),
        ClaimVerdict(
            key="state_parent_rank1_proxy_passed",
            answer=YES if state_parent_pass else incomplete_answer,
            reason=(
                "A state-level affine parent plus rank-one state residuals beats "
                "matched VAR after controls."
                if state_parent_pass
                else "The state-level low-rank proxy does not pass all controls."
            ),
        ),
        ClaimVerdict(
            key="task_inheritance_tree_identified",
            answer=NO,
            reason=(
                "State-level residual sharing is not task inheritance; justified "
                "cross-task common coordinates and lineage evidence are absent."
            ),
        ),
        ClaimVerdict(
            key="task_inheritance_architecture_exists_or_is_absent",
            answer=TEST_UNAVAILABLE,
            reason=(
                "The snapshot cannot decide universal existence or absence of a "
                "task-inheritance architecture."
            ),
        ),
        ClaimVerdict(
            key="frozen_cross_dimension_switching_transfer_passed",
            answer=YES if transfer_pass else incomplete_answer,
            reason=(
                "Source representation, gate, and operator beat a target-refit "
                "operator in both directions, both animals, and both sensitivities."
                if transfer_pass
                else "Frozen D1↔D3 transfer does not pass every declared gate."
            ),
        ),
        ClaimVerdict(
            key="continuous_dynamics_ruled_out",
            answer=NO,
            reason=(
                "Beating or losing to one matched VAR does not rule out the wider "
                "continuous nonlinear dynamics family."
            ),
        ),
        ClaimVerdict(
            key="causal_call_return_identified",
            answer=NO,
            reason=(
                "No caller/callee perturbation, saved continuation state, return "
                "branch intervention, or rescue is available."
            ),
        ),
        ClaimVerdict(
            key="causal_call_return_architecture_exists_or_is_absent",
            answer=TEST_UNAVAILABLE,
            reason=(
                "The released observational snapshot cannot decide universal "
                "existence or absence of causal call/return machinery."
            ),
        ),
        ClaimVerdict(
            key="brain_programming_language_identified",
            answer=NO,
            reason=(
                "Observational switching and convergence proxies are not an opcode "
                "library, grammar, unseen composition, or causal semantics."
            ),
        ),
    )


def validate_call_graph_claim_locks(locks: CallGraphClaimLocks) -> None:
    """Reject any scientific inference that the probe cannot unlock."""

    if not isinstance(locks, CallGraphClaimLocks):
        raise TypeError("locks must be CallGraphClaimLocks")
    unlocked = tuple(name for name, value in asdict(locks).items() if value is not False)
    if unlocked:
        raise ValueError(f"call-graph claim locks must remain false: {unlocked}")


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


def run_tafazoli_call_graph_probe_from_arrays(
    dimension_one_train: np.ndarray,
    dimension_three_train: np.ndarray,
    *,
    config: CallGraphProbeConfig = CallGraphProbeConfig(),
    session_specs: Sequence[SessionSpec] | None = None,
) -> TafazoliCallGraphProbeReport:
    """Run the NumPy-only call-graph proxy core on allowed training tensors."""

    if not isinstance(config, CallGraphProbeConfig):
        raise TypeError("config must be CallGraphProbeConfig")
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
    if dim1.shape[0] < config.fold_count:
        raise ValueError("not enough pseudo-trial rows for whole-trial CV")
    maximum_depth = max((*config.state_counts, *config.history_depths))
    if dim1.shape[2] <= maximum_depth * config.lag_bins:
        raise ValueError("not enough timepoints for the requested histories")
    if config.primary_stride_bins < config.lag_bins:
        raise ValueError("primary stride must not reuse overlapping count windows")
    if session_specs is None:
        specs = recovered_session_specs()
    else:
        specs = tuple(session_specs)
    specs = _validate_session_specs(specs, neuron_count=dim1.shape[1])
    folds = make_whole_trial_folds(
        dim1.shape[0],
        fold_count=config.fold_count,
        seed=config.seed,
    )
    dimensions = {1: dim1, 3: dim3}
    model_results = _run_session_model_grid(
        dimensions,
        specs,
        folds,
        config=config,
        event_mean_removed=False,
    )
    transfer_results = _run_frozen_transfer_grid(
        dimensions,
        specs,
        folds,
        config=config,
        event_mean_removed=False,
    )
    if config.run_event_mean_removed_sensitivity:
        residual_results = _run_session_model_grid(
            dimensions,
            specs,
            folds,
            config=config,
            event_mean_removed=True,
        )
        residual_transfer_results = _run_frozen_transfer_grid(
            dimensions,
            specs,
            folds,
            config=config,
            event_mean_removed=True,
        )
    else:
        residual_results = ()
        residual_transfer_results = ()
    aggregates = aggregate_session_models((*model_results, *residual_results))
    verdicts = _build_verdicts(
        aggregates,
        (*transfer_results, *residual_transfer_results),
        config=config,
    )
    locks = CallGraphClaimLocks()
    validate_call_graph_claim_locks(locks)
    inherited_locks = StationaryClaimLocks()
    if any(asdict(inherited_locks).values()):
        raise ValueError("inherited stationary claim locks must remain false")
    report = TafazoliCallGraphProbeReport(
        schema_version=SCHEMA_VERSION,
        scope=PROBE_SCOPE,
        method_status="CALL_GRAPH_OBSERVATIONAL_PROXY_PROBE_COMPLETE",
        source_file_md5=None,
        official_checksum_verified=False,
        config=config,
        session_specs=specs,
        fields_used_for_fitting=(
            "ClassifierOpts.Dimpredictors[dimension=1].train",
            "ClassifierOpts.Dimpredictors[dimension=3].train",
        ),
        blind_fields_used=(),
        saved_test_role="not_used",
        train_only_preprocessing=True,
        primary_inference_unit="recording_session_x_dimension",
        codelength_name="heldout Gaussian codelength/BIC proxy",
        model_results=model_results,
        event_mean_removed_results=tuple(residual_results),
        frozen_transfer_results=transfer_results,
        event_mean_removed_frozen_transfer_results=tuple(residual_transfer_results),
        aggregates=aggregates,
        verdicts=verdicts,
        claim_locks=locks,
        inherited_stationary_claim_locks=inherited_locks,
        limitations=(
            "The 36 rows are saved pseudo-trials from one overwritten classifier fold.",
            "The 27 recovered sessions, not neurons or anchors, are analysis units.",
            "Only two animals are available; no population-level inference is claimed.",
            "The primary 100 ms stride avoids counting 90%-overlapping windows as samples.",
            "The codelength is a held-out Gaussian/BIC proxy, not strict prequential MDL.",
            "D1 and D3 row numbers are never interpreted as paired biological trials.",
            "A train-selected latent hub can also be a continuous attractor or event funnel.",
            "State parent-plus-rank1 sharing is not task inheritance.",
            "No intervention, rescue, unseen composition, or call-return recording exists.",
        ),
        conclusion=(
            "This probe can accept or reject narrow observational switching, "
            "common-successor, frozen-transfer, and state-low-rank proxies. "
            "It cannot identify a biological common callee, task-inheritance "
            "tree, causal call/return graph, or brain programming language."
        ),
    )
    validate_call_graph_probe_report(report)
    return report


def run_tafazoli_call_graph_probe(
    classifier_file: str | Path,
    *,
    config: CallGraphProbeConfig = CallGraphProbeConfig(),
) -> TafazoliCallGraphProbeReport:
    """Checksum-lock the official MAT snapshot and run the strict core."""

    observed_md5 = verify_official_classifier_checksum(classifier_file)
    dim1, dim3 = load_tafazoli_train_dimensions(classifier_file)
    report = run_tafazoli_call_graph_probe_from_arrays(
        dim1,
        dim3,
        config=config,
    )
    verified = replace(
        report,
        source_file_md5=observed_md5,
        official_checksum_verified=True,
    )
    validate_call_graph_probe_report(verified)
    return verified


def validate_call_graph_probe_report(
    report: TafazoliCallGraphProbeReport,
) -> None:
    """Reject leakage, pseudopopulation fitting, and claim upgrades."""

    if not isinstance(report, TafazoliCallGraphProbeReport):
        raise TypeError("report must be TafazoliCallGraphProbeReport")
    if report.schema_version != SCHEMA_VERSION or report.scope != PROBE_SCOPE:
        raise ValueError("unexpected call-graph report schema or scope")
    if report.method_status != "CALL_GRAPH_OBSERVATIONAL_PROXY_PROBE_COMPLETE":
        raise ValueError("unexpected method status")
    if report.codelength_name != "heldout Gaussian codelength/BIC proxy":
        raise ValueError("codelength must not be mislabeled strict or prequential MDL")
    if report.blind_fields_used:
        raise ValueError("labels or factor metadata entered the blind probe")
    if report.saved_test_role != "not_used":
        raise ValueError("saved classifier test tensors must remain unused")
    if not report.train_only_preprocessing:
        raise ValueError("all representation and gate fitting must be train-only")
    if report.primary_inference_unit != "recording_session_x_dimension":
        raise ValueError("anchors or neurons cannot be declared inference units")
    if report.config.primary_stride_bins < report.config.lag_bins:
        raise ValueError("primary count windows must not overlap")
    validate_call_graph_claim_locks(report.claim_locks)
    if any(asdict(report.inherited_stationary_claim_locks).values()):
        raise ValueError("stationary claim locks were unlocked")
    if report.source_file_md5 is None:
        if report.official_checksum_verified:
            raise ValueError("array report cannot claim official checksum")
    elif report.source_file_md5 != OFFICIAL_CLASSIFIER_MD5 or not report.official_checksum_verified:
        raise ValueError("official checksum must be exact and verified")

    expected_model_count = (
        len(report.session_specs)
        * 2
        * len(report.config.state_counts)
        * len(report.config.history_depths)
    )
    if len(report.model_results) != expected_model_count:
        raise ValueError("primary grid does not cover every session×dimension")
    if report.config.run_event_mean_removed_sensitivity:
        if len(report.event_mean_removed_results) != expected_model_count:
            raise ValueError("event-mean sensitivity grid is incomplete")
    elif report.event_mean_removed_results:
        raise ValueError("disabled event-mean sensitivity produced results")
    all_models = (
        *report.model_results,
        *report.event_mean_removed_results,
    )
    if not all(item.parameter_matched_dynamic_block for item in all_models):
        raise ValueError("switching and VAR dynamic parameter counts must match")
    if any(item.stride_bins < report.config.lag_bins for item in all_models):
        raise ValueError("a model result reused overlapping count windows")
    for item in all_models:
        if item.dimension not in (1, 3):
            raise ValueError("only dimensions 1 and 3 are allowed")
        expected_session = report.session_specs[item.session_index_one_based - 1]
        if (
            item.neuron_count != expected_session.neuron_count
            or item.animal != expected_session.animal
        ):
            raise ValueError("session-local result crossed a session boundary")
        if (
            item.hub_total_fold_count != len(item.fold_results)
            or item.hub_available_fold_count
            != sum(fold.hub.available for fold in item.fold_results)
            or item.hub_all_folds_available
            != (item.hub_available_fold_count == item.hub_total_fold_count)
        ):
            raise ValueError("hub fold availability was silently filtered")
        for fold in item.fold_results:
            if not fold.gate_uses_current_and_past_only or fold.test_target_passed_to_gate:
                raise ValueError("held-out future entered a gate")

    expected_transfer_count = len(report.session_specs) * 2 * len(report.config.state_counts)
    if len(report.frozen_transfer_results) != expected_transfer_count:
        raise ValueError("frozen D1↔D3 transfer grid is incomplete")
    if report.config.run_event_mean_removed_sensitivity:
        if len(report.event_mean_removed_frozen_transfer_results) != expected_transfer_count:
            raise ValueError("event-mean transfer sensitivity is incomplete")
    elif report.event_mean_removed_frozen_transfer_results:
        raise ValueError("disabled transfer sensitivity produced results")
    all_transfers = (
        *report.frozen_transfer_results,
        *report.event_mean_removed_frozen_transfer_results,
    )
    if not all(
        item.source_representation_and_gate_frozen and not item.target_rows_paired_to_source_rows
        for item in all_transfers
    ):
        raise ValueError("transfer must remain frozen and row-unpaired")

    verdict_map = {item.key: item.answer for item in report.verdicts}
    required = {
        "past_only_gate_contract_verified",
        "nonoverlap_session_local_proxy_completed",
        "state_dependent_switching_proxy_passed",
        "state_dependent_switching_operator_identified",
        "latent_common_successor_proxy_passed",
        "frontend_to_common_callee_observational_candidate_supported",
        "biological_common_callee_assembly_identified",
        "common_callee_architecture_exists_or_is_absent",
        "state_parent_rank1_proxy_passed",
        "task_inheritance_tree_identified",
        "task_inheritance_architecture_exists_or_is_absent",
        "frozen_cross_dimension_switching_transfer_passed",
        "continuous_dynamics_ruled_out",
        "causal_call_return_identified",
        "causal_call_return_architecture_exists_or_is_absent",
        "brain_programming_language_identified",
    }
    if set(verdict_map) != required:
        raise ValueError("verdict keys do not match the locked interface")
    fixed_answers = {
        "past_only_gate_contract_verified": YES,
        "nonoverlap_session_local_proxy_completed": YES,
        "biological_common_callee_assembly_identified": NO,
        "common_callee_architecture_exists_or_is_absent": TEST_UNAVAILABLE,
        "task_inheritance_tree_identified": NO,
        "task_inheritance_architecture_exists_or_is_absent": TEST_UNAVAILABLE,
        "continuous_dynamics_ruled_out": NO,
        "causal_call_return_identified": NO,
        "causal_call_return_architecture_exists_or_is_absent": TEST_UNAVAILABLE,
        "brain_programming_language_identified": NO,
    }
    if any(verdict_map[key] != answer for key, answer in fixed_answers.items()):
        raise ValueError("a locked scientific verdict was upgraded")
    if verdict_map["state_dependent_switching_proxy_passed"] == YES:
        if verdict_map["state_dependent_switching_operator_identified"] != PENDING:
            raise ValueError("a proxy pass cannot identify a biological operator")
    elif verdict_map["state_dependent_switching_operator_identified"] != NO:
        raise ValueError("failed switching proxy must not identify an operator")
    expected_frontend_candidate = (
        YES
        if report.config.run_event_mean_removed_sensitivity
        and any(
            _hub_proxy_passes(
                report.aggregates,
                state_count=state_count,
            )
            and _frozen_transfer_passes(
                all_transfers,
                state_count=state_count,
            )
            for state_count in report.config.state_counts
        )
        else (PENDING if not report.config.run_event_mean_removed_sensitivity else NO)
    )
    if (
        verdict_map["frontend_to_common_callee_observational_candidate_supported"]
        != expected_frontend_candidate
    ):
        raise ValueError("frontend-to-callee verdict mixed incompatible state counts")
    if verdict_map["frontend_to_common_callee_observational_candidate_supported"] == YES and (
        verdict_map["latent_common_successor_proxy_passed"] != YES
        or verdict_map["frozen_cross_dimension_switching_transfer_passed"] != YES
    ):
        raise ValueError("frontend-to-callee candidate lacks hub or transfer")


__all__ = [
    "AggregateResult",
    "CallGraphClaimLocks",
    "CallGraphProbeConfig",
    "CodelengthResult",
    "FoldComparison",
    "FrozenTransferResult",
    "HubFoldResult",
    "NO",
    "PENDING",
    "PROBE_SCOPE",
    "PastOnlyGate",
    "SCHEMA_VERSION",
    "SessionModelResult",
    "TEST_UNAVAILABLE",
    "TafazoliCallGraphProbeReport",
    "TrajectoryDesign",
    "WhitenedLatentFold",
    "YES",
    "aggregate_session_models",
    "apply_state_parent_rank1_predictor",
    "apply_switching_predictor",
    "apply_var_predictor",
    "assign_past_only_states",
    "build_trajectory_design",
    "centroid_parameter_count",
    "evaluate_fold_comparison",
    "evaluate_hub_fold",
    "fit_past_only_gate",
    "fit_state_parent_rank1_predictor",
    "fit_switching_predictor",
    "fit_var_predictor",
    "model_selection_bits",
    "run_tafazoli_call_graph_probe",
    "run_tafazoli_call_graph_probe_from_arrays",
    "score_heldout_gaussian_codelength",
    "state_parent_rank1_parameter_count",
    "summarize_session_model",
    "switching_parameter_count",
    "validate_call_graph_claim_locks",
    "validate_call_graph_probe_report",
    "var_parameter_count",
    "whiten_prepared_latent_fold",
]
