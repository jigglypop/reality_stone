"""Isolated NumPy benchmark for Phase A causal recurrent geometry.

This module implements a deliberately narrow synthetic identification task.  It
does not join the Clarus runtime and it does not provide evidence about SCCs,
memory, biology, consciousness, or AGI.  Matrix orientation is always
``transition[target, source]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any, Literal, Mapping, Sequence

import numpy as np


ObservationKind = Literal["known_identity", "known_mask", "unknown_mix"]
ModelKind = Literal["context_shared_input", "pooled_shared_input"]

SCHEMA = "ce.causal_recurrent_geometry.phase_a.v1"
RESULT_SCHEMA = "ce.causal_recurrent_geometry.phase_a.development-result.v1"
_RNG_ROLES = frozenset(
    {
        "graph",
        "train_trajectory",
        "heldout_trajectory",
        "intervention",
        "train_noise",
        "evaluation_noise",
        "shuffle",
        "bootstrap",
    }
)


def _finite_float(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _readonly_float_array(
    name: str,
    value: object,
    *,
    ndim: int,
    shape: tuple[int | None, ...] | None = None,
) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if shape is not None:
        if len(shape) != ndim:
            raise AssertionError("internal shape contract mismatch")
        for axis, expected in enumerate(shape):
            if expected is not None and array.shape[axis] != expected:
                raise ValueError(
                    f"{name} axis {axis} must have length {expected}; "
                    f"got {array.shape[axis]}"
                )
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be nonempty and finite")
    array.setflags(write=False)
    return array


def _readonly_context_array(name: str, value: object, *, rows: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.shape[0] != rows:
        raise ValueError(f"{name} must have shape ({rows},)")
    if raw.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must contain only integer labels")
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError(f"{name} contains a label outside int64")
    array = np.array(raw, dtype=np.int64, copy=True)
    if array.size == 0 or np.any(array < 0):
        raise ValueError(f"{name} must be nonempty and nonnegative")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PhaseAConfig:
    """Frozen generator and estimator choices for the V1 development task."""

    experiment_version: str
    master_seed: int
    state_dimension: int
    input_dimension: int
    context_count: int
    train_steps: int
    heldout_steps: int
    noise_sigma: float
    ridge: float
    train_intervention_scale: float
    heldout_intervention_scale: float
    context_heterogeneity: float

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_version, str) or not self.experiment_version:
            raise ValueError("experiment_version must be nonempty")
        if not self.experiment_version.isascii():
            raise ValueError("experiment_version must be ASCII")
        master_seed = _nonnegative_int("master_seed", self.master_seed)
        state_dimension = _positive_int("state_dimension", self.state_dimension)
        input_dimension = _positive_int("input_dimension", self.input_dimension)
        context_count = _positive_int("context_count", self.context_count)
        if self.context_count < 2:
            raise ValueError("context_count must be at least two")
        train_steps = _positive_int("train_steps", self.train_steps)
        heldout_steps = _positive_int("heldout_steps", self.heldout_steps)
        noise_sigma = _finite_float("noise_sigma", self.noise_sigma, positive=True)
        ridge = _finite_float("ridge", self.ridge)
        if ridge < 0.0:
            raise ValueError("ridge must be nonnegative")
        train_intervention_scale = _finite_float(
            "train_intervention_scale", self.train_intervention_scale, positive=True
        )
        heldout_intervention_scale = _finite_float(
            "heldout_intervention_scale",
            self.heldout_intervention_scale,
            positive=True,
        )
        context_heterogeneity = _finite_float(
            "context_heterogeneity", self.context_heterogeneity, positive=True
        )
        object.__setattr__(self, "master_seed", master_seed)
        object.__setattr__(self, "state_dimension", state_dimension)
        object.__setattr__(self, "input_dimension", input_dimension)
        object.__setattr__(self, "context_count", context_count)
        object.__setattr__(self, "train_steps", train_steps)
        object.__setattr__(self, "heldout_steps", heldout_steps)
        object.__setattr__(self, "noise_sigma", noise_sigma)
        object.__setattr__(self, "ridge", ridge)
        object.__setattr__(self, "train_intervention_scale", train_intervention_scale)
        object.__setattr__(
            self, "heldout_intervention_scale", heldout_intervention_scale
        )
        object.__setattr__(self, "context_heterogeneity", context_heterogeneity)


@dataclass(frozen=True)
class GroundTruth:
    """Generator-owned coefficients, never accepted by a learner function."""

    context_transitions: np.ndarray
    shared_input: np.ndarray
    graph_index: int

    def __post_init__(self) -> None:
        transitions = _readonly_float_array(
            "context_transitions", self.context_transitions, ndim=3
        )
        if transitions.shape[0] < 2 or transitions.shape[1] != transitions.shape[2]:
            raise ValueError("context_transitions must have shape (K,n,n), K >= 2")
        shared_input = _readonly_float_array(
            "shared_input",
            self.shared_input,
            ndim=2,
            shape=(transitions.shape[1], None),
        )
        _nonnegative_int("graph_index", self.graph_index)
        object.__setattr__(self, "context_transitions", transitions)
        object.__setattr__(self, "shared_input", shared_input)


@dataclass(frozen=True)
class TransitionBatch:
    """Learner-visible transition rows and their declared observation chart."""

    state: np.ndarray
    intervention: np.ndarray
    context: np.ndarray
    next_state: np.ndarray
    observation_kind: ObservationKind = "known_identity"

    def __post_init__(self) -> None:
        state = _readonly_float_array("state", self.state, ndim=2)
        rows, dimension = state.shape
        intervention = _readonly_float_array(
            "intervention", self.intervention, ndim=2, shape=(rows, None)
        )
        next_state = _readonly_float_array(
            "next_state", self.next_state, ndim=2, shape=(rows, dimension)
        )
        context = _readonly_context_array("context", self.context, rows=rows)
        if self.observation_kind not in {
            "known_identity",
            "known_mask",
            "unknown_mix",
        }:
            raise ValueError("unknown observation_kind")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "intervention", intervention)
        object.__setattr__(self, "next_state", next_state)
        object.__setattr__(self, "context", context)


@dataclass(frozen=True)
class DimensionlessCertificate:
    normalized_coordinates: bool
    finite_positive_reference_scales: bool
    state_dimension_tag: str
    input_dimension_tag: str
    noise_dimension_tag: str
    gaussian_residual_dimension_tag: str
    passed: bool


@dataclass(frozen=True)
class DesignCertificate:
    joint_singular_values: tuple[float, ...]
    joint_rank: int
    required_rank: int
    context_state_ranks: tuple[int, ...]
    residualized_input_singular_values: tuple[float, ...]
    residualized_input_rank: int
    residualized_input_rank_tolerance: float
    rank_tolerance: float
    finite_valid_inputs: bool
    full_rank: bool


@dataclass(frozen=True)
class FitResult:
    """Frozen fitted predictive coefficients and design accounting."""

    model_kind: ModelKind
    observation_kind: ObservationKind
    transitions: np.ndarray
    shared_input: np.ndarray
    context_count: int
    ridge: float
    design: DesignCertificate
    nominal_dof: int
    effective_dof: float

    def __post_init__(self) -> None:
        transitions = _readonly_float_array("transitions", self.transitions, ndim=3)
        shared_input = _readonly_float_array(
            "shared_input",
            self.shared_input,
            ndim=2,
            shape=(transitions.shape[1], None),
        )
        if transitions.shape[1] != transitions.shape[2]:
            raise ValueError("transitions must be square")
        if transitions.shape[0] != self.context_count:
            raise ValueError("transition count must match context_count")
        if self.model_kind not in {"context_shared_input", "pooled_shared_input"}:
            raise ValueError("unknown model_kind")
        if self.observation_kind not in {
            "known_identity",
            "known_mask",
            "unknown_mix",
        }:
            raise ValueError("unknown observation_kind")
        ridge = _finite_float("ridge", self.ridge)
        if ridge < 0.0:
            raise ValueError("ridge must be nonnegative")
        _positive_int("nominal_dof", self.nominal_dof)
        effective_dof = _finite_float("effective_dof", self.effective_dof)
        if effective_dof < 0.0 or effective_dof > self.nominal_dof + 1.0e-8:
            raise ValueError("effective_dof is outside [0, nominal_dof]")
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "shared_input", shared_input)


@dataclass(frozen=True)
class ClaimCertificate:
    known_identity: bool
    declared_linear_class: bool
    full_rank: bool
    finite_valid_inputs: bool
    exact_edge_allowed: bool
    exact_edge_identifiability_conditions_met: bool
    predictive_transition_available: bool
    anatomical_graph_available: bool
    latent_causal_support_identifiable_in_declared_class: bool
    scc_evidence: bool
    memory_evidence: bool
    biology_evidence: bool
    consciousness_evidence: bool
    agi_evidence: bool


@dataclass(frozen=True)
class SimilarityNoGoFixture:
    transition_a: np.ndarray
    transition_b: np.ndarray
    observation_a: np.ndarray
    observation_b: np.ndarray
    observed_trajectory_a: np.ndarray
    observed_trajectory_b: np.ndarray
    support_differs: bool
    observations_identical: bool


def canonical_json_bytes(payload: object) -> bytes:
    """Return the one registered finite canonical JSON representation."""

    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def role_digest(
    experiment_version: str,
    master_seed: int,
    role: str,
    graph_index: int,
    replicate_index: int = 0,
) -> str:
    """Derive a stable SHA-256 namespace token for an allowed development role."""

    if not isinstance(experiment_version, str) or not experiment_version.isascii():
        raise ValueError("experiment_version must be ASCII")
    if role not in _RNG_ROLES:
        raise ValueError(f"unregistered RNG role: {role!r}")
    master = _nonnegative_int("master_seed", master_seed)
    graph = _nonnegative_int("graph_index", graph_index)
    replicate = _nonnegative_int("replicate_index", replicate_index)
    payload = canonical_json_bytes(
        {
            "experiment_version": experiment_version,
            "graph_index": graph,
            "master_seed": master,
            "replicate_index": replicate,
            "role": role,
            "schema": "CE-PHASE-A-RNG-V1",
        }
    )
    return hashlib.sha256(payload).hexdigest()


def role_rng(
    config: PhaseAConfig,
    role: str,
    graph_index: int,
    replicate_index: int = 0,
) -> np.random.Generator:
    digest = role_digest(
        config.experiment_version,
        config.master_seed,
        role,
        graph_index,
        replicate_index,
    )
    return np.random.default_rng(int(digest, 16))


def dimensionless_certificate(manifest_section: Mapping[str, Any]) -> DimensionlessCertificate:
    """Validate the isolated normalized-coordinate contract in a manifest section."""

    if not isinstance(manifest_section, Mapping):
        raise TypeError("dimensionless section must be a mapping")
    normalized = manifest_section.get("normalized_coordinates") is True
    tags = manifest_section.get("dimension_tags")
    if not isinstance(tags, Mapping):
        raise ValueError("dimension_tags must be a mapping")
    expected_keys = {"state", "input", "noise", "gaussian_residual"}
    if set(tags) != expected_keys:
        raise ValueError("dimension_tags must have the exact registered keys")
    if any(tags[key] != "DIMENSIONLESS" for key in expected_keys):
        raise ValueError("all registered dimension tags must be DIMENSIONLESS")
    scales = manifest_section.get("reference_scales")
    if not isinstance(scales, Mapping) or set(scales) != {"state", "input", "noise"}:
        raise ValueError("reference_scales must have state, input, and noise")
    finite_positive = True
    for name, raw_values in scales.items():
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"reference scale {name!r} must be a nonempty list")
        for raw in raw_values:
            _finite_float(f"reference scale {name!r}", raw, positive=True)
    if not normalized:
        finite_positive = False
    passed = normalized and finite_positive
    return DimensionlessCertificate(
        normalized_coordinates=normalized,
        finite_positive_reference_scales=finite_positive,
        state_dimension_tag=str(tags["state"]),
        input_dimension_tag=str(tags["input"]),
        noise_dimension_tag=str(tags["noise"]),
        gaussian_residual_dimension_tag=str(tags["gaussian_residual"]),
        passed=passed,
    )


def _scale_stable(matrix: np.ndarray, radius: float) -> np.ndarray:
    eigenvalues = np.linalg.eigvals(matrix)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    if not math.isfinite(spectral_radius):
        raise FloatingPointError("nonfinite graph spectral radius")
    if spectral_radius == 0.0:
        return np.array(matrix, dtype=np.float64, copy=True)
    return np.array(matrix * (radius / spectral_radius), dtype=np.float64, copy=True)


def _generate_ground_truth(config: PhaseAConfig, graph_index: int) -> GroundTruth:
    """Generate stable context transitions and one nonzero shared input matrix."""

    graph = _nonnegative_int("graph_index", graph_index)
    rng = role_rng(config, "graph", graph)
    n = config.state_dimension
    k = config.context_count
    m = config.input_dimension
    base = _scale_stable(rng.normal(size=(n, n)), 0.34)
    transitions = np.empty((k, n, n), dtype=np.float64)
    centred = np.arange(k, dtype=np.float64) - (k - 1.0) / 2.0
    for context in range(k):
        direction = np.diag(
            np.roll(np.linspace(-1.0, 1.0, n, dtype=np.float64), context)
        )
        interaction = rng.normal(scale=0.16, size=(n, n))
        np.fill_diagonal(interaction, 0.0)
        candidate = (
            base
            + config.context_heterogeneity * centred[context] * direction
            + config.context_heterogeneity * interaction
        )
        transitions[context] = _scale_stable(candidate, 0.70)
    shared_input = rng.normal(scale=0.48, size=(n, m))
    if float(np.linalg.norm(shared_input)) <= 0.25:
        shared_input[0, 0] += 0.75
    return GroundTruth(transitions, shared_input, graph)


def _generate_transition_batch(
    config: PhaseAConfig,
    truth: GroundTruth,
    *,
    split: Literal["train", "heldout"],
) -> TransitionBatch:
    """Generate a balanced-context recurrent transition batch from separated RNGs."""

    if truth.context_transitions.shape != (
        config.context_count,
        config.state_dimension,
        config.state_dimension,
    ) or truth.shared_input.shape != (
        config.state_dimension,
        config.input_dimension,
    ):
        raise ValueError("truth shapes do not match config")
    if split == "train":
        rows = config.train_steps
        trajectory_role = "train_trajectory"
        intervention_scale = config.train_intervention_scale
        noise_role = "train_noise"
        replicate = 0
    elif split == "heldout":
        rows = config.heldout_steps
        trajectory_role = "heldout_trajectory"
        intervention_scale = config.heldout_intervention_scale
        noise_role = "evaluation_noise"
        replicate = 1
    else:
        raise ValueError("split must be 'train' or 'heldout'")
    trajectory_rng = role_rng(config, trajectory_role, truth.graph_index, replicate)
    intervention_rng = role_rng(
        config, "intervention", truth.graph_index, replicate
    )
    noise_rng = role_rng(config, noise_role, truth.graph_index, replicate)
    context = np.resize(np.arange(config.context_count, dtype=np.int64), rows)
    trajectory_rng.shuffle(context)
    intervention = intervention_rng.normal(
        scale=intervention_scale,
        size=(rows, config.input_dimension),
    )
    state = np.empty((rows, config.state_dimension), dtype=np.float64)
    next_state = np.empty_like(state)
    current = trajectory_rng.normal(scale=0.55, size=config.state_dimension)
    for index in range(rows):
        state[index] = current
        target = (
            truth.context_transitions[context[index]] @ current
            + truth.shared_input @ intervention[index]
            + noise_rng.normal(scale=config.noise_sigma, size=config.state_dimension)
        )
        if not np.all(np.isfinite(target)):
            raise FloatingPointError("generated transition is nonfinite")
        next_state[index] = target
        current = target
    return TransitionBatch(state, intervention, context, next_state)


@dataclass(frozen=True)
class DevelopmentGenerator:
    """Typed generator that refuses every seed outside one registered dev block."""

    config: PhaseAConfig
    registered_graph_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        registered = _seed_list(
            "registered_graph_seeds", self.registered_graph_seeds
        )
        object.__setattr__(self, "registered_graph_seeds", registered)

    def ground_truth(self, graph_seed: int) -> GroundTruth:
        seed = _nonnegative_int("graph_seed", graph_seed)
        if seed not in self.registered_graph_seeds:
            raise PermissionError("graph seed is outside the registered development block")
        return _generate_ground_truth(self.config, seed)

    def transition_batch(
        self,
        truth: GroundTruth,
        *,
        split: Literal["train", "heldout"],
    ) -> TransitionBatch:
        if truth.graph_index not in self.registered_graph_seeds:
            raise PermissionError("truth is outside the registered development block")
        return _generate_transition_batch(self.config, truth, split=split)


def observe_batch(
    batch: TransitionBatch,
    observation_kind: ObservationKind,
    observation_matrix: object | None = None,
) -> TransitionBatch:
    """Construct an observation-coordinate prediction fixture.

    The operation does not assert that the transformed coordinates identify the
    latent graph.  The exact-edge certificate handles that boundary separately.
    """

    dimension = batch.state.shape[1]
    if observation_kind == "known_identity":
        if observation_matrix is not None:
            matrix = _readonly_float_array(
                "observation_matrix",
                observation_matrix,
                ndim=2,
                shape=(dimension, dimension),
            )
            if not np.array_equal(matrix, np.eye(dimension)):
                raise ValueError("known_identity requires the identity matrix")
        return TransitionBatch(
            batch.state,
            batch.intervention,
            batch.context,
            batch.next_state,
            "known_identity",
        )
    if observation_matrix is None:
        raise ValueError("non-identity observation requires a matrix")
    matrix = _readonly_float_array("observation_matrix", observation_matrix, ndim=2)
    if matrix.shape[1] != dimension:
        raise ValueError("observation matrix input dimension mismatch")
    if observation_kind == "known_mask":
        if matrix.shape[0] >= dimension:
            raise ValueError("known_mask must strictly reduce the state dimension")
        if not np.all(np.logical_or(matrix == 0.0, matrix == 1.0)):
            raise ValueError("known_mask must contain only zero and one")
        if not np.all(np.sum(matrix, axis=1) == 1.0):
            raise ValueError("known_mask rows must each select one coordinate")
        if len(set(np.argmax(matrix, axis=1).tolist())) != matrix.shape[0]:
            raise ValueError("known_mask may not duplicate selected coordinates")
    elif observation_kind == "unknown_mix":
        if matrix.shape != (dimension, dimension):
            raise ValueError("unknown_mix must be square and invertible")
        if np.linalg.matrix_rank(matrix) != dimension:
            raise ValueError("unknown_mix must be invertible")
    else:
        raise ValueError("unknown observation_kind")
    return TransitionBatch(
        batch.state @ matrix.T,
        batch.intervention,
        batch.context,
        batch.next_state @ matrix.T,
        observation_kind,
    )


def _rank_from_singular_values(
    values: np.ndarray, shape: tuple[int, int]
) -> tuple[int, float]:
    maximum = float(values[0]) if values.size else 0.0
    tolerance = max(shape) * np.finfo(np.float64).eps * maximum
    rank = int(np.count_nonzero(values > tolerance))
    return rank, tolerance


def _context_design(batch: TransitionBatch, context_count: int) -> np.ndarray:
    rows, dimension = batch.state.shape
    design = np.zeros((rows, context_count * dimension + batch.intervention.shape[1]))
    for context in range(context_count):
        selected = batch.context == context
        start = context * dimension
        design[selected, start : start + dimension] = batch.state[selected]
    design[:, context_count * dimension :] = batch.intervention
    return design


def design_certificate(
    batch: TransitionBatch, context_count: int
) -> DesignCertificate:
    """Certify the joint ``Kn+m`` design and residualized input excitation."""

    k = _positive_int("context_count", context_count)
    if np.any(batch.context >= k):
        raise ValueError("batch context lies outside context_count")
    counts = np.bincount(batch.context, minlength=k)
    if np.any(counts == 0):
        raise ValueError("every declared context must be nonempty")
    design = _context_design(batch, k)
    singular_values = np.linalg.svd(design, compute_uv=False)
    rank, tolerance = _rank_from_singular_values(singular_values, design.shape)
    context_ranks = tuple(
        int(np.linalg.matrix_rank(batch.state[batch.context == context]))
        for context in range(k)
    )
    state_blocks = design[:, : k * batch.state.shape[1]]
    projected_input = state_blocks @ np.linalg.lstsq(
        state_blocks, batch.intervention, rcond=None
    )[0]
    residualized_input = batch.intervention - projected_input
    residual_values = np.linalg.svd(residualized_input, compute_uv=False)
    input_values = np.linalg.svd(batch.intervention, compute_uv=False)
    input_scale = float(input_values[0]) if input_values.size else 0.0
    residual_tolerance = (
        max(residualized_input.shape) * np.finfo(np.float64).eps * input_scale
    )
    residual_rank = int(np.count_nonzero(residual_values > residual_tolerance))
    required_rank = k * batch.state.shape[1] + batch.intervention.shape[1]
    finite = bool(
        np.all(np.isfinite(design))
        and np.all(np.isfinite(batch.next_state))
        and np.all(np.isfinite(residual_values))
    )
    full_rank = bool(
        finite
        and rank == required_rank
        and all(value == batch.state.shape[1] for value in context_ranks)
        and residual_rank == batch.intervention.shape[1]
    )
    return DesignCertificate(
        joint_singular_values=tuple(float(value) for value in singular_values),
        joint_rank=rank,
        required_rank=required_rank,
        context_state_ranks=context_ranks,
        residualized_input_singular_values=tuple(
            float(value) for value in residual_values
        ),
        residualized_input_rank=residual_rank,
        residualized_input_rank_tolerance=float(residual_tolerance),
        rank_tolerance=float(tolerance),
        finite_valid_inputs=finite,
        full_rank=full_rank,
    )


def _ridge_coefficients(
    design: np.ndarray, targets: np.ndarray, ridge: float
) -> tuple[np.ndarray, np.ndarray]:
    if ridge == 0.0:
        coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
    else:
        gram = design.T @ design
        coefficients = np.linalg.solve(
            gram + ridge * np.eye(gram.shape[0]), design.T @ targets
        )
    singular_values = np.linalg.svd(design, compute_uv=False)
    return coefficients, singular_values


def _effective_dof(
    singular_values: np.ndarray, ridge: float, output_dimension: int
) -> float:
    if ridge == 0.0:
        rank, _ = _rank_from_singular_values(
            singular_values, (singular_values.size, singular_values.size)
        )
        return float(output_dimension * rank)
    squared = singular_values * singular_values
    return output_dimension * float(np.sum(squared / (squared + ridge)))


def fit_context_shared_input(
    batch: TransitionBatch,
    *,
    context_count: int,
    ridge: float,
) -> FitResult:
    """Fit R1 using only learner-visible training rows and fixed hyperparameters."""

    k = _positive_int("context_count", context_count)
    penalty = _finite_float("ridge", ridge)
    if penalty < 0.0:
        raise ValueError("ridge must be nonnegative")
    certificate = design_certificate(batch, k)
    design = _context_design(batch, k)
    coefficients, singular_values = _ridge_coefficients(
        design, batch.next_state, penalty
    )
    n = batch.state.shape[1]
    m = batch.intervention.shape[1]
    transitions = np.stack(
        [coefficients[context * n : (context + 1) * n].T for context in range(k)]
    )
    shared_input = coefficients[k * n : k * n + m].T
    nominal = n * (k * n + m)
    effective = _effective_dof(singular_values, penalty, n)
    return FitResult(
        "context_shared_input",
        batch.observation_kind,
        transitions,
        shared_input,
        k,
        penalty,
        certificate,
        nominal,
        effective,
    )


def fit_pooled_shared_input(batch: TransitionBatch, *, ridge: float) -> FitResult:
    """Fit the mandatory R3 pooled baseline on the same learner-visible rows."""

    penalty = _finite_float("ridge", ridge)
    if penalty < 0.0:
        raise ValueError("ridge must be nonnegative")
    context_count = int(np.max(batch.context)) + 1
    joint_certificate = design_certificate(batch, context_count)
    design = np.concatenate([batch.state, batch.intervention], axis=1)
    coefficients, singular_values = _ridge_coefficients(
        design, batch.next_state, penalty
    )
    n = batch.state.shape[1]
    m = batch.intervention.shape[1]
    transition = coefficients[:n].T
    shared_input = coefficients[n : n + m].T
    pooled_values = np.linalg.svd(design, compute_uv=False)
    pooled_rank, pooled_tolerance = _rank_from_singular_values(
        pooled_values, design.shape
    )
    pooled_required = n + m
    pooled_certificate = DesignCertificate(
        joint_singular_values=tuple(float(value) for value in pooled_values),
        joint_rank=pooled_rank,
        required_rank=pooled_required,
        context_state_ranks=joint_certificate.context_state_ranks,
        residualized_input_singular_values=(
            joint_certificate.residualized_input_singular_values
        ),
        residualized_input_rank=joint_certificate.residualized_input_rank,
        residualized_input_rank_tolerance=(
            joint_certificate.residualized_input_rank_tolerance
        ),
        rank_tolerance=float(pooled_tolerance),
        finite_valid_inputs=joint_certificate.finite_valid_inputs,
        full_rank=bool(
            joint_certificate.finite_valid_inputs and pooled_rank == pooled_required
        ),
    )
    nominal = n * (n + m)
    effective = _effective_dof(singular_values, penalty, n)
    return FitResult(
        "pooled_shared_input",
        batch.observation_kind,
        np.repeat(transition[None, :, :], context_count, axis=0),
        shared_input,
        context_count,
        penalty,
        pooled_certificate,
        nominal,
        effective,
    )


def predict(fit: FitResult, batch: TransitionBatch) -> np.ndarray:
    if batch.observation_kind != fit.observation_kind:
        raise ValueError("batch observation kind does not match fitted chart")
    if batch.state.shape[1] != fit.transitions.shape[1]:
        raise ValueError("batch state dimension does not match fit")
    if batch.intervention.shape[1] != fit.shared_input.shape[1]:
        raise ValueError("batch input dimension does not match fit")
    if np.any(batch.context >= fit.context_count):
        raise ValueError("batch context lies outside fitted contexts")
    prediction = np.einsum(
        "tij,tj->ti", fit.transitions[batch.context], batch.state, optimize=True
    ) + batch.intervention @ fit.shared_input.T
    if not np.all(np.isfinite(prediction)):
        raise FloatingPointError("prediction is nonfinite")
    return prediction


def gaussian_nll(fit: FitResult, batch: TransitionBatch, *, sigma: float) -> float:
    """Score with a caller-owned common sigma that no learner receives."""

    scale = _finite_float("sigma", sigma, positive=True)
    residual = batch.next_state - predict(fit, batch)
    value = 0.5 * (
        residual.size * math.log(2.0 * math.pi * scale * scale)
        + float(np.sum(residual * residual)) / (scale * scale)
    )
    if not math.isfinite(value):
        raise FloatingPointError("Gaussian NLL is nonfinite")
    return value


def coefficient_errors(fit: FitResult, truth: GroundTruth) -> dict[str, float]:
    """Evaluator-only coefficient score; this function is never used during fit."""

    if (
        fit.model_kind != "context_shared_input"
        or fit.observation_kind != "known_identity"
        or not fit.design.full_rank
        or not fit.design.finite_valid_inputs
    ):
        raise PermissionError(
            "coefficient error is not claimable outside a finite full-rank "
            "known-identity context fit"
        )
    if fit.transitions.shape != truth.context_transitions.shape:
        raise ValueError("fit and truth transition shapes differ")
    if fit.shared_input.shape != truth.shared_input.shape:
        raise ValueError("fit and truth input shapes differ")
    return {
        "max_transition_error": float(
            np.max(np.abs(fit.transitions - truth.context_transitions))
        ),
        "max_shared_input_error": float(
            np.max(np.abs(fit.shared_input - truth.shared_input))
        ),
    }


def claim_certificate(
    fit: FitResult,
    *,
    declared_linear_class: bool,
) -> ClaimCertificate:
    known_identity = fit.observation_kind == "known_identity"
    declared = declared_linear_class is True
    full_rank = fit.design.full_rank
    finite = fit.design.finite_valid_inputs
    exact = bool(known_identity and declared and full_rank and finite)
    return ClaimCertificate(
        known_identity=known_identity,
        declared_linear_class=declared,
        full_rank=full_rank,
        finite_valid_inputs=finite,
        exact_edge_allowed=exact,
        exact_edge_identifiability_conditions_met=exact,
        predictive_transition_available=True,
        anatomical_graph_available=False,
        latent_causal_support_identifiable_in_declared_class=exact,
        scc_evidence=False,
        memory_evidence=False,
        biology_evidence=False,
        consciousness_evidence=False,
        agi_evidence=False,
    )


def shuffle_intervention_time(
    batch: TransitionBatch, config: PhaseAConfig, graph_index: int
) -> TransitionBatch:
    """Destroy input timing while preserving its marginal values and all other rows."""

    rng = role_rng(config, "shuffle", graph_index)
    permutation = rng.permutation(batch.state.shape[0])
    if np.array_equal(permutation, np.arange(batch.state.shape[0])):
        permutation = np.roll(permutation, 1)
    return TransitionBatch(
        batch.state,
        batch.intervention[permutation],
        batch.context,
        batch.next_state,
        batch.observation_kind,
    )


def similarity_no_go_fixture(steps: int = 8) -> SimilarityNoGoFixture:
    """Return two support-distinct latent systems with exactly equal observations."""

    count = _positive_int("steps", steps)
    transition_a = np.array([[0.25, 0.0], [0.0, 0.5]], dtype=np.float64)
    transform = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    inverse = np.array([[1.0, -1.0], [0.0, 1.0]], dtype=np.float64)
    transition_b = transform @ transition_a @ inverse
    observation_a = np.eye(2, dtype=np.float64)
    observation_b = inverse
    state_a = np.array([0.5, 0.25], dtype=np.float64)
    state_b = transform @ state_a
    observed_a = []
    observed_b = []
    for _ in range(count):
        observed_a.append(observation_a @ state_a)
        observed_b.append(observation_b @ state_b)
        state_a = transition_a @ state_a
        state_b = transition_b @ state_b
    trajectory_a = np.stack(observed_a)
    trajectory_b = np.stack(observed_b)
    support_a = transition_a != 0.0
    support_b = transition_b != 0.0
    return SimilarityNoGoFixture(
        _readonly_float_array("transition_a", transition_a, ndim=2),
        _readonly_float_array("transition_b", transition_b, ndim=2),
        _readonly_float_array("observation_a", observation_a, ndim=2),
        _readonly_float_array("observation_b", observation_b, ndim=2),
        _readonly_float_array("trajectory_a", trajectory_a, ndim=2),
        _readonly_float_array("trajectory_b", trajectory_b, ndim=2),
        bool(not np.array_equal(support_a, support_b)),
        bool(np.array_equal(trajectory_a, trajectory_b)),
    )


def paired_bootstrap_interval(
    values: Sequence[float],
    *,
    config: PhaseAConfig,
    bootstrap_seed: int,
    samples: int,
) -> tuple[float, float]:
    """Bootstrap graph-seed means; frames are never resampling units."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a finite vector of length >= 2")
    count = _positive_int("samples", samples)
    replicate = _nonnegative_int("bootstrap_seed", bootstrap_seed)
    rng = role_rng(config, "bootstrap", 0, replicate)
    indices = rng.integers(0, array.size, size=(count, array.size))
    means = np.mean(array[indices], axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def _seed_list(name: str, values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers")
    result = tuple(_nonnegative_int(name, value) for value in values)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be nonempty and unique")
    return result


def _dof_accounting(config: PhaseAConfig) -> dict[str, int]:
    n = config.state_dimension
    m = config.input_dimension
    k = config.context_count
    pooled = n * (n + m)
    factorized = n * (k * n + m)
    return {
        "factorized_nominal_dof": factorized,
        "pooled_nominal_dof": pooled,
        "factorized_minus_pooled": (k - 1) * n * n,
    }


def run_development_benchmark(
    config: PhaseAConfig,
    *,
    graph_seeds: Sequence[int],
    registered_development_graph_seeds: Sequence[int],
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Run exactly the registered development block and no other seed role."""

    requested = _seed_list("graph_seeds", graph_seeds)
    registered = _seed_list(
        "registered_development_graph_seeds", registered_development_graph_seeds
    )
    # This check precedes every role_digest/role_rng call in this function.
    if requested != registered:
        raise PermissionError("runner may evaluate only the exact development seed block")
    generator = DevelopmentGenerator(config, registered)
    per_seed: list[dict[str, Any]] = []
    for graph_seed in requested:
        truth = generator.ground_truth(graph_seed)
        training = generator.transition_batch(truth, split="train")
        heldout = generator.transition_batch(truth, split="heldout")
        factorized = fit_context_shared_input(
            training, context_count=config.context_count, ridge=config.ridge
        )
        pooled = fit_pooled_shared_input(training, ridge=config.ridge)
        shuffled_training = shuffle_intervention_time(training, config, graph_seed)
        shuffled = fit_context_shared_input(
            shuffled_training, context_count=config.context_count, ridge=config.ridge
        )
        factorized_nll = gaussian_nll(
            factorized, heldout, sigma=config.noise_sigma
        )
        pooled_nll = gaussian_nll(pooled, heldout, sigma=config.noise_sigma)
        shuffled_nll = gaussian_nll(shuffled, heldout, sigma=config.noise_sigma)
        exact = claim_certificate(
            factorized,
            declared_linear_class=True,
        )
        errors = coefficient_errors(factorized, truth)
        per_seed.append(
            {
                "claim_certificate": {
                    "exact_edge_allowed": exact.exact_edge_allowed,
                    "finite_valid_inputs": exact.finite_valid_inputs,
                    "full_rank": exact.full_rank,
                },
                "coefficient_errors": errors,
                "delta_nll_pooled_minus_factorized": pooled_nll - factorized_nll,
                "factorized_effective_dof": factorized.effective_dof,
                "factorized_nll": factorized_nll,
                "graph_seed": graph_seed,
                "joint_design": {
                    "context_state_ranks": list(
                        factorized.design.context_state_ranks
                    ),
                    "joint_rank": factorized.design.joint_rank,
                    "joint_singular_values": list(
                        factorized.design.joint_singular_values
                    ),
                    "rank_tolerance": factorized.design.rank_tolerance,
                    "required_rank": factorized.design.required_rank,
                    "residualized_input_rank": (
                        factorized.design.residualized_input_rank
                    ),
                    "residualized_input_rank_tolerance": (
                        factorized.design.residualized_input_rank_tolerance
                    ),
                    "residualized_input_singular_values": list(
                        factorized.design.residualized_input_singular_values
                    ),
                },
                "pooled_effective_dof": pooled.effective_dof,
                "pooled_nll": pooled_nll,
                "residual_scalar_count": int(heldout.next_state.size),
                "shuffle_penalty_nll": shuffled_nll - factorized_nll,
                "shuffled_input_nll": shuffled_nll,
            }
        )
    deltas = np.array(
        [item["delta_nll_pooled_minus_factorized"] for item in per_seed],
        dtype=np.float64,
    )
    shuffle_penalties = np.array(
        [item["shuffle_penalty_nll"] for item in per_seed], dtype=np.float64
    )
    delta_interval = paired_bootstrap_interval(
        deltas,
        config=config,
        bootstrap_seed=bootstrap_seed,
        samples=bootstrap_samples,
    )
    shuffle_interval = paired_bootstrap_interval(
        shuffle_penalties,
        config=config,
        bootstrap_seed=bootstrap_seed,
        samples=bootstrap_samples,
    )
    h1_pass = bool(
        np.mean(deltas) > 0.0
        and np.median(deltas) > 0.0
        and delta_interval[0] > 0.0
    )
    h2_pass = bool(
        np.mean(shuffle_penalties) > 0.0
        and np.median(shuffle_penalties) > 0.0
        and shuffle_interval[0] > 0.0
    )
    all_finite = bool(
        np.all(np.isfinite(deltas))
        and np.all(np.isfinite(shuffle_penalties))
    )
    all_exact = all(
        item["claim_certificate"]["exact_edge_allowed"] for item in per_seed
    )
    result = {
        "aggregate": {
            "delta_nll_mean": float(np.mean(deltas)),
            "delta_nll_median": float(np.median(deltas)),
            "delta_nll_paired_bootstrap_95": list(delta_interval),
            "graph_seed_count": len(per_seed),
            "shuffle_penalty_mean": float(np.mean(shuffle_penalties)),
            "shuffle_penalty_median": float(np.median(shuffle_penalties)),
            "shuffle_penalty_paired_bootstrap_95": list(shuffle_interval),
        },
        "claim_status": {
            "PA-H1": "GO" if h1_pass else "STOP",
            "PA-H2": "GO" if h2_pass else "STOP",
            "PA-I3": "PASS" if all_exact else "FAIL",
        },
        "dof_accounting": _dof_accounting(config),
        "exclusions": {
            "agi_evidence": False,
            "biology_evidence": False,
            "consciousness_evidence": False,
            "memory_evidence": False,
            "scc_evidence": False,
        },
        "finite": all_finite,
        "mode": "development",
        "per_graph_seed": per_seed,
        "primary_gate": h1_pass,
        "protocol": {
            "common_manifest_sigma_scorer_only": config.noise_sigma,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_stream_shared_across_registered_endpoints": True,
            "factorized_arm": "context-specific A_z plus shared B",
            "frame_is_statistical_unit": False,
            "graph_seed_is_statistical_unit": True,
            "nll_aggregation": "total_over_graph_seed_transitions_and_coordinates",
            "pooled_arm": "single A plus shared B",
            "ridge_equal_across_arms": config.ridge,
            "same_training_and_heldout_batches": True,
        },
        "schema": RESULT_SCHEMA,
        "shuffle_integrity_gate": h2_pass,
    }
    # Enforce canonical finite serialization before returning scored material.
    canonical_json_bytes(result)
    return result


__all__ = [
    "ClaimCertificate",
    "DesignCertificate",
    "DevelopmentGenerator",
    "DimensionlessCertificate",
    "FitResult",
    "GroundTruth",
    "ObservationKind",
    "PhaseAConfig",
    "RESULT_SCHEMA",
    "SCHEMA",
    "SimilarityNoGoFixture",
    "TransitionBatch",
    "canonical_json_bytes",
    "claim_certificate",
    "coefficient_errors",
    "design_certificate",
    "dimensionless_certificate",
    "fit_context_shared_input",
    "fit_pooled_shared_input",
    "gaussian_nll",
    "observe_batch",
    "paired_bootstrap_interval",
    "predict",
    "role_digest",
    "role_rng",
    "run_development_benchmark",
    "shuffle_intervention_time",
    "similarity_no_go_fixture",
]
