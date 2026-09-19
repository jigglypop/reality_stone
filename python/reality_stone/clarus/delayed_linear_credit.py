"""Finite reward-decoded delayed linear credit for the V18b fixture.

This module intentionally has no repository-local imports.  It implements the
five learners/controls registered by the V18b contract and exposes every
semantic state field in frozen dataclasses.  A public marker decides when
``write_cue`` is called; learning that marker is outside this fixture.

The result is a deterministic, dimensionless synthetic credit-assignment
construction.  It is not evidence for general delayed credit, biological
fidelity, recursive-agent scaling, cosmology, or AGI.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from numbers import Integral, Real
from typing import Literal, Sequence

import numpy as np


VectorTuple = tuple[float, ...]
FactorTuple = tuple[tuple[float, ...], ...]
RouteName = Literal[
    "eligibility",
    "hard_latch",
    "homogeneous_factor",
    "strict_metric_control",
    "no_trace_control",
]

_DEFAULT_DIMENSION = 8
_DEFAULT_ETA = 0.25
_STRICT_ENSEMBLE_SIZES = (1, 2, 4, 8, 16, 64)


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive built-in integer")
    return value


def _learning_rate(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("eta must be a finite real in (0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError("eta must be a finite real in (0, 1]")
    return result


def _finite_vector(
    values: Sequence[float] | np.ndarray,
    *,
    dimension: int,
    name: str,
    nonzero: bool,
) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite vector of length {dimension}") from error
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain only real numeric values")
    try:
        vector = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite vector of length {dimension}") from error
    if vector.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    if nonzero and not np.any(vector != 0.0):
        raise ValueError(f"{name} must be nonzero")
    return vector.copy()


def _binary_action(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("action must be exactly -1 or +1")
    action = float(value)
    if not math.isfinite(action) or action not in (-1.0, 1.0):
        raise ValueError("action must be exactly -1 or +1")
    return int(action)


def _binary_reward(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if not isinstance(value, Integral):
        raise ValueError("reward must be exactly 0 or 1")
    reward = int(value)
    if reward not in (0, 1):
        raise ValueError("reward must be exactly 0 or 1")
    return reward


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a built-in bool")
    return value


def _canonical_float(value: float) -> float:
    result = float(value)
    return 0.0 if result == 0.0 else result


def _vector_tuple(vector: np.ndarray) -> VectorTuple:
    return tuple(_canonical_float(value) for value in vector)


def _factor_tuple(factor: np.ndarray) -> FactorTuple:
    return tuple(tuple(_canonical_float(value) for value in row) for row in factor)


def _validate_observation_batch(
    observations: Sequence[Sequence[float] | np.ndarray] | np.ndarray,
    *,
    dimension: int,
) -> int:
    if isinstance(observations, (str, bytes)):
        raise ValueError("observations must be a non-string finite sequence")
    try:
        raw = np.asarray(observations)
    except (TypeError, ValueError) as error:
        raise ValueError("observations must be a finite rectangular sequence") from error
    if raw.size == 0 and raw.shape == (0,):
        return 0
    expected_suffix = (dimension,)
    if raw.ndim != 2 or raw.shape[1:] != expected_suffix:
        raise ValueError(f"observations must have shape (K, {dimension})")
    if raw.dtype.kind not in "iuf":
        raise ValueError("observations must contain only real numeric values")
    try:
        array = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("observations must be finite binary64 vectors") from error
    finite_rows = np.all(np.isfinite(array), axis=1)
    if not np.all(finite_rows):
        first_invalid = int(np.flatnonzero(~finite_rows)[0])
        raise ValueError(f"observations[{first_invalid}] must contain only finite values")
    return int(array.shape[0])


def _tie_positive(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("decision score must be finite")
    return +1 if value >= 0.0 else -1


def decode_binary_reward(action: object, reward: object) -> int:
    """Recover the hidden binary label from an action and correctness reward."""

    return _binary_action(action) * (2 * _binary_reward(reward) - 1)


@dataclass(frozen=True)
class EligibilityState:
    classifier: VectorTuple
    trace: VectorTuple
    active: bool


@dataclass(frozen=True)
class HardLatchState:
    classifier: VectorTuple
    latch: VectorTuple
    active: bool


@dataclass(frozen=True)
class HomogeneousCreditState:
    classifier: VectorTuple
    factor: FactorTuple
    active: bool


@dataclass(frozen=True)
class StrictMetricState:
    factor: FactorTuple


@dataclass(frozen=True)
class NoTraceState:
    classifier: VectorTuple
    active: bool


@dataclass(frozen=True)
class DelayedCreditCertificate:
    route: RouteName
    spatial_dimension: int
    eta: float
    state_fields: tuple[str, ...]
    classifier_real_coordinates: int
    episodic_real_coordinates: int
    episodic_serialized_entries: int
    active_tag_count: int
    factor_only_episodic_memory: bool
    hidden_cue_field_present: bool
    cue_write_uses_public_marker: bool
    unmarked_distractors_are_exact_noops: bool
    reward_reads_current_declared_state: bool
    atomic_clear_after_reward: bool
    state_coordinate_matched_to_eligibility: bool
    flop_or_wall_time_matched: bool
    deterministic_tie_is_positive: bool
    general_delayed_credit_verified: bool
    learned_event_selection_verified: bool
    noisy_reward_learning_verified: bool
    infinite_scc_intelligence_growth_verified: bool
    biological_fidelity_verified: bool
    cosmological_identity_verified: bool
    agi_evidence: bool


class _VectorCreditLearner:
    """Shared implementation for the explicit eligibility and hard latch routes."""

    state_type: type[EligibilityState] | type[HardLatchState]
    memory_field: Literal["trace", "latch"]
    route: Literal["eligibility", "hard_latch"]

    def __init__(self, dimension: int = _DEFAULT_DIMENSION, eta: float = _DEFAULT_ETA) -> None:
        self.dimension = _positive_int(dimension, "dimension")
        self.eta = _learning_rate(eta)

    def _make_state(
        self,
        classifier: np.ndarray,
        memory: np.ndarray,
        active: bool,
    ) -> EligibilityState | HardLatchState:
        payload = {
            "classifier": _vector_tuple(classifier),
            self.memory_field: _vector_tuple(memory),
            "active": bool(active),
        }
        return self.state_type(**payload)

    def _validated(
        self,
        state: EligibilityState | HardLatchState,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        if type(state) is not self.state_type:
            raise ValueError(f"state must be an exact {self.state_type.__name__}")
        classifier = _finite_vector(
            state.classifier,
            dimension=self.dimension,
            name="state.classifier",
            nonzero=False,
        )
        memory = _finite_vector(
            getattr(state, self.memory_field),
            dimension=self.dimension,
            name=f"state.{self.memory_field}",
            nonzero=False,
        )
        active = _bool(state.active, "state.active")
        if not active and np.any(memory != 0.0):
            raise ValueError(f"cleared state.{self.memory_field} must be zero")
        return classifier, memory, active

    def identity_state(self) -> EligibilityState | HardLatchState:
        zero = np.zeros(self.dimension, dtype=np.float64)
        return self._make_state(zero, zero, False)

    def write_cue(
        self,
        state: EligibilityState | HardLatchState,
        cue: Sequence[float] | np.ndarray,
    ) -> EligibilityState | HardLatchState:
        classifier, _, active = self._validated(state)
        if active:
            raise ValueError("cannot write a second cue while an episode is active")
        vector = _finite_vector(
            cue,
            dimension=self.dimension,
            name="cue",
            nonzero=True,
        )
        return self._make_state(classifier, vector, True)

    def distract(
        self,
        state: EligibilityState | HardLatchState,
        observation: Sequence[float] | np.ndarray,
    ) -> EligibilityState | HardLatchState:
        _, _, active = self._validated(state)
        if not active:
            raise ValueError("distractor requires an active episode")
        _finite_vector(
            observation,
            dimension=self.dimension,
            name="observation",
            nonzero=False,
        )
        return state

    def distract_many(
        self,
        state: EligibilityState | HardLatchState,
        observations: Sequence[Sequence[float] | np.ndarray] | np.ndarray,
    ) -> EligibilityState | HardLatchState:
        _, _, active = self._validated(state)
        if not active:
            raise ValueError("distractors require an active episode")
        _validate_observation_batch(observations, dimension=self.dimension)
        return state

    def action(self, state: EligibilityState | HardLatchState) -> int:
        classifier, memory, active = self._validated(state)
        if not active:
            raise ValueError("action requires an active episode")
        return _tie_positive(float(classifier @ memory))

    def reward(
        self,
        state: EligibilityState | HardLatchState,
        action: object,
        reward: object,
        *,
        invert_reward: bool = False,
    ) -> EligibilityState | HardLatchState:
        classifier, memory, active = self._validated(state)
        if not active:
            raise ValueError("reward requires an active episode")
        invert = _bool(invert_reward, "invert_reward")
        observed = _binary_reward(reward)
        decoded = decode_binary_reward(action, 1 - observed if invert else observed)
        updated = classifier + self.eta * decoded * memory
        if not np.all(np.isfinite(updated)):
            raise OverflowError("classifier update is not finite")
        return self._make_state(updated, np.zeros(self.dimension), False)

    def trace_lesion(
        self,
        state: EligibilityState | HardLatchState,
    ) -> EligibilityState | HardLatchState:
        classifier, _, active = self._validated(state)
        if not active:
            raise ValueError("trace lesion requires an active episode")
        return self._make_state(classifier, np.zeros(self.dimension), True)

    def snapshot(
        self,
        state: EligibilityState | HardLatchState,
    ) -> EligibilityState | HardLatchState:
        classifier, memory, active = self._validated(state)
        return self._make_state(classifier, memory, active)

    def from_snapshot(
        self,
        snapshot: EligibilityState | HardLatchState,
    ) -> EligibilityState | HardLatchState:
        return self.snapshot(snapshot)

    def certificate(
        self,
        state: EligibilityState | HardLatchState,
    ) -> DelayedCreditCertificate:
        self._validated(state)
        return _certificate(
            route=self.route,
            dimension=self.dimension,
            eta=self.eta,
            state_type=self.state_type,
            classifier_coordinates=self.dimension,
            episodic_coordinates=self.dimension,
            episodic_entries=self.dimension,
            active_tags=1,
            factor_only=False,
            reward_reads_state=True,
            atomic_clear=True,
            coordinate_matched=self.route == "hard_latch",
        )


class EligibilityLearner(_VectorCreditLearner):
    state_type = EligibilityState
    memory_field = "trace"
    route = "eligibility"


class HardLatchLearner(_VectorCreditLearner):
    state_type = HardLatchState
    memory_field = "latch"
    route = "hard_latch"


class HomogeneousLearner:
    """Classifier plus one factor-valued homogeneous episodic memory."""

    def __init__(self, dimension: int = _DEFAULT_DIMENSION, eta: float = _DEFAULT_ETA) -> None:
        self.dimension = _positive_int(dimension, "dimension")
        self.ambient_dimension = self.dimension + 1
        self.eta = _learning_rate(eta)

    def _identity_factor(self) -> np.ndarray:
        return np.eye(self.ambient_dimension, dtype=np.float64)

    def _make_state(
        self,
        classifier: np.ndarray,
        factor: np.ndarray,
        active: bool,
    ) -> HomogeneousCreditState:
        return HomogeneousCreditState(
            classifier=_vector_tuple(classifier),
            factor=_factor_tuple(factor),
            active=bool(active),
        )

    def _validated(
        self,
        state: HomogeneousCreditState,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        if type(state) is not HomogeneousCreditState:
            raise ValueError("state must be an exact HomogeneousCreditState")
        classifier = _finite_vector(
            state.classifier,
            dimension=self.dimension,
            name="state.classifier",
            nonzero=False,
        )
        try:
            raw_factor = np.asarray(state.factor)
        except (TypeError, ValueError) as error:
            raise ValueError("state.factor must be a finite canonical factor") from error
        if raw_factor.dtype.kind not in "iuf":
            raise ValueError("state.factor must contain only real numeric values")
        factor = raw_factor.astype(np.float64, copy=True)
        expected = (self.ambient_dimension, self.ambient_dimension)
        if factor.shape != expected:
            raise ValueError(f"state.factor must have shape {expected}")
        if not np.all(np.isfinite(factor)):
            raise ValueError("state.factor must contain only finite values")
        if not np.all(np.triu(factor, 1) == 0.0):
            raise ValueError("state.factor must be lower triangular")
        if not np.all(np.diag(factor) > 0.0):
            raise ValueError("state.factor must have a positive diagonal")
        active = _bool(state.active, "state.active")
        if not active and not np.array_equal(factor, self._identity_factor()):
            raise ValueError("cleared homogeneous factor must be canonical identity")
        return classifier, factor.copy(), active

    def identity_state(self) -> HomogeneousCreditState:
        return self._make_state(
            np.zeros(self.dimension, dtype=np.float64),
            self._identity_factor(),
            False,
        )

    def write_cue(
        self,
        state: HomogeneousCreditState,
        cue: Sequence[float] | np.ndarray,
    ) -> HomogeneousCreditState:
        classifier, _, active = self._validated(state)
        if active:
            raise ValueError("cannot write a second cue while an episode is active")
        vector = _finite_vector(
            cue,
            dimension=self.dimension,
            name="cue",
            nonzero=True,
        )
        lifted = np.concatenate((vector, np.ones(1, dtype=np.float64)))
        metric = self._identity_factor() + 0.5 * np.outer(lifted, lifted)
        try:
            factor = np.linalg.cholesky(metric)
        except np.linalg.LinAlgError as error:
            raise ArithmeticError("homogeneous cue metric lost positive definiteness") from error
        if not np.all(np.isfinite(factor)):
            raise OverflowError("homogeneous cue factor is not finite")
        return self._make_state(classifier, factor, True)

    def metric(self, state: HomogeneousCreditState) -> np.ndarray:
        _, factor, _ = self._validated(state)
        metric = factor @ factor.T
        if not np.all(np.isfinite(metric)):
            raise OverflowError("reconstructed homogeneous metric is not finite")
        return metric

    def eligibility(self, state: HomogeneousCreditState) -> np.ndarray:
        _, _, active = self._validated(state)
        if not active:
            raise ValueError("eligibility requires an active episode")
        return 2.0 * self.metric(state)[:-1, -1]

    def spatial_metric(self, state: HomogeneousCreditState) -> np.ndarray:
        return self.metric(state)[:-1, :-1]

    def distract(
        self,
        state: HomogeneousCreditState,
        observation: Sequence[float] | np.ndarray,
    ) -> HomogeneousCreditState:
        _, _, active = self._validated(state)
        if not active:
            raise ValueError("distractor requires an active episode")
        _finite_vector(
            observation,
            dimension=self.dimension,
            name="observation",
            nonzero=False,
        )
        return state

    def distract_many(
        self,
        state: HomogeneousCreditState,
        observations: Sequence[Sequence[float] | np.ndarray] | np.ndarray,
    ) -> HomogeneousCreditState:
        _, _, active = self._validated(state)
        if not active:
            raise ValueError("distractors require an active episode")
        _validate_observation_batch(observations, dimension=self.dimension)
        return state

    def action(self, state: HomogeneousCreditState) -> int:
        classifier, _, active = self._validated(state)
        if not active:
            raise ValueError("action requires an active episode")
        return _tie_positive(float(classifier @ self.eligibility(state)))

    def reward(
        self,
        state: HomogeneousCreditState,
        action: object,
        reward: object,
        *,
        invert_reward: bool = False,
    ) -> HomogeneousCreditState:
        classifier, _, active = self._validated(state)
        if not active:
            raise ValueError("reward requires an active episode")
        current_eligibility = self.eligibility(state)
        invert = _bool(invert_reward, "invert_reward")
        observed = _binary_reward(reward)
        decoded = decode_binary_reward(action, 1 - observed if invert else observed)
        updated = classifier + self.eta * decoded * current_eligibility
        if not np.all(np.isfinite(updated)):
            raise OverflowError("classifier update is not finite")
        return self._make_state(updated, self._identity_factor(), False)

    def trace_lesion(self, state: HomogeneousCreditState) -> HomogeneousCreditState:
        classifier, _, active = self._validated(state)
        if not active:
            raise ValueError("trace lesion requires an active episode")
        return self._make_state(classifier, self._identity_factor(), True)

    def snapshot(self, state: HomogeneousCreditState) -> HomogeneousCreditState:
        classifier, factor, active = self._validated(state)
        return self._make_state(classifier, factor, active)

    def from_snapshot(self, snapshot: HomogeneousCreditState) -> HomogeneousCreditState:
        return self.snapshot(snapshot)

    def serialize_factor(self, state: HomogeneousCreditState) -> bytes:
        _, factor, _ = self._validated(state)
        return _serialize_floats(factor.ravel())

    def certificate(self, state: HomogeneousCreditState) -> DelayedCreditCertificate:
        self._validated(state)
        ambient_dof = self.ambient_dimension * (self.ambient_dimension + 1) // 2
        return _certificate(
            route="homogeneous_factor",
            dimension=self.dimension,
            eta=self.eta,
            state_type=HomogeneousCreditState,
            classifier_coordinates=self.dimension,
            episodic_coordinates=ambient_dof,
            episodic_entries=self.ambient_dimension**2,
            active_tags=1,
            factor_only=True,
            reward_reads_state=True,
            atomic_clear=True,
            coordinate_matched=False,
        )


class StrictMetricControl:
    """Original-space SPD control with an exactly sign-even cue update."""

    ensemble_sizes = _STRICT_ENSEMBLE_SIZES

    def __init__(self, dimension: int = _DEFAULT_DIMENSION) -> None:
        self.dimension = _positive_int(dimension, "dimension")

    def _make_state(self, factor: np.ndarray) -> StrictMetricState:
        return StrictMetricState(_factor_tuple(factor))

    def _validated(self, state: StrictMetricState) -> np.ndarray:
        if type(state) is not StrictMetricState:
            raise ValueError("state must be an exact StrictMetricState")
        try:
            raw_factor = np.asarray(state.factor)
        except (TypeError, ValueError) as error:
            raise ValueError("state.factor must be a finite canonical factor") from error
        if raw_factor.dtype.kind not in "iuf":
            raise ValueError("state.factor must contain only real numeric values")
        factor = raw_factor.astype(np.float64, copy=True)
        expected = (self.dimension, self.dimension)
        if factor.shape != expected:
            raise ValueError(f"state.factor must have shape {expected}")
        if not np.all(np.isfinite(factor)):
            raise ValueError("state.factor must contain only finite values")
        if not np.all(np.triu(factor, 1) == 0.0):
            raise ValueError("state.factor must be lower triangular")
        if not np.all(np.diag(factor) > 0.0):
            raise ValueError("state.factor must have a positive diagonal")
        return factor.copy()

    def identity_state(self) -> StrictMetricState:
        return self._make_state(np.eye(self.dimension, dtype=np.float64))

    def make_state_from_metric(
        self,
        metric: Sequence[Sequence[float]] | np.ndarray,
    ) -> StrictMetricState:
        """Create a canonical state from a finite symmetric positive metric.

        This is the registered entry point for evaluator-owned, independently
        seeded ensemble members.  The seed and metric construction remain
        outside production; only the declared SPD state is persisted here.
        """

        try:
            raw = np.asarray(metric)
        except (TypeError, ValueError) as error:
            raise ValueError("metric must be a finite real square array") from error
        if raw.dtype.kind not in "iuf":
            raise ValueError("metric must contain only real numeric values")
        try:
            array = raw.astype(np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("metric must be a finite real square array") from error
        expected = (self.dimension, self.dimension)
        if array.shape != expected:
            raise ValueError(f"metric must have shape {expected}")
        if not np.all(np.isfinite(array)):
            raise ValueError("metric must contain only finite values")
        scale = max(1.0, float(np.max(np.abs(array))))
        tolerance = 64.0 * np.finfo(np.float64).eps * scale
        if not np.allclose(array, array.T, rtol=0.0, atol=tolerance):
            raise ValueError("metric must be symmetric")
        symmetric = 0.5 * (array + array.T)
        try:
            factor = np.linalg.cholesky(symmetric)
        except np.linalg.LinAlgError as error:
            raise ValueError("metric must be positive definite") from error
        if not np.all(np.isfinite(factor)) or not np.all(np.diag(factor) > 0.0):
            raise OverflowError("metric factor is not representable as finite binary64")
        return self._make_state(factor)

    def metric(self, state: StrictMetricState) -> np.ndarray:
        factor = self._validated(state)
        metric = factor @ factor.T
        if not np.all(np.isfinite(metric)):
            raise OverflowError("reconstructed strict metric is not finite")
        return metric

    def _projective_representative(
        self,
        cue: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        vector = _finite_vector(
            cue,
            dimension=self.dimension,
            name="cue",
            nonzero=True,
        )
        first_nonzero = int(np.flatnonzero(vector != 0.0)[0])
        if vector[first_nonzero] < 0.0:
            vector *= -1.0
        vector[vector == 0.0] = 0.0
        return vector

    def write_cue(
        self,
        state: StrictMetricState,
        cue: Sequence[float] | np.ndarray,
    ) -> StrictMetricState:
        return self.write_ensemble((state,), cue)[0]

    def write_ensemble(
        self,
        states: Sequence[StrictMetricState],
        cue: Sequence[float] | np.ndarray,
    ) -> tuple[StrictMetricState, ...]:
        """Apply one sign-even cue to a registered ensemble in one batch."""

        if isinstance(states, (str, bytes)):
            raise ValueError("states must be a registered non-string ensemble")
        try:
            size = len(states)
        except TypeError as error:
            raise ValueError("states must be a registered finite sequence") from error
        if size not in self.ensemble_sizes:
            raise ValueError(f"ensemble size must be one of {self.ensemble_sizes}")
        factors = np.stack([self._validated(state) for state in states], axis=0)
        vector = self._projective_representative(cue)
        # Keep the per-member multiplication order identical to the N=1 path;
        # batched BLAS is permitted to choose a different reduction order and
        # can otherwise change hexadecimal serialization by one ulp.
        metrics = np.stack([factor @ factor.T for factor in factors], axis=0)
        metric_vectors = np.stack([metric @ vector for metric in metrics], axis=0)
        predictions = np.asarray(
            [float(vector @ metric_vector) for metric_vector in metric_vectors],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(predictions)) or not np.all(predictions > 0.0):
            raise ArithmeticError("strict cue predictions must be positive and finite")
        updated = np.stack(
            [
                metric
                + 0.5 * np.outer(metric_vector, metric_vector) / prediction
                for metric, metric_vector, prediction in zip(
                    metrics, metric_vectors, predictions, strict=True
                )
            ],
            axis=0,
        )
        try:
            updated_factors = np.linalg.cholesky(updated)
        except np.linalg.LinAlgError as error:
            raise ArithmeticError("strict metric ensemble update lost positive definiteness") from error
        if not np.all(np.isfinite(updated_factors)):
            raise OverflowError("strict metric ensemble factors are not finite")
        if not np.all(np.diagonal(updated_factors, axis1=1, axis2=2) > 0.0):
            raise OverflowError("strict metric ensemble factors lack positive diagonals")
        return tuple(self._make_state(factor) for factor in updated_factors)

    def write_cue_many(
        self,
        states: Sequence[StrictMetricState],
        cue: Sequence[float] | np.ndarray,
    ) -> tuple[StrictMetricState, ...]:
        """Compatibility spelling for :meth:`write_ensemble`."""

        return self.write_ensemble(states, cue)

    def distract(
        self,
        state: StrictMetricState,
        observation: Sequence[float] | np.ndarray,
    ) -> StrictMetricState:
        self._validated(state)
        _finite_vector(
            observation,
            dimension=self.dimension,
            name="observation",
            nonzero=False,
        )
        return state

    def distract_many(
        self,
        state: StrictMetricState,
        observations: Sequence[Sequence[float] | np.ndarray] | np.ndarray,
    ) -> StrictMetricState:
        self._validated(state)
        _validate_observation_batch(observations, dimension=self.dimension)
        return state

    def distract_ensemble(
        self,
        states: Sequence[StrictMetricState],
        observations: Sequence[Sequence[float] | np.ndarray] | np.ndarray,
    ) -> tuple[StrictMetricState, ...]:
        """Validate every ensemble member and every event, then make no update."""

        if isinstance(states, (str, bytes)):
            raise ValueError("states must be a registered non-string ensemble")
        try:
            size = len(states)
        except TypeError as error:
            raise ValueError("states must be a registered finite sequence") from error
        if size not in self.ensemble_sizes:
            raise ValueError(f"ensemble size must be one of {self.ensemble_sizes}")
        for state in states:
            self._validated(state)
        _validate_observation_batch(observations, dimension=self.dimension)
        return states if type(states) is tuple else tuple(states)

    def action(self, state: StrictMetricState) -> int:
        self._validated(state)
        return +1

    def reward(
        self,
        state: StrictMetricState,
        action: object,
        reward: object,
    ) -> StrictMetricState:
        """Validate the terminal event and persist the metric unchanged."""

        self._validated(state)
        _binary_action(action)
        _binary_reward(reward)
        return state

    def snapshot(self, state: StrictMetricState) -> StrictMetricState:
        return self._make_state(self._validated(state))

    def from_snapshot(self, snapshot: StrictMetricState) -> StrictMetricState:
        return self.snapshot(snapshot)

    def serialize_state(self, state: StrictMetricState) -> bytes:
        return _serialize_floats(self._validated(state).ravel())

    def serialize_ensemble(self, states: Sequence[StrictMetricState]) -> bytes:
        if isinstance(states, (str, bytes)):
            raise ValueError("states must be a registered non-string ensemble")
        serialized = sorted(self.serialize_state(state) for state in states)
        if len(serialized) not in self.ensemble_sizes:
            raise ValueError(f"ensemble size must be one of {self.ensemble_sizes}")
        framed = [len(serialized).to_bytes(2, "big")]
        for payload in serialized:
            framed.append(len(payload).to_bytes(4, "big"))
            framed.append(payload)
        return b"".join(framed)

    def aggregate_action(self, states: Sequence[StrictMetricState]) -> int:
        self.serialize_ensemble(states)
        return +1

    def certificate(self, state: StrictMetricState) -> DelayedCreditCertificate:
        self._validated(state)
        coordinates = self.dimension * (self.dimension + 1) // 2
        return _certificate(
            route="strict_metric_control",
            dimension=self.dimension,
            eta=0.0,
            state_type=StrictMetricState,
            classifier_coordinates=0,
            episodic_coordinates=coordinates,
            episodic_entries=self.dimension**2,
            active_tags=0,
            factor_only=True,
            reward_reads_state=False,
            atomic_clear=False,
            coordinate_matched=False,
        )


class NoTraceControl:
    """Classifier-budget control whose reward-time eligibility is always zero."""

    def __init__(self, dimension: int = _DEFAULT_DIMENSION, eta: float = _DEFAULT_ETA) -> None:
        self.dimension = _positive_int(dimension, "dimension")
        self.eta = _learning_rate(eta)

    def _make_state(self, classifier: np.ndarray, active: bool) -> NoTraceState:
        return NoTraceState(_vector_tuple(classifier), bool(active))

    def _validated(self, state: NoTraceState) -> tuple[np.ndarray, bool]:
        if type(state) is not NoTraceState:
            raise ValueError("state must be an exact NoTraceState")
        classifier = _finite_vector(
            state.classifier,
            dimension=self.dimension,
            name="state.classifier",
            nonzero=False,
        )
        return classifier, _bool(state.active, "state.active")

    def identity_state(self) -> NoTraceState:
        return self._make_state(np.zeros(self.dimension, dtype=np.float64), False)

    def write_cue(
        self,
        state: NoTraceState,
        cue: Sequence[float] | np.ndarray,
    ) -> NoTraceState:
        classifier, active = self._validated(state)
        if active:
            raise ValueError("cannot write a second cue while an episode is active")
        _finite_vector(cue, dimension=self.dimension, name="cue", nonzero=True)
        return self._make_state(classifier, True)

    def distract(
        self,
        state: NoTraceState,
        observation: Sequence[float] | np.ndarray,
    ) -> NoTraceState:
        _, active = self._validated(state)
        if not active:
            raise ValueError("distractor requires an active episode")
        _finite_vector(
            observation,
            dimension=self.dimension,
            name="observation",
            nonzero=False,
        )
        return state

    def distract_many(
        self,
        state: NoTraceState,
        observations: Sequence[Sequence[float] | np.ndarray] | np.ndarray,
    ) -> NoTraceState:
        _, active = self._validated(state)
        if not active:
            raise ValueError("distractors require an active episode")
        _validate_observation_batch(observations, dimension=self.dimension)
        return state

    def action(self, state: NoTraceState) -> int:
        _, active = self._validated(state)
        if not active:
            raise ValueError("action requires an active episode")
        return +1

    def reward(
        self,
        state: NoTraceState,
        action: object,
        reward: object,
        *,
        invert_reward: bool = False,
    ) -> NoTraceState:
        classifier, active = self._validated(state)
        if not active:
            raise ValueError("reward requires an active episode")
        invert = _bool(invert_reward, "invert_reward")
        observed = _binary_reward(reward)
        decode_binary_reward(action, 1 - observed if invert else observed)
        return self._make_state(classifier, False)

    def snapshot(self, state: NoTraceState) -> NoTraceState:
        classifier, active = self._validated(state)
        return self._make_state(classifier, active)

    def from_snapshot(self, snapshot: NoTraceState) -> NoTraceState:
        return self.snapshot(snapshot)

    def certificate(self, state: NoTraceState) -> DelayedCreditCertificate:
        self._validated(state)
        return _certificate(
            route="no_trace_control",
            dimension=self.dimension,
            eta=self.eta,
            state_type=NoTraceState,
            classifier_coordinates=self.dimension,
            episodic_coordinates=0,
            episodic_entries=0,
            active_tags=1,
            factor_only=False,
            reward_reads_state=True,
            atomic_clear=True,
            coordinate_matched=False,
        )


def _serialize_floats(values: np.ndarray) -> bytes:
    spellings = (
        "0x0.0p+0" if float(value) == 0.0 else float(value).hex() for value in values
    )
    return ("|".join(spellings)).encode("ascii")


def _certificate(
    *,
    route: RouteName,
    dimension: int,
    eta: float,
    state_type: type[object],
    classifier_coordinates: int,
    episodic_coordinates: int,
    episodic_entries: int,
    active_tags: int,
    factor_only: bool,
    reward_reads_state: bool,
    atomic_clear: bool,
    coordinate_matched: bool,
) -> DelayedCreditCertificate:
    return DelayedCreditCertificate(
        route=route,
        spatial_dimension=dimension,
        eta=eta,
        state_fields=tuple(field.name for field in fields(state_type)),
        classifier_real_coordinates=classifier_coordinates,
        episodic_real_coordinates=episodic_coordinates,
        episodic_serialized_entries=episodic_entries,
        active_tag_count=active_tags,
        factor_only_episodic_memory=factor_only,
        hidden_cue_field_present=False,
        cue_write_uses_public_marker=True,
        unmarked_distractors_are_exact_noops=True,
        reward_reads_current_declared_state=reward_reads_state,
        atomic_clear_after_reward=atomic_clear,
        state_coordinate_matched_to_eligibility=coordinate_matched,
        flop_or_wall_time_matched=False,
        deterministic_tie_is_positive=True,
        general_delayed_credit_verified=False,
        learned_event_selection_verified=False,
        noisy_reward_learning_verified=False,
        infinite_scc_intelligence_growth_verified=False,
        biological_fidelity_verified=False,
        cosmological_identity_verified=False,
        agi_evidence=False,
    )


__all__ = [
    "VectorTuple",
    "FactorTuple",
    "EligibilityState",
    "HardLatchState",
    "HomogeneousCreditState",
    "StrictMetricState",
    "NoTraceState",
    "DelayedCreditCertificate",
    "decode_binary_reward",
    "EligibilityLearner",
    "HardLatchLearner",
    "HomogeneousLearner",
    "StrictMetricControl",
    "NoTraceControl",
]
