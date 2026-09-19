"""A narrow one-factor homogeneous memory for the V17 signed-cue fixture.

The persistent candidate state is one canonical factor encoding an augmented
metric ``G in SPD(d + 1)``.  For a public spatial reference ``u`` and cue sign
``s``, the fixed analytic write applies the V16 metric flow to
``z_s = (s*u, 1)`` with ``eta=1`` and observed cost ``c=4``.  Terminal actions
use ``y_a = (a*u, -1)`` and select the lower quadratic cost.

The extra homogeneous coordinate is semantic structure: in block form it
packs a spatial covector and one scalar, adding ``d + 1`` real coordinates.
Its declared chart law is only ``diag(J, 1)`` for spatial ``J in GL(d)``.
This exact one-cue memory fixture is not general delayed credit or AGI.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from numbers import Real
from typing import Literal, Sequence

import numpy as np

from .covariant_metric_flow import (
    CovariantMetricConfig,
    CovariantMetricFlow,
    CovariantMetricState,
)


FactorTuple = tuple[tuple[float, ...], ...]

_CUE_COST = 4.0
_ETA = 1.0
_PREWRITE_PREDICTION = 2.0
_PREWRITE_TOLERANCE_MULTIPLIER = 256.0


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive built-in integer")
    return value


def _finite_vector(
    values: Sequence[float] | np.ndarray,
    *,
    dimension: int,
    name: str,
) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite vector of length {dimension}") from error
    if vector.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.any(vector != 0.0):
        raise ValueError(f"{name} must be nonzero")
    return vector.copy()


def _sign(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be exactly -1 or +1")
    result = float(value)
    if not math.isfinite(result) or result not in (-1.0, 1.0):
        raise ValueError(f"{name} must be exactly -1 or +1")
    return int(result)


def _factor_tuple(factor: np.ndarray) -> FactorTuple:
    return tuple(
        tuple(0.0 if float(value) == 0.0 else float(value) for value in row) for row in factor
    )


@dataclass(frozen=True)
class HomogeneousSignedCueState:
    """The sole persistent field: a factor for the augmented SPD metric."""

    factor: FactorTuple


@dataclass(frozen=True)
class SignedCueReadout:
    """Quadratic terminal decision in fixed action order ``(-1, +1)``."""

    action_signs: tuple[int, int]
    costs: tuple[float, float]
    selected_sign: int
    selected_cost: float
    unique: bool
    wrong_minus_selected_margin: float


@dataclass(frozen=True)
class HomogeneousSignedCueCertificate:
    spatial_dimension: int
    ambient_dimension: int
    eta: float
    observed_cue_cost: float
    registered_prewrite_prediction: float
    prewrite_prediction_tolerance_multiplier: float
    persistent_state: Literal["factor_only_augmented_metric_encoding"]
    persistent_state_field_count: int
    ambient_real_state_coordinates: int
    original_metric_degrees_of_freedom: int
    added_ambient_coordinates: int
    packed_covector_coordinates: int
    packed_scalar_coordinates: int
    optimizer_state_field_count: int
    declared_chart_group: Literal["diag(GL(d),1)_only"]
    embedded_spatial_chart_covariance_in_exact_arithmetic: bool
    general_ambient_gl_semantics_verified: bool
    homogeneous_splitting_is_extra_structure: bool
    single_cue_public_reference_memory_only: bool
    general_delayed_credit_verified: bool
    infinite_scc_intelligence_growth_verified: bool
    biological_fidelity_verified: bool
    cosmological_identity_verified: bool
    agi_evidence: bool


class HomogeneousSignedCue:
    """Fixed V17 homogeneous lift backed by the V16 factor flow."""

    def __init__(self, dimension: int) -> None:
        self.dimension = _positive_int(dimension, "dimension")
        self.ambient_dimension = self.dimension + 1
        config = CovariantMetricConfig(eta=_ETA)
        self._ambient_flow = CovariantMetricFlow(self.ambient_dimension, config)
        self._strict_flow = CovariantMetricFlow(self.dimension, config)

    @staticmethod
    def _to_augmented_state(
        state: HomogeneousSignedCueState,
    ) -> CovariantMetricState:
        if type(state) is not HomogeneousSignedCueState:
            raise ValueError("state must be an exact HomogeneousSignedCueState")
        return CovariantMetricState(state.factor)

    @staticmethod
    def _from_augmented_state(
        state: CovariantMetricState,
    ) -> HomogeneousSignedCueState:
        factor = np.asarray(state.factor, dtype=np.float64)
        return HomogeneousSignedCueState(_factor_tuple(factor))

    def identity_state(self) -> HomogeneousSignedCueState:
        return self._from_augmented_state(self._ambient_flow.identity_state())

    def make_state_from_metric(
        self,
        metric: Sequence[Sequence[float]] | np.ndarray,
    ) -> HomogeneousSignedCueState:
        """Create a state without resetting a transported chart metric."""

        return self._from_augmented_state(self._ambient_flow.make_state_from_metric(metric))

    def metric(self, state: HomogeneousSignedCueState) -> np.ndarray:
        return self._ambient_flow.metric(self._to_augmented_state(state))

    def lift_cue(
        self,
        public_reference: Sequence[float] | np.ndarray,
        sign: int,
    ) -> tuple[float, ...]:
        reference = _finite_vector(
            public_reference,
            dimension=self.dimension,
            name="public_reference",
        )
        cue_sign = _sign(sign, "sign")
        lifted = np.empty(self.ambient_dimension, dtype=np.float64)
        lifted[:-1] = cue_sign * reference
        lifted[-1] = 1.0
        return tuple(float(value) for value in lifted)

    def lift_action(
        self,
        public_reference: Sequence[float] | np.ndarray,
        action: int,
    ) -> tuple[float, ...]:
        reference = _finite_vector(
            public_reference,
            dimension=self.dimension,
            name="public_reference",
        )
        action_sign = _sign(action, "action")
        lifted = np.empty(self.ambient_dimension, dtype=np.float64)
        lifted[:-1] = action_sign * reference
        lifted[-1] = -1.0
        return tuple(float(value) for value in lifted)

    def write_cue(
        self,
        state: HomogeneousSignedCueState,
        public_reference: Sequence[float] | np.ndarray,
        sign: int,
    ) -> HomogeneousSignedCueState:
        internal = self._to_augmented_state(state)
        lifted_cue = self.lift_cue(public_reference, sign)
        prediction = self._ambient_flow.predict(internal, lifted_cue)
        tolerance = (
            _PREWRITE_TOLERANCE_MULTIPLIER
            * np.finfo(np.float64).eps
            * max(1.0, abs(prediction), _PREWRITE_PREDICTION)
        )
        if abs(prediction - _PREWRITE_PREDICTION) > tolerance:
            raise ValueError(
                "registered cue write requires pre-update prediction p=2; "
                "supply a unit original fixture or its metric-transported chart"
            )
        updated = self._ambient_flow.update(
            internal,
            lifted_cue,
            _CUE_COST,
        )
        return self._from_augmented_state(updated)

    def terminal_costs(
        self,
        state: HomogeneousSignedCueState,
        public_reference: Sequence[float] | np.ndarray,
    ) -> tuple[float, float]:
        internal = self._to_augmented_state(state)
        return (
            self._ambient_flow.predict(
                internal,
                self.lift_action(public_reference, -1),
            ),
            self._ambient_flow.predict(
                internal,
                self.lift_action(public_reference, +1),
            ),
        )

    def readout(
        self,
        state: HomogeneousSignedCueState,
        public_reference: Sequence[float] | np.ndarray,
    ) -> SignedCueReadout:
        costs = self.terminal_costs(state, public_reference)
        selected_index = 0 if costs[0] <= costs[1] else 1
        other_index = 1 - selected_index
        margin = costs[other_index] - costs[selected_index]
        return SignedCueReadout(
            action_signs=(-1, +1),
            costs=costs,
            selected_sign=(-1, +1)[selected_index],
            selected_cost=costs[selected_index],
            unique=margin > 0.0,
            wrong_minus_selected_margin=margin,
        )

    def snapshot(
        self,
        state: HomogeneousSignedCueState,
    ) -> HomogeneousSignedCueState:
        internal = self._ambient_flow.snapshot(self._to_augmented_state(state))
        return self._from_augmented_state(internal)

    def from_snapshot(
        self,
        snapshot: HomogeneousSignedCueState,
    ) -> HomogeneousSignedCueState:
        return self.snapshot(snapshot)

    def strict_identity_state(self) -> CovariantMetricState:
        """Identity state for the registered original-space no-go control."""

        return self._strict_flow.identity_state()

    def _projective_representative(
        self,
        signed_cue: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Choose one binary64 representative of ``{x, -x}``.

        The exact V16 update is sign-even.  This representative additionally
        prevents QR roundoff and signed-zero spelling from making paired
        serialized controls differ even though their exact metric is equal.
        """

        cue = _finite_vector(
            signed_cue,
            dimension=self.dimension,
            name="signed_cue",
        )
        first_nonzero = int(np.flatnonzero(cue != 0.0)[0])
        if cue[first_nonzero] < 0.0:
            cue *= -1.0
        cue[cue == 0.0] = 0.0
        return cue

    def strict_write(
        self,
        state: CovariantMetricState,
        signed_cue: Sequence[float] | np.ndarray,
    ) -> CovariantMetricState:
        if type(state) is not CovariantMetricState:
            raise ValueError("state must be an exact CovariantMetricState")
        updated = self._strict_flow.update(
            state,
            self._projective_representative(signed_cue),
            _CUE_COST,
        )
        factor = np.asarray(updated.factor, dtype=np.float64)
        return CovariantMetricState(_factor_tuple(factor))

    def serialize_strict_state(self, state: CovariantMetricState) -> bytes:
        """Serialize every factor float by exact hexadecimal spelling."""

        validated = self._strict_flow.snapshot(state)
        spellings = (
            "0x0.0p+0" if value == 0.0 else float(value).hex()
            for row in validated.factor
            for value in row
        )
        return ("|".join(spellings)).encode("ascii")

    def strict_terminal_distribution(
        self,
        state: CovariantMetricState,
        public_reference: Sequence[float] | np.ndarray,
    ) -> tuple[float, float]:
        """Registered sign-blind control law in action order ``(-1, +1)``."""

        self._strict_flow.snapshot(state)
        _finite_vector(
            public_reference,
            dimension=self.dimension,
            name="public_reference",
        )
        return (0.5, 0.5)

    def certificate(
        self,
        state: HomogeneousSignedCueState,
    ) -> HomogeneousSignedCueCertificate:
        self._ambient_flow.snapshot(self._to_augmented_state(state))
        original_dof = self.dimension * (self.dimension + 1) // 2
        ambient_dof = self.ambient_dimension * (self.ambient_dimension + 1) // 2
        return HomogeneousSignedCueCertificate(
            spatial_dimension=self.dimension,
            ambient_dimension=self.ambient_dimension,
            eta=_ETA,
            observed_cue_cost=_CUE_COST,
            registered_prewrite_prediction=_PREWRITE_PREDICTION,
            prewrite_prediction_tolerance_multiplier=(_PREWRITE_TOLERANCE_MULTIPLIER),
            persistent_state="factor_only_augmented_metric_encoding",
            persistent_state_field_count=len(fields(HomogeneousSignedCueState)),
            ambient_real_state_coordinates=ambient_dof,
            original_metric_degrees_of_freedom=original_dof,
            added_ambient_coordinates=ambient_dof - original_dof,
            packed_covector_coordinates=self.dimension,
            packed_scalar_coordinates=1,
            optimizer_state_field_count=0,
            declared_chart_group="diag(GL(d),1)_only",
            embedded_spatial_chart_covariance_in_exact_arithmetic=True,
            general_ambient_gl_semantics_verified=False,
            homogeneous_splitting_is_extra_structure=True,
            single_cue_public_reference_memory_only=True,
            general_delayed_credit_verified=False,
            infinite_scc_intelligence_growth_verified=False,
            biological_fidelity_verified=False,
            cosmological_identity_verified=False,
            agi_evidence=False,
        )


__all__ = [
    "HomogeneousSignedCueState",
    "SignedCueReadout",
    "HomogeneousSignedCueCertificate",
    "HomogeneousSignedCue",
]
