"""One-state affine-covariant rank-one metric flow for the V16.1 track.

The sole persistent semantic state is a canonical lower-triangular factor
``L`` with positive diagonal, encoding ``g = L L.T``.  The implementation does
not persist a second copy of ``g``, optimizer moments, replay, or role heads.

For an executed nonzero displacement ``x`` and observed positive squared cost
``c``, the update is the factor/congruence implementation of

    p = x.T g x,  r = log(p / c),
    g+ = g + ((exp(-eta*r) - 1) / p) (g x) (g x).T.

No spectral projection is used.  Binary64 results outside the representable
public domain are rejected explicitly rather than returned as zero, infinity,
NaN, or an invalid factor.  This finite vector-observation primitive is not a
raw sensory model, delayed-credit learner, continuum geometry, or AGI claim.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from numbers import Real
from typing import Literal, Sequence

import numpy as np


FactorTuple = tuple[tuple[float, ...], ...]
Route = Sequence[Sequence[float] | np.ndarray]

_FLOAT_INFO = np.finfo(np.float64)
_LOG_TWO = math.log(2.0)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


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
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite vector of length {dimension}") from error
    if result.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.any(result != 0.0):
        raise ValueError(f"{name} must be nonzero")
    return result.copy()


def _factor_tuple(factor: np.ndarray) -> FactorTuple:
    return tuple(tuple(float(value) for value in row) for row in factor)


def _normalised_binary_pair(mantissa: float, exponent: int) -> tuple[float, int]:
    """Return ``(m, e)`` with the same value and ``abs(m) in [0.5, 1)``."""

    if mantissa == 0.0:
        return 0.0, 0
    normalised, shift = math.frexp(mantissa)
    return normalised, exponent + shift


def _scaled_dot_pair(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    """Compute a signed dot product as a binary mantissa/exponent pair.

    Individual products are never materialised, so a representable cancelled
    result is not rejected merely because an intermediate product overflows.
    """

    terms: list[tuple[float, int]] = []
    for left_value, right_value in zip(left, right, strict=True):
        a = float(left_value)
        b = float(right_value)
        if a == 0.0 or b == 0.0:
            continue
        a_mantissa, a_exponent = math.frexp(a)
        b_mantissa, b_exponent = math.frexp(b)
        terms.append((a_mantissa * b_mantissa, a_exponent + b_exponent))
    if not terms:
        return 0.0, 0
    common_exponent = max(exponent for _, exponent in terms)
    scaled_sum = math.fsum(
        math.ldexp(mantissa, exponent - common_exponent)
        for mantissa, exponent in terms
    )
    return _normalised_binary_pair(scaled_sum, common_exponent)


def _pair_to_float(pair: tuple[float, int], name: str) -> float:
    mantissa, exponent = pair
    if mantissa == 0.0:
        return 0.0
    try:
        result = math.ldexp(mantissa, exponent)
    except OverflowError as error:
        raise OverflowError(f"{name} is not representable as finite binary64") from error
    if not math.isfinite(result):
        raise OverflowError(f"{name} is not representable as finite binary64")
    if result == 0.0:
        raise OverflowError(f"{name} is not representable as positive binary64")
    return result


def _log_norm_and_unit_from_pairs(
    pairs: Sequence[tuple[float, int]],
    *,
    name: str,
) -> tuple[float, np.ndarray]:
    nonzero_exponents = [exponent for mantissa, exponent in pairs if mantissa != 0.0]
    if not nonzero_exponents:
        raise ValueError(f"{name} must be nonzero after factor transport")
    common_exponent = max(nonzero_exponents)
    scaled = np.array(
        [
            math.ldexp(mantissa, exponent - common_exponent)
            if mantissa != 0.0
            else 0.0
            for mantissa, exponent in pairs
        ],
        dtype=np.float64,
    )
    norm = math.hypot(*(float(value) for value in scaled))
    if not math.isfinite(norm) or norm <= 0.0:  # pragma: no cover - construction guard
        raise FloatingPointError(f"failed to normalise {name}")
    return math.log(norm) + common_exponent * _LOG_TWO, scaled / norm


def _exp_representable(log_value: float, name: str) -> float:
    try:
        result = math.exp(log_value)
    except OverflowError as error:
        raise OverflowError(f"{name} is not representable as finite binary64") from error
    if not math.isfinite(result):
        raise OverflowError(f"{name} is not representable as finite binary64")
    if result <= 0.0:
        raise OverflowError(f"{name} is not representable as positive binary64")
    return result


def _scale_by_log(value: float, log_scale: float, name: str) -> float:
    if value == 0.0:
        return 0.0
    magnitude = _exp_representable(math.log(abs(value)) + log_scale, name)
    return math.copysign(magnitude, value)


def _sqrt_pair(value: float) -> tuple[float, int]:
    """Return the positive square root as a binary mantissa/exponent pair."""

    mantissa, exponent = math.frexp(value)
    quotient, remainder = divmod(exponent, 2)
    return _normalised_binary_pair(math.sqrt(math.ldexp(mantissa, remainder)), quotient)


def _scale_by_sqrt_ratio(
    value: float,
    numerator: float,
    denominator: float,
    name: str,
) -> float:
    """Compute ``value * sqrt(numerator / denominator)`` without the ratio."""

    if value == 0.0:
        return 0.0
    value_mantissa, value_exponent = math.frexp(abs(value))
    numerator_mantissa, numerator_exponent = _sqrt_pair(numerator)
    denominator_mantissa, denominator_exponent = _sqrt_pair(denominator)
    pair = _normalised_binary_pair(
        value_mantissa * numerator_mantissa / denominator_mantissa,
        value_exponent + numerator_exponent - denominator_exponent,
    )
    return math.copysign(_pair_to_float(pair, name), value)


@dataclass(frozen=True)
class CovariantMetricConfig:
    """Dimensionless V16.1 learning rate and declared route tie tolerance."""

    eta: float = 0.4
    tie_tolerance_multiplier: float = 64.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "eta", _finite_float(self.eta, "eta"))
        object.__setattr__(
            self,
            "tie_tolerance_multiplier",
            _finite_float(self.tie_tolerance_multiplier, "tie_tolerance_multiplier"),
        )
        if not 0.0 < self.eta <= 1.0:
            raise ValueError("eta must lie in (0, 1]")
        if self.tie_tolerance_multiplier < 0.0:
            raise ValueError("tie_tolerance_multiplier must be nonnegative")


@dataclass(frozen=True)
class CovariantMetricState:
    """The only persistent state: a canonical factor encoding ``g = L L.T``."""

    factor: FactorTuple


@dataclass(frozen=True)
class RouteChoice:
    costs: tuple[float, ...]
    minimizers: tuple[int, ...]
    selected_index: int
    selected_cost: float
    unique: bool
    tie_tolerance: float
    tie_policy: str


@dataclass(frozen=True)
class MetricFlowCertificate:
    dimension: int
    eta: float
    tie_tolerance_multiplier: float
    persistent_state: Literal["factor_only_metric_encoding"]
    persistent_state_field_count: int
    semantic_state_degrees_of_freedom: int
    optimizer_state_field_count: int
    canonical_lower_triangular_positive_diagonal: bool
    factor_congruence_update: bool
    exact_rank_one_update_complexity: Literal["O(d^2)"]
    canonical_qr_retriangularization_complexity: Literal["O(d^3)"]
    spectral_projection_used: bool
    affine_update_covariant_in_exact_arithmetic: bool
    spd_preserved_in_exact_arithmetic: bool
    same_observation_contraction_in_exact_arithmetic: bool
    airm_natural_gradient_identity: bool
    full_metric_identifiable_without_spanning_measurements: bool
    fixed_rate_noisy_point_convergence: bool
    raw_perception_verified: bool
    delayed_credit_verified: bool
    continuum_geometry_verified: bool
    agi_evidence: bool


class CovariantMetricFlow:
    """Canonical factor implementation of the one-state V16.1 metric flow.

    The mathematical rank-one congruence has ``O(d^2)`` structure.  This
    reference binary64 implementation deliberately uses a full ``O(d^3)`` QR
    after it, so that every persisted state is again the unique lower-triangular
    factor with positive diagonal.  It does not claim an ``O(d^2)`` numerical
    Cholesky update/downdate implementation.
    """

    def __init__(
        self,
        dimension: int,
        config: CovariantMetricConfig = CovariantMetricConfig(),
    ) -> None:
        self.dimension = _positive_int(dimension, "dimension")
        if type(config) is not CovariantMetricConfig:
            raise ValueError("config must be an exact CovariantMetricConfig")
        self.config = config

    def _validated_factor(self, state: CovariantMetricState) -> np.ndarray:
        if type(state) is not CovariantMetricState:
            raise ValueError("state must be an exact CovariantMetricState")
        try:
            factor = np.asarray(state.factor, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("state.factor must be a finite square array") from error
        expected = (self.dimension, self.dimension)
        if factor.shape != expected:
            raise ValueError(f"state.factor must have shape {expected}")
        if not np.all(np.isfinite(factor)):
            raise ValueError("state.factor must contain only finite values")
        if np.any(np.triu(factor, 1) != 0.0):
            raise ValueError("state.factor must be lower triangular")
        if np.any(np.diag(factor) <= 0.0):
            raise ValueError("state.factor must have a strictly positive diagonal")
        return factor.copy()

    def identity_state(self) -> CovariantMetricState:
        return CovariantMetricState(_factor_tuple(np.eye(self.dimension, dtype=np.float64)))

    def make_state_from_metric(
        self,
        metric: Sequence[Sequence[float]] | np.ndarray,
    ) -> CovariantMetricState:
        try:
            array = np.asarray(metric, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("metric must be a finite square array") from error
        expected = (self.dimension, self.dimension)
        if array.shape != expected:
            raise ValueError(f"metric must have shape {expected}")
        if not np.all(np.isfinite(array)):
            raise ValueError("metric must contain only finite values")
        scale = max(1.0, float(np.max(np.abs(array))))
        tolerance = 64.0 * _FLOAT_INFO.eps * scale
        if not np.allclose(array, array.T, rtol=0.0, atol=tolerance):
            raise ValueError("metric must be symmetric")
        symmetric = 0.5 * array + 0.5 * array.T
        try:
            factor = np.linalg.cholesky(symmetric)
        except np.linalg.LinAlgError as error:
            raise ValueError("metric must be positive definite") from error
        if not np.all(np.isfinite(factor)) or np.any(np.diag(factor) <= 0.0):
            raise OverflowError("metric factor is not representable as finite binary64")
        return CovariantMetricState(_factor_tuple(factor))

    def _transport_pairs(
        self,
        factor: np.ndarray,
        displacement: np.ndarray,
    ) -> tuple[tuple[float, int], ...]:
        return tuple(
            _scaled_dot_pair(factor[:, column], displacement)
            for column in range(self.dimension)
        )

    def _log_prediction_and_direction(
        self,
        state: CovariantMetricState,
        displacement: Sequence[float] | np.ndarray,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        factor = self._validated_factor(state)
        vector = _finite_vector(
            displacement,
            dimension=self.dimension,
            name="displacement",
        )
        pairs = self._transport_pairs(factor, vector)
        log_norm, direction = _log_norm_and_unit_from_pairs(
            pairs,
            name="L.T @ displacement",
        )
        common_exponent = max(
            exponent for mantissa, exponent in pairs if mantissa != 0.0
        )
        scaled = [
            math.ldexp(mantissa, exponent - common_exponent)
            if mantissa != 0.0
            else 0.0
            for mantissa, exponent in pairs
        ]
        squared_mantissa = math.fsum(value * value for value in scaled)
        prediction_pair = _normalised_binary_pair(
            squared_mantissa,
            2 * common_exponent,
        )
        prediction = _pair_to_float(prediction_pair, "prediction")
        return 2.0 * log_norm, prediction, direction, factor

    def metric(self, state: CovariantMetricState) -> np.ndarray:
        factor = self._validated_factor(state)
        result = np.empty((self.dimension, self.dimension), dtype=np.float64)
        for row in range(self.dimension):
            for column in range(row + 1):
                value = _pair_to_float(
                    _scaled_dot_pair(factor[row], factor[column]),
                    "metric entry",
                )
                result[row, column] = value
                result[column, row] = value
        try:
            np.linalg.cholesky(result)
        except np.linalg.LinAlgError as error:
            raise OverflowError(
                "metric is not representable as a positive-definite binary64 array"
            ) from error
        return result

    def predict(
        self,
        state: CovariantMetricState,
        displacement: Sequence[float] | np.ndarray,
    ) -> float:
        _, prediction, _, _ = self._log_prediction_and_direction(state, displacement)
        return prediction

    def residual(
        self,
        state: CovariantMetricState,
        displacement: Sequence[float] | np.ndarray,
        observed_cost: float,
    ) -> float:
        cost = _finite_float(observed_cost, "observed_cost")
        if cost <= 0.0:
            raise ValueError("observed_cost must be positive")
        log_prediction, prediction, _, _ = self._log_prediction_and_direction(
            state,
            displacement,
        )
        delta = prediction - cost
        if -0.5 * cost <= delta <= cost:
            relative_delta = delta / cost
            return math.log1p(relative_delta)
        return log_prediction - math.log(cost)

    def _orthogonal_basis(self, direction: np.ndarray) -> np.ndarray:
        """Return a deterministic orthogonal matrix whose first column is direction."""

        first = np.zeros(self.dimension, dtype=np.float64)
        first[0] = 1.0
        # Select the noncancelling Householder form.  ``e1 - direction`` is
        # ill-conditioned near +e1, while ``e1 + direction`` is ill-conditioned
        # near -e1; the sign branch keeps the normalising vector well scaled.
        if direction[0] >= 0.0:
            reflector = first + direction
            reflector /= math.hypot(*(float(value) for value in reflector))
            basis = 2.0 * np.outer(reflector, reflector) - np.eye(self.dimension)
        else:
            reflector = first - direction
            reflector /= math.hypot(*(float(value) for value in reflector))
            basis = np.eye(self.dimension) - 2.0 * np.outer(reflector, reflector)
        tolerance = 128.0 * _FLOAT_INFO.eps * max(1, self.dimension)
        if not np.allclose(
            basis.T @ basis,
            np.eye(self.dimension),
            rtol=0.0,
            atol=tolerance,
        ) or not np.allclose(basis[:, 0], direction, rtol=0.0, atol=tolerance):
            raise FloatingPointError("failed to construct an orthogonal congruence basis")
        return basis

    def update(
        self,
        state: CovariantMetricState,
        displacement: Sequence[float] | np.ndarray,
        observed_cost: float,
    ) -> CovariantMetricState:
        cost = _finite_float(observed_cost, "observed_cost")
        if cost <= 0.0:
            raise ValueError("observed_cost must be positive")
        log_prediction, prediction, direction, factor = self._log_prediction_and_direction(
            state,
            displacement,
        )
        delta = prediction - cost
        if -0.5 * cost <= delta <= cost:
            relative_delta = delta / cost
            residual = math.log1p(relative_delta)
        else:
            residual = log_prediction - math.log(cost)
        log_factor_ratio = -0.5 * self.config.eta * residual

        basis = self._orthogonal_basis(direction)
        transported = factor @ basis
        for row in range(self.dimension):
            if self.config.eta == 1.0:
                transported[row, 0] = _scale_by_sqrt_ratio(
                    float(transported[row, 0]),
                    cost,
                    prediction,
                    "updated congruence factor entry",
                )
            else:
                transported[row, 0] = _scale_by_log(
                    float(transported[row, 0]),
                    log_factor_ratio,
                    "updated congruence factor entry",
                )
        if not np.all(np.isfinite(transported)):
            raise OverflowError("updated congruence factor is not finite binary64")
        try:
            _, upper = np.linalg.qr(transported.T)
        except np.linalg.LinAlgError as error:
            raise FloatingPointError("updated congruence factor QR failed") from error
        diagonal = np.diag(upper)
        if not np.all(np.isfinite(upper)) or np.any(diagonal == 0.0):
            raise OverflowError(
                "updated canonical factor is not representable with positive binary64 diagonal"
            )
        signs = np.where(diagonal < 0.0, -1.0, 1.0)
        canonical = np.tril((signs[:, None] * upper).T)
        if not np.all(np.isfinite(canonical)) or np.any(np.diag(canonical) <= 0.0):
            raise OverflowError("updated canonical factor is not finite positive-diagonal")
        return CovariantMetricState(_factor_tuple(canonical))

    def route_costs(
        self,
        state: CovariantMetricState,
        routes: Sequence[Route],
    ) -> tuple[float, ...]:
        if isinstance(routes, (str, bytes)) or len(routes) == 0:
            raise ValueError("routes must be a nonempty sequence")
        costs: list[float] = []
        for route in routes:
            if isinstance(route, (str, bytes)) or len(route) == 0:
                raise ValueError("every route must contain at least one displacement")
            cost = math.fsum(self.predict(state, displacement) for displacement in route)
            if not math.isfinite(cost) or cost <= 0.0:
                raise OverflowError("route cost is not representable as positive binary64")
            costs.append(cost)
        return tuple(costs)

    def choose_route(
        self,
        state: CovariantMetricState,
        routes: Sequence[Route],
    ) -> RouteChoice:
        costs = self.route_costs(state, routes)
        minimum = min(costs)
        tolerance = (
            self.config.tie_tolerance_multiplier
            * _FLOAT_INFO.eps
            * max(1.0, max(abs(cost) for cost in costs))
        )
        minimizers = tuple(
            index for index, cost in enumerate(costs) if abs(cost - minimum) <= tolerance
        )
        selected = minimizers[0]
        return RouteChoice(
            costs=costs,
            minimizers=minimizers,
            selected_index=selected,
            selected_cost=costs[selected],
            unique=len(minimizers) == 1,
            tie_tolerance=tolerance,
            tie_policy="declared absolute-relative tolerance; lowest index representative",
        )

    def snapshot(self, state: CovariantMetricState) -> CovariantMetricState:
        return CovariantMetricState(_factor_tuple(self._validated_factor(state)))

    def from_snapshot(self, snapshot: CovariantMetricState) -> CovariantMetricState:
        return self.snapshot(snapshot)

    def certificate(self, state: CovariantMetricState) -> MetricFlowCertificate:
        self._validated_factor(state)
        return MetricFlowCertificate(
            dimension=self.dimension,
            eta=self.config.eta,
            tie_tolerance_multiplier=self.config.tie_tolerance_multiplier,
            persistent_state="factor_only_metric_encoding",
            persistent_state_field_count=len(fields(CovariantMetricState)),
            semantic_state_degrees_of_freedom=self.dimension * (self.dimension + 1) // 2,
            optimizer_state_field_count=0,
            canonical_lower_triangular_positive_diagonal=True,
            factor_congruence_update=True,
            exact_rank_one_update_complexity="O(d^2)",
            canonical_qr_retriangularization_complexity="O(d^3)",
            spectral_projection_used=False,
            affine_update_covariant_in_exact_arithmetic=True,
            spd_preserved_in_exact_arithmetic=True,
            same_observation_contraction_in_exact_arithmetic=True,
            airm_natural_gradient_identity=True,
            full_metric_identifiable_without_spanning_measurements=False,
            fixed_rate_noisy_point_convergence=False,
            raw_perception_verified=False,
            delayed_credit_verified=False,
            continuum_geometry_verified=False,
            agi_evidence=False,
        )


__all__ = [
    "CovariantMetricConfig",
    "CovariantMetricState",
    "RouteChoice",
    "MetricFlowCertificate",
    "CovariantMetricFlow",
]
