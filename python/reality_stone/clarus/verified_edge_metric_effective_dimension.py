"""Exact finite bridge from positive edge metrics to observed ridge dimension."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations


Matrix = tuple[tuple[Fraction, ...], ...]


def _q(value: object, name: str) -> Fraction:
    if isinstance(value, bool) or isinstance(value, float) or not isinstance(value, (int, Fraction)):
        raise ValueError(f"{name} must be an exact integer or Fraction")
    return Fraction(value)


def _matrix(value: object, name: str, *, columns: int | None = None) -> Matrix:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{name} must be a nonempty matrix")
    rows = []
    width = columns
    for row in value:
        if not isinstance(row, (tuple, list)) or not row:
            raise ValueError(f"{name} rows must be nonempty")
        parsed = tuple(_q(item, name) for item in row)
        width = len(parsed) if width is None else width
        if len(parsed) != width:
            raise ValueError(f"{name} must be rectangular with the declared width")
        rows.append(parsed)
    return tuple(rows)


def _transpose(a: Matrix) -> Matrix:
    return tuple(tuple(a[i][j] for i in range(len(a))) for j in range(len(a[0])))


def _add(a: Matrix, b: Matrix) -> Matrix:
    return tuple(tuple(x + y for x, y in zip(ar, br, strict=True)) for ar, br in zip(a, b, strict=True))


def _sub(a: Matrix, b: Matrix) -> Matrix:
    return tuple(tuple(x - y for x, y in zip(ar, br, strict=True)) for ar, br in zip(a, b, strict=True))


def _scale(value: Fraction, a: Matrix) -> Matrix:
    return tuple(tuple(value * item for item in row) for row in a)


def _mul(a: Matrix, b: Matrix) -> Matrix:
    bt = _transpose(b)
    return tuple(tuple(sum(x * y for x, y in zip(row, column, strict=True)) for column in bt) for row in a)


def _identity(size: int) -> Matrix:
    return tuple(tuple(Fraction(i == j) for j in range(size)) for i in range(size))


def _symmetric(a: Matrix) -> bool:
    return len(a) == len(a[0]) and a == _transpose(a)


def _determinant(a: Matrix) -> Fraction:
    if len(a) != len(a[0]):
        raise ValueError("determinant requires a square matrix")
    work = [list(row) for row in a]
    determinant = Fraction(1)
    for column in range(len(work)):
        pivot = next((row for row in range(column, len(work)) if work[row][column]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            determinant = -determinant
        pivot_value = work[column][column]
        determinant *= pivot_value
        for row in range(column + 1, len(work)):
            factor = work[row][column] / pivot_value
            for index in range(column + 1, len(work)):
                work[row][index] -= factor * work[column][index]
    return determinant


def _inverse(a: Matrix) -> Matrix:
    if len(a) != len(a[0]):
        raise ValueError("inverse requires a square matrix")
    size = len(a)
    work = [list(row + identity_row) for row, identity_row in zip(a, _identity(size), strict=True)]
    for column in range(size):
        pivot = next((row for row in range(column, size) if work[row][column]), None)
        if pivot is None:
            raise ValueError("matrix is singular")
        work[column], work[pivot] = work[pivot], work[column]
        pivot_value = work[column][column]
        work[column] = [item / pivot_value for item in work[column]]
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            work[row] = [x - factor * y for x, y in zip(work[row], work[column], strict=True)]
    return tuple(tuple(row[size:]) for row in work)


def _principal_submatrix(a: Matrix, indices: tuple[int, ...]) -> Matrix:
    return tuple(tuple(a[i][j] for j in indices) for i in indices)


def _positive_semidefinite(a: Matrix) -> bool:
    if not _symmetric(a):
        return False
    indices = range(len(a))
    return all(
        _determinant(_principal_submatrix(a, choice)) >= 0
        for size in range(1, len(a) + 1)
        for choice in combinations(indices, size)
    )


def _positive_definite(a: Matrix) -> bool:
    return _symmetric(a) and all(
        _determinant(tuple(tuple(a[i][j] for j in range(size)) for i in range(size))) > 0
        for size in range(1, len(a) + 1)
    )


def _rank(a: Matrix) -> int:
    work = [list(row) for row in a]
    row = 0
    for column in range(len(a[0])):
        pivot = next((index for index in range(row, len(work)) if work[index][column]), None)
        if pivot is None:
            continue
        work[row], work[pivot] = work[pivot], work[row]
        pivot_value = work[row][column]
        work[row] = [item / pivot_value for item in work[row]]
        for index in range(len(work)):
            if index != row and work[index][column]:
                factor = work[index][column]
                work[index] = [x - factor * y for x, y in zip(work[index], work[row], strict=True)]
        row += 1
        if row == len(work):
            break
    return row


def _trace(a: Matrix) -> Fraction:
    return sum((a[i][i] for i in range(len(a))), Fraction(0))


def _metric(base: Matrix, terms: tuple[Matrix, ...], weights: tuple[Fraction, ...]) -> Matrix:
    result = base
    for weight, term in zip(weights, terms, strict=True):
        result = _add(result, _scale(weight, term))
    return result


def _gram(measurement: Matrix, metric: Matrix) -> Matrix:
    return _mul(_mul(measurement, _inverse(metric)), _transpose(measurement))


def _ridge_dimension(gram: Matrix, ridge: Fraction) -> Fraction:
    regularized = _add(gram, _scale(ridge, _identity(len(gram))))
    return _trace(_mul(gram, _inverse(regularized)))


def _participation_ratio(gram: Matrix) -> Fraction | None:
    denominator = _trace(_mul(gram, gram))
    return None if denominator == 0 else _trace(gram) ** 2 / denominator


@dataclass(frozen=True)
class VerifiedEdgeMetricEffectiveDimension:
    status: str
    validation_level: str
    metric_at_lower_weights: Matrix
    metric_at_upper_weights: Matrix
    observed_gram_at_lower_weights: Matrix
    observed_gram_at_upper_weights: Matrix
    ridge_dimension_at_lower_weights: Fraction
    ridge_dimension_at_upper_weights: Fraction
    ridge_dimension_drop: Fraction
    ridge_dimension_drop_upper_bound: Fraction
    edge_derivatives_at_lower_weights: tuple[Fraction, ...]
    measurement_rank: int
    hard_rank_at_lower_weights: int
    hard_rank_at_upper_weights: int
    participation_ratio_at_lower_weights: Fraction | None
    participation_ratio_at_upper_weights: Fraction | None
    edge_weights_coordinatewise_strengthened: bool
    metric_loewner_increase_verified: bool
    inverse_metric_loewner_decrease_verified: bool
    observed_gram_loewner_decrease_verified: bool
    ridge_dimension_monotonicity_verified: bool
    hard_rank_invariance_verified: bool
    robust_candidate_band_verified: bool
    candidate_band: tuple[int, int]
    participation_ratio_monotonicity_claim_admitted: bool
    selected_signal_rank_claim_admitted: bool
    consciousness_dimension_claim_admitted: bool


def verified_edge_metric_effective_dimension(
    *,
    baseline_metric: object,
    edge_metric_terms: object,
    lower_edge_weights: object,
    upper_edge_weights: object,
    measurement_operator: object,
    ridge_parameter: object,
    candidate_band: object,
) -> VerifiedEdgeMetricEffectiveDimension:
    base = _matrix(baseline_metric, "baseline_metric")
    if not _positive_definite(base):
        raise ValueError("baseline_metric must be exact symmetric positive definite")
    dimension = len(base)
    if not isinstance(edge_metric_terms, (tuple, list)):
        raise ValueError("edge_metric_terms must be a sequence")
    terms = tuple(_matrix(term, "edge_metric_term", columns=dimension) for term in edge_metric_terms)
    if any(len(term) != dimension or not _positive_semidefinite(term) for term in terms):
        raise ValueError("every edge_metric_term must be exact symmetric positive semidefinite")
    if not isinstance(lower_edge_weights, (tuple, list)) or not isinstance(upper_edge_weights, (tuple, list)):
        raise ValueError("edge weights must be sequences")
    lower = tuple(_q(value, "lower_edge_weight") for value in lower_edge_weights)
    upper = tuple(_q(value, "upper_edge_weight") for value in upper_edge_weights)
    if len(lower) != len(terms) or len(upper) != len(terms):
        raise ValueError("edge weights must match edge term count")
    if any(value < 0 or value > 1 for value in lower + upper):
        raise ValueError("edge weights must lie in the closed unit interval")
    coordinatewise = all(left <= right for left, right in zip(lower, upper, strict=True))
    if not coordinatewise:
        raise ValueError("upper_edge_weights must strengthen every edge coordinatewise")
    measurement = _matrix(measurement_operator, "measurement_operator", columns=dimension)
    ridge = _q(ridge_parameter, "ridge_parameter")
    if ridge <= 0:
        raise ValueError("ridge_parameter must be strictly positive")
    if (
        not isinstance(candidate_band, (tuple, list))
        or len(candidate_band) != 2
        or any(type(value) is not int for value in candidate_band)
        or candidate_band[0] < 0
        or candidate_band[0] > candidate_band[1]
    ):
        raise ValueError("candidate_band must be two ordered nonnegative built-in integers")
    band = tuple(candidate_band)

    metric_lower = _metric(base, terms, lower)
    metric_upper = _metric(base, terms, upper)
    inverse_lower = _inverse(metric_lower)
    inverse_upper = _inverse(metric_upper)
    gram_lower = _gram(measurement, metric_lower)
    gram_upper = _gram(measurement, metric_upper)
    dimension_lower = _ridge_dimension(gram_lower, ridge)
    dimension_upper = _ridge_dimension(gram_upper, ridge)
    drop = dimension_lower - dimension_upper
    gram_drop = _sub(gram_lower, gram_upper)
    drop_upper = _trace(gram_drop) / ridge

    inverse_metric = inverse_lower
    regularized_inverse = _inverse(_add(gram_lower, _scale(ridge, _identity(len(gram_lower)))))
    derivatives = []
    for term in terms:
        core = _mul(
            _mul(
                _mul(
                    _mul(
                        _mul(regularized_inverse, measurement), inverse_metric
                    ),
                    term,
                ),
                inverse_metric,
            ),
            _mul(_transpose(measurement), regularized_inverse),
        )
        derivatives.append(-ridge * _trace(core))

    measurement_rank = _rank(measurement)
    hard_lower = _rank(gram_lower)
    hard_upper = _rank(gram_upper)
    metric_increase = _positive_semidefinite(_sub(metric_upper, metric_lower))
    inverse_decrease = _positive_semidefinite(_sub(inverse_lower, inverse_upper))
    gram_decrease = _positive_semidefinite(gram_drop)
    monotone = dimension_lower >= dimension_upper and drop >= 0 and drop <= drop_upper
    hard_invariant = hard_lower == hard_upper == measurement_rank
    robust_band = Fraction(band[0]) <= dimension_upper and dimension_lower <= Fraction(band[1])
    return VerifiedEdgeMetricEffectiveDimension(
        status="VERIFIED_EDGE_METRIC_TO_OBSERVED_RIDGE_DIMENSION_BRIDGE",
        validation_level="VERIFIED_EDGE_METRIC_TO_OBSERVED_RIDGE_DIMENSION_BRIDGE",
        metric_at_lower_weights=metric_lower,
        metric_at_upper_weights=metric_upper,
        observed_gram_at_lower_weights=gram_lower,
        observed_gram_at_upper_weights=gram_upper,
        ridge_dimension_at_lower_weights=dimension_lower,
        ridge_dimension_at_upper_weights=dimension_upper,
        ridge_dimension_drop=drop,
        ridge_dimension_drop_upper_bound=drop_upper,
        edge_derivatives_at_lower_weights=tuple(derivatives),
        measurement_rank=measurement_rank,
        hard_rank_at_lower_weights=hard_lower,
        hard_rank_at_upper_weights=hard_upper,
        participation_ratio_at_lower_weights=_participation_ratio(gram_lower),
        participation_ratio_at_upper_weights=_participation_ratio(gram_upper),
        edge_weights_coordinatewise_strengthened=coordinatewise,
        metric_loewner_increase_verified=metric_increase,
        inverse_metric_loewner_decrease_verified=inverse_decrease,
        observed_gram_loewner_decrease_verified=gram_decrease,
        ridge_dimension_monotonicity_verified=monotone,
        hard_rank_invariance_verified=hard_invariant,
        robust_candidate_band_verified=robust_band,
        candidate_band=band,
        participation_ratio_monotonicity_claim_admitted=False,
        selected_signal_rank_claim_admitted=False,
        consciousness_dimension_claim_admitted=False,
    )


__all__ = ["VerifiedEdgeMetricEffectiveDimension", "verified_edge_metric_effective_dimension"]
