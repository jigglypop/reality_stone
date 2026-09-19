"""Exact commuting theorem and noncommuting boundary for edge/noise dimension."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_edge_metric_effective_dimension import (
        Matrix, _add, _identity, _inverse, _matrix, _metric, _mul,
        _positive_definite, _positive_semidefinite, _q, _rank, _scale,
        _sub, _trace, _transpose,
    )
else:
    from verified_edge_metric_effective_dimension import (  # type: ignore[no-redef]
        Matrix, _add, _identity, _inverse, _matrix, _metric, _mul,
        _positive_definite, _positive_semidefinite, _q, _rank, _scale,
        _sub, _trace, _transpose,
    )


def _commutes(left: Matrix, right: Matrix) -> bool:
    return _mul(left, right) == _mul(right, left)


def _pairwise_commuting(matrices: tuple[Matrix, ...]) -> bool:
    return all(_commutes(matrices[i], matrices[j]) for i in range(len(matrices)) for j in range(i + 1, len(matrices)))


def _response_covariance(metric: Matrix, forcing: Matrix) -> Matrix:
    inverse = _inverse(metric)
    return _mul(_mul(inverse, forcing), inverse)


def _observed(measurement: Matrix, covariance: Matrix) -> Matrix:
    return _mul(_mul(measurement, covariance), _transpose(measurement))


def _ridge(gram: Matrix, ridge: Fraction) -> Fraction:
    return _trace(_mul(gram, _inverse(_add(gram, _scale(ridge, _identity(len(gram)))))))


def _ridge_derivative(
    *, metric: Matrix, forcing: Matrix, edge: Matrix, measurement: Matrix, ridge: Fraction
) -> Fraction:
    inverse = _inverse(metric)
    left = _mul(_mul(_mul(_mul(inverse, edge), inverse), forcing), inverse)
    right = _mul(_mul(_mul(_mul(inverse, forcing), inverse), edge), inverse)
    covariance_derivative = _scale(Fraction(-1), _add(left, right))
    gram = _observed(measurement, _response_covariance(metric, forcing))
    gram_derivative = _observed(measurement, covariance_derivative)
    regularized_inverse = _inverse(_add(gram, _scale(ridge, _identity(len(gram)))))
    return ridge * _trace(_mul(_mul(regularized_inverse, regularized_inverse), gram_derivative))


@dataclass(frozen=True)
class VerifiedEdgeNoiseEffectiveDimension:
    status: str
    validation_level: str
    metric_at_lower_weights: Matrix
    metric_at_upper_weights: Matrix
    forcing_covariance: Matrix
    response_covariance_at_lower_weights: Matrix
    response_covariance_at_upper_weights: Matrix
    observed_gram_at_lower_weights: Matrix
    observed_gram_at_upper_weights: Matrix
    ridge_dimension_at_lower_weights: Fraction
    ridge_dimension_at_upper_weights: Fraction
    ridge_dimension_change_upper_minus_lower: Fraction
    edge_derivatives_at_lower_weights: tuple[Fraction, ...]
    edge_derivatives_at_upper_weights: tuple[Fraction, ...]
    pairwise_commutation_verified: bool
    commuting_noise_monotonicity_theorem_admitted: bool
    response_covariance_loewner_decrease_verified: bool
    observed_gram_loewner_decrease_verified: bool
    hard_observed_rank_at_lower_weights: int
    hard_observed_rank_at_upper_weights: int
    hard_rank_invariance_under_commuting_family_verified: bool
    robust_candidate_band_verified: bool
    candidate_band: tuple[int, int]
    noncommuting_monotonicity_claim_admitted: bool
    selected_signal_rank_claim_admitted: bool
    consciousness_dimension_claim_admitted: bool


def verified_edge_noise_effective_dimension(
    *,
    baseline_metric: object,
    edge_metric_terms: object,
    lower_edge_weights: object,
    upper_edge_weights: object,
    forcing_covariance: object,
    measurement_operator: object,
    ridge_parameter: object,
    candidate_band: object,
) -> VerifiedEdgeNoiseEffectiveDimension:
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
    if any(left > right for left, right in zip(lower, upper, strict=True)):
        raise ValueError("upper_edge_weights must strengthen every edge coordinatewise")
    forcing = _matrix(forcing_covariance, "forcing_covariance", columns=dimension)
    if len(forcing) != dimension or not _positive_semidefinite(forcing):
        raise ValueError("forcing_covariance must be exact symmetric positive semidefinite")
    measurement = _matrix(measurement_operator, "measurement_operator", columns=dimension)
    ridge = _q(ridge_parameter, "ridge_parameter")
    if ridge <= 0:
        raise ValueError("ridge_parameter must be strictly positive")
    if (
        not isinstance(candidate_band, (tuple, list)) or len(candidate_band) != 2
        or any(type(value) is not int for value in candidate_band)
        or candidate_band[0] < 0 or candidate_band[0] > candidate_band[1]
    ):
        raise ValueError("candidate_band must be two ordered nonnegative built-in integers")
    band = tuple(candidate_band)

    metric_lower = _metric(base, terms, lower)
    metric_upper = _metric(base, terms, upper)
    covariance_lower = _response_covariance(metric_lower, forcing)
    covariance_upper = _response_covariance(metric_upper, forcing)
    gram_lower = _observed(measurement, covariance_lower)
    gram_upper = _observed(measurement, covariance_upper)
    dimension_lower = _ridge(gram_lower, ridge)
    dimension_upper = _ridge(gram_upper, ridge)
    commuting = _pairwise_commuting((base, forcing, *terms))
    covariance_decrease = _positive_semidefinite(_sub(covariance_lower, covariance_upper))
    gram_decrease = _positive_semidefinite(_sub(gram_lower, gram_upper))
    derivatives_lower = tuple(
        _ridge_derivative(metric=metric_lower, forcing=forcing, edge=term, measurement=measurement, ridge=ridge)
        for term in terms
    )
    derivatives_upper = tuple(
        _ridge_derivative(metric=metric_upper, forcing=forcing, edge=term, measurement=measurement, ridge=ridge)
        for term in terms
    )
    hard_lower = _rank(gram_lower)
    hard_upper = _rank(gram_upper)
    monotone = commuting and covariance_decrease and gram_decrease and dimension_upper <= dimension_lower and all(
        derivative <= 0 for derivative in derivatives_lower + derivatives_upper
    )
    hard_invariant = commuting and hard_lower == hard_upper
    robust_band = (
        monotone and Fraction(band[0]) <= dimension_upper and dimension_lower <= Fraction(band[1])
    )
    status = (
        "VERIFIED_COMMUTING_EDGE_NOISE_RIDGE_DIMENSION_BRIDGE"
        if commuting
        else "VERIFIED_NONCOMMUTING_EDGE_NOISE_MONOTONICITY_BOUNDARY"
    )
    return VerifiedEdgeNoiseEffectiveDimension(
        status=status,
        validation_level=status,
        metric_at_lower_weights=metric_lower,
        metric_at_upper_weights=metric_upper,
        forcing_covariance=forcing,
        response_covariance_at_lower_weights=covariance_lower,
        response_covariance_at_upper_weights=covariance_upper,
        observed_gram_at_lower_weights=gram_lower,
        observed_gram_at_upper_weights=gram_upper,
        ridge_dimension_at_lower_weights=dimension_lower,
        ridge_dimension_at_upper_weights=dimension_upper,
        ridge_dimension_change_upper_minus_lower=dimension_upper - dimension_lower,
        edge_derivatives_at_lower_weights=derivatives_lower,
        edge_derivatives_at_upper_weights=derivatives_upper,
        pairwise_commutation_verified=commuting,
        commuting_noise_monotonicity_theorem_admitted=monotone,
        response_covariance_loewner_decrease_verified=covariance_decrease,
        observed_gram_loewner_decrease_verified=gram_decrease,
        hard_observed_rank_at_lower_weights=hard_lower,
        hard_observed_rank_at_upper_weights=hard_upper,
        hard_rank_invariance_under_commuting_family_verified=hard_invariant,
        robust_candidate_band_verified=robust_band,
        candidate_band=band,
        noncommuting_monotonicity_claim_admitted=False,
        selected_signal_rank_claim_admitted=False,
        consciousness_dimension_claim_admitted=False,
    )


__all__ = ["VerifiedEdgeNoiseEffectiveDimension", "verified_edge_noise_effective_dimension"]
