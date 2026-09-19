"""Exact diagonal-similarity tightening for componentwise residual contours."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_interval_residual import (
        QMatrix,
        RMatrix,
        ResidualNodeCertificate,
        VerifiedResidualIntervalCircle,
        _node_certificate,
        verified_componentwise_residual_circle,
    )
    from .verified_rational_contour import (
        ONE,
        ZERO,
        QComplex,
        _fraction,
        _matrix,
        parse_qcomplex,
    )
else:
    from verified_interval_residual import (  # type: ignore[no-redef]
        QMatrix,
        RMatrix,
        ResidualNodeCertificate,
        VerifiedResidualIntervalCircle,
        _node_certificate,
        verified_componentwise_residual_circle,
    )
    from verified_rational_contour import (  # type: ignore[no-redef]
        ONE,
        ZERO,
        QComplex,
        _fraction,
        _matrix,
        parse_qcomplex,
    )


@dataclass(frozen=True)
class WeightedResidualNodeCertificate:
    unweighted: ResidualNodeCertificate
    weighted: ResidualNodeCertificate
    translated_weighted_node_sigma_lower: Fraction | None
    weight_condition_two: Fraction
    selected_node_sigma_lower: Fraction | None
    selected_method: str | None


@dataclass(frozen=True)
class VerifiedWeightedResidualIntervalCircle:
    status: str
    validation_level: str | None
    rank_preserved_for_entire_family: bool
    base_circle: VerifiedResidualIntervalCircle
    normalized_weights: tuple[Fraction, ...]
    weight_condition_two: Fraction
    nodes: tuple[WeightedResidualNodeCertificate, ...]
    normalized_robust_delta_lower: Fraction | None
    raw_robust_delta_lower: Fraction | None
    normalized_robust_resolvent_upper: Fraction | None
    raw_robust_resolvent_upper: Fraction | None
    projector_perturbation_upper: Fraction | None


def _weights(value: object, n: int) -> tuple[Fraction, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != n:
        raise ValueError("diagonal_weights must contain one exact positive entry per matrix row")
    parsed = tuple(_fraction(entry, f"diagonal_weights[{i}]") for i, entry in enumerate(value))
    if any(entry <= 0 for entry in parsed):
        raise ValueError("diagonal_weights must be strictly positive")
    minimum = min(parsed)
    return tuple(entry / minimum for entry in parsed)


def _similarity_q(matrix: QMatrix, weights: tuple[Fraction, ...]) -> QMatrix:
    n = len(matrix)
    return tuple(
        tuple(matrix[i][j] * weights[j] / weights[i] for j in range(n))
        for i in range(n)
    )


def _similarity_r(matrix: RMatrix, weights: tuple[Fraction, ...]) -> RMatrix:
    n = len(matrix)
    return tuple(
        tuple(matrix[i][j] * weights[j] / weights[i] for j in range(n))
        for i in range(n)
    )


def verified_weighted_componentwise_residual_circle(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    approximate_inverses: object,
    diagonal_weights: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    nodes: int = 4,
    sqrt_precision: int = 32,
) -> VerifiedWeightedResidualIntervalCircle:
    """Apply one declared diagonal similarity and translate back to the 2-norm."""
    base = verified_componentwise_residual_circle(
        nominal_transition,
        uncertainty_radii=uncertainty_radii,
        approximate_inverses=approximate_inverses,
        center=center,
        radius=radius,
        spectral_reference_scale=spectral_reference_scale,
        nodes=nodes,
        sqrt_precision=sqrt_precision,
    )
    matrix = _matrix(nominal_transition, "nominal_transition")
    n = len(matrix)
    weights = _weights(diagonal_weights, n)
    condition = max(weights)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    radius_q = _fraction(radius, "radius")
    u = tuple(tuple(entry / scale for entry in row) for row in matrix)
    c = parse_qcomplex(center, "center") / scale
    r = radius_q / scale
    if not isinstance(approximate_inverses, (tuple, list)) or len(approximate_inverses) != 4:
        raise ValueError("approximate_inverses must contain exactly four matrices")
    witnesses = tuple(
        _matrix(value, f"approximate_inverses[{k}]")
        for k, value in enumerate(approximate_inverses)
    )
    uncertainty_upper = tuple(
        tuple(base.normalized_uncertainty_entry_brackets[i][j][1] for j in range(n))
        for i in range(n)
    )
    weighted_uncertainty = _similarity_r(uncertainty_upper, weights)
    directions = (
        ONE,
        QComplex(Fraction(0), Fraction(1)),
        QComplex(Fraction(-1)),
        QComplex(Fraction(0), Fraction(-1)),
    )
    weighted_results = []
    for index, direction in enumerate(directions):
        z = c + r * direction
        a0 = tuple(
            tuple((z if i == j else ZERO) - u[i][j] for j in range(n))
            for i in range(n)
        )
        weighted_node = _node_certificate(
            _similarity_q(a0, weights),
            _similarity_q(witnesses[index], weights),
            weighted_uncertainty,
            sqrt_precision,
        )
        translated = (
            None
            if weighted_node.node_sigma_lower is None
            else weighted_node.node_sigma_lower / condition
        )
        unweighted = base.nodes[index]
        candidates = []
        if unweighted.node_sigma_lower is not None:
            candidates.append((unweighted.node_sigma_lower, "UNWEIGHTED"))
        if translated is not None:
            candidates.append((translated, "WEIGHTED_TRANSLATED"))
        if candidates:
            selected_lower, selected_method = max(candidates, key=lambda item: (item[0], item[1] == "UNWEIGHTED"))
        else:
            selected_lower, selected_method = None, None
        weighted_results.append(
            WeightedResidualNodeCertificate(
                unweighted=unweighted,
                weighted=weighted_node,
                translated_weighted_node_sigma_lower=translated,
                weight_condition_two=condition,
                selected_node_sigma_lower=selected_lower,
                selected_method=selected_method,
            )
        )
    node_tuple = tuple(weighted_results)
    common = dict(
        base_circle=base,
        normalized_weights=weights,
        weight_condition_two=condition,
        nodes=node_tuple,
    )
    if any(node.selected_node_sigma_lower is None for node in node_tuple):
        return VerifiedWeightedResidualIntervalCircle(
            status="VERIFIED_WEIGHTED_RESIDUAL_NODE_CONTRACTION_UNAVAILABLE",
            validation_level=None,
            rank_preserved_for_entire_family=False,
            normalized_robust_delta_lower=None,
            raw_robust_delta_lower=None,
            normalized_robust_resolvent_upper=None,
            raw_robust_resolvent_upper=None,
            projector_perturbation_upper=None,
            **common,
        )
    selected = tuple(node.selected_node_sigma_lower for node in node_tuple)
    assert all(value is not None for value in selected)
    delta = min(value for value in selected if value is not None) - base.normalized_chord_upper
    if delta <= 0:
        return VerifiedWeightedResidualIntervalCircle(
            status="VERIFIED_WEIGHTED_RESIDUAL_FULL_CIRCLE_LOWER_NONPOSITIVE",
            validation_level=None,
            rank_preserved_for_entire_family=False,
            normalized_robust_delta_lower=delta,
            raw_robust_delta_lower=delta * scale,
            normalized_robust_resolvent_upper=None,
            raw_robust_resolvent_upper=None,
            projector_perturbation_upper=None,
            **common,
        )
    resolvent = Fraction(1) / delta
    projector = (
        r
        * base.tightening.normalized_selected_uncertainty_upper
        * resolvent
        * resolvent
    )
    return VerifiedWeightedResidualIntervalCircle(
        status="VERIFIED_RATIONAL_WEIGHTED_RESIDUAL_CONTOUR_BRIDGE",
        validation_level="VERIFIED_RATIONAL_WEIGHTED_RESIDUAL_CONTOUR_BRIDGE",
        rank_preserved_for_entire_family=True,
        normalized_robust_delta_lower=delta,
        raw_robust_delta_lower=delta * scale,
        normalized_robust_resolvent_upper=resolvent,
        raw_robust_resolvent_upper=resolvent / scale,
        projector_perturbation_upper=projector,
        **common,
    )


__all__ = [
    "VerifiedWeightedResidualIntervalCircle",
    "WeightedResidualNodeCertificate",
    "verified_weighted_componentwise_residual_circle",
]

