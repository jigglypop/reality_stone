"""Exact dense-similarity residual certificates translated to the original 2-norm."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_interval_residual import (
        QMatrix,
        RMatrix,
        ResidualNodeCertificate,
        VerifiedResidualIntervalCircle,
        _infinity_norm,
        _magnitude_enclosures,
        _node_certificate,
        _nonnegative_matmul,
        _one_norm,
        verified_componentwise_residual_circle,
    )
    from .verified_rational_contour import (
        ONE,
        ZERO,
        QComplex,
        _dyadic_sqrt,
        _fraction,
        _inverse,
        _matmul,
        _matrix,
        parse_qcomplex,
    )
else:
    from verified_interval_residual import (  # type: ignore[no-redef]
        QMatrix,
        RMatrix,
        ResidualNodeCertificate,
        VerifiedResidualIntervalCircle,
        _infinity_norm,
        _magnitude_enclosures,
        _node_certificate,
        _nonnegative_matmul,
        _one_norm,
        verified_componentwise_residual_circle,
    )
    from verified_rational_contour import (  # type: ignore[no-redef]
        ONE,
        ZERO,
        QComplex,
        _dyadic_sqrt,
        _fraction,
        _inverse,
        _matmul,
        _matrix,
        parse_qcomplex,
    )


@dataclass(frozen=True)
class DenseSimilarityNodeCertificate:
    untransformed: ResidualNodeCertificate
    transformed: ResidualNodeCertificate
    translated_transformed_node_sigma_lower: Fraction | None
    similarity_condition_two_upper: Fraction
    selected_node_sigma_lower: Fraction | None
    selected_method: str | None


@dataclass(frozen=True)
class VerifiedDenseSimilarityResidualIntervalCircle:
    status: str
    validation_level: str | None
    rank_preserved_for_entire_family: bool
    base_circle: VerifiedResidualIntervalCircle
    similarity: QMatrix
    similarity_inverse: QMatrix
    inverse_identity_checks: tuple[bool, bool]
    similarity_magnitude_upper: RMatrix
    similarity_inverse_magnitude_upper: RMatrix
    magnitude_self_checks: tuple[
        tuple[tuple[tuple[bool, bool], ...], ...],
        tuple[tuple[tuple[bool, bool], ...], ...],
    ]
    transformed_uncertainty_upper: RMatrix
    similarity_two_norm_upper: Fraction
    similarity_inverse_two_norm_upper: Fraction
    similarity_condition_two_upper: Fraction
    norm_sqrt_self_checks: tuple[tuple[bool, bool], tuple[bool, bool]]
    nodes: tuple[DenseSimilarityNodeCertificate, ...]
    dense_only_robust_delta_lower: Fraction | None
    normalized_robust_delta_lower: Fraction | None
    raw_robust_delta_lower: Fraction | None
    normalized_robust_resolvent_upper: Fraction | None
    raw_robust_resolvent_upper: Fraction | None
    projector_perturbation_upper: Fraction | None
    empirical_matrix_provenance_verified: bool
    dense_similarity_selected_from_data: bool
    supplied_similarity_not_optimized: bool


def _similarity(matrix: QMatrix, transform: QMatrix, inverse: QMatrix) -> QMatrix:
    return _matmul(inverse, _matmul(matrix, transform))


def _two_norm_upper_from_magnitudes(
    magnitudes: RMatrix, precision: int
) -> tuple[Fraction, tuple[bool, bool]]:
    lower, upper, lower_ok, upper_ok = _dyadic_sqrt(
        _one_norm(magnitudes) * _infinity_norm(magnitudes), precision
    )
    return upper, (lower_ok, upper_ok)


def verified_dense_similarity_componentwise_residual_circle(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    approximate_inverses: object,
    similarity: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    nodes: int = 4,
    sqrt_precision: int = 32,
) -> VerifiedDenseSimilarityResidualIntervalCircle:
    """Verify one exact invertible similarity and translate back to the original norm."""
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
    transform = _matrix(similarity, "similarity")
    if len(transform) != n:
        raise ValueError("similarity dimension must match nominal_transition")
    inverse = _inverse(transform)
    if inverse is None:
        raise ValueError("similarity must be exactly invertible")
    identity = tuple(
        tuple(ONE if i == j else ZERO for j in range(n)) for i in range(n)
    )
    inverse_checks = (
        _matmul(inverse, transform) == identity,
        _matmul(transform, inverse) == identity,
    )
    if not all(inverse_checks):
        raise AssertionError("exact similarity inverse failed a two-sided identity check")

    _, transform_checks, transform_upper = _magnitude_enclosures(transform, sqrt_precision)
    _, inverse_magnitude_checks, inverse_upper = _magnitude_enclosures(inverse, sqrt_precision)
    transform_norm, transform_norm_checks = _two_norm_upper_from_magnitudes(
        transform_upper, sqrt_precision
    )
    inverse_norm, inverse_norm_checks = _two_norm_upper_from_magnitudes(
        inverse_upper, sqrt_precision
    )
    condition = transform_norm * inverse_norm
    original_uncertainty_upper = tuple(
        tuple(base.normalized_uncertainty_entry_brackets[i][j][1] for j in range(n))
        for i in range(n)
    )
    transformed_uncertainty = _nonnegative_matmul(
        inverse_upper,
        _nonnegative_matmul(original_uncertainty_upper, transform_upper),
    )

    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    radius_q = _fraction(radius, "radius")
    u = tuple(tuple(entry / scale for entry in row) for row in matrix)
    c = parse_qcomplex(center, "center") / scale
    r = radius_q / scale
    if not isinstance(approximate_inverses, (tuple, list)) or len(approximate_inverses) != 4:
        raise ValueError("approximate_inverses must contain exactly four matrices")
    witnesses = tuple(
        _matrix(value, f"approximate_inverses[{index}]")
        for index, value in enumerate(approximate_inverses)
    )
    directions = (
        ONE,
        QComplex(Fraction(0), Fraction(1)),
        QComplex(Fraction(-1)),
        QComplex(Fraction(0), Fraction(-1)),
    )
    node_results = []
    for index, direction in enumerate(directions):
        z = c + r * direction
        a0 = tuple(
            tuple((z if i == j else ZERO) - u[i][j] for j in range(n))
            for i in range(n)
        )
        transformed_node = _node_certificate(
            _similarity(a0, transform, inverse),
            _similarity(witnesses[index], transform, inverse),
            transformed_uncertainty,
            sqrt_precision,
        )
        translated = (
            None
            if transformed_node.node_sigma_lower is None
            else transformed_node.node_sigma_lower / condition
        )
        untransformed = base.nodes[index]
        candidates = []
        if untransformed.node_sigma_lower is not None:
            candidates.append((untransformed.node_sigma_lower, "UNTRANSFORMED"))
        if translated is not None:
            candidates.append((translated, "DENSE_SIMILARITY_TRANSLATED"))
        if candidates:
            selected_lower, selected_method = max(
                candidates, key=lambda item: (item[0], item[1] == "UNTRANSFORMED")
            )
        else:
            selected_lower, selected_method = None, None
        node_results.append(
            DenseSimilarityNodeCertificate(
                untransformed=untransformed,
                transformed=transformed_node,
                translated_transformed_node_sigma_lower=translated,
                similarity_condition_two_upper=condition,
                selected_node_sigma_lower=selected_lower,
                selected_method=selected_method,
            )
        )
    node_tuple = tuple(node_results)
    translated_values = tuple(
        node.translated_transformed_node_sigma_lower for node in node_tuple
    )
    dense_delta = None
    if all(value is not None for value in translated_values):
        dense_delta = (
            min(value for value in translated_values if value is not None)
            - base.normalized_chord_upper
        )
    common = dict(
        base_circle=base,
        similarity=transform,
        similarity_inverse=inverse,
        inverse_identity_checks=inverse_checks,
        similarity_magnitude_upper=transform_upper,
        similarity_inverse_magnitude_upper=inverse_upper,
        magnitude_self_checks=(transform_checks, inverse_magnitude_checks),
        transformed_uncertainty_upper=transformed_uncertainty,
        similarity_two_norm_upper=transform_norm,
        similarity_inverse_two_norm_upper=inverse_norm,
        similarity_condition_two_upper=condition,
        norm_sqrt_self_checks=(transform_norm_checks, inverse_norm_checks),
        nodes=node_tuple,
        dense_only_robust_delta_lower=dense_delta,
        empirical_matrix_provenance_verified=False,
        dense_similarity_selected_from_data=False,
        supplied_similarity_not_optimized=True,
    )
    if any(node.selected_node_sigma_lower is None for node in node_tuple):
        return VerifiedDenseSimilarityResidualIntervalCircle(
            status="VERIFIED_DENSE_SIMILARITY_NODE_CONTRACTION_UNAVAILABLE",
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
    delta = min(value for value in selected if value is not None) - base.normalized_chord_upper
    if delta <= 0:
        return VerifiedDenseSimilarityResidualIntervalCircle(
            status="VERIFIED_DENSE_SIMILARITY_FULL_CIRCLE_LOWER_NONPOSITIVE",
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
    return VerifiedDenseSimilarityResidualIntervalCircle(
        status="VERIFIED_RATIONAL_DENSE_SIMILARITY_RESIDUAL_CONTOUR_BRIDGE",
        validation_level="VERIFIED_RATIONAL_DENSE_SIMILARITY_RESIDUAL_CONTOUR_BRIDGE",
        rank_preserved_for_entire_family=True,
        normalized_robust_delta_lower=delta,
        raw_robust_delta_lower=delta * scale,
        normalized_robust_resolvent_upper=resolvent,
        raw_robust_resolvent_upper=resolvent / scale,
        projector_perturbation_upper=projector,
        **common,
    )


__all__ = [
    "DenseSimilarityNodeCertificate",
    "VerifiedDenseSimilarityResidualIntervalCircle",
    "verified_dense_similarity_componentwise_residual_circle",
]
