"""Residual certificates for a supplied interval-valued similarity family."""

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
    from .verified_interval_tightening import _entry_magnitude_enclosures
    from .verified_rational_contour import (
        ONE,
        ZERO,
        QComplex,
        _dyadic_sqrt,
        _fraction,
        _identity,
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
    from verified_interval_tightening import _entry_magnitude_enclosures  # type: ignore[no-redef]
    from verified_rational_contour import (  # type: ignore[no-redef]
        ONE,
        ZERO,
        QComplex,
        _dyadic_sqrt,
        _fraction,
        _identity,
        _inverse,
        _matmul,
        _matrix,
        parse_qcomplex,
    )


@dataclass(frozen=True)
class IntervalSimilarityNodeCertificate:
    transformed: ResidualNodeCertificate
    translated_node_sigma_lower: Fraction | None


@dataclass(frozen=True)
class VerifiedIntervalSimilarityResidualCircle:
    status: str
    validation_level: str | None
    rank_preserved_for_entire_family: bool
    base_circle: VerifiedResidualIntervalCircle
    nominal_similarity: QMatrix
    nominal_similarity_inverse: QMatrix
    similarity_uncertainty_upper: RMatrix
    relative_similarity_uncertainty_upper: RMatrix
    relative_infinity_contraction_upper: Fraction
    inverse_similarity_magnitude_upper: RMatrix | None
    transformed_uncertainty_upper: RMatrix | None
    similarity_two_norm_upper: Fraction | None
    inverse_similarity_two_norm_upper: Fraction | None
    similarity_condition_two_upper: Fraction | None
    nodes: tuple[IntervalSimilarityNodeCertificate, ...]
    normalized_robust_delta_lower: Fraction | None
    raw_robust_delta_lower: Fraction | None
    normalized_robust_resolvent_upper: Fraction | None
    raw_robust_resolvent_upper: Fraction | None
    projector_perturbation_upper: Fraction | None
    empirical_matrix_provenance_verified: bool
    interval_similarity_selected_from_data: bool
    supplied_interval_similarity_not_optimized: bool


def _add(left: RMatrix, right: RMatrix) -> RMatrix:
    return tuple(
        tuple(left[i][j] + right[i][j] for j in range(len(left)))
        for i in range(len(left))
    )


def _two_norm_upper(matrix: RMatrix, precision: int) -> Fraction:
    _, upper, lower_ok, upper_ok = _dyadic_sqrt(
        _one_norm(matrix) * _infinity_norm(matrix), precision
    )
    if not (lower_ok and upper_ok):
        raise AssertionError("dyadic square-root enclosure self-check failed")
    return upper


def _real_nonnegative(matrix: QMatrix, name: str) -> RMatrix:
    rows = []
    for row in matrix:
        values = []
        for entry in row:
            if entry.imag != 0 or entry.real < 0:
                raise AssertionError(f"{name} must be real and componentwise nonnegative")
            values.append(entry.real)
        rows.append(tuple(values))
    return tuple(rows)


def _similarity(matrix: QMatrix, transform: QMatrix, inverse: QMatrix) -> QMatrix:
    return _matmul(inverse, _matmul(matrix, transform))


def verified_interval_similarity_componentwise_residual_circle(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    approximate_inverses: object,
    nominal_similarity: object,
    similarity_uncertainty_radii: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    nodes: int = 4,
    sqrt_precision: int = 32,
) -> VerifiedIntervalSimilarityResidualCircle:
    """Certify every exact similarity in a componentwise interval around ``T0``."""
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
    transform = _matrix(nominal_similarity, "nominal_similarity")
    if len(transform) != n:
        raise ValueError("nominal_similarity dimension must match nominal_transition")
    inverse = _inverse(transform)
    if inverse is None:
        raise ValueError("nominal_similarity must be exactly invertible")

    _, _, transform_upper = _magnitude_enclosures(transform, sqrt_precision)
    _, _, inverse_upper = _magnitude_enclosures(inverse, sqrt_precision)
    _, _, transform_error = _entry_magnitude_enclosures(
        similarity_uncertainty_radii,
        n=n,
        scale=Fraction(1),
        precision=sqrt_precision,
    )
    relative_error = _nonnegative_matmul(inverse_upper, transform_error)
    theta = _infinity_norm(relative_error)
    common = dict(
        base_circle=base,
        nominal_similarity=transform,
        nominal_similarity_inverse=inverse,
        similarity_uncertainty_upper=transform_error,
        relative_similarity_uncertainty_upper=relative_error,
        relative_infinity_contraction_upper=theta,
        empirical_matrix_provenance_verified=False,
        interval_similarity_selected_from_data=False,
        supplied_interval_similarity_not_optimized=True,
    )
    if theta >= 1:
        return VerifiedIntervalSimilarityResidualCircle(
            status="INTERVAL_SIMILARITY_INVERTIBILITY_UNAVAILABLE",
            validation_level=None,
            rank_preserved_for_entire_family=False,
            inverse_similarity_magnitude_upper=None,
            transformed_uncertainty_upper=None,
            similarity_two_norm_upper=None,
            inverse_similarity_two_norm_upper=None,
            similarity_condition_two_upper=None,
            nodes=(),
            normalized_robust_delta_lower=None,
            raw_robust_delta_lower=None,
            normalized_robust_resolvent_upper=None,
            raw_robust_resolvent_upper=None,
            projector_perturbation_upper=None,
            **common,
        )

    identity = _identity(n)
    neumann_majorant = tuple(
        tuple(
            identity[i][j] - QComplex(relative_error[i][j])
            for j in range(n)
        )
        for i in range(n)
    )
    neumann_inverse = _inverse(neumann_majorant)
    if neumann_inverse is None:
        raise AssertionError("strict infinity contraction must make I-Q invertible")
    inverse_envelope = _nonnegative_matmul(
        _real_nonnegative(neumann_inverse, "Neumann inverse"), inverse_upper
    )
    transform_envelope = _add(transform_upper, transform_error)

    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    radius_q = _fraction(radius, "radius")
    normalized_matrix = tuple(tuple(entry / scale for entry in row) for row in matrix)
    _, _, normalized_matrix_upper = _magnitude_enclosures(
        normalized_matrix, sqrt_precision
    )
    original_uncertainty = tuple(
        tuple(base.normalized_uncertainty_entry_brackets[i][j][1] for j in range(n))
        for i in range(n)
    )

    data_term = _nonnegative_matmul(
        inverse_envelope,
        _nonnegative_matmul(original_uncertainty, transform_envelope),
    )
    right_transform_term = _nonnegative_matmul(
        inverse_envelope,
        _nonnegative_matmul(normalized_matrix_upper, transform_error),
    )
    inverse_transform_term = _nonnegative_matmul(
        inverse_envelope,
        _nonnegative_matmul(
            transform_error,
            _nonnegative_matmul(
                inverse_upper,
                _nonnegative_matmul(normalized_matrix_upper, transform_upper),
            ),
        ),
    )
    transformed_uncertainty = _add(
        _add(data_term, right_transform_term), inverse_transform_term
    )
    transform_norm = _two_norm_upper(transform_envelope, sqrt_precision)
    inverse_norm = _two_norm_upper(inverse_envelope, sqrt_precision)
    condition = transform_norm * inverse_norm

    if not isinstance(approximate_inverses, (tuple, list)) or len(approximate_inverses) != 4:
        raise ValueError("approximate_inverses must contain exactly four matrices")
    witnesses = tuple(
        _matrix(value, f"approximate_inverses[{index}]")
        for index, value in enumerate(approximate_inverses)
    )
    c = parse_qcomplex(center, "center") / scale
    r = radius_q / scale
    directions = (
        ONE,
        QComplex(Fraction(0), Fraction(1)),
        QComplex(Fraction(-1)),
        QComplex(Fraction(0), Fraction(-1)),
    )
    node_results = []
    for index, direction in enumerate(directions):
        z = c + r * direction
        node = tuple(
            tuple(
                (z if i == j else ZERO) - normalized_matrix[i][j]
                for j in range(n)
            )
            for i in range(n)
        )
        transformed_node = _node_certificate(
            _similarity(node, transform, inverse),
            _similarity(witnesses[index], transform, inverse),
            transformed_uncertainty,
            sqrt_precision,
        )
        translated = (
            None
            if transformed_node.node_sigma_lower is None
            else transformed_node.node_sigma_lower / condition
        )
        node_results.append(
            IntervalSimilarityNodeCertificate(
                transformed=transformed_node,
                translated_node_sigma_lower=translated,
            )
        )
    node_tuple = tuple(node_results)
    result_common = dict(
        inverse_similarity_magnitude_upper=inverse_envelope,
        transformed_uncertainty_upper=transformed_uncertainty,
        similarity_two_norm_upper=transform_norm,
        inverse_similarity_two_norm_upper=inverse_norm,
        similarity_condition_two_upper=condition,
        nodes=node_tuple,
        **common,
    )
    if any(node.translated_node_sigma_lower is None for node in node_tuple):
        return VerifiedIntervalSimilarityResidualCircle(
            status="INTERVAL_SIMILARITY_NODE_CONTRACTION_UNAVAILABLE",
            validation_level=None,
            rank_preserved_for_entire_family=False,
            normalized_robust_delta_lower=None,
            raw_robust_delta_lower=None,
            normalized_robust_resolvent_upper=None,
            raw_robust_resolvent_upper=None,
            projector_perturbation_upper=None,
            **result_common,
        )
    lowers = tuple(node.translated_node_sigma_lower for node in node_tuple)
    delta = min(value for value in lowers if value is not None) - base.normalized_chord_upper
    if delta <= 0:
        return VerifiedIntervalSimilarityResidualCircle(
            status="INTERVAL_SIMILARITY_FULL_CIRCLE_LOWER_NONPOSITIVE",
            validation_level=None,
            rank_preserved_for_entire_family=False,
            normalized_robust_delta_lower=delta,
            raw_robust_delta_lower=delta * scale,
            normalized_robust_resolvent_upper=None,
            raw_robust_resolvent_upper=None,
            projector_perturbation_upper=None,
            **result_common,
        )
    resolvent = Fraction(1) / delta
    projector = (
        r
        * base.tightening.normalized_selected_uncertainty_upper
        * resolvent
        * resolvent
    )
    return VerifiedIntervalSimilarityResidualCircle(
        status="VERIFIED_RATIONAL_INTERVAL_SIMILARITY_RESIDUAL_CONTOUR_BRIDGE",
        validation_level="VERIFIED_RATIONAL_INTERVAL_SIMILARITY_RESIDUAL_CONTOUR_BRIDGE",
        rank_preserved_for_entire_family=True,
        normalized_robust_delta_lower=delta,
        raw_robust_delta_lower=delta * scale,
        normalized_robust_resolvent_upper=resolvent,
        raw_robust_resolvent_upper=resolvent / scale,
        projector_perturbation_upper=projector,
        **result_common,
    )


__all__ = [
    "IntervalSimilarityNodeCertificate",
    "VerifiedIntervalSimilarityResidualCircle",
    "verified_interval_similarity_componentwise_residual_circle",
]
