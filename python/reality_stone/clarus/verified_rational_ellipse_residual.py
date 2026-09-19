"""Verified residual certificates on rational affine-image ellipse contours."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_interval_residual import (
        BracketMatrix,
        CheckMatrix,
        QMatrix,
        ResidualNodeCertificate,
        _entry_magnitude_enclosures,
        _infinity_norm,
        _node_certificate,
        _one_norm,
    )
    from .verified_rational_contour import (
        ZERO,
        QComplex,
        _dyadic_sqrt,
        _fraction,
        _inverse,
        _matrix,
        parse_qcomplex,
    )
    from .verified_rational_mesh_residual import (
        RationalUnitCircleMesh,
        verified_rational_unit_circle_mesh,
    )
else:
    from verified_interval_residual import (  # type: ignore[no-redef]
        BracketMatrix,
        CheckMatrix,
        QMatrix,
        ResidualNodeCertificate,
        _entry_magnitude_enclosures,
        _infinity_norm,
        _node_certificate,
        _one_norm,
    )
    from verified_rational_contour import (  # type: ignore[no-redef]
        ZERO,
        QComplex,
        _dyadic_sqrt,
        _fraction,
        _inverse,
        _matrix,
        parse_qcomplex,
    )
    from verified_rational_mesh_residual import (  # type: ignore[no-redef]
        RationalUnitCircleMesh,
        verified_rational_unit_circle_mesh,
    )


@dataclass(frozen=True)
class RationalEllipseContour:
    center: QComplex
    axis_u: QComplex
    axis_v: QComplex
    orientation_determinant: Fraction
    gram_trace: Fraction
    gram_discriminant: Fraction
    axis_operator_norm_bracket: tuple[Fraction, Fraction]
    axis_norm_sqrt_self_checks: tuple[bool, ...]
    mesh: RationalUnitCircleMesh
    contour_nodes: tuple[QComplex, ...]
    normalized_cover_chord_upper: Fraction
    normalized_perimeter_over_two_pi_upper: Fraction


@dataclass(frozen=True)
class ExactEllipseResidualWitnessConstruction:
    status: str
    validation_level: str | None
    ellipse: RationalEllipseContour
    normalized_node_matrices: tuple[QMatrix, ...]
    approximate_inverses: tuple[QMatrix, ...] | None
    failing_node_index: int | None
    spectral_reference_scale: Fraction


@dataclass(frozen=True)
class VerifiedResidualRationalEllipse:
    status: str
    validation_level: str | None
    rank_preserved_for_entire_family: bool
    ellipse: RationalEllipseContour
    nodes: tuple[ResidualNodeCertificate, ...]
    normalized_uncertainty_entry_brackets: BracketMatrix
    uncertainty_entry_self_checks: CheckMatrix
    normalized_frobenius_uncertainty_upper: Fraction
    normalized_induced_uncertainty_upper: Fraction
    normalized_selected_uncertainty_upper: Fraction
    selected_uncertainty_method: str
    normalized_robust_delta_lower: Fraction | None
    raw_robust_delta_lower: Fraction | None
    normalized_robust_resolvent_upper: Fraction | None
    raw_robust_resolvent_upper: Fraction | None
    projector_perturbation_upper: Fraction | None
    spectral_reference_scale: Fraction
    sqrt_precision: int
    empirical_matrix_provenance_verified: bool
    smooth_nonpolygonal_contour_verified: bool


def _verified_ellipse_geometry(
    *,
    center: QComplex,
    axis_u: QComplex,
    axis_v: QComplex,
    directions: object,
    sqrt_precision: int,
) -> RationalEllipseContour:
    determinant = axis_u.real * axis_v.imag - axis_u.imag * axis_v.real
    if determinant <= 0:
        raise ValueError("ellipse axes must have strictly positive orientation determinant")
    trace = axis_u.abs_squared() + axis_v.abs_squared()
    discriminant = trace * trace - 4 * determinant * determinant
    if discriminant < 0:
        raise AssertionError("a real 2x2 Gram discriminant cannot be negative")
    disc_lower, disc_upper, disc_lower_ok, disc_upper_ok = _dyadic_sqrt(
        discriminant, sqrt_precision
    )
    lambda_lower = (trace + disc_lower) / 2
    lambda_upper = (trace + disc_upper) / 2
    norm_lower, _, norm_lower_ok, norm_lower_upper_ok = _dyadic_sqrt(
        lambda_lower, sqrt_precision
    )
    _, norm_upper, norm_upper_lower_ok, norm_upper_ok = _dyadic_sqrt(
        lambda_upper, sqrt_precision
    )
    mesh = verified_rational_unit_circle_mesh(
        directions, sqrt_precision=sqrt_precision
    )
    nodes = tuple(
        center + axis_u * direction.real + axis_v * direction.imag
        for direction in mesh.directions
    )
    return RationalEllipseContour(
        center=center,
        axis_u=axis_u,
        axis_v=axis_v,
        orientation_determinant=determinant,
        gram_trace=trace,
        gram_discriminant=discriminant,
        axis_operator_norm_bracket=(norm_lower, norm_upper),
        axis_norm_sqrt_self_checks=(
            disc_lower_ok,
            disc_upper_ok,
            norm_lower_ok,
            norm_lower_upper_ok,
            norm_upper_lower_ok,
            norm_upper_ok,
        ),
        mesh=mesh,
        contour_nodes=nodes,
        normalized_cover_chord_upper=norm_upper * mesh.maximum_chord_factor_upper,
        normalized_perimeter_over_two_pi_upper=norm_upper,
    )


def exact_ellipse_nominal_inverse_witnesses(
    nominal_transition: object,
    *,
    center: object,
    axis_u: object,
    axis_v: object,
    directions: object,
    spectral_reference_scale: object,
    sqrt_precision: int = 32,
) -> ExactEllipseResidualWitnessConstruction:
    matrix = _matrix(nominal_transition, "nominal_transition")
    n = len(matrix)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    ellipse = _verified_ellipse_geometry(
        center=parse_qcomplex(center, "center") / scale,
        axis_u=parse_qcomplex(axis_u, "axis_u") / scale,
        axis_v=parse_qcomplex(axis_v, "axis_v") / scale,
        directions=directions,
        sqrt_precision=sqrt_precision,
    )
    normalized_matrix = tuple(tuple(entry / scale for entry in row) for row in matrix)
    node_matrices = tuple(
        tuple(
            tuple(
                (node if i == j else ZERO) - normalized_matrix[i][j]
                for j in range(n)
            )
            for i in range(n)
        )
        for node in ellipse.contour_nodes
    )
    inverses = []
    for index, node in enumerate(node_matrices):
        inverse = _inverse(node)
        if inverse is None:
            return ExactEllipseResidualWitnessConstruction(
                status="EXACT_RATIONAL_ELLIPSE_NOMINAL_NODE_SINGULAR",
                validation_level=None,
                ellipse=ellipse,
                normalized_node_matrices=node_matrices,
                approximate_inverses=None,
                failing_node_index=index,
                spectral_reference_scale=scale,
            )
        inverses.append(inverse)
    return ExactEllipseResidualWitnessConstruction(
        status="EXACT_RATIONAL_ELLIPSE_NOMINAL_INVERSES_CONSTRUCTED",
        validation_level="EXACT_RATIONAL_ELLIPSE_NOMINAL_INVERSES_CONSTRUCTED",
        ellipse=ellipse,
        normalized_node_matrices=node_matrices,
        approximate_inverses=tuple(inverses),
        failing_node_index=None,
        spectral_reference_scale=scale,
    )


def verified_componentwise_residual_rational_ellipse(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    center: object,
    axis_u: object,
    axis_v: object,
    directions: object,
    approximate_inverses: object,
    spectral_reference_scale: object,
    sqrt_precision: int = 32,
) -> VerifiedResidualRationalEllipse:
    matrix = _matrix(nominal_transition, "nominal_transition")
    n = len(matrix)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    ellipse = _verified_ellipse_geometry(
        center=parse_qcomplex(center, "center") / scale,
        axis_u=parse_qcomplex(axis_u, "axis_u") / scale,
        axis_v=parse_qcomplex(axis_v, "axis_v") / scale,
        directions=directions,
        sqrt_precision=sqrt_precision,
    )
    if not isinstance(approximate_inverses, (tuple, list)) or len(approximate_inverses) != len(ellipse.contour_nodes):
        raise ValueError("approximate_inverses must contain one matrix per ellipse node")
    witnesses = tuple(
        _matrix(value, f"approximate_inverses[{index}]")
        for index, value in enumerate(approximate_inverses)
    )
    if any(len(witness) != n for witness in witnesses):
        raise ValueError("every approximate inverse must shape-match the nominal matrix")
    uncertainty_brackets, uncertainty_checks, uncertainty_upper = _entry_magnitude_enclosures(
        uncertainty_radii, n=n, scale=scale, precision=sqrt_precision
    )
    _, frobenius_upper, _, _ = _dyadic_sqrt(
        sum((entry * entry for row in uncertainty_upper for entry in row), Fraction(0)),
        sqrt_precision,
    )
    _, induced_upper, _, _ = _dyadic_sqrt(
        _one_norm(uncertainty_upper) * _infinity_norm(uncertainty_upper),
        sqrt_precision,
    )
    if induced_upper < frobenius_upper:
        selected_uncertainty = induced_upper
        selected_method = "INDUCED_ONE_INFINITY"
    else:
        selected_uncertainty = frobenius_upper
        selected_method = "FROBENIUS"
    normalized_matrix = tuple(tuple(entry / scale for entry in row) for row in matrix)
    node_tuple = tuple(
        _node_certificate(
            tuple(
                tuple(
                    (node if i == j else ZERO) - normalized_matrix[i][j]
                    for j in range(n)
                )
                for i in range(n)
            ),
            witness,
            uncertainty_upper,
            sqrt_precision,
        )
        for node, witness in zip(ellipse.contour_nodes, witnesses)
    )
    common = dict(
        ellipse=ellipse,
        nodes=node_tuple,
        normalized_uncertainty_entry_brackets=uncertainty_brackets,
        uncertainty_entry_self_checks=uncertainty_checks,
        normalized_frobenius_uncertainty_upper=frobenius_upper,
        normalized_induced_uncertainty_upper=induced_upper,
        normalized_selected_uncertainty_upper=selected_uncertainty,
        selected_uncertainty_method=selected_method,
        spectral_reference_scale=scale,
        sqrt_precision=sqrt_precision,
        empirical_matrix_provenance_verified=False,
        smooth_nonpolygonal_contour_verified=True,
    )
    if any(node.node_sigma_lower is None for node in node_tuple):
        return VerifiedResidualRationalEllipse(
            status="VERIFIED_RATIONAL_ELLIPSE_RESIDUAL_NODE_CONTRACTION_UNAVAILABLE",
            validation_level=None,
            rank_preserved_for_entire_family=False,
            normalized_robust_delta_lower=None,
            raw_robust_delta_lower=None,
            normalized_robust_resolvent_upper=None,
            raw_robust_resolvent_upper=None,
            projector_perturbation_upper=None,
            **common,
        )
    lowers = tuple(node.node_sigma_lower for node in node_tuple)
    delta = min(value for value in lowers if value is not None) - ellipse.normalized_cover_chord_upper
    if delta <= 0:
        return VerifiedResidualRationalEllipse(
            status="VERIFIED_RATIONAL_ELLIPSE_RESIDUAL_COVER_NONPOSITIVE",
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
        ellipse.normalized_perimeter_over_two_pi_upper
        * selected_uncertainty
        * resolvent
        * resolvent
    )
    return VerifiedResidualRationalEllipse(
        status="VERIFIED_RATIONAL_ELLIPSE_COMPONENTWISE_RESIDUAL_CONTOUR_BRIDGE",
        validation_level="VERIFIED_RATIONAL_ELLIPSE_COMPONENTWISE_RESIDUAL_CONTOUR_BRIDGE",
        rank_preserved_for_entire_family=True,
        normalized_robust_delta_lower=delta,
        raw_robust_delta_lower=delta * scale,
        normalized_robust_resolvent_upper=resolvent,
        raw_robust_resolvent_upper=resolvent / scale,
        projector_perturbation_upper=projector,
        **common,
    )


__all__ = [
    "ExactEllipseResidualWitnessConstruction",
    "RationalEllipseContour",
    "VerifiedResidualRationalEllipse",
    "exact_ellipse_nominal_inverse_witnesses",
    "verified_componentwise_residual_rational_ellipse",
]
