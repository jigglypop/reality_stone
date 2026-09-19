"""Verified residuals on smooth rational affine-linear radial contours."""

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
    from .verified_rational_ellipse_residual import (
        RationalEllipseContour,
        _verified_ellipse_geometry,
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
    from verified_rational_ellipse_residual import (  # type: ignore[no-redef]
        RationalEllipseContour,
        _verified_ellipse_geometry,
    )


@dataclass(frozen=True)
class RationalAffineRadialContour:
    affine_geometry: RationalEllipseContour
    radial_base: Fraction
    radial_cos: Fraction
    radial_sin: Fraction
    radial_linear_norm_bracket: tuple[Fraction, Fraction]
    radial_norm_sqrt_self_checks: tuple[bool, bool]
    radial_minimum_lower: Fraction
    radial_maximum_upper: Fraction
    radial_lipschitz_before_affine_upper: Fraction
    full_lipschitz_upper: Fraction
    contour_nodes: tuple[QComplex, ...]
    normalized_cover_chord_upper: Fraction
    normalized_perimeter_over_two_pi_upper: Fraction
    genuinely_nonaffine_radial_contour: bool


@dataclass(frozen=True)
class ExactRadialResidualWitnessConstruction:
    status: str
    validation_level: str | None
    contour: RationalAffineRadialContour
    normalized_node_matrices: tuple[QMatrix, ...]
    approximate_inverses: tuple[QMatrix, ...] | None
    failing_node_index: int | None
    spectral_reference_scale: Fraction


@dataclass(frozen=True)
class VerifiedResidualRationalRadialContour:
    status: str
    validation_level: str | None
    rank_preserved_for_entire_family: bool
    contour: RationalAffineRadialContour
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
    smooth_nonellipse_contour_verified: bool


def _verified_radial_geometry(
    *,
    center: QComplex,
    axis_u: QComplex,
    axis_v: QComplex,
    radial_base: Fraction,
    radial_cos: Fraction,
    radial_sin: Fraction,
    directions: object,
    sqrt_precision: int,
) -> RationalAffineRadialContour:
    affine = _verified_ellipse_geometry(
        center=center,
        axis_u=axis_u,
        axis_v=axis_v,
        directions=directions,
        sqrt_precision=sqrt_precision,
    )
    radial_squared = radial_cos * radial_cos + radial_sin * radial_sin
    radial_lower, radial_upper, lower_ok, upper_ok = _dyadic_sqrt(
        radial_squared, sqrt_precision
    )
    radial_minimum = radial_base - radial_upper
    if radial_minimum <= 0:
        raise ValueError(
            "radial_base must strictly exceed the radial linear coefficient norm"
        )
    radial_maximum = radial_base + radial_upper
    before_affine = radial_base + 2 * radial_upper
    axis_norm_upper = affine.axis_operator_norm_bracket[1]
    full_lipschitz = axis_norm_upper * before_affine
    nodes = tuple(
        center
        + (
            axis_u * direction.real + axis_v * direction.imag
        )
        * (
            radial_base
            + radial_cos * direction.real
            + radial_sin * direction.imag
        )
        for direction in affine.mesh.directions
    )
    return RationalAffineRadialContour(
        affine_geometry=affine,
        radial_base=radial_base,
        radial_cos=radial_cos,
        radial_sin=radial_sin,
        radial_linear_norm_bracket=(radial_lower, radial_upper),
        radial_norm_sqrt_self_checks=(lower_ok, upper_ok),
        radial_minimum_lower=radial_minimum,
        radial_maximum_upper=radial_maximum,
        radial_lipschitz_before_affine_upper=before_affine,
        full_lipschitz_upper=full_lipschitz,
        contour_nodes=nodes,
        normalized_cover_chord_upper=(
            full_lipschitz * affine.mesh.maximum_chord_factor_upper
        ),
        normalized_perimeter_over_two_pi_upper=full_lipschitz,
        genuinely_nonaffine_radial_contour=(radial_cos != 0 or radial_sin != 0),
    )


def exact_radial_nominal_inverse_witnesses(
    nominal_transition: object,
    *,
    center: object,
    axis_u: object,
    axis_v: object,
    radial_base: object,
    radial_cos: object,
    radial_sin: object,
    directions: object,
    spectral_reference_scale: object,
    sqrt_precision: int = 32,
) -> ExactRadialResidualWitnessConstruction:
    matrix = _matrix(nominal_transition, "nominal_transition")
    n = len(matrix)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    contour = _verified_radial_geometry(
        center=parse_qcomplex(center, "center") / scale,
        axis_u=parse_qcomplex(axis_u, "axis_u") / scale,
        axis_v=parse_qcomplex(axis_v, "axis_v") / scale,
        radial_base=_fraction(radial_base, "radial_base"),
        radial_cos=_fraction(radial_cos, "radial_cos"),
        radial_sin=_fraction(radial_sin, "radial_sin"),
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
        for node in contour.contour_nodes
    )
    inverses = []
    for index, node in enumerate(node_matrices):
        inverse = _inverse(node)
        if inverse is None:
            return ExactRadialResidualWitnessConstruction(
                status="EXACT_RATIONAL_RADIAL_NOMINAL_NODE_SINGULAR",
                validation_level=None,
                contour=contour,
                normalized_node_matrices=node_matrices,
                approximate_inverses=None,
                failing_node_index=index,
                spectral_reference_scale=scale,
            )
        inverses.append(inverse)
    return ExactRadialResidualWitnessConstruction(
        status="EXACT_RATIONAL_RADIAL_NOMINAL_INVERSES_CONSTRUCTED",
        validation_level="EXACT_RATIONAL_RADIAL_NOMINAL_INVERSES_CONSTRUCTED",
        contour=contour,
        normalized_node_matrices=node_matrices,
        approximate_inverses=tuple(inverses),
        failing_node_index=None,
        spectral_reference_scale=scale,
    )


def verified_componentwise_residual_rational_radial_contour(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    center: object,
    axis_u: object,
    axis_v: object,
    radial_base: object,
    radial_cos: object,
    radial_sin: object,
    directions: object,
    approximate_inverses: object,
    spectral_reference_scale: object,
    sqrt_precision: int = 32,
) -> VerifiedResidualRationalRadialContour:
    matrix = _matrix(nominal_transition, "nominal_transition")
    n = len(matrix)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    contour = _verified_radial_geometry(
        center=parse_qcomplex(center, "center") / scale,
        axis_u=parse_qcomplex(axis_u, "axis_u") / scale,
        axis_v=parse_qcomplex(axis_v, "axis_v") / scale,
        radial_base=_fraction(radial_base, "radial_base"),
        radial_cos=_fraction(radial_cos, "radial_cos"),
        radial_sin=_fraction(radial_sin, "radial_sin"),
        directions=directions,
        sqrt_precision=sqrt_precision,
    )
    if not isinstance(approximate_inverses, (tuple, list)) or len(approximate_inverses) != len(contour.contour_nodes):
        raise ValueError("approximate_inverses must contain one matrix per radial node")
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
        for node, witness in zip(contour.contour_nodes, witnesses)
    )
    common = dict(
        contour=contour,
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
        smooth_nonellipse_contour_verified=contour.genuinely_nonaffine_radial_contour,
    )
    if any(node.node_sigma_lower is None for node in node_tuple):
        return VerifiedResidualRationalRadialContour(
            status="VERIFIED_RATIONAL_RADIAL_RESIDUAL_NODE_CONTRACTION_UNAVAILABLE",
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
    delta = min(value for value in lowers if value is not None) - contour.normalized_cover_chord_upper
    if delta <= 0:
        return VerifiedResidualRationalRadialContour(
            status="VERIFIED_RATIONAL_RADIAL_RESIDUAL_COVER_NONPOSITIVE",
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
        contour.normalized_perimeter_over_two_pi_upper
        * selected_uncertainty
        * resolvent
        * resolvent
    )
    return VerifiedResidualRationalRadialContour(
        status="VERIFIED_RATIONAL_AFFINE_RADIAL_COMPONENTWISE_RESIDUAL_CONTOUR_BRIDGE",
        validation_level="VERIFIED_RATIONAL_AFFINE_RADIAL_COMPONENTWISE_RESIDUAL_CONTOUR_BRIDGE",
        rank_preserved_for_entire_family=True,
        normalized_robust_delta_lower=delta,
        raw_robust_delta_lower=delta * scale,
        normalized_robust_resolvent_upper=resolvent,
        raw_robust_resolvent_upper=resolvent / scale,
        projector_perturbation_upper=projector,
        **common,
    )


__all__ = [
    "ExactRadialResidualWitnessConstruction",
    "RationalAffineRadialContour",
    "VerifiedResidualRationalRadialContour",
    "exact_radial_nominal_inverse_witnesses",
    "verified_componentwise_residual_rational_radial_contour",
]
