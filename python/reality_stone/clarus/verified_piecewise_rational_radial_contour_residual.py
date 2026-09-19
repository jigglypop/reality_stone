"""Verified residuals on periodic piecewise-rational radial C^q contours."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb

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
    from .verified_piecewise_polynomial_radial_contour_residual import (
        PolynomialMap,
        _evaluate_map,
        _polynomial_map,
        _rotation_derivative,
    )
    from .verified_polynomial_radial_contour_residual import (
        PolynomialTerm,
        _polynomial_terms,
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
    from verified_piecewise_polynomial_radial_contour_residual import (  # type: ignore[no-redef]
        PolynomialMap,
        _evaluate_map,
        _polynomial_map,
        _rotation_derivative,
    )
    from verified_polynomial_radial_contour_residual import (  # type: ignore[no-redef]
        PolynomialTerm,
        _polynomial_terms,
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
class PolynomialBoundReceipt:
    base: Fraction
    terms: tuple[PolynomialTerm, ...]
    degree: int
    minimum_lower: Fraction
    maximum_upper: Fraction
    gradient_upper: Fraction


@dataclass(frozen=True)
class RationalRadialPatch:
    numerator: PolynomialBoundReceipt
    denominator: PolynomialBoundReceipt
    radial_minimum_lower: Fraction
    radial_maximum_upper: Fraction
    radial_gradient_upper: Fraction
    radial_lipschitz_before_affine_upper: Fraction


@dataclass(frozen=True)
class RationalPiecewiseRationalRadialContour:
    affine_geometry: RationalEllipseContour
    patches: tuple[RationalRadialPatch, ...]
    junction_order: int
    junction_quotient_jets: tuple[tuple[Fraction, ...], ...]
    junction_self_checks: tuple[tuple[bool, ...], ...]
    minimum_denominator_lower: Fraction
    minimum_radial_lower: Fraction
    maximum_patch_lipschitz_before_affine_upper: Fraction
    full_lipschitz_upper: Fraction
    contour_nodes: tuple[QComplex, ...]
    normalized_cover_chord_upper: Fraction
    normalized_perimeter_over_two_pi_upper: Fraction
    denominator_pole_exclusion_verified: bool
    periodic_rational_cq_contour_verified: bool


@dataclass(frozen=True)
class ExactPiecewiseRationalRadialResidualWitnessConstruction:
    status: str
    validation_level: str | None
    contour: RationalPiecewiseRationalRadialContour
    normalized_node_matrices: tuple[QMatrix, ...]
    approximate_inverses: tuple[QMatrix, ...] | None
    failing_node_index: int | None
    spectral_reference_scale: Fraction


@dataclass(frozen=True)
class VerifiedResidualPiecewiseRationalRadialContour:
    status: str
    validation_level: str | None
    rank_preserved_for_entire_family: bool
    contour: RationalPiecewiseRationalRadialContour
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
    piecewise_rational_cq_contour_verified: bool


def _polynomial_bounds(
    raw: object, *, name: str, sqrt_precision: int
) -> PolynomialBoundReceipt:
    if not isinstance(raw, (tuple, list)) or len(raw) != 2:
        raise ValueError(f"{name} must have shape (base, terms)")
    base = _fraction(raw[0], f"{name}.base")
    terms = _polynomial_terms(raw[1])
    term_map = {(i, j): coefficient for i, j, coefficient in terms}
    px = term_map.get((1, 0), Fraction(0))
    py = term_map.get((0, 1), Fraction(0))
    _, linear_upper, _, _ = _dyadic_sqrt(px * px + py * py, sqrt_precision)
    higher = tuple(term for term in terms if term[0] + term[1] >= 2)
    amplitude = sum((abs(term[2]) for term in higher), Fraction(0))
    gradient_high = sum(
        (abs(coefficient) * (i + j) for i, j, coefficient in higher),
        Fraction(0),
    )
    return PolynomialBoundReceipt(
        base=base,
        terms=terms,
        degree=max((i + j for i, j, _ in terms), default=0),
        minimum_lower=base - linear_upper - amplitude,
        maximum_upper=base + linear_upper + amplitude,
        gradient_upper=linear_upper + gradient_high,
    )


def _parse_rational_patches(
    value: object, *, count: int, sqrt_precision: int
) -> tuple[RationalRadialPatch, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != count:
        raise ValueError("rational_patches must contain one (numerator, denominator) per mesh arc")
    patches = []
    for index, raw in enumerate(value):
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise ValueError(
                f"rational_patches[{index}] must have shape (numerator, denominator)"
            )
        numerator = _polynomial_bounds(
            raw[0], name=f"rational_patches[{index}].numerator", sqrt_precision=sqrt_precision
        )
        denominator = _polynomial_bounds(
            raw[1], name=f"rational_patches[{index}].denominator", sqrt_precision=sqrt_precision
        )
        if numerator.minimum_lower <= 0:
            raise ValueError(f"rational_patches[{index}] numerator positivity is nonpositive")
        if denominator.minimum_lower <= 0:
            raise ValueError(f"rational_patches[{index}] denominator pole margin is nonpositive")
        radial_minimum = numerator.minimum_lower / denominator.maximum_upper
        radial_maximum = numerator.maximum_upper / denominator.minimum_lower
        radial_gradient = (
            numerator.gradient_upper / denominator.minimum_lower
            + numerator.maximum_upper
            * denominator.gradient_upper
            / (denominator.minimum_lower * denominator.minimum_lower)
        )
        patches.append(
            RationalRadialPatch(
                numerator=numerator,
                denominator=denominator,
                radial_minimum_lower=radial_minimum,
                radial_maximum_upper=radial_maximum,
                radial_gradient_upper=radial_gradient,
                radial_lipschitz_before_affine_upper=radial_maximum + radial_gradient,
            )
        )
    return tuple(patches)


def _derivative_maps(polynomial: PolynomialMap, order: int) -> tuple[PolynomialMap, ...]:
    values = [polynomial]
    for _ in range(order):
        polynomial = _rotation_derivative(polynomial)
        values.append(polynomial)
    return tuple(values)


def _quotient_jet(
    numerator: tuple[PolynomialMap, ...],
    denominator: tuple[PolynomialMap, ...],
    direction: QComplex,
) -> tuple[Fraction, ...]:
    n_values = tuple(_evaluate_map(value, direction) for value in numerator)
    d_values = tuple(_evaluate_map(value, direction) for value in denominator)
    if d_values[0] == 0:
        raise AssertionError("certified positive denominator vanished at a knot")
    jets = []
    for order in range(len(n_values)):
        correction = sum(
            (
                Fraction(comb(order, k)) * d_values[k] * jets[order - k]
                for k in range(1, order + 1)
            ),
            Fraction(0),
        )
        jets.append((n_values[order] - correction) / d_values[0])
    return tuple(jets)


def _verified_piecewise_rational_geometry(
    *,
    center: QComplex,
    axis_u: QComplex,
    axis_v: QComplex,
    rational_patches: object,
    junction_order: int,
    directions: object,
    sqrt_precision: int,
) -> RationalPiecewiseRationalRadialContour:
    if isinstance(junction_order, bool) or not isinstance(junction_order, int) or junction_order < 1:
        raise ValueError("junction_order must be an integer at least one")
    affine = _verified_ellipse_geometry(
        center=center,
        axis_u=axis_u,
        axis_v=axis_v,
        directions=directions,
        sqrt_precision=sqrt_precision,
    )
    patches = _parse_rational_patches(
        rational_patches, count=len(affine.mesh.directions), sqrt_precision=sqrt_precision
    )
    maps = []
    for patch in patches:
        numerator = _derivative_maps(
            _polynomial_map(patch.numerator.base, patch.numerator.terms), junction_order
        )
        denominator = _derivative_maps(
            _polynomial_map(patch.denominator.base, patch.denominator.terms), junction_order
        )
        maps.append((numerator, denominator))
    junction_jets = []
    junction_checks = []
    for knot, direction in enumerate(affine.mesh.directions):
        left = _quotient_jet(*maps[(knot - 1) % len(patches)], direction)
        right = _quotient_jet(*maps[knot], direction)
        checks = tuple(left[order] == right[order] for order in range(junction_order + 1))
        for order, check in enumerate(checks):
            if not check:
                raise ValueError(
                    f"rational patch junction mismatch at knot {knot}, derivative order {order}"
                )
        junction_jets.append(right)
        junction_checks.append(checks)
    nodes = tuple(
        center
        + (axis_u * direction.real + axis_v * direction.imag)
        * junction_jets[index][0]
        for index, direction in enumerate(affine.mesh.directions)
    )
    before_affine = max(
        patch.radial_lipschitz_before_affine_upper for patch in patches
    )
    full_lipschitz = affine.axis_operator_norm_bracket[1] * before_affine
    return RationalPiecewiseRationalRadialContour(
        affine_geometry=affine,
        patches=patches,
        junction_order=junction_order,
        junction_quotient_jets=tuple(junction_jets),
        junction_self_checks=tuple(junction_checks),
        minimum_denominator_lower=min(
            patch.denominator.minimum_lower for patch in patches
        ),
        minimum_radial_lower=min(patch.radial_minimum_lower for patch in patches),
        maximum_patch_lipschitz_before_affine_upper=before_affine,
        full_lipschitz_upper=full_lipschitz,
        contour_nodes=nodes,
        normalized_cover_chord_upper=(
            full_lipschitz * affine.mesh.maximum_chord_factor_upper
        ),
        normalized_perimeter_over_two_pi_upper=full_lipschitz,
        denominator_pole_exclusion_verified=True,
        periodic_rational_cq_contour_verified=True,
    )


def exact_piecewise_rational_radial_nominal_inverse_witnesses(
    nominal_transition: object,
    *,
    center: object,
    axis_u: object,
    axis_v: object,
    rational_patches: object,
    junction_order: int,
    directions: object,
    spectral_reference_scale: object,
    sqrt_precision: int = 32,
) -> ExactPiecewiseRationalRadialResidualWitnessConstruction:
    matrix = _matrix(nominal_transition, "nominal_transition")
    n = len(matrix)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    contour = _verified_piecewise_rational_geometry(
        center=parse_qcomplex(center, "center") / scale,
        axis_u=parse_qcomplex(axis_u, "axis_u") / scale,
        axis_v=parse_qcomplex(axis_v, "axis_v") / scale,
        rational_patches=rational_patches,
        junction_order=junction_order,
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
            return ExactPiecewiseRationalRadialResidualWitnessConstruction(
                status="EXACT_PIECEWISE_RATIONAL_RADIAL_NODE_SINGULAR",
                validation_level=None,
                contour=contour,
                normalized_node_matrices=node_matrices,
                approximate_inverses=None,
                failing_node_index=index,
                spectral_reference_scale=scale,
            )
        inverses.append(inverse)
    return ExactPiecewiseRationalRadialResidualWitnessConstruction(
        status="EXACT_PIECEWISE_RATIONAL_RADIAL_INVERSES_CONSTRUCTED",
        validation_level="EXACT_PIECEWISE_RATIONAL_RADIAL_INVERSES_CONSTRUCTED",
        contour=contour,
        normalized_node_matrices=node_matrices,
        approximate_inverses=tuple(inverses),
        failing_node_index=None,
        spectral_reference_scale=scale,
    )


def verified_componentwise_residual_piecewise_rational_radial_contour(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    center: object,
    axis_u: object,
    axis_v: object,
    rational_patches: object,
    junction_order: int,
    directions: object,
    approximate_inverses: object,
    spectral_reference_scale: object,
    sqrt_precision: int = 32,
) -> VerifiedResidualPiecewiseRationalRadialContour:
    matrix = _matrix(nominal_transition, "nominal_transition")
    n = len(matrix)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    contour = _verified_piecewise_rational_geometry(
        center=parse_qcomplex(center, "center") / scale,
        axis_u=parse_qcomplex(axis_u, "axis_u") / scale,
        axis_v=parse_qcomplex(axis_v, "axis_v") / scale,
        rational_patches=rational_patches,
        junction_order=junction_order,
        directions=directions,
        sqrt_precision=sqrt_precision,
    )
    if not isinstance(approximate_inverses, (tuple, list)) or len(approximate_inverses) != len(contour.contour_nodes):
        raise ValueError("approximate_inverses must contain one matrix per rational-spline knot")
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
        piecewise_rational_cq_contour_verified=(
            contour.periodic_rational_cq_contour_verified
        ),
    )
    if any(node.node_sigma_lower is None for node in node_tuple):
        return VerifiedResidualPiecewiseRationalRadialContour(
            status="VERIFIED_PIECEWISE_RATIONAL_RADIAL_NODE_CONTRACTION_UNAVAILABLE",
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
        return VerifiedResidualPiecewiseRationalRadialContour(
            status="VERIFIED_PIECEWISE_RATIONAL_RADIAL_COVER_NONPOSITIVE",
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
    return VerifiedResidualPiecewiseRationalRadialContour(
        status="VERIFIED_PIECEWISE_RATIONAL_RADIAL_CQ_RESIDUAL_CONTOUR",
        validation_level="VERIFIED_PIECEWISE_RATIONAL_RADIAL_CQ_RESIDUAL_CONTOUR",
        rank_preserved_for_entire_family=True,
        normalized_robust_delta_lower=delta,
        raw_robust_delta_lower=delta * scale,
        normalized_robust_resolvent_upper=resolvent,
        raw_robust_resolvent_upper=resolvent / scale,
        projector_perturbation_upper=projector,
        **common,
    )


__all__ = [
    "ExactPiecewiseRationalRadialResidualWitnessConstruction",
    "PolynomialBoundReceipt",
    "RationalPiecewiseRationalRadialContour",
    "RationalRadialPatch",
    "VerifiedResidualPiecewiseRationalRadialContour",
    "exact_piecewise_rational_radial_nominal_inverse_witnesses",
    "verified_componentwise_residual_piecewise_rational_radial_contour",
]
