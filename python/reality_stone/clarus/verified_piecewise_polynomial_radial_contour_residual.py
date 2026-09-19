"""Verified residuals on periodic piecewise-polynomial radial C^q contours."""

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
    from .verified_polynomial_radial_contour_residual import (
        PolynomialTerm,
        _evaluate_polynomial_radial,
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
    from verified_polynomial_radial_contour_residual import (  # type: ignore[no-redef]
        PolynomialTerm,
        _evaluate_polynomial_radial,
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


PolynomialMap = dict[tuple[int, int], Fraction]


@dataclass(frozen=True)
class PolynomialRadialPatch:
    radial_base: Fraction
    radial_terms: tuple[PolynomialTerm, ...]
    radial_degree: int
    linear_norm_bracket: tuple[Fraction, Fraction]
    higher_amplitude_upper: Fraction
    higher_gradient_upper: Fraction
    radial_minimum_lower: Fraction
    radial_lipschitz_before_affine_upper: Fraction


@dataclass(frozen=True)
class RationalPiecewisePolynomialRadialContour:
    affine_geometry: RationalEllipseContour
    patches: tuple[PolynomialRadialPatch, ...]
    junction_order: int
    junction_derivative_values: tuple[tuple[Fraction, ...], ...]
    junction_self_checks: tuple[tuple[bool, ...], ...]
    minimum_radial_lower: Fraction
    maximum_patch_lipschitz_before_affine_upper: Fraction
    full_lipschitz_upper: Fraction
    contour_nodes: tuple[QComplex, ...]
    normalized_cover_chord_upper: Fraction
    normalized_perimeter_over_two_pi_upper: Fraction
    periodic_cq_radial_contour_verified: bool


@dataclass(frozen=True)
class ExactPiecewisePolynomialRadialResidualWitnessConstruction:
    status: str
    validation_level: str | None
    contour: RationalPiecewisePolynomialRadialContour
    normalized_node_matrices: tuple[QMatrix, ...]
    approximate_inverses: tuple[QMatrix, ...] | None
    failing_node_index: int | None
    spectral_reference_scale: Fraction


@dataclass(frozen=True)
class VerifiedResidualPiecewisePolynomialRadialContour:
    status: str
    validation_level: str | None
    rank_preserved_for_entire_family: bool
    contour: RationalPiecewisePolynomialRadialContour
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
    piecewise_polynomial_cq_contour_verified: bool


def _polynomial_map(base: Fraction, terms: tuple[PolynomialTerm, ...]) -> PolynomialMap:
    result: PolynomialMap = {(0, 0): base}
    for i, j, coefficient in terms:
        result[(i, j)] = result.get((i, j), Fraction(0)) + coefficient
    return {key: value for key, value in result.items() if value != 0}


def _rotation_derivative(polynomial: PolynomialMap) -> PolynomialMap:
    result: PolynomialMap = {}
    for (i, j), coefficient in polynomial.items():
        if i:
            key = (i - 1, j + 1)
            result[key] = result.get(key, Fraction(0)) - coefficient * i
        if j:
            key = (i + 1, j - 1)
            result[key] = result.get(key, Fraction(0)) + coefficient * j
    return {key: value for key, value in result.items() if value != 0}


def _evaluate_map(polynomial: PolynomialMap, direction: QComplex) -> Fraction:
    x, y = direction.real, direction.imag
    return sum(
        (coefficient * x**i * y**j for (i, j), coefficient in polynomial.items()),
        Fraction(0),
    )


def _parse_patches(
    value: object, *, count: int, sqrt_precision: int
) -> tuple[PolynomialRadialPatch, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != count:
        raise ValueError("radial_patches must contain one (base, terms) patch per mesh arc")
    patches = []
    for index, raw in enumerate(value):
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise ValueError(f"radial_patches[{index}] must have shape (base, terms)")
        base = _fraction(raw[0], f"radial_patches[{index}].base")
        terms = _polynomial_terms(raw[1])
        term_map = {(i, j): coefficient for i, j, coefficient in terms}
        px = term_map.get((1, 0), Fraction(0))
        py = term_map.get((0, 1), Fraction(0))
        _, linear_upper, _, _ = _dyadic_sqrt(px * px + py * py, sqrt_precision)
        higher = tuple(term for term in terms if term[0] + term[1] >= 2)
        amplitude = sum((abs(term[2]) for term in higher), Fraction(0))
        gradient = sum(
            (abs(coefficient) * (i + j) for i, j, coefficient in higher),
            Fraction(0),
        )
        minimum = base - linear_upper - amplitude
        if minimum <= 0:
            raise ValueError(
                f"radial_patches[{index}] fails strict polynomial amplitude positivity"
            )
        patches.append(
            PolynomialRadialPatch(
                radial_base=base,
                radial_terms=terms,
                radial_degree=max((i + j for i, j, _ in terms), default=0),
                linear_norm_bracket=(
                    _dyadic_sqrt(px * px + py * py, sqrt_precision)[0],
                    linear_upper,
                ),
                higher_amplitude_upper=amplitude,
                higher_gradient_upper=gradient,
                radial_minimum_lower=minimum,
                radial_lipschitz_before_affine_upper=(
                    base + 2 * linear_upper + amplitude + gradient
                ),
            )
        )
    return tuple(patches)


def _verified_piecewise_geometry(
    *,
    center: QComplex,
    axis_u: QComplex,
    axis_v: QComplex,
    radial_patches: object,
    junction_order: int,
    directions: object,
    sqrt_precision: int,
) -> RationalPiecewisePolynomialRadialContour:
    if isinstance(junction_order, bool) or not isinstance(junction_order, int) or junction_order < 1:
        raise ValueError("junction_order must be an integer at least one")
    affine = _verified_ellipse_geometry(
        center=center,
        axis_u=axis_u,
        axis_v=axis_v,
        directions=directions,
        sqrt_precision=sqrt_precision,
    )
    patches = _parse_patches(
        radial_patches, count=len(affine.mesh.directions), sqrt_precision=sqrt_precision
    )
    derivative_maps: list[tuple[PolynomialMap, ...]] = []
    for patch in patches:
        current = _polynomial_map(patch.radial_base, patch.radial_terms)
        derivatives = [current]
        for _ in range(junction_order):
            current = _rotation_derivative(current)
            derivatives.append(current)
        derivative_maps.append(tuple(derivatives))
    junction_values = []
    junction_checks = []
    for knot, direction in enumerate(affine.mesh.directions):
        left = derivative_maps[(knot - 1) % len(patches)]
        right = derivative_maps[knot]
        values = []
        checks = []
        for order in range(junction_order + 1):
            left_value = _evaluate_map(left[order], direction)
            right_value = _evaluate_map(right[order], direction)
            values.append(right_value)
            checks.append(left_value == right_value)
            if left_value != right_value:
                raise ValueError(
                    f"radial patch junction mismatch at knot {knot}, derivative order {order}"
                )
        junction_values.append(tuple(values))
        junction_checks.append(tuple(checks))
    nodes = tuple(
        center
        + (axis_u * direction.real + axis_v * direction.imag)
        * junction_values[index][0]
        for index, direction in enumerate(affine.mesh.directions)
    )
    before_affine = max(
        patch.radial_lipschitz_before_affine_upper for patch in patches
    )
    full_lipschitz = affine.axis_operator_norm_bracket[1] * before_affine
    return RationalPiecewisePolynomialRadialContour(
        affine_geometry=affine,
        patches=patches,
        junction_order=junction_order,
        junction_derivative_values=tuple(junction_values),
        junction_self_checks=tuple(junction_checks),
        minimum_radial_lower=min(patch.radial_minimum_lower for patch in patches),
        maximum_patch_lipschitz_before_affine_upper=before_affine,
        full_lipschitz_upper=full_lipschitz,
        contour_nodes=nodes,
        normalized_cover_chord_upper=(
            full_lipschitz * affine.mesh.maximum_chord_factor_upper
        ),
        normalized_perimeter_over_two_pi_upper=full_lipschitz,
        periodic_cq_radial_contour_verified=True,
    )


def exact_piecewise_polynomial_radial_nominal_inverse_witnesses(
    nominal_transition: object,
    *,
    center: object,
    axis_u: object,
    axis_v: object,
    radial_patches: object,
    junction_order: int,
    directions: object,
    spectral_reference_scale: object,
    sqrt_precision: int = 32,
) -> ExactPiecewisePolynomialRadialResidualWitnessConstruction:
    matrix = _matrix(nominal_transition, "nominal_transition")
    n = len(matrix)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    contour = _verified_piecewise_geometry(
        center=parse_qcomplex(center, "center") / scale,
        axis_u=parse_qcomplex(axis_u, "axis_u") / scale,
        axis_v=parse_qcomplex(axis_v, "axis_v") / scale,
        radial_patches=radial_patches,
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
            return ExactPiecewisePolynomialRadialResidualWitnessConstruction(
                status="EXACT_PIECEWISE_POLYNOMIAL_RADIAL_NODE_SINGULAR",
                validation_level=None,
                contour=contour,
                normalized_node_matrices=node_matrices,
                approximate_inverses=None,
                failing_node_index=index,
                spectral_reference_scale=scale,
            )
        inverses.append(inverse)
    return ExactPiecewisePolynomialRadialResidualWitnessConstruction(
        status="EXACT_PIECEWISE_POLYNOMIAL_RADIAL_INVERSES_CONSTRUCTED",
        validation_level="EXACT_PIECEWISE_POLYNOMIAL_RADIAL_INVERSES_CONSTRUCTED",
        contour=contour,
        normalized_node_matrices=node_matrices,
        approximate_inverses=tuple(inverses),
        failing_node_index=None,
        spectral_reference_scale=scale,
    )


def verified_componentwise_residual_piecewise_polynomial_radial_contour(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    center: object,
    axis_u: object,
    axis_v: object,
    radial_patches: object,
    junction_order: int,
    directions: object,
    approximate_inverses: object,
    spectral_reference_scale: object,
    sqrt_precision: int = 32,
) -> VerifiedResidualPiecewisePolynomialRadialContour:
    matrix = _matrix(nominal_transition, "nominal_transition")
    n = len(matrix)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    contour = _verified_piecewise_geometry(
        center=parse_qcomplex(center, "center") / scale,
        axis_u=parse_qcomplex(axis_u, "axis_u") / scale,
        axis_v=parse_qcomplex(axis_v, "axis_v") / scale,
        radial_patches=radial_patches,
        junction_order=junction_order,
        directions=directions,
        sqrt_precision=sqrt_precision,
    )
    if not isinstance(approximate_inverses, (tuple, list)) or len(approximate_inverses) != len(contour.contour_nodes):
        raise ValueError("approximate_inverses must contain one matrix per spline knot")
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
        piecewise_polynomial_cq_contour_verified=(
            contour.periodic_cq_radial_contour_verified
        ),
    )
    if any(node.node_sigma_lower is None for node in node_tuple):
        return VerifiedResidualPiecewisePolynomialRadialContour(
            status="VERIFIED_PIECEWISE_POLYNOMIAL_RADIAL_NODE_CONTRACTION_UNAVAILABLE",
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
        return VerifiedResidualPiecewisePolynomialRadialContour(
            status="VERIFIED_PIECEWISE_POLYNOMIAL_RADIAL_COVER_NONPOSITIVE",
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
    return VerifiedResidualPiecewisePolynomialRadialContour(
        status="VERIFIED_PIECEWISE_POLYNOMIAL_RADIAL_CQ_RESIDUAL_CONTOUR",
        validation_level="VERIFIED_PIECEWISE_POLYNOMIAL_RADIAL_CQ_RESIDUAL_CONTOUR",
        rank_preserved_for_entire_family=True,
        normalized_robust_delta_lower=delta,
        raw_robust_delta_lower=delta * scale,
        normalized_robust_resolvent_upper=resolvent,
        raw_robust_resolvent_upper=resolvent / scale,
        projector_perturbation_upper=projector,
        **common,
    )


__all__ = [
    "ExactPiecewisePolynomialRadialResidualWitnessConstruction",
    "PolynomialRadialPatch",
    "RationalPiecewisePolynomialRadialContour",
    "VerifiedResidualPiecewisePolynomialRadialContour",
    "exact_piecewise_polynomial_radial_nominal_inverse_witnesses",
    "verified_componentwise_residual_piecewise_polynomial_radial_contour",
]
