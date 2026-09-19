"""Exact componentwise residual certificates on ordered rational unit-circle meshes."""

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


@dataclass(frozen=True)
class RationalUnitCircleMesh:
    directions: tuple[QComplex, ...]
    consecutive_dot_products: tuple[Fraction, ...]
    consecutive_cross_products: tuple[Fraction, ...]
    gap_chord_factor_brackets: tuple[tuple[Fraction, Fraction], ...]
    gap_sqrt_self_checks: tuple[tuple[bool, bool, bool, bool], ...]
    maximum_chord_factor_upper: Fraction
    sqrt_precision: int


@dataclass(frozen=True)
class ExactMeshResidualWitnessConstruction:
    status: str
    validation_level: str | None
    mesh: RationalUnitCircleMesh
    normalized_node_matrices: tuple[QMatrix, ...]
    approximate_inverses: tuple[QMatrix, ...] | None
    failing_node_index: int | None
    spectral_reference_scale: Fraction


@dataclass(frozen=True)
class VerifiedResidualRationalMeshCircle:
    status: str
    validation_level: str | None
    rank_preserved_for_entire_family: bool
    mesh: RationalUnitCircleMesh
    nodes: tuple[ResidualNodeCertificate, ...]
    normalized_uncertainty_entry_brackets: BracketMatrix
    uncertainty_entry_self_checks: CheckMatrix
    normalized_frobenius_uncertainty_upper: Fraction
    normalized_induced_uncertainty_upper: Fraction
    normalized_selected_uncertainty_upper: Fraction
    selected_uncertainty_method: str
    normalized_chord_upper: Fraction
    raw_chord_upper: Fraction
    normalized_robust_delta_lower: Fraction | None
    raw_robust_delta_lower: Fraction | None
    normalized_robust_resolvent_upper: Fraction | None
    raw_robust_resolvent_upper: Fraction | None
    projector_perturbation_upper: Fraction | None
    spectral_reference_scale: Fraction
    sqrt_precision: int
    empirical_matrix_provenance_verified: bool


def verified_rational_unit_circle_mesh(
    directions: object, *, sqrt_precision: int = 32
) -> RationalUnitCircleMesh:
    if type(sqrt_precision) is not int or sqrt_precision < 0:
        raise ValueError("square-root precision must be a nonnegative built-in integer")
    if not isinstance(directions, (tuple, list)) or len(directions) < 3:
        raise ValueError("directions must contain at least three ordered unit directions")
    parsed = tuple(
        parse_qcomplex(value, f"directions[{index}]")
        for index, value in enumerate(directions)
    )
    if any(value.abs_squared() != 1 for value in parsed):
        raise ValueError("every mesh direction must lie exactly on the rational unit circle")
    if len(set(parsed)) != len(parsed):
        raise ValueError("mesh directions must be unique")
    first = parsed[0]
    relative = tuple(
        (
            first.real * value.real + first.imag * value.imag,
            first.real * value.imag - first.imag * value.real,
        )
        for value in parsed
    )

    def polar_less(left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]) -> bool:
        left_half = 0 if left[1] > 0 or (left[1] == 0 and left[0] >= 0) else 1
        right_half = 0 if right[1] > 0 or (right[1] == 0 and right[0] >= 0) else 1
        if left_half != right_half:
            return left_half < right_half
        return left[0] * right[1] - left[1] * right[0] > 0

    if any(
        not polar_less(relative[index], relative[index + 1])
        for index in range(len(relative) - 1)
    ):
        raise ValueError("directions must make exactly one counterclockwise cyclic traversal")
    dots = []
    crosses = []
    brackets = []
    checks = []
    for index, left in enumerate(parsed):
        right = parsed[(index + 1) % len(parsed)]
        dot = left.real * right.real + left.imag * right.imag
        cross = left.real * right.imag - left.imag * right.real
        if not (cross > 0 or (cross == 0 and dot == -1)):
            raise ValueError(
                "directions must be strictly counterclockwise with every cyclic gap at most pi"
            )
        twice_half_cos_lower, _, half_lower_ok, half_upper_ok = _dyadic_sqrt(
            2 + 2 * dot, sqrt_precision
        )
        chord_squared_upper = 2 - twice_half_cos_lower
        chord_lower, chord_upper, chord_lower_ok, chord_upper_ok = _dyadic_sqrt(
            chord_squared_upper, sqrt_precision
        )
        dots.append(dot)
        crosses.append(cross)
        brackets.append((chord_lower, chord_upper))
        checks.append((half_lower_ok, half_upper_ok, chord_lower_ok, chord_upper_ok))
    return RationalUnitCircleMesh(
        directions=parsed,
        consecutive_dot_products=tuple(dots),
        consecutive_cross_products=tuple(crosses),
        gap_chord_factor_brackets=tuple(brackets),
        gap_sqrt_self_checks=tuple(checks),
        maximum_chord_factor_upper=max(upper for _, upper in brackets),
        sqrt_precision=sqrt_precision,
    )


def exact_mesh_nominal_inverse_witnesses(
    nominal_transition: object,
    *,
    directions: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    sqrt_precision: int = 32,
) -> ExactMeshResidualWitnessConstruction:
    mesh = verified_rational_unit_circle_mesh(
        directions, sqrt_precision=sqrt_precision
    )
    matrix = _matrix(nominal_transition, "nominal_transition")
    n = len(matrix)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    radius_q = _fraction(radius, "radius")
    if scale <= 0 or radius_q <= 0:
        raise ValueError("radius and spectral_reference_scale must be positive")
    u = tuple(tuple(entry / scale for entry in row) for row in matrix)
    c = parse_qcomplex(center, "center") / scale
    r = radius_q / scale
    node_matrices = tuple(
        tuple(
            tuple(
                (c + r * direction if i == j else ZERO) - u[i][j]
                for j in range(n)
            )
            for i in range(n)
        )
        for direction in mesh.directions
    )
    inverses = []
    for index, node in enumerate(node_matrices):
        inverse = _inverse(node)
        if inverse is None:
            return ExactMeshResidualWitnessConstruction(
                status="EXACT_RATIONAL_MESH_NOMINAL_NODE_SINGULAR",
                validation_level=None,
                mesh=mesh,
                normalized_node_matrices=node_matrices,
                approximate_inverses=None,
                failing_node_index=index,
                spectral_reference_scale=scale,
            )
        inverses.append(inverse)
    return ExactMeshResidualWitnessConstruction(
        status="EXACT_RATIONAL_MESH_NOMINAL_INVERSES_CONSTRUCTED",
        validation_level="EXACT_RATIONAL_MESH_NOMINAL_INVERSES_CONSTRUCTED",
        mesh=mesh,
        normalized_node_matrices=node_matrices,
        approximate_inverses=tuple(inverses),
        failing_node_index=None,
        spectral_reference_scale=scale,
    )


def verified_componentwise_residual_rational_mesh_circle(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    directions: object,
    approximate_inverses: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    sqrt_precision: int = 32,
) -> VerifiedResidualRationalMeshCircle:
    mesh = verified_rational_unit_circle_mesh(
        directions, sqrt_precision=sqrt_precision
    )
    matrix = _matrix(nominal_transition, "nominal_transition")
    n = len(matrix)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    radius_q = _fraction(radius, "radius")
    if scale <= 0 or radius_q <= 0:
        raise ValueError("radius and spectral_reference_scale must be positive")
    if not isinstance(approximate_inverses, (tuple, list)) or len(approximate_inverses) != len(mesh.directions):
        raise ValueError("approximate_inverses must contain one matrix per mesh direction")
    witnesses = tuple(
        _matrix(value, f"approximate_inverses[{index}]")
        for index, value in enumerate(approximate_inverses)
    )
    if any(len(witness) != n for witness in witnesses):
        raise ValueError("every approximate inverse must shape-match the nominal matrix")
    uncertainty_brackets, uncertainty_checks, uncertainty_upper = _entry_magnitude_enclosures(
        uncertainty_radii, n=n, scale=scale, precision=sqrt_precision
    )
    frobenius_lower, frobenius_upper, _, _ = _dyadic_sqrt(
        sum((entry * entry for row in uncertainty_upper for entry in row), Fraction(0)),
        sqrt_precision,
    )
    induced_lower, induced_upper, _, _ = _dyadic_sqrt(
        _one_norm(uncertainty_upper) * _infinity_norm(uncertainty_upper),
        sqrt_precision,
    )
    if induced_upper < frobenius_upper:
        selected_uncertainty = induced_upper
        selected_method = "INDUCED_ONE_INFINITY"
    else:
        selected_uncertainty = frobenius_upper
        selected_method = "FROBENIUS"
    assert frobenius_lower <= frobenius_upper and induced_lower <= induced_upper
    u = tuple(tuple(entry / scale for entry in row) for row in matrix)
    c = parse_qcomplex(center, "center") / scale
    r = radius_q / scale
    node_results = []
    for direction, witness in zip(mesh.directions, witnesses):
        z = c + r * direction
        a0 = tuple(
            tuple((z if i == j else ZERO) - u[i][j] for j in range(n))
            for i in range(n)
        )
        node_results.append(
            _node_certificate(a0, witness, uncertainty_upper, sqrt_precision)
        )
    node_tuple = tuple(node_results)
    chord = r * mesh.maximum_chord_factor_upper
    common = dict(
        mesh=mesh,
        nodes=node_tuple,
        normalized_uncertainty_entry_brackets=uncertainty_brackets,
        uncertainty_entry_self_checks=uncertainty_checks,
        normalized_frobenius_uncertainty_upper=frobenius_upper,
        normalized_induced_uncertainty_upper=induced_upper,
        normalized_selected_uncertainty_upper=selected_uncertainty,
        selected_uncertainty_method=selected_method,
        normalized_chord_upper=chord,
        raw_chord_upper=chord * scale,
        spectral_reference_scale=scale,
        sqrt_precision=sqrt_precision,
        empirical_matrix_provenance_verified=False,
    )
    if any(node.node_sigma_lower is None for node in node_tuple):
        return VerifiedResidualRationalMeshCircle(
            status="VERIFIED_RATIONAL_MESH_RESIDUAL_NODE_CONTRACTION_UNAVAILABLE",
            validation_level=None,
            rank_preserved_for_entire_family=False,
            normalized_robust_delta_lower=None,
            raw_robust_delta_lower=None,
            normalized_robust_resolvent_upper=None,
            raw_robust_resolvent_upper=None,
            projector_perturbation_upper=None,
            **common,
        )
    node_lowers = tuple(node.node_sigma_lower for node in node_tuple)
    delta = min(value for value in node_lowers if value is not None) - chord
    if delta <= 0:
        return VerifiedResidualRationalMeshCircle(
            status="VERIFIED_RATIONAL_MESH_RESIDUAL_FULL_CIRCLE_LOWER_NONPOSITIVE",
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
    projector = r * selected_uncertainty * resolvent * resolvent
    return VerifiedResidualRationalMeshCircle(
        status="VERIFIED_RATIONAL_MESH_COMPONENTWISE_RESIDUAL_CONTOUR_BRIDGE",
        validation_level="VERIFIED_RATIONAL_MESH_COMPONENTWISE_RESIDUAL_CONTOUR_BRIDGE",
        rank_preserved_for_entire_family=True,
        normalized_robust_delta_lower=delta,
        raw_robust_delta_lower=delta * scale,
        normalized_robust_resolvent_upper=resolvent,
        raw_robust_resolvent_upper=resolvent / scale,
        projector_perturbation_upper=projector,
        **common,
    )


__all__ = [
    "ExactMeshResidualWitnessConstruction",
    "RationalUnitCircleMesh",
    "VerifiedResidualRationalMeshCircle",
    "exact_mesh_nominal_inverse_witnesses",
    "verified_componentwise_residual_rational_mesh_circle",
    "verified_rational_unit_circle_mesh",
]
