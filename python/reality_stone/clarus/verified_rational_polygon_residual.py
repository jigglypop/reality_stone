"""Exact residual certification on simple rational polygonal Jordan contours."""

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
class RationalJordanPolygon:
    vertices: tuple[QComplex, ...]
    signed_double_area: Fraction
    edge_squared_lengths: tuple[Fraction, ...]
    edge_length_brackets: tuple[tuple[Fraction, Fraction], ...]
    half_edge_length_uppers: tuple[Fraction, ...]
    perimeter_upper: Fraction
    length_sqrt_self_checks: tuple[tuple[bool, bool], ...]
    nonadjacent_edge_pairs_checked: int
    sqrt_precision: int


@dataclass(frozen=True)
class ExactPolygonResidualWitnessConstruction:
    status: str
    validation_level: str | None
    polygon: RationalJordanPolygon
    normalized_node_matrices: tuple[QMatrix, ...]
    approximate_inverses: tuple[QMatrix, ...] | None
    failing_node_index: int | None
    spectral_reference_scale: Fraction


@dataclass(frozen=True)
class VerifiedResidualRationalPolygon:
    status: str
    validation_level: str | None
    rank_preserved_for_entire_family: bool
    polygon: RationalJordanPolygon
    nodes: tuple[ResidualNodeCertificate, ...]
    edge_robust_delta_lowers: tuple[Fraction, ...] | None
    normalized_uncertainty_entry_brackets: BracketMatrix
    uncertainty_entry_self_checks: CheckMatrix
    normalized_frobenius_uncertainty_upper: Fraction
    normalized_induced_uncertainty_upper: Fraction
    normalized_selected_uncertainty_upper: Fraction
    selected_uncertainty_method: str
    normalized_perimeter_upper: Fraction
    raw_perimeter_upper: Fraction
    normalized_robust_delta_lower: Fraction | None
    raw_robust_delta_lower: Fraction | None
    normalized_robust_resolvent_upper: Fraction | None
    raw_robust_resolvent_upper: Fraction | None
    projector_prefactor_upper: Fraction
    projector_perturbation_upper: Fraction | None
    spectral_reference_scale: Fraction
    sqrt_precision: int
    empirical_matrix_provenance_verified: bool
    nominal_rank_value_supplied: bool


def _cross(a: QComplex, b: QComplex, c: QComplex) -> Fraction:
    return (b.real - a.real) * (c.imag - a.imag) - (
        b.imag - a.imag
    ) * (c.real - a.real)


def _on_segment(a: QComplex, b: QComplex, p: QComplex) -> bool:
    return (
        _cross(a, b, p) == 0
        and min(a.real, b.real) <= p.real <= max(a.real, b.real)
        and min(a.imag, b.imag) <= p.imag <= max(a.imag, b.imag)
    )


def _segments_intersect(
    a: QComplex, b: QComplex, c: QComplex, d: QComplex
) -> bool:
    o1 = _cross(a, b, c)
    o2 = _cross(a, b, d)
    o3 = _cross(c, d, a)
    o4 = _cross(c, d, b)
    if ((o1 > 0 and o2 < 0) or (o1 < 0 and o2 > 0)) and (
        (o3 > 0 and o4 < 0) or (o3 < 0 and o4 > 0)
    ):
        return True
    return (
        (o1 == 0 and _on_segment(a, b, c))
        or (o2 == 0 and _on_segment(a, b, d))
        or (o3 == 0 and _on_segment(c, d, a))
        or (o4 == 0 and _on_segment(c, d, b))
    )


def verified_rational_jordan_polygon(
    vertices: object, *, sqrt_precision: int = 32
) -> RationalJordanPolygon:
    """Verify one simple counterclockwise polygon, allowing forward edge refinement."""
    if type(sqrt_precision) is not int or sqrt_precision < 0:
        raise ValueError("square-root precision must be a nonnegative built-in integer")
    if not isinstance(vertices, (tuple, list)) or len(vertices) < 3:
        raise ValueError("vertices must contain at least three ordered points")
    parsed = tuple(
        parse_qcomplex(value, f"vertices[{index}]")
        for index, value in enumerate(vertices)
    )
    if len(set(parsed)) != len(parsed):
        raise ValueError("polygon vertices must be unique")
    count = len(parsed)
    for index, current in enumerate(parsed):
        nxt = parsed[(index + 1) % count]
        if current == nxt:
            raise ValueError("polygon edges must have positive length")
        following = parsed[(index + 2) % count]
        if _cross(current, nxt, following) == 0:
            first = (nxt.real - current.real, nxt.imag - current.imag)
            second = (following.real - nxt.real, following.imag - nxt.imag)
            if first[0] * second[0] + first[1] * second[1] <= 0:
                raise ValueError("adjacent collinear edges may not reverse or backtrack")

    checked = 0
    for left in range(count):
        a = parsed[left]
        b = parsed[(left + 1) % count]
        for right in range(left + 1, count):
            if right == left + 1 or (left == 0 and right == count - 1):
                continue
            checked += 1
            c = parsed[right]
            d = parsed[(right + 1) % count]
            if _segments_intersect(a, b, c, d):
                raise ValueError("nonadjacent polygon edges must not intersect or touch")

    signed_double_area = sum(
        (
            parsed[index].real * parsed[(index + 1) % count].imag
            - parsed[index].imag * parsed[(index + 1) % count].real
        )
        for index in range(count)
    )
    if signed_double_area <= 0:
        raise ValueError("polygon must be nondegenerate and counterclockwise")

    squared = []
    brackets = []
    checks = []
    half_uppers = []
    for index, left in enumerate(parsed):
        right = parsed[(index + 1) % count]
        dx = right.real - left.real
        dy = right.imag - left.imag
        length_squared = dx * dx + dy * dy
        lower, upper, lower_ok, upper_ok = _dyadic_sqrt(
            length_squared, sqrt_precision
        )
        squared.append(length_squared)
        brackets.append((lower, upper))
        checks.append((lower_ok, upper_ok))
        half_uppers.append(upper / 2)
    return RationalJordanPolygon(
        vertices=parsed,
        signed_double_area=signed_double_area,
        edge_squared_lengths=tuple(squared),
        edge_length_brackets=tuple(brackets),
        half_edge_length_uppers=tuple(half_uppers),
        perimeter_upper=sum((upper for _, upper in brackets), Fraction(0)),
        length_sqrt_self_checks=tuple(checks),
        nonadjacent_edge_pairs_checked=checked,
        sqrt_precision=sqrt_precision,
    )


def exact_polygon_nominal_inverse_witnesses(
    nominal_transition: object,
    *,
    vertices: object,
    spectral_reference_scale: object,
    sqrt_precision: int = 32,
) -> ExactPolygonResidualWitnessConstruction:
    matrix = _matrix(nominal_transition, "nominal_transition")
    n = len(matrix)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    raw_vertices = tuple(
        parse_qcomplex(value, f"vertices[{index}]")
        for index, value in enumerate(vertices)  # type: ignore[arg-type]
    )
    polygon = verified_rational_jordan_polygon(
        tuple(value / scale for value in raw_vertices),
        sqrt_precision=sqrt_precision,
    )
    normalized_matrix = tuple(tuple(entry / scale for entry in row) for row in matrix)
    node_matrices = tuple(
        tuple(
            tuple(
                (vertex if i == j else ZERO) - normalized_matrix[i][j]
                for j in range(n)
            )
            for i in range(n)
        )
        for vertex in polygon.vertices
    )
    inverses = []
    for index, node in enumerate(node_matrices):
        inverse = _inverse(node)
        if inverse is None:
            return ExactPolygonResidualWitnessConstruction(
                status="EXACT_RATIONAL_POLYGON_NOMINAL_NODE_SINGULAR",
                validation_level=None,
                polygon=polygon,
                normalized_node_matrices=node_matrices,
                approximate_inverses=None,
                failing_node_index=index,
                spectral_reference_scale=scale,
            )
        inverses.append(inverse)
    return ExactPolygonResidualWitnessConstruction(
        status="EXACT_RATIONAL_POLYGON_NOMINAL_INVERSES_CONSTRUCTED",
        validation_level="EXACT_RATIONAL_POLYGON_NOMINAL_INVERSES_CONSTRUCTED",
        polygon=polygon,
        normalized_node_matrices=node_matrices,
        approximate_inverses=tuple(inverses),
        failing_node_index=None,
        spectral_reference_scale=scale,
    )


def verified_componentwise_residual_rational_polygon(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    vertices: object,
    approximate_inverses: object,
    spectral_reference_scale: object,
    sqrt_precision: int = 32,
) -> VerifiedResidualRationalPolygon:
    matrix = _matrix(nominal_transition, "nominal_transition")
    n = len(matrix)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    raw_vertices = tuple(
        parse_qcomplex(value, f"vertices[{index}]")
        for index, value in enumerate(vertices)  # type: ignore[arg-type]
    )
    polygon = verified_rational_jordan_polygon(
        tuple(value / scale for value in raw_vertices),
        sqrt_precision=sqrt_precision,
    )
    if not isinstance(approximate_inverses, (tuple, list)) or len(approximate_inverses) != len(polygon.vertices):
        raise ValueError("approximate_inverses must contain one matrix per polygon vertex")
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
    assert frobenius_lower <= frobenius_upper and induced_lower <= induced_upper
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
                    (vertex if i == j else ZERO) - normalized_matrix[i][j]
                    for j in range(n)
                )
                for i in range(n)
            ),
            witness,
            uncertainty_upper,
            sqrt_precision,
        )
        for vertex, witness in zip(polygon.vertices, witnesses)
    )
    projector_prefactor = polygon.perimeter_upper / 6
    common = dict(
        polygon=polygon,
        nodes=node_tuple,
        normalized_uncertainty_entry_brackets=uncertainty_brackets,
        uncertainty_entry_self_checks=uncertainty_checks,
        normalized_frobenius_uncertainty_upper=frobenius_upper,
        normalized_induced_uncertainty_upper=induced_upper,
        normalized_selected_uncertainty_upper=selected_uncertainty,
        selected_uncertainty_method=selected_method,
        normalized_perimeter_upper=polygon.perimeter_upper,
        raw_perimeter_upper=polygon.perimeter_upper * scale,
        projector_prefactor_upper=projector_prefactor,
        spectral_reference_scale=scale,
        sqrt_precision=sqrt_precision,
        empirical_matrix_provenance_verified=False,
        nominal_rank_value_supplied=False,
    )
    if any(node.node_sigma_lower is None for node in node_tuple):
        return VerifiedResidualRationalPolygon(
            status="VERIFIED_RATIONAL_POLYGON_RESIDUAL_NODE_CONTRACTION_UNAVAILABLE",
            validation_level=None,
            rank_preserved_for_entire_family=False,
            edge_robust_delta_lowers=None,
            normalized_robust_delta_lower=None,
            raw_robust_delta_lower=None,
            normalized_robust_resolvent_upper=None,
            raw_robust_resolvent_upper=None,
            projector_perturbation_upper=None,
            **common,
        )
    node_lowers = tuple(node.node_sigma_lower for node in node_tuple)
    edge_deltas = tuple(
        min(
            node_lowers[index],
            node_lowers[(index + 1) % len(node_lowers)],
        )
        - polygon.half_edge_length_uppers[index]
        for index in range(len(node_lowers))
    )
    assert all(value is not None for value in edge_deltas)
    exact_edge_deltas = tuple(value for value in edge_deltas if value is not None)
    delta = min(exact_edge_deltas)
    if delta <= 0:
        return VerifiedResidualRationalPolygon(
            status="VERIFIED_RATIONAL_POLYGON_RESIDUAL_EDGE_COVER_NONPOSITIVE",
            validation_level=None,
            rank_preserved_for_entire_family=False,
            edge_robust_delta_lowers=exact_edge_deltas,
            normalized_robust_delta_lower=delta,
            raw_robust_delta_lower=delta * scale,
            normalized_robust_resolvent_upper=None,
            raw_robust_resolvent_upper=None,
            projector_perturbation_upper=None,
            **common,
        )
    resolvent = Fraction(1) / delta
    projector = projector_prefactor * selected_uncertainty * resolvent * resolvent
    return VerifiedResidualRationalPolygon(
        status="VERIFIED_RATIONAL_POLYGON_COMPONENTWISE_RESIDUAL_CONTOUR_BRIDGE",
        validation_level="VERIFIED_RATIONAL_POLYGON_COMPONENTWISE_RESIDUAL_CONTOUR_BRIDGE",
        rank_preserved_for_entire_family=True,
        edge_robust_delta_lowers=exact_edge_deltas,
        normalized_robust_delta_lower=delta,
        raw_robust_delta_lower=delta * scale,
        normalized_robust_resolvent_upper=resolvent,
        raw_robust_resolvent_upper=resolvent / scale,
        projector_perturbation_upper=projector,
        **common,
    )


__all__ = [
    "ExactPolygonResidualWitnessConstruction",
    "RationalJordanPolygon",
    "VerifiedResidualRationalPolygon",
    "exact_polygon_nominal_inverse_witnesses",
    "verified_componentwise_residual_rational_polygon",
    "verified_rational_jordan_polygon",
]
