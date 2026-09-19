"""Verified midpoint Riesz-rank quadrature on rational polygonal contours."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_rational_polygon_residual import (
        VerifiedResidualRationalPolygon,
        exact_polygon_nominal_inverse_witnesses,
        verified_componentwise_residual_rational_polygon,
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
    from verified_rational_polygon_residual import (  # type: ignore[no-redef]
        VerifiedResidualRationalPolygon,
        exact_polygon_nominal_inverse_witnesses,
        verified_componentwise_residual_rational_polygon,
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


QMatrix = tuple[tuple[QComplex, ...], ...]


@dataclass(frozen=True)
class RationalPiBracket:
    lower: Fraction
    upper: Fraction
    machin_terms: int
    lower_positive: bool
    ordered: bool
    archimedean_sanity: bool


@dataclass(frozen=True)
class VerifiedPolygonRieszQuadrature:
    status: str
    validation_level: str | None
    polygon_certificate: VerifiedResidualRationalPolygon
    subdivisions_per_edge: tuple[int, ...]
    midpoint_count: int
    normalized_midpoint_nodes: tuple[QComplex, ...]
    midpoint_inverse_identity_checks: tuple[tuple[bool, bool], ...]
    unscaled_projector_quadrature: QMatrix
    unscaled_trace_quadrature: QComplex
    inverse_two_pi_bracket: tuple[Fraction, Fraction]
    inverse_two_pi_midpoint: Fraction
    scaled_projector_quadrature: QMatrix
    unscaled_quadrature_norm_upper: Fraction
    edge_quadrature_error_uppers: tuple[Fraction, ...]
    unscaled_operator_quadrature_error_upper: Fraction
    scaled_projector_operator_error_upper: Fraction
    unscaled_trace_quadrature_error_upper: Fraction
    pi_bracket: RationalPiBracket
    possible_ranks: tuple[int, ...]
    certified_nominal_rank: int | None
    nominal_rank_verified: bool
    family_rank_verified: bool
    empirical_matrix_provenance_verified: bool
    adaptive_subdivision_used: bool


def _alternating_arctan_bracket(x: Fraction, terms: int) -> tuple[Fraction, Fraction]:
    if type(terms) is not int or terms < 1:
        raise ValueError("Machin terms must be a positive built-in integer")

    def partial(count: int) -> Fraction:
        return sum(
            ((-1 if k % 2 else 1) * x ** (2 * k + 1) / (2 * k + 1) for k in range(count)),
            Fraction(0),
        )

    first = partial(terms)
    second = partial(terms + 1)
    return min(first, second), max(first, second)


def rational_machin_pi_bracket(*, terms: int = 8) -> RationalPiBracket:
    """Enclose pi using pi/4 = 4 atan(1/5) - atan(1/239)."""
    atan5_lower, atan5_upper = _alternating_arctan_bracket(Fraction(1, 5), terms)
    atan239_lower, atan239_upper = _alternating_arctan_bracket(
        Fraction(1, 239), terms
    )
    lower = 4 * (4 * atan5_lower - atan239_upper)
    upper = 4 * (4 * atan5_upper - atan239_lower)
    return RationalPiBracket(
        lower=lower,
        upper=upper,
        machin_terms=terms,
        lower_positive=lower > 0,
        ordered=lower < upper,
        archimedean_sanity=Fraction(3) < lower < upper < Fraction(22, 7),
    )


def _identity(n: int) -> QMatrix:
    return tuple(
        tuple(QComplex(1) if i == j else ZERO for j in range(n))
        for i in range(n)
    )


def _matmul(left: QMatrix, right: QMatrix) -> QMatrix:
    n = len(left)
    return tuple(
        tuple(
            sum((left[i][k] * right[k][j] for k in range(n)), ZERO)
            for j in range(n)
        )
        for i in range(n)
    )


def _parse_subdivisions(value: object, edge_count: int) -> tuple[int, ...]:
    if type(value) is int:
        subdivisions = (value,) * edge_count
    elif isinstance(value, (tuple, list)) and len(value) == edge_count:
        subdivisions = tuple(value)
    else:
        raise ValueError("subdivisions must be one integer or one integer per edge")
    if any(type(item) is not int or item <= 0 for item in subdivisions):
        raise ValueError("every edge subdivision must be a positive built-in integer")
    return subdivisions


def verified_polygon_midpoint_riesz_rank(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    vertices: object,
    spectral_reference_scale: object,
    subdivisions: object,
    sqrt_precision: int = 32,
    machin_terms: int = 8,
) -> VerifiedPolygonRieszQuadrature:
    """Verify a nominal and family Riesz rank using polygon midpoint quadrature."""
    matrix = _matrix(nominal_transition, "nominal_transition")
    n = len(matrix)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    raw_vertices = tuple(
        parse_qcomplex(value, f"vertices[{index}]")
        for index, value in enumerate(vertices)  # type: ignore[arg-type]
    )
    construction = exact_polygon_nominal_inverse_witnesses(
        matrix,
        vertices=raw_vertices,
        spectral_reference_scale=scale,
        sqrt_precision=sqrt_precision,
    )
    if construction.approximate_inverses is None:
        raise ValueError("nominal polygon has a singular vertex node")
    polygon_certificate = verified_componentwise_residual_rational_polygon(
        matrix,
        uncertainty_radii=uncertainty_radii,
        vertices=raw_vertices,
        approximate_inverses=construction.approximate_inverses,
        spectral_reference_scale=scale,
        sqrt_precision=sqrt_precision,
    )
    if polygon_certificate.validation_level is None:
        raise ValueError("nominal polygon residual edge cover is unavailable")
    assert polygon_certificate.edge_robust_delta_lowers is not None
    polygon = polygon_certificate.polygon
    per_edge = _parse_subdivisions(subdivisions, len(polygon.vertices))
    normalized_matrix = tuple(tuple(entry / scale for entry in row) for row in matrix)
    identity = _identity(n)
    minus_i = QComplex(0, -1)
    accumulator = [[ZERO for _ in range(n)] for _ in range(n)]
    midpoint_nodes = []
    inverse_checks = []
    for edge, left in enumerate(polygon.vertices):
        right = polygon.vertices[(edge + 1) % len(polygon.vertices)]
        count = per_edge[edge]
        step = (right - left) / count
        for index in range(count):
            midpoint = left + (right - left) * Fraction(2 * index + 1, 2 * count)
            node = tuple(
                tuple(
                    (midpoint if i == j else ZERO) - normalized_matrix[i][j]
                    for j in range(n)
                )
                for i in range(n)
            )
            inverse = _inverse(node)
            if inverse is None:
                raise AssertionError("a certified polygon edge cannot have a singular midpoint")
            checks = (_matmul(inverse, node) == identity, _matmul(node, inverse) == identity)
            if not all(checks):
                raise AssertionError("exact midpoint inverse failed a two-sided identity check")
            weight = minus_i * step
            for i in range(n):
                for j in range(n):
                    accumulator[i][j] = accumulator[i][j] + weight * inverse[i][j]
            midpoint_nodes.append(midpoint)
            inverse_checks.append(checks)
    quadrature = tuple(tuple(row) for row in accumulator)
    trace = sum((quadrature[i][i] for i in range(n)), ZERO)
    edge_errors = tuple(
        polygon.edge_length_brackets[edge][1] ** 3
        / (
            12
            * per_edge[edge] ** 2
            * polygon_certificate.edge_robust_delta_lowers[edge] ** 3
        )
        for edge in range(len(polygon.vertices))
    )
    operator_error = sum(edge_errors, Fraction(0))
    trace_error = n * operator_error
    pi_bracket = rational_machin_pi_bracket(terms=machin_terms)
    if not (
        pi_bracket.lower_positive
        and pi_bracket.ordered
        and pi_bracket.archimedean_sanity
    ):
        raise AssertionError("rational pi enclosure failed its self-checks")
    inverse_two_pi_lower = Fraction(1) / (2 * pi_bracket.upper)
    inverse_two_pi_upper = Fraction(1) / (2 * pi_bracket.lower)
    inverse_two_pi_midpoint = (inverse_two_pi_lower + inverse_two_pi_upper) / 2
    inverse_two_pi_radius = (inverse_two_pi_upper - inverse_two_pi_lower) / 2
    scaled_quadrature = tuple(
        tuple(inverse_two_pi_midpoint * entry for entry in row)
        for row in quadrature
    )
    _, quadrature_norm_upper, quadrature_norm_lower_ok, quadrature_norm_upper_ok = _dyadic_sqrt(
        sum((entry.abs_squared() for row in quadrature for entry in row), Fraction(0)),
        sqrt_precision,
    )
    if not (quadrature_norm_lower_ok and quadrature_norm_upper_ok):
        raise AssertionError("quadrature Frobenius enclosure self-check failed")
    scaled_projector_error = (
        inverse_two_pi_upper * operator_error
        + inverse_two_pi_radius * quadrature_norm_upper
    )
    possible = []
    for rank in range(n + 1):
        lower = 2 * pi_bracket.lower * rank
        upper = 2 * pi_bracket.upper * rank
        if trace.real < lower:
            horizontal = lower - trace.real
        elif trace.real > upper:
            horizontal = trace.real - upper
        else:
            horizontal = Fraction(0)
        distance_squared = horizontal * horizontal + trace.imag * trace.imag
        if distance_squared <= trace_error * trace_error:
            possible.append(rank)
    if not possible:
        raise AssertionError("outward quadrature bounds excluded every possible rank")
    certified = possible[0] if len(possible) == 1 else None
    verified = certified is not None
    return VerifiedPolygonRieszQuadrature(
        status=(
            "VERIFIED_RATIONAL_POLYGON_MIDPOINT_RIESZ_RANK"
            if verified
            else "RATIONAL_POLYGON_MIDPOINT_RIESZ_RANK_UNRESOLVED"
        ),
        validation_level=(
            "VERIFIED_RATIONAL_POLYGON_MIDPOINT_RIESZ_RANK" if verified else None
        ),
        polygon_certificate=polygon_certificate,
        subdivisions_per_edge=per_edge,
        midpoint_count=sum(per_edge),
        normalized_midpoint_nodes=tuple(midpoint_nodes),
        midpoint_inverse_identity_checks=tuple(inverse_checks),
        unscaled_projector_quadrature=quadrature,
        unscaled_trace_quadrature=trace,
        inverse_two_pi_bracket=(inverse_two_pi_lower, inverse_two_pi_upper),
        inverse_two_pi_midpoint=inverse_two_pi_midpoint,
        scaled_projector_quadrature=scaled_quadrature,
        unscaled_quadrature_norm_upper=quadrature_norm_upper,
        edge_quadrature_error_uppers=edge_errors,
        unscaled_operator_quadrature_error_upper=operator_error,
        scaled_projector_operator_error_upper=scaled_projector_error,
        unscaled_trace_quadrature_error_upper=trace_error,
        pi_bracket=pi_bracket,
        possible_ranks=tuple(possible),
        certified_nominal_rank=certified,
        nominal_rank_verified=verified,
        family_rank_verified=verified,
        empirical_matrix_provenance_verified=False,
        adaptive_subdivision_used=False,
    )


__all__ = [
    "RationalPiBracket",
    "VerifiedPolygonRieszQuadrature",
    "rational_machin_pi_bracket",
    "verified_polygon_midpoint_riesz_rank",
]
