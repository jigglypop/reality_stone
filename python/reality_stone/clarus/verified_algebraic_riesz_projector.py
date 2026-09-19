"""Exact algebraic Riesz-projector witness without diagonalization."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_interval_residual import (
        BracketMatrix,
        CheckMatrix,
        RMatrix,
        _infinity_norm,
        _magnitude_enclosures,
        _one_norm,
    )
    from .verified_rational_contour import (
        ONE,
        ZERO,
        QComplex,
        _dyadic_sqrt,
        _fraction,
        _matmul,
        _matrix,
        parse_qcomplex,
    )
else:
    from verified_interval_residual import (  # type: ignore[no-redef]
        BracketMatrix,
        CheckMatrix,
        RMatrix,
        _infinity_norm,
        _magnitude_enclosures,
        _one_norm,
    )
    from verified_rational_contour import (  # type: ignore[no-redef]
        ONE,
        ZERO,
        QComplex,
        _dyadic_sqrt,
        _fraction,
        _matmul,
        _matrix,
        parse_qcomplex,
    )


@dataclass(frozen=True)
class AlgebraicOperatorNormCertificate:
    magnitude_brackets: BracketMatrix
    magnitude_self_checks: CheckMatrix
    magnitude_upper: RMatrix
    frobenius_upper: Fraction
    induced_one_infinity_upper: Fraction
    selected_two_norm_upper: Fraction
    selected_method: str
    sqrt_self_checks: tuple[tuple[bool, bool], tuple[bool, bool]]


@dataclass(frozen=True)
class VerifiedAlgebraicRieszProjector:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    projector_verified: bool
    normalized_transition: tuple[tuple[QComplex, ...], ...]
    normalized_center: QComplex
    normalized_radius: Fraction
    projector: tuple[tuple[QComplex, ...], ...]
    complement: tuple[tuple[QComplex, ...], ...]
    exterior_inverse: tuple[tuple[QComplex, ...], ...]
    idempotence_check: bool
    commutation_checks: tuple[bool, bool]
    exterior_support_checks: tuple[bool, bool, bool]
    exterior_inverse_identity_checks: tuple[bool, bool]
    projector_trace: QComplex
    projector_rank: int | None
    inside_operator_norm: AlgebraicOperatorNormCertificate
    exterior_inverse_norm: AlgebraicOperatorNormCertificate
    normalized_inside_margin: Fraction
    raw_inside_margin: Fraction
    exterior_reciprocal_margin: Fraction
    spectral_reference_scale: Fraction
    sqrt_precision: int
    diagonalization_witness_required: bool
    empirical_matrix_provenance_verified: bool


def _identity(n: int) -> tuple[tuple[QComplex, ...], ...]:
    return tuple(tuple(ONE if i == j else ZERO for j in range(n)) for i in range(n))


def _subtract(left, right):
    return tuple(
        tuple(left[i][j] - right[i][j] for j in range(len(left)))
        for i in range(len(left))
    )


def _operator_norm_certificate(matrix, precision: int) -> AlgebraicOperatorNormCertificate:
    brackets, checks, magnitudes = _magnitude_enclosures(matrix, precision)
    _, frobenius_upper, frobenius_lower_ok, frobenius_upper_ok = _dyadic_sqrt(
        sum((entry.abs_squared() for row in matrix for entry in row), Fraction(0)),
        precision,
    )
    _, induced_upper, induced_lower_ok, induced_upper_ok = _dyadic_sqrt(
        _one_norm(magnitudes) * _infinity_norm(magnitudes), precision
    )
    if induced_upper < frobenius_upper:
        selected = induced_upper
        method = "INDUCED_ONE_INFINITY"
    else:
        selected = frobenius_upper
        method = "FROBENIUS"
    return AlgebraicOperatorNormCertificate(
        magnitude_brackets=brackets,
        magnitude_self_checks=checks,
        magnitude_upper=magnitudes,
        frobenius_upper=frobenius_upper,
        induced_one_infinity_upper=induced_upper,
        selected_two_norm_upper=selected,
        selected_method=method,
        sqrt_self_checks=(
            (frobenius_lower_ok, frobenius_upper_ok),
            (induced_lower_ok, induced_upper_ok),
        ),
    )


def verified_algebraic_riesz_projector(
    nominal_transition: object,
    *,
    projector: object,
    exterior_centered_inverse: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    sqrt_precision: int = 32,
) -> VerifiedAlgebraicRieszProjector:
    """Verify a commuting invariant split with inside/outside spectral norm gates."""
    if type(sqrt_precision) is not int or sqrt_precision < 0:
        raise ValueError("square-root precision must be a nonnegative built-in integer")
    raw_transition = _matrix(nominal_transition, "nominal_transition")
    n = len(raw_transition)
    p = _matrix(projector, "projector")
    exterior_inverse = _matrix(exterior_centered_inverse, "exterior_centered_inverse")
    if len(p) != n or len(exterior_inverse) != n:
        raise ValueError("projector and exterior inverse must shape-match nominal_transition")
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    radius_q = _fraction(radius, "radius")
    if scale <= 0 or radius_q <= 0:
        raise ValueError("radius and spectral_reference_scale must be positive")
    u = tuple(tuple(entry / scale for entry in row) for row in raw_transition)
    c = parse_qcomplex(center, "center") / scale
    r = radius_q / scale
    identity = _identity(n)
    q = _subtract(identity, p)
    centered = tuple(
        tuple(u[i][j] - (c if i == j else ZERO) for j in range(n))
        for i in range(n)
    )
    p_squared = _matmul(p, p)
    pu = _matmul(p, u)
    up = _matmul(u, p)
    rq = _matmul(exterior_inverse, q)
    qr = _matmul(q, exterior_inverse)
    qrq = _matmul(q, rq)
    raq = _matmul(exterior_inverse, _matmul(centered, q))
    ar = _matmul(centered, exterior_inverse)
    trace = sum((p[i][i] for i in range(n)), ZERO)
    rank = None
    if trace.imag == 0 and trace.real.denominator == 1:
        integer_trace = int(trace.real)
        if 0 <= integer_trace <= n:
            rank = integer_trace
    inside_operator = _matmul(centered, p)
    inside_norm = _operator_norm_certificate(inside_operator, sqrt_precision)
    exterior_norm = _operator_norm_certificate(exterior_inverse, sqrt_precision)
    inside_margin = r - inside_norm.selected_two_norm_upper
    reciprocal_margin = 1 - r * exterior_norm.selected_two_norm_upper
    failures = []
    idempotence = p_squared == p
    commutation = (pu == up, _matmul(p, centered) == _matmul(centered, p))
    support = (rq == exterior_inverse, qr == exterior_inverse, qrq == exterior_inverse)
    inverse_checks = (raq == q, ar == q)
    if not idempotence:
        failures.append("ALGEBRAIC_PROJECTOR_NOT_IDEMPOTENT")
    if not all(commutation):
        failures.append("ALGEBRAIC_PROJECTOR_NOT_COMMUTING")
    if rank is None:
        failures.append("ALGEBRAIC_PROJECTOR_TRACE_NOT_AN_INTEGER_RANK")
    if not all(support):
        failures.append("ALGEBRAIC_EXTERIOR_INVERSE_NOT_COMPLEMENT_SUPPORTED")
    if not all(inverse_checks):
        failures.append("ALGEBRAIC_EXTERIOR_INVERSE_IDENTITY_FAILED")
    if inside_margin <= 0:
        failures.append("ALGEBRAIC_INSIDE_NORM_NOT_STRICTLY_INSIDE")
    if reciprocal_margin <= 0:
        failures.append("ALGEBRAIC_EXTERIOR_INVERSE_NORM_NOT_STRICTLY_OUTSIDE")
    status = failures[0] if failures else "VERIFIED_EXACT_ALGEBRAIC_RIESZ_PROJECTOR"
    return VerifiedAlgebraicRieszProjector(
        status=status,
        validation_level=None if failures else status,
        failure_codes=tuple(failures),
        projector_verified=not failures,
        normalized_transition=u,
        normalized_center=c,
        normalized_radius=r,
        projector=p,
        complement=q,
        exterior_inverse=exterior_inverse,
        idempotence_check=idempotence,
        commutation_checks=commutation,
        exterior_support_checks=support,
        exterior_inverse_identity_checks=inverse_checks,
        projector_trace=trace,
        projector_rank=rank,
        inside_operator_norm=inside_norm,
        exterior_inverse_norm=exterior_norm,
        normalized_inside_margin=inside_margin,
        raw_inside_margin=inside_margin * scale,
        exterior_reciprocal_margin=reciprocal_margin,
        spectral_reference_scale=scale,
        sqrt_precision=sqrt_precision,
        diagonalization_witness_required=False,
        empirical_matrix_provenance_verified=False,
    )


__all__ = [
    "AlgebraicOperatorNormCertificate",
    "VerifiedAlgebraicRieszProjector",
    "verified_algebraic_riesz_projector",
]
