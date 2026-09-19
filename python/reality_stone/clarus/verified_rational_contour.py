"""Dependency-free, exact rational certificates for the fixed four-node contour.

The input is deliberately small: matrices over Q(i), a positive rational
reference scale, and (for the strip result) an exact diagonalization witness.
No binary floating point is accepted or produced by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
import re


@dataclass(frozen=True)
class QComplex:
    real: Fraction = Fraction(0)
    imag: Fraction = Fraction(0)

    def __add__(self, other: object) -> "QComplex":
        other_q = parse_qcomplex(other, "operand")
        return QComplex(self.real + other_q.real, self.imag + other_q.imag)

    def __radd__(self, other: object) -> "QComplex":
        return self + other

    def __sub__(self, other: object) -> "QComplex":
        other_q = parse_qcomplex(other, "operand")
        return QComplex(self.real - other_q.real, self.imag - other_q.imag)

    def __rsub__(self, other: object) -> "QComplex":
        return parse_qcomplex(other, "operand") - self

    def __neg__(self) -> "QComplex":
        return QComplex(-self.real, -self.imag)

    def __mul__(self, other: object) -> "QComplex":
        other_q = parse_qcomplex(other, "operand")
        return QComplex(self.real * other_q.real - self.imag * other_q.imag,
                        self.real * other_q.imag + self.imag * other_q.real)

    def __rmul__(self, other: object) -> "QComplex":
        return self * other

    def __truediv__(self, other: object) -> "QComplex":
        other_q = parse_qcomplex(other, "operand")
        denominator = other_q.abs_squared()
        if denominator == 0:
            raise ZeroDivisionError("division by zero in Q(i)")
        return QComplex((self.real * other_q.real + self.imag * other_q.imag) / denominator,
                        (self.imag * other_q.real - self.real * other_q.imag) / denominator)

    def abs_squared(self) -> Fraction:
        return self.real * self.real + self.imag * self.imag

    def is_zero(self) -> bool:
        return self.real == 0 and self.imag == 0


ZERO = QComplex()
ONE = QComplex(Fraction(1))


def _fraction(value: object, name: str) -> Fraction:
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError(f"{name} must be an exact integer, Fraction, or canonical rational string")
    if isinstance(value, Fraction):
        return value
    if type(value) is int:
        return Fraction(value)
    if isinstance(value, str):
        # Canonical finite decimals have one digit before the point unless the
        # integer part is nonzero, and no trailing fractional zero.  Integer
        # and a/b forms must round-trip through Fraction's canonical printer.
        decimal = re.fullmatch(r"-?(?:0|[1-9][0-9]*)\.[0-9]*[1-9]", value)
        integer_or_fraction = re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:/(?:[1-9][0-9]*))?", value)
        if decimal is None and integer_or_fraction is None:
            raise ValueError(f"{name} is not a canonical rational string")
        try:
            parsed = Fraction(value)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(f"{name} is not a canonical rational string") from error
        if parsed == 0 and value.startswith("-"):
            raise ValueError(f"{name} is not a canonical rational string")
        if integer_or_fraction is not None and str(parsed) != value:
            raise ValueError(f"{name} is not a canonical rational string")
        return parsed
    raise ValueError(f"{name} must be an exact integer, Fraction, or canonical rational string")


def parse_qcomplex(value: object, name: str = "value") -> QComplex:
    """Parse one exact Q(i) scalar; Python complex and all floats are rejected."""
    if isinstance(value, QComplex):
        return value
    if isinstance(value, complex):
        raise ValueError(f"{name} must not be a Python complex value")
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} Q(i) pair must have exactly two entries")
        return QComplex(_fraction(value[0], f"{name}.real"), _fraction(value[1], f"{name}.imag"))
    return QComplex(_fraction(value, name))


def _matrix(value: object, name: str) -> tuple[tuple[QComplex, ...], ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{name} must be a nonempty square Q(i) matrix")
    rows = tuple(tuple(parse_qcomplex(entry, f"{name}[{i}][{j}]") for j, entry in enumerate(row))
                 if isinstance(row, (tuple, list)) else () for i, row in enumerate(value))
    n = len(rows)
    if any(len(row) != n for row in rows):
        raise ValueError(f"{name} must be a nonempty square Q(i) matrix")
    return rows


def _identity(n: int) -> tuple[tuple[QComplex, ...], ...]:
    return tuple(tuple(ONE if i == j else ZERO for j in range(n)) for i in range(n))


def _matmul(left: tuple[tuple[QComplex, ...], ...], right: tuple[tuple[QComplex, ...], ...]) -> tuple[tuple[QComplex, ...], ...]:
    n, m, p = len(left), len(right), len(right[0])
    if len(left[0]) != m:
        raise ValueError("matrix dimensions do not agree")
    return tuple(tuple(sum((left[i][k] * right[k][j] for k in range(m)), ZERO) for j in range(p)) for i in range(n))


def _inverse(matrix: tuple[tuple[QComplex, ...], ...]) -> tuple[tuple[QComplex, ...], ...] | None:
    n = len(matrix)
    work = [[*row, *_identity(n)[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next((row for row in range(col, n) if not work[row][col].is_zero()), None)
        if pivot is None:
            return None
        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]
        pivot_value = work[col][col]
        work[col] = [entry / pivot_value for entry in work[col]]
        for row in range(n):
            if row == col:
                continue
            factor = work[row][col]
            if not factor.is_zero():
                work[row] = [entry - factor * pivot_entry for entry, pivot_entry in zip(work[row], work[col])]
    inverse = tuple(tuple(row[n:]) for row in work)
    if _matmul(matrix, inverse) != _identity(n) or _matmul(inverse, matrix) != _identity(n):
        raise AssertionError("exact Gaussian inverse residual check failed")
    return inverse


def _dyadic_sqrt(x: Fraction, precision: int) -> tuple[Fraction, Fraction, bool, bool]:
    if x < 0 or type(precision) is not int or precision < 0:
        raise ValueError("square-root precision must be a nonnegative built-in integer")
    scaled_numerator = x.numerator << (2 * precision)
    t = isqrt(scaled_numerator // x.denominator)
    lower, upper = Fraction(t, 1 << precision), Fraction(t, 1 << precision)
    if lower * lower != x:
        upper = Fraction(t + 1, 1 << precision)
    lower_ok, upper_ok = lower * lower <= x, x <= upper * upper
    if not (lower_ok and upper_ok):
        raise AssertionError("exact dyadic square-root enclosure self-check failed")
    return lower, upper, lower_ok, upper_ok


@dataclass(frozen=True)
class VerifiedRationalCircle:
    status: str
    validation_level: str | None
    projection_p4: tuple[tuple[QComplex, ...], ...] | None
    normalized_node_sigma_lowers: tuple[Fraction, ...]
    raw_node_sigma_lowers: tuple[Fraction, ...]
    normalized_node_sqrt_brackets: tuple[tuple[Fraction, Fraction], ...]
    sqrt_self_checks: tuple[tuple[bool, bool], ...]
    sqrt2_bracket: tuple[Fraction, Fraction] | None
    sqrt2_self_checks: tuple[bool, bool] | None
    chord_factor_bracket: tuple[Fraction, Fraction] | None
    chord_factor_self_checks: tuple[bool, bool] | None
    normalized_chord_upper: Fraction | None
    raw_chord_upper: Fraction | None
    normalized_delta_lower: Fraction | None
    raw_delta_lower: Fraction | None
    normalized_resolvent_upper: Fraction | None
    raw_resolvent_upper: Fraction | None
    spectral_reference_scale: Fraction
    sqrt_precision: int


def verified_rational_circle(
    transition: object, *, center: object, radius: object, spectral_reference_scale: object,
    nodes: int = 4, sqrt_precision: int = 32,
) -> VerifiedRationalCircle:
    """Return V1/V2 certificate data for the fixed N=4 mesh, fail-closed."""
    if type(nodes) is not int or nodes != 4:
        raise ValueError("only the exact four-node mesh (nodes=4) is supported")
    if type(sqrt_precision) is not int or sqrt_precision < 0:
        raise ValueError("square-root precision must be a nonnegative built-in integer")
    matrix = _matrix(transition, "transition")
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    radius_q = _fraction(radius, "radius")
    if scale <= 0 or radius_q <= 0:
        raise ValueError("radius and spectral_reference_scale must be positive")
    center_q = parse_qcomplex(center, "center")
    u = tuple(tuple(entry / scale for entry in row) for row in matrix)
    c, r = center_q / scale, radius_q / scale
    n = len(u)
    directions = (ONE, QComplex(Fraction(0), Fraction(1)), QComplex(Fraction(-1)), QComplex(Fraction(0), Fraction(-1)))
    inverses: list[tuple[tuple[QComplex, ...], ...]] = []
    lowers: list[Fraction] = []
    brackets: list[tuple[Fraction, Fraction]] = []
    checks: list[tuple[bool, bool]] = []
    for direction in directions:
        z = c + r * direction
        a = tuple(tuple((z if i == j else ZERO) - u[i][j] for j in range(n)) for i in range(n))
        inverse = _inverse(a)
        if inverse is None:
            return VerifiedRationalCircle("VERIFIED_SAMPLED_NODE_SINGULAR", None, None, tuple(lowers), tuple(value * scale for value in lowers), tuple(brackets), tuple(checks), None, None, None, None, None, None, None, None, None, None, scale, sqrt_precision)
        squared_frobenius = sum((entry.abs_squared() for row in inverse for entry in row), Fraction(0))
        lo, hi, lo_ok, hi_ok = _dyadic_sqrt(squared_frobenius, sqrt_precision)
        inverses.append(inverse); lowers.append(Fraction(1, 1) / hi); brackets.append((lo, hi)); checks.append((lo_ok, hi_ok))
    s2_lo, s2_hi, s2_lo_ok, s2_hi_ok = _dyadic_sqrt(Fraction(2), sqrt_precision)
    h_lo, h_hi, h_lo_ok, h_hi_ok = _dyadic_sqrt(Fraction(2) - s2_lo, sqrt_precision)
    chord = r * h_hi
    delta = min(lowers) - chord
    projection = tuple(tuple(sum((r * direction * inverse[i][j] for direction, inverse in zip(directions, inverses)), ZERO) / 4 for j in range(n)) for i in range(n))
    if delta <= 0:
        return VerifiedRationalCircle("VERIFIED_LOWER_BOUND_NONPOSITIVE", None, projection, tuple(lowers), tuple(value * scale for value in lowers), tuple(brackets), tuple(checks), (s2_lo, s2_hi), (s2_lo_ok, s2_hi_ok), (h_lo, h_hi), (h_lo_ok, h_hi_ok), chord, chord * scale, delta, delta * scale, None, None, scale, sqrt_precision)
    resolvent = Fraction(1, 1) / delta
    return VerifiedRationalCircle("VERIFIED_RATIONAL_FULL_CIRCLE_ENCLOSURE", "VERIFIED_RATIONAL_FULL_CIRCLE_ENCLOSURE", projection, tuple(lowers), tuple(value * scale for value in lowers), tuple(brackets), tuple(checks), (s2_lo, s2_hi), (s2_lo_ok, s2_hi_ok), (h_lo, h_hi), (h_lo_ok, h_hi_ok), chord, chord * scale, delta, delta * scale, resolvent, resolvent / scale, scale, sqrt_precision)


@dataclass(frozen=True)
class VerifiedRationalStrip:
    status: str
    validation_level: str | None
    central: VerifiedRationalCircle
    inner: VerifiedRationalCircle | None
    outer: VerifiedRationalCircle | None
    exact_projection: tuple[tuple[QComplex, ...], ...] | None
    annulus_status: str
    normalized_m_upper: Fraction | None
    error_upper: Fraction | None
    expansion_factor: Fraction


def verified_rational_analytic_strip(
    transition: object, *, center: object, radius: object, spectral_reference_scale: object,
    expansion_factor: object, eigenvectors: object, eigenvalues: object, nodes: int = 4,
    sqrt_precision: int = 32,
) -> VerifiedRationalStrip:
    """V3 certificate restricted to a supplied exact rational diagonalization."""
    q = _fraction(expansion_factor, "expansion_factor")
    if q <= 1:
        raise ValueError("expansion_factor must be rational and greater than one")
    central = verified_rational_circle(transition, center=center, radius=radius, spectral_reference_scale=spectral_reference_scale, nodes=nodes, sqrt_precision=sqrt_precision)
    matrix, vectors, values = _matrix(transition, "transition"), _matrix(eigenvectors, "eigenvectors"), _matrix(eigenvalues, "eigenvalues")
    n = len(matrix)
    if len(vectors) != n or len(values) != n or any(values[i][j] != ZERO for i in range(n) for j in range(n) if i != j):
        return VerifiedRationalStrip("INVALID_DIAGONALIZATION_WITNESS", None, central, None, None, None, "INVALID_DIAGONALIZATION_WITNESS", None, None, q)
    if _inverse(vectors) is None or _matmul(matrix, vectors) != _matmul(vectors, values):
        return VerifiedRationalStrip("INVALID_DIAGONALIZATION_WITNESS", None, central, None, None, None, "INVALID_DIAGONALIZATION_WITNESS", None, None, q)
    c, r = parse_qcomplex(center, "center"), _fraction(radius, "radius")
    inner_r, outer_r = r / q, r * q
    inside: list[bool] = []
    for i in range(n):
        distance2 = (values[i][i] - c).abs_squared()
        if not (distance2 < inner_r * inner_r or distance2 > outer_r * outer_r):
            return VerifiedRationalStrip("VERIFIED_CLOSED_ANNULUS_NOT_CLEAR", None, central, None, None, None, "VERIFIED_CLOSED_ANNULUS_NOT_CLEAR", None, None, q)
        inside.append(distance2 < r * r)
    inner = verified_rational_circle(transition, center=center, radius=inner_r, spectral_reference_scale=spectral_reference_scale, nodes=nodes, sqrt_precision=sqrt_precision)
    outer = verified_rational_circle(transition, center=center, radius=outer_r, spectral_reference_scale=spectral_reference_scale, nodes=nodes, sqrt_precision=sqrt_precision)
    if central.validation_level is None or inner.validation_level is None or outer.validation_level is None:
        return VerifiedRationalStrip("VERIFIED_BOUNDARY_CERTIFICATE_UNAVAILABLE", None, central, inner, outer, None, "VERIFIED_CLOSED_ANNULUS_CLEAR", None, None, q)
    v_inverse = _inverse(vectors)
    assert v_inverse is not None
    diag = tuple(tuple(ONE if i == j and inside[i] else ZERO for j in range(n)) for i in range(n))
    exact_projection = _matmul(_matmul(vectors, diag), v_inverse)
    assert inner.normalized_delta_lower is not None and outer.normalized_delta_lower is not None
    inner_norm_r = (r / _fraction(spectral_reference_scale, "spectral_reference_scale")) / q
    outer_norm_r = (r / _fraction(spectral_reference_scale, "spectral_reference_scale")) * q
    m_upper = max(inner_norm_r / inner.normalized_delta_lower, outer_norm_r / outer.normalized_delta_lower)
    return VerifiedRationalStrip("VERIFIED_RATIONAL_ANALYTIC_STRIP_ENCLOSURE", "VERIFIED_RATIONAL_ANALYTIC_STRIP_ENCLOSURE", central, inner, outer, exact_projection, "VERIFIED_CLOSED_ANNULUS_CLEAR", m_upper, 2 * m_upper / (q ** 4 - 1), q)


__all__ = ["QComplex", "VerifiedRationalCircle", "VerifiedRationalStrip", "parse_qcomplex", "verified_rational_circle", "verified_rational_analytic_strip"]
