"""Construct an exact Riesz-projector witness from coprime spectral factors."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_algebraic_riesz_projector import (
        VerifiedAlgebraicRieszProjector,
        verified_algebraic_riesz_projector,
    )
    from .verified_rational_contour import (
        ONE,
        ZERO,
        QComplex,
        _fraction,
        _inverse,
        _matmul,
        _matrix,
        parse_qcomplex,
    )
else:
    from verified_algebraic_riesz_projector import (  # type: ignore[no-redef]
        VerifiedAlgebraicRieszProjector,
        verified_algebraic_riesz_projector,
    )
    from verified_rational_contour import (  # type: ignore[no-redef]
        ONE,
        ZERO,
        QComplex,
        _fraction,
        _inverse,
        _matmul,
        _matrix,
        parse_qcomplex,
    )


QPolynomial = tuple[Fraction, ...]
QMatrix = tuple[tuple[QComplex, ...], ...]


@dataclass(frozen=True)
class VerifiedPolynomialSpectralProjectorConstruction:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    normalized_inside_factor: QPolynomial
    normalized_outside_factor: QPolynomial
    bezout_inside_coefficient: QPolynomial
    bezout_outside_coefficient: QPolynomial
    bezout_identity_verified: bool
    factor_product_annihilates_transition: bool
    constructed_projector: QMatrix | None
    constructed_complement: QMatrix | None
    constructed_exterior_centered_inverse: QMatrix | None
    algebraic_certificate: VerifiedAlgebraicRieszProjector | None
    projector_rank: int | None
    spectral_reference_scale: Fraction
    polynomial_factor_split_supplied: bool
    characteristic_factorization_automatically_discovered: bool
    projector_automatically_constructed: bool
    exterior_inverse_automatically_constructed: bool
    diagonalization_witness_required: bool
    empirical_matrix_provenance_verified: bool


def _trim(polynomial: list[Fraction] | QPolynomial) -> QPolynomial:
    values = list(polynomial)
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return tuple(values or [Fraction(0)])


def _polynomial(value: object, name: str) -> QPolynomial:
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        raise ValueError(f"{name} must contain at least two ascending coefficients")
    result = _trim([_fraction(coefficient, f"{name}[{i}]") for i, coefficient in enumerate(value)])
    if len(result) < 2:
        raise ValueError(f"{name} must be nonconstant")
    leading = result[-1]
    return tuple(coefficient / leading for coefficient in result)


def _poly_add(left: QPolynomial, right: QPolynomial) -> QPolynomial:
    size = max(len(left), len(right))
    return _trim(
        [
            (left[i] if i < len(left) else Fraction(0))
            + (right[i] if i < len(right) else Fraction(0))
            for i in range(size)
        ]
    )


def _poly_neg(value: QPolynomial) -> QPolynomial:
    return tuple(-coefficient for coefficient in value)


def _poly_sub(left: QPolynomial, right: QPolynomial) -> QPolynomial:
    return _poly_add(left, _poly_neg(right))


def _poly_mul(left: QPolynomial, right: QPolynomial) -> QPolynomial:
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return _trim(result)


def _poly_divmod(numerator: QPolynomial, denominator: QPolynomial):
    numerator_list = list(numerator)
    quotient = [Fraction(0)] * max(1, len(numerator) - len(denominator) + 1)
    while len(_trim(numerator_list)) >= len(denominator) and _trim(numerator_list) != (Fraction(0),):
        current = _trim(numerator_list)
        degree = len(current) - len(denominator)
        coefficient = current[-1] / denominator[-1]
        quotient[degree] += coefficient
        for i, value in enumerate(denominator):
            numerator_list[i + degree] -= coefficient * value
        numerator_list = list(_trim(numerator_list))
    return _trim(quotient), _trim(numerator_list)


def _extended_gcd(left: QPolynomial, right: QPolynomial):
    old_r, r = left, right
    old_s, s = (Fraction(1),), (Fraction(0),)
    old_t, t = (Fraction(0),), (Fraction(1),)
    while r != (Fraction(0),):
        quotient, remainder = _poly_divmod(old_r, r)
        old_r, r = r, remainder
        old_s, s = s, _poly_sub(old_s, _poly_mul(quotient, s))
        old_t, t = t, _poly_sub(old_t, _poly_mul(quotient, t))
    leading = old_r[-1]
    return (
        tuple(value / leading for value in old_r),
        tuple(value / leading for value in old_s),
        tuple(value / leading for value in old_t),
    )


def _identity(n: int) -> QMatrix:
    return tuple(tuple(ONE if i == j else ZERO for j in range(n)) for i in range(n))


def _zero(n: int) -> QMatrix:
    return tuple(tuple(ZERO for _ in range(n)) for _ in range(n))


def _matrix_add(left: QMatrix, right: QMatrix) -> QMatrix:
    return tuple(
        tuple(left[i][j] + right[i][j] for j in range(len(left)))
        for i in range(len(left))
    )


def _matrix_sub(left: QMatrix, right: QMatrix) -> QMatrix:
    return tuple(
        tuple(left[i][j] - right[i][j] for j in range(len(left)))
        for i in range(len(left))
    )


def _matrix_scale(matrix: QMatrix, scalar: Fraction) -> QMatrix:
    return tuple(tuple(entry * scalar for entry in row) for row in matrix)


def _evaluate_matrix_polynomial(polynomial: QPolynomial, matrix: QMatrix) -> QMatrix:
    n = len(matrix)
    result = _zero(n)
    identity = _identity(n)
    for coefficient in reversed(polynomial):
        result = _matrix_add(_matmul(result, matrix), _matrix_scale(identity, coefficient))
    return result


def verified_projector_from_coprime_spectral_factors(
    nominal_transition: object,
    *,
    inside_factor: object,
    outside_factor: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    sqrt_precision: int = 32,
) -> VerifiedPolynomialSpectralProjectorConstruction:
    """Build P and the exterior inverse from a supplied coprime factor split."""
    raw = _matrix(nominal_transition, "nominal_transition")
    n = len(raw)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    radius_q = _fraction(radius, "radius")
    if scale <= 0 or radius_q <= 0:
        raise ValueError("radius and spectral_reference_scale must be positive")
    normalized = tuple(tuple(entry / scale for entry in row) for row in raw)
    normalized_center = parse_qcomplex(center, "center") / scale
    inside = _polynomial(inside_factor, "inside_factor")
    outside = _polynomial(outside_factor, "outside_factor")
    gcd, bezout_inside, bezout_outside = _extended_gcd(inside, outside)
    if gcd != (Fraction(1),):
        raise ValueError("inside_factor and outside_factor must be coprime")
    bezout_check = _poly_add(
        _poly_mul(bezout_inside, inside),
        _poly_mul(bezout_outside, outside),
    ) == (Fraction(1),)
    product = _poly_mul(inside, outside)
    annihilation = _evaluate_matrix_polynomial(product, normalized) == _zero(n)
    failures = []
    if not bezout_check:
        failures.append("SPECTRAL_FACTOR_BEZOUT_IDENTITY_FAILED")
    if not annihilation:
        failures.append("SPECTRAL_FACTOR_PRODUCT_DOES_NOT_ANNIHILATE_TRANSITION")
    if failures:
        return VerifiedPolynomialSpectralProjectorConstruction(
            status=failures[0],
            validation_level=None,
            failure_codes=tuple(failures),
            normalized_inside_factor=inside,
            normalized_outside_factor=outside,
            bezout_inside_coefficient=bezout_inside,
            bezout_outside_coefficient=bezout_outside,
            bezout_identity_verified=bezout_check,
            factor_product_annihilates_transition=annihilation,
            constructed_projector=None,
            constructed_complement=None,
            constructed_exterior_centered_inverse=None,
            algebraic_certificate=None,
            projector_rank=None,
            spectral_reference_scale=scale,
            polynomial_factor_split_supplied=True,
            characteristic_factorization_automatically_discovered=False,
            projector_automatically_constructed=False,
            exterior_inverse_automatically_constructed=False,
            diagonalization_witness_required=False,
            empirical_matrix_provenance_verified=False,
        )
    outside_value = _evaluate_matrix_polynomial(outside, normalized)
    bezout_outside_value = _evaluate_matrix_polynomial(bezout_outside, normalized)
    projector = _matmul(bezout_outside_value, outside_value)
    identity = _identity(n)
    complement = _matrix_sub(identity, projector)
    centered = tuple(
        tuple(
            normalized[i][j] - (normalized_center if i == j else ZERO)
            for j in range(n)
        )
        for i in range(n)
    )
    supported_centered = _matmul(complement, _matmul(centered, complement))
    augmented = _matrix_add(projector, supported_centered)
    augmented_inverse = _inverse(augmented)
    if augmented_inverse is None:
        failure = "SPECTRAL_FACTOR_COMPLEMENT_CENTERED_OPERATOR_SINGULAR"
        return VerifiedPolynomialSpectralProjectorConstruction(
            status=failure,
            validation_level=None,
            failure_codes=(failure,),
            normalized_inside_factor=inside,
            normalized_outside_factor=outside,
            bezout_inside_coefficient=bezout_inside,
            bezout_outside_coefficient=bezout_outside,
            bezout_identity_verified=True,
            factor_product_annihilates_transition=True,
            constructed_projector=projector,
            constructed_complement=complement,
            constructed_exterior_centered_inverse=None,
            algebraic_certificate=None,
            projector_rank=None,
            spectral_reference_scale=scale,
            polynomial_factor_split_supplied=True,
            characteristic_factorization_automatically_discovered=False,
            projector_automatically_constructed=True,
            exterior_inverse_automatically_constructed=False,
            diagonalization_witness_required=False,
            empirical_matrix_provenance_verified=False,
        )
    exterior_inverse = _matmul(complement, _matmul(augmented_inverse, complement))
    algebraic = verified_algebraic_riesz_projector(
        nominal_transition,
        projector=projector,
        exterior_centered_inverse=exterior_inverse,
        center=center,
        radius=radius,
        spectral_reference_scale=scale,
        sqrt_precision=sqrt_precision,
    )
    status = (
        "VERIFIED_EXACT_PROJECTOR_CONSTRUCTED_FROM_COPRIME_SPECTRAL_FACTORS"
        if algebraic.validation_level is not None
        else algebraic.status
    )
    return VerifiedPolynomialSpectralProjectorConstruction(
        status=status,
        validation_level=(status if algebraic.validation_level is not None else None),
        failure_codes=algebraic.failure_codes,
        normalized_inside_factor=inside,
        normalized_outside_factor=outside,
        bezout_inside_coefficient=bezout_inside,
        bezout_outside_coefficient=bezout_outside,
        bezout_identity_verified=True,
        factor_product_annihilates_transition=True,
        constructed_projector=projector,
        constructed_complement=complement,
        constructed_exterior_centered_inverse=exterior_inverse,
        algebraic_certificate=algebraic,
        projector_rank=algebraic.projector_rank,
        spectral_reference_scale=scale,
        polynomial_factor_split_supplied=True,
        characteristic_factorization_automatically_discovered=False,
        projector_automatically_constructed=True,
        exterior_inverse_automatically_constructed=True,
        diagonalization_witness_required=False,
        empirical_matrix_provenance_verified=False,
    )


__all__ = [
    "QPolynomial",
    "VerifiedPolynomialSpectralProjectorConstruction",
    "verified_projector_from_coprime_spectral_factors",
]
