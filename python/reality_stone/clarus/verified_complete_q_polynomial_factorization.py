"""Exact, proof-carrying factorization of one polynomial over Q.

The implementation uses Gauss' lemma and Kronecker's finite evaluation
method.  A successful result is a complete factorization certificate for the
supplied polynomial, not a floating-point root heuristic.  The explicit
candidate budget is a resource guard: exhaustion fails closed and never
produces a completeness claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import gcd, lcm


QPolynomial = tuple[Fraction, ...]
ZPolynomial = tuple[int, ...]


@dataclass(frozen=True)
class KroneckerSearchRecord:
    primitive_polynomial: ZPolynomial
    trial_factor_degree: int
    interpolation_points: tuple[int, ...]
    candidate_value_tuples: int
    candidates_examined: int
    proper_factor_found: ZPolynomial | None
    search_space_exhausted: bool


@dataclass(frozen=True)
class VerifiedQIrreducibleFactor:
    primitive_integer_factor: ZPolynomial
    monic_factor: QPolynomial
    multiplicity: int
    primary_factor: QPolynomial
    irreducibility_verified_by_kronecker: bool


@dataclass(frozen=True)
class VerifiedCompleteQPolynomialFactorization:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    normalized_input: QPolynomial
    primitive_integer_input: ZPolynomial
    factors: tuple[VerifiedQIrreducibleFactor, ...]
    search_records: tuple[KroneckerSearchRecord, ...]
    maximum_candidate_value_tuples: int
    candidate_value_tuples_examined: int
    product_reconstruction_verified: bool
    gauss_lemma_applicable: bool
    kronecker_completeness_verified: bool
    full_arbitrary_degree_q_factorization_verified: bool
    numerical_root_finding_used: bool


class _CandidateBudgetExceeded(Exception):
    pass


def _trim_q(values: list[Fraction] | QPolynomial) -> QPolynomial:
    result = list(values)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return tuple(result or [Fraction(0)])


def _trim_z(values: list[int] | ZPolynomial) -> ZPolynomial:
    result = list(values)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return tuple(result or [0])


def _as_fraction(value: object, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ValueError(f"{name} must be an exact integer or Fraction")
    return Fraction(value)


def _normalize_polynomial(value: object) -> QPolynomial:
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        raise ValueError("polynomial must contain at least two ascending coefficients")
    polynomial = _trim_q(
        [_as_fraction(coefficient, f"polynomial[{i}]") for i, coefficient in enumerate(value)]
    )
    if len(polynomial) < 2:
        raise ValueError("polynomial must be nonconstant")
    leading = polynomial[-1]
    return tuple(coefficient / leading for coefficient in polynomial)


def _primitive_integer(polynomial: QPolynomial) -> ZPolynomial:
    denominator_lcm = 1
    for coefficient in polynomial:
        denominator_lcm = lcm(denominator_lcm, coefficient.denominator)
    integers = [int(coefficient * denominator_lcm) for coefficient in polynomial]
    content = 0
    for coefficient in integers:
        content = gcd(content, abs(coefficient))
    content = max(content, 1)
    integers = [coefficient // content for coefficient in integers]
    if integers[-1] < 0:
        integers = [-coefficient for coefficient in integers]
    return _trim_z(integers)


def _primitive_part(polynomial: ZPolynomial) -> ZPolynomial:
    polynomial = _trim_z(polynomial)
    content = 0
    for coefficient in polynomial:
        content = gcd(content, abs(coefficient))
    if content == 0:
        return (0,)
    result = tuple(coefficient // content for coefficient in polynomial)
    return tuple(-coefficient for coefficient in result) if result[-1] < 0 else result


def _poly_add(left: QPolynomial, right: QPolynomial) -> QPolynomial:
    size = max(len(left), len(right))
    return _trim_q([
        (left[i] if i < len(left) else Fraction(0))
        + (right[i] if i < len(right) else Fraction(0))
        for i in range(size)
    ])


def _poly_mul(left: QPolynomial, right: QPolynomial) -> QPolynomial:
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return _trim_q(result)


def _poly_divmod(numerator: QPolynomial, denominator: QPolynomial) -> tuple[QPolynomial, QPolynomial]:
    remainder = list(numerator)
    quotient = [Fraction(0)] * max(1, len(numerator) - len(denominator) + 1)
    while len(_trim_q(remainder)) >= len(denominator) and _trim_q(remainder) != (Fraction(0),):
        current = _trim_q(remainder)
        degree = len(current) - len(denominator)
        coefficient = current[-1] / denominator[-1]
        quotient[degree] += coefficient
        for i, value in enumerate(denominator):
            remainder[i + degree] -= coefficient * value
        remainder = list(_trim_q(remainder))
    return _trim_q(quotient), _trim_q(remainder)


def _evaluate_integer(polynomial: ZPolynomial, point: int) -> int:
    result = 0
    for coefficient in reversed(polynomial):
        result = result * point + coefficient
    return result


def _positive_divisors(value: int) -> tuple[int, ...]:
    value = abs(value)
    small: list[int] = []
    large: list[int] = []
    divisor = 1
    while divisor * divisor <= value:
        if value % divisor == 0:
            small.append(divisor)
            if divisor * divisor != value:
                large.append(value // divisor)
        divisor += 1
    return tuple(small + list(reversed(large)))


def _signed_divisors(value: int) -> tuple[int, ...]:
    if value == 0:
        raise ValueError("Kronecker points must have nonzero polynomial values")
    positives = _positive_divisors(value)
    return tuple(item for divisor in positives for item in (divisor, -divisor))


def _evaluation_points(polynomial: ZPolynomial, count: int) -> tuple[int, ...]:
    points: list[int] = []
    offset = 0
    while len(points) < count:
        candidates = (0,) if offset == 0 else (offset, -offset)
        for point in candidates:
            if _evaluate_integer(polynomial, point) != 0:
                points.append(point)
                if len(points) == count:
                    break
        offset += 1
    return tuple(points)


def _interpolate(points: tuple[int, ...], values: tuple[int, ...]) -> QPolynomial:
    result: QPolynomial = (Fraction(0),)
    for i, (point_i, value_i) in enumerate(zip(points, values, strict=True)):
        basis: QPolynomial = (Fraction(1),)
        denominator = 1
        for j, point_j in enumerate(points):
            if i == j:
                continue
            basis = _poly_mul(basis, (Fraction(-point_j), Fraction(1)))
            denominator *= point_i - point_j
        scaled = tuple(coefficient * Fraction(value_i, denominator) for coefficient in basis)
        result = _poly_add(result, scaled)
    return _trim_q(result)


def _q_from_z(polynomial: ZPolynomial) -> QPolynomial:
    return tuple(Fraction(coefficient) for coefficient in polynomial)


def _monic(polynomial: ZPolynomial) -> QPolynomial:
    leading = Fraction(polynomial[-1])
    return tuple(Fraction(coefficient) / leading for coefficient in polynomial)


def _proper_factor_search(
    polynomial: ZPolynomial,
    *,
    maximum: int,
    state: dict[str, int],
    records: list[KroneckerSearchRecord],
) -> ZPolynomial | None:
    degree = len(polynomial) - 1
    for trial_degree in range(1, degree // 2 + 1):
        points = _evaluation_points(polynomial, trial_degree + 1)
        divisor_sets = tuple(
            _signed_divisors(_evaluate_integer(polynomial, point)) for point in points
        )
        combinations = 1
        for divisors in divisor_sets:
            combinations *= len(divisors)
        if state["examined"] + combinations > maximum:
            records.append(KroneckerSearchRecord(
                primitive_polynomial=polynomial,
                trial_factor_degree=trial_degree,
                interpolation_points=points,
                candidate_value_tuples=combinations,
                candidates_examined=0,
                proper_factor_found=None,
                search_space_exhausted=False,
            ))
            raise _CandidateBudgetExceeded
        examined_here = 0
        seen: set[ZPolynomial] = set()
        for values in product(*divisor_sets):
            examined_here += 1
            state["examined"] += 1
            interpolated = _interpolate(points, values)
            if len(interpolated) != trial_degree + 1:
                continue
            if any(coefficient.denominator != 1 for coefficient in interpolated):
                continue
            candidate = _primitive_part(tuple(int(coefficient) for coefficient in interpolated))
            if candidate in seen or len(candidate) != trial_degree + 1:
                continue
            seen.add(candidate)
            _, remainder = _poly_divmod(_q_from_z(polynomial), _q_from_z(candidate))
            if remainder != (Fraction(0),):
                continue
            records.append(KroneckerSearchRecord(
                primitive_polynomial=polynomial,
                trial_factor_degree=trial_degree,
                interpolation_points=points,
                candidate_value_tuples=combinations,
                candidates_examined=examined_here,
                proper_factor_found=candidate,
                search_space_exhausted=False,
            ))
            return candidate
        records.append(KroneckerSearchRecord(
            primitive_polynomial=polynomial,
            trial_factor_degree=trial_degree,
            interpolation_points=points,
            candidate_value_tuples=combinations,
            candidates_examined=examined_here,
            proper_factor_found=None,
            search_space_exhausted=True,
        ))
    return None


def _factor_recursive(
    polynomial: ZPolynomial,
    *,
    maximum: int,
    state: dict[str, int],
    records: list[KroneckerSearchRecord],
    cache: dict[ZPolynomial, tuple[ZPolynomial, ...]],
) -> tuple[ZPolynomial, ...]:
    polynomial = _primitive_part(polynomial)
    if polynomial in cache:
        return cache[polynomial]
    if len(polynomial) <= 2:
        result = (polynomial,)
    else:
        factor = _proper_factor_search(
            polynomial, maximum=maximum, state=state, records=records
        )
        if factor is None:
            result = (polynomial,)
        else:
            quotient, remainder = _poly_divmod(_q_from_z(polynomial), _q_from_z(factor))
            if remainder != (Fraction(0),):
                raise AssertionError("accepted Kronecker factor must divide exactly")
            quotient_z = _primitive_integer(quotient)
            result = (
                *_factor_recursive(
                    factor, maximum=maximum, state=state, records=records, cache=cache
                ),
                *_factor_recursive(
                    quotient_z, maximum=maximum, state=state, records=records, cache=cache
                ),
            )
    cache[polynomial] = result
    return result


def verified_complete_q_polynomial_factorization(
    polynomial: object,
    *,
    maximum_candidate_value_tuples: int = 1_000_000,
) -> VerifiedCompleteQPolynomialFactorization:
    """Return a complete exact Q-factorization or an explicit budget refusal."""
    if (
        type(maximum_candidate_value_tuples) is not int
        or maximum_candidate_value_tuples < 1
    ):
        raise ValueError("maximum_candidate_value_tuples must be a positive built-in integer")
    normalized = _normalize_polynomial(polynomial)
    primitive = _primitive_integer(normalized)
    records: list[KroneckerSearchRecord] = []
    state = {"examined": 0}
    try:
        raw_factors = _factor_recursive(
            primitive,
            maximum=maximum_candidate_value_tuples,
            state=state,
            records=records,
            cache={},
        )
    except _CandidateBudgetExceeded:
        failure = "Q_FACTORIZATION_CANDIDATE_BUDGET_EXCEEDED"
        return VerifiedCompleteQPolynomialFactorization(
            status=failure,
            validation_level=None,
            failure_codes=(failure,),
            normalized_input=normalized,
            primitive_integer_input=primitive,
            factors=(),
            search_records=tuple(records),
            maximum_candidate_value_tuples=maximum_candidate_value_tuples,
            candidate_value_tuples_examined=state["examined"],
            product_reconstruction_verified=False,
            gauss_lemma_applicable=True,
            kronecker_completeness_verified=False,
            full_arbitrary_degree_q_factorization_verified=False,
            numerical_root_finding_used=False,
        )

    counts: dict[ZPolynomial, int] = {}
    for factor in raw_factors:
        canonical = _primitive_part(factor)
        counts[canonical] = counts.get(canonical, 0) + 1
    verified_factors: list[VerifiedQIrreducibleFactor] = []
    reconstructed: QPolynomial = (Fraction(1),)
    for factor, multiplicity in sorted(
        counts.items(), key=lambda item: (len(item[0]), item[0])
    ):
        monic = _monic(factor)
        primary: QPolynomial = (Fraction(1),)
        for _ in range(multiplicity):
            primary = _poly_mul(primary, monic)
        reconstructed = _poly_mul(reconstructed, primary)
        verified_factors.append(VerifiedQIrreducibleFactor(
            primitive_integer_factor=factor,
            monic_factor=monic,
            multiplicity=multiplicity,
            primary_factor=primary,
            irreducibility_verified_by_kronecker=True,
        ))
    reconstruction = reconstructed == normalized
    failures = () if reconstruction else ("Q_FACTORIZATION_PRODUCT_RECONSTRUCTION_FAILED",)
    status = (
        "VERIFIED_COMPLETE_Q_POLYNOMIAL_FACTORIZATION"
        if not failures
        else failures[0]
    )
    return VerifiedCompleteQPolynomialFactorization(
        status=status,
        validation_level=None if failures else status,
        failure_codes=failures,
        normalized_input=normalized,
        primitive_integer_input=primitive,
        factors=tuple(verified_factors),
        search_records=tuple(records),
        maximum_candidate_value_tuples=maximum_candidate_value_tuples,
        candidate_value_tuples_examined=state["examined"],
        product_reconstruction_verified=reconstruction,
        gauss_lemma_applicable=True,
        kronecker_completeness_verified=not failures,
        full_arbitrary_degree_q_factorization_verified=not failures,
        numerical_root_finding_used=False,
    )


__all__ = [
    "KroneckerSearchRecord",
    "VerifiedCompleteQPolynomialFactorization",
    "VerifiedQIrreducibleFactor",
    "verified_complete_q_polynomial_factorization",
]
