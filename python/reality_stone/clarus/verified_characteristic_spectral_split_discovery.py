"""Discover exact spectral splits from a rational characteristic polynomial."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import gcd, lcm

if __package__:
    from .verified_complete_q_polynomial_factorization import (
        VerifiedCompleteQPolynomialFactorization,
        verified_complete_q_polynomial_factorization,
    )
    from .verified_polynomial_spectral_projector_construction import (
        QPolynomial,
        VerifiedPolynomialSpectralProjectorConstruction,
        _evaluate_matrix_polynomial,
        _poly_divmod,
        _poly_mul,
        _trim,
        verified_projector_from_coprime_spectral_factors,
    )
    from .verified_rational_contour import (
        ONE,
        ZERO,
        QComplex,
        _fraction,
        _matmul,
        _matrix,
        parse_qcomplex,
    )
else:
    from verified_complete_q_polynomial_factorization import (  # type: ignore[no-redef]
        VerifiedCompleteQPolynomialFactorization,
        verified_complete_q_polynomial_factorization,
    )
    from verified_polynomial_spectral_projector_construction import (  # type: ignore[no-redef]
        QPolynomial,
        VerifiedPolynomialSpectralProjectorConstruction,
        _evaluate_matrix_polynomial,
        _poly_divmod,
        _poly_mul,
        _trim,
        verified_projector_from_coprime_spectral_factors,
    )
    from verified_rational_contour import (  # type: ignore[no-redef]
        ONE,
        ZERO,
        QComplex,
        _fraction,
        _matmul,
        _matrix,
        parse_qcomplex,
    )


QMatrix = tuple[tuple[QComplex, ...], ...]


@dataclass(frozen=True)
class RationalRootFactor:
    root: Fraction
    multiplicity: int
    factor: QPolynomial


@dataclass(frozen=True)
class VerifiedCharacteristicSpectralSplitDiscovery:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    normalized_transition: QMatrix
    characteristic_polynomial: QPolynomial
    cayley_hamilton_verified: bool
    rational_root_factors: tuple[RationalRootFactor, ...]
    residual_factor: QPolynomial
    residual_degree: int
    residual_irreducible_over_q_verified: bool
    q_factorization_certificate: VerifiedCompleteQPolynomialFactorization
    complete_q_factorization_verified: bool
    atomic_factors: tuple[QPolynomial, ...]
    maximum_partitions: int
    maximum_factor_candidate_value_tuples: int
    factor_candidate_value_tuples_examined: int
    partitions_evaluated: int
    passing_constructions: tuple[VerifiedPolynomialSpectralProjectorConstruction, ...]
    selected_inside_factor: QPolynomial | None
    selected_outside_factor: QPolynomial | None
    selected_construction: VerifiedPolynomialSpectralProjectorConstruction | None
    projector_rank: int | None
    characteristic_polynomial_automatically_constructed: bool
    rational_linear_factors_automatically_discovered: bool
    characteristic_factorization_automatically_discovered: bool
    full_arbitrary_degree_q_factorization_verified: bool
    complex_rational_input_processed: bool
    real_center_conjugation_invariant_contour_verified: bool
    realification_characteristic_envelope_used: bool
    center_shift_to_zero_verified: bool
    characteristic_variable_center: QComplex
    projector_rank_convention: str
    diagonalization_witness_required: bool
    empirical_matrix_provenance_verified: bool


def _identity(n: int) -> QMatrix:
    return tuple(tuple(ONE if i == j else ZERO for j in range(n)) for i in range(n))


def _zero(n: int) -> QMatrix:
    return tuple(tuple(ZERO for _ in range(n)) for _ in range(n))


def _matrix_add(left: QMatrix, right: QMatrix) -> QMatrix:
    return tuple(
        tuple(left[i][j] + right[i][j] for j in range(len(left)))
        for i in range(len(left))
    )


def _matrix_scale(matrix: QMatrix, scalar: Fraction) -> QMatrix:
    return tuple(tuple(entry * scalar for entry in row) for row in matrix)


def _trace(matrix: QMatrix) -> QComplex:
    return sum((matrix[i][i] for i in range(len(matrix))), ZERO)


def _realification(matrix: QMatrix) -> QMatrix:
    """Return [[Re U,-Im U],[Im U,Re U]] over exact real rationals."""
    n = len(matrix)
    return tuple(
        tuple(
            QComplex(
                matrix[i][j].real
                if i < n and j < n
                else -matrix[i][j - n].imag
                if i < n and j >= n
                else matrix[i - n][j].imag
                if i >= n and j < n
                else matrix[i - n][j - n].real,
                Fraction(0),
            )
            for j in range(2 * n)
        )
        for i in range(2 * n)
    )


def _characteristic_polynomial(matrix: QMatrix) -> QPolynomial:
    n = len(matrix)
    identity = _identity(n)
    b = identity
    descending = [Fraction(1)]
    for k in range(1, n + 1):
        ab = _matmul(matrix, b)
        trace = _trace(ab)
        if trace.imag != 0:
            raise ValueError("normalized transition must have a real rational characteristic polynomial")
        coefficient = -trace.real / k
        descending.append(coefficient)
        b = _matrix_add(ab, _matrix_scale(identity, coefficient))
    return tuple(reversed(descending))


def _integer_primitive(polynomial: QPolynomial) -> tuple[int, ...]:
    denominator_lcm = 1
    for coefficient in polynomial:
        denominator_lcm = lcm(denominator_lcm, coefficient.denominator)
    integers = [int(coefficient * denominator_lcm) for coefficient in polynomial]
    divisor = 0
    for value in integers:
        divisor = gcd(divisor, abs(value))
    divisor = max(divisor, 1)
    integers = [value // divisor for value in integers]
    if integers[-1] < 0:
        integers = [-value for value in integers]
    return tuple(integers)


def _positive_divisors(value: int) -> tuple[int, ...]:
    value = abs(value)
    if value == 0:
        return (0,)
    small = []
    large = []
    i = 1
    while i * i <= value:
        if value % i == 0:
            small.append(i)
            if i * i != value:
                large.append(value // i)
        i += 1
    return tuple(small + list(reversed(large)))


def _evaluate_scalar(polynomial: QPolynomial, value: Fraction) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(polynomial):
        result = result * value + coefficient
    return result


def _rational_roots(polynomial: QPolynomial) -> tuple[Fraction, ...]:
    if polynomial[0] == 0:
        return (Fraction(0),)
    integers = _integer_primitive(polynomial)
    candidates = set()
    for numerator in _positive_divisors(integers[0]):
        for denominator in _positive_divisors(integers[-1]):
            if denominator == 0:
                continue
            candidate = Fraction(numerator, denominator)
            candidates.add(candidate)
            candidates.add(-candidate)
    return tuple(
        candidate
        for candidate in sorted(candidates)
        if _evaluate_scalar(polynomial, candidate) == 0
    )


def _factor_rational_roots(polynomial: QPolynomial):
    remainder = polynomial
    factors = []
    while len(remainder) > 1:
        roots = _rational_roots(remainder)
        if not roots:
            break
        root = roots[0]
        linear = (-root, Fraction(1))
        multiplicity = 0
        power = (Fraction(1),)
        while True:
            quotient, residual = _poly_divmod(remainder, linear)
            if residual != (Fraction(0),):
                break
            remainder = quotient
            multiplicity += 1
            power = _poly_mul(power, linear)
            if len(remainder) == 1:
                break
        factors.append(RationalRootFactor(root, multiplicity, power))
    leading = remainder[-1]
    remainder = tuple(value / leading for value in remainder)
    return tuple(factors), remainder


def _product(polynomials: tuple[QPolynomial, ...]) -> QPolynomial:
    result = (Fraction(1),)
    for polynomial in polynomials:
        result = _poly_mul(result, polynomial)
    return result


def _dummy_coprime_linear(characteristic: QPolynomial) -> QPolynomial:
    candidates = [0]
    for value in range(1, 1000):
        candidates.extend((value, -value))
    for value in candidates:
        if _evaluate_scalar(characteristic, Fraction(value)) != 0:
            return (Fraction(-value), Fraction(1))
    raise ValueError("could not construct a bounded dummy coprime linear factor")


def verified_characteristic_spectral_split_discovery(
    nominal_transition: object,
    *,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    maximum_partitions: int = 1024,
    maximum_factor_candidate_value_tuples: int = 1_000_000,
    sqrt_precision: int = 32,
) -> VerifiedCharacteristicSpectralSplitDiscovery:
    """Construct charpoly, factor rational atoms, and classify a unique split."""
    if type(maximum_partitions) is not int or maximum_partitions < 2:
        raise ValueError("maximum_partitions must be a built-in integer at least two")
    raw = _matrix(nominal_transition, "nominal_transition")
    complex_input = any(entry.imag != 0 for row in raw for entry in row)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    normalized = tuple(tuple(entry / scale for entry in row) for row in raw)
    raw_center = parse_qcomplex(center, "center")
    normalized_center = raw_center / scale
    envelope_required = complex_input or normalized_center.imag != 0
    working_raw = (
        tuple(
            tuple(entry - (raw_center if i == j else ZERO) for j, entry in enumerate(row))
            for i, row in enumerate(raw)
        )
        if envelope_required
        else raw
    )
    working_normalized = (
        tuple(
            tuple(
                entry - (normalized_center if i == j else ZERO)
                for j, entry in enumerate(row)
            )
            for i, row in enumerate(normalized)
        )
        if envelope_required
        else normalized
    )
    working_center = ZERO if envelope_required else normalized_center
    characteristic_source = (
        _realification(working_normalized) if envelope_required else working_normalized
    )
    characteristic = _characteristic_polynomial(characteristic_source)
    cayley_hamilton = (
        _evaluate_matrix_polynomial(characteristic, working_normalized) == _zero(len(raw))
    )
    root_factors, residual = _factor_rational_roots(characteristic)
    residual_degree = len(residual) - 1
    residual_irreducible = (
        residual_degree in (2, 3) and not _rational_roots(residual)
    )
    q_factorization = verified_complete_q_polynomial_factorization(
        characteristic,
        maximum_candidate_value_tuples=maximum_factor_candidate_value_tuples,
    )
    complete_factorization = q_factorization.validation_level is not None
    atoms = tuple(item.primary_factor for item in q_factorization.factors)
    if not complete_factorization:
        failure = "CHARACTERISTIC_Q_FACTORIZATION_BUDGET_EXCEEDED"
        return VerifiedCharacteristicSpectralSplitDiscovery(
            status=failure,
            validation_level=None,
            failure_codes=(failure,),
            normalized_transition=normalized,
            characteristic_polynomial=characteristic,
            cayley_hamilton_verified=cayley_hamilton,
            rational_root_factors=root_factors,
            residual_factor=residual,
            residual_degree=residual_degree,
            residual_irreducible_over_q_verified=residual_irreducible,
            q_factorization_certificate=q_factorization,
            complete_q_factorization_verified=False,
            atomic_factors=(),
            maximum_partitions=maximum_partitions,
            maximum_factor_candidate_value_tuples=maximum_factor_candidate_value_tuples,
            factor_candidate_value_tuples_examined=q_factorization.candidate_value_tuples_examined,
            partitions_evaluated=0,
            passing_constructions=(),
            selected_inside_factor=None,
            selected_outside_factor=None,
            selected_construction=None,
            projector_rank=None,
            characteristic_polynomial_automatically_constructed=True,
            rational_linear_factors_automatically_discovered=True,
            characteristic_factorization_automatically_discovered=False,
            full_arbitrary_degree_q_factorization_verified=False,
            complex_rational_input_processed=complex_input,
            real_center_conjugation_invariant_contour_verified=working_center.imag == 0,
            realification_characteristic_envelope_used=envelope_required,
            center_shift_to_zero_verified=envelope_required,
            characteristic_variable_center=working_center,
            projector_rank_convention="COMPLEX_RANK_OF_ORIGINAL_MATRIX",
            diagonalization_witness_required=False,
            empirical_matrix_provenance_verified=False,
        )
    partition_count = 2 ** len(atoms)
    if partition_count > maximum_partitions:
        failure = "CHARACTERISTIC_FACTOR_PARTITION_BUDGET_EXCEEDED"
        return VerifiedCharacteristicSpectralSplitDiscovery(
            status=failure,
            validation_level=None,
            failure_codes=(failure,),
            normalized_transition=normalized,
            characteristic_polynomial=characteristic,
            cayley_hamilton_verified=cayley_hamilton,
            rational_root_factors=root_factors,
            residual_factor=residual,
            residual_degree=residual_degree,
            residual_irreducible_over_q_verified=residual_irreducible,
            q_factorization_certificate=q_factorization,
            complete_q_factorization_verified=complete_factorization,
            atomic_factors=atoms,
            maximum_partitions=maximum_partitions,
            maximum_factor_candidate_value_tuples=maximum_factor_candidate_value_tuples,
            factor_candidate_value_tuples_examined=q_factorization.candidate_value_tuples_examined,
            partitions_evaluated=0,
            passing_constructions=(),
            selected_inside_factor=None,
            selected_outside_factor=None,
            selected_construction=None,
            projector_rank=None,
            characteristic_polynomial_automatically_constructed=True,
            rational_linear_factors_automatically_discovered=True,
            characteristic_factorization_automatically_discovered=complete_factorization,
            full_arbitrary_degree_q_factorization_verified=complete_factorization,
            complex_rational_input_processed=complex_input,
            real_center_conjugation_invariant_contour_verified=working_center.imag == 0,
            realification_characteristic_envelope_used=envelope_required,
            center_shift_to_zero_verified=envelope_required,
            characteristic_variable_center=working_center,
            projector_rank_convention="COMPLEX_RANK_OF_ORIGINAL_MATRIX",
            diagonalization_witness_required=False,
            empirical_matrix_provenance_verified=False,
        )
    dummy = _dummy_coprime_linear(characteristic)
    passing = []
    passing_factors = []
    evaluated = 0
    indices = tuple(range(len(atoms)))
    for size in range(len(atoms) + 1):
        for inside_indices in combinations(indices, size):
            inside_set = set(inside_indices)
            inside_atoms = tuple(atoms[i] for i in indices if i in inside_set)
            outside_atoms = tuple(atoms[i] for i in indices if i not in inside_set)
            inside_factor = _product(inside_atoms) if inside_atoms else dummy
            outside_factor = _product(outside_atoms) if outside_atoms else dummy
            evaluated += 1
            try:
                construction = verified_projector_from_coprime_spectral_factors(
                    working_raw,
                    inside_factor=inside_factor,
                    outside_factor=outside_factor,
                    center=ZERO if envelope_required else center,
                    radius=radius,
                    spectral_reference_scale=scale,
                    sqrt_precision=sqrt_precision,
                )
            except ValueError:
                continue
            if construction.validation_level is not None:
                passing.append(construction)
                passing_factors.append((inside_factor, outside_factor))
    failures = []
    if not cayley_hamilton:
        failures.append("CHARACTERISTIC_CAYLEY_HAMILTON_CHECK_FAILED")
    if len(passing) == 0:
        failures.append("CHARACTERISTIC_NO_CERTIFIED_INSIDE_OUTSIDE_PARTITION")
    elif len(passing) > 1:
        failures.append("CHARACTERISTIC_INSIDE_OUTSIDE_PARTITION_NOT_UNIQUE")
    selected = passing[0] if len(passing) == 1 else None
    selected_factors = passing_factors[0] if len(passing_factors) == 1 else (None, None)
    status = (
        failures[0]
        if failures
        else "VERIFIED_CHARACTERISTIC_SPECTRAL_SPLIT_AND_PROJECTOR_DISCOVERED"
    )
    return VerifiedCharacteristicSpectralSplitDiscovery(
        status=status,
        validation_level=None if failures else status,
        failure_codes=tuple(failures),
        normalized_transition=normalized,
        characteristic_polynomial=characteristic,
        cayley_hamilton_verified=cayley_hamilton,
        rational_root_factors=root_factors,
        residual_factor=residual,
        residual_degree=residual_degree,
        residual_irreducible_over_q_verified=residual_irreducible,
        q_factorization_certificate=q_factorization,
        complete_q_factorization_verified=complete_factorization,
        atomic_factors=atoms,
        maximum_partitions=maximum_partitions,
        maximum_factor_candidate_value_tuples=maximum_factor_candidate_value_tuples,
        factor_candidate_value_tuples_examined=q_factorization.candidate_value_tuples_examined,
        partitions_evaluated=evaluated,
        passing_constructions=tuple(passing),
        selected_inside_factor=selected_factors[0],
        selected_outside_factor=selected_factors[1],
        selected_construction=selected,
        projector_rank=None if selected is None else selected.projector_rank,
        characteristic_polynomial_automatically_constructed=True,
        rational_linear_factors_automatically_discovered=True,
        characteristic_factorization_automatically_discovered=complete_factorization,
        full_arbitrary_degree_q_factorization_verified=complete_factorization,
        complex_rational_input_processed=complex_input,
        real_center_conjugation_invariant_contour_verified=working_center.imag == 0,
        realification_characteristic_envelope_used=envelope_required,
        center_shift_to_zero_verified=envelope_required,
        characteristic_variable_center=working_center,
        projector_rank_convention="COMPLEX_RANK_OF_ORIGINAL_MATRIX",
        diagonalization_witness_required=False,
        empirical_matrix_provenance_verified=False,
    )


__all__ = [
    "RationalRootFactor",
    "VerifiedCharacteristicSpectralSplitDiscovery",
    "verified_characteristic_spectral_split_discovery",
]
