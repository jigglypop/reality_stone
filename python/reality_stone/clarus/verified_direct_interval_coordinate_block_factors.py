"""Direct interval characteristic-factor coefficients for invariant coordinate blocks."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
from math import factorial

if __package__:
    from .verified_interval_characteristic_spectral_split import (
        VerifiedIntervalCharacteristicSpectralSplit,
        verified_interval_characteristic_spectral_split,
    )
    from .verified_rational_contour import ONE, ZERO, QComplex, _fraction, _matrix
else:
    from verified_interval_characteristic_spectral_split import (  # type: ignore[no-redef]
        VerifiedIntervalCharacteristicSpectralSplit,
        verified_interval_characteristic_spectral_split,
    )
    from verified_rational_contour import ONE, ZERO, QComplex, _fraction, _matrix  # type: ignore[no-redef]


@dataclass(frozen=True)
class QComplexRectangle:
    real_lower: Fraction
    real_upper: Fraction
    imag_lower: Fraction
    imag_upper: Fraction


@dataclass(frozen=True)
class VerifiedDirectIntervalCoordinateBlockFactors:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    interval_spectral_split: VerifiedIntervalCharacteristicSpectralSplit
    coordinate_inside_indices: tuple[int, ...]
    coordinate_outside_indices: tuple[int, ...]
    coordinate_projector_matches_nominal_discovery: bool
    nominal_and_uncertainty_block_diagonal_verified: bool
    determinant_terms_required: int
    maximum_determinant_terms: int
    normalized_inside_factor_coefficient_boxes: tuple[QComplexRectangle, ...]
    normalized_outside_factor_coefficient_boxes: tuple[QComplexRectangle, ...]
    normalized_factor_product_coefficient_boxes: tuple[QComplexRectangle, ...]
    normalized_direct_full_characteristic_coefficient_boxes: tuple[QComplexRectangle, ...]
    product_and_direct_coefficient_intersections_verified: bool
    inside_factor_degree: int
    outside_factor_degree: int
    factor_degrees_match_family_rank: bool
    monic_factor_coefficients_verified: bool
    direct_interval_characteristic_factor_output_verified: bool
    interval_root_tracking_required: bool
    empirical_uncertainty_provenance_verified: bool


def _real_mul(left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]):
    values = tuple(a * b for a in left for b in right)
    return min(values), max(values)


def _real_add(left, right):
    return left[0] + right[0], left[1] + right[1]


def _real_sub(left, right):
    return left[0] - right[1], left[1] - right[0]


def _cadd(a: QComplexRectangle, b: QComplexRectangle) -> QComplexRectangle:
    return QComplexRectangle(a.real_lower + b.real_lower, a.real_upper + b.real_upper, a.imag_lower + b.imag_lower, a.imag_upper + b.imag_upper)


def _cneg(a: QComplexRectangle) -> QComplexRectangle:
    return QComplexRectangle(-a.real_upper, -a.real_lower, -a.imag_upper, -a.imag_lower)


def _cmul(a: QComplexRectangle, b: QComplexRectangle) -> QComplexRectangle:
    ar, ai = (a.real_lower, a.real_upper), (a.imag_lower, a.imag_upper)
    br, bi = (b.real_lower, b.real_upper), (b.imag_lower, b.imag_upper)
    real = _real_sub(_real_mul(ar, br), _real_mul(ai, bi))
    imag = _real_add(_real_mul(ar, bi), _real_mul(ai, br))
    return QComplexRectangle(real[0], real[1], imag[0], imag[1])


ZERO_BOX = QComplexRectangle(Fraction(0), Fraction(0), Fraction(0), Fraction(0))
ONE_BOX = QComplexRectangle(Fraction(1), Fraction(1), Fraction(0), Fraction(0))


def _padd(a, b):
    size = max(len(a), len(b))
    return tuple(_cadd(a[i] if i < len(a) else ZERO_BOX, b[i] if i < len(b) else ZERO_BOX) for i in range(size))


def _pmul(a, b):
    result = [ZERO_BOX] * (len(a) + len(b) - 1)
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            result[i + j] = _cadd(result[i + j], _cmul(left, right))
    return tuple(result)


def _parity(permutation: tuple[int, ...]) -> int:
    inversions = sum(1 for i in range(len(permutation)) for j in range(i + 1, len(permutation)) if permutation[i] > permutation[j])
    return -1 if inversions % 2 else 1


def _characteristic_boxes(entry_boxes, indices: tuple[int, ...]):
    size = len(indices)
    if size == 0:
        return (ONE_BOX,)
    total = (ZERO_BOX,)
    for perm in permutations(range(size)):
        term = (ONE_BOX,)
        for local_i, local_j in enumerate(perm):
            entry = entry_boxes[indices[local_i]][indices[local_j]]
            constant = _cneg(entry)
            polynomial = (constant, ONE_BOX) if local_i == local_j else (constant,)
            term = _pmul(term, polynomial)
        if _parity(perm) < 0:
            term = tuple(_cneg(value) for value in term)
        total = _padd(total, term)
    return total


def _contains_zero_intersection(a: QComplexRectangle, b: QComplexRectangle) -> bool:
    return max(a.real_lower, b.real_lower) <= min(a.real_upper, b.real_upper) and max(a.imag_lower, b.imag_lower) <= min(a.imag_upper, b.imag_upper)


def verified_direct_interval_coordinate_block_factors(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    coordinate_inside_indices: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    maximum_determinant_terms: int = 100_000,
    maximum_partitions: int = 1024,
    maximum_factor_candidate_value_tuples: int = 1_000_000,
    nodes: int = 4,
    sqrt_precision: int = 48,
) -> VerifiedDirectIntervalCoordinateBlockFactors:
    matrix = _matrix(nominal_transition, "nominal_transition")
    n = len(matrix)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    if not isinstance(coordinate_inside_indices, (tuple, list)):
        raise ValueError("coordinate_inside_indices must be a sequence")
    if any(type(index) is not int for index in coordinate_inside_indices):
        raise ValueError("coordinate indices must be built-in integers")
    inside = tuple(sorted(coordinate_inside_indices))
    if len(set(inside)) != len(inside) or any(index < 0 or index >= n for index in inside):
        raise ValueError("coordinate inside indices must be unique and in range")
    outside = tuple(index for index in range(n) if index not in set(inside))
    if type(maximum_determinant_terms) is not int or maximum_determinant_terms < 1:
        raise ValueError("maximum_determinant_terms must be a positive built-in integer")
    if not isinstance(uncertainty_radii, (tuple, list)) or len(uncertainty_radii) != n:
        raise ValueError("uncertainty_radii must shape-match the matrix")
    boxes = []
    parsed_radii = []
    for i in range(n):
        if not isinstance(uncertainty_radii[i], (tuple, list)) or len(uncertainty_radii[i]) != n:
            raise ValueError("uncertainty_radii must shape-match the matrix")
        box_row, radius_row = [], []
        for j in range(n):
            item = uncertainty_radii[i][j]
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("every uncertainty entry must be a real/imaginary radius pair")
            rr = _fraction(item[0], f"uncertainty_radii[{i}][{j}].real")
            ri = _fraction(item[1], f"uncertainty_radii[{i}][{j}].imag")
            if rr < 0 or ri < 0:
                raise ValueError("uncertainty radii must be nonnegative")
            value = matrix[i][j] / scale
            rr, ri = rr / scale, ri / scale
            box_row.append(QComplexRectangle(value.real - rr, value.real + rr, value.imag - ri, value.imag + ri))
            radius_row.append((rr, ri))
        boxes.append(tuple(box_row)); parsed_radii.append(tuple(radius_row))
    boxes = tuple(boxes); parsed_radii = tuple(parsed_radii)
    block_ok = all(
        (i in inside) == (j in inside) or (matrix[i][j] == ZERO and parsed_radii[i][j] == (0, 0))
        for i in range(n) for j in range(n)
    )
    required = factorial(len(inside)) + factorial(len(outside)) + factorial(n)
    failures: list[str] = []
    if not block_ok:
        failures.append("DIRECT_INTERVAL_FACTOR_COORDINATE_BLOCK_NOT_INVARIANT")
    if required > maximum_determinant_terms:
        failures.append("DIRECT_INTERVAL_FACTOR_DETERMINANT_TERM_BUDGET_EXCEEDED")
    interval = verified_interval_characteristic_spectral_split(
        nominal_transition, uncertainty_radii=uncertainty_radii, center=center,
        radius=radius, spectral_reference_scale=scale,
        maximum_partitions=maximum_partitions,
        maximum_factor_candidate_value_tuples=maximum_factor_candidate_value_tuples,
        nodes=nodes, sqrt_precision=sqrt_precision,
    )
    if interval.validation_level is None:
        failures.append("DIRECT_INTERVAL_FACTOR_IVSPEC_BRIDGE_FAILED")
    coordinate_projector = tuple(tuple(ONE if i == j and i in inside else ZERO for j in range(n)) for i in range(n))
    projector_match = interval.nominal_exact_projector == coordinate_projector
    if not projector_match:
        failures.append("DIRECT_INTERVAL_FACTOR_COORDINATE_PROJECTOR_MISMATCH")
    inside_factor = _characteristic_boxes(boxes, inside) if required <= maximum_determinant_terms else ()
    outside_factor = _characteristic_boxes(boxes, outside) if required <= maximum_determinant_terms else ()
    product = _pmul(inside_factor, outside_factor) if inside_factor and outside_factor else ()
    direct = _characteristic_boxes(boxes, tuple(range(n))) if required <= maximum_determinant_terms else ()
    intersections = bool(product and direct and len(product) == len(direct) and all(_contains_zero_intersection(a, b) for a, b in zip(product, direct, strict=True)))
    if not intersections:
        failures.append("DIRECT_INTERVAL_FACTOR_PRODUCT_DIRECT_INTERSECTION_FAILED")
    rank_match = interval.family_projector_rank == len(inside)
    if not rank_match:
        failures.append("DIRECT_INTERVAL_FACTOR_DEGREE_RANK_MISMATCH")
    monic = bool(inside_factor and outside_factor and inside_factor[-1] == ONE_BOX and outside_factor[-1] == ONE_BOX)
    if not monic:
        failures.append("DIRECT_INTERVAL_FACTOR_MONIC_CHECK_FAILED")
    success = not failures
    status = "VERIFIED_DIRECT_INTERVAL_COORDINATE_BLOCK_CHARACTERISTIC_FACTORS" if success else failures[0]
    return VerifiedDirectIntervalCoordinateBlockFactors(
        status=status, validation_level=status if success else None,
        failure_codes=tuple(failures), interval_spectral_split=interval,
        coordinate_inside_indices=inside, coordinate_outside_indices=outside,
        coordinate_projector_matches_nominal_discovery=projector_match,
        nominal_and_uncertainty_block_diagonal_verified=block_ok,
        determinant_terms_required=required,
        maximum_determinant_terms=maximum_determinant_terms,
        normalized_inside_factor_coefficient_boxes=inside_factor,
        normalized_outside_factor_coefficient_boxes=outside_factor,
        normalized_factor_product_coefficient_boxes=product,
        normalized_direct_full_characteristic_coefficient_boxes=direct,
        product_and_direct_coefficient_intersections_verified=intersections,
        inside_factor_degree=len(inside), outside_factor_degree=len(outside),
        factor_degrees_match_family_rank=rank_match,
        monic_factor_coefficients_verified=monic,
        direct_interval_characteristic_factor_output_verified=success,
        interval_root_tracking_required=False,
        empirical_uncertainty_provenance_verified=False,
    )


__all__ = ["QComplexRectangle", "VerifiedDirectIntervalCoordinateBlockFactors", "verified_direct_interval_coordinate_block_factors"]
