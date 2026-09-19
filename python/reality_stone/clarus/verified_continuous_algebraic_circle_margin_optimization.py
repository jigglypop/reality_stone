"""Continuous circle optimization without eigenvectors or diagonalization."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_characteristic_spectral_split_discovery import (
        VerifiedCharacteristicSpectralSplitDiscovery,
        verified_characteristic_spectral_split_discovery,
    )
    from .verified_rational_contour import (
        ONE,
        ZERO,
        QComplex,
        _dyadic_sqrt,
        _fraction,
        _inverse,
        _matmul,
        _matrix,
        parse_qcomplex,
    )
else:
    from verified_characteristic_spectral_split_discovery import (  # type: ignore[no-redef]
        VerifiedCharacteristicSpectralSplitDiscovery,
        verified_characteristic_spectral_split_discovery,
    )
    from verified_rational_contour import (  # type: ignore[no-redef]
        ONE,
        ZERO,
        QComplex,
        _dyadic_sqrt,
        _fraction,
        _inverse,
        _matmul,
        _matrix,
        parse_qcomplex,
    )


QMatrix = tuple[tuple[QComplex, ...], ...]


@dataclass(frozen=True)
class ContinuousAlgebraicCircleCellReceipt:
    center_real_interval: tuple[Fraction, Fraction]
    center_imag_interval: tuple[Fraction, Fraction]
    radius_interval: tuple[Fraction, Fraction]
    midpoint_center: QComplex
    midpoint_radius: Fraction
    inside_frobenius_squared: Fraction
    exterior_inverse_frobenius_squared: Fraction
    midpoint_algebraic_margin: Fraction
    inside_margin_cell_upper: Fraction
    outside_margin_cell_upper: Fraction
    cell_global_upper: Fraction
    center_displacement_upper: Fraction
    exterior_inverse_difference_upper: Fraction


@dataclass(frozen=True)
class VerifiedContinuousAlgebraicCircleMarginOptimization:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    reference_discovery: VerifiedCharacteristicSpectralSplitDiscovery
    reference_projector: QMatrix | None
    reference_projector_rank: int | None
    normalized_center_real_interval: tuple[Fraction, Fraction]
    normalized_center_imag_interval: tuple[Fraction, Fraction]
    normalized_radius_interval: tuple[Fraction, Fraction]
    normalized_optimality_tolerance: Fraction
    uniform_exterior_center_domain_verified: bool
    reference_to_box_center_displacement_upper: Fraction | None
    reference_exterior_inverse_frobenius_upper: Fraction | None
    maximum_cells: int
    cells_evaluated: int
    terminal_cells: tuple[ContinuousAlgebraicCircleCellReceipt, ...]
    selected_normalized_center: QComplex | None
    selected_normalized_radius: Fraction | None
    selected_algebraic_margin: Fraction | None
    certified_global_score_upper: Fraction | None
    certified_optimality_gap: Fraction | None
    epsilon_global_optimality_verified: bool
    positive_algebraic_margin_verified: bool
    selected_discovery: VerifiedCharacteristicSpectralSplitDiscovery | None
    selected_projector_rank: int | None
    projector_identity_preserved: bool
    diagonalization_witness_required: bool
    continuous_parameter_box_covered: bool
    empirical_matrix_provenance_verified: bool


def _interval(value: object, name: str) -> tuple[Fraction, Fraction]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be an exact (lower, upper) pair")
    lower = _fraction(value[0], f"{name}.lower")
    upper = _fraction(value[1], f"{name}.upper")
    if lower > upper:
        raise ValueError(f"{name} lower must not exceed upper")
    return lower, upper


def _midpoint(interval: tuple[Fraction, Fraction]) -> Fraction:
    return (interval[0] + interval[1]) / 2


def _half_width(interval: tuple[Fraction, Fraction]) -> Fraction:
    return (interval[1] - interval[0]) / 2


def _identity(n: int) -> QMatrix:
    return tuple(tuple(ONE if i == j else ZERO for j in range(n)) for i in range(n))


def _add(left: QMatrix, right: QMatrix) -> QMatrix:
    return tuple(
        tuple(left[i][j] + right[i][j] for j in range(len(left)))
        for i in range(len(left))
    )


def _sub(left: QMatrix, right: QMatrix) -> QMatrix:
    return tuple(
        tuple(left[i][j] - right[i][j] for j in range(len(left)))
        for i in range(len(left))
    )


def _centered(matrix: QMatrix, center: QComplex) -> QMatrix:
    return tuple(
        tuple(entry - (center if i == j else ZERO) for j, entry in enumerate(row))
        for i, row in enumerate(matrix)
    )


def _frobenius_squared(matrix: QMatrix) -> Fraction:
    return sum((entry.abs_squared() for row in matrix for entry in row), Fraction(0))


def _exterior_inverse(
    matrix: QMatrix,
    center: QComplex,
    projector: QMatrix,
    complement: QMatrix,
) -> QMatrix | None:
    centered = _centered(matrix, center)
    block = _add(projector, _matmul(_matmul(complement, centered), complement))
    inverse = _inverse(block)
    return None if inverse is None else _matmul(_matmul(complement, inverse), complement)


def _sqrt_bracket(value: Fraction, precision: int):
    lower, upper, _, _ = _dyadic_sqrt(value, precision)
    return lower, upper


def _cell_receipt(
    intervals,
    matrix: QMatrix,
    projector: QMatrix,
    complement: QMatrix,
    projector_frobenius_upper: Fraction,
    sqrt_precision: int,
) -> ContinuousAlgebraicCircleCellReceipt:
    real_interval, imag_interval, radius_interval = intervals
    center = QComplex(_midpoint(real_interval), _midpoint(imag_interval))
    radius = _midpoint(radius_interval)
    centered = _centered(matrix, center)
    inside = _matmul(centered, projector)
    exterior = _exterior_inverse(matrix, center, projector, complement)
    if exterior is None:
        raise AssertionError("uniform Neumann domain must keep every midpoint invertible")
    inside_squared = _frobenius_squared(inside)
    exterior_squared = _frobenius_squared(exterior)
    inside_score = radius * radius - inside_squared
    outside_score = Fraction(1) - radius * radius * exterior_squared
    hx = _half_width(real_interval)
    hy = _half_width(imag_interval)
    hr = _half_width(radius_interval)
    displacement = hx + hy
    if hx == 0 and hy == 0 and hr == 0:
        exterior_difference = Fraction(0)
        inside_upper = inside_score
        outside_upper = outside_score
    else:
        inside_lower, _ = _sqrt_bracket(inside_squared, sqrt_precision)
        exterior_lower, exterior_upper = _sqrt_bracket(exterior_squared, sqrt_precision)
        inside_norm_lower = max(
            Fraction(0), inside_lower - displacement * projector_frobenius_upper
        )
        inside_upper = (
            radius_interval[1] * radius_interval[1]
            - inside_norm_lower * inside_norm_lower
        )
        denominator = Fraction(1) - displacement * exterior_upper
        if denominator <= 0:
            exterior_difference = exterior_lower
            exterior_norm_lower = Fraction(0)
        else:
            exterior_difference = (
                displacement * exterior_upper * exterior_upper / denominator
            )
            exterior_norm_lower = max(
                Fraction(0), exterior_lower - exterior_difference
            )
        outside_upper = (
            Fraction(1)
            - radius_interval[0] * radius_interval[0]
            * exterior_norm_lower * exterior_norm_lower
        )
    return ContinuousAlgebraicCircleCellReceipt(
        center_real_interval=real_interval,
        center_imag_interval=imag_interval,
        radius_interval=radius_interval,
        midpoint_center=center,
        midpoint_radius=radius,
        inside_frobenius_squared=inside_squared,
        exterior_inverse_frobenius_squared=exterior_squared,
        midpoint_algebraic_margin=min(inside_score, outside_score),
        inside_margin_cell_upper=inside_upper,
        outside_margin_cell_upper=outside_upper,
        cell_global_upper=min(inside_upper, outside_upper),
        center_displacement_upper=displacement,
        exterior_inverse_difference_upper=exterior_difference,
    )


def _split_cell(cell: ContinuousAlgebraicCircleCellReceipt):
    intervals = (
        cell.center_real_interval,
        cell.center_imag_interval,
        cell.radius_interval,
    )
    widths = tuple(interval[1] - interval[0] for interval in intervals)
    axis = max(range(3), key=lambda index: (widths[index], -index))
    if widths[axis] == 0:
        return None
    midpoint = _midpoint(intervals[axis])
    left, right = list(intervals), list(intervals)
    left[axis] = (intervals[axis][0], midpoint)
    right[axis] = (midpoint, intervals[axis][1])
    return tuple(left), tuple(right)


def verified_continuous_algebraic_circle_margin_optimization(
    nominal_transition: object,
    *,
    reference_center: object,
    reference_radius: object,
    center_real_interval: object,
    center_imag_interval: object,
    radius_interval: object,
    spectral_reference_scale: object,
    normalized_optimality_tolerance: object,
    maximum_cells: int = 4096,
    maximum_partitions: int = 1024,
    maximum_factor_candidate_value_tuples: int = 1_000_000,
    sqrt_precision: int = 32,
) -> VerifiedContinuousAlgebraicCircleMarginOptimization:
    """Optimize a Frobenius algebraic Riesz margin on a continuous circle box."""
    raw = _matrix(nominal_transition, "nominal_transition")
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    tolerance = _fraction(normalized_optimality_tolerance, "normalized_optimality_tolerance")
    radius_reference = _fraction(reference_radius, "reference_radius")
    center_reference_raw = parse_qcomplex(reference_center, "reference_center")
    if scale <= 0 or radius_reference <= 0 or tolerance < 0:
        raise ValueError("scale/reference radius must be positive and tolerance nonnegative")
    if type(maximum_cells) is not int or maximum_cells < 1:
        raise ValueError("maximum_cells must be a positive built-in integer")
    real_raw = _interval(center_real_interval, "center_real_interval")
    imag_raw = _interval(center_imag_interval, "center_imag_interval")
    radius_raw = _interval(radius_interval, "radius_interval")
    if radius_raw[0] <= 0:
        raise ValueError("radius_interval must be strictly positive")
    center_reference = center_reference_raw / scale
    normalized = tuple(tuple(entry / scale for entry in row) for row in raw)
    intervals = (
        (real_raw[0] / scale, real_raw[1] / scale),
        (imag_raw[0] / scale, imag_raw[1] / scale),
        (radius_raw[0] / scale, radius_raw[1] / scale),
    )
    if not (
        intervals[0][0] <= center_reference.real <= intervals[0][1]
        and intervals[1][0] <= center_reference.imag <= intervals[1][1]
    ):
        raise ValueError("reference_center must lie inside the center parameter box")
    reference = verified_characteristic_spectral_split_discovery(
        nominal_transition,
        center=center_reference_raw,
        radius=radius_reference,
        spectral_reference_scale=scale,
        maximum_partitions=maximum_partitions,
        maximum_factor_candidate_value_tuples=maximum_factor_candidate_value_tuples,
        sqrt_precision=sqrt_precision,
    )
    construction = reference.selected_construction
    projector = None if construction is None else construction.constructed_projector
    exterior_reference = (
        None if construction is None else construction.constructed_exterior_centered_inverse
    )
    rank = reference.projector_rank
    common = dict(
        reference_discovery=reference,
        reference_projector=projector,
        reference_projector_rank=rank,
        normalized_center_real_interval=intervals[0],
        normalized_center_imag_interval=intervals[1],
        normalized_radius_interval=intervals[2],
        normalized_optimality_tolerance=tolerance,
        maximum_cells=maximum_cells,
        diagonalization_witness_required=False,
        empirical_matrix_provenance_verified=False,
    )
    if reference.validation_level is None or projector is None or exterior_reference is None or rank is None:
        failure = "ALGEBRAIC_CONTINUOUS_REFERENCE_DISCOVERY_UNAVAILABLE"
        return VerifiedContinuousAlgebraicCircleMarginOptimization(
            status=failure, validation_level=None, failure_codes=(failure,),
            uniform_exterior_center_domain_verified=False,
            reference_to_box_center_displacement_upper=None,
            reference_exterior_inverse_frobenius_upper=None,
            cells_evaluated=0, terminal_cells=(), selected_normalized_center=None,
            selected_normalized_radius=None, selected_algebraic_margin=None,
            certified_global_score_upper=None, certified_optimality_gap=None,
            epsilon_global_optimality_verified=False,
            positive_algebraic_margin_verified=False, selected_discovery=None,
            selected_projector_rank=None, projector_identity_preserved=False,
            continuous_parameter_box_covered=False, **common,
        )
    n = len(normalized)
    complement = _sub(_identity(n), projector)
    p2 = _frobenius_squared(projector)
    _, p_upper = _sqrt_bracket(p2, sqrt_precision)
    r2 = _frobenius_squared(exterior_reference)
    _, exterior_reference_upper = _sqrt_bracket(r2, sqrt_precision)
    box_displacement = (
        max(abs(intervals[0][0] - center_reference.real), abs(intervals[0][1] - center_reference.real))
        + max(abs(intervals[1][0] - center_reference.imag), abs(intervals[1][1] - center_reference.imag))
    )
    uniform_domain = box_displacement * exterior_reference_upper < 1
    if not uniform_domain:
        failure = "ALGEBRAIC_CONTINUOUS_EXTERIOR_CENTER_DOMAIN_NOT_UNIFORM"
        return VerifiedContinuousAlgebraicCircleMarginOptimization(
            status=failure, validation_level=None, failure_codes=(failure,),
            uniform_exterior_center_domain_verified=False,
            reference_to_box_center_displacement_upper=box_displacement,
            reference_exterior_inverse_frobenius_upper=exterior_reference_upper,
            cells_evaluated=0, terminal_cells=(), selected_normalized_center=None,
            selected_normalized_radius=None, selected_algebraic_margin=None,
            certified_global_score_upper=None, certified_optimality_gap=None,
            epsilon_global_optimality_verified=False,
            positive_algebraic_margin_verified=False, selected_discovery=None,
            selected_projector_rank=None, projector_identity_preserved=False,
            continuous_parameter_box_covered=False, **common,
        )
    root = _cell_receipt(
        intervals, normalized, projector, complement, p_upper, sqrt_precision
    )
    cells = [root]
    incumbent = root
    evaluated = 1
    budget_failure = False
    while True:
        global_upper = max(cell.cell_global_upper for cell in cells)
        if global_upper - incumbent.midpoint_algebraic_margin <= tolerance:
            break
        if evaluated + 2 > maximum_cells:
            budget_failure = True
            break
        index = max(range(len(cells)), key=lambda i: (cells[i].cell_global_upper, -i))
        chosen = cells.pop(index)
        children = _split_cell(chosen)
        if children is None:
            cells.append(chosen)
            break
        for child_intervals in children:
            child = _cell_receipt(
                child_intervals, normalized, projector, complement, p_upper, sqrt_precision
            )
            cells.append(child)
            evaluated += 1
            if child.midpoint_algebraic_margin > incumbent.midpoint_algebraic_margin:
                incumbent = child
    global_upper = max(cell.cell_global_upper for cell in cells)
    gap = global_upper - incumbent.midpoint_algebraic_margin
    failures: list[str] = []
    if budget_failure or gap > tolerance:
        failures.append("ALGEBRAIC_CONTINUOUS_OPTIMIZATION_CELL_BUDGET_EXCEEDED")
    if incumbent.midpoint_algebraic_margin <= 0:
        failures.append("ALGEBRAIC_CONTINUOUS_NO_POSITIVE_MARGIN")
    selected = None
    selected_rank = None
    projector_identity = False
    if not failures:
        selected = verified_characteristic_spectral_split_discovery(
            nominal_transition,
            center=incumbent.midpoint_center * scale,
            radius=incumbent.midpoint_radius * scale,
            spectral_reference_scale=scale,
            maximum_partitions=maximum_partitions,
            maximum_factor_candidate_value_tuples=maximum_factor_candidate_value_tuples,
            sqrt_precision=sqrt_precision,
        )
        selected_rank = selected.projector_rank
        selected_projector = (
            None if selected.selected_construction is None
            else selected.selected_construction.constructed_projector
        )
        projector_identity = (
            selected.validation_level is not None
            and selected_rank == rank
            and selected_projector == projector
        )
        if not projector_identity:
            failures.append("ALGEBRAIC_CONTINUOUS_SELECTED_PROJECTOR_VALIDATION_FAILED")
    success = not failures
    status = (
        "VERIFIED_CONTINUOUS_ALGEBRAIC_CIRCLE_MARGIN_OPTIMUM"
        if success else failures[0]
    )
    return VerifiedContinuousAlgebraicCircleMarginOptimization(
        status=status, validation_level=status if success else None,
        failure_codes=tuple(failures),
        uniform_exterior_center_domain_verified=True,
        reference_to_box_center_displacement_upper=box_displacement,
        reference_exterior_inverse_frobenius_upper=exterior_reference_upper,
        cells_evaluated=evaluated, terminal_cells=tuple(cells),
        selected_normalized_center=incumbent.midpoint_center if success else None,
        selected_normalized_radius=incumbent.midpoint_radius if success else None,
        selected_algebraic_margin=incumbent.midpoint_algebraic_margin if success else None,
        certified_global_score_upper=global_upper if success else None,
        certified_optimality_gap=gap if success else None,
        epsilon_global_optimality_verified=success and gap <= tolerance,
        positive_algebraic_margin_verified=success and incumbent.midpoint_algebraic_margin > 0,
        selected_discovery=selected,
        selected_projector_rank=selected_rank if success else None,
        projector_identity_preserved=success and projector_identity,
        continuous_parameter_box_covered=not budget_failure and gap <= tolerance,
        **common,
    )


__all__ = [
    "ContinuousAlgebraicCircleCellReceipt",
    "VerifiedContinuousAlgebraicCircleMarginOptimization",
    "verified_continuous_algebraic_circle_margin_optimization",
]
