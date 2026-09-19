"""Certified continuous optimization of a circular spectral split margin."""

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
        _fraction,
        _inverse,
        _matmul,
        _matrix,
        parse_qcomplex,
    )


QMatrix = tuple[tuple[QComplex, ...], ...]


@dataclass(frozen=True)
class ContinuousCircleCellReceipt:
    center_real_interval: tuple[Fraction, Fraction]
    center_imag_interval: tuple[Fraction, Fraction]
    radius_interval: tuple[Fraction, Fraction]
    midpoint_center: QComplex
    midpoint_radius: Fraction
    midpoint_signed_squared_margin: Fraction
    cell_global_upper: Fraction
    per_spectral_value_variation_uppers: tuple[Fraction, ...]


@dataclass(frozen=True)
class VerifiedContinuousCircleSpectralMarginOptimization:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    normalized_transition: QMatrix
    normalized_eigenvalues: tuple[QComplex, ...]
    target_inside_labels: tuple[bool, ...]
    exact_diagonalization_witness_verified: bool
    normalized_center_real_interval: tuple[Fraction, Fraction]
    normalized_center_imag_interval: tuple[Fraction, Fraction]
    normalized_radius_interval: tuple[Fraction, Fraction]
    normalized_optimality_tolerance: Fraction
    maximum_cells: int
    cells_evaluated: int
    terminal_cells: tuple[ContinuousCircleCellReceipt, ...]
    selected_normalized_center: QComplex | None
    selected_normalized_radius: Fraction | None
    selected_signed_squared_margin: Fraction | None
    certified_global_score_upper: Fraction | None
    certified_optimality_gap: Fraction | None
    epsilon_global_optimality_verified: bool
    positive_target_margin_verified: bool
    selected_characteristic_discovery: VerifiedCharacteristicSpectralSplitDiscovery | None
    selected_projector_rank: int | None
    target_rank: int
    rank_consistency_verified: bool
    continuous_parameter_box_covered: bool
    finite_grid_only: bool
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


def _signed_margin(
    center: QComplex,
    radius: Fraction,
    eigenvalue: QComplex,
    inside: bool,
) -> Fraction:
    distance_squared = (eigenvalue - center).abs_squared()
    return radius * radius - distance_squared if inside else distance_squared - radius * radius


def _cell_receipt(
    intervals: tuple[
        tuple[Fraction, Fraction],
        tuple[Fraction, Fraction],
        tuple[Fraction, Fraction],
    ],
    eigenvalues: tuple[QComplex, ...],
    labels: tuple[bool, ...],
) -> ContinuousCircleCellReceipt:
    real_interval, imag_interval, radius_interval = intervals
    x = _midpoint(real_interval)
    y = _midpoint(imag_interval)
    radius = _midpoint(radius_interval)
    center = QComplex(x, y)
    hx = _half_width(real_interval)
    hy = _half_width(imag_interval)
    hr = _half_width(radius_interval)
    scores: list[Fraction] = []
    variations: list[Fraction] = []
    for eigenvalue, inside in zip(eigenvalues, labels, strict=True):
        score = _signed_margin(center, radius, eigenvalue, inside)
        max_dx = max(abs(real_interval[0] - eigenvalue.real), abs(real_interval[1] - eigenvalue.real))
        max_dy = max(abs(imag_interval[0] - eigenvalue.imag), abs(imag_interval[1] - eigenvalue.imag))
        max_r = max(abs(radius_interval[0]), abs(radius_interval[1]))
        variation = 2 * max_dx * hx + 2 * max_dy * hy + 2 * max_r * hr
        scores.append(score)
        variations.append(variation)
    midpoint_score = min(scores)
    upper = min(score + variation for score, variation in zip(scores, variations, strict=True))
    return ContinuousCircleCellReceipt(
        center_real_interval=real_interval,
        center_imag_interval=imag_interval,
        radius_interval=radius_interval,
        midpoint_center=center,
        midpoint_radius=radius,
        midpoint_signed_squared_margin=midpoint_score,
        cell_global_upper=upper,
        per_spectral_value_variation_uppers=tuple(variations),
    )


def _split_cell(cell: ContinuousCircleCellReceipt):
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
    left = list(intervals)
    right = list(intervals)
    left[axis] = (intervals[axis][0], midpoint)
    right[axis] = (midpoint, intervals[axis][1])
    return tuple(left), tuple(right)


def _diagonal(values: tuple[QComplex, ...]) -> QMatrix:
    n = len(values)
    return tuple(
        tuple(values[i] if i == j else ZERO for j in range(n)) for i in range(n)
    )


def verified_continuous_circle_spectral_margin_optimization(
    nominal_transition: object,
    *,
    eigenvectors: object,
    eigenvalues: object,
    target_inside_labels: object,
    center_real_interval: object,
    center_imag_interval: object,
    radius_interval: object,
    spectral_reference_scale: object,
    normalized_optimality_tolerance: object,
    maximum_cells: int = 4096,
    maximum_partitions: int = 1024,
    maximum_factor_candidate_value_tuples: int = 1_000_000,
    sqrt_precision: int = 32,
) -> VerifiedContinuousCircleSpectralMarginOptimization:
    """Return a tolerance-global optimum over one continuous circle box."""
    raw = _matrix(nominal_transition, "nominal_transition")
    n = len(raw)
    vectors = _matrix(eigenvectors, "eigenvectors")
    if len(vectors) != n:
        raise ValueError("eigenvectors must shape-match nominal_transition")
    if not isinstance(eigenvalues, (tuple, list)) or len(eigenvalues) != n:
        raise ValueError("eigenvalues must contain exactly n exact Q(i) values")
    values = tuple(parse_qcomplex(value, f"eigenvalues[{i}]") for i, value in enumerate(eigenvalues))
    if not isinstance(target_inside_labels, (tuple, list)) or len(target_inside_labels) != n:
        raise ValueError("target_inside_labels must contain exactly n booleans")
    if any(type(label) is not bool for label in target_inside_labels):
        raise ValueError("target_inside_labels entries must be built-in booleans")
    labels = tuple(target_inside_labels)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    tolerance = _fraction(normalized_optimality_tolerance, "normalized_optimality_tolerance")
    if scale <= 0 or tolerance < 0:
        raise ValueError("spectral_reference_scale must be positive and tolerance nonnegative")
    if type(maximum_cells) is not int or maximum_cells < 1:
        raise ValueError("maximum_cells must be a positive built-in integer")
    real_raw = _interval(center_real_interval, "center_real_interval")
    imag_raw = _interval(center_imag_interval, "center_imag_interval")
    radius_raw = _interval(radius_interval, "radius_interval")
    if radius_raw[0] <= 0:
        raise ValueError("radius_interval must be strictly positive")
    normalized = tuple(tuple(entry / scale for entry in row) for row in raw)
    normalized_vectors = vectors
    normalized_values = tuple(value / scale for value in values)
    witness = (
        _inverse(normalized_vectors) is not None
        and _matmul(normalized, normalized_vectors)
        == _matmul(normalized_vectors, _diagonal(normalized_values))
    )
    normalized_intervals = (
        (real_raw[0] / scale, real_raw[1] / scale),
        (imag_raw[0] / scale, imag_raw[1] / scale),
        (radius_raw[0] / scale, radius_raw[1] / scale),
    )
    root = _cell_receipt(normalized_intervals, normalized_values, labels)
    cells = [root]
    evaluated = 1
    incumbent = root
    budget_failure = False
    while True:
        best_upper = max(cell.cell_global_upper for cell in cells)
        if best_upper - incumbent.midpoint_signed_squared_margin <= tolerance:
            break
        if evaluated + 2 > maximum_cells:
            budget_failure = True
            break
        chosen_index = max(
            range(len(cells)),
            key=lambda index: (cells[index].cell_global_upper, -index),
        )
        chosen = cells.pop(chosen_index)
        children = _split_cell(chosen)
        if children is None:
            cells.append(chosen)
            break
        for child_intervals in children:
            child = _cell_receipt(child_intervals, normalized_values, labels)
            cells.append(child)
            evaluated += 1
            if child.midpoint_signed_squared_margin > incumbent.midpoint_signed_squared_margin:
                incumbent = child
    global_upper = max(cell.cell_global_upper for cell in cells)
    gap = global_upper - incumbent.midpoint_signed_squared_margin
    failures: list[str] = []
    if not witness:
        failures.append("CONTINUOUS_CIRCLE_EXACT_DIAGONALIZATION_WITNESS_FAILED")
    if budget_failure or gap > tolerance:
        failures.append("CONTINUOUS_CIRCLE_OPTIMIZATION_CELL_BUDGET_EXCEEDED")
    if incumbent.midpoint_signed_squared_margin <= 0:
        failures.append("CONTINUOUS_CIRCLE_NO_POSITIVE_TARGET_MARGIN")
    discovery = None
    rank = None
    rank_consistency = False
    if not failures:
        selected_center_raw = incumbent.midpoint_center * scale
        selected_radius_raw = incumbent.midpoint_radius * scale
        discovery = verified_characteristic_spectral_split_discovery(
            nominal_transition,
            center=selected_center_raw,
            radius=selected_radius_raw,
            spectral_reference_scale=scale,
            maximum_partitions=maximum_partitions,
            maximum_factor_candidate_value_tuples=maximum_factor_candidate_value_tuples,
            sqrt_precision=sqrt_precision,
        )
        rank = discovery.projector_rank
        rank_consistency = (
            discovery.validation_level is not None and rank == sum(labels)
        )
        if not rank_consistency:
            failures.append("CONTINUOUS_CIRCLE_SELECTED_CONTOUR_RANK_VALIDATION_FAILED")
    success = not failures
    status = (
        "VERIFIED_CONTINUOUS_CIRCLE_SPECTRAL_MARGIN_OPTIMUM"
        if success
        else failures[0]
    )
    return VerifiedContinuousCircleSpectralMarginOptimization(
        status=status,
        validation_level=status if success else None,
        failure_codes=tuple(failures),
        normalized_transition=normalized,
        normalized_eigenvalues=normalized_values,
        target_inside_labels=labels,
        exact_diagonalization_witness_verified=witness,
        normalized_center_real_interval=normalized_intervals[0],
        normalized_center_imag_interval=normalized_intervals[1],
        normalized_radius_interval=normalized_intervals[2],
        normalized_optimality_tolerance=tolerance,
        maximum_cells=maximum_cells,
        cells_evaluated=evaluated,
        terminal_cells=tuple(cells),
        selected_normalized_center=incumbent.midpoint_center if success else None,
        selected_normalized_radius=incumbent.midpoint_radius if success else None,
        selected_signed_squared_margin=(
            incumbent.midpoint_signed_squared_margin if success else None
        ),
        certified_global_score_upper=global_upper if success else None,
        certified_optimality_gap=gap if success else None,
        epsilon_global_optimality_verified=success and gap <= tolerance,
        positive_target_margin_verified=(
            success and incumbent.midpoint_signed_squared_margin > 0
        ),
        selected_characteristic_discovery=discovery,
        selected_projector_rank=rank if success else None,
        target_rank=sum(labels),
        rank_consistency_verified=success and rank_consistency,
        continuous_parameter_box_covered=not budget_failure and gap <= tolerance,
        finite_grid_only=False,
        empirical_matrix_provenance_verified=False,
    )


__all__ = [
    "ContinuousCircleCellReceipt",
    "VerifiedContinuousCircleSpectralMarginOptimization",
    "verified_continuous_circle_spectral_margin_optimization",
]
