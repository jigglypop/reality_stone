"""Certified continuous optimization of a fixed-orientation ellipse spectral margin."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
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
QInterval = tuple[Fraction, Fraction]


@dataclass(frozen=True)
class ContinuousEllipseCellReceipt:
    center_real_interval: QInterval
    center_imag_interval: QInterval
    semiaxis_u_interval: QInterval
    semiaxis_v_interval: QInterval
    midpoint_center: QComplex
    midpoint_semiaxis_u: Fraction
    midpoint_semiaxis_v: Fraction
    midpoint_signed_quartic_margin: Fraction
    cell_global_upper: Fraction
    per_spectral_value_score_intervals: tuple[QInterval, ...]


@dataclass(frozen=True)
class VerifiedContinuousAxisAlignedEllipseSpectralMarginOptimization:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    normalized_transition: QMatrix
    normalized_eigenvalues: tuple[QComplex, ...]
    orientation_unit_u: QComplex
    orientation_unit_v: QComplex
    target_inside_labels: tuple[bool, ...]
    exact_diagonalization_witness_verified: bool
    normalized_center_real_interval: QInterval
    normalized_center_imag_interval: QInterval
    normalized_semiaxis_u_interval: QInterval
    normalized_semiaxis_v_interval: QInterval
    normalized_quartic_optimality_tolerance: Fraction
    maximum_cells: int
    cells_evaluated: int
    terminal_cells: tuple[ContinuousEllipseCellReceipt, ...]
    selected_normalized_center: QComplex | None
    selected_normalized_semiaxis_u: Fraction | None
    selected_normalized_semiaxis_v: Fraction | None
    selected_signed_quartic_margin: Fraction | None
    certified_global_score_upper: Fraction | None
    certified_optimality_gap: Fraction | None
    epsilon_global_optimality_verified: bool
    positive_target_margin_verified: bool
    selected_projector: QMatrix | None
    selected_projector_rank: int | None
    projector_identity_verified: bool
    selected_axes_positive_orientation_verified: bool
    target_rank: int
    continuous_parameter_box_covered: bool
    finite_grid_only: bool
    orientation_optimized: bool
    empirical_matrix_provenance_verified: bool


def _interval(value: object, name: str) -> QInterval:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be an exact (lower, upper) pair")
    lower = _fraction(value[0], f"{name}.lower")
    upper = _fraction(value[1], f"{name}.upper")
    if lower > upper:
        raise ValueError(f"{name} lower must not exceed upper")
    return lower, upper


def _midpoint(interval: QInterval) -> Fraction:
    return (interval[0] + interval[1]) / 2


def _add_interval(left: QInterval, right: QInterval) -> QInterval:
    return left[0] + right[0], left[1] + right[1]


def _scale_interval(value: QInterval, scalar: Fraction) -> QInterval:
    endpoints = scalar * value[0], scalar * value[1]
    return min(endpoints), max(endpoints)


def _square_interval(value: QInterval) -> QInterval:
    lower, upper = value
    high = max(lower * lower, upper * upper)
    low = Fraction(0) if lower <= 0 <= upper else min(lower * lower, upper * upper)
    return low, high


def _nonnegative_product(left: QInterval, right: QInterval) -> QInterval:
    if left[0] < 0 or right[0] < 0:
        raise AssertionError("nonnegative interval product received a negative bound")
    return left[0] * right[0], left[1] * right[1]


def _diagonal(values: tuple[QComplex, ...]) -> QMatrix:
    n = len(values)
    return tuple(
        tuple(values[i] if i == j else ZERO for j in range(n)) for i in range(n)
    )


def _projector(
    eigenvectors: QMatrix,
    labels: tuple[bool, ...],
) -> QMatrix | None:
    inverse = _inverse(eigenvectors)
    if inverse is None:
        return None
    selector = _diagonal(tuple(ONE if label else ZERO for label in labels))
    return _matmul(_matmul(eigenvectors, selector), inverse)


def _coordinate_intervals(
    eigenvalue: QComplex,
    center_real: QInterval,
    center_imag: QInterval,
    unit_u: QComplex,
) -> tuple[QInterval, QInterval]:
    dx = eigenvalue.real - center_real[1], eigenvalue.real - center_real[0]
    dy = eigenvalue.imag - center_imag[1], eigenvalue.imag - center_imag[0]
    xi = _add_interval(
        _scale_interval(dx, unit_u.real),
        _scale_interval(dy, unit_u.imag),
    )
    eta = _add_interval(
        _scale_interval(dx, -unit_u.imag),
        _scale_interval(dy, unit_u.real),
    )
    return xi, eta


def _score_interval(
    eigenvalue: QComplex,
    inside: bool,
    center_real: QInterval,
    center_imag: QInterval,
    semiaxis_u: QInterval,
    semiaxis_v: QInterval,
    unit_u: QComplex,
) -> QInterval:
    xi, eta = _coordinate_intervals(
        eigenvalue, center_real, center_imag, unit_u
    )
    xi_squared = _square_interval(xi)
    eta_squared = _square_interval(eta)
    a_squared = _square_interval(semiaxis_u)
    b_squared = _square_interval(semiaxis_v)
    area_squared = _nonnegative_product(a_squared, b_squared)
    b_xi = _nonnegative_product(b_squared, xi_squared)
    a_eta = _nonnegative_product(a_squared, eta_squared)
    radial = _add_interval(b_xi, a_eta)
    signed = area_squared[0] - radial[1], area_squared[1] - radial[0]
    return signed if inside else (-signed[1], -signed[0])


def _point_score(
    eigenvalue: QComplex,
    inside: bool,
    center: QComplex,
    semiaxis_u: Fraction,
    semiaxis_v: Fraction,
    unit_u: QComplex,
) -> Fraction:
    delta = eigenvalue - center
    xi = unit_u.real * delta.real + unit_u.imag * delta.imag
    eta = -unit_u.imag * delta.real + unit_u.real * delta.imag
    signed = (
        semiaxis_u * semiaxis_u * semiaxis_v * semiaxis_v
        - semiaxis_v * semiaxis_v * xi * xi
        - semiaxis_u * semiaxis_u * eta * eta
    )
    return signed if inside else -signed


def _cell_receipt(
    intervals: tuple[QInterval, QInterval, QInterval, QInterval],
    eigenvalues: tuple[QComplex, ...],
    labels: tuple[bool, ...],
    unit_u: QComplex,
) -> ContinuousEllipseCellReceipt:
    center_real, center_imag, semiaxis_u, semiaxis_v = intervals
    center = QComplex(_midpoint(center_real), _midpoint(center_imag))
    a = _midpoint(semiaxis_u)
    b = _midpoint(semiaxis_v)
    point_scores = tuple(
        _point_score(value, label, center, a, b, unit_u)
        for value, label in zip(eigenvalues, labels, strict=True)
    )
    score_intervals = tuple(
        _score_interval(
            value,
            label,
            center_real,
            center_imag,
            semiaxis_u,
            semiaxis_v,
            unit_u,
        )
        for value, label in zip(eigenvalues, labels, strict=True)
    )
    return ContinuousEllipseCellReceipt(
        center_real_interval=center_real,
        center_imag_interval=center_imag,
        semiaxis_u_interval=semiaxis_u,
        semiaxis_v_interval=semiaxis_v,
        midpoint_center=center,
        midpoint_semiaxis_u=a,
        midpoint_semiaxis_v=b,
        midpoint_signed_quartic_margin=min(point_scores),
        cell_global_upper=min(interval[1] for interval in score_intervals),
        per_spectral_value_score_intervals=score_intervals,
    )


def _split_cell(cell: ContinuousEllipseCellReceipt):
    intervals = (
        cell.center_real_interval,
        cell.center_imag_interval,
        cell.semiaxis_u_interval,
        cell.semiaxis_v_interval,
    )
    widths = tuple(interval[1] - interval[0] for interval in intervals)
    axis = max(range(4), key=lambda index: (widths[index], -index))
    if widths[axis] == 0:
        return None
    midpoint = _midpoint(intervals[axis])
    left, right = list(intervals), list(intervals)
    left[axis] = intervals[axis][0], midpoint
    right[axis] = midpoint, intervals[axis][1]
    return tuple(left), tuple(right)


def verified_continuous_axis_aligned_ellipse_spectral_margin_optimization(
    nominal_transition: object,
    *,
    eigenvectors: object,
    eigenvalues: object,
    target_inside_labels: object,
    orientation_unit_u: object,
    center_real_interval: object,
    center_imag_interval: object,
    semiaxis_u_interval: object,
    semiaxis_v_interval: object,
    spectral_reference_scale: object,
    normalized_quartic_optimality_tolerance: object,
    maximum_cells: int = 4096,
) -> VerifiedContinuousAxisAlignedEllipseSpectralMarginOptimization:
    """Optimize a frozen spectral split over a continuous ellipse-shape box."""
    raw = _matrix(nominal_transition, "nominal_transition")
    n = len(raw)
    vectors = _matrix(eigenvectors, "eigenvectors")
    if len(vectors) != n:
        raise ValueError("eigenvectors must shape-match nominal_transition")
    if not isinstance(eigenvalues, (tuple, list)) or len(eigenvalues) != n:
        raise ValueError("eigenvalues must contain exactly n exact Q(i) values")
    values = tuple(
        parse_qcomplex(value, f"eigenvalues[{index}]")
        for index, value in enumerate(eigenvalues)
    )
    if not isinstance(target_inside_labels, (tuple, list)) or len(target_inside_labels) != n:
        raise ValueError("target_inside_labels must contain exactly n booleans")
    if any(type(label) is not bool for label in target_inside_labels):
        raise ValueError("target_inside_labels entries must be built-in booleans")
    labels = tuple(target_inside_labels)
    unit_u = parse_qcomplex(orientation_unit_u, "orientation_unit_u")
    if unit_u.abs_squared() != 1:
        raise ValueError("orientation_unit_u must be an exact rational unit direction")
    unit_v = QComplex(-unit_u.imag, unit_u.real)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    tolerance = _fraction(
        normalized_quartic_optimality_tolerance,
        "normalized_quartic_optimality_tolerance",
    )
    if scale <= 0 or tolerance < 0:
        raise ValueError("spectral_reference_scale must be positive and tolerance nonnegative")
    if type(maximum_cells) is not int or maximum_cells < 1:
        raise ValueError("maximum_cells must be a positive built-in integer")
    raw_intervals = (
        _interval(center_real_interval, "center_real_interval"),
        _interval(center_imag_interval, "center_imag_interval"),
        _interval(semiaxis_u_interval, "semiaxis_u_interval"),
        _interval(semiaxis_v_interval, "semiaxis_v_interval"),
    )
    if raw_intervals[2][0] <= 0 or raw_intervals[3][0] <= 0:
        raise ValueError("both semiaxis intervals must be strictly positive")
    normalized = tuple(tuple(entry / scale for entry in row) for row in raw)
    normalized_values = tuple(value / scale for value in values)
    normalized_intervals = tuple(
        (interval[0] / scale, interval[1] / scale) for interval in raw_intervals
    )
    witness = (
        _inverse(vectors) is not None
        and _matmul(normalized, vectors)
        == _matmul(vectors, _diagonal(normalized_values))
    )
    root = _cell_receipt(normalized_intervals, normalized_values, labels, unit_u)
    cells = [root]
    incumbent = root
    evaluated = 1
    budget_failure = False
    while True:
        global_upper = max(cell.cell_global_upper for cell in cells)
        if global_upper - incumbent.midpoint_signed_quartic_margin <= tolerance:
            break
        if evaluated + 2 > maximum_cells:
            budget_failure = True
            break
        chosen_index = max(
            range(len(cells)), key=lambda index: (cells[index].cell_global_upper, -index)
        )
        chosen = cells.pop(chosen_index)
        children = _split_cell(chosen)
        if children is None:
            cells.append(chosen)
            break
        for child_intervals in children:
            child = _cell_receipt(child_intervals, normalized_values, labels, unit_u)
            cells.append(child)
            evaluated += 1
            if child.midpoint_signed_quartic_margin > incumbent.midpoint_signed_quartic_margin:
                incumbent = child
    global_upper = max(cell.cell_global_upper for cell in cells)
    gap = global_upper - incumbent.midpoint_signed_quartic_margin
    failures: list[str] = []
    if not witness:
        failures.append("CONTINUOUS_ELLIPSE_EXACT_DIAGONALIZATION_WITNESS_FAILED")
    if budget_failure or gap > tolerance:
        failures.append("CONTINUOUS_ELLIPSE_OPTIMIZATION_CELL_BUDGET_EXCEEDED")
    if incumbent.midpoint_signed_quartic_margin <= 0:
        failures.append("CONTINUOUS_ELLIPSE_NO_POSITIVE_TARGET_MARGIN")
    selected_projector = _projector(vectors, labels) if witness else None
    projector_identity = bool(
        selected_projector is not None
        and _matmul(selected_projector, selected_projector) == selected_projector
        and _matmul(normalized, selected_projector)
        == _matmul(selected_projector, normalized)
    )
    if not projector_identity:
        failures.append("CONTINUOUS_ELLIPSE_PROJECTOR_IDENTITY_FAILED")
    selected_orientation = (
        incumbent.midpoint_semiaxis_u > 0
        and incumbent.midpoint_semiaxis_v > 0
        and unit_u.real * unit_v.imag - unit_u.imag * unit_v.real == 1
    )
    if not selected_orientation:
        failures.append("CONTINUOUS_ELLIPSE_SELECTED_ORIENTATION_FAILED")
    success = not failures
    status = (
        "VERIFIED_CONTINUOUS_AXIS_ALIGNED_ELLIPSE_SPECTRAL_MARGIN_OPTIMUM"
        if success
        else failures[0]
    )
    return VerifiedContinuousAxisAlignedEllipseSpectralMarginOptimization(
        status=status,
        validation_level=status if success else None,
        failure_codes=tuple(failures),
        normalized_transition=normalized,
        normalized_eigenvalues=normalized_values,
        orientation_unit_u=unit_u,
        orientation_unit_v=unit_v,
        target_inside_labels=labels,
        exact_diagonalization_witness_verified=witness,
        normalized_center_real_interval=normalized_intervals[0],
        normalized_center_imag_interval=normalized_intervals[1],
        normalized_semiaxis_u_interval=normalized_intervals[2],
        normalized_semiaxis_v_interval=normalized_intervals[3],
        normalized_quartic_optimality_tolerance=tolerance,
        maximum_cells=maximum_cells,
        cells_evaluated=evaluated,
        terminal_cells=tuple(cells),
        selected_normalized_center=incumbent.midpoint_center if success else None,
        selected_normalized_semiaxis_u=(incumbent.midpoint_semiaxis_u if success else None),
        selected_normalized_semiaxis_v=(incumbent.midpoint_semiaxis_v if success else None),
        selected_signed_quartic_margin=(incumbent.midpoint_signed_quartic_margin if success else None),
        certified_global_score_upper=global_upper if success else None,
        certified_optimality_gap=gap if success else None,
        epsilon_global_optimality_verified=success and gap <= tolerance,
        positive_target_margin_verified=success and incumbent.midpoint_signed_quartic_margin > 0,
        selected_projector=selected_projector if success else None,
        selected_projector_rank=sum(labels) if success else None,
        projector_identity_verified=success and projector_identity,
        selected_axes_positive_orientation_verified=success and selected_orientation,
        target_rank=sum(labels),
        continuous_parameter_box_covered=not budget_failure and gap <= tolerance,
        finite_grid_only=False,
        orientation_optimized=False,
        empirical_matrix_provenance_verified=False,
    )


__all__ = [
    "ContinuousEllipseCellReceipt",
    "VerifiedContinuousAxisAlignedEllipseSpectralMarginOptimization",
    "verified_continuous_axis_aligned_ellipse_spectral_margin_optimization",
]
