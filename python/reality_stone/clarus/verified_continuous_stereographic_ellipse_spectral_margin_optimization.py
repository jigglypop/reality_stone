"""Certified continuous ellipse optimization including stereographic orientation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_continuous_axis_aligned_ellipse_spectral_margin_optimization import (
        QInterval,
        QMatrix,
        _add_interval,
        _diagonal,
        _interval,
        _midpoint,
        _nonnegative_product,
        _projector,
        _square_interval,
    )
    from .verified_rational_contour import (
        QComplex,
        _fraction,
        _inverse,
        _matmul,
        _matrix,
        parse_qcomplex,
    )
else:
    from verified_continuous_axis_aligned_ellipse_spectral_margin_optimization import (  # type: ignore[no-redef]
        QInterval,
        QMatrix,
        _add_interval,
        _diagonal,
        _interval,
        _midpoint,
        _nonnegative_product,
        _projector,
        _square_interval,
    )
    from verified_rational_contour import (  # type: ignore[no-redef]
        QComplex,
        _fraction,
        _inverse,
        _matmul,
        _matrix,
        parse_qcomplex,
    )


@dataclass(frozen=True)
class ContinuousStereographicEllipseCellReceipt:
    center_real_interval: QInterval
    center_imag_interval: QInterval
    semiaxis_u_interval: QInterval
    semiaxis_v_interval: QInterval
    orientation_parameter_interval: QInterval
    midpoint_center: QComplex
    midpoint_semiaxis_u: Fraction
    midpoint_semiaxis_v: Fraction
    midpoint_orientation_parameter: Fraction
    midpoint_orientation_unit_u: QComplex
    midpoint_signed_ellipse_margin: Fraction
    cell_global_upper: Fraction
    per_spectral_value_score_intervals: tuple[QInterval, ...]


@dataclass(frozen=True)
class VerifiedContinuousStereographicEllipseSpectralMarginOptimization:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    normalized_transition: QMatrix
    normalized_eigenvalues: tuple[QComplex, ...]
    target_inside_labels: tuple[bool, ...]
    exact_diagonalization_witness_verified: bool
    normalized_center_real_interval: QInterval
    normalized_center_imag_interval: QInterval
    normalized_semiaxis_u_interval: QInterval
    normalized_semiaxis_v_interval: QInterval
    orientation_parameter_interval: QInterval
    normalized_optimality_tolerance: Fraction
    maximum_cells: int
    cells_evaluated: int
    terminal_cells: tuple[ContinuousStereographicEllipseCellReceipt, ...]
    selected_normalized_center: QComplex | None
    selected_normalized_semiaxis_u: Fraction | None
    selected_normalized_semiaxis_v: Fraction | None
    selected_orientation_parameter: Fraction | None
    selected_orientation_unit_u: QComplex | None
    selected_orientation_unit_v: QComplex | None
    selected_signed_ellipse_margin: Fraction | None
    certified_global_score_upper: Fraction | None
    certified_optimality_gap: Fraction | None
    epsilon_global_optimality_verified: bool
    positive_target_margin_verified: bool
    selected_projector: QMatrix | None
    selected_projector_rank: int | None
    projector_identity_verified: bool
    selected_orientation_unit_verified: bool
    target_rank: int
    continuous_parameter_box_covered: bool
    finite_grid_only: bool
    orientation_optimized: bool
    empirical_matrix_provenance_verified: bool


def _multiply_interval(left: QInterval, right: QInterval) -> QInterval:
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    return min(products), max(products)


def _negate_interval(value: QInterval) -> QInterval:
    return -value[1], -value[0]


def _subtract_interval(left: QInterval, right: QInterval) -> QInterval:
    return left[0] - right[1], left[1] - right[0]


def _divide_by_positive_interval(
    numerator: QInterval, denominator: QInterval
) -> QInterval:
    if denominator[0] <= 0:
        raise AssertionError("interval division requires a positive denominator")
    quotients = (
        numerator[0] / denominator[0],
        numerator[0] / denominator[1],
        numerator[1] / denominator[0],
        numerator[1] / denominator[1],
    )
    return min(quotients), max(quotients)


def _orientation_polynomial_intervals(
    parameter: QInterval,
) -> tuple[QInterval, QInterval, QInterval, QInterval]:
    t_squared = _square_interval(parameter)
    denominator = 1 + t_squared[0], 1 + t_squared[1]
    numerator_real = 1 - t_squared[1], 1 - t_squared[0]
    numerator_imag = 2 * parameter[0], 2 * parameter[1]
    denominator_squared = _square_interval(denominator)
    return numerator_real, numerator_imag, denominator, denominator_squared


def _numerator_coordinate_intervals(
    eigenvalue: QComplex,
    center_real: QInterval,
    center_imag: QInterval,
    orientation_parameter: QInterval,
) -> tuple[QInterval, QInterval, QInterval]:
    u, v, _, denominator_squared = _orientation_polynomial_intervals(
        orientation_parameter
    )
    dx = eigenvalue.real - center_real[1], eigenvalue.real - center_real[0]
    dy = eigenvalue.imag - center_imag[1], eigenvalue.imag - center_imag[0]
    xi_numerator = _add_interval(
        _multiply_interval(u, dx), _multiply_interval(v, dy)
    )
    eta_numerator = _add_interval(
        _multiply_interval(_negate_interval(v), dx),
        _multiply_interval(u, dy),
    )
    return xi_numerator, eta_numerator, denominator_squared


def _score_interval(
    eigenvalue: QComplex,
    inside: bool,
    center_real: QInterval,
    center_imag: QInterval,
    semiaxis_u: QInterval,
    semiaxis_v: QInterval,
    orientation_parameter: QInterval,
) -> QInterval:
    xi, eta, denominator_squared = _numerator_coordinate_intervals(
        eigenvalue, center_real, center_imag, orientation_parameter
    )
    a_squared = _square_interval(semiaxis_u)
    b_squared = _square_interval(semiaxis_v)
    area_squared = _nonnegative_product(a_squared, b_squared)
    positive = _nonnegative_product(denominator_squared, area_squared)
    xi_term = _nonnegative_product(b_squared, _square_interval(xi))
    eta_term = _nonnegative_product(a_squared, _square_interval(eta))
    signed_polynomial = _subtract_interval(
        _subtract_interval(positive, xi_term), eta_term
    )
    signed = _divide_by_positive_interval(
        signed_polynomial, denominator_squared
    )
    return signed if inside else _negate_interval(signed)


def _orientation_at(parameter: Fraction) -> tuple[QComplex, QComplex]:
    denominator = 1 + parameter * parameter
    unit_u = QComplex(
        (1 - parameter * parameter) / denominator,
        2 * parameter / denominator,
    )
    return unit_u, QComplex(-unit_u.imag, unit_u.real)


def _point_score(
    eigenvalue: QComplex,
    inside: bool,
    center: QComplex,
    semiaxis_u: Fraction,
    semiaxis_v: Fraction,
    orientation_parameter: Fraction,
) -> Fraction:
    t = orientation_parameter
    u = 1 - t * t
    v = 2 * t
    denominator = 1 + t * t
    delta = eigenvalue - center
    xi = u * delta.real + v * delta.imag
    eta = -v * delta.real + u * delta.imag
    signed_polynomial = (
        denominator * denominator * semiaxis_u * semiaxis_u
        * semiaxis_v * semiaxis_v
        - semiaxis_v * semiaxis_v * xi * xi
        - semiaxis_u * semiaxis_u * eta * eta
    )
    signed = signed_polynomial / (denominator * denominator)
    return signed if inside else -signed


def _cell_receipt(
    intervals: tuple[QInterval, QInterval, QInterval, QInterval, QInterval],
    eigenvalues: tuple[QComplex, ...],
    labels: tuple[bool, ...],
) -> ContinuousStereographicEllipseCellReceipt:
    center_real, center_imag, semiaxis_u, semiaxis_v, parameter = intervals
    center = QComplex(_midpoint(center_real), _midpoint(center_imag))
    a = _midpoint(semiaxis_u)
    b = _midpoint(semiaxis_v)
    t = _midpoint(parameter)
    unit_u, _ = _orientation_at(t)
    point_scores = tuple(
        _point_score(value, label, center, a, b, t)
        for value, label in zip(eigenvalues, labels, strict=True)
    )
    ranges = tuple(
        _score_interval(
            value,
            label,
            center_real,
            center_imag,
            semiaxis_u,
            semiaxis_v,
            parameter,
        )
        for value, label in zip(eigenvalues, labels, strict=True)
    )
    return ContinuousStereographicEllipseCellReceipt(
        center_real_interval=center_real,
        center_imag_interval=center_imag,
        semiaxis_u_interval=semiaxis_u,
        semiaxis_v_interval=semiaxis_v,
        orientation_parameter_interval=parameter,
        midpoint_center=center,
        midpoint_semiaxis_u=a,
        midpoint_semiaxis_v=b,
        midpoint_orientation_parameter=t,
        midpoint_orientation_unit_u=unit_u,
        midpoint_signed_ellipse_margin=min(point_scores),
        cell_global_upper=min(value[1] for value in ranges),
        per_spectral_value_score_intervals=ranges,
    )


def _split_cell(cell: ContinuousStereographicEllipseCellReceipt):
    intervals = (
        cell.center_real_interval,
        cell.center_imag_interval,
        cell.semiaxis_u_interval,
        cell.semiaxis_v_interval,
        cell.orientation_parameter_interval,
    )
    widths = tuple(interval[1] - interval[0] for interval in intervals)
    axis = max(range(5), key=lambda index: (widths[index], -index))
    if widths[axis] == 0:
        return None
    midpoint = _midpoint(intervals[axis])
    left, right = list(intervals), list(intervals)
    left[axis] = intervals[axis][0], midpoint
    right[axis] = midpoint, intervals[axis][1]
    return tuple(left), tuple(right)


def verified_continuous_stereographic_ellipse_spectral_margin_optimization(
    nominal_transition: object,
    *,
    eigenvectors: object,
    eigenvalues: object,
    target_inside_labels: object,
    center_real_interval: object,
    center_imag_interval: object,
    semiaxis_u_interval: object,
    semiaxis_v_interval: object,
    orientation_parameter_interval: object,
    spectral_reference_scale: object,
    normalized_optimality_tolerance: object,
    maximum_cells: int = 8192,
) -> VerifiedContinuousStereographicEllipseSpectralMarginOptimization:
    """Optimize center, axes, and orientation over one compact exact box."""
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
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    tolerance = _fraction(
        normalized_optimality_tolerance,
        "normalized_optimality_tolerance",
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
    parameter_interval = _interval(
        orientation_parameter_interval, "orientation_parameter_interval"
    )
    if raw_intervals[2][0] <= 0 or raw_intervals[3][0] <= 0:
        raise ValueError("both semiaxis intervals must be strictly positive")
    normalized = tuple(tuple(entry / scale for entry in row) for row in raw)
    normalized_values = tuple(value / scale for value in values)
    normalized_spectral_intervals = tuple(
        (value[0] / scale, value[1] / scale) for value in raw_intervals
    )
    intervals = normalized_spectral_intervals + (parameter_interval,)
    witness = (
        _inverse(vectors) is not None
        and _matmul(normalized, vectors)
        == _matmul(vectors, _diagonal(normalized_values))
    )
    root = _cell_receipt(intervals, normalized_values, labels)
    cells = [root]
    incumbent = root
    evaluated = 1
    budget_failure = False
    while True:
        global_upper = max(cell.cell_global_upper for cell in cells)
        if global_upper - incumbent.midpoint_signed_ellipse_margin <= tolerance:
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
            child = _cell_receipt(child_intervals, normalized_values, labels)
            cells.append(child)
            evaluated += 1
            if child.midpoint_signed_ellipse_margin > incumbent.midpoint_signed_ellipse_margin:
                incumbent = child
    global_upper = max(cell.cell_global_upper for cell in cells)
    gap = global_upper - incumbent.midpoint_signed_ellipse_margin
    failures: list[str] = []
    if not witness:
        failures.append("STEREOGRAPHIC_ELLIPSE_EXACT_DIAGONALIZATION_WITNESS_FAILED")
    if budget_failure or gap > tolerance:
        failures.append("STEREOGRAPHIC_ELLIPSE_OPTIMIZATION_CELL_BUDGET_EXCEEDED")
    if incumbent.midpoint_signed_ellipse_margin <= 0:
        failures.append("STEREOGRAPHIC_ELLIPSE_NO_POSITIVE_TARGET_MARGIN")
    selected_projector = _projector(vectors, labels) if witness else None
    projector_identity = bool(
        selected_projector is not None
        and _matmul(selected_projector, selected_projector) == selected_projector
        and _matmul(normalized, selected_projector)
        == _matmul(selected_projector, normalized)
    )
    if not projector_identity:
        failures.append("STEREOGRAPHIC_ELLIPSE_PROJECTOR_IDENTITY_FAILED")
    selected_unit_u, selected_unit_v = _orientation_at(
        incumbent.midpoint_orientation_parameter
    )
    unit_verified = (
        selected_unit_u.abs_squared() == 1
        and selected_unit_v.abs_squared() == 1
        and selected_unit_u.real * selected_unit_v.real
        + selected_unit_u.imag * selected_unit_v.imag == 0
        and selected_unit_u.real * selected_unit_v.imag
        - selected_unit_u.imag * selected_unit_v.real == 1
    )
    if not unit_verified:
        failures.append("STEREOGRAPHIC_ELLIPSE_SELECTED_ORIENTATION_FAILED")
    success = not failures
    status = (
        "VERIFIED_CONTINUOUS_STEREOGRAPHIC_ELLIPSE_SPECTRAL_MARGIN_OPTIMUM"
        if success
        else failures[0]
    )
    return VerifiedContinuousStereographicEllipseSpectralMarginOptimization(
        status=status,
        validation_level=status if success else None,
        failure_codes=tuple(failures),
        normalized_transition=normalized,
        normalized_eigenvalues=normalized_values,
        target_inside_labels=labels,
        exact_diagonalization_witness_verified=witness,
        normalized_center_real_interval=normalized_spectral_intervals[0],
        normalized_center_imag_interval=normalized_spectral_intervals[1],
        normalized_semiaxis_u_interval=normalized_spectral_intervals[2],
        normalized_semiaxis_v_interval=normalized_spectral_intervals[3],
        orientation_parameter_interval=parameter_interval,
        normalized_optimality_tolerance=tolerance,
        maximum_cells=maximum_cells,
        cells_evaluated=evaluated,
        terminal_cells=tuple(cells),
        selected_normalized_center=incumbent.midpoint_center if success else None,
        selected_normalized_semiaxis_u=incumbent.midpoint_semiaxis_u if success else None,
        selected_normalized_semiaxis_v=incumbent.midpoint_semiaxis_v if success else None,
        selected_orientation_parameter=incumbent.midpoint_orientation_parameter if success else None,
        selected_orientation_unit_u=selected_unit_u if success else None,
        selected_orientation_unit_v=selected_unit_v if success else None,
        selected_signed_ellipse_margin=incumbent.midpoint_signed_ellipse_margin if success else None,
        certified_global_score_upper=global_upper if success else None,
        certified_optimality_gap=gap if success else None,
        epsilon_global_optimality_verified=success and gap <= tolerance,
        positive_target_margin_verified=success and incumbent.midpoint_signed_ellipse_margin > 0,
        selected_projector=selected_projector if success else None,
        selected_projector_rank=sum(labels) if success else None,
        projector_identity_verified=success and projector_identity,
        selected_orientation_unit_verified=success and unit_verified,
        target_rank=sum(labels),
        continuous_parameter_box_covered=not budget_failure and gap <= tolerance,
        finite_grid_only=False,
        orientation_optimized=True,
        empirical_matrix_provenance_verified=False,
    )


__all__ = [
    "ContinuousStereographicEllipseCellReceipt",
    "VerifiedContinuousStereographicEllipseSpectralMarginOptimization",
    "verified_continuous_stereographic_ellipse_spectral_margin_optimization",
]
