"""Certified continuous optimization of general affine ellipse spectral margins."""

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
    from .verified_continuous_stereographic_ellipse_spectral_margin_optimization import (
        _divide_by_positive_interval,
        _multiply_interval,
        _negate_interval,
        _numerator_coordinate_intervals,
        _orientation_at,
        _subtract_interval,
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
    from verified_continuous_stereographic_ellipse_spectral_margin_optimization import (  # type: ignore[no-redef]
        _divide_by_positive_interval,
        _multiply_interval,
        _negate_interval,
        _numerator_coordinate_intervals,
        _orientation_at,
        _subtract_interval,
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
class ContinuousGeneralAffineEllipseCellReceipt:
    center_real_interval: QInterval
    center_imag_interval: QInterval
    semiaxis_u_interval: QInterval
    semiaxis_v_interval: QInterval
    shear_interval: QInterval
    orientation_parameter_interval: QInterval
    midpoint_center: QComplex
    midpoint_semiaxis_u: Fraction
    midpoint_semiaxis_v: Fraction
    midpoint_shear: Fraction
    midpoint_orientation_parameter: Fraction
    midpoint_axis_u: QComplex
    midpoint_axis_v: QComplex
    midpoint_orientation_determinant: Fraction
    midpoint_signed_affine_margin: Fraction
    cell_global_upper: Fraction
    per_spectral_value_score_intervals: tuple[QInterval, ...]


@dataclass(frozen=True)
class VerifiedContinuousGeneralAffineEllipseSpectralMarginOptimization:
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
    normalized_shear_interval: QInterval
    orientation_parameter_interval: QInterval
    normalized_optimality_tolerance: Fraction
    maximum_cells: int
    cells_evaluated: int
    terminal_cells: tuple[ContinuousGeneralAffineEllipseCellReceipt, ...]
    selected_normalized_center: QComplex | None
    selected_normalized_semiaxis_u: Fraction | None
    selected_normalized_semiaxis_v: Fraction | None
    selected_normalized_shear: Fraction | None
    selected_orientation_parameter: Fraction | None
    selected_axis_u: QComplex | None
    selected_axis_v: QComplex | None
    selected_orientation_determinant: Fraction | None
    selected_signed_affine_margin: Fraction | None
    certified_global_score_upper: Fraction | None
    certified_optimality_gap: Fraction | None
    epsilon_global_optimality_verified: bool
    positive_target_margin_verified: bool
    selected_projector: QMatrix | None
    selected_projector_rank: int | None
    projector_identity_verified: bool
    selected_general_affine_geometry_verified: bool
    target_rank: int
    continuous_parameter_box_covered: bool
    finite_grid_only: bool
    orientation_optimized: bool
    shear_optimized: bool
    empirical_matrix_provenance_verified: bool


def _score_interval(
    eigenvalue: QComplex,
    inside: bool,
    center_real: QInterval,
    center_imag: QInterval,
    semiaxis_u: QInterval,
    semiaxis_v: QInterval,
    shear: QInterval,
    orientation_parameter: QInterval,
) -> QInterval:
    xi, eta, denominator_squared = _numerator_coordinate_intervals(
        eigenvalue, center_real, center_imag, orientation_parameter
    )
    a_squared = _square_interval(semiaxis_u)
    b_squared = _square_interval(semiaxis_v)
    area_squared = _nonnegative_product(a_squared, b_squared)
    positive = _nonnegative_product(denominator_squared, area_squared)
    coupled = _subtract_interval(
        _multiply_interval(semiaxis_v, xi),
        _multiply_interval(shear, eta),
    )
    coupled_squared = _square_interval(coupled)
    eta_term = _nonnegative_product(a_squared, _square_interval(eta))
    numerator = _subtract_interval(
        _subtract_interval(positive, coupled_squared), eta_term
    )
    signed = _divide_by_positive_interval(numerator, denominator_squared)
    return signed if inside else _negate_interval(signed)


def _point_geometry(
    semiaxis_u: Fraction,
    semiaxis_v: Fraction,
    shear: Fraction,
    orientation_parameter: Fraction,
) -> tuple[QComplex, QComplex, Fraction]:
    unit_u, unit_v = _orientation_at(orientation_parameter)
    axis_u = unit_u * semiaxis_u
    axis_v = unit_u * shear + unit_v * semiaxis_v
    determinant = axis_u.real * axis_v.imag - axis_u.imag * axis_v.real
    return axis_u, axis_v, determinant


def _point_score(
    eigenvalue: QComplex,
    inside: bool,
    center: QComplex,
    semiaxis_u: Fraction,
    semiaxis_v: Fraction,
    shear: Fraction,
    orientation_parameter: Fraction,
) -> Fraction:
    t = orientation_parameter
    u = 1 - t * t
    v = 2 * t
    denominator = 1 + t * t
    delta = eigenvalue - center
    xi = u * delta.real + v * delta.imag
    eta = -v * delta.real + u * delta.imag
    coupled = semiaxis_v * xi - shear * eta
    numerator = (
        denominator * denominator * semiaxis_u * semiaxis_u
        * semiaxis_v * semiaxis_v
        - coupled * coupled
        - semiaxis_u * semiaxis_u * eta * eta
    )
    signed = numerator / (denominator * denominator)
    return signed if inside else -signed


def _cell_receipt(
    intervals: tuple[
        QInterval, QInterval, QInterval, QInterval, QInterval, QInterval
    ],
    eigenvalues: tuple[QComplex, ...],
    labels: tuple[bool, ...],
) -> ContinuousGeneralAffineEllipseCellReceipt:
    center_real, center_imag, semiaxis_u, semiaxis_v, shear, parameter = intervals
    center = QComplex(_midpoint(center_real), _midpoint(center_imag))
    a = _midpoint(semiaxis_u)
    b = _midpoint(semiaxis_v)
    s = _midpoint(shear)
    t = _midpoint(parameter)
    axis_u, axis_v, determinant = _point_geometry(a, b, s, t)
    point_scores = tuple(
        _point_score(value, label, center, a, b, s, t)
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
            shear,
            parameter,
        )
        for value, label in zip(eigenvalues, labels, strict=True)
    )
    return ContinuousGeneralAffineEllipseCellReceipt(
        center_real_interval=center_real,
        center_imag_interval=center_imag,
        semiaxis_u_interval=semiaxis_u,
        semiaxis_v_interval=semiaxis_v,
        shear_interval=shear,
        orientation_parameter_interval=parameter,
        midpoint_center=center,
        midpoint_semiaxis_u=a,
        midpoint_semiaxis_v=b,
        midpoint_shear=s,
        midpoint_orientation_parameter=t,
        midpoint_axis_u=axis_u,
        midpoint_axis_v=axis_v,
        midpoint_orientation_determinant=determinant,
        midpoint_signed_affine_margin=min(point_scores),
        cell_global_upper=min(value[1] for value in ranges),
        per_spectral_value_score_intervals=ranges,
    )


def _split_cell(cell: ContinuousGeneralAffineEllipseCellReceipt):
    intervals = (
        cell.center_real_interval,
        cell.center_imag_interval,
        cell.semiaxis_u_interval,
        cell.semiaxis_v_interval,
        cell.shear_interval,
        cell.orientation_parameter_interval,
    )
    widths = tuple(interval[1] - interval[0] for interval in intervals)
    axis = max(range(6), key=lambda index: (widths[index], -index))
    if widths[axis] == 0:
        return None
    midpoint = _midpoint(intervals[axis])
    left, right = list(intervals), list(intervals)
    left[axis] = intervals[axis][0], midpoint
    right[axis] = midpoint, intervals[axis][1]
    return tuple(left), tuple(right)


def verified_continuous_general_affine_ellipse_spectral_margin_optimization(
    nominal_transition: object,
    *,
    eigenvectors: object,
    eigenvalues: object,
    target_inside_labels: object,
    center_real_interval: object,
    center_imag_interval: object,
    semiaxis_u_interval: object,
    semiaxis_v_interval: object,
    shear_interval: object,
    orientation_parameter_interval: object,
    spectral_reference_scale: object,
    normalized_optimality_tolerance: object,
    maximum_cells: int = 16384,
) -> VerifiedContinuousGeneralAffineEllipseSpectralMarginOptimization:
    """Optimize one general affine ellipse box with exact outward enclosures."""
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
        normalized_optimality_tolerance, "normalized_optimality_tolerance"
    )
    if scale <= 0 or tolerance < 0:
        raise ValueError("spectral_reference_scale must be positive and tolerance nonnegative")
    if type(maximum_cells) is not int or maximum_cells < 1:
        raise ValueError("maximum_cells must be a positive built-in integer")
    raw_spectral_intervals = (
        _interval(center_real_interval, "center_real_interval"),
        _interval(center_imag_interval, "center_imag_interval"),
        _interval(semiaxis_u_interval, "semiaxis_u_interval"),
        _interval(semiaxis_v_interval, "semiaxis_v_interval"),
        _interval(shear_interval, "shear_interval"),
    )
    parameter_interval = _interval(
        orientation_parameter_interval, "orientation_parameter_interval"
    )
    if raw_spectral_intervals[2][0] <= 0 or raw_spectral_intervals[3][0] <= 0:
        raise ValueError("both semiaxis intervals must be strictly positive")
    normalized = tuple(tuple(entry / scale for entry in row) for row in raw)
    normalized_values = tuple(value / scale for value in values)
    normalized_spectral_intervals = tuple(
        (value[0] / scale, value[1] / scale)
        for value in raw_spectral_intervals
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
        if global_upper - incumbent.midpoint_signed_affine_margin <= tolerance:
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
            if child.midpoint_signed_affine_margin > incumbent.midpoint_signed_affine_margin:
                incumbent = child
    global_upper = max(cell.cell_global_upper for cell in cells)
    gap = global_upper - incumbent.midpoint_signed_affine_margin
    failures: list[str] = []
    if not witness:
        failures.append("GENERAL_AFFINE_ELLIPSE_EXACT_DIAGONALIZATION_WITNESS_FAILED")
    if budget_failure or gap > tolerance:
        failures.append("GENERAL_AFFINE_ELLIPSE_OPTIMIZATION_CELL_BUDGET_EXCEEDED")
    if incumbent.midpoint_signed_affine_margin <= 0:
        failures.append("GENERAL_AFFINE_ELLIPSE_NO_POSITIVE_TARGET_MARGIN")
    selected_projector = _projector(vectors, labels) if witness else None
    projector_identity = bool(
        selected_projector is not None
        and _matmul(selected_projector, selected_projector) == selected_projector
        and _matmul(normalized, selected_projector)
        == _matmul(selected_projector, normalized)
    )
    if not projector_identity:
        failures.append("GENERAL_AFFINE_ELLIPSE_PROJECTOR_IDENTITY_FAILED")
    axis_u, axis_v, determinant = _point_geometry(
        incumbent.midpoint_semiaxis_u,
        incumbent.midpoint_semiaxis_v,
        incumbent.midpoint_shear,
        incumbent.midpoint_orientation_parameter,
    )
    geometry_verified = (
        determinant
        == incumbent.midpoint_semiaxis_u * incumbent.midpoint_semiaxis_v
        and determinant > 0
    )
    if not geometry_verified:
        failures.append("GENERAL_AFFINE_ELLIPSE_SELECTED_GEOMETRY_FAILED")
    success = not failures
    status = (
        "VERIFIED_CONTINUOUS_GENERAL_AFFINE_ELLIPSE_SPECTRAL_MARGIN_OPTIMUM"
        if success
        else failures[0]
    )
    return VerifiedContinuousGeneralAffineEllipseSpectralMarginOptimization(
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
        normalized_shear_interval=normalized_spectral_intervals[4],
        orientation_parameter_interval=parameter_interval,
        normalized_optimality_tolerance=tolerance,
        maximum_cells=maximum_cells,
        cells_evaluated=evaluated,
        terminal_cells=tuple(cells),
        selected_normalized_center=incumbent.midpoint_center if success else None,
        selected_normalized_semiaxis_u=incumbent.midpoint_semiaxis_u if success else None,
        selected_normalized_semiaxis_v=incumbent.midpoint_semiaxis_v if success else None,
        selected_normalized_shear=incumbent.midpoint_shear if success else None,
        selected_orientation_parameter=incumbent.midpoint_orientation_parameter if success else None,
        selected_axis_u=axis_u if success else None,
        selected_axis_v=axis_v if success else None,
        selected_orientation_determinant=determinant if success else None,
        selected_signed_affine_margin=incumbent.midpoint_signed_affine_margin if success else None,
        certified_global_score_upper=global_upper if success else None,
        certified_optimality_gap=gap if success else None,
        epsilon_global_optimality_verified=success and gap <= tolerance,
        positive_target_margin_verified=success and incumbent.midpoint_signed_affine_margin > 0,
        selected_projector=selected_projector if success else None,
        selected_projector_rank=sum(labels) if success else None,
        projector_identity_verified=success and projector_identity,
        selected_general_affine_geometry_verified=success and geometry_verified,
        target_rank=sum(labels),
        continuous_parameter_box_covered=not budget_failure and gap <= tolerance,
        finite_grid_only=False,
        orientation_optimized=True,
        shear_optimized=True,
        empirical_matrix_provenance_verified=False,
    )


__all__ = [
    "ContinuousGeneralAffineEllipseCellReceipt",
    "VerifiedContinuousGeneralAffineEllipseSpectralMarginOptimization",
    "verified_continuous_general_affine_ellipse_spectral_margin_optimization",
]
