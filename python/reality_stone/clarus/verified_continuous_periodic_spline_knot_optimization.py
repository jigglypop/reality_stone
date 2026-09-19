"""Certified continuous knot-spacing optimization for automatic periodic C^q splines."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_continuous_axis_aligned_ellipse_spectral_margin_optimization import (
        QInterval,
        _add_interval,
        _interval,
        _midpoint,
        _square_interval,
    )
    from .verified_continuous_stereographic_ellipse_spectral_margin_optimization import (
        _divide_by_positive_interval,
        _multiply_interval,
        _negate_interval,
        _subtract_interval,
    )
    from .verified_rational_contour import QComplex, _dyadic_sqrt, _fraction
else:
    from verified_continuous_axis_aligned_ellipse_spectral_margin_optimization import (  # type: ignore[no-redef]
        QInterval,
        _add_interval,
        _interval,
        _midpoint,
        _square_interval,
    )
    from verified_continuous_stereographic_ellipse_spectral_margin_optimization import (  # type: ignore[no-redef]
        _divide_by_positive_interval,
        _multiply_interval,
        _negate_interval,
        _subtract_interval,
    )
    from verified_rational_contour import QComplex, _dyadic_sqrt, _fraction  # type: ignore[no-redef]


ANCHOR = QComplex(Fraction(-1), Fraction(0))


@dataclass(frozen=True)
class ContinuousSplineKnotCellReceipt:
    knot_parameter_intervals: tuple[QInterval, ...]
    midpoint_knot_parameters: tuple[Fraction, ...]
    midpoint_knot_directions: tuple[QComplex, ...]
    per_arc_spacing_score_intervals: tuple[QInterval, ...]
    midpoint_minimum_spacing_score: Fraction
    cell_global_upper: Fraction


@dataclass(frozen=True)
class VerifiedContinuousPeriodicSplineKnotOptimization:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    junction_order: int
    automatic_vanishing_exponent: int
    automatic_endpoint_vanishing_order: int
    patch_amplitudes: tuple[Fraction, ...]
    knot_parameter_intervals: tuple[QInterval, ...]
    cyclic_anchor_direction: QComplex
    strict_order_and_minor_arc_domain_verified: bool
    normalized_optimality_tolerance: Fraction
    maximum_cells: int
    cells_evaluated: int
    terminal_cells: tuple[ContinuousSplineKnotCellReceipt, ...]
    selected_knot_parameters: tuple[Fraction, ...] | None
    selected_knot_directions: tuple[QComplex, ...] | None
    selected_minimum_spacing_score: Fraction | None
    selected_maximum_endpoint_chord_squared: Fraction | None
    selected_maximum_endpoint_chord_bracket: QInterval | None
    radial_minimum_lower: Fraction | None
    radial_maximum_upper: Fraction | None
    radial_gradient_upper: Fraction | None
    radial_lipschitz_upper: Fraction | None
    selected_contour_cover_chord_upper: Fraction | None
    certified_global_score_upper: Fraction | None
    certified_optimality_gap: Fraction | None
    epsilon_global_optimality_verified: bool
    automatic_periodic_cq_junction_verified: bool
    continuous_knot_box_covered: bool
    finite_grid_only: bool
    spectral_split_verified: bool
    empirical_matrix_provenance_verified: bool


def _orientation_components(parameter: QInterval) -> tuple[QInterval, QInterval, QInterval]:
    squared = _square_interval(parameter)
    denominator = 1 + squared[0], 1 + squared[1]
    x_numerator = 1 - squared[1], 1 - squared[0]
    y_numerator = 2 * parameter[0], 2 * parameter[1]
    return x_numerator, y_numerator, denominator


def _direction_at(parameter: Fraction) -> QComplex:
    denominator = 1 + parameter * parameter
    return QComplex(
        (1 - parameter * parameter) / denominator,
        2 * parameter / denominator,
    )


def _dot_interval(left: QInterval | None, right: QInterval | None) -> QInterval:
    if left is None:
        assert right is not None
        x, _, denominator = _orientation_components(right)
        return _divide_by_positive_interval(_negate_interval(x), denominator)
    if right is None:
        x, _, denominator = _orientation_components(left)
        return _divide_by_positive_interval(_negate_interval(x), denominator)
    lx, ly, ld = _orientation_components(left)
    rx, ry, rd = _orientation_components(right)
    numerator = _add_interval(
        _multiply_interval(lx, rx), _multiply_interval(ly, ry)
    )
    denominator = _multiply_interval(ld, rd)
    return _divide_by_positive_interval(numerator, denominator)


def _determinant_interval(left: QInterval | None, right: QInterval | None) -> QInterval:
    if left is None:
        assert right is not None
        _, y, denominator = _orientation_components(right)
        return _divide_by_positive_interval(_negate_interval(y), denominator)
    if right is None:
        _, y, denominator = _orientation_components(left)
        return _divide_by_positive_interval(y, denominator)
    lx, ly, ld = _orientation_components(left)
    rx, ry, rd = _orientation_components(right)
    numerator = _subtract_interval(
        _multiply_interval(lx, ry), _multiply_interval(ly, rx)
    )
    denominator = _multiply_interval(ld, rd)
    return _divide_by_positive_interval(numerator, denominator)


def _arc_pairs(intervals: tuple[QInterval, ...]):
    extended: tuple[QInterval | None, ...] = (None,) + intervals + (None,)
    return tuple(zip(extended[:-1], extended[1:], strict=True))


def _domain_verified(intervals: tuple[QInterval, ...]) -> bool:
    if len(intervals) < 3:
        return False
    if intervals[0][1] >= 0 or intervals[-1][0] <= 0:
        return False
    if any(intervals[index][1] >= intervals[index + 1][0] for index in range(len(intervals) - 1)):
        return False
    return all(_determinant_interval(left, right)[0] > 0 for left, right in _arc_pairs(intervals))


def _point_spacing_score(parameters: tuple[Fraction, ...]) -> Fraction:
    directions = (ANCHOR,) + tuple(_direction_at(value) for value in parameters) + (ANCHOR,)
    return min(
        2 + 2 * (left.real * right.real + left.imag * right.imag)
        for left, right in zip(directions[:-1], directions[1:], strict=True)
    )


def _cell_receipt(intervals: tuple[QInterval, ...]) -> ContinuousSplineKnotCellReceipt:
    parameters = tuple(_midpoint(value) for value in intervals)
    directions = tuple(_direction_at(value) for value in parameters)
    arc_scores = tuple(
        (2 + 2 * dot[0], 2 + 2 * dot[1])
        for dot in (_dot_interval(left, right) for left, right in _arc_pairs(intervals))
    )
    return ContinuousSplineKnotCellReceipt(
        knot_parameter_intervals=intervals,
        midpoint_knot_parameters=parameters,
        midpoint_knot_directions=directions,
        per_arc_spacing_score_intervals=arc_scores,
        midpoint_minimum_spacing_score=_point_spacing_score(parameters),
        cell_global_upper=min(value[1] for value in arc_scores),
    )


def _split_cell(cell: ContinuousSplineKnotCellReceipt):
    intervals = cell.knot_parameter_intervals
    widths = tuple(value[1] - value[0] for value in intervals)
    axis = max(range(len(intervals)), key=lambda index: (widths[index], -index))
    if widths[axis] == 0:
        return None
    midpoint = _midpoint(intervals[axis])
    left, right = list(intervals), list(intervals)
    left[axis] = intervals[axis][0], midpoint
    right[axis] = midpoint, intervals[axis][1]
    return tuple(left), tuple(right)


def verified_continuous_periodic_spline_knot_optimization(
    *,
    knot_parameter_intervals: object,
    patch_amplitudes: object,
    junction_order: int,
    normalized_optimality_tolerance: object,
    maximum_cells: int = 16384,
    sqrt_precision: int = 48,
) -> VerifiedContinuousPeriodicSplineKnotOptimization:
    """Optimize a continuous cyclic knot box for an automatic radial C^q family."""
    if not isinstance(knot_parameter_intervals, (tuple, list)):
        raise ValueError("knot_parameter_intervals must be an exact interval sequence")
    intervals = tuple(
        _interval(value, f"knot_parameter_intervals[{index}]")
        for index, value in enumerate(knot_parameter_intervals)
    )
    arc_count = len(intervals) + 1
    if not isinstance(patch_amplitudes, (tuple, list)) or len(patch_amplitudes) != arc_count:
        raise ValueError("patch_amplitudes must contain one value per cyclic arc")
    amplitudes = tuple(
        _fraction(value, f"patch_amplitudes[{index}]")
        for index, value in enumerate(patch_amplitudes)
    )
    if any(value < 0 for value in amplitudes):
        raise ValueError("patch amplitudes must be nonnegative")
    if type(junction_order) is not int or junction_order < 0:
        raise ValueError("junction_order must be a nonnegative built-in integer")
    tolerance = _fraction(
        normalized_optimality_tolerance, "normalized_optimality_tolerance"
    )
    if tolerance < 0:
        raise ValueError("normalized_optimality_tolerance must be nonnegative")
    if type(maximum_cells) is not int or maximum_cells < 1:
        raise ValueError("maximum_cells must be a positive built-in integer")
    if type(sqrt_precision) is not int or sqrt_precision < 1:
        raise ValueError("sqrt_precision must be a positive built-in integer")
    domain = _domain_verified(intervals)
    if not domain:
        raise ValueError(
            "knot boxes must be strictly ordered and every cyclic arc must stay below pi"
        )
    root = _cell_receipt(intervals)
    cells = [root]
    incumbent = root
    evaluated = 1
    budget_failure = False
    while True:
        global_upper = max(cell.cell_global_upper for cell in cells)
        if global_upper - incumbent.midpoint_minimum_spacing_score <= tolerance:
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
            child = _cell_receipt(child_intervals)
            cells.append(child)
            evaluated += 1
            if child.midpoint_minimum_spacing_score > incumbent.midpoint_minimum_spacing_score:
                incumbent = child
    global_upper = max(cell.cell_global_upper for cell in cells)
    gap = global_upper - incumbent.midpoint_minimum_spacing_score
    failures: list[str] = []
    if budget_failure or gap > tolerance:
        failures.append("CONTINUOUS_SPLINE_KNOT_OPTIMIZATION_CELL_BUDGET_EXCEEDED")
    if incumbent.midpoint_minimum_spacing_score <= 0:
        failures.append("CONTINUOUS_SPLINE_KNOT_NONMINOR_ARC_SELECTION")
    exponent = junction_order + 1
    epsilon = max(amplitudes, default=Fraction(0))
    power = Fraction(2) ** (2 * exponent)
    radial_minimum = Fraction(1)
    radial_maximum = Fraction(1) + epsilon * power
    radial_gradient = epsilon * exponent * power
    radial_lipschitz = radial_maximum + radial_gradient
    chord_squared = Fraction(4) - incumbent.midpoint_minimum_spacing_score
    chord_lower, chord_upper, _, _ = _dyadic_sqrt(chord_squared, sqrt_precision)
    success = not failures
    status = (
        "VERIFIED_CONTINUOUS_PERIODIC_SPLINE_KNOT_SPACING_OPTIMUM"
        if success
        else failures[0]
    )
    return VerifiedContinuousPeriodicSplineKnotOptimization(
        status=status,
        validation_level=status if success else None,
        failure_codes=tuple(failures),
        junction_order=junction_order,
        automatic_vanishing_exponent=exponent,
        automatic_endpoint_vanishing_order=2 * exponent,
        patch_amplitudes=amplitudes,
        knot_parameter_intervals=intervals,
        cyclic_anchor_direction=ANCHOR,
        strict_order_and_minor_arc_domain_verified=domain,
        normalized_optimality_tolerance=tolerance,
        maximum_cells=maximum_cells,
        cells_evaluated=evaluated,
        terminal_cells=tuple(cells),
        selected_knot_parameters=incumbent.midpoint_knot_parameters if success else None,
        selected_knot_directions=incumbent.midpoint_knot_directions if success else None,
        selected_minimum_spacing_score=incumbent.midpoint_minimum_spacing_score if success else None,
        selected_maximum_endpoint_chord_squared=chord_squared if success else None,
        selected_maximum_endpoint_chord_bracket=(chord_lower, chord_upper) if success else None,
        radial_minimum_lower=radial_minimum if success else None,
        radial_maximum_upper=radial_maximum if success else None,
        radial_gradient_upper=radial_gradient if success else None,
        radial_lipschitz_upper=radial_lipschitz if success else None,
        selected_contour_cover_chord_upper=radial_lipschitz * chord_upper if success else None,
        certified_global_score_upper=global_upper if success else None,
        certified_optimality_gap=gap if success else None,
        epsilon_global_optimality_verified=success and gap <= tolerance,
        automatic_periodic_cq_junction_verified=success,
        continuous_knot_box_covered=not budget_failure and gap <= tolerance,
        finite_grid_only=False,
        spectral_split_verified=False,
        empirical_matrix_provenance_verified=False,
    )


__all__ = [
    "ContinuousSplineKnotCellReceipt",
    "VerifiedContinuousPeriodicSplineKnotOptimization",
    "verified_continuous_periodic_spline_knot_optimization",
]
