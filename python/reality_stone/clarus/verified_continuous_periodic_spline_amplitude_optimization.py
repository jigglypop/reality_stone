"""Exact continuous amplitude-box optimization for automatic-Cq radial splines."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_continuous_periodic_spline_knot_optimization import (
        VerifiedContinuousPeriodicSplineKnotOptimization,
        verified_continuous_periodic_spline_knot_optimization,
    )
    from .verified_rational_contour import _fraction
else:
    from verified_continuous_periodic_spline_knot_optimization import (  # type: ignore[no-redef]
        VerifiedContinuousPeriodicSplineKnotOptimization,
        verified_continuous_periodic_spline_knot_optimization,
    )
    from verified_rational_contour import _fraction  # type: ignore[no-redef]


QInterval = tuple[Fraction, Fraction]


@dataclass(frozen=True)
class VerifiedContinuousPeriodicSplineAmplitudeOptimization:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    knot_optimization: VerifiedContinuousPeriodicSplineKnotOptimization | None
    junction_order: int
    automatic_vanishing_exponent: int
    amplitude_intervals: tuple[QInterval, ...]
    normalized_radial_maximum_cap: Fraction
    per_patch_amplitude_cap: Fraction
    selected_amplitudes: tuple[Fraction, ...] | None
    selected_admissible_amplitude_box: tuple[QInterval, ...] | None
    selected_total_amplitude_objective: Fraction | None
    certified_global_total_amplitude_upper: Fraction | None
    exact_global_amplitude_optimum_verified: bool
    original_box_radial_maximum_upper: Fraction
    selected_box_radial_minimum_lower: Fraction | None
    selected_box_radial_maximum_upper: Fraction | None
    selected_box_radial_gradient_upper: Fraction | None
    selected_box_radial_lipschitz_upper: Fraction | None
    selected_box_automatic_periodic_cq_junction_verified: bool
    continuous_amplitude_box_covered: bool
    finite_grid_only: bool
    spectral_split_verified: bool
    empirical_matrix_provenance_verified: bool


def _intervals(value: object) -> tuple[QInterval, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError("amplitude_intervals must be a nonempty sequence")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"amplitude_intervals[{index}] must be an exact pair")
        lower = _fraction(item[0], f"amplitude_intervals[{index}].lower")
        upper = _fraction(item[1], f"amplitude_intervals[{index}].upper")
        if lower < 0 or lower > upper:
            raise ValueError("amplitude intervals must be nonnegative and ordered")
        result.append((lower, upper))
    return tuple(result)


def verified_continuous_periodic_spline_amplitude_optimization(
    *,
    knot_parameter_intervals: object,
    amplitude_intervals: object,
    junction_order: int,
    normalized_radial_maximum_cap: object,
    normalized_knot_optimality_tolerance: object,
    maximum_knot_cells: int = 16384,
    sqrt_precision: int = 48,
) -> VerifiedContinuousPeriodicSplineAmplitudeOptimization:
    intervals = _intervals(amplitude_intervals)
    if type(junction_order) is not int or junction_order < 0:
        raise ValueError("junction_order must be a nonnegative built-in integer")
    cap = _fraction(normalized_radial_maximum_cap, "normalized_radial_maximum_cap")
    if cap < 1:
        raise ValueError("normalized_radial_maximum_cap must be at least one")
    exponent = junction_order + 1
    power = Fraction(2) ** (2 * exponent)
    amplitude_cap = (cap - 1) / power
    selected = tuple(min(upper, amplitude_cap) for _, upper in intervals)
    feasible = all(lower <= amplitude_cap for lower, _ in intervals)
    original_epsilon = max(upper for _, upper in intervals)
    original_radial_maximum = 1 + original_epsilon * power
    failures: list[str] = []
    if not feasible:
        failures.append("CONTINUOUS_SPLINE_AMPLITUDE_CAP_INFEASIBLE")
    knot = None
    if feasible:
        knot = verified_continuous_periodic_spline_knot_optimization(
            knot_parameter_intervals=knot_parameter_intervals,
            patch_amplitudes=selected,
            junction_order=junction_order,
            normalized_optimality_tolerance=normalized_knot_optimality_tolerance,
            maximum_cells=maximum_knot_cells,
            sqrt_precision=sqrt_precision,
        )
        if knot.validation_level is None:
            failures.append("CONTINUOUS_SPLINE_AMPLITUDE_KNOT_OPTIMIZATION_FAILED")
    success = not failures and knot is not None
    status = (
        "VERIFIED_CONTINUOUS_PERIODIC_SPLINE_AMPLITUDE_GLOBAL_OPTIMUM"
        if success else failures[0]
    )
    objective = sum(selected, Fraction(0)) if feasible else None
    selected_epsilon = max(selected) if feasible else None
    selected_radial_maximum = 1 + selected_epsilon * power if selected_epsilon is not None else None
    selected_gradient = selected_epsilon * exponent * power if selected_epsilon is not None else None
    selected_lipschitz = selected_radial_maximum + selected_gradient if selected_gradient is not None else None
    return VerifiedContinuousPeriodicSplineAmplitudeOptimization(
        status=status, validation_level=status if success else None,
        failure_codes=tuple(failures), knot_optimization=knot,
        junction_order=junction_order, automatic_vanishing_exponent=exponent,
        amplitude_intervals=intervals, normalized_radial_maximum_cap=cap,
        per_patch_amplitude_cap=amplitude_cap,
        selected_amplitudes=selected if success else None,
        selected_admissible_amplitude_box=(
            tuple((lower, chosen) for (lower, _), chosen in zip(intervals, selected, strict=True))
            if success else None
        ),
        selected_total_amplitude_objective=objective if success else None,
        certified_global_total_amplitude_upper=objective if success else None,
        exact_global_amplitude_optimum_verified=success,
        original_box_radial_maximum_upper=original_radial_maximum,
        selected_box_radial_minimum_lower=Fraction(1) if success else None,
        selected_box_radial_maximum_upper=selected_radial_maximum if success else None,
        selected_box_radial_gradient_upper=selected_gradient if success else None,
        selected_box_radial_lipschitz_upper=selected_lipschitz if success else None,
        selected_box_automatic_periodic_cq_junction_verified=bool(success and knot.automatic_periodic_cq_junction_verified),
        continuous_amplitude_box_covered=success,
        finite_grid_only=False,
        spectral_split_verified=False,
        empirical_matrix_provenance_verified=False,
    )


__all__ = ["VerifiedContinuousPeriodicSplineAmplitudeOptimization", "verified_continuous_periodic_spline_amplitude_optimization"]
