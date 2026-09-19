"""Exact box optimization for general automatic-Cq spline coefficients."""

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
class VerifiedContinuousPeriodicSplineCoefficientOptimization:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    knot_optimization: VerifiedContinuousPeriodicSplineKnotOptimization | None
    junction_order: int
    automatic_endpoint_vanishing_order: int
    basis_powers: tuple[int, ...]
    coefficient_intervals: tuple[tuple[QInterval, ...], ...]
    objective_weights: tuple[tuple[Fraction, ...], ...]
    basis_radial_weights: tuple[Fraction, ...]
    basis_gradient_weights: tuple[Fraction, ...]
    normalized_radial_maximum_cap: Fraction
    selected_coefficients: tuple[tuple[Fraction, ...], ...] | None
    selected_admissible_coefficient_box: tuple[tuple[QInterval, ...], ...] | None
    selected_weighted_objective: Fraction | None
    certified_global_weighted_objective_upper: Fraction | None
    per_patch_selected_radial_excess: tuple[Fraction, ...] | None
    original_box_radial_maximum_upper: Fraction
    selected_box_radial_minimum_lower: Fraction | None
    selected_box_radial_maximum_upper: Fraction | None
    selected_box_radial_gradient_upper: Fraction | None
    selected_box_radial_lipschitz_upper: Fraction | None
    exact_continuous_box_linear_optimum_verified: bool
    selected_box_automatic_periodic_cq_junction_verified: bool
    continuous_coefficient_box_covered: bool
    finite_grid_only: bool
    spectral_split_verified: bool
    empirical_matrix_provenance_verified: bool


def _parse_rows(value: object, name: str, *, intervals: bool):
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{name} must be a nonempty rectangular sequence")
    rows = []
    width = None
    for i, row in enumerate(value):
        if not isinstance(row, (tuple, list)) or not row:
            raise ValueError(f"{name}[{i}] must be a nonempty sequence")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError(f"{name} must be rectangular")
        parsed = []
        for j, item in enumerate(row):
            if intervals:
                if not isinstance(item, (tuple, list)) or len(item) != 2:
                    raise ValueError(f"{name}[{i}][{j}] must be an exact pair")
                lower = _fraction(item[0], f"{name}[{i}][{j}].lower")
                upper = _fraction(item[1], f"{name}[{i}][{j}].upper")
                if lower < 0 or lower > upper:
                    raise ValueError("coefficient intervals must be nonnegative and ordered")
                parsed.append((lower, upper))
            else:
                weight = _fraction(item, f"{name}[{i}][{j}]")
                if weight < 0:
                    raise ValueError("objective weights must be nonnegative")
                parsed.append(weight)
        rows.append(tuple(parsed))
    return tuple(rows)


def verified_continuous_periodic_spline_coefficient_optimization(
    *,
    knot_parameter_intervals: object,
    coefficient_intervals: object,
    objective_weights: object,
    basis_powers: object,
    junction_order: int,
    normalized_radial_maximum_cap: object,
    normalized_knot_optimality_tolerance: object,
    maximum_knot_cells: int = 16384,
    sqrt_precision: int = 48,
) -> VerifiedContinuousPeriodicSplineCoefficientOptimization:
    if type(junction_order) is not int or junction_order < 0:
        raise ValueError("junction_order must be a nonnegative built-in integer")
    vanishing = junction_order + 1
    if not isinstance(basis_powers, (tuple, list)) or not basis_powers:
        raise ValueError("basis_powers must be a nonempty sequence")
    if any(type(power) is not int or power < vanishing for power in basis_powers):
        raise ValueError("basis powers must be built-in integers at least junction_order + 1")
    powers = tuple(basis_powers)
    if len(set(powers)) != len(powers):
        raise ValueError("basis powers must be unique")
    intervals = _parse_rows(coefficient_intervals, "coefficient_intervals", intervals=True)
    weights = _parse_rows(objective_weights, "objective_weights", intervals=False)
    if len(intervals) != len(weights) or any(len(a) != len(b) for a, b in zip(intervals, weights, strict=False)):
        raise ValueError("objective_weights must shape-match coefficient_intervals")
    if any(len(row) != len(powers) for row in intervals):
        raise ValueError("basis_powers must match the coefficient mode count")
    cap = _fraction(normalized_radial_maximum_cap, "normalized_radial_maximum_cap")
    if cap < 1:
        raise ValueError("normalized_radial_maximum_cap must be at least one")
    radial_weights = tuple(Fraction(4) ** power for power in powers)
    gradient_weights = tuple(power * radial for power, radial in zip(powers, radial_weights, strict=True))
    capacity = cap - 1
    selected_rows = []
    radial_excesses = []
    feasible = True
    for row, objective in zip(intervals, weights, strict=True):
        chosen = [lower for lower, _ in row]
        used = sum((lower * cost for (lower, _), cost in zip(row, radial_weights, strict=True)), Fraction(0))
        if used > capacity:
            feasible = False
        remaining = capacity - used
        order = sorted(
            range(len(row)),
            key=lambda j: (objective[j] / radial_weights[j], objective[j], -j),
            reverse=True,
        )
        if feasible:
            for j in order:
                room = row[j][1] - chosen[j]
                addition = min(room, remaining / radial_weights[j])
                chosen[j] += addition
                remaining -= addition * radial_weights[j]
        selected_rows.append(tuple(chosen))
        radial_excesses.append(sum((value * cost for value, cost in zip(chosen, radial_weights, strict=True)), Fraction(0)))
    failures: list[str] = []
    if not feasible:
        failures.append("CONTINUOUS_SPLINE_COEFFICIENT_CAP_INFEASIBLE")
    knot = None
    if feasible:
        base_weight = Fraction(4) ** vanishing
        effective_amplitudes = tuple(excess / base_weight for excess in radial_excesses)
        knot = verified_continuous_periodic_spline_knot_optimization(
            knot_parameter_intervals=knot_parameter_intervals,
            patch_amplitudes=effective_amplitudes,
            junction_order=junction_order,
            normalized_optimality_tolerance=normalized_knot_optimality_tolerance,
            maximum_cells=maximum_knot_cells,
            sqrt_precision=sqrt_precision,
        )
        if knot.validation_level is None:
            failures.append("CONTINUOUS_SPLINE_COEFFICIENT_KNOT_OPTIMIZATION_FAILED")
    success = not failures and knot is not None
    selected = tuple(selected_rows)
    objective = sum(
        (value * weight for row, weight_row in zip(selected, weights, strict=True) for value, weight in zip(row, weight_row, strict=True)),
        Fraction(0),
    )
    original_radial = 1 + max(
        sum((upper * cost for (_, upper), cost in zip(row, radial_weights, strict=True)), Fraction(0))
        for row in intervals
    )
    selected_radial = 1 + max(radial_excesses)
    selected_gradient = max(
        sum((value * cost for value, cost in zip(row, gradient_weights, strict=True)), Fraction(0))
        for row in selected
    )
    status = "VERIFIED_CONTINUOUS_PERIODIC_SPLINE_COEFFICIENT_GLOBAL_OPTIMUM" if success else failures[0]
    return VerifiedContinuousPeriodicSplineCoefficientOptimization(
        status=status, validation_level=status if success else None,
        failure_codes=tuple(failures), knot_optimization=knot,
        junction_order=junction_order, automatic_endpoint_vanishing_order=vanishing,
        basis_powers=powers, coefficient_intervals=intervals, objective_weights=weights,
        basis_radial_weights=radial_weights, basis_gradient_weights=gradient_weights,
        normalized_radial_maximum_cap=cap,
        selected_coefficients=selected if success else None,
        selected_admissible_coefficient_box=(
            tuple(tuple((interval[0], value) for interval, value in zip(row, values, strict=True)) for row, values in zip(intervals, selected, strict=True))
            if success else None
        ),
        selected_weighted_objective=objective if success else None,
        certified_global_weighted_objective_upper=objective if success else None,
        per_patch_selected_radial_excess=tuple(radial_excesses) if success else None,
        original_box_radial_maximum_upper=original_radial,
        selected_box_radial_minimum_lower=Fraction(1) if success else None,
        selected_box_radial_maximum_upper=selected_radial if success else None,
        selected_box_radial_gradient_upper=selected_gradient if success else None,
        selected_box_radial_lipschitz_upper=(selected_radial + selected_gradient) if success else None,
        exact_continuous_box_linear_optimum_verified=success,
        selected_box_automatic_periodic_cq_junction_verified=bool(success and knot.automatic_periodic_cq_junction_verified),
        continuous_coefficient_box_covered=success, finite_grid_only=False,
        spectral_split_verified=False, empirical_matrix_provenance_verified=False,
    )


__all__ = ["VerifiedContinuousPeriodicSplineCoefficientOptimization", "verified_continuous_periodic_spline_coefficient_optimization"]
