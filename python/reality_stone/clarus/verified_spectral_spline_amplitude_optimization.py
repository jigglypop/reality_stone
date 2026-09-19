"""Spectral-derived continuous amplitude optimization for moving-knot splines."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_continuous_periodic_spline_amplitude_optimization import (
        VerifiedContinuousPeriodicSplineAmplitudeOptimization,
        verified_continuous_periodic_spline_amplitude_optimization,
    )
    from .verified_moving_knot_spline_spectral_rank_bridge import (
        VerifiedMovingKnotSplineSpectralRankBridge,
        _affine_inverse_coordinate_squared,
        verified_moving_knot_spline_spectral_rank_bridge,
    )
    from .verified_rational_contour import _dyadic_sqrt, _fraction, parse_qcomplex
else:
    from verified_continuous_periodic_spline_amplitude_optimization import (  # type: ignore[no-redef]
        VerifiedContinuousPeriodicSplineAmplitudeOptimization,
        verified_continuous_periodic_spline_amplitude_optimization,
    )
    from verified_moving_knot_spline_spectral_rank_bridge import (  # type: ignore[no-redef]
        VerifiedMovingKnotSplineSpectralRankBridge,
        _affine_inverse_coordinate_squared,
        verified_moving_knot_spline_spectral_rank_bridge,
    )
    from verified_rational_contour import _dyadic_sqrt, _fraction, parse_qcomplex  # type: ignore[no-redef]


@dataclass(frozen=True)
class VerifiedSpectralSplineAmplitudeOptimization:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    normalized_affine_inverse_coordinate_squared: tuple[Fraction, ...]
    per_outside_eigenvalue_radius_lower: tuple[Fraction, ...]
    normalized_radial_safety_margin: Fraction
    normalized_radial_cap_ceiling: Fraction
    spectral_derived_radial_cap: Fraction
    amplitude_optimization: VerifiedContinuousPeriodicSplineAmplitudeOptimization | None
    spectral_bridge: VerifiedMovingKnotSplineSpectralRankBridge | None
    selected_amplitudes: tuple[Fraction, ...] | None
    selected_admissible_amplitude_box: tuple[tuple[Fraction, Fraction], ...] | None
    selected_total_amplitude_objective: Fraction | None
    exact_global_amplitude_optimum_verified: bool
    entire_selected_amplitude_and_knot_box_spectral_split_verified: bool
    exact_projector_rank: int | None
    quantitative_resolvent_norm_bound: Fraction | None
    finite_grid_only: bool
    empirical_matrix_provenance_verified: bool


def verified_spectral_spline_amplitude_optimization(
    nominal_transition: object,
    *,
    eigenvectors: object,
    eigenvalues: object,
    target_inside_labels: object,
    center: object,
    axis_u: object,
    axis_v: object,
    spectral_reference_scale: object,
    amplitude_intervals: object,
    normalized_radial_safety_margin: object,
    normalized_radial_cap_ceiling: object,
    knot_parameter_intervals: object,
    junction_order: int,
    normalized_knot_optimality_tolerance: object,
    maximum_knot_cells: int = 16384,
    sqrt_precision: int = 48,
) -> VerifiedSpectralSplineAmplitudeOptimization:
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    safety = _fraction(normalized_radial_safety_margin, "normalized_radial_safety_margin")
    ceiling = _fraction(normalized_radial_cap_ceiling, "normalized_radial_cap_ceiling")
    if scale <= 0 or safety <= 0 or ceiling < 1:
        raise ValueError("scale/safety must be positive and radial cap ceiling at least one")
    if not isinstance(eigenvalues, (tuple, list)) or not isinstance(target_inside_labels, (tuple, list)):
        raise ValueError("eigenvalues and labels must be sequences")
    if len(eigenvalues) != len(target_inside_labels) or not eigenvalues:
        raise ValueError("eigenvalues and labels must have the same positive length")
    if any(type(label) is not bool for label in target_inside_labels):
        raise ValueError("target_inside_labels entries must be built-in booleans")
    values = tuple(parse_qcomplex(value, f"eigenvalues[{i}]") / scale for i, value in enumerate(eigenvalues))
    c = parse_qcomplex(center, "center") / scale
    u = parse_qcomplex(axis_u, "axis_u") / scale
    v = parse_qcomplex(axis_v, "axis_v") / scale
    determinant = u.real * v.imag - u.imag * v.real
    if determinant <= 0:
        raise ValueError("affine axes must have strictly positive orientation determinant")
    radii_squared = tuple(_affine_inverse_coordinate_squared(value, c, u, v, determinant) for value in values)
    outside_lowers = []
    for squared, inside in zip(radii_squared, target_inside_labels, strict=True):
        if not inside:
            lower, _, lower_ok, upper_ok = _dyadic_sqrt(squared, sqrt_precision)
            if not lower_ok or not upper_ok:
                raise ValueError("outside eigenvalue square-root enclosure failed")
            outside_lowers.append(lower)
    outside_cap = min(outside_lowers) - safety if outside_lowers else ceiling
    derived_cap = min(ceiling, outside_cap)
    failures: list[str] = []
    if any(squared >= 1 for squared, inside in zip(radii_squared, target_inside_labels, strict=True) if inside):
        failures.append("SPECTRAL_AMPLITUDE_INSIDE_UNIT_CORE_VIOLATED")
    if derived_cap < 1:
        failures.append("SPECTRAL_AMPLITUDE_RADIAL_CAP_BELOW_UNIT_CORE")
    amplitude = None
    bridge = None
    if not failures:
        amplitude = verified_continuous_periodic_spline_amplitude_optimization(
            knot_parameter_intervals=knot_parameter_intervals,
            amplitude_intervals=amplitude_intervals,
            junction_order=junction_order,
            normalized_radial_maximum_cap=derived_cap,
            normalized_knot_optimality_tolerance=normalized_knot_optimality_tolerance,
            maximum_knot_cells=maximum_knot_cells,
            sqrt_precision=sqrt_precision,
        )
        if amplitude.validation_level is None or amplitude.selected_amplitudes is None:
            failures.append("SPECTRAL_AMPLITUDE_GEOMETRY_OPTIMIZATION_FAILED")
        else:
            bridge = verified_moving_knot_spline_spectral_rank_bridge(
                nominal_transition, eigenvectors=eigenvectors, eigenvalues=eigenvalues,
                target_inside_labels=target_inside_labels, center=center,
                axis_u=axis_u, axis_v=axis_v, spectral_reference_scale=scale,
                knot_parameter_intervals=knot_parameter_intervals,
                patch_amplitudes=amplitude.selected_amplitudes,
                junction_order=junction_order,
                normalized_knot_optimality_tolerance=normalized_knot_optimality_tolerance,
                maximum_knot_cells=maximum_knot_cells, sqrt_precision=sqrt_precision,
            )
            if bridge.validation_level is None:
                failures.append("SPECTRAL_AMPLITUDE_FINAL_SPECTRAL_BRIDGE_FAILED")
    success = not failures and amplitude is not None and bridge is not None
    status = "VERIFIED_SPECTRAL_SPLINE_AMPLITUDE_GLOBAL_OPTIMUM" if success else failures[0]
    return VerifiedSpectralSplineAmplitudeOptimization(
        status=status, validation_level=status if success else None,
        failure_codes=tuple(failures),
        normalized_affine_inverse_coordinate_squared=radii_squared,
        per_outside_eigenvalue_radius_lower=tuple(outside_lowers),
        normalized_radial_safety_margin=safety,
        normalized_radial_cap_ceiling=ceiling,
        spectral_derived_radial_cap=derived_cap,
        amplitude_optimization=amplitude, spectral_bridge=bridge,
        selected_amplitudes=amplitude.selected_amplitudes if success else None,
        selected_admissible_amplitude_box=amplitude.selected_admissible_amplitude_box if success else None,
        selected_total_amplitude_objective=amplitude.selected_total_amplitude_objective if success else None,
        exact_global_amplitude_optimum_verified=bool(success and amplitude.exact_global_amplitude_optimum_verified),
        entire_selected_amplitude_and_knot_box_spectral_split_verified=success,
        exact_projector_rank=bridge.selected_projector_rank if success else None,
        quantitative_resolvent_norm_bound=bridge.quantitative_resolvent_norm_bound if success else None,
        finite_grid_only=False,
        empirical_matrix_provenance_verified=False,
    )


__all__ = ["VerifiedSpectralSplineAmplitudeOptimization", "verified_spectral_spline_amplitude_optimization"]
