"""Automatic defective/shear spectral amplitude optimization with interval control."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_automatic_interval_defective_general_affine_moving_knot_spline_bridge import (
        VerifiedAutomaticIntervalDefectiveGeneralAffineMovingKnotSplineBridge,
        verified_automatic_interval_defective_general_affine_moving_knot_spline_bridge,
    )
    from .verified_continuous_periodic_spline_amplitude_optimization import (
        VerifiedContinuousPeriodicSplineAmplitudeOptimization,
        verified_continuous_periodic_spline_amplitude_optimization,
    )
    from .verified_defective_general_affine_moving_knot_spline_bridge import (
        VerifiedDefectiveGeneralAffineMovingKnotSplineBridge,
        verified_defective_general_affine_moving_knot_spline_bridge,
    )
    from .verified_rational_contour import _fraction
else:
    from verified_automatic_interval_defective_general_affine_moving_knot_spline_bridge import (  # type: ignore[no-redef]
        VerifiedAutomaticIntervalDefectiveGeneralAffineMovingKnotSplineBridge,
        verified_automatic_interval_defective_general_affine_moving_knot_spline_bridge,
    )
    from verified_continuous_periodic_spline_amplitude_optimization import (  # type: ignore[no-redef]
        VerifiedContinuousPeriodicSplineAmplitudeOptimization,
        verified_continuous_periodic_spline_amplitude_optimization,
    )
    from verified_defective_general_affine_moving_knot_spline_bridge import (  # type: ignore[no-redef]
        VerifiedDefectiveGeneralAffineMovingKnotSplineBridge,
        verified_defective_general_affine_moving_knot_spline_bridge,
    )
    from verified_rational_contour import _fraction  # type: ignore[no-redef]


@dataclass(frozen=True)
class VerifiedDefectiveAlgebraicSplineAmplitudeOptimization:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    seed_automatic_discovery: VerifiedAutomaticIntervalDefectiveGeneralAffineMovingKnotSplineBridge | None
    amplitude_optimization: VerifiedContinuousPeriodicSplineAmplitudeOptimization | None
    final_general_affine_bridge: VerifiedDefectiveGeneralAffineMovingKnotSplineBridge | None
    normalized_exterior_safety_margin: Fraction
    normalized_radial_cap_ceiling: Fraction
    algebraic_derived_radial_cap: Fraction | None
    selected_amplitudes: tuple[Fraction, ...] | None
    selected_admissible_amplitude_box: tuple[tuple[Fraction, Fraction], ...] | None
    selected_total_amplitude_objective: Fraction | None
    normalized_matrix_uncertainty_frobenius_upper: Fraction
    final_algebraic_resolvent_norm_upper: Fraction | None
    interval_neumann_product_upper: Fraction | None
    interval_neumann_margin_lower: Fraction | None
    perturbed_algebraic_resolvent_norm_upper: Fraction | None
    interval_projector_perturbation_norm_upper: Fraction | None
    exact_projector_rank: int | None
    entire_amplitude_knot_matrix_product_family_verified: bool
    diagonalization_witness_required: bool
    supplied_projector_or_inverse_required: bool
    finite_grid_only: bool
    empirical_matrix_provenance_verified: bool


def verified_defective_algebraic_spline_amplitude_optimization(
    nominal_transition: object,
    *,
    center: object,
    axis_u: object,
    axis_v: object,
    spectral_reference_scale: object,
    amplitude_intervals: object,
    normalized_exterior_safety_margin: object,
    normalized_radial_cap_ceiling: object,
    matrix_uncertainty_frobenius_upper: object,
    knot_parameter_intervals: object,
    junction_order: int,
    normalized_knot_optimality_tolerance: object,
    maximum_knot_cells: int = 16384,
    maximum_partitions: int = 1024,
    maximum_factor_candidate_value_tuples: int = 1_000_000,
    sqrt_precision: int = 48,
) -> VerifiedDefectiveAlgebraicSplineAmplitudeOptimization:
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    safety = _fraction(normalized_exterior_safety_margin, "normalized_exterior_safety_margin")
    ceiling = _fraction(normalized_radial_cap_ceiling, "normalized_radial_cap_ceiling")
    raw_uncertainty = _fraction(matrix_uncertainty_frobenius_upper, "matrix_uncertainty_frobenius_upper")
    if scale <= 0 or safety <= 0 or safety >= 1 or ceiling < 1 or raw_uncertainty < 0:
        raise ValueError("require positive scale, safety in (0,1), ceiling >=1, and nonnegative uncertainty")
    if not isinstance(amplitude_intervals, (tuple, list)) or not amplitude_intervals:
        raise ValueError("amplitude_intervals must be nonempty")
    if not isinstance(knot_parameter_intervals, (tuple, list)):
        raise ValueError("knot_parameter_intervals must be a sequence")
    zero_amplitudes = (Fraction(0),) * (len(knot_parameter_intervals) + 1)
    seed = verified_automatic_interval_defective_general_affine_moving_knot_spline_bridge(
        nominal_transition, center=center, axis_u=axis_u, axis_v=axis_v,
        spectral_reference_scale=scale, matrix_uncertainty_frobenius_upper=0,
        knot_parameter_intervals=knot_parameter_intervals,
        patch_amplitudes=zero_amplitudes, junction_order=junction_order,
        normalized_knot_optimality_tolerance=normalized_knot_optimality_tolerance,
        maximum_knot_cells=maximum_knot_cells, maximum_partitions=maximum_partitions,
        maximum_factor_candidate_value_tuples=maximum_factor_candidate_value_tuples,
        sqrt_precision=sqrt_precision,
    )
    failures: list[str] = []
    if seed.validation_level is None or seed.general_affine_bridge is None or seed.characteristic_discovery is None:
        failures.append("DEFECTIVE_ALGEBRAIC_AMPLITUDE_SEED_DISCOVERY_FAILED")
    derived_cap = None
    amplitude = None
    final_bridge = None
    construction = None
    if not failures:
        seed_bridge = seed.general_affine_bridge
        construction = seed.characteristic_discovery.selected_construction
        exterior_norm = seed_bridge.exterior_inverse_norm_upper
        affine_upper = seed_bridge.affine_operator_norm_upper
        if exterior_norm == 0:
            algebraic_cap = ceiling
        else:
            algebraic_cap = (1 - safety) / (affine_upper * exterior_norm)
        derived_cap = min(ceiling, algebraic_cap)
        if derived_cap < 1:
            failures.append("DEFECTIVE_ALGEBRAIC_AMPLITUDE_CAP_BELOW_UNIT_CORE")
        elif construction is None or construction.constructed_projector is None or construction.constructed_exterior_centered_inverse is None:
            failures.append("DEFECTIVE_ALGEBRAIC_AMPLITUDE_WITNESS_MISSING")
        else:
            amplitude = verified_continuous_periodic_spline_amplitude_optimization(
                knot_parameter_intervals=knot_parameter_intervals,
                amplitude_intervals=amplitude_intervals, junction_order=junction_order,
                normalized_radial_maximum_cap=derived_cap,
                normalized_knot_optimality_tolerance=normalized_knot_optimality_tolerance,
                maximum_knot_cells=maximum_knot_cells, sqrt_precision=sqrt_precision,
            )
            if amplitude.validation_level is None or amplitude.selected_amplitudes is None:
                failures.append("DEFECTIVE_ALGEBRAIC_AMPLITUDE_OPTIMIZATION_FAILED")
            else:
                final_bridge = verified_defective_general_affine_moving_knot_spline_bridge(
                    nominal_transition, projector=construction.constructed_projector,
                    exterior_centered_inverse=construction.constructed_exterior_centered_inverse,
                    center=center, axis_u=axis_u, axis_v=axis_v,
                    spectral_reference_scale=scale,
                    knot_parameter_intervals=knot_parameter_intervals,
                    patch_amplitudes=amplitude.selected_amplitudes,
                    junction_order=junction_order,
                    normalized_knot_optimality_tolerance=normalized_knot_optimality_tolerance,
                    maximum_knot_cells=maximum_knot_cells, sqrt_precision=sqrt_precision,
                )
                if final_bridge.validation_level is None:
                    failures.append("DEFECTIVE_ALGEBRAIC_AMPLITUDE_FINAL_BRIDGE_FAILED")
    uncertainty = raw_uncertainty / scale
    resolvent = final_bridge.algebraic_uniform_resolvent_norm_upper if final_bridge is not None else None
    product = resolvent * uncertainty if resolvent is not None else None
    margin = 1 - product if product is not None else None
    if margin is None or margin <= 0:
        failures.append("DEFECTIVE_ALGEBRAIC_AMPLITUDE_INTERVAL_NEUMANN_MARGIN_NONPOSITIVE")
    perturbed = None
    projector_bound = None
    if not failures and final_bridge is not None and resolvent is not None and margin is not None:
        perturbed = resolvent / margin
        length_factor = final_bridge.affine_operator_norm_upper * final_bridge.knot_optimization.radial_lipschitz_upper
        projector_bound = length_factor * uncertainty * resolvent * resolvent / margin
    success = not failures and amplitude is not None and final_bridge is not None
    status = "VERIFIED_DEFECTIVE_ALGEBRAIC_SPLINE_AMPLITUDE_GLOBAL_OPTIMUM" if success else failures[0]
    return VerifiedDefectiveAlgebraicSplineAmplitudeOptimization(
        status=status, validation_level=status if success else None,
        failure_codes=tuple(failures), seed_automatic_discovery=seed,
        amplitude_optimization=amplitude, final_general_affine_bridge=final_bridge,
        normalized_exterior_safety_margin=safety,
        normalized_radial_cap_ceiling=ceiling,
        algebraic_derived_radial_cap=derived_cap,
        selected_amplitudes=amplitude.selected_amplitudes if success else None,
        selected_admissible_amplitude_box=amplitude.selected_admissible_amplitude_box if success else None,
        selected_total_amplitude_objective=amplitude.selected_total_amplitude_objective if success else None,
        normalized_matrix_uncertainty_frobenius_upper=uncertainty,
        final_algebraic_resolvent_norm_upper=resolvent,
        interval_neumann_product_upper=product,
        interval_neumann_margin_lower=margin,
        perturbed_algebraic_resolvent_norm_upper=perturbed,
        interval_projector_perturbation_norm_upper=projector_bound,
        exact_projector_rank=final_bridge.exact_projector_rank if success else None,
        entire_amplitude_knot_matrix_product_family_verified=success,
        diagonalization_witness_required=False,
        supplied_projector_or_inverse_required=False,
        finite_grid_only=False,
        empirical_matrix_provenance_verified=False,
    )


__all__ = ["VerifiedDefectiveAlgebraicSplineAmplitudeOptimization", "verified_defective_algebraic_spline_amplitude_optimization"]
