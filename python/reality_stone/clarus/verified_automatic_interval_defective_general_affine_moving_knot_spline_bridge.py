"""Automatic and interval successor for defective general-affine knot families."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_characteristic_spectral_split_discovery import (
        VerifiedCharacteristicSpectralSplitDiscovery,
        verified_characteristic_spectral_split_discovery,
    )
    from .verified_continuous_periodic_spline_knot_optimization import verified_continuous_periodic_spline_knot_optimization
    from .verified_defective_general_affine_moving_knot_spline_bridge import (
        VerifiedDefectiveGeneralAffineMovingKnotSplineBridge,
        verified_defective_general_affine_moving_knot_spline_bridge,
    )
    from .verified_rational_contour import _dyadic_sqrt, _fraction, parse_qcomplex
else:
    from verified_characteristic_spectral_split_discovery import (  # type: ignore[no-redef]
        VerifiedCharacteristicSpectralSplitDiscovery,
        verified_characteristic_spectral_split_discovery,
    )
    from verified_continuous_periodic_spline_knot_optimization import verified_continuous_periodic_spline_knot_optimization  # type: ignore[no-redef]
    from verified_defective_general_affine_moving_knot_spline_bridge import (  # type: ignore[no-redef]
        VerifiedDefectiveGeneralAffineMovingKnotSplineBridge,
        verified_defective_general_affine_moving_knot_spline_bridge,
    )
    from verified_rational_contour import _dyadic_sqrt, _fraction, parse_qcomplex  # type: ignore[no-redef]


@dataclass(frozen=True)
class VerifiedAutomaticIntervalDefectiveGeneralAffineMovingKnotSplineBridge:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    characteristic_discovery: VerifiedCharacteristicSpectralSplitDiscovery | None
    general_affine_bridge: VerifiedDefectiveGeneralAffineMovingKnotSplineBridge | None
    reference_radius_raw: Fraction | None
    projector_automatically_constructed: bool
    exterior_inverse_automatically_constructed: bool
    normalized_matrix_uncertainty_frobenius_upper: Fraction
    nominal_algebraic_resolvent_norm_upper: Fraction | None
    neumann_product_upper: Fraction | None
    neumann_margin_lower: Fraction | None
    perturbed_algebraic_resolvent_norm_upper: Fraction | None
    normalized_contour_length_over_two_pi_upper: Fraction | None
    interval_projector_perturbation_norm_upper: Fraction | None
    exact_projector_rank: int | None
    automatic_general_shear_interval_rank_verified: bool
    diagonalization_witness_required: bool
    supplied_projector_or_inverse_required: bool
    empirical_matrix_provenance_verified: bool


def verified_automatic_interval_defective_general_affine_moving_knot_spline_bridge(
    nominal_transition: object,
    *,
    center: object,
    axis_u: object,
    axis_v: object,
    spectral_reference_scale: object,
    matrix_uncertainty_frobenius_upper: object,
    knot_parameter_intervals: object,
    patch_amplitudes: object,
    junction_order: int,
    normalized_knot_optimality_tolerance: object,
    maximum_knot_cells: int = 16384,
    maximum_partitions: int = 1024,
    maximum_factor_candidate_value_tuples: int = 1_000_000,
    sqrt_precision: int = 48,
) -> VerifiedAutomaticIntervalDefectiveGeneralAffineMovingKnotSplineBridge:
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    raw_uncertainty = _fraction(matrix_uncertainty_frobenius_upper, "matrix_uncertainty_frobenius_upper")
    if scale <= 0 or raw_uncertainty < 0:
        raise ValueError("spectral scale must be positive and uncertainty nonnegative")
    uncertainty = raw_uncertainty / scale
    u = parse_qcomplex(axis_u, "axis_u") / scale
    v = parse_qcomplex(axis_v, "axis_v") / scale
    determinant = u.real * v.imag - u.imag * v.real
    if determinant <= 0:
        raise ValueError("affine axes must have strictly positive orientation determinant")
    frobenius_squared = u.abs_squared() + v.abs_squared()
    min_lower, _, _, _ = _dyadic_sqrt(determinant * determinant / frobenius_squared, sqrt_precision)
    _, operator_upper, _, _ = _dyadic_sqrt(frobenius_squared, sqrt_precision)
    knot = verified_continuous_periodic_spline_knot_optimization(
        knot_parameter_intervals=knot_parameter_intervals, patch_amplitudes=patch_amplitudes,
        junction_order=junction_order,
        normalized_optimality_tolerance=normalized_knot_optimality_tolerance,
        maximum_cells=maximum_knot_cells, sqrt_precision=sqrt_precision,
    )
    rho_min, rho_max = knot.radial_minimum_lower, knot.radial_maximum_upper
    reference_radius = (
        scale * (min_lower * rho_min + operator_upper * rho_max) / 2
        if rho_min is not None and rho_max is not None else None
    )
    failures: list[str] = []
    discovery = None
    construction = None
    bridge = None
    if knot.validation_level is None or reference_radius is None:
        failures.append("AUTO_INTERVAL_GADEF_KNOT_GEOMETRY_FAILED")
    else:
        discovery = verified_characteristic_spectral_split_discovery(
            nominal_transition, center=center, radius=reference_radius,
            spectral_reference_scale=scale, maximum_partitions=maximum_partitions,
            maximum_factor_candidate_value_tuples=maximum_factor_candidate_value_tuples,
            sqrt_precision=sqrt_precision,
        )
        construction = discovery.selected_construction
        if discovery.validation_level is None or construction is None:
            failures.append("AUTO_INTERVAL_GADEF_CHARACTERISTIC_DISCOVERY_FAILED")
        elif construction.constructed_projector is None or construction.constructed_exterior_centered_inverse is None:
            failures.append("AUTO_INTERVAL_GADEF_WITNESS_CONSTRUCTION_FAILED")
        else:
            bridge = verified_defective_general_affine_moving_knot_spline_bridge(
                nominal_transition, projector=construction.constructed_projector,
                exterior_centered_inverse=construction.constructed_exterior_centered_inverse,
                center=center, axis_u=axis_u, axis_v=axis_v,
                spectral_reference_scale=scale,
                knot_parameter_intervals=knot_parameter_intervals,
                patch_amplitudes=patch_amplitudes, junction_order=junction_order,
                normalized_knot_optimality_tolerance=normalized_knot_optimality_tolerance,
                maximum_knot_cells=maximum_knot_cells, sqrt_precision=sqrt_precision,
            )
            if bridge.validation_level is None:
                failures.append("AUTO_INTERVAL_GADEF_FINAL_NOMINAL_BRIDGE_FAILED")
    resolvent = bridge.algebraic_uniform_resolvent_norm_upper if bridge is not None else None
    product = resolvent * uncertainty if resolvent is not None else None
    margin = 1 - product if product is not None else None
    if margin is None or margin <= 0:
        failures.append("AUTO_INTERVAL_GADEF_NEUMANN_MARGIN_NONPOSITIVE")
    perturbed = None
    length_over_two_pi = None
    projector_bound = None
    if not failures and bridge is not None and resolvent is not None and margin is not None:
        perturbed = resolvent / margin
        radial_lipschitz = bridge.knot_optimization.radial_lipschitz_upper
        if radial_lipschitz is None:
            failures.append("AUTO_INTERVAL_GADEF_LENGTH_BOUND_FAILED")
        else:
            length_over_two_pi = bridge.affine_operator_norm_upper * radial_lipschitz
            projector_bound = length_over_two_pi * uncertainty * resolvent * resolvent / margin
    success = not failures and bridge is not None and construction is not None
    status = "VERIFIED_AUTOMATIC_INTERVAL_DEFECTIVE_GENERAL_AFFINE_MOVING_KNOT_SPLINE_BRIDGE" if success else failures[0]
    return VerifiedAutomaticIntervalDefectiveGeneralAffineMovingKnotSplineBridge(
        status=status, validation_level=status if success else None,
        failure_codes=tuple(failures), characteristic_discovery=discovery,
        general_affine_bridge=bridge, reference_radius_raw=reference_radius,
        projector_automatically_constructed=bool(success and construction.projector_automatically_constructed),
        exterior_inverse_automatically_constructed=bool(success and construction.exterior_inverse_automatically_constructed),
        normalized_matrix_uncertainty_frobenius_upper=uncertainty,
        nominal_algebraic_resolvent_norm_upper=resolvent,
        neumann_product_upper=product, neumann_margin_lower=margin,
        perturbed_algebraic_resolvent_norm_upper=perturbed,
        normalized_contour_length_over_two_pi_upper=length_over_two_pi,
        interval_projector_perturbation_norm_upper=projector_bound,
        exact_projector_rank=bridge.exact_projector_rank if success else None,
        automatic_general_shear_interval_rank_verified=success,
        diagonalization_witness_required=False,
        supplied_projector_or_inverse_required=False,
        empirical_matrix_provenance_verified=False,
    )


__all__ = ["VerifiedAutomaticIntervalDefectiveGeneralAffineMovingKnotSplineBridge", "verified_automatic_interval_defective_general_affine_moving_knot_spline_bridge"]
