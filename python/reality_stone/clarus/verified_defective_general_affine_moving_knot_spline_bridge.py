"""Defective general-affine moving-knot bridge via singular-value sandwich."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_algebraic_riesz_projector import (
        VerifiedAlgebraicRieszProjector,
        _operator_norm_certificate,
        verified_algebraic_riesz_projector,
    )
    from .verified_continuous_periodic_spline_knot_optimization import (
        VerifiedContinuousPeriodicSplineKnotOptimization,
        verified_continuous_periodic_spline_knot_optimization,
    )
    from .verified_rational_contour import _dyadic_sqrt, _fraction, _matrix, parse_qcomplex
else:
    from verified_algebraic_riesz_projector import (  # type: ignore[no-redef]
        VerifiedAlgebraicRieszProjector,
        _operator_norm_certificate,
        verified_algebraic_riesz_projector,
    )
    from verified_continuous_periodic_spline_knot_optimization import (  # type: ignore[no-redef]
        VerifiedContinuousPeriodicSplineKnotOptimization,
        verified_continuous_periodic_spline_knot_optimization,
    )
    from verified_rational_contour import _dyadic_sqrt, _fraction, _matrix, parse_qcomplex  # type: ignore[no-redef]


@dataclass(frozen=True)
class VerifiedDefectiveGeneralAffineMovingKnotSplineBridge:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    knot_optimization: VerifiedContinuousPeriodicSplineKnotOptimization
    algebraic_projector: VerifiedAlgebraicRieszProjector
    normalized_axis_u: object
    normalized_axis_v: object
    affine_orientation_determinant: Fraction
    affine_frobenius_norm_squared: Fraction
    affine_minimum_singular_value_squared_lower: Fraction
    affine_minimum_singular_value_lower: Fraction
    affine_operator_norm_upper: Fraction
    singular_value_sqrt_self_checks: tuple[bool, bool, bool, bool]
    uniform_euclidean_inner_radius_lower: Fraction | None
    uniform_euclidean_outer_radius_upper: Fraction | None
    inside_block_norm_upper: Fraction
    exterior_inverse_norm_upper: Fraction
    projector_norm_upper: Fraction
    inside_affine_gap_lower: Fraction | None
    exterior_affine_reciprocal_gap_lower: Fraction | None
    algebraic_uniform_resolvent_norm_upper: Fraction | None
    exact_projector_rank: int | None
    diagonalization_witness_required: bool
    defective_matrix_admitted: bool
    general_shear_affine_verified: bool
    entire_moving_knot_box_rank_preserved: bool
    empirical_matrix_provenance_verified: bool


def verified_defective_general_affine_moving_knot_spline_bridge(
    nominal_transition: object,
    *,
    projector: object,
    exterior_centered_inverse: object,
    center: object,
    axis_u: object,
    axis_v: object,
    spectral_reference_scale: object,
    knot_parameter_intervals: object,
    patch_amplitudes: object,
    junction_order: int,
    normalized_knot_optimality_tolerance: object,
    maximum_knot_cells: int = 16384,
    sqrt_precision: int = 48,
) -> VerifiedDefectiveGeneralAffineMovingKnotSplineBridge:
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    u = parse_qcomplex(axis_u, "axis_u") / scale
    v = parse_qcomplex(axis_v, "axis_v") / scale
    determinant = u.real * v.imag - u.imag * v.real
    if determinant <= 0:
        raise ValueError("affine axes must have strictly positive orientation determinant")
    frobenius_squared = u.abs_squared() + v.abs_squared()
    minimum_squared_lower = determinant * determinant / frobenius_squared
    min_lower, _, min_lower_ok, min_upper_ok = _dyadic_sqrt(
        minimum_squared_lower, sqrt_precision
    )
    _, operator_upper, op_lower_ok, op_upper_ok = _dyadic_sqrt(
        frobenius_squared, sqrt_precision
    )
    knot = verified_continuous_periodic_spline_knot_optimization(
        knot_parameter_intervals=knot_parameter_intervals,
        patch_amplitudes=patch_amplitudes,
        junction_order=junction_order,
        normalized_optimality_tolerance=normalized_knot_optimality_tolerance,
        maximum_cells=maximum_knot_cells,
        sqrt_precision=sqrt_precision,
    )
    rho_min = knot.radial_minimum_lower
    rho_max = knot.radial_maximum_upper
    inner_radius = min_lower * rho_min if rho_min is not None else None
    outer_radius = operator_upper * rho_max if rho_max is not None else None
    reference_radius_raw = (
        scale * (inner_radius + outer_radius) / 2
        if inner_radius is not None and outer_radius is not None
        else scale
    )
    algebraic = verified_algebraic_riesz_projector(
        nominal_transition,
        projector=projector,
        exterior_centered_inverse=exterior_centered_inverse,
        center=center,
        radius=reference_radius_raw,
        spectral_reference_scale=scale,
        sqrt_precision=sqrt_precision,
    )
    inside_norm = algebraic.inside_operator_norm.selected_two_norm_upper
    exterior_norm = algebraic.exterior_inverse_norm.selected_two_norm_upper
    projector_norm = _operator_norm_certificate(
        _matrix(projector, "projector"), sqrt_precision
    ).selected_two_norm_upper
    inside_gap = inner_radius - inside_norm if inner_radius is not None else None
    exterior_gap = (
        1 - outer_radius * exterior_norm if outer_radius is not None else None
    )
    failures: list[str] = []
    if not all((min_lower_ok, min_upper_ok, op_lower_ok, op_upper_ok)):
        failures.append("DEFECTIVE_GENERAL_AFFINE_SQRT_BOUND_FAILED")
    if knot.validation_level is None:
        failures.append("DEFECTIVE_GENERAL_AFFINE_KNOT_GEOMETRY_FAILED")
    if algebraic.validation_level is None:
        failures.append("DEFECTIVE_GENERAL_AFFINE_ALGEBRAIC_PROJECTOR_FAILED")
    if inside_gap is None or inside_gap <= 0:
        failures.append("DEFECTIVE_GENERAL_AFFINE_INSIDE_GAP_NONPOSITIVE")
    if exterior_gap is None or exterior_gap <= 0:
        failures.append("DEFECTIVE_GENERAL_AFFINE_EXTERIOR_GAP_NONPOSITIVE")
    resolvent_upper = None
    if inside_gap is not None and inside_gap > 0 and exterior_gap is not None and exterior_gap > 0:
        resolvent_upper = projector_norm / inside_gap + exterior_norm / exterior_gap
    success = not failures and resolvent_upper is not None
    status = "VERIFIED_DEFECTIVE_GENERAL_AFFINE_MOVING_KNOT_SPLINE_BRIDGE" if success else failures[0]
    return VerifiedDefectiveGeneralAffineMovingKnotSplineBridge(
        status=status, validation_level=status if success else None,
        failure_codes=tuple(failures), knot_optimization=knot,
        algebraic_projector=algebraic, normalized_axis_u=u, normalized_axis_v=v,
        affine_orientation_determinant=determinant,
        affine_frobenius_norm_squared=frobenius_squared,
        affine_minimum_singular_value_squared_lower=minimum_squared_lower,
        affine_minimum_singular_value_lower=min_lower,
        affine_operator_norm_upper=operator_upper,
        singular_value_sqrt_self_checks=(min_lower_ok, min_upper_ok, op_lower_ok, op_upper_ok),
        uniform_euclidean_inner_radius_lower=inner_radius,
        uniform_euclidean_outer_radius_upper=outer_radius,
        inside_block_norm_upper=inside_norm,
        exterior_inverse_norm_upper=exterior_norm,
        projector_norm_upper=projector_norm,
        inside_affine_gap_lower=inside_gap,
        exterior_affine_reciprocal_gap_lower=exterior_gap,
        algebraic_uniform_resolvent_norm_upper=resolvent_upper if success else None,
        exact_projector_rank=algebraic.projector_rank if success else None,
        diagonalization_witness_required=False,
        defective_matrix_admitted=success,
        general_shear_affine_verified=success,
        entire_moving_knot_box_rank_preserved=success,
        empirical_matrix_provenance_verified=False,
    )


__all__ = ["VerifiedDefectiveGeneralAffineMovingKnotSplineBridge", "verified_defective_general_affine_moving_knot_spline_bridge"]
