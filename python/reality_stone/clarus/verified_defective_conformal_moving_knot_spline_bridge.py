"""Defective conformal moving-knot bridge without diagonalization."""

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
    from .verified_rational_contour import _fraction, _matrix, parse_qcomplex
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
    from verified_rational_contour import _fraction, _matrix, parse_qcomplex  # type: ignore[no-redef]


@dataclass(frozen=True)
class VerifiedDefectiveConformalMovingKnotSplineBridge:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    knot_optimization: VerifiedContinuousPeriodicSplineKnotOptimization
    algebraic_projector: VerifiedAlgebraicRieszProjector
    normalized_conformal_scale: Fraction
    conformal_unit_squared_norm: Fraction
    uniform_inner_radius_lower: Fraction | None
    uniform_outer_radius_upper: Fraction | None
    inside_block_norm_upper: Fraction
    exterior_inverse_norm_upper: Fraction
    projector_norm_upper: Fraction
    inside_algebraic_gap_lower: Fraction | None
    exterior_reciprocal_gap_lower: Fraction | None
    algebraic_uniform_resolvent_norm_upper: Fraction | None
    exact_projector_rank: int | None
    diagonalization_witness_required: bool
    defective_matrix_admitted: bool
    entire_moving_knot_box_rank_preserved: bool
    general_shear_affine_verified: bool
    empirical_matrix_provenance_verified: bool


def verified_defective_conformal_moving_knot_spline_bridge(
    nominal_transition: object,
    *,
    projector: object,
    exterior_centered_inverse: object,
    center: object,
    conformal_scale: object,
    conformal_unit: object,
    spectral_reference_scale: object,
    knot_parameter_intervals: object,
    patch_amplitudes: object,
    junction_order: int,
    normalized_knot_optimality_tolerance: object,
    maximum_knot_cells: int = 16384,
    sqrt_precision: int = 48,
) -> VerifiedDefectiveConformalMovingKnotSplineBridge:
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    conformal_scale_raw = _fraction(conformal_scale, "conformal_scale")
    if scale <= 0 or conformal_scale_raw <= 0:
        raise ValueError("spectral_reference_scale and conformal_scale must be positive")
    unit = parse_qcomplex(conformal_unit, "conformal_unit")
    unit_squared = unit.abs_squared()
    if unit_squared != 1:
        raise ValueError("conformal_unit must have exact squared norm one")
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
    if rho_min is None or rho_max is None:
        reference_radius_raw = conformal_scale_raw
    else:
        reference_radius_raw = conformal_scale_raw * (rho_min + rho_max) / 2
    algebraic = verified_algebraic_riesz_projector(
        nominal_transition,
        projector=projector,
        exterior_centered_inverse=exterior_centered_inverse,
        center=center,
        radius=reference_radius_raw,
        spectral_reference_scale=scale,
        sqrt_precision=sqrt_precision,
    )
    normalized_conformal_scale = conformal_scale_raw / scale
    inside_norm = algebraic.inside_operator_norm.selected_two_norm_upper
    exterior_norm = algebraic.exterior_inverse_norm.selected_two_norm_upper
    projector_norm = _operator_norm_certificate(
        _matrix(projector, "projector"), sqrt_precision
    ).selected_two_norm_upper
    inner_radius = (
        normalized_conformal_scale * rho_min if rho_min is not None else None
    )
    outer_radius = (
        normalized_conformal_scale * rho_max if rho_max is not None else None
    )
    inside_gap = inner_radius - inside_norm if inner_radius is not None else None
    exterior_gap = (
        1 - outer_radius * exterior_norm if outer_radius is not None else None
    )
    failures: list[str] = []
    if knot.validation_level is None:
        failures.append("DEFECTIVE_CONFORMAL_KNOT_GEOMETRY_FAILED")
    if algebraic.validation_level is None:
        failures.append("DEFECTIVE_CONFORMAL_ALGEBRAIC_PROJECTOR_FAILED")
    if inside_gap is None or inside_gap <= 0:
        failures.append("DEFECTIVE_CONFORMAL_INSIDE_GAP_NONPOSITIVE")
    if exterior_gap is None or exterior_gap <= 0:
        failures.append("DEFECTIVE_CONFORMAL_EXTERIOR_GAP_NONPOSITIVE")
    resolvent_upper = None
    if inside_gap is not None and inside_gap > 0 and exterior_gap is not None and exterior_gap > 0:
        resolvent_upper = (
            projector_norm / inside_gap + exterior_norm / exterior_gap
        )
    success = not failures and resolvent_upper is not None
    status = (
        "VERIFIED_DEFECTIVE_CONFORMAL_MOVING_KNOT_SPLINE_BRIDGE"
        if success
        else failures[0]
    )
    return VerifiedDefectiveConformalMovingKnotSplineBridge(
        status=status,
        validation_level=status if success else None,
        failure_codes=tuple(failures),
        knot_optimization=knot,
        algebraic_projector=algebraic,
        normalized_conformal_scale=normalized_conformal_scale,
        conformal_unit_squared_norm=unit_squared,
        uniform_inner_radius_lower=inner_radius,
        uniform_outer_radius_upper=outer_radius,
        inside_block_norm_upper=inside_norm,
        exterior_inverse_norm_upper=exterior_norm,
        projector_norm_upper=projector_norm,
        inside_algebraic_gap_lower=inside_gap,
        exterior_reciprocal_gap_lower=exterior_gap,
        algebraic_uniform_resolvent_norm_upper=resolvent_upper if success else None,
        exact_projector_rank=algebraic.projector_rank if success else None,
        diagonalization_witness_required=False,
        defective_matrix_admitted=success,
        entire_moving_knot_box_rank_preserved=success,
        general_shear_affine_verified=False,
        empirical_matrix_provenance_verified=False,
    )


__all__ = [
    "VerifiedDefectiveConformalMovingKnotSplineBridge",
    "verified_defective_conformal_moving_knot_spline_bridge",
]
