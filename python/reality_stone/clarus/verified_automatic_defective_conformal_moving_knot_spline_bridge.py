"""Automatic characteristic-factor witness discovery for DEFCON."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_characteristic_spectral_split_discovery import (
        VerifiedCharacteristicSpectralSplitDiscovery,
        verified_characteristic_spectral_split_discovery,
    )
    from .verified_continuous_periodic_spline_knot_optimization import (
        VerifiedContinuousPeriodicSplineKnotOptimization,
        verified_continuous_periodic_spline_knot_optimization,
    )
    from .verified_defective_conformal_moving_knot_spline_bridge import (
        VerifiedDefectiveConformalMovingKnotSplineBridge,
        verified_defective_conformal_moving_knot_spline_bridge,
    )
    from .verified_rational_contour import _fraction, parse_qcomplex
else:
    from verified_characteristic_spectral_split_discovery import (  # type: ignore[no-redef]
        VerifiedCharacteristicSpectralSplitDiscovery,
        verified_characteristic_spectral_split_discovery,
    )
    from verified_continuous_periodic_spline_knot_optimization import (  # type: ignore[no-redef]
        VerifiedContinuousPeriodicSplineKnotOptimization,
        verified_continuous_periodic_spline_knot_optimization,
    )
    from verified_defective_conformal_moving_knot_spline_bridge import (  # type: ignore[no-redef]
        VerifiedDefectiveConformalMovingKnotSplineBridge,
        verified_defective_conformal_moving_knot_spline_bridge,
    )
    from verified_rational_contour import _fraction, parse_qcomplex  # type: ignore[no-redef]


@dataclass(frozen=True)
class VerifiedAutomaticDefectiveConformalMovingKnotSplineBridge:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    knot_optimization: VerifiedContinuousPeriodicSplineKnotOptimization
    reference_radius_raw: Fraction | None
    characteristic_discovery: VerifiedCharacteristicSpectralSplitDiscovery | None
    defective_bridge: VerifiedDefectiveConformalMovingKnotSplineBridge | None
    projector_automatically_constructed: bool
    exterior_inverse_automatically_constructed: bool
    characteristic_factorization_automatically_discovered: bool
    exact_projector_rank: int | None
    diagonalization_witness_required: bool
    supplied_projector_or_inverse_required: bool
    general_shear_affine_verified: bool
    empirical_matrix_provenance_verified: bool


def verified_automatic_defective_conformal_moving_knot_spline_bridge(
    nominal_transition: object,
    *,
    center: object,
    conformal_scale: object,
    conformal_unit: object,
    spectral_reference_scale: object,
    knot_parameter_intervals: object,
    patch_amplitudes: object,
    junction_order: int,
    normalized_knot_optimality_tolerance: object,
    maximum_knot_cells: int = 16384,
    maximum_partitions: int = 1024,
    maximum_factor_candidate_value_tuples: int = 1_000_000,
    sqrt_precision: int = 48,
) -> VerifiedAutomaticDefectiveConformalMovingKnotSplineBridge:
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    conformal_scale_raw = _fraction(conformal_scale, "conformal_scale")
    parse_qcomplex(center, "center")
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
    reference_radius = (
        conformal_scale_raw * (rho_min + rho_max) / 2
        if rho_min is not None and rho_max is not None
        else None
    )
    failures: list[str] = []
    if scale <= 0 or conformal_scale_raw <= 0:
        raise ValueError("spectral_reference_scale and conformal_scale must be positive")
    if knot.validation_level is None or reference_radius is None:
        failures.append("AUTOMATIC_DEFECTIVE_CONFORMAL_KNOT_GEOMETRY_FAILED")
    discovery = None
    construction = None
    defective = None
    if not failures and reference_radius is not None:
        discovery = verified_characteristic_spectral_split_discovery(
            nominal_transition,
            center=center,
            radius=reference_radius,
            spectral_reference_scale=scale,
            maximum_partitions=maximum_partitions,
            maximum_factor_candidate_value_tuples=maximum_factor_candidate_value_tuples,
            sqrt_precision=sqrt_precision,
        )
        construction = discovery.selected_construction
        if discovery.validation_level is None or construction is None:
            failures.append("AUTOMATIC_DEFECTIVE_CONFORMAL_CHARACTERISTIC_DISCOVERY_FAILED")
        elif (
            construction.constructed_projector is None
            or construction.constructed_exterior_centered_inverse is None
        ):
            failures.append("AUTOMATIC_DEFECTIVE_CONFORMAL_WITNESS_CONSTRUCTION_FAILED")
        else:
            defective = verified_defective_conformal_moving_knot_spline_bridge(
                nominal_transition,
                projector=construction.constructed_projector,
                exterior_centered_inverse=construction.constructed_exterior_centered_inverse,
                center=center,
                conformal_scale=conformal_scale_raw,
                conformal_unit=conformal_unit,
                spectral_reference_scale=scale,
                knot_parameter_intervals=knot_parameter_intervals,
                patch_amplitudes=patch_amplitudes,
                junction_order=junction_order,
                normalized_knot_optimality_tolerance=normalized_knot_optimality_tolerance,
                maximum_knot_cells=maximum_knot_cells,
                sqrt_precision=sqrt_precision,
            )
            if defective.validation_level is None:
                failures.append("AUTOMATIC_DEFECTIVE_CONFORMAL_FINAL_BRIDGE_FAILED")
    success = not failures and discovery is not None and construction is not None and defective is not None
    status = (
        "VERIFIED_AUTOMATIC_DEFECTIVE_CONFORMAL_MOVING_KNOT_SPLINE_BRIDGE"
        if success
        else failures[0]
    )
    return VerifiedAutomaticDefectiveConformalMovingKnotSplineBridge(
        status=status,
        validation_level=status if success else None,
        failure_codes=tuple(failures),
        knot_optimization=knot,
        reference_radius_raw=reference_radius,
        characteristic_discovery=discovery,
        defective_bridge=defective,
        projector_automatically_constructed=bool(success and construction.projector_automatically_constructed),
        exterior_inverse_automatically_constructed=bool(success and construction.exterior_inverse_automatically_constructed),
        characteristic_factorization_automatically_discovered=bool(success and discovery.characteristic_factorization_automatically_discovered),
        exact_projector_rank=defective.exact_projector_rank if success else None,
        diagonalization_witness_required=False,
        supplied_projector_or_inverse_required=False,
        general_shear_affine_verified=False,
        empirical_matrix_provenance_verified=False,
    )


__all__ = [
    "VerifiedAutomaticDefectiveConformalMovingKnotSplineBridge",
    "verified_automatic_defective_conformal_moving_knot_spline_bridge",
]
