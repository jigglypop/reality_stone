"""Interval perturbation bridge for defective conformal moving-knot families."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_defective_conformal_moving_knot_spline_bridge import (
        VerifiedDefectiveConformalMovingKnotSplineBridge,
        verified_defective_conformal_moving_knot_spline_bridge,
    )
    from .verified_rational_contour import _fraction
else:
    from verified_defective_conformal_moving_knot_spline_bridge import (  # type: ignore[no-redef]
        VerifiedDefectiveConformalMovingKnotSplineBridge,
        verified_defective_conformal_moving_knot_spline_bridge,
    )
    from verified_rational_contour import _fraction  # type: ignore[no-redef]


@dataclass(frozen=True)
class VerifiedIntervalDefectiveConformalMovingKnotSplineBridge:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    nominal_defective_bridge: VerifiedDefectiveConformalMovingKnotSplineBridge
    normalized_matrix_uncertainty_frobenius_upper: Fraction
    nominal_algebraic_resolvent_norm_upper: Fraction | None
    neumann_product_upper: Fraction | None
    neumann_margin_lower: Fraction | None
    perturbed_algebraic_resolvent_norm_upper: Fraction | None
    normalized_contour_length_over_two_pi_upper: Fraction | None
    interval_projector_perturbation_norm_upper: Fraction | None
    defective_interval_matrix_family_resolvent_verified: bool
    defective_interval_matrix_family_rank_preserved: bool
    diagonalization_witness_required: bool
    general_shear_affine_verified: bool
    empirical_matrix_provenance_verified: bool


def verified_interval_defective_conformal_moving_knot_spline_bridge(
    nominal_transition: object,
    *,
    matrix_uncertainty_frobenius_upper: object,
    **nominal_kwargs: object,
) -> VerifiedIntervalDefectiveConformalMovingKnotSplineBridge:
    nominal = verified_defective_conformal_moving_knot_spline_bridge(
        nominal_transition, **nominal_kwargs
    )
    scale = _fraction(
        nominal_kwargs.get("spectral_reference_scale"), "spectral_reference_scale"
    )
    raw_uncertainty = _fraction(
        matrix_uncertainty_frobenius_upper,
        "matrix_uncertainty_frobenius_upper",
    )
    if raw_uncertainty < 0:
        raise ValueError("matrix_uncertainty_frobenius_upper must be nonnegative")
    uncertainty = raw_uncertainty / scale
    resolvent = nominal.algebraic_uniform_resolvent_norm_upper
    product = resolvent * uncertainty if resolvent is not None else None
    margin = 1 - product if product is not None else None
    failures: list[str] = []
    if nominal.validation_level is None:
        failures.append("INTERVAL_DEFECTIVE_CONFORMAL_NOMINAL_BRIDGE_FAILED")
    if margin is None or margin <= 0:
        failures.append("INTERVAL_DEFECTIVE_CONFORMAL_NEUMANN_MARGIN_NONPOSITIVE")
    perturbed_resolvent = None
    length_over_two_pi = None
    projector_bound = None
    if not failures and resolvent is not None and margin is not None:
        perturbed_resolvent = resolvent / margin
        radial_lipschitz = nominal.knot_optimization.radial_lipschitz_upper
        if radial_lipschitz is None:
            failures.append("INTERVAL_DEFECTIVE_CONFORMAL_LENGTH_BOUND_FAILED")
        else:
            length_over_two_pi = (
                nominal.normalized_conformal_scale * radial_lipschitz
            )
            projector_bound = (
                length_over_two_pi
                * uncertainty
                * resolvent
                * resolvent
                / margin
            )
    success = not failures
    status = (
        "VERIFIED_INTERVAL_DEFECTIVE_CONFORMAL_MOVING_KNOT_SPLINE_BRIDGE"
        if success
        else failures[0]
    )
    return VerifiedIntervalDefectiveConformalMovingKnotSplineBridge(
        status=status,
        validation_level=status if success else None,
        failure_codes=tuple(failures),
        nominal_defective_bridge=nominal,
        normalized_matrix_uncertainty_frobenius_upper=uncertainty,
        nominal_algebraic_resolvent_norm_upper=resolvent,
        neumann_product_upper=product,
        neumann_margin_lower=margin,
        perturbed_algebraic_resolvent_norm_upper=perturbed_resolvent,
        normalized_contour_length_over_two_pi_upper=length_over_two_pi,
        interval_projector_perturbation_norm_upper=projector_bound,
        defective_interval_matrix_family_resolvent_verified=success,
        defective_interval_matrix_family_rank_preserved=success,
        diagonalization_witness_required=False,
        general_shear_affine_verified=False,
        empirical_matrix_provenance_verified=False,
    )


__all__ = [
    "VerifiedIntervalDefectiveConformalMovingKnotSplineBridge",
    "verified_interval_defective_conformal_moving_knot_spline_bridge",
]
