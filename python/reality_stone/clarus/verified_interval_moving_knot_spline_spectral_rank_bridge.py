"""Interval-matrix successor for the moving-knot spline spectral bridge."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_moving_knot_spline_spectral_rank_bridge import (
        VerifiedMovingKnotSplineSpectralRankBridge,
        verified_moving_knot_spline_spectral_rank_bridge,
    )
    from .verified_rational_contour import _dyadic_sqrt, _fraction
else:
    from verified_moving_knot_spline_spectral_rank_bridge import (  # type: ignore[no-redef]
        VerifiedMovingKnotSplineSpectralRankBridge,
        verified_moving_knot_spline_spectral_rank_bridge,
    )
    from verified_rational_contour import _dyadic_sqrt, _fraction  # type: ignore[no-redef]


@dataclass(frozen=True)
class VerifiedIntervalMovingKnotSplineSpectralRankBridge:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    nominal_bridge: VerifiedMovingKnotSplineSpectralRankBridge
    normalized_matrix_uncertainty_frobenius_upper: Fraction
    nominal_uniform_resolvent_norm_upper: Fraction | None
    neumann_product_upper: Fraction | None
    neumann_margin_lower: Fraction | None
    perturbed_uniform_resolvent_norm_upper: Fraction | None
    affine_frobenius_norm_upper: Fraction | None
    normalized_contour_length_over_two_pi_upper: Fraction | None
    interval_projector_perturbation_norm_upper: Fraction | None
    interval_matrix_family_resolvent_verified: bool
    interval_matrix_family_rank_preserved: bool
    interval_matrix_family_verified: bool
    defective_or_no_witness_family_verified: bool
    empirical_matrix_provenance_verified: bool


def verified_interval_moving_knot_spline_spectral_rank_bridge(
    nominal_transition: object,
    *,
    matrix_uncertainty_frobenius_upper: object,
    **nominal_kwargs: object,
) -> VerifiedIntervalMovingKnotSplineSpectralRankBridge:
    """Certify every matrix in one Frobenius ball on every moving contour."""
    nominal = verified_moving_knot_spline_spectral_rank_bridge(
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
    resolvent = nominal.quantitative_resolvent_norm_bound
    product = resolvent * uncertainty if resolvent is not None else None
    margin = 1 - product if product is not None else None
    failures: list[str] = []
    if nominal.validation_level is None:
        failures.append("INTERVAL_MOVING_KNOT_NOMINAL_BRIDGE_FAILED")
    if margin is None or margin <= 0:
        failures.append("INTERVAL_MOVING_KNOT_NEUMANN_MARGIN_NONPOSITIVE")

    perturbed_resolvent = None
    affine_frobenius_upper = None
    length_over_two_pi = None
    projector_bound = None
    if not failures and resolvent is not None and margin is not None:
        perturbed_resolvent = resolvent / margin
        _, affine_frobenius_upper, lower_ok, upper_ok = _dyadic_sqrt(
            nominal.affine_frobenius_norm_squared,
            nominal_kwargs.get("sqrt_precision", 48),
        )
        if not lower_ok or not upper_ok:
            failures.append("INTERVAL_MOVING_KNOT_AFFINE_SQRT_BOUND_FAILED")
        else:
            radial_lipschitz = nominal.knot_optimization.radial_lipschitz_upper
            if radial_lipschitz is None:
                failures.append("INTERVAL_MOVING_KNOT_LENGTH_BOUND_FAILED")
            else:
                length_over_two_pi = affine_frobenius_upper * radial_lipschitz
                projector_bound = (
                    length_over_two_pi
                    * uncertainty
                    * resolvent
                    * resolvent
                    / margin
                )
    success = not failures
    status = (
        "VERIFIED_INTERVAL_MOVING_KNOT_SPLINE_SPECTRAL_RANK_BRIDGE"
        if success
        else failures[0]
    )
    return VerifiedIntervalMovingKnotSplineSpectralRankBridge(
        status=status,
        validation_level=status if success else None,
        failure_codes=tuple(failures),
        nominal_bridge=nominal,
        normalized_matrix_uncertainty_frobenius_upper=uncertainty,
        nominal_uniform_resolvent_norm_upper=resolvent,
        neumann_product_upper=product,
        neumann_margin_lower=margin,
        perturbed_uniform_resolvent_norm_upper=perturbed_resolvent,
        affine_frobenius_norm_upper=affine_frobenius_upper,
        normalized_contour_length_over_two_pi_upper=length_over_two_pi,
        interval_projector_perturbation_norm_upper=projector_bound,
        interval_matrix_family_resolvent_verified=success,
        interval_matrix_family_rank_preserved=success,
        interval_matrix_family_verified=success,
        defective_or_no_witness_family_verified=False,
        empirical_matrix_provenance_verified=False,
    )


__all__ = [
    "VerifiedIntervalMovingKnotSplineSpectralRankBridge",
    "verified_interval_moving_knot_spline_spectral_rank_bridge",
]
