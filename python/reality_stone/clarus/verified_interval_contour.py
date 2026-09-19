"""Exact perturbation bridge from a rational nominal contour certificate.

The unknown matrix may have arbitrary complex entries.  What is exact here is
the declared rational rectangular uncertainty box and the proof that every
matrix in that box satisfies the returned bounds.  This module does not infer
or calibrate uncertainty radii from observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_rational_contour import (
        VerifiedRationalCircle,
        VerifiedRationalStrip,
        _dyadic_sqrt,
        _fraction,
        _matrix,
        verified_rational_analytic_strip,
        verified_rational_circle,
    )
else:  # Standalone focused-test loading without importing the large package.
    from verified_rational_contour import (  # type: ignore[no-redef]
        VerifiedRationalCircle,
        VerifiedRationalStrip,
        _dyadic_sqrt,
        _fraction,
        _matrix,
        verified_rational_analytic_strip,
        verified_rational_circle,
    )


@dataclass(frozen=True)
class VerifiedIntervalCircleBridge:
    status: str
    validation_level: str | None
    rank_preserved_for_entire_family: bool
    nominal: VerifiedRationalCircle
    normalized_uncertainty_squared: Fraction
    normalized_uncertainty_sqrt_bracket: tuple[Fraction, Fraction]
    uncertainty_sqrt_self_checks: tuple[bool, bool]
    normalized_uncertainty_upper: Fraction
    raw_uncertainty_upper: Fraction
    normalized_robust_delta_lower: Fraction | None
    raw_robust_delta_lower: Fraction | None
    normalized_robust_resolvent_upper: Fraction | None
    raw_robust_resolvent_upper: Fraction | None
    projector_perturbation_upper: Fraction | None
    spectral_reference_scale: Fraction
    sqrt_precision: int


@dataclass(frozen=True)
class VerifiedIntervalProjectorBridge:
    status: str
    validation_level: str | None
    circle_bridge: VerifiedIntervalCircleBridge
    nominal_strip: VerifiedRationalStrip
    nominal_quadrature_error_upper: Fraction | None
    uncertainty_projector_error_upper: Fraction | None
    total_projector_error_upper: Fraction | None


def _rectangular_uncertainty_squared(
    value: object, *, n: int, scale: Fraction
) -> Fraction:
    if not isinstance(value, (tuple, list)) or len(value) != n:
        raise ValueError("uncertainty_radii must shape-match the nominal square matrix")
    total = Fraction(0)
    for i, row in enumerate(value):
        if not isinstance(row, (tuple, list)) or len(row) != n:
            raise ValueError("uncertainty_radii must shape-match the nominal square matrix")
        for j, entry in enumerate(row):
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError(
                    f"uncertainty_radii[{i}][{j}] must be a (real_radius, imag_radius) pair"
                )
            real_radius = _fraction(entry[0], f"uncertainty_radii[{i}][{j}].real")
            imag_radius = _fraction(entry[1], f"uncertainty_radii[{i}][{j}].imag")
            if real_radius < 0 or imag_radius < 0:
                raise ValueError("uncertainty radii must be nonnegative")
            normalized_real = real_radius / scale
            normalized_imag = imag_radius / scale
            total += normalized_real * normalized_real + normalized_imag * normalized_imag
    return total


def verified_interval_family_circle(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    nodes: int = 4,
    sqrt_precision: int = 32,
) -> VerifiedIntervalCircleBridge:
    """Certify one circle uniformly over a rectangular matrix uncertainty box."""
    matrix = _matrix(nominal_transition, "nominal_transition")
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    if type(sqrt_precision) is not int or sqrt_precision < 0:
        raise ValueError("square-root precision must be a nonnegative built-in integer")
    uncertainty_squared = _rectangular_uncertainty_squared(
        uncertainty_radii, n=len(matrix), scale=scale
    )
    sqrt_lower, sqrt_upper, lower_ok, upper_ok = _dyadic_sqrt(
        uncertainty_squared, sqrt_precision
    )
    nominal = verified_rational_circle(
        nominal_transition,
        center=center,
        radius=radius,
        spectral_reference_scale=scale,
        nodes=nodes,
        sqrt_precision=sqrt_precision,
    )
    common = dict(
        nominal=nominal,
        normalized_uncertainty_squared=uncertainty_squared,
        normalized_uncertainty_sqrt_bracket=(sqrt_lower, sqrt_upper),
        uncertainty_sqrt_self_checks=(lower_ok, upper_ok),
        normalized_uncertainty_upper=sqrt_upper,
        raw_uncertainty_upper=sqrt_upper * scale,
        spectral_reference_scale=scale,
        sqrt_precision=sqrt_precision,
    )
    if nominal.validation_level is None or nominal.normalized_delta_lower is None:
        return VerifiedIntervalCircleBridge(
            status="VERIFIED_NOMINAL_CIRCLE_CERTIFICATE_UNAVAILABLE",
            validation_level=None,
            rank_preserved_for_entire_family=False,
            normalized_robust_delta_lower=None,
            raw_robust_delta_lower=None,
            normalized_robust_resolvent_upper=None,
            raw_robust_resolvent_upper=None,
            projector_perturbation_upper=None,
            **common,
        )
    nominal_delta = nominal.normalized_delta_lower
    robust_delta = nominal_delta - sqrt_upper
    if robust_delta <= 0:
        return VerifiedIntervalCircleBridge(
            status="VERIFIED_INTERVAL_UNCERTAINTY_NOT_BELOW_MARGIN",
            validation_level=None,
            rank_preserved_for_entire_family=False,
            normalized_robust_delta_lower=robust_delta,
            raw_robust_delta_lower=robust_delta * scale,
            normalized_robust_resolvent_upper=None,
            raw_robust_resolvent_upper=None,
            projector_perturbation_upper=None,
            **common,
        )
    normalized_radius = _fraction(radius, "radius") / scale
    projector_bound = (
        normalized_radius * sqrt_upper / (nominal_delta * robust_delta)
    )
    normalized_resolvent = Fraction(1) / robust_delta
    return VerifiedIntervalCircleBridge(
        status="VERIFIED_RATIONAL_INTERVAL_FAMILY_CONTOUR_BRIDGE",
        validation_level="VERIFIED_RATIONAL_INTERVAL_FAMILY_CONTOUR_BRIDGE",
        rank_preserved_for_entire_family=True,
        normalized_robust_delta_lower=robust_delta,
        raw_robust_delta_lower=robust_delta * scale,
        normalized_robust_resolvent_upper=normalized_resolvent,
        raw_robust_resolvent_upper=normalized_resolvent / scale,
        projector_perturbation_upper=projector_bound,
        **common,
    )


def verified_interval_family_projector(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    expansion_factor: object,
    eigenvectors: object,
    eigenvalues: object,
    nodes: int = 4,
    sqrt_precision: int = 32,
) -> VerifiedIntervalProjectorBridge:
    """Bound each exact family projector against the nominal four-node P4."""
    circle_bridge = verified_interval_family_circle(
        nominal_transition,
        uncertainty_radii=uncertainty_radii,
        center=center,
        radius=radius,
        spectral_reference_scale=spectral_reference_scale,
        nodes=nodes,
        sqrt_precision=sqrt_precision,
    )
    nominal_strip = verified_rational_analytic_strip(
        nominal_transition,
        center=center,
        radius=radius,
        spectral_reference_scale=spectral_reference_scale,
        expansion_factor=expansion_factor,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        nodes=nodes,
        sqrt_precision=sqrt_precision,
    )
    if circle_bridge.validation_level is None:
        return VerifiedIntervalProjectorBridge(
            status=circle_bridge.status,
            validation_level=None,
            circle_bridge=circle_bridge,
            nominal_strip=nominal_strip,
            nominal_quadrature_error_upper=nominal_strip.error_upper,
            uncertainty_projector_error_upper=None,
            total_projector_error_upper=None,
        )
    if nominal_strip.validation_level is None or nominal_strip.error_upper is None:
        return VerifiedIntervalProjectorBridge(
            status="VERIFIED_NOMINAL_STRIP_CERTIFICATE_UNAVAILABLE",
            validation_level=None,
            circle_bridge=circle_bridge,
            nominal_strip=nominal_strip,
            nominal_quadrature_error_upper=None,
            uncertainty_projector_error_upper=circle_bridge.projector_perturbation_upper,
            total_projector_error_upper=None,
        )
    assert circle_bridge.projector_perturbation_upper is not None
    total = circle_bridge.projector_perturbation_upper + nominal_strip.error_upper
    return VerifiedIntervalProjectorBridge(
        status="VERIFIED_RATIONAL_INTERVAL_FAMILY_PROJECTOR_BRIDGE",
        validation_level="VERIFIED_RATIONAL_INTERVAL_FAMILY_PROJECTOR_BRIDGE",
        circle_bridge=circle_bridge,
        nominal_strip=nominal_strip,
        nominal_quadrature_error_upper=nominal_strip.error_upper,
        uncertainty_projector_error_upper=circle_bridge.projector_perturbation_upper,
        total_projector_error_upper=total,
    )


__all__ = [
    "VerifiedIntervalCircleBridge",
    "VerifiedIntervalProjectorBridge",
    "verified_interval_family_circle",
    "verified_interval_family_projector",
]
