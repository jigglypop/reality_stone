"""Exact physical-scale map for a homogeneous scalar gradient-flow chart."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re


def _exact_fraction(value: object, name: str) -> Fraction:
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError(f"{name} must be an exact integer, Fraction, or canonical rational string")
    if isinstance(value, Fraction):
        return value
    if type(value) is int:
        return Fraction(value)
    if isinstance(value, str):
        decimal = re.fullmatch(r"-?(?:0|[1-9][0-9]*)\.[0-9]*[1-9]", value)
        rational = re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:/(?:[1-9][0-9]*))?", value)
        if decimal is None and rational is None:
            raise ValueError(f"{name} is not a canonical rational string")
        try:
            parsed = Fraction(value)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(f"{name} is not a canonical rational string") from error
        if parsed == 0 and value.startswith("-"):
            raise ValueError(f"{name} is not a canonical rational string")
        if rational is not None and str(parsed) != value:
            raise ValueError(f"{name} is not a canonical rational string")
        return parsed
    raise ValueError(f"{name} must be an exact integer, Fraction, or canonical rational string")


@dataclass(frozen=True)
class PhysicalScaleMobilityCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    state_reference_scale: Fraction
    energy_reference_scale: Fraction
    time_reference_scale: Fraction
    physical_mobility: Fraction
    dimensionless_window: Fraction
    physical_window: Fraction
    normalized_mobility: Fraction
    velocity_reference_scale: Fraction
    power_reference_scale: Fraction
    dimensionless_gradient_norm_upper: Fraction
    physical_speed_upper: Fraction
    euclidean_dissipation_upper: Fraction
    contraction_factor_upper: Fraction
    logarithm_series_terms: int
    logarithm_lower: Fraction | None
    logarithm_upper: Fraction | None
    decay_rate_lower: Fraction | None
    decay_rate_upper: Fraction | None


def _negative_log_enclosure(q: Fraction, terms: int) -> tuple[Fraction, Fraction]:
    """Enclose -log(q) using the positive atanh series; require 0 < q < 1."""
    z = (1 - q) / (1 + q)
    lower = 2 * sum(
        (z ** (2 * k + 1)) / (2 * k + 1)
        for k in range(terms)
    )
    tail = 2 * z ** (2 * terms + 1) / ((2 * terms + 1) * (1 - z * z))
    return lower, lower + tail


def physical_scale_mobility_certificate(
    *,
    state_reference_scale: object,
    energy_reference_scale: object,
    time_reference_scale: object,
    physical_mobility: object,
    dimensionless_window: object,
    contraction_factor_upper: object,
    dimensionless_gradient_norm_upper: object,
    logarithm_series_terms: int = 8,
) -> PhysicalScaleMobilityCertificate:
    """Build an exact scale map without supplying or inferring biological values."""
    x0 = _exact_fraction(state_reference_scale, "state_reference_scale")
    v0 = _exact_fraction(energy_reference_scale, "energy_reference_scale")
    t0 = _exact_fraction(time_reference_scale, "time_reference_scale")
    mobility = _exact_fraction(physical_mobility, "physical_mobility")
    window = _exact_fraction(dimensionless_window, "dimensionless_window")
    q = _exact_fraction(contraction_factor_upper, "contraction_factor_upper")
    gradient = _exact_fraction(
        dimensionless_gradient_norm_upper,
        "dimensionless_gradient_norm_upper",
    )
    if any(value <= 0 for value in (x0, v0, t0, mobility, window)):
        raise ValueError("reference scales, physical mobility, and window must be positive")
    if gradient < 0:
        raise ValueError("dimensionless_gradient_norm_upper must be nonnegative")
    if q < 0 or q > 1:
        raise ValueError("contraction_factor_upper must lie in the closed interval [0, 1]")
    if type(logarithm_series_terms) is not int or logarithm_series_terms <= 0:
        raise ValueError("logarithm_series_terms must be a positive built-in integer")

    normalized_mobility = mobility * v0 * t0 / (x0 * x0)
    physical_window = t0 * window
    velocity_scale = x0 / t0
    power_scale = v0 / t0
    speed_upper = velocity_scale * normalized_mobility * gradient
    dissipation_upper = power_scale * normalized_mobility * gradient * gradient
    common = dict(
        state_reference_scale=x0,
        energy_reference_scale=v0,
        time_reference_scale=t0,
        physical_mobility=mobility,
        dimensionless_window=window,
        physical_window=physical_window,
        normalized_mobility=normalized_mobility,
        velocity_reference_scale=velocity_scale,
        power_reference_scale=power_scale,
        dimensionless_gradient_norm_upper=gradient,
        physical_speed_upper=speed_upper,
        euclidean_dissipation_upper=dissipation_upper,
        contraction_factor_upper=q,
        logarithm_series_terms=logarithm_series_terms,
    )
    if q == 0:
        return PhysicalScaleMobilityCertificate(
            status="VERIFIED_FINITE_WINDOW_COLLAPSE_SCALE_MAP",
            validation_level="VERIFIED_FINITE_WINDOW_COLLAPSE_SCALE_MAP",
            failure_codes=(),
            logarithm_lower=None,
            logarithm_upper=None,
            decay_rate_lower=None,
            decay_rate_upper=None,
            **common,
        )
    if q == 1:
        return PhysicalScaleMobilityCertificate(
            status="NO_POSITIVE_CONTRACTION_RATE",
            validation_level=None,
            failure_codes=("CONTRACTION_NOT_STRICT",),
            logarithm_lower=None,
            logarithm_upper=None,
            decay_rate_lower=Fraction(0),
            decay_rate_upper=Fraction(0),
            **common,
        )

    log_lower, log_upper = _negative_log_enclosure(q, logarithm_series_terms)
    return PhysicalScaleMobilityCertificate(
        status="VERIFIED_PHYSICAL_SCALE_MOBILITY_MAP",
        validation_level="VERIFIED_PHYSICAL_SCALE_MOBILITY_MAP",
        failure_codes=(),
        logarithm_lower=log_lower,
        logarithm_upper=log_upper,
        decay_rate_lower=log_lower / physical_window,
        decay_rate_upper=log_upper / physical_window,
        **common,
    )


__all__ = [
    "PhysicalScaleMobilityCertificate",
    "physical_scale_mobility_certificate",
]

