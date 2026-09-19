"""Exact constant certificate for a triangular nonautonomous graph transform."""

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
class QuantitativeGraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    base_dimension: int
    base_reference_scale: Fraction
    fiber_reference_scale: Fraction
    raw_fiber_radius: Fraction
    raw_forcing_at_zero_upper: Fraction
    raw_base_to_fiber_lipschitz: Fraction
    raw_graph_slope_upper: Fraction
    normalized_fiber_radius: Fraction
    normalized_forcing_at_zero_upper: Fraction
    normalized_base_to_fiber_lipschitz: Fraction
    normalized_graph_slope_upper: Fraction
    base_inverse_lipschitz: Fraction
    fiber_linear_norm_upper: Fraction
    fiber_nonlinear_lipschitz: Fraction
    contraction_factor_upper: Fraction
    contraction_margin: Fraction
    tube_margin: Fraction
    slope_margin: Fraction
    graph_real_dimension: int | None


def quantitative_triangular_graph_transform(
    *,
    base_dimension: int,
    base_reference_scale: object,
    fiber_reference_scale: object,
    fiber_radius: object,
    forcing_at_zero_upper: object,
    base_inverse_lipschitz: object,
    fiber_linear_norm_upper: object,
    base_to_fiber_lipschitz: object,
    fiber_nonlinear_lipschitz: object,
    graph_slope_upper: object,
) -> QuantitativeGraphTransformCertificate:
    """Check G1a--G1c exactly for the declared triangular chart constants."""
    if type(base_dimension) is not int or base_dimension <= 0:
        raise ValueError("base_dimension must be a positive built-in integer")
    x_scale = _exact_fraction(base_reference_scale, "base_reference_scale")
    y_scale = _exact_fraction(fiber_reference_scale, "fiber_reference_scale")
    radius_raw = _exact_fraction(fiber_radius, "fiber_radius")
    forcing_raw = _exact_fraction(forcing_at_zero_upper, "forcing_at_zero_upper")
    mu = _exact_fraction(base_inverse_lipschitz, "base_inverse_lipschitz")
    b = _exact_fraction(fiber_linear_norm_upper, "fiber_linear_norm_upper")
    lx_raw = _exact_fraction(base_to_fiber_lipschitz, "base_to_fiber_lipschitz")
    ly = _exact_fraction(fiber_nonlinear_lipschitz, "fiber_nonlinear_lipschitz")
    slope_raw = _exact_fraction(graph_slope_upper, "graph_slope_upper")
    if x_scale <= 0 or y_scale <= 0 or radius_raw <= 0 or mu <= 0:
        raise ValueError("reference scales, fiber radius, and base inverse Lipschitz must be positive")
    if any(value < 0 for value in (forcing_raw, b, lx_raw, ly, slope_raw)):
        raise ValueError("graph-transform bounds must be nonnegative")
    radius = radius_raw / y_scale
    forcing = forcing_raw / y_scale
    lx = lx_raw * x_scale / y_scale
    slope = slope_raw * x_scale / y_scale
    q = b + ly
    contraction_margin = 1 - q
    tube_margin = radius - (q * radius + forcing)
    slope_margin = slope - mu * (q * slope + lx)
    failures = []
    if contraction_margin <= 0:
        failures.append("GRAPH_TRANSFORM_CONTRACTION_NOT_STRICT")
    if tube_margin < 0:
        failures.append("GRAPH_TRANSFORM_TUBE_NOT_INVARIANT")
    if slope_margin < 0:
        failures.append("GRAPH_TRANSFORM_SLOPE_NOT_INVARIANT")
    common = dict(
        failure_codes=tuple(failures),
        base_dimension=base_dimension,
        base_reference_scale=x_scale,
        fiber_reference_scale=y_scale,
        raw_fiber_radius=radius_raw,
        raw_forcing_at_zero_upper=forcing_raw,
        raw_base_to_fiber_lipschitz=lx_raw,
        raw_graph_slope_upper=slope_raw,
        normalized_fiber_radius=radius,
        normalized_forcing_at_zero_upper=forcing,
        normalized_base_to_fiber_lipschitz=lx,
        normalized_graph_slope_upper=slope,
        base_inverse_lipschitz=mu,
        fiber_linear_norm_upper=b,
        fiber_nonlinear_lipschitz=ly,
        contraction_factor_upper=q,
        contraction_margin=contraction_margin,
        tube_margin=tube_margin,
        slope_margin=slope_margin,
    )
    if failures:
        primary = failures[0]
        return QuantitativeGraphTransformCertificate(
            status=primary,
            validation_level=None,
            robust_interior=False,
            graph_real_dimension=None,
            **common,
        )
    robust = tube_margin > 0 and slope_margin > 0
    return QuantitativeGraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_TRIANGULAR_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_TRIANGULAR_GRAPH_TRANSFORM",
        robust_interior=robust,
        graph_real_dimension=base_dimension,
        **common,
    )


def graph_tracking_bound(
    certificate: QuantitativeGraphTransformCertificate,
    *,
    initial_fiber_distance: object,
    steps: int,
) -> Fraction:
    """Return the exact raw-fiber-unit bound q**steps * initial distance."""
    if certificate.validation_level is None:
        raise ValueError("tracking requires a verified graph-transform certificate")
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative built-in integer")
    distance = _exact_fraction(initial_fiber_distance, "initial_fiber_distance")
    if distance < 0:
        raise ValueError("initial_fiber_distance must be nonnegative")
    return certificate.contraction_factor_upper ** steps * distance


__all__ = [
    "QuantitativeGraphTransformCertificate",
    "quantitative_triangular_graph_transform",
    "graph_tracking_bound",
]
