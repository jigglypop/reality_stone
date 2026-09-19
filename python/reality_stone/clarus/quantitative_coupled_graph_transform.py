"""Exact constant gate for an affine-base, Lipschitz coupled graph transform."""

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
class QuantitativeCoupledGraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    base_dimension: int
    base_reference_scale: Fraction
    fiber_reference_scale: Fraction
    normalized_fiber_radius: Fraction
    normalized_forcing_at_zero_upper: Fraction
    base_inverse_lipschitz: Fraction
    fiber_linear_norm_upper: Fraction
    normalized_base_self_lipschitz: Fraction
    normalized_fiber_to_base_lipschitz: Fraction
    normalized_base_to_fiber_lipschitz: Fraction
    fiber_self_lipschitz: Fraction
    normalized_graph_slope_upper: Fraction
    fiber_factor_upper: Fraction
    base_invertibility_lower: Fraction
    base_fixed_point_factor_upper: Fraction
    transform_contraction_factor_upper: Fraction | None
    base_invertibility_margin: Fraction
    tube_margin: Fraction
    slope_margin: Fraction
    transform_contraction_margin: Fraction | None
    graph_real_dimension: int | None


def quantitative_coupled_graph_transform(
    *,
    base_dimension: int,
    base_reference_scale: object,
    fiber_reference_scale: object,
    fiber_radius: object,
    forcing_at_zero_upper: object,
    base_inverse_lipschitz: object,
    fiber_linear_norm_upper: object,
    base_self_lipschitz: object,
    fiber_to_base_lipschitz: object,
    base_to_fiber_lipschitz: object,
    fiber_self_lipschitz: object,
    graph_slope_upper: object,
) -> QuantitativeCoupledGraphTransformCertificate:
    """Check the affine-base coupled Lipschitz graph-transform conditions exactly."""
    if type(base_dimension) is not int or base_dimension <= 0:
        raise ValueError("base_dimension must be a positive built-in integer")
    x_scale = _exact_fraction(base_reference_scale, "base_reference_scale")
    y_scale = _exact_fraction(fiber_reference_scale, "fiber_reference_scale")
    radius_raw = _exact_fraction(fiber_radius, "fiber_radius")
    forcing_raw = _exact_fraction(forcing_at_zero_upper, "forcing_at_zero_upper")
    mu = _exact_fraction(base_inverse_lipschitz, "base_inverse_lipschitz")
    b = _exact_fraction(fiber_linear_norm_upper, "fiber_linear_norm_upper")
    fx = _exact_fraction(base_self_lipschitz, "base_self_lipschitz")
    fy_raw = _exact_fraction(fiber_to_base_lipschitz, "fiber_to_base_lipschitz")
    gx_raw = _exact_fraction(base_to_fiber_lipschitz, "base_to_fiber_lipschitz")
    gy = _exact_fraction(fiber_self_lipschitz, "fiber_self_lipschitz")
    slope_raw = _exact_fraction(graph_slope_upper, "graph_slope_upper")
    if any(value <= 0 for value in (x_scale, y_scale, radius_raw, mu)):
        raise ValueError("reference scales, fiber radius, and base inverse Lipschitz must be positive")
    if any(value < 0 for value in (forcing_raw, b, fx, fy_raw, gx_raw, gy, slope_raw)):
        raise ValueError("coupled graph-transform bounds must be nonnegative")

    radius = radius_raw / y_scale
    forcing = forcing_raw / y_scale
    fy = fy_raw * y_scale / x_scale
    gx = gx_raw * x_scale / y_scale
    slope = slope_raw * x_scale / y_scale
    q = b + gy
    fixed_point_factor = mu * (fx + fy * slope)
    alpha = Fraction(1) / mu - fx - fy * slope
    tube_margin = radius - (q * radius + forcing)
    slope_margin = slope * alpha - (q * slope + gx)
    transform_factor = None if alpha <= 0 else q + (q * slope + gx) * fy / alpha
    transform_margin = None if transform_factor is None else 1 - transform_factor
    failures = []
    if alpha <= 0:
        failures.append("COUPLED_BASE_INVERTIBILITY_NOT_STRICT")
    if tube_margin < 0:
        failures.append("COUPLED_GRAPH_TUBE_NOT_INVARIANT")
    if slope_margin < 0:
        failures.append("COUPLED_GRAPH_SLOPE_NOT_INVARIANT")
    if transform_margin is None or transform_margin <= 0:
        failures.append("COUPLED_GRAPH_TRANSFORM_CONTRACTION_NOT_STRICT")
    common = dict(
        failure_codes=tuple(failures),
        base_dimension=base_dimension,
        base_reference_scale=x_scale,
        fiber_reference_scale=y_scale,
        normalized_fiber_radius=radius,
        normalized_forcing_at_zero_upper=forcing,
        base_inverse_lipschitz=mu,
        fiber_linear_norm_upper=b,
        normalized_base_self_lipschitz=fx,
        normalized_fiber_to_base_lipschitz=fy,
        normalized_base_to_fiber_lipschitz=gx,
        fiber_self_lipschitz=gy,
        normalized_graph_slope_upper=slope,
        fiber_factor_upper=q,
        base_invertibility_lower=alpha,
        base_fixed_point_factor_upper=fixed_point_factor,
        transform_contraction_factor_upper=transform_factor,
        base_invertibility_margin=alpha,
        tube_margin=tube_margin,
        slope_margin=slope_margin,
        transform_contraction_margin=transform_margin,
    )
    if failures:
        return QuantitativeCoupledGraphTransformCertificate(
            status=failures[0],
            validation_level=None,
            robust_interior=False,
            graph_real_dimension=None,
            **common,
        )
    robust = tube_margin > 0 and slope_margin > 0
    return QuantitativeCoupledGraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_COUPLED_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_COUPLED_GRAPH_TRANSFORM",
        robust_interior=robust,
        graph_real_dimension=base_dimension,
        **common,
    )


__all__ = [
    "QuantitativeCoupledGraphTransformCertificate",
    "quantitative_coupled_graph_transform",
]

