"""Exact C1 bunching certificate for a triangular C1-diffeomorphism base."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_graph_transform import (
        QuantitativeGraphTransformCertificate,
        _exact_fraction,
        quantitative_triangular_graph_transform,
    )
else:
    from quantitative_graph_transform import (  # type: ignore[no-redef]
        QuantitativeGraphTransformCertificate,
        _exact_fraction,
        quantitative_triangular_graph_transform,
    )


@dataclass(frozen=True)
class QuantitativeC1GraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    lipschitz_certificate: QuantitativeGraphTransformCertificate
    raw_base_derivative_fiber_variation: Fraction
    raw_fiber_derivative_fiber_variation: Fraction
    normalized_base_derivative_fiber_variation: Fraction
    normalized_fiber_derivative_fiber_variation: Fraction
    derivative_bunching_factor_upper: Fraction
    derivative_bunching_margin: Fraction
    derivative_cross_coefficient_upper: Fraction
    c1_graph_real_dimension: int | None


@dataclass(frozen=True)
class C1GraphIterationBound:
    steps: int
    value_distance_upper: Fraction
    derivative_distance_upper: Fraction


def sine_perturbed_base_inverse_lipschitz(amplitude_upper: object) -> Fraction:
    """Return 1/(1-a) for phi(x)=x+a sin(x), requiring exact 0 <= a < 1."""
    amplitude = _exact_fraction(amplitude_upper, "amplitude_upper")
    if amplitude < 0 or amplitude >= 1:
        raise ValueError("amplitude_upper must lie in the exact interval [0, 1)")
    return Fraction(1) / (1 - amplitude)


def quantitative_c1_triangular_graph_transform(
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
    base_derivative_fiber_variation: object,
    fiber_derivative_fiber_variation: object,
) -> QuantitativeC1GraphTransformCertificate:
    """Compose the exact Lipschitz gate with the strict C1 bunching gate."""
    lipschitz = quantitative_triangular_graph_transform(
        base_dimension=base_dimension,
        base_reference_scale=base_reference_scale,
        fiber_reference_scale=fiber_reference_scale,
        fiber_radius=fiber_radius,
        forcing_at_zero_upper=forcing_at_zero_upper,
        base_inverse_lipschitz=base_inverse_lipschitz,
        fiber_linear_norm_upper=fiber_linear_norm_upper,
        base_to_fiber_lipschitz=base_to_fiber_lipschitz,
        fiber_nonlinear_lipschitz=fiber_nonlinear_lipschitz,
        graph_slope_upper=graph_slope_upper,
    )
    hx_raw = _exact_fraction(
        base_derivative_fiber_variation,
        "base_derivative_fiber_variation",
    )
    hy_raw = _exact_fraction(
        fiber_derivative_fiber_variation,
        "fiber_derivative_fiber_variation",
    )
    if hx_raw < 0 or hy_raw < 0:
        raise ValueError("derivative-variation bounds must be nonnegative")
    hx = hx_raw * lipschitz.base_reference_scale
    hy = hy_raw * lipschitz.fiber_reference_scale
    beta = lipschitz.base_inverse_lipschitz * lipschitz.contraction_factor_upper
    margin = 1 - beta
    cross = lipschitz.base_inverse_lipschitz * (
        hy * lipschitz.normalized_graph_slope_upper + hx
    )
    failures = list(lipschitz.failure_codes)
    if margin <= 0:
        failures.append("C1_DERIVATIVE_BUNCHING_NOT_STRICT")
    common = dict(
        failure_codes=tuple(failures),
        lipschitz_certificate=lipschitz,
        raw_base_derivative_fiber_variation=hx_raw,
        raw_fiber_derivative_fiber_variation=hy_raw,
        normalized_base_derivative_fiber_variation=hx,
        normalized_fiber_derivative_fiber_variation=hy,
        derivative_bunching_factor_upper=beta,
        derivative_bunching_margin=margin,
        derivative_cross_coefficient_upper=cross,
    )
    if failures:
        return QuantitativeC1GraphTransformCertificate(
            status=failures[0],
            validation_level=None,
            c1_graph_real_dimension=None,
            **common,
        )
    return QuantitativeC1GraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_C1_TRIANGULAR_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_C1_TRIANGULAR_GRAPH_TRANSFORM",
        c1_graph_real_dimension=base_dimension,
        **common,
    )


def c1_graph_iteration_bound(
    certificate: QuantitativeC1GraphTransformCertificate,
    *,
    initial_value_distance: object,
    initial_derivative_distance: object,
    steps: int,
) -> C1GraphIterationBound:
    """Iterate delta'<=q delta and d'<=beta d+c_D delta exactly."""
    if certificate.validation_level is None:
        raise ValueError("C1 iteration requires a verified C1 graph-transform certificate")
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative built-in integer")
    value_distance = _exact_fraction(initial_value_distance, "initial_value_distance")
    derivative_distance = _exact_fraction(
        initial_derivative_distance,
        "initial_derivative_distance",
    )
    if value_distance < 0 or derivative_distance < 0:
        raise ValueError("initial distances must be nonnegative")
    q = certificate.lipschitz_certificate.contraction_factor_upper
    beta = certificate.derivative_bunching_factor_upper
    cross = certificate.derivative_cross_coefficient_upper
    for _ in range(steps):
        derivative_distance = beta * derivative_distance + cross * value_distance
        value_distance = q * value_distance
    return C1GraphIterationBound(
        steps=steps,
        value_distance_upper=value_distance,
        derivative_distance_upper=derivative_distance,
    )


__all__ = [
    "C1GraphIterationBound",
    "QuantitativeC1GraphTransformCertificate",
    "c1_graph_iteration_bound",
    "quantitative_c1_triangular_graph_transform",
    "sine_perturbed_base_inverse_lipschitz",
]
