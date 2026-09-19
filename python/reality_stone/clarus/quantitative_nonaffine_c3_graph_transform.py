"""Exact C3 certificate for a graph-independent nonaffine triangular base."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_nonaffine_c2_graph_transform import (
        QuantitativeNonaffineC2GraphTransformCertificate,
        quantitative_nonaffine_c2_triangular_graph_transform,
    )
    from .quantitative_graph_transform import _exact_fraction
else:
    from quantitative_nonaffine_c2_graph_transform import (  # type: ignore[no-redef]
        QuantitativeNonaffineC2GraphTransformCertificate,
        quantitative_nonaffine_c2_triangular_graph_transform,
    )
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]


@dataclass(frozen=True)
class NonaffineBaseC3Bounds:
    inverse_derivative_upper: Fraction
    inverse_hessian_upper: Fraction
    inverse_third_derivative_upper: Fraction


@dataclass(frozen=True)
class QuantitativeNonaffineC3GraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    nonaffine_c2_certificate: QuantitativeNonaffineC2GraphTransformCertificate
    base_inverse_third_derivative_upper: Fraction
    normalized_map_third_derivative_upper: Fraction
    normalized_map_third_derivative_fiber_lipschitz: Fraction
    normalized_graph_third_derivative_upper: Fraction
    output_graph_third_derivative_upper: Fraction
    graph_third_derivative_margin: Fraction
    third_derivative_bunching_factor_upper: Fraction
    third_derivative_bunching_margin: Fraction
    hessian_to_third_derivative_cross_coefficient_upper: Fraction
    derivative_to_third_derivative_cross_coefficient_upper: Fraction
    value_to_third_derivative_cross_coefficient_upper: Fraction
    c3_graph_real_dimension: int | None


@dataclass(frozen=True)
class NonaffineC3GraphIterationBound:
    steps: int
    value_distance_upper: Fraction
    derivative_distance_upper: Fraction
    hessian_distance_upper: Fraction
    third_derivative_distance_upper: Fraction


def sine_perturbed_base_c3_bounds(amplitude_upper: object) -> NonaffineBaseC3Bounds:
    """Inverse C3 bounds for phi(x)=x+a sin(x), exact on 0 <= a < 1."""
    amplitude = _exact_fraction(amplitude_upper, "amplitude_upper")
    if amplitude < 0 or amplitude >= 1:
        raise ValueError("amplitude_upper must lie in the exact interval [0, 1)")
    gap = 1 - amplitude
    return NonaffineBaseC3Bounds(
        inverse_derivative_upper=1 / gap,
        inverse_hessian_upper=amplitude / gap**3,
        inverse_third_derivative_upper=(amplitude + 4 * amplitude**2) / gap**5,
    )


def quantitative_nonaffine_c3_triangular_graph_transform(
    *,
    base_dimension: int,
    base_reference_scale: object,
    fiber_reference_scale: object,
    fiber_radius: object,
    forcing_at_zero_upper: object,
    base_inverse_lipschitz: object,
    base_inverse_hessian_upper: object,
    base_inverse_third_derivative_upper: object,
    fiber_linear_norm_upper: object,
    base_to_fiber_lipschitz: object,
    fiber_nonlinear_lipschitz: object,
    graph_slope_upper: object,
    base_derivative_fiber_variation: object,
    fiber_derivative_fiber_variation: object,
    normalized_map_hessian_upper: object,
    normalized_map_third_derivative_upper: object,
    normalized_map_third_derivative_fiber_lipschitz: object,
    normalized_graph_hessian_upper: object,
    normalized_graph_third_derivative_upper: object,
) -> QuantitativeNonaffineC3GraphTransformCertificate:
    """Compose nonaffine C2 with the exact third-order inverse chain rule."""
    tau = _exact_fraction(
        base_inverse_third_derivative_upper,
        "base_inverse_third_derivative_upper",
    )
    k3 = _exact_fraction(
        normalized_map_third_derivative_upper,
        "normalized_map_third_derivative_upper",
    )
    k4 = _exact_fraction(
        normalized_map_third_derivative_fiber_lipschitz,
        "normalized_map_third_derivative_fiber_lipschitz",
    )
    lambda3 = _exact_fraction(
        normalized_graph_third_derivative_upper,
        "normalized_graph_third_derivative_upper",
    )
    if min(tau, k3, k4, lambda3) < 0:
        raise ValueError("normalized inverse/map/graph C3 bounds must be nonnegative")

    c2 = quantitative_nonaffine_c2_triangular_graph_transform(
        base_dimension=base_dimension,
        base_reference_scale=base_reference_scale,
        fiber_reference_scale=fiber_reference_scale,
        fiber_radius=fiber_radius,
        forcing_at_zero_upper=forcing_at_zero_upper,
        base_inverse_lipschitz=base_inverse_lipschitz,
        base_inverse_hessian_upper=base_inverse_hessian_upper,
        fiber_linear_norm_upper=fiber_linear_norm_upper,
        base_to_fiber_lipschitz=base_to_fiber_lipschitz,
        fiber_nonlinear_lipschitz=fiber_nonlinear_lipschitz,
        graph_slope_upper=graph_slope_upper,
        base_derivative_fiber_variation=base_derivative_fiber_variation,
        fiber_derivative_fiber_variation=fiber_derivative_fiber_variation,
        normalized_map_hessian_upper=normalized_map_hessian_upper,
        normalized_map_hessian_fiber_lipschitz=k3,
        normalized_graph_hessian_upper=normalized_graph_hessian_upper,
    )
    c1 = c2.c1_certificate
    lipschitz = c1.lipschitz_certificate
    q = lipschitz.contraction_factor_upper
    mu = lipschitz.base_inverse_lipschitz
    nu = c2.base_inverse_hessian_upper
    kappa = lipschitz.normalized_graph_slope_upper
    lx = lipschitz.normalized_base_to_fiber_lipschitz
    hx = c1.normalized_base_derivative_fiber_variation
    hy = c1.normalized_fiber_derivative_fiber_variation
    k2 = c2.normalized_map_hessian_upper
    lambda2 = c2.normalized_graph_hessian_upper
    r = 1 + kappa
    s = q * kappa + lx
    a2 = q * lambda2 + k2 * r**2
    a3 = q * lambda3 + 3 * k2 * lambda2 * r + k3 * r**3

    output_third = mu**3 * a3 + 3 * mu * nu * a2 + s * tau
    class_margin = lambda3 - output_third
    beta3 = q * mu**3
    bunching_margin = 1 - beta3
    c32 = 3 * mu**3 * k2 * r + 3 * mu * nu * q
    c31 = (
        3 * mu**3 * (k2 * lambda2 + k3 * r**2)
        + 6 * mu * nu * k2 * r + tau * q
    )
    c30 = (
        mu**3 * (k2 * lambda3 + 3 * k3 * lambda2 * r + k4 * r**3)
        + 3 * mu * nu * (k2 * lambda2 + k3 * r**2)
        + tau * (hy * kappa + hx)
    )

    failures = list(c2.failure_codes)
    if class_margin < 0:
        failures.append("NONAFFINE_C3_GRAPH_THIRD_DERIVATIVE_CLASS_NOT_INVARIANT")
    if bunching_margin <= 0:
        failures.append("NONAFFINE_C3_THIRD_DERIVATIVE_BUNCHING_NOT_STRICT")
    common = dict(
        failure_codes=tuple(failures), nonaffine_c2_certificate=c2,
        base_inverse_third_derivative_upper=tau,
        normalized_map_third_derivative_upper=k3,
        normalized_map_third_derivative_fiber_lipschitz=k4,
        normalized_graph_third_derivative_upper=lambda3,
        output_graph_third_derivative_upper=output_third,
        graph_third_derivative_margin=class_margin,
        third_derivative_bunching_factor_upper=beta3,
        third_derivative_bunching_margin=bunching_margin,
        hessian_to_third_derivative_cross_coefficient_upper=c32,
        derivative_to_third_derivative_cross_coefficient_upper=c31,
        value_to_third_derivative_cross_coefficient_upper=c30,
    )
    if failures:
        return QuantitativeNonaffineC3GraphTransformCertificate(
            status=failures[0], validation_level=None, robust_interior=False,
            c3_graph_real_dimension=None, **common,
        )
    return QuantitativeNonaffineC3GraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_NONAFFINE_C3_TRIANGULAR_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_NONAFFINE_C3_TRIANGULAR_GRAPH_TRANSFORM",
        robust_interior=c2.robust_interior and class_margin > 0,
        c3_graph_real_dimension=base_dimension, **common,
    )


def nonaffine_c3_graph_iteration_bound(
    certificate: QuantitativeNonaffineC3GraphTransformCertificate,
    *, initial_value_distance: object, initial_derivative_distance: object,
    initial_hessian_distance: object, initial_third_derivative_distance: object,
    steps: int,
) -> NonaffineC3GraphIterationBound:
    """Iterate the exact common-inverse C0/C1/C2/C3 recurrence."""
    if certificate.validation_level is None:
        raise ValueError("C3 iteration requires a verified nonaffine C3 certificate")
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative built-in integer")
    values = tuple(
        _exact_fraction(value, name)
        for value, name in (
            (initial_value_distance, "initial_value_distance"),
            (initial_derivative_distance, "initial_derivative_distance"),
            (initial_hessian_distance, "initial_hessian_distance"),
            (initial_third_derivative_distance, "initial_third_derivative_distance"),
        )
    )
    if min(values) < 0:
        raise ValueError("initial distances must be nonnegative")
    delta, derivative, hessian, third = values
    c2 = certificate.nonaffine_c2_certificate
    c1 = c2.c1_certificate
    q = c1.lipschitz_certificate.contraction_factor_upper
    for _ in range(steps):
        next_third = (
            certificate.third_derivative_bunching_factor_upper * third
            + certificate.hessian_to_third_derivative_cross_coefficient_upper * hessian
            + certificate.derivative_to_third_derivative_cross_coefficient_upper * derivative
            + certificate.value_to_third_derivative_cross_coefficient_upper * delta
        )
        next_hessian = (
            c2.second_derivative_bunching_factor_upper * hessian
            + c2.derivative_to_hessian_cross_coefficient_upper * derivative
            + c2.value_to_hessian_cross_coefficient_upper * delta
        )
        next_derivative = (
            c1.derivative_bunching_factor_upper * derivative
            + c1.derivative_cross_coefficient_upper * delta
        )
        delta, derivative, hessian, third = (
            q * delta, next_derivative, next_hessian, next_third
        )
    return NonaffineC3GraphIterationBound(steps, delta, derivative, hessian, third)


__all__ = [
    "NonaffineBaseC3Bounds",
    "NonaffineC3GraphIterationBound",
    "QuantitativeNonaffineC3GraphTransformCertificate",
    "nonaffine_c3_graph_iteration_bound",
    "quantitative_nonaffine_c3_triangular_graph_transform",
    "sine_perturbed_base_c3_bounds",
]
