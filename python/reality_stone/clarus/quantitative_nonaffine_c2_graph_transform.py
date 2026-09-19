"""Exact C2 certificate for a graph-independent nonaffine triangular base."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_c1_graph_transform import (
        QuantitativeC1GraphTransformCertificate,
        quantitative_c1_triangular_graph_transform,
    )
    from .quantitative_graph_transform import _exact_fraction
else:
    from quantitative_c1_graph_transform import (  # type: ignore[no-redef]
        QuantitativeC1GraphTransformCertificate,
        quantitative_c1_triangular_graph_transform,
    )
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]


@dataclass(frozen=True)
class NonaffineBaseC2Bounds:
    inverse_derivative_upper: Fraction
    inverse_hessian_upper: Fraction


@dataclass(frozen=True)
class QuantitativeNonaffineC2GraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    c1_certificate: QuantitativeC1GraphTransformCertificate
    base_inverse_hessian_upper: Fraction
    normalized_map_hessian_upper: Fraction
    normalized_map_hessian_fiber_lipschitz: Fraction
    normalized_graph_hessian_upper: Fraction
    output_graph_hessian_upper: Fraction
    graph_hessian_margin: Fraction
    second_derivative_bunching_factor_upper: Fraction
    second_derivative_bunching_margin: Fraction
    derivative_to_hessian_cross_coefficient_upper: Fraction
    value_to_hessian_cross_coefficient_upper: Fraction
    c2_graph_real_dimension: int | None


@dataclass(frozen=True)
class NonaffineC2GraphIterationBound:
    steps: int
    value_distance_upper: Fraction
    derivative_distance_upper: Fraction
    hessian_distance_upper: Fraction


def sine_perturbed_base_c2_bounds(amplitude_upper: object) -> NonaffineBaseC2Bounds:
    """Bounds for the inverse of phi(x)=x+a sin(x), with exact 0 <= a < 1."""
    amplitude = _exact_fraction(amplitude_upper, "amplitude_upper")
    if amplitude < 0 or amplitude >= 1:
        raise ValueError("amplitude_upper must lie in the exact interval [0, 1)")
    gap = 1 - amplitude
    return NonaffineBaseC2Bounds(
        inverse_derivative_upper=Fraction(1) / gap,
        inverse_hessian_upper=amplitude / (gap * gap * gap),
    )


def quantitative_nonaffine_c2_triangular_graph_transform(
    *,
    base_dimension: int,
    base_reference_scale: object,
    fiber_reference_scale: object,
    fiber_radius: object,
    forcing_at_zero_upper: object,
    base_inverse_lipschitz: object,
    base_inverse_hessian_upper: object,
    fiber_linear_norm_upper: object,
    base_to_fiber_lipschitz: object,
    fiber_nonlinear_lipschitz: object,
    graph_slope_upper: object,
    base_derivative_fiber_variation: object,
    fiber_derivative_fiber_variation: object,
    normalized_map_hessian_upper: object,
    normalized_map_hessian_fiber_lipschitz: object,
    normalized_graph_hessian_upper: object,
) -> QuantitativeNonaffineC2GraphTransformCertificate:
    """Compose the nonaffine C1 gate with exact inverse-Hessian C2 terms."""
    c1 = quantitative_c1_triangular_graph_transform(
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
        base_derivative_fiber_variation=base_derivative_fiber_variation,
        fiber_derivative_fiber_variation=fiber_derivative_fiber_variation,
    )
    nu = _exact_fraction(base_inverse_hessian_upper, "base_inverse_hessian_upper")
    k2 = _exact_fraction(normalized_map_hessian_upper, "normalized_map_hessian_upper")
    k3 = _exact_fraction(
        normalized_map_hessian_fiber_lipschitz,
        "normalized_map_hessian_fiber_lipschitz",
    )
    lambda2 = _exact_fraction(normalized_graph_hessian_upper, "normalized_graph_hessian_upper")
    if nu < 0 or k2 < 0 or k3 < 0 or lambda2 < 0:
        raise ValueError("inverse and normalized Hessian bounds must be nonnegative")

    lipschitz = c1.lipschitz_certificate
    q = lipschitz.contraction_factor_upper
    mu = lipschitz.base_inverse_lipschitz
    kappa = lipschitz.normalized_graph_slope_upper
    lx = lipschitz.normalized_base_to_fiber_lipschitz
    hx = c1.normalized_base_derivative_fiber_variation
    hy = c1.normalized_fiber_derivative_fiber_variation
    mu2 = mu * mu
    one_plus_kappa = 1 + kappa
    slope_image = q * kappa + lx

    output_hessian = (
        mu2 * (q * lambda2 + k2 * one_plus_kappa * one_plus_kappa)
        + slope_image * nu
    )
    class_margin = lambda2 - output_hessian
    beta2 = q * mu2
    bunching_margin = 1 - beta2
    derivative_cross = 2 * mu2 * k2 * one_plus_kappa + q * nu
    value_cross = (
        mu2 * (k2 * lambda2 + k3 * one_plus_kappa * one_plus_kappa)
        + nu * (hy * kappa + hx)
    )

    failures = list(c1.failure_codes)
    if class_margin < 0:
        failures.append("NONAFFINE_C2_GRAPH_HESSIAN_CLASS_NOT_INVARIANT")
    if bunching_margin <= 0:
        failures.append("NONAFFINE_C2_SECOND_DERIVATIVE_BUNCHING_NOT_STRICT")
    common = dict(
        failure_codes=tuple(failures),
        c1_certificate=c1,
        base_inverse_hessian_upper=nu,
        normalized_map_hessian_upper=k2,
        normalized_map_hessian_fiber_lipschitz=k3,
        normalized_graph_hessian_upper=lambda2,
        output_graph_hessian_upper=output_hessian,
        graph_hessian_margin=class_margin,
        second_derivative_bunching_factor_upper=beta2,
        second_derivative_bunching_margin=bunching_margin,
        derivative_to_hessian_cross_coefficient_upper=derivative_cross,
        value_to_hessian_cross_coefficient_upper=value_cross,
    )
    if failures:
        return QuantitativeNonaffineC2GraphTransformCertificate(
            status=failures[0],
            validation_level=None,
            robust_interior=False,
            c2_graph_real_dimension=None,
            **common,
        )
    robust = lipschitz.robust_interior and class_margin > 0
    return QuantitativeNonaffineC2GraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_NONAFFINE_C2_TRIANGULAR_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_NONAFFINE_C2_TRIANGULAR_GRAPH_TRANSFORM",
        robust_interior=robust,
        c2_graph_real_dimension=base_dimension,
        **common,
    )


def nonaffine_c2_graph_iteration_bound(
    certificate: QuantitativeNonaffineC2GraphTransformCertificate,
    *,
    initial_value_distance: object,
    initial_derivative_distance: object,
    initial_hessian_distance: object,
    steps: int,
) -> NonaffineC2GraphIterationBound:
    """Iterate the exact value/derivative/nonaffine-Hessian recurrence."""
    if certificate.validation_level is None:
        raise ValueError("C2 iteration requires a verified nonaffine C2 certificate")
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative built-in integer")
    delta = _exact_fraction(initial_value_distance, "initial_value_distance")
    derivative = _exact_fraction(initial_derivative_distance, "initial_derivative_distance")
    hessian = _exact_fraction(initial_hessian_distance, "initial_hessian_distance")
    if delta < 0 or derivative < 0 or hessian < 0:
        raise ValueError("initial distances must be nonnegative")

    c1 = certificate.c1_certificate
    q = c1.lipschitz_certificate.contraction_factor_upper
    for _ in range(steps):
        next_hessian = (
            certificate.second_derivative_bunching_factor_upper * hessian
            + certificate.derivative_to_hessian_cross_coefficient_upper * derivative
            + certificate.value_to_hessian_cross_coefficient_upper * delta
        )
        next_derivative = (
            c1.derivative_bunching_factor_upper * derivative
            + c1.derivative_cross_coefficient_upper * delta
        )
        next_delta = q * delta
        delta, derivative, hessian = next_delta, next_derivative, next_hessian
    return NonaffineC2GraphIterationBound(steps, delta, derivative, hessian)


__all__ = [
    "NonaffineBaseC2Bounds",
    "NonaffineC2GraphIterationBound",
    "QuantitativeNonaffineC2GraphTransformCertificate",
    "nonaffine_c2_graph_iteration_bound",
    "quantitative_nonaffine_c2_triangular_graph_transform",
    "sine_perturbed_base_c2_bounds",
]

