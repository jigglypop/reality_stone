"""Exact C2 certificate for an affine triangular graph transform."""

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
class QuantitativeC2GraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    c1_certificate: QuantitativeC1GraphTransformCertificate
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
class C2GraphIterationBound:
    steps: int
    value_distance_upper: Fraction
    derivative_distance_upper: Fraction
    hessian_distance_upper: Fraction


def quantitative_c2_triangular_graph_transform(
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
    normalized_map_hessian_upper: object,
    normalized_map_hessian_fiber_lipschitz: object,
    normalized_graph_hessian_upper: object,
) -> QuantitativeC2GraphTransformCertificate:
    """Compose exact Lipschitz/C1 gates with Hessian invariance and C2 bunching."""
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
    k2 = _exact_fraction(normalized_map_hessian_upper, "normalized_map_hessian_upper")
    k3 = _exact_fraction(
        normalized_map_hessian_fiber_lipschitz,
        "normalized_map_hessian_fiber_lipschitz",
    )
    lambda2 = _exact_fraction(
        normalized_graph_hessian_upper,
        "normalized_graph_hessian_upper",
    )
    if k2 < 0 or k3 < 0 or lambda2 < 0:
        raise ValueError("normalized Hessian and Hessian-modulus bounds must be nonnegative")

    lipschitz = c1.lipschitz_certificate
    q = lipschitz.contraction_factor_upper
    mu = lipschitz.base_inverse_lipschitz
    kappa = lipschitz.normalized_graph_slope_upper
    mu_squared = mu * mu
    one_plus_kappa = 1 + kappa

    output_hessian = mu_squared * (
        q * lambda2 + k2 * one_plus_kappa * one_plus_kappa
    )
    class_margin = lambda2 - output_hessian
    beta2 = q * mu_squared
    bunching_margin = 1 - beta2
    derivative_cross = 2 * mu_squared * k2 * one_plus_kappa
    value_cross = mu_squared * (
        k2 * lambda2 + k3 * one_plus_kappa * one_plus_kappa
    )

    failures = list(c1.failure_codes)
    if class_margin < 0:
        failures.append("C2_GRAPH_HESSIAN_CLASS_NOT_INVARIANT")
    if bunching_margin <= 0:
        failures.append("C2_SECOND_DERIVATIVE_BUNCHING_NOT_STRICT")

    common = dict(
        failure_codes=tuple(failures),
        c1_certificate=c1,
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
        return QuantitativeC2GraphTransformCertificate(
            status=failures[0],
            validation_level=None,
            robust_interior=False,
            c2_graph_real_dimension=None,
            **common,
        )
    robust = lipschitz.robust_interior and class_margin > 0
    return QuantitativeC2GraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_C2_TRIANGULAR_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_C2_TRIANGULAR_GRAPH_TRANSFORM",
        robust_interior=robust,
        c2_graph_real_dimension=base_dimension,
        **common,
    )


def c2_graph_iteration_bound(
    certificate: QuantitativeC2GraphTransformCertificate,
    *,
    initial_value_distance: object,
    initial_derivative_distance: object,
    initial_hessian_distance: object,
    steps: int,
) -> C2GraphIterationBound:
    """Iterate the exact upper-triangular value/derivative/Hessian recurrence."""
    if certificate.validation_level is None:
        raise ValueError("C2 iteration requires a verified C2 graph-transform certificate")
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative built-in integer")
    value_distance = _exact_fraction(initial_value_distance, "initial_value_distance")
    derivative_distance = _exact_fraction(
        initial_derivative_distance,
        "initial_derivative_distance",
    )
    hessian_distance = _exact_fraction(
        initial_hessian_distance,
        "initial_hessian_distance",
    )
    if value_distance < 0 or derivative_distance < 0 or hessian_distance < 0:
        raise ValueError("initial distances must be nonnegative")

    c1 = certificate.c1_certificate
    q = c1.lipschitz_certificate.contraction_factor_upper
    beta1 = c1.derivative_bunching_factor_upper
    c10 = c1.derivative_cross_coefficient_upper
    beta2 = certificate.second_derivative_bunching_factor_upper
    c21 = certificate.derivative_to_hessian_cross_coefficient_upper
    c20 = certificate.value_to_hessian_cross_coefficient_upper
    for _ in range(steps):
        next_hessian = (
            beta2 * hessian_distance
            + c21 * derivative_distance
            + c20 * value_distance
        )
        next_derivative = beta1 * derivative_distance + c10 * value_distance
        next_value = q * value_distance
        value_distance = next_value
        derivative_distance = next_derivative
        hessian_distance = next_hessian
    return C2GraphIterationBound(
        steps=steps,
        value_distance_upper=value_distance,
        derivative_distance_upper=derivative_distance,
        hessian_distance_upper=hessian_distance,
    )


__all__ = [
    "C2GraphIterationBound",
    "QuantitativeC2GraphTransformCertificate",
    "c2_graph_iteration_bound",
    "quantitative_c2_triangular_graph_transform",
]
