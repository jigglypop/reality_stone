"""Exact C3 certificate for an affine triangular graph transform."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_c2_graph_transform import (
        QuantitativeC2GraphTransformCertificate,
        quantitative_c2_triangular_graph_transform,
    )
    from .quantitative_graph_transform import _exact_fraction
else:
    from quantitative_c2_graph_transform import (  # type: ignore[no-redef]
        QuantitativeC2GraphTransformCertificate,
        quantitative_c2_triangular_graph_transform,
    )
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]


@dataclass(frozen=True)
class QuantitativeC3GraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    c2_certificate: QuantitativeC2GraphTransformCertificate
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
class C3GraphIterationBound:
    steps: int
    value_distance_upper: Fraction
    derivative_distance_upper: Fraction
    hessian_distance_upper: Fraction
    third_derivative_distance_upper: Fraction


def quantitative_c3_triangular_graph_transform(
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
    normalized_map_third_derivative_upper: object,
    normalized_map_third_derivative_fiber_lipschitz: object,
    normalized_graph_hessian_upper: object,
    normalized_graph_third_derivative_upper: object,
) -> QuantitativeC3GraphTransformCertificate:
    """Compose the exact C2 gates with C3 class invariance and bunching.

    The third-derivative bound also supplies the fiber-Lipschitz modulus of
    the map Hessian used by the C2 predecessor.  The final fiber modulus is a
    D4-level hypothesis: without it two values of D3 g cannot be compared.
    """
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
    if k3 < 0 or k4 < 0 or lambda3 < 0:
        raise ValueError("normalized third-derivative and modulus bounds must be nonnegative")

    c2 = quantitative_c2_triangular_graph_transform(
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
        normalized_map_hessian_upper=normalized_map_hessian_upper,
        normalized_map_hessian_fiber_lipschitz=k3,
        normalized_graph_hessian_upper=normalized_graph_hessian_upper,
    )
    lipschitz = c2.c1_certificate.lipschitz_certificate
    q = lipschitz.contraction_factor_upper
    mu = lipschitz.base_inverse_lipschitz
    kappa = lipschitz.normalized_graph_slope_upper
    k2 = c2.normalized_map_hessian_upper
    lambda2 = c2.normalized_graph_hessian_upper
    mu_cubed = mu * mu * mu
    one_plus_kappa = 1 + kappa

    output_third = mu_cubed * (
        q * lambda3
        + 3 * k2 * lambda2 * one_plus_kappa
        + k3 * one_plus_kappa**3
    )
    class_margin = lambda3 - output_third
    beta3 = q * mu_cubed
    bunching_margin = 1 - beta3
    hessian_cross = 3 * mu_cubed * k2 * one_plus_kappa
    derivative_cross = 3 * mu_cubed * (
        k2 * lambda2 + k3 * one_plus_kappa**2
    )
    value_cross = mu_cubed * (
        k2 * lambda3
        + 3 * k3 * lambda2 * one_plus_kappa
        + k4 * one_plus_kappa**3
    )

    failures = list(c2.failure_codes)
    if class_margin < 0:
        failures.append("C3_GRAPH_THIRD_DERIVATIVE_CLASS_NOT_INVARIANT")
    if bunching_margin <= 0:
        failures.append("C3_THIRD_DERIVATIVE_BUNCHING_NOT_STRICT")
    common = dict(
        failure_codes=tuple(failures),
        c2_certificate=c2,
        normalized_map_third_derivative_upper=k3,
        normalized_map_third_derivative_fiber_lipschitz=k4,
        normalized_graph_third_derivative_upper=lambda3,
        output_graph_third_derivative_upper=output_third,
        graph_third_derivative_margin=class_margin,
        third_derivative_bunching_factor_upper=beta3,
        third_derivative_bunching_margin=bunching_margin,
        hessian_to_third_derivative_cross_coefficient_upper=hessian_cross,
        derivative_to_third_derivative_cross_coefficient_upper=derivative_cross,
        value_to_third_derivative_cross_coefficient_upper=value_cross,
    )
    if failures:
        return QuantitativeC3GraphTransformCertificate(
            status=failures[0], validation_level=None, robust_interior=False,
            c3_graph_real_dimension=None, **common,
        )
    return QuantitativeC3GraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_C3_TRIANGULAR_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_C3_TRIANGULAR_GRAPH_TRANSFORM",
        robust_interior=c2.robust_interior and class_margin > 0,
        c3_graph_real_dimension=base_dimension,
        **common,
    )


def c3_graph_iteration_bound(
    certificate: QuantitativeC3GraphTransformCertificate,
    *,
    initial_value_distance: object,
    initial_derivative_distance: object,
    initial_hessian_distance: object,
    initial_third_derivative_distance: object,
    steps: int,
) -> C3GraphIterationBound:
    """Iterate the exact upper-triangular C0/C1/C2/C3 recurrence."""
    if certificate.validation_level is None:
        raise ValueError("C3 iteration requires a verified C3 graph-transform certificate")
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative built-in integer")
    names = (
        (initial_value_distance, "initial_value_distance"),
        (initial_derivative_distance, "initial_derivative_distance"),
        (initial_hessian_distance, "initial_hessian_distance"),
        (initial_third_derivative_distance, "initial_third_derivative_distance"),
    )
    delta, derivative, hessian, third = (
        _exact_fraction(value, name) for value, name in names
    )
    if min(delta, derivative, hessian, third) < 0:
        raise ValueError("initial distances must be nonnegative")

    c2 = certificate.c2_certificate
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
    return C3GraphIterationBound(steps, delta, derivative, hessian, third)


__all__ = [
    "C3GraphIterationBound",
    "QuantitativeC3GraphTransformCertificate",
    "c3_graph_iteration_bound",
    "quantitative_c3_triangular_graph_transform",
]
