"""Exact C4 certificate for an affine triangular graph transform."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_c3_graph_transform import (
        QuantitativeC3GraphTransformCertificate,
        quantitative_c3_triangular_graph_transform,
    )
    from .quantitative_graph_transform import _exact_fraction
else:
    from quantitative_c3_graph_transform import (  # type: ignore[no-redef]
        QuantitativeC3GraphTransformCertificate,
        quantitative_c3_triangular_graph_transform,
    )
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]


@dataclass(frozen=True)
class QuantitativeC4GraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    c3_certificate: QuantitativeC3GraphTransformCertificate
    normalized_map_fourth_derivative_upper: Fraction
    normalized_map_fourth_derivative_fiber_lipschitz: Fraction
    normalized_graph_fourth_derivative_upper: Fraction
    output_graph_fourth_derivative_upper: Fraction
    graph_fourth_derivative_margin: Fraction
    fourth_derivative_bunching_factor_upper: Fraction
    fourth_derivative_bunching_margin: Fraction
    third_to_fourth_derivative_cross_coefficient_upper: Fraction
    hessian_to_fourth_derivative_cross_coefficient_upper: Fraction
    derivative_to_fourth_derivative_cross_coefficient_upper: Fraction
    value_to_fourth_derivative_cross_coefficient_upper: Fraction
    c4_graph_real_dimension: int | None


@dataclass(frozen=True)
class C4GraphIterationBound:
    steps: int
    value_distance_upper: Fraction
    derivative_distance_upper: Fraction
    hessian_distance_upper: Fraction
    third_derivative_distance_upper: Fraction
    fourth_derivative_distance_upper: Fraction


def quantitative_c4_triangular_graph_transform(
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
    normalized_map_fourth_derivative_upper: object,
    normalized_map_fourth_derivative_fiber_lipschitz: object,
    normalized_graph_hessian_upper: object,
    normalized_graph_third_derivative_upper: object,
    normalized_graph_fourth_derivative_upper: object,
) -> QuantitativeC4GraphTransformCertificate:
    """Compose the exact C3 gates with C4 class and five-layer bunching."""
    k4 = _exact_fraction(
        normalized_map_fourth_derivative_upper,
        "normalized_map_fourth_derivative_upper",
    )
    k5 = _exact_fraction(
        normalized_map_fourth_derivative_fiber_lipschitz,
        "normalized_map_fourth_derivative_fiber_lipschitz",
    )
    lambda4 = _exact_fraction(
        normalized_graph_fourth_derivative_upper,
        "normalized_graph_fourth_derivative_upper",
    )
    if min(k4, k5, lambda4) < 0:
        raise ValueError("normalized fourth-derivative and modulus bounds must be nonnegative")

    c3 = quantitative_c3_triangular_graph_transform(
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
        normalized_map_third_derivative_upper=normalized_map_third_derivative_upper,
        normalized_map_third_derivative_fiber_lipschitz=k4,
        normalized_graph_hessian_upper=normalized_graph_hessian_upper,
        normalized_graph_third_derivative_upper=normalized_graph_third_derivative_upper,
    )
    c2 = c3.c2_certificate
    lipschitz = c2.c1_certificate.lipschitz_certificate
    q = lipschitz.contraction_factor_upper
    mu = lipschitz.base_inverse_lipschitz
    kappa = lipschitz.normalized_graph_slope_upper
    k2 = c2.normalized_map_hessian_upper
    k3 = c3.normalized_map_third_derivative_upper
    lambda2 = c2.normalized_graph_hessian_upper
    lambda3 = c3.normalized_graph_third_derivative_upper
    r = 1 + kappa
    mu4 = mu**4

    output_fourth = mu4 * (
        q * lambda4
        + 4 * k2 * lambda3 * r
        + 3 * k2 * lambda2**2
        + 6 * k3 * lambda2 * r**2
        + k4 * r**4
    )
    class_margin = lambda4 - output_fourth
    beta4 = q * mu4
    bunching_margin = 1 - beta4
    c43 = mu4 * 4 * k2 * r
    c42 = mu4 * (6 * k2 * lambda2 + 6 * k3 * r**2)
    c41 = mu4 * (
        4 * k2 * lambda3 + 12 * k3 * lambda2 * r + 4 * k4 * r**3
    )
    c40 = mu4 * (
        k2 * lambda4
        + 4 * k3 * lambda3 * r
        + 3 * k3 * lambda2**2
        + 6 * k4 * lambda2 * r**2
        + k5 * r**4
    )

    failures = list(c3.failure_codes)
    if class_margin < 0:
        failures.append("C4_GRAPH_FOURTH_DERIVATIVE_CLASS_NOT_INVARIANT")
    if bunching_margin <= 0:
        failures.append("C4_FOURTH_DERIVATIVE_BUNCHING_NOT_STRICT")
    common = dict(
        failure_codes=tuple(failures),
        c3_certificate=c3,
        normalized_map_fourth_derivative_upper=k4,
        normalized_map_fourth_derivative_fiber_lipschitz=k5,
        normalized_graph_fourth_derivative_upper=lambda4,
        output_graph_fourth_derivative_upper=output_fourth,
        graph_fourth_derivative_margin=class_margin,
        fourth_derivative_bunching_factor_upper=beta4,
        fourth_derivative_bunching_margin=bunching_margin,
        third_to_fourth_derivative_cross_coefficient_upper=c43,
        hessian_to_fourth_derivative_cross_coefficient_upper=c42,
        derivative_to_fourth_derivative_cross_coefficient_upper=c41,
        value_to_fourth_derivative_cross_coefficient_upper=c40,
    )
    if failures:
        return QuantitativeC4GraphTransformCertificate(
            status=failures[0],
            validation_level=None,
            robust_interior=False,
            c4_graph_real_dimension=None,
            **common,
        )
    return QuantitativeC4GraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_C4_TRIANGULAR_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_C4_TRIANGULAR_GRAPH_TRANSFORM",
        robust_interior=c3.robust_interior and class_margin > 0,
        c4_graph_real_dimension=base_dimension,
        **common,
    )


def c4_graph_iteration_bound(
    certificate: QuantitativeC4GraphTransformCertificate,
    *,
    initial_value_distance: object,
    initial_derivative_distance: object,
    initial_hessian_distance: object,
    initial_third_derivative_distance: object,
    initial_fourth_derivative_distance: object,
    steps: int,
) -> C4GraphIterationBound:
    """Iterate the exact upper-triangular C0/C1/C2/C3/C4 recurrence."""
    if certificate.validation_level is None:
        raise ValueError("C4 iteration requires a verified C4 graph-transform certificate")
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative built-in integer")
    values = tuple(
        _exact_fraction(value, name)
        for value, name in (
            (initial_value_distance, "initial_value_distance"),
            (initial_derivative_distance, "initial_derivative_distance"),
            (initial_hessian_distance, "initial_hessian_distance"),
            (initial_third_derivative_distance, "initial_third_derivative_distance"),
            (initial_fourth_derivative_distance, "initial_fourth_derivative_distance"),
        )
    )
    if min(values) < 0:
        raise ValueError("initial distances must be nonnegative")
    delta, derivative, hessian, third, fourth = values
    c3 = certificate.c3_certificate
    c2 = c3.c2_certificate
    c1 = c2.c1_certificate
    q = c1.lipschitz_certificate.contraction_factor_upper
    for _ in range(steps):
        next_fourth = (
            certificate.fourth_derivative_bunching_factor_upper * fourth
            + certificate.third_to_fourth_derivative_cross_coefficient_upper * third
            + certificate.hessian_to_fourth_derivative_cross_coefficient_upper * hessian
            + certificate.derivative_to_fourth_derivative_cross_coefficient_upper * derivative
            + certificate.value_to_fourth_derivative_cross_coefficient_upper * delta
        )
        next_third = (
            c3.third_derivative_bunching_factor_upper * third
            + c3.hessian_to_third_derivative_cross_coefficient_upper * hessian
            + c3.derivative_to_third_derivative_cross_coefficient_upper * derivative
            + c3.value_to_third_derivative_cross_coefficient_upper * delta
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
        delta, derivative, hessian, third, fourth = (
            q * delta,
            next_derivative,
            next_hessian,
            next_third,
            next_fourth,
        )
    return C4GraphIterationBound(steps, delta, derivative, hessian, third, fourth)


__all__ = [
    "C4GraphIterationBound",
    "QuantitativeC4GraphTransformCertificate",
    "c4_graph_iteration_bound",
    "quantitative_c4_triangular_graph_transform",
]
