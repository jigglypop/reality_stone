"""Exact C1 certificate for a nonaffine graph-independent coupled base."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_coupled_graph_transform import (
        QuantitativeCoupledGraphTransformCertificate,
        _exact_fraction,
        quantitative_coupled_graph_transform,
    )
else:
    from quantitative_coupled_graph_transform import (  # type: ignore[no-redef]
        QuantitativeCoupledGraphTransformCertificate,
        _exact_fraction,
        quantitative_coupled_graph_transform,
    )


@dataclass(frozen=True)
class SinePerturbedCoupledBaseC1Bounds:
    inverse_lipschitz_upper: Fraction
    base_map_hessian_upper: Fraction


@dataclass(frozen=True)
class QuantitativeNonaffineCoupledC1GraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    coupled_lipschitz_certificate: QuantitativeCoupledGraphTransformCertificate
    normalized_base_map_hessian_upper: Fraction
    normalized_base_jacobian_lipschitz: Fraction
    normalized_fiber_jacobian_lipschitz: Fraction
    graph_derivative_lipschitz_upper: Fraction
    base_jacobian_lipschitz_upper: Fraction | None
    fiber_jacobian_lipschitz_upper: Fraction | None
    output_graph_derivative_lipschitz_upper: Fraction | None
    graph_derivative_lipschitz_margin: Fraction | None
    preimage_value_coupling_upper: Fraction | None
    state_value_coupling_upper: Fraction | None
    base_jacobian_difference_value_coefficient_upper: Fraction | None
    fiber_jacobian_difference_value_coefficient_upper: Fraction | None
    derivative_bunching_factor_upper: Fraction | None
    derivative_bunching_margin: Fraction | None
    derivative_cross_coefficient_upper: Fraction | None
    c1_graph_real_dimension: int | None


@dataclass(frozen=True)
class NonaffineCoupledC1GraphIterationBound:
    steps: int
    value_distance_upper: Fraction
    derivative_distance_upper: Fraction


def sine_perturbed_coupled_base_c1_bounds(
    *,
    linear_coefficient: object,
    amplitude_upper: object,
) -> SinePerturbedCoupledBaseC1Bounds:
    """Bounds for phi(x)=lambda*x+a*sin(x), requiring exact lambda>a>=0."""
    linear = _exact_fraction(linear_coefficient, "linear_coefficient")
    amplitude = _exact_fraction(amplitude_upper, "amplitude_upper")
    if linear <= 0 or amplitude < 0 or linear <= amplitude:
        raise ValueError("linear_coefficient must be positive and strictly exceed amplitude_upper")
    return SinePerturbedCoupledBaseC1Bounds(
        inverse_lipschitz_upper=Fraction(1) / (linear - amplitude),
        base_map_hessian_upper=amplitude,
    )


def quantitative_nonaffine_coupled_c1_graph_transform(
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
    normalized_base_map_hessian_upper: object,
    normalized_base_jacobian_lipschitz: object,
    normalized_fiber_jacobian_lipschitz: object,
    graph_derivative_lipschitz_upper: object,
) -> QuantitativeNonaffineCoupledC1GraphTransformCertificate:
    """Add base-map curvature to the exact coupled C1 graph-transform gates."""
    base = quantitative_coupled_graph_transform(
        base_dimension=base_dimension,
        base_reference_scale=base_reference_scale,
        fiber_reference_scale=fiber_reference_scale,
        fiber_radius=fiber_radius,
        forcing_at_zero_upper=forcing_at_zero_upper,
        base_inverse_lipschitz=base_inverse_lipschitz,
        fiber_linear_norm_upper=fiber_linear_norm_upper,
        base_self_lipschitz=base_self_lipschitz,
        fiber_to_base_lipschitz=fiber_to_base_lipschitz,
        base_to_fiber_lipschitz=base_to_fiber_lipschitz,
        fiber_self_lipschitz=fiber_self_lipschitz,
        graph_slope_upper=graph_slope_upper,
    )
    hphi = _exact_fraction(
        normalized_base_map_hessian_upper,
        "normalized_base_map_hessian_upper",
    )
    hf = _exact_fraction(
        normalized_base_jacobian_lipschitz,
        "normalized_base_jacobian_lipschitz",
    )
    hg = _exact_fraction(
        normalized_fiber_jacobian_lipschitz,
        "normalized_fiber_jacobian_lipschitz",
    )
    lambda2 = _exact_fraction(
        graph_derivative_lipschitz_upper,
        "graph_derivative_lipschitz_upper",
    )
    if hphi < 0 or hf < 0 or hg < 0 or lambda2 < 0:
        raise ValueError("normalized base curvature, Jacobian, and graph bounds must be nonnegative")

    failures = list(base.failure_codes)
    common = dict(
        coupled_lipschitz_certificate=base,
        normalized_base_map_hessian_upper=hphi,
        normalized_base_jacobian_lipschitz=hf,
        normalized_fiber_jacobian_lipschitz=hg,
        graph_derivative_lipschitz_upper=lambda2,
    )
    alpha = base.base_invertibility_lower
    q_value = base.transform_contraction_factor_upper
    if alpha <= 0 or q_value is None:
        return QuantitativeNonaffineCoupledC1GraphTransformCertificate(
            status=failures[0],
            validation_level=None,
            failure_codes=tuple(failures),
            robust_interior=False,
            base_jacobian_lipschitz_upper=None,
            fiber_jacobian_lipschitz_upper=None,
            output_graph_derivative_lipschitz_upper=None,
            graph_derivative_lipschitz_margin=None,
            preimage_value_coupling_upper=None,
            state_value_coupling_upper=None,
            base_jacobian_difference_value_coefficient_upper=None,
            fiber_jacobian_difference_value_coefficient_upper=None,
            derivative_bunching_factor_upper=None,
            derivative_bunching_margin=None,
            derivative_cross_coefficient_upper=None,
            c1_graph_real_dimension=None,
            **common,
        )

    kappa = base.normalized_graph_slope_upper
    lfy = base.normalized_fiber_to_base_lipschitz
    q = base.fiber_factor_upper
    lgx = base.normalized_base_to_fiber_lipschitz
    s = q * kappa + lgx
    one_plus_kappa = 1 + kappa
    cf = hphi + hf * one_plus_kappa * one_plus_kappa + lfy * lambda2
    cy = q * lambda2 + hg * one_plus_kappa * one_plus_kappa
    output_lipschitz = cy / (alpha**2) + s * cf / (alpha**3)
    class_margin = lambda2 - output_lipschitz
    if class_margin < 0:
        failures.append("NONAFFINE_COUPLED_C11_GRAPH_CLASS_NOT_INVARIANT")

    r_x = lfy / alpha
    z_factor = 1 + one_plus_kappa * r_x
    a_delta = (
        hphi * r_x
        + hf * z_factor * one_plus_kappa
        + lfy * lambda2 * r_x
    )
    y_delta = q * lambda2 * r_x + hg * z_factor * one_plus_kappa
    beta1 = q_value / alpha
    beta1_margin = 1 - beta1
    if beta1_margin <= 0:
        failures.append("NONAFFINE_COUPLED_C1_DERIVATIVE_BUNCHING_NOT_STRICT")
    c10 = y_delta / alpha + s * a_delta / (alpha**2)

    detailed = dict(
        failure_codes=tuple(failures),
        base_jacobian_lipschitz_upper=cf,
        fiber_jacobian_lipschitz_upper=cy,
        output_graph_derivative_lipschitz_upper=output_lipschitz,
        graph_derivative_lipschitz_margin=class_margin,
        preimage_value_coupling_upper=r_x,
        state_value_coupling_upper=z_factor,
        base_jacobian_difference_value_coefficient_upper=a_delta,
        fiber_jacobian_difference_value_coefficient_upper=y_delta,
        derivative_bunching_factor_upper=beta1,
        derivative_bunching_margin=beta1_margin,
        derivative_cross_coefficient_upper=c10,
    )
    if failures:
        return QuantitativeNonaffineCoupledC1GraphTransformCertificate(
            status=failures[0],
            validation_level=None,
            robust_interior=False,
            c1_graph_real_dimension=None,
            **detailed,
            **common,
        )
    return QuantitativeNonaffineCoupledC1GraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_NONAFFINE_COUPLED_C1_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_NONAFFINE_COUPLED_C1_GRAPH_TRANSFORM",
        robust_interior=base.robust_interior and class_margin > 0,
        c1_graph_real_dimension=base_dimension,
        **detailed,
        **common,
    )


def nonaffine_coupled_c1_graph_iteration_bound(
    certificate: QuantitativeNonaffineCoupledC1GraphTransformCertificate,
    *,
    initial_value_distance: object,
    initial_derivative_distance: object,
    steps: int,
) -> NonaffineCoupledC1GraphIterationBound:
    """Iterate the exact nonaffine coupled value/derivative recurrence."""
    if certificate.validation_level is None:
        raise ValueError("iteration requires a verified nonaffine coupled C1 certificate")
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative built-in integer")
    delta = _exact_fraction(initial_value_distance, "initial_value_distance")
    derivative = _exact_fraction(initial_derivative_distance, "initial_derivative_distance")
    if delta < 0 or derivative < 0:
        raise ValueError("initial distances must be nonnegative")
    q_value = certificate.coupled_lipschitz_certificate.transform_contraction_factor_upper
    beta1 = certificate.derivative_bunching_factor_upper
    c10 = certificate.derivative_cross_coefficient_upper
    assert q_value is not None and beta1 is not None and c10 is not None
    for _ in range(steps):
        next_derivative = beta1 * derivative + c10 * delta
        next_delta = q_value * delta
        delta, derivative = next_delta, next_derivative
    return NonaffineCoupledC1GraphIterationBound(steps, delta, derivative)


__all__ = [
    "NonaffineCoupledC1GraphIterationBound",
    "QuantitativeNonaffineCoupledC1GraphTransformCertificate",
    "SinePerturbedCoupledBaseC1Bounds",
    "nonaffine_coupled_c1_graph_iteration_bound",
    "quantitative_nonaffine_coupled_c1_graph_transform",
    "sine_perturbed_coupled_base_c1_bounds",
]

