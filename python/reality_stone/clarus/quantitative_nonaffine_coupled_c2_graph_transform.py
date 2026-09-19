"""Exact C2 certificate for a nonaffine graph-independent coupled base."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_coupled_graph_transform import _exact_fraction
    from .quantitative_nonaffine_coupled_c1_graph_transform import (
        QuantitativeNonaffineCoupledC1GraphTransformCertificate,
        quantitative_nonaffine_coupled_c1_graph_transform,
    )
else:
    from quantitative_coupled_graph_transform import _exact_fraction  # type: ignore[no-redef]
    from quantitative_nonaffine_coupled_c1_graph_transform import (  # type: ignore[no-redef]
        QuantitativeNonaffineCoupledC1GraphTransformCertificate,
        quantitative_nonaffine_coupled_c1_graph_transform,
    )


@dataclass(frozen=True)
class SinePerturbedCoupledBaseC2Bounds:
    inverse_lipschitz_upper: Fraction
    base_map_hessian_upper: Fraction
    base_map_hessian_lipschitz_upper: Fraction


@dataclass(frozen=True)
class QuantitativeNonaffineCoupledC2GraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    nonaffine_coupled_c1_certificate: QuantitativeNonaffineCoupledC1GraphTransformCertificate
    normalized_base_map_hessian_lipschitz: Fraction
    normalized_base_hessian_lipschitz: Fraction
    normalized_fiber_hessian_lipschitz: Fraction
    graph_hessian_lipschitz_upper: Fraction
    graph_hessian_lipschitz_required: bool
    base_hessian_lipschitz_upper: Fraction | None
    fiber_hessian_lipschitz_upper: Fraction | None
    first_derivative_transform_lipschitz_upper: Fraction | None
    modified_hessian_upper: Fraction | None
    modified_hessian_lipschitz_upper: Fraction | None
    output_graph_hessian_lipschitz_upper: Fraction | None
    graph_hessian_lipschitz_margin: Fraction | None
    base_hessian_difference_derivative_coefficient_upper: Fraction | None
    base_hessian_difference_value_coefficient_upper: Fraction | None
    fiber_hessian_difference_derivative_coefficient_upper: Fraction | None
    fiber_hessian_difference_value_coefficient_upper: Fraction | None
    modified_hessian_difference_derivative_coefficient_upper: Fraction | None
    modified_hessian_difference_value_coefficient_upper: Fraction | None
    second_derivative_bunching_factor_upper: Fraction | None
    second_derivative_bunching_margin: Fraction | None
    derivative_to_hessian_cross_coefficient_upper: Fraction | None
    value_to_hessian_cross_coefficient_upper: Fraction | None
    c2_graph_real_dimension: int | None


@dataclass(frozen=True)
class NonaffineCoupledC2GraphIterationBound:
    steps: int
    value_distance_upper: Fraction
    derivative_distance_upper: Fraction
    hessian_distance_upper: Fraction


def sine_perturbed_coupled_base_c2_bounds(
    *,
    linear_coefficient: object,
    amplitude_upper: object,
) -> SinePerturbedCoupledBaseC2Bounds:
    """Bounds for phi(x)=lambda*x+a*sin(x), requiring exact lambda>a>=0."""
    linear = _exact_fraction(linear_coefficient, "linear_coefficient")
    amplitude = _exact_fraction(amplitude_upper, "amplitude_upper")
    if linear <= 0 or amplitude < 0 or linear <= amplitude:
        raise ValueError("linear_coefficient must be positive and strictly exceed amplitude_upper")
    return SinePerturbedCoupledBaseC2Bounds(
        inverse_lipschitz_upper=Fraction(1) / (linear - amplitude),
        base_map_hessian_upper=amplitude,
        base_map_hessian_lipschitz_upper=amplitude,
    )


def quantitative_nonaffine_coupled_c2_graph_transform(
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
    normalized_base_map_hessian_lipschitz: object,
    normalized_base_hessian_lipschitz: object,
    normalized_fiber_hessian_lipschitz: object,
    graph_hessian_lipschitz_upper: object,
) -> QuantitativeNonaffineCoupledC2GraphTransformCertificate:
    """Compose nonaffine coupled C1 with C2,1 and second bunching gates."""
    c1 = quantitative_nonaffine_coupled_c1_graph_transform(
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
        normalized_base_map_hessian_upper=normalized_base_map_hessian_upper,
        normalized_base_jacobian_lipschitz=normalized_base_jacobian_lipschitz,
        normalized_fiber_jacobian_lipschitz=normalized_fiber_jacobian_lipschitz,
        graph_derivative_lipschitz_upper=graph_derivative_lipschitz_upper,
    )
    tphi = _exact_fraction(
        normalized_base_map_hessian_lipschitz,
        "normalized_base_map_hessian_lipschitz",
    )
    tf = _exact_fraction(
        normalized_base_hessian_lipschitz,
        "normalized_base_hessian_lipschitz",
    )
    tg = _exact_fraction(
        normalized_fiber_hessian_lipschitz,
        "normalized_fiber_hessian_lipschitz",
    )
    xi = _exact_fraction(
        graph_hessian_lipschitz_upper,
        "graph_hessian_lipschitz_upper",
    )
    if tphi < 0 or tf < 0 or tg < 0 or xi < 0:
        raise ValueError("normalized base-map, map, and graph Hessian moduli must be nonnegative")

    base = c1.coupled_lipschitz_certificate
    lfy = base.normalized_fiber_to_base_lipschitz
    xi_required = lfy > 0
    common = dict(
        nonaffine_coupled_c1_certificate=c1,
        normalized_base_map_hessian_lipschitz=tphi,
        normalized_base_hessian_lipschitz=tf,
        normalized_fiber_hessian_lipschitz=tg,
        graph_hessian_lipschitz_upper=xi,
        graph_hessian_lipschitz_required=xi_required,
    )
    alpha = base.base_invertibility_lower
    q_value = base.transform_contraction_factor_upper
    if alpha <= 0 or q_value is None:
        return QuantitativeNonaffineCoupledC2GraphTransformCertificate(
            status=c1.failure_codes[0],
            validation_level=None,
            failure_codes=c1.failure_codes,
            robust_interior=False,
            base_hessian_lipschitz_upper=None,
            fiber_hessian_lipschitz_upper=None,
            first_derivative_transform_lipschitz_upper=None,
            modified_hessian_upper=None,
            modified_hessian_lipschitz_upper=None,
            output_graph_hessian_lipschitz_upper=None,
            graph_hessian_lipschitz_margin=None,
            base_hessian_difference_derivative_coefficient_upper=None,
            base_hessian_difference_value_coefficient_upper=None,
            fiber_hessian_difference_derivative_coefficient_upper=None,
            fiber_hessian_difference_value_coefficient_upper=None,
            modified_hessian_difference_derivative_coefficient_upper=None,
            modified_hessian_difference_value_coefficient_upper=None,
            second_derivative_bunching_factor_upper=None,
            second_derivative_bunching_margin=None,
            derivative_to_hessian_cross_coefficient_upper=None,
            value_to_hessian_cross_coefficient_upper=None,
            c2_graph_real_dimension=None,
            **common,
        )

    assert c1.base_jacobian_lipschitz_upper is not None
    assert c1.fiber_jacobian_lipschitz_upper is not None
    assert c1.preimage_value_coupling_upper is not None
    assert c1.state_value_coupling_upper is not None
    assert c1.base_jacobian_difference_value_coefficient_upper is not None
    assert c1.derivative_bunching_factor_upper is not None
    assert c1.derivative_cross_coefficient_upper is not None
    hf = c1.normalized_base_jacobian_lipschitz
    hg = c1.normalized_fiber_jacobian_lipschitz
    lambda2 = c1.graph_derivative_lipschitz_upper
    cf = c1.base_jacobian_lipschitz_upper
    cy = c1.fiber_jacobian_lipschitz_upper
    r_x = c1.preimage_value_coupling_upper
    z_factor = c1.state_value_coupling_upper
    a_delta = c1.base_jacobian_difference_value_coefficient_upper
    beta1 = c1.derivative_bunching_factor_upper
    c1_cross = c1.derivative_cross_coefficient_upper
    kappa = base.normalized_graph_slope_upper
    lgx = base.normalized_base_to_fiber_lipschitz
    q = base.fiber_factor_upper
    s = q * kappa + lgx
    rho = s / alpha
    one_plus_kappa = 1 + kappa

    modified_hessian = cy + rho * cf
    transform_lipschitz = cy / alpha + s * cf / (alpha**2)
    cp = (
        tphi
        + tf * one_plus_kappa**3
        + 3 * hf * one_plus_kappa * lambda2
        + lfy * xi
    )
    cr = q * xi + 3 * hg * one_plus_kappa * lambda2 + tg * one_plus_kappa**3
    cn = cr + transform_lipschitz * cf + rho * cp
    xi_output = cn / (alpha**3) + 2 * modified_hessian * cf / (alpha**4)
    xi_margin = xi - xi_output

    pd = 2 * hf * one_plus_kappa
    p_delta = (
        tphi * r_x
        + lfy * xi * r_x
        + hf * z_factor * lambda2
        + 2 * hf * one_plus_kappa * lambda2 * r_x
        + tf * z_factor * one_plus_kappa**2
    )
    rd = 2 * hg * one_plus_kappa
    r_delta = (
        q * xi * r_x
        + hg * z_factor * lambda2
        + 2 * hg * one_plus_kappa * lambda2 * r_x
        + tg * z_factor * one_plus_kappa**2
    )
    nd = rd + cf * beta1 + rho * pd
    n_delta = r_delta + cf * c1_cross + rho * p_delta
    beta2 = q_value / (alpha**2)
    beta2_margin = 1 - beta2
    c21 = nd / (alpha**2) + 2 * modified_hessian * lfy / (alpha**3)
    c20 = n_delta / (alpha**2) + 2 * modified_hessian * a_delta / (alpha**3)

    failures = list(c1.failure_codes)
    if xi_required and xi_margin < 0:
        failures.append("NONAFFINE_COUPLED_C21_GRAPH_CLASS_NOT_INVARIANT")
    if beta2_margin <= 0:
        failures.append("NONAFFINE_COUPLED_C2_SECOND_DERIVATIVE_BUNCHING_NOT_STRICT")
    detailed = dict(
        failure_codes=tuple(failures),
        base_hessian_lipschitz_upper=cp,
        fiber_hessian_lipschitz_upper=cr,
        first_derivative_transform_lipschitz_upper=transform_lipschitz,
        modified_hessian_upper=modified_hessian,
        modified_hessian_lipschitz_upper=cn,
        output_graph_hessian_lipschitz_upper=xi_output,
        graph_hessian_lipschitz_margin=xi_margin,
        base_hessian_difference_derivative_coefficient_upper=pd,
        base_hessian_difference_value_coefficient_upper=p_delta,
        fiber_hessian_difference_derivative_coefficient_upper=rd,
        fiber_hessian_difference_value_coefficient_upper=r_delta,
        modified_hessian_difference_derivative_coefficient_upper=nd,
        modified_hessian_difference_value_coefficient_upper=n_delta,
        second_derivative_bunching_factor_upper=beta2,
        second_derivative_bunching_margin=beta2_margin,
        derivative_to_hessian_cross_coefficient_upper=c21,
        value_to_hessian_cross_coefficient_upper=c20,
    )
    if failures:
        return QuantitativeNonaffineCoupledC2GraphTransformCertificate(
            status=failures[0],
            validation_level=None,
            robust_interior=False,
            c2_graph_real_dimension=None,
            **detailed,
            **common,
        )
    robust = c1.robust_interior and (not xi_required or xi_margin > 0)
    return QuantitativeNonaffineCoupledC2GraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_NONAFFINE_COUPLED_C2_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_NONAFFINE_COUPLED_C2_GRAPH_TRANSFORM",
        robust_interior=robust,
        c2_graph_real_dimension=base_dimension,
        **detailed,
        **common,
    )


def nonaffine_coupled_c2_graph_iteration_bound(
    certificate: QuantitativeNonaffineCoupledC2GraphTransformCertificate,
    *,
    initial_value_distance: object,
    initial_derivative_distance: object,
    initial_hessian_distance: object,
    steps: int,
) -> NonaffineCoupledC2GraphIterationBound:
    """Iterate the exact nonaffine coupled C0/C1/C2 recurrence."""
    if certificate.validation_level is None:
        raise ValueError("iteration requires a verified nonaffine coupled C2 certificate")
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative built-in integer")
    delta = _exact_fraction(initial_value_distance, "initial_value_distance")
    derivative = _exact_fraction(initial_derivative_distance, "initial_derivative_distance")
    hessian = _exact_fraction(initial_hessian_distance, "initial_hessian_distance")
    if delta < 0 or derivative < 0 or hessian < 0:
        raise ValueError("initial distances must be nonnegative")
    c1 = certificate.nonaffine_coupled_c1_certificate
    q_value = c1.coupled_lipschitz_certificate.transform_contraction_factor_upper
    beta1 = c1.derivative_bunching_factor_upper
    c10 = c1.derivative_cross_coefficient_upper
    beta2 = certificate.second_derivative_bunching_factor_upper
    c21 = certificate.derivative_to_hessian_cross_coefficient_upper
    c20 = certificate.value_to_hessian_cross_coefficient_upper
    assert None not in (q_value, beta1, c10, beta2, c21, c20)
    for _ in range(steps):
        next_hessian = beta2 * hessian + c21 * derivative + c20 * delta
        next_derivative = beta1 * derivative + c10 * delta
        next_delta = q_value * delta
        delta, derivative, hessian = next_delta, next_derivative, next_hessian
    return NonaffineCoupledC2GraphIterationBound(steps, delta, derivative, hessian)


__all__ = [
    "NonaffineCoupledC2GraphIterationBound",
    "QuantitativeNonaffineCoupledC2GraphTransformCertificate",
    "SinePerturbedCoupledBaseC2Bounds",
    "nonaffine_coupled_c2_graph_iteration_bound",
    "quantitative_nonaffine_coupled_c2_graph_transform",
    "sine_perturbed_coupled_base_c2_bounds",
]
