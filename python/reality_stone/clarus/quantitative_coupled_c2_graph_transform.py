"""Exact C2 certificate for an affine coupled-base graph transform."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_coupled_c1_graph_transform import (
        QuantitativeCoupledC1GraphTransformCertificate,
        quantitative_coupled_c1_graph_transform,
    )
    from .quantitative_coupled_graph_transform import _exact_fraction
else:
    from quantitative_coupled_c1_graph_transform import (  # type: ignore[no-redef]
        QuantitativeCoupledC1GraphTransformCertificate,
        quantitative_coupled_c1_graph_transform,
    )
    from quantitative_coupled_graph_transform import _exact_fraction  # type: ignore[no-redef]


@dataclass(frozen=True)
class QuantitativeCoupledC2GraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    coupled_c1_certificate: QuantitativeCoupledC1GraphTransformCertificate
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
class CoupledC2GraphIterationBound:
    steps: int
    value_distance_upper: Fraction
    derivative_distance_upper: Fraction
    hessian_distance_upper: Fraction


def quantitative_coupled_c2_graph_transform(
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
    normalized_base_jacobian_lipschitz: object,
    normalized_fiber_jacobian_lipschitz: object,
    graph_derivative_lipschitz_upper: object,
    normalized_base_hessian_lipschitz: object,
    normalized_fiber_hessian_lipschitz: object,
    graph_hessian_lipschitz_upper: object,
) -> QuantitativeCoupledC2GraphTransformCertificate:
    """Compose coupled C1 with C2,1 invariance and strict second bunching."""
    c1 = quantitative_coupled_c1_graph_transform(
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
        normalized_base_jacobian_lipschitz=normalized_base_jacobian_lipschitz,
        normalized_fiber_jacobian_lipschitz=normalized_fiber_jacobian_lipschitz,
        graph_derivative_lipschitz_upper=graph_derivative_lipschitz_upper,
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
    if tf < 0 or tg < 0 or xi < 0:
        raise ValueError("normalized map and graph Hessian-Lipschitz bounds must be nonnegative")

    base = c1.coupled_lipschitz_certificate
    fy = base.normalized_fiber_to_base_lipschitz
    xi_required = fy > 0
    common = dict(
        coupled_c1_certificate=c1,
        normalized_base_hessian_lipschitz=tf,
        normalized_fiber_hessian_lipschitz=tg,
        graph_hessian_lipschitz_upper=xi,
        graph_hessian_lipschitz_required=xi_required,
    )
    alpha = base.base_invertibility_lower
    q_value = base.transform_contraction_factor_upper
    if alpha <= 0 or q_value is None:
        return QuantitativeCoupledC2GraphTransformCertificate(
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
    assert c1.derivative_bunching_factor_upper is not None
    assert c1.derivative_cross_coefficient_upper is not None
    hf = c1.normalized_base_jacobian_lipschitz
    hg = c1.normalized_fiber_jacobian_lipschitz
    lambda2 = c1.graph_derivative_lipschitz_upper
    cf = c1.base_jacobian_lipschitz_upper
    cy = c1.fiber_jacobian_lipschitz_upper
    r_x = c1.preimage_value_coupling_upper
    z_factor = c1.state_value_coupling_upper
    beta1 = c1.derivative_bunching_factor_upper
    c1_cross = c1.derivative_cross_coefficient_upper
    kappa = base.normalized_graph_slope_upper
    lfy = base.normalized_fiber_to_base_lipschitz
    lgx = base.normalized_base_to_fiber_lipschitz
    q = base.fiber_factor_upper
    s = q * kappa + lgx
    rho = s / alpha
    one_plus_kappa = 1 + kappa

    modified_hessian = cy + rho * cf
    transform_lipschitz = cy / alpha + s * cf / (alpha * alpha)
    cp = (
        tf * one_plus_kappa * one_plus_kappa * one_plus_kappa
        + 3 * hf * one_plus_kappa * lambda2
        + lfy * xi
    )
    cr = (
        q * xi
        + 3 * hg * one_plus_kappa * lambda2
        + tg * one_plus_kappa * one_plus_kappa * one_plus_kappa
    )
    cn = cr + transform_lipschitz * cf + rho * cp
    xi_output = cn / (alpha**3) + 2 * modified_hessian * cf / (alpha**4)
    xi_margin = xi - xi_output

    a_delta = (
        hf * z_factor * one_plus_kappa
        + lfy * lambda2 * r_x
    )
    pd = 2 * hf * one_plus_kappa
    p_delta = (
        lfy * xi * r_x
        + hf * z_factor * lambda2
        + 2 * hf * one_plus_kappa * lambda2 * r_x
        + tf * z_factor * one_plus_kappa * one_plus_kappa
    )
    rd = 2 * hg * one_plus_kappa
    r_delta = (
        q * xi * r_x
        + hg * z_factor * lambda2
        + 2 * hg * one_plus_kappa * lambda2 * r_x
        + tg * z_factor * one_plus_kappa * one_plus_kappa
    )
    nd = rd + cf * beta1 + rho * pd
    n_delta = r_delta + cf * c1_cross + rho * p_delta
    beta2 = q_value / (alpha * alpha)
    beta2_margin = 1 - beta2
    c21 = nd / (alpha * alpha) + 2 * modified_hessian * lfy / (alpha**3)
    c20 = n_delta / (alpha * alpha) + 2 * modified_hessian * a_delta / (alpha**3)

    failures = list(c1.failure_codes)
    if xi_required and xi_margin < 0:
        failures.append("COUPLED_C21_GRAPH_CLASS_NOT_INVARIANT")
    if beta2_margin <= 0:
        failures.append("COUPLED_C2_SECOND_DERIVATIVE_BUNCHING_NOT_STRICT")
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
        return QuantitativeCoupledC2GraphTransformCertificate(
            status=failures[0],
            validation_level=None,
            robust_interior=False,
            c2_graph_real_dimension=None,
            **detailed,
            **common,
        )
    robust = c1.robust_interior and (not xi_required or xi_margin > 0)
    return QuantitativeCoupledC2GraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_COUPLED_C2_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_COUPLED_C2_GRAPH_TRANSFORM",
        robust_interior=robust,
        c2_graph_real_dimension=base_dimension,
        **detailed,
        **common,
    )


def coupled_c2_graph_iteration_bound(
    certificate: QuantitativeCoupledC2GraphTransformCertificate,
    *,
    initial_value_distance: object,
    initial_derivative_distance: object,
    initial_hessian_distance: object,
    steps: int,
) -> CoupledC2GraphIterationBound:
    """Iterate the exact coupled value/derivative/Hessian recurrence."""
    if certificate.validation_level is None:
        raise ValueError("iteration requires a verified coupled C2 certificate")
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

    c1 = certificate.coupled_c1_certificate
    q_value = c1.coupled_lipschitz_certificate.transform_contraction_factor_upper
    beta1 = c1.derivative_bunching_factor_upper
    c10 = c1.derivative_cross_coefficient_upper
    beta2 = certificate.second_derivative_bunching_factor_upper
    c21 = certificate.derivative_to_hessian_cross_coefficient_upper
    c20 = certificate.value_to_hessian_cross_coefficient_upper
    assert None not in (q_value, beta1, c10, beta2, c21, c20)
    for _ in range(steps):
        next_hessian = (
            beta2 * hessian_distance
            + c21 * derivative_distance
            + c20 * value_distance
        )
        next_derivative = beta1 * derivative_distance + c10 * value_distance
        next_value = q_value * value_distance
        value_distance = next_value
        derivative_distance = next_derivative
        hessian_distance = next_hessian
    return CoupledC2GraphIterationBound(
        steps,
        value_distance,
        derivative_distance,
        hessian_distance,
    )


__all__ = [
    "CoupledC2GraphIterationBound",
    "QuantitativeCoupledC2GraphTransformCertificate",
    "coupled_c2_graph_iteration_bound",
    "quantitative_coupled_c2_graph_transform",
]
