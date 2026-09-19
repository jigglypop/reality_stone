"""Exact C3 certificate for a graph-independent nonaffine coupled base."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_coupled_graph_transform import _exact_fraction
    from .quantitative_nonaffine_coupled_c2_graph_transform import (
        QuantitativeNonaffineCoupledC2GraphTransformCertificate,
        quantitative_nonaffine_coupled_c2_graph_transform,
    )
else:
    from quantitative_coupled_graph_transform import _exact_fraction  # type: ignore[no-redef]
    from quantitative_nonaffine_coupled_c2_graph_transform import (  # type: ignore[no-redef]
        QuantitativeNonaffineCoupledC2GraphTransformCertificate,
        quantitative_nonaffine_coupled_c2_graph_transform,
    )


@dataclass(frozen=True)
class SinePerturbedCoupledBaseC3Bounds:
    inverse_lipschitz_upper: Fraction
    base_map_hessian_upper: Fraction
    base_map_hessian_lipschitz_upper: Fraction
    base_map_third_derivative_lipschitz_upper: Fraction


@dataclass(frozen=True)
class QuantitativeNonaffineCoupledC3GraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    nonaffine_coupled_c2_certificate: QuantitativeNonaffineCoupledC2GraphTransformCertificate
    normalized_base_map_third_derivative_lipschitz: Fraction
    normalized_base_third_derivative_lipschitz: Fraction
    normalized_fiber_third_derivative_lipschitz: Fraction
    normalized_graph_third_derivative_upper: Fraction
    graph_third_derivative_lipschitz_upper: Fraction
    graph_third_derivative_lipschitz_required: bool
    base_third_derivative_upper: Fraction | None
    fiber_third_derivative_upper: Fraction | None
    modified_third_derivative_upper: Fraction | None
    output_graph_third_derivative_upper: Fraction | None
    graph_third_derivative_margin: Fraction | None
    base_third_derivative_lipschitz_upper: Fraction | None
    fiber_third_derivative_lipschitz_upper: Fraction | None
    modified_third_derivative_lipschitz_upper: Fraction | None
    output_graph_third_derivative_lipschitz_upper: Fraction | None
    graph_third_derivative_lipschitz_margin: Fraction | None
    modified_third_difference_hessian_coefficient_upper: Fraction | None
    modified_third_difference_derivative_coefficient_upper: Fraction | None
    modified_third_difference_value_coefficient_upper: Fraction | None
    third_derivative_bunching_factor_upper: Fraction | None
    third_derivative_bunching_margin: Fraction | None
    hessian_to_third_derivative_cross_coefficient_upper: Fraction | None
    derivative_to_third_derivative_cross_coefficient_upper: Fraction | None
    value_to_third_derivative_cross_coefficient_upper: Fraction | None
    c3_graph_real_dimension: int | None


@dataclass(frozen=True)
class NonaffineCoupledC3GraphIterationBound:
    steps: int
    value_distance_upper: Fraction
    derivative_distance_upper: Fraction
    hessian_distance_upper: Fraction
    third_derivative_distance_upper: Fraction


def sine_perturbed_coupled_base_c3_bounds(
    *,
    linear_coefficient: object,
    amplitude_upper: object,
) -> SinePerturbedCoupledBaseC3Bounds:
    """Bounds for phi(x)=lambda*x+a*sin(x), requiring exact lambda>a>=0."""
    linear = _exact_fraction(linear_coefficient, "linear_coefficient")
    amplitude = _exact_fraction(amplitude_upper, "amplitude_upper")
    if linear <= 0 or amplitude < 0 or linear <= amplitude:
        raise ValueError("linear_coefficient must be positive and strictly exceed amplitude_upper")
    return SinePerturbedCoupledBaseC3Bounds(
        inverse_lipschitz_upper=Fraction(1) / (linear - amplitude),
        base_map_hessian_upper=amplitude,
        base_map_hessian_lipschitz_upper=amplitude,
        base_map_third_derivative_lipschitz_upper=amplitude,
    )


def quantitative_nonaffine_coupled_c3_graph_transform(
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
    normalized_base_map_third_derivative_lipschitz: object,
    normalized_base_third_derivative_lipschitz: object,
    normalized_fiber_third_derivative_lipschitz: object,
    normalized_graph_third_derivative_upper: object,
    graph_third_derivative_lipschitz_upper: object,
) -> QuantitativeNonaffineCoupledC3GraphTransformCertificate:
    """Compose nonaffine coupled C2 with exact C3/C3,1 and bunching gates."""
    uphi = _exact_fraction(
        normalized_base_map_third_derivative_lipschitz,
        "normalized_base_map_third_derivative_lipschitz",
    )
    uf = _exact_fraction(
        normalized_base_third_derivative_lipschitz,
        "normalized_base_third_derivative_lipschitz",
    )
    ug = _exact_fraction(
        normalized_fiber_third_derivative_lipschitz,
        "normalized_fiber_third_derivative_lipschitz",
    )
    lambda3 = _exact_fraction(
        normalized_graph_third_derivative_upper,
        "normalized_graph_third_derivative_upper",
    )
    xi3 = _exact_fraction(
        graph_third_derivative_lipschitz_upper,
        "graph_third_derivative_lipschitz_upper",
    )
    if min(uphi, uf, ug, lambda3, xi3) < 0:
        raise ValueError("normalized base-map, map, and graph C3 moduli must be nonnegative")

    c2 = quantitative_nonaffine_coupled_c2_graph_transform(
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
        normalized_base_map_hessian_lipschitz=normalized_base_map_hessian_lipschitz,
        normalized_base_hessian_lipschitz=normalized_base_hessian_lipschitz,
        normalized_fiber_hessian_lipschitz=normalized_fiber_hessian_lipschitz,
        graph_hessian_lipschitz_upper=lambda3,
    )
    c1 = c2.nonaffine_coupled_c1_certificate
    base = c1.coupled_lipschitz_certificate
    xi3_required = base.normalized_fiber_to_base_lipschitz > 0
    common = dict(
        nonaffine_coupled_c2_certificate=c2,
        normalized_base_map_third_derivative_lipschitz=uphi,
        normalized_base_third_derivative_lipschitz=uf,
        normalized_fiber_third_derivative_lipschitz=ug,
        normalized_graph_third_derivative_upper=lambda3,
        graph_third_derivative_lipschitz_upper=xi3,
        graph_third_derivative_lipschitz_required=xi3_required,
    )

    alpha = base.base_invertibility_lower
    q_value = base.transform_contraction_factor_upper
    if alpha <= 0 or q_value is None or c2.validation_level is None:
        failures = c2.failure_codes
        return QuantitativeNonaffineCoupledC3GraphTransformCertificate(
            status=failures[0], validation_level=None, failure_codes=failures,
            robust_interior=False, base_third_derivative_upper=None,
            fiber_third_derivative_upper=None, modified_third_derivative_upper=None,
            output_graph_third_derivative_upper=None, graph_third_derivative_margin=None,
            base_third_derivative_lipschitz_upper=None,
            fiber_third_derivative_lipschitz_upper=None,
            modified_third_derivative_lipschitz_upper=None,
            output_graph_third_derivative_lipschitz_upper=None,
            graph_third_derivative_lipschitz_margin=None,
            modified_third_difference_hessian_coefficient_upper=None,
            modified_third_difference_derivative_coefficient_upper=None,
            modified_third_difference_value_coefficient_upper=None,
            third_derivative_bunching_factor_upper=None,
            third_derivative_bunching_margin=None,
            hessian_to_third_derivative_cross_coefficient_upper=None,
            derivative_to_third_derivative_cross_coefficient_upper=None,
            value_to_third_derivative_cross_coefficient_upper=None,
            c3_graph_real_dimension=None, **common,
        )

    assert None not in (
        c2.base_hessian_lipschitz_upper,
        c2.fiber_hessian_lipschitz_upper,
        c2.first_derivative_transform_lipschitz_upper,
        c2.modified_hessian_upper,
        c2.modified_hessian_lipschitz_upper,
        c2.base_hessian_difference_derivative_coefficient_upper,
        c2.base_hessian_difference_value_coefficient_upper,
        c2.modified_hessian_difference_derivative_coefficient_upper,
        c2.modified_hessian_difference_value_coefficient_upper,
        c2.second_derivative_bunching_factor_upper,
        c2.derivative_to_hessian_cross_coefficient_upper,
        c2.value_to_hessian_cross_coefficient_upper,
        c1.base_jacobian_lipschitz_upper,
        c1.fiber_jacobian_lipschitz_upper,
        c1.preimage_value_coupling_upper,
        c1.state_value_coupling_upper,
        c1.base_jacobian_difference_value_coefficient_upper,
        c1.derivative_bunching_factor_upper,
        c1.derivative_cross_coefficient_upper,
    )
    kappa = base.normalized_graph_slope_upper
    r = 1 + kappa
    lfy = base.normalized_fiber_to_base_lipschitz
    q = base.fiber_factor_upper
    hf = c1.normalized_base_jacobian_lipschitz
    hg = c1.normalized_fiber_jacobian_lipschitz
    tphi = c2.normalized_base_map_hessian_lipschitz
    tf = c2.normalized_base_hessian_lipschitz
    tg = c2.normalized_fiber_hessian_lipschitz
    lambda2 = c1.graph_derivative_lipschitz_upper
    cf = c1.base_jacobian_lipschitz_upper
    cy = c1.fiber_jacobian_lipschitz_upper
    rho = (q * kappa + base.normalized_base_to_fiber_lipschitz) / alpha
    ct = c2.first_derivative_transform_lipschitz_upper
    n = c2.modified_hessian_upper
    cn = c2.modified_hessian_lipschitz_upper
    cp = c2.base_hessian_lipschitz_upper
    rx = c1.preimage_value_coupling_upper
    z = c1.state_value_coupling_upper
    beta1 = c1.derivative_bunching_factor_upper
    c10 = c1.derivative_cross_coefficient_upper
    pd = c2.base_hessian_difference_derivative_coefficient_upper
    pdelta = c2.base_hessian_difference_value_coefficient_upper
    nd = c2.modified_hessian_difference_derivative_coefficient_upper
    ndelta = c2.modified_hessian_difference_value_coefficient_upper
    adelta = c1.base_jacobian_difference_value_coefficient_upper

    cu = tphi + lfy * lambda3 + 3 * hf * lambda2 * r + tf * r**3
    cv = q * lambda3 + 3 * hg * lambda2 * r + tg * r**3
    m = cv + rho * cu + 3 * n * cf / alpha
    lambda3_output = m / alpha**3
    lambda3_margin = lambda3 - lambda3_output

    cu_lip = (
        uphi + lfy * xi3 + 4 * hf * r * lambda3 + 3 * hf * lambda2**2
        + 6 * tf * lambda2 * r**2 + uf * r**4
    )
    cv_lip = (
        q * xi3 + 4 * hg * r * lambda3 + 3 * hg * lambda2**2
        + 6 * tg * lambda2 * r**2 + ug * r**4
    )
    m_lip = (
        cv_lip + ct * cu + rho * cu_lip
        + 3 * (cn * cf / alpha + n * cf**2 / alpha**2 + n * cp / alpha)
    )
    xi3_output = m_lip / alpha**4 + 3 * m * cf / alpha**5
    xi3_margin = xi3 - xi3_output

    ud = 3 * hf * lambda2 + 3 * tf * r**2
    ue = 3 * hf * r
    udelta = (
        uphi * rx + lfy * xi3 * rx + hf * z * lambda3
        + 3 * hf * r * lambda3 * rx + 3 * hf * lambda2**2 * rx
        + 3 * tf * z * lambda2 * r + 3 * tf * r**2 * lambda2 * rx
        + uf * z * r**3
    )
    vd = 3 * hg * lambda2 + 3 * tg * r**2
    ve = 3 * hg * r
    vdelta = (
        q * xi3 * rx + hg * z * lambda3
        + 3 * hg * r * lambda3 * rx + 3 * hg * lambda2**2 * rx
        + 3 * tg * z * lambda2 * r + 3 * tg * r**2 * lambda2 * rx
        + ug * z * r**3
    )
    correction_e = q_value * cf / alpha + n * lfy / alpha
    correction_d = nd * cf / alpha + n * (pd / alpha + cf * lfy / alpha**2)
    correction_delta = (
        ndelta * cf / alpha + n * (pdelta / alpha + cf * adelta / alpha**2)
    )
    me = ve + rho * ue + 3 * correction_e
    md = vd + rho * ud + cu * beta1 + 3 * correction_d
    mdelta = vdelta + rho * udelta + cu * c10 + 3 * correction_delta

    beta3 = q_value / alpha**3
    beta3_margin = 1 - beta3
    c32 = me / alpha**3
    c31 = md / alpha**3 + 3 * m * lfy / alpha**4
    c30 = mdelta / alpha**3 + 3 * m * adelta / alpha**4

    failures = list(c2.failure_codes)
    if lambda3_margin < 0:
        failures.append("NONAFFINE_COUPLED_C3_GRAPH_THIRD_DERIVATIVE_CLASS_NOT_INVARIANT")
    if xi3_required and xi3_margin < 0:
        failures.append("NONAFFINE_COUPLED_C31_GRAPH_CLASS_NOT_INVARIANT")
    if beta3_margin <= 0:
        failures.append("NONAFFINE_COUPLED_C3_THIRD_DERIVATIVE_BUNCHING_NOT_STRICT")
    detailed = dict(
        failure_codes=tuple(failures), base_third_derivative_upper=cu,
        fiber_third_derivative_upper=cv, modified_third_derivative_upper=m,
        output_graph_third_derivative_upper=lambda3_output,
        graph_third_derivative_margin=lambda3_margin,
        base_third_derivative_lipschitz_upper=cu_lip,
        fiber_third_derivative_lipschitz_upper=cv_lip,
        modified_third_derivative_lipschitz_upper=m_lip,
        output_graph_third_derivative_lipschitz_upper=xi3_output,
        graph_third_derivative_lipschitz_margin=xi3_margin,
        modified_third_difference_hessian_coefficient_upper=me,
        modified_third_difference_derivative_coefficient_upper=md,
        modified_third_difference_value_coefficient_upper=mdelta,
        third_derivative_bunching_factor_upper=beta3,
        third_derivative_bunching_margin=beta3_margin,
        hessian_to_third_derivative_cross_coefficient_upper=c32,
        derivative_to_third_derivative_cross_coefficient_upper=c31,
        value_to_third_derivative_cross_coefficient_upper=c30,
    )
    if failures:
        return QuantitativeNonaffineCoupledC3GraphTransformCertificate(
            status=failures[0], validation_level=None, robust_interior=False,
            c3_graph_real_dimension=None, **detailed, **common,
        )
    robust = (
        c2.robust_interior and lambda3_margin > 0
        and (not xi3_required or xi3_margin > 0)
    )
    return QuantitativeNonaffineCoupledC3GraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_NONAFFINE_COUPLED_C3_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_NONAFFINE_COUPLED_C3_GRAPH_TRANSFORM",
        robust_interior=robust, c3_graph_real_dimension=base_dimension,
        **detailed, **common,
    )


def nonaffine_coupled_c3_graph_iteration_bound(
    certificate: QuantitativeNonaffineCoupledC3GraphTransformCertificate,
    *, initial_value_distance: object, initial_derivative_distance: object,
    initial_hessian_distance: object, initial_third_derivative_distance: object,
    steps: int,
) -> NonaffineCoupledC3GraphIterationBound:
    """Iterate the exact nonaffine coupled C0/C1/C2/C3 recurrence."""
    if certificate.validation_level is None:
        raise ValueError("iteration requires a verified nonaffine coupled C3 certificate")
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
    c2 = certificate.nonaffine_coupled_c2_certificate
    c1 = c2.nonaffine_coupled_c1_certificate
    q0 = c1.coupled_lipschitz_certificate.transform_contraction_factor_upper
    assert None not in (
        q0, c1.derivative_bunching_factor_upper, c1.derivative_cross_coefficient_upper,
        c2.second_derivative_bunching_factor_upper,
        c2.derivative_to_hessian_cross_coefficient_upper,
        c2.value_to_hessian_cross_coefficient_upper,
        certificate.third_derivative_bunching_factor_upper,
        certificate.hessian_to_third_derivative_cross_coefficient_upper,
        certificate.derivative_to_third_derivative_cross_coefficient_upper,
        certificate.value_to_third_derivative_cross_coefficient_upper,
    )
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
            q0 * delta, next_derivative, next_hessian, next_third
        )
    return NonaffineCoupledC3GraphIterationBound(steps, delta, derivative, hessian, third)


__all__ = [
    "NonaffineCoupledC3GraphIterationBound",
    "QuantitativeNonaffineCoupledC3GraphTransformCertificate",
    "SinePerturbedCoupledBaseC3Bounds",
    "nonaffine_coupled_c3_graph_iteration_bound",
    "quantitative_nonaffine_coupled_c3_graph_transform",
    "sine_perturbed_coupled_base_c3_bounds",
]
