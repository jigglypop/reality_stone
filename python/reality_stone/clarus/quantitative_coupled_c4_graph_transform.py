"""Exact C4 certificate for an affine graph-dependent coupled base."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_coupled_c3_graph_transform import (
        QuantitativeCoupledC3GraphTransformCertificate,
        quantitative_coupled_c3_graph_transform,
    )
    from .quantitative_coupled_graph_transform import _exact_fraction
else:
    from quantitative_coupled_c3_graph_transform import (  # type: ignore[no-redef]
        QuantitativeCoupledC3GraphTransformCertificate,
        quantitative_coupled_c3_graph_transform,
    )
    from quantitative_coupled_graph_transform import _exact_fraction  # type: ignore[no-redef]


Vector5 = tuple[Fraction, Fraction, Fraction, Fraction, Fraction]


def _vadd(*vectors: Vector5) -> Vector5:
    return tuple(sum(vector[index] for vector in vectors) for index in range(5))  # type: ignore[return-value]


def _vscale(scale: Fraction, vector: Vector5) -> Vector5:
    return tuple(scale * value for value in vector)  # type: ignore[return-value]


@dataclass(frozen=True)
class QuantitativeCoupledC4GraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    coupled_c3_certificate: QuantitativeCoupledC3GraphTransformCertificate
    normalized_base_fourth_derivative_lipschitz: Fraction
    normalized_fiber_fourth_derivative_lipschitz: Fraction
    normalized_graph_fourth_derivative_upper: Fraction
    graph_fourth_derivative_lipschitz_upper: Fraction
    graph_fourth_derivative_lipschitz_required: bool
    base_fourth_derivative_upper: Fraction | None
    fiber_fourth_derivative_upper: Fraction | None
    modified_fourth_derivative_upper: Fraction | None
    output_graph_fourth_derivative_upper: Fraction | None
    graph_fourth_derivative_margin: Fraction | None
    base_fourth_derivative_lipschitz_upper: Fraction | None
    fiber_fourth_derivative_lipschitz_upper: Fraction | None
    modified_fourth_derivative_lipschitz_upper: Fraction | None
    output_graph_fourth_derivative_lipschitz_upper: Fraction | None
    graph_fourth_derivative_lipschitz_margin: Fraction | None
    modified_fourth_difference_third_coefficient_upper: Fraction | None
    modified_fourth_difference_hessian_coefficient_upper: Fraction | None
    modified_fourth_difference_derivative_coefficient_upper: Fraction | None
    modified_fourth_difference_value_coefficient_upper: Fraction | None
    fourth_derivative_bunching_factor_upper: Fraction | None
    fourth_derivative_bunching_margin: Fraction | None
    third_to_fourth_derivative_cross_coefficient_upper: Fraction | None
    hessian_to_fourth_derivative_cross_coefficient_upper: Fraction | None
    derivative_to_fourth_derivative_cross_coefficient_upper: Fraction | None
    value_to_fourth_derivative_cross_coefficient_upper: Fraction | None
    c4_graph_real_dimension: int | None


@dataclass(frozen=True)
class CoupledC4GraphIterationBound:
    steps: int
    value_distance_upper: Fraction
    derivative_distance_upper: Fraction
    hessian_distance_upper: Fraction
    third_derivative_distance_upper: Fraction
    fourth_derivative_distance_upper: Fraction


def quantitative_coupled_c4_graph_transform(
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
    normalized_base_third_derivative_lipschitz: object,
    normalized_fiber_third_derivative_lipschitz: object,
    normalized_graph_third_derivative_upper: object,
    graph_third_derivative_lipschitz_upper: object,
    normalized_base_fourth_derivative_lipschitz: object,
    normalized_fiber_fourth_derivative_lipschitz: object,
    normalized_graph_fourth_derivative_upper: object,
    graph_fourth_derivative_lipschitz_upper: object,
) -> QuantitativeCoupledC4GraphTransformCertificate:
    """Compose coupled C3 with the exact modified-fourth-tensor gates."""
    vf = _exact_fraction(
        normalized_base_fourth_derivative_lipschitz,
        "normalized_base_fourth_derivative_lipschitz",
    )
    vg = _exact_fraction(
        normalized_fiber_fourth_derivative_lipschitz,
        "normalized_fiber_fourth_derivative_lipschitz",
    )
    lambda4 = _exact_fraction(
        normalized_graph_fourth_derivative_upper,
        "normalized_graph_fourth_derivative_upper",
    )
    xi4 = _exact_fraction(
        graph_fourth_derivative_lipschitz_upper,
        "graph_fourth_derivative_lipschitz_upper",
    )
    if min(vf, vg, lambda4, xi4) < 0:
        raise ValueError("normalized C4 and D5-level modulus bounds must be nonnegative")

    c3 = quantitative_coupled_c3_graph_transform(
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
        normalized_base_hessian_lipschitz=normalized_base_hessian_lipschitz,
        normalized_fiber_hessian_lipschitz=normalized_fiber_hessian_lipschitz,
        normalized_base_third_derivative_lipschitz=normalized_base_third_derivative_lipschitz,
        normalized_fiber_third_derivative_lipschitz=normalized_fiber_third_derivative_lipschitz,
        normalized_graph_third_derivative_upper=normalized_graph_third_derivative_upper,
        graph_third_derivative_lipschitz_upper=graph_third_derivative_lipschitz_upper,
    )
    c2 = c3.coupled_c2_certificate
    c1 = c2.coupled_c1_certificate
    base = c1.coupled_lipschitz_certificate
    xi4_required = base.normalized_fiber_to_base_lipschitz > 0
    common = dict(
        coupled_c3_certificate=c3,
        normalized_base_fourth_derivative_lipschitz=vf,
        normalized_fiber_fourth_derivative_lipschitz=vg,
        normalized_graph_fourth_derivative_upper=lambda4,
        graph_fourth_derivative_lipschitz_upper=xi4,
        graph_fourth_derivative_lipschitz_required=xi4_required,
    )

    alpha = base.base_invertibility_lower
    q_value = base.transform_contraction_factor_upper
    if alpha <= 0 or q_value is None or c3.validation_level is None:
        failures = c3.failure_codes
        empty = dict(
            base_fourth_derivative_upper=None,
            fiber_fourth_derivative_upper=None,
            modified_fourth_derivative_upper=None,
            output_graph_fourth_derivative_upper=None,
            graph_fourth_derivative_margin=None,
            base_fourth_derivative_lipschitz_upper=None,
            fiber_fourth_derivative_lipschitz_upper=None,
            modified_fourth_derivative_lipschitz_upper=None,
            output_graph_fourth_derivative_lipschitz_upper=None,
            graph_fourth_derivative_lipschitz_margin=None,
            modified_fourth_difference_third_coefficient_upper=None,
            modified_fourth_difference_hessian_coefficient_upper=None,
            modified_fourth_difference_derivative_coefficient_upper=None,
            modified_fourth_difference_value_coefficient_upper=None,
            fourth_derivative_bunching_factor_upper=None,
            fourth_derivative_bunching_margin=None,
            third_to_fourth_derivative_cross_coefficient_upper=None,
            hessian_to_fourth_derivative_cross_coefficient_upper=None,
            derivative_to_fourth_derivative_cross_coefficient_upper=None,
            value_to_fourth_derivative_cross_coefficient_upper=None,
        )
        return QuantitativeCoupledC4GraphTransformCertificate(
            status=failures[0], validation_level=None, failure_codes=failures,
            robust_interior=False, c4_graph_real_dimension=None,
            **empty, **common,
        )

    assert None not in (
        c1.base_jacobian_lipschitz_upper,
        c1.fiber_jacobian_lipschitz_upper,
        c1.preimage_value_coupling_upper,
        c1.state_value_coupling_upper,
        c1.derivative_bunching_factor_upper,
        c1.derivative_cross_coefficient_upper,
        c2.base_hessian_lipschitz_upper,
        c2.first_derivative_transform_lipschitz_upper,
        c2.modified_hessian_upper,
        c2.modified_hessian_lipschitz_upper,
        c2.base_hessian_difference_derivative_coefficient_upper,
        c2.base_hessian_difference_value_coefficient_upper,
        c2.modified_hessian_difference_derivative_coefficient_upper,
        c2.modified_hessian_difference_value_coefficient_upper,
        c3.base_third_derivative_upper,
        c3.fiber_third_derivative_upper,
        c3.modified_third_derivative_upper,
        c3.base_third_derivative_lipschitz_upper,
        c3.modified_third_derivative_lipschitz_upper,
        c3.modified_third_difference_hessian_coefficient_upper,
        c3.modified_third_difference_derivative_coefficient_upper,
        c3.modified_third_difference_value_coefficient_upper,
    )
    kappa = base.normalized_graph_slope_upper
    r = 1 + kappa
    lfy = base.normalized_fiber_to_base_lipschitz
    q = base.fiber_factor_upper
    hf = c1.normalized_base_jacobian_lipschitz
    hg = c1.normalized_fiber_jacobian_lipschitz
    tf = c2.normalized_base_hessian_lipschitz
    tg = c2.normalized_fiber_hessian_lipschitz
    uf = c3.normalized_base_third_derivative_lipschitz
    ug = c3.normalized_fiber_third_derivative_lipschitz
    lambda2 = c1.graph_derivative_lipschitz_upper
    lambda3 = c3.normalized_graph_third_derivative_upper
    xi3 = c3.graph_third_derivative_lipschitz_upper
    cf = c1.base_jacobian_lipschitz_upper
    rho = (q * kappa + base.normalized_base_to_fiber_lipschitz) / alpha
    ct = c2.first_derivative_transform_lipschitz_upper
    n = c2.modified_hessian_upper
    cn = c2.modified_hessian_lipschitz_upper
    cp = c2.base_hessian_lipschitz_upper
    cu = c3.base_third_derivative_upper
    cu_lip = c3.base_third_derivative_lipschitz_upper
    m = c3.modified_third_derivative_upper
    cm = c3.modified_third_derivative_lipschitz_upper
    rx = c1.preimage_value_coupling_upper
    state = c1.state_value_coupling_upper
    beta1 = c1.derivative_bunching_factor_upper
    c10 = c1.derivative_cross_coefficient_upper
    pd = c2.base_hessian_difference_derivative_coefficient_upper
    pdelta = c2.base_hessian_difference_value_coefficient_upper
    nd = c2.modified_hessian_difference_derivative_coefficient_upper
    ndelta = c2.modified_hessian_difference_value_coefficient_upper
    me = c3.modified_third_difference_hessian_coefficient_upper
    md = c3.modified_third_difference_derivative_coefficient_upper
    mdelta = c3.modified_third_difference_value_coefficient_upper
    adelta = hf * state * r + lfy * lambda2 * rx

    cw = (
        lfy * lambda4 + 4 * hf * lambda3 * r + 3 * hf * lambda2**2
        + 6 * tf * lambda2 * r**2 + uf * r**4
    )
    cz = (
        q * lambda4 + 4 * hg * lambda3 * r + 3 * hg * lambda2**2
        + 6 * tg * lambda2 * r**2 + ug * r**4
    )
    co = (
        cz + rho * cw + 4 * n * cu / alpha
        + 3 * n * cf**2 / alpha**2 + 6 * m * cf / alpha
    )
    lambda4_output = co / alpha**4
    lambda4_margin = lambda4 - lambda4_output

    cw_lip = (
        lfy * xi4 + 5 * hf * lambda4 * r + 10 * hf * lambda3 * lambda2
        + 10 * tf * lambda3 * r**2 + 15 * tf * lambda2**2 * r
        + 10 * uf * lambda2 * r**3 + vf * r**5
    )
    cz_lip = (
        q * xi4 + 5 * hg * lambda4 * r + 10 * hg * lambda3 * lambda2
        + 10 * tg * lambda3 * r**2 + 15 * tg * lambda2**2 * r
        + 10 * ug * lambda2 * r**3 + vg * r**5
    )
    l_lip = cf / alpha**2
    lp = cf / alpha
    lp_lip = cf * l_lip + cp / alpha
    co_lip = (
        cz_lip + ct * cw + rho * cw_lip
        + 4 * (cn * cu / alpha + n * l_lip * cu + n * cu_lip / alpha)
        + 3 * (cn * lp**2 + 2 * n * lp * lp_lip)
        + 6 * (cm * lp + m * lp_lip)
    )
    xi4_output = co_lip / alpha**5 + 4 * co * cf / alpha**6
    xi4_margin = xi4 - xi4_output

    wf = 4 * hf * r
    we = 6 * hf * lambda2 + 6 * tf * r**2
    wd = 4 * hf * lambda3 + 12 * tf * lambda2 * r + 4 * uf * r**3
    wdelta = (
        lfy * xi4 * rx + hf * state * lambda4
        + 4 * (tf * state * lambda3 * r + hf * xi3 * rx * r + hf * lambda3 * lambda2 * rx)
        + 3 * (tf * state * lambda2**2 + 2 * hf * lambda2 * lambda3 * rx)
        + 6 * (uf * state * lambda2 * r**2 + tf * lambda3 * rx * r**2 + 2 * tf * lambda2**2 * r * rx)
        + vf * state * r**4 + 4 * uf * lambda2 * rx * r**3
    )
    zf = 4 * hg * r
    ze = 6 * hg * lambda2 + 6 * tg * r**2
    zd = 4 * hg * lambda3 + 12 * tg * lambda2 * r + 4 * ug * r**3
    zdelta = (
        q * xi4 * rx + hg * state * lambda4
        + 4 * (tg * state * lambda3 * r + hg * xi3 * rx * r + hg * lambda3 * lambda2 * rx)
        + 3 * (tg * state * lambda2**2 + 2 * hg * lambda2 * lambda3 * rx)
        + 6 * (ug * state * lambda2 * r**2 + tg * lambda3 * rx * r**2 + 2 * tg * lambda2**2 * r * rx)
        + vg * state * r**4 + 4 * ug * lambda2 * rx * r**3
    )

    zero = Fraction(0)
    dw: Vector5 = (lfy, wf, we, wd, wdelta)
    dz: Vector5 = (q, zf, ze, zd, zdelta)
    dt: Vector5 = (zero, zero, zero, beta1, c10)
    dn: Vector5 = (zero, zero, q_value, nd, ndelta)
    dm: Vector5 = (zero, q_value, me, md, mdelta)
    dl: Vector5 = (zero, zero, zero, lfy / alpha**2, adelta / alpha**2)
    dp: Vector5 = (zero, zero, lfy, pd, pdelta)
    # Recompute the predecessor U-difference value coefficient explicitly;
    # it is not recoverable from the already combined M-difference coefficient.
    udelta = (
        lfy * xi3 * rx + hf * state * lambda3
        + 3 * hf * r * lambda3 * rx + 3 * hf * lambda2**2 * rx
        + 3 * tf * state * lambda2 * r + 3 * tf * r**2 * lambda2 * rx
        + uf * state * r**3
    )
    du = (zero, lfy, 3 * hf * r, 3 * hf * lambda2 + 3 * tf * r**2, udelta)
    dlp = _vadd(_vscale(cf, dl), _vscale(Fraction(1, 1) / alpha, dp))
    do = _vadd(
        dz,
        _vscale(rho, dw),
        _vscale(cw, dt),
        _vscale(4 * cu / alpha, dn),
        _vscale(4 * n * cu, dl),
        _vscale(4 * n / alpha, du),
        _vscale(3 * lp**2, dn),
        _vscale(6 * n * lp, dlp),
        _vscale(6 * lp, dm),
        _vscale(6 * m, dlp),
    )
    oj, of, oe, od, odelta = do
    beta4 = oj / alpha**4
    beta4_margin = 1 - beta4
    c43 = of / alpha**4
    c42 = oe / alpha**4
    c41 = od / alpha**4 + 4 * co * lfy / alpha**5
    c40 = odelta / alpha**4 + 4 * co * adelta / alpha**5

    failures = list(c3.failure_codes)
    if lambda4_margin < 0:
        failures.append("COUPLED_C4_GRAPH_FOURTH_DERIVATIVE_CLASS_NOT_INVARIANT")
    if xi4_required and xi4_margin < 0:
        failures.append("COUPLED_C41_GRAPH_CLASS_NOT_INVARIANT")
    if beta4_margin <= 0:
        failures.append("COUPLED_C4_FOURTH_DERIVATIVE_BUNCHING_NOT_STRICT")
    detailed = dict(
        failure_codes=tuple(failures),
        base_fourth_derivative_upper=cw,
        fiber_fourth_derivative_upper=cz,
        modified_fourth_derivative_upper=co,
        output_graph_fourth_derivative_upper=lambda4_output,
        graph_fourth_derivative_margin=lambda4_margin,
        base_fourth_derivative_lipschitz_upper=cw_lip,
        fiber_fourth_derivative_lipschitz_upper=cz_lip,
        modified_fourth_derivative_lipschitz_upper=co_lip,
        output_graph_fourth_derivative_lipschitz_upper=xi4_output,
        graph_fourth_derivative_lipschitz_margin=xi4_margin,
        modified_fourth_difference_third_coefficient_upper=of,
        modified_fourth_difference_hessian_coefficient_upper=oe,
        modified_fourth_difference_derivative_coefficient_upper=od,
        modified_fourth_difference_value_coefficient_upper=odelta,
        fourth_derivative_bunching_factor_upper=beta4,
        fourth_derivative_bunching_margin=beta4_margin,
        third_to_fourth_derivative_cross_coefficient_upper=c43,
        hessian_to_fourth_derivative_cross_coefficient_upper=c42,
        derivative_to_fourth_derivative_cross_coefficient_upper=c41,
        value_to_fourth_derivative_cross_coefficient_upper=c40,
    )
    if failures:
        return QuantitativeCoupledC4GraphTransformCertificate(
            status=failures[0], validation_level=None, robust_interior=False,
            c4_graph_real_dimension=None, **detailed, **common,
        )
    robust = (
        c3.robust_interior and lambda4_margin > 0
        and (not xi4_required or xi4_margin > 0)
    )
    return QuantitativeCoupledC4GraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_COUPLED_C4_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_COUPLED_C4_GRAPH_TRANSFORM",
        robust_interior=robust,
        c4_graph_real_dimension=base_dimension,
        **detailed,
        **common,
    )


def coupled_c4_graph_iteration_bound(
    certificate: QuantitativeCoupledC4GraphTransformCertificate,
    *, initial_value_distance: object, initial_derivative_distance: object,
    initial_hessian_distance: object, initial_third_derivative_distance: object,
    initial_fourth_derivative_distance: object, steps: int,
) -> CoupledC4GraphIterationBound:
    """Iterate the exact coupled C0/C1/C2/C3/C4 recurrence."""
    if certificate.validation_level is None:
        raise ValueError("iteration requires a verified coupled C4 certificate")
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
    c3 = certificate.coupled_c3_certificate
    c2 = c3.coupled_c2_certificate
    c1 = c2.coupled_c1_certificate
    q0 = c1.coupled_lipschitz_certificate.transform_contraction_factor_upper
    assert None not in (
        q0, c1.derivative_bunching_factor_upper, c1.derivative_cross_coefficient_upper,
        c2.second_derivative_bunching_factor_upper,
        c2.derivative_to_hessian_cross_coefficient_upper,
        c2.value_to_hessian_cross_coefficient_upper,
        c3.third_derivative_bunching_factor_upper,
        c3.hessian_to_third_derivative_cross_coefficient_upper,
        c3.derivative_to_third_derivative_cross_coefficient_upper,
        c3.value_to_third_derivative_cross_coefficient_upper,
        certificate.fourth_derivative_bunching_factor_upper,
        certificate.third_to_fourth_derivative_cross_coefficient_upper,
        certificate.hessian_to_fourth_derivative_cross_coefficient_upper,
        certificate.derivative_to_fourth_derivative_cross_coefficient_upper,
        certificate.value_to_fourth_derivative_cross_coefficient_upper,
    )
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
            q0 * delta, next_derivative, next_hessian, next_third, next_fourth
        )
    return CoupledC4GraphIterationBound(steps, delta, derivative, hessian, third, fourth)


__all__ = [
    "CoupledC4GraphIterationBound",
    "QuantitativeCoupledC4GraphTransformCertificate",
    "coupled_c4_graph_iteration_bound",
    "quantitative_coupled_c4_graph_transform",
]
