"""Exact C1,1-class certificate for an affine coupled-base graph transform."""

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
class QuantitativeCoupledC1GraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    coupled_lipschitz_certificate: QuantitativeCoupledGraphTransformCertificate
    normalized_base_jacobian_lipschitz: Fraction
    normalized_fiber_jacobian_lipschitz: Fraction
    graph_derivative_lipschitz_upper: Fraction
    base_jacobian_lipschitz_upper: Fraction | None
    fiber_jacobian_lipschitz_upper: Fraction | None
    output_graph_derivative_lipschitz_upper: Fraction | None
    graph_derivative_lipschitz_margin: Fraction | None
    preimage_value_coupling_upper: Fraction | None
    state_value_coupling_upper: Fraction | None
    derivative_bunching_factor_upper: Fraction | None
    derivative_bunching_margin: Fraction | None
    derivative_cross_coefficient_upper: Fraction | None
    c1_graph_real_dimension: int | None


@dataclass(frozen=True)
class CoupledC1GraphIterationBound:
    steps: int
    value_distance_upper: Fraction
    derivative_distance_upper: Fraction


def quantitative_coupled_c1_graph_transform(
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
) -> QuantitativeCoupledC1GraphTransformCertificate:
    """Check coupled Lipschitz, C1,1 invariance, and derivative bunching exactly."""
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
    hf = _exact_fraction(
        normalized_base_jacobian_lipschitz,
        "normalized_base_jacobian_lipschitz",
    )
    hg = _exact_fraction(
        normalized_fiber_jacobian_lipschitz,
        "normalized_fiber_jacobian_lipschitz",
    )
    derivative_lipschitz = _exact_fraction(
        graph_derivative_lipschitz_upper,
        "graph_derivative_lipschitz_upper",
    )
    if hf < 0 or hg < 0 or derivative_lipschitz < 0:
        raise ValueError("normalized Jacobian and graph-derivative Lipschitz bounds must be nonnegative")
    failures = list(base.failure_codes)
    alpha = base.base_invertibility_lower
    common = dict(
        coupled_lipschitz_certificate=base,
        normalized_base_jacobian_lipschitz=hf,
        normalized_fiber_jacobian_lipschitz=hg,
        graph_derivative_lipschitz_upper=derivative_lipschitz,
    )
    if alpha <= 0 or base.transform_contraction_factor_upper is None:
        return QuantitativeCoupledC1GraphTransformCertificate(
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
            derivative_bunching_factor_upper=None,
            derivative_bunching_margin=None,
            derivative_cross_coefficient_upper=None,
            c1_graph_real_dimension=None,
            **common,
        )
    kappa = base.normalized_graph_slope_upper
    fy = base.normalized_fiber_to_base_lipschitz
    q = base.fiber_factor_upper
    gx = base.normalized_base_to_fiber_lipschitz
    s = q * kappa + gx
    one_plus_kappa = 1 + kappa
    c_f = hf * one_plus_kappa * one_plus_kappa + fy * derivative_lipschitz
    c_y = q * derivative_lipschitz + hg * one_plus_kappa * one_plus_kappa
    output_lipschitz = c_y / (alpha * alpha) + s * c_f / (alpha * alpha * alpha)
    class_margin = derivative_lipschitz - output_lipschitz
    if class_margin < 0:
        failures.append("COUPLED_C11_GRAPH_CLASS_NOT_INVARIANT")
    preimage_coupling = fy / alpha
    state_coupling = 1 + one_plus_kappa * preimage_coupling
    a_delta = (
        hf * state_coupling * one_plus_kappa
        + fy * derivative_lipschitz * preimage_coupling
    )
    y_delta = (
        q * derivative_lipschitz * preimage_coupling
        + hg * state_coupling * one_plus_kappa
    )
    transform_factor = base.transform_contraction_factor_upper
    derivative_factor = transform_factor / alpha
    derivative_margin = 1 - derivative_factor
    if derivative_margin <= 0:
        failures.append("COUPLED_C1_DERIVATIVE_BUNCHING_NOT_STRICT")
    derivative_cross = y_delta / alpha + s * a_delta / (alpha * alpha)
    detailed = dict(
        failure_codes=tuple(failures),
        base_jacobian_lipschitz_upper=c_f,
        fiber_jacobian_lipschitz_upper=c_y,
        output_graph_derivative_lipschitz_upper=output_lipschitz,
        graph_derivative_lipschitz_margin=class_margin,
        preimage_value_coupling_upper=preimage_coupling,
        state_value_coupling_upper=state_coupling,
        derivative_bunching_factor_upper=derivative_factor,
        derivative_bunching_margin=derivative_margin,
        derivative_cross_coefficient_upper=derivative_cross,
    )
    if failures:
        return QuantitativeCoupledC1GraphTransformCertificate(
            status=failures[0],
            validation_level=None,
            robust_interior=False,
            c1_graph_real_dimension=None,
            **detailed,
            **common,
        )
    robust = base.robust_interior and class_margin > 0
    return QuantitativeCoupledC1GraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_COUPLED_C1_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_COUPLED_C1_GRAPH_TRANSFORM",
        robust_interior=robust,
        c1_graph_real_dimension=base_dimension,
        **detailed,
        **common,
    )


def coupled_c1_graph_iteration_bound(
    certificate: QuantitativeCoupledC1GraphTransformCertificate,
    *,
    initial_value_distance: object,
    initial_derivative_distance: object,
    steps: int,
) -> CoupledC1GraphIterationBound:
    """Iterate the exact coupled C0/C1 upper-triangular recurrence."""
    if certificate.validation_level is None:
        raise ValueError("iteration requires a verified coupled C1 certificate")
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative built-in integer")
    value_distance = _exact_fraction(initial_value_distance, "initial_value_distance")
    derivative_distance = _exact_fraction(
        initial_derivative_distance,
        "initial_derivative_distance",
    )
    if value_distance < 0 or derivative_distance < 0:
        raise ValueError("initial distances must be nonnegative")
    q_value = certificate.coupled_lipschitz_certificate.transform_contraction_factor_upper
    beta = certificate.derivative_bunching_factor_upper
    cross = certificate.derivative_cross_coefficient_upper
    assert q_value is not None and beta is not None and cross is not None
    for _ in range(steps):
        derivative_distance = beta * derivative_distance + cross * value_distance
        value_distance = q_value * value_distance
    return CoupledC1GraphIterationBound(steps, value_distance, derivative_distance)


__all__ = [
    "CoupledC1GraphIterationBound",
    "QuantitativeCoupledC1GraphTransformCertificate",
    "coupled_c1_graph_iteration_bound",
    "quantitative_coupled_c1_graph_transform",
]

