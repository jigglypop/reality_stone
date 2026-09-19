"""Exact local-domain wrapper for the nonaffine coupled C2 transform."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_coupled_graph_transform import _exact_fraction
    from .quantitative_nonaffine_coupled_c2_graph_transform import (
        NonaffineCoupledC2GraphIterationBound,
        QuantitativeNonaffineCoupledC2GraphTransformCertificate,
        nonaffine_coupled_c2_graph_iteration_bound,
        quantitative_nonaffine_coupled_c2_graph_transform,
    )
else:
    from quantitative_coupled_graph_transform import _exact_fraction  # type: ignore[no-redef]
    from quantitative_nonaffine_coupled_c2_graph_transform import (  # type: ignore[no-redef]
        NonaffineCoupledC2GraphIterationBound,
        QuantitativeNonaffineCoupledC2GraphTransformCertificate,
        nonaffine_coupled_c2_graph_iteration_bound,
        quantitative_nonaffine_coupled_c2_graph_transform,
    )


LOCAL_NONAFFINE_COUPLED_INVARIANCE_KIND = (
    "GRAPH_DEPENDENT_BACKWARD_COVERED_OVERFLOW_INVARIANT"
)


@dataclass(frozen=True)
class ExpandingSineCoupledLocalBaseC2Bounds:
    base_inverse_lipschitz_upper: Fraction
    coupled_inverse_lipschitz_upper: Fraction
    base_map_hessian_upper: Fraction
    base_map_hessian_lipschitz_upper: Fraction
    uniform_inverse_image_radius_upper: Fraction


@dataclass(frozen=True)
class QuantitativeLocalNonaffineCoupledC2GraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    nonaffine_coupled_c2_certificate: QuantitativeNonaffineCoupledC2GraphTransformCertificate
    raw_input_base_domain_radius: Fraction
    raw_output_base_domain_radius: Fraction
    raw_uniform_inverse_base_image_radius_upper: Fraction
    normalized_input_base_domain_radius: Fraction
    normalized_output_base_domain_radius: Fraction
    normalized_uniform_inverse_base_image_radius_upper: Fraction
    inverse_domain_coverage_margin: Fraction
    local_invariance_kind: str
    forward_retention_certified: bool
    local_c2_graph_real_dimension: int | None


def expanding_sine_coupled_local_base_c2_bounds(
    *,
    linear_coefficient: object,
    amplitude_upper: object,
    fiber_coupling_upper: object,
    graph_slope_upper: object,
    fiber_radius: object,
    output_base_domain_radius: object,
) -> ExpandingSineCoupledLocalBaseC2Bounds:
    """Uniform scalar bound for lambda*x+a*sin(x)+epsilon*h(x)."""
    linear = _exact_fraction(linear_coefficient, "linear_coefficient")
    amplitude = _exact_fraction(amplitude_upper, "amplitude_upper")
    coupling = _exact_fraction(fiber_coupling_upper, "fiber_coupling_upper")
    slope = _exact_fraction(graph_slope_upper, "graph_slope_upper")
    fiber = _exact_fraction(fiber_radius, "fiber_radius")
    output = _exact_fraction(output_base_domain_radius, "output_base_domain_radius")
    if linear <= 0 or amplitude < 0 or linear <= amplitude:
        raise ValueError("linear_coefficient must be positive and strictly exceed amplitude_upper")
    if coupling < 0 or slope < 0 or fiber <= 0 or output <= 0:
        raise ValueError("coupling and slope must be nonnegative and both radii must be positive")
    base_gap = linear - amplitude
    coupled_gap = base_gap - coupling * slope
    if coupled_gap <= 0:
        raise ValueError("fiber coupling times graph slope must be strictly below the base gap")
    base_inverse = Fraction(1) / base_gap
    return ExpandingSineCoupledLocalBaseC2Bounds(
        base_inverse_lipschitz_upper=base_inverse,
        coupled_inverse_lipschitz_upper=Fraction(1) / coupled_gap,
        base_map_hessian_upper=amplitude,
        base_map_hessian_lipschitz_upper=amplitude,
        uniform_inverse_image_radius_upper=base_inverse * (output + coupling * fiber),
    )


def quantitative_local_nonaffine_coupled_c2_graph_transform(
    *,
    base_dimension: int,
    base_reference_scale: object,
    fiber_reference_scale: object,
    input_base_domain_radius: object,
    output_base_domain_radius: object,
    uniform_inverse_base_image_radius_upper: object,
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
) -> QuantitativeLocalNonaffineCoupledC2GraphTransformCertificate:
    """Add uniform graph-dependent inverse coverage to nonaffine coupled C2."""
    predecessor = quantitative_nonaffine_coupled_c2_graph_transform(
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
        graph_hessian_lipschitz_upper=graph_hessian_lipschitz_upper,
    )
    input_raw = _exact_fraction(input_base_domain_radius, "input_base_domain_radius")
    output_raw = _exact_fraction(output_base_domain_radius, "output_base_domain_radius")
    inverse_raw = _exact_fraction(
        uniform_inverse_base_image_radius_upper,
        "uniform_inverse_base_image_radius_upper",
    )
    if input_raw <= 0 or output_raw <= 0:
        raise ValueError("input and output base-domain radii must be positive")
    if inverse_raw < 0:
        raise ValueError("uniform inverse-base image radius must be nonnegative")

    base = predecessor.nonaffine_coupled_c1_certificate.coupled_lipschitz_certificate
    x_scale = base.base_reference_scale
    input_radius = input_raw / x_scale
    output_radius = output_raw / x_scale
    inverse_radius = inverse_raw / x_scale
    coverage_margin = input_radius - inverse_radius
    failures = list(predecessor.failure_codes)
    if coverage_margin < 0:
        failures.append("LOCAL_NONAFFINE_COUPLED_C2_INVERSE_BASE_DOMAIN_NOT_COVERED")
    common = dict(
        failure_codes=tuple(failures),
        nonaffine_coupled_c2_certificate=predecessor,
        raw_input_base_domain_radius=input_raw,
        raw_output_base_domain_radius=output_raw,
        raw_uniform_inverse_base_image_radius_upper=inverse_raw,
        normalized_input_base_domain_radius=input_radius,
        normalized_output_base_domain_radius=output_radius,
        normalized_uniform_inverse_base_image_radius_upper=inverse_radius,
        inverse_domain_coverage_margin=coverage_margin,
        local_invariance_kind=LOCAL_NONAFFINE_COUPLED_INVARIANCE_KIND,
        forward_retention_certified=False,
    )
    if failures:
        return QuantitativeLocalNonaffineCoupledC2GraphTransformCertificate(
            status=failures[0],
            validation_level=None,
            robust_interior=False,
            local_c2_graph_real_dimension=None,
            **common,
        )
    return QuantitativeLocalNonaffineCoupledC2GraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_LOCAL_NONAFFINE_COUPLED_C2_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_LOCAL_NONAFFINE_COUPLED_C2_GRAPH_TRANSFORM",
        robust_interior=predecessor.robust_interior and coverage_margin > 0,
        local_c2_graph_real_dimension=base_dimension,
        **common,
    )


def local_nonaffine_coupled_c2_graph_iteration_bound(
    certificate: QuantitativeLocalNonaffineCoupledC2GraphTransformCertificate,
    *,
    initial_value_distance: object,
    initial_derivative_distance: object,
    initial_hessian_distance: object,
    steps: int,
) -> NonaffineCoupledC2GraphIterationBound:
    """Iterate the unchanged recurrence after local inverse coverage passes."""
    if certificate.validation_level is None:
        raise ValueError("local nonaffine coupled C2 iteration requires a verified certificate")
    return nonaffine_coupled_c2_graph_iteration_bound(
        certificate.nonaffine_coupled_c2_certificate,
        initial_value_distance=initial_value_distance,
        initial_derivative_distance=initial_derivative_distance,
        initial_hessian_distance=initial_hessian_distance,
        steps=steps,
    )


__all__ = [
    "ExpandingSineCoupledLocalBaseC2Bounds",
    "LOCAL_NONAFFINE_COUPLED_INVARIANCE_KIND",
    "QuantitativeLocalNonaffineCoupledC2GraphTransformCertificate",
    "expanding_sine_coupled_local_base_c2_bounds",
    "local_nonaffine_coupled_c2_graph_iteration_bound",
    "quantitative_local_nonaffine_coupled_c2_graph_transform",
]
