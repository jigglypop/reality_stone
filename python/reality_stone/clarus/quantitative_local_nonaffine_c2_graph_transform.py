"""Exact local-domain wrapper for the graph-independent nonaffine C2 transform."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_graph_transform import _exact_fraction
    from .quantitative_nonaffine_c2_graph_transform import (
        NonaffineC2GraphIterationBound,
        QuantitativeNonaffineC2GraphTransformCertificate,
        nonaffine_c2_graph_iteration_bound,
        quantitative_nonaffine_c2_triangular_graph_transform,
    )
else:
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]
    from quantitative_nonaffine_c2_graph_transform import (  # type: ignore[no-redef]
        NonaffineC2GraphIterationBound,
        QuantitativeNonaffineC2GraphTransformCertificate,
        nonaffine_c2_graph_iteration_bound,
        quantitative_nonaffine_c2_triangular_graph_transform,
    )


LOCAL_INVARIANCE_KIND = "BACKWARD_COVERED_OVERFLOW_INVARIANT"


@dataclass(frozen=True)
class ExpandingSineLocalBaseC2Bounds:
    inverse_derivative_upper: Fraction
    inverse_hessian_upper: Fraction
    inverse_image_radius_upper: Fraction


@dataclass(frozen=True)
class QuantitativeLocalNonaffineC2GraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    nonaffine_c2_certificate: QuantitativeNonaffineC2GraphTransformCertificate
    raw_base_domain_radius: Fraction
    raw_inverse_base_image_radius_upper: Fraction
    normalized_base_domain_radius: Fraction
    normalized_inverse_base_image_radius_upper: Fraction
    inverse_domain_coverage_margin: Fraction
    local_invariance_kind: str
    forward_retention_certified: bool
    local_c2_graph_real_dimension: int | None


def expanding_sine_local_base_c2_bounds(
    *,
    linear_coefficient: object,
    amplitude_upper: object,
    base_domain_radius: object,
) -> ExpandingSineLocalBaseC2Bounds:
    """Exact inverse and centered-ball coverage bounds for lambda*x+a*sin(x)."""
    linear = _exact_fraction(linear_coefficient, "linear_coefficient")
    amplitude = _exact_fraction(amplitude_upper, "amplitude_upper")
    radius = _exact_fraction(base_domain_radius, "base_domain_radius")
    if linear <= 0 or amplitude < 0 or linear <= amplitude:
        raise ValueError("linear_coefficient must be positive and strictly exceed amplitude_upper")
    if radius <= 0:
        raise ValueError("base_domain_radius must be positive")
    derivative_gap = linear - amplitude
    mu = Fraction(1) / derivative_gap
    return ExpandingSineLocalBaseC2Bounds(
        inverse_derivative_upper=mu,
        inverse_hessian_upper=amplitude / (derivative_gap ** 3),
        inverse_image_radius_upper=mu * radius,
    )


def quantitative_local_nonaffine_c2_triangular_graph_transform(
    *,
    base_dimension: int,
    base_reference_scale: object,
    fiber_reference_scale: object,
    base_domain_radius: object,
    inverse_base_image_radius_upper: object,
    fiber_radius: object,
    forcing_at_zero_upper: object,
    base_inverse_lipschitz: object,
    base_inverse_hessian_upper: object,
    fiber_linear_norm_upper: object,
    base_to_fiber_lipschitz: object,
    fiber_nonlinear_lipschitz: object,
    graph_slope_upper: object,
    base_derivative_fiber_variation: object,
    fiber_derivative_fiber_variation: object,
    normalized_map_hessian_upper: object,
    normalized_map_hessian_fiber_lipschitz: object,
    normalized_graph_hessian_upper: object,
) -> QuantitativeLocalNonaffineC2GraphTransformCertificate:
    """Add exact inverse-domain coverage to the nonaffine triangular C2 gates."""
    predecessor = quantitative_nonaffine_c2_triangular_graph_transform(
        base_dimension=base_dimension,
        base_reference_scale=base_reference_scale,
        fiber_reference_scale=fiber_reference_scale,
        fiber_radius=fiber_radius,
        forcing_at_zero_upper=forcing_at_zero_upper,
        base_inverse_lipschitz=base_inverse_lipschitz,
        base_inverse_hessian_upper=base_inverse_hessian_upper,
        fiber_linear_norm_upper=fiber_linear_norm_upper,
        base_to_fiber_lipschitz=base_to_fiber_lipschitz,
        fiber_nonlinear_lipschitz=fiber_nonlinear_lipschitz,
        graph_slope_upper=graph_slope_upper,
        base_derivative_fiber_variation=base_derivative_fiber_variation,
        fiber_derivative_fiber_variation=fiber_derivative_fiber_variation,
        normalized_map_hessian_upper=normalized_map_hessian_upper,
        normalized_map_hessian_fiber_lipschitz=normalized_map_hessian_fiber_lipschitz,
        normalized_graph_hessian_upper=normalized_graph_hessian_upper,
    )
    radius_raw = _exact_fraction(base_domain_radius, "base_domain_radius")
    inverse_radius_raw = _exact_fraction(
        inverse_base_image_radius_upper,
        "inverse_base_image_radius_upper",
    )
    if radius_raw <= 0:
        raise ValueError("base_domain_radius must be positive")
    if inverse_radius_raw < 0:
        raise ValueError("inverse_base_image_radius_upper must be nonnegative")

    x_scale = predecessor.c1_certificate.lipschitz_certificate.base_reference_scale
    radius = radius_raw / x_scale
    inverse_radius = inverse_radius_raw / x_scale
    coverage_margin = radius - inverse_radius
    failures = list(predecessor.failure_codes)
    if coverage_margin < 0:
        failures.append("LOCAL_C2_INVERSE_BASE_DOMAIN_NOT_COVERED")
    common = dict(
        failure_codes=tuple(failures),
        nonaffine_c2_certificate=predecessor,
        raw_base_domain_radius=radius_raw,
        raw_inverse_base_image_radius_upper=inverse_radius_raw,
        normalized_base_domain_radius=radius,
        normalized_inverse_base_image_radius_upper=inverse_radius,
        inverse_domain_coverage_margin=coverage_margin,
        local_invariance_kind=LOCAL_INVARIANCE_KIND,
        forward_retention_certified=False,
    )
    if failures:
        return QuantitativeLocalNonaffineC2GraphTransformCertificate(
            status=failures[0],
            validation_level=None,
            robust_interior=False,
            local_c2_graph_real_dimension=None,
            **common,
        )
    return QuantitativeLocalNonaffineC2GraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_LOCAL_NONAFFINE_C2_TRIANGULAR_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_LOCAL_NONAFFINE_C2_TRIANGULAR_GRAPH_TRANSFORM",
        robust_interior=predecessor.robust_interior and coverage_margin > 0,
        local_c2_graph_real_dimension=base_dimension,
        **common,
    )


def local_nonaffine_c2_graph_iteration_bound(
    certificate: QuantitativeLocalNonaffineC2GraphTransformCertificate,
    *,
    initial_value_distance: object,
    initial_derivative_distance: object,
    initial_hessian_distance: object,
    steps: int,
) -> NonaffineC2GraphIterationBound:
    """Iterate the predecessor recurrence after the local coverage gate passes."""
    if certificate.validation_level is None:
        raise ValueError("local C2 iteration requires a verified local certificate")
    return nonaffine_c2_graph_iteration_bound(
        certificate.nonaffine_c2_certificate,
        initial_value_distance=initial_value_distance,
        initial_derivative_distance=initial_derivative_distance,
        initial_hessian_distance=initial_hessian_distance,
        steps=steps,
    )


__all__ = [
    "ExpandingSineLocalBaseC2Bounds",
    "LOCAL_INVARIANCE_KIND",
    "QuantitativeLocalNonaffineC2GraphTransformCertificate",
    "expanding_sine_local_base_c2_bounds",
    "local_nonaffine_c2_graph_iteration_bound",
    "quantitative_local_nonaffine_c2_triangular_graph_transform",
]

