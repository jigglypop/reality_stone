"""Exact matched-ball full-forward wrapper for local nonaffine C2 graphs."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_graph_transform import _exact_fraction
    from .quantitative_local_nonaffine_c2_graph_transform import (
        QuantitativeLocalNonaffineC2GraphTransformCertificate,
        local_nonaffine_c2_graph_iteration_bound,
        quantitative_local_nonaffine_c2_triangular_graph_transform,
    )
    from .quantitative_nonaffine_c2_graph_transform import NonaffineC2GraphIterationBound
else:
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]
    from quantitative_local_nonaffine_c2_graph_transform import (  # type: ignore[no-redef]
        QuantitativeLocalNonaffineC2GraphTransformCertificate,
        local_nonaffine_c2_graph_iteration_bound,
        quantitative_local_nonaffine_c2_triangular_graph_transform,
    )
    from quantitative_nonaffine_c2_graph_transform import (  # type: ignore[no-redef]
        NonaffineC2GraphIterationBound,
    )


MATCHED_LOCAL_INVARIANCE_KIND = "EXACT_MATCHED_DOMAIN_FULL_FORWARD_BACKWARD_INVARIANT"


@dataclass(frozen=True)
class BoundaryFixedCubicUnitIntervalC2Bounds:
    inverse_derivative_upper: Fraction
    inverse_hessian_upper: Fraction
    forward_image_radius_exact: Fraction
    inverse_image_radius_exact: Fraction


@dataclass(frozen=True)
class QuantitativeMatchedLocalNonaffineC2GraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    differential_robust_interior: bool
    domain_contact_robust_interior: bool
    local_c2_certificate: QuantitativeLocalNonaffineC2GraphTransformCertificate
    raw_output_base_domain_radius: Fraction
    raw_forward_base_image_radius_exact: Fraction
    normalized_output_base_domain_radius: Fraction
    normalized_forward_base_image_radius_exact: Fraction
    inverse_boundary_contact_residual: Fraction
    forward_boundary_contact_residual: Fraction
    matched_domain_identity: str
    full_forward_retention_certified: bool
    matched_local_c2_graph_real_dimension: int | None


def boundary_fixed_cubic_unit_interval_c2_bounds(
    amplitude_upper: object,
) -> BoundaryFixedCubicUnitIntervalC2Bounds:
    """Exact bounds for phi_a(x)=x+a*x*(1-x**2) on the unit interval."""
    amplitude = _exact_fraction(amplitude_upper, "amplitude_upper")
    if amplitude < 0 or amplitude >= Fraction(1, 2):
        raise ValueError("amplitude_upper must lie in the exact interval [0, 1/2)")
    derivative_gap = 1 - 2 * amplitude
    return BoundaryFixedCubicUnitIntervalC2Bounds(
        inverse_derivative_upper=Fraction(1) / derivative_gap,
        inverse_hessian_upper=6 * amplitude / (derivative_gap ** 3),
        forward_image_radius_exact=Fraction(1),
        inverse_image_radius_exact=Fraction(1),
    )


def quantitative_matched_local_nonaffine_c2_triangular_graph_transform(
    *,
    base_dimension: int,
    base_reference_scale: object,
    fiber_reference_scale: object,
    base_domain_radius: object,
    output_base_domain_radius: object,
    inverse_base_image_radius_exact: object,
    forward_base_image_radius_exact: object,
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
) -> QuantitativeMatchedLocalNonaffineC2GraphTransformCertificate:
    """Require exact forward and inverse ball-boundary contact for full invariance."""
    local = quantitative_local_nonaffine_c2_triangular_graph_transform(
        base_dimension=base_dimension,
        base_reference_scale=base_reference_scale,
        fiber_reference_scale=fiber_reference_scale,
        base_domain_radius=base_domain_radius,
        inverse_base_image_radius_upper=inverse_base_image_radius_exact,
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
    output_radius_raw = _exact_fraction(
        output_base_domain_radius,
        "output_base_domain_radius",
    )
    forward_radius_raw = _exact_fraction(
        forward_base_image_radius_exact,
        "forward_base_image_radius_exact",
    )
    if output_radius_raw <= 0:
        raise ValueError("output_base_domain_radius must be positive")
    if forward_radius_raw < 0:
        raise ValueError("forward_base_image_radius_exact must be nonnegative")

    x_scale = local.nonaffine_c2_certificate.c1_certificate.lipschitz_certificate.base_reference_scale
    output_radius = output_radius_raw / x_scale
    forward_radius = forward_radius_raw / x_scale
    inverse_contact = local.inverse_domain_coverage_margin
    forward_contact = output_radius - forward_radius
    failures = list(local.failure_codes)
    if inverse_contact > 0:
        failures.append("MATCHED_LOCAL_C2_INVERSE_BOUNDARY_CONTACT_NOT_EXACT")
    if forward_contact < 0:
        failures.append("MATCHED_LOCAL_C2_FORWARD_BASE_DOMAIN_NOT_COVERED")
    elif forward_contact > 0:
        failures.append("MATCHED_LOCAL_C2_FORWARD_BOUNDARY_CONTACT_NOT_EXACT")

    common = dict(
        failure_codes=tuple(failures),
        local_c2_certificate=local,
        raw_output_base_domain_radius=output_radius_raw,
        raw_forward_base_image_radius_exact=forward_radius_raw,
        normalized_output_base_domain_radius=output_radius,
        normalized_forward_base_image_radius_exact=forward_radius,
        inverse_boundary_contact_residual=inverse_contact,
        forward_boundary_contact_residual=forward_contact,
        matched_domain_identity=MATCHED_LOCAL_INVARIANCE_KIND,
        full_forward_retention_certified=False,
    )
    if failures:
        return QuantitativeMatchedLocalNonaffineC2GraphTransformCertificate(
            status=failures[0],
            validation_level=None,
            robust_interior=False,
            differential_robust_interior=local.nonaffine_c2_certificate.robust_interior,
            domain_contact_robust_interior=False,
            matched_local_c2_graph_real_dimension=None,
            **common,
        )

    differential_robust = local.nonaffine_c2_certificate.robust_interior
    common["full_forward_retention_certified"] = True
    return QuantitativeMatchedLocalNonaffineC2GraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_MATCHED_LOCAL_NONAFFINE_C2_TRIANGULAR_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_MATCHED_LOCAL_NONAFFINE_C2_TRIANGULAR_GRAPH_TRANSFORM",
        robust_interior=False,
        differential_robust_interior=differential_robust,
        domain_contact_robust_interior=False,
        matched_local_c2_graph_real_dimension=base_dimension,
        **common,
    )


def matched_local_nonaffine_c2_graph_iteration_bound(
    certificate: QuantitativeMatchedLocalNonaffineC2GraphTransformCertificate,
    *,
    initial_value_distance: object,
    initial_derivative_distance: object,
    initial_hessian_distance: object,
    steps: int,
) -> NonaffineC2GraphIterationBound:
    """Iterate the unchanged C2 recurrence after exact domain matching passes."""
    if certificate.validation_level is None:
        raise ValueError("matched local C2 iteration requires a verified certificate")
    return local_nonaffine_c2_graph_iteration_bound(
        certificate.local_c2_certificate,
        initial_value_distance=initial_value_distance,
        initial_derivative_distance=initial_derivative_distance,
        initial_hessian_distance=initial_hessian_distance,
        steps=steps,
    )


__all__ = [
    "BoundaryFixedCubicUnitIntervalC2Bounds",
    "MATCHED_LOCAL_INVARIANCE_KIND",
    "QuantitativeMatchedLocalNonaffineC2GraphTransformCertificate",
    "boundary_fixed_cubic_unit_interval_c2_bounds",
    "matched_local_nonaffine_c2_graph_iteration_bound",
    "quantitative_matched_local_nonaffine_c2_triangular_graph_transform",
]
