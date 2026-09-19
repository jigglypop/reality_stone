"""Exact matched-domain wrapper for local nonaffine coupled C2 graphs."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_coupled_graph_transform import _exact_fraction
    from .quantitative_local_nonaffine_coupled_c2_graph_transform import (
        QuantitativeLocalNonaffineCoupledC2GraphTransformCertificate,
        local_nonaffine_coupled_c2_graph_iteration_bound,
        quantitative_local_nonaffine_coupled_c2_graph_transform,
    )
    from .quantitative_nonaffine_coupled_c2_graph_transform import (
        NonaffineCoupledC2GraphIterationBound,
    )
else:
    from quantitative_coupled_graph_transform import _exact_fraction  # type: ignore[no-redef]
    from quantitative_local_nonaffine_coupled_c2_graph_transform import (  # type: ignore[no-redef]
        QuantitativeLocalNonaffineCoupledC2GraphTransformCertificate,
        local_nonaffine_coupled_c2_graph_iteration_bound,
        quantitative_local_nonaffine_coupled_c2_graph_transform,
    )
    from quantitative_nonaffine_coupled_c2_graph_transform import (  # type: ignore[no-redef]
        NonaffineCoupledC2GraphIterationBound,
    )


MATCHED_LOCAL_NONAFFINE_COUPLED_INVARIANCE_KIND = (
    "GRAPH_DEPENDENT_EXACT_MATCHED_DOMAIN_FULL_FORWARD_BACKWARD_INVARIANT"
)


@dataclass(frozen=True)
class BoundaryAnchoredCubicCoupledUnitIntervalC2Bounds:
    base_inverse_lipschitz_upper: Fraction
    coupled_inverse_lipschitz_upper: Fraction
    base_map_hessian_upper: Fraction
    base_map_hessian_lipschitz_upper: Fraction
    forward_image_radius_exact: Fraction
    inverse_image_radius_exact: Fraction
    graph_boundary_value_exact: Fraction


@dataclass(frozen=True)
class QuantitativeMatchedLocalNonaffineCoupledC2GraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    differential_robust_interior: bool
    domain_contact_robust_interior: bool
    local_nonaffine_coupled_c2_certificate: QuantitativeLocalNonaffineCoupledC2GraphTransformCertificate
    raw_uniform_forward_base_image_radius_exact: Fraction
    normalized_uniform_forward_base_image_radius_exact: Fraction
    raw_graph_boundary_value_upper: Fraction
    normalized_graph_boundary_value_upper: Fraction
    raw_fiber_boundary_forcing_upper: Fraction
    normalized_fiber_boundary_forcing_upper: Fraction
    inverse_boundary_contact_residual: Fraction
    forward_boundary_contact_residual: Fraction
    matched_domain_identity: str
    full_forward_retention_certified: bool
    boundary_anchored_graph_class_certified: bool
    matched_local_c2_graph_real_dimension: int | None


def boundary_anchored_cubic_coupled_unit_interval_c2_bounds(
    *,
    amplitude_upper: object,
    fiber_coupling_upper: object,
    graph_slope_upper: object,
) -> BoundaryAnchoredCubicCoupledUnitIntervalC2Bounds:
    """Bounds for x+a*x*(1-x**2)+epsilon*h with h(-1)=h(1)=0."""
    amplitude = _exact_fraction(amplitude_upper, "amplitude_upper")
    coupling = _exact_fraction(fiber_coupling_upper, "fiber_coupling_upper")
    slope = _exact_fraction(graph_slope_upper, "graph_slope_upper")
    if amplitude < 0 or amplitude >= Fraction(1, 2):
        raise ValueError("amplitude_upper must lie in the exact interval [0, 1/2)")
    if coupling < 0 or slope < 0:
        raise ValueError("fiber coupling and graph slope must be nonnegative")
    base_gap = 1 - 2 * amplitude
    coupled_gap = base_gap - coupling * slope
    if coupled_gap <= 0:
        raise ValueError("fiber coupling times graph slope must be strictly below the base gap")
    curvature = 6 * amplitude
    return BoundaryAnchoredCubicCoupledUnitIntervalC2Bounds(
        base_inverse_lipschitz_upper=Fraction(1) / base_gap,
        coupled_inverse_lipschitz_upper=Fraction(1) / coupled_gap,
        base_map_hessian_upper=curvature,
        base_map_hessian_lipschitz_upper=curvature,
        forward_image_radius_exact=Fraction(1),
        inverse_image_radius_exact=Fraction(1),
        graph_boundary_value_exact=Fraction(0),
    )


def quantitative_matched_local_nonaffine_coupled_c2_graph_transform(
    *,
    base_dimension: int,
    base_reference_scale: object,
    fiber_reference_scale: object,
    input_base_domain_radius: object,
    output_base_domain_radius: object,
    uniform_inverse_base_image_radius_exact: object,
    uniform_forward_base_image_radius_exact: object,
    graph_boundary_value_upper: object,
    fiber_boundary_forcing_upper: object,
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
) -> QuantitativeMatchedLocalNonaffineCoupledC2GraphTransformCertificate:
    """Require exact contacts and boundary preservation for full invariance."""
    local = quantitative_local_nonaffine_coupled_c2_graph_transform(
        base_dimension=base_dimension,
        base_reference_scale=base_reference_scale,
        fiber_reference_scale=fiber_reference_scale,
        input_base_domain_radius=input_base_domain_radius,
        output_base_domain_radius=output_base_domain_radius,
        uniform_inverse_base_image_radius_upper=uniform_inverse_base_image_radius_exact,
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
    forward_raw = _exact_fraction(
        uniform_forward_base_image_radius_exact,
        "uniform_forward_base_image_radius_exact",
    )
    graph_boundary_raw = _exact_fraction(
        graph_boundary_value_upper,
        "graph_boundary_value_upper",
    )
    fiber_boundary_raw = _exact_fraction(
        fiber_boundary_forcing_upper,
        "fiber_boundary_forcing_upper",
    )
    if forward_raw < 0 or graph_boundary_raw < 0 or fiber_boundary_raw < 0:
        raise ValueError("forward radius and boundary residuals must be nonnegative")

    base = local.nonaffine_coupled_c2_certificate.nonaffine_coupled_c1_certificate.coupled_lipschitz_certificate
    x_scale = base.base_reference_scale
    y_scale = base.fiber_reference_scale
    forward_radius = forward_raw / x_scale
    graph_boundary = graph_boundary_raw / y_scale
    fiber_boundary = fiber_boundary_raw / y_scale
    inverse_contact = local.inverse_domain_coverage_margin
    forward_contact = local.normalized_output_base_domain_radius - forward_radius
    failures = list(local.failure_codes)
    if inverse_contact > 0:
        failures.append("MATCHED_LOCAL_NONAFFINE_COUPLED_C2_INVERSE_BOUNDARY_CONTACT_NOT_EXACT")
    if forward_contact < 0:
        failures.append("MATCHED_LOCAL_NONAFFINE_COUPLED_C2_FORWARD_BASE_DOMAIN_NOT_COVERED")
    elif forward_contact > 0:
        failures.append("MATCHED_LOCAL_NONAFFINE_COUPLED_C2_FORWARD_BOUNDARY_CONTACT_NOT_EXACT")
    if graph_boundary > 0:
        failures.append("MATCHED_LOCAL_NONAFFINE_COUPLED_C2_GRAPH_BOUNDARY_NOT_ANCHORED")
    if fiber_boundary > 0:
        failures.append("MATCHED_LOCAL_NONAFFINE_COUPLED_C2_FIBER_BOUNDARY_NOT_PRESERVED")

    common = dict(
        failure_codes=tuple(failures),
        local_nonaffine_coupled_c2_certificate=local,
        raw_uniform_forward_base_image_radius_exact=forward_raw,
        normalized_uniform_forward_base_image_radius_exact=forward_radius,
        raw_graph_boundary_value_upper=graph_boundary_raw,
        normalized_graph_boundary_value_upper=graph_boundary,
        raw_fiber_boundary_forcing_upper=fiber_boundary_raw,
        normalized_fiber_boundary_forcing_upper=fiber_boundary,
        inverse_boundary_contact_residual=inverse_contact,
        forward_boundary_contact_residual=forward_contact,
        matched_domain_identity=MATCHED_LOCAL_NONAFFINE_COUPLED_INVARIANCE_KIND,
        full_forward_retention_certified=False,
        boundary_anchored_graph_class_certified=False,
    )
    if failures:
        return QuantitativeMatchedLocalNonaffineCoupledC2GraphTransformCertificate(
            status=failures[0],
            validation_level=None,
            robust_interior=False,
            differential_robust_interior=local.nonaffine_coupled_c2_certificate.robust_interior,
            domain_contact_robust_interior=False,
            matched_local_c2_graph_real_dimension=None,
            **common,
        )
    common["full_forward_retention_certified"] = True
    common["boundary_anchored_graph_class_certified"] = True
    return QuantitativeMatchedLocalNonaffineCoupledC2GraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_MATCHED_LOCAL_NONAFFINE_COUPLED_C2_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_MATCHED_LOCAL_NONAFFINE_COUPLED_C2_GRAPH_TRANSFORM",
        robust_interior=False,
        differential_robust_interior=local.nonaffine_coupled_c2_certificate.robust_interior,
        domain_contact_robust_interior=False,
        matched_local_c2_graph_real_dimension=base_dimension,
        **common,
    )


def matched_local_nonaffine_coupled_c2_graph_iteration_bound(
    certificate: QuantitativeMatchedLocalNonaffineCoupledC2GraphTransformCertificate,
    *,
    initial_value_distance: object,
    initial_derivative_distance: object,
    initial_hessian_distance: object,
    steps: int,
) -> NonaffineCoupledC2GraphIterationBound:
    """Iterate the unchanged recurrence after every matched gate passes."""
    if certificate.validation_level is None:
        raise ValueError("matched local nonaffine coupled C2 iteration requires a verified certificate")
    return local_nonaffine_coupled_c2_graph_iteration_bound(
        certificate.local_nonaffine_coupled_c2_certificate,
        initial_value_distance=initial_value_distance,
        initial_derivative_distance=initial_derivative_distance,
        initial_hessian_distance=initial_hessian_distance,
        steps=steps,
    )


__all__ = [
    "BoundaryAnchoredCubicCoupledUnitIntervalC2Bounds",
    "MATCHED_LOCAL_NONAFFINE_COUPLED_INVARIANCE_KIND",
    "QuantitativeMatchedLocalNonaffineCoupledC2GraphTransformCertificate",
    "boundary_anchored_cubic_coupled_unit_interval_c2_bounds",
    "matched_local_nonaffine_coupled_c2_graph_iteration_bound",
    "quantitative_matched_local_nonaffine_coupled_c2_graph_transform",
]
