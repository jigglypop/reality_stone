"""Exact matched-domain wrapper for local coupled nonaffine C4 graphs."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_coupled_graph_transform import _exact_fraction
    from .quantitative_local_nonaffine_coupled_c4_graph_transform import (
        QuantitativeLocalNonaffineCoupledC4GraphTransformCertificate,
        local_nonaffine_coupled_c4_graph_iteration_bound,
        quantitative_local_nonaffine_coupled_c4_graph_transform,
    )
    from .quantitative_matched_local_nonaffine_coupled_c3_graph_transform import (
        boundary_anchored_cubic_coupled_unit_interval_c3_bounds,
    )
    from .quantitative_nonaffine_coupled_c4_graph_transform import (
        NonaffineCoupledC4GraphIterationBound,
    )
else:
    from quantitative_coupled_graph_transform import _exact_fraction  # type: ignore[no-redef]
    from quantitative_local_nonaffine_coupled_c4_graph_transform import (  # type: ignore[no-redef]
        QuantitativeLocalNonaffineCoupledC4GraphTransformCertificate,
        local_nonaffine_coupled_c4_graph_iteration_bound,
        quantitative_local_nonaffine_coupled_c4_graph_transform,
    )
    from quantitative_matched_local_nonaffine_coupled_c3_graph_transform import (  # type: ignore[no-redef]
        boundary_anchored_cubic_coupled_unit_interval_c3_bounds,
    )
    from quantitative_nonaffine_coupled_c4_graph_transform import (  # type: ignore[no-redef]
        NonaffineCoupledC4GraphIterationBound,
    )


MATCHED_LOCAL_NONAFFINE_COUPLED_C4_INVARIANCE_KIND = (
    "GRAPH_DEPENDENT_EXACT_MATCHED_DOMAIN_C4_COLLAR_FULL_INVARIANT"
)


@dataclass(frozen=True)
class BoundaryAnchoredCubicCoupledUnitIntervalC4Bounds:
    base_inverse_lipschitz_upper: Fraction
    coupled_inverse_lipschitz_upper: Fraction
    coupled_collar_inverse_lipschitz_upper: Fraction
    base_map_hessian_upper: Fraction
    base_map_hessian_lipschitz_upper: Fraction
    base_map_third_derivative_lipschitz_upper: Fraction
    base_map_fourth_derivative_lipschitz_upper: Fraction
    forward_image_radius_exact: Fraction
    inverse_image_radius_exact: Fraction
    input_c4_extension_collar_radius: Fraction
    output_c4_extension_collar_radius: Fraction
    inverse_c4_collar_image_radius_upper: Fraction
    graph_boundary_value_exact: Fraction


@dataclass(frozen=True)
class QuantitativeMatchedLocalNonaffineCoupledC4GraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    differential_and_collar_robust_interior: bool
    domain_contact_robust_interior: bool
    local_nonaffine_coupled_c4_certificate: QuantitativeLocalNonaffineCoupledC4GraphTransformCertificate
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
    matched_local_c4_graph_real_dimension: int | None


def boundary_anchored_cubic_coupled_unit_interval_c4_bounds(
    *, amplitude_upper: object, fiber_coupling_upper: object,
    graph_slope_upper: object, input_c4_extension_collar_radius: object,
) -> BoundaryAnchoredCubicCoupledUnitIntervalC4Bounds:
    """Lift the exact cubic collar witness; its D4 and D5 base terms vanish."""
    c3 = boundary_anchored_cubic_coupled_unit_interval_c3_bounds(
        amplitude_upper=amplitude_upper,
        fiber_coupling_upper=fiber_coupling_upper,
        graph_slope_upper=graph_slope_upper,
        input_c3_extension_collar_radius=input_c4_extension_collar_radius,
    )
    return BoundaryAnchoredCubicCoupledUnitIntervalC4Bounds(
        base_inverse_lipschitz_upper=c3.base_inverse_lipschitz_upper,
        coupled_inverse_lipschitz_upper=c3.coupled_inverse_lipschitz_upper,
        coupled_collar_inverse_lipschitz_upper=c3.coupled_collar_inverse_lipschitz_upper,
        base_map_hessian_upper=c3.base_map_hessian_upper,
        base_map_hessian_lipschitz_upper=c3.base_map_hessian_lipschitz_upper,
        base_map_third_derivative_lipschitz_upper=c3.base_map_third_derivative_lipschitz_upper,
        base_map_fourth_derivative_lipschitz_upper=Fraction(0),
        forward_image_radius_exact=c3.forward_image_radius_exact,
        inverse_image_radius_exact=c3.inverse_image_radius_exact,
        input_c4_extension_collar_radius=c3.input_c3_extension_collar_radius,
        output_c4_extension_collar_radius=c3.output_c3_extension_collar_radius,
        inverse_c4_collar_image_radius_upper=c3.inverse_c3_collar_image_radius_upper,
        graph_boundary_value_exact=c3.graph_boundary_value_exact,
    )


def quantitative_matched_local_nonaffine_coupled_c4_graph_transform(
    *, input_base_domain_radius: object, output_base_domain_radius: object,
    uniform_inverse_base_image_radius_exact: object,
    uniform_forward_base_image_radius_exact: object,
    input_c4_extension_collar_radius: object,
    output_c4_extension_collar_radius: object,
    uniform_inverse_c4_collar_image_radius_upper: object,
    graph_boundary_value_upper: object, fiber_boundary_forcing_upper: object,
    **global_c4_inputs: object,
) -> QuantitativeMatchedLocalNonaffineCoupledC4GraphTransformCertificate:
    """Require open C4 collars, exact contacts, and anchored boundaries."""
    local = quantitative_local_nonaffine_coupled_c4_graph_transform(
        input_base_domain_radius=input_base_domain_radius,
        output_base_domain_radius=output_base_domain_radius,
        uniform_inverse_base_image_radius_upper=uniform_inverse_base_image_radius_exact,
        input_c4_extension_collar_radius=input_c4_extension_collar_radius,
        output_c4_extension_collar_radius=output_c4_extension_collar_radius,
        uniform_inverse_c4_collar_image_radius_upper=uniform_inverse_c4_collar_image_radius_upper,
        **global_c4_inputs,
    )
    forward_raw = _exact_fraction(
        uniform_forward_base_image_radius_exact,
        "uniform_forward_base_image_radius_exact",
    )
    graph_boundary_raw = _exact_fraction(graph_boundary_value_upper, "graph_boundary_value_upper")
    fiber_boundary_raw = _exact_fraction(fiber_boundary_forcing_upper, "fiber_boundary_forcing_upper")
    if forward_raw < 0 or graph_boundary_raw < 0 or fiber_boundary_raw < 0:
        raise ValueError("forward radius and boundary residuals must be nonnegative")

    base = local.nonaffine_coupled_c4_certificate.nonaffine_coupled_c3_certificate.nonaffine_coupled_c2_certificate.nonaffine_coupled_c1_certificate.coupled_lipschitz_certificate
    forward_radius = forward_raw / base.base_reference_scale
    graph_boundary = graph_boundary_raw / base.fiber_reference_scale
    fiber_boundary = fiber_boundary_raw / base.fiber_reference_scale
    inverse_contact = local.inverse_domain_coverage_margin
    forward_contact = local.normalized_output_base_domain_radius - forward_radius
    failures = list(local.failure_codes)
    if inverse_contact > 0:
        failures.append("MATCHED_LOCAL_NONAFFINE_COUPLED_C4_INVERSE_BOUNDARY_CONTACT_NOT_EXACT")
    if forward_contact < 0:
        failures.append("MATCHED_LOCAL_NONAFFINE_COUPLED_C4_FORWARD_BASE_DOMAIN_NOT_COVERED")
    elif forward_contact > 0:
        failures.append("MATCHED_LOCAL_NONAFFINE_COUPLED_C4_FORWARD_BOUNDARY_CONTACT_NOT_EXACT")
    if graph_boundary > 0:
        failures.append("MATCHED_LOCAL_NONAFFINE_COUPLED_C4_GRAPH_BOUNDARY_NOT_ANCHORED")
    if fiber_boundary > 0:
        failures.append("MATCHED_LOCAL_NONAFFINE_COUPLED_C4_FIBER_BOUNDARY_NOT_PRESERVED")

    differential_collar_robust = (
        local.nonaffine_coupled_c4_certificate.robust_interior
        and local.inverse_c4_collar_coverage_margin > 0
    )
    common = dict(
        failure_codes=tuple(failures), local_nonaffine_coupled_c4_certificate=local,
        raw_uniform_forward_base_image_radius_exact=forward_raw,
        normalized_uniform_forward_base_image_radius_exact=forward_radius,
        raw_graph_boundary_value_upper=graph_boundary_raw,
        normalized_graph_boundary_value_upper=graph_boundary,
        raw_fiber_boundary_forcing_upper=fiber_boundary_raw,
        normalized_fiber_boundary_forcing_upper=fiber_boundary,
        inverse_boundary_contact_residual=inverse_contact,
        forward_boundary_contact_residual=forward_contact,
        matched_domain_identity=MATCHED_LOCAL_NONAFFINE_COUPLED_C4_INVARIANCE_KIND,
        full_forward_retention_certified=False,
        boundary_anchored_graph_class_certified=False,
    )
    if failures:
        return QuantitativeMatchedLocalNonaffineCoupledC4GraphTransformCertificate(
            status=failures[0], validation_level=None, robust_interior=False,
            differential_and_collar_robust_interior=differential_collar_robust,
            domain_contact_robust_interior=False,
            matched_local_c4_graph_real_dimension=None, **common,
        )
    common["full_forward_retention_certified"] = True
    common["boundary_anchored_graph_class_certified"] = True
    return QuantitativeMatchedLocalNonaffineCoupledC4GraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_MATCHED_LOCAL_NONAFFINE_COUPLED_C4_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_MATCHED_LOCAL_NONAFFINE_COUPLED_C4_GRAPH_TRANSFORM",
        robust_interior=False,
        differential_and_collar_robust_interior=differential_collar_robust,
        domain_contact_robust_interior=False,
        matched_local_c4_graph_real_dimension=local.local_c4_graph_real_dimension,
        **common,
    )


def matched_local_nonaffine_coupled_c4_graph_iteration_bound(
    certificate: QuantitativeMatchedLocalNonaffineCoupledC4GraphTransformCertificate,
    *, initial_value_distance: object, initial_derivative_distance: object,
    initial_hessian_distance: object, initial_third_derivative_distance: object,
    initial_fourth_derivative_distance: object, steps: int,
) -> NonaffineCoupledC4GraphIterationBound:
    if certificate.validation_level is None:
        raise ValueError("matched local nonaffine coupled C4 iteration requires a verified certificate")
    return local_nonaffine_coupled_c4_graph_iteration_bound(
        certificate.local_nonaffine_coupled_c4_certificate,
        initial_value_distance=initial_value_distance,
        initial_derivative_distance=initial_derivative_distance,
        initial_hessian_distance=initial_hessian_distance,
        initial_third_derivative_distance=initial_third_derivative_distance,
        initial_fourth_derivative_distance=initial_fourth_derivative_distance,
        steps=steps,
    )


__all__ = [
    "BoundaryAnchoredCubicCoupledUnitIntervalC4Bounds",
    "MATCHED_LOCAL_NONAFFINE_COUPLED_C4_INVARIANCE_KIND",
    "QuantitativeMatchedLocalNonaffineCoupledC4GraphTransformCertificate",
    "boundary_anchored_cubic_coupled_unit_interval_c4_bounds",
    "matched_local_nonaffine_coupled_c4_graph_iteration_bound",
    "quantitative_matched_local_nonaffine_coupled_c4_graph_transform",
]
