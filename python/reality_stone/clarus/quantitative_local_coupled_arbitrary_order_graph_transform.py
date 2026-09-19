"""Local and exact-matched wrappers for conditional coupled finite C^n jets."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_coupled_arbitrary_order_graph_transform import (
        QuantitativeCoupledArbitraryOrderGraphTransformCertificate,
    )
    from .quantitative_coupled_arbitrary_order_implicit_jet import (
        CoupledImplicitJetIterationBound,
        coupled_implicit_jet_iteration_bound,
    )
    from .quantitative_graph_transform import _exact_fraction
else:
    from quantitative_coupled_arbitrary_order_graph_transform import (  # type: ignore[no-redef]
        QuantitativeCoupledArbitraryOrderGraphTransformCertificate,
    )
    from quantitative_coupled_arbitrary_order_implicit_jet import (  # type: ignore[no-redef]
        CoupledImplicitJetIterationBound,
        coupled_implicit_jet_iteration_bound,
    )
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]


LOCAL_COUPLED_ARBITRARY_ORDER_INVARIANCE_KIND = (
    "GRAPH_DEPENDENT_BACKWARD_COVERED_FINITE_CN_COLLAR_INVARIANT"
)
MATCHED_LOCAL_COUPLED_ARBITRARY_ORDER_INVARIANCE_KIND = (
    "GRAPH_DEPENDENT_EXACT_MATCHED_DOMAIN_FINITE_CN_COLLAR_FULL_INVARIANT"
)


@dataclass(frozen=True)
class QuantitativeLocalCoupledArbitraryOrderGraphTransformCertificate:
    status: str
    validation_level: str | None
    claim_scope: str
    failure_codes: tuple[str, ...]
    robust_interior: bool
    global_certificate: QuantitativeCoupledArbitraryOrderGraphTransformCertificate
    base_reference_scale: Fraction
    collar_moduli_covered_through_order: int
    raw_input_base_domain_radius: Fraction
    raw_output_base_domain_radius: Fraction
    raw_uniform_inverse_base_image_radius_upper: Fraction
    raw_input_extension_collar_radius: Fraction
    raw_output_extension_collar_radius: Fraction
    raw_uniform_inverse_collar_image_radius_upper: Fraction
    normalized_input_base_domain_radius: Fraction
    normalized_output_base_domain_radius: Fraction
    normalized_uniform_inverse_base_image_radius_upper: Fraction
    normalized_input_extension_collar_radius: Fraction
    normalized_output_extension_collar_radius: Fraction
    normalized_uniform_inverse_collar_image_radius_upper: Fraction
    inverse_domain_coverage_margin: Fraction
    inverse_collar_coverage_margin: Fraction
    extension_collar_certified: bool
    local_invariance_kind: str
    forward_retention_certified: bool
    local_cn_graph_real_dimension: int | None


@dataclass(frozen=True)
class QuantitativeMatchedLocalCoupledArbitraryOrderGraphTransformCertificate:
    status: str
    validation_level: str | None
    claim_scope: str
    failure_codes: tuple[str, ...]
    robust_interior: bool
    differential_and_collar_robust_interior: bool
    domain_contact_robust_interior: bool
    local_certificate: QuantitativeLocalCoupledArbitraryOrderGraphTransformCertificate
    fiber_reference_scale: Fraction
    raw_uniform_inverse_base_image_radius_exact: Fraction
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
    matched_local_cn_graph_real_dimension: int | None


def quantitative_local_coupled_arbitrary_order_graph_transform(
    *,
    global_certificate: QuantitativeCoupledArbitraryOrderGraphTransformCertificate,
    base_reference_scale: object,
    collar_moduli_covered_through_order: int,
    input_base_domain_radius: object,
    output_base_domain_radius: object,
    uniform_inverse_base_image_radius_upper: object,
    input_extension_collar_radius: object,
    output_extension_collar_radius: object,
    uniform_inverse_collar_image_radius_upper: object,
) -> QuantitativeLocalCoupledArbitraryOrderGraphTransformCertificate:
    """Require core and open-collar inverse coverage for every supplied jet."""
    if not isinstance(global_certificate, QuantitativeCoupledArbitraryOrderGraphTransformCertificate):
        raise ValueError("global_certificate must be a coupled arbitrary-order certificate")
    if type(collar_moduli_covered_through_order) is not int or collar_moduli_covered_through_order < 0:
        raise ValueError("collar_moduli_covered_through_order must be a nonnegative integer")
    scale = _exact_fraction(base_reference_scale, "base_reference_scale")
    input_raw = _exact_fraction(input_base_domain_radius, "input_base_domain_radius")
    output_raw = _exact_fraction(output_base_domain_radius, "output_base_domain_radius")
    inverse_raw = _exact_fraction(
        uniform_inverse_base_image_radius_upper,
        "uniform_inverse_base_image_radius_upper",
    )
    input_collar_raw = _exact_fraction(input_extension_collar_radius, "input_extension_collar_radius")
    output_collar_raw = _exact_fraction(output_extension_collar_radius, "output_extension_collar_radius")
    collar_inverse_raw = _exact_fraction(
        uniform_inverse_collar_image_radius_upper,
        "uniform_inverse_collar_image_radius_upper",
    )
    if scale <= 0 or input_raw <= 0 or output_raw <= 0:
        raise ValueError("reference scale and base-domain radii must be positive")
    if inverse_raw < 0 or collar_inverse_raw < 0:
        raise ValueError("inverse-image radii must be nonnegative")
    if input_collar_raw < 0 or output_collar_raw < 0:
        raise ValueError("extension-collar radii must be nonnegative")

    input_radius = input_raw / scale
    output_radius = output_raw / scale
    inverse_radius = inverse_raw / scale
    input_collar = input_collar_raw / scale
    output_collar = output_collar_raw / scale
    collar_inverse = collar_inverse_raw / scale
    core_margin = input_radius - inverse_radius
    collar_margin = input_radius + input_collar - collar_inverse
    required_coverage_order = global_certificate.maximum_order + 1
    failures = list(global_certificate.failure_codes)
    if global_certificate.validation_level is None and not failures:
        failures.append("LOCAL_COUPLED_CN_GLOBAL_DERIVATIVE_CERTIFICATE_NOT_VERIFIED")
    if collar_moduli_covered_through_order < required_coverage_order:
        failures.append("LOCAL_COUPLED_CN_COLLAR_MODULI_ORDER_INSUFFICIENT")
    if core_margin < 0:
        failures.append("LOCAL_COUPLED_CN_INVERSE_BASE_DOMAIN_NOT_COVERED")
    if input_collar <= 0:
        failures.append("LOCAL_COUPLED_CN_INPUT_EXTENSION_COLLAR_NOT_OPEN")
    if output_collar <= 0:
        failures.append("LOCAL_COUPLED_CN_OUTPUT_EXTENSION_COLLAR_NOT_OPEN")
    if collar_inverse < inverse_radius:
        failures.append("LOCAL_COUPLED_CN_COLLAR_INVERSE_BOUND_BELOW_CORE")
    if collar_margin < 0:
        failures.append("LOCAL_COUPLED_CN_INVERSE_EXTENSION_COLLAR_NOT_COVERED")

    scope = (
        "CONDITIONAL_LOCAL_FINITE_CN_DERIVATIVE_JET_THEOREM; COLLAR_MODULI_"
        "COVERAGE_IS_A_DECLARED_HYPOTHESIS_AND_FORWARD_RETENTION_IS_SEPARATE"
    )
    common = dict(
        claim_scope=scope, failure_codes=tuple(failures),
        global_certificate=global_certificate, base_reference_scale=scale,
        collar_moduli_covered_through_order=collar_moduli_covered_through_order,
        raw_input_base_domain_radius=input_raw,
        raw_output_base_domain_radius=output_raw,
        raw_uniform_inverse_base_image_radius_upper=inverse_raw,
        raw_input_extension_collar_radius=input_collar_raw,
        raw_output_extension_collar_radius=output_collar_raw,
        raw_uniform_inverse_collar_image_radius_upper=collar_inverse_raw,
        normalized_input_base_domain_radius=input_radius,
        normalized_output_base_domain_radius=output_radius,
        normalized_uniform_inverse_base_image_radius_upper=inverse_radius,
        normalized_input_extension_collar_radius=input_collar,
        normalized_output_extension_collar_radius=output_collar,
        normalized_uniform_inverse_collar_image_radius_upper=collar_inverse,
        inverse_domain_coverage_margin=core_margin,
        inverse_collar_coverage_margin=collar_margin,
        local_invariance_kind=LOCAL_COUPLED_ARBITRARY_ORDER_INVARIANCE_KIND,
        forward_retention_certified=False,
    )
    if failures:
        return QuantitativeLocalCoupledArbitraryOrderGraphTransformCertificate(
            status=failures[0], validation_level=None, robust_interior=False,
            extension_collar_certified=False, local_cn_graph_real_dimension=None,
            **common,
        )
    status = f"VERIFIED_CONDITIONAL_LOCAL_COUPLED_C{global_certificate.maximum_order}_DERIVATIVE_JET_GRAPH_TRANSFORM"
    return QuantitativeLocalCoupledArbitraryOrderGraphTransformCertificate(
        status=status, validation_level=status,
        robust_interior=(global_certificate.robust_interior and core_margin > 0 and collar_margin > 0),
        extension_collar_certified=True,
        local_cn_graph_real_dimension=global_certificate.cn_graph_real_dimension,
        **common,
    )


def quantitative_matched_local_coupled_arbitrary_order_graph_transform(
    *,
    local_certificate: QuantitativeLocalCoupledArbitraryOrderGraphTransformCertificate,
    fiber_reference_scale: object,
    uniform_inverse_base_image_radius_exact: object,
    uniform_forward_base_image_radius_exact: object,
    graph_boundary_value_upper: object,
    fiber_boundary_forcing_upper: object,
) -> QuantitativeMatchedLocalCoupledArbitraryOrderGraphTransformCertificate:
    """Add exact inverse/forward contacts and anchored graph/fiber boundaries."""
    if not isinstance(local_certificate, QuantitativeLocalCoupledArbitraryOrderGraphTransformCertificate):
        raise ValueError("local_certificate must be a local coupled Cn certificate")
    fiber_scale = _exact_fraction(fiber_reference_scale, "fiber_reference_scale")
    inverse_exact_raw = _exact_fraction(
        uniform_inverse_base_image_radius_exact,
        "uniform_inverse_base_image_radius_exact",
    )
    forward_raw = _exact_fraction(
        uniform_forward_base_image_radius_exact,
        "uniform_forward_base_image_radius_exact",
    )
    graph_boundary_raw = _exact_fraction(graph_boundary_value_upper, "graph_boundary_value_upper")
    fiber_boundary_raw = _exact_fraction(fiber_boundary_forcing_upper, "fiber_boundary_forcing_upper")
    if fiber_scale <= 0:
        raise ValueError("fiber_reference_scale must be positive")
    if min(inverse_exact_raw, forward_raw, graph_boundary_raw, fiber_boundary_raw) < 0:
        raise ValueError("exact radii and boundary residuals must be nonnegative")

    scale = local_certificate.base_reference_scale
    inverse_exact = inverse_exact_raw / scale
    forward_radius = forward_raw / scale
    graph_boundary = graph_boundary_raw / fiber_scale
    fiber_boundary = fiber_boundary_raw / fiber_scale
    inverse_contact = local_certificate.normalized_input_base_domain_radius - inverse_exact
    forward_contact = local_certificate.normalized_output_base_domain_radius - forward_radius
    failures = list(local_certificate.failure_codes)
    if inverse_exact_raw != local_certificate.raw_uniform_inverse_base_image_radius_upper:
        failures.append("MATCHED_LOCAL_COUPLED_CN_INVERSE_EXACT_BOUND_MISMATCH")
    if inverse_contact != 0:
        failures.append("MATCHED_LOCAL_COUPLED_CN_INVERSE_BOUNDARY_CONTACT_NOT_EXACT")
    if forward_contact < 0:
        failures.append("MATCHED_LOCAL_COUPLED_CN_FORWARD_BASE_DOMAIN_NOT_COVERED")
    elif forward_contact > 0:
        failures.append("MATCHED_LOCAL_COUPLED_CN_FORWARD_BOUNDARY_CONTACT_NOT_EXACT")
    if graph_boundary > 0:
        failures.append("MATCHED_LOCAL_COUPLED_CN_GRAPH_BOUNDARY_NOT_ANCHORED")
    if fiber_boundary > 0:
        failures.append("MATCHED_LOCAL_COUPLED_CN_FIBER_BOUNDARY_NOT_PRESERVED")

    differential_collar_robust = (
        local_certificate.global_certificate.robust_interior
        and local_certificate.inverse_collar_coverage_margin > 0
    )
    scope = (
        "CONDITIONAL_EXACT_MATCHED_LOCAL_FINITE_CN_DERIVATIVE_JET_THEOREM; "
        "EXACT_DOMAIN_CONTACTS_ARE_NOT_ROBUST_INTERIOR_MARGINS"
    )
    common = dict(
        claim_scope=scope, failure_codes=tuple(failures),
        local_certificate=local_certificate, fiber_reference_scale=fiber_scale,
        raw_uniform_inverse_base_image_radius_exact=inverse_exact_raw,
        raw_uniform_forward_base_image_radius_exact=forward_raw,
        normalized_uniform_forward_base_image_radius_exact=forward_radius,
        raw_graph_boundary_value_upper=graph_boundary_raw,
        normalized_graph_boundary_value_upper=graph_boundary,
        raw_fiber_boundary_forcing_upper=fiber_boundary_raw,
        normalized_fiber_boundary_forcing_upper=fiber_boundary,
        inverse_boundary_contact_residual=inverse_contact,
        forward_boundary_contact_residual=forward_contact,
        matched_domain_identity=MATCHED_LOCAL_COUPLED_ARBITRARY_ORDER_INVARIANCE_KIND,
        full_forward_retention_certified=False,
        boundary_anchored_graph_class_certified=False,
    )
    if failures:
        return QuantitativeMatchedLocalCoupledArbitraryOrderGraphTransformCertificate(
            status=failures[0], validation_level=None, robust_interior=False,
            differential_and_collar_robust_interior=differential_collar_robust,
            domain_contact_robust_interior=False,
            matched_local_cn_graph_real_dimension=None, **common,
        )
    common["full_forward_retention_certified"] = True
    common["boundary_anchored_graph_class_certified"] = True
    order = local_certificate.global_certificate.maximum_order
    status = f"VERIFIED_CONDITIONAL_MATCHED_LOCAL_COUPLED_C{order}_DERIVATIVE_JET_GRAPH_TRANSFORM"
    return QuantitativeMatchedLocalCoupledArbitraryOrderGraphTransformCertificate(
        status=status, validation_level=status, robust_interior=False,
        differential_and_collar_robust_interior=differential_collar_robust,
        domain_contact_robust_interior=False,
        matched_local_cn_graph_real_dimension=local_certificate.local_cn_graph_real_dimension,
        **common,
    )


def local_coupled_arbitrary_order_iteration_bound(
    certificate: QuantitativeLocalCoupledArbitraryOrderGraphTransformCertificate,
    *, initial_distance_by_derivative_order: object, steps: int,
) -> CoupledImplicitJetIterationBound:
    if certificate.validation_level is None:
        raise ValueError("local iteration requires a verified local Cn certificate")
    return coupled_implicit_jet_iteration_bound(
        certificate.global_certificate.implicit_jet_certificate,
        initial_distance_by_derivative_order=initial_distance_by_derivative_order,
        steps=steps,
    )


def matched_local_coupled_arbitrary_order_iteration_bound(
    certificate: QuantitativeMatchedLocalCoupledArbitraryOrderGraphTransformCertificate,
    *, initial_distance_by_derivative_order: object, steps: int,
) -> CoupledImplicitJetIterationBound:
    if certificate.validation_level is None:
        raise ValueError("matched local iteration requires a verified matched Cn certificate")
    return local_coupled_arbitrary_order_iteration_bound(
        certificate.local_certificate,
        initial_distance_by_derivative_order=initial_distance_by_derivative_order,
        steps=steps,
    )


__all__ = [
    "LOCAL_COUPLED_ARBITRARY_ORDER_INVARIANCE_KIND",
    "MATCHED_LOCAL_COUPLED_ARBITRARY_ORDER_INVARIANCE_KIND",
    "QuantitativeLocalCoupledArbitraryOrderGraphTransformCertificate",
    "QuantitativeMatchedLocalCoupledArbitraryOrderGraphTransformCertificate",
    "local_coupled_arbitrary_order_iteration_bound",
    "matched_local_coupled_arbitrary_order_iteration_bound",
    "quantitative_local_coupled_arbitrary_order_graph_transform",
    "quantitative_matched_local_coupled_arbitrary_order_graph_transform",
]
