"""Exact local-domain and extension-collar wrapper for coupled nonaffine C4."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_coupled_graph_transform import _exact_fraction
    from .quantitative_nonaffine_coupled_c4_graph_transform import (
        NonaffineCoupledC4GraphIterationBound,
        QuantitativeNonaffineCoupledC4GraphTransformCertificate,
        nonaffine_coupled_c4_graph_iteration_bound,
        quantitative_nonaffine_coupled_c4_graph_transform,
    )
else:
    from quantitative_coupled_graph_transform import _exact_fraction  # type: ignore[no-redef]
    from quantitative_nonaffine_coupled_c4_graph_transform import (  # type: ignore[no-redef]
        NonaffineCoupledC4GraphIterationBound,
        QuantitativeNonaffineCoupledC4GraphTransformCertificate,
        nonaffine_coupled_c4_graph_iteration_bound,
        quantitative_nonaffine_coupled_c4_graph_transform,
    )


LOCAL_NONAFFINE_COUPLED_C4_INVARIANCE_KIND = (
    "GRAPH_DEPENDENT_BACKWARD_COVERED_C4_COLLAR_INVARIANT"
)


@dataclass(frozen=True)
class QuantitativeLocalNonaffineCoupledC4GraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    nonaffine_coupled_c4_certificate: QuantitativeNonaffineCoupledC4GraphTransformCertificate
    raw_input_base_domain_radius: Fraction
    raw_output_base_domain_radius: Fraction
    raw_uniform_inverse_base_image_radius_upper: Fraction
    raw_input_c4_extension_collar_radius: Fraction
    raw_output_c4_extension_collar_radius: Fraction
    raw_uniform_inverse_c4_collar_image_radius_upper: Fraction
    normalized_input_base_domain_radius: Fraction
    normalized_output_base_domain_radius: Fraction
    normalized_uniform_inverse_base_image_radius_upper: Fraction
    normalized_input_c4_extension_collar_radius: Fraction
    normalized_output_c4_extension_collar_radius: Fraction
    normalized_uniform_inverse_c4_collar_image_radius_upper: Fraction
    inverse_domain_coverage_margin: Fraction
    inverse_c4_collar_coverage_margin: Fraction
    c4_extension_collar_certified: bool
    local_invariance_kind: str
    forward_retention_certified: bool
    local_c4_graph_real_dimension: int | None


def quantitative_local_nonaffine_coupled_c4_graph_transform(
    *, input_base_domain_radius: object, output_base_domain_radius: object,
    uniform_inverse_base_image_radius_upper: object,
    input_c4_extension_collar_radius: object,
    output_c4_extension_collar_radius: object,
    uniform_inverse_c4_collar_image_radius_upper: object,
    **global_c4_inputs: object,
) -> QuantitativeLocalNonaffineCoupledC4GraphTransformCertificate:
    """Compose global C4 gates with core and open-collar inverse coverage."""
    predecessor = quantitative_nonaffine_coupled_c4_graph_transform(**global_c4_inputs)
    input_raw = _exact_fraction(input_base_domain_radius, "input_base_domain_radius")
    output_raw = _exact_fraction(output_base_domain_radius, "output_base_domain_radius")
    inverse_raw = _exact_fraction(
        uniform_inverse_base_image_radius_upper,
        "uniform_inverse_base_image_radius_upper",
    )
    input_collar_raw = _exact_fraction(
        input_c4_extension_collar_radius, "input_c4_extension_collar_radius"
    )
    output_collar_raw = _exact_fraction(
        output_c4_extension_collar_radius, "output_c4_extension_collar_radius"
    )
    collar_inverse_raw = _exact_fraction(
        uniform_inverse_c4_collar_image_radius_upper,
        "uniform_inverse_c4_collar_image_radius_upper",
    )
    if input_raw <= 0 or output_raw <= 0:
        raise ValueError("input and output base-domain radii must be positive")
    if inverse_raw < 0 or collar_inverse_raw < 0:
        raise ValueError("inverse-image radii must be nonnegative")
    if input_collar_raw < 0 or output_collar_raw < 0:
        raise ValueError("C4 extension-collar radii must be nonnegative")

    base = predecessor.nonaffine_coupled_c3_certificate.nonaffine_coupled_c2_certificate.nonaffine_coupled_c1_certificate.coupled_lipschitz_certificate
    x_scale = base.base_reference_scale
    input_radius = input_raw / x_scale
    output_radius = output_raw / x_scale
    inverse_radius = inverse_raw / x_scale
    input_collar = input_collar_raw / x_scale
    output_collar = output_collar_raw / x_scale
    collar_inverse = collar_inverse_raw / x_scale
    core_margin = input_radius - inverse_radius
    collar_margin = input_radius + input_collar - collar_inverse

    failures = list(predecessor.failure_codes)
    if core_margin < 0:
        failures.append("LOCAL_NONAFFINE_COUPLED_C4_INVERSE_BASE_DOMAIN_NOT_COVERED")
    if input_collar <= 0:
        failures.append("LOCAL_NONAFFINE_COUPLED_C4_INPUT_EXTENSION_COLLAR_NOT_OPEN")
    if output_collar <= 0:
        failures.append("LOCAL_NONAFFINE_COUPLED_C4_OUTPUT_EXTENSION_COLLAR_NOT_OPEN")
    if collar_inverse < inverse_radius:
        failures.append("LOCAL_NONAFFINE_COUPLED_C4_COLLAR_INVERSE_BOUND_BELOW_CORE")
    if collar_margin < 0:
        failures.append("LOCAL_NONAFFINE_COUPLED_C4_INVERSE_EXTENSION_COLLAR_NOT_COVERED")

    common = dict(
        failure_codes=tuple(failures), nonaffine_coupled_c4_certificate=predecessor,
        raw_input_base_domain_radius=input_raw,
        raw_output_base_domain_radius=output_raw,
        raw_uniform_inverse_base_image_radius_upper=inverse_raw,
        raw_input_c4_extension_collar_radius=input_collar_raw,
        raw_output_c4_extension_collar_radius=output_collar_raw,
        raw_uniform_inverse_c4_collar_image_radius_upper=collar_inverse_raw,
        normalized_input_base_domain_radius=input_radius,
        normalized_output_base_domain_radius=output_radius,
        normalized_uniform_inverse_base_image_radius_upper=inverse_radius,
        normalized_input_c4_extension_collar_radius=input_collar,
        normalized_output_c4_extension_collar_radius=output_collar,
        normalized_uniform_inverse_c4_collar_image_radius_upper=collar_inverse,
        inverse_domain_coverage_margin=core_margin,
        inverse_c4_collar_coverage_margin=collar_margin,
        local_invariance_kind=LOCAL_NONAFFINE_COUPLED_C4_INVARIANCE_KIND,
        forward_retention_certified=False,
    )
    if failures:
        return QuantitativeLocalNonaffineCoupledC4GraphTransformCertificate(
            status=failures[0], validation_level=None, robust_interior=False,
            c4_extension_collar_certified=False,
            local_c4_graph_real_dimension=None, **common,
        )
    return QuantitativeLocalNonaffineCoupledC4GraphTransformCertificate(
        status="VERIFIED_QUANTITATIVE_LOCAL_NONAFFINE_COUPLED_C4_GRAPH_TRANSFORM",
        validation_level="VERIFIED_QUANTITATIVE_LOCAL_NONAFFINE_COUPLED_C4_GRAPH_TRANSFORM",
        robust_interior=(predecessor.robust_interior and core_margin > 0 and collar_margin > 0),
        c4_extension_collar_certified=True,
        local_c4_graph_real_dimension=predecessor.c4_graph_real_dimension,
        **common,
    )


def local_nonaffine_coupled_c4_graph_iteration_bound(
    certificate: QuantitativeLocalNonaffineCoupledC4GraphTransformCertificate,
    *, initial_value_distance: object, initial_derivative_distance: object,
    initial_hessian_distance: object, initial_third_derivative_distance: object,
    initial_fourth_derivative_distance: object, steps: int,
) -> NonaffineCoupledC4GraphIterationBound:
    if certificate.validation_level is None:
        raise ValueError("local nonaffine coupled C4 iteration requires a verified certificate")
    return nonaffine_coupled_c4_graph_iteration_bound(
        certificate.nonaffine_coupled_c4_certificate,
        initial_value_distance=initial_value_distance,
        initial_derivative_distance=initial_derivative_distance,
        initial_hessian_distance=initial_hessian_distance,
        initial_third_derivative_distance=initial_third_derivative_distance,
        initial_fourth_derivative_distance=initial_fourth_derivative_distance,
        steps=steps,
    )


__all__ = [
    "LOCAL_NONAFFINE_COUPLED_C4_INVARIANCE_KIND",
    "QuantitativeLocalNonaffineCoupledC4GraphTransformCertificate",
    "local_nonaffine_coupled_c4_graph_iteration_bound",
    "quantitative_local_nonaffine_coupled_c4_graph_transform",
]
