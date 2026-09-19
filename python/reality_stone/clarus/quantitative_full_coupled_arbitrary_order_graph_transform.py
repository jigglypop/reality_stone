"""Same-version C0-to-Cn certificates for the coupled graph transform.

This adapter closes a deliberately exposed gap between the C0 invariant-ball
and contraction theorem and the conditional finite-jet theorem.  It does not
derive new map estimates: it checks that both predecessor certificates refer
to the same normalized constants, then iterates their bounds synchronously.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_coupled_graph_transform import QuantitativeCoupledGraphTransformCertificate
    from .quantitative_coupled_arbitrary_order_graph_transform import (
        QuantitativeCoupledArbitraryOrderGraphTransformCertificate,
    )
    from .quantitative_local_coupled_arbitrary_order_graph_transform import (
        QuantitativeLocalCoupledArbitraryOrderGraphTransformCertificate,
        QuantitativeMatchedLocalCoupledArbitraryOrderGraphTransformCertificate,
    )
    from .quantitative_graph_transform import _exact_fraction
else:
    from quantitative_coupled_graph_transform import QuantitativeCoupledGraphTransformCertificate  # type: ignore[no-redef]
    from quantitative_coupled_arbitrary_order_graph_transform import (  # type: ignore[no-redef]
        QuantitativeCoupledArbitraryOrderGraphTransformCertificate,
    )
    from quantitative_local_coupled_arbitrary_order_graph_transform import (  # type: ignore[no-redef]
        QuantitativeLocalCoupledArbitraryOrderGraphTransformCertificate,
        QuantitativeMatchedLocalCoupledArbitraryOrderGraphTransformCertificate,
    )
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]


@dataclass(frozen=True)
class QuantitativeFullCoupledArbitraryOrderGraphTransformCertificate:
    status: str
    validation_level: str | None
    claim_scope: str
    failure_codes: tuple[str, ...]
    robust_interior: bool
    differential_robust_interior: bool
    constant_contact_robust_interior: bool
    c0_certificate: QuantitativeCoupledGraphTransformCertificate
    derivative_certificate: QuantitativeCoupledArbitraryOrderGraphTransformCertificate
    maximum_order: int
    expected_preimage_value_coupling_upper: Fraction
    c0_fiber_first_jet_numerator_upper: Fraction
    full_cn_graph_real_dimension: int | None


@dataclass(frozen=True)
class QuantitativeFullLocalCoupledArbitraryOrderGraphTransformCertificate:
    status: str
    validation_level: str | None
    claim_scope: str
    failure_codes: tuple[str, ...]
    robust_interior: bool
    differential_and_collar_robust_interior: bool
    constant_contact_robust_interior: bool
    global_certificate: QuantitativeFullCoupledArbitraryOrderGraphTransformCertificate
    derivative_local_certificate: QuantitativeLocalCoupledArbitraryOrderGraphTransformCertificate
    full_local_cn_graph_real_dimension: int | None


@dataclass(frozen=True)
class QuantitativeFullMatchedLocalCoupledArbitraryOrderGraphTransformCertificate:
    status: str
    validation_level: str | None
    claim_scope: str
    failure_codes: tuple[str, ...]
    robust_interior: bool
    differential_and_collar_robust_interior: bool
    constant_and_domain_contact_robust_interior: bool
    local_certificate: QuantitativeFullLocalCoupledArbitraryOrderGraphTransformCertificate
    derivative_matched_certificate: QuantitativeMatchedLocalCoupledArbitraryOrderGraphTransformCertificate
    full_matched_local_cn_graph_real_dimension: int | None


@dataclass(frozen=True)
class FullCoupledArbitraryOrderIterationBound:
    steps: int
    distance_by_derivative_order_upper: tuple[Fraction, ...]


def quantitative_full_coupled_arbitrary_order_graph_transform(
    *,
    c0_certificate: QuantitativeCoupledGraphTransformCertificate,
    derivative_certificate: QuantitativeCoupledArbitraryOrderGraphTransformCertificate,
) -> QuantitativeFullCoupledArbitraryOrderGraphTransformCertificate:
    """Join verified C0 and Cn certificates through exact normalized contacts."""
    if not isinstance(c0_certificate, QuantitativeCoupledGraphTransformCertificate):
        raise ValueError("c0_certificate must be a quantitative coupled C0 certificate")
    if not isinstance(derivative_certificate, QuantitativeCoupledArbitraryOrderGraphTransformCertificate):
        raise ValueError("derivative_certificate must be a coupled arbitrary-order certificate")

    failures = list(c0_certificate.failure_codes)
    if c0_certificate.validation_level is None and not failures:
        failures.append("FULL_COUPLED_CN_C0_CERTIFICATE_NOT_VERIFIED")
    failures.extend(derivative_certificate.failure_codes)
    if derivative_certificate.validation_level is None and not derivative_certificate.failure_codes:
        failures.append("FULL_COUPLED_CN_DERIVATIVE_CERTIFICATE_NOT_VERIFIED")

    implicit = derivative_certificate.implicit_jet_certificate
    alpha = c0_certificate.base_invertibility_lower
    expected_rx = (
        Fraction(0) if alpha <= 0
        else c0_certificate.normalized_fiber_to_base_lipschitz / alpha
    )
    slope_numerator = (
        c0_certificate.fiber_factor_upper * c0_certificate.normalized_graph_slope_upper
        + c0_certificate.normalized_base_to_fiber_lipschitz
    )
    if c0_certificate.base_dimension != derivative_certificate.base_dimension:
        failures.append("FULL_COUPLED_CN_BASE_DIMENSION_MISMATCH")
    if implicit.base_invertibility_lower != alpha:
        failures.append("FULL_COUPLED_CN_BASE_INVERTIBILITY_LOWER_MISMATCH")
    if derivative_certificate.normalized_graph_derivative_bounds_with_point_modulus[0] != c0_certificate.normalized_graph_slope_upper:
        failures.append("FULL_COUPLED_CN_GRAPH_SLOPE_MISMATCH")
    if (
        derivative_certificate.base_raw_jet_envelope.preimage_value_coupling_upper != expected_rx
        or derivative_certificate.fiber_raw_jet_envelope.preimage_value_coupling_upper != expected_rx
    ):
        failures.append("FULL_COUPLED_CN_PREIMAGE_COUPLING_MISMATCH")
    if derivative_certificate.fiber_raw_jet_envelope.composite_jet_bounds[0] < slope_numerator:
        failures.append("FULL_COUPLED_CN_FIBER_FIRST_JET_UNDER_C0_SLOPE_NUMERATOR")

    scope = (
        "CONDITIONAL_SAME_VERSION_NORMALIZED_COUPLED_C0_TO_FINITE_CN_THEOREM; "
        "EXACT_CONSTANT_CONTACTS_IDENTIFY_THE_TWO_PREDECESSOR_CERTIFICATES"
    )
    common = dict(
        claim_scope=scope,
        failure_codes=tuple(failures),
        c0_certificate=c0_certificate,
        derivative_certificate=derivative_certificate,
        maximum_order=derivative_certificate.maximum_order,
        expected_preimage_value_coupling_upper=expected_rx,
        c0_fiber_first_jet_numerator_upper=slope_numerator,
    )
    if failures:
        return QuantitativeFullCoupledArbitraryOrderGraphTransformCertificate(
            status=failures[0], validation_level=None, robust_interior=False,
            differential_robust_interior=False,
            constant_contact_robust_interior=False, full_cn_graph_real_dimension=None,
            **common,
        )
    status = f"VERIFIED_CONDITIONAL_FULL_COUPLED_C0_TO_C{derivative_certificate.maximum_order}_GRAPH_TRANSFORM"
    return QuantitativeFullCoupledArbitraryOrderGraphTransformCertificate(
        status=status, validation_level=status,
        robust_interior=False,
        differential_robust_interior=(
            c0_certificate.robust_interior and derivative_certificate.robust_interior
        ),
        constant_contact_robust_interior=False,
        full_cn_graph_real_dimension=c0_certificate.base_dimension,
        **common,
    )


def quantitative_full_local_coupled_arbitrary_order_graph_transform(
    *,
    global_certificate: QuantitativeFullCoupledArbitraryOrderGraphTransformCertificate,
    derivative_local_certificate: QuantitativeLocalCoupledArbitraryOrderGraphTransformCertificate,
) -> QuantitativeFullLocalCoupledArbitraryOrderGraphTransformCertificate:
    """Attach the already-certified finite-Cn local collar to the full global gate."""
    if not isinstance(global_certificate, QuantitativeFullCoupledArbitraryOrderGraphTransformCertificate):
        raise ValueError("global_certificate must be a full coupled C0-to-Cn certificate")
    if not isinstance(derivative_local_certificate, QuantitativeLocalCoupledArbitraryOrderGraphTransformCertificate):
        raise ValueError("derivative_local_certificate must be a local coupled Cn certificate")
    failures = list(global_certificate.failure_codes)
    failures.extend(derivative_local_certificate.failure_codes)
    if derivative_local_certificate.validation_level is None and not derivative_local_certificate.failure_codes:
        failures.append("FULL_LOCAL_COUPLED_CN_DERIVATIVE_LOCAL_CERTIFICATE_NOT_VERIFIED")
    if derivative_local_certificate.global_certificate != global_certificate.derivative_certificate:
        failures.append("FULL_LOCAL_COUPLED_CN_DERIVATIVE_CERTIFICATE_MISMATCH")
    if derivative_local_certificate.base_reference_scale != global_certificate.c0_certificate.base_reference_scale:
        failures.append("FULL_LOCAL_COUPLED_CN_BASE_REFERENCE_SCALE_MISMATCH")
    scope = (
        "CONDITIONAL_SAME_VERSION_LOCAL_COUPLED_C0_TO_FINITE_CN_THEOREM; "
        "THE_OPEN_COLLAR_AND_GLOBAL_CONSTANT CONTACTS_ARE_BOTH_REQUIRED"
    )
    common = dict(
        claim_scope=scope, failure_codes=tuple(failures),
        global_certificate=global_certificate,
        derivative_local_certificate=derivative_local_certificate,
    )
    if failures:
        return QuantitativeFullLocalCoupledArbitraryOrderGraphTransformCertificate(
            status=failures[0], validation_level=None, robust_interior=False,
            differential_and_collar_robust_interior=False,
            constant_contact_robust_interior=False, full_local_cn_graph_real_dimension=None,
            **common,
        )
    order = global_certificate.maximum_order
    status = f"VERIFIED_CONDITIONAL_FULL_LOCAL_COUPLED_C0_TO_C{order}_GRAPH_TRANSFORM"
    return QuantitativeFullLocalCoupledArbitraryOrderGraphTransformCertificate(
        status=status, validation_level=status,
        robust_interior=False,
        differential_and_collar_robust_interior=(
            global_certificate.differential_robust_interior
            and derivative_local_certificate.global_certificate.robust_interior
            and derivative_local_certificate.inverse_collar_coverage_margin > 0
        ),
        constant_contact_robust_interior=False,
        full_local_cn_graph_real_dimension=global_certificate.full_cn_graph_real_dimension,
        **common,
    )


def quantitative_full_matched_local_coupled_arbitrary_order_graph_transform(
    *,
    local_certificate: QuantitativeFullLocalCoupledArbitraryOrderGraphTransformCertificate,
    derivative_matched_certificate: QuantitativeMatchedLocalCoupledArbitraryOrderGraphTransformCertificate,
) -> QuantitativeFullMatchedLocalCoupledArbitraryOrderGraphTransformCertificate:
    """Attach exact domain/boundary matching without calling equalities robust."""
    if not isinstance(local_certificate, QuantitativeFullLocalCoupledArbitraryOrderGraphTransformCertificate):
        raise ValueError("local_certificate must be a full local C0-to-Cn certificate")
    if not isinstance(derivative_matched_certificate, QuantitativeMatchedLocalCoupledArbitraryOrderGraphTransformCertificate):
        raise ValueError("derivative_matched_certificate must be a matched local Cn certificate")
    failures = list(local_certificate.failure_codes)
    failures.extend(derivative_matched_certificate.failure_codes)
    if derivative_matched_certificate.validation_level is None and not derivative_matched_certificate.failure_codes:
        failures.append("FULL_MATCHED_LOCAL_COUPLED_CN_DERIVATIVE_MATCHED_CERTIFICATE_NOT_VERIFIED")
    if derivative_matched_certificate.local_certificate != local_certificate.derivative_local_certificate:
        failures.append("FULL_MATCHED_LOCAL_COUPLED_CN_LOCAL_CERTIFICATE_MISMATCH")
    if derivative_matched_certificate.fiber_reference_scale != local_certificate.global_certificate.c0_certificate.fiber_reference_scale:
        failures.append("FULL_MATCHED_LOCAL_COUPLED_CN_FIBER_REFERENCE_SCALE_MISMATCH")
    scope = (
        "CONDITIONAL_SAME_VERSION_EXACT_MATCHED_LOCAL_COUPLED_C0_TO_FINITE_CN_THEOREM; "
        "EXACT_CONSTANT_AND_DOMAIN CONTACTS_ARE_NOT ROBUST INTERIOR MARGINS"
    )
    differential = (
        local_certificate.differential_and_collar_robust_interior
        and derivative_matched_certificate.differential_and_collar_robust_interior
    )
    common = dict(
        claim_scope=scope, failure_codes=tuple(failures),
        local_certificate=local_certificate,
        derivative_matched_certificate=derivative_matched_certificate,
        differential_and_collar_robust_interior=differential,
        constant_and_domain_contact_robust_interior=False,
    )
    if failures:
        return QuantitativeFullMatchedLocalCoupledArbitraryOrderGraphTransformCertificate(
            status=failures[0], validation_level=None, robust_interior=False,
            full_matched_local_cn_graph_real_dimension=None, **common,
        )
    order = local_certificate.global_certificate.maximum_order
    status = f"VERIFIED_CONDITIONAL_FULL_MATCHED_LOCAL_COUPLED_C0_TO_C{order}_GRAPH_TRANSFORM"
    return QuantitativeFullMatchedLocalCoupledArbitraryOrderGraphTransformCertificate(
        status=status, validation_level=status, robust_interior=False,
        full_matched_local_cn_graph_real_dimension=local_certificate.full_local_cn_graph_real_dimension,
        **common,
    )


def full_coupled_arbitrary_order_iteration_bound(
    certificate: QuantitativeFullCoupledArbitraryOrderGraphTransformCertificate,
    *, initial_distance_by_derivative_order: object, steps: int,
) -> FullCoupledArbitraryOrderIterationBound:
    """Synchronously iterate C0 contraction and all triangular C1..Cn rows."""
    if certificate.validation_level is None:
        raise ValueError("iteration requires a verified full C0-to-Cn certificate")
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative built-in integer")
    if not isinstance(initial_distance_by_derivative_order, (tuple, list)):
        raise ValueError("initial distances must be a tuple or list")
    values = tuple(
        _exact_fraction(value, f"initial_distance_by_derivative_order[{index}]")
        for index, value in enumerate(initial_distance_by_derivative_order)
    )
    if len(values) != certificate.maximum_order + 1:
        raise ValueError("initial distances must cover D0..Dn")
    if min(values, default=Fraction(0)) < 0:
        raise ValueError("initial distances must be nonnegative")
    q0 = certificate.c0_certificate.transform_contraction_factor_upper
    if q0 is None:
        raise ValueError("verified full certificate must have a C0 contraction factor")
    levels = certificate.derivative_certificate.implicit_jet_certificate.levels
    for _ in range(steps):
        old = values
        values = (q0 * old[0],) + tuple(
            sum(coefficient * old[index] for index, coefficient in enumerate(level.coefficient_by_input_order_upper))
            for level in levels
        )
    return FullCoupledArbitraryOrderIterationBound(
        steps=steps, distance_by_derivative_order_upper=values
    )


def full_local_coupled_arbitrary_order_iteration_bound(
    certificate: QuantitativeFullLocalCoupledArbitraryOrderGraphTransformCertificate,
    *, initial_distance_by_derivative_order: object, steps: int,
) -> FullCoupledArbitraryOrderIterationBound:
    if certificate.validation_level is None:
        raise ValueError("iteration requires a verified full local certificate")
    return full_coupled_arbitrary_order_iteration_bound(
        certificate.global_certificate,
        initial_distance_by_derivative_order=initial_distance_by_derivative_order,
        steps=steps,
    )


def full_matched_local_coupled_arbitrary_order_iteration_bound(
    certificate: QuantitativeFullMatchedLocalCoupledArbitraryOrderGraphTransformCertificate,
    *, initial_distance_by_derivative_order: object, steps: int,
) -> FullCoupledArbitraryOrderIterationBound:
    if certificate.validation_level is None:
        raise ValueError("iteration requires a verified full matched local certificate")
    return full_local_coupled_arbitrary_order_iteration_bound(
        certificate.local_certificate,
        initial_distance_by_derivative_order=initial_distance_by_derivative_order,
        steps=steps,
    )


__all__ = [
    "FullCoupledArbitraryOrderIterationBound",
    "QuantitativeFullCoupledArbitraryOrderGraphTransformCertificate",
    "QuantitativeFullLocalCoupledArbitraryOrderGraphTransformCertificate",
    "QuantitativeFullMatchedLocalCoupledArbitraryOrderGraphTransformCertificate",
    "full_coupled_arbitrary_order_iteration_bound",
    "full_local_coupled_arbitrary_order_iteration_bound",
    "full_matched_local_coupled_arbitrary_order_iteration_bound",
    "quantitative_full_coupled_arbitrary_order_graph_transform",
    "quantitative_full_local_coupled_arbitrary_order_graph_transform",
    "quantitative_full_matched_local_coupled_arbitrary_order_graph_transform",
]
