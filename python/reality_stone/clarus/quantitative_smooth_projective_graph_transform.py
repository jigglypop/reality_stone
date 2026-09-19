"""Finite-prefix audit for a nonanalytic projective C-infinity route.

No finite execution can verify a statement quantified over every derivative
order.  This module therefore certifies compatible finite prefixes and their
joint triangular convergence, while permanently refusing to promote a finite
prefix to a C-infinity conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_full_coupled_arbitrary_order_graph_transform import (
        FullCoupledArbitraryOrderIterationBound,
        QuantitativeFullCoupledArbitraryOrderGraphTransformCertificate,
        full_coupled_arbitrary_order_iteration_bound,
    )
else:
    from quantitative_full_coupled_arbitrary_order_graph_transform import (  # type: ignore[no-redef]
        FullCoupledArbitraryOrderIterationBound,
        QuantitativeFullCoupledArbitraryOrderGraphTransformCertificate,
        full_coupled_arbitrary_order_iteration_bound,
    )


@dataclass(frozen=True)
class QuantitativeSmoothProjectivePrefixCertificate:
    status: str
    validation_level: str | None
    claim_scope: str
    failure_codes: tuple[str, ...]
    robust_interior: bool
    certificates_by_maximum_order: tuple[QuantitativeFullCoupledArbitraryOrderGraphTransformCertificate, ...]
    covered_orders: tuple[int, ...]
    maximum_verified_order: int
    transition_matrix_upper: tuple[tuple[Fraction, ...], ...]
    maximum_diagonal_factor_upper: Fraction | None
    triangular_convergence_margin: Fraction | None
    every_finite_order_hypothesis_verified: bool
    projective_limit_completeness_hypothesis_verified: bool
    common_orbit_hypothesis_verified: bool
    cinfinity_claim_admitted: bool
    cn_graph_real_dimension: int | None


def _raw_signature(envelope, order: int):
    return (
        envelope.preimage_value_coupling_upper,
        envelope.state_value_coupling_upper,
        envelope.graph_embedding_jet_bounds[:order],
        tuple(row[: index + 2] for index, row in enumerate(envelope.graph_embedding_difference_coefficients_upper[:order])),
        envelope.map_derivative_bounds[:order],
        envelope.map_derivative_point_lipschitz_bounds[:order],
        envelope.composite_jet_bounds[:order],
        tuple(row[: index + 2] for index, row in enumerate(envelope.composite_difference_coefficients_upper[:order])),
    )


def _prefix_signature(certificate, order: int):
    derivative = certificate.derivative_certificate
    implicit = derivative.implicit_jet_certificate
    return (
        certificate.c0_certificate,
        derivative.base_dimension,
        derivative.normalized_graph_derivative_bounds_with_point_modulus[: order + 1],
        _raw_signature(derivative.base_raw_jet_envelope, order),
        _raw_signature(derivative.fiber_raw_jet_envelope, order),
        implicit.base_invertibility_lower,
        implicit.normalized_graph_derivative_bounds[:order],
        implicit.forward_jet_bounds[:order],
        implicit.observed_output_jet_bounds[:order],
        implicit.forward_difference_coefficients_upper[:order],
        implicit.observed_output_difference_coefficients_upper[:order],
        implicit.levels[:order],
    )


def _transition_matrix(certificate):
    size = certificate.maximum_order + 1
    q0 = certificate.c0_certificate.transform_contraction_factor_upper
    if q0 is None:
        return ()
    rows = [(q0,) + (Fraction(0),) * (size - 1)]
    for level in certificate.derivative_certificate.implicit_jet_certificate.levels:
        row = level.coefficient_by_input_order_upper
        rows.append(row + (Fraction(0),) * (size - len(row)))
    return tuple(rows)


def quantitative_smooth_projective_prefix_graph_transform(
    *, certificates_by_maximum_order: object,
) -> QuantitativeSmoothProjectivePrefixCertificate:
    """Audit a compatible C2..CN prefix without claiming C-infinity."""
    if not isinstance(certificates_by_maximum_order, (tuple, list)):
        raise ValueError("certificates_by_maximum_order must be a tuple or list")
    certificates = tuple(certificates_by_maximum_order)
    if not certificates:
        raise ValueError("at least one full finite-order certificate is required")
    if any(not isinstance(item, QuantitativeFullCoupledArbitraryOrderGraphTransformCertificate) for item in certificates):
        raise ValueError("every item must be a full coupled C0-to-Cn certificate")

    orders = tuple(item.maximum_order for item in certificates)
    failures: list[str] = []
    if orders != tuple(range(2, orders[-1] + 1)):
        failures.append("SMOOTH_PROJECTIVE_PREFIX_ORDERS_NOT_CONSECUTIVE_FROM_C2")
    for item in certificates:
        failures.extend(item.failure_codes)
        if item.validation_level is None and not item.failure_codes:
            failures.append("SMOOTH_PROJECTIVE_PREFIX_MEMBER_NOT_VERIFIED")
    for lower, higher in zip(certificates, certificates[1:]):
        if _prefix_signature(higher, lower.maximum_order) != _prefix_signature(lower, lower.maximum_order):
            failures.append(
                f"SMOOTH_PROJECTIVE_C{lower.maximum_order}_PREFIX_INCOMPATIBLE_WITH_C{higher.maximum_order}"
            )

    highest = certificates[-1]
    matrix = _transition_matrix(highest)
    diagonal = tuple(matrix[index][index] for index in range(len(matrix))) if matrix else ()
    maximum_diagonal = max(diagonal) if diagonal else None
    margin = None if maximum_diagonal is None else 1 - maximum_diagonal
    if margin is None or margin <= 0:
        failures.append("SMOOTH_PROJECTIVE_PREFIX_TRIANGULAR_CONVERGENCE_NOT_STRICT")

    scope = (
        "VERIFIED_COMPATIBLE_FINITE_CN_PREFIX_AND_TRIANGULAR_CONVERGENCE; "
        "CINFINITY_REQUIRES_ALL_NATURAL_ORDERS_PROJECTIVE_COMPLETENESS_AND_ONE_COMMON_ORBIT"
    )
    common = dict(
        claim_scope=scope,
        failure_codes=tuple(failures),
        certificates_by_maximum_order=certificates,
        covered_orders=orders,
        maximum_verified_order=orders[-1],
        transition_matrix_upper=matrix,
        maximum_diagonal_factor_upper=maximum_diagonal,
        triangular_convergence_margin=margin,
        every_finite_order_hypothesis_verified=False,
        projective_limit_completeness_hypothesis_verified=False,
        common_orbit_hypothesis_verified=False,
        cinfinity_claim_admitted=False,
    )
    if failures:
        return QuantitativeSmoothProjectivePrefixCertificate(
            status=failures[0], validation_level=None, robust_interior=False,
            cn_graph_real_dimension=None, **common,
        )
    status = f"VERIFIED_COMPATIBLE_SMOOTH_PROJECTIVE_C2_TO_C{orders[-1]}_PREFIX"
    return QuantitativeSmoothProjectivePrefixCertificate(
        status=status, validation_level=status, robust_interior=False,
        cn_graph_real_dimension=highest.full_cn_graph_real_dimension,
        **common,
    )


def smooth_projective_prefix_iteration_bound(
    certificate: QuantitativeSmoothProjectivePrefixCertificate,
    *, initial_distance_by_derivative_order: object, steps: int,
) -> FullCoupledArbitraryOrderIterationBound:
    if certificate.validation_level is None:
        raise ValueError("iteration requires a verified compatible finite prefix")
    return full_coupled_arbitrary_order_iteration_bound(
        certificate.certificates_by_maximum_order[-1],
        initial_distance_by_derivative_order=initial_distance_by_derivative_order,
        steps=steps,
    )


__all__ = [
    "QuantitativeSmoothProjectivePrefixCertificate",
    "quantitative_smooth_projective_prefix_graph_transform",
    "smooth_projective_prefix_iteration_bound",
]
