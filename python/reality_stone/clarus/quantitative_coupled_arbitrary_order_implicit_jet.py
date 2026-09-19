"""Finite-order Bell solver for graph-dependent implicit jet equations.

The module is deliberately an algebraic layer.  It certifies the passage from
raw envelopes for ``Y_h = T_h compose F_h`` to envelopes for ``T_h``.  A map-
specific graph-transform theorem must additionally certify the supplied raw
``F`` and ``Y`` jets.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_arbitrary_order_graph_transform import (
        BellPartitionTerm,
        bell_partition_terms,
    )
    from .quantitative_graph_transform import _exact_fraction
else:
    from quantitative_arbitrary_order_graph_transform import (  # type: ignore[no-redef]
        BellPartitionTerm,
        bell_partition_terms,
    )
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]


@dataclass(frozen=True)
class CoupledImplicitJetLevelCertificate:
    order: int
    partition_terms: tuple[BellPartitionTerm, ...]
    isolated_all_singletons: tuple[int, ...]
    lower_composition_numerator_upper: Fraction
    transform_jet_upper: Fraction
    graph_jet_margin: Fraction
    numerator_difference_coefficients_upper: tuple[Fraction, ...]
    inverse_slot_correction_coefficients_upper: tuple[Fraction, ...]
    coefficient_by_input_order_upper: tuple[Fraction, ...]
    derivative_bunching_factor_upper: Fraction
    derivative_bunching_margin: Fraction


@dataclass(frozen=True)
class QuantitativeCoupledArbitraryOrderImplicitJetCertificate:
    status: str
    validation_level: str | None
    claim_scope: str
    failure_codes: tuple[str, ...]
    robust_interior: bool
    base_dimension: int
    maximum_order: int
    base_invertibility_lower: Fraction
    normalized_graph_derivative_bounds: tuple[Fraction, ...]
    forward_jet_bounds: tuple[Fraction, ...]
    observed_output_jet_bounds: tuple[Fraction, ...]
    forward_difference_coefficients_upper: tuple[tuple[Fraction, ...], ...]
    observed_output_difference_coefficients_upper: tuple[tuple[Fraction, ...], ...]
    levels: tuple[CoupledImplicitJetLevelCertificate, ...]
    cn_graph_real_dimension: int | None


@dataclass(frozen=True)
class CoupledImplicitJetIterationBound:
    steps: int
    distance_by_derivative_order_upper: tuple[Fraction, ...]


def _product(
    bounds: tuple[Fraction, ...], multiplicities: tuple[int, ...],
    *, removed_order: int | None = None,
) -> Fraction:
    value = Fraction(1)
    for order, count in enumerate(multiplicities, start=1):
        exponent = count - (1 if removed_order == order else 0)
        if exponent:
            value *= bounds[order - 1] ** exponent
    return value


def _fraction_tuple(values: object, name: str) -> tuple[Fraction, ...]:
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{name} must be a tuple or list")
    result = tuple(
        _exact_fraction(value, f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if min(result, default=Fraction(0)) < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _triangular_coefficients(
    rows: object, *, maximum_order: int, name: str,
) -> tuple[tuple[Fraction, ...], ...]:
    if not isinstance(rows, (tuple, list)) or len(rows) != maximum_order:
        raise ValueError(f"{name} must contain one row for each order")
    result: list[tuple[Fraction, ...]] = []
    for order, row in enumerate(rows, start=1):
        values = _fraction_tuple(row, f"{name}[{order - 1}]")
        if len(values) != order + 1:
            raise ValueError(f"{name} order {order} must cover D0..D{order}")
        result.append(values)
    return tuple(result)


def _pad(row: tuple[Fraction, ...], length: int) -> tuple[Fraction, ...]:
    return row + (Fraction(0),) * (length - len(row))


def quantitative_coupled_arbitrary_order_implicit_jet_recurrence(
    *,
    base_dimension: int,
    base_invertibility_lower: object,
    normalized_graph_derivative_bounds: object,
    forward_jet_bounds: object,
    observed_output_jet_bounds: object,
    forward_difference_coefficients_upper: object,
    observed_output_difference_coefficients_upper: object,
) -> QuantitativeCoupledArbitraryOrderImplicitJetCertificate:
    """Solve raw finite jets of ``Y_h=T_h compose F_h`` for the jets of ``T_h``.

    All quantities are normalized operator-norm envelopes.  Difference rows
    are upper-triangular coefficient rows over graph distances D0..Dj.  The
    result is conditional on those raw rows being valid for the concrete map.
    """
    if type(base_dimension) is not int or base_dimension < 1:
        raise ValueError("base_dimension must be a positive built-in integer")
    alpha = _exact_fraction(base_invertibility_lower, "base_invertibility_lower")
    if alpha <= 0:
        raise ValueError("base_invertibility_lower must be positive")
    graph = _fraction_tuple(
        normalized_graph_derivative_bounds, "normalized_graph_derivative_bounds"
    )
    forward = _fraction_tuple(forward_jet_bounds, "forward_jet_bounds")
    observed = _fraction_tuple(observed_output_jet_bounds, "observed_output_jet_bounds")
    maximum_order = len(graph)
    if maximum_order < 2 or len(forward) != maximum_order or len(observed) != maximum_order:
        raise ValueError("graph, forward, and observed jets must cover D1..Dn with n>=2")
    if forward[0] < alpha:
        raise ValueError("the DF norm upper bound cannot be below its conorm lower bound")
    delta_forward = _triangular_coefficients(
        forward_difference_coefficients_upper,
        maximum_order=maximum_order,
        name="forward_difference_coefficients_upper",
    )
    delta_observed = _triangular_coefficients(
        observed_output_difference_coefficients_upper,
        maximum_order=maximum_order,
        name="observed_output_difference_coefficients_upper",
    )

    transform: list[Fraction] = []
    transform_delta: list[tuple[Fraction, ...]] = []
    levels: list[CoupledImplicitJetLevelCertificate] = []
    failures: list[str] = []
    for order in range(1, maximum_order + 1):
        partitions = bell_partition_terms(order)
        # D^n T is carried by n singleton input blocks, not by one n-block.
        all_singletons = (order,) + (0,) * (order - 1)
        lower_terms = tuple(
            term for term in partitions if term.multiplicities != all_singletons
        )
        numerator = observed[order - 1]
        numerator_delta = list(_pad(delta_observed[order - 1], order + 1))
        for term in lower_terms:
            outer_order = term.block_count
            outer_size = transform[outer_order - 1]
            product = _product(forward, term.multiplicities)
            coefficient = Fraction(term.combinatorial_coefficient)
            numerator += coefficient * outer_size * product

            outer_delta = _pad(transform_delta[outer_order - 1], order + 1)
            for index in range(order + 1):
                numerator_delta[index] += coefficient * product * outer_delta[index]
            for slot_order, multiplicity in enumerate(term.multiplicities, start=1):
                if not multiplicity:
                    continue
                slot_weight = (
                    coefficient * outer_size * multiplicity
                    * _product(forward, term.multiplicities, removed_order=slot_order)
                )
                slot_delta = _pad(delta_forward[slot_order - 1], order + 1)
                for index in range(order + 1):
                    numerator_delta[index] += slot_weight * slot_delta[index]

        inverse_scale = alpha ** (-order)
        transform_size = inverse_scale * numerator
        inverse_weight = order * numerator * alpha ** (-(order + 1))
        inverse_correction = tuple(
            inverse_weight * value
            for value in _pad(delta_forward[0], order + 1)
        )
        coefficients = tuple(
            inverse_scale * numerator_delta[index] + inverse_correction[index]
            for index in range(order + 1)
        )
        beta = coefficients[order]
        margin = graph[order - 1] - transform_size
        beta_margin = 1 - beta
        if margin < 0:
            failures.append(f"CONDITIONAL_IMPLICIT_C{order}_GRAPH_JET_CLASS_NOT_INVARIANT")
        if beta_margin <= 0:
            failures.append(f"CONDITIONAL_IMPLICIT_C{order}_JET_BUNCHING_NOT_STRICT")
        transform.append(transform_size)
        transform_delta.append(coefficients)
        levels.append(CoupledImplicitJetLevelCertificate(
            order=order,
            partition_terms=partitions,
            isolated_all_singletons=all_singletons,
            lower_composition_numerator_upper=numerator,
            transform_jet_upper=transform_size,
            graph_jet_margin=margin,
            numerator_difference_coefficients_upper=tuple(numerator_delta),
            inverse_slot_correction_coefficients_upper=inverse_correction,
            coefficient_by_input_order_upper=coefficients,
            derivative_bunching_factor_upper=beta,
            derivative_bunching_margin=beta_margin,
        ))

    scope = (
        "CONDITIONAL_ALGEBRAIC_JET_SOLVER; RAW_F_AND_Y_JET_ENVELOPES_REQUIRE_"
        "A_SEPARATE_MAP_SPECIFIC_PROOF"
    )
    common = dict(
        claim_scope=scope,
        base_dimension=base_dimension,
        maximum_order=maximum_order,
        base_invertibility_lower=alpha,
        normalized_graph_derivative_bounds=graph,
        forward_jet_bounds=forward,
        observed_output_jet_bounds=observed,
        forward_difference_coefficients_upper=delta_forward,
        observed_output_difference_coefficients_upper=delta_observed,
        levels=tuple(levels),
    )
    robust = not failures and all(level.graph_jet_margin > 0 for level in levels)
    if failures:
        return QuantitativeCoupledArbitraryOrderImplicitJetCertificate(
            status=failures[0], validation_level=None,
            failure_codes=tuple(failures), robust_interior=False,
            cn_graph_real_dimension=None, **common,
        )
    status = f"VERIFIED_CONDITIONAL_COUPLED_C{maximum_order}_IMPLICIT_JET_RECURRENCE"
    return QuantitativeCoupledArbitraryOrderImplicitJetCertificate(
        status=status, validation_level=status, failure_codes=(),
        robust_interior=robust, cn_graph_real_dimension=base_dimension, **common,
    )


def coupled_implicit_jet_iteration_bound(
    certificate: QuantitativeCoupledArbitraryOrderImplicitJetCertificate,
    *, initial_distance_by_derivative_order: object, steps: int,
) -> CoupledImplicitJetIterationBound:
    """Synchronously iterate the certified upper-triangular D0..Dn system."""
    if certificate.validation_level is None:
        raise ValueError("iteration requires a verified conditional jet certificate")
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative built-in integer")
    values = _fraction_tuple(
        initial_distance_by_derivative_order, "initial_distance_by_derivative_order"
    )
    if len(values) != certificate.maximum_order + 1:
        raise ValueError("initial distances must cover D0..Dn")
    # The raw recurrence does not by itself certify the C0 graph transform.
    # Preserve D0 as an external input while iterating the conditional jet rows.
    for _ in range(steps):
        old = values
        values = (old[0],) + tuple(
            sum(coefficient * old[index] for index, coefficient in enumerate(level.coefficient_by_input_order_upper))
            for level in certificate.levels
        )
    return CoupledImplicitJetIterationBound(steps=steps, distance_by_derivative_order_upper=values)


__all__ = [
    "CoupledImplicitJetIterationBound",
    "CoupledImplicitJetLevelCertificate",
    "QuantitativeCoupledArbitraryOrderImplicitJetCertificate",
    "coupled_implicit_jet_iteration_bound",
    "quantitative_coupled_arbitrary_order_implicit_jet_recurrence",
]
