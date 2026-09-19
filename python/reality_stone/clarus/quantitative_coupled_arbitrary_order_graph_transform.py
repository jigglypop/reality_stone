"""Conditional finite C^n coupled graph transform from normalized map moduli."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_arbitrary_order_graph_transform import bell_partition_terms
    from .quantitative_coupled_arbitrary_order_implicit_jet import (
        QuantitativeCoupledArbitraryOrderImplicitJetCertificate,
        quantitative_coupled_arbitrary_order_implicit_jet_recurrence,
    )
    from .quantitative_graph_transform import _exact_fraction
else:
    from quantitative_arbitrary_order_graph_transform import bell_partition_terms  # type: ignore[no-redef]
    from quantitative_coupled_arbitrary_order_implicit_jet import (  # type: ignore[no-redef]
        QuantitativeCoupledArbitraryOrderImplicitJetCertificate,
        quantitative_coupled_arbitrary_order_implicit_jet_recurrence,
    )
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]


@dataclass(frozen=True)
class CoupledCompositeRawJetEnvelope:
    maximum_order: int
    preimage_value_coupling_upper: Fraction
    state_value_coupling_upper: Fraction
    graph_embedding_jet_bounds: tuple[Fraction, ...]
    graph_embedding_difference_coefficients_upper: tuple[tuple[Fraction, ...], ...]
    map_derivative_bounds: tuple[Fraction, ...]
    map_derivative_point_lipschitz_bounds: tuple[Fraction, ...]
    composite_jet_bounds: tuple[Fraction, ...]
    composite_difference_coefficients_upper: tuple[tuple[Fraction, ...], ...]


@dataclass(frozen=True)
class QuantitativeCoupledArbitraryOrderGraphTransformCertificate:
    status: str
    validation_level: str | None
    claim_scope: str
    failure_codes: tuple[str, ...]
    robust_interior: bool
    base_dimension: int
    maximum_order: int
    normalized_graph_derivative_bounds_with_point_modulus: tuple[Fraction, ...]
    base_raw_jet_envelope: CoupledCompositeRawJetEnvelope
    fiber_raw_jet_envelope: CoupledCompositeRawJetEnvelope
    implicit_jet_certificate: QuantitativeCoupledArbitraryOrderImplicitJetCertificate
    cn_graph_real_dimension: int | None


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


def _product(
    radii: tuple[Fraction, ...], multiplicities: tuple[int, ...],
    *, removed_order: int | None = None,
) -> Fraction:
    value = Fraction(1)
    for order, count in enumerate(multiplicities, start=1):
        exponent = count - (1 if removed_order == order else 0)
        if exponent:
            value *= radii[order - 1] ** exponent
    return value


def coupled_composite_raw_jet_envelopes(
    *,
    normalized_graph_derivative_bounds_with_point_modulus: object,
    map_derivative_bounds: object,
    map_derivative_point_lipschitz_bounds: object,
    preimage_value_coupling_upper: object,
) -> CoupledCompositeRawJetEnvelope:
    """Bound jets of ``a compose (Id,h)`` at graph-dependent matched preimages.

    Graph bounds cover D1..D(n+1); the last entry controls displacement of
    D^n h between the two preimages.  Map rows cover D1..Dn and the pointwise
    Lipschitz moduli of those same derivatives.
    """
    graph = _fraction_tuple(
        normalized_graph_derivative_bounds_with_point_modulus,
        "normalized_graph_derivative_bounds_with_point_modulus",
    )
    maps = _fraction_tuple(map_derivative_bounds, "map_derivative_bounds")
    map_lip = _fraction_tuple(
        map_derivative_point_lipschitz_bounds,
        "map_derivative_point_lipschitz_bounds",
    )
    maximum_order = len(maps)
    if maximum_order < 1 or len(map_lip) != maximum_order or len(graph) != maximum_order + 1:
        raise ValueError("graph D1..D(n+1) and map D1..Dn size/Lipschitz rows are required")
    rx = _exact_fraction(preimage_value_coupling_upper, "preimage_value_coupling_upper")
    if rx < 0:
        raise ValueError("preimage_value_coupling_upper must be nonnegative")

    # Product norm on (x,h(x)): ||D(Id,h)|| <= 1+kappa.
    embedding = (1 + graph[0],) + graph[1:maximum_order]
    state = 1 + embedding[0] * rx
    embedding_delta: list[tuple[Fraction, ...]] = []
    for order in range(1, maximum_order + 1):
        row = [Fraction(0) for _ in range(order + 1)]
        row[0] = graph[order] * rx
        row[order] = 1
        embedding_delta.append(tuple(row))

    jet_bounds: list[Fraction] = []
    difference_rows: list[tuple[Fraction, ...]] = []
    for order in range(1, maximum_order + 1):
        size = Fraction(0)
        row = [Fraction(0) for _ in range(order + 1)]
        for term in bell_partition_terms(order):
            outer_order = term.block_count
            bell = Fraction(term.combinatorial_coefficient)
            product = _product(embedding, term.multiplicities)
            outer_size = maps[outer_order - 1]
            size += bell * outer_size * product
            # Change of D^b a between the matched graph states.
            row[0] += bell * map_lip[outer_order - 1] * state * product
            # Change of each D^j(Id,h) slot, including preimage displacement.
            for slot_order, multiplicity in enumerate(term.multiplicities, start=1):
                if not multiplicity:
                    continue
                slot_weight = (
                    bell * outer_size * multiplicity
                    * _product(embedding, term.multiplicities, removed_order=slot_order)
                )
                slot_delta = embedding_delta[slot_order - 1]
                for index, coefficient in enumerate(slot_delta):
                    row[index] += slot_weight * coefficient
        jet_bounds.append(size)
        difference_rows.append(tuple(row))

    return CoupledCompositeRawJetEnvelope(
        maximum_order=maximum_order,
        preimage_value_coupling_upper=rx,
        state_value_coupling_upper=state,
        graph_embedding_jet_bounds=embedding,
        graph_embedding_difference_coefficients_upper=tuple(embedding_delta),
        map_derivative_bounds=maps,
        map_derivative_point_lipschitz_bounds=map_lip,
        composite_jet_bounds=tuple(jet_bounds),
        composite_difference_coefficients_upper=tuple(difference_rows),
    )


def quantitative_coupled_arbitrary_order_graph_transform(
    *,
    base_dimension: int,
    base_invertibility_lower: object,
    preimage_value_coupling_upper: object,
    normalized_graph_derivative_bounds_with_point_modulus: object,
    base_map_derivative_bounds: object,
    base_map_derivative_point_lipschitz_bounds: object,
    fiber_map_derivative_bounds: object,
    fiber_map_derivative_point_lipschitz_bounds: object,
) -> QuantitativeCoupledArbitraryOrderGraphTransformCertificate:
    """Generate raw coupled jets and solve the graph-dependent implicit system."""
    if type(base_dimension) is not int or base_dimension < 1:
        raise ValueError("base_dimension must be a positive built-in integer")
    graph = _fraction_tuple(
        normalized_graph_derivative_bounds_with_point_modulus,
        "normalized_graph_derivative_bounds_with_point_modulus",
    )
    base_raw = coupled_composite_raw_jet_envelopes(
        normalized_graph_derivative_bounds_with_point_modulus=graph,
        map_derivative_bounds=base_map_derivative_bounds,
        map_derivative_point_lipschitz_bounds=base_map_derivative_point_lipschitz_bounds,
        preimage_value_coupling_upper=preimage_value_coupling_upper,
    )
    fiber_raw = coupled_composite_raw_jet_envelopes(
        normalized_graph_derivative_bounds_with_point_modulus=graph,
        map_derivative_bounds=fiber_map_derivative_bounds,
        map_derivative_point_lipschitz_bounds=fiber_map_derivative_point_lipschitz_bounds,
        preimage_value_coupling_upper=preimage_value_coupling_upper,
    )
    implicit = quantitative_coupled_arbitrary_order_implicit_jet_recurrence(
        base_dimension=base_dimension,
        base_invertibility_lower=base_invertibility_lower,
        normalized_graph_derivative_bounds=graph[:-1],
        forward_jet_bounds=base_raw.composite_jet_bounds,
        observed_output_jet_bounds=fiber_raw.composite_jet_bounds,
        forward_difference_coefficients_upper=base_raw.composite_difference_coefficients_upper,
        observed_output_difference_coefficients_upper=fiber_raw.composite_difference_coefficients_upper,
    )
    scope = (
        "CONDITIONAL_NORMALIZED_COUPLED_DERIVATIVE_JET_THEOREM; REQUIRES_"
        "SUPPLIED_GLOBAL_MAP_MODULI_PREIMAGE_COUPLING_AND_A_SEPARATE_C0_GATE"
    )
    failures = implicit.failure_codes
    common = dict(
        claim_scope=scope,
        base_dimension=base_dimension,
        maximum_order=base_raw.maximum_order,
        normalized_graph_derivative_bounds_with_point_modulus=graph,
        base_raw_jet_envelope=base_raw,
        fiber_raw_jet_envelope=fiber_raw,
        implicit_jet_certificate=implicit,
    )
    if implicit.validation_level is None:
        return QuantitativeCoupledArbitraryOrderGraphTransformCertificate(
            status=implicit.status, validation_level=None,
            failure_codes=failures, robust_interior=False,
            cn_graph_real_dimension=None, **common,
        )
    status = f"VERIFIED_CONDITIONAL_COUPLED_C{base_raw.maximum_order}_DERIVATIVE_JET_GRAPH_TRANSFORM"
    return QuantitativeCoupledArbitraryOrderGraphTransformCertificate(
        status=status, validation_level=status, failure_codes=(),
        robust_interior=implicit.robust_interior,
        cn_graph_real_dimension=base_dimension, **common,
    )


__all__ = [
    "CoupledCompositeRawJetEnvelope",
    "QuantitativeCoupledArbitraryOrderGraphTransformCertificate",
    "coupled_composite_raw_jet_envelopes",
    "quantitative_coupled_arbitrary_order_graph_transform",
]
