"""Exact Bell-polynomial hierarchy for affine triangular C^n graph transforms."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import factorial

if __package__:
    from .quantitative_c4_graph_transform import QuantitativeC4GraphTransformCertificate
    from .quantitative_graph_transform import _exact_fraction
else:
    from quantitative_c4_graph_transform import QuantitativeC4GraphTransformCertificate  # type: ignore[no-redef]
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]


@dataclass(frozen=True)
class BellPartitionTerm:
    order: int
    multiplicities: tuple[int, ...]
    block_count: int
    combinatorial_coefficient: int


@dataclass(frozen=True)
class ArbitraryOrderLevelCertificate:
    order: int
    partition_terms: tuple[BellPartitionTerm, ...]
    raw_composite_derivative_upper: Fraction
    output_graph_derivative_upper: Fraction
    graph_derivative_margin: Fraction
    derivative_bunching_factor_upper: Fraction
    derivative_bunching_margin: Fraction
    coefficient_by_input_order_upper: tuple[Fraction, ...]


@dataclass(frozen=True)
class QuantitativeArbitraryOrderGraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    c4_certificate: QuantitativeC4GraphTransformCertificate
    maximum_order: int
    normalized_graph_derivative_bounds: tuple[Fraction, ...]
    normalized_map_derivative_bounds: tuple[Fraction, ...]
    levels: tuple[ArbitraryOrderLevelCertificate, ...]
    cn_graph_real_dimension: int | None


@dataclass(frozen=True)
class ArbitraryOrderGraphIterationBound:
    steps: int
    distance_by_derivative_order_upper: tuple[Fraction, ...]


def bell_partition_terms(order: int) -> tuple[BellPartitionTerm, ...]:
    """Enumerate multiplicity vectors m_j with sum(j*m_j)=order exactly."""
    if type(order) is not int or order < 1:
        raise ValueError("order must be a positive built-in integer")
    terms: list[BellPartitionTerm] = []

    def visit(j: int, remaining: int, reversed_counts: list[int]) -> None:
        if j == 0:
            if remaining != 0:
                return
            multiplicities = tuple(reversed(reversed_counts))
            blocks = sum(multiplicities)
            denominator = 1
            for slot_order, count in enumerate(multiplicities, start=1):
                denominator *= factorial(slot_order) ** count * factorial(count)
            coefficient = factorial(order) // denominator
            terms.append(BellPartitionTerm(order, multiplicities, blocks, coefficient))
            return
        for count in range(remaining // j, -1, -1):
            visit(j - 1, remaining - count * j, reversed_counts + [count])

    visit(order, order, [])
    return tuple(sorted(terms, key=lambda term: term.multiplicities, reverse=True))


def _product_with_removed_slot(
    radii: tuple[Fraction, ...], multiplicities: tuple[int, ...], removed_order: int | None,
) -> Fraction:
    product = Fraction(1)
    for slot_order, count in enumerate(multiplicities, start=1):
        effective = count - (1 if removed_order == slot_order else 0)
        if effective:
            product *= radii[slot_order - 1] ** effective
    return product


def _level(
    *, order: int, q: Fraction, mu: Fraction,
    graph_bounds: tuple[Fraction, ...], map_bounds: tuple[Fraction, ...],
) -> ArbitraryOrderLevelCertificate:
    partitions = bell_partition_terms(order)
    singleton = (0,) * (order - 1) + (1,)
    nonlinear_terms = tuple(term for term in partitions if term.multiplicities != singleton)
    radii = (1 + graph_bounds[0],) + graph_bounds[1:order]
    lambda_n = graph_bounds[order - 1]
    raw = q * lambda_n
    coefficients = [Fraction(0) for _ in range(order + 1)]
    coefficients[order] = q
    coefficients[0] = map_bounds[0] * lambda_n
    for term in nonlinear_terms:
        k_m = map_bounds[term.block_count - 2]
        product = _product_with_removed_slot(radii, term.multiplicities, None)
        raw += term.combinatorial_coefficient * k_m * product
        coefficients[0] += (
            term.combinatorial_coefficient
            * map_bounds[term.block_count - 1]
            * product
        )
        for input_order in range(1, order):
            multiplicity = term.multiplicities[input_order - 1]
            if multiplicity:
                coefficients[input_order] += (
                    term.combinatorial_coefficient * k_m * multiplicity
                    * _product_with_removed_slot(
                        radii, term.multiplicities, input_order
                    )
                )
    scale = mu**order
    coefficients = [scale * value for value in coefficients]
    output = scale * raw
    beta = coefficients[order]
    return ArbitraryOrderLevelCertificate(
        order=order,
        partition_terms=partitions,
        raw_composite_derivative_upper=raw,
        output_graph_derivative_upper=output,
        graph_derivative_margin=lambda_n - output,
        derivative_bunching_factor_upper=beta,
        derivative_bunching_margin=1 - beta,
        coefficient_by_input_order_upper=tuple(coefficients),
    )


def quantitative_affine_triangular_arbitrary_order_graph_transform(
    *,
    c4_certificate: QuantitativeC4GraphTransformCertificate,
    normalized_graph_derivative_bounds: object,
    normalized_map_derivative_bounds: object,
) -> QuantitativeArbitraryOrderGraphTransformCertificate:
    """Certify every affine triangular derivative level from C4 through C^n."""
    if not isinstance(c4_certificate, QuantitativeC4GraphTransformCertificate):
        raise ValueError("c4_certificate must be an exact C4 certificate")
    if c4_certificate.validation_level is None:
        raise ValueError("arbitrary-order extension requires a verified C4 predecessor")
    if not isinstance(normalized_graph_derivative_bounds, (tuple, list)):
        raise ValueError("normalized_graph_derivative_bounds must be a tuple or list")
    if not isinstance(normalized_map_derivative_bounds, (tuple, list)):
        raise ValueError("normalized_map_derivative_bounds must be a tuple or list")
    graph = tuple(
        _exact_fraction(value, f"normalized_graph_derivative_bounds[{index}]")
        for index, value in enumerate(normalized_graph_derivative_bounds)
    )
    maps = tuple(
        _exact_fraction(value, f"normalized_map_derivative_bounds[{index}]")
        for index, value in enumerate(normalized_map_derivative_bounds)
    )
    maximum_order = len(graph)
    if maximum_order < 4 or len(maps) != maximum_order:
        raise ValueError("graph bounds must cover D1..Dn and map bounds K2..K(n+1), n>=4")
    if min(graph + maps) < 0:
        raise ValueError("all normalized derivative bounds must be nonnegative")

    c3 = c4_certificate.c3_certificate
    c2 = c3.c2_certificate
    lipschitz = c2.c1_certificate.lipschitz_certificate
    expected_graph = (
        lipschitz.normalized_graph_slope_upper,
        c2.normalized_graph_hessian_upper,
        c3.normalized_graph_third_derivative_upper,
        c4_certificate.normalized_graph_fourth_derivative_upper,
    )
    expected_maps = (
        c2.normalized_map_hessian_upper,
        c3.normalized_map_third_derivative_upper,
        c4_certificate.normalized_map_fourth_derivative_upper,
        c4_certificate.normalized_map_fourth_derivative_fiber_lipschitz,
    )
    if graph[:4] != expected_graph:
        raise ValueError("D1..D4 graph bounds must exactly match the C4 predecessor")
    if maps[:4] != expected_maps:
        raise ValueError("K2..K5 map bounds must exactly match the C4 predecessor")
    q = lipschitz.contraction_factor_upper
    mu = lipschitz.base_inverse_lipschitz
    levels = tuple(
        _level(order=order, q=q, mu=mu, graph_bounds=graph, map_bounds=maps)
        for order in range(4, maximum_order + 1)
    )
    c4_level = levels[0]
    expected_c4_coefficients = (
        c4_certificate.value_to_fourth_derivative_cross_coefficient_upper,
        c4_certificate.derivative_to_fourth_derivative_cross_coefficient_upper,
        c4_certificate.hessian_to_fourth_derivative_cross_coefficient_upper,
        c4_certificate.third_to_fourth_derivative_cross_coefficient_upper,
        c4_certificate.fourth_derivative_bunching_factor_upper,
    )
    if (
        c4_level.output_graph_derivative_upper
        != c4_certificate.output_graph_fourth_derivative_upper
        or c4_level.coefficient_by_input_order_upper != expected_c4_coefficients
    ):
        raise ValueError("Bell hierarchy does not exactly reproduce the C4 predecessor")

    failures = list(c4_certificate.failure_codes)
    for level in levels[1:]:
        if level.graph_derivative_margin < 0:
            failures.append(f"C{level.order}_GRAPH_DERIVATIVE_CLASS_NOT_INVARIANT")
        if level.derivative_bunching_margin <= 0:
            failures.append(f"C{level.order}_DERIVATIVE_BUNCHING_NOT_STRICT")
    robust = c4_certificate.robust_interior and all(
        level.graph_derivative_margin > 0 for level in levels[1:]
    )
    if failures:
        return QuantitativeArbitraryOrderGraphTransformCertificate(
            status=failures[0], validation_level=None,
            failure_codes=tuple(failures), robust_interior=False,
            c4_certificate=c4_certificate, maximum_order=maximum_order,
            normalized_graph_derivative_bounds=graph,
            normalized_map_derivative_bounds=maps, levels=levels,
            cn_graph_real_dimension=None,
        )
    status = f"VERIFIED_QUANTITATIVE_AFFINE_TRIANGULAR_C{maximum_order}_GRAPH_TRANSFORM"
    return QuantitativeArbitraryOrderGraphTransformCertificate(
        status=status, validation_level=status, failure_codes=(),
        robust_interior=robust, c4_certificate=c4_certificate,
        maximum_order=maximum_order,
        normalized_graph_derivative_bounds=graph,
        normalized_map_derivative_bounds=maps, levels=levels,
        cn_graph_real_dimension=c4_certificate.c4_graph_real_dimension,
    )


def arbitrary_order_graph_iteration_bound(
    certificate: QuantitativeArbitraryOrderGraphTransformCertificate,
    *, initial_distance_by_derivative_order: object, steps: int,
) -> ArbitraryOrderGraphIterationBound:
    """Iterate all C0..Cn upper-triangular layers from the same previous state."""
    if certificate.validation_level is None:
        raise ValueError("iteration requires a verified arbitrary-order certificate")
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative built-in integer")
    if not isinstance(initial_distance_by_derivative_order, (tuple, list)):
        raise ValueError("initial distances must be a tuple or list")
    values = tuple(
        _exact_fraction(value, f"initial_distance_by_derivative_order[{index}]")
        for index, value in enumerate(initial_distance_by_derivative_order)
    )
    if len(values) != certificate.maximum_order + 1 or min(values) < 0:
        raise ValueError("initial distances must be nonnegative and cover D0..Dn")
    c4 = certificate.c4_certificate
    c3 = c4.c3_certificate
    c2 = c3.c2_certificate
    c1 = c2.c1_certificate
    q = c1.lipschitz_certificate.contraction_factor_upper
    level_by_order = {level.order: level for level in certificate.levels}
    for _ in range(steps):
        old = values
        new = [q * old[0]]
        new.append(c1.derivative_cross_coefficient_upper * old[0] + c1.derivative_bunching_factor_upper * old[1])
        new.append(c2.value_to_hessian_cross_coefficient_upper * old[0] + c2.derivative_to_hessian_cross_coefficient_upper * old[1] + c2.second_derivative_bunching_factor_upper * old[2])
        new.append(c3.value_to_third_derivative_cross_coefficient_upper * old[0] + c3.derivative_to_third_derivative_cross_coefficient_upper * old[1] + c3.hessian_to_third_derivative_cross_coefficient_upper * old[2] + c3.third_derivative_bunching_factor_upper * old[3])
        for order in range(4, certificate.maximum_order + 1):
            coefficients = level_by_order[order].coefficient_by_input_order_upper
            new.append(sum(coefficients[index] * old[index] for index in range(order + 1)))
        values = tuple(new)
    return ArbitraryOrderGraphIterationBound(steps, values)


def positive_part_power(value: object, order: int) -> Fraction:
    """Exact C^(n-1)/non-C^n equality witness h(x)=max(x,0)^n."""
    x = _exact_fraction(value, "value")
    if type(order) is not int or order < 1:
        raise ValueError("order must be a positive built-in integer")
    return max(x, Fraction(0)) ** order


__all__ = [
    "ArbitraryOrderGraphIterationBound",
    "ArbitraryOrderLevelCertificate",
    "BellPartitionTerm",
    "QuantitativeArbitraryOrderGraphTransformCertificate",
    "arbitrary_order_graph_iteration_bound",
    "bell_partition_terms",
    "positive_part_power",
    "quantitative_affine_triangular_arbitrary_order_graph_transform",
]
