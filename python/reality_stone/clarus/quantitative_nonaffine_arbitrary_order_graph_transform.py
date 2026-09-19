"""Exact finite Bell hierarchy for common-inverse nonaffine triangular graphs."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_arbitrary_order_graph_transform import bell_partition_terms
    from .quantitative_graph_transform import _exact_fraction
    from .quantitative_nonaffine_c4_graph_transform import (
        QuantitativeNonaffineC4GraphTransformCertificate,
    )
else:
    from quantitative_arbitrary_order_graph_transform import bell_partition_terms  # type: ignore[no-redef]
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]
    from quantitative_nonaffine_c4_graph_transform import (  # type: ignore[no-redef]
        QuantitativeNonaffineC4GraphTransformCertificate,
    )


@dataclass(frozen=True)
class NonaffineArbitraryOrderLevelCertificate:
    order: int
    inverse_derivative_upper: Fraction
    raw_graph_map_derivative_upper: Fraction
    output_graph_derivative_upper: Fraction
    graph_derivative_margin: Fraction
    derivative_bunching_factor_upper: Fraction
    derivative_bunching_margin: Fraction
    coefficient_by_input_order_upper: tuple[Fraction, ...]


@dataclass(frozen=True)
class QuantitativeNonaffineArbitraryOrderGraphTransformCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    nonaffine_c4_certificate: QuantitativeNonaffineC4GraphTransformCertificate
    maximum_order: int
    normalized_base_map_derivative_bounds: tuple[Fraction, ...]
    normalized_inverse_derivative_bounds: tuple[Fraction, ...]
    normalized_graph_derivative_bounds: tuple[Fraction, ...]
    normalized_map_derivative_bounds: tuple[Fraction, ...]
    levels: tuple[NonaffineArbitraryOrderLevelCertificate, ...]
    cn_graph_real_dimension: int | None


@dataclass(frozen=True)
class NonaffineArbitraryOrderGraphIterationBound:
    steps: int
    distance_by_derivative_order_upper: tuple[Fraction, ...]


def inverse_derivative_bounds_from_forward_map(
    *, inverse_lipschitz_upper: object, normalized_base_map_derivative_bounds: object,
) -> tuple[Fraction, ...]:
    """Generate I1..In from D2phi..Dnphi by D^n(phi compose psi)=0."""
    mu = _exact_fraction(inverse_lipschitz_upper, "inverse_lipschitz_upper")
    if mu <= 0:
        raise ValueError("inverse_lipschitz_upper must be positive")
    if not isinstance(normalized_base_map_derivative_bounds, (tuple, list)):
        raise ValueError("normalized_base_map_derivative_bounds must be a tuple or list")
    forward = tuple(
        _exact_fraction(value, f"normalized_base_map_derivative_bounds[{index}]")
        for index, value in enumerate(normalized_base_map_derivative_bounds)
    )
    if min(forward, default=Fraction(0)) < 0:
        raise ValueError("base-map derivative bounds must be nonnegative")
    inverse = [mu]
    for order in range(2, len(forward) + 2):
        total = Fraction(0)
        singleton = (0,) * (order - 1) + (1,)
        for term in bell_partition_terms(order):
            if term.multiplicities == singleton:
                continue
            product = Fraction(1)
            for slot_order, count in enumerate(term.multiplicities, start=1):
                if count:
                    product *= inverse[slot_order - 1] ** count
            total += (
                term.combinatorial_coefficient
                * forward[term.block_count - 2]
                * product
            )
        inverse.append(mu * total)
    return tuple(inverse)


def _product(bounds: tuple[Fraction, ...], multiplicities: tuple[int, ...]) -> Fraction:
    value = Fraction(1)
    for order, count in enumerate(multiplicities, start=1):
        if count:
            value *= bounds[order - 1] ** count
    return value


def _raw_graph_map_level(
    *, order: int, q: Fraction, graph: tuple[Fraction, ...], maps: tuple[Fraction, ...],
) -> tuple[Fraction, tuple[Fraction, ...]]:
    """Return the D^order S_h size and its graph-difference coefficient vector."""
    if order == 1:
        raise ValueError("order-one raw level requires the anisotropic fiber variations")
    singleton = (0,) * (order - 1) + (1,)
    radii = (1 + graph[0],) + graph[1:order]
    raw = q * graph[order - 1]
    coefficients = [Fraction(0) for _ in range(order + 1)]
    coefficients[order] = q
    coefficients[0] = maps[0] * graph[order - 1]
    for term in bell_partition_terms(order):
        if term.multiplicities == singleton:
            continue
        k_m = maps[term.block_count - 2]
        product = _product(radii, term.multiplicities)
        raw += term.combinatorial_coefficient * k_m * product
        coefficients[0] += (
            term.combinatorial_coefficient * maps[term.block_count - 1] * product
        )
        for input_order in range(1, order):
            count = term.multiplicities[input_order - 1]
            if not count:
                continue
            reduced = list(term.multiplicities)
            reduced[input_order - 1] -= 1
            coefficients[input_order] += (
                term.combinatorial_coefficient * k_m * count
                * _product(radii, tuple(reduced))
            )
    return raw, tuple(coefficients)


def quantitative_nonaffine_triangular_arbitrary_order_graph_transform(
    *, nonaffine_c4_certificate: QuantitativeNonaffineC4GraphTransformCertificate,
    normalized_base_map_derivative_bounds: object,
    normalized_graph_derivative_bounds: object,
    normalized_map_derivative_bounds: object,
) -> QuantitativeNonaffineArbitraryOrderGraphTransformCertificate:
    """Certify every common-inverse nonaffine triangular level C4..Cn."""
    if not isinstance(nonaffine_c4_certificate, QuantitativeNonaffineC4GraphTransformCertificate):
        raise ValueError("nonaffine_c4_certificate must be an exact nonaffine C4 certificate")
    if nonaffine_c4_certificate.validation_level is None:
        raise ValueError("extension requires a verified nonaffine C4 predecessor")
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
        raise ValueError("graph D1..Dn and map K2..K(n+1) bounds are required, n>=4")
    if min(graph + maps) < 0:
        raise ValueError("graph and map derivative bounds must be nonnegative")

    c3 = nonaffine_c4_certificate.nonaffine_c3_certificate
    c2 = c3.nonaffine_c2_certificate
    c1 = c2.c1_certificate
    lip = c1.lipschitz_certificate
    mu = lip.base_inverse_lipschitz
    inverse = inverse_derivative_bounds_from_forward_map(
        inverse_lipschitz_upper=mu,
        normalized_base_map_derivative_bounds=normalized_base_map_derivative_bounds,
    )
    forward = tuple(
        _exact_fraction(value, f"normalized_base_map_derivative_bounds[{index}]")
        for index, value in enumerate(normalized_base_map_derivative_bounds)
    )
    if len(forward) != maximum_order - 1:
        raise ValueError("base-map bounds must cover D2phi..Dnphi")
    expected_graph = (
        lip.normalized_graph_slope_upper, c2.normalized_graph_hessian_upper,
        c3.normalized_graph_third_derivative_upper,
        nonaffine_c4_certificate.normalized_graph_fourth_derivative_upper,
    )
    expected_maps = (
        c2.normalized_map_hessian_upper, c3.normalized_map_third_derivative_upper,
        nonaffine_c4_certificate.normalized_map_fourth_derivative_upper,
        nonaffine_c4_certificate.normalized_map_fourth_derivative_fiber_lipschitz,
    )
    expected_inverse = (
        mu, c2.base_inverse_hessian_upper,
        c3.base_inverse_third_derivative_upper,
        nonaffine_c4_certificate.base_inverse_fourth_derivative_upper,
    )
    if graph[:4] != expected_graph or maps[:4] != expected_maps:
        raise ValueError("graph/map D1..D4/K2..K5 bounds must match the predecessor")
    if inverse[:4] != expected_inverse:
        raise ValueError("forward base-map bounds must reproduce the predecessor inverse D1..D4 bounds")

    q = lip.contraction_factor_upper
    s = q * graph[0] + lip.normalized_base_to_fiber_lipschitz
    a1_delta = (
        c1.normalized_base_derivative_fiber_variation
        + c1.normalized_fiber_derivative_fiber_variation * graph[0]
    )
    raw_sizes: dict[int, Fraction] = {1: s}
    raw_vectors: dict[int, tuple[Fraction, ...]] = {1: (a1_delta, q)}
    for order in range(2, maximum_order + 1):
        raw_sizes[order], raw_vectors[order] = _raw_graph_map_level(
            order=order, q=q, graph=graph, maps=maps
        )

    levels: list[NonaffineArbitraryOrderLevelCertificate] = []
    for order in range(4, maximum_order + 1):
        output = Fraction(0)
        coefficients = [Fraction(0) for _ in range(order + 1)]
        for term in bell_partition_terms(order):
            transport = term.combinatorial_coefficient * _product(inverse, term.multiplicities)
            outer_order = term.block_count
            output += transport * raw_sizes[outer_order]
            raw_vector = raw_vectors[outer_order]
            for input_order, coefficient in enumerate(raw_vector):
                coefficients[input_order] += transport * coefficient
        beta = coefficients[order]
        levels.append(NonaffineArbitraryOrderLevelCertificate(
            order=order, inverse_derivative_upper=inverse[order - 1],
            raw_graph_map_derivative_upper=raw_sizes[order],
            output_graph_derivative_upper=output,
            graph_derivative_margin=graph[order - 1] - output,
            derivative_bunching_factor_upper=beta,
            derivative_bunching_margin=1 - beta,
            coefficient_by_input_order_upper=tuple(coefficients),
        ))

    c4_level = levels[0]
    expected_c4_coefficients = (
        nonaffine_c4_certificate.value_to_fourth_derivative_cross_coefficient_upper,
        nonaffine_c4_certificate.derivative_to_fourth_derivative_cross_coefficient_upper,
        nonaffine_c4_certificate.hessian_to_fourth_derivative_cross_coefficient_upper,
        nonaffine_c4_certificate.third_to_fourth_derivative_cross_coefficient_upper,
        nonaffine_c4_certificate.fourth_derivative_bunching_factor_upper,
    )
    if (
        c4_level.output_graph_derivative_upper
        != nonaffine_c4_certificate.output_graph_fourth_derivative_upper
        or c4_level.coefficient_by_input_order_upper != expected_c4_coefficients
    ):
        raise ValueError("nonaffine Bell hierarchy does not reproduce the C4 predecessor")
    failures = list(nonaffine_c4_certificate.failure_codes)
    for level in levels[1:]:
        if level.graph_derivative_margin < 0:
            failures.append(f"NONAFFINE_C{level.order}_GRAPH_DERIVATIVE_CLASS_NOT_INVARIANT")
        if level.derivative_bunching_margin <= 0:
            failures.append(f"NONAFFINE_C{level.order}_DERIVATIVE_BUNCHING_NOT_STRICT")
    robust = nonaffine_c4_certificate.robust_interior and all(
        level.graph_derivative_margin > 0 for level in levels[1:]
    )
    common = dict(
        nonaffine_c4_certificate=nonaffine_c4_certificate,
        maximum_order=maximum_order,
        normalized_base_map_derivative_bounds=forward,
        normalized_inverse_derivative_bounds=inverse,
        normalized_graph_derivative_bounds=graph,
        normalized_map_derivative_bounds=maps,
        levels=tuple(levels),
    )
    if failures:
        return QuantitativeNonaffineArbitraryOrderGraphTransformCertificate(
            status=failures[0], validation_level=None,
            failure_codes=tuple(failures), robust_interior=False,
            cn_graph_real_dimension=None, **common,
        )
    status = f"VERIFIED_QUANTITATIVE_NONAFFINE_TRIANGULAR_C{maximum_order}_GRAPH_TRANSFORM"
    return QuantitativeNonaffineArbitraryOrderGraphTransformCertificate(
        status=status, validation_level=status, failure_codes=(),
        robust_interior=robust,
        cn_graph_real_dimension=nonaffine_c4_certificate.c4_graph_real_dimension,
        **common,
    )


def nonaffine_arbitrary_order_graph_iteration_bound(
    certificate: QuantitativeNonaffineArbitraryOrderGraphTransformCertificate,
    *, initial_distance_by_derivative_order: object, steps: int,
) -> NonaffineArbitraryOrderGraphIterationBound:
    if certificate.validation_level is None:
        raise ValueError("iteration requires a verified nonaffine arbitrary-order certificate")
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
    c4 = certificate.nonaffine_c4_certificate
    c3 = c4.nonaffine_c3_certificate
    c2 = c3.nonaffine_c2_certificate
    c1 = c2.c1_certificate
    q = c1.lipschitz_certificate.contraction_factor_upper
    levels = {level.order: level for level in certificate.levels}
    for _ in range(steps):
        old = values
        new = [q * old[0]]
        new.append(c1.derivative_cross_coefficient_upper * old[0] + c1.derivative_bunching_factor_upper * old[1])
        new.append(c2.value_to_hessian_cross_coefficient_upper * old[0] + c2.derivative_to_hessian_cross_coefficient_upper * old[1] + c2.second_derivative_bunching_factor_upper * old[2])
        new.append(c3.value_to_third_derivative_cross_coefficient_upper * old[0] + c3.derivative_to_third_derivative_cross_coefficient_upper * old[1] + c3.hessian_to_third_derivative_cross_coefficient_upper * old[2] + c3.third_derivative_bunching_factor_upper * old[3])
        for order in range(4, certificate.maximum_order + 1):
            new.append(sum(
                coefficient * old[index]
                for index, coefficient in enumerate(levels[order].coefficient_by_input_order_upper)
            ))
        values = tuple(new)
    return NonaffineArbitraryOrderGraphIterationBound(steps, values)


__all__ = [
    "NonaffineArbitraryOrderGraphIterationBound",
    "NonaffineArbitraryOrderLevelCertificate",
    "QuantitativeNonaffineArbitraryOrderGraphTransformCertificate",
    "inverse_derivative_bounds_from_forward_map",
    "nonaffine_arbitrary_order_graph_iteration_bound",
    "quantitative_nonaffine_triangular_arbitrary_order_graph_transform",
]
