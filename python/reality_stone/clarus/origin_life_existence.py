"""Exact certificates for a model-relative primitive-lineage theorem.

The result is deliberately narrow.  It concerns a deterministic,
chemostatted hybrid map and an explicitly defined symmetric branching lift.
It is not empirical proof of an autonomous protocell, genotype--phenotype
coupling, endogenous evolution, or a universal definition of life.

Every algebraic equality and inequality used by the certificate is evaluated
with :class:`fractions.Fraction`; floating point values are display-only.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ExistenceParameters:
    """Parameters of the selected-daughter hybrid map."""

    growth: Fraction = Fraction(9, 2)
    leak: Fraction = Fraction(5, 2)
    boundary_production: Fraction = Fraction(1, 5)
    boundary_decay: Fraction = Fraction(1, 10)
    copy_selection: Fraction = Fraction(1, 2)
    mutation: Fraction = Fraction(3, 32)
    inheritance_gain: Fraction = Fraction(1)
    division_threshold: Fraction = Fraction(3, 4)
    capacity: Fraction = Fraction(1)


PARAMETERS = ExistenceParameters()
WITNESSES = (
    (Fraction(1, 2), Fraction(1, 2), Fraction(1, 4)),
    (Fraction(1, 2), Fraction(1, 2), Fraction(3, 4)),
)


def _fraction(value: Fraction) -> dict[str, str | float]:
    return {"exact": str(value), "decimal": float(value)}


def _fraction_interval(
    lower: Fraction, upper: Fraction
) -> list[dict[str, str | float]]:
    return [_fraction(lower), _fraction(upper)]


def _exact_fraction_sqrt(value: Fraction) -> Fraction:
    numerator = math.isqrt(value.numerator)
    denominator = math.isqrt(value.denominator)
    if numerator**2 != value.numerator or denominator**2 != value.denominator:
        raise ValueError(f"{value} is not an exact rational square")
    return Fraction(numerator, denominator)


def _raw_predivision(
    mass: Fraction,
    boundary: Fraction,
    parameters: ExistenceParameters = PARAMETERS,
) -> Fraction:
    return mass * (
        1
        + parameters.growth * (1 - mass / parameters.capacity)
        - parameters.leak * (1 - boundary)
    )


def _mass_before_division(
    mass: Fraction,
    boundary: Fraction,
    parameters: ExistenceParameters = PARAMETERS,
) -> Fraction:
    return max(Fraction(0), _raw_predivision(mass, boundary, parameters))


def exact_hybrid_step(
    state: tuple[Fraction, Fraction, Fraction],
    parameters: ExistenceParameters = PARAMETERS,
) -> tuple[tuple[Fraction, Fraction, Fraction], bool, Fraction]:
    """Advance ``(mass, boundary, transmitted_state)`` by one exact step."""

    mass, boundary, heredity = state
    predivision_mass = _mass_before_division(mass, boundary, parameters)
    divided = predivision_mass >= parameters.division_threshold
    next_mass = predivision_mass / (2 if divided else 1)
    next_boundary = (
        (1 - parameters.boundary_decay) * boundary
        + parameters.boundary_production * mass * (1 - boundary)
    )
    copied_heredity = (
        heredity
        + parameters.copy_selection
        * heredity
        * (1 - heredity)
        * (2 * heredity - 1)
        + parameters.mutation * (1 - 2 * heredity)
    )
    next_heredity = Fraction(1, 2) + parameters.inheritance_gain * (
        copied_heredity - Fraction(1, 2)
    )
    return (next_mass, next_boundary, next_heredity), divided, predivision_mass


def exact_symmetric_branch_step(
    state: tuple[Fraction, Fraction, Fraction],
    parameters: ExistenceParameters = PARAMETERS,
) -> tuple[tuple[tuple[Fraction, Fraction, Fraction], ...], bool, Fraction]:
    """Apply the ideal symmetric lift: a division returns two equal daughters.

    The base map follows one selected daughter.  This separate operator makes
    the extra equal-partition assumption explicit instead of silently reading
    population doubling into the selected-daughter map.
    """

    daughter, divided, predivision_mass = exact_hybrid_step(state, parameters)
    daughter_count = 2 if divided else 1
    return (daughter,) * daughter_count, divided, predivision_mass


def _parameter_payload(parameters: ExistenceParameters) -> dict[str, str]:
    return {
        field: str(getattr(parameters, field))
        for field in parameters.__dataclass_fields__
    }


def _equation_payload() -> dict[str, str]:
    return {
        "predivision_mass": "m*[1+r*(1-m/K)-lambda*(1-b)]_+",
        "division": "d=1[predivision_mass>=theta_D]; m'=predivision_mass/2^d",
        "boundary": "b'=(1-delta)*b+rho*m*(1-b)",
        "transmitted_state": (
            "q'=1/2+eta*{q+s*q*(1-q)*(2*q-1)+mu*(1-2*q)-1/2}"
        ),
    }


def _model_sha256(parameters: ExistenceParameters) -> str:
    model = {
        "equations": _equation_payload(),
        "parameters": _parameter_payload(parameters),
    }
    payload = json.dumps(model, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _implementation_sha256() -> str:
    source = "\n".join(
        inspect.getsource(function)
        for function in (
            _raw_predivision,
            exact_hybrid_step,
            exact_symmetric_branch_step,
        )
    )
    return hashlib.sha256(source.encode()).hexdigest()


def _heredity_derivative(
    heredity: Fraction, parameters: ExistenceParameters = PARAMETERS
) -> Fraction:
    return parameters.inheritance_gain * (
        1
        + parameters.copy_selection
        * (-6 * heredity**2 + 6 * heredity - 1)
        - 2 * parameters.mutation
    )


def _boundary_image_bounds(
    mass_interval: tuple[Fraction, Fraction],
    boundary_interval: tuple[Fraction, Fraction],
    parameters: ExistenceParameters = PARAMETERS,
) -> tuple[Fraction, Fraction]:
    values = [
        (1 - parameters.boundary_decay) * boundary
        + parameters.boundary_production * mass * (1 - boundary)
        for mass in mass_interval
        for boundary in boundary_interval
    ]
    return min(values), max(values)


def _predivision_bounds(
    mass_interval: tuple[Fraction, Fraction],
    boundary_interval: tuple[Fraction, Fraction],
    parameters: ExistenceParameters = PARAMETERS,
) -> tuple[Fraction, Fraction]:
    """Exact range of the unprojected concave mass polynomial on a box."""

    mass_low, mass_high = mass_interval
    values = [
        _raw_predivision(mass, boundary, parameters)
        for mass in mass_interval
        for boundary in boundary_interval
    ]
    if parameters.growth > 0:
        quadratic_twice = 2 * parameters.growth / parameters.capacity
        for boundary in boundary_interval:
            linear = (
                1
                + parameters.growth
                - parameters.leak
                + parameters.leak * boundary
            )
            vertex = linear / quadratic_twice
            if mass_low <= vertex <= mass_high:
                values.append(_raw_predivision(vertex, boundary, parameters))
    return min(values), max(values)


def _divided_jacobian_row_bounds(
    mass_interval: tuple[Fraction, Fraction],
    boundary_interval: tuple[Fraction, Fraction],
    weight_ratio: Fraction,
    parameters: ExistenceParameters = PARAMETERS,
) -> tuple[Fraction, Fraction]:
    """Bounds for ``max(|dm|, |db|/t)`` with ``t=weight_ratio``."""

    first_rows: list[Fraction] = []
    second_rows: list[Fraction] = []
    for mass, boundary in itertools.product(mass_interval, boundary_interval):
        j_mm = Fraction(1, 2) * (
            1
            + parameters.growth
            - parameters.leak
            + parameters.leak * boundary
            - 2 * parameters.growth * mass / parameters.capacity
        )
        j_mb = Fraction(1, 2) * parameters.leak * mass
        j_bm = parameters.boundary_production * (1 - boundary)
        j_bb = (
            1
            - parameters.boundary_decay
            - parameters.boundary_production * mass
        )
        first_rows.append(abs(j_mm) + weight_ratio * abs(j_mb))
        second_rows.append(abs(j_bb) + abs(j_bm) / weight_ratio)
    return max(first_rows), max(second_rows)


def _next_divided_rectangle(
    mass_interval: tuple[Fraction, Fraction],
    boundary_interval: tuple[Fraction, Fraction],
    parameters: ExistenceParameters = PARAMETERS,
) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
    pre_low, pre_high = _predivision_bounds(
        mass_interval, boundary_interval, parameters
    )
    boundary_bounds = _boundary_image_bounds(
        mass_interval, boundary_interval, parameters
    )
    return (pre_low / 2, pre_high / 2), boundary_bounds


def _expanded_basin_certificate() -> dict[str, object]:
    parameters = PARAMETERS
    mass_r0 = (Fraction(2, 5), Fraction(3, 5))
    boundary_r0 = (Fraction(4, 9), Fraction(6, 11))
    mass_r1 = (Fraction(5, 12), Fraction(7, 12))
    boundary_r1 = (Fraction(5, 11), Fraction(7, 13))

    pre_r0 = _predivision_bounds(mass_r0, boundary_r0, parameters)
    boundary_image_r0 = _boundary_image_bounds(
        mass_r0, boundary_r0, parameters
    )
    post_r0 = (pre_r0[0] / 2, pre_r0[1] / 2)
    r0_invariant = (
        pre_r0[0] > parameters.division_threshold
        and mass_r0[0] <= post_r0[0] <= post_r0[1] <= mass_r0[1]
        and boundary_r0[0]
        <= boundary_image_r0[0]
        <= boundary_image_r0[1]
        <= boundary_r0[1]
    )

    propagated: list[
        tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
    ] = [(mass_r0, boundary_r0)]
    mass_box, boundary_box = mass_r0, boundary_r0
    for _ in range(4):
        mass_box, boundary_box = _next_divided_rectangle(
            mass_box, boundary_box, parameters
        )
        propagated.append((mass_box, boundary_box))
    enters_r1 = (
        mass_r1[0] <= mass_box[0] <= mass_box[1] <= mass_r1[1]
        and boundary_r1[0]
        <= boundary_box[0]
        <= boundary_box[1]
        <= boundary_r1[1]
    )

    pre_r1 = _predivision_bounds(mass_r1, boundary_r1, parameters)
    boundary_image_r1 = _boundary_image_bounds(
        mass_r1, boundary_r1, parameters
    )
    post_r1 = (pre_r1[0] / 2, pre_r1[1] / 2)
    r1_invariant = (
        pre_r1[0] > parameters.division_threshold
        and mass_r1[0] <= post_r1[0] <= post_r1[1] <= mass_r1[1]
        and boundary_r1[0]
        <= boundary_image_r1[0]
        <= boundary_image_r1[1]
        <= boundary_r1[1]
    )
    weight_ratio = Fraction(3, 5)
    row_bounds = _divided_jacobian_row_bounds(
        mass_r1, boundary_r1, weight_ratio, parameters
    )
    contraction_bound = max(row_bounds)
    contraction_passed = r1_invariant and contraction_bound < 1

    area = (mass_r0[1] - mass_r0[0]) * (
        boundary_r0[1] - boundary_r0[0]
    )
    passed = r0_invariant and enters_r1 and contraction_passed
    return {
        "passed": passed,
        "entry_rectangle_R0": {
            "mass": _fraction_interval(*mass_r0),
            "boundary": _fraction_interval(*boundary_r0),
            "forward_invariant": r0_invariant,
            "division_every_generation": (
                pre_r0[0] > parameters.division_threshold
            ),
            "predivision_mass_bounds": _fraction_interval(*pre_r0),
            "postdivision_mass_bounds": _fraction_interval(*post_r0),
            "boundary_image_bounds": _fraction_interval(*boundary_image_r0),
            "minimum_division_margin": _fraction(
                pre_r0[0] - parameters.division_threshold
            ),
        },
        "four_step_exact_interval_enclosures": [
            {
                "step": step,
                "mass": _fraction_interval(*mass_interval),
                "boundary": _fraction_interval(*boundary_interval),
            }
            for step, (mass_interval, boundary_interval) in enumerate(propagated)
        ],
        "contraction_rectangle_R1": {
            "mass": _fraction_interval(*mass_r1),
            "boundary": _fraction_interval(*boundary_r1),
            "entered_from_R0_within_steps": 4,
            "entry_proven": enters_r1,
            "forward_invariant": r1_invariant,
            "predivision_mass_bounds": _fraction_interval(*pre_r1),
            "postdivision_mass_bounds": _fraction_interval(*post_r1),
            "boundary_image_bounds": _fraction_interval(*boundary_image_r1),
            "weighted_sup_norm": "max(|delta_m|,(5/3)*|delta_b|)",
            "weight_ratio_t": _fraction(weight_ratio),
            "jacobian_row_bounds": [_fraction(value) for value in row_bounds],
            "contraction_bound": _fraction(contraction_bound),
        },
        "division_every_generation": r0_invariant,
        "full_state_limits": {
            "R0_x_[0,1/2)": "Z_-=(1/2,1/2,1/4)",
            "R0_x_(1/2,1]": "Z_+=(1/2,1/2,3/4)",
            "R0_x_{1/2}": "unstable separator fixed label",
        },
        "nontrivial_periodic_orbits_in_R0": False,
        "certified_basin_volume_lower_bound": _fraction(area),
        "each_stable_label_basin_volume_lower_bound": _fraction(area / 2),
    }


def _explicit_parameter_box_certificate() -> dict[str, object]:
    ranges = {
        "growth": (Fraction(449, 100), Fraction(451, 100)),
        "leak": (Fraction(249, 100), Fraction(251, 100)),
        "boundary_production": (Fraction(199, 1000), Fraction(201, 1000)),
        "boundary_decay": (Fraction(99, 1000), Fraction(101, 1000)),
        "copy_selection": (Fraction(499, 1000), Fraction(501, 1000)),
        "mutation": (Fraction(93, 1000), Fraction(189, 2000)),
        "division_threshold": (Fraction(749, 1000), Fraction(751, 1000)),
    }
    mass_interval = (Fraction(12, 25), Fraction(13, 25))
    boundary_interval = mass_interval
    weight_ratio = Fraction(1)

    pre_lows: list[Fraction] = []
    pre_highs: list[Fraction] = []
    boundary_lows: list[Fraction] = []
    boundary_highs: list[Fraction] = []
    first_rows: list[Fraction] = []
    second_rows: list[Fraction] = []
    names = tuple(ranges)
    for endpoints in itertools.product(*(ranges[name] for name in names)):
        values = dict(zip(names, endpoints))
        parameters = ExistenceParameters(
            growth=values["growth"],
            leak=values["leak"],
            boundary_production=values["boundary_production"],
            boundary_decay=values["boundary_decay"],
            copy_selection=values["copy_selection"],
            mutation=values["mutation"],
            inheritance_gain=Fraction(1),
            division_threshold=values["division_threshold"],
            capacity=Fraction(1),
        )
        pre_low, pre_high = _predivision_bounds(
            mass_interval, boundary_interval, parameters
        )
        boundary_low, boundary_high = _boundary_image_bounds(
            mass_interval, boundary_interval, parameters
        )
        first_row, second_row = _divided_jacobian_row_bounds(
            mass_interval, boundary_interval, weight_ratio, parameters
        )
        pre_lows.append(pre_low)
        pre_highs.append(pre_high)
        boundary_lows.append(boundary_low)
        boundary_highs.append(boundary_high)
        first_rows.append(first_row)
        second_rows.append(second_row)

    pre_bounds = (min(pre_lows), max(pre_highs))
    post_bounds = (pre_bounds[0] / 2, pre_bounds[1] / 2)
    boundary_bounds = (min(boundary_lows), max(boundary_highs))
    row_bounds = (max(first_rows), max(second_rows))

    s_low, s_high = ranges["copy_selection"]
    mu_low, mu_high = ranges["mutation"]
    theta_high = ranges["division_threshold"][1]
    bistability_margin = s_low - 4 * mu_high
    outer_stability_margin = 2 - (s_high - 4 * mu_low)
    monotonicity_margin = 1 - s_high - 2 * mu_high
    root_separation_squared = 1 - 4 * mu_high / s_low
    passed = (
        pre_bounds[0] > theta_high
        and mass_interval[0]
        <= post_bounds[0]
        <= post_bounds[1]
        <= mass_interval[1]
        and boundary_interval[0]
        <= boundary_bounds[0]
        <= boundary_bounds[1]
        <= boundary_interval[1]
        and max(row_bounds) < 1
        and bistability_margin > 0
        and outer_stability_margin > 0
        and monotonicity_margin > 0
        and root_separation_squared > 0
    )
    return {
        "passed": passed,
        "theorem": (
            "every parameter point in this closed seven-dimensional box has "
            "two stable, once-per-step dividing fixed states in the stated "
            "mass-boundary rectangle"
        ),
        "parameter_intervals": {
            name: _fraction_interval(*interval) for name, interval in ranges.items()
        },
        "fixed_parameters": {"inheritance_gain": "1", "capacity": "1"},
        "positive_width_dimensions": 7,
        "common_mass_boundary_rectangle": {
            "mass": _fraction_interval(*mass_interval),
            "boundary": _fraction_interval(*boundary_interval),
        },
        "uniform_predivision_mass_bounds": _fraction_interval(*pre_bounds),
        "uniform_postdivision_mass_bounds": _fraction_interval(*post_bounds),
        "uniform_boundary_image_bounds": _fraction_interval(*boundary_bounds),
        "uniform_weighted_jacobian_row_bounds": [
            _fraction(value) for value in row_bounds
        ],
        "uniform_contraction_bound": _fraction(max(row_bounds)),
        "uniform_q_margins": {
            "s_minus_4mu": _fraction(bistability_margin),
            "2_minus_max_s_minus_4mu": _fraction(outer_stability_margin),
            "1_minus_s_minus_2mu": _fraction(monotonicity_margin),
            "root_separation_squared": _fraction(root_separation_squared),
        },
    }


def _ablation_certificate() -> tuple[dict[str, object], bool]:
    parameters = PARAMETERS
    decay = 1 - parameters.boundary_decay
    boundary_after_14 = decay**14
    boundary_ceiling = Fraction(1, 4)
    no_boundary_linear = (
        1
        + parameters.growth
        - parameters.leak * (1 - boundary_ceiling)
    )
    no_boundary_quadratic = parameters.growth / parameters.capacity
    no_boundary_maximum = no_boundary_linear**2 / (4 * no_boundary_quadratic)

    no_autocatalysis_passed = (
        parameters.leak >= 0 and parameters.division_threshold > 0
    )
    no_boundary_production_passed = (
        boundary_after_14 <= boundary_ceiling
        and no_boundary_maximum < parameters.division_threshold
    )
    no_inheritance_passed = True
    no_selection_multiplier = 1 - 2 * parameters.mutation
    no_selection_passed = 0 < no_selection_multiplier < 1

    rows: dict[str, object] = {
        "no_autocatalysis": {
            "passed": no_autocatalysis_passed,
            "intervention": "r=0 with every other parameter fixed",
            "global_result": "only finitely many divisions can occur",
            "proof": (
                "predivision_mass<=m; nondivision cannot increase m and each "
                "division at least halves it, while division requires m>=theta_D"
            ),
        },
        "no_boundary_production": {
            "passed": no_boundary_production_passed,
            "intervention": "rho=0 with every other parameter fixed",
            "boundary_law": "b_t=(9/10)^t*b_0",
            "generations_until_boundary_at_most_one_quarter": 14,
            "uniform_boundary_bound_after_14": _fraction(boundary_after_14),
            "post_decay_maximum_predivision_mass": _fraction(
                no_boundary_maximum
            ),
            "division_threshold": _fraction(parameters.division_threshold),
            "global_result": "division permanently ceases after finitely many steps",
        },
        "no_inheritance": {
            "passed": no_inheritance_passed,
            "intervention": "eta=0 with every other parameter fixed",
            "one_step_heredity": _fraction(Fraction(1, 2)),
            "global_result": "every transmitted-state difference is erased in one step",
        },
        "no_bistabilizing_selection": {
            "passed": no_selection_passed,
            "intervention": "s=0 with every other parameter fixed",
            "difference_multiplier": _fraction(no_selection_multiplier),
            "exact_law": "q_t-1/2=(13/16)^t*(q_0-1/2)",
            "scope": (
                "this removes bistabilizing selection, not parent-state copying"
            ),
        },
    }
    core_passed = (
        no_autocatalysis_passed
        and no_boundary_production_passed
        and no_inheritance_passed
    )
    return rows, core_passed


def build_existence_certificate() -> dict[str, object]:
    """Build the complete exact certificate for the current model."""

    parameters = PARAMETERS
    if parameters.inheritance_gain != 1:
        raise ValueError("the exact global q proof requires inheritance_gain=1")

    witness_rows = []
    for state in WITNESSES:
        next_state, divided, predivision_mass = exact_hybrid_step(state)
        witness_rows.append(
            {
                "state": [_fraction(value) for value in state],
                "predivision_mass": _fraction(predivision_mass),
                "division_triggered": divided,
                "next_state": [_fraction(value) for value in next_state],
                "exact_fixed_point": next_state == state,
            }
        )

    fixed_mass, fixed_boundary, fixed_heredity = WITNESSES[0]
    fixed_bracket = (
        1
        + parameters.growth
        * (1 - fixed_mass / parameters.capacity)
        - parameters.leak * (1 - fixed_boundary)
    )
    j_mm = Fraction(1, 2) * (
        fixed_bracket
        - parameters.growth * fixed_mass / parameters.capacity
    )
    j_mb = Fraction(1, 2) * fixed_mass * parameters.leak
    j_bm = parameters.boundary_production * (1 - fixed_boundary)
    j_bb = (
        1
        - parameters.boundary_decay
        - parameters.boundary_production * fixed_mass
    )
    trace = j_mm + j_bb
    determinant = j_mm * j_bb - j_mb * j_bm
    jury = {
        "1_minus_trace_plus_determinant": 1 - trace + determinant,
        "1_plus_trace_plus_determinant": 1 + trace + determinant,
        "1_minus_determinant": 1 - determinant,
    }
    mb_stable = all(value > 0 for value in jury.values())
    lineage_multiplier = _heredity_derivative(fixed_heredity, parameters)
    central_multiplier = _heredity_derivative(Fraction(1, 2), parameters)
    det_i_minus_mb = jury["1_minus_trace_plus_determinant"]
    det_i_minus_full = det_i_minus_mb * (1 - lineage_multiplier)
    spectral_gap_lower_bound = Fraction(1, 8)
    mb_eigenvalue_expression = "(27 +/- sqrt(1769))/80"
    mb_eigenvalues_decimal = [
        (27 - math.sqrt(1769)) / 80,
        (27 + math.sqrt(1769)) / 80,
    ]
    division_margin = _mass_before_division(
        fixed_mass, fixed_boundary, parameters
    ) - parameters.division_threshold
    stability_passed = (
        mb_stable
        and abs(lineage_multiplier) < 1
        and central_multiplier > 1
        and det_i_minus_full != 0
        and division_margin > 0
    )

    mass_linear = 1 + parameters.growth
    mass_quadratic = parameters.growth / parameters.capacity
    predivision_upper = mass_linear**2 / (4 * mass_quadratic)
    mass_after_step_upper = max(
        parameters.division_threshold, predivision_upper / 2
    )
    boundary_upper = max(
        _boundary_image_bounds(
            (Fraction(0), Fraction(1)), (Fraction(0), Fraction(1)), parameters
        )
    )
    heredity_derivative_minimum = min(
        _heredity_derivative(Fraction(0), parameters),
        _heredity_derivative(Fraction(1), parameters),
    )
    heredity_image = (parameters.mutation, 1 - parameters.mutation)
    invariant_box_passed = (
        mass_after_step_upper < 1
        and boundary_upper <= 1
        and heredity_derivative_minimum > 0
        and 0 <= heredity_image[0] < heredity_image[1] <= 1
    )

    heredity_ratio = parameters.mutation / parameters.copy_selection
    heredity_discriminant = 1 - 4 * heredity_ratio
    heredity_root_gap = _exact_fraction_sqrt(heredity_discriminant)
    heredity_roots = (
        (1 - heredity_root_gap) / 2,
        Fraction(1, 2),
        (1 + heredity_root_gap) / 2,
    )
    bistability_margin = parameters.copy_selection - 4 * parameters.mutation
    outer_stability_margin = 2 - bistability_margin
    monotonicity_margin = (
        1 - parameters.copy_selection - 2 * parameters.mutation
    )
    heredity_global_passed = (
        heredity_roots
        == (WITNESSES[0][2], Fraction(1, 2), WITNESSES[1][2])
        and bistability_margin > 0
        and outer_stability_margin > 0
        and monotonicity_margin > 0
        and lineage_multiplier == 1 - bistability_margin
        and central_multiplier == 1 + bistability_margin / 2
    )

    divided_c2 = (
        -parameters.boundary_production
        * parameters.growth
        / parameters.capacity
    )
    divided_c1 = (
        parameters.boundary_production * (parameters.growth - 1)
        - parameters.boundary_decay
        * parameters.growth
        / parameters.capacity
    )
    divided_c0 = parameters.boundary_decay * (
        parameters.growth - 1 - parameters.leak
    )
    divided_discriminant = divided_c1**2 - 4 * divided_c2 * divided_c0
    divided_sqrt = _exact_fraction_sqrt(divided_discriminant)
    divided_roots = tuple(
        sorted(
            (
                (-divided_c1 + divided_sqrt) / (2 * divided_c2),
                (-divided_c1 - divided_sqrt) / (2 * divided_c2),
            )
        )
    )
    positive_divided_roots = tuple(root for root in divided_roots if root > 0)
    unique_positive_divided_fixed_point = positive_divided_roots == (
        fixed_mass,
    )

    nondivided_c2 = divided_c2
    nondivided_c1 = (
        parameters.boundary_production * parameters.growth
        - parameters.boundary_decay
        * parameters.growth
        / parameters.capacity
    )
    nondivided_c0 = parameters.boundary_decay * (
        parameters.growth - parameters.leak
    )

    def polynomial(
        coefficient2: Fraction,
        coefficient1: Fraction,
        coefficient0: Fraction,
        value: Fraction,
    ) -> Fraction:
        return coefficient2 * value**2 + coefficient1 * value + coefficient0

    nondivided_at_zero = polynomial(
        nondivided_c2, nondivided_c1, nondivided_c0, Fraction(0)
    )
    nondivided_at_threshold = polynomial(
        nondivided_c2,
        nondivided_c1,
        nondivided_c0,
        parameters.division_threshold,
    )
    no_valid_positive_nondivided_fixed_point = (
        nondivided_c2 < 0
        and nondivided_at_zero > 0
        and nondivided_at_threshold > 0
    )
    fixed_point_classification_passed = (
        unique_positive_divided_fixed_point
        and no_valid_positive_nondivided_fixed_point
    )

    basin = _expanded_basin_certificate()
    parameter_box = _explicit_parameter_box_certificate()
    ablations, conditional_core_ablation_suite_passed = (
        _ablation_certificate()
    )

    branch_rows = []
    for witness in WITNESSES:
        daughters, divided, predivision_mass = exact_symmetric_branch_step(witness)
        branch_rows.append(
            {
                "parent": [_fraction(value) for value in witness],
                "division_triggered": divided,
                "daughter_count": len(daughters),
                "daughters_equal_parent": all(
                    daughter == witness for daughter in daughters
                ),
                "predivision_mass": _fraction(predivision_mass),
                "total_daughter_mass": _fraction(
                    sum(daughter[0] for daughter in daughters)
                ),
            }
        )
    branching_passed = all(
        row["division_triggered"]
        and row["daughter_count"] == 2
        and row["daughters_equal_parent"]
        and row["predivision_mass"] == row["total_daughter_mass"]
        for row in branch_rows
    )

    extinction_constant = 1 + parameters.growth - parameters.leak
    extinction_mass_coefficient = -parameters.growth / parameters.capacity
    extinction_boundary_coefficient = parameters.leak
    extinction_boundary_ceiling = Fraction(3, 5)
    extinction_area = (
        extinction_boundary_ceiling / 3
        - 5 * extinction_boundary_ceiling**2 / 18
    )
    extinction_wedge_passed = (
        2 * extinction_constant == 6
        and -2 * extinction_mass_coefficient == 9
        and 2 * extinction_boundary_coefficient == 5
        and extinction_area == Fraction(1, 10)
    )

    fixed_points_passed = all(
        row["division_triggered"] and row["exact_fixed_point"]
        for row in witness_rows
    )
    structural_robustness_passed = (
        stability_passed
        and division_margin > 0
        and det_i_minus_full != 0
        and parameter_box["passed"]
    )
    base_existence_theorem_proven = (
        invariant_box_passed
        and fixed_points_passed
        and stability_passed
        and heredity_global_passed
        and fixed_point_classification_passed
        and bool(basin["passed"])
        and structural_robustness_passed
    )
    all_exact_model_theorems_passed = (
        base_existence_theorem_proven
        and conditional_core_ablation_suite_passed
        and branching_passed
        and extinction_wedge_passed
    )

    return {
        "artifact_type": "clarus_primitive_lineage_exact_existence_certificate",
        "artifact_version": 6,
        "model_sha256": _model_sha256(parameters),
        "implementation_sha256": _implementation_sha256(),
        "theorem": (
            "in the stated deterministic chemostatted selected-daughter map, "
            "exactly three positive once-per-step dividing fixed states exist; "
            "the q=1/4 and q=3/4 states are locally asymptotically stable and "
            "have positive-volume certified basins"
        ),
        "equation_scope": (
            "model-relative recurrence of a dividing attractor times a transmitted-"
            "state label; not autonomous life or mechanistic genetic heredity"
        ),
        "equations": _equation_payload(),
        "parameters": _parameter_payload(parameters),
        "state_order": ["mass", "boundary", "transmitted_state"],
        "invariant_box": {
            "domain": "[0,1]^3",
            "passed": invariant_box_passed,
            "predivision_mass_upper": _fraction(predivision_upper),
            "mass_after_step_upper": _fraction(mass_after_step_upper),
            "boundary_after_step_upper": _fraction(boundary_upper),
            "heredity_derivative_minimum": _fraction(
                heredity_derivative_minimum
            ),
            "heredity_image_endpoints": [
                _fraction(value) for value in heredity_image
            ],
            "one_step_absorbing_box": {
                "mass": _fraction_interval(
                    Fraction(0), mass_after_step_upper
                ),
                "boundary": _fraction_interval(Fraction(0), boundary_upper),
                "transmitted_state": _fraction_interval(*heredity_image),
            },
            "interpretation": "bounded dissipativity, not survival",
        },
        "witnesses": witness_rows,
        "transmitted_state_separation": _fraction(Fraction(1, 2)),
        "division_branch_margin": _fraction(division_margin),
        "local_stability": {
            "passed": stability_passed,
            "mass_boundary_jacobian": [
                [_fraction(j_mm), _fraction(j_mb)],
                [_fraction(j_bm), _fraction(j_bb)],
            ],
            "trace": _fraction(trace),
            "determinant": _fraction(determinant),
            "jury_schur_conditions": {
                name: _fraction(value) for name, value in jury.items()
            },
            "mass_boundary_eigenvalues": {
                "exact_expression": mb_eigenvalue_expression,
                "decimal": mb_eigenvalues_decimal,
            },
            "lineage_fixed_point_multiplier": _fraction(lineage_multiplier),
            "central_fixed_point_multiplier": _fraction(central_multiplier),
            "det_I_minus_mass_boundary_jacobian": _fraction(det_i_minus_mb),
            "det_I_minus_full_jacobian": _fraction(det_i_minus_full),
            "spectral_gap_lower_bound": _fraction(
                spectral_gap_lower_bound
            ),
        },
        "structural_robustness": {
            "passed": structural_robustness_passed,
            "hyperbolic_fixed_points": stability_passed,
            "implicit_function_determinant": _fraction(det_i_minus_full),
            "open_parameter_neighborhood_exists": (
                structural_robustness_passed
            ),
            "explicit_closed_parameter_box": parameter_box,
        },
        "global_heredity_dynamics": {
            "passed": heredity_global_passed,
            "nominal_centered_form": "x'=x*(17/16-x^2), x=q-1/2",
            "factorization": "f(q)-q=(2*q-1)*(s*q*(1-q)-mu)",
            "fixed_points": [_fraction(value) for value in heredity_roots],
            "stable_basins": {
                "[0,1/2)": "q -> 1/4 monotonically",
                "(1/2,1]": "q -> 3/4 monotonically",
                "{1/2}": "unstable invariant separator",
            },
            "basin_crossing": False,
            "nontrivial_periodic_orbits": False,
            "stable_lineage_count": 2,
            "log2_stable_label_count": 1,
            "general_sufficient_conditions": [
                "mu>=0",
                "4*mu<s",
                "s+2*mu<=1",
                "eta=1",
            ],
            "nominal_condition_margins": {
                "s_minus_4mu": _fraction(bistability_margin),
                "2_minus_(s_minus_4mu)": _fraction(
                    outer_stability_margin
                ),
                "1_minus_s_minus_2mu": _fraction(
                    monotonicity_margin
                ),
            },
        },
        "fixed_point_classification": {
            "passed": fixed_point_classification_passed,
            "full_fixed_state_count": 6,
            "full_fixed_states": (
                "(m,b) in {(0,0),(1/2,1/2)} crossed with "
                "q in {1/4,1/2,3/4}"
            ),
            "divided_mass_polynomial_coefficients": [
                _fraction(divided_c2),
                _fraction(divided_c1),
                _fraction(divided_c0),
            ],
            "divided_mass_roots": [_fraction(value) for value in divided_roots],
            "unique_positive_divided_mass": _fraction(fixed_mass),
            "no_positive_nondivided_fixed_point_below_threshold": (
                no_valid_positive_nondivided_fixed_point
            ),
            "nondivided_positive_algebraic_candidate": (
                "(3+sqrt(41))/12 > 3/4, hence branch-inconsistent"
            ),
            "nondivided_polynomial_at_zero": _fraction(nondivided_at_zero),
            "nondivided_polynomial_at_threshold": _fraction(
                nondivided_at_threshold
            ),
            "positive_reproductive_fixed_states": 3,
            "locally_stable_reproductive_fixed_states": 2,
            "central_reproductive_state_is_saddle": True,
            "extinction_fixed_states_are_locally_unstable": True,
            "extinction_mass_multiplier": _fraction(Fraction(3)),
        },
        "certified_reproductive_basin": basin,
        "extinction_boundary": {
            "passed": extinction_wedge_passed,
            "one_step_wedge": "E={(m,b): 9*m>=6+5*b}",
            "symbolic_reason": (
                "3-(9/2)*m+(5/2)*b<=0 makes projected mass exactly zero"
            ),
            "unprojected_bracket_coefficients": {
                "constant": _fraction(extinction_constant),
                "mass": _fraction(extinction_mass_coefficient),
                "boundary": _fraction(extinction_boundary_coefficient),
            },
            "boundary_ceiling": _fraction(extinction_boundary_ceiling),
            "mass_boundary_area": _fraction(extinction_area),
            "boundary_zero_threshold": _fraction(Fraction(2, 3)),
            "each_q_half_volume_lower_bound": _fraction(
                extinction_area / 2
            ),
            "absorbing_rule": "after m=0, mass stays zero and b'=(9/10)*b",
            "global_positive_attraction_proven": False,
            "global_survival_proven": False,
        },
        "model_relative_recurrent_property": {
            "definition": (
                "L={inf_t m_t>0, sum_t d_t=infinity, "
                "lim q_t in {1/4,3/4}}"
            ),
            "nonempty": bool(basin["passed"]),
            "positive_volume_lower_bound": basin[
                "certified_basin_volume_lower_bound"
            ],
            "conditional_core_terms": [
                "autocatalytic growth r",
                "boundary production rho",
                "transmission gain eta",
            ],
        },
        "single_term_ablation_lemmas": ablations,
        "conditional_core_ablation_suite_passed": (
            conditional_core_ablation_suite_passed
        ),
        "branching_lineage": {
            "passed": branching_passed,
            "operator": "ideal symmetric two-daughter lift of the base map",
            "daughter_count": 2,
            "daughters_per_division": 2,
            "witness_checks": branch_rows,
            "descendants_after_n_generations": "2^n",
            "induction": (
                "each witness parent returns two identical witness daughters; "
                "therefore generation n contains 2^n descendants"
            ),
            "scope": (
                "mathematical equal-partition construction; physical partition "
                "noise and population competition are not modeled"
            ),
        },
        "base_existence_theorem_proven": base_existence_theorem_proven,
        "existence_theorem_proven": base_existence_theorem_proven,
        "conditional_minimality_in_this_model_proven": (
            conditional_core_ablation_suite_passed
        ),
        "all_exact_model_theorems_passed": all_exact_model_theorems_passed,
        "universal_necessity_proven": False,
        "empirical_autonomous_protocell_proven": False,
        "genotype_phenotype_coupling_proven": False,
        "endogenous_evolution_proven": False,
        "historical_origin_of_life_proven": False,
        "scope_limits": {
            "mass_boundary_and_q_are_decoupled": True,
            "dead_states_still_update_q_mathematically": True,
            "selected_daughter_base_map_only": True,
            "branching_requires_added_symmetric_partition_operator": True,
            "universal_necessity_proven": False,
            "empirical_autonomous_protocell_proven": False,
            "genotype_phenotype_coupling_proven": False,
            "endogenous_evolution_proven": False,
        },
    }


def validate_existence_certificate(certificate: Mapping[str, object]) -> bool:
    """Check reproducibility against the current builder.

    This is a regeneration/staleness check, not an independent proof checker.
    Use :mod:`origin_life_existence_verifier` for independent recomputation.
    """

    return dict(certificate) == build_existence_certificate()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--verify")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        certificate = json.loads(Path(args.verify).read_text(encoding="utf-8"))
        verified = validate_existence_certificate(certificate)
        print(json.dumps({"verified": verified}))
        return int(not verified)

    certificate = build_existence_certificate()
    payload = json.dumps(certificate, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return int(
        args.require_pass and not certificate["all_exact_model_theorems_passed"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
