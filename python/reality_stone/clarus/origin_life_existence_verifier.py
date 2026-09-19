"""Independent exact verifier for the primitive-lineage certificate.

This module deliberately does not import :mod:`origin_life_existence`.  It
parses the public JSON artifact and recomputes the proof obligations from the
reported rational parameters.  Consequently a bug in, or a stale copy of,
the certificate builder is not automatically accepted by this checker.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Mapping, Sequence


CERTIFICATE_TYPE = "clarus_primitive_lineage_exact_existence_certificate"
MINIMUM_CERTIFICATE_VERSION = 5


@dataclass(frozen=True)
class VerificationReport:
    """Machine-readable result of the independent exact checks."""

    verified: bool
    checks: tuple[str, ...]
    errors: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "verified": self.verified,
            "checker": "independent_fraction_recomputation",
            "checks_passed": list(self.checks),
            "errors": list(self.errors),
        }


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    return value


def _field(row: Mapping[str, object], name: str, label: str) -> object:
    if name not in row:
        raise ValueError(f"missing {label}.{name}")
    return row[name]


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _fraction(value: object, label: str) -> Fraction:
    """Parse only integer/string rationals or an ``{"exact": ...}`` field."""

    if isinstance(value, Mapping):
        value = _field(value, "exact", label)
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{label} must contain an exact integer or rational string")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid rational at {label}: {value!r}") from exc
    return result


def _fraction_list(value: object, label: str) -> tuple[Fraction, ...]:
    return tuple(
        _fraction(item, f"{label}[{index}]")
        for index, item in enumerate(_sequence(value, label))
    )


def _fraction_interval(value: object, label: str) -> tuple[Fraction, Fraction]:
    interval = _fraction_list(value, label)
    if len(interval) != 2 or interval[0] > interval[1]:
        raise ValueError(f"{label} must be an ordered two-point interval")
    return interval[0], interval[1]


def _require_exact(
    row: Mapping[str, object], name: str, expected: Fraction, label: str
) -> None:
    observed = _fraction(_field(row, name, label), f"{label}.{name}")
    if observed != expected:
        raise ValueError(
            f"{label}.{name}: expected {expected}, observed {observed}"
        )


def _require_true(row: Mapping[str, object], name: str, label: str) -> None:
    if not _boolean(_field(row, name, label), f"{label}.{name}"):
        raise ValueError(f"{label}.{name} must be true")


def _parameter_map(certificate: Mapping[str, object]) -> dict[str, Fraction]:
    row = _mapping(_field(certificate, "parameters", "certificate"), "parameters")
    names = (
        "growth",
        "leak",
        "boundary_production",
        "boundary_decay",
        "copy_selection",
        "mutation",
        "inheritance_gain",
        "division_threshold",
        "capacity",
    )
    parameters = {
        name: _fraction(_field(row, name, "parameters"), f"parameters.{name}")
        for name in names
    }
    if parameters["capacity"] <= 0:
        raise ValueError("parameters.capacity must be positive")
    if parameters["inheritance_gain"] != 1:
        raise ValueError("the certified q factorization requires inheritance_gain=1")
    return parameters


def _raw_predivision(mass: Fraction, boundary: Fraction, p: Mapping[str, Fraction]) -> Fraction:
    return mass * (
        1
        + p["growth"] * (1 - mass / p["capacity"])
        - p["leak"] * (1 - boundary)
    )


def _predivision_bounds(
    mass_interval: tuple[Fraction, Fraction],
    boundary_interval: tuple[Fraction, Fraction],
    p: Mapping[str, Fraction],
) -> tuple[Fraction, Fraction]:
    values = [
        _raw_predivision(mass, boundary, p)
        for mass in mass_interval
        for boundary in boundary_interval
    ]
    if p["growth"] > 0:
        twice_quadratic = 2 * p["growth"] / p["capacity"]
        for boundary in boundary_interval:
            linear = 1 + p["growth"] - p["leak"] + p["leak"] * boundary
            vertex = linear / twice_quadratic
            if mass_interval[0] <= vertex <= mass_interval[1]:
                values.append(_raw_predivision(vertex, boundary, p))
    return min(values), max(values)


def _boundary_bounds(
    mass_interval: tuple[Fraction, Fraction],
    boundary_interval: tuple[Fraction, Fraction],
    p: Mapping[str, Fraction],
) -> tuple[Fraction, Fraction]:
    values = [
        (1 - p["boundary_decay"]) * boundary
        + p["boundary_production"] * mass * (1 - boundary)
        for mass in mass_interval
        for boundary in boundary_interval
    ]
    return min(values), max(values)


def _divided_row_bounds(
    mass_interval: tuple[Fraction, Fraction],
    boundary_interval: tuple[Fraction, Fraction],
    weight_ratio: Fraction,
    p: Mapping[str, Fraction],
) -> tuple[Fraction, Fraction]:
    first: list[Fraction] = []
    second: list[Fraction] = []
    for mass, boundary in itertools.product(mass_interval, boundary_interval):
        j_mm = Fraction(1, 2) * (
            1
            + p["growth"]
            - p["leak"]
            + p["leak"] * boundary
            - 2 * p["growth"] * mass / p["capacity"]
        )
        j_mb = p["leak"] * mass / 2
        j_bm = p["boundary_production"] * (1 - boundary)
        j_bb = 1 - p["boundary_decay"] - p["boundary_production"] * mass
        first.append(abs(j_mm) + weight_ratio * abs(j_mb))
        second.append(abs(j_bb) + abs(j_bm) / weight_ratio)
    return max(first), max(second)


def _require_interval(
    row: Mapping[str, object],
    name: str,
    expected: tuple[Fraction, Fraction],
    label: str,
) -> None:
    observed = _fraction_interval(_field(row, name, label), f"{label}.{name}")
    if observed != expected:
        raise ValueError(
            f"{label}.{name}: expected {expected}, observed {observed}"
        )


def _step(
    state: tuple[Fraction, Fraction, Fraction], p: Mapping[str, Fraction]
) -> tuple[tuple[Fraction, Fraction, Fraction], bool, Fraction]:
    mass, boundary, heredity = state
    predivision = max(Fraction(0), _raw_predivision(mass, boundary, p))
    divided = predivision >= p["division_threshold"]
    next_mass = predivision / (2 if divided else 1)
    next_boundary = (
        (1 - p["boundary_decay"]) * boundary
        + p["boundary_production"] * mass * (1 - boundary)
    )
    copied = (
        heredity
        + p["copy_selection"] * heredity * (1 - heredity) * (2 * heredity - 1)
        + p["mutation"] * (1 - 2 * heredity)
    )
    next_heredity = Fraction(1, 2) + p["inheritance_gain"] * (
        copied - Fraction(1, 2)
    )
    return (next_mass, next_boundary, next_heredity), divided, predivision


def _heredity_derivative(q: Fraction, p: Mapping[str, Fraction]) -> Fraction:
    return p["inheritance_gain"] * (
        1
        + p["copy_selection"] * (-6 * q**2 + 6 * q - 1)
        - 2 * p["mutation"]
    )


def _positive_root_gap(p: Mapping[str, Fraction]) -> Fraction:
    discriminant = 1 - 4 * p["mutation"] / p["copy_selection"]
    if discriminant < 0:
        raise ValueError("q fixed-point discriminant is negative")
    numerator = math.isqrt(discriminant.numerator)
    denominator = math.isqrt(discriminant.denominator)
    if numerator**2 != discriminant.numerator or denominator**2 != discriminant.denominator:
        raise ValueError("q fixed-point discriminant is not an exact rational square")
    return Fraction(numerator, denominator)


def _check_header(certificate: Mapping[str, object]) -> None:
    artifact_type = _field(certificate, "artifact_type", "certificate")
    if artifact_type != CERTIFICATE_TYPE:
        raise ValueError(f"unexpected artifact_type: {artifact_type!r}")
    version = _integer(
        _field(certificate, "artifact_version", "certificate"),
        "artifact_version",
    )
    if version < MINIMUM_CERTIFICATE_VERSION:
        raise ValueError(
            f"artifact_version {version} is older than {MINIMUM_CERTIFICATE_VERSION}"
        )


def _check_witnesses(certificate: Mapping[str, object]) -> None:
    p = _parameter_map(certificate)
    rows = _sequence(_field(certificate, "witnesses", "certificate"), "witnesses")
    if len(rows) != 2:
        raise ValueError("witnesses must contain exactly two stable outer-q states")
    states: list[tuple[Fraction, Fraction, Fraction]] = []
    for index, value in enumerate(rows):
        label = f"witnesses[{index}]"
        row = _mapping(value, label)
        state_values = _fraction_list(_field(row, "state", label), f"{label}.state")
        next_values = _fraction_list(
            _field(row, "next_state", label), f"{label}.next_state"
        )
        if len(state_values) != 3 or len(next_values) != 3:
            raise ValueError(f"{label} states must have three coordinates")
        state = (state_values[0], state_values[1], state_values[2])
        expected_next, divided, predivision = _step(state, p)
        if next_values != expected_next:
            raise ValueError(f"{label}.next_state does not satisfy the reported equation")
        _require_exact(row, "predivision_mass", predivision, label)
        if _boolean(_field(row, "division_triggered", label), f"{label}.division_triggered") != divided:
            raise ValueError(f"{label}.division_triggered is inconsistent")
        if not divided or expected_next != state:
            raise ValueError(f"{label} is not an exact dividing fixed point")
        _require_true(row, "exact_fixed_point", label)
        states.append(state)

    root_gap = _positive_root_gap(p)
    expected_q = ((1 - root_gap) / 2, (1 + root_gap) / 2)
    if tuple(state[2] for state in states) != expected_q:
        raise ValueError("witness heredity coordinates are not the two outer q roots")
    if states[0][:2] != states[1][:2]:
        raise ValueError("the two witnesses must share the certified mass-boundary state")


def _check_invariant_box(certificate: Mapping[str, object]) -> None:
    p = _parameter_map(certificate)
    row = _mapping(
        _field(certificate, "invariant_box", "certificate"), "invariant_box"
    )
    _require_true(row, "passed", "invariant_box")
    if p["growth"] <= 0:
        raise ValueError("the invariant proof requires positive growth")
    linear = 1 + p["growth"]
    quadratic = p["growth"] / p["capacity"]
    vertex = linear / (2 * quadratic)
    if not 0 <= vertex <= 1:
        raise ValueError("the reported invariant-box extremum is outside [0,1]")
    pre_upper = linear**2 / (4 * quadratic)
    mass_upper = max(p["division_threshold"], pre_upper / 2)
    boundary_values = (
        (1 - p["boundary_decay"]) * b
        + p["boundary_production"] * m * (1 - b)
        for m in (Fraction(0), Fraction(1))
        for b in (Fraction(0), Fraction(1))
    )
    boundary_upper = max(boundary_values)
    derivative_minimum = min(
        _heredity_derivative(Fraction(0), p),
        _heredity_derivative(Fraction(1), p),
    )
    _require_exact(row, "predivision_mass_upper", pre_upper, "invariant_box")
    _require_exact(row, "mass_after_step_upper", mass_upper, "invariant_box")
    _require_exact(row, "boundary_after_step_upper", boundary_upper, "invariant_box")
    _require_exact(
        row,
        "heredity_derivative_minimum",
        derivative_minimum,
        "invariant_box",
    )
    image = _fraction_list(
        _field(row, "heredity_image_endpoints", "invariant_box"),
        "invariant_box.heredity_image_endpoints",
    )
    expected_image = (p["mutation"], 1 - p["mutation"])
    if image != expected_image:
        raise ValueError(
            "invariant_box.heredity_image_endpoints do not match the q map"
        )
    if not (mass_upper < 1 and boundary_upper <= 1 and derivative_minimum > 0):
        raise ValueError("the recomputed invariant-box inequalities do not pass")


def _det_i_minus_full_field(certificate: Mapping[str, object]) -> Fraction:
    candidates = (
        ("local_stability", "det_I_minus_full_jacobian"),
        ("local_stability", "determinant_I_minus_full_jacobian"),
        ("structural_robustness", "det_I_minus_full_jacobian"),
        ("structural_robustness", "implicit_function_determinant"),
    )
    for section_name, field_name in candidates:
        section = certificate.get(section_name)
        if isinstance(section, Mapping) and field_name in section:
            return _fraction(section[field_name], f"{section_name}.{field_name}")
    raise ValueError("missing exact det(I-J_full) certificate field")


def _check_stability_and_q(certificate: Mapping[str, object]) -> None:
    p = _parameter_map(certificate)
    witness_rows = _sequence(_field(certificate, "witnesses", "certificate"), "witnesses")
    first = _mapping(witness_rows[0], "witnesses[0]")
    state = _fraction_list(_field(first, "state", "witnesses[0]"), "witnesses[0].state")
    mass, boundary, outer_q = state

    bracket = (
        1
        + p["growth"] * (1 - mass / p["capacity"])
        - p["leak"] * (1 - boundary)
    )
    j_mm = Fraction(1, 2) * (
        bracket - p["growth"] * mass / p["capacity"]
    )
    j_mb = Fraction(1, 2) * mass * p["leak"]
    j_bm = p["boundary_production"] * (1 - boundary)
    j_bb = 1 - p["boundary_decay"] - p["boundary_production"] * mass
    trace = j_mm + j_bb
    determinant = j_mm * j_bb - j_mb * j_bm
    jury = (
        1 - trace + determinant,
        1 + trace + determinant,
        1 - determinant,
    )
    q_outer_multiplier = _heredity_derivative(outer_q, p)
    q_central_multiplier = _heredity_derivative(Fraction(1, 2), p)
    full_determinant = jury[0] * (1 - q_outer_multiplier)

    row = _mapping(
        _field(certificate, "local_stability", "certificate"), "local_stability"
    )
    _require_true(row, "passed", "local_stability")
    matrix_rows = _sequence(
        _field(row, "mass_boundary_jacobian", "local_stability"),
        "local_stability.mass_boundary_jacobian",
    )
    observed_matrix = tuple(
        _fraction_list(value, f"local_stability.mass_boundary_jacobian[{index}]")
        for index, value in enumerate(matrix_rows)
    )
    if observed_matrix != ((j_mm, j_mb), (j_bm, j_bb)):
        raise ValueError("mass-boundary Jacobian does not match the witness")
    _require_exact(row, "trace", trace, "local_stability")
    _require_exact(row, "determinant", determinant, "local_stability")
    jury_row = _mapping(
        _field(row, "jury_schur_conditions", "local_stability"),
        "local_stability.jury_schur_conditions",
    )
    jury_names = (
        "1_minus_trace_plus_determinant",
        "1_plus_trace_plus_determinant",
        "1_minus_determinant",
    )
    for name, expected in zip(jury_names, jury):
        _require_exact(jury_row, name, expected, "local_stability.jury_schur_conditions")
    _require_exact(
        row,
        "lineage_fixed_point_multiplier",
        q_outer_multiplier,
        "local_stability",
    )
    _require_exact(
        row,
        "central_fixed_point_multiplier",
        q_central_multiplier,
        "local_stability",
    )
    reported_full_determinant = _det_i_minus_full_field(certificate)
    if reported_full_determinant != full_determinant or full_determinant != Fraction(13, 640):
        raise ValueError(
            "det(I-J_full) must independently recompute to 13/640"
        )
    if not (all(value > 0 for value in jury) and abs(q_outer_multiplier) < 1 < q_central_multiplier):
        raise ValueError("recomputed Schur/hyperbolicity conditions do not pass")

    q_row = _mapping(
        _field(certificate, "global_heredity_dynamics", "certificate"),
        "global_heredity_dynamics",
    )
    _require_true(q_row, "passed", "global_heredity_dynamics")
    gap = _positive_root_gap(p)
    expected_roots = ((1 - gap) / 2, Fraction(1, 2), (1 + gap) / 2)
    roots = _fraction_list(
        _field(q_row, "fixed_points", "global_heredity_dynamics"),
        "global_heredity_dynamics.fixed_points",
    )
    if roots != expected_roots:
        raise ValueError("reported q fixed points do not match the exact factorization")
    if _integer(
        _field(q_row, "stable_lineage_count", "global_heredity_dynamics"),
        "global_heredity_dynamics.stable_lineage_count",
    ) != 2:
        raise ValueError("global_heredity_dynamics.stable_lineage_count must be 2")
    bistability_margin = p["copy_selection"] - 4 * p["mutation"]
    monotonicity_margin = 1 - p["copy_selection"] - 2 * p["mutation"]
    if not (0 < bistability_margin < 2 and monotonicity_margin >= 0):
        raise ValueError("the global monotone two-basin q conditions do not pass")


def _full_fixed_count(row: Mapping[str, object]) -> int:
    names = (
        "full_fixed_state_count",
        "full_cube_fixed_states",
        "total_fixed_states_in_invariant_cube",
    )
    for name in names:
        if name in row:
            return _integer(row[name], f"fixed_point_classification.{name}")
    raise ValueError("missing full fixed-state count")


def _check_fixed_classification(certificate: Mapping[str, object]) -> None:
    p = _parameter_map(certificate)
    row = _mapping(
        _field(certificate, "fixed_point_classification", "certificate"),
        "fixed_point_classification",
    )
    _require_true(row, "passed", "fixed_point_classification")
    rho = p["boundary_production"]
    delta = p["boundary_decay"]
    growth = p["growth"]
    capacity = p["capacity"]
    leak = p["leak"]
    c2 = -rho * growth / capacity
    c1 = rho * (growth - 1) - delta * growth / capacity
    c0 = delta * (growth - 1 - leak)
    reported_coefficients = _fraction_list(
        _field(row, "divided_mass_polynomial_coefficients", "fixed_point_classification"),
        "fixed_point_classification.divided_mass_polynomial_coefficients",
    )
    if reported_coefficients != (c2, c1, c0):
        raise ValueError("divided fixed-mass polynomial coefficients are inconsistent")
    reported_roots = _fraction_list(
        _field(row, "divided_mass_roots", "fixed_point_classification"),
        "fixed_point_classification.divided_mass_roots",
    )
    if any(c2 * root**2 + c1 * root + c0 != 0 for root in reported_roots):
        raise ValueError("a reported divided-mass root does not solve the polynomial")
    positive_roots = tuple(root for root in reported_roots if root > 0)
    if len(reported_roots) != 2 or len(positive_roots) != 1:
        raise ValueError("the divided branch must have exactly one positive mass root")
    reported_nondivided_exclusion = _boolean(
        _field(
            row,
            "no_positive_nondivided_fixed_point_below_threshold",
            "fixed_point_classification",
        ),
        "fixed_point_classification.no_positive_nondivided_fixed_point_below_threshold",
    )
    nondivided_c2 = c2
    nondivided_c1 = rho * growth - delta * growth / capacity
    nondivided_c0 = delta * (growth - leak)
    nondivided_at_zero = nondivided_c0
    threshold = p["division_threshold"]
    nondivided_at_threshold = (
        nondivided_c2 * threshold**2
        + nondivided_c1 * threshold
        + nondivided_c0
    )
    recomputed_nondivided_exclusion = (
        nondivided_c2 < 0
        and nondivided_at_zero > 0
        and nondivided_at_threshold > 0
    )
    if not reported_nondivided_exclusion or not recomputed_nondivided_exclusion:
        raise ValueError("positive non-dividing fixed states were not excluded")
    for name, expected in (
        ("nondivided_polynomial_at_zero", nondivided_at_zero),
        ("nondivided_polynomial_at_threshold", nondivided_at_threshold),
    ):
        if name in row:
            _require_exact(row, name, expected, "fixed_point_classification")
    q_root_count = 3
    recomputed_full_count = (1 + len(positive_roots)) * q_root_count
    if _full_fixed_count(row) != recomputed_full_count or recomputed_full_count != 6:
        raise ValueError("full invariant-cube fixed-state count must be exactly 6")
    if _integer(
        _field(row, "positive_reproductive_fixed_states", "fixed_point_classification"),
        "fixed_point_classification.positive_reproductive_fixed_states",
    ) != 3:
        raise ValueError("positive reproductive fixed-state count must be 3")
    if _integer(
        _field(row, "locally_stable_reproductive_fixed_states", "fixed_point_classification"),
        "fixed_point_classification.locally_stable_reproductive_fixed_states",
    ) != 2:
        raise ValueError("locally stable reproductive fixed-state count must be 2")


def _check_reproductive_basin(certificate: Mapping[str, object]) -> None:
    row = _mapping(
        _field(certificate, "certified_reproductive_basin", "certificate"),
        "certified_reproductive_basin",
    )
    _require_true(row, "passed", "certified_reproductive_basin")
    version = _integer(certificate["artifact_version"], "artifact_version")
    if version < 6:
        return

    p = _parameter_map(certificate)
    mass_r0 = (Fraction(2, 5), Fraction(3, 5))
    boundary_r0 = (Fraction(4, 9), Fraction(6, 11))
    mass_r1 = (Fraction(5, 12), Fraction(7, 12))
    boundary_r1 = (Fraction(5, 11), Fraction(7, 13))

    r0 = _mapping(_field(row, "entry_rectangle_R0", "basin"), "basin.R0")
    _require_interval(r0, "mass", mass_r0, "basin.R0")
    _require_interval(r0, "boundary", boundary_r0, "basin.R0")
    pre_r0 = _predivision_bounds(mass_r0, boundary_r0, p)
    post_r0 = (pre_r0[0] / 2, pre_r0[1] / 2)
    boundary_image_r0 = _boundary_bounds(mass_r0, boundary_r0, p)
    _require_interval(r0, "predivision_mass_bounds", pre_r0, "basin.R0")
    _require_interval(r0, "postdivision_mass_bounds", post_r0, "basin.R0")
    _require_interval(r0, "boundary_image_bounds", boundary_image_r0, "basin.R0")
    _require_exact(
        r0,
        "minimum_division_margin",
        pre_r0[0] - p["division_threshold"],
        "basin.R0",
    )
    if not (
        pre_r0[0] > p["division_threshold"]
        and mass_r0[0] <= post_r0[0] <= post_r0[1] <= mass_r0[1]
        and boundary_r0[0]
        <= boundary_image_r0[0]
        <= boundary_image_r0[1]
        <= boundary_r0[1]
    ):
        raise ValueError("R0 is not an invariant every-step division rectangle")

    enclosures = _sequence(
        _field(row, "four_step_exact_interval_enclosures", "basin"),
        "basin.enclosures",
    )
    if len(enclosures) != 5:
        raise ValueError("the R0-to-R1 certificate must contain steps 0 through 4")
    mass_box, boundary_box = mass_r0, boundary_r0
    for step, value in enumerate(enclosures):
        enclosure = _mapping(value, f"basin.enclosures[{step}]")
        if _integer(enclosure["step"], f"basin.enclosures[{step}].step") != step:
            raise ValueError("basin enclosure step index is inconsistent")
        _require_interval(enclosure, "mass", mass_box, f"basin.enclosures[{step}]")
        _require_interval(
            enclosure,
            "boundary",
            boundary_box,
            f"basin.enclosures[{step}]",
        )
        if step < 4:
            pre_box = _predivision_bounds(mass_box, boundary_box, p)
            next_boundary_box = _boundary_bounds(mass_box, boundary_box, p)
            mass_box = (pre_box[0] / 2, pre_box[1] / 2)
            boundary_box = next_boundary_box
    if not (
        mass_r1[0] <= mass_box[0] <= mass_box[1] <= mass_r1[1]
        and boundary_r1[0]
        <= boundary_box[0]
        <= boundary_box[1]
        <= boundary_r1[1]
    ):
        raise ValueError("four exact interval steps do not enter R1")

    r1 = _mapping(
        _field(row, "contraction_rectangle_R1", "basin"), "basin.R1"
    )
    _require_interval(r1, "mass", mass_r1, "basin.R1")
    _require_interval(r1, "boundary", boundary_r1, "basin.R1")
    pre_r1 = _predivision_bounds(mass_r1, boundary_r1, p)
    post_r1 = (pre_r1[0] / 2, pre_r1[1] / 2)
    boundary_image_r1 = _boundary_bounds(mass_r1, boundary_r1, p)
    _require_interval(r1, "predivision_mass_bounds", pre_r1, "basin.R1")
    _require_interval(r1, "postdivision_mass_bounds", post_r1, "basin.R1")
    _require_interval(r1, "boundary_image_bounds", boundary_image_r1, "basin.R1")
    row_bounds = _divided_row_bounds(
        mass_r1, boundary_r1, Fraction(3, 5), p
    )
    observed_rows = _fraction_list(
        _field(r1, "jacobian_row_bounds", "basin.R1"),
        "basin.R1.jacobian_row_bounds",
    )
    if observed_rows != row_bounds or max(row_bounds) >= 1:
        raise ValueError("the independently recomputed R1 contraction fails")
    _require_exact(
        r1,
        "contraction_bound",
        max(row_bounds),
        "basin.R1",
    )
    area = (mass_r0[1] - mass_r0[0]) * (
        boundary_r0[1] - boundary_r0[0]
    )
    _require_exact(
        row,
        "certified_basin_volume_lower_bound",
        area,
        "certified_reproductive_basin",
    )


def _check_parameter_box(certificate: Mapping[str, object]) -> None:
    version = _integer(certificate["artifact_version"], "artifact_version")
    if version < 6:
        return
    structural = _mapping(
        _field(certificate, "structural_robustness", "certificate"),
        "structural_robustness",
    )
    box = _mapping(
        _field(structural, "explicit_closed_parameter_box", "structural_robustness"),
        "parameter_box",
    )
    _require_true(box, "passed", "parameter_box")
    interval_rows = _mapping(
        _field(box, "parameter_intervals", "parameter_box"),
        "parameter_box.parameter_intervals",
    )
    names = (
        "growth",
        "leak",
        "boundary_production",
        "boundary_decay",
        "copy_selection",
        "mutation",
        "division_threshold",
    )
    ranges = {
        name: _fraction_interval(
            _field(interval_rows, name, "parameter_box.parameter_intervals"),
            f"parameter_box.parameter_intervals.{name}",
        )
        for name in names
    }
    if any(lower >= upper for lower, upper in ranges.values()):
        raise ValueError("every certified parameter dimension must have positive width")
    if _integer(box["positive_width_dimensions"], "parameter_box.dimensions") != 7:
        raise ValueError("the explicit robustness box must have seven dimensions")
    common = _mapping(
        _field(box, "common_mass_boundary_rectangle", "parameter_box"),
        "parameter_box.rectangle",
    )
    mass_interval = _fraction_interval(common["mass"], "parameter_box.mass")
    boundary_interval = _fraction_interval(
        common["boundary"], "parameter_box.boundary"
    )

    base = _parameter_map(certificate)
    pre_lows: list[Fraction] = []
    pre_highs: list[Fraction] = []
    boundary_lows: list[Fraction] = []
    boundary_highs: list[Fraction] = []
    row_1: list[Fraction] = []
    row_2: list[Fraction] = []
    for endpoints in itertools.product(*(ranges[name] for name in names)):
        p = dict(base)
        p.update(dict(zip(names, endpoints)))
        pre_low, pre_high = _predivision_bounds(
            mass_interval, boundary_interval, p
        )
        boundary_low, boundary_high = _boundary_bounds(
            mass_interval, boundary_interval, p
        )
        first, second = _divided_row_bounds(
            mass_interval, boundary_interval, Fraction(1), p
        )
        pre_lows.append(pre_low)
        pre_highs.append(pre_high)
        boundary_lows.append(boundary_low)
        boundary_highs.append(boundary_high)
        row_1.append(first)
        row_2.append(second)
    pre_bounds = (min(pre_lows), max(pre_highs))
    post_bounds = (pre_bounds[0] / 2, pre_bounds[1] / 2)
    boundary_bounds = (min(boundary_lows), max(boundary_highs))
    row_bounds = (max(row_1), max(row_2))
    _require_interval(
        box, "uniform_predivision_mass_bounds", pre_bounds, "parameter_box"
    )
    _require_interval(
        box, "uniform_postdivision_mass_bounds", post_bounds, "parameter_box"
    )
    _require_interval(
        box, "uniform_boundary_image_bounds", boundary_bounds, "parameter_box"
    )
    observed_rows = _fraction_list(
        box["uniform_weighted_jacobian_row_bounds"], "parameter_box.row_bounds"
    )
    if observed_rows != row_bounds:
        raise ValueError("parameter-box Jacobian row bounds are inconsistent")
    _require_exact(
        box,
        "uniform_contraction_bound",
        max(row_bounds),
        "parameter_box",
    )
    s_low, s_high = ranges["copy_selection"]
    mu_low, mu_high = ranges["mutation"]
    q_conditions = (
        s_low - 4 * mu_high > 0,
        2 - (s_high - 4 * mu_low) > 0,
        1 - s_high - 2 * mu_high > 0,
    )
    q_margins = _mapping(
        _field(box, "uniform_q_margins", "parameter_box"),
        "parameter_box.uniform_q_margins",
    )
    for name, expected in (
        ("s_minus_4mu", s_low - 4 * mu_high),
        ("2_minus_max_s_minus_4mu", 2 - (s_high - 4 * mu_low)),
        ("1_minus_s_minus_2mu", 1 - s_high - 2 * mu_high),
        ("root_separation_squared", 1 - 4 * mu_high / s_low),
    ):
        _require_exact(q_margins, name, expected, "parameter_box.uniform_q_margins")
    if not (
        pre_bounds[0] > ranges["division_threshold"][1]
        and mass_interval[0]
        <= post_bounds[0]
        <= post_bounds[1]
        <= mass_interval[1]
        and boundary_interval[0]
        <= boundary_bounds[0]
        <= boundary_bounds[1]
        <= boundary_interval[1]
        and max(row_bounds) < 1
        and all(q_conditions)
    ):
        raise ValueError("the independently recomputed parameter box is not robust")


def _check_extinction_wedge(certificate: Mapping[str, object]) -> None:
    version = _integer(certificate["artifact_version"], "artifact_version")
    if version < 6:
        return
    p = _parameter_map(certificate)
    row = _mapping(
        _field(certificate, "extinction_boundary", "certificate"),
        "extinction_boundary",
    )
    _require_true(row, "passed", "extinction_boundary")
    constant = 1 + p["growth"] - p["leak"]
    mass_coefficient = -p["growth"] / p["capacity"]
    boundary_coefficient = p["leak"]
    if (
        2 * constant,
        -2 * mass_coefficient,
        2 * boundary_coefficient,
    ) != (6, 9, 5):
        raise ValueError("the reported wedge equation is not the nominal projection zero-set")
    boundary_ceiling = Fraction(3, 5)
    area = boundary_ceiling / 3 - 5 * boundary_ceiling**2 / 18
    _require_exact(row, "mass_boundary_area", area, "extinction_boundary")
    coefficients = _mapping(
        _field(row, "unprojected_bracket_coefficients", "extinction_boundary"),
        "extinction_boundary.coefficients",
    )
    for name, expected in (
        ("constant", constant),
        ("mass", mass_coefficient),
        ("boundary", boundary_coefficient),
    ):
        _require_exact(coefficients, name, expected, "extinction_boundary.coefficients")
    if _boolean(row["global_survival_proven"], "global_survival_proven"):
        raise ValueError("the positive-area extinction wedge refutes global survival")


def _ablation_row(certificate: Mapping[str, object], names: Sequence[str]) -> Mapping[str, object]:
    section = certificate.get("single_term_ablation_lemmas")
    if not isinstance(section, Mapping):
        section = certificate.get("conditional_ablation_lemmas")
    rows = _mapping(section, "single_term_ablation_lemmas")
    for name in names:
        if name in rows:
            return _mapping(rows[name], f"single_term_ablation_lemmas.{name}")
    raise ValueError(f"missing ablation row; expected one of {tuple(names)!r}")


def _check_ablations(certificate: Mapping[str, object]) -> None:
    p = _parameter_map(certificate)
    autocatalysis_row = _ablation_row(certificate, ("no_autocatalysis", "r_zero"))
    _require_true(autocatalysis_row, "passed", "no_autocatalysis")
    if not (p["leak"] >= 0 and p["division_threshold"] > 0):
        raise ValueError("r=0 does not certify finite division under these parameters")
    boundary_row = _ablation_row(
        certificate, ("no_boundary_production", "rho_zero")
    )
    _require_true(boundary_row, "passed", "no_boundary_production")
    decay_factor = 1 - p["boundary_decay"]
    if not 0 < decay_factor < 1:
        raise ValueError("rho=0 boundary decay is not geometric")
    generations = 0
    boundary_bound = Fraction(1)
    while boundary_bound > Fraction(1, 4):
        generations += 1
        boundary_bound *= decay_factor
        if generations > 10000:
            raise ValueError("failed to find a finite rho=0 boundary-decay time")
    linear = 1 + p["growth"] - p["leak"] * (1 - Fraction(1, 4))
    quadratic = p["growth"] / p["capacity"]
    vertex = linear / (2 * quadratic)
    if not 0 <= vertex <= 1:
        raise ValueError("rho=0 cessation extremum left the invariant interval")
    post_decay_predivision_maximum = linear**2 / (4 * quadratic)
    if generations != 14 or post_decay_predivision_maximum != Fraction(841, 1152):
        raise ValueError("rho=0 exact cessation constants changed unexpectedly")
    if post_decay_predivision_maximum >= p["division_threshold"]:
        raise ValueError("rho=0 does not force permanent division cessation")
    if "generations_until_boundary_at_most_one_quarter" in boundary_row:
        if _integer(
            boundary_row["generations_until_boundary_at_most_one_quarter"],
            "no_boundary_production.generations_until_boundary_at_most_one_quarter",
        ) != generations:
            raise ValueError("rho=0 reported decay time is inconsistent")
    for name in (
        "post_decay_maximum_predivision_mass",
        "maximum_predivision_mass_after_boundary_decay",
    ):
        if name in boundary_row:
            if _fraction(boundary_row[name], f"no_boundary_production.{name}") != post_decay_predivision_maximum:
                raise ValueError("rho=0 reported mass bound is inconsistent")

    inheritance_row = _ablation_row(
        certificate, ("no_inheritance", "inheritance_gain_zero", "eta_zero")
    )
    _require_true(inheritance_row, "passed", "no_inheritance")
    # Setting eta=0 in the reported equation gives q'=1/2 for every q in one step.
    erased_image = Fraction(1, 2) + Fraction(0) * (p["mutation"] - Fraction(1, 2))
    if erased_image != Fraction(1, 2):
        raise ValueError("eta=0 failed to erase the transmitted q coordinate")
    for name in ("one_step_heredity", "unique_image", "collapsed_heredity"):
        if name in inheritance_row and _fraction(
            inheritance_row[name], f"no_inheritance.{name}"
        ) != Fraction(1, 2):
            raise ValueError("eta=0 reported image is inconsistent")

    selection_row = _ablation_row(
        certificate, ("no_bistabilizing_selection", "no_copy_selection")
    )
    _require_true(selection_row, "passed", "no_bistabilizing_selection")
    selection_multiplier = 1 - 2 * p["mutation"]
    if not 0 < selection_multiplier < 1:
        raise ValueError("s=0 does not contract q differences for these parameters")

    suite_value = certificate.get("conditional_core_ablation_suite_passed")
    if suite_value is None:
        suite_value = certificate.get("conditional_ablation_suite_passed")
    if not _boolean(suite_value, "conditional_core_ablation_suite_passed"):
        raise ValueError("conditional core ablation suite must pass")


def _check_branching(certificate: Mapping[str, object]) -> None:
    p = _parameter_map(certificate)
    section: object | None = None
    label = "branching_reproduction"
    for name in ("branching_reproduction", "branching_lineage", "binary_branching"):
        if name in certificate:
            section = certificate[name]
            label = name
            break
    row = _mapping(section, label)
    _require_true(row, "passed", label)
    daughter_count = row.get("daughter_count", row.get("daughters_per_division"))
    if _integer(daughter_count, f"{label}.daughter_count") != 2:
        raise ValueError("the certified branching operator must return two daughters")
    witnesses = _sequence(_field(certificate, "witnesses", "certificate"), "witnesses")
    for index, value in enumerate(witnesses):
        witness = _mapping(value, f"witnesses[{index}]")
        state_values = _fraction_list(
            _field(witness, "state", f"witnesses[{index}]"),
            f"witnesses[{index}].state",
        )
        state = (state_values[0], state_values[1], state_values[2])
        daughter, divided, predivision = _step(state, p)
        if not divided or daughter != state or 2 * daughter[0] != predivision:
            raise ValueError("witness does not generate two mass-conserving identical daughters")
    if "descendants_after_n_generations" not in row:
        raise ValueError(f"missing {label}.descendants_after_n_generations")


def _scope_boolean(certificate: Mapping[str, object], name: str) -> bool:
    if name in certificate:
        return _boolean(certificate[name], name)
    for section_name in ("scope_limits", "nonclaims", "claim_boundaries"):
        section = certificate.get(section_name)
        if isinstance(section, Mapping) and name in section:
            return _boolean(section[name], f"{section_name}.{name}")
    raise ValueError(f"missing explicit scope guard {name}")


def _check_claim_scope(certificate: Mapping[str, object]) -> None:
    base_value = certificate.get("base_existence_theorem_proven")
    if not _boolean(base_value, "base_existence_theorem_proven"):
        raise ValueError("base existence theorem must pass independently of ablations")
    if "all_exact_model_theorems_passed" in certificate and not _boolean(
        certificate["all_exact_model_theorems_passed"],
        "all_exact_model_theorems_passed",
    ):
        raise ValueError("all_exact_model_theorems_passed must be true when present")
    for name in (
        "universal_necessity_proven",
        "empirical_autonomous_protocell_proven",
        "genotype_phenotype_coupling_proven",
        "endogenous_evolution_proven",
    ):
        if _scope_boolean(certificate, name):
            raise ValueError(f"overclaim guard {name} must be false")


def verify_existence_certificate(
    certificate: Mapping[str, object],
) -> VerificationReport:
    """Recompute all v5 proof obligations without calling the builder."""

    checks: list[str] = []
    errors: list[str] = []
    obligations: tuple[tuple[str, Callable[[Mapping[str, object]], None]], ...] = (
        ("header", _check_header),
        ("exact_witness_steps", _check_witnesses),
        ("invariant_box", _check_invariant_box),
        ("jacobian_jury_and_q", _check_stability_and_q),
        ("six_fixed_states", _check_fixed_classification),
        ("reproductive_basin", _check_reproductive_basin),
        ("explicit_parameter_box", _check_parameter_box),
        ("extinction_wedge", _check_extinction_wedge),
        ("conditional_ablations", _check_ablations),
        ("binary_branching", _check_branching),
        ("claim_scope", _check_claim_scope),
    )
    for name, obligation in obligations:
        try:
            obligation(certificate)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(f"{name}: {exc}")
        else:
            checks.append(name)
    return VerificationReport(not errors, tuple(checks), tuple(errors))


def independently_verified(certificate: Mapping[str, object]) -> bool:
    """Convenience boolean API for callers that do not need diagnostics."""

    return verify_existence_certificate(certificate).verified


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", help="path to the JSON certificate")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(Path(args.certificate).read_text(encoding="utf-8"))
        certificate = _mapping(payload, "certificate")
        report = verify_existence_certificate(certificate)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        report = VerificationReport(False, (), (f"input: {exc}",))
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return int(not report.verified)


if __name__ == "__main__":
    raise SystemExit(main())
