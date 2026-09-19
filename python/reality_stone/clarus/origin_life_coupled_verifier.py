"""Independent exact verifier for the coupled heredity-selection certificate.

The checker deliberately does not import the certificate builder.  It parses
the artifact fail-closed and recomputes every numerical proof obligation with
``Fraction`` under a separately encoded canonical model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class CoupledVerificationReport:
    verified: bool
    checks: tuple[str, ...]
    errors: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "verified": self.verified,
            "checks": list(self.checks),
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


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _fraction(value: object, label: str) -> Fraction:
    if isinstance(value, Mapping):
        row = _mapping(value, label)
        if "exact" not in row:
            raise ValueError(f"{label}.exact is required")
        exact = row["exact"]
        if not isinstance(exact, (str, int)) or isinstance(exact, bool):
            raise ValueError(f"{label}.exact must be a rational string or integer")
        result = Fraction(exact)
        if "decimal" in row:
            decimal = row["decimal"]
            if not isinstance(decimal, (int, float)) or isinstance(decimal, bool):
                raise ValueError(f"{label}.decimal must be numeric")
            if abs(float(result) - float(decimal)) > 1e-12:
                raise ValueError(f"{label}.decimal disagrees with exact value")
        return result
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ValueError(f"{label} must be an exact rational, not a float")
    return Fraction(value)


def _require_equal(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label} is {observed!r}; expected {expected!r}")


def _require_fraction(
    row: Mapping[str, object],
    field: str,
    expected: Fraction,
    label: str,
) -> None:
    if field not in row:
        raise ValueError(f"missing {label}.{field}")
    observed = _fraction(row[field], f"{label}.{field}")
    if observed != expected:
        raise ValueError(
            f"{label}.{field} is {observed}; expected exact value {expected}"
        )


def _require_passed(row: Mapping[str, object], label: str) -> None:
    if "passed" not in row or not _boolean(row["passed"], f"{label}.passed"):
        raise ValueError(f"{label}.passed must be true")


def _proof_section(
    certificate: Mapping[str, object], name: str
) -> Mapping[str, object]:
    obligations = _mapping(certificate.get("proof_obligations"), "proof_obligations")
    if name not in obligations:
        raise ValueError(f"missing proof_obligations.{name}")
    return _mapping(obligations[name], f"proof_obligations.{name}")


def _canonical_parameters(certificate: Mapping[str, object]) -> dict[str, Fraction | int]:
    model = _mapping(certificate.get("model"), "model")
    parameters = _mapping(model.get("parameters"), "model.parameters")
    expected: dict[str, Fraction | int] = {
        "growth_intercept": Fraction(3, 16),
        "growth_slope": Fraction(1, 2),
        "division_threshold": Fraction(1),
        "daughter_reset": Fraction(1, 2),
        "slow_type": Fraction(1, 4),
        "fast_type": Fraction(3, 4),
        "forward_mutation_probability": Fraction(1, 16),
        "essential_module_count": 7,
    }
    if set(parameters) != set(expected):
        raise ValueError("model.parameters must contain exactly the canonical fields")
    for name, expected_value in expected.items():
        value = parameters[name]
        if name == "essential_module_count":
            observed: Fraction | int = _integer(value, f"model.parameters.{name}")
        else:
            observed = _fraction(value, f"model.parameters.{name}")
        if observed != expected_value:
            raise ValueError(
                f"model.parameters.{name} is {observed}; expected {expected_value}"
            )
    return expected


def _check_header(certificate: Mapping[str, object]) -> None:
    _require_equal(
        certificate.get("artifact_type"),
        "clarus_coupled_heredity_selection_exact_certificate",
        "artifact_type",
    )
    _require_equal(
        _integer(certificate.get("artifact_version"), "artifact_version"),
        1,
        "artifact_version",
    )
    _require_equal(
        certificate.get("arithmetic"),
        "fractions.Fraction exact rational arithmetic",
        "arithmetic",
    )
    model = _mapping(certificate.get("model"), "model")
    _require_equal(model.get("state"), "(cell_cycle_phase, transmitted_type)", "state")
    semantics = _mapping(model.get("semantics"), "model.semantics")
    expected_semantics = {
        "division_comparator": ">=",
        "update_order": "advance_phase_then_test_then_reset",
        "tick_rule": "each cell present at tick start updates exactly once",
        "daughter_activation": "daughters first update on the next tick",
        "parent_fate": "division replaces parent by exactly two daughters",
        "phase_interpretation": "cell-cycle accumulator, not conserved mass",
        "founder_phase": "all certified founders start at daughter_reset",
        "resource_regime": "unlimited and noninteracting",
    }
    if dict(semantics) != expected_semantics:
        raise ValueError("model.semantics changed the certified transition meaning")
    equations = _mapping(model.get("equations"), "model.equations")
    expected_equations = {
        "phenotype": "g(q)=3/16+q/2",
        "clock": "a_pre=a+g(q)",
        "nondivision": "a_pre<1 => one successor (a_pre,q)",
        "division": "a_pre>=1 => two daughters (1/2,q)",
        "mutation_kernel": (
            "at slow-type division only, each daughter changes slow->fast "
            "with probability nu; no back mutation"
        ),
    }
    if dict(equations) != expected_equations:
        raise ValueError("model.equations changed the canonical map")
    _canonical_parameters(certificate)


def _increments(parameters: Mapping[str, Fraction | int]) -> tuple[Fraction, Fraction]:
    intercept = Fraction(parameters["growth_intercept"])
    slope = Fraction(parameters["growth_slope"])
    slow = intercept + slope * Fraction(parameters["slow_type"])
    fast = intercept + slope * Fraction(parameters["fast_type"])
    return slow, fast


def _check_coupling(certificate: Mapping[str, object]) -> None:
    p = _canonical_parameters(certificate)
    slow, fast = _increments(p)
    row = _proof_section(certificate, "phenotype_coupling")
    _require_passed(row, "phenotype_coupling")
    _require_equal(
        row.get("equation"),
        "g(q)=growth_intercept+growth_slope*q",
        "phenotype_coupling.equation",
    )
    _require_fraction(row, "dg_dq", Fraction(1, 2), "phenotype_coupling")
    _require_fraction(row, "slow_increment", slow, "phenotype_coupling")
    _require_fraction(row, "fast_increment", fast, "phenotype_coupling")
    _require_fraction(
        row, "increment_difference", fast - slow, "phenotype_coupling"
    )
    if not (slow == Fraction(5, 16) < fast == Fraction(9, 16)):
        raise ValueError("canonical type-to-phenotype coupling is not distinct")


def _exact_trace(
    transmitted_type: Fraction,
    ticks: int,
    parameters: Mapping[str, Fraction | int],
) -> list[dict[str, object]]:
    phase = Fraction(parameters["daughter_reset"])
    threshold = Fraction(parameters["division_threshold"])
    intercept = Fraction(parameters["growth_intercept"])
    slope = Fraction(parameters["growth_slope"])
    increment = intercept + slope * transmitted_type
    descendants = 1
    rows: list[dict[str, object]] = []
    for tick in range(1, ticks + 1):
        phase_before = phase
        predivision = phase + increment
        divided = predivision >= threshold
        daughter_count = 2 if divided else 1
        phase = Fraction(parameters["daughter_reset"]) if divided else predivision
        descendants *= daughter_count
        rows.append(
            {
                "tick": tick,
                "phase_before": phase_before,
                "predivision_phase": predivision,
                "divided": divided,
                "daughter_count_per_parent": daughter_count,
                "representative_phase_after": phase,
                "transmitted_type_after": transmitted_type,
                "total_descendants": descendants,
            }
        )
    return rows


def _compare_trace(
    observed_value: object,
    expected: list[dict[str, object]],
    label: str,
) -> None:
    observed = _sequence(observed_value, label)
    if len(observed) != len(expected):
        raise ValueError(f"{label} has the wrong number of ticks")
    fraction_fields = (
        "phase_before",
        "predivision_phase",
        "representative_phase_after",
        "transmitted_type_after",
    )
    integer_fields = ("tick", "daughter_count_per_parent", "total_descendants")
    for index, expected_row in enumerate(expected):
        row = _mapping(observed[index], f"{label}[{index}]")
        if set(row) != set(expected_row):
            raise ValueError(f"{label}[{index}] fields changed")
        for field in fraction_fields:
            value = _fraction(row[field], f"{label}[{index}].{field}")
            if value != expected_row[field]:
                raise ValueError(f"{label}[{index}].{field} is inconsistent")
        for field in integer_fields:
            value = _integer(row[field], f"{label}[{index}].{field}")
            if value != expected_row[field]:
                raise ValueError(f"{label}[{index}].{field} is inconsistent")
        if _boolean(row["divided"], f"{label}[{index}].divided") != expected_row[
            "divided"
        ]:
            raise ValueError(f"{label}[{index}].divided is inconsistent")


def _check_transmission(certificate: Mapping[str, object]) -> None:
    p = _canonical_parameters(certificate)
    row = _proof_section(certificate, "division_gated_transmission")
    _require_passed(row, "division_gated_transmission")
    _require_equal(
        _integer(row.get("nondivision_output_count"), "nondivision_output_count"),
        1,
        "nondivision_output_count",
    )
    _require_equal(
        _integer(row.get("division_output_count"), "division_output_count"),
        2,
        "division_output_count",
    )
    _require_equal(
        row.get("copy_event_condition"),
        "predivision_phase>=division_threshold",
        "copy_event_condition",
    )
    fast = Fraction(p["fast_type"])
    slow = Fraction(p["slow_type"])
    _compare_trace(row.get("fast_trace"), _exact_trace(fast, 4, p), "fast_trace")
    _compare_trace(row.get("slow_trace"), _exact_trace(slow, 4, p), "slow_trace")


def _interval(value: object, label: str) -> tuple[Fraction, Fraction]:
    values = _sequence(value, label)
    if len(values) != 2:
        raise ValueError(f"{label} must contain two endpoints")
    return _fraction(values[0], f"{label}[0]"), _fraction(values[1], f"{label}[1]")


def _check_robustness(certificate: Mapping[str, object]) -> None:
    row = _proof_section(certificate, "open_parameter_plateau")
    _require_passed(row, "open_parameter_plateau")
    intercept = _interval(
        row.get("growth_intercept_interval"), "growth_intercept_interval"
    )
    slope = _interval(row.get("growth_slope_interval"), "growth_slope_interval")
    if intercept != (Fraction(11, 64), Fraction(13, 64)):
        raise ValueError("growth-intercept robustness interval changed")
    if slope != (Fraction(15, 32), Fraction(17, 32)):
        raise ValueError("growth-slope robustness interval changed")
    slow = (
        intercept[0] + slope[0] * Fraction(1, 4),
        intercept[1] + slope[1] * Fraction(1, 4),
    )
    fast = (
        intercept[0] + slope[0] * Fraction(3, 4),
        intercept[1] + slope[1] * Fraction(3, 4),
    )
    if _interval(row.get("slow_increment_bounds"), "slow_increment_bounds") != slow:
        raise ValueError("slow increment bounds are inconsistent")
    if _interval(row.get("fast_increment_bounds"), "fast_increment_bounds") != fast:
        raise ValueError("fast increment bounds are inconsistent")
    _require_fraction(
        row,
        "slow_two_tick_lower_margin",
        slow[0] - Fraction(1, 4),
        "open_parameter_plateau",
    )
    _require_fraction(
        row,
        "slow_one_tick_upper_margin",
        Fraction(1, 2) - slow[1],
        "open_parameter_plateau",
    )
    _require_fraction(
        row,
        "fast_one_tick_lower_margin",
        fast[0] - Fraction(1, 2),
        "open_parameter_plateau",
    )
    if not (slow[0] > Fraction(1, 4) and slow[1] < Fraction(1, 2)):
        raise ValueError("the slow cycle-length plateau is not robust")
    if not fast[0] > Fraction(1, 2):
        raise ValueError("the fast one-tick division plateau is not robust")


def _check_lineage_counts(certificate: Mapping[str, object]) -> None:
    row = _proof_section(certificate, "differential_lineage_growth")
    _require_passed(row, "differential_lineage_growth")
    expected_literals = {
        "initial_condition": "all founders are reset-phase newborns",
        "fast_cycle_length_ticks": 1,
        "slow_cycle_length_ticks": 2,
        "fast_count": "N_fast(t)=2^t",
        "slow_count": "N_slow(t)=2^floor(t/2)",
        "relative_count": "N_fast(t)/N_slow(t)=2^ceil(t/2)",
        "fast_cumulative_divisions": "C_fast(t)=N_fast(0)*(2^t-1)",
        "slow_cumulative_divisions": (
            "C_slow(t)=N_slow(0)*(2^floor(t/2)-1)"
        ),
    }
    for field, expected in expected_literals.items():
        _require_equal(row.get(field), expected, f"differential_lineage_growth.{field}")
    _require_fraction(row, "two_tick_fast_multiplier", Fraction(4), "lineage")
    _require_fraction(row, "two_tick_slow_multiplier", Fraction(2), "lineage")
    _require_fraction(
        row, "two_tick_relative_selection_factor", Fraction(2), "lineage"
    )
    if _boolean(row.get("slow_absolute_extinction"), "slow_absolute_extinction"):
        raise ValueError("the slow absolute population grows and is not extinct")
    samples = _sequence(row.get("samples"), "lineage.samples")
    if len(samples) != 9:
        raise ValueError("lineage.samples must certify ticks 0 through 8")
    for ticks, value in enumerate(samples):
        sample = _mapping(value, f"lineage.samples[{ticks}]")
        fast = 2**ticks
        slow = 2 ** (ticks // 2)
        _require_equal(
            _integer(sample.get("ticks"), f"samples[{ticks}].ticks"),
            ticks,
            f"samples[{ticks}].ticks",
        )
        _require_equal(
            _integer(sample.get("fast_descendants"), "fast_descendants"),
            fast,
            "fast_descendants",
        )
        _require_equal(
            _integer(sample.get("slow_descendants"), "slow_descendants"),
            slow,
            "slow_descendants",
        )
        _require_fraction(
            sample, "fast_to_slow_ratio", Fraction(fast, slow), f"samples[{ticks}]"
        )


def _expected_counts(
    blocks: int, mutation_probability: Fraction
) -> tuple[Fraction, Fraction]:
    slow_eigenvalue = 2 * (1 - mutation_probability)
    slow = slow_eigenvalue**blocks
    if blocks == 0:
        return Fraction(0), slow
    fast = (
        mutation_probability
        * (4**blocks - slow_eigenvalue**blocks)
        / (1 + mutation_probability)
    )
    return fast, slow


def _check_mutation_selection(certificate: Mapping[str, object]) -> None:
    p = _canonical_parameters(certificate)
    nu = Fraction(p["forward_mutation_probability"])
    row = _proof_section(certificate, "division_gated_mutation_selection")
    _require_passed(row, "division_gated_mutation_selection")
    expected_metadata = {
        "scope": "expectation under an explicit forward-mutation kernel",
        "population_vector_order": ["fast", "slow"],
        "matrix_orientation": "rows_child_columns_parent",
        "population_quantity": "expected_counts",
        "matrix_time_unit_ticks": 2,
        "mutation_timing": "at_slow_type_division_on_second_tick",
        "block_census_phase": "all surviving descendants reset",
    }
    for field, expected in expected_metadata.items():
        _require_equal(row.get(field), expected, f"mutation_selection.{field}")
    _require_fraction(row, "mutation_probability", nu, "mutation_selection")
    slow_eigenvalue = 2 * (1 - nu)
    matrix = _sequence(row.get("two_tick_matrix"), "two_tick_matrix")
    expected_matrix = ((Fraction(4), 2 * nu), (Fraction(0), slow_eigenvalue))
    if len(matrix) != 2:
        raise ValueError("two_tick_matrix must have two rows")
    for i, expected_row in enumerate(expected_matrix):
        observed_row = _sequence(matrix[i], f"two_tick_matrix[{i}]")
        if len(observed_row) != 2:
            raise ValueError("two_tick_matrix rows must have two columns")
        for j, expected in enumerate(expected_row):
            if _fraction(observed_row[j], f"two_tick_matrix[{i}][{j}]") != expected:
                raise ValueError("two_tick mutation-selection matrix is inconsistent")
    _require_fraction(row, "dominant_eigenvalue", Fraction(4), "mutation_selection")
    _require_fraction(row, "slow_eigenvalue", slow_eigenvalue, "mutation_selection")
    _require_fraction(
        row, "spectral_gap", 4 - slow_eigenvalue, "mutation_selection"
    )
    _require_fraction(
        row,
        "slow_to_fast_eigenvalue_ratio",
        slow_eigenvalue / 4,
        "mutation_selection",
    )
    if not (0 < nu <= 1 and slow_eigenvalue / 4 < 1):
        raise ValueError("the mean fast-frequency convergence condition failed")
    closed = _mapping(row.get("closed_forms"), "mutation_selection.closed_forms")
    expected_closed = {
        "slow": "S_n=[2(1-nu)]^n*S_0",
        "fast": (
            "F_n=4^n*F_0+nu/(1+nu)*(4^n-[2(1-nu)]^n)*S_0"
        ),
        "frequency_limit": "F_n/(F_n+S_n)->1 for S_0>0 and 0<nu<=1",
    }
    if dict(closed) != expected_closed:
        raise ValueError("mutation-selection closed forms changed")
    samples = _sequence(
        row.get("samples_from_F0_0_S0_1"), "mutation_selection.samples"
    )
    if len(samples) != 7:
        raise ValueError("mutation-selection samples must cover blocks 0 through 6")
    fast_recurrence, slow_recurrence = Fraction(0), Fraction(1)
    for blocks, value in enumerate(samples):
        sample = _mapping(value, f"mutation_selection.samples[{blocks}]")
        closed_fast, closed_slow = _expected_counts(blocks, nu)
        _require_equal(
            _integer(sample.get("blocks"), f"samples[{blocks}].blocks"),
            blocks,
            f"samples[{blocks}].blocks",
        )
        _require_fraction(sample, "fast_expected", fast_recurrence, "sample")
        _require_fraction(sample, "slow_expected", slow_recurrence, "sample")
        _require_fraction(sample, "closed_form_fast", closed_fast, "sample")
        _require_fraction(sample, "closed_form_slow", closed_slow, "sample")
        if not _boolean(
            sample.get("recurrence_matches_closed_form"),
            "recurrence_matches_closed_form",
        ):
            raise ValueError("mutation recurrence/closed-form equality must pass")
        fast_recurrence, slow_recurrence = (
            4 * fast_recurrence + 2 * nu * slow_recurrence,
            2 * (1 - nu) * slow_recurrence,
        )


def _partition_values(
    modules: int, copies: int
) -> tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction]:
    specified = (1 - Fraction(1, 2) ** copies) ** modules
    both = (1 - 2 * Fraction(1, 2) ** copies) ** modules
    reproduction = 2 * specified
    probability_two = both
    probability_one = 2 * (specified - both)
    probability_zero = 1 - 2 * specified + both
    return (
        specified,
        both,
        reproduction,
        probability_zero,
        probability_one,
        probability_two,
    )


def _check_partition(certificate: Mapping[str, object]) -> None:
    row = _proof_section(certificate, "stochastic_partition_threshold")
    _require_passed(row, "stochastic_partition_threshold")
    expected_formulas = {
        "specified_daughter_formula": "(1-2^(-k))^L",
        "both_daughters_formula": "(1-2^(1-k))^L",
        "expected_complete_daughters_formula": "2*(1-2^(-k))^L",
    }
    for field, expected in expected_formulas.items():
        _require_equal(row.get(field), expected, f"partition.{field}")
    modules = _integer(row.get("essential_module_count"), "essential_module_count")
    if modules != 7:
        raise ValueError("the canonical partition theorem uses seven modules")
    minimum = _integer(
        row.get("minimum_supercritical_copies_per_module"), "minimum_copies"
    )
    if minimum != 4:
        raise ValueError("the exact L=7 threshold must be k=4")
    rows = _sequence(row.get("rows"), "partition.rows")
    if len(rows) != 4:
        raise ValueError("partition.rows must cover k=1 through 4")
    expected_by_k: dict[int, tuple[Fraction, ...]] = {}
    for copies, value in enumerate(rows, start=1):
        part = _mapping(value, f"partition.rows[{copies - 1}]")
        _require_equal(
            _integer(part.get("copies_per_module"), "copies_per_module"),
            copies,
            "copies_per_module",
        )
        values = _partition_values(modules, copies)
        expected_by_k[copies] = values
        specified, both, reproduction, p_zero, p_one, p_two = values
        _require_fraction(
            part, "specified_daughter_complete_probability", specified, "partition row"
        )
        _require_fraction(
            part, "both_daughters_complete_probability", both, "partition row"
        )
        _require_fraction(
            part, "expected_complete_daughters", reproduction, "partition row"
        )
        distribution = _mapping(
            part.get("complete_daughter_count_distribution"), "offspring distribution"
        )
        _require_fraction(distribution, "P_X_0", p_zero, "offspring distribution")
        _require_fraction(distribution, "P_X_1", p_one, "offspring distribution")
        _require_fraction(distribution, "P_X_2", p_two, "offspring distribution")
        if p_zero + p_one + p_two != 1 or min(p_zero, p_one, p_two) < 0:
            raise ValueError("offspring probabilities are invalid")
        observed_supercritical = _boolean(
            part.get("supercritical_in_expectation"), "supercritical_in_expectation"
        )
        if observed_supercritical != (reproduction > 1):
            raise ValueError("partition criticality flag is inconsistent")
    below = expected_by_k[3][2]
    above = expected_by_k[4][2]
    _require_fraction(
        row, "subcritical_margin_at_k_3", 1 - below, "partition threshold"
    )
    _require_fraction(
        row, "supercritical_margin_at_k_4", above - 1, "partition threshold"
    )
    if not below < 1 < above:
        raise ValueError("k=3/k=4 do not straddle the mean threshold")
    galton_watson = _mapping(
        row.get("conditional_iid_Galton_Watson_at_k_4"), "Galton-Watson"
    )
    p_zero = expected_by_k[4][3]
    p_two = expected_by_k[4][5]
    extinction = p_zero / p_two
    survival = 1 - extinction
    _require_fraction(galton_watson, "extinction_probability", extinction, "GW")
    _require_fraction(
        galton_watson, "positive_survival_probability", survival, "GW"
    )
    if _boolean(galton_watson.get("certain_survival"), "GW.certain_survival"):
        raise ValueError("a supercritical Galton-Watson lineage can still go extinct")
    if not (0 < survival < 1):
        raise ValueError("the conditional survival probability must lie in (0,1)")


def _check_claim_scope(certificate: Mapping[str, object]) -> None:
    scope = _mapping(certificate.get("claim_scope"), "claim_scope")
    true_fields = {
        "model_relative_division_gated_copying_proven",
        "model_relative_genotype_phenotype_coupling_proven",
        "model_relative_differential_selection_proven",
        "model_relative_mutation_selection_in_expectation_proven",
        "model_relative_partition_mean_threshold_proven",
        "conditional_iid_partition_lineage_positive_survival_proven",
    }
    false_fields = {
        "sample_path_fixation_proven",
        "certain_partition_lineage_survival_proven",
        "molecular_copying_mechanism_proven",
        "endogenous_mutation_chemistry_proven",
        "empirical_autonomous_protocell_proven",
        "universal_life_theorem_proven",
    }
    if set(scope) != true_fields | false_fields:
        raise ValueError("claim_scope fields must be complete and fail closed")
    for field in true_fields:
        if not _boolean(scope[field], f"claim_scope.{field}"):
            raise ValueError(f"claim_scope.{field} must be true")
    for field in false_fields:
        if _boolean(scope[field], f"claim_scope.{field}"):
            raise ValueError(f"overclaim guard claim_scope.{field} must be false")
    if not _boolean(
        certificate.get("all_exact_model_obligations_passed"),
        "all_exact_model_obligations_passed",
    ):
        raise ValueError("all exact model obligations must pass")


def verify_coupled_certificate(
    certificate: Mapping[str, object],
) -> CoupledVerificationReport:
    """Independently recompute every coupled-model proof obligation."""

    checks: list[str] = []
    errors: list[str] = []
    obligations: tuple[
        tuple[str, Callable[[Mapping[str, object]], None]], ...
    ] = (
        ("header_and_semantics", _check_header),
        ("phenotype_coupling", _check_coupling),
        ("division_gated_transmission", _check_transmission),
        ("open_parameter_plateau", _check_robustness),
        ("differential_lineage_growth", _check_lineage_counts),
        ("mutation_selection_expectation", _check_mutation_selection),
        ("stochastic_partition", _check_partition),
        ("claim_scope", _check_claim_scope),
    )
    for name, obligation in obligations:
        try:
            obligation(certificate)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(f"{name}: {exc}")
        else:
            checks.append(name)
    return CoupledVerificationReport(not errors, tuple(checks), tuple(errors))


def independently_verified(certificate: Mapping[str, object]) -> bool:
    return verify_coupled_certificate(certificate).verified


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", help="path to the JSON certificate")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(Path(args.certificate).read_text(encoding="utf-8"))
        report = verify_coupled_certificate(_mapping(payload, "certificate"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        report = CoupledVerificationReport(False, (), (f"input: {exc}",))
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return int(not report.verified)


if __name__ == "__main__":
    raise SystemExit(main())
