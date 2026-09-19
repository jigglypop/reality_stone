"""Independent verifier for the age-structured branching certificate.

The verifier intentionally does not import the certificate builder or the
older coupled-model module.  It re-encodes the canonical process and checks
all exact values with :class:`fractions.Fraction`.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class BranchingVerificationReport:
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


def _fraction(value: object, label: str) -> Fraction:
    if isinstance(value, Mapping):
        row = _mapping(value, label)
        if set(row) != {"exact", "decimal"}:
            raise ValueError(f"{label} must contain exact and decimal")
        exact = row["exact"]
        if not isinstance(exact, (str, int)) or isinstance(exact, bool):
            raise ValueError(f"{label}.exact must be a rational string")
        result = Fraction(exact)
        decimal = row["decimal"]
        if not isinstance(decimal, (int, float)) or isinstance(decimal, bool):
            raise ValueError(f"{label}.decimal must be numeric")
        if not math.isfinite(float(decimal)):
            raise ValueError(f"{label}.decimal must be finite")
        if abs(float(result) - float(decimal)) > 1e-12:
            raise ValueError(f"{label}.decimal disagrees with exact")
        return result
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ValueError(f"{label} must be an exact rational, not a float")
    return Fraction(value)


def _require_equal(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label} is {observed!r}; expected {expected!r}")


def _require_exact_keys(
    row: Mapping[str, object], expected: set[str], label: str
) -> None:
    observed = set(row)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{label} keys changed; missing={missing}, extra={extra}")


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
        raise ValueError(f"{label}.{field} is {observed}; expected {expected}")


def _require_passed(row: Mapping[str, object], label: str) -> None:
    if not _boolean(row.get("passed"), f"{label}.passed"):
        raise ValueError(f"{label}.passed must be true")


def _section(
    certificate: Mapping[str, object], name: str
) -> Mapping[str, object]:
    obligations = _mapping(certificate.get("proof_obligations"), "proof_obligations")
    if name not in obligations:
        raise ValueError(f"missing proof_obligations.{name}")
    return _mapping(obligations[name], f"proof_obligations.{name}")


def _canonical_values() -> dict[str, Fraction | int]:
    return {
        "essential_module_count": 7,
        "copies_per_module": 4,
        "forward_mutation_probability": Fraction(1, 16),
        "fast_cycle_ticks": 1,
        "slow_cycle_ticks": 2,
    }


def _partition_values(
    modules: int, copies: int
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    half_power = Fraction(1, 2) ** copies
    specified = (1 - half_power) ** modules
    both = (1 - 2 * half_power) ** modules
    probability_two = both
    probability_one = 2 * (specified - both)
    probability_zero = 1 - probability_one - probability_two
    reproduction = probability_one + 2 * probability_two
    return probability_zero, probability_one, probability_two, reproduction


def _canonical_process() -> dict[str, object]:
    values = _canonical_values()
    modules = int(values["essential_module_count"])
    copies = int(values["copies_per_module"])
    nu = Fraction(values["forward_mutation_probability"])
    probability_zero, probability_one, probability_two, reproduction = (
        _partition_values(modules, copies)
    )
    keep = 1 - nu
    return {
        "nu": nu,
        "keep": keep,
        "p0": probability_zero,
        "p1": probability_one,
        "p2": probability_two,
        "reproduction": reproduction,
        "fast_joint": {
            (0, 0): probability_zero,
            (1, 0): probability_one,
            (2, 0): probability_two,
        },
        "slow_joint": {
            (0, 0): probability_zero,
            (1, 0): probability_one * nu,
            (0, 1): probability_one * keep,
            (2, 0): probability_two * nu**2,
            (1, 1): 2 * probability_two * nu * keep,
            (0, 2): probability_two * keep**2,
        },
    }


def _parse_matrix(
    value: object,
    rows: int,
    columns: int,
    label: str,
) -> tuple[tuple[Fraction, ...], ...]:
    outer = _sequence(value, label)
    if len(outer) != rows:
        raise ValueError(f"{label} must have {rows} rows")
    parsed: list[tuple[Fraction, ...]] = []
    for row_index, row_value in enumerate(outer):
        row = _sequence(row_value, f"{label}[{row_index}]")
        if len(row) != columns:
            raise ValueError(f"{label}[{row_index}] must have {columns} columns")
        parsed.append(
            tuple(
                _fraction(cell, f"{label}[{row_index}][{column_index}]")
                for column_index, cell in enumerate(row)
            )
        )
    return tuple(parsed)


def _parse_joint_distribution(
    value: object,
    label: str,
) -> dict[tuple[int, int], Fraction]:
    rows = _sequence(value, label)
    parsed: dict[tuple[int, int], Fraction] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"{label}[{index}]")
        if set(row) != {
            "fast_daughters",
            "slow_newborn_daughters",
            "probability",
        }:
            raise ValueError(f"{label}[{index}] has unexpected fields")
        key = (
            _integer(row["fast_daughters"], f"{label}.fast_daughters"),
            _integer(
                row["slow_newborn_daughters"],
                f"{label}.slow_newborn_daughters",
            ),
        )
        if key in parsed:
            raise ValueError(f"{label} contains duplicate offspring {key}")
        parsed[key] = _fraction(row["probability"], f"{label}.probability")
    return parsed


def _check_header(certificate: Mapping[str, object]) -> None:
    _require_exact_keys(
        certificate,
        {
            "artifact_type",
            "artifact_version",
            "arithmetic",
            "model",
            "proof_obligations",
            "claim_scope",
            "all_exact_model_obligations_passed",
        },
        "certificate",
    )
    _require_equal(
        certificate.get("artifact_type"),
        "clarus_age_structured_multitype_branching_exact_certificate",
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
    _require_exact_keys(model, {"state_types", "parameters", "semantics"}, "model")
    _require_equal(
        model.get("state_types"),
        ["fast_newborn", "slow_newborn", "slow_aged"],
        "model.state_types",
    )
    parameters = _mapping(model.get("parameters"), "model.parameters")
    expected = _canonical_values()
    if set(parameters) != set(expected):
        raise ValueError("model.parameters must contain exactly canonical fields")
    for field, expected_value in expected.items():
        if isinstance(expected_value, int):
            observed: Fraction | int = _integer(
                parameters[field], f"model.parameters.{field}"
            )
        else:
            observed = _fraction(parameters[field], f"model.parameters.{field}")
        if observed != expected_value:
            raise ValueError(
                f"model.parameters.{field} is {observed}; expected {expected_value}"
            )
    semantics = _mapping(model.get("semantics"), "model.semantics")
    expected_semantics = {
        "fast_transition": "fast_newborn divides every tick",
        "slow_age_transition": "slow_newborn becomes slow_aged without division",
        "slow_division_transition": "slow_aged divides on the next tick",
        "parent_fate": "division replaces the parent",
        "incomplete_daughter_fate": "sterile and removed from the reproductive process",
        "copy_restoration": "each complete daughter restores k copies before its next division",
        "parent_independence": "partition and mutation events are iid across parents",
        "resource_regime": "unlimited and noninteracting",
        "mutation_rule": "complete slow daughters mutate independently to fast with probability nu; no back mutation",
    }
    if dict(semantics) != expected_semantics:
        raise ValueError("model.semantics changed the certified process")
    obligations = _mapping(
        certificate.get("proof_obligations"), "proof_obligations"
    )
    _require_exact_keys(
        obligations,
        {
            "partition_mutation_kernel",
            "age_structured_mean_operator",
            "copy_number_survival_threshold",
            "total_sample_path_extinction",
            "persistent_slow_sublineage",
            "perfect_partition_limit",
        },
        "proof_obligations",
    )


def _check_kernel(certificate: Mapping[str, object]) -> None:
    process = _canonical_process()
    row = _section(certificate, "partition_mutation_kernel")
    _require_passed(row, "partition_mutation_kernel")
    _require_exact_keys(
        row,
        {
            "passed",
            "complete_daughter_distribution",
            "expected_complete_daughters",
            "specified_daughter_complete_probability",
            "daughter_completeness_covariance",
            "fast_parent_joint_offspring",
            "slow_parent_joint_offspring",
            "daughter_completeness_is_independent",
            "mutation_conditional_on_completeness",
        },
        "partition_mutation_kernel",
    )
    distribution = _mapping(
        row.get("complete_daughter_distribution"),
        "complete_daughter_distribution",
    )
    if set(distribution) != {"P_X_0", "P_X_1", "P_X_2"}:
        raise ValueError("complete-daughter distribution fields changed")
    for field, key in (("P_X_0", "p0"), ("P_X_1", "p1"), ("P_X_2", "p2")):
        _require_fraction(
            distribution,
            field,
            Fraction(process[key]),
            "complete_daughter_distribution",
        )
    _require_fraction(
        row,
        "expected_complete_daughters",
        Fraction(process["reproduction"]),
        "partition_mutation_kernel",
    )
    reproduction = Fraction(process["reproduction"])
    specified = reproduction / 2
    covariance = Fraction(process["p2"]) - specified**2
    _require_fraction(
        row,
        "specified_daughter_complete_probability",
        specified,
        "partition_mutation_kernel",
    )
    _require_fraction(
        row,
        "daughter_completeness_covariance",
        covariance,
        "partition_mutation_kernel",
    )
    if covariance >= 0:
        raise ValueError("canonical daughter completeness covariance must be negative")
    fast = _parse_joint_distribution(
        row.get("fast_parent_joint_offspring"), "fast_parent_joint_offspring"
    )
    slow = _parse_joint_distribution(
        row.get("slow_parent_joint_offspring"), "slow_parent_joint_offspring"
    )
    if fast != process["fast_joint"] or slow != process["slow_joint"]:
        raise ValueError("joint mutation-partition offspring law changed")
    if _boolean(
        row.get("daughter_completeness_is_independent"),
        "daughter_completeness_is_independent",
    ):
        raise ValueError("daughter completeness must retain partition correlation")
    if not _boolean(
        row.get("mutation_conditional_on_completeness"),
        "mutation_conditional_on_completeness",
    ):
        raise ValueError("mutation must be conditional on daughter completeness")


def _check_mean_operator(certificate: Mapping[str, object]) -> None:
    process = _canonical_process()
    row = _section(certificate, "age_structured_mean_operator")
    _require_passed(row, "age_structured_mean_operator")
    _require_exact_keys(
        row,
        {
            "passed",
            "state_order",
            "matrix_orientation",
            "one_tick_mean_matrix",
            "characteristic_factorization",
            "dominant_eigenvalue",
            "slow_block_eigenvalue_squared",
            "dominant_separation_squared",
            "two_tick_reset_census_matrix",
        },
        "age_structured_mean_operator",
    )
    reproduction = Fraction(process["reproduction"])
    nu = Fraction(process["nu"])
    keep = Fraction(process["keep"])
    expected_matrix = (
        (reproduction, Fraction(0), reproduction * nu),
        (Fraction(0), Fraction(0), reproduction * keep),
        (Fraction(0), Fraction(1), Fraction(0)),
    )
    observed_matrix = _parse_matrix(
        row.get("one_tick_mean_matrix"), 3, 3, "one_tick_mean_matrix"
    )
    if observed_matrix != expected_matrix:
        raise ValueError("one-tick mean matrix changed")
    expected_two_tick = (
        (reproduction**2, reproduction * nu),
        (Fraction(0), reproduction * keep),
    )
    observed_two_tick = _parse_matrix(
        row.get("two_tick_reset_census_matrix"),
        2,
        2,
        "two_tick_reset_census_matrix",
    )
    if observed_two_tick != expected_two_tick:
        raise ValueError("two-tick newborn census matrix changed")
    _require_equal(
        row.get("state_order"),
        ["fast_newborn", "slow_newborn", "slow_aged"],
        "state_order",
    )
    _require_equal(
        row.get("matrix_orientation"),
        "rows_child_columns_parent",
        "matrix_orientation",
    )
    _require_equal(
        row.get("characteristic_factorization"),
        "(lambda-R)*(lambda^2-R*(1-nu))",
        "characteristic_factorization",
    )
    _require_fraction(row, "dominant_eigenvalue", reproduction, "mean_operator")
    _require_fraction(
        row,
        "slow_block_eigenvalue_squared",
        reproduction * keep,
        "mean_operator",
    )
    _require_fraction(
        row,
        "dominant_separation_squared",
        reproduction**2 - reproduction * keep,
        "mean_operator",
    )


def _check_threshold(certificate: Mapping[str, object]) -> None:
    process = _canonical_process()
    row = _section(certificate, "copy_number_survival_threshold")
    _require_passed(row, "copy_number_survival_threshold")
    _require_exact_keys(
        row,
        {
            "passed",
            "essential_module_count",
            "minimum_supercritical_copies_per_module",
            "k3_reproduction_mean",
            "k3_extinction_probability",
            "k4_reproduction_mean",
            "k4_supercritical_margin",
        },
        "copy_number_survival_threshold",
    )
    _, _, _, reproduction_k3 = _partition_values(7, 3)
    reproduction_k4 = Fraction(process["reproduction"])
    _require_equal(
        _integer(row.get("essential_module_count"), "essential_module_count"),
        7,
        "essential_module_count",
    )
    _require_equal(
        _integer(
            row.get("minimum_supercritical_copies_per_module"),
            "minimum_supercritical_copies_per_module",
        ),
        4,
        "minimum_supercritical_copies_per_module",
    )
    _require_fraction(row, "k3_reproduction_mean", reproduction_k3, "threshold")
    _require_fraction(row, "k3_extinction_probability", Fraction(1), "threshold")
    _require_fraction(row, "k4_reproduction_mean", reproduction_k4, "threshold")
    _require_fraction(
        row, "k4_supercritical_margin", reproduction_k4 - 1, "threshold"
    )
    if not reproduction_k3 < 1 < reproduction_k4:
        raise ValueError("k=3/k=4 no longer straddle the survival threshold")


def _check_total_extinction(certificate: Mapping[str, object]) -> None:
    process = _canonical_process()
    row = _section(certificate, "total_sample_path_extinction")
    _require_passed(row, "total_sample_path_extinction")
    _require_exact_keys(
        row,
        {
            "passed",
            "embedded_generation_argument",
            "offspring_pgf",
            "fixed_point_factorization",
            "fast_founder_extinction_probability",
            "slow_founder_extinction_probability",
            "positive_survival_probability",
            "fixed_point_residual",
            "certain_survival",
        },
        "total_sample_path_extinction",
    )
    probability_zero = Fraction(process["p0"])
    probability_one = Fraction(process["p1"])
    probability_two = Fraction(process["p2"])
    extinction = probability_zero / probability_two
    survival = 1 - extinction
    _require_equal(
        row.get("embedded_generation_argument"),
        (
            "both types eventually divide and share the same complete-daughter "
            "count law, so clock age and mutation do not change total "
            "genealogical extinction"
        ),
        "embedded_generation_argument",
    )
    _require_equal(
        row.get("offspring_pgf"),
        "f(z)=P0+P1*z+P2*z^2",
        "offspring_pgf",
    )
    _require_equal(
        row.get("fixed_point_factorization"),
        "f(z)-z=(z-1)*(P2*z-P0)",
        "fixed_point_factorization",
    )
    _require_fraction(
        row, "fast_founder_extinction_probability", extinction, "extinction"
    )
    _require_fraction(
        row, "slow_founder_extinction_probability", extinction, "extinction"
    )
    _require_fraction(
        row, "positive_survival_probability", survival, "extinction"
    )
    _require_fraction(row, "fixed_point_residual", Fraction(0), "extinction")
    if _boolean(row.get("certain_survival"), "certain_survival"):
        raise ValueError("supercritical survival cannot be marked certain")
    residual = (
        probability_zero
        + probability_one * extinction
        + probability_two * extinction**2
        - extinction
    )
    if residual != 0 or not 0 < extinction < 1:
        raise ValueError("extinction root is invalid")


def _check_slow_persistence(certificate: Mapping[str, object]) -> None:
    process = _canonical_process()
    row = _section(certificate, "persistent_slow_sublineage")
    _require_passed(row, "persistent_slow_sublineage")
    _require_exact_keys(
        row,
        {
            "passed",
            "initial_condition",
            "slow_offspring_pgf",
            "slow_reproduction_mean",
            "quadratic_coefficients_for_f_s_minus_z",
            "slow_sublineage_extinction_probability",
            "slow_sublineage_survival_probability",
            "probability_slow_persists_from_slow_founder_given_total_survival",
            "probability_strict_fast_fixation_from_slow_founder_given_total_survival",
            "strict_fast_fixation_definition",
            "strict_fast_fixation_from_slow_founder_almost_sure",
            "relative_fast_frequency_limit_proven",
        },
        "persistent_slow_sublineage",
    )
    probability_zero = Fraction(process["p0"])
    probability_one = Fraction(process["p1"])
    probability_two = Fraction(process["p2"])
    reproduction = Fraction(process["reproduction"])
    nu = Fraction(process["nu"])
    keep = Fraction(process["keep"])
    leading = probability_two * keep**2
    linear = probability_one * keep + 2 * probability_two * nu * keep - 1
    constant = probability_zero + probability_one * nu + probability_two * nu**2
    extinction = constant / leading
    survival = 1 - extinction
    total_survival = 1 - probability_zero / probability_two
    strict_fixation_given_survival = (
        (extinction - probability_zero / probability_two) / total_survival
    )
    slow_persistence_given_survival = survival / total_survival
    _require_equal(
        row.get("initial_condition"),
        "one_slow_newborn_founder",
        "initial_condition",
    )
    _require_equal(
        row.get("slow_offspring_pgf"),
        "f_s(z)=f(nu+(1-nu)*z)",
        "slow_offspring_pgf",
    )
    _require_fraction(
        row, "slow_reproduction_mean", reproduction * keep, "slow_persistence"
    )
    coefficients = _mapping(
        row.get("quadratic_coefficients_for_f_s_minus_z"),
        "slow quadratic coefficients",
    )
    _require_exact_keys(coefficients, {"A", "B", "C"}, "slow coefficients")
    _require_fraction(coefficients, "A", leading, "slow coefficients")
    _require_fraction(coefficients, "B", linear, "slow coefficients")
    _require_fraction(coefficients, "C", constant, "slow coefficients")
    _require_fraction(
        row,
        "slow_sublineage_extinction_probability",
        extinction,
        "slow_persistence",
    )
    _require_fraction(
        row,
        "slow_sublineage_survival_probability",
        survival,
        "slow_persistence",
    )
    _require_fraction(
        row,
        "probability_slow_persists_from_slow_founder_given_total_survival",
        slow_persistence_given_survival,
        "slow_persistence",
    )
    _require_fraction(
        row,
        "probability_strict_fast_fixation_from_slow_founder_given_total_survival",
        strict_fixation_given_survival,
        "slow_persistence",
    )
    _require_equal(
        row.get("strict_fast_fixation_definition"),
        (
            "there exists a finite tick after which no reproductive slow cell "
            "remains"
        ),
        "strict_fast_fixation_definition",
    )
    if _boolean(
        row.get("strict_fast_fixation_from_slow_founder_almost_sure"),
        "strict_fast_fixation_from_slow_founder_almost_sure",
    ):
        raise ValueError("positive slow survival refutes almost-sure strict fixation")
    if _boolean(
        row.get("relative_fast_frequency_limit_proven"),
        "relative_fast_frequency_limit_proven",
    ):
        raise ValueError("relative-frequency convergence remains unproved")
    if (
        leading + linear + constant != 0
        or not 0 < survival < total_survival
        or strict_fixation_given_survival + slow_persistence_given_survival != 1
    ):
        raise ValueError("slow-sublineage roots or ordering are invalid")


def _check_perfect_limit(certificate: Mapping[str, object]) -> None:
    process = _canonical_process()
    row = _section(certificate, "perfect_partition_limit")
    _require_passed(row, "perfect_partition_limit")
    _require_exact_keys(
        row,
        {
            "passed",
            "limit_mean_complete_daughters",
            "recovered_two_tick_matrix",
            "prior_separated_mean_matrix",
        },
        "perfect_partition_limit",
    )
    nu = Fraction(process["nu"])
    expected = (
        (Fraction(4), 2 * nu),
        (Fraction(0), 2 * (1 - nu)),
    )
    _require_fraction(
        row,
        "limit_mean_complete_daughters",
        Fraction(2),
        "perfect_partition_limit",
    )
    recovered = _parse_matrix(
        row.get("recovered_two_tick_matrix"),
        2,
        2,
        "recovered_two_tick_matrix",
    )
    prior = _parse_matrix(
        row.get("prior_separated_mean_matrix"),
        2,
        2,
        "prior_separated_mean_matrix",
    )
    if recovered != expected or prior != expected:
        raise ValueError("perfect-partition limit does not recover the prior model")


def _check_claim_scope(certificate: Mapping[str, object]) -> None:
    scope = _mapping(certificate.get("claim_scope"), "claim_scope")
    true_fields = {
        "unified_age_structured_sample_path_process_defined",
        "model_relative_total_extinction_probability_proven",
        "model_relative_positive_survival_from_both_types_proven",
        "model_relative_positive_slow_persistence_proven",
        "almost_sure_strict_fast_fixation_from_slow_founder_refuted",
        "perfect_partition_limit_recovers_prior_mean_model",
    }
    false_fields = {
        "relative_frequency_fixation_proven",
        "certain_survival_proven",
        "finite_resource_survival_proven",
        "molecular_copy_restoration_proven",
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


def verify_branching_certificate(
    certificate: Mapping[str, object],
) -> BranchingVerificationReport:
    """Independently recompute every integrated branching obligation."""

    checks: list[str] = []
    errors: list[str] = []
    obligations: tuple[
        tuple[str, Callable[[Mapping[str, object]], None]], ...
    ] = (
        ("header_and_semantics", _check_header),
        ("partition_mutation_kernel", _check_kernel),
        ("age_structured_mean_operator", _check_mean_operator),
        ("copy_number_survival_threshold", _check_threshold),
        ("total_sample_path_extinction", _check_total_extinction),
        ("persistent_slow_sublineage", _check_slow_persistence),
        ("perfect_partition_limit", _check_perfect_limit),
        ("claim_scope", _check_claim_scope),
    )
    for name, obligation in obligations:
        try:
            obligation(certificate)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(f"{name}: {exc}")
        else:
            checks.append(name)
    return BranchingVerificationReport(not errors, tuple(checks), tuple(errors))


def independently_verified(certificate: Mapping[str, object]) -> bool:
    return verify_branching_certificate(certificate).verified


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", help="path to the JSON certificate")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(Path(args.certificate).read_text(encoding="utf-8"))
        report = verify_branching_certificate(_mapping(payload, "certificate"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        report = BranchingVerificationReport(False, (), (f"input: {exc}",))
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return int(not report.verified)


if __name__ == "__main__":
    raise SystemExit(main())
