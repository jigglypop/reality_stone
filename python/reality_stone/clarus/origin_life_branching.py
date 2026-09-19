"""Exact age-structured multitype branching certificate.

This module joins two pieces that were separate in ``origin_life_coupled``:
the fast/slow cell-cycle mutation model and the stochastic partition model for
essential genome modules.  The resulting process is a discrete-time,
age-structured multitype branching process with an explicit sample-path
offspring kernel.

The theorem is conditional on the declared branching assumptions.  It does
not establish autonomous chemistry, resource-limited survival, molecular copy
restoration, or an empirical origin of life.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class BranchingParameters:
    """Canonical parameters for the integrated branching theorem."""

    essential_module_count: int = 7
    copies_per_module: int = 4
    forward_mutation_probability: Fraction = Fraction(1, 16)
    fast_cycle_ticks: int = 1
    slow_cycle_ticks: int = 2


PARAMETERS = BranchingParameters()
STATE_ORDER = ("fast_newborn", "slow_newborn", "slow_aged")


def _fraction(value: Fraction | int) -> dict[str, str | float]:
    exact = Fraction(value)
    return {"exact": str(exact), "decimal": float(exact)}


def _validate_partition_parameters(module_count: int, copies: int) -> None:
    if module_count < 1:
        raise ValueError("module_count must be positive")
    if copies < 1:
        raise ValueError("copies_per_module must be positive")


def _validate_mutation_probability(probability: Fraction) -> Fraction:
    value = Fraction(probability)
    if not 0 <= value <= 1:
        raise ValueError("mutation_probability must lie in [0, 1]")
    return value


def _validate_age_state_schema(parameters: BranchingParameters) -> None:
    """Reject cycle lengths that the fixed ``(F, S0, S1)`` schema cannot encode."""

    if (
        not isinstance(parameters.fast_cycle_ticks, int)
        or isinstance(parameters.fast_cycle_ticks, bool)
        or parameters.fast_cycle_ticks != 1
        or not isinstance(parameters.slow_cycle_ticks, int)
        or isinstance(parameters.slow_cycle_ticks, bool)
        or parameters.slow_cycle_ticks != 2
    ):
        raise ValueError(
            "the fixed (fast_newborn, slow_newborn, slow_aged) schema requires "
            "fast_cycle_ticks=1 and slow_cycle_ticks=2"
        )


def complete_daughter_distribution(
    module_count: int,
    copies_per_module: int,
) -> tuple[Fraction, Fraction, Fraction]:
    """Return ``P(X=0), P(X=1), P(X=2)`` for complete daughters.

    Each of the ``copies_per_module`` copies of every essential module chooses
    one of the two daughters independently with probability one half.  Events
    for the two daughters are correlated because all copies are conserved.
    """

    _validate_partition_parameters(module_count, copies_per_module)
    half_power = Fraction(1, 2) ** copies_per_module
    specified = (1 - half_power) ** module_count
    both = (1 - 2 * half_power) ** module_count
    probability_two = both
    probability_one = 2 * (specified - both)
    probability_zero = 1 - probability_one - probability_two
    return probability_zero, probability_one, probability_two


def expected_complete_daughters(
    module_count: int,
    copies_per_module: int,
) -> Fraction:
    """Return the exact mean number of complete daughters per division."""

    _, probability_one, probability_two = complete_daughter_distribution(
        module_count, copies_per_module
    )
    return probability_one + 2 * probability_two


def minimum_supercritical_copy_number(module_count: int) -> int:
    """Return the least copy number whose complete-daughter mean exceeds one."""

    if module_count < 1:
        raise ValueError("module_count must be positive")
    copies = 1
    while expected_complete_daughters(module_count, copies) <= 1:
        copies += 1
    return copies


def division_offspring_distribution(
    parent_type: str,
    module_count: int = PARAMETERS.essential_module_count,
    copies_per_module: int = PARAMETERS.copies_per_module,
    mutation_probability: Fraction = PARAMETERS.forward_mutation_probability,
) -> dict[tuple[int, int], Fraction]:
    """Return the joint ``(fast, slow-newborn)`` offspring distribution.

    A fast parent has no back mutation.  For a slow parent, each complete
    daughter independently changes to fast with probability ``nu``.  The
    partition correlation is retained through the complete-daughter count
    distribution before daughter types are sampled.
    """

    probability_zero, probability_one, probability_two = (
        complete_daughter_distribution(module_count, copies_per_module)
    )
    nu = _validate_mutation_probability(mutation_probability)
    if parent_type == "fast":
        return {
            (0, 0): probability_zero,
            (1, 0): probability_one,
            (2, 0): probability_two,
        }
    if parent_type != "slow":
        raise ValueError("parent_type must be 'fast' or 'slow'")
    keep = 1 - nu
    return {
        (0, 0): probability_zero,
        (1, 0): probability_one * nu,
        (0, 1): probability_one * keep,
        (2, 0): probability_two * nu**2,
        (1, 1): 2 * probability_two * nu * keep,
        (0, 2): probability_two * keep**2,
    }


def age_structured_mean_matrix(
    parameters: BranchingParameters = PARAMETERS,
) -> tuple[tuple[Fraction, ...], ...]:
    """Return the one-tick mean matrix in ``STATE_ORDER``.

    Matrix orientation is rows-child, columns-parent.  A slow newborn ages
    deterministically for one tick; a slow aged cell divides on the next tick.
    """

    _validate_age_state_schema(parameters)
    reproduction = expected_complete_daughters(
        parameters.essential_module_count,
        parameters.copies_per_module,
    )
    nu = _validate_mutation_probability(
        parameters.forward_mutation_probability
    )
    return (
        (reproduction, Fraction(0), reproduction * nu),
        (Fraction(0), Fraction(0), reproduction * (1 - nu)),
        (Fraction(0), Fraction(1), Fraction(0)),
    )


def reset_census_two_tick_matrix(
    parameters: BranchingParameters = PARAMETERS,
) -> tuple[tuple[Fraction, ...], ...]:
    """Return the two-tick mean map on reset ``(fast, slow)`` cells."""

    _validate_age_state_schema(parameters)
    reproduction = expected_complete_daughters(
        parameters.essential_module_count,
        parameters.copies_per_module,
    )
    nu = _validate_mutation_probability(
        parameters.forward_mutation_probability
    )
    return (
        (reproduction**2, reproduction * nu),
        (Fraction(0), reproduction * (1 - nu)),
    )


def _offspring_pgf(
    distribution: tuple[Fraction, Fraction, Fraction],
    value: Fraction,
) -> Fraction:
    probability_zero, probability_one, probability_two = distribution
    point = Fraction(value)
    return probability_zero + probability_one * point + probability_two * point**2


def total_extinction_probability(
    module_count: int = PARAMETERS.essential_module_count,
    copies_per_module: int = PARAMETERS.copies_per_module,
) -> Fraction:
    """Return extinction probability for either founder type.

    Ignoring clock time, every complete fast or slow cell eventually divides
    and has the same total complete-daughter distribution.  The embedded
    genealogical process is therefore a one-type Galton--Watson process.
    """

    distribution = complete_daughter_distribution(
        module_count, copies_per_module
    )
    reproduction = distribution[1] + 2 * distribution[2]
    if distribution[0] == 0:
        return Fraction(0)
    if reproduction <= 1:
        return Fraction(1)
    probability_zero, _, probability_two = distribution
    if probability_two <= 0:
        raise ValueError("a supercritical quadratic offspring law needs P(X=2)>0")
    extinction = probability_zero / probability_two
    if not 0 <= extinction < 1:
        raise ValueError("invalid supercritical extinction root")
    return extinction


def slow_sublineage_extinction_probability(
    module_count: int = PARAMETERS.essential_module_count,
    copies_per_module: int = PARAMETERS.copies_per_module,
    mutation_probability: Fraction = PARAMETERS.forward_mutation_probability,
) -> Fraction:
    """Return extinction probability of descendants that remain slow.

    Fast mutants are counted as leaving the slow sublineage.  Its offspring
    pgf is ``f(nu + (1-nu) z)``.  Since one fixed point is one, the other exact
    quadratic root is the constant coefficient divided by the leading one.
    """

    probability_zero, probability_one, probability_two = (
        complete_daughter_distribution(module_count, copies_per_module)
    )
    nu = _validate_mutation_probability(mutation_probability)
    keep = 1 - nu
    reproduction = probability_one + 2 * probability_two
    zero_slow_offspring = (
        probability_zero
        + probability_one * nu
        + probability_two * nu**2
    )
    if zero_slow_offspring == 0:
        return Fraction(0)
    if reproduction * keep <= 1:
        return Fraction(1)
    leading = probability_two * keep**2
    constant = zero_slow_offspring
    if leading <= 0:
        raise ValueError("a supercritical slow sublineage needs a quadratic term")
    extinction = constant / leading
    if not 0 <= extinction < 1:
        raise ValueError("invalid slow-sublineage extinction root")
    return extinction


def _offspring_rows(
    distribution: Mapping[tuple[int, int], Fraction],
) -> list[dict[str, object]]:
    return [
        {
            "fast_daughters": fast,
            "slow_newborn_daughters": slow,
            "probability": _fraction(probability),
        }
        for (fast, slow), probability in distribution.items()
    ]


def _matrix_rows(
    matrix: Sequence[Sequence[Fraction]],
) -> list[list[dict[str, str | float]]]:
    return [[_fraction(value) for value in row] for row in matrix]


def build_branching_certificate() -> dict[str, object]:
    """Build the exact integrated age-structured branching certificate."""

    parameters = PARAMETERS
    modules = parameters.essential_module_count
    copies = parameters.copies_per_module
    nu = parameters.forward_mutation_probability
    keep = 1 - nu
    distribution = complete_daughter_distribution(modules, copies)
    probability_zero, probability_one, probability_two = distribution
    reproduction = expected_complete_daughters(modules, copies)
    specified_daughter_probability = reproduction / 2
    daughter_completeness_covariance = (
        probability_two - specified_daughter_probability**2
    )
    fast_distribution = division_offspring_distribution("fast")
    slow_distribution = division_offspring_distribution("slow")

    kernel_passed = (
        sum(distribution) == 1
        and min(distribution) >= 0
        and sum(fast_distribution.values()) == 1
        and sum(slow_distribution.values()) == 1
        and sum(
            (fast + slow) * probability
            for (fast, slow), probability in slow_distribution.items()
        )
        == reproduction
    )

    mean_matrix = age_structured_mean_matrix(parameters)
    two_tick_matrix = reset_census_two_tick_matrix(parameters)
    expected_slow_fast = sum(
        fast * probability
        for (fast, _), probability in slow_distribution.items()
    )
    expected_slow_slow = sum(
        slow * probability
        for (_, slow), probability in slow_distribution.items()
    )
    slow_block_growth_squared = reproduction * keep
    dominant_separation_squared = reproduction**2 - slow_block_growth_squared
    mean_passed = (
        expected_slow_fast == reproduction * nu
        and expected_slow_slow == slow_block_growth_squared
        and mean_matrix[0][2] == expected_slow_fast
        and mean_matrix[1][2] == expected_slow_slow
        and reproduction > 1
        and dominant_separation_squared > 0
    )

    minimum_copies = minimum_supercritical_copy_number(modules)
    reproduction_k3 = expected_complete_daughters(modules, 3)
    extinction_k3 = total_extinction_probability(modules, 3)
    threshold_passed = (
        minimum_copies == 4
        and reproduction_k3 < 1 < reproduction
        and extinction_k3 == 1
    )

    total_extinction = total_extinction_probability(modules, copies)
    total_survival = 1 - total_extinction
    total_fixed_point_residual = (
        _offspring_pgf(distribution, total_extinction) - total_extinction
    )
    total_extinction_passed = (
        total_fixed_point_residual == 0
        and _offspring_pgf(distribution, Fraction(1)) == 1
        and total_extinction == probability_zero / probability_two
        and 0 < total_survival < 1
    )

    slow_extinction = slow_sublineage_extinction_probability(
        modules, copies, nu
    )
    slow_survival = 1 - slow_extinction
    strict_fixation_given_survival = (
        (slow_extinction - total_extinction) / total_survival
    )
    slow_persistence_given_survival = slow_survival / total_survival
    slow_pgf_value = _offspring_pgf(
        distribution,
        nu + keep * slow_extinction,
    )
    slow_leading = probability_two * keep**2
    slow_linear = (
        probability_one * keep
        + 2 * probability_two * nu * keep
        - 1
    )
    slow_constant = (
        probability_zero
        + probability_one * nu
        + probability_two * nu**2
    )
    slow_persistence_passed = (
        reproduction * keep > 1
        and slow_pgf_value == slow_extinction
        and slow_extinction == slow_constant / slow_leading
        and slow_leading + slow_linear + slow_constant == 0
        and 0 < slow_survival < total_survival
        and strict_fixation_given_survival + slow_persistence_given_survival == 1
    )

    perfect_reproduction = Fraction(2)
    perfect_limit = (
        (perfect_reproduction**2, perfect_reproduction * nu),
        (Fraction(0), perfect_reproduction * keep),
    )
    prior_matrix = (
        (Fraction(4), 2 * nu),
        (Fraction(0), 2 * keep),
    )
    integration_passed = perfect_limit == prior_matrix

    proof_obligations = {
        "partition_mutation_kernel": {
            "passed": kernel_passed,
            "complete_daughter_distribution": {
                "P_X_0": _fraction(probability_zero),
                "P_X_1": _fraction(probability_one),
                "P_X_2": _fraction(probability_two),
            },
            "expected_complete_daughters": _fraction(reproduction),
            "specified_daughter_complete_probability": _fraction(
                specified_daughter_probability
            ),
            "daughter_completeness_covariance": _fraction(
                daughter_completeness_covariance
            ),
            "fast_parent_joint_offspring": _offspring_rows(fast_distribution),
            "slow_parent_joint_offspring": _offspring_rows(slow_distribution),
            "daughter_completeness_is_independent": False,
            "mutation_conditional_on_completeness": True,
        },
        "age_structured_mean_operator": {
            "passed": mean_passed,
            "state_order": list(STATE_ORDER),
            "matrix_orientation": "rows_child_columns_parent",
            "one_tick_mean_matrix": _matrix_rows(mean_matrix),
            "characteristic_factorization": (
                "(lambda-R)*(lambda^2-R*(1-nu))"
            ),
            "dominant_eigenvalue": _fraction(reproduction),
            "slow_block_eigenvalue_squared": _fraction(
                slow_block_growth_squared
            ),
            "dominant_separation_squared": _fraction(
                dominant_separation_squared
            ),
            "two_tick_reset_census_matrix": _matrix_rows(two_tick_matrix),
        },
        "copy_number_survival_threshold": {
            "passed": threshold_passed,
            "essential_module_count": modules,
            "minimum_supercritical_copies_per_module": minimum_copies,
            "k3_reproduction_mean": _fraction(reproduction_k3),
            "k3_extinction_probability": _fraction(extinction_k3),
            "k4_reproduction_mean": _fraction(reproduction),
            "k4_supercritical_margin": _fraction(reproduction - 1),
        },
        "total_sample_path_extinction": {
            "passed": total_extinction_passed,
            "embedded_generation_argument": (
                "both types eventually divide and share the same complete-"
                "daughter count law, so clock age and mutation do not change "
                "total genealogical extinction"
            ),
            "offspring_pgf": "f(z)=P0+P1*z+P2*z^2",
            "fixed_point_factorization": "f(z)-z=(z-1)*(P2*z-P0)",
            "fast_founder_extinction_probability": _fraction(total_extinction),
            "slow_founder_extinction_probability": _fraction(total_extinction),
            "positive_survival_probability": _fraction(total_survival),
            "fixed_point_residual": _fraction(total_fixed_point_residual),
            "certain_survival": False,
        },
        "persistent_slow_sublineage": {
            "passed": slow_persistence_passed,
            "initial_condition": "one_slow_newborn_founder",
            "slow_offspring_pgf": "f_s(z)=f(nu+(1-nu)*z)",
            "slow_reproduction_mean": _fraction(reproduction * keep),
            "quadratic_coefficients_for_f_s_minus_z": {
                "A": _fraction(slow_leading),
                "B": _fraction(slow_linear),
                "C": _fraction(slow_constant),
            },
            "slow_sublineage_extinction_probability": _fraction(slow_extinction),
            "slow_sublineage_survival_probability": _fraction(slow_survival),
            "probability_slow_persists_from_slow_founder_given_total_survival": _fraction(
                slow_persistence_given_survival
            ),
            "probability_strict_fast_fixation_from_slow_founder_given_total_survival": _fraction(
                strict_fixation_given_survival
            ),
            "strict_fast_fixation_definition": (
                "there exists a finite tick after which no reproductive slow "
                "cell remains"
            ),
            "strict_fast_fixation_from_slow_founder_almost_sure": False,
            "relative_fast_frequency_limit_proven": False,
        },
        "perfect_partition_limit": {
            "passed": integration_passed,
            "limit_mean_complete_daughters": _fraction(perfect_reproduction),
            "recovered_two_tick_matrix": _matrix_rows(perfect_limit),
            "prior_separated_mean_matrix": _matrix_rows(prior_matrix),
        },
    }
    all_passed = all(
        bool(section["passed"]) for section in proof_obligations.values()
    )
    return {
        "artifact_type": (
            "clarus_age_structured_multitype_branching_exact_certificate"
        ),
        "artifact_version": 1,
        "arithmetic": "fractions.Fraction exact rational arithmetic",
        "model": {
            "state_types": list(STATE_ORDER),
            "parameters": {
                "essential_module_count": modules,
                "copies_per_module": copies,
                "forward_mutation_probability": str(nu),
                "fast_cycle_ticks": parameters.fast_cycle_ticks,
                "slow_cycle_ticks": parameters.slow_cycle_ticks,
            },
            "semantics": {
                "fast_transition": "fast_newborn divides every tick",
                "slow_age_transition": "slow_newborn becomes slow_aged without division",
                "slow_division_transition": "slow_aged divides on the next tick",
                "parent_fate": "division replaces the parent",
                "incomplete_daughter_fate": "sterile and removed from the reproductive process",
                "copy_restoration": "each complete daughter restores k copies before its next division",
                "parent_independence": "partition and mutation events are iid across parents",
                "resource_regime": "unlimited and noninteracting",
                "mutation_rule": "complete slow daughters mutate independently to fast with probability nu; no back mutation",
            },
        },
        "proof_obligations": proof_obligations,
        "claim_scope": {
            "unified_age_structured_sample_path_process_defined": True,
            "model_relative_total_extinction_probability_proven": True,
            "model_relative_positive_survival_from_both_types_proven": True,
            "model_relative_positive_slow_persistence_proven": True,
            "almost_sure_strict_fast_fixation_from_slow_founder_refuted": True,
            "perfect_partition_limit_recovers_prior_mean_model": True,
            "relative_frequency_fixation_proven": False,
            "certain_survival_proven": False,
            "finite_resource_survival_proven": False,
            "molecular_copy_restoration_proven": False,
            "endogenous_mutation_chemistry_proven": False,
            "empirical_autonomous_protocell_proven": False,
            "universal_life_theorem_proven": False,
        },
        "all_exact_model_obligations_passed": all_passed,
    }


def validate_branching_certificate(certificate: Mapping[str, object]) -> bool:
    """Check exact equality with a fresh deterministic build."""

    return dict(certificate) == build_branching_certificate()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="artifacts/biology/origin_life_branching_certificate.json",
        help="certificate output path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that an existing artifact equals a fresh build",
    )
    args = parser.parse_args(argv)
    output = Path(args.output)
    certificate = build_branching_certificate()
    if args.check:
        try:
            observed = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 1
        return int(observed != certificate)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(certificate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
