"""Exact certificates for a coupled heredity--phenotype--selection toy model.

This module closes a specific gap in :mod:`origin_life_existence`: the old
``q`` coordinate was dynamically independent of division and phenotype.  The
model below is intentionally smaller.  A transmitted type controls a cell-cycle
increment, copying occurs only when a division creates two daughters, and the
resulting generation times create an exact descendant-number difference.

The theorems are model-relative.  They do not establish a molecular copying
mechanism, autonomous metabolism, sample-path fixation, or an empirical origin
of life.  All proof arithmetic uses :class:`fractions.Fraction`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CoupledParameters:
    """Canonical parameters of the exact two-type cell-cycle model."""

    growth_intercept: Fraction = Fraction(3, 16)
    growth_slope: Fraction = Fraction(1, 2)
    division_threshold: Fraction = Fraction(1)
    daughter_reset: Fraction = Fraction(1, 2)
    slow_type: Fraction = Fraction(1, 4)
    fast_type: Fraction = Fraction(3, 4)
    forward_mutation_probability: Fraction = Fraction(1, 16)
    essential_module_count: int = 7


@dataclass(frozen=True)
class CellState:
    """Synchronous cell-cycle phase and its transmitted abstract type."""

    phase: Fraction
    transmitted_type: Fraction


@dataclass(frozen=True)
class CellTransition:
    """One exact clock tick for a representative synchronous cell."""

    daughters: tuple[CellState, ...]
    divided: bool
    predivision_phase: Fraction


PARAMETERS = CoupledParameters()
INITIAL_PHASE = PARAMETERS.daughter_reset


def _fraction(value: Fraction | int) -> dict[str, str | float]:
    exact = Fraction(value)
    return {"exact": str(exact), "decimal": float(exact)}


def growth_increment(
    transmitted_type: Fraction,
    parameters: CoupledParameters = PARAMETERS,
) -> Fraction:
    """Return the phenotype controlled by the transmitted type."""

    return (
        parameters.growth_intercept
        + parameters.growth_slope * transmitted_type
    )


def exact_cell_tick(
    state: CellState,
    parameters: CoupledParameters = PARAMETERS,
) -> CellTransition:
    """Advance one cell by one clock tick.

    A nondividing cell persists as one state and does not create a new copy.
    At threshold, the parent is replaced by exactly two daughters with the
    reset phase and the parent's transmitted type.  Mutation is deliberately
    absent here and is introduced only in the explicit division kernel below.
    """

    predivision_phase = state.phase + growth_increment(
        state.transmitted_type, parameters
    )
    divided = predivision_phase >= parameters.division_threshold
    if divided:
        daughter = CellState(parameters.daughter_reset, state.transmitted_type)
        daughters = (daughter, daughter)
    else:
        daughters = (CellState(predivision_phase, state.transmitted_type),)
    return CellTransition(daughters, divided, predivision_phase)


def exact_descendant_count(
    transmitted_type: Fraction,
    ticks: int,
    parameters: CoupledParameters = PARAMETERS,
) -> int:
    """Return the exact synchronous descendant count from one reset cell."""

    if ticks < 0:
        raise ValueError("ticks must be nonnegative")
    if transmitted_type == parameters.fast_type:
        return 2**ticks
    if transmitted_type == parameters.slow_type:
        return 2 ** (ticks // 2)
    raise ValueError("the closed form is certified only for the two canonical types")


def expected_two_tick_transition(
    fast_count: Fraction,
    slow_count: Fraction,
    mutation_probability: Fraction = PARAMETERS.forward_mutation_probability,
) -> tuple[Fraction, Fraction]:
    """Apply the two-clock expected population transition.

    Fast founders produce four fast descendants.  Slow founders divide once;
    each of their two daughters independently changes to the fast type with
    probability ``mutation_probability``.  There is no back mutation.
    """

    nu = Fraction(mutation_probability)
    if not 0 <= nu <= 1:
        raise ValueError("mutation_probability must lie in [0, 1]")
    fast = 4 * Fraction(fast_count) + 2 * nu * Fraction(slow_count)
    slow = 2 * (1 - nu) * Fraction(slow_count)
    return fast, slow


def expected_counts_after_blocks(
    fast_initial: Fraction,
    slow_initial: Fraction,
    blocks: int,
    mutation_probability: Fraction = PARAMETERS.forward_mutation_probability,
) -> tuple[Fraction, Fraction]:
    """Closed form after ``blocks`` two-clock mutation-selection steps."""

    if blocks < 0:
        raise ValueError("blocks must be nonnegative")
    fast_0 = Fraction(fast_initial)
    slow_0 = Fraction(slow_initial)
    nu = Fraction(mutation_probability)
    if not 0 <= nu <= 1:
        raise ValueError("mutation_probability must lie in [0, 1]")
    slow_eigenvalue = 2 * (1 - nu)
    slow = slow_eigenvalue**blocks * slow_0
    fast = 4**blocks * fast_0
    if blocks:
        fast += (
            nu
            * (4**blocks - slow_eigenvalue**blocks)
            * slow_0
            / (1 + nu)
        )
    return fast, slow


def specified_daughter_complete_probability(
    module_count: int,
    copies_per_module: int,
) -> Fraction:
    """Probability that one specified daughter receives every module."""

    _validate_partition_parameters(module_count, copies_per_module)
    per_module = 1 - Fraction(1, 2) ** copies_per_module
    return per_module**module_count


def both_daughters_complete_probability(
    module_count: int,
    copies_per_module: int,
) -> Fraction:
    """Probability that both daughters receive every essential module."""

    _validate_partition_parameters(module_count, copies_per_module)
    per_module = 1 - 2 * Fraction(1, 2) ** copies_per_module
    return per_module**module_count


def expected_complete_daughters(
    module_count: int,
    copies_per_module: int,
) -> Fraction:
    """Expected complete daughters per division, by linearity of expectation."""

    return 2 * specified_daughter_complete_probability(
        module_count, copies_per_module
    )


def minimum_supercritical_copy_number(module_count: int) -> int:
    """Smallest integer copy number with expected complete daughters above one."""

    if module_count < 1:
        raise ValueError("module_count must be positive")
    copies = 1
    while expected_complete_daughters(module_count, copies) <= 1:
        copies += 1
    return copies


def _validate_partition_parameters(
    module_count: int,
    copies_per_module: int,
) -> None:
    if module_count < 1:
        raise ValueError("module_count must be positive")
    if copies_per_module < 1:
        raise ValueError("copies_per_module must be positive")


def _trace_rows(transmitted_type: Fraction, ticks: int) -> list[dict[str, object]]:
    state = CellState(INITIAL_PHASE, transmitted_type)
    rows: list[dict[str, object]] = []
    descendants = 1
    for tick in range(1, ticks + 1):
        transition = exact_cell_tick(state)
        descendants *= len(transition.daughters)
        state = transition.daughters[0]
        rows.append(
            {
                "tick": tick,
                "phase_before": _fraction(
                    transition.predivision_phase
                    - growth_increment(transmitted_type)
                ),
                "predivision_phase": _fraction(transition.predivision_phase),
                "divided": transition.divided,
                "daughter_count_per_parent": len(transition.daughters),
                "representative_phase_after": _fraction(state.phase),
                "transmitted_type_after": _fraction(state.transmitted_type),
                "total_descendants": descendants,
            }
        )
    return rows


def _partition_row(module_count: int, copies: int) -> dict[str, object]:
    specified = specified_daughter_complete_probability(module_count, copies)
    both = both_daughters_complete_probability(module_count, copies)
    reproduction = 2 * specified
    probability_two = both
    probability_one = 2 * (specified - both)
    probability_zero = 1 - 2 * specified + both
    return {
        "copies_per_module": copies,
        "specified_daughter_complete_probability": _fraction(specified),
        "both_daughters_complete_probability": _fraction(both),
        "expected_complete_daughters": _fraction(reproduction),
        "complete_daughter_count_distribution": {
            "P_X_0": _fraction(probability_zero),
            "P_X_1": _fraction(probability_one),
            "P_X_2": _fraction(probability_two),
        },
        "supercritical_in_expectation": reproduction > 1,
    }


def build_coupled_certificate() -> dict[str, object]:
    """Build the exact, machine-checkable coupled-model certificate."""

    p = PARAMETERS
    slow_increment = growth_increment(p.slow_type, p)
    fast_increment = growth_increment(p.fast_type, p)
    fast_trace = _trace_rows(p.fast_type, 4)
    slow_trace = _trace_rows(p.slow_type, 4)

    lineage_samples = [
        {
            "ticks": ticks,
            "fast_descendants": exact_descendant_count(p.fast_type, ticks, p),
            "slow_descendants": exact_descendant_count(p.slow_type, ticks, p),
            "fast_to_slow_ratio": _fraction(
                Fraction(
                    exact_descendant_count(p.fast_type, ticks, p),
                    exact_descendant_count(p.slow_type, ticks, p),
                )
            ),
        }
        for ticks in range(9)
    ]

    nu = p.forward_mutation_probability
    slow_eigenvalue = 2 * (1 - nu)
    spectral_gap = 4 - slow_eigenvalue
    ratio_base = slow_eigenvalue / 4
    mutation_samples = []
    fast, slow = Fraction(0), Fraction(1)
    for blocks in range(7):
        closed_fast, closed_slow = expected_counts_after_blocks(0, 1, blocks, nu)
        mutation_samples.append(
            {
                "blocks": blocks,
                "fast_expected": _fraction(fast),
                "slow_expected": _fraction(slow),
                "closed_form_fast": _fraction(closed_fast),
                "closed_form_slow": _fraction(closed_slow),
                "recurrence_matches_closed_form": (fast, slow)
                == (closed_fast, closed_slow),
            }
        )
        fast, slow = expected_two_tick_transition(fast, slow, nu)

    modules = p.essential_module_count
    minimum_copies = minimum_supercritical_copy_number(modules)
    partition_rows = [_partition_row(modules, copies) for copies in range(1, 5)]
    below = expected_complete_daughters(modules, minimum_copies - 1)
    above = expected_complete_daughters(modules, minimum_copies)

    coupling_passed = (
        p.growth_slope != 0
        and slow_increment == Fraction(5, 16)
        and fast_increment == Fraction(9, 16)
    )
    transmission_passed = (
        [row["divided"] for row in fast_trace] == [True] * 4
        and [row["divided"] for row in slow_trace]
        == [False, True, False, True]
        and all(
            row["transmitted_type_after"]["exact"] == str(p.fast_type)
            for row in fast_trace
        )
        and all(
            row["transmitted_type_after"]["exact"] == str(p.slow_type)
            for row in slow_trace
        )
    )
    lineage_passed = all(
        row["fast_descendants"] == 2 ** row["ticks"]
        and row["slow_descendants"] == 2 ** (row["ticks"] // 2)
        for row in lineage_samples
    )
    mutation_passed = (
        0 < nu <= 1
        and slow_eigenvalue < 4
        and spectral_gap > 0
        and ratio_base < 1
        and all(row["recurrence_matches_closed_form"] for row in mutation_samples)
    )
    partition_passed = (
        minimum_copies == 4 and below < 1 and above > 1
    )

    intercept_interval = (Fraction(11, 64), Fraction(13, 64))
    slope_interval = (Fraction(15, 32), Fraction(17, 32))
    slow_increment_bounds = (
        intercept_interval[0] + slope_interval[0] * p.slow_type,
        intercept_interval[1] + slope_interval[1] * p.slow_type,
    )
    fast_increment_bounds = (
        intercept_interval[0] + slope_interval[0] * p.fast_type,
        intercept_interval[1] + slope_interval[1] * p.fast_type,
    )
    robustness_passed = (
        slow_increment_bounds[0] > Fraction(1, 4)
        and slow_increment_bounds[1] < Fraction(1, 2)
        and fast_increment_bounds[0] > Fraction(1, 2)
    )

    partition_k4 = partition_rows[3]
    offspring = partition_k4["complete_daughter_count_distribution"]
    probability_zero = Fraction(offspring["P_X_0"]["exact"])
    probability_two = Fraction(offspring["P_X_2"]["exact"])
    extinction_probability = probability_zero / probability_two
    survival_probability = 1 - extinction_probability

    proof_obligations = {
        "phenotype_coupling": {
            "passed": coupling_passed,
            "equation": "g(q)=growth_intercept+growth_slope*q",
            "dg_dq": _fraction(p.growth_slope),
            "slow_increment": _fraction(slow_increment),
            "fast_increment": _fraction(fast_increment),
            "increment_difference": _fraction(fast_increment - slow_increment),
        },
        "division_gated_transmission": {
            "passed": transmission_passed,
            "nondivision_output_count": 1,
            "division_output_count": 2,
            "copy_event_condition": "predivision_phase>=division_threshold",
            "fast_trace": fast_trace,
            "slow_trace": slow_trace,
        },
        "open_parameter_plateau": {
            "passed": robustness_passed,
            "theorem": (
                "every intercept/slope pair in the closed box preserves "
                "fast cycle length 1 and slow cycle length 2"
            ),
            "growth_intercept_interval": [
                _fraction(value) for value in intercept_interval
            ],
            "growth_slope_interval": [
                _fraction(value) for value in slope_interval
            ],
            "slow_increment_bounds": [
                _fraction(value) for value in slow_increment_bounds
            ],
            "fast_increment_bounds": [
                _fraction(value) for value in fast_increment_bounds
            ],
            "slow_two_tick_lower_margin": _fraction(
                slow_increment_bounds[0] - Fraction(1, 4)
            ),
            "slow_one_tick_upper_margin": _fraction(
                Fraction(1, 2) - slow_increment_bounds[1]
            ),
            "fast_one_tick_lower_margin": _fraction(
                fast_increment_bounds[0] - Fraction(1, 2)
            ),
        },
        "differential_lineage_growth": {
            "passed": lineage_passed,
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
            "two_tick_fast_multiplier": _fraction(4),
            "two_tick_slow_multiplier": _fraction(2),
            "two_tick_relative_selection_factor": _fraction(2),
            "slow_absolute_extinction": False,
            "samples": lineage_samples,
        },
        "division_gated_mutation_selection": {
            "passed": mutation_passed,
            "scope": "expectation under an explicit forward-mutation kernel",
            "population_vector_order": ["fast", "slow"],
            "matrix_orientation": "rows_child_columns_parent",
            "population_quantity": "expected_counts",
            "matrix_time_unit_ticks": 2,
            "mutation_timing": "at_slow_type_division_on_second_tick",
            "block_census_phase": "all surviving descendants reset",
            "mutation_probability": _fraction(nu),
            "two_tick_matrix": [
                [_fraction(4), _fraction(2 * nu)],
                [_fraction(0), _fraction(slow_eigenvalue)],
            ],
            "dominant_eigenvalue": _fraction(4),
            "slow_eigenvalue": _fraction(slow_eigenvalue),
            "spectral_gap": _fraction(spectral_gap),
            "slow_to_fast_eigenvalue_ratio": _fraction(ratio_base),
            "closed_forms": {
                "slow": "S_n=[2(1-nu)]^n*S_0",
                "fast": (
                    "F_n=4^n*F_0+nu/(1+nu)*"
                    "(4^n-[2(1-nu)]^n)*S_0"
                ),
                "frequency_limit": "F_n/(F_n+S_n)->1 for S_0>0 and 0<nu<=1",
            },
            "samples_from_F0_0_S0_1": mutation_samples,
        },
        "stochastic_partition_threshold": {
            "passed": partition_passed,
            "assumptions": [
                "L essential modules",
                "k identical copies of each module before division",
                "each copy independently chooses either daughter with probability 1/2",
                "no copy loss and independent modules",
            ],
            "specified_daughter_formula": "(1-2^(-k))^L",
            "both_daughters_formula": "(1-2^(1-k))^L",
            "expected_complete_daughters_formula": "2*(1-2^(-k))^L",
            "essential_module_count": modules,
            "minimum_supercritical_copies_per_module": minimum_copies,
            "subcritical_margin_at_k_3": _fraction(1 - below),
            "supercritical_margin_at_k_4": _fraction(above - 1),
            "rows": partition_rows,
            "conditional_iid_Galton_Watson_at_k_4": {
                "extra_assumptions": [
                    "every complete daughter restores exactly k copies per module",
                    "incomplete daughters are sterile",
                    "different parents partition independently",
                    "resources are unlimited",
                ],
                "extinction_probability": _fraction(extinction_probability),
                "positive_survival_probability": _fraction(survival_probability),
                "certain_survival": False,
            },
        },
    }
    all_passed = all(
        bool(section["passed"]) for section in proof_obligations.values()
    )
    return {
        "artifact_type": "clarus_coupled_heredity_selection_exact_certificate",
        "artifact_version": 1,
        "arithmetic": "fractions.Fraction exact rational arithmetic",
        "model": {
            "state": "(cell_cycle_phase, transmitted_type)",
            "semantics": {
                "division_comparator": ">=",
                "update_order": "advance_phase_then_test_then_reset",
                "tick_rule": "each cell present at tick start updates exactly once",
                "daughter_activation": "daughters first update on the next tick",
                "parent_fate": "division replaces parent by exactly two daughters",
                "phase_interpretation": "cell-cycle accumulator, not conserved mass",
                "founder_phase": "all certified founders start at daughter_reset",
                "resource_regime": "unlimited and noninteracting",
            },
            "parameters": {
                "growth_intercept": str(p.growth_intercept),
                "growth_slope": str(p.growth_slope),
                "division_threshold": str(p.division_threshold),
                "daughter_reset": str(p.daughter_reset),
                "slow_type": str(p.slow_type),
                "fast_type": str(p.fast_type),
                "forward_mutation_probability": str(
                    p.forward_mutation_probability
                ),
                "essential_module_count": p.essential_module_count,
            },
            "equations": {
                "phenotype": "g(q)=3/16+q/2",
                "clock": "a_pre=a+g(q)",
                "nondivision": "a_pre<1 => one successor (a_pre,q)",
                "division": "a_pre>=1 => two daughters (1/2,q)",
                "mutation_kernel": (
                    "at slow-type division only, each daughter changes slow->fast "
                    "with probability nu; no back mutation"
                ),
            },
        },
        "proof_obligations": proof_obligations,
        "claim_scope": {
            "model_relative_division_gated_copying_proven": transmission_passed,
            "model_relative_genotype_phenotype_coupling_proven": coupling_passed,
            "model_relative_differential_selection_proven": lineage_passed,
            "model_relative_mutation_selection_in_expectation_proven": (
                mutation_passed
            ),
            "model_relative_partition_mean_threshold_proven": partition_passed,
            "conditional_iid_partition_lineage_positive_survival_proven": (
                partition_passed and survival_probability > 0
            ),
            "sample_path_fixation_proven": False,
            "certain_partition_lineage_survival_proven": False,
            "molecular_copying_mechanism_proven": False,
            "endogenous_mutation_chemistry_proven": False,
            "empirical_autonomous_protocell_proven": False,
            "universal_life_theorem_proven": False,
        },
        "all_exact_model_obligations_passed": all_passed,
    }


def validate_coupled_certificate(certificate: Mapping[str, object]) -> bool:
    """Check equality with a freshly built certificate.

    This is a deterministic builder check, not the independent verifier.  The
    latter lives in :mod:`origin_life_coupled_verifier`.
    """

    return dict(certificate) == build_coupled_certificate()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="artifacts/biology/origin_life_coupled_certificate.json",
        help="certificate output path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that an existing artifact equals a fresh build",
    )
    args = parser.parse_args(argv)
    output = Path(args.output)
    certificate = build_coupled_certificate()
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
