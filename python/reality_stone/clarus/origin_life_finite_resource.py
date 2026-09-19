"""Finite-resource sample paths for the first protocell reconstruction loop.

The older origin-life certificates solve an age-structured branching model in
which every complete daughter is assumed to restore all essential genome
modules before its next division.  This module makes that restoration an
explicit, resource-consuming process.  It also adds individual identities,
parent links, starvation deaths, competition, and exact integer material
accounting.

This remains an engineering model.  Passing its gates does not establish an
autonomous metabolism, a prebiotic synthesis route, or an empirical origin of
life.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class FiniteResourceParameters:
    """Canonical integer-token ecology used by the first reconstruction gate."""

    essential_module_count: int = 7
    target_copies_per_module: int = 4
    target_boundary_units: int = 2
    initial_external_resource: int = 64
    reservoir_resource: int = 4096
    inflow_per_tick: int = 64
    external_capacity: int = 64
    uptake_per_cell_per_tick: int = 8
    maintenance_period_ticks: int = 8
    maintenance_cost: int = 1
    horizon_ticks: int = 64


@dataclass(frozen=True)
class ProtocellCondition:
    """Terms that can be removed in paired engineering ablations."""

    uptake: bool = True
    template_copying: bool = True
    boundary_synthesis: bool = True


FULL_CONDITION = ProtocellCondition()
CONDITIONS = {
    "full": FULL_CONDITION,
    "no_uptake": replace(FULL_CONDITION, uptake=False),
    "no_template_copying": replace(FULL_CONDITION, template_copying=False),
    "no_boundary_synthesis": replace(FULL_CONDITION, boundary_synthesis=False),
}
PARAMETERS = FiniteResourceParameters()


@dataclass
class _Cell:
    cell_id: int
    parent_id: int | None
    lineage_id: int
    lineage_path: tuple[int, ...]
    generation: int
    age_ticks: int
    module_copies: list[int]
    boundary_units: int
    energy_units: int


def _validate_parameters(parameters: FiniteResourceParameters) -> None:
    integer_fields = asdict(parameters)
    for name, value in integer_fields.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    positive = {
        "essential_module_count",
        "target_copies_per_module",
        "target_boundary_units",
        "external_capacity",
        "uptake_per_cell_per_tick",
        "maintenance_period_ticks",
        "horizon_ticks",
    }
    for name in positive:
        if integer_fields[name] < 1:
            raise ValueError(f"{name} must be positive")
    nonnegative = set(integer_fields) - positive
    for name in nonnegative:
        if integer_fields[name] < 0:
            raise ValueError(f"{name} must be nonnegative")
    if parameters.initial_external_resource > parameters.external_capacity:
        raise ValueError("initial_external_resource exceeds external_capacity")
    if parameters.target_copies_per_module < 2:
        raise ValueError("division requires at least two copies per module")
    if parameters.target_boundary_units != 2:
        raise ValueError("the current two-daughter schema requires two boundary units")


def _cell_mass(cell: _Cell) -> int:
    return sum(cell.module_copies) + cell.boundary_units + cell.energy_units


def _total_mass(
    cells: Mapping[int, _Cell],
    *,
    external_resource: int,
    reservoir_resource: int,
    waste: int,
) -> int:
    return (
        external_resource
        + reservoir_resource
        + waste
        + sum(_cell_mass(cell) for cell in cells.values())
    )


def _offspring_probabilities(
    module_count: int,
    copies_per_module: int,
) -> tuple[Fraction, Fraction, Fraction]:
    """Exact P(0), P(1), P(2) complete daughters under fair partition."""

    if module_count < 1 or copies_per_module < 1:
        raise ValueError("module and copy counts must be positive")
    one_daughter_misses_a_module = Fraction(1, 2) ** copies_per_module
    specified_complete = (1 - one_daughter_misses_a_module) ** module_count
    both_complete = (1 - 2 * one_daughter_misses_a_module) ** module_count
    probability_two = both_complete
    probability_one = 2 * (specified_complete - both_complete)
    probability_zero = 1 - probability_one - probability_two
    return probability_zero, probability_one, probability_two


def finite_generation_survival_probability(
    generations: int,
    *,
    module_count: int = PARAMETERS.essential_module_count,
    copies_per_module: int = PARAMETERS.target_copies_per_module,
) -> Fraction:
    """Return exact P(Z_generations > 0) for the low-density branching oracle."""

    if generations < 0:
        raise ValueError("generations must be nonnegative")
    probability_zero, probability_one, probability_two = _offspring_probabilities(
        module_count,
        copies_per_module,
    )
    extinct = Fraction(0)
    for _ in range(generations):
        extinct = (
            probability_zero
            + probability_one * extinct
            + probability_two * extinct**2
        )
    return 1 - extinct


def _record_event(
    events: list[dict[str, object]],
    *,
    tick: int,
    kind: str,
    cell: _Cell,
    **payload: object,
) -> None:
    events.append(
        {
            "tick": tick,
            "kind": kind,
            "cell_id": cell.cell_id,
            "parent_id": cell.parent_id,
            "lineage_id": cell.lineage_id,
            "lineage_path": list(cell.lineage_path),
            "generation": cell.generation,
            **payload,
        }
    )


def _stream_seed(*parts: object) -> int:
    """Return a stable counter-style seed independent of process hash state."""

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _ordered_cell_ids(
    cells: Mapping[int, _Cell],
    *,
    master_seed: int,
    tick: int,
    event_kind: str,
) -> list[int]:
    return sorted(
        cells,
        key=lambda cell_id: (
            _stream_seed(
                master_seed,
                tick,
                event_kind,
                cells[cell_id].lineage_id,
                *cells[cell_id].lineage_path,
            ),
            cells[cell_id].lineage_path,
        ),
    )


def _partition_modules(
    module_copies: Sequence[int],
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    daughters = ([0] * len(module_copies), [0] * len(module_copies))
    for module_index, copies in enumerate(module_copies):
        for _ in range(copies):
            daughters[rng.randrange(2)][module_index] += 1
    return daughters


def sample_complete_daughter_count(
    seed: int,
    *,
    module_count: int = PARAMETERS.essential_module_count,
    copies_per_module: int = PARAMETERS.target_copies_per_module,
) -> int:
    """Sample the same fair module partition kernel used by the lineage runner."""

    if module_count < 1 or copies_per_module < 1:
        raise ValueError("module and copy counts must be positive")
    rng = random.Random(_stream_seed(seed, "registered_partition_sampler"))
    daughters = _partition_modules([copies_per_module] * module_count, rng)
    return sum(min(modules) >= 1 for modules in daughters)


def simulate_finite_resource_lineage(
    *,
    seed: int,
    founder_count: int = 1,
    parameters: FiniteResourceParameters = PARAMETERS,
    condition: ProtocellCondition = FULL_CONDITION,
    include_events: bool = False,
) -> dict[str, object]:
    """Run one exact-integer protocell ecology sample path.

    Nutrient tokens move from a finite reservoir to the external pool, into a
    cell's energy store, and then into genome modules, membrane, or waste.
    Division only redistributes existing tokens.  Consequently

    ``reservoir + external + waste + sum(cell material)``

    is an exact invariant at every tick.
    """

    _validate_parameters(parameters)
    if not isinstance(founder_count, int) or isinstance(founder_count, bool):
        raise ValueError("founder_count must be an integer")
    if founder_count < 1:
        raise ValueError("founder_count must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    cells: dict[int, _Cell] = {}
    next_cell_id = 0
    for lineage_id in range(founder_count):
        cells[next_cell_id] = _Cell(
            cell_id=next_cell_id,
            parent_id=None,
            lineage_id=lineage_id,
            lineage_path=(lineage_id,),
            generation=0,
            age_ticks=0,
            module_copies=[1] * parameters.essential_module_count,
            boundary_units=1,
            energy_units=0,
        )
        next_cell_id += 1

    external_resource = parameters.initial_external_resource
    reservoir_resource = parameters.reservoir_resource
    waste = 0
    initial_total_mass = _total_mass(
        cells,
        external_resource=external_resource,
        reservoir_resource=reservoir_resource,
        waste=waste,
    )
    minimum_complete_cell_mass = parameters.essential_module_count + 1
    initial_template_units = founder_count * parameters.essential_module_count
    initial_boundary_units = founder_count
    population_mass_bound = initial_total_mass // minimum_complete_cell_mass
    events: list[dict[str, object]] = []
    division_events = 0
    complete_daughters = 0
    incomplete_daughters = 0
    starvation_deaths = 0
    max_population = len(cells)
    max_generation = 0
    mass_balance_error_max = 0
    template_synthesis_events = 0
    boundary_synthesis_events = 0
    template_units_to_waste = 0
    boundary_units_to_waste = 0
    module_partition_residual_max = 0
    population_history = [len(cells)]

    for tick in range(1, parameters.horizon_ticks + 1):
        inflow = min(
            parameters.inflow_per_tick,
            parameters.external_capacity - external_resource,
            reservoir_resource,
        )
        reservoir_resource -= inflow
        external_resource += inflow

        cell_order = _ordered_cell_ids(
            cells,
            master_seed=seed,
            tick=tick,
            event_kind="uptake_and_metabolism_order",
        )
        if condition.uptake:
            for cell_id in cell_order:
                cell = cells[cell_id]
                uptake = min(parameters.uptake_per_cell_per_tick, external_resource)
                external_resource -= uptake
                cell.energy_units += uptake

        for cell_id in cell_order:
            cell = cells.get(cell_id)
            if cell is None:
                continue
            cell.age_ticks += 1
            if cell.age_ticks % parameters.maintenance_period_ticks == 0:
                if cell.energy_units < parameters.maintenance_cost:
                    starvation_deaths += 1
                    _record_event(events, tick=tick, kind="starvation", cell=cell)
                    template_units_to_waste += sum(cell.module_copies)
                    boundary_units_to_waste += cell.boundary_units
                    waste += _cell_mass(cell)
                    del cells[cell_id]
                    continue
                cell.energy_units -= parameters.maintenance_cost
                waste += parameters.maintenance_cost

            if condition.template_copying:
                module_order = sorted(
                    range(parameters.essential_module_count),
                    key=lambda module_index: _stream_seed(
                        seed,
                        tick,
                        "copy_order",
                        cell.lineage_id,
                        *cell.lineage_path,
                        module_index,
                    ),
                )
                for module_index in module_order:
                    copies = cell.module_copies[module_index]
                    needed = parameters.target_copies_per_module - copies
                    converted = min(needed, cell.energy_units)
                    cell.module_copies[module_index] += converted
                    cell.energy_units -= converted
                    template_synthesis_events += converted
                    if cell.energy_units == 0:
                        break

            if condition.boundary_synthesis:
                needed = parameters.target_boundary_units - cell.boundary_units
                converted = min(needed, cell.energy_units)
                cell.boundary_units += converted
                cell.energy_units -= converted
                boundary_synthesis_events += converted

        division_order = _ordered_cell_ids(
            cells,
            master_seed=seed,
            tick=tick,
            event_kind="division_order",
        )
        for cell_id in division_order:
            parent = cells.get(cell_id)
            if parent is None:
                continue
            ready = (
                min(parent.module_copies) >= parameters.target_copies_per_module
                and parent.boundary_units >= parameters.target_boundary_units
            )
            if not ready:
                continue

            partition_seed = _stream_seed(
                seed,
                "cell_partition",
                parent.lineage_id,
                *parent.lineage_path,
                parent.generation,
            )
            partition_rng = random.Random(partition_seed)
            daughter_modules = _partition_modules(
                parent.module_copies,
                partition_rng,
            )
            daughter_energy = [0, 0]
            for _ in range(parent.energy_units):
                daughter_energy[partition_rng.randrange(2)] += 1
            module_partition_residual = sum(
                abs(
                    daughter_modules[0][module_index]
                    + daughter_modules[1][module_index]
                    - parent.module_copies[module_index]
                )
                for module_index in range(parameters.essential_module_count)
            )
            module_partition_residual_max = max(
                module_partition_residual_max,
                module_partition_residual,
            )

            del cells[cell_id]
            division_events += 1
            daughter_ids: list[int] = []
            daughter_complete: list[bool] = []
            for daughter_index in range(2):
                modules = daughter_modules[daughter_index]
                complete = min(modules) >= 1
                daughter_complete.append(complete)
                if complete:
                    daughter = _Cell(
                        cell_id=next_cell_id,
                        parent_id=parent.cell_id,
                        lineage_id=parent.lineage_id,
                        lineage_path=parent.lineage_path + (daughter_index,),
                        generation=parent.generation + 1,
                        age_ticks=0,
                        module_copies=modules,
                        boundary_units=1,
                        energy_units=daughter_energy[daughter_index],
                    )
                    cells[next_cell_id] = daughter
                    daughter_ids.append(next_cell_id)
                    next_cell_id += 1
                    complete_daughters += 1
                    max_generation = max(max_generation, daughter.generation)
                else:
                    template_units_to_waste += sum(modules)
                    boundary_units_to_waste += 1
                    waste += sum(modules) + 1 + daughter_energy[daughter_index]
                    incomplete_daughters += 1
            _record_event(
                events,
                tick=tick,
                kind="division",
                cell=parent,
                daughter_ids=daughter_ids,
                daughter_complete=daughter_complete,
                parent_module_copies=list(parent.module_copies),
                daughter_module_copies=daughter_modules,
                partition_rng_key=f"{partition_seed:016x}",
            )

        observed_mass = _total_mass(
            cells,
            external_resource=external_resource,
            reservoir_resource=reservoir_resource,
            waste=waste,
        )
        mass_error = abs(observed_mass - initial_total_mass)
        mass_balance_error_max = max(mass_balance_error_max, mass_error)
        if mass_error:
            raise RuntimeError("integer material balance was violated")
        max_population = max(max_population, len(cells))
        if len(cells) > population_mass_bound:
            raise RuntimeError("population exceeded its finite material bound")
        population_history.append(len(cells))

    alive_lineages = sorted({cell.lineage_id for cell in cells.values()})
    final_template_units = sum(sum(cell.module_copies) for cell in cells.values())
    final_boundary_units = sum(cell.boundary_units for cell in cells.values())
    template_ledger_residual = (
        final_template_units
        + template_units_to_waste
        - initial_template_units
        - template_synthesis_events
    )
    boundary_ledger_residual = (
        final_boundary_units
        + boundary_units_to_waste
        - initial_boundary_units
        - boundary_synthesis_events
    )
    result: dict[str, object] = {
        "seed": seed,
        "founder_count": founder_count,
        "condition": asdict(condition),
        "horizon_ticks": parameters.horizon_ticks,
        "initial_total_mass": initial_total_mass,
        "final_total_mass": _total_mass(
            cells,
            external_resource=external_resource,
            reservoir_resource=reservoir_resource,
            waste=waste,
        ),
        "mass_balance_error_max": mass_balance_error_max,
        "template_ledger_residual": template_ledger_residual,
        "boundary_ledger_residual": boundary_ledger_residual,
        "module_partition_residual_max": module_partition_residual_max,
        "template_synthesis_events": template_synthesis_events,
        "boundary_synthesis_events": boundary_synthesis_events,
        "population_mass_bound": population_mass_bound,
        "max_population": max_population,
        "final_population": len(cells),
        "max_generation": max_generation,
        "division_events": division_events,
        "complete_daughters": complete_daughters,
        "incomplete_daughters": incomplete_daughters,
        "starvation_deaths": starvation_deaths,
        "alive_lineages": alive_lineages,
        "final_external_resource": external_resource,
        "final_reservoir_resource": reservoir_resource,
        "final_waste": waste,
        "population_history": population_history,
    }
    if include_events:
        result["events"] = events
    return result


def _fraction(value: Fraction) -> dict[str, str | float]:
    return {"exact": str(value), "decimal": float(value)}


def _panel_summary(runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not runs:
        raise ValueError("panel must not be empty")
    return {
        "runs": len(runs),
        "seeds": [int(run["seed"]) for run in runs],
        "max_mass_balance_error": max(int(run["mass_balance_error_max"]) for run in runs),
        "max_template_ledger_residual": max(
            abs(int(run["template_ledger_residual"])) for run in runs
        ),
        "max_boundary_ledger_residual": max(
            abs(int(run["boundary_ledger_residual"])) for run in runs
        ),
        "max_module_partition_residual": max(
            int(run["module_partition_residual_max"]) for run in runs
        ),
        "mean_divisions_per_founder": sum(
            int(run["division_events"]) / int(run["founder_count"]) for run in runs
        )
        / len(runs),
        "mean_final_population_per_founder": sum(
            int(run["final_population"]) / int(run["founder_count"]) for run in runs
        )
        / len(runs),
        "mean_peak_population_per_founder": sum(
            int(run["max_population"]) / int(run["founder_count"]) for run in runs
        )
        / len(runs),
        "max_population": max(int(run["max_population"]) for run in runs),
        "max_generation": max(int(run["max_generation"]) for run in runs),
        "runs_reaching_generation_3": sum(
            int(run["max_generation"]) >= 3 for run in runs
        ),
        "runs_with_division": sum(int(run["division_events"]) > 0 for run in runs),
        "runs_extinct_at_horizon": sum(int(run["final_population"]) == 0 for run in runs),
        "total_starvation_deaths": sum(int(run["starvation_deaths"]) for run in runs),
    }


def build_finite_resource_certificate() -> dict[str, object]:
    """Build the deterministic certificate for reconstruction loop stage P0."""

    parameters = PARAMETERS
    seeds = list(range(32))
    full_runs = [
        simulate_finite_resource_lineage(seed=seed, parameters=parameters)
        for seed in seeds
    ]
    full_summary = _panel_summary(full_runs)

    ablation_summaries: dict[str, dict[str, object]] = {}
    for name in ("no_uptake", "no_template_copying", "no_boundary_synthesis"):
        runs = [
            simulate_finite_resource_lineage(
                seed=seed,
                parameters=parameters,
                condition=CONDITIONS[name],
            )
            for seed in seeds[:8]
        ]
        ablation_summaries[name] = _panel_summary(runs)

    starvation_parameters = replace(
        parameters,
        initial_external_resource=0,
        reservoir_resource=0,
        inflow_per_tick=0,
        horizon_ticks=parameters.maintenance_period_ticks,
    )
    starvation_runs = [
        simulate_finite_resource_lineage(
            seed=seed,
            parameters=starvation_parameters,
        )
        for seed in seeds[:8]
    ]
    starvation_summary = _panel_summary(starvation_runs)

    competition_parameters = replace(
        parameters,
        initial_external_resource=16,
        reservoir_resource=512,
        inflow_per_tick=16,
        external_capacity=16,
        horizon_ticks=48,
    )
    low_density_runs = [
        simulate_finite_resource_lineage(
            seed=seed,
            founder_count=2,
            parameters=competition_parameters,
        )
        for seed in seeds[:16]
    ]
    high_density_runs = [
        simulate_finite_resource_lineage(
            seed=seed,
            founder_count=16,
            parameters=competition_parameters,
        )
        for seed in seeds[:16]
    ]
    low_density_summary = _panel_summary(low_density_runs)
    high_density_summary = _panel_summary(high_density_runs)

    probability_zero, probability_one, probability_two = _offspring_probabilities(
        parameters.essential_module_count,
        parameters.target_copies_per_module,
    )
    expected_daughters = probability_one + 2 * probability_two
    generation_3_survival = finite_generation_survival_probability(3)
    observed_generation_3_survival = (
        int(full_summary["runs_reaching_generation_3"]) / int(full_summary["runs"])
    )
    sampler_draws = 8192
    sampler_counts = [0, 0, 0]
    for sampler_seed in range(sampler_draws):
        sampler_counts[sample_complete_daughter_count(sampler_seed)] += 1
    sampler_probabilities = [count / sampler_draws for count in sampler_counts]
    exact_probabilities = [
        float(probability_zero),
        float(probability_one),
        float(probability_two),
    ]
    sampler_max_absolute_error = max(
        abs(observed - expected)
        for observed, expected in zip(sampler_probabilities, exact_probabilities)
    )
    sampler_tolerance = 0.02

    partition_passed = (
        probability_zero + probability_one + probability_two == 1
        and min(probability_zero, probability_one, probability_two) >= 0
        and expected_daughters > 1
        and sampler_max_absolute_error <= sampler_tolerance
    )
    conservation_passed = (
        int(full_summary["max_mass_balance_error"]) == 0
        and all(
            int(summary["max_mass_balance_error"]) == 0
            for summary in ablation_summaries.values()
        )
        and int(starvation_summary["max_mass_balance_error"]) == 0
        and int(low_density_summary["max_mass_balance_error"]) == 0
        and int(high_density_summary["max_mass_balance_error"]) == 0
        and all(
            int(summary[field]) == 0
            for summary in (
                full_summary,
                *ablation_summaries.values(),
                starvation_summary,
                low_density_summary,
                high_density_summary,
            )
            for field in (
                "max_template_ledger_residual",
                "max_boundary_ledger_residual",
                "max_module_partition_residual",
            )
        )
    )
    recurrence_passed = (
        int(full_summary["runs_reaching_generation_3"]) >= 20
        and int(full_summary["max_generation"]) >= 3
        and observed_generation_3_survival >= 0.60
        and abs(observed_generation_3_survival - float(generation_3_survival)) <= 0.20
    )
    ablations_passed = all(
        int(summary["runs_with_division"]) == 0
        for summary in ablation_summaries.values()
    )
    starvation_passed = (
        int(starvation_summary["runs_extinct_at_horizon"])
        == int(starvation_summary["runs"])
        and int(starvation_summary["total_starvation_deaths"])
        == int(starvation_summary["runs"])
    )
    competition_passed = (
        float(high_density_summary["mean_divisions_per_founder"])
        < float(low_density_summary["mean_divisions_per_founder"])
        and float(high_density_summary["mean_peak_population_per_founder"])
        < float(low_density_summary["mean_peak_population_per_founder"])
    )
    finite_bound_passed = all(
        int(run["max_population"]) <= int(run["population_mass_bound"])
        for run in (
            full_runs
            + starvation_runs
            + low_density_runs
            + high_density_runs
        )
    )

    gates = {
        "exact_partition_bridge": {
            "passed": partition_passed,
            "P_X_0": _fraction(probability_zero),
            "P_X_1": _fraction(probability_one),
            "P_X_2": _fraction(probability_two),
            "expected_complete_daughters": _fraction(expected_daughters),
            "low_density_generation_3_survival": _fraction(generation_3_survival),
            "sampler_draws": sampler_draws,
            "sampler_counts_P0_P1_P2": sampler_counts,
            "sampler_probabilities_P0_P1_P2": sampler_probabilities,
            "sampler_max_absolute_error": sampler_max_absolute_error,
            "sampler_tolerance": sampler_tolerance,
        },
        "integer_material_conservation": {
            "passed": conservation_passed,
            "invariant": (
                "reservoir + external + waste + "
                "sum(cell_energy + cell_boundary + sum(cell_module_copies))"
            ),
            "max_balance_error": int(full_summary["max_mass_balance_error"]),
            "max_template_ledger_residual": int(
                full_summary["max_template_ledger_residual"]
            ),
            "max_boundary_ledger_residual": int(
                full_summary["max_boundary_ledger_residual"]
            ),
        },
        "explicit_copy_recurrence": {
            "passed": recurrence_passed,
            "panel": full_summary,
            "observed_generation_3_survival": observed_generation_3_survival,
            "required_generation_3_runs": 20,
            "automatic_copy_reset_used": not (
                int(full_summary["max_template_ledger_residual"]) == 0
                and int(full_summary["max_module_partition_residual"]) == 0
            ),
        },
        "paired_single_term_ablations": {
            "passed": ablations_passed,
            "panels": ablation_summaries,
        },
        "starvation_death": {
            "passed": starvation_passed,
            "panel": starvation_summary,
        },
        "density_competition": {
            "passed": competition_passed,
            "low_density": low_density_summary,
            "high_density": high_density_summary,
        },
        "finite_material_population_bound": {
            "passed": finite_bound_passed,
            "minimum_complete_cell_mass": (
                parameters.essential_module_count + 1
            ),
            "bound_definition": "floor(initial_total_material/minimum_complete_cell_mass)",
        },
    }
    all_passed = all(bool(gate["passed"]) for gate in gates.values())
    return {
        "artifact_type": "clarus_finite_resource_protocell_engineering_certificate",
        "artifact_version": 2,
        "arithmetic": "integer token conservation plus exact rational partition oracle",
        "model": {
            "parameters": asdict(parameters),
            "state": (
                "finite reservoir/external/waste plus individual cells with "
                "lineage, parent, age, energy, boundary, and per-module copies"
            ),
            "division_rule": (
                "division requires four copies of each of seven essential modules "
                "and two boundary units; every copy partitions independently and "
                "incomplete daughters become waste"
            ),
            "resource_rule": (
                "all uptake and synthesis are integer transfers from a finite "
                "initial material budget; no token is created"
            ),
        },
        "gates": gates,
        "claim_scope": {
            "finite_resource_accounting_proven_for_declared_simulator": conservation_passed,
            "explicit_token_consuming_copy_bookkeeping_implemented": recurrence_passed,
            "finite_horizon_multigeneration_lineage_observed": recurrence_passed,
            "density_competition_observed_in_registered_seed_panel": competition_passed,
            "starvation_death_observed_in_registered_seed_panel": starvation_passed,
            "prior_low_density_partition_kernel_recovered": partition_passed,
            "indefinite_survival_proven": False,
            "autonomous_metabolism_proven": False,
            "prebiotic_synthesis_route_proven": False,
            "empirical_autonomous_protocell_proven": False,
            "universal_minimum_life_theorem_proven": False,
        },
        "all_engineering_gates_passed": all_passed,
    }


def validate_finite_resource_certificate(certificate: Mapping[str, object]) -> bool:
    """Check byte-structure equality with a fresh deterministic build."""

    return dict(certificate) == build_finite_resource_certificate()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="artifacts/biology/origin_life_finite_resource_certificate.json",
    )
    parser.add_argument("--verify")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(argv)
    certificate = build_finite_resource_certificate()
    if args.verify:
        observed = json.loads(Path(args.verify).read_text(encoding="utf-8"))
        return int(observed != certificate)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(certificate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(certificate, ensure_ascii=False, indent=2))
    return int(args.require_pass and not certificate["all_engineering_gates_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
