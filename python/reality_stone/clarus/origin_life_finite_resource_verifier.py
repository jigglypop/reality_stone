"""Independent verifier for the finite-resource protocell certificate.

The verifier deliberately does not import ``origin_life_finite_resource``.  It
reimplements the registered integer-token transition and exact partition
oracle, then compares every certificate section fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence


_PARAMETERS = {
    "essential_module_count": 7,
    "target_copies_per_module": 4,
    "target_boundary_units": 2,
    "initial_external_resource": 64,
    "reservoir_resource": 4096,
    "inflow_per_tick": 64,
    "external_capacity": 64,
    "uptake_per_cell_per_tick": 8,
    "maintenance_period_ticks": 8,
    "maintenance_cost": 1,
    "horizon_ticks": 64,
}
_FULL = {
    "uptake": True,
    "template_copying": True,
    "boundary_synthesis": True,
}


@dataclass
class _Cell:
    cell_id: int
    lineage_id: int
    lineage_path: tuple[int, ...]
    generation: int
    age_ticks: int
    module_copies: list[int]
    boundary_units: int
    energy_units: int


@dataclass(frozen=True)
class FiniteResourceVerificationReport:
    verified: bool
    checks: tuple[str, ...]
    errors: tuple[str, ...]


def _cell_mass(cell: _Cell) -> int:
    return sum(cell.module_copies) + cell.boundary_units + cell.energy_units


def _total_mass(
    cells: Mapping[int, _Cell],
    external: int,
    reservoir: int,
    waste: int,
) -> int:
    return external + reservoir + waste + sum(_cell_mass(cell) for cell in cells.values())


def _stream_seed(*parts: object) -> int:
    """Return the registered stable counter-style seed."""

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


def _sample_complete_daughter_count(
    seed: int,
    *,
    module_count: int = _PARAMETERS["essential_module_count"],
    copies_per_module: int = _PARAMETERS["target_copies_per_module"],
) -> int:
    rng = random.Random(_stream_seed(seed, "registered_partition_sampler"))
    daughters = _partition_modules([copies_per_module] * module_count, rng)
    return sum(min(modules) >= 1 for modules in daughters)


def _simulate(
    *,
    seed: int,
    founders: int,
    parameters: Mapping[str, int],
    condition: Mapping[str, bool],
) -> dict[str, int]:
    """Independent compact implementation of the registered transition."""

    module_count = parameters["essential_module_count"]
    target_copies = parameters["target_copies_per_module"]
    target_boundary = parameters["target_boundary_units"]
    cells: dict[int, _Cell] = {}
    next_id = 0
    for lineage_id in range(founders):
        cells[next_id] = _Cell(
            cell_id=next_id,
            lineage_id=lineage_id,
            lineage_path=(lineage_id,),
            generation=0,
            age_ticks=0,
            module_copies=[1] * module_count,
            boundary_units=1,
            energy_units=0,
        )
        next_id += 1

    external = parameters["initial_external_resource"]
    reservoir = parameters["reservoir_resource"]
    waste = 0
    initial_mass = _total_mass(cells, external, reservoir, waste)
    population_bound = initial_mass // (module_count + 1)
    initial_template_units = founders * module_count
    initial_boundary_units = founders
    divisions = 0
    starvation = 0
    max_population = founders
    max_generation = 0
    template_synthesis_events = 0
    boundary_synthesis_events = 0
    template_units_to_waste = 0
    boundary_units_to_waste = 0
    module_partition_residual_max = 0

    for tick in range(1, parameters["horizon_ticks"] + 1):
        inflow = min(
            parameters["inflow_per_tick"],
            parameters["external_capacity"] - external,
            reservoir,
        )
        external += inflow
        reservoir -= inflow

        order = _ordered_cell_ids(
            cells,
            master_seed=seed,
            tick=tick,
            event_kind="uptake_and_metabolism_order",
        )
        if condition["uptake"]:
            for cell_id in order:
                uptake = min(parameters["uptake_per_cell_per_tick"], external)
                external -= uptake
                cells[cell_id].energy_units += uptake

        for cell_id in order:
            cell = cells.get(cell_id)
            if cell is None:
                continue
            cell.age_ticks += 1
            if cell.age_ticks % parameters["maintenance_period_ticks"] == 0:
                cost = parameters["maintenance_cost"]
                if cell.energy_units < cost:
                    starvation += 1
                    template_units_to_waste += sum(cell.module_copies)
                    boundary_units_to_waste += cell.boundary_units
                    waste += _cell_mass(cell)
                    del cells[cell_id]
                    continue
                cell.energy_units -= cost
                waste += cost

            if condition["template_copying"]:
                module_order = sorted(
                    range(module_count),
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
                    converted = min(target_copies - copies, cell.energy_units)
                    cell.module_copies[module_index] += converted
                    cell.energy_units -= converted
                    template_synthesis_events += converted
                    if cell.energy_units == 0:
                        break
            if condition["boundary_synthesis"]:
                converted = min(target_boundary - cell.boundary_units, cell.energy_units)
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
            if not (
                min(parent.module_copies) >= target_copies
                and parent.boundary_units >= target_boundary
            ):
                continue

            partition_seed = _stream_seed(
                seed,
                "cell_partition",
                parent.lineage_id,
                *parent.lineage_path,
                parent.generation,
            )
            partition_rng = random.Random(partition_seed)
            module_partition = _partition_modules(
                parent.module_copies,
                partition_rng,
            )
            energy_partition = [0, 0]
            for _ in range(parent.energy_units):
                energy_partition[partition_rng.randrange(2)] += 1
            module_partition_residual = sum(
                abs(
                    module_partition[0][module_index]
                    + module_partition[1][module_index]
                    - parent.module_copies[module_index]
                )
                for module_index in range(module_count)
            )
            module_partition_residual_max = max(
                module_partition_residual_max,
                module_partition_residual,
            )

            del cells[cell_id]
            divisions += 1
            for daughter_index in range(2):
                modules = module_partition[daughter_index]
                if min(modules) >= 1:
                    daughter = _Cell(
                        cell_id=next_id,
                        lineage_id=parent.lineage_id,
                        lineage_path=parent.lineage_path + (daughter_index,),
                        generation=parent.generation + 1,
                        age_ticks=0,
                        module_copies=modules,
                        boundary_units=1,
                        energy_units=energy_partition[daughter_index],
                    )
                    cells[next_id] = daughter
                    next_id += 1
                    max_generation = max(max_generation, daughter.generation)
                else:
                    template_units_to_waste += sum(modules)
                    boundary_units_to_waste += 1
                    waste += sum(modules) + 1 + energy_partition[daughter_index]

        if _total_mass(cells, external, reservoir, waste) != initial_mass:
            raise AssertionError("reference material ledger failed")
        if len(cells) > population_bound:
            raise AssertionError("reference population bound failed")
        max_population = max(max_population, len(cells))

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

    return {
        "seed": seed,
        "founder_count": founders,
        "mass_balance_error_max": 0,
        "template_ledger_residual": template_ledger_residual,
        "boundary_ledger_residual": boundary_ledger_residual,
        "module_partition_residual_max": module_partition_residual_max,
        "template_synthesis_events": template_synthesis_events,
        "boundary_synthesis_events": boundary_synthesis_events,
        "population_mass_bound": population_bound,
        "max_population": max_population,
        "final_population": len(cells),
        "max_generation": max_generation,
        "division_events": divisions,
        "starvation_deaths": starvation,
    }


def _summary(runs: Sequence[Mapping[str, int]]) -> dict[str, object]:
    return {
        "runs": len(runs),
        "seeds": [run["seed"] for run in runs],
        "max_mass_balance_error": max(run["mass_balance_error_max"] for run in runs),
        "max_template_ledger_residual": max(
            abs(run["template_ledger_residual"]) for run in runs
        ),
        "max_boundary_ledger_residual": max(
            abs(run["boundary_ledger_residual"]) for run in runs
        ),
        "max_module_partition_residual": max(
            run["module_partition_residual_max"] for run in runs
        ),
        "mean_divisions_per_founder": sum(
            run["division_events"] / run["founder_count"] for run in runs
        )
        / len(runs),
        "mean_final_population_per_founder": sum(
            run["final_population"] / run["founder_count"] for run in runs
        )
        / len(runs),
        "mean_peak_population_per_founder": sum(
            run["max_population"] / run["founder_count"] for run in runs
        )
        / len(runs),
        "max_population": max(run["max_population"] for run in runs),
        "max_generation": max(run["max_generation"] for run in runs),
        "runs_reaching_generation_3": sum(run["max_generation"] >= 3 for run in runs),
        "runs_with_division": sum(run["division_events"] > 0 for run in runs),
        "runs_extinct_at_horizon": sum(run["final_population"] == 0 for run in runs),
        "total_starvation_deaths": sum(run["starvation_deaths"] for run in runs),
    }


def _partition() -> tuple[Fraction, Fraction, Fraction]:
    modules = _PARAMETERS["essential_module_count"]
    copies = _PARAMETERS["target_copies_per_module"]
    miss = Fraction(1, 2) ** copies
    specified = (1 - miss) ** modules
    both = (1 - 2 * miss) ** modules
    return 1 - 2 * specified + both, 2 * (specified - both), both


def _finite_survival(generations: int) -> Fraction:
    p0, p1, p2 = _partition()
    extinct = Fraction(0)
    for _ in range(generations):
        extinct = p0 + p1 * extinct + p2 * extinct**2
    return 1 - extinct


def _fraction(value: Fraction) -> dict[str, str | float]:
    return {"exact": str(value), "decimal": float(value)}


def _expected_certificate() -> dict[str, object]:
    seeds = list(range(32))
    full_runs = [
        _simulate(seed=seed, founders=1, parameters=_PARAMETERS, condition=_FULL)
        for seed in seeds
    ]
    full_summary = _summary(full_runs)

    conditions = {
        "no_uptake": {**_FULL, "uptake": False},
        "no_template_copying": {**_FULL, "template_copying": False},
        "no_boundary_synthesis": {**_FULL, "boundary_synthesis": False},
    }
    ablations = {
        name: _summary(
            [
                _simulate(
                    seed=seed,
                    founders=1,
                    parameters=_PARAMETERS,
                    condition=condition,
                )
                for seed in seeds[:8]
            ]
        )
        for name, condition in conditions.items()
    }

    starvation_parameters = {
        **_PARAMETERS,
        "initial_external_resource": 0,
        "reservoir_resource": 0,
        "inflow_per_tick": 0,
        "horizon_ticks": _PARAMETERS["maintenance_period_ticks"],
    }
    starvation_runs = [
        _simulate(
            seed=seed,
            founders=1,
            parameters=starvation_parameters,
            condition=_FULL,
        )
        for seed in seeds[:8]
    ]
    starvation = _summary(starvation_runs)

    competition_parameters = {
        **_PARAMETERS,
        "initial_external_resource": 16,
        "reservoir_resource": 512,
        "inflow_per_tick": 16,
        "external_capacity": 16,
        "horizon_ticks": 48,
    }
    low_runs = [
        _simulate(
            seed=seed,
            founders=2,
            parameters=competition_parameters,
            condition=_FULL,
        )
        for seed in seeds[:16]
    ]
    high_runs = [
        _simulate(
            seed=seed,
            founders=16,
            parameters=competition_parameters,
            condition=_FULL,
        )
        for seed in seeds[:16]
    ]
    low = _summary(low_runs)
    high = _summary(high_runs)

    p0, p1, p2 = _partition()
    expected_daughters = p1 + 2 * p2
    generation_3_survival = _finite_survival(3)
    observed = full_summary["runs_reaching_generation_3"] / full_summary["runs"]
    sampler_draws = 8192
    sampler_counts = [0, 0, 0]
    for sampler_seed in range(sampler_draws):
        sampler_counts[_sample_complete_daughter_count(sampler_seed)] += 1
    sampler_probabilities = [count / sampler_draws for count in sampler_counts]
    exact_probabilities = [float(p0), float(p1), float(p2)]
    sampler_max_absolute_error = max(
        abs(observed_probability - expected_probability)
        for observed_probability, expected_probability in zip(
            sampler_probabilities,
            exact_probabilities,
        )
    )
    sampler_tolerance = 0.02
    partition_pass = (
        p0 + p1 + p2 == 1
        and min(p0, p1, p2) >= 0
        and expected_daughters > 1
        and sampler_max_absolute_error <= sampler_tolerance
    )
    summaries = [full_summary, *ablations.values(), starvation, low, high]
    conservation_pass = (
        all(summary["max_mass_balance_error"] == 0 for summary in summaries)
        and all(
            summary[field] == 0
            for summary in summaries
            for field in (
                "max_template_ledger_residual",
                "max_boundary_ledger_residual",
                "max_module_partition_residual",
            )
        )
    )
    recurrence_pass = (
        full_summary["runs_reaching_generation_3"] >= 20
        and full_summary["max_generation"] >= 3
        and observed >= 0.60
        and abs(observed - float(generation_3_survival)) <= 0.20
    )
    ablation_pass = all(summary["runs_with_division"] == 0 for summary in ablations.values())
    starvation_pass = (
        starvation["runs_extinct_at_horizon"] == starvation["runs"]
        and starvation["total_starvation_deaths"] == starvation["runs"]
    )
    competition_pass = (
        high["mean_divisions_per_founder"] < low["mean_divisions_per_founder"]
        and high["mean_peak_population_per_founder"]
        < low["mean_peak_population_per_founder"]
    )
    bound_pass = all(
        run["max_population"] <= run["population_mass_bound"]
        for run in [*full_runs, *starvation_runs, *low_runs, *high_runs]
    )

    gates = {
        "exact_partition_bridge": {
            "passed": partition_pass,
            "P_X_0": _fraction(p0),
            "P_X_1": _fraction(p1),
            "P_X_2": _fraction(p2),
            "expected_complete_daughters": _fraction(expected_daughters),
            "low_density_generation_3_survival": _fraction(generation_3_survival),
            "sampler_draws": sampler_draws,
            "sampler_counts_P0_P1_P2": sampler_counts,
            "sampler_probabilities_P0_P1_P2": sampler_probabilities,
            "sampler_max_absolute_error": sampler_max_absolute_error,
            "sampler_tolerance": sampler_tolerance,
        },
        "integer_material_conservation": {
            "passed": conservation_pass,
            "invariant": (
                "reservoir + external + waste + "
                "sum(cell_energy + cell_boundary + sum(cell_module_copies))"
            ),
            "max_balance_error": full_summary["max_mass_balance_error"],
            "max_template_ledger_residual": full_summary[
                "max_template_ledger_residual"
            ],
            "max_boundary_ledger_residual": full_summary[
                "max_boundary_ledger_residual"
            ],
        },
        "explicit_copy_recurrence": {
            "passed": recurrence_pass,
            "panel": full_summary,
            "observed_generation_3_survival": observed,
            "required_generation_3_runs": 20,
            "automatic_copy_reset_used": not (
                full_summary["max_template_ledger_residual"] == 0
                and full_summary["max_module_partition_residual"] == 0
            ),
        },
        "paired_single_term_ablations": {
            "passed": ablation_pass,
            "panels": ablations,
        },
        "starvation_death": {"passed": starvation_pass, "panel": starvation},
        "density_competition": {
            "passed": competition_pass,
            "low_density": low,
            "high_density": high,
        },
        "finite_material_population_bound": {
            "passed": bound_pass,
            "minimum_complete_cell_mass": _PARAMETERS["essential_module_count"] + 1,
            "bound_definition": "floor(initial_total_material/minimum_complete_cell_mass)",
        },
    }
    all_passed = all(gate["passed"] for gate in gates.values())
    return {
        "artifact_type": "clarus_finite_resource_protocell_engineering_certificate",
        "artifact_version": 2,
        "arithmetic": "integer token conservation plus exact rational partition oracle",
        "model": {
            "parameters": dict(_PARAMETERS),
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
            "finite_resource_accounting_proven_for_declared_simulator": conservation_pass,
            "explicit_token_consuming_copy_bookkeeping_implemented": recurrence_pass,
            "finite_horizon_multigeneration_lineage_observed": recurrence_pass,
            "density_competition_observed_in_registered_seed_panel": competition_pass,
            "starvation_death_observed_in_registered_seed_panel": starvation_pass,
            "prior_low_density_partition_kernel_recovered": partition_pass,
            "indefinite_survival_proven": False,
            "autonomous_metabolism_proven": False,
            "prebiotic_synthesis_route_proven": False,
            "empirical_autonomous_protocell_proven": False,
            "universal_minimum_life_theorem_proven": False,
        },
        "all_engineering_gates_passed": all_passed,
    }


def _assert_finite(value: object, path: str = "certificate") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")


def verify_finite_resource_certificate(
    certificate: Mapping[str, object],
) -> FiniteResourceVerificationReport:
    checks: list[str] = []
    errors: list[str] = []
    try:
        if not isinstance(certificate, Mapping):
            raise ValueError("certificate must be a mapping")
        _assert_finite(certificate)
        expected = _expected_certificate()
        expected_keys = set(expected)
        if set(certificate) != expected_keys:
            raise ValueError("top-level schema does not match the registered certificate")
        checks.append("schema_and_finite_numbers")

        for section in (
            "artifact_type",
            "artifact_version",
            "arithmetic",
            "model",
            "gates",
            "claim_scope",
            "all_engineering_gates_passed",
        ):
            if certificate.get(section) != expected[section]:
                raise ValueError(f"{section} differs from independent recomputation")
            checks.append(section)
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return FiniteResourceVerificationReport(
        verified=not errors,
        checks=tuple(checks),
        errors=tuple(errors),
    )


def independently_verified(certificate: Mapping[str, object]) -> bool:
    return verify_finite_resource_certificate(certificate).verified


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate")
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.certificate).read_text(encoding="utf-8"))
    report = verify_finite_resource_certificate(payload)
    print(json.dumps({"verified": report.verified, "checks": report.checks, "errors": report.errors}))
    return int(not report.verified)


if __name__ == "__main__":
    raise SystemExit(main())
