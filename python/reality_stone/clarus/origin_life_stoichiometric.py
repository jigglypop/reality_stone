"""Exact P0.2 stoichiometric, free-energy, and membrane-geometry gates.

The model is deliberately small.  Integer reaction events move two conserved
moieties (material and carrier) through named pools.  A declared standard-state
free-energy potential is converted to an exact heat ledger.  Membrane geometry
is certified with the rational upper bound ``pi < 355/113``; floating-point
geometry is readout only and never decides a gate.

This is a coarse-grained engineering witness.  It is not calibrated chemistry,
an autonomous protocell, or evidence about the historical origin of life.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence


RAW = "R"
PRECURSOR = "P"
CHARGED = "F"
DISCHARGED = "D"
WASTE = "W"
BOUNDARY = "B"
MODULE_COUNT = 7
TEMPLATES = tuple(f"T{index}" for index in range(MODULE_COUNT))
CELL_SPECIES = (RAW, PRECURSOR, CHARGED, DISCHARGED, WASTE, BOUNDARY, *TEMPLATES)
ENVIRONMENT_SPECIES = (RAW, DISCHARGED, WASTE)

# Left-null vectors of the registered stoichiometric matrix.
MATERIAL_WEIGHT = {
    RAW: 1,
    PRECURSOR: 1,
    CHARGED: 0,
    DISCHARGED: 0,
    WASTE: 1,
    BOUNDARY: 2,
    **{template: 4 for template in TEMPLATES},
}
CARRIER_WEIGHT = {
    RAW: 0,
    PRECURSOR: 0,
    CHARGED: 1,
    DISCHARGED: 1,
    WASTE: 0,
    BOUNDARY: 0,
    **{template: 0 for template in TEMPLATES},
}

# A declared standard-state potential in integer energy quanta.  It is a
# bookkeeping potential, not a fitted chemical potential.
STANDARD_FREE_ENERGY_QUANTA = {
    RAW: 8,
    PRECURSOR: 4,
    CHARGED: 4,
    DISCHARGED: 0,
    WASTE: 0,
    BOUNDARY: 4,
    **{template: 8 for template in TEMPLATES},
}

# Exact SI scale manifest.  B is a coarse bilayer patch, not one amphiphile.
VOLUME_QUANTUM_M3 = Fraction(1, 72 * 10**18)
BOUNDARY_PATCH_AREA_M2 = Fraction(31, 10**14)
FREE_ENERGY_QUANTUM_J = Fraction(1, 10**20)
PI_UPPER = Fraction(355, 113)

# Backward-compatible names, now exact Fractions with corrected semantics.
MOLECULAR_VOLUME_M3 = VOLUME_QUANTUM_M3
BOUNDARY_AREA_PER_MOLECULE_M2 = BOUNDARY_PATCH_AREA_M2


@dataclass(frozen=True)
class Reaction:
    """One integer-event reaction in qualified compartment notation."""

    name: str
    reactants: tuple[tuple[str, int], ...]
    products: tuple[tuple[str, int], ...]
    stochastic_rate_per_second: Fraction

    @property
    def rate_constant_per_second(self) -> Fraction:
        """Compatibility alias for the registered stochastic event rate."""

        return self.stochastic_rate_per_second

    @property
    def order(self) -> int:
        return sum(coefficient for _, coefficient in self.reactants)

    @property
    def rate_constant_unit(self) -> str:
        return "s^-1 per available reactant combination"

    @property
    def concentration_rate_constant_unit(self) -> str:
        if self.order == 1:
            return "s^-1"
        return f"m^{3 * (self.order - 1)} molecule^{1 - self.order} s^-1"


def _q(compartment: str, species: str) -> str:
    if compartment not in {"cell", "environment"}:
        raise ValueError("unknown compartment")
    return f"{compartment}:{species}"


def _reaction(
    name: str,
    reactants: Mapping[str, int],
    products: Mapping[str, int],
    rate: Fraction,
) -> Reaction:
    if not reactants or not products:
        raise ValueError("registered reactions need reactants and products")
    for side in (reactants, products):
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in side.values()
        ):
            raise ValueError("stoichiometric coefficients must be positive integers")
    return Reaction(
        name=name,
        reactants=tuple(sorted(reactants.items())),
        products=tuple(sorted(products.items())),
        stochastic_rate_per_second=Fraction(rate),
    )


def canonical_reactions() -> tuple[Reaction, ...]:
    """Return the canonical balanced P0.2 reaction/transport manifest."""

    cell = lambda species: _q("cell", species)
    environment = lambda species: _q("environment", species)
    reactions = [
        _reaction(
            "carrier_recharge",
            {cell(RAW): 1, cell(DISCHARGED): 1},
            {cell(WASTE): 1, cell(CHARGED): 1},
            Fraction(1, 2),
        ),
        _reaction(
            "precursor_activation",
            {cell(RAW): 1, cell(CHARGED): 1},
            {cell(PRECURSOR): 1, cell(DISCHARGED): 1},
            Fraction(1, 4),
        ),
        _reaction(
            "boundary_synthesis",
            {cell(PRECURSOR): 2, cell(CHARGED): 1},
            {cell(BOUNDARY): 1, cell(DISCHARGED): 1},
            Fraction(1, 4),
        ),
        _reaction(
            "maintenance_discharge",
            {cell(CHARGED): 1},
            {cell(DISCHARGED): 1},
            Fraction(1, 64),
        ),
        _reaction(
            "precursor_decay",
            {cell(PRECURSOR): 1},
            {cell(WASTE): 1},
            Fraction(1, 512),
        ),
        _reaction(
            "boundary_decay",
            {cell(BOUNDARY): 1},
            {cell(WASTE): 2},
            Fraction(1, 2048),
        ),
        _reaction(
            "raw_uptake",
            {environment(RAW): 1},
            {cell(RAW): 1},
            Fraction(1, 16),
        ),
        _reaction(
            "discharged_carrier_uptake",
            {environment(DISCHARGED): 1},
            {cell(DISCHARGED): 1},
            Fraction(1, 16),
        ),
        _reaction(
            "waste_export",
            {cell(WASTE): 1},
            {environment(WASTE): 1},
            Fraction(1, 16),
        ),
    ]
    for template in TEMPLATES:
        reactions.extend(
            [
                _reaction(
                    f"copy_{template}",
                    {
                        cell(template): 1,
                        cell(PRECURSOR): 4,
                        cell(CHARGED): 2,
                    },
                    {cell(template): 2, cell(DISCHARGED): 2},
                    Fraction(1, 8),
                ),
                _reaction(
                    f"decay_{template}",
                    {cell(template): 1},
                    {cell(WASTE): 4},
                    Fraction(1, 1024),
                ),
            ]
        )
    return tuple(reactions)


def _base_species(qualified: str) -> str:
    try:
        compartment, species = qualified.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"invalid qualified species {qualified!r}") from exc
    if compartment == "cell":
        allowed = CELL_SPECIES
    elif compartment == "environment":
        allowed = ENVIRONMENT_SPECIES
    else:
        raise ValueError(f"unknown qualified species {qualified!r}")
    if species not in allowed:
        raise ValueError(f"unknown qualified species {qualified!r}")
    return species


def _weighted_total(state: Mapping[str, int], weights: Mapping[str, int]) -> int:
    return sum(weights[_base_species(species)] * count for species, count in state.items())


def material_total(state: Mapping[str, int]) -> int:
    return _weighted_total(state, MATERIAL_WEIGHT)


def carrier_total(state: Mapping[str, int]) -> int:
    return _weighted_total(state, CARRIER_WEIGHT)


def free_energy_total(state: Mapping[str, int]) -> int:
    return _weighted_total(state, STANDARD_FREE_ENERGY_QUANTA)


def reaction_balance(reaction: Reaction) -> tuple[int, int]:
    material = carrier = 0
    for sign, side in ((-1, reaction.reactants), (1, reaction.products)):
        for species, coefficient in side:
            base = _base_species(species)
            material += sign * MATERIAL_WEIGHT[base] * coefficient
            carrier += sign * CARRIER_WEIGHT[base] * coefficient
    return material, carrier


def reaction_free_energy_delta(reaction: Reaction) -> int:
    delta = 0
    for sign, side in ((-1, reaction.reactants), (1, reaction.products)):
        for species, coefficient in side:
            delta += sign * STANDARD_FREE_ENERGY_QUANTA[_base_species(species)] * coefficient
    return delta


def reaction_heat_quanta(reaction: Reaction, *, events: int = 1) -> int:
    if not isinstance(events, int) or isinstance(events, bool) or events < 0:
        raise ValueError("events must be a nonnegative integer")
    delta = reaction_free_energy_delta(reaction)
    if delta > 0:
        raise ValueError(f"{reaction.name} is uphill in the declared potential")
    return -delta * events


def _validate_state(state: Mapping[str, int]) -> None:
    for species, count in state.items():
        _base_species(species)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"{species} must have a nonnegative integer count")


def apply_reaction(
    state: Mapping[str, int],
    reaction: Reaction,
    *,
    events: int = 1,
) -> dict[str, int]:
    """Apply exact events transactionally; shortage is rejected, never clipped."""

    _validate_state(state)
    if not isinstance(events, int) or isinstance(events, bool) or events < 1:
        raise ValueError("events must be a positive integer")
    updated = dict(state)
    for species, coefficient in reaction.reactants:
        required = coefficient * events
        if updated.get(species, 0) < required:
            raise ValueError(f"insufficient reactant {species} for {reaction.name}")
    for species, coefficient in reaction.reactants:
        updated[species] = updated.get(species, 0) - coefficient * events
    for species, coefficient in reaction.products:
        updated[species] = updated.get(species, 0) + coefficient * events
    _validate_state(updated)
    return updated


def _reaction_by_name() -> dict[str, Reaction]:
    return {reaction.name: reaction for reaction in canonical_reactions()}


def _canonical_state() -> dict[str, int]:
    state = {
        **{_q("cell", species): 0 for species in CELL_SPECIES},
        **{_q("environment", species): 0 for species in ENVIRONMENT_SPECIES},
    }
    state.update(
        {
            _q("cell", RAW): 128,
            _q("cell", PRECURSOR): 96,
            _q("cell", CHARGED): 128,
            _q("cell", DISCHARGED): 128,
            _q("cell", BOUNDARY): 80,
            _q("environment", RAW): 256,
            _q("environment", DISCHARGED): 256,
            **{_q("cell", template): 4 for template in TEMPLATES},
        }
    )
    return state


def _apply_schedule(
    state: Mapping[str, int],
    schedule: Sequence[tuple[str, int]],
) -> tuple[dict[str, int], int, int]:
    reactions = _reaction_by_name()
    current = dict(state)
    heat = 0
    maximum_event_energy_residual = 0
    for name, events in schedule:
        reaction = reactions[name]
        before = free_energy_total(current)
        current = apply_reaction(current, reaction, events=events)
        event_heat = reaction_heat_quanta(reaction, events=events)
        after = free_energy_total(current)
        residual = before - after - event_heat
        maximum_event_energy_residual = max(maximum_event_energy_residual, abs(residual))
        heat += event_heat
    return current, heat, maximum_event_energy_residual


def run_closed_ledger() -> dict[str, object]:
    state = _canonical_state()
    initial_state = dict(state)
    initial_material = material_total(state)
    initial_carrier = carrier_total(state)
    initial_energy = free_energy_total(state)
    schedule = [
        ("raw_uptake", 8),
        ("discharged_carrier_uptake", 8),
        ("carrier_recharge", 16),
        ("precursor_activation", 16),
        *[(f"copy_{template}", 2) for template in TEMPLATES],
        ("boundary_synthesis", 8),
        ("maintenance_discharge", 4),
        *[(f"decay_{template}", 1) for template in TEMPLATES],
        ("boundary_decay", 2),
        ("precursor_decay", 2),
        ("waste_export", 10),
    ]
    state, heat, event_energy_error = _apply_schedule(state, schedule)
    final_material = material_total(state)
    final_carrier = carrier_total(state)
    final_energy = free_energy_total(state)
    energy_residual = initial_energy - final_energy - heat
    passed = (
        initial_material == final_material
        and initial_carrier == final_carrier
        and energy_residual == 0
        and event_energy_error == 0
        and min(state.values()) >= 0
    )
    return {
        "passed": passed,
        "event_batches": len(schedule),
        "initial_state": initial_state,
        "schedule": [{"reaction": name, "events": events} for name, events in schedule],
        "initial_material": initial_material,
        "final_material": final_material,
        "material_residual": final_material - initial_material,
        "initial_carrier": initial_carrier,
        "final_carrier": final_carrier,
        "carrier_residual": final_carrier - initial_carrier,
        "initial_standard_free_energy_quanta": initial_energy,
        "final_standard_free_energy_quanta": final_energy,
        "heat_quanta": heat,
        "free_energy_heat_residual": energy_residual,
        "maximum_event_energy_residual": event_energy_error,
        "heat_joule": {
            "exact": str(heat * FREE_ENERGY_QUANTUM_J),
            "decimal": float(heat * FREE_ENERGY_QUANTUM_J),
        },
        "minimum_final_count": min(state.values()),
    }


def _open_inflow(state: Mapping[str, int], species: str, amount: int) -> dict[str, int]:
    if species not in ENVIRONMENT_SPECIES:
        raise ValueError("invalid open-inflow species")
    if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
        raise ValueError("open inflow must be a nonnegative integer")
    updated = dict(state)
    key = _q("environment", species)
    updated[key] = updated.get(key, 0) + amount
    _validate_state(updated)
    return updated


def _open_outflow(state: Mapping[str, int], species: str, amount: int) -> dict[str, int]:
    if species not in ENVIRONMENT_SPECIES:
        raise ValueError("invalid open-outflow species")
    if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
        raise ValueError("open outflow must be a nonnegative integer")
    key = _q("environment", species)
    if state.get(key, 0) < amount:
        raise ValueError("open outflow exceeds environment count")
    updated = dict(state)
    updated[key] -= amount
    _validate_state(updated)
    return updated


def run_open_flow_ledger(*, ticks: int = 16) -> dict[str, object]:
    if not isinstance(ticks, int) or isinstance(ticks, bool) or ticks < 1:
        raise ValueError("ticks must be a positive integer")
    state = _canonical_state()
    initial_state = dict(state)
    initial_material = material_total(state)
    initial_carrier = carrier_total(state)
    initial_energy = free_energy_total(state)
    material_in = material_out = carrier_in = carrier_out = 0
    free_energy_in = free_energy_out = heat = 0
    maximum_event_energy_residual = 0
    intervention = {
        "raw_inflow_per_tick": 4,
        "discharged_carrier_inflow_per_tick": 4,
        "waste_outflow_per_tick": 1,
    }
    schedule_label = [
        "raw_uptake x2",
        "discharged_carrier_uptake x2",
        "carrier_recharge",
        "precursor_activation",
        "copy_T[tick mod 7]",
        "boundary_synthesis",
        "maintenance_discharge",
        "waste_export",
    ]
    for tick in range(ticks):
        for species, amount in (
            (RAW, intervention["raw_inflow_per_tick"]),
            (DISCHARGED, intervention["discharged_carrier_inflow_per_tick"]),
        ):
            state = _open_inflow(state, species, amount)
            material_in += MATERIAL_WEIGHT[species] * amount
            carrier_in += CARRIER_WEIGHT[species] * amount
            free_energy_in += STANDARD_FREE_ENERGY_QUANTA[species] * amount
        schedule = [
            ("raw_uptake", 2),
            ("discharged_carrier_uptake", 2),
            ("carrier_recharge", 1),
            ("precursor_activation", 1),
            (f"copy_{TEMPLATES[tick % MODULE_COUNT]}", 1),
            ("boundary_synthesis", 1),
            ("maintenance_discharge", 1),
            ("waste_export", 1),
        ]
        state, tick_heat, tick_error = _apply_schedule(state, schedule)
        heat += tick_heat
        maximum_event_energy_residual = max(maximum_event_energy_residual, tick_error)
        amount = intervention["waste_outflow_per_tick"]
        state = _open_outflow(state, WASTE, amount)
        material_out += MATERIAL_WEIGHT[WASTE] * amount
        carrier_out += CARRIER_WEIGHT[WASTE] * amount
        free_energy_out += STANDARD_FREE_ENERGY_QUANTA[WASTE] * amount

    final_material = material_total(state)
    final_carrier = carrier_total(state)
    final_energy = free_energy_total(state)
    material_residual = initial_material + material_in - material_out - final_material
    carrier_residual = initial_carrier + carrier_in - carrier_out - final_carrier
    energy_residual = initial_energy + free_energy_in - free_energy_out - final_energy - heat
    passed = (
        material_residual == 0
        and carrier_residual == 0
        and energy_residual == 0
        and maximum_event_energy_residual == 0
        and min(state.values()) >= 0
    )
    return {
        "passed": passed,
        "ticks": ticks,
        "initial_state": initial_state,
        "intervention": intervention,
        "reaction_schedule_per_tick": schedule_label,
        "schedule_constant_after_initialization": True,
        "initial_material": initial_material,
        "final_material": final_material,
        "material_in": material_in,
        "material_out": material_out,
        "material_residual": material_residual,
        "initial_carrier": initial_carrier,
        "final_carrier": final_carrier,
        "carrier_in": carrier_in,
        "carrier_out": carrier_out,
        "carrier_residual": carrier_residual,
        "initial_standard_free_energy_quanta": initial_energy,
        "final_standard_free_energy_quanta": final_energy,
        "standard_free_energy_in": free_energy_in,
        "standard_free_energy_out": free_energy_out,
        "heat_quanta": heat,
        "free_energy_heat_residual": energy_residual,
        "maximum_event_energy_residual": maximum_event_energy_residual,
        "minimum_final_count": min(state.values()),
    }


def _validate_cell_counts(counts: Mapping[str, int]) -> None:
    if set(counts) != set(CELL_SPECIES):
        raise ValueError("cell state requires every registered cell species exactly once")
    for species, count in counts.items():
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"invalid count for {species}")


def _unqualified_total(counts: Mapping[str, int], weights: Mapping[str, int]) -> int:
    return sum(weights[species] * count for species, count in counts.items())


def _geometry_exact(counts: Mapping[str, int]) -> tuple[Fraction, Fraction, int]:
    _validate_cell_counts(counts)
    material = _unqualified_total(counts, MATERIAL_WEIGHT)
    carrier = _unqualified_total(counts, CARRIER_WEIGHT)
    volume_quanta = material + carrier
    if volume_quanta < 1:
        raise ValueError("cell volume must be positive")
    volume = VOLUME_QUANTUM_M3 * volume_quanta
    area = BOUNDARY_PATCH_AREA_M2 * counts[BOUNDARY]
    return volume, area, volume_quanta


def cell_geometry(counts: Mapping[str, int]) -> dict[str, object]:
    """Return exact geometry predicates plus non-normative decimal readouts."""

    volume, area, volume_quanta = _geometry_exact(counts)
    enclosure_lhs = area**3
    enclosure_rhs = 36 * PI_UPPER * volume**2
    division_rhs = 72 * PI_UPPER * volume**2
    sphere_area = (36.0 * math.pi) ** (1.0 / 3.0) * float(volume) ** (2.0 / 3.0)
    return {
        "volume_quanta": volume_quanta,
        "volume_m3": float(volume),
        "volume_m3_exact": str(volume),
        "membrane_area_m2": float(area),
        "membrane_area_m2_exact": str(area),
        "minimum_sphere_area_m2_readout": sphere_area,
        "membrane_slack_readout": float(area) / sphere_area,
        "pi_upper_exact": str(PI_UPPER),
        "enclosure_lhs_A_cubed_exact": str(enclosure_lhs),
        "enclosure_rhs_36_pi_upper_V_squared_exact": str(enclosure_rhs),
        "division_rhs_72_pi_upper_V_squared_exact": str(division_rhs),
        "enclosure_certified": enclosure_lhs >= enclosure_rhs,
        "symmetric_division_geometry_certified": enclosure_lhs >= division_rhs,
        "gate_uses_float": False,
    }


def partition_cell_counts(
    parent: Mapping[str, int],
    *,
    seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """Partition every extensive count; membrane patches split floor/ceiling."""

    _validate_cell_counts(parent)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    first = {species: 0 for species in CELL_SPECIES}
    second = {species: 0 for species in CELL_SPECIES}
    for species, count in parent.items():
        if species == BOUNDARY:
            first[species] = count // 2
            second[species] = count - first[species]
            continue
        for _ in range(count):
            if rng.randrange(2) == 0:
                first[species] += 1
            else:
                second[species] += 1
    return first, second


def run_division_geometry_gate() -> dict[str, object]:
    parent = {
        RAW: 20,
        PRECURSOR: 20,
        CHARGED: 20,
        DISCHARGED: 20,
        WASTE: 20,
        BOUNDARY: 80,
        **{template: 4 for template in TEMPLATES},
    }
    equal_daughter = {species: count // 2 for species, count in parent.items()}
    parent_geometry = cell_geometry(parent)
    daughter_geometry = cell_geometry(equal_daughter)
    parent_volume, parent_area, _ = _geometry_exact(parent)
    daughter_volume, daughter_area, _ = _geometry_exact(equal_daughter)
    equal_split_passed = (
        parent_volume == 2 * daughter_volume
        and parent_area == 2 * daughter_area
        and bool(parent_geometry["symmetric_division_geometry_certified"])
        and bool(daughter_geometry["enclosure_certified"])
        and parent_area**3 >= 72 * PI_UPPER * parent_volume**2
        and daughter_area**3 >= 36 * PI_UPPER * daughter_volume**2
    )

    parent_material = _unqualified_total(parent, MATERIAL_WEIGHT)
    parent_carrier = _unqualified_total(parent, CARRIER_WEIGHT)
    parent_energy = _unqualified_total(parent, STANDARD_FREE_ENERGY_QUANTA)
    maximum_residual = 0
    all_nonnegative = True
    enclosed_daughters = 0
    for seed in range(64):
        first, second = partition_cell_counts(parent, seed=seed)
        residual = sum(
            abs(first[species] + second[species] - parent[species])
            for species in CELL_SPECIES
        )
        residual += abs(
            _unqualified_total(first, MATERIAL_WEIGHT)
            + _unqualified_total(second, MATERIAL_WEIGHT)
            - parent_material
        )
        residual += abs(
            _unqualified_total(first, CARRIER_WEIGHT)
            + _unqualified_total(second, CARRIER_WEIGHT)
            - parent_carrier
        )
        residual += abs(
            _unqualified_total(first, STANDARD_FREE_ENERGY_QUANTA)
            + _unqualified_total(second, STANDARD_FREE_ENERGY_QUANTA)
            - parent_energy
        )
        maximum_residual = max(maximum_residual, residual)
        all_nonnegative = all_nonnegative and min(first.values()) >= 0 and min(second.values()) >= 0
        enclosed_daughters += int(cell_geometry(first)["enclosure_certified"])
        enclosed_daughters += int(cell_geometry(second)["enclosure_certified"])

    passed = equal_split_passed and maximum_residual == 0 and all_nonnegative
    return {
        "passed": passed,
        "parent_counts": parent,
        "parent_geometry": parent_geometry,
        "equal_daughter_geometry": daughter_geometry,
        "exact_equal_split_volume_conservation": parent_volume == 2 * daughter_volume,
        "exact_equal_split_area_conservation": parent_area == 2 * daughter_area,
        "rational_division_implication": (
            "A_parent^3 >= 72*(355/113)*V_parent^2 implies "
            "(A_parent/2)^3 >= 36*(355/113)*(V_parent/2)^2"
        ),
        "stochastic_partition_trials": 64,
        "sampled_enclosed_daughters": enclosed_daughters,
        "sampled_daughters": 128,
        "maximum_species_material_carrier_or_energy_residual": maximum_residual,
        "hidden_division_injection": 0,
        "membrane_partition_rule": "B1=floor(B/2), B2=B-B1; no target reset",
        "all_daughter_counts_nonnegative": all_nonnegative,
    }


def _reaction_row(reaction: Reaction) -> dict[str, object]:
    material, carrier = reaction_balance(reaction)
    delta_g = reaction_free_energy_delta(reaction)
    return {
        "name": reaction.name,
        "reactants": dict(reaction.reactants),
        "products": dict(reaction.products),
        "order": reaction.order,
        "stochastic_rate": {
            "exact": str(reaction.stochastic_rate_per_second),
            "decimal": float(reaction.stochastic_rate_per_second),
        },
        "stochastic_rate_unit": reaction.rate_constant_unit,
        "concentration_rate_constant_unit": reaction.concentration_rate_constant_unit,
        "material_delta": material,
        "carrier_delta": carrier,
        "standard_free_energy_delta_quanta": delta_g,
        "heat_quanta_per_event": -delta_g,
    }


def build_stoichiometric_certificate() -> dict[str, object]:
    reactions = canonical_reactions()
    rows = [_reaction_row(reaction) for reaction in reactions]
    transport_names = {"raw_uptake", "discharged_carrier_uptake", "waste_export"}
    stoichiometry_passed = all(
        row["material_delta"] == 0 and row["carrier_delta"] == 0 for row in rows
    )
    free_energy_passed = all(
        (row["standard_free_energy_delta_quanta"] == 0)
        if row["name"] in transport_names
        else (row["standard_free_energy_delta_quanta"] < 0)
        for row in rows
    )
    unit_passed = all(
        reaction.order >= 1
        and reaction.stochastic_rate_per_second > 0
        and reaction.rate_constant_unit == "s^-1 per available reactant combination"
        and reaction.concentration_rate_constant_unit.endswith("s^-1")
        for reaction in reactions
    )
    no_spontaneous_template = all(
        any(species == _q("cell", template) for species, _ in reaction.reactants)
        for template in TEMPLATES
        for reaction in reactions
        if any(species == _q("cell", template) for species, _ in reaction.products)
    )
    closed = run_closed_ledger()
    open_flow = run_open_flow_ledger()
    division = run_division_geometry_gate()

    shortage_rejected = False
    empty = {key: 0 for key in _canonical_state()}
    unchanged_after_rejection = dict(empty)
    try:
        apply_reaction(empty, _reaction_by_name()["carrier_recharge"])
    except ValueError as error:
        shortage_rejected = "insufficient reactant" in str(error)
    positivity_passed = (
        shortage_rejected
        and empty == unchanged_after_rejection
        and int(closed["minimum_final_count"]) >= 0
        and int(open_flow["minimum_final_count"]) >= 0
    )

    gates = {
        "reaction_stoichiometry": {
            "passed": stoichiometry_passed,
            "reactions": rows,
            "material_weights": MATERIAL_WEIGHT,
            "carrier_weights": CARRIER_WEIGHT,
        },
        "si_dimension_manifest": {
            "passed": unit_passed,
            "base_units": {"time": "s", "length": "m", "entity_count": "molecule or coarse entity"},
            "derived_units": {
                "volume": "m^3",
                "area": "m^2",
                "concentration": "entity m^-3",
                "stochastic_propensity": "s^-1",
                "permeability": "m s^-1",
                "free_energy": "J",
            },
            "volume_quantum_m3": {"exact": str(VOLUME_QUANTUM_M3), "decimal": float(VOLUME_QUANTUM_M3)},
            "boundary_patch_area_m2": {"exact": str(BOUNDARY_PATCH_AREA_M2), "decimal": float(BOUNDARY_PATCH_AREA_M2)},
            "free_energy_quantum_joule": {"exact": str(FREE_ENERGY_QUANTUM_J), "decimal": float(FREE_ENERGY_QUANTUM_J)},
            "pi_upper_exact": str(PI_UPPER),
            "boundary_entity_semantics": "one coarse bilayer patch, not one amphiphile",
        },
        "declared_standard_state_free_energy": {
            "passed": free_energy_passed and bool(closed["passed"]) and bool(open_flow["passed"]),
            "species_free_energy_quanta": STANDARD_FREE_ENERGY_QUANTA,
            "internal_reactions_are_strictly_downhill": free_energy_passed,
            "mixing_and_concentration_chemical_potentials_modelled": False,
            "local_detailed_balance_calibrated": False,
        },
        "closed_batch_ledger": closed,
        "open_flow_ledger": open_flow,
        "positivity_without_clipping": {
            "passed": positivity_passed,
            "insufficient_reactant_event_rejected": shortage_rejected,
            "input_mapping_unchanged_after_rejection": empty == unchanged_after_rejection,
            "negative_count_repair_used": False,
        },
        "no_spontaneous_template": {
            "passed": no_spontaneous_template,
            "rule": "every reaction producing T_l also consumes the same T_l template",
        },
        "division_geometry_and_conservation": division,
        "constant_external_intervention": {
            "passed": bool(open_flow["schedule_constant_after_initialization"]),
            "schedule": open_flow["intervention"],
            "per_generation_manual_reset": False,
        },
    }
    all_passed = all(bool(gate["passed"]) for gate in gates.values())
    return {
        "artifact_type": "clarus_stoichiometric_protocell_ledger_certificate",
        "artifact_version": 2,
        "arithmetic": (
            "integer reaction events and ledgers; Fraction SI scales and "
            "355/113 geometry gates; floats are readout only"
        ),
        "model": {
            "cell_species": list(CELL_SPECIES),
            "environment_species": list(ENVIRONMENT_SPECIES),
            "module_count": MODULE_COUNT,
            "material_ledger": "R+P+W+2B+4*sum(T_l)",
            "carrier_moiety_ledger": "F+D",
            "standard_state_potential": "8R+4P+4F+4B+8*sum(T_l) energy quanta",
            "geometry": "V=v_q*(material+carrier); A=a_patch*B",
            "reaction_execution": "transactional integer events; shortages rejected before mutation",
        },
        "gates": gates,
        "claim_scope": {
            "registered_reactions_are_stoichiometrically_balanced": stoichiometry_passed,
            "closed_and_open_material_carrier_ledgers_verified": bool(closed["passed"] and open_flow["passed"]),
            "declared_standard_state_free_energy_and_heat_ledger_verified": free_energy_passed,
            "rational_upper_bound_membrane_geometry_verified": bool(division["passed"]),
            "division_has_no_hidden_material_carrier_template_or_energy_injection": bool(division["passed"]),
            "positivity_is_enforced_without_clipping": positivity_passed,
            "autonomous_metabolism_proven": False,
            "thermodynamic_feasibility_calibrated": False,
            "mature_offspring_supercriticality_proven": False,
            "empirical_autonomous_protocell_proven": False,
            "historical_origin_of_life_inferred": False,
        },
        "all_stoichiometric_gates_passed": all_passed,
    }


def validate_stoichiometric_certificate(certificate: Mapping[str, object]) -> bool:
    return dict(certificate) == build_stoichiometric_certificate()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="artifacts/biology/origin_life_stoichiometric_certificate.json",
    )
    parser.add_argument("--verify")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(argv)
    certificate = build_stoichiometric_certificate()
    if args.verify:
        observed = json.loads(Path(args.verify).read_text(encoding="utf-8"))
        return int(observed != certificate)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(certificate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(certificate, ensure_ascii=False, indent=2))
    return int(args.require_pass and not certificate["all_stoichiometric_gates_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
