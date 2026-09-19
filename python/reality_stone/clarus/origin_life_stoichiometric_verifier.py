"""Independent verifier for the P0.2 stoichiometric protocell ledger.

No simulator module is imported.  The registered reaction graph, event
schedules, ledgers, units, and division geometry are rebuilt here and compared
fail-closed with the submitted certificate.
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


RAW, PRECURSOR, CHARGED, DISCHARGED, WASTE, BOUNDARY = "R", "P", "F", "D", "W", "B"
TEMPLATES = tuple(f"T{index}" for index in range(7))
CELL_SPECIES = (RAW, PRECURSOR, CHARGED, DISCHARGED, WASTE, BOUNDARY, *TEMPLATES)
ENVIRONMENT_SPECIES = (RAW, DISCHARGED, WASTE)
MATERIAL = {
    RAW: 1,
    PRECURSOR: 1,
    CHARGED: 0,
    DISCHARGED: 0,
    WASTE: 1,
    BOUNDARY: 2,
    **{template: 4 for template in TEMPLATES},
}
CARRIER = {
    RAW: 0,
    PRECURSOR: 0,
    CHARGED: 1,
    DISCHARGED: 1,
    WASTE: 0,
    BOUNDARY: 0,
    **{template: 0 for template in TEMPLATES},
}
STANDARD_FREE_ENERGY = {
    RAW: 8,
    PRECURSOR: 4,
    CHARGED: 4,
    DISCHARGED: 0,
    WASTE: 0,
    BOUNDARY: 4,
    **{template: 8 for template in TEMPLATES},
}
VOLUME_QUANTUM_M3 = Fraction(1, 72 * 10**18)
BOUNDARY_PATCH_AREA_M2 = Fraction(31, 10**14)
FREE_ENERGY_QUANTUM_J = Fraction(1, 10**20)
PI_UPPER = Fraction(355, 113)


@dataclass(frozen=True)
class _Reaction:
    name: str
    reactants: tuple[tuple[str, int], ...]
    products: tuple[tuple[str, int], ...]
    rate: Fraction

    @property
    def order(self) -> int:
        return sum(value for _, value in self.reactants)


@dataclass(frozen=True)
class StoichiometricVerificationReport:
    verified: bool
    checks: tuple[str, ...]
    errors: tuple[str, ...]


def _q(compartment: str, species: str) -> str:
    return f"{compartment}:{species}"


def _rxn(
    name: str,
    reactants: Mapping[str, int],
    products: Mapping[str, int],
    rate: Fraction,
) -> _Reaction:
    return _Reaction(name, tuple(sorted(reactants.items())), tuple(sorted(products.items())), rate)


def _reactions() -> tuple[_Reaction, ...]:
    def cell(species: str) -> str:
        return _q("cell", species)

    def env(species: str) -> str:
        return _q("environment", species)

    values = [
        _rxn(
            "carrier_recharge",
            {cell(RAW): 1, cell(DISCHARGED): 1},
            {cell(WASTE): 1, cell(CHARGED): 1},
            Fraction(1, 2),
        ),
        _rxn(
            "precursor_activation",
            {cell(RAW): 1, cell(CHARGED): 1},
            {cell(PRECURSOR): 1, cell(DISCHARGED): 1},
            Fraction(1, 4),
        ),
        _rxn(
            "boundary_synthesis",
            {cell(PRECURSOR): 2, cell(CHARGED): 1},
            {cell(BOUNDARY): 1, cell(DISCHARGED): 1},
            Fraction(1, 4),
        ),
        _rxn(
            "maintenance_discharge",
            {cell(CHARGED): 1},
            {cell(DISCHARGED): 1},
            Fraction(1, 64),
        ),
        _rxn(
            "precursor_decay",
            {cell(PRECURSOR): 1},
            {cell(WASTE): 1},
            Fraction(1, 512),
        ),
        _rxn(
            "boundary_decay",
            {cell(BOUNDARY): 1},
            {cell(WASTE): 2},
            Fraction(1, 2048),
        ),
        _rxn("raw_uptake", {env(RAW): 1}, {cell(RAW): 1}, Fraction(1, 16)),
        _rxn(
            "discharged_carrier_uptake",
            {env(DISCHARGED): 1},
            {cell(DISCHARGED): 1},
            Fraction(1, 16),
        ),
        _rxn("waste_export", {cell(WASTE): 1}, {env(WASTE): 1}, Fraction(1, 16)),
    ]
    for template in TEMPLATES:
        values.extend(
            [
                _rxn(
                    f"copy_{template}",
                    {cell(template): 1, cell(PRECURSOR): 4, cell(CHARGED): 2},
                    {cell(template): 2, cell(DISCHARGED): 2},
                    Fraction(1, 8),
                ),
                _rxn(
                    f"decay_{template}",
                    {cell(template): 1},
                    {cell(WASTE): 4},
                    Fraction(1, 1024),
                ),
            ]
        )
    return tuple(values)


def _base(qualified: str) -> str:
    parts = qualified.split(":", 1)
    if len(parts) != 2:
        raise ValueError("invalid qualified species")
    compartment, species = parts
    allowed = CELL_SPECIES if compartment == "cell" else ENVIRONMENT_SPECIES
    if compartment not in {"cell", "environment"} or species not in allowed:
        raise ValueError("unknown qualified species")
    return species


def _total(state: Mapping[str, int], weights: Mapping[str, int]) -> int:
    return sum(weights[_base(species)] * count for species, count in state.items())


def _free_energy_total(state: Mapping[str, int]) -> int:
    return _total(state, STANDARD_FREE_ENERGY)


def _balance(reaction: _Reaction) -> tuple[int, int]:
    material = carrier = 0
    for sign, side in ((-1, reaction.reactants), (1, reaction.products)):
        for species, coefficient in side:
            base = _base(species)
            material += sign * MATERIAL[base] * coefficient
            carrier += sign * CARRIER[base] * coefficient
    return material, carrier


def _free_energy_delta(reaction: _Reaction) -> int:
    delta = 0
    for sign, side in ((-1, reaction.reactants), (1, reaction.products)):
        for species, coefficient in side:
            delta += sign * STANDARD_FREE_ENERGY[_base(species)] * coefficient
    return delta


def _reaction_heat(reaction: _Reaction, events: int) -> int:
    delta = _free_energy_delta(reaction)
    if delta > 0:
        raise ValueError(f"{reaction.name} is uphill in the declared potential")
    return -delta * events


def _apply(state: Mapping[str, int], reaction: _Reaction, events: int = 1) -> dict[str, int]:
    if events < 1:
        raise ValueError("events must be positive")
    updated = dict(state)
    for species, coefficient in reaction.reactants:
        required = coefficient * events
        if updated.get(species, 0) < required:
            raise ValueError("insufficient reactant")
    for species, coefficient in reaction.reactants:
        updated[species] -= coefficient * events
    for species, coefficient in reaction.products:
        updated[species] = updated.get(species, 0) + coefficient * events
    if min(updated.values()) < 0:
        raise AssertionError("negative reference count")
    return updated


def _apply_schedule(
    state: Mapping[str, int],
    schedule: Sequence[tuple[str, int]],
) -> tuple[dict[str, int], int, int]:
    reactions = {reaction.name: reaction for reaction in _reactions()}
    current = dict(state)
    heat = 0
    maximum_event_energy_residual = 0
    for name, events in schedule:
        reaction = reactions[name]
        before = _free_energy_total(current)
        current = _apply(current, reaction, events)
        event_heat = _reaction_heat(reaction, events)
        after = _free_energy_total(current)
        maximum_event_energy_residual = max(
            maximum_event_energy_residual,
            abs(before - after - event_heat),
        )
        heat += event_heat
    return current, heat, maximum_event_energy_residual


def _state() -> dict[str, int]:
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
            _q("cell", WASTE): 0,
            _q("cell", BOUNDARY): 80,
            _q("environment", RAW): 256,
            _q("environment", DISCHARGED): 256,
            _q("environment", WASTE): 0,
            **{_q("cell", template): 4 for template in TEMPLATES},
        }
    )
    return state


def _closed() -> dict[str, object]:
    state = _state()
    initial_state = dict(state)
    initial_material = _total(state, MATERIAL)
    initial_carrier = _total(state, CARRIER)
    initial_energy = _free_energy_total(state)
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
    final_material = _total(state, MATERIAL)
    final_carrier = _total(state, CARRIER)
    final_energy = _free_energy_total(state)
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


def _open() -> dict[str, object]:
    state = _state()
    initial_state = dict(state)
    initial_material = _total(state, MATERIAL)
    initial_carrier = _total(state, CARRIER)
    initial_energy = _free_energy_total(state)
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
    material_in = material_out = carrier_in = carrier_out = 0
    free_energy_in = free_energy_out = heat = 0
    maximum_event_energy_residual = 0
    for tick in range(16):
        state[_q("environment", RAW)] += intervention["raw_inflow_per_tick"]
        material_in += MATERIAL[RAW] * intervention["raw_inflow_per_tick"]
        carrier_in += CARRIER[RAW] * intervention["raw_inflow_per_tick"]
        free_energy_in += STANDARD_FREE_ENERGY[RAW] * intervention["raw_inflow_per_tick"]
        state[_q("environment", DISCHARGED)] += intervention["discharged_carrier_inflow_per_tick"]
        material_in += MATERIAL[DISCHARGED] * intervention["discharged_carrier_inflow_per_tick"]
        carrier_in += CARRIER[DISCHARGED] * intervention["discharged_carrier_inflow_per_tick"]
        free_energy_in += (
            STANDARD_FREE_ENERGY[DISCHARGED] * intervention["discharged_carrier_inflow_per_tick"]
        )
        schedule = [
            ("raw_uptake", 2),
            ("discharged_carrier_uptake", 2),
            ("carrier_recharge", 1),
            ("precursor_activation", 1),
            (f"copy_{TEMPLATES[tick % 7]}", 1),
            ("boundary_synthesis", 1),
            ("maintenance_discharge", 1),
            ("waste_export", 1),
        ]
        state, tick_heat, tick_error = _apply_schedule(state, schedule)
        heat += tick_heat
        maximum_event_energy_residual = max(maximum_event_energy_residual, tick_error)
        outflow = intervention["waste_outflow_per_tick"]
        state[_q("environment", WASTE)] -= outflow
        if state[_q("environment", WASTE)] < 0:
            raise AssertionError("reference outflow exceeded waste")
        material_out += MATERIAL[WASTE] * outflow
        carrier_out += CARRIER[WASTE] * outflow
        free_energy_out += STANDARD_FREE_ENERGY[WASTE] * outflow
    final_material = _total(state, MATERIAL)
    final_carrier = _total(state, CARRIER)
    final_energy = _free_energy_total(state)
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
        "ticks": 16,
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
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0
        for count in counts.values()
    ):
        raise ValueError("cell counts must be nonnegative integers")


def _unqualified_total(counts: Mapping[str, int], weights: Mapping[str, int]) -> int:
    return sum(weights[species] * count for species, count in counts.items())


def _geometry_exact(counts: Mapping[str, int]) -> tuple[Fraction, Fraction, int]:
    _validate_cell_counts(counts)
    material = _unqualified_total(counts, MATERIAL)
    carrier = _unqualified_total(counts, CARRIER)
    volume_quanta = material + carrier
    if volume_quanta < 1:
        raise ValueError("cell volume must be positive")
    volume = VOLUME_QUANTUM_M3 * volume_quanta
    area = BOUNDARY_PATCH_AREA_M2 * counts[BOUNDARY]
    return volume, area, volume_quanta


def _geometry(counts: Mapping[str, int]) -> dict[str, object]:
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


def _partition(
    parent: Mapping[str, int],
    seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    _validate_cell_counts(parent)
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


def _division() -> dict[str, object]:
    parent = {
        RAW: 20,
        PRECURSOR: 20,
        CHARGED: 20,
        DISCHARGED: 20,
        WASTE: 20,
        BOUNDARY: 80,
        **{template: 4 for template in TEMPLATES},
    }
    equal = {species: count // 2 for species, count in parent.items()}
    parent_geometry = _geometry(parent)
    daughter_geometry = _geometry(equal)
    parent_volume, parent_area, _ = _geometry_exact(parent)
    daughter_volume, daughter_area, _ = _geometry_exact(equal)
    equal_pass = (
        parent_volume == 2 * daughter_volume
        and parent_area == 2 * daughter_area
        and bool(parent_geometry["symmetric_division_geometry_certified"])
        and bool(daughter_geometry["enclosure_certified"])
        and parent_area**3 >= 72 * PI_UPPER * parent_volume**2
        and daughter_area**3 >= 36 * PI_UPPER * daughter_volume**2
    )
    residual_max = 0
    nonnegative = True
    enclosed_daughters = 0
    parent_material = _unqualified_total(parent, MATERIAL)
    parent_carrier = _unqualified_total(parent, CARRIER)
    parent_energy = _unqualified_total(parent, STANDARD_FREE_ENERGY)
    for seed in range(64):
        first, second = _partition(parent, seed)
        residual = sum(abs(first[s] + second[s] - parent[s]) for s in CELL_SPECIES)
        residual += abs(
            _unqualified_total(first, MATERIAL)
            + _unqualified_total(second, MATERIAL)
            - parent_material
        )
        residual += abs(
            _unqualified_total(first, CARRIER)
            + _unqualified_total(second, CARRIER)
            - parent_carrier
        )
        residual += abs(
            _unqualified_total(first, STANDARD_FREE_ENERGY)
            + _unqualified_total(second, STANDARD_FREE_ENERGY)
            - parent_energy
        )
        residual_max = max(residual_max, residual)
        nonnegative = nonnegative and min(first.values()) >= 0 and min(second.values()) >= 0
        enclosed_daughters += int(_geometry(first)["enclosure_certified"])
        enclosed_daughters += int(_geometry(second)["enclosure_certified"])
    return {
        "passed": equal_pass and residual_max == 0 and nonnegative,
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
        "maximum_species_material_carrier_or_energy_residual": residual_max,
        "hidden_division_injection": 0,
        "membrane_partition_rule": "B1=floor(B/2), B2=B-B1; no target reset",
        "all_daughter_counts_nonnegative": nonnegative,
    }


def _reaction_row(reaction: _Reaction) -> dict[str, object]:
    material, carrier = _balance(reaction)
    delta_g = _free_energy_delta(reaction)
    concentration_unit = (
        "s^-1"
        if reaction.order == 1
        else f"m^{3 * (reaction.order - 1)} molecule^{1 - reaction.order} s^-1"
    )
    return {
        "name": reaction.name,
        "reactants": dict(reaction.reactants),
        "products": dict(reaction.products),
        "order": reaction.order,
        "stochastic_rate": {"exact": str(reaction.rate), "decimal": float(reaction.rate)},
        "stochastic_rate_unit": "s^-1 per available reactant combination",
        "concentration_rate_constant_unit": concentration_unit,
        "material_delta": material,
        "carrier_delta": carrier,
        "standard_free_energy_delta_quanta": delta_g,
        "heat_quanta_per_event": -delta_g,
    }


def _expected() -> dict[str, object]:
    reactions = _reactions()
    rows = [_reaction_row(reaction) for reaction in reactions]
    transport_names = {"raw_uptake", "discharged_carrier_uptake", "waste_export"}
    stoich = all(row["material_delta"] == 0 and row["carrier_delta"] == 0 for row in rows)
    free_energy = all(
        (row["standard_free_energy_delta_quanta"] == 0)
        if row["name"] in transport_names
        else (row["standard_free_energy_delta_quanta"] < 0)
        for row in rows
    )
    units = all(
        reaction.order >= 1
        and reaction.rate > 0
        and _reaction_row(reaction)["stochastic_rate_unit"]
        == "s^-1 per available reactant combination"
        and str(_reaction_row(reaction)["concentration_rate_constant_unit"]).endswith("s^-1")
        for reaction in reactions
    )
    no_spontaneous = all(
        any(species == _q("cell", template) for species, _ in reaction.reactants)
        for template in TEMPLATES
        for reaction in reactions
        if any(species == _q("cell", template) for species, _ in reaction.products)
    )
    closed = _closed()
    opened = _open()
    division = _division()
    shortage = False
    empty = {key: 0 for key in _state()}
    unchanged = dict(empty)
    try:
        _apply(empty, {r.name: r for r in reactions}["carrier_recharge"])
    except ValueError as error:
        shortage = "insufficient reactant" in str(error)
    positivity = (
        shortage
        and empty == unchanged
        and closed["minimum_final_count"] >= 0
        and opened["minimum_final_count"] >= 0
    )
    gates = {
        "reaction_stoichiometry": {
            "passed": stoich,
            "reactions": rows,
            "material_weights": MATERIAL,
            "carrier_weights": CARRIER,
        },
        "si_dimension_manifest": {
            "passed": units,
            "base_units": {
                "time": "s",
                "length": "m",
                "entity_count": "molecule or coarse entity",
            },
            "derived_units": {
                "volume": "m^3",
                "area": "m^2",
                "concentration": "entity m^-3",
                "stochastic_propensity": "s^-1",
                "permeability": "m s^-1",
                "free_energy": "J",
            },
            "volume_quantum_m3": {
                "exact": str(VOLUME_QUANTUM_M3),
                "decimal": float(VOLUME_QUANTUM_M3),
            },
            "boundary_patch_area_m2": {
                "exact": str(BOUNDARY_PATCH_AREA_M2),
                "decimal": float(BOUNDARY_PATCH_AREA_M2),
            },
            "free_energy_quantum_joule": {
                "exact": str(FREE_ENERGY_QUANTUM_J),
                "decimal": float(FREE_ENERGY_QUANTUM_J),
            },
            "pi_upper_exact": str(PI_UPPER),
            "boundary_entity_semantics": "one coarse bilayer patch, not one amphiphile",
        },
        "declared_standard_state_free_energy": {
            "passed": free_energy and closed["passed"] and opened["passed"],
            "species_free_energy_quanta": STANDARD_FREE_ENERGY,
            "internal_reactions_are_strictly_downhill": free_energy,
            "mixing_and_concentration_chemical_potentials_modelled": False,
            "local_detailed_balance_calibrated": False,
        },
        "closed_batch_ledger": closed,
        "open_flow_ledger": opened,
        "positivity_without_clipping": {
            "passed": positivity,
            "insufficient_reactant_event_rejected": shortage,
            "input_mapping_unchanged_after_rejection": empty == unchanged,
            "negative_count_repair_used": False,
        },
        "no_spontaneous_template": {
            "passed": no_spontaneous,
            "rule": "every reaction producing T_l also consumes the same T_l template",
        },
        "division_geometry_and_conservation": division,
        "constant_external_intervention": {
            "passed": opened["schedule_constant_after_initialization"],
            "schedule": opened["intervention"],
            "per_generation_manual_reset": False,
        },
    }
    all_passed = all(gate["passed"] for gate in gates.values())
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
            "module_count": 7,
            "material_ledger": "R+P+W+2B+4*sum(T_l)",
            "carrier_moiety_ledger": "F+D",
            "standard_state_potential": "8R+4P+4F+4B+8*sum(T_l) energy quanta",
            "geometry": "V=v_q*(material+carrier); A=a_patch*B",
            "reaction_execution": "transactional integer events; shortages rejected before mutation",
        },
        "gates": gates,
        "claim_scope": {
            "registered_reactions_are_stoichiometrically_balanced": stoich,
            "closed_and_open_material_carrier_ledgers_verified": (
                closed["passed"] and opened["passed"]
            ),
            "declared_standard_state_free_energy_and_heat_ledger_verified": free_energy,
            "rational_upper_bound_membrane_geometry_verified": division["passed"],
            "division_has_no_hidden_material_carrier_template_or_energy_injection": division[
                "passed"
            ],
            "positivity_is_enforced_without_clipping": positivity,
            "autonomous_metabolism_proven": False,
            "thermodynamic_feasibility_calibrated": False,
            "mature_offspring_supercriticality_proven": False,
            "empirical_autonomous_protocell_proven": False,
            "historical_origin_of_life_inferred": False,
        },
        "all_stoichiometric_gates_passed": all_passed,
    }


def _finite(value: object, path: str = "certificate") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")


def verify_stoichiometric_certificate(
    certificate: Mapping[str, object],
) -> StoichiometricVerificationReport:
    checks: list[str] = []
    errors: list[str] = []
    try:
        if not isinstance(certificate, Mapping):
            raise ValueError("certificate must be a mapping")
        _finite(certificate)
        expected = _expected()
        if set(certificate) != set(expected):
            raise ValueError("top-level schema differs from the registered certificate")
        checks.append("schema_and_finite_numbers")
        for section in (
            "artifact_type",
            "artifact_version",
            "arithmetic",
            "model",
            "gates",
            "claim_scope",
            "all_stoichiometric_gates_passed",
        ):
            if certificate.get(section) != expected[section]:
                raise ValueError(f"{section} differs from independent recomputation")
            checks.append(section)
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
    return StoichiometricVerificationReport(not errors, tuple(checks), tuple(errors))


def independently_verified(certificate: Mapping[str, object]) -> bool:
    return verify_stoichiometric_certificate(certificate).verified


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate")
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.certificate).read_text(encoding="utf-8"))
    report = verify_stoichiometric_certificate(payload)
    print(
        json.dumps({"verified": report.verified, "checks": report.checks, "errors": report.errors})
    )
    return int(not report.verified)


if __name__ == "__main__":
    raise SystemExit(main())
