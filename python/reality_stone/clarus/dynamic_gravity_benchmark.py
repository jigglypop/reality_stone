"""Loop 8F finite-response gravitational decision benchmark."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .brain_geometry_benchmark import _lcb, _residual_trials, _run_residual_arm, _stream
from .gravitational_decision_benchmark import (
    GravityDecisionBenchConfig,
    _FieldBasis,
    _evidence_path,
    _field_basis,
    _gravity_trial,
    _reflect,
    _sigmoid,
    _threshold_trial,
)


@dataclass(frozen=True)
class DynamicGravityBenchConfig:
    base: GravityDecisionBenchConfig = GravityDecisionBenchConfig()
    field_time: float = 1.0
    damping_ratio: float = 0.80
    wave_speed: float = 1.0
    capture_error_probability: float = 0.05
    capture_persistence: int = 3


def _laplacian(values: np.ndarray, spacing: float) -> np.ndarray:
    result = np.empty_like(values)
    inverse_square = 1.0 / spacing**2
    result[0] = (values[0] - values[1]) * inverse_square
    result[-1] = (values[-1] - values[-2]) * inverse_square
    result[1:-1] = (
        2.0 * values[1:-1] - values[:-2] - values[2:]
    ) * inverse_square
    return result


def _source(
    evidence: float,
    basis: _FieldBasis,
    config: DynamicGravityBenchConfig,
    *,
    flip: bool,
    contrast: bool = False,
) -> np.ndarray:
    plus_mass = _sigmoid(config.base.evidence_temperature_inverse * evidence)
    if flip:
        plus_mass = 1.0 - plus_mass
    minus_mass = 1.0 - plus_mass
    if contrast:
        return (
            (plus_mass - 0.5) * basis.plus_source
            + (minus_mass - 0.5) * basis.minus_source
        )
    return plus_mass * basis.plus_source + minus_mass * basis.minus_source


def _field_step(
    potential: np.ndarray,
    field_velocity: np.ndarray,
    source: np.ndarray,
    basis: _FieldBasis,
    config: DynamicGravityBenchConfig,
) -> tuple[np.ndarray, np.ndarray]:
    base = config.base
    acceleration = (
        -2.0 * config.damping_ratio * config.field_time * field_velocity
        - config.wave_speed**2 * _laplacian(potential, basis.spacing)
        - base.screening**2 * potential
        - base.coupling * source
    ) / config.field_time**2
    next_velocity = field_velocity + base.dt * acceleration
    next_potential = potential + base.dt * next_velocity
    return next_potential, next_velocity


def _field_preflight(basis: _FieldBasis, config: DynamicGravityBenchConfig) -> dict[str, float]:
    zero = np.zeros_like(basis.grid)
    potential, velocity = _field_step(zero.copy(), zero.copy(), zero, basis, config)
    zero_error = float(max(np.max(np.abs(potential)), np.max(np.abs(velocity))))
    potential = zero.copy()
    velocity = zero.copy()
    equal_source = 0.5 * basis.plus_source + 0.5 * basis.minus_source
    max_center_force = 0.0
    center = len(basis.grid) // 2
    for _ in range(100):
        potential, velocity = _field_step(potential, velocity, equal_source, basis, config)
        force = -np.gradient(potential, basis.spacing, edge_order=2)
        max_center_force = max(max_center_force, abs(float(force[center])))
    cfl = config.wave_speed * config.base.dt / (config.field_time * basis.spacing)
    return {
        "cfl": cfl,
        "zero_field_error": zero_error,
        "equal_mass_max_center_force": max_center_force,
    }


def _dynamic_trial(
    path: tuple[float, ...],
    target: int,
    seed: int,
    trial_index: int,
    basis: _FieldBasis,
    config: DynamicGravityBenchConfig,
    *,
    flip: bool = False,
    contrast: bool = False,
) -> dict[str, float | bool | int]:
    base = config.base
    potential = np.zeros_like(basis.grid)
    field_velocity = np.zeros_like(basis.grid)
    position = particle_velocity = 0.0
    motion_rng = _stream(seed * 1061 + trial_index, 503)
    persistence = 0
    persistence_side = 0
    captured = False
    captured_side = 0
    capture_step = base.max_steps
    saddle_position = 0.0
    max_field_energy = 0.0
    max_abs_velocity = 0.0
    bounded = True

    for index, evidence in enumerate(path):
        source = _source(evidence, basis, config, flip=flip, contrast=contrast)
        potential, field_velocity = _field_step(
            potential, field_velocity, source, basis, config
        )
        force = -np.gradient(potential, basis.spacing, edge_order=2)
        gradient = np.gradient(potential, basis.spacing, edge_order=2)
        field_energy = 0.5 * basis.spacing * float(np.sum(
            config.field_time**2 * field_velocity**2
            + config.wave_speed**2 * gradient**2
            + base.screening**2 * potential**2
        ))
        max_field_energy = max(max_field_energy, abs(field_energy))

        between = np.where(
            (basis.grid >= -base.source_position)
            & (basis.grid <= base.source_position)
        )[0]
        saddle_index = int(between[np.argmax(potential[between])])
        saddle_position = float(basis.grid[saddle_index])
        saddle_potential = float(potential[saddle_index])
        old_velocity = particle_velocity
        position += old_velocity * base.dt
        particle_velocity += (
            (-base.friction * old_velocity + float(np.interp(position, basis.grid, force)))
            * base.dt
            + math.sqrt(
                2.0 * base.friction * base.decision_temperature * base.dt
            )
            * motion_rng.gauss(0.0, 1.0)
        )
        position, particle_velocity = _reflect(position, particle_velocity, base)
        max_abs_velocity = max(max_abs_velocity, abs(particle_velocity))

        side = 1 if position > saddle_position else -1
        side_indices = np.where(
            basis.grid >= saddle_position if side > 0 else basis.grid <= saddle_position
        )[0]
        basin_minimum = float(np.min(potential[side_indices]))
        barrier = saddle_potential - basin_minimum
        energy = (
            0.5 * particle_velocity**2
            + float(np.interp(position, basis.grid, potential))
        )
        barrier_required = base.decision_temperature * math.log(
            1.0 / config.capture_error_probability
        )
        condition = (
            barrier >= barrier_required
            and energy < saddle_potential - base.capture_tolerance
        )
        if condition and side == persistence_side:
            persistence += 1
        elif condition:
            persistence_side = side
            persistence = 1
        else:
            persistence = 0
            persistence_side = 0
        bounded = (
            bounded
            and np.all(np.isfinite(potential))
            and np.all(np.isfinite(field_velocity))
            and math.isfinite(position)
            and math.isfinite(particle_velocity)
            and max_field_energy <= 1e4
        )
        if persistence >= config.capture_persistence:
            captured = True
            captured_side = side
            capture_step = index + 1
            break

    if not captured:
        captured_side = 1 if position >= saddle_position else -1

    frozen_force = -np.gradient(potential, basis.spacing, edge_order=2)
    side_flipped = False
    for _ in range(base.post_capture_steps):
        old_velocity = particle_velocity
        position += old_velocity * base.dt
        particle_velocity += (
            (-base.friction * old_velocity + float(np.interp(position, basis.grid, frozen_force)))
            * base.dt
            + math.sqrt(
                2.0 * base.friction * base.decision_temperature * base.dt
            )
            * motion_rng.gauss(0.0, 1.0)
        )
        position, particle_velocity = _reflect(position, particle_velocity, base)
        current_side = 1 if position > saddle_position else -1
        side_flipped = side_flipped or current_side != captured_side
        max_abs_velocity = max(max_abs_velocity, abs(particle_velocity))
        bounded = bounded and math.isfinite(position) and math.isfinite(particle_velocity)

    correct = captured_side == target
    return {
        "correct": correct,
        "steps": capture_step,
        "utility": (1.0 if correct else -1.0) - base.time_cost * capture_step,
        "captured": captured,
        "side_flipped": side_flipped,
        "bounded": bounded,
        "max_abs_velocity": max_abs_velocity,
        "max_field_energy": max_field_energy,
    }


def _domain(basis: _FieldBasis, config: DynamicGravityBenchConfig, *, ood: bool) -> dict[str, object]:
    arms = ("fixed_ddm", "quasi_static", "dynamic_gravity", "dynamic_shuffle", "dynamic_sign_flip")
    per_seed: dict[str, list[dict[str, float]]] = {arm: [] for arm in arms}
    levels = config.base.ood_coherences if ood else config.base.id_coherences
    trace_identity = True
    for offset in range(config.base.residual.base.seeds):
        seed = 860_000 + offset
        trials = _residual_trials(seed, config.base.residual, ood=ood, stationary=False)
        memory_result = _run_residual_arm(
            trials, "residual_replay", seed, config.base.residual, ood=ood, return_trace=True
        )
        trace = memory_result["decision_trace"]
        frozen_trace = tuple(trace)
        coherence_rng = _stream(seed, 601 if not ood else 603)
        coherences = [levels[coherence_rng.randrange(len(levels))] for _ in trace]
        paths = [
            _evidence_path(signal, target, coherences[index], seed, index, config.base, ood=ood)
            for index, (signal, target) in enumerate(trace)
        ]
        permutation = list(range(len(paths)))
        _stream(seed, 607 if not ood else 609).shuffle(permutation)
        low_levels = set(sorted(levels)[:2])
        high_levels = set(sorted(levels)[2:])
        for arm in arms:
            rows: list[dict[str, float | bool | int]] = []
            for index, (_, target) in enumerate(trace):
                if arm == "fixed_ddm":
                    row = _threshold_trial(paths[index], target, config.base.ddm_boundary, config.base)
                    row["max_field_energy"] = 0.0
                elif arm == "quasi_static":
                    row = _gravity_trial(paths[index], target, seed, index, basis, config.base)
                    row["max_field_energy"] = 0.0
                elif arm == "dynamic_shuffle":
                    row = _dynamic_trial(paths[permutation[index]], target, seed, index, basis, config)
                else:
                    row = _dynamic_trial(
                        paths[index], target, seed, index, basis, config,
                        flip=arm == "dynamic_sign_flip",
                    )
                rows.append(row)
            low_indices = [index for index, value in enumerate(coherences) if value in low_levels]
            high_indices = [index for index, value in enumerate(coherences) if value in high_levels]
            per_seed[arm].append({
                "accuracy": sum(float(row["correct"]) for row in rows) / len(rows),
                "utility": sum(float(row["utility"]) for row in rows) / len(rows),
                "steps": sum(float(row["steps"]) for row in rows) / len(rows),
                "low_coherence_steps": sum(float(rows[i]["steps"]) for i in low_indices) / len(low_indices),
                "high_coherence_steps": sum(float(rows[i]["steps"]) for i in high_indices) / len(high_indices),
                "capture_rate": sum(float(row["captured"]) for row in rows) / len(rows),
                "flip_rate": sum(float(row["side_flipped"]) for row in rows) / len(rows),
                "bounded": float(all(bool(row["bounded"]) for row in rows)),
                "max_field_energy": max(float(row["max_field_energy"]) for row in rows),
            })
            trace_identity = trace_identity and tuple(trace) == frozen_trace

    summary: dict[str, object] = {}
    for arm in arms:
        summary[arm] = {
            key: sum(row[key] for row in per_seed[arm]) / len(per_seed[arm])
            for key in per_seed[arm][0]
        }

    def difference(left: str, right: str, metric: str) -> list[float]:
        return [per_seed[left][i][metric] - per_seed[right][i][metric] for i in range(len(per_seed[left]))]

    tag = int(ood) * 100
    summary["effects"] = {
        "dynamic_minus_quasi_accuracy_lcb": _lcb(difference("dynamic_gravity", "quasi_static", "accuracy"), seed=20261201 + tag),
        "dynamic_minus_ddm_accuracy_lcb": _lcb(difference("dynamic_gravity", "fixed_ddm", "accuracy"), seed=20261202 + tag),
        "dynamic_minus_ddm_utility_lcb": _lcb(difference("dynamic_gravity", "fixed_ddm", "utility"), seed=20261203 + tag),
        "dynamic_minus_shuffle_accuracy_lcb": _lcb(difference("dynamic_gravity", "dynamic_shuffle", "accuracy"), seed=20261204 + tag),
        "dynamic_minus_sign_flip_accuracy_lcb": _lcb(difference("dynamic_gravity", "dynamic_sign_flip", "accuracy"), seed=20261205 + tag),
    }
    summary["memory_trace_identical"] = trace_identity
    return summary


def evaluate_dynamic_gravity(config: DynamicGravityBenchConfig | None = None) -> dict[str, object]:
    cfg = config or DynamicGravityBenchConfig()
    basis = _field_basis(cfg.base)
    preflight = _field_preflight(basis, cfg)
    id_result = _domain(basis, cfg, ood=False)
    ood_result = _domain(basis, cfg, ood=True)
    id_effects = id_result["effects"]
    ood_effects = ood_result["effects"]
    dynamic_rows = (id_result["dynamic_gravity"], ood_result["dynamic_gravity"])
    gates = {
        "field_dynamics_stable": (
            preflight["cfl"] <= 1.0
            and preflight["zero_field_error"] == 0.0
            and preflight["equal_mass_max_center_force"] <= 1e-10
        ),
        "beats_quasi_static": (
            id_effects["dynamic_minus_quasi_accuracy_lcb"] >= 0.05
            and ood_effects["dynamic_minus_quasi_accuracy_lcb"] >= 0.05
        ),
        "ddm_accuracy_noninferior": (
            id_effects["dynamic_minus_ddm_accuracy_lcb"] >= -0.01
            and ood_effects["dynamic_minus_ddm_accuracy_lcb"] >= -0.01
        ),
        "ddm_utility_noninferior": (
            id_effects["dynamic_minus_ddm_utility_lcb"] >= 0.0
            and ood_effects["dynamic_minus_ddm_utility_lcb"] >= 0.0
        ),
        "source_alignment": (
            id_effects["dynamic_minus_shuffle_accuracy_lcb"] >= 0.10
            and ood_effects["dynamic_minus_shuffle_accuracy_lcb"] >= 0.10
        ),
        "source_sign": (
            id_effects["dynamic_minus_sign_flip_accuracy_lcb"] >= 0.20
            and ood_effects["dynamic_minus_sign_flip_accuracy_lcb"] >= 0.20
        ),
        "graded_capture_time": all(
            row["low_coherence_steps"] - row["high_coherence_steps"] >= 10.0
            and row["steps"] > 10.0
            for row in dynamic_rows
        ),
        "capture_stable": all(
            row["capture_rate"] >= 0.90
            and row["flip_rate"] <= 0.02
            and bool(row["bounded"])
            and row["max_field_energy"] <= 1e4
            for row in dynamic_rows
        ),
        "integrity": bool(id_result["memory_trace_identical"]) and bool(ood_result["memory_trace_identical"]),
    }
    hard_gate = all(gates.values())
    return {
        "schema": "clarus.dynamic-gravity.validation.v1",
        "config": asdict(cfg),
        "preflight": preflight,
        "id": id_result,
        "ood": ood_result,
        "future_reads": 0,
        "environment_clone_calls": 0,
        "gates": gates,
        "hard_gate": hard_gate,
        "score": 100 if hard_gate else 0,
        "decision": "GO" if hard_gate else "STOP",
    }


def small_dynamic_config() -> DynamicGravityBenchConfig:
    from .gravitational_decision_benchmark import small_gravity_config

    return DynamicGravityBenchConfig(base=small_gravity_config(trials=16, seeds=2))


def _contrast_domain(
    basis: _FieldBasis,
    config: DynamicGravityBenchConfig,
    *,
    ood: bool,
) -> dict[str, object]:
    arms = ("fixed_ddm", "contrast_gravity", "contrast_shuffle", "contrast_sign_flip")
    per_seed: dict[str, list[dict[str, float]]] = {arm: [] for arm in arms}
    levels = config.base.ood_coherences if ood else config.base.id_coherences
    trace_identity = True
    for offset in range(config.base.residual.base.seeds):
        seed = 860_000 + offset
        trials = _residual_trials(seed, config.base.residual, ood=ood, stationary=False)
        memory_result = _run_residual_arm(
            trials, "residual_replay", seed, config.base.residual, ood=ood, return_trace=True
        )
        trace = memory_result["decision_trace"]
        frozen_trace = tuple(trace)
        coherence_rng = _stream(seed, 601 if not ood else 603)
        coherences = [levels[coherence_rng.randrange(len(levels))] for _ in trace]
        paths = [
            _evidence_path(signal, target, coherences[index], seed, index, config.base, ood=ood)
            for index, (signal, target) in enumerate(trace)
        ]
        permutation = list(range(len(paths)))
        _stream(seed, 607 if not ood else 609).shuffle(permutation)
        low_levels = set(sorted(levels)[:2])
        high_levels = set(sorted(levels)[2:])
        for arm in arms:
            rows: list[dict[str, float | bool | int]] = []
            for index, (_, target) in enumerate(trace):
                if arm == "fixed_ddm":
                    row = _threshold_trial(paths[index], target, config.base.ddm_boundary, config.base)
                    row["max_field_energy"] = 0.0
                elif arm == "contrast_shuffle":
                    row = _dynamic_trial(
                        paths[permutation[index]], target, seed, index, basis, config,
                        contrast=True,
                    )
                else:
                    row = _dynamic_trial(
                        paths[index], target, seed, index, basis, config,
                        flip=arm == "contrast_sign_flip", contrast=True,
                    )
                rows.append(row)
            low_indices = [index for index, value in enumerate(coherences) if value in low_levels]
            high_indices = [index for index, value in enumerate(coherences) if value in high_levels]
            per_seed[arm].append({
                "accuracy": sum(float(row["correct"]) for row in rows) / len(rows),
                "utility": sum(float(row["utility"]) for row in rows) / len(rows),
                "steps": sum(float(row["steps"]) for row in rows) / len(rows),
                "low_coherence_steps": sum(float(rows[i]["steps"]) for i in low_indices) / len(low_indices),
                "high_coherence_steps": sum(float(rows[i]["steps"]) for i in high_indices) / len(high_indices),
                "capture_rate": sum(float(row["captured"]) for row in rows) / len(rows),
                "flip_rate": sum(float(row["side_flipped"]) for row in rows) / len(rows),
                "bounded": float(all(bool(row["bounded"]) for row in rows)),
                "max_field_energy": max(float(row["max_field_energy"]) for row in rows),
            })
            trace_identity = trace_identity and tuple(trace) == frozen_trace

    summary: dict[str, object] = {}
    for arm in arms:
        summary[arm] = {
            key: sum(row[key] for row in per_seed[arm]) / len(per_seed[arm])
            for key in per_seed[arm][0]
        }

    def difference(left: str, right: str, metric: str) -> list[float]:
        return [per_seed[left][i][metric] - per_seed[right][i][metric] for i in range(len(per_seed[left]))]

    tag = int(ood) * 100
    summary["effects"] = {
        "contrast_minus_ddm_accuracy_lcb": _lcb(difference("contrast_gravity", "fixed_ddm", "accuracy"), seed=20261301 + tag),
        "contrast_minus_ddm_utility_lcb": _lcb(difference("contrast_gravity", "fixed_ddm", "utility"), seed=20261302 + tag),
        "contrast_minus_shuffle_accuracy_lcb": _lcb(difference("contrast_gravity", "contrast_shuffle", "accuracy"), seed=20261303 + tag),
        "contrast_minus_sign_flip_accuracy_lcb": _lcb(difference("contrast_gravity", "contrast_sign_flip", "accuracy"), seed=20261304 + tag),
    }
    summary["memory_trace_identical"] = trace_identity
    return summary


def evaluate_density_contrast(
    config: DynamicGravityBenchConfig | None = None,
) -> dict[str, object]:
    cfg = config or DynamicGravityBenchConfig()
    basis = _field_basis(cfg.base)
    zero_source = _source(0.0, basis, cfg, flip=False, contrast=True)
    potential = np.zeros_like(basis.grid)
    velocity = np.zeros_like(basis.grid)
    max_zero_field = 0.0
    for _ in range(100):
        potential, velocity = _field_step(potential, velocity, zero_source, basis, cfg)
        max_zero_field = max(
            max_zero_field,
            float(np.max(np.abs(potential))),
            float(np.max(np.abs(velocity))),
        )
    id_result = _contrast_domain(basis, cfg, ood=False)
    ood_result = _contrast_domain(basis, cfg, ood=True)
    id_effects = id_result["effects"]
    ood_effects = ood_result["effects"]
    rows = (id_result["contrast_gravity"], ood_result["contrast_gravity"])
    gates = {
        "equal_prior_zero_field": (
            float(np.max(np.abs(zero_source))) <= 1e-12 and max_zero_field <= 1e-12
        ),
        "ddm_accuracy_noninferior": (
            id_effects["contrast_minus_ddm_accuracy_lcb"] >= -0.01
            and ood_effects["contrast_minus_ddm_accuracy_lcb"] >= -0.01
        ),
        "ddm_utility_noninferior": (
            id_effects["contrast_minus_ddm_utility_lcb"] >= 0.0
            and ood_effects["contrast_minus_ddm_utility_lcb"] >= 0.0
        ),
        "source_alignment": (
            id_effects["contrast_minus_shuffle_accuracy_lcb"] >= 0.10
            and ood_effects["contrast_minus_shuffle_accuracy_lcb"] >= 0.10
        ),
        "source_sign": (
            id_effects["contrast_minus_sign_flip_accuracy_lcb"] >= 0.20
            and ood_effects["contrast_minus_sign_flip_accuracy_lcb"] >= 0.20
        ),
        "graded_capture_time": all(
            row["low_coherence_steps"] - row["high_coherence_steps"] >= 10.0
            for row in rows
        ),
        "capture_stable": all(
            row["capture_rate"] >= 0.90
            and row["flip_rate"] <= 0.02
            and bool(row["bounded"])
            and row["max_field_energy"] <= 1e4
            for row in rows
        ),
        "integrity": bool(id_result["memory_trace_identical"]) and bool(ood_result["memory_trace_identical"]),
    }
    hard_gate = all(gates.values())
    return {
        "schema": "clarus.density-contrast-gravity.validation.v1",
        "config": asdict(cfg),
        "equal_prior": {
            "source_max_abs": float(np.max(np.abs(zero_source))),
            "field_max_abs": max_zero_field,
        },
        "id": id_result,
        "ood": ood_result,
        "future_reads": 0,
        "environment_clone_calls": 0,
        "gates": gates,
        "hard_gate": hard_gate,
        "score": 100 if hard_gate else 0,
        "decision": "GO" if hard_gate else "STOP",
    }
