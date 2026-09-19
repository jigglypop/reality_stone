"""Loop 8E screened-field decision benchmark.

The gravitational language denotes an effective source-field-motion model on
a decision space. It is not a physical gravity or biological identity claim.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .brain_geometry_benchmark import (
    BrainGeometryBenchConfig,
    ResidualReplayBenchConfig,
    _lcb,
    _residual_trials,
    _run_residual_arm,
    _stream,
)


@dataclass(frozen=True)
class GravityDecisionBenchConfig:
    residual: ResidualReplayBenchConfig = ResidualReplayBenchConfig()
    grid_min: float = -3.0
    grid_max: float = 3.0
    grid_points: int = 241
    source_position: float = 1.0
    source_width: float = 0.30
    screening: float = 0.60
    coupling: float = 1.00
    dt: float = 0.02
    evidence_temperature_inverse: float = 2.0
    memory_weight: float = 0.25
    id_coherences: tuple[float, ...] = (0.10, 0.20, 0.40, 0.70)
    ood_coherences: tuple[float, ...] = (0.05, 0.15, 0.30, 0.60)
    id_evidence_noise: float = 0.35
    ood_evidence_noise: float = 0.45
    friction: float = 1.0
    decision_temperature: float = 0.005
    max_steps: int = 500
    minimum_capture_steps: int = 5
    capture_tolerance: float = 1e-6
    post_capture_steps: int = 50
    ddm_boundary: float = 1.0
    linear_low_boundary: float = 0.70
    conflict_reference: float = 0.70
    time_cost: float = 0.002


@dataclass(frozen=True)
class _FieldBasis:
    grid: np.ndarray
    spacing: float
    plus_source: np.ndarray
    minus_source: np.ndarray
    plus_potential: np.ndarray
    minus_potential: np.ndarray
    plus_force: np.ndarray
    minus_force: np.ndarray
    plus_residual: float
    minus_residual: float


def _field_basis(config: GravityDecisionBenchConfig) -> _FieldBasis:
    grid = np.linspace(config.grid_min, config.grid_max, config.grid_points)
    spacing = float(grid[1] - grid[0])
    laplacian = np.zeros((config.grid_points, config.grid_points), dtype=np.float64)
    diagonal = np.full(config.grid_points, 2.0 / spacing**2)
    diagonal[0] = diagonal[-1] = 1.0 / spacing**2
    np.fill_diagonal(laplacian, diagonal)
    off_diagonal = np.full(config.grid_points - 1, -1.0 / spacing**2)
    laplacian[np.arange(config.grid_points - 1), np.arange(1, config.grid_points)] = off_diagonal
    laplacian[np.arange(1, config.grid_points), np.arange(config.grid_points - 1)] = off_diagonal
    operator = laplacian + config.screening**2 * np.eye(config.grid_points)

    potentials: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    residuals: list[float] = []
    forces: list[np.ndarray] = []
    for position in (config.source_position, -config.source_position):
        density = np.exp(-0.5 * ((grid - position) / config.source_width) ** 2)
        density /= float(density.sum() * spacing)
        source = density - density.mean()
        sources.append(source)
        right_hand_side = -config.coupling * source
        potential = np.linalg.solve(operator, right_hand_side)
        residuals.append(float(np.max(np.abs(operator @ potential - right_hand_side))))
        potentials.append(potential)
        forces.append(-np.gradient(potential, spacing, edge_order=2))
    return _FieldBasis(
        grid=grid,
        spacing=spacing,
        plus_source=sources[0],
        minus_source=sources[1],
        plus_potential=potentials[0],
        minus_potential=potentials[1],
        plus_force=forces[0],
        minus_force=forces[1],
        plus_residual=residuals[0],
        minus_residual=residuals[1],
    )


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _combine_field(
    evidence: float,
    basis: _FieldBasis,
    config: GravityDecisionBenchConfig,
    *,
    flip: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    plus_mass = _sigmoid(config.evidence_temperature_inverse * evidence)
    if flip:
        plus_mass = 1.0 - plus_mass
    minus_mass = 1.0 - plus_mass
    potential = plus_mass * basis.plus_potential + minus_mass * basis.minus_potential
    force = plus_mass * basis.plus_force + minus_mass * basis.minus_force
    return potential, force


def _central_force(
    plus_mass: float, basis: _FieldBasis
) -> float:
    force = plus_mass * basis.plus_force + (1.0 - plus_mass) * basis.minus_force
    center = len(basis.grid) // 2
    return float(force[center])


def _evidence_path(
    memory_signal: float,
    target: int,
    coherence: float,
    seed: int,
    trial_index: int,
    config: GravityDecisionBenchConfig,
    *,
    ood: bool,
) -> tuple[float, ...]:
    noise_scale = config.ood_evidence_noise if ood else config.id_evidence_noise
    rng = _stream(seed * 1049 + trial_index, 301)
    evidence = config.memory_weight * memory_signal
    path: list[float] = []
    for _ in range(config.max_steps):
        evidence += (
            target * coherence * config.dt
            + noise_scale * math.sqrt(config.dt) * rng.gauss(0.0, 1.0)
        )
        path.append(evidence)
    return tuple(path)


def _reflect(position: float, velocity: float, config: GravityDecisionBenchConfig) -> tuple[float, float]:
    if position < config.grid_min:
        return 2.0 * config.grid_min - position, -velocity
    if position > config.grid_max:
        return 2.0 * config.grid_max - position, -velocity
    return position, velocity


def _gravity_trial(
    path: tuple[float, ...],
    target: int,
    seed: int,
    trial_index: int,
    basis: _FieldBasis,
    config: GravityDecisionBenchConfig,
    *,
    flip: bool = False,
) -> dict[str, float | bool | int]:
    motion_rng = _stream(seed * 1051 + trial_index, 307)
    position = velocity = 0.0
    captured = False
    captured_side = 0
    capture_step = config.max_steps
    frozen_potential: np.ndarray | None = None
    frozen_force: np.ndarray | None = None
    frozen_saddle = 0.0
    bounded = True
    max_abs_velocity = 0.0

    for index, evidence in enumerate(path):
        potential, force = _combine_field(evidence, basis, config, flip=flip)
        between = np.where(
            (basis.grid >= -config.source_position)
            & (basis.grid <= config.source_position)
        )[0]
        saddle_index = int(between[np.argmax(potential[between])])
        saddle_position = float(basis.grid[saddle_index])
        saddle_potential = float(potential[saddle_index])
        local_force = float(np.interp(position, basis.grid, force))
        old_velocity = velocity
        position += old_velocity * config.dt
        velocity += (
            (-config.friction * old_velocity + local_force) * config.dt
            + math.sqrt(
                2.0
                * config.friction
                * config.decision_temperature
                * config.dt
            )
            * motion_rng.gauss(0.0, 1.0)
        )
        position, velocity = _reflect(position, velocity, config)
        max_abs_velocity = max(max_abs_velocity, abs(velocity))
        bounded = bounded and math.isfinite(position) and math.isfinite(velocity)
        energy = 0.5 * velocity * velocity + float(np.interp(position, basis.grid, potential))
        if (
            index + 1 >= config.minimum_capture_steps
            and energy < saddle_potential - config.capture_tolerance
            and abs(position - saddle_position) > 1e-12
        ):
            captured = True
            captured_side = 1 if position > saddle_position else -1
            capture_step = index + 1
            frozen_potential = potential
            frozen_force = force
            frozen_saddle = saddle_position
            break

    if not captured:
        captured_side = 1 if position >= 0.0 else -1
        frozen_potential, frozen_force = _combine_field(path[-1], basis, config, flip=flip)
        between = np.where(
            (basis.grid >= -config.source_position)
            & (basis.grid <= config.source_position)
        )[0]
        frozen_saddle = float(basis.grid[int(between[np.argmax(frozen_potential[between])])])

    side_flipped = False
    initial_side = captured_side
    assert frozen_force is not None and frozen_potential is not None
    for _ in range(config.post_capture_steps):
        local_force = float(np.interp(position, basis.grid, frozen_force))
        old_velocity = velocity
        position += old_velocity * config.dt
        velocity += (
            (-config.friction * old_velocity + local_force) * config.dt
            + math.sqrt(
                2.0
                * config.friction
                * config.decision_temperature
                * config.dt
            )
            * motion_rng.gauss(0.0, 1.0)
        )
        position, velocity = _reflect(position, velocity, config)
        current_side = 1 if position > frozen_saddle else -1
        side_flipped = side_flipped or current_side != initial_side
        max_abs_velocity = max(max_abs_velocity, abs(velocity))
        bounded = bounded and math.isfinite(position) and math.isfinite(velocity)

    correct = captured_side == target
    return {
        "correct": correct,
        "steps": capture_step,
        "utility": (1.0 if correct else -1.0) - config.time_cost * capture_step,
        "captured": captured,
        "side_flipped": side_flipped,
        "bounded": bounded,
        "max_abs_velocity": max_abs_velocity,
    }


def _threshold_trial(
    path: tuple[float, ...],
    target: int,
    boundary: float,
    config: GravityDecisionBenchConfig,
) -> dict[str, float | bool | int]:
    steps = config.max_steps
    evidence = path[-1]
    for index, value in enumerate(path):
        if abs(value) >= boundary:
            steps = index + 1
            evidence = value
            break
    choice = 1 if evidence >= 0.0 else -1
    correct = choice == target
    return {
        "correct": correct,
        "steps": steps,
        "utility": (1.0 if correct else -1.0) - config.time_cost * steps,
        "captured": True,
        "side_flipped": False,
        "bounded": True,
        "max_abs_velocity": 0.0,
    }


def _domain(
    basis: _FieldBasis,
    config: GravityDecisionBenchConfig,
    *,
    ood: bool,
) -> dict[str, object]:
    arms = ("fixed_ddm", "linear_stn", "gravity_capture", "gravity_mass_shuffle", "gravity_sign_flip")
    per_seed = {arm: [] for arm in arms}
    trace_identity = True
    levels = config.ood_coherences if ood else config.id_coherences
    for offset in range(config.residual.base.seeds):
        seed = 850_000 + offset
        trials = _residual_trials(seed, config.residual, ood=ood, stationary=False)
        memory_result = _run_residual_arm(
            trials,
            "residual_replay",
            seed,
            config.residual,
            ood=ood,
            return_trace=True,
        )
        trace = memory_result["decision_trace"]
        frozen_trace = tuple(trace)
        coherence_rng = _stream(seed, 401 if not ood else 403)
        coherences = [levels[coherence_rng.randrange(len(levels))] for _ in trace]
        paths = [
            _evidence_path(signal, target, coherences[index], seed, index, config, ood=ood)
            for index, (signal, target) in enumerate(trace)
        ]
        permutation = list(range(len(paths)))
        _stream(seed, 409 if not ood else 411).shuffle(permutation)
        for arm in arms:
            rows: list[dict[str, float | bool | int]] = []
            for index, (_, target) in enumerate(trace):
                if arm == "fixed_ddm":
                    row = _threshold_trial(paths[index], target, config.ddm_boundary, config)
                elif arm == "linear_stn":
                    conflict = min(
                        1.0,
                        max(0.0, 1.0 - coherences[index] / config.conflict_reference),
                    )
                    row = _threshold_trial(
                        paths[index],
                        target,
                        config.linear_low_boundary + conflict,
                        config,
                    )
                elif arm == "gravity_mass_shuffle":
                    row = _gravity_trial(
                        paths[permutation[index]], target, seed, index, basis, config
                    )
                else:
                    row = _gravity_trial(
                        paths[index],
                        target,
                        seed,
                        index,
                        basis,
                        config,
                        flip=arm == "gravity_sign_flip",
                    )
                rows.append(row)
            per_seed[arm].append((rows, tuple(coherences)))
            trace_identity = trace_identity and tuple(trace) == frozen_trace

    summary: dict[str, object] = {}
    for arm in arms:
        seed_metrics: list[dict[str, float]] = []
        for rows, coherences in per_seed[arm]:
            low_levels = set(sorted(levels)[:2])
            high_levels = set(sorted(levels)[2:])
            low_indices = [index for index, value in enumerate(coherences) if value in low_levels]
            high_indices = [index for index, value in enumerate(coherences) if value in high_levels]
            seed_metrics.append({
                "accuracy": sum(float(row["correct"]) for row in rows) / len(rows),
                "utility": sum(float(row["utility"]) for row in rows) / len(rows),
                "steps": sum(float(row["steps"]) for row in rows) / len(rows),
                "low_coherence_steps": sum(float(rows[index]["steps"]) for index in low_indices) / len(low_indices),
                "high_coherence_steps": sum(float(rows[index]["steps"]) for index in high_indices) / len(high_indices),
                "capture_rate": sum(float(row["captured"]) for row in rows) / len(rows),
                "flip_rate": sum(float(row["side_flipped"]) for row in rows) / len(rows),
                "bounded": float(all(bool(row["bounded"]) for row in rows)),
                "max_abs_velocity": max(float(row["max_abs_velocity"]) for row in rows),
            })
        summary[arm] = {
            key: sum(row[key] for row in seed_metrics) / len(seed_metrics)
            for key in seed_metrics[0]
        }
        per_seed[arm] = seed_metrics

    def difference(left: str, right: str, metric: str) -> list[float]:
        return [
            float(per_seed[left][index][metric]) - float(per_seed[right][index][metric])
            for index in range(config.residual.base.seeds)
        ]

    tag = int(ood) * 100
    summary["effects"] = {
        "gravity_minus_ddm_accuracy_lcb": _lcb(
            difference("gravity_capture", "fixed_ddm", "accuracy"), seed=20261101 + tag
        ),
        "gravity_minus_stn_utility_lcb": _lcb(
            difference("gravity_capture", "linear_stn", "utility"), seed=20261102 + tag
        ),
        "gravity_minus_shuffle_accuracy_lcb": _lcb(
            difference("gravity_capture", "gravity_mass_shuffle", "accuracy"), seed=20261103 + tag
        ),
        "gravity_minus_sign_flip_accuracy_lcb": _lcb(
            difference("gravity_capture", "gravity_sign_flip", "accuracy"), seed=20261104 + tag
        ),
    }
    summary["memory_trace_identical"] = trace_identity
    return summary


def evaluate_gravitational_decision(
    config: GravityDecisionBenchConfig | None = None,
) -> dict[str, object]:
    cfg = config or GravityDecisionBenchConfig()
    basis = _field_basis(cfg)
    equal_force = _central_force(0.5, basis)
    positive_force = _central_force(0.7, basis)
    negative_force = _central_force(0.3, basis)
    id_result = _domain(basis, cfg, ood=False)
    ood_result = _domain(basis, cfg, ood=True)
    id_effects = id_result["effects"]
    ood_effects = ood_result["effects"]
    gravity_rows = (id_result["gravity_capture"], ood_result["gravity_capture"])
    gates = {
        "field_residual": max(basis.plus_residual, basis.minus_residual) <= 1e-10,
        "equal_mass_symmetry": abs(equal_force) <= 1e-12,
        "force_direction": (
            positive_force > 0.0
            and negative_force < 0.0
            and abs(abs(positive_force) - abs(negative_force)) <= 1e-12
        ),
        "beats_ddm_accuracy": (
            id_effects["gravity_minus_ddm_accuracy_lcb"] >= 0.02
            and ood_effects["gravity_minus_ddm_accuracy_lcb"] >= 0.0
        ),
        "beats_linear_stn_utility": (
            id_effects["gravity_minus_stn_utility_lcb"] >= 0.0
            and ood_effects["gravity_minus_stn_utility_lcb"] >= 0.0
        ),
        "source_alignment": (
            id_effects["gravity_minus_shuffle_accuracy_lcb"] >= 0.10
            and ood_effects["gravity_minus_shuffle_accuracy_lcb"] >= 0.10
        ),
        "source_sign": (
            id_effects["gravity_minus_sign_flip_accuracy_lcb"] >= 0.20
            and ood_effects["gravity_minus_sign_flip_accuracy_lcb"] >= 0.20
        ),
        "coherence_orders_time": all(
            row["low_coherence_steps"] > row["high_coherence_steps"]
            for row in gravity_rows
        ),
        "capture_stable": all(
            row["capture_rate"] >= 0.95
            and row["flip_rate"] <= 0.02
            and bool(row["bounded"])
            for row in gravity_rows
        ),
        "integrity": (
            bool(id_result["memory_trace_identical"])
            and bool(ood_result["memory_trace_identical"])
        ),
    }
    hard_gate = all(gates.values())
    return {
        "schema": "clarus.gravitational-decision.validation.v1",
        "config": asdict(cfg),
        "field": {
            "plus_residual": basis.plus_residual,
            "minus_residual": basis.minus_residual,
            "equal_mass_central_force": equal_force,
            "positive_mass_difference_force": positive_force,
            "negative_mass_difference_force": negative_force,
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


def small_gravity_config(*, trials: int = 24, seeds: int = 2) -> GravityDecisionBenchConfig:
    base = BrainGeometryBenchConfig(trials=trials, seeds=seeds, blocks=(8, 12))
    residual = ResidualReplayBenchConfig(base=base)
    return GravityDecisionBenchConfig(residual=residual, max_steps=80, post_capture_steps=10)
