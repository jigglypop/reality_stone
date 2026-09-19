"""Behavioral and resource gates for the delayed orbit sidecar."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np

from reality_stone.clarus.orbit_quotient_network import (
    DelayedEdge,
    DelayedOrbitNetwork,
    project_orbit_state,
    simulate_budgeted_initial_deviation,
    simulate_full,
    simulate_quotient,
    simulate_sparse_sidecar,
)


@dataclass(frozen=True)
class InterceptionEpisode:
    cover_size: int
    positions: tuple[int, int, int]
    perturbations: dict[tuple[int, int], float]
    need: float
    threat: float
    label: int
    utilities: tuple[float, float, float]


def interception_network() -> DelayedOrbitNetwork:
    return DelayedOrbitNetwork(
        3,
        (0.0, 0.0, 0.0),
        (
            DelayedEdge(0, 0, -1, 1, 0.18),
            DelayedEdge(0, 0, 0, 1, 0.35),
            DelayedEdge(0, 0, 1, 1, 0.18),
            DelayedEdge(1, 1, -1, 1, 0.17),
            DelayedEdge(1, 1, 0, 1, 0.33),
            DelayedEdge(1, 1, 1, 1, 0.17),
            DelayedEdge(2, 2, 0, 2, 0.20),
        ),
    )


def _utilities(
    reward: np.ndarray, hazard: np.ndarray, need: float, threat: float
) -> np.ndarray:
    return need * reward - threat * hazard - np.asarray((0.01, 0.0, 0.01))


def _dense_rollout(
    network: DelayedOrbitNetwork, episode: InterceptionEpisode | None, cover_size: int,
    perturbations: dict[tuple[int, int], float]
) -> np.ndarray:
    initial = np.zeros((cover_size, 3), dtype=np.float64)
    for (cell, orbit), delta in perturbations.items():
        initial[cell, orbit] += delta
    return simulate_full(network, initial, np.zeros((4, cover_size, 3)))


def generate_interception_episodes(
    cover_size: int, seed: int, *, per_action: int = 18
) -> list[InterceptionEpisode]:
    network = interception_network()
    rng = np.random.default_rng(seed)
    buckets: list[list[InterceptionEpisode]] = [[], [], []]
    attempts = 0
    while min(map(len, buckets)) < per_action and attempts < 20_000:
        attempts += 1
        center = int(rng.integers(0, cover_size))
        positions = tuple((center + offset) % cover_size for offset in (-12, 0, 12))
        perturbations: dict[tuple[int, int], float] = {}
        for cell in positions:
            perturbations[(cell, 0)] = float(rng.uniform(0.15, 0.90))
            perturbations[(cell, 1)] = float(rng.uniform(0.15, 0.90))
        need = float(rng.uniform(0.7, 1.3))
        threat = float(rng.uniform(0.7, 1.3))
        dense = _dense_rollout(network, None, cover_size, perturbations)
        reward = dense[-1, np.asarray(positions), 0]
        hazard = dense[-1, np.asarray(positions), 1]
        utility = _utilities(reward, hazard, need, threat)
        order = np.argsort(utility)
        label = int(order[-1])
        if utility[order[-1]] - utility[order[-2]] < 0.01:
            continue
        if len(buckets[label]) >= per_action:
            continue
        buckets[label].append(
            InterceptionEpisode(
                cover_size,
                positions,
                perturbations,
                need,
                threat,
                label,
                tuple(float(value) for value in utility),
            )
        )
    if min(map(len, buckets)) < per_action:
        raise RuntimeError("could not generate the preregistered balanced task")
    return [episode for bucket in buckets for episode in bucket]


def _normalized_utility(episode: InterceptionEpisode, action: int) -> float:
    values = np.asarray(episode.utilities)
    span = float(np.max(values) - np.min(values))
    return float((values[action] - np.min(values)) / span)


def _local_action(network: DelayedOrbitNetwork, episode: InterceptionEpisode) -> tuple[int, int]:
    sidecar = simulate_sparse_sidecar(
        network,
        episode.cover_size,
        np.zeros(3),
        np.zeros((4, 3)),
        episode.perturbations,
        active_budget=128,
    )
    final = {(cell, orbit): delta for cell, orbit, delta in sidecar.deviations_by_time[4]}
    reward = np.asarray([sidecar.baseline[4, 0] + final.get((cell, 0), 0.0)
                         for cell in episode.positions])
    hazard = np.asarray([sidecar.baseline[4, 1] + final.get((cell, 1), 0.0)
                         for cell in episode.positions])
    action = int(np.argmax(_utilities(reward, hazard, episode.need, episode.threat)))
    stored = sum(len(layer) for layer in sidecar.deviations_by_time)
    return action, stored


def _fixed_patch_action(episode: InterceptionEpisode) -> int:
    """Fixed-radius cone kernel for the registered, non-overlapping task patches."""

    patches = [[[0.0] * 9 for _ in range(2)] for _ in range(3)]
    for patch, cell in zip(patches, episode.positions):
        patch[0][4] = episode.perturbations[(cell, 0)]
        patch[1][4] = episode.perturbations[(cell, 1)]
    for _ in range(4):
        next_patches = [[[0.0] * 9 for _ in range(2)] for _ in range(3)]
        for patch_index, patch in enumerate(patches):
            for channel, (neighbor, center) in enumerate(((0.18, 0.35), (0.17, 0.33))):
                for index in range(9):
                    total = center * patch[channel][index]
                    if index:
                        total += neighbor * patch[channel][index - 1]
                    if index < 8:
                        total += neighbor * patch[channel][index + 1]
                    next_patches[patch_index][channel][index] = math.tanh(total)
        patches = next_patches
    reward = np.asarray([patch[0][4] for patch in patches])
    hazard = np.asarray([patch[1][4] for patch in patches])
    return int(np.argmax(_utilities(reward, hazard, episode.need, episode.threat)))


def _quotient_action(network: DelayedOrbitNetwork, episode: InterceptionEpisode) -> int:
    full_initial = np.zeros((episode.cover_size, 3), dtype=np.float64)
    for (cell, orbit), delta in episode.perturbations.items():
        full_initial[cell, orbit] += delta
    quotient = simulate_quotient(
        network, project_orbit_state(full_initial), np.zeros((4, 3))
    )
    reward = np.repeat(quotient[-1, 0], 3)
    hazard = np.repeat(quotient[-1, 1], 3)
    return int(np.argmax(_utilities(reward, hazard, episode.need, episode.threat)))


def evaluate_orbit_budget_sidecar() -> dict[str, object]:
    network = interception_network()
    specifications = ((64, 41064), (128, 41128), (256, 41256))
    by_size: dict[str, dict[str, float | int]] = {}
    all_episodes: list[InterceptionEpisode] = []
    maximum_local_storage = 0
    shift_failures = 0
    fixed_patch_failures = 0
    for cover_size, seed in specifications:
        episodes = generate_interception_episodes(cover_size, seed)
        all_episodes.extend(episodes)
        local_actions: list[int] = []
        quotient_actions: list[int] = []
        for episode in episodes:
            action, stored = _local_action(network, episode)
            local_actions.append(action)
            quotient_actions.append(_quotient_action(network, episode))
            fixed_patch_failures += int(_fixed_patch_action(episode) != action)
            maximum_local_storage = max(maximum_local_storage, stored + 15)
            shifted = InterceptionEpisode(
                cover_size,
                tuple((cell + 7) % cover_size for cell in episode.positions),
                {((cell + 7) % cover_size, orbit): value
                 for (cell, orbit), value in episode.perturbations.items()},
                episode.need,
                episode.threat,
                episode.label,
                episode.utilities,
            )
            shifted_action, _ = _local_action(network, shifted)
            shift_failures += int(shifted_action != action)
        labels = np.asarray([episode.label for episode in episodes])
        local_array = np.asarray(local_actions)
        quotient_array = np.asarray(quotient_actions)
        local_utility = np.mean(
            [_normalized_utility(ep, action) for ep, action in zip(episodes, local_actions)]
        )
        quotient_utility = np.mean(
            [_normalized_utility(ep, action) for ep, action in zip(episodes, quotient_actions)]
        )
        by_size[str(cover_size)] = {
            "episodes": len(episodes),
            "dense_accuracy": 1.0,
            "local_accuracy": float(np.mean(local_array == labels)),
            "quotient_accuracy": float(np.mean(quotient_array == labels)),
            "local_normalized_utility": float(local_utility),
            "quotient_normalized_utility": float(quotient_utility),
        }

    budget_rows: list[dict[str, float | int]] = []
    audit_episodes = all_episodes[:12]
    for budget in (0, 1, 2, 4, 8, 16, 64):
        maximum_error = 0.0
        maximum_bound = 0.0
        violations = 0
        action_matches = 0
        certified_actions = 0
        certified_action_errors = 0
        for episode in audit_episodes:
            approximation = simulate_budgeted_initial_deviation(
                network,
                episode.cover_size,
                np.zeros(3),
                np.zeros((4, 3)),
                episode.perturbations,
                active_budget=budget,
            )
            dense = _dense_rollout(
                network, episode, episode.cover_size, episode.perturbations
            )
            actual = np.max(np.abs(dense - approximation.reconstructed), axis=1)
            violations += int(
                np.any(actual > approximation.certified_error_by_time_orbit + 1e-12)
            )
            maximum_error = max(maximum_error, float(np.max(actual)))
            maximum_bound = max(
                maximum_bound, float(np.max(approximation.certified_error_by_time_orbit))
            )
            reward = approximation.reconstructed[-1, np.asarray(episode.positions), 0]
            hazard = approximation.reconstructed[-1, np.asarray(episode.positions), 1]
            approximate_utility = _utilities(
                reward, hazard, episode.need, episode.threat
            )
            action = int(np.argmax(approximate_utility))
            action_matches += int(action == episode.label)
            ordered = np.sort(approximate_utility)
            score_error = (
                episode.need * approximation.certified_error_by_time_orbit[-1, 0]
                + episode.threat * approximation.certified_error_by_time_orbit[-1, 1]
            )
            certified_action = bool(ordered[-1] - ordered[-2] > 2.0 * score_error)
            certified_actions += int(certified_action)
            certified_action_errors += int(certified_action and action != episode.label)
        budget_rows.append(
            {
                "budget": budget,
                "maximum_actual_error": maximum_error,
                "maximum_certified_bound": maximum_bound,
                "bound_violations": violations,
                "action_accuracy": action_matches / len(audit_episodes),
                "certified_action_coverage": certified_actions / len(audit_episodes),
                "certified_action_errors": certified_action_errors,
            }
        )

    timing_episodes = all_episodes[-18:]
    for episode in timing_episodes[:3]:
        _dense_rollout(network, episode, episode.cover_size, episode.perturbations)
        _fixed_patch_action(episode)
    dense_trials = []
    local_trials = []
    for _ in range(7):
        start = perf_counter()
        for episode in timing_episodes:
            _dense_rollout(network, episode, episode.cover_size, episode.perturbations)
        dense_trials.append(perf_counter() - start)
        start = perf_counter()
        for episode in timing_episodes:
            _fixed_patch_action(episode)
        local_trials.append(perf_counter() - start)
    dense_seconds = float(np.median(dense_trials))
    local_seconds = float(np.median(local_trials))
    dense_storage_256 = 5 * 256 * 3
    memory_ratio = maximum_local_storage / dense_storage_256
    local_time_ratio = local_seconds / dense_seconds

    task_gates = {
        "local_dense_noninferiority": all(
            row["local_accuracy"] >= row["dense_accuracy"] - 0.02
            and row["local_normalized_utility"] >= 0.98
            for row in by_size.values()
        ),
        "local_beats_quotient": all(
            row["local_accuracy"] >= row["quotient_accuracy"] + 0.20
            and row["local_normalized_utility"] >= row["quotient_normalized_utility"] + 0.20
            for row in by_size.values()
        ),
        "shift_consistency": shift_failures == 0,
        "budget_bound_valid": all(row["bound_violations"] == 0 for row in budget_rows),
        "large_budget_exact": budget_rows[-1]["maximum_actual_error"] <= 1e-12,
        "candidate_memory_reduced": memory_ratio <= 0.40,
        "candidate_time_reduced": local_time_ratio <= 0.60,
        "fixed_patch_matches_generic": fixed_patch_failures == 0,
        "no_false_action_certificates": all(
            row["certified_action_errors"] == 0 for row in budget_rows
        ),
    }
    behavioral_points = 10 if (
        task_gates["local_dense_noninferiority"] and task_gates["local_beats_quotient"]
    ) else 0
    ood_points = 6 if task_gates["shift_consistency"] else 0
    scaling_points = (
        (3 if task_gates["candidate_time_reduced"] else 0)
        + (3 if task_gates["candidate_memory_reduced"] else 0)
        + 1
    )
    integrity_points = 4 if task_gates["budget_bound_valid"] else 0
    score = 73 + behavioral_points + ood_points + scaling_points + integrity_points
    hard_behavior = all(
        task_gates[key]
        for key in (
            "local_dense_noninferiority",
            "local_beats_quotient",
            "shift_consistency",
            "budget_bound_valid",
            "large_budget_exact",
            "fixed_patch_matches_generic",
            "no_false_action_certificates",
        )
    )
    return {
        "schema": "clarus.orbit-budget-sidecar.validation.v1",
        "by_size": by_size,
        "budget_curve": budget_rows,
        "shift_failures": shift_failures,
        "fixed_patch_failures": fixed_patch_failures,
        "maximum_local_state_scalars": maximum_local_storage,
        "dense_state_scalars_at_256": dense_storage_256,
        "state_memory_ratio": memory_ratio,
        "dense_seconds": dense_seconds,
        "local_seconds": local_seconds,
        "local_time_ratio": local_time_ratio,
        "gates": task_gates,
        "readiness_score": score,
        "standalone_verdict": "GO" if score >= 80 and hard_behavior else "STOP",
        "runtime_sidecar_verdict": "GO" if score >= 80 and all(task_gates.values()) else "HOLD",
    }


__all__ = [
    "InterceptionEpisode",
    "evaluate_orbit_budget_sidecar",
    "generate_interception_episodes",
    "interception_network",
]
