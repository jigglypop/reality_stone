"""Delayed translation-equivariant networks with exact finite orbit quotients.

The full carrier is a cyclic cover ``C_N x Q``.  The quotient represents only
the spatially constant sector; localized deviations are executed on their
finite causal cone.  All activities and gains are normalized/dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
Node = tuple[int, int]


@dataclass(frozen=True)
class DelayedEdge:
    target_orbit: int
    source_orbit: int
    shift: int
    delay: int
    weight: float


@dataclass(frozen=True)
class DelayedOrbitNetwork:
    orbit_count: int
    bias: tuple[float, ...]
    edges: tuple[DelayedEdge, ...]

    def __post_init__(self) -> None:
        if self.orbit_count <= 0 or len(self.bias) != self.orbit_count:
            raise ValueError("bias must contain one value per orbit")
        if not all(math.isfinite(value) for value in self.bias):
            raise ValueError("bias must be finite")
        for edge in self.edges:
            if not 0 <= edge.target_orbit < self.orbit_count:
                raise ValueError("target orbit is out of range")
            if not 0 <= edge.source_orbit < self.orbit_count:
                raise ValueError("source orbit is out of range")
            if edge.delay < 1:
                raise ValueError("delays must be positive; same-tick reads are forbidden")
            if not math.isfinite(edge.weight):
                raise ValueError("edge weights must be finite")

    @property
    def maximum_delay(self) -> int:
        return max((edge.delay for edge in self.edges), default=1)

    @property
    def maximum_shift(self) -> int:
        return max((abs(edge.shift) for edge in self.edges), default=0)

    @property
    def small_gain(self) -> float:
        return max(
            sum(abs(edge.weight) for edge in self.edges if edge.target_orbit == target)
            for target in range(self.orbit_count)
        )

    @property
    def quotient_work(self) -> int:
        return self.orbit_count + len(self.edges)


@dataclass(frozen=True)
class DelayedSnapshot:
    time: int
    history: tuple[FloatArray, ...]


@dataclass(frozen=True)
class SparseConeResult:
    baseline: FloatArray
    reconstructed: FloatArray
    active_by_time: tuple[tuple[Node, ...], ...]
    work: int


@dataclass(frozen=True)
class ApproximateConeResult:
    baseline: FloatArray
    reconstructed: FloatArray
    retained_by_time: tuple[tuple[Node, ...], ...]
    dropped_by_time: tuple[tuple[Node, ...], ...]
    candidate_counts: tuple[int, ...]
    local_residual_by_time_orbit: FloatArray
    certified_error_by_time_orbit: FloatArray
    exhausted_by_time: tuple[bool, ...]
    tie_drops_by_time: tuple[int, ...]
    candidate_evaluations: int
    edge_evaluations: int


@dataclass(frozen=True)
class SparseSidecarResult:
    baseline: FloatArray
    deviations_by_time: tuple[tuple[tuple[int, int, float], ...], ...]
    work: int

    def value(self, time: int, cell: int, orbit: int, cover_size: int) -> float:
        value = float(self.baseline[time, orbit])
        key = (int(cell) % cover_size, int(orbit))
        for stored_cell, stored_orbit, delta in self.deviations_by_time[time]:
            if (stored_cell, stored_orbit) == key:
                return value + delta
        return value


@lru_cache(maxsize=32)
def _edge_groups(
    network: DelayedOrbitNetwork,
) -> tuple[tuple[tuple[DelayedEdge, ...], ...], tuple[tuple[DelayedEdge, ...], ...]]:
    incoming = tuple(
        tuple(edge for edge in network.edges if edge.target_orbit == orbit)
        for orbit in range(network.orbit_count)
    )
    outgoing = tuple(
        tuple(edge for edge in network.edges if edge.source_orbit == orbit)
        for orbit in range(network.orbit_count)
    )
    return incoming, outgoing


def _vector(values: ArrayLike, size: int, name: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return result


def lift_orbit_trajectory(trajectory: ArrayLike, cover_size: int) -> FloatArray:
    state = np.asarray(trajectory, dtype=np.float64)
    if cover_size <= 0 or state.ndim < 1:
        raise ValueError("cover_size must be positive and trajectory must have an orbit axis")
    return np.repeat(state[..., np.newaxis, :], cover_size, axis=-2)


def project_orbit_state(state: ArrayLike) -> FloatArray:
    full = np.asarray(state, dtype=np.float64)
    if full.ndim != 2 or full.shape[0] == 0 or not np.all(np.isfinite(full)):
        raise ValueError("state must be a finite non-empty (cover, orbit) array")
    return np.mean(full, axis=0)


def initial_snapshot(network: DelayedOrbitNetwork, state: ArrayLike) -> DelayedSnapshot:
    current = np.asarray(state, dtype=np.float64)
    if current.shape[-1:] != (network.orbit_count,) or not np.all(np.isfinite(current)):
        raise ValueError("initial state has the wrong orbit dimension or is non-finite")
    zeros = tuple(np.zeros_like(current) for _ in range(network.maximum_delay - 1))
    return DelayedSnapshot(0, zeros + (current.copy(),))


def advance_quotient(
    network: DelayedOrbitNetwork,
    snapshot: DelayedSnapshot,
    external_input: ArrayLike,
) -> DelayedSnapshot:
    drive = _vector(external_input, network.orbit_count, "external_input")
    total = np.asarray(network.bias, dtype=np.float64) + drive
    for edge in network.edges:
        total[edge.target_orbit] += (
            edge.weight * snapshot.history[-edge.delay][edge.source_orbit]
        )
    next_state = np.tanh(total)
    history = snapshot.history[1:] + (next_state,)
    return DelayedSnapshot(snapshot.time + 1, history)


def advance_full(
    network: DelayedOrbitNetwork,
    snapshot: DelayedSnapshot,
    external_input: ArrayLike,
    *,
    boundary: str = "cyclic",
) -> DelayedSnapshot:
    current = snapshot.history[-1]
    if current.ndim != 2:
        raise ValueError("full state must have shape (cover, orbit)")
    cover_size = current.shape[0]
    drive = np.asarray(external_input, dtype=np.float64)
    if drive.shape != current.shape or not np.all(np.isfinite(drive)):
        raise ValueError("external_input must match the full state")
    if boundary not in {"cyclic", "open"}:
        raise ValueError("boundary must be 'cyclic' or 'open'")
    total = np.broadcast_to(np.asarray(network.bias), current.shape).copy() + drive
    for edge in network.edges:
        source = snapshot.history[-edge.delay][:, edge.source_orbit]
        if boundary == "cyclic":
            shifted = np.roll(source, edge.shift)
            total[:, edge.target_orbit] += edge.weight * shifted
        else:
            target_start = max(edge.shift, 0)
            target_stop = min(cover_size + edge.shift, cover_size)
            if target_start < target_stop:
                source_start = target_start - edge.shift
                source_stop = target_stop - edge.shift
                total[target_start:target_stop, edge.target_orbit] += (
                    edge.weight * source[source_start:source_stop]
                )
    next_state = np.tanh(total)
    history = snapshot.history[1:] + (next_state,)
    return DelayedSnapshot(snapshot.time + 1, history)


def simulate_quotient(
    network: DelayedOrbitNetwork,
    initial_state: ArrayLike,
    inputs: ArrayLike,
) -> FloatArray:
    drive = np.asarray(inputs, dtype=np.float64)
    if drive.ndim != 2 or drive.shape[1] != network.orbit_count:
        raise ValueError("inputs must have shape (steps, orbit)")
    snapshot = initial_snapshot(network, _vector(initial_state, network.orbit_count, "state"))
    states = [snapshot.history[-1].copy()]
    for row in drive:
        snapshot = advance_quotient(network, snapshot, row)
        states.append(snapshot.history[-1].copy())
    return np.asarray(states)


def simulate_full(
    network: DelayedOrbitNetwork,
    initial_state: ArrayLike,
    inputs: ArrayLike,
    *,
    boundary: str = "cyclic",
) -> FloatArray:
    state = np.asarray(initial_state, dtype=np.float64)
    drive = np.asarray(inputs, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != network.orbit_count:
        raise ValueError("initial_state must have shape (cover, orbit)")
    if drive.ndim != 3 or drive.shape[1:] != state.shape:
        raise ValueError("inputs must have shape (steps, cover, orbit)")
    snapshot = initial_snapshot(network, state)
    states = [snapshot.history[-1].copy()]
    for row in drive:
        snapshot = advance_full(network, snapshot, row, boundary=boundary)
        states.append(snapshot.history[-1].copy())
    return np.asarray(states)


def simulate_sparse_initial_deviation(
    network: DelayedOrbitNetwork,
    cover_size: int,
    initial_background: ArrayLike,
    inputs: ArrayLike,
    perturbations: Mapping[Node, float],
    *,
    active_budget: int | None = None,
) -> SparseConeResult:
    """Execute an exact localized deviation over a quotient background.

    The budget is checked per time slice.  Overflow raises instead of silently
    selecting an index-dependent subset that would destroy equivariance.
    """

    if cover_size <= 0:
        raise ValueError("cover_size must be positive")
    if active_budget is not None and active_budget <= 0:
        raise ValueError("active_budget must be positive")
    background = simulate_quotient(network, initial_background, inputs)
    steps = background.shape[0] - 1
    deviations: list[dict[Node, float]] = [dict() for _ in range(steps + 1)]
    for (cell, orbit), delta in perturbations.items():
        if not 0 <= orbit < network.orbit_count or not math.isfinite(delta):
            raise ValueError("invalid perturbation")
        key = (int(cell) % cover_size, int(orbit))
        deviations[0][key] = deviations[0].get(key, 0.0) + float(delta)
    if active_budget is not None and len(deviations[0]) > active_budget:
        raise RuntimeError("active causal cone exceeds budget")

    reconstructed = lift_orbit_trajectory(background, cover_size)
    for node, delta in deviations[0].items():
        reconstructed[0, node[0], node[1]] += delta
    work = len(deviations[0])
    drive = np.asarray(inputs, dtype=np.float64)
    for time in range(1, steps + 1):
        candidates: set[Node] = set()
        for edge in network.edges:
            source_time = time - edge.delay
            if source_time < 0:
                continue
            for cell, orbit in deviations[source_time]:
                if orbit == edge.source_orbit:
                    candidates.add(((cell + edge.shift) % cover_size, edge.target_orbit))
        if active_budget is not None and len(candidates) > active_budget:
            raise RuntimeError("active causal cone exceeds budget")
        for cell, target_orbit in candidates:
            total = network.bias[target_orbit] + drive[time - 1, target_orbit]
            for edge in network.edges:
                if edge.target_orbit != target_orbit:
                    continue
                source_time = time - edge.delay
                if source_time < 0:
                    source_value = 0.0
                else:
                    source_cell = (cell - edge.shift) % cover_size
                    source_value = background[source_time, edge.source_orbit]
                    source_value += deviations[source_time].get(
                        (source_cell, edge.source_orbit), 0.0
                    )
                total += edge.weight * source_value
            actual = math.tanh(total)
            delta = actual - background[time, target_orbit]
            if delta != 0.0:
                deviations[time][(cell, target_orbit)] = delta
                reconstructed[time, cell, target_orbit] = actual
        work += len(candidates) * (1 + len(network.edges))
    active = tuple(tuple(sorted(layer)) for layer in deviations)
    return SparseConeResult(background, reconstructed, active, work)


def simulate_sparse_sidecar(
    network: DelayedOrbitNetwork,
    cover_size: int,
    initial_background: ArrayLike,
    inputs: ArrayLike,
    perturbations: Mapping[Node, float],
    *,
    active_budget: int | None = None,
) -> SparseSidecarResult:
    """Exact sparse execution without materializing an ``N x Q`` trajectory."""

    if cover_size <= 0:
        raise ValueError("cover_size must be positive")
    if active_budget is not None and active_budget <= 0:
        raise ValueError("active_budget must be positive")
    initial_vector = _vector(initial_background, network.orbit_count, "state")
    drive = np.asarray(inputs, dtype=np.float64)
    if (
        not np.any(initial_vector)
        and not np.any(drive)
        and not any(network.bias)
    ):
        background = np.zeros((drive.shape[0] + 1, network.orbit_count))
    else:
        background = simulate_quotient(network, initial_vector, drive)
    steps = background.shape[0] - 1
    incoming, outgoing = _edge_groups(network)
    deviations: list[dict[Node, float]] = [dict() for _ in range(steps + 1)]
    for (cell, orbit), delta in perturbations.items():
        if not 0 <= orbit < network.orbit_count or not math.isfinite(delta):
            raise ValueError("invalid perturbation")
        key = (int(cell) % cover_size, int(orbit))
        deviations[0][key] = deviations[0].get(key, 0.0) + float(delta)
    if active_budget is not None and len(deviations[0]) > active_budget:
        raise RuntimeError("active causal cone exceeds budget")
    work = len(deviations[0])
    for time in range(1, steps + 1):
        candidates: set[Node] = set()
        for source_time in range(time):
            for cell, orbit in deviations[source_time]:
                for edge in outgoing[orbit]:
                    if time - edge.delay != source_time:
                        continue
                    candidates.add(((cell + edge.shift) % cover_size, edge.target_orbit))
        if active_budget is not None and len(candidates) > active_budget:
            raise RuntimeError("active causal cone exceeds budget")
        for cell, target_orbit in candidates:
            total = network.bias[target_orbit] + drive[time - 1, target_orbit]
            for edge in incoming[target_orbit]:
                source_time = time - edge.delay
                if source_time < 0:
                    source_value = 0.0
                else:
                    source_cell = (cell - edge.shift) % cover_size
                    source_value = background[source_time, edge.source_orbit]
                    source_value += deviations[source_time].get(
                        (source_cell, edge.source_orbit), 0.0
                    )
                total += edge.weight * source_value
            delta = math.tanh(total) - background[time, target_orbit]
            if delta != 0.0:
                deviations[time][(cell, target_orbit)] = delta
        work += len(candidates) * (1 + len(network.edges))
    records = tuple(
        tuple((cell, orbit, value) for (cell, orbit), value in layer.items())
        for layer in deviations
    )
    return SparseSidecarResult(background, records, work)


def _equivariant_magnitude_budget(
    candidates: Mapping[Node, float], active_budget: int
) -> tuple[dict[Node, float], dict[Node, float], int]:
    """Select complete magnitude tie classes without an absolute-index tie break."""

    if active_budget < 0:
        raise ValueError("active_budget must be nonnegative")
    if len(candidates) <= active_budget:
        return dict(candidates), {}, 0
    by_magnitude: dict[float, list[tuple[Node, float]]] = {}
    for node, value in candidates.items():
        by_magnitude.setdefault(abs(value), []).append((node, value))
    retained: dict[Node, float] = {}
    boundary_tie_drops = 0
    stopped = False
    for magnitude in sorted(by_magnitude, reverse=True):
        group = by_magnitude[magnitude]
        if not stopped and len(retained) + len(group) <= active_budget:
            retained.update(group)
        else:
            if not stopped and len(group) > 1:
                boundary_tie_drops = len(group)
            stopped = True
    dropped = {node: value for node, value in candidates.items() if node not in retained}
    return retained, dropped, boundary_tie_drops


def simulate_budgeted_initial_deviation(
    network: DelayedOrbitNetwork,
    cover_size: int,
    initial_background: ArrayLike,
    inputs: ArrayLike,
    perturbations: Mapping[Node, float],
    *,
    active_budget: int,
) -> ApproximateConeResult:
    """Approximate a local deviation and certify its orbit-wise sup error.

    Unlike :func:`simulate_sparse_initial_deviation`, this path may discard
    deviations.  It reports every discarded candidate and propagates a
    Lipschitz error certificate over the fixed orbit-delay quotient.
    """

    if cover_size <= 0:
        raise ValueError("cover_size must be positive")
    if active_budget < 0:
        raise ValueError("active_budget must be nonnegative")
    background = simulate_quotient(network, initial_background, inputs)
    steps = background.shape[0] - 1
    drive = np.asarray(inputs, dtype=np.float64)
    retained_layers: list[dict[Node, float]] = [dict() for _ in range(steps + 1)]
    dropped_layers: list[dict[Node, float]] = [dict() for _ in range(steps + 1)]
    initial_candidates: dict[Node, float] = {}
    for (cell, orbit), delta in perturbations.items():
        if not 0 <= orbit < network.orbit_count or not math.isfinite(delta):
            raise ValueError("invalid perturbation")
        key = (int(cell) % cover_size, int(orbit))
        initial_candidates[key] = initial_candidates.get(key, 0.0) + float(delta)
    retained_layers[0], dropped_layers[0], initial_ties = _equivariant_magnitude_budget(
        initial_candidates, active_budget
    )

    reconstructed = lift_orbit_trajectory(background, cover_size)
    for node, delta in retained_layers[0].items():
        reconstructed[0, node[0], node[1]] += delta
    residual = np.zeros((steps + 1, network.orbit_count), dtype=np.float64)
    for (_, orbit), delta in dropped_layers[0].items():
        residual[0, orbit] = max(residual[0, orbit], abs(delta))
    certified = np.zeros_like(residual)
    certified[0] = residual[0]
    candidate_counts = [len(initial_candidates)]
    exhausted = [bool(dropped_layers[0])]
    tie_drops = [initial_ties]
    candidate_evaluations = len(initial_candidates)
    edge_evaluations = 0

    for time in range(1, steps + 1):
        candidate_nodes: set[Node] = set()
        for edge in network.edges:
            source_time = time - edge.delay
            if source_time < 0:
                continue
            for cell, orbit in retained_layers[source_time]:
                if orbit == edge.source_orbit:
                    candidate_nodes.add(((cell + edge.shift) % cover_size, edge.target_orbit))
        candidates: dict[Node, float] = {}
        for cell, target_orbit in candidate_nodes:
            total = network.bias[target_orbit] + drive[time - 1, target_orbit]
            for edge in network.edges:
                if edge.target_orbit != target_orbit:
                    continue
                edge_evaluations += 1
                source_time = time - edge.delay
                if source_time < 0:
                    source_value = 0.0
                else:
                    source_cell = (cell - edge.shift) % cover_size
                    source_value = background[source_time, edge.source_orbit]
                    source_value += retained_layers[source_time].get(
                        (source_cell, edge.source_orbit), 0.0
                    )
                total += edge.weight * source_value
            delta = math.tanh(total) - background[time, target_orbit]
            if delta != 0.0:
                candidates[(cell, target_orbit)] = delta
        retained, dropped, ties = _equivariant_magnitude_budget(candidates, active_budget)
        retained_layers[time] = retained
        dropped_layers[time] = dropped
        for node, delta in retained.items():
            reconstructed[time, node[0], node[1]] += delta
        for (_, orbit), delta in dropped.items():
            residual[time, orbit] = max(residual[time, orbit], abs(delta))
        certified[time] = residual[time]
        for edge in network.edges:
            source_time = time - edge.delay
            if source_time >= 0:
                certified[time, edge.target_orbit] += (
                    abs(edge.weight) * certified[source_time, edge.source_orbit]
                )
        candidate_counts.append(len(candidates))
        exhausted.append(bool(dropped))
        tie_drops.append(ties)
        candidate_evaluations += len(candidates)

    return ApproximateConeResult(
        baseline=background,
        reconstructed=reconstructed,
        retained_by_time=tuple(tuple(sorted(layer)) for layer in retained_layers),
        dropped_by_time=tuple(tuple(sorted(layer)) for layer in dropped_layers),
        candidate_counts=tuple(candidate_counts),
        local_residual_by_time_orbit=residual,
        certified_error_by_time_orbit=certified,
        exhausted_by_time=tuple(exhausted),
        tie_drops_by_time=tuple(tie_drops),
        candidate_evaluations=candidate_evaluations,
        edge_evaluations=edge_evaluations,
    )


def translate_cells(state: ArrayLike, shift: int) -> FloatArray:
    values = np.asarray(state, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("state must contain cell and orbit axes")
    return np.roll(values, int(shift), axis=-2)


__all__ = [
    "ApproximateConeResult",
    "DelayedEdge",
    "DelayedOrbitNetwork",
    "DelayedSnapshot",
    "SparseConeResult",
    "SparseSidecarResult",
    "advance_full",
    "advance_quotient",
    "initial_snapshot",
    "lift_orbit_trajectory",
    "project_orbit_state",
    "simulate_full",
    "simulate_budgeted_initial_deviation",
    "simulate_quotient",
    "simulate_sparse_initial_deviation",
    "simulate_sparse_sidecar",
    "translate_cells",
]
