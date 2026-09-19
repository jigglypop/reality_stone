"""Bounded graph-field memory primitive for the Clarus-field research track.

The module implements only the portion covered by CF-1--CF-3 in
``_workspace/ce/agi-clarus-field-20260812``:

* a finite, undirected graph carries a non-negative damped diffusion field;
* an externally supplied salience score passes through an exact hard gate;
* an externally supplied write is projected into the unit ball and then used
  in a convex memory update;
* phase occupancy is descriptive and never forced toward a cosmological
  target.

This is a research primitive, not a biological brain model, a cosmology
solver, or evidence of AGI.  In particular, the conditional CF-3 equilibrium
theorem additionally requires an i.i.d. exogenous/common-write input process
and non-atomic phase thresholds; Python cannot infer those statistical facts
from one call to :meth:`ClarusField.step`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal, Sequence

import numpy as np


PhaseLabel = Literal["active", "structural", "frozen"]


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive built-in integer")
    return value


def _array(
    values: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite array with shape {shape}") from error
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


def _tuples(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _vector_tuple(vector: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in vector)


@dataclass(frozen=True)
class ClarusFieldConfig:
    """Dimensionless coefficients for one graph-field tick.

    ``structure_threshold`` is applied to ``field_decay * phi / source_cap``.
    It is therefore dimensionless even when a later application supplies a
    dimensional interpretation for ``phi``.
    """

    width: int
    field_decay: float = 0.25
    diffusion_strength: float = 1.0
    tick_duration: float = 1.0
    source_cap: float = 1.0
    gate_threshold: float = 0.5
    structure_threshold: float = 0.25

    def __post_init__(self) -> None:
        _positive_int(self.width, "width")
        for name in (
            "field_decay",
            "diffusion_strength",
            "tick_duration",
            "source_cap",
            "gate_threshold",
            "structure_threshold",
        ):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        if self.field_decay <= 0.0:
            raise ValueError("field_decay must be positive")
        if self.diffusion_strength < 0.0:
            raise ValueError("diffusion_strength must be nonnegative")
        if self.tick_duration <= 0.0:
            raise ValueError("tick_duration must be positive")
        if self.source_cap <= 0.0:
            raise ValueError("source_cap must be positive")
        if not 0.0 <= self.gate_threshold < 1.0:
            raise ValueError("gate_threshold must lie in [0, 1)")
        if self.structure_threshold < 0.0:
            raise ValueError("structure_threshold must be nonnegative")


@dataclass(frozen=True)
class ClarusFieldState:
    tick: int
    memory: tuple[tuple[float, ...], ...]
    field: tuple[float, ...]


@dataclass(frozen=True)
class ClarusFieldDrive:
    """One exogenous gate/write event.

    CF-3 covers a stream of these drives only when gate scores and candidates
    are common functions of an exogenous i.i.d. input, not functions of the
    current :class:`ClarusFieldState`.
    """

    gate_scores: tuple[float, ...]
    write_candidates: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class PhaseOccupancy:
    active: float
    structural: float
    frozen: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.active, self.structural, self.frozen)


@dataclass(frozen=True)
class ClarusFieldCertificate:
    node_count: int
    laplacian_eigenvalues: tuple[float, ...]
    field_contraction: float
    stationary_field_bound: float
    cf1_bounded_positive_field: bool
    cf2_exact_closed_gate: bool
    cf3_scope: str
    p_star_self_convergence: bool
    v14_route_l_inherited: bool


@dataclass(frozen=True)
class ClarusFieldStep:
    state: ClarusFieldState
    effective_gate: tuple[float, ...]
    phase_labels: tuple[PhaseLabel, ...]
    occupancy: PhaseOccupancy
    source: tuple[float, ...]
    memory_bound: float
    field_bound: float


def normalized_graph_laplacian(adjacency: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return the symmetric normalized Laplacian of a connected graph."""

    try:
        raw = np.asarray(adjacency, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("adjacency must be a finite square matrix") from error
    if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[0] != raw.shape[1]:
        raise ValueError("adjacency must be a nonempty square matrix")
    if not np.all(np.isfinite(raw)):
        raise ValueError("adjacency must contain only finite values")
    if np.any(raw < 0.0):
        raise ValueError("adjacency weights must be nonnegative")
    if not np.allclose(raw, raw.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("field adjacency must be symmetric")
    if not np.allclose(np.diag(raw), 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("field adjacency must have a zero diagonal")

    graph = raw > 0.0
    visited = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in np.flatnonzero(graph[node]):
            index = int(neighbor)
            if index not in visited:
                visited.add(index)
                frontier.append(index)
    if len(visited) != raw.shape[0]:
        raise ValueError("field adjacency must describe a connected graph")

    if raw.shape[0] == 1:
        return np.zeros((1, 1), dtype=np.float64)
    degree = raw.sum(axis=1)
    if np.any(degree <= 0.0):  # pragma: no cover - excluded by connectivity
        raise ValueError("connected adjacency must have positive degrees")
    inverse_sqrt = degree ** -0.5
    normalized = raw * inverse_sqrt[:, None] * inverse_sqrt[None, :]
    laplacian = np.eye(raw.shape[0], dtype=np.float64) - normalized
    return 0.5 * (laplacian + laplacian.T)


def project_rows_to_unit_ball(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Project the last-axis row vectors onto the closed Euclidean unit ball."""

    try:
        raw = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("values must be a finite rank-1 or rank-2 array") from error
    if raw.ndim not in {1, 2} or raw.shape[-1] < 1:
        raise ValueError("values must be a nonempty rank-1 or rank-2 array")
    if not np.all(np.isfinite(raw)):
        raise ValueError("values must contain only finite values")
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    return raw / np.maximum(norms, 1.0)


def prediction_error_gate_scores(
    observation: Sequence[Sequence[float]] | np.ndarray,
    prediction: Sequence[Sequence[float]] | np.ndarray,
    *,
    reference_scale: float,
    gain: float = 8.0,
    bias: float = -4.0,
) -> np.ndarray:
    """Return sign-invariant salience scores from normalized prediction error.

    The sigmoid argument is ``gain * ||(observation-prediction)/scale||^2 + bias``.
    A positive reference scale is mandatory so the exponential core receives a
    dimensionless argument.  The returned soft scores still pass through the
    exact hard threshold inside :meth:`ClarusField.step`.
    """

    try:
        observed = np.asarray(observation, dtype=np.float64)
        predicted = np.asarray(prediction, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("observation and prediction must be finite rank-2 arrays") from error
    if observed.ndim != 2 or observed.shape != predicted.shape or observed.shape[1] < 1:
        raise ValueError("observation and prediction must be same-shape rank-2 arrays")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
        raise ValueError("observation and prediction must contain only finite values")
    scale = _finite_float(reference_scale, "reference_scale")
    gain_value = _finite_float(gain, "gain")
    bias_value = _finite_float(bias, "bias")
    if scale <= 0.0:
        raise ValueError("reference_scale must be positive")
    if gain_value < 0.0:
        raise ValueError("gain must be nonnegative")
    normalized_error = (observed - predicted) / scale
    logits = gain_value * np.sum(normalized_error * normalized_error, axis=1) + bias_value
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))


def bounded_hrr_bind(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Circular-convolution binding projected into the unit ball.

    This is a bounded readout helper.  It is deliberately not used by the
    certified state transition, so the unstable recurrent V14 route-L update
    cannot be inherited accidentally.
    """

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.ndim != 1 or left_array.shape != right_array.shape or left_array.size < 1:
        raise ValueError("left and right must be same-width nonempty vectors")
    if not np.all(np.isfinite(left_array)) or not np.all(np.isfinite(right_array)):
        raise ValueError("left and right must contain only finite values")
    width = left_array.size
    bound = np.zeros(width, dtype=np.float64)
    for index in range(width):
        bound[index] = sum(
            left_array[offset] * right_array[(index - offset) % width]
            for offset in range(width)
        )
    return project_rows_to_unit_ball(bound)


class ClarusField:
    """Stateless transition operator for bounded graph memory and field state."""

    def __init__(
        self,
        adjacency: Sequence[Sequence[float]] | np.ndarray,
        config: ClarusFieldConfig,
    ) -> None:
        if type(config) is not ClarusFieldConfig:
            raise ValueError("config must be an exact ClarusFieldConfig")
        self.config = config
        self._laplacian = normalized_graph_laplacian(adjacency)
        self.node_count = int(self._laplacian.shape[0])

        operator = (
            self.config.diffusion_strength * self._laplacian
            + self.config.field_decay * np.eye(self.node_count, dtype=np.float64)
        )
        eigenvalues, eigenvectors = np.linalg.eigh(operator)
        if float(np.min(eigenvalues)) <= 0.0:  # pragma: no cover - guarded by decay
            raise ValueError("damped field operator must be positive definite")
        decay = np.exp(-eigenvalues * self.config.tick_duration)
        source_gain = (1.0 - decay) / eigenvalues
        self._propagator = (eigenvectors * decay) @ eigenvectors.T
        self._source_integral = (eigenvectors * source_gain) @ eigenvectors.T
        laplacian_eigenvalues = np.linalg.eigvalsh(self._laplacian)
        self.certificate = ClarusFieldCertificate(
            node_count=self.node_count,
            laplacian_eigenvalues=tuple(float(value) for value in laplacian_eigenvalues),
            field_contraction=float(np.max(decay)),
            stationary_field_bound=(
                math.sqrt(self.node_count)
                * self.config.source_cap
                / self.config.field_decay
            ),
            cf1_bounded_positive_field=True,
            cf2_exact_closed_gate=True,
            cf3_scope="conditional:iid-exogenous-common-write+nonatomic-thresholds",
            p_star_self_convergence=False,
            v14_route_l_inherited=False,
        )

    @property
    def laplacian(self) -> np.ndarray:
        return self._laplacian.copy()

    def zero_state(self) -> ClarusFieldState:
        return ClarusFieldState(
            tick=0,
            memory=tuple((0.0,) * self.config.width for _ in range(self.node_count)),
            field=(0.0,) * self.node_count,
        )

    def make_state(
        self,
        memory: Sequence[Sequence[float]] | np.ndarray,
        field: Sequence[float] | np.ndarray,
        *,
        tick: int = 0,
    ) -> ClarusFieldState:
        if type(tick) is not int or tick < 0:
            raise ValueError("tick must be a nonnegative built-in integer")
        memory_array = _array(
            memory,
            shape=(self.node_count, self.config.width),
            name="memory",
        )
        field_array = _array(field, shape=(self.node_count,), name="field")
        tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(abs(field_array))))
        if np.any(field_array < -tolerance):
            raise ValueError("field must be componentwise nonnegative")
        field_array = np.maximum(field_array, 0.0)
        return ClarusFieldState(tick=tick, memory=_tuples(memory_array), field=_vector_tuple(field_array))

    def make_drive(
        self,
        gate_scores: Sequence[float] | np.ndarray,
        write_candidates: Sequence[Sequence[float]] | np.ndarray,
    ) -> ClarusFieldDrive:
        gates = _array(gate_scores, shape=(self.node_count,), name="gate_scores")
        if np.any(gates < 0.0) or np.any(gates > 1.0):
            raise ValueError("gate_scores must lie in [0, 1]")
        candidates = _array(
            write_candidates,
            shape=(self.node_count, self.config.width),
            name="write_candidates",
        )
        return ClarusFieldDrive(
            gate_scores=_vector_tuple(gates),
            write_candidates=_tuples(candidates),
        )

    def snapshot(self, state: ClarusFieldState) -> ClarusFieldState:
        memory, field = self._validated_state(state)
        return ClarusFieldState(state.tick, _tuples(memory), _vector_tuple(field))

    def from_snapshot(self, snapshot: ClarusFieldState) -> ClarusFieldState:
        return self.snapshot(snapshot)

    def _validated_state(self, state: ClarusFieldState) -> tuple[np.ndarray, np.ndarray]:
        if type(state) is not ClarusFieldState:
            raise ValueError("state must be an exact ClarusFieldState")
        if type(state.tick) is not int or state.tick < 0:
            raise ValueError("state.tick must be a nonnegative built-in integer")
        memory = _array(
            state.memory,
            shape=(self.node_count, self.config.width),
            name="state.memory",
        )
        field = _array(state.field, shape=(self.node_count,), name="state.field")
        tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(abs(field))))
        if np.any(field < -tolerance):
            raise ValueError("state.field must be componentwise nonnegative")
        return memory, np.maximum(field, 0.0)

    def _validated_drive(self, drive: ClarusFieldDrive) -> tuple[np.ndarray, np.ndarray]:
        if type(drive) is not ClarusFieldDrive:
            raise ValueError("drive must be an exact ClarusFieldDrive")
        gates = _array(drive.gate_scores, shape=(self.node_count,), name="drive.gate_scores")
        if np.any(gates < 0.0) or np.any(gates > 1.0):
            raise ValueError("drive.gate_scores must lie in [0, 1]")
        candidates = _array(
            drive.write_candidates,
            shape=(self.node_count, self.config.width),
            name="drive.write_candidates",
        )
        return gates, project_rows_to_unit_ball(candidates)

    def step(self, state: ClarusFieldState, drive: ClarusFieldDrive) -> ClarusFieldStep:
        memory, field = self._validated_state(state)
        gates, candidates = self._validated_drive(drive)

        source = np.minimum(np.linalg.norm(memory, axis=1), self.config.source_cap)
        next_field = self._propagator @ field + self._source_integral @ source
        scale = max(1.0, float(np.linalg.norm(next_field)))
        tolerance = 256.0 * np.finfo(np.float64).eps * scale
        if float(np.min(next_field)) < -tolerance:
            raise FloatingPointError("field integrator violated positivity beyond roundoff")
        next_field = np.maximum(next_field, 0.0)

        effective_gate = np.where(gates > self.config.gate_threshold, gates, 0.0)
        next_memory = memory.copy()
        open_nodes = np.flatnonzero(effective_gate > 0.0)
        if open_nodes.size:
            weights = effective_gate[open_nodes, None]
            next_memory[open_nodes] = (
                (1.0 - weights) * memory[open_nodes] + weights * candidates[open_nodes]
            )

        dimensionless_field = (
            self.config.field_decay * next_field / self.config.source_cap
        )
        active = effective_gate > 0.0
        structural = (~active) & (dimensionless_field > self.config.structure_threshold)
        labels: tuple[PhaseLabel, ...] = tuple(
            "active" if active[index] else "structural" if structural[index] else "frozen"
            for index in range(self.node_count)
        )
        occupancy = PhaseOccupancy(
            active=float(np.mean(active)),
            structural=float(np.mean(structural)),
            frozen=float(np.mean(~active & ~structural)),
        )

        current_memory_bound = max(float(np.max(np.linalg.norm(memory, axis=1))), 1.0)
        field_bound = max(float(np.linalg.norm(field)), self.certificate.stationary_field_bound)
        next_state = ClarusFieldState(
            tick=state.tick + 1,
            memory=_tuples(next_memory),
            field=_vector_tuple(next_field),
        )
        return ClarusFieldStep(
            state=next_state,
            effective_gate=_vector_tuple(effective_gate),
            phase_labels=labels,
            occupancy=occupancy,
            source=_vector_tuple(source),
            memory_bound=current_memory_bound,
            field_bound=field_bound,
        )


__all__ = [
    "PhaseLabel",
    "ClarusFieldConfig",
    "ClarusFieldState",
    "ClarusFieldDrive",
    "PhaseOccupancy",
    "ClarusFieldCertificate",
    "ClarusFieldStep",
    "ClarusField",
    "normalized_graph_laplacian",
    "project_rows_to_unit_ball",
    "prediction_error_gate_scores",
    "bounded_hrr_bind",
]
