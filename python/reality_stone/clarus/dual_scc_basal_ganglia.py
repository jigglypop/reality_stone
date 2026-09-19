"""Certified two-timescale recurrent controller for basal-ganglia research.

The implementation is an engineering model, not a claim of literal neural
identity.  ``slow`` and ``fast`` name two coloured recurrent edge layers.  If
reciprocal cross-layer edges are ignored, each layer is strongly connected;
their uncoloured union is normally one larger SCC.

All values entering ``tanh``, ``exp`` and probabilities are dimensionless.
Callers must normalize physical observations, rewards, rates and costs before
constructing the drives passed to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


class DualSCCConvergenceError(RuntimeError):
    """Raised when a certified fixed-point solve exhausts its iteration budget."""


@dataclass(frozen=True)
class LayerEdge:
    source: int
    target: int
    layer: str


@dataclass(frozen=True)
class TopologyAudit:
    slow_component_count: int
    fast_component_count: int
    union_component_count: int
    slow_is_strongly_connected: bool
    fast_is_strongly_connected: bool
    union_is_single_macro_scc: bool


@dataclass(frozen=True)
class SmallGainCertificate:
    gain_matrix: tuple[tuple[float, float], tuple[float, float]]
    spectral_radius: float
    determinant_margin: float
    block_weights: tuple[float, float]
    weighted_contraction: float
    certified: bool


@dataclass(frozen=True)
class DualSCCConfig:
    """Dimensionless coefficients for the two recurrent layers."""

    action_count: int = 2
    slow_recurrence: float = 0.34
    fast_recurrence: float = 0.30
    slow_from_fast: float = 0.08
    fast_from_slow: float = 1.00
    tolerance: float = 1e-10
    max_iterations: int = 128
    policy_temperature: float = 0.55
    hold_temperature: float = 0.20
    hold_bias: float = -1.10
    hold_conflict_gain: float = 1.00

    def __post_init__(self) -> None:
        if self.action_count < 2:
            raise ValueError("action_count must be at least two")
        for name, value in (
            ("slow_recurrence", self.slow_recurrence),
            ("fast_recurrence", self.fast_recurrence),
            ("slow_from_fast", self.slow_from_fast),
            ("fast_from_slow", self.fast_from_slow),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (
            ("tolerance", self.tolerance),
            ("policy_temperature", self.policy_temperature),
            ("hold_temperature", self.hold_temperature),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("hold_bias", self.hold_bias),
            ("hold_conflict_gain", self.hold_conflict_gain),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.hold_conflict_gain < 0.0:
            raise ValueError("hold_conflict_gain must be nonnegative")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")


@dataclass(frozen=True)
class DualSCCPolicy:
    action_probabilities: tuple[float, ...]
    conditional_action_probabilities: tuple[float, ...]
    hold_probability: float
    conflict: float
    normalization_error: float
    selected_action: int | None


@dataclass(frozen=True)
class DualSCCResult:
    slow_state: tuple[float, ...]
    fast_state: tuple[float, ...]
    iterations: int
    residual: float
    residual_by_layer: tuple[float, float]
    error_bound: float
    error_bound_by_layer: tuple[float, float]
    certificate: SmallGainCertificate
    policy: DualSCCPolicy


def _cycle_matrix(size: int) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.float64)
    for target in range(size):
        matrix[target, (target - 1) % size] = 1.0
    return matrix


def _default_cross_matrices(action_count: int) -> tuple[np.ndarray, np.ndarray]:
    slow_from_fast = np.zeros((2, action_count + 1), dtype=np.float64)
    slow_from_fast[0, 0] = 1.0
    slow_from_fast[1, 1] = 1.0

    fast_from_slow = np.zeros((action_count + 1, 2), dtype=np.float64)
    for action in range(action_count):
        fast_from_slow[action, action % 2] = 1.0
    fast_from_slow[-1, :] = 0.5
    return slow_from_fast, fast_from_slow


def _induced_infinity_norm(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    return float(np.max(np.sum(np.abs(matrix), axis=1)))


def _stable_softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values / temperature
    scaled -= float(np.max(scaled))
    weights = np.exp(scaled)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        raise FloatingPointError("softmax normalization is nonfinite")
    return weights / total


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _components(node_count: int, edges: Iterable[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for source, target in edges:
        if source < 0 or source >= node_count or target < 0 or target >= node_count:
            raise ValueError("edge references an unknown node")
        adjacency[source].append(target)

    index = 0
    indices = [-1] * node_count
    lowlink = [0] * node_count
    stack: list[int] = []
    on_stack = [False] * node_count
    found: list[tuple[int, ...]] = []

    def visit(node: int) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack[node] = True
        for target in adjacency[node]:
            if indices[target] < 0:
                visit(target)
                lowlink[node] = min(lowlink[node], lowlink[target])
            elif on_stack[target]:
                lowlink[node] = min(lowlink[node], indices[target])
        if lowlink[node] != indices[node]:
            return
        component: list[int] = []
        while True:
            member = stack.pop()
            on_stack[member] = False
            component.append(member)
            if member == node:
                break
        found.append(tuple(sorted(component)))

    for node in range(node_count):
        if indices[node] < 0:
            visit(node)
    return tuple(sorted(found))


class DualSCCBasalGanglia:
    """Two coupled recurrent layers with a computable small-gain certificate."""

    def __init__(
        self,
        config: DualSCCConfig = DualSCCConfig(),
        *,
        slow_matrix: np.ndarray | None = None,
        fast_matrix: np.ndarray | None = None,
        slow_from_fast_matrix: np.ndarray | None = None,
        fast_from_slow_matrix: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.slow_size = 2
        self.fast_size = config.action_count + 1
        default_slow_from_fast, default_fast_from_slow = _default_cross_matrices(
            config.action_count
        )
        self.slow_matrix = self._validated_matrix(
            _cycle_matrix(self.slow_size) if slow_matrix is None else slow_matrix,
            (self.slow_size, self.slow_size),
            "slow_matrix",
        )
        self.fast_matrix = self._validated_matrix(
            _cycle_matrix(self.fast_size) if fast_matrix is None else fast_matrix,
            (self.fast_size, self.fast_size),
            "fast_matrix",
        )
        self.slow_from_fast_matrix = self._validated_matrix(
            default_slow_from_fast
            if slow_from_fast_matrix is None
            else slow_from_fast_matrix,
            (self.slow_size, self.fast_size),
            "slow_from_fast_matrix",
        )
        self.fast_from_slow_matrix = self._validated_matrix(
            default_fast_from_slow
            if fast_from_slow_matrix is None
            else fast_from_slow_matrix,
            (self.fast_size, self.slow_size),
            "fast_from_slow_matrix",
        )
        self.certificate = self._small_gain_certificate()
        self._topology = self._compute_topology_audit()

    @staticmethod
    def _validated_matrix(matrix: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
        result = np.asarray(matrix, dtype=np.float64)
        if result.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be finite")
        return result.copy()

    def _small_gain_certificate(self) -> SmallGainCertificate:
        gain = np.asarray(
            (
                (
                    self.config.slow_recurrence * _induced_infinity_norm(self.slow_matrix),
                    self.config.slow_from_fast
                    * _induced_infinity_norm(self.slow_from_fast_matrix),
                ),
                (
                    self.config.fast_from_slow
                    * _induced_infinity_norm(self.fast_from_slow_matrix),
                    self.config.fast_recurrence * _induced_infinity_norm(self.fast_matrix),
                ),
            ),
            dtype=np.float64,
        )
        if np.any(gain < 0.0) or not np.all(np.isfinite(gain)):
            raise ValueError("block gain matrix must be finite and nonnegative")
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(gain))))
        if spectral_radius < 1.0:
            weights = np.linalg.solve(np.eye(2, dtype=np.float64) - gain, np.ones(2))
            ratios = gain @ weights / weights
            contraction = float(np.max(ratios))
            certified = bool(
                np.all(weights > 0.0)
                and math.isfinite(contraction)
                and contraction < 1.0
            )
        else:
            weights = np.ones(2, dtype=np.float64)
            contraction = spectral_radius
            certified = False
        return SmallGainCertificate(
            gain_matrix=tuple(tuple(float(value) for value in row) for row in gain),
            spectral_radius=spectral_radius,
            determinant_margin=float(np.linalg.det(np.eye(2) - gain)),
            block_weights=(float(weights[0]), float(weights[1])),
            weighted_contraction=contraction,
            certified=certified,
        )

    def _compute_topology_audit(self) -> TopologyAudit:
        slow_edges = tuple(
            (source, target)
            for target in range(self.slow_size)
            for source in range(self.slow_size)
            if self.slow_matrix[target, source] != 0.0
        )
        fast_edges = tuple(
            (source, target)
            for target in range(self.fast_size)
            for source in range(self.fast_size)
            if self.fast_matrix[target, source] != 0.0
        )
        slow_components = _components(self.slow_size, slow_edges)
        fast_components = _components(self.fast_size, fast_edges)

        offset = self.slow_size
        union_edges = list(slow_edges)
        union_edges.extend((offset + source, offset + target) for source, target in fast_edges)
        if self.config.slow_from_fast > 0.0:
            union_edges.extend(
                (offset + source, target)
                for target in range(self.slow_size)
                for source in range(self.fast_size)
                if self.slow_from_fast_matrix[target, source] != 0.0
            )
        if self.config.fast_from_slow > 0.0:
            union_edges.extend(
                (source, offset + target)
                for target in range(self.fast_size)
                for source in range(self.slow_size)
                if self.fast_from_slow_matrix[target, source] != 0.0
            )
        union_components = _components(self.slow_size + self.fast_size, union_edges)
        return TopologyAudit(
            slow_component_count=len(slow_components),
            fast_component_count=len(fast_components),
            union_component_count=len(union_components),
            slow_is_strongly_connected=len(slow_components) == 1,
            fast_is_strongly_connected=len(fast_components) == 1,
            union_is_single_macro_scc=len(union_components) == 1,
        )

    def topology_audit(self) -> TopologyAudit:
        return self._topology

    def update(
        self,
        slow_state: Sequence[float],
        fast_state: Sequence[float],
        slow_drive: Sequence[float],
        fast_drive: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        slow = self._vector(slow_state, self.slow_size, "slow_state", bounded=True)
        fast = self._vector(fast_state, self.fast_size, "fast_state", bounded=True)
        slow_input = self._vector(slow_drive, self.slow_size, "slow_drive")
        fast_input = self._vector(fast_drive, self.fast_size, "fast_drive")
        next_slow = np.tanh(
            slow_input
            + self.config.slow_recurrence * (self.slow_matrix @ slow)
            + self.config.slow_from_fast * (self.slow_from_fast_matrix @ fast)
        )
        next_fast = np.tanh(
            fast_input
            + self.config.fast_recurrence * (self.fast_matrix @ fast)
            + self.config.fast_from_slow * (self.fast_from_slow_matrix @ slow)
        )
        return next_slow, next_fast

    @staticmethod
    def _vector(
        values: Sequence[float],
        size: int,
        name: str,
        *,
        bounded: bool = False,
    ) -> np.ndarray:
        result = np.asarray(tuple(values), dtype=np.float64)
        if result.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},)")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be finite")
        if bounded and np.any(np.abs(result) > 1.0 + 1e-12):
            raise ValueError(f"{name} must lie in [-1, 1]")
        return result

    def _block_norm(self, slow: np.ndarray, fast: np.ndarray) -> float:
        weights = self.certificate.block_weights
        return max(
            float(np.max(np.abs(slow))) / weights[0],
            float(np.max(np.abs(fast))) / weights[1],
        )

    def settle(
        self,
        slow_drive: Sequence[float],
        fast_drive: Sequence[float],
        *,
        initial_slow: Sequence[float] | None = None,
        initial_fast: Sequence[float] | None = None,
        hold_bias_delta: float = 0.0,
    ) -> DualSCCResult:
        """Solve the frozen-input coupled map and fail closed without a certificate."""

        if not self.certificate.certified:
            raise ValueError(
                "small-gain certificate failed: spectral radius must be below one"
            )
        topology = self.topology_audit()
        if not topology.slow_is_strongly_connected or not topology.fast_is_strongly_connected:
            raise ValueError("each coloured recurrent layer must be strongly connected")
        if not math.isfinite(hold_bias_delta):
            raise ValueError("hold_bias_delta must be finite and dimensionless")
        slow_input = self._vector(slow_drive, self.slow_size, "slow_drive")
        fast_input = self._vector(fast_drive, self.fast_size, "fast_drive")
        slow = (
            np.zeros(self.slow_size, dtype=np.float64)
            if initial_slow is None
            else self._vector(initial_slow, self.slow_size, "initial_slow", bounded=True)
        )
        fast = (
            np.zeros(self.fast_size, dtype=np.float64)
            if initial_fast is None
            else self._vector(initial_fast, self.fast_size, "initial_fast", bounded=True)
        )
        contraction = self.certificate.weighted_contraction
        residual = math.inf
        residual_by_layer = (math.inf, math.inf)
        error_bound = math.inf
        error_bound_by_layer = (math.inf, math.inf)
        for iteration in range(1, self.config.max_iterations + 1):
            slow, fast = self.update(slow, fast, slow_input, fast_input)
            check_slow, check_fast = self.update(slow, fast, slow_input, fast_input)
            slow_residual = float(np.max(np.abs(check_slow - slow)))
            fast_residual = float(np.max(np.abs(check_fast - fast)))
            residual_by_layer = (slow_residual, fast_residual)
            residual = self._block_norm(check_slow - slow, check_fast - fast)
            error_bound = residual / (1.0 - contraction)
            gain = np.asarray(self.certificate.gain_matrix, dtype=np.float64)
            component_bound = np.linalg.solve(
                np.eye(2, dtype=np.float64) - gain,
                np.asarray(residual_by_layer, dtype=np.float64),
            )
            error_bound_by_layer = (
                float(component_bound[0]),
                float(component_bound[1]),
            )
            if error_bound <= self.config.tolerance:
                policy = self.policy(fast, hold_bias_delta=hold_bias_delta)
                return DualSCCResult(
                    slow_state=tuple(float(value) for value in slow),
                    fast_state=tuple(float(value) for value in fast),
                    iterations=iteration,
                    residual=residual,
                    residual_by_layer=residual_by_layer,
                    error_bound=error_bound,
                    error_bound_by_layer=error_bound_by_layer,
                    certificate=self.certificate,
                    policy=policy,
                )
        raise DualSCCConvergenceError(
            "dual-SCC solve exhausted its finite budget: "
            f"bound={error_bound:.6g}, tolerance={self.config.tolerance:.6g}"
        )

    def policy(
        self,
        fast_state: Sequence[float],
        *,
        hold_bias_delta: float = 0.0,
    ) -> DualSCCPolicy:
        fast = self._vector(fast_state, self.fast_size, "fast_state", bounded=True)
        if not math.isfinite(hold_bias_delta):
            raise ValueError("hold_bias_delta must be finite and dimensionless")
        conditional = _stable_softmax(
            fast[: self.config.action_count], self.config.policy_temperature
        )
        if self.config.action_count == 1:
            conflict = 0.0
        else:
            conflict = -float(
                np.sum(conditional * np.log(np.clip(conditional, 1e-15, 1.0)))
            ) / math.log(float(self.config.action_count))
        hold_logit = (
            self.config.hold_bias
            + hold_bias_delta
            + self.config.hold_conflict_gain * conflict
            + float(fast[-1])
        )
        hold_probability = _sigmoid(hold_logit / self.config.hold_temperature)
        action_probabilities = (1.0 - hold_probability) * conditional
        normalization_error = abs(
            float(np.sum(action_probabilities)) + hold_probability - 1.0
        )
        all_probabilities = np.concatenate(
            (action_probabilities, np.asarray((hold_probability,), dtype=np.float64))
        )
        winner = int(np.argmax(all_probabilities))
        selected_action = None if winner == self.config.action_count else winner
        return DualSCCPolicy(
            action_probabilities=tuple(float(value) for value in action_probabilities),
            conditional_action_probabilities=tuple(float(value) for value in conditional),
            hold_probability=hold_probability,
            conflict=conflict,
            normalization_error=normalization_error,
            selected_action=selected_action,
        )


__all__ = [
    "DualSCCBasalGanglia",
    "DualSCCConfig",
    "DualSCCConvergenceError",
    "DualSCCPolicy",
    "DualSCCResult",
    "LayerEdge",
    "SmallGainCertificate",
    "TopologyAudit",
]
