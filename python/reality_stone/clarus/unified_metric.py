"""Finite one-metric research core for the V15 unified-metric track.

The module implements a deliberately narrow object: a fixed finite graph whose
nodes carry one symmetric positive-definite metric tensor.  The metric is the
only persistent semantic state.  All other inputs are transient queries or
external source tensors.

The implemented readouts are finite local quadratic lengths, metric-graph
costs, shortest paths, a dimensionless surprise gate, metric deformation, and
tie-preserving minimum-cost targets.  They are not a continuum geodesic solver,
a curvature implementation, an irreversible world model, or evidence of AGI,
biology, or cosmology.

Affine tensor covariance and fixed-chart spectral stabilisation are kept
separate.  Eigenvalue clipping is useful numerically but is not covariant under
general affine changes of chart.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import heapq
import math
from numbers import Real
from typing import Literal, Sequence

import numpy as np

from .clarus_field import normalized_graph_laplacian


MetricNestedTuple = tuple[tuple[tuple[float, ...], ...], ...]

_FLOAT_EPSILON = float(np.finfo(np.float64).eps)
_FLOAT_MAX_LOG = math.log(float(np.finfo(np.float64).max))
_FLOAT_MIN_SUBNORMAL = float(np.nextafter(0.0, 1.0))
_FLOAT_MIN_LOG = math.log(_FLOAT_MIN_SUBNORMAL)


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


def _node_index(value: object, node_count: int, name: str) -> int:
    if type(value) is not int or not 0 <= value < node_count:
        raise ValueError(f"{name} must be a built-in integer in [0, {node_count})")
    return value


def _finite_array(
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


def _safe_symmetrize(array: np.ndarray, *, name: str) -> np.ndarray:
    """Average a matrix with its transpose without overflowing finite entries."""

    with np.errstate(over="ignore", invalid="ignore"):
        result = 0.5 * array + 0.5 * np.swapaxes(array, -1, -2)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(f"{name} symmetrization is not representable")
    return result


def _positive_product_parts(
    factors: Sequence[float],
    *,
    name: str,
) -> tuple[float, int]:
    """Return ``mantissa, exponent`` for a positive product without forming it."""

    mantissa = 1.0
    exponent = 0
    for factor in factors:
        if not math.isfinite(factor) or factor <= 0.0:
            raise FloatingPointError(f"{name} has a nonpositive or nonfinite factor")
        part_mantissa, part_exponent = math.frexp(factor)
        mantissa *= part_mantissa
        exponent += part_exponent
        mantissa, shift = math.frexp(mantissa)
        exponent += shift
    return mantissa, exponent


def _representable_positive_product(
    factors: Sequence[float],
    *,
    name: str,
) -> float:
    mantissa, exponent = _positive_product_parts(factors, name=name)
    try:
        result = math.ldexp(mantissa, exponent)
    except OverflowError as error:
        raise FloatingPointError(f"{name} is not representable") from error
    if not math.isfinite(result) or result <= 0.0:
        raise FloatingPointError(f"{name} is not representable")
    return result


def _representable_positive_sqrt_product(
    factors: Sequence[float],
    *,
    name: str,
) -> float:
    mantissa, exponent = _positive_product_parts(factors, name=name)
    if exponent % 2:
        mantissa *= 2.0
        exponent -= 1
    try:
        result = math.ldexp(math.sqrt(mantissa), exponent // 2)
    except OverflowError as error:
        raise FloatingPointError(f"{name} is not representable") from error
    if not math.isfinite(result) or result <= 0.0:
        raise FloatingPointError(f"{name} is not representable")
    return result


def _quadratic_factors(
    vector: np.ndarray,
    metric: np.ndarray,
    *,
    name: str,
) -> tuple[float, ...] | None:
    """Factor ``x.T @ g @ x`` into bounded pieces, or return ``None`` for zero."""

    vector_scale = float(np.max(np.abs(vector)))
    if vector_scale == 0.0:
        return None
    metric_scale = float(np.max(np.abs(metric)))
    if not math.isfinite(metric_scale) or metric_scale <= 0.0:
        raise FloatingPointError(f"{name} metric scale is invalid")

    normalized_vector = vector / vector_scale
    normalized_metric = metric / metric_scale
    try:
        factor = np.linalg.cholesky(normalized_metric)
    except np.linalg.LinAlgError as error:
        raise FloatingPointError(f"{name} metric factorization failed") from error
    weighted = factor.T @ normalized_vector
    unit_quadratic = math.fsum(float(value) * float(value) for value in weighted)
    if not math.isfinite(unit_quadratic) or unit_quadratic <= 0.0:
        raise FloatingPointError(f"{name} is not numerically positive")
    return (unit_quadratic, metric_scale, vector_scale, vector_scale)


def _product_log(factors: Sequence[float]) -> float:
    return math.fsum(math.log(factor) for factor in factors)


def _saturating_exp(log_value: float) -> float:
    if log_value > _FLOAT_MAX_LOG:
        return math.inf
    if log_value < _FLOAT_MIN_LOG:
        return 0.0
    return math.exp(log_value)


def _metric_tuples(metric: np.ndarray) -> MetricNestedTuple:
    return tuple(
        tuple(tuple(float(value) for value in row) for row in node_metric) for node_metric in metric
    )


def _metric_array(
    values: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    *,
    node_count: int,
    dimension: int,
    name: str,
    require_spd: bool,
) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a finite array with shape ({node_count}, {dimension}, {dimension})"
        ) from error
    expected = (node_count, dimension, dimension)
    if result.shape != expected:
        raise ValueError(f"{name} must have shape {expected}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    scale = float(np.max(np.abs(result)))
    tolerance = 128.0 * _FLOAT_EPSILON * scale
    if not np.allclose(result, np.swapaxes(result, 1, 2), rtol=0.0, atol=tolerance):
        raise ValueError(f"{name} must be symmetric at every node")
    try:
        result = _safe_symmetrize(result, name=name)
    except FloatingPointError as error:
        raise ValueError(f"{name} cannot be safely symmetrized") from error
    if require_spd:
        try:
            eigenvalues = np.linalg.eigvalsh(result)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"{name} eigenvalue validation failed") from error
        if not np.all(np.isfinite(eigenvalues)) or float(np.min(eigenvalues)) <= 0.0:
            raise ValueError(f"{name} must be positive definite at every node")
    return result.copy()


@dataclass(frozen=True)
class UnifiedMetricConfig:
    """Dimensionless fixed-chart stabilisation and common source-update rate."""

    min_eigenvalue: float = 0.25
    max_eigenvalue: float = 4.0
    source_rate: float = 0.1

    def __post_init__(self) -> None:
        for name in ("min_eigenvalue", "max_eigenvalue", "source_rate"):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        if self.min_eigenvalue <= 0.0:
            raise ValueError("min_eigenvalue must be positive")
        if self.max_eigenvalue < self.min_eigenvalue:
            raise ValueError("max_eigenvalue must be at least min_eigenvalue")
        if not 0.0 <= self.source_rate <= 1.0:
            raise ValueError("source_rate must lie in [0, 1]")


@dataclass(frozen=True)
class UnifiedMetricState:
    """The only persistent semantic state: one SPD tensor at each graph node."""

    metric: MetricNestedTuple


@dataclass(frozen=True)
class MetricPath:
    """A representative minimum-cost graph path and its tie status."""

    nodes: tuple[int, ...]
    cost: float
    unique: bool
    tie_policy: str


@dataclass(frozen=True)
class MetricGoalReadout:
    """Candidate costs with every numerical minimizer preserved."""

    costs: tuple[tuple[int, float], ...]
    minimizers: tuple[int, ...]
    unique: bool


@dataclass(frozen=True)
class MetricSurprise:
    squared_length: float
    normalized_squared_length: float
    hard_gate: int


@dataclass(frozen=True)
class UnifiedMetricCertificate:
    node_count: int
    dimension: int
    observed_min_eigenvalue: float
    observed_max_eigenvalue: float
    condition_number: float
    configured_condition_bound: float
    within_configured_bounds: bool
    persistent_state: Literal["metric_only"]
    persistent_state_field_count: int
    role_parameter_count: int
    geometry_scope: Literal["finite-point-local-quadratic+metric-graph"]
    world_scope: Literal["metric_cost_substrate"]
    affine_readout_covariant: bool
    projection_affine_covariant: bool
    full_geodesic_verified: bool
    connection_verified: bool
    curvature_verified: bool
    heat_kernel_verified: bool
    continuum_limit_verified: bool
    irreversible_world_dynamics_verified: bool
    agi_evidence: bool
    biological_evidence: bool
    cosmological_evidence: bool


def affine_chart_change(
    points: Sequence[Sequence[float]] | np.ndarray,
    metric: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    jacobian: Sequence[Sequence[float]] | np.ndarray,
    offset: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Transport points and a covariant metric under ``y = J x + b``.

    No spectral projection is performed.  Applying fixed-chart eigenvalue
    clipping after this transform would generally destroy affine covariance.
    """

    try:
        point_array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("points must be a finite rank-2 array") from error
    if point_array.ndim != 2 or point_array.shape[0] < 1 or point_array.shape[1] < 2:
        raise ValueError("points must be a nonempty rank-2 array with dimension at least 2")
    if not np.all(np.isfinite(point_array)):
        raise ValueError("points must contain only finite values")
    node_count, dimension = point_array.shape
    metric_array = _metric_array(
        metric,
        node_count=node_count,
        dimension=dimension,
        name="metric",
        require_spd=True,
    )
    jacobian_array = _finite_array(
        jacobian,
        shape=(dimension, dimension),
        name="jacobian",
    )
    try:
        inverse = np.linalg.inv(jacobian_array)
    except np.linalg.LinAlgError as error:
        raise ValueError("jacobian must be invertible") from error
    if not np.all(np.isfinite(inverse)):
        raise ValueError("jacobian inverse must be finite")
    if offset is None:
        offset_array = np.zeros(dimension, dtype=np.float64)
    else:
        offset_array = _finite_array(offset, shape=(dimension,), name="offset")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        transformed_points = point_array @ jacobian_array.T + offset_array
    if not np.all(np.isfinite(transformed_points)):
        raise FloatingPointError("affine point transport is not representable")
    transformed_metric = np.empty_like(metric_array)
    for node in range(node_count):
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            transported = inverse.T @ metric_array[node] @ inverse
        if not np.all(np.isfinite(transported)):
            raise FloatingPointError("affine metric transport is not representable")
        transported = _safe_symmetrize(
            transported,
            name="affine metric transport",
        )
        try:
            eigenvalues = np.linalg.eigvalsh(transported)
        except np.linalg.LinAlgError as error:
            raise FloatingPointError("affine metric transport SPD validation failed") from error
        if not np.all(np.isfinite(eigenvalues)) or float(np.min(eigenvalues)) <= 0.0:
            raise FloatingPointError("affine metric transport is not representable as SPD")
        transformed_metric[node] = transported
    return transformed_points, transformed_metric


class UnifiedMetricCore:
    """Finite graph readouts driven by one shared SPD metric state."""

    def __init__(
        self,
        points: Sequence[Sequence[float]] | np.ndarray,
        adjacency: Sequence[Sequence[float]] | np.ndarray,
        config: UnifiedMetricConfig = UnifiedMetricConfig(),
    ) -> None:
        if type(config) is not UnifiedMetricConfig:
            raise ValueError("config must be an exact UnifiedMetricConfig")
        try:
            point_array = np.asarray(points, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("points must be a finite rank-2 array") from error
        if point_array.ndim != 2 or point_array.shape[0] < 1 or point_array.shape[1] < 2:
            raise ValueError("points must be a nonempty rank-2 array with dimension at least 2")
        if not np.all(np.isfinite(point_array)):
            raise ValueError("points must contain only finite values")
        if len({tuple(float(value) for value in row) for row in point_array}) != len(point_array):
            raise ValueError("points must be distinct")
        self.config = config
        self.node_count = int(point_array.shape[0])
        self.dimension = int(point_array.shape[1])
        try:
            adjacency_array = np.asarray(adjacency, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("adjacency must be a finite square matrix") from error
        normalized_graph_laplacian(adjacency_array)
        if adjacency_array.shape != (self.node_count, self.node_count):
            raise ValueError("adjacency size must match the number of points")
        self._points = point_array.copy()
        self._edge_mask = adjacency_array > 0.0

    @property
    def points(self) -> np.ndarray:
        return self._points.copy()

    @property
    def adjacency_mask(self) -> np.ndarray:
        return self._edge_mask.copy()

    def identity_state(self) -> UnifiedMetricState:
        metric = np.repeat(
            np.eye(self.dimension, dtype=np.float64)[None, :, :],
            self.node_count,
            axis=0,
        )
        metric = self._project_array(metric, name="identity metric")
        return UnifiedMetricState(_metric_tuples(metric))

    def make_state(
        self,
        metric: Sequence[Sequence[Sequence[float]]] | np.ndarray,
        *,
        project: bool = False,
    ) -> UnifiedMetricState:
        if type(project) is not bool:
            raise ValueError("project must be a built-in bool")
        if project:
            array = self._project_array(metric, name="metric")
        else:
            array = _metric_array(
                metric,
                node_count=self.node_count,
                dimension=self.dimension,
                name="metric",
                require_spd=True,
            )
        return UnifiedMetricState(_metric_tuples(array))

    def project_metric(
        self,
        metric: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    ) -> UnifiedMetricState:
        return UnifiedMetricState(_metric_tuples(self._project_array(metric, name="metric")))

    def _project_array(
        self,
        metric: Sequence[Sequence[Sequence[float]]] | np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        array = _metric_array(
            metric,
            node_count=self.node_count,
            dimension=self.dimension,
            name=name,
            require_spd=False,
        )
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(array)
        except np.linalg.LinAlgError as error:
            raise FloatingPointError(f"{name} projection eigensolver failed") from error
        if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
            raise FloatingPointError(f"{name} projection eigensystem is not finite")
        clipped = np.clip(
            eigenvalues,
            self.config.min_eigenvalue,
            self.config.max_eigenvalue,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            projected = (eigenvectors * clipped[:, None, :]) @ np.swapaxes(
                eigenvectors,
                1,
                2,
            )
        projected = _safe_symmetrize(projected, name=f"{name} projection")
        try:
            validated = _metric_array(
                projected,
                node_count=self.node_count,
                dimension=self.dimension,
                name=f"{name} projection",
                require_spd=True,
            )
        except ValueError as error:
            raise FloatingPointError(
                f"{name} projection did not produce a finite SPD metric"
            ) from error
        return validated

    def _validated_state(self, state: UnifiedMetricState) -> np.ndarray:
        if type(state) is not UnifiedMetricState:
            raise ValueError("state must be an exact UnifiedMetricState")
        return _metric_array(
            state.metric,
            node_count=self.node_count,
            dimension=self.dimension,
            name="state.metric",
            require_spd=True,
        )

    def snapshot(self, state: UnifiedMetricState) -> UnifiedMetricState:
        return UnifiedMetricState(_metric_tuples(self._validated_state(state)))

    def from_snapshot(self, snapshot: UnifiedMetricState) -> UnifiedMetricState:
        return self.snapshot(snapshot)

    def apply_source_metric(
        self,
        state: UnifiedMetricState,
        source_metric: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    ) -> UnifiedMetricState:
        """Apply one common bounded source update in the configured chart.

        Both the current tensor and external source tensor are spectrally
        projected before their convex combination.  This is a stable
        fixed-chart operation and is deliberately not claimed affine-covariant.
        """

        current = self._project_array(self._validated_state(state), name="state.metric")
        source = self._project_array(source_metric, name="source_metric")
        rate = self.config.source_rate
        with np.errstate(over="ignore", invalid="ignore"):
            updated = (1.0 - rate) * current + rate * source
        updated = _safe_symmetrize(updated, name="source metric update")
        try:
            validated = _metric_array(
                updated,
                node_count=self.node_count,
                dimension=self.dimension,
                name="source metric update",
                require_spd=True,
            )
        except ValueError as error:  # pragma: no cover - convex SPD arithmetic guard
            raise FloatingPointError("source metric update is not finite SPD") from error
        return UnifiedMetricState(_metric_tuples(validated))

    def metric_deformation(
        self,
        state: UnifiedMetricState,
        reference: UnifiedMetricState,
    ) -> MetricNestedTuple:
        """Return the covariant tensor difference used as the memory readout."""

        current = self._validated_state(state)
        baseline = self._validated_state(reference)
        return _metric_tuples(current - baseline)

    def local_length_squared(
        self,
        state: UnifiedMetricState,
        node: int,
        displacement: Sequence[float] | np.ndarray,
    ) -> float:
        metric = self._validated_state(state)
        node_index = _node_index(node, self.node_count, "node")
        vector = _finite_array(
            displacement,
            shape=(self.dimension,),
            name="displacement",
        )
        factors = _quadratic_factors(
            vector,
            metric[node_index],
            name="local squared length",
        )
        if factors is None:
            return 0.0
        return _representable_positive_product(
            factors,
            name="local squared length",
        )

    def edge_lengths(self, state: UnifiedMetricState) -> np.ndarray:
        """Return finite lengths on topology edges and infinity elsewhere."""

        metric = self._validated_state(state)
        lengths = np.full((self.node_count, self.node_count), np.inf, dtype=np.float64)
        np.fill_diagonal(lengths, 0.0)
        for source in range(self.node_count):
            for target in np.flatnonzero(self._edge_mask[source]):
                target_index = int(target)
                if target_index <= source:
                    continue
                with np.errstate(over="ignore", invalid="ignore"):
                    displacement = self._points[target_index] - self._points[source]
                if not np.all(np.isfinite(displacement)):
                    raise FloatingPointError("an edge displacement is not representable")
                endpoint_metric = _safe_symmetrize(
                    0.5 * metric[source] + 0.5 * metric[target_index],
                    name="endpoint metric average",
                )
                factors = _quadratic_factors(
                    displacement,
                    endpoint_metric,
                    name="edge squared length",
                )
                if factors is None:  # pragma: no cover - distinct points are validated
                    raise FloatingPointError("an edge has zero metric length")
                length = _representable_positive_sqrt_product(
                    factors,
                    name="edge length",
                )
                lengths[source, target_index] = length
                lengths[target_index, source] = length
        return lengths

    def _dijkstra(
        self,
        state: UnifiedMetricState,
        source: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source_index = _node_index(source, self.node_count, "source")
        lengths = self.edge_lengths(state)
        distances = np.full(self.node_count, np.inf, dtype=np.float64)
        predecessors = np.full(self.node_count, -1, dtype=np.int64)
        distances[source_index] = 0.0
        pending: list[tuple[float, int]] = [(0.0, source_index)]
        while pending:
            distance, node = heapq.heappop(pending)
            if distance != distances[node]:
                continue
            for neighbor in np.flatnonzero(np.isfinite(lengths[node])):
                neighbor_index = int(neighbor)
                if neighbor_index == node:
                    continue
                edge_length = float(lengths[node, neighbor_index])
                with np.errstate(over="ignore", invalid="ignore"):
                    candidate = distance + edge_length
                if not math.isfinite(candidate) or candidate <= distance:
                    continue
                incumbent = float(distances[neighbor_index])
                if candidate < incumbent:
                    distances[neighbor_index] = candidate
                    predecessors[neighbor_index] = node
                    heapq.heappush(pending, (candidate, neighbor_index))

        if not np.all(np.isfinite(distances)):
            raise FloatingPointError("a finite graph path cost is not representable")

        path_counts = np.zeros(self.node_count, dtype=np.int8)
        path_counts[source_index] = 1
        distance_order = sorted(
            range(self.node_count),
            key=lambda node: (float(distances[node]), node),
        )
        for node in distance_order:
            if path_counts[node] == 0:
                continue
            for neighbor in np.flatnonzero(np.isfinite(lengths[node])):
                neighbor_index = int(neighbor)
                if distances[node] >= distances[neighbor_index]:
                    continue
                candidate = float(distances[node]) + float(lengths[node, neighbor_index])
                if not math.isfinite(candidate):
                    continue
                target_distance = float(distances[neighbor_index])
                scale = max(abs(candidate), abs(target_distance))
                tie_tolerance = 128.0 * _FLOAT_EPSILON * scale
                if abs(candidate - target_distance) <= tie_tolerance:
                    path_counts[neighbor_index] = min(
                        2,
                        int(path_counts[neighbor_index]) + int(path_counts[node]),
                    )
        return distances, predecessors, path_counts

    def shortest_path(
        self,
        state: UnifiedMetricState,
        source: int,
        target: int,
    ) -> MetricPath:
        source_index = _node_index(source, self.node_count, "source")
        target_index = _node_index(target, self.node_count, "target")
        distances, predecessors, path_counts = self._dijkstra(state, source_index)
        path = [target_index]
        visited = {target_index}
        cursor = target_index
        for _ in range(self.node_count - 1):
            if cursor == source_index:
                break
            cursor = int(predecessors[cursor])
            if cursor < 0:  # pragma: no cover - topology connectivity is validated
                raise RuntimeError("connected metric graph produced no path")
            if cursor in visited:
                raise RuntimeError("shortest-path predecessor cycle detected")
            visited.add(cursor)
            path.append(cursor)
        if cursor != source_index:
            raise RuntimeError("shortest-path reconstruction exceeded N-1 hops")
        path.reverse()
        return MetricPath(
            nodes=tuple(path),
            cost=float(distances[target_index]),
            unique=bool(path_counts[target_index] == 1),
            tie_policy="lowest-index representative; uniqueness reported separately",
        )

    def surprise_gate(
        self,
        state: UnifiedMetricState,
        node: int,
        observed: Sequence[float] | np.ndarray,
        predicted: Sequence[float] | np.ndarray,
        *,
        reference_scale: float,
        threshold: float,
    ) -> MetricSurprise:
        observed_array = _finite_array(
            observed,
            shape=(self.dimension,),
            name="observed",
        )
        predicted_array = _finite_array(
            predicted,
            shape=(self.dimension,),
            name="predicted",
        )
        scale = _finite_float(reference_scale, "reference_scale")
        threshold_value = _finite_float(threshold, "threshold")
        if scale <= 0.0:
            raise ValueError("reference_scale must be positive")
        if threshold_value < 0.0:
            raise ValueError("threshold must be nonnegative")
        metric = self._validated_state(state)
        node_index = _node_index(node, self.node_count, "node")
        with np.errstate(over="ignore", invalid="ignore"):
            displacement = observed_array - predicted_array
        if not np.all(np.isfinite(displacement)):
            raise FloatingPointError("surprise displacement is not representable")
        factors = _quadratic_factors(
            displacement,
            metric[node_index],
            name="surprise squared length",
        )
        if factors is None:
            squared = 0.0
            normalized = 0.0
            hard_gate = 0
        else:
            squared_log = _product_log(factors)
            squared = _saturating_exp(squared_log)
            ratio_log = squared_log - 2.0 * math.log(scale)
            normalized = _saturating_exp(ratio_log)
            if threshold_value == 0.0:
                hard_gate = 1
            else:
                threshold_log = math.log(threshold_value)
                comparison_tolerance = (
                    128.0
                    * _FLOAT_EPSILON
                    * max(
                        1.0,
                        abs(ratio_log),
                        abs(threshold_log),
                    )
                )
                hard_gate = int(ratio_log > threshold_log + comparison_tolerance)
        return MetricSurprise(
            squared_length=squared,
            normalized_squared_length=normalized,
            hard_gate=hard_gate,
        )

    def minimum_cost_targets(
        self,
        state: UnifiedMetricState,
        source: int,
        candidates: Sequence[int],
    ) -> MetricGoalReadout:
        source_index = _node_index(source, self.node_count, "source")
        if isinstance(candidates, (str, bytes)):
            raise ValueError("candidates must be a nonempty sequence of node indices")
        ordered: list[int] = []
        for candidate in candidates:
            ordered.append(_node_index(candidate, self.node_count, "candidate"))
        if not ordered:
            raise ValueError("candidates must be nonempty")
        if len(set(ordered)) != len(ordered):
            raise ValueError("candidates must be unique")
        ordered.sort()
        distances, _, _ = self._dijkstra(state, source_index)
        costs = tuple((candidate, float(distances[candidate])) for candidate in ordered)
        minimum = min(cost for _, cost in costs)
        minimizers = tuple(
            candidate
            for candidate, cost in costs
            if abs(cost - minimum) <= 256.0 * _FLOAT_EPSILON * max(abs(cost), abs(minimum))
        )
        return MetricGoalReadout(
            costs=costs,
            minimizers=minimizers,
            unique=len(minimizers) == 1,
        )

    def certificate(self, state: UnifiedMetricState) -> UnifiedMetricCertificate:
        metric = self._validated_state(state)
        eigenvalues = np.linalg.eigvalsh(metric)
        minimum = float(np.min(eigenvalues))
        maximum = float(np.max(eigenvalues))
        tolerance = 256.0 * np.finfo(np.float64).eps * max(1.0, abs(maximum))
        within = (
            minimum >= self.config.min_eigenvalue - tolerance
            and maximum <= self.config.max_eigenvalue + tolerance
        )
        return UnifiedMetricCertificate(
            node_count=self.node_count,
            dimension=self.dimension,
            observed_min_eigenvalue=minimum,
            observed_max_eigenvalue=maximum,
            condition_number=maximum / minimum,
            configured_condition_bound=(self.config.max_eigenvalue / self.config.min_eigenvalue),
            within_configured_bounds=within,
            persistent_state="metric_only",
            persistent_state_field_count=len(fields(UnifiedMetricState)),
            role_parameter_count=0,
            geometry_scope="finite-point-local-quadratic+metric-graph",
            world_scope="metric_cost_substrate",
            affine_readout_covariant=True,
            projection_affine_covariant=False,
            full_geodesic_verified=False,
            connection_verified=False,
            curvature_verified=False,
            heat_kernel_verified=False,
            continuum_limit_verified=False,
            irreversible_world_dynamics_verified=False,
            agi_evidence=False,
            biological_evidence=False,
            cosmological_evidence=False,
        )


__all__ = [
    "UnifiedMetricConfig",
    "UnifiedMetricState",
    "MetricPath",
    "MetricGoalReadout",
    "MetricSurprise",
    "UnifiedMetricCertificate",
    "UnifiedMetricCore",
    "affine_chart_change",
]
