"""Finite SCC quotient and certificate utilities for isolated CE research.

The module keeps four objects separate:

* maximal strongly connected components of one fixed directed graph;
* merge-only SCC filtrations obtained by adding edges on that same node set;
* positive-delay event unrolls, whose SCCs are necessarily singletons; and
* dynamical certificates, which require gains in addition to graph topology.

It is not wired into the canonical Clarus runtime and does not identify a
biological whole-brain parcellation.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from types import MappingProxyType
from typing import Generic, Hashable, Iterable, Mapping, Sequence, TypeVar

import numpy as np
from numpy.typing import ArrayLike


NodeT = TypeVar("NodeT", bound=Hashable)


@dataclass(frozen=True)
class SCCDecomposition(Generic[NodeT]):
    """Deterministic maximal-SCC partition of a finite declared graph."""

    nodes: tuple[NodeT, ...]
    edges: tuple[tuple[NodeT, NodeT], ...]
    components: tuple[tuple[NodeT, ...], ...]
    component_of: Mapping[NodeT, int]
    condensation_edges: tuple[tuple[int, int], ...]
    topological_order: tuple[int, ...]
    component_is_recurrent: tuple[bool, ...]


@dataclass(frozen=True)
class ThresholdLevel(Generic[NodeT]):
    """One level of a decreasing-threshold, edge-addition filtration.

    ``parent_of_previous`` maps every component at the immediately preceding
    (higher) threshold to its unique containing component at this threshold.
    It is ``None`` at the first level.
    """

    threshold: float
    retained_edges: tuple[tuple[NodeT, NodeT], ...]
    decomposition: SCCDecomposition[NodeT]
    parent_of_previous: tuple[int, ...] | None


@dataclass(frozen=True)
class ThresholdFiltration(Generic[NodeT]):
    """A merge-only SCC tower for one fixed graph specification."""

    nodes: tuple[NodeT, ...]
    edge_semantics: str
    layer: str
    score_name: str
    tie_rule: str
    levels: tuple[ThresholdLevel[NodeT], ...]

    def assert_compatible(
        self,
        *,
        nodes: Sequence[NodeT],
        edge_semantics: str,
        layer: str,
        score_name: str,
    ) -> None:
        """Reject comparisons that are not a fixed-node, fixed-semantics filtration."""

        candidate_nodes, _ = _normalise_nodes(nodes)
        if candidate_nodes != self.nodes:
            raise ValueError("node set or declared node order changed across the filtration")
        if edge_semantics != self.edge_semantics:
            raise ValueError("edge semantics changed across the filtration")
        if layer != self.layer:
            raise ValueError("edge layer changed across the filtration")
        if score_name != self.score_name:
            raise ValueError("edge score definition changed across the filtration")


@dataclass(frozen=True)
class ForwardTimeUnroll(Generic[NodeT]):
    """Finite-horizon event graph for strictly positive template delays."""

    template_nodes: tuple[NodeT, ...]
    delayed_template_edges: tuple[tuple[NodeT, NodeT, int], ...]
    horizon: int
    event_nodes: tuple[tuple[NodeT, int], ...]
    event_edges: tuple[tuple[tuple[NodeT, int], tuple[NodeT, int]], ...]
    projected_template_edges: tuple[tuple[NodeT, NodeT], ...]
    decomposition: SCCDecomposition[tuple[NodeT, int]]


@dataclass(frozen=True)
class Arch1Validation(Generic[NodeT]):
    """Result of checking the finite DAG module-realization theorem."""

    valid: bool
    errors: tuple[str, ...]
    modules: tuple[tuple[NodeT, ...], ...]
    target_edges: tuple[tuple[int, int], ...]
    target_topological_order: tuple[int, ...]
    module_component_ids: tuple[int, ...]
    decomposition: SCCDecomposition[NodeT]


@dataclass(frozen=True)
class Arch1Construction(Generic[NodeT]):
    """Constructed graph whose SCC condensation is a declared finite DAG."""

    nodes: tuple[NodeT, ...]
    edges: tuple[tuple[NodeT, NodeT], ...]
    modules: tuple[tuple[NodeT, ...], ...]
    target_edges: tuple[tuple[int, int], ...]
    validation: Arch1Validation[NodeT]


@dataclass(frozen=True)
class BlockGainCertificate:
    """Finite simultaneous-update small-gain certificate.

    The locked orientation is ``gain_matrix[target, source]``.  The function
    issuing this object rejects cyclic off-diagonal support, schedule ambiguity,
    non-normalized metrics, and numerically unusable Neumann solves.
    """

    gain_matrix: tuple[tuple[float, ...], ...]
    gain_orientation: str
    schedule: str
    normalization_scales: tuple[float, ...]
    topological_order: tuple[int, ...]
    spectral_radius: float
    neumann_weights: tuple[float, ...]
    contraction_factor: float
    condition_number: float
    condition_limit: float
    solve_relative_residual: float
    neumann_inverse: tuple[tuple[float, ...], ...]
    certified: bool

    def residual_bound(self, block_residuals: Sequence[float]) -> tuple[float, ...]:
        """Return ``(I-M)^-1 e`` for nonnegative normalized block residuals."""

        residual = np.asarray(block_residuals, dtype=np.float64)
        size = len(self.gain_matrix)
        if residual.shape != (size,):
            raise ValueError("block_residuals must contain one value per block")
        if not np.all(np.isfinite(residual)) or np.any(residual < 0.0):
            raise ValueError("block_residuals must be finite and nonnegative")
        bound = np.asarray(self.neumann_inverse, dtype=np.float64) @ residual
        if not np.all(np.isfinite(bound)) or np.any(bound < 0.0):
            raise FloatingPointError("residual bound is numerically unusable")
        return tuple(float(value) for value in bound)


@dataclass(frozen=True)
class GeometricErrorBound:
    """Typed scalar rollout bound for one explicitly named contraction side."""

    premise: str
    steps: int
    contraction_factor: float
    one_step_defect: float
    initial_error: float
    finite_horizon_bound: float
    asymptotic_bound: float


def _normalise_nodes(nodes: Sequence[NodeT]) -> tuple[tuple[NodeT, ...], dict[NodeT, int]]:
    ordered = tuple(nodes)
    node_index: dict[NodeT, int] = {}
    for index, node in enumerate(ordered):
        try:
            duplicate = node in node_index
        except TypeError as error:
            raise TypeError("nodes must be hashable") from error
        if duplicate:
            raise ValueError("nodes must be unique in their declared order")
        node_index[node] = index
    return ordered, node_index


def _normalise_edges(
    edges: Iterable[tuple[NodeT, NodeT]],
    node_index: Mapping[NodeT, int],
) -> tuple[tuple[NodeT, NodeT], ...]:
    indexed: set[tuple[int, int]] = set()
    for source, target in edges:
        if source not in node_index or target not in node_index:
            raise ValueError("edge references a node outside the declared node set")
        indexed.add((node_index[source], node_index[target]))
    nodes_by_index = tuple(node_index)
    return tuple(
        (nodes_by_index[source], nodes_by_index[target]) for source, target in sorted(indexed)
    )


def _stable_topological_order(
    node_count: int,
    edges: Iterable[tuple[int, int]],
) -> tuple[int, ...]:
    adjacency: list[set[int]] = [set() for _ in range(node_count)]
    indegree = [0] * node_count
    for source, target in edges:
        if not 0 <= source < node_count or not 0 <= target < node_count:
            raise ValueError("DAG edge references an unknown vertex")
        if target not in adjacency[source]:
            adjacency[source].add(target)
            indegree[target] += 1
    ready = [node for node, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        source = heapq.heappop(ready)
        order.append(source)
        for target in sorted(adjacency[source]):
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)
    if len(order) != node_count:
        raise ValueError("directed graph must be acyclic")
    return tuple(order)


def decompose_scc(
    nodes: Sequence[NodeT],
    edges: Iterable[tuple[NodeT, NodeT]],
) -> SCCDecomposition[NodeT]:
    """Return the deterministic maximal-SCC partition and condensation DAG.

    Component members and component identifiers follow the declared node order;
    the condensation receives a separate stable topological order.
    """

    ordered_nodes, node_index = _normalise_nodes(nodes)
    ordered_edges = _normalise_edges(edges, node_index)
    node_count = len(ordered_nodes)
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    reverse: list[list[int]] = [[] for _ in range(node_count)]
    self_loops: set[int] = set()
    for source, target in ordered_edges:
        source_index = node_index[source]
        target_index = node_index[target]
        adjacency[source_index].append(target_index)
        reverse[target_index].append(source_index)
        if source_index == target_index:
            self_loops.add(source_index)
    for neighbours in adjacency:
        neighbours.sort()
    for neighbours in reverse:
        neighbours.sort()

    seen = [False] * node_count
    finish_order: list[int] = []
    for root in range(node_count):
        if seen[root]:
            continue
        seen[root] = True
        stack: list[tuple[int, int]] = [(root, 0)]
        while stack:
            node, next_neighbour = stack[-1]
            if next_neighbour == len(adjacency[node]):
                stack.pop()
                finish_order.append(node)
                continue
            target = adjacency[node][next_neighbour]
            stack[-1] = (node, next_neighbour + 1)
            if not seen[target]:
                seen[target] = True
                stack.append((target, 0))

    assigned = [False] * node_count
    component_indices: list[tuple[int, ...]] = []
    for root in reversed(finish_order):
        if assigned[root]:
            continue
        assigned[root] = True
        members: list[int] = []
        pending = [root]
        while pending:
            node = pending.pop()
            members.append(node)
            for source in reversed(reverse[node]):
                if not assigned[source]:
                    assigned[source] = True
                    pending.append(source)
        component_indices.append(tuple(sorted(members)))
    component_indices.sort(key=lambda members: members[0])

    components = tuple(
        tuple(ordered_nodes[index] for index in members) for members in component_indices
    )
    mutable_component_of: dict[NodeT, int] = {}
    component_index_of_node = [-1] * node_count
    for component_id, members in enumerate(component_indices):
        for node in members:
            component_index_of_node[node] = component_id
            mutable_component_of[ordered_nodes[node]] = component_id

    condensation = {
        (component_index_of_node[node_index[source]], component_index_of_node[node_index[target]])
        for source, target in ordered_edges
        if component_index_of_node[node_index[source]]
        != component_index_of_node[node_index[target]]
    }
    condensation_edges = tuple(sorted(condensation))
    topological_order = _stable_topological_order(len(components), condensation_edges)
    recurrent = tuple(len(members) > 1 or members[0] in self_loops for members in component_indices)
    return SCCDecomposition(
        nodes=ordered_nodes,
        edges=ordered_edges,
        components=components,
        component_of=MappingProxyType(mutable_component_of),
        condensation_edges=condensation_edges,
        topological_order=topological_order,
        component_is_recurrent=recurrent,
    )


def threshold_scc_filtration(
    nodes: Sequence[NodeT],
    weighted_edges: Iterable[tuple[NodeT, NodeT, float]],
    thresholds: Sequence[float],
    *,
    edge_semantics: str,
    layer: str,
    score_name: str,
    tie_rule: str = ">=",
) -> ThresholdFiltration[NodeT]:
    """Build a fixed-node merge filtration as thresholds decrease."""

    ordered_nodes, node_index = _normalise_nodes(nodes)
    for name, value in (
        ("edge_semantics", edge_semantics),
        ("layer", layer),
        ("score_name", score_name),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty declaration")
    if tie_rule not in {">=", ">"}:
        raise ValueError("tie_rule must be '>=' or '>'")

    ordered_thresholds = tuple(float(value) for value in thresholds)
    if not ordered_thresholds:
        raise ValueError("at least one threshold is required")
    if any(not math.isfinite(value) or value < 0.0 for value in ordered_thresholds):
        raise ValueError("thresholds must be finite and nonnegative")
    if any(
        ordered_thresholds[index] <= ordered_thresholds[index + 1]
        for index in range(len(ordered_thresholds) - 1)
    ):
        raise ValueError("thresholds must be strictly decreasing")

    scored: dict[tuple[int, int], float] = {}
    for source, target, score_value in weighted_edges:
        if source not in node_index or target not in node_index:
            raise ValueError("weighted edge references an unknown node")
        score = float(score_value)
        if not math.isfinite(score) or score < 0.0:
            raise ValueError("edge scores must be finite and nonnegative")
        key = (node_index[source], node_index[target])
        if key in scored:
            raise ValueError("weighted directed edges must be unique")
        scored[key] = score

    levels: list[ThresholdLevel[NodeT]] = []
    previous: SCCDecomposition[NodeT] | None = None
    for threshold in ordered_thresholds:
        if tie_rule == ">=":
            retained_indices = [edge for edge, score in scored.items() if score >= threshold]
        else:
            retained_indices = [edge for edge, score in scored.items() if score > threshold]
        retained = tuple(
            (ordered_nodes[source], ordered_nodes[target])
            for source, target in sorted(retained_indices)
        )
        decomposition = decompose_scc(ordered_nodes, retained)
        parents: tuple[int, ...] | None = None
        if previous is not None:
            mutable_parents: list[int] = []
            for component in previous.components:
                parent = decomposition.component_of[component[0]]
                if any(decomposition.component_of[node] != parent for node in component):
                    raise RuntimeError("edge-addition filtration split an existing SCC")
                mutable_parents.append(parent)
            parents = tuple(mutable_parents)
        levels.append(
            ThresholdLevel(
                threshold=threshold,
                retained_edges=retained,
                decomposition=decomposition,
                parent_of_previous=parents,
            )
        )
        previous = decomposition
    return ThresholdFiltration(
        nodes=ordered_nodes,
        edge_semantics=edge_semantics,
        layer=layer,
        score_name=score_name,
        tie_rule=tie_rule,
        levels=tuple(levels),
    )


def forward_time_unroll(
    nodes: Sequence[NodeT],
    delayed_edges: Iterable[tuple[NodeT, NodeT, int]],
    *,
    horizon: int,
) -> ForwardTimeUnroll[NodeT]:
    """Unroll positive-delay template edges over integer times ``0..horizon``."""

    ordered_nodes, node_index = _normalise_nodes(nodes)
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer")
    delayed_index_edges: set[tuple[int, int, int]] = set()
    for source, target, delay in delayed_edges:
        if source not in node_index or target not in node_index:
            raise ValueError("delayed edge references an unknown node")
        if isinstance(delay, bool) or not isinstance(delay, int) or delay <= 0:
            raise ValueError("every unroll delay must be a positive integer")
        delayed_index_edges.add((node_index[source], node_index[target], delay))
    ordered_delayed = tuple(
        (ordered_nodes[source], ordered_nodes[target], delay)
        for source, target, delay in sorted(delayed_index_edges)
    )
    event_nodes = tuple((node, time) for time in range(horizon + 1) for node in ordered_nodes)
    mutable_event_edges: list[tuple[tuple[NodeT, int], tuple[NodeT, int]]] = []
    for time in range(horizon + 1):
        for source, target, delay in ordered_delayed:
            if time + delay <= horizon:
                mutable_event_edges.append(((source, time), (target, time + delay)))
    event_edges = tuple(mutable_event_edges)
    decomposition = decompose_scc(event_nodes, event_edges)
    if any(len(component) != 1 for component in decomposition.components):
        raise RuntimeError("positive-delay forward unroll unexpectedly contains a cycle")
    projected = tuple(
        (ordered_nodes[source], ordered_nodes[target])
        for source, target in sorted(
            {(source, target) for source, target, _ in delayed_index_edges}
        )
    )
    return ForwardTimeUnroll(
        template_nodes=ordered_nodes,
        delayed_template_edges=ordered_delayed,
        horizon=horizon,
        event_nodes=event_nodes,
        event_edges=event_edges,
        projected_template_edges=projected,
        decomposition=decomposition,
    )


def project_time_coordinate(event_vertex: tuple[NodeT, int]) -> NodeT:
    """Apply the separately declared time-coordinate quotient ``(v,t) -> v``."""

    node, time = event_vertex
    if isinstance(time, bool) or not isinstance(time, int) or time < 0:
        raise ValueError("event time must be a nonnegative integer")
    return node


def _normalise_modules(
    modules: Sequence[Sequence[NodeT]],
) -> tuple[tuple[tuple[NodeT, ...], ...], tuple[NodeT, ...], dict[NodeT, int]]:
    ordered_modules = tuple(tuple(module) for module in modules)
    if not ordered_modules or any(not module for module in ordered_modules):
        raise ValueError("ARCH-1 requires at least one nonempty module")
    flattened = tuple(node for module in ordered_modules for node in module)
    ordered_nodes, node_index = _normalise_nodes(flattened)
    return ordered_modules, ordered_nodes, node_index


def _normalise_target_edges(
    module_count: int,
    target_edges: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    normalized: set[tuple[int, int]] = set()
    for source, target in target_edges:
        if (
            isinstance(source, bool)
            or isinstance(target, bool)
            or not isinstance(source, int)
            or not isinstance(target, int)
        ):
            raise TypeError("target module identifiers must be integers")
        if not 0 <= source < module_count or not 0 <= target < module_count:
            raise ValueError("target edge references an unknown module")
        if source == target:
            raise ValueError("target condensation edges cannot be self-loops")
        normalized.add((source, target))
    return tuple(sorted(normalized))


def validate_arch1(
    modules: Sequence[Sequence[NodeT]],
    graph_edges: Iterable[tuple[NodeT, NodeT]],
    target_edges: Iterable[tuple[int, int]],
) -> Arch1Validation[NodeT]:
    """Validate that modules are exactly the SCCs and realize the target DAG."""

    ordered_modules, ordered_nodes, node_index = _normalise_modules(modules)
    ordered_edges = _normalise_edges(graph_edges, node_index)
    ordered_target = _normalise_target_edges(len(ordered_modules), target_edges)
    errors: list[str] = []
    try:
        target_order = _stable_topological_order(len(ordered_modules), ordered_target)
    except ValueError:
        target_order = ()
        errors.append("target module graph is not a DAG")

    module_of = {
        node: module_id for module_id, module in enumerate(ordered_modules) for node in module
    }
    cross_pairs = {
        (module_of[source], module_of[target])
        for source, target in ordered_edges
        if module_of[source] != module_of[target]
    }
    if cross_pairs != set(ordered_target):
        errors.append("cross-module edge relation does not exactly match the target graph")

    for module_id, module in enumerate(ordered_modules):
        internal_edges = tuple(
            (source, target)
            for source, target in ordered_edges
            if module_of[source] == module_id and module_of[target] == module_id
        )
        if len(decompose_scc(module, internal_edges).components) != 1:
            errors.append(f"module {module_id} is not internally strongly connected")

    decomposition = decompose_scc(ordered_nodes, ordered_edges)
    actual_components = {frozenset(component) for component in decomposition.components}
    expected_components = {frozenset(module) for module in ordered_modules}
    if actual_components != expected_components:
        errors.append("declared modules are not exactly the maximal SCCs")
    module_component_ids = tuple(
        decomposition.component_of[module[0]] for module in ordered_modules
    )
    if actual_components == expected_components:
        induced_target = {
            (module_component_ids[source], module_component_ids[target])
            for source, target in ordered_target
        }
        if induced_target != set(decomposition.condensation_edges):
            errors.append("condensation edges do not exactly realize the target graph")
    return Arch1Validation(
        valid=not errors,
        errors=tuple(errors),
        modules=ordered_modules,
        target_edges=ordered_target,
        target_topological_order=target_order,
        module_component_ids=module_component_ids,
        decomposition=decomposition,
    )


def construct_arch1(
    modules: Sequence[Sequence[NodeT]],
    target_edges: Iterable[tuple[int, int]],
) -> Arch1Construction[NodeT]:
    """Construct one exact finite ARCH-1 realization using cycle modules."""

    ordered_modules, ordered_nodes, _ = _normalise_modules(modules)
    ordered_target = _normalise_target_edges(len(ordered_modules), target_edges)
    try:
        _stable_topological_order(len(ordered_modules), ordered_target)
    except ValueError as error:
        raise ValueError("ARCH-1 target module graph must be a DAG") from error
    edges: set[tuple[NodeT, NodeT]] = set()
    for module in ordered_modules:
        for index, source in enumerate(module):
            edges.add((source, module[(index + 1) % len(module)]))
    for source_module, target_module in ordered_target:
        edges.add((ordered_modules[source_module][0], ordered_modules[target_module][0]))
    ordered_edges = _normalise_edges(edges, {node: i for i, node in enumerate(ordered_nodes)})
    validation = validate_arch1(ordered_modules, ordered_edges, ordered_target)
    if not validation.valid:
        raise RuntimeError("internal ARCH-1 construction failed validation")
    return Arch1Construction(
        nodes=ordered_nodes,
        edges=ordered_edges,
        modules=ordered_modules,
        target_edges=ordered_target,
        validation=validation,
    )


def certify_dag_block_gain(
    gain_matrix: ArrayLike,
    *,
    normalization_scales: Sequence[float],
    schedule: str,
    condition_limit: float = 1e12,
    residual_tolerance: float = 1e-12,
) -> BlockGainCertificate:
    """Certify a finite DAG block-gain map in a weighted maximum norm.

    ``gain_matrix[target, source]`` is dimensionless in the declared normalized
    block metrics.  Only the simultaneous frozen-map schedule is covered here.
    """

    matrix = np.asarray(gain_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("gain_matrix must be a nonempty finite square matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("gain_matrix must be finite and nonnegative")
    if schedule != "simultaneous":
        raise ValueError("certificate only covers the declared simultaneous schedule")
    size = matrix.shape[0]
    scales = np.asarray(normalization_scales, dtype=np.float64)
    if scales.shape != (size,):
        raise ValueError("normalization_scales must contain one value per block")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("normalization_scales must be finite and positive")
    if not math.isfinite(condition_limit) or condition_limit <= 1.0:
        raise ValueError("condition_limit must be finite and greater than one")
    if not math.isfinite(residual_tolerance) or residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and positive")
    diagonal = np.diag(matrix)
    if np.any(diagonal >= 1.0):
        raise ValueError("every diagonal self-gain must be strictly below one")

    dependency_edges = tuple(
        (source, target)
        for target in range(size)
        for source in range(size)
        if source != target and matrix[target, source] > 0.0
    )
    try:
        topological_order = _stable_topological_order(size, dependency_edges)
    except ValueError as error:
        raise ValueError("off-diagonal gain support must be a finite DAG") from error

    system = np.eye(size, dtype=np.float64) - matrix
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        condition_number = float(np.linalg.cond(system, p=np.inf))
    if not math.isfinite(condition_number) or condition_number > condition_limit:
        raise FloatingPointError("Neumann system exceeds the declared condition limit")
    ones = np.ones(size, dtype=np.float64)
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            weights = np.linalg.solve(system, ones)
            inverse = np.linalg.solve(system, np.eye(size, dtype=np.float64))
            weighted_image = matrix @ weights
    except np.linalg.LinAlgError as error:
        raise FloatingPointError("Neumann system solve failed") from error
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise FloatingPointError("Neumann weights or inverse are numerically unusable")
    if not np.all(np.isfinite(inverse)):
        raise FloatingPointError("Neumann weights or inverse are numerically unusable")
    inverse_negative_tolerance = (
        64.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(inverse))))
    )
    if np.any(inverse < -inverse_negative_tolerance):
        raise FloatingPointError("Neumann weights or inverse are numerically unusable")
    inverse = np.maximum(inverse, 0.0)
    if not np.all(np.isfinite(weighted_image)):
        raise FloatingPointError("Neumann weights or inverse are numerically unusable")
    denominator = max(
        1.0,
        float(np.linalg.norm(system, ord=np.inf)) * float(np.linalg.norm(weights, ord=np.inf)),
    )
    solve_relative_residual = float(
        np.linalg.norm(system @ weights - ones, ord=np.inf) / denominator
    )
    if not math.isfinite(solve_relative_residual) or solve_relative_residual > residual_tolerance:
        raise FloatingPointError("Neumann solve residual exceeds the declared tolerance")
    ratios = weighted_image / weights
    if not np.all(np.isfinite(ratios)) or np.any(ratios < 0.0):
        raise FloatingPointError("weighted contraction ratios are numerically unusable")
    contraction_factor = float(np.max(ratios))
    if not contraction_factor < 1.0:
        raise FloatingPointError("finite precision did not certify q < 1")
    spectral_radius = float(np.max(diagonal))
    return BlockGainCertificate(
        gain_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        gain_orientation="M[target, source]",
        schedule=schedule,
        normalization_scales=tuple(float(value) for value in scales),
        topological_order=topological_order,
        spectral_radius=spectral_radius,
        neumann_weights=tuple(float(value) for value in weights),
        contraction_factor=contraction_factor,
        condition_number=condition_number,
        condition_limit=float(condition_limit),
        solve_relative_residual=solve_relative_residual,
        neumann_inverse=tuple(tuple(float(value) for value in row) for row in inverse),
        certified=True,
    )


def _geometric_error_bound(
    *,
    premise: str,
    initial_error: float,
    one_step_defect: float,
    contraction_factor: float,
    steps: int,
) -> GeometricErrorBound:
    values = (initial_error, one_step_defect, contraction_factor)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("errors, defect, and contraction factor must be finite")
    if initial_error < 0.0 or one_step_defect < 0.0:
        raise ValueError("initial error and one-step defect must be nonnegative")
    if not 0.0 <= contraction_factor < 1.0:
        raise ValueError("contraction factor must lie in [0, 1)")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    decay = contraction_factor**steps
    asymptotic = one_step_defect / (1.0 - contraction_factor)
    finite = decay * initial_error + (1.0 - decay) * asymptotic
    if not math.isfinite(finite) or not math.isfinite(asymptotic):
        raise FloatingPointError("geometric rollout bound is nonfinite")
    return GeometricErrorBound(
        premise=premise,
        steps=steps,
        contraction_factor=contraction_factor,
        one_step_defect=one_step_defect,
        initial_error=initial_error,
        finite_horizon_bound=finite,
        asymptotic_bound=asymptotic,
    )


def decoder_f_contraction_error_bound(
    *,
    initial_decoder_error: float,
    decoder_defect: float,
    f_contraction: float,
    steps: int,
) -> GeometricErrorBound:
    """Bound decoded microstate error assuming contraction of micro map ``F``."""

    return _geometric_error_bound(
        premise="decoder defect with F contraction",
        initial_error=initial_decoder_error,
        one_step_defect=decoder_defect,
        contraction_factor=f_contraction,
        steps=steps,
    )


def encoder_phi_contraction_error_bound(
    *,
    initial_encoder_error: float,
    encoder_defect: float,
    phi_contraction: float,
    steps: int,
) -> GeometricErrorBound:
    """Bound encoded macrostate error assuming contraction of macro map ``Phi``."""

    return _geometric_error_bound(
        premise="encoder defect with Phi contraction",
        initial_error=initial_encoder_error,
        one_step_defect=encoder_defect,
        contraction_factor=phi_contraction,
        steps=steps,
    )
