"""Finite, generated prefixes of an ideal nested strongly connected tower.

This module is an isolated research helper.  It proves no biological identity
and never materialises an infinite graph.  The generator exposes finite
bidirected shell/path prefixes, a complete local predecessor rule, and a
depth-independent global-coordinate-sup-norm contraction certificate for the
*declared* Jacobi update.

Graph nesting, dynamical compatibility, and contraction are deliberately
separate certificates.  In particular, the default cross-boundary coupling
makes append-zero inclusion incompatible away from the invariant zero fixture.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Iterable, Literal, Sequence

import numpy as np


Node = tuple[int, int]
Edge = tuple[Node, Node]


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, not bool or encoded text")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite and dimensionless")
    return result


def _finite_vector(
    values: Sequence[float],
    *,
    name: str,
    width: int,
    bounded: bool = False,
) -> np.ndarray:
    try:
        raw = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a finite width-{width} vector") from error
    if len(raw) != width:
        raise ValueError(f"{name} must have shape ({width},)")
    result = np.asarray(
        tuple(_finite_float(value, f"{name}[{index}]") for index, value in enumerate(raw)),
        dtype=np.float64,
    )
    if bounded and np.any(np.abs(result) > 1.0):
        raise ValueError("normalized tower state must lie in [-1, 1]")
    return result


@dataclass(frozen=True)
class TowerSpec:
    """Finite generator parameters for the isolated nested-tower fixture.

    ``observation_scales`` are named reference scales: raw observations are
    divided by them before entering the dimensionless recurrent core.
    All gains, states, tolerances, and contraction factors are dimensionless.
    """

    shell_width: int = 3
    maximum_depth: int = 4
    observation_scales: tuple[float, ...] = ()
    recurrence_gain: float = 0.24
    upward_gain: float = 0.16
    downward_gain: float = 0.14
    input_gain: float = 0.45
    level_decay: float = 0.72
    contraction_cap: float = 0.95
    update_schedule: str = "previous_tick_jacobi"

    def __post_init__(self) -> None:
        if type(self.shell_width) is not int or self.shell_width < 1:
            raise ValueError("shell_width must be a positive integer")
        if type(self.maximum_depth) is not int or self.maximum_depth < 0:
            raise ValueError("maximum_depth must be a nonnegative integer")
        if type(self.update_schedule) is not str or self.update_schedule != "previous_tick_jacobi":
            raise ValueError("only previous_tick_jacobi is declared by this fixture")

        for name in (
            "recurrence_gain",
            "upward_gain",
            "downward_gain",
            "input_gain",
        ):
            value = _finite_float(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        level_decay = _finite_float(self.level_decay, "level_decay")
        if not 0.0 < level_decay <= 1.0:
            raise ValueError("level_decay must lie in (0, 1]")
        object.__setattr__(self, "level_decay", level_decay)
        contraction_cap = _finite_float(self.contraction_cap, "contraction_cap")
        if not 0.0 < contraction_cap < 1.0:
            raise ValueError("contraction_cap must lie strictly between zero and one")
        object.__setattr__(self, "contraction_cap", contraction_cap)

        scales = self.observation_scales
        if type(scales) is not tuple:
            raise ValueError(
                "observation_scales must be a tuple; only the exact empty tuple uses defaults"
            )
        if len(scales) == 0:
            scales = (1.0,) * self.shell_width
            object.__setattr__(self, "observation_scales", scales)
        if len(scales) != self.shell_width:
            raise ValueError("observation_scales must match shell_width")
        normalized_scales = tuple(
            _finite_float(value, f"observation_scales[{index}]")
            for index, value in enumerate(scales)
        )
        if len(normalized_scales) != self.shell_width:
            raise ValueError("canonical observation_scales must match shell_width")
        if any(value <= 0.0 for value in normalized_scales):
            raise ValueError("every observation reference scale must be positive")
        object.__setattr__(self, "observation_scales", normalized_scales)


@dataclass(frozen=True)
class TowerManifest:
    generator: str
    spec_hash: str
    parameter_hash: str
    update_schedule: str
    state_domain: tuple[float, float]
    serialized_operator_scalar_count: int
    maximum_in_degree: int


@dataclass(frozen=True)
class PrefixGraph:
    depth: int
    vertices: tuple[Node, ...]
    edges: tuple[Edge, ...]
    manifest_hash: str


@dataclass(frozen=True)
class PrefixAudit:
    depth: int
    vertex_count: int
    edge_count: int
    component_count: int
    components: tuple[tuple[Node, ...], ...]
    is_strongly_connected: bool
    nested_with_previous: bool
    manifest_hash: str


@dataclass(frozen=True)
class ContractionCertificate:
    depth: int
    schedule: str
    metric: Literal["global_coordinate_sup"]
    level_independent_bound: float
    contraction_cap: float
    certified: bool
    state_domain: tuple[float, float]
    reason: str


@dataclass(frozen=True)
class CompatibilityCertificate:
    lower_depth: int
    embedding: str
    declared_domain: str
    certified: bool
    witness_defect: float
    reason: str


@dataclass(frozen=True)
class InfiniteTailCertificate:
    prefix_depth: int
    schedule: str
    metric: Literal["global_coordinate_sup"]
    uniform_contraction_bound: float
    level_decay: float
    boundary_defect_bound: float
    fixed_point_error_bound: float
    certified: bool
    reason: str


@dataclass(frozen=True)
class RolloutTailCertificate:
    prefix_depth: int
    horizon: int
    metric: Literal["global_coordinate_sup"]
    uniform_contraction_bound: float
    initial_error_bound: float
    boundary_defect_bound: float
    rollout_error_bound: float
    certified: bool
    reason: str


@dataclass(frozen=True)
class CausalConeCertificate:
    query_nodes: tuple[Node, ...]
    horizon: int
    nodes: tuple[Node, ...]
    maximum_birth_depth: int
    maximum_in_degree: int
    cardinality_bound: int
    predecessor_complete: bool
    predecessor_manifest_hash: str


@dataclass(frozen=True)
class EventUnrollAudit:
    template_depth: int
    horizon: int
    vertex_count: int
    edge_count: int
    acyclic: bool
    singleton_component_count: int
    manifest_hash: str


def strongly_connected_components(
    vertices: Iterable[Node], edges: Iterable[Edge]
) -> tuple[tuple[Node, ...], ...]:
    """Return the deterministic maximal-SCC partition of a finite graph.

    The implementation is intentionally independent from ``scc_atlas``.  For
    each remaining root it intersects forward and reverse reachability; this is
    slower than Tarjan but makes the audit logic compact and transparent.
    """

    ordered_vertices = tuple(sorted(set(vertices)))
    vertex_set = set(ordered_vertices)
    adjacency = {node: set() for node in ordered_vertices}
    reverse = {node: set() for node in ordered_vertices}
    for source, target in set(edges):
        if source not in vertex_set or target not in vertex_set:
            raise ValueError("every edge endpoint must be in the declared vertex set")
        adjacency[source].add(target)
        reverse[target].add(source)

    def reachable(root: Node, graph: dict[Node, set[Node]]) -> set[Node]:
        found = {root}
        pending = [root]
        while pending:
            node = pending.pop()
            for target in graph[node]:
                if target not in found:
                    found.add(target)
                    pending.append(target)
        return found

    unassigned = set(ordered_vertices)
    components: list[tuple[Node, ...]] = []
    while unassigned:
        root = min(unassigned)
        component = reachable(root, adjacency) & reachable(root, reverse)
        components.append(tuple(sorted(component)))
        unassigned.difference_update(component)
    return tuple(sorted(components, key=lambda component: component[0]))


class NestedTowerGenerator:
    """A finite rule for nested bidirected shell/path prefixes.

    A node is ``(level, within_shell_index)``.  Each shell is a bidirected
    path (with self dependencies), and equal indices in adjacent shells have
    reciprocal bridges.  Therefore every finite prefix is strongly connected.
    """

    __slots__ = (
        "_identity",
        "_manifest",
        "_sealed_manifest_hash",
        "_sealed_parameter_hash",
        "_sealed_spec_hash",
        "_spec",
        "_within_base",
    )

    def __init__(self, spec: TowerSpec | None = None) -> None:
        if spec is None:
            spec = TowerSpec()
        if type(spec) is not TowerSpec:
            raise TypeError("spec must be a TowerSpec")
        self._spec = spec
        self._within_base = self._make_within_base(spec.shell_width)
        self._identity = np.eye(spec.shell_width, dtype=np.float64)
        self._within_base.setflags(write=False)
        self._identity.setflags(write=False)
        self._manifest = self._build_manifest()
        self._sealed_parameter_hash = self._manifest.parameter_hash
        self._sealed_spec_hash = self._manifest.spec_hash
        self._sealed_manifest_hash = _canonical_hash(asdict(self._manifest))

    @property
    def spec(self) -> TowerSpec:
        if hasattr(self, "_sealed_spec_hash"):
            self.assert_integrity()
        return self._spec

    @staticmethod
    def _make_within_base(width: int) -> np.ndarray:
        matrix = np.eye(width, dtype=np.float64)
        for index in range(width - 1):
            matrix[index, index + 1] = 1.0
            matrix[index + 1, index] = 1.0
        row_sums = np.sum(np.abs(matrix), axis=1, keepdims=True)
        return matrix / row_sums

    def _build_manifest(self) -> TowerManifest:
        spec_payload = asdict(self._spec)
        spec_hash = _canonical_hash(spec_payload)
        parameter_hash = _canonical_hash(self._parameter_payload(spec_hash))
        width = self._spec.shell_width
        return TowerManifest(
            generator="bidirected_path_shells_v1",
            spec_hash=spec_hash,
            parameter_hash=parameter_hash,
            update_schedule=self._spec.update_schedule,
            state_domain=(-1.0, 1.0),
            # Two serialized width-by-width operator templates, one scale per
            # coordinate, and six floating dynamics coefficients.  This is
            # metadata, not a trainable-parameter, capacity, or MAC count.
            serialized_operator_scalar_count=2 * width * width + width + 6,
            maximum_in_degree=5,
        )

    def _parameter_payload(self, spec_hash: str | None = None) -> dict[str, object]:
        live_spec_hash = _canonical_hash(asdict(self._spec)) if spec_hash is None else spec_hash
        return {
            "spec_hash": live_spec_hash,
            "within_base": self._within_base.tolist(),
            "bridge_base": self._identity.tolist(),
            "level_rule": "gain * level_decay ** (bridge_level + 1)",
            "input_rule": "level_zero_only",
        }

    def assert_integrity(self) -> None:
        """Fail closed if sealed generator arrays or specification changed."""

        live_spec_hash = _canonical_hash(asdict(self._spec))
        live_parameter_hash = _canonical_hash(self._parameter_payload(live_spec_hash))
        if (
            self._within_base.flags.writeable
            or self._identity.flags.writeable
            or self._manifest.spec_hash != self._sealed_spec_hash
            or self._manifest.parameter_hash != self._sealed_parameter_hash
            or _canonical_hash(asdict(self._manifest)) != self._sealed_manifest_hash
            or live_spec_hash != self._sealed_spec_hash
            or live_parameter_hash != self._sealed_parameter_hash
        ):
            raise ValueError("nested tower generator integrity seal mismatch")

    @property
    def manifest(self) -> TowerManifest:
        self.assert_integrity()
        return self._manifest

    def _validate_depth(self, depth: int) -> int:
        self.assert_integrity()
        if type(depth) is not int:
            raise TypeError("depth must be an integer")
        if depth < 0 or depth > self.spec.maximum_depth:
            raise ValueError(f"depth must lie in [0, {self.spec.maximum_depth}]")
        return depth

    def _validate_infinite_node(self, node: Node) -> Node:
        self.assert_integrity()
        if type(node) is not tuple or len(node) != 2:
            raise TypeError("node must be an integer pair (level, index)")
        level, index = node
        if type(level) is not int or type(index) is not int:
            raise TypeError("node must be an integer pair (level, index)")
        if level < 0 or not 0 <= index < self.spec.shell_width:
            raise ValueError("node is outside the generated infinite tower")
        return (level, index)

    def vertices(self, depth: int) -> tuple[Node, ...]:
        self.assert_integrity()
        depth = self._validate_depth(depth)
        return tuple(
            (level, index) for level in range(depth + 1) for index in range(self.spec.shell_width)
        )

    def edges(self, depth: int) -> tuple[Edge, ...]:
        self.assert_integrity()
        depth = self._validate_depth(depth)
        edges: set[Edge] = set()
        for level in range(depth + 1):
            for index in range(self.spec.shell_width):
                node = (level, index)
                edges.add((node, node))
                if index + 1 < self.spec.shell_width:
                    neighbor = (level, index + 1)
                    edges.add((node, neighbor))
                    edges.add((neighbor, node))
                if level < depth:
                    outer = (level + 1, index)
                    edges.add((node, outer))
                    edges.add((outer, node))
        return tuple(sorted(edges))

    def prefix(self, depth: int) -> PrefixGraph:
        self.assert_integrity()
        vertices = self.vertices(depth)
        edges = self.edges(depth)
        payload = {
            "depth": depth,
            "parameter_hash": self.manifest.parameter_hash,
            "vertices": vertices,
            "edges": edges,
        }
        return PrefixGraph(
            depth=depth,
            vertices=vertices,
            edges=edges,
            manifest_hash=_canonical_hash(payload),
        )

    def audit_prefix(self, depth: int) -> PrefixAudit:
        self.assert_integrity()
        prefix = self.prefix(depth)
        components = strongly_connected_components(prefix.vertices, prefix.edges)
        nested = True
        if depth > 0:
            previous = self.prefix(depth - 1)
            nested = set(previous.vertices) < set(prefix.vertices) and set(previous.edges).issubset(
                prefix.edges
            )
        return PrefixAudit(
            depth=depth,
            vertex_count=len(prefix.vertices),
            edge_count=len(prefix.edges),
            component_count=len(components),
            components=components,
            is_strongly_connected=len(components) == 1,
            nested_with_previous=nested,
            manifest_hash=prefix.manifest_hash,
        )

    def predecessors(self, node: Node) -> tuple[Node, ...]:
        """Return every incoming neighbour in the ideal union graph."""

        self.assert_integrity()
        level, index = self._validate_infinite_node(node)
        found = {node, (level + 1, index)}
        if level > 0:
            found.add((level - 1, index))
        if index > 0:
            found.add((level, index - 1))
        if index + 1 < self.spec.shell_width:
            found.add((level, index + 1))
        return tuple(sorted(found))

    def backward_causal_cone(
        self, query_nodes: Sequence[Node], horizon: int
    ) -> CausalConeCertificate:
        self.assert_integrity()
        if type(horizon) is not int or horizon < 0:
            raise ValueError("horizon must be a nonnegative integer")
        try:
            raw_queries = tuple(query_nodes)
        except TypeError as error:
            raise ValueError("query_nodes must be a finite node sequence") from error
        if not raw_queries:
            raise ValueError("query_nodes must be nonempty")
        queries = tuple(sorted({self._validate_infinite_node(node) for node in raw_queries}))
        distances = {node: 0 for node in queries}
        pending: deque[Node] = deque(queries)
        predecessor_rows: list[tuple[Node, tuple[Node, ...]]] = []
        while pending:
            node = pending.popleft()
            distance = distances[node]
            predecessors = self.predecessors(node)
            predecessor_rows.append((node, predecessors))
            if distance == horizon:
                continue
            for predecessor in predecessors:
                if predecessor not in distances:
                    distances[predecessor] = distance + 1
                    pending.append(predecessor)
        nodes = tuple(sorted(distances))
        degree = self.manifest.maximum_in_degree
        cardinality_bound = len(queries) * sum(degree**step for step in range(horizon + 1))
        return CausalConeCertificate(
            query_nodes=queries,
            horizon=horizon,
            nodes=nodes,
            maximum_birth_depth=max(level for level, _ in nodes),
            maximum_in_degree=degree,
            cardinality_bound=cardinality_bound,
            predecessor_complete=True,
            predecessor_manifest_hash=_canonical_hash(predecessor_rows),
        )

    def forward_unroll(self, depth: int, horizon: int) -> EventUnrollAudit:
        self.assert_integrity()
        prefix = self.prefix(depth)
        if type(horizon) is not int or horizon < 0:
            raise ValueError("horizon must be a nonnegative integer")
        event_vertices = tuple(
            (node, tick) for tick in range(horizon + 1) for node in prefix.vertices
        )
        event_edges = tuple(
            sorted(
                ((source, tick), (target, tick + 1))
                for tick in range(horizon)
                for source, target in prefix.edges
            )
        )
        return EventUnrollAudit(
            template_depth=depth,
            horizon=horizon,
            vertex_count=len(event_vertices),
            edge_count=len(event_edges),
            acyclic=all(source[1] < target[1] for source, target in event_edges),
            singleton_component_count=len(event_vertices),
            manifest_hash=_canonical_hash({"vertices": event_vertices, "edges": event_edges}),
        )

    def recurrence_operator(self) -> np.ndarray:
        self.assert_integrity()
        return (self.spec.recurrence_gain * self._within_base).copy()

    def upward_operator(self, bridge_level: int) -> np.ndarray:
        self.assert_integrity()
        bridge_level = self._validate_bridge(bridge_level)
        scale = self.spec.level_decay ** (bridge_level + 1)
        return (self.spec.upward_gain * scale * self._identity).copy()

    def downward_operator(self, bridge_level: int) -> np.ndarray:
        self.assert_integrity()
        bridge_level = self._validate_bridge(bridge_level)
        scale = self.spec.level_decay ** (bridge_level + 1)
        return (self.spec.downward_gain * scale * self._identity).copy()

    def _validate_bridge(self, bridge_level: int) -> int:
        self.assert_integrity()
        if type(bridge_level) is not int:
            raise TypeError("bridge_level must be an integer")
        if not 0 <= bridge_level < self.spec.maximum_depth:
            raise ValueError("bridge_level must connect two registered levels")
        return bridge_level

    def normalize_observation(self, observation: Sequence[float]) -> np.ndarray:
        self.assert_integrity()
        result = _finite_vector(
            observation,
            name="observation",
            width=self.spec.shell_width,
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            normalized = result / np.asarray(self.spec.observation_scales, dtype=np.float64)
        if not np.all(np.isfinite(normalized)):
            raise ValueError(
                "normalized observation must remain finite under the frozen reference scales"
            )
        return normalized

    def _validated_states(self, states: Sequence[Sequence[float]]) -> tuple[np.ndarray, ...]:
        try:
            raw_states = tuple(states)
        except TypeError as error:
            raise ValueError("states must be a finite outer sequence") from error
        if not raw_states:
            raise ValueError("states must contain at least level zero")
        depth = len(raw_states) - 1
        self._validate_depth(depth)
        result: list[np.ndarray] = []
        for level, values in enumerate(raw_states):
            state = _finite_vector(
                values,
                name=f"state at level {level}",
                width=self.spec.shell_width,
                bounded=True,
            )
            result.append(state.copy())
        return tuple(result)

    def bridge_messages(
        self, states: Sequence[Sequence[float]]
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        self.assert_integrity()
        old = self._validated_states(states)
        with np.errstate(invalid="ignore", over="ignore"):
            upward_raw = tuple(
                self.upward_operator(bridge) @ old[bridge] for bridge in range(len(old) - 1)
            )
            downward_raw = tuple(
                self.downward_operator(bridge) @ old[bridge + 1] for bridge in range(len(old) - 1)
            )
        upward = tuple(
            _finite_vector(
                values,
                name=f"generated upward message[{bridge}]",
                width=self.spec.shell_width,
            )
            for bridge, values in enumerate(upward_raw)
        )
        downward = tuple(
            _finite_vector(
                values,
                name=f"generated downward message[{bridge}]",
                width=self.spec.shell_width,
            )
            for bridge, values in enumerate(downward_raw)
        )
        return upward, downward

    def step_with_messages(
        self,
        states: Sequence[Sequence[float]],
        normalized_observation: Sequence[float],
        upward_messages: Sequence[Sequence[float]],
        downward_messages: Sequence[Sequence[float]],
    ) -> tuple[np.ndarray, ...]:
        """Apply one synchronous previous-tick/Jacobi update."""

        self.assert_integrity()
        old = self._validated_states(states)
        observation = _finite_vector(
            normalized_observation,
            name="normalized_observation",
            width=self.spec.shell_width,
        )
        bridge_count = len(old) - 1
        try:
            raw_upward = tuple(upward_messages)
            raw_downward = tuple(downward_messages)
        except TypeError as error:
            raise ValueError("message collections must be finite outer sequences") from error
        if len(raw_upward) != bridge_count or len(raw_downward) != bridge_count:
            raise ValueError("message sequences must match the active bridge count")

        def message(values: Sequence[float], name: str) -> np.ndarray:
            return _finite_vector(
                values,
                name=name,
                width=self.spec.shell_width,
            )

        upward = tuple(
            message(values, f"upward_messages[{index}]") for index, values in enumerate(raw_upward)
        )
        downward = tuple(
            message(values, f"downward_messages[{index}]")
            for index, values in enumerate(raw_downward)
        )
        recurrence = self.recurrence_operator()
        updated: list[np.ndarray] = []
        for level, state in enumerate(old):
            with np.errstate(invalid="ignore", over="ignore"):
                drive = recurrence @ state
                if level == 0:
                    drive = drive + self.spec.input_gain * observation
                if level > 0:
                    drive = drive + upward[level - 1]
                if level < bridge_count:
                    drive = drive + downward[level]
            if not np.all(np.isfinite(drive)):
                raise ValueError(f"dimensionless recurrent drive at level {level} overflowed")
            next_state = np.tanh(drive)
            if not np.all(np.isfinite(next_state)):
                raise ValueError(f"updated state at level {level} must remain finite")
            updated.append(next_state)
        return tuple(updated)

    def step(
        self,
        states: Sequence[Sequence[float]],
        normalized_observation: Sequence[float],
    ) -> tuple[np.ndarray, ...]:
        self.assert_integrity()
        try:
            raw_states = tuple(states)
        except TypeError as error:
            raise ValueError("states must be a finite outer sequence") from error
        upward, downward = self.bridge_messages(raw_states)
        return self.step_with_messages(raw_states, normalized_observation, upward, downward)

    def certify_prefix(self, depth: int, *, schedule: str | None = None) -> ContractionCertificate:
        self.assert_integrity()
        depth = self._validate_depth(depth)
        requested_schedule = self.spec.update_schedule if schedule is None else schedule
        if type(requested_schedule) is not str:
            raise ValueError("requested schedule must be an exact string")
        q = self.spec.recurrence_gain + self.spec.upward_gain + self.spec.downward_gain
        topology_ok = self.audit_prefix(depth).is_strongly_connected
        schedule_ok = requested_schedule == self.spec.update_schedule
        certified = bool(
            topology_ok
            and schedule_ok
            and math.isfinite(q)
            and q < 1.0
            and q <= self.spec.contraction_cap
        )
        if not topology_ok:
            reason = "no global-coordinate-sup certificate: prefix topology is not strong"
        elif not schedule_ok:
            reason = (
                "the global-coordinate-sup Jacobi gain bound cannot certify a different "
                "update schedule"
            )
        elif q >= 1.0:
            reason = "the global-coordinate-sup level-independent Lipschitz bound is not strict"
        elif q > self.spec.contraction_cap:
            reason = "the global-coordinate-sup strict bound exceeds the registered contraction cap"
        else:
            reason = (
                "in the global coordinate sup norm, tanh is one-Lipschitz and "
                "every block row sum is bounded by q"
            )
        return ContractionCertificate(
            depth=depth,
            schedule=requested_schedule,
            metric="global_coordinate_sup",
            level_independent_bound=q,
            contraction_cap=self.spec.contraction_cap,
            certified=certified,
            state_domain=(-1.0, 1.0),
            reason=reason,
        )

    def append_zero_defect(self, lower_depth: int) -> float:
        """Return an explicit nonzero witness to generic inclusion failure."""

        self.assert_integrity()
        lower_depth = self._validate_depth(lower_depth)
        if lower_depth >= self.spec.maximum_depth:
            raise ValueError("an upper registered level is required")
        lower = [np.zeros(self.spec.shell_width) for _ in range(lower_depth + 1)]
        lower[-1].fill(1.0)
        observation = np.zeros(self.spec.shell_width)
        lower_next = self.step(lower, observation)
        embedded = (*lower, np.zeros(self.spec.shell_width))
        upper_next = self.step(embedded, observation)
        embedded_lower_next = (*lower_next, np.zeros(self.spec.shell_width))
        return max(
            float(np.max(np.abs(left - right)))
            for left, right in zip(upper_next, embedded_lower_next)
        )

    def compatibility_certificate(
        self,
        lower_depth: int,
        *,
        domain: Literal["zero_state_zero_input", "append_zero_unit_cube"],
    ) -> CompatibilityCertificate:
        self.assert_integrity()
        lower_depth = self._validate_depth(lower_depth)
        if lower_depth >= self.spec.maximum_depth:
            raise ValueError("an upper registered level is required")
        if type(domain) is not str:
            raise ValueError("compatibility domain must be an exact string")
        if domain == "zero_state_zero_input":
            lower = tuple(np.zeros(self.spec.shell_width) for _ in range(lower_depth + 1))
            observation = np.zeros(self.spec.shell_width)
            lower_next = self.step(lower, observation)
            upper_next = self.step((*lower, np.zeros(self.spec.shell_width)), observation)
            target = (*lower_next, np.zeros(self.spec.shell_width))
            defect = max(
                float(np.max(np.abs(left - right))) for left, right in zip(upper_next, target)
            )
            return CompatibilityCertificate(
                lower_depth=lower_depth,
                embedding="append_zero",
                declared_domain=domain,
                certified=defect == 0.0,
                witness_defect=defect,
                reason=("the zero-state/zero-input singleton is an invariant embedding fixture"),
            )
        if domain != "append_zero_unit_cube":
            raise ValueError("unknown compatibility domain")
        defect = self.append_zero_defect(lower_depth)
        structurally_decoupled = self.spec.upward_gain == 0.0
        certified = structurally_decoupled and defect == 0.0
        reason = (
            "append-zero compatibility holds because upward boundary coupling is zero"
            if certified
            else "refused: nonzero upward cross-boundary coupling activates the appended shell"
        )
        return CompatibilityCertificate(
            lower_depth=lower_depth,
            embedding="append_zero",
            declared_domain=domain,
            certified=certified,
            witness_defect=defect,
            reason=reason,
        )

    def infinite_tail_certificate(self, prefix_depth: int) -> InfiniteTailCertificate:
        """Certify a uniform infinite-tail approximation without exact commutation.

        The ideal operator acts on the unit ball of the shellwise l-infinity
        product space.  Appending a zero tail leaves one boundary residual at
        level ``prefix_depth + 1``.  This is an analytic uniform-domain bound,
        not a sampled state defect and not an exact direct-limit certificate.
        """

        self.assert_integrity()
        prefix_depth = self._validate_depth(prefix_depth)
        contraction = self.certify_prefix(prefix_depth)
        q = contraction.level_independent_bound
        decay = self.spec.level_decay
        defect = self.spec.upward_gain * decay ** (prefix_depth + 1)
        strict_tail = decay < 1.0
        certified = bool(contraction.certified and strict_tail)
        error = defect / (1.0 - q) if q < 1.0 else math.inf
        if not contraction.certified:
            reason = "no infinite-tail bound: the uniform finite schedule is not contractive"
        elif not strict_tail:
            reason = "no convergent prefix approximation: level_decay must be strictly below one"
        else:
            reason = (
                "the nonzero append-zero boundary residual is uniformly bounded by "
                "upward_gain * level_decay ** (prefix_depth + 1), and the contraction "
                "resolvent bounds the infinite fixed-point tail"
            )
        return InfiniteTailCertificate(
            prefix_depth=prefix_depth,
            schedule=self.spec.update_schedule,
            metric="global_coordinate_sup",
            uniform_contraction_bound=q,
            level_decay=decay,
            boundary_defect_bound=defect,
            fixed_point_error_bound=error,
            certified=certified,
            reason=reason,
        )

    def rollout_tail_certificate(
        self,
        prefix_depth: int,
        horizon: int,
        *,
        initial_error_bound: float = 0.0,
    ) -> RolloutTailCertificate:
        """Propagate the uniform boundary residual for a finite causal rollout."""

        self.assert_integrity()
        if type(horizon) is not int or horizon < 0:
            raise ValueError("horizon must be a nonnegative integer")
        initial_error = _finite_float(initial_error_bound, "initial_error_bound")
        if initial_error < 0.0:
            raise ValueError("initial_error_bound must be nonnegative")
        tail = self.infinite_tail_certificate(prefix_depth)
        q = tail.uniform_contraction_bound
        if q < 1.0:
            geometric = (1.0 - q**horizon) / (1.0 - q)
            error = q**horizon * initial_error + geometric * tail.boundary_defect_bound
        else:
            error = math.inf
        return RolloutTailCertificate(
            prefix_depth=tail.prefix_depth,
            horizon=horizon,
            metric=tail.metric,
            uniform_contraction_bound=q,
            initial_error_bound=initial_error,
            boundary_defect_bound=tail.boundary_defect_bound,
            rollout_error_bound=error,
            certified=tail.certified,
            reason=(
                "the recursive error inequality E[t+1] <= q E[t] + boundary_defect "
                "was summed over the registered horizon"
                if tail.certified
                else tail.reason
            ),
        )

    def requires_extension(self, depth: int) -> bool:
        """Conservatively grow while generic append-zero compatibility fails.

        This is not a truncation-error certificate.  The controller never uses
        a sampled current-state defect to deactivate a level.
        """

        self.assert_integrity()
        depth = self._validate_depth(depth)
        if depth >= self.spec.maximum_depth:
            return False
        return not self.compatibility_certificate(depth, domain="append_zero_unit_cube").certified


__all__ = [
    "CausalConeCertificate",
    "CompatibilityCertificate",
    "ContractionCertificate",
    "Edge",
    "EventUnrollAudit",
    "InfiniteTailCertificate",
    "NestedTowerGenerator",
    "Node",
    "PrefixAudit",
    "PrefixGraph",
    "RolloutTailCertificate",
    "TowerManifest",
    "TowerSpec",
    "strongly_connected_components",
]
