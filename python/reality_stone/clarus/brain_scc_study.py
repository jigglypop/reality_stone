"""Typed design contract for a future scale-indexed brain SCC study.

This module does not contain biological data and never infers a brain identity.
It keeps SCC decompositions within each fixed directed graph separate from
cross-scale maps between graphs with different registered semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

from .scc_atlas import SCCDecomposition, decompose_scc


@dataclass(frozen=True)
class DirectedScaleGraph:
    scale_id: str
    scale_rank: int
    node_semantics: str
    edge_semantics: str
    direction_source: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for name in ("scale_id", "node_semantics", "edge_semantics", "direction_source"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a nonempty exact string")
        if type(self.scale_rank) is not int or self.scale_rank < 0:
            raise ValueError("scale_rank must be an exact nonnegative integer")
        if type(self.nodes) is not tuple or not self.nodes:
            raise ValueError("nodes must be a nonempty exact tuple")
        if any(type(node) is not str or not node for node in self.nodes):
            raise ValueError("every node must be a nonempty exact string")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("nodes must be unique")
        if type(self.edges) is not tuple:
            raise ValueError("edges must be an exact tuple")
        node_set = set(self.nodes)
        for edge in self.edges:
            if (
                type(edge) is not tuple
                or len(edge) != 2
                or any(type(node) is not str for node in edge)
                or edge[0] not in node_set
                or edge[1] not in node_set
            ):
                raise ValueError("every directed edge must use declared exact-string nodes")


@dataclass(frozen=True)
class CrossScaleNodeMap:
    fine_scale_id: str
    coarse_scale_id: str
    mapping: tuple[tuple[str, str], ...]
    mapping_semantics: str

    def __post_init__(self) -> None:
        for name in ("fine_scale_id", "coarse_scale_id", "mapping_semantics"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a nonempty exact string")
        if type(self.mapping) is not tuple:
            raise ValueError("mapping must be an exact tuple")
        if any(
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) is not str
            for pair in self.mapping
        ):
            raise ValueError("mapping rows must be exact string pairs")


@dataclass(frozen=True)
class ScaleSCCSummary:
    scale_id: str
    node_count: int
    component_count: int
    largest_component_fraction: float
    decomposition: SCCDecomposition[str]


@dataclass(frozen=True)
class BrainSCCStudyAudit:
    summaries: tuple[ScaleSCCSummary, ...]
    cross_scale_component_violations: int
    complete_node_maps: bool
    scale_compatible: bool
    fixed_graph_nested_maximal_claim_allowed: bool
    biological_identity_established: bool


def audit_scale_indexed_scc_study(
    views: tuple[DirectedScaleGraph, ...],
    maps: tuple[CrossScaleNodeMap, ...],
) -> BrainSCCStudyAudit:
    if type(views) is not tuple or not views:
        raise ValueError("views must be a nonempty exact tuple")
    if type(maps) is not tuple:
        raise ValueError("maps must be an exact tuple")
    if any(type(view) is not DirectedScaleGraph for view in views):
        raise ValueError("every view must be an exact DirectedScaleGraph")
    if any(type(mapping) is not CrossScaleNodeMap for mapping in maps):
        raise ValueError("every map must be an exact CrossScaleNodeMap")
    by_id = {view.scale_id: view for view in views}
    if len(by_id) != len(views):
        raise ValueError("scale ids must be unique")
    if len({view.scale_rank for view in views}) != len(views):
        raise ValueError("scale ranks must be unique")

    decompositions = {view.scale_id: decompose_scc(view.nodes, view.edges) for view in views}
    summaries = tuple(
        ScaleSCCSummary(
            scale_id=view.scale_id,
            node_count=len(view.nodes),
            component_count=len(decompositions[view.scale_id].components),
            largest_component_fraction=max(
                len(component) for component in decompositions[view.scale_id].components
            )
            / len(view.nodes),
            decomposition=decompositions[view.scale_id],
        )
        for view in sorted(views, key=lambda item: item.scale_rank)
    )

    violations = 0
    complete = True
    seen_pairs: set[tuple[str, str]] = set()
    for mapping in maps:
        if mapping.fine_scale_id not in by_id or mapping.coarse_scale_id not in by_id:
            raise ValueError("cross-scale map references an unknown scale")
        fine = by_id[mapping.fine_scale_id]
        coarse = by_id[mapping.coarse_scale_id]
        if fine.scale_rank >= coarse.scale_rank:
            raise ValueError("cross-scale maps must point from finer to coarser rank")
        pair = (fine.scale_id, coarse.scale_id)
        if pair in seen_pairs:
            raise ValueError("duplicate cross-scale map")
        seen_pairs.add(pair)
        rows = dict(mapping.mapping)
        if len(rows) != len(mapping.mapping):
            raise ValueError("every fine node must have exactly one mapping row")
        if set(rows) != set(fine.nodes) or any(
            target not in coarse.nodes for target in rows.values()
        ):
            complete = False
            continue
        coarse_component = decompositions[coarse.scale_id].component_of
        for component in decompositions[fine.scale_id].components:
            image_components = {coarse_component[rows[node]] for node in component}
            if len(image_components) != 1:
                violations += 1

    return BrainSCCStudyAudit(
        summaries=summaries,
        cross_scale_component_violations=violations,
        complete_node_maps=complete,
        scale_compatible=complete and violations == 0,
        # Maximal SCCs are disjoint inside every fixed graph.  Cross-scale
        # compatibility is a separate relation and never changes this no-go.
        fixed_graph_nested_maximal_claim_allowed=False,
        # Data provenance, estimator validity, and interventions are outside
        # this structural audit.
        biological_identity_established=False,
    )


__all__ = [
    "BrainSCCStudyAudit",
    "CrossScaleNodeMap",
    "DirectedScaleGraph",
    "ScaleSCCSummary",
    "audit_scale_indexed_scc_study",
]
