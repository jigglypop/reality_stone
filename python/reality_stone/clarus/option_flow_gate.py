"""Mass-conserving option routing with an explicit basal-ganglia HOLD channel.

The graph is an engineering abstraction evaluated as one finite decision slice.
It is not a claim that cortico-basal-ganglia-thalamic anatomy is acyclic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class OptionNode:
    node_id: int
    depth: int
    kind: str
    action_label: int | None = None


@dataclass(frozen=True)
class OptionEdge:
    source: int
    target: int


@dataclass(frozen=True)
class HoldGateConfig:
    tonic_inhibition: float = 0.0
    stn_bias: float = -2.0
    stn_conflict_gain: float = 4.0
    stn_to_gpi: float = 1.0
    direct_to_gpi: float = 1.0
    indirect_to_gpi: float = 1.0
    gpi_temperature: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("tonic_inhibition", self.tonic_inhibition),
            ("stn_bias", self.stn_bias),
            ("stn_conflict_gain", self.stn_conflict_gain),
            ("stn_to_gpi", self.stn_to_gpi),
            ("direct_to_gpi", self.direct_to_gpi),
            ("indirect_to_gpi", self.indirect_to_gpi),
            ("gpi_temperature", self.gpi_temperature),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.stn_conflict_gain < 0.0:
            raise ValueError("stn_conflict_gain must be nonnegative")
        if self.stn_to_gpi < 0.0:
            raise ValueError("stn_to_gpi must be nonnegative")
        if self.direct_to_gpi < 0.0 or self.indirect_to_gpi < 0.0:
            raise ValueError("striatal pathway gains must be nonnegative")
        if self.gpi_temperature <= 0.0:
            raise ValueError("gpi_temperature must be positive")


@dataclass(frozen=True)
class HoldGateOutput:
    hold_probability: float
    action_probabilities: tuple[float, ...]
    conditional_action_probabilities: tuple[float, ...]
    proposal_conflict: float
    stn_drive: float
    gpi_inhibition: tuple[float, ...]
    normalization_error: float


@dataclass(frozen=True)
class OptionFlowOutput:
    action_probabilities: tuple[tuple[int, float], ...]
    hold_probability: float
    node_flows: tuple[tuple[int, float], ...]
    normalization_error: float
    evaluated_nodes: int
    evaluated_edges: int

    def probability_of(self, action_label: int) -> float:
        return dict(self.action_probabilities).get(action_label, 0.0)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def _softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def gpi_hold_gate(
    proposal_logits: Sequence[float],
    direct_drive: Sequence[float],
    indirect_drive: Sequence[float],
    config: HoldGateConfig = HoldGateConfig(),
) -> HoldGateOutput:
    """Convert co-active striatal channels into action-or-HOLD probabilities.

    All inputs are dimensionless. STN contributes a common GPi inhibition, but
    HOLD is a separate reference logit, so this common term cannot cancel.
    """

    proposal = np.asarray(proposal_logits, dtype=np.float64)
    direct = np.asarray(direct_drive, dtype=np.float64)
    indirect = np.asarray(indirect_drive, dtype=np.float64)
    if proposal.ndim != 1 or proposal.size < 2:
        raise ValueError("at least two one-dimensional proposal channels are required")
    if direct.shape != proposal.shape or indirect.shape != proposal.shape:
        raise ValueError("proposal, direct, and indirect channels must have equal shape")
    if not np.all(np.isfinite(proposal)):
        raise ValueError("proposal logits must be finite")
    if not np.all(np.isfinite(direct)) or not np.all(np.isfinite(indirect)):
        raise ValueError("striatal drives must be finite")
    if np.any(direct < 0.0) or np.any(indirect < 0.0):
        raise ValueError("striatal drives must be nonnegative")

    proposal_probabilities = _softmax(proposal)
    entropy = -float(np.sum(proposal_probabilities * np.log(proposal_probabilities)))
    conflict = entropy / math.log(float(proposal.size))
    stn_drive = _softplus(config.stn_bias + config.stn_conflict_gain * conflict)
    gpi = (
        config.tonic_inhibition
        + config.stn_to_gpi * stn_drive
        - config.direct_to_gpi * direct
        + config.indirect_to_gpi * indirect
    )

    action_logits = -gpi / config.gpi_temperature
    joint_logits = np.concatenate((np.zeros(1, dtype=np.float64), action_logits))
    joint = _softmax(joint_logits)
    hold_probability = float(joint[0])
    actions = joint[1:]
    conditional = actions / float(np.sum(actions))
    normalization_error = abs(hold_probability + float(np.sum(actions)) - 1.0)
    return HoldGateOutput(
        hold_probability=hold_probability,
        action_probabilities=tuple(float(value) for value in actions),
        conditional_action_probabilities=tuple(float(value) for value in conditional),
        proposal_conflict=conflict,
        stn_drive=stn_drive,
        gpi_inhibition=tuple(float(value) for value in gpi),
        normalization_error=normalization_error,
    )


def validate_option_dag(nodes: Sequence[OptionNode], edges: Sequence[OptionEdge]) -> None:
    node_map = {node.node_id: node for node in nodes}
    if len(node_map) != len(nodes):
        raise ValueError("option node identifiers must be unique")
    if not nodes:
        raise ValueError("option DAG must contain at least one node")
    for node in nodes:
        if node.depth < 0:
            raise ValueError("option node depth must be nonnegative")
        if node.kind == "leaf" and node.action_label is None:
            raise ValueError("leaf nodes require an action label")
        if node.kind != "leaf" and node.action_label is not None:
            raise ValueError("only leaf nodes may carry action labels")
    for edge in edges:
        if edge.source not in node_map or edge.target not in node_map:
            raise ValueError("option edge references an unknown node")
        if node_map[edge.source].depth >= node_map[edge.target].depth:
            raise ValueError("every within-tick option edge must increase depth")


def route_option_flow(
    nodes: Sequence[OptionNode],
    edges: Sequence[OptionEdge],
    *,
    root_id: int,
    edge_probabilities: Mapping[tuple[int, int], float],
    hold_probabilities: Mapping[int, float],
) -> OptionFlowOutput:
    """Propagate one unit of probability mass through a reconvergent DAG."""

    validate_option_dag(nodes, edges)
    node_map = {node.node_id: node for node in nodes}
    if root_id not in node_map:
        raise ValueError("root_id must reference an option node")
    if node_map[root_id].depth != min(node.depth for node in nodes):
        raise ValueError("root node must have minimum depth")

    children: dict[int, list[int]] = {node.node_id: [] for node in nodes}
    for edge in edges:
        children[edge.source].append(edge.target)
    flows = {node.node_id: 0.0 for node in nodes}
    flows[root_id] = 1.0
    held = 0.0
    action_mass: dict[int, float] = {}

    for node in sorted(nodes, key=lambda item: (item.depth, item.node_id)):
        flow = flows[node.node_id]
        outgoing = children[node.node_id]
        if node.kind == "leaf":
            if outgoing:
                raise ValueError("leaf nodes cannot have outgoing edges")
            label = int(node.action_label)
            action_mass[label] = action_mass.get(label, 0.0) + flow
            continue
        if not outgoing:
            raise ValueError("non-leaf option nodes require outgoing edges")
        hold = float(hold_probabilities.get(node.node_id, 0.0))
        probabilities = [
            float(edge_probabilities.get((node.node_id, target), -1.0))
            for target in outgoing
        ]
        if not math.isfinite(hold) or any(not math.isfinite(value) for value in probabilities):
            raise ValueError("routing probabilities must be finite")
        if hold < 0.0 or any(value < 0.0 for value in probabilities):
            raise ValueError("routing probabilities must be nonnegative")
        if not math.isclose(hold + sum(probabilities), 1.0, abs_tol=1e-12):
            raise ValueError("each option node must distribute exactly one unit of local mass")
        held += flow * hold
        for target, probability in zip(outgoing, probabilities, strict=True):
            flows[target] += flow * probability

    total = held + sum(action_mass.values())
    return OptionFlowOutput(
        action_probabilities=tuple(sorted(action_mass.items())),
        hold_probability=held,
        node_flows=tuple(sorted(flows.items())),
        normalization_error=abs(total - 1.0),
        evaluated_nodes=len(nodes),
        evaluated_edges=len(edges),
    )


def edge_responsibilities(
    nodes: Sequence[OptionNode],
    edges: Sequence[OptionEdge],
    *,
    chosen_action: int,
    flow: OptionFlowOutput,
    edge_probabilities: Mapping[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    """Return posterior edge-use probabilities conditioned on a realized action."""

    validate_option_dag(nodes, edges)
    action_probability = flow.probability_of(chosen_action)
    if action_probability <= 0.0:
        raise ValueError("chosen action must have positive routed probability")
    node_flows = dict(flow.node_flows)
    children: dict[int, list[int]] = {node.node_id: [] for node in nodes}
    for edge in edges:
        children[edge.source].append(edge.target)
    backward: dict[int, float] = {}
    for node in sorted(nodes, key=lambda item: (item.depth, item.node_id), reverse=True):
        if node.kind == "leaf":
            backward[node.node_id] = float(node.action_label == chosen_action)
        else:
            backward[node.node_id] = sum(
                float(edge_probabilities[(node.node_id, target)]) * backward[target]
                for target in children[node.node_id]
            )
    return {
        (edge.source, edge.target): (
            node_flows[edge.source]
            * float(edge_probabilities[(edge.source, edge.target)])
            * backward[edge.target]
            / action_probability
        )
        for edge in edges
    }


__all__ = [
    "HoldGateConfig",
    "HoldGateOutput",
    "OptionEdge",
    "OptionFlowOutput",
    "OptionNode",
    "edge_responsibilities",
    "gpi_hold_gate",
    "route_option_flow",
    "validate_option_dag",
]
