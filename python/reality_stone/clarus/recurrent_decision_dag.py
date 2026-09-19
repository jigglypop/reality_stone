"""Finite DAG decision slices with feedback committed across time steps.

This is an engineering abstraction of recurrent action-selection circuits.  It
does not assert that basal-ganglia anatomy is itself acyclic or tree-shaped.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DagNode:
    node_id: int
    depth: int
    kind: str


@dataclass(frozen=True)
class DagEdge:
    source: int
    target: int


@dataclass(frozen=True)
class RecurrentDagConfig:
    action_count: int = 8
    context_masks: tuple[int, ...] = (0, 3, 5, 6)
    state_decay: float = 0.82
    cue_gain: float = 1.0
    feedback_gain: float = 1.35
    policy_temperature: float = 0.35
    inhibition_strength: float = 0.20
    state_norm_cap: float = 6.0
    soft_content: bool = False
    content_temperature: float = 1.0
    strict_causal_order: bool = False

    def __post_init__(self) -> None:
        if self.action_count < 2 or self.action_count & (self.action_count - 1):
            raise ValueError("action_count must be a power of two")
        if not self.context_masks:
            raise ValueError("at least one context mask is required")
        if any(mask < 0 or mask >= self.action_count for mask in self.context_masks):
            raise ValueError("context masks must index the action space")
        if not 0.0 <= self.state_decay < 1.0:
            raise ValueError("state_decay must lie in [0, 1)")
        for name, value in (
            ("cue_gain", self.cue_gain),
            ("feedback_gain", self.feedback_gain),
            ("policy_temperature", self.policy_temperature),
            ("state_norm_cap", self.state_norm_cap),
            ("content_temperature", self.content_temperature),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.inhibition_strength) or self.inhibition_strength < 0.0:
            raise ValueError("inhibition_strength must be finite and nonnegative")
        if not isinstance(self.soft_content, bool) or not isinstance(self.strict_causal_order, bool):
            raise TypeError("soft_content and strict_causal_order must be booleans")


@dataclass(frozen=True)
class RecurrentDagOutput:
    action: int
    probabilities: tuple[float, ...]
    context_probabilities: tuple[float, ...]
    base_action: int
    predicted_actions_by_context: tuple[int, ...]
    action_support_by_context: tuple[tuple[float, ...], ...]
    evaluated_nodes: int
    evaluated_edges: int


@dataclass(frozen=True)
class ContextBoundaryResult:
    mode: str
    confidence: float
    reset_strength: float
    state_norm_before: float
    state_norm_after_labilization: float
    orthogonal_error: float


@dataclass(frozen=True)
class OutcomePosteriorConfig:
    switch_hazard: float = 0.06

    def __post_init__(self) -> None:
        if not math.isfinite(self.switch_hazard) or not 0.0 < self.switch_hazard < 1.0:
            raise ValueError("switch_hazard must lie strictly between zero and one")


@dataclass(frozen=True)
class OutcomePosteriorResult:
    outcome: int
    prior_context_probabilities: tuple[float, ...]
    outcome_likelihoods: tuple[float, ...]
    evidence: float
    posterior_context_probabilities: tuple[float, ...]
    predicted_next_prior: tuple[float, ...]
    posterior_sum_error: float
    bayes_residual: float
    transition_sum_error: float
    degenerate_evidence: bool


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = values / temperature
    scaled = scaled - float(np.max(scaled))
    weights = np.exp(scaled)
    return weights / float(np.sum(weights))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def context_action_topology(config: RecurrentDagConfig) -> tuple[tuple[DagNode, ...], tuple[DagEdge, ...]]:
    nodes = [DagNode(0, 0, "input")]
    context_offset = 1
    action_offset = context_offset + len(config.context_masks)
    nodes.extend(
        DagNode(context_offset + index, 1, "context")
        for index in range(len(config.context_masks))
    )
    nodes.extend(
        DagNode(action_offset + index, 2, "action")
        for index in range(config.action_count)
    )
    edges = [
        DagEdge(0, context_offset + index)
        for index in range(len(config.context_masks))
    ]
    edges.extend(
        DagEdge(context_offset + context, action_offset + action)
        for context in range(len(config.context_masks))
        for action in range(config.action_count)
    )
    return tuple(nodes), tuple(edges)


def validate_topology(nodes: Sequence[DagNode], edges: Sequence[DagEdge]) -> None:
    node_map = {node.node_id: node for node in nodes}
    if len(node_map) != len(nodes):
        raise ValueError("DAG node identifiers must be unique")
    for edge in edges:
        if edge.source not in node_map or edge.target not in node_map:
            raise ValueError("DAG edge references an unknown node")
        source = node_map[edge.source]
        target = node_map[edge.target]
        if source.depth >= target.depth:
            raise ValueError("every within-step edge must increase topological depth")


class RecurrentDecisionDag:
    """Conditional context-to-action DAG with delayed feedback recurrence."""

    def __init__(self, config: RecurrentDagConfig = RecurrentDagConfig()) -> None:
        self.config = config
        self.nodes, self.edges = context_action_topology(config)
        validate_topology(self.nodes, self.edges)
        self._state = np.zeros(len(config.context_masks), dtype=np.float64)
        self._pending: RecurrentDagOutput | None = None
        self.commit_count = 0
        self.nonfinite_count = 0

    @property
    def state(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._state)

    def reset(self) -> None:
        self._state.fill(0.0)
        self._pending = None
        self.commit_count = 0
        self.nonfinite_count = 0

    def forward_step(
        self,
        content_evidence: Sequence[float],
        context_logits: Sequence[float],
    ) -> RecurrentDagOutput:
        if self.config.strict_causal_order and self._pending is not None:
            raise RuntimeError("a pending decision must receive feedback before another forward step")
        bit_count = int(math.log2(self.config.action_count))
        content = np.asarray(content_evidence, dtype=np.float64)
        cues = np.asarray(context_logits, dtype=np.float64)
        if content.shape != (bit_count,):
            raise ValueError("content evidence has the wrong bit dimension")
        if cues.shape != self._state.shape:
            raise ValueError("context logits have the wrong dimension")
        if not np.all(np.isfinite(content)) or not np.all(np.isfinite(cues)):
            raise ValueError("DAG inputs must be finite")

        base_action = sum(int(value >= 0.0) << index for index, value in enumerate(content))
        context_probabilities = _softmax(
            self.config.cue_gain * cues + self._state
        )
        predicted = tuple(base_action ^ mask for mask in self.config.context_masks)
        if self.config.soft_content:
            bit_probabilities = _sigmoid(content / self.config.content_temperature)
            base_probabilities = np.ones(self.config.action_count, dtype=np.float64)
            for candidate in range(self.config.action_count):
                for bit, probability in enumerate(bit_probabilities):
                    base_probabilities[candidate] *= (
                        probability if candidate & (1 << bit) else 1.0 - probability
                    )
        else:
            base_probabilities = np.zeros(self.config.action_count, dtype=np.float64)
            base_probabilities[base_action] = 1.0
        support_by_context = np.zeros(
            (len(self.config.context_masks), self.config.action_count),
            dtype=np.float64,
        )
        for context, mask in enumerate(self.config.context_masks):
            for candidate, probability in enumerate(base_probabilities):
                support_by_context[context, candidate ^ mask] += probability
        proposal = np.full(self.config.action_count, 1e-12, dtype=np.float64)
        if self.config.soft_content:
            proposal.fill(0.0)
        for context in range(len(self.config.context_masks)):
            proposal += context_probabilities[context] * support_by_context[context]
        promotion = np.log(proposal)

        inhibition = np.zeros_like(promotion)
        for action in range(self.config.action_count):
            competitors = np.delete(promotion, action)
            maximum = float(np.max(competitors))
            log_mean_exp = maximum + math.log(
                float(np.mean(np.exp(competitors - maximum)))
            )
            inhibition[action] = self.config.inhibition_strength * log_mean_exp
        probabilities = _softmax(
            promotion - inhibition,
            self.config.policy_temperature,
        )
        if not np.all(np.isfinite(probabilities)):
            self.nonfinite_count += 1
            raise FloatingPointError("nonfinite DAG policy")
        output = RecurrentDagOutput(
            action=int(np.argmax(probabilities)),
            probabilities=tuple(float(value) for value in probabilities),
            context_probabilities=tuple(float(value) for value in context_probabilities),
            base_action=base_action,
            predicted_actions_by_context=predicted,
            action_support_by_context=tuple(
                tuple(float(value) for value in row) for row in support_by_context
            ),
            evaluated_nodes=len(self.nodes),
            evaluated_edges=len(self.edges),
        )
        self._pending = output
        return output

    def commit_feedback(
        self,
        signed_feedback: float,
        *,
        eligibility_permutation: Sequence[int] | None = None,
        flip_sign: bool = False,
    ) -> None:
        self._commit_feedback(
            signed_feedback,
            eligibility_permutation=eligibility_permutation,
            flip_sign=flip_sign,
            context_boundary_mode="none",
        )

    def commit_feedback_with_context_boundary(
        self,
        signed_feedback: float,
        *,
        mode: str = "surprise_directional",
    ) -> ContextBoundaryResult:
        return self._commit_feedback(
            signed_feedback,
            eligibility_permutation=None,
            flip_sign=False,
            context_boundary_mode=mode,
        )

    def commit_outcome_posterior(
        self,
        signed_feedback: float,
        *,
        config: OutcomePosteriorConfig = OutcomePosteriorConfig(),
        support_permutation: Sequence[int] | None = None,
        flip_outcome: bool = False,
    ) -> OutcomePosteriorResult:
        if self._pending is None:
            raise RuntimeError("outcome posterior requires one preceding forward step")
        if signed_feedback not in (-1.0, 1.0):
            raise ValueError("outcome posterior requires feedback exactly -1 or +1")
        count = len(self.config.context_masks)
        chosen = self._pending.action
        support = np.asarray(
            [row[chosen] for row in self._pending.action_support_by_context],
            dtype=np.float64,
        )
        if support_permutation is not None:
            permutation = np.asarray(tuple(support_permutation), dtype=np.int64)
            if sorted(permutation.tolist()) != list(range(count)):
                raise ValueError("support permutation must be a full permutation")
            support = support[permutation]
        outcome = int(signed_feedback > 0.0)
        if flip_outcome:
            outcome = 1 - outcome
        likelihood = support if outcome else 1.0 - support
        prior = np.asarray(self._pending.context_probabilities, dtype=np.float64)
        joint = prior * likelihood
        evidence = float(np.sum(joint))
        if not math.isfinite(evidence) or evidence <= 0.0:
            raise FloatingPointError("outcome has zero or nonfinite model evidence")
        posterior = joint / evidence
        hazard = config.switch_hazard
        predicted = (1.0 - hazard) * posterior + hazard * (1.0 - posterior) / (count - 1)
        if np.any(predicted <= 0.0) or not np.all(np.isfinite(predicted)):
            raise FloatingPointError("transition prediction left the probability simplex")
        self._state = np.log(predicted)
        self._state -= float(np.mean(self._state))
        if not np.all(np.isfinite(self._state)):
            self.nonfinite_count += 1
            raise FloatingPointError("nonfinite posterior state")
        result = OutcomePosteriorResult(
            outcome=outcome,
            prior_context_probabilities=tuple(float(value) for value in prior),
            outcome_likelihoods=tuple(float(value) for value in likelihood),
            evidence=evidence,
            posterior_context_probabilities=tuple(float(value) for value in posterior),
            predicted_next_prior=tuple(float(value) for value in predicted),
            posterior_sum_error=abs(float(np.sum(posterior)) - 1.0),
            bayes_residual=float(np.max(np.abs(posterior * evidence - joint))),
            transition_sum_error=abs(float(np.sum(predicted)) - 1.0),
            degenerate_evidence=False,
        )
        self.commit_count += 1
        self._pending = None
        return result

    def _commit_feedback(
        self,
        signed_feedback: float,
        *,
        eligibility_permutation: Sequence[int] | None,
        flip_sign: bool,
        context_boundary_mode: str,
    ) -> ContextBoundaryResult:
        if self._pending is None:
            raise RuntimeError("feedback requires one preceding forward step")
        if not math.isfinite(signed_feedback) or abs(signed_feedback) > 1.0:
            raise ValueError("signed feedback must be finite and lie in [-1, 1]")
        feedback = -signed_feedback if flip_sign else signed_feedback
        chosen = self._pending.action
        confidence = self._pending.probabilities[chosen]
        count = len(self.config.context_masks)
        eligibility = np.asarray(
            [row[chosen] for row in self._pending.action_support_by_context],
            dtype=np.float64,
        )
        eligibility -= float(np.mean(eligibility))
        maximum = float(np.max(np.abs(eligibility)))
        if maximum > 0.0:
            eligibility /= maximum
        if eligibility_permutation is not None:
            permutation = np.asarray(tuple(eligibility_permutation), dtype=np.int64)
            if sorted(permutation.tolist()) != list(range(count)):
                raise ValueError("eligibility permutation must be a full permutation")
            eligibility = eligibility[permutation]
        allowed_modes = {
            "none",
            "surprise_directional",
            "negative_directional",
            "generic_forgetting",
            "full_reset",
        }
        if context_boundary_mode not in allowed_modes:
            raise ValueError(f"unknown context-boundary mode: {context_boundary_mode}")
        state_before = self._state.copy()
        reset_strength = 0.0
        orthogonal_error = 0.0
        if context_boundary_mode == "surprise_directional" and feedback < 0.0:
            reset_strength = confidence
        elif context_boundary_mode == "negative_directional" and feedback < 0.0:
            reset_strength = 1.0
        elif context_boundary_mode == "generic_forgetting":
            reset_strength = confidence if feedback < 0.0 else 1.0 - confidence
            self._state *= 1.0 - reset_strength
        elif context_boundary_mode == "full_reset" and feedback < 0.0:
            reset_strength = 1.0
            self._state.fill(0.0)
        if context_boundary_mode in {"surprise_directional", "negative_directional"}:
            squared_norm = float(np.dot(eligibility, eligibility))
            if squared_norm > 0.0 and reset_strength > 0.0:
                alignment = max(float(np.dot(eligibility, self._state)), 0.0) / squared_norm
                self._state -= reset_strength * alignment * eligibility
                before_parallel = float(np.dot(eligibility, state_before)) / squared_norm
                after_parallel = float(np.dot(eligibility, self._state)) / squared_norm
                before_orthogonal = state_before - before_parallel * eligibility
                after_orthogonal = self._state - after_parallel * eligibility
                orthogonal_error = float(np.linalg.norm(after_orthogonal - before_orthogonal))
        state_after_labilization = self._state.copy()
        update = self.config.feedback_gain * feedback * eligibility
        self._state = self.config.state_decay * self._state + update
        self._state -= float(np.mean(self._state))
        norm = float(np.linalg.norm(self._state))
        if norm > self.config.state_norm_cap:
            self._state *= self.config.state_norm_cap / norm
        if not np.all(np.isfinite(self._state)):
            self.nonfinite_count += 1
            raise FloatingPointError("nonfinite recurrent state")
        self.commit_count += 1
        self._pending = None
        return ContextBoundaryResult(
            mode=context_boundary_mode,
            confidence=float(confidence),
            reset_strength=float(reset_strength),
            state_norm_before=float(np.linalg.norm(state_before)),
            state_norm_after_labilization=float(np.linalg.norm(state_after_labilization)),
            orthogonal_error=orthogonal_error,
        )


__all__ = [
    "DagEdge",
    "DagNode",
    "ContextBoundaryResult",
    "RecurrentDagConfig",
    "RecurrentDagOutput",
    "RecurrentDecisionDag",
    "OutcomePosteriorConfig",
    "OutcomePosteriorResult",
    "context_action_topology",
    "validate_topology",
]
