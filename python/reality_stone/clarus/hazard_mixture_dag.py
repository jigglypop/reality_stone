"""Finite joint hazard-by-context filter for the recurrent DAG benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .recurrent_decision_dag import (
    RecurrentDagConfig,
    _sigmoid,
    _softmax,
    context_action_topology,
    validate_topology,
)


@dataclass(frozen=True)
class HazardMixtureConfig:
    base: RecurrentDagConfig = RecurrentDagConfig(
        soft_content=True,
        strict_causal_order=True,
    )
    hazards: tuple[float, ...] = (0.0, 0.03, 0.06, 0.12, 0.24)
    freeze_model_weights: bool = False

    def __post_init__(self) -> None:
        if not self.base.soft_content or not self.base.strict_causal_order:
            raise ValueError("hazard mixture requires soft content and strict causal order")
        if not self.hazards or len(set(self.hazards)) != len(self.hazards):
            raise ValueError("hazards must be a nonempty unique tuple")
        if any(not math.isfinite(value) or not 0.0 <= value < 1.0 for value in self.hazards):
            raise ValueError("hazards must lie in [0, 1)")
        if not isinstance(self.freeze_model_weights, bool):
            raise TypeError("freeze_model_weights must be boolean")


@dataclass(frozen=True)
class HazardMixtureOutput:
    action: int
    probabilities: tuple[float, ...]
    raw_action_probabilities: tuple[float, ...]
    context_probabilities: tuple[float, ...]
    hazard_weights: tuple[float, ...]
    expected_hazard: float
    action_support_by_context: tuple[tuple[float, ...], ...]
    joint_sum_error: float
    action_mixture_residual: float


@dataclass(frozen=True)
class HazardMixtureCommitResult:
    outcome: int
    evidence: float
    joint_sum_error: float
    minimum_joint_mass: float
    hazard_weights: tuple[float, ...]
    expected_hazard: float
    effective_model_count: float
    outcome_bayes_residual: float
    degenerate_evidence: bool


@dataclass(frozen=True)
class _Pending:
    output: HazardMixtureOutput
    cue_joint: np.ndarray
    support: np.ndarray


class HazardMixtureDecisionDag:
    """Exact joint filtering under a declared finite pseudo-likelihood model."""

    def __init__(self, config: HazardMixtureConfig = HazardMixtureConfig()) -> None:
        self.config = config
        self.nodes, self.edges = context_action_topology(config.base)
        validate_topology(self.nodes, self.edges)
        shape = (len(config.hazards), len(config.base.context_masks))
        self._joint = np.full(shape, 1.0 / (shape[0] * shape[1]), dtype=np.float64)
        self._pending: _Pending | None = None
        self.commit_count = 0
        self.nonfinite_count = 0

    def reset(self) -> None:
        self._joint.fill(1.0 / self._joint.size)
        self._pending = None
        self.commit_count = 0
        self.nonfinite_count = 0

    @property
    def joint_posterior(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(value) for value in row) for row in self._joint)

    def _predict(self) -> np.ndarray:
        count = self._joint.shape[1]
        predicted = np.empty_like(self._joint)
        for model, hazard in enumerate(self.config.hazards):
            row = self._joint[model]
            predicted[model] = (
                (1.0 - hazard) * row
                + hazard * (float(np.sum(row)) - row) / (count - 1)
            )
        return predicted

    def _soft_support(self, content: np.ndarray) -> np.ndarray:
        base = self.config.base
        bit_probabilities = _sigmoid(content / base.content_temperature)
        base_probabilities = np.ones(base.action_count, dtype=np.float64)
        for candidate in range(base.action_count):
            for bit, probability in enumerate(bit_probabilities):
                base_probabilities[candidate] *= (
                    probability if candidate & (1 << bit) else 1.0 - probability
                )
        support = np.zeros(
            (len(base.context_masks), base.action_count), dtype=np.float64
        )
        for context, mask in enumerate(base.context_masks):
            for candidate, probability in enumerate(base_probabilities):
                support[context, candidate ^ mask] += probability
        return support

    def forward_step(
        self,
        content_evidence: Sequence[float],
        cue_log_likelihoods: Sequence[float],
    ) -> HazardMixtureOutput:
        if self._pending is not None:
            raise RuntimeError("a pending mixture decision requires outcome feedback")
        base = self.config.base
        content = np.asarray(content_evidence, dtype=np.float64)
        cues = np.asarray(cue_log_likelihoods, dtype=np.float64)
        if content.shape != (int(math.log2(base.action_count)),):
            raise ValueError("content evidence has the wrong bit dimension")
        if cues.shape != (len(base.context_masks),):
            raise ValueError("cue log likelihoods have the wrong dimension")
        if not np.all(np.isfinite(content)) or not np.all(np.isfinite(cues)):
            raise ValueError("mixture inputs must be finite")

        predicted = self._predict()
        cue_likelihood = np.exp(cues - float(np.max(cues)))
        unnormalized = predicted * cue_likelihood[None, :]
        if self.config.freeze_model_weights:
            conditional = unnormalized / np.sum(unnormalized, axis=1, keepdims=True)
            cue_joint = conditional / len(self.config.hazards)
        else:
            evidence = float(np.sum(unnormalized))
            if evidence <= 0.0 or not math.isfinite(evidence):
                raise FloatingPointError("cue has zero or nonfinite mixture evidence")
            cue_joint = unnormalized / evidence
        context = np.sum(cue_joint, axis=0)
        support = self._soft_support(content)
        proposal = context @ support
        promotion = np.log(proposal)
        inhibition = np.zeros_like(promotion)
        for action in range(base.action_count):
            competitors = np.delete(promotion, action)
            maximum = float(np.max(competitors))
            inhibition[action] = base.inhibition_strength * (
                maximum + math.log(float(np.mean(np.exp(competitors - maximum))))
            )
        probabilities = _softmax(
            promotion - inhibition,
            base.policy_temperature,
        )
        model_weights = np.sum(cue_joint, axis=1)
        per_model_context = np.divide(
            cue_joint,
            model_weights[:, None],
            out=np.zeros_like(cue_joint),
            where=model_weights[:, None] > 0.0,
        )
        per_model_action = per_model_context @ support
        action_mixture = model_weights @ per_model_action
        output = HazardMixtureOutput(
            action=int(np.argmax(probabilities)),
            probabilities=tuple(float(value) for value in probabilities),
            raw_action_probabilities=tuple(float(value) for value in proposal),
            context_probabilities=tuple(float(value) for value in context),
            hazard_weights=tuple(float(value) for value in model_weights),
            expected_hazard=float(np.dot(model_weights, self.config.hazards)),
            action_support_by_context=tuple(
                tuple(float(value) for value in row) for row in support
            ),
            joint_sum_error=abs(float(np.sum(cue_joint)) - 1.0),
            action_mixture_residual=float(np.max(np.abs(action_mixture - proposal))),
        )
        self._pending = _Pending(output=output, cue_joint=cue_joint, support=support)
        return output

    def commit_outcome(
        self,
        signed_feedback: float,
        *,
        support_permutation: Sequence[int] | None = None,
        flip_outcome: bool = False,
    ) -> HazardMixtureCommitResult:
        if self._pending is None:
            raise RuntimeError("mixture outcome requires one preceding forward step")
        if signed_feedback not in (-1.0, 1.0):
            raise ValueError("mixture outcome requires feedback exactly -1 or +1")
        support = self._pending.support[:, self._pending.output.action]
        if support_permutation is not None:
            permutation = np.asarray(tuple(support_permutation), dtype=np.int64)
            count = len(self.config.base.context_masks)
            if sorted(permutation.tolist()) != list(range(count)):
                raise ValueError("support permutation must be a full permutation")
            support = support[permutation]
        outcome = int(signed_feedback > 0.0)
        if flip_outcome:
            outcome = 1 - outcome
        likelihood = support if outcome else 1.0 - support
        unnormalized = self._pending.cue_joint * likelihood[None, :]
        evidence = float(np.sum(unnormalized))
        if evidence <= 0.0 or not math.isfinite(evidence):
            raise FloatingPointError("outcome has zero or nonfinite mixture evidence")
        posterior = unnormalized / evidence
        if self.config.freeze_model_weights:
            row_sums = np.sum(posterior, axis=1, keepdims=True)
            conditional = np.divide(
                posterior,
                row_sums,
                out=np.full_like(posterior, 1.0 / posterior.shape[1]),
                where=row_sums > 0.0,
            )
            posterior = conditional / len(self.config.hazards)
        model_weights = np.sum(posterior, axis=1)
        expected_hazard = float(np.dot(model_weights, self.config.hazards))
        effective = 1.0 / float(np.sum(model_weights**2))
        bayes_residual = float(
            np.max(np.abs(posterior * float(np.sum(unnormalized)) - unnormalized))
        )
        self._joint = posterior
        self._pending = None
        self.commit_count += 1
        if not np.all(np.isfinite(self._joint)):
            self.nonfinite_count += 1
            raise FloatingPointError("nonfinite mixture posterior")
        return HazardMixtureCommitResult(
            outcome=outcome,
            evidence=evidence,
            joint_sum_error=abs(float(np.sum(posterior)) - 1.0),
            minimum_joint_mass=float(np.min(posterior)),
            hazard_weights=tuple(float(value) for value in model_weights),
            expected_hazard=expected_hazard,
            effective_model_count=effective,
            outcome_bayes_residual=bayes_residual,
            degenerate_evidence=False,
        )


__all__ = [
    "HazardMixtureCommitResult",
    "HazardMixtureConfig",
    "HazardMixtureDecisionDag",
    "HazardMixtureOutput",
]
