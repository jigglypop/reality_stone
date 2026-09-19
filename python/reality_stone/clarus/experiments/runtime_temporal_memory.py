"""Opt-in temporal-memory controller composed around :class:`RuntimeAgent`.

The wrapper keeps the existing RuntimeAgent untouched. Every call advances the
base agent; only a routed fact query may override the outward action with the
configured answer or abstain action. The default is disabled and therefore
preserves the legacy action path exactly.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch
import torch.nn.functional as F

from ..temporal_memory import RecallMode, TemporalAuditedMemory, TemporalMemoryRecall


Route = Literal["disabled", "bypass", "context", "memory"]
TaskKind = Literal["fact", "context", "control"]


@dataclass(frozen=True, slots=True)
class TemporalAgentQuery:
    query_id: str
    kind: TaskKind
    subject: str
    relation: str | None = None
    mode: RecallMode = "current"
    as_of_session: int | None = None
    path: tuple[str, ...] = ()
    context_value: str | None = None
    context_evidence_id: str | None = None


@dataclass(frozen=True, slots=True)
class TemporalAgentDecision:
    route: Route
    value: str | None
    evidence_id: str | None
    valid_session: int | None
    abstained: bool
    recall_calls: int
    recall_cost: int


@dataclass(frozen=True, slots=True)
class RuntimeTemporalAgentStep:
    base_step: Any
    action_index: int
    value: str | None
    evidence_id: str | None
    decision: TemporalAgentDecision
    memory_context_norm: float


class RuntimeAgentLike(Protocol):
    runtime: Any
    action_embeddings: torch.Tensor

    def step(self, **kwargs: Any) -> Any: ...


@dataclass
class TemporalMemoryController:
    memory: TemporalAuditedMemory
    enabled: bool = False
    prefer_context: bool = True

    @staticmethod
    def _decision_from_recall(
        recall: TemporalMemoryRecall,
        *,
        calls: int,
        cost: int,
    ) -> TemporalAgentDecision:
        return TemporalAgentDecision(
            route="memory",
            value=recall.value,
            evidence_id=recall.evidence_id,
            valid_session=recall.valid_session,
            abstained=bool(recall.abstained or recall.value is None),
            recall_calls=calls,
            recall_cost=cost,
        )

    def decide(self, query: TemporalAgentQuery) -> TemporalAgentDecision:
        if not self.enabled:
            return TemporalAgentDecision("disabled", None, None, None, False, 0, 0)
        if query.kind == "control":
            return TemporalAgentDecision("bypass", None, None, None, False, 0, 0)
        if query.context_value is not None and self.prefer_context:
            return TemporalAgentDecision(
                "context",
                query.context_value,
                query.context_evidence_id,
                None,
                False,
                0,
                0,
            )
        if query.path:
            current = query.subject
            calls = 0
            total_cost = 0
            last: TemporalMemoryRecall | None = None
            for relation in query.path:
                last = self.memory.recall(current, relation)
                calls += 1
                total_cost += last.cost
                if last.abstained or last.value is None:
                    return self._decision_from_recall(
                        last,
                        calls=calls,
                        cost=total_cost,
                    )
                current = last.value
            if last is None:
                return TemporalAgentDecision("memory", None, None, None, True, 0, 0)
            return TemporalAgentDecision(
                "memory",
                current,
                last.evidence_id,
                last.valid_session,
                False,
                calls,
                total_cost,
            )
        if query.relation is None:
            raise ValueError("relation is required when path is empty")
        recall = self.memory.recall(
            query.subject,
            query.relation,
            mode=query.mode,
            as_of_session=query.as_of_session,
        )
        return self._decision_from_recall(recall, calls=1, cost=recall.cost)


def _encode_context(decision: TemporalAgentDecision, dim: int, device: torch.device) -> torch.Tensor:
    if decision.route in {"disabled", "bypass"}:
        return torch.zeros(dim, device=device)
    payload = "|".join(
        [
            decision.route,
            decision.value or "<ABSTAIN>",
            decision.evidence_id or "<NO-EVIDENCE>",
            str(decision.valid_session or 0),
        ]
    )
    vector = torch.zeros(dim, dtype=torch.float32, device=device)
    for position, term in enumerate(payload.split("|")):
        digest = hashlib.blake2b(
            f"{position}:{term}".encode("utf-8"),
            digest_size=8,
        ).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign / math.sqrt(position + 1.0)
    return F.normalize(vector, dim=0) if vector.norm().item() > 0.0 else vector


class RuntimeTemporalAgent:
    def __init__(
        self,
        runtime_agent: RuntimeAgentLike,
        *,
        controller: TemporalMemoryController,
        answer_action_index: int = 0,
        abstain_action_index: int = 1,
        memory_context_gain: float = 0.25,
    ) -> None:
        self.runtime_agent = runtime_agent
        self.controller = controller
        self.answer_action_index = int(answer_action_index)
        self.abstain_action_index = int(abstain_action_index)
        self.memory_context_gain = float(memory_context_gain)
        action_count = int(runtime_agent.action_embeddings.shape[0])
        if not 0 <= self.answer_action_index < action_count:
            raise ValueError("answer_action_index is outside the action space")
        if not 0 <= self.abstain_action_index < action_count:
            raise ValueError("abstain_action_index is outside the action space")
        if self.answer_action_index == self.abstain_action_index:
            raise ValueError("answer and abstain actions must differ")
        if not math.isfinite(self.memory_context_gain) or self.memory_context_gain < 0.0:
            raise ValueError("memory_context_gain must be finite and non-negative")

    def ingest(self, event: Any) -> str:
        return self.controller.memory.ingest(event)

    def step(
        self,
        *,
        query: TemporalAgentQuery,
        external_input: torch.Tensor | None = None,
        observation: torch.Tensor | None = None,
        **runtime_kwargs: Any,
    ) -> RuntimeTemporalAgentStep:
        decision = self.controller.decide(query)
        runtime = self.runtime_agent.runtime
        dim = int(runtime.config.dim)
        device = runtime.device
        base_observation = observation
        if base_observation is None:
            base_observation = external_input
        if base_observation is None:
            base_observation = torch.zeros(dim, device=device)
        base_observation = base_observation.detach().float().to(device).view(-1)
        if base_observation.numel() != dim or not torch.isfinite(base_observation).all():
            raise ValueError("observation must be finite and match runtime dim")

        memory_context = _encode_context(decision, dim, device)
        fused = base_observation + self.memory_context_gain * memory_context
        if fused.norm().item() > 0.0:
            fused = F.normalize(fused, dim=0)
        base_step = self.runtime_agent.step(
            external_input=fused,
            observation=fused,
            **runtime_kwargs,
        )

        if decision.route in {"disabled", "bypass"}:
            action_index = int(base_step.action_index)
            value = None
            evidence_id = None
        elif decision.abstained or decision.value is None:
            action_index = self.abstain_action_index
            value = None
            evidence_id = None
        else:
            action_index = self.answer_action_index
            value = decision.value
            evidence_id = decision.evidence_id
        return RuntimeTemporalAgentStep(
            base_step=base_step,
            action_index=action_index,
            value=value,
            evidence_id=evidence_id,
            decision=decision,
            memory_context_norm=float(memory_context.norm().item()),
        )
