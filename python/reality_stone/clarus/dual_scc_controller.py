"""Causal controller protocol around the certified dual-SCC fixed-point core."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .dual_scc_basal_ganglia import (
    DualSCCBasalGanglia,
    DualSCCConfig,
    DualSCCResult,
    SmallGainCertificate,
    TopologyAudit,
)


@dataclass(frozen=True)
class DualSCCControllerConfig:
    probe_limit: int = 2
    probe_cost: float = 0.08
    feedback_gain: float = 0.24
    slow_memory_gain: float = 0.72
    fast_memory_gain: float = 0.72

    def __post_init__(self) -> None:
        if self.probe_limit < 0:
            raise ValueError("probe_limit must be nonnegative")
        for name, value in (
            ("probe_cost", self.probe_cost),
            ("feedback_gain", self.feedback_gain),
            ("slow_memory_gain", self.slow_memory_gain),
            ("fast_memory_gain", self.fast_memory_gain),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class DecisionToken:
    sequence: int
    trial: int
    kind: str
    action: int | None


@dataclass(frozen=True)
class DualSCCObservation:
    trial: int
    observation_index: int
    result: DualSCCResult


@dataclass(frozen=True)
class DualSCCDecision:
    token: DecisionToken
    action: int | None
    hold_probability: float
    action_probabilities: tuple[float, ...]
    conditional_action_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class PendingFeedback:
    token: DecisionToken
    due_tick: int
    signed_summary: tuple[float, float]


@dataclass(frozen=True)
class ControllerAudit:
    topology: TopologyAudit
    certificate: SmallGainCertificate
    dimensionless_contract: bool


class DualSCCController:
    """Stateful observe/decide/commit protocol with single-use feedback tokens.

    Frozen observation updates use the certified Jacobi map.  Delayed feedback
    is an explicitly separate bounded state update and is not covered by the
    frozen-map contraction theorem.
    """

    def __init__(
        self,
        core: DualSCCBasalGanglia | None = None,
        *,
        config: DualSCCControllerConfig = DualSCCControllerConfig(),
    ) -> None:
        self.core = core or DualSCCBasalGanglia()
        self.config = config
        self._slow_anchor = np.zeros(self.core.slow_size, dtype=np.float64)
        self._fast_anchor = np.zeros(self.core.fast_size, dtype=np.float64)
        self._trial = -1
        self._trial_open = False
        self._observation_index = 0
        self._probe_count = 0
        self._tick = 0
        self._next_sequence = 0
        self._last_observation: DualSCCObservation | None = None
        self._last_decision: DualSCCDecision | None = None
        self._pending: dict[int, PendingFeedback] = {}
        self._consumed: set[int] = set()
        self._total_probe_cost = 0.0

    @property
    def slow_state(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._slow_anchor)

    @property
    def fast_state(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._fast_anchor)

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def total_probe_cost(self) -> float:
        return self._total_probe_cost

    @property
    def pending_feedback_count(self) -> int:
        return len(self._pending)

    def audit(self) -> ControllerAudit:
        return ControllerAudit(
            topology=self.core.topology_audit(),
            certificate=self.core.certificate,
            dimensionless_contract=True,
        )

    def begin_trial(self) -> int:
        if self._trial_open:
            raise RuntimeError("cannot begin a trial before the current trial closes")
        if self._last_decision is not None or self._last_observation is not None:
            raise RuntimeError("stale decision state must be committed before a new trial")
        self._trial += 1
        self._trial_open = True
        self._observation_index = 0
        self._probe_count = 0
        self._fast_anchor.fill(0.0)
        return self._trial

    def observe(
        self,
        slow_drive: Sequence[float],
        fast_drive: Sequence[float],
        *,
        hold_bias_delta: float = 0.0,
    ) -> DualSCCObservation:
        if not self._trial_open:
            raise RuntimeError("begin_trial must precede observe")
        if self._last_decision is not None:
            raise RuntimeError("commit the pending decision before another observation")
        normalized_slow_drive = self.core._vector(  # noqa: SLF001 - shared typed boundary
            slow_drive, self.core.slow_size, "slow_drive"
        )
        # A converged fixed point is independent of its initial condition.  The
        # previous slow state must therefore enter as a delayed, frozen input
        # if it is to be genuine across-trial memory.  This does not change the
        # within-observation small-gain certificate; closed-loop memory over
        # trials is deliberately outside that theorem.
        normalized_fast_drive = self.core._vector(  # noqa: SLF001 - shared typed boundary
            fast_drive, self.core.fast_size, "fast_drive"
        )
        effective_slow_drive = (
            normalized_slow_drive + self.config.slow_memory_gain * self._slow_anchor
        )
        effective_fast_drive = (
            normalized_fast_drive + self.config.fast_memory_gain * self._fast_anchor
        )
        result = self.core.settle(
            effective_slow_drive,
            effective_fast_drive,
            initial_slow=self._slow_anchor,
            initial_fast=self._fast_anchor,
            hold_bias_delta=hold_bias_delta,
        )
        self._slow_anchor = np.asarray(result.slow_state, dtype=np.float64)
        self._fast_anchor = np.asarray(result.fast_state, dtype=np.float64)
        observation = DualSCCObservation(
            trial=self._trial,
            observation_index=self._observation_index,
            result=result,
        )
        self._observation_index += 1
        self._last_observation = observation
        return observation

    def decide(self) -> DualSCCDecision:
        if not self._trial_open or self._last_observation is None:
            raise RuntimeError("a certified observation must precede decide")
        if self._last_decision is not None:
            raise RuntimeError("the previous decision token is still active")
        policy = self._last_observation.result.policy
        action = policy.selected_action
        kind = "probe" if action is None else "action"
        token = DecisionToken(
            sequence=self._next_sequence,
            trial=self._trial,
            kind=kind,
            action=action,
        )
        self._next_sequence += 1
        decision = DualSCCDecision(
            token=token,
            action=action,
            hold_probability=policy.hold_probability,
            action_probabilities=policy.action_probabilities,
            conditional_action_probabilities=policy.conditional_action_probabilities,
        )
        self._last_decision = decision
        return decision

    def commit_probe(self, token: DecisionToken) -> float:
        decision = self._require_active_token(token, kind="probe")
        if decision.action is not None:
            raise RuntimeError("an action decision cannot be committed as a probe")
        if self._probe_count >= self.config.probe_limit:
            raise RuntimeError("probe limit exhausted")
        self._probe_count += 1
        self._total_probe_cost += self.config.probe_cost
        self._last_decision = None
        self._last_observation = None
        return self.config.probe_cost

    def commit_action(
        self,
        token: DecisionToken,
        *,
        feedback_delay: int = 0,
    ) -> PendingFeedback:
        decision = self._require_active_token(token, kind="action")
        if decision.action is None:
            raise RuntimeError("a probe decision cannot be committed as an action")
        if feedback_delay < 0:
            raise ValueError("feedback_delay must be nonnegative")
        sign = 1.0 if decision.action % 2 else -1.0
        confidence = max(decision.conditional_action_probabilities)
        summary = (-sign * confidence, sign * confidence)
        pending = PendingFeedback(
            token=decision.token,
            due_tick=self._tick + int(feedback_delay),
            signed_summary=summary,
        )
        self._pending[token.sequence] = pending
        self._trial_open = False
        self._last_decision = None
        self._last_observation = None
        self._fast_anchor.fill(0.0)
        return pending

    def advance_time(self, steps: int = 1) -> int:
        if steps < 0:
            raise ValueError("time cannot move backwards")
        self._tick += int(steps)
        return self._tick

    def commit_feedback(self, token: DecisionToken, normalized_reward: float) -> None:
        if token.sequence in self._consumed:
            raise RuntimeError("feedback token was already consumed")
        pending = self._pending.get(token.sequence)
        if pending is None or pending.token != token:
            raise RuntimeError("unknown or mismatched feedback token")
        if self._tick < pending.due_tick:
            raise RuntimeError("feedback token arrived before its causal due tick")
        reward = float(normalized_reward)
        if not math.isfinite(reward) or abs(reward) > 1.0:
            raise ValueError("normalized_reward must be finite and lie in [-1, 1]")
        update = self.config.feedback_gain * reward * np.asarray(
            pending.signed_summary, dtype=np.float64
        )
        self._slow_anchor = np.tanh(self._slow_anchor + update)
        if not np.all(np.isfinite(self._slow_anchor)):
            raise FloatingPointError("feedback produced a nonfinite slow state")
        del self._pending[token.sequence]
        self._consumed.add(token.sequence)

    def _require_active_token(self, token: DecisionToken, *, kind: str) -> DualSCCDecision:
        if self._last_decision is None:
            raise RuntimeError("no active decision token")
        if self._last_decision.token != token:
            raise RuntimeError("decision token does not match the active decision")
        if token.kind != kind:
            raise RuntimeError(f"expected a {kind} token")
        return self._last_decision

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": "clarus.dual-scc-controller.snapshot.v1",
            "controller_config": asdict(self.config),
            "core_config": asdict(self.core.config),
            "slow_anchor": self.slow_state,
            "fast_anchor": self.fast_state,
            "trial": self._trial,
            "trial_open": self._trial_open,
            "observation_index": self._observation_index,
            "probe_count": self._probe_count,
            "tick": self._tick,
            "next_sequence": self._next_sequence,
            "last_decision": None
            if self._last_decision is None
            else asdict(self._last_decision),
            "pending": tuple(asdict(item) for _, item in sorted(self._pending.items())),
            "consumed": tuple(sorted(self._consumed)),
            "total_probe_cost": self._total_probe_cost,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("schema") != "clarus.dual-scc-controller.snapshot.v1":
            raise ValueError("unsupported controller snapshot schema")
        if state.get("controller_config") != asdict(self.config):
            raise ValueError("controller snapshot config mismatch")
        if state.get("core_config") != asdict(self.core.config):
            raise ValueError("core snapshot config mismatch")
        self._slow_anchor = self.core._vector(  # noqa: SLF001 - exact snapshot validation
            state["slow_anchor"], self.core.slow_size, "slow_anchor", bounded=True
        )
        self._fast_anchor = self.core._vector(  # noqa: SLF001 - exact snapshot validation
            state["fast_anchor"], self.core.fast_size, "fast_anchor", bounded=True
        )
        self._trial = int(state["trial"])
        self._trial_open = bool(state["trial_open"])
        self._observation_index = int(state["observation_index"])
        self._probe_count = int(state["probe_count"])
        self._tick = int(state["tick"])
        self._next_sequence = int(state["next_sequence"])
        self._last_observation = None
        last = state["last_decision"]
        if last is None:
            self._last_decision = None
        else:
            last_map = dict(last)
            token_map = dict(last_map["token"])
            token = DecisionToken(**token_map)
            self._last_decision = DualSCCDecision(
                token=token,
                action=last_map["action"],
                hold_probability=float(last_map["hold_probability"]),
                action_probabilities=tuple(last_map["action_probabilities"]),
                conditional_action_probabilities=tuple(
                    last_map["conditional_action_probabilities"]
                ),
            )
        self._pending = {}
        for raw in state["pending"]:
            item = dict(raw)
            token = DecisionToken(**dict(item["token"]))
            pending = PendingFeedback(
                token=token,
                due_tick=int(item["due_tick"]),
                signed_summary=tuple(item["signed_summary"]),
            )
            self._pending[token.sequence] = pending
        self._consumed = {int(value) for value in state["consumed"]}
        self._total_probe_cost = float(state["total_probe_cost"])
        if self._last_decision is not None and self._last_decision.token.sequence in self._consumed:
            raise ValueError("active decision token cannot already be consumed")


def default_dual_scc_controller() -> DualSCCController:
    return DualSCCController(DualSCCBasalGanglia(DualSCCConfig()))


__all__ = [
    "ControllerAudit",
    "DecisionToken",
    "DualSCCController",
    "DualSCCControllerConfig",
    "DualSCCDecision",
    "DualSCCObservation",
    "PendingFeedback",
    "default_dual_scc_controller",
]
