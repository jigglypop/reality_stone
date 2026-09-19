"""Signed temporal credit and separately signed homeostatic control."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TemporalCreditConfig:
    state_count: int
    action_count: int
    discount: float = 0.95
    trace_decay: float = 0.8
    learning_rate: float = 0.12

    def __post_init__(self) -> None:
        if self.state_count < 1 or self.action_count < 1:
            raise ValueError("state_count and action_count must be positive")
        if not 0.0 <= self.discount <= 1.0 or not 0.0 <= self.trace_decay <= 1.0:
            raise ValueError("discount and trace_decay must lie in [0, 1]")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")


class EligibilityQLearner:
    """Small tabular TD(lambda) probe for delayed signed credit."""

    def __init__(self, config: TemporalCreditConfig, *, absolute_td: bool = False) -> None:
        self.config = config
        self.absolute_td = bool(absolute_td)
        self.q = torch.zeros(config.state_count, config.action_count)
        self.eligibility = torch.zeros_like(self.q)

    def start_episode(self) -> None:
        self.eligibility.zero_()

    def decay_eligibility(self) -> None:
        self.eligibility.mul_(self.config.discount * self.config.trace_decay)

    def mark_eligibility(self, state: int, action: int) -> None:
        if not 0 <= state < self.config.state_count or not 0 <= action < self.config.action_count:
            raise ValueError("state or action out of bounds")
        self.decay_eligibility()
        self.eligibility[state, action] += 1.0

    def apply_credit(self, signal: float) -> None:
        signed = abs(float(signal)) if self.absolute_td else float(signal)
        self.q.add_(self.config.learning_rate * signed * self.eligibility)
        if not torch.isfinite(self.q).all():
            raise FloatingPointError("non-finite temporal-credit state")

    def act(self, state: int, *, rng: random.Random | None = None, epsilon: float = 0.0) -> int:
        if not 0 <= state < self.config.state_count:
            raise ValueError("state out of bounds")
        if rng is not None and rng.random() < epsilon:
            return rng.randrange(self.config.action_count)
        values = self.q[state]
        return int(values.argmax().item())

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int | None,
        *,
        done: bool,
    ) -> float:
        if not 0 <= state < self.config.state_count or not 0 <= action < self.config.action_count:
            raise ValueError("state or action out of bounds")
        if not done and (next_state is None or not 0 <= next_state < self.config.state_count):
            raise ValueError("non-terminal update requires a valid next_state")
        bootstrap = 0.0 if done else self.config.discount * float(self.q[next_state].max().item())
        delta = float(reward) + bootstrap - float(self.q[state, action].item())
        self.mark_eligibility(state, action)
        self.apply_credit(delta)
        return delta


@dataclass(frozen=True)
class HomeostaticConfig:
    unit_count: int
    target_rate: float
    averaging_rate: float = 0.02
    threshold_learning_rate: float = 0.01

    def __post_init__(self) -> None:
        if self.unit_count < 1 or not 0.0 < self.target_rate < 1.0:
            raise ValueError("invalid unit_count or target_rate")
        if not 0.0 < self.averaging_rate <= 1.0 or self.threshold_learning_rate <= 0.0:
            raise ValueError("invalid homeostatic rates")


class SignedHomeostaticController:
    """Slow firing-rate controller kept separate from task TD error."""

    def __init__(self, config: HomeostaticConfig) -> None:
        self.config = config
        self.mean_rate = torch.full((config.unit_count,), config.target_rate)
        self.threshold = torch.zeros(config.unit_count)

    def update(self, activity: torch.Tensor) -> torch.Tensor:
        active = activity.detach().float().view(-1)
        if active.numel() != self.config.unit_count or not torch.isfinite(active).all():
            raise ValueError("activity must be a finite unit vector")
        beta = self.config.averaging_rate
        self.mean_rate = (1.0 - beta) * self.mean_rate + beta * active
        signed_error = self.mean_rate - self.config.target_rate
        self.threshold = self.threshold + self.config.threshold_learning_rate * signed_error
        return signed_error.clone()
