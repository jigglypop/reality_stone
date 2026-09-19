"""Minimal prefrontal executive state for hidden-rule maintenance and switch."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExecutiveConfig:
    rule_count: int = 3
    action_count: int = 4
    base_hazard: float = 0.02
    surprise_hazard: float = 0.35
    surprise_threshold: float = 0.20
    feedback_error: float = 0.05

    def __post_init__(self) -> None:
        if self.rule_count < 2 or self.action_count < 2:
            raise ValueError("rule_count and action_count must be at least two")
        for name in ("base_hazard", "surprise_hazard", "surprise_threshold", "feedback_error"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True)
class ExecutiveUpdate:
    predictive_probability: float
    surprise: float
    hazard_used: float
    switched_attention: bool
    feedback_observed: bool


class ExecutiveRuleController:
    """Categorical rule belief with surprise-triggered flexibility."""

    def __init__(self, config: ExecutiveConfig | None = None) -> None:
        self.config = config or ExecutiveConfig()
        self.belief = torch.full((self.config.rule_count,), 1.0 / self.config.rule_count)
        self._next_hazard = self.config.base_hazard
        self.update_count = 0
        self.switch_release_count = 0

    def _prior(self) -> tuple[torch.Tensor, float]:
        hazard = self._next_hazard
        uniform = torch.full_like(self.belief, 1.0 / self.config.rule_count)
        prior = (1.0 - hazard) * self.belief + hazard * uniform
        self._next_hazard = self.config.base_hazard
        return prior, hazard

    def choose(self, features: tuple[int, ...]) -> int:
        if len(features) != self.config.rule_count:
            raise ValueError("features must contain one action per rule")
        scores = torch.zeros(self.config.action_count)
        for rule, action in enumerate(features):
            if not 0 <= int(action) < self.config.action_count:
                raise ValueError("feature action is outside action space")
            scores[int(action)] += self.belief[rule]
        return int(scores.argmax().item())

    def update(self, features: tuple[int, ...], action: int, feedback: bool | None) -> ExecutiveUpdate:
        prior, hazard = self._prior()
        if feedback is None:
            self.belief = prior / prior.sum()
            return ExecutiveUpdate(1.0, 0.0, hazard, False, False)
        likelihood = torch.empty(self.config.rule_count)
        for rule, expected_action in enumerate(features):
            predicts_success = int(action) == int(expected_action)
            matches = bool(feedback) == predicts_success
            likelihood[rule] = 1.0 - self.config.feedback_error if matches else self.config.feedback_error
        predictive = float((prior * likelihood).sum().clamp(min=1e-12).item())
        posterior = prior * likelihood
        if posterior.sum().item() <= 0.0 or not torch.isfinite(posterior).all():
            posterior = torch.full_like(prior, 1.0 / self.config.rule_count)
        else:
            posterior = posterior / posterior.sum()
        surprise = -math.log(max(predictive, 1e-12))
        released = predictive < self.config.surprise_threshold
        if released:
            self._next_hazard = self.config.surprise_hazard
            self.switch_release_count += 1
        self.belief = posterior
        self.update_count += 1
        return ExecutiveUpdate(predictive, surprise, hazard, released, True)

    def reset_goal(self) -> None:
        self.belief.fill_(1.0 / self.config.rule_count)
        self._next_hazard = self.config.base_hazard

    def simplex_valid(self) -> bool:
        return bool(
            torch.isfinite(self.belief).all()
            and (self.belief >= 0.0).all()
            and abs(float(self.belief.sum().item()) - 1.0) <= 1e-6
        )


class ActiveExecutiveController(ExecutiveRuleController):
    """Executive controller that trades immediate reward for rule information."""

    def __init__(
        self,
        config: ExecutiveConfig | None = None,
        *,
        reward_weight: float = 1.0,
        information_weight: float = 0.25,
    ) -> None:
        super().__init__(config or ExecutiveConfig(surprise_hazard=0.02))
        self.reward_weight = float(reward_weight)
        self.information_weight = float(information_weight)
        if self.reward_weight < 0.0 or self.information_weight < 0.0:
            raise ValueError("active executive weights must be non-negative")

    @staticmethod
    def _entropy(probabilities: torch.Tensor) -> float:
        positive = probabilities.clamp(min=1e-12)
        return float(-(positive * positive.log()).sum().item())

    def _information_gain(self, features: tuple[int, ...], action: int) -> float:
        error = self.config.feedback_error
        success_likelihood = torch.tensor([
            1.0 - error if int(action) == int(expected) else error
            for expected in features
        ])
        p_success = float((self.belief * success_likelihood).sum().clamp(1e-12, 1.0 - 1e-12).item())
        success_post = self.belief * success_likelihood
        success_post = success_post / success_post.sum().clamp(min=1e-12)
        failure_likelihood = 1.0 - success_likelihood
        failure_post = self.belief * failure_likelihood
        failure_post = failure_post / failure_post.sum().clamp(min=1e-12)
        expected_entropy = (
            p_success * self._entropy(success_post)
            + (1.0 - p_success) * self._entropy(failure_post)
        )
        return max(self._entropy(self.belief) - expected_entropy, 0.0)

    def choose(self, features: tuple[int, ...]) -> int:
        if len(features) != self.config.rule_count:
            raise ValueError("features must contain one action per rule")
        best_action = 0
        best_score = float("-inf")
        for action in range(self.config.action_count):
            immediate = sum(
                float(self.belief[rule].item())
                for rule, expected in enumerate(features)
                if int(expected) == action
            )
            information = self._information_gain(features, action)
            score = self.reward_weight * immediate + self.information_weight * information
            if score > best_score + 1e-12:
                best_score = score
                best_action = action
        return best_action


__all__ = ["ActiveExecutiveController", "ExecutiveConfig", "ExecutiveRuleController", "ExecutiveUpdate"]
