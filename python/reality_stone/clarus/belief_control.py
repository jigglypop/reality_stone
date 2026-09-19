"""Action-conditioned rank-1 belief control and short-horizon planning.

This module is deliberately independent from :mod:`clarus.runtime`.  It owns
only an observation-space belief and never mutates a BrainRuntime while
evaluating candidate action sequences.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Callable, Mapping

import torch


@dataclass(frozen=True)
class BeliefControlConfig:
    observation_dim: int
    action_count: int
    horizon: int = 2
    discount: float = 0.95
    latent_pole: float = 0.95
    process_variance: float = 0.01
    observation_variance: float = 0.05
    robust_threshold: float = 3.0
    action_effect_lr: float = 0.15
    goal_weight: float = 1.0
    uncertainty_weight: float = 0.05
    action_cost: float = 0.0
    variance_floor: float = 1e-6

    def __post_init__(self) -> None:
        if self.observation_dim < 1 or self.action_count < 1:
            raise ValueError("observation_dim and action_count must be positive")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must lie in [0, 1]")
        if not 0.0 <= self.latent_pole < 1.0:
            raise ValueError("latent_pole must lie in [0, 1)")
        for name in (
            "process_variance", "observation_variance", "robust_threshold",
            "action_effect_lr", "goal_weight", "uncertainty_weight",
            "variance_floor",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.action_cost < 0.0:
            raise ValueError("action_cost must be non-negative")


@dataclass(frozen=True)
class BeliefUpdate:
    innovation_norm: float
    mahalanobis: float
    robust_weight: float
    posterior_mean: float
    posterior_variance: float
    trust: float


@dataclass(frozen=True)
class BeliefPlan:
    action_index: int
    sequence: tuple[int, ...]
    sequence_cost: float
    action_costs: torch.Tensor
    predicted_observation: torch.Tensor
    predicted_variance: torch.Tensor
    trust: float


def _finite_vector(value: torch.Tensor, dim: int, name: str) -> torch.Tensor:
    out = value.detach().float().view(-1)
    if out.numel() != dim:
        raise ValueError(f"{name} must contain {dim} values")
    if not torch.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


class BeliefController:
    """Rank-1 robust observer with learned action effects and pure MPC.

    Learning occurs only when :meth:`observe` receives the observation after a
    committed action.  :meth:`plan` is pure and can be called repeatedly
    without changing controller state.
    """

    SCHEMA = "clarus.belief_control.v1"

    def __init__(
        self,
        config: BeliefControlConfig,
        *,
        loading: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        dim = config.observation_dim
        if loading is None:
            loading = torch.ones(dim, dtype=torch.float32) / math.sqrt(dim)
        self.loading = _finite_vector(loading, dim, "loading").to(self.device)
        norm = self.loading.norm().clamp(min=config.variance_floor)
        self.loading = self.loading / norm
        self.posterior_mean = torch.tensor(0.0, device=self.device)
        self.posterior_variance = torch.tensor(1.0, device=self.device)
        self.observation_variance = torch.full(
            (dim,), config.observation_variance, device=self.device
        )
        self.action_effect = torch.zeros(dim, config.action_count, device=self.device)
        self.action_counts = torch.zeros(config.action_count, dtype=torch.long, device=self.device)
        self.last_observation: torch.Tensor | None = None
        self.last_action: int | None = None
        self.last_base_prediction: torch.Tensor | None = None
        self.update_count = 0
        self.last_robust_weight = 1.0
        self.last_trust = 0.0

    def _predicted_belief(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.config.latent_pole * self.posterior_mean
        variance = (
            self.config.latent_pole ** 2 * self.posterior_variance
            + self.config.process_variance
        ).clamp(min=self.config.variance_floor)
        return mean, variance

    def _trust(self, mean: torch.Tensor, variance: torch.Tensor) -> float:
        correction = self.loading * mean
        signal = correction.square().sum()
        uncertainty = (
            variance * self.loading.square() + self.observation_variance
        ).sum()
        trust = signal / (signal + uncertainty + self.config.variance_floor)
        return float(trust.clamp(0.0, 1.0).item())

    def observe(self, observation: torch.Tensor) -> BeliefUpdate:
        """Assimilate a current observation using only the committed past."""
        obs = _finite_vector(observation, self.config.observation_dim, "observation").to(self.device)
        mean_minus, variance_minus = self._predicted_belief()

        if self.last_observation is None:
            self.last_observation = obs.clone()
            self.posterior_mean = mean_minus
            self.posterior_variance = variance_minus
            self.last_trust = self._trust(mean_minus, variance_minus)
            return BeliefUpdate(0.0, 0.0, 1.0, float(mean_minus), float(variance_minus), self.last_trust)

        if self.last_action is None or self.last_base_prediction is None:
            raise RuntimeError("observe requires a committed action after initialization")

        action = self.last_action
        predicted = self.last_base_prediction + self.action_effect[:, action] + self.loading * mean_minus
        innovation = obs - predicted
        innovation_variance = (
            variance_minus * self.loading.square() + self.observation_variance
        ).clamp(min=self.config.variance_floor)
        mahalanobis = float((innovation.square() / innovation_variance).sum().item())
        normalized_distance = math.sqrt(max(mahalanobis / self.config.observation_dim, 0.0))
        robust_weight = min(
            1.0,
            self.config.robust_threshold / max(normalized_distance, self.config.variance_floor),
        )

        # Scalar-latent diagonal-observation Kalman update.  Robustness acts in
        # the likelihood/update, not as an output interpolation gain.
        precision = (
            self.loading.square() / innovation_variance
        ).sum().clamp(min=self.config.variance_floor)
        gain_scale = variance_minus / (1.0 + variance_minus * precision)
        mean_update = gain_scale * (self.loading * innovation / innovation_variance).sum()
        mean_post = mean_minus + robust_weight * mean_update
        variance_post = (
            (1.0 - robust_weight) * variance_minus
            + robust_weight * gain_scale
        ).clamp(min=self.config.variance_floor)

        # Learn only the selected action column from the causal transition.
        residual_for_action = obs - self.last_base_prediction - self.loading * mean_post
        lr = self.config.action_effect_lr * robust_weight
        self.action_effect[:, action] = (
            self.action_effect[:, action]
            + lr * (residual_for_action - self.action_effect[:, action])
        )
        self.action_counts[action] += 1

        self.posterior_mean = mean_post
        self.posterior_variance = variance_post
        self.last_observation = obs.clone()
        self.last_action = None
        self.last_base_prediction = None
        self.update_count += 1
        self.last_robust_weight = robust_weight
        self.last_trust = self._trust(mean_post, variance_post)
        return BeliefUpdate(
            innovation_norm=float(innovation.norm().item()),
            mahalanobis=mahalanobis,
            robust_weight=robust_weight,
            posterior_mean=float(mean_post.item()),
            posterior_variance=float(variance_post.item()),
            trust=self.last_trust,
        )

    def _rollout_step(
        self,
        observation: torch.Tensor,
        mean: torch.Tensor,
        variance: torch.Tensor,
        action: int,
        action_free_base_transition: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        base = _finite_vector(
            action_free_base_transition(observation),
            self.config.observation_dim,
            "base prediction",
        ).to(self.device)
        next_mean = self.config.latent_pole * mean
        next_variance = (
            self.config.latent_pole ** 2 * variance + self.config.process_variance
        ).clamp(min=self.config.variance_floor)
        trust = self._trust(next_mean, next_variance)
        # Covariance influences the Kalman posterior and the explicit planning
        # risk.  It is not allowed to become a scalar interpolation of a
        # completed forecast (the failed V8 family).
        next_obs = base + self.action_effect[:, action] + self.loading * next_mean
        obs_variance = next_variance * self.loading.square() + self.observation_variance
        return next_obs, next_mean, next_variance, trust

    def plan(
        self,
        observation: torch.Tensor,
        task_goal: torch.Tensor,
        *,
        action_free_base_transition: Callable[[torch.Tensor], torch.Tensor],
        action_mask: torch.Tensor | None = None,
    ) -> BeliefPlan:
        """Choose the first action of a deterministic finite-horizon rollout."""
        obs = _finite_vector(observation, self.config.observation_dim, "observation").to(self.device)
        goal = _finite_vector(task_goal, self.config.observation_dim, "task_goal").to(self.device)
        if action_mask is None:
            legal = tuple(range(self.config.action_count))
        else:
            mask = action_mask.detach().bool().view(-1)
            if mask.numel() != self.config.action_count or not mask.any():
                raise ValueError("action_mask must contain at least one legal action")
            legal = tuple(int(i) for i in torch.nonzero(mask, as_tuple=False).view(-1).tolist())

        costs = torch.full((self.config.action_count,), float("inf"))
        best_cost = float("inf")
        best_sequence: tuple[int, ...] | None = None
        best_observation = obs
        best_variance = self.posterior_variance * self.loading.square() + self.observation_variance
        best_trust = self.last_trust

        for sequence in itertools.product(legal, repeat=self.config.horizon):
            rollout_obs = obs.clone()
            rollout_mean = self.posterior_mean.clone()
            rollout_variance = self.posterior_variance.clone()
            total = 0.0
            trust = self.last_trust
            obs_variance = best_variance
            for depth, action in enumerate(sequence):
                rollout_obs, rollout_mean, rollout_variance, trust = self._rollout_step(
                    rollout_obs,
                    rollout_mean,
                    rollout_variance,
                    action,
                    action_free_base_transition,
                )
                obs_variance = (
                    rollout_variance * self.loading.square() + self.observation_variance
                )
                stage_cost = (
                    self.config.goal_weight * (rollout_obs - goal).square().mean()
                    + self.config.uncertainty_weight * obs_variance.mean()
                    + self.config.action_cost
                )
                total += (self.config.discount ** depth) * float(stage_cost.item())
            first = sequence[0]
            costs[first] = min(costs[first], total)
            if total < best_cost - 1e-12 or (
                abs(total - best_cost) <= 1e-12 and (best_sequence is None or sequence < best_sequence)
            ):
                best_cost = total
                best_sequence = sequence
                best_observation = rollout_obs.clone()
                best_variance = obs_variance.clone()
                best_trust = trust

        if best_sequence is None:
            raise RuntimeError("planner found no legal action sequence")
        return BeliefPlan(
            action_index=best_sequence[0],
            sequence=best_sequence,
            sequence_cost=best_cost,
            action_costs=costs,
            predicted_observation=best_observation,
            predicted_variance=best_variance,
            trust=best_trust,
        )

    def commit(
        self,
        plan: BeliefPlan,
        *,
        base_prediction: torch.Tensor,
    ) -> None:
        """Commit one real action; its effect is learned at the next observe."""
        if self.last_observation is None:
            raise RuntimeError("observe an initial state before commit")
        if self.last_action is not None:
            raise RuntimeError("cannot commit twice without a new observation")
        if not 0 <= plan.action_index < self.config.action_count:
            raise ValueError("plan action is out of bounds")
        self.last_action = int(plan.action_index)
        self.last_base_prediction = _finite_vector(
            base_prediction, self.config.observation_dim, "base_prediction"
        ).to(self.device).clone()

    def state_dict(self) -> dict[str, object]:
        def cpu_clone(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.detach().cpu().clone()

        return {
            "schema": self.SCHEMA,
            "config": asdict(self.config),
            "loading": cpu_clone(self.loading),
            "posterior_mean": cpu_clone(self.posterior_mean),
            "posterior_variance": cpu_clone(self.posterior_variance),
            "observation_variance": cpu_clone(self.observation_variance),
            "action_effect": cpu_clone(self.action_effect),
            "action_counts": cpu_clone(self.action_counts),
            "last_observation": cpu_clone(self.last_observation),
            "last_action": self.last_action,
            "last_base_prediction": cpu_clone(self.last_base_prediction),
            "update_count": self.update_count,
            "last_robust_weight": self.last_robust_weight,
            "last_trust": self.last_trust,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("schema") != self.SCHEMA:
            raise ValueError("unsupported belief controller schema")
        saved_config = state.get("config")
        if not isinstance(saved_config, Mapping):
            raise ValueError("missing belief controller config")
        if int(saved_config.get("observation_dim", -1)) != self.config.observation_dim:
            raise ValueError("observation dimension mismatch")
        if int(saved_config.get("action_count", -1)) != self.config.action_count:
            raise ValueError("action count mismatch")

        def tensor(name: str, shape: tuple[int, ...]) -> torch.Tensor:
            value = state.get(name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"invalid {name}")
            out = value.detach().float().to(self.device).clone()
            if not torch.isfinite(out).all():
                raise ValueError(f"non-finite {name}")
            return out

        dim, actions = self.config.observation_dim, self.config.action_count
        self.loading = tensor("loading", (dim,))
        self.posterior_mean = tensor("posterior_mean", ())
        self.posterior_variance = tensor("posterior_variance", ())
        self.observation_variance = tensor("observation_variance", (dim,))
        self.action_effect = tensor("action_effect", (dim, actions))
        counts = state.get("action_counts")
        if not isinstance(counts, torch.Tensor) or tuple(counts.shape) != (actions,):
            raise ValueError("invalid action_counts")
        self.action_counts = counts.detach().long().to(self.device).clone()
        if self.posterior_variance.item() < 0 or (self.observation_variance < 0).any():
            raise ValueError("variances must be non-negative")
        last_obs = state.get("last_observation")
        self.last_observation = None if last_obs is None else _finite_vector(last_obs, dim, "last_observation").to(self.device).clone()
        last_base = state.get("last_base_prediction")
        self.last_base_prediction = None if last_base is None else _finite_vector(last_base, dim, "last_base_prediction").to(self.device).clone()
        last_action = state.get("last_action")
        if last_action is not None and not 0 <= int(last_action) < actions:
            raise ValueError("invalid last_action")
        self.last_action = None if last_action is None else int(last_action)
        self.update_count = int(state.get("update_count", 0))
        self.last_robust_weight = float(state.get("last_robust_weight", 1.0))
        self.last_trust = float(state.get("last_trust", 0.0))
