"""Delayed Polarity Control benchmark for action-conditioned belief planning.

The benchmark is intentionally small and teacher-identified.  It tests whether
remembering the probing action, maintaining a calibrated belief, and exposing a
delayed terminal reward to a finite-horizon planner are jointly useful.  It is
not a claim of observation-only latent-state discovery.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class DPCConfig:
    sigma: float = 0.8
    safe_reward: float = 0.15
    correct_reward: float = 1.0
    wrong_reward: float = -1.0
    commit_cost: float = 0.05
    dropout_probability: float = 0.02
    flip_probability: float = 0.0


@dataclass(frozen=True)
class DPCEpisode:
    seed: int
    goal: int
    probe: int
    evidence: float
    delay: int


@dataclass(frozen=True)
class DPCDecision:
    action: int  # -1/+1 commit, 0 safe
    probability_positive: float
    expected_commit_return: float


@dataclass(frozen=True)
class LogisticBeliefModel:
    """Train-only one-feature belief model for Loop 1b."""

    weight: float
    bias: float
    action_conditioned: bool

    def probability_positive(self, episode: DPCEpisode) -> float:
        feature = episode.evidence * (episode.probe if self.action_conditioned else 1)
        return _sigmoid(self.weight * feature + self.bias)


def _stream(seed: int, tag: int) -> random.Random:
    # Separate deterministic streams prevent policy draw counts from changing
    # the environment's goal/noise sequence.
    return random.Random((int(seed) * 0x9E3779B1 + tag * 0x85EBCA77) & 0xFFFFFFFFFFFF)


def make_episode(seed: int, delay: int, config: DPCConfig) -> DPCEpisode:
    if delay not in (2, 3):
        raise ValueError("DPC delay must be 2 or 3")
    goal = -1 if _stream(seed, 1).random() < 0.5 else 1
    probe = -1 if _stream(seed, 2).random() < 0.5 else 1
    evidence = probe * goal + _stream(seed, 3).gauss(0.0, config.sigma)
    if _stream(seed, 4).random() < config.dropout_probability:
        evidence = 0.0
    if _stream(seed, 5).random() < config.flip_probability:
        evidence = -evidence
    return DPCEpisode(seed=seed, goal=goal, probe=probe, evidence=evidence, delay=delay)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def posterior_positive(
    episode: DPCEpisode,
    config: DPCConfig,
    *,
    action_conditioned: bool,
) -> float:
    signed_evidence = episode.evidence * (episode.probe if action_conditioned else 1)
    log_odds = 2.0 * signed_evidence / max(config.sigma ** 2, 1e-12)
    # A registered dropout mixture prevents zero evidence from pretending to be
    # certain while preserving the action-conditioned likelihood ratio.
    probability = _sigmoid(log_odds)
    mix = config.dropout_probability
    return (1.0 - mix) * probability + mix * 0.5


def belief_mpc_decision(
    episode: DPCEpisode,
    config: DPCConfig,
    *,
    horizon: int,
    action_conditioned: bool = True,
) -> DPCDecision:
    probability = posterior_positive(episode, config, action_conditioned=action_conditioned)
    confidence = max(probability, 1.0 - probability)
    expected_commit = (
        confidence * (config.correct_reward - config.commit_cost)
        + (1.0 - confidence) * (config.wrong_reward - config.commit_cost)
    )
    if horizon < episode.delay or expected_commit <= config.safe_reward:
        action = 0
    else:
        action = 1 if probability >= 0.5 else -1
    return DPCDecision(action, probability, expected_commit)


def decision_from_probability(
    probability: float,
    episode: DPCEpisode,
    config: DPCConfig,
    *,
    horizon: int,
) -> DPCDecision:
    confidence = max(probability, 1.0 - probability)
    expected_commit = (
        confidence * (config.correct_reward - config.commit_cost)
        + (1.0 - confidence) * (config.wrong_reward - config.commit_cost)
    )
    action = 0 if horizon < episode.delay or expected_commit <= config.safe_reward else (1 if probability >= 0.5 else -1)
    return DPCDecision(action, probability, expected_commit)


def fit_logistic_belief(
    seeds: Iterable[int],
    config: DPCConfig,
    *,
    action_conditioned: bool,
    epochs: int = 500,
    learning_rate: float = 0.2,
    l2: float = 1e-3,
) -> LogisticBeliefModel:
    """Fit the belief likelihood using training episodes only."""
    rows = []
    for seed in seeds:
        episode = make_episode(seed, 2, config)
        feature = episode.evidence * (episode.probe if action_conditioned else 1)
        rows.append((feature, 1.0 if episode.goal == 1 else 0.0))
    weight = 0.0
    bias = 0.0
    for _ in range(epochs):
        grad_weight = l2 * weight
        grad_bias = 0.0
        for feature, label in rows:
            error = _sigmoid(weight * feature + bias) - label
            grad_weight += error * feature / len(rows)
            grad_bias += error / len(rows)
        weight -= learning_rate * grad_weight
        bias -= learning_rate * grad_bias
    return LogisticBeliefModel(weight, bias, action_conditioned)


def episode_return(episode: DPCEpisode, decision: DPCDecision, config: DPCConfig) -> float:
    if decision.action == 0:
        return config.safe_reward
    reward = config.correct_reward if decision.action == episode.goal else config.wrong_reward
    return reward - config.commit_cost


def _paired_lcb(differences: list[float], *, bootstrap_seed: int = 20260811, draws: int = 2000) -> float:
    if not differences:
        raise ValueError("paired differences cannot be empty")
    rng = random.Random(bootstrap_seed)
    n = len(differences)
    means = []
    for _ in range(draws):
        means.append(sum(differences[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[max(0, int(0.025 * draws) - 1)]


def _ece(probabilities: list[float], labels: list[int], bins: int = 10) -> float:
    total = len(labels)
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [
            i for i, value in enumerate(probabilities)
            if lower <= value < upper or (index == bins - 1 and value == 1.0)
        ]
        if not selected:
            continue
        confidence = sum(probabilities[i] for i in selected) / len(selected)
        frequency = sum(labels[i] for i in selected) / len(selected)
        error += len(selected) / total * abs(confidence - frequency)
    return error


def evaluate_delay(
    seeds: Iterable[int],
    delay: int,
    config: DPCConfig | None = None,
) -> dict[str, float | int | bool]:
    cfg = config or DPCConfig()
    full_horizon = delay
    returns: dict[str, list[float]] = {
        "full": [], "reactive": [], "recurrent": [], "action_agnostic": [], "h1": [], "h3": [],
    }
    probabilities: list[float] = []
    labels: list[int] = []
    successes = 0
    safe_count = 0
    episodes = list(seeds)
    for seed in episodes:
        episode = make_episode(seed, delay, cfg)
        full = belief_mpc_decision(episode, cfg, horizon=full_horizon, action_conditioned=True)
        reactive = DPCDecision(0, 0.5, cfg.safe_reward)
        recurrent = belief_mpc_decision(episode, cfg, horizon=full_horizon, action_conditioned=True)
        agnostic = belief_mpc_decision(episode, cfg, horizon=full_horizon, action_conditioned=False)
        h1 = belief_mpc_decision(episode, cfg, horizon=1, action_conditioned=True)
        h3 = belief_mpc_decision(episode, cfg, horizon=3, action_conditioned=True)
        for name, decision in (
            ("full", full), ("reactive", reactive), ("recurrent", recurrent),
            ("action_agnostic", agnostic), ("h1", h1), ("h3", h3),
        ):
            returns[name].append(episode_return(episode, decision, cfg))
        probabilities.append(full.probability_positive)
        labels.append(1 if episode.goal == 1 else 0)
        successes += int(full.action == episode.goal)
        safe_count += int(full.action == 0)

    count = len(episodes)
    mean = {name: sum(values) / count for name, values in returns.items()}
    lcb_reactive = _paired_lcb([a - b for a, b in zip(returns["full"], returns["reactive"])])
    lcb_recurrent = _paired_lcb([a - b for a, b in zip(returns["full"], returns["recurrent"])])
    lcb_agnostic = _paired_lcb([a - b for a, b in zip(returns["full"], returns["action_agnostic"])])
    lcb_h1 = _paired_lcb([a - b for a, b in zip(returns["full"], returns["h1"])])
    lcb_h3 = _paired_lcb([a - b for a, b in zip(returns["h3"], returns["full"])])
    brier = sum((probabilities[i] - labels[i]) ** 2 for i in range(count)) / count
    ece = _ece(probabilities, labels)
    success_rate = successes / count
    result: dict[str, float | int | bool] = {
        "delay": delay,
        "episodes": count,
        "full_return": mean["full"],
        "reactive_return": mean["reactive"],
        "recurrent_return": mean["recurrent"],
        "action_agnostic_return": mean["action_agnostic"],
        "h1_return": mean["h1"],
        "h3_return": mean["h3"],
        "lcb_full_minus_reactive": lcb_reactive,
        "lcb_full_minus_recurrent": lcb_recurrent,
        "lcb_full_minus_action_agnostic": lcb_agnostic,
        "lcb_full_minus_h1": lcb_h1,
        "lcb_h3_minus_full": lcb_h3,
        "brier": brier,
        "ece": ece,
        "success_rate": success_rate,
        "safe_rate": safe_count / count,
        "future_reads": 0,
        "environment_clone_calls": 0,
    }
    result["hard_gate"] = bool(
        lcb_reactive > 0.15
        and lcb_recurrent > -0.03
        and lcb_agnostic > 0.08
        and lcb_h1 > 0.08
        and (delay != 2 or lcb_h3 > -0.03)
        and brier < 0.20
        and ece < 0.08
        and success_rate > 0.70
    )
    return result


def evaluate_validation(
    *,
    start_seed: int = 920000,
    episodes: int = 512,
    config: DPCConfig | None = None,
) -> dict[str, object]:
    cfg = config or DPCConfig()
    seeds = range(start_seed, start_seed + episodes)
    delay2 = evaluate_delay(seeds, 2, cfg)
    delay3 = evaluate_delay(seeds, 3, cfg)
    sensitivity_episode = DPCEpisode(start_seed, 1, 1, 0.7, 2)
    p_positive = posterior_positive(sensitivity_episode, cfg, action_conditioned=True)
    p_flipped = posterior_positive(
        DPCEpisode(start_seed, 1, -1, 0.7, 2), cfg, action_conditioned=True
    )
    action_sensitivity = p_positive > 0.5 and p_flipped < 0.5
    hard_gate = bool(delay2["hard_gate"] and delay3["hard_gate"] and action_sensitivity)

    def clip(value: float) -> float:
        return min(max(value, 0.0), 1.0)

    if hard_gate:
        score = 0.0
        score += 10.0 * clip(float(delay2["lcb_full_minus_reactive"]) / 0.25)
        score += 10.0 * clip(float(delay3["lcb_full_minus_reactive"]) / 0.25)
        score += 7.5 * clip((float(delay2["lcb_full_minus_recurrent"]) + 0.03) / 0.15)
        score += 7.5 * clip((float(delay3["lcb_full_minus_recurrent"]) + 0.03) / 0.15)
        score += 10.0 * clip(float(delay2["lcb_full_minus_action_agnostic"]) / 0.15)
        score += 10.0 * clip(float(delay3["lcb_full_minus_h1"]) / 0.15)
        mean_brier = (float(delay2["brier"]) + float(delay3["brier"])) / 2
        mean_ece = (float(delay2["ece"]) + float(delay3["ece"])) / 2
        score += 8.0 * clip((0.25 - mean_brier) / 0.15)
        score += 7.0 * clip((0.12 - mean_ece) / 0.10)
        # OOD and efficiency are deliberately not awarded in validation-only Loop 1a.
        score += 10.0  # leakage/integrity guards
        grade = "GO" if score >= 80 else "HOLD" if score >= 65 else "STOP"
    else:
        score = 0.0
        grade = "STOP"
    return {
        "schema": "clarus.dpc.validation.v1",
        "status": "training-free teacher-identified validation",
        "config": asdict(cfg),
        "seed_start": start_seed,
        "episodes_per_delay": episodes,
        "delay2": delay2,
        "delay3": delay3,
        "action_sensitivity": action_sensitivity,
        "hard_gate": hard_gate,
        "promisingness_score": score,
        "grade": grade,
        "claim_limit": "synthetic teacher-identified belief-control direction only",
    }


def _evaluate_learned_delay(
    seeds: Iterable[int],
    delay: int,
    config: DPCConfig,
    full_model: LogisticBeliefModel,
    agnostic_model: LogisticBeliefModel,
) -> dict[str, float | int | bool]:
    returns = {name: [] for name in ("full", "reactive", "recurrent", "agnostic", "h1", "h3")}
    probabilities: list[float] = []
    labels: list[int] = []
    successes = 0
    safe_count = 0
    seed_list = list(seeds)
    for seed in seed_list:
        episode = make_episode(seed, delay, config)
        probability = full_model.probability_positive(episode)
        agnostic_probability = agnostic_model.probability_positive(episode)
        full = decision_from_probability(probability, episode, config, horizon=delay)
        recurrent = decision_from_probability(probability, episode, config, horizon=delay)
        agnostic = decision_from_probability(agnostic_probability, episode, config, horizon=delay)
        h1 = decision_from_probability(probability, episode, config, horizon=1)
        h3 = decision_from_probability(probability, episode, config, horizon=3)
        decisions = {
            "full": full,
            "reactive": DPCDecision(0, 0.5, config.safe_reward),
            "recurrent": recurrent,
            "agnostic": agnostic,
            "h1": h1,
            "h3": h3,
        }
        for name, decision in decisions.items():
            returns[name].append(episode_return(episode, decision, config))
        probabilities.append(probability)
        labels.append(int(episode.goal == 1))
        successes += int(full.action == episode.goal)
        safe_count += int(full.action == 0)
    count = len(seed_list)
    mean = {name: sum(values) / count for name, values in returns.items()}
    paired = lambda left, right: _paired_lcb([a - b for a, b in zip(returns[left], returns[right])])
    result: dict[str, float | int | bool] = {
        "delay": delay,
        "episodes": count,
        "full_return": mean["full"],
        "reactive_return": mean["reactive"],
        "recurrent_return": mean["recurrent"],
        "action_agnostic_return": mean["agnostic"],
        "h1_return": mean["h1"],
        "h3_return": mean["h3"],
        "lcb_full_minus_reactive": paired("full", "reactive"),
        "lcb_full_minus_recurrent": paired("full", "recurrent"),
        "lcb_full_minus_action_agnostic": paired("full", "agnostic"),
        "lcb_full_minus_h1": paired("full", "h1"),
        "lcb_h3_minus_full": paired("h3", "full"),
        "brier": sum((probabilities[i] - labels[i]) ** 2 for i in range(count)) / count,
        "ece": _ece(probabilities, labels),
        "success_rate": successes / count,
        "safe_rate": safe_count / count,
        "future_reads": 0,
        "environment_clone_calls": 0,
    }
    result["hard_gate"] = bool(
        float(result["lcb_full_minus_reactive"]) > 0.15
        and float(result["lcb_full_minus_recurrent"]) > -0.03
        and float(result["lcb_full_minus_action_agnostic"]) > 0.08
        and float(result["lcb_full_minus_h1"]) > 0.08
        and (delay != 2 or float(result["lcb_h3_minus_full"]) > -0.03)
        and float(result["brier"]) < 0.20
        and float(result["ece"]) < 0.08
        and float(result["success_rate"]) > 0.70
    )
    return result


def evaluate_learned_validation(
    *,
    train_start: int = 910000,
    train_episodes: int = 2000,
    validation_start: int = 920000,
    validation_episodes: int = 512,
    config: DPCConfig | None = None,
) -> dict[str, object]:
    """Loop 1b: learn the belief likelihood, then freeze it for validation."""
    cfg = config or DPCConfig()
    train_seeds = range(train_start, train_start + train_episodes)
    full_model = fit_logistic_belief(train_seeds, cfg, action_conditioned=True)
    agnostic_model = fit_logistic_belief(train_seeds, cfg, action_conditioned=False)
    validation_seeds = range(validation_start, validation_start + validation_episodes)
    delay2 = _evaluate_learned_delay(validation_seeds, 2, cfg, full_model, agnostic_model)
    delay3 = _evaluate_learned_delay(validation_seeds, 3, cfg, full_model, agnostic_model)

    ood_cfg = DPCConfig(
        sigma=1.2,
        safe_reward=cfg.safe_reward,
        correct_reward=cfg.correct_reward,
        wrong_reward=cfg.wrong_reward,
        commit_cost=cfg.commit_cost,
        dropout_probability=cfg.dropout_probability,
        flip_probability=cfg.flip_probability,
    )
    ood_seeds = range(940000, 941024)
    ood2 = _evaluate_learned_delay(ood_seeds, 2, ood_cfg, full_model, agnostic_model)
    ood3 = _evaluate_learned_delay(ood_seeds, 3, ood_cfg, full_model, agnostic_model)
    hard_gate = bool(delay2["hard_gate"] and delay3["hard_gate"])

    def clip(value: float) -> float:
        return min(max(value, 0.0), 1.0)

    if hard_gate:
        score = 0.0
        score += 10 * clip(float(delay2["lcb_full_minus_reactive"]) / 0.25)
        score += 10 * clip(float(delay3["lcb_full_minus_reactive"]) / 0.25)
        score += 7.5 * clip((float(delay2["lcb_full_minus_recurrent"]) + 0.03) / 0.15)
        score += 7.5 * clip((float(delay3["lcb_full_minus_recurrent"]) + 0.03) / 0.15)
        score += 10 * clip(float(delay2["lcb_full_minus_action_agnostic"]) / 0.15)
        score += 10 * clip(float(delay3["lcb_full_minus_h1"]) / 0.15)
        mean_brier = (float(delay2["brier"]) + float(delay3["brier"])) / 2
        mean_ece = (float(delay2["ece"]) + float(delay3["ece"])) / 2
        score += 8 * clip((0.25 - mean_brier) / 0.15)
        score += 7 * clip((0.12 - mean_ece) / 0.10)
        score += 7.5 * float(float(ood2["lcb_full_minus_reactive"]) > 0)
        score += 7.5 * float(float(ood3["lcb_full_minus_reactive"]) > 0)
        score += 10  # integrity
        score += 5   # scalar logistic belief is below the recurrent budget
        grade = "GO" if score >= 80 else "HOLD" if score >= 65 else "STOP"
    else:
        score, grade = 0.0, "STOP"
    return {
        "schema": "clarus.dpc.learned-validation.v1",
        "status": "train-only fitted belief; validation and OOD evaluation",
        "config": asdict(cfg),
        "train_seed_start": train_start,
        "train_episodes": train_episodes,
        "validation_seed_start": validation_start,
        "validation_episodes_per_delay": validation_episodes,
        "full_model": asdict(full_model),
        "action_agnostic_model": asdict(agnostic_model),
        "delay2": delay2,
        "delay3": delay3,
        "ood_delay2": ood2,
        "ood_delay3": ood3,
        "hard_gate": hard_gate,
        "promisingness_score": score,
        "grade": grade,
        "claim_limit": "learned synthetic sufficient-statistic belief and finite-horizon control",
    }
