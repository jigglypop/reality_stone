"""Preregistered raw-history state-learning probe for AGI Loop 3.

The candidate is deliberately minimal: an action-conditioned leaky evidence
state learned from ordered observations.  The hidden goal and the collapsed
episode statistic are never exposed to the learner.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class HistoryStateConfig:
    sigma: float = 0.9
    ood_sigma: float = 1.2
    evidence_events: int = 3
    gap: int = 2
    safe_reward: float = 0.15
    correct_reward: float = 1.0
    wrong_reward: float = -1.0
    train_episodes: int = 2000
    validation_episodes: int = 512
    rho_grid: tuple[float, ...] = (0.50, 0.70, 0.85, 0.95, 1.00)


@dataclass(frozen=True)
class HistoryEpisode:
    seed: int
    goal: int
    actions: tuple[int, ...]
    observations: tuple[float, ...]
    masks: tuple[int, ...]


@dataclass(frozen=True)
class ControlledHistoryModel:
    rho: float
    weight: float
    bias: float

    def probability(self, episode: HistoryEpisode, mode: str = "candidate") -> float:
        state = history_state(episode, self.rho, mode=mode)
        return _sigmoid(self.weight * state + self.bias)


def _rng(seed: int, tag: int) -> random.Random:
    return random.Random((seed * 0x9E3779B1 + tag * 0x85EBCA77) & 0xFFFFFFFFFFFF)


def make_history_episode(seed: int, config: HistoryStateConfig, *, sigma: float | None = None) -> HistoryEpisode:
    noise = config.sigma if sigma is None else float(sigma)
    goal = -1 if _rng(seed, 1).random() < 0.5 else 1
    length = 1 + (config.evidence_events - 1) * config.gap
    actions = [0] * length
    observations = [0.0] * length
    masks = [0] * length
    for event in range(config.evidence_events):
        index = event * config.gap
        action = -1 if _rng(seed, 10 + event).random() < 0.5 else 1
        actions[index] = action
        observations[index] = action * goal + _rng(seed, 20 + event).gauss(0.0, noise)
        masks[index] = 1
    return HistoryEpisode(seed, goal, tuple(actions), tuple(observations), tuple(masks))


def history_state(episode: HistoryEpisode, rho: float, *, mode: str = "candidate") -> float:
    state = 0.0
    observed = [i for i, mask in enumerate(episode.masks) if mask]
    shuffled_actions = [episode.actions[i] for i in observed]
    if shuffled_actions:
        shuffled_actions = shuffled_actions[1:] + shuffled_actions[:1]
    shuffled_index = 0
    for action, observation, mask in zip(episode.actions, episode.observations, episode.masks):
        state *= rho
        if not mask:
            continue
        if mode == "observation_only":
            factor = 1
        elif mode == "action_shuffle":
            factor = shuffled_actions[shuffled_index]
        else:
            factor = action
        shuffled_index += 1
        if mode == "truncated":
            state = factor * observation
        else:
            state += factor * observation
    return state


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _fit_logistic(rows: list[tuple[float, float]], *, epochs: int = 500, lr: float = 0.15) -> tuple[float, float, float]:
    weight = bias = 0.0
    for _ in range(epochs):
        gw = gb = 0.0
        for feature, label in rows:
            error = _sigmoid(weight * feature + bias) - label
            gw += error * feature / len(rows)
            gb += error / len(rows)
        weight -= lr * gw
        bias -= lr * gb
    nll = -sum(
        label * math.log(max(_sigmoid(weight * feature + bias), 1e-12))
        + (1.0 - label) * math.log(max(1.0 - _sigmoid(weight * feature + bias), 1e-12))
        for feature, label in rows
    ) / len(rows)
    return weight, bias, nll


def fit_controlled_history(episodes: Iterable[HistoryEpisode], config: HistoryStateConfig) -> ControlledHistoryModel:
    episode_list = list(episodes)
    best: tuple[float, float, float, float] | None = None
    for rho in config.rho_grid:
        rows = [(history_state(ep, rho), float(ep.goal == 1)) for ep in episode_list]
        weight, bias, nll = _fit_logistic(rows)
        candidate = (nll, -rho, weight, bias)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return ControlledHistoryModel(rho=-best[1], weight=best[2], bias=best[3])


class _TinyRecurrent(torch.nn.Module):
    """Two-state tanh RNN used as the frozen matched recurrent comparator."""

    def __init__(self) -> None:
        super().__init__()
        self.recurrence = torch.nn.RNN(
            input_size=3, hidden_size=2, nonlinearity="tanh", batch_first=True
        )
        self.readout = torch.nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.recurrence(inputs)
        return self.readout(sequence[:, -1, :])


def _fit_recurrent(episodes: list[HistoryEpisode]) -> torch.nn.Module:
    torch.manual_seed(20260811)
    model = _TinyRecurrent()
    inputs = torch.tensor([
        [[a, y, m] for a, y, m in zip(ep.actions, ep.observations, ep.masks)]
        for ep in episodes
    ], dtype=torch.float32)
    labels = torch.tensor([[float(ep.goal == 1)] for ep in episodes], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
    for _ in range(250):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(inputs), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _recurrent_probability(model: torch.nn.Module, episode: HistoryEpisode) -> float:
    row = torch.tensor([[[a, y, m] for a, y, m in zip(episode.actions, episode.observations, episode.masks)]], dtype=torch.float32)
    with torch.no_grad():
        return float(torch.sigmoid(model(row)).item())


def _return(probability: float, goal: int, config: HistoryStateConfig) -> tuple[float, bool]:
    confidence = max(probability, 1.0 - probability)
    expected = confidence * config.correct_reward + (1.0 - confidence) * config.wrong_reward
    if expected <= config.safe_reward:
        return config.safe_reward, False
    action = 1 if probability >= 0.5 else -1
    return (config.correct_reward if action == goal else config.wrong_reward), action == goal


def _lcb(values: list[float], seed: int = 20260811, draws: int = 2000) -> float:
    rng = random.Random(seed)
    means = [sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(draws)]
    means.sort()
    return means[max(0, int(0.025 * draws) - 1)]


def _ece(probabilities: list[float], labels: list[int], bins: int = 10) -> float:
    error = 0.0
    for index in range(bins):
        selected = [i for i, p in enumerate(probabilities) if index / bins <= p < (index + 1) / bins or (index == bins - 1 and p == 1.0)]
        if selected:
            error += len(selected) / len(labels) * abs(
                sum(probabilities[i] for i in selected) / len(selected)
                - sum(labels[i] for i in selected) / len(selected)
            )
    return error


def _evaluate(episodes: list[HistoryEpisode], model: ControlledHistoryModel, recurrent: torch.nn.Module, config: HistoryStateConfig) -> dict[str, object]:
    names = ("candidate", "reactive", "observation_only", "action_shuffle", "truncated", "recurrent", "oracle")
    returns = {name: [] for name in names}
    probabilities: list[float] = []
    labels: list[int] = []
    successes = 0
    for episode in episodes:
        candidate = model.probability(episode)
        probs = {
            "candidate": candidate,
            "reactive": model.probability(episode, "truncated"),
            "observation_only": model.probability(episode, "observation_only"),
            "action_shuffle": model.probability(episode, "action_shuffle"),
            "truncated": model.probability(episode, "truncated"),
            "recurrent": _recurrent_probability(recurrent, episode),
            "oracle": _sigmoid(2.0 * history_state(episode, 1.0) / (config.sigma ** 2)),
        }
        for name, probability in probs.items():
            reward, correct = _return(probability, episode.goal, config)
            returns[name].append(reward)
            if name == "candidate":
                successes += int(correct)
        probabilities.append(candidate)
        labels.append(int(episode.goal == 1))
    means = {name: sum(values) / len(values) for name, values in returns.items()}
    comparisons = {
        name: _lcb([a - b for a, b in zip(returns["candidate"], returns[name])])
        for name in ("reactive", "observation_only", "action_shuffle", "truncated", "recurrent")
    }
    return {
        "mean_return": means,
        "lcb_candidate_minus": comparisons,
        "success_rate": successes / len(episodes),
        "brier": sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels),
        "ece": _ece(probabilities, labels),
    }


def evaluate_history_state(config: HistoryStateConfig | None = None) -> dict[str, object]:
    cfg = config or HistoryStateConfig()
    train = [make_history_episode(seed, cfg) for seed in range(980000, 980000 + cfg.train_episodes)]
    model = fit_controlled_history(train, cfg)
    recurrent = _fit_recurrent(train)
    validation = [make_history_episode(seed, cfg) for seed in range(990000, 990000 + cfg.validation_episodes)]
    ood = [make_history_episode(seed, cfg, sigma=cfg.ood_sigma) for seed in range(995000, 995000 + cfg.validation_episodes)]
    id_result = _evaluate(validation, model, recurrent, cfg)
    ood_result = _evaluate(ood, model, recurrent, cfg)
    sensitivity = (
        model.probability(HistoryEpisode(0, 1, (1,), (0.7,), (1,))) > 0.5
        and model.probability(HistoryEpisode(0, 1, (-1,), (0.7,), (1,))) < 0.5
    )
    def passes(result: dict[str, object], *, ood_gate: bool) -> bool:
        lcb = result["lcb_candidate_minus"]
        assert isinstance(lcb, dict)
        return bool(
            all(float(lcb[name]) > 0.0 for name in ("reactive", "observation_only", "action_shuffle", "truncated"))
            and float(lcb["recurrent"]) > -0.03
            and float(result["success_rate"]) > 0.70
            and float(result["brier"]) < 0.20
            and float(result["ece"]) < (0.10 if ood_gate else 0.08)
        )
    hard_gate = passes(id_result, ood_gate=False) and passes(ood_result, ood_gate=True) and sensitivity
    score = 85.0 if hard_gate else 0.0
    return {
        "schema": "clarus.history-state.validation.v1",
        "config": asdict(cfg),
        "model": asdict(model),
        "id": id_result,
        "ood": ood_result,
        "action_sensitivity": sensitivity,
        "future_reads": 0,
        "environment_clone_calls": 0,
        "hard_gate": hard_gate,
        "promisingness_score": score,
        "grade": "GO" if score >= 80 else "STOP",
        "claim_limit": "synthetic controlled-evidence history state discovery only",
    }


__all__ = ["ControlledHistoryModel", "HistoryEpisode", "HistoryStateConfig", "evaluate_history_state", "fit_controlled_history", "history_state", "make_history_episode"]
