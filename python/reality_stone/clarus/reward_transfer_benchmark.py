"""Frozen-belief reward transfer benchmark for AGI Loop 4."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import torch

from .history_state_benchmark import (
    HistoryEpisode,
    HistoryStateConfig,
    fit_controlled_history,
    history_state,
    make_history_episode,
)


@dataclass(frozen=True)
class UtilityContext:
    safe_reward: float
    wrong_loss: float
    commit_cost: float
    correct_reward: float = 1.0


@dataclass(frozen=True)
class RewardTransferConfig:
    train_episodes: int = 2000
    validation_episodes: int = 512
    sigma: float = 0.9
    ood_sigma: float = 1.2
    policy_epochs: int = 250


TRAIN_CONTEXTS = tuple(
    UtilityContext(safe, loss, cost)
    for safe in (0.05, 0.25)
    for loss in (0.8, 1.4)
    for cost in (0.0, 0.10)
)
TRANSFER_CONTEXTS = {
    "balanced": UtilityContext(0.15, 1.10, 0.05),
    "cautious": UtilityContext(0.45, 1.80, 0.10),
    "asymmetric": UtilityContext(0.30, 2.20, 0.00),
}
STALE_CONTEXT = UtilityContext(0.15, 1.0, 0.0)


def choose_action(probability_positive: float, context: UtilityContext) -> int:
    positive_value = (
        probability_positive * (context.correct_reward - context.commit_cost)
        + (1.0 - probability_positive) * (-context.wrong_loss - context.commit_cost)
    )
    negative_value = (
        (1.0 - probability_positive) * (context.correct_reward - context.commit_cost)
        + probability_positive * (-context.wrong_loss - context.commit_cost)
    )
    best_commit = max(positive_value, negative_value)
    if best_commit <= context.safe_reward:
        return 0
    return 1 if positive_value >= negative_value else -1


def action_return(action: int, goal: int, context: UtilityContext) -> float:
    if action == 0:
        return context.safe_reward
    if action == goal:
        return context.correct_reward - context.commit_cost
    return -context.wrong_loss - context.commit_cost


def _oracle_probability(episode: HistoryEpisode, sigma: float) -> float:
    log_odds = 2.0 * history_state(episode, 1.0) / max(sigma * sigma, 1e-12)
    if log_odds >= 0.0:
        return 1.0 / (1.0 + math.exp(-log_odds))
    exp_value = math.exp(log_odds)
    return exp_value / (1.0 + exp_value)


class _ContextPolicyRNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recurrence = torch.nn.RNN(6, 2, nonlinearity="tanh", batch_first=True)
        self.readout = torch.nn.Linear(2, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.recurrence(inputs)
        return self.readout(sequence[:, -1, :])


def _policy_row(episode: HistoryEpisode, context: UtilityContext) -> list[list[float]]:
    return [
        [float(a), float(y), float(m), context.safe_reward, context.wrong_loss, context.commit_cost]
        for a, y, m in zip(episode.actions, episode.observations, episode.masks)
    ]


def _fit_policy(episodes: list[HistoryEpisode], sigma: float, epochs: int) -> _ContextPolicyRNN:
    torch.manual_seed(20260811)
    rows: list[list[list[float]]] = []
    labels: list[int] = []
    for episode in episodes:
        probability = _oracle_probability(episode, sigma)
        for context in TRAIN_CONTEXTS:
            rows.append(_policy_row(episode, context))
            labels.append(choose_action(probability, context) + 1)
    inputs = torch.tensor(rows, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.long)
    model = _ContextPolicyRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _policy_action(model: _ContextPolicyRNN, episode: HistoryEpisode, context: UtilityContext) -> int:
    row = torch.tensor([_policy_row(episode, context)], dtype=torch.float32)
    with torch.no_grad():
        return int(model(row).argmax(dim=1).item()) - 1


def _lcb(values: list[float], *, seed: int, draws: int = 2000) -> float:
    rng = random.Random(seed)
    means = [sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(draws)]
    means.sort()
    return means[max(0, int(0.025 * draws) - 1)]


def _evaluate_context(
    episodes: list[HistoryEpisode],
    sigma: float,
    context: UtilityContext,
    belief,
    policy: _ContextPolicyRNN,
    *,
    bootstrap_seed: int,
) -> dict[str, object]:
    returns = {name: [] for name in ("candidate", "context_rnn", "stale", "reactive", "oracle")}
    committed = correct_commits = 0
    for episode in episodes:
        probability = belief.probability(episode)
        actions = {
            "candidate": choose_action(probability, context),
            "context_rnn": _policy_action(policy, episode, context),
            "stale": choose_action(probability, STALE_CONTEXT),
            "reactive": choose_action(belief.probability(episode, "truncated"), context),
            "oracle": choose_action(_oracle_probability(episode, sigma), context),
        }
        for name, action in actions.items():
            returns[name].append(action_return(action, episode.goal, context))
        if actions["candidate"] != 0:
            committed += 1
            correct_commits += int(actions["candidate"] == episode.goal)
    means = {name: sum(values) / len(values) for name, values in returns.items()}
    lcb = {
        name: _lcb(
            [left - right for left, right in zip(returns["candidate"], returns[name])],
            seed=bootstrap_seed + index,
        )
        for index, name in enumerate(("context_rnn", "stale", "reactive"))
    }
    return {
        "context": asdict(context),
        "mean_return": means,
        "lcb_candidate_minus": lcb,
        "oracle_gap": means["oracle"] - means["candidate"],
        "commit_rate": committed / len(episodes),
        "success_among_commits": correct_commits / max(committed, 1),
    }


def evaluate_reward_transfer(config: RewardTransferConfig | None = None) -> dict[str, object]:
    cfg = config or RewardTransferConfig()
    history_cfg = HistoryStateConfig(
        sigma=cfg.sigma,
        ood_sigma=cfg.ood_sigma,
        train_episodes=cfg.train_episodes,
        validation_episodes=cfg.validation_episodes,
    )
    train = [make_history_episode(seed, history_cfg) for seed in range(980000, 980000 + cfg.train_episodes)]
    belief = fit_controlled_history(train, history_cfg)
    policy = _fit_policy(train, cfg.sigma, cfg.policy_epochs)
    id_episodes = [make_history_episode(seed, history_cfg) for seed in range(996000, 996000 + cfg.validation_episodes)]
    ood_episodes = [make_history_episode(seed, history_cfg, sigma=cfg.ood_sigma) for seed in range(997000, 997000 + cfg.validation_episodes)]
    results: dict[str, object] = {"id": {}, "ood": {}}
    lcb_rnn: list[float] = []
    hard_parts: list[bool] = []
    oracle_gaps: list[float] = []
    for domain_index, (domain, episodes, sigma) in enumerate((
        ("id", id_episodes, cfg.sigma), ("ood", ood_episodes, cfg.ood_sigma)
    )):
        domain_results = results[domain]
        assert isinstance(domain_results, dict)
        for context_index, (name, context) in enumerate(TRANSFER_CONTEXTS.items()):
            result = _evaluate_context(
                episodes, sigma, context, belief, policy,
                bootstrap_seed=20260811 + 100 * domain_index + 10 * context_index,
            )
            domain_results[name] = result
            lcb = result["lcb_candidate_minus"]
            assert isinstance(lcb, dict)
            lcb_rnn.append(float(lcb["context_rnn"]))
            oracle_gaps.append(float(result["oracle_gap"]))
            hard_parts.append(
                float(lcb["stale"]) > 0.0
                and float(lcb["reactive"]) > 0.0
                and float(lcb["context_rnn"]) > -0.03
                and float(result["oracle_gap"]) <= 0.02
                and float(result["success_among_commits"]) > 0.85
            )
    mean_rnn_lcb = sum(lcb_rnn) / len(lcb_rnn)
    hard_gate = bool(
        all(hard_parts)
        and mean_rnn_lcb > 0.02
        and max(oracle_gaps) <= 0.02
    )
    return {
        "schema": "clarus.reward-transfer.validation.v1",
        "config": asdict(cfg),
        "belief_model": asdict(belief),
        "training_contexts": [asdict(context) for context in TRAIN_CONTEXTS],
        "transfer_contexts": {name: asdict(context) for name, context in TRANSFER_CONTEXTS.items()},
        "results": results,
        "mean_lcb_candidate_minus_context_rnn": mean_rnn_lcb,
        "belief_updates_during_transfer": 0,
        "policy_updates_during_transfer": 0,
        "future_reads": 0,
        "environment_clone_calls": 0,
        "hard_gate": hard_gate,
        "promisingness_score": 90.0 if hard_gate else 0.0,
        "grade": "GO" if hard_gate else "STOP",
        "claim_limit": "synthetic modular reward-transfer only",
    }


__all__ = ["RewardTransferConfig", "UtilityContext", "action_return", "choose_action", "evaluate_reward_transfer"]
