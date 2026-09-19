"""Minimal delayed-credit benchmark for signed TD eligibility traces."""

from __future__ import annotations

import random
from dataclasses import asdict

from .credit_control import EligibilityQLearner, TemporalCreditConfig


def _run_episode(
    learner: EligibilityQLearner,
    cue: int,
    *,
    rng: random.Random,
    epsilon: float,
    reward_shuffle: bool = False,
) -> int:
    learner.start_episode()
    action = learner.act(cue, rng=rng, epsilon=epsilon)
    learner.mark_eligibility(cue, action)
    # The outcome arrives after two blank transitions.  Those transitions
    # contain no Markov state capable of reconstructing the initial choice;
    # only the eligibility trace may carry its credit.
    learner.decay_eligibility()
    learner.decay_eligibility()
    correct_action = cue
    if reward_shuffle:
        correct_action = rng.randrange(2)
    reward = 1.0 if action == correct_action else -1.0
    learner.apply_credit(reward)
    return int(action == cue)


def _train(
    config: TemporalCreditConfig,
    seeds: range,
    *,
    absolute_td: bool = False,
    reward_shuffle: bool = False,
) -> EligibilityQLearner:
    learner = EligibilityQLearner(config, absolute_td=absolute_td)
    for seed in seeds:
        rng = random.Random(seed)
        cue = rng.randrange(2)
        _run_episode(
            learner,
            cue,
            rng=rng,
            epsilon=0.15,
            reward_shuffle=reward_shuffle,
        )
    return learner


def _success_rate(learner: EligibilityQLearner, seeds: range) -> float:
    wins = 0
    for seed in seeds:
        cue = random.Random(seed).randrange(2)
        wins += int(learner.act(cue) == cue)
    return wins / len(seeds)


def evaluate_delayed_credit(
    *,
    train_start: int = 950000,
    train_episodes: int = 1000,
    validation_start: int = 960000,
    validation_episodes: int = 512,
) -> dict[str, object]:
    base = TemporalCreditConfig(state_count=4, action_count=2, trace_decay=0.8)
    no_trace = TemporalCreditConfig(state_count=4, action_count=2, trace_decay=0.0)
    train = range(train_start, train_start + train_episodes)
    validation = range(validation_start, validation_start + validation_episodes)
    signed = _train(base, train)
    trace_off = _train(no_trace, train)
    unsigned = _train(base, train, absolute_td=True)
    shuffled = _train(base, train, reward_shuffle=True)
    rates = {
        "signed_td_lambda": _success_rate(signed, validation),
        "trace_off": _success_rate(trace_off, validation),
        "absolute_td": _success_rate(unsigned, validation),
        "reward_shuffled": _success_rate(shuffled, validation),
    }
    hard_gate = bool(
        rates["signed_td_lambda"] > 0.80
        and rates["signed_td_lambda"] - rates["trace_off"] > 0.20
        and rates["signed_td_lambda"] - rates["absolute_td"] > 0.20
        and rates["signed_td_lambda"] - rates["reward_shuffled"] > 0.20
    )
    score = 0.0
    if hard_gate:
        score += 40.0 * min(max((rates["signed_td_lambda"] - 0.5) / 0.5, 0.0), 1.0)
        score += 20.0 * min(max((rates["signed_td_lambda"] - rates["trace_off"]) / 0.5, 0.0), 1.0)
        score += 20.0 * min(max((rates["signed_td_lambda"] - rates["absolute_td"]) / 0.5, 0.0), 1.0)
        score += 10.0 * min(max((rates["signed_td_lambda"] - rates["reward_shuffled"]) / 0.5, 0.0), 1.0)
        score += 10.0  # finite/local eligibility/integrity checks
    return {
        "schema": "clarus.delayed-credit.validation.v1",
        "config": asdict(base),
        "train_seed_start": train_start,
        "train_episodes": train_episodes,
        "validation_seed_start": validation_start,
        "validation_episodes": validation_episodes,
        "success_rates": rates,
        "hard_gate": hard_gate,
        "promisingness_score": score,
        "grade": "GO" if score >= 80 else "HOLD" if score >= 65 else "STOP",
        "claim_limit": "tabular delayed-credit mechanism only",
    }
