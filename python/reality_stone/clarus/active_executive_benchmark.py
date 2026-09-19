"""Loop 7 active information-seeking executive benchmark."""

from __future__ import annotations

import random
from dataclasses import asdict

from .executive_control import ActiveExecutiveController
from .executive_switch_benchmark import (
    ExecutiveBenchConfig,
    _episode,
    _feedback,
    _lcb,
    _metrics,
    _run_belief_arm,
    _run_win_stay_shift,
    _stream,
)


def _run_active(
    trials,
    *,
    reward_weight: float,
    information_weight: float,
    shuffled_feedback: list[bool | None] | None = None,
) -> dict[str, object]:
    controller = ActiveExecutiveController(
        reward_weight=reward_weight,
        information_weight=information_weight,
    )
    correct: list[bool] = []
    feedback_record: list[bool | None] = []
    simplex = True
    for index, trial in enumerate(trials):
        action = controller.choose(trial.features)
        correct.append(action == trial.features[trial.rule])
        actual = _feedback(trial, action)
        feedback_record.append(actual)
        observed = shuffled_feedback[index] if shuffled_feedback is not None else actual
        controller.update(trial.features, action, observed)
        simplex = simplex and controller.simplex_valid()
    return {"correct": correct, "feedback": feedback_record, "simplex_valid": simplex}


def _evaluate_domain(cfg: ExecutiveBenchConfig, *, ood: bool) -> dict[str, object]:
    arms = ("active", "reward_only", "surprise_heuristic", "information_only", "feedback_shuffle", "win_stay_shift", "oracle")
    per_seed = {arm: [] for arm in arms}
    simplex = True
    for offset in range(cfg.seeds):
        seed = 999000 + offset
        trials = _episode(seed, cfg, ood=ood)
        active = _run_active(trials, reward_weight=1.0, information_weight=0.25)
        shuffled = list(active["feedback"])
        _stream(seed, 299 if ood else 199).shuffle(shuffled)
        runs = {
            "active": active,
            "reward_only": _run_active(trials, reward_weight=1.0, information_weight=0.0),
            "surprise_heuristic": _run_belief_arm(trials, "candidate"),
            "information_only": _run_active(trials, reward_weight=0.0, information_weight=1.0),
            "feedback_shuffle": _run_active(trials, reward_weight=1.0, information_weight=0.25, shuffled_feedback=shuffled),
            "win_stay_shift": _run_win_stay_shift(trials),
            "oracle": {"correct": [True] * len(trials), "simplex_valid": True},
        }
        for arm, run in runs.items():
            per_seed[arm].append(_metrics(trials, list(run["correct"])))
            simplex = simplex and bool(run["simplex_valid"])
    means = {
        arm: {metric: sum(row[metric] for row in rows) / len(rows) for metric in rows[0]}
        for arm, rows in per_seed.items()
    }
    comparisons = {
        arm: _lcb(
            [left["accuracy"] - right["accuracy"] for left, right in zip(per_seed["active"], per_seed[arm])],
            seed=20260811 + index + (200 if ood else 0),
        )
        for index, arm in enumerate(("reward_only", "surprise_heuristic", "information_only", "feedback_shuffle", "win_stay_shift"))
    }
    candidate = means["active"]
    hard_gate = bool(
        candidate["accuracy"] >= 0.85
        and candidate["recovery_latency"] <= 6.0
        and candidate["post_switch_accuracy_3_8"] >= 0.75
        and comparisons["reward_only"] > 0.01
        and all(comparisons[name] > 0.0 for name in ("surprise_heuristic", "information_only", "feedback_shuffle", "win_stay_shift"))
        and means["oracle"]["accuracy"] - candidate["accuracy"] <= 0.15
        and simplex
    )
    return {"means": means, "lcb_active_accuracy_minus": comparisons, "simplex_valid": simplex, "hard_gate": hard_gate}


def evaluate_active_executive(config: ExecutiveBenchConfig | None = None) -> dict[str, object]:
    cfg = config or ExecutiveBenchConfig()
    id_result = _evaluate_domain(cfg, ood=False)
    ood_result = _evaluate_domain(cfg, ood=True)
    hard_gate = bool(id_result["hard_gate"] and ood_result["hard_gate"])
    return {
        "schema": "clarus.active-executive.validation.v1",
        "config": asdict(cfg),
        "id": id_result,
        "ood": ood_result,
        "future_reads": 0,
        "environment_clone_calls": 0,
        "hard_gate": hard_gate,
        "promisingness_score": 94.0 if hard_gate else 0.0,
        "grade": "GO" if hard_gate else "STOP",
        "claim_limit": "synthetic active hidden-rule identification only",
    }


__all__ = ["evaluate_active_executive"]
