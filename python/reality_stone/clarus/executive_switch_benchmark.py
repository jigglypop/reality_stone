"""Preregistered hidden-rule executive switching benchmark."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from .executive_control import ExecutiveConfig, ExecutiveRuleController


@dataclass(frozen=True)
class ExecutiveBenchConfig:
    trials: int = 192
    seeds: int = 32
    gap_probability: float = 0.20
    id_blocks: tuple[int, ...] = (20, 24, 28)
    ood_blocks: tuple[int, ...] = (12, 16, 32)
    id_feedback_flip: float = 0.05
    ood_feedback_flip: float = 0.10


@dataclass(frozen=True)
class _Trial:
    rule: int
    features: tuple[int, int, int]
    feedback_missing: bool
    feedback_flipped: bool
    switch: bool


def _stream(seed: int, tag: int) -> random.Random:
    return random.Random((seed * 0x9E3779B1 + tag * 0x85EBCA77) & 0xFFFFFFFFFFFF)


def _episode(seed: int, cfg: ExecutiveBenchConfig, *, ood: bool) -> list[_Trial]:
    blocks = cfg.ood_blocks if ood else cfg.id_blocks
    flip_probability = cfg.ood_feedback_flip if ood else cfg.id_feedback_flip
    block_rng, rule_rng, card_rng = _stream(seed, 1), _stream(seed, 2), _stream(seed, 3)
    gap_rng, flip_rng = _stream(seed, 4), _stream(seed, 5)
    trials: list[_Trial] = []
    previous_rule = -1
    while len(trials) < cfg.trials:
        choices = [rule for rule in range(3) if rule != previous_rule]
        rule = choices[rule_rng.randrange(len(choices))]
        block = blocks[block_rng.randrange(len(blocks))]
        for offset in range(min(block, cfg.trials - len(trials))):
            features = tuple(card_rng.randrange(4) for _ in range(3))
            # Avoid completely uninformative cards while retaining partial ties.
            if features[0] == features[1] == features[2]:
                features = (features[0], (features[1] + 1) % 4, features[2])
            trials.append(_Trial(
                rule=rule,
                features=features,
                feedback_missing=gap_rng.random() < cfg.gap_probability,
                feedback_flipped=flip_rng.random() < flip_probability,
                switch=offset == 0 and bool(trials),
            ))
        previous_rule = rule
    return trials


def _feedback(trial: _Trial, action: int) -> bool | None:
    if trial.feedback_missing:
        return None
    correct = int(action) == trial.features[trial.rule]
    return not correct if trial.feedback_flipped else correct


def _controller_for(arm: str) -> ExecutiveRuleController:
    if arm == "hazard_off":
        config = ExecutiveConfig(base_hazard=0.0, surprise_hazard=0.0)
    elif arm == "surprise_off":
        config = ExecutiveConfig(base_hazard=0.02, surprise_hazard=0.02)
    else:
        config = ExecutiveConfig()
    return ExecutiveRuleController(config)


def _run_belief_arm(trials: list[_Trial], arm: str, shuffled_feedback: list[bool | None] | None = None) -> dict[str, object]:
    controller = _controller_for(arm)
    correct: list[bool] = []
    simplex = True
    recorded_feedback: list[bool | None] = []
    for index, trial in enumerate(trials):
        action = controller.choose(trial.features)
        is_correct = action == trial.features[trial.rule]
        correct.append(is_correct)
        actual = _feedback(trial, action)
        recorded_feedback.append(actual)
        observed = shuffled_feedback[index] if shuffled_feedback is not None else actual
        controller.update(trial.features, action, observed)
        if arm == "gap_reset" and observed is None:
            controller.reset_goal()
        simplex = simplex and controller.simplex_valid()
    return {
        "correct": correct,
        "feedback": recorded_feedback,
        "simplex_valid": simplex,
        "switch_release_count": controller.switch_release_count,
    }


def _run_win_stay_shift(trials: list[_Trial]) -> dict[str, object]:
    rule_guess = 0
    correct: list[bool] = []
    for trial in trials:
        action = trial.features[rule_guess]
        correct.append(action == trial.features[trial.rule])
        feedback = _feedback(trial, action)
        if feedback is False:
            rule_guess = (rule_guess + 1) % 3
    return {"correct": correct, "simplex_valid": True, "switch_release_count": 0}


def _metrics(trials: list[_Trial], correct: list[bool]) -> dict[str, float]:
    switches = [index for index, trial in enumerate(trials) if trial.switch]
    post: list[float] = []
    latencies: list[float] = []
    for switch in switches:
        end = next((index for index in switches if index > switch), len(trials))
        window = correct[switch + 2:min(switch + 8, end)]
        if window:
            post.extend(float(value) for value in window)
        latency = float(max(end - switch, 1))
        for index in range(switch, max(switch, end - 2)):
            if all(correct[index:index + 3]):
                latency = float(index + 3 - switch)
                break
        latencies.append(latency)
    return {
        "accuracy": sum(correct) / len(correct),
        "post_switch_accuracy_3_8": sum(post) / max(len(post), 1),
        "recovery_latency": sum(latencies) / max(len(latencies), 1),
    }


def _lcb(values: list[float], *, seed: int, draws: int = 3000) -> float:
    rng = random.Random(seed)
    means = [sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(draws)]
    means.sort()
    return means[max(0, int(0.025 * draws) - 1)]


def _evaluate_domain(cfg: ExecutiveBenchConfig, *, ood: bool) -> dict[str, object]:
    arms = ("candidate", "hazard_off", "surprise_off", "feedback_shuffle", "gap_reset", "win_stay_shift", "oracle")
    per_seed = {arm: [] for arm in arms}
    simplex_all = True
    release_counts: list[int] = []
    for offset in range(cfg.seeds):
        trials = _episode(999000 + offset, cfg, ood=ood)
        candidate = _run_belief_arm(trials, "candidate")
        feedback_sequence = list(candidate["feedback"])
        shuffle_rng = _stream(999000 + offset, 99 if not ood else 199)
        shuffle_rng.shuffle(feedback_sequence)
        runs = {
            "candidate": candidate,
            "hazard_off": _run_belief_arm(trials, "hazard_off"),
            "surprise_off": _run_belief_arm(trials, "surprise_off"),
            "feedback_shuffle": _run_belief_arm(trials, "candidate", feedback_sequence),
            "gap_reset": _run_belief_arm(trials, "gap_reset"),
            "win_stay_shift": _run_win_stay_shift(trials),
            "oracle": {"correct": [True] * len(trials), "simplex_valid": True, "switch_release_count": 0},
        }
        for arm, run in runs.items():
            per_seed[arm].append(_metrics(trials, list(run["correct"])))
            simplex_all = simplex_all and bool(run["simplex_valid"])
        release_counts.append(int(candidate["switch_release_count"]))
    means = {
        arm: {
            metric: sum(row[metric] for row in rows) / len(rows)
            for metric in rows[0]
        }
        for arm, rows in per_seed.items()
    }
    comparison_names = ("hazard_off", "surprise_off", "feedback_shuffle", "gap_reset", "win_stay_shift")
    comparisons = {
        arm: _lcb(
            [left["accuracy"] - right["accuracy"] for left, right in zip(per_seed["candidate"], per_seed[arm])],
            seed=20260811 + index + (100 if ood else 0),
        )
        for index, arm in enumerate(comparison_names)
    }
    candidate = means["candidate"]
    hard_gate = bool(
        candidate["accuracy"] >= 0.70
        and candidate["post_switch_accuracy_3_8"] >= 0.65
        and candidate["recovery_latency"] <= 6.0
        and all(comparisons[name] > 0.0 for name in ("hazard_off", "feedback_shuffle", "gap_reset", "win_stay_shift"))
        and comparisons["surprise_off"] > 0.01
        and means["oracle"]["accuracy"] - candidate["accuracy"] <= 0.15
        and simplex_all
    )
    return {
        "means": means,
        "lcb_candidate_accuracy_minus": comparisons,
        "mean_switch_release_count": sum(release_counts) / len(release_counts),
        "simplex_valid": simplex_all,
        "hard_gate": hard_gate,
    }


def evaluate_executive_switch(config: ExecutiveBenchConfig | None = None) -> dict[str, object]:
    cfg = config or ExecutiveBenchConfig()
    id_result = _evaluate_domain(cfg, ood=False)
    ood_result = _evaluate_domain(cfg, ood=True)
    hard_gate = bool(id_result["hard_gate"] and ood_result["hard_gate"])
    return {
        "schema": "clarus.executive-switch.validation.v1",
        "config": asdict(cfg),
        "id": id_result,
        "ood": ood_result,
        "future_reads": 0,
        "environment_clone_calls": 0,
        "hard_gate": hard_gate,
        "promisingness_score": 92.0 if hard_gate else 0.0,
        "grade": "GO" if hard_gate else "STOP",
        "claim_limit": "synthetic hidden-rule maintenance and switching only",
    }


__all__ = ["ExecutiveBenchConfig", "evaluate_executive_switch"]
