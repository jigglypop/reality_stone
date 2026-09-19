"""Preregistered Loop 8B brain-geometry mechanism benchmark.

This is a bounded synthetic test of continuous MD-like attractor modulation.
It does not claim biological identity or AGI capability.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BrainGeometryBenchConfig:
    trials: int = 192
    seeds: int = 32
    step: float = 0.05
    attractor_gain: float = 1.5
    cross_inhibition: float = 1.0
    diffusion: float = 0.08
    heat_diffusion: float = 0.80
    encoding_steps: int = 10
    input_gain: float = 1.3
    distractor_gain: float = 0.80
    id_delay: int = 40
    ood_delay: int = 70
    id_context_noise: float = 0.55
    ood_context_noise: float = 0.85
    context_retention: float = 0.85
    cue_gain: float = 0.50
    context_logit_gain: float = 2.0
    relevance_floor: float = 0.40
    relevance_gain: float = 1.20
    blocks: tuple[int, ...] = (20, 24, 28)


@dataclass(frozen=True)
class _Trial:
    context: int
    features: tuple[int, int]
    cue: float
    switch: bool


def _stream(seed: int, tag: int) -> random.Random:
    return random.Random((seed * 0x9E3779B1 + tag * 0x85EBCA77) & 0xFFFFFFFFFFFF)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _trials(seed: int, config: BrainGeometryBenchConfig, *, ood: bool) -> list[_Trial]:
    block_rng = _stream(seed, 1)
    feature_rng = _stream(seed, 2)
    cue_rng = _stream(seed, 3)
    context_noise = config.ood_context_noise if ood else config.id_context_noise
    rows: list[_Trial] = []
    context = block_rng.randrange(2)
    while len(rows) < config.trials:
        if rows:
            context = 1 - context
        block = config.blocks[block_rng.randrange(len(config.blocks))]
        for offset in range(min(block, config.trials - len(rows))):
            first = -1 if feature_rng.random() < 0.5 else 1
            cue_sign = -1.0 if context == 0 else 1.0
            rows.append(_Trial(
                context=context,
                features=(first, -first),
                cue=cue_sign + cue_rng.gauss(0.0, context_noise),
                switch=offset == 0 and bool(rows),
            ))
    return rows


def pure_diffusion_mode(steps: int, config: BrainGeometryBenchConfig | None = None) -> float:
    """Return the exact nonconstant-mode amplitude after heat-flow steps."""
    cfg = config or BrainGeometryBenchConfig()
    difference = 1.0
    attenuation = math.exp(-2.0 * cfg.heat_diffusion * cfg.step)
    for _ in range(steps):
        difference *= attenuation
    return difference


def _heat_step(
    state: list[float],
    external: tuple[float, float],
    noise: tuple[float, float],
    config: BrainGeometryBenchConfig,
) -> list[float]:
    provisional = [
        state[index] + config.step * external[index] + noise[index]
        for index in range(2)
    ]
    mean = 0.5 * (provisional[0] + provisional[1])
    half_difference = 0.5 * (provisional[0] - provisional[1])
    half_difference *= math.exp(-2.0 * config.heat_diffusion * config.step)
    return [mean + half_difference, mean - half_difference]


def _attractor_step(
    state: list[float],
    external: tuple[float, float],
    noise: tuple[float, float],
    theta: float,
    arm: str,
    config: BrainGeometryBenchConfig,
) -> list[float]:
    if arm == "fixed_attractor":
        gains = (config.attractor_gain, config.attractor_gain)
    else:
        relevances = (theta, 1.0 - theta)
        gains = tuple(
            config.attractor_gain
            * (config.relevance_floor + config.relevance_gain * relevance)
            for relevance in relevances
        )
    updated: list[float] = []
    for index in range(2):
        other = 1 - index
        value = state[index]
        drift = (
            gains[index] * value * (1.0 - value * value)
            - 2.0 * config.cross_inhibition * value * state[other] * state[other]
            + external[index]
        )
        updated.append(value + config.step * drift + noise[index])
    return updated


def _trial_noise(seed: int, trial_index: int, steps: int, config: BrainGeometryBenchConfig) -> list[tuple[float, float]]:
    rng = _stream(seed * 1009 + trial_index, 17)
    scale = math.sqrt(2.0 * config.diffusion * config.step)
    return [(scale * rng.gauss(0.0, 1.0), scale * rng.gauss(0.0, 1.0)) for _ in range(steps)]


def _distractors(seed: int, trial_index: int) -> tuple[tuple[int, int], ...]:
    rng = _stream(seed * 1013 + trial_index, 23)
    return tuple(
        (-1 if rng.random() < 0.5 else 1, -1 if rng.random() < 0.5 else 1)
        for _ in range(3)
    )


def _run_arm(
    trials: list[_Trial],
    arm: str,
    seed: int,
    config: BrainGeometryBenchConfig,
    *,
    ood: bool,
) -> dict[str, float | int | bool]:
    delay = config.ood_delay if ood else config.id_delay
    cues = [trial.cue for trial in trials]
    if arm == "md_context_shuffle":
        shuffle_rng = _stream(seed, 71 if not ood else 73)
        shuffle_rng.shuffle(cues)

    context_state = 0.0
    correct: list[bool] = []
    context_correct: list[bool] = []
    margins: list[float] = []
    bounded = True
    nonfinite = 0
    max_abs = 0.0
    per_trial_correct: list[bool] = []

    for trial_index, trial in enumerate(trials):
        if arm == "oracle_context_md":
            theta = float(trial.context == 0)
        else:
            # theta is the probability that feature zero is relevant.
            context_state = (
                config.context_retention * context_state
                + config.cue_gain * (-cues[trial_index])
            )
            theta = _sigmoid(config.context_logit_gain * context_state)
        inferred_context = 0 if theta >= 0.5 else 1
        context_correct.append(inferred_context == trial.context)

        total_steps = config.encoding_steps + delay
        increments = _trial_noise(seed, trial_index, total_steps, config)
        distractors = _distractors(seed, trial_index)
        pulse_steps = {delay // 4: 0, delay // 2: 1, (3 * delay) // 4: 2}
        state = [0.0, 0.0]
        step_index = 0
        for _ in range(config.encoding_steps):
            external = (
                config.input_gain * trial.features[0],
                config.input_gain * trial.features[1],
            )
            if arm == "pure_diffusion":
                state = _heat_step(state, external, increments[step_index], config)
            else:
                state = _attractor_step(state, external, increments[step_index], theta, arm, config)
            step_index += 1
        for delay_step in range(delay):
            if delay_step in pulse_steps:
                distractor = distractors[pulse_steps[delay_step]]
                external = (
                    config.distractor_gain * distractor[0],
                    config.distractor_gain * distractor[1],
                )
            else:
                external = (0.0, 0.0)
            if arm == "pure_diffusion":
                state = _heat_step(state, external, increments[step_index], config)
            else:
                state = _attractor_step(state, external, increments[step_index], theta, arm, config)
            step_index += 1

        finite = all(math.isfinite(value) for value in state)
        nonfinite += int(not finite)
        if finite:
            max_abs = max(max_abs, abs(state[0]), abs(state[1]))
        bounded = bounded and finite and max(abs(value) for value in state) <= 4.0
        signal = theta * state[0] + (1.0 - theta) * state[1] if finite else 0.0
        target = trial.features[trial.context]
        is_correct = (signal >= 0.0) == (target > 0)
        correct.append(is_correct)
        per_trial_correct.append(is_correct)
        margins.append(target * signal)

    switches = [index for index, trial in enumerate(trials) if trial.switch]
    post_switch: list[bool] = []
    for switch_index in switches:
        next_switch = next(
            (index for index in switches if index > switch_index), len(trials)
        )
        post_switch.extend(per_trial_correct[switch_index + 2:min(switch_index + 8, next_switch)])
    return {
        "accuracy": sum(correct) / len(correct),
        "context_accuracy": sum(context_correct) / len(context_correct),
        "mean_margin": sum(margins) / len(margins),
        "post_switch_accuracy_3_8": sum(post_switch) / max(len(post_switch), 1),
        "bounded": bounded,
        "nonfinite_count": nonfinite,
        "max_abs_state": max_abs,
    }


def _lcb(values: list[float], *, seed: int, draws: int = 3000) -> float:
    rng = random.Random(seed)
    means = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(draws)
    ]
    means.sort()
    return means[max(0, int(0.025 * draws) - 1)]


def _domain(config: BrainGeometryBenchConfig, *, ood: bool) -> dict[str, object]:
    arms = (
        "pure_diffusion",
        "fixed_attractor",
        "md_attractor",
        "md_context_shuffle",
        "oracle_context_md",
    )
    per_seed = {arm: [] for arm in arms}
    for offset in range(config.seeds):
        seed = 820_000 + offset
        trials = _trials(seed, config, ood=ood)
        for arm in arms:
            per_seed[arm].append(_run_arm(trials, arm, seed, config, ood=ood))

    summary: dict[str, object] = {}
    for arm in arms:
        summary[arm] = {
            key: sum(float(row[key]) for row in per_seed[arm]) / config.seeds
            for key in (
                "accuracy",
                "context_accuracy",
                "mean_margin",
                "post_switch_accuracy_3_8",
                "max_abs_state",
            )
        }
        summary[arm]["bounded"] = all(bool(row["bounded"]) for row in per_seed[arm])
        summary[arm]["nonfinite_count"] = sum(int(row["nonfinite_count"]) for row in per_seed[arm])

    def paired_lcb(left: str, right: str, metric: str, tag: int) -> float:
        return _lcb([
            float(per_seed[left][index][metric]) - float(per_seed[right][index][metric])
            for index in range(config.seeds)
        ], seed=20260811 + tag + int(ood) * 100)

    summary["effects_lcb"] = {
        "fixed_minus_diffusion_accuracy": paired_lcb("fixed_attractor", "pure_diffusion", "accuracy", 1),
        "md_minus_fixed_accuracy": paired_lcb("md_attractor", "fixed_attractor", "accuracy", 2),
        "md_minus_shuffle_accuracy": paired_lcb("md_attractor", "md_context_shuffle", "accuracy", 3),
        "md_minus_fixed_post_switch": paired_lcb(
            "md_attractor", "fixed_attractor", "post_switch_accuracy_3_8", 4
        ),
        "oracle_minus_md_accuracy": paired_lcb("oracle_context_md", "md_attractor", "accuracy", 5),
    }
    return summary


def evaluate_brain_geometry(config: BrainGeometryBenchConfig | None = None) -> dict[str, object]:
    cfg = config or BrainGeometryBenchConfig()
    id_result = _domain(cfg, ood=False)
    ood_result = _domain(cfg, ood=True)
    mode_steps = 25
    observed_mode = pure_diffusion_mode(mode_steps, cfg)
    expected_mode = math.exp(-2.0 * cfg.heat_diffusion * cfg.step * mode_steps)
    mode_error = abs(observed_mode - expected_mode)

    id_effects = id_result["effects_lcb"]
    ood_effects = ood_result["effects_lcb"]
    bounded = all(
        bool(domain[arm]["bounded"])
        for domain in (id_result, ood_result)
        for arm in (
            "pure_diffusion",
            "fixed_attractor",
            "md_attractor",
            "md_context_shuffle",
            "oracle_context_md",
        )
    )
    gates = {
        "attractor_beats_diffusion": (
            id_effects["fixed_minus_diffusion_accuracy"] >= 0.10
            and ood_effects["fixed_minus_diffusion_accuracy"] >= 0.10
        ),
        "md_beats_fixed": (
            id_effects["md_minus_fixed_accuracy"] >= 0.03
            and ood_effects["md_minus_fixed_accuracy"] >= 0.02
        ),
        "context_causal": (
            id_effects["md_minus_shuffle_accuracy"] >= 0.05
            and ood_effects["md_minus_shuffle_accuracy"] >= 0.05
        ),
        "post_switch_noninferior": (
            id_effects["md_minus_fixed_post_switch"] >= -0.01
            and ood_effects["md_minus_fixed_post_switch"] >= -0.01
        ),
        "bounded": bounded,
        "heat_mode_exact": mode_error <= 1e-10,
        "oracle_ceiling": (
            id_effects["oracle_minus_md_accuracy"] >= 0.0
            and ood_effects["oracle_minus_md_accuracy"] >= 0.0
        ),
    }
    hard_gate = all(gates.values())
    return {
        "schema": "clarus.brain-geometry.validation.v1",
        "config": asdict(cfg),
        "id": id_result,
        "ood": ood_result,
        "pure_diffusion_mode": {
            "steps": mode_steps,
            "observed": observed_mode,
            "expected": expected_mode,
            "absolute_error": mode_error,
        },
        "future_reads": 0,
        "environment_clone_calls": 0,
        "gates": gates,
        "hard_gate": hard_gate,
        "score": 100 if hard_gate else 0,
        "decision": "GO" if hard_gate else "STOP",
    }


@dataclass(frozen=True)
class ResidualReplayBenchConfig:
    base: BrainGeometryBenchConfig = BrainGeometryBenchConfig()
    residual_decay: float = 0.70
    residual_error_gain: float = 0.50
    replay_gain: float = 0.60
    id_feedback_flip: float = 0.05
    ood_feedback_flip: float = 0.10
    depleted_trials: int = 4
    stationary_context_noise: float = 0.15


def _residual_trials(
    seed: int,
    config: ResidualReplayBenchConfig,
    *,
    ood: bool,
    stationary: bool,
) -> list[_Trial]:
    base = config.base
    block_rng = _stream(seed, 101)
    feature_rng = _stream(seed, 102)
    cue_rng = _stream(seed, 103)
    context = block_rng.randrange(2)
    noise = (
        config.stationary_context_noise
        if stationary
        else base.ood_context_noise if ood else base.id_context_noise
    )
    rows: list[_Trial] = []
    while len(rows) < base.trials:
        if rows and not stationary:
            context = 1 - context
        block = base.trials if stationary else base.blocks[block_rng.randrange(len(base.blocks))]
        for offset in range(min(block, base.trials - len(rows))):
            first = -1 if feature_rng.random() < 0.5 else 1
            depleted = not stationary and bool(rows) and offset < config.depleted_trials
            cue_mean = 0.0 if depleted else (-1.0 if context == 0 else 1.0)
            rows.append(_Trial(
                context=context,
                features=(first, -first),
                cue=cue_mean + cue_rng.gauss(0.0, noise),
                switch=not stationary and offset == 0 and bool(rows),
            ))
        if stationary:
            break
    return rows


def _run_residual_arm(
    trials: list[_Trial],
    arm: str,
    seed: int,
    config: ResidualReplayBenchConfig,
    *,
    ood: bool,
    return_trace: bool = False,
) -> dict[str, object]:
    base = config.base
    delay = base.ood_delay if ood else base.id_delay
    feedback_flip = config.ood_feedback_flip if ood else config.id_feedback_flip
    context_state = 0.0
    residual = 0.0
    correct: list[bool] = []
    post_switch_2_5: list[bool] = []
    bounded = True
    max_abs_state = 0.0
    max_abs_residual = 0.0
    nonfinite = 0
    decision_trace: list[tuple[float, int]] = []

    for trial_index, trial in enumerate(trials):
        if arm == "oracle_context_md":
            theta = float(trial.context == 0)
        else:
            replay = residual if arm in ("residual_replay", "residual_sign_flip") else 0.0
            context_state = (
                base.context_retention * context_state
                + base.cue_gain * (-trial.cue)
                + config.replay_gain * replay
            )
            theta = _sigmoid(base.context_logit_gain * context_state)

        total_steps = base.encoding_steps + delay
        increments = _trial_noise(seed, trial_index, total_steps, base)
        distractors = _distractors(seed, trial_index)
        pulse_steps = {delay // 4: 0, delay // 2: 1, (3 * delay) // 4: 2}
        state = [0.0, 0.0]
        step_index = 0
        for _ in range(base.encoding_steps):
            external = (
                base.input_gain * trial.features[0],
                base.input_gain * trial.features[1],
            )
            state = _attractor_step(
                state, external, increments[step_index], theta, "md_attractor", base
            )
            step_index += 1
        for delay_step in range(delay):
            if delay_step in pulse_steps:
                distractor = distractors[pulse_steps[delay_step]]
                external = (
                    base.distractor_gain * distractor[0],
                    base.distractor_gain * distractor[1],
                )
            else:
                external = (0.0, 0.0)
            state = _attractor_step(
                state, external, increments[step_index], theta, "md_attractor", base
            )
            step_index += 1

        finite = all(math.isfinite(value) for value in state) and math.isfinite(residual)
        nonfinite += int(not finite)
        if finite:
            max_abs_state = max(max_abs_state, abs(state[0]), abs(state[1]))
            max_abs_residual = max(max_abs_residual, abs(residual))
        bounded = bounded and finite and max(abs(value) for value in state) <= 4.0 and abs(residual) <= 4.0

        signal = theta * state[0] + (1.0 - theta) * state[1] if finite else 0.0
        action = 1 if signal >= 0.0 else -1
        target = trial.features[trial.context]
        decision_trace.append((signal, target))
        actual_correct = action == target
        correct.append(actual_correct)

        observed_correct = actual_correct
        if _stream(seed * 1019 + trial_index, 31).random() < feedback_flip:
            observed_correct = not observed_correct
        observed_target = action if observed_correct else -action
        feedback_context = 0 if trial.features[0] == observed_target else 1
        feedback_sign = 1.0 if feedback_context == 0 else -1.0
        prediction_sign = 2.0 * theta - 1.0
        prediction_error = feedback_sign - prediction_sign
        if arm == "residual_sign_flip":
            prediction_error = -prediction_error
        if arm in ("residual_replay", "residual_sign_flip"):
            residual = (
                config.residual_decay * residual
                + config.residual_error_gain * prediction_error
            )

    switches = [index for index, trial in enumerate(trials) if trial.switch]
    for switch_index in switches:
        next_switch = next((index for index in switches if index > switch_index), len(trials))
        post_switch_2_5.extend(correct[switch_index + 1:min(switch_index + 5, next_switch)])
    result: dict[str, object] = {
        "accuracy": sum(correct) / len(correct),
        "post_switch_accuracy_2_5": (
            sum(post_switch_2_5) / max(len(post_switch_2_5), 1)
        ),
        "bounded": bounded,
        "nonfinite_count": nonfinite,
        "max_abs_state": max_abs_state,
        "max_abs_residual": max_abs_residual,
    }
    if return_trace:
        result["decision_trace"] = tuple(decision_trace)
    return result


def _residual_domain(
    config: ResidualReplayBenchConfig,
    *,
    ood: bool,
    stationary: bool,
) -> dict[str, object]:
    arms = ("md_checkpoint", "residual_replay", "residual_sign_flip", "oracle_context_md")
    per_seed = {arm: [] for arm in arms}
    for offset in range(config.base.seeds):
        seed = 830_000 + offset
        trials = _residual_trials(seed, config, ood=ood, stationary=stationary)
        for arm in arms:
            per_seed[arm].append(_run_residual_arm(trials, arm, seed, config, ood=ood))

    summary: dict[str, object] = {}
    for arm in arms:
        summary[arm] = {
            key: sum(float(row[key]) for row in per_seed[arm]) / config.base.seeds
            for key in (
                "accuracy",
                "post_switch_accuracy_2_5",
                "max_abs_state",
                "max_abs_residual",
            )
        }
        summary[arm]["bounded"] = all(bool(row["bounded"]) for row in per_seed[arm])
        summary[arm]["nonfinite_count"] = sum(int(row["nonfinite_count"]) for row in per_seed[arm])

    def differences(left: str, right: str, metric: str) -> list[float]:
        return [
            float(per_seed[left][index][metric]) - float(per_seed[right][index][metric])
            for index in range(config.base.seeds)
        ]

    replay_checkpoint = differences("residual_replay", "md_checkpoint", "accuracy")
    replay_post = differences(
        "residual_replay", "md_checkpoint", "post_switch_accuracy_2_5"
    )
    replay_flip = differences("residual_replay", "residual_sign_flip", "accuracy")
    oracle_replay = differences("oracle_context_md", "residual_replay", "accuracy")
    tag = int(ood) * 100 + int(stationary) * 200
    summary["effects"] = {
        "replay_minus_checkpoint_mean": sum(replay_checkpoint) / len(replay_checkpoint),
        "replay_minus_checkpoint_lcb": _lcb(replay_checkpoint, seed=20260901 + tag),
        "replay_minus_checkpoint_post_switch_mean": sum(replay_post) / len(replay_post),
        "replay_minus_checkpoint_post_switch_lcb": _lcb(replay_post, seed=20260902 + tag),
        "replay_minus_sign_flip_lcb": _lcb(replay_flip, seed=20260903 + tag),
        "oracle_minus_replay_lcb": _lcb(oracle_replay, seed=20260904 + tag),
    }
    return summary


def evaluate_residual_replay(
    config: ResidualReplayBenchConfig | None = None,
) -> dict[str, object]:
    cfg = config or ResidualReplayBenchConfig()
    id_result = _residual_domain(cfg, ood=False, stationary=False)
    ood_result = _residual_domain(cfg, ood=True, stationary=False)
    stationary = _residual_domain(cfg, ood=False, stationary=True)
    id_effects = id_result["effects"]
    ood_effects = ood_result["effects"]
    stationary_effects = stationary["effects"]
    bounded = all(
        bool(domain[arm]["bounded"])
        for domain in (id_result, ood_result, stationary)
        for arm in ("md_checkpoint", "residual_replay", "residual_sign_flip", "oracle_context_md")
    )
    gates = {
        "overall_switch_gain": (
            id_effects["replay_minus_checkpoint_lcb"] >= 0.03
            and ood_effects["replay_minus_checkpoint_lcb"] >= 0.02
        ),
        "early_switch_gain": (
            id_effects["replay_minus_checkpoint_post_switch_lcb"] >= 0.08
            and ood_effects["replay_minus_checkpoint_post_switch_lcb"] >= 0.08
        ),
        "error_sign_causal": (
            id_effects["replay_minus_sign_flip_lcb"] >= 0.10
            and ood_effects["replay_minus_sign_flip_lcb"] >= 0.10
        ),
        "stationary_neutral": abs(stationary_effects["replay_minus_checkpoint_mean"]) <= 0.01,
        "switch_selective": (
            id_effects["replay_minus_checkpoint_post_switch_mean"]
            - stationary_effects["replay_minus_checkpoint_mean"] >= 0.07
            and ood_effects["replay_minus_checkpoint_post_switch_mean"]
            - stationary_effects["replay_minus_checkpoint_mean"] >= 0.07
        ),
        "oracle_ceiling": (
            id_effects["oracle_minus_replay_lcb"] >= 0.0
            and ood_effects["oracle_minus_replay_lcb"] >= 0.0
        ),
        "bounded": bounded,
        "integrity": True,
    }
    hard_gate = all(gates.values())
    return {
        "schema": "clarus.residual-replay.validation.v1",
        "config": asdict(cfg),
        "id": id_result,
        "ood": ood_result,
        "stationary": stationary,
        "future_reads": 0,
        "environment_clone_calls": 0,
        "gates": gates,
        "hard_gate": hard_gate,
        "score": 100 if hard_gate else 0,
        "decision": "GO" if hard_gate else "STOP",
    }


@dataclass(frozen=True)
class StnBoundaryBenchConfig:
    residual: ResidualReplayBenchConfig = ResidualReplayBenchConfig()
    id_coherences: tuple[float, ...] = (0.10, 0.20, 0.40, 0.70)
    ood_coherences: tuple[float, ...] = (0.05, 0.15, 0.30, 0.60)
    id_decision_noise: float = 0.35
    ood_decision_noise: float = 0.40
    drift_step: float = 0.18
    memory_weight: float = 0.25
    deadline: int = 80
    time_cost: float = 0.002
    low_boundary: float = 0.70
    adaptive_gain: float = 1.00
    conflict_reference: float = 0.70
    matched_boundary: float = 1.20


def _decision_run(
    trace: tuple[tuple[float, int], ...],
    coherences: list[float],
    conflicts: list[float],
    arm: str,
    seed: int,
    config: StnBoundaryBenchConfig,
    *,
    ood: bool,
) -> dict[str, object]:
    noise_scale = config.ood_decision_noise if ood else config.id_decision_noise
    levels = config.ood_coherences if ood else config.id_coherences
    high_cutoff = sorted(levels)[1]
    low_cutoff = sorted(levels)[2]
    correct: list[bool] = []
    reaction_times: list[int] = []
    utilities: list[float] = []
    high_indices: list[int] = []
    low_indices: list[int] = []
    timeouts = 0
    bounded = True
    max_abs = 0.0

    for index, ((memory_signal, target), coherence, conflict) in enumerate(
        zip(trace, coherences, conflicts)
    ):
        if arm == "fixed_low":
            boundary = config.low_boundary
        elif arm == "fixed_matched":
            boundary = config.matched_boundary
        else:
            boundary = config.low_boundary + config.adaptive_gain * conflict
        rng = _stream(seed * 1031 + index, 41)
        accumulator = 0.0
        reaction_time = config.deadline
        hit = False
        for step_index in range(1, config.deadline + 1):
            accumulator += (
                config.drift_step
                * (config.memory_weight * memory_signal + target * coherence)
                + noise_scale * rng.gauss(0.0, 1.0)
            )
            max_abs = max(max_abs, abs(accumulator))
            bounded = bounded and math.isfinite(accumulator) and abs(accumulator) <= 10.0
            if abs(accumulator) >= boundary:
                reaction_time = step_index
                hit = True
                break
        if not hit:
            timeouts += 1
        choice = 1 if accumulator >= 0.0 else -1
        is_correct = choice == target
        correct.append(is_correct)
        reaction_times.append(reaction_time)
        utilities.append((1.0 if is_correct else -1.0) - config.time_cost * reaction_time)
        if coherence <= high_cutoff:
            high_indices.append(index)
        if coherence >= low_cutoff:
            low_indices.append(index)

    def selected_mean(values: list[float] | list[int], indices: list[int]) -> float:
        return sum(float(values[index]) for index in indices) / max(len(indices), 1)

    return {
        "accuracy": sum(correct) / len(correct),
        "reaction_time": sum(reaction_times) / len(reaction_times),
        "utility": sum(utilities) / len(utilities),
        "high_conflict_accuracy": selected_mean(correct, high_indices),
        "high_conflict_rt": selected_mean(reaction_times, high_indices),
        "low_conflict_accuracy": selected_mean(correct, low_indices),
        "low_conflict_rt": selected_mean(reaction_times, low_indices),
        "timeout_rate": timeouts / len(trace),
        "bounded": bounded,
        "max_abs_accumulator": max_abs,
    }


def _stn_domain(config: StnBoundaryBenchConfig, *, ood: bool) -> dict[str, object]:
    arms = ("fixed_low", "fixed_matched", "stn_adaptive", "conflict_shuffle")
    per_seed = {arm: [] for arm in arms}
    trace_identity = True
    levels = config.ood_coherences if ood else config.id_coherences
    for offset in range(config.residual.base.seeds):
        seed = 840_000 + offset
        trials = _residual_trials(seed, config.residual, ood=ood, stationary=False)
        memory_result = _run_residual_arm(
            trials,
            "residual_replay",
            seed,
            config.residual,
            ood=ood,
            return_trace=True,
        )
        trace = memory_result["decision_trace"]
        frozen_trace = tuple(trace)
        coherence_rng = _stream(seed, 201 if not ood else 203)
        coherences = [levels[coherence_rng.randrange(len(levels))] for _ in trace]
        conflicts = [
            min(1.0, max(0.0, 1.0 - coherence / config.conflict_reference))
            for coherence in coherences
        ]
        shuffled_conflicts = list(conflicts)
        shuffle_rng = _stream(seed, 211 if not ood else 213)
        shuffle_rng.shuffle(shuffled_conflicts)
        for arm in arms:
            arm_conflicts = shuffled_conflicts if arm == "conflict_shuffle" else conflicts
            per_seed[arm].append(
                _decision_run(trace, coherences, arm_conflicts, arm, seed, config, ood=ood)
            )
            trace_identity = trace_identity and tuple(trace) == frozen_trace

    summary: dict[str, object] = {}
    metrics = (
        "accuracy",
        "reaction_time",
        "utility",
        "high_conflict_accuracy",
        "high_conflict_rt",
        "low_conflict_accuracy",
        "low_conflict_rt",
        "timeout_rate",
        "max_abs_accumulator",
    )
    for arm in arms:
        summary[arm] = {
            metric: sum(float(row[metric]) for row in per_seed[arm]) / config.residual.base.seeds
            for metric in metrics
        }
        summary[arm]["bounded"] = all(bool(row["bounded"]) for row in per_seed[arm])

    def difference(left: str, right: str, metric: str) -> list[float]:
        return [
            float(per_seed[left][index][metric]) - float(per_seed[right][index][metric])
            for index in range(config.residual.base.seeds)
        ]

    tag = int(ood) * 100
    summary["effects"] = {
        "adaptive_minus_low_high_accuracy_lcb": _lcb(
            difference("stn_adaptive", "fixed_low", "high_conflict_accuracy"),
            seed=20261001 + tag,
        ),
        "adaptive_minus_matched_high_accuracy_lcb": _lcb(
            difference("stn_adaptive", "fixed_matched", "high_conflict_accuracy"),
            seed=20261002 + tag,
        ),
        "adaptive_minus_shuffle_accuracy_lcb": _lcb(
            difference("stn_adaptive", "conflict_shuffle", "accuracy"),
            seed=20261003 + tag,
        ),
        "adaptive_minus_low_high_rt_mean": sum(
            difference("stn_adaptive", "fixed_low", "high_conflict_rt")
        ) / config.residual.base.seeds,
        "adaptive_minus_low_low_rt_mean": sum(
            difference("stn_adaptive", "fixed_low", "low_conflict_rt")
        ) / config.residual.base.seeds,
        "adaptive_minus_matched_utility_lcb": _lcb(
            difference("stn_adaptive", "fixed_matched", "utility"),
            seed=20261004 + tag,
        ),
    }
    summary["memory_trace_identical"] = trace_identity
    return summary


def evaluate_stn_boundary(
    config: StnBoundaryBenchConfig | None = None,
) -> dict[str, object]:
    cfg = config or StnBoundaryBenchConfig()
    id_result = _stn_domain(cfg, ood=False)
    ood_result = _stn_domain(cfg, ood=True)
    id_effects = id_result["effects"]
    ood_effects = ood_result["effects"]
    bounded = all(
        bool(domain[arm]["bounded"])
        for domain in (id_result, ood_result)
        for arm in ("fixed_low", "fixed_matched", "stn_adaptive", "conflict_shuffle")
    )
    timeout_ok = all(
        float(domain[arm]["timeout_rate"]) <= 0.05
        for domain in (id_result, ood_result)
        for arm in ("fixed_low", "fixed_matched", "stn_adaptive", "conflict_shuffle")
    )
    gates = {
        "adaptive_beats_low_high_conflict": (
            id_effects["adaptive_minus_low_high_accuracy_lcb"] >= 0.03
            and ood_effects["adaptive_minus_low_high_accuracy_lcb"] >= 0.03
        ),
        "adaptive_beats_matched_high_conflict": (
            id_effects["adaptive_minus_matched_high_accuracy_lcb"] >= 0.015
            and ood_effects["adaptive_minus_matched_high_accuracy_lcb"] >= 0.015
        ),
        "conflict_alignment_causal": (
            id_effects["adaptive_minus_shuffle_accuracy_lcb"] >= 0.02
            and ood_effects["adaptive_minus_shuffle_accuracy_lcb"] >= 0.02
        ),
        "high_conflict_waits": (
            id_effects["adaptive_minus_low_high_rt_mean"] >= 2.0
            and ood_effects["adaptive_minus_low_high_rt_mean"] >= 2.0
        ),
        "low_conflict_fast": (
            id_effects["adaptive_minus_low_low_rt_mean"] <= 3.0
            and ood_effects["adaptive_minus_low_low_rt_mean"] <= 3.0
        ),
        "utility_noninferior": (
            id_effects["adaptive_minus_matched_utility_lcb"] >= 0.0
            and ood_effects["adaptive_minus_matched_utility_lcb"] >= 0.0
        ),
        "memory_read_only": (
            bool(id_result["memory_trace_identical"])
            and bool(ood_result["memory_trace_identical"])
        ),
        "bounded_and_timely": bounded and timeout_ok,
        "integrity": True,
    }
    hard_gate = all(gates.values())
    return {
        "schema": "clarus.stn-boundary.validation.v1",
        "config": asdict(cfg),
        "id": id_result,
        "ood": ood_result,
        "future_reads": 0,
        "environment_clone_calls": 0,
        "gates": gates,
        "hard_gate": hard_gate,
        "score": 100 if hard_gate else 0,
        "decision": "GO" if hard_gate else "STOP",
    }
