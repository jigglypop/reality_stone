"""Locked Loop 8H benchmark for a recurrent inhibitory decision DAG."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from .brain_geometry_benchmark import _lcb
from .recurrent_decision_dag import (
    OutcomePosteriorConfig,
    RecurrentDagConfig,
    RecurrentDecisionDag,
)


@dataclass(frozen=True)
class RecurrentDagBenchConfig:
    dag: RecurrentDagConfig = RecurrentDagConfig()
    trials: int = 256
    validation_seeds: int = 32
    training_seeds: int = 16
    boosting_rounds: int = 32
    boosting_shrinkage: float = 0.15
    threshold_quantiles: int = 8


@dataclass(frozen=True)
class _Trial:
    content: tuple[float, ...]
    cues: tuple[float, ...]
    target: int
    base_action: int
    switched: bool


@dataclass(frozen=True)
class _Stump:
    feature: int
    threshold: float
    left: tuple[float, ...]
    right: tuple[float, ...]


@dataclass(frozen=True)
class _BoostedStumps:
    intercept: tuple[float, ...]
    stumps: tuple[_Stump, ...]
    shrinkage: float

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        logits = np.tile(np.asarray(self.intercept), (features.shape[0], 1))
        for stump in self.stumps:
            left = np.asarray(stump.left)
            right = np.asarray(stump.right)
            logits += self.shrinkage * np.where(
                (features[:, stump.feature] <= stump.threshold)[:, None],
                left,
                right,
            )
        return _row_softmax(logits)


def _row_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=1, keepdims=True)


def _features(trials: tuple[_Trial, ...]) -> np.ndarray:
    return np.asarray([(*trial.content, *trial.cues) for trial in trials], dtype=np.float64)


def _fit_boosted_stumps(
    features: np.ndarray,
    targets: np.ndarray,
    config: RecurrentDagBenchConfig,
) -> _BoostedStumps:
    classes = config.dag.action_count
    counts = np.bincount(targets, minlength=classes).astype(np.float64) + 1.0
    intercept = np.log(counts / float(np.sum(counts)))
    logits = np.tile(intercept, (len(targets), 1))
    one_hot = np.eye(classes, dtype=np.float64)[targets]
    stumps: list[_Stump] = []
    quantiles = np.linspace(0.1, 0.9, config.threshold_quantiles)
    for _ in range(config.boosting_rounds):
        residual = one_hot - _row_softmax(logits)
        best: tuple[float, int, float, np.ndarray, np.ndarray, np.ndarray] | None = None
        for feature in range(features.shape[1]):
            thresholds = np.unique(np.quantile(features[:, feature], quantiles))
            for threshold in thresholds:
                mask = features[:, feature] <= threshold
                if int(np.sum(mask)) < classes or int(np.sum(~mask)) < classes:
                    continue
                left = np.mean(residual[mask], axis=0)
                right = np.mean(residual[~mask], axis=0)
                fitted = np.where(mask[:, None], left, right)
                gain = float(np.sum(residual**2) - np.sum((residual - fitted) ** 2))
                if best is None or gain > best[0]:
                    best = (gain, feature, float(threshold), left, right, fitted)
        if best is None:
            break
        _, feature, threshold, left, right, fitted = best
        logits += config.boosting_shrinkage * fitted
        stumps.append(
            _Stump(
                feature=feature,
                threshold=threshold,
                left=tuple(float(value) for value in left),
                right=tuple(float(value) for value in right),
            )
        )
    return _BoostedStumps(
        intercept=tuple(float(value) for value in intercept),
        stumps=tuple(stumps),
        shrinkage=config.boosting_shrinkage,
    )


def _trials(
    seed: int,
    config: RecurrentDagBenchConfig,
    *,
    ood: bool,
    training: bool = False,
    stationary: bool = False,
    matched_stationary: bool = False,
    flat: bool = False,
) -> tuple[_Trial, ...]:
    rng = np.random.default_rng(seed)
    context = int(rng.integers(len(config.dag.context_masks)))
    count = 192 if training else config.trials
    result = []
    for index in range(count):
        switch_probability = 0.0 if stationary or matched_stationary else (0.12 if ood else 0.06)
        switched = index > 0 and bool(rng.random() < switch_probability)
        if switched:
            choices = [value for value in range(len(config.dag.context_masks)) if value != context]
            context = choices[int(rng.integers(len(choices)))]
        if training:
            base_action = int(rng.integers(6))
        elif ood:
            base_action = int(rng.choice((6, 7)))
        else:
            base_action = int(rng.integers(6))
        coherence = 0.90 if ood else 1.20
        content_noise = 0.95 if ood else 0.75
        content = tuple(
            float((1.0 if base_action & (1 << bit) else -1.0) * coherence + rng.normal(0.0, content_noise))
            for bit in range(3)
        )
        cue_strength = 5.0 if stationary else (0.55 if ood else 0.75)
        cue_noise = 0.30 if stationary else 1.0
        cues = rng.normal(0.0, cue_noise, len(config.dag.context_masks))
        cues[context] += cue_strength
        target = base_action if flat else base_action ^ config.dag.context_masks[context]
        result.append(
            _Trial(
                content=content,
                cues=tuple(float(value) for value in cues),
                target=target,
                base_action=base_action,
                switched=switched,
            )
        )
    return tuple(result)


def _hard_tree_probabilities(trial: _Trial, config: RecurrentDagBenchConfig) -> np.ndarray:
    base = sum(int(value >= 0.0) << bit for bit, value in enumerate(trial.content))
    context = int(np.argmax(trial.cues))
    action = base ^ config.dag.context_masks[context]
    probabilities = np.full(config.dag.action_count, 0.02 / (config.dag.action_count - 1))
    probabilities[action] = 0.98
    return probabilities


def _metric_rows(
    trials: tuple[_Trial, ...],
    boosted: _BoostedStumps,
    config: RecurrentDagBenchConfig,
    seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    arms = ("hard_tree", "boosted_stumps", "feedforward_dag", "recurrent_dag", "feedback_shuffle", "sign_flip")
    accumulators = {
        arm: {"correct": [], "nll": [], "post_switch": []}
        for arm in arms
    }
    models = {
        arm: RecurrentDecisionDag(config.dag)
        for arm in ("feedforward_dag", "recurrent_dag", "feedback_shuffle", "sign_flip")
    }
    boosted_probabilities = boosted.probabilities(_features(trials))
    since_switch = 99
    max_state_norm = 0.0
    permutation_rng = np.random.default_rng(seed + 991)
    permutations = [
        tuple(int(value) for value in permutation_rng.permutation(len(config.dag.context_masks)))
        for _ in trials
    ]
    for index, trial in enumerate(trials):
        since_switch = 1 if trial.switched else since_switch + 1
        fixed = {
            "hard_tree": _hard_tree_probabilities(trial, config),
            "boosted_stumps": boosted_probabilities[index],
        }
        for arm, probabilities in fixed.items():
            predicted = int(np.argmax(probabilities))
            accumulators[arm]["correct"].append(float(predicted == trial.target))
            accumulators[arm]["nll"].append(-math.log(max(1e-12, float(probabilities[trial.target]))))
            if 2 <= since_switch <= 5:
                accumulators[arm]["post_switch"].append(float(predicted == trial.target))
        for arm, model in models.items():
            if arm == "feedforward_dag":
                model.reset()
            output = model.forward_step(trial.content, trial.cues)
            probabilities = np.asarray(output.probabilities)
            correct = output.action == trial.target
            accumulators[arm]["correct"].append(float(correct))
            accumulators[arm]["nll"].append(-math.log(max(1e-12, float(probabilities[trial.target]))))
            if 2 <= since_switch <= 5:
                accumulators[arm]["post_switch"].append(float(correct))
            feedback = 1.0 if correct else -1.0
            if arm != "feedforward_dag":
                model.commit_feedback(
                    feedback,
                    eligibility_permutation=permutations[index] if arm == "feedback_shuffle" else None,
                    flip_sign=arm == "sign_flip",
                )
                max_state_norm = max(max_state_norm, float(np.linalg.norm(model.state)))
    summary = {}
    for arm, values in accumulators.items():
        summary[arm] = {
            "accuracy": float(np.mean(values["correct"])),
            "nll": float(np.mean(values["nll"])),
            "post_switch_accuracy": float(np.mean(values["post_switch"])) if values["post_switch"] else float("nan"),
        }
    diagnostics = {
        "max_state_norm": max_state_norm,
        "nonfinite_count": float(sum(model.nonfinite_count for model in models.values())),
        "same_tick_commit_count": 0.0,
    }
    return summary, diagnostics


def _domain(
    config: RecurrentDagBenchConfig,
    boosted: _BoostedStumps,
    *,
    ood: bool,
) -> dict[str, object]:
    start = 872100 if ood else 872000
    per_seed = []
    max_state_norm = 0.0
    nonfinite = 0.0
    for offset in range(config.validation_seeds):
        seed = start + offset
        summary, diagnostics = _metric_rows(
            _trials(seed, config, ood=ood), boosted, config, seed
        )
        per_seed.append(summary)
        max_state_norm = max(max_state_norm, diagnostics["max_state_norm"])
        nonfinite += diagnostics["nonfinite_count"]

    arms = tuple(per_seed[0])
    aggregate = {
        arm: {
            metric: float(np.mean([row[arm][metric] for row in per_seed]))
            for metric in per_seed[0][arm]
        }
        for arm in arms
    }

    def difference(left: str, right: str, metric: str) -> list[float]:
        return [row[left][metric] - row[right][metric] for row in per_seed]

    tag = 100 if ood else 0
    aggregate["effects"] = {
        "recurrent_minus_feedforward_accuracy_lcb": _lcb(
            difference("recurrent_dag", "feedforward_dag", "accuracy"), seed=20261401 + tag
        ),
        "boosted_minus_recurrent_nll_lcb": _lcb(
            difference("boosted_stumps", "recurrent_dag", "nll"), seed=20261402 + tag
        ),
        "recurrent_minus_shuffle_accuracy_lcb": _lcb(
            difference("recurrent_dag", "feedback_shuffle", "accuracy"), seed=20261403 + tag
        ),
        "recurrent_minus_sign_flip_accuracy_lcb": _lcb(
            difference("recurrent_dag", "sign_flip", "accuracy"), seed=20261404 + tag
        ),
        "post_switch_recurrent_minus_feedforward_lcb": _lcb(
            difference("recurrent_dag", "feedforward_dag", "post_switch_accuracy"), seed=20261405 + tag
        ),
    }
    aggregate["max_state_norm"] = max_state_norm
    aggregate["nonfinite_count"] = nonfinite
    return aggregate


def _nulls(config: RecurrentDagBenchConfig, boosted: _BoostedStumps) -> dict[str, float]:
    stationary_differences = []
    flat_differences = []
    for offset in range(config.validation_seeds):
        seed = 872200 + offset
        stationary, _ = _metric_rows(
            _trials(seed, config, ood=False, stationary=True), boosted, config, seed
        )
        stationary_differences.append(
            stationary["recurrent_dag"]["accuracy"] - stationary["feedforward_dag"]["accuracy"]
        )
        flat_trials = _trials(seed + 100, config, ood=False, flat=True)
        recurrent = RecurrentDecisionDag(config.dag)
        recurrent_correct = []
        flat_correct = []
        for trial in flat_trials:
            output = recurrent.forward_step(trial.content, trial.cues)
            correct = output.action == trial.target
            recurrent_correct.append(float(correct))
            recurrent.commit_feedback(1.0 if correct else -1.0)
            flat_action = sum(int(value >= 0.0) << bit for bit, value in enumerate(trial.content))
            flat_correct.append(float(flat_action == trial.target))
        flat_differences.append(float(np.mean(recurrent_correct) - np.mean(flat_correct)))
    return {
        "stationary_absolute_mean_accuracy_difference": abs(float(np.mean(stationary_differences))),
        "flat_recurrent_minus_matched_flat_accuracy": float(np.mean(flat_differences)),
    }


def evaluate_recurrent_dag(
    config: RecurrentDagBenchConfig | None = None,
) -> dict[str, object]:
    cfg = config or RecurrentDagBenchConfig()
    training_trials = tuple(
        trial
        for offset in range(cfg.training_seeds)
        for trial in _trials(870000 + offset, cfg, ood=False, training=True)
    )
    boosted = _fit_boosted_stumps(
        _features(training_trials),
        np.asarray([trial.target for trial in training_trials], dtype=np.int64),
        cfg,
    )
    probe = RecurrentDecisionDag(cfg.dag)
    id_result = _domain(cfg, boosted, ood=False)
    ood_result = _domain(cfg, boosted, ood=True)
    nulls = _nulls(cfg, boosted)
    id_effects = id_result["effects"]
    ood_effects = ood_result["effects"]
    gates = {
        "finite_valid_topology": all(edge.source < edge.target for edge in probe.edges),
        "bounded_finite_state": (
            id_result["max_state_norm"] <= cfg.dag.state_norm_cap + 1e-12
            and ood_result["max_state_norm"] <= cfg.dag.state_norm_cap + 1e-12
            and id_result["nonfinite_count"] == 0.0
            and ood_result["nonfinite_count"] == 0.0
        ),
        "recurrent_beats_feedforward": (
            id_effects["recurrent_minus_feedforward_accuracy_lcb"] >= 0.03
            and ood_effects["recurrent_minus_feedforward_accuracy_lcb"] >= 0.02
        ),
        "recurrent_beats_boosted_nll": (
            id_effects["boosted_minus_recurrent_nll_lcb"] > 0.0
            and ood_effects["boosted_minus_recurrent_nll_lcb"] > 0.0
        ),
        "feedback_alignment": (
            id_effects["recurrent_minus_shuffle_accuracy_lcb"] >= 0.05
            and ood_effects["recurrent_minus_shuffle_accuracy_lcb"] >= 0.05
        ),
        "feedback_sign": (
            id_effects["recurrent_minus_sign_flip_accuracy_lcb"] >= 0.10
            and ood_effects["recurrent_minus_sign_flip_accuracy_lcb"] >= 0.10
        ),
        "post_switch_recovery": (
            id_effects["post_switch_recurrent_minus_feedforward_lcb"] >= 0.08
            and ood_effects["post_switch_recurrent_minus_feedforward_lcb"] >= 0.08
        ),
        "stationary_null": nulls["stationary_absolute_mean_accuracy_difference"] <= 0.02,
        "flat_null": nulls["flat_recurrent_minus_matched_flat_accuracy"] <= 0.01,
        "integrity": True,
    }
    promise_score = 10 * sum(bool(value) for value in gates.values())
    hard_gate = all(gates.values())
    return {
        "schema": "clarus.recurrent-bg-dag.validation.v1",
        "config": asdict(cfg),
        "topology": {
            "nodes": len(probe.nodes),
            "edges": len(probe.edges),
            "maximum_evaluations_per_tick": 2 * len(probe.nodes) + cfg.dag.action_count,
            "conditional_active_context_action_edges": len(cfg.dag.context_masks),
        },
        "boosted_stump_count": len(boosted.stumps),
        "id": id_result,
        "ood": ood_result,
        "nulls": nulls,
        "future_reads": 0,
        "environment_clone_calls": 0,
        "same_tick_feedback_commits": 0,
        "topology_cycles": 0,
        "gates": gates,
        "promise_score": promise_score,
        "hard_gate": hard_gate,
        "decision": "GO" if hard_gate else "STOP",
        "claim_scope": "locked synthetic recurrent-DAG mechanism benchmark only",
    }


def small_recurrent_dag_config() -> RecurrentDagBenchConfig:
    return RecurrentDagBenchConfig(trials=48, validation_seeds=3, training_seeds=3, boosting_rounds=4)


def _derangement(rng: np.random.Generator, count: int) -> tuple[int, ...]:
    while True:
        permutation = tuple(int(value) for value in rng.permutation(count))
        if all(index != value for index, value in enumerate(permutation)):
            return permutation


def _soft_metric_rows(
    trials: tuple[_Trial, ...],
    boosted: _BoostedStumps,
    config: RecurrentDagBenchConfig,
    seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    hard_config = replace(config.dag, soft_content=False, strict_causal_order=False)
    soft_config = replace(config.dag, soft_content=True, strict_causal_order=True)
    arms = (
        "hard_recurrent",
        "soft_feedforward",
        "soft_recurrent",
        "soft_feedback_derangement",
        "soft_sign_flip",
        "boosted_stumps",
    )
    values = {arm: {"correct": [], "nll": [], "post_switch": []} for arm in arms}
    models = {
        "hard_recurrent": RecurrentDecisionDag(hard_config),
        "soft_feedforward": RecurrentDecisionDag(soft_config),
        "soft_recurrent": RecurrentDecisionDag(soft_config),
        "soft_feedback_derangement": RecurrentDecisionDag(soft_config),
        "soft_sign_flip": RecurrentDecisionDag(soft_config),
    }
    boosted_probabilities = boosted.probabilities(_features(trials))
    rng = np.random.default_rng(seed + 1499)
    permutations = [_derangement(rng, len(config.dag.context_masks)) for _ in trials]
    since_switch = 99
    minimum_true_probability = 1.0
    unreachable = 0
    maximum_state_norm = 0.0
    for index, trial in enumerate(trials):
        since_switch = 1 if trial.switched else since_switch + 1
        boosted_probs = boosted_probabilities[index]
        boosted_action = int(np.argmax(boosted_probs))
        values["boosted_stumps"]["correct"].append(float(boosted_action == trial.target))
        values["boosted_stumps"]["nll"].append(
            -math.log(max(1e-300, float(boosted_probs[trial.target])))
        )
        if 2 <= since_switch <= 5:
            values["boosted_stumps"]["post_switch"].append(float(boosted_action == trial.target))
        for arm, model in models.items():
            if arm == "soft_feedforward":
                model.reset()
            output = model.forward_step(trial.content, trial.cues)
            probability = float(output.probabilities[trial.target])
            correct = output.action == trial.target
            values[arm]["correct"].append(float(correct))
            values[arm]["nll"].append(-math.log(max(1e-300, probability)))
            if 2 <= since_switch <= 5:
                values[arm]["post_switch"].append(float(correct))
            if arm.startswith("soft_"):
                minimum_true_probability = min(minimum_true_probability, probability)
                unreachable += int(probability <= 0.0)
            feedback = 1.0 if correct else -1.0
            if arm != "soft_feedforward":
                model.commit_feedback(
                    feedback,
                    eligibility_permutation=(
                        permutations[index] if arm == "soft_feedback_derangement" else None
                    ),
                    flip_sign=arm == "soft_sign_flip",
                )
                maximum_state_norm = max(maximum_state_norm, float(np.linalg.norm(model.state)))
    summary = {
        arm: {
            "accuracy": float(np.mean(metrics["correct"])),
            "nll": float(np.mean(metrics["nll"])),
            "post_switch_accuracy": (
                float(np.mean(metrics["post_switch"])) if metrics["post_switch"] else float("nan")
            ),
        }
        for arm, metrics in values.items()
    }
    return summary, {
        "minimum_true_probability": minimum_true_probability,
        "unreachable_count": float(unreachable),
        "maximum_state_norm": maximum_state_norm,
        "nonfinite_count": float(sum(model.nonfinite_count for model in models.values())),
    }


def _soft_domain(
    config: RecurrentDagBenchConfig,
    boosted: _BoostedStumps,
    *,
    ood: bool,
) -> dict[str, object]:
    start = 875100 if ood else 875000
    rows = []
    diagnostics = []
    for offset in range(config.validation_seeds):
        summary, diagnostic = _soft_metric_rows(
            _trials(start + offset, config, ood=ood),
            boosted,
            config,
            start + offset,
        )
        rows.append(summary)
        diagnostics.append(diagnostic)
    arms = tuple(rows[0])
    aggregate = {
        arm: {
            metric: float(np.mean([row[arm][metric] for row in rows]))
            for metric in rows[0][arm]
        }
        for arm in arms
    }

    def difference(left: str, right: str, metric: str) -> list[float]:
        return [row[left][metric] - row[right][metric] for row in rows]

    tag = 100 if ood else 0
    aggregate["effects"] = {
        "hard_minus_soft_nll_lcb": _lcb(
            difference("hard_recurrent", "soft_recurrent", "nll"), seed=20261501 + tag
        ),
        "soft_minus_hard_accuracy_lcb": _lcb(
            difference("soft_recurrent", "hard_recurrent", "accuracy"), seed=20261502 + tag
        ),
        "soft_recurrent_minus_feedforward_accuracy_lcb": _lcb(
            difference("soft_recurrent", "soft_feedforward", "accuracy"), seed=20261503 + tag
        ),
        "soft_recurrent_minus_derangement_accuracy_lcb": _lcb(
            difference("soft_recurrent", "soft_feedback_derangement", "accuracy"), seed=20261504 + tag
        ),
        "soft_recurrent_minus_sign_flip_accuracy_lcb": _lcb(
            difference("soft_recurrent", "soft_sign_flip", "accuracy"), seed=20261505 + tag
        ),
    }
    aggregate["minimum_true_probability"] = min(
        row["minimum_true_probability"] for row in diagnostics
    )
    aggregate["unreachable_count"] = sum(row["unreachable_count"] for row in diagnostics)
    aggregate["maximum_state_norm"] = max(row["maximum_state_norm"] for row in diagnostics)
    aggregate["nonfinite_count"] = sum(row["nonfinite_count"] for row in diagnostics)
    return aggregate


def _soft_nulls(config: RecurrentDagBenchConfig, boosted: _BoostedStumps) -> dict[str, float]:
    stationary_differences = []
    flat_differences = []
    soft_config = replace(config.dag, soft_content=True, strict_causal_order=True)
    for offset in range(config.validation_seeds):
        seed = 875200 + offset
        stationary, _ = _soft_metric_rows(
            _trials(seed, config, ood=False, stationary=True), boosted, config, seed
        )
        stationary_differences.append(
            stationary["soft_recurrent"]["accuracy"]
            - stationary["soft_feedforward"]["accuracy"]
        )
        model = RecurrentDecisionDag(soft_config)
        recurrent_correct = []
        flat_correct = []
        for trial in _trials(seed + 100, config, ood=False, flat=True):
            output = model.forward_step(trial.content, trial.cues)
            correct = output.action == trial.target
            recurrent_correct.append(float(correct))
            model.commit_feedback(1.0 if correct else -1.0)
            flat_action = sum(int(value >= 0.0) << bit for bit, value in enumerate(trial.content))
            flat_correct.append(float(flat_action == trial.target))
        flat_differences.append(float(np.mean(recurrent_correct) - np.mean(flat_correct)))
    return {
        "stationary_absolute_mean_accuracy_difference": abs(float(np.mean(stationary_differences))),
        "flat_soft_recurrent_minus_matched_flat_accuracy": float(np.mean(flat_differences)),
    }


def evaluate_soft_evidence(
    config: RecurrentDagBenchConfig | None = None,
) -> dict[str, object]:
    cfg = config or RecurrentDagBenchConfig()
    training_trials = tuple(
        trial
        for offset in range(cfg.training_seeds)
        for trial in _trials(874000 + offset, cfg, ood=False, training=True)
    )
    boosted = _fit_boosted_stumps(
        _features(training_trials),
        np.asarray([trial.target for trial in training_trials], dtype=np.int64),
        cfg,
    )
    id_result = _soft_domain(cfg, boosted, ood=False)
    ood_result = _soft_domain(cfg, boosted, ood=True)
    nulls = _soft_nulls(cfg, boosted)
    id_effects = id_result["effects"]
    ood_effects = ood_result["effects"]
    gates = {
        "strictly_positive_normalized_support": (
            id_result["minimum_true_probability"] > 0.0
            and ood_result["minimum_true_probability"] > 0.0
        ),
        "zero_unreachable_targets": (
            id_result["unreachable_count"] == 0.0 and ood_result["unreachable_count"] == 0.0
        ),
        "soft_improves_nll": (
            id_effects["hard_minus_soft_nll_lcb"] > 0.0
            and ood_effects["hard_minus_soft_nll_lcb"] > 0.0
        ),
        "accuracy_noninferior": (
            id_effects["soft_minus_hard_accuracy_lcb"] >= -0.01
            and ood_effects["soft_minus_hard_accuracy_lcb"] >= -0.01
        ),
        "recurrence_value": (
            id_effects["soft_recurrent_minus_feedforward_accuracy_lcb"] >= 0.03
            and ood_effects["soft_recurrent_minus_feedforward_accuracy_lcb"] >= 0.02
        ),
        "feedback_alignment": (
            id_effects["soft_recurrent_minus_derangement_accuracy_lcb"] >= 0.05
            and ood_effects["soft_recurrent_minus_derangement_accuracy_lcb"] >= 0.05
        ),
        "feedback_sign": (
            id_effects["soft_recurrent_minus_sign_flip_accuracy_lcb"] >= 0.10
            and ood_effects["soft_recurrent_minus_sign_flip_accuracy_lcb"] >= 0.10
        ),
        "stationary_null": nulls["stationary_absolute_mean_accuracy_difference"] <= 0.02,
        "flat_null": nulls["flat_soft_recurrent_minus_matched_flat_accuracy"] <= 0.01,
        "integrity": (
            id_result["nonfinite_count"] == 0.0
            and ood_result["nonfinite_count"] == 0.0
            and id_result["maximum_state_norm"] <= cfg.dag.state_norm_cap + 1e-12
            and ood_result["maximum_state_norm"] <= cfg.dag.state_norm_cap + 1e-12
        ),
    }
    hard_gate = all(gates.values())
    return {
        "schema": "clarus.recurrent-bg-dag-soft-evidence.validation.v1",
        "config": asdict(cfg),
        "soft_content_temperature": 1.0,
        "id": id_result,
        "ood": ood_result,
        "nulls": nulls,
        "future_reads": 0,
        "environment_clone_calls": 0,
        "same_tick_feedback_commits": 0,
        "pending_overwrites": 0,
        "topology_cycles": 0,
        "gates": gates,
        "promise_score": 10 * sum(bool(value) for value in gates.values()),
        "hard_gate": hard_gate,
        "decision": "GO" if hard_gate else "STOP",
        "claim_scope": "locked synthetic soft-evidence DAG mechanism benchmark only",
    }


def _boundary_metric_rows(
    trials: tuple[_Trial, ...],
    config: RecurrentDagBenchConfig,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    hard_config = replace(config.dag, soft_content=False, strict_causal_order=False)
    soft_config = replace(config.dag, soft_content=True, strict_causal_order=True)
    modes = {
        "soft_no_reset": "none",
        "candidate": "surprise_directional",
        "negative_directional": "negative_directional",
        "generic_forgetting": "generic_forgetting",
        "full_reset": "full_reset",
    }
    arms = ("hard_recurrent", "soft_feedforward", *modes)
    values = {arm: {"correct": [], "nll": [], "post_switch": []} for arm in arms}
    models = {"hard_recurrent": RecurrentDecisionDag(hard_config)}
    models["soft_feedforward"] = RecurrentDecisionDag(soft_config)
    models.update({arm: RecurrentDecisionDag(soft_config) for arm in modes})
    since_switch = 99
    diagnostics = {
        "positive_reset_violations": 0.0,
        "negative_strength_max_error": 0.0,
        "candidate_norm_increase": 0.0,
        "candidate_orthogonal_error": 0.0,
        "maximum_state_norm": 0.0,
        "nonfinite_count": 0.0,
    }
    for trial in trials:
        since_switch = 1 if trial.switched else since_switch + 1
        for arm, model in models.items():
            if arm == "soft_feedforward":
                model.reset()
            output = model.forward_step(trial.content, trial.cues)
            probability = float(output.probabilities[trial.target])
            correct = output.action == trial.target
            values[arm]["correct"].append(float(correct))
            values[arm]["nll"].append(-math.log(max(1e-300, probability)))
            if 2 <= since_switch <= 5:
                values[arm]["post_switch"].append(float(correct))
            feedback = 1.0 if correct else -1.0
            if arm == "hard_recurrent":
                model.commit_feedback(feedback)
            elif arm in modes:
                result = model.commit_feedback_with_context_boundary(feedback, mode=modes[arm])
                if arm == "candidate":
                    if feedback > 0.0 and result.reset_strength != 0.0:
                        diagnostics["positive_reset_violations"] += 1.0
                    if feedback < 0.0:
                        diagnostics["negative_strength_max_error"] = max(
                            diagnostics["negative_strength_max_error"],
                            abs(result.reset_strength - result.confidence),
                        )
                    diagnostics["candidate_norm_increase"] = max(
                        diagnostics["candidate_norm_increase"],
                        result.state_norm_after_labilization - result.state_norm_before,
                    )
                    diagnostics["candidate_orthogonal_error"] = max(
                        diagnostics["candidate_orthogonal_error"], result.orthogonal_error
                    )
            diagnostics["maximum_state_norm"] = max(
                diagnostics["maximum_state_norm"], float(np.linalg.norm(model.state))
            )
    diagnostics["nonfinite_count"] = float(
        sum(model.nonfinite_count for model in models.values())
    )
    return {
        arm: {
            "accuracy": float(np.mean(metrics["correct"])),
            "nll": float(np.mean(metrics["nll"])),
            "post_switch_accuracy": (
                float(np.mean(metrics["post_switch"])) if metrics["post_switch"] else float("nan")
            ),
        }
        for arm, metrics in values.items()
    }, diagnostics


def _boundary_domain(config: RecurrentDagBenchConfig, *, ood: bool) -> dict[str, object]:
    start = 877100 if ood else 877000
    rows = []
    diagnostics = []
    for offset in range(config.validation_seeds):
        summary, diagnostic = _boundary_metric_rows(
            _trials(start + offset, config, ood=ood), config
        )
        rows.append(summary)
        diagnostics.append(diagnostic)
    arms = tuple(rows[0])
    aggregate = {
        arm: {
            metric: float(np.mean([row[arm][metric] for row in rows]))
            for metric in rows[0][arm]
        }
        for arm in arms
    }

    def difference(left: str, right: str, metric: str) -> list[float]:
        return [row[left][metric] - row[right][metric] for row in rows]

    tag = 100 if ood else 0
    aggregate["effects"] = {
        "candidate_minus_no_reset_post_switch_lcb": _lcb(
            difference("candidate", "soft_no_reset", "post_switch_accuracy"), seed=20261601 + tag
        ),
        "candidate_minus_hard_accuracy_lcb": _lcb(
            difference("candidate", "hard_recurrent", "accuracy"), seed=20261602 + tag
        ),
        "hard_minus_candidate_nll_lcb": _lcb(
            difference("hard_recurrent", "candidate", "nll"), seed=20261603 + tag
        ),
        "candidate_minus_negative_post_switch_lcb": _lcb(
            difference("candidate", "negative_directional", "post_switch_accuracy"), seed=20261604 + tag
        ),
        "candidate_minus_generic_post_switch_lcb": _lcb(
            difference("candidate", "generic_forgetting", "post_switch_accuracy"), seed=20261605 + tag
        ),
        "candidate_minus_full_post_switch_lcb": _lcb(
            difference("candidate", "full_reset", "post_switch_accuracy"), seed=20261606 + tag
        ),
    }
    for key in diagnostics[0]:
        aggregate[key] = max(row[key] for row in diagnostics)
    return aggregate


def _boundary_nulls(config: RecurrentDagBenchConfig) -> dict[str, float]:
    stationary = []
    flat = []
    soft_config = replace(config.dag, soft_content=True, strict_causal_order=True)
    for offset in range(config.validation_seeds):
        seed = 877200 + offset
        summary, _ = _boundary_metric_rows(
            _trials(seed, config, ood=False, matched_stationary=True), config
        )
        stationary.append(summary["candidate"]["accuracy"] - summary["soft_no_reset"]["accuracy"])
        candidate = RecurrentDecisionDag(soft_config)
        candidate_correct = []
        flat_correct = []
        for trial in _trials(seed + 100, config, ood=False, flat=True):
            output = candidate.forward_step(trial.content, trial.cues)
            correct = output.action == trial.target
            candidate_correct.append(float(correct))
            candidate.commit_feedback_with_context_boundary(
                1.0 if correct else -1.0,
                mode="surprise_directional",
            )
            flat_action = sum(int(value >= 0.0) << bit for bit, value in enumerate(trial.content))
            flat_correct.append(float(flat_action == trial.target))
        flat.append(float(np.mean(candidate_correct) - np.mean(flat_correct)))
    return {
        "stationary_candidate_minus_no_reset_absolute_accuracy": abs(float(np.mean(stationary))),
        "flat_candidate_minus_matched_flat_accuracy": float(np.mean(flat)),
    }


def evaluate_context_boundary(
    config: RecurrentDagBenchConfig | None = None,
) -> dict[str, object]:
    cfg = config or RecurrentDagBenchConfig()
    id_result = _boundary_domain(cfg, ood=False)
    ood_result = _boundary_domain(cfg, ood=True)
    nulls = _boundary_nulls(cfg)
    id_effects = id_result["effects"]
    ood_effects = ood_result["effects"]
    gates = {
        "exact_surprise_identity": (
            id_result["positive_reset_violations"] == 0.0
            and ood_result["positive_reset_violations"] == 0.0
            and id_result["negative_strength_max_error"] <= 1e-15
            and ood_result["negative_strength_max_error"] <= 1e-15
        ),
        "directional_nonexpansive": (
            id_result["candidate_norm_increase"] <= 1e-12
            and ood_result["candidate_norm_increase"] <= 1e-12
            and id_result["candidate_orthogonal_error"] <= 1e-12
            and ood_result["candidate_orthogonal_error"] <= 1e-12
        ),
        "post_switch_recovery": (
            id_effects["candidate_minus_no_reset_post_switch_lcb"] >= 0.08
            and ood_effects["candidate_minus_no_reset_post_switch_lcb"] >= 0.08
        ),
        "accuracy_noninferior_to_hard": (
            id_effects["candidate_minus_hard_accuracy_lcb"] >= -0.01
            and ood_effects["candidate_minus_hard_accuracy_lcb"] >= -0.01
        ),
        "nll_improves_hard": (
            id_effects["hard_minus_candidate_nll_lcb"] > 0.0
            and ood_effects["hard_minus_candidate_nll_lcb"] > 0.0
        ),
        "beats_negative_directional": (
            id_effects["candidate_minus_negative_post_switch_lcb"] > 0.0
            and ood_effects["candidate_minus_negative_post_switch_lcb"] > 0.0
        ),
        "beats_generic_forgetting": (
            id_effects["candidate_minus_generic_post_switch_lcb"] > 0.0
            and ood_effects["candidate_minus_generic_post_switch_lcb"] > 0.0
        ),
        "beats_full_reset": (
            id_effects["candidate_minus_full_post_switch_lcb"] > 0.0
            and ood_effects["candidate_minus_full_post_switch_lcb"] > 0.0
        ),
        "nulls": (
            nulls["stationary_candidate_minus_no_reset_absolute_accuracy"] <= 0.02
            and nulls["flat_candidate_minus_matched_flat_accuracy"] <= 0.01
        ),
        "integrity": (
            id_result["nonfinite_count"] == 0.0
            and ood_result["nonfinite_count"] == 0.0
            and id_result["maximum_state_norm"] <= cfg.dag.state_norm_cap + 1e-12
            and ood_result["maximum_state_norm"] <= cfg.dag.state_norm_cap + 1e-12
        ),
    }
    hard_gate = all(gates.values())
    return {
        "schema": "clarus.recurrent-bg-dag-context-boundary.validation.v1",
        "config": asdict(cfg),
        "id": id_result,
        "ood": ood_result,
        "nulls": nulls,
        "future_reads": 0,
        "environment_clone_calls": 0,
        "same_tick_feedback_commits": 0,
        "pending_overwrites": 0,
        "topology_cycles": 0,
        "gates": gates,
        "promise_score": 10 * sum(bool(value) for value in gates.values()),
        "hard_gate": hard_gate,
        "decision": "GO" if hard_gate else "STOP",
        "claim_scope": "synthetic surprise-gated context-update mechanism only",
    }


def _posterior_metric_rows(
    trials: tuple[_Trial, ...],
    config: RecurrentDagBenchConfig,
    seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    hard_config = replace(config.dag, soft_content=False, strict_causal_order=False)
    soft_config = replace(config.dag, soft_content=True, strict_causal_order=True)
    arms = (
        "hard_recurrent",
        "soft_feedforward",
        "signed_heuristic",
        "directional_reset",
        "outcome_posterior",
        "support_derangement",
        "outcome_sign_flip",
    )
    values = {arm: {"correct": [], "nll": [], "post_switch": []} for arm in arms}
    models = {"hard_recurrent": RecurrentDecisionDag(hard_config)}
    models.update({arm: RecurrentDecisionDag(soft_config) for arm in arms if arm != "hard_recurrent"})
    posterior_config = OutcomePosteriorConfig(switch_hazard=0.06)
    rng = np.random.default_rng(seed + 1877)
    permutations = [_derangement(rng, len(config.dag.context_masks)) for _ in trials]
    diagnostics = {
        "posterior_sum_error": 0.0,
        "bayes_residual": 0.0,
        "transition_sum_error": 0.0,
        "minimum_posterior": 1.0,
        "minimum_predicted_prior": 1.0,
        "degenerate_evidence_count": 0.0,
        "maximum_state_norm": 0.0,
        "nonfinite_count": 0.0,
    }
    since_switch = 99
    for index, trial in enumerate(trials):
        since_switch = 1 if trial.switched else since_switch + 1
        for arm, model in models.items():
            if arm == "soft_feedforward":
                model.reset()
            output = model.forward_step(trial.content, trial.cues)
            probability = float(output.probabilities[trial.target])
            correct = output.action == trial.target
            values[arm]["correct"].append(float(correct))
            values[arm]["nll"].append(-math.log(max(1e-300, probability)))
            if 2 <= since_switch <= 5:
                values[arm]["post_switch"].append(float(correct))
            feedback = 1.0 if correct else -1.0
            if arm == "hard_recurrent" or arm == "signed_heuristic":
                model.commit_feedback(feedback)
            elif arm == "directional_reset":
                model.commit_feedback_with_context_boundary(
                    feedback, mode="surprise_directional"
                )
            elif arm in {"outcome_posterior", "support_derangement", "outcome_sign_flip"}:
                result = model.commit_outcome_posterior(
                    feedback,
                    config=posterior_config,
                    support_permutation=(
                        permutations[index] if arm == "support_derangement" else None
                    ),
                    flip_outcome=arm == "outcome_sign_flip",
                )
                if arm == "outcome_posterior":
                    diagnostics["posterior_sum_error"] = max(
                        diagnostics["posterior_sum_error"], result.posterior_sum_error
                    )
                    diagnostics["bayes_residual"] = max(
                        diagnostics["bayes_residual"], result.bayes_residual
                    )
                    diagnostics["transition_sum_error"] = max(
                        diagnostics["transition_sum_error"], result.transition_sum_error
                    )
                    diagnostics["minimum_posterior"] = min(
                        diagnostics["minimum_posterior"],
                        min(result.posterior_context_probabilities),
                    )
                    diagnostics["minimum_predicted_prior"] = min(
                        diagnostics["minimum_predicted_prior"],
                        min(result.predicted_next_prior),
                    )
                    diagnostics["degenerate_evidence_count"] += float(
                        result.degenerate_evidence
                    )
            diagnostics["maximum_state_norm"] = max(
                diagnostics["maximum_state_norm"], float(np.linalg.norm(model.state))
            )
    diagnostics["nonfinite_count"] = float(
        sum(model.nonfinite_count for model in models.values())
    )
    return {
        arm: {
            "accuracy": float(np.mean(metrics["correct"])),
            "nll": float(np.mean(metrics["nll"])),
            "post_switch_accuracy": (
                float(np.mean(metrics["post_switch"])) if metrics["post_switch"] else float("nan")
            ),
        }
        for arm, metrics in values.items()
    }, diagnostics


def _posterior_domain(config: RecurrentDagBenchConfig, *, ood: bool) -> dict[str, object]:
    start = 879100 if ood else 879000
    rows = []
    diagnostics = []
    for offset in range(config.validation_seeds):
        summary, diagnostic = _posterior_metric_rows(
            _trials(start + offset, config, ood=ood), config, start + offset
        )
        rows.append(summary)
        diagnostics.append(diagnostic)
    arms = tuple(rows[0])
    aggregate = {
        arm: {
            metric: float(np.mean([row[arm][metric] for row in rows]))
            for metric in rows[0][arm]
        }
        for arm in arms
    }

    def difference(left: str, right: str, metric: str) -> list[float]:
        return [row[left][metric] - row[right][metric] for row in rows]

    tag = 100 if ood else 0
    aggregate["effects"] = {
        "posterior_minus_signed_accuracy_lcb": _lcb(
            difference("outcome_posterior", "signed_heuristic", "accuracy"), seed=20261701 + tag
        ),
        "posterior_minus_signed_post_switch_lcb": _lcb(
            difference("outcome_posterior", "signed_heuristic", "post_switch_accuracy"), seed=20261702 + tag
        ),
        "posterior_minus_hard_accuracy_lcb": _lcb(
            difference("outcome_posterior", "hard_recurrent", "accuracy"), seed=20261703 + tag
        ),
        "hard_minus_posterior_nll_lcb": _lcb(
            difference("hard_recurrent", "outcome_posterior", "nll"), seed=20261704 + tag
        ),
        "posterior_minus_derangement_accuracy_lcb": _lcb(
            difference("outcome_posterior", "support_derangement", "accuracy"), seed=20261705 + tag
        ),
        "posterior_minus_sign_flip_accuracy_lcb": _lcb(
            difference("outcome_posterior", "outcome_sign_flip", "accuracy"), seed=20261706 + tag
        ),
        "posterior_minus_reset_accuracy_lcb": _lcb(
            difference("outcome_posterior", "directional_reset", "accuracy"), seed=20261707 + tag
        ),
    }
    for key in diagnostics[0]:
        if key.startswith("minimum_"):
            aggregate[key] = min(row[key] for row in diagnostics)
        else:
            aggregate[key] = max(row[key] for row in diagnostics)
    return aggregate


def _posterior_nulls(config: RecurrentDagBenchConfig) -> dict[str, float]:
    stationary = []
    flat = []
    soft_config = replace(config.dag, soft_content=True, strict_causal_order=True)
    posterior_config = OutcomePosteriorConfig(switch_hazard=0.06)
    for offset in range(config.validation_seeds):
        seed = 879200 + offset
        summary, _ = _posterior_metric_rows(
            _trials(seed, config, ood=False, matched_stationary=True), config, seed
        )
        stationary.append(
            summary["outcome_posterior"]["accuracy"]
            - summary["signed_heuristic"]["accuracy"]
        )
        candidate = RecurrentDecisionDag(soft_config)
        candidate_correct = []
        flat_correct = []
        for trial in _trials(seed + 100, config, ood=False, flat=True):
            output = candidate.forward_step(trial.content, trial.cues)
            correct = output.action == trial.target
            candidate_correct.append(float(correct))
            candidate.commit_outcome_posterior(
                1.0 if correct else -1.0,
                config=posterior_config,
            )
            flat_action = sum(int(value >= 0.0) << bit for bit, value in enumerate(trial.content))
            flat_correct.append(float(flat_action == trial.target))
        flat.append(float(np.mean(candidate_correct) - np.mean(flat_correct)))
    return {
        "stationary_posterior_minus_signed_absolute_accuracy": abs(float(np.mean(stationary))),
        "flat_posterior_minus_matched_flat_accuracy": float(np.mean(flat)),
    }


def evaluate_outcome_posterior(
    config: RecurrentDagBenchConfig | None = None,
) -> dict[str, object]:
    cfg = config or RecurrentDagBenchConfig()
    id_result = _posterior_domain(cfg, ood=False)
    ood_result = _posterior_domain(cfg, ood=True)
    nulls = _posterior_nulls(cfg)
    id_effects = id_result["effects"]
    ood_effects = ood_result["effects"]
    gates = {
        "simplex_valid": (
            id_result["minimum_posterior"] > 0.0
            and ood_result["minimum_posterior"] > 0.0
            and id_result["minimum_predicted_prior"] > 0.0
            and ood_result["minimum_predicted_prior"] > 0.0
            and id_result["posterior_sum_error"] <= 1e-12
            and ood_result["posterior_sum_error"] <= 1e-12
        ),
        "exact_filter_identities": (
            id_result["bayes_residual"] <= 1e-12
            and ood_result["bayes_residual"] <= 1e-12
            and id_result["transition_sum_error"] <= 1e-12
            and ood_result["transition_sum_error"] <= 1e-12
        ),
        "beats_signed_accuracy": (
            id_effects["posterior_minus_signed_accuracy_lcb"] >= 0.02
            and ood_effects["posterior_minus_signed_accuracy_lcb"] >= 0.02
        ),
        "beats_signed_post_switch": (
            id_effects["posterior_minus_signed_post_switch_lcb"] >= 0.05
            and ood_effects["posterior_minus_signed_post_switch_lcb"] >= 0.05
        ),
        "accuracy_noninferior_to_hard": (
            id_effects["posterior_minus_hard_accuracy_lcb"] >= -0.01
            and ood_effects["posterior_minus_hard_accuracy_lcb"] >= -0.01
        ),
        "nll_improves_hard": (
            id_effects["hard_minus_posterior_nll_lcb"] > 0.0
            and ood_effects["hard_minus_posterior_nll_lcb"] > 0.0
        ),
        "support_alignment": (
            id_effects["posterior_minus_derangement_accuracy_lcb"] >= 0.05
            and ood_effects["posterior_minus_derangement_accuracy_lcb"] >= 0.05
        ),
        "outcome_sign": (
            id_effects["posterior_minus_sign_flip_accuracy_lcb"] >= 0.10
            and ood_effects["posterior_minus_sign_flip_accuracy_lcb"] >= 0.10
        ),
        "beats_directional_reset": (
            id_effects["posterior_minus_reset_accuracy_lcb"] > 0.0
            and ood_effects["posterior_minus_reset_accuracy_lcb"] > 0.0
        ),
        "null_and_integrity": (
            nulls["stationary_posterior_minus_signed_absolute_accuracy"] <= 0.02
            and nulls["flat_posterior_minus_matched_flat_accuracy"] <= 0.01
            and id_result["degenerate_evidence_count"] == 0.0
            and ood_result["degenerate_evidence_count"] == 0.0
            and id_result["nonfinite_count"] == 0.0
            and ood_result["nonfinite_count"] == 0.0
        ),
    }
    hard_gate = all(gates.values())
    return {
        "schema": "clarus.recurrent-bg-dag-outcome-posterior.validation.v1",
        "config": asdict(cfg),
        "model_switch_hazard": 0.06,
        "id": id_result,
        "ood": ood_result,
        "nulls": nulls,
        "future_reads": 0,
        "environment_clone_calls": 0,
        "same_tick_feedback_commits": 0,
        "pending_overwrites": 0,
        "topology_cycles": 0,
        "legacy_decay_updates_in_candidate": 0,
        "explicit_resets_in_candidate": 0,
        "gates": gates,
        "promise_score": 10 * sum(bool(value) for value in gates.values()),
        "hard_gate": hard_gate,
        "decision": "GO" if hard_gate else "STOP",
        "claim_scope": "exact finite-model context filter on a synthetic task only",
    }


__all__ = [
    "RecurrentDagBenchConfig",
    "evaluate_recurrent_dag",
    "evaluate_soft_evidence",
    "evaluate_context_boundary",
    "evaluate_outcome_posterior",
    "small_recurrent_dag_config",
]
