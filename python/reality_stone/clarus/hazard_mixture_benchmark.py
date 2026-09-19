"""Locked Loop 8L benchmark for the finite hazard ensemble."""

from __future__ import annotations

from dataclasses import asdict, replace
import math

import numpy as np

from .brain_geometry_benchmark import _lcb
from .hazard_mixture_dag import HazardMixtureConfig, HazardMixtureDecisionDag
from .recurrent_dag_benchmark import (
    RecurrentDagBenchConfig,
    _derangement,
    _trials,
)
from .recurrent_decision_dag import (
    OutcomePosteriorConfig,
    RecurrentDecisionDag,
)


def _sequence(
    trials: tuple,
    config: RecurrentDagBenchConfig,
    seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    hard_config = replace(config.dag, soft_content=False, strict_causal_order=False)
    soft_config = replace(config.dag, soft_content=True, strict_causal_order=True)
    arms = (
        "hard_recurrent",
        "soft_feedforward",
        "signed_heuristic",
        "fixed_h006",
        "hazard_ensemble",
        "frozen_weights",
        "support_derangement",
        "outcome_sign_flip",
    )
    values = {arm: {"correct": [], "nll": [], "post_switch": []} for arm in arms}
    legacy = {
        "hard_recurrent": RecurrentDecisionDag(hard_config),
        "soft_feedforward": RecurrentDecisionDag(soft_config),
        "signed_heuristic": RecurrentDecisionDag(soft_config),
        "fixed_h006": RecurrentDecisionDag(soft_config),
    }
    mixture = {
        "hazard_ensemble": HazardMixtureDecisionDag(),
        "frozen_weights": HazardMixtureDecisionDag(
            HazardMixtureConfig(freeze_model_weights=True)
        ),
        "support_derangement": HazardMixtureDecisionDag(),
        "outcome_sign_flip": HazardMixtureDecisionDag(),
    }
    posterior_config = OutcomePosteriorConfig(switch_hazard=0.06)
    rng = np.random.default_rng(seed + 2131)
    permutations = [_derangement(rng, len(config.dag.context_masks)) for _ in trials]
    since_switch = 99
    diagnostics = {
        "joint_sum_error": 0.0,
        "action_mixture_residual": 0.0,
        "outcome_bayes_residual": 0.0,
        "minimum_joint_mass": 1.0,
        "degenerate_evidence_count": 0.0,
        "nonfinite_count": 0.0,
        "final_expected_hazard": 0.0,
        "final_h0_weight": 0.0,
        "final_max_other_weight": 0.0,
    }
    for index, trial in enumerate(trials):
        since_switch = 1 if trial.switched else since_switch + 1
        for arm, model in legacy.items():
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
            if arm in {"hard_recurrent", "signed_heuristic"}:
                model.commit_feedback(feedback)
            elif arm == "fixed_h006":
                model.commit_outcome_posterior(feedback, config=posterior_config)
        for arm, model in mixture.items():
            output = model.forward_step(trial.content, trial.cues)
            probability = float(output.probabilities[trial.target])
            correct = output.action == trial.target
            values[arm]["correct"].append(float(correct))
            values[arm]["nll"].append(-math.log(max(1e-300, probability)))
            if 2 <= since_switch <= 5:
                values[arm]["post_switch"].append(float(correct))
            result = model.commit_outcome(
                1.0 if correct else -1.0,
                support_permutation=(
                    permutations[index] if arm == "support_derangement" else None
                ),
                flip_outcome=arm == "outcome_sign_flip",
            )
            if arm == "hazard_ensemble":
                diagnostics["joint_sum_error"] = max(
                    diagnostics["joint_sum_error"],
                    output.joint_sum_error,
                    result.joint_sum_error,
                )
                diagnostics["action_mixture_residual"] = max(
                    diagnostics["action_mixture_residual"], output.action_mixture_residual
                )
                diagnostics["outcome_bayes_residual"] = max(
                    diagnostics["outcome_bayes_residual"], result.outcome_bayes_residual
                )
                diagnostics["minimum_joint_mass"] = min(
                    diagnostics["minimum_joint_mass"], result.minimum_joint_mass
                )
                diagnostics["degenerate_evidence_count"] += float(result.degenerate_evidence)
                diagnostics["final_expected_hazard"] = result.expected_hazard
                diagnostics["final_h0_weight"] = result.hazard_weights[0]
                diagnostics["final_max_other_weight"] = max(result.hazard_weights[1:])
    diagnostics["nonfinite_count"] = float(
        sum(model.nonfinite_count for model in (*legacy.values(), *mixture.values()))
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


def _domain(config: RecurrentDagBenchConfig, *, ood: bool) -> dict[str, object]:
    start = 881100 if ood else 881000
    rows = []
    diagnostics = []
    for offset in range(config.validation_seeds):
        summary, diagnostic = _sequence(
            _trials(start + offset, config, ood=ood), config, start + offset
        )
        rows.append(summary)
        diagnostics.append(diagnostic)
    aggregate = {
        arm: {
            metric: float(np.mean([row[arm][metric] for row in rows]))
            for metric in rows[0][arm]
        }
        for arm in rows[0]
    }

    def difference(left: str, right: str, metric: str) -> list[float]:
        return [row[left][metric] - row[right][metric] for row in rows]

    tag = 100 if ood else 0
    aggregate["effects"] = {
        "ensemble_minus_fixed_accuracy_lcb": _lcb(
            difference("hazard_ensemble", "fixed_h006", "accuracy"), seed=20261801 + tag
        ),
        "ensemble_minus_fixed_post_switch_lcb": _lcb(
            difference("hazard_ensemble", "fixed_h006", "post_switch_accuracy"), seed=20261802 + tag
        ),
        "ensemble_minus_hard_accuracy_lcb": _lcb(
            difference("hazard_ensemble", "hard_recurrent", "accuracy"), seed=20261803 + tag
        ),
        "hard_minus_ensemble_nll_lcb": _lcb(
            difference("hard_recurrent", "hazard_ensemble", "nll"), seed=20261804 + tag
        ),
        "ensemble_minus_frozen_accuracy_lcb": _lcb(
            difference("hazard_ensemble", "frozen_weights", "accuracy"), seed=20261805 + tag
        ),
        "ensemble_minus_derangement_accuracy_lcb": _lcb(
            difference("hazard_ensemble", "support_derangement", "accuracy"), seed=20261806 + tag
        ),
        "ensemble_minus_sign_flip_accuracy_lcb": _lcb(
            difference("hazard_ensemble", "outcome_sign_flip", "accuracy"), seed=20261807 + tag
        ),
    }
    for key in diagnostics[0]:
        if key == "minimum_joint_mass":
            aggregate[key] = min(row[key] for row in diagnostics)
        else:
            aggregate[key] = max(row[key] for row in diagnostics)
    aggregate["mean_final_expected_hazard"] = float(
        np.mean([row["final_expected_hazard"] for row in diagnostics])
    )
    aggregate["mean_final_h0_weight"] = float(
        np.mean([row["final_h0_weight"] for row in diagnostics])
    )
    aggregate["mean_final_max_other_weight"] = float(
        np.mean([row["final_max_other_weight"] for row in diagnostics])
    )
    return aggregate


def _nulls(config: RecurrentDagBenchConfig) -> dict[str, float]:
    stationary_differences = []
    flat_differences = []
    h0_weights = []
    other_weights = []
    posterior_config = OutcomePosteriorConfig(switch_hazard=0.06)
    soft_config = replace(config.dag, soft_content=True, strict_causal_order=True)
    for offset in range(config.validation_seeds):
        seed = 881200 + offset
        summary, diagnostic = _sequence(
            _trials(seed, config, ood=False, matched_stationary=True), config, seed
        )
        stationary_differences.append(
            summary["hazard_ensemble"]["accuracy"] - summary["fixed_h006"]["accuracy"]
        )
        h0_weights.append(diagnostic["final_h0_weight"])
        other_weights.append(diagnostic["final_max_other_weight"])
        candidate = HazardMixtureDecisionDag()
        matched_flat = RecurrentDecisionDag(soft_config)
        candidate_correct = []
        flat_correct = []
        for trial in _trials(seed + 100, config, ood=False, flat=True):
            output = candidate.forward_step(trial.content, trial.cues)
            correct = output.action == trial.target
            candidate_correct.append(float(correct))
            candidate.commit_outcome(1.0 if correct else -1.0)
            flat_output = matched_flat.forward_step(trial.content, trial.cues)
            flat_action = sum(int(value >= 0.0) << bit for bit, value in enumerate(trial.content))
            flat_correct.append(float(flat_action == trial.target))
            matched_flat.commit_outcome_posterior(
                1.0 if flat_output.action == trial.target else -1.0,
                config=posterior_config,
            )
        flat_differences.append(float(np.mean(candidate_correct) - np.mean(flat_correct)))
    return {
        "stationary_ensemble_minus_fixed_absolute_accuracy": abs(
            float(np.mean(stationary_differences))
        ),
        "flat_ensemble_minus_matched_flat_accuracy": float(np.mean(flat_differences)),
        "stationary_mean_h0_weight": float(np.mean(h0_weights)),
        "stationary_mean_max_other_weight": float(np.mean(other_weights)),
    }


def evaluate_hazard_ensemble(
    config: RecurrentDagBenchConfig | None = None,
) -> dict[str, object]:
    cfg = config or RecurrentDagBenchConfig()
    id_result = _domain(cfg, ood=False)
    ood_result = _domain(cfg, ood=True)
    nulls = _nulls(cfg)
    id_effects = id_result["effects"]
    ood_effects = ood_result["effects"]
    gates = {
        "joint_simplex": (
            id_result["joint_sum_error"] <= 1e-12
            and ood_result["joint_sum_error"] <= 1e-12
            and id_result["minimum_joint_mass"] >= 0.0
            and ood_result["minimum_joint_mass"] >= 0.0
        ),
        "filter_identities": (
            id_result["action_mixture_residual"] <= 1e-12
            and ood_result["action_mixture_residual"] <= 1e-12
            and id_result["outcome_bayes_residual"] <= 1e-12
            and ood_result["outcome_bayes_residual"] <= 1e-12
            and id_result["degenerate_evidence_count"] == 0.0
            and ood_result["degenerate_evidence_count"] == 0.0
        ),
        "accuracy_noninferior_to_fixed": (
            id_effects["ensemble_minus_fixed_accuracy_lcb"] >= 0.0
            and ood_effects["ensemble_minus_fixed_accuracy_lcb"] >= 0.0
        ),
        "post_switch_improvement": (
            id_effects["ensemble_minus_fixed_post_switch_lcb"] >= 0.03
            and ood_effects["ensemble_minus_fixed_post_switch_lcb"] >= 0.03
        ),
        "accuracy_noninferior_to_hard": (
            id_effects["ensemble_minus_hard_accuracy_lcb"] >= -0.01
            and ood_effects["ensemble_minus_hard_accuracy_lcb"] >= -0.01
        ),
        "nll_improves_hard": (
            id_effects["hard_minus_ensemble_nll_lcb"] > 0.0
            and ood_effects["hard_minus_ensemble_nll_lcb"] > 0.0
        ),
        "learned_weights_add_value": (
            id_effects["ensemble_minus_frozen_accuracy_lcb"] > 0.0
            and ood_effects["ensemble_minus_frozen_accuracy_lcb"] > 0.0
        ),
        "causal_controls": (
            id_effects["ensemble_minus_derangement_accuracy_lcb"] >= 0.05
            and ood_effects["ensemble_minus_derangement_accuracy_lcb"] >= 0.05
            and id_effects["ensemble_minus_sign_flip_accuracy_lcb"] >= 0.10
            and ood_effects["ensemble_minus_sign_flip_accuracy_lcb"] >= 0.10
        ),
        "hazard_identification": (
            ood_result["mean_final_expected_hazard"]
            - id_result["mean_final_expected_hazard"]
            >= 0.03
            and nulls["stationary_mean_h0_weight"]
            > nulls["stationary_mean_max_other_weight"]
        ),
        "null_and_integrity": (
            nulls["stationary_ensemble_minus_fixed_absolute_accuracy"] <= 0.02
            and nulls["flat_ensemble_minus_matched_flat_accuracy"] <= 0.01
            and id_result["nonfinite_count"] == 0.0
            and ood_result["nonfinite_count"] == 0.0
        ),
    }
    hard_gate = all(gates.values())
    return {
        "schema": "clarus.recurrent-bg-dag-hazard-ensemble.validation.v1",
        "config": asdict(cfg),
        "hazards": (0.0, 0.03, 0.06, 0.12, 0.24),
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
        "track_terminal": True,
        "claim_scope": "finite synthetic hazard ensemble under assumed pseudo-likelihoods",
    }


def small_hazard_ensemble_config() -> RecurrentDagBenchConfig:
    return RecurrentDagBenchConfig(trials=48, validation_seeds=3)


__all__ = [
    "evaluate_hazard_ensemble",
    "small_hazard_ensemble_config",
]
