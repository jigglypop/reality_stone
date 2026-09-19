"""Factorial costly-probe benchmark for the dual-SCC research controller.

The benchmark is deliberately synthetic.  It tests whether a slow hidden-rule
summary and a fast information-acquisition loop are both used causally.  It is
not evidence of literal basal-ganglia identity or general intelligence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from typing import Sequence

import numpy as np

from .dual_scc_basal_ganglia import DualSCCBasalGanglia, DualSCCConfig


@dataclass(frozen=True)
class DualSCCProbeBenchConfig:
    episodes_per_seed: int = 240
    context_bias: float = 1.10
    cue_mean_id: float = 0.28
    cue_mean_ood: float = 0.22
    cue_std: float = 1.0
    clear_noise_id: float = 0.24
    clear_noise_ood: float = 0.30
    evidence_noise_id: float = 1.35
    evidence_noise_ood: float = 1.50
    probe_noise_id: float = 0.12
    probe_noise_ood: float = 0.16
    model_clear_noise: float = 0.24
    model_evidence_noise: float = 1.35
    model_probe_noise: float = 0.12
    probe_cost: float = 0.30
    feedback_delay_min: int = 2
    feedback_delay_max: int = 4
    logit_clip: float = 7.0
    drive_scale: float = 3.0
    slow_memory_gain: float = 0.72
    quadrature_points: int = 9
    hidden_blocks_id: tuple[int, ...] = (29, 37, 31)
    hidden_blocks_ood: tuple[int, ...] = (23, 41, 29)

    def __post_init__(self) -> None:
        if self.episodes_per_seed <= 0:
            raise ValueError("episodes_per_seed must be positive")
        if self.feedback_delay_min < 0 or self.feedback_delay_max < self.feedback_delay_min:
            raise ValueError("feedback delay range is invalid")
        if self.quadrature_points < 5:
            raise ValueError("quadrature_points must be at least five")
        if not self.hidden_blocks_id or not self.hidden_blocks_ood:
            raise ValueError("hidden block schedules cannot be empty")
        if any(length <= 0 for length in (*self.hidden_blocks_id, *self.hidden_blocks_ood)):
            raise ValueError("hidden block lengths must be positive")
        for name, value in asdict(self).items():
            if name in {
                "episodes_per_seed",
                "feedback_delay_min",
                "feedback_delay_max",
                "quadrature_points",
                "hidden_blocks_id",
                "hidden_blocks_ood",
            }:
                continue
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class _EpisodeStream:
    context: np.ndarray
    latent: np.ndarray
    evidence_first: np.ndarray
    evidence_probe: np.ndarray
    cue: np.ndarray
    feedback_delay: np.ndarray


@dataclass(frozen=True)
class _PendingOutcome:
    due: int
    label: int
    evidence: tuple[float, ...]
    noises: tuple[float, ...]


@dataclass(frozen=True)
class _ArmMetrics:
    accuracy: float
    utility: float
    brier: float
    nll: float
    hold_rate: float
    high_minus_low_conflict_hold_rate: float
    max_simplex_error: float
    max_residual_bound: float
    nonfinite_events: int
    future_reads: int
    duplicate_feedback: int


_ARMS = (
    "dual",
    "slow_only",
    "fast_only",
    "cross_cut",
    "summary_shuffle",
    "feedback_sign_flip",
    "time_shift",
    "clock_swap",
    "monolithic",
    "always_hold",
    "never_hold",
    "oracle",
)

_CELLS = ("S0F0", "S1F0", "S0F1", "S1F1")


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _clip_probability(value: float) -> float:
    return min(max(float(value), 1e-9), 1.0 - 1e-9)


def _logit(value: float) -> float:
    probability = _clip_probability(value)
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _posterior_latent(
    evidence: Sequence[float],
    noises: Sequence[float],
) -> tuple[float, float]:
    if len(evidence) != len(noises) or not evidence:
        raise ValueError("evidence and noise sequences must be nonempty and aligned")
    precision = 1.0
    weighted_sum = 0.0
    for observation, noise in zip(evidence, noises, strict=True):
        if not math.isfinite(observation) or not math.isfinite(noise) or noise <= 0.0:
            raise ValueError("evidence must be finite and noise positive")
        inverse_variance = 1.0 / noise**2
        precision += inverse_variance
        weighted_sum += observation * inverse_variance
    variance = 1.0 / precision
    return variance * weighted_sum, variance


def _label_probability(
    context_probability: float,
    evidence: Sequence[float],
    noises: Sequence[float],
    context_bias: float,
) -> tuple[float, float, float]:
    mean, variance = _posterior_latent(evidence, noises)
    scale = math.sqrt(variance)
    positive_if_context_positive = _normal_cdf((mean + context_bias) / scale)
    positive_if_context_negative = _normal_cdf((mean - context_bias) / scale)
    probability = (
        context_probability * positive_if_context_positive
        + (1.0 - context_probability) * positive_if_context_negative
    )
    return _clip_probability(probability), mean, variance


def _outcome_log_likelihood_ratio(
    label: int,
    evidence: Sequence[float],
    noises: Sequence[float],
    context_bias: float,
) -> float:
    _, mean, variance = _label_probability(0.5, evidence, noises, context_bias)
    scale = math.sqrt(variance)
    positive_plus = _normal_cdf((mean + context_bias) / scale)
    positive_minus = _normal_cdf((mean - context_bias) / scale)
    likelihood_plus = positive_plus if label > 0 else 1.0 - positive_plus
    likelihood_minus = positive_minus if label > 0 else 1.0 - positive_minus
    return math.log(_clip_probability(likelihood_plus)) - math.log(
        _clip_probability(likelihood_minus)
    )


def _expected_probe_accuracy(
    context_probability: float,
    evidence: Sequence[float],
    noises: Sequence[float],
    *,
    context_bias: float,
    probe_noise: float,
    quadrature_points: int,
) -> float:
    mean, variance = _posterior_latent(evidence, noises)
    predictive_std = math.sqrt(variance + probe_noise**2)
    nodes, weights = _hermgauss(quadrature_points)
    total = 0.0
    for node, weight in zip(nodes, weights, strict=True):
        next_evidence = mean + math.sqrt(2.0) * predictive_std * float(node)
        probability, _, _ = _label_probability(
            context_probability,
            (*evidence, next_evidence),
            (*noises, probe_noise),
            context_bias,
        )
        total += float(weight) * max(probability, 1.0 - probability)
    return total / math.sqrt(math.pi)


@lru_cache(maxsize=None)
def _hermgauss(points: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite.hermgauss(points)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _context_sequence(count: int, blocks: Sequence[int], initial: int) -> np.ndarray:
    result = np.empty(count, dtype=np.int8)
    context = 1 if initial >= 0 else -1
    position = 0
    block_index = 0
    while position < count:
        length = int(blocks[block_index % len(blocks)])
        stop = min(position + length, count)
        result[position:stop] = context
        context *= -1
        position = stop
        block_index += 1
    return result


def _stream(
    config: DualSCCProbeBenchConfig,
    seed: int,
    *,
    regime: str,
    slow_demand: bool,
    fast_demand: bool,
) -> _EpisodeStream:
    rng = np.random.default_rng(seed)
    count = config.episodes_per_seed
    if slow_demand:
        blocks = config.hidden_blocks_id if regime == "ID" else config.hidden_blocks_ood
        context = _context_sequence(count, blocks, int(rng.choice((-1, 1))))
    else:
        context = np.ones(count, dtype=np.int8)
    latent = rng.normal(0.0, 1.0, size=count)
    if fast_demand:
        evidence_noise = (
            config.evidence_noise_id if regime == "ID" else config.evidence_noise_ood
        )
        probe_noise = config.probe_noise_id if regime == "ID" else config.probe_noise_ood
    else:
        evidence_noise = config.clear_noise_id if regime == "ID" else config.clear_noise_ood
        probe_noise = 1e6
    cue_mean = config.cue_mean_id if regime == "ID" else config.cue_mean_ood
    evidence_first = latent + rng.normal(0.0, evidence_noise, size=count)
    evidence_probe = latent + rng.normal(0.0, probe_noise, size=count)
    cue = context * cue_mean + rng.normal(0.0, config.cue_std, size=count)
    delay = rng.integers(
        config.feedback_delay_min,
        config.feedback_delay_max + 1,
        size=count,
    )
    return _EpisodeStream(context, latent, evidence_first, evidence_probe, cue, delay)


def _model_noises(
    config: DualSCCProbeBenchConfig,
    *,
    regime: str,
    fast_demand: bool,
) -> tuple[float, float]:
    if fast_demand:
        if regime == "ID":
            return config.model_evidence_noise, config.model_probe_noise
        return config.evidence_noise_ood, config.probe_noise_ood
    if regime == "ID":
        return config.model_clear_noise, 1e6
    return config.clear_noise_ood, 1e6


def _fixed_point_policy(
    core: DualSCCBasalGanglia,
    context_probability: float,
    evidence_probability: float,
    hold_advantage: float,
    *,
    initial_slow: Sequence[float],
    initial_fast: Sequence[float],
    logit_clip: float,
    drive_scale: float,
    slow_memory_gain: float,
) -> tuple[float, float, tuple[float, ...], tuple[float, ...], float, float]:
    context_logit = float(np.clip(_logit(context_probability), -logit_clip, logit_clip))
    label_logit = float(np.clip(_logit(evidence_probability), -logit_clip, logit_clip))
    entropy = -(
        evidence_probability * math.log(evidence_probability)
        + (1.0 - evidence_probability) * math.log(1.0 - evidence_probability)
    ) / math.log(2.0)
    prior_slow = np.asarray(tuple(initial_slow), dtype=np.float64)
    slow_drive = (
        -context_logit / drive_scale + slow_memory_gain * float(prior_slow[0]),
        context_logit / drive_scale + slow_memory_gain * float(prior_slow[1]),
    )
    fast_drive = (
        -label_logit / drive_scale,
        label_logit / drive_scale,
        2.0 * entropy - 1.0,
    )
    result = core.settle(
        slow_drive,
        fast_drive,
        initial_slow=initial_slow,
        initial_fast=initial_fast,
    )
    fast = np.asarray(result.fast_state, dtype=np.float64)
    policy = core.policy(fast, hold_bias_delta=float(hold_advantage))
    # Both returned choices are read from the recurrent state.  The analytic
    # observation model only supplies normalized drives and a value-of-
    # information offset; it is not allowed to bypass the recurrent readout.
    return (
        policy.conditional_action_probabilities[1],
        policy.hold_probability,
        result.slow_state,
        result.fast_state,
        policy.normalization_error,
        result.error_bound,
    )


def _evaluate_arm(
    config: DualSCCProbeBenchConfig,
    core: DualSCCBasalGanglia,
    stream: _EpisodeStream,
    *,
    arm: str,
    regime: str,
    slow_demand: bool,
    fast_demand: bool,
) -> _ArmMetrics:
    if arm not in _ARMS:
        raise ValueError(f"unknown benchmark arm: {arm}")
    count = config.episodes_per_seed
    first_noise, probe_model_noise = _model_noises(
        config,
        regime=regime,
        fast_demand=fast_demand,
    )
    cue_mean = config.cue_mean_id if regime == "ID" else config.cue_mean_ood
    belief_logit = 0.0
    previous_decision_probability = 0.5
    pending: list[_PendingOutcome] = []
    slow_anchor = np.zeros(core.slow_size, dtype=np.float64)
    correct = np.zeros(count, dtype=bool)
    utilities = np.zeros(count, dtype=np.float64)
    briers = np.zeros(count, dtype=np.float64)
    nll = np.zeros(count, dtype=np.float64)
    holds = np.zeros(count, dtype=bool)
    margins = np.abs(stream.latent + config.context_bias * stream.context)
    max_simplex_error = 0.0
    max_residual_bound = 0.0
    duplicate_feedback = 0

    for trial in range(count):
        remaining: list[_PendingOutcome] = []
        for outcome in pending:
            if outcome.due > trial:
                remaining.append(outcome)
                continue
            label = -outcome.label if arm == "feedback_sign_flip" else outcome.label
            belief_logit += _outcome_log_likelihood_ratio(
                label,
                outcome.evidence,
                outcome.noises,
                config.context_bias,
            )
        pending = remaining
        belief_logit = float(np.clip(belief_logit, -config.logit_clip, config.logit_clip))

        true_context = int(stream.context[trial])
        if not slow_demand:
            internal_probability = 1.0 if true_context > 0 else 0.0
        else:
            cue_increment = 2.0 * cue_mean * float(stream.cue[trial]) / config.cue_std**2
            belief_logit = float(
                np.clip(
                    belief_logit + cue_increment,
                    -config.logit_clip,
                    config.logit_clip,
                )
            )
            internal_probability = _sigmoid(belief_logit)

        if arm in {"fast_only", "cross_cut"} and slow_demand:
            decision_context_probability = 0.5
        elif arm == "summary_shuffle" and slow_demand:
            decision_context_probability = 1.0 - internal_probability
        elif arm == "time_shift" and slow_demand:
            decision_context_probability = previous_decision_probability
        elif arm == "clock_swap" and slow_demand:
            cue_only = 2.0 * cue_mean * float(stream.cue[trial]) / config.cue_std**2
            decision_context_probability = _sigmoid(cue_only)
        elif arm == "oracle":
            decision_context_probability = 1.0 if true_context > 0 else 0.0
        else:
            decision_context_probability = internal_probability
        previous_decision_probability = internal_probability

        evidence = (float(stream.evidence_first[trial]),)
        noises = (first_noise,)
        label_probability, _, _ = _label_probability(
            decision_context_probability,
            evidence,
            noises,
            config.context_bias,
        )
        evidence_probability, _, _ = _label_probability(
            0.5,
            evidence,
            noises,
            config.context_bias,
        )
        current_accuracy = max(label_probability, 1.0 - label_probability)
        if fast_demand:
            expected_accuracy = _expected_probe_accuracy(
                decision_context_probability,
                evidence,
                noises,
                context_bias=config.context_bias,
                probe_noise=probe_model_noise,
                quadrature_points=config.quadrature_points,
            )
            hold_advantage = 2.0 * (expected_accuracy - current_accuracy) - config.probe_cost
        else:
            hold_advantage = -config.probe_cost
        if arm in {"slow_only", "clock_swap", "never_hold"}:
            hold_advantage = -abs(config.probe_cost)
        elif arm == "always_hold" and fast_demand:
            hold_advantage = abs(config.probe_cost)

        action_probability, hold_probability, slow_state, fast_state, simplex, bound = (
            _fixed_point_policy(
                core,
                decision_context_probability,
                evidence_probability,
                hold_advantage,
                initial_slow=slow_anchor,
                initial_fast=np.zeros(core.fast_size, dtype=np.float64),
                logit_clip=config.logit_clip,
                drive_scale=config.drive_scale,
                slow_memory_gain=config.slow_memory_gain,
            )
        )
        max_simplex_error = max(max_simplex_error, simplex)
        max_residual_bound = max(max_residual_bound, bound)
        slow_anchor = np.asarray(slow_state, dtype=np.float64)
        fast_anchor = np.asarray(fast_state, dtype=np.float64)
        hold = bool(fast_demand and hold_probability > 0.5)
        if arm == "always_hold" and fast_demand:
            hold = True
        if arm in {"slow_only", "clock_swap", "never_hold"}:
            hold = False

        if hold:
            holds[trial] = True
            evidence = (*evidence, float(stream.evidence_probe[trial]))
            noises = (*noises, probe_model_noise)
            label_probability, _, _ = _label_probability(
                decision_context_probability,
                evidence,
                noises,
                config.context_bias,
            )
            evidence_probability, _, _ = _label_probability(
                0.5,
                evidence,
                noises,
                config.context_bias,
            )
            action_probability, _, slow_state, _, simplex, bound = _fixed_point_policy(
                core,
                decision_context_probability,
                evidence_probability,
                -abs(config.probe_cost),
                initial_slow=slow_anchor,
                initial_fast=fast_anchor,
                logit_clip=config.logit_clip,
                drive_scale=config.drive_scale,
                slow_memory_gain=config.slow_memory_gain,
            )
            slow_anchor = np.asarray(slow_state, dtype=np.float64)
            max_simplex_error = max(max_simplex_error, simplex)
            max_residual_bound = max(max_residual_bound, bound)

        action = 1 if action_probability >= 0.5 else -1
        true_label = 1 if stream.latent[trial] + config.context_bias * true_context >= 0.0 else -1
        correct[trial] = action == true_label
        utilities[trial] = (1.0 if correct[trial] else -1.0) - (
            config.probe_cost if hold else 0.0
        )
        target_probability = 1.0 if true_label > 0 else 0.0
        clipped_action_probability = _clip_probability(action_probability)
        briers[trial] = 2.0 * (clipped_action_probability - target_probability) ** 2
        nll[trial] = -math.log(
            clipped_action_probability
            if true_label > 0
            else 1.0 - clipped_action_probability
        )

        if slow_demand and arm != "fast_only":
            due = trial + int(stream.feedback_delay[trial])
            pending.append(
                _PendingOutcome(
                    due=due,
                    label=true_label,
                    evidence=tuple(evidence),
                    noises=tuple(noises),
                )
            )

    lower = margins >= np.quantile(margins, 0.75)
    upper = margins <= np.quantile(margins, 0.25)
    conflict_gap = float(np.mean(holds[upper]) - np.mean(holds[lower]))
    return _ArmMetrics(
        accuracy=float(np.mean(correct)),
        utility=float(np.mean(utilities)),
        brier=float(np.mean(briers)),
        nll=float(np.mean(nll)),
        hold_rate=float(np.mean(holds)),
        high_minus_low_conflict_hold_rate=conflict_gap,
        max_simplex_error=max_simplex_error,
        max_residual_bound=max_residual_bound,
        nonfinite_events=0,
        future_reads=0,
        duplicate_feedback=duplicate_feedback,
    )


def _mean_interval(values: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(values))
    if values.size <= 1:
        return {"mean": mean, "lcb": mean, "ucb": mean}
    standard_error = float(np.std(values, ddof=1) / math.sqrt(values.size))
    return {
        "mean": mean,
        "lcb": mean - 1.96 * standard_error,
        "ucb": mean + 1.96 * standard_error,
    }


def _arm_values(
    runs: dict[str, dict[str, dict[str, list[_ArmMetrics]]]],
    regime: str,
    cell: str,
    arm: str,
    metric: str,
) -> np.ndarray:
    return np.asarray(
        [getattr(item, metric) for item in runs[regime][cell][arm]],
        dtype=np.float64,
    )


def evaluate_dual_scc_probe_benchmark(
    *,
    seeds: Sequence[int],
    config: DualSCCProbeBenchConfig = DualSCCProbeBenchConfig(),
    role: str = "development",
) -> dict[str, object]:
    if role not in {"development", "validation", "test"}:
        raise ValueError("role must be development, validation, or test")
    if len(seeds) < 2:
        raise ValueError("at least two paired seeds are required")
    if len(set(int(seed) for seed in seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    core = DualSCCBasalGanglia(DualSCCConfig())
    audit = core.topology_audit()
    runs: dict[str, dict[str, dict[str, list[_ArmMetrics]]]] = {
        regime: {
            cell: {arm: [] for arm in _ARMS}
            for cell in _CELLS
        }
        for regime in ("ID", "OOD")
    }
    for regime_index, regime in enumerate(("ID", "OOD")):
        for cell_index, cell in enumerate(_CELLS):
            slow_demand = cell[1] == "1"
            fast_demand = cell[3] == "1"
            for seed in seeds:
                stream_seed = int(seed) + 100_000 * regime_index + 10_000 * cell_index
                episode_stream = _stream(
                    config,
                    stream_seed,
                    regime=regime,
                    slow_demand=slow_demand,
                    fast_demand=fast_demand,
                )
                aliases: dict[str, str] = {"monolithic": "dual", "cross_cut": "fast_only"}
                if cell == "S0F0":
                    aliases.update({arm: "dual" for arm in _ARMS if arm != "oracle"})
                elif cell == "S1F0":
                    aliases.update(
                        {
                            "slow_only": "dual",
                            "always_hold": "dual",
                            "never_hold": "dual",
                        }
                    )
                elif cell == "S0F1":
                    aliases.update(
                        {
                            "fast_only": "dual",
                            "cross_cut": "dual",
                            "summary_shuffle": "dual",
                            "feedback_sign_flip": "dual",
                            "time_shift": "dual",
                            "monolithic": "dual",
                            "oracle": "dual",
                            "clock_swap": "slow_only",
                            "never_hold": "slow_only",
                        }
                    )
                cached: dict[str, _ArmMetrics] = {}
                for arm in _ARMS:
                    evaluated_arm = aliases.get(arm, arm)
                    if evaluated_arm not in cached:
                        cached[evaluated_arm] = _evaluate_arm(
                            config,
                            core,
                            episode_stream,
                            arm=evaluated_arm,
                            regime=regime,
                            slow_demand=slow_demand,
                            fast_demand=fast_demand,
                        )
                    runs[regime][cell][arm].append(cached[evaluated_arm])

    summaries: dict[str, object] = {}
    for regime in ("ID", "OOD"):
        summaries[regime] = {}
        for cell in _CELLS:
            summaries[regime][cell] = {}
            for arm in _ARMS:
                summaries[regime][cell][arm] = {
                    metric: float(np.mean(_arm_values(runs, regime, cell, arm, metric)))
                    for metric in (
                        "accuracy",
                        "utility",
                        "brier",
                        "nll",
                        "hold_rate",
                        "high_minus_low_conflict_hold_rate",
                        "max_simplex_error",
                        "max_residual_bound",
                    )
                }

    effects: dict[str, object] = {}
    gates: dict[str, bool] = {
        "slow_layer_scc": audit.slow_is_strongly_connected,
        "fast_layer_scc": audit.fast_is_strongly_connected,
        "truthful_union_macro_scc": audit.union_is_single_macro_scc,
        "small_gain": core.certificate.certified,
        "finite_state": True,
    }
    for regime in ("ID", "OOD"):
        candidate = _arm_values(runs, regime, "S1F1", "dual", "utility")
        for comparator in ("slow_only", "fast_only"):
            comparator_values = _arm_values(
                runs, regime, "S1F1", comparator, "utility"
            )
            effect = _mean_interval(candidate - comparator_values)
            effects[f"{regime}_S1F1_dual_minus_{comparator}_utility"] = effect
            gates[f"{regime}_S1F1_utility_over_{comparator}"] = effect["lcb"] > 0.0

            candidate_brier = _arm_values(runs, regime, "S1F1", "dual", "brier")
            comparator_brier = _arm_values(runs, regime, "S1F1", comparator, "brier")
            brier_effect = _mean_interval(comparator_brier - candidate_brier)
            effects[f"{regime}_S1F1_{comparator}_minus_dual_brier"] = brier_effect
            gates[f"{regime}_S1F1_brier_over_{comparator}"] = brier_effect["lcb"] > 0.0

        monolithic = _arm_values(runs, regime, "S1F1", "monolithic", "utility")
        monolithic_effect = _mean_interval(candidate - monolithic)
        effects[f"{regime}_S1F1_dual_minus_monolithic_utility"] = monolithic_effect
        gates[f"{regime}_monolithic_noninferior"] = monolithic_effect["lcb"] >= -0.02

        for comparator, threshold in (
            ("summary_shuffle", 0.03),
            ("feedback_sign_flip", 0.03),
            ("time_shift", 0.02),
            ("clock_swap", 0.02),
        ):
            control = _arm_values(runs, regime, "S1F1", comparator, "utility")
            effect = _mean_interval(candidate - control)
            effects[f"{regime}_S1F1_dual_minus_{comparator}_utility"] = effect
            gates[f"{regime}_causal_{comparator}"] = (
                effect["mean"] >= threshold and effect["lcb"] > 0.0
            )

        for comparator in ("always_hold", "never_hold"):
            control = _arm_values(runs, regime, "S1F1", comparator, "utility")
            effect = _mean_interval(candidate - control)
            effects[f"{regime}_S1F1_dual_minus_{comparator}_utility"] = effect
            gates[f"{regime}_utility_over_{comparator}"] = effect["lcb"] > 0.0

        hold_gap = float(
            np.mean(
                _arm_values(
                    runs,
                    regime,
                    "S1F1",
                    "dual",
                    "high_minus_low_conflict_hold_rate",
                )
            )
        )
        effects[f"{regime}_S1F1_conflict_hold_gap"] = hold_gap
        gates[f"{regime}_conflict_selective_hold"] = hold_gap >= 0.15

        d_values: dict[str, np.ndarray] = {}
        for cell in _CELLS:
            dual = _arm_values(runs, regime, cell, "dual", "utility")
            slow = _arm_values(runs, regime, cell, "slow_only", "utility")
            fast = _arm_values(runs, regime, cell, "fast_only", "utility")
            d_values[cell] = dual - np.maximum(slow, fast)
        interaction = d_values["S1F1"] - d_values["S1F0"] - d_values["S0F1"] + d_values["S0F0"]
        interaction_effect = _mean_interval(interaction)
        effects[f"{regime}_factorial_interaction"] = interaction_effect
        gates[f"{regime}_factorial_interaction"] = (
            interaction_effect["mean"] > 0.02 and interaction_effect["lcb"] > 0.02
        )

        slow_null = _mean_interval(
            _arm_values(runs, regime, "S1F0", "dual", "utility")
            - _arm_values(runs, regime, "S1F0", "slow_only", "utility")
        )
        fast_null = _mean_interval(
            _arm_values(runs, regime, "S0F1", "dual", "utility")
            - _arm_values(runs, regime, "S0F1", "fast_only", "utility")
        )
        effects[f"{regime}_S1F0_dual_minus_slow_null"] = slow_null
        effects[f"{regime}_S0F1_dual_minus_fast_null"] = fast_null
        gates[f"{regime}_slow_only_null"] = slow_null["lcb"] >= -0.02 and slow_null["ucb"] <= 0.02
        gates[f"{regime}_fast_only_null"] = fast_null["lcb"] >= -0.02 and fast_null["ucb"] <= 0.02

    maximum_simplex_error = max(
        item.max_simplex_error
        for regime in runs.values()
        for cell in regime.values()
        for arm in cell.values()
        for item in arm
    )
    maximum_residual_bound = max(
        item.max_residual_bound
        for regime in runs.values()
        for cell in regime.values()
        for arm in cell.values()
        for item in arm
    )
    integrity_totals = {
        name: sum(
            getattr(item, name)
            for regime in runs.values()
            for cell in regime.values()
            for arm in cell.values()
            for item in arm
        )
        for name in ("nonfinite_events", "future_reads", "duplicate_feedback")
    }
    gates["simplex"] = maximum_simplex_error <= 1e-12
    gates["residual_certificate"] = maximum_residual_bound <= core.config.tolerance
    # This reduced benchmark has immutable streams, but it does not yet
    # instrument attempted future reads or implement distinct capacity-matched
    # monolithic/shuffle controls.  Zero observed events must not be promoted
    # into a causal-integrity pass.
    integrity_instrumented = False
    gates["causal_integrity_instrumented"] = integrity_instrumented
    diagnostic_go = all(gates.values())
    return {
        "schema": "clarus.dual-scc-costly-probe.diagnostic.v2",
        "role": str(role),
        "config": asdict(config),
        "core_config": asdict(core.config),
        "seed_count": len(seeds),
        "seeds": tuple(int(seed) for seed in seeds),
        "cells": _CELLS,
        "arms": _ARMS,
        "topology": asdict(audit),
        "certificate": asdict(core.certificate),
        "summaries": summaries,
        "effects": effects,
        "integrity": {
            **integrity_totals,
            "instrumented": integrity_instrumented,
            "maximum_simplex_error": maximum_simplex_error,
            "maximum_residual_bound": maximum_residual_bound,
        },
        "promotion_eligibility": {
            "eligible": False,
            "status": "UNTESTED_INVALID_DESIGN",
            "reasons": (
                "external Bayesian context filtering remains outside the slow recurrent state",
                "monolithic and some null-cell arms are algebraic aliases rather than matched controllers",
                "shuffle/time controls do not instrument the internal cross-summary tensor",
                "attempted future reads are not independently instrumented",
                "the reduced 2+3-state analytic diagnostic is not the preregistered learned 16+16 route",
            ),
        },
        "gates": gates,
        "score": 100.0 * sum(gates.values()) / len(gates),
        "diagnostic_verdict": "GO" if diagnostic_go else "STOP",
        "verdict": "HOLD",
    }


__all__ = [
    "DualSCCProbeBenchConfig",
    "evaluate_dual_scc_probe_benchmark",
]
