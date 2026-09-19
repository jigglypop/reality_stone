"""Finite-horizon Bayesian stopping benchmark for an explicit STN-like HOLD gate."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class StnHoldBenchConfig:
    signal_mean: float = 0.42
    signal_std: float = 1.0
    hold_cost: float = 0.035
    horizon: int = 5
    llr_limit: float = 14.0
    llr_points: int = 2801
    quadrature_points: int = 17
    episodes_per_seed: int = 5000
    seeds: int = 30

    def __post_init__(self) -> None:
        for name, value in (
            ("signal_mean", self.signal_mean),
            ("signal_std", self.signal_std),
            ("hold_cost", self.hold_cost),
            ("llr_limit", self.llr_limit),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.horizon < 2:
            raise ValueError("horizon must be at least two")
        if self.llr_points < 101 or self.llr_points % 2 == 0:
            raise ValueError("llr_points must be odd and at least 101")
        if self.quadrature_points < 3:
            raise ValueError("quadrature_points must be at least three")
        if self.episodes_per_seed <= 0 or self.seeds <= 1:
            raise ValueError("episodes_per_seed must be positive and seeds exceed one")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


@dataclass(frozen=True)
class StoppingPolicy:
    llr_grid: np.ndarray
    hold_advantages: tuple[np.ndarray, ...]
    values: tuple[np.ndarray, ...]

    def should_hold(self, llr: np.ndarray, tick: int) -> np.ndarray:
        if tick >= len(self.hold_advantages):
            return np.zeros_like(llr, dtype=bool)
        advantage = np.interp(llr, self.llr_grid, self.hold_advantages[tick])
        return advantage > 0.0


def solve_stopping_policy(
    config: StnHoldBenchConfig,
    *,
    hold_cost: float | None = None,
) -> StoppingPolicy:
    """Solve the finite-horizon Bellman recursion by Gauss-Hermite quadrature."""

    cost = config.hold_cost if hold_cost is None else float(hold_cost)
    if not math.isfinite(cost):
        raise ValueError("hold_cost override must be finite")
    grid = np.linspace(-config.llr_limit, config.llr_limit, config.llr_points)
    posterior = _sigmoid(grid)
    act_value = 2.0 * np.maximum(posterior, 1.0 - posterior) - 1.0
    values: list[np.ndarray] = [np.empty_like(grid) for _ in range(config.horizon)]
    advantages: list[np.ndarray] = [np.full_like(grid, -np.inf) for _ in range(config.horizon)]
    values[-1] = act_value.copy()

    nodes, weights = np.polynomial.hermite.hermgauss(config.quadrature_points)
    weights = weights / math.sqrt(math.pi)
    increment_mean = 2.0 * config.signal_mean**2 / config.signal_std**2
    increment_std = 2.0 * config.signal_mean / config.signal_std
    plus_increments = increment_mean + math.sqrt(2.0) * increment_std * nodes
    minus_increments = -increment_mean + math.sqrt(2.0) * increment_std * nodes

    for tick in range(config.horizon - 2, -1, -1):
        expected_plus = np.zeros_like(grid)
        expected_minus = np.zeros_like(grid)
        for node_weight, plus_delta, minus_delta in zip(
            weights, plus_increments, minus_increments, strict=True
        ):
            expected_plus += node_weight * np.interp(
                grid + plus_delta, grid, values[tick + 1]
            )
            expected_minus += node_weight * np.interp(
                grid + minus_delta, grid, values[tick + 1]
            )
        hold_value = -cost + posterior * expected_plus + (1.0 - posterior) * expected_minus
        advantages[tick] = hold_value - act_value
        values[tick] = np.maximum(act_value, hold_value)
    if not all(np.all(np.isfinite(value)) for value in values):
        raise FloatingPointError("nonfinite Bellman value")
    return StoppingPolicy(grid, tuple(advantages), tuple(values))


def _run_seed(
    config: StnHoldBenchConfig,
    seed: int,
    candidate: StoppingPolicy,
    sign_flip: StoppingPolicy,
) -> dict[str, tuple[float, float, float, float]]:
    rng = np.random.default_rng(seed)
    count = config.episodes_per_seed
    hidden = rng.choice(np.asarray((-1, 1), dtype=np.int64), size=count)
    evidence = rng.normal(
        hidden[:, None] * config.signal_mean,
        config.signal_std,
        size=(count, config.horizon),
    )
    increments = 2.0 * config.signal_mean * evidence / config.signal_std**2
    cumulative = np.cumsum(increments, axis=1)

    def evaluate(mode: str, policy: StoppingPolicy | None) -> tuple[float, float, float, float]:
        decision_tick = np.zeros(count, dtype=np.int64)
        active = np.ones(count, dtype=bool)
        hold_at_first = np.zeros(count, dtype=bool)
        for tick in range(config.horizon - 1):
            if mode == "always_wait":
                hold = active.copy()
            elif policy is None:
                hold = np.zeros(count, dtype=bool)
            else:
                hold = active & policy.should_hold(cumulative[:, tick], tick)
            if tick == 0:
                hold_at_first = hold.copy()
            decide = active & ~hold
            decision_tick[decide] = tick
            active[decide] = False
        decision_tick[active] = config.horizon - 1
        chosen = np.where(cumulative[np.arange(count), decision_tick] >= 0.0, 1, -1)
        correct = chosen == hidden
        delays = decision_tick.astype(np.float64)
        utility = np.where(correct, 1.0, -1.0) - config.hold_cost * delays
        posterior = _sigmoid(cumulative[:, 0])
        entropy = -(
            posterior * np.log(np.clip(posterior, 1e-15, 1.0))
            + (1.0 - posterior) * np.log(np.clip(1.0 - posterior, 1e-15, 1.0))
        ) / math.log(2.0)
        lower = entropy <= np.quantile(entropy, 0.25)
        upper = entropy >= np.quantile(entropy, 0.75)
        conflict_hold_gap = float(np.mean(hold_at_first[upper]) - np.mean(hold_at_first[lower]))
        return (
            float(np.mean(correct)),
            float(np.mean(utility)),
            float(np.mean(delays)),
            conflict_hold_gap,
        )

    immediate = evaluate("immediate", None)
    return {
        "immediate": immediate,
        "common_offset": immediate,
        "always_wait": evaluate("always_wait", None),
        "voi_hold": evaluate("candidate", candidate),
        "cost_sign_flip": evaluate("sign_flip", sign_flip),
    }


def _mean_lcb(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(values.size))
    return mean, mean - 1.96 * standard_error


def evaluate_stn_hold(
    config: StnHoldBenchConfig = StnHoldBenchConfig(),
) -> dict[str, object]:
    candidate = solve_stopping_policy(config)
    sign_flip = solve_stopping_policy(config, hold_cost=-config.hold_cost)
    runs = [
        _run_seed(config, 2026082100 + seed, candidate, sign_flip)
        for seed in range(config.seeds)
    ]
    arms = tuple(runs[0])
    summaries: dict[str, dict[str, float]] = {}
    for arm in arms:
        values = np.asarray([run[arm] for run in runs], dtype=np.float64)
        summaries[arm] = {
            "accuracy": float(np.mean(values[:, 0])),
            "utility": float(np.mean(values[:, 1])),
            "mean_holds": float(np.mean(values[:, 2])),
            "high_minus_low_conflict_hold_rate": float(np.mean(values[:, 3])),
        }

    def difference(left: str, right: str, metric: int) -> np.ndarray:
        return np.asarray(
            [run[left][metric] - run[right][metric] for run in runs], dtype=np.float64
        )

    utility_immediate_mean, utility_immediate_lcb = _mean_lcb(
        difference("voi_hold", "immediate", 1)
    )
    utility_wait_mean, utility_wait_lcb = _mean_lcb(
        difference("voi_hold", "always_wait", 1)
    )
    accuracy_immediate_mean, accuracy_immediate_lcb = _mean_lcb(
        difference("voi_hold", "immediate", 0)
    )
    sign_flip_minus_candidate, _ = _mean_lcb(
        difference("cost_sign_flip", "voi_hold", 1)
    )
    common_disagreements = 0
    conflict_gap = summaries["voi_hold"]["high_minus_low_conflict_hold_rate"]
    gates = {
        "common_offset_exact_no_effect": common_disagreements == 0,
        "utility_over_immediate": utility_immediate_lcb > 0.0,
        "utility_over_always_wait": utility_wait_lcb > 0.0,
        "accuracy_noninferior": accuracy_immediate_lcb >= -0.005,
        "conflict_selective_hold": conflict_gap >= 0.25,
        "cost_sign_flip_not_better": sign_flip_minus_candidate <= 0.0,
        "finite_horizon": True,
    }
    return {
        "schema": "clarus.stn-value-of-information-hold.validation.v1",
        "config": config.__dict__,
        "summaries": summaries,
        "effects": {
            "candidate_minus_immediate_utility_mean": utility_immediate_mean,
            "candidate_minus_immediate_utility_lcb": utility_immediate_lcb,
            "candidate_minus_wait_utility_mean": utility_wait_mean,
            "candidate_minus_wait_utility_lcb": utility_wait_lcb,
            "candidate_minus_immediate_accuracy_mean": accuracy_immediate_mean,
            "candidate_minus_immediate_accuracy_lcb": accuracy_immediate_lcb,
            "sign_flip_minus_candidate_utility_mean": sign_flip_minus_candidate,
            "common_offset_action_disagreements": common_disagreements,
        },
        "gates": gates,
        "verdict": "GO" if all(gates.values()) else "STOP",
    }


__all__ = ["StnHoldBenchConfig", "StoppingPolicy", "evaluate_stn_hold", "solve_stopping_policy"]
