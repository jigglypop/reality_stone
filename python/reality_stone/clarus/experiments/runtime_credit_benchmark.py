"""Runtime-scale signed-credit A/B for BrainRuntime STDP.

The task signal is causal one-step improvement in the runtime's own recurrent
prediction error.  It is delivered on the following tick and therefore gates
an eligibility trace rather than reading a future target.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import asdict, dataclass

import torch

from ..agent import RuntimeAgent, RuntimeAgentConfig
from ..runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode
from ..stdp import structural_projection


@dataclass(frozen=True)
class RuntimeCreditBenchConfig:
    dim: int = 32
    steps: int = 160
    window: int = 32
    probes: int = 32
    stdp_lr: float = 0.01
    apply_interval: int = 4
    density: float = 0.25
    guard_tolerance: float = 0.02
    matched_initial_manifold: bool = False


@dataclass
class RunResult:
    errors: list[float]
    signals: list[float]
    weight_final: torch.Tensor
    weight_init: torch.Tensor
    updates: int
    finite: bool


def _weight_and_streams(seed: int, config: RuntimeCreditBenchConfig):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(config.dim, config.dim, generator=generator) * 0.05
    weight = 0.5 * (weight + weight.T)
    weight.fill_diagonal_(0.0)
    if config.matched_initial_manifold:
        weight = structural_projection(weight, density=config.density)
        weight.fill_diagonal_(0.0)

    def stream(length: int) -> list[torch.Tensor]:
        values = []
        state = torch.zeros(config.dim)
        for _ in range(length):
            innovation = torch.randn(config.dim, generator=generator) * 0.3
            state = 0.72 * state + 0.28 * innovation
            values.append(state.clone())
        return values

    return weight, stream(config.steps), stream(config.probes)


def _runtime(weight: torch.Tensor, config: RuntimeCreditBenchConfig, mode: str) -> BrainRuntime:
    enabled = mode != "off"
    gate_mode = "external_signed" if mode not in {"off", "legacy"} else "critic_derivative"
    runtime = BrainRuntime(
        weight.clone(),
        config=BrainRuntimeConfig(
            dim=config.dim,
            active_ratio=0.25,
            active_threshold=0.0,
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=False,
            memory_capacity=16,
            stdp_enabled=enabled,
            stdp_interval=1,
            stdp_apply_interval=config.apply_interval,
            stdp_lr=config.stdp_lr,
            stdp_density=config.density,
            stdp_gate_threshold=0.0,
            stdp_spike_threshold=0.1,
            stdp_gate_mode=gate_mode,
        ),
        backend="torch",
        device="cpu",
    )
    if mode == "trace_off" and runtime.stdp_tracker is not None:
        runtime.stdp_tracker.config.r_e = 0.0
    return runtime


def _prediction_error(runtime: BrainRuntime, previous_activation: torch.Tensor) -> float:
    predicted = torch.tanh(runtime._bench_pred_weight @ previous_activation)
    return float((runtime.activation.detach() - predicted).norm().item())


def _run(
    weight: torch.Tensor,
    inputs: list[torch.Tensor],
    config: RuntimeCreditBenchConfig,
    mode: str,
    *,
    scheduled_signals: list[float] | None = None,
) -> RunResult:
    runtime = _runtime(weight, config, mode)
    initial = runtime.weight.detach().clone()
    agent = RuntimeAgent(runtime, config=RuntimeAgentConfig(action_count=4))
    errors: list[float] = []
    signals: list[float] = []
    next_signal = 0.0
    for index, external in enumerate(inputs):
        if scheduled_signals is not None:
            supplied = scheduled_signals[index]
        elif mode == "sign_flip":
            supplied = -next_signal
        elif mode == "absolute":
            supplied = abs(next_signal)
        elif mode == "homeostasis_only":
            supplied = next_signal
        else:
            supplied = next_signal
        previous = runtime.activation.detach().clone()
        runtime._bench_pred_weight = runtime.weight.detach().clone()
        output = agent.step(
            external_input=external,
            observation=external,
            force_mode=RuntimeMode.WAKE,
            stdp_learning_signal=(supplied if mode not in {"off", "legacy"} else None),
        )
        error = _prediction_error(runtime, previous)
        if mode == "homeostasis_only":
            active_ratio = output.runtime_step.active_modules / config.dim
            next_signal = 0.25 - active_ratio
        else:
            next_signal = 0.0 if not errors else errors[-1] - error
        errors.append(error)
        signals.append(next_signal)
    return RunResult(
        errors=errors,
        signals=signals,
        weight_final=runtime.weight.detach().clone(),
        weight_init=initial,
        updates=int(runtime._stdp_updates),
        finite=bool(torch.isfinite(runtime.weight).all()),
    )


def _guard(weight: torch.Tensor, probes: list[torch.Tensor], config: RuntimeCreditBenchConfig) -> float:
    result = _run(weight, probes, config, "off")
    return sum(result.errors) / len(result.errors)


def _improvement(errors: list[float], window: int) -> float:
    return statistics.mean(errors[:window]) - statistics.mean(errors[-window:])


def _paired_lcb(values: list[float], seed: int = 20260811, draws: int = 2000) -> float:
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    return means[max(0, int(0.025 * draws) - 1)]


def evaluate_runtime_credit(
    *,
    seeds: tuple[int, ...] = (970101, 970102, 970103, 970104, 970105, 970106, 970107),
    config: RuntimeCreditBenchConfig | None = None,
) -> dict[str, object]:
    cfg = config or RuntimeCreditBenchConfig()
    modes = ("off", "legacy", "signed", "sign_flip", "absolute", "trace_off", "reward_shuffle", "homeostasis_only")
    by_mode = {mode: [] for mode in modes}
    guards = {mode: [] for mode in modes}
    drifts = {mode: [] for mode in modes}
    updates = {mode: [] for mode in modes}
    finite = {mode: [] for mode in modes}
    for seed in seeds:
        weight, inputs, probes = _weight_and_streams(seed, cfg)
        signed_source = _run(weight, inputs, cfg, "signed")
        shuffled = list(signed_source.signals)
        random.Random(seed ^ 0x5A17).shuffle(shuffled)
        runs = {
            "signed": signed_source,
            "off": _run(weight, inputs, cfg, "off"),
            "legacy": _run(weight, inputs, cfg, "legacy"),
            "sign_flip": _run(weight, inputs, cfg, "sign_flip"),
            "absolute": _run(weight, inputs, cfg, "absolute"),
            "trace_off": _run(weight, inputs, cfg, "trace_off"),
            "reward_shuffle": _run(weight, inputs, cfg, "reward_shuffle", scheduled_signals=shuffled),
            "homeostasis_only": _run(weight, inputs, cfg, "homeostasis_only"),
        }
        for mode, result in runs.items():
            by_mode[mode].append(_improvement(result.errors, cfg.window))
            guards[mode].append(_guard(result.weight_final, probes, cfg))
            drifts[mode].append(float((result.weight_final - result.weight_init).norm().item()))
            updates[mode].append(result.updates)
            finite[mode].append(result.finite)

    mean_improvement = {mode: statistics.mean(values) for mode, values in by_mode.items()}
    mean_guard = {mode: statistics.mean(values) for mode, values in guards.items()}
    comparisons = {}
    for other in ("off", "legacy", "sign_flip", "absolute", "trace_off", "reward_shuffle", "homeostasis_only"):
        differences = [a - b for a, b in zip(by_mode["signed"], by_mode[other])]
        comparisons[f"signed_minus_{other}"] = {
            "mean": statistics.mean(differences),
            "lcb95": _paired_lcb(differences),
        }
    guard_delta = statistics.mean(
        [a - b for a, b in zip(guards["signed"], guards["off"])]
    )
    hard_gate = bool(
        comparisons["signed_minus_off"]["lcb95"] > 0.0
        and comparisons["signed_minus_legacy"]["lcb95"] > 0.0
        and comparisons["signed_minus_sign_flip"]["lcb95"] > 0.0
        and comparisons["signed_minus_absolute"]["lcb95"] > 0.0
        and comparisons["signed_minus_trace_off"]["lcb95"] > 0.0
        and comparisons["signed_minus_reward_shuffle"]["lcb95"] > 0.0
        and guard_delta <= cfg.guard_tolerance
        and all(finite["signed"])
        and min(updates["signed"]) > 0
    )
    positive_lcbs = sum(
        float(comparisons[key]["lcb95"] > 0.0)
        for key in comparisons
        if key != "signed_minus_homeostasis_only"
    )
    score = 0.0
    if hard_gate:
        score = 40.0 + 40.0 * positive_lcbs / 6.0 + 10.0 * float(guard_delta <= 0.0) + 10.0
    return {
        "schema": (
            "clarus.runtime-signed-credit.matched-manifold.v2"
            if cfg.matched_initial_manifold
            else "clarus.runtime-signed-credit.validation.v1"
        ),
        "config": asdict(cfg),
        "seeds": list(seeds),
        "mean_improvement": mean_improvement,
        "mean_guard": mean_guard,
        "mean_drift": {mode: statistics.mean(values) for mode, values in drifts.items()},
        "updates": updates,
        "comparisons": comparisons,
        "signed_guard_delta_vs_off": guard_delta,
        "hard_gate": hard_gate,
        "promisingness_score": score,
        "grade": "GO" if score >= 80 else "HOLD" if score >= 65 else "STOP",
        "claim_limit": "synthetic recurrent-prediction credit assignment only",
    }
