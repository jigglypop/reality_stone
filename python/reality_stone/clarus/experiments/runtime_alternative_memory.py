"""Alternative native recurrent-memory mechanisms.

The predecessor's causal-STDP result is frozen.  This module keeps two new
questions isolated from it:

* M0: how much supervised low-rank recurrent structure is sufficient?
* M1: can signed local eligibility, accumulated over a replay block and
  modulated by a target-blind block-end clock, acquire the association?

Every scored probe uses the real :class:`BrainRuntime` weight and activation
after both external stores have been physically removed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch

from ..runtime import BrainRuntime, BrainRuntimeConfig, HippocampusMemory, RuntimeMode
from .runtime_native_loops import (
    NativeLoopsConfig,
    _codebook,
    _decode,
    _detach,
    _loop8_replay_source_audit,
    _probe_rollout,
    _runtime,
    _unit,
)
from ..stdp import structural_projection


@dataclass(frozen=True)
class AlternativeMemoryConfig:
    dim: int = 48
    replay_epochs: int = 12
    replay_ticks: int = 3
    rollout_horizon: int = 6
    cue_corruption: float = 0.15
    cue_drive_gain: float = 5.0
    max_write_norm: float = 5.0
    m1_lr: float = 0.8
    m1_trace_decay: float = 0.95
    m1_eligibility_decay: float = 0.99
    m1_ltp: float = 1.0
    m1_ltd: float = 0.20
    m1_abstain_threshold: float = 0.20
    neuronwise_active_threshold: tuple[float, ...] | None = None
    neuronwise_bit_lower_threshold: tuple[float, ...] | None = None
    neuronwise_bit_upper_threshold: tuple[float, ...] | None = None
    seed: int = 97201

    def native(self) -> NativeLoopsConfig:
        return NativeLoopsConfig(
            dim=self.dim,
            replay_epochs=self.replay_epochs,
            rollout_horizon=self.rollout_horizon,
            cue_corruption=self.cue_corruption,
            cue_drive_gain=self.cue_drive_gain,
            bounded_write_gain=1.0,
            seed=self.seed,
        )


class DelayedSignedEligibility:
    """Signed rate eligibility with an explicit row-post/column-pre convention.

    The previous presynaptic trace is used before the current activation is
    folded into it.  A cue followed by a replayed value therefore contributes
    ``outer(value, cue)`` rather than a same-tick autocorrelation.  Weight is
    never owned or mutated by this object.
    """

    def __init__(self, config: AlternativeMemoryConfig, *, device: str | torch.device = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.pre_trace = torch.zeros(config.dim, device=self.device)
        self.post_trace = torch.zeros(config.dim, device=self.device)
        self.eligibility = torch.zeros(config.dim, config.dim, device=self.device)
        self.observations = 0

    def observe(self, activation: torch.Tensor) -> None:
        current = activation.detach().float().to(self.device).view(self.config.dim)
        previous_pre = self.pre_trace.clone()
        previous_post = self.post_trace.clone()
        ltp = self.config.m1_ltp * torch.outer(current, previous_pre)
        ltd = self.config.m1_ltd * torch.outer(previous_post, current)
        self.eligibility = self.config.m1_eligibility_decay * self.eligibility + ltp - ltd
        self.pre_trace = self.config.m1_trace_decay * previous_pre + current
        self.post_trace = self.config.m1_trace_decay * previous_post + current
        self.observations += 1

    def reset(self) -> None:
        self.pre_trace.zero_()
        self.post_trace.zero_()
        self.eligibility.zero_()
        self.observations = 0


def _association_contrast(delta: torch.Tensor, cues: torch.Tensor, targets: torch.Tensor, indices: list[int]) -> float:
    terms: list[torch.Tensor] = []
    for index in indices:
        correct = targets[index] @ delta @ cues[index]
        alternatives = [targets[other] @ delta @ cues[index] for other in indices if other != index]
        incorrect = torch.stack(alternatives).mean() if alternatives else torch.zeros_like(correct)
        terms.append(correct - incorrect)
    return float(torch.stack(terms).mean().item()) if terms else 0.0


def _dense_sparse_parity(runtime: BrainRuntime) -> bool:
    return bool(torch.allclose(runtime.weight, runtime.sparse_weight.to_dense(), atol=1e-7, rtol=0.0))


def _native_state(runtime: BrainRuntime, vector: torch.Tensor, config: AlternativeMemoryConfig) -> torch.Tensor:
    runtime.reset_evaluation_state()
    runtime.hippocampus = HippocampusMemory(config.dim, capacity=runtime.config.memory_capacity, device=runtime.device)
    runtime.step(
        external_input=config.cue_drive_gain * vector,
        force_mode=RuntimeMode.WAKE,
        learning_signal=0.0,
    )
    return _unit(runtime.activation.clone())


def _svd_truncate(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, list[float], int]:
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    numerical_rank = int((singular > 1e-6).sum().item())
    retained = min(max(int(rank), 1), max(numerical_rank, 1))
    truncated = (u[:, :retained] * singular[:retained]) @ vh[:retained]
    return truncated, [float(value) for value in singular[:retained]], numerical_rank


def _random_same_spectrum(seed: int, dim: int, singular: list[float]) -> torch.Tensor:
    rank = len(singular)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    left, _ = torch.linalg.qr(torch.randn(dim, rank, generator=generator), mode="reduced")
    right, _ = torch.linalg.qr(torch.randn(dim, rank, generator=generator), mode="reduced")
    values = torch.tensor(singular, dtype=torch.float32)
    return (left * values) @ right.T


def _evaluate_sealed(
    runtime: BrainRuntime,
    temporal: object,
    cues: torch.Tensor,
    targets: torch.Tensor,
    indices: list[int],
    config: AlternativeMemoryConfig,
    *,
    abstain_threshold: float = 0.15,
) -> dict[str, Any]:
    cutoff = _detach(runtime, temporal)  # type: ignore[arg-type]
    sealed = runtime.snapshot()
    restored = BrainRuntime.from_snapshot(sealed, backend="torch", device="cpu")
    snapshot_restore_parity = bool(torch.equal(restored.weight, runtime.weight) and len(restored.hippocampus) == 0)
    clean: list[bool] = []
    corrupt: list[bool] = []
    gains: list[float] = []
    known_target_cosines: list[float] = []
    rows_after: list[int] = []
    for index in indices:
        cue_state, final, rows = _probe_rollout(sealed, cues[index], config.native())
        clean.append(_decode(final, targets, abstain_threshold=abstain_threshold) == index)
        known_target_cosines.append(float((targets @ _unit(final))[index].item()))
        gains.append(float(torch.nn.functional.cosine_similarity(final, targets[index], dim=0)
                           - torch.nn.functional.cosine_similarity(cue_state, targets[index], dim=0)))
        rows_after.append(rows)
        noisy = cues[index].clone()
        noisy[: max(1, int(config.dim * config.cue_corruption))] = 0.0
        _, noisy_final, rows = _probe_rollout(sealed, noisy, config.native())
        corrupt.append(_decode(noisy_final, targets, abstain_threshold=abstain_threshold) == index)
        rows_after.append(rows)
    deleted = sorted(set(range(len(cues))) - set(indices))
    deleted_abstentions: list[bool] = []
    deleted_max_cosines: list[float] = []
    for index in deleted:
        _, final, rows = _probe_rollout(sealed, cues[index], config.native())
        deleted_abstentions.append(_decode(final, targets, abstain_threshold=abstain_threshold) is None)
        deleted_max_cosines.append(float((targets @ _unit(final)).max().item()))
        rows_after.append(rows)
    return {
        "clean_accuracy": sum(clean) / max(1, len(clean)),
        "corrupt_accuracy": sum(corrupt) / max(1, len(corrupt)),
        "deleted_abstention": sum(deleted_abstentions) / max(1, len(deleted_abstentions)),
        "unknown_abstention": float(_decode(
            torch.zeros(config.dim), targets, abstain_threshold=abstain_threshold,
        ) is None),
        "abstain_threshold": float(abstain_threshold),
        "attractor_cosine_gain": sum(gains) / max(1, len(gains)),
        "known_target_cosines": known_target_cosines,
        "known_min_target_cosine": min(known_target_cosines, default=0.0),
        "deleted_max_cosines": deleted_max_cosines,
        "deleted_max_cosine": max(deleted_max_cosines, default=0.0),
        "cutoff_audit": cutoff,
        "hippocampal_rows_after_rollout": max(rows_after, default=0),
        "snapshot_restore_parity": snapshot_restore_parity,
        "dense_sparse_parity": _dense_sparse_parity(runtime),
        "finite": bool(torch.isfinite(runtime.weight).all()),
    }


def _m0_condition(seed: int, config: AlternativeMemoryConfig, rank: int, condition: str) -> dict[str, Any]:
    native = config.native()
    runtime = _runtime(seed, native)
    runtime.config.hippocampal_encoding_enabled = False
    temporal, source, _ = _loop8_replay_source_audit()
    indices = [int(entry["value"]) for entry in source]
    cues, targets = _codebook(seed, config.dim)
    initial = runtime.weight.clone()

    cue_states = {index: _native_state(runtime, cues[index], config) for index in indices}
    target_states = {index: _native_state(runtime, targets[index], config) for index in indices}
    raw = torch.zeros_like(runtime.weight)
    shifted = indices[1:] + indices[:1]
    for position, index in enumerate(indices):
        paired = shifted[position] if condition == "target_shuffled" else index
        cue_state = cue_states[index]
        target_state = target_states[paired]
        raw += torch.outer(target_state, cue_state) + 0.65 * torch.outer(target_state, target_state)

    association, singular, numerical_rank = _svd_truncate(raw, rank)
    desired = association
    spectrum_parity = True
    if condition == "random_low_rank":
        desired = _random_same_spectrum(seed + 4049 + rank, config.dim, singular)
        spectrum_parity = bool(torch.allclose(
            torch.linalg.svdvals(desired)[: len(singular)],
            torch.tensor(singular), atol=1e-5, rtol=1e-5,
        ))
    elif condition == "cue_only":
        cue_write = sum((torch.outer(cue_states[index], cue_states[index]) for index in indices), torch.zeros_like(raw))
        desired, _, _ = _svd_truncate(cue_write, rank)
        desired = desired * (association.norm() / desired.norm().clamp_min(1e-12))

    installed = 0.0
    if condition != "no_write":
        installed = runtime.install_bounded_recurrent_delta(
            desired - runtime.weight,
            max_frobenius_norm=config.max_write_norm,
        )
    result = _evaluate_sealed(runtime, temporal, cues, targets, indices, config)
    result.update({
        "condition": condition,
        "requested_rank": rank,
        "raw_numerical_rank": numerical_rank,
        "retained_singular_values": singular,
        "raw_write_norm": float(raw.norm().item()),
        "desired_write_norm": float(desired.norm().item()),
        "installed_write_norm": installed,
        "weight_drift": float((runtime.weight - initial).norm().item()),
        "random_spectrum_parity": spectrum_parity,
        "source_indices": indices,
        "source_manifest": source,
        "dale_law": runtime.config.dale_law,
        "structural_projection_used": False,
    })
    return result


def m0_capacity_rank_sweep(seed: int, config: AlternativeMemoryConfig | None = None) -> dict[str, Any]:
    config = config or AlternativeMemoryConfig(seed=seed)
    config = AlternativeMemoryConfig(**{**asdict(config), "seed": seed})
    ranks = [1, 2, 4, config.dim]
    results: dict[str, Any] = {}
    for rank in ranks:
        label = "full" if rank == config.dim else str(rank)
        learned = _m0_condition(seed, config, rank, "supervised")
        controls = {
            name: _m0_condition(seed, config, rank, name)
            for name in ("no_write", "target_shuffled", "random_low_rank", "cue_only")
        }
        advantage = learned["clean_accuracy"] - max(control["clean_accuracy"] for control in controls.values())
        learned["controls"] = controls
        learned["control_advantage"] = advantage
        learned["status"] = "GO" if (
            learned["installed_write_norm"] > 0.0
            and learned["clean_accuracy"] >= 0.80
            and learned["corrupt_accuracy"] >= 0.65
            and learned["deleted_abstention"] >= 0.95
            and learned["unknown_abstention"] >= 0.95
            and learned["attractor_cosine_gain"] >= 0.05
            and advantage >= 0.20
            and learned["hippocampal_rows_after_rollout"] == 0
            and learned["snapshot_restore_parity"]
            and learned["dense_sparse_parity"]
            and learned["finite"]
            and all(control["random_spectrum_parity"] for control in controls.values())
        ) else "STOP"
        results[label] = learned
    return {
        "seed": seed,
        "route": "M0_supervised_low_rank_capacity_ceiling",
        "rank_order": ["1", "2", "4", "full"],
        "results": results,
        "passing_ranks": [label for label in ("1", "2", "4", "full") if results[label]["status"] == "GO"],
    }


def _m1_runtime(config: AlternativeMemoryConfig) -> BrainRuntime:
    return BrainRuntime(
        torch.zeros(config.dim, config.dim),
        config=BrainRuntimeConfig(
            dim=config.dim,
            active_ratio=0.25,
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=False,
            f1_self_measure=False,
            stdp_enabled=False,
            memory_capacity=16,
            replay_gain=1.0,
            hippocampal_encoding_enabled=False,
            neuronwise_active_threshold=config.neuronwise_active_threshold,
            neuronwise_bit_lower_threshold=config.neuronwise_bit_lower_threshold,
            neuronwise_bit_upper_threshold=config.neuronwise_bit_upper_threshold,
        ),
        backend="torch",
        device="cpu",
    )


def _m1_apply_block(
    runtime: BrainRuntime,
    tracker: DelayedSignedEligibility,
    gate: float,
    config: AlternativeMemoryConfig,
) -> dict[str, float]:
    before = runtime.weight.clone()
    candidate = structural_projection(
        before + config.m1_lr * gate * tracker.eligibility,
        density=1.0,
        theta_on=1e-6,
        theta_off=5e-7,
    )
    candidate.fill_diagonal_(0.0)
    installed = runtime.install_bounded_recurrent_delta(
        candidate - before,
        max_frobenius_norm=config.max_write_norm,
    ) if gate != 0.0 else 0.0
    applied = runtime.weight - before
    audit = {
        "gate": float(gate),
        "eligibility_norm": float(tracker.eligibility.norm().item()),
        "candidate_delta_norm": float((candidate - before).norm().item()),
        "installed_delta_norm": installed,
        "applied_delta_norm": float(applied.norm().item()),
    }
    tracker.reset()
    return audit


def _m1_condition(seed: int, config: AlternativeMemoryConfig, condition: str) -> dict[str, Any]:
    runtime = _m1_runtime(config)
    temporal, source, _ = _loop8_replay_source_audit()
    indices = [int(entry["value"]) for entry in source]
    shifted = indices[1:] + indices[:1]
    cues, targets = _codebook(seed, config.dim)
    initial = runtime.weight.clone()
    tracker = DelayedSignedEligibility(config)
    block_audits: list[dict[str, float]] = []
    mid_block_unchanged = True
    runtime_ticks = 0
    event_count = 0
    pulse_count = 0
    interphase_reset_count = 0

    for _epoch in range(config.replay_epochs):
        for position, index in enumerate(indices):
            paired = shifted[position] if condition == "target_shuffled" else index
            cue, target = cues[index], targets[paired]
            runtime.reset_evaluation_state()
            runtime.hippocampus = HippocampusMemory(config.dim, capacity=runtime.config.memory_capacity, device=runtime.device)
            block_weight = runtime.weight.clone()

            def observe_step(*, external: torch.Tensor, cue_arg: torch.Tensor, mode: RuntimeMode) -> None:
                nonlocal mid_block_unchanged, runtime_ticks, event_count
                runtime.step(external_input=external, cue=cue_arg, force_mode=mode, learning_signal=0.0)
                tracker.observe(runtime.activation)
                mid_block_unchanged = mid_block_unchanged and torch.equal(runtime.weight, block_weight)
                runtime_ticks += 1
                event_count += 1

            target_first = condition == "time_reversed"
            if target_first:
                runtime.hippocampus.encode(cue, value=target, priority=1.0)
                for _ in range(config.replay_ticks):
                    observe_step(external=torch.zeros(config.dim), cue_arg=cue, mode=RuntimeMode.NREM)
                runtime.hippocampus = HippocampusMemory(config.dim, capacity=runtime.config.memory_capacity, device=runtime.device)
                runtime.reset_evaluation_state()
                interphase_reset_count += 1
                observe_step(external=config.cue_drive_gain * cue, cue_arg=cue, mode=RuntimeMode.WAKE)
            else:
                observe_step(external=config.cue_drive_gain * cue, cue_arg=cue, mode=RuntimeMode.WAKE)
                if condition == "eligibility_reset":
                    tracker.reset()
                if condition != "no_replay":
                    runtime.hippocampus.encode(cue, value=target, priority=1.0)
                runtime.reset_evaluation_state()
                interphase_reset_count += 1
                for _ in range(config.replay_ticks):
                    observe_step(external=torch.zeros(config.dim), cue_arg=cue, mode=RuntimeMode.NREM)

            gate = 0.0 if condition == "zero_gate" else (-1.0 if condition == "sign_flipped" else 1.0)
            block_audits.append(_m1_apply_block(runtime, tracker, gate, config))
            pulse_count += 1

    delta = runtime.weight - initial
    result = _evaluate_sealed(
        runtime, temporal, cues, targets, indices, config,
        abstain_threshold=config.m1_abstain_threshold,
    )
    result.update({
        "condition": condition,
        "weight_drift": float(delta.norm().item()),
        "association_contrast": _association_contrast(delta, cues, targets, indices),
        "source_indices": indices,
        "source_manifest": source,
        "block_count": config.replay_epochs * len(indices),
        "pulse_count": pulse_count,
        "event_count": event_count,
        "runtime_tick_count": runtime_ticks,
        "interphase_reset_count": interphase_reset_count,
        "expected_event_count": config.replay_epochs * len(indices) * (config.replay_ticks + 1),
        "mid_block_weight_unchanged": bool(mid_block_unchanged),
        "block_end_apply_only": True,
        "gate_audit": {
            "source": "fixed_block_end_clock",
            "base_value": 1.0,
            "reads_runtime_state": False,
            "reads_reward": False,
            "reads_target_identity": False,
            "reads_decoder": False,
            "reads_memory_value": False,
            "reads_condition_flag": False,
            "lesion_override": condition if condition in {"zero_gate", "sign_flipped"} else "none",
        },
        "block_apply_audits": block_audits,
    })
    return result


def m1_delayed_three_factor(seed: int, config: AlternativeMemoryConfig | None = None) -> dict[str, Any]:
    config = config or AlternativeMemoryConfig(seed=seed)
    config = AlternativeMemoryConfig(**{**asdict(config), "seed": seed})
    learned = _m1_condition(seed, config, "fixed_clock")
    controls = {
        name: _m1_condition(seed, config, name)
        for name in (
            "zero_gate",
            "sign_flipped",
            "time_reversed",
            "eligibility_reset",
            "no_replay",
            "target_shuffled",
        )
    }
    strongest_control = max(control["clean_accuracy"] for control in controls.values())
    advantage = learned["clean_accuracy"] - strongest_control
    contrast_margin = learned["association_contrast"] - controls["target_shuffled"]["association_contrast"]
    schedule_fields = (
        "block_count", "pulse_count", "event_count", "runtime_tick_count",
        "interphase_reset_count", "expected_event_count",
    )
    schedule_parity = all(
        all(control[field] == learned[field] for field in schedule_fields)
        for control in controls.values()
    )
    learned["route"] = "M1_fixed_clock_delayed_three_factor"
    learned["controls"] = controls
    learned["control_advantage"] = advantage
    learned["target_shuffled_contrast_margin"] = contrast_margin
    learned["schedule_parity"] = schedule_parity
    learned["status"] = "GO" if (
        learned["weight_drift"] > 0.0
        and learned["association_contrast"] > 1e-6
        and contrast_margin > 1e-6
        and learned["clean_accuracy"] >= 0.80
        and learned["corrupt_accuracy"] >= 0.65
        and learned["deleted_abstention"] >= 0.95
        and learned["unknown_abstention"] >= 0.95
        and learned["attractor_cosine_gain"] >= 0.05
        and advantage >= 0.20
        and learned["mid_block_weight_unchanged"]
        and learned["hippocampal_rows_after_rollout"] == 0
        and learned["snapshot_restore_parity"]
        and learned["dense_sparse_parity"]
        and learned["finite"]
        and schedule_parity
    ) else "STOP"
    return learned


def run_alternative_memory(seed: int = 97201, *, config: AlternativeMemoryConfig | None = None) -> dict[str, Any]:
    config = config or AlternativeMemoryConfig(seed=seed)
    config = AlternativeMemoryConfig(**{**asdict(config), "seed": seed})
    return {
        "seed": seed,
        "config": asdict(config),
        "m0": m0_capacity_rank_sweep(seed, config),
        "m1": m1_delayed_three_factor(seed, config),
    }


def run_seed_range(seeds: Iterable[int], *, config: AlternativeMemoryConfig | None = None) -> list[dict[str, Any]]:
    return [run_alternative_memory(int(seed), config=config) for seed in seeds]
