"""T1/M2/M3 continuation for alternative recurrent-memory mechanisms.

The previously confirmed M1 implementation is imported read-only.  New
mechanisms are isolated here so their development cannot alter its frozen
binding result.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from typing import Any

import torch

from ..constants import STP_TAU_FAC_INV, STP_TAU_REC, STP_U_BASE
from ..runtime import BrainRuntime, BrainRuntimeConfig, BrainRuntimeSnapshot, HippocampusMemory, RuntimeMode
from .runtime_alternative_memory import (
    AlternativeMemoryConfig,
    DelayedSignedEligibility,
    _association_contrast,
    _dense_sparse_parity,
    _evaluate_sealed,
    _m1_apply_block,
    _m1_runtime,
)
from .runtime_native_loops import (
    _codebook,
    _decode,
    _detach,
    _loop8_replay_source_audit,
    _orthonormal_rows,
    _probe_rollout,
    _unit,
)
from ..stdp import structural_projection
from ..temporal_memory import TemporalAuditedMemory, TemporalMemoryEvent


DEVELOPMENT_SEEDS = range(97301, 97317)
CONFIRMATION_SEEDS = range(99301, 99333)


def _factor_codebooks(seed: int, dim: int) -> dict[str, Any]:
    if dim % 4:
        raise ValueError("factorized routes require dim divisible by four")
    generator = torch.Generator(device="cpu").manual_seed(seed + 2003)
    width = dim // 4
    factor_a = _orthonormal_rows(generator, 2, width)
    factor_b = _orthonormal_rows(generator, 2, width)
    combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def code(ai: int, bi: int, *, target: bool) -> torch.Tensor:
        vector = torch.zeros(dim)
        offset = 2 * width if target else 0
        vector[offset : offset + width] = factor_a[ai]
        vector[offset + width : offset + 2 * width] = factor_b[bi]
        return _unit(vector)

    cues = torch.stack([code(ai, bi, target=False) for ai, bi in combinations])
    targets = torch.stack([code(ai, bi, target=True) for ai, bi in combinations])
    digest = hashlib.sha256(cues.numpy().tobytes() + targets.numpy().tobytes()).hexdigest()
    return {
        "factor_a": factor_a,
        "factor_b": factor_b,
        "combinations": combinations,
        "train": combinations[:-1],
        "held_out": combinations[-1],
        "cues": cues,
        "targets": targets,
        "sha256": digest,
    }


def _factor_mapping_sensitivity(
    weight: torch.Tensor,
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    dim: int,
) -> dict[str, Any]:
    width = dim // 4
    scores: dict[str, list[float]] = {"A": [], "B": []}
    for label, factors, cue_offset, target_offset in (
        ("A", factor_a, 0, 2 * width),
        ("B", factor_b, width, 3 * width),
    ):
        for value in range(2):
            cue = torch.zeros(dim); cue[cue_offset : cue_offset + width] = factors[value]
            target = torch.zeros(dim); target[target_offset : target_offset + width] = factors[value]
            scores[label].append(float(target @ weight @ cue))
    flat = [abs(value) for values in scores.values() for value in values]
    ratio = max(flat) / max(min(flat), 1e-12)
    return {"mapping_scores": scores, "max_to_min_abs_ratio": ratio}


def _t1_condition(seed: int, config: AlternativeMemoryConfig, condition: str) -> dict[str, Any]:
    books = _factor_codebooks(seed, config.dim)
    combinations: list[tuple[int, int]] = books["combinations"]
    train: list[tuple[int, int]] = books["train"]
    held_out: tuple[int, int] = books["held_out"]
    cues: torch.Tensor = books["cues"]
    targets: torch.Tensor = books["targets"]
    runtime = _m1_runtime(config)
    temporal = TemporalAuditedMemory(capacity=32)
    for index, (ai, bi) in enumerate(train):
        temporal.ingest(TemporalMemoryEvent(
            f"factor-{ai}-{bi}", "target", f"{ai}:{bi}", 1, index, f"t1-{index}",
        ))
    tracker = DelayedSignedEligibility(config)
    initial = runtime.weight.clone()
    shifted = train[1:] + train[:1]
    block_count = pulse_count = event_count = runtime_tick_count = interphase_reset_count = 0
    mid_block_unchanged = True
    block_audits: list[dict[str, float]] = []
    staged_combinations: list[list[int]] = []
    schedule_blocks: list[dict[str, Any]] = []

    for epoch in range(config.replay_epochs):
        for position, combination in enumerate(train):
            cue_index = combinations.index(combination)
            paired_combination = shifted[position] if condition == "target_shuffled" else combination
            target_index = combinations.index(paired_combination)
            cue, target = cues[cue_index], targets[target_index]
            runtime.reset_evaluation_state()
            runtime.hippocampus = HippocampusMemory(config.dim, capacity=runtime.config.memory_capacity, device=runtime.device)
            block_weight = runtime.weight.clone()

            def observe_step(*, external: torch.Tensor, cue_arg: torch.Tensor, mode: RuntimeMode) -> None:
                nonlocal event_count, runtime_tick_count, mid_block_unchanged
                runtime.step(external_input=external, cue=cue_arg, force_mode=mode, learning_signal=0.0)
                tracker.observe(runtime.activation)
                mid_block_unchanged = mid_block_unchanged and torch.equal(runtime.weight, block_weight)
                event_count += 1; runtime_tick_count += 1

            if condition == "time_reversed":
                runtime.hippocampus.encode(cue, value=target, priority=1.0)
                staged_combinations.append(list(paired_combination))
                for _ in range(config.replay_ticks):
                    observe_step(external=torch.zeros(config.dim), cue_arg=cue, mode=RuntimeMode.NREM)
                runtime.hippocampus = HippocampusMemory(config.dim, capacity=runtime.config.memory_capacity, device=runtime.device)
                runtime.reset_evaluation_state(); interphase_reset_count += 1
                observe_step(external=config.cue_drive_gain * cue, cue_arg=cue, mode=RuntimeMode.WAKE)
                mode_sequence = ["NREM"] * config.replay_ticks + ["WAKE"]
                replay_sequence = [True] * config.replay_ticks + [False]
                reset_after_tick = config.replay_ticks - 1
            else:
                observe_step(external=config.cue_drive_gain * cue, cue_arg=cue, mode=RuntimeMode.WAKE)
                if condition == "eligibility_reset":
                    tracker.reset()
                if condition != "no_replay":
                    runtime.hippocampus.encode(cue, value=target, priority=1.0)
                    staged_combinations.append(list(paired_combination))
                runtime.reset_evaluation_state(); interphase_reset_count += 1
                for _ in range(config.replay_ticks):
                    observe_step(external=torch.zeros(config.dim), cue_arg=cue, mode=RuntimeMode.NREM)
                mode_sequence = ["WAKE"] + ["NREM"] * config.replay_ticks
                replay_sequence = [False] + [condition != "no_replay"] * config.replay_ticks
                reset_after_tick = 0

            gate = 0.0 if condition == "zero_gate" else (-1.0 if condition == "sign_flipped" else 1.0)
            block_audits.append(_m1_apply_block(runtime, tracker, gate, config))
            schedule_blocks.append({
                "epoch": epoch,
                "position": position,
                "cue": list(combination),
                "staged_target": None if condition == "no_replay" else list(paired_combination),
                "mode_sequence": mode_sequence,
                "external_sequence": ["cue" if mode == "WAKE" else "zero" for mode in mode_sequence],
                "replay_present_sequence": replay_sequence,
                "reset_after_tick": reset_after_tick,
                "eligibility_reset_between_phases": condition == "eligibility_reset",
                "clock": gate,
            })
            block_count += 1; pulse_count += 1

    delta = runtime.weight - initial
    sensitivity = _factor_mapping_sensitivity(
        delta, books["factor_a"], books["factor_b"], config.dim,
    )
    cutoff = _detach(runtime, temporal)
    sealed = runtime.snapshot()
    restored = BrainRuntime.from_snapshot(sealed, backend="torch", device="cpu")
    snapshot_restore_parity = bool(
        torch.equal(restored.weight, runtime.weight)
        and len(restored.hippocampus) == 0
    )
    held_index = combinations.index(held_out)
    cue_state, final, rows = _probe_rollout(sealed, cues[held_index], config.native())
    decoded = _decode(final, targets, abstain_threshold=config.m1_abstain_threshold)
    target_cosine = float((targets @ _unit(final))[held_index].item())
    gain = float(torch.nn.functional.cosine_similarity(final, targets[held_index], dim=0)
                 - torch.nn.functional.cosine_similarity(cue_state, targets[held_index], dim=0))
    staged_set = {tuple(value) for value in staged_combinations}
    heldout_absence = held_out not in train and held_out not in staged_set
    decoder_only = _decode(cues[held_index], targets, abstain_threshold=config.m1_abstain_threshold)
    return {
        "seed": seed,
        "condition": condition,
        "held_out_accuracy": float(decoded == held_index),
        "decoded_index": decoded,
        "held_out_target_cosine": target_cosine,
        "attractor_cosine_gain": gain,
        "weight_drift": float(delta.norm().item()),
        "factor_codebook_sha256": books["sha256"],
        "train_combinations": [list(value) for value in train],
        "held_out_combination": list(held_out),
        "heldout_absence_audit": heldout_absence,
        "staged_unique_combinations": [list(value) for value in sorted(staged_set)],
        "factor_value_counts": {"A": {"0": 2, "1": 1}, "B": {"0": 2, "1": 1}},
        "factor_frequency_sensitivity": sensitivity,
        "schedule_blocks": schedule_blocks,
        "chance_baseline": 0.25,
        "decoder_only_baseline_accuracy": float(decoder_only == held_index),
        "block_count": block_count,
        "pulse_count": pulse_count,
        "event_count": event_count,
        "runtime_tick_count": runtime_tick_count,
        "interphase_reset_count": interphase_reset_count,
        "expected_event_count": config.replay_epochs * len(train) * (config.replay_ticks + 1),
        "mid_block_weight_unchanged": bool(mid_block_unchanged),
        "block_end_apply_only": True,
        "block_apply_audits": block_audits,
        "cutoff_audit": cutoff,
        "hippocampal_rows_after_rollout": rows,
        "snapshot_restore_parity": snapshot_restore_parity,
        "dense_sparse_parity": _dense_sparse_parity(runtime),
        "finite": bool(torch.isfinite(runtime.weight).all()),
        "abstain_threshold": config.m1_abstain_threshold,
    }


def t1_m1_factor_transfer(seed: int, config: AlternativeMemoryConfig | None = None) -> dict[str, Any]:
    config = config or AlternativeMemoryConfig(seed=seed)
    config = AlternativeMemoryConfig(**{**asdict(config), "seed": seed})
    learned = _t1_condition(seed, config, "fixed_clock")
    controls = {
        name: _t1_condition(seed, config, name)
        for name in (
            "zero_gate", "sign_flipped", "time_reversed",
            "eligibility_reset", "no_replay", "target_shuffled",
        )
    }
    advantage = learned["held_out_accuracy"] - max(value["held_out_accuracy"] for value in controls.values())
    schedule_fields = (
        "block_count", "pulse_count", "event_count", "runtime_tick_count",
        "interphase_reset_count", "expected_event_count",
    )
    schedule_parity = all(
        all(control[field] == learned[field] for field in schedule_fields)
        for control in controls.values()
    )
    codebook_parity = all(
        control["factor_codebook_sha256"] == learned["factor_codebook_sha256"]
        for control in controls.values()
    )
    learned_schedule = learned["schedule_blocks"]

    def masked_schedule(name: str, row: dict[str, Any]) -> list[dict[str, Any]]:
        masked: list[dict[str, Any]] = []
        for block in row["schedule_blocks"]:
            copy = dict(block)
            if name in {"zero_gate", "sign_flipped"}:
                copy["clock"] = 1.0
            if name == "eligibility_reset":
                copy["eligibility_reset_between_phases"] = False
            if name == "no_replay":
                copy["staged_target"] = list(copy["cue"])
                copy["replay_present_sequence"] = [False] + [True] * config.replay_ticks
            if name == "target_shuffled":
                copy["staged_target"] = list(copy["cue"])
            if name == "time_reversed":
                copy["mode_sequence"] = ["WAKE"] + ["NREM"] * config.replay_ticks
                copy["external_sequence"] = ["cue"] + ["zero"] * config.replay_ticks
                copy["replay_present_sequence"] = [False] + [True] * config.replay_ticks
                copy["reset_after_tick"] = 0
            masked.append(copy)
        return masked

    schedule_contract = all(
        masked_schedule(name, control) == learned_schedule
        for name, control in controls.items()
    )
    frozen_protocol = asdict(config) == asdict(AlternativeMemoryConfig(seed=seed))
    learned["route"] = "T1_frozen_M1_factor_transfer"
    learned["controls"] = controls
    learned["control_advantage"] = advantage
    learned["schedule_parity"] = schedule_parity
    learned["schedule_contract"] = schedule_contract
    learned["frozen_protocol"] = frozen_protocol
    learned["config"] = asdict(config)
    learned["codebook_parity"] = codebook_parity
    learned["status"] = "GO" if (
        learned["held_out_accuracy"] >= 0.70
        and advantage >= 0.20
        and learned["heldout_absence_audit"]
        and learned["mid_block_weight_unchanged"]
        and learned["hippocampal_rows_after_rollout"] == 0
        and learned["snapshot_restore_parity"]
        and learned["cutoff_audit"]["temporal_rows_after"] == 0
        and learned["cutoff_audit"]["hippocampal_rows_after"] == 0
        and learned["dense_sparse_parity"]
        and learned["finite"]
        and schedule_parity
        and schedule_contract
        and frozen_protocol
        and codebook_parity
    ) else "STOP"
    return learned


M2_PROJECTION = {"density": 1.0, "theta_on": 1e-6, "theta_off": 5e-7}
M2_CONDITIONS = (
    "no_write",
    "target_shuffled",
    "identical_phase",
    "positive_only",
    "negative_only",
    "sign_reversed",
)


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _m2_runtime(seed: int, config: AlternativeMemoryConfig) -> tuple[BrainRuntime, float, int]:
    generator = torch.Generator(device="cpu").manual_seed(seed + 4001)
    gaussian = torch.randn(config.dim, config.dim, generator=generator)
    gaussian.fill_diagonal_(0.0)
    # Repeated float32 row normalization can enter a one-ulp cycle whose
    # aggregate Frobenius residual exceeds the frozen 1e-7 gate.  Derive the
    # support and signs from the Gaussian draw, but use an exactly unit-norm
    # binary amplitude (4 entries of .5, or 16 entries of .25) before applying
    # the declared projection.  This yields a real fixed point without relaxing
    # the tolerance or special-casing the identical-phase control.
    support = 16 if config.dim - 1 >= 16 else 4
    if config.dim - 1 < support:
        raise ValueError("M2 requires at least five recurrent coordinates")
    amplitude = 0.25 if support == 16 else 0.5
    fixed_seed = torch.zeros_like(gaussian)
    for row in range(config.dim):
        scores = gaussian[row].abs().clone()
        scores[row] = -1.0
        indices = torch.topk(scores, support).indices
        signs = torch.where(gaussian[row, indices] >= 0.0, 1.0, -1.0)
        fixed_seed[row, indices] = amplitude * signs
    fixed = structural_projection(fixed_seed, **M2_PROJECTION)
    fixed.fill_diagonal_(0.0)
    projection_passes = 1
    verification = structural_projection(fixed, **M2_PROJECTION)
    verification.fill_diagonal_(0.0)
    residual = float((verification - fixed).norm().item())
    if residual > 1e-7:
        raise RuntimeError(f"M2 projection fixed-point residual {residual} exceeds 1e-7")
    runtime = BrainRuntime(
        fixed,
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
        ),
        backend="torch",
        device="cpu",
    )
    if runtime.stdp_tracker is not None or runtime._stdp_updates != 0:
        raise RuntimeError("M2 requires automatic STDP to remain disabled")
    if not _dense_sparse_parity(runtime):
        raise RuntimeError("M2 base runtime failed dense/sparse parity")
    return runtime, residual, projection_passes


def _m2_collect_phase(
    base: BrainRuntimeSnapshot,
    cue: torch.Tensor,
    replay_value: torch.Tensor,
    config: AlternativeMemoryConfig,
) -> dict[str, Any]:
    """Collect the declared virtual cached-cue plus three-state correlation."""
    runtime = BrainRuntime.from_snapshot(base, backend="torch", device="cpu")
    runtime.hippocampus = HippocampusMemory(
        config.dim, capacity=runtime.config.memory_capacity, device=runtime.device,
    )
    before = runtime.weight.clone()
    runtime.reset_evaluation_state()
    runtime.step(
        external_input=config.cue_drive_gain * cue,
        cue=cue,
        force_mode=RuntimeMode.WAKE,
        learning_signal=0.0,
    )
    cached_cue = runtime.activation.detach().clone()
    # Stage only after the cue observation.  Otherwise WAKE's nonzero replay
    # mix would leak the target into the cached cue and make C+ and C- start
    # from different predecessor vectors.
    runtime.hippocampus.encode(cue, value=replay_value, priority=1.0)
    runtime.reset_evaluation_state()
    previous = cached_cue
    terms: list[torch.Tensor] = []
    activations: list[torch.Tensor] = []
    for _ in range(3):
        runtime.step(
            external_input=torch.zeros(config.dim),
            cue=cue,
            force_mode=RuntimeMode.NREM,
            learning_signal=0.0,
        )
        current = runtime.activation.detach().clone()
        terms.append(torch.outer(current, previous))
        activations.append(current)
        previous = current
    correlation = torch.stack(terms).mean(dim=0)
    first_term_residual = float((terms[0] - torch.outer(activations[0], cached_cue)).norm().item())
    return {
        "correlation": correlation,
        "cached_cue": cached_cue,
        "activations": activations,
        "first_term_residual": first_term_residual,
        "weight_unchanged": bool(torch.equal(runtime.weight, before)),
        "stdp_updates": int(runtime._stdp_updates),
        "hippocampal_rows": len(runtime.hippocampus),
    }


def _m2_factor_evaluation(
    runtime: BrainRuntime,
    temporal: TemporalAuditedMemory,
    books: dict[str, Any],
    config: AlternativeMemoryConfig,
) -> dict[str, Any]:
    cutoff = _detach(runtime, temporal)
    sealed = runtime.snapshot()
    restored = BrainRuntime.from_snapshot(sealed, backend="torch", device="cpu")
    snapshot_restore_parity = bool(
        torch.equal(restored.weight, runtime.weight) and len(restored.hippocampus) == 0
    )
    held_index = books["combinations"].index(books["held_out"])
    cue_state, final, rows = _probe_rollout(sealed, books["cues"][held_index], config.native())
    decoded = _decode(final, books["targets"], abstain_threshold=0.20)
    target_cosine = float((books["targets"] @ _unit(final))[held_index].item())
    gain = float(
        torch.nn.functional.cosine_similarity(final, books["targets"][held_index], dim=0)
        - torch.nn.functional.cosine_similarity(cue_state, books["targets"][held_index], dim=0)
    )
    return {
        "held_out_accuracy": float(decoded == held_index),
        "decoded_index": decoded,
        "held_out_target_cosine": target_cosine,
        "attractor_cosine_gain": gain,
        "chance_baseline": 0.25,
        "decoder_only_baseline_accuracy": float(
            _decode(books["cues"][held_index], books["targets"], abstain_threshold=0.20) == held_index
        ),
        "cutoff_audit": cutoff,
        "hippocampal_rows_after_rollout": rows,
        "snapshot_restore_parity": snapshot_restore_parity,
        "dense_sparse_parity": _dense_sparse_parity(runtime),
        "finite": bool(torch.isfinite(runtime.weight).all()),
        "abstain_threshold": 0.20,
    }


def _m2_condition(
    seed: int,
    config: AlternativeMemoryConfig,
    condition: str,
    task: str,
) -> dict[str, Any]:
    if task not in {"binding", "factor_transfer"}:
        raise ValueError("M2 task must be binding or factor_transfer")
    runtime, fixed_point_residual, projection_passes = _m2_runtime(seed, config)
    base = runtime.snapshot()
    before = runtime.weight.clone()
    if task == "binding":
        temporal, source, _ = _loop8_replay_source_audit()
        train_labels: list[Any] = [int(entry["value"]) for entry in source]
        cues, targets = _codebook(seed, config.dim)
        books = None
    else:
        books = _factor_codebooks(seed, config.dim)
        train_labels = list(books["train"])
        cues, targets = books["cues"], books["targets"]
        temporal = TemporalAuditedMemory(capacity=32)
        for position, (ai, bi) in enumerate(train_labels):
            temporal.ingest(TemporalMemoryEvent(
                f"factor-{ai}-{bi}", "target", f"{ai}:{bi}", 1, position, f"m2-{position}",
            ))

    label_to_index = {
        label: (books["combinations"].index(label) if books is not None else int(label))
        for label in train_labels
    }
    shifted = train_labels[1:] + train_labels[:1]
    cplus_total = torch.zeros_like(before)
    cminus_total = torch.zeros_like(before)
    phase_count = runtime_tick_count = 0
    phase_weight_unchanged = True
    first_term_residual = 0.0
    stdp_updates = 0
    schedule_core: list[dict[str, Any]] = []
    learning_pairs: list[dict[str, Any]] = []

    for epoch in range(config.replay_epochs):
        for position, label in enumerate(train_labels):
            cue_index = label_to_index[label]
            paired_label = shifted[position] if condition == "target_shuffled" else label
            target_index = label_to_index[paired_label]
            positive_value = targets[target_index]
            negative_value = positive_value if condition == "identical_phase" else torch.zeros(config.dim)
            positive = _m2_collect_phase(base, cues[cue_index], positive_value, config)
            negative = _m2_collect_phase(base, cues[cue_index], negative_value, config)
            cplus_total += positive["correlation"]
            cminus_total += negative["correlation"]
            phase_count += 2
            runtime_tick_count += 8
            phase_weight_unchanged = phase_weight_unchanged and positive["weight_unchanged"] and negative["weight_unchanged"]
            first_term_residual = max(
                first_term_residual,
                positive["first_term_residual"],
                negative["first_term_residual"],
            )
            stdp_updates += positive["stdp_updates"] + negative["stdp_updates"]
            schedule_core.append({
                "epoch": epoch,
                "position": position,
                "cue": list(label) if isinstance(label, tuple) else int(label),
                "phase_modes": [["WAKE", "NREM", "NREM", "NREM"]] * 2,
                "phase_rows": [1, 1],
            })
            learning_pairs.append({
                "cue": list(label) if isinstance(label, tuple) else int(label),
                "positive_target": list(paired_label) if isinstance(paired_label, tuple) else int(paired_label),
                "negative": "same_as_positive" if condition == "identical_phase" else "zero",
            })

    block_count = config.replay_epochs * len(train_labels)
    cplus = cplus_total / block_count
    cminus = cminus_total / block_count
    if condition == "positive_only":
        raw = 0.8 * cplus
    elif condition == "negative_only":
        raw = -0.8 * cminus
    elif condition == "sign_reversed":
        raw = 0.8 * (cminus - cplus)
    else:
        raw = 0.8 * (cplus - cminus)

    runtime = BrainRuntime.from_snapshot(base, backend="torch", device="cpu")
    proposed = structural_projection(runtime.weight + raw, **M2_PROJECTION)
    proposed.fill_diagonal_(0.0)
    requested = proposed - runtime.weight
    requested_norm = float(requested.norm().item())
    clipped = requested.clone()
    if requested_norm > config.max_write_norm:
        clipped *= config.max_write_norm / requested_norm
    if condition == "no_write":
        clipped.zero_()
    installed_norm = runtime.install_bounded_recurrent_delta(
        clipped, max_frobenius_norm=config.max_write_norm,
    )
    applied = runtime.weight - before
    applied_reconstruction_residual = float((applied - clipped).norm().item())
    weight_drift = float(applied.norm().item())

    common = {
        "seed": seed,
        "task": task,
        "condition": condition,
        "weight_drift": weight_drift,
        "association_contrast": _association_contrast(
            applied, cues, targets, [label_to_index[label] for label in train_labels],
        ),
        "positive_correlation_norm": float(cplus.norm().item()),
        "negative_correlation_norm": float(cminus.norm().item()),
        "raw_delta_norm": float(raw.norm().item()),
        "proposed_delta_norm": requested_norm,
        "clipped_delta_norm": float(clipped.norm().item()),
        "installed_delta_norm": installed_norm,
        "applied_delta_norm": weight_drift,
        "applied_delta_sha256": _tensor_sha256(applied),
        "applied_reconstruction_residual": applied_reconstruction_residual,
        "projection_fixed_point_residual": fixed_point_residual,
        "projection_passes": projection_passes,
        "projection": dict(M2_PROJECTION),
        "diagonal_max_abs": float(runtime.weight.diagonal().abs().max().item()),
        "block_count": block_count,
        "phase_count": phase_count,
        "runtime_tick_count": runtime_tick_count,
        "install_count": 1,
        "phase_weight_unchanged": bool(phase_weight_unchanged),
        "first_term_orientation_residual": first_term_residual,
        "automatic_stdp_updates": stdp_updates + int(runtime._stdp_updates),
        "schedule_core": schedule_core,
        "schedule_core_sha256": hashlib.sha256(repr(schedule_core).encode("utf-8")).hexdigest(),
        "learning_pairs": learning_pairs,
        "finite_update": bool(torch.isfinite(applied).all()),
    }
    if task == "binding":
        common.update(_evaluate_sealed(
            runtime, temporal, cues, targets,
            [label_to_index[label] for label in train_labels],
            config, abstain_threshold=0.20,
        ))
    else:
        assert books is not None
        common.update(_m2_factor_evaluation(runtime, temporal, books, config))
        common.update({
            "factor_codebook_sha256": books["sha256"],
            "train_combinations": [list(value) for value in books["train"]],
            "held_out_combination": list(books["held_out"]),
            "heldout_absence_audit": (
                books["held_out"] not in books["train"]
                and all(tuple(pair["positive_target"]) != books["held_out"] for pair in learning_pairs)
            ),
        })
    return common


def _m2_integrity(row: dict[str, Any]) -> bool:
    return bool(
        row["projection_fixed_point_residual"] <= 1e-7
        and row["phase_weight_unchanged"]
        and row["first_term_orientation_residual"] <= 1e-7
        and row["automatic_stdp_updates"] == 0
        and row["applied_reconstruction_residual"] <= 1e-7
        and row["diagonal_max_abs"] <= 1e-7
        and row["snapshot_restore_parity"]
        and row["dense_sparse_parity"]
        and row["finite"]
        and row["finite_update"]
        and row["hippocampal_rows_after_rollout"] == 0
        and row["cutoff_audit"]["temporal_rows_after"] == 0
        and row["cutoff_audit"]["hippocampal_rows_after"] == 0
    )


def m2_lagged_contrastive_binding(
    seed: int,
    config: AlternativeMemoryConfig | None = None,
) -> dict[str, Any]:
    config = config or AlternativeMemoryConfig(seed=seed)
    config = AlternativeMemoryConfig(**{**asdict(config), "seed": seed})
    learned = _m2_condition(seed, config, "lagged_contrast", "binding")
    controls = {name: _m2_condition(seed, config, name, "binding") for name in M2_CONDITIONS}
    adverse_names = ("no_write", "target_shuffled", "identical_phase", "sign_reversed")
    strongest_adverse = max(controls[name]["clean_accuracy"] for name in adverse_names)
    advantage = learned["clean_accuracy"] - strongest_adverse
    task_gate = bool(
        learned["clean_accuracy"] >= 0.80
        and learned["corrupt_accuracy"] >= 0.65
        and learned["deleted_abstention"] >= 0.95
        and learned["unknown_abstention"] >= 0.95
        and learned["attractor_cosine_gain"] >= 0.05
    )
    schedule_parity = all(
        control["schedule_core_sha256"] == learned["schedule_core_sha256"]
        for control in controls.values()
    )
    identical_zero = bool(
        controls["identical_phase"]["raw_delta_norm"] <= 1e-7
        and controls["identical_phase"]["applied_delta_norm"] <= 1e-7
    )
    positive_only_same = (
        controls["positive_only"]["applied_delta_sha256"] == learned["applied_delta_sha256"]
    )
    learned.update({
        "route": "M2_lagged_contrastive_binding",
        "controls": controls,
        "task_gate_without_controls": task_gate,
        "control_advantage": advantage,
        "schedule_parity": schedule_parity,
        "identical_phase_zero_update": identical_zero,
        "positive_only_applied_delta_same": positive_only_same,
        "contrastive_negative_nonzero": learned["negative_correlation_norm"] > 1e-7,
    })
    learned["status"] = "GO" if (
        task_gate
        and advantage >= 0.20
        and learned["weight_drift"] > 0.0
        and learned["association_contrast"] > 1e-6
        and learned["contrastive_negative_nonzero"]
        and not positive_only_same
        and identical_zero
        and schedule_parity
        and _m2_integrity(learned)
        and all(_m2_integrity(control) for control in controls.values())
    ) else "STOP"
    return learned


def m2_lagged_contrastive_factor_transfer(
    seed: int,
    config: AlternativeMemoryConfig | None = None,
) -> dict[str, Any]:
    config = config or AlternativeMemoryConfig(seed=seed)
    config = AlternativeMemoryConfig(**{**asdict(config), "seed": seed})
    learned = _m2_condition(seed, config, "lagged_contrast", "factor_transfer")
    controls = {name: _m2_condition(seed, config, name, "factor_transfer") for name in M2_CONDITIONS}
    adverse_names = ("no_write", "target_shuffled", "identical_phase", "sign_reversed")
    strongest_adverse = max(controls[name]["held_out_accuracy"] for name in adverse_names)
    advantage = learned["held_out_accuracy"] - strongest_adverse
    schedule_parity = all(
        control["schedule_core_sha256"] == learned["schedule_core_sha256"]
        for control in controls.values()
    )
    codebook_parity = all(
        control["factor_codebook_sha256"] == learned["factor_codebook_sha256"]
        for control in controls.values()
    )
    identical_zero = bool(
        controls["identical_phase"]["raw_delta_norm"] <= 1e-7
        and controls["identical_phase"]["applied_delta_norm"] <= 1e-7
    )
    positive_only_same = (
        controls["positive_only"]["applied_delta_sha256"] == learned["applied_delta_sha256"]
    )
    learned.update({
        "route": "M2_lagged_contrastive_factor_transfer",
        "controls": controls,
        "control_advantage": advantage,
        "schedule_parity": schedule_parity,
        "codebook_parity": codebook_parity,
        "identical_phase_zero_update": identical_zero,
        "positive_only_applied_delta_same": positive_only_same,
        "contrastive_negative_nonzero": learned["negative_correlation_norm"] > 1e-7,
    })
    learned["status"] = "GO" if (
        learned["held_out_accuracy"] >= 0.70
        and advantage >= 0.20
        and learned["heldout_absence_audit"]
        and learned["contrastive_negative_nonzero"]
        and not positive_only_same
        and identical_zero
        and schedule_parity
        and codebook_parity
        and _m2_integrity(learned)
        and all(_m2_integrity(control) for control in controls.values())
    ) else "STOP"
    return learned


M3_FEATURE_SCHEMA = (
    "activation",
    "refractory",
    "memory_trace",
    "adaptation",
    "stp_u",
    "stp_x",
    "bitfield",
    "lifecycle",
    "inactive_steps",
    "goal",
    "external_drive",
    "effective_replay_drive",
    "forced_mode_onehot_WAKE_NREM_REM",
    "replay_present",
    "bias",
)


def _snapshot_sha256(snapshot: BrainRuntimeSnapshot) -> str:
    digest = hashlib.sha256(repr(asdict(snapshot.config)).encode("utf-8"))
    for name in (
        "weight", "activation", "refractory", "memory_trace", "adaptation",
        "stp_u", "stp_x", "bitfield", "goal", "lifecycle", "inactive_steps",
    ):
        value = getattr(snapshot, name)
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    digest.update(snapshot.mode.value.encode("ascii"))
    digest.update(str(snapshot.step).encode("ascii"))
    return digest.hexdigest()


def _m3_runtime(seed: int, config: AlternativeMemoryConfig) -> BrainRuntime:
    # Use the same exact fixed-point constructor with a disjoint seed offset.
    # This keeps a zero replay-residual update from acquiring an unlogged
    # projection-only weight change at M3's block boundary.
    runtime, residual, _ = _m2_runtime(seed + 1000, config)
    if residual > 1e-7:
        raise RuntimeError("M3 source recurrent matrix is not a projection fixed point")
    return runtime


def _m3_feature(
    runtime: BrainRuntime,
    external: torch.Tensor,
    effective_replay: torch.Tensor,
    mode: RuntimeMode,
    *,
    replay_present: bool,
) -> torch.Tensor:
    mode_onehot = torch.tensor(
        [float(mode is RuntimeMode.WAKE), float(mode is RuntimeMode.NREM), float(mode is RuntimeMode.REM)],
        device=runtime.device,
    )
    feature = torch.cat((
        runtime.activation.float(),
        runtime.refractory.float(),
        runtime.memory_trace.float(),
        runtime.adaptation.float(),
        runtime.stp_u.float(),
        runtime.stp_x.float(),
        runtime.bitfield.float(),
        runtime.lifecycle.float(),
        runtime.inactive_steps.float(),
        runtime.goal.float(),
        external.detach().float().to(runtime.device).view(runtime.config.dim),
        effective_replay.detach().float().to(runtime.device).view(runtime.config.dim),
        mode_onehot,
        torch.tensor([float(replay_present), 1.0], device=runtime.device),
    ))
    expected = 12 * runtime.config.dim + 5
    if feature.numel() != expected:
        raise RuntimeError(f"M3 feature has {feature.numel()} entries; expected {expected}")
    return feature


def _m3_effective_replay(
    runtime: BrainRuntime,
    cue: torch.Tensor,
    mode: RuntimeMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    replay = runtime.hippocampus.recall(cue, topk=runtime.config.memory_topk)
    if mode is not RuntimeMode.WAKE and len(runtime.hippocampus) > 0:
        replay = 0.5 * replay + 0.5 * runtime.hippocampus.replay(mode)
    return replay, runtime.config.replay_mix(mode) * replay


def _m3_actual_gated_pre(runtime: BrainRuntime) -> torch.Tensor:
    active = runtime.active_mask().float()
    stp_u = runtime.stp_u + (
        -STP_TAU_FAC_INV * runtime.stp_u
        + STP_U_BASE * (1.0 - runtime.stp_u) * active
    )
    stp_x = runtime.stp_x + (
        STP_TAU_REC * (1.0 - runtime.stp_x)
        - runtime.stp_u * runtime.stp_x * active
    )
    return stp_u.clamp(0.0, 1.0) * stp_x.clamp(0.0, 1.0) * runtime.activation * active


def _m3_stage_calibration_replay(
    runtime: BrainRuntime,
    replay_vector: torch.Tensor | None,
) -> tuple[torch.Tensor, bool]:
    runtime.hippocampus = HippocampusMemory(
        runtime.config.dim, capacity=runtime.config.memory_capacity, device=runtime.device,
    )
    if replay_vector is None:
        return torch.zeros(runtime.config.dim, device=runtime.device), False
    key = _unit(replay_vector).to(runtime.device)
    runtime.hippocampus.encode(key, value=replay_vector, priority=1.0)
    return key, True


def _m3_transition(
    runtime: BrainRuntime,
    external: torch.Tensor,
    mode: RuntimeMode,
    replay_vector: torch.Tensor | None,
) -> dict[str, Any]:
    cue, replay_present = _m3_stage_calibration_replay(runtime, replay_vector)
    raw_replay, effective_replay = _m3_effective_replay(runtime, cue, mode)
    feature = _m3_feature(
        runtime, external, effective_replay, mode, replay_present=replay_present,
    )
    persistence = runtime.activation.detach().clone()
    before_weight = runtime.weight.clone()
    step = runtime.step(
        external_input=external,
        cue=cue,
        force_mode=mode,
        learning_signal=0.0,
    )
    post = runtime.activation.detach().clone()
    effective_norm_residual = abs(
        float(effective_replay.norm().item())
        - runtime.config.replay_mix(mode) * float(step.replay_norm)
    )
    effective_vector_residual = float(
        (effective_replay - runtime.config.replay_mix(mode) * raw_replay).norm().item()
    )
    return {
        "feature": feature,
        "post": post,
        "persistence": persistence,
        "raw_replay": raw_replay,
        "effective_replay": effective_replay,
        "replay_present": replay_present,
        "effective_norm_residual": effective_norm_residual,
        "effective_vector_residual": effective_vector_residual,
        "weight_unchanged": bool(torch.equal(runtime.weight, before_weight)),
        "stdp_updates": int(runtime._stdp_updates),
    }


def _m3_calibration_codebooks(seed: int, dim: int) -> dict[str, torch.Tensor]:
    action_generator = torch.Generator(device="cpu").manual_seed(seed + 5101)
    fit_replay_generator = torch.Generator(device="cpu").manual_seed(seed + 5102)
    score_replay_generator = torch.Generator(device="cpu").manual_seed(seed + 5103)
    return {
        "actions": _orthonormal_rows(action_generator, 4, dim),
        "fit_replay": _orthonormal_rows(fit_replay_generator, 4, dim),
        "score_replay": _orthonormal_rows(score_replay_generator, 4, dim),
    }


def _fit_m3_predictor(seed: int, config: AlternativeMemoryConfig) -> dict[str, Any]:
    runtime = _m3_runtime(seed, config)
    source_snapshot = runtime.snapshot()
    initial_weight = runtime.weight.clone()
    codebooks = _m3_calibration_codebooks(seed, config.dim)
    actions = codebooks["actions"]
    fit_replay = codebooks["fit_replay"]
    fit_modes = (RuntimeMode.WAKE, RuntimeMode.NREM, RuntimeMode.REM, RuntimeMode.WAKE)
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    fit_hashes: list[str] = []
    replay_norm_residual = 0.0
    replay_vector_residual = 0.0
    weight_unchanged = True
    stdp_updates = 0
    for row in range(64):
        action_index = row % 4
        replay_vector = fit_replay[(row // 2) % 4] if row % 2 else None
        transition = _m3_transition(
            runtime,
            actions[action_index],
            fit_modes[row % 4],
            replay_vector,
        )
        features.append(transition["feature"])
        targets.append(transition["post"])
        fit_hashes.append(_tensor_sha256(transition["feature"]))
        replay_norm_residual = max(replay_norm_residual, transition["effective_norm_residual"])
        replay_vector_residual = max(replay_vector_residual, transition["effective_vector_residual"])
        weight_unchanged = weight_unchanged and transition["weight_unchanged"]
        stdp_updates += transition["stdp_updates"]

    x = torch.stack(features).double()
    y = torch.stack(targets).double()
    ridge = 1e-4
    # This dual solve is algebraically identical to the frozen primal ridge
    # expression and avoids forming a 581x581 inverse from only 64 rows.
    theta = x.T @ torch.linalg.solve(
        x @ x.T + ridge * torch.eye(x.shape[0], dtype=x.dtype),
        y,
    )
    theta_hash_before_score = _tensor_sha256(theta)
    pre_score_snapshot = runtime.snapshot()

    score_replay = codebooks["score_replay"]
    score_actions = (3, 2, 1, 0) * 4
    score_modes = (RuntimeMode.WAKE, RuntimeMode.REM, RuntimeMode.NREM, RuntimeMode.WAKE) * 4
    model_squared_errors: list[float] = []
    persistence_squared_errors: list[float] = []
    score_hashes: list[str] = []
    for row, (action_index, mode) in enumerate(zip(score_actions, score_modes)):
        fork = BrainRuntime.from_snapshot(pre_score_snapshot, backend="torch", device="cpu")
        replay_vector = score_replay[(row // 2) % 4] if row % 2 else None
        transition = _m3_transition(
            fork,
            1.5 * actions[action_index],
            mode,
            replay_vector,
        )
        feature = transition["feature"].double()
        prediction = feature @ theta
        actual = transition["post"].double()
        persistence = transition["persistence"].double()
        model_squared_errors.append(float(torch.mean((prediction - actual) ** 2).item()))
        persistence_squared_errors.append(float(torch.mean((persistence - actual) ** 2).item()))
        score_hashes.append(_tensor_sha256(transition["feature"]))
        replay_norm_residual = max(replay_norm_residual, transition["effective_norm_residual"])
        replay_vector_residual = max(replay_vector_residual, transition["effective_vector_residual"])
        weight_unchanged = weight_unchanged and transition["weight_unchanged"]
        stdp_updates += transition["stdp_updates"]

    model_mse = sum(model_squared_errors) / len(model_squared_errors)
    persistence_mse = sum(persistence_squared_errors) / len(persistence_squared_errors)
    theta_hash_after_score = _tensor_sha256(theta)
    task_cues, task_targets = _codebook(seed, config.dim)
    calibration = torch.cat((fit_replay, score_replay), dim=0)
    task_vectors = torch.cat((task_cues, task_targets), dim=0)
    max_task_overlap = float((calibration @ task_vectors.T).abs().max().item())
    result = {
        "seed": seed,
        "route": "M3_frozen_native_transition_predictor",
        "fit_row_count": 64,
        "heldout_row_count": 16,
        "feature_dim": int(x.shape[1]),
        "expected_feature_dim": 12 * config.dim + 5,
        "feature_schema": list(M3_FEATURE_SCHEMA),
        "feature_schema_sha256": hashlib.sha256(repr(M3_FEATURE_SCHEMA).encode("utf-8")).hexdigest(),
        "ridge": ridge,
        "theta_shape": list(theta.shape),
        "theta_sha256": theta_hash_before_score,
        "theta_frozen_during_score": theta_hash_before_score == theta_hash_after_score,
        "source_snapshot_sha256": _snapshot_sha256(source_snapshot),
        "pre_score_snapshot_sha256": _snapshot_sha256(pre_score_snapshot),
        "fit_feature_sha256": fit_hashes,
        "score_feature_sha256": score_hashes,
        "fit_score_row_disjoint": set(fit_hashes).isdisjoint(score_hashes),
        "model_mse": model_mse,
        "persistence_mse": persistence_mse,
        "mse_ratio": model_mse / max(persistence_mse, 1e-15),
        "effective_replay_norm_residual_max": replay_norm_residual,
        "effective_replay_vector_residual_max": replay_vector_residual,
        "fit_replay_present_rows": 32,
        "heldout_replay_present_rows": 8,
        "calibration_seed_offsets": {"actions": 5101, "fit_replay": 5102, "score_replay": 5103},
        "calibration_codebook_sha256": {
            name: _tensor_sha256(value) for name, value in codebooks.items()
        },
        "calibration_task_max_abs_cosine": max_task_overlap,
        "automatic_stdp_updates": stdp_updates + int(runtime._stdp_updates),
        "weight_unchanged": bool(weight_unchanged and torch.equal(runtime.weight, initial_weight)),
        "finite": bool(torch.isfinite(theta).all() and torch.isfinite(torch.tensor(model_mse))),
    }
    result["status"] = "GO" if (
        result["feature_dim"] == result["expected_feature_dim"]
        and result["theta_frozen_during_score"]
        and result["fit_score_row_disjoint"]
        and result["effective_replay_vector_residual_max"] <= 1e-7
        and result["automatic_stdp_updates"] == 0
        and result["weight_unchanged"]
        and result["finite"]
        and model_mse <= 0.90 * persistence_mse
    ) else "STOP"
    return {"result": result, "theta": theta, "source_snapshot": source_snapshot}


def m3_predictor_audit(
    seed: int,
    config: AlternativeMemoryConfig | None = None,
) -> dict[str, Any]:
    config = config or AlternativeMemoryConfig(seed=seed)
    config = AlternativeMemoryConfig(**{**asdict(config), "seed": seed})
    return _fit_m3_predictor(seed, config)["result"]


M3_CONDITIONS = (
    "predictor_only",
    "transition_order_shuffled",
    "one_block_delayed_error",
    "sign_flipped_error",
    "no_replay",
    "target_shuffled",
)


def _m3_install_block(
    runtime: BrainRuntime,
    raw: torch.Tensor,
    config: AlternativeMemoryConfig,
    *,
    write_enabled: bool,
) -> dict[str, Any]:
    before = runtime.weight.clone()
    proposed = structural_projection(before + raw, **M2_PROJECTION)
    proposed.fill_diagonal_(0.0)
    requested = proposed - before
    requested_norm = float(requested.norm().item())
    clipped = requested.clone()
    if requested_norm > config.max_write_norm:
        clipped *= config.max_write_norm / requested_norm
    if not write_enabled:
        clipped.zero_()
    installed = runtime.install_bounded_recurrent_delta(
        clipped, max_frobenius_norm=config.max_write_norm,
    )
    applied = runtime.weight - before
    clipped_norm = float(clipped.norm().item())
    return {
        "raw_norm": float(raw.norm().item()),
        "raw_sha256": _tensor_sha256(raw),
        "proposed_delta_norm": requested_norm,
        "proposed_weight_sha256": _tensor_sha256(proposed),
        "requested_delta_sha256": _tensor_sha256(requested),
        "clipped_delta_norm": clipped_norm,
        "clipped_delta_sha256": _tensor_sha256(clipped),
        "installed_delta_norm": installed,
        "installed_norm_residual": abs(installed - clipped_norm),
        "applied_delta_norm": float(applied.norm().item()),
        "applied_delta_sha256": _tensor_sha256(applied),
        "applied_reconstruction_residual": float((applied - clipped).norm().item()),
        "clipped": requested_norm > config.max_write_norm,
        "diagonal_max_abs": float(runtime.weight.diagonal().abs().max().item()),
        "dense_sparse_parity": _dense_sparse_parity(runtime),
        "finite": bool(torch.isfinite(runtime.weight).all()),
    }


def _m3_condition(
    seed: int,
    config: AlternativeMemoryConfig,
    condition: str,
    task: str,
    predictor_bundle: dict[str, Any],
) -> dict[str, Any]:
    if task not in {"binding", "factor_transfer"}:
        raise ValueError("M3 task must be binding or factor_transfer")
    theta: torch.Tensor = predictor_bundle["theta"]
    source_snapshot: BrainRuntimeSnapshot = predictor_bundle["source_snapshot"]
    predictor_result: dict[str, Any] = predictor_bundle["result"]
    runtime = BrainRuntime.from_snapshot(source_snapshot, backend="torch", device="cpu")
    initial_weight = runtime.weight.clone()
    if task == "binding":
        temporal, source, _ = _loop8_replay_source_audit()
        train_labels: list[Any] = [int(entry["value"]) for entry in source]
        cues, targets = _codebook(seed, config.dim)
        books = None
    else:
        books = _factor_codebooks(seed, config.dim)
        train_labels = list(books["train"])
        cues, targets = books["cues"], books["targets"]
        temporal = TemporalAuditedMemory(capacity=32)
        for position, (ai, bi) in enumerate(train_labels):
            temporal.ingest(TemporalMemoryEvent(
                f"factor-{ai}-{bi}", "target", f"{ai}:{bi}", 1, position, f"m3-{position}",
            ))
    label_to_index = {
        label: (books["combinations"].index(label) if books is not None else int(label))
        for label in train_labels
    }
    shifted = train_labels[1:] + train_labels[:1]
    previous_block_errors: list[torch.Tensor] | None = None
    block_audits: list[dict[str, Any]] = []
    schedule_core: list[dict[str, Any]] = []
    learning_pairs: list[dict[str, Any]] = []
    q_credit_norms: list[float] = []
    q_credit_hashes: list[str] = []
    recurrent_pre_norms: list[list[float]] = []
    mid_block_weight_unchanged = True
    update_formula_residual = 0.0
    replay_vector_residual = 0.0
    automatic_stdp_updates = 0

    for epoch in range(config.replay_epochs):
        for position, label in enumerate(train_labels):
            cue_index = label_to_index[label]
            paired_label = shifted[position] if condition == "target_shuffled" else label
            target_index = label_to_index[paired_label]
            cue, target = cues[cue_index], targets[target_index]
            runtime.reset_evaluation_state()
            runtime.hippocampus = HippocampusMemory(
                config.dim, capacity=runtime.config.memory_capacity, device=runtime.device,
            )
            before_phase = runtime.weight.clone()
            runtime.step(
                external_input=config.cue_drive_gain * cue,
                cue=cue,
                force_mode=RuntimeMode.WAKE,
                learning_signal=0.0,
            )
            q_credit = _unit(runtime.activation.detach().clone())
            q_credit_norms.append(float(q_credit.norm().item()))
            q_credit_hashes.append(_tensor_sha256(q_credit))
            staged_value = torch.zeros_like(target) if condition == "no_replay" else target
            runtime.hippocampus.encode(cue, value=staged_value, priority=1.0)
            runtime.reset_evaluation_state()

            errors: list[torch.Tensor] = []
            credits: list[torch.Tensor] = []
            pre_norms: list[float] = []
            feature_hashes: list[str] = []
            replay_hashes: list[str] = []
            pre_hashes: list[str] = []
            for replay_tick in range(3):
                raw_replay, effective_replay = _m3_effective_replay(
                    runtime, cue, RuntimeMode.NREM,
                )
                feature = _m3_feature(
                    runtime,
                    torch.zeros(config.dim),
                    effective_replay,
                    RuntimeMode.NREM,
                    replay_present=True,
                )
                actual_pre = _m3_actual_gated_pre(runtime).detach().clone()
                feature_hashes.append(_tensor_sha256(feature))
                replay_hashes.append(_tensor_sha256(effective_replay))
                pre_hashes.append(_tensor_sha256(actual_pre))
                runtime.step(
                    external_input=torch.zeros(config.dim),
                    cue=cue,
                    force_mode=RuntimeMode.NREM,
                    learning_signal=0.0,
                )
                prediction = feature.double() @ theta
                error = (runtime.activation.double() - prediction).float()
                errors.append(error)
                credits.append(q_credit if replay_tick == 0 else actual_pre)
                pre_norms.append(float(actual_pre.norm().item()))
                replay_vector_residual = max(
                    replay_vector_residual,
                    float((effective_replay - runtime.config.replay_mix(RuntimeMode.NREM) * raw_replay).norm().item()),
                )
                mid_block_weight_unchanged = mid_block_weight_unchanged and torch.equal(
                    runtime.weight, before_phase,
                )
                automatic_stdp_updates += int(runtime._stdp_updates)
            recurrent_pre_norms.append(pre_norms)

            if condition == "transition_order_shuffled":
                used_errors = [errors[index] for index in (1, 2, 0)]
            elif condition == "one_block_delayed_error":
                used_errors = (
                    [torch.zeros_like(error) for error in errors]
                    if previous_block_errors is None
                    else [error.clone() for error in previous_block_errors]
                )
            elif condition == "sign_flipped_error":
                used_errors = [-error for error in errors]
            else:
                used_errors = errors
            formula_terms = [torch.outer(error, credit) for error, credit in zip(used_errors, credits)]
            raw = (0.8 / 3.0) * torch.stack(formula_terms).sum(dim=0)
            reconstructed = (0.8 / 3.0) * (
                torch.outer(used_errors[0], credits[0])
                + torch.outer(used_errors[1], credits[1])
                + torch.outer(used_errors[2], credits[2])
            )
            update_formula_residual = max(
                update_formula_residual, float((raw - reconstructed).norm().item()),
            )
            previous_block_errors = [error.detach().clone() for error in errors]
            apply_audit = _m3_install_block(
                runtime,
                raw,
                config,
                write_enabled=condition != "predictor_only",
            )
            block_audits.append({
                "epoch": epoch,
                "position": position,
                "cue": list(label) if isinstance(label, tuple) else int(label),
                "target": list(paired_label) if isinstance(paired_label, tuple) else int(paired_label),
                "error_norms": [float(error.norm().item()) for error in errors],
                "credit_norms": [float(credit.norm().item()) for credit in credits],
                "feature_sha256": feature_hashes,
                "effective_replay_sha256": replay_hashes,
                "actual_pre_sha256": pre_hashes,
                "residual_credit_permutation": (
                    [1, 2, 0] if condition == "transition_order_shuffled" else [0, 1, 2]
                ),
                "delayed_source_block": (
                    len(block_audits) - 1 if condition == "one_block_delayed_error" else None
                ),
                **apply_audit,
            })
            schedule_core.append({
                "epoch": epoch,
                "position": position,
                "cue": list(label) if isinstance(label, tuple) else int(label),
                "mode_sequence": ["WAKE", "NREM", "NREM", "NREM"],
                "reset_after_tick": 0,
                "install_count": 1,
            })
            learning_pairs.append({
                "cue": list(label) if isinstance(label, tuple) else int(label),
                "target": list(paired_label) if isinstance(paired_label, tuple) else int(paired_label),
                "replay_present": True,
                "replay_value": "zero" if condition == "no_replay" else "assigned_target",
            })

    weight_delta = runtime.weight - initial_weight
    common = {
        "seed": seed,
        "task": task,
        "condition": condition,
        "predictor_status": predictor_result["status"],
        "predictor_mse_ratio": predictor_result["mse_ratio"],
        "predictor_theta_sha256": predictor_result["theta_sha256"],
        "predictor_source_snapshot_sha256": predictor_result["source_snapshot_sha256"],
        "predictor_frozen": _tensor_sha256(theta) == predictor_result["theta_sha256"],
        "weight_drift": float(weight_delta.norm().item()),
        "weight_delta_sha256": _tensor_sha256(weight_delta),
        "association_contrast": _association_contrast(
            weight_delta, cues, targets, [label_to_index[label] for label in train_labels],
        ),
        "block_count": len(block_audits),
        "runtime_tick_count": 4 * len(block_audits),
        "install_count": len(block_audits),
        "mid_block_weight_unchanged": bool(mid_block_weight_unchanged),
        "automatic_stdp_updates": automatic_stdp_updates + int(runtime._stdp_updates),
        "update_formula_residual_max": update_formula_residual,
        "applied_reconstruction_residual_max": max(
            audit["applied_reconstruction_residual"] for audit in block_audits
        ),
        "installed_norm_residual_max": max(
            audit["installed_norm_residual"] for audit in block_audits
        ),
        "all_block_updates_finite": all(audit["finite"] for audit in block_audits),
        "all_block_dense_sparse_parity": all(audit["dense_sparse_parity"] for audit in block_audits),
        "all_block_diagonal_zero": all(audit["diagonal_max_abs"] <= 1e-7 for audit in block_audits),
        "all_writes_at_block_end": True,
        "q_credit_norm_min": min(q_credit_norms),
        "q_credit_norm_max": max(q_credit_norms),
        "q_credit_sha256": q_credit_hashes,
        "recurrent_pre_norms": recurrent_pre_norms,
        "effective_replay_vector_residual_max": replay_vector_residual,
        "schedule_core": schedule_core,
        "schedule_core_sha256": hashlib.sha256(repr(schedule_core).encode("utf-8")).hexdigest(),
        "learning_pairs": learning_pairs,
        "block_apply_audits": block_audits,
        "target_identifier_in_predictor_feature": False,
        "decoder_output_in_update": False,
        "teacher_signal_boundary": "continuous_replay_state_is_predictor_input_and_teacher",
    }
    if task == "binding":
        common.update(_evaluate_sealed(
            runtime, temporal, cues, targets,
            [label_to_index[label] for label in train_labels],
            config, abstain_threshold=0.20,
        ))
    else:
        assert books is not None
        common.update(_m2_factor_evaluation(runtime, temporal, books, config))
        common.update({
            "factor_codebook_sha256": books["sha256"],
            "train_combinations": [list(value) for value in books["train"]],
            "held_out_combination": list(books["held_out"]),
            "heldout_absence_audit": (
                books["held_out"] not in books["train"]
                and all(tuple(pair["target"]) != books["held_out"] for pair in learning_pairs)
            ),
        })
    return common


def _m3_memory_integrity(row: dict[str, Any]) -> bool:
    return bool(
        row["predictor_frozen"]
        and row["mid_block_weight_unchanged"]
        and row["automatic_stdp_updates"] == 0
        and row["update_formula_residual_max"] <= 1e-7
        and row["applied_reconstruction_residual_max"] <= 1e-7
        and row["installed_norm_residual_max"] <= 1e-7
        and row["all_block_updates_finite"]
        and row["all_block_dense_sparse_parity"]
        and row["all_block_diagonal_zero"]
        and row["q_credit_norm_min"] >= 1.0 - 1e-6
        and row["q_credit_norm_max"] <= 1.0 + 1e-6
        and row["effective_replay_vector_residual_max"] <= 1e-7
        and not row["target_identifier_in_predictor_feature"]
        and not row["decoder_output_in_update"]
        and row["snapshot_restore_parity"]
        and row["dense_sparse_parity"]
        and row["finite"]
        and row["hippocampal_rows_after_rollout"] == 0
        and row["cutoff_audit"]["temporal_rows_after"] == 0
        and row["cutoff_audit"]["hippocampal_rows_after"] == 0
    )


def m3_replay_residual_binding(
    seed: int,
    config: AlternativeMemoryConfig | None = None,
) -> dict[str, Any]:
    config = config or AlternativeMemoryConfig(seed=seed)
    config = AlternativeMemoryConfig(**{**asdict(config), "seed": seed})
    predictor = _fit_m3_predictor(seed, config)
    learned = _m3_condition(seed, config, "replay_residual", "binding", predictor)
    controls = {
        name: _m3_condition(seed, config, name, "binding", predictor)
        for name in M3_CONDITIONS
    }
    strongest_control = max(control["clean_accuracy"] for control in controls.values())
    advantage = learned["clean_accuracy"] - strongest_control
    schedule_parity = all(
        control["schedule_core_sha256"] == learned["schedule_core_sha256"]
        for control in controls.values()
    )
    task_gate = bool(
        learned["clean_accuracy"] >= 0.80
        and learned["corrupt_accuracy"] >= 0.65
        and learned["deleted_abstention"] >= 0.95
        and learned["unknown_abstention"] >= 0.95
        and learned["attractor_cosine_gain"] >= 0.05
    )
    learned.update({
        "route": "M3_teacher_forced_replay_residual_binding",
        "controls": controls,
        "task_gate_without_controls": task_gate,
        "control_advantage": advantage,
        "schedule_parity": schedule_parity,
        "prediction_gate_passed": predictor["result"]["status"] == "GO",
    })
    learned["status"] = "GO" if (
        task_gate
        and advantage >= 0.20
        and learned["weight_drift"] > 0.0
        and learned["association_contrast"] > 1e-6
        and schedule_parity
        and _m3_memory_integrity(learned)
        and all(_m3_memory_integrity(control) for control in controls.values())
    ) else "STOP"
    return learned


def m3_replay_residual_factor_transfer(
    seed: int,
    config: AlternativeMemoryConfig | None = None,
) -> dict[str, Any]:
    config = config or AlternativeMemoryConfig(seed=seed)
    config = AlternativeMemoryConfig(**{**asdict(config), "seed": seed})
    predictor = _fit_m3_predictor(seed, config)
    learned = _m3_condition(seed, config, "replay_residual", "factor_transfer", predictor)
    controls = {
        name: _m3_condition(seed, config, name, "factor_transfer", predictor)
        for name in M3_CONDITIONS
    }
    strongest_control = max(control["held_out_accuracy"] for control in controls.values())
    advantage = learned["held_out_accuracy"] - strongest_control
    schedule_parity = all(
        control["schedule_core_sha256"] == learned["schedule_core_sha256"]
        for control in controls.values()
    )
    codebook_parity = all(
        control["factor_codebook_sha256"] == learned["factor_codebook_sha256"]
        for control in controls.values()
    )
    learned.update({
        "route": "M3_teacher_forced_replay_residual_factor_transfer",
        "controls": controls,
        "control_advantage": advantage,
        "schedule_parity": schedule_parity,
        "codebook_parity": codebook_parity,
        "prediction_gate_passed": predictor["result"]["status"] == "GO",
    })
    learned["status"] = "GO" if (
        learned["held_out_accuracy"] >= 0.70
        and advantage >= 0.20
        and learned["heldout_absence_audit"]
        and schedule_parity
        and codebook_parity
        and _m3_memory_integrity(learned)
        and all(_m3_memory_integrity(control) for control in controls.values())
    ) else "STOP"
    return learned
