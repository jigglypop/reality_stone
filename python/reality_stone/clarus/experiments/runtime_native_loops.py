"""Deterministic, opt-in native BrainRuntime experiment for memory Loops 6--10.

This is deliberately a harness, not a new default runtime policy.  In
particular, all reported recall is decoded from the live runtime activation
after the episodic stores have been physically replaced by empty stores.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from ..agent import RuntimeAgent, RuntimeAgentConfig
from ..runtime import BrainRuntime, BrainRuntimeConfig, HippocampusMemory, RuntimeMode
from .runtime_temporal_memory import RuntimeTemporalAgent, TemporalAgentQuery, TemporalMemoryController
from ..temporal_memory import TemporalAuditedMemory, TemporalMemoryEvent, TemporalOperation


@dataclass(frozen=True)
class NativeLoopsConfig:
    dim: int = 48
    replay_epochs: int = 12
    rollout_horizon: int = 6
    cue_corruption: float = 0.15
    # Unit-norm codebook coordinates are about 1/sqrt(dim/2).  A fixed gain of
    # five puts a typical coordinate above the runtime's active/schmitt
    # thresholds after the external and WAKE gains are applied.
    cue_drive_gain: float = 5.0
    bounded_write_gain: float = 1.0
    seed: int = 97101


def _unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=0) if float(x.norm()) else x.float()


def _orthonormal_rows(generator: torch.Generator, count: int, width: int) -> torch.Tensor:
    """Seed-fixed codewords with no cue/value pairing information."""
    q, _ = torch.linalg.qr(torch.randn(width, count, generator=generator), mode="reduced")
    return q.T.contiguous()


def _runtime(seed: int, config: NativeLoopsConfig) -> BrainRuntime:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight = torch.randn(config.dim, config.dim, generator=generator) * 0.025
    weight.fill_diagonal_(0.0)
    return BrainRuntime(weight, config=BrainRuntimeConfig(
        dim=config.dim, active_ratio=0.25, noise_sigma=0.0, dale_law=False,
        axon_delay=False, f1_self_measure=False, stdp_enabled=True,
        stdp_interval=1, stdp_apply_interval=1, stdp_lr=0.08,
        stdp_density=1.0, stdp_gate_threshold=0.0, stdp_spike_threshold=0.05,
        stdp_gate_mode="external_signed", stdp_orientation="causal",
        memory_capacity=16, replay_gain=1.0,
    ), backend="torch", device="cpu")


def _codebook(seed: int, dim: int, count: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed + 1009)
    half = dim // 2
    cues, targets = torch.zeros(count, dim), torch.zeros(count, dim)
    cues[:, :half] = _orthonormal_rows(gen, count, half)
    targets[:, half:] = _orthonormal_rows(gen, count, dim - half)
    return cues, targets


def _decode(state: torch.Tensor, codebook: torch.Tensor, *, abstain_threshold: float = 0.15) -> int | None:
    scores = codebook @ _unit(state)
    best = int(scores.argmax().item())
    return best if float(scores[best]) >= abstain_threshold else None


def _detach(runtime: BrainRuntime, temporal: TemporalAuditedMemory) -> dict[str, int]:
    """Remove both stores, rather than merely disabling their read methods."""
    temporal_rows = len(temporal)
    hippocampal_rows = len(runtime.hippocampus)
    temporal._versions.clear()
    temporal._evidence_ids.clear()
    runtime.hippocampus = HippocampusMemory(runtime.config.dim, capacity=runtime.config.memory_capacity, device=runtime.device)
    runtime.config.hippocampal_encoding_enabled = False
    runtime.reset_evaluation_state()
    return {"temporal_rows_removed": temporal_rows, "hippocampal_rows_removed": hippocampal_rows,
            "temporal_rows_after": len(temporal), "hippocampal_rows_after": len(runtime.hippocampus)}


def _probe_rollout(snapshot: object, cue: torch.Tensor, config: NativeLoopsConfig) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Run one query from an identical, sealed consolidated snapshot."""
    runtime = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    runtime.step(external_input=config.cue_drive_gain * cue, force_mode=RuntimeMode.WAKE, learning_signal=0.0)
    cue_state = runtime.activation.clone()
    for _ in range(config.rollout_horizon):
        runtime.step(external_input=torch.zeros(config.dim), force_mode=RuntimeMode.WAKE, learning_signal=0.0)
    return cue_state, runtime.activation.clone(), len(runtime.hippocampus)


def loop6_temporal_selection() -> dict[str, Any]:
    memory = TemporalAuditedMemory(capacity=32)
    events = [
        TemporalMemoryEvent("a", "r", "old", 1, 1, "e1"),
        TemporalMemoryEvent("a", "r", "new", 3, 1, "e3"),
        TemporalMemoryEvent("b", "r", "gone", 1, 1, "e4"),
        TemporalMemoryEvent("b", "r", None, 2, 1, "e5", TemporalOperation.DELETE),
    ]
    for event in (events[1], events[0], events[3], events[2]):
        memory.ingest(event)
    correct = memory.recall("a", "r").value == "new"
    abstain = memory.recall("b", "r").abstained and memory.recall("missing", "r").abstained
    # Arrival-order ablation is intentionally wrong for the late old version.
    arrival_latest = events[0].value
    shuffle_accuracy = float(arrival_latest == "new")
    return {"selected_version_accuracy": float(correct), "abstention_accuracy": float(abstain),
            "temporal_shuffle_latest_accuracy": shuffle_accuracy,
            "shuffle_drop": 1.0 - shuffle_accuracy,
            "status": "GO" if correct and abstain and (1.0 - shuffle_accuracy) >= .20 else "STOP"}


def loop7_agent_route(seed: int, config: NativeLoopsConfig) -> dict[str, Any]:
    runtime = _runtime(seed, config)
    agent = RuntimeAgent(runtime, config=RuntimeAgentConfig(action_count=3))
    memory = TemporalAuditedMemory(capacity=8)
    memory.ingest(TemporalMemoryEvent("a", "r", "v", 1, 1, "route"))
    wrapper = RuntimeTemporalAgent(agent, controller=TemporalMemoryController(memory, enabled=True), answer_action_index=0, abstain_action_index=1)
    observation = torch.ones(config.dim)
    fact = wrapper.step(query=TemporalAgentQuery("q", "fact", "a", relation="r"), observation=observation, force_mode=RuntimeMode.WAKE)
    control = wrapper.step(query=TemporalAgentQuery("q2", "control", "a"), observation=observation, force_mode=RuntimeMode.WAKE)
    disabled = RuntimeTemporalAgent(agent, controller=TemporalMemoryController(memory, enabled=False), answer_action_index=0, abstain_action_index=1)
    reads_before = memory.recall_count
    disabled_out = disabled.step(query=TemporalAgentQuery("q3", "fact", "a", relation="r"), observation=observation, force_mode=RuntimeMode.WAKE)
    disabled_reads = memory.recall_count - reads_before
    precision = float(fact.decision.route == "memory" and control.decision.route != "memory")
    # A supplied fact is authoritative for this query even when the temporal
    # store contains a conflicting older value.  Record the read delta here,
    # rather than inferring it from the decision route, so the benchmark
    # catches a future fallback read.
    context_reads_before = memory.recall_count
    context = wrapper.step(
        query=TemporalAgentQuery(
            "q-context", "fact", "a", relation="r",
            context_value="supplied-v", context_evidence_id="supplied-e",
        ),
        observation=observation,
        force_mode=RuntimeMode.WAKE,
    )
    context_reads = memory.recall_count - context_reads_before
    context_accuracy = float(
        context.decision.route == "context"
        and context.value == "supplied-v"
        and context.evidence_id == "supplied-e"
    )
    return {"route_precision": precision, "route_recall_accuracy": float(fact.value == "v"),
            "disabled_temporal_reads": disabled_reads, "disabled_matches_base": disabled_out.action_index == disabled_out.base_step.action_index,
            "context_precedence_accuracy": context_accuracy, "context_temporal_reads": context_reads,
            "status": "GO" if (precision == 1.0 and fact.value == "v"
                                 and context_accuracy == 1.0 and context_reads == 0
                                 and disabled_reads == 0
                                 and disabled_out.action_index == disabled_out.base_step.action_index) else "STOP"}


def _loop8_replay_source_audit(*, reverse_arrival: bool = False) -> tuple[TemporalAuditedMemory, list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve replay inputs exclusively through latest-valid temporal reads.

    The intentionally adversarial arrival order puts current UPSERT and DELETE
    events before their stale predecessors.  ``arrival_last`` is retained only
    as a negative control; it is never used to form a replay pair.
    """
    events = [
        TemporalMemoryEvent("episode-0", "target", "0", 3, 1, "e0-current"),
        TemporalMemoryEvent("episode-0", "target", "3", 1, 1, "e0-stale"),
        TemporalMemoryEvent("episode-1", "target", None, 3, 1, "e1-delete", TemporalOperation.DELETE),
        TemporalMemoryEvent("episode-1", "target", "1", 1, 1, "e1-stale"),
        TemporalMemoryEvent("episode-2", "target", "2", 2, 1, "e2-current"),
        TemporalMemoryEvent("episode-3", "target", "3", 2, 1, "e3-current"),
    ]
    arrival = list(reversed(events)) if reverse_arrival else events
    memory = TemporalAuditedMemory(capacity=32, max_versions_per_key=3)
    arrival_last: dict[tuple[str, str], TemporalMemoryEvent] = {}
    for event in arrival:
        memory.ingest(event)
        arrival_last[event.key] = event

    resolved: list[dict[str, Any]] = []
    for subject in ("episode-0", "episode-1", "episode-2", "episode-3"):
        recall = memory.recall(subject, "target")
        if recall.abstained or recall.value is None:
            continue
        resolved.append({"key": [subject, "target"], "value": recall.value,
                         "evidence": recall.evidence_id,
                         "session": recall.valid_session})
    arrival_manifest = [
        {"key": [key[0], key[1]], "value": event.value,
         "evidence": event.evidence_id, "session": event.valid_session}
        for key, event in sorted(arrival_last.items())
        if event.operation is TemporalOperation.UPSERT and event.value is not None
    ]
    return memory, resolved, arrival_manifest


def _loop8_condition(seed: int, config: NativeLoopsConfig, condition: str) -> dict[str, Any]:
    runtime = _runtime(seed, config)
    temporal, source, arrival_last = _loop8_replay_source_audit()
    _, reversed_source, _ = _loop8_replay_source_audit(reverse_arrival=True)
    cues, targets = _codebook(seed, config.dim)
    before = runtime.weight.clone()
    replay_source_audit = [dict(entry, replayed=condition != "no_replay") for entry in source]
    # The temporal selector determines which cue/value rows are staged.  Once
    # staged, consolidation reads the value through HippocampusMemory during a
    # zero-input NREM step; target identity never enters the scalar gate.
    runtime.config.hippocampal_encoding_enabled = False
    for source_index, entry in enumerate(source):
        index = int(entry["value"])
        cue, target = cues[index], targets[index]
        paired = targets[(int(source[(source_index + 1) % len(source)]["value"]))] if condition == "target_shuffled" else target
        if condition != "no_replay":
            # Increasing priority makes the current episode the global NREM
            # replay item while exact cue recall returns the same paired value.
            runtime.hippocampus.encode(cue, value=paired, priority=float(source_index + 1))
        for _ in range(config.replay_epochs):
            runtime.reset_evaluation_state()
            runtime.step(
                external_input=config.cue_drive_gain * cue,
                cue=cue,
                force_mode=RuntimeMode.WAKE,
                learning_signal=0.0,
            )
            runtime.step(
                external_input=torch.zeros(config.dim),
                cue=cue,
                force_mode=RuntimeMode.NREM,
                learning_signal=1.0 if condition != "no_replay" else 0.0,
            )
    cutoff = _detach(runtime, temporal)
    sealed_snapshot = runtime.snapshot()
    restored = BrainRuntime.from_snapshot(sealed_snapshot, backend="torch", device="cpu")
    snapshot_restore_parity = bool(torch.equal(restored.weight, runtime.weight) and len(restored.hippocampus) == 0)
    predicted, predicted_ids, cosine_gain, final_target_cosines, final_max_cosines, corrupted = [], [], [], [], [], []
    rows_after_rollout = []
    evaluable_indices = [int(entry["value"]) for entry in source]
    for index in evaluable_indices:
        cue = cues[index]
        cue_state, final, rows = _probe_rollout(sealed_snapshot, cue, config)
        decoded = _decode(final, targets)
        scores = targets @ _unit(final)
        rows_after_rollout.append(rows); predicted.append(decoded == index); predicted_ids.append(decoded)
        final_target_cosines.append(float(scores[index])); final_max_cosines.append(float(scores.max()))
        cosine_gain.append(float(F.cosine_similarity(final, targets[index], dim=0) - F.cosine_similarity(cue_state, targets[index], dim=0)))
        noisy = cue.clone(); noisy[:max(1, int(config.dim * config.cue_corruption))] = 0.0
        _, noisy_final, rows = _probe_rollout(sealed_snapshot, noisy, config)
        rows_after_rollout.append(rows); corrupted.append(_decode(noisy_final, targets) == index)
    deleted_indices = sorted(set(range(len(cues))) - set(evaluable_indices))
    deleted_abstentions = []
    for index in deleted_indices:
        _, deleted_final, rows = _probe_rollout(sealed_snapshot, cues[index], config)
        rows_after_rollout.append(rows)
        deleted_abstentions.append(_decode(deleted_final, targets) is None)
    unknown = _decode(torch.zeros(config.dim), targets) is None
    return {"condition": condition, "clean_accuracy": sum(predicted) / len(predicted),
            "corrupt_accuracy": sum(corrupted) / len(corrupted),
            "deleted_abstention": sum(deleted_abstentions) / max(1, len(deleted_abstentions)),
            "predicted_ids": predicted_ids,
            "mean_final_target_cosine": sum(final_target_cosines) / len(final_target_cosines),
            "mean_final_max_cosine": sum(final_max_cosines) / len(final_max_cosines),
            "unknown_abstention": float(unknown), "attractor_cosine_gain": sum(cosine_gain) / len(cosine_gain),
            "weight_drift": float((runtime.weight - before).norm()), "cutoff_audit": cutoff,
            "hippocampal_rows_after_rollout": max(rows_after_rollout, default=0),
            "snapshot_restore_parity": snapshot_restore_parity,
            "finite": bool(torch.isfinite(runtime.weight).all()),
            "replay_source_audit": replay_source_audit,
            "replay_source_reverse_audit": reversed_source,
            "replay_source_reverse_equal": source == reversed_source,
            "arrival_last_negative_control": arrival_last,
            "arrival_last_mismatch": source != arrival_last}


def loop8_native_replay(seed: int, config: NativeLoopsConfig) -> dict[str, Any]:
    native = _loop8_condition(seed, config, "native")
    no_replay = _loop8_condition(seed, config, "no_replay")
    shuffled = _loop8_condition(seed, config, "target_shuffled")
    advantage = native["clean_accuracy"] - max(no_replay["clean_accuracy"], shuffled["clean_accuracy"])
    native["controls"] = {"no_replay": no_replay, "target_shuffled": shuffled}
    native["control_advantage"] = advantage
    native["status"] = "GO" if (native["weight_drift"] > 0 and native["finite"] and native["clean_accuracy"] >= .8 and native["corrupt_accuracy"] >= .65 and native["deleted_abstention"] >= .95 and native["unknown_abstention"] >= .95 and advantage >= .2 and native["attractor_cosine_gain"] >= .05 and native["cutoff_audit"]["temporal_rows_after"] == 0 and native["cutoff_audit"]["hippocampal_rows_after"] == 0 and native["hippocampal_rows_after_rollout"] == 0 and native["snapshot_restore_parity"] and native["replay_source_reverse_equal"] and native["arrival_last_mismatch"]) else "STOP"
    return native


def _route_b_condition(seed: int, config: NativeLoopsConfig, condition: str) -> dict[str, Any]:
    """Separate fallback: bounded low-rank write computed from native states."""
    runtime, temporal = _runtime(seed, config), TemporalAuditedMemory(capacity=32)
    cues, targets = _codebook(seed, config.dim); before = runtime.weight.clone()
    write = torch.zeros_like(runtime.weight)
    for index, (cue, target) in enumerate(zip(cues, targets)):
        temporal.ingest(TemporalMemoryEvent(f"b{index}", "target", str(index), 1, index, f"route-b-{index}"))
        runtime.step(external_input=cue, force_mode=RuntimeMode.NREM, learning_signal=0.0)
        cue_state = _unit(runtime.activation.clone())
        paired = targets[(index + 1) % len(targets)] if condition == "target_shuffled" else target
        runtime.step(external_input=paired, force_mode=RuntimeMode.NREM, learning_signal=0.0)
        target_state = _unit(runtime.activation.clone())
        # row=post, column=pre; the auto term keeps the learned attractor alive.
        if condition != "no_write":
            write += torch.outer(target_state, cue_state) + 0.65 * torch.outer(target_state, target_state)
    # Four approximately unit-norm associations require a global bound above
    # two to avoid shrinking every mapping below the recurrent fixed-point
    # gain.  The bound remains finite and is audited in the result.
    desired_weight = config.bounded_write_gain * write
    installed_norm = 0.0 if condition == "no_write" else runtime.install_bounded_recurrent_delta(
        desired_weight - runtime.weight, max_frobenius_norm=5.0,
    )
    cutoff = _detach(runtime, temporal); sealed = runtime.snapshot()
    clean, corrupt, rows, gains = [], [], [], []
    for index, cue in enumerate(cues):
        cue_state, final, count = _probe_rollout(sealed, cue, config)
        clean.append(_decode(final, targets) == index); rows.append(count)
        gains.append(float(F.cosine_similarity(final, targets[index], dim=0) - F.cosine_similarity(cue_state, targets[index], dim=0)))
        noisy = cue.clone(); noisy[: max(1, int(config.dim * config.cue_corruption))] = 0.0
        _, final_noisy, count = _probe_rollout(sealed, noisy, config)
        corrupt.append(_decode(final_noisy, targets) == index); rows.append(count)
    return {"condition": condition, "clean_accuracy": sum(clean) / len(clean), "corrupt_accuracy": sum(corrupt) / len(corrupt),
            "unknown_abstention": float(_decode(torch.zeros(config.dim), targets) is None),
            "attractor_cosine_gain": sum(gains) / len(gains), "installed_write_norm": installed_norm,
            "weight_drift": float((runtime.weight - before).norm()), "cutoff_audit": cutoff,
            "hippocampal_rows_after_rollout": max(rows, default=0), "finite": bool(torch.isfinite(runtime.weight).all())}


def loop8_route_b(seed: int, config: NativeLoopsConfig) -> dict[str, Any]:
    """Route B fallback result.  It is intentionally not a Route A result."""
    native = _route_b_condition(seed, config, "bounded_write")
    no_write = _route_b_condition(seed, config, "no_write")
    shuffled = _route_b_condition(seed, config, "target_shuffled")
    advantage = native["clean_accuracy"] - max(no_write["clean_accuracy"], shuffled["clean_accuracy"])
    native["route"] = "B_bounded_supervised_recurrent_projection"
    native["controls"] = {"no_write": no_write, "target_shuffled": shuffled}
    native["control_advantage"] = advantage
    native["status"] = "GO" if (native["installed_write_norm"] > 0.0 and native["weight_drift"] > 0.0 and native["clean_accuracy"] >= .8 and native["corrupt_accuracy"] >= .65 and native["unknown_abstention"] >= .95 and native["attractor_cosine_gain"] >= .05 and advantage >= .2 and native["hippocampal_rows_after_rollout"] == 0) else "STOP"
    return native


def loop9_intervention(seed: int, config: NativeLoopsConfig, loop8: dict[str, Any]) -> dict[str, Any]:
    gen = torch.Generator(device="cpu").manual_seed(seed + 2003)
    width = config.dim // 4
    a = _orthonormal_rows(gen, 2, width)
    b = _orthonormal_rows(gen, 2, width)
    # Predeclared factorized codebooks: cue(A,B), target(A,B).  (1,1) never
    # appears in replay and is evaluated only after do(B=1) from (1,0).
    combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
    train_combinations, held_out = combinations[:-1], [(1, 1)]
    def encode_pair(ai: int, bi: int, *, target: bool) -> torch.Tensor:
        vector = torch.zeros(config.dim)
        offset = 2 * width if target else 0
        vector[offset : offset + width] = a[ai]
        vector[offset + width : offset + 2 * width] = b[bi]
        return _unit(vector)
    target_book = torch.stack([encode_pair(ai, bi, target=True) for ai, bi in combinations])
    def evaluate(*, shuffled: bool) -> tuple[float, dict[str, int]]:
        runtime, temporal = _runtime(seed, config), TemporalAuditedMemory(capacity=32)
        for index, (ai, bi) in enumerate(train_combinations):
            cue = encode_pair(ai, bi, target=False)
            temporal.ingest(TemporalMemoryEvent(f"i{index}", "target", f"{ai}:{bi}", 1, index, f"intervention-{index}"))
            target = encode_pair(*train_combinations[(index + 1) % len(train_combinations)], target=True) if shuffled else encode_pair(ai, bi, target=True)
            runtime.step(external_input=cue, force_mode=RuntimeMode.WAKE, learning_signal=0.0)
            for _ in range(config.replay_epochs):
                runtime.step(external_input=cue, force_mode=RuntimeMode.NREM, learning_signal=1.0)
                runtime.step(external_input=target, force_mode=RuntimeMode.NREM, learning_signal=1.0)
        cutoff = _detach(runtime, temporal); sealed_snapshot = runtime.snapshot(); outcomes = []
        for ai, bi in held_out:
            # do(B=bi) applied to the observed (A=ai, B=0) coordinate block.
            intervened = encode_pair(ai, bi, target=False)
            _, final, rows = _probe_rollout(sealed_snapshot, intervened, config)
            cutoff["hippocampal_rows_after_rollout"] = rows
            outcomes.append(_decode(final, target_book) == combinations.index((ai, bi)))
        return sum(outcomes) / len(outcomes), cutoff
    accuracy, cutoff = evaluate(shuffled=False)
    shuffled_accuracy, shuffled_cutoff = evaluate(shuffled=True)
    return {"train_combinations": train_combinations, "held_out_combinations": held_out, "intervention": "do(B=1)", "native_accuracy": accuracy, "target_shuffled_accuracy": shuffled_accuracy,
            "advantage": accuracy - shuffled_accuracy, "frozen_codebook": True,
            "cutoff_audit": cutoff,
            "target_shuffled_cutoff_audit": shuffled_cutoff,
            "status": "GO" if accuracy >= .70 and accuracy - shuffled_accuracy >= .20 else "STOP"}


def loop9_route_b(seed: int, config: NativeLoopsConfig) -> dict[str, Any]:
    """Route-B factorized held-out intervention, isolated from Route A."""
    gen = torch.Generator(device="cpu").manual_seed(seed + 2003); width = config.dim // 4
    a = _orthonormal_rows(gen, 2, width)
    b = _orthonormal_rows(gen, 2, width)
    combinations, train, held_out = [(0, 0), (0, 1), (1, 0), (1, 1)], [(0, 0), (0, 1), (1, 0)], [(1, 1)]
    def code(ai: int, bi: int, target: bool) -> torch.Tensor:
        v = torch.zeros(config.dim); offset = 2 * width if target else 0
        v[offset:offset + width] = a[ai]; v[offset + width:offset + 2 * width] = b[bi]
        return _unit(v)
    book = torch.stack([code(ai, bi, True) for ai, bi in combinations])
    def evaluate(condition: str) -> tuple[float, float, dict[str, int]]:
        runtime, temporal, write = _runtime(seed, config), TemporalAuditedMemory(), None
        runtime.config.hippocampal_encoding_enabled = False
        write = torch.zeros_like(runtime.weight)
        a_counts = {value: sum(ai == value for ai, _ in train) for value in range(2)}
        b_counts = {value: sum(bi == value for _, bi in train) for value in range(2)}
        def factor_state(value: torch.Tensor) -> torch.Tensor:
            runtime.reset_evaluation_state()
            runtime.step(
                external_input=config.cue_drive_gain * value,
                force_mode=RuntimeMode.WAKE,
                learning_signal=0.0,
            )
            return _unit(runtime.activation.clone())
        for index, (ai, bi) in enumerate(train):
            target_ai, target_bi = ai, bi
            if condition == "target_shuffled":
                target_ai, target_bi = train[(index + 1) % len(train)]
            temporal.ingest(TemporalMemoryEvent(f"b9{index}", "target", f"{ai}:{bi}", 1, index, f"route-b9-{index}"))
            cue_a = torch.zeros(config.dim); cue_a[:width] = a[ai]
            cue_b = torch.zeros(config.dim); cue_b[width:2 * width] = b[bi]
            target_a = torch.zeros(config.dim); target_a[2 * width:3 * width] = a[target_ai]
            target_b = torch.zeros(config.dim); target_b[3 * width:4 * width] = b[target_bi]
            pre_a, pre_b = factor_state(cue_a), factor_state(cue_b)
            post_a, post_b = factor_state(target_a), factor_state(target_b)
            if condition != "no_write":
                # Each factor value contributes unit total mass regardless of
                # how often it occurs in the three observed combinations.
                wa, wb = 1.0 / a_counts[ai], 1.0 / b_counts[bi]
                write += wa * (torch.outer(post_a, pre_a) + .65 * torch.outer(post_a, post_a))
                write += wb * (torch.outer(post_b, pre_b) + .65 * torch.outer(post_b, post_b))
        desired_weight = config.bounded_write_gain * write
        write_norm = 0.0 if condition == "no_write" else runtime.install_bounded_recurrent_delta(
            desired_weight - runtime.weight, max_frobenius_norm=5.0,
        )
        cutoff = _detach(runtime, temporal); sealed = runtime.snapshot(); wins = []
        for ai, bi in held_out:
            _, final, rows = _probe_rollout(sealed, code(ai, bi, False), config)
            cutoff["hippocampal_rows_after_rollout"] = rows
            wins.append(_decode(final, book) == combinations.index((ai, bi)))
        return sum(wins) / len(wins), write_norm, cutoff
    accuracy, norm, cutoff = evaluate("bounded_write")
    no_write, _, no_write_cutoff = evaluate("no_write")
    shuffled, _, shuffled_cutoff = evaluate("target_shuffled")
    advantage = accuracy - max(no_write, shuffled)
    return {"route": "B_bounded_supervised_recurrent_projection", "train_combinations": train, "held_out_combinations": held_out,
            "intervention": "do(B=1)", "native_accuracy": accuracy, "no_write_accuracy": no_write,
            "target_shuffled_accuracy": shuffled, "control_advantage": advantage, "installed_write_norm": norm,
            "cutoff_audit": cutoff, "no_write_cutoff_audit": no_write_cutoff, "target_shuffled_cutoff_audit": shuffled_cutoff,
            "status": "GO" if accuracy >= .70 and advantage >= .20 and norm > 0 and cutoff["hippocampal_rows_after_rollout"] == 0 else "STOP"}


def loop10_self_prediction(seed: int, config: NativeLoopsConfig) -> dict[str, Any]:
    runtime = _runtime(seed, config)
    action_count = 4
    action_drives = torch.zeros(action_count, config.dim)
    for action in range(action_count):
        action_drives[action, action * (config.dim // action_count) : (action + 1) * (config.dim // action_count)] = 1.0
    def summary(rt: BrainRuntime) -> torch.Tensor:
        state = rt.activation
        return torch.tensor([float(state.mean()), float(state.norm()), float(state.abs().max()), float((state.abs() > .1).float().mean())])
    def feature(rt: BrainRuntime, action: int) -> torch.Tensor:
        one_hot = F.one_hot(torch.tensor(action), num_classes=action_count).float()
        # Pre-transition native observables only.  Keeping the vector-valued
        # state avoids throwing away the dynamics before the frozen readout.
        return torch.cat((rt.activation, rt.refractory, rt.memory_trace,
                          rt.adaptation, rt.stp_u, rt.stp_x,
                          rt.lifecycle.float(), one_hot, torch.ones(1)))
    # Fit only pre-confirmation current-observable/action -> next-summary rows.
    features: list[torch.Tensor] = []; nexts: list[torch.Tensor] = []
    for action in ([0, 1, 2, 3] * 12):
        features.append(feature(runtime, action))
        runtime.step(external_input=action_drives[action], force_mode=RuntimeMode.WAKE, learning_signal=0.0)
        nexts.append(summary(runtime))
    x, y = torch.stack(features), torch.stack(nexts)
    ridge = 1e-4
    model = torch.linalg.solve(x.T @ x + ridge * torch.eye(x.shape[1]), x.T @ y)  # frozen before confirmation.
    sealed_snapshot = runtime.snapshot()
    model_mse = persistence_mse = 0.0; corrections = []; errors = []
    # A distinct, reversed confirmation trajectory is never used for fitting.
    for action in [3, 2, 1, 0] * 3:
        for intervened in (False, True):
            trial = BrainRuntime.from_snapshot(sealed_snapshot, backend="torch", device="cpu")
            prediction = feature(trial, action) @ model
            persistence = summary(trial)
            drive = action_drives[action] * (3.0 if intervened else 1.0)  # OOD high-magnitude do(action-drive).
            trial.step(external_input=drive, force_mode=RuntimeMode.WAKE, learning_signal=0.0)
            actual = summary(trial)
            error = float(torch.mean((prediction - actual) ** 2)); errors.append(error)
            model_mse += error; persistence_mse += float(torch.mean((persistence - actual) ** 2))
            corrections.append((intervened, min(1.0, error)))
    model_mse /= len(errors); persistence_mse /= len(errors)
    intervened_depth = sum(v for flag, v in corrections if flag) / 8; plain_depth = sum(v for flag, v in corrections if not flag) / 8
    return {"next_state_mse": model_mse, "persistence_mse": persistence_mse, "improvement": 1.0 - model_mse / max(persistence_mse, 1e-12),
            "error_finite": all(torch.isfinite(torch.tensor(errors)).tolist()), "intervened_correction_depth": intervened_depth,
            "plain_correction_depth": plain_depth, "leakage_audit": {"post_state_used_as_feature": False, "model_frozen_before_confirmation": True, "deterministic_ridge": ridge},
            "status": "GO" if model_mse <= .9 * persistence_mse and intervened_depth > plain_depth else "STOP"}


def run_native_loops(seed: int = 97101, *, config: NativeLoopsConfig | None = None) -> dict[str, Any]:
    config = config or NativeLoopsConfig(seed=seed)
    config = NativeLoopsConfig(**{**asdict(config), "seed": seed})
    six = loop6_temporal_selection(); seven = loop7_agent_route(seed, config); eight = loop8_native_replay(seed, config)
    return {"seed": seed, "config": asdict(config), "loop6": six, "loop7": seven, "loop8": eight,
            "loop9": loop9_intervention(seed, config, eight), "loop10": loop10_self_prediction(seed, config)}


def run_seed_range(seeds: Iterable[int]) -> list[dict[str, Any]]:
    return [run_native_loops(int(seed)) for seed in seeds]


def run_route_b_seed_range(seeds: Iterable[int]) -> list[dict[str, Any]]:
    """Route B fallback results; Route A is not recomputed."""
    return [{"seed": int(seed), "route": "B_bounded_supervised_recurrent_projection",
             "loop8_route_b": loop8_route_b(int(seed), NativeLoopsConfig(seed=int(seed))),
             "loop9_route_b": loop9_route_b(int(seed), NativeLoopsConfig(seed=int(seed)))} for seed in seeds]
