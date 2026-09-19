"""BA-TR23: experience-learned context-to-packet relevance gating."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ..runtime import BrainRuntime, RuntimeMode
from .runtime_all_input_packet_factorization import _all_input_snapshot
from .runtime_binding_composition_no_go import COMPOSITION_PAIRS, TARGET_MAPPING
from .runtime_context_branch_routing import architectural_blocks
from .runtime_experience_attenuation_binding import (
    _experience_block_compensated,
    generate_fresh_inputs,
)
from .runtime_experience_delayed_binding import (
    MIN_DECODE_ACTIVATION,
    PAIR_TICKS,
    PRESYNAPTIC_EVENT_THRESHOLD,
    _blocks,
    _external,
    _seal,
)
from .runtime_three_event_relevance_no_go import (
    DISTRACTOR_INDEX,
    _matched_distractor_snapshot,
)


CALIBRATION_SEEDS = (110001,)
DEVELOPMENT_SEEDS = tuple(range(110101, 110117))
GATE_THRESHOLD = 0.5


@dataclass(frozen=True)
class ContextPacketGateSnapshot:
    association: torch.Tensor
    context_codes: torch.Tensor
    update_count: int


def _gate_hash(gate: ContextPacketGateSnapshot) -> str:
    digest = hashlib.sha256()
    digest.update(gate.association.detach().cpu().numpy().tobytes())
    digest.update(gate.context_codes.detach().cpu().numpy().tobytes())
    digest.update(str(int(gate.update_count)).encode("ascii"))
    return digest.hexdigest()


def train_context_packet_gate(seed: int) -> ContextPacketGateSnapshot:
    """Learn context/event co-occurrence with no target-side signal."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 230_009)
    q, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator))
    context_codes = q.T.contiguous()
    association = torch.zeros(20, 4)
    source, _hidden, _target = _blocks()
    for context_index, pair in enumerate(COMPOSITION_PAIRS):
        event = torch.zeros(20)
        event[source[pair[0]]] = 1.0
        event[source[pair[1]]] = 1.0
        association += torch.outer(event, context_codes[context_index])
    return ContextPacketGateSnapshot(
        association=association,
        context_codes=context_codes,
        update_count=len(COMPOSITION_PAIRS),
    )


def compile_context_packet_indices(
    gate: ContextPacketGateSnapshot,
    context_code: torch.Tensor,
) -> tuple[int, ...]:
    association = torch.as_tensor(gate.association, dtype=torch.float32)
    cue = torch.as_tensor(context_code, dtype=torch.float32).view(-1)
    if association.shape != (20, 4) or cue.shape != (4,):
        raise ValueError("context packet gate has the wrong shape")
    if not torch.isfinite(association).all() or not torch.isfinite(cue).all():
        raise ValueError("context packet gate must be finite")
    scores = association @ cue
    selected = tuple(
        int(index)
        for index in torch.nonzero(scores > GATE_THRESHOLD, as_tuple=False).view(-1)
    )
    if len(selected) != 2:
        raise RuntimeError("context packet gate did not select exactly two inputs")
    return selected


def _target_set(values: torch.Tensor) -> tuple[int, ...]:
    packed = torch.as_tensor(values, dtype=torch.float64).view(4)
    return tuple(
        int(index)
        for index in torch.nonzero(
            packed >= MIN_DECODE_ACTIVATION, as_tuple=False
        ).view(-1)
    )


def _gated_three_event_probe(
    snapshot: Any,
    left: int,
    right: int,
    selected_indices: tuple[int, ...] | None,
) -> dict[str, Any]:
    source, hidden, target = _blocks()
    missing = tuple(slot for slot in range(4) if slot not in (left, right))
    distractor_source_slot = int(missing[0])
    routed = _matched_distractor_snapshot(snapshot, distractor_source_slot)
    if selected_indices is not None:
        config = replace(
            routed.config,
            competition_input_indices=tuple(int(value) for value in selected_indices),
            competition_k_from_delayed_input=False,
            competition_factorize_delayed_input=True,
        )
        routed = replace(routed, config=config)
    else:
        routed = _all_input_snapshot(routed)
    runtime = BrainRuntime.from_snapshot(routed, backend="torch", device="cpu")
    event_indices = torch.tensor(
        tuple(source) + tuple(int(v) for v in architectural_blocks(20)[1])
    )
    hidden_idx = torch.tensor(hidden)
    target_idx = torch.tensor(target)
    packet_counts: list[int] = []
    hidden_first = torch.zeros(4)
    target_final = torch.zeros(4)
    for tick in range(PAIR_TICKS):
        if runtime._delay_buffer is None:
            raise RuntimeError("context relevance test requires the delay ring")
        ring_slot = runtime._delay_idx % runtime.config.max_axon_delay
        packet_counts.append(int(torch.count_nonzero(
            runtime._delay_buffer[ring_slot, event_indices].abs()
            > runtime.config.competition_epsilon
        ).item()))
        external = torch.zeros(20)
        if tick == 0:
            external = (
                _external(source[left])
                + _external(source[right])
                + _external(DISTRACTOR_INDEX)
            )
        runtime.step(
            external_input=external,
            force_mode=RuntimeMode.WAKE,
            learning_signal=0.0,
        )
        if tick != 1:
            runtime._delay_buffer[ring_slot, event_indices] = 0.0
        if tick == 3:
            hidden_first = runtime.activation[hidden_idx].detach().clone()
        if tick == 6:
            target_final = runtime.activation[target_idx].detach().clone()
    decoded = _target_set(target_final)
    desired = tuple(sorted((TARGET_MAPPING[left], TARGET_MAPPING[right])))
    return {
        "source_pair": [left, right],
        "selected_input_indices": None if selected_indices is None else list(selected_indices),
        "decoded_target_set": list(decoded),
        "desired_target_set": list(desired),
        "success": decoded == desired,
        "hidden_positive_count": int(torch.count_nonzero(
            hidden_first > PRESYNAPTIC_EVENT_THRESHOLD
        ).item()),
        "target_at_6": [float(value) for value in target_final.tolist()],
        "input_packet_count_by_tick": packet_counts,
        "hippocampal_rows_after": len(runtime.hippocampus),
    }


def analyze_context_gate_row(seed: int, B: torch.Tensor) -> dict[str, Any]:
    block = _experience_block_compensated(B, condition="target_shuffle")
    base_snapshot, cutoff = _seal(block["runtime"])
    gate = train_context_packet_gate(seed)
    gate_before = _gate_hash(gate)
    source, _hidden, _target = _blocks()
    learned: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    shuffled: list[dict[str, Any]] = []
    static: list[dict[str, Any]] = []
    no_context: list[dict[str, Any]] = []
    selected_receipts: list[list[int]] = []
    static_indices = compile_context_packet_indices(gate, gate.context_codes[0])
    for context_index, pair in enumerate(COMPOSITION_PAIRS):
        selected = compile_context_packet_indices(
            gate, gate.context_codes[context_index]
        )
        selected_receipts.append(list(selected))
        oracle_indices = (source[pair[0]], source[pair[1]])
        learned.append(_gated_three_event_probe(base_snapshot, *pair, selected))
        oracle.append(_gated_three_event_probe(base_snapshot, *pair, oracle_indices))
        shuffled.append(_gated_three_event_probe(
            base_snapshot,
            *pair,
            compile_context_packet_indices(
                gate, gate.context_codes[(context_index + 1) % len(COMPOSITION_PAIRS)]
            ),
        ))
        static.append(_gated_three_event_probe(base_snapshot, *pair, static_indices))
        no_context.append(_gated_three_event_probe(base_snapshot, *pair, None))
    gate_after = _gate_hash(gate)
    exact_compiler = all(
        tuple(selected_receipts[index])
        == (source[pair[0]], source[pair[1]])
        for index, pair in enumerate(COMPOSITION_PAIRS)
    )
    gates = {
        "context_codes_orthonormal": bool(torch.allclose(
            gate.context_codes @ gate.context_codes.T,
            torch.eye(4),
            atol=1e-6,
            rtol=0.0,
        )),
        "local_cooccurrence_compiler_exact": exact_compiler,
        "learned_gate_recovers_pair": all(
            row["success"] and row["hidden_positive_count"] == 2
            for row in learned
        ),
        "oracle_bit_exact": all(
            learned[index]["target_at_6"] == oracle[index]["target_at_6"]
            for index in range(len(COMPOSITION_PAIRS))
        ),
        "context_shuffle_fails": not any(row["success"] for row in shuffled),
        "static_gate_not_general": sum(row["success"] for row in static) == 1,
        "no_context_retains_relevance_no_go": not any(
            row["success"] for row in no_context
        ),
        "one_shot_three_event_receipt": all(
            row["input_packet_count_by_tick"] == [0, 0, 0, 3, 0, 0, 0]
            for row in learned
        ),
        "gate_frozen_during_probe": gate_before == gate_after,
        "stores_zero": bool(
            cutoff["temporal_rows_after"] == 0
            and cutoff["hippocampal_rows_after"] == 0
            and all(row["hippocampal_rows_after"] == 0 for row in learned)
        ),
    }
    return {
        "seed": int(seed),
        "status": "CONTEXT_PACKET_RELEVANCE_GATE_PASS" if all(gates.values()) else "CONTEXT_PACKET_RELEVANCE_GATE_FAIL",
        "gates": gates,
        "gate_hash": gate_before,
        "gate_update_count": int(gate.update_count),
        "gate_training_inputs": ["context_code", "cooccurring_event_coordinates"],
        "selected_indices_by_context": selected_receipts,
        "learned_success_count": sum(row["success"] for row in learned),
        "oracle_success_count": sum(row["success"] for row in oracle),
        "context_shuffle_success_count": sum(row["success"] for row in shuffled),
        "static_success_count": sum(row["success"] for row in static),
        "no_context_success_count": sum(row["success"] for row in no_context),
        "learned": learned,
        "endpoint_opened": False,
        "claim_scope": "synthetic context/event-cooccurrence packet relevance lookup",
    }


def analyze_context_gate_artifact(path: str | Path, *, stage: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "FRESH_INPUTS_READY":
        raise RuntimeError("fresh source-code inputs did not pass producer gates")
    rows = [
        analyze_context_gate_row(int(row["seed"]), torch.tensor(row["candidate_weights"]))
        for row in payload["rows"]
    ]
    expected_count = 1 if stage == "calibration" else len(DEVELOPMENT_SEEDS)
    passed = len(rows) == expected_count and all(
        row["status"] == "CONTEXT_PACKET_RELEVANCE_GATE_PASS" for row in rows
    )
    return {
        "status": (
            "CONTEXT_PACKET_GATE_CALIBRATION_PASS"
            if passed and stage == "calibration"
            else "CONTEXT_PACKET_GATE_DEVELOPMENT_GO"
            if passed and stage == "development"
            else "CONTEXT_PACKET_GATE_STOP"
        ),
        "stage": stage,
        "seed_count": len(rows),
        "pass_count": sum(row["status"] == "CONTEXT_PACKET_RELEVANCE_GATE_PASS" for row in rows),
        "learned_success_total": sum(row["learned_success_count"] for row in rows),
        "oracle_success_total": sum(row["oracle_success_count"] for row in rows),
        "context_shuffle_success_total": sum(row["context_shuffle_success_count"] for row in rows),
        "static_success_total": sum(row["static_success_count"] for row in rows),
        "no_context_success_total": sum(row["no_context_success_count"] for row in rows),
        "endpoint_opened": False,
        "confirmation_opened": False,
        "rows": rows,
    }


__all__ = [
    "CALIBRATION_SEEDS",
    "DEVELOPMENT_SEEDS",
    "ContextPacketGateSnapshot",
    "generate_fresh_inputs",
    "train_context_packet_gate",
    "compile_context_packet_indices",
    "analyze_context_gate_artifact",
]

