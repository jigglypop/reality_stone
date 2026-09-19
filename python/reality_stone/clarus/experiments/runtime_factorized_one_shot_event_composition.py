"""BA-TR20: one-shot source-event falsifier for factorized composition."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch

from ..runtime import BrainRuntime, RuntimeMode
from .runtime_binding_composition_no_go import COMPOSITION_PAIRS, TARGET_MAPPING
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
from .runtime_factorized_competition_composition import _factorized_snapshot


CALIBRATION_SEEDS = (107003,)
DEVELOPMENT_SEEDS = tuple(range(107101, 107117))


def _target_set(values: torch.Tensor) -> tuple[int, ...]:
    packed = torch.as_tensor(values, dtype=torch.float64).view(4)
    return tuple(
        int(index)
        for index in torch.nonzero(
            packed >= MIN_DECODE_ACTIVATION, as_tuple=False
        ).view(-1)
    )


def _event_probe(
    snapshot: Any,
    source_slots: Iterable[int],
    *,
    emission: str,
) -> dict[str, Any]:
    """Run one probe while gating only source writes into the delay ring.

    ``one_shot`` retains the source-coordinate packet written at tick 1 and
    zeros source coordinates in every other newly written ring slot.
    ``suppressed`` zeros those coordinates in every written slot. ``stream``
    is the unmodified persistent-activation control.
    """
    if emission not in {"one_shot", "suppressed", "stream"}:
        raise ValueError(f"unknown source emission mode: {emission}")
    slots = tuple(int(slot) for slot in source_slots)
    if not slots or len(set(slots)) != len(slots):
        raise ValueError("source slots must be a nonempty unique sequence")
    source, hidden, target = _blocks()
    if any(slot < 0 or slot >= len(source) for slot in slots):
        raise ValueError("source slot is out of range")
    source_idx = torch.tensor(source)
    hidden_idx = torch.tensor(hidden)
    target_idx = torch.tensor(target)
    runtime = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    packet_counts: list[int] = []
    written_packet_counts: list[int] = []
    source_max_after_step: list[float] = []
    hidden_first = torch.zeros(4)
    target_final = torch.zeros(4)
    for tick in range(PAIR_TICKS):
        if runtime._delay_buffer is None:
            raise RuntimeError("one-shot composition requires the delay ring")
        slot = runtime._delay_idx % runtime.config.max_axon_delay
        delivered = runtime._delay_buffer[slot]
        packet_counts.append(int(torch.count_nonzero(
            delivered[source_idx].abs() > runtime.config.competition_epsilon
        ).item()))
        external = torch.zeros(20)
        if tick == 0:
            for source_slot in slots:
                external = external + _external(source[source_slot])
        runtime.step(
            external_input=external,
            force_mode=RuntimeMode.WAKE,
            learning_signal=0.0,
        )
        if emission == "suppressed" or (emission == "one_shot" and tick != 1):
            runtime._delay_buffer[slot, source_idx] = 0.0
        written_packet_counts.append(int(torch.count_nonzero(
            runtime._delay_buffer[slot, source_idx].abs()
            > runtime.config.competition_epsilon
        ).item()))
        source_max_after_step.append(float(runtime.activation[source_idx].abs().max().item()))
        if tick == 3:
            hidden_first = runtime.activation[hidden_idx].detach().clone()
        if tick == 6:
            target_final = runtime.activation[target_idx].detach().clone()
    return {
        "source_slots": list(slots),
        "emission": emission,
        "decoded_target_set": list(_target_set(target_final)),
        "hidden_first_arrival": [float(value) for value in hidden_first.tolist()],
        "hidden_positive_count": int(torch.count_nonzero(
            hidden_first > PRESYNAPTIC_EVENT_THRESHOLD
        ).item()),
        "target_at_6": [float(value) for value in target_final.tolist()],
        "source_packet_count_by_tick": packet_counts,
        "source_written_packet_count_by_tick": written_packet_counts,
        "source_max_after_step": source_max_after_step,
        "external_nonzero_ticks": [0],
        "hippocampal_rows_after": len(runtime.hippocampus),
    }


def _expected(slots: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(TARGET_MAPPING[int(slot)] for slot in slots))


def _success(row: dict[str, Any]) -> bool:
    return tuple(row["decoded_target_set"]) == _expected(row["source_slots"])


def _independent_union(snapshot: Any, left: int, right: int) -> dict[str, Any]:
    left_probe = _event_probe(snapshot, (left,), emission="one_shot")
    right_probe = _event_probe(snapshot, (right,), emission="one_shot")
    union = torch.maximum(
        torch.tensor(left_probe["target_at_6"]),
        torch.tensor(right_probe["target_at_6"]),
    )
    decoded = _target_set(union)
    expected = _expected((left, right))
    return {
        "source_slots": [left, right],
        "expected_target_set": list(expected),
        "decoded_target_set": list(decoded),
        "success": decoded == expected,
        "union_target": [float(value) for value in union.tolist()],
    }


def analyze_one_shot_row(seed: int, B: torch.Tensor) -> dict[str, Any]:
    block = _experience_block_compensated(B, condition="target_shuffle")
    base_snapshot, cutoff = _seal(block["runtime"])
    factorized_snapshot = _factorized_snapshot(base_snapshot, aligned=True)
    misaligned_snapshot = _factorized_snapshot(base_snapshot, aligned=False)

    base_atomic = [
        _event_probe(base_snapshot, (slot,), emission="one_shot")
        for slot in range(4)
    ]
    factorized_atomic = [
        _event_probe(factorized_snapshot, (slot,), emission="one_shot")
        for slot in range(4)
    ]
    singleton_parity = all(
        base_atomic[index]["hidden_first_arrival"]
        == factorized_atomic[index]["hidden_first_arrival"]
        and base_atomic[index]["target_at_6"] == factorized_atomic[index]["target_at_6"]
        for index in range(4)
    )
    factorized_pairs = [
        _event_probe(factorized_snapshot, pair, emission="one_shot")
        for pair in COMPOSITION_PAIRS
    ]
    legacy_pairs = [
        _event_probe(base_snapshot, pair, emission="one_shot")
        for pair in COMPOSITION_PAIRS
    ]
    misaligned_pairs = [
        _event_probe(misaligned_snapshot, pair, emission="one_shot")
        for pair in COMPOSITION_PAIRS
    ]
    suppressed_pairs = [
        _event_probe(factorized_snapshot, pair, emission="suppressed")
        for pair in COMPOSITION_PAIRS
    ]
    stream_pairs = [
        _event_probe(factorized_snapshot, pair, emission="stream")
        for pair in COMPOSITION_PAIRS
    ]
    independent = [
        _independent_union(factorized_snapshot, *pair)
        for pair in COMPOSITION_PAIRS
    ]

    gates = {
        "singleton_exact_parity_under_latch": singleton_parity,
        "atomic_one_shot_memory_intact": all(_success(row) for row in factorized_atomic),
        "factorized_one_shot_pair_composition": all(
            _success(row) and row["hidden_positive_count"] == 2
            for row in factorized_pairs
        ),
        "legacy_one_shot_global_wta_fails": not any(_success(row) for row in legacy_pairs),
        "misaligned_one_shot_receipt_fails": not any(
            _success(row) for row in misaligned_pairs
        ),
        "independent_one_shot_union_recovers": all(
            row["success"] for row in independent
        ),
        "exact_one_shot_pair_receipt": all(
            row["source_packet_count_by_tick"] == [0, 0, 0, 2, 0, 0, 0]
            and row["source_written_packet_count_by_tick"] == [0, 2, 0, 0, 0, 0, 0]
            for row in factorized_pairs
        ),
        "exact_one_shot_atomic_receipt": all(
            row["source_packet_count_by_tick"] == [0, 0, 0, 1, 0, 0, 0]
            and row["source_written_packet_count_by_tick"] == [0, 1, 0, 0, 0, 0, 0]
            for row in factorized_atomic
        ),
        "suppressed_event_has_no_route": all(
            row["source_packet_count_by_tick"] == [0] * PAIR_TICKS
            and row["source_written_packet_count_by_tick"] == [0] * PAIR_TICKS
            and row["decoded_target_set"] == []
            for row in suppressed_pairs
        ),
        "persistent_stream_control_retained": all(
            row["source_packet_count_by_tick"] == [0, 0, 0, 2, 2, 2, 2]
            and _success(row)
            for row in stream_pairs
        ),
        "stores_zero": bool(
            cutoff["temporal_rows_after"] == 0
            and cutoff["hippocampal_rows_after"] == 0
            and all(row["hippocampal_rows_after"] == 0 for row in factorized_pairs)
        ),
    }
    return {
        "seed": int(seed),
        "status": "FACTORIZED_ONE_SHOT_EVENT_PASS" if all(gates.values()) else "FACTORIZED_ONE_SHOT_EVENT_FAIL",
        "gates": gates,
        "atomic_success_count": sum(_success(row) for row in factorized_atomic),
        "factorized_pair_success_count": sum(_success(row) for row in factorized_pairs),
        "legacy_pair_success_count": sum(_success(row) for row in legacy_pairs),
        "misaligned_pair_success_count": sum(_success(row) for row in misaligned_pairs),
        "independent_union_success_count": sum(row["success"] for row in independent),
        "suppressed_pair_success_count": sum(_success(row) for row in suppressed_pairs),
        "stream_pair_success_count": sum(_success(row) for row in stream_pairs),
        "factorized_atomic": factorized_atomic,
        "factorized_pairs": factorized_pairs,
        "legacy_pairs": legacy_pairs,
        "misaligned_pairs": misaligned_pairs,
        "suppressed_pairs": suppressed_pairs,
        "stream_pairs": stream_pairs,
        "independent_union": independent,
        "endpoint_opened": False,
        "claim_scope": "synthetic factorized composition under an externally clamped one-shot source event",
    }


def analyze_one_shot_artifact(path: str | Path, *, stage: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "FRESH_INPUTS_READY":
        raise RuntimeError("fresh source-code inputs did not pass producer gates")
    rows = [
        analyze_one_shot_row(int(row["seed"]), torch.tensor(row["candidate_weights"]))
        for row in payload["rows"]
    ]
    expected_count = 1 if stage == "calibration" else len(DEVELOPMENT_SEEDS)
    passed = len(rows) == expected_count and all(
        row["status"] == "FACTORIZED_ONE_SHOT_EVENT_PASS" for row in rows
    )
    return {
        "status": (
            "FACTORIZED_ONE_SHOT_CALIBRATION_PASS"
            if passed and stage == "calibration"
            else "FACTORIZED_ONE_SHOT_DEVELOPMENT_GO"
            if passed and stage == "development"
            else "FACTORIZED_ONE_SHOT_STOP"
        ),
        "stage": stage,
        "seed_count": len(rows),
        "pass_count": sum(row["status"] == "FACTORIZED_ONE_SHOT_EVENT_PASS" for row in rows),
        "atomic_success_total": sum(row["atomic_success_count"] for row in rows),
        "factorized_pair_success_total": sum(row["factorized_pair_success_count"] for row in rows),
        "legacy_pair_success_total": sum(row["legacy_pair_success_count"] for row in rows),
        "misaligned_pair_success_total": sum(row["misaligned_pair_success_count"] for row in rows),
        "independent_union_success_total": sum(row["independent_union_success_count"] for row in rows),
        "suppressed_pair_success_total": sum(row["suppressed_pair_success_count"] for row in rows),
        "stream_pair_success_total": sum(row["stream_pair_success_count"] for row in rows),
        "endpoint_opened": False,
        "confirmation_opened": False,
        "rows": rows,
    }


__all__ = [
    "CALIBRATION_SEEDS",
    "DEVELOPMENT_SEEDS",
    "generate_fresh_inputs",
    "_event_probe",
    "analyze_one_shot_artifact",
]
