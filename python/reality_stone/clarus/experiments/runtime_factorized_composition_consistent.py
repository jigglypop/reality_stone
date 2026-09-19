"""BA-TR19: decoder-consistent source-factorized composition."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..runtime import BrainRuntime, RuntimeMode
from .runtime_adaptive_competition_composition import _adaptive_snapshot
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
    _probe,
    _seal,
)
from .runtime_factorized_competition_composition import _factorized_snapshot


CALIBRATION_SEEDS = (106001,)
DEVELOPMENT_SEEDS = tuple(range(106101, 106117))


def _target_set(values: torch.Tensor) -> tuple[int, ...]:
    packed = torch.as_tensor(values, dtype=torch.float64).view(4)
    return tuple(
        int(index)
        for index in torch.nonzero(
            packed >= MIN_DECODE_ACTIVATION, as_tuple=False
        ).view(-1)
    )


def _pair_probe(snapshot: Any, left: int, right: int) -> dict[str, Any]:
    source, hidden, target = _blocks()
    source_idx = torch.tensor(source)
    hidden_idx = torch.tensor(hidden)
    target_idx = torch.tensor(target)
    runtime = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    packet_counts = []
    hidden_first = torch.zeros(4)
    target_final = torch.zeros(4)
    for tick in range(PAIR_TICKS):
        if runtime._delay_buffer is None:
            raise RuntimeError("composition requires the delayed packet ring")
        slot = runtime._delay_idx % runtime.config.max_axon_delay
        delivered = runtime._delay_buffer[slot]
        packet_counts.append(int(torch.count_nonzero(
            delivered[source_idx].abs() > runtime.config.competition_epsilon
        ).item()))
        external = (
            _external(source[left]) + _external(source[right])
            if tick == 0
            else torch.zeros(20)
        )
        runtime.step(
            external_input=external,
            force_mode=RuntimeMode.WAKE,
            learning_signal=0.0,
        )
        if tick == 3:
            hidden_first = runtime.activation[hidden_idx].detach().clone()
        if tick == 6:
            target_final = runtime.activation[target_idx].detach().clone()
    expected = tuple(sorted((TARGET_MAPPING[left], TARGET_MAPPING[right])))
    decoded = _target_set(target_final)
    return {
        "source_pair": [left, right],
        "expected_target_set": list(expected),
        "decoded_target_set": list(decoded),
        "success": decoded == expected,
        "hidden_first_arrival": [float(value) for value in hidden_first.tolist()],
        "hidden_positive_count": int(torch.count_nonzero(
            hidden_first > PRESYNAPTIC_EVENT_THRESHOLD
        ).item()),
        "target_at_6": [float(value) for value in target_final.tolist()],
        "source_packet_count_by_tick": packet_counts,
        "hippocampal_rows_after": len(runtime.hippocampus),
    }


def _consistent_union(snapshot: Any, left: int, right: int) -> dict[str, Any]:
    left_probe = _probe(snapshot, left)
    right_probe = _probe(snapshot, right)
    union = torch.maximum(
        torch.tensor(left_probe["target_at_6"]),
        torch.tensor(right_probe["target_at_6"]),
    )
    expected = tuple(sorted((TARGET_MAPPING[left], TARGET_MAPPING[right])))
    decoded = _target_set(union)
    return {
        "source_pair": [left, right],
        "expected_target_set": list(expected),
        "decoded_target_set": list(decoded),
        "success": decoded == expected,
        "union_target": [float(value) for value in union.tolist()],
    }


def _atomic(snapshot: Any) -> list[dict[str, Any]]:
    return [_probe(snapshot, slot) for slot in range(4)]


def _atomic_success(rows: list[dict[str, Any]]) -> int:
    return sum(
        row["decoded_target"] == TARGET_MAPPING[row["source_slot"]]
        for row in rows
    )


def analyze_consistent_row(seed: int, B: torch.Tensor) -> dict[str, Any]:
    block = _experience_block_compensated(B, condition="target_shuffle")
    base_snapshot, cutoff = _seal(block["runtime"])
    factorized_snapshot = _factorized_snapshot(base_snapshot, aligned=True)
    misaligned_snapshot = _factorized_snapshot(base_snapshot, aligned=False)
    adaptive_snapshot = _adaptive_snapshot(base_snapshot, aligned=True)
    base_atomic = _atomic(base_snapshot)
    factorized_atomic = _atomic(factorized_snapshot)
    singleton_parity = all(
        base_atomic[index]["hidden_first_arrival"]
        == factorized_atomic[index]["hidden_first_arrival"]
        and base_atomic[index]["target_at_6"]
        == factorized_atomic[index]["target_at_6"]
        for index in range(4)
    )
    factorized_pairs = [_pair_probe(factorized_snapshot, *pair) for pair in COMPOSITION_PAIRS]
    legacy_pairs = [_pair_probe(base_snapshot, *pair) for pair in COMPOSITION_PAIRS]
    adaptive_pairs = [_pair_probe(adaptive_snapshot, *pair) for pair in COMPOSITION_PAIRS]
    misaligned_pairs = [_pair_probe(misaligned_snapshot, *pair) for pair in COMPOSITION_PAIRS]
    independent = [_consistent_union(base_snapshot, *pair) for pair in COMPOSITION_PAIRS]
    gates = {
        "singleton_exact_parity": singleton_parity,
        "atomic_memory_intact": _atomic_success(factorized_atomic) == 4,
        "factorized_pair_composition": all(
            row["success"] and row["hidden_positive_count"] == 2
            for row in factorized_pairs
        ),
        "legacy_global_wta_fails": not any(row["success"] for row in legacy_pairs),
        "misaligned_source_receipt_fails": not any(row["success"] for row in misaligned_pairs),
        "independent_union_recovers": all(row["success"] for row in independent),
        "delay_aligned_packet_count": all(
            row["source_packet_count_by_tick"][3] == 2 for row in factorized_pairs
        ),
        "persistent_packet_stream_disclosed": all(
            sum(count > 0 for count in row["source_packet_count_by_tick"]) > 1
            for row in factorized_pairs
        ),
        "stores_zero": bool(
            cutoff["temporal_rows_after"] == 0
            and cutoff["hippocampal_rows_after"] == 0
            and all(row["hippocampal_rows_after"] == 0 for row in factorized_pairs)
        ),
    }
    return {
        "seed": int(seed),
        "status": "FACTORIZED_COMPOSITION_CONSISTENT_PASS" if all(gates.values()) else "FACTORIZED_COMPOSITION_CONSISTENT_FAIL",
        "gates": gates,
        "atomic_success_count": _atomic_success(factorized_atomic),
        "factorized_pair_success_count": sum(bool(row["success"]) for row in factorized_pairs),
        "legacy_pair_success_count": sum(bool(row["success"]) for row in legacy_pairs),
        "adaptive_pair_success_count": sum(bool(row["success"]) for row in adaptive_pairs),
        "misaligned_pair_success_count": sum(bool(row["success"]) for row in misaligned_pairs),
        "independent_union_success_count": sum(bool(row["success"]) for row in independent),
        "factorized_pairs": factorized_pairs,
        "legacy_pairs": legacy_pairs,
        "adaptive_pairs": adaptive_pairs,
        "misaligned_pairs": misaligned_pairs,
        "independent_union": independent,
        "endpoint_opened": False,
        "claim_scope": "synthetic decoder-consistent source-factorized packet-stream composition",
    }


def analyze_consistent_artifact(path: str | Path, *, stage: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "FRESH_INPUTS_READY":
        raise RuntimeError("fresh source-code inputs did not pass producer gates")
    rows = [
        analyze_consistent_row(int(row["seed"]), torch.tensor(row["candidate_weights"]))
        for row in payload["rows"]
    ]
    expected_count = 1 if stage == "calibration" else len(DEVELOPMENT_SEEDS)
    passed = len(rows) == expected_count and all(
        row["status"] == "FACTORIZED_COMPOSITION_CONSISTENT_PASS" for row in rows
    )
    return {
        "status": (
            "FACTORIZED_COMPOSITION_CALIBRATION_PASS"
            if passed and stage == "calibration"
            else "FACTORIZED_COMPOSITION_DEVELOPMENT_GO"
            if passed and stage == "development"
            else "FACTORIZED_COMPOSITION_STOP"
        ),
        "stage": stage,
        "seed_count": len(rows),
        "pass_count": sum(row["status"] == "FACTORIZED_COMPOSITION_CONSISTENT_PASS" for row in rows),
        "atomic_success_total": sum(row["atomic_success_count"] for row in rows),
        "factorized_pair_success_total": sum(row["factorized_pair_success_count"] for row in rows),
        "legacy_pair_success_total": sum(row["legacy_pair_success_count"] for row in rows),
        "adaptive_pair_success_total": sum(row["adaptive_pair_success_count"] for row in rows),
        "misaligned_pair_success_total": sum(row["misaligned_pair_success_count"] for row in rows),
        "independent_union_success_total": sum(row["independent_union_success_count"] for row in rows),
        "endpoint_opened": False,
        "confirmation_opened": False,
        "rows": rows,
    }


__all__ = [
    "CALIBRATION_SEEDS",
    "DEVELOPMENT_SEEDS",
    "generate_fresh_inputs",
    "analyze_consistent_artifact",
]

