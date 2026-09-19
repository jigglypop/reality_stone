"""BA-TR22: relevance-selection no-go with a matched third event."""
from __future__ import annotations

from dataclasses import replace
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
from .runtime_factorized_one_shot_event_composition import _event_probe, _success


CALIBRATION_SEEDS = (109001,)
DEVELOPMENT_SEEDS = tuple(range(109101, 109117))
DISTRACTOR_INDEX = int(architectural_blocks(20)[1][0])


def _target_set(values: torch.Tensor) -> tuple[int, ...]:
    packed = torch.as_tensor(values, dtype=torch.float64).view(4)
    return tuple(
        int(index)
        for index in torch.nonzero(
            packed >= MIN_DECODE_ACTIVATION, as_tuple=False
        ).view(-1)
    )


def _matched_distractor_snapshot(snapshot: Any, source_slot: int) -> Any:
    """Copy one learned source column onto an outcome-blind distractor input."""
    source, hidden, _target = _blocks()
    packed = snapshot.weight.detach().clone()
    hidden_idx = torch.tensor(hidden)
    packed[hidden_idx, DISTRACTOR_INDEX] = packed[hidden_idx, source[source_slot]]
    return _all_input_snapshot(replace(snapshot, weight=packed))


def _three_event_probe(
    snapshot: Any,
    left: int,
    right: int,
) -> dict[str, Any]:
    source, hidden, target = _blocks()
    missing = tuple(slot for slot in range(4) if slot not in (left, right))
    distractor_source_slot = int(missing[0])
    routed = _matched_distractor_snapshot(snapshot, distractor_source_slot)
    runtime = BrainRuntime.from_snapshot(routed, backend="torch", device="cpu")
    input_gate_indices = torch.tensor(
        tuple(source) + tuple(int(v) for v in architectural_blocks(20)[1])
    )
    hidden_idx = torch.tensor(hidden)
    target_idx = torch.tensor(target)
    packet_counts: list[int] = []
    written_counts: list[int] = []
    hidden_first = torch.zeros(4)
    target_final = torch.zeros(4)
    for tick in range(PAIR_TICKS):
        if runtime._delay_buffer is None:
            raise RuntimeError("three-event test requires the delay ring")
        ring_slot = runtime._delay_idx % runtime.config.max_axon_delay
        delivered = runtime._delay_buffer[ring_slot]
        packet_counts.append(int(torch.count_nonzero(
            delivered[input_gate_indices].abs() > runtime.config.competition_epsilon
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
            runtime._delay_buffer[ring_slot, input_gate_indices] = 0.0
        written_counts.append(int(torch.count_nonzero(
            runtime._delay_buffer[ring_slot, input_gate_indices].abs()
            > runtime.config.competition_epsilon
        ).item()))
        if tick == 3:
            hidden_first = runtime.activation[hidden_idx].detach().clone()
        if tick == 6:
            target_final = runtime.activation[target_idx].detach().clone()
    desired_pair = tuple(sorted((TARGET_MAPPING[left], TARGET_MAPPING[right])))
    routed_three = tuple(sorted(desired_pair + (TARGET_MAPPING[distractor_source_slot],)))
    decoded = _target_set(target_final)
    return {
        "source_pair": [left, right],
        "distractor_index": DISTRACTOR_INDEX,
        "distractor_copied_source_slot": distractor_source_slot,
        "desired_pair_target_set": list(desired_pair),
        "routed_three_target_set": list(routed_three),
        "decoded_target_set": list(decoded),
        "desired_pair_success": decoded == desired_pair,
        "three_route_identity": decoded == routed_three,
        "hidden_first_arrival": [float(value) for value in hidden_first.tolist()],
        "hidden_positive_count": int(torch.count_nonzero(
            hidden_first > PRESYNAPTIC_EVENT_THRESHOLD
        ).item()),
        "target_at_6": [float(value) for value in target_final.tolist()],
        "input_packet_count_by_tick": packet_counts,
        "input_written_count_by_tick": written_counts,
        "hippocampal_rows_after": len(runtime.hippocampus),
    }


def analyze_relevance_no_go_row(seed: int, B: torch.Tensor) -> dict[str, Any]:
    block = _experience_block_compensated(B, condition="target_shuffle")
    base_snapshot, cutoff = _seal(block["runtime"])
    all_input = _all_input_snapshot(base_snapshot)
    pair_only = [
        _event_probe(all_input, pair, emission="one_shot")
        for pair in COMPOSITION_PAIRS
    ]
    three_event = [
        _three_event_probe(base_snapshot, *pair)
        for pair in COMPOSITION_PAIRS
    ]
    source, hidden, _target = _blocks()
    hidden_idx = torch.tensor(hidden)
    matched_columns = all(
        torch.equal(
            _matched_distractor_snapshot(base_snapshot, row["distractor_copied_source_slot"]).weight[
                hidden_idx, DISTRACTOR_INDEX
            ],
            base_snapshot.weight[
                hidden_idx, source[row["distractor_copied_source_slot"]]
            ],
        )
        for row in three_event
    )
    gates = {
        "pair_only_route_intact": all(_success(row) for row in pair_only),
        "matched_distractor_columns": matched_columns,
        "three_event_exact_packet_receipt": all(
            row["input_packet_count_by_tick"] == [0, 0, 0, 3, 0, 0, 0]
            and row["input_written_count_by_tick"] == [0, 3, 0, 0, 0, 0, 0]
            for row in three_event
        ),
        "three_local_routes_survive": all(
            row["three_route_identity"] and row["hidden_positive_count"] == 3
            for row in three_event
        ),
        "desired_pair_is_not_identified": not any(
            row["desired_pair_success"] for row in three_event
        ),
        "stores_zero": bool(
            cutoff["temporal_rows_after"] == 0
            and cutoff["hippocampal_rows_after"] == 0
            and all(row["hippocampal_rows_after"] == 0 for row in three_event)
        ),
    }
    return {
        "seed": int(seed),
        "status": "THREE_EVENT_RELEVANCE_NO_GO_WITNESS" if all(gates.values()) else "THREE_EVENT_APPARATUS_OR_BOUND_MISMATCH",
        "gates": gates,
        "pair_only_success_count": sum(_success(row) for row in pair_only),
        "desired_pair_success_count": sum(row["desired_pair_success"] for row in three_event),
        "three_route_identity_count": sum(row["three_route_identity"] for row in three_event),
        "three_event": three_event,
        "endpoint_opened": False,
        "claim_scope": "synthetic relevance-selection no-go for three locally valid one-shot packet routes",
    }


def analyze_relevance_no_go_artifact(path: str | Path, *, stage: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "FRESH_INPUTS_READY":
        raise RuntimeError("fresh source-code inputs did not pass producer gates")
    rows = [
        analyze_relevance_no_go_row(int(row["seed"]), torch.tensor(row["candidate_weights"]))
        for row in payload["rows"]
    ]
    expected_count = 1 if stage == "calibration" else len(DEVELOPMENT_SEEDS)
    witnessed = len(rows) == expected_count and all(
        row["status"] == "THREE_EVENT_RELEVANCE_NO_GO_WITNESS" for row in rows
    )
    return {
        "status": (
            "THREE_EVENT_RELEVANCE_CALIBRATION_PASS"
            if witnessed and stage == "calibration"
            else "THREE_EVENT_RELEVANCE_NO_GO_CONFIRMED"
            if witnessed and stage == "development"
            else "THREE_EVENT_RELEVANCE_APPARATUS_STOP"
        ),
        "stage": stage,
        "seed_count": len(rows),
        "witness_count": sum(row["status"] == "THREE_EVENT_RELEVANCE_NO_GO_WITNESS" for row in rows),
        "pair_only_success_total": sum(row["pair_only_success_count"] for row in rows),
        "desired_pair_success_total": sum(row["desired_pair_success_count"] for row in rows),
        "three_route_identity_total": sum(row["three_route_identity_count"] for row in rows),
        "endpoint_opened": False,
        "confirmation_opened": False,
        "rows": rows,
    }


__all__ = [
    "CALIBRATION_SEEDS",
    "DEVELOPMENT_SEEDS",
    "generate_fresh_inputs",
    "_three_event_probe",
    "analyze_relevance_no_go_artifact",
]

