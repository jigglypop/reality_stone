"""BA-TR16: direct runtime witness for the global-WTA composition no-go."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from ..runtime import BrainRuntime, RuntimeMode
from .runtime_experience_attenuation_binding import (
    MIN_COMPENSATED_MARGIN,
    _experience_block_compensated,
    generate_fresh_inputs,
)
from .runtime_experience_delayed_binding import (
    PAIR_TICKS,
    PRESYNAPTIC_EVENT_THRESHOLD,
    _blocks,
    _external,
    _probe,
    _seal,
)


CALIBRATION_SEEDS = (103001,)
DEVELOPMENT_SEEDS = tuple(range(103101, 103117))
TARGET_MAPPING = (1, 2, 3, 0)
COMPOSITION_PAIRS = ((0, 2), (0, 3), (1, 2), (1, 3))


def _global_wta(values: torch.Tensor) -> torch.Tensor:
    packed = torch.as_tensor(values, dtype=torch.float64).view(-1)
    peers = packed.unsqueeze(0).expand(packed.numel(), -1).clone()
    peers.fill_diagonal_(float("-inf"))
    return (packed - peers.max(dim=1).values).clamp_min(0.0)


def _active_target_set(values: torch.Tensor) -> tuple[int, ...]:
    packed = torch.as_tensor(values, dtype=torch.float64).view(4)
    return tuple(
        int(index)
        for index in torch.nonzero(packed >= MIN_COMPENSATED_MARGIN, as_tuple=False).view(-1)
    )


def _combined_probe(snapshot: Any, left: int, right: int) -> dict[str, Any]:
    source, hidden, target = _blocks()
    runtime = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    hidden_first = torch.zeros(4)
    target_final = torch.zeros(4)
    for tick in range(PAIR_TICKS):
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
            hidden_first = runtime.activation[torch.tensor(hidden)].detach().clone()
        if tick == 6:
            target_final = runtime.activation[torch.tensor(target)].detach().clone()
    expected = tuple(sorted((TARGET_MAPPING[left], TARGET_MAPPING[right])))
    decoded = _active_target_set(target_final)
    return {
        "source_pair": [left, right],
        "expected_target_set": list(expected),
        "decoded_target_set": list(decoded),
        "success": decoded == expected,
        "hidden_first_arrival": [float(value) for value in hidden_first.tolist()],
        "hidden_positive_count": int(
            torch.count_nonzero(hidden_first > PRESYNAPTIC_EVENT_THRESHOLD).item()
        ),
        "target_at_6": [float(value) for value in target_final.tolist()],
        "external_nonzero_ticks": [0],
        "zero_input_ticks": [1, 2, 3, 4, 5, 6],
        "hippocampal_rows_after": len(runtime.hippocampus),
    }


def _independent_union(snapshot: Any, left: int, right: int) -> dict[str, Any]:
    left_probe = _probe(snapshot, left)
    right_probe = _probe(snapshot, right)
    union = torch.maximum(
        torch.tensor(left_probe["target_at_6"]),
        torch.tensor(right_probe["target_at_6"]),
    )
    expected = tuple(sorted((TARGET_MAPPING[left], TARGET_MAPPING[right])))
    decoded = _active_target_set(union)
    return {
        "source_pair": [left, right],
        "expected_target_set": list(expected),
        "decoded_target_set": list(decoded),
        "success": decoded == expected,
        "union_target": [float(value) for value in union.tolist()],
    }


def analyze_composition_row(seed: int, B: torch.Tensor) -> dict[str, Any]:
    block = _experience_block_compensated(B, condition="target_shuffle")
    snapshot, cutoff = _seal(block["runtime"])
    atomic = [_probe(snapshot, slot) for slot in range(4)]
    atomic_success = sum(
        row["decoded_target"] == TARGET_MAPPING[row["source_slot"]]
        for row in atomic
    )
    combined = [_combined_probe(snapshot, left, right) for left, right in COMPOSITION_PAIRS]
    independent = [_independent_union(snapshot, left, right) for left, right in COMPOSITION_PAIRS]
    gates = {
        "atomic_nonidentity_memory_intact": atomic_success == 4,
        "global_wta_capacity_bound_observed": all(
            row["hidden_positive_count"] <= 1 for row in combined
        ),
        "simultaneous_composition_fails": not any(row["success"] for row in combined),
        "independent_union_recovers_both": all(row["success"] for row in independent),
        "stores_zero": bool(
            cutoff["temporal_rows_after"] == 0
            and cutoff["hippocampal_rows_after"] == 0
            and all(row["hippocampal_rows_after"] == 0 for row in combined)
        ),
    }
    return {
        "seed": int(seed),
        "status": "GLOBAL_WTA_COMPOSITION_NO_GO_WITNESS" if all(gates.values()) else "COMPOSITION_APPARATUS_OR_BOUND_MISMATCH",
        "gates": gates,
        "atomic_success_count": atomic_success,
        "simultaneous_success_count": sum(bool(row["success"]) for row in combined),
        "independent_union_success_count": sum(bool(row["success"]) for row in independent),
        "atomic": atomic,
        "combined": combined,
        "independent_union": independent,
        "endpoint_opened": False,
        "claim_scope": "synthetic global-WTA simultaneous composition no-go",
    }


def analyze_composition_artifact(path: str | Path, *, stage: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "FRESH_INPUTS_READY":
        raise RuntimeError("fresh source-code inputs did not pass producer gates")
    rows = [
        analyze_composition_row(int(row["seed"]), torch.tensor(row["candidate_weights"]))
        for row in payload["rows"]
    ]
    witnessed = all(row["status"] == "GLOBAL_WTA_COMPOSITION_NO_GO_WITNESS" for row in rows)
    expected_count = 1 if stage == "calibration" else len(DEVELOPMENT_SEEDS)
    confirmed = witnessed and len(rows) == expected_count
    return {
        "status": (
            "GLOBAL_WTA_COMPOSITION_NO_GO_CONFIRMED"
            if confirmed
            else "GLOBAL_WTA_COMPOSITION_NO_GO_NOT_CONFIRMED"
        ),
        "stage": stage,
        "seed_count": len(rows),
        "witness_count": sum(row["status"] == "GLOBAL_WTA_COMPOSITION_NO_GO_WITNESS" for row in rows),
        "atomic_success_total": sum(row["atomic_success_count"] for row in rows),
        "simultaneous_success_total": sum(row["simultaneous_success_count"] for row in rows),
        "independent_union_success_total": sum(row["independent_union_success_count"] for row in rows),
        "endpoint_opened": False,
        "confirmation_opened": False,
        "rows": rows,
    }


__all__ = [
    "CALIBRATION_SEEDS",
    "DEVELOPMENT_SEEDS",
    "generate_fresh_inputs",
    "_global_wta",
    "analyze_composition_artifact",
]

