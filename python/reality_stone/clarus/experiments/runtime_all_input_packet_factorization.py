"""BA-TR21: factorize every arriving presynaptic packet coordinate."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import torch

from .runtime_binding_composition_no_go import COMPOSITION_PAIRS, TARGET_MAPPING
from .runtime_experience_attenuation_binding import (
    _experience_block_compensated,
    generate_fresh_inputs,
)
from .runtime_experience_delayed_binding import _blocks, _seal
from .runtime_factorized_one_shot_event_composition import (
    _event_probe,
    _success,
)


CALIBRATION_SEEDS = (108001,)
DEVELOPMENT_SEEDS = tuple(range(108101, 108117))


def _all_input_snapshot(snapshot: Any) -> Any:
    """Enable factorization over the whole runtime coordinate chart."""
    config = replace(
        snapshot.config,
        competition_input_indices=tuple(range(int(snapshot.config.dim))),
        competition_k_from_delayed_input=False,
        competition_factorize_delayed_input=True,
    )
    return replace(snapshot, config=config)


def _source_projected_snapshot(snapshot: Any) -> Any:
    source, _hidden, _target = _blocks()
    config = replace(
        snapshot.config,
        competition_input_indices=tuple(source),
        competition_k_from_delayed_input=False,
        competition_factorize_delayed_input=True,
    )
    return replace(snapshot, config=config)


def _shift_source_columns(snapshot: Any) -> Any:
    """Adverse control: break packet-column identity without changing norms."""
    source, hidden, _target = _blocks()
    packed = snapshot.weight.detach().clone()
    source_idx = torch.tensor(source)
    hidden_idx = torch.tensor(hidden)
    original = packed[hidden_idx[:, None], source_idx].clone()
    packed[hidden_idx[:, None], source_idx] = original[:, [1, 2, 3, 0]]
    return _all_input_snapshot(replace(snapshot, weight=packed))


def _expected(slots: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(TARGET_MAPPING[slot] for slot in slots))


def _independent_union(snapshot: Any, left: int, right: int) -> dict[str, Any]:
    left_row = _event_probe(snapshot, (left,), emission="one_shot")
    right_row = _event_probe(snapshot, (right,), emission="one_shot")
    values = torch.maximum(
        torch.tensor(left_row["target_at_6"]),
        torch.tensor(right_row["target_at_6"]),
    )
    decoded = tuple(
        int(index)
        for index in torch.nonzero(values >= 1e-5, as_tuple=False).view(-1)
    )
    expected = _expected((left, right))
    return {
        "source_slots": [left, right],
        "decoded_target_set": list(decoded),
        "expected_target_set": list(expected),
        "success": decoded == expected,
    }


def analyze_all_input_row(seed: int, B: torch.Tensor) -> dict[str, Any]:
    block = _experience_block_compensated(B, condition="target_shuffle")
    base_snapshot, cutoff = _seal(block["runtime"])
    all_input = _all_input_snapshot(base_snapshot)
    source_projected = _source_projected_snapshot(base_snapshot)
    shifted = _shift_source_columns(base_snapshot)

    all_atomic = [
        _event_probe(all_input, (slot,), emission="one_shot")
        for slot in range(4)
    ]
    projected_atomic = [
        _event_probe(source_projected, (slot,), emission="one_shot")
        for slot in range(4)
    ]
    all_pairs = [
        _event_probe(all_input, pair, emission="one_shot")
        for pair in COMPOSITION_PAIRS
    ]
    projected_pairs = [
        _event_probe(source_projected, pair, emission="one_shot")
        for pair in COMPOSITION_PAIRS
    ]
    legacy_pairs = [
        _event_probe(base_snapshot, pair, emission="one_shot")
        for pair in COMPOSITION_PAIRS
    ]
    shifted_pairs = [
        _event_probe(shifted, pair, emission="one_shot")
        for pair in COMPOSITION_PAIRS
    ]
    suppressed_pairs = [
        _event_probe(all_input, pair, emission="suppressed")
        for pair in COMPOSITION_PAIRS
    ]
    independent = [
        _independent_union(all_input, *pair)
        for pair in COMPOSITION_PAIRS
    ]
    parity = all(
        all_atomic[index]["hidden_first_arrival"]
        == projected_atomic[index]["hidden_first_arrival"]
        and all_atomic[index]["target_at_6"]
        == projected_atomic[index]["target_at_6"]
        for index in range(4)
    ) and all(
        all_pairs[index]["hidden_first_arrival"]
        == projected_pairs[index]["hidden_first_arrival"]
        and all_pairs[index]["target_at_6"]
        == projected_pairs[index]["target_at_6"]
        for index in range(len(COMPOSITION_PAIRS))
    )
    gates = {
        "all_input_exact_projector_parity": parity,
        "all_input_atomic_memory_intact": all(_success(row) for row in all_atomic),
        "all_input_pair_composition": all(
            _success(row)
            and row["hidden_positive_count"] == 2
            and row["source_packet_count_by_tick"] == [0, 0, 0, 2, 0, 0, 0]
            for row in all_pairs
        ),
        "legacy_global_wta_fails": not any(_success(row) for row in legacy_pairs),
        "shifted_packet_columns_fail": not any(_success(row) for row in shifted_pairs),
        "suppressed_event_fails": not any(_success(row) for row in suppressed_pairs),
        "independent_union_recovers": all(row["success"] for row in independent),
        "all_coordinate_input_set": tuple(all_input.config.competition_input_indices or ())
        == tuple(range(int(all_input.config.dim))),
        "stores_zero": bool(
            cutoff["temporal_rows_after"] == 0
            and cutoff["hippocampal_rows_after"] == 0
            and all(row["hippocampal_rows_after"] == 0 for row in all_pairs)
        ),
    }
    return {
        "seed": int(seed),
        "status": "ALL_INPUT_PACKET_FACTORIZATION_PASS" if all(gates.values()) else "ALL_INPUT_PACKET_FACTORIZATION_FAIL",
        "gates": gates,
        "atomic_success_count": sum(_success(row) for row in all_atomic),
        "all_input_pair_success_count": sum(_success(row) for row in all_pairs),
        "source_projected_pair_success_count": sum(_success(row) for row in projected_pairs),
        "legacy_pair_success_count": sum(_success(row) for row in legacy_pairs),
        "shifted_column_pair_success_count": sum(_success(row) for row in shifted_pairs),
        "suppressed_pair_success_count": sum(_success(row) for row in suppressed_pairs),
        "independent_union_success_count": sum(row["success"] for row in independent),
        "all_input_pairs": all_pairs,
        "source_projected_pairs": projected_pairs,
        "shifted_column_pairs": shifted_pairs,
        "endpoint_opened": False,
        "claim_scope": "synthetic one-shot composition using all arriving packet coordinates and sparse learned support",
    }


def analyze_all_input_artifact(path: str | Path, *, stage: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "FRESH_INPUTS_READY":
        raise RuntimeError("fresh source-code inputs did not pass producer gates")
    rows = [
        analyze_all_input_row(int(row["seed"]), torch.tensor(row["candidate_weights"]))
        for row in payload["rows"]
    ]
    expected_count = 1 if stage == "calibration" else len(DEVELOPMENT_SEEDS)
    passed = len(rows) == expected_count and all(
        row["status"] == "ALL_INPUT_PACKET_FACTORIZATION_PASS" for row in rows
    )
    return {
        "status": (
            "ALL_INPUT_PACKET_CALIBRATION_PASS"
            if passed and stage == "calibration"
            else "ALL_INPUT_PACKET_DEVELOPMENT_GO"
            if passed and stage == "development"
            else "ALL_INPUT_PACKET_STOP"
        ),
        "stage": stage,
        "seed_count": len(rows),
        "pass_count": sum(row["status"] == "ALL_INPUT_PACKET_FACTORIZATION_PASS" for row in rows),
        "atomic_success_total": sum(row["atomic_success_count"] for row in rows),
        "all_input_pair_success_total": sum(row["all_input_pair_success_count"] for row in rows),
        "source_projected_pair_success_total": sum(row["source_projected_pair_success_count"] for row in rows),
        "legacy_pair_success_total": sum(row["legacy_pair_success_count"] for row in rows),
        "shifted_column_pair_success_total": sum(row["shifted_column_pair_success_count"] for row in rows),
        "suppressed_pair_success_total": sum(row["suppressed_pair_success_count"] for row in rows),
        "independent_union_success_total": sum(row["independent_union_success_count"] for row in rows),
        "endpoint_opened": False,
        "confirmation_opened": False,
        "rows": rows,
    }


__all__ = [
    "CALIBRATION_SEEDS",
    "DEVELOPMENT_SEEDS",
    "generate_fresh_inputs",
    "_all_input_snapshot",
    "analyze_all_input_artifact",
]

