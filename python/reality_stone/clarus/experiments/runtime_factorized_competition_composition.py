"""BA-TR18: source-factorized delayed competition before route summation."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import torch

from .runtime_adaptive_competition_composition import _adaptive_snapshot
from .runtime_binding_composition_no_go import (
    COMPOSITION_PAIRS,
    TARGET_MAPPING,
    _combined_probe,
    _independent_union,
)
from .runtime_experience_attenuation_binding import (
    _experience_block_compensated,
    generate_fresh_inputs,
)
from .runtime_experience_delayed_binding import _blocks, _probe, _seal


CALIBRATION_SEEDS = (105001,)
DEVELOPMENT_SEEDS = tuple(range(105101, 105117))


def _factorized_snapshot(snapshot: Any, *, aligned: bool) -> Any:
    source, _hidden, target = _blocks()
    config = replace(
        snapshot.config,
        competition_input_indices=tuple(source if aligned else target),
        competition_k_from_delayed_input=False,
        competition_factorize_delayed_input=True,
    )
    return replace(snapshot, config=config)


def _atomic(snapshot: Any) -> list[dict[str, Any]]:
    return [_probe(snapshot, slot) for slot in range(4)]


def _atomic_success(rows: list[dict[str, Any]]) -> int:
    return sum(
        row["decoded_target"] == TARGET_MAPPING[row["source_slot"]]
        for row in rows
    )


def analyze_factorized_row(seed: int, B: torch.Tensor) -> dict[str, Any]:
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
    factorized_pairs = [
        _combined_probe(factorized_snapshot, left, right)
        for left, right in COMPOSITION_PAIRS
    ]
    legacy_pairs = [
        _combined_probe(base_snapshot, left, right)
        for left, right in COMPOSITION_PAIRS
    ]
    adaptive_pairs = [
        _combined_probe(adaptive_snapshot, left, right)
        for left, right in COMPOSITION_PAIRS
    ]
    misaligned_pairs = [
        _combined_probe(misaligned_snapshot, left, right)
        for left, right in COMPOSITION_PAIRS
    ]
    independent = [
        _independent_union(base_snapshot, left, right)
        for left, right in COMPOSITION_PAIRS
    ]
    gates = {
        "singleton_exact_parity": singleton_parity,
        "atomic_memory_intact": _atomic_success(factorized_atomic) == 4,
        "factorized_pair_composition": all(
            row["success"] and row["hidden_positive_count"] == 2
            for row in factorized_pairs
        ),
        "legacy_global_wta_fails": not any(row["success"] for row in legacy_pairs),
        "misaligned_source_receipt_fails": not any(
            row["success"] for row in misaligned_pairs
        ),
        "independent_union_recovers": all(row["success"] for row in independent),
        "stores_zero": bool(
            cutoff["temporal_rows_after"] == 0
            and cutoff["hippocampal_rows_after"] == 0
            and all(row["hippocampal_rows_after"] == 0 for row in factorized_pairs)
        ),
    }
    return {
        "seed": int(seed),
        "status": "FACTORIZED_COMPETITION_COMPOSITION_PASS" if all(gates.values()) else "FACTORIZED_COMPETITION_COMPOSITION_FAIL",
        "gates": gates,
        "factorized_atomic_success_count": _atomic_success(factorized_atomic),
        "factorized_pair_success_count": sum(bool(row["success"]) for row in factorized_pairs),
        "legacy_pair_success_count": sum(bool(row["success"]) for row in legacy_pairs),
        "adaptive_top2_pair_success_count": sum(bool(row["success"]) for row in adaptive_pairs),
        "misaligned_pair_success_count": sum(bool(row["success"]) for row in misaligned_pairs),
        "independent_union_success_count": sum(bool(row["success"]) for row in independent),
        "base_atomic": base_atomic,
        "factorized_atomic": factorized_atomic,
        "factorized_pairs": factorized_pairs,
        "legacy_pairs": legacy_pairs,
        "adaptive_top2_pairs": adaptive_pairs,
        "misaligned_pairs": misaligned_pairs,
        "independent_union": independent,
        "endpoint_opened": False,
        "claim_scope": "synthetic source-factorized two-packet competition",
    }


def analyze_factorized_artifact(path: str | Path, *, stage: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "FRESH_INPUTS_READY":
        raise RuntimeError("fresh source-code inputs did not pass producer gates")
    rows = [
        analyze_factorized_row(int(row["seed"]), torch.tensor(row["candidate_weights"]))
        for row in payload["rows"]
    ]
    expected_count = 1 if stage == "calibration" else len(DEVELOPMENT_SEEDS)
    passed = len(rows) == expected_count and all(
        row["status"] == "FACTORIZED_COMPETITION_COMPOSITION_PASS" for row in rows
    )
    return {
        "status": (
            "FACTORIZED_COMPETITION_CALIBRATION_PASS"
            if passed and stage == "calibration"
            else "FACTORIZED_COMPETITION_DEVELOPMENT_GO"
            if passed and stage == "development"
            else "FACTORIZED_COMPETITION_STOP"
        ),
        "stage": stage,
        "seed_count": len(rows),
        "pass_count": sum(row["status"] == "FACTORIZED_COMPETITION_COMPOSITION_PASS" for row in rows),
        "atomic_success_total": sum(row["factorized_atomic_success_count"] for row in rows),
        "factorized_pair_success_total": sum(row["factorized_pair_success_count"] for row in rows),
        "legacy_pair_success_total": sum(row["legacy_pair_success_count"] for row in rows),
        "adaptive_top2_pair_success_total": sum(row["adaptive_top2_pair_success_count"] for row in rows),
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
    "_factorized_snapshot",
    "analyze_factorized_artifact",
]

