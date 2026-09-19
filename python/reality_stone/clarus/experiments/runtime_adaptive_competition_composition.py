"""BA-TR17: delay-aligned input-count adaptive competition composition."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import torch

from ..runtime import BrainRuntime
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


CALIBRATION_SEEDS = (104001,)
DEVELOPMENT_SEEDS = tuple(range(104101, 104117))


def _adaptive_snapshot(snapshot: Any, *, aligned: bool) -> Any:
    source, _hidden, target = _blocks()
    input_indices = source if aligned else target
    config = replace(
        snapshot.config,
        competition_input_indices=tuple(input_indices),
        competition_k_from_delayed_input=True,
    )
    return replace(snapshot, config=config)


def _atomic_probes(snapshot: Any) -> list[dict[str, Any]]:
    return [_probe(snapshot, slot) for slot in range(4)]


def _atomic_success(rows: list[dict[str, Any]]) -> int:
    return sum(
        row["decoded_target"] == TARGET_MAPPING[row["source_slot"]]
        for row in rows
    )


def analyze_adaptive_row(seed: int, B: torch.Tensor) -> dict[str, Any]:
    block = _experience_block_compensated(B, condition="target_shuffle")
    base_snapshot, cutoff = _seal(block["runtime"])
    adaptive_snapshot = _adaptive_snapshot(base_snapshot, aligned=True)
    misaligned_snapshot = _adaptive_snapshot(base_snapshot, aligned=False)

    legacy_atomic = _atomic_probes(base_snapshot)
    adaptive_atomic = _atomic_probes(adaptive_snapshot)
    singleton_parity = all(
        legacy_atomic[index]["hidden_first_arrival"]
        == adaptive_atomic[index]["hidden_first_arrival"]
        and legacy_atomic[index]["target_at_6"]
        == adaptive_atomic[index]["target_at_6"]
        for index in range(4)
    )
    adaptive_pairs = [
        _combined_probe(adaptive_snapshot, left, right)
        for left, right in COMPOSITION_PAIRS
    ]
    legacy_pairs = [
        _combined_probe(base_snapshot, left, right)
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
        "atomic_memory_intact": _atomic_success(adaptive_atomic) == 4,
        "adaptive_pair_composition": all(
            row["success"] and row["hidden_positive_count"] == 2
            for row in adaptive_pairs
        ),
        "count_blind_control_fails": not any(row["success"] for row in legacy_pairs),
        "misaligned_count_control_fails": not any(
            row["success"] for row in misaligned_pairs
        ),
        "independent_union_recovers": all(row["success"] for row in independent),
        "stores_zero": bool(
            cutoff["temporal_rows_after"] == 0
            and cutoff["hippocampal_rows_after"] == 0
            and all(row["hippocampal_rows_after"] == 0 for row in adaptive_pairs)
        ),
    }
    return {
        "seed": int(seed),
        "status": "ADAPTIVE_COMPETITION_COMPOSITION_PASS" if all(gates.values()) else "ADAPTIVE_COMPETITION_COMPOSITION_FAIL",
        "gates": gates,
        "adaptive_atomic_success_count": _atomic_success(adaptive_atomic),
        "adaptive_pair_success_count": sum(bool(row["success"]) for row in adaptive_pairs),
        "legacy_pair_success_count": sum(bool(row["success"]) for row in legacy_pairs),
        "misaligned_pair_success_count": sum(bool(row["success"]) for row in misaligned_pairs),
        "independent_union_success_count": sum(bool(row["success"]) for row in independent),
        "legacy_atomic": legacy_atomic,
        "adaptive_atomic": adaptive_atomic,
        "adaptive_pairs": adaptive_pairs,
        "legacy_pairs": legacy_pairs,
        "misaligned_pairs": misaligned_pairs,
        "independent_union": independent,
        "endpoint_opened": False,
        "claim_scope": "synthetic delay-aligned two-packet adaptive competition",
    }


def analyze_adaptive_artifact(path: str | Path, *, stage: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "FRESH_INPUTS_READY":
        raise RuntimeError("fresh source-code inputs did not pass producer gates")
    rows = [
        analyze_adaptive_row(int(row["seed"]), torch.tensor(row["candidate_weights"]))
        for row in payload["rows"]
    ]
    expected_count = 1 if stage == "calibration" else len(DEVELOPMENT_SEEDS)
    passed = len(rows) == expected_count and all(
        row["status"] == "ADAPTIVE_COMPETITION_COMPOSITION_PASS" for row in rows
    )
    return {
        "status": (
            "ADAPTIVE_COMPETITION_CALIBRATION_PASS"
            if passed and stage == "calibration"
            else "ADAPTIVE_COMPETITION_DEVELOPMENT_GO"
            if passed and stage == "development"
            else "ADAPTIVE_COMPETITION_STOP"
        ),
        "stage": stage,
        "seed_count": len(rows),
        "pass_count": sum(row["status"] == "ADAPTIVE_COMPETITION_COMPOSITION_PASS" for row in rows),
        "atomic_success_total": sum(row["adaptive_atomic_success_count"] for row in rows),
        "adaptive_pair_success_total": sum(row["adaptive_pair_success_count"] for row in rows),
        "legacy_pair_success_total": sum(row["legacy_pair_success_count"] for row in rows),
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
    "_adaptive_snapshot",
    "analyze_adaptive_artifact",
]

