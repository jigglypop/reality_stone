"""BA-TR24: held-out factor composition for context packet relevance."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .runtime_binding_composition_no_go import COMPOSITION_PAIRS
from .runtime_context_packet_relevance_gate import _gated_three_event_probe
from .runtime_experience_attenuation_binding import (
    _experience_block_compensated,
    generate_fresh_inputs,
)
from .runtime_experience_delayed_binding import _blocks, _seal


CALIBRATION_SEEDS = (111001,)
DEVELOPMENT_SEEDS = tuple(range(111101, 111117))
TRAIN_CONTEXTS = (0, 1, 2)
HELDOUT_CONTEXT = 3
FACTOR_CONTEXTS = ((0, 0), (0, 1), (1, 0), (1, 1))
GATE_THRESHOLD = 0.5


@dataclass(frozen=True)
class FactorContextGateSnapshot:
    theta_a: torch.Tensor
    theta_b: torch.Tensor
    codes_a: torch.Tensor
    codes_b: torch.Tensor
    counts_a: torch.Tensor
    counts_b: torch.Tensor
    training_contexts: tuple[int, ...]


def _factor_codes(seed: int, offset: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + offset)
    q, _ = torch.linalg.qr(torch.randn(2, 2, generator=generator))
    return q.T.contiguous()


def train_factor_context_gate(seed: int) -> FactorContextGateSnapshot:
    """Count-normalized local context-factor/event co-occurrence."""
    codes_a = _factor_codes(seed, 240_007)
    codes_b = _factor_codes(seed, 250_013)
    uses_a = torch.zeros(20, 2)
    uses_b = torch.zeros(20, 2)
    counts_a = torch.zeros(2)
    counts_b = torch.zeros(2)
    source, _hidden, _target = _blocks()
    for context_index in TRAIN_CONTEXTS:
        a, b = FACTOR_CONTEXTS[context_index]
        uses_a[source[a], a] += 1.0
        uses_b[source[2 + b], b] += 1.0
        counts_a[a] += 1.0
        counts_b[b] += 1.0
    if torch.any(counts_a <= 0.0) or torch.any(counts_b <= 0.0):
        raise RuntimeError("every context factor value must appear in training")
    normalized_a = uses_a / counts_a.unsqueeze(0)
    normalized_b = uses_b / counts_b.unsqueeze(0)
    return FactorContextGateSnapshot(
        theta_a=normalized_a @ codes_a,
        theta_b=normalized_b @ codes_b,
        codes_a=codes_a,
        codes_b=codes_b,
        counts_a=counts_a,
        counts_b=counts_b,
        training_contexts=TRAIN_CONTEXTS,
    )


def compile_factor_context_indices(
    gate: FactorContextGateSnapshot,
    code_a: torch.Tensor,
    code_b: torch.Tensor,
) -> tuple[int, ...]:
    qa = torch.as_tensor(code_a, dtype=torch.float32).view(2)
    qb = torch.as_tensor(code_b, dtype=torch.float32).view(2)
    scores = gate.theta_a @ qa + gate.theta_b @ qb
    selected = tuple(
        int(index)
        for index in torch.nonzero(scores > GATE_THRESHOLD, as_tuple=False).view(-1)
    )
    if len(selected) != 2:
        raise RuntimeError("factor context gate did not select exactly two inputs")
    return selected


def _gate_hash(gate: FactorContextGateSnapshot) -> str:
    digest = hashlib.sha256()
    for tensor in (
        gate.theta_a,
        gate.theta_b,
        gate.codes_a,
        gate.codes_b,
        gate.counts_a,
        gate.counts_b,
    ):
        digest.update(tensor.detach().cpu().numpy().tobytes())
    digest.update(repr(gate.training_contexts).encode("ascii"))
    return digest.hexdigest()


def _compile_context(gate: FactorContextGateSnapshot, context_index: int) -> tuple[int, ...]:
    a, b = FACTOR_CONTEXTS[context_index]
    return compile_factor_context_indices(
        gate,
        gate.codes_a[a],
        gate.codes_b[b],
    )


def analyze_factor_context_row(seed: int, B: torch.Tensor) -> dict[str, Any]:
    block = _experience_block_compensated(B, condition="target_shuffle")
    base_snapshot, cutoff = _seal(block["runtime"])
    gate = train_factor_context_gate(seed)
    before = _gate_hash(gate)
    source, _hidden, _target = _blocks()
    compiled = [_compile_context(gate, index) for index in range(4)]
    expected_indices = [
        (source[pair[0]], source[pair[1]]) for pair in COMPOSITION_PAIRS
    ]
    training_rows = [
        _gated_three_event_probe(
            base_snapshot,
            *COMPOSITION_PAIRS[index],
            compiled[index],
        )
        for index in TRAIN_CONTEXTS
    ]
    pair = COMPOSITION_PAIRS[HELDOUT_CONTEXT]
    heldout = _gated_three_event_probe(
        base_snapshot,
        *pair,
        compiled[HELDOUT_CONTEXT],
    )
    oracle = _gated_three_event_probe(
        base_snapshot,
        *pair,
        expected_indices[HELDOUT_CONTEXT],
    )
    joint_unseen_fallback = _gated_three_event_probe(
        base_snapshot,
        *pair,
        expected_indices[0],
    )
    a, b = FACTOR_CONTEXTS[HELDOUT_CONTEXT]
    shuffle_a = _gated_three_event_probe(
        base_snapshot,
        *pair,
        compile_factor_context_indices(
            gate,
            gate.codes_a[1 - a],
            gate.codes_b[b],
        ),
    )
    shuffle_b = _gated_three_event_probe(
        base_snapshot,
        *pair,
        compile_factor_context_indices(
            gate,
            gate.codes_a[a],
            gate.codes_b[1 - b],
        ),
    )
    no_context = _gated_three_event_probe(base_snapshot, *pair, None)
    after = _gate_hash(gate)
    gates = {
        "heldout_absent_from_training": HELDOUT_CONTEXT not in gate.training_contexts,
        "every_factor_value_observed": bool(
            torch.equal(gate.counts_a, torch.tensor([2.0, 1.0]))
            and torch.equal(gate.counts_b, torch.tensor([2.0, 1.0]))
        ),
        "all_four_factor_compilers_exact": compiled == expected_indices,
        "training_contexts_recalled": all(row["success"] for row in training_rows),
        "heldout_factor_composition": bool(
            heldout["success"] and heldout["hidden_positive_count"] == 2
        ),
        "heldout_oracle_bit_exact": heldout["target_at_6"] == oracle["target_at_6"],
        "joint_lookup_unseen_fails": not joint_unseen_fallback["success"],
        "factor_a_shuffle_fails": not shuffle_a["success"],
        "factor_b_shuffle_fails": not shuffle_b["success"],
        "no_context_fails": not no_context["success"],
        "gate_frozen": before == after,
        "stores_zero": bool(
            cutoff["temporal_rows_after"] == 0
            and cutoff["hippocampal_rows_after"] == 0
            and heldout["hippocampal_rows_after"] == 0
        ),
    }
    return {
        "seed": int(seed),
        "status": "HELDOUT_FACTOR_CONTEXT_RELEVANCE_PASS" if all(gates.values()) else "HELDOUT_FACTOR_CONTEXT_RELEVANCE_FAIL",
        "gates": gates,
        "training_contexts": list(gate.training_contexts),
        "heldout_context": HELDOUT_CONTEXT,
        "counts_a": [float(value) for value in gate.counts_a.tolist()],
        "counts_b": [float(value) for value in gate.counts_b.tolist()],
        "compiled_indices": [list(values) for values in compiled],
        "gate_hash": before,
        "heldout_success": bool(heldout["success"]),
        "oracle_success": bool(oracle["success"]),
        "joint_lookup_success": bool(joint_unseen_fallback["success"]),
        "factor_a_shuffle_success": bool(shuffle_a["success"]),
        "factor_b_shuffle_success": bool(shuffle_b["success"]),
        "no_context_success": bool(no_context["success"]),
        "heldout": heldout,
        "endpoint_opened": False,
        "claim_scope": "synthetic held-out composition of two declared context factors",
    }


def analyze_factor_context_artifact(path: str | Path, *, stage: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "FRESH_INPUTS_READY":
        raise RuntimeError("fresh source-code inputs did not pass producer gates")
    rows = [
        analyze_factor_context_row(int(row["seed"]), torch.tensor(row["candidate_weights"]))
        for row in payload["rows"]
    ]
    expected_count = 1 if stage == "calibration" else len(DEVELOPMENT_SEEDS)
    passed = len(rows) == expected_count and all(
        row["status"] == "HELDOUT_FACTOR_CONTEXT_RELEVANCE_PASS" for row in rows
    )
    return {
        "status": (
            "FACTOR_CONTEXT_CALIBRATION_PASS"
            if passed and stage == "calibration"
            else "FACTOR_CONTEXT_DEVELOPMENT_GO"
            if passed and stage == "development"
            else "FACTOR_CONTEXT_STOP"
        ),
        "stage": stage,
        "seed_count": len(rows),
        "pass_count": sum(row["status"] == "HELDOUT_FACTOR_CONTEXT_RELEVANCE_PASS" for row in rows),
        "heldout_success_total": sum(row["heldout_success"] for row in rows),
        "oracle_success_total": sum(row["oracle_success"] for row in rows),
        "joint_lookup_success_total": sum(row["joint_lookup_success"] for row in rows),
        "factor_a_shuffle_success_total": sum(row["factor_a_shuffle_success"] for row in rows),
        "factor_b_shuffle_success_total": sum(row["factor_b_shuffle_success"] for row in rows),
        "no_context_success_total": sum(row["no_context_success"] for row in rows),
        "endpoint_opened": False,
        "confirmation_opened": False,
        "rows": rows,
    }


__all__ = [
    "CALIBRATION_SEEDS",
    "DEVELOPMENT_SEEDS",
    "TRAIN_CONTEXTS",
    "HELDOUT_CONTEXT",
    "generate_fresh_inputs",
    "train_factor_context_gate",
    "compile_factor_context_indices",
    "analyze_factor_context_artifact",
]

