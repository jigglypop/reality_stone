"""Held-out composition of two independent learned context gates.

The apparatus is the direct product of two frozen 20-dimensional TR3 payload
circuits.  Gate experience contains only the joint contexts 00, 01, and 10;
the factorwise count-normalized gates are then frozen before the 11 endpoint
is opened.  The module intentionally tests declared-factor composition, not
factor discovery or cross-factor interaction.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import inspect
import math
import textwrap
from typing import Any, Sequence

import torch

from ..runtime import RuntimeMode
from .runtime_context_branch_routing import (
    EPSILON,
    ApparatusInvalid,
    ContextBranchConfig,
    ExactDelayEligibility,
    _codebook_hash,
    _decode_y,
    _learn,
    _preflight as branch_preflight,
    _rollout,
    _runtime,
    _snapshot_hash,
    construct_context_branch_mask,
)
from .runtime_context_learned_gate import _branch_use, _entry_and_trunk_masks


TRAIN_CONTEXTS = ((0, 0), (0, 1), (1, 0))
HELDOUT_CONTEXT = (1, 1)
ROUTES = (
    "ORACLE",
    "FACTORWISE_LEARNED",
    "A_FACTOR_SHUFFLE_TRAIN",
    "B_FACTOR_SHUFFLE_TRAIN",
    "A_LESION_STATIC_0",
    "B_LESION_STATIC_0",
    "STATIC_00",
    "STATIC_01",
    "STATIC_10",
    "STATIC_11",
    "RANDOM_MATCHED_24",
    "FULL_32",
)
EXACT_24_NONORACLE_CONTROLS = (
    "A_FACTOR_SHUFFLE_TRAIN",
    "B_FACTOR_SHUFFLE_TRAIN",
    "A_LESION_STATIC_0",
    "B_LESION_STATIC_0",
    "STATIC_00",
    "STATIC_01",
    "STATIC_10",
    "STATIC_11",
    "RANDOM_MATCHED_24",
)
FORBIDDEN_FACTOR_GATE_NAMES = {
    "answer",
    "decoder",
    "endpoint",
    "expected",
    "factor_name",
    "joint_context",
    "oracle",
    "other_factor",
    "payload",
    "reward",
    "route",
    "schedule",
    "seed",
    "sigma",
    "target",
    "task",
}


@dataclass(frozen=True)
class FactorCompositionConfig:
    seed: int = 97701
    gate_min_logit_margin: float = 1e-6

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.gate_min_logit_margin)):
            raise ValueError("gate margin must be finite")
        if self.gate_min_logit_margin <= 0.0:
            raise ValueError("gate margin must be positive")


@dataclass(frozen=True)
class CountNormalizedGateSnapshot:
    theta: torch.Tensor
    accumulator: torch.Tensor
    counts: torch.Tensor
    update_count: int
    min_logit_margin: float


class CountNormalizedFactorGate:
    """Two-actuator local association normalized by factor-cue exposure."""

    def __init__(self, min_logit_margin: float = 1e-6) -> None:
        self.min_logit_margin = float(min_logit_margin)
        if not math.isfinite(self.min_logit_margin) or self.min_logit_margin <= 0.0:
            raise ValueError("gate margin must be finite and positive")
        self.accumulator = torch.zeros(2, 2, dtype=torch.float64)
        self.counts = torch.zeros(2, dtype=torch.float64)
        self.theta = torch.zeros(2, 2, dtype=torch.float64)
        self.update_count = 0

    def observe(self, factor_cue: torch.Tensor, branch_use: torch.Tensor) -> float:
        cue = torch.as_tensor(factor_cue, dtype=torch.float64).view(-1)
        use = torch.as_tensor(branch_use, dtype=torch.float64).view(-1)
        if cue.shape != (2,) or use.shape != (2,):
            raise ApparatusInvalid("APPARATUS_INVALID: invalid factor-gate observation shape")
        if not torch.isfinite(cue).all() or not torch.isfinite(use).all():
            raise ApparatusInvalid("APPARATUS_INVALID: nonfinite factor-gate observation")
        if torch.any(use < 0.0):
            raise ApparatusInvalid("APPARATUS_INVALID: branch use must be nonnegative")
        if not (
            torch.all((cue == 0.0) | (cue == 1.0))
            and float(cue.sum().item()) == 1.0
        ):
            raise ApparatusInvalid("APPARATUS_INVALID: factor cue must be exactly one-hot")
        before = self.theta.clone()
        self.accumulator += torch.outer(use, cue)
        self.counts += cue
        positive = self.counts > 0.0
        self.theta[:, positive] = self.accumulator[:, positive] / self.counts[positive]
        self.update_count += 1
        return float((self.theta - before).norm().item())

    def snapshot(self) -> CountNormalizedGateSnapshot:
        return CountNormalizedGateSnapshot(
            theta=self.theta.detach().clone(),
            accumulator=self.accumulator.detach().clone(),
            counts=self.counts.detach().clone(),
            update_count=int(self.update_count),
            min_logit_margin=self.min_logit_margin,
        )


def _validate_gate_snapshot(snapshot: CountNormalizedGateSnapshot) -> torch.Tensor:
    theta = torch.as_tensor(snapshot.theta, dtype=torch.float64)
    accumulator = torch.as_tensor(snapshot.accumulator, dtype=torch.float64)
    counts = torch.as_tensor(snapshot.counts, dtype=torch.float64)
    if theta.shape != (2, 2) or accumulator.shape != (2, 2) or counts.shape != (2,):
        raise ApparatusInvalid("APPARATUS_INVALID: invalid frozen factor-gate shape")
    if not torch.isfinite(theta).all() or not torch.isfinite(accumulator).all() or not torch.isfinite(counts).all():
        raise ApparatusInvalid("APPARATUS_INVALID: nonfinite frozen factor gate")
    if torch.any(counts <= 0.0) or snapshot.update_count <= 0:
        raise ApparatusInvalid("APPARATUS_INVALID: factor gate has an unobserved cue")
    reference = accumulator / counts.view(1, 2)
    if not torch.equal(theta, reference):
        raise ApparatusInvalid("APPARATUS_INVALID: frozen factor gate is not count-normalized")
    if not math.isfinite(float(snapshot.min_logit_margin)) or snapshot.min_logit_margin <= 0.0:
        raise ApparatusInvalid("APPARATUS_INVALID: invalid frozen factor-gate margin")
    return theta


def compile_factor_mask(
    gate_snapshot: CountNormalizedGateSnapshot,
    factor_cue: torch.Tensor,
    weight: torch.Tensor,
    blocks: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compile one branch mask from frozen local state and one factor cue."""
    theta = _validate_gate_snapshot(gate_snapshot)
    cue = torch.as_tensor(factor_cue, dtype=torch.float64).view(-1)
    if cue.shape != (2,) or not torch.isfinite(cue).all():
        raise ApparatusInvalid("APPARATUS_INVALID: invalid factor cue")
    logits = theta @ cue
    if not torch.isfinite(logits).all():
        raise ApparatusInvalid("APPARATUS_INVALID: nonfinite factor-gate logits")
    margin = float(torch.abs(logits[0] - logits[1]).item())
    if margin < gate_snapshot.min_logit_margin:
        raise ApparatusInvalid("APPARATUS_INVALID: unresolved factor-gate tie")
    selected = int(torch.argmax(logits).item())
    branches, trunk = _entry_and_trunk_masks(weight, blocks)
    mask = trunk | branches[selected]
    return mask.to(weight.dtype), {
        "selected_branch": selected,
        "logits": [float(value) for value in logits.tolist()],
        "logit_margin": margin,
        "mask_edges": int(mask.sum().item()),
    }


def _gate_hash(snapshot: CountNormalizedGateSnapshot) -> str:
    digest = hashlib.sha256()
    for tensor in (snapshot.theta, snapshot.accumulator, snapshot.counts):
        packed = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(packed.shape)).encode())
        digest.update(str(packed.dtype).encode())
        digest.update(packed.numpy().tobytes())
    digest.update(repr((snapshot.update_count, snapshot.min_logit_margin)).encode())
    return digest.hexdigest()


def _reference_action(
    snapshot: CountNormalizedGateSnapshot,
    cue: torch.Tensor,
) -> tuple[int, list[float], float]:
    theta = torch.as_tensor(snapshot.theta, dtype=torch.float64).detach().clone()
    vector = torch.as_tensor(cue, dtype=torch.float64).detach().clone().view(2)
    scores = torch.mv(theta, vector)
    margin = float(torch.abs(scores[0] - scores[1]).item())
    if not torch.isfinite(scores).all() or margin < snapshot.min_logit_margin:
        raise ApparatusInvalid("APPARATUS_INVALID: reference factor action unresolved")
    return int(torch.argmax(scores).item()), [float(value) for value in scores.tolist()], margin


def _function_identifiers(function: Any) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _factor_task(seed: int) -> dict[str, Any]:
    parity_a = int(seed) & 1
    parity_b = (int(seed) >> 1) & 1
    return {
        "cues": torch.eye(2, dtype=torch.float64),
        "mapping_A": (parity_a, 1 - parity_a),
        "mapping_B": (parity_b, 1 - parity_b),
        "parity_pair": (parity_a, parity_b),
        "source_seed_A": int(seed) + 1_000_003,
        "source_seed_B": int(seed) + 2_000_003,
    }


def _factor_episode_use(
    books: dict[str, Any],
    source_snapshot: Any,
    branch: int,
    payload: int,
    base: ContextBranchConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    runtime = _runtime(base)
    tracker = ExactDelayEligibility(
        base.dim,
        base.delay_ticks,
        base.eligibility_decay,
        base.ltd,
    )
    source = books[f"S{int(branch)}"][int(payload)]
    hidden = books[f"H{int(branch)}"][int(payload)]
    for tick in range(base.delay_ticks + 1):
        external = torch.zeros(base.dim)
        if tick == 0:
            external = base.cue_drive_gain * source
        elif tick == base.delay_ticks:
            external = base.cue_drive_gain * hidden
        runtime.step(
            external_input=external,
            force_mode=RuntimeMode.WAKE,
            learning_signal=0.0,
        )
        tracker.observe(runtime.activation)
    branch_masks, _ = _entry_and_trunk_masks(source_snapshot.weight, books["blocks"])
    use = _branch_use(tracker.eligibility, branch_masks)
    return use, {
        "experienced_branch": int(branch),
        "payload_slot": int(payload),
        "branch_use": [float(value) for value in use.tolist()],
        "experienced_branch_use": float(use[int(branch)].item()),
        "other_branch_use": float(use[1 - int(branch)].item()),
        "paired_observations": int(tracker.paired_observations),
        "target_pulse_count": 0,
        "decoder_read_count": 0,
        "endpoint_read_count": 0,
    }


def _collect_experience(
    source_A: Any,
    books_A: dict[str, Any],
    source_B: Any,
    books_B: dict[str, Any],
    task: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_A = ContextBranchConfig(seed=int(task["source_seed_A"]))
    base_B = ContextBranchConfig(seed=int(task["source_seed_B"]))
    hashes_before = (_snapshot_hash(source_A), _snapshot_hash(source_B))
    rows: list[dict[str, Any]] = []
    for a, b in TRAIN_CONTEXTS:
        branch_A = int(task["mapping_A"][a])
        branch_B = int(task["mapping_B"][b])
        for payload in range(base_A.payload_width):
            use_A, receipt_A = _factor_episode_use(
                books_A, source_A, branch_A, payload, base_A,
            )
            use_B, receipt_B = _factor_episode_use(
                books_B, source_B, branch_B, payload, base_B,
            )
            rows.append({
                "context": (int(a), int(b)),
                "payload_repetition": int(payload),
                "cue_A": [float(value) for value in task["cues"][a].tolist()],
                "cue_B": [float(value) for value in task["cues"][b].tolist()],
                "use_A": [float(value) for value in use_A.tolist()],
                "use_B": [float(value) for value in use_B.tolist()],
                "factor_A": receipt_A,
                "factor_B": receipt_B,
            })
    hashes_after = (_snapshot_hash(source_A), _snapshot_hash(source_B))
    return rows, {
        "training_contexts": [list(value) for value in TRAIN_CONTEXTS],
        "training_row_count": len(rows),
        "source_hashes_before": hashes_before,
        "source_hashes_after": hashes_after,
        "sources_immutable": hashes_before == hashes_after,
        "target_pulse_count": 0,
        "decoder_read_count": 0,
        "endpoint_read_count": 0,
    }


def _fit_factor_gate(
    rows: list[dict[str, Any]],
    factor_slot: int,
    min_logit_margin: float,
    *,
    shuffle: bool,
) -> tuple[CountNormalizedGateSnapshot, dict[str, Any]]:
    gate = CountNormalizedFactorGate(min_logit_margin)
    label = "A" if factor_slot == 0 else "B"
    update_rows: list[dict[str, Any]] = []
    for row in rows:
        cue_slot = int(row["context"][factor_slot])
        observed_slot = 1 - cue_slot if shuffle else cue_slot
        cue = torch.tensor((1.0, 0.0) if observed_slot == 0 else (0.0, 1.0), dtype=torch.float64)
        use = torch.tensor(row[f"use_{label}"], dtype=torch.float64)
        change = gate.observe(cue, use)
        update_rows.append({
            "factor_cue_slot": cue_slot,
            "observed_cue_slot": observed_slot,
            "theta_change_norm": change,
        })
    frozen = gate.snapshot()
    return frozen, {
        "shuffle": bool(shuffle),
        "update_count": frozen.update_count,
        "counts": [float(value) for value in frozen.counts.tolist()],
        "accumulator": [[float(value) for value in row] for row in frozen.accumulator.tolist()],
        "theta": [[float(value) for value in row] for row in frozen.theta.tolist()],
        "gate_sha256": _gate_hash(frozen),
        "target_read_count": 0,
        "decoder_read_count": 0,
        "endpoint_read_count": 0,
        "rows": update_rows,
    }


def _joint_lookup_receipt(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accumulator = torch.zeros(4, 4, dtype=torch.float64)
    counts = torch.zeros(4, dtype=torch.float64)
    for row in rows:
        a, b = (int(value) for value in row["context"])
        column = 2 * a + b
        use_A = torch.tensor(row["use_A"], dtype=torch.float64)
        use_B = torch.tensor(row["use_B"], dtype=torch.float64)
        accumulator[:, column] += torch.kron(use_A, use_B)
        counts[column] += 1.0
    theta = torch.zeros_like(accumulator)
    positive = counts > 0.0
    theta[:, positive] = accumulator[:, positive] / counts[positive]
    heldout_logits = theta[:, 3]
    return {
        "counts": [float(value) for value in counts.tolist()],
        "heldout_column": [float(value) for value in theta[:, 3].tolist()],
        "heldout_logits": [float(value) for value in heldout_logits.tolist()],
        "heldout_tie": bool(torch.count_nonzero(heldout_logits) == 0),
        "heldout_abstains": bool(counts[3] == 0.0 and torch.count_nonzero(heldout_logits) == 0),
        "endpoint_opened": False,
    }


def _mask_for_branch(
    weight: torch.Tensor,
    blocks: Sequence[Sequence[int]],
    branch: int,
) -> torch.Tensor:
    return construct_context_branch_mask(weight, int(branch), blocks, 0, "CORRECT")


def _compose_pair_mask(mask_A: torch.Tensor, mask_B: torch.Tensor) -> torch.Tensor:
    return torch.block_diag(mask_A.bool(), mask_B.bool())


def _factor_gate_receipts(
    label: str,
    source: Any,
    books: dict[str, Any],
    mapping: tuple[int, int],
    cues: torch.Tensor,
    normal: CountNormalizedGateSnapshot,
    shuffled: CountNormalizedGateSnapshot,
    normal_receipt: dict[str, Any],
    shuffled_receipt: dict[str, Any],
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    learned = [compile_factor_mask(normal, cues[index], source.weight, books["blocks"])
               for index in (0, 1)]
    adverse = [compile_factor_mask(shuffled, cues[index], source.weight, books["blocks"])
               for index in (0, 1)]
    references = [_reference_action(normal, cues[index]) for index in (0, 1)]
    learned_masks = [row[0].bool() for row in learned]
    learned_info = [row[1] for row in learned]
    adverse_info = [row[1] for row in adverse]
    branches, trunk = _entry_and_trunk_masks(source.weight, books["blocks"])
    expected_masks = [trunk | branches[mapping[index]] for index in (0, 1)]

    metadata_only = {"seed": -1, "sigma": (9, 9), "schedule": "not-an-input", "factor": label}
    del metadata_only
    repeated = [compile_factor_mask(normal, cues[index], source.weight, books["blocks"])[0].bool()
                for index in (0, 1)]
    swapped_cues = [compile_factor_mask(normal, cues[1 - index], source.weight, books["blocks"])[0].bool()
                    for index in (0, 1)]
    row_swapped = CountNormalizedGateSnapshot(
        theta=normal.theta.flip(0).clone(),
        accumulator=normal.accumulator.flip(0).clone(),
        counts=normal.counts.clone(),
        update_count=normal.update_count,
        min_logit_margin=normal.min_logit_margin,
    )
    counterfactual = [compile_factor_mask(
        row_swapped, cues[index], source.weight, books["blocks"],
    ) for index in (0, 1)]
    counter_refs = [_reference_action(row_swapped, cues[index]) for index in (0, 1)]
    recomputed = normal.accumulator / normal.counts.view(1, 2)
    actions = tuple(int(row["selected_branch"]) for row in learned_info)
    shuffled_actions = tuple(int(row["selected_branch"]) for row in adverse_info)
    gates = {
        "positive_unequal_counts": bool(torch.all(normal.counts > 0.0) and normal.counts[0] != normal.counts[1]),
        "exact_count_normalization": bool(torch.equal(normal.theta, recomputed)),
        "normal_count_receipt_matches": normal_receipt["counts"] == [float(value) for value in normal.counts.tolist()],
        "shuffled_count_receipt_matches": shuffled_receipt["counts"] == [float(value) for value in shuffled.counts.tolist()],
        "independent_theta_q_reference": all(
            references[index][0] == learned_info[index]["selected_branch"]
            and references[index][1] == learned_info[index]["logits"]
            for index in (0, 1)
        ),
        "learned_actions_match_mapping": actions == mapping,
        "shuffled_actions_reverse_mapping": shuffled_actions == tuple(1 - value for value in mapping),
        "expected_masks": all(torch.equal(learned_masks[index], expected_masks[index]) for index in (0, 1)),
        "metadata_invariance": all(torch.equal(repeated[index], learned_masks[index]) for index in (0, 1)),
        "cue_swap_equivariance": all(torch.equal(swapped_cues[index], learned_masks[1 - index]) for index in (0, 1)),
        "theta_counterfactual_dependence": all(
            counterfactual[index][1]["selected_branch"] == counter_refs[index][0]
            and counterfactual[index][1]["selected_branch"] == 1 - learned_info[index]["selected_branch"]
            for index in (0, 1)
        ),
        "mask_budget": [int(mask.sum().item()) for mask in learned_masks] == [12, 12],
        "mask_difference": int((learned_masks[0] != learned_masks[1]).sum().item()) == 8,
        "common_trunk": bool(torch.equal(learned_masks[0] & trunk, learned_masks[1] & trunk))
            and int((learned_masks[0] & trunk).sum().item()) == 8,
    }
    return {
        "all_pass": all(gates.values()),
        "gates": gates,
        "mapping": mapping,
        "learned_actions": actions,
        "shuffled_actions": shuffled_actions,
        "counts": tuple(float(value) for value in normal.counts.tolist()),
        "shuffled_counts": tuple(float(value) for value in shuffled.counts.tolist()),
        "theta": tuple(tuple(float(value) for value in row) for row in normal.theta.tolist()),
        "gate_sha256": _gate_hash(normal),
        "shuffled_gate_sha256": _gate_hash(shuffled),
    }, learned_masks


def _preflight(
    source_A: Any,
    books_A: dict[str, Any],
    recurrent_A: dict[str, Any],
    source_B: Any,
    books_B: dict[str, Any],
    recurrent_B: dict[str, Any],
    task: dict[str, Any],
    experience_rows: list[dict[str, Any]],
    experience_receipt: dict[str, Any],
    gate_A: CountNormalizedGateSnapshot,
    gate_B: CountNormalizedGateSnapshot,
    shuffled_A: CountNormalizedGateSnapshot,
    shuffled_B: CountNormalizedGateSnapshot,
    gate_receipts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base_A = branch_preflight(source_A, books_A, recurrent_A, int(task["source_seed_A"]))
    base_B = branch_preflight(source_B, books_B, recurrent_B, int(task["source_seed_B"]))
    cues = torch.as_tensor(task["cues"], dtype=torch.float64)
    mapping_A = tuple(int(value) for value in task["mapping_A"])
    mapping_B = tuple(int(value) for value in task["mapping_B"])
    receipt_A, masks_A = _factor_gate_receipts(
        "A", source_A, books_A, mapping_A, cues, gate_A, shuffled_A,
        gate_receipts["A"], gate_receipts["A_shuffled"],
    )
    receipt_B, masks_B = _factor_gate_receipts(
        "B", source_B, books_B, mapping_B, cues, gate_B, shuffled_B,
        gate_receipts["B"], gate_receipts["B_shuffled"],
    )

    pair_masks = {
        (a, b): _compose_pair_mask(masks_A[a], masks_B[b])
        for a in (0, 1) for b in (0, 1)
    }
    _, trunk_A = _entry_and_trunk_masks(source_A.weight, books_A["blocks"])
    _, trunk_B = _entry_and_trunk_masks(source_B.weight, books_B["blocks"])
    common_trunk = _compose_pair_mask(trunk_A, trunk_B)
    joint_weight = torch.block_diag(source_A.weight, source_B.weight)
    direct_sum_cross_nonzero = int(torch.count_nonzero(joint_weight[:20, 20:]).item()) + int(
        torch.count_nonzero(joint_weight[20:, :20]).item()
    )
    contexts = [tuple(int(value) for value in row["context"]) for row in experience_rows]
    context_counts = {key: contexts.count(key) for key in (*TRAIN_CONTEXTS, HELDOUT_CONTEXT)}
    local_separation = all(
        row["factor_A"]["experienced_branch_use"] > row["factor_A"]["other_branch_use"] + 1e-6
        and row["factor_B"]["experienced_branch_use"] > row["factor_B"]["other_branch_use"] + 1e-6
        for row in experience_rows
    )
    compiler_signature = tuple(inspect.signature(compile_factor_mask).parameters)
    observe_signature = tuple(inspect.signature(CountNormalizedFactorGate.observe).parameters)
    identifiers = _function_identifiers(compile_factor_mask) | _function_identifiers(
        CountNormalizedFactorGate.observe
    )
    joint_lookup = _joint_lookup_receipt(experience_rows)
    source_hashes = (_snapshot_hash(source_A), _snapshot_hash(source_B))
    decoder_hashes = (_codebook_hash(books_A), _codebook_hash(books_B))
    pair_counts = {str(key): int(mask.sum().item()) for key, mask in pair_masks.items()}
    pair_trunks = {
        key: int((mask & common_trunk).sum().item()) for key, mask in pair_masks.items()
    }
    hamming = {
        "00_to_10": int((pair_masks[(0, 0)] != pair_masks[(1, 0)]).sum().item()),
        "00_to_01": int((pair_masks[(0, 0)] != pair_masks[(0, 1)]).sum().item()),
        "00_to_11": int((pair_masks[(0, 0)] != pair_masks[(1, 1)]).sum().item()),
    }
    gates = {
        "base_A_preflight": bool(base_A["all_pass"]),
        "base_B_preflight": bool(base_B["all_pass"]),
        "exact_training_multiset": context_counts == {
            (0, 0): 4, (0, 1): 4, (1, 0): 4, (1, 1): 0,
        },
        "heldout_absent": HELDOUT_CONTEXT not in contexts,
        "factor_values_observed": all(any(row["context"][slot] == value for row in experience_rows)
                                      for slot in (0, 1) for value in (0, 1)),
        "local_branch_use_separation": local_separation,
        "no_target_decoder_endpoint_reads": bool(
            experience_receipt["target_pulse_count"] == 0
            and experience_receipt["decoder_read_count"] == 0
            and experience_receipt["endpoint_read_count"] == 0
        ),
        "sources_immutable_during_experience": bool(experience_receipt["sources_immutable"]),
        "factor_A_gate": bool(receipt_A["all_pass"]),
        "factor_B_gate": bool(receipt_B["all_pass"]),
        "normal_counts": receipt_A["counts"] == (8.0, 4.0) and receipt_B["counts"] == (8.0, 4.0),
        "shuffled_counts": receipt_A["shuffled_counts"] == (4.0, 8.0)
            and receipt_B["shuffled_counts"] == (4.0, 8.0),
        "factor_gate_input_signature": bool(
            compiler_signature == ("gate_snapshot", "factor_cue", "weight", "blocks")
            and observe_signature == ("self", "factor_cue", "branch_use")
            and compile_factor_mask.__closure__ is None
            and not bool(identifiers & FORBIDDEN_FACTOR_GATE_NAMES)
        ),
        "pair_mask_budget": all(value == 24 for value in pair_counts.values()),
        "common_output_trunk": all(value == 16 for value in pair_trunks.values()),
        "pair_mask_hamming": hamming == {"00_to_10": 8, "00_to_01": 8, "00_to_11": 16},
        "direct_sum_cross_support_zero": direct_sum_cross_nonzero == 0,
        "separate_runtime_objects": source_A is not source_B and books_A is not books_B,
        "joint_lookup_holdout_abstains": bool(joint_lookup["heldout_abstains"]),
        "source_cutoff": bool(
            recurrent_A["cutoff"]["hippocampal_rows_after"] == 0
            and recurrent_B["cutoff"]["hippocampal_rows_after"] == 0
        ),
    }
    return {
        "all_pass": all(gates.values()),
        "gates": gates,
        "base_A": base_A,
        "base_B": base_B,
        "factor_A": receipt_A,
        "factor_B": receipt_B,
        "training_context_counts": {"".join(map(str, key)): value for key, value in context_counts.items()},
        "compiler_signature": compiler_signature,
        "observe_signature": observe_signature,
        "compiler_identifiers": sorted(identifiers),
        "pair_mask_edge_counts": pair_counts,
        "pair_mask_hamming": hamming,
        "common_trunk_edges": int(common_trunk.sum().item()),
        "direct_sum_shape": tuple(joint_weight.shape),
        "direct_sum_cross_nonzero": direct_sum_cross_nonzero,
        "source_snapshot_sha256": source_hashes,
        "decoder_sha256": decoder_hashes,
        "joint_lookup_holdout_abstain": joint_lookup,
    }


def _factor_mask_for_route(
    route_name: str,
    factor_slot: int,
    source: Any,
    books: dict[str, Any],
    task: dict[str, Any],
    normal: CountNormalizedGateSnapshot,
    shuffled: CountNormalizedGateSnapshot,
) -> tuple[torch.Tensor, dict[str, Any]]:
    label = "A" if factor_slot == 0 else "B"
    mapping = task[f"mapping_{label}"]
    cue = task["cues"][1]
    blocks = books["blocks"]
    weight = source.weight
    if route_name == "ORACLE":
        selected = int(mapping[1])
        return _mask_for_branch(weight, blocks, selected), {"selected_branch": selected, "logit_margin": None}
    if route_name == "FACTORWISE_LEARNED":
        return compile_factor_mask(normal, cue, weight, blocks)
    if route_name == "A_FACTOR_SHUFFLE_TRAIN" and factor_slot == 0:
        return compile_factor_mask(shuffled, cue, weight, blocks)
    if route_name == "B_FACTOR_SHUFFLE_TRAIN" and factor_slot == 1:
        return compile_factor_mask(shuffled, cue, weight, blocks)
    if route_name in {"A_FACTOR_SHUFFLE_TRAIN", "B_FACTOR_SHUFFLE_TRAIN"}:
        return compile_factor_mask(normal, cue, weight, blocks)
    if route_name == "A_LESION_STATIC_0" and factor_slot == 0:
        return _mask_for_branch(weight, blocks, 0), {"selected_branch": 0, "logit_margin": None}
    if route_name == "B_LESION_STATIC_0" and factor_slot == 1:
        return _mask_for_branch(weight, blocks, 0), {"selected_branch": 0, "logit_margin": None}
    if route_name in {"A_LESION_STATIC_0", "B_LESION_STATIC_0"}:
        return compile_factor_mask(normal, cue, weight, blocks)
    if route_name.startswith("STATIC_"):
        pair = route_name.removeprefix("STATIC_")
        selected = int(pair[factor_slot])
        return _mask_for_branch(weight, blocks, selected), {"selected_branch": selected, "logit_margin": None}
    if route_name == "RANDOM_MATCHED_24":
        random_key = int(task["source_seed_A" if factor_slot == 0 else "source_seed_B"]) + 31_337
        mask = construct_context_branch_mask(weight, 0, blocks, random_key, "RANDOM_MATCHED")
        return mask, {"selected_branch": -1, "logit_margin": None}
    if route_name == "FULL_32":
        mask = construct_context_branch_mask(weight, 0, blocks, 0, "FULL")
        return mask, {"selected_branch": -1, "logit_margin": None}
    raise ValueError(f"unknown route {route_name!r}")


def _evaluate_factor(
    route_name: str,
    factor_slot: int,
    source: Any,
    books: dict[str, Any],
    task: dict[str, Any],
    normal: CountNormalizedGateSnapshot,
    shuffled: CountNormalizedGateSnapshot,
) -> dict[str, Any]:
    label = "A" if factor_slot == 0 else "B"
    base = ContextBranchConfig(seed=int(task[f"source_seed_{label}"]))
    expected_branch = int(task[f"mapping_{label}"][1])
    mask, gate_info = _factor_mask_for_route(
        route_name, factor_slot, source, books, task, normal, shuffled,
    )
    rows: list[dict[str, Any]] = []
    for left in range(base.payload_width):
        for right in range(base.payload_width):
            if left == right:
                continue
            expected = left if expected_branch == 0 else right
            opposite = right if expected_branch == 0 else left
            sensory = books["S0"][left] + books["S1"][right]
            final, metrics = _rollout(source, mask, mask, sensory, base, books["blocks"])
            hidden_norms = metrics.pop("hidden_norms_at_arrival")
            decoded = _decode_y(final, books, expected, opposite, base)
            rows.append({
                "left_payload": left,
                "right_payload": right,
                "expected_branch": expected_branch,
                "selected_branch": int(gate_info["selected_branch"]),
                "mask_edges": int(mask.sum().item()),
                "hidden_norms_at_arrival": hidden_norms,
                **decoded,
                **metrics,
            })
    count = len(rows)
    return {
        "factor": label,
        "accuracy": sum(int(row["success"]) for row in rows) / count,
        "opposite_delivery": sum(int(row["opposite_delivery"]) for row in rows) / count,
        "mean_margin": sum(float(row["margin"]) for row in rows) / count,
        "mean_runtime_energy_proxy": sum(float(row["runtime_energy_proxy"]) for row in rows) / count,
        "mean_active_fraction": sum(float(row["active_fraction"]) for row in rows) / count,
        "mask_edges": int(mask.sum().item()),
        "selected_branch": int(gate_info["selected_branch"]),
        "expected_branch": expected_branch,
        "logit_margin": gate_info.get("logit_margin"),
        "hippocampal_rows_after": max(int(row["hippocampal_rows_after"]) for row in rows),
        "trials": rows,
    }


def _evaluate_pair_route(
    route_name: str,
    source_A: Any,
    books_A: dict[str, Any],
    source_B: Any,
    books_B: dict[str, Any],
    task: dict[str, Any],
    gate_A: CountNormalizedGateSnapshot,
    gate_B: CountNormalizedGateSnapshot,
    shuffled_A: CountNormalizedGateSnapshot,
    shuffled_B: CountNormalizedGateSnapshot,
) -> dict[str, Any]:
    factor_A = _evaluate_factor(route_name, 0, source_A, books_A, task, gate_A, shuffled_A)
    factor_B = _evaluate_factor(route_name, 1, source_B, books_B, task, gate_B, shuffled_B)
    cartesian_outcomes = bytes(
        int(row_A["success"] and row_B["success"])
        for row_A in factor_A["trials"]
        for row_B in factor_B["trials"]
    )
    factor_A_trial_receipt = tuple(
        (row["left_payload"], row["right_payload"], row["prediction"], row["success"], row["opposite_delivery"])
        for row in factor_A["trials"]
    )
    factor_B_trial_receipt = tuple(
        (row["left_payload"], row["right_payload"], row["prediction"], row["success"], row["opposite_delivery"])
        for row in factor_B["trials"]
    )
    factor_A_summary = {key: value for key, value in factor_A.items() if key != "trials"}
    factor_B_summary = {key: value for key, value in factor_B.items() if key != "trials"}
    count = len(cartesian_outcomes)
    success_count = sum(cartesian_outcomes)
    return {
        "route": route_name,
        "joint_accuracy": success_count / count,
        "A_accuracy": factor_A["accuracy"],
        "B_accuracy": factor_B["accuracy"],
        "A_opposite_delivery": factor_A["opposite_delivery"],
        "B_opposite_delivery": factor_B["opposite_delivery"],
        "mask_edges": factor_A["mask_edges"] + factor_B["mask_edges"],
        "mean_runtime_energy_proxy": factor_A["mean_runtime_energy_proxy"]
            + factor_B["mean_runtime_energy_proxy"],
        "mean_active_fraction": 0.5 * (
            factor_A["mean_active_fraction"] + factor_B["mean_active_fraction"]
        ),
        "cartesian_trial_count": count,
        "cartesian_success_count": success_count,
        "cartesian_conjunction_sha256": hashlib.sha256(cartesian_outcomes).hexdigest(),
        "cartesian_rule": "A_success AND B_success",
        "factor_A_trials_sha256": hashlib.sha256(repr(factor_A_trial_receipt).encode()).hexdigest(),
        "factor_B_trials_sha256": hashlib.sha256(repr(factor_B_trial_receipt).encode()).hexdigest(),
        "factor_A": factor_A_summary,
        "factor_B": factor_B_summary,
    }


def run_factor_composition_seed(
    seed: int = 97701,
    *,
    config: FactorCompositionConfig | None = None,
) -> dict[str, Any]:
    selected = config or FactorCompositionConfig(seed=int(seed))
    config = FactorCompositionConfig(**{**asdict(selected), "seed": int(seed)})
    task = _factor_task(int(seed))
    base_A = ContextBranchConfig(seed=int(task["source_seed_A"]))
    base_B = ContextBranchConfig(seed=int(task["source_seed_B"]))
    source_A, books_A, recurrent_A = _learn(int(task["source_seed_A"]), base_A)
    source_B, books_B, recurrent_B = _learn(int(task["source_seed_B"]), base_B)
    experience_rows, experience_receipt = _collect_experience(
        source_A, books_A, source_B, books_B, task,
    )
    gate_A, receipt_A = _fit_factor_gate(
        experience_rows, 0, config.gate_min_logit_margin, shuffle=False,
    )
    gate_B, receipt_B = _fit_factor_gate(
        experience_rows, 1, config.gate_min_logit_margin, shuffle=False,
    )
    shuffled_A, shuffled_receipt_A = _fit_factor_gate(
        experience_rows, 0, config.gate_min_logit_margin, shuffle=True,
    )
    shuffled_B, shuffled_receipt_B = _fit_factor_gate(
        experience_rows, 1, config.gate_min_logit_margin, shuffle=True,
    )
    gate_receipts = {
        "A": receipt_A,
        "B": receipt_B,
        "A_shuffled": shuffled_receipt_A,
        "B_shuffled": shuffled_receipt_B,
    }
    preflight = _preflight(
        source_A, books_A, recurrent_A,
        source_B, books_B, recurrent_B,
        task, experience_rows, experience_receipt,
        gate_A, gate_B, shuffled_A, shuffled_B, gate_receipts,
    )
    if not preflight["all_pass"]:
        return {
            "seed": int(seed),
            "status": "APPARATUS_INVALID",
            "endpoint_opened": False,
            "config": asdict(config),
            "task": {
                "mapping_A": task["mapping_A"],
                "mapping_B": task["mapping_B"],
                "parity_pair": task["parity_pair"],
            },
            "preflight": preflight,
            "experience": experience_receipt,
            "gate_learning": gate_receipts,
        }

    source_hashes_before = (_snapshot_hash(source_A), _snapshot_hash(source_B))
    gate_hashes_before = tuple(_gate_hash(value) for value in (gate_A, gate_B, shuffled_A, shuffled_B))
    routes = {
        route: _evaluate_pair_route(
            route, source_A, books_A, source_B, books_B, task,
            gate_A, gate_B, shuffled_A, shuffled_B,
        )
        for route in ROUTES
    }
    source_hashes_after = (_snapshot_hash(source_A), _snapshot_hash(source_B))
    gate_hashes_after = tuple(_gate_hash(value) for value in (gate_A, gate_B, shuffled_A, shuffled_B))
    frozen_after = source_hashes_before == source_hashes_after and gate_hashes_before == gate_hashes_after
    learned = routes["FACTORWISE_LEARNED"]["joint_accuracy"]
    oracle = routes["ORACLE"]["joint_accuracy"]
    shuffle_A = routes["A_FACTOR_SHUFFLE_TRAIN"]
    shuffle_B = routes["B_FACTOR_SHUFFLE_TRAIN"]
    stores_zero = all(
        route["factor_A"]["hippocampal_rows_after"] == 0
        and route["factor_B"]["hippocampal_rows_after"] == 0
        for route in routes.values()
    )
    seed_pass = bool(
        learned >= 0.95
        and oracle >= 0.95
        and oracle - learned <= 0.05
        and shuffle_A["joint_accuracy"] <= 0.05
        and shuffle_A["A_opposite_delivery"] >= 0.95
        and shuffle_A["B_accuracy"] >= 0.95
        and shuffle_B["joint_accuracy"] <= 0.05
        and shuffle_B["B_opposite_delivery"] >= 0.95
        and shuffle_B["A_accuracy"] >= 0.95
        and frozen_after
        and stores_zero
    )
    return {
        "seed": int(seed),
        "status": "FACTOR_COMPOSITION_PASS" if seed_pass else "FACTOR_COMPOSITION_NOT_IDENTIFIED",
        "endpoint_opened": True,
        "heldout_context": HELDOUT_CONTEXT,
        "config": asdict(config),
        "task": {
            "mapping_A": task["mapping_A"],
            "mapping_B": task["mapping_B"],
            "parity_pair": task["parity_pair"],
            "source_seed_A": task["source_seed_A"],
            "source_seed_B": task["source_seed_B"],
        },
        "preflight": preflight,
        "experience": experience_receipt,
        "gate_learning": gate_receipts,
        "routes": routes,
        "learned_oracle_gap": oracle - learned,
        "all_frozen_after_evaluation": frozen_after,
        "stores_zero_after_evaluation": stores_zero,
        "source_snapshot_sha256_before_evaluation": source_hashes_before,
        "source_snapshot_sha256_after_evaluation": source_hashes_after,
        "gate_sha256_before_evaluation": gate_hashes_before,
        "gate_sha256_after_evaluation": gate_hashes_after,
    }
