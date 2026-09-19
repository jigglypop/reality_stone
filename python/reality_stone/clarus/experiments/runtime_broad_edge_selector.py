"""Context-conditioned local edge selection on a uniform broad substrate.

Each factor has 32 equally weighted S->H candidate edges and a fixed
four-edge H->Y trunk.  Exact-delay experience supplies a normalized local
edge field; a frozen factor cue selects four candidates without reading the
recurrent weight matrix, a branch family, or an endpoint.
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

from ..runtime import BrainRuntime, RuntimeMode
from .runtime_context_branch_routing import (
    EPSILON,
    ApparatusInvalid,
    ContextBranchConfig,
    ExactDelayEligibility,
    _block_matrix,
    _codebook_hash,
    _decode_y,
    _learn,
    _matrix_rank,
    _rollout,
    _runtime,
    _snapshot_hash,
)


TRAIN_CONTEXTS = ((0, 0), (0, 1), (1, 0))
HELDOUT_CONTEXT = (1, 1)
ROUTES = (
    "ORACLE",
    "EDGE_FIELD_LEARNED",
    "A_FACTOR_SHUFFLE_TRAIN",
    "B_FACTOR_SHUFFLE_TRAIN",
    "STATIC_00",
    "STATIC_01",
    "STATIC_10",
    "STATIC_11",
    "RANDOM_MATCHED_16",
    "FULL_72",
)
EXACT_16_NONORACLE_CONTROLS = (
    "A_FACTOR_SHUFFLE_TRAIN",
    "B_FACTOR_SHUFFLE_TRAIN",
    "STATIC_00",
    "STATIC_01",
    "STATIC_10",
    "STATIC_11",
    "RANDOM_MATCHED_16",
)
FORBIDDEN_EDGE_SELECTOR_NAMES = {
    "answer",
    "branch",
    "branches",
    "decoder",
    "endpoint",
    "expected",
    "factor_name",
    "joint_context",
    "mapping",
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
    "weight",
}


@dataclass(frozen=True)
class BroadEdgeSelectorConfig:
    seed: int = 97801
    selected_edges: int = 4
    min_boundary_gap: float = 1e-6

    def __post_init__(self) -> None:
        if self.selected_edges != 4:
            raise ValueError("the frozen selector requires exactly four entry edges")
        if not math.isfinite(float(self.min_boundary_gap)) or self.min_boundary_gap <= 0.0:
            raise ValueError("boundary gap must be finite and positive")


@dataclass(frozen=True)
class EdgeFieldSnapshot:
    theta: torch.Tensor
    accumulator: torch.Tensor
    counts: torch.Tensor
    update_count: int
    selected_edges: int
    min_boundary_gap: float


class CountNormalizedEdgeField:
    """Count-normalized dimensionless field over a fixed edge list."""

    def __init__(self, edge_count: int, selected_edges: int, min_boundary_gap: float) -> None:
        self.edge_count = int(edge_count)
        self.selected_edges = int(selected_edges)
        self.min_boundary_gap = float(min_boundary_gap)
        if self.edge_count <= self.selected_edges or self.selected_edges <= 0:
            raise ValueError("invalid edge-field dimensions")
        if not math.isfinite(self.min_boundary_gap) or self.min_boundary_gap <= 0.0:
            raise ValueError("invalid boundary gap")
        self.accumulator = torch.zeros(self.edge_count, 2, dtype=torch.float64)
        self.counts = torch.zeros(2, dtype=torch.float64)
        self.theta = torch.zeros(self.edge_count, 2, dtype=torch.float64)
        self.update_count = 0

    def observe(self, factor_cue: torch.Tensor, edge_use: torch.Tensor) -> float:
        cue = torch.as_tensor(factor_cue, dtype=torch.float64).view(-1)
        use = torch.as_tensor(edge_use, dtype=torch.float64).view(-1)
        if cue.shape != (2,) or use.shape != (self.edge_count,):
            raise ApparatusInvalid("APPARATUS_INVALID: invalid edge-field observation shape")
        if not torch.isfinite(cue).all() or not torch.isfinite(use).all():
            raise ApparatusInvalid("APPARATUS_INVALID: nonfinite edge-field observation")
        if torch.any(use < 0.0):
            raise ApparatusInvalid("APPARATUS_INVALID: edge use must be nonnegative")
        if not (
            torch.all((cue == 0.0) | (cue == 1.0))
            and float(cue.sum().item()) == 1.0
        ):
            raise ApparatusInvalid("APPARATUS_INVALID: factor cue must be exactly one-hot")
        before = self.theta.clone()
        self.accumulator += torch.outer(use, cue)
        self.counts += cue
        seen = self.counts > 0.0
        self.theta[:, seen] = self.accumulator[:, seen] / self.counts[seen]
        self.update_count += 1
        return float((self.theta - before).norm().item())

    def snapshot(self) -> EdgeFieldSnapshot:
        return EdgeFieldSnapshot(
            theta=self.theta.detach().clone(),
            accumulator=self.accumulator.detach().clone(),
            counts=self.counts.detach().clone(),
            update_count=int(self.update_count),
            selected_edges=self.selected_edges,
            min_boundary_gap=self.min_boundary_gap,
        )


def _support_indices(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    packed = torch.as_tensor(mask, dtype=torch.bool)
    if packed.ndim != 2 or packed.shape[0] != packed.shape[1]:
        raise ApparatusInvalid("APPARATUS_INVALID: support must be square")
    return torch.where(packed)


def _validate_edge_field(
    snapshot: EdgeFieldSnapshot,
    candidate_support: torch.Tensor,
    trunk_support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate = torch.as_tensor(candidate_support, dtype=torch.bool)
    trunk = torch.as_tensor(trunk_support, dtype=torch.bool)
    if candidate.shape != trunk.shape or candidate.ndim != 2 or candidate.shape[0] != candidate.shape[1]:
        raise ApparatusInvalid("APPARATUS_INVALID: incompatible edge supports")
    if torch.any(candidate & trunk):
        raise ApparatusInvalid("APPARATUS_INVALID: candidate and trunk supports overlap")
    rows, cols = _support_indices(candidate)
    edge_count = int(rows.numel())
    theta = torch.as_tensor(snapshot.theta, dtype=torch.float64)
    accumulator = torch.as_tensor(snapshot.accumulator, dtype=torch.float64)
    counts = torch.as_tensor(snapshot.counts, dtype=torch.float64)
    if theta.shape != (edge_count, 2) or accumulator.shape != theta.shape or counts.shape != (2,):
        raise ApparatusInvalid("APPARATUS_INVALID: invalid frozen edge-field shape")
    if not torch.isfinite(theta).all() or not torch.isfinite(accumulator).all() or not torch.isfinite(counts).all():
        raise ApparatusInvalid("APPARATUS_INVALID: nonfinite frozen edge field")
    if torch.any(counts <= 0.0) or snapshot.update_count <= 0:
        raise ApparatusInvalid("APPARATUS_INVALID: edge field has an unobserved cue")
    if not torch.equal(theta, accumulator / counts.view(1, 2)):
        raise ApparatusInvalid("APPARATUS_INVALID: frozen edge field is not count-normalized")
    if snapshot.selected_edges <= 0 or snapshot.selected_edges >= edge_count:
        raise ApparatusInvalid("APPARATUS_INVALID: invalid frozen edge budget")
    if not math.isfinite(float(snapshot.min_boundary_gap)) or snapshot.min_boundary_gap <= 0.0:
        raise ApparatusInvalid("APPARATUS_INVALID: invalid frozen edge margin")
    return theta, rows, cols


def compile_edge_field_mask(
    gate_snapshot: EdgeFieldSnapshot,
    factor_cue: torch.Tensor,
    candidate_support: torch.Tensor,
    trunk_support: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select a local top-m edge set without recurrent-weight access."""
    theta, rows, cols = _validate_edge_field(gate_snapshot, candidate_support, trunk_support)
    cue = torch.as_tensor(factor_cue, dtype=torch.float64).view(-1)
    if cue.shape != (2,) or not torch.isfinite(cue).all():
        raise ApparatusInvalid("APPARATUS_INVALID: invalid factor cue")
    scores = theta @ cue
    order = torch.argsort(scores, descending=True, stable=True)
    boundary = float((scores[order[gate_snapshot.selected_edges - 1]]
                      - scores[order[gate_snapshot.selected_edges]]).item())
    if not math.isfinite(boundary) or boundary < gate_snapshot.min_boundary_gap:
        raise ApparatusInvalid("APPARATUS_INVALID: unresolved top-m boundary tie")
    chosen = order[:gate_snapshot.selected_edges]
    mask = torch.as_tensor(trunk_support, dtype=torch.bool).clone()
    mask[rows[chosen], cols[chosen]] = True
    return mask.to(torch.float32), {
        "selected_candidate_indices": [int(value) for value in chosen.tolist()],
        "boundary_gap": boundary,
        "entry_edges": int(chosen.numel()),
        "trunk_edges": int(torch.count_nonzero(torch.as_tensor(trunk_support)).item()),
        "mask_edges": int(mask.sum().item()),
    }


def _edge_field_hash(snapshot: EdgeFieldSnapshot) -> str:
    digest = hashlib.sha256()
    for tensor in (snapshot.theta, snapshot.accumulator, snapshot.counts):
        packed = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(packed.shape)).encode())
        digest.update(str(packed.dtype).encode())
        digest.update(packed.numpy().tobytes())
    digest.update(repr((snapshot.update_count, snapshot.selected_edges, snapshot.min_boundary_gap)).encode())
    return digest.hexdigest()


def _rectangle(dim: int, destination: Sequence[int], source: Sequence[int]) -> torch.Tensor:
    mask = torch.zeros(dim, dim, dtype=torch.bool)
    rows = torch.tensor(tuple(destination), dtype=torch.long)
    cols = torch.tensor(tuple(source), dtype=torch.long)
    mask[rows[:, None], cols] = True
    return mask


def _broad_supports(
    blocks: Sequence[Sequence[int]],
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed = tuple(tuple(int(value) for value in block) for block in blocks)
    if len(packed) != 5 or any(len(block) != 4 for block in packed):
        raise ApparatusInvalid("APPARATUS_INVALID: expected five width-four blocks")
    source = (*packed[0], *packed[1])
    hidden = packed[2]
    output = packed[4]
    return _rectangle(dim, hidden, source), _rectangle(dim, output, hidden)


def _shared_hidden_source(
    seed: int,
    base: ContextBranchConfig,
) -> tuple[Any, dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor]:
    inherited, books, inherited_receipt = _learn(int(seed), base)
    candidate, trunk_region = _broad_supports(books["blocks"], base.dim)
    inherited_trunk = (inherited.weight != 0.0) & trunk_region
    if int(inherited_trunk.sum().item()) != base.payload_width:
        raise ApparatusInvalid("APPARATUS_INVALID: inherited shared trunk is incomplete")
    runtime = BrainRuntime.from_snapshot(inherited, backend="torch", device="cpu")
    matrix = torch.zeros_like(runtime.weight)
    matrix[candidate] = 1.0
    matrix[inherited_trunk] = inherited.weight[inherited_trunk]
    runtime.weight = matrix
    runtime._rebuild_sparse()
    runtime.reset_evaluation_state()
    source = runtime.snapshot()
    outside = ~(candidate | inherited_trunk)
    trunk_matrix = _block_matrix(source.weight, books["blocks"][4], books["blocks"][2])
    receipt = {
        "candidate_edges": int(candidate.sum().item()),
        "candidate_all_nonzero": bool(torch.all(source.weight[candidate] != 0.0)),
        "candidate_all_equal_one": bool(torch.equal(
            source.weight[candidate], torch.ones(int(candidate.sum().item()))
        )),
        "candidate_unique_weight_values": sorted(set(float(value) for value in source.weight[candidate].tolist())),
        "trunk_edges": int(inherited_trunk.sum().item()),
        "trunk_rank": _matrix_rank(trunk_matrix),
        "outside_nonzero": int(torch.count_nonzero(source.weight[outside]).item()),
        "hippocampal_rows_after": len(runtime.hippocampus),
        "delay_ring_zero": bool(runtime._delay_buffer is not None and torch.count_nonzero(runtime._delay_buffer) == 0),
        "delay_index_after": int(runtime._delay_idx),
        "inherited_trunk_learning": inherited_receipt,
    }
    return source, books, receipt, candidate, inherited_trunk


def _factor_task(seed: int) -> dict[str, Any]:
    parity_A = int(seed) & 1
    parity_B = (int(seed) >> 1) & 1
    return {
        "cues": torch.eye(2, dtype=torch.float64),
        "mapping_A": (parity_A, 1 - parity_A),
        "mapping_B": (parity_B, 1 - parity_B),
        "parity_pair": (parity_A, parity_B),
        "source_seed_A": int(seed) + 3_000_017,
        "source_seed_B": int(seed) + 4_000_037,
    }


def _episode_edge_use(
    books: dict[str, Any],
    source_slot: int,
    payload_slot: int,
    candidate_support: torch.Tensor,
    base: ContextBranchConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    runtime = _runtime(base)
    tracker = ExactDelayEligibility(
        base.dim, base.delay_ticks, base.eligibility_decay, base.ltd,
    )
    source = books[f"S{int(source_slot)}"][int(payload_slot)]
    hidden = books["H0"][int(payload_slot)]
    for tick in range(base.delay_ticks + 1):
        external = torch.zeros(base.dim)
        if tick == 0:
            external = base.cue_drive_gain * source
        elif tick == base.delay_ticks:
            external = base.cue_drive_gain * hidden
        runtime.step(external_input=external, force_mode=RuntimeMode.WAKE, learning_signal=0.0)
        tracker.observe(runtime.activation)
    rows, cols = _support_indices(candidate_support)
    positive = tracker.eligibility.double().clamp_min(0.0)[rows, cols]
    total = float(positive.sum().item())
    if not math.isfinite(total) or total <= EPSILON:
        raise ApparatusInvalid("APPARATUS_INVALID: empty positive edge experience")
    normalized = positive / (EPSILON + total)
    peak = int(torch.argmax(normalized).item())
    sorted_values = torch.sort(normalized, descending=True).values
    return normalized, {
        "source_slot": int(source_slot),
        "payload_slot": int(payload_slot),
        "positive_sum": total,
        "normalized_sum": float(normalized.sum().item()),
        "peak_candidate_index": peak,
        "peak_value": float(sorted_values[0].item()),
        "runner_up_value": float(sorted_values[1].item()),
        "paired_observations": int(tracker.paired_observations),
        "target_pulse_count": 0,
        "decoder_read_count": 0,
        "endpoint_read_count": 0,
    }


def _collect_experience(
    source_A: Any,
    books_A: dict[str, Any],
    candidate_A: torch.Tensor,
    source_B: Any,
    books_B: dict[str, Any],
    candidate_B: torch.Tensor,
    task: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_A = ContextBranchConfig(seed=int(task["source_seed_A"]))
    base_B = ContextBranchConfig(seed=int(task["source_seed_B"]))
    hashes_before = (_snapshot_hash(source_A), _snapshot_hash(source_B))
    rows: list[dict[str, Any]] = []
    for a, b in TRAIN_CONTEXTS:
        source_slot_A = int(task["mapping_A"][a])
        source_slot_B = int(task["mapping_B"][b])
        for payload in range(base_A.payload_width):
            use_A, receipt_A = _episode_edge_use(
                books_A, source_slot_A, payload, candidate_A, base_A,
            )
            use_B, receipt_B = _episode_edge_use(
                books_B, source_slot_B, payload, candidate_B, base_B,
            )
            rows.append({
                "context": (int(a), int(b)),
                "payload_repetition": int(payload),
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


def _fit_edge_field(
    rows: list[dict[str, Any]],
    factor_slot: int,
    config: BroadEdgeSelectorConfig,
    *,
    shuffle: bool,
) -> tuple[EdgeFieldSnapshot, dict[str, Any]]:
    label = "A" if factor_slot == 0 else "B"
    field = CountNormalizedEdgeField(32, config.selected_edges, config.min_boundary_gap)
    update_rows: list[dict[str, Any]] = []
    for row in rows:
        cue_slot = int(row["context"][factor_slot])
        observed_slot = 1 - cue_slot if shuffle else cue_slot
        cue = torch.tensor((1.0, 0.0) if observed_slot == 0 else (0.0, 1.0), dtype=torch.float64)
        use = torch.tensor(row[f"use_{label}"], dtype=torch.float64)
        change = field.observe(cue, use)
        update_rows.append({
            "factor_cue_slot": cue_slot,
            "observed_cue_slot": observed_slot,
            "theta_change_norm": change,
        })
    frozen = field.snapshot()
    return frozen, {
        "shuffle": bool(shuffle),
        "update_count": frozen.update_count,
        "counts": [float(value) for value in frozen.counts.tolist()],
        "theta_norm": float(frozen.theta.norm().item()),
        "field_sha256": _edge_field_hash(frozen),
        "target_read_count": 0,
        "decoder_read_count": 0,
        "endpoint_read_count": 0,
        "rows": update_rows,
    }


def _function_identifiers(function: Any) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _matching_entry_mask(
    books: dict[str, Any],
    source_slot: int,
) -> torch.Tensor:
    mask = torch.zeros(20, 20, dtype=torch.bool)
    for payload in range(4):
        source_index = int(torch.argmax(books[f"S{int(source_slot)}"][payload]).item())
        hidden_index = int(torch.argmax(books["H0"][payload]).item())
        mask[hidden_index, source_index] = True
    return mask


def _pooled_boundary_gap(snapshot: EdgeFieldSnapshot) -> float:
    scores = 0.5 * (snapshot.theta[:, 0] + snapshot.theta[:, 1])
    order = torch.argsort(scores, descending=True, stable=True)
    return float((scores[order[snapshot.selected_edges - 1]]
                  - scores[order[snapshot.selected_edges]]).item())


def _joint_lookup_receipt(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accumulator = torch.zeros(64, 4, dtype=torch.float64)
    counts = torch.zeros(4, dtype=torch.float64)
    for row in rows:
        a, b = (int(value) for value in row["context"])
        column = 2 * a + b
        joined = torch.cat((torch.tensor(row["use_A"]), torch.tensor(row["use_B"]))).double()
        accumulator[:, column] += joined
        counts[column] += 1.0
    theta = torch.zeros_like(accumulator)
    seen = counts > 0.0
    theta[:, seen] = accumulator[:, seen] / counts[seen]
    return {
        "counts": [float(value) for value in counts.tolist()],
        "heldout_column_nonzero": int(torch.count_nonzero(theta[:, 3]).item()),
        "heldout_abstains": bool(counts[3] == 0.0 and torch.count_nonzero(theta[:, 3]) == 0),
        "endpoint_opened": False,
    }


def _preflight_factor(
    source: Any,
    books: dict[str, Any],
    source_receipt: dict[str, Any],
    candidate: torch.Tensor,
    trunk: torch.Tensor,
    normal: EdgeFieldSnapshot,
    shuffled: EdgeFieldSnapshot,
    cues: torch.Tensor,
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    learned = [compile_edge_field_mask(normal, cues[index], candidate, trunk) for index in (0, 1)]
    adverse = [compile_edge_field_mask(shuffled, cues[index], candidate, trunk) for index in (0, 1)]
    learned_masks = [row[0].bool() for row in learned]
    learned_info = [row[1] for row in learned]
    adverse_masks = [row[0].bool() for row in adverse]
    repeated = [compile_edge_field_mask(normal, cues[index], candidate, trunk)[0].bool()
                for index in (0, 1)]
    cue_swap = [compile_edge_field_mask(normal, cues[1 - index], candidate, trunk)[0].bool()
                for index in (0, 1)]
    row_swapped = EdgeFieldSnapshot(
        theta=normal.theta.flip(0).clone(),
        accumulator=normal.accumulator.flip(0).clone(),
        counts=normal.counts.clone(),
        update_count=normal.update_count,
        selected_edges=normal.selected_edges,
        min_boundary_gap=normal.min_boundary_gap,
    )
    counterfactual = [compile_edge_field_mask(row_swapped, cues[index], candidate, trunk)[0].bool()
                      for index in (0, 1)]
    original_entries = [mask & candidate for mask in learned_masks]
    counter_entries = [mask & candidate for mask in counterfactual]
    gates = {
        "source_candidate_count": source_receipt["candidate_edges"] == 32,
        "source_uniform_nonzero": bool(
            source_receipt["candidate_all_nonzero"]
            and source_receipt["candidate_all_equal_one"]
            and source_receipt["candidate_unique_weight_values"] == [1.0]
        ),
        "source_trunk": source_receipt["trunk_edges"] == 4 and source_receipt["trunk_rank"] == 4,
        "source_outside_zero": source_receipt["outside_nonzero"] == 0,
        "source_cutoff": bool(
            source_receipt["hippocampal_rows_after"] == 0
            and source_receipt["delay_ring_zero"]
            and source_receipt["delay_index_after"] == 0
        ),
        "positive_unequal_counts": tuple(normal.counts.tolist()) == (8.0, 4.0),
        "exact_count_normalization": bool(torch.equal(
            normal.theta, normal.accumulator / normal.counts.view(1, 2)
        )),
        "learned_budget": all(info["entry_edges"] == 4 and info["trunk_edges"] == 4
                              and info["mask_edges"] == 8 for info in learned_info),
        "learned_sets_distinct": int((original_entries[0] != original_entries[1]).sum().item()) == 8,
        "cue_swap_equivariance": all(torch.equal(cue_swap[index], learned_masks[1 - index])
                                     for index in (0, 1)),
        "shuffle_exchanges_sets": all(torch.equal(adverse_masks[index], learned_masks[1 - index])
                                      for index in (0, 1)),
        "metadata_invariance": all(torch.equal(repeated[index], learned_masks[index]) for index in (0, 1)),
        "theta_counterfactual_dependence": all(
            not torch.equal(counter_entries[index], original_entries[index]) for index in (0, 1)
        ),
        "strict_boundaries": min(info["boundary_gap"] for info in learned_info) >= normal.min_boundary_gap,
        "weight_only_abstains": len(set(source_receipt["candidate_unique_weight_values"])) == 1,
        "pooled_static_abstains": abs(_pooled_boundary_gap(normal)) <= 1e-12,
        "snapshot_finite": bool(torch.isfinite(source.weight).all()),
    }
    return {
        "all_pass": all(gates.values()),
        "gates": gates,
        "counts": tuple(float(value) for value in normal.counts.tolist()),
        "shuffled_counts": tuple(float(value) for value in shuffled.counts.tolist()),
        "boundary_gaps": tuple(float(info["boundary_gap"]) for info in learned_info),
        "pooled_boundary_gap": _pooled_boundary_gap(normal),
        "selected_candidate_indices": tuple(tuple(info["selected_candidate_indices"])
                                            for info in learned_info),
        "field_sha256": _edge_field_hash(normal),
        "shuffled_field_sha256": _edge_field_hash(shuffled),
        "source_snapshot_sha256": _snapshot_hash(source),
        "decoder_sha256": _codebook_hash(books),
    }, learned_masks


def _preflight(
    source_A: Any,
    books_A: dict[str, Any],
    source_receipt_A: dict[str, Any],
    candidate_A: torch.Tensor,
    trunk_A: torch.Tensor,
    source_B: Any,
    books_B: dict[str, Any],
    source_receipt_B: dict[str, Any],
    candidate_B: torch.Tensor,
    trunk_B: torch.Tensor,
    task: dict[str, Any],
    rows: list[dict[str, Any]],
    experience: dict[str, Any],
    normal_A: EdgeFieldSnapshot,
    normal_B: EdgeFieldSnapshot,
    shuffled_A: EdgeFieldSnapshot,
    shuffled_B: EdgeFieldSnapshot,
) -> dict[str, Any]:
    cues = torch.as_tensor(task["cues"], dtype=torch.float64)
    factor_A, masks_A = _preflight_factor(
        source_A, books_A, source_receipt_A, candidate_A, trunk_A,
        normal_A, shuffled_A, cues,
    )
    factor_B, masks_B = _preflight_factor(
        source_B, books_B, source_receipt_B, candidate_B, trunk_B,
        normal_B, shuffled_B, cues,
    )
    contexts = [tuple(int(value) for value in row["context"]) for row in rows]
    counts = {key: contexts.count(key) for key in (*TRAIN_CONTEXTS, HELDOUT_CONTEXT)}
    compiler_signature = tuple(inspect.signature(compile_edge_field_mask).parameters)
    observe_signature = tuple(inspect.signature(CountNormalizedEdgeField.observe).parameters)
    identifiers = _function_identifiers(compile_edge_field_mask) | _function_identifiers(
        CountNormalizedEdgeField.observe
    )
    pair_masks = {(a, b): torch.block_diag(masks_A[a], masks_B[b])
                  for a in (0, 1) for b in (0, 1)}
    local_peaks = all(
        row["factor_A"]["peak_value"] > row["factor_A"]["runner_up_value"] + 1e-6
        and row["factor_B"]["peak_value"] > row["factor_B"]["runner_up_value"] + 1e-6
        for row in rows
    )
    gates = {
        "factor_A_preflight": bool(factor_A["all_pass"]),
        "factor_B_preflight": bool(factor_B["all_pass"]),
        "exact_training_multiset": counts == {
            (0, 0): 4, (0, 1): 4, (1, 0): 4, (1, 1): 0,
        },
        "heldout_absent": HELDOUT_CONTEXT not in contexts,
        "factor_values_observed": all(any(row["context"][slot] == value for row in rows)
                                      for slot in (0, 1) for value in (0, 1)),
        "episode_local_peak": local_peaks,
        "no_target_decoder_endpoint_reads": bool(
            experience["target_pulse_count"] == 0
            and experience["decoder_read_count"] == 0
            and experience["endpoint_read_count"] == 0
        ),
        "sources_immutable": bool(experience["sources_immutable"]),
        "field_input_signature": bool(
            compiler_signature == (
                "gate_snapshot", "factor_cue", "candidate_support", "trunk_support",
            )
            and observe_signature == ("self", "factor_cue", "edge_use")
            and compile_edge_field_mask.__closure__ is None
            and not bool(identifiers & FORBIDDEN_EDGE_SELECTOR_NAMES)
        ),
        "pair_mask_budget": all(int(mask.sum().item()) == 16 for mask in pair_masks.values()),
        "pair_mask_hamming": bool(
            int((pair_masks[(0, 0)] != pair_masks[(1, 0)]).sum().item()) == 8
            and int((pair_masks[(0, 0)] != pair_masks[(0, 1)]).sum().item()) == 8
            and int((pair_masks[(0, 0)] != pair_masks[(1, 1)]).sum().item()) == 16
        ),
        "direct_sum_cross_support_zero": int(torch.count_nonzero(
            torch.block_diag(source_A.weight, source_B.weight)[:20, 20:]
        ).item()) == 0,
        "joint_lookup_holdout_abstains": bool(_joint_lookup_receipt(rows)["heldout_abstains"]),
    }
    return {
        "all_pass": all(gates.values()),
        "gates": gates,
        "factor_A": factor_A,
        "factor_B": factor_B,
        "training_context_counts": {"".join(map(str, key)): value for key, value in counts.items()},
        "compiler_signature": compiler_signature,
        "observe_signature": observe_signature,
        "compiler_identifiers": sorted(identifiers),
        "pair_mask_edge_counts": {str(key): int(mask.sum().item()) for key, mask in pair_masks.items()},
        "joint_lookup_holdout_abstain": _joint_lookup_receipt(rows),
    }


def _random_entry_mask(candidate: torch.Tensor, key: int, count: int = 4) -> torch.Tensor:
    rows, cols = _support_indices(candidate)
    order = torch.randperm(int(rows.numel()), generator=torch.Generator().manual_seed(int(key)))
    mask = torch.zeros_like(candidate, dtype=torch.bool)
    chosen = order[:count]
    mask[rows[chosen], cols[chosen]] = True
    return mask


def _factor_mask_for_route(
    route_name: str,
    factor_slot: int,
    source: Any,
    books: dict[str, Any],
    candidate: torch.Tensor,
    trunk: torch.Tensor,
    task: dict[str, Any],
    normal: EdgeFieldSnapshot,
    shuffled: EdgeFieldSnapshot,
) -> tuple[torch.Tensor, dict[str, Any]]:
    label = "A" if factor_slot == 0 else "B"
    cue = task["cues"][1]
    source_slot = int(task[f"mapping_{label}"][1])
    if route_name == "ORACLE":
        mask = trunk | _matching_entry_mask(books, source_slot)
        return mask.float(), {"selected_source_slot": source_slot, "boundary_gap": None}
    if route_name == "EDGE_FIELD_LEARNED":
        return compile_edge_field_mask(normal, cue, candidate, trunk)
    if route_name == "A_FACTOR_SHUFFLE_TRAIN" and factor_slot == 0:
        return compile_edge_field_mask(shuffled, cue, candidate, trunk)
    if route_name == "B_FACTOR_SHUFFLE_TRAIN" and factor_slot == 1:
        return compile_edge_field_mask(shuffled, cue, candidate, trunk)
    if route_name in {"A_FACTOR_SHUFFLE_TRAIN", "B_FACTOR_SHUFFLE_TRAIN"}:
        return compile_edge_field_mask(normal, cue, candidate, trunk)
    if route_name.startswith("STATIC_"):
        fixed = route_name.removeprefix("STATIC_")
        selected = int(fixed[factor_slot])
        mask = trunk | _matching_entry_mask(books, selected)
        return mask.float(), {"selected_source_slot": selected, "boundary_gap": None}
    if route_name == "RANDOM_MATCHED_16":
        key = int(task["source_seed_A" if factor_slot == 0 else "source_seed_B"]) + 71_113
        mask = trunk | _random_entry_mask(candidate, key)
        return mask.float(), {"selected_source_slot": -1, "boundary_gap": None}
    if route_name == "FULL_72":
        return (candidate | trunk).float(), {"selected_source_slot": -1, "boundary_gap": None}
    raise ValueError(f"unknown route {route_name!r}")


def _evaluate_factor(
    route_name: str,
    factor_slot: int,
    source: Any,
    books: dict[str, Any],
    candidate: torch.Tensor,
    trunk: torch.Tensor,
    task: dict[str, Any],
    normal: EdgeFieldSnapshot,
    shuffled: EdgeFieldSnapshot,
) -> dict[str, Any]:
    label = "A" if factor_slot == 0 else "B"
    base = ContextBranchConfig(seed=int(task[f"source_seed_{label}"]))
    expected_source = int(task[f"mapping_{label}"][1])
    mask, selector_info = _factor_mask_for_route(
        route_name, factor_slot, source, books, candidate, trunk, task, normal, shuffled,
    )
    trial_rows: list[dict[str, Any]] = []
    for left in range(base.payload_width):
        for right in range(base.payload_width):
            if left == right:
                continue
            expected = left if expected_source == 0 else right
            opposite = right if expected_source == 0 else left
            sensory = books["S0"][left] + books["S1"][right]
            final, metrics = _rollout(source, mask, mask, sensory, base, books["blocks"])
            metrics.pop("hidden_norms_at_arrival")
            decoded = _decode_y(final, books, expected, opposite, base)
            trial_rows.append({
                "left_payload": left,
                "right_payload": right,
                "prediction": decoded["prediction"],
                "success": decoded["success"],
                "opposite_delivery": decoded["opposite_delivery"],
                "runtime_energy_proxy": metrics["runtime_energy_proxy"],
                "active_fraction": metrics["active_fraction"],
                "hippocampal_rows_after": metrics["hippocampal_rows_after"],
            })
    count = len(trial_rows)
    return {
        "factor": label,
        "accuracy": sum(int(row["success"]) for row in trial_rows) / count,
        "opposite_delivery": sum(int(row["opposite_delivery"]) for row in trial_rows) / count,
        "mean_runtime_energy_proxy": sum(float(row["runtime_energy_proxy"]) for row in trial_rows) / count,
        "mean_active_fraction": sum(float(row["active_fraction"]) for row in trial_rows) / count,
        "mask_edges": int(mask.sum().item()),
        "expected_source_slot": expected_source,
        "selector_info": selector_info,
        "hippocampal_rows_after": max(int(row["hippocampal_rows_after"]) for row in trial_rows),
        "trials": trial_rows,
    }


def _evaluate_pair_route(
    route_name: str,
    source_A: Any,
    books_A: dict[str, Any],
    candidate_A: torch.Tensor,
    trunk_A: torch.Tensor,
    source_B: Any,
    books_B: dict[str, Any],
    candidate_B: torch.Tensor,
    trunk_B: torch.Tensor,
    task: dict[str, Any],
    normal_A: EdgeFieldSnapshot,
    normal_B: EdgeFieldSnapshot,
    shuffled_A: EdgeFieldSnapshot,
    shuffled_B: EdgeFieldSnapshot,
) -> dict[str, Any]:
    factor_A = _evaluate_factor(
        route_name, 0, source_A, books_A, candidate_A, trunk_A, task, normal_A, shuffled_A,
    )
    factor_B = _evaluate_factor(
        route_name, 1, source_B, books_B, candidate_B, trunk_B, task, normal_B, shuffled_B,
    )
    outcomes = bytes(
        int(row_A["success"] and row_B["success"])
        for row_A in factor_A["trials"] for row_B in factor_B["trials"]
    )
    receipts_A = tuple((row["left_payload"], row["right_payload"], row["prediction"],
                        row["success"], row["opposite_delivery"]) for row in factor_A["trials"])
    receipts_B = tuple((row["left_payload"], row["right_payload"], row["prediction"],
                        row["success"], row["opposite_delivery"]) for row in factor_B["trials"])
    summary_A = {key: value for key, value in factor_A.items() if key != "trials"}
    summary_B = {key: value for key, value in factor_B.items() if key != "trials"}
    return {
        "route": route_name,
        "joint_accuracy": sum(outcomes) / len(outcomes),
        "A_accuracy": factor_A["accuracy"],
        "B_accuracy": factor_B["accuracy"],
        "A_opposite_delivery": factor_A["opposite_delivery"],
        "B_opposite_delivery": factor_B["opposite_delivery"],
        "mask_edges": factor_A["mask_edges"] + factor_B["mask_edges"],
        "cartesian_trial_count": len(outcomes),
        "cartesian_success_count": sum(outcomes),
        "cartesian_conjunction_sha256": hashlib.sha256(outcomes).hexdigest(),
        "factor_A_trials_sha256": hashlib.sha256(repr(receipts_A).encode()).hexdigest(),
        "factor_B_trials_sha256": hashlib.sha256(repr(receipts_B).encode()).hexdigest(),
        "factor_A": summary_A,
        "factor_B": summary_B,
    }


def run_broad_edge_selector_seed(
    seed: int = 97801,
    *,
    config: BroadEdgeSelectorConfig | None = None,
) -> dict[str, Any]:
    selected = config or BroadEdgeSelectorConfig(seed=int(seed))
    config = BroadEdgeSelectorConfig(**{**asdict(selected), "seed": int(seed)})
    task = _factor_task(int(seed))
    base_A = ContextBranchConfig(seed=int(task["source_seed_A"]))
    base_B = ContextBranchConfig(seed=int(task["source_seed_B"]))
    source_A, books_A, source_receipt_A, candidate_A, trunk_A = _shared_hidden_source(
        int(task["source_seed_A"]), base_A,
    )
    source_B, books_B, source_receipt_B, candidate_B, trunk_B = _shared_hidden_source(
        int(task["source_seed_B"]), base_B,
    )
    rows, experience = _collect_experience(
        source_A, books_A, candidate_A, source_B, books_B, candidate_B, task,
    )
    normal_A, receipt_A = _fit_edge_field(rows, 0, config, shuffle=False)
    normal_B, receipt_B = _fit_edge_field(rows, 1, config, shuffle=False)
    shuffled_A, shuffled_receipt_A = _fit_edge_field(rows, 0, config, shuffle=True)
    shuffled_B, shuffled_receipt_B = _fit_edge_field(rows, 1, config, shuffle=True)
    preflight = _preflight(
        source_A, books_A, source_receipt_A, candidate_A, trunk_A,
        source_B, books_B, source_receipt_B, candidate_B, trunk_B,
        task, rows, experience, normal_A, normal_B, shuffled_A, shuffled_B,
    )
    task_receipt = {
        "mapping_A": task["mapping_A"],
        "mapping_B": task["mapping_B"],
        "parity_pair": task["parity_pair"],
        "source_seed_A": task["source_seed_A"],
        "source_seed_B": task["source_seed_B"],
    }
    field_receipts = {
        "A": receipt_A,
        "B": receipt_B,
        "A_shuffled": shuffled_receipt_A,
        "B_shuffled": shuffled_receipt_B,
    }
    if not preflight["all_pass"]:
        return {
            "seed": int(seed),
            "status": "APPARATUS_INVALID",
            "endpoint_opened": False,
            "config": asdict(config),
            "task": task_receipt,
            "preflight": preflight,
            "experience": experience,
            "edge_field_learning": field_receipts,
        }

    sources_before = (_snapshot_hash(source_A), _snapshot_hash(source_B))
    fields_before = tuple(_edge_field_hash(value) for value in (
        normal_A, normal_B, shuffled_A, shuffled_B,
    ))
    routes = {
        route: _evaluate_pair_route(
            route,
            source_A, books_A, candidate_A, trunk_A,
            source_B, books_B, candidate_B, trunk_B,
            task, normal_A, normal_B, shuffled_A, shuffled_B,
        )
        for route in ROUTES
    }
    sources_after = (_snapshot_hash(source_A), _snapshot_hash(source_B))
    fields_after = tuple(_edge_field_hash(value) for value in (
        normal_A, normal_B, shuffled_A, shuffled_B,
    ))
    frozen_after = sources_before == sources_after and fields_before == fields_after
    stores_zero = all(
        route["factor_A"]["hippocampal_rows_after"] == 0
        and route["factor_B"]["hippocampal_rows_after"] == 0
        for route in routes.values()
    )
    learned = routes["EDGE_FIELD_LEARNED"]["joint_accuracy"]
    oracle = routes["ORACLE"]["joint_accuracy"]
    adverse_A = routes["A_FACTOR_SHUFFLE_TRAIN"]
    adverse_B = routes["B_FACTOR_SHUFFLE_TRAIN"]
    seed_pass = bool(
        learned >= 0.95
        and oracle >= 0.95
        and oracle - learned <= 0.05
        and adverse_A["joint_accuracy"] <= 0.05
        and adverse_A["A_opposite_delivery"] >= 0.95
        and adverse_A["B_accuracy"] >= 0.95
        and adverse_B["joint_accuracy"] <= 0.05
        and adverse_B["B_opposite_delivery"] >= 0.95
        and adverse_B["A_accuracy"] >= 0.95
        and frozen_after
        and stores_zero
    )
    return {
        "seed": int(seed),
        "status": "BROAD_EDGE_SELECTOR_PASS" if seed_pass else "BROAD_EDGE_SELECTOR_NOT_IDENTIFIED",
        "endpoint_opened": True,
        "heldout_context": HELDOUT_CONTEXT,
        "config": asdict(config),
        "task": task_receipt,
        "preflight": preflight,
        "experience": experience,
        "edge_field_learning": field_receipts,
        "routes": routes,
        "learned_oracle_gap": oracle - learned,
        "all_frozen_after_evaluation": frozen_after,
        "stores_zero_after_evaluation": stores_zero,
        "source_snapshot_sha256_before_evaluation": sources_before,
        "source_snapshot_sha256_after_evaluation": sources_after,
        "edge_field_sha256_before_evaluation": fields_before,
        "edge_field_sha256_after_evaluation": fields_after,
    }
