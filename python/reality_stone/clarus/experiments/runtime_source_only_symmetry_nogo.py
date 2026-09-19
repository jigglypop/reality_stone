"""Source-only symmetry no-go for the uniform BA-TR6 substrate."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch

from ..runtime import BrainRuntime, RuntimeMode
from .runtime_context_branch_routing import (
    EPSILON,
    ApparatusInvalid,
    ContextBranchConfig,
    ExactDelayEligibility,
    _snapshot_hash,
)
from .runtime_broad_edge_selector import (
    CountNormalizedEdgeField,
    _edge_field_hash,
    _factor_task,
    _shared_hidden_source,
    _support_indices,
    compile_edge_field_mask,
)


TRAIN_CONTEXTS = ((0, 0), (0, 1), (1, 0))
HELDOUT_CONTEXT = (1, 1)


@dataclass(frozen=True)
class SourceOnlySymmetryConfig:
    seed: int = 97901
    equality_tolerance: float = 1e-12
    min_positive: float = 1e-8

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(value)) for value in (
            self.equality_tolerance, self.min_positive,
        )):
            raise ValueError("tolerances must be finite")
        if self.equality_tolerance < 0.0 or self.min_positive <= 0.0:
            raise ValueError("invalid symmetry tolerances")


def _reverse_hidden_thresholds(runtime: BrainRuntime, hidden: tuple[int, ...]) -> None:
    for name in (
        "neuronwise_active_threshold",
        "neuronwise_bit_lower_threshold",
        "neuronwise_bit_upper_threshold",
    ):
        values = getattr(runtime.config, name)
        if values is None:
            continue
        packed = list(values)
        reversed_values = [packed[index] for index in reversed(hidden)]
        for index, value in zip(hidden, reversed_values):
            packed[index] = value
        setattr(runtime.config, name, tuple(packed))


def _source_only_episode(
    source_snapshot: Any,
    books: dict[str, Any],
    source_slot: int,
    payload_slot: int,
    candidate_support: torch.Tensor,
    base: ContextBranchConfig,
    *,
    reverse_hidden_thresholds: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    runtime = BrainRuntime.from_snapshot(source_snapshot, backend="torch", device="cpu")
    runtime.reset_evaluation_state()
    hidden = tuple(int(value) for value in books["blocks"][2])
    if reverse_hidden_thresholds:
        _reverse_hidden_thresholds(runtime, hidden)
    tracker = ExactDelayEligibility(
        base.dim, base.delay_ticks, base.eligibility_decay, base.ltd,
    )
    hidden_history: list[torch.Tensor] = []
    bit_history: list[torch.Tensor] = []
    for tick in range(base.delay_ticks + 2):
        external = (
            base.cue_drive_gain * books[f"S{int(source_slot)}"][int(payload_slot)]
            if tick == 0 else torch.zeros(base.dim)
        )
        runtime.step(external_input=external, force_mode=RuntimeMode.WAKE, learning_signal=0.0)
        tracker.observe(runtime.activation)
        hidden_history.append(runtime.activation[torch.tensor(hidden)].detach().clone())
        bit_history.append(runtime.bitfield[torch.tensor(hidden)].detach().clone())
    rows, cols = _support_indices(candidate_support)
    positive = tracker.eligibility.double().clamp_min(0.0)[rows, cols]
    positive_sum = float(positive.sum().item())
    if not math.isfinite(positive_sum) or positive_sum <= EPSILON:
        raise ApparatusInvalid("APPARATUS_INVALID: no source-only eligibility at true arrival")
    normalized = positive / (EPSILON + positive_sum)
    arrival = hidden_history[-1]
    prearrival = torch.stack(hidden_history[:-1])
    active_values = normalized[normalized > EPSILON]
    return normalized, {
        "source_slot": int(source_slot),
        "payload_slot": int(payload_slot),
        "pulse_ticks": [0],
        "observed_ticks": list(range(base.delay_ticks + 2)),
        "prearrival_hidden_max_abs": float(prearrival.abs().max().item()),
        "arrival_hidden": [float(value) for value in arrival.tolist()],
        "arrival_hidden_range": float((arrival.max() - arrival.min()).abs().item()),
        "arrival_hidden_min": float(arrival.min().item()),
        "arrival_bitfield": [int(value) for value in bit_history[-1].tolist()],
        "positive_candidate_count": int(active_values.numel()),
        "positive_candidate_range": float(
            (active_values.max() - active_values.min()).abs().item()
        ),
        "positive_sum": positive_sum,
        "normalized_sum": float(normalized.sum().item()),
        "target_pulse_count": 0,
        "hidden_pulse_count": 0,
        "output_pulse_count": 0,
        "decoder_read_count": 0,
        "endpoint_read_count": 0,
    }


def _fit_source_only_field(
    rows: list[dict[str, Any]],
    factor_slot: int,
) -> Any:
    label = "A" if factor_slot == 0 else "B"
    field = CountNormalizedEdgeField(32, 4, 1e-6)
    for row in rows:
        cue_slot = int(row["context"][factor_slot])
        cue = torch.tensor((1.0, 0.0) if cue_slot == 0 else (0.0, 1.0), dtype=torch.float64)
        field.observe(cue, torch.tensor(row[f"use_{label}"], dtype=torch.float64))
    return field.snapshot()


def _field_symmetry_receipt(
    field: Any,
    candidate: torch.Tensor,
    trunk: torch.Tensor,
    cues: torch.Tensor,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_abstain = True
    for cue_slot in (0, 1):
        scores = field.theta @ cues[cue_slot]
        order = torch.argsort(scores, descending=True, stable=True)
        gap = float((scores[order[3]] - scores[order[4]]).item())
        positive = scores[scores > EPSILON]
        try:
            compile_edge_field_mask(field, cues[cue_slot], candidate, trunk)
            abstains = False
        except ApparatusInvalid as error:
            abstains = "boundary tie" in str(error)
        all_abstain = all_abstain and abstains
        rows.append({
            "cue_slot": cue_slot,
            "positive_support_count": int(positive.numel()),
            "positive_score_range": float((positive.max() - positive.min()).abs().item()),
            "top4_boundary_gap": gap,
            "compiler_abstains": abstains,
        })
    return {
        "rows": rows,
        "all_compiler_abstain": all_abstain,
        "counts": [float(value) for value in field.counts.tolist()],
        "field_sha256": _edge_field_hash(field),
    }


def run_source_only_symmetry_seed(
    seed: int = 97901,
    *,
    config: SourceOnlySymmetryConfig | None = None,
) -> dict[str, Any]:
    selected = config or SourceOnlySymmetryConfig(seed=int(seed))
    config = SourceOnlySymmetryConfig(**{**asdict(selected), "seed": int(seed)})
    task = _factor_task(int(seed))
    base_A = ContextBranchConfig(seed=int(task["source_seed_A"]))
    base_B = ContextBranchConfig(seed=int(task["source_seed_B"]))
    source_A, books_A, source_receipt_A, candidate_A, trunk_A = _shared_hidden_source(
        int(task["source_seed_A"]), base_A,
    )
    source_B, books_B, source_receipt_B, candidate_B, trunk_B = _shared_hidden_source(
        int(task["source_seed_B"]), base_B,
    )
    source_hashes_before = (_snapshot_hash(source_A), _snapshot_hash(source_B))
    rows: list[dict[str, Any]] = []
    for a, b in TRAIN_CONTEXTS:
        source_slot_A = int(task["mapping_A"][a])
        source_slot_B = int(task["mapping_B"][b])
        for payload in range(base_A.payload_width):
            use_A, receipt_A = _source_only_episode(
                source_A, books_A, source_slot_A, payload, candidate_A, base_A,
            )
            use_B, receipt_B = _source_only_episode(
                source_B, books_B, source_slot_B, payload, candidate_B, base_B,
            )
            rows.append({
                "context": (int(a), int(b)),
                "payload_repetition": int(payload),
                "use_A": [float(value) for value in use_A.tolist()],
                "use_B": [float(value) for value in use_B.tolist()],
                "factor_A": receipt_A,
                "factor_B": receipt_B,
            })
    field_A = _fit_source_only_field(rows, 0)
    field_B = _fit_source_only_field(rows, 1)
    cues = torch.as_tensor(task["cues"], dtype=torch.float64)
    symmetry_A = _field_symmetry_receipt(field_A, candidate_A, trunk_A, cues)
    symmetry_B = _field_symmetry_receipt(field_B, candidate_B, trunk_B, cues)

    probe_A, threshold_A = _source_only_episode(
        source_A, books_A, int(task["mapping_A"][0]), 0, candidate_A, base_A,
        reverse_hidden_thresholds=True,
    )
    reference_A = next(row["factor_A"] for row in rows if row["context"] == (0, 0)
                       and row["payload_repetition"] == 0)
    source_hashes_after = (_snapshot_hash(source_A), _snapshot_hash(source_B))
    contexts = [tuple(int(value) for value in row["context"]) for row in rows]
    no_reads = all(
        factor["target_pulse_count"] == 0
        and factor["hidden_pulse_count"] == 0
        and factor["output_pulse_count"] == 0
        and factor["decoder_read_count"] == 0
        and factor["endpoint_read_count"] == 0
        for row in rows for factor in (row["factor_A"], row["factor_B"])
    )
    episode_receipts = [factor for row in rows for factor in (row["factor_A"], row["factor_B"])]
    field_rows = (*symmetry_A["rows"], *symmetry_B["rows"])
    gates = {
        "uniform_source_apparatus": bool(
            source_receipt_A["candidate_all_equal_one"]
            and source_receipt_B["candidate_all_equal_one"]
        ),
        "exact_training_multiset": all(contexts.count(value) == 4 for value in TRAIN_CONTEXTS)
            and HELDOUT_CONTEXT not in contexts,
        "zero_through_tick_L": max(value["prearrival_hidden_max_abs"] for value in episode_receipts)
            <= config.equality_tolerance,
        "nonzero_at_tick_L_plus_1": min(value["arrival_hidden_min"] for value in episode_receipts)
            > config.min_positive,
        "hidden_row_symmetry": max(value["arrival_hidden_range"] for value in episode_receipts)
            <= config.equality_tolerance,
        "per_payload_four_equal_edges": all(
            value["positive_candidate_count"] == 4
            and value["positive_candidate_range"] <= config.equality_tolerance
            for value in episode_receipts
        ),
        "cue_field_has_sixteen_tied_edges": all(
            value["positive_support_count"] == 16
            and value["positive_score_range"] <= config.equality_tolerance
            for value in field_rows
        ),
        "top4_boundary_tie": all(abs(value["top4_boundary_gap"]) <= config.equality_tolerance
                                 for value in field_rows),
        "compiler_abstains": bool(
            symmetry_A["all_compiler_abstain"] and symmetry_B["all_compiler_abstain"]
        ),
        "count_normalization": symmetry_A["counts"] == [8.0, 4.0]
            and symmetry_B["counts"] == [8.0, 4.0],
        "threshold_permutation_first_arrival_invariant": bool(
            threshold_A["arrival_hidden"] == reference_A["arrival_hidden"]
            and torch.equal(probe_A, torch.tensor(rows[0]["use_A"], dtype=torch.float64))
        ),
        "no_hidden_target_decoder_endpoint_reads": no_reads,
        "sources_immutable": source_hashes_before == source_hashes_after,
    }
    no_go = all(gates.values())
    return {
        "seed": int(seed),
        "status": "SOURCE_ONLY_SYMMETRY_NO_GO" if no_go else "APPARATUS_INVALID",
        "endpoint_opened": False,
        "config": asdict(config),
        "task": {
            "mapping_A": task["mapping_A"],
            "mapping_B": task["mapping_B"],
            "parity_pair": task["parity_pair"],
        },
        "gates": gates,
        "factor_A_symmetry": symmetry_A,
        "factor_B_symmetry": symmetry_B,
        "representative_arrival": reference_A,
        "threshold_permuted_arrival": threshold_A,
        "source_snapshot_sha256_before": source_hashes_before,
        "source_snapshot_sha256_after": source_hashes_after,
        "rows": rows,
    }
