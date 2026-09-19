"""BA-TR25: rank-two mixed-cue transfer through current packet content."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Sequence

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


CALIBRATION_SEEDS = (112001,)
DEVELOPMENT_SEEDS = tuple(range(112101, 112117))
TRAINING_ROWS = (0, 1, 2)
HELDOUT_ROW = 3
CUE_DIMENSION = 6
MIN_AXIS_SINGULAR_VALUE = 1e-8
MIN_BINDING_MARGIN = 1e-8


@dataclass(frozen=True)
class MixedCueContentGateSnapshot:
    cue_anchor: torch.Tensor
    content_anchor: torch.Tensor
    cue_axes: torch.Tensor
    content_axes: torch.Tensor
    gram_inverse: torch.Tensor
    cue_rank: int
    content_rank: int
    training_count: int


def _operational_rank(matrix: torch.Tensor) -> tuple[int, torch.Tensor]:
    singular = torch.linalg.svdvals(torch.as_tensor(matrix, dtype=torch.float64))
    rank = int(torch.count_nonzero(singular > MIN_AXIS_SINGULAR_VALUE).item())
    return rank, singular


def train_mixed_cue_content_gate(
    raw_cues: torch.Tensor,
    observed_content_sums: torch.Tensor,
) -> MixedCueContentGateSnapshot:
    """Fit a generic affine rank-two relation from three unlabeled episodes."""
    cues = torch.as_tensor(raw_cues, dtype=torch.float64)
    contents = torch.as_tensor(observed_content_sums, dtype=torch.float64)
    if cues.ndim != 2 or contents.ndim != 2 or cues.shape[0] != 3:
        raise ValueError("mixed-cue training requires three cue/content rows")
    if contents.shape[0] != 3 or cues.shape[1] < 2 or contents.shape[1] < 2:
        raise ValueError("mixed-cue training arrays have the wrong shape")
    if not torch.isfinite(cues).all() or not torch.isfinite(contents).all():
        raise ValueError("mixed-cue training arrays must be finite")
    cue_axes = torch.stack((cues[1] - cues[0], cues[2] - cues[0]), dim=1)
    content_axes = torch.stack(
        (contents[1] - contents[0], contents[2] - contents[0]), dim=1
    )
    cue_rank, _ = _operational_rank(cue_axes)
    content_rank, _ = _operational_rank(content_axes)
    if cue_rank != 2 or content_rank != 2:
        raise RuntimeError("mixed-cue/content contrasts are not rank two")
    gram = cue_axes.T @ cue_axes
    return MixedCueContentGateSnapshot(
        cue_anchor=cues[0].clone(),
        content_anchor=contents[0].clone(),
        cue_axes=cue_axes,
        content_axes=content_axes,
        gram_inverse=torch.linalg.inv(gram),
        cue_rank=cue_rank,
        content_rank=content_rank,
        training_count=3,
    )


def predict_mixed_cue_content(
    gate: MixedCueContentGateSnapshot,
    raw_cue: torch.Tensor,
) -> torch.Tensor:
    cue = torch.as_tensor(raw_cue, dtype=torch.float64).view(-1)
    if cue.shape != gate.cue_anchor.shape or not torch.isfinite(cue).all():
        raise ValueError("mixed cue has the wrong shape or a nonfinite value")
    alpha = gate.gram_inverse @ gate.cue_axes.T @ (cue - gate.cue_anchor)
    return gate.content_anchor + gate.content_axes @ alpha


def _content_binding_receipt(
    gate: MixedCueContentGateSnapshot,
    raw_cue: torch.Tensor,
    arrived_packet_indices: Sequence[int],
    weight: torch.Tensor,
    response_indices: Sequence[int],
) -> dict[str, Any]:
    packet_indices = tuple(int(value) for value in arrived_packet_indices)
    response = tuple(int(value) for value in response_indices)
    packed = torch.as_tensor(weight, dtype=torch.float64)
    if packed.ndim != 2 or packed.shape[0] != packed.shape[1]:
        raise ValueError("content binding requires a square weight matrix")
    if len(packet_indices) < 3 or len(set(packet_indices)) != len(packet_indices):
        raise ValueError("content binding requires at least three distinct packets")
    if not response or len(set(response)) != len(response):
        raise ValueError("response coordinates must be distinct")
    if min(packet_indices + response) < 0 or max(packet_indices + response) >= packed.shape[0]:
        raise ValueError("content binding coordinate is out of range")
    rows = torch.tensor(response, dtype=torch.long)
    columns = torch.tensor(packet_indices, dtype=torch.long)
    descriptors = packed.index_select(0, rows).index_select(1, columns).T
    norms = descriptors.norm(dim=1, keepdim=True)
    if torch.any(norms <= 0.0) or not torch.isfinite(descriptors).all():
        raise RuntimeError("arrived packet content is zero or nonfinite")
    descriptors = descriptors / norms
    prediction = predict_mixed_cue_content(gate, raw_cue)
    if prediction.numel() != descriptors.shape[1]:
        raise ValueError("predicted and arrived packet content dimensions differ")
    candidates: list[tuple[float, tuple[int, int]]] = []
    for left, right in itertools.combinations(range(len(packet_indices)), 2):
        residual = float(
            torch.linalg.vector_norm(
                prediction - descriptors[left] - descriptors[right]
            ).item()
        )
        candidates.append((residual, (left, right)))
    candidates.sort(key=lambda item: (item[0], item[1]))
    margin = candidates[1][0] - candidates[0][0]
    if not margin > MIN_BINDING_MARGIN:
        raise RuntimeError("packet-content pair binding is tied")
    best = candidates[0][1]
    return {
        "selected_indices": [packet_indices[best[0]], packet_indices[best[1]]],
        "best_residual": candidates[0][0],
        "second_residual": candidates[1][0],
        "binding_margin": margin,
        "predicted_content": [float(value) for value in prediction.tolist()],
    }


def compile_arrived_packet_indices(
    gate: MixedCueContentGateSnapshot,
    raw_cue: torch.Tensor,
    arrived_packet_indices: Sequence[int],
    weight: torch.Tensor,
    response_indices: Sequence[int],
) -> tuple[int, ...]:
    """Bind a predicted content sum to the current, possibly remapped packets."""
    receipt = _content_binding_receipt(
        gate,
        raw_cue,
        arrived_packet_indices,
        weight,
        response_indices,
    )
    return tuple(int(value) for value in receipt["selected_indices"])


def _gate_hash(gate: MixedCueContentGateSnapshot) -> str:
    digest = hashlib.sha256()
    for tensor in (
        gate.cue_anchor,
        gate.content_anchor,
        gate.cue_axes,
        gate.content_axes,
        gate.gram_inverse,
    ):
        digest.update(tensor.detach().cpu().numpy().tobytes())
    digest.update(
        repr((gate.cue_rank, gate.content_rank, gate.training_count)).encode("ascii")
    )
    return digest.hexdigest()


def _mixed_cues(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 260_003)
    chart, _ = torch.linalg.qr(
        torch.randn(CUE_DIMENSION, 3, generator=generator, dtype=torch.float64)
    )
    anchor = 1.25 * chart[:, 0]
    first = chart[:, 1]
    second = chart[:, 2]
    return torch.stack(
        (anchor, anchor + first, anchor + second, anchor + first + second)
    )


def _role_content(B: torch.Tensor) -> torch.Tensor:
    packed = torch.as_tensor(B, dtype=torch.float64)
    if packed.shape != (4, 4) or not torch.isfinite(packed).all():
        raise ValueError("packet content source must be finite 4x4")
    columns = packed.T.contiguous()
    norms = columns.norm(dim=1, keepdim=True)
    if torch.any(norms <= 0.0):
        raise RuntimeError("packet content source has a zero column")
    return columns / norms


def _observed_content_sums(content: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [content[torch.tensor(pair)].sum(dim=0) for pair in COMPOSITION_PAIRS]
    )


def _heldout_coordinate_map(seed: int) -> tuple[int, ...]:
    remap_block = tuple(int(value) for value in architectural_blocks(20)[1])
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 270_019)
    order = torch.randperm(4, generator=generator).tolist()
    return tuple(remap_block[index] for index in order)


def _snapshot_with_role_columns(snapshot: Any, role_coordinates: Sequence[int]) -> Any:
    coordinates = tuple(int(value) for value in role_coordinates)
    if len(coordinates) != 4 or len(set(coordinates)) != 4:
        raise ValueError("role coordinate map must contain four distinct entries")
    source, hidden, _target = _blocks()
    pool = tuple(source) + tuple(int(value) for value in architectural_blocks(20)[1])
    packed = snapshot.weight.detach().clone()
    hidden_idx = torch.tensor(hidden, dtype=torch.long)
    source_idx = torch.tensor(source, dtype=torch.long)
    pool_idx = torch.tensor(pool, dtype=torch.long)
    original = packed[hidden_idx[:, None], source_idx].clone()
    packed[hidden_idx[:, None], pool_idx] = 0.0
    for role, coordinate in enumerate(coordinates):
        packed[hidden_idx, coordinate] = original[:, role]
    return replace(snapshot, weight=packed)


def _target_set(values: torch.Tensor) -> tuple[int, ...]:
    packed = torch.as_tensor(values, dtype=torch.float64).view(4)
    return tuple(
        int(value)
        for value in torch.nonzero(
            packed >= MIN_DECODE_ACTIVATION, as_tuple=False
        ).view(-1)
    )


def _three_packet_probe(
    snapshot: Any,
    role_coordinates: Sequence[int],
    relevant_roles: Sequence[int],
    distractor_role: int,
    selected_indices: tuple[int, ...] | None,
) -> dict[str, Any]:
    coordinates = tuple(int(value) for value in role_coordinates)
    relevant = tuple(int(value) for value in relevant_roles)
    if len(relevant) != 2 or len(set(relevant + (int(distractor_role),))) != 3:
        raise ValueError("probe requires two relevant roles and one distractor")
    routed = _snapshot_with_role_columns(snapshot, coordinates)
    if selected_indices is None:
        routed = _all_input_snapshot(routed)
    else:
        config = replace(
            routed.config,
            competition_input_indices=tuple(int(value) for value in selected_indices),
            competition_k_from_delayed_input=False,
            competition_factorize_delayed_input=True,
        )
        routed = replace(routed, config=config)
    runtime = BrainRuntime.from_snapshot(routed, backend="torch", device="cpu")
    source, hidden, target = _blocks()
    pool = tuple(source) + tuple(int(value) for value in architectural_blocks(20)[1])
    pool_idx = torch.tensor(pool, dtype=torch.long)
    hidden_idx = torch.tensor(hidden, dtype=torch.long)
    target_idx = torch.tensor(target, dtype=torch.long)
    event_roles = relevant + (int(distractor_role),)
    event_coordinates = tuple(coordinates[role] for role in event_roles)
    packet_counts: list[int] = []
    written_counts: list[int] = []
    hidden_first = torch.zeros(4)
    target_final = torch.zeros(4)
    for tick in range(PAIR_TICKS):
        if runtime._delay_buffer is None:
            raise RuntimeError("mixed-cue transfer requires the delay ring")
        ring_slot = runtime._delay_idx % runtime.config.max_axon_delay
        packet_counts.append(
            int(
                torch.count_nonzero(
                    runtime._delay_buffer[ring_slot, pool_idx].abs()
                    > runtime.config.competition_epsilon
                ).item()
            )
        )
        external = torch.zeros(20)
        if tick == 0:
            for coordinate in event_coordinates:
                external += _external(coordinate)
        runtime.step(
            external_input=external,
            force_mode=RuntimeMode.WAKE,
            learning_signal=0.0,
        )
        if tick != 1:
            runtime._delay_buffer[ring_slot, pool_idx] = 0.0
        written_counts.append(
            int(
                torch.count_nonzero(
                    runtime._delay_buffer[ring_slot, pool_idx].abs()
                    > runtime.config.competition_epsilon
                ).item()
            )
        )
        if tick == 3:
            hidden_first = runtime.activation[hidden_idx].detach().clone()
        if tick == 6:
            target_final = runtime.activation[target_idx].detach().clone()
    desired = tuple(sorted(TARGET_MAPPING[role] for role in relevant))
    decoded = _target_set(target_final)
    return {
        "relevant_roles": list(relevant),
        "distractor_role": int(distractor_role),
        "event_coordinates": list(event_coordinates),
        "selected_indices": None if selected_indices is None else list(selected_indices),
        "desired_target_set": list(desired),
        "decoded_target_set": list(decoded),
        "success": decoded == desired,
        "hidden_positive_count": int(
            torch.count_nonzero(
                hidden_first > PRESYNAPTIC_EVENT_THRESHOLD
            ).item()
        ),
        "hidden_first_arrival": [float(value) for value in hidden_first.tolist()],
        "target_at_6": [float(value) for value in target_final.tolist()],
        "input_packet_count_by_tick": packet_counts,
        "input_written_count_by_tick": written_counts,
        "hippocampal_rows_after": len(runtime.hippocampus),
    }


def _rank_one_ablation(
    gate: MixedCueContentGateSnapshot,
) -> MixedCueContentGateSnapshot:
    content_axes = gate.content_axes.clone()
    content_axes[:, 1] = 0.0
    return replace(gate, content_axes=content_axes, content_rank=1)


def analyze_mixed_cue_content_row(seed: int, B: torch.Tensor) -> dict[str, Any]:
    block = _experience_block_compensated(B, condition="target_shuffle")
    base_snapshot, cutoff = _seal(block["runtime"])
    source, hidden, _target = _blocks()
    content = _role_content(B)
    content_sums = _observed_content_sums(content)
    cues = _mixed_cues(seed)
    gate = train_mixed_cue_content_gate(cues[:3], content_sums[:3])
    gate_before = _gate_hash(gate)

    training_rows: list[dict[str, Any]] = []
    training_compiled: list[list[int]] = []
    for row_index in TRAINING_ROWS:
        pair = COMPOSITION_PAIRS[row_index]
        missing = tuple(role for role in range(4) if role not in pair)
        arrived = tuple(source[role] for role in pair + (missing[0],))
        selected = compile_arrived_packet_indices(
            gate,
            cues[row_index],
            arrived,
            base_snapshot.weight,
            hidden,
        )
        training_compiled.append(list(selected))
        training_rows.append(
            _three_packet_probe(
                base_snapshot,
                source,
                pair,
                missing[0],
                selected,
            )
        )

    heldout_pair = COMPOSITION_PAIRS[HELDOUT_ROW]
    heldout_missing = tuple(role for role in range(4) if role not in heldout_pair)
    distractor_role = int(heldout_missing[0])
    role_coordinates = _heldout_coordinate_map(seed)
    remapped_snapshot = _snapshot_with_role_columns(base_snapshot, role_coordinates)
    arrived = tuple(
        role_coordinates[role]
        for role in heldout_pair + (distractor_role,)
    )
    binding = _content_binding_receipt(
        gate,
        cues[HELDOUT_ROW],
        arrived,
        remapped_snapshot.weight,
        hidden,
    )
    learned_indices = tuple(int(value) for value in binding["selected_indices"])
    oracle_indices = tuple(role_coordinates[role] for role in heldout_pair)

    learned = _three_packet_probe(
        base_snapshot,
        role_coordinates,
        heldout_pair,
        distractor_role,
        learned_indices,
    )
    oracle = _three_packet_probe(
        base_snapshot,
        role_coordinates,
        heldout_pair,
        distractor_role,
        oracle_indices,
    )
    joint_lookup = _three_packet_probe(
        base_snapshot,
        role_coordinates,
        heldout_pair,
        distractor_role,
        tuple(role_coordinates[role] for role in COMPOSITION_PAIRS[0]),
    )
    coordinate_memorizer = _three_packet_probe(
        base_snapshot,
        role_coordinates,
        heldout_pair,
        distractor_role,
        tuple(source[role] for role in heldout_pair),
    )
    wrong_cue_indices = compile_arrived_packet_indices(
        gate,
        cues[1],
        arrived,
        remapped_snapshot.weight,
        hidden,
    )
    wrong_cue = _three_packet_probe(
        base_snapshot,
        role_coordinates,
        heldout_pair,
        distractor_role,
        wrong_cue_indices,
    )

    hidden_idx = torch.tensor(hidden, dtype=torch.long)
    arrived_idx = torch.tensor(arrived, dtype=torch.long)
    shuffled_weight = remapped_snapshot.weight.detach().clone()
    original_arrived = shuffled_weight[hidden_idx[:, None], arrived_idx].clone()
    shuffled_weight[hidden_idx[:, None], arrived_idx] = original_arrived[:, [1, 2, 0]]
    shuffled_binding_indices = compile_arrived_packet_indices(
        gate,
        cues[HELDOUT_ROW],
        arrived,
        shuffled_weight,
        hidden,
    )
    shuffled_binding = _three_packet_probe(
        base_snapshot,
        role_coordinates,
        heldout_pair,
        distractor_role,
        shuffled_binding_indices,
    )

    rank_one_indices = compile_arrived_packet_indices(
        _rank_one_ablation(gate),
        cues[HELDOUT_ROW],
        arrived,
        remapped_snapshot.weight,
        hidden,
    )
    rank_one = _three_packet_probe(
        base_snapshot,
        role_coordinates,
        heldout_pair,
        distractor_role,
        rank_one_indices,
    )
    no_context = _three_packet_probe(
        base_snapshot,
        role_coordinates,
        heldout_pair,
        distractor_role,
        None,
    )

    predicted_training = torch.stack(
        [predict_mixed_cue_content(gate, cues[index]) for index in TRAINING_ROWS]
    )
    predicted_heldout = predict_mixed_cue_content(gate, cues[HELDOUT_ROW])
    cue_residual = float(
        torch.linalg.vector_norm(cues[3] - cues[1] - cues[2] + cues[0]).item()
    )
    additive_residual = float(
        torch.linalg.vector_norm(predicted_heldout - content_sums[3]).item()
    )
    alternative_completion_distance = float(
        torch.linalg.vector_norm(content_sums[3] - content_sums[0]).item()
    )
    gate_after = _gate_hash(gate)
    expected_receipt = [0, 0, 0, 3, 0, 0, 0]
    expected_written = [0, 3, 0, 0, 0, 0, 0]
    gates = {
        "rank_two_mixed_cue_and_content": bool(
            gate.cue_rank == 2 and gate.content_rank == 2
        ),
        "cue_parallelogram_predeclared": cue_residual <= 1e-12,
        "training_rows_reconstructed": bool(
            torch.allclose(predicted_training, content_sums[:3], atol=1e-10, rtol=0.0)
            and all(row["success"] for row in training_rows)
        ),
        "heldout_additive_content_exact": additive_residual <= 1e-10,
        "all_semantic_columns_moved_to_unseen_coordinates": bool(
            set(role_coordinates) == set(architectural_blocks(20)[1])
            and all(role_coordinates[index] != source[index] for index in range(4))
        ),
        "content_binding_unique": binding["binding_margin"] > MIN_BINDING_MARGIN,
        "heldout_remapped_transfer": bool(
            learned["success"]
            and learned["hidden_positive_count"] == 2
            and set(learned_indices) == set(oracle_indices)
        ),
        "oracle_bit_exact": learned["target_at_6"] == oracle["target_at_6"],
        "joint_lookup_fails": not joint_lookup["success"],
        "absolute_coordinate_memorizer_fails": not coordinate_memorizer["success"],
        "wrong_mixed_cue_fails": not wrong_cue["success"],
        "packet_content_shuffle_fails": not shuffled_binding["success"],
        "rank_one_ablation_fails": not rank_one["success"],
        "no_context_all_packet_fails": not no_context["success"],
        "one_shot_three_packet_receipt": bool(
            learned["input_packet_count_by_tick"] == expected_receipt
            and learned["input_written_count_by_tick"] == expected_written
        ),
        "same_training_admits_alternative_completion": alternative_completion_distance > 1e-6,
        "gate_frozen": gate_before == gate_after,
        "stores_zero": bool(
            cutoff["temporal_rows_after"] == 0
            and cutoff["hippocampal_rows_after"] == 0
            and learned["hippocampal_rows_after"] == 0
        ),
    }
    return {
        "seed": int(seed),
        "status": (
            "CONDITIONAL_RANK2_CONTENT_TRANSFER_PASS"
            if all(gates.values())
            else "CONDITIONAL_RANK2_CONTENT_TRANSFER_STOP"
        ),
        "gates": gates,
        "training_inputs": ["raw_mixed_cue", "cooccurring_packet_content_sum"],
        "cue_rank": gate.cue_rank,
        "content_rank": gate.content_rank,
        "cue_parallelogram_residual": cue_residual,
        "heldout_content_residual": additive_residual,
        "alternative_completion_distance": alternative_completion_distance,
        "role_coordinate_map": list(role_coordinates),
        "training_compiled_indices": training_compiled,
        "heldout_binding": binding,
        "learned_success": bool(learned["success"]),
        "oracle_success": bool(oracle["success"]),
        "joint_lookup_success": bool(joint_lookup["success"]),
        "coordinate_memorizer_success": bool(coordinate_memorizer["success"]),
        "wrong_cue_success": bool(wrong_cue["success"]),
        "binding_shuffle_success": bool(shuffled_binding["success"]),
        "rank_one_success": bool(rank_one["success"]),
        "no_context_success": bool(no_context["success"]),
        "learned": learned,
        "gate_hash": gate_before,
        "endpoint_opened": False,
        "claim_scope": (
            "synthetic conditional rank-two additive cue/content subspace "
            "with current-packet coordinate transfer"
        ),
    }


def analyze_mixed_cue_content_artifact(
    path: str | Path,
    *,
    stage: str,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "FRESH_INPUTS_READY":
        raise RuntimeError("fresh source-code inputs did not pass producer gates")
    rows = [
        analyze_mixed_cue_content_row(
            int(row["seed"]), torch.tensor(row["candidate_weights"])
        )
        for row in payload["rows"]
    ]
    if stage not in {"calibration", "development"}:
        raise ValueError("stage must be calibration or development")
    expected_count = 1 if stage == "calibration" else len(DEVELOPMENT_SEEDS)
    pass_count = sum(
        row["status"] == "CONDITIONAL_RANK2_CONTENT_TRANSFER_PASS"
        for row in rows
    )
    passed = len(rows) == expected_count and pass_count == expected_count
    return {
        "status": (
            "MIXED_CUE_CONTENT_CALIBRATION_PASS"
            if passed and stage == "calibration"
            else "MIXED_CUE_CONTENT_DEVELOPMENT_GO"
            if passed
            else "MIXED_CUE_CONTENT_STOP"
        ),
        "stage": stage,
        "seed_count": len(rows),
        "pass_count": pass_count,
        "learned_success_total": sum(row["learned_success"] for row in rows),
        "oracle_success_total": sum(row["oracle_success"] for row in rows),
        "joint_lookup_success_total": sum(row["joint_lookup_success"] for row in rows),
        "coordinate_memorizer_success_total": sum(
            row["coordinate_memorizer_success"] for row in rows
        ),
        "wrong_cue_success_total": sum(row["wrong_cue_success"] for row in rows),
        "binding_shuffle_success_total": sum(
            row["binding_shuffle_success"] for row in rows
        ),
        "rank_one_success_total": sum(row["rank_one_success"] for row in rows),
        "no_context_success_total": sum(row["no_context_success"] for row in rows),
        "maximum_cue_parallelogram_residual": max(
            row["cue_parallelogram_residual"] for row in rows
        ),
        "maximum_heldout_content_residual": max(
            row["heldout_content_residual"] for row in rows
        ),
        "endpoint_opened": False,
        "confirmation_opened": False,
        "rows": rows,
    }


__all__ = [
    "CALIBRATION_SEEDS",
    "DEVELOPMENT_SEEDS",
    "MixedCueContentGateSnapshot",
    "generate_fresh_inputs",
    "train_mixed_cue_content_gate",
    "predict_mixed_cue_content",
    "compile_arrived_packet_indices",
    "analyze_mixed_cue_content_artifact",
]
