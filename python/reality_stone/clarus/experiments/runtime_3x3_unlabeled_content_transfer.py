"""BA-TR26: conditional 3x3 affine-content transfer from opaque episodes.

The learner in this module receives only raw cue rows and contemporaneous
packet-content sums.  Grid coordinates and semantic role names exist solely
in the synthetic harness that produces the falsifiable fixture.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from ..runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode
from .runtime_context_branch_routing import architectural_blocks


DIMENSION = 30
CUE_DIMENSION = 8
CONTENT_DIMENSION = 6
PAIR_TICKS = 7
CALIBRATION_SEEDS = (113001,)
DEVELOPMENT_SEEDS = tuple(range(113101, 113117))
RANK_RELATIVE_TOLERANCE = 1e-10
MAX_CONDITION_NUMBER = 1e6
MAX_RELATIVE_FIT_ERROR = 1e-10
MAX_RELATIVE_SPAN_ERROR = 1e-10
MAX_RECTANGLE_RESIDUAL = 1e-10
MIN_RELATIVE_BINDING_MARGIN = 1e-6
MIN_TARGET_ACTIVATION = 1e-5
EXTERNAL_DRIVE = 5.0
CONTENT_OFF_LEVEL = 0.12
CONTENT_PEAK_LEVEL = 1.20
TARGET_TRUNK_WEIGHT = 1.20
TRAINING_CELL_COUNT = 8
EXPECTED_RECTANGLE_COUNT = 5
HELDOUT_CELL = (2, 2)


@dataclass(frozen=True)
class AffineContentGateSnapshot:
    cue_mean: torch.Tensor
    content_mean: torch.Tensor
    cue_basis: torch.Tensor
    cue_right_basis: torch.Tensor
    cue_singular_values: torch.Tensor
    centered_content: torch.Tensor
    linear_map: torch.Tensor
    cue_rank: int
    content_rank: int
    condition_number: float
    relative_fit_error: float
    rectangles: tuple[tuple[int, int, int, int], ...]
    rectangle_content_residuals: tuple[float, ...]
    training_count: int


def _relative_rank(matrix: torch.Tensor) -> tuple[int, torch.Tensor]:
    packed = torch.as_tensor(matrix, dtype=torch.float64)
    singular = torch.linalg.svdvals(packed)
    if singular.numel() == 0 or float(singular[0].item()) == 0.0:
        return 0, singular
    cutoff = RANK_RELATIVE_TOLERANCE * singular[0]
    return int(torch.count_nonzero(singular > cutoff).item()), singular


def _relative_residual(value: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = max(
        float(torch.linalg.vector_norm(reference).item()),
        torch.finfo(torch.float64).eps,
    )
    return float(torch.linalg.vector_norm(value).item()) / denominator


def discover_unlabeled_parallelograms(
    raw_cues: torch.Tensor,
    observed_content_sums: torch.Tensor,
) -> tuple[tuple[tuple[int, int, int, int], ...], tuple[float, ...]]:
    """Find cue parallelograms without row labels and audit their contents."""
    cues = torch.as_tensor(raw_cues, dtype=torch.float64)
    contents = torch.as_tensor(observed_content_sums, dtype=torch.float64)
    if cues.ndim != 2 or contents.ndim != 2 or cues.shape[0] != contents.shape[0]:
        raise ValueError("cue/content rows must be aligned matrices")
    if not torch.isfinite(cues).all() or not torch.isfinite(contents).all():
        raise ValueError("cue/content rows must be finite")
    rectangles: list[tuple[int, int, int, int]] = []
    content_residuals: list[float] = []
    for rows in itertools.combinations(range(cues.shape[0]), 4):
        a, b, c, d = rows
        pairings = (
            ((a, b), (c, d)),
            ((a, c), (b, d)),
            ((a, d), (b, c)),
        )
        for (left_a, left_b), (right_a, right_b) in pairings:
            cue_delta = cues[left_a] + cues[left_b] - cues[right_a] - cues[right_b]
            cue_scale = max(
                1.0,
                float(
                    torch.linalg.vector_norm(
                        cues[left_a] + cues[left_b]
                    ).item()
                ),
                float(
                    torch.linalg.vector_norm(
                        cues[right_a] + cues[right_b]
                    ).item()
                ),
            )
            if float(torch.linalg.vector_norm(cue_delta).item()) > (
                MAX_RECTANGLE_RESIDUAL * cue_scale
            ):
                continue
            content_delta = (
                contents[left_a]
                + contents[left_b]
                - contents[right_a]
                - contents[right_b]
            )
            content_scale = max(
                1.0,
                float(
                    torch.linalg.vector_norm(
                        contents[left_a] + contents[left_b]
                    ).item()
                ),
                float(
                    torch.linalg.vector_norm(
                        contents[right_a] + contents[right_b]
                    ).item()
                ),
            )
            residual = float(torch.linalg.vector_norm(content_delta).item()) / content_scale
            rectangles.append((left_a, left_b, right_a, right_b))
            content_residuals.append(residual)
    return tuple(rectangles), tuple(content_residuals)


def train_affine_content_gate(
    raw_cues: torch.Tensor,
    observed_content_sums: torch.Tensor,
) -> AffineContentGateSnapshot:
    """Fit the frozen rank-four affine map from eight opaque observations."""
    cues = torch.as_tensor(raw_cues, dtype=torch.float64)
    contents = torch.as_tensor(observed_content_sums, dtype=torch.float64)
    if cues.ndim != 2 or contents.ndim != 2:
        raise ValueError("affine-content training inputs must be matrices")
    if cues.shape[0] != TRAINING_CELL_COUNT or contents.shape[0] != cues.shape[0]:
        raise ValueError("affine-content training requires eight aligned rows")
    if cues.shape[1] < 4 or contents.shape[1] < 4:
        raise ValueError("affine-content training requires at least four coordinates")
    if not torch.isfinite(cues).all() or not torch.isfinite(contents).all():
        raise ValueError("affine-content training inputs must be finite")

    cue_mean = cues.mean(dim=0)
    content_mean = contents.mean(dim=0)
    centered_cue = (cues - cue_mean).T.contiguous()
    centered_content = (contents - content_mean).T.contiguous()
    cue_rank, cue_singular = _relative_rank(centered_cue)
    content_rank, _ = _relative_rank(centered_content)
    if cue_rank != 4 or content_rank != 4:
        raise RuntimeError("centered cue and content matrices must both have rank four")

    left, singular, right_h = torch.linalg.svd(centered_cue, full_matrices=False)
    left4 = left[:, :4]
    singular4 = singular[:4]
    right4 = right_h[:4].T
    condition_number = float((singular4[0] / singular4[-1]).item())
    if condition_number > MAX_CONDITION_NUMBER:
        raise RuntimeError("centered cue matrix is too ill-conditioned")
    linear_map = (
        centered_content
        @ right4
        @ torch.diag(singular4.reciprocal())
        @ left4.T
    )
    relative_fit_error = _relative_residual(
        centered_content - linear_map @ centered_cue,
        centered_content,
    )
    if relative_fit_error > MAX_RELATIVE_FIT_ERROR:
        raise RuntimeError("one global affine cue/content law does not fit training rows")

    rectangles, content_residuals = discover_unlabeled_parallelograms(cues, contents)
    covered = {index for rectangle in rectangles for index in rectangle}
    if len(rectangles) != EXPECTED_RECTANGLE_COUNT or covered != set(range(8)):
        raise RuntimeError("unlabeled cue rectangles do not match the frozen 8/9 fixture")
    if any(value > MAX_RECTANGLE_RESIDUAL for value in content_residuals):
        raise RuntimeError("content sums violate an observed cue parallelogram")

    return AffineContentGateSnapshot(
        cue_mean=cue_mean.clone(),
        content_mean=content_mean.clone(),
        cue_basis=left4.clone(),
        cue_right_basis=right4.clone(),
        cue_singular_values=singular4.clone(),
        centered_content=centered_content.clone(),
        linear_map=linear_map.clone(),
        cue_rank=cue_rank,
        content_rank=content_rank,
        condition_number=condition_number,
        relative_fit_error=relative_fit_error,
        rectangles=rectangles,
        rectangle_content_residuals=content_residuals,
        training_count=8,
    )


def affine_content_prediction_receipt(
    gate: AffineContentGateSnapshot,
    raw_cue: torch.Tensor,
) -> dict[str, Any]:
    cue = torch.as_tensor(raw_cue, dtype=torch.float64).view(-1)
    if cue.shape != gate.cue_mean.shape or not torch.isfinite(cue).all():
        raise ValueError("affine-content cue has the wrong shape or is nonfinite")
    centered = cue - gate.cue_mean
    projection = gate.cue_basis @ (gate.cue_basis.T @ centered)
    span_error = _relative_residual(centered - projection, centered)
    if span_error > MAX_RELATIVE_SPAN_ERROR:
        raise RuntimeError("affine-content query is outside the observed cue span")
    prediction = gate.content_mean + gate.linear_map @ centered
    return {
        "prediction": prediction,
        "relative_span_error": span_error,
    }


def predict_affine_content(
    gate: AffineContentGateSnapshot,
    raw_cue: torch.Tensor,
) -> torch.Tensor:
    return affine_content_prediction_receipt(gate, raw_cue)["prediction"]


def _bind_prediction_to_packets(
    prediction: torch.Tensor,
    arrived_packet_indices: Sequence[int],
    weight: torch.Tensor,
    response_indices: Sequence[int],
) -> dict[str, Any]:
    packet_indices = tuple(int(value) for value in arrived_packet_indices)
    response = tuple(int(value) for value in response_indices)
    packed = torch.as_tensor(weight, dtype=torch.float64)
    if packed.ndim != 2 or packed.shape[0] != packed.shape[1]:
        raise ValueError("packet binding requires a square weight matrix")
    if len(packet_indices) < 3 or len(set(packet_indices)) != len(packet_indices):
        raise ValueError("packet binding requires at least three distinct arrivals")
    if not response or len(set(response)) != len(response):
        raise ValueError("response coordinates must be nonempty and distinct")
    all_indices = packet_indices + response
    if min(all_indices) < 0 or max(all_indices) >= packed.shape[0]:
        raise ValueError("packet binding coordinate is out of range")
    rows = torch.tensor(response, dtype=torch.long)
    columns = torch.tensor(packet_indices, dtype=torch.long)
    descriptors = packed.index_select(0, rows).index_select(1, columns).T
    norms = descriptors.norm(dim=1, keepdim=True)
    if torch.any(norms <= 0.0) or not torch.isfinite(descriptors).all():
        raise RuntimeError("arrived packet descriptor is zero or nonfinite")
    descriptors = descriptors / norms
    predicted = torch.as_tensor(prediction, dtype=torch.float64).view(-1)
    if predicted.numel() != descriptors.shape[1]:
        raise ValueError("prediction and packet descriptors have different dimensions")
    candidates: list[tuple[float, tuple[int, int]]] = []
    for left, right in itertools.combinations(range(len(packet_indices)), 2):
        residual = float(
            torch.linalg.vector_norm(
                predicted - descriptors[left] - descriptors[right]
            ).item()
        )
        candidates.append((residual, (left, right)))
    candidates.sort(key=lambda item: (item[0], item[1]))
    scale = max(1.0, float(torch.linalg.vector_norm(predicted).item()))
    relative_margin = (candidates[1][0] - candidates[0][0]) / scale
    if relative_margin <= MIN_RELATIVE_BINDING_MARGIN:
        raise RuntimeError("current-packet content binding is tied")
    best = candidates[0][1]
    return {
        "selected_indices": [packet_indices[best[0]], packet_indices[best[1]]],
        "best_residual": candidates[0][0],
        "second_residual": candidates[1][0],
        "relative_binding_margin": relative_margin,
        "predicted_content": [float(value) for value in predicted.tolist()],
    }


def compile_current_packet_indices(
    gate: AffineContentGateSnapshot,
    raw_cue: torch.Tensor,
    arrived_packet_indices: Sequence[int],
    weight: torch.Tensor,
    response_indices: Sequence[int],
) -> tuple[int, ...]:
    """Compile two current coordinates from predicted and arrived content."""
    prediction = predict_affine_content(gate, raw_cue)
    receipt = _bind_prediction_to_packets(
        prediction,
        arrived_packet_indices,
        weight,
        response_indices,
    )
    return tuple(int(value) for value in receipt["selected_indices"])


def _gate_hash(gate: AffineContentGateSnapshot) -> str:
    digest = hashlib.sha256()
    for tensor in (
        gate.cue_mean,
        gate.content_mean,
        gate.cue_basis,
        gate.cue_right_basis,
        gate.cue_singular_values,
        gate.centered_content,
        gate.linear_map,
    ):
        digest.update(tensor.detach().cpu().numpy().tobytes())
    digest.update(
        repr(
            (
                gate.cue_rank,
                gate.content_rank,
                gate.condition_number,
                gate.relative_fit_error,
                gate.rectangles,
                gate.rectangle_content_residuals,
                gate.training_count,
            )
        ).encode("ascii")
    )
    return digest.hexdigest()


def _rank_three_ablation(
    gate: AffineContentGateSnapshot,
) -> AffineContentGateSnapshot:
    left3 = gate.cue_basis[:, :3]
    right3 = gate.cue_right_basis[:, :3]
    singular3 = gate.cue_singular_values[:3]
    linear_map = (
        gate.centered_content
        @ right3
        @ torch.diag(singular3.reciprocal())
        @ left3.T
    )
    return replace(
        gate,
        linear_map=linear_map,
        cue_rank=3,
        content_rank=3,
    )


def _blocks() -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    blocks = architectural_blocks(DIMENSION)
    return tuple(blocks[0] + blocks[1]), tuple(blocks[2]), tuple(blocks[4])


def _content_dictionary(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 310_019)
    winners = torch.randperm(CONTENT_DIMENSION, generator=generator)
    matrix = torch.full(
        (CONTENT_DIMENSION, CONTENT_DIMENSION),
        CONTENT_OFF_LEVEL,
        dtype=torch.float64,
    )
    for column, row in enumerate(winners.tolist()):
        matrix[row, column] = CONTENT_PEAK_LEVEL
    return matrix


def _tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(
        torch.as_tensor(value).detach().cpu().numpy().tobytes()
    ).hexdigest()


def generate_fresh_inputs(seeds: Sequence[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed_value in seeds:
        seed = int(seed_value)
        matrix = _content_dictionary(seed)
        norms = matrix.norm(dim=0)
        rank = int(torch.linalg.matrix_rank(matrix).item())
        winners = torch.argmax(matrix, dim=0)
        gates = {
            "finite": bool(torch.isfinite(matrix).all()),
            "strictly_positive": bool(torch.all(matrix > 0.0)),
            "equal_column_norm": float((norms.max() - norms.min()).item()) <= 1e-12,
            "full_rank": rank == CONTENT_DIMENSION,
            "unique_winner_per_column": len(set(winners.tolist())) == CONTENT_DIMENSION,
        }
        rows.append(
            {
                "seed": seed,
                "content_columns": [
                    [float(value) for value in row] for row in matrix.tolist()
                ],
                "content_sha256": _tensor_hash(matrix),
                "column_norm_spread": float((norms.max() - norms.min()).item()),
                "matrix_rank": rank,
                "winner_rows": [int(value) for value in winners.tolist()],
                "gates": gates,
            }
        )
    ready = bool(rows) and all(all(row["gates"].values()) for row in rows)
    return {
        "status": "FRESH_SIX_CONTENT_INPUTS_READY" if ready else "FRESH_INPUTS_STOP",
        "seed_count": len(rows),
        "rows": rows,
    }


def _base_snapshot(content_columns: torch.Tensor) -> Any:
    matrix = torch.as_tensor(content_columns, dtype=torch.float32)
    if matrix.shape != (CONTENT_DIMENSION, CONTENT_DIMENSION):
        raise ValueError("content source must be 6x6")
    input_pool, hidden, target = _blocks()
    canonical = input_pool[:CONTENT_DIMENSION]
    weight = torch.zeros(DIMENSION, DIMENSION, dtype=torch.float32)
    weight[torch.tensor(hidden)[:, None], torch.tensor(canonical)] = matrix
    weight[torch.tensor(target)[:, None], torch.tensor(hidden)] = (
        TARGET_TRUNK_WEIGHT * torch.eye(CONTENT_DIMENSION)
    )
    config = BrainRuntimeConfig(
        dim=DIMENSION,
        active_ratio=1.0,
        active_threshold=0.0,
        force_all_active_selection=True,
        noise_sigma=0.0,
        dale_law=False,
        axon_delay=True,
        max_axon_delay=2,
        f1_self_measure=False,
        stdp_enabled=False,
        memory_capacity=1,
        hippocampal_encoding_enabled=False,
        competition_indices=hidden,
        competition_lateral_gain=1.0,
        competition_homeostasis_gain=0.0,
        competition_homeostasis_rate=0.0,
        competition_homeostasis_decay=0.0,
        competition_jitter_sigma=0.0,
        competition_input_indices=(canonical[0],),
        competition_k_from_delayed_input=False,
        competition_factorize_delayed_input=True,
    )
    runtime = BrainRuntime(weight, config=config, backend="torch", device="cpu")
    return runtime.snapshot()


def _snapshot_with_content_columns(
    snapshot: Any,
    content_columns: torch.Tensor,
    role_coordinates: Sequence[int],
) -> Any:
    coordinates = tuple(int(value) for value in role_coordinates)
    input_pool, hidden, _ = _blocks()
    if len(coordinates) != CONTENT_DIMENSION or len(set(coordinates)) != CONTENT_DIMENSION:
        raise ValueError("role coordinate map must contain six distinct entries")
    if not set(coordinates).issubset(input_pool):
        raise ValueError("role coordinates must lie in the declared input pool")
    matrix = torch.as_tensor(content_columns, dtype=torch.float32)
    packed = snapshot.weight.detach().clone()
    hidden_idx = torch.tensor(hidden, dtype=torch.long)
    packed[hidden_idx[:, None], torch.tensor(input_pool)] = 0.0
    for role, coordinate in enumerate(coordinates):
        packed[hidden_idx, coordinate] = matrix[:, role]
    return replace(snapshot, weight=packed)


def _episode_coordinate_maps(seed: int) -> tuple[tuple[int, ...], ...]:
    input_pool, _, _ = _blocks()
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 320_023)
    maps: list[tuple[int, ...]] = []
    attempts = 0
    while len(maps) < TRAINING_CELL_COUNT:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("could not construct distinct training coordinate maps")
        proposal = tuple(
            input_pool[index]
            for index in torch.randperm(len(input_pool), generator=generator)[:6].tolist()
        )
        if proposal not in maps:
            maps.append(proposal)
    second_block = tuple(architectural_blocks(DIMENSION)[1])
    while True:
        heldout = tuple(
            second_block[index]
            for index in torch.randperm(6, generator=generator).tolist()
        )
        if heldout not in maps:
            break
    return tuple(maps + [heldout])


def _raw_cues(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 330_037)
    chart, _ = torch.linalg.qr(
        torch.randn(CUE_DIMENSION, 5, generator=generator, dtype=torch.float64)
    )
    base = 1.25 * chart[:, 0]
    first_levels = (torch.zeros(CUE_DIMENSION, dtype=torch.float64), chart[:, 1], chart[:, 2])
    second_levels = (torch.zeros(CUE_DIMENSION, dtype=torch.float64), chart[:, 3], chart[:, 4])
    return torch.stack(
        [base + first_levels[first] + second_levels[second]
         for first in range(3) for second in range(3)]
    )


def _content_descriptors(
    weight: torch.Tensor,
    packet_indices: Sequence[int],
    response_indices: Sequence[int],
) -> torch.Tensor:
    rows = torch.tensor(tuple(int(value) for value in response_indices), dtype=torch.long)
    columns = torch.tensor(tuple(int(value) for value in packet_indices), dtype=torch.long)
    descriptors = torch.as_tensor(weight, dtype=torch.float64).index_select(0, rows).index_select(1, columns).T
    norms = descriptors.norm(dim=1, keepdim=True)
    if torch.any(norms <= 0.0):
        raise RuntimeError("content observation encountered a zero packet column")
    return descriptors / norms


def _external(indices: Sequence[int]) -> torch.Tensor:
    value = torch.zeros(DIMENSION, dtype=torch.float32)
    for index in indices:
        value[int(index)] += EXTERNAL_DRIVE
    return value


def _target_set(values: torch.Tensor) -> tuple[int, ...]:
    packed = torch.as_tensor(values, dtype=torch.float64).view(CONTENT_DIMENSION)
    return tuple(
        int(value)
        for value in torch.nonzero(
            packed >= MIN_TARGET_ACTIVATION,
            as_tuple=False,
        ).view(-1)
    )


def _packet_probe(
    snapshot: Any,
    content_columns: torch.Tensor,
    role_coordinates: Sequence[int],
    event_roles: Sequence[int],
    selected_indices: Sequence[int],
) -> dict[str, Any]:
    coordinates = tuple(int(value) for value in role_coordinates)
    roles = tuple(int(value) for value in event_roles)
    selected = tuple(int(value) for value in selected_indices)
    if not roles or len(set(roles)) != len(roles):
        raise ValueError("packet probe events must be nonempty and distinct")
    routed = _snapshot_with_content_columns(snapshot, content_columns, coordinates)
    config = replace(
        routed.config,
        competition_input_indices=selected,
        competition_k_from_delayed_input=False,
        competition_factorize_delayed_input=True,
    )
    routed = replace(routed, config=config)
    runtime = BrainRuntime.from_snapshot(routed, backend="torch", device="cpu")
    input_pool, hidden, target = _blocks()
    input_idx = torch.tensor(input_pool, dtype=torch.long)
    hidden_idx = torch.tensor(hidden, dtype=torch.long)
    target_idx = torch.tensor(target, dtype=torch.long)
    event_coordinates = tuple(coordinates[role] for role in roles)
    packet_counts: list[int] = []
    written_counts: list[int] = []
    hidden_first = torch.zeros(CONTENT_DIMENSION)
    target_final = torch.zeros(CONTENT_DIMENSION)
    for tick in range(PAIR_TICKS):
        if runtime._delay_buffer is None:
            raise RuntimeError("3x3 content transfer requires the delay ring")
        ring_slot = runtime._delay_idx % runtime.config.max_axon_delay
        packet_counts.append(
            int(
                torch.count_nonzero(
                    runtime._delay_buffer[ring_slot, input_idx].abs()
                    > runtime.config.competition_epsilon
                ).item()
            )
        )
        runtime.step(
            external_input=_external(event_coordinates) if tick == 0 else _external(()),
            force_mode=RuntimeMode.WAKE,
            learning_signal=0.0,
        )
        if tick != 1:
            runtime._delay_buffer[ring_slot, input_idx] = 0.0
        written_counts.append(
            int(
                torch.count_nonzero(
                    runtime._delay_buffer[ring_slot, input_idx].abs()
                    > runtime.config.competition_epsilon
                ).item()
            )
        )
        if tick == 3:
            hidden_first = runtime.activation[hidden_idx].detach().clone()
        if tick == 6:
            target_final = runtime.activation[target_idx].detach().clone()
    return {
        "event_roles": list(roles),
        "event_coordinates": list(event_coordinates),
        "selected_indices": list(selected),
        "decoded_target_set": list(_target_set(target_final)),
        "hidden_positive_count": int(
            torch.count_nonzero(hidden_first > runtime.config.competition_epsilon).item()
        ),
        "hidden_first_arrival": [float(value) for value in hidden_first.tolist()],
        "target_at_6": [float(value) for value in target_final.tolist()],
        "input_packet_count_by_tick": packet_counts,
        "input_written_count_by_tick": written_counts,
        "hippocampal_rows_after": len(runtime.hippocampus),
    }


def _compile_or_abstain(
    gate: AffineContentGateSnapshot,
    cue: torch.Tensor,
    arrived: Sequence[int],
    weight: torch.Tensor,
    hidden: Sequence[int],
) -> tuple[tuple[int, ...] | None, str | None]:
    try:
        return compile_current_packet_indices(gate, cue, arrived, weight, hidden), None
    except (RuntimeError, ValueError) as exc:
        return None, str(exc)


def _control_probe(
    snapshot: Any,
    content_columns: torch.Tensor,
    coordinates: Sequence[int],
    events: Sequence[int],
    selected: tuple[int, ...] | None,
    expected: tuple[int, ...],
) -> dict[str, Any]:
    if selected is None:
        return {"success": False, "abstained": True}
    receipt = _packet_probe(snapshot, content_columns, coordinates, events, selected)
    receipt["success"] = tuple(receipt["decoded_target_set"]) == expected
    receipt["abstained"] = False
    return receipt


def analyze_3x3_unlabeled_content_row(
    seed: int,
    content_columns: torch.Tensor,
) -> dict[str, Any]:
    seed = int(seed)
    content_columns = torch.as_tensor(content_columns, dtype=torch.float64)
    base_snapshot = _base_snapshot(content_columns)
    input_pool, hidden, _ = _blocks()
    maps = _episode_coordinate_maps(seed)
    cues = _raw_cues(seed)
    cells = tuple((first, second) for first in range(3) for second in range(3))

    observed: list[torch.Tensor] = []
    for row, (first, second) in enumerate(cells[:8]):
        moved = _snapshot_with_content_columns(base_snapshot, content_columns, maps[row])
        descriptors = _content_descriptors(moved.weight, maps[row], hidden)
        observed.append(descriptors[first] + descriptors[3 + second])
    observed_content = torch.stack(observed)

    generator = torch.Generator(device="cpu").manual_seed(seed + 340_039)
    training_order = torch.randperm(8, generator=generator)
    training_cues = cues[:8].index_select(0, training_order)
    training_content = observed_content.index_select(0, training_order)
    gate = train_affine_content_gate(training_cues, training_content)
    gate_before = _gate_hash(gate)
    predicted_training = torch.stack(
        [predict_affine_content(gate, row) for row in training_cues]
    )

    query_receipt = affine_content_prediction_receipt(gate, cues[8])
    heldout_prediction = query_receipt["prediction"]
    heldout_coordinates = maps[8]
    heldout_snapshot = _snapshot_with_content_columns(
        base_snapshot,
        content_columns,
        heldout_coordinates,
    )
    heldout_descriptors = _content_descriptors(
        heldout_snapshot.weight,
        heldout_coordinates,
        hidden,
    )
    expected_content = heldout_descriptors[2] + heldout_descriptors[5]
    heldout_content_error = _relative_residual(
        heldout_prediction - expected_content,
        expected_content,
    )
    event_roles = (2, 5, 0)
    arrived = tuple(heldout_coordinates[role] for role in event_roles)
    binding = _bind_prediction_to_packets(
        heldout_prediction,
        arrived,
        heldout_snapshot.weight,
        hidden,
    )
    learned_indices = tuple(int(value) for value in binding["selected_indices"])
    oracle_indices = tuple(heldout_coordinates[role] for role in (2, 5))

    association_shuffle_rejected = False
    association_shuffle_error = None
    try:
        train_affine_content_gate(
            training_cues,
            training_content.roll(shifts=1, dims=0),
        )
    except (RuntimeError, ValueError) as exc:
        association_shuffle_rejected = True
        association_shuffle_error = str(exc)

    reverse = torch.arange(7, -1, -1)
    row_gate = train_affine_content_gate(
        training_cues.index_select(0, reverse),
        training_content.index_select(0, reverse),
    )
    row_order_prediction_error = float(
        torch.linalg.vector_norm(
            predict_affine_content(row_gate, cues[8]) - heldout_prediction
        ).item()
    )
    chart_generator = torch.Generator(device="cpu").manual_seed(seed + 350_041)
    chart, _ = torch.linalg.qr(
        torch.randn(CUE_DIMENSION, CUE_DIMENSION, generator=chart_generator, dtype=torch.float64)
    )
    chart_gate = train_affine_content_gate(training_cues @ chart.T, training_content)
    chart_prediction_error = float(
        torch.linalg.vector_norm(
            predict_affine_content(chart_gate, cues[8] @ chart.T) - heldout_prediction
        ).item()
    )
    all_maps_fresh = len(set(maps)) == 9
    heldout_in_second_block = set(heldout_coordinates) == set(architectural_blocks(DIMENSION)[1])
    training_reconstruction_error = _relative_residual(
        predicted_training - training_content,
        training_content,
    )
    alternative_delta = torch.linspace(0.1, 0.6, CONTENT_DIMENSION, dtype=torch.float64)
    alternative_completion_distance = float(torch.linalg.vector_norm(alternative_delta).item())

    preflight_gates = {
        "rank_four_cue_and_content": gate.cue_rank == 4 and gate.content_rank == 4,
        "five_unlabeled_training_rectangles": len(gate.rectangles) == 5,
        "rectangle_hypergraph_covers_all_rows": (
            {index for rectangle in gate.rectangles for index in rectangle} == set(range(8))
        ),
        "training_affine_fit": training_reconstruction_error <= MAX_RELATIVE_FIT_ERROR,
        "heldout_query_in_span": query_receipt["relative_span_error"] <= MAX_RELATIVE_SPAN_ERROR,
        "conditional_heldout_content_exact": heldout_content_error <= MAX_RELATIVE_FIT_ERROR,
        "current_packet_binding_unique": (
            binding["relative_binding_margin"] > MIN_RELATIVE_BINDING_MARGIN
        ),
        "episode_coordinate_maps_fresh": all_maps_fresh,
        "heldout_all_columns_in_disjoint_second_block": heldout_in_second_block,
        "association_shuffle_rejected_pre_endpoint": association_shuffle_rejected,
        "row_order_invariant": row_order_prediction_error <= 1e-10,
        "orthogonal_cue_chart_invariant": chart_prediction_error <= 1e-10,
        "alternative_heldout_completion_exists": alternative_completion_distance > 1e-6,
    }
    if not all(preflight_gates.values()):
        return {
            "seed": seed,
            "status": "CONDITIONAL_3X3_AFFINE_CONTENT_STOP",
            "gates": preflight_gates,
            "association_shuffle_error": association_shuffle_error,
            "endpoint_opened": False,
            "claim_scope": "pre-endpoint conditional affine apparatus only",
        }

    atomic = [
        _packet_probe(
            base_snapshot,
            content_columns,
            heldout_coordinates,
            (role,),
            (heldout_coordinates[role],),
        )
        for role in (2, 5)
    ]
    expected_target = tuple(
        sorted(
            set(atomic[0]["decoded_target_set"])
            | set(atomic[1]["decoded_target_set"])
        )
    )
    learned = _control_probe(
        base_snapshot,
        content_columns,
        heldout_coordinates,
        event_roles,
        learned_indices,
        expected_target,
    )
    oracle = _control_probe(
        base_snapshot,
        content_columns,
        heldout_coordinates,
        event_roles,
        oracle_indices,
        expected_target,
    )
    joint_lookup = _control_probe(
        base_snapshot,
        content_columns,
        heldout_coordinates,
        event_roles,
        tuple(heldout_coordinates[role] for role in (0, 3)),
        expected_target,
    )
    coordinate_memorizer = _control_probe(
        base_snapshot,
        content_columns,
        heldout_coordinates,
        event_roles,
        tuple(input_pool[role] for role in (2, 5)),
        expected_target,
    )
    wrong_cue_indices, wrong_cue_error = _compile_or_abstain(
        gate,
        cues[2],
        arrived,
        heldout_snapshot.weight,
        hidden,
    )
    wrong_cue = _control_probe(
        base_snapshot,
        content_columns,
        heldout_coordinates,
        event_roles,
        wrong_cue_indices,
        expected_target,
    )

    hidden_idx = torch.tensor(hidden, dtype=torch.long)
    arrived_idx = torch.tensor(arrived, dtype=torch.long)
    shuffled_weight = heldout_snapshot.weight.detach().clone()
    original_arrived = shuffled_weight[hidden_idx[:, None], arrived_idx].clone()
    shuffled_weight[hidden_idx[:, None], arrived_idx] = original_arrived[:, [1, 2, 0]]
    binding_shuffle_indices, binding_shuffle_error = _compile_or_abstain(
        gate,
        cues[8],
        arrived,
        shuffled_weight,
        hidden,
    )
    binding_shuffle = _control_probe(
        base_snapshot,
        content_columns,
        heldout_coordinates,
        event_roles,
        binding_shuffle_indices,
        expected_target,
    )
    rank_three_indices, rank_three_error = _compile_or_abstain(
        _rank_three_ablation(gate),
        cues[8],
        arrived,
        heldout_snapshot.weight,
        hidden,
    )
    rank_three = _control_probe(
        base_snapshot,
        content_columns,
        heldout_coordinates,
        event_roles,
        rank_three_indices,
        expected_target,
    )
    no_context = _control_probe(
        base_snapshot,
        content_columns,
        heldout_coordinates,
        event_roles,
        tuple(input_pool),
        expected_target,
    )

    expected_packet_receipt = [0, 0, 0, 3, 0, 0, 0]
    expected_written_receipt = [0, 3, 0, 0, 0, 0, 0]
    gate_after = _gate_hash(gate)
    endpoint_gates = {
        "learned_heldout_transfer": bool(
            learned["success"]
            and learned["hidden_positive_count"] == 2
            and set(learned_indices) == set(oracle_indices)
        ),
        "oracle_matches_atomic_union": bool(
            oracle["success"]
            and len(expected_target) == 2
            and all(len(item["decoded_target_set"]) == 1 for item in atomic)
        ),
        "learned_oracle_bit_exact": learned["target_at_6"] == oracle["target_at_6"],
        "joint_lookup_fails": not joint_lookup["success"],
        "absolute_coordinate_memorizer_fails": not coordinate_memorizer["success"],
        "wrong_raw_cue_fails": not wrong_cue["success"],
        "packet_binding_shuffle_fails": not binding_shuffle["success"],
        "rank_three_ablation_fails": not rank_three["success"],
        "no_context_all_packet_fails": not no_context["success"],
        "one_shot_three_packet_receipt": bool(
            learned["input_packet_count_by_tick"] == expected_packet_receipt
            and learned["input_written_count_by_tick"] == expected_written_receipt
        ),
        "gate_frozen": gate_before == gate_after,
        "stores_zero": bool(
            len(base_snapshot.hippocampus.get("priority", [])) == 0
            and learned["hippocampal_rows_after"] == 0
            and oracle["hippocampal_rows_after"] == 0
        ),
    }
    gates = {**preflight_gates, **endpoint_gates}
    return {
        "seed": seed,
        "status": (
            "CONDITIONAL_3X3_AFFINE_CONTENT_PASS"
            if all(gates.values())
            else "CONDITIONAL_3X3_AFFINE_CONTENT_STOP"
        ),
        "gates": gates,
        "training_inputs": ["opaque_raw_cue", "contemporaneous_packet_content_sum"],
        "cue_rank": gate.cue_rank,
        "content_rank": gate.content_rank,
        "condition_number": gate.condition_number,
        "relative_fit_error": gate.relative_fit_error,
        "heldout_relative_span_error": query_receipt["relative_span_error"],
        "heldout_content_error": heldout_content_error,
        "rectangle_count": len(gate.rectangles),
        "rectangle_content_residual_max": max(gate.rectangle_content_residuals),
        "training_order": [int(value) for value in training_order.tolist()],
        "coordinate_maps": [list(item) for item in maps],
        "heldout_binding": binding,
        "expected_target_set_from_atomic_union": list(expected_target),
        "learned_success": bool(learned["success"]),
        "oracle_success": bool(oracle["success"]),
        "joint_lookup_success": bool(joint_lookup["success"]),
        "coordinate_memorizer_success": bool(coordinate_memorizer["success"]),
        "wrong_cue_success": bool(wrong_cue["success"]),
        "binding_shuffle_success": bool(binding_shuffle["success"]),
        "rank_three_success": bool(rank_three["success"]),
        "no_context_success": bool(no_context["success"]),
        "wrong_cue_error": wrong_cue_error,
        "binding_shuffle_error": binding_shuffle_error,
        "rank_three_error": rank_three_error,
        "association_shuffle_error": association_shuffle_error,
        "row_order_prediction_error": row_order_prediction_error,
        "orthogonal_chart_prediction_error": chart_prediction_error,
        "alternative_heldout_delta_norm": alternative_completion_distance,
        "learned": learned,
        "oracle": oracle,
        "atomic": atomic,
        "gate_hash": gate_before,
        "content_sha256": _tensor_hash(content_columns),
        "endpoint_opened": True,
        "claim_scope": (
            "synthetic conditional rank-four global-affine cue/content transfer "
            "with current-column coordinate binding"
        ),
    }


def analyze_3x3_unlabeled_content_artifact(
    path: str | Path,
    *,
    stage: str,
) -> dict[str, Any]:
    if stage not in {"calibration", "development"}:
        raise ValueError("stage must be calibration or development")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "FRESH_SIX_CONTENT_INPUTS_READY":
        raise RuntimeError("fresh six-content inputs did not pass producer gates")
    expected_seeds = CALIBRATION_SEEDS if stage == "calibration" else DEVELOPMENT_SEEDS
    actual_seeds = tuple(int(row["seed"]) for row in payload.get("rows", ()))
    if actual_seeds != expected_seeds:
        raise RuntimeError("fresh input seed order does not match the frozen stage")
    rows = [
        analyze_3x3_unlabeled_content_row(
            int(row["seed"]),
            torch.tensor(row["content_columns"], dtype=torch.float64),
        )
        for row in payload["rows"]
    ]
    pass_count = sum(
        row["status"] == "CONDITIONAL_3X3_AFFINE_CONTENT_PASS" for row in rows
    )
    passed = pass_count == len(expected_seeds)
    return {
        "status": (
            "UNLABELED_3X3_AFFINE_CONTENT_CALIBRATION_PASS"
            if passed and stage == "calibration"
            else "UNLABELED_3X3_AFFINE_CONTENT_DEVELOPMENT_GO"
            if passed
            else "UNLABELED_3X3_AFFINE_CONTENT_STOP"
        ),
        "stage": stage,
        "seed_count": len(rows),
        "pass_count": pass_count,
        "learned_success_total": sum(row.get("learned_success", False) for row in rows),
        "oracle_success_total": sum(row.get("oracle_success", False) for row in rows),
        "joint_lookup_success_total": sum(row.get("joint_lookup_success", False) for row in rows),
        "coordinate_memorizer_success_total": sum(
            row.get("coordinate_memorizer_success", False) for row in rows
        ),
        "wrong_cue_success_total": sum(row.get("wrong_cue_success", False) for row in rows),
        "binding_shuffle_success_total": sum(
            row.get("binding_shuffle_success", False) for row in rows
        ),
        "rank_three_success_total": sum(row.get("rank_three_success", False) for row in rows),
        "no_context_success_total": sum(row.get("no_context_success", False) for row in rows),
        "maximum_relative_fit_error": max(
            row.get("relative_fit_error", float("inf")) for row in rows
        ),
        "maximum_heldout_content_error": max(
            row.get("heldout_content_error", float("inf")) for row in rows
        ),
        "minimum_binding_margin": min(
            row.get("heldout_binding", {}).get("relative_binding_margin", float("-inf"))
            for row in rows
        ),
        "endpoint_opened": any(row.get("endpoint_opened", False) for row in rows),
        "confirmation_opened": False,
        "rows": rows,
    }


__all__ = [
    "AffineContentGateSnapshot",
    "CALIBRATION_SEEDS",
    "DEVELOPMENT_SEEDS",
    "discover_unlabeled_parallelograms",
    "train_affine_content_gate",
    "predict_affine_content",
    "compile_current_packet_indices",
    "generate_fresh_inputs",
    "analyze_3x3_unlabeled_content_row",
    "analyze_3x3_unlabeled_content_artifact",
]
