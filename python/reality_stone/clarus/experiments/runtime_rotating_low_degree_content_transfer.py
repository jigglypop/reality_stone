"""BA-TR28: rotating-holdout transfer with a generic degree-two operator.

The learner sees only dimensionless raw cue vectors and contemporaneous
packet-content vectors.  Grid coordinates, held-out identities, runtime
coordinates, and endpoint responses belong to the synthetic harness and are
never passed to the learner or compiler.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from ..runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode
from .runtime_3x3_unlabeled_content_transfer import (
    CONTENT_DIMENSION,
    DIMENSION,
    EXTERNAL_DRIVE,
    PAIR_TICKS,
    TARGET_TRUNK_WEIGHT,
    _blocks,
    _snapshot_with_content_columns,
)


CALIBRATION_SEEDS = (115001,)
DEVELOPMENT_SEEDS = tuple(range(115101, 115117))
GRID_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)
GRID_CELL_COUNT = len(GRID_LEVELS) ** 2
CUE_DIMENSION = 8
INTRINSIC_CUE_RANK = 2
FEATURE_COUNT = 6
RANK_RELATIVE_TOLERANCE = 1e-10
MAX_CONDITION_NUMBER = 1e4
MAX_RELATIVE_FIT_ERROR = 1e-10
MAX_RELATIVE_SPAN_ERROR = 1e-10
MAX_RELATIVE_QUERY_ERROR = 1e-9
MIN_RELATIVE_BINDING_MARGIN = 1e-4
MIN_QUADRATIC_FRACTION = 0.05
MIN_AFFINE_MEAN_ERROR = 1e-3
MAX_AFFINE_BINDING_SUCCESS_FRACTION = 0.50
MAX_ROUTE_ERROR = 1e-6
MIN_ROUTE_SEPARATION = 1e-5
MAX_CANDIDATE_NORM_RATIO = 1.75
EXACT_LOOKUP_TOLERANCE = 1e-12


@dataclass(frozen=True)
class LowDegreeContentGateSnapshot:
    cue_mean: torch.Tensor
    cue_basis: torch.Tensor
    coefficients: torch.Tensor
    cue_singular_values: torch.Tensor
    feature_singular_values: torch.Tensor
    feature_right_basis: torch.Tensor
    cue_rank: int
    feature_rank: int
    condition_number: float
    relative_fit_error: float
    training_count: int


def _relative_rank(matrix: torch.Tensor) -> tuple[int, torch.Tensor]:
    singular = torch.linalg.svdvals(torch.as_tensor(matrix, dtype=torch.float64))
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


def _quadratic_features(coordinates: torch.Tensor) -> torch.Tensor:
    packed = torch.as_tensor(coordinates, dtype=torch.float64)
    if packed.ndim == 1:
        packed = packed.view(1, -1)
    if packed.ndim != 2 or packed.shape[1] != INTRINSIC_CUE_RANK:
        raise ValueError("degree-two features require two intrinsic coordinates")
    first = packed[:, 0]
    second = packed[:, 1]
    return torch.stack(
        (
            torch.ones_like(first),
            first,
            second,
            first.square(),
            first * second,
            second.square(),
        ),
        dim=1,
    )


def _prepare_cue_plane(raw_cues: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cues = torch.as_tensor(raw_cues, dtype=torch.float64)
    if cues.ndim != 2 or cues.shape[0] < FEATURE_COUNT or cues.shape[1] < 2:
        raise ValueError("low-degree fitting requires at least six cue rows")
    if not torch.isfinite(cues).all():
        raise ValueError("raw cues must be finite")
    mean = cues.mean(dim=0)
    centered = cues - mean
    _, singular, right = torch.linalg.svd(centered, full_matrices=False)
    rank, _ = _relative_rank(centered)
    if rank != INTRINSIC_CUE_RANK:
        raise RuntimeError("raw cue observations do not identify a rank-two plane")
    basis = right[:INTRINSIC_CUE_RANK].T.contiguous()
    return mean, basis, singular[:INTRINSIC_CUE_RANK]


def train_low_degree_content_gate(
    raw_cues: torch.Tensor,
    observed_content: torch.Tensor,
) -> LowDegreeContentGateSnapshot:
    """Fit one full degree-two vector operator on an opaque cue plane."""
    cues = torch.as_tensor(raw_cues, dtype=torch.float64)
    content = torch.as_tensor(observed_content, dtype=torch.float64)
    if content.ndim != 2 or content.shape != (cues.shape[0], CONTENT_DIMENSION):
        raise ValueError("content observations must align with cue rows")
    if not torch.isfinite(content).all():
        raise ValueError("content observations must be finite")
    cue_mean, cue_basis, cue_singular = _prepare_cue_plane(cues)
    coordinates = (cues - cue_mean) @ cue_basis
    design = _quadratic_features(coordinates)
    _, feature_singular, feature_right = torch.linalg.svd(
        design, full_matrices=False
    )
    feature_rank, _ = _relative_rank(design)
    if feature_rank != FEATURE_COUNT:
        raise RuntimeError("degree-two feature design is rank deficient")
    condition = float((feature_singular[0] / feature_singular[-1]).item())
    if condition > MAX_CONDITION_NUMBER:
        raise RuntimeError("degree-two feature design is ill conditioned")
    coefficients = torch.linalg.pinv(
        design,
        atol=0.0,
        rtol=RANK_RELATIVE_TOLERANCE,
    ) @ content
    fitted = design @ coefficients
    fit_error = _relative_residual(fitted - content, content)
    if fit_error > MAX_RELATIVE_FIT_ERROR:
        raise RuntimeError("observed content is outside the degree-two class")
    return LowDegreeContentGateSnapshot(
        cue_mean=cue_mean,
        cue_basis=cue_basis,
        coefficients=coefficients,
        cue_singular_values=cue_singular,
        feature_singular_values=feature_singular,
        feature_right_basis=feature_right[:feature_rank].T.contiguous(),
        cue_rank=INTRINSIC_CUE_RANK,
        feature_rank=feature_rank,
        condition_number=condition,
        relative_fit_error=fit_error,
        training_count=int(cues.shape[0]),
    )


def low_degree_prediction_receipt(
    gate: LowDegreeContentGateSnapshot,
    raw_cue: torch.Tensor,
) -> dict[str, Any]:
    cue = torch.as_tensor(raw_cue, dtype=torch.float64).view(-1)
    if cue.shape != gate.cue_mean.shape or not torch.isfinite(cue).all():
        raise ValueError("low-degree query cue has the wrong shape or is nonfinite")
    centered = cue - gate.cue_mean
    projection = gate.cue_basis @ (gate.cue_basis.T @ centered)
    span_scale = max(
        1.0,
        float(torch.linalg.vector_norm(centered).item()),
        float(torch.linalg.vector_norm(gate.cue_mean).item()),
    )
    span_error = float(torch.linalg.vector_norm(centered - projection).item()) / span_scale
    if span_error > MAX_RELATIVE_SPAN_ERROR:
        raise RuntimeError("low-degree query cue is outside the observed cue plane")
    coordinates = centered @ gate.cue_basis
    feature = _quadratic_features(coordinates).view(-1)
    feature_projection = gate.feature_right_basis @ (
        gate.feature_right_basis.T @ feature
    )
    feature_span_error = _relative_residual(feature - feature_projection, feature)
    return {
        "prediction": feature @ gate.coefficients,
        "relative_cue_span_error": span_error,
        "relative_feature_span_error": feature_span_error,
    }


def predict_low_degree_content(
    gate: LowDegreeContentGateSnapshot,
    raw_cue: torch.Tensor,
) -> torch.Tensor:
    return low_degree_prediction_receipt(gate, raw_cue)["prediction"]


def _bind_prediction_to_current_packet(
    prediction: torch.Tensor,
    arrived_packet_indices: Sequence[int],
    weight: torch.Tensor,
    response_indices: Sequence[int],
) -> dict[str, Any]:
    packet_indices = tuple(int(value) for value in arrived_packet_indices)
    response = tuple(int(value) for value in response_indices)
    packed = torch.as_tensor(weight, dtype=torch.float64)
    if packed.ndim != 2 or packed.shape[0] != packed.shape[1]:
        raise ValueError("current-packet binding requires a square weight matrix")
    if len(packet_indices) < 3 or len(set(packet_indices)) != len(packet_indices):
        raise ValueError("current-packet binding requires three distinct arrivals")
    rows = torch.tensor(response, dtype=torch.long)
    columns = torch.tensor(packet_indices, dtype=torch.long)
    descriptors = packed.index_select(0, rows).index_select(1, columns).T
    if torch.any(descriptors <= 0.0) or not torch.isfinite(descriptors).all():
        raise RuntimeError("current packet descriptors must be positive and finite")
    predicted = torch.as_tensor(prediction, dtype=torch.float64).view(-1)
    residuals = torch.linalg.vector_norm(descriptors - predicted, dim=1)
    ordered = torch.argsort(residuals, stable=True)
    best = int(ordered[0].item())
    second = int(ordered[1].item())
    scale = max(
        float(torch.linalg.vector_norm(predicted).item()),
        torch.finfo(torch.float64).eps,
    )
    relative_margin = float((residuals[second] - residuals[best]).item()) / scale
    if relative_margin <= MIN_RELATIVE_BINDING_MARGIN:
        raise RuntimeError("current-packet low-degree binding is tied")
    return {
        "selected_index": packet_indices[best],
        "best_residual": float(residuals[best].item()),
        "second_residual": float(residuals[second].item()),
        "relative_binding_margin": relative_margin,
        "predicted_content": [float(value) for value in predicted.tolist()],
    }


def compile_low_degree_packet_index(
    gate: LowDegreeContentGateSnapshot,
    raw_cue: torch.Tensor,
    arrived_packet_indices: Sequence[int],
    weight: torch.Tensor,
    response_indices: Sequence[int],
) -> int:
    prediction = predict_low_degree_content(gate, raw_cue)
    receipt = _bind_prediction_to_current_packet(
        prediction,
        arrived_packet_indices,
        weight,
        response_indices,
    )
    return int(receipt["selected_index"])


def _gate_hash(gate: LowDegreeContentGateSnapshot) -> str:
    digest = hashlib.sha256()
    for tensor in (
        gate.cue_mean,
        gate.cue_basis,
        gate.coefficients,
        gate.cue_singular_values,
        gate.feature_singular_values,
        gate.feature_right_basis,
    ):
        digest.update(tensor.detach().cpu().numpy().tobytes())
    digest.update(
        repr(
            (
                gate.cue_rank,
                gate.feature_rank,
                gate.condition_number,
                gate.relative_fit_error,
                gate.training_count,
            )
        ).encode("ascii")
    )
    return digest.hexdigest()


def _truth_features() -> torch.Tensor:
    points = torch.tensor(
        [(first, second) for first in GRID_LEVELS for second in GRID_LEVELS],
        dtype=torch.float64,
    )
    return _quadratic_features(points)


def _raw_cues(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 610_019)
    chart, _ = torch.linalg.qr(
        torch.randn(CUE_DIMENSION, 3, generator=generator, dtype=torch.float64)
    )
    base = 0.75 * chart[:, 0]
    points = torch.tensor(
        [(first, second) for first in GRID_LEVELS for second in GRID_LEVELS],
        dtype=torch.float64,
    )
    return base + points @ chart[:, 1:3].T


def _coefficient_matrix(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 620_023)
    coefficients = torch.zeros(FEATURE_COUNT, CONTENT_DIMENSION, dtype=torch.float64)
    coefficients[0] = torch.linspace(0.78, 1.18, CONTENT_DIMENSION)
    coefficients[1:3] = 0.055 * torch.randn(
        2, CONTENT_DIMENSION, generator=generator, dtype=torch.float64
    )
    coefficients[3:] = 0.095 * torch.randn(
        3, CONTENT_DIMENSION, generator=generator, dtype=torch.float64
    )
    signs = torch.tensor((1.0, -1.0, 1.0, -1.0, -1.0, 1.0), dtype=torch.float64)
    coefficients[4] += 0.11 * signs
    values = _truth_features() @ coefficients
    minimum = values.min(dim=0).values
    coefficients[0] += (0.30 - minimum).clamp_min(0.0)
    maximum = float((_truth_features() @ coefficients).max().item())
    if maximum > 1.45:
        coefficients *= 1.45 / maximum
    return coefficients


def _tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(
        torch.as_tensor(value).detach().cpu().numpy().tobytes()
    ).hexdigest()


def generate_low_degree_inputs(seeds: Sequence[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    features = _truth_features()
    for seed_value in seeds:
        seed = int(seed_value)
        cues = _raw_cues(seed)
        coefficients = _coefficient_matrix(seed)
        content = features @ coefficients
        quadratic_fraction = float(
            torch.linalg.vector_norm(coefficients[3:]).item()
            / torch.linalg.vector_norm(coefficients).item()
        )
        cue_rank, _ = _relative_rank(cues - cues.mean(dim=0))
        feature_rank, feature_singular = _relative_rank(features)
        gates = {
            "finite": bool(torch.isfinite(cues).all() and torch.isfinite(content).all()),
            "positive_content": bool(torch.all(content > 0.0)),
            "cue_rank_two": cue_rank == INTRINSIC_CUE_RANK,
            "feature_rank_six": feature_rank == FEATURE_COUNT,
            "quadratic_signal_present": quadratic_fraction >= MIN_QUADRATIC_FRACTION,
            "feature_condition_bounded": bool(
                float((feature_singular[0] / feature_singular[-1]).item())
                <= MAX_CONDITION_NUMBER
            ),
        }
        rows.append(
            {
                "seed": seed,
                "raw_cues": [[float(value) for value in row] for row in cues.tolist()],
                "coefficients": [
                    [float(value) for value in row] for row in coefficients.tolist()
                ],
                "cue_sha256": _tensor_hash(cues),
                "coefficient_sha256": _tensor_hash(coefficients),
                "quadratic_fraction": quadratic_fraction,
                "content_minimum": float(content.min().item()),
                "content_maximum": float(content.max().item()),
                "gates": gates,
            }
        )
    ready = bool(rows) and all(all(row["gates"].values()) for row in rows)
    return {
        "status": "LOW_DEGREE_ROTATING_INPUTS_READY" if ready else "LOW_DEGREE_INPUTS_STOP",
        "seed_count": len(rows),
        "rows": rows,
    }


def _runtime_snapshot(content_columns: torch.Tensor) -> Any:
    matrix = torch.as_tensor(content_columns, dtype=torch.float32)
    if matrix.shape != (CONTENT_DIMENSION, CONTENT_DIMENSION):
        raise ValueError("runtime content columns must be 6x6")
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
    )
    return BrainRuntime(weight, config=config, backend="torch", device="cpu").snapshot()


def _external(indices: Sequence[int]) -> torch.Tensor:
    value = torch.zeros(DIMENSION, dtype=torch.float32)
    for index in indices:
        value[int(index)] += EXTERNAL_DRIVE
    return value


def _routed_packet_probe(
    snapshot: Any,
    content_columns: torch.Tensor,
    coordinates: Sequence[int],
    event_roles: Sequence[int],
    selected_indices: Sequence[int],
) -> dict[str, Any]:
    coordinate_map = tuple(int(value) for value in coordinates)
    events = tuple(int(value) for value in event_roles)
    selected = tuple(int(value) for value in selected_indices)
    moved = _snapshot_with_content_columns(snapshot, content_columns, coordinate_map)
    input_pool, hidden, target = _blocks()
    hidden_idx = torch.tensor(hidden, dtype=torch.long)
    input_idx = torch.tensor(input_pool, dtype=torch.long)
    target_idx = torch.tensor(target, dtype=torch.long)
    masked = moved.weight.detach().clone()
    preserved = masked.index_select(0, hidden_idx).index_select(
        1, torch.tensor(selected, dtype=torch.long)
    ).clone()
    masked[hidden_idx[:, None], input_idx] = 0.0
    if selected:
        masked[hidden_idx[:, None], torch.tensor(selected, dtype=torch.long)] = preserved
    runtime = BrainRuntime.from_snapshot(
        replace(moved, weight=masked), backend="torch", device="cpu"
    )
    event_coordinates = tuple(coordinate_map[role] for role in events)
    packet_counts: list[int] = []
    written_counts: list[int] = []
    hidden_first = torch.zeros(CONTENT_DIMENSION)
    target_final = torch.zeros(CONTENT_DIMENSION)
    for tick in range(PAIR_TICKS):
        if runtime._delay_buffer is None:
            raise RuntimeError("rotating content transfer requires the delay ring")
        slot = runtime._delay_idx % runtime.config.max_axon_delay
        packet_counts.append(
            int(
                torch.count_nonzero(
                    runtime._delay_buffer[slot, input_idx].abs()
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
            runtime._delay_buffer[slot, input_idx] = 0.0
        written_counts.append(
            int(
                torch.count_nonzero(
                    runtime._delay_buffer[slot, input_idx].abs()
                    > runtime.config.competition_epsilon
                ).item()
            )
        )
        if tick == 3:
            hidden_first = runtime.activation[hidden_idx].detach().clone()
        if tick == 6:
            target_final = runtime.activation[target_idx].detach().clone()
    return {
        "event_coordinates": list(event_coordinates),
        "selected_indices": list(selected),
        "hidden_first_arrival": [float(value) for value in hidden_first.tolist()],
        "hidden_positive_count": int(
            torch.count_nonzero(hidden_first > runtime.config.competition_epsilon).item()
        ),
        "target_at_6": [float(value) for value in target_final.tolist()],
        "input_packet_count_by_tick": packet_counts,
        "input_written_count_by_tick": written_counts,
        "hippocampal_rows_after": len(runtime.hippocampus),
    }


def _route_error(receipt: dict[str, Any], expected: torch.Tensor) -> float:
    actual = torch.tensor(receipt["target_at_6"], dtype=torch.float64)
    return _relative_residual(actual - expected, expected)


def _affine_prediction(
    raw_cues: torch.Tensor,
    content: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    cues = torch.as_tensor(raw_cues, dtype=torch.float64)
    values = torch.as_tensor(content, dtype=torch.float64)
    mean, basis, _ = _prepare_cue_plane(cues)
    design = torch.cat(
        (torch.ones(cues.shape[0], 1, dtype=torch.float64), (cues - mean) @ basis),
        dim=1,
    )
    coefficients = torch.linalg.pinv(
        design, atol=0.0, rtol=RANK_RELATIVE_TOLERANCE
    ) @ values
    query_coordinate = (torch.as_tensor(query, dtype=torch.float64) - mean) @ basis
    query_design = torch.cat((torch.ones(1, dtype=torch.float64), query_coordinate))
    return query_design @ coefficients


def _candidate_fixture(
    seed: int,
    query_index: int,
    content: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...], int, int]:
    offsets = (0, 1, 7, 11, 13, 17, 19, 23)
    cells: list[int] = []
    for offset in offsets:
        candidate = (int(query_index) + offset) % GRID_CELL_COUNT
        if candidate not in cells:
            cells.append(candidate)
        if len(cells) == CONTENT_DIMENSION:
            break
    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + 630_029 + 1009 * int(query_index)
    )
    first_three = torch.randperm(3, generator=generator).tolist()
    event_cells = [cells[index] for index in first_three]
    event_cells.extend(cells[3:])
    matrix = torch.stack([content[index] for index in event_cells], dim=1)
    input_pool, _, _ = _blocks()
    available = input_pool[1:]
    permutation = torch.randperm(len(available), generator=generator)[:CONTENT_DIMENSION]
    coordinates = tuple(available[index] for index in permutation.tolist())
    correct_role = event_cells.index(int(query_index))
    wrong_role = event_cells.index((int(query_index) + 1) % GRID_CELL_COUNT)
    return matrix, coordinates, tuple(event_cells[:3]), correct_role, wrong_role


def _compile_or_abstain(
    prediction: torch.Tensor,
    arrived: Sequence[int],
    weight: torch.Tensor,
    hidden: Sequence[int],
) -> tuple[int | None, str | None]:
    try:
        receipt = _bind_prediction_to_current_packet(prediction, arrived, weight, hidden)
        return int(receipt["selected_index"]), None
    except (RuntimeError, ValueError) as exc:
        return None, str(exc)


def analyze_rotating_low_degree_row(
    seed: int,
    raw_cues: torch.Tensor,
    coefficients: torch.Tensor,
) -> dict[str, Any]:
    seed = int(seed)
    cues = torch.as_tensor(raw_cues, dtype=torch.float64)
    coefficient_matrix = torch.as_tensor(coefficients, dtype=torch.float64)
    if cues.shape != (GRID_CELL_COUNT, CUE_DIMENSION):
        raise ValueError("rotating raw cue input has the wrong shape")
    if coefficient_matrix.shape != (FEATURE_COUNT, CONTENT_DIMENSION):
        raise ValueError("rotating coefficient input has the wrong shape")
    content = _truth_features() @ coefficient_matrix
    quadratic_fraction = float(
        torch.linalg.vector_norm(coefficient_matrix[3:]).item()
        / torch.linalg.vector_norm(coefficient_matrix).item()
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 640_031)
    rotation_order = torch.randperm(GRID_CELL_COUNT, generator=generator).tolist()
    fold_receipts: list[dict[str, Any]] = []
    all_gates: list[bool] = []
    affine_errors: list[float] = []
    affine_binding_successes = 0
    minimum_binding_margin = float("inf")
    minimum_route_separation = float("inf")
    maximum_query_error = 0.0
    maximum_fit_error = 0.0
    maximum_chart_error = 0.0
    maximum_row_order_error = 0.0
    maximum_candidate_norm_ratio = 0.0
    association_shuffle_rejections = 0
    expected_packet = [0, 0, 0, 3, 0, 0, 0]
    expected_written = [0, 3, 0, 0, 0, 0, 0]

    for fold_number, query_index in enumerate(rotation_order):
        training_indices = [index for index in range(GRID_CELL_COUNT) if index != query_index]
        order_generator = torch.Generator(device="cpu").manual_seed(
            seed + 650_033 + 1009 * fold_number
        )
        order = torch.randperm(len(training_indices), generator=order_generator)
        index_tensor = torch.tensor(training_indices, dtype=torch.long).index_select(0, order)
        training_cues = cues.index_select(0, index_tensor)
        training_content = content.index_select(0, index_tensor)
        gate = train_low_degree_content_gate(training_cues, training_content)
        gate_before = _gate_hash(gate)
        prediction_receipt = low_degree_prediction_receipt(gate, cues[query_index])
        prediction = prediction_receipt["prediction"]
        query_error = _relative_residual(
            prediction - content[query_index], content[query_index]
        )
        affine_prediction = _affine_prediction(
            training_cues, training_content, cues[query_index]
        )
        affine_error = _relative_residual(
            affine_prediction - content[query_index], content[query_index]
        )
        affine_errors.append(affine_error)
        maximum_query_error = max(maximum_query_error, query_error)
        maximum_fit_error = max(maximum_fit_error, gate.relative_fit_error)

        reverse = torch.arange(len(training_indices) - 1, -1, -1)
        reordered = train_low_degree_content_gate(
            training_cues.index_select(0, reverse),
            training_content.index_select(0, reverse),
        )
        row_order_error = _relative_residual(
            predict_low_degree_content(reordered, cues[query_index]) - prediction,
            prediction,
        )
        maximum_row_order_error = max(maximum_row_order_error, row_order_error)

        chart_generator = torch.Generator(device="cpu").manual_seed(
            seed + 660_037 + 1009 * fold_number
        )
        chart, _ = torch.linalg.qr(
            torch.randn(CUE_DIMENSION, CUE_DIMENSION, generator=chart_generator, dtype=torch.float64)
        )
        chart_gate = train_low_degree_content_gate(training_cues @ chart.T, training_content)
        chart_prediction = predict_low_degree_content(
            chart_gate, cues[query_index] @ chart.T
        )
        chart_error = _relative_residual(chart_prediction - prediction, prediction)
        maximum_chart_error = max(maximum_chart_error, chart_error)

        shuffle_rejected = False
        try:
            train_low_degree_content_gate(training_cues, training_content.roll(1, dims=0))
        except (RuntimeError, ValueError):
            shuffle_rejected = True
            association_shuffle_rejections += 1

        matrix, coordinates, event_cells, correct_role, wrong_role = _candidate_fixture(
            seed, query_index, content
        )
        base_snapshot = _runtime_snapshot(matrix)
        moved = _snapshot_with_content_columns(base_snapshot, matrix, coordinates)
        _, hidden, _ = _blocks()
        arrived = tuple(coordinates[index] for index in range(3))
        binding = _bind_prediction_to_current_packet(
            prediction, arrived, moved.weight, hidden
        )
        learned_index = int(binding["selected_index"])
        oracle_index = int(coordinates[correct_role])
        wrong_index = int(coordinates[wrong_role])
        minimum_binding_margin = min(
            minimum_binding_margin, float(binding["relative_binding_margin"])
        )
        candidate_norms = matrix[:, :3].norm(dim=0)
        norm_ratio = float((candidate_norms.max() / candidate_norms.min()).item())
        maximum_candidate_norm_ratio = max(maximum_candidate_norm_ratio, norm_ratio)

        oracle = _routed_packet_probe(
            base_snapshot, matrix, coordinates, (0, 1, 2), (oracle_index,)
        )
        oracle_target = torch.tensor(oracle["target_at_6"], dtype=torch.float64)
        learned = _routed_packet_probe(
            base_snapshot, matrix, coordinates, (0, 1, 2), (learned_index,)
        )
        wrong = _routed_packet_probe(
            base_snapshot, matrix, coordinates, (0, 1, 2), (wrong_index,)
        )
        wrong_error = _route_error(wrong, oracle_target)
        minimum_route_separation = min(minimum_route_separation, wrong_error)

        affine_index, _ = _compile_or_abstain(
            affine_prediction, arrived, moved.weight, hidden
        )
        if affine_index is not None:
            affine_probe = _routed_packet_probe(
                base_snapshot, matrix, coordinates, (0, 1, 2), (affine_index,)
            )
            affine_success = _route_error(affine_probe, oracle_target) <= MAX_ROUTE_ERROR
        else:
            affine_success = False
        affine_binding_successes += int(affine_success)

        wrong_cue_prediction = predict_low_degree_content(
            gate, cues[(query_index + 1) % GRID_CELL_COUNT]
        )
        wrong_cue_index, _ = _compile_or_abstain(
            wrong_cue_prediction, arrived, moved.weight, hidden
        )
        wrong_cue_success = False
        if wrong_cue_index is not None:
            wrong_cue_probe = _routed_packet_probe(
                base_snapshot, matrix, coordinates, (0, 1, 2), (wrong_cue_index,)
            )
            wrong_cue_success = _route_error(wrong_cue_probe, oracle_target) <= MAX_ROUTE_ERROR

        shuffled_weight = moved.weight.detach().clone()
        hidden_idx = torch.tensor(hidden, dtype=torch.long)
        arrived_idx = torch.tensor(arrived, dtype=torch.long)
        original = shuffled_weight[hidden_idx[:, None], arrived_idx].clone()
        shuffled_weight[hidden_idx[:, None], arrived_idx] = original[:, [1, 2, 0]]
        shuffled_index, _ = _compile_or_abstain(
            prediction, arrived, shuffled_weight, hidden
        )
        binding_shuffle_success = False
        if shuffled_index is not None:
            shuffled_probe = _routed_packet_probe(
                base_snapshot, matrix, coordinates, (0, 1, 2), (shuffled_index,)
            )
            binding_shuffle_success = (
                _route_error(shuffled_probe, oracle_target) <= MAX_ROUTE_ERROR
            )

        input_pool, _, _ = _blocks()
        canonical_probe = _routed_packet_probe(
            base_snapshot, matrix, coordinates, (0, 1, 2), (input_pool[0],)
        )
        no_context_probe = _routed_packet_probe(
            base_snapshot, matrix, coordinates, (0, 1, 2), arrived
        )
        alternative_delta = 0.20 * torch.tensor(
            (1.0, -0.8, 0.6, -0.4, 0.2, -0.1), dtype=torch.float64
        )
        alternative = content[query_index] + alternative_delta
        alternative_error = _relative_residual(prediction - alternative, alternative)
        lookup_scale = max(
            1.0,
            float(torch.linalg.vector_norm(cues[query_index]).item()),
        )
        lookup_abstains = bool(
            torch.all(
                torch.linalg.vector_norm(training_cues - cues[query_index], dim=1)
                > EXACT_LOOKUP_TOLERANCE * lookup_scale
            )
        )
        fold_gates = {
            "cue_rank_two": gate.cue_rank == INTRINSIC_CUE_RANK,
            "feature_rank_six": gate.feature_rank == FEATURE_COUNT,
            "condition_bounded": gate.condition_number <= MAX_CONDITION_NUMBER,
            "training_fit": gate.relative_fit_error <= MAX_RELATIVE_FIT_ERROR,
            "query_cue_in_span": (
                prediction_receipt["relative_cue_span_error"] <= MAX_RELATIVE_SPAN_ERROR
            ),
            "query_feature_identified": (
                prediction_receipt["relative_feature_span_error"] <= MAX_RELATIVE_SPAN_ERROR
            ),
            "conditional_query_exact": query_error <= MAX_RELATIVE_QUERY_ERROR,
            "binding_unique": (
                binding["relative_binding_margin"] > MIN_RELATIVE_BINDING_MARGIN
            ),
            "learned_selects_oracle_coordinate": learned_index == oracle_index,
            "learned_oracle_runtime_equal": _route_error(learned, oracle_target) <= MAX_ROUTE_ERROR,
            "wrong_current_packet_separated": wrong_error >= MIN_ROUTE_SEPARATION,
            "wrong_cue_fails": not wrong_cue_success,
            "binding_shuffle_fails": not binding_shuffle_success,
            "canonical_coordinate_fails": (
                _route_error(canonical_probe, oracle_target) > MAX_ROUTE_ERROR
            ),
            "no_context_all_packet_fails": (
                _route_error(no_context_probe, oracle_target) > MAX_ROUTE_ERROR
            ),
            "finite_lookup_abstains": lookup_abstains,
            "association_shuffle_rejected": shuffle_rejected,
            "row_order_invariant": row_order_error <= 1e-10,
            "orthogonal_chart_invariant": chart_error <= 1e-10,
            "candidate_scales_matched": norm_ratio <= MAX_CANDIDATE_NORM_RATIO,
            "one_shot_three_packet_receipt": (
                learned["input_packet_count_by_tick"] == expected_packet
                and learned["input_written_count_by_tick"] == expected_written
            ),
            "stores_zero": (
                learned["hippocampal_rows_after"] == 0
                and oracle["hippocampal_rows_after"] == 0
            ),
            "query_only_delta_nonidentifiable": (
                alternative_error >= 0.01 and query_error <= MAX_RELATIVE_QUERY_ERROR
            ),
            "gate_frozen": gate_before == _gate_hash(gate),
        }
        all_gates.append(all(fold_gates.values()))
        fold_receipts.append(
            {
                "query_index": int(query_index),
                "event_cells": [int(value) for value in event_cells],
                "query_error": query_error,
                "affine_error": affine_error,
                "affine_binding_success": affine_success,
                "binding_margin": float(binding["relative_binding_margin"]),
                "route_separation": wrong_error,
                "candidate_norm_ratio": norm_ratio,
                "alternative_query_error": alternative_error,
                "gates": fold_gates,
            }
        )

    affine_mean_error = sum(affine_errors) / len(affine_errors)
    affine_success_fraction = affine_binding_successes / GRID_CELL_COUNT
    aggregate_gates = {
        "all_rotating_folds_pass": all(all_gates),
        "all_twenty_five_cells_held_out_once": set(rotation_order) == set(range(GRID_CELL_COUNT)),
        "quadratic_signal_present": quadratic_fraction >= MIN_QUADRATIC_FRACTION,
        "affine_prediction_separated": affine_mean_error >= MIN_AFFINE_MEAN_ERROR,
        "affine_binding_loses": (
            affine_success_fraction <= MAX_AFFINE_BINDING_SUCCESS_FRACTION
        ),
        "finite_lookup_abstains_all": all(
            fold["gates"]["finite_lookup_abstains"] for fold in fold_receipts
        ),
        "association_shuffle_rejected_all": (
            association_shuffle_rejections == GRID_CELL_COUNT
        ),
    }
    status = (
        "ROTATING_LOW_DEGREE_CONTENT_PASS"
        if all(aggregate_gates.values())
        else "ROTATING_LOW_DEGREE_CONTENT_STOP"
    )
    return {
        "seed": seed,
        "status": status,
        "gates": aggregate_gates,
        "rotation_order": [int(value) for value in rotation_order],
        "quadratic_fraction": quadratic_fraction,
        "maximum_query_error": maximum_query_error,
        "maximum_fit_error": maximum_fit_error,
        "maximum_row_order_error": maximum_row_order_error,
        "maximum_orthogonal_chart_error": maximum_chart_error,
        "minimum_binding_margin": minimum_binding_margin,
        "minimum_route_separation": minimum_route_separation,
        "maximum_candidate_norm_ratio": maximum_candidate_norm_ratio,
        "affine_mean_error": affine_mean_error,
        "affine_binding_success_fraction": affine_success_fraction,
        "association_shuffle_rejections": association_shuffle_rejections,
        "folds": fold_receipts,
        "cue_sha256": _tensor_hash(cues),
        "coefficient_sha256": _tensor_hash(coefficient_matrix),
        "endpoint_opened": True,
        "confirmation_opened": False,
        "claim_scope": (
            "conditional synthetic full degree-two transfer on a learned rank-two "
            "cue plane with rotating heldout cells and current-packet binding"
        ),
    }


def analyze_rotating_low_degree_artifact(
    path: str | Path,
    *,
    stage: str,
) -> dict[str, Any]:
    if stage not in {"calibration", "development"}:
        raise ValueError("stage must be calibration or development")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "LOW_DEGREE_ROTATING_INPUTS_READY":
        raise RuntimeError("rotating low-degree inputs did not pass producer gates")
    expected = CALIBRATION_SEEDS if stage == "calibration" else DEVELOPMENT_SEEDS
    actual = tuple(int(row["seed"]) for row in payload.get("rows", ()))
    if actual != expected:
        raise RuntimeError("rotating low-degree seed order does not match the frozen stage")
    rows = [
        analyze_rotating_low_degree_row(
            int(row["seed"]),
            torch.tensor(row["raw_cues"], dtype=torch.float64),
            torch.tensor(row["coefficients"], dtype=torch.float64),
        )
        for row in payload["rows"]
    ]
    pass_count = sum(row["status"] == "ROTATING_LOW_DEGREE_CONTENT_PASS" for row in rows)
    passed = pass_count == len(expected)
    return {
        "status": (
            "ROTATING_LOW_DEGREE_CALIBRATION_PASS"
            if passed and stage == "calibration"
            else "ROTATING_LOW_DEGREE_DEVELOPMENT_GO"
            if passed
            else "ROTATING_LOW_DEGREE_STOP"
        ),
        "stage": stage,
        "seed_count": len(rows),
        "pass_count": pass_count,
        "rotating_fold_count": len(rows) * GRID_CELL_COUNT,
        "maximum_query_error": max(row["maximum_query_error"] for row in rows),
        "maximum_fit_error": max(row["maximum_fit_error"] for row in rows),
        "minimum_binding_margin": min(row["minimum_binding_margin"] for row in rows),
        "minimum_route_separation": min(row["minimum_route_separation"] for row in rows),
        "maximum_candidate_norm_ratio": max(
            row["maximum_candidate_norm_ratio"] for row in rows
        ),
        "minimum_affine_mean_error": min(row["affine_mean_error"] for row in rows),
        "maximum_affine_binding_success_fraction": max(
            row["affine_binding_success_fraction"] for row in rows
        ),
        "association_shuffle_rejection_total": sum(
            row["association_shuffle_rejections"] for row in rows
        ),
        "endpoint_opened": any(row["endpoint_opened"] for row in rows),
        "confirmation_opened": False,
        "rows": rows,
    }


__all__ = [
    "CALIBRATION_SEEDS",
    "DEVELOPMENT_SEEDS",
    "LowDegreeContentGateSnapshot",
    "train_low_degree_content_gate",
    "low_degree_prediction_receipt",
    "predict_low_degree_content",
    "compile_low_degree_packet_index",
    "generate_low_degree_inputs",
    "analyze_rotating_low_degree_row",
    "analyze_rotating_low_degree_artifact",
]
