"""BA-TR29: low-degree routing under nearest and affine hard negatives."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .runtime_rotating_low_degree_content_transfer import (
    CONTENT_DIMENSION,
    CUE_DIMENSION,
    EXACT_LOOKUP_TOLERANCE,
    FEATURE_COUNT,
    GRID_CELL_COUNT,
    MAX_RELATIVE_FIT_ERROR,
    MAX_RELATIVE_QUERY_ERROR,
    MAX_ROUTE_ERROR,
    _affine_prediction,
    _bind_prediction_to_current_packet,
    _blocks,
    _gate_hash,
    _relative_residual,
    _routed_packet_probe,
    _runtime_snapshot,
    _tensor_hash,
    _truth_features,
    generate_low_degree_inputs,
    predict_low_degree_content,
    train_low_degree_content_gate,
)
from .runtime_3x3_unlabeled_content_transfer import _snapshot_with_content_columns


CALIBRATION_SEEDS = (116002,)
DEVELOPMENT_SEEDS = tuple(range(116101, 116117))
MIN_MODEL_SEPARATION = 1e-2
MIN_PANEL_COMPONENT = 0.1
MAX_HARD_PANEL_NORM_RATIO = 1.75
MAX_NEAREST_PANEL_NORM_RATIO = 1.25
MIN_HARD_BINDING_MARGIN = 1e-3
MIN_NEAREST_BINDING_MARGIN = 5e-3
MIN_NEAREST_DESCRIPTOR_SEPARATION = 5e-3
MIN_ROUTE_SEPARATION = 1e-4
MAX_NEAREST_AFFINE_SUCCESS_FRACTION = 0.25


def _coordinate_map(seed: int, query_index: int, panel: int) -> tuple[int, ...]:
    input_pool, _, _ = _blocks()
    available = input_pool[1:]
    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + 710_021 + 1009 * int(query_index) + 104_729 * int(panel)
    )
    order = torch.randperm(len(available), generator=generator)[:CONTENT_DIMENSION]
    return tuple(available[index] for index in order.tolist())


def _hard_panel(
    seed: int,
    query_index: int,
    truth: torch.Tensor,
    affine: torch.Tensor,
    content: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...], int, int, int]:
    truth_value = torch.as_tensor(truth, dtype=torch.float64)
    affine_value = torch.as_tensor(affine, dtype=torch.float64)
    midpoint = 0.5 * (truth_value + affine_value)
    panel = torch.stack((truth_value, affine_value, midpoint), dim=1)
    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + 720_023 + 1009 * int(query_index)
    )
    order = torch.randperm(3, generator=generator)
    panel = panel.index_select(1, order)
    labels = order.tolist()
    extra = []
    for offset in (1, 7, 11, 13, 17, 19):
        candidate = (int(query_index) + offset) % GRID_CELL_COUNT
        if candidate != int(query_index) and candidate not in extra:
            extra.append(candidate)
        if len(extra) == 3:
            break
    panel = torch.cat((panel, content[torch.tensor(extra, dtype=torch.long)].T), dim=1)
    coordinates = _coordinate_map(seed, query_index, 0)
    return (
        panel,
        coordinates,
        labels.index(0),
        labels.index(1),
        labels.index(2),
    )


def _nearest_panel(
    seed: int,
    query_index: int,
    content: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...], int, int]:
    distances = torch.linalg.vector_norm(content - content[int(query_index)], dim=1)
    nearest = torch.argsort(distances, stable=True)[:CONTENT_DIMENSION]
    cells = tuple(int(value) for value in nearest.tolist())
    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + 730_027 + 1009 * int(query_index)
    )
    order = torch.randperm(CONTENT_DIMENSION, generator=generator)
    cells = tuple(cells[index] for index in order.tolist())
    matrix = torch.stack([content[index] for index in cells], dim=1)
    coordinates = _coordinate_map(seed, query_index, 1)
    return matrix, coordinates, cells, cells.index(int(query_index)), cells.index(int(nearest[1]))


def _probe_error(receipt: dict[str, Any], expected: torch.Tensor) -> float:
    actual = torch.tensor(receipt["target_at_6"], dtype=torch.float64)
    return _relative_residual(actual - expected, expected)


def _compile(
    prediction: torch.Tensor,
    arrived: Sequence[int],
    weight: torch.Tensor,
    hidden: Sequence[int],
) -> tuple[int | None, dict[str, Any] | None]:
    try:
        receipt = _bind_prediction_to_current_packet(
            prediction, arrived, weight, hidden
        )
        return int(receipt["selected_index"]), receipt
    except (RuntimeError, ValueError):
        return None, None


def analyze_low_degree_hard_negative_row(
    seed: int,
    raw_cues: torch.Tensor,
    coefficients: torch.Tensor,
) -> dict[str, Any]:
    seed = int(seed)
    cues = torch.as_tensor(raw_cues, dtype=torch.float64)
    coefficient_matrix = torch.as_tensor(coefficients, dtype=torch.float64)
    if cues.shape != (GRID_CELL_COUNT, CUE_DIMENSION):
        raise ValueError("hard-negative cue input has the wrong shape")
    if coefficient_matrix.shape != (FEATURE_COUNT, CONTENT_DIMENSION):
        raise ValueError("hard-negative coefficient input has the wrong shape")
    content = _truth_features() @ coefficient_matrix
    rotation_generator = torch.Generator(device="cpu").manual_seed(seed + 740_029)
    rotation_order = torch.randperm(GRID_CELL_COUNT, generator=rotation_generator).tolist()
    fold_rows: list[dict[str, Any]] = []
    fold_passes: list[bool] = []
    hard_affine_decoy_selections = 0
    hard_affine_truth_selections = 0
    nearest_affine_truth_selections = 0
    association_shuffle_rejections = 0
    maximum_query_error = 0.0
    minimum_model_separation = float("inf")
    minimum_hard_margin = float("inf")
    minimum_nearest_margin = float("inf")
    minimum_nearest_separation = float("inf")
    minimum_route_separation = float("inf")
    maximum_hard_norm_ratio = 0.0
    maximum_nearest_norm_ratio = 0.0
    expected_hard_packet = [0, 0, 0, 3, 0, 0, 0]
    expected_hard_written = [0, 3, 0, 0, 0, 0, 0]
    expected_nearest_packet = [0, 0, 0, 6, 0, 0, 0]
    expected_nearest_written = [0, 6, 0, 0, 0, 0, 0]

    for fold_number, query_index in enumerate(rotation_order):
        training = [index for index in range(GRID_CELL_COUNT) if index != query_index]
        order_generator = torch.Generator(device="cpu").manual_seed(
            seed + 750_031 + 1009 * fold_number
        )
        order = torch.randperm(len(training), generator=order_generator)
        indices = torch.tensor(training, dtype=torch.long).index_select(0, order)
        training_cues = cues.index_select(0, indices)
        training_content = content.index_select(0, indices)
        gate = train_low_degree_content_gate(training_cues, training_content)
        gate_before = _gate_hash(gate)
        prediction = predict_low_degree_content(gate, cues[query_index])
        affine = _affine_prediction(training_cues, training_content, cues[query_index])
        truth = content[query_index]
        query_error = _relative_residual(prediction - truth, truth)
        separation_scale = max(
            float(torch.linalg.vector_norm(truth).item()),
            float(torch.linalg.vector_norm(affine).item()),
            torch.finfo(torch.float64).eps,
        )
        model_separation = float(torch.linalg.vector_norm(affine - truth).item()) / separation_scale
        maximum_query_error = max(maximum_query_error, query_error)
        minimum_model_separation = min(minimum_model_separation, model_separation)

        shuffle_rejected = False
        try:
            train_low_degree_content_gate(training_cues, training_content.roll(1, dims=0))
        except (RuntimeError, ValueError):
            shuffle_rejected = True
            association_shuffle_rejections += 1

        hard_matrix, hard_coordinates, truth_role, affine_role, skew_role = _hard_panel(
            seed, query_index, truth, affine, content
        )
        hard_snapshot = _runtime_snapshot(hard_matrix)
        hard_moved = _snapshot_with_content_columns(
            hard_snapshot, hard_matrix, hard_coordinates
        )
        _, hidden, _ = _blocks()
        hard_arrived = tuple(hard_coordinates[index] for index in range(3))
        learned_hard_index, learned_hard_binding = _compile(
            prediction, hard_arrived, hard_moved.weight, hidden
        )
        affine_hard_index, affine_hard_binding = _compile(
            affine, hard_arrived, hard_moved.weight, hidden
        )
        oracle_hard_index = hard_coordinates[truth_role]
        affine_decoy_index = hard_coordinates[affine_role]
        skew_index = hard_coordinates[skew_role]
        if affine_hard_index == affine_decoy_index:
            hard_affine_decoy_selections += 1
        if affine_hard_index == oracle_hard_index:
            hard_affine_truth_selections += 1
        hard_norms = hard_matrix[:, :3].norm(dim=0)
        hard_norm_ratio = float((hard_norms.max() / hard_norms.min()).item())
        maximum_hard_norm_ratio = max(maximum_hard_norm_ratio, hard_norm_ratio)
        hard_margin = min(
            float(learned_hard_binding["relative_binding_margin"])
            if learned_hard_binding is not None
            else 0.0,
            float(affine_hard_binding["relative_binding_margin"])
            if affine_hard_binding is not None
            else 0.0,
        )
        minimum_hard_margin = min(minimum_hard_margin, hard_margin)
        oracle_hard = _routed_packet_probe(
            hard_snapshot, hard_matrix, hard_coordinates, (0, 1, 2), (oracle_hard_index,)
        )
        oracle_hard_target = torch.tensor(oracle_hard["target_at_6"], dtype=torch.float64)
        learned_hard = _routed_packet_probe(
            hard_snapshot,
            hard_matrix,
            hard_coordinates,
            (0, 1, 2),
            (learned_hard_index,) if learned_hard_index is not None else (),
        )
        affine_hard_probe = _routed_packet_probe(
            hard_snapshot,
            hard_matrix,
            hard_coordinates,
            (0, 1, 2),
            (affine_hard_index,) if affine_hard_index is not None else (),
        )
        skew_probe = _routed_packet_probe(
            hard_snapshot, hard_matrix, hard_coordinates, (0, 1, 2), (skew_index,)
        )
        hard_route_separation = min(
            _probe_error(affine_hard_probe, oracle_hard_target),
            _probe_error(skew_probe, oracle_hard_target),
        )
        minimum_route_separation = min(
            minimum_route_separation, hard_route_separation
        )

        near_matrix, near_coordinates, near_cells, near_truth_role, near_wrong_role = _nearest_panel(
            seed, query_index, content
        )
        near_snapshot = _runtime_snapshot(near_matrix)
        near_moved = _snapshot_with_content_columns(
            near_snapshot, near_matrix, near_coordinates
        )
        near_arrived = tuple(near_coordinates)
        learned_near_index, learned_near_binding = _compile(
            prediction, near_arrived, near_moved.weight, hidden
        )
        affine_near_index, _ = _compile(
            affine, near_arrived, near_moved.weight, hidden
        )
        oracle_near_index = near_coordinates[near_truth_role]
        wrong_near_index = near_coordinates[near_wrong_role]
        nearest_affine_truth_selections += int(affine_near_index == oracle_near_index)
        near_norms = near_matrix.norm(dim=0)
        near_norm_ratio = float((near_norms.max() / near_norms.min()).item())
        maximum_nearest_norm_ratio = max(maximum_nearest_norm_ratio, near_norm_ratio)
        nearest_margin = (
            float(learned_near_binding["relative_binding_margin"])
            if learned_near_binding is not None
            else 0.0
        )
        minimum_nearest_margin = min(minimum_nearest_margin, nearest_margin)
        nearest_descriptor_separation = _relative_residual(
            content[near_cells[near_wrong_role]] - truth, truth
        )
        minimum_nearest_separation = min(
            minimum_nearest_separation, nearest_descriptor_separation
        )
        oracle_near = _routed_packet_probe(
            near_snapshot,
            near_matrix,
            near_coordinates,
            tuple(range(CONTENT_DIMENSION)),
            (oracle_near_index,),
        )
        oracle_near_target = torch.tensor(oracle_near["target_at_6"], dtype=torch.float64)
        learned_near = _routed_packet_probe(
            near_snapshot,
            near_matrix,
            near_coordinates,
            tuple(range(CONTENT_DIMENSION)),
            (learned_near_index,) if learned_near_index is not None else (),
        )
        wrong_cue_prediction = predict_low_degree_content(
            gate, cues[near_cells[near_wrong_role]]
        )
        wrong_cue_index, _ = _compile(
            wrong_cue_prediction, near_arrived, near_moved.weight, hidden
        )
        wrong_cue_probe = _routed_packet_probe(
            near_snapshot,
            near_matrix,
            near_coordinates,
            tuple(range(CONTENT_DIMENSION)),
            (wrong_cue_index,) if wrong_cue_index is not None else (),
        )

        shuffled_weight = near_moved.weight.detach().clone()
        hidden_idx = torch.tensor(hidden, dtype=torch.long)
        arrived_idx = torch.tensor(near_arrived, dtype=torch.long)
        original = shuffled_weight[hidden_idx[:, None], arrived_idx].clone()
        shuffled_weight[hidden_idx[:, None], arrived_idx] = original.roll(1, dims=1)
        shuffled_index, _ = _compile(
            prediction, near_arrived, shuffled_weight, hidden
        )
        shuffled_probe = _routed_packet_probe(
            near_snapshot,
            near_matrix,
            near_coordinates,
            tuple(range(CONTENT_DIMENSION)),
            (shuffled_index,) if shuffled_index is not None else (),
        )
        input_pool, _, _ = _blocks()
        canonical_probe = _routed_packet_probe(
            near_snapshot,
            near_matrix,
            near_coordinates,
            tuple(range(CONTENT_DIMENSION)),
            (input_pool[0],),
        )
        no_context_probe = _routed_packet_probe(
            near_snapshot,
            near_matrix,
            near_coordinates,
            tuple(range(CONTENT_DIMENSION)),
            near_arrived,
        )
        alternative = truth + 0.20 * torch.tensor(
            (1.0, -0.8, 0.6, -0.4, 0.2, -0.1), dtype=torch.float64
        )
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
            "training_fit": gate.relative_fit_error <= MAX_RELATIVE_FIT_ERROR,
            "conditional_query_exact": query_error <= MAX_RELATIVE_QUERY_ERROR,
            "model_separation_present": model_separation >= MIN_MODEL_SEPARATION,
            "hard_panel_positive": bool(torch.all(hard_matrix[:, :3] >= MIN_PANEL_COMPONENT)),
            "hard_panel_scale_matched": hard_norm_ratio <= MAX_HARD_PANEL_NORM_RATIO,
            "hard_panel_both_bindings_unique": hard_margin >= MIN_HARD_BINDING_MARGIN,
            "quadratic_selects_hard_truth": learned_hard_index == oracle_hard_index,
            "affine_selects_own_decoy": affine_hard_index == affine_decoy_index,
            "quadratic_hard_route_matches": (
                _probe_error(learned_hard, oracle_hard_target) <= MAX_ROUTE_ERROR
            ),
            "affine_hard_route_differs": (
                _probe_error(affine_hard_probe, oracle_hard_target) >= MIN_ROUTE_SEPARATION
            ),
            "skew_hard_route_differs": (
                _probe_error(skew_probe, oracle_hard_target) >= MIN_ROUTE_SEPARATION
            ),
            "nearest_panel_scale_matched": near_norm_ratio <= MAX_NEAREST_PANEL_NORM_RATIO,
            "nearest_descriptor_separated": (
                nearest_descriptor_separation >= MIN_NEAREST_DESCRIPTOR_SEPARATION
            ),
            "nearest_binding_unique": nearest_margin >= MIN_NEAREST_BINDING_MARGIN,
            "quadratic_selects_nearest_truth": learned_near_index == oracle_near_index,
            "quadratic_nearest_route_matches": (
                _probe_error(learned_near, oracle_near_target) <= MAX_ROUTE_ERROR
            ),
            "wrong_cue_fails": (
                _probe_error(wrong_cue_probe, oracle_near_target) >= MIN_ROUTE_SEPARATION
            ),
            "binding_shuffle_fails": (
                _probe_error(shuffled_probe, oracle_near_target) >= MIN_ROUTE_SEPARATION
            ),
            "canonical_coordinate_fails": (
                _probe_error(canonical_probe, oracle_near_target) >= MIN_ROUTE_SEPARATION
            ),
            "no_context_all_packet_fails": (
                _probe_error(no_context_probe, oracle_near_target) >= MIN_ROUTE_SEPARATION
            ),
            "hard_one_shot_receipt": (
                learned_hard["input_packet_count_by_tick"] == expected_hard_packet
                and learned_hard["input_written_count_by_tick"] == expected_hard_written
            ),
            "nearest_one_shot_receipt": (
                learned_near["input_packet_count_by_tick"] == expected_nearest_packet
                and learned_near["input_written_count_by_tick"] == expected_nearest_written
            ),
            "association_shuffle_rejected": shuffle_rejected,
            "finite_lookup_abstains": lookup_abstains,
            "query_only_delta_nonidentifiable": (
                _relative_residual(prediction - alternative, alternative) >= 0.01
            ),
            "stores_zero": (
                learned_hard["hippocampal_rows_after"] == 0
                and learned_near["hippocampal_rows_after"] == 0
            ),
            "gate_frozen": gate_before == _gate_hash(gate),
        }
        fold_passes.append(all(fold_gates.values()))
        fold_rows.append(
            {
                "query_index": int(query_index),
                "model_separation": model_separation,
                "query_error": query_error,
                "hard_binding_margin": hard_margin,
                "nearest_binding_margin": nearest_margin,
                "nearest_descriptor_separation": nearest_descriptor_separation,
                "hard_norm_ratio": hard_norm_ratio,
                "nearest_norm_ratio": near_norm_ratio,
                "hard_route_separation": hard_route_separation,
                "affine_hard_selected_decoy": affine_hard_index == affine_decoy_index,
                "affine_nearest_selected_truth": affine_near_index == oracle_near_index,
                "gates": fold_gates,
            }
        )

    nearest_affine_fraction = nearest_affine_truth_selections / GRID_CELL_COUNT
    aggregate_gates = {
        "all_rotating_folds_pass": all(fold_passes),
        "all_cells_held_out_once": set(rotation_order) == set(range(GRID_CELL_COUNT)),
        "hard_affine_decoy_25_of_25": hard_affine_decoy_selections == GRID_CELL_COUNT,
        "hard_affine_truth_0_of_25": hard_affine_truth_selections == 0,
        "nearest_affine_loses": (
            nearest_affine_fraction <= MAX_NEAREST_AFFINE_SUCCESS_FRACTION
        ),
        "association_shuffle_rejected_all": (
            association_shuffle_rejections == GRID_CELL_COUNT
        ),
    }
    return {
        "seed": seed,
        "status": (
            "LOW_DEGREE_HARD_NEGATIVE_PASS"
            if all(aggregate_gates.values())
            else "LOW_DEGREE_HARD_NEGATIVE_STOP"
        ),
        "gates": aggregate_gates,
        "rotation_order": [int(value) for value in rotation_order],
        "maximum_query_error": maximum_query_error,
        "minimum_model_separation": minimum_model_separation,
        "minimum_hard_binding_margin": minimum_hard_margin,
        "minimum_nearest_binding_margin": minimum_nearest_margin,
        "minimum_nearest_descriptor_separation": minimum_nearest_separation,
        "minimum_route_separation": minimum_route_separation,
        "maximum_hard_norm_ratio": maximum_hard_norm_ratio,
        "maximum_nearest_norm_ratio": maximum_nearest_norm_ratio,
        "hard_affine_decoy_selections": hard_affine_decoy_selections,
        "hard_affine_truth_selections": hard_affine_truth_selections,
        "nearest_affine_truth_selections": nearest_affine_truth_selections,
        "nearest_affine_truth_fraction": nearest_affine_fraction,
        "association_shuffle_rejections": association_shuffle_rejections,
        "folds": fold_rows,
        "cue_sha256": _tensor_hash(cues),
        "coefficient_sha256": _tensor_hash(coefficient_matrix),
        "endpoint_opened": True,
        "confirmation_opened": False,
        "claim_scope": (
            "conditional synthetic degree-two current-packet discrimination "
            "against nearest and observed-only affine hard negatives"
        ),
    }


def analyze_low_degree_hard_negative_artifact(
    path: str | Path,
    *,
    stage: str,
) -> dict[str, Any]:
    if stage not in {"calibration", "development"}:
        raise ValueError("stage must be calibration or development")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "LOW_DEGREE_ROTATING_INPUTS_READY":
        raise RuntimeError("hard-negative input producer did not pass")
    expected = CALIBRATION_SEEDS if stage == "calibration" else DEVELOPMENT_SEEDS
    actual = tuple(int(row["seed"]) for row in payload.get("rows", ()))
    if actual != expected:
        raise RuntimeError("hard-negative seed order does not match the frozen stage")
    rows = [
        analyze_low_degree_hard_negative_row(
            int(row["seed"]),
            torch.tensor(row["raw_cues"], dtype=torch.float64),
            torch.tensor(row["coefficients"], dtype=torch.float64),
        )
        for row in payload["rows"]
    ]
    pass_count = sum(row["status"] == "LOW_DEGREE_HARD_NEGATIVE_PASS" for row in rows)
    passed = pass_count == len(expected)
    return {
        "status": (
            "LOW_DEGREE_HARD_NEGATIVE_CALIBRATION_PASS"
            if passed and stage == "calibration"
            else "LOW_DEGREE_HARD_NEGATIVE_DEVELOPMENT_GO"
            if passed
            else "LOW_DEGREE_HARD_NEGATIVE_STOP"
        ),
        "stage": stage,
        "seed_count": len(rows),
        "pass_count": pass_count,
        "rotating_fold_count": len(rows) * GRID_CELL_COUNT,
        "maximum_query_error": max(row["maximum_query_error"] for row in rows),
        "minimum_model_separation": min(row["minimum_model_separation"] for row in rows),
        "minimum_hard_binding_margin": min(
            row["minimum_hard_binding_margin"] for row in rows
        ),
        "minimum_nearest_binding_margin": min(
            row["minimum_nearest_binding_margin"] for row in rows
        ),
        "minimum_nearest_descriptor_separation": min(
            row["minimum_nearest_descriptor_separation"] for row in rows
        ),
        "minimum_route_separation": min(row["minimum_route_separation"] for row in rows),
        "maximum_hard_norm_ratio": max(row["maximum_hard_norm_ratio"] for row in rows),
        "maximum_nearest_norm_ratio": max(
            row["maximum_nearest_norm_ratio"] for row in rows
        ),
        "hard_affine_decoy_selection_total": sum(
            row["hard_affine_decoy_selections"] for row in rows
        ),
        "hard_affine_truth_selection_total": sum(
            row["hard_affine_truth_selections"] for row in rows
        ),
        "nearest_affine_truth_selection_total": sum(
            row["nearest_affine_truth_selections"] for row in rows
        ),
        "maximum_nearest_affine_truth_fraction": max(
            row["nearest_affine_truth_fraction"] for row in rows
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
    "generate_low_degree_inputs",
    "analyze_low_degree_hard_negative_row",
    "analyze_low_degree_hard_negative_artifact",
]
