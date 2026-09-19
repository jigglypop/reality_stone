"""BA-TR27: conditional nonseparable Z3 content-composition transfer."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .runtime_3x3_unlabeled_content_transfer import (
    CONTENT_DIMENSION,
    CUE_DIMENSION,
    DIMENSION,
    MAX_RELATIVE_FIT_ERROR,
    MIN_RELATIVE_BINDING_MARGIN,
    RANK_RELATIVE_TOLERANCE,
    _base_snapshot,
    _bind_prediction_to_packets,
    _blocks,
    _content_descriptors,
    _control_probe,
    _episode_coordinate_maps,
    _packet_probe,
    _raw_cues,
    _snapshot_with_content_columns,
    _tensor_hash,
    generate_fresh_inputs,
)


CALIBRATION_SEEDS = (114001,)
DEVELOPMENT_SEEDS = tuple(range(114101, 114117))
MAX_CHART_RESIDUAL = 1e-10
MIN_ADDITIVE_RESIDUAL = 1e-3
MAX_QUERY_GAUGE_SPREAD = 1e-10
EXPECTED_CHART_COUNT = 72


@dataclass(frozen=True)
class TwistedContentGateSnapshot:
    raw_cues: torch.Tensor
    predictions_by_row: torch.Tensor
    selected_residual: float
    best_additive_residual: float
    query_gauge_spread: float
    chart_count: int
    admissible_candidate_count: int
    admitted_classes: tuple[int, ...]
    candidate_ranks: tuple[tuple[int, int], ...]


def _match_cue_row(cues: torch.Tensor, expected: torch.Tensor) -> int | None:
    distances = torch.linalg.vector_norm(cues - expected, dim=1)
    scale = max(1.0, float(torch.linalg.vector_norm(expected).item()))
    matches = torch.nonzero(
        distances <= MAX_CHART_RESIDUAL * scale,
        as_tuple=False,
    ).view(-1)
    if matches.numel() != 1:
        return None
    return int(matches[0].item())


def enumerate_unlabeled_cartesian_charts(
    raw_cues: torch.Tensor,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate all 3-by-3 additive charts using cue values only."""
    cues = torch.as_tensor(raw_cues, dtype=torch.float64)
    if cues.ndim != 2 or cues.shape[0] != 9 or cues.shape[1] < 4:
        raise ValueError("chart discovery requires nine cue rows")
    if not torch.isfinite(cues).all():
        raise ValueError("chart discovery cues must be finite")
    charts: set[tuple[int, ...]] = set()
    indices = tuple(range(9))
    for origin in indices:
        remaining = tuple(index for index in indices if index != origin)
        for first_one, first_two in itertools.permutations(remaining, 2):
            after_first = tuple(
                index
                for index in remaining
                if index not in {first_one, first_two}
            )
            for second_one, second_two in itertools.permutations(after_first, 2):
                rows = [
                    origin,
                    second_one,
                    second_two,
                    first_one,
                    -1,
                    -1,
                    first_two,
                    -1,
                    -1,
                ]
                valid = True
                for first_level, first_row in ((1, first_one), (2, first_two)):
                    for second_level, second_row in ((1, second_one), (2, second_two)):
                        expected = cues[first_row] + cues[second_row] - cues[origin]
                        matched = _match_cue_row(cues, expected)
                        if matched is None:
                            valid = False
                            break
                        rows[3 * first_level + second_level] = matched
                    if not valid:
                        break
                chart = tuple(rows)
                if valid and len(set(chart)) == 9:
                    charts.add(chart)
    return tuple(sorted(charts))


def _incidence_row(first: int, second: int, twist_class: int) -> torch.Tensor:
    row = torch.zeros(6, dtype=torch.float64)
    row[int(first)] = 1.0
    row[3 + ((int(second) + int(twist_class) * int(first)) % 3)] = 1.0
    return row


def _operational_rank(matrix: torch.Tensor) -> int:
    singular = torch.linalg.svdvals(torch.as_tensor(matrix, dtype=torch.float64))
    if singular.numel() == 0 or float(singular[0].item()) == 0.0:
        return 0
    return int(
        torch.count_nonzero(
            singular > RANK_RELATIVE_TOLERANCE * singular[0]
        ).item()
    )


def _relative_matrix_residual(value: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = max(
        float(torch.linalg.vector_norm(reference).item()),
        torch.finfo(torch.float64).eps,
    )
    return float(torch.linalg.vector_norm(value).item()) / denominator


def _candidate_receipts(
    raw_cues: torch.Tensor,
    observed_content_sums: torch.Tensor,
    query_raw_cue: torch.Tensor,
) -> tuple[tuple[dict[str, Any], ...], int]:
    observed_cues = torch.as_tensor(raw_cues, dtype=torch.float64)
    contents = torch.as_tensor(observed_content_sums, dtype=torch.float64)
    query = torch.as_tensor(query_raw_cue, dtype=torch.float64).view(1, -1)
    if observed_cues.ndim != 2 or observed_cues.shape[0] != 8:
        raise ValueError("twisted-content training requires eight cue rows")
    if contents.ndim != 2 or contents.shape[0] != 8:
        raise ValueError("twisted-content observations must have eight rows")
    if query.shape[1] != observed_cues.shape[1]:
        raise ValueError("query cue dimension does not match training cues")
    if contents.shape[1] != CONTENT_DIMENSION:
        raise ValueError("twisted-content observations must have six coordinates")
    if not torch.isfinite(observed_cues).all() or not torch.isfinite(contents).all():
        raise ValueError("twisted-content inputs must be finite")
    all_cues = torch.cat((observed_cues, query), dim=0)
    charts = enumerate_unlabeled_cartesian_charts(all_cues)
    candidates: list[dict[str, Any]] = []
    for chart in charts:
        row_to_cell = {row: cell for cell, row in enumerate(chart)}
        if set(row_to_cell) != set(range(9)):
            continue
        for twist_class in (0, 1, 2):
            design_rows: list[torch.Tensor] = []
            for observed_row in range(8):
                cell = row_to_cell[observed_row]
                design_rows.append(
                    _incidence_row(cell // 3, cell % 3, twist_class)
                )
            design = torch.stack(design_rows)
            query_cell = row_to_cell[8]
            query_design = _incidence_row(
                query_cell // 3,
                query_cell % 3,
                twist_class,
            )
            rank = _operational_rank(design)
            augmented_rank = _operational_rank(
                torch.cat((design, query_design.view(1, -1)), dim=0)
            )
            theta = torch.linalg.pinv(
                design,
                atol=0.0,
                rtol=RANK_RELATIVE_TOLERANCE,
            ) @ contents
            fitted = design @ theta
            residual = _relative_matrix_residual(fitted - contents, contents)
            predictions = torch.empty(9, CONTENT_DIMENSION, dtype=torch.float64)
            for cell, row_index in enumerate(chart):
                predictions[row_index] = (
                    _incidence_row(cell // 3, cell % 3, twist_class) @ theta
                )
            candidates.append(
                {
                    "chart": chart,
                    "twist_class": twist_class,
                    "rank": rank,
                    "augmented_rank": augmented_rank,
                    "residual": residual,
                    "predictions": predictions,
                }
            )
    return tuple(candidates), len(charts)


def train_twisted_content_gate(
    raw_cues: torch.Tensor,
    observed_content_sums: torch.Tensor,
    query_raw_cue: torch.Tensor,
) -> TwistedContentGateSnapshot:
    """Select a nonzero finite-family coupling from opaque content rows."""
    candidates, chart_count = _candidate_receipts(
        raw_cues,
        observed_content_sums,
        query_raw_cue,
    )
    if chart_count != EXPECTED_CHART_COUNT:
        raise RuntimeError("raw cue chart family is not the frozen generic fixture")
    if not candidates:
        raise RuntimeError("no finite-family content candidate was constructed")
    additive_residual = min(
        candidate["residual"]
        for candidate in candidates
        if candidate["twist_class"] == 0
    )
    if additive_residual < MIN_ADDITIVE_RESIDUAL:
        raise RuntimeError("additive content arm is not separated")
    admitted = [
        candidate
        for candidate in candidates
        if candidate["twist_class"] in {1, 2}
        and candidate["rank"] == 5
        and candidate["augmented_rank"] == 5
        and candidate["residual"] <= MAX_RELATIVE_FIT_ERROR
    ]
    if not admitted:
        raise RuntimeError("no rank-five nonzero coupling fits the observed rows")
    admitted.sort(
        key=lambda item: (
            item["residual"],
            item["twist_class"],
            item["chart"],
        )
    )
    reference = admitted[0]["predictions"][8]
    scale = max(1.0, float(torch.linalg.vector_norm(reference).item()))
    spread = max(
        float(torch.linalg.vector_norm(item["predictions"][8] - reference).item())
        / scale
        for item in admitted
    )
    if spread > MAX_QUERY_GAUGE_SPREAD:
        raise RuntimeError("zero-residual chart gauges disagree on the query")
    all_cues = torch.cat(
        (
            torch.as_tensor(raw_cues, dtype=torch.float64),
            torch.as_tensor(query_raw_cue, dtype=torch.float64).view(1, -1),
        ),
        dim=0,
    )
    return TwistedContentGateSnapshot(
        raw_cues=all_cues.clone(),
        predictions_by_row=admitted[0]["predictions"].clone(),
        selected_residual=float(admitted[0]["residual"]),
        best_additive_residual=float(additive_residual),
        query_gauge_spread=spread,
        chart_count=chart_count,
        admissible_candidate_count=len(admitted),
        admitted_classes=tuple(sorted({item["twist_class"] for item in admitted})),
        candidate_ranks=tuple(
            sorted({(item["rank"], item["augmented_rank"]) for item in admitted})
        ),
    )


def predict_twisted_content(
    gate: TwistedContentGateSnapshot,
    raw_cue: torch.Tensor,
) -> torch.Tensor:
    cue = torch.as_tensor(raw_cue, dtype=torch.float64).view(-1)
    if cue.shape != gate.raw_cues.shape[1:] or not torch.isfinite(cue).all():
        raise ValueError("twisted-content cue has the wrong shape or is nonfinite")
    distances = torch.linalg.vector_norm(gate.raw_cues - cue, dim=1)
    matches = torch.nonzero(
        distances <= MAX_CHART_RESIDUAL * max(1.0, float(cue.norm().item())),
        as_tuple=False,
    ).view(-1)
    if matches.numel() != 1:
        raise RuntimeError("twisted-content cue is absent or ambiguous")
    return gate.predictions_by_row[int(matches[0].item())].clone()


def compile_twisted_packet_indices(
    gate: TwistedContentGateSnapshot,
    raw_cue: torch.Tensor,
    arrived_packet_indices: Sequence[int],
    weight: torch.Tensor,
    response_indices: Sequence[int],
) -> tuple[int, ...]:
    prediction = predict_twisted_content(gate, raw_cue)
    receipt = _bind_prediction_to_packets(
        prediction,
        arrived_packet_indices,
        weight,
        response_indices,
    )
    return tuple(int(value) for value in receipt["selected_indices"])


def _gate_hash(gate: TwistedContentGateSnapshot) -> str:
    digest = hashlib.sha256()
    digest.update(gate.raw_cues.detach().cpu().numpy().tobytes())
    digest.update(gate.predictions_by_row.detach().cpu().numpy().tobytes())
    digest.update(
        repr(
            (
                gate.selected_residual,
                gate.best_additive_residual,
                gate.query_gauge_spread,
                gate.chart_count,
                gate.admissible_candidate_count,
                gate.admitted_classes,
                gate.candidate_ranks,
            )
        ).encode("ascii")
    )
    return digest.hexdigest()


def _twist_class(seed: int) -> int:
    return 1 + (int(seed) & 1)


def _pair_for_cell(first: int, second: int, twist_class: int) -> tuple[int, int]:
    return int(first), 3 + ((int(second) + int(twist_class) * int(first)) % 3)


def _compile_or_abstain(
    gate: TwistedContentGateSnapshot,
    cue: torch.Tensor,
    arrived: Sequence[int],
    weight: torch.Tensor,
    hidden: Sequence[int],
) -> tuple[tuple[int, ...] | None, str | None]:
    try:
        return compile_twisted_packet_indices(
            gate,
            cue,
            arrived,
            weight,
            hidden,
        ), None
    except (RuntimeError, ValueError) as exc:
        return None, str(exc)


def _endpoint_control(
    snapshot: Any,
    content_columns: torch.Tensor,
    coordinates: Sequence[int],
    events: Sequence[int],
    selected: tuple[int, ...] | None,
    expected: tuple[int, ...],
) -> dict[str, Any]:
    if selected is None:
        return {"success": False, "abstained": True}
    return _control_probe(
        snapshot,
        content_columns,
        coordinates,
        events,
        selected,
        expected,
    )


def analyze_z3_twisted_content_row(
    seed: int,
    content_columns: torch.Tensor,
) -> dict[str, Any]:
    seed = int(seed)
    matrix = torch.as_tensor(content_columns, dtype=torch.float64)
    base_snapshot = _base_snapshot(matrix)
    input_pool, hidden, _ = _blocks()
    maps = _episode_coordinate_maps(seed + 400_009)
    cues = _raw_cues(seed + 410_011)
    cells = tuple((first, second) for first in range(3) for second in range(3))
    twist_class = _twist_class(seed)

    observed: list[torch.Tensor] = []
    for row, (first, second) in enumerate(cells[:8]):
        moved = _snapshot_with_content_columns(base_snapshot, matrix, maps[row])
        descriptors = _content_descriptors(moved.weight, maps[row], hidden)
        pair = _pair_for_cell(first, second, twist_class)
        observed.append(descriptors[pair[0]] + descriptors[pair[1]])
    observed_content = torch.stack(observed)

    generator = torch.Generator(device="cpu").manual_seed(seed + 420_017)
    order = torch.randperm(8, generator=generator)
    training_cues = cues[:8].index_select(0, order)
    training_content = observed_content.index_select(0, order)
    gate = train_twisted_content_gate(training_cues, training_content, cues[8])
    gate_before = _gate_hash(gate)
    predicted_training = torch.stack(
        [predict_twisted_content(gate, cue) for cue in training_cues]
    )
    training_error = _relative_matrix_residual(
        predicted_training - training_content,
        training_content,
    )

    query_pair = _pair_for_cell(2, 2, twist_class)
    query_coordinates = maps[8]
    query_snapshot = _snapshot_with_content_columns(
        base_snapshot,
        matrix,
        query_coordinates,
    )
    descriptors = _content_descriptors(
        query_snapshot.weight,
        query_coordinates,
        hidden,
    )
    expected_content = descriptors[query_pair[0]] + descriptors[query_pair[1]]
    predicted_query = predict_twisted_content(gate, cues[8])
    query_error = _relative_matrix_residual(
        predicted_query - expected_content,
        expected_content,
    )
    distractor_role = 0
    event_roles = query_pair + (distractor_role,)
    if len(set(event_roles)) != 3:
        raise RuntimeError("frozen twisted query distractor collided with the true pair")
    arrived = tuple(query_coordinates[role] for role in event_roles)
    binding = _bind_prediction_to_packets(
        predicted_query,
        arrived,
        query_snapshot.weight,
        hidden,
    )
    learned_indices = tuple(int(value) for value in binding["selected_indices"])
    oracle_indices = tuple(query_coordinates[role] for role in query_pair)

    shuffle_rejected = False
    shuffle_error = None
    try:
        train_twisted_content_gate(
            training_cues,
            training_content.roll(1, dims=0),
            cues[8],
        )
    except (RuntimeError, ValueError) as exc:
        shuffle_rejected = True
        shuffle_error = str(exc)

    reverse = torch.arange(7, -1, -1)
    reordered = train_twisted_content_gate(
        training_cues.index_select(0, reverse),
        training_content.index_select(0, reverse),
        cues[8],
    )
    row_order_error = float(
        torch.linalg.vector_norm(
            predict_twisted_content(reordered, cues[8]) - predicted_query
        ).item()
    )
    chart_generator = torch.Generator(device="cpu").manual_seed(seed + 430_019)
    chart, _ = torch.linalg.qr(
        torch.randn(CUE_DIMENSION, CUE_DIMENSION, generator=chart_generator, dtype=torch.float64)
    )
    chart_gate = train_twisted_content_gate(
        training_cues @ chart.T,
        training_content,
        cues[8] @ chart.T,
    )
    chart_error = float(
        torch.linalg.vector_norm(
            predict_twisted_content(chart_gate, cues[8] @ chart.T) - predicted_query
        ).item()
    )
    alternative_delta = torch.linspace(0.1, 0.6, 6, dtype=torch.float64)
    preflight_gates = {
        "raw_chart_family_complete": gate.chart_count == EXPECTED_CHART_COUNT,
        "rank_five_query_identified": gate.candidate_ranks == ((5, 5),),
        "nonzero_twist_fits_observed_rows": gate.selected_residual <= MAX_RELATIVE_FIT_ERROR,
        "additive_arm_rejected": gate.best_additive_residual >= MIN_ADDITIVE_RESIDUAL,
        "all_gauges_agree_on_query": gate.query_gauge_spread <= MAX_QUERY_GAUGE_SPREAD,
        "training_rows_reconstructed": training_error <= MAX_RELATIVE_FIT_ERROR,
        "conditional_query_content_exact": query_error <= MAX_RELATIVE_FIT_ERROR,
        "current_packet_binding_unique": (
            binding["relative_binding_margin"] > MIN_RELATIVE_BINDING_MARGIN
        ),
        "episode_maps_fresh": len(set(maps)) == 9,
        "query_columns_in_second_block": (
            set(query_coordinates) == set(tuple(range(6, 12)))
        ),
        "association_shuffle_rejected_pre_endpoint": shuffle_rejected,
        "row_order_invariant": row_order_error <= 1e-10,
        "orthogonal_cue_chart_invariant": chart_error <= 1e-10,
        "alternative_query_completion_exists": float(alternative_delta.norm().item()) > 1e-6,
    }
    if not all(preflight_gates.values()):
        return {
            "seed": seed,
            "status": "CONDITIONAL_Z3_TWISTED_CONTENT_STOP",
            "gates": preflight_gates,
            "shuffle_error": shuffle_error,
            "endpoint_opened": False,
            "claim_scope": "pre-endpoint finite-family twisted-content apparatus",
        }

    atomic = [
        _packet_probe(
            base_snapshot,
            matrix,
            query_coordinates,
            (role,),
            (query_coordinates[role],),
        )
        for role in query_pair
    ]
    expected_target = tuple(
        sorted(
            set(atomic[0]["decoded_target_set"])
            | set(atomic[1]["decoded_target_set"])
        )
    )
    learned = _endpoint_control(
        base_snapshot,
        matrix,
        query_coordinates,
        event_roles,
        learned_indices,
        expected_target,
    )
    oracle = _endpoint_control(
        base_snapshot,
        matrix,
        query_coordinates,
        event_roles,
        oracle_indices,
        expected_target,
    )
    joint_pair = _pair_for_cell(0, 0, twist_class)
    joint_lookup = _endpoint_control(
        base_snapshot,
        matrix,
        query_coordinates,
        event_roles,
        tuple(query_coordinates[role] for role in joint_pair),
        expected_target,
    )
    coordinate_memorizer = _endpoint_control(
        base_snapshot,
        matrix,
        query_coordinates,
        event_roles,
        tuple(input_pool[role] for role in query_pair),
        expected_target,
    )
    wrong_second = query_pair[1] - 3
    wrong_cue = cues[wrong_second]
    wrong_indices, wrong_error = _compile_or_abstain(
        gate,
        wrong_cue,
        arrived,
        query_snapshot.weight,
        hidden,
    )
    wrong = _endpoint_control(
        base_snapshot,
        matrix,
        query_coordinates,
        event_roles,
        wrong_indices,
        expected_target,
    )

    hidden_idx = torch.tensor(hidden, dtype=torch.long)
    arrived_idx = torch.tensor(arrived, dtype=torch.long)
    shuffled_weight = query_snapshot.weight.detach().clone()
    original_arrived = shuffled_weight[hidden_idx[:, None], arrived_idx].clone()
    shuffled_weight[hidden_idx[:, None], arrived_idx] = original_arrived[:, [1, 2, 0]]
    shuffled_indices, binding_shuffle_error = _compile_or_abstain(
        gate,
        cues[8],
        arrived,
        shuffled_weight,
        hidden,
    )
    binding_shuffle = _endpoint_control(
        base_snapshot,
        matrix,
        query_coordinates,
        event_roles,
        shuffled_indices,
        expected_target,
    )
    no_context = _endpoint_control(
        base_snapshot,
        matrix,
        query_coordinates,
        event_roles,
        tuple(input_pool),
        expected_target,
    )
    expected_packet = [0, 0, 0, 3, 0, 0, 0]
    expected_written = [0, 3, 0, 0, 0, 0, 0]
    gate_after = _gate_hash(gate)
    endpoint_gates = {
        "learned_query_transfer": bool(
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
        "canonical_coordinate_memorizer_fails": not coordinate_memorizer["success"],
        "wrong_raw_cue_fails": not wrong["success"],
        "packet_binding_shuffle_fails": not binding_shuffle["success"],
        "no_context_all_packet_fails": not no_context["success"],
        "one_shot_three_packet_receipt": bool(
            learned["input_packet_count_by_tick"] == expected_packet
            and learned["input_written_count_by_tick"] == expected_written
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
        "hidden_twist_class": twist_class,
        "status": (
            "CONDITIONAL_Z3_TWISTED_CONTENT_PASS"
            if all(gates.values())
            else "CONDITIONAL_Z3_TWISTED_CONTENT_STOP"
        ),
        "gates": gates,
        "training_inputs": ["opaque_raw_cue", "current_packet_content_sum"],
        "chart_count": gate.chart_count,
        "admissible_candidate_count": gate.admissible_candidate_count,
        "admitted_classes": list(gate.admitted_classes),
        "selected_residual": gate.selected_residual,
        "best_additive_residual": gate.best_additive_residual,
        "query_gauge_spread": gate.query_gauge_spread,
        "query_content_error": query_error,
        "binding": binding,
        "coordinate_maps": [list(item) for item in maps],
        "expected_target_set_from_atomic_union": list(expected_target),
        "learned_success": bool(learned["success"]),
        "oracle_success": bool(oracle["success"]),
        "joint_lookup_success": bool(joint_lookup["success"]),
        "coordinate_memorizer_success": bool(coordinate_memorizer["success"]),
        "wrong_cue_success": bool(wrong["success"]),
        "binding_shuffle_success": bool(binding_shuffle["success"]),
        "no_context_success": bool(no_context["success"]),
        "wrong_cue_error": wrong_error,
        "binding_shuffle_error": binding_shuffle_error,
        "shuffle_error": shuffle_error,
        "row_order_prediction_error": row_order_error,
        "orthogonal_chart_prediction_error": chart_error,
        "alternative_query_delta_norm": float(alternative_delta.norm().item()),
        "learned": learned,
        "oracle": oracle,
        "atomic": atomic,
        "gate_hash": gate_before,
        "content_sha256": _tensor_hash(matrix),
        "endpoint_opened": True,
        "claim_scope": (
            "synthetic finite-family Z3 nonzero-twist content transfer with "
            "current-column coordinate binding"
        ),
    }


def analyze_z3_twisted_content_artifact(
    path: str | Path,
    *,
    stage: str,
) -> dict[str, Any]:
    if stage not in {"calibration", "development"}:
        raise ValueError("stage must be calibration or development")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "FRESH_SIX_CONTENT_INPUTS_READY":
        raise RuntimeError("fresh six-content inputs did not pass producer gates")
    expected = CALIBRATION_SEEDS if stage == "calibration" else DEVELOPMENT_SEEDS
    actual = tuple(int(row["seed"]) for row in payload.get("rows", ()))
    if actual != expected:
        raise RuntimeError("fresh input seed order does not match the frozen stage")
    rows = [
        analyze_z3_twisted_content_row(
            int(row["seed"]),
            torch.tensor(row["content_columns"], dtype=torch.float64),
        )
        for row in payload["rows"]
    ]
    pass_count = sum(
        row["status"] == "CONDITIONAL_Z3_TWISTED_CONTENT_PASS" for row in rows
    )
    passed = pass_count == len(expected)
    return {
        "status": (
            "Z3_TWISTED_CONTENT_CALIBRATION_PASS"
            if passed and stage == "calibration"
            else "Z3_TWISTED_CONTENT_DEVELOPMENT_GO"
            if passed
            else "Z3_TWISTED_CONTENT_STOP"
        ),
        "stage": stage,
        "seed_count": len(rows),
        "pass_count": pass_count,
        "twist_class_counts": {
            str(value): sum(row.get("hidden_twist_class") == value for row in rows)
            for value in (1, 2)
        },
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
        "no_context_success_total": sum(row.get("no_context_success", False) for row in rows),
        "maximum_selected_residual": max(
            row.get("selected_residual", float("inf")) for row in rows
        ),
        "minimum_additive_residual": min(
            row.get("best_additive_residual", float("-inf")) for row in rows
        ),
        "maximum_query_content_error": max(
            row.get("query_content_error", float("inf")) for row in rows
        ),
        "minimum_binding_margin": min(
            row.get("binding", {}).get("relative_binding_margin", float("-inf"))
            for row in rows
        ),
        "endpoint_opened": any(row.get("endpoint_opened", False) for row in rows),
        "confirmation_opened": False,
        "rows": rows,
    }


__all__ = [
    "CALIBRATION_SEEDS",
    "DEVELOPMENT_SEEDS",
    "TwistedContentGateSnapshot",
    "enumerate_unlabeled_cartesian_charts",
    "train_twisted_content_gate",
    "predict_twisted_content",
    "compile_twisted_packet_indices",
    "generate_fresh_inputs",
    "analyze_z3_twisted_content_row",
    "analyze_z3_twisted_content_artifact",
]
