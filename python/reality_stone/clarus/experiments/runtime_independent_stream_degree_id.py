"""BA-TR30: polynomial degree/noise model identification, independent stream.

The frozen learner sees only dimensionless cue vectors, contemporaneous
noisy content observations, and the fold's true noise level (supplied by
axiom, contract section 4.1).  Generator degree, held-out clean truth, and
the candidate bank belong to the synthetic harness.  The bank is a function
of (seed, generator, non-query cells) only; it is serialized and SHA-256
hashed strictly before any prediction is issued, and the order receipt is a
mandatory artifact (contract section 4.4, `BANK_RECEIPT_FAIL` otherwise).

This run is a simulator model-class identification experiment.  It is not
evidence about real brains, memory, consciousness, or AGI.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .runtime_rotating_low_degree_content_transfer import (
    _relative_residual,
    _tensor_hash,
)


CALIBRATION_SEEDS = (117001,)
DEVELOPMENT_SEEDS = tuple(range(117101, 117117))
CUE_DIMENSION = 2
CONTENT_DIMENSION = 6
TRAINING_ROWS = 24
DEGREE_SET = (1, 2, 3)
NOISE_SET = (0.0, 1e-3, 1e-2)
WITNESS_DEGREE = 4
WITNESS_NOISE = 1e-3
PARSIMONY_RHO = 0.5
PARSIMONY_FLOOR = 1e-8
CLASS_TAU_FLOOR = 1e-8
CLASS_TAU_NOISE_FACTOR = 8.0
MAX_PHI3_CONDITION = 1e6
PREDICTION_GATES = {0.0: 1e-10, 1e-3: 2e-2, 1e-2: 2e-1}
BANK_SIZE = 8
BANK_OTHER_CELL_COUNT = 4
BANK_DISTRACTOR_COUNT = 3
FOLD_SCHEDULE = tuple(
    [(degree, eta) for degree in DEGREE_SET for eta in NOISE_SET]
    + [(WITNESS_DEGREE, WITNESS_NOISE)]
)


def _monomial_exponents(degree: int) -> tuple[tuple[int, int], ...]:
    exponents: list[tuple[int, int]] = []
    for total in range(int(degree) + 1):
        for first in range(total, -1, -1):
            exponents.append((first, total - first))
    return tuple(exponents)


def _poly_features(cues: torch.Tensor, degree: int) -> torch.Tensor:
    packed = torch.as_tensor(cues, dtype=torch.float64)
    if packed.ndim == 1:
        packed = packed.view(1, -1)
    if packed.ndim != 2 or packed.shape[1] != CUE_DIMENSION:
        raise ValueError("total-degree features require two cue coordinates")
    first = packed[:, 0]
    second = packed[:, 1]
    columns = [
        first.pow(a) * second.pow(b) for a, b in _monomial_exponents(degree)
    ]
    return torch.stack(columns, dim=1)


def _studentized_press(
    design: torch.Tensor,
    observations: torch.Tensor,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Frozen operator: C_d = pinv(Phi) Y and studentized PRESS via the hat
    identity  s'_d = (1/N) sum_i ||e_i^{loo}|| sqrt(1-h_ii)
              = (1/N) sum_i ||e_i|| / sqrt(1-h_ii),
    equivalence with explicit refits confirmed by 11-math item (b)."""
    pseudoinverse = torch.linalg.pinv(design)
    coefficients = pseudoinverse @ observations
    leverages = torch.diagonal(design @ pseudoinverse)
    residuals = observations - design @ coefficients
    denominator = torch.sqrt((1.0 - leverages).clamp_min(0.0))
    row_norms = torch.linalg.vector_norm(residuals, dim=1)
    ratios = torch.where(
        denominator > 0.0,
        row_norms / denominator,
        torch.full_like(row_norms, float("inf")),
    )
    return float(ratios.mean().item()), leverages, coefficients


def _condition_number(design: torch.Tensor) -> float:
    singular = torch.linalg.svdvals(design)
    smallest = float(singular[-1].item())
    if smallest <= 0.0:
        return float("inf")
    return float(singular[0].item()) / smallest


def _select_degree(s_prime: dict[int, float], slack: float) -> int:
    minimum = min(s_prime.values())
    for degree in DEGREE_SET:
        if s_prime[degree] <= (1.0 + slack) * minimum + PARSIMONY_FLOOR:
            return degree
    return DEGREE_SET[-1]


def _candidate_bank(
    seed: int,
    fold_index: int,
    clean_training_content: torch.Tensor,
    clean_query_content: torch.Tensor,
) -> dict[str, Any]:
    """Contract section 4.4: one truth element (generator output at the query
    cell, not via the model) plus K-1 elements that are functions of
    (seed, generator, non-query cells) only.  Distractor norms match the mean
    norm of the non-query cell contents, never the truth norm."""
    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + 940_009 + 1009 * int(fold_index)
    )
    other_cells = torch.randperm(TRAINING_ROWS, generator=generator)[
        :BANK_OTHER_CELL_COUNT
    ].tolist()
    matched_norm = float(
        torch.linalg.vector_norm(clean_training_content, dim=1).mean().item()
    )
    directions = torch.randn(
        BANK_DISTRACTOR_COUNT,
        CONTENT_DIMENSION,
        generator=generator,
        dtype=torch.float64,
    )
    scales = torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    distractors = directions / scales * matched_norm
    elements = torch.cat(
        (
            clean_query_content.view(1, -1),
            clean_training_content.index_select(
                0, torch.tensor(other_cells, dtype=torch.long)
            ),
            distractors,
        ),
        dim=0,
    )
    order = torch.randperm(BANK_SIZE, generator=generator)
    bank = elements.index_select(0, order)
    positions = {int(element): slot for slot, element in enumerate(order.tolist())}
    return {
        "values": bank,
        "sha256": _tensor_hash(bank),
        "truth_position": positions[0],
        "other_cell_positions": [
            positions[1 + index] for index in range(BANK_OTHER_CELL_COUNT)
        ],
        "other_cells": other_cells,
        "matched_norm": matched_norm,
    }


def _bank_selection(prediction: torch.Tensor, bank: torch.Tensor) -> int:
    distances = torch.linalg.vector_norm(bank - prediction.view(1, -1), dim=1)
    return int(torch.argmin(distances).item())


def generate_degree_id_inputs(seeds: Sequence[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed_value in seeds:
        seed = int(seed_value)
        folds: list[dict[str, Any]] = []
        for fold_index, (degree, eta) in enumerate(FOLD_SCHEDULE):
            cue_generator = torch.Generator(device="cpu").manual_seed(
                seed + 910_001 + 1009 * fold_index
            )
            cues = torch.randn(
                TRAINING_ROWS + 1,
                CUE_DIMENSION,
                generator=cue_generator,
                dtype=torch.float64,
            )
            coefficient_generator = torch.Generator(device="cpu").manual_seed(
                seed + 920_003 + 1009 * fold_index
            )
            coefficients = (
                2.0
                * torch.rand(
                    len(_monomial_exponents(degree)),
                    CONTENT_DIMENSION,
                    generator=coefficient_generator,
                    dtype=torch.float64,
                )
                - 1.0
            )
            noise_generator = torch.Generator(device="cpu").manual_seed(
                seed + 930_007 + 1009 * fold_index
            )
            noise = torch.randn(
                TRAINING_ROWS,
                CONTENT_DIMENSION,
                generator=noise_generator,
                dtype=torch.float64,
            )
            folds.append(
                {
                    "fold_index": fold_index,
                    "degree": int(degree),
                    "eta": float(eta),
                    "z_train": [[float(v) for v in row] for row in cues[:TRAINING_ROWS].tolist()],
                    "z_query": [float(v) for v in cues[TRAINING_ROWS].tolist()],
                    "coefficients": [
                        [float(v) for v in row] for row in coefficients.tolist()
                    ],
                    "noise": [[float(v) for v in row] for row in noise.tolist()],
                    "cue_sha256": _tensor_hash(cues),
                    "coefficient_sha256": _tensor_hash(coefficients),
                    "noise_sha256": _tensor_hash(noise),
                    "finite": bool(
                        torch.isfinite(cues).all()
                        and torch.isfinite(coefficients).all()
                        and torch.isfinite(noise).all()
                    ),
                }
            )
        rows.append({"seed": seed, "folds": folds})
    ready = bool(rows) and all(
        fold["finite"] for row in rows for fold in row["folds"]
    )
    return {
        "status": "DEGREE_ID_INPUTS_READY" if ready else "DEGREE_ID_INPUTS_STOP",
        "seed_count": len(rows),
        "fold_schedule": [[int(d), float(e)] for d, e in FOLD_SCHEDULE],
        "rows": rows,
    }


def _fit_operator(
    training_cues: torch.Tensor,
    observations: torch.Tensor,
    eta: float,
) -> dict[str, Any]:
    """Frozen learner (contract section 4.2).  The true fold noise level eta
    is supplied by axiom; eta estimation is outside this run's claim scope."""
    s_prime: dict[int, float] = {}
    coefficients: dict[int, torch.Tensor] = {}
    for degree in DEGREE_SET:
        design = _poly_features(training_cues, degree)
        value, _, fit = _studentized_press(design, observations)
        s_prime[degree] = value
        coefficients[degree] = fit
    tau_class = max(CLASS_TAU_FLOOR, CLASS_TAU_NOISE_FACTOR * float(eta))
    minimum = min(s_prime.values())
    abstained = minimum > tau_class
    degree_hat = None if abstained else _select_degree(s_prime, PARSIMONY_RHO)
    degree_hat_ablation = _select_degree(s_prime, 0.0)
    return {
        "s_prime": s_prime,
        "coefficients": coefficients,
        "tau_class": tau_class,
        "min_s_prime": minimum,
        "abstained": abstained,
        "degree_hat": degree_hat,
        "degree_hat_ablation": degree_hat_ablation,
    }


def _analyze_fold(seed: int, fold: dict[str, Any]) -> dict[str, Any]:
    fold_index = int(fold["fold_index"])
    degree_star = int(fold["degree"])
    eta = float(fold["eta"])
    is_witness = degree_star == WITNESS_DEGREE
    training_cues = torch.tensor(fold["z_train"], dtype=torch.float64)
    query_cue = torch.tensor(fold["z_query"], dtype=torch.float64)
    coefficients_star = torch.tensor(fold["coefficients"], dtype=torch.float64)
    noise = torch.tensor(fold["noise"], dtype=torch.float64)
    if training_cues.shape != (TRAINING_ROWS, CUE_DIMENSION):
        raise ValueError("degree-id training cues have the wrong shape")
    if coefficients_star.shape != (
        len(_monomial_exponents(degree_star)),
        CONTENT_DIMENSION,
    ):
        raise ValueError("degree-id coefficient input has the wrong shape")
    clean_training = _poly_features(training_cues, degree_star) @ coefficients_star
    observations = clean_training + eta * noise
    clean_query = (_poly_features(query_cue, degree_star) @ coefficients_star).view(-1)

    phi3_condition = _condition_number(_poly_features(training_cues, 3))
    cond_admitted = phi3_condition <= MAX_PHI3_CONDITION
    counter = 0

    # Candidate bank: serialized and hashed strictly before the learner emits
    # any prediction (order receipt, contract section 4.4).
    bank = _candidate_bank(seed, fold_index, clean_training, clean_query)
    counter += 1
    bank_counter = counter

    operator = _fit_operator(training_cues, observations, eta)
    counter += 1
    decision_counter = counter

    prediction = None
    prediction_error = None
    bank_selection = None
    if cond_admitted and not operator["abstained"]:
        degree_hat = int(operator["degree_hat"])
        prediction = (
            _poly_features(query_cue, degree_hat) @ operator["coefficients"][degree_hat]
        ).view(-1)
        prediction_error = _relative_residual(prediction - clean_query, clean_query)
        bank_selection = _bank_selection(prediction, bank["values"])

    receipt_valid = (
        bank["sha256"] == _tensor_hash(bank["values"])
        and bank_counter < decision_counter
    )

    # Controls (contract section 4.5).
    shuffled = _fit_operator(training_cues, observations.roll(1, dims=0), eta)
    shuffle_abstained = bool(shuffled["abstained"])

    wrong_cue_selection = None
    wrong_cue_truth_selected = None
    if prediction is not None:
        wrong_cell = int(bank["other_cells"][0])
        degree_hat = int(operator["degree_hat"])
        wrong_prediction = (
            _poly_features(training_cues[wrong_cell], degree_hat)
            @ operator["coefficients"][degree_hat]
        ).view(-1)
        wrong_cue_selection = _bank_selection(wrong_prediction, bank["values"])
        wrong_cue_truth_selected = wrong_cue_selection == bank["truth_position"]

    forced_affine_error = None
    forced_affine_gate_failed = None
    if not is_witness and degree_star in (2, 3):
        affine_prediction = (
            _poly_features(query_cue, 1) @ operator["coefficients"][1]
        ).view(-1)
        forced_affine_error = _relative_residual(
            affine_prediction - clean_query, clean_query
        )
        forced_affine_gate_failed = forced_affine_error > PREDICTION_GATES[eta]

    if is_witness:
        gates = {
            "cond_admitted": cond_admitted,
            "bank_receipt_valid": receipt_valid,
            "witness_abstains": bool(operator["abstained"]),
            "association_shuffle_rejected": shuffle_abstained,
        }
    else:
        gates = {
            "cond_admitted": cond_admitted,
            "bank_receipt_valid": receipt_valid,
            "no_class_abstain": not operator["abstained"],
            "degree_identified": operator["degree_hat"] == degree_star,
            "prediction_within_gate": (
                prediction_error is not None
                and prediction_error <= PREDICTION_GATES[eta]
            ),
            "bank_truth_selected": bank_selection == bank["truth_position"],
            "association_shuffle_rejected": shuffle_abstained,
            "wrong_cue_truth_rejected": wrong_cue_truth_selected is False,
        }
        if degree_star in (2, 3):
            gates["forced_affine_gate_fails"] = bool(forced_affine_gate_failed)

    if not cond_admitted:
        fold_status = "CUE_DEGENERATE"
    elif not receipt_valid:
        fold_status = "BANK_RECEIPT_FAIL"
    elif operator["abstained"]:
        fold_status = "CLASS_EXTERNAL_ABSTAIN"
    else:
        fold_status = "OK"

    return {
        "fold_index": fold_index,
        "kind": "witness" if is_witness else "main",
        "degree_star": degree_star,
        "eta": eta,
        "fold_status": fold_status,
        "phi3_condition": phi3_condition,
        "s_prime": {str(d): operator["s_prime"][d] for d in DEGREE_SET},
        "tau_class": operator["tau_class"],
        "abstained": bool(operator["abstained"]),
        "abstain_margin": operator["min_s_prime"] / operator["tau_class"],
        "degree_hat": operator["degree_hat"],
        "degree_hat_ablation": operator["degree_hat_ablation"],
        "prediction_error": prediction_error,
        "prediction_gate": None if is_witness else PREDICTION_GATES[eta],
        "bank_receipt": {
            "sha256": bank["sha256"],
            "bank_counter": bank_counter,
            "decision_counter": decision_counter,
            "truth_position": bank["truth_position"],
            "other_cells": bank["other_cells"],
            "other_cell_positions": bank["other_cell_positions"],
            "matched_norm": bank["matched_norm"],
            "values": [[float(v) for v in row] for row in bank["values"].tolist()],
        },
        "bank_selection": bank_selection,
        "controls": {
            "shuffle_abstained": shuffle_abstained,
            "shuffle_min_s_prime": shuffled["min_s_prime"],
            "wrong_cue_selection": wrong_cue_selection,
            "wrong_cue_truth_selected": wrong_cue_truth_selected,
            "forced_affine_error": forced_affine_error,
            "forced_affine_gate_failed": forced_affine_gate_failed,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def analyze_degree_id_row(seed: int, folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    seed = int(seed)
    schedule = tuple((int(fold["degree"]), float(fold["eta"])) for fold in folds)
    if schedule != FOLD_SCHEDULE:
        raise ValueError("degree-id fold schedule does not match the frozen contract")
    fold_rows = [_analyze_fold(seed, fold) for fold in folds]
    main_rows = [row for row in fold_rows if row["kind"] == "main"]
    witness_rows = [row for row in fold_rows if row["kind"] == "witness"]
    aggregate_gates = {
        "all_main_folds_pass": all(row["passed"] for row in main_rows),
        "all_witness_folds_pass": all(row["passed"] for row in witness_rows),
        "main_fold_count_nine": len(main_rows) == 9,
        "witness_fold_count_one": len(witness_rows) == 1,
    }
    return {
        "seed": seed,
        "status": (
            "INDEPENDENT_STREAM_DEGREE_ID_PASS"
            if all(aggregate_gates.values())
            else "INDEPENDENT_STREAM_DEGREE_ID_STOP"
        ),
        "gates": aggregate_gates,
        "degree_identifications": sum(
            row["degree_hat"] == row["degree_star"] for row in main_rows
        ),
        "bank_truth_selections": sum(
            row["bank_selection"] == row["bank_receipt"]["truth_position"]
            for row in main_rows
        ),
        "witness_abstains": sum(row["abstained"] for row in witness_rows),
        "witness_margin": min(row["abstain_margin"] for row in witness_rows),
        "shuffle_rejections": sum(
            row["controls"]["shuffle_abstained"] for row in fold_rows
        ),
        "wrong_cue_truth_selections": sum(
            bool(row["controls"]["wrong_cue_truth_selected"]) for row in main_rows
        ),
        "forced_affine_gate_failures": sum(
            bool(row["controls"]["forced_affine_gate_failed"])
            for row in main_rows
            if row["degree_star"] in (2, 3)
        ),
        "ablation_degree_matches": sum(
            row["degree_hat_ablation"] == row["degree_star"] for row in main_rows
        ),
        "maximum_prediction_error_by_eta": {
            str(eta): max(
                (
                    row["prediction_error"]
                    for row in main_rows
                    if row["eta"] == eta and row["prediction_error"] is not None
                ),
                default=None,
            )
            for eta in NOISE_SET
        },
        "maximum_phi3_condition": max(row["phi3_condition"] for row in fold_rows),
        "folds": fold_rows,
        "endpoint_opened": True,
        "confirmation_opened": False,
        "claim_scope": (
            "synthetic model-class identification and direct content prediction "
            "over a declared finite polynomial family and noise set with an "
            "independent candidate stream"
        ),
    }


def analyze_degree_id_artifact(path: str | Path, *, stage: str) -> dict[str, Any]:
    if stage not in {"calibration", "development"}:
        raise ValueError("stage must be calibration or development")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "DEGREE_ID_INPUTS_READY":
        raise RuntimeError("degree-id input producer did not pass")
    expected = CALIBRATION_SEEDS if stage == "calibration" else DEVELOPMENT_SEEDS
    actual = tuple(int(row["seed"]) for row in payload.get("rows", ()))
    if actual != expected:
        raise RuntimeError("degree-id seed order does not match the frozen stage")
    rows = [
        analyze_degree_id_row(int(row["seed"]), row["folds"])
        for row in payload["rows"]
    ]
    pass_count = sum(
        row["status"] == "INDEPENDENT_STREAM_DEGREE_ID_PASS" for row in rows
    )
    passed = pass_count == len(expected)
    maxima: dict[str, float | None] = {}
    for eta in NOISE_SET:
        values = [
            row["maximum_prediction_error_by_eta"][str(eta)]
            for row in rows
            if row["maximum_prediction_error_by_eta"][str(eta)] is not None
        ]
        maxima[str(eta)] = max(values) if values else None
    return {
        "status": (
            "DEGREE_ID_CALIBRATION_PASS"
            if passed and stage == "calibration"
            else "DEGREE_ID_DEVELOPMENT_GO"
            if passed
            else "DEGREE_ID_STOP"
        ),
        "stage": stage,
        "seed_count": len(rows),
        "pass_count": pass_count,
        "main_fold_count": sum(
            1 for row in rows for fold in row["folds"] if fold["kind"] == "main"
        ),
        "witness_fold_count": sum(
            1 for row in rows for fold in row["folds"] if fold["kind"] == "witness"
        ),
        "degree_identification_count": sum(
            row["degree_identifications"] for row in rows
        ),
        "bank_truth_selection_count": sum(row["bank_truth_selections"] for row in rows),
        "witness_abstain_count": sum(row["witness_abstains"] for row in rows),
        "minimum_witness_margin": min(row["witness_margin"] for row in rows),
        "shuffle_rejection_count": sum(row["shuffle_rejections"] for row in rows),
        "wrong_cue_truth_selection_count": sum(
            row["wrong_cue_truth_selections"] for row in rows
        ),
        "forced_affine_gate_failure_count": sum(
            row["forced_affine_gate_failures"] for row in rows
        ),
        "ablation_degree_match_count": sum(
            row["ablation_degree_matches"] for row in rows
        ),
        "maximum_prediction_error_by_eta": maxima,
        "maximum_phi3_condition": max(row["maximum_phi3_condition"] for row in rows),
        "endpoint_opened": any(row["endpoint_opened"] for row in rows),
        "confirmation_opened": False,
        "rows": rows,
    }


__all__ = [
    "CALIBRATION_SEEDS",
    "DEVELOPMENT_SEEDS",
    "FOLD_SCHEDULE",
    "generate_degree_id_inputs",
    "analyze_degree_id_row",
    "analyze_degree_id_artifact",
]
