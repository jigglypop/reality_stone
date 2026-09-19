"""BA-TR15: bounded local attenuation compensation for delayed binding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from ..runtime import RuntimeMode
from .runtime_experience_delayed_binding import (
    PAIR_TICKS,
    PRESYNAPTIC_EVENT_THRESHOLD,
    _blocks,
    _delivered_packet,
    _evaluate,
    _external,
    _random_same_norm_snapshot,
    _runtime,
    _seal,
    _uniform_source_snapshot,
)
from .runtime_local_stochastic_binding import (
    LocalStochasticBindingConfig,
    run_local_stochastic_binding_seed,
)


CALIBRATION_SEEDS = (102001,)
DEVELOPMENT_SEEDS = tuple(range(102101, 102117))
PACKET_REFERENCE = 1e-4
PACKET_EPSILON = 1e-12
MAX_COMPENSATION = 16.0
LEARNING_RATE = 1.0
MAX_EDGE_WRITE = 13.0
MAX_WRITE_NORM = 26.0
MIN_COMPENSATED_MARGIN = 2e-5


def _matrix_hash(matrix: Sequence[Sequence[float]]) -> str:
    payload = json.dumps(matrix, separators=(",", ":"), sort_keys=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def generate_fresh_inputs(seeds: Sequence[int]) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        result = run_local_stochastic_binding_seed(
            int(seed), config=LocalStochasticBindingConfig(seed=int(seed))
        )
        weights = result["learned"]["candidate_weights"]
        rows.append({
            "seed": int(seed),
            "producer_status": result["status"],
            "producer_gates": result["gates"],
            "candidate_weights": weights,
            "candidate_weights_sha256": _matrix_hash(weights),
            "producer_snapshot_sha256": result["learned"]["trained_snapshot_sha256"],
        })
    ready = bool(rows) and all(
        row["producer_status"] == "LOCAL_STOCHASTIC_WEIGHT_CODE_PASS"
        and all(bool(value) for value in row["producer_gates"].values())
        for row in rows
    )
    return {
        "status": "FRESH_INPUTS_READY" if ready else "FRESH_INPUTS_STOP",
        "seeds": [int(seed) for seed in seeds],
        "row_count": len(rows),
        "source_endpoint_opened": False,
        "rows": rows,
    }


def _compensation_vector(packet: torch.Tensor, trace: torch.Tensor) -> torch.Tensor:
    p = torch.as_tensor(packet, dtype=torch.float32).view(4).clamp_min(0.0)
    z = torch.as_tensor(trace, dtype=torch.float32).view(4).clamp(0.0, 1.0)
    present = (p > PACKET_EPSILON).to(torch.float32)
    ratio = (PACKET_REFERENCE / (PACKET_EPSILON + p)).clamp(1.0, MAX_COMPENSATION)
    return z * present * ratio


def _experience_block_compensated(
    B: torch.Tensor,
    *,
    condition: str,
) -> dict[str, Any]:
    allowed = {
        "compensated",
        "attenuation_off",
        "packet_shuffle",
        "target_shuffle",
        "time_reversed",
        "no_target",
        "zero_eta",
    }
    if condition not in allowed:
        raise ValueError("unknown compensated experience condition")
    runtime = _runtime(B)
    source, hidden, target = _blocks()
    source_idx = torch.tensor(source)
    hidden_idx = torch.tensor(hidden)
    target_idx = torch.tensor(target)
    support = torch.zeros(20, 20, dtype=torch.bool)
    support[target_idx[:, None], hidden_idx] = True
    initial = runtime.weight.detach().clone()
    episode_state: list[dict[str, Any]] = []
    mid_block_unchanged = True

    for source_slot in range(4):
        runtime.reset_evaluation_state()
        target_slot = (source_slot + 1) % 4 if condition == "target_shuffle" else source_slot
        hidden_history = []
        trace = torch.zeros(4)
        packet = torch.zeros(4)
        post = torch.zeros(4)
        for tick in range(PAIR_TICKS):
            if condition == "time_reversed":
                external_index = target[target_slot] if tick == 0 else (
                    source[source_slot] if tick == 6 else None
                )
            else:
                external_index = source[source_slot] if tick == 0 else (
                    target[target_slot]
                    if tick == 6 and condition != "no_target"
                    else None
                )
            delivered = _delivered_packet(runtime)
            runtime.step(
                external_input=_external(external_index),
                force_mode=RuntimeMode.WAKE,
                learning_signal=0.0,
            )
            hidden_history.append(runtime.activation[hidden_idx].detach().clone())
            if tick == 3:
                trace = (
                    runtime.activation[hidden_idx].detach()
                    > PRESYNAPTIC_EVENT_THRESHOLD
                ).to(torch.float32)
            if tick == 6:
                packet = delivered[hidden_idx].clamp_min(0.0)
                post = runtime.activation[target_idx].detach().clamp_min(0.0)
            mid_block_unchanged = mid_block_unchanged and torch.equal(runtime.weight, initial)
        packet_scalar = float((packet * trace).sum().item())
        episode_state.append({
            "source_slot": source_slot,
            "experienced_target_slot": target_slot,
            "trace": trace,
            "packet": packet,
            "post": post,
            "packet_scalar": packet_scalar,
            "hidden_prearrival_max": float(torch.stack(hidden_history[:3]).abs().max().item()),
            "hidden_first_arrival": hidden_history[3],
        })

    accumulator = torch.zeros(4, 4)
    receipts = []
    packet_scalars = [float(row["packet_scalar"]) for row in episode_state]
    for index, row in enumerate(episode_state):
        trace = row["trace"]
        packet = row["packet"]
        if condition == "attenuation_off":
            factor = trace * (packet > PACKET_EPSILON).to(torch.float32)
        elif condition == "packet_shuffle":
            shifted = packet_scalars[(index + 1) % len(packet_scalars)]
            scalar_factor = 0.0 if shifted <= PACKET_EPSILON else max(
                1.0, min(MAX_COMPENSATION, PACKET_REFERENCE / (PACKET_EPSILON + shifted))
            )
            factor = trace * (packet > PACKET_EPSILON).to(torch.float32) * scalar_factor
        else:
            factor = _compensation_vector(packet, trace)
        accumulator += torch.outer(row["post"], factor)
        positive_factor = factor[factor > 0.0]
        receipts.append({
            "source_slot": int(row["source_slot"]),
            "experienced_target_slot": int(row["experienced_target_slot"]),
            "hidden_prearrival_max": float(row["hidden_prearrival_max"]),
            "hidden_first_arrival": [float(value) for value in row["hidden_first_arrival"].tolist()],
            "presynaptic_event_trace": [float(value) for value in trace.tolist()],
            "terminal_packet": [float(value) for value in packet.tolist()],
            "terminal_packet_scalar": float(row["packet_scalar"]),
            "target_post": [float(value) for value in row["post"].tolist()],
            "compensation_factor": [float(value) for value in factor.tolist()],
            "positive_factor_min": float(positive_factor.min().item()) if positive_factor.numel() else 0.0,
            "positive_factor_max": float(positive_factor.max().item()) if positive_factor.numel() else 0.0,
        })

    effective_lr = 0.0 if condition == "zero_eta" else LEARNING_RATE
    raw_block_unclipped = effective_lr * accumulator
    raw_block = raw_block_unclipped.clamp(0.0, MAX_EDGE_WRITE)
    raw_delta = torch.zeros_like(runtime.weight)
    raw_delta[target_idx[:, None], hidden_idx] = raw_block
    raw_norm = float(raw_delta.norm().item())
    if raw_norm > MAX_WRITE_NORM + 1e-7:
        raise RuntimeError("raw compensated write exceeds the frozen no-scaling bound")
    installed_norm = 0.0
    if raw_norm > 0.0:
        installed_norm = runtime.install_bounded_recurrent_delta(
            raw_delta, max_frobenius_norm=MAX_WRITE_NORM
        )
    actual_delta = runtime.weight - initial
    return {
        "runtime": runtime,
        "condition": condition,
        "accumulator_norm": float(accumulator.norm().item()),
        "raw_unclipped_max": float(raw_block_unclipped.max().item()),
        "edge_cap_hit_count": int(torch.count_nonzero(raw_block_unclipped > MAX_EDGE_WRITE).item()),
        "raw_delta_norm": raw_norm,
        "installed_delta_norm": installed_norm,
        "raw_install_max_error": float((actual_delta - raw_delta).abs().max().item()),
        "outside_support_delta_norm": float(actual_delta[~support].norm().item()),
        "mid_block_weight_unchanged": mid_block_unchanged,
        "block_boundary_count": 1,
        "mutation_count": int(raw_norm > 0.0),
        "source_hidden_weight_unchanged": bool(
            torch.equal(actual_delta[hidden_idx[:, None], source_idx], torch.zeros(4, 4))
        ),
        "episodes": receipts,
    }


def analyze_fresh_row(seed: int, B: torch.Tensor) -> dict[str, Any]:
    learned = _experience_block_compensated(B, condition="compensated")
    learned_snapshot, cutoff = _seal(learned["runtime"])
    learned_eval = _evaluate(learned_snapshot, (0, 1, 2, 3))

    controls: dict[str, Any] = {}
    for condition in (
        "attenuation_off",
        "packet_shuffle",
        "zero_eta",
        "target_shuffle",
        "time_reversed",
        "no_target",
    ):
        block = _experience_block_compensated(B, condition=condition)
        snapshot, control_cutoff = _seal(block["runtime"])
        controls[condition] = {
            "block": {key: value for key, value in block.items() if key != "runtime"},
            "cutoff": control_cutoff,
            "evaluation": _evaluate(snapshot, (0, 1, 2, 3)),
        }
        if condition == "target_shuffle":
            controls[condition]["experienced_mapping_accuracy"] = _evaluate(
                snapshot, (1, 2, 3, 0)
            )["accuracy"]

    uniform_snapshot = _uniform_source_snapshot(learned_snapshot)
    controls["source_code_reset"] = {
        "evaluation": _evaluate(uniform_snapshot, (0, 1, 2, 3))
    }
    random_snapshot = _random_same_norm_snapshot(B, learned_snapshot, int(seed))
    controls["random_same_norm"] = {
        "evaluation": _evaluate(random_snapshot, (0, 1, 2, 3))
    }

    learned_receipt = {key: value for key, value in learned.items() if key != "runtime"}
    learned_min_margin = min(float(row["decode_margin"]) for row in learned_eval["probes"])
    attenuation_eval = controls["attenuation_off"]["evaluation"]
    attenuation_min_margin = min(
        float(row["decode_margin"]) for row in attenuation_eval["probes"]
    )
    strongest_identity_control = max(
        float(controls[name]["evaluation"]["accuracy"])
        for name in ("zero_eta", "time_reversed", "no_target", "source_code_reset", "random_same_norm")
    )
    gates = {
        "true_delay_local_packet": all(
            row["hidden_prearrival_max"] <= 1e-7
            and sum(value > 0.0 for value in row["presynaptic_event_trace"]) == 1
            and row["terminal_packet_scalar"] > PACKET_EPSILON
            for row in learned_receipt["episodes"]
        ),
        "bounded_local_compensation": all(
            1.0 <= row["positive_factor_min"] <= row["positive_factor_max"] <= MAX_COMPENSATION
            for row in learned_receipt["episodes"]
        ) and learned_receipt["edge_cap_hit_count"] == 0,
        "single_support_only_write": bool(
            learned_receipt["mid_block_weight_unchanged"]
            and learned_receipt["block_boundary_count"] == 1
            and learned_receipt["mutation_count"] == 1
            and learned_receipt["outside_support_delta_norm"] == 0.0
            and learned_receipt["source_hidden_weight_unchanged"]
        ),
        "raw_install_parity": learned_receipt["raw_install_max_error"] <= 1e-7,
        "compensated_zero_store_recall": bool(
            learned_eval["accuracy"] == 1.0
            and learned_min_margin >= MIN_COMPENSATED_MARGIN
            and learned_eval["stores_zero"]
            and learned_eval["order_parity"]
            and learned_eval["snapshot_immutable"]
        ),
        "target_shuffle_follows_experience": bool(
            controls["target_shuffle"]["evaluation"]["accuracy"] == 0.0
            and controls["target_shuffle"]["experienced_mapping_accuracy"] == 1.0
        ),
        "zero_temporal_and_source_controls_fail": all(
            controls[name]["evaluation"]["accuracy"] == 0.0
            for name in ("zero_eta", "time_reversed", "no_target", "source_code_reset")
        ),
        "random_same_norm_not_equivalent": controls["random_same_norm"]["evaluation"]["accuracy"] <= 0.5,
        "stores_physically_cut_off": bool(
            cutoff["temporal_rows_after"] == 0 and cutoff["hippocampal_rows_after"] == 0
        ),
    }
    return {
        "seed": int(seed),
        "status": "ATTENUATION_COMPENSATED_BINDING_PASS" if all(gates.values()) else "ATTENUATION_COMPENSATED_BINDING_FAIL",
        "gates": gates,
        "learned_block": learned_receipt,
        "learned_cutoff": cutoff,
        "learned_evaluation": learned_eval,
        "learned_minimum_margin": learned_min_margin,
        "attenuation_off_accuracy": attenuation_eval["accuracy"],
        "attenuation_off_minimum_margin": attenuation_min_margin,
        "strongest_identity_control_accuracy": strongest_identity_control,
        "controls": controls,
        "endpoint_opened": False,
        "claim_scope": "synthetic bounded local attenuation-compensated delayed association",
    }


def analyze_fresh_input_artifact(path: str | Path, *, stage: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "FRESH_INPUTS_READY":
        raise RuntimeError("fresh source-code inputs did not pass their producer gates")
    rows = [
        analyze_fresh_row(int(row["seed"]), torch.tensor(row["candidate_weights"]))
        for row in payload["rows"]
    ]
    all_rows_pass = all(row["status"] == "ATTENUATION_COMPENSATED_BINDING_PASS" for row in rows)
    compensated_pass_count = sum(row["learned_evaluation"]["accuracy"] == 1.0 for row in rows)
    attenuation_pass_count = sum(row["attenuation_off_accuracy"] == 1.0 for row in rows)
    minimum_margin = min(row["learned_minimum_margin"] for row in rows)
    minimum_off_margin = min(row["attenuation_off_minimum_margin"] for row in rows)
    comparative_gate = bool(
        compensated_pass_count > attenuation_pass_count
        and minimum_margin > minimum_off_margin
    )
    if stage == "calibration":
        go = len(rows) == 1 and all_rows_pass
        status = "ATTENUATION_CALIBRATION_PASS" if go else "ATTENUATION_CALIBRATION_STOP"
    elif stage == "development":
        go = len(rows) == len(DEVELOPMENT_SEEDS) and all_rows_pass and comparative_gate
        status = "ATTENUATION_BINDING_DEVELOPMENT_GO" if go else "ATTENUATION_BINDING_STOP"
    else:
        raise ValueError("stage must be calibration or development")
    return {
        "status": status,
        "stage": stage,
        "seed_count": len(rows),
        "pass_count": sum(row["status"] == "ATTENUATION_COMPENSATED_BINDING_PASS" for row in rows),
        "compensated_identity_pass_count": compensated_pass_count,
        "attenuation_off_identity_pass_count": attenuation_pass_count,
        "minimum_compensated_margin": minimum_margin,
        "minimum_attenuation_off_margin": minimum_off_margin,
        "comparative_gate": comparative_gate,
        "maximum_identity_control_accuracy": max(row["strongest_identity_control_accuracy"] for row in rows),
        "maximum_raw_install_error": max(row["learned_block"]["raw_install_max_error"] for row in rows),
        "endpoint_opened": False,
        "confirmation_opened": False,
        "rows": rows,
    }

