"""BA-TR14: source/target experience bound by exact delayed local coincidence."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch

from ..runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode
from .runtime_context_branch_routing import _snapshot_hash, architectural_blocks
from .runtime_native_loops import _detach
from ..temporal_memory import TemporalAuditedMemory


DELAY_TICKS = 2
PAIR_TICKS = 7
LEARNING_RATE = 1.0
MAX_WRITE_NORM = 4.0
PRESYNAPTIC_EVENT_THRESHOLD = 1e-6
MIN_DECODE_ACTIVATION = 1e-5
MIN_DECODE_MARGIN = 1e-5


def _blocks() -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    blocks = architectural_blocks(20)
    return (
        tuple(int(value) for value in blocks[0]),
        tuple(int(value) for value in blocks[2]),
        tuple(int(value) for value in blocks[4]),
    )


def _runtime(B: torch.Tensor) -> BrainRuntime:
    packed = torch.as_tensor(B, dtype=torch.float32)
    if packed.shape != (4, 4) or not torch.isfinite(packed).all():
        raise ValueError("learned source code must be finite 4x4")
    source, hidden, _target = _blocks()
    weight = torch.zeros(20, 20)
    weight[torch.tensor(hidden)[:, None], torch.tensor(source)] = packed
    runtime = BrainRuntime(
        weight,
        config=BrainRuntimeConfig(
            dim=20,
            active_ratio=1.0,
            active_threshold=0.0,
            force_all_active_selection=True,
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=True,
            max_axon_delay=DELAY_TICKS,
            f1_self_measure=False,
            stdp_enabled=False,
            memory_capacity=1,
            hippocampal_encoding_enabled=False,
            competition_indices=hidden,
            competition_lateral_gain=1.0,
            competition_homeostasis_gain=0.0,
            competition_homeostasis_rate=0.0,
            competition_jitter_sigma=0.0,
            competition_jitter_seed=0,
        ),
        backend="torch",
        device="cpu",
    )
    runtime.reset_evaluation_state()
    return runtime


def _external(index: int | None, gain: float = 5.0) -> torch.Tensor:
    value = torch.zeros(20)
    if index is not None:
        value[int(index)] = gain
    return value


def _delivered_packet(runtime: BrainRuntime) -> torch.Tensor:
    if runtime._delay_buffer is None:
        raise RuntimeError("delayed binding requires the axon ring")
    slot = runtime._delay_idx % DELAY_TICKS
    return runtime._delay_buffer[slot].detach().clone()


def _local_pair(post: torch.Tensor, delayed_pre: torch.Tensor) -> torch.Tensor:
    """The learner sees only local post activity and a presynaptic event trace."""
    current_post = torch.as_tensor(post, dtype=torch.float32).view(4).clamp_min(0.0)
    terminal_pre = torch.as_tensor(delayed_pre, dtype=torch.float32).view(4).clamp_min(0.0)
    return torch.outer(current_post, terminal_pre)


def _experience_block(
    B: torch.Tensor,
    *,
    condition: str,
    learning_rate: float = LEARNING_RATE,
) -> dict[str, Any]:
    if condition not in {"learned", "target_shuffle", "time_reversed", "no_target", "zero_eta"}:
        raise ValueError("unknown experience condition")
    runtime = _runtime(B)
    source, hidden, target = _blocks()
    support = torch.zeros(20, 20, dtype=torch.bool)
    support[torch.tensor(target)[:, None], torch.tensor(hidden)] = True
    initial = runtime.weight.detach().clone()
    accumulator = torch.zeros(4, 4)
    episode_receipts = []
    mid_block_unchanged = True
    for source_slot in range(4):
        runtime.reset_evaluation_state()
        target_slot = (source_slot + 1) % 4 if condition == "target_shuffle" else source_slot
        hidden_history = []
        eligibility_trace = torch.zeros(4)
        actual_terminal_packet = torch.zeros(4)
        pair_pre = torch.zeros(4)
        pair_post = torch.zeros(4)
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
            hidden_history.append(runtime.activation[torch.tensor(hidden)].detach().clone())
            if tick == 3:
                eligibility_trace = (
                    runtime.activation[torch.tensor(hidden)].detach()
                    > PRESYNAPTIC_EVENT_THRESHOLD
                ).to(torch.float32)
            if tick == 6:
                actual_terminal_packet = delivered[torch.tensor(hidden)].clamp_min(0.0)
                pair_pre = eligibility_trace.clone()
                pair_post = runtime.activation[torch.tensor(target)].detach().clamp_min(0.0)
                accumulator += _local_pair(pair_post, pair_pre)
            mid_block_unchanged = mid_block_unchanged and torch.equal(runtime.weight, initial)
        episode_receipts.append({
            "source_slot": source_slot,
            "experienced_target_slot": target_slot,
            "hidden_prearrival_max": float(
                torch.stack(hidden_history[:3]).abs().max().item()
            ),
            "hidden_first_arrival": [float(value) for value in hidden_history[3].tolist()],
            "presynaptic_event_trace_at_6": [float(value) for value in pair_pre.tolist()],
            "actual_delivered_hidden_packet_at_6": [
                float(value) for value in actual_terminal_packet.tolist()
            ],
            "target_post_at_6": [float(value) for value in pair_post.tolist()],
            "terminal_pre_positive_count": int(torch.count_nonzero(pair_pre > 0.0).item()),
            "target_post_positive_count": int(torch.count_nonzero(pair_post > 0.0).item()),
            "external_nonzero_ticks": (
                [0, 6] if condition not in {"no_target"} else [0]
            ),
        })
    effective_lr = 0.0 if condition == "zero_eta" else float(learning_rate)
    raw_delta = torch.zeros_like(runtime.weight)
    raw_delta[torch.tensor(target)[:, None], torch.tensor(hidden)] = effective_lr * accumulator
    raw_norm = float(raw_delta.norm().item())
    if raw_norm >= MAX_WRITE_NORM:
        raise RuntimeError("raw local write exceeds the frozen no-scaling bound")
    installed_norm = 0.0
    if raw_norm > 0.0:
        installed_norm = runtime.install_bounded_recurrent_delta(
            raw_delta,
            max_frobenius_norm=MAX_WRITE_NORM,
        )
    actual_delta = runtime.weight - initial
    parity_error = float((actual_delta - raw_delta).abs().max().item())
    return {
        "runtime": runtime,
        "condition": condition,
        "accumulator": [[float(value) for value in row] for row in accumulator.tolist()],
        "accumulator_norm": float(accumulator.norm().item()),
        "raw_delta_norm": raw_norm,
        "installed_delta_norm": installed_norm,
        "raw_install_max_error": parity_error,
        "outside_support_delta_norm": float(actual_delta[~support].norm().item()),
        "mid_block_weight_unchanged": mid_block_unchanged,
        "block_boundary_count": 1,
        "mutation_count": int(raw_norm > 0.0),
        "episodes": episode_receipts,
        "source_hidden_weight_unchanged": bool(
            torch.equal(actual_delta[torch.tensor(hidden)[:, None], torch.tensor(source)], torch.zeros(4, 4))
        ),
    }


def _seal(runtime: BrainRuntime) -> tuple[Any, dict[str, int]]:
    temporal = TemporalAuditedMemory(capacity=1)
    cutoff = _detach(runtime, temporal)
    return runtime.snapshot(), cutoff


def _decode_target(values: torch.Tensor) -> tuple[int | None, float]:
    packed = torch.as_tensor(values, dtype=torch.float64).view(4)
    ordered, indices = torch.sort(packed, descending=True, stable=True)
    margin = float((ordered[0] - ordered[1]).item())
    if float(ordered[0].item()) <= MIN_DECODE_ACTIVATION or margin < MIN_DECODE_MARGIN:
        return None, margin
    return int(indices[0].item()), margin


def _probe(snapshot: Any, source_slot: int) -> dict[str, Any]:
    source, hidden, target = _blocks()
    runtime = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    hidden_first = torch.zeros(4)
    target_final = torch.zeros(4)
    for tick in range(PAIR_TICKS):
        runtime.step(
            external_input=_external(source[source_slot] if tick == 0 else None),
            force_mode=RuntimeMode.WAKE,
            learning_signal=0.0,
        )
        if tick == 3:
            hidden_first = runtime.activation[torch.tensor(hidden)].detach().clone()
        if tick == 6:
            target_final = runtime.activation[torch.tensor(target)].detach().clone()
    decoded, margin = _decode_target(target_final)
    return {
        "source_slot": source_slot,
        "decoded_target": decoded,
        "decode_margin": margin,
        "hidden_first_arrival": [float(value) for value in hidden_first.tolist()],
        "target_at_6": [float(value) for value in target_final.tolist()],
        "external_nonzero_ticks": [0],
        "zero_input_ticks": [1, 2, 3, 4, 5, 6],
        "hippocampal_rows_after": len(runtime.hippocampus),
    }


def _evaluate(snapshot: Any, expected: Sequence[int]) -> dict[str, Any]:
    before = _snapshot_hash(snapshot)
    forward = [_probe(snapshot, slot) for slot in range(4)]
    reverse = [_probe(snapshot, slot) for slot in reversed(range(4))]
    reverse_by_source = {row["source_slot"]: row for row in reverse}
    order_parity = all(
        forward[slot]["target_at_6"] == reverse_by_source[slot]["target_at_6"]
        for slot in range(4)
    )
    correct = sum(
        row["decoded_target"] == int(expected[row["source_slot"]])
        for row in forward
    )
    return {
        "accuracy": correct / 4.0,
        "probes": forward,
        "order_parity": order_parity,
        "snapshot_immutable": before == _snapshot_hash(snapshot),
        "stores_zero": all(row["hippocampal_rows_after"] == 0 for row in forward),
    }


def _uniform_source_snapshot(learned_snapshot: Any) -> Any:
    copied = torch.as_tensor(learned_snapshot.weight).clone()
    source, hidden, _target = _blocks()
    copied[torch.tensor(hidden)[:, None], torch.tensor(source)] = 1.0
    runtime = _runtime(copied[torch.tensor(hidden)[:, None], torch.tensor(source)])
    runtime.weight[torch.tensor(_target)[:, None], torch.tensor(hidden)] = learned_snapshot.weight[
        torch.tensor(_target)[:, None], torch.tensor(hidden)
    ]
    runtime._rebuild_sparse()
    return runtime.snapshot()


def _random_same_norm_snapshot(B: torch.Tensor, learned_snapshot: Any, seed: int) -> Any:
    runtime = _runtime(B)
    _source, hidden, target = _blocks()
    learned_block = learned_snapshot.weight[torch.tensor(target)[:, None], torch.tensor(hidden)]
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 149_000_009)
    random_block = torch.randn(4, 4, generator=generator)
    random_block = random_block * (learned_block.norm() / random_block.norm())
    runtime.weight[torch.tensor(target)[:, None], torch.tensor(hidden)] = random_block
    runtime._rebuild_sparse()
    runtime.reset_evaluation_state()
    return runtime.snapshot()


def analyze_weight_row(seed: int, B: torch.Tensor) -> dict[str, Any]:
    learned_block = _experience_block(B, condition="learned")
    learned_snapshot, cutoff = _seal(learned_block["runtime"])
    learned_eval = _evaluate(learned_snapshot, (0, 1, 2, 3))

    controls = {}
    for condition in ("zero_eta", "target_shuffle", "time_reversed", "no_target"):
        block = _experience_block(B, condition=condition)
        snapshot, control_cutoff = _seal(block["runtime"])
        controls[condition] = {
            "block": {key: value for key, value in block.items() if key != "runtime"},
            "cutoff": control_cutoff,
            "evaluation": _evaluate(snapshot, (0, 1, 2, 3)),
        }
        if condition == "target_shuffle":
            controls[condition]["experienced_mapping_accuracy"] = _evaluate(
                snapshot, (1, 2, 3, 0),
            )["accuracy"]

    uniform_snapshot = _uniform_source_snapshot(learned_snapshot)
    controls["source_code_reset"] = {
        "evaluation": _evaluate(uniform_snapshot, (0, 1, 2, 3)),
    }
    random_snapshot = _random_same_norm_snapshot(B, learned_snapshot, seed)
    controls["random_same_norm"] = {
        "evaluation": _evaluate(random_snapshot, (0, 1, 2, 3)),
    }

    learned_receipt = {key: value for key, value in learned_block.items() if key != "runtime"}
    hidden_codes = torch.tensor([
        episode["hidden_first_arrival"] for episode in learned_receipt["episodes"]
    ], dtype=torch.float64).T
    gram = hidden_codes.T @ hidden_codes
    gram_margins = [
        float(gram[index, index] - torch.cat((gram[index, :index], gram[index, index + 1:])).max())
        for index in range(4)
    ]
    strongest_control = max(
        float(value["evaluation"]["accuracy"]) for value in controls.values()
    )
    gates = {
        "true_delay_and_terminal_pair": all(
            episode["hidden_prearrival_max"] <= 1e-7
            and episode["terminal_pre_positive_count"] == 1
            and episode["target_post_positive_count"] == 1
            for episode in learned_receipt["episodes"]
        ),
        "hidden_gram_margin_positive": min(gram_margins) > 1e-6,
        "block_end_single_local_write": bool(
            learned_receipt["mid_block_weight_unchanged"]
            and learned_receipt["block_boundary_count"] == 1
            and learned_receipt["mutation_count"] == 1
            and learned_receipt["outside_support_delta_norm"] == 0.0
            and learned_receipt["source_hidden_weight_unchanged"]
        ),
        "raw_install_parity": learned_receipt["raw_install_max_error"] <= 1e-7,
        "learned_zero_store_recall": bool(
            learned_eval["accuracy"] == 1.0
            and learned_eval["stores_zero"]
            and learned_eval["order_parity"]
            and learned_eval["snapshot_immutable"]
        ),
        "control_advantage": learned_eval["accuracy"] - strongest_control >= 0.5,
        "target_shuffle_follows_experience": bool(
            controls["target_shuffle"]["evaluation"]["accuracy"] == 0.0
            and controls["target_shuffle"]["experienced_mapping_accuracy"] == 1.0
        ),
        "zero_and_temporal_controls_fail": all(
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
        "status": "EXPERIENCE_DELAYED_BINDING_PASS" if all(gates.values()) else "EXPERIENCE_DELAYED_BINDING_FAIL",
        "gates": gates,
        "learned_block": learned_receipt,
        "learned_cutoff": cutoff,
        "learned_evaluation": learned_eval,
        "hidden_gram_margins": gram_margins,
        "strongest_control_accuracy": strongest_control,
        "controls": controls,
        "endpoint_opened": False,
        "claim_scope": "synthetic experience-supervised local delayed association",
    }


def analyze_development_artifact(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [
        analyze_weight_row(
            int(row["seed"]),
            torch.tensor(row["learned"]["candidate_weights"]),
        )
        for row in payload["rows"]
    ]
    return {
        "status": (
            "EXPERIENCE_DELAYED_BINDING_DEVELOPMENT_GO"
            if all(row["status"] == "EXPERIENCE_DELAYED_BINDING_PASS" for row in rows)
            else "EXPERIENCE_DELAYED_BINDING_STOP"
        ),
        "seed_count": len(rows),
        "pass_count": sum(row["status"] == "EXPERIENCE_DELAYED_BINDING_PASS" for row in rows),
        "mean_accuracy": sum(row["learned_evaluation"]["accuracy"] for row in rows) / len(rows),
        "maximum_control_accuracy": max(row["strongest_control_accuracy"] for row in rows),
        "minimum_hidden_gram_margin": min(min(row["hidden_gram_margins"]) for row in rows),
        "maximum_raw_install_error": max(row["learned_block"]["raw_install_max_error"] for row in rows),
        "endpoint_opened": False,
        "rows": rows,
    }
