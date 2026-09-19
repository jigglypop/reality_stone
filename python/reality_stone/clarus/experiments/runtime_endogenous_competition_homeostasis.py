"""BA-TR9: endogenous delayed competition/homeostasis, endpoint closed.

The experiment keeps one BrainRuntime alive across four source pulses.  A
seed-only microscopic edge code supplies the first symmetry break.  Runtime
state then performs max-relative continuous competition and delayed usage
homeostasis.  Winners are observational receipts only and never feed back into
the runtime transition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import torch

from ..runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode
from .runtime_context_branch_routing import (
    ApparatusInvalid,
    _snapshot_hash,
    architectural_blocks,
)
from .runtime_source_seeded_competition import (
    _move_old_rows_to_new,
    _source_independent_bias,
    _tensor_hash,
    seeded_edge_code,
)


CALIBRATION_SEED = 97091
DEVELOPMENT_SEEDS = tuple(range(98201, 98217))
CONFIRMATION_SEEDS = tuple(range(101801, 101833))


@dataclass(frozen=True)
class EndogenousCompetitionConfig:
    seed: int = CALIBRATION_SEED
    dim: int = 20
    width: int = 4
    axon_delay_ticks: int = 2
    cue_drive_gain: float = 5.0
    heterogeneity_epsilon: float = 0.20
    lateral_gain: float = 1.0
    homeostasis_gain: float = 1.0
    homeostasis_rate: float = 1.0
    homeostasis_decay: float = 0.0
    novelty_decay: float = 0.8
    competition_delay_ticks: int = 1
    competition_epsilon: float = 1e-8
    min_winner_margin: float = 1e-6
    min_positive_activation: float = 1e-8
    washout_tolerance: float = 1e-5
    max_washout_ticks: int = 512
    equality_tolerance: float = 2e-7

    def __post_init__(self) -> None:
        if self.dim != 5 * self.width or self.width != 4:
            raise ValueError("the frozen fixture requires five width-four blocks")
        if self.axon_delay_ticks != 2 or self.competition_delay_ticks != 1:
            raise ValueError("the frozen fixture requires axon delay two and usage delay one")
        finite = (
            self.cue_drive_gain,
            self.heterogeneity_epsilon,
            self.lateral_gain,
            self.homeostasis_gain,
            self.homeostasis_rate,
            self.homeostasis_decay,
            self.novelty_decay,
            self.competition_epsilon,
            self.min_winner_margin,
            self.min_positive_activation,
            self.washout_tolerance,
            self.equality_tolerance,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise ValueError("configuration values must be finite")
        max_level = 3.0 / math.sqrt(20.0)
        if not 0.0 < self.heterogeneity_epsilon < 1.0 / max_level:
            raise ValueError("heterogeneity must keep candidate weights positive")
        if self.lateral_gain != 1.0:
            raise ValueError("the frozen max-relative competition gain is one")
        if self.homeostasis_gain < 0.0:
            raise ValueError("homeostasis gain must be nonnegative")
        if not 0.0 <= self.homeostasis_rate <= 1.0:
            raise ValueError("homeostasis rate must be in [0, 1]")
        if not 0.0 <= self.homeostasis_decay <= 1.0:
            raise ValueError("homeostasis decay must be in [0, 1]")
        if not 0.0 <= self.novelty_decay < 1.0:
            raise ValueError("novelty decay must be in [0, 1)")
        if min(
            self.competition_epsilon,
            self.min_winner_margin,
            self.min_positive_activation,
            self.washout_tolerance,
        ) <= 0.0:
            raise ValueError("stabilizers and tolerances must be positive")
        if int(self.max_washout_ticks) < 1:
            raise ValueError("max_washout_ticks must be positive")


def _candidate_support(
    dim: int,
) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    blocks = architectural_blocks(int(dim))
    source = tuple(int(value) for value in blocks[0])
    hidden = tuple(int(value) for value in blocks[2])
    mask = torch.zeros(dim, dim, dtype=torch.bool)
    mask[torch.tensor(hidden)[:, None], torch.tensor(source)] = True
    return mask, source, hidden


def _runtime_config(
    config: EndogenousCompetitionConfig,
    hidden: Sequence[int],
) -> BrainRuntimeConfig:
    return BrainRuntimeConfig(
        dim=config.dim,
        active_ratio=1.0,
        active_threshold=0.22,
        bit_lower_threshold=0.10,
        bit_upper_threshold=0.30,
        noise_sigma=0.0,
        dale_law=False,
        axon_delay=True,
        max_axon_delay=config.axon_delay_ticks,
        f1_self_measure=False,
        stdp_enabled=False,
        memory_capacity=1,
        hippocampal_encoding_enabled=False,
        competition_indices=tuple(int(value) for value in hidden),
        competition_lateral_gain=config.lateral_gain,
        competition_homeostasis_gain=config.homeostasis_gain,
        competition_homeostasis_rate=config.homeostasis_rate,
        competition_homeostasis_decay=config.homeostasis_decay,
        competition_novelty_decay=config.novelty_decay,
        competition_delay_ticks=config.competition_delay_ticks,
        competition_epsilon=config.competition_epsilon,
    )


def _build_source_snapshot(
    config: EndogenousCompetitionConfig,
    code: torch.Tensor,
) -> tuple[Any, tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    candidate, source, hidden = _candidate_support(config.dim)
    packed = torch.as_tensor(code, dtype=torch.float64)
    if packed.shape != (config.width, config.width) or not torch.isfinite(packed).all():
        raise ApparatusInvalid("APPARATUS_INVALID: invalid source edge code")
    weights = 1.0 + config.heterogeneity_epsilon * packed
    if torch.any(weights <= 0.0):
        raise ApparatusInvalid("APPARATUS_INVALID: nonpositive candidate weight")
    matrix = torch.zeros(config.dim, config.dim)
    matrix[torch.tensor(hidden)[:, None], torch.tensor(source)] = weights.float()
    runtime = BrainRuntime(
        matrix,
        config=_runtime_config(config, hidden),
        backend="torch",
        device="cpu",
    )
    runtime.reset_evaluation_state()
    snapshot = runtime.snapshot()
    values = snapshot.weight[candidate].double()
    receipt = {
        "candidate_edges": int(candidate.sum().item()),
        "outside_candidate_nonzero": int(torch.count_nonzero(snapshot.weight[~candidate]).item()),
        "candidate_min": float(values.min().item()),
        "candidate_max": float(values.max().item()),
        "code_column_sums": [float(value) for value in packed.sum(dim=0).tolist()],
        "code_column_norms": [float(value) for value in packed.norm(dim=0).tolist()],
        "code_unique_per_column": [
            int(torch.unique(packed[:, column]).numel()) for column in range(config.width)
        ],
        "code_sha256": _tensor_hash(packed),
        "snapshot_sha256": _snapshot_hash(snapshot),
        "scalar_thresholds_only": bool(
            snapshot.config.neuronwise_active_threshold is None
            and snapshot.config.neuronwise_bit_lower_threshold is None
            and snapshot.config.neuronwise_bit_upper_threshold is None
        ),
        "competition_state_in_snapshot": bool(
            snapshot.competition_homeostasis is not None
            and snapshot.competition_usage_buffer is not None
            and snapshot.competition_packet_envelope is not None
        ),
        "output_weight_count": 0,
        "hidden_pulse_count": 0,
        "decoder_read_count": 0,
        "reward_read_count": 0,
        "endpoint_read_count": 0,
    }
    return snapshot, source, hidden, receipt


def _strict_observational_winner(
    hidden_activation: torch.Tensor,
    config: EndogenousCompetitionConfig,
) -> tuple[int | None, float]:
    packed = torch.as_tensor(hidden_activation, dtype=torch.float64).view(config.width)
    sorted_values, sorted_indices = torch.sort(packed, descending=True, stable=True)
    margin = float((sorted_values[0] - sorted_values[1]).item())
    if (
        not math.isfinite(margin)
        or float(sorted_values[0].item()) <= config.min_positive_activation
        or margin < config.min_winner_margin
    ):
        return None, margin
    return int(sorted_indices[0].item()), margin


def _fast_positive_residual(runtime: BrainRuntime) -> dict[str, float]:
    activation = float(runtime.activation.clamp_min(0.0).max().item())
    axon = 0.0 if runtime._delay_buffer is None else float(
        runtime._delay_buffer.clamp_min(0.0).max().item()
    )
    usage = 0.0 if runtime._competition_usage_buffer is None else float(
        runtime._competition_usage_buffer.abs().max().item()
    )
    envelope = 0.0 if runtime.competition_packet_envelope is None else float(
        runtime.competition_packet_envelope.item()
    )
    return {
        "positive_activation": activation,
        "positive_axon_packet": axon,
        "usage_packet": usage,
        "packet_envelope": envelope,
    }


def _washout(runtime: BrainRuntime, config: EndogenousCompetitionConfig) -> dict[str, Any]:
    zero = torch.zeros(config.dim)
    for tick in range(1, config.max_washout_ticks + 1):
        runtime.step(
            external_input=zero,
            force_mode=RuntimeMode.WAKE,
            learning_signal=0.0,
        )
        residual = _fast_positive_residual(runtime)
        if max(residual.values()) <= config.washout_tolerance:
            return {
                "washout_ticks": tick,
                "residual": residual,
                "passed": True,
            }
    return {
        "washout_ticks": config.max_washout_ticks,
        "residual": _fast_positive_residual(runtime),
        "passed": False,
    }


def _source_episode(
    runtime: BrainRuntime,
    source_index: int,
    hidden: Sequence[int],
    config: EndogenousCompetitionConfig,
) -> dict[str, Any]:
    hidden_index = torch.tensor(tuple(int(value) for value in hidden), dtype=torch.long)
    histories: list[torch.Tensor] = []
    assert runtime.competition_homeostasis is not None
    homeostasis_before = runtime.competition_homeostasis[hidden_index].detach().clone()
    envelope_before = float(runtime.competition_packet_envelope.item())
    for relative_tick in range(config.axon_delay_ticks + 2):
        external = torch.zeros(config.dim)
        if relative_tick == 0:
            external[int(source_index)] = config.cue_drive_gain
        runtime.step(
            external_input=external,
            force_mode=RuntimeMode.WAKE,
            learning_signal=0.0,
        )
        histories.append(runtime.activation[hidden_index].detach().clone())
    arrival = histories[-1]
    winner, margin = _strict_observational_winner(arrival, config)
    homeostasis_at_arrival = (
        runtime.competition_homeostasis[hidden_index].detach().clone()
    )
    washout = _washout(runtime, config)
    assert runtime.competition_homeostasis is not None
    return {
        "source_index": int(source_index),
        "pulse_ticks": [0],
        "observed_ticks": list(range(config.axon_delay_ticks + 2)),
        "prearrival_positive_hidden_max": float(
            torch.stack(histories[:-1]).clamp_min(0.0).max().item()
        ),
        "arrival_hidden": [float(value) for value in arrival.tolist()],
        "winner": winner,
        "winner_margin": margin,
        "homeostasis_before": [float(value) for value in homeostasis_before.tolist()],
        "homeostasis_at_arrival": [
            float(value) for value in homeostasis_at_arrival.tolist()
        ],
        "homeostasis_after_washout": [
            float(value)
            for value in runtime.competition_homeostasis[hidden_index].tolist()
        ],
        "packet_envelope_before": envelope_before,
        "washout": washout,
        "hidden_pulse_count": 0,
        "output_pulse_count": 0,
        "decoder_read_count": 0,
        "reward_read_count": 0,
        "endpoint_read_count": 0,
    }


def _run_tail(
    runtime: BrainRuntime,
    source: Sequence[int],
    hidden: Sequence[int],
    source_slots: Sequence[int],
    config: EndogenousCompetitionConfig,
) -> list[dict[str, Any]]:
    return [
        _source_episode(runtime, int(source[int(slot)]), hidden, config)
        for slot in source_slots
    ]


def _allocation_summary(
    episodes: Sequence[dict[str, Any]],
    source: Sequence[int],
    hidden: Sequence[int],
) -> dict[str, Any]:
    source_to_slot = {int(value): index for index, value in enumerate(source)}
    winner_by_source = [-1] * len(source)
    winners: list[int] = []
    margins: list[float] = []
    for episode in episodes:
        winner = episode["winner"]
        if winner is None:
            continue
        slot = source_to_slot[int(episode["source_index"])]
        winner_by_source[slot] = int(winner)
        winners.append(int(winner))
        margins.append(float(episode["winner_margin"]))
    unique = len(set(winners))
    return {
        "winner_by_source": winner_by_source,
        "winner_coordinates": [
            -1 if value < 0 else int(hidden[value]) for value in winner_by_source
        ],
        "winner_margins": margins,
        "abstention_count": len(source) - len(winners),
        "collision_fraction": 1.0 - unique / len(source),
        "is_bijection": bool(
            len(winners) == len(source) and unique == len(source)
        ),
    }


def _run_curriculum(
    snapshot: Any,
    source: Sequence[int],
    hidden: Sequence[int],
    source_order: Sequence[int],
    config: EndogenousCompetitionConfig,
) -> dict[str, Any]:
    order = tuple(int(value) for value in source_order)
    if sorted(order) != list(range(config.width)):
        raise ApparatusInvalid("APPARATUS_INVALID: source order must be a permutation")
    runtime = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    first = _run_tail(runtime, source, hidden, order[:2], config)
    midpoint = runtime.snapshot()
    branch_a = BrainRuntime.from_snapshot(midpoint, backend="torch", device="cpu")
    branch_b = BrainRuntime.from_snapshot(midpoint, backend="torch", device="cpu")
    tail_a = _run_tail(branch_a, source, hidden, order[2:], config)
    tail_b = _run_tail(branch_b, source, hidden, order[2:], config)
    hash_a = _snapshot_hash(branch_a.snapshot())
    hash_b = _snapshot_hash(branch_b.snapshot())
    episodes = [*first, *tail_a]
    return {
        "source_order": list(order),
        "episodes": episodes,
        "allocation": _allocation_summary(episodes, source, hidden),
        "midpoint_snapshot_sha256": _snapshot_hash(midpoint),
        "continuation_hash_a": hash_a,
        "continuation_hash_b": hash_b,
        "snapshot_continuation_exact": bool(hash_a == hash_b and tail_a == tail_b),
        "final_snapshot_sha256": hash_a,
    }


def _move_vector_old_to_new(
    values: Sequence[float],
    old_to_new: torch.Tensor,
) -> list[float]:
    moved = [0.0] * len(values)
    for old, value in enumerate(values):
        moved[int(old_to_new[old].item())] = float(value)
    return moved


def _row_covariance(
    reference: dict[str, Any],
    permuted: dict[str, Any],
    old_to_new: torch.Tensor,
    tolerance: float,
) -> bool:
    expected_winners = [
        -1 if value < 0 else int(old_to_new[value].item())
        for value in reference["allocation"]["winner_by_source"]
    ]
    if permuted["allocation"]["winner_by_source"] != expected_winners:
        return False
    for left, right in zip(reference["episodes"], permuted["episodes"]):
        expected_arrival = _move_vector_old_to_new(left["arrival_hidden"], old_to_new)
        expected_homeostasis = _move_vector_old_to_new(
            left["homeostasis_after_washout"], old_to_new,
        )
        if max(abs(a - b) for a, b in zip(expected_arrival, right["arrival_hidden"])) > tolerance:
            return False
        if max(
            abs(a - b)
            for a, b in zip(expected_homeostasis, right["homeostasis_after_washout"])
        ) > tolerance:
            return False
    return True


def _serializable_curriculum(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_order": value["source_order"],
        "episodes": value["episodes"],
        "allocation": value["allocation"],
        "midpoint_snapshot_sha256": value["midpoint_snapshot_sha256"],
        "continuation_hash_a": value["continuation_hash_a"],
        "continuation_hash_b": value["continuation_hash_b"],
        "snapshot_continuation_exact": value["snapshot_continuation_exact"],
        "final_snapshot_sha256": value["final_snapshot_sha256"],
    }


def run_endogenous_competition_seed(
    seed: int = CALIBRATION_SEED,
    *,
    config: EndogenousCompetitionConfig | None = None,
) -> dict[str, Any]:
    selected = config or EndogenousCompetitionConfig(seed=int(seed))
    config = EndogenousCompetitionConfig(**{**asdict(selected), "seed": int(seed)})
    heterogeneity_seed = int(seed) + 91_300_019
    order_seed = int(seed) + 93_100_033
    code = seeded_edge_code(heterogeneity_seed)
    uniform = torch.zeros_like(code)
    bias = _source_independent_bias(heterogeneity_seed)
    shift = 1 + int(seed) % 3
    old_to_new = torch.remainder(torch.arange(config.width) + shift, config.width)
    permuted_code = _move_old_rows_to_new(code, old_to_new)
    order_generator = torch.Generator(device="cpu").manual_seed(order_seed)
    source_order = tuple(
        int(value) for value in torch.randperm(config.width, generator=order_generator).tolist()
    )

    positive_snapshot, source, hidden, positive_receipt = _build_source_snapshot(config, code)
    positive = _run_curriculum(
        positive_snapshot, source, hidden, source_order, config,
    )

    no_homeostasis_config = EndogenousCompetitionConfig(
        **{**asdict(config), "homeostasis_gain": 0.0}
    )
    no_homeostasis_snapshot, _, _, no_homeostasis_receipt = _build_source_snapshot(
        no_homeostasis_config, code,
    )
    no_homeostasis = _run_curriculum(
        no_homeostasis_snapshot,
        source,
        hidden,
        source_order,
        no_homeostasis_config,
    )

    uniform_snapshot, _, _, uniform_receipt = _build_source_snapshot(config, uniform)
    uniform_runtime = BrainRuntime.from_snapshot(uniform_snapshot, backend="torch", device="cpu")
    uniform_episode = _source_episode(
        uniform_runtime, int(source[source_order[0]]), hidden, config,
    )

    bias_snapshot, _, _, bias_receipt = _build_source_snapshot(config, bias)
    bias_result = _run_curriculum(
        bias_snapshot, source, hidden, source_order, config,
    )

    permuted_snapshot, _, _, permuted_receipt = _build_source_snapshot(
        config, permuted_code,
    )
    permuted = _run_curriculum(
        permuted_snapshot, source, hidden, source_order, config,
    )

    max_level = 3.0 / math.sqrt(20.0)
    b_min = 1.0 - config.heterogeneity_epsilon * max_level
    b_max = 1.0 + config.heterogeneity_epsilon * max_level
    all_episodes = [
        *positive["episodes"],
        *no_homeostasis["episodes"],
        uniform_episode,
        *bias_result["episodes"],
        *permuted["episodes"],
    ]
    no_reads = all(
        episode["hidden_pulse_count"] == 0
        and episode["output_pulse_count"] == 0
        and episode["decoder_read_count"] == 0
        and episode["reward_read_count"] == 0
        and episode["endpoint_read_count"] == 0
        for episode in all_episodes
    )
    uniform_homeostasis = uniform_runtime.competition_homeostasis
    assert uniform_homeostasis is not None
    source_hashes_before = {
        "positive": positive_receipt["snapshot_sha256"],
        "no_homeostasis": no_homeostasis_receipt["snapshot_sha256"],
        "uniform": uniform_receipt["snapshot_sha256"],
        "bias": bias_receipt["snapshot_sha256"],
        "permuted": permuted_receipt["snapshot_sha256"],
    }
    source_hashes_after = {
        "positive": _snapshot_hash(positive_snapshot),
        "no_homeostasis": _snapshot_hash(no_homeostasis_snapshot),
        "uniform": _snapshot_hash(uniform_snapshot),
        "bias": _snapshot_hash(bias_snapshot),
        "permuted": _snapshot_hash(permuted_snapshot),
    }
    gates = {
        "analytic_homeostasis_bound": bool(
            config.homeostasis_gain > math.log(b_max / b_min)
        ),
        "balanced_distinct_seed_code": bool(
            max(abs(value) for value in positive_receipt["code_column_sums"])
            <= config.equality_tolerance
            and all(
                abs(value - 1.0) <= config.equality_tolerance
                for value in positive_receipt["code_column_norms"]
            )
            and positive_receipt["code_unique_per_column"] == [4, 4, 4, 4]
        ),
        "only_source_hidden_weights": all(
            receipt["candidate_edges"] == 16
            and receipt["outside_candidate_nonzero"] == 0
            and receipt["output_weight_count"] == 0
            for receipt in (
                positive_receipt,
                no_homeostasis_receipt,
                uniform_receipt,
                bias_receipt,
                permuted_receipt,
            )
        ),
        "scalar_thresholds_only": all(
            receipt["scalar_thresholds_only"]
            for receipt in (
                positive_receipt,
                no_homeostasis_receipt,
                uniform_receipt,
                bias_receipt,
                permuted_receipt,
            )
        ),
        "runtime_competition_state_snapshotted": all(
            receipt["competition_state_in_snapshot"]
            for receipt in (
                positive_receipt,
                no_homeostasis_receipt,
                uniform_receipt,
                bias_receipt,
                permuted_receipt,
            )
        ),
        "true_delayed_first_arrival": bool(
            max(episode["prearrival_positive_hidden_max"] for episode in all_episodes)
            <= config.washout_tolerance
        ),
        "all_washouts_pass": all(
            bool(episode["washout"]["passed"]) for episode in all_episodes
        ),
        "persistent_homeostasis_bijection": bool(
            positive["allocation"]["is_bijection"]
            and positive["allocation"]["abstention_count"] == 0
            and min(positive["allocation"]["winner_margins"])
            >= config.min_winner_margin
        ),
        "uniform_tie_abstains_without_state_write": bool(
            uniform_episode["winner"] is None
            and float(uniform_homeostasis.abs().max().item()) <= config.equality_tolerance
        ),
        "source_independent_bias_marked_unidentified": bool(
            torch.equal(bias[:, :1].expand_as(bias), bias)
        ),
        "hidden_row_permutation_covariant": _row_covariance(
            positive, permuted, old_to_new, config.equality_tolerance,
        ),
        "snapshot_continuation_exact": bool(
            positive["snapshot_continuation_exact"]
            and no_homeostasis["snapshot_continuation_exact"]
            and bias_result["snapshot_continuation_exact"]
            and permuted["snapshot_continuation_exact"]
        ),
        "source_snapshots_immutable": source_hashes_before == source_hashes_after,
        "no_hidden_output_decoder_reward_endpoint_reads": no_reads,
    }
    apparatus_pass = all(gates.values())
    return {
        "seed": int(seed),
        "status": (
            "ENDOGENOUS_SOURCE_ALLOCATION_PASS"
            if apparatus_pass
            else "APPARATUS_OR_MECHANISM_FAIL"
        ),
        "endpoint_opened": False,
        "output_identity_status": "NONIDENTIFIED_ENDPOINT_CLOSED",
        "config": asdict(config),
        "heterogeneity_seed": heterogeneity_seed,
        "source_order_seed": order_seed,
        "source_order": list(source_order),
        "analytic_bound": {
            "b_min": b_min,
            "b_max": b_max,
            "required_gain": math.log(b_max / b_min),
            "frozen_gain": config.homeostasis_gain,
        },
        "gates": gates,
        "persistent": _serializable_curriculum(positive),
        "no_homeostasis": _serializable_curriculum(no_homeostasis),
        "uniform": {
            "status": "ABSTAIN_BOUNDARY_TIE" if uniform_episode["winner"] is None else "SELECTED",
            "episode": uniform_episode,
            "final_homeostasis": [
                float(value) for value in uniform_homeostasis.tolist()
            ],
        },
        "source_independent_bias": {
            "status": "SOURCE_UNIDENTIFIED",
            "prehistory_columns_identical": bool(
                torch.equal(bias[:, :1].expand_as(bias), bias)
            ),
            "curriculum": _serializable_curriculum(bias_result),
        },
        "row_permuted": {
            "old_to_new": [int(value) for value in old_to_new.tolist()],
            "curriculum": _serializable_curriculum(permuted),
        },
        "collision_fraction": float(positive["allocation"]["collision_fraction"]),
        "no_homeostasis_collision_fraction": float(
            no_homeostasis["allocation"]["collision_fraction"]
        ),
        "source_snapshot_sha256_before": source_hashes_before,
        "source_snapshot_sha256_after": source_hashes_after,
        "confirmation_opened": False,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
    }
