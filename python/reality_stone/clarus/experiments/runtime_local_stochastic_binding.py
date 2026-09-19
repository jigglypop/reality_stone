"""BA-TR10: local stochastic symmetry breaking consolidated into recurrent weights.

All source-to-hidden weights start exactly equal.  Exchangeable multiplicative
jitter is applied only to a genuinely delivered recurrent packet, and a local
Oja update writes the resulting pre/post coincidence into the real recurrent
matrix.  Jitter is disabled for every evaluation probe.  No output, decoder,
reward, target, or semantic identity is present in this experiment.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import torch

from ..runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode
from .runtime_context_branch_routing import ApparatusInvalid, _snapshot_hash, architectural_blocks


CALIBRATION_SEED = 98301
DEVELOPMENT_SEEDS = tuple(range(98501, 98517))
CONFIRMATION_SEEDS = tuple(range(101901, 101933))


@dataclass(frozen=True)
class LocalStochasticBindingConfig:
    seed: int = CALIBRATION_SEED
    dim: int = 20
    width: int = 4
    axon_delay_ticks: int = 2
    cue_drive_gain: float = 5.0
    jitter_sigma: float = 0.35
    oja_lr: float = 4.0
    oja_beta: float = 1.0
    weight_min: float = 0.20
    weight_max: float = 2.00
    max_update_norm: float = 1.0
    lateral_gain: float = 1.0
    homeostasis_gain: float = 1.0
    homeostasis_rate: float = 1.0
    homeostasis_decay: float = 0.0
    novelty_decay: float = 0.8
    competition_delay_ticks: int = 1
    competition_epsilon: float = 1e-8
    min_winner_margin: float = 1e-6
    min_positive_activation: float = 1e-8
    min_column_distance: float = 1e-4
    washout_tolerance: float = 1e-5
    max_washout_ticks: int = 512

    def __post_init__(self) -> None:
        if self.dim != 5 * self.width or self.width != 4:
            raise ValueError("the frozen fixture requires five width-four blocks")
        if self.axon_delay_ticks != 2 or self.competition_delay_ticks != 1:
            raise ValueError("the frozen fixture requires axon delay two and usage delay one")
        finite = (
            self.cue_drive_gain,
            self.jitter_sigma,
            self.oja_lr,
            self.oja_beta,
            self.weight_min,
            self.weight_max,
            self.max_update_norm,
            self.lateral_gain,
            self.homeostasis_gain,
            self.homeostasis_rate,
            self.homeostasis_decay,
            self.novelty_decay,
            self.competition_epsilon,
            self.min_winner_margin,
            self.min_positive_activation,
            self.min_column_distance,
            self.washout_tolerance,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise ValueError("configuration values must be finite")
        if not 0.0 <= self.jitter_sigma < 1.0:
            raise ValueError("jitter_sigma must be in [0, 1)")
        if self.oja_lr < 0.0 or self.oja_beta < 0.0:
            raise ValueError("jitter and learning coefficients must be nonnegative")
        if not 0.0 < self.weight_min < 1.0 < self.weight_max:
            raise ValueError("weight bounds must strictly contain the uniform initial weight")
        if self.max_update_norm <= 0.0:
            raise ValueError("max_update_norm must be positive")
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
            self.min_column_distance,
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
    config: LocalStochasticBindingConfig,
    hidden: Sequence[int],
    *,
    jitter_sigma: float,
    homeostasis_gain: float | None = None,
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
        competition_homeostasis_gain=(
            config.homeostasis_gain if homeostasis_gain is None else float(homeostasis_gain)
        ),
        competition_homeostasis_rate=config.homeostasis_rate,
        competition_homeostasis_decay=config.homeostasis_decay,
        competition_novelty_decay=config.novelty_decay,
        competition_delay_ticks=config.competition_delay_ticks,
        competition_epsilon=config.competition_epsilon,
        competition_jitter_sigma=float(jitter_sigma),
        competition_jitter_seed=int(config.seed) + 121_000_003,
    )


def _uniform_source_snapshot(
    config: LocalStochasticBindingConfig,
    *,
    jitter_sigma: float,
    homeostasis_gain: float | None = None,
) -> tuple[Any, tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    candidate, source, hidden = _candidate_support(config.dim)
    weight = torch.zeros(config.dim, config.dim)
    weight[torch.tensor(hidden)[:, None], torch.tensor(source)] = 1.0
    runtime = BrainRuntime(
        weight,
        config=_runtime_config(
            config,
            hidden,
            jitter_sigma=jitter_sigma,
            homeostasis_gain=homeostasis_gain,
        ),
        backend="torch",
        device="cpu",
    )
    runtime.reset_evaluation_state()
    snapshot = runtime.snapshot()
    values = snapshot.weight[candidate]
    receipt = {
        "candidate_edges": int(candidate.sum().item()),
        "candidate_all_exactly_one": bool(torch.equal(values, torch.ones_like(values))),
        "candidate_unique_values": int(torch.unique(values).numel()),
        "outside_candidate_nonzero": int(torch.count_nonzero(snapshot.weight[~candidate]).item()),
        "jitter_sigma": float(snapshot.config.competition_jitter_sigma),
        "jitter_seed": int(snapshot.config.competition_jitter_seed),
        "snapshot_sha256": _snapshot_hash(snapshot),
        "hidden_pulse_count": 0,
        "output_weight_count": 0,
        "decoder_read_count": 0,
        "reward_read_count": 0,
        "endpoint_read_count": 0,
    }
    return snapshot, source, hidden, receipt


def _strict_winner(
    hidden_activation: torch.Tensor,
    config: LocalStochasticBindingConfig,
) -> tuple[int | None, float]:
    packed = torch.as_tensor(hidden_activation, dtype=torch.float64).view(config.width)
    values, indices = torch.sort(packed, descending=True, stable=True)
    margin = float((values[0] - values[1]).item())
    if (
        not math.isfinite(margin)
        or float(values[0].item()) <= config.min_positive_activation
        or margin < config.min_winner_margin
    ):
        return None, margin
    return int(indices[0].item()), margin


def _residual(runtime: BrainRuntime) -> dict[str, float]:
    return {
        "positive_activation": float(runtime.activation.clamp_min(0.0).max().item()),
        "positive_axon_packet": (
            0.0
            if runtime._delay_buffer is None
            else float(runtime._delay_buffer.clamp_min(0.0).max().item())
        ),
        "usage_packet": (
            0.0
            if runtime._competition_usage_buffer is None
            else float(runtime._competition_usage_buffer.abs().max().item())
        ),
        "packet_envelope": (
            0.0
            if runtime.competition_packet_envelope is None
            else float(runtime.competition_packet_envelope.item())
        ),
    }


def _washout(runtime: BrainRuntime, config: LocalStochasticBindingConfig) -> dict[str, Any]:
    zero = torch.zeros(config.dim)
    for tick in range(1, config.max_washout_ticks + 1):
        runtime.step(external_input=zero, force_mode=RuntimeMode.WAKE, learning_signal=0.0)
        residual = _residual(runtime)
        if max(residual.values()) <= config.washout_tolerance:
            return {"washout_ticks": tick, "residual": residual, "passed": True}
    return {
        "washout_ticks": config.max_washout_ticks,
        "residual": _residual(runtime),
        "passed": False,
    }


def _jitter_receipt(runtime: BrainRuntime, hidden_count: int) -> dict[str, Any]:
    sigma = float(runtime.config.competition_jitter_sigma)
    generator = torch.Generator(device="cpu")
    seed = (
        int(runtime.config.competition_jitter_seed)
        + 104_729 * int(runtime.step_index)
    ) & 0x7FFF_FFFF_FFFF_FFFF
    generator.manual_seed(seed)
    normal = torch.randn(hidden_count, generator=generator)
    factor = 1.0 + sigma * torch.tanh(normal)
    return {
        "step_index": int(runtime.step_index),
        "derived_seed": int(seed),
        "normal": [float(value) for value in normal.tolist()],
        "factor": [float(value) for value in factor.tolist()],
        "normal_mean": float(normal.mean().item()),
        "normal_variance": float(normal.var(unbiased=False).item()),
    }


def _oja_update(
    runtime: BrainRuntime,
    delivered_pre: torch.Tensor,
    source: Sequence[int],
    hidden: Sequence[int],
    config: LocalStochasticBindingConfig,
    *,
    learning_rate: float,
) -> dict[str, Any]:
    source_index = torch.tensor(tuple(int(value) for value in source), dtype=torch.long)
    hidden_index = torch.tensor(tuple(int(value) for value in hidden), dtype=torch.long)
    pre = delivered_pre[source_index].clamp_min(0.0)
    post = runtime.activation[hidden_index].clamp_min(0.0)
    before = runtime.weight[hidden_index[:, None], source_index].detach().clone()
    raw_delta = float(learning_rate) * post[:, None] * (
        pre[None, :] - config.oja_beta * post[:, None] * before
    )
    after = (before + raw_delta).clamp(config.weight_min, config.weight_max)
    local_delta = after - before
    delta = torch.zeros_like(runtime.weight)
    delta[hidden_index[:, None], source_index] = local_delta
    installed_norm = 0.0
    if float(learning_rate) > 0.0 and float(delta.norm().item()) > 0.0:
        installed_norm = runtime.install_bounded_recurrent_delta(
            delta,
            max_frobenius_norm=config.max_update_norm,
        )
    candidate, _, _ = _candidate_support(config.dim)
    return {
        "delivered_pre": [float(value) for value in pre.tolist()],
        "post_activation": [float(value) for value in post.tolist()],
        "pre_positive_count": int(torch.count_nonzero(pre > 0.0).item()),
        "raw_delta_norm": float(raw_delta.norm().item()),
        "installed_delta_norm": float(installed_norm),
        "outside_candidate_delta_norm": float(delta[~candidate].norm().item()),
        "weight_min_after": float(runtime.weight[candidate].min().item()),
        "weight_max_after": float(runtime.weight[candidate].max().item()),
    }


def _episode(
    runtime: BrainRuntime,
    source_index: int,
    source: Sequence[int],
    hidden: Sequence[int],
    config: LocalStochasticBindingConfig,
    *,
    learning_rate: float,
    washout: bool,
) -> dict[str, Any]:
    hidden_index = torch.tensor(tuple(int(value) for value in hidden), dtype=torch.long)
    history: list[torch.Tensor] = []
    delivered = torch.zeros(config.dim)
    jitter: dict[str, Any] | None = None
    update: dict[str, Any] | None = None
    for relative_tick in range(config.axon_delay_ticks + 2):
        external = torch.zeros(config.dim)
        if relative_tick == 0:
            external[int(source_index)] = config.cue_drive_gain
        if runtime._delay_buffer is not None:
            slot = runtime._delay_idx % config.axon_delay_ticks
            delivered = runtime._delay_buffer[slot].detach().clone()
        if relative_tick == config.axon_delay_ticks + 1:
            jitter = _jitter_receipt(runtime, len(hidden))
        runtime.step(external_input=external, force_mode=RuntimeMode.WAKE, learning_signal=0.0)
        history.append(runtime.activation[hidden_index].detach().clone())
        if relative_tick == config.axon_delay_ticks + 1:
            update = _oja_update(
                runtime,
                delivered,
                source,
                hidden,
                config,
                learning_rate=learning_rate,
            )
    arrival = history[-1]
    winner, margin = _strict_winner(arrival, config)
    return {
        "source_index": int(source_index),
        "pulse_ticks": [0],
        "observed_ticks": list(range(config.axon_delay_ticks + 2)),
        "prearrival_positive_hidden_max": float(
            torch.stack(history[:-1]).clamp_min(0.0).max().item()
        ),
        "arrival_hidden": [float(value) for value in arrival.tolist()],
        "winner": winner,
        "winner_margin": margin,
        "jitter": jitter,
        "oja": update,
        "washout": _washout(runtime, config) if washout else None,
        "hidden_pulse_count": 0,
        "output_pulse_count": 0,
        "decoder_read_count": 0,
        "reward_read_count": 0,
        "endpoint_read_count": 0,
    }


def _allocation(
    episodes: Sequence[dict[str, Any]],
    source: Sequence[int],
) -> dict[str, Any]:
    source_to_slot = {int(value): slot for slot, value in enumerate(source)}
    winner_by_source = [-1] * len(source)
    margins: list[float] = []
    for episode in episodes:
        winner = episode["winner"]
        if winner is None:
            continue
        winner_by_source[source_to_slot[int(episode["source_index"])]] = int(winner)
        margins.append(float(episode["winner_margin"]))
    selected = [value for value in winner_by_source if value >= 0]
    return {
        "winner_by_source": winner_by_source,
        "winner_margins": margins,
        "abstention_count": len(source) - len(selected),
        "collision_fraction": 1.0 - len(set(selected)) / len(source),
        "is_bijection": bool(len(selected) == len(source) and len(set(selected)) == len(source)),
    }


def _noise_off_snapshot(snapshot: Any) -> Any:
    copied = deepcopy(snapshot)
    copied.config.competition_jitter_sigma = 0.0
    copied.config.validate_local_competition()
    return copied


def _train(
    snapshot: Any,
    source: Sequence[int],
    hidden: Sequence[int],
    order: Sequence[int],
    config: LocalStochasticBindingConfig,
    *,
    learning_rate: float,
) -> dict[str, Any]:
    runtime = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    episodes = [
        _episode(
            runtime,
            int(source[int(slot)]),
            source,
            hidden,
            config,
            learning_rate=learning_rate,
            washout=True,
        )
        for slot in order
    ]
    trained = runtime.snapshot()
    candidate, _, _ = _candidate_support(config.dim)
    values = trained.weight[candidate].view(config.width, config.width)
    columns = values / values.norm(dim=0, keepdim=True).clamp_min(config.competition_epsilon)
    pairwise = [
        float((columns[:, left] - columns[:, right]).norm().item())
        for left in range(config.width)
        for right in range(left + 1, config.width)
    ]
    return {
        "episodes": episodes,
        "trained_snapshot": trained,
        "trained_snapshot_sha256": _snapshot_hash(trained),
        "candidate_weights": [[float(value) for value in row] for row in values.tolist()],
        "candidate_weight_min": float(values.min().item()),
        "candidate_weight_max": float(values.max().item()),
        "candidate_weight_delta_norm": float((values - 1.0).norm().item()),
        "minimum_normalized_column_distance": min(pairwise),
        "all_updates_local": all(
            episode["oja"] is not None
            and episode["oja"]["outside_candidate_delta_norm"] == 0.0
            for episode in episodes
        ),
    }


def _evaluate_fresh(
    snapshot: Any,
    source: Sequence[int],
    hidden: Sequence[int],
    config: LocalStochasticBindingConfig,
    order: Sequence[int],
) -> dict[str, Any]:
    noise_off = _noise_off_snapshot(snapshot)
    episodes: list[dict[str, Any]] = []
    before_hash = _snapshot_hash(noise_off)
    for slot in order:
        runtime = BrainRuntime.from_snapshot(noise_off, backend="torch", device="cpu")
        runtime.reset_evaluation_state()
        episodes.append(
            _episode(
                runtime,
                int(source[int(slot)]),
                source,
                hidden,
                config,
                learning_rate=0.0,
                washout=False,
            )
        )
    after_hash = _snapshot_hash(noise_off)
    return {
        "order": [int(value) for value in order],
        "episodes": episodes,
        "allocation": _allocation(episodes, source),
        "jitter_sigma": float(noise_off.config.competition_jitter_sigma),
        "source_snapshot_immutable": before_hash == after_hash,
    }


def _source_independent_bias_snapshot(
    config: LocalStochasticBindingConfig,
) -> tuple[Any, tuple[int, ...], tuple[int, ...]]:
    candidate, source, hidden = _candidate_support(config.dim)
    row_bias = torch.tensor((0.85, 0.95, 1.05, 1.15))
    weight = torch.zeros(config.dim, config.dim)
    weight[torch.tensor(hidden)[:, None], torch.tensor(source)] = row_bias[:, None]
    runtime = BrainRuntime(
        weight,
        config=_runtime_config(config, hidden, jitter_sigma=0.0),
        backend="torch",
        device="cpu",
    )
    snapshot = runtime.snapshot()
    assert torch.count_nonzero(snapshot.weight[~candidate]) == 0
    return snapshot, source, hidden


def run_local_stochastic_binding_seed(
    seed: int = CALIBRATION_SEED,
    *,
    config: LocalStochasticBindingConfig | None = None,
) -> dict[str, Any]:
    selected = config or LocalStochasticBindingConfig(seed=int(seed))
    config = LocalStochasticBindingConfig(**{**asdict(selected), "seed": int(seed)})
    order_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 123_700_021)
    order = tuple(int(value) for value in torch.randperm(config.width, generator=order_generator))
    evaluation_order = tuple(reversed(order))

    source_snapshot, source, hidden, source_receipt = _uniform_source_snapshot(
        config,
        jitter_sigma=config.jitter_sigma,
    )
    learned = _train(
        source_snapshot,
        source,
        hidden,
        order,
        config,
        learning_rate=config.oja_lr,
    )
    learned_eval = _evaluate_fresh(
        learned["trained_snapshot"], source, hidden, config, evaluation_order,
    )

    deterministic_snapshot, _, _, deterministic_receipt = _uniform_source_snapshot(
        config,
        jitter_sigma=0.0,
    )
    deterministic = _train(
        deterministic_snapshot,
        source,
        hidden,
        order,
        config,
        learning_rate=config.oja_lr,
    )
    deterministic_eval = _evaluate_fresh(
        deterministic["trained_snapshot"], source, hidden, config, evaluation_order,
    )

    no_learning = _train(
        source_snapshot,
        source,
        hidden,
        order,
        config,
        learning_rate=0.0,
    )
    no_learning_eval = _evaluate_fresh(
        no_learning["trained_snapshot"], source, hidden, config, evaluation_order,
    )

    no_homeostasis_snapshot, _, _, no_homeostasis_receipt = _uniform_source_snapshot(
        config,
        jitter_sigma=config.jitter_sigma,
        homeostasis_gain=0.0,
    )
    no_homeostasis = _train(
        no_homeostasis_snapshot,
        source,
        hidden,
        order,
        config,
        learning_rate=config.oja_lr,
    )
    no_homeostasis_eval = _evaluate_fresh(
        no_homeostasis["trained_snapshot"], source, hidden, config, evaluation_order,
    )

    bias_snapshot, _, _ = _source_independent_bias_snapshot(config)
    bias_eval = _evaluate_fresh(
        bias_snapshot, source, hidden, config, evaluation_order,
    )

    all_training_episodes = [
        *learned["episodes"],
        *deterministic["episodes"],
        *no_learning["episodes"],
        *no_homeostasis["episodes"],
    ]
    all_evaluation_episodes = [
        *learned_eval["episodes"],
        *deterministic_eval["episodes"],
        *no_learning_eval["episodes"],
        *no_homeostasis_eval["episodes"],
        *bias_eval["episodes"],
    ]
    no_forbidden_reads = all(
        episode["hidden_pulse_count"] == 0
        and episode["output_pulse_count"] == 0
        and episode["decoder_read_count"] == 0
        and episode["reward_read_count"] == 0
        and episode["endpoint_read_count"] == 0
        for episode in [*all_training_episodes, *all_evaluation_episodes]
    )
    learned_allocation = learned_eval["allocation"]
    deterministic_allocation = deterministic_eval["allocation"]
    no_learning_allocation = no_learning_eval["allocation"]
    bias_columns = torch.as_tensor(bias_snapshot.weight)[torch.tensor(hidden)[:, None], torch.tensor(source)]
    gates = {
        "bounded_jitter_homeostasis_inequality": bool(
            config.homeostasis_gain
            > math.log((1.0 + config.jitter_sigma) / (1.0 - config.jitter_sigma))
        ),
        "uniform_initial_weights": bool(
            source_receipt["candidate_all_exactly_one"]
            and source_receipt["candidate_unique_values"] == 1
            and source_receipt["outside_candidate_nonzero"] == 0
        ),
        "packet_local_true_delay": bool(
            max(episode["prearrival_positive_hidden_max"] for episode in all_training_episodes)
            <= config.washout_tolerance
            and all(
                episode["oja"] is not None
                and episode["oja"]["pre_positive_count"] == 1
                for episode in all_training_episodes
            )
        ),
        "all_training_washouts_pass": all(
            bool(episode["washout"]["passed"]) for episode in all_training_episodes
        ),
        "local_oja_only": bool(
            learned["all_updates_local"]
            and learned["candidate_weight_delta_norm"] > 0.0
        ),
        "noise_off_fresh_evaluation_bijection": bool(
            learned_eval["jitter_sigma"] == 0.0
            and learned_allocation["is_bijection"]
            and learned_allocation["abstention_count"] == 0
        ),
        "durable_distinct_weight_columns": bool(
            learned["minimum_normalized_column_distance"] >= config.min_column_distance
        ),
        "deterministic_symmetry_no_go": bool(
            deterministic["candidate_weight_delta_norm"] == 0.0
            and deterministic_allocation["abstention_count"] == config.width
        ),
        "learning_ablation_no_code": bool(
            no_learning["candidate_weight_delta_norm"] == 0.0
            and no_learning_allocation["abstention_count"] == config.width
        ),
        "source_independent_bias_rejected": bool(
            torch.equal(bias_columns[:, :1].expand_as(bias_columns), bias_columns)
            and not bias_eval["allocation"]["is_bijection"]
        ),
        "evaluation_order_differs_from_training": list(evaluation_order) != list(order),
        "snapshots_immutable_during_evaluation": all(
            item["source_snapshot_immutable"]
            for item in (
                learned_eval,
                deterministic_eval,
                no_learning_eval,
                no_homeostasis_eval,
                bias_eval,
            )
        ),
        "no_hidden_output_decoder_reward_endpoint_reads": no_forbidden_reads,
    }
    status = "LOCAL_STOCHASTIC_WEIGHT_CODE_PASS" if all(gates.values()) else "MECHANISM_FAIL"
    return {
        "seed": int(seed),
        "status": status,
        "endpoint_opened": False,
        "output_identity_status": "NONIDENTIFIED_ENDPOINT_CLOSED",
        "claim_scope": "stochastic source-column code in a declared synthetic support",
        "config": asdict(config),
        "training_order": list(order),
        "evaluation_order": list(evaluation_order),
        "source_receipt": source_receipt,
        "deterministic_receipt": deterministic_receipt,
        "no_homeostasis_receipt": no_homeostasis_receipt,
        "gates": gates,
        "learned": {
            key: value for key, value in learned.items() if key != "trained_snapshot"
        },
        "learned_evaluation": learned_eval,
        "deterministic_no_jitter": {
            "training": {key: value for key, value in deterministic.items() if key != "trained_snapshot"},
            "evaluation": deterministic_eval,
        },
        "no_learning": {
            "training": {key: value for key, value in no_learning.items() if key != "trained_snapshot"},
            "evaluation": no_learning_eval,
        },
        "no_homeostasis": {
            "training": {key: value for key, value in no_homeostasis.items() if key != "trained_snapshot"},
            "evaluation": no_homeostasis_eval,
        },
        "source_independent_bias": {
            "status": "SOURCE_UNIDENTIFIED",
            "evaluation": bias_eval,
        },
        "confirmation_opened": False,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
    }
