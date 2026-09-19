"""Outcome-blind seeded source allocation after the BA-TR7 symmetry no-go.

The BrainRuntime part of this probe supplies only a delayed source-to-hidden
eligibility observation.  A separate hard WTA/capacity state then allocates
one hidden coordinate per physical source coordinate.  No output codebook,
decoder, reward, or task endpoint is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Sequence

import torch

from ..runtime import BrainRuntime, RuntimeMode
from .runtime_context_branch_routing import (
    EPSILON,
    ApparatusInvalid,
    ContextBranchConfig,
    ExactDelayEligibility,
    _runtime,
    _snapshot_hash,
    architectural_blocks,
)


@dataclass(frozen=True)
class SeededCompetitionConfig:
    seed: int = 98001
    heterogeneity_epsilon: float = 0.20
    capacity_penalty: float = 1.10
    min_winner_margin: float = 1e-6
    equality_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        finite = (
            self.heterogeneity_epsilon,
            self.capacity_penalty,
            self.min_winner_margin,
            self.equality_tolerance,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise ValueError("configuration values must be finite")
        max_level = 3.0 / math.sqrt(20.0)
        if not 0.0 < self.heterogeneity_epsilon < 1.0 / max_level:
            raise ValueError("heterogeneity must keep every candidate weight positive")
        if self.capacity_penalty <= 1.0:
            raise ValueError("capacity penalty must exceed normalized local evidence")
        if self.min_winner_margin <= 0.0 or self.equality_tolerance < 0.0:
            raise ValueError("invalid winner/equality tolerance")


def seeded_edge_code(heterogeneity_seed: int, width: int = 4) -> torch.Tensor:
    """Return a seed-only, column-balanced edge code.

    Every column is an independent permutation of four fixed levels.  The
    function intentionally has no task, label, output, decoder, or schedule
    input.
    """
    if int(width) != 4:
        raise ValueError("the frozen fixture requires width four")
    levels = torch.tensor((-3.0, -1.0, 1.0, 3.0), dtype=torch.float64) / math.sqrt(20.0)
    generator = torch.Generator(device="cpu").manual_seed(int(heterogeneity_seed))
    columns = [levels[torch.randperm(width, generator=generator)] for _ in range(width)]
    return torch.stack(columns, dim=1)


def _source_independent_bias(heterogeneity_seed: int, width: int = 4) -> torch.Tensor:
    levels = seeded_edge_code(int(heterogeneity_seed), int(width))[:, 0]
    return levels.view(width, 1).expand(width, width).clone()


def _tensor_hash(tensor: torch.Tensor) -> str:
    packed = torch.as_tensor(tensor).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(packed.shape)).encode())
    digest.update(str(packed.dtype).encode())
    digest.update(packed.numpy().tobytes())
    return digest.hexdigest()


def _candidate_support(
    blocks: Sequence[Sequence[int]],
    dim: int,
) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    source = tuple(int(value) for value in blocks[0])
    hidden = tuple(int(value) for value in blocks[2])
    if len(source) != 4 or len(hidden) != 4:
        raise ApparatusInvalid("APPARATUS_INVALID: expected width-four source and hidden blocks")
    mask = torch.zeros(dim, dim, dtype=torch.bool)
    mask[torch.tensor(hidden)[:, None], torch.tensor(source)] = True
    return mask, source, hidden


def _move_old_rows_to_new(code: torch.Tensor, old_to_new: torch.Tensor) -> torch.Tensor:
    packed = torch.as_tensor(old_to_new, dtype=torch.long).view(-1)
    if packed.shape != (4,) or sorted(int(value) for value in packed.tolist()) != list(range(4)):
        raise ValueError("row mapping must be a permutation of four coordinates")
    moved = torch.empty_like(code)
    moved[packed, :] = code
    return moved


def _build_source_snapshot(
    config: SeededCompetitionConfig,
    code: torch.Tensor,
) -> tuple[Any, ContextBranchConfig, torch.Tensor, tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    base = ContextBranchConfig(seed=int(config.seed) + 5_000_003)
    runtime = _runtime(base)
    blocks = architectural_blocks(base.dim)
    candidate, source, hidden = _candidate_support(blocks, base.dim)
    packed = torch.as_tensor(code, dtype=torch.float64)
    if packed.shape != (4, 4) or not torch.isfinite(packed).all():
        raise ApparatusInvalid("APPARATUS_INVALID: invalid seeded edge code")
    weights = 1.0 + float(config.heterogeneity_epsilon) * packed
    if torch.any(weights <= 0.0):
        raise ApparatusInvalid("APPARATUS_INVALID: nonpositive seeded candidate")
    matrix = torch.zeros_like(runtime.weight)
    matrix[torch.tensor(hidden)[:, None], torch.tensor(source)] = weights.to(matrix)
    runtime.weight = matrix
    runtime._rebuild_sparse()
    runtime.reset_evaluation_state()
    snapshot = runtime.snapshot()
    candidate_values = snapshot.weight[candidate].double()
    receipt = {
        "candidate_edges": int(candidate.sum().item()),
        "outside_nonzero": int(torch.count_nonzero(snapshot.weight[~candidate]).item()),
        "candidate_min": float(candidate_values.min().item()),
        "candidate_max": float(candidate_values.max().item()),
        "code_column_sums": [float(value) for value in packed.sum(dim=0).tolist()],
        "code_column_norms": [float(value) for value in packed.norm(dim=0).tolist()],
        "code_unique_per_column": [int(torch.unique(packed[:, column]).numel()) for column in range(4)],
        "code_sha256": _tensor_hash(packed),
        "source_snapshot_sha256": _snapshot_hash(snapshot),
        "delay_ring_zero": bool(
            runtime._delay_buffer is not None
            and torch.count_nonzero(runtime._delay_buffer) == 0
        ),
        "hippocampal_rows_after": len(runtime.hippocampus),
        "output_weight_count": 0,
    }
    return snapshot, base, candidate, source, hidden, receipt


def _source_evidence(
    snapshot: Any,
    base: ContextBranchConfig,
    source_index: int,
    hidden: Sequence[int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    runtime = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    runtime.reset_evaluation_state()
    tracker = ExactDelayEligibility(base.dim, base.delay_ticks, base.eligibility_decay, base.ltd)
    hidden_index = torch.tensor(tuple(int(value) for value in hidden), dtype=torch.long)
    hidden_history: list[torch.Tensor] = []
    for tick in range(base.delay_ticks + 2):
        external = torch.zeros(base.dim)
        if tick == 0:
            external[int(source_index)] = float(base.cue_drive_gain)
        runtime.step(external_input=external, force_mode=RuntimeMode.WAKE, learning_signal=0.0)
        tracker.observe(runtime.activation)
        hidden_history.append(runtime.activation[hidden_index].detach().clone())
    local = tracker.eligibility.double().clamp_min(0.0)[hidden_index, int(source_index)]
    total = float(local.sum().item())
    if not math.isfinite(total) or total <= EPSILON:
        raise ApparatusInvalid("APPARATUS_INVALID: no positive source-only eligibility")
    evidence = local / (EPSILON + total)
    prearrival = torch.stack(hidden_history[:-1])
    return evidence, {
        "source_index": int(source_index),
        "pulse_ticks": [0],
        "observed_ticks": list(range(base.delay_ticks + 2)),
        "prearrival_hidden_max_abs": float(prearrival.abs().max().item()),
        "arrival_hidden": [float(value) for value in hidden_history[-1].tolist()],
        "arrival_hidden_min": float(hidden_history[-1].min().item()),
        "evidence": [float(value) for value in evidence.tolist()],
        "evidence_sum": float(evidence.sum().item()),
        "hidden_pulse_count": 0,
        "output_pulse_count": 0,
        "decoder_read_count": 0,
        "reward_read_count": 0,
        "endpoint_read_count": 0,
    }


def _collect_evidence(
    snapshot: Any,
    base: ContextBranchConfig,
    source: Sequence[int],
    hidden: Sequence[int],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    columns: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for source_index in source:
        evidence, receipt = _source_evidence(snapshot, base, int(source_index), hidden)
        columns.append(evidence)
        rows.append(receipt)
    return torch.stack(columns, dim=1), rows


def allocate_source_bindings(
    evidence: torch.Tensor,
    source_order: Sequence[int],
    capacity_penalty: float,
    min_winner_margin: float,
    *,
    use_capacity: bool,
) -> dict[str, Any]:
    """Allocate source columns using only local evidence and hidden occupancy."""
    packed = torch.as_tensor(evidence, dtype=torch.float64)
    order = tuple(int(value) for value in source_order)
    if packed.shape != (4, 4) or not torch.isfinite(packed).all() or torch.any(packed < 0.0):
        raise ApparatusInvalid("APPARATUS_INVALID: invalid local evidence matrix")
    if sorted(order) != list(range(4)):
        raise ApparatusInvalid("APPARATUS_INVALID: source order must be a permutation")
    penalty = float(capacity_penalty)
    margin_floor = float(min_winner_margin)
    if not math.isfinite(penalty) or not math.isfinite(margin_floor) or margin_floor <= 0.0:
        raise ApparatusInvalid("APPARATUS_INVALID: invalid allocation constants")
    occupancy = torch.zeros(4, dtype=torch.float64)
    binding = torch.zeros(4, 4, dtype=torch.bool)
    rows: list[dict[str, Any]] = []
    winners: list[int] = []
    for source_slot in order:
        scores = packed[:, source_slot] - (penalty * occupancy if use_capacity else 0.0)
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, stable=True)
        margin = float((sorted_scores[0] - sorted_scores[1]).item())
        if not math.isfinite(margin) or margin < margin_floor:
            return {
                "status": "ABSTAIN_BOUNDARY_TIE",
                "abstained_source_slot": int(source_slot),
                "source_order": list(order),
                "use_capacity": bool(use_capacity),
                "binding": binding,
                "winners_by_processing_order": winners,
                "winner_by_source": [-1] * 4,
                "winner_margins": [float(row["winner_margin"]) for row in rows],
                "collision_fraction": 1.0,
                "is_bijection": False,
                "rows": rows,
            }
        winner = int(sorted_indices[0].item())
        binding[winner, source_slot] = True
        winners.append(winner)
        rows.append({
            "source_slot": int(source_slot),
            "winner": winner,
            "winner_margin": margin,
            "occupancy_before": [float(value) for value in occupancy.tolist()],
            "scores": [float(value) for value in scores.tolist()],
        })
        if use_capacity:
            occupancy[winner] = 1.0
    winner_by_source = [-1] * 4
    for row in rows:
        winner_by_source[int(row["source_slot"])] = int(row["winner"])
    unique = len(set(winners))
    return {
        "status": "ALLOCATED",
        "abstained_source_slot": None,
        "source_order": list(order),
        "use_capacity": bool(use_capacity),
        "binding": binding,
        "winners_by_processing_order": winners,
        "winner_by_source": winner_by_source,
        "winner_margins": [float(row["winner_margin"]) for row in rows],
        "collision_fraction": 1.0 - unique / 4.0,
        "is_bijection": bool(unique == 4 and int(binding.sum().item()) == 4),
        "rows": rows,
    }


def _serializable_allocation(result: dict[str, Any]) -> dict[str, Any]:
    packed = dict(result)
    binding = torch.as_tensor(packed.pop("binding"), dtype=torch.bool)
    packed["binding_edges"] = [
        [int(row), int(column)] for row, column in zip(*torch.where(binding))
    ]
    packed["binding_sha256"] = _tensor_hash(binding)
    return packed


def run_seeded_source_competition_seed(
    seed: int = 98001,
    *,
    config: SeededCompetitionConfig | None = None,
) -> dict[str, Any]:
    selected = config or SeededCompetitionConfig(seed=int(seed))
    config = SeededCompetitionConfig(**{**asdict(selected), "seed": int(seed)})
    heterogeneity_seed = int(seed) + 71_900_003
    order_seed = int(seed) + 81_700_019
    code = seeded_edge_code(heterogeneity_seed)
    uniform = torch.zeros_like(code)
    bias = _source_independent_bias(heterogeneity_seed)
    order_generator = torch.Generator(device="cpu").manual_seed(order_seed)
    source_order = tuple(int(value) for value in torch.randperm(4, generator=order_generator).tolist())
    alternate_order = tuple(reversed(source_order))
    shift = 1 + (int(seed) % 3)
    old_to_new = torch.remainder(torch.arange(4) + shift, 4)
    permuted_code = _move_old_rows_to_new(code, old_to_new)

    built: dict[str, Any] = {}
    for name, candidate_code in (
        ("seeded", code),
        ("uniform", uniform),
        ("bias", bias),
        ("row_permuted", permuted_code),
    ):
        snapshot, base, candidate, source, hidden, receipt = _build_source_snapshot(
            config, candidate_code,
        )
        before = _snapshot_hash(snapshot)
        evidence, episodes = _collect_evidence(snapshot, base, source, hidden)
        after = _snapshot_hash(snapshot)
        built[name] = {
            "snapshot": snapshot,
            "base": base,
            "candidate": candidate,
            "source": source,
            "hidden": hidden,
            "receipt": receipt,
            "evidence": evidence,
            "episodes": episodes,
            "snapshot_immutable": before == after,
        }

    seeded_raw = allocate_source_bindings(
        built["seeded"]["evidence"], source_order,
        config.capacity_penalty, config.min_winner_margin, use_capacity=False,
    )
    seeded_capacity = allocate_source_bindings(
        built["seeded"]["evidence"], source_order,
        config.capacity_penalty, config.min_winner_margin, use_capacity=True,
    )
    alternate_capacity = allocate_source_bindings(
        built["seeded"]["evidence"], alternate_order,
        config.capacity_penalty, config.min_winner_margin, use_capacity=True,
    )
    uniform_raw = allocate_source_bindings(
        built["uniform"]["evidence"], source_order,
        config.capacity_penalty, config.min_winner_margin, use_capacity=False,
    )
    uniform_capacity = allocate_source_bindings(
        built["uniform"]["evidence"], source_order,
        config.capacity_penalty, config.min_winner_margin, use_capacity=True,
    )
    bias_raw = allocate_source_bindings(
        built["bias"]["evidence"], source_order,
        config.capacity_penalty, config.min_winner_margin, use_capacity=False,
    )
    row_permuted = allocate_source_bindings(
        built["row_permuted"]["evidence"], source_order,
        config.capacity_penalty, config.min_winner_margin, use_capacity=True,
    )

    expected_permuted = [int(old_to_new[value].item()) for value in seeded_capacity["winner_by_source"]]
    all_episodes = [episode for value in built.values() for episode in value["episodes"]]
    seeded_receipt = built["seeded"]["receipt"]
    gates = {
        "seed_code_pure_and_reproducible": bool(
            torch.equal(code, seeded_edge_code(heterogeneity_seed))
            and code.shape == (4, 4)
        ),
        "balanced_distinct_edge_code": bool(
            max(abs(value) for value in seeded_receipt["code_column_sums"])
            <= config.equality_tolerance
            and all(abs(value - 1.0) <= config.equality_tolerance
                    for value in seeded_receipt["code_column_norms"])
            and seeded_receipt["code_unique_per_column"] == [4, 4, 4, 4]
        ),
        "only_source_hidden_candidates_exist": all(
            value["receipt"]["candidate_edges"] == 16
            and value["receipt"]["outside_nonzero"] == 0
            and value["receipt"]["output_weight_count"] == 0
            for value in built.values()
        ),
        "all_candidate_weights_positive": all(
            value["receipt"]["candidate_min"] > 0.0 for value in built.values()
        ),
        "true_delay_arrival": bool(
            max(episode["prearrival_hidden_max_abs"] for episode in all_episodes)
            <= config.equality_tolerance
            and min(episode["arrival_hidden_min"] for episode in all_episodes) > 0.0
            and all(episode["observed_ticks"] == [0, 1, 2, 3] for episode in all_episodes)
        ),
        "seeded_source_local_margins": bool(
            seeded_raw["status"] == "ALLOCATED"
            and min(seeded_raw["winner_margins"]) >= config.min_winner_margin
        ),
        "uniform_no_capacity_abstains": uniform_raw["status"] == "ABSTAIN_BOUNDARY_TIE",
        "competition_only_uniform_abstains": (
            uniform_capacity["status"] == "ABSTAIN_BOUNDARY_TIE"
        ),
        "source_independent_bias_collapses": bool(
            bias_raw["status"] == "ALLOCATED"
            and bias_raw["collision_fraction"] == 0.75
        ),
        "seeded_capacity_bijection": bool(
            seeded_capacity["status"] == "ALLOCATED"
            and seeded_capacity["is_bijection"]
            and min(seeded_capacity["winner_margins"]) >= config.min_winner_margin
        ),
        "alternate_order_remains_bijection": bool(
            alternate_capacity["status"] == "ALLOCATED"
            and alternate_capacity["is_bijection"]
        ),
        "hidden_row_permutation_covariant": bool(
            row_permuted["status"] == "ALLOCATED"
            and row_permuted["winner_by_source"] == expected_permuted
        ),
        "source_snapshots_immutable": all(
            bool(value["snapshot_immutable"]) for value in built.values()
        ),
        "no_hidden_output_decoder_reward_endpoint_reads": all(
            episode["hidden_pulse_count"] == 0
            and episode["output_pulse_count"] == 0
            and episode["decoder_read_count"] == 0
            and episode["reward_read_count"] == 0
            and episode["endpoint_read_count"] == 0
            for episode in all_episodes
        ),
    }
    passed = all(gates.values())
    controls = {
        "seeded_no_capacity": _serializable_allocation(seeded_raw),
        "seeded_capacity": _serializable_allocation(seeded_capacity),
        "seeded_capacity_alternate_order": _serializable_allocation(alternate_capacity),
        "uniform_no_capacity": _serializable_allocation(uniform_raw),
        "uniform_capacity": _serializable_allocation(uniform_capacity),
        "source_independent_bias_no_capacity": _serializable_allocation(bias_raw),
        "hidden_row_permuted_capacity": _serializable_allocation(row_permuted),
    }
    return {
        "seed": int(seed),
        "status": "SEEDED_SOURCE_ALLOCATION_PASS" if passed else "APPARATUS_INVALID",
        "endpoint_opened": False,
        "output_identity_status": "NONIDENTIFIED_ENDPOINT_CLOSED",
        "config": asdict(config),
        "heterogeneity_seed": heterogeneity_seed,
        "source_order_seed": order_seed,
        "source_order": list(source_order),
        "alternate_source_order": list(alternate_order),
        "hidden_old_to_new_permutation": [int(value) for value in old_to_new.tolist()],
        "seeded_code_sha256": _tensor_hash(code),
        "gates": gates,
        "raw_collision_fraction": float(seeded_raw["collision_fraction"]),
        "capacity_collision_fraction": float(seeded_capacity["collision_fraction"]),
        "order_changed_binding": bool(
            seeded_capacity["winner_by_source"] != alternate_capacity["winner_by_source"]
        ),
        "controls": controls,
        "seeded_source_receipt": seeded_receipt,
        "representative_episodes": built["seeded"]["episodes"],
    }
