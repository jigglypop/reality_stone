"""Cue-conditioned sparse recurrent masks for the frozen M1 apparatus.

This is deliberately a small synthetic mechanism module.  Mask construction
is a pure function of a learned matrix, the present cue, the declared blocks,
and a public seed; it has no access to codebook outputs or rollout results.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import math
from typing import Any, Sequence

import torch

from ..runtime import BrainRuntime, BrainRuntimeConfig, HippocampusMemory, RuntimeMode
from .runtime_alternative_memory import AlternativeMemoryConfig, DelayedSignedEligibility, _m1_apply_block
from .runtime_contrastive_predictive_memory import _factor_codebooks
from .runtime_native_loops import _codebook, _decode, _loop8_replay_source_audit, _unit
from ..temporal_memory import TemporalAuditedMemory, TemporalMemoryEvent


EPSILON = 1e-8
ROUTES = frozenset({
    "FULL", "WEIGHT", "CLUSTER", "PATH_ONLY", "TOPOLOGY",
    "RETURN_SHUFFLED", "RANDOM_MATCHED", "WRONG_CONTEXT",
})


class ApparatusInvalid(ValueError):
    """Raised when the frozen routing apparatus cannot form an admissible mask."""


def _edge_order(scores: torch.Tensor, allowed: torch.Tensor, count: int) -> torch.Tensor:
    rows, cols = torch.where(allowed)
    if int(rows.numel()) < count:
        raise ApparatusInvalid("APPARATUS_INVALID: admissible support is below budget")
    triples = [(-float(scores[row, col]), int(row), int(col)) for row, col in zip(rows.tolist(), cols.tolist())]
    triples.sort()
    chosen = triples[:count]
    out = torch.zeros_like(allowed, dtype=torch.bool)
    for _, row, col in chosen:
        out[row, col] = True
    return out


def _validate_blocks(blocks: Sequence[Sequence[int]], dim: int) -> tuple[tuple[int, ...], ...]:
    packed = tuple(tuple(int(i) for i in block) for block in blocks)
    if not packed or any(not block for block in packed):
        raise ApparatusInvalid("APPARATUS_INVALID: empty architectural block")
    flattened = [i for block in packed for i in block]
    if sorted(flattened) != list(range(dim)):
        raise ApparatusInvalid("APPARATUS_INVALID: blocks must partition coordinates")
    return packed


def _cluster_admissible(
    support: torch.Tensor,
    learned: torch.Tensor,
    magnitude: torch.Tensor,
    partition: tuple[tuple[int, ...], ...],
) -> torch.Tensor:
    if float(magnitude.max().item()) == 0.0:
        raise ApparatusInvalid("APPARATUS_INVALID: zero cue")
    active = [
        block_index
        for block_index, block in enumerate(partition)
        if float(magnitude[list(block)].sum().item()) > EPSILON
    ]
    if not active:
        raise ApparatusInvalid("APPARATUS_INVALID: no cue-active source block")
    admissible = torch.zeros_like(learned)
    for source in active:
        source_ix = list(partition[source])
        candidates: list[tuple[float, int]] = []
        for destination, block in enumerate(partition):
            if destination not in active:
                candidates.append(
                    (-float(support[list(block)][:, source_ix].sum().item()), destination)
                )
        if not candidates:
            raise ApparatusInvalid("APPARATUS_INVALID: no non-source downstream block")
        _, destination = min(candidates)
        destination_ix = list(partition[destination])
        dst = torch.tensor(destination_ix, dtype=torch.long)
        src = torch.tensor(source_ix, dtype=torch.long)
        admissible[dst[:, None], src] = learned[dst[:, None], src]
        admissible[src[:, None], src] |= learned[src[:, None], src]
        admissible[dst[:, None], dst] |= learned[dst[:, None], dst]
    return admissible


def construct_route_mask(weight, cue, blocks, seed, route, budget):
    """Construct a target-free exact-budget mask.

    The six-argument signature is an intentional audit boundary.  In
    particular, this function receives neither a target codebook nor a
    decoder, endpoint, or rollout state.
    """
    if route not in ROUTES:
        raise ValueError(f"unknown route {route!r}")
    matrix = torch.as_tensor(weight, dtype=torch.float64)
    signal = torch.as_tensor(cue, dtype=torch.float64).flatten()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != signal.numel():
        raise ApparatusInvalid("APPARATUS_INVALID: incompatible matrix/cue")
    if not torch.isfinite(matrix).all() or not torch.isfinite(signal).all():
        raise ApparatusInvalid("APPARATUS_INVALID: nonfinite input")
    partition = _validate_blocks(blocks, int(signal.numel()))
    support = matrix.abs()
    support.fill_diagonal_(0.0)
    learned = support > 0.0
    total = int(learned.sum().item())
    if total == 0 or int(budget) <= 0:
        raise ApparatusInvalid("APPARATUS_INVALID: empty learned support or budget")
    budget = int(budget)
    if route == "FULL":
        return learned.to(weight.dtype if isinstance(weight, torch.Tensor) else torch.float32)
    if total < budget:
        raise ApparatusInvalid("APPARATUS_INVALID: learned support is below budget")
    if route == "RANDOM_MATCHED":
        values = torch.randperm(total, generator=torch.Generator(device="cpu").manual_seed(int(seed)))
        rows, cols = torch.where(learned)
        result = torch.zeros_like(learned)
        result[rows[values[:budget]], cols[values[:budget]]] = True
        return result.to(weight.dtype if isinstance(weight, torch.Tensor) else torch.float32)
    if route == "WEIGHT":
        return _edge_order(support, learned, budget).to(weight.dtype if isinstance(weight, torch.Tensor) else torch.float32)

    magnitude = signal.abs()
    admissible = _cluster_admissible(support, learned, magnitude, partition)
    if route == "CLUSTER":
        return _edge_order(support, admissible, budget).to(weight.dtype if isinstance(weight, torch.Tensor) else torch.float32)
    forward = magnitude + support @ magnitude + support @ (support @ magnitude)
    if route == "PATH_ONLY":
        score = support * (EPSILON + forward.unsqueeze(0))
    else:
        returned = support.T + (support @ support).T
        if route == "RETURN_SHUFFLED":
            rows, cols = torch.where(admissible)
            shuffled = returned[rows, cols][torch.randperm(int(rows.numel()), generator=torch.Generator(device="cpu").manual_seed(int(seed)))]
            copy = torch.zeros_like(returned); copy[rows, cols] = shuffled; returned = copy
        score = support * (EPSILON + forward.unsqueeze(0)) * (1.0 + returned / (returned.max() + EPSILON))
    return _edge_order(score, admissible, budget).to(weight.dtype if isinstance(weight, torch.Tensor) else torch.float32)


@dataclass(frozen=True)
class TopologyRoutingConfig:
    dim: int = 48
    replay_epochs: int = 12
    replay_ticks: int = 3
    rollout_horizon: int = 6
    cue_corruption: float = 0.15
    cue_drive_gain: float = 5.0
    m1_lr: float = 0.8
    max_write_norm: float = 5.0
    seed: int = 97301

    def alternative(self) -> AlternativeMemoryConfig:
        d = self.dim
        return AlternativeMemoryConfig(
            dim=d, replay_epochs=self.replay_epochs, replay_ticks=self.replay_ticks,
            rollout_horizon=self.rollout_horizon, cue_corruption=self.cue_corruption,
            cue_drive_gain=self.cue_drive_gain,
            m1_lr=self.m1_lr, max_write_norm=self.max_write_norm, seed=self.seed,
            neuronwise_active_threshold=tuple(0.18 + 0.08 * i / (d - 1) for i in range(d)),
            neuronwise_bit_lower_threshold=tuple(0.06 + 0.08 * i / (d - 1) for i in range(d)),
            neuronwise_bit_upper_threshold=tuple(0.24 + 0.12 * i / (d - 1) for i in range(d)),
        )


def _runtime(config: AlternativeMemoryConfig) -> BrainRuntime:
    return BrainRuntime(torch.zeros(config.dim, config.dim), config=BrainRuntimeConfig(
        dim=config.dim, active_ratio=0.25, noise_sigma=0.0, dale_law=False,
        axon_delay=True, max_axon_delay=2, f1_self_measure=False, stdp_enabled=False,
        memory_capacity=16, replay_gain=1.0, hippocampal_encoding_enabled=False,
        neuronwise_active_threshold=config.neuronwise_active_threshold,
        neuronwise_bit_lower_threshold=config.neuronwise_bit_lower_threshold,
        neuronwise_bit_upper_threshold=config.neuronwise_bit_upper_threshold,
    ), backend="torch", device="cpu")


def _learn_weight(
    config: AlternativeMemoryConfig,
    cues: torch.Tensor,
    targets: torch.Tensor,
    indices: Sequence[int],
) -> BrainRuntime:
    runtime = _runtime(config); tracker = DelayedSignedEligibility(config)
    for _ in range(config.replay_epochs):
        for index in indices:
            cue, value = cues[index], targets[index]
            runtime.reset_evaluation_state(); runtime.hippocampus = HippocampusMemory(config.dim, capacity=16)
            runtime.step(external_input=config.cue_drive_gain * cue, cue=cue, force_mode=RuntimeMode.WAKE, learning_signal=0.0)
            tracker.observe(runtime.activation)
            runtime.hippocampus.encode(cue, value=value, priority=1.0); runtime.reset_evaluation_state()
            for _ in range(config.replay_ticks):
                runtime.step(external_input=torch.zeros(config.dim), cue=cue, force_mode=RuntimeMode.NREM, learning_signal=0.0)
                tracker.observe(runtime.activation)
            _m1_apply_block(runtime, tracker, 1.0, config)
    return runtime


def _learn_factor_weight(seed: int, config: AlternativeMemoryConfig) -> tuple[BrainRuntime, dict[str, Any]]:
    books = _factor_codebooks(seed, config.dim)
    indices = [books["combinations"].index(value) for value in books["train"]]
    runtime = _learn_weight(config, books["cues"], books["targets"], indices)
    return runtime, books


def _hash_value(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    elif is_dataclass(value):
        for field in fields(value):
            digest.update(field.name.encode()); _hash_value(digest, getattr(value, field.name))
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            digest.update(repr(key).encode()); _hash_value(digest, value[key])
    elif isinstance(value, (tuple, list)):
        for item in value: _hash_value(digest, item)
    else:
        digest.update(repr(value).encode())


def _snapshot_hash(snapshot) -> str:
    digest = hashlib.sha256(); _hash_value(digest, snapshot)
    return digest.hexdigest()


def _rollout(
    snapshot,
    cue: torch.Tensor,
    horizon: int,
    gain: float,
    original_nonzero: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    runtime = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    energy = 0.0; active = 0.0; exposed = 0.0
    for tick in range(horizon + 1):
        pre_active = runtime.active_mask().bool()
        exposed += float(((runtime.weight != 0) & pre_active.unsqueeze(0)).sum().item()) / max(1, original_nonzero)
        step = runtime.step(external_input=(gain * cue if tick == 0 else torch.zeros_like(cue)), force_mode=RuntimeMode.WAKE, learning_signal=0.0)
        energy += step.energy; active += step.active_modules / runtime.config.dim
    return runtime.activation.clone(), {"runtime_energy": energy, "active_fraction": active / (horizon + 1), "exposed_edge_fraction": exposed / (horizon + 1), "hippocampal_rows_after_rollout": float(len(runtime.hippocampus))}


def _architectural_blocks(dim: int) -> tuple[tuple[int, ...], ...]:
    if dim % 4:
        raise ApparatusInvalid("APPARATUS_INVALID: dimension must be divisible by four")
    width = dim // 4
    return tuple(tuple(range(i * width, (i + 1) * width)) for i in range(4))


def _shared_sparse_budget(
    weight: torch.Tensor,
    cues: torch.Tensor,
    blocks: Sequence[Sequence[int]],
    *,
    fraction: float = 0.25,
) -> tuple[int, int]:
    """Return a cue-shared budget from the tightest admissible support.

    This is computed before rollout from the same allowlisted construction
    inputs as the masks. It prevents a full-graph budget from exceeding the
    cluster/path support while keeping every sparse arm on one exact budget.
    """
    matrix = torch.as_tensor(weight, dtype=torch.float64)
    cue_matrix = torch.as_tensor(cues, dtype=torch.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ApparatusInvalid("APPARATUS_INVALID: weight must be square")
    if cue_matrix.ndim != 2 or cue_matrix.shape[1] != matrix.shape[0]:
        raise ApparatusInvalid("APPARATUS_INVALID: incompatible cue matrix")
    if not torch.isfinite(matrix).all() or not torch.isfinite(cue_matrix).all():
        raise ApparatusInvalid("APPARATUS_INVALID: nonfinite budget input")
    if not 0.0 < float(fraction) <= 1.0:
        raise ApparatusInvalid("APPARATUS_INVALID: budget fraction must be in (0, 1]")
    partition = _validate_blocks(blocks, int(matrix.shape[0]))
    support = matrix.abs()
    support.fill_diagonal_(0.0)
    learned = support > 0.0
    if int(learned.sum().item()) == 0:
        raise ApparatusInvalid("APPARATUS_INVALID: empty learned support")
    counts = [
        int(_cluster_admissible(support, learned, cue.abs(), partition).sum().item())
        for cue in cue_matrix
    ]
    minimum = min(counts)
    if minimum <= 0:
        raise ApparatusInvalid("APPARATUS_INVALID: empty shared admissible support")
    return max(1, math.ceil(float(fraction) * minimum)), minimum


def _offdiagonal_count(weight: torch.Tensor) -> int:
    eye = torch.eye(weight.shape[0], dtype=torch.bool, device=weight.device)
    return int(((weight != 0) & ~eye).sum().item())


def _seal(learner: BrainRuntime, temporal: TemporalAuditedMemory) -> tuple[Any, dict[str, int]]:
    removed = {"temporal_rows_removed": len(temporal), "hippocampal_rows_removed": len(learner.hippocampus)}
    temporal._versions.clear(); temporal._evidence_ids.clear()
    learner.hippocampus = HippocampusMemory(learner.config.dim, capacity=16)
    learner.config.hippocampal_encoding_enabled = False; learner.reset_evaluation_state()
    removed.update({"temporal_rows_after": len(temporal), "hippocampal_rows_after": len(learner.hippocampus)})
    return learner.snapshot(), removed


def _mask_sequence(
    weight: torch.Tensor,
    cues: torch.Tensor,
    blocks: tuple[tuple[int, ...], ...],
    seed: int,
    route: str,
    budget: int,
) -> list[torch.Tensor]:
    masks = []
    for index in range(len(cues)):
        cue_index = (index + 1) % len(cues) if route == "WRONG_CONTEXT" else index
        masks.append(construct_route_mask(weight, cues[cue_index], blocks, seed, route, budget))
    return masks


def _mask_metrics(
    weight: torch.Tensor,
    cues: torch.Tensor,
    blocks: tuple[tuple[int, ...], ...],
    seed: int,
    route: str,
    budget: int,
) -> tuple[list[torch.Tensor], float, float, bool]:
    masks = _mask_sequence(weight, cues, blocks, seed, route, budget)
    nonzero = max(1, _offdiagonal_count(weight))
    switch = sum(float((masks[i] != masks[(i + 1) % len(masks)]).sum().item()) / nonzero for i in range(len(masks))) / len(masks)
    topology_path = 0.0; topology_path_applicable = route in {"TOPOLOGY", "PATH_ONLY", "RETURN_SHUFFLED"}
    if topology_path_applicable:
        topology = _mask_sequence(weight, cues, blocks, seed, "TOPOLOGY", budget)
        path = _mask_sequence(weight, cues, blocks, seed, "PATH_ONLY", budget)
        topology_path = sum(float((a != b).sum().item()) / nonzero for a, b in zip(topology, path)) / len(topology)
    return masks, switch, topology_path, topology_path_applicable


def _masked_rollout(
    sealed: Any,
    mask: torch.Tensor,
    cue: torch.Tensor,
    config: TopologyRoutingConfig,
    original_nonzero: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    masked = BrainRuntime.from_snapshot(sealed, backend="torch", device="cpu")
    masked.weight = masked.weight * mask.to(masked.weight); masked._rebuild_sparse()
    return _rollout(masked.snapshot(), cue, config.rollout_horizon, config.cue_drive_gain, original_nonzero)


def _evaluate_factor_route(
    seed: int,
    config: TopologyRoutingConfig,
    route: str,
    sealed: Any,
    books: dict[str, Any],
    cutoff: dict[str, int],
) -> dict[str, Any]:
    before = _snapshot_hash(sealed)
    blocks = _architectural_blocks(config.dim); original_nonzero = _offdiagonal_count(sealed.weight)
    budget, budget_basis = _shared_sparse_budget(sealed.weight, books["cues"], blocks)
    masks, switch_cost, topology_path_hamming, topology_path_applicable = _mask_metrics(
        sealed.weight, books["cues"], blocks, seed, route, budget,
    )
    held = books["combinations"].index(books["held_out"]); mask = masks[held]; cue = books["cues"][held]
    final, metrics = _masked_rollout(sealed, mask, cue, config, original_nonzero)
    cosine = books["targets"] @ _unit(final); correct = float(cosine[held]); wrong = float(torch.cat((cosine[:held], cosine[held + 1:])).max())
    after = _snapshot_hash(sealed); retained = int(mask.sum().item())
    return {"seed": seed, "route": route, "held_out_accuracy": float(_decode(final, books["targets"], abstain_threshold=0.20) == held),
            "separation": correct - wrong, "edge_budget": budget, "retained_edges": retained,
            "budget_exact": route == "FULL" or retained == budget,
            "budget_basis_min_admissible": budget_basis,
            "budget_fraction_of_min_admissible": 0.25,
            "snapshot_immutable": before == after, "temporal_rows_after": cutoff["temporal_rows_after"], "cutoff_audit": cutoff,
            "finite": bool(torch.isfinite(mask).all() and torch.isfinite(final).all()), "delay_ring_length": 2,
            "source_snapshot_sha256": before, "retained_edge_fraction": retained / max(1, original_nonzero),
            "switch_cost": switch_cost, "topology_path_hamming": topology_path_hamming,
            "topology_path_hamming_applicable": topology_path_applicable, **metrics}


def _factor_snapshot(seed: int, config: TopologyRoutingConfig) -> tuple[Any, dict[str, Any], dict[str, int]]:
    learner, books = _learn_factor_weight(seed, config.alternative())
    temporal = TemporalAuditedMemory(capacity=8)
    for position, pair in enumerate(books["train"]):
        temporal.ingest(TemporalMemoryEvent("factor", "pair", str(pair), 1, position, str(position)))
    sealed, cutoff = _seal(learner, temporal)
    return sealed, books, cutoff


def run_topology_route(seed: int = 97301, *, config: TopologyRoutingConfig | None = None, route: str = "TOPOLOGY") -> dict[str, Any]:
    """One actual delayed-Torch factor-transfer circuit, evaluated from a clone."""
    config = config or TopologyRoutingConfig(seed=seed)
    config = TopologyRoutingConfig(**{**asdict(config), "seed": seed})
    sealed, books, cutoff = _factor_snapshot(seed, config)
    return _evaluate_factor_route(seed, config, route, sealed, books, cutoff)


def run_topology_circuit(seed: int = 97301, *, config: TopologyRoutingConfig | None = None) -> dict[str, Any]:
    """Evaluate every frozen route from one shared learned sealed snapshot."""
    config = config or TopologyRoutingConfig(seed=seed)
    config = TopologyRoutingConfig(**{**asdict(config), "seed": seed})
    sealed, books, cutoff = _factor_snapshot(seed, config)
    ordered = ("FULL", "WEIGHT", "CLUSTER", "PATH_ONLY", "TOPOLOGY", "RETURN_SHUFFLED", "RANDOM_MATCHED", "WRONG_CONTEXT")
    rows = {route: _evaluate_factor_route(seed, config, route, sealed, books, cutoff) for route in ordered}
    return {"seed": seed, "source_snapshot_sha256": _snapshot_hash(sealed), "routes": rows}


def run_binding_route(seed: int = 97201, *, config: TopologyRoutingConfig | None = None, route: str = "TOPOLOGY") -> dict[str, Any]:
    """Pairwise M1 binding baseline under the same delayed routed apparatus."""
    config = config or TopologyRoutingConfig(seed=seed)
    config = TopologyRoutingConfig(**{**asdict(config), "seed": seed})
    temporal, source, _ = _loop8_replay_source_audit(); indices = [int(row["value"]) for row in source]
    cues, targets = _codebook(seed, config.dim)
    learner = _learn_weight(config.alternative(), cues, targets, indices)
    sealed, cutoff = _seal(learner, temporal); before = _snapshot_hash(sealed)
    blocks = _architectural_blocks(config.dim); original_nonzero = _offdiagonal_count(sealed.weight)
    budget, budget_basis = _shared_sparse_budget(sealed.weight, cues, blocks)
    masks, switch_cost, topology_path_hamming, topology_path_applicable = _mask_metrics(sealed.weight, cues, blocks, seed, route, budget)
    clean: list[bool] = []; corrupt: list[bool] = []; finite = True; rows = 0.0
    energy = active = exposed = 0.0
    for index in indices:
        final, metrics = _masked_rollout(sealed, masks[index], cues[index], config, original_nonzero)
        clean.append(_decode(final, targets, abstain_threshold=0.20) == index)
        noisy = cues[index].clone(); noisy[:max(1, int(config.dim * config.cue_corruption))] = 0.0
        noisy_cues = cues.clone(); noisy_cues[index] = noisy
        noisy_masks = _mask_sequence(sealed.weight, noisy_cues, blocks, seed, route, budget)
        noisy_final, noisy_metrics = _masked_rollout(sealed, noisy_masks[index], noisy, config, original_nonzero)
        corrupt.append(_decode(noisy_final, targets, abstain_threshold=0.20) == index)
        finite = finite and bool(torch.isfinite(final).all() and torch.isfinite(noisy_final).all())
        for receipt in (metrics, noisy_metrics):
            energy += receipt["runtime_energy"]; active += receipt["active_fraction"]; exposed += receipt["exposed_edge_fraction"]
            rows = max(rows, receipt["hippocampal_rows_after_rollout"])
    after = _snapshot_hash(sealed); retained = int(masks[indices[0]].sum().item()); observations = 2 * len(indices)
    return {"seed": seed, "route": route, "clean_accuracy": sum(clean) / len(clean), "corrupt_accuracy": sum(corrupt) / len(corrupt),
            "edge_budget": budget, "retained_edges": retained, "budget_exact": route == "FULL" or retained == budget,
            "budget_basis_min_admissible": budget_basis,
            "budget_fraction_of_min_admissible": 0.25,
            "snapshot_immutable": before == after, "source_snapshot_sha256": before, "cutoff_audit": cutoff,
            "temporal_rows_after": len(temporal), "hippocampal_rows_after_rollout": rows,
            "finite": finite, "delay_ring_length": 2, "retained_edge_fraction": retained / max(1, original_nonzero),
            "runtime_energy": energy / observations, "active_fraction": active / observations,
            "exposed_edge_fraction": exposed / observations, "switch_cost": switch_cost,
            "topology_path_hamming": topology_path_hamming,
            "topology_path_hamming_applicable": topology_path_applicable}
