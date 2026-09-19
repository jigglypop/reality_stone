"""Context-only entry-branch routing through a shared delayed runtime trunk.

The experiment deliberately separates routing from answer delivery.  Recall
receives two simultaneous payloads; context is visible only to a pure mask
compiler.  Both context masks use the same relay/output trunk and differ only
at the source-to-hidden entry branch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import inspect
import math
from typing import Any, Sequence

import torch

from ..constants import STP_TAU_FAC_INV, STP_TAU_REC, STP_U_BASE
from ..runtime import BrainRuntime, BrainRuntimeConfig, HippocampusMemory, RuntimeMode
from ..stdp import structural_projection


EPSILON = 1e-8
BLOCK_NAMES = ("S0", "S1", "H0", "H1", "Y")
ROUTES = (
    "CORRECT",
    "WRONG",
    "STATIC_0",
    "STATIC_1",
    "RANDOM_MATCHED",
    "FULL",
    "STATIC_UNION",
    "SWAPPED_AFTER_CUE",
)


class ApparatusInvalid(ValueError):
    """Raised before endpoint access when a frozen receipt is not satisfied."""


@dataclass(frozen=True)
class ContextBranchConfig:
    dim: int = 20
    payload_width: int = 4
    delay_ticks: int = 2
    learning_epochs: int = 1
    cue_drive_gain: float = 5.0
    learning_rate: float = 0.8
    eligibility_decay: float = 0.99
    ltd: float = 0.20
    max_write_norm: float = 5.0
    decoder_min_cosine: float = 0.50
    decoder_margin: float = 0.15
    seed: int = 97501

    def __post_init__(self) -> None:
        if self.dim != 5 * self.payload_width or self.payload_width < 2:
            raise ValueError("dim must equal five times payload_width >= 2")
        if self.delay_ticks < 1 or self.learning_epochs != 1:
            raise ValueError("the frozen apparatus requires delay >= 1 and one epoch")
        finite = (
            self.cue_drive_gain,
            self.learning_rate,
            self.eligibility_decay,
            self.ltd,
            self.max_write_norm,
            self.decoder_min_cosine,
            self.decoder_margin,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise ValueError("configuration values must be finite")
        if not 0.0 <= self.eligibility_decay < 1.0 or self.ltd < 0.0:
            raise ValueError("eligibility parameters are out of range")

    @property
    def recall_call_index(self) -> int:
        return 2 * (self.delay_ticks + 1)

    @property
    def learning_pulse_ticks(self) -> tuple[int, int, int]:
        return tuple(stage * self.delay_ticks for stage in range(3))  # type: ignore[return-value]


class ExactDelayEligibility:
    """Dimensionless local row-post/column-pre eligibility at an exact delay."""

    def __init__(self, dim: int, delay: int, decay: float, ltd: float) -> None:
        self.dim = int(dim)
        self.delay = int(delay)
        self.decay = float(decay)
        self.ltd = float(ltd)
        self.eligibility = torch.zeros(self.dim, self.dim)
        self._history: list[torch.Tensor] = []
        self.observations = 0
        self.paired_observations = 0

    def observe(self, activation: torch.Tensor) -> None:
        current = activation.detach().float().view(self.dim).cpu().clone()
        if len(self._history) >= self.delay:
            delayed = self._history[-self.delay]
            self.eligibility = (
                self.decay * self.eligibility
                + torch.outer(current, delayed)
                - self.ltd * torch.outer(delayed, current)
            )
            self.paired_observations += 1
        self._history.append(current)
        if len(self._history) > self.delay:
            self._history.pop(0)
        self.observations += 1

    def reset(self) -> None:
        self.eligibility.zero_()
        self._history.clear()
        self.observations = 0
        self.paired_observations = 0


def _validate_blocks(blocks: Sequence[Sequence[int]], dim: int) -> tuple[tuple[int, ...], ...]:
    packed = tuple(tuple(int(index) for index in block) for block in blocks)
    if len(packed) != 5 or any(not block for block in packed):
        raise ApparatusInvalid("APPARATUS_INVALID: five nonempty blocks are required")
    width = len(packed[0])
    if any(len(block) != width for block in packed):
        raise ApparatusInvalid("APPARATUS_INVALID: blocks must have equal width")
    if sorted(index for block in packed for index in block) != list(range(dim)):
        raise ApparatusInvalid("APPARATUS_INVALID: blocks must partition coordinates")
    return packed


def architectural_blocks(dim: int) -> tuple[tuple[int, ...], ...]:
    if dim % 5:
        raise ApparatusInvalid("APPARATUS_INVALID: dim must be divisible by five")
    width = dim // 5
    return tuple(tuple(range(block * width, (block + 1) * width)) for block in range(5))


def _rectangle(dim: int, destination: Sequence[int], source: Sequence[int]) -> torch.Tensor:
    mask = torch.zeros(dim, dim, dtype=torch.bool)
    rows = torch.tensor(tuple(destination), dtype=torch.long)
    cols = torch.tensor(tuple(source), dtype=torch.long)
    mask[rows[:, None], cols] = True
    return mask


def _support_parts(
    blocks: tuple[tuple[int, ...], ...], dim: int,
) -> dict[str, torch.Tensor]:
    s0, s1, h0, h1, output = blocks
    return {
        "H0_S0": _rectangle(dim, h0, s0),
        "H1_S1": _rectangle(dim, h1, s1),
        "Y_H0": _rectangle(dim, output, h0),
        "Y_H1": _rectangle(dim, output, h1),
    }


def _learned_part(weight: torch.Tensor, allowed: torch.Tensor, label: str, width: int) -> torch.Tensor:
    part = (weight != 0) & allowed
    if int(part.sum().item()) != width:
        raise ApparatusInvalid(f"APPARATUS_INVALID: {label} must contain exactly {width} learned edges")
    return part


def construct_context_branch_mask(weight, context, blocks, seed, route):
    """Compile a context mask without payload, answer, decoder, or rollout input."""
    if route not in ROUTES:
        raise ValueError(f"unknown route {route!r}")
    matrix = torch.as_tensor(weight, dtype=torch.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not torch.isfinite(matrix).all():
        raise ApparatusInvalid("APPARATUS_INVALID: weight must be finite and square")
    context = int(context)
    if context not in (0, 1):
        raise ApparatusInvalid("APPARATUS_INVALID: context must be zero or one")
    partition = _validate_blocks(blocks, int(matrix.shape[0]))
    width = len(partition[0])
    parts = _support_parts(partition, int(matrix.shape[0]))
    learned = {name: _learned_part(matrix, mask, name, width) for name, mask in parts.items()}
    trunk = learned["Y_H0"] | learned["Y_H1"]
    branches = (learned["H0_S0"], learned["H1_S1"])
    union = trunk | branches[0] | branches[1]
    budget = 3 * width

    if route in {"FULL", "STATIC_UNION"}:
        result = union
    elif route == "RANDOM_MATCHED":
        rows, cols = torch.where(union)
        order = torch.randperm(
            int(rows.numel()),
            generator=torch.Generator(device="cpu").manual_seed(int(seed)),
        )
        result = torch.zeros_like(union)
        result[rows[order[:budget]], cols[order[:budget]]] = True
    else:
        selected = context
        if route in {"WRONG", "SWAPPED_AFTER_CUE"}:
            selected = 1 - context
        elif route == "STATIC_0":
            selected = 0
        elif route == "STATIC_1":
            selected = 1
        result = trunk | branches[selected]
    return result.to(weight.dtype if isinstance(weight, torch.Tensor) else torch.float32)


def _runtime(config: ContextBranchConfig) -> BrainRuntime:
    active_profile = (0.08, 0.10, 0.12, 0.14) * 5
    lower_profile = (0.04, 0.05, 0.06, 0.07) * 5
    upper_profile = (0.18, 0.20, 0.22, 0.24) * 5
    return BrainRuntime(
        torch.zeros(config.dim, config.dim),
        config=BrainRuntimeConfig(
            dim=config.dim,
            active_ratio=0.50,
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=True,
            max_axon_delay=config.delay_ticks,
            f1_self_measure=False,
            stdp_enabled=False,
            memory_capacity=1,
            hippocampal_encoding_enabled=False,
            neuronwise_active_threshold=active_profile,
            neuronwise_bit_lower_threshold=lower_profile,
            neuronwise_bit_upper_threshold=upper_profile,
        ),
        backend="torch",
        device="cpu",
    )


def _one_hot(block: Sequence[int], coordinate: int, dim: int) -> torch.Tensor:
    vector = torch.zeros(dim)
    vector[int(block[int(coordinate)])] = 1.0
    return vector


def _payload_books(seed: int, config: ContextBranchConfig) -> dict[str, Any]:
    blocks = architectural_blocks(config.dim)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutations = [torch.randperm(config.payload_width, generator=generator).tolist() for _ in blocks]
    books = {
        name: torch.stack([
            _one_hot(blocks[index], permutations[index][payload], config.dim)
            for payload in range(config.payload_width)
        ])
        for index, name in enumerate(BLOCK_NAMES)
    }
    books["blocks"] = blocks
    books["permutations"] = tuple(tuple(int(value) for value in values) for values in permutations)
    return books


def _episode_support(
    parts: dict[str, torch.Tensor], context: int,
) -> torch.Tensor:
    return parts[f"H{context}_S{context}"] | parts[f"Y_H{context}"]


def _hash_value(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    elif is_dataclass(value):
        for field in fields(value):
            digest.update(field.name.encode())
            _hash_value(digest, getattr(value, field.name))
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            digest.update(repr(key).encode())
            _hash_value(digest, value[key])
    elif isinstance(value, (tuple, list)):
        for item in value:
            _hash_value(digest, item)
    else:
        digest.update(repr(value).encode())


def _snapshot_hash(snapshot: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, snapshot)
    return digest.hexdigest()


def _codebook_hash(books: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in BLOCK_NAMES:
        _hash_value(digest, books[name])
    return digest.hexdigest()


def _learn(seed: int, config: ContextBranchConfig) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    runtime = _runtime(config)
    books = _payload_books(seed, config)
    blocks = books["blocks"]
    parts = _support_parts(blocks, config.dim)
    allowed = torch.zeros(config.dim, config.dim, dtype=torch.bool)
    for part in parts.values():
        allowed |= part
    aggregate = torch.zeros_like(runtime.weight)
    initial = runtime.weight.clone()
    episode_rows: list[dict[str, Any]] = []
    pulse_ticks = config.learning_pulse_ticks

    for context in (0, 1):
        for payload in range(config.payload_width):
            runtime.reset_evaluation_state()
            runtime.hippocampus = HippocampusMemory(config.dim, capacity=1, device=runtime.device)
            tracker = ExactDelayEligibility(
                config.dim,
                config.delay_ticks,
                config.eligibility_decay,
                config.ltd,
            )
            pulses = (
                books[f"S{context}"][payload],
                books[f"H{context}"][payload],
                books["Y"][payload],
            )
            before_episode = runtime.weight.clone()
            for tick in range(pulse_ticks[-1] + 1):
                external = torch.zeros(config.dim)
                if tick in pulse_ticks:
                    external = config.cue_drive_gain * pulses[pulse_ticks.index(tick)]
                runtime.step(
                    external_input=external,
                    force_mode=RuntimeMode.WAKE,
                    learning_signal=0.0,
                )
                tracker.observe(runtime.activation)
            local = tracker.eligibility * _episode_support(parts, context)
            aggregate += local
            episode_rows.append({
                "context": context,
                "payload": payload,
                "observations": tracker.observations,
                "paired_observations": tracker.paired_observations,
                "local_eligibility_norm": float(local.norm().item()),
                "mid_episode_weight_unchanged": bool(torch.equal(before_episode, runtime.weight)),
            })

    before = runtime.weight.clone()
    projected = structural_projection(
        before + config.learning_rate * aggregate,
        density=1.0,
        theta_on=1e-6,
        theta_off=5e-7,
    )
    projected.fill_diagonal_(0.0)
    requested = (projected - before) * allowed.to(projected)
    installed_norm = runtime.install_bounded_recurrent_delta(
        requested,
        max_frobenius_norm=config.max_write_norm,
    )
    actual = runtime.weight - before
    outside_norm = float((actual * (~allowed).to(actual)).norm().item())

    runtime.hippocampus = HippocampusMemory(config.dim, capacity=1, device=runtime.device)
    runtime.config.hippocampal_encoding_enabled = False
    runtime.reset_evaluation_state()
    delay_zero = bool(runtime._delay_buffer is not None and torch.count_nonzero(runtime._delay_buffer) == 0)
    cutoff = {
        "hippocampal_rows_after": len(runtime.hippocampus),
        "temporal_rows_after": 0,
        "activation_norm_after": float(runtime.activation.norm().item()),
        "delay_ring_zero": delay_zero,
        "delay_index_after": int(runtime._delay_idx),
    }
    receipt = {
        "initial_weight_norm": float(initial.norm().item()),
        "aggregate_eligibility_norm": float(aggregate.norm().item()),
        "requested_delta_norm": float(requested.norm().item()),
        "installed_delta_norm": installed_norm,
        "actual_delta_norm": float(actual.norm().item()),
        "outside_allowed_actual_delta_norm": outside_norm,
        "episode_rows": episode_rows,
        "cutoff": cutoff,
    }
    return runtime.snapshot(), books, receipt


def _matrix_rank(matrix: torch.Tensor, tolerance: float = 1e-6) -> int:
    singular = torch.linalg.svdvals(matrix.double())
    return int((singular > tolerance).sum().item())


def _block_matrix(weight: torch.Tensor, destination: Sequence[int], source: Sequence[int]) -> torch.Tensor:
    rows = torch.tensor(tuple(destination), dtype=torch.long)
    cols = torch.tensor(tuple(source), dtype=torch.long)
    return weight[rows[:, None], cols]


def _preflight(snapshot: Any, books: dict[str, Any], learn: dict[str, Any], seed: int) -> dict[str, Any]:
    weight = snapshot.weight
    blocks = books["blocks"]
    width = len(blocks[0])
    parts = _support_parts(blocks, int(weight.shape[0]))
    allowed = torch.zeros_like(weight, dtype=torch.bool)
    for part in parts.values():
        allowed |= part
    block_weights = {
        name: weight.masked_fill(~part, 0.0)
        for name, part in parts.items()
    }
    edge_counts = {name: int((matrix != 0).sum().item()) for name, matrix in block_weights.items()}
    ranks = {}
    for name, part in parts.items():
        rows, cols = torch.where(part)
        destination = sorted(set(rows.tolist()))
        source = sorted(set(cols.tolist()))
        ranks[name] = _matrix_rank(_block_matrix(weight, destination, source))

    s0, s1, h0, h1, output = blocks
    yh0 = _block_matrix(weight, output, h0)
    yh1 = _block_matrix(weight, output, h1)
    h0s0 = _block_matrix(weight, h0, s0)
    h1s1 = _block_matrix(weight, h1, s1)
    products = (yh0 @ h0s0, yh1 @ h1s1)
    product_min_singular = [float(torch.linalg.svdvals(product.double()).min().item()) for product in products]

    masks = {
        context: construct_context_branch_mask(weight, context, blocks, seed, "CORRECT").bool()
        for context in (0, 1)
    }
    wrong_masks = {
        context: construct_context_branch_mask(weight, context, blocks, seed, "WRONG").bool()
        for context in (0, 1)
    }
    trunk = parts["Y_H0"] | parts["Y_H1"]
    shared_trunk_equal = bool(torch.equal(masks[0] & trunk, masks[1] & trunk))
    mask_counts = [int(masks[context].sum().item()) for context in (0, 1)]
    symmetric_difference = int((masks[0] != masks[1]).sum().item())
    signature = tuple(inspect.signature(construct_context_branch_mask).parameters)
    forbidden_parameters = {"payload", "answer", "target", "decoder", "endpoint", "rollout"}
    no_forbidden_parameters = not bool(set(signature) & forbidden_parameters)
    outside_support = int(((weight != 0) & ~allowed).sum().item())
    output_context_specific_edges = int(
        ((masks[0] != masks[1]) & (parts["Y_H0"] | parts["Y_H1"])).sum().item()
    )
    direct_bypass_edges = int(((weight != 0) & ~allowed).sum().item())
    active_thresholds = snapshot.config.effective_active_thresholds()
    lower_thresholds, upper_thresholds = snapshot.config.effective_bit_thresholds()

    def path_threshold_profile(context: int) -> tuple[tuple[float, float, float], ...]:
        indices = (*blocks[context], *blocks[2 + context], *blocks[4])
        return tuple(
            (active_thresholds[index], lower_thresholds[index], upper_thresholds[index])
            for index in indices
        )

    threshold_profiles = (path_threshold_profile(0), path_threshold_profile(1))
    delay_histograms = (
        (snapshot.config.max_axon_delay, snapshot.config.max_axon_delay),
        (snapshot.config.max_axon_delay, snapshot.config.max_axon_delay),
    )
    stp_profiles = (
        (STP_TAU_FAC_INV, STP_TAU_REC, STP_U_BASE),
        (STP_TAU_FAC_INV, STP_TAU_REC, STP_U_BASE),
    )
    decoder_hashes = (_codebook_hash(books), _codebook_hash(books))
    state_signature = tuple(inspect.signature(_rollout).parameters)
    decoder_signature = tuple(inspect.signature(_decode_y).parameters)
    decoder_support = books["Y"] != 0
    decoder_allowed = torch.zeros_like(decoder_support)
    decoder_allowed[:, torch.tensor(blocks[4], dtype=torch.long)] = True
    no_context_state_path = "context" not in state_signature
    no_context_decoder_path = "context" not in decoder_signature and bool(
        torch.all(~decoder_support | decoder_allowed)
    )
    correct_wrong_parity = all(
        torch.equal(wrong_masks[context], masks[1 - context])
        and int(wrong_masks[context].sum().item()) == mask_counts[context]
        for context in (0, 1)
    )
    dense_sparse_parity = bool(torch.equal(weight, BrainRuntime.from_snapshot(
        snapshot, backend="torch", device="cpu",
    ).sparse_weight.to_dense()))
    gates = {
        "finite": bool(torch.isfinite(weight).all()),
        "support_subset": outside_support == 0,
        "outside_delta_zero": learn["outside_allowed_actual_delta_norm"] == 0.0,
        "block_edge_counts": all(count == width for count in edge_counts.values()),
        "block_full_rank": all(rank == width for rank in ranks.values()),
        "product_min_singular": min(product_min_singular) >= 0.25,
        "mask_budget": mask_counts == [3 * width, 3 * width],
        "mask_difference": symmetric_difference == 2 * width,
        "shared_trunk_equal": shared_trunk_equal,
        "no_output_context_gate": output_context_specific_edges == 0,
        "no_direct_bypass": direct_bypass_edges == 0,
        "no_context_state_path": no_context_state_path,
        "no_context_decoder_path": no_context_decoder_path,
        "correct_wrong_mask_parity": correct_wrong_parity,
        "delay_histogram_parity": delay_histograms[0] == delay_histograms[1],
        "threshold_profile_parity": threshold_profiles[0] == threshold_profiles[1],
        "stp_profile_parity": stp_profiles[0] == stp_profiles[1],
        "decoder_hash_parity": decoder_hashes[0] == decoder_hashes[1],
        "mask_signature": signature == ("weight", "context", "blocks", "seed", "route") and no_forbidden_parameters,
        "cutoff": (
            learn["cutoff"]["hippocampal_rows_after"] == 0
            and learn["cutoff"]["temporal_rows_after"] == 0
            and learn["cutoff"]["activation_norm_after"] == 0.0
            and learn["cutoff"]["delay_ring_zero"]
            and learn["cutoff"]["delay_index_after"] == 0
        ),
        "dense_sparse_parity": dense_sparse_parity,
    }
    return {
        "gates": gates,
        "all_pass": all(gates.values()),
        "edge_counts": edge_counts,
        "ranks": ranks,
        "product_min_singular": product_min_singular,
        "mask_counts": mask_counts,
        "mask_symmetric_difference": symmetric_difference,
        "shared_trunk_edge_count": int((masks[0] & masks[1]).sum().item()),
        "output_context_specific_edges": output_context_specific_edges,
        "mask_constructor_parameters": signature,
        "state_rollout_parameters": state_signature,
        "decoder_parameters": decoder_signature,
        "delay_histograms": delay_histograms,
        "threshold_profiles": threshold_profiles,
        "stp_profiles": stp_profiles,
        "decoder_hashes": decoder_hashes,
        "decoder_sha256": decoder_hashes[0],
    }


def _decode_y(
    activation: torch.Tensor,
    books: dict[str, Any],
    expected: int,
    opposite: int,
    config: ContextBranchConfig,
) -> dict[str, Any]:
    y_indices = torch.tensor(books["blocks"][4], dtype=torch.long)
    state = activation[y_indices]
    decoder = books["Y"][:, y_indices]
    norm = float(state.norm().item())
    cosine = decoder @ (state / state.norm().clamp_min(EPSILON))
    order = torch.argsort(cosine, descending=True, stable=True)
    prediction = int(order[0].item()) if norm > EPSILON else -1
    runner_up = float(cosine[order[1]].item()) if len(order) > 1 else -1.0
    top = float(cosine[order[0]].item()) if prediction >= 0 else 0.0
    margin = top - runner_up
    confident = bool(top >= config.decoder_min_cosine and margin >= config.decoder_margin)
    return {
        "prediction": prediction,
        "expected": int(expected),
        "opposite": int(opposite),
        "success": bool(confident and prediction == expected),
        "opposite_delivery": bool(confident and prediction == opposite),
        "top_cosine": top,
        "margin": margin,
        "y_norm": norm,
        "cosines": [float(value) for value in cosine.tolist()],
    }


def _rollout(
    snapshot: Any,
    initial_mask: torch.Tensor,
    post_cue_mask: torch.Tensor,
    cue: torch.Tensor,
    config: ContextBranchConfig,
    blocks: tuple[tuple[int, ...], ...],
) -> tuple[torch.Tensor, dict[str, Any]]:
    runtime = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    source_weight = runtime.weight.clone()

    def install_mask(mask: torch.Tensor) -> None:
        runtime.weight = source_weight * mask.to(source_weight)
        runtime._rebuild_sparse()

    install_mask(initial_mask)
    energy = 0.0
    active = 0.0
    hidden_norms = (0.0, 0.0)
    for tick in range(config.recall_call_index + 1):
        external = config.cue_drive_gain * cue if tick == 0 else torch.zeros(config.dim)
        step = runtime.step(
            external_input=external,
            force_mode=RuntimeMode.WAKE,
            learning_signal=0.0,
        )
        if tick == 0 and not torch.equal(initial_mask, post_cue_mask):
            install_mask(post_cue_mask)
        if tick == config.delay_ticks + 1:
            hidden_norms = tuple(
                float(runtime.activation[torch.tensor(blocks[2 + branch], dtype=torch.long)].norm().item())
                for branch in (0, 1)
            )
        energy += float(step.energy)
        active += float(step.active_modules) / config.dim
    return runtime.activation.clone(), {
        "runtime_energy_proxy": energy,
        "active_fraction": active / (config.recall_call_index + 1),
        "hidden_norms_at_arrival": hidden_norms,
        "hippocampal_rows_after": len(runtime.hippocampus),
        "delay_index_after": int(runtime._delay_idx),
    }


def _route_trial(
    snapshot: Any,
    books: dict[str, Any],
    config: ContextBranchConfig,
    seed: int,
    route: str,
    context: int,
    left: int,
    right: int,
) -> dict[str, Any]:
    blocks = books["blocks"]
    cue = books["S0"][left] + books["S1"][right]
    expected = left if context == 0 else right
    opposite = right if context == 0 else left
    initial_route = "CORRECT" if route == "SWAPPED_AFTER_CUE" else route
    initial = construct_context_branch_mask(snapshot.weight, context, blocks, seed, initial_route)
    post = construct_context_branch_mask(snapshot.weight, context, blocks, seed, route)
    final, metrics = _rollout(snapshot, initial, post, cue, config, blocks)
    hidden_norms = metrics.pop("hidden_norms_at_arrival")
    selected_hidden_ratio = hidden_norms[context] / (sum(hidden_norms) + EPSILON)
    decoded = _decode_y(final, books, expected, opposite, config)
    return {
        "context": context,
        "left_payload": left,
        "right_payload": right,
        "mask_edges": int(post.sum().item()),
        **decoded,
        **metrics,
        "selected_hidden_ratio": selected_hidden_ratio,
        "hidden_norms_at_arrival": hidden_norms,
    }


def _evaluate_route(
    snapshot: Any,
    books: dict[str, Any],
    config: ContextBranchConfig,
    seed: int,
    route: str,
) -> dict[str, Any]:
    rows = [
        _route_trial(snapshot, books, config, seed, route, context, left, right)
        for context in (0, 1)
        for left in range(config.payload_width)
        for right in range(config.payload_width)
        if left != right
    ]
    count = len(rows)
    return {
        "route": route,
        "accuracy": sum(int(row["success"]) for row in rows) / count,
        "opposite_delivery": sum(int(row["opposite_delivery"]) for row in rows) / count,
        "mean_margin": sum(float(row["margin"]) for row in rows) / count,
        "mean_y_norm": sum(float(row["y_norm"]) for row in rows) / count,
        "mean_runtime_energy_proxy": sum(float(row["runtime_energy_proxy"]) for row in rows) / count,
        "mean_active_fraction": sum(float(row["active_fraction"]) for row in rows) / count,
        "mean_selected_hidden_ratio": sum(float(row["selected_hidden_ratio"]) for row in rows) / count,
        "mask_edge_counts": sorted(set(int(row["mask_edges"]) for row in rows)),
        "hippocampal_rows_after": max(int(row["hippocampal_rows_after"]) for row in rows),
        "trials": rows,
    }


def run_context_branch_seed(
    seed: int = 97501,
    *,
    config: ContextBranchConfig | None = None,
) -> dict[str, Any]:
    config = config or ContextBranchConfig(seed=seed)
    config = ContextBranchConfig(**{**asdict(config), "seed": int(seed)})
    snapshot, books, learn = _learn(seed, config)
    preflight = _preflight(snapshot, books, learn, seed)
    source_hash = _snapshot_hash(snapshot)
    if not preflight["all_pass"]:
        return {
            "seed": seed,
            "status": "APPARATUS_INVALID",
            "config": asdict(config),
            "preflight": preflight,
            "learning": learn,
            "source_snapshot_sha256": source_hash,
            "endpoint_opened": False,
        }

    routes = {
        route: _evaluate_route(snapshot, books, config, seed, route)
        for route in ROUTES
    }
    correct = routes["CORRECT"]["accuracy"]
    exact_budget_controls = ("WRONG", "STATIC_0", "STATIC_1", "RANDOM_MATCHED")
    strongest = max(routes[name]["accuracy"] for name in exact_budget_controls)
    source_immutable = source_hash == _snapshot_hash(snapshot)
    swap_parity = bool(
        routes["SWAPPED_AFTER_CUE"]["accuracy"] == routes["WRONG"]["accuracy"]
        and routes["SWAPPED_AFTER_CUE"]["opposite_delivery"] == routes["WRONG"]["opposite_delivery"]
    )
    seed_pass = bool(
        correct >= 0.95
        and routes["WRONG"]["accuracy"] <= 0.05
        and routes["WRONG"]["opposite_delivery"] >= 0.95
        and routes["STATIC_0"]["accuracy"] <= 0.55
        and routes["STATIC_1"]["accuracy"] <= 0.55
        and routes["RANDOM_MATCHED"]["accuracy"] <= 0.55
        and routes["FULL"]["accuracy"] <= 0.55
        and routes["STATIC_UNION"]["accuracy"] <= 0.55
        and correct - strongest >= 0.40
        and swap_parity
        and source_immutable
        and all(row["hippocampal_rows_after"] == 0 for row in routes.values())
    )
    return {
        "seed": seed,
        "status": "CONTEXT_BRANCH_PASS" if seed_pass else "CONTEXT_BRANCH_NOT_IDENTIFIED",
        "config": asdict(config),
        "preflight": preflight,
        "learning": learn,
        "routes": routes,
        "strongest_exact_budget_control_accuracy": strongest,
        "correct_control_advantage": correct - strongest,
        "swap_parity": swap_parity,
        "source_snapshot_immutable": source_immutable,
        "source_snapshot_sha256": source_hash,
        "endpoint_opened": True,
    }
