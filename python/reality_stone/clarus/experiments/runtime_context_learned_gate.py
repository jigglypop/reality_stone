"""Experience-learned context gate for the frozen TR3 branch apparatus.

The recurrent payload circuit is inherited unchanged from TR3.  A separate
two-actuator gate learns only a context-cue/branch-use association.  At recall
the frozen gate sees a context code, selects one entry branch, and then hands a
binary recurrent mask to the unchanged zero-store rollout.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import inspect
import math
import textwrap
from typing import Any, Sequence

import torch

from ..runtime import RuntimeMode
from .runtime_context_branch_routing import (
    EPSILON,
    ApparatusInvalid,
    ContextBranchConfig,
    ExactDelayEligibility,
    _codebook_hash,
    _decode_y,
    _learn,
    _learned_part,
    _preflight as branch_preflight,
    _rollout,
    _runtime,
    _snapshot_hash,
    _support_parts,
    construct_context_branch_mask,
)


ROUTES = (
    "ORACLE",
    "LEARNED",
    "CONTEXT_SHUFFLE_TRAIN",
    "WRONG_CUE",
    "POST_CUE_SWAP",
    "GATE_LESION_STATIC_0",
    "STATIC_1",
    "CANONICAL_CUE_MAP",
    "RANDOM_MATCHED",
    "FULL",
)

FORBIDDEN_GATE_NAMES = {
    "answer",
    "context_index",
    "decoder",
    "endpoint",
    "expected",
    "oracle",
    "payload",
    "route",
    "schedule",
    "seed",
    "sigma",
    "target",
    "task",
}


@dataclass(frozen=True)
class LearnedContextGateConfig:
    cue_dim: int = 4
    gate_learning_rate: float = 1.0
    gate_weight_clip: float = 4.0
    gate_min_logit_margin: float = 1e-6
    seed: int = 97601

    def __post_init__(self) -> None:
        if self.cue_dim < 2:
            raise ValueError("cue_dim must be at least two")
        values = (
            self.gate_learning_rate,
            self.gate_weight_clip,
            self.gate_min_logit_margin,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("gate configuration values must be finite")
        if self.gate_learning_rate <= 0.0 or self.gate_weight_clip <= 0.0:
            raise ValueError("gate learning rate and clip must be positive")
        if self.gate_min_logit_margin <= 0.0:
            raise ValueError("gate margin must be positive")


@dataclass(frozen=True)
class GateSnapshot:
    theta: torch.Tensor
    cue_dim: int
    update_count: int
    learning_rate: float
    weight_clip: float
    min_logit_margin: float


class LocalContextGate:
    """Bounded dimensionless Hebbian association for two gate actuators."""

    def __init__(self, config: LearnedContextGateConfig) -> None:
        self.cue_dim = int(config.cue_dim)
        self.learning_rate = float(config.gate_learning_rate)
        self.weight_clip = float(config.gate_weight_clip)
        self.min_logit_margin = float(config.gate_min_logit_margin)
        self.theta = torch.zeros(2, self.cue_dim, dtype=torch.float64)
        self.update_count = 0

    def observe(self, context_cue: torch.Tensor, branch_use: torch.Tensor) -> float:
        cue = torch.as_tensor(context_cue, dtype=torch.float64).view(-1)
        use = torch.as_tensor(branch_use, dtype=torch.float64).view(-1)
        if cue.shape != (self.cue_dim,) or use.shape != (2,):
            raise ApparatusInvalid("APPARATUS_INVALID: invalid gate observation shape")
        if not torch.isfinite(cue).all() or not torch.isfinite(use).all():
            raise ApparatusInvalid("APPARATUS_INVALID: nonfinite gate observation")
        if torch.any(use < 0.0):
            raise ApparatusInvalid("APPARATUS_INVALID: branch use must be nonnegative")
        delta = self.learning_rate * torch.outer(use, cue)
        self.theta = (self.theta + delta).clamp(-self.weight_clip, self.weight_clip)
        self.update_count += 1
        return float(delta.norm().item())

    def snapshot(self) -> GateSnapshot:
        return GateSnapshot(
            theta=self.theta.detach().clone(),
            cue_dim=self.cue_dim,
            update_count=self.update_count,
            learning_rate=self.learning_rate,
            weight_clip=self.weight_clip,
            min_logit_margin=self.min_logit_margin,
        )


def _gate_hash(snapshot: GateSnapshot) -> str:
    digest = hashlib.sha256()
    tensor = snapshot.theta.detach().cpu().contiguous()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(tensor.numpy().tobytes())
    digest.update(repr((
        snapshot.cue_dim,
        snapshot.update_count,
        snapshot.learning_rate,
        snapshot.weight_clip,
        snapshot.min_logit_margin,
    )).encode())
    return digest.hexdigest()


def _validate_gate_snapshot(snapshot: GateSnapshot) -> torch.Tensor:
    theta = torch.as_tensor(snapshot.theta, dtype=torch.float64)
    if theta.shape != (2, int(snapshot.cue_dim)) or not torch.isfinite(theta).all():
        raise ApparatusInvalid("APPARATUS_INVALID: invalid frozen gate matrix")
    if snapshot.update_count <= 0:
        raise ApparatusInvalid("APPARATUS_INVALID: gate has no learning updates")
    if snapshot.min_logit_margin <= 0.0:
        raise ApparatusInvalid("APPARATUS_INVALID: invalid frozen gate margin")
    return theta


def _entry_and_trunk_masks(
    weight: torch.Tensor,
    blocks: Sequence[Sequence[int]],
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    matrix = torch.as_tensor(weight)
    packed = tuple(tuple(int(index) for index in block) for block in blocks)
    parts = _support_parts(packed, int(matrix.shape[0]))
    width = len(packed[0])
    branches = (
        _learned_part(matrix, parts["H0_S0"], "H0_S0", width).bool(),
        _learned_part(matrix, parts["H1_S1"], "H1_S1", width).bool(),
    )
    trunk = (
        _learned_part(matrix, parts["Y_H0"], "Y_H0", width)
        | _learned_part(matrix, parts["Y_H1"], "Y_H1", width)
    ).bool()
    return branches, trunk


def compile_learned_mask(
    gate_snapshot: GateSnapshot,
    context_cue: torch.Tensor,
    weight: torch.Tensor,
    blocks: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compile from frozen gate, cue, weight, and blocks only."""
    theta = _validate_gate_snapshot(gate_snapshot)
    cue = torch.as_tensor(context_cue, dtype=torch.float64).view(-1)
    if cue.shape != (gate_snapshot.cue_dim,) or not torch.isfinite(cue).all():
        raise ApparatusInvalid("APPARATUS_INVALID: invalid context cue")
    logits = theta @ cue
    if not torch.isfinite(logits).all():
        raise ApparatusInvalid("APPARATUS_INVALID: nonfinite gate logits")
    margin = float(torch.abs(logits[0] - logits[1]).item())
    if margin < gate_snapshot.min_logit_margin:
        raise ApparatusInvalid("APPARATUS_INVALID: unresolved gate tie")
    selected = int(torch.argmax(logits).item())
    branches, trunk = _entry_and_trunk_masks(weight, blocks)
    mask = trunk | branches[selected]
    return mask.to(weight.dtype), {
        "selected_branch": selected,
        "logits": [float(value) for value in logits.tolist()],
        "logit_margin": margin,
        "mask_edges": int(mask.sum().item()),
    }


def _reference_action(snapshot: GateSnapshot, cue: torch.Tensor) -> tuple[int, list[float], float]:
    """Independent serialized-state evaluator used only for anti-oracle receipts."""
    theta = torch.as_tensor(snapshot.theta, dtype=torch.float64).detach().clone()
    vector = torch.as_tensor(cue, dtype=torch.float64).detach().clone().view(snapshot.cue_dim)
    scores = torch.mv(theta, vector)
    separation = float(abs(float(scores[0].item()) - float(scores[1].item())))
    if not torch.isfinite(scores).all() or separation < snapshot.min_logit_margin:
        raise ApparatusInvalid("APPARATUS_INVALID: reference gate action unresolved")
    return int(torch.argmax(scores).item()), [float(value) for value in scores.tolist()], separation


def _context_task(seed: int, cue_dim: int) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 704_003)
    raw = torch.randn(2, cue_dim, generator=generator, dtype=torch.float64)
    first = raw[0] / raw[0].norm().clamp_min(EPSILON)
    second_raw = raw[1] - torch.dot(raw[1], first) * first
    if float(second_raw.norm().item()) <= 1e-6:
        raise ApparatusInvalid("APPARATUS_INVALID: context-code construction is singular")
    second = second_raw / second_raw.norm()
    cues = torch.stack((first, second))
    first_branch = int(seed) % 2
    mapping = (first_branch, 1 - first_branch)
    return {"cues": cues, "mapping": mapping}


def _branch_use(
    eligibility: torch.Tensor,
    branch_masks: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    positive = torch.as_tensor(eligibility, dtype=torch.float64).clamp_min(0.0)
    return torch.stack(tuple(
        (positive * mask.to(positive)).sum() / int(mask.sum().item())
        for mask in branch_masks
    ))


def _train_gate(
    source_snapshot: Any,
    books: dict[str, Any],
    task: dict[str, Any],
    config: LearnedContextGateConfig,
    *,
    shuffle_context: bool,
) -> tuple[GateSnapshot, dict[str, Any]]:
    base = ContextBranchConfig(seed=int(config.seed))
    runtime = _runtime(base)
    blocks = books["blocks"]
    branch_masks, _ = _entry_and_trunk_masks(source_snapshot.weight, blocks)
    gate = LocalContextGate(config)
    rows: list[dict[str, Any]] = []
    source_hash_before = _snapshot_hash(source_snapshot)

    for payload in range(base.payload_width):
        for context_slot in (0, 1):
            runtime.reset_evaluation_state()
            tracker = ExactDelayEligibility(
                base.dim,
                base.delay_ticks,
                base.eligibility_decay,
                base.ltd,
            )
            experienced_branch = int(task["mapping"][context_slot])
            source = books[f"S{experienced_branch}"][payload]
            hidden = books[f"H{experienced_branch}"][payload]
            for tick in range(base.delay_ticks + 1):
                external = torch.zeros(base.dim)
                if tick == 0:
                    external = base.cue_drive_gain * source
                elif tick == base.delay_ticks:
                    external = base.cue_drive_gain * hidden
                runtime.step(
                    external_input=external,
                    force_mode=RuntimeMode.WAKE,
                    learning_signal=0.0,
                )
                tracker.observe(runtime.activation)
            use = _branch_use(tracker.eligibility, branch_masks)
            observed_slot = 1 - context_slot if shuffle_context else context_slot
            update_norm = gate.observe(task["cues"][observed_slot], use)
            rows.append({
                "context_slot": context_slot,
                "observed_context_slot": observed_slot,
                "experienced_branch": experienced_branch,
                "payload_slot": payload,
                "branch_use": [float(value) for value in use.tolist()],
                "experienced_branch_use": float(use[experienced_branch].item()),
                "other_branch_use": float(use[1 - experienced_branch].item()),
                "update_norm": update_norm,
                "paired_observations": tracker.paired_observations,
            })

    frozen = gate.snapshot()
    receipt = {
        "shuffle_context": bool(shuffle_context),
        "update_count": frozen.update_count,
        "theta_norm": float(frozen.theta.norm().item()),
        "theta": [[float(value) for value in row] for row in frozen.theta.tolist()],
        "gate_sha256": _gate_hash(frozen),
        "source_snapshot_sha256_before": source_hash_before,
        "source_snapshot_sha256_after": _snapshot_hash(source_snapshot),
        "source_snapshot_immutable": source_hash_before == _snapshot_hash(source_snapshot),
        "target_pulse_count": 0,
        "decoder_read_count": 0,
        "endpoint_read_count": 0,
        "rows": rows,
    }
    return frozen, receipt


def _function_identifiers(function: Any) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _preflight(
    source_snapshot: Any,
    books: dict[str, Any],
    recurrent_receipt: dict[str, Any],
    task: dict[str, Any],
    gate_snapshot: GateSnapshot,
    gate_receipt: dict[str, Any],
    shuffled_snapshot: GateSnapshot,
    shuffled_receipt: dict[str, Any],
    config: LearnedContextGateConfig,
) -> dict[str, Any]:
    base_receipt = branch_preflight(
        source_snapshot,
        books,
        recurrent_receipt,
        int(config.seed),
    )
    cues = torch.as_tensor(task["cues"], dtype=torch.float64)
    mapping = tuple(int(value) for value in task["mapping"])
    learned = [compile_learned_mask(gate_snapshot, cues[index], source_snapshot.weight, books["blocks"])
               for index in (0, 1)]
    shuffled = [compile_learned_mask(shuffled_snapshot, cues[index], source_snapshot.weight, books["blocks"])
                for index in (0, 1)]
    references = [_reference_action(gate_snapshot, cues[index]) for index in (0, 1)]
    learned_masks = [row[0].bool() for row in learned]
    learned_info = [row[1] for row in learned]
    shuffled_masks = [row[0].bool() for row in shuffled]
    shuffled_info = [row[1] for row in shuffled]
    branch_masks, trunk = _entry_and_trunk_masks(source_snapshot.weight, books["blocks"])
    expected_masks = [trunk | branch_masks[mapping[index]] for index in (0, 1)]

    compiler_signature = tuple(inspect.signature(compile_learned_mask).parameters)
    update_signature = tuple(inspect.signature(LocalContextGate.observe).parameters)
    identifiers = _function_identifiers(compile_learned_mask) | _function_identifiers(LocalContextGate.observe)
    compiler_no_closure = compile_learned_mask.__closure__ is None
    independent_reference = all(
        references[index][0] == learned_info[index]["selected_branch"]
        and references[index][1] == learned_info[index]["logits"]
        and torch.equal(learned_masks[index], expected_masks[index])
        for index in (0, 1)
    )

    metadata_variant = {
        "seed": int(config.seed) + 1,
        "mapping": tuple(reversed(mapping)),
        "schedule": "adverse-metadata-only",
    }
    del metadata_variant
    metadata_invariant_masks = [
        compile_learned_mask(gate_snapshot, cues[index], source_snapshot.weight, books["blocks"])[0].bool()
        for index in (0, 1)
    ]
    metadata_invariance = all(
        torch.equal(learned_masks[index], metadata_invariant_masks[index]) for index in (0, 1)
    )
    cue_swap_masks = [
        compile_learned_mask(gate_snapshot, cues[1 - index], source_snapshot.weight, books["blocks"])[0].bool()
        for index in (0, 1)
    ]
    cue_swap_equivariance = all(
        torch.equal(cue_swap_masks[index], learned_masks[1 - index]) for index in (0, 1)
    )
    theta_swapped_snapshot = GateSnapshot(
        theta=gate_snapshot.theta.flip(0).clone(),
        cue_dim=gate_snapshot.cue_dim,
        update_count=gate_snapshot.update_count,
        learning_rate=gate_snapshot.learning_rate,
        weight_clip=gate_snapshot.weight_clip,
        min_logit_margin=gate_snapshot.min_logit_margin,
    )
    theta_swapped = [
        compile_learned_mask(
            theta_swapped_snapshot,
            cues[index],
            source_snapshot.weight,
            books["blocks"],
        )
        for index in (0, 1)
    ]
    theta_swapped_references = [
        _reference_action(theta_swapped_snapshot, cues[index]) for index in (0, 1)
    ]
    theta_counterfactual_dependence = all(
        theta_swapped[index][1]["selected_branch"] == theta_swapped_references[index][0]
        and theta_swapped[index][1]["selected_branch"] == 1 - learned_info[index]["selected_branch"]
        and torch.equal(theta_swapped[index][0].bool(), learned_masks[1 - index])
        for index in (0, 1)
    )

    norms = [float(cues[index].norm().item()) for index in (0, 1)]
    orthogonality = float(abs(torch.dot(cues[0], cues[1]).item()))
    mask_counts = [int(mask.sum().item()) for mask in learned_masks]
    mask_difference = int((learned_masks[0] != learned_masks[1]).sum().item())
    shared_trunk = bool(torch.equal(learned_masks[0] & trunk, learned_masks[1] & trunk))
    experienced_separation = all(
        row["experienced_branch_use"] > row["other_branch_use"] + 1e-6
        for row in gate_receipt["rows"]
    )
    learned_actions = tuple(row["selected_branch"] for row in learned_info)
    shuffled_actions = tuple(row["selected_branch"] for row in shuffled_info)
    expected_shuffled = tuple(1 - value for value in mapping)
    gate_norm_parity = abs(gate_receipt["theta_norm"] - shuffled_receipt["theta_norm"]) <= 1e-10
    source_hash = _snapshot_hash(source_snapshot)

    gates = {
        "base_preflight": bool(base_receipt["all_pass"]),
        "context_code_shape": tuple(cues.shape) == (2, config.cue_dim),
        "context_code_finite": bool(torch.isfinite(cues).all()),
        "context_code_unit": max(abs(value - 1.0) for value in norms) <= 1e-10,
        "context_code_orthogonal": orthogonality <= 1e-10,
        "mapping_bijection": sorted(mapping) == [0, 1],
        "gate_update_count": gate_snapshot.update_count == shuffled_snapshot.update_count == 8,
        "gate_finite_nonzero": bool(
            torch.isfinite(gate_snapshot.theta).all() and gate_snapshot.theta.norm() > 0.0
        ),
        "local_branch_use_separation": experienced_separation,
        "no_target_decoder_endpoint_reads": (
            gate_receipt["target_pulse_count"] == 0
            and gate_receipt["decoder_read_count"] == 0
            and gate_receipt["endpoint_read_count"] == 0
        ),
        "source_snapshot_immutable_during_gate_learning": bool(
            gate_receipt["source_snapshot_immutable"]
            and shuffled_receipt["source_snapshot_immutable"]
            and source_hash == gate_receipt["source_snapshot_sha256_before"]
        ),
        "gate_input_signature": (
            compiler_signature == ("gate_snapshot", "context_cue", "weight", "blocks")
            and update_signature == ("self", "context_cue", "branch_use")
            and not bool(identifiers & FORBIDDEN_GATE_NAMES)
            and compiler_no_closure
        ),
        "independent_theta_q_reference": independent_reference,
        "seed_sigma_schedule_metadata_invariance": metadata_invariance,
        "cue_swap_equivariance": cue_swap_equivariance,
        "theta_counterfactual_dependence": theta_counterfactual_dependence,
        "learned_selects_experienced_mapping": learned_actions == mapping,
        "shuffled_training_reverses_mapping": shuffled_actions == expected_shuffled,
        "gate_norm_update_parity": gate_norm_parity,
        "learned_mask_budget": mask_counts == [12, 12],
        "learned_mask_difference": mask_difference == 8,
        "shared_output_trunk": shared_trunk and int((learned_masks[0] & trunk).sum().item()) == 8,
        "gate_margins": min(row["logit_margin"] for row in learned_info) >= config.gate_min_logit_margin,
    }
    return {
        "all_pass": all(gates.values()),
        "gates": gates,
        "base_preflight": base_receipt,
        "context_codes": [[float(value) for value in row] for row in cues.tolist()],
        "context_code_norms": norms,
        "context_code_inner_product_abs": orthogonality,
        "task_mapping": mapping,
        "learned_actions": learned_actions,
        "shuffled_actions": shuffled_actions,
        "reference_actions": tuple(row[0] for row in references),
        "theta_swapped_reference_actions": tuple(row[0] for row in theta_swapped_references),
        "learned_logits": tuple(row["logits"] for row in learned_info),
        "learned_margins": tuple(row["logit_margin"] for row in learned_info),
        "mask_counts": mask_counts,
        "mask_difference": mask_difference,
        "compiler_signature": compiler_signature,
        "update_signature": update_signature,
        "compiler_identifiers": sorted(identifiers),
        "gate_sha256": _gate_hash(gate_snapshot),
        "shuffled_gate_sha256": _gate_hash(shuffled_snapshot),
        "runtime_snapshot_sha256": source_hash,
        "decoder_sha256": _codebook_hash(books),
    }


def _mask_for_branch(
    weight: torch.Tensor,
    blocks: Sequence[Sequence[int]],
    selected_branch: int,
) -> torch.Tensor:
    return construct_context_branch_mask(weight, int(selected_branch), blocks, 0, "CORRECT")


def _route_masks(
    route_name: str,
    context_slot: int,
    source_snapshot: Any,
    books: dict[str, Any],
    task: dict[str, Any],
    gate_snapshot: GateSnapshot,
    shuffled_snapshot: GateSnapshot,
    random_key: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    weight = source_snapshot.weight
    blocks = books["blocks"]
    cue = task["cues"][context_slot]
    wrong_cue = task["cues"][1 - context_slot]
    mapping = task["mapping"]
    gate_info: dict[str, Any] = {"gate_logit_margin": None, "selected_branch": -1}

    if route_name == "ORACLE":
        selected = int(mapping[context_slot])
        initial = post = _mask_for_branch(weight, blocks, selected)
        gate_info["selected_branch"] = selected
    elif route_name == "LEARNED":
        initial, info = compile_learned_mask(gate_snapshot, cue, weight, blocks)
        post = initial
        gate_info.update(selected_branch=info["selected_branch"], gate_logit_margin=info["logit_margin"])
    elif route_name == "CONTEXT_SHUFFLE_TRAIN":
        initial, info = compile_learned_mask(shuffled_snapshot, cue, weight, blocks)
        post = initial
        gate_info.update(selected_branch=info["selected_branch"], gate_logit_margin=info["logit_margin"])
    elif route_name == "WRONG_CUE":
        initial, info = compile_learned_mask(gate_snapshot, wrong_cue, weight, blocks)
        post = initial
        gate_info.update(selected_branch=info["selected_branch"], gate_logit_margin=info["logit_margin"])
    elif route_name == "POST_CUE_SWAP":
        initial, info = compile_learned_mask(gate_snapshot, cue, weight, blocks)
        post, wrong_info = compile_learned_mask(gate_snapshot, wrong_cue, weight, blocks)
        gate_info.update(
            selected_branch=wrong_info["selected_branch"],
            initial_selected_branch=info["selected_branch"],
            gate_logit_margin=wrong_info["logit_margin"],
        )
    elif route_name == "GATE_LESION_STATIC_0":
        initial = post = _mask_for_branch(weight, blocks, 0)
        gate_info["selected_branch"] = 0
    elif route_name == "STATIC_1":
        initial = post = _mask_for_branch(weight, blocks, 1)
        gate_info["selected_branch"] = 1
    elif route_name == "CANONICAL_CUE_MAP":
        initial = post = _mask_for_branch(weight, blocks, context_slot)
        gate_info["selected_branch"] = context_slot
    elif route_name == "RANDOM_MATCHED":
        initial = post = construct_context_branch_mask(weight, 0, blocks, int(random_key), "RANDOM_MATCHED")
    elif route_name == "FULL":
        initial = post = construct_context_branch_mask(weight, 0, blocks, 0, "FULL")
    else:
        raise ValueError(f"unknown route {route_name!r}")
    return initial, post, gate_info


def _route_trial(
    route_name: str,
    context_slot: int,
    left_payload: int,
    right_payload: int,
    source_snapshot: Any,
    books: dict[str, Any],
    task: dict[str, Any],
    gate_snapshot: GateSnapshot,
    shuffled_snapshot: GateSnapshot,
    config: LearnedContextGateConfig,
) -> dict[str, Any]:
    base = ContextBranchConfig(seed=int(config.seed))
    expected_branch = int(task["mapping"][context_slot])
    expected_payload = left_payload if expected_branch == 0 else right_payload
    opposite_payload = right_payload if expected_branch == 0 else left_payload
    sensory = books["S0"][left_payload] + books["S1"][right_payload]
    initial, post, gate_info = _route_masks(
        route_name,
        context_slot,
        source_snapshot,
        books,
        task,
        gate_snapshot,
        shuffled_snapshot,
        int(config.seed),
    )
    final, metrics = _rollout(source_snapshot, initial, post, sensory, base, books["blocks"])
    hidden_norms = metrics.pop("hidden_norms_at_arrival")
    decoded = _decode_y(final, books, expected_payload, opposite_payload, base)
    expected_hidden_ratio = hidden_norms[expected_branch] / (sum(hidden_norms) + EPSILON)
    return {
        "route": route_name,
        "context_slot": context_slot,
        "expected_branch": expected_branch,
        "left_payload": left_payload,
        "right_payload": right_payload,
        "mask_edges": int(post.sum().item()),
        "expected_hidden_ratio": float(expected_hidden_ratio),
        "hidden_norms_at_arrival": hidden_norms,
        **gate_info,
        **decoded,
        **metrics,
    }


def _evaluate_route(
    route_name: str,
    source_snapshot: Any,
    books: dict[str, Any],
    task: dict[str, Any],
    gate_snapshot: GateSnapshot,
    shuffled_snapshot: GateSnapshot,
    config: LearnedContextGateConfig,
) -> dict[str, Any]:
    rows = [
        _route_trial(
            route_name,
            context_slot,
            left,
            right,
            source_snapshot,
            books,
            task,
            gate_snapshot,
            shuffled_snapshot,
            config,
        )
        for context_slot in (0, 1)
        for left in range(ContextBranchConfig().payload_width)
        for right in range(ContextBranchConfig().payload_width)
        if left != right
    ]
    count = len(rows)
    return {
        "route": route_name,
        "accuracy": sum(int(row["success"]) for row in rows) / count,
        "opposite_delivery": sum(int(row["opposite_delivery"]) for row in rows) / count,
        "mean_margin": sum(float(row["margin"]) for row in rows) / count,
        "mean_y_norm": sum(float(row["y_norm"]) for row in rows) / count,
        "mean_runtime_energy_proxy": sum(float(row["runtime_energy_proxy"]) for row in rows) / count,
        "mean_active_fraction": sum(float(row["active_fraction"]) for row in rows) / count,
        "mean_expected_hidden_ratio": sum(float(row["expected_hidden_ratio"]) for row in rows) / count,
        "mask_edge_counts": sorted(set(int(row["mask_edges"]) for row in rows)),
        "hippocampal_rows_after": max(int(row["hippocampal_rows_after"]) for row in rows),
        "trials": rows,
    }


def run_learned_context_gate_seed(
    seed: int = 97601,
    *,
    config: LearnedContextGateConfig | None = None,
) -> dict[str, Any]:
    selected = config or LearnedContextGateConfig(seed=int(seed))
    config = LearnedContextGateConfig(**{**asdict(selected), "seed": int(seed)})
    base = ContextBranchConfig(seed=int(seed))
    source_snapshot, books, recurrent_receipt = _learn(int(seed), base)
    task = _context_task(int(seed), config.cue_dim)
    gate_snapshot, gate_receipt = _train_gate(
        source_snapshot,
        books,
        task,
        config,
        shuffle_context=False,
    )
    shuffled_snapshot, shuffled_receipt = _train_gate(
        source_snapshot,
        books,
        task,
        config,
        shuffle_context=True,
    )
    preflight = _preflight(
        source_snapshot,
        books,
        recurrent_receipt,
        task,
        gate_snapshot,
        gate_receipt,
        shuffled_snapshot,
        shuffled_receipt,
        config,
    )
    if not preflight["all_pass"]:
        return {
            "seed": int(seed),
            "status": "APPARATUS_INVALID",
            "endpoint_opened": False,
            "config": asdict(config),
            "branch_config": asdict(base),
            "preflight": preflight,
            "gate_learning": gate_receipt,
            "shuffled_gate_learning": shuffled_receipt,
        }

    gate_hash_before = _gate_hash(gate_snapshot)
    shuffled_hash_before = _gate_hash(shuffled_snapshot)
    runtime_hash_before = _snapshot_hash(source_snapshot)
    routes = {
        route_name: _evaluate_route(
            route_name,
            source_snapshot,
            books,
            task,
            gate_snapshot,
            shuffled_snapshot,
            config,
        )
        for route_name in ROUTES
    }
    frozen_after = bool(
        gate_hash_before == _gate_hash(gate_snapshot)
        and shuffled_hash_before == _gate_hash(shuffled_snapshot)
        and runtime_hash_before == _snapshot_hash(source_snapshot)
    )
    learned = routes["LEARNED"]["accuracy"]
    oracle = routes["ORACLE"]["accuracy"]
    exact_budget_controls = (
        "CONTEXT_SHUFFLE_TRAIN",
        "WRONG_CUE",
        "GATE_LESION_STATIC_0",
        "STATIC_1",
        "RANDOM_MATCHED",
    )
    strongest = max(routes[name]["accuracy"] for name in exact_budget_controls)
    swap_parity = bool(
        routes["POST_CUE_SWAP"]["accuracy"] == routes["WRONG_CUE"]["accuracy"]
        and routes["POST_CUE_SWAP"]["opposite_delivery"] == routes["WRONG_CUE"]["opposite_delivery"]
    )
    seed_pass = bool(
        learned >= 0.95
        and oracle >= 0.95
        and oracle - learned <= 0.05
        and routes["CONTEXT_SHUFFLE_TRAIN"]["accuracy"] <= 0.05
        and routes["CONTEXT_SHUFFLE_TRAIN"]["opposite_delivery"] >= 0.95
        and routes["WRONG_CUE"]["accuracy"] <= 0.05
        and routes["WRONG_CUE"]["opposite_delivery"] >= 0.95
        and routes["GATE_LESION_STATIC_0"]["accuracy"] <= 0.55
        and routes["STATIC_1"]["accuracy"] <= 0.55
        and routes["RANDOM_MATCHED"]["accuracy"] <= 0.55
        and routes["FULL"]["accuracy"] <= 0.55
        and learned - strongest >= 0.40
        and swap_parity
        and frozen_after
        and all(route["hippocampal_rows_after"] == 0 for route in routes.values())
    )
    return {
        "seed": int(seed),
        "status": "LEARNED_CONTEXT_GATE_PASS" if seed_pass else "LEARNED_CONTEXT_GATE_NOT_IDENTIFIED",
        "endpoint_opened": True,
        "config": asdict(config),
        "branch_config": asdict(base),
        "preflight": preflight,
        "gate_learning": gate_receipt,
        "shuffled_gate_learning": shuffled_receipt,
        "routes": routes,
        "strongest_exact_budget_nonoracle_control_accuracy": strongest,
        "learned_control_advantage": learned - strongest,
        "learned_oracle_gap": oracle - learned,
        "post_cue_swap_parity": swap_parity,
        "all_frozen_after_evaluation": frozen_after,
        "gate_sha256_before_evaluation": gate_hash_before,
        "gate_sha256_after_evaluation": _gate_hash(gate_snapshot),
        "runtime_snapshot_sha256_before_evaluation": runtime_hash_before,
        "runtime_snapshot_sha256_after_evaluation": _snapshot_hash(source_snapshot),
        "gate_compute_proxy_multiply_adds_per_selection": 2 * 2 * config.cue_dim,
    }
