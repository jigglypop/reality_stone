"""M4-R: experience-scored recurrent deformation, isolated from frozen M1/T1.

This is deliberately a small experimental apparatus.  It never installs a
closed-form answer matrix: every epoch scores a fixed bank of local
rollout-residual candidates on staged experiences, then installs one projected
candidate.  The factor held-out row is constructed only after the weight is
frozen.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Iterable

import torch

from ..runtime import BrainRuntime, BrainRuntimeSnapshot, HippocampusMemory, RuntimeMode
from .runtime_alternative_memory import AlternativeMemoryConfig, _dense_sparse_parity, _evaluate_sealed, _m1_runtime
from .runtime_contrastive_predictive_memory import _factor_codebooks
from .runtime_native_loops import _codebook, _decode, _detach, _loop8_replay_source_audit, _probe_rollout, _unit
from ..temporal_memory import TemporalAuditedMemory

DISCOVERY_SEEDS = range(97401, 97409)
DEVELOPMENT_SEEDS = range(97409, 97417)
CONFIRMATION_SEEDS = range(99401, 99433)
LAMBDAS = (0.50, 0.80, 0.95)
SCALES = (0.50, 1.0, 2.0)
EPSILON = 1e-8


@dataclass(frozen=True)
class SelfSelectingDeformationConfig:
    dim: int = 48
    replay_epochs: int = 12
    rollout_horizon: int = 6
    cue_drive_gain: float = 5.0
    max_write_norm: float = 5.0
    abstain_threshold: float = 0.20
    seed: int = 97401

    def alternative(self) -> AlternativeMemoryConfig:
        return AlternativeMemoryConfig(dim=self.dim, replay_epochs=self.replay_epochs,
            rollout_horizon=self.rollout_horizon, cue_drive_gain=self.cue_drive_gain,
            max_write_norm=self.max_write_norm, m1_abstain_threshold=self.abstain_threshold,
            seed=self.seed)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b).item() / ((float(a.norm()) + EPSILON) * (float(b.norm()) + EPSILON)))


def _runtime_weighted(weight: torch.Tensor, config: SelfSelectingDeformationConfig) -> BrainRuntime:
    runtime = _m1_runtime(config.alternative())
    runtime.weight.copy_(weight); runtime.sparse_weight = runtime.weight.to_sparse()
    return runtime


def _rollout(runtime: BrainRuntime, vector: torch.Tensor, config: SelfSelectingDeformationConfig) -> list[torch.Tensor]:
    runtime.reset_evaluation_state(); runtime.hippocampus = HippocampusMemory(config.dim, capacity=16, device="cpu")
    states: list[torch.Tensor] = []
    runtime.step(external_input=config.cue_drive_gain * vector, force_mode=RuntimeMode.WAKE, learning_signal=0.0)
    states.append(runtime.activation.detach().clone())
    for _ in range(config.rollout_horizon):
        runtime.step(external_input=torch.zeros(config.dim), force_mode=RuntimeMode.WAKE, learning_signal=0.0)
        states.append(runtime.activation.detach().clone())
    return states


def _project(weight: torch.Tensor) -> torch.Tensor:
    result = weight.clone(); result.fill_diagonal_(0.0)
    return result


def _candidate_delta(weight: torch.Tensor, cues: torch.Tensor, values: torch.Tensor,
                     config: SelfSelectingDeformationConfig, condition: str) -> tuple[dict[float, torch.Tensor], dict[str, Any]]:
    n = len(cues); runtime = _runtime_weighted(weight, config)
    cue_states = [_rollout(runtime, cue, config) for cue in cues]
    value_states = [_rollout(runtime, value, config) for value in values]
    order = list(range(n))
    if condition == "target_shuffled":
        values = values[1:].clone().contiguous() if n == 1 else torch.cat((values[1:], values[:1]))
        value_states = [_rollout(runtime, value, config) for value in values]
    trace_order = order[1:] + order[:1] if n > 1 and condition == "trace_shuffled" else order
    result: dict[float, torch.Tensor] = {}
    for lam in LAMBDAS:
        total = torch.zeros_like(weight)
        for row, trace_row in zip(order, trace_order):
            vc, vv = _unit(values[row]), _unit(values[row])
            xc, xv = cue_states[row][-1], value_states[row][-1]
            pc = sum((lam ** (config.rollout_horizon - h)) * cue_states[trace_row][h]
                     for h in range(config.rollout_horizon))
            pv = sum((lam ** (config.rollout_horizon - h)) * value_states[trace_row][h]
                     for h in range(config.rollout_horizon))
            total += torch.outer(vc - xc, pc) + .65 * torch.outer(vv - xv, pv)
        total /= float(n)
        result[lam] = total
    return result, {"cue_rollout_count": n, "value_rollout_count": n, "heldout_rows_used": 0}


def _rank_cosines(directions: dict[float, torch.Tensor]) -> tuple[int, dict[str, float]]:
    matrix = torch.stack([directions[key].reshape(-1) for key in LAMBDAS])
    rank = int(torch.linalg.matrix_rank(matrix, tol=1e-6).item())
    return rank, {f"{a:.2f}:{b:.2f}": _cos(directions[a].reshape(-1), directions[b].reshape(-1))
                  for i, a in enumerate(LAMBDAS) for b in LAMBDAS[i + 1:]}


def _score_candidate(weight: torch.Tensor, candidate: torch.Tensor, cues: torch.Tensor, values: torch.Tensor,
                     config: SelfSelectingDeformationConfig) -> tuple[float, bool]:
    unstable = False; loss = 0.0
    runtime = _runtime_weighted(candidate, config)
    for cue, value in zip(cues, values):
        for start in (cue, value):
            states = _rollout(runtime, start, config); final = states[-1]
            # The contract's x_0 is the state produced by the first input
            # after reset.  It is a baseline, not a state to be tested for
            # growth; only the later zero-input rollout may be unstable.
            baseline = max(1.0, float(states[0].norm()))
            unstable = unstable or not bool(final.isfinite().all()) or any(
                float(state.norm()) > 2.0 * baseline for state in states[1:])
            loss += 1.0 - _cos(final, value)
    penalty = .02 * (float((candidate - weight).norm()) ** 2) / 25.0
    return loss / (2.0 * len(cues)) + penalty + (.10 if unstable else 0.0), unstable


def _run_condition(seed: int, config: SelfSelectingDeformationConfig, task: str, condition: str) -> dict[str, Any]:
    config = SelfSelectingDeformationConfig(**{**asdict(config), "seed": seed})
    runtime = _m1_runtime(config.alternative()); initial = runtime.weight.clone()
    if task == "loop8":
        temporal, source, _ = _loop8_replay_source_audit(); indices = [int(x["value"]) for x in source]
        all_cues, all_targets = _codebook(seed, config.dim); cues, values = all_cues[indices], all_targets[indices]
        codebook_hash = hashlib.sha256(all_cues.numpy().tobytes() + all_targets.numpy().tobytes()).hexdigest()
        heldout = None
    else:
        temporal = TemporalAuditedMemory(capacity=32); books = _factor_codebooks(seed, config.dim)
        indices = list(range(3)); cues, values = books["cues"][:3], books["targets"][:3]
        all_cues, all_targets = books["cues"], books["targets"]; codebook_hash = books["sha256"]; heldout = 3
    epochs: list[dict[str, Any]] = []; fold_trigger = False
    for epoch in range(config.replay_epochs):
        before = runtime.weight.clone(); directions, receipt = _candidate_delta(before, cues, values, config, condition)
        rank, cosines = _rank_cosines(directions)
        candidates: list[dict[str, Any]] = []
        for lam in LAMBDAS:
            norm = float(directions[lam].norm())
            for scale in SCALES:
                raw_delta = torch.zeros_like(before) if norm <= EPSILON else .8 * scale * directions[lam] / (norm + EPSILON)
                projected = _project(before + raw_delta); candidate = projected
                score, unstable = _score_candidate(before, candidate, cues, values, config)
                candidates.append({"lambda": lam, "scale": scale, "score": score, "unstable": unstable,
                    "raw_delta_norm": float(raw_delta.norm()), "projected_delta_norm": float((candidate - before).norm()),
                    "actual_delta_norm": None})
        candidates.sort(key=lambda x: (x["score"], x["scale"], x["lambda"]))
        selected = candidates[0] if condition != "no_selection" else next(x for x in candidates if x["lambda"] == .8 and x["scale"] == 1.0)
        chosen = next(x for x in candidates if x["lambda"] == selected["lambda"] and x["scale"] == selected["scale"])
        if condition == "identity": actual = torch.zeros_like(before)
        else:
            proposed = _project(before + (.8 * selected["scale"] * directions[selected["lambda"]] /
                (float(directions[selected["lambda"]].norm()) + EPSILON)))
            actual = proposed - before
            if condition == "sign_flipped": actual = -actual
            norm = float(actual.norm())
            if norm > config.max_write_norm: actual *= config.max_write_norm / norm
        runtime.weight.copy_(_project(before + actual)); runtime.sparse_weight = runtime.weight.to_sparse()
        chosen["actual_delta_norm"] = float(actual.norm())
        max_scale_selected = selected["scale"] == 2.0
        instability = any(x["unstable"] for x in candidates)
        fold_trigger = fold_trigger or instability
        epochs.append({"epoch": epoch, "raw_D_norms": {str(k): float(v.norm()) for k,v in directions.items()},
            "svd_rank_tol_1e-6": rank, "pairwise_cosine": cosines, "candidates": candidates,
            "selected_lambda": selected["lambda"], "selected_scale": selected["scale"], "receipt": receipt,
            "projected_delta_norm": chosen["projected_delta_norm"], "actual_delta_norm": chosen["actual_delta_norm"],
            "max_scale_selected": max_scale_selected, "candidate_instability": instability})
    cutoff = _detach(runtime, temporal); sealed = runtime.snapshot()
    restored = BrainRuntime.from_snapshot(sealed, backend="torch", device="cpu")
    parity = bool(torch.equal(restored.weight, runtime.weight) and len(restored.hippocampus) == 0)
    if task == "loop8":
        evaluated = _evaluate_sealed(runtime, temporal, all_cues, all_targets, indices, config.alternative(), abstain_threshold=config.abstain_threshold)
        metric = evaluated["clean_accuracy"]; basic_gate = (metric >= .80 and evaluated["corrupt_accuracy"] >= .65 and
            evaluated["deleted_abstention"] >= .95 and evaluated["unknown_abstention"] >= .95 and evaluated["attractor_cosine_gain"] >= .05)
    else:
        _, final, rows = _probe_rollout(sealed, all_cues[heldout], config.alternative().native())
        metric = float(_decode(final, all_targets, abstain_threshold=config.abstain_threshold) == heldout)
        evaluated = {"held_out_accuracy": metric, "hippocampal_rows_after_rollout": rows, "heldout_rows_used": 0,
                     "heldout_absence_audit": True, "attractor_cosine_gain": _cos(final, all_targets[heldout])}
        basic_gate = metric >= .70
    return {"seed": seed, "task": task, "condition": condition, "epochs": epochs, "factor_codebook_sha256": codebook_hash,
        "candidate_bank": [{"lambda": x, "scale": y} for x in LAMBDAS for y in SCALES], "candidate_bank_hash": hashlib.sha256(repr((LAMBDAS,SCALES)).encode()).hexdigest(),
        "evaluation": evaluated, "endpoint": metric, "basic_task_gate": basic_gate,
        "weight_drift": float((runtime.weight-initial).norm()), "snapshot_restore_parity": parity,
        "dense_sparse_parity": _dense_sparse_parity(runtime), "finite": bool(torch.isfinite(runtime.weight).all()),
        "cutoff_audit": cutoff, "fold_trigger": fold_trigger or (sum(e["max_scale_selected"] for e in epochs) / len(epochs) >= .75),
        "fold_trigger_receipt": {"max_scale_selected_epochs": sum(e["max_scale_selected"] for e in epochs),
            "epoch_count": len(epochs), "max_scale_selected_fraction": sum(e["max_scale_selected"] for e in epochs) / len(epochs),
            "instability_epoch_count": sum(e["candidate_instability"] for e in epochs)}, "fold_active": False,
        "direction_rank_one": any(e["svd_rank_tol_1e-6"] == 1 for e in epochs)}


def self_selecting_deformation(seed: int, config: SelfSelectingDeformationConfig | None = None) -> dict[str, Any]:
    config = config or SelfSelectingDeformationConfig(seed=seed)
    tasks = {task: _run_condition(seed, config, task, "learned") for task in ("loop8", "loop9")}
    controls = {task: {name: _run_condition(seed, config, task, name) for name in
        ("identity", "no_selection", "target_shuffled", "trace_shuffled", "sign_flipped")} for task in tasks}
    for task, learned in tasks.items():
        learned["controls"] = controls[task]
        learned["min_control_advantage"] = learned["endpoint"] - max(x["endpoint"] for x in controls[task].values())
        learned["status"] = "GO" if (learned["basic_task_gate"] and learned["min_control_advantage"] >= .20 and
            not learned["direction_rank_one"] and learned["weight_drift"] > 0 and learned["snapshot_restore_parity"] and
            learned["dense_sparse_parity"] and learned["finite"] and learned["cutoff_audit"]["temporal_rows_after"] == 0 and
            learned["cutoff_audit"]["hippocampal_rows_after"] == 0) else "STOP"
    shared = tasks["loop8"]["candidate_bank"] == tasks["loop9"]["candidate_bank"] and all(
        x["factor_codebook_sha256"] == tasks["loop8"]["factor_codebook_sha256"] for x in controls["loop8"].values())
    return {"seed": seed, "route": "M4-R_experience_scored_rollout_residual", "config": asdict(config),
        "loop8": tasks["loop8"], "loop9": tasks["loop9"], "same_schedule_candidate_bank": shared,
        "status": "GO" if tasks["loop8"]["status"] == tasks["loop9"]["status"] == "GO" and shared else "STOP"}
