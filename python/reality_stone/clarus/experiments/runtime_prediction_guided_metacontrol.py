"""Frozen C1 predictor-to-policy intervention for :class:`BrainRuntime`.

The route is deliberately narrow.  A predictor is fitted on independent
one-step transitions, frozen, and queried algebraically for three candidate
actions.  Policy episodes execute only the selected action.  The module is a
simulator experiment, not a biological or consciousness model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch

from ..runtime import BrainRuntime, BrainRuntimeConfig, BrainRuntimeSnapshot, RuntimeMode


DEVELOPMENT_SEEDS = range(97901, 97917)
CONFIRMATION_SEEDS = range(99901, 99933)
ACTION_VALUES = (-1, 0, 1)
ADVERSE_ARMS = (
    "edge_shuffle",
    "persistence",
    "random_balanced",
    "error_magnitude_only",
    "reactive_mean_effect",
)
ALL_ARMS = ("intact", "readout_shuffle", *ADVERSE_ARMS)
EDGE_DERANGEMENT = (1, 2, 0)
RESULT_SCHEMA = "clarus.runtime_prediction_guided_metacontrol.c1.seed.v1"
ARTIFACT_SCHEMA = "clarus.runtime_prediction_guided_metacontrol.c1.v1"
ROUTE = "C1_prediction_guided_metacontrol"
CLAIM_BOUNDARY = "SIMULATOR_PREDICTOR_TO_POLICY_ONLY"
PREDECESSOR_ARTIFACT_SHA256 = (
    "2fd40c7e32f2ed8b143701bc517393b7df279d36a293483a6846279863726633"
)

_REPOSITORY = Path(__file__).resolve().parents[5]
_PREDECESSOR_ARTIFACT = _REPOSITORY / (
    "_workspace/ce/brainruntime-native-all-loops-p1-20260819/"
    "artifacts/confirmation-results.json"
)
_FREEZE_FILES = (
    Path("reality_stone/python/reality_stone/clarus/experiments/runtime_prediction_guided_metacontrol.py"),
    Path("reality_stone/python/reality_stone/clarus/experiments/runtime_prediction_guided_metacontrol_benchmark.py"),
    Path("reality_stone/python/reality_stone/clarus/runtime.py"),
    Path("tests/test_runtime_prediction_guided_metacontrol.py"),
    Path("_workspace/ce/brainruntime-prediction-guided-metacontrol-20260820/00-contract.md"),
)


@dataclass(frozen=True)
class C1MetacontrolConfig:
    dim: int = 48
    fit_states: int = 128
    audit_states: int = 48
    policy_states: int = 64
    warmup_ticks: int = 4
    warmup_norm: float = 0.35
    context_norm: float = 0.50
    correction_gain: float = 0.75
    goal_shift: float = 0.25
    ridge: float = 1e-4
    summary_threshold: float = 0.10
    scale_floor: float = 1e-8
    prediction_ratio_threshold: float = 0.90
    advantage_threshold: float = 0.05
    edge_change_threshold: float = 0.20
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 97_998
    action_seed: int = 97_900
    goal_bank_seed: int = 97_901
    warmup_seed_offset: int = 170_000
    context_seed_offset: int = 180_000
    active_ratio: float = 0.25
    seed: int = 97_901

    def native(self) -> BrainRuntimeConfig:
        return BrainRuntimeConfig(
            dim=self.dim,
            active_ratio=self.active_ratio,
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=False,
            f1_self_measure=False,
            stdp_enabled=False,
            hippocampal_encoding_enabled=False,
            replay_gain=0.0,
        )


_FROZEN_PARAMETERS = {
    key: value
    for key, value in asdict(C1MetacontrolConfig()).items()
    if key != "seed"
}


@dataclass(frozen=True)
class _Fixture:
    base_snapshot: BrainRuntimeSnapshot
    snapshots: tuple[BrainRuntimeSnapshot, ...]
    contexts: torch.Tensor
    warmups: torch.Tensor
    snapshot_ids: tuple[str, ...]
    snapshot_hashes: tuple[str, ...]
    audit: dict[str, Any]


@dataclass(frozen=True)
class _Predictor:
    theta: torch.Tensor
    target_mean: torch.Tensor
    target_scale: torch.Tensor
    mean_effect: torch.Tensor
    alarm_median: float
    audit: dict[str, Any]


@dataclass
class _StepLedger:
    """Observed calls through the only C1 runtime-transition boundary."""

    selection_calls: int = 0
    actual_calls: int = 0

    def record(self, phase: str) -> None:
        if phase == "selection":
            self.selection_calls += 1
        elif phase == "actual":
            self.actual_calls += 1
        else:
            raise ValueError("C1 step phase must be selection or actual")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def c1_source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _FREEZE_FILES:
        path = _REPOSITORY / relative
        if not path.is_file():
            raise RuntimeError(f"C1 freeze file is missing: {relative.as_posix()}")
        result[relative.as_posix()] = _file_sha256(path)
    return result


def _current_environment() -> dict[str, str]:
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_device": "cpu",
    }


def _update_json(digest: Any, label: str, value: Any) -> None:
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )
    digest.update(b"\0")


def _update_tensor(digest: Any, label: str, tensor: torch.Tensor | None) -> None:
    digest.update(label.encode("utf-8") + b"\0")
    if tensor is None:
        digest.update(b"NONE\0")
        return
    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
    digest.update(value.numpy().tobytes(order="C"))
    digest.update(b"\0")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    _update_tensor(digest, "tensor", tensor)
    return digest.hexdigest()


def _snapshot_sha256(snapshot: BrainRuntimeSnapshot) -> str:
    digest = hashlib.sha256()
    _update_json(digest, "config", asdict(snapshot.config))
    for name in (
        "weight",
        "activation",
        "refractory",
        "memory_trace",
        "adaptation",
        "stp_u",
        "stp_x",
        "bitfield",
        "goal",
        "lifecycle",
        "inactive_steps",
    ):
        _update_tensor(digest, name, getattr(snapshot, name))
    _update_json(digest, "mode", snapshot.mode.value)
    for name in (
        "sleep_pressure",
        "arousal",
        "step",
        "active_ratio_ema",
        "stdp_prev_critic_score",
        "stdp_updates",
        "circadian_phase",
        "circadian_value",
        "nrem_cycle_count",
        "delay_idx",
        "last_stdp_gate",
        "stdp_pending_learning_signal",
    ):
        _update_json(digest, name, getattr(snapshot, name))
    _update_json(digest, "mode_occupancy", dict(sorted(snapshot.mode_occupancy.items())))
    _update_json(digest, "brainwave_history", list(snapshot.brainwave_history))
    hippocampus = snapshot.hippocampus
    _update_json(digest, "hippocampus_dim", hippocampus.get("dim"))
    _update_json(digest, "hippocampus_capacity", hippocampus.get("capacity"))
    _update_tensor(digest, "hippocampus_keys", hippocampus.get("keys"))
    _update_tensor(digest, "hippocampus_values", hippocampus.get("values"))
    _update_json(digest, "hippocampus_priority", list(hippocampus.get("priority", [])))
    if snapshot.stdp_tracker is None:
        _update_json(digest, "stdp_tracker", None)
    else:
        _update_json(digest, "stdp_tracker", "present")
        for name in ("pre_trace", "post_trace", "eligibility"):
            _update_tensor(digest, f"stdp_tracker_{name}", snapshot.stdp_tracker.get(name))
    _update_tensor(digest, "delay_buffer", snapshot.delay_buffer)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _unit_rows(table: torch.Tensor, norm: float) -> torch.Tensor:
    flat = table.reshape(-1, table.shape[-1])
    lengths = flat.double().norm(dim=1)
    if not torch.isfinite(lengths).all() or bool((lengths <= 0.0).any()):
        raise RuntimeError("C1 random tape contains a zero or nonfinite row")
    scaled = flat.double() * (float(norm) / lengths).unsqueeze(1)
    return scaled.reshape(table.shape).float()


def _action_vector(dim: int, seed: int = 97_900) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    bits = torch.randint(0, 2, (dim,), generator=generator)
    vector = bits.double().mul(2.0).sub(1.0)
    return vector / vector.norm()


def _goal_bank(seed: int = 97_901) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    raw = torch.randn(16, 4, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(raw, mode="reduced")
    diagonal = torch.diagonal(r)
    signs = torch.where(diagonal < 0.0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    signed = q * signs.unsqueeze(0)
    lengths = signed.norm(dim=1)
    if not torch.isfinite(lengths).all() or bool((lengths <= 0.0).any()):
        raise RuntimeError("C1 goal bank contains a zero or nonfinite row")
    bank = signed / lengths.unsqueeze(1)
    if float((bank.norm(dim=1) - 1.0).abs().max()) > 1e-12:
        raise RuntimeError("C1 goal bank unit-norm invariant failed")
    return signed.contiguous(), bank.contiguous()


def _global_apparatus(config: C1MetacontrolConfig) -> dict[str, Any]:
    action = _action_vector(config.dim, config.action_seed)
    goal_pre, goals = _goal_bank(config.goal_bank_seed)
    random_schedule = tuple([-1, 0, 1] * 21 + [-1])
    sign_schedule = tuple([-1, 1] * 32)
    return {
        "action_vector": action,
        "goal_pre": goal_pre,
        "goals": goals,
        "random_schedule": random_schedule,
        "sign_schedule": sign_schedule,
        "hashes": {
            "action_vector_sha256": _tensor_sha256(action),
            "goal_pre_sha256": _tensor_sha256(goal_pre),
            "goal_bank_sha256": _tensor_sha256(goals),
            "random_schedule_sha256": _canonical_sha256(random_schedule),
            "sign_schedule_sha256": _canonical_sha256(sign_schedule),
            "edge_derangement_sha256": _canonical_sha256(EDGE_DERANGEMENT),
        },
    }


def _c1_runtime(seed: int, config: C1MetacontrolConfig) -> BrainRuntime:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    weight = torch.randn(
        config.dim, config.dim, generator=generator, dtype=torch.float32,
    ) * 0.025
    weight.fill_diagonal_(0.0)
    return BrainRuntime(
        weight,
        config=config.native(),
        backend="torch",
        device="cpu",
    )


def _dense_sparse_parity(runtime: BrainRuntime) -> bool:
    return bool(torch.equal(runtime.weight, runtime.sparse_weight.to_dense()))


def _split_ids(config: C1MetacontrolConfig) -> tuple[str, ...]:
    return tuple(
        [f"fit-{index:03d}" for index in range(config.fit_states)]
        + [f"audit-{index:03d}" for index in range(config.audit_states)]
        + [f"policy-{index:03d}" for index in range(config.policy_states)]
    )


def _build_fixture(seed: int, config: C1MetacontrolConfig) -> _Fixture:
    runtime = _c1_runtime(seed, config)
    base = runtime.snapshot()
    total = config.fit_states + config.audit_states + config.policy_states
    warmup_generator = torch.Generator(device="cpu").manual_seed(
        int(seed + config.warmup_seed_offset)
    )
    context_generator = torch.Generator(device="cpu").manual_seed(
        int(seed + config.context_seed_offset)
    )
    warmups = _unit_rows(
        torch.randn(
            total, config.warmup_ticks, config.dim,
            generator=warmup_generator, dtype=torch.float32,
        ),
        config.warmup_norm,
    )
    contexts = _unit_rows(
        torch.randn(total, config.dim, generator=context_generator, dtype=torch.float32),
        config.context_norm,
    )
    ids = _split_ids(config)
    snapshots: list[BrainRuntimeSnapshot] = []
    hashes: list[str] = []
    transition_count = 0
    for index in range(total):
        trial = BrainRuntime.from_snapshot(base, backend="torch", device="cpu")
        for tick in range(config.warmup_ticks):
            trial.step(
                external_input=warmups[index, tick],
                force_mode=RuntimeMode.WAKE,
                learning_signal=0.0,
            )
            transition_count += 1
        snapshot = trial.snapshot()
        snapshots.append(snapshot)
        hashes.append(_snapshot_sha256(snapshot))
    base_weight_hash = _tensor_sha256(base.weight)
    sparse_hash = _tensor_sha256(runtime.sparse_weight.to_dense())
    row_associations = [
        {
            "id": ids[index],
            "warmup_sha256": _tensor_sha256(warmups[index]),
            "context_sha256": _tensor_sha256(contexts[index]),
            "snapshot_sha256": hashes[index],
        }
        for index in range(total)
    ]
    audit = {
        "total_states": total,
        "fit_states": config.fit_states,
        "audit_states": config.audit_states,
        "policy_states": config.policy_states,
        "warmup_ticks_per_state": config.warmup_ticks,
        "warmup_transition_count": transition_count,
        "warmup_table_sha256": _tensor_sha256(warmups),
        "context_table_sha256": _tensor_sha256(contexts),
        "ordered_ids_sha256": _canonical_sha256(ids),
        "associations_sha256": _canonical_sha256(row_associations),
        "snapshot_hashes_sha256": _canonical_sha256(hashes),
        "split_ids_disjoint": len(set(ids)) == len(ids),
        "base_weight_sha256": base_weight_hash,
        "base_sparse_weight_sha256": sparse_hash,
        "base_dense_sparse_parity": _dense_sparse_parity(runtime),
        "base_hippocampal_rows": len(runtime.hippocampus),
        "base_temporal_store_present": False,
        "base_temporal_rows": 0,
        "temporal_zero_provenance": "BrainRuntime_has_no_temporal_store_member",
        "automatic_stdp_updates": int(runtime._stdp_updates),
    }
    return _Fixture(
        base_snapshot=base,
        snapshots=tuple(snapshots),
        contexts=contexts,
        warmups=warmups,
        snapshot_ids=ids,
        snapshot_hashes=tuple(hashes),
        audit=audit,
    )


def _summary_from_activation(activation: torch.Tensor, threshold: float) -> torch.Tensor:
    value = activation.detach().cpu().double()
    return torch.stack(
        (
            value.mean(),
            value.norm(),
            value.abs().max(),
            (value.abs() > float(threshold)).double().mean(),
        )
    )


def _snapshot_summary(snapshot: BrainRuntimeSnapshot, config: C1MetacontrolConfig) -> torch.Tensor:
    return _summary_from_activation(snapshot.activation, config.summary_threshold)


def _drive(
    context: torch.Tensor,
    action_index: int,
    action_vector: torch.Tensor,
    config: C1MetacontrolConfig,
) -> torch.Tensor:
    action = ACTION_VALUES[int(action_index)]
    return (context.double() + config.correction_gain * action * action_vector).float()


def _feature(
    snapshot: BrainRuntimeSnapshot,
    drive: torch.Tensor,
    action_index: int,
) -> torch.Tensor:
    one_hot = torch.zeros(3, dtype=torch.float64)
    one_hot[int(action_index)] = 1.0
    feature = torch.cat(
        (
            snapshot.activation.double(),
            snapshot.refractory.double(),
            snapshot.memory_trace.double(),
            snapshot.adaptation.double(),
            snapshot.stp_u.double(),
            snapshot.stp_x.double(),
            snapshot.lifecycle.double(),
            drive.double(),
            one_hot,
            torch.ones(1, dtype=torch.float64),
        )
    )
    return feature


def _transition_integrity(
    runtime: BrainRuntime,
    *,
    before_weight_hash: str,
    before_step: int,
) -> dict[str, Any]:
    state_tensors = (
        runtime.activation,
        runtime.refractory,
        runtime.memory_trace,
        runtime.adaptation,
        runtime.stp_u,
        runtime.stp_x,
    )
    return {
        "step_before": before_step,
        "step_after": int(runtime.step_index),
        "step_delta": int(runtime.step_index - before_step),
        "one_actual_step": runtime.step_index - before_step == 1,
        "weight_unchanged": _tensor_sha256(runtime.weight) == before_weight_hash,
        "weight_sha256": _tensor_sha256(runtime.weight),
        "dense_sparse_parity": _dense_sparse_parity(runtime),
        "stdp_updates": int(runtime._stdp_updates),
        "hippocampal_rows": len(runtime.hippocampus),
        "temporal_store_present": False,
        "temporal_rows": 0,
        "temporal_zero_provenance": "BrainRuntime_has_no_temporal_store_member",
        "state_finite": all(bool(torch.isfinite(value).all()) for value in state_tensors),
    }


def _one_transition(
    snapshot: BrainRuntimeSnapshot,
    drive: torch.Tensor,
    config: C1MetacontrolConfig,
    *,
    ledger: _StepLedger | None = None,
    phase: str = "actual",
) -> tuple[torch.Tensor, dict[str, Any]]:
    runtime = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    before_step = runtime.step_index
    before_weight_hash = _tensor_sha256(runtime.weight)
    if ledger is not None:
        ledger.record(phase)
    runtime.step(
        external_input=drive,
        force_mode=RuntimeMode.WAKE,
        learning_signal=0.0,
    )
    audit = _transition_integrity(
        runtime,
        before_weight_hash=before_weight_hash,
        before_step=before_step,
    )
    audit["starting_snapshot_sha256"] = _snapshot_sha256(snapshot)
    audit["ending_snapshot_sha256"] = _snapshot_sha256(runtime.snapshot())
    audit["actual_drive_sha256"] = _tensor_sha256(drive)
    return _summary_from_activation(runtime.activation, config.summary_threshold), audit


def _fit_predictor(
    fixture: _Fixture,
    apparatus: Mapping[str, Any],
    config: C1MetacontrolConfig,
) -> _Predictor:
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    transition_integrity: list[bool] = []
    action_vector = apparatus["action_vector"]
    for index in range(config.fit_states):
        snapshot = fixture.snapshots[index]
        context = fixture.contexts[index]
        for action_index in range(3):
            drive = _drive(context, action_index, action_vector, config)
            features.append(_feature(snapshot, drive, action_index))
            target, audit = _one_transition(snapshot, drive, config)
            targets.append(target)
            transition_integrity.append(_transition_audit_ok(audit))
    x = torch.stack(features).double()
    y = torch.stack(targets).double()
    identity = torch.eye(x.shape[1], dtype=torch.float64)
    theta = torch.linalg.solve(x.T @ x + config.ridge * identity, x.T @ y)
    target_mean = y.mean(dim=0)
    target_scale = y.std(dim=0, correction=0).clamp_min(config.scale_floor)

    predicted = x @ theta
    row_actions = torch.tensor(
        [index for _ in range(config.fit_states) for index in range(3)], dtype=torch.int64,
    )
    current = torch.stack(
        [
            _snapshot_summary(fixture.snapshots[index], config)
            for index in range(config.fit_states)
            for _ in range(3)
        ]
    )
    standardized_prediction = (predicted - target_mean) / target_scale
    standardized_current = (current - target_mean) / target_scale
    mean_effect = torch.stack(
        [
            (standardized_prediction[row_actions == action_index]
             - standardized_current[row_actions == action_index]).mean(dim=0)
            for action_index in range(3)
        ]
    )

    goals = apparatus["goals"]
    zero_costs: list[float] = []
    for index in range(config.fit_states):
        row = index * 3 + 1
        goal = standardized_current[row] + config.goal_shift * goals[index % 16]
        cost = (standardized_prediction[row] - goal).square().sum()
        zero_costs.append(float(cost))
    sorted_costs = sorted(zero_costs)
    middle = len(sorted_costs) // 2
    alarm_median = (
        float(sorted_costs[middle])
        if len(sorted_costs) % 2
        else 0.5 * float(sorted_costs[middle - 1] + sorted_costs[middle])
    )
    schema = (
        "activation|refractory|memory_trace|adaptation|stp_u|stp_x|"
        "lifecycle|exact_drive|action_onehot_3|bias -> raw_activation_summary_4"
    )
    audit = {
        "fit_rows": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "output_dim": int(y.shape[1]),
        "ridge": config.ridge,
        "theta_sha256": _tensor_sha256(theta),
        "target_mean_sha256": _tensor_sha256(target_mean),
        "target_scale_sha256": _tensor_sha256(target_scale),
        "mean_effect_sha256": _tensor_sha256(mean_effect),
        "alarm_median": alarm_median,
        "alarm_costs_sha256": _canonical_sha256(zero_costs),
        "feature_schema": schema,
        "feature_schema_sha256": hashlib.sha256(schema.encode("utf-8")).hexdigest(),
        "fit_transition_integrity": all(transition_integrity),
        "parameters_finite": bool(
            torch.isfinite(theta).all()
            and torch.isfinite(target_mean).all()
            and torch.isfinite(target_scale).all()
            and torch.isfinite(mean_effect).all()
            and math.isfinite(alarm_median)
        ),
        "standardizer_correction": 0,
        "standardizer_floor": config.scale_floor,
        "model_frozen_before_policy": True,
    }
    return _Predictor(
        theta=theta.detach().clone(),
        target_mean=target_mean.detach().clone(),
        target_scale=target_scale.detach().clone(),
        mean_effect=mean_effect.detach().clone(),
        alarm_median=alarm_median,
        audit=audit,
    )


def _transition_audit_ok(audit: Mapping[str, Any]) -> bool:
    return bool(
        audit.get("one_actual_step")
        and audit.get("weight_unchanged")
        and audit.get("dense_sparse_parity")
        and audit.get("stdp_updates") == 0
        and audit.get("hippocampal_rows") == 0
        and audit.get("temporal_rows") == 0
        and audit.get("state_finite")
    )


def _predict_three(
    snapshot: BrainRuntimeSnapshot,
    context: torch.Tensor,
    action_vector: torch.Tensor,
    predictor: _Predictor,
    config: C1MetacontrolConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    drives = torch.stack(
        [_drive(context, index, action_vector, config) for index in range(3)]
    )
    features = torch.stack(
        [_feature(snapshot, drives[index], index) for index in range(3)]
    )
    return features @ predictor.theta, drives


def _standardize(value: torch.Tensor, predictor: _Predictor) -> torch.Tensor:
    return (value.double() - predictor.target_mean) / predictor.target_scale


def _predictor_audit(
    fixture: _Fixture,
    apparatus: Mapping[str, Any],
    predictor: _Predictor,
    config: C1MetacontrolConfig,
) -> dict[str, Any]:
    start = config.fit_states
    predictor_sse = 0.0
    persistence_sse = 0.0
    rows = 0
    integrity: list[bool] = []
    for offset in range(config.audit_states):
        index = start + offset
        snapshot = fixture.snapshots[index]
        context = fixture.contexts[index]
        predictions, drives = _predict_three(
            snapshot, context, apparatus["action_vector"], predictor, config,
        )
        current = _snapshot_summary(snapshot, config)
        for action_index in range(3):
            actual, transition_audit = _one_transition(
                snapshot, drives[action_index], config,
            )
            predictor_sse += float((predictions[action_index] - actual).square().sum())
            persistence_sse += float((current - actual).square().sum())
            rows += 1
            integrity.append(_transition_audit_ok(transition_audit))
    predictor_mse = predictor_sse / (rows * 4.0)
    persistence_mse = persistence_sse / (rows * 4.0)
    ratio = (
        predictor_mse / persistence_mse
        if math.isfinite(persistence_mse) and persistence_mse > 0.0
        else float("inf")
    )
    return {
        "rows": rows,
        "predictor_mse_raw_summary": predictor_mse,
        "persistence_mse_raw_summary": persistence_mse,
        "mse_ratio": ratio,
        "denominator_finite_positive": math.isfinite(persistence_mse) and persistence_mse > 0.0,
        "transition_integrity": all(integrity),
        "fit_rows_reused": False,
        "policy_rows_used": False,
    }


def _lowest_index_argmin(costs: torch.Tensor) -> int:
    if costs.shape != (3,) or not bool(torch.isfinite(costs).all()):
        raise ValueError("planner costs must be three finite values")
    return int(torch.argmin(costs).item())


def _policy_actions(
    predictions: torch.Tensor,
    current_z: torch.Tensor,
    goal: torch.Tensor,
    predictor: _Predictor,
    episode_index: int,
    apparatus: Mapping[str, Any],
) -> tuple[dict[str, int], torch.Tensor, torch.Tensor]:
    predicted_z = _standardize(predictions, predictor)
    true_costs = (predicted_z - goal.unsqueeze(0)).square().sum(dim=1)
    intact = _lowest_index_argmin(true_costs)
    edge_costs = true_costs[list(EDGE_DERANGEMENT)]
    edge = _lowest_index_argmin(edge_costs)
    zero_cost = float(true_costs[1])
    if zero_cost <= predictor.alarm_median:
        magnitude = 1
    else:
        sign = apparatus["sign_schedule"][episode_index]
        magnitude = 0 if sign == -1 else 2
    reactive_predictions = current_z.unsqueeze(0) + predictor.mean_effect
    reactive_costs = (reactive_predictions - goal.unsqueeze(0)).square().sum(dim=1)
    actions = {
        "intact": intact,
        "readout_shuffle": intact,
        "edge_shuffle": edge,
        "persistence": 1,
        "random_balanced": ACTION_VALUES.index(apparatus["random_schedule"][episode_index]),
        "error_magnitude_only": magnitude,
        "reactive_mean_effect": _lowest_index_argmin(reactive_costs),
    }
    return actions, true_costs, reactive_costs


def _evaluate_policy(
    fixture: _Fixture,
    apparatus: Mapping[str, Any],
    predictor: _Predictor,
    config: C1MetacontrolConfig,
) -> dict[str, Any]:
    start = config.fit_states + config.audit_states
    goals = apparatus["goals"]
    losses = {arm: [] for arm in ALL_ARMS}
    actions = {arm: [] for arm in ALL_ARMS}
    traces: list[dict[str, Any]] = []
    integrity_rows: list[bool] = []
    readout_equivalence = True
    edge_port_identity = True
    snapshot_identity = True
    selection_step_calls = 0
    actual_step_calls = 0
    for episode in range(config.policy_states):
        index = start + episode
        snapshot = fixture.snapshots[index]
        context = fixture.contexts[index]
        predictions, drives = _predict_three(
            snapshot, context, apparatus["action_vector"], predictor, config,
        )
        current_z = _standardize(_snapshot_summary(snapshot, config), predictor)
        goal = current_z + config.goal_shift * goals[episode % 16]
        step_ledger = _StepLedger()
        selected, true_costs, reactive_costs = _policy_actions(
            predictions, current_z, goal, predictor, episode, apparatus,
        )
        # Selection is tensor algebra over an immutable snapshot.  The ledger
        # is not exposed to the planner and must still be zero here.
        selection_calls_after_planner = step_ledger.selection_calls
        predicted_z = _standardize(predictions, predictor)
        edge_predictions = predictions[list(EDGE_DERANGEMENT)]
        edge_costs = true_costs[list(EDGE_DERANGEMENT)]
        display_predictions = predictions[list(EDGE_DERANGEMENT)]
        arm_logs: dict[str, Any] = {}
        for arm in ALL_ARMS:
            action_index = selected[arm]
            actual, transition_audit = _one_transition(
                snapshot,
                drives[action_index],
                config,
                ledger=step_ledger,
                phase="actual",
            )
            loss = float((_standardize(actual, predictor) - goal).square().mean())
            losses[arm].append(loss)
            actions[arm].append(ACTION_VALUES[action_index])
            integrity_rows.append(_transition_audit_ok(transition_audit))
            expected_snapshot_hash = fixture.snapshot_hashes[index]
            same_start = (
                transition_audit["starting_snapshot_sha256"] == expected_snapshot_hash
            )
            snapshot_identity = bool(snapshot_identity and same_start)
            planner_predictions = edge_predictions if arm == "edge_shuffle" else predictions
            planner_costs = edge_costs if arm == "edge_shuffle" else true_costs
            displayed = display_predictions if arm == "readout_shuffle" else predictions
            arm_logs[arm] = {
                "pre_map_forecasts": predictions.tolist(),
                "pre_map_forecasts_sha256": _tensor_sha256(predictions),
                "pre_map_costs": true_costs.tolist(),
                "planner_port_permutation": (
                    list(EDGE_DERANGEMENT) if arm == "edge_shuffle" else [0, 1, 2]
                ),
                "planner_port_forecasts": planner_predictions.tolist(),
                "planner_port_costs": planner_costs.tolist(),
                "displayed_forecasts": displayed.tolist(),
                "selected_action_index": action_index,
                "selected_action": ACTION_VALUES[action_index],
                "tie_rule": "lowest_action_index",
                "actual_drive": drives[action_index].tolist(),
                "actual_drive_sha256": _tensor_sha256(drives[action_index]),
                "candidate_runtime_steps": selection_calls_after_planner,
                "selection_step_calls_observed": selection_calls_after_planner,
                "actual_runtime_steps": transition_audit["step_delta"],
                "starting_snapshot_sha256": transition_audit["starting_snapshot_sha256"],
                "expected_starting_snapshot_sha256": expected_snapshot_hash,
                "starting_snapshot_matches_fixture": same_start,
                "ending_snapshot_sha256": transition_audit["ending_snapshot_sha256"],
                "post_weight_sha256": transition_audit["weight_sha256"],
                "loss": loss,
                "transition_integrity": _transition_audit_ok(transition_audit),
            }
        readout_equivalence = bool(
            readout_equivalence
            and selected["readout_shuffle"] == selected["intact"]
            and arm_logs["readout_shuffle"]["actual_drive_sha256"]
            == arm_logs["intact"]["actual_drive_sha256"]
            and abs(losses["readout_shuffle"][-1] - losses["intact"][-1]) <= 1e-12
        )
        edge_port_identity = bool(
            edge_port_identity
            and torch.equal(
                torch.sort(edge_costs).values,
                torch.sort(true_costs).values,
            )
            and arm_logs["edge_shuffle"]["pre_map_forecasts_sha256"]
            == arm_logs["intact"]["pre_map_forecasts_sha256"]
            and tuple(arm_logs["edge_shuffle"]["planner_port_permutation"])
            == EDGE_DERANGEMENT
        )
        traces.append(
            {
                "episode": episode,
                "snapshot_id": fixture.snapshot_ids[index],
                "snapshot_sha256": fixture.snapshot_hashes[index],
                "context_sha256": _tensor_sha256(context),
                "goal": goal.tolist(),
                "goal_sha256": _tensor_sha256(goal),
                "predicted_standardized": predicted_z.tolist(),
                "reactive_costs": reactive_costs.tolist(),
                "selection_step_calls_observed": selection_calls_after_planner,
                "actual_step_calls_observed": step_ledger.actual_calls,
                "arms": arm_logs,
            }
        )
        selection_step_calls += step_ledger.selection_calls
        actual_step_calls += step_ledger.actual_calls
    mean_losses = {
        arm: float(torch.tensor(values, dtype=torch.float64).mean())
        for arm, values in losses.items()
    }
    advantages: dict[str, float] = {}
    denominators: dict[str, float] = {}
    for arm in ADVERSE_ARMS:
        denominator = mean_losses[arm] + mean_losses["intact"] + 1e-12
        denominators[arm] = denominator
        advantages[arm] = (
            (mean_losses[arm] - mean_losses["intact"]) / denominator
            if denominator > 1e-12 and math.isfinite(denominator)
            else float("-inf")
        )
    primary_means = [mean_losses["intact"]] + [mean_losses[arm] for arm in ADVERSE_ARMS]
    degenerate = max(primary_means) - min(primary_means) <= 1e-12
    edge_changes = sum(
        left != right
        for left, right in zip(actions["edge_shuffle"], actions["intact"])
    )
    action_trace_hashes = {
        arm: _canonical_sha256(values) for arm, values in actions.items()
    }
    readout_trace_equal = (
        action_trace_hashes["readout_shuffle"] == action_trace_hashes["intact"]
    )
    return {
        "mean_losses": mean_losses,
        "advantages": advantages,
        "advantage_denominators": denominators,
        "minimum_advantage": min(advantages.values()),
        "edge_action_change_rate": edge_changes / config.policy_states,
        "action_traces": actions,
        "action_trace_hashes": action_trace_hashes,
        "readout_action_trace_hash_equal": readout_trace_equal,
        "readout_equivalence": readout_equivalence,
        "edge_port_identity": edge_port_identity,
        "starting_snapshot_identity": snapshot_identity,
        "constant_degenerate": degenerate,
        "all_transition_integrity": all(integrity_rows),
        "candidate_runtime_steps": selection_step_calls,
        "actual_runtime_steps": actual_step_calls,
        "episode_trace_sha256": _canonical_sha256(traces),
        "episodes": traces,
    }


def _frozen_protocol(config: C1MetacontrolConfig) -> bool:
    current = {key: value for key, value in asdict(config).items() if key != "seed"}
    return current == _FROZEN_PARAMETERS


def _seed_integrity(
    fixture: _Fixture,
    predictor: _Predictor,
    prediction: Mapping[str, Any],
    policy: Mapping[str, Any],
    predecessor_lock: bool,
    frozen: bool,
) -> bool:
    fixture_ok = bool(
        fixture.audit["split_ids_disjoint"]
        and fixture.audit["base_dense_sparse_parity"]
        and fixture.audit["base_hippocampal_rows"] == 0
        and fixture.audit["base_temporal_rows"] == 0
        and fixture.audit["automatic_stdp_updates"] == 0
    )
    denominators_ok = all(
        math.isfinite(float(value)) and float(value) > 1e-12
        for value in policy["advantage_denominators"].values()
    )
    return bool(
        frozen
        and predecessor_lock
        and fixture_ok
        and predictor.audit["fit_transition_integrity"]
        and predictor.audit["parameters_finite"]
        and prediction["denominator_finite_positive"]
        and prediction["transition_integrity"]
        and policy["all_transition_integrity"]
        and policy["candidate_runtime_steps"] == 0
        and policy["actual_runtime_steps"]
        == fixture.audit["policy_states"] * len(ALL_ARMS)
        and policy["readout_equivalence"]
        and policy["readout_action_trace_hash_equal"]
        and policy["edge_port_identity"]
        and policy["starting_snapshot_identity"]
        and not policy["constant_degenerate"]
        and denominators_ok
    )


def _c1_prediction_guided_metacontrol_unchecked(
    seed: int,
    config: C1MetacontrolConfig | None = None,
    *,
    _confirmation_manifest: Path | None = None,
) -> dict[str, Any]:
    seed = int(seed)
    if seed in CONFIRMATION_SEEDS:
        if _confirmation_manifest is None:
            raise RuntimeError(
                "official C1 confirmation seeds require a verified development manifest"
            )
        verify_c1_confirmation_manifest(_confirmation_manifest)
    config = replace(config or C1MetacontrolConfig(), seed=seed)
    apparatus = _global_apparatus(config)
    fixture = _build_fixture(seed, config)
    predictor = _fit_predictor(fixture, apparatus, config)
    prediction = _predictor_audit(fixture, apparatus, predictor, config)
    policy = _evaluate_policy(fixture, apparatus, predictor, config)
    predecessor_lock = bool(
        _PREDECESSOR_ARTIFACT.is_file()
        and _file_sha256(_PREDECESSOR_ARTIFACT) == PREDECESSOR_ARTIFACT_SHA256
    )
    frozen = _frozen_protocol(config)
    integrity = _seed_integrity(
        fixture, predictor, prediction, policy, predecessor_lock, frozen,
    )
    local_pass = bool(
        integrity
        and prediction["mse_ratio"] <= config.prediction_ratio_threshold
        and policy["minimum_advantage"] > config.advantage_threshold
        and policy["edge_action_change_rate"] >= config.edge_change_threshold
    )
    return {
        "result_schema": RESULT_SCHEMA,
        "route": ROUTE,
        "seed": seed,
        "config": asdict(config),
        "frozen_parameters": _FROZEN_PARAMETERS,
        "frozen_protocol": frozen,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_sha256": c1_source_hashes(),
        "environment": _current_environment(),
        "predecessor_artifact": str(_PREDECESSOR_ARTIFACT.relative_to(_REPOSITORY)),
        "predecessor_artifact_sha256": PREDECESSOR_ARTIFACT_SHA256,
        "predecessor_lock": predecessor_lock,
        "apparatus_hashes": apparatus["hashes"],
        "fixture": fixture.audit,
        "predictor": predictor.audit,
        "prediction_audit": prediction,
        "policy": policy,
        "integrity": integrity,
        "local_gate_pass": local_pass,
        "status": "PASS_CANDIDATE" if local_pass else "STOP",
    }


def c1_prediction_guided_metacontrol(
    seed: int,
    config: C1MetacontrolConfig | None = None,
) -> dict[str, Any]:
    seed = int(seed)
    if seed in CONFIRMATION_SEEDS:
        raise RuntimeError("official C1 confirmation seeds require manifest-verified run_c1_stage")
    return _c1_prediction_guided_metacontrol_unchecked(seed, config=config)


def run_c1_seed_range(
    seeds: Iterable[int],
    *,
    config: C1MetacontrolConfig | None = None,
) -> list[dict[str, Any]]:
    seed_list = [int(seed) for seed in seeds]
    if any(seed in CONFIRMATION_SEEDS for seed in seed_list):
        raise RuntimeError("official C1 confirmation seeds require manifest-verified run_c1_stage")
    return [c1_prediction_guided_metacontrol(seed, config=config) for seed in seed_list]


def _bootstrap_c1(
    results: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not results or samples <= 0:
        raise ValueError("C1 bootstrap requires results and a positive sample count")
    ratio = torch.tensor(
        [row["prediction_audit"]["mse_ratio"] for row in results], dtype=torch.float64,
    )
    advantage = torch.tensor(
        [row["policy"]["minimum_advantage"] for row in results], dtype=torch.float64,
    )
    edge_change = torch.tensor(
        [row["policy"]["edge_action_change_rate"] for row in results], dtype=torch.float64,
    )
    if not all(bool(torch.isfinite(value).all()) for value in (ratio, advantage, edge_change)):
        raise ValueError("C1 bootstrap inputs must be finite")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        0, len(results), (int(samples), len(results)), generator=generator,
    )
    ratio_means = ratio[indices].mean(dim=1)
    advantage_means = advantage[indices].mean(dim=1)
    edge_means = edge_change[indices].mean(dim=1)
    if not all(
        bool(torch.isfinite(value).all())
        for value in (ratio_means, advantage_means, edge_means)
    ):
        raise ValueError("C1 bootstrap produced a nonfinite replicate")
    lower_index = max(0, int(0.05 * samples) - 1)
    upper_index = min(samples - 1, int(0.95 * samples))
    sorted_ratio = torch.sort(ratio_means).values
    sorted_advantage = torch.sort(advantage_means).values
    sorted_edge = torch.sort(edge_means).values
    return {
        "samples": int(samples),
        "seed": int(seed),
        "resampled_seed_units": len(results),
        "lower_order_index_zero_based": lower_index,
        "upper_order_index_zero_based": upper_index,
        "mean_prediction_ratio_ucb_95": float(sorted_ratio[upper_index]),
        "mean_minimum_advantage_lcb_95": float(sorted_advantage[lower_index]),
        "mean_edge_change_rate_lcb_95": float(sorted_edge[lower_index]),
    }


def _validate_stage_results(
    results: list[dict[str, Any]],
    stage: str,
) -> dict[str, Any]:
    if stage not in {"development", "confirmation"}:
        raise ValueError("C1 stage must be development or confirmation")
    expected_range = DEVELOPMENT_SEEDS if stage == "development" else CONFIRMATION_SEEDS
    expected = list(expected_range)
    actual = [int(row.get("seed", -1)) for row in results]
    if actual != expected or len(set(actual)) != len(expected):
        raise ValueError(
            f"C1 {stage} requires the exact ordered unique seed block "
            f"{expected[0]}..{expected[-1]}"
        )
    current_sources = c1_source_hashes()
    expected_apparatus = _global_apparatus(C1MetacontrolConfig())["hashes"]
    expected_environment = _current_environment()
    for row, seed in zip(results, expected):
        expected_config = asdict(C1MetacontrolConfig(seed=seed))
        if row.get("result_schema") != RESULT_SCHEMA or row.get("route") != ROUTE:
            raise ValueError("mixed or unknown C1 result rows")
        if row.get("config") != expected_config or not row.get("frozen_protocol"):
            raise ValueError("nonfrozen or mixed C1 configuration")
        if row.get("frozen_parameters") != _FROZEN_PARAMETERS:
            raise ValueError("mixed C1 frozen-parameter manifest")
        if row.get("claim_boundary") != CLAIM_BOUNDARY:
            raise ValueError("C1 simulator claim boundary is not frozen")
        if row.get("source_sha256") != current_sources:
            raise ValueError("C1 source freeze mismatch")
        if row.get("environment") != expected_environment:
            raise ValueError("C1 execution environment mismatch")
        if row.get("apparatus_hashes") != expected_apparatus:
            raise ValueError("mixed C1 global apparatus")
        if (
            row.get("predecessor_artifact_sha256") != PREDECESSOR_ARTIFACT_SHA256
            or not row.get("predecessor_lock")
        ):
            raise ValueError("C1 predecessor evidence lock failed")
        predictor = row.get("predictor", {})
        if predictor.get("feature_dim") != 388 or predictor.get("output_dim") != 4:
            raise ValueError("C1 predictor schema or dimension mismatch")
        fixture = row.get("fixture", {})
        if (
            fixture.get("fit_states") != 128
            or fixture.get("audit_states") != 48
            or fixture.get("policy_states") != 64
        ):
            raise ValueError("C1 split counts are not frozen")
    return {
        "stage": stage,
        "seed_start": expected[0],
        "seed_stop_inclusive": expected[-1],
        "seed_count": len(expected),
        "unique": True,
        "ordered_exact_block": True,
    }


def summarize_c1(
    results: list[dict[str, Any]],
    *,
    stage: str = "development",
    confirmation_manifest: Path | None = None,
) -> dict[str, Any]:
    if not results:
        raise ValueError("C1 summary requires at least one circuit")
    if stage == "confirmation":
        if confirmation_manifest is None:
            raise RuntimeError("C1 confirmation summary requires a verified development manifest")
        verify_c1_confirmation_manifest(confirmation_manifest)
    elif confirmation_manifest is not None:
        raise RuntimeError("C1 development summary does not accept a confirmation manifest")
    stage_audit = _validate_stage_results(results, stage)
    config = C1MetacontrolConfig(**results[0]["config"])
    bootstrap = _bootstrap_c1(
        results,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed,
    )
    prediction_pass = [
        row["prediction_audit"]["mse_ratio"] <= config.prediction_ratio_threshold
        for row in results
    ]
    advantage_pass = [
        row["policy"]["minimum_advantage"] > config.advantage_threshold
        for row in results
    ]
    edge_pass = [
        row["policy"]["edge_action_change_rate"] >= config.edge_change_threshold
        for row in results
    ]
    integrity_all = all(row.get("integrity") for row in results)
    prediction_fraction = sum(prediction_pass) / len(results)
    advantage_fraction = sum(advantage_pass) / len(results)
    edge_fraction = sum(edge_pass) / len(results)
    pass_gate = bool(
        integrity_all
        and prediction_fraction >= 0.80
        and bootstrap["mean_prediction_ratio_ucb_95"] <= config.prediction_ratio_threshold
        and advantage_fraction >= 0.80
        and bootstrap["mean_minimum_advantage_lcb_95"] > config.advantage_threshold
        and edge_fraction >= 0.80
        and bootstrap["mean_edge_change_rate_lcb_95"] > config.edge_change_threshold
    )
    return {
        "stage_audit": stage_audit,
        "circuit_count": len(results),
        "integrity_all": integrity_all,
        "prediction_pass_count": sum(prediction_pass),
        "prediction_pass_fraction": prediction_fraction,
        "advantage_pass_count": sum(advantage_pass),
        "advantage_pass_fraction": advantage_fraction,
        "edge_change_pass_count": sum(edge_pass),
        "edge_change_pass_fraction": edge_fraction,
        "bootstrap": bootstrap,
        "claim_boundary": CLAIM_BOUNDARY,
        "route_verdict": "GO" if pass_gate else "STOP",
    }


def _read_strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def verify_c1_confirmation_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.resolve()
    manifest = _read_strict_json(manifest_path)
    if manifest.get("status") != "FROZEN":
        raise RuntimeError("C1 confirmation manifest status must be FROZEN")
    if manifest.get("development_route_verdict") != "GO":
        raise RuntimeError("C1 confirmation is sealed because development did not pass")
    current_sources = c1_source_hashes()
    current_environment = _current_environment()
    if manifest.get("files") != current_sources:
        raise RuntimeError("C1 confirmation source hash mismatch")
    if manifest.get("environment") != current_environment:
        raise RuntimeError("C1 confirmation environment mismatch")

    artifact_name = manifest.get("development_artifact")
    artifact_hash = manifest.get("development_artifact_sha256")
    results_hash = manifest.get("development_results_sha256")
    if not all(isinstance(value, str) for value in (artifact_name, artifact_hash, results_hash)):
        raise RuntimeError("C1 development artifact provenance is incomplete")
    artifact_path = Path(artifact_name)
    if not artifact_path.is_absolute():
        artifact_path = _REPOSITORY / artifact_path
    artifact_path = artifact_path.resolve()
    if not artifact_path.is_file() or _file_sha256(artifact_path) != artifact_hash:
        raise RuntimeError("C1 development artifact hash mismatch")
    artifact = _read_strict_json(artifact_path)
    if artifact.get("schema") != ARTIFACT_SCHEMA or artifact.get("mode") != "development":
        raise RuntimeError("development artifact is not a C1 development artifact")
    if (
        artifact.get("seed_start") != min(DEVELOPMENT_SEEDS)
        or artifact.get("seed_stop_inclusive") != max(DEVELOPMENT_SEEDS)
        or artifact.get("result_count") != len(DEVELOPMENT_SEEDS)
    ):
        raise RuntimeError("C1 development artifact seed block is invalid")
    results = artifact.get("results")
    if not isinstance(results, list):
        raise RuntimeError("C1 development artifact results are missing")
    _validate_stage_results(results, "development")
    try:
        canonical = json.dumps(
            results, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("C1 development results are not canonical finite JSON") from exc
    canonical_hash = hashlib.sha256(canonical).hexdigest()
    if artifact.get("results_sha256") != canonical_hash or results_hash != canonical_hash:
        raise RuntimeError("C1 development result hash mismatch")
    if artifact.get("source_sha256") != current_sources:
        raise RuntimeError("C1 development artifact source hashes do not match")
    if artifact.get("environment") != current_environment:
        raise RuntimeError("C1 development artifact environment does not match")
    recomputed = summarize_c1(results, stage="development")
    if artifact.get("summary") != recomputed:
        raise RuntimeError("C1 development artifact summary does not reproduce")
    if recomputed.get("route_verdict") != "GO":
        raise RuntimeError("C1 development artifact did not pass")
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "development_artifact": str(artifact_path),
        "development_artifact_sha256": artifact_hash,
        "development_results_sha256": canonical_hash,
        "source_sha256": current_sources,
        "environment": current_environment,
    }


def run_c1_stage(
    stage: str,
    *,
    confirmation_manifest: Path | None = None,
) -> list[dict[str, Any]]:
    if stage == "development":
        if confirmation_manifest is not None:
            raise RuntimeError("C1 development execution does not accept a confirmation manifest")
        results = run_c1_seed_range(DEVELOPMENT_SEEDS)
    elif stage == "confirmation":
        if confirmation_manifest is None:
            raise RuntimeError("C1 confirmation execution requires a verified development manifest")
        verify_c1_confirmation_manifest(confirmation_manifest)
        results = [
            _c1_prediction_guided_metacontrol_unchecked(
                seed,
                _confirmation_manifest=confirmation_manifest,
            )
            for seed in CONFIRMATION_SEEDS
        ]
    else:
        raise ValueError("C1 stage must be development or confirmation")
    _validate_stage_results(results, stage)
    return results
