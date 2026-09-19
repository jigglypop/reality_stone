"""G3-D: a non-mediational response/recall diagnostic over frozen M1.

The route deliberately cannot identify ``response summary -> recall``.  It
asks only whether a task-independent SPD response summary and continuous
zero-store recall have predeclared, same-control contrasts across frozen M1
circuits, while actively searching for a summary-null weight lesion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from ..runtime import BrainRuntime, BrainRuntimeSnapshot, HippocampusMemory, RuntimeMode
from .runtime_alternative_memory import (
    AlternativeMemoryConfig,
    DelayedSignedEligibility,
    _association_contrast,
    _dense_sparse_parity,
    _evaluate_sealed,
    _m1_apply_block,
    _m1_runtime,
)
from .runtime_native_loops import _codebook, _loop8_replay_source_audit, _unit
from ..temporal_memory import TemporalAuditedMemory


DEVELOPMENT_SEEDS = range(97801, 97817)
RETIRED_DEVELOPMENT_SEEDS = range(97701, 97717)
CONFIRMATION_SEEDS = range(99701, 99733)
ADVERSE_CONDITIONS = ("target_shuffled", "no_replay", "weight_permuted")
M1_PARITY_SMOKE_SEED = 97699
M1_SOURCE_SHA256 = "be708bac30bb4e7e681990f838159e70efb9ed36061cef602771e86c8248c27a"
_M1_SOURCE = Path(__file__).with_name("runtime_alternative_memory.py")
_REPOSITORY = Path(__file__).resolve().parents[5]
G3_FREEZE_FILES = (
    "reality_stone/python/reality_stone/clarus/experiments/runtime_alternative_memory.py",
    "reality_stone/python/reality_stone/clarus/experiments/runtime_metric_memory_diagnostic.py",
    "reality_stone/python/reality_stone/clarus/experiments/runtime_metric_memory_diagnostic_benchmark.py",
    "tests/test_runtime_metric_memory_diagnostic.py",
    "_workspace/ce/brainruntime-weight-metric-dynamics-intervention-20260819/02-g3-diagnostic-contract.md",
    "_workspace/ce/brainruntime-weight-metric-dynamics-intervention-20260819/22-g3-audit.md",
)


@dataclass(frozen=True)
class G3DiagnosticConfig:
    dim: int = 48
    replay_epochs: int = 12
    replay_ticks: int = 3
    rollout_horizon: int = 6
    cue_corruption: float = 0.15
    cue_drive_gain: float = 5.0
    max_write_norm: float = 5.0
    m1_lr: float = 0.8
    m1_trace_decay: float = 0.95
    m1_eligibility_decay: float = 0.99
    m1_ltp: float = 1.0
    m1_ltd: float = 0.20
    m1_abstain_threshold: float = 0.20
    calibration_amplitude: float = 5.0
    response_regularizer: float = 1e-3
    lesion_direction_count: int = 8
    lesion_frobenius_norm: float = 0.25
    lesion_norm_tolerance: float = 1e-6
    lesion_quantization_tolerance: float = 1e-6
    lesion_install_bound: float = 0.250001
    lesion_target_tolerance: float = 1e-7
    lesion_stack_tolerance: float = 0.02
    lesion_airm_tolerance: float = 0.02
    lesion_recall_shift: float = 0.05
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 97898
    seed: int = 97801

    def __post_init__(self) -> None:
        if self.dim < 12 or self.dim % 6:
            raise ValueError("G3-D dimension must be at least 12 and divisible by six")
        if self.replay_epochs < 1 or self.replay_ticks < 1 or self.rollout_horizon < 1:
            raise ValueError("training and rollout counts must be positive")
        if not 0.0 <= self.cue_corruption < 1.0:
            raise ValueError("cue_corruption must be in [0,1)")
        if self.calibration_amplitude <= 0.0 or self.response_regularizer <= 0.0:
            raise ValueError("calibration amplitude and regularizer must be positive")
        if self.lesion_direction_count < 1 or self.lesion_frobenius_norm <= 0.0:
            raise ValueError("lesion bank parameters must be positive")
        if (
            self.lesion_norm_tolerance <= 0.0
            or self.lesion_quantization_tolerance <= 0.0
            or self.lesion_install_bound < self.lesion_frobenius_norm
            or self.lesion_target_tolerance <= 0.0
        ):
            raise ValueError("lesion representation tolerances and headroom are invalid")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")

    def alternative(self) -> AlternativeMemoryConfig:
        return AlternativeMemoryConfig(
            dim=self.dim,
            replay_epochs=self.replay_epochs,
            replay_ticks=self.replay_ticks,
            rollout_horizon=self.rollout_horizon,
            cue_corruption=self.cue_corruption,
            cue_drive_gain=self.cue_drive_gain,
            max_write_norm=self.max_write_norm,
            m1_lr=self.m1_lr,
            m1_trace_decay=self.m1_trace_decay,
            m1_eligibility_decay=self.m1_eligibility_decay,
            m1_ltp=self.m1_ltp,
            m1_ltd=self.m1_ltd,
            m1_abstain_threshold=self.m1_abstain_threshold,
            seed=self.seed,
        )


_FROZEN_G3_PARAMETERS = {
    key: value for key, value in asdict(G3DiagnosticConfig()).items() if key != "seed"
}


@dataclass
class _TrainingRun:
    initial_snapshot: BrainRuntimeSnapshot
    sealed_snapshot: BrainRuntimeSnapshot
    cues: torch.Tensor
    targets: torch.Tensor
    indices: list[int]
    report: dict[str, Any]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def g3_source_hashes() -> dict[str, str]:
    return {name: _file_sha256(_REPOSITORY / name) for name in G3_FREEZE_FILES}


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _frozen_protocol(config: G3DiagnosticConfig) -> bool:
    values = asdict(config)
    return all(values[name] == expected for name, expected in _FROZEN_G3_PARAMETERS.items())


def _train_m1_arm(seed: int, config: G3DiagnosticConfig, condition: str) -> _TrainingRun:
    """Duplicate the frozen M1 body while retaining its pre/post snapshots.

    A focused parity test compares this function with the untouched predecessor.
    """
    if condition not in {"fixed_clock", "target_shuffled", "no_replay", "zero_gate"}:
        raise ValueError(f"unsupported G3-D M1 condition: {condition}")
    alternative = config.alternative()
    runtime = _m1_runtime(alternative)
    temporal, source, _ = _loop8_replay_source_audit()
    indices = [int(entry["value"]) for entry in source]
    shifted = indices[1:] + indices[:1]
    cues, targets = _codebook(seed, config.dim)
    initial_snapshot = runtime.snapshot()
    initial = runtime.weight.clone()
    tracker = DelayedSignedEligibility(alternative)
    block_audits: list[dict[str, float]] = []
    mid_block_unchanged = True
    runtime_ticks = 0
    event_count = 0
    pulse_count = 0
    interphase_reset_count = 0

    for _epoch in range(config.replay_epochs):
        for position, index in enumerate(indices):
            paired = shifted[position] if condition == "target_shuffled" else index
            cue, target = cues[index], targets[paired]
            runtime.reset_evaluation_state()
            runtime.hippocampus = HippocampusMemory(
                config.dim,
                capacity=runtime.config.memory_capacity,
                device=runtime.device,
            )
            block_weight = runtime.weight.clone()

            def observe_step(*, external: torch.Tensor, cue_arg: torch.Tensor, mode: RuntimeMode) -> None:
                nonlocal mid_block_unchanged, runtime_ticks, event_count
                runtime.step(
                    external_input=external,
                    cue=cue_arg,
                    force_mode=mode,
                    learning_signal=0.0,
                )
                tracker.observe(runtime.activation)
                mid_block_unchanged = mid_block_unchanged and torch.equal(
                    runtime.weight, block_weight,
                )
                runtime_ticks += 1
                event_count += 1

            observe_step(
                external=config.cue_drive_gain * cue,
                cue_arg=cue,
                mode=RuntimeMode.WAKE,
            )
            if condition != "no_replay":
                runtime.hippocampus.encode(cue, value=target, priority=1.0)
            runtime.reset_evaluation_state()
            interphase_reset_count += 1
            for _ in range(config.replay_ticks):
                observe_step(
                    external=torch.zeros(config.dim),
                    cue_arg=cue,
                    mode=RuntimeMode.NREM,
                )

            gate = 0.0 if condition == "zero_gate" else 1.0
            block_audits.append(_m1_apply_block(runtime, tracker, gate, alternative))
            pulse_count += 1

    delta = runtime.weight - initial
    result = _evaluate_sealed(
        runtime,
        temporal,
        cues,
        targets,
        indices,
        alternative,
        abstain_threshold=config.m1_abstain_threshold,
    )
    result.update({
        "condition": condition,
        "weight_drift": float(delta.norm().item()),
        "association_contrast": _association_contrast(delta, cues, targets, indices),
        "source_indices": indices,
        "source_manifest": source,
        "block_count": config.replay_epochs * len(indices),
        "pulse_count": pulse_count,
        "event_count": event_count,
        "runtime_tick_count": runtime_ticks,
        "interphase_reset_count": interphase_reset_count,
        "expected_event_count": config.replay_epochs * len(indices) * (config.replay_ticks + 1),
        "mid_block_weight_unchanged": bool(mid_block_unchanged),
        "block_end_apply_only": True,
        "gate_audit": {
            "source": "fixed_block_end_clock",
            "base_value": 1.0,
            "reads_runtime_state": False,
            "reads_reward": False,
            "reads_target_identity": False,
            "reads_decoder": False,
            "reads_memory_value": False,
            "reads_condition_flag": False,
            "lesion_override": condition if condition == "zero_gate" else "none",
        },
        "block_apply_audits": block_audits,
        "automatic_stdp_updates": int(runtime._stdp_updates),
        "post_weight_sha256": _tensor_sha256(runtime.weight),
        "initial_weight_sha256": _tensor_sha256(initial),
        "codebook_sha256": _tensor_sha256(torch.cat((cues, targets), dim=0)),
    })
    return _TrainingRun(
        initial_snapshot=initial_snapshot,
        sealed_snapshot=runtime.snapshot(),
        cues=cues,
        targets=targets,
        indices=indices,
        report=result,
    )


def _storage_pointer(value: torch.Tensor) -> int:
    return int(value.untyped_storage().data_ptr())


def _weight_permuted_arm(
    seed: int,
    config: G3DiagnosticConfig,
    matched: _TrainingRun,
    zero_gate: _TrainingRun,
) -> _TrainingRun:
    runtime = BrainRuntime.from_snapshot(zero_gate.sealed_snapshot, backend="torch", device="cpu")
    before = runtime.weight.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed + 97901)
    permutation = torch.randperm(config.dim, generator=generator)
    p_matrix = torch.eye(config.dim)[permutation]
    desired = p_matrix @ matched.sealed_snapshot.weight @ p_matrix.T
    requested = desired - before
    requested_norm = float(requested.norm().item())
    no_alias = bool(
        _storage_pointer(runtime.weight) != _storage_pointer(matched.sealed_snapshot.weight)
        and _storage_pointer(desired) != _storage_pointer(matched.sealed_snapshot.weight)
    )
    installed = runtime.install_bounded_recurrent_delta(
        requested, max_frobenius_norm=10.0,
    )
    applied = runtime.weight - before
    reconstruction = float((runtime.weight - desired).norm().item())
    no_clipping = abs(installed - requested_norm) <= 1e-7
    matched_singular = torch.linalg.svdvals(matched.sealed_snapshot.weight)
    applied_singular = torch.linalg.svdvals(runtime.weight)
    matched_row_norms = torch.sort(matched.sealed_snapshot.weight.norm(dim=1)).values
    applied_row_norms = torch.sort(runtime.weight.norm(dim=1)).values
    matched_col_norms = torch.sort(matched.sealed_snapshot.weight.norm(dim=0)).values
    applied_col_norms = torch.sort(runtime.weight.norm(dim=0)).values
    structural_audit = {
        "provenance": "matched_post_learning_coordinate_permutation",
        "randomized_learning_contingency": False,
        "permutation": permutation.tolist(),
        "permutation_matrix": p_matrix.tolist(),
        "permutation_sha256": _tensor_sha256(p_matrix),
        "fresh_zero_gate_initial_sha256": _tensor_sha256(before),
        "matched_weight_sha256": _tensor_sha256(matched.sealed_snapshot.weight),
        "desired_weight_sha256": _tensor_sha256(desired),
        "requested_delta_sha256": _tensor_sha256(requested),
        "applied_delta_sha256": _tensor_sha256(applied),
        "final_weight_sha256": _tensor_sha256(runtime.weight),
        "requested_delta_norm": requested_norm,
        "installed_delta_norm": installed,
        "applied_delta_norm": float(applied.norm().item()),
        "reconstruction_residual": reconstruction,
        "no_clipping": no_clipping,
        "no_tensor_storage_alias": no_alias,
        "frobenius_residual": abs(
            float(runtime.weight.norm().item())
            - float(matched.sealed_snapshot.weight.norm().item())
        ),
        "singular_spectrum_max_residual": float(
            (matched_singular - applied_singular).abs().max().item()
        ),
        "density_equal": int(torch.count_nonzero(runtime.weight))
            == int(torch.count_nonzero(matched.sealed_snapshot.weight)),
        "diagonal_zero": int(torch.count_nonzero(torch.diag(runtime.weight))) == 0,
        "row_norm_multiset_max_residual": float(
            (matched_row_norms - applied_row_norms).abs().max().item()
        ),
        "column_norm_multiset_max_residual": float(
            (matched_col_norms - applied_col_norms).abs().max().item()
        ),
        "dense_sparse_parity": _dense_sparse_parity(runtime),
        "finite": bool(torch.isfinite(runtime.weight).all()),
    }
    final_evaluation = _evaluate_sealed(
        runtime,
        TemporalAuditedMemory(capacity=1),
        zero_gate.cues,
        zero_gate.targets,
        zero_gate.indices,
        config.alternative(),
        abstain_threshold=config.m1_abstain_threshold,
    )
    report = dict(zero_gate.report)
    report.update(final_evaluation)
    report.update({
        "condition": "weight_permuted",
        "weight_drift": float(runtime.weight.norm().item()),
        "association_contrast": _association_contrast(
            runtime.weight,
            zero_gate.cues,
            zero_gate.targets,
            zero_gate.indices,
        ),
        "post_weight_sha256": _tensor_sha256(runtime.weight),
        "structural_control": structural_audit,
    })
    return _TrainingRun(
        initial_snapshot=zero_gate.initial_snapshot,
        sealed_snapshot=runtime.snapshot(),
        cues=zero_gate.cues,
        targets=zero_gate.targets,
        indices=zero_gate.indices,
        report=report,
    )


def _seal_snapshot(snapshot: BrainRuntimeSnapshot) -> tuple[BrainRuntimeSnapshot, dict[str, Any]]:
    runtime = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    temporal = TemporalAuditedMemory(capacity=1)
    temporal_rows_before = len(temporal)
    temporal._versions.clear()
    temporal._evidence_ids.clear()
    rows_before = len(runtime.hippocampus)
    runtime.hippocampus = HippocampusMemory(
        runtime.config.dim,
        capacity=runtime.config.memory_capacity,
        device=runtime.device,
    )
    runtime.config.hippocampal_encoding_enabled = False
    runtime.reset_evaluation_state()
    sealed = runtime.snapshot()
    return sealed, {
        "hippocampal_rows_removed": rows_before,
        "hippocampal_rows_after": len(runtime.hippocampus),
        "temporal_rows_before": temporal_rows_before,
        "temporal_rows_removed": temporal_rows_before,
        "temporal_rows_after": len(temporal),
        "hippocampal_encoding_enabled": runtime.config.hippocampal_encoding_enabled,
        "automatic_stdp_updates": int(runtime._stdp_updates),
        "dense_sparse_parity": _dense_sparse_parity(runtime),
        "weight_sha256": _tensor_sha256(runtime.weight),
    }


def _probe_matrix(dim: int) -> torch.Tensor:
    width = dim // 3
    injection = torch.zeros(dim, 3)
    amplitude = 1.0 / math.sqrt(width)
    for axis in range(3):
        injection[axis * width : (axis + 1) * width, axis] = amplitude
    return injection


def _pulse_trajectory(
    snapshot: BrainRuntimeSnapshot,
    injection: torch.Tensor,
    axis: int,
    amplitude: float,
    config: G3DiagnosticConfig,
) -> dict[str, Any]:
    runtime = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    temporal = TemporalAuditedMemory(capacity=1)
    runtime.reset_evaluation_state()
    restored_step_before = int(runtime.step_index)
    restored_activation_zero = int(torch.count_nonzero(runtime.activation)) == 0
    before = runtime.weight.clone()
    rows_before = len(runtime.hippocampus)
    runtime.step(
        external_input=amplitude * injection[:, axis],
        force_mode=RuntimeMode.WAKE,
        learning_signal=0.0,
    )
    active_count = int(runtime.active_mask().sum().item())
    trajectory: list[torch.Tensor] = []
    for _ in range(config.rollout_horizon):
        runtime.step(
            external_input=torch.zeros(config.dim),
            force_mode=RuntimeMode.WAKE,
            learning_signal=0.0,
        )
        trajectory.append((injection.T @ runtime.activation).detach().double())
    return {
        "trajectory": torch.stack(trajectory),
        "active_count_after_pulse": active_count,
        "hippocampal_rows_before": rows_before,
        "hippocampal_rows_after": len(runtime.hippocampus),
        "temporal_rows_before": 0,
        "temporal_rows_after": len(temporal),
        "restored_step_before": restored_step_before,
        "restored_activation_zero": restored_activation_zero,
        "restored_weight_sha256": _tensor_sha256(before),
        "hippocampal_encoding_enabled": runtime.config.hippocampal_encoding_enabled,
        "weight_unchanged": bool(torch.equal(runtime.weight, before)),
        "automatic_stdp_updates": int(runtime._stdp_updates),
        "dense_sparse_parity": _dense_sparse_parity(runtime),
        "finite": bool(torch.isfinite(runtime.activation).all()),
    }


def _calibrate_response(
    snapshot: BrainRuntimeSnapshot,
    injection: torch.Tensor,
    config: G3DiagnosticConfig,
) -> dict[str, Any]:
    positive: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    for axis in range(3):
        positive.append(_pulse_trajectory(
            snapshot, injection, axis, config.calibration_amplitude, config,
        ))
        negative.append(_pulse_trajectory(
            snapshot, injection, axis, -config.calibration_amplitude, config,
        ))
    b_h: list[torch.Tensor] = []
    for horizon in range(config.rollout_horizon):
        columns = [
            (positive[axis]["trajectory"][horizon] - negative[axis]["trajectory"][horizon])
            / (2.0 * config.calibration_amplitude)
            for axis in range(3)
        ]
        b_h.append(torch.stack(columns, dim=1).double())
    covariance = (
        b_h[-1] @ b_h[-1].T
        + config.response_regularizer * torch.eye(3, dtype=torch.float64)
    )
    eigenvalues = torch.linalg.eigvalsh(covariance)
    expected_active = max(1, int(round(config.dim * 0.25)))
    probes = positive + negative
    integrity = bool(
        torch.isfinite(covariance).all()
        and bool((eigenvalues > 1e-12).all())
        and all(float(matrix.norm().item()) > 0.0 for matrix in b_h)
        and all(
            probe["active_count_after_pulse"] == expected_active
            and probe["hippocampal_rows_before"] == 0
            and probe["hippocampal_rows_after"] == 0
            and probe["temporal_rows_before"] == 0
            and probe["temporal_rows_after"] == 0
            and probe["restored_step_before"] == 0
            and probe["restored_activation_zero"]
            and not probe["hippocampal_encoding_enabled"]
            and probe["weight_unchanged"]
            and probe["automatic_stdp_updates"] == 0
            and probe["dense_sparse_parity"]
            and probe["finite"]
            for probe in probes
        )
    )
    return {
        "B_h": b_h,
        "C": covariance,
        "eigenvalues_C": [float(value) for value in eigenvalues],
        "probes": probes,
        "integrity": integrity,
    }


def _airm_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.detach().double()
    second = second.detach().double()
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError("AIRM inputs must be finite")
    if not torch.allclose(first, first.T, atol=1e-12, rtol=0.0):
        raise ValueError("AIRM reference must be symmetric")
    if not torch.allclose(second, second.T, atol=1e-12, rtol=0.0):
        raise ValueError("AIRM second input must be symmetric")
    first_spectrum, first_vectors = torch.linalg.eigh(first)
    if not bool((first_spectrum > 1e-12).all()):
        raise ValueError("AIRM reference must be strictly SPD")
    second_spectrum = torch.linalg.eigvalsh(second)
    if not torch.isfinite(second_spectrum).all() or not bool((second_spectrum > 1e-12).all()):
        raise ValueError("AIRM second input must be strictly SPD")
    inverse_sqrt = (
        first_vectors
        @ torch.diag(first_spectrum.rsqrt())
        @ first_vectors.T
    )
    relative = inverse_sqrt @ second @ inverse_sqrt
    relative = 0.5 * (relative + relative.T)
    spectrum = torch.linalg.eigvalsh(relative)
    if not torch.isfinite(spectrum).all() or not bool((spectrum > 1e-12).all()):
        raise ValueError("AIRM generalized spectrum must be finite and positive")
    return float(torch.log(spectrum).norm().item())


def _recall_probe(
    snapshot: BrainRuntimeSnapshot,
    cue: torch.Tensor,
    targets: torch.Tensor,
    target_index: int,
    config: G3DiagnosticConfig,
) -> dict[str, Any]:
    runtime = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    temporal = TemporalAuditedMemory(capacity=1)
    runtime.reset_evaluation_state()
    restored_step_before = int(runtime.step_index)
    restored_activation_zero = int(torch.count_nonzero(runtime.activation)) == 0
    before = runtime.weight.clone()
    rows_before = len(runtime.hippocampus)
    runtime.step(
        external_input=config.cue_drive_gain * cue,
        force_mode=RuntimeMode.WAKE,
        learning_signal=0.0,
    )
    for _ in range(config.rollout_horizon):
        runtime.step(
            external_input=torch.zeros(config.dim),
            force_mode=RuntimeMode.WAKE,
            learning_signal=0.0,
        )
    final = _unit(runtime.activation)
    scores = targets @ final
    alternatives = torch.cat((scores[:target_index], scores[target_index + 1 :]))
    margin = float((scores[target_index] - alternatives.mean()).item())
    return {
        "margin": margin,
        "target_cosine": float(scores[target_index].item()),
        "max_other_cosine": float(alternatives.max().item()),
        "hippocampal_rows_before": rows_before,
        "hippocampal_rows_after": len(runtime.hippocampus),
        "temporal_rows_before": 0,
        "temporal_rows_after": len(temporal),
        "restored_step_before": restored_step_before,
        "restored_activation_zero": restored_activation_zero,
        "restored_weight_sha256": _tensor_sha256(before),
        "hippocampal_encoding_enabled": runtime.config.hippocampal_encoding_enabled,
        "weight_unchanged": bool(torch.equal(runtime.weight, before)),
        "automatic_stdp_updates": int(runtime._stdp_updates),
        "dense_sparse_parity": _dense_sparse_parity(runtime),
        "finite": bool(torch.isfinite(runtime.activation).all()),
    }


def _continuous_recall(
    snapshot: BrainRuntimeSnapshot,
    cues: torch.Tensor,
    targets: torch.Tensor,
    indices: list[int],
    config: G3DiagnosticConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    corruption_width = max(1, int(config.dim * config.cue_corruption))
    for index in indices:
        rows.append({
            "source_index": index,
            "cue_condition": "clean",
            **_recall_probe(snapshot, cues[index], targets, index, config),
        })
        corrupted = cues[index].clone()
        corrupted[:corruption_width] = 0.0
        rows.append({
            "source_index": index,
            "cue_condition": "corrupt_15pct_prefix",
            **_recall_probe(snapshot, corrupted, targets, index, config),
        })
    integrity = all(
        row["hippocampal_rows_before"] == 0
        and row["hippocampal_rows_after"] == 0
        and row["temporal_rows_before"] == 0
        and row["temporal_rows_after"] == 0
        and row["restored_step_before"] == 0
        and row["restored_activation_zero"]
        and not row["hippocampal_encoding_enabled"]
        and row["weight_unchanged"]
        and row["automatic_stdp_updates"] == 0
        and row["dense_sparse_parity"]
        and row["finite"]
        for row in rows
    )
    return {
        "R": sum(row["margin"] for row in rows) / len(rows),
        "rows": rows,
        "integrity": bool(integrity),
    }


def _serialize_calibration(calibration: dict[str, Any]) -> dict[str, Any]:
    return {
        "B_h": [matrix.tolist() for matrix in calibration["B_h"]],
        "B_h_sha256": [_tensor_sha256(matrix) for matrix in calibration["B_h"]],
        "C": calibration["C"].tolist(),
        "C_sha256": _tensor_sha256(calibration["C"]),
        "eigenvalues_C": calibration["eigenvalues_C"],
        "probes": [
            {
                **{key: value for key, value in probe.items() if key != "trajectory"},
                "trajectory": probe["trajectory"].tolist(),
            }
            for probe in calibration["probes"]
        ],
        "integrity": calibration["integrity"],
    }


def _training_integrity(report: dict[str, Any], reference: dict[str, Any]) -> bool:
    schedule_fields = (
        "block_count",
        "pulse_count",
        "event_count",
        "runtime_tick_count",
        "interphase_reset_count",
        "expected_event_count",
    )
    return bool(
        all(report[field] == reference[field] for field in schedule_fields)
        and report["event_count"] == report["expected_event_count"]
        and report["mid_block_weight_unchanged"]
        and report["block_end_apply_only"]
        and report["hippocampal_rows_after_rollout"] == 0
        and report["cutoff_audit"]["temporal_rows_after"] == 0
        and report["cutoff_audit"]["hippocampal_rows_after"] == 0
        and report["snapshot_restore_parity"]
        and report["dense_sparse_parity"]
        and report["finite"]
        and report["automatic_stdp_updates"] == 0
        and report["source_indices"] == reference["source_indices"]
        and report["source_manifest"] == reference["source_manifest"]
        and report["codebook_sha256"] == reference["codebook_sha256"]
        and all(
            math.isfinite(float(value))
            for audit in report["block_apply_audits"]
            for value in audit.values()
        )
    )


def _lesion_bank(seed: int, config: G3DiagnosticConfig) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(seed + 97999)
    bank: list[dict[str, Any]] = []
    for direction_index in range(config.lesion_direction_count):
        base = torch.randn(config.dim, config.dim, generator=generator)
        base.fill_diagonal_(0.0)
        base = base * (config.lesion_frobenius_norm / base.norm().clamp_min(1e-12))
        for sign in (1, -1):
            intended = float(sign) * base
            bank.append({
                "direction_index": direction_index,
                "sign": sign,
                "intended_delta": intended,
                "intended_delta_sha256": _tensor_sha256(intended),
                "intended_delta_norm": float(intended.double().norm().item()),
            })
    return bank


def _norm64(value: torch.Tensor) -> float:
    return float(value.detach().double().norm().item())


def _install_candidate(
    matched_snapshot: BrainRuntimeSnapshot,
    intended_delta: torch.Tensor,
    config: G3DiagnosticConfig,
) -> tuple[BrainRuntimeSnapshot | None, dict[str, Any]]:
    runtime = BrainRuntime.from_snapshot(matched_snapshot, backend="torch", device="cpu")
    before = runtime.weight.clone()
    intended = intended_delta.detach().float().to(runtime.device)
    target = (before + intended).float()
    actual = (target - before).float()
    intended_norm = _norm64(intended)
    actual_norm = _norm64(actual)
    intended_norm_error = abs(intended_norm - config.lesion_frobenius_norm)
    norm_error = abs(actual_norm - config.lesion_frobenius_norm)
    quantization_residual = _norm64(actual - intended)
    preinstall_admitted = bool(
        torch.isfinite(intended).all()
        and torch.isfinite(target).all()
        and torch.isfinite(actual).all()
        and intended_norm_error <= config.lesion_norm_tolerance
        and norm_error <= config.lesion_norm_tolerance
        and quantization_residual <= config.lesion_quantization_tolerance
        and actual_norm <= config.lesion_install_bound
    )
    audit: dict[str, Any] = {
        "matched_weight_sha256": _tensor_sha256(before),
        "intended_delta_sha256": _tensor_sha256(intended),
        "intended_delta_norm_float64": intended_norm,
        "intended_norm_error_from_declared": intended_norm_error,
        "target_weight_sha256": _tensor_sha256(target),
        "actual_delta_sha256": _tensor_sha256(actual),
        "actual_delta_norm_float64": actual_norm,
        "actual_norm_error_from_declared": norm_error,
        "intended_to_actual_residual_float64": quantization_residual,
        "install_bound": config.lesion_install_bound,
        "preinstall_representable": preinstall_admitted,
        "install_performed": False,
        "installed_delta_norm": 0.0,
        "applied_delta_sha256": None,
        "applied_delta_norm_float64": 0.0,
        "additive_reconstruction_residual_float64": 0.0,
        "target_reconstruction_residual_float64": 0.0,
        "reconstruction_residual": 0.0,
        "no_clipping": False,
        "final_weight_sha256": None,
        "hippocampal_rows_after": None,
        "dense_sparse_parity": False,
        "finite": False,
    }
    if not preinstall_admitted:
        return None, audit

    native_requested_norm = float(actual.norm().item())
    installed = runtime.install_bounded_recurrent_delta(
        actual,
        max_frobenius_norm=config.lesion_install_bound,
    )
    applied = runtime.weight - before
    additive_reconstruction = _norm64(applied - actual)
    target_reconstruction = _norm64(runtime.weight - target)
    runtime.hippocampus = HippocampusMemory(
        config.dim,
        capacity=runtime.config.memory_capacity,
        device=runtime.device,
    )
    runtime.config.hippocampal_encoding_enabled = False
    runtime.reset_evaluation_state()
    audit.update({
        "install_performed": True,
        "native_requested_delta_norm_float32": native_requested_norm,
        "installed_delta_norm": installed,
        "applied_delta_sha256": _tensor_sha256(applied),
        "applied_delta_norm_float64": _norm64(applied),
        "additive_reconstruction_residual_float64": additive_reconstruction,
        "target_reconstruction_residual_float64": target_reconstruction,
        "reconstruction_residual": target_reconstruction,
        "no_clipping": abs(installed - native_requested_norm) <= 1e-7,
        "final_weight_sha256": _tensor_sha256(runtime.weight),
        "hippocampal_rows_after": len(runtime.hippocampus),
        "dense_sparse_parity": _dense_sparse_parity(runtime),
        "finite": bool(torch.isfinite(runtime.weight).all()),
    })
    return runtime.snapshot(), audit


def _calibration_null_lesion(
    seed: int,
    matched_snapshot: BrainRuntimeSnapshot,
    matched_calibration: dict[str, Any],
    matched_recall: dict[str, Any],
    cues: torch.Tensor,
    targets: torch.Tensor,
    indices: list[int],
    injection: torch.Tensor,
    config: G3DiagnosticConfig,
) -> dict[str, Any]:
    baseline_stack = torch.stack(matched_calibration["B_h"], dim=0)
    denominator = float(baseline_stack.norm().item()) + 1e-12
    candidate_rows: list[dict[str, Any]] = []
    bank = _lesion_bank(seed, config)
    for order, item in enumerate(bank):
        snapshot, install = _install_candidate(
            matched_snapshot, item["intended_delta"], config,
        )
        if snapshot is None:
            candidate_rows.append({
                "order": order,
                "direction_index": item["direction_index"],
                "sign": item["sign"],
                "intended_delta_sha256": item["intended_delta_sha256"],
                "intended_delta_norm": item["intended_delta_norm"],
                "q_stack_relative": 1e30,
                "airm_C_from_matched": 1e30,
                "B_h_sha256": [],
                "C_sha256": None,
                "calibration_integrity": False,
                "install_audit": install,
            })
            continue
        calibration = _calibrate_response(snapshot, injection, config)
        candidate_stack = torch.stack(calibration["B_h"], dim=0)
        q_value = float((candidate_stack - baseline_stack).norm().item()) / denominator
        airm = _airm_distance(matched_calibration["C"], calibration["C"])
        candidate_rows.append({
            "order": order,
            "direction_index": item["direction_index"],
            "sign": item["sign"],
            "intended_delta_sha256": item["intended_delta_sha256"],
            "intended_delta_norm": item["intended_delta_norm"],
            "q_stack_relative": q_value,
            "airm_C_from_matched": airm,
            "B_h_sha256": [_tensor_sha256(matrix) for matrix in calibration["B_h"]],
            "C_sha256": _tensor_sha256(calibration["C"]),
            "calibration_integrity": calibration["integrity"],
            "install_audit": install,
        })
    selected = min(candidate_rows, key=lambda row: (row["q_stack_relative"], row["order"]))
    selected_item = bank[int(selected["order"])]
    selected_snapshot, repeat_install = _install_candidate(
        matched_snapshot, selected_item["intended_delta"], config,
    )
    recall = (
        _continuous_recall(selected_snapshot, cues, targets, indices, config)
        if selected_snapshot is not None
        else {"R": matched_recall["R"], "rows": [], "integrity": False}
    )
    recall_shift = abs(float(recall["R"]) - float(matched_recall["R"]))
    repeat_hash_matches = bool(
        repeat_install["target_weight_sha256"]
        == selected["install_audit"]["target_weight_sha256"]
        and repeat_install["actual_delta_sha256"]
        == selected["install_audit"]["actual_delta_sha256"]
        and repeat_install["final_weight_sha256"] is not None
        and repeat_install["final_weight_sha256"]
        == selected["install_audit"]["final_weight_sha256"]
    )
    falsifier = bool(
        selected["q_stack_relative"] <= config.lesion_stack_tolerance
        and selected["airm_C_from_matched"] <= config.lesion_airm_tolerance
        and recall_shift >= config.lesion_recall_shift
    )
    integrity = bool(
        len(candidate_rows) == 2 * config.lesion_direction_count
        and all(
            row["calibration_integrity"]
            and row["install_audit"]["preinstall_representable"]
            and row["install_audit"]["install_performed"]
            and row["install_audit"]["no_clipping"]
            and row["install_audit"]["intended_norm_error_from_declared"]
                <= config.lesion_norm_tolerance
            and row["install_audit"]["actual_norm_error_from_declared"]
                <= config.lesion_norm_tolerance
            and row["install_audit"]["intended_to_actual_residual_float64"]
                <= config.lesion_quantization_tolerance
            and row["install_audit"]["actual_delta_norm_float64"]
                <= config.lesion_install_bound
            and row["install_audit"]["target_reconstruction_residual_float64"]
                <= config.lesion_target_tolerance
            and row["install_audit"]["hippocampal_rows_after"] == 0
            and row["install_audit"]["dense_sparse_parity"]
            and row["install_audit"]["finite"]
            for row in candidate_rows
        )
        and repeat_hash_matches
        and repeat_install["preinstall_representable"]
        and repeat_install["install_performed"]
        and repeat_install["no_clipping"]
        and repeat_install["target_reconstruction_residual_float64"]
            <= config.lesion_target_tolerance
        and recall["integrity"]
    )
    return {
        "selection_uses_recall": False,
        "candidate_count": len(candidate_rows),
        "candidate_order": "direction_0_positive_negative_then_increasing_direction",
        "candidates": candidate_rows,
        "selected_order": selected["order"],
        "selected_direction_index": selected["direction_index"],
        "selected_sign": selected["sign"],
        "selected_q_stack_relative": selected["q_stack_relative"],
        "selected_airm_C_from_matched": selected["airm_C_from_matched"],
        "selected_recall": recall,
        "matched_recall_R": matched_recall["R"],
        "absolute_recall_shift": recall_shift,
        "repeat_install": repeat_install,
        "repeat_hash_matches": repeat_hash_matches,
        "falsifier_found": falsifier,
        "integrity": integrity,
    }


def _g3_response_recall_diagnostic_unchecked(
    seed: int,
    config: G3DiagnosticConfig | None = None,
) -> dict[str, Any]:
    config = config or G3DiagnosticConfig(seed=seed)
    config = G3DiagnosticConfig(**{**asdict(config), "seed": seed})
    source_hash = _file_sha256(_M1_SOURCE)
    frozen_protocol = _frozen_protocol(config)
    source_lock = source_hash == M1_SOURCE_SHA256

    matched = _train_m1_arm(seed, config, "fixed_clock")
    shuffled = _train_m1_arm(seed, config, "target_shuffled")
    no_replay = _train_m1_arm(seed, config, "no_replay")
    zero_gate = _train_m1_arm(seed, config, "zero_gate")
    structural = _weight_permuted_arm(seed, config, matched, zero_gate)
    training_runs = {
        "matched": matched,
        "target_shuffled": shuffled,
        "no_replay": no_replay,
        "weight_permuted": structural,
    }

    all_initial_hashes = {
        name: _tensor_sha256(run.initial_snapshot.weight)
        for name, run in {**training_runs, "zero_gate": zero_gate}.items()
    }
    common_initial = len(set(all_initial_hashes.values())) == 1
    injection = _probe_matrix(config.dim)
    injection_orthonormal = bool(torch.equal(
        injection.T @ injection,
        torch.eye(3),
    ))
    pre_snapshot, pre_cutoff = _seal_snapshot(matched.initial_snapshot)
    pre_calibration = _calibrate_response(pre_snapshot, injection, config)
    pre_recall = _continuous_recall(
        pre_snapshot, matched.cues, matched.targets, matched.indices, config,
    )

    arms: dict[str, dict[str, Any]] = {}
    internal_calibrations: dict[str, dict[str, Any]] = {}
    internal_snapshots: dict[str, BrainRuntimeSnapshot] = {}
    for name, run in training_runs.items():
        sealed, cutoff = _seal_snapshot(run.sealed_snapshot)
        calibration = _calibrate_response(sealed, injection, config)
        recall = _continuous_recall(
            sealed, run.cues, run.targets, run.indices, config,
        )
        response_change = _airm_distance(pre_calibration["C"], calibration["C"])
        internal_calibrations[name] = calibration
        internal_snapshots[name] = sealed
        arms[name] = {
            "training": run.report,
            "cutoff": cutoff,
            "calibration": _serialize_calibration(calibration),
            "response_change_airm": response_change,
            "recall": recall,
        }

    lesion = _calibration_null_lesion(
        seed,
        internal_snapshots["matched"],
        internal_calibrations["matched"],
        arms["matched"]["recall"],
        matched.cues,
        matched.targets,
        matched.indices,
        injection,
        config,
    )
    contrasts: dict[str, dict[str, float]] = {}
    for name in ADVERSE_CONDITIONS:
        contrasts[name] = {
            "delta_S": float(
                arms["matched"]["response_change_airm"]
                - arms[name]["response_change_airm"]
            ),
            "delta_R": float(arms["matched"]["recall"]["R"] - arms[name]["recall"]["R"]),
        }
    delta_s_min = min(row["delta_S"] for row in contrasts.values())
    delta_r_min = min(row["delta_R"] for row in contrasts.values())

    reference = matched.report
    training_integrity = all(
        _training_integrity(run.report, reference) for run in training_runs.values()
    ) and _training_integrity(zero_gate.report, reference)
    structural_audit = structural.report["structural_control"]
    structural_integrity = bool(
        structural_audit["no_tensor_storage_alias"]
        and structural_audit["no_clipping"]
        and structural_audit["reconstruction_residual"] <= 1e-7
        and structural_audit["frobenius_residual"] <= 1e-6
        and structural_audit["singular_spectrum_max_residual"] <= 1e-5
        and structural_audit["density_equal"]
        and structural_audit["diagonal_zero"]
        and structural_audit["row_norm_multiset_max_residual"] <= 1e-6
        and structural_audit["column_norm_multiset_max_residual"] <= 1e-6
        and structural_audit["dense_sparse_parity"]
        and structural_audit["finite"]
    )
    condition_integrity = all(
        arm["cutoff"]["hippocampal_rows_after"] == 0
        and arm["cutoff"]["temporal_rows_after"] == 0
        and not arm["cutoff"]["hippocampal_encoding_enabled"]
        and arm["cutoff"]["automatic_stdp_updates"] == 0
        and arm["cutoff"]["dense_sparse_parity"]
        and arm["calibration"]["integrity"]
        and arm["recall"]["integrity"]
        for arm in arms.values()
    )
    probe_matrix_inputs = list(
        _probe_matrix.__code__.co_varnames[: _probe_matrix.__code__.co_argcount]
    )
    calibration_inputs = list(
        _calibrate_response.__code__.co_varnames[: _calibrate_response.__code__.co_argcount]
    )
    recall_inputs = list(
        _continuous_recall.__code__.co_varnames[: _continuous_recall.__code__.co_argcount]
    )
    fresh_calibration_probes = all(
        probe["restored_step_before"] == 0
        and probe["restored_activation_zero"]
        and probe["temporal_rows_before"] == probe["temporal_rows_after"] == 0
        for calibration in internal_calibrations.values()
        for probe in calibration["probes"]
    )
    fresh_recall_probes = all(
        row["restored_step_before"] == 0
        and row["restored_activation_zero"]
        and row["temporal_rows_before"] == row["temporal_rows_after"] == 0
        for arm in arms.values()
        for row in arm["recall"]["rows"]
    )
    separation_audit = {
        "fresh_calibration_probe_restores_observed": fresh_calibration_probes,
        "fresh_recall_probe_restores_observed": fresh_recall_probes,
        "calibration_and_recall_use_separate_snapshot_restores": bool(
            fresh_calibration_probes and fresh_recall_probes
        ),
        "probe_matrix_builder_inputs": probe_matrix_inputs,
        "calibration_builder_inputs": calibration_inputs,
        "recall_builder_inputs": recall_inputs,
        "calibration_reads_task_codebook": any(
            name in {"cue", "cues", "target", "targets", "codebook"}
            for name in calibration_inputs
        ),
        "recall_reads_calibration_state": any(
            "calibration" in name or name in {"B_h", "C", "g"}
            for name in recall_inputs
        ),
        "probe_matrix_reads_codebook_or_outcome": probe_matrix_inputs != ["dim"],
        "probe_matrix_sha256": _tensor_sha256(injection),
        "codebook_sha256": matched.report["codebook_sha256"],
        "hashes_distinct": _tensor_sha256(injection) != matched.report["codebook_sha256"],
    }
    integrity = bool(
        frozen_protocol
        and source_lock
        and common_initial
        and injection_orthonormal
        and pre_cutoff["hippocampal_rows_after"] == 0
        and pre_cutoff["temporal_rows_after"] == 0
        and not pre_cutoff["hippocampal_encoding_enabled"]
        and pre_calibration["integrity"]
        and pre_recall["integrity"]
        and training_integrity
        and structural_integrity
        and condition_integrity
        and lesion["integrity"]
        and separation_audit["calibration_and_recall_use_separate_snapshot_restores"]
        and not separation_audit["calibration_reads_task_codebook"]
        and not separation_audit["recall_reads_calibration_state"]
        and not separation_audit["probe_matrix_reads_codebook_or_outcome"]
        and separation_audit["hashes_distinct"]
    )
    directional_pass = bool(
        integrity and delta_s_min > 0.0 and delta_r_min > 0.0 and not lesion["falsifier_found"]
    )
    return {
        "result_schema": "clarus.runtime_metric_memory_diagnostic.g3d.seed.v1",
        "seed": seed,
        "route": "G3_D_nonmediational_response_recall_diagnostic",
        "config": asdict(config),
        "frozen_protocol": frozen_protocol,
        "frozen_parameters": dict(_FROZEN_G3_PARAMETERS),
        "m1_source_sha256": source_hash,
        "m1_source_lock": source_lock,
        "mediation_status": "BLOCKED_NOT_IDENTIFIED",
        "common_initial_snapshot": common_initial,
        "initial_weight_hashes": all_initial_hashes,
        "probe_matrix": injection.tolist(),
        "probe_matrix_sha256": _tensor_sha256(injection),
        "probe_matrix_orthonormal": injection_orthonormal,
        "pre": {
            "cutoff": pre_cutoff,
            "calibration": _serialize_calibration(pre_calibration),
            "recall": pre_recall,
        },
        "arms": arms,
        "contrasts": contrasts,
        "delta_S_min": delta_s_min,
        "delta_R_min": delta_r_min,
        "calibration_null_lesion": lesion,
        "training_integrity": training_integrity,
        "structural_integrity": structural_integrity,
        "condition_integrity": condition_integrity,
        "separation_audit": separation_audit,
        "integrity": integrity,
        "directional_pass": directional_pass,
        "status": "DIRECTIONAL_PASS" if directional_pass else "STOP",
    }


def _pearson(first: torch.Tensor, second: torch.Tensor) -> float | None:
    first = first.double()
    second = second.double()
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        return None
    centered_first = first - first.mean()
    centered_second = second - second.mean()
    denominator = centered_first.norm() * centered_second.norm()
    if float(denominator.item()) <= 1e-15:
        return None
    return float((centered_first @ centered_second / denominator).item())


def _bootstrap_family(
    results: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    delta_s = torch.tensor([
        [row["contrasts"][name]["delta_S"] for name in ADVERSE_CONDITIONS]
        for row in results
    ], dtype=torch.float64)
    delta_r = torch.tensor([
        [row["contrasts"][name]["delta_R"] for name in ADVERSE_CONDITIONS]
        for row in results
    ], dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(0, len(results), (samples, len(results)), generator=generator)
    minimum_mean_s = torch.empty(samples, dtype=torch.float64)
    minimum_mean_r = torch.empty(samples, dtype=torch.float64)
    minimum_rho = torch.empty(samples, dtype=torch.float64)
    for sample_index in range(samples):
        selected_s = delta_s[indices[sample_index]]
        selected_r = delta_r[indices[sample_index]]
        minimum_mean_s[sample_index] = selected_s.mean(dim=0).min()
        minimum_mean_r[sample_index] = selected_r.mean(dim=0).min()
        correlations = []
        for condition_index in range(len(ADVERSE_CONDITIONS)):
            rho = _pearson(selected_s[:, condition_index], selected_r[:, condition_index])
            correlations.append(-1.0 if rho is None else rho)
        minimum_rho[sample_index] = min(correlations)
    return {
        "mean_delta_S_simultaneous_lcb_95": float(torch.quantile(minimum_mean_s, 0.05).item()),
        "mean_delta_R_simultaneous_lcb_95": float(torch.quantile(minimum_mean_r, 0.05).item()),
        "same_arm_rho_simultaneous_lcb_95": float(torch.quantile(minimum_rho, 0.05).item()),
        "samples": samples,
        "seed": seed,
    }


def _validate_stage_results(results: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    if stage not in {"development", "confirmation"}:
        raise ValueError("G3-D stage must be development or confirmation")
    expected_range = DEVELOPMENT_SEEDS if stage == "development" else CONFIRMATION_SEEDS
    expected_seeds = list(expected_range)
    actual_seeds = [int(row.get("seed", -1)) for row in results]
    if actual_seeds != expected_seeds or len(set(actual_seeds)) != len(expected_seeds):
        raise ValueError(
            f"G3-D {stage} requires the exact ordered unique seed block {expected_seeds[0]}..{expected_seeds[-1]}"
        )
    expected_probe_hash = _tensor_sha256(_probe_matrix(48))
    for row, seed in zip(results, expected_seeds):
        expected_config = asdict(G3DiagnosticConfig(seed=seed))
        if row.get("result_schema") != "clarus.runtime_metric_memory_diagnostic.g3d.seed.v1":
            raise ValueError("mixed or unknown G3-D seed result schema")
        if row.get("route") != "G3_D_nonmediational_response_recall_diagnostic":
            raise ValueError("mixed G3-D route rows")
        if row.get("config") != expected_config or not row.get("frozen_protocol"):
            raise ValueError("nonfrozen or mixed G3-D configuration")
        if row.get("frozen_parameters") != _FROZEN_G3_PARAMETERS:
            raise ValueError("mixed frozen-parameter manifest")
        if row.get("m1_source_sha256") != M1_SOURCE_SHA256 or not row.get("m1_source_lock"):
            raise ValueError("mixed or unlocked M1 source")
        if row.get("probe_matrix_sha256") != expected_probe_hash:
            raise ValueError("mixed G3-D probe matrix")
        if row.get("mediation_status") != "BLOCKED_NOT_IDENTIFIED":
            raise ValueError("G3-D mediation boundary is not frozen")
    return {
        "stage": stage,
        "seed_start": expected_seeds[0],
        "seed_stop_inclusive": expected_seeds[-1],
        "seed_count": len(expected_seeds),
        "unique": True,
        "ordered_exact_block": True,
    }


def summarize_g3(
    results: list[dict[str, Any]],
    *,
    stage: str = "development",
    confirmation_manifest: Path | None = None,
) -> dict[str, Any]:
    if not results:
        raise ValueError("G3-D summary requires at least one circuit")
    if stage == "confirmation":
        if confirmation_manifest is None:
            raise RuntimeError("G3-D confirmation summary requires a verified development manifest")
        verify_g3_confirmation_manifest(confirmation_manifest)
    elif confirmation_manifest is not None:
        raise RuntimeError("development summary does not accept a confirmation manifest")
    stage_audit = _validate_stage_results(results, stage)
    config = G3DiagnosticConfig(**results[0]["config"])
    correlations = {
        name: _pearson(
            torch.tensor([row["contrasts"][name]["delta_S"] for row in results]),
            torch.tensor([row["contrasts"][name]["delta_R"] for row in results]),
        )
        for name in ADVERSE_CONDITIONS
    }
    uninformative = any(value is None for value in correlations.values())
    bootstrap = _bootstrap_family(
        results,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed,
    )
    joint_positive = [
        row["delta_S_min"] > 0.0 and row["delta_R_min"] > 0.0
        for row in results
    ]
    falsifier_count = sum(
        row["calibration_null_lesion"]["falsifier_found"] for row in results
    )
    integrity_all = all(row["integrity"] for row in results)
    pass_gate = bool(
        not uninformative
        and integrity_all
        and sum(joint_positive) / len(results) >= 0.80
        and bootstrap["mean_delta_S_simultaneous_lcb_95"] > 0.0
        and bootstrap["mean_delta_R_simultaneous_lcb_95"] > 0.0
        and bootstrap["same_arm_rho_simultaneous_lcb_95"] > 0.0
        and falsifier_count == 0
        and all(row["mediation_status"] == "BLOCKED_NOT_IDENTIFIED" for row in results)
    )
    verdict = "DIAGNOSTIC_PASS" if pass_gate else (
        "UNINFORMATIVE_STOP" if uninformative else "STOP"
    )
    return {
        "stage_audit": stage_audit,
        "circuit_count": len(results),
        "directional_pass_count": sum(row["directional_pass"] for row in results),
        "joint_positive_count": sum(joint_positive),
        "joint_positive_fraction": sum(joint_positive) / len(results),
        "same_arm_correlations": correlations,
        "uninformative_correlation": uninformative,
        "bootstrap": bootstrap,
        "falsifier_count": falsifier_count,
        "integrity_all": integrity_all,
        "mediation_status": "BLOCKED_NOT_IDENTIFIED",
        "route_verdict": verdict,
    }


def verify_g3_confirmation_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN":
        raise RuntimeError("confirmation manifest status must be FROZEN")
    if manifest.get("development_route_verdict") != "DIAGNOSTIC_PASS":
        raise RuntimeError("confirmation is sealed because development did not pass")
    current_sources = g3_source_hashes()
    if manifest.get("files") != current_sources:
        raise RuntimeError("confirmation source hash mismatch")

    artifact_name = manifest.get("development_artifact")
    artifact_hash = manifest.get("development_artifact_sha256")
    results_hash = manifest.get("development_results_sha256")
    if not all(isinstance(value, str) for value in (artifact_name, artifact_hash, results_hash)):
        raise RuntimeError("development artifact provenance is incomplete")
    artifact_path = Path(artifact_name)
    if not artifact_path.is_absolute():
        artifact_path = _REPOSITORY / artifact_path
    artifact_path = artifact_path.resolve()
    if not artifact_path.is_file() or _file_sha256(artifact_path) != artifact_hash:
        raise RuntimeError("development artifact hash mismatch")

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("schema") != "clarus.runtime_metric_memory_diagnostic.g3d.v1":
        raise RuntimeError("development artifact is not a G3-D artifact")
    if artifact.get("mode") != "development":
        raise RuntimeError("confirmation requires a development-mode artifact")
    if (
        artifact.get("seed_start") != min(DEVELOPMENT_SEEDS)
        or artifact.get("seed_stop_inclusive") != max(DEVELOPMENT_SEEDS)
        or artifact.get("result_count") != len(DEVELOPMENT_SEEDS)
    ):
        raise RuntimeError("development artifact seed block is invalid")
    results = artifact.get("results")
    if not isinstance(results, list):
        raise RuntimeError("development artifact results are missing")
    _validate_stage_results(results, "development")
    try:
        canonical = json.dumps(
            results, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("development artifact results are not canonical finite JSON") from exc
    canonical_hash = hashlib.sha256(canonical).hexdigest()
    if artifact.get("results_sha256") != canonical_hash or results_hash != canonical_hash:
        raise RuntimeError("development result hash mismatch")
    if artifact.get("source_sha256") != current_sources:
        raise RuntimeError("development artifact source hashes do not match the freeze")
    recomputed_summary = summarize_g3(results, stage="development")
    if artifact.get("summary") != recomputed_summary:
        raise RuntimeError("development artifact summary does not reproduce")
    if recomputed_summary.get("route_verdict") != "DIAGNOSTIC_PASS":
        raise RuntimeError("development artifact did not pass G3-D")
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "development_artifact": str(artifact_path),
        "development_artifact_sha256": artifact_hash,
        "development_results_sha256": canonical_hash,
        "source_sha256": current_sources,
    }


def run_g3_seed_range(
    seeds: Iterable[int],
    *,
    config: G3DiagnosticConfig | None = None,
) -> list[dict[str, Any]]:
    seed_list = [int(seed) for seed in seeds]
    if any(seed in RETIRED_DEVELOPMENT_SEEDS for seed in seed_list):
        raise RuntimeError("retired apparatus-invalid G3-D development seeds cannot be rerun")
    if any(seed in CONFIRMATION_SEEDS for seed in seed_list):
        raise RuntimeError(
            "official G3-D confirmation seeds require manifest-verified run_g3_stage"
        )
    return [g3_response_recall_diagnostic(seed, config=config) for seed in seed_list]


def g3_response_recall_diagnostic(
    seed: int,
    config: G3DiagnosticConfig | None = None,
) -> dict[str, Any]:
    seed = int(seed)
    if seed in RETIRED_DEVELOPMENT_SEEDS:
        raise RuntimeError("retired apparatus-invalid G3-D development seeds cannot be rerun")
    if seed in CONFIRMATION_SEEDS:
        raise RuntimeError(
            "official G3-D confirmation seeds require manifest-verified run_g3_stage"
        )
    return _g3_response_recall_diagnostic_unchecked(seed, config=config)


def run_g3_stage(
    stage: str,
    *,
    confirmation_manifest: Path | None = None,
) -> list[dict[str, Any]]:
    if stage == "development":
        if confirmation_manifest is not None:
            raise RuntimeError("development execution does not accept a confirmation manifest")
        seeds = DEVELOPMENT_SEEDS
    elif stage == "confirmation":
        if confirmation_manifest is None:
            raise RuntimeError("G3-D confirmation execution requires a verified development manifest")
        verify_g3_confirmation_manifest(confirmation_manifest)
        seeds = CONFIRMATION_SEEDS
    else:
        raise ValueError("G3-D stage must be development or confirmation")
    results = (
        run_g3_seed_range(seeds)
        if stage == "development"
        else [_g3_response_recall_diagnostic_unchecked(seed) for seed in seeds]
    )
    _validate_stage_results(results, stage)
    return results
