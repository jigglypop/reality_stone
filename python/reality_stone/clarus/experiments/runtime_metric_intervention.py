"""Direct recurrent-weight intervention and finite-horizon SPD response audit."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Iterable

import torch

from ..runtime import BrainRuntime, BrainRuntimeConfig, BrainRuntimeSnapshot, RuntimeMode


DEVELOPMENT_SEEDS = range(97401, 97417)
CONFIRMATION_SEEDS = range(99401, 99433)
ARM_NAMES = ("treatment", "sham", "scrambled", "gain_only", "noise_only")
_FROZEN_G1_PARAMETERS = {
    "dim": 48,
    "background_sigma": 0.01,
    "edge_value": 0.08,
    "calibration_amplitude": 0.50,
    "heldout_amplitude": 0.65,
    "horizon": 6,
    "regularizer": 1e-3,
    "first_passage_threshold": 0.05,
    "gain_control": 0.60,
    "noise_control": 0.02,
    "max_write_norm": 5.0,
    "active_threshold": 0.04,
}


@dataclass(frozen=True)
class MetricInterventionConfig:
    dim: int = 48
    background_sigma: float = 0.01
    edge_value: float = 0.08
    calibration_amplitude: float = 0.50
    heldout_amplitude: float = 0.65
    horizon: int = 6
    regularizer: float = 1e-3
    first_passage_threshold: float = 0.05
    gain_control: float = 0.60
    noise_control: float = 0.02
    max_write_norm: float = 5.0
    active_threshold: float = 0.04
    seed: int = 97401

    def __post_init__(self) -> None:
        if self.dim < 6 or self.dim % 3:
            raise ValueError("metric intervention dimension must be divisible by three")
        if self.background_sigma < 0.0:
            raise ValueError("background_sigma must be non-negative")
        if self.edge_value <= 0.0:
            raise ValueError("edge_value must be positive")
        if self.calibration_amplitude <= 0.0 or self.heldout_amplitude <= 0.0:
            raise ValueError("pulse amplitudes must be positive")
        if self.horizon < 1 or self.regularizer <= 0.0:
            raise ValueError("horizon and regularizer must be positive")
        if self.first_passage_threshold <= 0.0 or self.max_write_norm <= 0.0:
            raise ValueError("threshold and write bound must be positive")
        if self.active_threshold < 0.0:
            raise ValueError("active_threshold must be non-negative")


def _frozen_protocol(config: MetricInterventionConfig) -> bool:
    values = asdict(config)
    return all(values[name] == expected for name, expected in _FROZEN_G1_PARAMETERS.items())


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _dense_sparse_parity(runtime: BrainRuntime) -> bool:
    return bool(torch.allclose(
        runtime.weight, runtime.sparse_weight.to_dense(), atol=1e-7, rtol=0.0,
    ))


def _fixed_transform() -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.2, 0.0], [0.0, 1.0, 0.1], [0.1, 0.0, 1.1]],
        dtype=torch.float64,
    )


def _base_fixture(
    seed: int,
    config: MetricInterventionConfig,
) -> tuple[BrainRuntimeSnapshot, torch.Tensor, dict[str, torch.Tensor]]:
    weight_generator = torch.Generator(device="cpu").manual_seed(seed + 6001)
    permutation_generator = torch.Generator(device="cpu").manual_seed(seed + 6002)
    weight = config.background_sigma * torch.randn(
        config.dim, config.dim, generator=weight_generator,
    )
    weight.fill_diagonal_(0.0)
    permutation = torch.randperm(config.dim, generator=permutation_generator)
    width = config.dim // 3
    groups = {
        "S": permutation[:width].clone(),
        "T": permutation[width : 2 * width].clone(),
        "N": permutation[2 * width :].clone(),
    }
    injection = torch.zeros(config.dim, 3)
    amplitude = 1.0 / (width ** 0.5)
    for column, name in enumerate(("S", "T", "N")):
        injection[groups[name], column] = amplitude
    runtime = BrainRuntime(
        weight,
        config=BrainRuntimeConfig(
            dim=config.dim,
            active_ratio=1.0,
            external_gain=0.45,
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=False,
            f1_self_measure=False,
            stdp_enabled=False,
            memory_capacity=4,
            replay_gain=0.0,
            hippocampal_encoding_enabled=False,
            active_threshold=config.active_threshold,
        ),
        backend="torch",
        device="cpu",
    )
    runtime.reset_evaluation_state()
    if len(runtime.hippocampus) != 0 or not _dense_sparse_parity(runtime):
        raise RuntimeError("invalid G1 base fixture")
    return runtime.snapshot(), injection, groups


def _edge_delta(
    config: MetricInterventionConfig,
    groups: dict[str, torch.Tensor],
    receiver: str,
) -> torch.Tensor:
    delta = torch.zeros(config.dim, config.dim)
    rows = groups[receiver]
    columns = groups["S"]
    delta[rows[:, None], columns[None, :]] = config.edge_value
    return delta


def _arm_snapshot(
    base: BrainRuntimeSnapshot,
    injection: torch.Tensor,
    groups: dict[str, torch.Tensor],
    config: MetricInterventionConfig,
    arm: str,
) -> tuple[BrainRuntimeSnapshot, dict[str, Any]]:
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown G1 arm {arm}")
    runtime = BrainRuntime.from_snapshot(base, backend="torch", device="cpu")
    before = runtime.weight.clone()
    delta = torch.zeros_like(before)
    installed = 0.0
    receiver: str | None = None
    if arm == "treatment":
        receiver = "T"
        delta = _edge_delta(config, groups, "T")
        installed = runtime.install_bounded_recurrent_delta(
            delta, max_frobenius_norm=config.max_write_norm,
        )
    elif arm == "scrambled":
        receiver = "N"
        delta = _edge_delta(config, groups, "N")
        installed = runtime.install_bounded_recurrent_delta(
            delta, max_frobenius_norm=config.max_write_norm,
        )
    elif arm == "gain_only":
        runtime.config.external_gain = config.gain_control
    elif arm == "noise_only":
        runtime.config.noise_sigma = config.noise_control
    runtime.reset_evaluation_state()
    applied = runtime.weight - before
    expected_count = (config.dim // 3) ** 2 if arm in {"treatment", "scrambled"} else 0
    expected_norm = abs(config.edge_value) * (expected_count ** 0.5)
    support_threshold = 1e-7
    declared_mask = torch.zeros_like(applied, dtype=torch.bool)
    if receiver is not None:
        declared_mask[groups[receiver][:, None], groups["S"][None, :]] = True
    inside_error = (
        float((applied[declared_mask] - config.edge_value).abs().max().item())
        if bool(declared_mask.any()) else 0.0
    )
    outside_max = (
        float(applied[~declared_mask].abs().max().item())
        if bool((~declared_mask).any()) else 0.0
    )
    reconstruction_residual = float((applied - delta).norm().item())
    audit = {
        "arm": arm,
        "base_weight_sha256": _tensor_sha256(before),
        "delta_sha256": _tensor_sha256(delta),
        "delta_nonzero_count": int((delta != 0).sum().item()),
        "expected_delta_nonzero_count": expected_count,
        "delta_positive_count": int((delta > 0).sum().item()),
        "delta_frobenius_norm": float(delta.norm().item()),
        "expected_delta_frobenius_norm": expected_norm,
        "installed_delta_norm": installed,
        "applied_delta_sha256": _tensor_sha256(applied),
        "applied_nonzero_count": int((applied.abs() > support_threshold).sum().item()),
        "applied_positive_count": int((applied > support_threshold).sum().item()),
        "applied_frobenius_norm": float(applied.norm().item()),
        "applied_reconstruction_residual": reconstruction_residual,
        "declared_block_inside_max_error": inside_error,
        "declared_block_outside_max_abs": outside_max,
        "only_declared_block": bool(
            reconstruction_residual <= 1e-7
            and inside_error <= 1e-7
            and outside_max <= 1e-7
        ),
        "declared_receiver": receiver,
        "declared_receiver_indices": (
            groups[receiver].tolist() if receiver is not None else []
        ),
        "declared_sender_indices": groups["S"].tolist() if receiver is not None else [],
        "external_gain": float(runtime.config.external_gain),
        "noise_sigma": float(runtime.config.noise_sigma),
        "active_threshold": float(runtime.config.active_threshold),
        "stdp_enabled": bool(runtime.config.stdp_enabled),
        "hippocampal_encoding_enabled": bool(runtime.config.hippocampal_encoding_enabled),
        "hippocampal_rows": len(runtime.hippocampus),
        "dense_sparse_parity": _dense_sparse_parity(runtime),
        "finite": bool(torch.isfinite(runtime.weight).all()),
        "injection_sha256": _tensor_sha256(injection),
    }
    return runtime.snapshot(), audit


def _chart(injection: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    return injection.T @ activation


def _pulse_rollout(
    snapshot: BrainRuntimeSnapshot,
    injection: torch.Tensor,
    axis: int,
    amplitude: float,
    config: MetricInterventionConfig,
) -> dict[str, Any]:
    runtime = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    runtime.reset_evaluation_state()
    restored_weight = runtime.weight.clone()
    rows_before = len(runtime.hippocampus)
    runtime.step(
        external_input=amplitude * injection[:, axis],
        force_mode=RuntimeMode.WAKE,
        learning_signal=0.0,
    )
    trajectory = [_chart(injection, runtime.activation).detach().double()]
    active_after_pulse = runtime.active_mask().detach().clone()
    driven_support = injection[:, axis] != 0
    rows_after_pulse = len(runtime.hippocampus)
    first_passage = config.horizon + 1
    for tick in range(1, config.horizon + 1):
        runtime.step(
            external_input=torch.zeros(config.dim),
            force_mode=RuntimeMode.WAKE,
            learning_signal=0.0,
        )
        point = _chart(injection, runtime.activation).detach().double()
        trajectory.append(point)
        if first_passage == config.horizon + 1 and abs(float(point[1])) >= config.first_passage_threshold:
            first_passage = tick
    return {
        "axis": axis,
        "amplitude": amplitude,
        "trajectory": torch.stack(trajectory),
        "endpoint": trajectory[-1],
        "first_passage": first_passage,
        "rows_before": rows_before,
        "rows_after_pulse": rows_after_pulse,
        "active_count_after_pulse": int(active_after_pulse.sum().item()),
        "driven_coordinates_active": bool(active_after_pulse[driven_support].all()),
        "rows_after_rollout": len(runtime.hippocampus),
        "weight_unchanged": bool(torch.equal(runtime.weight, restored_weight)),
        "automatic_stdp_updates": int(runtime._stdp_updates),
        "dense_sparse_parity": _dense_sparse_parity(runtime),
        "finite": bool(torch.isfinite(runtime.activation).all()),
    }


def _metric_from_calibration(
    snapshot: BrainRuntimeSnapshot,
    injection: torch.Tensor,
    config: MetricInterventionConfig,
) -> dict[str, Any]:
    columns: list[torch.Tensor] = []
    probes: list[dict[str, Any]] = []
    for axis in range(3):
        positive = _pulse_rollout(
            snapshot, injection, axis, config.calibration_amplitude, config,
        )
        negative = _pulse_rollout(
            snapshot, injection, axis, -config.calibration_amplitude, config,
        )
        probes.extend((positive, negative))
        columns.append(
            (positive["endpoint"] - negative["endpoint"])
            / (2.0 * config.calibration_amplitude)
        )
    b_matrix = torch.stack(columns, dim=1).double()
    reference = torch.eye(3, dtype=torch.float64)
    covariance = b_matrix @ b_matrix.T + config.regularizer * reference
    metric = torch.linalg.inv(covariance)
    source_covariance = (
        torch.outer(b_matrix[:, 0], b_matrix[:, 0])
        + config.regularizer * reference
    )
    transform = _fixed_transform()
    transformed_b = transform @ b_matrix
    transformed_reference = transform @ reference @ transform.T
    transformed_covariance = (
        transformed_b @ transformed_b.T
        + config.regularizer * transformed_reference
    )
    transformed_metric = torch.linalg.inv(transformed_covariance)
    inverse_transform = torch.linalg.inv(transform)
    covariance_residual = float(
        (transformed_covariance - transform @ covariance @ transform.T).abs().max().item()
    )
    metric_residual = float(
        (
            transformed_metric
            - inverse_transform.T @ metric @ inverse_transform
        ).abs().max().item()
    )
    target_variance = float(source_covariance[1, 1].item())
    cross_response = abs(float(b_matrix[1, 0].item()))
    transform_determinant = float(torch.linalg.det(transform).item())
    transform_condition = float(torch.linalg.cond(transform).item())
    return {
        "B": b_matrix,
        "C": covariance,
        "g": metric,
        "source_C": source_covariance,
        "cross_response": cross_response,
        "target_variance": target_variance,
        "target_variance_identity_residual": abs(
            target_variance - (cross_response ** 2 + config.regularizer)
        ),
        "calibration_probes": probes,
        "covariance_transform_residual": covariance_residual,
        "metric_transform_residual": metric_residual,
        "transform": transform,
        "transform_sha256": _tensor_sha256(transform),
        "transform_bytes_hex": transform.detach().cpu().contiguous().numpy().tobytes().hex(),
        "transform_determinant": transform_determinant,
        "transform_condition_2": transform_condition,
        "transform_finite": bool(
            torch.isfinite(transform).all()
            and torch.isfinite(torch.tensor(transform_determinant))
            and torch.isfinite(torch.tensor(transform_condition))
        ),
        "transform_invertible": abs(transform_determinant) > 1e-12,
        "eigenvalues_C": [float(value) for value in torch.linalg.eigvalsh(covariance)],
        "eigenvalues_g": [float(value) for value in torch.linalg.eigvalsh(metric)],
    }


def _heldout_evaluation(
    snapshot: BrainRuntimeSnapshot,
    injection: torch.Tensor,
    calibration: dict[str, Any],
    config: MetricInterventionConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for amplitude in (config.heldout_amplitude, -config.heldout_amplitude):
        probe = _pulse_rollout(snapshot, injection, 0, amplitude, config)
        control = torch.tensor([amplitude, 0.0, 0.0], dtype=torch.float64)
        prediction = calibration["B"] @ control
        error = float((prediction - probe["endpoint"]).norm().item())
        rows.append({
            "amplitude": amplitude,
            "endpoint": [float(value) for value in probe["endpoint"]],
            "predicted_endpoint": [float(value) for value in prediction],
            "linearization_error": error,
            "absolute_target_endpoint": abs(float(probe["endpoint"][1].item())),
            "signed_target_endpoint": float(probe["endpoint"][1].item()),
            "first_passage": int(probe["first_passage"]),
            "rows_before": probe["rows_before"],
            "rows_after_pulse": probe["rows_after_pulse"],
            "rows_after_rollout": probe["rows_after_rollout"],
            "active_count_after_pulse": probe["active_count_after_pulse"],
            "driven_coordinates_active": probe["driven_coordinates_active"],
            "weight_unchanged": probe["weight_unchanged"],
            "automatic_stdp_updates": probe["automatic_stdp_updates"],
            "dense_sparse_parity": probe["dense_sparse_parity"],
            "finite": probe["finite"],
        })
    return {
        "signs": rows,
        "mean_absolute_target_endpoint": sum(row["absolute_target_endpoint"] for row in rows) / 2.0,
        "max_linearization_error": max(row["linearization_error"] for row in rows),
        "first_passages": [row["first_passage"] for row in rows],
    }


def _airm_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    eigenvalues, eigenvectors = torch.linalg.eigh(first)
    inverse_sqrt = eigenvectors @ torch.diag(eigenvalues.clamp_min(1e-15).rsqrt()) @ eigenvectors.T
    relative = inverse_sqrt @ second @ inverse_sqrt
    spectrum = torch.linalg.eigvalsh(relative).clamp_min(1e-15)
    return float(torch.log(spectrum).norm().item())


def _arm_integrity(arm: dict[str, Any], config: MetricInterventionConfig) -> bool:
    calibration = arm["calibration"]
    install = arm["install_audit"]
    arm_name = install["arm"]
    expected_count = 256 if arm_name in {"treatment", "scrambled"} else 0
    expected_norm = 1.28 if arm_name in {"treatment", "scrambled"} else 0.0
    expected_gain = config.gain_control if arm_name == "gain_only" else 0.45
    expected_noise = config.noise_control if arm_name == "noise_only" else 0.0
    all_probes = calibration["calibration_probes"]
    heldout_rows = arm["heldout"]["signs"]
    return bool(
        _frozen_protocol(config)
        and install["delta_nonzero_count"] == expected_count
        and install["expected_delta_nonzero_count"] == expected_count
        and install["delta_positive_count"] == expected_count
        and install["applied_nonzero_count"] == expected_count
        and install["applied_positive_count"] == expected_count
        and abs(install["delta_frobenius_norm"] - expected_norm) <= 1e-6
        and abs(install["expected_delta_frobenius_norm"] - expected_norm) <= 1e-12
        and abs(install["installed_delta_norm"] - expected_norm) <= 1e-6
        and abs(install["applied_frobenius_norm"] - expected_norm) <= 1e-6
        and install["applied_reconstruction_residual"] <= 1e-7
        and install["declared_block_inside_max_error"] <= 1e-7
        and install["declared_block_outside_max_abs"] <= 1e-7
        and install["only_declared_block"]
        and abs(install["external_gain"] - expected_gain) <= 1e-12
        and abs(install["noise_sigma"] - expected_noise) <= 1e-12
        and abs(install["active_threshold"] - config.active_threshold) <= 1e-12
        and install["dense_sparse_parity"]
        and install["finite"]
        and not install["stdp_enabled"]
        and not install["hippocampal_encoding_enabled"]
        and install["hippocampal_rows"] == 0
        and calibration["covariance_transform_residual"] <= 1e-6
        and calibration["metric_transform_residual"] <= 1e-6
        and calibration["target_variance_identity_residual"] <= 1e-8
        and calibration["transform_finite"]
        and calibration["transform_invertible"]
        and calibration["transform_sha256"] == _tensor_sha256(_fixed_transform())
        and calibration["transform_bytes_hex"]
            == _fixed_transform().detach().cpu().contiguous().numpy().tobytes().hex()
        and min(calibration["eigenvalues_C"]) > 0.0
        and min(calibration["eigenvalues_g"]) > 0.0
        and all(
            probe["rows_before"] == 0
            and probe["rows_after_pulse"] == 0
            and probe["rows_after_rollout"] == 0
            and probe["driven_coordinates_active"]
            and probe["weight_unchanged"]
            and probe["automatic_stdp_updates"] == 0
            and probe["dense_sparse_parity"]
            and probe["finite"]
            for probe in all_probes
        )
        and all(
            row["rows_before"] == 0
            and row["rows_after_pulse"] == 0
            and row["rows_after_rollout"] == 0
            and row["driven_coordinates_active"]
            and row["weight_unchanged"]
            and row["automatic_stdp_updates"] == 0
            and row["dense_sparse_parity"]
            and row["finite"]
            for row in heldout_rows
        )
    )


def g1_edge_intervention(
    seed: int,
    config: MetricInterventionConfig | None = None,
) -> dict[str, Any]:
    config = config or MetricInterventionConfig(seed=seed)
    config = MetricInterventionConfig(**{**asdict(config), "seed": seed})
    frozen_protocol = _frozen_protocol(config)
    base, injection, groups = _base_fixture(seed, config)
    restored = BrainRuntime.from_snapshot(base, backend="torch", device="cpu")
    snapshot_restore_parity = bool(
        torch.equal(restored.weight, base.weight)
        and torch.equal(restored.activation, base.activation)
        and len(restored.hippocampus) == 0
    )
    arms: dict[str, dict[str, Any]] = {}
    for name in ARM_NAMES:
        snapshot, install_audit = _arm_snapshot(
            base, injection, groups, config, name,
        )
        calibration = _metric_from_calibration(snapshot, injection, config)
        heldout = _heldout_evaluation(snapshot, injection, calibration, config)
        arms[name] = {
            "install_audit": install_audit,
            "calibration": calibration,
            "heldout": heldout,
        }

    treatment = arms["treatment"]
    controls = {name: arms[name] for name in ARM_NAMES if name != "treatment"}
    strongest_endpoint = max(
        arm["heldout"]["mean_absolute_target_endpoint"] for arm in controls.values()
    )
    strongest_cross = max(
        arm["calibration"]["cross_response"] for arm in controls.values()
    )
    endpoint_advantage = treatment["heldout"]["mean_absolute_target_endpoint"] - strongest_endpoint
    cross_advantage = treatment["calibration"]["cross_response"] - strongest_cross
    first_passage_by_sign = []
    for sign_index in range(2):
        treatment_tick = treatment["heldout"]["first_passages"][sign_index]
        earliest_control = min(
            arm["heldout"]["first_passages"][sign_index] for arm in controls.values()
        )
        first_passage_by_sign.append({
            "treatment": treatment_tick,
            "earliest_control": earliest_control,
            "passes": treatment_tick <= earliest_control - 1,
        })

    target_delta = treatment["install_audit"]
    scramble_delta = arms["scrambled"]["install_audit"]
    edge_match = bool(
        target_delta["delta_nonzero_count"] == scramble_delta["delta_nonzero_count"]
        and target_delta["delta_positive_count"] == scramble_delta["delta_positive_count"]
        and abs(target_delta["delta_frobenius_norm"] - scramble_delta["delta_frobenius_norm"]) <= 1e-7
        and target_delta["only_declared_block"]
        and scramble_delta["only_declared_block"]
        and target_delta["delta_nonzero_count"] == 256
        and target_delta["delta_positive_count"] == 256
        and target_delta["applied_nonzero_count"] == 256
        and target_delta["applied_positive_count"] == 256
        and abs(target_delta["delta_frobenius_norm"] - 1.28) <= 1e-6
        and abs(target_delta["installed_delta_norm"] - 1.28) <= 1e-6
        and abs(target_delta["applied_frobenius_norm"] - 1.28) <= 1e-6
        and target_delta["applied_reconstruction_residual"] <= 1e-7
        and scramble_delta["applied_reconstruction_residual"] <= 1e-7
    )
    integrity = bool(
        frozen_protocol
        and snapshot_restore_parity
        and edge_match
        and all(_arm_integrity(arm, config) for arm in arms.values())
    )
    per_circuit_go = bool(
        integrity
        and cross_advantage > 0.0
        and endpoint_advantage >= 0.05
        and all(row["passes"] for row in first_passage_by_sign)
        and treatment["heldout"]["max_linearization_error"] <= 0.10
    )
    serializable_arms: dict[str, Any] = {}
    for name, arm in arms.items():
        calibration = dict(arm["calibration"])
        calibration.update({
            "B": calibration["B"].tolist(),
            "C": calibration["C"].tolist(),
            "g": calibration["g"].tolist(),
            "source_C": calibration["source_C"].tolist(),
            "transform": calibration["transform"].tolist(),
            "calibration_probes": [
                {
                    key: (
                        value.tolist() if isinstance(value, torch.Tensor) else value
                    )
                    for key, value in probe.items()
                }
                for probe in calibration["calibration_probes"]
            ],
        })
        serializable_arms[name] = {
            "install_audit": arm["install_audit"],
            "calibration": calibration,
            "heldout": arm["heldout"],
        }
    return {
        "seed": seed,
        "route": "G1_precommitted_directed_edge_intervention",
        "config": asdict(config),
        "frozen_protocol": frozen_protocol,
        "frozen_protocol_parameters": dict(_FROZEN_G1_PARAMETERS),
        "base_weight_sha256": _tensor_sha256(base.weight),
        "coordinate_permutation": torch.cat((groups["S"], groups["T"], groups["N"])).tolist(),
        "injection_sha256": _tensor_sha256(injection),
        "snapshot_restore_parity": snapshot_restore_parity,
        "edge_match": edge_match,
        "endpoint_advantage": endpoint_advantage,
        "cross_response_advantage": cross_advantage,
        "first_passage_by_sign": first_passage_by_sign,
        "airm_treatment_vs_sham": _airm_distance(
            torch.tensor(serializable_arms["sham"]["calibration"]["C"], dtype=torch.float64),
            torch.tensor(serializable_arms["treatment"]["calibration"]["C"], dtype=torch.float64),
        ),
        "integrity": integrity,
        "status": "GO" if per_circuit_go else "STOP",
        "arms": serializable_arms,
    }


def _bootstrap_lcb(values: list[float], *, seed: int = 97499, samples: int = 10_000) -> float:
    data = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        0, len(values), (samples, len(values)), generator=generator,
    )
    means = data[indices].mean(dim=1)
    return float(torch.quantile(means, 0.025).item())


def summarize_g1(results: list[dict[str, Any]]) -> dict[str, Any]:
    go_count = sum(row["status"] == "GO" for row in results)
    advantages = [float(row["endpoint_advantage"]) for row in results]
    lcb = _bootstrap_lcb(advantages)
    route_go = bool(
        go_count / len(results) >= 0.80
        and lcb > 0.0
        and all(row["integrity"] for row in results)
    )
    return {
        "circuit_count": len(results),
        "go_count": go_count,
        "go_fraction": go_count / len(results),
        "endpoint_advantage_mean": sum(advantages) / len(advantages),
        "endpoint_advantage_min": min(advantages),
        "endpoint_advantage_bootstrap_lcb_95": lcb,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 97499,
        "cross_response_advantage_min": min(row["cross_response_advantage"] for row in results),
        "integrity_all": all(row["integrity"] for row in results),
        "route_verdict": "GO" if route_go else "STOP",
    }


def run_g1_seed_range(
    seeds: Iterable[int],
    *,
    config: MetricInterventionConfig | None = None,
) -> list[dict[str, Any]]:
    return [g1_edge_intervention(int(seed), config=config) for seed in seeds]
