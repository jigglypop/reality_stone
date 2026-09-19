"""Fixed-weight G2 compressed metric-feature utility experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Iterable

import torch

from ..runtime import BrainRuntime, BrainRuntimeConfig, BrainRuntimeSnapshot, RuntimeMode


G2_DEVELOPMENT_SEEDS = range(97601, 97617)
G2_CONFIRMATION_SEEDS = range(99601, 99633)
G2_ENVIRONMENTS = ((0.40, 0.0), (0.40, 0.02), (0.60, 0.0), (0.60, 0.02))
G2_ADVERSE_MODELS = (
    "D",
    "D+C",
    "D+E",
    "D+perm",
    "D+Bpath",
    "D+Qraw",
    "D+Cterms",
    "D+Craw",
    "D2",
    "persistence",
    "global_mean",
    "raw_Bpath",
)
_G2_FROZEN = {
    "dim": 48,
    "background_sigma": 0.01,
    "calibration_amplitude": 0.50,
    "horizon": 6,
    "regularizer": 1e-3,
    "active_threshold": 0.0,
    "force_all_active_selection": True,
    "fit_rows": 64,
    "test_rows": 32,
    "codebook_seed": 97599,
    "ridge": 1e-4,
    "score_variance": 1e-4,
}


@dataclass(frozen=True)
class G2Config:
    dim: int = 48
    background_sigma: float = 0.01
    calibration_amplitude: float = 0.50
    horizon: int = 6
    regularizer: float = 1e-3
    active_threshold: float = 0.0
    force_all_active_selection: bool = True
    fit_rows: int = 64
    test_rows: int = 32
    codebook_seed: int = 97599
    ridge: float = 1e-4
    score_variance: float = 1e-4
    seed: int = 97501

    def __post_init__(self) -> None:
        if self.dim != 48 or self.dim % 3:
            raise ValueError("G2 requires the frozen 48-dimensional chart")
        if self.horizon < 1 or self.regularizer <= 0.0:
            raise ValueError("G2 horizon and regularizer must be positive")
        if self.fit_rows < 1 or self.test_rows < 1:
            raise ValueError("G2 fit/test row counts must be positive")
        if self.ridge <= 0.0 or self.score_variance <= 0.0:
            raise ValueError("G2 ridge and score variance must be positive")
        object.__setattr__(self, "force_all_active_selection", bool(self.force_all_active_selection))


def _g2_frozen_protocol(config: G2Config) -> bool:
    values = asdict(config)
    return all(values[name] == expected for name, expected in _G2_FROZEN.items())


def _sha_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _dense_sparse_parity(runtime: BrainRuntime) -> bool:
    return bool(torch.allclose(
        runtime.weight, runtime.sparse_weight.to_dense(), atol=1e-7, rtol=0.0,
    ))


def _g2_transform() -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.2, 0.0], [0.0, 1.0, 0.1], [0.1, 0.0, 1.1]],
        dtype=torch.float64,
    )


def _g2_fixture(
    seed: int,
    config: G2Config,
) -> tuple[BrainRuntimeSnapshot, torch.Tensor, dict[str, torch.Tensor]]:
    """Dedicated fixture; intentionally independent of G1 frozen-protocol logic."""
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
    for column, name in enumerate(("S", "T", "N")):
        injection[groups[name], column] = 1.0 / math.sqrt(width)
    runtime = BrainRuntime(
        weight,
        config=BrainRuntimeConfig(
            dim=config.dim,
            active_ratio=1.0,
            active_threshold=config.active_threshold,
            force_all_active_selection=config.force_all_active_selection,
            external_gain=G2_ENVIRONMENTS[0][0],
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=False,
            f1_self_measure=False,
            stdp_enabled=False,
            memory_capacity=4,
            replay_gain=0.0,
            hippocampal_encoding_enabled=False,
        ),
        backend="torch",
        device="cpu",
    )
    runtime.reset_evaluation_state()
    if len(runtime.hippocampus) or not _dense_sparse_parity(runtime):
        raise RuntimeError("invalid G2 fixture")
    return runtime.snapshot(), injection, groups


def _g2_environment_snapshot(
    base: BrainRuntimeSnapshot,
    gain: float,
    noise: float,
) -> tuple[BrainRuntimeSnapshot, dict[str, Any]]:
    runtime = BrainRuntime.from_snapshot(base, backend="torch", device="cpu")
    before = runtime.weight.clone()
    runtime.config.external_gain = float(gain)
    runtime.config.noise_sigma = float(noise)
    runtime.reset_evaluation_state()
    audit = {
        "gain": float(gain),
        "noise_sigma": float(noise),
        "weight_sha256": _sha_tensor(runtime.weight),
        "weight_unchanged": bool(torch.equal(before, runtime.weight)),
        "dense_sparse_parity": _dense_sparse_parity(runtime),
        "hippocampal_rows": len(runtime.hippocampus),
        "stdp_enabled": bool(runtime.config.stdp_enabled),
        "hippocampal_encoding_enabled": bool(runtime.config.hippocampal_encoding_enabled),
        "active_threshold": float(runtime.config.active_threshold),
    }
    return runtime.snapshot(), audit


def _g2_codebook(config: G2Config) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(config.codebook_seed)
    accepted: list[torch.Tensor] = []
    attempts = 0
    required = config.fit_rows + config.test_rows
    while len(accepted) < required:
        attempts += 1
        candidate = torch.randn(3, generator=generator, dtype=torch.float64)
        norm = candidate.norm()
        if not torch.isfinite(candidate).all() or not torch.isfinite(norm) or norm <= 0.0:
            continue
        candidate = candidate / norm
        if float(candidate.abs().max().item()) > 0.95:
            continue
        if any(float(torch.dot(candidate, previous).abs().item()) >= 1.0 - 1e-10 for previous in accepted):
            continue
        accepted.append(candidate)
    directions = torch.stack(accepted)
    fit = directions[: config.fit_rows].clone()
    test = directions[config.fit_rows :].clone()
    fit_amplitudes = torch.tensor(
        [(.60, .75)[(index // 4) % 2] for index in range(config.fit_rows)],
        dtype=torch.float64,
    )
    test_amplitudes = torch.tensor(
        [(.90, 1.05)[(index // 4) % 2] for index in range(config.test_rows)],
        dtype=torch.float64,
    )
    fit_pairs = torch.cat((fit, fit_amplitudes.unsqueeze(1)), dim=1)
    test_pairs = torch.cat((test, test_amplitudes.unsqueeze(1)), dim=1)
    gram = directions @ directions.T
    gram.fill_diagonal_(0.0)
    audit = {
        "attempts": attempts,
        "direction_sha256": _sha_tensor(directions),
        "fit_direction_sha256": _sha_tensor(fit),
        "test_direction_sha256": _sha_tensor(test),
        "fit_amplitude_sha256": _sha_tensor(fit_amplitudes),
        "test_amplitude_sha256": _sha_tensor(test_amplitudes),
        "fit_pair_sha256": _sha_tensor(fit_pairs),
        "test_pair_sha256": _sha_tensor(test_pairs),
        "max_axis_alignment": float(directions.abs().max().item()),
        "max_pair_alignment": float(gram.abs().max().item()),
        "fit_test_pair_disjoint": not any(
            torch.equal(first, second) for first in fit_pairs for second in test_pairs
        ),
        "finite": bool(torch.isfinite(directions).all()),
    }
    return {
        "directions": directions,
        "fit": fit,
        "test": test,
        "fit_amplitudes": fit_amplitudes,
        "test_amplitudes": test_amplitudes,
        "audit": audit,
    }


def _g2_noise_start(seed: int, environment_index: int, split: str, local_rank: int) -> int:
    if split == "calibration":
        local_id = local_rank
        if not 0 <= local_rank < 3:
            raise ValueError("calibration local rank must be 0..2")
    elif split == "fit":
        local_id = 3 + local_rank
        if not 0 <= local_rank < 16:
            raise ValueError("fit local rank must be 0..15")
    elif split == "test":
        local_id = 19 + local_rank
        if not 0 <= local_rank < 8:
            raise ValueError("test local rank must be 0..7")
    else:
        raise ValueError(f"unknown G2 split {split}")
    if not 0 <= environment_index < len(G2_ENVIRONMENTS):
        raise ValueError("environment index must be 0..3")
    return 8 * (int(seed) * 512 + environment_index * 128 + local_id)


def _g2_noise_schedule(seed: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for environment_index in range(len(G2_ENVIRONMENTS)):
        for split, count in (("calibration", 3), ("fit", 16), ("test", 8)):
            for local_rank in range(count):
                start = _g2_noise_start(seed, environment_index, split, local_rank)
                rows.append({
                    "environment_index": environment_index,
                    "split": split,
                    "local_rank": local_rank,
                    "start": start,
                    "stop_inclusive": start + 6,
                })
    intervals = [(row["start"], row["stop_inclusive"]) for row in rows]
    disjoint = all(
        first_stop < second_start or second_stop < first_start
        for index, (first_start, first_stop) in enumerate(intervals)
        for second_start, second_stop in intervals[index + 1 :]
    )
    starts = torch.tensor([row["start"] for row in rows], dtype=torch.int64)
    return {
        "rows": rows,
        "interval_count": len(rows),
        "pairwise_disjoint": disjoint,
        "max_start_below_2e9": max(row["start"] for row in rows) < 2_000_000_000,
        "starts_sha256": _sha_tensor(starts),
    }


def _g2_rollout(
    snapshot: BrainRuntimeSnapshot,
    injection: torch.Tensor,
    control: torch.Tensor,
    noise_start: int,
    config: G2Config,
) -> dict[str, Any]:
    runtime = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    runtime.reset_evaluation_state()
    reset_parity = bool(
        runtime.step_index == 0
        and torch.count_nonzero(runtime.activation) == 0
        and torch.count_nonzero(runtime.refractory) == 0
        and len(runtime.hippocampus) == 0
    )
    runtime.step_index = int(noise_start)
    weight_before = runtime.weight.clone()
    external = (injection.double() @ control.double()).float()
    runtime.step(
        external_input=external,
        force_mode=RuntimeMode.WAKE,
        learning_signal=0.0,
    )
    y0 = (injection.T @ runtime.activation).detach().double()
    active_masks = [runtime.active_mask().detach().cpu().clone()]
    path: list[torch.Tensor] = []
    first_passage = config.horizon + 1
    for tick in range(1, config.horizon + 1):
        runtime.step(
            external_input=torch.zeros(config.dim),
            force_mode=RuntimeMode.WAKE,
            learning_signal=0.0,
        )
        point = (injection.T @ runtime.activation).detach().double()
        path.append(point)
        active_masks.append(runtime.active_mask().detach().cpu().clone())
        if first_passage == config.horizon + 1 and abs(float(point[1])) >= 0.05:
            first_passage = tick
    path_tensor = torch.stack(path)
    mask_tensor = torch.stack(active_masks)
    native_noise_seeds = [
        (noise_start + offset) * 31337 + 7 for offset in range(config.horizon + 1)
    ]
    integrity = bool(
        reset_parity
        and torch.equal(runtime.weight, weight_before)
        and _dense_sparse_parity(runtime)
        and len(runtime.hippocampus) == 0
        and runtime._stdp_updates == 0
        and torch.isfinite(y0).all()
        and torch.isfinite(path_tensor).all()
        and bool(mask_tensor.all())
        and runtime.step_index == noise_start + config.horizon + 1
    )
    return {
        "control": control.detach().double().clone(),
        "noise_start": int(noise_start),
        "native_noise_seeds": native_noise_seeds,
        "y0": y0,
        "path": path_tensor,
        "target": float(path_tensor[:, 1].abs().mean().item()),
        "first_passage": first_passage,
        "active_counts": [int(mask.sum().item()) for mask in mask_tensor],
        "active_mask_sha256": _sha_tensor(mask_tensor),
        "present_state_sha256": _sha_tensor(y0),
        "reset_parity": reset_parity,
        "weight_unchanged": bool(torch.equal(runtime.weight, weight_before)),
        "weight_sha256": _sha_tensor(runtime.weight),
        "dense_sparse_parity": _dense_sparse_parity(runtime),
        "hippocampal_rows": len(runtime.hippocampus),
        "automatic_stdp_updates": int(runtime._stdp_updates),
        "finite": bool(torch.isfinite(y0).all() and torch.isfinite(path_tensor).all()),
        "integrity": integrity,
    }


def _g2_calibrate(
    snapshot: BrainRuntimeSnapshot,
    injection: torch.Tensor,
    seed: int,
    environment_index: int,
    config: G2Config,
) -> dict[str, Any]:
    positive_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    for axis in range(3):
        start = _g2_noise_start(seed, environment_index, "calibration", axis)
        control = torch.zeros(3, dtype=torch.float64)
        control[axis] = config.calibration_amplitude
        positive_rows.append(_g2_rollout(snapshot, injection, control, start, config))
        negative_rows.append(_g2_rollout(snapshot, injection, -control, start, config))
    b_h = torch.empty(config.horizon, 3, 3, dtype=torch.float64)
    denominator = 2.0 * config.calibration_amplitude
    for horizon_index in range(config.horizon):
        for axis in range(3):
            b_h[horizon_index, :, axis] = (
                positive_rows[axis]["path"][horizon_index]
                - negative_rows[axis]["path"][horizon_index]
            ) / denominator
    b_matrix = b_h[-1].clone()
    reference = torch.eye(3, dtype=torch.float64)
    covariance = b_matrix @ b_matrix.T + config.regularizer * reference
    metric = torch.linalg.inv(covariance)
    endpoints = torch.stack([
        row["path"][-1] for pair in zip(positive_rows, negative_rows) for row in pair
    ])
    endpoint_covariance = torch.einsum("ni,nj->ij", endpoints, endpoints) / endpoints.shape[0]
    q_raw = torch.linalg.inv(endpoint_covariance + config.regularizer * reference)
    transform = _g2_transform()
    transformed_b = transform @ b_matrix
    transformed_reference = transform @ reference @ transform.T
    transformed_covariance = (
        transformed_b @ transformed_b.T + config.regularizer * transformed_reference
    )
    transformed_metric = torch.linalg.inv(transformed_covariance)
    inverse_transform = torch.linalg.inv(transform)
    covariance_residual = float(
        (transformed_covariance - transform @ covariance @ transform.T).abs().max().item()
    )
    metric_residual = float(
        (transformed_metric - inverse_transform.T @ metric @ inverse_transform).abs().max().item()
    )
    all_rows = positive_rows + negative_rows
    integrity = bool(
        all(row["integrity"] for row in all_rows)
        and torch.isfinite(b_h).all()
        and torch.isfinite(covariance).all()
        and torch.isfinite(metric).all()
        and torch.isfinite(q_raw).all()
        and min(torch.linalg.eigvalsh(covariance)) > 0.0
        and min(torch.linalg.eigvalsh(metric)) > 0.0
        and min(torch.linalg.eigvalsh(q_raw)) > 0.0
        and covariance_residual <= 1e-6
        and metric_residual <= 1e-6
    )
    return {
        "B_h": b_h,
        "B": b_matrix,
        "C": covariance,
        "g": metric,
        "Q_raw": q_raw,
        "g_transformed": transformed_metric,
        "transform": transform,
        "transform_covariance_residual": covariance_residual,
        "transform_metric_residual": metric_residual,
        "eigenvalues_C": torch.linalg.eigvalsh(covariance),
        "eigenvalues_g": torch.linalg.eigvalsh(metric),
        "eigenvalues_Q_raw": torch.linalg.eigvalsh(q_raw),
        "positive": positive_rows,
        "negative": negative_rows,
        "integrity": integrity,
    }


def _g2_features(
    present: torch.Tensor,
    control: torch.Tensor,
    calibration: dict[str, Any],
    gain: float,
    noise: float,
) -> dict[str, torch.Tensor]:
    """Build features only from pre-free-tick y0 and calibration/exogenous values."""
    u = present.detach().double().clone()
    c_matrix = calibration["C"]
    metric = calibration["g"]
    q_raw = calibration["Q_raw"]
    permutation = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    permuted_metric = permutation.T @ metric @ permutation
    c_terms = torch.stack((
        c_matrix[0, 0] * u[0].square(),
        c_matrix[1, 1] * u[1].square(),
        c_matrix[2, 2] * u[2].square(),
        2.0 * c_matrix[0, 1] * u[0] * u[1],
        2.0 * c_matrix[0, 2] * u[0] * u[2],
        2.0 * c_matrix[1, 2] * u[1] * u[2],
    ))
    c_raw = torch.stack((
        c_matrix[0, 0], c_matrix[1, 1], c_matrix[2, 2],
        c_matrix[0, 1], c_matrix[0, 2], c_matrix[1, 2],
    ))
    direct_path = torch.stack([
        (calibration["B_h"][index] @ control)[1].abs()
        for index in range(calibration["B_h"].shape[0])
    ]).mean()
    copied_c = c_matrix.detach().clone()
    copied_metric = torch.linalg.inv(copied_c)
    transform = calibration["transform"]
    transformed_present = transform @ u
    metric_invariance_residual = float(abs(
        transformed_present @ calibration["g_transformed"] @ transformed_present
        - u @ metric @ u
    ).item())
    return {
        "base": torch.cat((u, torch.tensor([gain, noise], dtype=torch.float64))),
        "g": (u @ metric @ u).reshape(1),
        "g_from_C": (u @ copied_metric @ u).reshape(1),
        "C": (u @ c_matrix @ u).reshape(1),
        "Cterms": c_terms,
        "Craw": c_raw,
        "E": (u @ u).reshape(1),
        "perm": (u @ permuted_metric @ u).reshape(1),
        "Bpath": direct_path.reshape(1),
        "Qraw": (u @ q_raw @ u).reshape(1),
        "metric_invariance_residual": torch.tensor(
            [metric_invariance_residual], dtype=torch.float64,
        ),
        "g_c_matrix_alias": torch.tensor(
            [float(metric.data_ptr() == copied_metric.data_ptr())], dtype=torch.float64,
        ),
    }


def _g2_dataset(
    seed: int,
    environment_snapshots: list[BrainRuntimeSnapshot],
    injection: torch.Tensor,
    calibrations: list[dict[str, Any]],
    codebook: dict[str, Any],
    split: str,
    config: G2Config,
) -> list[dict[str, Any]]:
    if split == "fit":
        directions = codebook["fit"]
        amplitudes = codebook["fit_amplitudes"]
    elif split == "test":
        directions = codebook["test"]
        amplitudes = codebook["test_amplitudes"]
    else:
        raise ValueError("G2 dataset split must be fit or test")
    rows: list[dict[str, Any]] = []
    for row_index, (direction, amplitude) in enumerate(zip(directions, amplitudes)):
        environment_index = row_index % len(G2_ENVIRONMENTS)
        local_rank = row_index // len(G2_ENVIRONMENTS)
        gain, noise = G2_ENVIRONMENTS[environment_index]
        control = float(amplitude) * direction
        start = _g2_noise_start(seed, environment_index, split, local_rank)
        rollout = _g2_rollout(
            environment_snapshots[environment_index], injection, control, start, config,
        )
        features = _g2_features(
            rollout["y0"], control, calibrations[environment_index], gain, noise,
        )
        rows.append({
            "split": split,
            "row_index": row_index,
            "environment_index": environment_index,
            "environment": [gain, noise],
            "local_rank": local_rank,
            "direction": direction.detach().clone(),
            "amplitude": float(amplitude),
            "control": control.detach().clone(),
            "rollout": rollout,
            "features": features,
        })
    return rows


@dataclass
class _RidgeFit:
    mean: torch.Tensor
    scale: torch.Tensor
    theta: torch.Tensor
    ridge: float


def _fit_ridge(features: torch.Tensor, target: torch.Tensor, ridge: float) -> _RidgeFit:
    features = features.detach().double().clone()
    target = target.detach().double().clone()
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False)
    scale = torch.where(scale <= 1e-12, torch.ones_like(scale), scale)
    standardized = (features - mean) / scale
    design = torch.cat((torch.ones(features.shape[0], 1, dtype=torch.float64), standardized), dim=1)
    penalty = torch.eye(design.shape[1], dtype=torch.float64)
    penalty[0, 0] = 0.0
    theta = torch.linalg.solve(
        design.T @ design + ridge * penalty,
        design.T @ target,
    )
    return _RidgeFit(mean=mean, scale=scale, theta=theta, ridge=ridge)


def _predict_ridge(fit: _RidgeFit, features: torch.Tensor) -> torch.Tensor:
    standardized = (features.detach().double() - fit.mean) / fit.scale
    design = torch.cat((torch.ones(features.shape[0], 1, dtype=torch.float64), standardized), dim=1)
    return design @ fit.theta


def _stack_feature(rows: list[dict[str, Any]], name: str) -> torch.Tensor:
    base = torch.stack([row["features"]["base"] for row in rows])
    if name == "D":
        return base
    mapping = {
        "D+g": "g",
        "D+g_from_C": "g_from_C",
        "D+C": "C",
        "D+E": "E",
        "D+perm": "perm",
        "D+Bpath": "Bpath",
        "D+Qraw": "Qraw",
        "D+Cterms": "Cterms",
        "D+Craw": "Craw",
    }
    if name not in mapping:
        raise ValueError(f"unknown learned G2 model {name}")
    extra = torch.stack([row["features"][mapping[name]] for row in rows])
    return torch.cat((base, extra), dim=1)


def _quadratic_features(
    train_base: torch.Tensor,
    test_base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    mean = train_base.mean(dim=0)
    scale = train_base.std(dim=0, unbiased=False)
    scale = torch.where(scale <= 1e-12, torch.ones_like(scale), scale)
    train_z = (train_base - mean) / scale
    test_z = (test_base - mean) / scale
    indices = [(row, column) for row in range(5) for column in range(row, 5)]
    train_products = torch.stack([train_z[:, i] * train_z[:, j] for i, j in indices], dim=1)
    test_products = torch.stack([test_z[:, i] * test_z[:, j] for i, j in indices], dim=1)
    product_mean = train_products.mean(dim=0)
    product_scale = train_products.std(dim=0, unbiased=False)
    product_scale = torch.where(product_scale <= 1e-12, torch.ones_like(product_scale), product_scale)
    train = torch.cat((train_z, (train_products - product_mean) / product_scale), dim=1)
    test = torch.cat((test_z, (test_products - product_mean) / product_scale), dim=1)
    return train, test, {
        "base_mean": mean,
        "base_scale": scale,
        "product_mean": product_mean,
        "product_scale": product_scale,
    }


def _fit_standardized_ridge(features: torch.Tensor, target: torch.Tensor, ridge: float) -> torch.Tensor:
    design = torch.cat((torch.ones(features.shape[0], 1, dtype=torch.float64), features), dim=1)
    penalty = torch.eye(design.shape[1], dtype=torch.float64)
    penalty[0, 0] = 0.0
    return torch.linalg.solve(design.T @ design + ridge * penalty, design.T @ target)


def _predict_standardized_ridge(theta: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    design = torch.cat((torch.ones(features.shape[0], 1, dtype=torch.float64), features), dim=1)
    return design @ theta


def _score(target: torch.Tensor, prediction: torch.Tensor, variance: float) -> tuple[torch.Tensor, torch.Tensor]:
    squared = (target - prediction).square()
    loss = 0.5 * math.log(2.0 * math.pi * variance) + squared / (2.0 * variance)
    return loss, squared


def _fit_models(
    fit_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    config: G2Config,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_fit = torch.tensor([row["rollout"]["target"] for row in fit_rows], dtype=torch.float64)
    target_test = torch.tensor([row["rollout"]["target"] for row in test_rows], dtype=torch.float64)
    learned = (
        "D", "D+g", "D+g_from_C", "D+C", "D+E", "D+perm", "D+Bpath",
        "D+Qraw", "D+Cterms", "D+Craw",
    )
    predictions: dict[str, torch.Tensor] = {}
    audits: dict[str, Any] = {}
    fitted_records: dict[str, tuple[_RidgeFit, torch.Tensor, torch.Tensor]] = {}
    for name in learned:
        train_feature = _stack_feature(fit_rows, name).detach().clone()
        test_feature = _stack_feature(test_rows, name).detach().clone()
        fitted = _fit_ridge(train_feature, target_fit, config.ridge)
        predictions[name] = _predict_ridge(fitted, test_feature)
        fitted_records[name] = (fitted, train_feature, test_feature)
        audits[name] = {
            "feature_dim": train_feature.shape[1],
            "coefficient_count": fitted.theta.numel(),
            "fit_rows": train_feature.shape[0],
            "test_rows": test_feature.shape[0],
            "feature_sha256": _sha_tensor(train_feature),
            "test_feature_sha256": _sha_tensor(test_feature),
            "mean_sha256": _sha_tensor(fitted.mean),
            "scale_sha256": _sha_tensor(fitted.scale),
            "theta_sha256": _sha_tensor(fitted.theta),
            "finite": bool(torch.isfinite(fitted.theta).all()),
        }
    train_base = _stack_feature(fit_rows, "D")
    test_base = _stack_feature(test_rows, "D")
    train_quadratic, test_quadratic, quadratic_scaler = _quadratic_features(train_base, test_base)
    d2_theta = _fit_standardized_ridge(train_quadratic, target_fit, config.ridge)
    predictions["D2"] = _predict_standardized_ridge(d2_theta, test_quadratic)
    audits["D2"] = {
        "feature_dim": train_quadratic.shape[1],
        "coefficient_count": d2_theta.numel(),
        "fit_rows": train_quadratic.shape[0],
        "test_rows": test_quadratic.shape[0],
        "feature_sha256": _sha_tensor(train_quadratic),
        "test_feature_sha256": _sha_tensor(test_quadratic),
        "theta_sha256": _sha_tensor(d2_theta),
        "scaler_sha256": _sha_text("|".join(
            _sha_tensor(quadratic_scaler[name]) for name in sorted(quadratic_scaler)
        )),
        "finite": bool(torch.isfinite(d2_theta).all()),
    }
    predictions["persistence"] = torch.tensor(
        [abs(float(row["rollout"]["y0"][1])) for row in test_rows], dtype=torch.float64,
    )
    predictions["global_mean"] = torch.full_like(target_test, float(target_fit.mean().item()))
    predictions["raw_Bpath"] = torch.tensor(
        [float(row["features"]["Bpath"][0]) for row in test_rows], dtype=torch.float64,
    )
    audits["persistence"] = {"feature_dim": 0, "coefficient_count": 0, "finite": True}
    audits["global_mean"] = {"feature_dim": 0, "coefficient_count": 1, "finite": True}
    audits["raw_Bpath"] = {"feature_dim": 1, "coefficient_count": 0, "finite": True}

    models: dict[str, Any] = {}
    for name, prediction in predictions.items():
        loss, squared = _score(target_test, prediction, config.score_variance)
        models[name] = {
            "mean_loss": float(loss.mean().item()),
            "mean_squared_error": float(squared.mean().item()),
            "predictions": [float(value) for value in prediction],
            "losses": [float(value) for value in loss],
            "audit": audits[name],
        }
    raw_g_fit = torch.stack([row["features"]["g"] for row in fit_rows]).detach().clone()
    copy_g_fit = torch.stack([row["features"]["g_from_C"] for row in fit_rows]).detach().clone()
    raw_g_test = torch.stack([row["features"]["g"] for row in test_rows]).detach().clone()
    copy_g_test = torch.stack([row["features"]["g_from_C"] for row in test_rows]).detach().clone()
    g_fit, g_train_features, g_test_features = fitted_records["D+g"]
    copy_fit, copy_train_features, copy_test_features = fitted_records["D+g_from_C"]
    g_train_standardized = (g_train_features - g_fit.mean) / g_fit.scale
    copy_train_standardized = (copy_train_features - copy_fit.mean) / copy_fit.scale
    g_test_standardized = (g_test_features - g_fit.mean) / g_fit.scale
    copy_test_standardized = (copy_test_features - copy_fit.mean) / copy_fit.scale
    no_repackaging = {
        "fit_raw_feature_residual": float((raw_g_fit - copy_g_fit).abs().max().item()),
        "test_raw_feature_residual": float((raw_g_test - copy_g_test).abs().max().item()),
        "fit_standardized_feature_residual": float(
            (g_train_standardized - copy_train_standardized).abs().max().item()
        ),
        "test_standardized_feature_residual": float(
            (g_test_standardized - copy_test_standardized).abs().max().item()
        ),
        "prediction_residual": float(
            (predictions["D+g"] - predictions["D+g_from_C"]).abs().max().item()
        ),
        "nonaliased_feature_arrays": bool(
            raw_g_fit.data_ptr() != copy_g_fit.data_ptr()
            and raw_g_test.data_ptr() != copy_g_test.data_ptr()
            and g_train_features.data_ptr() != copy_train_features.data_ptr()
            and g_test_features.data_ptr() != copy_test_features.data_ptr()
        ),
        "nonaliased_calibration_matrices": all(
            not bool(row["features"]["g_c_matrix_alias"][0]) for row in fit_rows + test_rows
        ),
    }
    return models, no_repackaging


def _serialize_rollout(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "control": row["control"].tolist(),
        "noise_start": row["noise_start"],
        "native_noise_seeds": row["native_noise_seeds"],
        "y0": row["y0"].tolist(),
        "path": row["path"].tolist(),
        "target": row["target"],
        "first_passage": row["first_passage"],
        "active_counts": row["active_counts"],
        "active_mask_sha256": row["active_mask_sha256"],
        "present_state_sha256": row["present_state_sha256"],
        "reset_parity": row["reset_parity"],
        "weight_unchanged": row["weight_unchanged"],
        "weight_sha256": row["weight_sha256"],
        "dense_sparse_parity": row["dense_sparse_parity"],
        "hippocampal_rows": row["hippocampal_rows"],
        "automatic_stdp_updates": row["automatic_stdp_updates"],
        "finite": row["finite"],
        "integrity": row["integrity"],
    }


def _serialize_dataset_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": row["split"],
        "row_index": row["row_index"],
        "environment_index": row["environment_index"],
        "environment": row["environment"],
        "local_rank": row["local_rank"],
        "direction": row["direction"].tolist(),
        "amplitude": row["amplitude"],
        "control": row["control"].tolist(),
        "rollout": _serialize_rollout(row["rollout"]),
        "feature_sha256": {
            name: _sha_tensor(value) for name, value in row["features"].items()
        },
        "features": {
            name: value.tolist() for name, value in row["features"].items()
        },
    }


def _serialize_calibration(calibration: dict[str, Any]) -> dict[str, Any]:
    return {
        "B_h": calibration["B_h"].tolist(),
        "B": calibration["B"].tolist(),
        "C": calibration["C"].tolist(),
        "g": calibration["g"].tolist(),
        "Q_raw": calibration["Q_raw"].tolist(),
        "g_transformed": calibration["g_transformed"].tolist(),
        "transform": calibration["transform"].tolist(),
        "transform_covariance_residual": calibration["transform_covariance_residual"],
        "transform_metric_residual": calibration["transform_metric_residual"],
        "eigenvalues_C": calibration["eigenvalues_C"].tolist(),
        "eigenvalues_g": calibration["eigenvalues_g"].tolist(),
        "eigenvalues_Q_raw": calibration["eigenvalues_Q_raw"].tolist(),
        "positive": [_serialize_rollout(row) for row in calibration["positive"]],
        "negative": [_serialize_rollout(row) for row in calibration["negative"]],
        "integrity": calibration["integrity"],
    }


def g2_metric_feature_utility(
    seed: int,
    config: G2Config | None = None,
) -> dict[str, Any]:
    config = config or G2Config(seed=seed)
    config = G2Config(**{**asdict(config), "seed": seed})
    frozen_protocol = _g2_frozen_protocol(config)
    base, injection, groups = _g2_fixture(seed, config)
    codebook = _g2_codebook(config)
    noise_schedule = _g2_noise_schedule(seed)
    snapshots: list[BrainRuntimeSnapshot] = []
    environment_audits: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    for environment_index, (gain, noise) in enumerate(G2_ENVIRONMENTS):
        snapshot, audit = _g2_environment_snapshot(base, gain, noise)
        snapshots.append(snapshot)
        environment_audits.append(audit)
        calibrations.append(_g2_calibrate(
            snapshot, injection, seed, environment_index, config,
        ))
    fit_rows = _g2_dataset(
        seed, snapshots, injection, calibrations, codebook, "fit", config,
    )
    test_rows = _g2_dataset(
        seed, snapshots, injection, calibrations, codebook, "test", config,
    )
    models, no_repackaging = _fit_models(fit_rows, test_rows, config)
    metric_loss = models["D+g"]["mean_loss"]
    deltas = {
        name: models[name]["mean_loss"] - metric_loss for name in G2_ADVERSE_MODELS
    }
    delta_min = min(deltas.values())
    transform_feature_residual = max(
        float(row["features"]["metric_invariance_residual"][0])
        for row in fit_rows + test_rows
    )
    cterms_formula_residual = max(
        float(abs(row["features"]["Cterms"].sum() - row["features"]["C"][0]).item())
        for row in fit_rows + test_rows
    )
    codebook_integrity = bool(
        codebook["audit"]["finite"]
        and codebook["audit"]["max_axis_alignment"] <= 0.95
        and codebook["audit"]["max_pair_alignment"] < 1.0 - 1e-10
        and codebook["audit"]["fit_test_pair_disjoint"]
    )
    coefficient_ledger = {
        name: models[name]["audit"]["coefficient_count"] for name in models
    }
    coefficient_integrity = bool(
        coefficient_ledger["D"] == 6
        and all(coefficient_ledger[name] == 7 for name in (
            "D+g", "D+g_from_C", "D+C", "D+E", "D+perm", "D+Bpath", "D+Qraw",
        ))
        and coefficient_ledger["D+Cterms"] == 12
        and coefficient_ledger["D+Craw"] == 12
        and coefficient_ledger["D2"] == 21
        and coefficient_ledger["persistence"] == 0
        and coefficient_ledger["global_mean"] == 1
        and coefficient_ledger["raw_Bpath"] == 0
    )
    no_repackaging_integrity = bool(
        no_repackaging["fit_raw_feature_residual"] <= 1e-10
        and no_repackaging["test_raw_feature_residual"] <= 1e-10
        and no_repackaging["fit_standardized_feature_residual"] <= 1e-10
        and no_repackaging["test_standardized_feature_residual"] <= 1e-10
        and no_repackaging["prediction_residual"] <= 1e-8
        and no_repackaging["nonaliased_feature_arrays"]
        and no_repackaging["nonaliased_calibration_matrices"]
    )
    integrity = bool(
        frozen_protocol
        and codebook_integrity
        and noise_schedule["pairwise_disjoint"]
        and noise_schedule["max_start_below_2e9"]
        and all(audit["weight_unchanged"] for audit in environment_audits)
        and len({audit["weight_sha256"] for audit in environment_audits}) == 1
        and all(audit["dense_sparse_parity"] for audit in environment_audits)
        and all(calibration["integrity"] for calibration in calibrations)
        and all(row["rollout"]["integrity"] for row in fit_rows + test_rows)
        and transform_feature_residual <= 1e-6
        and cterms_formula_residual <= 1e-10
        and no_repackaging_integrity
        and coefficient_integrity
        and all(model["audit"]["finite"] for model in models.values())
        and len(fit_rows) == config.fit_rows
        and len(test_rows) == config.test_rows
    )
    per_circuit_go = bool(integrity and delta_min > 0.0)
    return {
        "seed": seed,
        "route": "G2_fixed_weight_compressed_metric_feature_utility",
        "config": asdict(config),
        "frozen_protocol": frozen_protocol,
        "base_weight_sha256": _sha_tensor(base.weight),
        "coordinate_permutation": torch.cat((groups["S"], groups["T"], groups["N"])).tolist(),
        "injection_sha256": _sha_tensor(injection),
        "dedicated_g2_fixture": True,
        "environment_audits": environment_audits,
        "codebook_audit": codebook["audit"],
        "noise_schedule": noise_schedule,
        "calibrations": [_serialize_calibration(row) for row in calibrations],
        "fit_rows": [_serialize_dataset_row(row) for row in fit_rows],
        "test_rows": [_serialize_dataset_row(row) for row in test_rows],
        "models": models,
        "coefficient_ledger": coefficient_ledger,
        "no_repackaging": no_repackaging,
        "transform_feature_residual_max": transform_feature_residual,
        "cterms_formula_residual_max": cterms_formula_residual,
        "deltas_vs_metric": deltas,
        "delta_min": delta_min,
        "integrity": integrity,
        "status": "GO" if per_circuit_go else "STOP",
    }


def _g2_bootstrap_lcb(values: list[float], *, seed: int = 97598, samples: int = 10_000) -> float:
    if not values:
        raise ValueError("G2 bootstrap requires at least one circuit")
    data = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(0, len(values), (samples, len(values)), generator=generator)
    means = data[indices].mean(dim=1)
    return float(torch.quantile(means, 0.05).item())


def summarize_g2(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("G2 summary requires at least one circuit")
    delta_min = [float(row["delta_min"]) for row in results]
    go_count = sum(row["status"] == "GO" for row in results)
    lcb = _g2_bootstrap_lcb(delta_min)
    intervals = sorted(
        (entry["start"], entry["stop_inclusive"])
        for row in results for entry in row["noise_schedule"]["rows"]
    )
    all_noise_intervals_disjoint = all(
        first[1] < second[0] for first, second in zip(intervals, intervals[1:])
    )
    route_go = bool(
        all(row["integrity"] for row in results)
        and all_noise_intervals_disjoint
        and go_count / len(results) >= 0.80
        and lcb > 0.0
    )
    return {
        "circuit_count": len(results),
        "go_count": go_count,
        "go_fraction": go_count / len(results),
        "integrity_all": all(row["integrity"] for row in results),
        "noise_intervals_pairwise_disjoint_all_seeds": all_noise_intervals_disjoint,
        "delta_min_mean": sum(delta_min) / len(delta_min),
        "delta_min_min": min(delta_min),
        "delta_min_max": max(delta_min),
        "delta_min_bootstrap_lcb_95_one_sided": lcb,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 97598,
        "mean_delta_by_comparator": {
            name: sum(float(row["deltas_vs_metric"][name]) for row in results) / len(results)
            for name in G2_ADVERSE_MODELS
        },
        "route_verdict": "GO" if route_go else "STOP",
    }


def run_g2_seed_range(
    seeds: Iterable[int],
    *,
    config: G2Config | None = None,
) -> list[dict[str, Any]]:
    return [g2_metric_feature_utility(int(seed), config=config) for seed in seeds]
