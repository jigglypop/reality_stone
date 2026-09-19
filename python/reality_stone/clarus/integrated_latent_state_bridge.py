"""Development-only adaptive causal belief-state bridge.

This module never opens a registered test split.  It fits stable residual
moments from inherited training episodes, adapts observation geometry from an
immutable prefix, filters a Gaussian belief state, and injects its posterior
mean inside each free-rollout transition.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import free_rollout_bridge as free
from . import latent_causal_bridge as latent
from . import parent_anchored_rollout_bridge as v8
from . import reliability_rollout_bridge as rel
from . import sparse_causal_bridge as base


DEVELOPMENT_SEEDS = tuple(range(82100, 82356))
CRITICAL_T_DF255 = 1.9693105698498752
MAX_LAG = 12
FAST_SIGNAL_FRACTION_MIN = 0.05
MODEL_NAMES = (
    "acbsm_sparse_rank2",
    "acbsm_sparse_rank1",
    "acbsm_dense_rank2",
    "acbsm_zero_bridge_rank2",
    "v5_sparse_parent",
    "persistence",
    "v8_r1",
)


@dataclass(frozen=True)
class ResidualDynamics:
    center: np.ndarray
    transition: np.ndarray
    training_loading: np.ndarray
    process_covariance: np.ndarray
    training_observation_covariance: np.ndarray
    rank: int
    fast_active: bool
    fast_signal_fraction: float
    fast_fold_support: float
    moment_fit_error: float


@dataclass(frozen=True)
class PrefixObservationGeometry:
    loading: np.ndarray
    observation_covariance: np.ndarray
    signal_eigenvalues: np.ndarray


@dataclass(frozen=True)
class FilteredResidualState:
    mean: np.ndarray
    covariance: np.ndarray
    geometry: PrefixObservationGeometry
    minimum_covariance_eigenvalue: float
    maximum_innovation_mahalanobis: float


@dataclass(frozen=True)
class ACBSMContext:
    v8_context: v8.V8TrainingContext
    sparse_rank2: ResidualDynamics
    sparse_rank1: ResidualDynamics
    dense_rank2: ResidualDynamics
    zero_rank2: ResidualDynamics


@dataclass(frozen=True)
class ACBSMPredictions:
    models: dict[str, np.ndarray]
    posterior: dict[str, dict]
    pathwise_jacobian_radii: dict[str, float]
    maximum_covariance_negative_eigenvalue: float


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sign(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=float).copy()
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result *= -1.0
    return result


def _project_rank_one_psd(matrix: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    values, vectors = np.linalg.eigh(symmetric)
    value = max(float(values[-1]), 0.0)
    vector = _canonical_sign(vectors[:, -1])
    return value * np.outer(vector, vector), value, vector


def _residual_sequences(
    episodes: Sequence[base.Episode], mechanism: base.BridgeModel
) -> list[np.ndarray]:
    return [
        episode.states[1:] - mechanism.predict(episode.states[:-1])
        for episode in episodes
    ]


def _lag_covariances(
    sequences: Sequence[np.ndarray], center: np.ndarray, maximum_lag: int = MAX_LAG
) -> tuple[np.ndarray, ...]:
    centered = [np.asarray(values, dtype=float) - center for values in sequences]
    result = []
    for lag in range(maximum_lag + 1):
        products = []
        for values in centered:
            if len(values) <= lag:
                raise ValueError("residual sequence is shorter than the registered lag")
            products.append(values[lag:].T @ values[: len(values) - lag])
        denominator = sum(len(values) - lag for values in centered)
        covariance = sum(products) / denominator
        result.append(0.5 * (covariance + covariance.T))
    return tuple(result)


def _moment_error(
    covariances: Sequence[np.ndarray], components: Sequence[tuple[np.ndarray, float]]
) -> float:
    numerator = 0.0
    denominator = 0.0
    for lag in range(1, min(len(covariances), MAX_LAG + 1)):
        reconstructed = sum(matrix * pole**lag for matrix, pole in components)
        numerator += float(np.sum((covariances[lag] - reconstructed) ** 2))
        denominator += float(np.sum(covariances[lag] ** 2))
    return numerator / max(denominator, 1e-15)


def _fit_rank_one_moments(
    covariances: Sequence[np.ndarray], pole_grid: np.ndarray
) -> tuple[float, np.ndarray, float]:
    best = None
    for pole in pole_grid:
        matrix, _, _ = _project_rank_one_psd(covariances[1] / pole)
        error = _moment_error(covariances, ((matrix, float(pole)),))
        if best is None or error < best[0]:
            best = (error, float(pole), matrix)
    assert best is not None
    return best[1], best[2], best[0]


def _fit_rank_two_moments(
    covariances: Sequence[np.ndarray],
) -> tuple[float, float, np.ndarray, np.ndarray, float]:
    best = None
    for fast in np.linspace(0.05, 0.80, 31):
        for slow in np.linspace(0.82, 0.98, 33):
            if slow - fast < 0.10:
                continue
            fast_raw = (slow * covariances[1] - covariances[2]) / (
                fast * (slow - fast)
            )
            slow_raw = (covariances[2] - fast * covariances[1]) / (
                slow * (slow - fast)
            )
            fast_matrix, _, _ = _project_rank_one_psd(fast_raw)
            slow_matrix, _, _ = _project_rank_one_psd(slow_raw)
            error = _moment_error(
                covariances,
                ((fast_matrix, float(fast)), (slow_matrix, float(slow))),
            )
            if best is None or error < best[0]:
                best = (error, float(fast), float(slow), fast_matrix, slow_matrix)
    assert best is not None
    return best[1], best[2], best[3], best[4], best[0]


def _loading_from_signal_matrices(matrices: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    columns = []
    eigenvalues = []
    for matrix in matrices:
        _, value, vector = _project_rank_one_psd(matrix)
        columns.append(math.sqrt(max(value, 0.0)) * vector)
        eigenvalues.append(value)
    return np.column_stack(columns), np.asarray(eigenvalues, dtype=float)


def fit_residual_dynamics(
    episodes: Sequence[base.Episode], mechanism: base.BridgeModel, *, rank: int
) -> ResidualDynamics:
    if rank not in {1, 2}:
        raise ValueError("the development model supports rank one or two")
    sequences = _residual_sequences(episodes, mechanism)
    center = np.mean(np.concatenate(sequences, axis=0), axis=0)
    covariances = _lag_covariances(sequences, center)
    if rank == 1:
        slow, slow_matrix, error = _fit_rank_one_moments(
            covariances, np.linspace(0.80, 0.98, 37)
        )
        matrices = (slow_matrix,)
        transition = np.asarray([slow], dtype=float)
        fast_active = False
        fast_fraction = 0.0
    else:
        fast, slow, fast_matrix, slow_matrix, error = _fit_rank_two_moments(covariances)
        fast_trace = max(float(np.trace(fast_matrix)), 0.0)
        slow_trace = max(float(np.trace(slow_matrix)), 0.0)
        fast_fraction = fast_trace / max(fast_trace + slow_trace, 1e-15)
        fast_active = fast_fraction >= FAST_SIGNAL_FRACTION_MIN
        if not fast_active:
            # Automatic rank-one collapse is part of the frozen model, not a
            # post-evaluation route change.
            slow, slow_matrix, error = _fit_rank_one_moments(
                covariances, np.linspace(0.80, 0.98, 37)
            )
            fast_matrix = np.zeros_like(slow_matrix)
            fast = min(0.80, slow - 0.10)
        matrices = (fast_matrix, slow_matrix)
        transition = np.asarray([fast, slow], dtype=float)
    loading, _ = _loading_from_signal_matrices(matrices)
    signal = sum(matrices)
    residual_diagonal = np.maximum(np.diag(covariances[0] - signal), 1e-6)
    observation_covariance = np.diag(residual_diagonal)
    process_covariance = np.diag(np.maximum(1.0 - transition * transition, 1e-6))
    if np.any(np.abs(transition) >= 0.98 + 1e-15):
        raise FloatingPointError("residual pole violates the development bound")
    return ResidualDynamics(
        center=center,
        transition=transition,
        training_loading=loading,
        process_covariance=process_covariance,
        training_observation_covariance=observation_covariance,
        rank=rank,
        fast_active=fast_active,
        fast_signal_fraction=fast_fraction,
        fast_fold_support=1.0 if fast_active else 0.0,
        moment_fit_error=float(error),
    )


def fit_stable_rank_two_dynamics(
    episodes: Sequence[base.Episode], mechanism: base.BridgeModel
) -> ResidualDynamics:
    """Require the optional fast mode to reproduce across episode folds."""

    raw = fit_residual_dynamics(episodes, mechanism, rank=2)
    rank_one = fit_residual_dynamics(episodes, mechanism, rank=1)
    fold_models = [
        fit_residual_dynamics(
            [episode for index, episode in enumerate(episodes) if index != held],
            mechanism,
            rank=2,
        )
        for held in range(len(episodes))
    ]
    support = float(np.mean([model.fast_active for model in fold_models]))
    active_fast_poles = [
        float(model.transition[0]) for model in fold_models if model.fast_active
    ]
    pole_mad = (
        float(
            np.median(
                np.abs(
                    np.asarray(active_fast_poles, dtype=float)
                    - np.median(active_fast_poles)
                )
            )
        )
        if active_fast_poles
        else float("inf")
    )
    stable = raw.fast_active and support >= 0.75 and pole_mad <= 0.08
    if stable:
        return ResidualDynamics(
            **{**raw.__dict__, "fast_fold_support": support}
        )
    slow = float(rank_one.transition[0])
    fast = min(0.80, slow - 0.10)
    transition = np.asarray([fast, slow], dtype=float)
    loading = np.column_stack((np.zeros(4), rank_one.training_loading[:, 0]))
    return ResidualDynamics(
        center=rank_one.center,
        transition=transition,
        training_loading=loading,
        process_covariance=np.diag(np.maximum(1.0 - transition * transition, 1e-6)),
        training_observation_covariance=rank_one.training_observation_covariance,
        rank=2,
        fast_active=False,
        fast_signal_fraction=raw.fast_signal_fraction,
        fast_fold_support=support,
        moment_fit_error=rank_one.moment_fit_error,
    )


def _prefix_geometry(
    residuals: np.ndarray, dynamics: ResidualDynamics
) -> PrefixObservationGeometry:
    covariances = _lag_covariances((residuals,), dynamics.center, maximum_lag=2)
    if dynamics.rank == 1 or not dynamics.fast_active:
        pole = float(dynamics.transition[-1])
        slow_matrix, _, _ = _project_rank_one_psd(covariances[1] / pole)
        if dynamics.rank == 1:
            matrices = (slow_matrix,)
        else:
            matrices = (np.zeros_like(slow_matrix), slow_matrix)
    else:
        fast, slow = map(float, dynamics.transition)
        fast_raw = (slow * covariances[1] - covariances[2]) / (
            fast * (slow - fast)
        )
        slow_raw = (covariances[2] - fast * covariances[1]) / (
            slow * (slow - fast)
        )
        fast_matrix, _, _ = _project_rank_one_psd(fast_raw)
        slow_matrix, _, _ = _project_rank_one_psd(slow_raw)
        matrices = (fast_matrix, slow_matrix)
    loading, values = _loading_from_signal_matrices(matrices)
    signal = sum(matrices)
    noise = np.maximum(np.diag(covariances[0] - signal), 1e-6)
    return PrefixObservationGeometry(
        loading=loading,
        observation_covariance=np.diag(noise),
        signal_eigenvalues=values,
    )


def filter_prefix(
    prefix_states: np.ndarray,
    mechanism: base.BridgeModel,
    dynamics: ResidualDynamics,
) -> FilteredResidualState:
    prefix = np.asarray(prefix_states, dtype=float)
    if prefix.ndim != 2 or len(prefix) < 5:
        raise ValueError("prefix must contain at least five observed states")
    residuals = prefix[1:] - mechanism.predict(prefix[:-1])
    geometry = _prefix_geometry(residuals, dynamics)
    transition = np.diag(dynamics.transition)
    mean = np.zeros(dynamics.rank, dtype=float)
    covariance = np.eye(dynamics.rank, dtype=float)
    identity = np.eye(dynamics.rank, dtype=float)
    minimum_eigenvalue = float("inf")
    maximum_mahalanobis = 0.0
    for observation in residuals:
        mean = transition @ mean
        covariance = transition @ covariance @ transition.T + dynamics.process_covariance
        innovation = observation - dynamics.center - geometry.loading @ mean
        innovation_covariance = (
            geometry.loading @ covariance @ geometry.loading.T
            + geometry.observation_covariance
        )
        inverse = np.linalg.pinv(innovation_covariance, hermitian=True)
        gain = covariance @ geometry.loading.T @ inverse
        mean = mean + gain @ innovation
        update = identity - gain @ geometry.loading
        covariance = (
            update @ covariance @ update.T
            + gain @ geometry.observation_covariance @ gain.T
        )
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
        minimum_eigenvalue = min(minimum_eigenvalue, eigenvalue)
        maximum_mahalanobis = max(
            maximum_mahalanobis, float(innovation @ inverse @ innovation)
        )
    if minimum_eigenvalue < -1e-10:
        raise FloatingPointError("posterior covariance is not positive semidefinite")
    return FilteredResidualState(
        mean=mean,
        covariance=covariance,
        geometry=geometry,
        minimum_covariance_eigenvalue=minimum_eigenvalue,
        maximum_innovation_mahalanobis=maximum_mahalanobis,
    )


def rollout_from_belief(
    x_anchor: np.ndarray,
    mechanism: base.BridgeModel,
    dynamics: ResidualDynamics,
    belief: FilteredResidualState,
    horizon: int,
    *,
    feed_correction_back: bool = True,
) -> tuple[np.ndarray, list[np.ndarray]]:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    transition = np.diag(dynamics.transition)
    state = np.asarray(x_anchor, dtype=float).copy()
    mechanism_state = state.copy()
    mean = belief.mean.copy()
    covariance = belief.covariance.copy()
    predictions = []
    covariances = []
    for _ in range(horizon):
        mean = transition @ mean
        covariance = transition @ covariance @ transition.T + dynamics.process_covariance
        correction = dynamics.center + belief.geometry.loading @ mean
        base_prediction = mechanism.predict(mechanism_state)[0]
        following = base_prediction + correction
        predictions.append(following.copy())
        covariances.append(
            belief.geometry.loading @ covariance @ belief.geometry.loading.T
            + belief.geometry.observation_covariance
        )
        state = following
        mechanism_state = state if feed_correction_back else base_prediction
    return np.asarray(predictions), covariances


def _zero_bridge(parent: rel.TrainingContext) -> base.BridgeModel:
    return latent.mechanism_model(
        "zero_bridge_acbsm",
        parent.sparse_mechanism.local_coefficients[:, 1],
        np.zeros_like(parent.sparse_mechanism.bridge),
        (),
    )


def build_context(config_path: Path, registration: dict) -> ACBSMContext:
    v8_context = v8._build_training_context(config_path, registration)
    role = registration["data_roles"]["observational_train"]
    episodes = [
        base.simulate_episode(
            int(seed), registration, environment=role["environment"],
            steps=int(role["steps_per_seed"])
        )
        for seed in role["seeds"]
    ]
    zero = _zero_bridge(v8_context.parent)
    return ACBSMContext(
        v8_context=v8_context,
        sparse_rank2=fit_stable_rank_two_dynamics(
            episodes, v8_context.parent.sparse_mechanism
        ),
        sparse_rank1=fit_residual_dynamics(
            episodes, v8_context.parent.sparse_mechanism, rank=1
        ),
        dense_rank2=fit_stable_rank_two_dynamics(
            episodes, v8_context.parent.dense_probe_mechanism
        ),
        zero_rank2=fit_stable_rank_two_dynamics(episodes, zero),
    )


def _one_acbsm_prediction(
    prefix: np.ndarray,
    mechanism: base.BridgeModel,
    dynamics: ResidualDynamics,
    horizon: int,
) -> tuple[np.ndarray, FilteredResidualState, list[np.ndarray]]:
    belief = filter_prefix(prefix, mechanism, dynamics)
    prediction, covariances = rollout_from_belief(
        prefix[-1], mechanism, dynamics, belief, horizon
    )
    return prediction, belief, covariances


def predict_from_prefix(
    prefix_states: np.ndarray, context: ACBSMContext, registration: dict
) -> ACBSMPredictions:
    prefix = np.asarray(prefix_states, dtype=float)
    origin = int(registration["parent_anchor"]["origin"])
    horizon = int(registration["parent_anchor"]["horizon"])
    if prefix.shape != (origin + 1, len(context.v8_context.parent.scales)):
        raise ValueError("prefix must end exactly at the frozen origin")
    prefix = prefix.copy()
    prefix.setflags(write=False)
    parent = context.v8_context.parent
    zero = _zero_bridge(parent)
    sparse2, sparse2_belief, sparse2_cov = _one_acbsm_prediction(
        prefix, parent.sparse_mechanism, context.sparse_rank2, horizon
    )
    sparse1, sparse1_belief, sparse1_cov = _one_acbsm_prediction(
        prefix, parent.sparse_mechanism, context.sparse_rank1, horizon
    )
    dense2, dense2_belief, dense2_cov = _one_acbsm_prediction(
        prefix, parent.dense_probe_mechanism, context.dense_rank2, horizon
    )
    zero2, zero2_belief, zero2_cov = _one_acbsm_prediction(
        prefix, zero, context.zero_rank2, horizon
    )
    legacy = rel._latent_rollout(
        parent.sparse_mechanism, parent.sparse_ar, prefix, horizon
    )
    persistence = np.repeat(prefix[-1][None, :], horizon, axis=0)
    r1 = persistence + context.v8_context.sparse_gain * (legacy - persistence)
    models = {
        "acbsm_sparse_rank2": sparse2,
        "acbsm_sparse_rank1": sparse1,
        "acbsm_dense_rank2": dense2,
        "acbsm_zero_bridge_rank2": zero2,
        "v5_sparse_parent": legacy,
        "persistence": persistence,
        "v8_r1": r1,
    }
    beliefs = {
        "sparse_rank2": (sparse2_belief, sparse2_cov, context.sparse_rank2),
        "sparse_rank1": (sparse1_belief, sparse1_cov, context.sparse_rank1),
        "dense_rank2": (dense2_belief, dense2_cov, context.dense_rank2),
        "zero_rank2": (zero2_belief, zero2_cov, context.zero_rank2),
    }
    posterior = {}
    maximum_negative = 0.0
    for name, (belief, covariances, dynamics) in beliefs.items():
        forecast_minimum = min(float(np.min(np.linalg.eigvalsh(value))) for value in covariances)
        minimum = min(belief.minimum_covariance_eigenvalue, forecast_minimum)
        maximum_negative = max(maximum_negative, max(0.0, -minimum))
        posterior[name] = {
            "mean": belief.mean.tolist(),
            "covariance": belief.covariance.tolist(),
            "transition": dynamics.transition.tolist(),
            "fast_active": dynamics.fast_active,
            "fast_signal_fraction": dynamics.fast_signal_fraction,
            "prefix_signal_eigenvalues": belief.geometry.signal_eigenvalues.tolist(),
            "minimum_covariance_eigenvalue": minimum,
            "maximum_innovation_mahalanobis": belief.maximum_innovation_mahalanobis,
            "forecast_trace": [float(np.trace(value)) for value in covariances],
        }
    radii = {
        "sparse_rank2": max(
            free._maximum_jacobian_radius(parent.sparse_mechanism, sparse2),
            float(np.max(np.abs(context.sparse_rank2.transition))),
        ),
        "sparse_rank1": max(
            free._maximum_jacobian_radius(parent.sparse_mechanism, sparse1),
            float(np.max(np.abs(context.sparse_rank1.transition))),
        ),
        "dense_rank2": max(
            free._maximum_jacobian_radius(parent.dense_probe_mechanism, dense2),
            float(np.max(np.abs(context.dense_rank2.transition))),
        ),
        "zero_rank2": max(
            free._maximum_jacobian_radius(zero, zero2),
            float(np.max(np.abs(context.zero_rank2.transition))),
        ),
    }
    return ACBSMPredictions(models, posterior, radii, maximum_negative)


def _paired_lower(baseline: Sequence[float], candidate: Sequence[float]) -> dict[str, float]:
    values = np.asarray(baseline) - np.asarray(candidate)
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    half = CRITICAL_T_DF255 * sd / math.sqrt(len(values))
    return {
        "mean_improvement": mean,
        "sample_sd": sd,
        "ci95_lower": mean - half,
        "ci95_upper": mean + half,
        "seed_win_fraction": float(np.mean(values > 0.0)),
    }


def _paired_log_upper(candidate: Sequence[float], baseline: Sequence[float]) -> dict[str, float]:
    values = np.log(np.asarray(candidate) / np.asarray(baseline))
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    half = CRITICAL_T_DF255 * sd / math.sqrt(len(values))
    return {
        "mean_log_ratio": mean,
        "ci95_lower": mean - half,
        "ci95_upper": mean + half,
        "geometric_mean_ratio": math.exp(mean),
    }


def _raw_historical_seeds(config_path: Path) -> set[int]:
    seeds: set[int] = set()
    for path in sorted(config_path.parent.glob("sparse_causal_bridge_v*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        roles = raw.get("overrides", {}).get("data_roles", raw.get("data_roles", {}))
        for role in roles.values():
            if isinstance(role, dict):
                seeds.update(map(int, role.get("seeds", [])))
    seeds.update(range(79100, 79356))
    return seeds


def _implementation_hashes(root: Path) -> dict[str, str]:
    paths = {
        "integrated_latent_state_bridge.py": Path(__file__).resolve(),
        "parent_anchored_rollout_bridge.py": Path(v8.__file__).resolve(),
        "reliability_rollout_bridge.py": Path(rel.__file__).resolve(),
        "free_rollout_bridge.py": Path(free.__file__).resolve(),
        "latent_causal_bridge.py": Path(latent.__file__).resolve(),
        "sparse_causal_bridge.py": Path(base.__file__).resolve(),
        "test_integrated_latent_state_bridge.py": root / "tests" / "test_integrated_latent_state_bridge.py",
    }
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError("ACBSM lock surface is incomplete")
    return {name: _sha256(path) for name, path in paths.items()}


def _score_promise(
    comparisons: dict, stability_ok: bool, integrity_ok: bool,
    context: ACBSMContext, dense_ok: bool
) -> dict:
    v5_lower = comparisons["vs_v5_parent"]["ci95_lower"]
    points = {
        "v5_transfer": 30.0 * float(np.clip(v5_lower / 0.005, 0.0, 1.0)),
        "persistence": 10.0 if comparisons["vs_persistence"]["ci95_lower"] > 0.0 else 0.0,
        "zero_bridge": 10.0 if comparisons["vs_zero_bridge"]["ci95_lower"] > 0.0 else 0.0,
        "rank_two_contribution": 10.0 if comparisons["vs_rank_one"]["ci95_lower"] > 0.0 else 0.0,
        "two_modes_identified": 5.0 if context.sparse_rank2.fast_active else 0.0,
        "dense_symmetry": 10.0 if dense_ok else 0.0,
        "stability": 10.0 if stability_ok else 0.0,
        "integrity": 10.0 if integrity_ok else 0.0,
        "parsimony": 5.0,
    }
    total = float(sum(points.values()))
    if not stability_ok or not integrity_ok:
        classification = "REJECT"
    elif total >= 75.0 and v5_lower >= 0.005:
        classification = "PROMISING"
    elif total >= 60.0:
        classification = "HOLD"
    else:
        classification = "REJECT"
    return {"total": total, "classification": classification, "points": points}


def run_development(
    config_path: Path, *, lock_path: Path, output_path: Path
) -> dict:
    """Consume the frozen development block once and save its full paired report."""

    if output_path.exists():
        raise FileExistsError("ACBSM development block was already consumed")
    started = time.perf_counter()
    registration, _ = base._load_registration(config_path)
    base._validate_registration(registration)
    if set(DEVELOPMENT_SEEDS) & _raw_historical_seeds(config_path):
        raise PermissionError("ACBSM development seeds overlap historical roles")
    root = config_path.resolve().parents[2]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    current_hashes = _implementation_hashes(root)
    if lock.get("implementation_sha256") != current_hashes:
        raise PermissionError("ACBSM implementation differs from the development lock")
    context = build_context(config_path, registration)
    errors = {name: [] for name in MODEL_NAMES}
    maximum_observed = -1
    future_reads = 0
    nonfinite = 0
    maximum_radius = 0.0
    maximum_covariance_negative = 0.0
    h5_exact = True
    posterior_rows = []
    for seed in DEVELOPMENT_SEEDS:
        episode = base.simulate_episode(seed, registration, environment="ood", steps=100)
        reader = rel.PrefixReader(episode.states, 80)
        predictions = predict_from_prefix(reader.through_origin(), context, registration)
        maximum_observed = max(maximum_observed, reader.max_observed_state_index)
        future_reads += reader.future_observation_reads
        maximum_radius = max(maximum_radius, *predictions.pathwise_jacobian_radii.values())
        maximum_covariance_negative = max(
            maximum_covariance_negative, predictions.maximum_covariance_negative_eigenvalue
        )
        truth = episode.states[81:101]
        for name, prediction in predictions.models.items():
            nonfinite += int(prediction.size - np.count_nonzero(np.isfinite(prediction)))
            h5_exact &= np.array_equal(prediction[:5], prediction[0:5])
            errors[name].append(
                rel._normalized_rmse(truth, prediction, context.v8_context.parent.scales)
            )
        posterior_rows.append(predictions.posterior)
    candidate = errors["acbsm_sparse_rank2"]
    comparisons = {
        "vs_v5_parent": _paired_lower(errors["v5_sparse_parent"], candidate),
        "vs_persistence": _paired_lower(errors["persistence"], candidate),
        "vs_zero_bridge": _paired_lower(errors["acbsm_zero_bridge_rank2"], candidate),
        "vs_rank_one": _paired_lower(errors["acbsm_sparse_rank1"], candidate),
        "vs_v8_r1": _paired_lower(errors["v8_r1"], candidate),
        "log_ratio_vs_dense": _paired_log_upper(candidate, errors["acbsm_dense_rank2"]),
    }
    stability_ok = (
        maximum_radius <= 0.98
        and maximum_covariance_negative <= 1e-10
        and nonfinite == 0
    )
    integrity_ok = (
        maximum_observed <= 80 and future_reads == 0 and h5_exact
        and not (set(DEVELOPMENT_SEEDS) & _raw_historical_seeds(config_path))
    )
    dense_ok = comparisons["log_ratio_vs_dense"]["ci95_upper"] <= math.log(1.02)
    promise = _score_promise(comparisons, stability_ok, integrity_ok, context, dense_ok)
    end_hashes = _implementation_hashes(root)
    if end_hashes != current_hashes:
        raise PermissionError("ACBSM implementation changed during development execution")
    report = {
        "status": "development_only_direction_score",
        "model": "adaptive_causal_belief_state_model_core",
        "v9_registered": False,
        "v8_locked_test_opened": False,
        "development_seeds": {
            "first": DEVELOPMENT_SEEDS[0], "last": DEVELOPMENT_SEEDS[-1],
            "count": len(DEVELOPMENT_SEEDS), "historical_overlap": []
        },
        "promise_score": promise,
        "models": {
            name: {
                "mean_h20_normalized_path_rmse": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)),
                "seed_h20_normalized_path_rmse": values,
            }
            for name, values in errors.items()
        },
        "comparisons": comparisons,
        "dynamics": {
            name: {
                "transition": dynamics.transition.tolist(),
                "rank": dynamics.rank,
                "fast_active": dynamics.fast_active,
                "fast_signal_fraction": dynamics.fast_signal_fraction,
                "fast_fold_support": dynamics.fast_fold_support,
                "moment_fit_error": dynamics.moment_fit_error,
            }
            for name, dynamics in {
                "sparse_rank2": context.sparse_rank2,
                "sparse_rank1": context.sparse_rank1,
                "dense_rank2": context.dense_rank2,
                "zero_rank2": context.zero_rank2,
            }.items()
        },
        "stability": {
            "passed": stability_ok,
            "maximum_augmented_pathwise_radius": maximum_radius,
            "maximum_covariance_negative_eigenvalue": maximum_covariance_negative,
            "nonfinite_prediction_count": nonfinite,
        },
        "integrity": {
            "passed": integrity_ok,
            "maximum_observed_state_index": maximum_observed,
            "future_observation_reads": future_reads,
            "h5_exact_h20_slice": h5_exact,
            "implementation_sha256": current_hashes,
        },
        "posterior_per_seed": posterior_rows,
        "resource_usage": {
            "wall_seconds": time.perf_counter() - started,
            "evaluation_seeds": len(DEVELOPMENT_SEEDS),
            "external_download_bytes": 0,
        },
        "environment_manifest": {
            "python": sys.version, "numpy": np.__version__, "platform": platform.platform()
        },
        "claim_boundary": (
            "Development route-priority score in one four-chart synthetic H20 family; "
            "not confirmation, AGI evidence, or a general world-model result."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "ACBSMContext",
    "ACBSMPredictions",
    "DEVELOPMENT_SEEDS",
    "FilteredResidualState",
    "ResidualDynamics",
    "build_context",
    "filter_prefix",
    "fit_residual_dynamics",
    "fit_stable_rank_two_dynamics",
    "predict_from_prefix",
    "rollout_from_belief",
    "run_development",
]
