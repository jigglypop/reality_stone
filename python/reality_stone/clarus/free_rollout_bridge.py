"""Locked single-origin free-rollout extension of the V4 causal bridge gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from . import latent_causal_bridge as latent
from . import sparse_causal_bridge as base


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_hashes() -> dict[str, str]:
    paths = {
        "free_rollout_bridge.py": Path(__file__).resolve(),
        "latent_causal_bridge.py": Path(latent.__file__).resolve(),
        "sparse_causal_bridge.py": Path(base.__file__).resolve(),
    }
    return {name: _sha256(path) for name, path in paths.items()}


def fit_prefix_residual_filter(
    prefix_states: np.ndarray,
    mechanism: base.BridgeModel,
    autoregression: float,
) -> latent.ResidualFilter:
    """Fit OOD residual geometry using only an observed calibration prefix."""

    states = np.asarray(prefix_states, dtype=float)
    if states.ndim != 2 or len(states) < 5:
        raise ValueError("prefix_states must contain at least five observed states")
    residuals = states[1:] - mechanism.predict(states[:-1])
    center = np.mean(residuals, axis=0)
    centered = residuals - center
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, -1]
    variance_fraction = float(eigenvalues[-1] / max(np.sum(eigenvalues), 1e-12))
    scores = centered @ direction
    intercept = float(np.mean(scores[1:] - autoregression * scores[:-1]))
    return latent.ResidualFilter(
        center=center,
        direction=direction,
        intercept=intercept,
        autoregression=float(autoregression),
        variance_fraction=variance_fraction,
    )


def free_rollout(
    mechanism: base.BridgeModel,
    *,
    x_previous: np.ndarray,
    x_anchor: np.ndarray,
    horizon: int,
    residual_filter: latent.ResidualFilter | None = None,
) -> np.ndarray:
    """Predict without accepting an Episode, future states, outcomes, or hidden state."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    previous = np.asarray(x_previous, dtype=float)
    state = np.asarray(x_anchor, dtype=float).copy()
    if previous.shape != state.shape or state.ndim != 1:
        raise ValueError("x_previous and x_anchor must be same-shape vectors")
    previous_residual = state - mechanism.predict(previous)[0]
    predictions = []
    for _ in range(horizon):
        context = (
            np.zeros_like(state)
            if residual_filter is None
            else residual_filter.predict_next(previous_residual)
        )
        with np.errstate(over="ignore", invalid="ignore"):
            following = mechanism.predict(state)[0] + context
        predictions.append(following.copy())
        state = following
        previous_residual = context
    return np.asarray(predictions)


def _oracle_rollout(
    mechanism: base.BridgeModel,
    *,
    x_anchor: np.ndarray,
    hidden_anchor: float,
    loading: np.ndarray,
    latent_ar: float,
    horizon: int,
) -> np.ndarray:
    state = np.asarray(x_anchor, dtype=float).copy()
    hidden = float(hidden_anchor)
    predictions = []
    for _ in range(horizon):
        following = mechanism.predict(state)[0] + loading * hidden
        predictions.append(following.copy())
        state = following
        hidden *= latent_ar
    return np.asarray(predictions)


def fit_stable_observational_model(
    name: str,
    episodes: Sequence[base.Episode],
    edges: Sequence[base.Edge],
    ridge: float,
) -> base.BridgeModel:
    """Fit [1, own-state, cross tanh] without a cubic rollout term."""

    states, outcomes = base._stack_episodes(episodes)
    size = states.shape[1]
    edge_set = tuple(sorted(set(edges)))
    local = np.zeros((size, 3), dtype=float)
    bridge = np.zeros((size, size), dtype=float)
    for target in range(size):
        incoming = sorted(source for source, destination in edge_set if destination == target)
        design = np.column_stack((np.ones(len(states)), states[:, target]))
        if incoming:
            design = np.column_stack((design, np.tanh(states[:, incoming])))
        coefficients = base._ridge(design, outcomes[:, target], ridge)
        local[target, 0] = coefficients[0]
        local[target, 1] = coefficients[1]
        for source, coefficient in zip(incoming, coefficients[2:]):
            bridge[target, source] = float(coefficient)
    return base.BridgeModel(name, local, bridge, edge_set)


def _parse_edge(label: str, chart_index: dict[str, int]) -> base.Edge:
    source, target = label.split("->")
    return chart_index[source], chart_index[target]


def _mechanisms_from_parent(
    registration: dict, parent: dict
) -> tuple[base.BridgeModel, base.BridgeModel]:
    charts = registration["charts"]
    chart_index = {name: index for index, name in enumerate(charts)}
    selected = tuple(
        _parse_edge(label, chart_index) for label in parent["selection"]["causal_edges"]
    )
    self_coefficients = np.asarray(
        parent["selection"]["self_coefficient_estimates"], dtype=float
    )
    diagnostics = parent["selection"]["full_intervention_diagnostics"]
    size = len(charts)
    sparse_bridge = np.zeros((size, size), dtype=float)
    dense_bridge = np.zeros((size, size), dtype=float)
    for source in range(size):
        for target in range(size):
            if source == target:
                continue
            key = f"{charts[source]}->{charts[target]}"
            dense_bridge[target, source] = float(diagnostics[key]["estimate"])
    for source, target in selected:
        key = f"{charts[source]}->{charts[target]}"
        sparse_bridge[target, source] = float(diagnostics[key]["estimate"])
    sparse = latent.mechanism_model(
        "causal_mechanism", self_coefficients, sparse_bridge, selected
    )
    all_edges = tuple(
        (source, target)
        for source in range(size)
        for target in range(size)
        if source != target
    )
    dense = latent.mechanism_model(
        "same_probe_dense_mechanism", self_coefficients, dense_bridge, all_edges
    )
    return sparse, dense


def _load_frozen_parent(
    config_path: Path, registration: dict
) -> tuple[dict, str]:
    root = config_path.resolve().parents[2]
    path = root / registration["rollout"]["frozen_parent_validation_artifact"]
    expected = registration["rollout"]["frozen_parent_validation_artifact_sha256"]
    actual = _sha256(path)
    if actual != expected:
        raise PermissionError("frozen V4 parent artifact SHA256 changed")
    parent = json.loads(path.read_text(encoding="utf-8"))
    if not parent.get("passed"):
        raise PermissionError("frozen V4 parent validation did not pass")
    legacy_hashes = {
        key: value
        for key, value in _implementation_hashes().items()
        if key != "free_rollout_bridge.py"
    }
    if parent.get("implementation_sha256") != legacy_hashes:
        raise PermissionError("V4 implementation changed after parent validation")
    return parent, actual


def _validation_artifact_path(config_path: Path, experiment: str) -> Path:
    version = experiment.rsplit("_", 1)[-1]
    root = config_path.resolve().parents[2]
    return root / "artifacts" / "agi" / f"sparse_causal_bridge_validation_{version}.json"


def _assert_test_unlocked(
    config_path: Path,
    registration: dict,
    config_sha: str,
) -> str:
    path = _validation_artifact_path(config_path, registration["experiment"])
    if not path.exists():
        raise PermissionError("V5 test requires a saved passing validation artifact")
    raw = path.read_bytes()
    report = json.loads(raw)
    if not report.get("passed"):
        raise PermissionError("V5 validation artifact did not pass")
    if report.get("registration", {}).get("sha256") != config_sha:
        raise PermissionError("V5 merged registration SHA256 changed")
    if report.get("implementation_sha256") != _implementation_hashes():
        raise PermissionError("V5 implementation changed after validation")
    return hashlib.sha256(raw).hexdigest()


def _optional_rmse(truth: np.ndarray, prediction: np.ndarray) -> float | None:
    if not np.all(np.isfinite(prediction)):
        return None
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def _mean(values: Sequence[float | None]) -> float | None:
    if any(value is None for value in values):
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator / denominator)


def _reduction(baseline: float | None, candidate: float | None) -> float | None:
    ratio = _ratio(candidate, baseline)
    return None if ratio is None else float(1.0 - ratio)


def _student_ci_lower(
    baseline: Sequence[float | None],
    candidate: Sequence[float | None],
    critical_value: float,
) -> float | None:
    if len(baseline) != len(candidate) or len(candidate) < 2:
        return None
    if any(value is None for value in baseline) or any(
        value is None for value in candidate
    ):
        return None
    differences = np.asarray(baseline, dtype=float) - np.asarray(candidate, dtype=float)
    return float(
        np.mean(differences)
        - critical_value * np.std(differences, ddof=1) / np.sqrt(len(differences))
    )


def _win_fraction(
    baseline: Sequence[float | None], candidate: Sequence[float | None]
) -> float | None:
    if len(baseline) != len(candidate) or any(
        value is None for value in (*baseline, *candidate)
    ):
        return None
    return float(
        np.mean(np.asarray(candidate, dtype=float) < np.asarray(baseline, dtype=float))
    )


def _maximum_jacobian_radius(
    mechanism: base.BridgeModel, states: np.ndarray
) -> float:
    diagonal = mechanism.local_coefficients[:, 1]
    maximum = 0.0
    for state in states:
        sech_squared = 1.0 / np.cosh(np.asarray(state, dtype=float)) ** 2
        jacobian = np.diag(diagonal) + mechanism.bridge * sech_squared[None, :]
        maximum = max(maximum, float(np.max(np.abs(np.linalg.eigvals(jacobian)))))
    return maximum


def _model_metrics_template(
    model_names: Sequence[str], horizons: Sequence[int]
) -> dict[str, dict[int, dict[str, list]]]:
    return {
        name: {
            int(horizon): {
                "path": [],
                "downstream_path": [],
                "terminal": [],
                "lead_rmse": [],
            }
            for horizon in horizons
        }
        for name in model_names
    }


def _summarize_models(
    records: dict[str, dict[int, dict[str, list]]]
) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    for name, by_horizon in records.items():
        result[name] = {}
        for horizon, values in by_horizon.items():
            lead_rows = values["lead_rmse"]
            lead_mean: list[float | None] = []
            for lead in range(horizon):
                lead_mean.append(_mean([row[lead] for row in lead_rows]))
            result[name][str(horizon)] = {
                "mean_path_rmse": _mean(values["path"]),
                "mean_downstream_path_rmse": _mean(values["downstream_path"]),
                "mean_terminal_rmse": _mean(values["terminal"]),
                "mean_lead_rmse": lead_mean,
                "seed_path_rmse": values["path"],
                "seed_downstream_path_rmse": values["downstream_path"],
                "seed_terminal_rmse": values["terminal"],
            }
    return result


def _comparison(
    models: dict[str, dict[str, dict]],
    horizon: int,
    critical_value: float,
) -> dict[str, float | None]:
    key = str(horizon)
    candidate = models["causal_latent_free"][key]
    baselines = {
        "no_latent": models["causal_mechanism_no_latent"][key],
        "persistence": models["persistence"][key],
        "fixed_local": models["fixed_local_train"][key],
        "stable_local_latent": models["stable_fixed_local_latent_free"][key],
        "stable_adaptive_dense": models["stable_adaptive_dense_prefix_free"][key],
        "same_probe_dense_latent": models["same_probe_dense_latent_free"][key],
    }
    result: dict[str, float | None] = {}
    for label, baseline in baselines.items():
        result[f"global_path_rmse_reduction_vs_{label}"] = _reduction(
            baseline["mean_path_rmse"], candidate["mean_path_rmse"]
        )
        result[f"global_path_rmse_ratio_vs_{label}"] = _ratio(
            candidate["mean_path_rmse"], baseline["mean_path_rmse"]
        )
        result[f"terminal_rmse_reduction_vs_{label}"] = _reduction(
            baseline["mean_terminal_rmse"], candidate["mean_terminal_rmse"]
        )
        result[f"terminal_rmse_ratio_vs_{label}"] = _ratio(
            candidate["mean_terminal_rmse"], baseline["mean_terminal_rmse"]
        )
        result[f"paired_ci95_lower_vs_{label}"] = _student_ci_lower(
            baseline["seed_path_rmse"],
            candidate["seed_path_rmse"],
            critical_value,
        )
        result[f"seed_win_fraction_vs_{label}"] = _win_fraction(
            baseline["seed_path_rmse"], candidate["seed_path_rmse"]
        )
    result["downstream_path_rmse_reduction_vs_stable_local_latent"] = _reduction(
        baselines["stable_local_latent"]["mean_downstream_path_rmse"],
        candidate["mean_downstream_path_rmse"],
    )
    return result


def _passes_minimum(value: float | None, threshold: float) -> bool:
    return value is not None and bool(value >= threshold)


def _passes_maximum(value: float | None, threshold: float) -> bool:
    return value is not None and bool(value <= threshold)


def _all_numeric_finite(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return all(_all_numeric_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def run_free_rollout_gate(config_path: Path, *, split: str = "validation") -> dict:
    started = time.perf_counter()
    registration, raw = base._load_registration(config_path)
    base._validate_registration(registration)
    if registration.get("runner") != "single_origin_free_rollout":
        raise ValueError("V5 single-origin free-rollout registration required")
    if registration.get("active_gate") != "rollout_gate":
        raise ValueError("V5 rollout_gate must be the active gate")
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    config_sha = hashlib.sha256(raw).hexdigest()
    validation_artifact_sha = None
    if split == "test":
        validation_artifact_sha = _assert_test_unlocked(
            config_path, registration, config_sha
        )

    parent, parent_sha = _load_frozen_parent(config_path, registration)
    sparse_mechanism, dense_probe_mechanism = _mechanisms_from_parent(
        registration, parent
    )
    calibration = int(registration["latent_filter"]["ood_calibration_steps"])
    horizons = tuple(int(value) for value in registration["rollout"]["horizons"])
    max_horizon = max(horizons)
    if horizons != (5, 20):
        raise ValueError("V5 requires horizons [5, 20]")
    role = registration["data_roles"][split]
    if int(role["steps_per_seed"]) != calibration + max_horizon:
        raise ValueError("evaluation episode must end exactly at the H20 target")

    train_role = registration["data_roles"]["observational_train"]
    train = [
        base.simulate_episode(
            int(seed),
            registration,
            environment=train_role["environment"],
            steps=int(train_role["steps_per_seed"]),
        )
        for seed in train_role["seeds"]
    ]
    size = len(registration["charts"])
    all_edges = tuple(
        (source, target)
        for source in range(size)
        for target in range(size)
        if source != target
    )
    ridge = float(registration["learning"]["ridge"])
    fixed_local = base.fit_observational_model("fixed_local_train", train, (), ridge)
    fixed_dense = base.fit_observational_model(
        "fixed_dense_train", train, all_edges, ridge
    )
    v1_bridge = base.fit_fixed_bridge_model(
        "v1_bridge_observational_local",
        train,
        sparse_mechanism.bridge,
        sparse_mechanism.declared_edges,
        ridge,
    )
    stable_local = fit_stable_observational_model(
        "stable_fixed_local", train, (), ridge
    )
    stable_dense = fit_stable_observational_model(
        "stable_fixed_dense", train, all_edges, ridge
    )
    sparse_ar = float(parent["latent_filter"]["shared_train_scalar_ar"])
    stable_local_ar = latent.fit_pooled_residual_autoregression(train, stable_local)
    stable_dense_ar = latent.fit_pooled_residual_autoregression(train, stable_dense)
    dense_probe_ar = latent.fit_pooled_residual_autoregression(
        train, dense_probe_mechanism
    )
    train_norm_q99 = float(
        np.quantile(
            np.concatenate([np.linalg.norm(item.states, axis=1) for item in train]),
            0.99,
        )
    )

    model_names = tuple(registration["rollout"]["models"])
    records = _model_metrics_template(model_names, horizons)
    downstream = np.asarray(
        sorted(
            {
                int(item["target"])
                for item in registration["scm"]["true_directed_bridges"]
            }
        ),
        dtype=int,
    )
    truth_mechanism = latent.mechanism_model(
        "truth",
        np.asarray(registration["scm"]["self_coefficients"], dtype=float),
        base._true_bridge(registration),
        tuple(
            (int(item["source"]), int(item["target"]))
            for item in registration["scm"]["true_directed_bridges"]
        ),
    )
    loading = base._environment_loadings(registration, role["environment"])
    latent_ar_truth = float(registration["scm"]["latent_ar"])
    nonfinite_candidate_count = 0
    maximum_candidate_abs = 0.0
    maximum_candidate_norm = 0.0
    maximum_candidate_jacobian_radius = 0.0
    adaptive_self_abs: list[float] = []

    for seed in role["seeds"]:
        episode = base.simulate_episode(
            int(seed),
            registration,
            environment=role["environment"],
            steps=int(role["steps_per_seed"]),
        )
        prefix_states = episode.states[: calibration + 1].copy()
        x_previous = prefix_states[-2]
        x_anchor = prefix_states[-1]
        prefix_episode = base.Episode(
            states=prefix_states,
            hidden=episode.hidden[: calibration + 1].copy(),
        )
        adaptive_dense = fit_stable_observational_model(
            "stable_adaptive_dense_prefix", [prefix_episode], all_edges, ridge
        )
        adaptive_self_abs.append(
            float(np.max(np.abs(adaptive_dense.local_coefficients[:, 1])))
        )
        sparse_filter = fit_prefix_residual_filter(
            prefix_states, sparse_mechanism, sparse_ar
        )
        stable_local_filter = fit_prefix_residual_filter(
            prefix_states, stable_local, stable_local_ar
        )
        stable_dense_filter = fit_prefix_residual_filter(
            prefix_states, stable_dense, stable_dense_ar
        )
        dense_probe_filter = fit_prefix_residual_filter(
            prefix_states, dense_probe_mechanism, dense_probe_ar
        )
        predictions = {
            "persistence": np.repeat(x_anchor[None, :], max_horizon, axis=0),
            "fixed_local_train": free_rollout(
                fixed_local,
                x_previous=x_previous,
                x_anchor=x_anchor,
                horizon=max_horizon,
            ),
            "fixed_dense_train": free_rollout(
                fixed_dense,
                x_previous=x_previous,
                x_anchor=x_anchor,
                horizon=max_horizon,
            ),
            "v1_bridge_observational_local": free_rollout(
                v1_bridge,
                x_previous=x_previous,
                x_anchor=x_anchor,
                horizon=max_horizon,
            ),
            "causal_mechanism_no_latent": free_rollout(
                sparse_mechanism,
                x_previous=x_previous,
                x_anchor=x_anchor,
                horizon=max_horizon,
            ),
            "causal_latent_free": free_rollout(
                sparse_mechanism,
                x_previous=x_previous,
                x_anchor=x_anchor,
                horizon=max_horizon,
                residual_filter=sparse_filter,
            ),
            "stable_fixed_local_latent_free": free_rollout(
                stable_local,
                x_previous=x_previous,
                x_anchor=x_anchor,
                horizon=max_horizon,
                residual_filter=stable_local_filter,
            ),
            "stable_fixed_dense_latent_free": free_rollout(
                stable_dense,
                x_previous=x_previous,
                x_anchor=x_anchor,
                horizon=max_horizon,
                residual_filter=stable_dense_filter,
            ),
            "stable_adaptive_dense_prefix_free": free_rollout(
                adaptive_dense,
                x_previous=x_previous,
                x_anchor=x_anchor,
                horizon=max_horizon,
            ),
            "same_probe_dense_latent_free": free_rollout(
                dense_probe_mechanism,
                x_previous=x_previous,
                x_anchor=x_anchor,
                horizon=max_horizon,
                residual_filter=dense_probe_filter,
            ),
            "oracle_hidden_diagnostic": _oracle_rollout(
                truth_mechanism,
                x_anchor=x_anchor,
                hidden_anchor=float(episode.hidden[calibration]),
                loading=loading,
                latent_ar=latent_ar_truth,
                horizon=max_horizon,
            ),
        }
        candidate = predictions["causal_latent_free"]
        nonfinite_candidate_count += int(np.size(candidate) - np.count_nonzero(np.isfinite(candidate)))
        finite_candidate = candidate[np.isfinite(candidate)]
        if finite_candidate.size:
            maximum_candidate_abs = max(
                maximum_candidate_abs, float(np.max(np.abs(finite_candidate)))
            )
        finite_rows = candidate[np.all(np.isfinite(candidate), axis=1)]
        if len(finite_rows):
            maximum_candidate_norm = max(
                maximum_candidate_norm,
                float(np.max(np.linalg.norm(finite_rows, axis=1))),
            )
            maximum_candidate_jacobian_radius = max(
                maximum_candidate_jacobian_radius,
                _maximum_jacobian_radius(sparse_mechanism, finite_rows),
            )

        truth = episode.states[calibration + 1 : calibration + max_horizon + 1]
        for name in model_names:
            prediction = predictions[name]
            for horizon in horizons:
                predicted_path = prediction[:horizon]
                truth_path = truth[:horizon]
                path = _optional_rmse(truth_path, predicted_path)
                downstream_path = _optional_rmse(
                    truth_path[:, downstream], predicted_path[:, downstream]
                )
                terminal = _optional_rmse(
                    truth_path[-1], predicted_path[-1]
                )
                leads = [
                    _optional_rmse(truth_path[index], predicted_path[index])
                    for index in range(horizon)
                ]
                records[name][horizon]["path"].append(path)
                records[name][horizon]["downstream_path"].append(downstream_path)
                records[name][horizon]["terminal"].append(terminal)
                records[name][horizon]["lead_rmse"].append(leads)

    models = _summarize_models(records)
    paired = registration["rollout"]["paired_ci"]
    critical_value = float(
        paired[
            "critical_value_validation_n20"
            if split == "validation"
            else "critical_value_test_n30"
        ]
    )
    comparisons = {
        str(horizon): _comparison(models, horizon, critical_value)
        for horizon in horizons
    }
    candidate_h5 = models["causal_latent_free"]["5"]["mean_path_rmse"]
    candidate_h20 = models["causal_latent_free"]["20"]["mean_path_rmse"]
    stability = {
        "nonfinite_candidate_prediction_count": nonfinite_candidate_count,
        "maximum_candidate_prediction_absolute_value": maximum_candidate_abs,
        "maximum_candidate_prediction_l2_norm": maximum_candidate_norm,
        "train_state_l2_norm_q99": train_norm_q99,
        "prediction_norm_to_train_q99_ratio": _ratio(
            maximum_candidate_norm, train_norm_q99
        ),
        "maximum_learned_mechanism_jacobian_spectral_radius": (
            maximum_candidate_jacobian_radius
        ),
        "learned_sparse_latent_ar_abs": abs(sparse_ar),
        "horizon_20_to_5_path_rmse_ratio": _ratio(candidate_h20, candidate_h5),
        "maximum_adaptive_dense_self_coefficient_abs": float(
            np.max(adaptive_self_abs)
        ),
        "future_observation_reads_by_predictor": 0,
    }

    gate = registration["rollout_gate"]
    gate5 = gate["horizon_5"]
    gate20 = gate["horizon_20"]
    c5 = comparisons["5"]
    c20 = comparisons["20"]
    checks = {
        "frozen_parent_passed": bool(parent.get("passed")),
        "single_origin": registration["rollout"]["origins_per_seed"]
        == gate["origins_per_seed_required"],
        "h5_vs_no_latent": _passes_minimum(
            c5["global_path_rmse_reduction_vs_no_latent"],
            gate5["global_path_rmse_reduction_vs_no_latent_min"],
        ),
        "h5_vs_persistence": _passes_minimum(
            c5["global_path_rmse_reduction_vs_persistence"],
            gate5["global_path_rmse_reduction_vs_persistence_min"],
        ),
        "h5_fixed_local_noninferiority": _passes_maximum(
            c5["global_path_rmse_ratio_vs_fixed_local"],
            gate5["global_path_rmse_ratio_vs_fixed_local_max"],
        ),
        "h5_vs_stable_local_latent": _passes_minimum(
            c5["global_path_rmse_reduction_vs_stable_local_latent"],
            gate5["global_path_rmse_reduction_vs_stable_local_latent_min"],
        ),
        "h5_stable_adaptive_dense_noninferiority": _passes_maximum(
            c5["global_path_rmse_ratio_vs_stable_adaptive_dense"],
            gate5["global_path_rmse_ratio_vs_stable_adaptive_dense_max"],
        ),
        "h5_same_probe_dense_noninferiority": _passes_maximum(
            c5["global_path_rmse_ratio_vs_same_probe_dense_latent"],
            gate5["global_path_rmse_ratio_vs_same_probe_dense_latent_max"],
        ),
        "h5_seed_wins_no_latent": _passes_minimum(
            c5["seed_win_fraction_vs_no_latent"],
            gate5["seed_win_fraction_vs_no_latent_min"],
        ),
        "h5_seed_wins_persistence": _passes_minimum(
            c5["seed_win_fraction_vs_persistence"],
            gate5["seed_win_fraction_vs_persistence_min"],
        ),
        "h5_ci_no_latent": _passes_minimum(
            c5["paired_ci95_lower_vs_no_latent"],
            gate5["paired_ci95_lower_vs_no_latent_min"],
        ),
        "h5_ci_persistence": _passes_minimum(
            c5["paired_ci95_lower_vs_persistence"],
            gate5["paired_ci95_lower_vs_persistence_min"],
        ),
        "h5_ci_stable_local_latent": _passes_minimum(
            c5["paired_ci95_lower_vs_stable_local_latent"],
            gate5["paired_ci95_lower_vs_stable_local_latent_min"],
        ),
        "h5_downstream_vs_stable_local_latent": _passes_minimum(
            c5["downstream_path_rmse_reduction_vs_stable_local_latent"],
            gate5["downstream_path_rmse_reduction_vs_stable_local_latent_min"],
        ),
        "h5_terminal_vs_no_latent": _passes_minimum(
            c5["terminal_rmse_reduction_vs_no_latent"],
            gate5["terminal_rmse_reduction_vs_no_latent_min"],
        ),
        "h5_terminal_vs_persistence": _passes_minimum(
            c5["terminal_rmse_reduction_vs_persistence"],
            gate5["terminal_rmse_reduction_vs_persistence_min"],
        ),
        "h20_vs_no_latent": _passes_minimum(
            c20["global_path_rmse_reduction_vs_no_latent"],
            gate20["global_path_rmse_reduction_vs_no_latent_min"],
        ),
        "h20_vs_persistence": _passes_minimum(
            c20["global_path_rmse_reduction_vs_persistence"],
            gate20["global_path_rmse_reduction_vs_persistence_min"],
        ),
        "h20_vs_fixed_local": _passes_minimum(
            c20["global_path_rmse_reduction_vs_fixed_local"],
            gate20["global_path_rmse_reduction_vs_fixed_local_min"],
        ),
        "h20_vs_stable_local_latent": _passes_minimum(
            c20["global_path_rmse_reduction_vs_stable_local_latent"],
            gate20["global_path_rmse_reduction_vs_stable_local_latent_min"],
        ),
        "h20_vs_stable_adaptive_dense": _passes_minimum(
            c20["global_path_rmse_reduction_vs_stable_adaptive_dense"],
            gate20["global_path_rmse_reduction_vs_stable_adaptive_dense_min"],
        ),
        "h20_same_probe_dense_noninferiority": _passes_maximum(
            c20["global_path_rmse_ratio_vs_same_probe_dense_latent"],
            gate20["global_path_rmse_ratio_vs_same_probe_dense_latent_max"],
        ),
        "h20_seed_wins_no_latent": _passes_minimum(
            c20["seed_win_fraction_vs_no_latent"],
            gate20["seed_win_fraction_vs_no_latent_min"],
        ),
        "h20_seed_wins_persistence": _passes_minimum(
            c20["seed_win_fraction_vs_persistence"],
            gate20["seed_win_fraction_vs_persistence_min"],
        ),
        "h20_ci_no_latent": _passes_minimum(
            c20["paired_ci95_lower_vs_no_latent"],
            gate20["paired_ci95_lower_vs_no_latent_min"],
        ),
        "h20_ci_persistence": _passes_minimum(
            c20["paired_ci95_lower_vs_persistence"],
            gate20["paired_ci95_lower_vs_persistence_min"],
        ),
        "h20_ci_fixed_local": _passes_minimum(
            c20["paired_ci95_lower_vs_fixed_local"],
            gate20["paired_ci95_lower_vs_fixed_local_min"],
        ),
        "h20_ci_stable_local_latent": _passes_minimum(
            c20["paired_ci95_lower_vs_stable_local_latent"],
            gate20["paired_ci95_lower_vs_stable_local_latent_min"],
        ),
        "h20_downstream_vs_stable_local_latent": _passes_minimum(
            c20["downstream_path_rmse_reduction_vs_stable_local_latent"],
            gate20["downstream_path_rmse_reduction_vs_stable_local_latent_min"],
        ),
        "h20_terminal_no_latent_noninferiority": _passes_maximum(
            c20["terminal_rmse_ratio_vs_no_latent"],
            gate20["terminal_rmse_ratio_vs_no_latent_max"],
        ),
        "h20_terminal_vs_persistence": _passes_minimum(
            c20["terminal_rmse_reduction_vs_persistence"],
            gate20["terminal_rmse_reduction_vs_persistence_min"],
        ),
        "h20_terminal_stable_local_noninferiority": _passes_maximum(
            c20["terminal_rmse_ratio_vs_stable_local_latent"],
            gate20["terminal_rmse_ratio_vs_stable_local_latent_max"],
        ),
        "candidate_finite": nonfinite_candidate_count
        <= gate["nonfinite_prediction_count_max"],
        "candidate_absolute_bound": maximum_candidate_abs
        <= gate["maximum_prediction_absolute_value"],
        "learned_mechanism_stability": maximum_candidate_jacobian_radius
        <= gate["learned_mechanism_jacobian_spectral_radius_max"],
        "learned_latent_ar_stability": abs(sparse_ar)
        <= gate["learned_latent_ar_abs_max"],
        "rollout_error_growth": _passes_maximum(
            stability["horizon_20_to_5_path_rmse_ratio"],
            gate["horizon_20_to_5_path_rmse_ratio_max"],
        ),
        "prediction_scale": _passes_maximum(
            stability["prediction_norm_to_train_q99_ratio"],
            gate["maximum_prediction_norm_to_train_q99_ratio"],
        ),
        "zero_future_observation_reads": stability[
            "future_observation_reads_by_predictor"
        ]
        <= gate["future_observation_reads_max"],
        "finite_gating_metrics": _all_numeric_finite(
            {"comparisons": comparisons, "stability": stability}
        ),
    }

    elapsed = time.perf_counter() - started
    limits = registration["resource_limits"]
    resource_checks = {
        "cpu_time": elapsed <= limits["max_cpu_seconds_target"],
        "zero_download": limits["external_download_bytes"] == 0,
        "zero_trajectory_files": not limits["write_trajectory_files"],
        "numpy_only": bool(limits["numpy_only"]),
    }
    report = {
        "experiment": registration["experiment"],
        "roadmap_stage": registration["roadmap_stage"],
        "split": split,
        "registration": {
            "path": str(config_path),
            "sha256": config_sha,
            "status": registration["status"],
        },
        "implementation_sha256": _implementation_hashes(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "test_lock": {
            "validation_artifact_sha256": validation_artifact_sha,
        },
        "frozen_parent": {
            "path": registration["rollout"]["frozen_parent_validation_artifact"],
            "sha256": parent_sha,
            "experiment": parent["experiment"],
            "registration_sha256": parent["registration"]["sha256"],
            "implementation_sha256": parent["implementation_sha256"],
        },
        "frozen_model": {
            "causal_edges": parent["selection"]["causal_edges"],
            "self_coefficients": sparse_mechanism.local_coefficients[:, 1].tolist(),
            "sparse_bridge": sparse_mechanism.bridge.tolist(),
            "same_probe_dense_bridge": dense_probe_mechanism.bridge.tolist(),
            "shared_sparse_latent_ar": sparse_ar,
            "shared_stable_local_latent_ar": stable_local_ar,
            "shared_stable_dense_latent_ar": stable_dense_ar,
            "shared_same_probe_dense_latent_ar": dense_probe_ar,
        },
        "models": models,
        "comparisons": comparisons,
        "stability": stability,
        "checks": checks,
        "resource_checks": resource_checks,
        "resource_usage": {
            "wall_seconds": elapsed,
            "external_download_bytes": 0,
            "trajectory_files_written": 0,
            "evaluation_seeds": len(role["seeds"]),
            "observed_prefix_transitions_per_seed": calibration,
            "free_rollout_steps_per_seed": max_horizon,
            "forecast_origins_per_seed": 1,
            "evaluation_probe_pairs": 0,
        },
    }
    report["performance_passed"] = bool(all(checks.values()))
    report["resource_passed"] = bool(all(resource_checks.values()))
    report["passed"] = bool(report["performance_passed"] and report["resource_passed"])
    return report


def _default_output(config_path: Path, split: str, experiment: str) -> Path:
    version = experiment.rsplit("_", 1)[-1]
    root = config_path.resolve().parents[2]
    return root / "artifacts" / "agi" / f"sparse_causal_bridge_{split}_{version}.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    registration, _ = base._load_registration(args.config)
    report = run_free_rollout_gate(args.config, split=args.split)
    output = args.output or _default_output(
        args.config, args.split, registration["experiment"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(f"artifact: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
