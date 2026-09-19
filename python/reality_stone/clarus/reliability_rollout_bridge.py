"""V7 symmetric consensus closure gate for the sparse causal bridge line.

This module implements a locked synthetic forecast experiment.  It does not
implement or measure AGI.  The sparse contribution is primary and may close
the route when the registered validation conjunction fails.
"""

from __future__ import annotations

import argparse
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
from . import sparse_causal_bridge as base


MODEL_NAMES = (
    "sparse_consensus",
    "no_sparse_consensus",
    "symmetric_dense_consensus",
    "v5_sparse_parent",
    "stable_adaptive_dense_prefix_free",
    "persistence",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _implementation_hashes() -> dict[str, str]:
    paths = {
        "reliability_rollout_bridge.py": Path(__file__).resolve(),
        "free_rollout_bridge.py": Path(free.__file__).resolve(),
        "latent_causal_bridge.py": Path(latent.__file__).resolve(),
        "sparse_causal_bridge.py": Path(base.__file__).resolve(),
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _test_hashes(root: Path) -> dict[str, str]:
    path = root / "tests" / "test_reliability_rollout_bridge.py"
    if not path.exists():
        return {"test_reliability_rollout_bridge.py": "MISSING"}
    return {"test_reliability_rollout_bridge.py": _sha256(path)}


class PrefixReader:
    """Copy a registered prefix while recording every accessible state index."""

    def __init__(self, states: np.ndarray, origin: int) -> None:
        values = np.asarray(states, dtype=float)
        if values.ndim != 2 or origin < 1 or origin >= len(values):
            raise ValueError("states must contain a valid registered origin")
        self._states = values
        self._origin = int(origin)
        self.max_observed_state_index = -1
        self.future_observation_reads = 0

    def through_origin(self) -> np.ndarray:
        stop = self._origin + 1
        if stop > self._origin + 1:
            self.future_observation_reads += stop - (self._origin + 1)
            raise PermissionError("future observation read attempted")
        self.max_observed_state_index = max(self.max_observed_state_index, self._origin)
        result = self._states[:stop].copy()
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class TrainingContext:
    sparse_mechanism: base.BridgeModel
    dense_probe_mechanism: base.BridgeModel
    sparse_ar: float
    dense_probe_ar: float
    all_edges: tuple[base.Edge, ...]
    ridge: float
    scales: np.ndarray
    train_normalized_norm_q99: float
    parent_report: dict
    parent_raw_sha256: str
    v5_failure_report: dict
    v5_failure_raw_sha256: str


@dataclass(frozen=True)
class PrefixPredictions:
    models: dict[str, np.ndarray]
    weights: dict[str, np.ndarray]
    pathwise_jacobian_radii: dict[str, float]
    component_rollouts: int


def _load_v5_failure(config_path: Path, registration: dict) -> tuple[dict, str]:
    root = config_path.resolve().parents[2]
    path = root / registration["failed_predecessor_artifact"]
    expected = registration["development_data_disclosure"]["v5_failure_artifact_sha256_raw_crlf"]
    actual = _sha256(path)
    if actual != expected:
        raise PermissionError("historical V5 failure artifact SHA256 changed")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("passed") is not False:
        raise PermissionError("historical V5 failure artifact is not a failure")
    expected_false = {
        "h20_ci_persistence",
        "h20_vs_stable_adaptive_dense",
        "h5_ci_persistence",
        "h5_seed_wins_persistence",
    }
    actual_false = {name for name, passed in report["checks"].items() if not passed}
    if actual_false != expected_false:
        raise PermissionError("historical V5 failure set changed")
    return report, actual


def _build_training_context(config_path: Path, registration: dict) -> TrainingContext:
    parent, parent_sha = free._load_frozen_parent(config_path, registration)
    sparse, dense_probe = free._mechanisms_from_parent(registration, parent)
    v5_failure, v5_failure_sha = _load_v5_failure(config_path, registration)

    role = registration["data_roles"]["observational_train"]
    train = [
        base.simulate_episode(
            int(seed),
            registration,
            environment=role["environment"],
            steps=int(role["steps_per_seed"]),
        )
        for seed in role["seeds"]
    ]
    states = np.concatenate([episode.states for episode in train], axis=0)
    ddof = int(registration["normalization"]["ddof"])
    scales = np.std(states, axis=0, ddof=ddof)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("training-only chart scales must be positive and finite")
    expected_scales = np.asarray(registration["normalization"]["expected_scales"], dtype=float)
    tolerance = float(registration["normalization"]["scale_match_absolute_tolerance"])
    if not np.allclose(scales, expected_scales, rtol=0.0, atol=tolerance):
        raise PermissionError("training-only normalization scales changed")

    size = len(registration["charts"])
    all_edges = tuple(
        (source, target) for source in range(size) for target in range(size) if source != target
    )
    dense_ar = latent.fit_pooled_residual_autoregression(train, dense_probe)
    normalized_norms = np.linalg.norm(states / scales, axis=1)
    return TrainingContext(
        sparse_mechanism=sparse,
        dense_probe_mechanism=dense_probe,
        sparse_ar=float(parent["latent_filter"]["shared_train_scalar_ar"]),
        dense_probe_ar=float(dense_ar),
        all_edges=all_edges,
        ridge=float(registration["learning"]["ridge"]),
        scales=scales,
        train_normalized_norm_q99=float(np.quantile(normalized_norms, 0.99)),
        parent_report=parent,
        parent_raw_sha256=parent_sha,
        v5_failure_report=v5_failure,
        v5_failure_raw_sha256=v5_failure_sha,
    )


def _latent_rollout(
    mechanism: base.BridgeModel,
    autoregression: float,
    prefix: np.ndarray,
    horizon: int,
) -> np.ndarray:
    residual_filter = free.fit_prefix_residual_filter(prefix, mechanism, autoregression)
    return free.free_rollout(
        mechanism,
        x_previous=prefix[-2],
        x_anchor=prefix[-1],
        horizon=horizon,
        residual_filter=residual_filter,
    )


def _fit_adaptive_dense(prefix: np.ndarray, context: TrainingContext) -> base.BridgeModel:
    # The historical fitter accepts Episode, but only reads states.  Real hidden
    # values are deliberately unavailable and replaced by inert zeros.
    episode = base.Episode(
        states=np.asarray(prefix, dtype=float),
        hidden=np.zeros(len(prefix), dtype=float),
    )
    return free.fit_stable_observational_model(
        "stable_adaptive_dense_prefix",
        [episode],
        context.all_edges,
        context.ridge,
    )


def _adaptive_rollout(
    prefix: np.ndarray, context: TrainingContext, horizon: int
) -> tuple[np.ndarray, base.BridgeModel]:
    model = _fit_adaptive_dense(prefix, context)
    prediction = free.free_rollout(
        model,
        x_previous=prefix[-2],
        x_anchor=prefix[-1],
        horizon=horizon,
    )
    return prediction, model


def _normalized_mse(truth: np.ndarray, prediction: np.ndarray, scales: np.ndarray) -> float:
    difference = (np.asarray(truth) - np.asarray(prediction)) / scales
    return float(np.mean(difference * difference))


def _normalized_rmse(truth: np.ndarray, prediction: np.ndarray, scales: np.ndarray) -> float:
    return float(math.sqrt(_normalized_mse(truth, prediction, scales)))


def _inverse_root_weights(errors: Sequence[float], epsilon: float) -> np.ndarray:
    values = np.asarray(errors, dtype=float)
    if values.ndim != 1 or len(values) < 2 or np.any(~np.isfinite(values)):
        raise ValueError("controller errors must be a finite one-dimensional vector")
    if np.any(values < 0) or epsilon <= 0:
        raise ValueError("controller errors and epsilon must be nonnegative")
    raw = np.power(values + epsilon, -0.5)
    weights = raw / np.sum(raw)
    if np.any(weights < 0) or not np.isclose(np.sum(weights), 1.0, atol=1e-12):
        raise FloatingPointError("invalid convex controller weights")
    return weights


def _combine(weights: np.ndarray, predictions: Sequence[np.ndarray]) -> np.ndarray:
    if len(weights) != len(predictions):
        raise ValueError("one weight is required per component prediction")
    result = np.zeros_like(np.asarray(predictions[0], dtype=float))
    for weight, prediction in zip(weights, predictions):
        values = np.asarray(prediction, dtype=float)
        if values.shape != result.shape:
            raise ValueError("component predictions must have identical shapes")
        result += float(weight) * values
    return result


def predict_from_prefix(
    prefix_states: np.ndarray,
    context: TrainingContext,
    registration: dict,
) -> PrefixPredictions:
    """Generate all V7 predictions from the immutable observed prefix only."""

    prefix = np.asarray(prefix_states, dtype=float)
    origin = int(registration["closure"]["origin"])
    pseudo_origin = int(registration["closure"]["pseudo_origin"])
    horizon = int(registration["closure"]["horizon"])
    if prefix.shape != (origin + 1, len(context.scales)):
        raise ValueError("prefix must end exactly at the registered origin")
    if prefix.flags.writeable:
        prefix = prefix.copy()
        prefix.setflags(write=False)
    inner = prefix[: pseudo_origin + 1]
    inner_truth = prefix[pseudo_origin + 1 : origin + 1]
    if len(inner_truth) != horizon:
        raise ValueError("registered pseudo-origin must expose one H20 backtest")

    sparse_inner = _latent_rollout(context.sparse_mechanism, context.sparse_ar, inner, horizon)
    dense_inner = _latent_rollout(
        context.dense_probe_mechanism, context.dense_probe_ar, inner, horizon
    )
    adaptive_inner, _ = _adaptive_rollout(inner, context, horizon)
    persistence_inner = np.repeat(inner[-1][None, :], horizon, axis=0)

    epsilon = float(registration["closure"]["weight_epsilon_dimensionless"])
    sparse_weights = _inverse_root_weights(
        [
            _normalized_mse(inner_truth, sparse_inner, context.scales),
            _normalized_mse(inner_truth, adaptive_inner, context.scales),
            _normalized_mse(inner_truth, persistence_inner, context.scales),
        ],
        epsilon,
    )
    dense_weights = _inverse_root_weights(
        [
            _normalized_mse(inner_truth, dense_inner, context.scales),
            _normalized_mse(inner_truth, adaptive_inner, context.scales),
            _normalized_mse(inner_truth, persistence_inner, context.scales),
        ],
        epsilon,
    )
    no_sparse_weights = _inverse_root_weights(
        [
            _normalized_mse(inner_truth, adaptive_inner, context.scales),
            _normalized_mse(inner_truth, persistence_inner, context.scales),
        ],
        epsilon,
    )

    sparse_outer = _latent_rollout(context.sparse_mechanism, context.sparse_ar, prefix, horizon)
    dense_outer = _latent_rollout(
        context.dense_probe_mechanism, context.dense_probe_ar, prefix, horizon
    )
    adaptive_outer, adaptive_model = _adaptive_rollout(prefix, context, horizon)
    persistence_outer = np.repeat(prefix[-1][None, :], horizon, axis=0)

    models = {
        "sparse_consensus": _combine(
            sparse_weights, [sparse_outer, adaptive_outer, persistence_outer]
        ),
        "no_sparse_consensus": _combine(no_sparse_weights, [adaptive_outer, persistence_outer]),
        "symmetric_dense_consensus": _combine(
            dense_weights, [dense_outer, adaptive_outer, persistence_outer]
        ),
        "v5_sparse_parent": sparse_outer,
        "stable_adaptive_dense_prefix_free": adaptive_outer,
        "persistence": persistence_outer,
    }
    radii = {
        "v5_sparse_parent": free._maximum_jacobian_radius(context.sparse_mechanism, sparse_outer),
        "same_probe_dense_latent_free": free._maximum_jacobian_radius(
            context.dense_probe_mechanism, dense_outer
        ),
        "stable_adaptive_dense_prefix_free": free._maximum_jacobian_radius(
            adaptive_model, adaptive_outer
        ),
    }
    return PrefixPredictions(
        models=models,
        weights={
            "sparse_consensus": sparse_weights,
            "symmetric_dense_consensus": dense_weights,
            "no_sparse_consensus": no_sparse_weights,
        },
        pathwise_jacobian_radii=radii,
        component_rollouts=8,
    )


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _sample_sd(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def _paired_lower(
    baseline: Sequence[float], candidate: Sequence[float], critical: float
) -> dict[str, float]:
    difference = np.asarray(baseline, dtype=float) - np.asarray(candidate, dtype=float)
    if len(difference) < 2:
        raise ValueError("paired inference requires at least two seeds")
    mean = float(np.mean(difference))
    sd = float(np.std(difference, ddof=1))
    half_width = float(critical * sd / math.sqrt(len(difference)))
    return {
        "mean_improvement": mean,
        "sample_sd": sd,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
        "seed_win_fraction": float(np.mean(difference > 0)),
    }


def _paired_log_ratio_upper(
    candidate: Sequence[float], baseline: Sequence[float], critical: float
) -> dict[str, float]:
    candidate_values = np.asarray(candidate, dtype=float)
    baseline_values = np.asarray(baseline, dtype=float)
    if np.any(candidate_values <= 0) or np.any(baseline_values <= 0):
        raise ValueError("log-ratio inference requires positive seed RMSE values")
    ratios = np.log(candidate_values / baseline_values)
    mean = float(np.mean(ratios))
    sd = float(np.std(ratios, ddof=1))
    half_width = float(critical * sd / math.sqrt(len(ratios)))
    return {
        "mean_log_ratio": mean,
        "sample_sd": sd,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
        "geometric_mean_ratio": float(math.exp(mean)),
    }


def _all_numeric_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_all_numeric_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _validation_artifact_path(config_path: Path) -> Path:
    root = config_path.resolve().parents[2]
    return root / "artifacts" / "agi" / "sparse_causal_bridge_validation_v7.json"


def _assert_test_unlocked(
    config_path: Path,
    raw_registration_sha: str,
    canonical_registration_sha: str,
) -> str:
    path = _validation_artifact_path(config_path)
    if not path.exists():
        raise PermissionError("V7 test requires a saved validation artifact")
    raw = path.read_bytes()
    report = json.loads(raw)
    if not report.get("passed"):
        raise PermissionError("V7 validation failed; locked test remains unopened")
    registration = report.get("registration", {})
    if registration.get("merged_raw_sha256") != raw_registration_sha:
        raise PermissionError("V7 merged registration changed after validation")
    if registration.get("canonical_sha256") != canonical_registration_sha:
        raise PermissionError("V7 canonical registration changed after validation")
    if report.get("implementation_sha256") != _implementation_hashes():
        raise PermissionError("V7 implementation changed after validation")
    root = config_path.resolve().parents[2]
    if report.get("test_sha256") != _test_hashes(root):
        raise PermissionError("V7 tests changed after validation")
    return hashlib.sha256(raw).hexdigest()


def run_reliability_closure_gate(config_path: Path, *, split: str = "validation") -> dict:
    """Run the locked V7 validation or conditionally unlocked test gate."""

    started = time.perf_counter()
    registration, raw = base._load_registration(config_path)
    base._validate_registration(registration)
    if registration.get("runner") != "symmetric_consensus_closure":
        raise ValueError("V7 symmetric consensus closure registration required")
    if registration.get("active_gate") != "closure_gate":
        raise ValueError("V7 closure_gate must be active")
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    raw_registration_sha = hashlib.sha256(raw).hexdigest()
    canonical_registration_sha = _canonical_json_sha256(registration)
    validation_artifact_sha = None
    if split == "test":
        validation_artifact_sha = _assert_test_unlocked(
            config_path, raw_registration_sha, canonical_registration_sha
        )

    context = _build_training_context(config_path, registration)
    role = registration["data_roles"][split]
    gate = registration["closure_gate"]
    expected_seed_count = int(
        gate["validation_seeds_required" if split == "validation" else "test_seeds_required"]
    )
    if len(role["seeds"]) != expected_seed_count:
        raise ValueError("registered V7 seed count changed")
    origin = int(registration["closure"]["origin"])
    horizon = int(registration["closure"]["horizon"])
    if int(role["steps_per_seed"]) != origin + horizon:
        raise ValueError("V7 episode must end exactly at its H20 target")

    seed_rmse = {name: [] for name in MODEL_NAMES}
    weight_rows = {
        "sparse_consensus": [],
        "symmetric_dense_consensus": [],
        "no_sparse_consensus": [],
    }
    maximum_observed_index = -1
    future_reads = 0
    nonfinite_count = 0
    maximum_component_abs = 0.0
    maximum_normalized_norm = 0.0
    maximum_pathwise_radius = 0.0
    component_rollouts = 0

    for seed in role["seeds"]:
        episode = base.simulate_episode(
            int(seed),
            registration,
            environment=role["environment"],
            steps=int(role["steps_per_seed"]),
        )
        reader = PrefixReader(episode.states, origin)
        prefix = reader.through_origin()
        prediction_set = predict_from_prefix(prefix, context, registration)
        maximum_observed_index = max(maximum_observed_index, reader.max_observed_state_index)
        future_reads += reader.future_observation_reads
        component_rollouts += prediction_set.component_rollouts

        truth = episode.states[origin + 1 : origin + horizon + 1]
        for name in MODEL_NAMES:
            prediction = prediction_set.models[name]
            nonfinite_count += int(prediction.size - np.count_nonzero(np.isfinite(prediction)))
            finite = prediction[np.isfinite(prediction)]
            if finite.size:
                maximum_component_abs = max(maximum_component_abs, float(np.max(np.abs(finite))))
            finite_rows = prediction[np.all(np.isfinite(prediction), axis=1)]
            if len(finite_rows):
                maximum_normalized_norm = max(
                    maximum_normalized_norm,
                    float(np.max(np.linalg.norm(finite_rows / context.scales, axis=1))),
                )
            seed_rmse[name].append(_normalized_rmse(truth, prediction, context.scales))
        for name, weights in prediction_set.weights.items():
            weight_rows[name].append(weights.tolist())
        maximum_pathwise_radius = max(
            maximum_pathwise_radius,
            *prediction_set.pathwise_jacobian_radii.values(),
        )

    models = {
        name: {
            "mean_h20_normalized_path_rmse": _mean(values),
            "seed_h20_normalized_path_rmse": values,
            "sample_sd": _sample_sd(values),
        }
        for name, values in seed_rmse.items()
    }
    critical = float(registration["closure"]["critical_value_n96_df95"])
    candidate = seed_rmse["sparse_consensus"]
    comparisons = {
        "sparse_contribution_vs_no_sparse": _paired_lower(
            seed_rmse["no_sparse_consensus"], candidate, critical
        ),
        "improvement_vs_v5_parent": _paired_lower(
            seed_rmse["v5_sparse_parent"], candidate, critical
        ),
        "improvement_vs_persistence": _paired_lower(seed_rmse["persistence"], candidate, critical),
        "log_ratio_vs_stable_adaptive_dense": _paired_log_ratio_upper(
            candidate,
            seed_rmse["stable_adaptive_dense_prefix_free"],
            critical,
        ),
        "log_ratio_vs_symmetric_dense_consensus": _paired_log_ratio_upper(
            candidate, seed_rmse["symmetric_dense_consensus"], critical
        ),
    }
    maximum_weight_sum_error = max(
        abs(sum(row) - 1.0) for rows in weight_rows.values() for row in rows
    )
    stability = {
        "nonfinite_prediction_count": nonfinite_count,
        "maximum_component_prediction_absolute_value": maximum_component_abs,
        "maximum_prediction_normalized_l2_norm": maximum_normalized_norm,
        "train_normalized_state_l2_norm_q99": context.train_normalized_norm_q99,
        "prediction_norm_to_train_q99_ratio": (
            maximum_normalized_norm / context.train_normalized_norm_q99
        ),
        "maximum_dynamic_component_pathwise_jacobian_radius": maximum_pathwise_radius,
        "maximum_latent_ar_abs": max(abs(context.sparse_ar), abs(context.dense_probe_ar)),
        "maximum_observed_state_index": maximum_observed_index,
        "future_observation_reads": future_reads,
        "maximum_weight_sum_error": maximum_weight_sum_error,
    }
    checks = {
        "historical_v4_parent_passed": bool(context.parent_report.get("passed")),
        "historical_v5_failure_preserved": context.v5_failure_report.get("passed") is False,
        "registered_seed_count": len(role["seeds"]) == expected_seed_count,
        "single_origin": int(gate["origins_per_seed_required"]) == 1,
        "h5_non_gating": gate["h5_is_gating"] is False,
        "sparse_contribution_ci": comparisons["sparse_contribution_vs_no_sparse"]["ci95_lower"]
        >= float(gate["paired_ci95_lower_sparse_contribution_vs_no_sparse_min"]),
        "v5_parent_repair_ci": comparisons["improvement_vs_v5_parent"]["ci95_lower"]
        >= float(gate["paired_ci95_lower_improvement_vs_v5_parent_min"]),
        "persistence_improvement_ci": comparisons["improvement_vs_persistence"]["ci95_lower"]
        >= float(gate["paired_ci95_lower_improvement_vs_persistence_min"]),
        "stable_adaptive_dense_noninferiority": comparisons["log_ratio_vs_stable_adaptive_dense"][
            "ci95_upper"
        ]
        <= float(gate["paired_log_ratio_ci95_upper_vs_stable_adaptive_dense_max"]),
        "symmetric_dense_consensus_noninferiority": comparisons[
            "log_ratio_vs_symmetric_dense_consensus"
        ]["ci95_upper"]
        <= float(gate["paired_log_ratio_ci95_upper_vs_symmetric_dense_consensus_max"]),
        "predictions_finite": nonfinite_count <= int(gate["nonfinite_prediction_count_max"]),
        "prediction_absolute_bound": maximum_component_abs
        <= float(gate["maximum_component_prediction_absolute_value"]),
        "prediction_scale": stability["prediction_norm_to_train_q99_ratio"]
        <= float(gate["maximum_prediction_norm_to_train_q99_ratio"]),
        "pathwise_dynamic_stability": maximum_pathwise_radius
        <= float(gate["maximum_dynamic_component_pathwise_jacobian_radius"]),
        "latent_ar_stability": stability["maximum_latent_ar_abs"]
        <= float(gate["maximum_latent_ar_abs"]),
        "observed_state_index_bound": maximum_observed_index
        <= int(gate["maximum_observed_state_index"]),
        "zero_future_observation_reads": future_reads <= int(gate["future_observation_reads_max"]),
        "convex_weight_sums": maximum_weight_sum_error <= float(gate["weights_sum_tolerance"]),
    }
    finite_metrics = _all_numeric_finite(
        {"models": models, "comparisons": comparisons, "stability": stability}
    )
    checks["finite_gating_metrics"] = finite_metrics is bool(gate["finite_metrics_required"])
    passed = all(checks.values())
    root = config_path.resolve().parents[2]
    result = {
        "experiment": registration["experiment"],
        "roadmap_stage": registration["roadmap_stage"],
        "split": split,
        "environment": role["environment"],
        "passed": passed,
        "closure_status": (
            "SPARSE_CONTRIBUTION_SUPPORTED" if passed else "SPARSE_ROUTE_CLOSED_ON_VALIDATION"
        ),
        "claim_boundary": registration["claim_boundary"],
        "checks": checks,
        "models": models,
        "comparisons": comparisons,
        "weights": {
            name: {
                "mean": np.mean(np.asarray(rows), axis=0).tolist(),
                "per_seed": rows,
            }
            for name, rows in weight_rows.items()
        },
        "normalization": {
            "training_only_scales": context.scales.tolist(),
            "ddof": registration["normalization"]["ddof"],
        },
        "stability": stability,
        "registration": {
            "path": str(config_path),
            "merged_raw_sha256": raw_registration_sha,
            "canonical_sha256": canonical_registration_sha,
        },
        "parent_artifacts": {
            "v4_validation_raw_sha256": context.parent_raw_sha256,
            "v5_failure_raw_sha256": context.v5_failure_raw_sha256,
        },
        "implementation_sha256": _implementation_hashes(),
        "test_sha256": _test_hashes(root),
        "test_lock": {
            "validation_artifact_sha256": validation_artifact_sha,
            "test_opened": split == "test",
            "test_unlocked": split == "test",
        },
        "resource_usage": {
            "evaluation_seeds": len(role["seeds"]),
            "forecast_origins_per_seed": 1,
            "component_rollouts_per_seed": component_rollouts // len(role["seeds"]),
            "total_component_rollouts": component_rollouts,
            "observed_prefix_transitions_per_seed": origin,
            "free_rollout_steps_per_component": horizon,
            "evaluation_probe_pairs": 0,
            "external_download_bytes": 0,
            "trajectory_files_written": 0,
            "wall_seconds": time.perf_counter() - started,
        },
        "environment_manifest": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "development_data_counted_as_v7_evidence": False,
    }
    return result


def _default_output(config_path: Path, split: str) -> Path:
    root = config_path.resolve().parents[2]
    return root / "artifacts" / "agi" / f"sparse_causal_bridge_{split}_v7.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/preregistration/sparse_causal_bridge_v7.json"),
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    report = run_reliability_closure_gate(args.config, split=args.split)
    if not args.no_save:
        output = args.output or _default_output(args.config, args.split)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(output)
    print(json.dumps({"passed": report["passed"], "checks": report["checks"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
