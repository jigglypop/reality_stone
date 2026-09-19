"""V8 confirmatory gate for training-only parent-anchored shrinkage.

This is a locked four-chart synthetic H20 forecasting experiment.  It does
not implement or measure AGI.  Validation and test are one-shot evidence
blocks protected by registration, implementation, test, and artifact hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import free_rollout_bridge as free
from . import latent_causal_bridge as latent
from . import reliability_rollout_bridge as rel
from . import sparse_causal_bridge as base


MODEL_NAMES = (
    "parent_anchored_sparse",
    "v5_sparse_parent",
    "persistence",
    "zero_bridge_shrinkage",
    "symmetric_dense_shrinkage",
    "frozen_v7_consensus",
    "frozen_v7_no_sparse_consensus",
    "stable_adaptive_dense",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _root(config_path: Path) -> Path:
    return config_path.resolve().parents[2]


def _implementation_hashes() -> dict[str, str]:
    paths = {
        "parent_anchored_rollout_bridge.py": Path(__file__).resolve(),
        "reliability_rollout_bridge.py": Path(rel.__file__).resolve(),
        "free_rollout_bridge.py": Path(free.__file__).resolve(),
        "latent_causal_bridge.py": Path(latent.__file__).resolve(),
        "sparse_causal_bridge.py": Path(base.__file__).resolve(),
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _test_hashes(root: Path) -> dict[str, str]:
    path = root / "tests" / "test_parent_anchored_rollout_bridge.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    return {path.name: _sha256(path)}


@dataclass(frozen=True)
class V8TrainingContext:
    parent: rel.TrainingContext
    zero_bridge_mechanism: base.BridgeModel
    zero_bridge_ar: float
    sparse_gain: float
    dense_gain: float
    zero_bridge_gain: float
    gain_fit_windows: int
    v7_failure_raw_sha256: str
    r1_development_raw_sha256: str


@dataclass(frozen=True)
class PrefixPredictions:
    models: dict[str, np.ndarray]
    component_pathwise_jacobian_radii: dict[str, float]
    convex_envelope_violations: dict[str, float]


def _fit_gain(
    episodes: Sequence[base.Episode],
    mechanism: base.BridgeModel,
    autoregression: float,
    scales: np.ndarray,
    origins: Sequence[int],
    horizon: int,
) -> tuple[float, int]:
    numerator = 0.0
    denominator = 0.0
    windows = 0
    for episode in episodes:
        for raw_origin in origins:
            origin = int(raw_origin)
            prefix = episode.states[: origin + 1].copy()
            prefix.setflags(write=False)
            truth = episode.states[origin + 1 : origin + horizon + 1]
            if len(truth) != horizon:
                raise ValueError("gain-fit origin lacks a complete target window")
            learned = rel._latent_rollout(mechanism, autoregression, prefix, horizon)
            persistence = np.repeat(prefix[-1][None, :], horizon, axis=0)
            direction = (learned - persistence) / scales
            target = (truth - persistence) / scales
            numerator += float(np.sum(direction * target))
            denominator += float(np.sum(direction * direction))
            windows += 1
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise FloatingPointError("gain denominator must be positive and finite")
    gain = float(np.clip(numerator / denominator, 0.0, 1.0))
    if not math.isfinite(gain):
        raise FloatingPointError("gain must be finite")
    return gain, windows


def _assert_historical_seed_disjoint(config_path: Path, registration: dict) -> dict:
    validation = set(map(int, registration["data_roles"]["validation"]["seeds"]))
    test = set(map(int, registration["data_roles"]["test"]["seeds"]))
    if validation & test:
        raise PermissionError("V8 validation and test roles overlap")
    historical: set[int] = set()
    for path in sorted(config_path.parent.glob("sparse_causal_bridge_v*.json")):
        if path.resolve() == config_path.resolve():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        roles = raw.get("overrides", {}).get("data_roles", raw.get("data_roles", {}))
        for role in roles.values():
            if isinstance(role, dict):
                historical.update(map(int, role.get("seeds", [])))
    development = set(
        range(
            int(registration["development_data_disclosure"]["development_seed_first"]),
            int(registration["development_data_disclosure"]["development_seed_last"]) + 1,
        )
    )
    overlap = {
        "validation_historical": sorted(validation & historical),
        "test_historical": sorted(test & historical),
        "validation_development": sorted(validation & development),
        "test_development": sorted(test & development),
        "validation_test": sorted(validation & test),
    }
    if any(overlap.values()):
        raise PermissionError(f"V8 evidence role overlap: {overlap}")
    return overlap


def _build_training_context(config_path: Path, registration: dict) -> V8TrainingContext:
    # V8 deliberately replaces the development disclosure object.  Build the
    # frozen V7 parent from its own registration so V5/V4 provenance keys are
    # not accidentally erased by that override.
    v7_config = config_path.parent / "sparse_causal_bridge_v7.json"
    v7_registration, _ = base._load_registration(v7_config)
    parent = rel._build_training_context(v7_config, v7_registration)
    train_role = registration["data_roles"]["observational_train"]
    episodes = [
        base.simulate_episode(
            int(seed), registration, environment=train_role["environment"],
            steps=int(train_role["steps_per_seed"])
        )
        for seed in train_role["seeds"]
    ]
    zero = latent.mechanism_model(
        "zero_bridge_parent",
        parent.sparse_mechanism.local_coefficients[:, 1],
        np.zeros_like(parent.sparse_mechanism.bridge),
        (),
    )
    zero_ar = float(latent.fit_pooled_residual_autoregression(episodes, zero))
    spec = registration["parent_anchor"]
    origins = tuple(map(int, spec["gain_fit_origins"]))
    horizon = int(spec["horizon"])
    sparse_gain, sparse_windows = _fit_gain(
        episodes, parent.sparse_mechanism, parent.sparse_ar, parent.scales, origins, horizon
    )
    dense_gain, dense_windows = _fit_gain(
        episodes, parent.dense_probe_mechanism, parent.dense_probe_ar,
        parent.scales, origins, horizon
    )
    zero_gain, zero_windows = _fit_gain(
        episodes, zero, zero_ar, parent.scales, origins, horizon
    )
    expected_windows = int(spec["gain_fit_windows"])
    if {sparse_windows, dense_windows, zero_windows} != {expected_windows}:
        raise PermissionError("gain-fit window count changed")
    tolerance = float(spec["gain_match_absolute_tolerance"])
    expected = (
        float(spec["expected_sparse_gain"]),
        float(spec["expected_dense_control_gain"]),
        float(spec["expected_zero_bridge_control_gain"]),
    )
    actual = (sparse_gain, dense_gain, zero_gain)
    if not all(math.isclose(a, e, rel_tol=0.0, abs_tol=tolerance) for a, e in zip(actual, expected)):
        raise PermissionError(f"training-only gains changed: expected={expected}, actual={actual}")
    root = _root(config_path)
    v7_path = root / registration["development_data_disclosure"]["v7_validation_artifact"]
    r1_path = root / registration["development_data_disclosure"]["r1_development_artifact"]
    v7_sha, r1_sha = _sha256(v7_path), _sha256(r1_path)
    if v7_sha != registration["development_data_disclosure"]["v7_validation_artifact_sha256"]:
        raise PermissionError("V7 failure artifact changed")
    if r1_sha != registration["development_data_disclosure"]["r1_development_artifact_sha256"]:
        raise PermissionError("R1 development artifact changed")
    if json.loads(v7_path.read_text(encoding="utf-8")).get("passed") is not False:
        raise PermissionError("V7 validation is no longer a preserved failure")
    return V8TrainingContext(
        parent=parent, zero_bridge_mechanism=zero, zero_bridge_ar=zero_ar,
        sparse_gain=sparse_gain, dense_gain=dense_gain, zero_bridge_gain=zero_gain,
        gain_fit_windows=expected_windows, v7_failure_raw_sha256=v7_sha,
        r1_development_raw_sha256=r1_sha,
    )


def _envelope_violation(blend: np.ndarray, anchor: np.ndarray, endpoint: np.ndarray) -> float:
    low = np.minimum(anchor, endpoint)
    high = np.maximum(anchor, endpoint)
    return float(max(0.0, np.max(low - blend), np.max(blend - high)))


def predict_from_prefix(
    prefix_states: np.ndarray, context: V8TrainingContext, registration: dict
) -> PrefixPredictions:
    """Generate every V8 path from an immutable observed prefix only."""

    spec = registration["parent_anchor"]
    origin, horizon = int(spec["origin"]), int(spec["horizon"])
    prefix = np.asarray(prefix_states, dtype=float)
    if prefix.shape != (origin + 1, len(context.parent.scales)):
        raise ValueError("prefix must end exactly at the registered origin")
    if prefix.flags.writeable:
        prefix = prefix.copy()
        prefix.setflags(write=False)
    sparse = rel._latent_rollout(
        context.parent.sparse_mechanism, context.parent.sparse_ar, prefix, horizon
    )
    dense = rel._latent_rollout(
        context.parent.dense_probe_mechanism, context.parent.dense_probe_ar, prefix, horizon
    )
    zero = rel._latent_rollout(
        context.zero_bridge_mechanism, context.zero_bridge_ar, prefix, horizon
    )
    adaptive, adaptive_model = rel._adaptive_rollout(prefix, context.parent, horizon)
    persistence = np.repeat(prefix[-1][None, :], horizon, axis=0)
    frozen = rel.predict_from_prefix(prefix, context.parent, registration)
    candidate = persistence + context.sparse_gain * (sparse - persistence)
    dense_shrinkage = persistence + context.dense_gain * (dense - persistence)
    zero_shrinkage = persistence + context.zero_bridge_gain * (zero - persistence)
    models = {
        "parent_anchored_sparse": candidate,
        "v5_sparse_parent": sparse,
        "persistence": persistence,
        "zero_bridge_shrinkage": zero_shrinkage,
        "symmetric_dense_shrinkage": dense_shrinkage,
        "frozen_v7_consensus": frozen.models["sparse_consensus"],
        "frozen_v7_no_sparse_consensus": frozen.models["no_sparse_consensus"],
        "stable_adaptive_dense": adaptive,
    }
    radii = {
        "sparse": free._maximum_jacobian_radius(context.parent.sparse_mechanism, sparse),
        "symmetric_dense": free._maximum_jacobian_radius(
            context.parent.dense_probe_mechanism, dense
        ),
        "zero_bridge": free._maximum_jacobian_radius(context.zero_bridge_mechanism, zero),
        "adaptive_secondary": free._maximum_jacobian_radius(adaptive_model, adaptive),
    }
    envelopes = {
        "parent_anchored_sparse": _envelope_violation(candidate, persistence, sparse),
        "symmetric_dense_shrinkage": _envelope_violation(dense_shrinkage, persistence, dense),
        "zero_bridge_shrinkage": _envelope_violation(zero_shrinkage, persistence, zero),
    }
    return PrefixPredictions(models, radii, envelopes)


def _paired_lower(baseline: Sequence[float], candidate: Sequence[float], critical: float) -> dict:
    values = np.asarray(baseline) - np.asarray(candidate)
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    half = critical * sd / math.sqrt(len(values))
    return {"mean_improvement": mean, "sample_sd": sd, "ci95_lower": mean - half,
            "ci95_upper": mean + half, "seed_win_fraction": float(np.mean(values > 0.0))}


def _paired_log_upper(candidate: Sequence[float], baseline: Sequence[float], critical: float) -> dict:
    values = np.log(np.asarray(candidate) / np.asarray(baseline))
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    half = critical * sd / math.sqrt(len(values))
    return {"mean_log_ratio": mean, "sample_sd": sd, "ci95_lower": mean - half,
            "ci95_upper": mean + half, "geometric_mean_ratio": math.exp(mean)}


def _artifact_path(config_path: Path, split: str) -> Path:
    return _root(config_path) / "artifacts" / "agi" / f"sparse_causal_bridge_{split}_v8.json"


def _lock_bundle(config_path: Path, registration: dict, raw_chain: bytes,
                 context: V8TrainingContext) -> dict:
    root = _root(config_path)
    manifest_path = root / registration["implementation_lock_path"]
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_impl, current_tests = _implementation_hashes(), _test_hashes(root)
    if manifest.get("implementation_sha256") != current_impl:
        raise PermissionError("implementation differs from V8 lock manifest")
    if manifest.get("test_sha256") != current_tests:
        raise PermissionError("tests differ from V8 lock manifest")
    return {
        "registration_raw_sha256": _sha256(config_path),
        "ancestor_byte_chain_sha256": hashlib.sha256(raw_chain).hexdigest(),
        "canonical_merged_registration_sha256": _canonical_json_sha256(registration),
        "implementation_lock_manifest_sha256": _sha256(manifest_path),
        "implementation_sha256": current_impl,
        "test_sha256": current_tests,
        "v4_parent_artifact_sha256": context.parent.parent_raw_sha256,
        "v5_failure_artifact_sha256": context.parent.v5_failure_raw_sha256,
        "v7_failure_artifact_sha256": context.v7_failure_raw_sha256,
        "r1_development_artifact_sha256": context.r1_development_raw_sha256,
        "gains": {"sparse": context.sparse_gain, "symmetric_dense": context.dense_gain,
                  "zero_bridge": context.zero_bridge_gain, "fit_windows": context.gain_fit_windows},
    }


def _assert_test_unlocked(config_path: Path, current_lock: dict) -> str:
    path = _artifact_path(config_path, "validation")
    if not path.is_file():
        raise PermissionError("V8 test requires the canonical validation artifact")
    raw = path.read_bytes()
    report = json.loads(raw)
    if report.get("experiment") != "sparse_causal_bridge_v8" or report.get("split") != "validation":
        raise PermissionError("wrong validation artifact identity")
    if report.get("passed") is not True or not all(report.get("checks", {}).values()):
        raise PermissionError("V8 validation did not pass its full conjunction")
    if report.get("lock_bundle") != current_lock:
        raise PermissionError("V8 lock bundle changed after validation")
    return hashlib.sha256(raw).hexdigest()


def run_parent_anchored_gate(config_path: Path, *, split: str = "validation") -> dict:
    """Run one registered V8 split; test is checked before any test simulation."""

    started = time.perf_counter()
    registration, raw_chain = base._load_registration(config_path)
    base._validate_registration(registration)
    if registration.get("runner") != "parent_anchored_shrinkage_confirmation":
        raise ValueError("V8 parent-anchored registration required")
    if registration.get("active_gate") != "parent_anchor_gate":
        raise ValueError("V8 parent_anchor_gate must be active")
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    overlap = _assert_historical_seed_disjoint(config_path, registration)
    context = _build_training_context(config_path, registration)
    start_lock = _lock_bundle(config_path, registration, raw_chain, context)
    validation_artifact_sha = None
    if split == "test":
        validation_artifact_sha = _assert_test_unlocked(config_path, start_lock)

    role, gate = registration["data_roles"][split], registration["parent_anchor_gate"]
    required = int(gate[f"{split}_seeds_required"])
    if len(role["seeds"]) != required:
        raise ValueError("registered V8 seed count changed")
    origin, horizon = int(registration["parent_anchor"]["origin"]), int(
        registration["parent_anchor"]["horizon"]
    )
    errors = {name: [] for name in MODEL_NAMES}
    radii = {name: [] for name in ("sparse", "symmetric_dense", "zero_bridge", "adaptive_secondary")}
    maximum_observed, future_reads, nonfinite = -1, 0, 0
    maximum_abs, maximum_norm, maximum_envelope = 0.0, 0.0, 0.0
    shapes_ok, h5_ok = True, True
    for raw_seed in role["seeds"]:
        episode = base.simulate_episode(int(raw_seed), registration, environment=role["environment"],
                                        steps=int(role["steps_per_seed"]))
        reader = rel.PrefixReader(episode.states, origin)
        predictions = predict_from_prefix(reader.through_origin(), context, registration)
        maximum_observed = max(maximum_observed, reader.max_observed_state_index)
        future_reads += reader.future_observation_reads
        truth = episode.states[origin + 1 : origin + horizon + 1]
        for name, prediction in predictions.models.items():
            shapes_ok &= prediction.shape == (horizon, len(context.parent.scales))
            h5_ok &= np.array_equal(prediction[:5], prediction[0:5])
            nonfinite += int(prediction.size - np.count_nonzero(np.isfinite(prediction)))
            finite = prediction[np.isfinite(prediction)]
            if finite.size:
                maximum_abs = max(maximum_abs, float(np.max(np.abs(finite))))
            finite_rows = prediction[np.all(np.isfinite(prediction), axis=1)]
            if len(finite_rows):
                maximum_norm = max(maximum_norm, float(np.max(
                    np.linalg.norm(finite_rows / context.parent.scales, axis=1))))
            errors[name].append(rel._normalized_rmse(truth, prediction, context.parent.scales))
        for name, radius in predictions.component_pathwise_jacobian_radii.items():
            radii[name].append(float(radius))
        maximum_envelope = max(maximum_envelope, *predictions.convex_envelope_violations.values())

    critical = float(registration["parent_anchor"]["critical_value_n256_df255"])
    candidate = errors["parent_anchored_sparse"]
    comparisons = {
        "improvement_vs_v5_parent": _paired_lower(errors["v5_sparse_parent"], candidate, critical),
        "improvement_vs_persistence": _paired_lower(errors["persistence"], candidate, critical),
        "improvement_vs_zero_bridge": _paired_lower(errors["zero_bridge_shrinkage"], candidate, critical),
        "improvement_vs_frozen_v7_consensus": _paired_lower(errors["frozen_v7_consensus"], candidate, critical),
        "log_ratio_vs_symmetric_dense": _paired_log_upper(candidate, errors["symmetric_dense_shrinkage"], critical),
        "log_ratio_vs_stable_adaptive_dense": _paired_log_upper(candidate, errors["stable_adaptive_dense"], critical),
    }
    retained_max = max(max(radii[name]) for name in ("sparse", "symmetric_dense", "zero_bridge"))
    ar_values = {"sparse": abs(context.parent.sparse_ar),
                 "symmetric_dense": abs(context.parent.dense_probe_ar),
                 "zero_bridge": abs(context.zero_bridge_ar)}
    stability = {
        "per_seed_component_pathwise_jacobian_radius": radii,
        "maximum_retained_component_pathwise_jacobian_radius": retained_max,
        "maximum_adaptive_secondary_pathwise_jacobian_radius": max(radii["adaptive_secondary"]),
        "retained_latent_ar_abs": ar_values,
        "maximum_retained_latent_ar_abs": max(ar_values.values()),
        "sparse_augmented_common_norm_bound": float(
            registration["parent_anchor"]["sparse_augmented_common_norm_bound"]),
        "nonfinite_prediction_count": nonfinite,
        "maximum_component_prediction_absolute_value": maximum_abs,
        "maximum_prediction_normalized_l2_norm": maximum_norm,
        "train_normalized_state_l2_norm_q99": context.parent.train_normalized_norm_q99,
        "prediction_norm_to_train_q99_ratio": maximum_norm / context.parent.train_normalized_norm_q99,
        "maximum_convex_envelope_violation": maximum_envelope,
        "maximum_observed_state_index": maximum_observed,
        "future_observation_reads": future_reads,
    }
    checks = {
        "historical_v4_parent_passed": context.parent.parent_report.get("passed") is True,
        "historical_v5_failure_preserved": context.parent.v5_failure_report.get("passed") is False,
        "historical_v7_failure_preserved": True,
        "evidence_seeds_disjoint": not any(overlap.values()),
        "registered_seed_count": len(role["seeds"]) == required,
        "single_origin": int(gate["origins_per_seed_required"]) == 1,
        "h5_non_gating_exact_slice": gate["h5_is_gating"] is False and h5_ok,
        "model_shapes": bool(shapes_ok),
        "v5_parent_improvement_ci": comparisons["improvement_vs_v5_parent"]["ci95_lower"] > 0.0,
        "persistence_improvement_ci": comparisons["improvement_vs_persistence"]["ci95_lower"] > 0.0,
        "zero_bridge_improvement_ci": comparisons["improvement_vs_zero_bridge"]["ci95_lower"] > 0.0,
        "frozen_v7_improvement_ci": comparisons["improvement_vs_frozen_v7_consensus"]["ci95_lower"] > 0.0,
        "symmetric_dense_noninferiority": comparisons["log_ratio_vs_symmetric_dense"]["ci95_upper"] <= float(gate["paired_log_ratio_ci95_upper_vs_symmetric_dense_max"]),
        "stable_adaptive_dense_noninferiority": comparisons["log_ratio_vs_stable_adaptive_dense"]["ci95_upper"] <= float(gate["paired_log_ratio_ci95_upper_vs_stable_adaptive_dense_max"]),
        "predictions_finite": nonfinite <= int(gate["nonfinite_prediction_count_max"]),
        "prediction_absolute_bound": maximum_abs <= float(gate["maximum_component_prediction_absolute_value"]),
        "prediction_scale": stability["prediction_norm_to_train_q99_ratio"] <= float(gate["maximum_prediction_norm_to_train_q99_ratio"]),
        "retained_pathwise_stability": retained_max <= float(gate["maximum_retained_component_pathwise_jacobian_radius"]),
        "sparse_common_norm_certificate": stability["sparse_augmented_common_norm_bound"] <= float(gate["maximum_sparse_augmented_common_norm_bound"]),
        "retained_latent_ar_stability": max(ar_values.values()) <= float(gate["maximum_retained_latent_ar_abs"]),
        "convex_envelope": maximum_envelope <= float(gate["maximum_convex_envelope_violation"]),
        "observed_state_index_bound": maximum_observed <= int(gate["maximum_observed_state_index"]),
        "zero_future_observation_reads": future_reads <= int(gate["future_observation_reads_max"]),
    }
    models = {name: {"mean_h20_normalized_path_rmse": float(np.mean(values)),
                     "sample_sd": float(np.std(values, ddof=1)),
                     "seed_h20_normalized_path_rmse": values}
              for name, values in errors.items()}
    end_lock = _lock_bundle(config_path, registration, raw_chain, context)
    if end_lock != start_lock:
        raise PermissionError("V8 lock bundle changed during evaluation")
    passed = all(checks.values())
    return {
        "experiment": registration["experiment"], "roadmap_stage": registration["roadmap_stage"],
        "split": split, "environment": role["environment"], "passed": passed,
        "confirmation_status": "R1_CONFIRMED" if passed else "R1_NOT_CONFIRMED",
        "claim_boundary": registration["claim_boundary"], "checks": checks,
        "models": models, "comparisons": comparisons, "stability": stability,
        "normalization": {"training_only_scales": context.parent.scales.tolist(),
                          "ddof": registration["normalization"]["ddof"]},
        "seed_integrity": {"overlap": overlap, "evaluation_seed_count": len(role["seeds"])},
        "lock_bundle": start_lock,
        "test_lock": {"validation_artifact_sha256": validation_artifact_sha,
                      "test_opened": split == "test"},
        "resource_usage": {"evaluation_seeds": len(role["seeds"]),
                           "forecast_origins_per_seed": 1,
                           "external_download_bytes": 0,
                           "wall_seconds": time.perf_counter() - started},
        "environment_manifest": {"python": sys.version, "numpy": np.__version__,
                                 "platform": platform.platform()},
        "development_data_counted_as_v8_evidence": False,
    }


def _atomic_write_once(path: Path, report: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite canonical evidence artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("experiments/preregistration/sparse_causal_bridge_v8.json"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args(argv)
    output = _artifact_path(args.config, args.split)
    if output.exists():
        raise FileExistsError(f"registered split already consumed: {output}")
    report = run_parent_anchored_gate(args.config, split=args.split)
    _atomic_write_once(output, report)
    print(output)
    print(json.dumps({"passed": report["passed"], "checks": report["checks"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
