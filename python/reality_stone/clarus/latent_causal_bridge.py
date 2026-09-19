"""V3 latent-context extension of the sparse causal bridge experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import sparse_causal_bridge as base


@dataclass(frozen=True)
class ResidualFilter:
    center: np.ndarray
    direction: np.ndarray
    intercept: float
    autoregression: float
    variance_fraction: float

    def predict_next(self, previous_residual: np.ndarray) -> np.ndarray:
        score = float(self.direction @ (previous_residual - self.center))
        next_score = self.intercept + self.autoregression * score
        return self.center + self.direction * next_score


def estimate_full_mechanism(
    probe: base.ProbeBatch, amplitude: float
) -> dict[base.Edge, dict[str, float | bool]]:
    """Estimate diagonal linear dynamics and off-diagonal tanh effects."""

    size = probe.x_plus.shape[1]
    result: dict[base.Edge, dict[str, float | bool]] = {}
    for source in range(size):
        rows = np.flatnonzero(probe.source == source)
        if len(rows) < 4:
            raise ValueError("each source requires at least four paired probes")
        for target in range(size):
            denominator = 2.0 * amplitude if source == target else 2.0 * np.tanh(amplitude)
            effects = (probe.y_plus[rows, target] - probe.y_minus[rows, target]) / denominator
            estimate = float(np.mean(effects))
            standard_error = float(np.std(effects, ddof=1) / np.sqrt(len(effects)))
            half = len(effects) // 2
            first = float(np.mean(effects[:half]))
            second = float(np.mean(effects[half:]))
            result[(source, target)] = {
                "estimate": estimate,
                "standard_error": standard_error,
                "z_score": abs(estimate) / max(standard_error, 1e-12),
                "split_half_sign_agreement": bool(
                    np.sign(first) == np.sign(second) and np.sign(first) != 0
                ),
                "samples": int(len(effects)),
            }
    return result


def mechanism_model(
    name: str,
    self_coefficients: np.ndarray,
    bridge: np.ndarray,
    edges: Sequence[base.Edge],
) -> base.BridgeModel:
    local = np.zeros((len(self_coefficients), 3), dtype=float)
    local[:, 1] = self_coefficients
    return base.BridgeModel(name, local, bridge.copy(), tuple(edges))


def fit_residual_filter(
    episode: base.Episode,
    mechanism: base.BridgeModel,
    calibration_steps: int,
    *,
    autoregression_override: float | None = None,
) -> ResidualFilter:
    if calibration_steps < 4 or calibration_steps >= len(episode.states) - 1:
        raise ValueError("calibration steps must leave evaluation rows")
    states = episode.states[:calibration_steps]
    outcomes = episode.states[1 : calibration_steps + 1]
    residuals = outcomes - mechanism.predict(states)
    center = np.mean(residuals, axis=0)
    centered = residuals - center
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, -1]
    variance_fraction = float(eigenvalues[-1] / max(np.sum(eigenvalues), 1e-12))
    scores = centered @ direction
    if autoregression_override is None:
        design = np.column_stack((np.ones(len(scores) - 1), scores[:-1]))
        intercept, autoregression = np.linalg.lstsq(design, scores[1:], rcond=None)[0]
    else:
        autoregression = float(autoregression_override)
        intercept = float(np.mean(scores[1:] - autoregression * scores[:-1]))
    return ResidualFilter(
        center=center,
        direction=direction,
        intercept=float(intercept),
        autoregression=float(autoregression),
        variance_fraction=variance_fraction,
    )


def fit_pooled_residual_autoregression(
    episodes: Sequence[base.Episode],
    mechanism: base.BridgeModel,
) -> float:
    """Estimate one invariant scalar AR without crossing episode boundaries."""

    residual_sequences = [
        episode.states[1:] - mechanism.predict(episode.states[:-1])
        for episode in episodes
    ]
    pooled = np.concatenate(residual_sequences)
    center = np.mean(pooled, axis=0)
    covariance = (pooled - center).T @ (pooled - center) / max(len(pooled) - 1, 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, -1]
    previous_scores = []
    next_scores = []
    for residuals in residual_sequences:
        scores = (residuals - center) @ direction
        previous_scores.append(scores[:-1])
        next_scores.append(scores[1:])
    previous = np.concatenate(previous_scores)
    following = np.concatenate(next_scores)
    design = np.column_stack((np.ones(len(previous)), previous))
    return float(np.linalg.lstsq(design, following, rcond=None)[0][1])


def sequential_filter_prediction(
    episode: base.Episode,
    mechanism: base.BridgeModel,
    residual_filter: ResidualFilter,
    calibration_steps: int,
    *,
    bridge_override: np.ndarray | None = None,
) -> np.ndarray:
    predictions = []
    prediction_model = mechanism
    if bridge_override is not None:
        prediction_model = base.BridgeModel(
            mechanism.name,
            mechanism.local_coefficients,
            bridge_override,
            mechanism.declared_edges,
        )
    for time_index in range(calibration_steps, len(episode.states) - 1):
        previous_residual = episode.states[time_index] - mechanism.predict(
            episode.states[time_index - 1]
        )[0]
        latent_prediction = residual_filter.predict_next(previous_residual)
        prediction = prediction_model.predict(episode.states[time_index])[0]
        predictions.append(prediction + latent_prediction)
    return np.asarray(predictions)


def _prefix_episode(episode: base.Episode, steps: int) -> base.Episode:
    return base.Episode(
        states=episode.states[: steps + 1],
        hidden=episode.hidden[: steps + 1],
    )


def _implementation_hashes() -> dict[str, str]:
    paths = {
        "latent_causal_bridge.py": Path(__file__).resolve(),
        "sparse_causal_bridge.py": Path(base.__file__).resolve(),
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def _assert_v3_test_unlocked(
    config_path: Path,
    registration: dict,
    config_sha: str,
) -> None:
    version = registration["experiment"].rsplit("_", 1)[-1]
    root = config_path.resolve().parents[2]
    path = root / "artifacts" / "agi" / f"sparse_causal_bridge_validation_{version}.json"
    if not path.exists():
        raise PermissionError("V3 test requires a saved passing validation artifact")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise PermissionError("V3 validation artifact did not pass")
    if report.get("registration", {}).get("sha256") != config_sha:
        raise PermissionError("V3 validation registration hash changed")
    if report.get("implementation_sha256") != _implementation_hashes():
        raise PermissionError("V3 implementation hash changed after validation")


def _evaluate_latent_models(
    registration: dict,
    split: str,
    fixed_models: dict[str, base.BridgeModel],
    mechanism: base.BridgeModel,
    truth_edges: Sequence[base.Edge],
    shared_autoregression: float | None,
) -> dict:
    role = registration["data_roles"][split]
    calibration_steps = int(registration["latent_filter"]["ood_calibration_steps"])
    downstream = np.asarray(sorted({target for _, target in truth_edges}), dtype=int)
    all_edges = tuple(
        (source, target)
        for source in range(len(registration["charts"]))
        for target in range(len(registration["charts"]))
        if source != target
    )
    names = (
        "fixed_local_train",
        "fixed_dense_train",
        "adaptive_local_prefix",
        "adaptive_dense_prefix",
        "v1_bridge_observational_local",
        "causal_mechanism_no_latent",
        "causal_latent_filter",
        "oracle_hidden_diagnostic",
    )
    values = {
        name: {"global": [], "downstream": [], "intervention_nrmse": []}
        for name in names
    }
    cosine_values: list[float] = []
    ar_errors: list[float] = []
    variance_fractions: list[float] = []
    lesion_increases: list[float] = []
    true_bridge = base._true_bridge(registration)
    true_self = np.asarray(registration["scm"]["self_coefficients"], dtype=float)
    true_mechanism = mechanism_model("true_mechanism", true_self, true_bridge, truth_edges)
    loading = base._environment_loadings(registration, role["environment"])
    loading_direction = loading / np.linalg.norm(loading)
    for seed in role["seeds"]:
        episode = base.simulate_episode(
            int(seed),
            registration,
            environment=role["environment"],
            steps=int(role["steps_per_seed"]),
        )
        prefix = _prefix_episode(episode, calibration_steps)
        adaptive_local = base.fit_observational_model(
            "adaptive_local_prefix",
            [prefix],
            (),
            registration["learning"]["ridge"],
        )
        adaptive_dense = base.fit_observational_model(
            "adaptive_dense_prefix", [prefix], all_edges, registration["learning"]["ridge"]
        )
        residual_filter = fit_residual_filter(
            episode,
            mechanism,
            calibration_steps,
            autoregression_override=shared_autoregression,
        )
        cosine_values.append(abs(float(residual_filter.direction @ loading_direction)))
        ar_errors.append(
            abs(residual_filter.autoregression - float(registration["scm"]["latent_ar"]))
        )
        variance_fractions.append(residual_filter.variance_fraction)
        states = episode.states[calibration_steps:-1]
        outcomes = episode.states[calibration_steps + 1 :]
        predictions = {
            "fixed_local_train": fixed_models["fixed_local_train"].predict(states),
            "fixed_dense_train": fixed_models["fixed_dense_train"].predict(states),
            "adaptive_local_prefix": adaptive_local.predict(states),
            "adaptive_dense_prefix": adaptive_dense.predict(states),
            "v1_bridge_observational_local": fixed_models[
                "v1_bridge_observational_local"
            ].predict(states),
            "causal_mechanism_no_latent": mechanism.predict(states),
            "causal_latent_filter": sequential_filter_prediction(
                episode, mechanism, residual_filter, calibration_steps
            ),
            "oracle_hidden_diagnostic": true_mechanism.predict(states)
            + episode.hidden[calibration_steps:-1, None] * loading[None, :],
        }
        effect_models = {
            "fixed_local_train": fixed_models["fixed_local_train"],
            "fixed_dense_train": fixed_models["fixed_dense_train"],
            "adaptive_local_prefix": adaptive_local,
            "adaptive_dense_prefix": adaptive_dense,
            "v1_bridge_observational_local": fixed_models[
                "v1_bridge_observational_local"
            ],
            "causal_mechanism_no_latent": mechanism,
            "causal_latent_filter": mechanism,
            "oracle_hidden_diagnostic": true_mechanism,
        }
        probe = base.generate_probe(
            int(seed) + 400_009,
            registration,
            environment=role["environment"],
            stationary_steps=max(int(role["steps_per_seed"]), 128),
            pairs_per_source=int(role["intervention_pairs_per_source_per_seed"]),
        )
        for name, prediction in predictions.items():
            values[name]["global"].append(base._rmse(outcomes, prediction))
            values[name]["downstream"].append(
                base._rmse(outcomes[:, downstream], prediction[:, downstream])
            )
            values[name]["intervention_nrmse"].append(
                base._probe_effect_nrmse(effect_models[name], probe, truth_edges)
            )
        intact = predictions["causal_latent_filter"]
        for source, target in truth_edges:
            lesioned_bridge = mechanism.bridge.copy()
            lesioned_bridge[target, source] = 0.0
            lesioned = sequential_filter_prediction(
                episode,
                mechanism,
                residual_filter,
                calibration_steps,
                bridge_override=lesioned_bridge,
            )
            intact_mse = float(np.mean((outcomes[:, target] - intact[:, target]) ** 2))
            lesion_mse = float(np.mean((outcomes[:, target] - lesioned[:, target]) ** 2))
            lesion_increases.append((lesion_mse - intact_mse) / max(intact_mse, 1e-12))
    models = {
        name: {
            "mean_global_rmse": float(np.mean(data["global"])),
            "mean_downstream_rmse": float(np.mean(data["downstream"])),
            "mean_intervention_nrmse": float(np.mean(data["intervention_nrmse"])),
            "seed_global_rmse": data["global"],
            "seed_downstream_rmse": data["downstream"],
        }
        for name, data in values.items()
    }
    fixed_improvement = np.asarray(values["fixed_local_train"]["global"]) - np.asarray(
        values["causal_latent_filter"]["global"]
    )
    return {
        "models": models,
        "filter": {
            "shared_train_scalar_ar": shared_autoregression,
            "mean_loading_subspace_cosine": float(np.mean(cosine_values)),
            "minimum_loading_subspace_cosine": float(np.min(cosine_values)),
            "mean_scalar_ar_abs_error": float(np.mean(ar_errors)),
            "mean_rank_one_variance_fraction": float(np.mean(variance_fractions)),
            "seed_loading_subspace_cosine": cosine_values,
            "seed_scalar_ar_abs_error": ar_errors,
        },
        "paired_ci95_lower_vs_fixed_local": base._paired_ci95_lower(fixed_improvement),
        "minimum_lesion_direct_target_mse_increase_fraction": float(
            np.min(lesion_increases)
        ),
    }


def run_latent_causal_bridge_gate(
    config_path: Path,
    *,
    split: str = "validation",
    enforce_test_lock: bool = True,
) -> dict:
    started = time.perf_counter()
    registration, raw = base._load_registration(config_path)
    base._validate_registration(registration)
    if registration.get("runner") != "latent_residual_filter":
        raise ValueError("V3 latent runner registration required")
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    config_sha = hashlib.sha256(raw).hexdigest()
    if split == "test" and enforce_test_lock:
        _assert_v3_test_unlocked(config_path, registration, config_sha)

    data = registration["data_roles"]
    train_role = data["observational_train"]
    holdout_role = data["observational_selector_holdout"]
    train = [
        base.simulate_episode(
            seed,
            registration,
            environment=train_role["environment"],
            steps=int(train_role["steps_per_seed"]),
        )
        for seed in train_role["seeds"]
    ]
    holdout = [
        base.simulate_episode(
            seed,
            registration,
            environment=holdout_role["environment"],
            steps=int(holdout_role["steps_per_seed"]),
        )
        for seed in holdout_role["seeds"]
    ]
    geometry = base.laplace_beltrami_proposal(registration)
    observational = base.observational_edge_diagnostics(
        train, holdout, registration["learning"]["ridge"]
    )
    probe_role = data["topology_intervention_probe"]
    probes = [
        base.generate_probe(
            seed,
            registration,
            environment=probe_role["environment"],
            stationary_steps=int(probe_role["stationary_steps_per_seed"]),
            pairs_per_source=int(probe_role["pairs_per_source_per_seed"]),
        )
        for seed in probe_role["seeds"]
    ]
    pooled_probe = base.combine_probes(probes)
    full_effects = estimate_full_mechanism(
        pooled_probe, float(registration["intervention"]["amplitude"])
    )
    causal_edges = base.select_causal_edges(
        full_effects, observational, geometry, registration["learning"]
    )
    size = len(registration["charts"])
    all_edges = tuple(
        (source, target)
        for source in range(size)
        for target in range(size)
        if source != target
    )
    bridge = base._bridge_from_estimates(size, full_effects, causal_edges)
    self_coefficients = np.asarray(
        [full_effects[(index, index)]["estimate"] for index in range(size)]
    )
    mechanism = mechanism_model(
        "causal_mechanism", self_coefficients, bridge, causal_edges
    )
    fixed_models = {
        "fixed_local_train": base.fit_observational_model(
            "fixed_local_train", train, (), registration["learning"]["ridge"]
        ),
        "fixed_dense_train": base.fit_observational_model(
            "fixed_dense_train", train, all_edges, registration["learning"]["ridge"]
        ),
        "v1_bridge_observational_local": base.fit_fixed_bridge_model(
            "v1_bridge_observational_local",
            train,
            bridge,
            causal_edges,
            registration["learning"]["ridge"],
        ),
    }
    truth_edges = tuple(
        (int(item["source"]), int(item["target"]))
        for item in registration["scm"]["true_directed_bridges"]
    )
    shared_autoregression = None
    if registration["latent_filter"].get("scalar_ar_source") == (
        "pooled_observational_train_mechanism_residuals"
    ):
        shared_autoregression = fit_pooled_residual_autoregression(train, mechanism)
    evaluation = _evaluate_latent_models(
        registration,
        split,
        fixed_models,
        mechanism,
        truth_edges,
        shared_autoregression,
    )

    per_probe_exact = []
    for probe in probes:
        effects = estimate_full_mechanism(
            probe, float(registration["intervention"]["amplitude"])
        )
        selected = base.select_causal_edges(
            effects, observational, geometry, registration["learning"]
        )
        per_probe_exact.append(set(selected) == set(truth_edges))
    controls = registration["negative_controls"]
    null_probe = base.generate_probe(
        int(controls["no_bridge_seed"]),
        registration,
        environment="train",
        stationary_steps=int(probe_role["stationary_steps_per_seed"]),
        pairs_per_source=int(probe_role["pairs_per_source_per_seed"]),
        bridge_override=np.zeros((size, size)),
    )
    null_effects = estimate_full_mechanism(
        null_probe, float(registration["intervention"]["amplitude"])
    )
    null_selected = base.select_causal_edges(
        null_effects, observational, geometry, registration["learning"]
    )
    permuted = base.permute_probe_signs(
        pooled_probe, int(controls["permuted_intervention_seed"])
    )
    permuted_effects = estimate_full_mechanism(
        permuted, float(registration["intervention"]["amplitude"])
    )
    permuted_selected = base.select_causal_edges(
        permuted_effects, observational, geometry, registration["learning"]
    )

    truth_bridge = base._true_bridge(registration)
    truth_self = np.asarray(registration["scm"]["self_coefficients"], dtype=float)
    graph = base._graph_metrics(causal_edges, truth_edges)
    common = tuple(registration["scm"]["registered_common_cause_pair"])
    common_edges = {common, common[::-1]}
    reverse_edges = {(target, source) for source, target in truth_edges}
    selected_set = set(causal_edges)
    bridge_mae = float(
        np.mean(
            [
                abs(bridge[target, source] - truth_bridge[target, source])
                for source, target in truth_edges
            ]
        )
    )
    self_mae = float(np.mean(np.abs(self_coefficients - truth_self)))
    sign_accuracy = float(
        np.mean(
            [
                np.sign(bridge[target, source]) == np.sign(truth_bridge[target, source])
                for source, target in truth_edges
            ]
        )
    )
    truth_pairs = {tuple(sorted(edge)) for edge in truth_edges}
    proposal_coverage = len(set(geometry["pairs"]) & truth_pairs) / len(truth_pairs)

    metrics = evaluation["models"]
    causal_global = metrics["causal_latent_filter"]["mean_global_rmse"]
    causal_downstream = metrics["causal_latent_filter"]["mean_downstream_rmse"]
    comparisons = {
        "global_reduction_vs_fixed_local": base._reduction(
            metrics["fixed_local_train"]["mean_global_rmse"], causal_global
        ),
        "global_reduction_vs_v1_bridge": base._reduction(
            metrics["v1_bridge_observational_local"]["mean_global_rmse"], causal_global
        ),
        "global_reduction_vs_no_latent": base._reduction(
            metrics["causal_mechanism_no_latent"]["mean_global_rmse"], causal_global
        ),
        "global_rmse_ratio_vs_adaptive_dense": causal_global
        / max(metrics["adaptive_dense_prefix"]["mean_global_rmse"], 1e-12),
        "downstream_reduction_vs_fixed_local": base._reduction(
            metrics["fixed_local_train"]["mean_downstream_rmse"], causal_downstream
        ),
        "paired_ci95_lower_vs_fixed_local": evaluation[
            "paired_ci95_lower_vs_fixed_local"
        ],
    }
    gate = registration["latent_gate"]
    checks = {
        "spectral_stability": base._joint_spectral_radius(registration)
        <= gate["max_joint_spectral_radius"],
        "geometry_truth_coverage": proposal_coverage
        >= gate["geometry_truth_pair_coverage_min"],
        "directed_precision": graph["precision"] >= gate["directed_precision_min"],
        "directed_recall": graph["recall"] >= gate["directed_recall_min"],
        "exact_graph_recovery_stability": float(np.mean(per_probe_exact))
        >= gate["exact_graph_recovery_probe_seed_fraction_min"],
        "common_cause_rejected": len(selected_set & common_edges)
        <= gate["common_cause_false_edges_max"],
        "reverse_edges_rejected": len(selected_set & reverse_edges)
        <= gate["reverse_true_edges_max"],
        "edge_budget": len(causal_edges) <= gate["selected_edges_max"],
        "self_coefficient_mae": self_mae <= gate["self_coefficient_mae_max"],
        "bridge_coefficient_mae": bridge_mae <= gate["bridge_coefficient_mae_max"],
        "true_effect_sign": sign_accuracy >= gate["true_effect_sign_accuracy_min"],
        "intervention_response": metrics["causal_latent_filter"][
            "mean_intervention_nrmse"
        ]
        <= gate["intervention_response_nrmse_max"],
        "latent_subspace_mean": evaluation["filter"][
            "mean_loading_subspace_cosine"
        ]
        >= gate["latent_subspace_cosine_mean_min"],
        "latent_subspace_seed_min": evaluation["filter"][
            "minimum_loading_subspace_cosine"
        ]
        >= gate["latent_subspace_cosine_seed_min"],
        "latent_ar_error": evaluation["filter"]["mean_scalar_ar_abs_error"]
        <= gate["latent_scalar_ar_abs_error_mean_max"],
        "rank_one_variance": evaluation["filter"][
            "mean_rank_one_variance_fraction"
        ]
        >= gate["rank_one_residual_variance_fraction_mean_min"],
        "global_vs_fixed_local": comparisons["global_reduction_vs_fixed_local"]
        >= gate["ood_global_rmse_reduction_vs_fixed_local_min"],
        "global_vs_v1_bridge": comparisons["global_reduction_vs_v1_bridge"]
        >= gate["ood_global_rmse_reduction_vs_v1_bridge_min"],
        "global_vs_no_latent": comparisons["global_reduction_vs_no_latent"]
        >= gate["ood_global_rmse_reduction_vs_no_latent_min"],
        "adaptive_dense_noninferiority": comparisons[
            "global_rmse_ratio_vs_adaptive_dense"
        ]
        <= gate["ood_global_rmse_ratio_vs_adaptive_dense_max"],
        "downstream_vs_fixed_local": comparisons[
            "downstream_reduction_vs_fixed_local"
        ]
        >= gate["ood_downstream_rmse_reduction_vs_fixed_local_min"],
        "paired_ci_vs_fixed_local": comparisons["paired_ci95_lower_vs_fixed_local"]
        > gate["paired_ci95_lower_vs_fixed_local_min"],
        "lesion_direct_target": evaluation[
            "minimum_lesion_direct_target_mse_increase_fraction"
        ]
        >= gate["lesion_direct_target_mse_increase_fraction_min"],
        "no_bridge_negative_control": len(null_selected)
        <= controls["max_selected_edges_each"],
        "permuted_intervention_negative_control": len(permuted_selected)
        <= controls["max_selected_edges_each"],
    }
    finite_payload = {
        "models": metrics,
        "filter": evaluation["filter"],
        "comparisons": comparisons,
    }
    checks["finite_metrics"] = base._all_finite(finite_payload)
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
        "selection": {
            "causal_edges": base._edge_list(causal_edges, registration["charts"]),
            "truth_edges_evaluation_only": base._edge_list(
                truth_edges, registration["charts"]
            ),
            "geometry_proposed_pairs": [
                f"{registration['charts'][left]}--{registration['charts'][right]}"
                for left, right in geometry["pairs"]
            ],
            "geometry_truth_pair_coverage": proposal_coverage,
            "exact_recovery_probe_seed_fraction": float(np.mean(per_probe_exact)),
            "self_coefficient_estimates": self_coefficients.tolist(),
            "full_intervention_diagnostics": base._serialize_edge_diagnostics(
                full_effects, registration["charts"]
            ),
        },
        "graph_metrics": {
            **graph,
            "common_cause_false_edges": len(selected_set & common_edges),
            "reverse_true_edges": len(selected_set & reverse_edges),
            "self_coefficient_mae": self_mae,
            "bridge_coefficient_mae": bridge_mae,
            "true_effect_sign_accuracy": sign_accuracy,
        },
        "models": metrics,
        "latent_filter": evaluation["filter"],
        "comparisons": comparisons,
        "lesion": {
            "minimum_direct_target_mse_increase_fraction": evaluation[
                "minimum_lesion_direct_target_mse_increase_fraction"
            ]
        },
        "negative_controls": {
            "no_bridge_selected_edges": base._edge_list(
                null_selected, registration["charts"]
            ),
            "permuted_intervention_selected_edges": base._edge_list(
                permuted_selected, registration["charts"]
            ),
        },
        "checks": checks,
        "resource_checks": resource_checks,
        "resource_usage": {
            "wall_seconds": elapsed,
            "external_download_bytes": 0,
            "trajectory_files_written": 0,
            "topology_probe_pairs": int(len(pooled_probe.source)),
            "ood_calibration_steps_per_seed": registration["latent_filter"][
                "ood_calibration_steps"
            ],
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
    report = run_latent_causal_bridge_gate(args.config, split=args.split)
    output = args.output or _default_output(args.config, args.split, registration["experiment"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"artifact: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
