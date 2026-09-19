"""Sparse causal bridge gate with a discrete Laplace--Beltrami proposal prior.

The experiment deliberately keeps chart identity fixed.  Geometry proposes a
small set of undirected contacts; randomized one-step interventions decide
which directed links exist and estimate their coefficients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


Edge = tuple[int, int]  # (source, target)


@dataclass(frozen=True)
class Episode:
    states: np.ndarray
    hidden: np.ndarray


@dataclass(frozen=True)
class ProbeBatch:
    source: np.ndarray
    x_plus: np.ndarray
    x_minus: np.ndarray
    y_plus: np.ndarray
    y_minus: np.ndarray

    def subset(self, source: int) -> "ProbeBatch":
        keep = self.source == source
        return ProbeBatch(
            source=self.source[keep],
            x_plus=self.x_plus[keep],
            x_minus=self.x_minus[keep],
            y_plus=self.y_plus[keep],
            y_minus=self.y_minus[keep],
        )


@dataclass(frozen=True)
class BridgeModel:
    name: str
    local_coefficients: np.ndarray
    bridge: np.ndarray
    declared_edges: tuple[Edge, ...]

    def predict(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        if states.ndim == 1:
            states = states[None, :]
        prediction = np.empty_like(states)
        for target in range(states.shape[1]):
            local = np.column_stack(
                (
                    np.ones(len(states)),
                    states[:, target],
                    states[:, target] ** 3,
                )
            )
            prediction[:, target] = local @ self.local_coefficients[target]
        prediction += np.tanh(states) @ self.bridge.T
        return prediction


def _edge_key(edge: Edge) -> str:
    return f"{edge[0]}->{edge[1]}"


def _edge_list(edges: Iterable[Edge], chart_names: Sequence[str]) -> list[str]:
    return [f"{chart_names[source]}->{chart_names[target]}" for source, target in edges]


def _true_bridge(registration: dict) -> np.ndarray:
    size = len(registration["charts"])
    bridge = np.zeros((size, size), dtype=float)
    for item in registration["scm"]["true_directed_bridges"]:
        bridge[int(item["target"]), int(item["source"])] = float(item["coefficient"])
    return bridge


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_registration(config_path: Path) -> tuple[dict, bytes]:
    raw = config_path.read_bytes()
    registration = json.loads(raw)
    if "extends" not in registration:
        return registration, raw
    base, base_raw = _load_registration(config_path.parent / registration["extends"])
    merged = _deep_merge(base, registration.get("overrides", {}))
    for key, value in registration.items():
        if key not in {"overrides"}:
            merged[key] = value
    return merged, base_raw + raw


def _validate_registration(registration: dict) -> None:
    charts = registration["charts"]
    size = len(charts)
    if size != 4 or len(set(charts)) != size:
        raise ValueError("V1 requires four uniquely named fixed charts")
    scm = registration["scm"]
    for key in ("self_coefficients", "train_hidden_loadings", "ood_hidden_loadings"):
        if len(scm[key]) != size:
            raise ValueError(f"{key} must have one value per chart")
    true_edges: set[Edge] = set()
    for item in scm["true_directed_bridges"]:
        edge = (int(item["source"]), int(item["target"]))
        if edge[0] == edge[1] or min(edge) < 0 or max(edge) >= size or edge in true_edges:
            raise ValueError("true directed bridges must be unique, valid, and off-diagonal")
        true_edges.add(edge)
    contact = np.asarray(registration["geometry_proposal"]["white_contact_affinity"])
    if contact.shape != (size, size) or not np.allclose(contact, contact.T):
        raise ValueError("white contact affinity must be a symmetric chart matrix")
    seed_groups = []
    for role in registration["data_roles"].values():
        seed_groups.extend(role["seeds"])
    seed_groups.extend(registration["negative_controls"].values())
    if len(seed_groups) != len(set(seed_groups)):
        raise ValueError("all registered data-role and control seeds must be disjoint")
    if registration["learning"]["directed_edge_budget"] < len(true_edges):
        raise ValueError("directed edge budget cannot be smaller than the registered truth")


def _joint_spectral_radius(registration: dict) -> float:
    bridge = _true_bridge(registration)
    observed = np.diag(registration["scm"]["self_coefficients"]) + bridge
    observed_radius = float(np.max(np.abs(np.linalg.eigvals(observed))))
    return max(observed_radius, abs(float(registration["scm"]["latent_ar"])))


def _environment_loadings(registration: dict, environment: str) -> np.ndarray:
    if environment == "train":
        key = "train_hidden_loadings"
    elif environment == "ood":
        key = "ood_hidden_loadings"
    else:
        raise ValueError("environment must be train or ood")
    return np.asarray(registration["scm"][key], dtype=float)


def _one_step(
    state: np.ndarray,
    hidden: float,
    process_noise: np.ndarray,
    registration: dict,
    environment: str,
    *,
    bridge_override: np.ndarray | None = None,
) -> np.ndarray:
    scm = registration["scm"]
    bridge = _true_bridge(registration) if bridge_override is None else bridge_override
    return (
        np.asarray(scm["self_coefficients"], dtype=float) * state
        + bridge @ np.tanh(state)
        + _environment_loadings(registration, environment) * hidden
        + float(scm["state_noise_std"]) * process_noise
    )


def simulate_episode(
    seed: int,
    registration: dict,
    *,
    environment: str,
    steps: int,
    bridge_override: np.ndarray | None = None,
) -> Episode:
    """Generate one stationary observational episode with separated RNG streams."""

    scm = registration["scm"]
    size = len(registration["charts"])
    state_seed, latent_seed, process_seed = np.random.SeedSequence(seed).spawn(3)
    state_rng = np.random.default_rng(state_seed)
    latent_rng = np.random.default_rng(latent_seed)
    process_rng = np.random.default_rng(process_seed)
    state = state_rng.normal(0.0, float(scm["initial_state_std"]), size=size)
    hidden = float(state_rng.normal(0.0, float(scm["initial_state_std"])))
    burn_in = int(scm["burn_in_steps"])
    states = np.empty((steps + 1, size), dtype=float)
    hidden_values = np.empty(steps + 1, dtype=float)
    for index in range(burn_in + steps):
        state = _one_step(
            state,
            hidden,
            process_rng.normal(size=size),
            registration,
            environment,
            bridge_override=bridge_override,
        )
        hidden = (
            float(scm["latent_ar"]) * hidden
            + float(scm["latent_noise_std"]) * float(latent_rng.normal())
        )
        if index >= burn_in - 1:
            output_index = index - burn_in + 1
            states[output_index] = state
            hidden_values[output_index] = hidden
    if not np.isfinite(states).all():
        raise FloatingPointError("non-finite state generated")
    return Episode(states=states, hidden=hidden_values)


def generate_probe(
    seed: int,
    registration: dict,
    *,
    environment: str,
    stationary_steps: int,
    pairs_per_source: int,
    bridge_override: np.ndarray | None = None,
) -> ProbeBatch:
    """Generate paired surgical interventions with shared process noise."""

    size = len(registration["charts"])
    episode = simulate_episode(
        seed,
        registration,
        environment=environment,
        steps=stationary_steps,
        bridge_override=bridge_override,
    )
    order_seed, process_seed, sensor_seed = np.random.SeedSequence(seed + 1_000_003).spawn(3)
    order_rng = np.random.default_rng(order_seed)
    process_rng = np.random.default_rng(process_seed)
    sensor_rng = np.random.default_rng(sensor_seed)
    amplitude = float(registration["intervention"]["amplitude"])
    sensor_std = float(registration["intervention"]["independent_sensor_noise_std"])
    records: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    available = np.arange(len(episode.states))
    for source in range(size):
        replace_indices = pairs_per_source > len(available)
        indices = order_rng.choice(available, size=pairs_per_source, replace=replace_indices)
        for index in indices:
            base = episode.states[index]
            hidden = float(episode.hidden[index])
            plus = base.copy()
            minus = base.copy()
            plus[source] = amplitude
            minus[source] = -amplitude
            shared_noise = process_rng.normal(size=size)
            y_plus = _one_step(
                plus,
                hidden,
                shared_noise,
                registration,
                environment,
                bridge_override=bridge_override,
            )
            y_minus = _one_step(
                minus,
                hidden,
                shared_noise,
                registration,
                environment,
                bridge_override=bridge_override,
            )
            y_plus = y_plus + sensor_rng.normal(0.0, sensor_std, size=size)
            y_minus = y_minus + sensor_rng.normal(0.0, sensor_std, size=size)
            records.append((source, plus, minus, y_plus, y_minus))
    order = order_rng.permutation(len(records))
    return ProbeBatch(
        source=np.asarray([records[index][0] for index in order], dtype=int),
        x_plus=np.asarray([records[index][1] for index in order]),
        x_minus=np.asarray([records[index][2] for index in order]),
        y_plus=np.asarray([records[index][3] for index in order]),
        y_minus=np.asarray([records[index][4] for index in order]),
    )


def combine_probes(probes: Sequence[ProbeBatch]) -> ProbeBatch:
    return ProbeBatch(
        source=np.concatenate([probe.source for probe in probes]),
        x_plus=np.concatenate([probe.x_plus for probe in probes]),
        x_minus=np.concatenate([probe.x_minus for probe in probes]),
        y_plus=np.concatenate([probe.y_plus for probe in probes]),
        y_minus=np.concatenate([probe.y_minus for probe in probes]),
    )


def permute_probe_signs(probe: ProbeBatch, seed: int) -> ProbeBatch:
    """Break intervention/effect alignment with balanced within-source arm swaps."""

    rng = np.random.default_rng(seed)
    x_plus = probe.x_plus.copy()
    x_minus = probe.x_minus.copy()
    y_plus = probe.y_plus.copy()
    y_minus = probe.y_minus.copy()
    for source in np.unique(probe.source):
        indices = np.flatnonzero(probe.source == source)
        indices = indices[rng.permutation(len(indices))]
        swap = indices[: len(indices) // 2]
        y_plus[swap], y_minus[swap] = y_minus[swap].copy(), y_plus[swap].copy()
        # Inputs keep their labels while half of the paired outcomes are reassigned.
    return ProbeBatch(probe.source.copy(), x_plus, x_minus, y_plus, y_minus)


def laplace_beltrami_proposal(registration: dict) -> dict:
    """Construct a finite-graph LB heat kernel and rank fold-contact pairs."""

    geometry = registration["geometry_proposal"]
    size = len(registration["charts"])
    adjacency = np.zeros((size, size), dtype=float)
    for left, right, weight in geometry["surface_edges"]:
        adjacency[int(left), int(right)] = float(weight)
        adjacency[int(right), int(left)] = float(weight)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    heat = (eigenvectors * np.exp(-float(geometry["heat_time"]) * eigenvalues)) @ eigenvectors.T
    pairs = [(left, right) for left in range(size) for right in range(left + 1, size)]
    off_diagonal = np.asarray([heat[left, right] for left, right in pairs])
    span = float(np.ptp(off_diagonal))
    normalized = (off_diagonal - float(np.min(off_diagonal))) / max(span, 1e-12)
    contact = np.asarray(geometry["white_contact_affinity"], dtype=float)
    scores = {
        pair: float(contact[pair] * (1.0 - normalized[index]))
        for index, pair in enumerate(pairs)
    }
    ranked = sorted(pairs, key=lambda pair: (-scores[pair], pair))
    proposed_pairs = ranked[: int(geometry["undirected_pair_budget"])]
    directed = tuple(
        edge
        for left, right in proposed_pairs
        for edge in ((left, right), (right, left))
    )
    return {
        "laplacian": laplacian,
        "eigenvalues": eigenvalues,
        "heat_kernel": heat,
        "pair_scores": scores,
        "pairs": tuple(proposed_pairs),
        "directed_edges": directed,
    }


def _stack_episodes(episodes: Sequence[Episode]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([episode.states[:-1] for episode in episodes]),
        np.concatenate([episode.states[1:] for episode in episodes]),
    )


def _ridge(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def _local_design(values: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(values)), values, values**3))


def fit_observational_model(
    name: str,
    episodes: Sequence[Episode],
    edges: Sequence[Edge],
    ridge: float,
) -> BridgeModel:
    states, outcomes = _stack_episodes(episodes)
    size = states.shape[1]
    bridge = np.zeros((size, size), dtype=float)
    local = np.zeros((size, 3), dtype=float)
    edge_set = tuple(sorted(set(edges)))
    for target in range(size):
        incoming = sorted(source for source, destination in edge_set if destination == target)
        design = _local_design(states[:, target])
        if incoming:
            design = np.column_stack((design, np.tanh(states[:, incoming])))
        coefficients = _ridge(design, outcomes[:, target], ridge)
        local[target] = coefficients[:3]
        for source, coefficient in zip(incoming, coefficients[3:]):
            bridge[target, source] = float(coefficient)
    return BridgeModel(name, local, bridge, edge_set)


def fit_fixed_bridge_model(
    name: str,
    episodes: Sequence[Episode],
    bridge: np.ndarray,
    declared_edges: Sequence[Edge],
    ridge: float,
) -> BridgeModel:
    states, outcomes = _stack_episodes(episodes)
    adjusted = outcomes - np.tanh(states) @ bridge.T
    local = np.vstack(
        [
            _ridge(_local_design(states[:, target]), adjusted[:, target], ridge)
            for target in range(states.shape[1])
        ]
    )
    return BridgeModel(name, local, bridge.copy(), tuple(sorted(set(declared_edges))))


def _target_mse(model: BridgeModel, episodes: Sequence[Episode], target: int) -> float:
    states, outcomes = _stack_episodes(episodes)
    error = outcomes[:, target] - model.predict(states)[:, target]
    return float(np.mean(error**2))


def observational_edge_diagnostics(
    train: Sequence[Episode],
    holdout: Sequence[Episode],
    ridge: float,
) -> dict[Edge, dict[str, float]]:
    size = train[0].states.shape[1]
    local = fit_observational_model("local", train, (), ridge)
    hold_states, hold_outcomes = _stack_episodes(holdout)
    diagnostics: dict[Edge, dict[str, float]] = {}
    for target in range(size):
        base_mse = _target_mse(local, holdout, target)
        for source in range(size):
            if source == target:
                continue
            edge = (source, target)
            augmented = fit_observational_model("one_edge", train, (edge,), ridge)
            augmented_mse = _target_mse(augmented, holdout, target)
            correlation = float(
                np.corrcoef(np.tanh(hold_states[:, source]), hold_outcomes[:, target])[0, 1]
            )
            diagnostics[edge] = {
                "raw_correlation": correlation,
                "gain_fraction": float((base_mse - augmented_mse) / max(base_mse, 1e-12)),
                "holdout_local_mse": base_mse,
                "holdout_augmented_mse": augmented_mse,
            }
    return diagnostics


def estimate_intervention_edges(probe: ProbeBatch, amplitude: float) -> dict[Edge, dict[str, float | bool]]:
    size = probe.x_plus.shape[1]
    scale = 2.0 * float(np.tanh(amplitude))
    result: dict[Edge, dict[str, float | bool]] = {}
    for source in range(size):
        rows = np.flatnonzero(probe.source == source)
        if len(rows) < 4:
            raise ValueError("each source requires at least four paired probes")
        for target in range(size):
            if source == target:
                continue
            effects = (probe.y_plus[rows, target] - probe.y_minus[rows, target]) / scale
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


def select_causal_edges(
    intervention: dict[Edge, dict[str, float | bool]],
    observational: dict[Edge, dict[str, float]],
    geometry: dict,
    learning: dict,
) -> tuple[Edge, ...]:
    candidates: list[tuple[float, Edge]] = []
    for edge in geometry["directed_edges"]:
        effect = intervention[edge]
        gain = observational[edge]["gain_fraction"]
        if (
            abs(float(effect["estimate"])) >= float(learning["causal_effect_abs_min"])
            and float(effect["z_score"]) >= float(learning["causal_z_min"])
            and bool(effect["split_half_sign_agreement"])
            and gain >= float(learning["observational_gain_min_fraction"])
        ):
            undirected = tuple(sorted(edge))
            score = (
                abs(float(effect["estimate"]))
                * max(gain, 0.0)
                * float(geometry["pair_scores"][undirected])
            )
            candidates.append((score, edge))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    budget = int(learning["directed_edge_budget"])
    return tuple(edge for _, edge in candidates[:budget])


def _top_edges(
    diagnostics: dict[Edge, dict[str, float]],
    budget: int,
    score_key: str,
    *,
    allowed: Sequence[Edge] | None = None,
    geometry_scores: dict[tuple[int, int], float] | None = None,
) -> tuple[Edge, ...]:
    edge_pool = list(diagnostics) if allowed is None else list(allowed)
    scored = []
    for edge in edge_pool:
        score = abs(diagnostics[edge][score_key]) if score_key == "raw_correlation" else diagnostics[edge][score_key]
        if geometry_scores is not None:
            score = max(score, 0.0) * geometry_scores[tuple(sorted(edge))]
        scored.append((float(score), edge))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(edge for _, edge in scored[:budget])


def _bridge_from_estimates(
    size: int,
    estimates: dict[Edge, dict[str, float | bool]],
    edges: Sequence[Edge],
) -> np.ndarray:
    bridge = np.zeros((size, size), dtype=float)
    for source, target in edges:
        bridge[target, source] = float(estimates[(source, target)]["estimate"])
    return bridge


def _rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


def _paired_ci95_lower(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return float("-inf")
    return float(np.mean(array) - 1.96 * np.std(array, ddof=1) / np.sqrt(len(array)))


def _reduction(baseline: float, candidate: float) -> float:
    return float((baseline - candidate) / max(baseline, 1e-12))


def _probe_effect_nrmse(model: BridgeModel, probe: ProbeBatch, truth_edges: Sequence[Edge]) -> float:
    predicted = model.predict(probe.x_plus) - model.predict(probe.x_minus)
    observed = probe.y_plus - probe.y_minus
    predicted_values = []
    observed_values = []
    for source, target in truth_edges:
        rows = probe.source == source
        predicted_values.extend(predicted[rows, target])
        observed_values.extend(observed[rows, target])
    return _rmse(np.asarray(observed_values), np.asarray(predicted_values)) / max(
        float(np.sqrt(np.mean(np.asarray(observed_values) ** 2))), 1e-12
    )


def evaluate_models(
    models: dict[str, BridgeModel],
    registration: dict,
    split: str,
    truth_edges: Sequence[Edge],
) -> dict:
    role = registration["data_roles"][split]
    downstream = np.asarray(sorted({target for _, target in truth_edges}), dtype=int)
    model_values = {
        name: {"global": [], "downstream": [], "intervention_nrmse": []}
        for name in models
    }
    episodes: list[Episode] = []
    for seed in role["seeds"]:
        episode = simulate_episode(
            int(seed),
            registration,
            environment=role["environment"],
            steps=int(role["steps_per_seed"]),
        )
        episodes.append(episode)
        states, outcomes = episode.states[:-1], episode.states[1:]
        probe = generate_probe(
            int(seed) + 400_009,
            registration,
            environment=role["environment"],
            stationary_steps=max(int(role["steps_per_seed"]), 128),
            pairs_per_source=int(role["intervention_pairs_per_source_per_seed"]),
        )
        for name, model in models.items():
            prediction = model.predict(states)
            model_values[name]["global"].append(_rmse(outcomes, prediction))
            model_values[name]["downstream"].append(
                _rmse(outcomes[:, downstream], prediction[:, downstream])
            )
            model_values[name]["intervention_nrmse"].append(
                _probe_effect_nrmse(model, probe, truth_edges)
            )
    summarized = {
        name: {
            "mean_global_rmse": float(np.mean(values["global"])),
            "mean_downstream_rmse": float(np.mean(values["downstream"])),
            "mean_intervention_nrmse": float(np.mean(values["intervention_nrmse"])),
            "seed_global_rmse": values["global"],
            "seed_downstream_rmse": values["downstream"],
        }
        for name, values in model_values.items()
    }
    observational_names = (
        "local_only",
        "dense_observational",
        "raw_correlation_top2",
        "predictive_gain_top2",
        "geometry_observational_top2",
    )
    best_by_seed = np.min(
        np.asarray([model_values[name]["global"] for name in observational_names]), axis=0
    )
    improvement = best_by_seed - np.asarray(model_values["causal_bridge"]["global"])
    return {
        "models": summarized,
        "paired_ci95_lower_vs_best_observational": _paired_ci95_lower(improvement),
        "episodes": episodes,
    }


def _graph_metrics(selected: Sequence[Edge], truth: Sequence[Edge]) -> dict:
    selected_set, truth_set = set(selected), set(truth)
    true_positive = len(selected_set & truth_set)
    precision = true_positive / len(selected_set) if selected_set else 0.0
    recall = true_positive / len(truth_set) if truth_set else 1.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "exact_recovery": selected_set == truth_set,
    }


def _lesion_metrics(
    causal: BridgeModel,
    episodes: Sequence[Episode],
    truth_edges: Sequence[Edge],
) -> dict:
    increases = []
    non_target_changes = []
    size = causal.bridge.shape[0]
    for source, target in truth_edges:
        lesioned_bridge = causal.bridge.copy()
        lesioned_bridge[target, source] = 0.0
        lesioned = replace(causal, name="lesioned", bridge=lesioned_bridge)
        for episode in episodes:
            states, outcomes = episode.states[:-1], episode.states[1:]
            intact_prediction = causal.predict(states)
            lesion_prediction = lesioned.predict(states)
            intact_mse = float(np.mean((outcomes[:, target] - intact_prediction[:, target]) ** 2))
            lesion_mse = float(np.mean((outcomes[:, target] - lesion_prediction[:, target]) ** 2))
            increases.append((lesion_mse - intact_mse) / max(intact_mse, 1e-12))
            other = np.asarray([index for index in range(size) if index != target], dtype=int)
            non_target_changes.append(
                float(np.max(np.abs(lesion_prediction[:, other] - intact_prediction[:, other])))
            )
    return {
        "minimum_direct_target_mse_increase_fraction": float(np.min(increases)),
        "maximum_non_target_prediction_change": float(np.max(non_target_changes)),
    }


def _serialize_edge_diagnostics(
    diagnostics: dict[Edge, dict], chart_names: Sequence[str]
) -> dict[str, dict]:
    result = {}
    for edge, values in sorted(diagnostics.items()):
        key = f"{chart_names[edge[0]]}->{chart_names[edge[1]]}"
        result[key] = {
            name: (bool(value) if isinstance(value, (bool, np.bool_)) else float(value))
            for name, value in values.items()
        }
    return result


def _all_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _validation_artifact_path(config_path: Path, experiment: str) -> Path:
    suffix = experiment.rsplit("_", 1)[-1]
    root = config_path.resolve().parents[2]
    return root / "artifacts" / "agi" / f"sparse_causal_bridge_validation_{suffix}.json"


def _assert_test_unlocked(config_path: Path, registration: dict, config_sha: str) -> None:
    path = _validation_artifact_path(config_path, registration["experiment"])
    if not path.exists():
        raise PermissionError("locked test requires a saved passing validation artifact")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("passed") or report.get("registration", {}).get("sha256") != config_sha:
        raise PermissionError("validation artifact did not pass under the identical registration")


def run_sparse_causal_bridge_gate(
    config_path: Path,
    *,
    split: str = "validation",
    enforce_test_lock: bool = True,
) -> dict:
    started = time.perf_counter()
    registration, raw = _load_registration(config_path)
    _validate_registration(registration)
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    config_sha = hashlib.sha256(raw).hexdigest()
    if split == "test" and enforce_test_lock:
        _assert_test_unlocked(config_path, registration, config_sha)

    charts = registration["charts"]
    size = len(charts)
    ridge = float(registration["learning"]["ridge"])
    data = registration["data_roles"]
    train = [
        simulate_episode(
            seed,
            registration,
            environment=role["environment"],
            steps=int(role["steps_per_seed"]),
        )
        for seed in data["observational_train"]["seeds"]
        for role in (data["observational_train"],)
    ]
    holdout = [
        simulate_episode(
            seed,
            registration,
            environment=role["environment"],
            steps=int(role["steps_per_seed"]),
        )
        for seed in data["observational_selector_holdout"]["seeds"]
        for role in (data["observational_selector_holdout"],)
    ]
    geometry = laplace_beltrami_proposal(registration)
    observational = observational_edge_diagnostics(train, holdout, ridge)
    probe_role = data["topology_intervention_probe"]
    probes = [
        generate_probe(
            seed,
            registration,
            environment=probe_role["environment"],
            stationary_steps=int(probe_role["stationary_steps_per_seed"]),
            pairs_per_source=int(probe_role["pairs_per_source_per_seed"]),
        )
        for seed in probe_role["seeds"]
    ]
    pooled_probe = combine_probes(probes)
    intervention = estimate_intervention_edges(
        pooled_probe, float(registration["intervention"]["amplitude"])
    )
    budget = int(registration["learning"]["directed_edge_budget"])
    raw_edges = _top_edges(observational, budget, "raw_correlation")
    gain_edges = _top_edges(observational, budget, "gain_fraction")
    geometry_edges = _top_edges(
        observational,
        budget,
        "gain_fraction",
        allowed=geometry["directed_edges"],
        geometry_scores=geometry["pair_scores"],
    )
    causal_edges = select_causal_edges(
        intervention, observational, geometry, registration["learning"]
    )
    all_edges = tuple(
        (source, target)
        for source in range(size)
        for target in range(size)
        if source != target
    )
    dense_probe_bridge = _bridge_from_estimates(size, intervention, all_edges)
    causal_bridge = _bridge_from_estimates(size, intervention, causal_edges)
    true_bridge = _true_bridge(registration)
    truth_edges = tuple(
        (int(item["source"]), int(item["target"]))
        for item in registration["scm"]["true_directed_bridges"]
    )
    models = {
        "local_only": fit_observational_model("local_only", train, (), ridge),
        "dense_observational": fit_observational_model(
            "dense_observational", train, all_edges, ridge
        ),
        "raw_correlation_top2": fit_observational_model(
            "raw_correlation_top2", train, raw_edges, ridge
        ),
        "predictive_gain_top2": fit_observational_model(
            "predictive_gain_top2", train, gain_edges, ridge
        ),
        "geometry_observational_top2": fit_observational_model(
            "geometry_observational_top2", train, geometry_edges, ridge
        ),
        "dense_probe": fit_fixed_bridge_model(
            "dense_probe", train, dense_probe_bridge, all_edges, ridge
        ),
        "causal_bridge": fit_fixed_bridge_model(
            "causal_bridge", train, causal_bridge, causal_edges, ridge
        ),
        "oracle_diagnostic": fit_fixed_bridge_model(
            "oracle_diagnostic", train, true_bridge, truth_edges, ridge
        ),
    }
    evaluation = evaluate_models(models, registration, split, truth_edges)
    lesion = _lesion_metrics(models["causal_bridge"], evaluation.pop("episodes"), truth_edges)

    per_probe_exact = []
    for probe in probes:
        estimates = estimate_intervention_edges(
            probe, float(registration["intervention"]["amplitude"])
        )
        selected = select_causal_edges(
            estimates, observational, geometry, registration["learning"]
        )
        per_probe_exact.append(set(selected) == set(truth_edges))

    zero_bridge = np.zeros_like(true_bridge)
    control_config = registration["negative_controls"]
    null_probe = generate_probe(
        int(control_config["no_bridge_seed"]),
        registration,
        environment="train",
        stationary_steps=int(probe_role["stationary_steps_per_seed"]),
        pairs_per_source=int(probe_role["pairs_per_source_per_seed"]),
        bridge_override=zero_bridge,
    )
    null_estimates = estimate_intervention_edges(
        null_probe, float(registration["intervention"]["amplitude"])
    )
    null_selected = select_causal_edges(
        null_estimates, observational, geometry, registration["learning"]
    )
    permuted = permute_probe_signs(pooled_probe, int(control_config["permuted_intervention_seed"]))
    permuted_estimates = estimate_intervention_edges(
        permuted, float(registration["intervention"]["amplitude"])
    )
    permuted_selected = select_causal_edges(
        permuted_estimates, observational, geometry, registration["learning"]
    )

    graph = _graph_metrics(causal_edges, truth_edges)
    common_pair = tuple(registration["scm"]["registered_common_cause_pair"])
    common_edges = {common_pair, common_pair[::-1]}
    reverse_edges = {(target, source) for source, target in truth_edges}
    selected_set = set(causal_edges)
    coefficient_errors = [
        abs(causal_bridge[target, source] - true_bridge[target, source])
        for source, target in truth_edges
    ]
    sign_accuracy = float(
        np.mean(
            [
                np.sign(causal_bridge[target, source]) == np.sign(true_bridge[target, source])
                for source, target in truth_edges
            ]
        )
    )
    proposed_pair_set = set(geometry["pairs"])
    truth_pair_set = {tuple(sorted(edge)) for edge in truth_edges}
    proposal_coverage = len(proposed_pair_set & truth_pair_set) / len(truth_pair_set)
    train_states, _ = _stack_episodes(train)
    common_correlation = abs(float(np.corrcoef(train_states[:, common_pair[0]], train_states[:, common_pair[1]])[0, 1]))

    metrics = evaluation["models"]
    causal_global = metrics["causal_bridge"]["mean_global_rmse"]
    causal_downstream = metrics["causal_bridge"]["mean_downstream_rmse"]
    comparisons = {
        "global_reduction_vs_local": _reduction(
            metrics["local_only"]["mean_global_rmse"], causal_global
        ),
        "global_reduction_vs_dense_observational": _reduction(
            metrics["dense_observational"]["mean_global_rmse"], causal_global
        ),
        "global_reduction_vs_raw_correlation": _reduction(
            metrics["raw_correlation_top2"]["mean_global_rmse"], causal_global
        ),
        "global_reduction_vs_predictive_gain": _reduction(
            metrics["predictive_gain_top2"]["mean_global_rmse"], causal_global
        ),
        "downstream_reduction_vs_local": _reduction(
            metrics["local_only"]["mean_downstream_rmse"], causal_downstream
        ),
        "downstream_reduction_vs_dense_observational": _reduction(
            metrics["dense_observational"]["mean_downstream_rmse"], causal_downstream
        ),
        "causal_rmse_ratio_vs_dense_probe": causal_global
        / max(metrics["dense_probe"]["mean_global_rmse"], 1e-12),
        "causal_rmse_ratio_vs_predictive_gain": causal_global
        / max(metrics["predictive_gain_top2"]["mean_global_rmse"], 1e-12),
        "causal_downstream_rmse_ratio_vs_dense_observational": causal_downstream
        / max(metrics["dense_observational"]["mean_downstream_rmse"], 1e-12),
        "paired_ci95_lower_vs_best_observational": evaluation[
            "paired_ci95_lower_vs_best_observational"
        ],
    }
    gate = registration["gate"]
    raw_has_common = bool(set(raw_edges) & common_edges)
    checks = {
        "spectral_stability": _joint_spectral_radius(registration)
        <= gate["max_joint_spectral_radius"],
        "common_cause_trap_present": common_correlation
        >= gate["train_abs_common_cause_correlation_min"],
        "raw_correlation_selects_common_cause": (
            raw_has_common if gate["raw_correlation_must_select_common_cause"] else True
        ),
        "geometry_truth_coverage": proposal_coverage
        >= gate["geometry_truth_pair_coverage_min"],
        "directed_precision": graph["precision"] >= gate["directed_precision_min"],
        "directed_recall": graph["recall"] >= gate["directed_recall_min"],
        "exact_recovery_stability": float(np.mean(per_probe_exact))
        >= gate["exact_graph_recovery_probe_seed_fraction_min"],
        "common_cause_rejected": len(selected_set & common_edges)
        <= gate["common_cause_false_edges_max"],
        "reverse_edges_rejected": len(selected_set & reverse_edges)
        <= gate["reverse_true_edges_max"],
        "edge_budget": len(causal_edges) <= gate["selected_edges_max"],
        "true_effect_sign": sign_accuracy >= gate["true_effect_sign_accuracy_min"],
        "bridge_coefficient_mae": float(np.mean(coefficient_errors))
        <= gate["bridge_coefficient_mae_max"],
        "intervention_response": metrics["causal_bridge"]["mean_intervention_nrmse"]
        <= gate["intervention_response_nrmse_max"],
        "global_vs_local": comparisons["global_reduction_vs_local"]
        >= gate["ood_global_rmse_reduction_vs_local_min"],
        "global_vs_dense_observational": comparisons[
            "global_reduction_vs_dense_observational"
        ]
        >= gate["ood_global_rmse_reduction_vs_dense_observational_min"],
        "global_vs_raw_correlation": comparisons["global_reduction_vs_raw_correlation"]
        >= gate["ood_global_rmse_reduction_vs_raw_correlation_min"],
        "global_vs_predictive_gain": comparisons["global_reduction_vs_predictive_gain"]
        >= gate["ood_global_rmse_reduction_vs_predictive_gain_min"],
        "downstream_vs_local": comparisons["downstream_reduction_vs_local"]
        >= gate["ood_downstream_rmse_reduction_vs_local_min"],
        "downstream_vs_dense_observational": comparisons[
            "downstream_reduction_vs_dense_observational"
        ]
        >= gate["ood_downstream_rmse_reduction_vs_dense_observational_min"],
        "paired_ci_vs_best_observational": comparisons[
            "paired_ci95_lower_vs_best_observational"
        ]
        > gate["paired_ci95_lower_vs_best_observational_min"],
        "dense_probe_noninferiority": comparisons["causal_rmse_ratio_vs_dense_probe"]
        <= gate["causal_rmse_ratio_vs_dense_probe_max"],
        "lesion_direct_target": lesion["minimum_direct_target_mse_increase_fraction"]
        >= gate["lesion_direct_target_mse_increase_fraction_min"],
        "lesion_locality": lesion["maximum_non_target_prediction_change"]
        <= gate["lesion_non_descendant_prediction_change_max"],
        "no_bridge_negative_control": len(null_selected)
        <= control_config["max_selected_edges_each"],
        "permuted_intervention_negative_control": len(permuted_selected)
        <= control_config["max_selected_edges_each"],
    }
    if "causal_rmse_ratio_vs_predictive_gain_max" in gate:
        checks["predictive_gain_noninferiority"] = comparisons[
            "causal_rmse_ratio_vs_predictive_gain"
        ] <= gate["causal_rmse_ratio_vs_predictive_gain_max"]
    if "causal_downstream_rmse_ratio_vs_dense_observational_max" in gate:
        checks["dense_observational_downstream_noninferiority"] = comparisons[
            "causal_downstream_rmse_ratio_vs_dense_observational"
        ] <= gate["causal_downstream_rmse_ratio_vs_dense_observational_max"]
    finite_payload = {"metrics": metrics, "comparisons": comparisons, "lesion": lesion}
    checks["finite_metrics"] = _all_finite(finite_payload)
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
        "selection": {
            "matrix_orientation": registration["matrix_orientation"],
            "laplace_beltrami_eigenvalues": geometry["eigenvalues"].tolist(),
            "geometry_pair_scores": {
                f"{charts[left]}--{charts[right]}": score
                for (left, right), score in sorted(geometry["pair_scores"].items())
            },
            "geometry_proposed_pairs": [
                f"{charts[left]}--{charts[right]}" for left, right in geometry["pairs"]
            ],
            "geometry_truth_pair_coverage": proposal_coverage,
            "raw_correlation_edges": _edge_list(raw_edges, charts),
            "predictive_gain_edges": _edge_list(gain_edges, charts),
            "geometry_observational_edges": _edge_list(geometry_edges, charts),
            "causal_edges": _edge_list(causal_edges, charts),
            "truth_edges_evaluation_only": _edge_list(truth_edges, charts),
            "causal_edge_count": len(causal_edges),
            "exact_recovery_probe_seed_fraction": float(np.mean(per_probe_exact)),
            "intervention_diagnostics": _serialize_edge_diagnostics(intervention, charts),
            "observational_diagnostics": _serialize_edge_diagnostics(observational, charts),
        },
        "validity": {
            "joint_spectral_radius": _joint_spectral_radius(registration),
            "train_abs_common_cause_correlation": common_correlation,
            "raw_correlation_selected_common_cause": raw_has_common,
        },
        "graph_metrics": {
            **graph,
            "common_cause_false_edges": len(selected_set & common_edges),
            "reverse_true_edges": len(selected_set & reverse_edges),
            "true_effect_sign_accuracy": sign_accuracy,
            "bridge_coefficient_mae": float(np.mean(coefficient_errors)),
        },
        "models": metrics,
        "comparisons": comparisons,
        "lesion": lesion,
        "negative_controls": {
            "no_bridge_selected_edges": _edge_list(null_selected, charts),
            "permuted_intervention_selected_edges": _edge_list(permuted_selected, charts),
        },
        "checks": checks,
        "resource_checks": resource_checks,
        "resource_usage": {
            "wall_seconds": elapsed,
            "external_download_bytes": 0,
            "trajectory_files_written": 0,
            "topology_probe_pairs": int(len(pooled_probe.source)),
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
    registration, _ = _load_registration(args.config)
    report = run_sparse_causal_bridge_gate(args.config, split=args.split)
    output = args.output or _default_output(args.config, args.split, registration["experiment"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"artifact: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
