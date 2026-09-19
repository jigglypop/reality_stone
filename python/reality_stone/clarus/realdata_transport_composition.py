"""Held-out transport-composition test for trial-resolved calcium imaging.

The test is deliberately observational.  It asks whether two fitted local
state transitions compose into a useful two-step predictor in a common,
train-only latent chart.  It does not identify synapses, anatomical edges, or
learning-induced shortcuts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_E17_ROOT = Path(
    "_workspace/ce/_archive/neural-riemannian-metric-validation-20260818/"
    "artifacts/realdata/NRM-E17-extracted/Figure2/Data"
)


@dataclass(frozen=True)
class TransportConfig:
    """Frozen analysis choices for the equal-lag delay-period test."""

    phase_centers_seconds: tuple[float, float, float] = (-1.5, -0.9, -0.3)
    phase_window_seconds: float = 0.2
    trial_start_seconds: float = -3.0
    trial_stop_seconds: float = 3.0
    latent_rank: int = 6
    ridge: float = 1.0
    folds: int = 5
    near_direct_g_tolerance: float = 0.10

    def validate(self) -> None:
        centers = np.asarray(self.phase_centers_seconds, dtype=float)
        if centers.shape != (3,) or not np.isfinite(centers).all():
            raise ValueError("phase_centers_seconds must contain three finite values")
        gaps = np.diff(centers)
        if np.any(gaps <= 0.0) or not np.allclose(gaps, gaps[0], atol=1e-12):
            raise ValueError("phase centers must be increasing and equally spaced")
        if not np.isfinite(self.phase_window_seconds) or self.phase_window_seconds <= 0:
            raise ValueError("phase_window_seconds must be positive and finite")
        if self.trial_stop_seconds <= self.trial_start_seconds:
            raise ValueError("trial time interval must be increasing")
        if self.latent_rank < 1 or self.folds < 2:
            raise ValueError("latent_rank and folds must be positive")
        if not np.isfinite(self.ridge) or self.ridge < 0.0:
            raise ValueError("ridge must be finite and nonnegative")
        if self.near_direct_g_tolerance < 0.0:
            raise ValueError("near_direct_g_tolerance must be nonnegative")


@dataclass(frozen=True)
class E17Block:
    session_id: str
    animal: str
    condition: str
    source_path: str
    source_sha256: str
    trials: np.ndarray


@dataclass(frozen=True)
class AffineMap:
    coefficient: np.ndarray
    intercept: np.ndarray

    def predict(self, values: np.ndarray) -> np.ndarray:
        return values @ self.coefficient + self.intercept


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trial_stack(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != object and array.ndim == 3:
        result = np.asarray(array, dtype=float)
    else:
        items = [np.asarray(item, dtype=float) for item in np.atleast_1d(value)]
        if not items or any(item.ndim != 2 for item in items):
            raise ValueError("trial field must contain time-by-ROI matrices")
        if any(item.shape != items[0].shape for item in items):
            raise ValueError("trial shapes change inside one condition")
        result = np.stack(items)
    if result.ndim != 3 or result.shape[1] != 180:
        raise ValueError("expected trial-by-180-frame-by-ROI activity")
    return result


def load_e17_blocks(
    root: str | Path = DEFAULT_E17_ROOT,
    *,
    signal_field: str = "dff",
) -> tuple[E17Block, ...]:
    """Load saline and DCZ trial blocks without pooling session identities."""

    try:
        from scipy.io import loadmat
    except ImportError as error:  # pragma: no cover - environment guard
        raise RuntimeError("SciPy is required for E17 MATLAB files") from error

    root_path = Path(root)
    paths = sorted(root_path.glob("DCO*_dff.mat"))
    if len(paths) != 11:
        raise ValueError(f"expected 11 E17 Figure 2 sessions, found {len(paths)}")

    blocks: list[E17Block] = []
    for path in paths:
        payload = loadmat(
            path,
            variable_names=("cont_data",),
            simplify_cells=True,
        )
        cont_data = payload.get("cont_data")
        if not isinstance(cont_data, dict):
            raise ValueError(f"{path} lacks the cont_data structure")
        animal = path.stem.split("_", 1)[0]
        source_hash = sha256_file(path)
        for condition_key, condition_name in (("Sal", "saline"), ("DCZ", "dcz")):
            condition = cont_data.get(condition_key)
            if not isinstance(condition, dict) or signal_field not in condition:
                raise KeyError(f"{path} lacks cont_data.{condition_key}.{signal_field}")
            trials = _trial_stack(condition[signal_field])
            blocks.append(
                E17Block(
                    session_id=path.stem,
                    animal=animal,
                    condition=condition_name,
                    source_path=path.as_posix(),
                    source_sha256=source_hash,
                    trials=trials,
                )
            )
    return tuple(blocks)


def phase_states(
    trials: np.ndarray,
    config: TransportConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average three equal-lag windows after cue and before the Go signal."""

    config.validate()
    trial_array = np.asarray(trials, dtype=float)
    if trial_array.ndim != 3 or trial_array.shape[1] != 180:
        raise ValueError("trials must have shape (trial, 180, ROI)")
    time = np.linspace(
        config.trial_start_seconds,
        config.trial_stop_seconds,
        trial_array.shape[1],
    )
    half = 0.5 * config.phase_window_seconds
    states: list[np.ndarray] = []
    for center in config.phase_centers_seconds:
        mask = (time >= center - half) & (time <= center + half)
        if int(np.sum(mask)) < 2:
            raise ValueError("a phase window contains fewer than two frames")
        states.append(np.nanmean(trial_array[:, mask, :], axis=1))
    finite = np.logical_and.reduce([np.isfinite(state).all(axis=1) for state in states])
    if int(np.sum(finite)) < max(15, config.folds * 2):
        raise ValueError("too few finite whole trials survive phase extraction")
    return tuple(state[finite] for state in states)  # type: ignore[return-value]


def _fit_affine(x: np.ndarray, y: np.ndarray, ridge: float) -> AffineMap:
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    x_centered = x - x_mean
    y_centered = y - y_mean
    gram = x_centered.T @ x_centered
    system = gram + ridge * np.eye(gram.shape[0])
    rhs = x_centered.T @ y_centered
    try:
        coefficient = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        coefficient = np.linalg.pinv(system, rcond=1e-12) @ rhs
    intercept = y_mean - x_mean @ coefficient
    return AffineMap(coefficient=coefficient, intercept=intercept)


def _latent_chart(
    states: Sequence[np.ndarray],
    train_indices: np.ndarray,
    rank: int,
) -> tuple[tuple[np.ndarray, ...], int]:
    train = np.vstack([state[train_indices] for state in states])
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    keep = np.isfinite(mean) & np.isfinite(scale) & (scale > 1e-10)
    if not np.any(keep):
        raise ValueError("no varying ROI survives the train-only chart")
    train_z = (train[:, keep] - mean[keep]) / scale[keep]
    _, _, right = np.linalg.svd(train_z, full_matrices=False)
    count = min(rank, right.shape[0], int(np.sum(keep)))
    if count < 1:
        raise ValueError("latent chart is empty")
    components = right[:count].T
    transformed = tuple(
        ((state[:, keep] - mean[keep]) / scale[keep]) @ components
        for state in states
    )
    return transformed, count


def _sse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sum(np.square(target - prediction)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, np.finfo(float).eps))


def _folds(count: int, fold_count: int) -> tuple[np.ndarray, ...]:
    if count < fold_count * 2:
        raise ValueError("not enough trials for the requested whole-trial folds")
    return tuple(np.asarray(part, dtype=int) for part in np.array_split(np.arange(count), fold_count))


def evaluate_transport_block(
    block: E17Block,
    *,
    config: TransportConfig = TransportConfig(),
) -> dict[str, Any]:
    """Evaluate phase-specific and stationary composition on held-out trials."""

    x0, x1, x2 = phase_states(block.trials, config)
    fold_rows: list[dict[str, float | int]] = []
    totals = {
        "mean": 0.0,
        "persistence": 0.0,
        "direct": 0.0,
        "composed": 0.0,
        "stationary_composed": 0.0,
        "deranged_composed": 0.0,
        "permuted_interface": 0.0,
        "reverse_mean": 0.0,
        "reverse_persistence": 0.0,
        "reverse_direct": 0.0,
        "reverse_composed": 0.0,
        "operator_difference": 0.0,
        "operator_direct": 0.0,
        "prediction_difference": 0.0,
        "prediction_direct": 0.0,
    }
    all_indices = np.arange(x0.shape[0])
    for fold_index, test_indices in enumerate(_folds(x0.shape[0], config.folds)):
        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
        (z0, z1, z2), rank = _latent_chart(
            (x0, x1, x2), train_indices, config.latent_rank
        )
        z0_train, z1_train, z2_train = (
            state[train_indices] for state in (z0, z1, z2)
        )
        z0_test, z1_test, z2_test = (state[test_indices] for state in (z0, z1, z2))

        transition_01 = _fit_affine(z0_train, z1_train, config.ridge)
        transition_12 = _fit_affine(z1_train, z2_train, config.ridge)
        transition_02 = _fit_affine(z0_train, z2_train, config.ridge)
        first = transition_01.predict(z0_test)
        composed = transition_12.predict(first)
        direct = transition_02.predict(z0_test)

        stationary = _fit_affine(
            np.vstack((z0_train, z1_train)),
            np.vstack((z1_train, z2_train)),
            config.ridge,
        )
        stationary_composed = stationary.predict(stationary.predict(z0_test))

        shift = max(1, z0_train.shape[0] // 3 + fold_index)
        deranged_01 = _fit_affine(z0_train, np.roll(z1_train, shift, axis=0), config.ridge)
        deranged_12 = _fit_affine(z1_train, np.roll(z2_train, -shift, axis=0), config.ridge)
        deranged = deranged_12.predict(deranged_01.predict(z0_test))

        coordinate_permutation = np.arange(rank)[::-1]
        permuted_interface = transition_12.predict(first[:, coordinate_permutation])

        reverse_21 = _fit_affine(z2_train, z1_train, config.ridge)
        reverse_10 = _fit_affine(z1_train, z0_train, config.ridge)
        reverse_20 = _fit_affine(z2_train, z0_train, config.ridge)
        reverse_composed = reverse_10.predict(reverse_21.predict(z2_test))
        reverse_direct = reverse_20.predict(z2_test)

        mean_prediction = np.broadcast_to(np.mean(z2_train, axis=0), z2_test.shape)
        reverse_mean = np.broadcast_to(np.mean(z0_train, axis=0), z0_test.shape)
        fold_sse = {
            "mean": _sse(z2_test, mean_prediction),
            "persistence": _sse(z2_test, z0_test),
            "direct": _sse(z2_test, direct),
            "composed": _sse(z2_test, composed),
            "stationary_composed": _sse(z2_test, stationary_composed),
            "deranged_composed": _sse(z2_test, deranged),
            "permuted_interface": _sse(z2_test, permuted_interface),
            "reverse_mean": _sse(z0_test, reverse_mean),
            "reverse_persistence": _sse(z0_test, z2_test),
            "reverse_direct": _sse(z0_test, reverse_direct),
            "reverse_composed": _sse(z0_test, reverse_composed),
        }
        for key, value in fold_sse.items():
            totals[key] += value

        composed_coefficient = transition_01.coefficient @ transition_12.coefficient
        centered_train = z0_train - np.mean(z0_train, axis=0)
        totals["operator_difference"] += _sse(
            np.zeros_like(centered_train),
            centered_train @ (transition_02.coefficient - composed_coefficient),
        )
        totals["operator_direct"] += _sse(
            np.zeros_like(centered_train), centered_train @ transition_02.coefficient
        )
        totals["prediction_difference"] += _sse(direct, composed)
        totals["prediction_direct"] += _sse(direct, mean_prediction)
        fold_rows.append(
            {
                "fold": fold_index,
                "n_train": int(train_indices.size),
                "n_test": int(test_indices.size),
                "latent_rank": rank,
                **{f"sse_{key}": value for key, value in fold_sse.items()},
            }
        )

    persistence = totals["persistence"]
    composed = totals["composed"]
    direct = totals["direct"]
    deranged = totals["deranged_composed"]
    score = {
        "session_id": block.session_id,
        "animal": block.animal,
        "condition": block.condition,
        "signal_field": "unknown",
        "source_path": block.source_path,
        "source_sha256": block.source_sha256,
        "n_trials": int(x0.shape[0]),
        "n_rois": int(block.trials.shape[2]),
        "phase_frame_counts": [
            int(
                np.sum(
                    (
                        np.linspace(
                            config.trial_start_seconds,
                            config.trial_stop_seconds,
                            block.trials.shape[1],
                        )
                        >= center - 0.5 * config.phase_window_seconds
                    )
                    & (
                        np.linspace(
                            config.trial_start_seconds,
                            config.trial_stop_seconds,
                            block.trials.shape[1],
                        )
                        <= center + 0.5 * config.phase_window_seconds
                    )
                )
            )
            for center in config.phase_centers_seconds
        ],
        **{f"sse_{key}": float(value) for key, value in totals.items() if not key.startswith("operator_") and not key.startswith("prediction_")},
        "g_composition_excess_over_direct": _safe_ratio(composed - direct, persistence),
        "composition_skill_vs_persistence": 1.0 - _safe_ratio(composed, persistence),
        "composition_skill_vs_mean": 1.0 - _safe_ratio(composed, totals["mean"]),
        "direct_skill_vs_persistence": 1.0 - _safe_ratio(direct, persistence),
        "composition_advantage_over_deranged": 1.0 - _safe_ratio(composed, deranged),
        "composition_advantage_over_permuted_interface": 1.0
        - _safe_ratio(composed, totals["permuted_interface"]),
        "stationary_skill_vs_persistence": 1.0
        - _safe_ratio(totals["stationary_composed"], persistence),
        "reverse_g_composition_excess_over_direct": _safe_ratio(
            totals["reverse_composed"] - totals["reverse_direct"],
            totals["reverse_persistence"],
        ),
        "covariance_weighted_operator_discrepancy": _safe_ratio(
            totals["operator_difference"], totals["operator_direct"]
        ),
        "heldout_composed_direct_prediction_discrepancy": _safe_ratio(
            totals["prediction_difference"], totals["prediction_direct"]
        ),
        "near_direct": bool(
            _safe_ratio(composed - direct, persistence)
            <= config.near_direct_g_tolerance
        ),
        "beats_persistence": bool(composed < persistence),
        "beats_mean": bool(composed < totals["mean"]),
        "beats_deranged": bool(composed < deranged),
        "folds": fold_rows,
    }
    score["core_consistent"] = bool(
        score["near_direct"]
        and score["beats_persistence"]
        and score["beats_mean"]
        and score["beats_deranged"]
    )
    return score


def _aggregate_scores(scores: Iterable[dict[str, Any]], label: str) -> dict[str, Any]:
    rows = tuple(scores)
    if not rows:
        raise ValueError(f"cannot aggregate empty {label} rows")
    summed = {
        key: float(sum(float(row[key]) for row in rows))
        for key in (
            "sse_mean",
            "sse_persistence",
            "sse_direct",
            "sse_composed",
            "sse_stationary_composed",
            "sse_deranged_composed",
            "sse_permuted_interface",
        )
    }
    persistence = summed["sse_persistence"]
    composed = summed["sse_composed"]
    direct = summed["sse_direct"]
    result = {
        "label": label,
        "n_rows": len(rows),
        "n_trials": int(sum(int(row["n_trials"]) for row in rows)),
        **summed,
        "g_composition_excess_over_direct": _safe_ratio(composed - direct, persistence),
        "composition_skill_vs_persistence": 1.0 - _safe_ratio(composed, persistence),
        "composition_skill_vs_mean": 1.0 - _safe_ratio(composed, summed["sse_mean"]),
        "direct_skill_vs_persistence": 1.0 - _safe_ratio(direct, persistence),
        "composition_advantage_over_deranged": 1.0
        - _safe_ratio(composed, summed["sse_deranged_composed"]),
        "composition_advantage_over_permuted_interface": 1.0
        - _safe_ratio(composed, summed["sse_permuted_interface"]),
    }
    return result


def evaluate_e17_panel(
    root: str | Path = DEFAULT_E17_ROOT,
    *,
    config: TransportConfig = TransportConfig(),
    signal_fields: Sequence[str] = ("dff", "branch"),
) -> dict[str, Any]:
    """Run the primary dF/F test and a predeclared branch-only sensitivity."""

    config.validate()
    panels: dict[str, Any] = {}
    for signal_field in signal_fields:
        block_scores: list[dict[str, Any]] = []
        for block in load_e17_blocks(root, signal_field=signal_field):
            score = evaluate_transport_block(block, config=config)
            score["signal_field"] = signal_field
            block_scores.append(score)

        session_rows: list[dict[str, Any]] = []
        for session_id in sorted({str(row["session_id"]) for row in block_scores}):
            subset = [row for row in block_scores if row["session_id"] == session_id]
            session_rows.append(_aggregate_scores(subset, session_id))
        animal_rows: list[dict[str, Any]] = []
        for animal in sorted({str(row["animal"]) for row in block_scores}):
            subset = [row for row in block_scores if row["animal"] == animal]
            animal_rows.append(_aggregate_scores(subset, animal))
        condition_rows: list[dict[str, Any]] = []
        for condition in sorted({str(row["condition"]) for row in block_scores}):
            subset = [row for row in block_scores if row["condition"] == condition]
            condition_rows.append(_aggregate_scores(subset, condition))
        overall = _aggregate_scores(block_scores, f"{signal_field}:all")
        for row in session_rows + animal_rows + condition_rows + [overall]:
            row["near_direct"] = bool(
                row["g_composition_excess_over_direct"]
                <= config.near_direct_g_tolerance
            )
            row["beats_persistence"] = bool(row["composition_skill_vs_persistence"] > 0.0)
            row["beats_mean"] = bool(row["composition_skill_vs_mean"] > 0.0)
            row["beats_deranged"] = bool(
                row["composition_advantage_over_deranged"] > 0.0
            )
            row["core_consistent"] = bool(
                row["near_direct"]
                and row["beats_persistence"]
                and row["beats_mean"]
                and row["beats_deranged"]
            )
        panels[signal_field] = {
            "block_scores": block_scores,
            "session_scores": session_rows,
            "animal_scores": animal_rows,
            "condition_scores": condition_rows,
            "overall": overall,
        }

    primary = panels[signal_fields[0]]
    animal_passes = int(sum(bool(row["core_consistent"]) for row in primary["animal_scores"]))
    overall_pass = bool(primary["overall"]["core_consistent"])
    status = (
        "OBSERVATIONAL_TRANSPORT_COMPOSITION_CONSISTENT"
        if overall_pass and animal_passes == len(primary["animal_scores"])
        else "OBSERVATIONAL_TRANSPORT_COMPOSITION_STOP"
    )
    return {
        "status": status,
        "claim_scope": (
            "session-local held-out predictive composition in an equal-lag "
            "delay-period calcium-state chart; no synaptic, anatomical, "
            "consolidation, or causal claim"
        ),
        "dataset": "Maristany de las Casas et al. Science 2026 Figure 2",
        "doi": "10.1126/science.adx4358",
        "archive_doi": "10.12751/g-node.etlk5k",
        "config": asdict(config),
        "primary_signal_field": signal_fields[0],
        "animal_passes": animal_passes,
        "animal_count": len(primary["animal_scores"]),
        "panels": panels,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("refusing to serialize a nonfinite numeric value")
        return numeric
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_E17_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_e17_panel(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "animal_passes": result["animal_passes"],
        "animal_count": result["animal_count"],
        "output": args.output.as_posix(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
