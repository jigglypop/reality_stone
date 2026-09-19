"""Evidence manifest checks for empirical Clarus gates.

The evolution ladder uses external datasets to promote theoretical terms into
reproducible gates. This module keeps the readiness judgment mechanical: a
dataset moves forward only when the required manifest fields are present.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


READINESS_ORDER = (
    "metadata-only",
    "download-ready",
    "field-ready",
    "gate-ready",
    "reproducible",
)

METADATA_FIELDS = (
    "source_id",
    "species",
    "paper_or_dataset_url",
    "license_or_access",
)

FIELD_FIELDS = (
    "neural_fields",
    "behavior_fields",
    "stimulus_fields",
    "timebase_fields",
    "subject_id_fields",
)

GATE_FIELDS = (
    "train_test_split_rule",
    "baseline_models",
    "null_models",
    "candidate_terms",
    "expected_gate",
)

REPRODUCIBLE_FIELDS = (
    "local_gate_command",
    "local_artifacts",
)

ARTIFACT_FIELDS = (
    "artifact_type",
    "artifact_version",
    "source_id",
    "criteria",
    "gate_passed",
    "result",
)

LOCOMOTION_ARTIFACT_TYPES = (
    "clarus_locomotion_gate",
    "clarus_locomotion_control_gate",
)


@dataclass(frozen=True)
class EvidenceCheck:
    """Result of checking one external evidence manifest."""

    source_id: str
    readiness: str
    missing: tuple[str, ...]
    next_action: str

    @property
    def is_gate_ready(self) -> bool:
        return READINESS_ORDER.index(self.readiness) >= READINESS_ORDER.index("gate-ready")

    @property
    def is_reproducible(self) -> bool:
        return self.readiness == "reproducible"


@dataclass(frozen=True)
class ArtifactCheck:
    """Result of validating a gate artifact."""

    source_id: str
    artifact_type: str
    passed: bool
    missing: tuple[str, ...]
    next_action: str

    @property
    def is_reproducible(self) -> bool:
        return self.passed and not self.missing

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "artifact_type": self.artifact_type,
            "passed": self.passed,
            "is_reproducible": self.is_reproducible,
            "missing": self.missing,
            "next_action": self.next_action,
        }


@dataclass(frozen=True)
class LinearDecoderGate:
    """Held-out linear decoder result for one neural-to-behavior target."""

    n_train: int
    n_test: int
    r2_model: float
    r2_baseline: float
    delta_r2: float
    p_value: float | None

    @property
    def passed(self) -> bool:
        return self.delta_r2 > 0.0 and (self.p_value is None or self.p_value < 0.05)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "r2_model": self.r2_model,
            "r2_baseline": self.r2_baseline,
            "delta_r2": self.delta_r2,
            "p_value": self.p_value,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class LocomotionGatePanel:
    """Panel-level C. elegans locomotion decoder gate result."""

    results: Mapping[str, Mapping[str, LinearDecoderGate]]

    @property
    def recording_count(self) -> int:
        return len(self.results)

    def pass_count(self, target: str) -> int:
        return sum(1 for result in self.results.values() if result[target].passed)

    def pass_rate(self, target: str) -> float:
        if not self.results:
            return 0.0
        return self.pass_count(target) / len(self.results)

    def summary(self, targets: tuple[str, ...] = ("velocity", "curvature")) -> dict[str, object]:
        target_summary: dict[str, object] = {}
        for target in targets:
            gates = [result[target] for result in self.results.values() if target in result]
            if not gates:
                target_summary[target] = {
                    "pass_count": 0,
                    "pass_rate": 0.0,
                    "mean_delta_r2": 0.0,
                    "mean_r2_model": 0.0,
                    "mean_r2_baseline": 0.0,
                }
                continue
            target_summary[target] = {
                "pass_count": sum(1 for gate in gates if gate.passed),
                "pass_rate": sum(1 for gate in gates if gate.passed) / len(gates),
                "mean_delta_r2": float(np.mean([gate.delta_r2 for gate in gates])),
                "mean_r2_model": float(np.mean([gate.r2_model for gate in gates])),
                "mean_r2_baseline": float(np.mean([gate.r2_baseline for gate in gates])),
            }
        return {
            "recording_count": self.recording_count,
            "targets": target_summary,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary(),
            "recordings": {
                recording_id: {
                    target: gate.to_dict()
                    for target, gate in target_results.items()
                }
                for recording_id, target_results in self.results.items()
            },
        }

    def passed(
        self,
        *,
        min_pass_rate: float,
        targets: tuple[str, ...] = ("velocity", "curvature"),
    ) -> bool:
        return all(self.pass_rate(target) >= min_pass_rate for target in targets)


@dataclass(frozen=True)
class LocomotionControlComparison:
    """Treatment-vs-control comparison for a locomotion decoder gate."""

    treatment: LocomotionGatePanel
    control: LocomotionGatePanel
    min_pass_rate: float
    min_control_delta: float
    targets: tuple[str, ...] = ("velocity", "curvature")

    def target_summary(self, target: str) -> dict[str, object]:
        treatment_rate = self.treatment.pass_rate(target)
        control_rate = self.control.pass_rate(target)
        delta = treatment_rate - control_rate
        return {
            "treatment_pass_rate": treatment_rate,
            "control_pass_rate": control_rate,
            "pass_rate_delta": delta,
            "passed": treatment_rate >= self.min_pass_rate and delta >= self.min_control_delta,
        }

    @property
    def passed(self) -> bool:
        return all(bool(self.target_summary(target)["passed"]) for target in self.targets)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "min_pass_rate": self.min_pass_rate,
            "min_control_delta": self.min_control_delta,
            "targets": {target: self.target_summary(target) for target in self.targets},
            "treatment": self.treatment.to_dict(),
            "control": self.control.to_dict(),
        }


def assess_manifest(manifest: Mapping[str, object]) -> EvidenceCheck:
    """Assess how far an external dataset can move through the gate pipeline."""

    source_id = str(manifest.get("source_id") or "unknown")
    missing_metadata = _missing(manifest, METADATA_FIELDS)
    if missing_metadata:
        return EvidenceCheck(
            source_id=source_id,
            readiness="metadata-only",
            missing=missing_metadata,
            next_action="fill required source metadata before using this dataset",
        )

    missing_download = _missing(manifest, ("raw_files",))
    if missing_download:
        return EvidenceCheck(
            source_id=source_id,
            readiness="metadata-only",
            missing=missing_download,
            next_action="locate public raw file URLs or document the API access path",
        )

    missing_fields = _missing(manifest, FIELD_FIELDS)
    if missing_fields:
        return EvidenceCheck(
            source_id=source_id,
            readiness="download-ready",
            missing=missing_fields,
            next_action="inspect files and map neural, behavior, stimulus, timebase, and subject fields",
        )

    missing_gate = _missing(manifest, GATE_FIELDS)
    if missing_gate:
        return EvidenceCheck(
            source_id=source_id,
            readiness="field-ready",
            missing=missing_gate,
            next_action="define split rule, baselines, nulls, candidate terms, and expected gate",
        )

    missing_repro = _missing(manifest, REPRODUCIBLE_FIELDS)
    if missing_repro:
        return EvidenceCheck(
            source_id=source_id,
            readiness="gate-ready",
            missing=missing_repro,
            next_action="add a local command and generated artifacts after running the gate",
        )

    return EvidenceCheck(
        source_id=source_id,
        readiness="reproducible",
        missing=(),
        next_action="rerun the local gate before promoting or editing equations",
    )


def validate_locomotion_artifact(artifact: Mapping[str, object]) -> ArtifactCheck:
    """Validate whether a locomotion JSON artifact can support promotion."""

    source_id = str(artifact.get("source_id") or "unknown")
    artifact_type = str(artifact.get("artifact_type") or "unknown")
    missing = _missing(artifact, ARTIFACT_FIELDS)
    if missing:
        return ArtifactCheck(
            source_id=source_id,
            artifact_type=artifact_type,
            passed=False,
            missing=missing,
            next_action="rerun the gate to produce a complete artifact",
        )
    if artifact_type not in LOCOMOTION_ARTIFACT_TYPES:
        return ArtifactCheck(
            source_id=source_id,
            artifact_type=artifact_type,
            passed=False,
            missing=("artifact_type",),
            next_action="use a supported locomotion artifact type",
        )
    if artifact.get("gate_passed") is not True:
        return ArtifactCheck(
            source_id=source_id,
            artifact_type=artifact_type,
            passed=False,
            missing=(),
            next_action="gate artifact is complete but did not pass",
        )
    return ArtifactCheck(
        source_id=source_id,
        artifact_type=artifact_type,
        passed=True,
        missing=(),
        next_action="attach this artifact as local_artifacts before promotion",
    )


def validate_locomotion_artifact_file(path: str | Path) -> ArtifactCheck:
    """Read and validate a locomotion gate artifact JSON file."""

    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        return ArtifactCheck(
            source_id="unknown",
            artifact_type="unknown",
            passed=False,
            missing=("artifact",),
            next_action="artifact JSON must be an object",
        )
    return validate_locomotion_artifact(artifact)


def celegans_elife_66135_manifest() -> dict[str, object]:
    """Known manifest facts for the first C. elegans locomotion target."""

    return {
        "source_id": "celegans_elife_66135_locomotion",
        "species": "C. elegans",
        "paper_or_dataset_url": "https://elifesciences.org/articles/66135",
        "license_or_access": "public OSF data and GPL-2.0 analysis code; verify file-level terms before redistribution",
        "raw_files": (
            "https://osf.io/dpr3h/",
            "https://github.com/leiferlab/PredictionCode",
        ),
        "recording_files": (
            "centerline.mat",
            "heatData.mat",
            "heatDataMS.mat",
            "pointStatsNew.mat",
            "positionDataMS.mat",
        ),
        "neural_fields": (
            "Neurons.I_smooth_interp_crop_noncontig",
            "derived neuron_derivatives = d/dt(Neurons.I_smooth_interp_crop_noncontig)",
        ),
        "behavior_fields": (
            "Behavior_crop_noncontig.CMSVelocity",
            "Behavior_crop_noncontig.Curvature",
        ),
        "stimulus_fields": (
            "strain_condition directory, e.g. AML310_moving, AML32_moving, AML18_moving",
            "BFP cutoff volume for AML310 identity recordings",
        ),
        "timebase_fields": (
            "Neurons.I_Time_crop_noncontig",
            "heatDataMS.hasPointsTime",
            "heatDataMS.clTime",
        ),
        "subject_id_fields": (
            "strain_condition",
            "recording folder key",
        ),
        "field_mapping_source": "leiferlab/PredictionCode utility/get_all_recordings.py and utility/data_handler.py",
        "train_test_split_rule": "held-out recording or blocked time split",
        "baseline_models": ("behavior autocorrelation baseline",),
        "null_models": ("time-shuffled neural activity", "recording-label permutation"),
        "candidate_terms": ("neural population -> velocity/curvature decoder",),
        "expected_gate": "positive held-out decoding over baseline and nulls",
    }


def linear_decoder_gate(
    features: object,
    target: object,
    *,
    train_fraction: float = 0.7,
    ridge: float = 1.0,
    n_permutations: int = 0,
    seed: int = 0,
) -> LinearDecoderGate:
    """Run a minimal held-out ridge decoder gate.

    ``features`` may be shaped as ``time x features`` or ``features x time``.
    The baseline is the training-set mean target. Permutations shuffle the
    training target, giving a one-sided p-value for ``delta_r2``.
    """

    x, y = _clean_xy(features, target)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    n_train = int(x.shape[0] * train_fraction)
    if n_train < 2 or x.shape[0] - n_train < 2:
        raise ValueError("need at least two train and two test samples after cleaning")

    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    x_train, x_test = _standardize_train_test(x_train, x_test)

    model_pred = _fit_predict_ridge(x_train, y_train, x_test, ridge)
    baseline_pred = np.full_like(y_test, float(np.mean(y_train)), dtype=float)
    r2_model = _r2(y_test, model_pred)
    r2_baseline = _r2(y_test, baseline_pred)
    delta_r2 = r2_model - r2_baseline

    p_value = None
    if n_permutations > 0:
        rng = np.random.default_rng(seed)
        null_delta = np.empty(n_permutations, dtype=float)
        for idx in range(n_permutations):
            shuffled = rng.permutation(y_train)
            shuffled_pred = _fit_predict_ridge(x_train, shuffled, x_test, ridge)
            null_delta[idx] = _r2(y_test, shuffled_pred) - r2_baseline
        p_value = float((np.sum(null_delta >= delta_r2) + 1) / (n_permutations + 1))

    return LinearDecoderGate(
        n_train=n_train,
        n_test=x.shape[0] - n_train,
        r2_model=float(r2_model),
        r2_baseline=float(r2_baseline),
        delta_r2=float(delta_r2),
        p_value=p_value,
    )


def celegans_locomotion_gate(
    recordings: Mapping[str, Mapping[str, object]],
    *,
    targets: tuple[str, ...] = ("velocity", "curvature"),
    include_derivatives: bool = True,
    train_fraction: float = 0.7,
    ridge: float = 1.0,
    n_permutations: int = 0,
    seed: int = 0,
) -> LocomotionGatePanel:
    """Run velocity/curvature gates on PredictionCode preprocessed recordings."""

    results: dict[str, dict[str, LinearDecoderGate]] = {}
    for recording_id, recording in recordings.items():
        features = _celegans_features(recording, include_derivatives=include_derivatives)
        target_results: dict[str, LinearDecoderGate] = {}
        for target in targets:
            if target not in recording:
                raise KeyError(f"recording {recording_id!r} is missing target {target!r}")
            target_results[target] = linear_decoder_gate(
                features,
                recording[target],
                train_fraction=train_fraction,
                ridge=ridge,
                n_permutations=n_permutations,
                seed=seed,
            )
        results[recording_id] = target_results
    return LocomotionGatePanel(results=results)


def celegans_locomotion_gate_from_pickle(
    path: str | Path,
    *,
    targets: tuple[str, ...] = ("velocity", "curvature"),
    include_derivatives: bool = True,
    train_fraction: float = 0.7,
    ridge: float = 1.0,
    n_permutations: int = 0,
    seed: int = 0,
) -> LocomotionGatePanel:
    """Load a PredictionCode ``*_recordings.dat`` pickle and run the panel gate."""

    with Path(path).open("rb") as handle:
        recordings = pickle.load(handle, encoding="latin1")
    return celegans_locomotion_gate(
        recordings,
        targets=targets,
        include_derivatives=include_derivatives,
        train_fraction=train_fraction,
        ridge=ridge,
        n_permutations=n_permutations,
        seed=seed,
    )


def build_locomotion_gate_artifact(
    panel: LocomotionGatePanel,
    *,
    source_id: str = "celegans_elife_66135_locomotion",
    recordings_pickle: str | None = None,
    min_pass_rate: float = 0.0,
    permutations: int = 0,
    ridge: float = 1.0,
    train_fraction: float = 0.7,
    include_derivatives: bool = True,
) -> dict[str, object]:
    """Build a self-describing JSON artifact for one locomotion gate run."""

    return {
        "artifact_type": "clarus_locomotion_gate",
        "artifact_version": 1,
        "source_id": source_id,
        "recordings_pickle": recordings_pickle,
        "criteria": {
            "min_pass_rate": min_pass_rate,
            "permutations": permutations,
            "ridge": ridge,
            "train_fraction": train_fraction,
            "include_derivatives": include_derivatives,
        },
        "gate_passed": panel.passed(min_pass_rate=min_pass_rate),
        "result": panel.to_dict(),
    }


def build_locomotion_control_artifact(
    comparison: LocomotionControlComparison,
    *,
    source_id: str = "celegans_elife_66135_locomotion",
    treatment_pickle: str | None = None,
    control_pickle: str | None = None,
    permutations: int = 0,
    ridge: float = 1.0,
    train_fraction: float = 0.7,
    include_derivatives: bool = True,
) -> dict[str, object]:
    """Build a self-describing JSON artifact for a treatment-vs-control gate."""

    return {
        "artifact_type": "clarus_locomotion_control_gate",
        "artifact_version": 1,
        "source_id": source_id,
        "treatment_pickle": treatment_pickle,
        "control_pickle": control_pickle,
        "criteria": {
            "min_pass_rate": comparison.min_pass_rate,
            "min_control_delta": comparison.min_control_delta,
            "permutations": permutations,
            "ridge": ridge,
            "train_fraction": train_fraction,
            "include_derivatives": include_derivatives,
        },
        "gate_passed": comparison.passed,
        "result": comparison.to_dict(),
    }


def compare_locomotion_to_control(
    treatment: LocomotionGatePanel,
    control: LocomotionGatePanel,
    *,
    min_pass_rate: float,
    min_control_delta: float,
    targets: tuple[str, ...] = ("velocity", "curvature"),
) -> LocomotionControlComparison:
    """Compare an experimental locomotion panel against a matched control panel."""

    return LocomotionControlComparison(
        treatment=treatment,
        control=control,
        min_pass_rate=min_pass_rate,
        min_control_delta=min_control_delta,
        targets=targets,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the C. elegans locomotion decoder gate on a PredictionCode pickle."
    )
    parser.add_argument(
        "recordings_pickle",
        nargs="?",
        help="Path to gcamp_recordings.dat or gfp_recordings.dat",
    )
    parser.add_argument("--validate-artifact", help="Validate an existing gate artifact JSON")
    parser.add_argument("--control-pickle", help="Optional GFP/control recordings pickle")
    parser.add_argument("--permutations", type=int, default=0, help="Permutation count for p-values")
    parser.add_argument("--ridge", type=float, default=1.0, help="Ridge penalty")
    parser.add_argument("--train-fraction", type=float, default=0.7, help="Blocked train fraction")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for permutations")
    parser.add_argument("--output", help="Optional path to write the JSON result artifact")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="Required pass rate for every target; returns exit code 2 on failure",
    )
    parser.add_argument(
        "--min-control-delta",
        type=float,
        default=0.0,
        help="Required treatment minus control pass-rate margin when --control-pickle is used",
    )
    parser.add_argument(
        "--no-derivatives",
        action="store_true",
        help="Use only neural activity, without neuron_derivatives",
    )
    args = parser.parse_args(argv)
    if args.validate_artifact:
        check = validate_locomotion_artifact_file(args.validate_artifact)
        print(json.dumps(check.to_dict(), indent=2, sort_keys=True))
        return 0 if check.is_reproducible else 2
    if not args.recordings_pickle:
        parser.error("recordings_pickle is required unless --validate-artifact is provided")
    if not 0.0 <= args.min_pass_rate <= 1.0:
        parser.error("--min-pass-rate must be between 0 and 1")
    if not 0.0 <= args.min_control_delta <= 1.0:
        parser.error("--min-control-delta must be between 0 and 1")
    panel = celegans_locomotion_gate_from_pickle(
        args.recordings_pickle,
        include_derivatives=not args.no_derivatives,
        train_fraction=args.train_fraction,
        ridge=args.ridge,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    if args.control_pickle:
        control = celegans_locomotion_gate_from_pickle(
            args.control_pickle,
            include_derivatives=not args.no_derivatives,
            train_fraction=args.train_fraction,
            ridge=args.ridge,
            n_permutations=args.permutations,
            seed=args.seed,
        )
        comparison = compare_locomotion_to_control(
            panel,
            control,
            min_pass_rate=args.min_pass_rate,
            min_control_delta=args.min_control_delta,
        )
        payload = build_locomotion_control_artifact(
            comparison,
            treatment_pickle=args.recordings_pickle,
            control_pickle=args.control_pickle,
            permutations=args.permutations,
            ridge=args.ridge,
            train_fraction=args.train_fraction,
            include_derivatives=not args.no_derivatives,
        )
        passed = comparison.passed
    else:
        payload = build_locomotion_gate_artifact(
            panel,
            recordings_pickle=args.recordings_pickle,
            min_pass_rate=args.min_pass_rate,
            permutations=args.permutations,
            ridge=args.ridge,
            train_fraction=args.train_fraction,
            include_derivatives=not args.no_derivatives,
        )
        passed = panel.passed(min_pass_rate=args.min_pass_rate)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if passed else 2


def _missing(manifest: Mapping[str, object], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(field for field in fields if _empty(manifest.get(field)))


def _empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        return len(value) == 0
    return False


def _clean_xy(features: object, target: object) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    if x.ndim != 2:
        raise ValueError("features must be a 2D array")
    if x.shape[0] != y.shape[0] and x.shape[1] == y.shape[0]:
        x = x.T
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and target must share the time dimension")

    finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
    return x[finite], y[finite]


def _celegans_features(
    recording: Mapping[str, object],
    *,
    include_derivatives: bool,
) -> np.ndarray:
    if "neurons" not in recording:
        raise KeyError("recording is missing neural activity field 'neurons'")

    neurons = np.asarray(recording["neurons"], dtype=float)
    if not include_derivatives:
        return neurons

    derivatives = recording.get("neuron_derivatives")
    if derivatives is None:
        return neurons
    return np.vstack([neurons, np.asarray(derivatives, dtype=float)])


def _standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std[std == 0.0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std


def _fit_predict_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    ridge: float,
) -> np.ndarray:
    x_train_i = np.column_stack([np.ones(x_train.shape[0]), x_train])
    x_test_i = np.column_stack([np.ones(x_test.shape[0]), x_test])
    penalty = np.eye(x_train_i.shape[1]) * ridge
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(x_train_i.T @ x_train_i + penalty, x_train_i.T @ y_train)
    return x_test_i @ weights


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom == 0.0:
        return 0.0
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom


__all__ = [
    "ArtifactCheck",
    "EvidenceCheck",
    "LinearDecoderGate",
    "LocomotionControlComparison",
    "LocomotionGatePanel",
    "READINESS_ORDER",
    "assess_manifest",
    "build_locomotion_control_artifact",
    "build_locomotion_gate_artifact",
    "celegans_elife_66135_manifest",
    "celegans_locomotion_gate",
    "celegans_locomotion_gate_from_pickle",
    "compare_locomotion_to_control",
    "linear_decoder_gate",
    "main",
    "validate_locomotion_artifact",
    "validate_locomotion_artifact_file",
]


if __name__ == "__main__":
    raise SystemExit(main())
