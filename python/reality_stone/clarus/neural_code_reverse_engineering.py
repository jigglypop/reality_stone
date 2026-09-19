"""Claim-by-claim reverse engineering of a neural-code candidate.

This module does two things that must not be conflated:

* it reconstructs the *task-design* factorization in the Tafazoli et al.
  experiment (read shape/colour, then route to response axis 1/2); and
* it audits whether the released processed neural artifacts can identify that
  factorization as a neural instruction set.

The answer is intentionally not one global ``maybe``.  Every claim receives a
separate ``YES``, ``NO``, or ``TEST_UNAVAILABLE`` answer.  Here ``NO`` answers
an identification question ("was it identified?"), not an ontological
question ("can it exist anywhere in the brain?").

The released ``PFC_ClassifierData.mat`` contains a 403-column
pseudopopulation.  Neurons sharing the same sampled-trial signature can be
grouped back into recording sessions.  This module performs that recovery and
forbids treating all 403 columns as one simultaneous neural state.

SciPy is imported only by :func:`run_tafazoli_classifier_snapshot_audit`, so
manifest validation and claim logic retain the package's base NumPy-only
dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import isclose, isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from reality_stone.clarus.tafazoli_processed_audit import (
    PROCESSED_NEURAL_FIGURE_AUDIT_PASS,
    TafazoliProcessedAuditReport,
)


SCHEMA_VERSION = "clarus-neural-code-reverse-engineering/v1"
REPORT_SCHEMA_VERSION = "clarus-neural-code-reverse-engineering-report/v1"
REVERSE_ENGINEERING_SCOPE = (
    "official_processed_neural_code_claim_by_claim_verdict"
)
CODE_SKELETON_ONLY_STATUS = (
    "TASK_CODE_SKELETON_RECONSTRUCTED_NEURAL_LANGUAGE_NOT_IDENTIFIED"
)

YES = "YES"
NO = "NO"
TEST_UNAVAILABLE = "TEST_UNAVAILABLE"
_ANSWERS = frozenset({YES, NO, TEST_UNAVAILABLE})

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "scope",
        "description",
        "classifier_snapshot",
        "task_grid",
        "published_reuse_thresholds",
        "identification_requirements",
        "current_capabilities",
        "excluded_inferences",
    }
)
_SNAPSHOT_KEYS = frozenset(
    {
        "filename",
        "md5",
        "expected_neuron_count",
        "expected_timepoint_count",
        "expected_time_step_seconds",
        "temporal_count_window_seconds",
        "expected_session_group_count",
        "expected_animals",
        "primary_discovery_dimensions_one_based",
        "excluded_discovery_dimensions_one_based",
    }
)
_TASK_GRID_KEYS = frozenset(
    {
        "read_primitives",
        "route_primitives",
        "observed_tasks",
        "predicted_missing_task",
    }
)
_TASK_KEYS = frozenset({"task", "read_primitive", "route_primitive"})
_THRESHOLD_KEYS = frozenset(
    {
        "bidirectional_cross_decoder_peak_accuracy_min",
        "bidirectional_cross_decoder_post_event_mean_accuracy_min",
    }
)
_REQUIREMENT_KEYS = frozenset(
    {
        "session_local_discovery",
        "heldout_session_frozen_operator_transfer",
        "both_animals_same_direction",
        "same_rank_continuous_dynamics_comparison",
        "fixed_neuron_opcode_comparison",
        "time_reversal_and_trial_shuffle_controls",
        "movement_reward_rt_history_conditioning",
        "complete_unseen_factorial_composition",
        "selective_perturbation_and_rescue_for_causality",
        "multi_area_local_dsl_vs_global_comparison",
        "minimal_sufficient_region_recruitment_curve",
        "optimizer_timescale_intervention",
    }
)
_CAPABILITY_KEYS = frozenset(
    {
        "official_author_processed_outputs",
        "raw_trial_archive",
        "simultaneous_403_neuron_population",
        "session_groups_recoverable",
        "all_classifier_resamples_available",
        "complete_factorial_task_grid",
        "unseen_composition_recorded",
        "movement_covariates_available",
        "causal_perturbation_and_rescue_available",
        "three_family_model_comparison_completed",
        "heldout_session_frozen_operator_transfer_completed",
    }
)
_REQUIRED_CROSS_DECODER_NAMES = (
    "cross_color_C1_to_C2",
    "cross_color_C2_to_C1",
    "cross_response_C1_to_S1",
    "cross_response_S1_to_C1",
)


@dataclass(frozen=True)
class ClassifierSnapshotSpec:
    """Integrity and layout declaration for one saved classifier snapshot."""

    filename: str
    md5: str
    expected_neuron_count: int
    expected_timepoint_count: int
    expected_time_step_seconds: float
    temporal_count_window_seconds: float
    expected_session_group_count: int
    expected_animals: tuple[str, ...]
    primary_discovery_dimensions_one_based: tuple[int, ...]
    excluded_discovery_dimensions_one_based: tuple[int, ...]


@dataclass(frozen=True)
class TaskInstruction:
    """One task expressed in the candidate two-slot instruction format."""

    task: str
    read_primitive: str
    route_primitive: str
    observed: bool

    @property
    def program(self) -> tuple[str, str]:
        return (self.read_primitive, self.route_primitive)


@dataclass(frozen=True)
class TaskGrid:
    """Observed three cells and the uniquely missing cell of a 2x2 grid."""

    read_primitives: tuple[str, ...]
    route_primitives: tuple[str, ...]
    observed_tasks: tuple[TaskInstruction, ...]
    predicted_missing_task: TaskInstruction


@dataclass(frozen=True)
class PublishedReuseThresholds:
    """Sanity thresholds for already-published decoder artifacts only."""

    bidirectional_cross_decoder_peak_accuracy_min: float
    bidirectional_cross_decoder_post_event_mean_accuracy_min: float


@dataclass(frozen=True)
class NeuralCodeReverseEngineeringManifest:
    """Strict preregistration of the current claim boundary."""

    schema_version: str
    scope: str
    description: str
    classifier_snapshot: ClassifierSnapshotSpec
    task_grid: TaskGrid
    published_reuse_thresholds: PublishedReuseThresholds
    identification_requirements: Mapping[str, bool]
    current_capabilities: Mapping[str, bool]
    excluded_inferences: tuple[str, ...]


@dataclass(frozen=True)
class SessionGroupAudit:
    """A contiguous recording-session group recovered from trial signatures."""

    session_index_one_based: int
    neuron_column_start_one_based: int
    neuron_column_end_one_based: int
    neuron_count: int
    animal: str


@dataclass(frozen=True)
class SnapshotDimensionAudit:
    """Shape and temporal-overlap diagnostic for one classifier dimension."""

    dimension_one_based: int
    target_factor: str
    train_shape: tuple[int, int, int]
    test_shape: tuple[int, int, int]
    train_dtype: str
    test_dtype: str
    mean_adjacent_time_bin_correlation: float
    primary_discovery_allowed: bool
    exclusion_reason: str | None


@dataclass(frozen=True)
class TafazoliClassifierSnapshotReport:
    """Data-fitness audit for the saved pseudopopulation snapshot."""

    filename: str
    expected_md5: str
    observed_md5: str
    checksum_matches: bool
    neuron_count: int
    timepoint_count: int
    time_start_seconds: float
    time_end_seconds: float
    time_step_seconds: float
    temporal_count_window_seconds: float
    adjacent_window_overlap_fraction: float
    classifier_snapshot_count: int
    full_pseudopopulation_is_simultaneous: bool
    session_groups_recoverable: bool
    session_groups: tuple[SessionGroupAudit, ...]
    animal_neuron_counts: tuple[tuple[str, int], ...]
    dimensions: tuple[SnapshotDimensionAudit, ...]
    dim_train_stim_indices_field_overwritten: bool
    dim2_transductive_mean_subtraction_warning: bool
    all_classifier_resamples_available: bool
    raw_trial_archive_available: bool
    session_local_operator_pilot_possible: bool
    full_neural_language_inverse_problem_possible: bool
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class PublishedDecoderMetric:
    """The two descriptive values used for one cross-task curve."""

    name: str
    peak_accuracy: float
    post_event_mean_accuracy: float


@dataclass(frozen=True)
class ClaimVerdict:
    """One scoped scientific question and an explicit answer."""

    key: str
    question: str
    answer: str
    claim_scope: str
    basis: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.answer not in _ANSWERS:
            raise ValueError(f"unsupported claim answer: {self.answer!r}")


@dataclass(frozen=True)
class NeuralCodeReverseEngineeringReport:
    """Serializable current verdict, with no global ontological shortcut."""

    schema_version: str
    scope: str
    method_status: str
    task_programs: tuple[TaskInstruction, ...]
    missing_composition_prediction: TaskInstruction
    snapshot: TafazoliClassifierSnapshotReport
    published_decoder_metrics: tuple[PublishedDecoderMetric, ...]
    published_cross_task_decoder_artifact_pass: bool
    competing_family_winner: str
    claim_verdicts: tuple[ClaimVerdict, ...]
    next_decisive_experiment: tuple[str, ...]
    excluded_inferences: tuple[str, ...]
    conclusion: str

    def claim(self, key: str) -> ClaimVerdict:
        """Return one verdict by stable key."""

        matches = tuple(item for item in self.claim_verdicts if item.key == key)
        if len(matches) != 1:
            raise KeyError(f"claim key is not unique or does not exist: {key!r}")
        return matches[0]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _require_exact_keys(
    value: Any,
    *,
    required: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    keys = frozenset(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        raise ValueError(f"{label} is missing required keys: {missing}")
    if unknown:
        raise ValueError(f"{label} has unknown keys: {unknown}")
    return value


def _strict_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _strict_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _strict_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _strict_bool_mapping(
    value: Any,
    *,
    required: frozenset[str],
    label: str,
) -> Mapping[str, bool]:
    raw = _require_exact_keys(value, required=required, label=label)
    result = {}
    for key in sorted(required):
        if type(raw[key]) is not bool:
            raise TypeError(f"{label}.{key} must be a boolean")
        result[key] = raw[key]
    return result


def _strict_string_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{label} must be a non-empty JSON array")
    result = tuple(
        _strict_string(item, label=f"{label} item") for item in value
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _strict_int_tuple(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{label} must be a non-empty JSON array")
    result = tuple(
        _strict_int(item, label=f"{label} item") for item in value
    )
    if min(result) < 1 or len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique positive integers")
    return result


def _parse_task(value: Any, *, label: str, observed: bool) -> TaskInstruction:
    raw = _require_exact_keys(value, required=_TASK_KEYS, label=label)
    return TaskInstruction(
        task=_strict_string(raw["task"], label=f"{label}.task"),
        read_primitive=_strict_string(
            raw["read_primitive"],
            label=f"{label}.read_primitive",
        ),
        route_primitive=_strict_string(
            raw["route_primitive"],
            label=f"{label}.route_primitive",
        ),
        observed=observed,
    )


def _validate_task_grid(task_grid: TaskGrid) -> None:
    expected_cells = {
        (read, route)
        for read in task_grid.read_primitives
        for route in task_grid.route_primitives
    }
    observed_cells = {item.program for item in task_grid.observed_tasks}
    if len(task_grid.read_primitives) != 2:
        raise ValueError("task_grid must declare exactly two read primitives")
    if len(task_grid.route_primitives) != 2:
        raise ValueError("task_grid must declare exactly two route primitives")
    if len(task_grid.observed_tasks) != 3:
        raise ValueError("task_grid must contain exactly three observed cells")
    if len({item.task for item in task_grid.observed_tasks}) != 3:
        raise ValueError("observed task names must be unique")
    if not observed_cells <= expected_cells:
        raise ValueError("an observed task is outside the declared 2x2 grid")
    missing_cells = expected_cells - observed_cells
    if missing_cells != {task_grid.predicted_missing_task.program}:
        raise ValueError(
            "predicted_missing_task must be the unique unobserved grid cell"
        )
    if task_grid.predicted_missing_task.task in {
        item.task for item in task_grid.observed_tasks
    }:
        raise ValueError("predicted missing task name must be unobserved")


def load_neural_code_reverse_engineering_manifest(
    path: str | Path,
) -> NeuralCodeReverseEngineeringManifest:
    """Strictly load and validate the reverse-engineering declaration."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    top = _require_exact_keys(
        payload,
        required=_TOP_LEVEL_KEYS,
        label="manifest",
    )
    schema_version = _strict_string(
        top["schema_version"],
        label="schema_version",
    )
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SCHEMA_VERSION!r}")
    scope = _strict_string(top["scope"], label="scope")
    if scope != REVERSE_ENGINEERING_SCOPE:
        raise ValueError(f"scope must equal {REVERSE_ENGINEERING_SCOPE!r}")

    snapshot_raw = _require_exact_keys(
        top["classifier_snapshot"],
        required=_SNAPSHOT_KEYS,
        label="classifier_snapshot",
    )
    md5 = _strict_string(
        snapshot_raw["md5"],
        label="classifier_snapshot.md5",
    ).lower()
    if len(md5) != 32 or any(
        character not in "0123456789abcdef" for character in md5
    ):
        raise ValueError("classifier_snapshot.md5 must be lowercase MD5 hex")
    snapshot = ClassifierSnapshotSpec(
        filename=_strict_string(
            snapshot_raw["filename"],
            label="classifier_snapshot.filename",
        ),
        md5=md5,
        expected_neuron_count=_strict_int(
            snapshot_raw["expected_neuron_count"],
            label="classifier_snapshot.expected_neuron_count",
        ),
        expected_timepoint_count=_strict_int(
            snapshot_raw["expected_timepoint_count"],
            label="classifier_snapshot.expected_timepoint_count",
        ),
        expected_time_step_seconds=_strict_float(
            snapshot_raw["expected_time_step_seconds"],
            label="classifier_snapshot.expected_time_step_seconds",
        ),
        temporal_count_window_seconds=_strict_float(
            snapshot_raw["temporal_count_window_seconds"],
            label="classifier_snapshot.temporal_count_window_seconds",
        ),
        expected_session_group_count=_strict_int(
            snapshot_raw["expected_session_group_count"],
            label="classifier_snapshot.expected_session_group_count",
        ),
        expected_animals=_strict_string_tuple(
            snapshot_raw["expected_animals"],
            label="classifier_snapshot.expected_animals",
        ),
        primary_discovery_dimensions_one_based=_strict_int_tuple(
            snapshot_raw["primary_discovery_dimensions_one_based"],
            label=(
                "classifier_snapshot."
                "primary_discovery_dimensions_one_based"
            ),
        ),
        excluded_discovery_dimensions_one_based=_strict_int_tuple(
            snapshot_raw["excluded_discovery_dimensions_one_based"],
            label=(
                "classifier_snapshot."
                "excluded_discovery_dimensions_one_based"
            ),
        ),
    )
    if min(
        snapshot.expected_neuron_count,
        snapshot.expected_timepoint_count,
        snapshot.expected_session_group_count,
    ) <= 0:
        raise ValueError("snapshot expected counts must be positive")
    if min(
        snapshot.expected_time_step_seconds,
        snapshot.temporal_count_window_seconds,
    ) <= 0:
        raise ValueError("snapshot time declarations must be positive")
    if set(snapshot.primary_discovery_dimensions_one_based) & set(
        snapshot.excluded_discovery_dimensions_one_based
    ):
        raise ValueError("primary and excluded dimensions must be disjoint")

    grid_raw = _require_exact_keys(
        top["task_grid"],
        required=_TASK_GRID_KEYS,
        label="task_grid",
    )
    observed_raw = grid_raw["observed_tasks"]
    if not isinstance(observed_raw, list):
        raise TypeError("task_grid.observed_tasks must be a JSON array")
    task_grid = TaskGrid(
        read_primitives=_strict_string_tuple(
            grid_raw["read_primitives"],
            label="task_grid.read_primitives",
        ),
        route_primitives=_strict_string_tuple(
            grid_raw["route_primitives"],
            label="task_grid.route_primitives",
        ),
        observed_tasks=tuple(
            _parse_task(
                value,
                label=f"task_grid.observed_tasks[{index}]",
                observed=True,
            )
            for index, value in enumerate(observed_raw)
        ),
        predicted_missing_task=_parse_task(
            grid_raw["predicted_missing_task"],
            label="task_grid.predicted_missing_task",
            observed=False,
        ),
    )
    _validate_task_grid(task_grid)

    threshold_raw = _require_exact_keys(
        top["published_reuse_thresholds"],
        required=_THRESHOLD_KEYS,
        label="published_reuse_thresholds",
    )
    thresholds = PublishedReuseThresholds(
        **{
            key: _strict_float(
                threshold_raw[key],
                label=f"published_reuse_thresholds.{key}",
            )
            for key in sorted(_THRESHOLD_KEYS)
        }
    )
    if not all(0.0 <= value <= 1.0 for value in asdict(thresholds).values()):
        raise ValueError("published decoder thresholds must be in [0, 1]")

    requirements = _strict_bool_mapping(
        top["identification_requirements"],
        required=_REQUIREMENT_KEYS,
        label="identification_requirements",
    )
    if not all(requirements.values()):
        raise ValueError("every identification requirement must stay enabled")
    capabilities = _strict_bool_mapping(
        top["current_capabilities"],
        required=_CAPABILITY_KEYS,
        label="current_capabilities",
    )
    required_capability_values = {
        "official_author_processed_outputs": True,
        "raw_trial_archive": False,
        "simultaneous_403_neuron_population": False,
        "session_groups_recoverable": True,
        "all_classifier_resamples_available": False,
        "complete_factorial_task_grid": False,
        "unseen_composition_recorded": False,
        "movement_covariates_available": False,
        "causal_perturbation_and_rescue_available": False,
        "three_family_model_comparison_completed": False,
        "heldout_session_frozen_operator_transfer_completed": False,
    }
    if dict(capabilities) != required_capability_values:
        raise ValueError("current capability locks do not match this dataset")

    return NeuralCodeReverseEngineeringManifest(
        schema_version=schema_version,
        scope=scope,
        description=_strict_string(top["description"], label="description"),
        classifier_snapshot=snapshot,
        task_grid=task_grid,
        published_reuse_thresholds=thresholds,
        identification_requirements=requirements,
        current_capabilities=capabilities,
        excluded_inferences=_strict_string_tuple(
            top["excluded_inferences"],
            label="excluded_inferences",
        ),
    )


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mat_field(value: Any, name: str, *, label: str) -> Any:
    if isinstance(value, Mapping):
        try:
            return value[name]
        except KeyError as error:
            raise ValueError(f"{label} is missing field {name!r}") from error
    try:
        return getattr(value, name)
    except AttributeError as error:
        raise ValueError(f"{label} is missing field {name!r}") from error


def _unwrap_singleton_object(value: Any) -> Any:
    result = value
    while (
        isinstance(result, np.ndarray)
        and result.dtype == object
        and result.size == 1
    ):
        result = result.reshape(-1)[0]
    return result


def _object_items(value: Any, *, expected: int, label: str) -> tuple[Any, ...]:
    unwrapped = _unwrap_singleton_object(value)
    array = np.asarray(unwrapped, dtype=object).reshape(-1)
    if array.size != expected:
        raise ValueError(f"{label} must contain exactly {expected} items")
    return tuple(_unwrap_singleton_object(item) for item in array)


def _numeric_tensor(value: Any, *, label: str) -> np.ndarray:
    result = np.asarray(_unwrap_singleton_object(value))
    if result.ndim != 3:
        raise ValueError(f"{label} must be trial x neuron x time")
    if not np.issubdtype(result.dtype, np.number):
        raise TypeError(f"{label} must be numeric")
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite and non-empty")
    return result


def _mean_adjacent_correlation(values: np.ndarray) -> float:
    correlations = []
    for time_index in range(values.shape[2] - 1):
        first = values[:, :, time_index].astype(np.float64).reshape(-1)
        second = values[:, :, time_index + 1].astype(np.float64).reshape(-1)
        first -= first.mean()
        second -= second.mean()
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator > 0:
            correlations.append(float(first @ second / denominator))
    if not correlations:
        raise ValueError("adjacent-bin correlation is undefined")
    return float(np.mean(correlations))


def _canonical_signature(value: Any) -> Any:
    unwrapped = _unwrap_singleton_object(value)
    if isinstance(unwrapped, np.ndarray) and unwrapped.dtype == object:
        return tuple(
            _canonical_signature(item) for item in unwrapped.reshape(-1)
        )
    array = np.asarray(unwrapped)
    return (array.shape, array.dtype.str, tuple(array.reshape(-1).tolist()))


def _recover_session_groups(
    train_stim_indices: Any,
    animal_labels: Any,
    *,
    expected_neuron_count: int,
) -> tuple[SessionGroupAudit, ...]:
    cells = np.asarray(train_stim_indices, dtype=object).reshape(-1)
    animals = np.asarray(animal_labels).astype(str).reshape(-1)
    if cells.size != expected_neuron_count:
        raise ValueError("TrainStimInds length does not match neuron count")
    if animals.size != expected_neuron_count:
        raise ValueError("animal label length does not match neuron count")
    signatures = tuple(_canonical_signature(item) for item in cells)

    boundaries = [0]
    for index in range(1, len(signatures)):
        if signatures[index] != signatures[index - 1]:
            boundaries.append(index)
    boundaries.append(len(signatures))

    groups = []
    for group_index, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:]),
        start=1,
    ):
        group_animals = set(animals[start:end].tolist())
        if len(group_animals) != 1:
            raise ValueError("a recovered session crosses animal identity")
        groups.append(
            SessionGroupAudit(
                session_index_one_based=group_index,
                neuron_column_start_one_based=start + 1,
                neuron_column_end_one_based=end,
                neuron_count=end - start,
                animal=group_animals.pop(),
            )
        )
    return tuple(groups)


def _target_factor(options: Any, dimension_one_based: int) -> str:
    suffix = "" if dimension_one_based == 1 else f"_{dimension_one_based}ndD"
    if dimension_one_based == 3:
        suffix = "_3ndD"
    values = np.asarray(
        _mat_field(
            options,
            f"TargetFactors{suffix}",
            label="ClassifierOpts",
        ),
        dtype=object,
    ).reshape(-1)
    if values.size < 1:
        raise ValueError("target factor declaration is empty")
    return str(values[0])


def summarize_tafazoli_classifier_snapshot(
    spec: ClassifierSnapshotSpec,
    classifier_options: Any,
    time: Any,
    *,
    observed_md5: str,
) -> TafazoliClassifierSnapshotReport:
    """Audit one saved fold snapshot without using task labels for discovery."""

    if not isinstance(spec, ClassifierSnapshotSpec):
        raise TypeError("spec must be ClassifierSnapshotSpec")
    time_values = np.asarray(time, dtype=np.float64).reshape(-1)
    if time_values.size != spec.expected_timepoint_count:
        raise ValueError("snapshot timepoint count does not match manifest")
    if not np.isfinite(time_values).all():
        raise ValueError("snapshot time must be finite")
    differences = np.diff(time_values)
    if not np.allclose(
        differences,
        spec.expected_time_step_seconds,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("snapshot time grid does not match declared step")

    dimension_entries = _object_items(
        _mat_field(
            classifier_options,
            "Dimpredictors",
            label="ClassifierOpts",
        ),
        expected=3,
        label="ClassifierOpts.Dimpredictors",
    )
    dimension_audits = []
    neuron_count = None
    for dimension_index, entry in enumerate(dimension_entries, start=1):
        train_raw, test_raw = _object_items(
            entry,
            expected=2,
            label=f"Dimpredictors dimension {dimension_index}",
        )
        train = _numeric_tensor(
            train_raw,
            label=f"Dimpredictors[{dimension_index}] train",
        )
        test = _numeric_tensor(
            test_raw,
            label=f"Dimpredictors[{dimension_index}] test",
        )
        if train.shape[1:] != test.shape[1:]:
            raise ValueError("train/test neuron-time shapes must match")
        if train.shape[2] != spec.expected_timepoint_count:
            raise ValueError("dimension timepoint count does not match manifest")
        if neuron_count is None:
            neuron_count = train.shape[1]
        elif neuron_count != train.shape[1]:
            raise ValueError("classifier dimensions have different neurons")
        primary = (
            dimension_index in spec.primary_discovery_dimensions_one_based
        )
        exclusion_reason = None
        if dimension_index in spec.excluded_discovery_dimensions_one_based:
            primary = False
            exclusion_reason = (
                "cross-rule dimension retains author mean subtraction that "
                "used the complete rule pool; raw factors are unavailable for "
                "train-only recentering"
            )
        dimension_audits.append(
            SnapshotDimensionAudit(
                dimension_one_based=dimension_index,
                target_factor=_target_factor(
                    classifier_options,
                    dimension_index,
                ),
                train_shape=tuple(int(item) for item in train.shape),
                test_shape=tuple(int(item) for item in test.shape),
                train_dtype=str(train.dtype),
                test_dtype=str(test.dtype),
                mean_adjacent_time_bin_correlation=(
                    _mean_adjacent_correlation(train)
                ),
                primary_discovery_allowed=primary,
                exclusion_reason=exclusion_reason,
            )
        )
    if neuron_count != spec.expected_neuron_count:
        raise ValueError("snapshot neuron count does not match manifest")

    groups = _recover_session_groups(
        _mat_field(
            classifier_options,
            "TrainStimInds",
            label="ClassifierOpts",
        ),
        _mat_field(
            classifier_options,
            "IncludedNeu4Ana_Animal",
            label="ClassifierOpts",
        ),
        expected_neuron_count=spec.expected_neuron_count,
    )
    if len(groups) != spec.expected_session_group_count:
        raise ValueError("recovered session count does not match manifest")
    animals = tuple(sorted({item.animal for item in groups}))
    if set(animals) != set(spec.expected_animals):
        raise ValueError("recovered animals do not match manifest")
    animal_neuron_counts = tuple(
        (
            animal,
            sum(item.neuron_count for item in groups if item.animal == animal),
        )
        for animal in animals
    )

    overlap = 1.0 - (
        spec.expected_time_step_seconds
        / spec.temporal_count_window_seconds
    )
    if not 0.0 <= overlap < 1.0:
        raise ValueError("declared temporal windows have invalid overlap")
    checksum_matches = observed_md5.lower() == spec.md5
    return TafazoliClassifierSnapshotReport(
        filename=spec.filename,
        expected_md5=spec.md5,
        observed_md5=observed_md5.lower(),
        checksum_matches=checksum_matches,
        neuron_count=spec.expected_neuron_count,
        timepoint_count=spec.expected_timepoint_count,
        time_start_seconds=float(time_values[0]),
        time_end_seconds=float(time_values[-1]),
        time_step_seconds=spec.expected_time_step_seconds,
        temporal_count_window_seconds=spec.temporal_count_window_seconds,
        adjacent_window_overlap_fraction=overlap,
        classifier_snapshot_count=1,
        full_pseudopopulation_is_simultaneous=False,
        session_groups_recoverable=True,
        session_groups=groups,
        animal_neuron_counts=animal_neuron_counts,
        dimensions=tuple(dimension_audits),
        dim_train_stim_indices_field_overwritten=True,
        dim2_transductive_mean_subtraction_warning=True,
        all_classifier_resamples_available=False,
        raw_trial_archive_available=False,
        session_local_operator_pilot_possible=True,
        full_neural_language_inverse_problem_possible=False,
        limitations=(
            "the 403 columns concatenate 27 recording-session groups",
            "the file stores one overwritten fold snapshot rather than all "
            "250 classifier resamples",
            "100 ms counting windows advance by 10 ms and overlap by 90%",
            "dimension 2 has a transductive mean-subtraction caveat",
            "test sampled-trial identities were not saved",
            "session-local pilots are possible, but the complete raw neural "
            "inverse problem is not",
        ),
    )


def run_tafazoli_classifier_snapshot_audit(
    manifest: NeuralCodeReverseEngineeringManifest,
    data_directory: str | Path,
) -> TafazoliClassifierSnapshotReport:
    """Load the checksum-locked MAT snapshot and recover session boundaries."""

    if not isinstance(manifest, NeuralCodeReverseEngineeringManifest):
        raise TypeError(
            "manifest must be NeuralCodeReverseEngineeringManifest"
        )
    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise RuntimeError(
            "SciPy is required for the Tafazoli classifier snapshot audit"
        ) from error

    path = Path(data_directory) / manifest.classifier_snapshot.filename
    observed_md5 = _md5(path)
    if observed_md5 != manifest.classifier_snapshot.md5:
        raise ValueError("classifier snapshot failed its MD5 checksum")
    payload = loadmat(path, simplify_cells=True)
    try:
        options = payload["ClassifierOpts"]
        time = payload["Time"]
    except KeyError as error:
        raise ValueError(
            f"classifier snapshot is missing variable {error.args[0]!r}"
        ) from error
    return summarize_tafazoli_classifier_snapshot(
        manifest.classifier_snapshot,
        options,
        time,
        observed_md5=observed_md5,
    )


def _published_decoder_metrics(
    processed_report: TafazoliProcessedAuditReport,
) -> tuple[PublishedDecoderMetric, ...]:
    curves = {item.name: item for item in processed_report.decoder_curves}
    missing = sorted(set(_REQUIRED_CROSS_DECODER_NAMES) - set(curves))
    if missing:
        raise ValueError(f"processed report is missing decoder curves: {missing}")
    return tuple(
        PublishedDecoderMetric(
            name=name,
            peak_accuracy=float(curves[name].raw_peak_accuracy),
            post_event_mean_accuracy=float(
                curves[name].raw_post_event_mean_accuracy
            ),
        )
        for name in _REQUIRED_CROSS_DECODER_NAMES
    )


def _published_artifact_passes(
    metrics: tuple[PublishedDecoderMetric, ...],
    thresholds: PublishedReuseThresholds,
) -> bool:
    return all(
        item.peak_accuracy
        >= thresholds.bidirectional_cross_decoder_peak_accuracy_min
        and item.post_event_mean_accuracy
        >= (
            thresholds
            .bidirectional_cross_decoder_post_event_mean_accuracy_min
        )
        for item in metrics
    )


def evaluate_neural_code_reverse_engineering(
    manifest: NeuralCodeReverseEngineeringManifest,
    snapshot_report: TafazoliClassifierSnapshotReport,
    published_decoder_metrics: tuple[PublishedDecoderMetric, ...],
    *,
    processed_artifact_integrity_passed: bool,
) -> NeuralCodeReverseEngineeringReport:
    """Evaluate claim logic from already-extracted, testable evidence."""

    if not isinstance(manifest, NeuralCodeReverseEngineeringManifest):
        raise TypeError(
            "manifest must be NeuralCodeReverseEngineeringManifest"
        )
    if not isinstance(
        snapshot_report,
        TafazoliClassifierSnapshotReport,
    ):
        raise TypeError(
            "snapshot_report must be TafazoliClassifierSnapshotReport"
        )
    if type(processed_artifact_integrity_passed) is not bool:
        raise TypeError("processed_artifact_integrity_passed must be boolean")
    if not processed_artifact_integrity_passed:
        raise ValueError("processed-output integrity audit did not pass")
    if not snapshot_report.checksum_matches:
        raise ValueError("classifier snapshot integrity audit did not pass")
    if snapshot_report.full_pseudopopulation_is_simultaneous:
        raise ValueError("403-column pseudopopulation cannot be simultaneous")
    if snapshot_report.full_neural_language_inverse_problem_possible:
        raise ValueError("snapshot must not unlock the full inverse problem")

    metrics = tuple(published_decoder_metrics)
    if not all(isinstance(item, PublishedDecoderMetric) for item in metrics):
        raise TypeError(
            "published_decoder_metrics must contain PublishedDecoderMetric"
        )
    metric_names = tuple(item.name for item in metrics)
    if set(metric_names) != set(_REQUIRED_CROSS_DECODER_NAMES):
        raise ValueError(
            "published_decoder_metrics must contain the four declared "
            "cross-task curves"
        )
    if len(metric_names) != len(set(metric_names)):
        raise ValueError("published decoder metric names must be unique")
    artifact_pass = _published_artifact_passes(
        metrics,
        manifest.published_reuse_thresholds,
    )
    if not artifact_pass:
        raise ValueError("published cross-task decoder sanity gate failed")

    verdicts = (
        ClaimVerdict(
            key="official_processed_artifacts_reproduced",
            question=(
                "Were the checksum-matched published cross-task artifacts "
                "reproduced?"
            ),
            answer=YES,
            claim_scope="official author-processed figure outputs",
            basis=(
                "the processed figure audit passed",
                "all four declared bidirectional cross-task curves exceed "
                "the preregistered descriptive thresholds",
            ),
        ),
        ClaimVerdict(
            key="task_design_two_slot_code_skeleton_reconstructed",
            question=(
                "Does the observed task table factor into a read selector and "
                "a response-axis router?"
            ),
            answer=YES,
            claim_scope="experimental task design, not neural implementation",
            basis=(
                "S1 = READ_SHAPE then ROUTE_AXIS_1",
                "C1 = READ_COLOR then ROUTE_AXIS_1",
                "C2 = READ_COLOR then ROUTE_AXIS_2",
                "the unique missing cell is S2 = READ_SHAPE then "
                "ROUTE_AXIS_2",
            ),
        ),
        ClaimVerdict(
            key="shared_interface_frontend_backend_candidate_supported",
            question=(
                "Do the processed results support a shared sensory-front-end "
                "and motor-back-end interface candidate?"
            ),
            answer=YES,
            claim_scope=(
                "representational interface skeleton, not a neural call graph"
            ),
            basis=(
                "colour decoding transfers bidirectionally between C1 and C2",
                "response decoding transfers bidirectionally between C1 and S1",
                "the task table routes a selected sensory category to one of "
                "two response axes",
            ),
        ),
        ClaimVerdict(
            key="common_callee_assembly_identified",
            question=(
                "Was a common cell assembly called by distinct neural "
                "front ends identified?"
            ),
            answer=NO,
            claim_scope="latent or biological run-time call structure",
            basis=(
                "cross-decoding identifies shared representations, not a "
                "caller, callee, or call boundary",
                "the session-local common-successor proxy did not beat "
                "time-only, reverse, event-mean, and same-state-count frozen "
                "D1-to-D3 transfer gates together",
                "return paths and causal call boundaries are absent from the "
                "released artifact",
            ),
        ),
        ClaimVerdict(
            key="hierarchical_inheritance_operator_identified",
            question=(
                "Was a parent neural operator with smaller task-specific "
                "child residuals identified?"
            ),
            answer=NO,
            claim_scope="hierarchical parameter-sharing architecture",
            basis=(
                "the state-parent-plus-rank-one proxy was less predictive and "
                "longer than matched stationary VAR",
                "state-level residual sharing is not task inheritance",
                "D1 and D3 rows cannot be paired to construct a justified "
                "cross-task parent and child hierarchy",
            ),
        ),
        ClaimVerdict(
            key="common_callee_or_hierarchy_refuted",
            question=(
                "Was every common-callee or hierarchical neural architecture "
                "refuted?"
            ),
            answer=TEST_UNAVAILABLE,
            claim_scope="absence of front-end dispatch or inheritance structure",
            basis=(
                "the released snapshot is a processed single-area "
                "pseudopopulation",
                "there is no selective caller/callee perturbation, rescue, or "
                "cross-region call-and-return recording in this artifact",
            ),
        ),
        ClaimVerdict(
            key="session_local_operator_pilot_possible",
            question=(
                "Can the saved snapshot support a restricted oracle-free "
                "session-local operator pilot?"
            ),
            answer=YES,
            claim_scope=(
                "dimensions 1 and 3 inside each recovered recording session"
            ),
            basis=(
                f"{len(snapshot_report.session_groups)} session groups were "
                "recovered from sampled-trial signatures",
                "the primary pilot forbids 403-wide latent fitting",
                "dimension 2 is excluded from primary discovery",
            ),
        ),
        ClaimVerdict(
            key="shared_population_transition_primitive_identified",
            question=(
                "Has a reusable population transition primitive been "
                "identified from these neural data?"
            ),
            answer=NO,
            claim_scope="strict neural-code identification in this dataset",
            basis=(
                "published cross-decoding supports a representation candidate "
                "but is not blind operator discovery",
                "session-local stationary and past-gated switching probes did "
                "not pass their reverse and event-time controls",
                "source-frozen D1-to-D3 switching transfer did not beat "
                "target refitting",
                "matched stationary VAR beat the tested switching and "
                "state-parent residual proxies",
            ),
        ),
        ClaimVerdict(
            key="fixed_neuron_opcode_identified",
            question="Was a fixed-neuron opcode dictionary identified?",
            answer=NO,
            claim_scope="this released processed dataset",
            basis=(
                "the 403 columns are a stitched pseudopopulation",
                "neuron identity is not tracked across recording sessions",
                "the released target labels were defined before this audit",
            ),
        ),
        ClaimVerdict(
            key="fixed_neuron_opcode_refuted",
            question="Was every fixed-neuron opcode account refuted?",
            answer=TEST_UNAVAILABLE,
            claim_scope="ontological rejection of a code family",
            basis=(
                "the required cross-session neuron identity is unavailable",
                "no matched fixed-opcode model comparison was run",
            ),
        ),
        ClaimVerdict(
            key="continuous_dynamics_ruled_out",
            question=(
                "Was a same-capacity continuous-dynamics account ruled out?"
            ),
            answer=NO,
            claim_scope="three-family model comparison",
            basis=(
                "matched stationary VAR is the relative winner over the tested "
                "switching and state-parent residual proxies",
                "one linear VAR comparison does not exhaust the wider "
                "continuous nonlinear family",
                "no complete fixed-opcode, compositional-program, and "
                "continuous-dynamics comparison is possible in this artifact",
            ),
        ),
        ClaimVerdict(
            key="neural_language_architecture_type_identified",
            question=(
                "Was a global language versus regional local-DSL architecture "
                "identified?"
            ),
            answer=NO,
            claim_scope="architecture identification in the current artifact",
            basis=(
                "the released local artifact is one processed LPFC "
                "pseudopopulation",
                "simultaneous multi-area raw trials and source-frozen "
                "communication interfaces are unavailable",
            ),
        ),
        ClaimVerdict(
            key="optimizer_mechanism_identified",
            question=(
                "Was the mechanism that optimizes the candidate code "
                "identified?"
            ),
            answer=NO,
            claim_scope="learning and consolidation mechanism identification",
            basis=(
                "the snapshot has no novice-to-expert learning trajectory",
                "synaptic, inhibitory-cell, neuromodulator, and sleep-replay "
                "measurements are unavailable",
            ),
        ),
        ClaimVerdict(
            key="monotonic_more_regions_worse_supported",
            question=(
                "Does the current evidence support a universal rule that "
                "recruiting more brain regions monotonically reduces function?"
            ),
            answer=NO,
            claim_scope="support for a universal monotonic recruitment claim",
            basis=(
                "region count is distinct from effective communication rank, "
                "representational overlap, shared noise, energy, and delay",
                "the current artifact contains no simultaneous multi-area "
                "recruitment manipulation",
            ),
        ),
        ClaimVerdict(
            key="minimal_sufficient_multi_area_circuit_test_available",
            question=(
                "Can this artifact locate a minimum sufficient multi-area "
                "circuit and interface width?"
            ),
            answer=TEST_UNAVAILABLE,
            claim_scope="multi-area recruitment and communication-cost curve",
            basis=(
                "the 403 columns are not a simultaneous multi-area population",
                "no selective recruitment, matched activity-cost, or "
                "interface-width intervention is present",
            ),
        ),
        ClaimVerdict(
            key="unseen_composition_validated",
            question="Was the predicted missing S2 composition validated?",
            answer=TEST_UNAVAILABLE,
            claim_scope="held-out factorial neural and behavioural prediction",
            basis=(
                "the recorded task grid contains S1, C1, and C2 but not S2",
                "published cross-decoding is not a substitute for a missing "
                "factorial cell",
            ),
        ),
        ClaimVerdict(
            key="causal_instruction_set_validated",
            question=(
                "Was a neural instruction selectively perturbed and rescued?"
            ),
            answer=TEST_UNAVAILABLE,
            claim_scope="causal instruction semantics",
            basis=(
                "the released artifacts contain no targeted perturbation and "
                "interface-level rescue experiment",
            ),
        ),
        ClaimVerdict(
            key="brain_programming_language_identified",
            question="Was a brain programming language identified?",
            answer=NO,
            claim_scope="identification by the current code and data",
            basis=(
                "no blind reusable operator library has been recovered",
                "unseen composition and causal semantics are not tested",
                "a task-design factorization is not a neural grammar",
            ),
        ),
        ClaimVerdict(
            key="brain_programming_language_exists",
            question="Does any programming language exist anywhere in a brain?",
            answer=TEST_UNAVAILABLE,
            claim_scope="whole-brain existence claim",
            basis=(
                "a processed LPFC task dataset cannot decide a universal "
                "existence or non-existence claim",
            ),
        ),
    )
    return NeuralCodeReverseEngineeringReport(
        schema_version=REPORT_SCHEMA_VERSION,
        scope=manifest.scope,
        method_status=CODE_SKELETON_ONLY_STATUS,
        task_programs=manifest.task_grid.observed_tasks,
        missing_composition_prediction=(
            manifest.task_grid.predicted_missing_task
        ),
        snapshot=snapshot_report,
        published_decoder_metrics=metrics,
        published_cross_task_decoder_artifact_pass=artifact_pass,
        competing_family_winner="nonidentifiable",
        claim_verdicts=verdicts,
        next_decisive_experiment=(
            "record raw trials simultaneously across candidate sensory, "
            "association, action-selection, and motor regions",
            "compare a global monolithic latent, independent regional models, "
            "and regional local DSLs joined by narrow communication subspaces",
            "increase the number of recruited regions under matched predictive, "
            "parameter, synchronization, and activity-cost budgets to locate "
            "the minimum sufficient circuit",
            "separate local plasticity and inhibitory competition, delayed "
            "reward or dopamine eligibility, and sleep or replay consolidation "
            "as competing optimizer timescales",
            "require source-frozen communication interfaces to transfer across "
            "sessions, tasks, Chico, and Silas",
            "break every candidate with time reversal, whole-trial shuffle, "
            "event-time-only, session-boundary, and nuisance-only controls",
            "record S2 and a counterbalanced second missing composition, then "
            "predict their full neural trajectory and behaviour without "
            "using either cell for alignment or model selection",
            "selectively perturb the inferred primitive and rescue its "
            "predicted downstream interface before using causal language",
        ),
        excluded_inferences=manifest.excluded_inferences,
        conclusion=(
            "The experimental task table yields a clear two-slot candidate "
            "program and predicts the missing S2 composition. The official "
            "processed artifacts support a shared sensory-front-end and "
            "motor-back-end interface candidate. They do not identify a "
            "state-dependent switching operator, common callee assembly, "
            "inheritance tree, neural operator library, fixed-neuron opcode, "
            "global-versus-local language architecture, optimizer, unseen "
            "composition, causal instruction set, or brain programming "
            "language."
        ),
    )


def build_neural_code_reverse_engineering_report(
    manifest: NeuralCodeReverseEngineeringManifest,
    processed_report: TafazoliProcessedAuditReport,
    snapshot_report: TafazoliClassifierSnapshotReport,
) -> NeuralCodeReverseEngineeringReport:
    """Combine the real processed-data audits into claim-local verdicts."""

    if not isinstance(processed_report, TafazoliProcessedAuditReport):
        raise TypeError("processed_report must be TafazoliProcessedAuditReport")
    return evaluate_neural_code_reverse_engineering(
        manifest,
        snapshot_report,
        _published_decoder_metrics(processed_report),
        processed_artifact_integrity_passed=(
            processed_report.method_status
            == PROCESSED_NEURAL_FIGURE_AUDIT_PASS
        ),
    )


def verify_report_internal_consistency(
    report: NeuralCodeReverseEngineeringReport,
) -> None:
    """Raise when a report silently upgrades a locked scientific claim."""

    if report.method_status != CODE_SKELETON_ONLY_STATUS:
        raise ValueError("unexpected reverse-engineering status")
    expected = {
        "official_processed_artifacts_reproduced": YES,
        "task_design_two_slot_code_skeleton_reconstructed": YES,
        "shared_interface_frontend_backend_candidate_supported": YES,
        "common_callee_assembly_identified": NO,
        "hierarchical_inheritance_operator_identified": NO,
        "common_callee_or_hierarchy_refuted": TEST_UNAVAILABLE,
        "session_local_operator_pilot_possible": YES,
        "shared_population_transition_primitive_identified": NO,
        "fixed_neuron_opcode_identified": NO,
        "fixed_neuron_opcode_refuted": TEST_UNAVAILABLE,
        "continuous_dynamics_ruled_out": NO,
        "neural_language_architecture_type_identified": NO,
        "optimizer_mechanism_identified": NO,
        "monotonic_more_regions_worse_supported": NO,
        "minimal_sufficient_multi_area_circuit_test_available": (
            TEST_UNAVAILABLE
        ),
        "unseen_composition_validated": TEST_UNAVAILABLE,
        "causal_instruction_set_validated": TEST_UNAVAILABLE,
        "brain_programming_language_identified": NO,
        "brain_programming_language_exists": TEST_UNAVAILABLE,
    }
    observed = {item.key: item.answer for item in report.claim_verdicts}
    if observed != expected:
        raise ValueError("claim verdicts do not match the locked boundary")
    if report.competing_family_winner != "nonidentifiable":
        raise ValueError("current data cannot select a competing family")
    if report.missing_composition_prediction.observed:
        raise ValueError("missing composition cannot be marked observed")
    if report.snapshot.full_pseudopopulation_is_simultaneous:
        raise ValueError("pseudopopulation cannot be marked simultaneous")
    if not isclose(
        report.snapshot.adjacent_window_overlap_fraction,
        0.9,
        abs_tol=1e-12,
    ):
        raise ValueError("unexpected temporal-window overlap")
