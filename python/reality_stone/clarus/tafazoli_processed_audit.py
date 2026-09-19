"""Narrow audit of official processed Tafazoli et al. figure outputs.

The input files are author-produced MATLAB outputs for published figures.  The
audit verifies their checksums, extracts declared classifier curves and
dynamic-correlation matrices, and reports descriptive values.  It deliberately
does not call this a raw-spike reanalysis, an independent replication, an
assembly-discovery result, or evidence of a neural programming language.

SciPy is imported only by :func:`run_tafazoli_processed_audit`, so the pure
summary functions remain usable with the base NumPy dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "clarus-tafazoli-processed-figure-audit/v1"
REPORT_SCHEMA_VERSION = "clarus-tafazoli-processed-figure-audit-report/v1"
PROCESSED_FIGURE_AUDIT_SCOPE = (
    "official_processed_neural_figure_outputs_integrity_and_descriptive_"
    "audit_only"
)
PROCESSED_NEURAL_FIGURE_AUDIT_PASS = (
    "PROCESSED_NEURAL_FIGURE_AUDIT_PASS"
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "scope",
        "description",
        "source",
        "required_files",
        "expected_shapes",
        "decoder_curves",
        "dynamic_correlations",
        "claim_locks",
        "excluded_inferences",
    }
)
_SOURCE_KEYS = frozenset(
    {"article_url", "data_doi", "code_url", "license"}
)
_FILE_KEYS = frozenset({"filename", "md5"})
_SHAPE_KEYS = frozenset(
    {
        "classifier_resample_count",
        "classifier_timepoint_count",
        "dynamic_timepoint_count",
        "decoder_moving_mean_width",
    }
)
_DECODER_KEYS = frozenset(
    {
        "name",
        "figure_panel",
        "feature",
        "train_task",
        "test_task",
        "trip_index_zero_based",
        "curve_index_zero_based",
        "stat_dimension_one_based",
        "stat_condition_one_based",
    }
)
_DYNAMIC_KEYS = frozenset(
    {
        "name",
        "figure_panel",
        "correlation_variable",
        "pvalue_variable",
    }
)
_CLAIM_LOCK_KEYS = frozenset(
    {
        "raw_trial_or_spike_reanalysis",
        "classifier_refit",
        "independent_replication",
        "independent_statistical_reanalysis",
        "transfer_entropy_validated",
        "causal_information_flow_validated",
        "unseen_composition_tested",
        "neural_clarus_assembly_validated",
        "causal_instruction_set_validated",
        "full_brain_language_identified",
    }
)


@dataclass(frozen=True)
class ProcessedSource:
    """Primary article, data, code, and license identifiers."""

    article_url: str
    data_doi: str
    code_url: str
    license: str


@dataclass(frozen=True)
class RequiredFile:
    """Checksum-locked processed input."""

    filename: str
    md5: str


@dataclass(frozen=True)
class ExpectedShapes:
    """Expected dimensions of the official processed arrays."""

    classifier_resample_count: int
    classifier_timepoint_count: int
    dynamic_timepoint_count: int
    decoder_moving_mean_width: int


@dataclass(frozen=True)
class DecoderCurveSpec:
    """Manifest mapping from a paper panel to one MATLAB cell."""

    name: str
    figure_panel: str
    feature: str
    train_task: str
    test_task: str
    trip_index_zero_based: int
    curve_index_zero_based: int
    stat_dimension_one_based: int
    stat_condition_one_based: int


@dataclass(frozen=True)
class DynamicCorrelationSpec:
    """Manifest mapping for one published dynamic-correlation matrix."""

    name: str
    figure_panel: str
    correlation_variable: str
    pvalue_variable: str


@dataclass(frozen=True)
class ProcessedClaimLocks:
    """Scientific claims that this processed-output audit cannot unlock."""

    raw_trial_or_spike_reanalysis: bool
    classifier_refit: bool
    independent_replication: bool
    independent_statistical_reanalysis: bool
    transfer_entropy_validated: bool
    causal_information_flow_validated: bool
    unseen_composition_tested: bool
    neural_clarus_assembly_validated: bool
    causal_instruction_set_validated: bool
    full_brain_language_identified: bool


@dataclass(frozen=True)
class TafazoliProcessedAuditManifest:
    """Strict, checksum-locked extraction declaration."""

    schema_version: str
    scope: str
    description: str
    source: ProcessedSource
    required_files: tuple[RequiredFile, ...]
    expected_shapes: ExpectedShapes
    decoder_curves: tuple[DecoderCurveSpec, ...]
    dynamic_correlations: tuple[DynamicCorrelationSpec, ...]
    claim_locks: ProcessedClaimLocks
    excluded_inferences: tuple[str, ...]


@dataclass(frozen=True)
class SourceFileAudit:
    """Observed checksum and size for a required input."""

    filename: str
    expected_md5: str
    observed_md5: str
    byte_count: int
    checksum_matches: bool


@dataclass(frozen=True)
class DecoderCurveAudit:
    """Descriptive extraction from one author-produced classifier curve."""

    name: str
    figure_panel: str
    feature: str
    train_task: str
    test_task: str
    classifier_resample_count: int
    timepoint_count: int
    time_start_seconds: float
    time_end_seconds: float
    moving_mean_width: int
    raw_peak_accuracy: float
    raw_peak_time_seconds: float
    raw_full_window_mean_accuracy: float
    raw_post_event_mean_accuracy: float
    plotted_smoothed_peak_accuracy: float
    plotted_smoothed_peak_time_seconds: float
    plotted_smoothed_post_event_mean_accuracy: float
    author_cluster_index_start_one_based: int | None
    author_cluster_index_end_one_based: int | None
    author_cluster_time_start_seconds: float | None
    author_cluster_time_end_seconds: float | None
    author_cluster_minimum_reported_p: float | None


@dataclass(frozen=True)
class DynamicCorrelationAudit:
    """Descriptive extraction from one author-produced correlation matrix."""

    name: str
    figure_panel: str
    timepoint_count: int
    declared_window_start_seconds: float
    declared_window_end_seconds: float
    grid_window_start_seconds: float
    grid_window_end_seconds: float
    grid_window_timepoint_count: int
    window_mean_correlation: float
    window_positive_value_mean: float
    pointwise_p_below_0_05_fraction: float
    pointwise_p_below_0_001_fraction: float
    positive_diagonal_projection_peak_lag_seconds: float
    positive_diagonal_projection_peak_value: float
    positive_diagonal_projection_weighted_lag_seconds: float
    zero_lag_positive_diagonal_mean: float
    window_source: str
    matrix_axis_convention: str
    pearson_decoder_score_correlation: bool
    transfer_entropy_computed: bool
    positive_diagonal_projection_uses_significance_mask: bool
    global_analysisopts_shadowing_warning: bool


@dataclass(frozen=True)
class TafazoliProcessedAuditReport:
    """Serializable audit with strict claim locks."""

    schema_version: str
    scope: str
    method_status: str
    source: ProcessedSource
    source_files: tuple[SourceFileAudit, ...]
    decoder_curves: tuple[DecoderCurveAudit, ...]
    dynamic_correlations: tuple[DynamicCorrelationAudit, ...]
    processed_neural_figure_artifact_used: bool
    cross_task_decoder_artifact_reproduced: bool
    dynamic_lag_projection_artifact_reproduced: bool
    raw_trial_or_spike_reanalysis: bool
    classifier_refit: bool
    independent_replication: bool
    independent_statistical_reanalysis: bool
    transfer_entropy_validated: bool
    causal_information_flow_validated: bool
    unseen_composition_tested: bool
    neural_clarus_assembly_validated: bool
    causal_instruction_set_validated: bool
    full_brain_language_identified: bool
    excluded_inferences: tuple[str, ...]
    limitations: tuple[str, ...]
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

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


def _strict_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a boolean")
    return value


def _strict_string_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{label} must be a non-empty JSON array")
    result = tuple(
        _strict_string(item, label=f"{label} item") for item in value
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _strict_object_array(
    value: Any,
    *,
    label: str,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{label} must be a non-empty JSON array")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"{label}[{index}] must be a JSON object")
        result.append(item)
    return tuple(result)


def load_tafazoli_processed_audit_manifest(
    path: str | Path,
) -> TafazoliProcessedAuditManifest:
    """Strictly load the processed-output extraction manifest."""

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
    if scope != PROCESSED_FIGURE_AUDIT_SCOPE:
        raise ValueError(
            f"scope must equal {PROCESSED_FIGURE_AUDIT_SCOPE!r}"
        )

    source_raw = _require_exact_keys(
        top["source"],
        required=_SOURCE_KEYS,
        label="source",
    )
    source = ProcessedSource(
        **{
            key: _strict_string(source_raw[key], label=f"source.{key}")
            for key in sorted(_SOURCE_KEYS)
        }
    )

    required_files = []
    for index, raw in enumerate(
        _strict_object_array(top["required_files"], label="required_files")
    ):
        item = _require_exact_keys(
            raw,
            required=_FILE_KEYS,
            label=f"required_files[{index}]",
        )
        filename = _strict_string(
            item["filename"],
            label=f"required_files[{index}].filename",
        )
        md5 = _strict_string(
            item["md5"],
            label=f"required_files[{index}].md5",
        ).lower()
        if len(md5) != 32 or any(
            character not in "0123456789abcdef" for character in md5
        ):
            raise ValueError(
                f"required_files[{index}].md5 must be 32 lowercase hex digits"
            )
        required_files.append(RequiredFile(filename=filename, md5=md5))
    if len({item.filename for item in required_files}) != len(required_files):
        raise ValueError("required_files filenames must be unique")

    shapes_raw = _require_exact_keys(
        top["expected_shapes"],
        required=_SHAPE_KEYS,
        label="expected_shapes",
    )
    expected_shapes = ExpectedShapes(
        **{
            key: _strict_int(
                shapes_raw[key],
                label=f"expected_shapes.{key}",
            )
            for key in sorted(_SHAPE_KEYS)
        }
    )
    if min(asdict(expected_shapes).values()) <= 0:
        raise ValueError("every expected shape must be positive")

    decoder_specs = []
    for index, raw in enumerate(
        _strict_object_array(top["decoder_curves"], label="decoder_curves")
    ):
        item = _require_exact_keys(
            raw,
            required=_DECODER_KEYS,
            label=f"decoder_curves[{index}]",
        )
        decoder_specs.append(
            DecoderCurveSpec(
                name=_strict_string(
                    item["name"],
                    label=f"decoder_curves[{index}].name",
                ),
                figure_panel=_strict_string(
                    item["figure_panel"],
                    label=f"decoder_curves[{index}].figure_panel",
                ),
                feature=_strict_string(
                    item["feature"],
                    label=f"decoder_curves[{index}].feature",
                ),
                train_task=_strict_string(
                    item["train_task"],
                    label=f"decoder_curves[{index}].train_task",
                ),
                test_task=_strict_string(
                    item["test_task"],
                    label=f"decoder_curves[{index}].test_task",
                ),
                trip_index_zero_based=_strict_int(
                    item["trip_index_zero_based"],
                    label=(
                        f"decoder_curves[{index}].trip_index_zero_based"
                    ),
                ),
                curve_index_zero_based=_strict_int(
                    item["curve_index_zero_based"],
                    label=(
                        f"decoder_curves[{index}].curve_index_zero_based"
                    ),
                ),
                stat_dimension_one_based=_strict_int(
                    item["stat_dimension_one_based"],
                    label=(
                        f"decoder_curves[{index}]."
                        "stat_dimension_one_based"
                    ),
                ),
                stat_condition_one_based=_strict_int(
                    item["stat_condition_one_based"],
                    label=(
                        f"decoder_curves[{index}]."
                        "stat_condition_one_based"
                    ),
                ),
            )
        )
    if len({item.name for item in decoder_specs}) != len(decoder_specs):
        raise ValueError("decoder curve names must be unique")
    for item in decoder_specs:
        if min(
            item.trip_index_zero_based,
            item.curve_index_zero_based,
        ) < 0:
            raise ValueError("decoder zero-based indices must be non-negative")
        if min(
            item.stat_dimension_one_based,
            item.stat_condition_one_based,
        ) < 1:
            raise ValueError("decoder stat indices must be one-based positive")

    dynamic_specs = []
    for index, raw in enumerate(
        _strict_object_array(
            top["dynamic_correlations"],
            label="dynamic_correlations",
        )
    ):
        item = _require_exact_keys(
            raw,
            required=_DYNAMIC_KEYS,
            label=f"dynamic_correlations[{index}]",
        )
        dynamic_specs.append(
            DynamicCorrelationSpec(
                **{
                    key: _strict_string(
                        item[key],
                        label=f"dynamic_correlations[{index}].{key}",
                    )
                    for key in sorted(_DYNAMIC_KEYS)
                }
            )
        )
    if len({item.name for item in dynamic_specs}) != len(dynamic_specs):
        raise ValueError("dynamic correlation names must be unique")

    locks_raw = _require_exact_keys(
        top["claim_locks"],
        required=_CLAIM_LOCK_KEYS,
        label="claim_locks",
    )
    locks = {
        key: _strict_bool(
            locks_raw[key],
            label=f"claim_locks.{key}",
        )
        for key in sorted(_CLAIM_LOCK_KEYS)
    }
    if any(locks.values()):
        raise ValueError("every processed-audit claim lock must remain false")

    return TafazoliProcessedAuditManifest(
        schema_version=schema_version,
        scope=scope,
        description=_strict_string(top["description"], label="description"),
        source=source,
        required_files=tuple(required_files),
        expected_shapes=expected_shapes,
        decoder_curves=tuple(decoder_specs),
        dynamic_correlations=tuple(dynamic_specs),
        claim_locks=ProcessedClaimLocks(**locks),
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


def audit_required_files(
    data_directory: str | Path,
    required_files: tuple[RequiredFile, ...],
) -> tuple[SourceFileAudit, ...]:
    """Verify exact processed inputs without downloading or modifying data."""

    root = Path(data_directory)
    audits = []
    for spec in required_files:
        path = root / spec.filename
        if not path.is_file():
            raise FileNotFoundError(f"required processed file not found: {path}")
        observed = _md5(path)
        audits.append(
            SourceFileAudit(
                filename=spec.filename,
                expected_md5=spec.md5,
                observed_md5=observed,
                byte_count=path.stat().st_size,
                checksum_matches=observed == spec.md5,
            )
        )
    return tuple(audits)


def _as_finite_vector(value: Any, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if not vector.size or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be a non-empty finite vector")
    return vector


def _flatten_numeric(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != object:
        return np.asarray(array, dtype=np.float64).reshape(-1)
    flattened = [
        _flatten_numeric(item)
        for item in array.reshape(-1)
        if np.asarray(item).size
    ]
    if not flattened:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(flattened)


def _centered_moving_mean(values: np.ndarray, width: int) -> np.ndarray:
    """Match MATLAB ``movmean`` endpoint shrinking for an odd window."""

    if width < 1 or width % 2 == 0:
        raise ValueError("moving-mean width must be a positive odd integer")
    half_width = width // 2
    return np.asarray(
        [
            values[
                max(0, index - half_width) : min(
                    values.size,
                    index + half_width + 1,
                )
            ].mean()
            for index in range(values.size)
        ],
        dtype=np.float64,
    )


def summarize_decoder_curve(
    spec: DecoderCurveSpec,
    time_seconds: Any,
    classifier_resamples: Any,
    *,
    expected_resample_count: int,
    expected_timepoint_count: int,
    moving_mean_width: int,
    author_cluster_indices_one_based: Any,
    author_cluster_reported_p: Any,
) -> DecoderCurveAudit:
    """Summarize one processed classifier curve without inferential reuse."""

    time = _as_finite_vector(time_seconds, label=f"{spec.name} time")
    if time.size != expected_timepoint_count:
        raise ValueError(
            f"{spec.name} timepoint count must equal "
            f"{expected_timepoint_count}"
        )
    if not np.all(np.diff(time) > 0.0):
        raise ValueError(f"{spec.name} time must be strictly increasing")

    values = np.asarray(classifier_resamples, dtype=np.float64)
    expected_shape = (expected_resample_count, expected_timepoint_count)
    if values.shape != expected_shape:
        raise ValueError(
            f"{spec.name} classifier array must have shape {expected_shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{spec.name} classifier values must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{spec.name} classifier values must be in [0, 1]")

    mean_trace = values.mean(axis=0)
    raw_peak_index = int(np.argmax(mean_trace))
    smoothed_trace = _centered_moving_mean(
        mean_trace,
        moving_mean_width,
    )
    smoothed_peak_index = int(np.argmax(smoothed_trace))
    post_event = time >= 0.0
    if not np.any(post_event):
        raise ValueError(f"{spec.name} time axis must include nonnegative time")

    cluster_indices_float = _flatten_numeric(
        author_cluster_indices_one_based
    )
    cluster_p = _flatten_numeric(author_cluster_reported_p)
    if cluster_indices_float.size:
        rounded = np.rint(cluster_indices_float)
        if not np.allclose(cluster_indices_float, rounded):
            raise ValueError(f"{spec.name} cluster indices must be integral")
        cluster_indices = rounded.astype(np.int64)
        if np.any(
            (cluster_indices < 1)
            | (cluster_indices > expected_timepoint_count)
        ):
            raise ValueError(f"{spec.name} cluster indices are out of range")
        start_index = int(cluster_indices.min())
        end_index = int(cluster_indices.max())
        start_time = float(time[start_index - 1])
        end_time = float(time[end_index - 1])
    else:
        start_index = None
        end_index = None
        start_time = None
        end_time = None

    if cluster_p.size:
        if not np.all(np.isfinite(cluster_p)):
            raise ValueError(f"{spec.name} reported p values must be finite")
        if np.any((cluster_p < 0.0) | (cluster_p > 1.0)):
            raise ValueError(f"{spec.name} reported p values must be in [0, 1]")
        minimum_p = float(cluster_p.min())
    else:
        minimum_p = None

    return DecoderCurveAudit(
        name=spec.name,
        figure_panel=spec.figure_panel,
        feature=spec.feature,
        train_task=spec.train_task,
        test_task=spec.test_task,
        classifier_resample_count=values.shape[0],
        timepoint_count=time.size,
        time_start_seconds=float(time[0]),
        time_end_seconds=float(time[-1]),
        moving_mean_width=moving_mean_width,
        raw_peak_accuracy=float(mean_trace[raw_peak_index]),
        raw_peak_time_seconds=float(time[raw_peak_index]),
        raw_full_window_mean_accuracy=float(values.mean()),
        raw_post_event_mean_accuracy=float(mean_trace[post_event].mean()),
        plotted_smoothed_peak_accuracy=float(
            smoothed_trace[smoothed_peak_index]
        ),
        plotted_smoothed_peak_time_seconds=float(
            time[smoothed_peak_index]
        ),
        plotted_smoothed_post_event_mean_accuracy=float(
            smoothed_trace[post_event].mean()
        ),
        author_cluster_index_start_one_based=start_index,
        author_cluster_index_end_one_based=end_index,
        author_cluster_time_start_seconds=start_time,
        author_cluster_time_end_seconds=end_time,
        author_cluster_minimum_reported_p=minimum_p,
    )


def summarize_dynamic_correlation(
    spec: DynamicCorrelationSpec,
    time_seconds: Any,
    correlation: Any,
    pointwise_p: Any,
    *,
    expected_timepoint_count: int,
    declared_window_start_seconds: float,
    declared_window_end_seconds: float,
    bin_shift_seconds: float,
) -> DynamicCorrelationAudit:
    """Summarize a processed dynamic-correlation matrix."""

    time = _as_finite_vector(time_seconds, label=f"{spec.name} time")
    if time.size != expected_timepoint_count:
        raise ValueError(
            f"{spec.name} timepoint count must equal "
            f"{expected_timepoint_count}"
        )
    if not np.all(np.diff(time) > 0.0):
        raise ValueError(f"{spec.name} time must be strictly increasing")

    matrix = np.asarray(correlation, dtype=np.float64)
    p_values = np.asarray(pointwise_p, dtype=np.float64)
    expected_shape = (expected_timepoint_count, expected_timepoint_count)
    if matrix.shape != expected_shape or p_values.shape != expected_shape:
        raise ValueError(
            f"{spec.name} matrices must both have shape {expected_shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{spec.name} correlations must be finite")
    if not np.all(np.isfinite(p_values)):
        raise ValueError(f"{spec.name} pointwise p values must be finite")
    if np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError(f"{spec.name} pointwise p values must be in [0, 1]")
    for label, value in (
        ("declared window start", declared_window_start_seconds),
        ("declared window end", declared_window_end_seconds),
        ("bin shift", bin_shift_seconds),
    ):
        if not isinstance(value, Real) or not isfinite(float(value)):
            raise TypeError(f"{spec.name} {label} must be finite")
    if declared_window_start_seconds >= declared_window_end_seconds:
        raise ValueError(f"{spec.name} declared window must increase")
    if bin_shift_seconds <= 0.0:
        raise ValueError(f"{spec.name} bin shift must be positive")

    window = (time >= declared_window_start_seconds) & (
        time <= declared_window_end_seconds
    )
    if np.count_nonzero(window) < 2:
        raise ValueError(f"{spec.name} declared window is empty")
    window_matrix = matrix[np.ix_(window, window)]
    window_p = p_values[np.ix_(window, window)]
    positive_values = window_matrix[window_matrix > 0.0]
    positive_mean = (
        float(positive_values.mean()) if positive_values.size else 0.0
    )

    size = window_matrix.shape[0]
    offsets = np.arange(-(size - 1), size, dtype=np.int64)
    positive_projection = []
    for offset in offsets:
        diagonal = np.diag(window_matrix, int(offset))
        positive = diagonal[diagonal > 0.0]
        positive_projection.append(
            float(positive.mean()) if positive.size else 0.0
        )
    projection = np.asarray(positive_projection, dtype=np.float64)
    lags = offsets.astype(np.float64) * float(bin_shift_seconds)
    peak_index = int(np.argmax(projection))
    zero_index = int(np.flatnonzero(offsets == 0)[0])
    projection_mass = float(projection.sum())
    weighted_lag = (
        float(np.dot(lags, projection) / projection_mass)
        if projection_mass > 0.0
        else 0.0
    )
    grid_time = time[window]

    return DynamicCorrelationAudit(
        name=spec.name,
        figure_panel=spec.figure_panel,
        timepoint_count=time.size,
        declared_window_start_seconds=float(
            declared_window_start_seconds
        ),
        declared_window_end_seconds=float(declared_window_end_seconds),
        grid_window_start_seconds=float(grid_time[0]),
        grid_window_end_seconds=float(grid_time[-1]),
        grid_window_timepoint_count=int(grid_time.size),
        window_mean_correlation=float(window_matrix.mean()),
        window_positive_value_mean=positive_mean,
        pointwise_p_below_0_05_fraction=float(np.mean(window_p < 0.05)),
        pointwise_p_below_0_001_fraction=float(
            np.mean(window_p < 0.001)
        ),
        positive_diagonal_projection_peak_lag_seconds=float(
            lags[peak_index]
        ),
        positive_diagonal_projection_peak_value=float(
            projection[peak_index]
        ),
        positive_diagonal_projection_weighted_lag_seconds=weighted_lag,
        zero_lag_positive_diagonal_mean=float(projection[zero_index]),
        window_source="embedded_mat_artifact",
        matrix_axis_convention=(
            "rows=response_time; columns=color_time; "
            "negative_lag=color_leads_response"
        ),
        pearson_decoder_score_correlation=True,
        transfer_entropy_computed=False,
        positive_diagonal_projection_uses_significance_mask=False,
        global_analysisopts_shadowing_warning=True,
    )


def _mat_struct_attribute(value: Any, name: str, *, label: str) -> Any:
    try:
        return getattr(value, name)
    except AttributeError as error:
        raise ValueError(f"{label} is missing MATLAB field {name!r}") from error


def run_tafazoli_processed_audit(
    manifest: TafazoliProcessedAuditManifest,
    data_directory: str | Path,
) -> TafazoliProcessedAuditReport:
    """Load checksum-matched MATLAB outputs and produce the narrow audit."""

    if not isinstance(manifest, TafazoliProcessedAuditManifest):
        raise TypeError(
            "manifest must be TafazoliProcessedAuditManifest"
        )
    if any(asdict(manifest.claim_locks).values()):
        raise ValueError("every processed-audit claim lock must remain false")

    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise RuntimeError(
            "SciPy is required for the MATLAB processed-output audit; "
            "install reality_stone[science]"
        ) from error

    root = Path(data_directory)
    source_files = audit_required_files(root, manifest.required_files)
    if not all(item.checksum_matches for item in source_files):
        raise ValueError("a required processed file failed its MD5 checksum")

    classifier_path = root / "PFC_ClassifierData.mat"
    dynamic_path = root / "DynamicTransformationData.mat"
    classifier = loadmat(
        classifier_path,
        squeeze_me=True,
        struct_as_record=False,
    )
    dynamic = loadmat(
        dynamic_path,
        squeeze_me=True,
        struct_as_record=False,
    )

    try:
        classifier_time = classifier["Time"]
        metric_cells = np.atleast_1d(
            classifier["MetricValsOrg_SuperImposed"]
        )
        stat_dimensions = np.atleast_1d(classifier["StatTest"])
    except KeyError as error:
        raise ValueError(
            f"classifier file is missing variable {error.args[0]!r}"
        ) from error

    decoder_audits = []
    for spec in manifest.decoder_curves:
        try:
            trip = np.atleast_1d(
                metric_cells[spec.trip_index_zero_based]
            )
            values = trip[spec.curve_index_zero_based]
            dimension = np.atleast_1d(
                stat_dimensions[spec.stat_dimension_one_based - 1]
            )
            statistic = dimension[spec.stat_condition_one_based - 1]
        except IndexError as error:
            raise ValueError(
                f"{spec.name} manifest index is outside the MATLAB cells"
            ) from error
        accuracy = _mat_struct_attribute(
            statistic,
            "Accuracy",
            label=f"{spec.name} statistic",
        )
        clusters = _mat_struct_attribute(
            accuracy,
            "clusters",
            label=f"{spec.name} Accuracy",
        )
        reported_p = _mat_struct_attribute(
            accuracy,
            "statsummery",
            label=f"{spec.name} Accuracy",
        )
        decoder_audits.append(
            summarize_decoder_curve(
                spec,
                classifier_time,
                values,
                expected_resample_count=(
                    manifest.expected_shapes.classifier_resample_count
                ),
                expected_timepoint_count=(
                    manifest.expected_shapes.classifier_timepoint_count
                ),
                moving_mean_width=(
                    manifest.expected_shapes.decoder_moving_mean_width
                ),
                author_cluster_indices_one_based=clusters,
                author_cluster_reported_p=reported_p,
            )
        )

    try:
        analysis_options = dynamic["AnalysisOpts"]
    except KeyError as error:
        raise ValueError(
            "dynamic file is missing variable 'AnalysisOpts'"
        ) from error
    dynamic_time = _mat_struct_attribute(
        analysis_options,
        "Time",
        label="dynamic AnalysisOpts",
    )
    window_start = float(
        _mat_struct_attribute(
            analysis_options,
            "ThisTimeAxisStart",
            label="dynamic AnalysisOpts",
        )
    )
    window_end = float(
        _mat_struct_attribute(
            analysis_options,
            "ThisTimeAxisEnd",
            label="dynamic AnalysisOpts",
        )
    )
    spike_options = _mat_struct_attribute(
        analysis_options,
        "SpkParams",
        label="dynamic AnalysisOpts",
    )
    bin_shift = float(
        _mat_struct_attribute(
            spike_options,
            "PSTH_BinShift",
            label="dynamic AnalysisOpts.SpkParams",
        )
    )

    dynamic_audits = []
    for spec in manifest.dynamic_correlations:
        try:
            correlation = dynamic[spec.correlation_variable]
            pointwise_p = dynamic[spec.pvalue_variable]
        except KeyError as error:
            raise ValueError(
                f"dynamic file is missing variable {error.args[0]!r}"
            ) from error
        dynamic_audits.append(
            summarize_dynamic_correlation(
                spec,
                dynamic_time,
                correlation,
                pointwise_p,
                expected_timepoint_count=(
                    manifest.expected_shapes.dynamic_timepoint_count
                ),
                declared_window_start_seconds=window_start,
                declared_window_end_seconds=window_end,
                bin_shift_seconds=bin_shift,
            )
        )

    locks = manifest.claim_locks
    return TafazoliProcessedAuditReport(
        schema_version=REPORT_SCHEMA_VERSION,
        scope=manifest.scope,
        method_status=PROCESSED_NEURAL_FIGURE_AUDIT_PASS,
        source=manifest.source,
        source_files=source_files,
        decoder_curves=tuple(decoder_audits),
        dynamic_correlations=tuple(dynamic_audits),
        processed_neural_figure_artifact_used=True,
        cross_task_decoder_artifact_reproduced=True,
        dynamic_lag_projection_artifact_reproduced=True,
        raw_trial_or_spike_reanalysis=locks.raw_trial_or_spike_reanalysis,
        classifier_refit=locks.classifier_refit,
        independent_replication=locks.independent_replication,
        independent_statistical_reanalysis=(
            locks.independent_statistical_reanalysis
        ),
        transfer_entropy_validated=locks.transfer_entropy_validated,
        causal_information_flow_validated=(
            locks.causal_information_flow_validated
        ),
        unseen_composition_tested=locks.unseen_composition_tested,
        neural_clarus_assembly_validated=(
            locks.neural_clarus_assembly_validated
        ),
        causal_instruction_set_validated=(
            locks.causal_instruction_set_validated
        ),
        full_brain_language_identified=(
            locks.full_brain_language_identified
        ),
        excluded_inferences=manifest.excluded_inferences,
        limitations=(
            "the MATLAB inputs are author-produced processed figure outputs",
            "the 250 classifier rows are resampling runs, not independent "
            "animals or biological replicates",
            "author-supplied cluster summaries are extracted, not "
            "independently recomputed from trials",
            "the dynamic artifact is a Pearson decoder-score correlation, "
            "not transfer entropy or directional causal information flow",
            "the embedded MATLAB time window is used explicitly because the "
            "standalone plotting code may shadow it through global state",
            "no raw spike train, trial-level holdout, assembly segmentation, "
            "or causal perturbation is analyzed",
        ),
        conclusion=(
            "The checksum-matched official processed outputs reproduce "
            "within-task and cross-task decoder summaries plus the published "
            "dynamic-correlation matrices. They support shared population "
            "representations as a candidate, but do not test a neural "
            "instruction set or programming language."
        ),
    )
