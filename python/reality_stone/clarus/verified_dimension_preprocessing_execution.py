"""Development-only exact centering and reference-scale preprocessing for dimension execution."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .conscious_moment_dimension_protocol import ConsciousMomentDimensionProtocolCertificate
    from .quantitative_graph_transform import _exact_fraction
    from .verified_dimension_projector_training_execution import (
        VerifiedDimensionProjectorTrainingExecution,
        verified_dimension_projector_training_execution,
    )
else:
    from conscious_moment_dimension_protocol import ConsciousMomentDimensionProtocolCertificate  # type: ignore[no-redef]
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]
    from verified_dimension_projector_training_execution import (  # type: ignore[no-redef]
        VerifiedDimensionProjectorTrainingExecution,
        verified_dimension_projector_training_execution,
    )


@dataclass(frozen=True)
class DimensionSessionPreprocessingReceipt:
    session_id: str
    ambient_dimension: int
    development_observation_count: int
    development_coordinate_mean: tuple[Fraction, ...]
    frozen_coordinate_reference_scales: tuple[Fraction, ...]
    normalized_development_coordinate_sum: tuple[Fraction, ...]
    development_mean_used_for_all_splits: bool
    heldout_statistics_reestimated: bool


@dataclass(frozen=True)
class VerifiedDimensionPreprocessingExecution:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    preprocessing_receipts: tuple[DimensionSessionPreprocessingReceipt, ...]
    projector_training_execution: VerifiedDimensionProjectorTrainingExecution | None
    raw_development_vectors_parsed_exactly: bool
    development_means_recomputed_exactly: bool
    positive_frozen_reference_scales_verified: bool
    development_only_transform_applied_to_all_splits: bool
    heldout_statistic_leakage_detected: bool
    preprocessing_to_training_and_heldout_chain_composed: bool
    external_raw_bytes_receipt_verified: bool
    source_locked_empirical_dimension_result: bool
    consciousness_dimension_claim_admitted: bool


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")
    return value


def _session_vectors(value: object, name: str, sessions: int, n: int, minimum: int):
    if not isinstance(value, (tuple, list)) or len(value) != sessions:
        raise ValueError(f"{name} must match session count")
    output = []
    for s, rows in enumerate(value):
        if not isinstance(rows, (tuple, list)) or len(rows) < minimum:
            raise ValueError(f"{name}[{s}] has too few observations")
        parsed = []
        for i, row in enumerate(rows):
            if not isinstance(row, (tuple, list)) or len(row) != n:
                raise ValueError(f"{name}[{s}][{i}] must have ambient width")
            parsed.append(tuple(_exact_fraction(x, f"{name}[{s}][{i}]") for x in row))
        output.append(tuple(parsed))
    return tuple(output)


def _window_vectors(value: object, name: str, sessions: int, n: int):
    if not isinstance(value, (tuple, list)) or len(value) != sessions:
        raise ValueError(f"{name} must match session count")
    output = []
    for s, windows in enumerate(value):
        if not isinstance(windows, (tuple, list)) or len(windows) < 5:
            raise ValueError(f"{name}[{s}] must contain at least five windows")
        parsed_windows = []
        for w, rows in enumerate(windows):
            if not isinstance(rows, (tuple, list)) or not rows:
                raise ValueError(f"{name}[{s}][{w}] must contain observations")
            parsed = []
            for i, row in enumerate(rows):
                if not isinstance(row, (tuple, list)) or len(row) != n:
                    raise ValueError(f"{name}[{s}][{w}][{i}] must have ambient width")
                parsed.append(tuple(_exact_fraction(x, f"{name}[{s}][{w}][{i}]") for x in row))
            parsed_windows.append(tuple(parsed))
        output.append(tuple(parsed_windows))
    return tuple(output)


def _normalize(vector, mean, scales):
    return tuple((value - center) / scale for value, center, scale in zip(vector, mean, scales, strict=True))


def verified_dimension_preprocessing_execution(
    *,
    base_dimension_certificate: ConsciousMomentDimensionProtocolCertificate,
    session_ids: object,
    raw_development_observations_by_session: object,
    raw_conscious_heldout_observations_by_session: object,
    raw_control_heldout_observations_by_session: object,
    coordinate_reference_scales_by_session: object,
    covariance_eigenbasis_witnesses_by_session: object,
    complexity_penalty: object,
    bootstrap_window_indices_by_session: object,
    frozen_block_length: int,
    window_partition_kind: str,
    block_bootstrap_kind: str,
    minimum_conscious_band_window_fraction: object,
    minimum_selected_rank_window_fraction: object,
    maximum_conscious_rank_transition_fraction: object,
    minimum_selected_rank_longest_dwell_fraction: object,
    minimum_conscious_control_band_gap: object,
    minimum_bootstrap_band_fraction: object,
    minimum_bootstrap_selected_rank_fraction: object,
    raw_development_sha256: object,
    raw_heldout_sha256: object,
    preprocessing_contract_sha256: object,
    eigenbasis_witness_sha256: object,
    bootstrap_schedule_sha256: object,
    execution_contract_sha256: object,
) -> VerifiedDimensionPreprocessingExecution:
    if not isinstance(session_ids, (tuple, list)) or len(session_ids) < 3:
        raise ValueError("at least three session IDs are required")
    ids = tuple(session_ids)
    if not isinstance(coordinate_reference_scales_by_session, (tuple, list)) or len(coordinate_reference_scales_by_session) != len(ids):
        raise ValueError("coordinate reference scales must match session count")
    first = coordinate_reference_scales_by_session[0]
    if not isinstance(first, (tuple, list)) or not first:
        raise ValueError("coordinate reference scale rows must be nonempty")
    n = len(first)
    scales = []
    for s, row in enumerate(coordinate_reference_scales_by_session):
        if not isinstance(row, (tuple, list)) or len(row) != n:
            raise ValueError("coordinate reference scale rows must share width")
        parsed = tuple(_exact_fraction(x, f"coordinate_reference_scales_by_session[{s}]") for x in row)
        if min(parsed) <= 0:
            raise ValueError("coordinate reference scales must be positive")
        scales.append(parsed)
    development = _session_vectors(raw_development_observations_by_session, "raw_development_observations_by_session", len(ids), n, 2)
    conscious = _window_vectors(raw_conscious_heldout_observations_by_session, "raw_conscious_heldout_observations_by_session", len(ids), n)
    control = _window_vectors(raw_control_heldout_observations_by_session, "raw_control_heldout_observations_by_session", len(ids), n)
    for value, name in (
        (raw_development_sha256, "raw_development_sha256"),
        (raw_heldout_sha256, "raw_heldout_sha256"),
        (preprocessing_contract_sha256, "preprocessing_contract_sha256"),
        (eigenbasis_witness_sha256, "eigenbasis_witness_sha256"),
        (bootstrap_schedule_sha256, "bootstrap_schedule_sha256"),
        (execution_contract_sha256, "execution_contract_sha256"),
    ):
        _hash(value, name)

    normalized_development = []
    normalized_conscious = []
    normalized_control = []
    receipts = []
    for session_id, development_rows, conscious_windows, control_windows, scale_row in zip(
        ids, development, conscious, control, scales, strict=True
    ):
        mean = tuple(sum((row[j] for row in development_rows), Fraction(0)) / len(development_rows) for j in range(n))
        normalized_dev = tuple(_normalize(row, mean, scale_row) for row in development_rows)
        normalized_c = tuple(tuple(_normalize(row, mean, scale_row) for row in window) for window in conscious_windows)
        normalized_ctl = tuple(tuple(_normalize(row, mean, scale_row) for row in window) for window in control_windows)
        sums = tuple(sum((row[j] for row in normalized_dev), Fraction(0)) for j in range(n))
        normalized_development.append(normalized_dev)
        normalized_conscious.append(normalized_c)
        normalized_control.append(normalized_ctl)
        receipts.append(DimensionSessionPreprocessingReceipt(
            session_id=session_id, ambient_dimension=n,
            development_observation_count=len(development_rows),
            development_coordinate_mean=mean,
            frozen_coordinate_reference_scales=scale_row,
            normalized_development_coordinate_sum=sums,
            development_mean_used_for_all_splits=True,
            heldout_statistics_reestimated=False,
        ))
    training = verified_dimension_projector_training_execution(
        base_dimension_certificate=base_dimension_certificate, session_ids=ids,
        centered_development_observations_by_session=tuple(normalized_development),
        covariance_eigenbasis_witnesses_by_session=covariance_eigenbasis_witnesses_by_session,
        conscious_heldout_observations_by_session=tuple(normalized_conscious),
        control_heldout_observations_by_session=tuple(normalized_control),
        complexity_penalty=complexity_penalty,
        bootstrap_window_indices_by_session=bootstrap_window_indices_by_session,
        frozen_block_length=frozen_block_length,
        window_partition_kind=window_partition_kind, block_bootstrap_kind=block_bootstrap_kind,
        minimum_conscious_band_window_fraction=minimum_conscious_band_window_fraction,
        minimum_selected_rank_window_fraction=minimum_selected_rank_window_fraction,
        maximum_conscious_rank_transition_fraction=maximum_conscious_rank_transition_fraction,
        minimum_selected_rank_longest_dwell_fraction=minimum_selected_rank_longest_dwell_fraction,
        minimum_conscious_control_band_gap=minimum_conscious_control_band_gap,
        minimum_bootstrap_band_fraction=minimum_bootstrap_band_fraction,
        minimum_bootstrap_selected_rank_fraction=minimum_bootstrap_selected_rank_fraction,
        development_observation_sha256=raw_development_sha256,
        eigenbasis_witness_sha256=eigenbasis_witness_sha256,
        heldout_observation_sha256=raw_heldout_sha256,
        bootstrap_schedule_sha256=bootstrap_schedule_sha256,
        execution_contract_sha256=execution_contract_sha256,
    )
    failures = list(training.failure_codes) if training.validation_level is None else []
    success = not failures
    status = "VERIFIED_DEVELOPMENT_ONLY_PREPROCESSING_TO_DIMENSION_STABILITY" if success else failures[0]
    return VerifiedDimensionPreprocessingExecution(
        status=status, validation_level=status if success else None,
        failure_codes=tuple(failures), preprocessing_receipts=tuple(receipts),
        projector_training_execution=training,
        raw_development_vectors_parsed_exactly=True,
        development_means_recomputed_exactly=True,
        positive_frozen_reference_scales_verified=True,
        development_only_transform_applied_to_all_splits=True,
        heldout_statistic_leakage_detected=False,
        preprocessing_to_training_and_heldout_chain_composed=success,
        external_raw_bytes_receipt_verified=False,
        source_locked_empirical_dimension_result=False,
        consciousness_dimension_claim_admitted=False,
    )


__all__ = ["DimensionSessionPreprocessingReceipt", "VerifiedDimensionPreprocessingExecution", "verified_dimension_preprocessing_execution"]
