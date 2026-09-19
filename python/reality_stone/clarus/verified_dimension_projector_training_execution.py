"""Verify exact development covariance eigenbases and execute held-out dimension gates."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .conscious_moment_dimension_protocol import ConsciousMomentDimensionProtocolCertificate, FROZEN_DIMENSION_MENU
    from .quantitative_graph_transform import _exact_fraction
    from .verified_dimension_time_series_execution import (
        VerifiedDimensionTimeSeriesExecution,
        verified_dimension_time_series_execution,
    )
else:
    from conscious_moment_dimension_protocol import ConsciousMomentDimensionProtocolCertificate, FROZEN_DIMENSION_MENU  # type: ignore[no-redef]
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]
    from verified_dimension_time_series_execution import (  # type: ignore[no-redef]
        VerifiedDimensionTimeSeriesExecution,
        verified_dimension_time_series_execution,
    )


@dataclass(frozen=True)
class DevelopmentSessionProjectorTrainingReceipt:
    session_id: str
    observation_count: int
    ambient_dimension: int
    exact_zero_mean_verified: bool
    covariance_matrix: tuple[tuple[Fraction, ...], ...]
    ordered_eigenvalues: tuple[Fraction, ...]
    orthonormal_eigenbasis_verified: bool
    covariance_eigen_equations_verified: bool
    strict_eigenvalue_order_verified: bool
    projection_menu: tuple[tuple[int, tuple[tuple[Fraction, ...], ...]], ...]


@dataclass(frozen=True)
class VerifiedDimensionProjectorTrainingExecution:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    training_receipts: tuple[DevelopmentSessionProjectorTrainingReceipt, ...]
    heldout_execution: VerifiedDimensionTimeSeriesExecution | None
    exact_development_covariance_recomputed: bool
    exact_eigenbasis_witness_verified: bool
    strict_all_rank_pca_order_verified: bool
    projector_menus_constructed_from_development_receipts: bool
    development_projector_training_verified: bool
    heldout_execution_composed: bool
    external_development_bytes_receipt_verified: bool
    preprocessing_execution_verified: bool
    source_locked_empirical_dimension_result: bool
    consciousness_dimension_claim_admitted: bool


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")
    return value


def _vectors(value: object, name: str, sessions: int):
    if not isinstance(value, (tuple, list)) or len(value) != sessions:
        raise ValueError(f"{name} must match session count")
    parsed_sessions = []
    ambient = None
    for s, rows in enumerate(value):
        if not isinstance(rows, (tuple, list)) or len(rows) < 2:
            raise ValueError(f"{name}[{s}] must contain at least two vectors")
        parsed_rows = []
        for i, row in enumerate(rows):
            if not isinstance(row, (tuple, list)) or not row:
                raise ValueError(f"{name}[{s}][{i}] must be a vector")
            if ambient is None:
                ambient = len(row)
            if len(row) != ambient:
                raise ValueError("all development vectors must share ambient dimension")
            parsed_rows.append(tuple(_exact_fraction(x, f"{name}[{s}][{i}]") for x in row))
        parsed_sessions.append(tuple(parsed_rows))
    assert ambient is not None
    return tuple(parsed_sessions), ambient


def _basis(value: object, sessions: int, n: int):
    if not isinstance(value, (tuple, list)) or len(value) != sessions:
        raise ValueError("eigenbasis witnesses must match session count")
    result = []
    for s, basis in enumerate(value):
        if not isinstance(basis, (tuple, list)) or len(basis) != n:
            raise ValueError(f"eigenbasis session {s} must contain ambient-dimension vectors")
        vectors = []
        for i, vector in enumerate(basis):
            if not isinstance(vector, (tuple, list)) or len(vector) != n:
                raise ValueError(f"eigenbasis session {s} vector {i} has wrong width")
            vectors.append(tuple(_exact_fraction(x, f"eigenbasis[{s}][{i}]") for x in vector))
        result.append(tuple(vectors))
    return tuple(result)


def _dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


def _covariance(rows):
    n = len(rows[0])
    return tuple(tuple(sum((row[i] * row[j] for row in rows), Fraction(0)) / len(rows) for j in range(n)) for i in range(n))


def _matvec(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _projector(vectors):
    n = len(vectors[0])
    return tuple(tuple(sum((vector[i] * vector[j] for vector in vectors), Fraction(0)) for j in range(n)) for i in range(n))


def verified_dimension_projector_training_execution(
    *,
    base_dimension_certificate: ConsciousMomentDimensionProtocolCertificate,
    session_ids: object,
    centered_development_observations_by_session: object,
    covariance_eigenbasis_witnesses_by_session: object,
    conscious_heldout_observations_by_session: object,
    control_heldout_observations_by_session: object,
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
    development_observation_sha256: object,
    eigenbasis_witness_sha256: object,
    heldout_observation_sha256: object,
    bootstrap_schedule_sha256: object,
    execution_contract_sha256: object,
) -> VerifiedDimensionProjectorTrainingExecution:
    if not isinstance(base_dimension_certificate, ConsciousMomentDimensionProtocolCertificate):
        raise ValueError("base_dimension_certificate must be canonical")
    if not isinstance(session_ids, (tuple, list)) or len(session_ids) < 3:
        raise ValueError("at least three sessions are required")
    ids = tuple(session_ids)
    development, n = _vectors(centered_development_observations_by_session, "centered_development_observations_by_session", len(ids))
    if n < max(FROZEN_DIMENSION_MENU):
        raise ValueError("ambient dimension must cover the frozen menu")
    bases = _basis(covariance_eigenbasis_witnesses_by_session, len(ids), n)
    for value, name in (
        (development_observation_sha256, "development_observation_sha256"),
        (eigenbasis_witness_sha256, "eigenbasis_witness_sha256"),
        (heldout_observation_sha256, "heldout_observation_sha256"),
        (bootstrap_schedule_sha256, "bootstrap_schedule_sha256"),
        (execution_contract_sha256, "execution_contract_sha256"),
    ):
        _hash(value, name)

    failures: list[str] = []
    receipts = []
    menus = []
    all_centered = all_orthonormal = all_eigen = all_strict = True
    for s, (session_id, rows, basis) in enumerate(zip(ids, development, bases, strict=True)):
        centered = all(sum((row[j] for row in rows), Fraction(0)) == 0 for j in range(n))
        covariance = _covariance(rows)
        orthonormal = all(_dot(basis[i], basis[j]) == (1 if i == j else 0) for i in range(n) for j in range(n))
        eigenvalues = tuple(_dot(vector, _matvec(covariance, vector)) for vector in basis)
        eigen = all(_matvec(covariance, vector) == tuple(value * x for x in vector) for vector, value in zip(basis, eigenvalues, strict=True))
        strict_order = all(eigenvalues[i] > eigenvalues[i + 1] for i in range(n - 1))
        all_centered &= centered; all_orthonormal &= orthonormal; all_eigen &= eigen; all_strict &= strict_order
        if not centered:
            failures.append(f"DIMENSION_TRAINING_SESSION_{s}_DEVELOPMENT_NOT_EXACTLY_CENTERED")
        if not orthonormal:
            failures.append(f"DIMENSION_TRAINING_SESSION_{s}_EIGENBASIS_NOT_ORTHONORMAL")
        if not eigen:
            failures.append(f"DIMENSION_TRAINING_SESSION_{s}_COVARIANCE_EIGEN_EQUATION_FAILED")
        if not strict_order:
            failures.append(f"DIMENSION_TRAINING_SESSION_{s}_EIGENVALUES_NOT_STRICTLY_ORDERED")
        menu = {d: _projector(basis[:d]) for d in FROZEN_DIMENSION_MENU}
        menus.append(menu)
        receipts.append(DevelopmentSessionProjectorTrainingReceipt(
            session_id=session_id, observation_count=len(rows), ambient_dimension=n,
            exact_zero_mean_verified=centered, covariance_matrix=covariance,
            ordered_eigenvalues=eigenvalues,
            orthonormal_eigenbasis_verified=orthonormal,
            covariance_eigen_equations_verified=eigen,
            strict_eigenvalue_order_verified=strict_order,
            projection_menu=tuple((d, menu[d]) for d in FROZEN_DIMENSION_MENU),
        ))
    training_ok = all_centered and all_orthonormal and all_eigen and all_strict
    heldout = None
    if training_ok:
        heldout = verified_dimension_time_series_execution(
            base_dimension_certificate=base_dimension_certificate, session_ids=ids,
            conscious_heldout_observations_by_session=conscious_heldout_observations_by_session,
            control_heldout_observations_by_session=control_heldout_observations_by_session,
            projection_menus_by_session=tuple(menus), complexity_penalty=complexity_penalty,
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
            heldout_observation_sha256=heldout_observation_sha256,
            projector_menu_sha256=eigenbasis_witness_sha256,
            bootstrap_schedule_sha256=bootstrap_schedule_sha256,
            execution_contract_sha256=execution_contract_sha256,
        )
        if heldout.validation_level is None:
            failures.extend(heldout.failure_codes)
    success = not failures and heldout is not None
    status = "VERIFIED_EXACT_DEVELOPMENT_PCA_AND_HELDOUT_DIMENSION_EXECUTION" if success else failures[0]
    return VerifiedDimensionProjectorTrainingExecution(
        status=status, validation_level=status if success else None,
        failure_codes=tuple(failures), training_receipts=tuple(receipts),
        heldout_execution=heldout,
        exact_development_covariance_recomputed=True,
        exact_eigenbasis_witness_verified=all_orthonormal and all_eigen,
        strict_all_rank_pca_order_verified=all_strict,
        projector_menus_constructed_from_development_receipts=training_ok,
        development_projector_training_verified=training_ok,
        heldout_execution_composed=success,
        external_development_bytes_receipt_verified=False,
        preprocessing_execution_verified=False,
        source_locked_empirical_dimension_result=False,
        consciousness_dimension_claim_admitted=False,
    )


__all__ = ["DevelopmentSessionProjectorTrainingReceipt", "VerifiedDimensionProjectorTrainingExecution", "verified_dimension_projector_training_execution"]
