"""Temporal and block-bootstrap falsification gates for a candidate 4--6 signal rank."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .conscious_moment_dimension_protocol import ConsciousMomentDimensionProtocolCertificate
    from .quantitative_graph_transform import _exact_fraction
else:
    from conscious_moment_dimension_protocol import ConsciousMomentDimensionProtocolCertificate  # type: ignore[no-redef]
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]


WINDOW_PARTITION_KIND = "PREDECLARED_NONOVERLAPPING_HELDOUT_WINDOWS"
BLOCK_BOOTSTRAP_KIND = "SESSION_BLOCK_BOOTSTRAP_WITH_FROZEN_BLOCK_LENGTH"


@dataclass(frozen=True)
class DimensionSessionStabilityMetrics:
    session_id: str
    window_count: int
    conscious_band_window_fraction: Fraction
    conscious_selected_rank_window_fraction: Fraction
    conscious_rank_transition_fraction: Fraction
    selected_rank_longest_dwell_fraction: Fraction
    control_band_window_fraction: Fraction
    conscious_minus_control_band_fraction: Fraction
    bootstrap_repetition_count: int
    bootstrap_band_fraction: Fraction
    bootstrap_selected_rank_fraction: Fraction


@dataclass(frozen=True)
class ConsciousMomentDimensionStabilityCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    base_dimension_certificate: ConsciousMomentDimensionProtocolCertificate
    selected_signal_rank: int | None
    candidate_band: tuple[int, int]
    window_partition_kind: str
    block_bootstrap_kind: str
    frozen_block_length: int
    minimum_conscious_band_window_fraction: Fraction
    minimum_selected_rank_window_fraction: Fraction
    maximum_conscious_rank_transition_fraction: Fraction
    minimum_selected_rank_longest_dwell_fraction: Fraction
    minimum_conscious_control_band_gap: Fraction
    minimum_bootstrap_band_fraction: Fraction
    minimum_bootstrap_selected_rank_fraction: Fraction
    session_metrics: tuple[DimensionSessionStabilityMetrics, ...]
    all_sessions_temporally_persistent: bool
    all_sessions_control_separated: bool
    all_sessions_block_bootstrap_stable: bool
    candidate_4_6_temporal_stability_supported_by_supplied_summary: bool
    external_time_series_receipt_verified: bool
    source_locked_empirical_temporal_result: bool
    signal_rank_is_not_consciousness_dimension: bool
    consciousness_dimension_claim_admitted: bool


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")
    return value


def _rank_rows(value: object, name: str, session_count: int, minimum_width: int) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (tuple, list)) or len(value) != session_count:
        raise ValueError(f"{name} must match the session count")
    rows = []
    for i, row in enumerate(value):
        if not isinstance(row, (tuple, list)) or len(row) < minimum_width:
            raise ValueError(f"{name}[{i}] has too few entries")
        if any(type(rank) is not int or rank < 1 for rank in row):
            raise ValueError("window and bootstrap ranks must be positive built-in integers")
        rows.append(tuple(row))
    return tuple(rows)


def _fraction_in(row: tuple[int, ...], allowed: set[int]) -> Fraction:
    return Fraction(sum(value in allowed for value in row), len(row))


def _transition_fraction(row: tuple[int, ...]) -> Fraction:
    return Fraction(sum(left != right for left, right in zip(row[:-1], row[1:], strict=True)), len(row) - 1)


def _longest_dwell_fraction(row: tuple[int, ...], selected: int | None) -> Fraction:
    if selected is None:
        return Fraction(0)
    longest = current = 0
    for value in row:
        current = current + 1 if value == selected else 0
        longest = max(longest, current)
    return Fraction(longest, len(row))


def conscious_moment_dimension_stability_protocol(
    *,
    base_dimension_certificate: ConsciousMomentDimensionProtocolCertificate,
    session_ids: object,
    conscious_window_ranks_by_session: object,
    control_window_ranks_by_session: object,
    block_bootstrap_ranks_by_session: object,
    window_partition_kind: str,
    block_bootstrap_kind: str,
    frozen_block_length: int,
    minimum_conscious_band_window_fraction: object,
    minimum_selected_rank_window_fraction: object,
    maximum_conscious_rank_transition_fraction: object,
    minimum_selected_rank_longest_dwell_fraction: object,
    minimum_conscious_control_band_gap: object,
    minimum_bootstrap_band_fraction: object,
    minimum_bootstrap_selected_rank_fraction: object,
    heldout_window_summary_sha256: object,
    bootstrap_summary_sha256: object,
    stability_contract_sha256: object,
) -> ConsciousMomentDimensionStabilityCertificate:
    if not isinstance(base_dimension_certificate, ConsciousMomentDimensionProtocolCertificate):
        raise ValueError("base_dimension_certificate must be canonical")
    if not isinstance(session_ids, (tuple, list)) or len(session_ids) < 3:
        raise ValueError("at least three session IDs are required")
    ids = tuple(session_ids)
    if any(not isinstance(value, str) or not value.isdecimal() or str(int(value)) != value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("session IDs must be unique canonical decimal strings")
    conscious = _rank_rows(conscious_window_ranks_by_session, "conscious_window_ranks_by_session", len(ids), 5)
    control = _rank_rows(control_window_ranks_by_session, "control_window_ranks_by_session", len(ids), 5)
    bootstrap = _rank_rows(block_bootstrap_ranks_by_session, "block_bootstrap_ranks_by_session", len(ids), 20)
    if any(len(a) != len(b) for a, b in zip(conscious, control, strict=True)):
        raise ValueError("conscious and control sessions must use the same window count")
    if type(frozen_block_length) is not int or frozen_block_length < 1:
        raise ValueError("frozen_block_length must be a positive built-in integer")
    thresholds = tuple(
        _exact_fraction(value, name)
        for value, name in (
            (minimum_conscious_band_window_fraction, "minimum_conscious_band_window_fraction"),
            (minimum_selected_rank_window_fraction, "minimum_selected_rank_window_fraction"),
            (maximum_conscious_rank_transition_fraction, "maximum_conscious_rank_transition_fraction"),
            (minimum_selected_rank_longest_dwell_fraction, "minimum_selected_rank_longest_dwell_fraction"),
            (minimum_conscious_control_band_gap, "minimum_conscious_control_band_gap"),
            (minimum_bootstrap_band_fraction, "minimum_bootstrap_band_fraction"),
            (minimum_bootstrap_selected_rank_fraction, "minimum_bootstrap_selected_rank_fraction"),
        )
    )
    if any(value < 0 or value > 1 for value in thresholds):
        raise ValueError("stability thresholds must lie in the unit interval")
    min_band, min_selected, max_transition, min_dwell, min_gap, min_boot_band, min_boot_selected = thresholds
    _hash(heldout_window_summary_sha256, "heldout_window_summary_sha256")
    _hash(bootstrap_summary_sha256, "bootstrap_summary_sha256")
    _hash(stability_contract_sha256, "stability_contract_sha256")

    selected = base_dimension_certificate.selected_signal_rank
    band = base_dimension_certificate.candidate_band
    allowed = set(range(band[0], band[1] + 1))
    metrics = []
    temporal = True
    separated = True
    stable = True
    failures: list[str] = []
    if base_dimension_certificate.validation_level is None:
        failures.append("DIMENSION_STABILITY_BASE_CERTIFICATE_NOT_VALIDATED")
    if selected is None:
        failures.append("DIMENSION_STABILITY_BASE_SELECTED_RANK_MISSING")
    if window_partition_kind != WINDOW_PARTITION_KIND:
        failures.append("DIMENSION_STABILITY_WINDOW_PARTITION_NOT_FROZEN")
    if block_bootstrap_kind != BLOCK_BOOTSTRAP_KIND:
        failures.append("DIMENSION_STABILITY_BOOTSTRAP_KIND_NOT_FROZEN")
    for index, (session_id, conscious_row, control_row, bootstrap_row) in enumerate(zip(ids, conscious, control, bootstrap, strict=True)):
        conscious_band = _fraction_in(conscious_row, allowed)
        selected_fraction = _fraction_in(conscious_row, {selected}) if selected is not None else Fraction(0)
        transition_fraction = _transition_fraction(conscious_row)
        dwell_fraction = _longest_dwell_fraction(conscious_row, selected)
        control_band = _fraction_in(control_row, allowed)
        gap = conscious_band - control_band
        bootstrap_band = _fraction_in(bootstrap_row, allowed)
        bootstrap_selected = _fraction_in(bootstrap_row, {selected}) if selected is not None else Fraction(0)
        metrics.append(DimensionSessionStabilityMetrics(
            session_id=session_id, window_count=len(conscious_row),
            conscious_band_window_fraction=conscious_band,
            conscious_selected_rank_window_fraction=selected_fraction,
            conscious_rank_transition_fraction=transition_fraction,
            selected_rank_longest_dwell_fraction=dwell_fraction,
            control_band_window_fraction=control_band,
            conscious_minus_control_band_fraction=gap,
            bootstrap_repetition_count=len(bootstrap_row),
            bootstrap_band_fraction=bootstrap_band,
            bootstrap_selected_rank_fraction=bootstrap_selected,
        ))
        if (
            conscious_band < min_band or selected_fraction < min_selected
            or transition_fraction > max_transition or dwell_fraction < min_dwell
        ):
            temporal = False
            failures.append(f"DIMENSION_STABILITY_SESSION_{index}_TEMPORAL_PERSISTENCE_FAILED")
        if gap < min_gap:
            separated = False
            failures.append(f"DIMENSION_STABILITY_SESSION_{index}_CONTROL_SEPARATION_FAILED")
        if bootstrap_band < min_boot_band or bootstrap_selected < min_boot_selected:
            stable = False
            failures.append(f"DIMENSION_STABILITY_SESSION_{index}_BLOCK_BOOTSTRAP_FAILED")
    support = not failures
    external = False
    common = dict(
        failure_codes=tuple(failures), base_dimension_certificate=base_dimension_certificate,
        selected_signal_rank=selected, candidate_band=band,
        window_partition_kind=window_partition_kind, block_bootstrap_kind=block_bootstrap_kind,
        frozen_block_length=frozen_block_length,
        minimum_conscious_band_window_fraction=min_band,
        minimum_selected_rank_window_fraction=min_selected,
        maximum_conscious_rank_transition_fraction=max_transition,
        minimum_selected_rank_longest_dwell_fraction=min_dwell,
        minimum_conscious_control_band_gap=min_gap,
        minimum_bootstrap_band_fraction=min_boot_band,
        minimum_bootstrap_selected_rank_fraction=min_boot_selected,
        session_metrics=tuple(metrics), all_sessions_temporally_persistent=temporal,
        all_sessions_control_separated=separated,
        all_sessions_block_bootstrap_stable=stable,
        candidate_4_6_temporal_stability_supported_by_supplied_summary=support,
        external_time_series_receipt_verified=external,
        source_locked_empirical_temporal_result=False,
        signal_rank_is_not_consciousness_dimension=True,
        consciousness_dimension_claim_admitted=False,
    )
    if failures:
        return ConsciousMomentDimensionStabilityCertificate(
            status=failures[0], validation_level=None, **common
        )
    status = "VALIDATED_SUPPLIED_4_6_TEMPORAL_AND_BLOCK_BOOTSTRAP_STABILITY_APPARATUS"
    return ConsciousMomentDimensionStabilityCertificate(
        status=status, validation_level=status, **common
    )


__all__ = [
    "BLOCK_BOOTSTRAP_KIND", "ConsciousMomentDimensionStabilityCertificate",
    "DimensionSessionStabilityMetrics", "WINDOW_PARTITION_KIND",
    "conscious_moment_dimension_stability_protocol",
]
