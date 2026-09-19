"""Exact held-out window scoring and moving-block rank stability execution."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .conscious_moment_dimension_protocol import (
        ConsciousMomentDimensionProtocolCertificate,
        FROZEN_DIMENSION_MENU,
    )
    from .conscious_moment_dimension_stability_protocol import (
        ConsciousMomentDimensionStabilityCertificate,
        conscious_moment_dimension_stability_protocol,
    )
    from .quantitative_graph_transform import _exact_fraction
else:
    from conscious_moment_dimension_protocol import (  # type: ignore[no-redef]
        ConsciousMomentDimensionProtocolCertificate,
        FROZEN_DIMENSION_MENU,
    )
    from conscious_moment_dimension_stability_protocol import (  # type: ignore[no-redef]
        ConsciousMomentDimensionStabilityCertificate,
        conscious_moment_dimension_stability_protocol,
    )
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]


@dataclass(frozen=True)
class ExactWindowDimensionScores:
    session_id: str
    conscious_scores_by_window: tuple[tuple[tuple[int, Fraction], ...], ...]
    control_scores_by_window: tuple[tuple[tuple[int, Fraction], ...], ...]
    conscious_selected_ranks: tuple[int, ...]
    control_selected_ranks: tuple[int, ...]
    aggregate_conscious_selected_rank: int | None
    bootstrap_selected_ranks: tuple[int, ...]


@dataclass(frozen=True)
class VerifiedDimensionTimeSeriesExecution:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    base_dimension_certificate: ConsciousMomentDimensionProtocolCertificate
    stability_certificate: ConsciousMomentDimensionStabilityCertificate | None
    session_ids: tuple[str, ...]
    ambient_dimension: int
    complexity_penalty: Fraction
    frozen_block_length: int
    exact_projection_menu_verified: bool
    moving_block_schedule_verified: bool
    window_execution_receipts: tuple[ExactWindowDimensionScores, ...]
    generated_conscious_window_ranks_by_session: tuple[tuple[int, ...], ...]
    generated_control_window_ranks_by_session: tuple[tuple[int, ...], ...]
    generated_block_bootstrap_ranks_by_session: tuple[tuple[int, ...], ...]
    heldout_scores_recomputed_from_observations: bool
    bootstrap_ranks_recomputed_from_frozen_schedule: bool
    stability_composed_from_generated_summaries: bool
    external_observation_bytes_receipt_verified: bool
    projector_training_execution_verified: bool
    source_locked_empirical_dimension_result: bool
    consciousness_dimension_claim_admitted: bool


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")
    return value


def _matrix(value: object, name: str, n: int) -> tuple[tuple[Fraction, ...], ...]:
    if not isinstance(value, (tuple, list)) or len(value) != n:
        raise ValueError(f"{name} must be an {n} by {n} matrix")
    rows = []
    for i, row in enumerate(value):
        if not isinstance(row, (tuple, list)) or len(row) != n:
            raise ValueError(f"{name}[{i}] must have width {n}")
        rows.append(tuple(_exact_fraction(item, f"{name}[{i}][{j}]") for j, item in enumerate(row)))
    return tuple(rows)


def _matmul(left, right):
    n = len(left)
    return tuple(tuple(sum((left[i][k] * right[k][j] for k in range(n)), Fraction(0)) for j in range(n)) for i in range(n))


def _projector_ok(matrix, rank: int) -> bool:
    n = len(matrix)
    return (
        matrix == tuple(tuple(matrix[j][i] for j in range(n)) for i in range(n))
        and _matmul(matrix, matrix) == matrix
        and sum(matrix[i][i] for i in range(n)) == rank
    )


def _windows(value: object, name: str, sessions: int, n: int):
    if not isinstance(value, (tuple, list)) or len(value) != sessions:
        raise ValueError(f"{name} must match session count")
    parsed_sessions = []
    for s, session in enumerate(value):
        if not isinstance(session, (tuple, list)) or len(session) < 5:
            raise ValueError(f"{name}[{s}] must contain at least five windows")
        parsed_windows = []
        for w, window in enumerate(session):
            if not isinstance(window, (tuple, list)) or not window:
                raise ValueError(f"{name}[{s}][{w}] must contain observations")
            observations = []
            for t, vector in enumerate(window):
                if not isinstance(vector, (tuple, list)) or len(vector) != n:
                    raise ValueError(f"{name}[{s}][{w}][{t}] must have ambient width")
                observations.append(tuple(_exact_fraction(x, f"{name}[{s}][{w}][{t}]") for x in vector))
            parsed_windows.append(tuple(observations))
        parsed_sessions.append(tuple(parsed_windows))
    return tuple(parsed_sessions)


def _score(window, projector, dimension: int, penalty: Fraction) -> Fraction:
    total = Fraction(0)
    for vector in window:
        projected = tuple(sum((projector[i][j] * vector[j] for j in range(len(vector))), Fraction(0)) for i in range(len(vector)))
        total += sum((vector[i] - projected[i]) ** 2 for i in range(len(vector)))
    return -(total / len(window)) - penalty * dimension


def _unique_winner(scores: dict[int, Fraction]) -> int | None:
    best = max(scores.values())
    winners = tuple(d for d, value in scores.items() if value == best)
    return winners[0] if len(winners) == 1 else None


def _schedule(value: object, sessions: int, window_counts: tuple[int, ...], block: int):
    if not isinstance(value, (tuple, list)) or len(value) != sessions:
        raise ValueError("bootstrap schedule must match session count")
    parsed = []
    valid = True
    for s, (replicates, count) in enumerate(zip(value, window_counts, strict=True)):
        if count % block:
            raise ValueError("window count must be divisible by frozen block length")
        if not isinstance(replicates, (tuple, list)) or len(replicates) < 20:
            raise ValueError(f"bootstrap schedule session {s} needs at least twenty replicates")
        session_rows = []
        for row in replicates:
            if not isinstance(row, (tuple, list)) or len(row) != count:
                raise ValueError("every bootstrap replicate must resample exactly the window count")
            if any(type(index) is not int or index < 0 or index >= count for index in row):
                raise ValueError("bootstrap indices must be built-in integers in range")
            indices = tuple(row)
            for start in range(0, count, block):
                chunk = indices[start:start + block]
                if any(chunk[j] != (chunk[0] + j) % count for j in range(block)):
                    valid = False
            session_rows.append(indices)
        parsed.append(tuple(session_rows))
    return tuple(parsed), valid


def verified_dimension_time_series_execution(
    *,
    base_dimension_certificate: ConsciousMomentDimensionProtocolCertificate,
    session_ids: object,
    conscious_heldout_observations_by_session: object,
    control_heldout_observations_by_session: object,
    projection_menus_by_session: object,
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
    heldout_observation_sha256: object,
    projector_menu_sha256: object,
    bootstrap_schedule_sha256: object,
    execution_contract_sha256: object,
) -> VerifiedDimensionTimeSeriesExecution:
    if not isinstance(base_dimension_certificate, ConsciousMomentDimensionProtocolCertificate):
        raise ValueError("base_dimension_certificate must be canonical")
    if not isinstance(session_ids, (tuple, list)) or len(session_ids) < 3:
        raise ValueError("at least three session IDs are required")
    ids = tuple(session_ids)
    if any(not isinstance(value, str) or not value.isdecimal() or str(int(value)) != value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("session IDs must be unique canonical decimal strings")
    if not isinstance(projection_menus_by_session, (tuple, list)) or len(projection_menus_by_session) != len(ids):
        raise ValueError("projection menus must match session count")
    first = projection_menus_by_session[0]
    if not isinstance(first, dict) or set(first) != set(FROZEN_DIMENSION_MENU):
        raise ValueError("every projection menu must contain the complete frozen dimension menu")
    sample = first[FROZEN_DIMENSION_MENU[-1]]
    if not isinstance(sample, (tuple, list)) or not sample:
        raise ValueError("projection matrices must be nonempty")
    n = len(sample)
    if n < max(FROZEN_DIMENSION_MENU):
        raise ValueError("ambient dimension must cover the frozen dimension menu")
    menus = []
    projector_ok = True
    for s, menu in enumerate(projection_menus_by_session):
        if not isinstance(menu, dict) or set(menu) != set(FROZEN_DIMENSION_MENU):
            raise ValueError("every projection menu must contain the complete frozen dimension menu")
        parsed = {}
        for d in FROZEN_DIMENSION_MENU:
            matrix = _matrix(menu[d], f"projection_menus_by_session[{s}][{d}]", n)
            projector_ok = projector_ok and _projector_ok(matrix, d)
            parsed[d] = matrix
        menus.append(parsed)
    conscious = _windows(conscious_heldout_observations_by_session, "conscious_heldout_observations_by_session", len(ids), n)
    control = _windows(control_heldout_observations_by_session, "control_heldout_observations_by_session", len(ids), n)
    if any(len(a) != len(b) for a, b in zip(conscious, control, strict=True)):
        raise ValueError("conscious and control sessions must share window counts")
    if type(frozen_block_length) is not int or frozen_block_length < 1:
        raise ValueError("frozen_block_length must be a positive built-in integer")
    schedule, schedule_ok = _schedule(
        bootstrap_window_indices_by_session, len(ids), tuple(len(row) for row in conscious), frozen_block_length
    )
    penalty = _exact_fraction(complexity_penalty, "complexity_penalty")
    if penalty < 0:
        raise ValueError("complexity_penalty must be nonnegative")
    for value, name in (
        (heldout_observation_sha256, "heldout_observation_sha256"),
        (projector_menu_sha256, "projector_menu_sha256"),
        (bootstrap_schedule_sha256, "bootstrap_schedule_sha256"),
        (execution_contract_sha256, "execution_contract_sha256"),
    ):
        _hash(value, name)

    failures: list[str] = []
    if not projector_ok:
        failures.append("DIMENSION_EXECUTION_PROJECTION_MENU_NOT_EXACT_ORTHOGONAL_PROJECTORS")
    if not schedule_ok:
        failures.append("DIMENSION_EXECUTION_BOOTSTRAP_SCHEDULE_NOT_MOVING_BLOCK")
    receipts = []
    conscious_ranks, control_ranks, bootstrap_ranks = [], [], []
    base_selected = base_dimension_certificate.selected_signal_rank
    for s, (session_id, conscious_windows, control_windows, menu, resamples) in enumerate(
        zip(ids, conscious, control, menus, schedule, strict=True)
    ):
        conscious_score_rows, control_score_rows = [], []
        session_conscious, session_control = [], []
        for condition, windows, score_rows, selected_rows in (
            ("CONSCIOUS", conscious_windows, conscious_score_rows, session_conscious),
            ("CONTROL", control_windows, control_score_rows, session_control),
        ):
            for w, window in enumerate(windows):
                scores = {d: _score(window, menu[d], d, penalty) for d in FROZEN_DIMENSION_MENU}
                winner = _unique_winner(scores)
                score_rows.append(tuple((d, scores[d]) for d in FROZEN_DIMENSION_MENU))
                if winner is None:
                    failures.append(f"DIMENSION_EXECUTION_SESSION_{s}_{condition}_WINDOW_{w}_WINNER_NOT_UNIQUE")
                    selected_rows.append(FROZEN_DIMENSION_MENU[0])
                else:
                    selected_rows.append(winner)
        aggregate_scores = {
            d: sum((dict(row)[d] for row in conscious_score_rows), Fraction(0)) / len(conscious_score_rows)
            for d in FROZEN_DIMENSION_MENU
        }
        aggregate = _unique_winner(aggregate_scores)
        if aggregate is None:
            failures.append(f"DIMENSION_EXECUTION_SESSION_{s}_AGGREGATE_WINNER_NOT_UNIQUE")
        elif aggregate != base_selected:
            failures.append(f"DIMENSION_EXECUTION_SESSION_{s}_AGGREGATE_RANK_DISAGREES_WITH_BASE")
        session_bootstrap = []
        for b, indices in enumerate(resamples):
            scores = {
                d: sum((dict(conscious_score_rows[index])[d] for index in indices), Fraction(0)) / len(indices)
                for d in FROZEN_DIMENSION_MENU
            }
            winner = _unique_winner(scores)
            if winner is None:
                failures.append(f"DIMENSION_EXECUTION_SESSION_{s}_BOOTSTRAP_{b}_WINNER_NOT_UNIQUE")
                session_bootstrap.append(FROZEN_DIMENSION_MENU[0])
            else:
                session_bootstrap.append(winner)
        conscious_ranks.append(tuple(session_conscious)); control_ranks.append(tuple(session_control)); bootstrap_ranks.append(tuple(session_bootstrap))
        receipts.append(ExactWindowDimensionScores(
            session_id=session_id,
            conscious_scores_by_window=tuple(conscious_score_rows),
            control_scores_by_window=tuple(control_score_rows),
            conscious_selected_ranks=tuple(session_conscious),
            control_selected_ranks=tuple(session_control),
            aggregate_conscious_selected_rank=aggregate,
            bootstrap_selected_ranks=tuple(session_bootstrap),
        ))
    stability = conscious_moment_dimension_stability_protocol(
        base_dimension_certificate=base_dimension_certificate, session_ids=ids,
        conscious_window_ranks_by_session=tuple(conscious_ranks),
        control_window_ranks_by_session=tuple(control_ranks),
        block_bootstrap_ranks_by_session=tuple(bootstrap_ranks),
        window_partition_kind=window_partition_kind,
        block_bootstrap_kind=block_bootstrap_kind,
        frozen_block_length=frozen_block_length,
        minimum_conscious_band_window_fraction=minimum_conscious_band_window_fraction,
        minimum_selected_rank_window_fraction=minimum_selected_rank_window_fraction,
        maximum_conscious_rank_transition_fraction=maximum_conscious_rank_transition_fraction,
        minimum_selected_rank_longest_dwell_fraction=minimum_selected_rank_longest_dwell_fraction,
        minimum_conscious_control_band_gap=minimum_conscious_control_band_gap,
        minimum_bootstrap_band_fraction=minimum_bootstrap_band_fraction,
        minimum_bootstrap_selected_rank_fraction=minimum_bootstrap_selected_rank_fraction,
        heldout_window_summary_sha256=heldout_observation_sha256,
        bootstrap_summary_sha256=bootstrap_schedule_sha256,
        stability_contract_sha256=execution_contract_sha256,
    )
    if stability.validation_level is None:
        failures.extend(stability.failure_codes)
    success = not failures
    status = "VERIFIED_EXACT_HELDOUT_WINDOW_AND_MOVING_BLOCK_DIMENSION_EXECUTION" if success else failures[0]
    return VerifiedDimensionTimeSeriesExecution(
        status=status, validation_level=status if success else None,
        failure_codes=tuple(failures), base_dimension_certificate=base_dimension_certificate,
        stability_certificate=stability, session_ids=ids, ambient_dimension=n,
        complexity_penalty=penalty, frozen_block_length=frozen_block_length,
        exact_projection_menu_verified=projector_ok,
        moving_block_schedule_verified=schedule_ok,
        window_execution_receipts=tuple(receipts),
        generated_conscious_window_ranks_by_session=tuple(conscious_ranks),
        generated_control_window_ranks_by_session=tuple(control_ranks),
        generated_block_bootstrap_ranks_by_session=tuple(bootstrap_ranks),
        heldout_scores_recomputed_from_observations=True,
        bootstrap_ranks_recomputed_from_frozen_schedule=True,
        stability_composed_from_generated_summaries=success,
        external_observation_bytes_receipt_verified=False,
        projector_training_execution_verified=False,
        source_locked_empirical_dimension_result=False,
        consciousness_dimension_claim_admitted=False,
    )


__all__ = ["ExactWindowDimensionScores", "VerifiedDimensionTimeSeriesExecution", "verified_dimension_time_series_execution"]
