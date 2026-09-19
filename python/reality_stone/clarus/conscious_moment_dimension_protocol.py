"""Held-out identification protocol for a candidate 4--6 neural signal rank."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .quantitative_graph_transform import _exact_fraction
else:
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]


FROZEN_DIMENSION_MENU = (1, 2, 3, 4, 5, 6, 8, 10, 12)
SOURCE_LOCKED_EMPIRICAL = "SOURCE_LOCKED_HELD_OUT_EMPIRICAL"
SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"
SIGNAL_SPECTRUM_KIND = "CROSS_VALIDATED_SIGNAL_COVARIANCE"
PERMUTATION_SCOPE = "MAX_OVER_FROZEN_MENU"


@dataclass(frozen=True)
class NeuralSignalDimensionSessionMetrics:
    recorded_unit_count: int
    positive_spectrum_rank: int
    participation_ratio: Fraction
    ridge_effective_dimension: Fraction
    stable_rank: Fraction


@dataclass(frozen=True)
class HeldOutDimensionSelectionGate:
    selected_dimension: int | None
    runner_up_dimension: int | None
    mean_score_by_dimension: tuple[tuple[int, Fraction], ...]
    selected_minus_runner_mean: Fraction | None
    selected_minus_runner_standard_error_squared: Fraction | None
    exceeds_two_standard_errors_strictly: bool


@dataclass(frozen=True)
class ConsciousMomentDimensionProtocolCertificate:
    status: str
    validation_level: str | None
    claim_scope: str
    failure_codes: tuple[str, ...]
    robust_interior: bool
    provenance_status: str
    spectrum_kind: str
    permutation_scope: str
    frozen_dimension_menu: tuple[int, ...]
    candidate_band: tuple[int, int]
    estimator_agreement_tolerance: Fraction
    ridge_regularization: Fraction
    rank_tolerance: Fraction
    session_metrics: tuple[NeuralSignalDimensionSessionMetrics, ...]
    conscious_selection: HeldOutDimensionSelectionGate
    conscious_minus_control_selection: HeldOutDimensionSelectionGate
    permutation_p_upper: Fraction
    permutation_alpha: Fraction
    selected_signal_rank: int | None
    candidate_4_6_signal_rank_supported_by_supplied_summary: bool
    external_source_receipt_verified: bool
    source_locked_empirical_signal_rank_result: bool
    consciousness_dimension_claim_admitted: bool
    neural_signal_rank_is_not_manifold_or_consciousness_dimension: bool


def _fraction_row(values: object, name: str) -> tuple[Fraction, ...]:
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{name} must be a tuple or list")
    row = tuple(_exact_fraction(value, f"{name}[{i}]") for i, value in enumerate(values))
    return row


def _score_rows(values: object, name: str) -> dict[int, tuple[Fraction, ...]]:
    if not isinstance(values, dict) or set(values) != set(FROZEN_DIMENSION_MENU):
        raise ValueError(f"{name} must contain exactly the frozen dimension menu")
    rows = {dimension: _fraction_row(values[dimension], f"{name}[{dimension}]") for dimension in FROZEN_DIMENSION_MENU}
    lengths = {len(row) for row in rows.values()}
    if len(lengths) != 1 or next(iter(lengths), 0) < 3:
        raise ValueError(f"{name} rows must share at least three held-out folds")
    return rows


def _mean(row: tuple[Fraction, ...]) -> Fraction:
    return sum(row, Fraction(0)) / len(row)


def _selection_gate(rows: dict[int, tuple[Fraction, ...]]) -> HeldOutDimensionSelectionGate:
    means = {dimension: _mean(row) for dimension, row in rows.items()}
    top = max(means.values())
    winners = tuple(dimension for dimension, value in means.items() if value == top)
    ordered_means = tuple((dimension, means[dimension]) for dimension in FROZEN_DIMENSION_MENU)
    if len(winners) != 1:
        return HeldOutDimensionSelectionGate(
            selected_dimension=None, runner_up_dimension=None,
            mean_score_by_dimension=ordered_means,
            selected_minus_runner_mean=None,
            selected_minus_runner_standard_error_squared=None,
            exceeds_two_standard_errors_strictly=False,
        )
    selected = winners[0]
    runner_value = max(value for dimension, value in means.items() if dimension != selected)
    runners = tuple(dimension for dimension, value in means.items() if dimension != selected and value == runner_value)
    if len(runners) != 1:
        return HeldOutDimensionSelectionGate(
            selected_dimension=selected, runner_up_dimension=None,
            mean_score_by_dimension=ordered_means,
            selected_minus_runner_mean=None,
            selected_minus_runner_standard_error_squared=None,
            exceeds_two_standard_errors_strictly=False,
        )
    runner = runners[0]
    differences = tuple(a - b for a, b in zip(rows[selected], rows[runner]))
    mean_difference = _mean(differences)
    sample_variance = sum(
        (value - mean_difference) ** 2 for value in differences
    ) / (len(differences) - 1)
    se_squared = sample_variance / len(differences)
    strict = mean_difference > 0 and mean_difference**2 > 4 * se_squared
    return HeldOutDimensionSelectionGate(
        selected_dimension=selected, runner_up_dimension=runner,
        mean_score_by_dimension=ordered_means,
        selected_minus_runner_mean=mean_difference,
        selected_minus_runner_standard_error_squared=se_squared,
        exceeds_two_standard_errors_strictly=strict,
    )


def conscious_moment_dimension_identification_protocol(
    *,
    signal_covariance_spectra: object,
    ridge_regularization: object,
    rank_tolerance: object,
    estimator_agreement_tolerance: object,
    conscious_fold_scores_by_dimension: object,
    control_fold_scores_by_dimension: object,
    permutation_exceedances: int,
    permutation_repetitions: int,
    permutation_alpha: object,
    provenance_status: str,
    spectrum_kind: str,
    permutation_scope: str,
) -> ConsciousMomentDimensionProtocolCertificate:
    """Audit a candidate neural signal rank without identifying consciousness itself."""
    if provenance_status not in (SYNTHETIC_FIXTURE, SOURCE_LOCKED_EMPIRICAL):
        raise ValueError("provenance_status must be a frozen protocol value")
    if not isinstance(spectrum_kind, str) or not isinstance(permutation_scope, str):
        raise ValueError("spectrum_kind and permutation_scope must be strings")
    ridge = _exact_fraction(ridge_regularization, "ridge_regularization")
    rank_tol = _exact_fraction(rank_tolerance, "rank_tolerance")
    agreement_tol = _exact_fraction(estimator_agreement_tolerance, "estimator_agreement_tolerance")
    alpha = _exact_fraction(permutation_alpha, "permutation_alpha")
    if ridge <= 0 or rank_tol < 0 or agreement_tol < 0 or alpha <= 0 or alpha >= 1:
        raise ValueError("ridge must be positive; tolerances nonnegative; alpha strictly between zero and one")
    if type(permutation_exceedances) is not int or type(permutation_repetitions) is not int:
        raise ValueError("permutation counts must be built-in integers")
    if permutation_repetitions < 1 or permutation_exceedances < 0 or permutation_exceedances > permutation_repetitions:
        raise ValueError("permutation counts are inconsistent")
    if not isinstance(signal_covariance_spectra, (tuple, list)) or len(signal_covariance_spectra) < 3:
        raise ValueError("at least three independent signal spectra are required")
    spectra = tuple(
        _fraction_row(row, f"signal_covariance_spectra[{index}]")
        for index, row in enumerate(signal_covariance_spectra)
    )
    widths = {len(row) for row in spectra}
    if len(widths) != 1 or next(iter(widths), 0) < max(FROZEN_DIMENSION_MENU):
        raise ValueError("spectra must share at least twelve recorded dimensions")
    if any(min(row) < 0 or sum(row) <= 0 for row in spectra):
        raise ValueError("signal spectra must be nonnegative with positive trace")

    conscious_rows = _score_rows(conscious_fold_scores_by_dimension, "conscious_fold_scores_by_dimension")
    control_rows = _score_rows(control_fold_scores_by_dimension, "control_fold_scores_by_dimension")
    if {len(row) for row in conscious_rows.values()} != {len(row) for row in control_rows.values()}:
        raise ValueError("conscious and control score rows must use the same fold count")
    contrast_rows = {
        dimension: tuple(a - b for a, b in zip(conscious_rows[dimension], control_rows[dimension]))
        for dimension in FROZEN_DIMENSION_MENU
    }
    conscious_gate = _selection_gate(conscious_rows)
    contrast_gate = _selection_gate(contrast_rows)

    metrics: list[NeuralSignalDimensionSessionMetrics] = []
    for spectrum in spectra:
        trace = sum(spectrum, Fraction(0))
        square_trace = sum(value**2 for value in spectrum)
        maximum = max(spectrum)
        metrics.append(NeuralSignalDimensionSessionMetrics(
            recorded_unit_count=len(spectrum),
            positive_spectrum_rank=sum(value > rank_tol for value in spectrum),
            participation_ratio=trace**2 / square_trace,
            ridge_effective_dimension=sum(value / (value + ridge) for value in spectrum),
            stable_rank=trace / maximum,
        ))

    p_upper = Fraction(permutation_exceedances + 1, permutation_repetitions + 1)
    failures: list[str] = []
    if spectrum_kind != SIGNAL_SPECTRUM_KIND:
        failures.append("DIMENSION_PROTOCOL_SPECTRUM_NOT_CROSS_VALIDATED_SIGNAL")
    if permutation_scope != PERMUTATION_SCOPE:
        failures.append("DIMENSION_PROTOCOL_PERMUTATION_NOT_MAX_OVER_FROZEN_MENU")
    if conscious_gate.selected_dimension is None:
        failures.append("DIMENSION_PROTOCOL_CONSCIOUS_HELDOUT_WINNER_NOT_UNIQUE")
    elif not conscious_gate.exceeds_two_standard_errors_strictly:
        failures.append("DIMENSION_PROTOCOL_CONSCIOUS_HELDOUT_ADVANTAGE_NOT_ABOVE_TWO_SE")
    if contrast_gate.selected_dimension is None:
        failures.append("DIMENSION_PROTOCOL_STATE_CONTRAST_WINNER_NOT_UNIQUE")
    elif not contrast_gate.exceeds_two_standard_errors_strictly:
        failures.append("DIMENSION_PROTOCOL_STATE_CONTRAST_NOT_ABOVE_TWO_SE")
    selected = conscious_gate.selected_dimension
    if selected is not None and contrast_gate.selected_dimension != selected:
        failures.append("DIMENSION_PROTOCOL_CONSCIOUS_AND_CONTRAST_WINNERS_DISAGREE")
    if selected is not None and not 4 <= selected <= 6:
        failures.append("DIMENSION_PROTOCOL_SELECTED_RANK_OUTSIDE_FROZEN_4_6_BAND")
    if selected is not None:
        lower = Fraction(selected) - agreement_tol
        upper = Fraction(selected) + agreement_tol
        for index, metric in enumerate(metrics):
            if metric.positive_spectrum_rank != selected:
                failures.append(f"DIMENSION_PROTOCOL_SESSION_{index}_HARD_RANK_DISAGREES")
            if not lower <= metric.participation_ratio <= upper:
                failures.append(f"DIMENSION_PROTOCOL_SESSION_{index}_PARTICIPATION_RATIO_DISAGREES")
            if not lower <= metric.ridge_effective_dimension <= upper:
                failures.append(f"DIMENSION_PROTOCOL_SESSION_{index}_RIDGE_DIMENSION_DISAGREES")
            if not lower <= metric.stable_rank <= upper:
                failures.append(f"DIMENSION_PROTOCOL_SESSION_{index}_STABLE_RANK_DISAGREES")
    if p_upper > alpha:
        failures.append("DIMENSION_PROTOCOL_MAX_MENU_PERMUTATION_NOT_SIGNIFICANT")

    support = not failures
    # The current protocol receives summaries but no authenticated external
    # observation receipt.  A caller-selected provenance string is not proof
    # that spectra and scores came from the pinned source, so empirical
    # promotion remains closed until a receipt-bearing adapter is added.
    external_source_receipt_verified = False
    empirical = False
    scope = (
        "HELD_OUT_NEURAL_SIGNAL_RANK_PROTOCOL; A_SELECTED_RANK_IS_NOT_AN_"
        "AMBIENT_MANIFOLD_PHENOMENOLOGICAL_OR_CONSCIOUSNESS_DIMENSION"
    )
    common = dict(
        claim_scope=scope, failure_codes=tuple(failures),
        provenance_status=provenance_status, spectrum_kind=spectrum_kind,
        permutation_scope=permutation_scope,
        frozen_dimension_menu=FROZEN_DIMENSION_MENU, candidate_band=(4, 6),
        estimator_agreement_tolerance=agreement_tol,
        ridge_regularization=ridge, rank_tolerance=rank_tol,
        session_metrics=tuple(metrics), conscious_selection=conscious_gate,
        conscious_minus_control_selection=contrast_gate,
        permutation_p_upper=p_upper, permutation_alpha=alpha,
        selected_signal_rank=selected,
        candidate_4_6_signal_rank_supported_by_supplied_summary=support,
        external_source_receipt_verified=external_source_receipt_verified,
        source_locked_empirical_signal_rank_result=empirical,
        consciousness_dimension_claim_admitted=False,
        neural_signal_rank_is_not_manifold_or_consciousness_dimension=True,
    )
    if failures:
        return ConsciousMomentDimensionProtocolCertificate(
            status=failures[0], validation_level=None,
            robust_interior=False, **common,
        )
    status = (
        "VALIDATED_DECLARED_SOURCE_LOCKED_4_6_SIGNAL_RANK_SUMMARY_ONLY"
        if provenance_status == SOURCE_LOCKED_EMPIRICAL
        else "VALIDATED_SYNTHETIC_4_6_DIMENSION_IDENTIFICATION_APPARATUS"
    )
    return ConsciousMomentDimensionProtocolCertificate(
        status=status, validation_level=status,
        robust_interior=False, **common,
    )


__all__ = [
    "ConsciousMomentDimensionProtocolCertificate",
    "FROZEN_DIMENSION_MENU",
    "HeldOutDimensionSelectionGate",
    "NeuralSignalDimensionSessionMetrics",
    "PERMUTATION_SCOPE",
    "SIGNAL_SPECTRUM_KIND",
    "SOURCE_LOCKED_EMPIRICAL",
    "SYNTHETIC_FIXTURE",
    "conscious_moment_dimension_identification_protocol",
]
