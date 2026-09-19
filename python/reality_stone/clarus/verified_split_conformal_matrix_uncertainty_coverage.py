"""Split-conformal simultaneous uncertainty radii for finite matrix families."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb

if __package__:
    from .quantitative_graph_transform import _exact_fraction
else:
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]


SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"
SOURCE_LOCKED_EMPIRICAL = "SOURCE_LOCKED_HELD_OUT_EMPIRICAL"
EXCHANGEABILITY_SCOPE = "CALIBRATION_AND_ONE_FUTURE_VECTOR_EXCHANGEABLE"
HELDOUT_SAMPLING_KIND = "INDEPENDENT_BERNOULLI_COVERAGE_AUDIT"


@dataclass(frozen=True)
class VerifiedSplitConformalMatrixUncertaintyCoverage:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    provenance_status: str
    component_labels: tuple[str, ...]
    component_scales: tuple[Fraction, ...]
    calibration_vector_count: int
    heldout_vector_count: int
    miscoverage_alpha: Fraction
    conformal_order_index: int
    finite_sample_coverage_lower: Fraction
    calibration_max_scores: tuple[Fraction, ...]
    conformal_score_threshold: Fraction | None
    simultaneous_component_radii: tuple[Fraction, ...] | None
    heldout_max_scores: tuple[Fraction, ...]
    heldout_covered_count: int
    heldout_empirical_coverage: Fraction
    heldout_null_coverage_floor: Fraction
    heldout_undercoverage_audit_p_upper: Fraction
    heldout_audit_alpha: Fraction
    exchangeability_scope: str
    heldout_sampling_kind: str
    simultaneous_family_coverage_verified_conditionally: bool
    heldout_undercoverage_falsifier_rejected: bool
    external_source_receipt_verified: bool
    source_locked_empirical_coverage_result: bool
    consciousness_claim_admitted: bool
    dimension_4_6_claim_admitted: bool


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")
    return value


def _rows(value: object, name: str, width: int) -> tuple[tuple[Fraction, ...], ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{name} must be a nonempty sequence")
    result = []
    for i, row in enumerate(value):
        if not isinstance(row, (tuple, list)) or len(row) != width:
            raise ValueError(f"{name}[{i}] must match the component family")
        parsed = tuple(_exact_fraction(item, f"{name}[{i}][{j}]") for j, item in enumerate(row))
        if min(parsed) < 0:
            raise ValueError("absolute errors must be nonnegative")
        result.append(parsed)
    return tuple(result)


def _ceil_fraction(value: Fraction) -> int:
    return (value.numerator + value.denominator - 1) // value.denominator


def _binomial_upper_tail(successes: int, trials: int, probability: Fraction) -> Fraction:
    return sum(
        (Fraction(comb(trials, j)) * probability**j * (1 - probability) ** (trials - j) for j in range(successes, trials + 1)),
        Fraction(0),
    )


def verified_split_conformal_matrix_uncertainty_coverage(
    *,
    component_labels: object,
    component_scales: object,
    calibration_absolute_error_vectors: object,
    heldout_absolute_error_vectors: object,
    miscoverage_alpha: object,
    heldout_null_coverage_floor: object,
    heldout_audit_alpha: object,
    exchangeability_scope: str,
    heldout_sampling_kind: str,
    provenance_status: str,
    calibration_data_sha256: object,
    heldout_data_sha256: object,
    coverage_contract_sha256: object,
) -> VerifiedSplitConformalMatrixUncertaintyCoverage:
    if provenance_status not in (SYNTHETIC_FIXTURE, SOURCE_LOCKED_EMPIRICAL):
        raise ValueError("provenance_status must be a frozen coverage value")
    if not isinstance(component_labels, (tuple, list)) or not component_labels:
        raise ValueError("component_labels must be a nonempty sequence")
    labels = tuple(component_labels)
    if any(not isinstance(label, str) or not label or label.strip() != label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("component labels must be unique nonempty canonical strings")
    if not isinstance(component_scales, (tuple, list)) or len(component_scales) != len(labels):
        raise ValueError("component_scales must match component_labels")
    scales = tuple(_exact_fraction(value, f"component_scales[{i}]") for i, value in enumerate(component_scales))
    if min(scales) <= 0:
        raise ValueError("component scales must be positive")
    calibration = _rows(calibration_absolute_error_vectors, "calibration_absolute_error_vectors", len(labels))
    heldout = _rows(heldout_absolute_error_vectors, "heldout_absolute_error_vectors", len(labels))
    alpha = _exact_fraction(miscoverage_alpha, "miscoverage_alpha")
    audit_floor = _exact_fraction(heldout_null_coverage_floor, "heldout_null_coverage_floor")
    audit_alpha = _exact_fraction(heldout_audit_alpha, "heldout_audit_alpha")
    if not 0 < alpha < 1 or not 0 < audit_floor < 1 or not 0 < audit_alpha < 1:
        raise ValueError("coverage probabilities must be strictly between zero and one")
    if audit_floor >= 1 - alpha:
        raise ValueError("heldout null coverage floor must be below the conformal target")
    _hash(calibration_data_sha256, "calibration_data_sha256")
    _hash(heldout_data_sha256, "heldout_data_sha256")
    _hash(coverage_contract_sha256, "coverage_contract_sha256")

    calibration_scores = tuple(max(error / scale for error, scale in zip(row, scales, strict=True)) for row in calibration)
    heldout_scores = tuple(max(error / scale for error, scale in zip(row, scales, strict=True)) for row in heldout)
    n = len(calibration_scores)
    order = _ceil_fraction(Fraction(n + 1) * (1 - alpha))
    finite_lower = Fraction(order, n + 1)
    failures: list[str] = []
    if exchangeability_scope != EXCHANGEABILITY_SCOPE:
        failures.append("CONFORMAL_COVERAGE_EXCHANGEABILITY_SCOPE_MISMATCH")
    if heldout_sampling_kind != HELDOUT_SAMPLING_KIND:
        failures.append("CONFORMAL_COVERAGE_HELDOUT_SAMPLING_KIND_MISMATCH")
    if order > n:
        failures.append("CONFORMAL_COVERAGE_CALIBRATION_COUNT_TOO_SMALL_FOR_ALPHA")
    threshold = sorted(calibration_scores)[order - 1] if order <= n else None
    radii = tuple(scale * threshold for scale in scales) if threshold is not None else None
    covered = sum(score <= threshold for score in heldout_scores) if threshold is not None else 0
    empirical = Fraction(covered, len(heldout_scores))
    p_upper = _binomial_upper_tail(covered, len(heldout_scores), audit_floor)
    falsifier_rejected = p_upper <= audit_alpha
    if not falsifier_rejected:
        failures.append("CONFORMAL_COVERAGE_HELDOUT_UNDERCOVERAGE_FALSIFIER_NOT_REJECTED")
    conditional = not failures
    external = False
    source_locked = False
    common = dict(
        failure_codes=tuple(failures), provenance_status=provenance_status,
        component_labels=labels, component_scales=scales,
        calibration_vector_count=n, heldout_vector_count=len(heldout_scores),
        miscoverage_alpha=alpha, conformal_order_index=order,
        finite_sample_coverage_lower=finite_lower,
        calibration_max_scores=calibration_scores,
        conformal_score_threshold=threshold,
        simultaneous_component_radii=radii,
        heldout_max_scores=heldout_scores, heldout_covered_count=covered,
        heldout_empirical_coverage=empirical,
        heldout_null_coverage_floor=audit_floor,
        heldout_undercoverage_audit_p_upper=p_upper,
        heldout_audit_alpha=audit_alpha,
        exchangeability_scope=exchangeability_scope,
        heldout_sampling_kind=heldout_sampling_kind,
        simultaneous_family_coverage_verified_conditionally=conditional,
        heldout_undercoverage_falsifier_rejected=falsifier_rejected,
        external_source_receipt_verified=external,
        source_locked_empirical_coverage_result=source_locked,
        consciousness_claim_admitted=False, dimension_4_6_claim_admitted=False,
    )
    if failures:
        return VerifiedSplitConformalMatrixUncertaintyCoverage(
            status=failures[0], validation_level=None, **common
        )
    status = (
        "VALIDATED_DECLARED_SOURCE_LOCKED_SPLIT_CONFORMAL_COVERAGE_APPARATUS_ONLY"
        if provenance_status == SOURCE_LOCKED_EMPIRICAL
        else "VERIFIED_SYNTHETIC_SPLIT_CONFORMAL_SIMULTANEOUS_COVERAGE"
    )
    return VerifiedSplitConformalMatrixUncertaintyCoverage(
        status=status, validation_level=status, **common
    )


__all__ = [
    "EXCHANGEABILITY_SCOPE", "HELDOUT_SAMPLING_KIND", "SOURCE_LOCKED_EMPIRICAL",
    "SYNTHETIC_FIXTURE", "VerifiedSplitConformalMatrixUncertaintyCoverage",
    "verified_split_conformal_matrix_uncertainty_coverage",
]
