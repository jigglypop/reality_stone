"""Split-conformal calibration adapter for interval defective edge rank."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re

if __package__:
    from .verified_interval_defective_edge_disconnection_spectral_rank import (
        VerifiedIntervalDefectiveEdgeDisconnectionSpectralRank,
        verified_interval_defective_edge_disconnection_spectral_rank,
    )
    from .verified_rational_contour import _matrix
    from .verified_split_conformal_matrix_uncertainty_coverage import (
        VerifiedSplitConformalMatrixUncertaintyCoverage,
        verified_split_conformal_matrix_uncertainty_coverage,
    )
    from .verified_edge_metric_effective_dimension import _q
else:
    from verified_interval_defective_edge_disconnection_spectral_rank import (  # type: ignore[no-redef]
        VerifiedIntervalDefectiveEdgeDisconnectionSpectralRank,
        verified_interval_defective_edge_disconnection_spectral_rank,
    )
    from verified_rational_contour import _matrix  # type: ignore[no-redef]
    from verified_split_conformal_matrix_uncertainty_coverage import (  # type: ignore[no-redef]
        VerifiedSplitConformalMatrixUncertaintyCoverage,
        verified_split_conformal_matrix_uncertainty_coverage,
    )
    from verified_edge_metric_effective_dimension import _q  # type: ignore[no-redef]


_LABEL = re.compile(r"^(baseline|coupling)\[(\d+),(\d+)\]$")


@dataclass(frozen=True)
class VerifiedConformalIntervalDefectiveEdgeDisconnectionSpectralRank:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    coverage_certificate: VerifiedSplitConformalMatrixUncertaintyCoverage
    interval_rank_certificate: VerifiedIntervalDefectiveEdgeDisconnectionSpectralRank | None
    baseline_radius_matrix: tuple[tuple[Fraction, ...], ...] | None
    coupling_radius_matrix: tuple[tuple[Fraction, ...], ...] | None
    simultaneous_coverage_lower: Fraction
    conditional_simultaneous_rank_coverage_admitted: bool
    external_source_receipt_verified: bool
    source_locked_empirical_rank_result: bool
    consciousness_dimension_claim_admitted: bool


def verified_conformal_interval_defective_edge_disconnection_spectral_rank(
    nominal_disconnected_transition: object,
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
    disconnected_edge_mask: object,
    maximum_connected_edge_mask: object,
    edge_coupling_operator: object,
    maximum_coupling_parameter: object,
    normalized_maximum_coupling_norm_upper: object,
    projector: object,
    exterior_centered_inverse: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    candidate_band: object,
    sqrt_precision: int = 32,
) -> VerifiedConformalIntervalDefectiveEdgeDisconnectionSpectralRank:
    coverage = verified_split_conformal_matrix_uncertainty_coverage(
        component_labels=component_labels, component_scales=component_scales,
        calibration_absolute_error_vectors=calibration_absolute_error_vectors,
        heldout_absolute_error_vectors=heldout_absolute_error_vectors,
        miscoverage_alpha=miscoverage_alpha,
        heldout_null_coverage_floor=heldout_null_coverage_floor,
        heldout_audit_alpha=heldout_audit_alpha,
        exchangeability_scope=exchangeability_scope,
        heldout_sampling_kind=heldout_sampling_kind,
        provenance_status=provenance_status,
        calibration_data_sha256=calibration_data_sha256,
        heldout_data_sha256=heldout_data_sha256,
        coverage_contract_sha256=coverage_contract_sha256,
    )
    size = len(_matrix(nominal_disconnected_transition, "nominal_disconnected_transition"))
    failures = list(coverage.failure_codes)
    interval = None
    baseline_matrix = None
    coupling_matrix = None
    if coverage.simultaneous_component_radii is not None:
        baseline = [[Fraction(0) for _ in range(size)] for _ in range(size)]
        coupling = [[Fraction(0) for _ in range(size)] for _ in range(size)]
        seen: set[tuple[str, int, int]] = set()
        for label, component_radius in zip(
            coverage.component_labels, coverage.simultaneous_component_radii, strict=True
        ):
            match = _LABEL.fullmatch(label)
            if match is None:
                raise ValueError("component labels must use baseline[i,j] or coupling[i,j]")
            family, i_text, j_text = match.groups()
            i, j = int(i_text), int(j_text)
            if i >= size or j >= size:
                raise ValueError("component label matrix index exceeds transition size")
            key = (family, i, j)
            if key in seen:
                raise ValueError("component labels may not duplicate a matrix entry")
            seen.add(key)
            (baseline if family == "baseline" else coupling)[i][j] = component_radius
        baseline_matrix = tuple(tuple(row) for row in baseline)
        coupling_matrix = tuple(tuple(row) for row in coupling)
        scale = _q(spectral_reference_scale, "spectral_reference_scale")
        maximum = _q(maximum_coupling_parameter, "maximum_coupling_parameter")
        baseline_upper = sum((item for row in baseline_matrix for item in row), Fraction(0)) / scale
        coupling_upper = maximum * sum((item for row in coupling_matrix for item in row), Fraction(0)) / scale
        interval = verified_interval_defective_edge_disconnection_spectral_rank(
            nominal_disconnected_transition,
            disconnected_edge_mask=disconnected_edge_mask,
            maximum_connected_edge_mask=maximum_connected_edge_mask,
            edge_coupling_operator=edge_coupling_operator,
            maximum_coupling_parameter=maximum_coupling_parameter,
            normalized_maximum_coupling_norm_upper=normalized_maximum_coupling_norm_upper,
            baseline_operator_uncertainty_radius=baseline_matrix,
            normalized_baseline_uncertainty_norm_upper=baseline_upper,
            edge_coupling_uncertainty_radius=coupling_matrix,
            normalized_maximum_coupling_uncertainty_norm_upper=coupling_upper,
            projector=projector, exterior_centered_inverse=exterior_centered_inverse,
            center=center, radius=radius,
            spectral_reference_scale=spectral_reference_scale,
            candidate_band=candidate_band, sqrt_precision=sqrt_precision,
        )
        failures.extend(interval.failure_codes)
    success = bool(
        coverage.validation_level is not None
        and interval is not None
        and interval.validation_level is not None
    )
    if not success and not failures:
        failures.append("CONFORMAL_INTERVAL_DEFECTIVE_COMPOSITION_FAILED")
    status = (
        "VERIFIED_CONDITIONAL_CONFORMAL_INTERVAL_DEFECTIVE_EDGE_RANK"
        if success else failures[0]
    )
    return VerifiedConformalIntervalDefectiveEdgeDisconnectionSpectralRank(
        status=status, validation_level=status if success else None,
        failure_codes=tuple(dict.fromkeys(failures)),
        coverage_certificate=coverage, interval_rank_certificate=interval,
        baseline_radius_matrix=baseline_matrix, coupling_radius_matrix=coupling_matrix,
        simultaneous_coverage_lower=coverage.finite_sample_coverage_lower,
        conditional_simultaneous_rank_coverage_admitted=success,
        external_source_receipt_verified=False,
        source_locked_empirical_rank_result=False,
        consciousness_dimension_claim_admitted=False,
    )


__all__ = [
    "VerifiedConformalIntervalDefectiveEdgeDisconnectionSpectralRank",
    "verified_conformal_interval_defective_edge_disconnection_spectral_rank",
]
