"""Join split-conformal matrix radii to the interval characteristic split theorem."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_interval_characteristic_spectral_split import (
        VerifiedIntervalCharacteristicSpectralSplit,
        verified_interval_characteristic_spectral_split,
    )
    from .verified_rational_contour import _matrix
    from .verified_split_conformal_matrix_uncertainty_coverage import (
        VerifiedSplitConformalMatrixUncertaintyCoverage,
    )
else:
    from verified_interval_characteristic_spectral_split import (  # type: ignore[no-redef]
        VerifiedIntervalCharacteristicSpectralSplit,
        verified_interval_characteristic_spectral_split,
    )
    from verified_rational_contour import _matrix  # type: ignore[no-redef]
    from verified_split_conformal_matrix_uncertainty_coverage import (  # type: ignore[no-redef]
        VerifiedSplitConformalMatrixUncertaintyCoverage,
    )


@dataclass(frozen=True)
class VerifiedConformalIntervalCharacteristicBridge:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    coverage_certificate: VerifiedSplitConformalMatrixUncertaintyCoverage
    interval_characteristic_split: VerifiedIntervalCharacteristicSpectralSplit | None
    matrix_dimension: int
    expected_component_labels: tuple[str, ...]
    component_family_complete_and_ordered: bool
    conformal_uncertainty_radii: tuple[tuple[tuple[Fraction, Fraction], ...], ...] | None
    conditional_simultaneous_matrix_coverage_lower: Fraction
    conditional_covered_family_rank: int | None
    conformal_coverage_and_interval_rank_composed: bool
    external_source_receipt_verified: bool
    source_locked_empirical_rank_result: bool
    consciousness_claim_admitted: bool
    dimension_4_6_claim_admitted: bool


def _labels(n: int) -> tuple[str, ...]:
    return tuple(
        f"A[{i},{j}].{part}"
        for i in range(n) for j in range(n) for part in ("real", "imag")
    )


def verified_conformal_interval_characteristic_bridge(
    nominal_transition: object,
    *,
    coverage_certificate: VerifiedSplitConformalMatrixUncertaintyCoverage,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    maximum_partitions: int = 1024,
    maximum_factor_candidate_value_tuples: int = 1_000_000,
    nodes: int = 4,
    sqrt_precision: int = 48,
) -> VerifiedConformalIntervalCharacteristicBridge:
    if not isinstance(coverage_certificate, VerifiedSplitConformalMatrixUncertaintyCoverage):
        raise ValueError("coverage_certificate must be canonical")
    matrix = _matrix(nominal_transition, "nominal_transition")
    n = len(matrix)
    expected = _labels(n)
    complete = coverage_certificate.component_labels == expected
    failures: list[str] = []
    if coverage_certificate.validation_level is None:
        failures.append("CONFORMAL_INTERVAL_BRIDGE_COVERAGE_NOT_VALIDATED")
    if not complete:
        failures.append("CONFORMAL_INTERVAL_BRIDGE_COMPONENT_FAMILY_INCOMPLETE_OR_UNORDERED")
    radii = None
    interval = None
    if complete and coverage_certificate.simultaneous_component_radii is not None:
        flat = coverage_certificate.simultaneous_component_radii
        radii = tuple(
            tuple((flat[2 * (i * n + j)], flat[2 * (i * n + j) + 1]) for j in range(n))
            for i in range(n)
        )
        interval = verified_interval_characteristic_spectral_split(
            nominal_transition, uncertainty_radii=radii, center=center, radius=radius,
            spectral_reference_scale=spectral_reference_scale,
            maximum_partitions=maximum_partitions,
            maximum_factor_candidate_value_tuples=maximum_factor_candidate_value_tuples,
            nodes=nodes, sqrt_precision=sqrt_precision,
        )
        if interval.validation_level is None:
            failures.append("CONFORMAL_INTERVAL_BRIDGE_IVSPEC_FAILED")
    elif complete:
        failures.append("CONFORMAL_INTERVAL_BRIDGE_RADII_MISSING")
    success = not failures and interval is not None
    status = "VERIFIED_CONDITIONAL_CONFORMAL_INTERVAL_FAMILY_RANK" if success else failures[0]
    return VerifiedConformalIntervalCharacteristicBridge(
        status=status, validation_level=status if success else None,
        failure_codes=tuple(failures), coverage_certificate=coverage_certificate,
        interval_characteristic_split=interval, matrix_dimension=n,
        expected_component_labels=expected,
        component_family_complete_and_ordered=complete,
        conformal_uncertainty_radii=radii,
        conditional_simultaneous_matrix_coverage_lower=coverage_certificate.finite_sample_coverage_lower,
        conditional_covered_family_rank=interval.family_projector_rank if success else None,
        conformal_coverage_and_interval_rank_composed=success,
        external_source_receipt_verified=False,
        source_locked_empirical_rank_result=False,
        consciousness_claim_admitted=False, dimension_4_6_claim_admitted=False,
    )


__all__ = ["VerifiedConformalIntervalCharacteristicBridge", "verified_conformal_interval_characteristic_bridge"]
