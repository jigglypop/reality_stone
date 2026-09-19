"""Interval-robust defective directed-edge disconnection rank certificate."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_defective_edge_disconnection_spectral_rank import (
        VerifiedDefectiveEdgeDisconnectionSpectralRank,
        verified_defective_edge_disconnection_spectral_rank,
    )
    from .verified_edge_metric_effective_dimension import Matrix, _matrix, _q
    from .verified_nonnormal_edge_disconnection_spectral_rank import _mask
else:
    from verified_defective_edge_disconnection_spectral_rank import (  # type: ignore[no-redef]
        VerifiedDefectiveEdgeDisconnectionSpectralRank,
        verified_defective_edge_disconnection_spectral_rank,
    )
    from verified_edge_metric_effective_dimension import (  # type: ignore[no-redef]
        Matrix, _matrix, _q,
    )
    from verified_nonnormal_edge_disconnection_spectral_rank import _mask  # type: ignore[no-redef]


def _nonnegative_radius_matrix(value: object, name: str, size: int) -> Matrix:
    matrix = _matrix(value, name, columns=size)
    if len(matrix) != size or any(item < 0 for row in matrix for item in row):
        raise ValueError(f"{name} must be a square nonnegative radius matrix")
    return matrix


def _frobenius_squared(matrix: Matrix) -> Fraction:
    return sum((item * item for row in matrix for item in row), Fraction(0))


@dataclass(frozen=True)
class VerifiedIntervalDefectiveEdgeDisconnectionSpectralRank:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    nominal_certificate: VerifiedDefectiveEdgeDisconnectionSpectralRank
    normalized_baseline_uncertainty_radius: Matrix
    normalized_coupling_uncertainty_radius: Matrix
    normalized_baseline_uncertainty_norm_upper: Fraction
    normalized_maximum_coupling_uncertainty_norm_upper: Fraction
    total_family_perturbation_norm_upper: Fraction
    interval_neumann_product_upper: Fraction | None
    interval_neumann_margin: Fraction | None
    interval_projector_perturbation_norm_upper: Fraction | None
    fixed_circle_rank: int | None
    entire_interval_family_rank_preserved: bool
    robust_candidate_band_verified: bool
    empirical_interval_coverage_verified: bool
    held_out_intervention_replication_verified: bool
    consciousness_dimension_claim_admitted: bool


def verified_interval_defective_edge_disconnection_spectral_rank(
    nominal_disconnected_transition: object,
    *,
    disconnected_edge_mask: object,
    maximum_connected_edge_mask: object,
    edge_coupling_operator: object,
    maximum_coupling_parameter: object,
    normalized_maximum_coupling_norm_upper: object,
    baseline_operator_uncertainty_radius: object,
    normalized_baseline_uncertainty_norm_upper: object,
    edge_coupling_uncertainty_radius: object,
    normalized_maximum_coupling_uncertainty_norm_upper: object,
    projector: object,
    exterior_centered_inverse: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    candidate_band: object,
    sqrt_precision: int = 32,
) -> VerifiedIntervalDefectiveEdgeDisconnectionSpectralRank:
    """Certify every operator in an entrywise uncertainty family by one base resolvent."""
    nominal = verified_defective_edge_disconnection_spectral_rank(
        nominal_disconnected_transition,
        disconnected_edge_mask=disconnected_edge_mask,
        maximum_connected_edge_mask=maximum_connected_edge_mask,
        edge_coupling_operator=edge_coupling_operator,
        maximum_coupling_parameter=maximum_coupling_parameter,
        normalized_maximum_coupling_norm_upper=normalized_maximum_coupling_norm_upper,
        projector=projector,
        exterior_centered_inverse=exterior_centered_inverse,
        center=center,
        radius=radius,
        spectral_reference_scale=spectral_reference_scale,
        candidate_band=candidate_band,
        sqrt_precision=sqrt_precision,
    )
    size = len(nominal.algebraic_projector.normalized_transition)
    scale = _q(spectral_reference_scale, "spectral_reference_scale")
    maximum = _q(maximum_coupling_parameter, "maximum_coupling_parameter")
    baseline_raw = _nonnegative_radius_matrix(
        baseline_operator_uncertainty_radius, "baseline_operator_uncertainty_radius", size
    )
    coupling_raw = _nonnegative_radius_matrix(
        edge_coupling_uncertainty_radius, "edge_coupling_uncertainty_radius", size
    )
    baseline = tuple(tuple(item / scale for item in row) for row in baseline_raw)
    coupling = tuple(tuple(item / scale for item in row) for row in coupling_raw)
    mask_zero = _mask(disconnected_edge_mask, "disconnected_edge_mask", size)
    mask_one = _mask(maximum_connected_edge_mask, "maximum_connected_edge_mask", size)
    for i in range(size):
        for j in range(size):
            added = i != j and bool(mask_one[i][j] and not mask_zero[i][j])
            if coupling[i][j] and not added:
                raise ValueError("coupling uncertainty support must lie in added directed edges")
    baseline_upper = _q(
        normalized_baseline_uncertainty_norm_upper,
        "normalized_baseline_uncertainty_norm_upper",
    )
    coupling_upper = _q(
        normalized_maximum_coupling_uncertainty_norm_upper,
        "normalized_maximum_coupling_uncertainty_norm_upper",
    )
    if baseline_upper < 0 or baseline_upper ** 2 < _frobenius_squared(baseline):
        raise ValueError("baseline uncertainty norm upper underreports entrywise radius family")
    if coupling_upper < 0 or coupling_upper ** 2 < maximum ** 2 * _frobenius_squared(coupling):
        raise ValueError("coupling uncertainty norm upper underreports entrywise radius family")
    total = (
        baseline_upper
        + nominal.normalized_maximum_coupling_norm_upper
        + coupling_upper
    )
    failures = list(nominal.failure_codes)
    resolvent = nominal.disconnected_resolvent_norm_upper
    product = resolvent * total if resolvent is not None else None
    margin = 1 - product if product is not None else None
    preserved = bool(
        nominal.algebraic_projector.projector_verified
        and product is not None and margin is not None and margin > 0
    )
    if not preserved:
        failures.append("INTERVAL_DEFECTIVE_EDGE_NEUMANN_MARGIN_NONPOSITIVE")
    movement = (
        nominal.algebraic_projector.normalized_radius
        * resolvent * resolvent * total / margin
        if preserved and resolvent is not None and margin is not None else None
    )
    rank = nominal.nominal_fixed_circle_rank
    band = nominal.candidate_band
    status = (
        "VERIFIED_INTERVAL_DEFECTIVE_DIRECTED_EDGE_RANK_PRESERVATION"
        if preserved else failures[0]
    )
    return VerifiedIntervalDefectiveEdgeDisconnectionSpectralRank(
        status=status,
        validation_level=status if preserved else None,
        failure_codes=tuple(dict.fromkeys(failures)),
        nominal_certificate=nominal,
        normalized_baseline_uncertainty_radius=baseline,
        normalized_coupling_uncertainty_radius=coupling,
        normalized_baseline_uncertainty_norm_upper=baseline_upper,
        normalized_maximum_coupling_uncertainty_norm_upper=coupling_upper,
        total_family_perturbation_norm_upper=total,
        interval_neumann_product_upper=product,
        interval_neumann_margin=margin,
        interval_projector_perturbation_norm_upper=movement,
        fixed_circle_rank=rank,
        entire_interval_family_rank_preserved=preserved,
        robust_candidate_band_verified=(
            preserved and rank is not None and band[0] <= rank <= band[1]
        ),
        empirical_interval_coverage_verified=False,
        held_out_intervention_replication_verified=False,
        consciousness_dimension_claim_admitted=False,
    )


__all__ = [
    "VerifiedIntervalDefectiveEdgeDisconnectionSpectralRank",
    "verified_interval_defective_edge_disconnection_spectral_rank",
]
