"""Exact defective-capable directed-edge disconnection rank certificate."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_algebraic_riesz_projector import (
        VerifiedAlgebraicRieszProjector,
        _operator_norm_certificate,
        verified_algebraic_riesz_projector,
    )
    from .verified_nonnormal_edge_disconnection_spectral_rank import _mask
    from .verified_rational_contour import ZERO, QComplex, _fraction, _matrix
else:
    from verified_algebraic_riesz_projector import (  # type: ignore[no-redef]
        VerifiedAlgebraicRieszProjector,
        _operator_norm_certificate,
        verified_algebraic_riesz_projector,
    )
    from verified_nonnormal_edge_disconnection_spectral_rank import _mask  # type: ignore[no-redef]
    from verified_rational_contour import (  # type: ignore[no-redef]
        ZERO, QComplex, _fraction, _matrix,
    )


QMatrix = tuple[tuple[QComplex, ...], ...]


def _scaled(matrix: QMatrix, scale: Fraction) -> QMatrix:
    return tuple(tuple(entry / scale for entry in row) for row in matrix)


@dataclass(frozen=True)
class VerifiedDefectiveEdgeDisconnectionSpectralRank:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    algebraic_projector: VerifiedAlgebraicRieszProjector
    normalized_directed_coupling: QMatrix
    normalized_maximum_coupling_norm_upper: Fraction
    normalized_maximum_coupling_frobenius_upper: Fraction
    projector_norm_upper: Fraction
    inside_gap_lower: Fraction
    exterior_reciprocal_gap_lower: Fraction
    disconnected_resolvent_norm_upper: Fraction | None
    neumann_product_upper: Fraction | None
    neumann_margin: Fraction | None
    projector_perturbation_norm_upper: Fraction | None
    nominal_fixed_circle_rank: int | None
    fixed_circle_rank_preservation_admitted: bool
    robust_candidate_band_verified: bool
    candidate_band: tuple[int, int]
    diagonalization_witness_required: bool
    defective_operator_admitted: bool
    eigenvalue_distance_only_certificate_admitted: bool
    empirical_operator_provenance_verified: bool
    consciousness_dimension_claim_admitted: bool


def verified_defective_edge_disconnection_spectral_rank(
    nominal_disconnected_transition: object,
    *,
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
) -> VerifiedDefectiveEdgeDisconnectionSpectralRank:
    """Compose an algebraic Riesz split with a directed-edge Neumann homotopy."""
    raw_transition = _matrix(nominal_disconnected_transition, "nominal_disconnected_transition")
    size = len(raw_transition)
    coupling_raw = _matrix(edge_coupling_operator, "edge_coupling_operator")
    if len(coupling_raw) != size or any(len(row) != size for row in coupling_raw):
        raise ValueError("edge_coupling_operator must shape-match transition")
    if any(coupling_raw[i][i] != ZERO for i in range(size)):
        raise ValueError("edge_coupling_operator diagonal must be zero")
    mask_zero = _mask(disconnected_edge_mask, "disconnected_edge_mask", size)
    mask_one = _mask(maximum_connected_edge_mask, "maximum_connected_edge_mask", size)
    if any(mask_zero[i][j] > mask_one[i][j] for i in range(size) for j in range(size)):
        raise ValueError("disconnected edge mask may remove but not add directed edges")
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            added = bool(mask_one[i][j] and not mask_zero[i][j])
            if (coupling_raw[i][j] != ZERO) != added:
                raise ValueError("directed coupling support must equal added edge-mask support")
    maximum = _fraction(maximum_coupling_parameter, "maximum_coupling_parameter")
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if maximum <= 0 or scale <= 0:
        raise ValueError("maximum coupling parameter and spectral scale must be positive")
    coupling = _scaled(coupling_raw, scale)
    coupling_certificate = _operator_norm_certificate(coupling, sqrt_precision)
    exact_frobenius_squared = sum(
        (entry.abs_squared() for row in coupling for entry in row), Fraction(0)
    ) * maximum * maximum
    supplied_coupling_upper = _fraction(
        normalized_maximum_coupling_norm_upper,
        "normalized_maximum_coupling_norm_upper",
    )
    if supplied_coupling_upper < 0 or supplied_coupling_upper ** 2 < exact_frobenius_squared:
        raise ValueError("normalized maximum coupling norm upper underreports exact Frobenius norm")
    algebraic = verified_algebraic_riesz_projector(
        nominal_disconnected_transition,
        projector=projector,
        exterior_centered_inverse=exterior_centered_inverse,
        center=center,
        radius=radius,
        spectral_reference_scale=scale,
        sqrt_precision=sqrt_precision,
    )
    projector_norm = _operator_norm_certificate(
        _matrix(projector, "projector"), sqrt_precision
    ).selected_two_norm_upper
    inside_gap = algebraic.normalized_inside_margin
    exterior_gap = algebraic.exterior_reciprocal_margin
    failures = list(algebraic.failure_codes)
    resolvent_upper = None
    product = None
    margin = None
    projector_movement = None
    preserved = False
    if algebraic.projector_verified and inside_gap > 0 and exterior_gap > 0:
        resolvent_upper = (
            projector_norm / inside_gap
            + algebraic.exterior_inverse_norm.selected_two_norm_upper / exterior_gap
        )
        product = resolvent_upper * supplied_coupling_upper
        margin = 1 - product
        preserved = margin > 0
        if not preserved:
            failures.append("DEFECTIVE_EDGE_NEUMANN_MARGIN_NONPOSITIVE")
        else:
            projector_movement = (
                algebraic.normalized_radius
                * resolvent_upper * resolvent_upper
                * supplied_coupling_upper / margin
            )
    elif not failures:
        failures.append("DEFECTIVE_EDGE_BASE_RESOLVENT_UNAVAILABLE")
    if (
        not isinstance(candidate_band, (tuple, list)) or len(candidate_band) != 2
        or any(type(value) is not int for value in candidate_band)
        or candidate_band[0] < 0 or candidate_band[0] > candidate_band[1]
    ):
        raise ValueError("candidate_band must be two ordered nonnegative built-in integers")
    band = (candidate_band[0], candidate_band[1])
    rank = algebraic.projector_rank
    success = preserved and rank is not None
    status = (
        "VERIFIED_DEFECTIVE_DIRECTED_EDGE_DISCONNECTION_RANK_PRESERVATION"
        if success else (failures[0] if failures else "DEFECTIVE_EDGE_RANK_UNRESOLVED")
    )
    return VerifiedDefectiveEdgeDisconnectionSpectralRank(
        status=status,
        validation_level=status if success else None,
        failure_codes=tuple(failures),
        algebraic_projector=algebraic,
        normalized_directed_coupling=coupling,
        normalized_maximum_coupling_norm_upper=supplied_coupling_upper,
        normalized_maximum_coupling_frobenius_upper=maximum * coupling_certificate.frobenius_upper,
        projector_norm_upper=projector_norm,
        inside_gap_lower=inside_gap,
        exterior_reciprocal_gap_lower=exterior_gap,
        disconnected_resolvent_norm_upper=resolvent_upper,
        neumann_product_upper=product,
        neumann_margin=margin,
        projector_perturbation_norm_upper=projector_movement,
        nominal_fixed_circle_rank=rank,
        fixed_circle_rank_preservation_admitted=success,
        robust_candidate_band_verified=(success and band[0] <= rank <= band[1]),
        candidate_band=band,
        diagonalization_witness_required=False,
        defective_operator_admitted=success,
        eigenvalue_distance_only_certificate_admitted=False,
        empirical_operator_provenance_verified=False,
        consciousness_dimension_claim_admitted=False,
    )


__all__ = [
    "VerifiedDefectiveEdgeDisconnectionSpectralRank",
    "verified_defective_edge_disconnection_spectral_rank",
]
