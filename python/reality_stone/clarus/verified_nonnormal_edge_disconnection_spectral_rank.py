"""Exact diagonalizable-nonnormal disconnection resolvent and rank certificate."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_edge_metric_effective_dimension import (
        Matrix, _add, _determinant, _identity, _inverse, _matrix, _mul,
        _q, _scale, _transpose,
    )
else:
    from verified_edge_metric_effective_dimension import (  # type: ignore[no-redef]
        Matrix, _add, _determinant, _identity, _inverse, _matrix, _mul,
        _q, _scale, _transpose,
    )


def _diagonal(values: tuple[Fraction, ...]) -> Matrix:
    return tuple(tuple(values[i] if i == j else Fraction(0) for j in range(len(values))) for i in range(len(values)))


def _frobenius_squared(matrix: Matrix) -> Fraction:
    return sum((item * item for row in matrix for item in row), Fraction(0))


def _mask(value: object, name: str, size: int) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise ValueError(f"{name} must match operator size")
    rows = []
    for i, row in enumerate(value):
        if not isinstance(row, (tuple, list)) or len(row) != size:
            raise ValueError(f"{name} must be square")
        parsed = []
        for j, item in enumerate(row):
            if type(item) is not int or item not in (0, 1):
                raise ValueError(f"{name} entries must be built-in binary integers")
            if i == j and item:
                raise ValueError(f"{name} diagonal must be zero")
            parsed.append(item)
        rows.append(tuple(parsed))
    return tuple(rows)


@dataclass(frozen=True)
class VerifiedNonnormalEdgeDisconnectionSpectralRank:
    status: str
    validation_level: str
    disconnected_operator: Matrix
    maximum_connected_operator: Matrix
    eigenvector_matrix: Matrix
    inverse_eigenvector_matrix: Matrix
    disconnected_eigenvalues: tuple[Fraction, ...]
    disconnected_operator_is_normal: bool
    fixed_circle_center: Fraction
    fixed_circle_radius: Fraction
    fixed_circle_spectral_gap: Fraction
    eigenvector_condition_frobenius_squared: Fraction
    eigenvector_condition_frobenius_upper: Fraction
    uniform_disconnected_resolvent_norm_upper: Fraction
    maximum_coupling_frobenius_squared: Fraction
    maximum_coupling_norm_upper: Fraction
    neumann_product_upper: Fraction
    neumann_margin: Fraction
    nominal_fixed_circle_rank: int
    fixed_circle_rank_preservation_admitted: bool
    projector_perturbation_norm_upper: Fraction | None
    robust_candidate_band_verified: bool
    candidate_band: tuple[int, int]
    eigenvalue_distance_only_certificate_admitted: bool
    defective_operator_branch_verified: bool
    consciousness_dimension_claim_admitted: bool


def verified_nonnormal_edge_disconnection_spectral_rank(
    *,
    disconnected_edge_mask: object,
    maximum_connected_edge_mask: object,
    disconnected_eigenvalues: object,
    eigenvector_matrix: object,
    edge_coupling_operator: object,
    maximum_coupling_parameter: object,
    fixed_circle_center: object,
    fixed_circle_radius: object,
    eigenvector_condition_frobenius_upper: object,
    maximum_coupling_norm_upper: object,
    candidate_band: object,
) -> VerifiedNonnormalEdgeDisconnectionSpectralRank:
    if not isinstance(disconnected_eigenvalues, (tuple, list)) or not disconnected_eigenvalues:
        raise ValueError("disconnected_eigenvalues must be a nonempty sequence")
    eigenvalues = tuple(_q(value, "disconnected_eigenvalue") for value in disconnected_eigenvalues)
    size = len(eigenvalues)
    basis = _matrix(eigenvector_matrix, "eigenvector_matrix", columns=size)
    if len(basis) != size or _determinant(basis) == 0:
        raise ValueError("eigenvector_matrix must be exact square invertible")
    inverse_basis = _inverse(basis)
    operator_zero = _mul(_mul(basis, _diagonal(eigenvalues)), inverse_basis)
    coupling = _matrix(edge_coupling_operator, "edge_coupling_operator", columns=size)
    if len(coupling) != size or any(coupling[i][i] for i in range(size)):
        raise ValueError("edge_coupling_operator must match size and have zero diagonal")
    mask_zero = _mask(disconnected_edge_mask, "disconnected_edge_mask", size)
    mask_one = _mask(maximum_connected_edge_mask, "maximum_connected_edge_mask", size)
    if any(mask_zero[i][j] > mask_one[i][j] for i in range(size) for j in range(size)):
        raise ValueError("disconnected edge mask may remove but not add directed edges")
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            added = bool(mask_one[i][j] and not mask_zero[i][j])
            if bool(coupling[i][j]) != added:
                raise ValueError("directed coupling support must equal added edge-mask support")
    maximum = _q(maximum_coupling_parameter, "maximum_coupling_parameter")
    if maximum <= 0:
        raise ValueError("maximum_coupling_parameter must be strictly positive")
    center = _q(fixed_circle_center, "fixed_circle_center")
    radius = _q(fixed_circle_radius, "fixed_circle_radius")
    if radius <= 0:
        raise ValueError("fixed_circle_radius must be strictly positive")
    distances = tuple(abs(abs(value - center) - radius) for value in eigenvalues)
    if any(distance == 0 for distance in distances):
        raise ValueError("fixed circle may not contain a disconnected eigenvalue")
    gap = min(distances)
    condition_squared = _frobenius_squared(basis) * _frobenius_squared(inverse_basis)
    condition_upper = _q(eigenvector_condition_frobenius_upper, "eigenvector_condition_frobenius_upper")
    if condition_upper <= 0 or condition_upper * condition_upper < condition_squared:
        raise ValueError("eigenvector condition upper does not enclose exact Frobenius product")
    coupling_squared = maximum * maximum * _frobenius_squared(coupling)
    coupling_upper = _q(maximum_coupling_norm_upper, "maximum_coupling_norm_upper")
    if coupling_upper < 0 or coupling_upper * coupling_upper < coupling_squared:
        raise ValueError("maximum coupling norm upper does not enclose Frobenius coupling")
    resolvent_upper = condition_upper / gap
    product = resolvent_upper * coupling_upper
    margin = 1 - product
    preserved = margin > 0
    projector_upper = (
        radius * resolvent_upper * resolvent_upper * coupling_upper / margin
        if preserved else None
    )
    rank = sum(abs(value - center) < radius for value in eigenvalues)
    if (
        not isinstance(candidate_band, (tuple, list)) or len(candidate_band) != 2
        or any(type(value) is not int for value in candidate_band)
        or candidate_band[0] < 0 or candidate_band[0] > candidate_band[1]
    ):
        raise ValueError("candidate_band must be two ordered nonnegative built-in integers")
    band = tuple(candidate_band)
    normal = _mul(operator_zero, _transpose(operator_zero)) == _mul(_transpose(operator_zero), operator_zero)
    status = (
        "VERIFIED_DIAGONALIZABLE_NONNORMAL_DISCONNECTION_RANK_PRESERVATION"
        if preserved else "VERIFIED_NONNORMAL_DISCONNECTION_PSEUDOSPECTRAL_GAP_UNRESOLVED"
    )
    return VerifiedNonnormalEdgeDisconnectionSpectralRank(
        status=status,
        validation_level=status,
        disconnected_operator=operator_zero,
        maximum_connected_operator=_add(operator_zero, _scale(maximum, coupling)),
        eigenvector_matrix=basis,
        inverse_eigenvector_matrix=inverse_basis,
        disconnected_eigenvalues=eigenvalues,
        disconnected_operator_is_normal=normal,
        fixed_circle_center=center,
        fixed_circle_radius=radius,
        fixed_circle_spectral_gap=gap,
        eigenvector_condition_frobenius_squared=condition_squared,
        eigenvector_condition_frobenius_upper=condition_upper,
        uniform_disconnected_resolvent_norm_upper=resolvent_upper,
        maximum_coupling_frobenius_squared=coupling_squared,
        maximum_coupling_norm_upper=coupling_upper,
        neumann_product_upper=product,
        neumann_margin=margin,
        nominal_fixed_circle_rank=rank,
        fixed_circle_rank_preservation_admitted=preserved,
        projector_perturbation_norm_upper=projector_upper,
        robust_candidate_band_verified=(preserved and band[0] <= rank <= band[1]),
        candidate_band=band,
        eigenvalue_distance_only_certificate_admitted=False,
        defective_operator_branch_verified=False,
        consciousness_dimension_claim_admitted=False,
    )


__all__ = [
    "VerifiedNonnormalEdgeDisconnectionSpectralRank",
    "verified_nonnormal_edge_disconnection_spectral_rank",
]
