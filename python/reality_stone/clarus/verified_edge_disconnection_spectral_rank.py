"""Exact finite Hermitian edge-disconnection and fixed-contour rank boundary."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_edge_metric_effective_dimension import (
        Matrix, _add, _identity, _matrix, _mul, _q, _scale, _transpose,
    )
else:
    from verified_edge_metric_effective_dimension import (  # type: ignore[no-redef]
        Matrix, _add, _identity, _matrix, _mul, _q, _scale, _transpose,
    )


def _diagonal(values: tuple[Fraction, ...]) -> Matrix:
    return tuple(tuple(values[i] if i == j else Fraction(0) for j in range(len(values))) for i in range(len(values)))


def _adjacency(value: object, name: str, size: int) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise ValueError(f"{name} must be a square adjacency matrix matching operator size")
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
    result = tuple(rows)
    if result != tuple(tuple(result[j][i] for j in range(size)) for i in range(size)):
        raise ValueError(f"{name} must be symmetric")
    return result


def _components(adjacency: tuple[tuple[int, ...], ...]) -> int:
    unseen = set(range(len(adjacency)))
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            neighbors = [j for j, edge in enumerate(adjacency[node]) if edge and j in unseen]
            for neighbor in neighbors:
                unseen.remove(neighbor)
                stack.append(neighbor)
    return count


def _columns(matrix: Matrix) -> tuple[tuple[Fraction, ...], ...]:
    return _transpose(matrix)


@dataclass(frozen=True)
class VerifiedEdgeDisconnectionSpectralRank:
    status: str
    validation_level: str
    disconnected_adjacency_components: int
    connected_adjacency_components: int
    topological_disconnection_verified: bool
    removed_edge_pairs: tuple[tuple[int, int], ...]
    disconnected_operator: Matrix
    connected_operator: Matrix
    disconnected_eigenvalues: tuple[Fraction, ...]
    connected_eigenvalues: tuple[Fraction, ...]
    disconnected_fixed_contour_rank: int
    connected_fixed_contour_rank: int
    fixed_contour_radius: Fraction
    disconnected_spectral_gap: Fraction
    coupling_frobenius_norm_squared: Fraction
    robust_gap_squared: Fraction
    strict_uniform_no_crossing_bound_verified: bool
    fixed_contour_rank_preservation_admitted: bool
    endpoint_rank_change_verified: bool
    intermediate_contour_crossing_mathematically_required: bool
    crossing_location_computed: bool
    topological_disconnection_causes_rank_change_claim_admitted: bool
    robust_candidate_band_verified: bool
    candidate_band: tuple[int, int]
    consciousness_dimension_claim_admitted: bool


def verified_edge_disconnection_spectral_rank(
    *,
    disconnected_adjacency: object,
    connected_adjacency: object,
    disconnected_eigenvalues: object,
    edge_coupling_in_disconnected_eigenbasis: object,
    maximum_coupling_parameter: object,
    connected_eigenvalues: object,
    connected_eigenbasis_columns: object,
    fixed_contour_radius: object,
    candidate_band: object,
) -> VerifiedEdgeDisconnectionSpectralRank:
    if not isinstance(disconnected_eigenvalues, (tuple, list)) or not disconnected_eigenvalues:
        raise ValueError("disconnected_eigenvalues must be a nonempty sequence")
    eigenvalues_zero = tuple(_q(value, "disconnected_eigenvalue") for value in disconnected_eigenvalues)
    size = len(eigenvalues_zero)
    adjacency_zero = _adjacency(disconnected_adjacency, "disconnected_adjacency", size)
    adjacency_one = _adjacency(connected_adjacency, "connected_adjacency", size)
    if any(adjacency_zero[i][j] > adjacency_one[i][j] for i in range(size) for j in range(size)):
        raise ValueError("disconnected adjacency may remove but not add connected edges")
    removed = tuple((i, j) for i in range(size) for j in range(i + 1, size) if adjacency_one[i][j] and not adjacency_zero[i][j])
    coupling = _matrix(edge_coupling_in_disconnected_eigenbasis, "edge_coupling", columns=size)
    if len(coupling) != size or coupling != _transpose(coupling):
        raise ValueError("edge_coupling must be exact symmetric and match operator size")
    if any(coupling[i][i] for i in range(size)):
        raise ValueError("edge_coupling diagonal must vanish for a pure removed-edge model")
    removed_set = set(removed)
    for i in range(size):
        for j in range(i + 1, size):
            if coupling[i][j] and (i, j) not in removed_set:
                raise ValueError("nonzero coupling must correspond to a removed adjacency edge")
            if (i, j) in removed_set and not coupling[i][j]:
                raise ValueError("every removed adjacency edge needs a nonzero operator coupling")
    maximum = _q(maximum_coupling_parameter, "maximum_coupling_parameter")
    if maximum <= 0:
        raise ValueError("maximum_coupling_parameter must be strictly positive")
    if not isinstance(connected_eigenvalues, (tuple, list)) or len(connected_eigenvalues) != size:
        raise ValueError("connected_eigenvalues must match operator size")
    eigenvalues_one = tuple(_q(value, "connected_eigenvalue") for value in connected_eigenvalues)
    basis = _matrix(connected_eigenbasis_columns, "connected_eigenbasis_columns", columns=size)
    if len(basis) != size or _mul(_transpose(basis), basis) != _identity(size):
        raise ValueError("connected eigenbasis columns must be exact orthonormal")
    operator_zero = _diagonal(eigenvalues_zero)
    operator_one = _add(operator_zero, _scale(maximum, coupling))
    if _mul(operator_one, basis) != _mul(basis, _diagonal(eigenvalues_one)):
        raise ValueError("connected eigenbasis/eigenvalue witness does not diagonalize connected operator")
    radius = _q(fixed_contour_radius, "fixed_contour_radius")
    if radius <= 0:
        raise ValueError("fixed_contour_radius must be strictly positive")
    if any(abs(value) == radius for value in eigenvalues_zero + eigenvalues_one):
        raise ValueError("fixed contour may not contain an endpoint eigenvalue")
    if (
        not isinstance(candidate_band, (tuple, list)) or len(candidate_band) != 2
        or any(type(value) is not int for value in candidate_band)
        or candidate_band[0] < 0 or candidate_band[0] > candidate_band[1]
    ):
        raise ValueError("candidate_band must be two ordered nonnegative built-in integers")
    band = tuple(candidate_band)
    rank_zero = sum(abs(value) < radius for value in eigenvalues_zero)
    rank_one = sum(abs(value) < radius for value in eigenvalues_one)
    gap = min(abs(abs(value) - radius) for value in eigenvalues_zero)
    coupling_squared = maximum * maximum * sum((item * item for row in coupling for item in row), Fraction(0))
    robust_gap_squared = gap * gap - coupling_squared
    robust = robust_gap_squared > 0
    if robust and rank_zero != rank_one:
        raise ValueError("connected eigenwitness contradicts the strict no-crossing perturbation bound")
    rank_change = rank_zero != rank_one
    components_zero = _components(adjacency_zero)
    components_one = _components(adjacency_one)
    topological = components_zero > components_one
    status = (
        "VERIFIED_TOPOLOGICAL_DISCONNECTION_WITH_FIXED_CONTOUR_RANK_PRESERVATION"
        if robust
        else (
            "VERIFIED_ENDPOINT_RANK_CHANGE_REQUIRES_INTERMEDIATE_CONTOUR_CROSSING"
            if rank_change
            else "VERIFIED_DISCONNECTION_SPECTRAL_GAP_UNRESOLVED"
        )
    )
    return VerifiedEdgeDisconnectionSpectralRank(
        status=status,
        validation_level=status,
        disconnected_adjacency_components=components_zero,
        connected_adjacency_components=components_one,
        topological_disconnection_verified=topological,
        removed_edge_pairs=removed,
        disconnected_operator=operator_zero,
        connected_operator=operator_one,
        disconnected_eigenvalues=eigenvalues_zero,
        connected_eigenvalues=eigenvalues_one,
        disconnected_fixed_contour_rank=rank_zero,
        connected_fixed_contour_rank=rank_one,
        fixed_contour_radius=radius,
        disconnected_spectral_gap=gap,
        coupling_frobenius_norm_squared=coupling_squared,
        robust_gap_squared=robust_gap_squared,
        strict_uniform_no_crossing_bound_verified=robust,
        fixed_contour_rank_preservation_admitted=robust,
        endpoint_rank_change_verified=rank_change,
        intermediate_contour_crossing_mathematically_required=rank_change,
        crossing_location_computed=False,
        topological_disconnection_causes_rank_change_claim_admitted=False,
        robust_candidate_band_verified=(robust and band[0] <= rank_zero <= band[1]),
        candidate_band=band,
        consciousness_dimension_claim_admitted=False,
    )


__all__ = ["VerifiedEdgeDisconnectionSpectralRank", "verified_edge_disconnection_spectral_rank"]
