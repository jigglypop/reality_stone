"""Finite L0 seam for audited history-edge metric and subspace formulas.

This module is deliberately separate from :mod:`unified_metric`: availability
weights deform a quadratic metric, while declared adjacency is immutable input
for a separate topology experiment.  It implements finite symmetric fixtures,
not a general Riesz calculus, biological identification, or a fixed dimension
claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

import numpy as np


_TOLERANCE = 1e-10


def _finite_scalar(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be a {qualifier} real number")
    return result


def _matrix(values: object, name: str, *, square: bool = True) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real matrix") from error
    if result.ndim != 2 or (square and result.shape[0] != result.shape[1]):
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


def _symmetric(values: object, name: str) -> np.ndarray:
    result = _matrix(values, name)
    if not np.allclose(result, result.T, atol=_TOLERANCE, rtol=_TOLERANCE):
        raise ValueError(f"{name} must be symmetric")
    # Tolerance acceptance is a projection rule, not permission to carry an
    # asymmetric representative into eigensolvers or quadratic forms.
    canonical = 0.5 * result + 0.5 * result.T
    if not np.all(np.isfinite(canonical)):
        raise ValueError(f"{name} symmetric canonicalization must be finite")
    return canonical


def _psd(values: object, name: str) -> np.ndarray:
    result = _symmetric(values, name)
    eigenvalues, vectors = np.linalg.eigh(result)
    if float(eigenvalues.min()) < -_TOLERANCE:
        raise ValueError(f"{name} must be positive semidefinite")
    # Values inside the accepted negative roundoff band are explicitly
    # projected to the PSD cone before downstream calculations.
    clipped = np.maximum(eigenvalues, 0.0)
    return (vectors * clipped) @ vectors.T


def _availability(values: Sequence[float], count: int, name: str) -> tuple[float, ...]:
    if len(values) != count:
        raise ValueError(f"{name} must have one value for each edge")
    result = tuple(_finite_scalar(value, f"{name}[{index}]") for index, value in enumerate(values))
    if any(value < 0.0 or value > 1.0 for value in result):
        raise ValueError(f"{name} entries must lie in [0, 1]")
    return result


@dataclass(frozen=True)
class EdgeContribution:
    """One finite edge contrast term ``b D.T @ K @ D``."""

    contrast: np.ndarray
    weight: np.ndarray
    availability: float = 1.0

    def __post_init__(self) -> None:
        contrast = _matrix(self.contrast, "contrast", square=False)
        weight = _psd(self.weight, "weight")
        if contrast.shape[0] != weight.shape[0]:
            raise ValueError("contrast row count must match weight size")
        object.__setattr__(self, "contrast", contrast)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "availability", _availability((self.availability,), 1, "availability")[0])

    def operator(self) -> np.ndarray:
        return self.contrast.T @ self.weight @ self.contrast


@dataclass(frozen=True)
class PerturbationCertificate:
    actual_spectral_norm: float
    theorem_upper_bound: float
    baseline_coercivity: float
    normalized_upper_bound: float
    small_perturbation: bool


@dataclass(frozen=True)
class ConcentrationCertificate:
    """Auditable finite certificate for orthogonal spectral concentration."""

    value: float
    canonical_projector: np.ndarray
    input_symmetry_residual: float
    input_idempotence_residual: float
    certificate_tolerance: float


@dataclass(frozen=True)
class SpectralSubspace:
    projector: np.ndarray
    selected_indices: tuple[int, ...]
    eigenvalues: np.ndarray
    spectral_gap: float
    numerical_rank: int
    real_dimension: int
    rank_tolerance: float
    input_canonicalization_residual: float
    normality_residual: float
    invariance_residual: float
    certificate_tolerance: float


@dataclass(frozen=True)
class EffectiveDimension:
    effective_dimension: float
    positive_spectrum_rank: int
    numerical_hard_rank: int
    eigenvalues: np.ndarray
    regularization: float
    rank_tolerance: float


@dataclass(frozen=True)
class FiniteEdgeMetric:
    """Validated finite coercive metric with separately declared topology."""

    baseline: np.ndarray
    edges: tuple[EdgeContribution, ...]
    declared_adjacency: tuple[bool, ...] | None = None

    def __post_init__(self) -> None:
        baseline = _symmetric(self.baseline, "baseline")
        m0 = float(np.linalg.eigvalsh(baseline).min())
        if m0 <= _TOLERANCE:
            raise ValueError("baseline must be coercive positive definite")
        edges = tuple(self.edges)
        for edge in edges:
            if not isinstance(edge, EdgeContribution):
                raise ValueError("edges must contain EdgeContribution values")
            if edge.contrast.shape[1] != baseline.shape[0]:
                raise ValueError("contrast column count must match baseline size")
        adjacency = self.declared_adjacency
        if adjacency is None:
            adjacency = tuple(True for _ in edges)
        elif len(adjacency) != len(edges) or not all(type(value) is bool for value in adjacency):
            raise ValueError("declared_adjacency must be one bool per edge")
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "declared_adjacency", tuple(adjacency))

    @property
    def baseline_coercivity(self) -> float:
        return float(np.linalg.eigvalsh(self.baseline).min())

    @property
    def availability(self) -> tuple[float, ...]:
        return tuple(edge.availability for edge in self.edges)

    def metric(self, availability: Sequence[float] | None = None) -> np.ndarray:
        values = self.availability if availability is None else _availability(availability, len(self.edges), "availability")
        result = self.baseline.copy()
        for value, edge in zip(values, self.edges):
            result += value * edge.operator()
        return result

    def perturbation_certificate(
        self,
        before: Sequence[float],
        after: Sequence[float],
    ) -> PerturbationCertificate:
        left = _availability(before, len(self.edges), "before")
        right = _availability(after, len(self.edges), "after")
        delta = self.metric(right) - self.metric(left)
        actual = float(np.linalg.norm(delta, ord=2))
        upper = sum(
            abs(new - old)
            * float(np.linalg.norm(edge.contrast, ord=2)) ** 2
            * float(np.linalg.norm(edge.weight, ord=2))
            for old, new, edge in zip(left, right, self.edges)
        )
        m0 = self.baseline_coercivity
        return PerturbationCertificate(actual, upper, m0, upper / m0, upper / m0 < 1.0)


def topology_with_deleted_edges(
    declared_adjacency: Sequence[bool], deleted_edge_indices: Sequence[int],
) -> tuple[bool, ...]:
    """Return a topology fixture; metric availability is intentionally untouched."""

    adjacency = tuple(declared_adjacency)
    if not all(type(value) is bool for value in adjacency):
        raise ValueError("declared_adjacency must contain bool values")
    deleted = tuple(deleted_edge_indices)
    if any(type(index) is not int or not 0 <= index < len(adjacency) for index in deleted):
        raise ValueError("deleted edge index is out of range")
    result = list(adjacency)
    for index in deleted:
        result[index] = False
    return tuple(result)


def spectral_subspace(
    transition: object,
    *,
    indices: Sequence[int] | None = None,
    interval: tuple[float, float] | None = None,
    min_gap: float = _TOLERANCE,
    rank_tolerance: float = _TOLERANCE,
    certificate_tolerance: float = _TOLERANCE,
) -> SpectralSubspace:
    """Select an isolated finite spectral cluster of a real symmetric/normal map.

    General nonnormal matrices are rejected: this finite seam must not be
    presented as a general Riesz-projection implementation.
    """

    input_matrix = _matrix(transition, "transition")
    gap_floor = _finite_scalar(min_gap, "min_gap", positive=True)
    rank_floor = _finite_scalar(rank_tolerance, "rank_tolerance", positive=True)
    certificate_floor = _finite_scalar(
        certificate_tolerance, "certificate_tolerance", positive=True
    )
    symmetric = np.allclose(
        input_matrix, input_matrix.T, atol=_TOLERANCE, rtol=_TOLERANCE
    )
    if symmetric:
        matrix = _symmetric(input_matrix, "transition")
        input_scale = max(1.0, float(np.linalg.norm(input_matrix, ord=2)))
        input_canonicalization_residual = float(
            np.linalg.norm(input_matrix - matrix, ord=2) / input_scale
        )
    else:
        matrix = input_matrix
        input_canonicalization_residual = 0.0
    norm_scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
    normality_residual = float(
        np.linalg.norm(matrix.T @ matrix - matrix @ matrix.T, ord=2) / norm_scale**2
    )
    if normality_residual > certificate_floor:
        raise ValueError("transition must be real symmetric or normal; nonnormal input is unsupported")
    if (indices is None) == (interval is None):
        raise ValueError("supply exactly one of indices or interval")

    if symmetric:
        eigenvalues, vectors = np.linalg.eigh(matrix)
    else:
        if interval is not None:
            raise ValueError("interval selection requires a real symmetric transition")
        eigenvalues, vectors = np.linalg.eig(matrix)

    if indices is not None:
        selected = tuple(indices)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("indices must be a nonempty unique cluster")
        if any(type(index) is not int or not 0 <= index < len(eigenvalues) for index in selected):
            raise ValueError("cluster index is out of range")
    else:
        assert interval is not None
        lower, upper = (_finite_scalar(interval[0], "interval lower"), _finite_scalar(interval[1], "interval upper"))
        if lower > upper:
            raise ValueError("interval lower bound must not exceed upper bound")
        selected = tuple(index for index, value in enumerate(eigenvalues) if lower <= float(value) <= upper)
        if not selected:
            raise ValueError("interval selects no eigenvalues")

    remaining = tuple(index for index in range(len(eigenvalues)) if index not in selected)
    if not remaining:
        raise ValueError("cluster must leave a complement to certify an explicit gap")
    gap = min(abs(eigenvalues[left] - eigenvalues[right]) for left in selected for right in remaining)
    if float(gap) <= gap_floor:
        raise ValueError("selected cluster is not separated by the declared spectral gap")

    if not symmetric:
        unmatched = set(selected)
        conjugate_tolerance = certificate_floor * max(1.0, float(np.max(np.abs(eigenvalues))))
        while unmatched:
            index = unmatched.pop()
            value = eigenvalues[index]
            if value.imag == 0.0:
                continue
            if abs(value.imag) <= conjugate_tolerance:
                raise ValueError("normal real transition has an ambiguous near-real eigenvalue")
            candidates = [
                candidate
                for candidate in unmatched
                if abs(eigenvalues[candidate] - value.conjugate()) <= conjugate_tolerance
            ]
            if not candidates:
                raise ValueError(
                    "normal real transition cluster must include conjugate eigenvalue pairs with matching multiplicity"
                )
            partner = min(candidates, key=lambda candidate: abs(eigenvalues[candidate] - value.conjugate()))
            unmatched.remove(partner)
        candidate = np.concatenate((vectors[:, selected].real, vectors[:, selected].imag), axis=1)
        q_basis, singular_values, _ = np.linalg.svd(candidate, full_matrices=False)
        rank = int(np.count_nonzero(singular_values > rank_floor))
        basis = q_basis[:, :rank]
    else:
        basis = vectors[:, selected]
        rank = len(selected)
    projector = basis @ basis.T
    projector = _symmetric(projector, "projector")
    invariance_residual = float(
        np.linalg.norm(matrix @ projector - projector @ matrix, ord=2) / norm_scale
    )
    if invariance_residual > certificate_floor:
        raise ArithmeticError("invariant-subspace residual exceeds certificate tolerance")
    return SpectralSubspace(
        projector,
        selected,
        eigenvalues,
        float(gap),
        rank,
        rank,
        rank_floor,
        input_canonicalization_residual,
        normality_residual,
        invariance_residual,
        certificate_floor,
    )


def orthogonal_concentration(
    projector: object,
    covariance: object,
    *,
    certificate_tolerance: float = _TOLERANCE,
) -> ConcentrationCertificate:
    """Certify ``tr(Q C Q) / tr(C)`` under numerical projector hypotheses.

    Small symmetry/idempotence errors are disclosed and projected to the nearest
    eigenvalue-thresholded orthogonal projector; material errors are rejected.
    """

    raw_projector = _matrix(projector, "projector")
    tolerance = _finite_scalar(
        certificate_tolerance, "certificate_tolerance", positive=True
    )
    input_scale = max(1.0, float(np.linalg.norm(raw_projector, ord=2)))
    symmetry_residual = float(
        np.linalg.norm(raw_projector - raw_projector.T, ord=2) / input_scale
    )
    if symmetry_residual > tolerance:
        raise ValueError("projector must be symmetric within certificate tolerance")
    q_symmetric = 0.5 * raw_projector + 0.5 * raw_projector.T
    idempotence_scale = max(1.0, float(np.linalg.norm(q_symmetric, ord=2)))
    idempotence_residual = float(
        np.linalg.norm(q_symmetric @ q_symmetric - q_symmetric, ord=2) / idempotence_scale
    )
    if idempotence_residual > tolerance:
        raise ValueError("projector must be idempotent within certificate tolerance")
    eigenvalues, vectors = np.linalg.eigh(q_symmetric)
    distances = np.minimum(np.abs(eigenvalues), np.abs(eigenvalues - 1.0))
    if float(distances.max()) > tolerance:
        raise ValueError("projector eigenvalues must lie near 0 or 1")
    q = (vectors * (eigenvalues >= 0.5)) @ vectors.T
    c = _psd(covariance, "covariance")
    if q.shape != c.shape:
        raise ValueError("projector and covariance must have the same shape")
    denominator = float(np.trace(c))
    if denominator <= _TOLERANCE:
        raise ValueError("covariance must have positive trace")
    value = float(np.trace(q @ c @ q) / denominator)
    if value < -_TOLERANCE or value > 1.0 + _TOLERANCE:
        raise ArithmeticError("orthogonal concentration escaped [0, 1]")
    return ConcentrationCertificate(
        min(1.0, max(0.0, value)),
        q,
        symmetry_residual,
        idempotence_residual,
        tolerance,
    )


def effective_dimension(gram: object, regularization: object) -> EffectiveDimension:
    """Compute observed effective dimension; it is distinct from numerical rank."""

    g = _psd(gram, "gram")
    lam = _finite_scalar(regularization, "regularization", positive=True)
    eigenvalues = np.maximum(np.linalg.eigvalsh(g), 0.0)
    positive_spectrum_rank = int(np.count_nonzero(eigenvalues > 0.0))
    numerical_hard_rank = int(np.count_nonzero(eigenvalues > _TOLERANCE))
    value = float(np.sum(eigenvalues / (eigenvalues + lam)))
    return EffectiveDimension(
        value,
        positive_spectrum_rank,
        numerical_hard_rank,
        eigenvalues,
        lam,
        _TOLERANCE,
    )


def mobility_scale(X0: object, V0: object, t0: object) -> float:
    """Return the dimensional coefficient ``X0**2 / (V0*t0)`` before scaling."""

    x0 = _finite_scalar(X0, "X0", positive=True)
    v0 = _finite_scalar(V0, "V0", positive=True)
    t0 = _finite_scalar(t0, "t0", positive=True)
    result = x0 * x0 / (v0 * t0)
    if not math.isfinite(result):
        raise ValueError("mobility scale must be finite")
    return result


__all__ = [
    "EdgeContribution",
    "ConcentrationCertificate",
    "EffectiveDimension",
    "FiniteEdgeMetric",
    "PerturbationCertificate",
    "SpectralSubspace",
    "effective_dimension",
    "mobility_scale",
    "orthogonal_concentration",
    "spectral_subspace",
    "topology_with_deleted_edges",
]
