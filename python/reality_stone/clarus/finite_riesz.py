"""Finite, a-posteriori contour approximations for nonnormal spectral blocks.

This is a matrix-only trapezoid implementation of a circular Riesz integral.
It reports sampled numerical evidence; it is not an analytic-strip error bound
or a general infinite-dimensional Riesz theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


_DEFAULT_TOLERANCE = 1e-8


def _finite_real(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be a {qualifier} real number")
    return result


def _finite_real_matrix(value: object, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real square matrix") from error
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a finite real square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix.copy()


@dataclass(frozen=True)
class FiniteRieszCertificate:
    """Finite numerical certificate, not an all-contour or analytic proof."""

    approximation: np.ndarray
    refined_approximation: np.ndarray
    orthogonal_range_projector: np.ndarray
    center: float
    radius: float
    spectral_reference_scale: float
    quadrature_nodes: int
    raw_sampled_min_singular_separation: float
    raw_sampled_max_resolvent_norm: float
    normalized_sampled_min_singular_separation: float
    normalized_sampled_max_resolvent_norm: float
    refinement_difference_norm: float
    normalized_refinement_residual: float
    idempotence_residual: float
    normalized_idempotence_residual: float
    commutator_residual: float
    normalized_commutator_residual: float
    imaginary_residual: float
    normalized_imaginary_residual: float
    projection_scale: float
    refinement_scale: float
    idempotence_scale: float
    commutator_scale: float
    realified_range_singular_values: np.ndarray
    realified_numerical_rank: int
    rank_tolerance: float
    effective_rank_threshold: float
    selected_eigenvalue_count: int
    finite_numerical_eigenvalue_margin: float | None
    normalized_eigenvalue_margin: float | None
    eigenvalue_margin_label: str | None
    quadrature_error_bound: None
    validation_level: str


def _left_singular_system(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Small-matrix singular data via a Hermitian Gram eigensystem.

    The finite certificate needs only singular values and left range vectors;
    this avoids invoking an extra full-SVD workspace for each contour sample.
    """

    eigenvalues, vectors = np.linalg.eigh(matrix @ matrix.conj().T)
    order = np.argsort(eigenvalues)[::-1]
    singular_values = np.sqrt(np.maximum(eigenvalues[order], 0.0))
    return singular_values, vectors[:, order]


def _trapezoid_projection(
    matrix: np.ndarray, center: float, radius: float, nodes: int
) -> tuple[np.ndarray, float, float]:
    identity = np.eye(matrix.shape[0])
    total = np.zeros(matrix.shape, dtype=np.complex128)
    min_separation = math.inf
    max_resolvent = 0.0
    for index in range(nodes):
        theta = 2.0 * math.pi * index / nodes
        direction = complex(math.cos(theta), math.sin(theta))
        z = center + radius * direction
        resolvent_matrix = z * identity - matrix
        singular_minimum = float(_left_singular_system(resolvent_matrix)[0][-1])
        if singular_minimum <= 0.0 or not math.isfinite(singular_minimum):
            raise ValueError("sampled contour resolvent is singular")
        try:
            resolvent = np.linalg.solve(resolvent_matrix, identity)
        except np.linalg.LinAlgError as error:
            raise ValueError("sampled contour resolvent is singular") from error
        min_separation = min(min_separation, singular_minimum)
        max_resolvent = max(max_resolvent, float(np.linalg.norm(resolvent, ord=2)))
        total += radius * direction * resolvent
    return total / nodes, min_separation, max_resolvent


def _classify_circle_spectrum(
    eigenvalues: np.ndarray, center: float, radius: float, tolerance: float
) -> tuple[tuple[int, ...], float]:
    radial_distances = np.abs(eigenvalues - center)
    margins = np.abs(radial_distances - radius)
    margin = float(np.min(margins))
    if margin <= tolerance:
        raise ValueError("computed eigenvalue is ambiguous or crosses the contour")
    selected = tuple(index for index, distance in enumerate(radial_distances) if distance < radius)
    if not selected or len(selected) == len(eigenvalues):
        raise ValueError("computed contour cluster must be nonempty and have a complement")

    unmatched = set(selected)
    conjugate_tolerance = tolerance
    while unmatched:
        index = unmatched.pop()
        value = eigenvalues[index]
        if value.imag == 0.0:
            continue
        if abs(value.imag) <= conjugate_tolerance:
            raise ValueError("computed eigenvalue is ambiguously near-real")
        candidates = [
            candidate
            for candidate in unmatched
            if abs(eigenvalues[candidate] - value.conjugate()) <= conjugate_tolerance
        ]
        if not candidates:
            raise ValueError("selected spectrum is not closed under conjugation")
        partner = min(candidates, key=lambda candidate: abs(eigenvalues[candidate] - value.conjugate()))
        unmatched.remove(partner)
    return selected, margin


def finite_riesz_projection(
    transition: object,
    *,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    nodes: int = 32,
    rank_tolerance: object = _DEFAULT_TOLERANCE,
    strict: bool = False,
    certificate_tolerance: object = 1e-6,
) -> FiniteRieszCertificate:
    """Approximate a circular finite Riesz projection with N and 2N nodes.

    ``spectral_reference_scale`` is a mandatory positive scale with the same
    units as ``transition``, ``center``, and ``radius``. ``strict`` compares
    only normalized finite residuals to ``certificate_tolerance``; it never
    manufactures an unverified quadrature-error bound.
    """

    matrix = _finite_real_matrix(transition, "transition")
    c = _finite_real(center, "center")
    r = _finite_real(radius, "radius", positive=True)
    reference_scale = _finite_real(
        spectral_reference_scale, "spectral_reference_scale", positive=True
    )
    if type(nodes) is not int or nodes < 4:
        raise ValueError("nodes must be a built-in integer at least 4")
    rank_floor = _finite_real(rank_tolerance, "rank_tolerance", positive=True)
    certificate_floor = _finite_real(
        certificate_tolerance, "certificate_tolerance", positive=True
    )

    eigenvalues = np.linalg.eigvals(matrix)
    spectral_scale = max(
        reference_scale,
        float(np.linalg.norm(matrix, ord=2)),
        abs(c),
        r,
        float(np.max(np.abs(eigenvalues))),
    )
    selected, eigenvalue_margin = _classify_circle_spectrum(
        eigenvalues, c, r, certificate_floor * spectral_scale
    )
    approximation, sampled_minimum_n, sampled_maximum_n = _trapezoid_projection(
        matrix, c, r, nodes
    )
    refined, sampled_minimum_2n, sampled_maximum_2n = _trapezoid_projection(
        matrix, c, r, 2 * nodes
    )
    sampled_minimum = min(sampled_minimum_n, sampled_minimum_2n)
    sampled_maximum = max(sampled_maximum_n, sampled_maximum_2n)
    refinement = float(np.linalg.norm(refined - approximation, ord=2))
    idempotence = float(np.linalg.norm(refined @ refined - refined, ord=2))
    commutator = float(np.linalg.norm(matrix @ refined - refined @ matrix, ord=2))
    imaginary = float(np.linalg.norm(refined.imag, ord=2))
    projection_scale = max(1.0, float(np.linalg.norm(refined, ord=2)))
    refinement_scale = max(
        1.0,
        float(np.linalg.norm(approximation, ord=2)),
        float(np.linalg.norm(refined, ord=2)),
    )
    idempotence_scale = max(1.0, projection_scale, projection_scale**2)
    commutator_scale = spectral_scale * projection_scale
    normalized_refinement = refinement / refinement_scale
    normalized_idempotence = idempotence / idempotence_scale
    normalized_commutator = commutator / commutator_scale
    normalized_imaginary = imaginary / projection_scale
    normalized_sampled_minimum = sampled_minimum / spectral_scale
    normalized_sampled_maximum = sampled_maximum * spectral_scale
    normalized_eigenvalue_margin = eigenvalue_margin / spectral_scale

    realified = np.concatenate((refined.real, refined.imag), axis=1)
    singular_values, left_vectors = _left_singular_system(realified)
    effective_rank_threshold = rank_floor * max(1.0, float(singular_values[0]))
    numerical_rank = int(np.count_nonzero(singular_values > effective_rank_threshold))
    ambiguity_band = 0.1 * effective_rank_threshold
    if np.any(np.abs(singular_values - effective_rank_threshold) <= ambiguity_band):
        raise ValueError("realified range rank is ambiguous at the declared tolerance")
    if numerical_rank != len(selected):
        raise ValueError("realified range numerical rank disagrees with selected spectrum")
    if numerical_rank == 0 or numerical_rank == matrix.shape[0]:
        raise ValueError("realified range rank must be nontrivial")
    basis = left_vectors[:, :numerical_rank]
    orthogonal_projector = basis @ basis.T

    if strict and max(
        normalized_refinement,
        normalized_idempotence,
        normalized_commutator,
        normalized_imaginary,
    ) > certificate_floor:
        raise ValueError("finite a-posteriori certificate residual exceeds strict tolerance")
    return FiniteRieszCertificate(
        approximation,
        refined,
        orthogonal_projector,
        c,
        r,
        reference_scale,
        nodes,
        sampled_minimum,
        sampled_maximum,
        normalized_sampled_minimum,
        normalized_sampled_maximum,
        refinement,
        normalized_refinement,
        idempotence,
        normalized_idempotence,
        commutator,
        normalized_commutator,
        imaginary,
        normalized_imaginary,
        projection_scale,
        refinement_scale,
        idempotence_scale,
        commutator_scale,
        singular_values,
        numerical_rank,
        rank_floor,
        effective_rank_threshold,
        len(selected),
        eigenvalue_margin,
        normalized_eigenvalue_margin,
        "FINITE_NUMERICAL_EIGENVALUE_MARGIN",
        None,
        "FINITE_A_POSTERIORI_CONTOUR_APPROXIMATION",
    )


__all__ = ["FiniteRieszCertificate", "finite_riesz_projection"]
