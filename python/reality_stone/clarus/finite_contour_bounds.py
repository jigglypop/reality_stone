"""Float64-only circle and annulus estimates for finite contour quadrature.

All outputs here are numerical estimates. They are deliberately not interval
enclosures, certificates, or verified full-circle bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


def _finite_real(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be a {'positive finite' if positive else 'finite'} real number")
    return result


def _finite_complex(value: object, name: str) -> complex:
    try:
        result = complex(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite complex number") from error
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} must be a finite complex number")
    return result


def _finite_complex_matrix(value: object, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite complex square matrix") from error
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a finite complex square matrix")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(matrix.real)) or not np.all(np.isfinite(matrix.imag)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix.copy()


def _positive_nodes(value: object) -> int:
    if type(value) is not int or value < 4:
        raise ValueError("nodes must be a built-in integer at least 4")
    return value


@dataclass(frozen=True)
class Float64CircleEstimate:
    projection_estimate: np.ndarray
    sampled_min_sigma_estimate: float
    chord_upper_estimate: float
    delta_hat_estimate: float
    delta_hat_positive: bool
    resolvent_upper_estimate: float | None
    normalized_sampled_min_sigma_estimate: float
    normalized_chord_upper_estimate: float
    normalized_delta_hat_estimate: float
    normalized_resolvent_upper_estimate: float | None
    center: complex
    radius: float
    nodes: int
    spectral_reference_scale: float
    validation_level: str = "FLOAT64_UNVERIFIED_FULL_CIRCLE_ESTIMATE"


@dataclass(frozen=True)
class Float64AnalyticStripEstimate:
    central_circle_estimate: Float64CircleEstimate
    inner_circle_estimate: Float64CircleEstimate | None
    outer_circle_estimate: Float64CircleEstimate | None
    strip_half_width: float
    numerical_annulus_eigenvalue_count: int
    numerical_annulus_margin_estimate: float | None
    normalized_numerical_annulus_margin_estimate: float | None
    annulus_status: str
    integrand_M_estimate: float | None
    normalized_integrand_M_estimate: float | None
    quadrature_error_estimate: float | None
    normalized_quadrature_error_estimate: float | None
    validation_level: str = "FLOAT64_UNVERIFIED_ANALYTIC_STRIP_ESTIMATE"


def circle_float64_estimate(
    transition: object,
    *,
    center: object,
    radius: object,
    nodes: int,
    spectral_reference_scale: object,
) -> Float64CircleEstimate:
    """Return float64 sampled-circle estimates, never verified bounds."""

    matrix = _finite_complex_matrix(transition, "transition")
    c = _finite_complex(center, "center")
    r = _finite_real(radius, "radius", positive=True)
    n = _positive_nodes(nodes)
    scale = _finite_real(spectral_reference_scale, "spectral_reference_scale", positive=True)
    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    projection = np.zeros_like(matrix)
    min_sigma = math.inf
    for index in range(n):
        theta = 2.0 * math.pi * index / n
        direction = complex(math.cos(theta), math.sin(theta))
        z = c + r * direction
        resolvent_matrix = z * identity - matrix
        singular_values = np.linalg.svd(resolvent_matrix, compute_uv=False)
        sigma = float(singular_values[-1])
        if sigma <= 0.0 or not math.isfinite(sigma):
            raise ValueError("sampled resolvent is singular in float64")
        projection += r * direction * np.linalg.solve(resolvent_matrix, identity)
        min_sigma = min(min_sigma, sigma)
    projection /= n
    chord_upper = 2.0 * r * math.sin(math.pi / (2.0 * n))
    delta_hat = min_sigma - chord_upper
    positive = delta_hat > 0.0
    resolvent_upper = 1.0 / delta_hat if positive else None
    return Float64CircleEstimate(
        projection,
        min_sigma,
        chord_upper,
        delta_hat,
        positive,
        resolvent_upper,
        min_sigma / scale,
        chord_upper / scale,
        delta_hat / scale,
        resolvent_upper * scale if resolvent_upper is not None else None,
        c,
        r,
        n,
        scale,
    )


def analytic_strip_float64_estimate(
    transition: object,
    *,
    center: object,
    radius: object,
    strip_half_width: object,
    nodes: int,
    spectral_reference_scale: object,
) -> Float64AnalyticStripEstimate:
    """Return annular float64 estimates; no strip theorem is certified."""

    matrix = _finite_complex_matrix(transition, "transition")
    c = _finite_complex(center, "center")
    r = _finite_real(radius, "radius", positive=True)
    a = _finite_real(strip_half_width, "strip_half_width", positive=True)
    n = _positive_nodes(nodes)
    scale = _finite_real(spectral_reference_scale, "spectral_reference_scale", positive=True)
    try:
        inner_factor = math.exp(-a)
        outer_factor = math.exp(a)
    except OverflowError as error:
        raise ValueError("strip radii are not representable in float64") from error
    inner_radius = r * inner_factor
    outer_radius = r * outer_factor
    if not (math.isfinite(inner_radius) and math.isfinite(outer_radius)) or inner_radius <= 0.0:
        raise ValueError("strip radii are not representable in float64")
    product = a * n
    if not math.isfinite(product):
        raise ValueError("strip exponent a*N is not representable in float64")
    try:
        denominator = math.expm1(product)
    except OverflowError as error:
        raise ValueError("strip exponent a*N is not representable in float64") from error
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("strip exponent a*N is not representable in float64")
    central = circle_float64_estimate(
        matrix, center=c, radius=r, nodes=n, spectral_reference_scale=scale
    )
    eigenvalues = np.linalg.eigvals(matrix)
    distances = np.abs(eigenvalues - c)
    spectral_scale = max(scale, float(np.linalg.norm(matrix, ord=2)), abs(c), r, float(np.max(distances)))
    ambiguity_tolerance = 32.0 * np.finfo(np.float64).eps * spectral_scale
    annulus = (distances >= inner_radius - ambiguity_tolerance) & (
        distances <= outer_radius + ambiguity_tolerance
    )
    annulus_count = int(np.count_nonzero(annulus))
    margin = float(np.min(np.minimum(np.abs(distances - inner_radius), np.abs(distances - outer_radius))))
    if annulus_count:
        inner = None
        outer = None
        annulus_status = "NUMERICAL_ANNULUS_OR_BOUNDARY_EIGENVALUE"
        m_estimate = None
        error_estimate = None
    else:
        inner = circle_float64_estimate(
            matrix, center=c, radius=inner_radius, nodes=n, spectral_reference_scale=scale
        )
        outer = circle_float64_estimate(
            matrix, center=c, radius=outer_radius, nodes=n, spectral_reference_scale=scale
        )
        annulus_status = "NUMERICAL_ANNULUS_CLEAR_FLOAT64"
        if inner.delta_hat_positive and outer.delta_hat_positive:
            m_estimate = max(
                inner_radius * inner.resolvent_upper_estimate,
                outer_radius * outer.resolvent_upper_estimate,
            )
            error_estimate = 2.0 * m_estimate / denominator
        else:
            m_estimate = None
            error_estimate = None
    return Float64AnalyticStripEstimate(
        central,
        inner,
        outer,
        a,
        annulus_count,
        margin,
        margin / scale,
        annulus_status,
        m_estimate,
        m_estimate,
        error_estimate,
        error_estimate,
    )


__all__ = [
    "Float64AnalyticStripEstimate",
    "Float64CircleEstimate",
    "analytic_strip_float64_estimate",
    "circle_float64_estimate",
]
