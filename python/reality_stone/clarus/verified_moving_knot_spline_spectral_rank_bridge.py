"""Uniform spectral/rank bridge for a continuously moving-knot radial spline family."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_continuous_axis_aligned_ellipse_spectral_margin_optimization import (
        QMatrix,
        _diagonal,
        _projector,
    )
    from .verified_continuous_periodic_spline_knot_optimization import (
        VerifiedContinuousPeriodicSplineKnotOptimization,
        verified_continuous_periodic_spline_knot_optimization,
    )
    from .verified_rational_contour import (
        QComplex,
        _dyadic_sqrt,
        _fraction,
        _inverse,
        _matmul,
        _matrix,
        parse_qcomplex,
    )
else:
    from verified_continuous_axis_aligned_ellipse_spectral_margin_optimization import (  # type: ignore[no-redef]
        QMatrix,
        _diagonal,
        _projector,
    )
    from verified_continuous_periodic_spline_knot_optimization import (  # type: ignore[no-redef]
        VerifiedContinuousPeriodicSplineKnotOptimization,
        verified_continuous_periodic_spline_knot_optimization,
    )
    from verified_rational_contour import (  # type: ignore[no-redef]
        QComplex,
        _dyadic_sqrt,
        _fraction,
        _inverse,
        _matmul,
        _matrix,
        parse_qcomplex,
    )


@dataclass(frozen=True)
class VerifiedMovingKnotSplineSpectralRankBridge:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    knot_optimization: VerifiedContinuousPeriodicSplineKnotOptimization
    normalized_transition: QMatrix
    normalized_eigenvalues: tuple[QComplex, ...]
    target_inside_labels: tuple[bool, ...]
    exact_diagonalization_witness_verified: bool
    normalized_center: QComplex
    normalized_axis_u: QComplex
    normalized_axis_v: QComplex
    affine_orientation_determinant: Fraction
    affine_inverse_coordinate_squared: tuple[Fraction, ...]
    uniform_radial_minimum_lower: Fraction | None
    uniform_radial_maximum_upper: Fraction | None
    per_eigenvalue_uniform_signed_squared_margins: tuple[Fraction, ...] | None
    uniform_family_margin: Fraction | None
    positive_uniform_family_margin_verified: bool
    affine_frobenius_norm_squared: Fraction
    affine_minimum_singular_value_squared_lower: Fraction
    eigenvector_frobenius_condition_squared_upper: Fraction | None
    per_eigenvalue_normalized_contour_distance_squared_lower: tuple[Fraction, ...] | None
    uniform_normalized_contour_distance_squared_lower: Fraction | None
    uniform_affine_contour_distance_squared_lower: Fraction | None
    quantitative_resolvent_norm_squared_upper: Fraction | None
    selected_projector: QMatrix | None
    selected_projector_rank: int | None
    projector_identity_verified: bool
    simple_periodic_cq_jordan_family_verified: bool
    entire_moving_knot_box_resolvent_verified: bool
    moving_knot_family_spectral_split_verified: bool
    moving_knot_family_rank_preserved: bool
    quantitative_resolvent_norm_bound: Fraction | None
    interval_matrix_family_verified: bool
    empirical_matrix_provenance_verified: bool


def _affine_inverse_coordinate_squared(
    value: QComplex,
    center: QComplex,
    axis_u: QComplex,
    axis_v: QComplex,
    determinant: Fraction,
) -> Fraction:
    delta = value - center
    x = (axis_v.imag * delta.real - axis_v.real * delta.imag) / determinant
    y = (-axis_u.imag * delta.real + axis_u.real * delta.imag) / determinant
    return x * x + y * y


def _frobenius_norm_squared(matrix: QMatrix) -> Fraction:
    return sum(
        (entry.abs_squared() for row in matrix for entry in row),
        Fraction(0),
    )


def verified_moving_knot_spline_spectral_rank_bridge(
    nominal_transition: object,
    *,
    eigenvectors: object,
    eigenvalues: object,
    target_inside_labels: object,
    center: object,
    axis_u: object,
    axis_v: object,
    spectral_reference_scale: object,
    knot_parameter_intervals: object,
    patch_amplitudes: object,
    junction_order: int,
    normalized_knot_optimality_tolerance: object,
    maximum_knot_cells: int = 16384,
    sqrt_precision: int = 48,
) -> VerifiedMovingKnotSplineSpectralRankBridge:
    """Certify one exact spectral split for every contour in a moving-knot box."""
    knot = verified_continuous_periodic_spline_knot_optimization(
        knot_parameter_intervals=knot_parameter_intervals,
        patch_amplitudes=patch_amplitudes,
        junction_order=junction_order,
        normalized_optimality_tolerance=normalized_knot_optimality_tolerance,
        maximum_cells=maximum_knot_cells,
        sqrt_precision=sqrt_precision,
    )
    raw = _matrix(nominal_transition, "nominal_transition")
    n = len(raw)
    vectors = _matrix(eigenvectors, "eigenvectors")
    if len(vectors) != n:
        raise ValueError("eigenvectors must shape-match nominal_transition")
    if not isinstance(eigenvalues, (tuple, list)) or len(eigenvalues) != n:
        raise ValueError("eigenvalues must contain exactly n exact Q(i) values")
    values = tuple(
        parse_qcomplex(value, f"eigenvalues[{index}]")
        for index, value in enumerate(eigenvalues)
    )
    if not isinstance(target_inside_labels, (tuple, list)) or len(target_inside_labels) != n:
        raise ValueError("target_inside_labels must contain exactly n booleans")
    if any(type(label) is not bool for label in target_inside_labels):
        raise ValueError("target_inside_labels entries must be built-in booleans")
    labels = tuple(target_inside_labels)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    normalized = tuple(tuple(entry / scale for entry in row) for row in raw)
    normalized_values = tuple(value / scale for value in values)
    normalized_center = parse_qcomplex(center, "center") / scale
    normalized_axis_u = parse_qcomplex(axis_u, "axis_u") / scale
    normalized_axis_v = parse_qcomplex(axis_v, "axis_v") / scale
    determinant = (
        normalized_axis_u.real * normalized_axis_v.imag
        - normalized_axis_u.imag * normalized_axis_v.real
    )
    if determinant <= 0:
        raise ValueError("affine axes must have strictly positive orientation determinant")
    inverse_vectors = _inverse(vectors)
    witness = (
        inverse_vectors is not None
        and _matmul(normalized, vectors)
        == _matmul(vectors, _diagonal(normalized_values))
    )
    coordinate_squared = tuple(
        _affine_inverse_coordinate_squared(
            value,
            normalized_center,
            normalized_axis_u,
            normalized_axis_v,
            determinant,
        )
        for value in normalized_values
    )
    radial_minimum = knot.radial_minimum_lower
    radial_maximum = knot.radial_maximum_upper
    margins = None
    family_margin = None
    if radial_minimum is not None and radial_maximum is not None:
        margins = tuple(
            (
                radial_minimum * radial_minimum - squared
                if inside
                else squared - radial_maximum * radial_maximum
            )
            for squared, inside in zip(coordinate_squared, labels, strict=True)
        )
        family_margin = min(margins)
    affine_frobenius_squared = (
        normalized_axis_u.abs_squared() + normalized_axis_v.abs_squared()
    )
    affine_minimum_singular_squared_lower = (
        determinant * determinant / affine_frobenius_squared
    )
    eigenvector_condition_squared_upper = None
    normalized_distance_squared_lower = None
    uniform_normalized_distance_squared_lower = None
    uniform_affine_distance_squared_lower = None
    resolvent_squared_upper = None
    resolvent_upper = None
    resolvent_sqrt_checks = (False, False)
    if witness and inverse_vectors is not None and margins is not None:
        eigenvector_condition_squared_upper = (
            _frobenius_norm_squared(vectors)
            * _frobenius_norm_squared(inverse_vectors)
        )
        normalized_distance_squared_lower = tuple(
            margin * margin
            / (
                2
                * (
                    squared
                    + (
                        radial_minimum * radial_minimum
                        if inside
                        else radial_maximum * radial_maximum
                    )
                )
            )
            for squared, inside, margin in zip(
                coordinate_squared, labels, margins, strict=True
            )
        )
        if normalized_distance_squared_lower:
            uniform_normalized_distance_squared_lower = min(
                normalized_distance_squared_lower
            )
            uniform_affine_distance_squared_lower = (
                affine_minimum_singular_squared_lower
                * uniform_normalized_distance_squared_lower
            )
            if uniform_affine_distance_squared_lower > 0:
                resolvent_squared_upper = (
                    eigenvector_condition_squared_upper
                    / uniform_affine_distance_squared_lower
                )
                _, resolvent_upper, lower_ok, upper_ok = _dyadic_sqrt(
                    resolvent_squared_upper, sqrt_precision
                )
                resolvent_sqrt_checks = (lower_ok, upper_ok)
    failures: list[str] = []
    if knot.validation_level is None:
        failures.append("MOVING_KNOT_SPLINE_GEOMETRY_OPTIMIZATION_FAILED")
    if not witness:
        failures.append("MOVING_KNOT_SPLINE_EXACT_DIAGONALIZATION_WITNESS_FAILED")
    if family_margin is None or family_margin <= 0:
        failures.append("MOVING_KNOT_SPLINE_UNIFORM_SPECTRAL_MARGIN_NONPOSITIVE")
    if resolvent_upper is None or not all(resolvent_sqrt_checks):
        failures.append("MOVING_KNOT_SPLINE_QUANTITATIVE_RESOLVENT_BOUND_FAILED")
    selected_projector = _projector(vectors, labels) if witness else None
    projector_identity = bool(
        selected_projector is not None
        and _matmul(selected_projector, selected_projector) == selected_projector
        and _matmul(normalized, selected_projector)
        == _matmul(selected_projector, normalized)
    )
    if not projector_identity:
        failures.append("MOVING_KNOT_SPLINE_PROJECTOR_IDENTITY_FAILED")
    simple_jordan = bool(
        knot.automatic_periodic_cq_junction_verified
        and radial_minimum is not None
        and radial_minimum > 0
        and determinant > 0
    )
    if not simple_jordan:
        failures.append("MOVING_KNOT_SPLINE_SIMPLE_JORDAN_FAMILY_FAILED")
    success = not failures
    status = (
        "VERIFIED_MOVING_KNOT_SPLINE_SPECTRAL_RANK_BRIDGE"
        if success
        else failures[0]
    )
    return VerifiedMovingKnotSplineSpectralRankBridge(
        status=status,
        validation_level=status if success else None,
        failure_codes=tuple(failures),
        knot_optimization=knot,
        normalized_transition=normalized,
        normalized_eigenvalues=normalized_values,
        target_inside_labels=labels,
        exact_diagonalization_witness_verified=witness,
        normalized_center=normalized_center,
        normalized_axis_u=normalized_axis_u,
        normalized_axis_v=normalized_axis_v,
        affine_orientation_determinant=determinant,
        affine_inverse_coordinate_squared=coordinate_squared,
        uniform_radial_minimum_lower=radial_minimum,
        uniform_radial_maximum_upper=radial_maximum,
        per_eigenvalue_uniform_signed_squared_margins=margins,
        uniform_family_margin=family_margin,
        positive_uniform_family_margin_verified=success and family_margin is not None and family_margin > 0,
        affine_frobenius_norm_squared=affine_frobenius_squared,
        affine_minimum_singular_value_squared_lower=affine_minimum_singular_squared_lower,
        eigenvector_frobenius_condition_squared_upper=eigenvector_condition_squared_upper,
        per_eigenvalue_normalized_contour_distance_squared_lower=normalized_distance_squared_lower,
        uniform_normalized_contour_distance_squared_lower=uniform_normalized_distance_squared_lower,
        uniform_affine_contour_distance_squared_lower=uniform_affine_distance_squared_lower,
        quantitative_resolvent_norm_squared_upper=resolvent_squared_upper,
        selected_projector=selected_projector if success else None,
        selected_projector_rank=sum(labels) if success else None,
        projector_identity_verified=success and projector_identity,
        simple_periodic_cq_jordan_family_verified=success and simple_jordan,
        entire_moving_knot_box_resolvent_verified=success,
        moving_knot_family_spectral_split_verified=success,
        moving_knot_family_rank_preserved=success,
        quantitative_resolvent_norm_bound=resolvent_upper if success else None,
        interval_matrix_family_verified=False,
        empirical_matrix_provenance_verified=False,
    )


__all__ = [
    "VerifiedMovingKnotSplineSpectralRankBridge",
    "verified_moving_knot_spline_spectral_rank_bridge",
]
