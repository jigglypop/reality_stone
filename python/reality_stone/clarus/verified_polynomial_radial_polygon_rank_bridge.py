"""Transfer polygon quadrature ranks to positive polynomial radial contours."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_adaptive_polygon_riesz import (
        VerifiedAdaptivePolygonRiesz,
        verified_adaptive_polygon_midpoint_riesz_rank,
    )
    from .verified_polynomial_radial_contour_residual import (
        ExactPolynomialRadialResidualWitnessConstruction,
        VerifiedResidualRationalPolynomialRadialContour,
        exact_polynomial_radial_nominal_inverse_witnesses,
        verified_componentwise_residual_polynomial_radial_contour,
    )
    from .verified_rational_contour import QComplex, _fraction, parse_qcomplex
else:
    from verified_adaptive_polygon_riesz import (  # type: ignore[no-redef]
        VerifiedAdaptivePolygonRiesz,
        verified_adaptive_polygon_midpoint_riesz_rank,
    )
    from verified_polynomial_radial_contour_residual import (  # type: ignore[no-redef]
        ExactPolynomialRadialResidualWitnessConstruction,
        VerifiedResidualRationalPolynomialRadialContour,
        exact_polynomial_radial_nominal_inverse_witnesses,
        verified_componentwise_residual_polynomial_radial_contour,
    )
    from verified_rational_contour import (  # type: ignore[no-redef]
        QComplex,
        _fraction,
        parse_qcomplex,
    )


@dataclass(frozen=True)
class VerifiedPolynomialRadialPolygonRankBridge:
    status: str
    validation_level: str | None
    polynomial_construction: ExactPolynomialRadialResidualWitnessConstruction
    polynomial_certificate: VerifiedResidualRationalPolynomialRadialContour | None
    adaptive_polygon: VerifiedAdaptivePolygonRiesz | None
    raw_inscribed_polygon_vertices: tuple[QComplex, ...]
    normalized_sector_homotopy_cover_upper: Fraction
    normalized_sector_homotopy_margin_lower: Fraction | None
    sector_homotopy_resolvent_verified: bool
    certified_nominal_rank: int | None
    polynomial_family_rank_verified: bool
    polynomial_projector_quadrature: tuple[tuple[QComplex, ...], ...] | None
    polynomial_projector_operator_error_upper: Fraction | None
    empirical_matrix_provenance_verified: bool
    smooth_contour_quadrature_rank_verified: bool
    contour_geometry_selected_from_data: bool


def verified_polynomial_radial_rank_via_inscribed_polygon(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    center: object,
    axis_u: object,
    axis_v: object,
    radial_base: object,
    radial_terms: object,
    directions: object,
    spectral_reference_scale: object,
    initial_subdivisions: object,
    maximum_refinements: int,
    strategy: str = "UNIFORM",
    subdivision_multiplier: int = 2,
    sqrt_precision: int = 32,
    machin_terms: int = 8,
) -> VerifiedPolynomialRadialPolygonRankBridge:
    """Certify a polynomial radial contour rank via its inscribed polygon."""
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    raw_center = parse_qcomplex(center, "center")
    raw_axis_u = parse_qcomplex(axis_u, "axis_u")
    raw_axis_v = parse_qcomplex(axis_v, "axis_v")
    construction = exact_polynomial_radial_nominal_inverse_witnesses(
        nominal_transition,
        center=raw_center,
        axis_u=raw_axis_u,
        axis_v=raw_axis_v,
        radial_base=radial_base,
        radial_terms=radial_terms,
        directions=directions,
        spectral_reference_scale=scale,
        sqrt_precision=sqrt_precision,
    )
    contour = construction.contour
    raw_vertices = tuple(node * scale for node in contour.contour_nodes)
    common = dict(
        polynomial_construction=construction,
        raw_inscribed_polygon_vertices=raw_vertices,
        normalized_sector_homotopy_cover_upper=contour.normalized_cover_chord_upper,
        empirical_matrix_provenance_verified=False,
        contour_geometry_selected_from_data=False,
    )
    if construction.approximate_inverses is None:
        return VerifiedPolynomialRadialPolygonRankBridge(
            status=construction.status,
            validation_level=None,
            polynomial_certificate=None,
            adaptive_polygon=None,
            normalized_sector_homotopy_margin_lower=None,
            sector_homotopy_resolvent_verified=False,
            certified_nominal_rank=None,
            polynomial_family_rank_verified=False,
            polynomial_projector_quadrature=None,
            polynomial_projector_operator_error_upper=None,
            smooth_contour_quadrature_rank_verified=False,
            **common,
        )
    certificate = verified_componentwise_residual_polynomial_radial_contour(
        nominal_transition,
        uncertainty_radii=uncertainty_radii,
        center=raw_center,
        axis_u=raw_axis_u,
        axis_v=raw_axis_v,
        radial_base=radial_base,
        radial_terms=radial_terms,
        directions=directions,
        approximate_inverses=construction.approximate_inverses,
        spectral_reference_scale=scale,
        sqrt_precision=sqrt_precision,
    )
    if certificate.validation_level is None:
        return VerifiedPolynomialRadialPolygonRankBridge(
            status=certificate.status,
            validation_level=None,
            polynomial_certificate=certificate,
            adaptive_polygon=None,
            normalized_sector_homotopy_margin_lower=(
                certificate.normalized_robust_delta_lower
            ),
            sector_homotopy_resolvent_verified=False,
            certified_nominal_rank=None,
            polynomial_family_rank_verified=False,
            polynomial_projector_quadrature=None,
            polynomial_projector_operator_error_upper=None,
            smooth_contour_quadrature_rank_verified=False,
            **common,
        )
    adaptive = verified_adaptive_polygon_midpoint_riesz_rank(
        nominal_transition,
        uncertainty_radii=uncertainty_radii,
        vertices=raw_vertices,
        spectral_reference_scale=scale,
        initial_subdivisions=initial_subdivisions,
        maximum_refinements=maximum_refinements,
        strategy=strategy,
        subdivision_multiplier=subdivision_multiplier,
        sqrt_precision=sqrt_precision,
        machin_terms=machin_terms,
    )
    rank_verified = adaptive.family_rank_verified
    final = adaptive.final_quadrature
    return VerifiedPolynomialRadialPolygonRankBridge(
        status=(
            "VERIFIED_POLYNOMIAL_RADIAL_POLYGON_HOMOTOPY_RIESZ_RANK"
            if rank_verified
            else "POLYNOMIAL_RADIAL_POLYGON_HOMOTOPY_RANK_UNRESOLVED"
        ),
        validation_level=(
            "VERIFIED_POLYNOMIAL_RADIAL_POLYGON_HOMOTOPY_RIESZ_RANK"
            if rank_verified
            else None
        ),
        polynomial_certificate=certificate,
        adaptive_polygon=adaptive,
        normalized_sector_homotopy_margin_lower=(
            certificate.normalized_robust_delta_lower
        ),
        sector_homotopy_resolvent_verified=True,
        certified_nominal_rank=adaptive.certified_nominal_rank,
        polynomial_family_rank_verified=rank_verified,
        polynomial_projector_quadrature=(
            final.scaled_projector_quadrature if rank_verified else None
        ),
        polynomial_projector_operator_error_upper=(
            final.scaled_projector_operator_error_upper if rank_verified else None
        ),
        smooth_contour_quadrature_rank_verified=rank_verified,
        **common,
    )


__all__ = [
    "VerifiedPolynomialRadialPolygonRankBridge",
    "verified_polynomial_radial_rank_via_inscribed_polygon",
]
