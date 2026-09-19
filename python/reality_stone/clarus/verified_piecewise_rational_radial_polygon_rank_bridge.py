"""Transfer polygon ranks to periodic piecewise-rational radial contours."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_adaptive_polygon_riesz import (
        VerifiedAdaptivePolygonRiesz,
        verified_adaptive_polygon_midpoint_riesz_rank,
    )
    from .verified_piecewise_rational_radial_contour_residual import (
        ExactPiecewiseRationalRadialResidualWitnessConstruction,
        VerifiedResidualPiecewiseRationalRadialContour,
        exact_piecewise_rational_radial_nominal_inverse_witnesses,
        verified_componentwise_residual_piecewise_rational_radial_contour,
    )
    from .verified_rational_contour import QComplex, _fraction, parse_qcomplex
else:
    from verified_adaptive_polygon_riesz import (  # type: ignore[no-redef]
        VerifiedAdaptivePolygonRiesz,
        verified_adaptive_polygon_midpoint_riesz_rank,
    )
    from verified_piecewise_rational_radial_contour_residual import (  # type: ignore[no-redef]
        ExactPiecewiseRationalRadialResidualWitnessConstruction,
        VerifiedResidualPiecewiseRationalRadialContour,
        exact_piecewise_rational_radial_nominal_inverse_witnesses,
        verified_componentwise_residual_piecewise_rational_radial_contour,
    )
    from verified_rational_contour import (  # type: ignore[no-redef]
        QComplex,
        _fraction,
        parse_qcomplex,
    )


@dataclass(frozen=True)
class VerifiedPiecewiseRationalRadialPolygonRankBridge:
    status: str
    validation_level: str | None
    rational_construction: ExactPiecewiseRationalRadialResidualWitnessConstruction
    rational_certificate: VerifiedResidualPiecewiseRationalRadialContour | None
    adaptive_polygon: VerifiedAdaptivePolygonRiesz | None
    raw_inscribed_polygon_vertices: tuple[QComplex, ...]
    normalized_sector_homotopy_cover_upper: Fraction
    normalized_sector_homotopy_margin_lower: Fraction | None
    sector_homotopy_resolvent_verified: bool
    certified_nominal_rank: int | None
    rational_family_rank_verified: bool
    rational_projector_quadrature: tuple[tuple[QComplex, ...], ...] | None
    rational_projector_operator_error_upper: Fraction | None
    empirical_matrix_provenance_verified: bool
    smooth_contour_quadrature_rank_verified: bool
    contour_geometry_selected_from_data: bool


def verified_piecewise_rational_radial_rank_via_inscribed_polygon(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    center: object,
    axis_u: object,
    axis_v: object,
    rational_patches: object,
    junction_order: int,
    directions: object,
    spectral_reference_scale: object,
    initial_subdivisions: object,
    maximum_refinements: int,
    strategy: str = "UNIFORM",
    subdivision_multiplier: int = 2,
    sqrt_precision: int = 32,
    machin_terms: int = 8,
) -> VerifiedPiecewiseRationalRadialPolygonRankBridge:
    """Certify a periodic rational radial Cq contour via its knot polygon."""
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    raw_center = parse_qcomplex(center, "center")
    raw_axis_u = parse_qcomplex(axis_u, "axis_u")
    raw_axis_v = parse_qcomplex(axis_v, "axis_v")
    construction = exact_piecewise_rational_radial_nominal_inverse_witnesses(
        nominal_transition,
        center=raw_center,
        axis_u=raw_axis_u,
        axis_v=raw_axis_v,
        rational_patches=rational_patches,
        junction_order=junction_order,
        directions=directions,
        spectral_reference_scale=scale,
        sqrt_precision=sqrt_precision,
    )
    contour = construction.contour
    raw_vertices = tuple(node * scale for node in contour.contour_nodes)
    common = dict(
        rational_construction=construction,
        raw_inscribed_polygon_vertices=raw_vertices,
        normalized_sector_homotopy_cover_upper=contour.normalized_cover_chord_upper,
        empirical_matrix_provenance_verified=False,
        contour_geometry_selected_from_data=False,
    )
    if construction.approximate_inverses is None:
        return VerifiedPiecewiseRationalRadialPolygonRankBridge(
            status=construction.status,
            validation_level=None,
            rational_certificate=None,
            adaptive_polygon=None,
            normalized_sector_homotopy_margin_lower=None,
            sector_homotopy_resolvent_verified=False,
            certified_nominal_rank=None,
            rational_family_rank_verified=False,
            rational_projector_quadrature=None,
            rational_projector_operator_error_upper=None,
            smooth_contour_quadrature_rank_verified=False,
            **common,
        )
    certificate = verified_componentwise_residual_piecewise_rational_radial_contour(
        nominal_transition,
        uncertainty_radii=uncertainty_radii,
        center=raw_center,
        axis_u=raw_axis_u,
        axis_v=raw_axis_v,
        rational_patches=rational_patches,
        junction_order=junction_order,
        directions=directions,
        approximate_inverses=construction.approximate_inverses,
        spectral_reference_scale=scale,
        sqrt_precision=sqrt_precision,
    )
    if certificate.validation_level is None:
        return VerifiedPiecewiseRationalRadialPolygonRankBridge(
            status=certificate.status,
            validation_level=None,
            rational_certificate=certificate,
            adaptive_polygon=None,
            normalized_sector_homotopy_margin_lower=(
                certificate.normalized_robust_delta_lower
            ),
            sector_homotopy_resolvent_verified=False,
            certified_nominal_rank=None,
            rational_family_rank_verified=False,
            rational_projector_quadrature=None,
            rational_projector_operator_error_upper=None,
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
    return VerifiedPiecewiseRationalRadialPolygonRankBridge(
        status=(
            "VERIFIED_PIECEWISE_RATIONAL_RADIAL_POLYGON_RIESZ_RANK"
            if rank_verified
            else "PIECEWISE_RATIONAL_RADIAL_POLYGON_RANK_UNRESOLVED"
        ),
        validation_level=(
            "VERIFIED_PIECEWISE_RATIONAL_RADIAL_POLYGON_RIESZ_RANK"
            if rank_verified
            else None
        ),
        rational_certificate=certificate,
        adaptive_polygon=adaptive,
        normalized_sector_homotopy_margin_lower=(
            certificate.normalized_robust_delta_lower
        ),
        sector_homotopy_resolvent_verified=True,
        certified_nominal_rank=adaptive.certified_nominal_rank,
        rational_family_rank_verified=rank_verified,
        rational_projector_quadrature=(
            final.scaled_projector_quadrature if rank_verified else None
        ),
        rational_projector_operator_error_upper=(
            final.scaled_projector_operator_error_upper if rank_verified else None
        ),
        smooth_contour_quadrature_rank_verified=rank_verified,
        **common,
    )


__all__ = [
    "VerifiedPiecewiseRationalRadialPolygonRankBridge",
    "verified_piecewise_rational_radial_rank_via_inscribed_polygon",
]
