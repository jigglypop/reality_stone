"""Transfer verified polygon quadrature ranks to smooth rational ellipses."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_adaptive_polygon_riesz import (
        VerifiedAdaptivePolygonRiesz,
        verified_adaptive_polygon_midpoint_riesz_rank,
    )
    from .verified_rational_contour import QComplex, _fraction, parse_qcomplex
    from .verified_rational_ellipse_residual import (
        ExactEllipseResidualWitnessConstruction,
        VerifiedResidualRationalEllipse,
        exact_ellipse_nominal_inverse_witnesses,
        verified_componentwise_residual_rational_ellipse,
    )
else:
    from verified_adaptive_polygon_riesz import (  # type: ignore[no-redef]
        VerifiedAdaptivePolygonRiesz,
        verified_adaptive_polygon_midpoint_riesz_rank,
    )
    from verified_rational_contour import (  # type: ignore[no-redef]
        QComplex,
        _fraction,
        parse_qcomplex,
    )
    from verified_rational_ellipse_residual import (  # type: ignore[no-redef]
        ExactEllipseResidualWitnessConstruction,
        VerifiedResidualRationalEllipse,
        exact_ellipse_nominal_inverse_witnesses,
        verified_componentwise_residual_rational_ellipse,
    )


@dataclass(frozen=True)
class VerifiedEllipsePolygonRankBridge:
    status: str
    validation_level: str | None
    ellipse_construction: ExactEllipseResidualWitnessConstruction
    ellipse_certificate: VerifiedResidualRationalEllipse | None
    adaptive_polygon: VerifiedAdaptivePolygonRiesz | None
    raw_inscribed_polygon_vertices: tuple[QComplex, ...]
    normalized_sector_homotopy_cover_upper: Fraction
    normalized_sector_homotopy_margin_lower: Fraction | None
    sector_homotopy_resolvent_verified: bool
    certified_nominal_rank: int | None
    ellipse_family_rank_verified: bool
    ellipse_projector_quadrature: tuple[tuple[QComplex, ...], ...] | None
    ellipse_projector_operator_error_upper: Fraction | None
    empirical_matrix_provenance_verified: bool
    smooth_contour_quadrature_rank_verified: bool
    contour_geometry_selected_from_data: bool


def verified_ellipse_rank_via_inscribed_polygon(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    center: object,
    axis_u: object,
    axis_v: object,
    directions: object,
    spectral_reference_scale: object,
    initial_subdivisions: object,
    maximum_refinements: int,
    strategy: str = "UNIFORM",
    subdivision_multiplier: int = 2,
    sqrt_precision: int = 32,
    machin_terms: int = 8,
) -> VerifiedEllipsePolygonRankBridge:
    """Certify an ellipse rank through a contour-free arc-to-chord homotopy."""
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    raw_center = parse_qcomplex(center, "center")
    raw_axis_u = parse_qcomplex(axis_u, "axis_u")
    raw_axis_v = parse_qcomplex(axis_v, "axis_v")
    construction = exact_ellipse_nominal_inverse_witnesses(
        nominal_transition,
        center=raw_center,
        axis_u=raw_axis_u,
        axis_v=raw_axis_v,
        directions=directions,
        spectral_reference_scale=scale,
        sqrt_precision=sqrt_precision,
    )
    ellipse = construction.ellipse
    raw_vertices = tuple(node * scale for node in ellipse.contour_nodes)
    if construction.approximate_inverses is None:
        return VerifiedEllipsePolygonRankBridge(
            status=construction.status,
            validation_level=None,
            ellipse_construction=construction,
            ellipse_certificate=None,
            adaptive_polygon=None,
            raw_inscribed_polygon_vertices=raw_vertices,
            normalized_sector_homotopy_cover_upper=ellipse.normalized_cover_chord_upper,
            normalized_sector_homotopy_margin_lower=None,
            sector_homotopy_resolvent_verified=False,
            certified_nominal_rank=None,
            ellipse_family_rank_verified=False,
            ellipse_projector_quadrature=None,
            ellipse_projector_operator_error_upper=None,
            empirical_matrix_provenance_verified=False,
            smooth_contour_quadrature_rank_verified=False,
            contour_geometry_selected_from_data=False,
        )
    ellipse_certificate = verified_componentwise_residual_rational_ellipse(
        nominal_transition,
        uncertainty_radii=uncertainty_radii,
        center=raw_center,
        axis_u=raw_axis_u,
        axis_v=raw_axis_v,
        directions=directions,
        approximate_inverses=construction.approximate_inverses,
        spectral_reference_scale=scale,
        sqrt_precision=sqrt_precision,
    )
    if ellipse_certificate.validation_level is None:
        return VerifiedEllipsePolygonRankBridge(
            status=ellipse_certificate.status,
            validation_level=None,
            ellipse_construction=construction,
            ellipse_certificate=ellipse_certificate,
            adaptive_polygon=None,
            raw_inscribed_polygon_vertices=raw_vertices,
            normalized_sector_homotopy_cover_upper=ellipse.normalized_cover_chord_upper,
            normalized_sector_homotopy_margin_lower=ellipse_certificate.normalized_robust_delta_lower,
            sector_homotopy_resolvent_verified=False,
            certified_nominal_rank=None,
            ellipse_family_rank_verified=False,
            ellipse_projector_quadrature=None,
            ellipse_projector_operator_error_upper=None,
            empirical_matrix_provenance_verified=False,
            smooth_contour_quadrature_rank_verified=False,
            contour_geometry_selected_from_data=False,
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
    return VerifiedEllipsePolygonRankBridge(
        status=(
            "VERIFIED_RATIONAL_ELLIPSE_POLYGON_HOMOTOPY_RIESZ_RANK"
            if rank_verified
            else "RATIONAL_ELLIPSE_POLYGON_HOMOTOPY_RANK_UNRESOLVED"
        ),
        validation_level=(
            "VERIFIED_RATIONAL_ELLIPSE_POLYGON_HOMOTOPY_RIESZ_RANK"
            if rank_verified
            else None
        ),
        ellipse_construction=construction,
        ellipse_certificate=ellipse_certificate,
        adaptive_polygon=adaptive,
        raw_inscribed_polygon_vertices=raw_vertices,
        normalized_sector_homotopy_cover_upper=ellipse.normalized_cover_chord_upper,
        normalized_sector_homotopy_margin_lower=ellipse_certificate.normalized_robust_delta_lower,
        sector_homotopy_resolvent_verified=True,
        certified_nominal_rank=adaptive.certified_nominal_rank,
        ellipse_family_rank_verified=rank_verified,
        ellipse_projector_quadrature=(
            final.scaled_projector_quadrature if rank_verified else None
        ),
        ellipse_projector_operator_error_upper=(
            final.scaled_projector_operator_error_upper if rank_verified else None
        ),
        empirical_matrix_provenance_verified=False,
        smooth_contour_quadrature_rank_verified=rank_verified,
        contour_geometry_selected_from_data=False,
    )


__all__ = [
    "VerifiedEllipsePolygonRankBridge",
    "verified_ellipse_rank_via_inscribed_polygon",
]
