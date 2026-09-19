"""Deterministic verified subdivision refinement for polygonal Riesz rank."""

from __future__ import annotations

from dataclasses import dataclass

if __package__:
    from .verified_polygon_riesz_quadrature import (
        VerifiedPolygonRieszQuadrature,
        verified_polygon_midpoint_riesz_rank,
    )
else:
    from verified_polygon_riesz_quadrature import (  # type: ignore[no-redef]
        VerifiedPolygonRieszQuadrature,
        verified_polygon_midpoint_riesz_rank,
    )


@dataclass(frozen=True)
class AdaptivePolygonRieszRound:
    round_index: int
    subdivisions: tuple[int, ...]
    quadrature: VerifiedPolygonRieszQuadrature
    refined_edges_after_round: tuple[int, ...]


@dataclass(frozen=True)
class VerifiedAdaptivePolygonRiesz:
    status: str
    validation_level: str | None
    strategy: str
    subdivision_multiplier: int
    maximum_refinements: int
    rounds: tuple[AdaptivePolygonRieszRound, ...]
    final_quadrature: VerifiedPolygonRieszQuadrature
    certified_nominal_rank: int | None
    family_rank_verified: bool
    refinement_budget_exhausted: bool
    deterministic_refinement_verified: bool
    polygon_geometry_held_fixed: bool
    empirical_matrix_provenance_verified: bool
    data_dependent_contour_discovery_used: bool


def verified_adaptive_polygon_midpoint_riesz_rank(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    vertices: object,
    spectral_reference_scale: object,
    initial_subdivisions: object,
    maximum_refinements: int,
    strategy: str = "UNIFORM",
    subdivision_multiplier: int = 2,
    sqrt_precision: int = 32,
    machin_terms: int = 8,
) -> VerifiedAdaptivePolygonRiesz:
    """Refine a fixed polygon by a frozen rule until its rank interval is unique."""
    if type(maximum_refinements) is not int or maximum_refinements < 0:
        raise ValueError("maximum_refinements must be a nonnegative built-in integer")
    if type(subdivision_multiplier) is not int or subdivision_multiplier < 2:
        raise ValueError("subdivision_multiplier must be a built-in integer at least two")
    if strategy not in {"UNIFORM", "MAX_ERROR_TIES"}:
        raise ValueError("strategy must be UNIFORM or MAX_ERROR_TIES")

    current: object = initial_subdivisions
    receipts = []
    final = None
    for round_index in range(maximum_refinements + 1):
        quadrature = verified_polygon_midpoint_riesz_rank(
            nominal_transition,
            uncertainty_radii=uncertainty_radii,
            vertices=vertices,
            spectral_reference_scale=spectral_reference_scale,
            subdivisions=current,
            sqrt_precision=sqrt_precision,
            machin_terms=machin_terms,
        )
        final = quadrature
        if quadrature.nominal_rank_verified or round_index == maximum_refinements:
            refined_edges: tuple[int, ...] = ()
        elif strategy == "UNIFORM":
            refined_edges = tuple(range(len(quadrature.subdivisions_per_edge)))
        else:
            largest = max(quadrature.edge_quadrature_error_uppers)
            refined_edges = tuple(
                index
                for index, error in enumerate(quadrature.edge_quadrature_error_uppers)
                if error == largest
            )
        receipts.append(
            AdaptivePolygonRieszRound(
                round_index=round_index,
                subdivisions=quadrature.subdivisions_per_edge,
                quadrature=quadrature,
                refined_edges_after_round=refined_edges,
            )
        )
        if quadrature.nominal_rank_verified or not refined_edges:
            break
        current = tuple(
            value * subdivision_multiplier if index in refined_edges else value
            for index, value in enumerate(quadrature.subdivisions_per_edge)
        )
    assert final is not None
    verified = final.nominal_rank_verified
    return VerifiedAdaptivePolygonRiesz(
        status=(
            "VERIFIED_ADAPTIVE_RATIONAL_POLYGON_RIESZ_RANK"
            if verified
            else "ADAPTIVE_RATIONAL_POLYGON_RIESZ_REFINEMENT_EXHAUSTED"
        ),
        validation_level=(
            "VERIFIED_ADAPTIVE_RATIONAL_POLYGON_RIESZ_RANK" if verified else None
        ),
        strategy=strategy,
        subdivision_multiplier=subdivision_multiplier,
        maximum_refinements=maximum_refinements,
        rounds=tuple(receipts),
        final_quadrature=final,
        certified_nominal_rank=final.certified_nominal_rank,
        family_rank_verified=verified and final.family_rank_verified,
        refinement_budget_exhausted=not verified,
        deterministic_refinement_verified=True,
        polygon_geometry_held_fixed=True,
        empirical_matrix_provenance_verified=False,
        data_dependent_contour_discovery_used=False,
    )


__all__ = [
    "AdaptivePolygonRieszRound",
    "VerifiedAdaptivePolygonRiesz",
    "verified_adaptive_polygon_midpoint_riesz_rank",
]
