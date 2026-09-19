"""Leakage-safe selection from a frozen finite rational radial contour menu."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re

if __package__:
    from .verified_piecewise_rational_radial_contour_residual import (
        _verified_piecewise_rational_geometry,
    )
    from .verified_piecewise_rational_radial_polygon_rank_bridge import (
        VerifiedPiecewiseRationalRadialPolygonRankBridge,
        verified_piecewise_rational_radial_rank_via_inscribed_polygon,
    )
    from .verified_rational_contour import QComplex, _fraction, _matrix, parse_qcomplex
else:
    from verified_piecewise_rational_radial_contour_residual import (  # type: ignore[no-redef]
        _verified_piecewise_rational_geometry,
    )
    from verified_piecewise_rational_radial_polygon_rank_bridge import (  # type: ignore[no-redef]
        VerifiedPiecewiseRationalRadialPolygonRankBridge,
        verified_piecewise_rational_radial_rank_via_inscribed_polygon,
    )
    from verified_rational_contour import (  # type: ignore[no-redef]
        QComplex,
        _fraction,
        _matrix,
        parse_qcomplex,
    )


_CANDIDATE_ID = re.compile(r"[a-z][a-z0-9_-]*")


@dataclass(frozen=True)
class PredeclaredRationalContourCandidate:
    candidate_id: str
    normalized_center: QComplex
    normalized_axis_u: QComplex
    normalized_axis_v: QComplex
    rational_patches: tuple[object, ...]
    junction_order: int
    directions: tuple[QComplex, ...]


@dataclass(frozen=True)
class PredeclaredRationalContourMenu:
    candidates: tuple[PredeclaredRationalContourCandidate, ...]
    canonical_menu_sha256: str


@dataclass(frozen=True)
class DevelopmentRationalContourCandidateResult:
    candidate: PredeclaredRationalContourCandidate
    certificate: VerifiedPiecewiseRationalRadialPolygonRankBridge
    selection_score: Fraction | None


@dataclass(frozen=True)
class PredeclaredRationalContourSelectionCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    menu: PredeclaredRationalContourMenu
    minimum_development_advantage: Fraction
    development_results: tuple[DevelopmentRationalContourCandidateResult, ...]
    selected_candidate_id: str | None
    runner_up_candidate_id: str | None
    development_advantage: Fraction | None
    development_selected_rank: int | None
    heldout_certificate: VerifiedPiecewiseRationalRadialPolygonRankBridge | None
    heldout_selected_robust_delta_lower: Fraction | None
    heldout_selected_rank: int | None
    heldout_alternative_candidates_evaluated: bool
    selected_contour_confirmed_on_heldout: bool
    heldout_rank_consistency_verified: bool
    empirical_matrix_provenance_verified: bool
    post_hoc_contour_search_admitted: bool
    continuous_contour_optimization_verified: bool


def _canonical_patches(contour) -> tuple[object, ...]:
    return tuple(
        (
            (patch.numerator.base, patch.numerator.terms),
            (patch.denominator.base, patch.denominator.terms),
        )
        for patch in contour.patches
    )


def _candidate_payload(candidate: PredeclaredRationalContourCandidate) -> object:
    def qc(value: QComplex):
        return str(value.real), str(value.imag)

    return (
        candidate.candidate_id,
        qc(candidate.normalized_center),
        qc(candidate.normalized_axis_u),
        qc(candidate.normalized_axis_v),
        tuple(
            (
                (str(numerator[0]), tuple((i, j, str(c)) for i, j, c in numerator[1])),
                (str(denominator[0]), tuple((i, j, str(c)) for i, j, c in denominator[1])),
            )
            for numerator, denominator in candidate.rational_patches
        ),
        candidate.junction_order,
        tuple(qc(direction) for direction in candidate.directions),
    )


def predeclared_rational_contour_menu(
    *, candidate_specs_by_id: object, sqrt_precision: int = 32
) -> PredeclaredRationalContourMenu:
    if not isinstance(candidate_specs_by_id, dict) or len(candidate_specs_by_id) < 2:
        raise ValueError("candidate_specs_by_id must contain at least two candidates")
    candidates = []
    seen = set()
    required = {
        "center",
        "axis_u",
        "axis_v",
        "rational_patches",
        "junction_order",
        "directions",
    }
    for candidate_id in sorted(candidate_specs_by_id):
        if not isinstance(candidate_id, str) or _CANDIDATE_ID.fullmatch(candidate_id) is None:
            raise ValueError("candidate IDs must be canonical lowercase identifiers")
        spec = candidate_specs_by_id[candidate_id]
        if not isinstance(spec, dict) or set(spec) != required:
            raise ValueError(f"candidate {candidate_id} must have exactly the canonical contour fields")
        contour = _verified_piecewise_rational_geometry(
            center=parse_qcomplex(spec["center"], f"{candidate_id}.center"),
            axis_u=parse_qcomplex(spec["axis_u"], f"{candidate_id}.axis_u"),
            axis_v=parse_qcomplex(spec["axis_v"], f"{candidate_id}.axis_v"),
            rational_patches=spec["rational_patches"],
            junction_order=spec["junction_order"],
            directions=spec["directions"],
            sqrt_precision=sqrt_precision,
        )
        candidate = PredeclaredRationalContourCandidate(
            candidate_id=candidate_id,
            normalized_center=contour.affine_geometry.center,
            normalized_axis_u=contour.affine_geometry.axis_u,
            normalized_axis_v=contour.affine_geometry.axis_v,
            rational_patches=_canonical_patches(contour),
            junction_order=contour.junction_order,
            directions=contour.affine_geometry.mesh.directions,
        )
        signature = _candidate_payload(candidate)[1:]
        if signature in seen:
            raise ValueError("candidate menu contains an exact duplicate contour specification")
        seen.add(signature)
        candidates.append(candidate)
    canonical = tuple(candidates)
    digest = hashlib.sha256(
        json.dumps(
            tuple(_candidate_payload(candidate) for candidate in canonical),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    return PredeclaredRationalContourMenu(canonical, digest)


def _candidate_spec(candidate: PredeclaredRationalContourCandidate) -> dict[str, object]:
    return {
        "center": candidate.normalized_center,
        "axis_u": candidate.normalized_axis_u,
        "axis_v": candidate.normalized_axis_v,
        "rational_patches": candidate.rational_patches,
        "junction_order": candidate.junction_order,
        "directions": candidate.directions,
    }


def _evaluate_candidate(
    candidate: PredeclaredRationalContourCandidate,
    *,
    nominal_transition: object,
    uncertainty_radii: object,
    scale: Fraction,
    initial_subdivisions: object,
    maximum_refinements: int,
    strategy: str,
    subdivision_multiplier: int,
    sqrt_precision: int,
    machin_terms: int,
) -> VerifiedPiecewiseRationalRadialPolygonRankBridge:
    return verified_piecewise_rational_radial_rank_via_inscribed_polygon(
        nominal_transition,
        uncertainty_radii=uncertainty_radii,
        center=candidate.normalized_center * scale,
        axis_u=candidate.normalized_axis_u * scale,
        axis_v=candidate.normalized_axis_v * scale,
        rational_patches=candidate.rational_patches,
        junction_order=candidate.junction_order,
        directions=candidate.directions,
        spectral_reference_scale=scale,
        initial_subdivisions=initial_subdivisions,
        maximum_refinements=maximum_refinements,
        strategy=strategy,
        subdivision_multiplier=subdivision_multiplier,
        sqrt_precision=sqrt_precision,
        machin_terms=machin_terms,
    )


def predeclared_rational_contour_selection(
    *,
    menu: PredeclaredRationalContourMenu,
    development_nominal_transition: object,
    development_uncertainty_radii: object,
    heldout_nominal_transition: object,
    heldout_uncertainty_radii: object,
    spectral_reference_scale: object,
    minimum_development_advantage: object,
    initial_subdivisions: object,
    maximum_refinements: int,
    strategy: str = "UNIFORM",
    subdivision_multiplier: int = 2,
    sqrt_precision: int = 32,
    machin_terms: int = 8,
) -> PredeclaredRationalContourSelectionCertificate:
    if not isinstance(menu, PredeclaredRationalContourMenu):
        raise ValueError("menu must be a predeclared rational-contour menu")
    rebuilt = predeclared_rational_contour_menu(
        candidate_specs_by_id={
            candidate.candidate_id: _candidate_spec(candidate)
            for candidate in menu.candidates
        },
        sqrt_precision=sqrt_precision,
    )
    if rebuilt != menu:
        raise ValueError("menu must be the unchanged output of predeclared_rational_contour_menu")
    development_matrix = _matrix(development_nominal_transition, "development_nominal_transition")
    heldout_matrix = _matrix(heldout_nominal_transition, "heldout_nominal_transition")
    if len(development_matrix) != len(heldout_matrix):
        raise ValueError("development and heldout matrix dimensions must agree")
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    minimum_advantage = _fraction(
        minimum_development_advantage, "minimum_development_advantage"
    )
    if minimum_advantage < 0:
        raise ValueError("minimum_development_advantage must be nonnegative")

    development = []
    eligible = []
    for candidate in menu.candidates:
        certificate = _evaluate_candidate(
            candidate,
            nominal_transition=development_nominal_transition,
            uncertainty_radii=development_uncertainty_radii,
            scale=scale,
            initial_subdivisions=initial_subdivisions,
            maximum_refinements=maximum_refinements,
            strategy=strategy,
            subdivision_multiplier=subdivision_multiplier,
            sqrt_precision=sqrt_precision,
            machin_terms=machin_terms,
        )
        residual = certificate.rational_certificate
        delta = None if residual is None else residual.normalized_robust_delta_lower
        score = (
            delta
            if certificate.validation_level is not None and delta is not None and delta > 0
            else None
        )
        item = DevelopmentRationalContourCandidateResult(candidate, certificate, score)
        development.append(item)
        if score is not None:
            eligible.append(item)

    failures = []
    selected = None
    runner = None
    advantage = None
    if not eligible:
        failures.append("CONTOUR_MENU_NO_DEVELOPMENT_CANDIDATE_CERTIFIED")
    else:
        top_score = max(item.selection_score for item in eligible if item.selection_score is not None)
        winners = tuple(item for item in eligible if item.selection_score == top_score)
        if len(winners) != 1:
            failures.append("CONTOUR_MENU_DEVELOPMENT_WINNER_NOT_UNIQUE")
        else:
            selected = winners[0]
            others = tuple(item for item in eligible if item is not selected)
            if not others:
                failures.append("CONTOUR_MENU_DEVELOPMENT_RUNNER_UP_UNAVAILABLE")
            else:
                runner_score = max(item.selection_score for item in others if item.selection_score is not None)
                runners = tuple(item for item in others if item.selection_score == runner_score)
                if len(runners) != 1:
                    failures.append("CONTOUR_MENU_DEVELOPMENT_RUNNER_UP_NOT_UNIQUE")
                else:
                    runner = runners[0]
                    assert selected.selection_score is not None and runner.selection_score is not None
                    advantage = selected.selection_score - runner.selection_score
                    if advantage <= minimum_advantage:
                        failures.append("CONTOUR_MENU_DEVELOPMENT_ADVANTAGE_NOT_STRICT")

    heldout = None
    heldout_delta = None
    rank_consistent = False
    if not failures and selected is not None:
        heldout = _evaluate_candidate(
            selected.candidate,
            nominal_transition=heldout_nominal_transition,
            uncertainty_radii=heldout_uncertainty_radii,
            scale=scale,
            initial_subdivisions=initial_subdivisions,
            maximum_refinements=maximum_refinements,
            strategy=strategy,
            subdivision_multiplier=subdivision_multiplier,
            sqrt_precision=sqrt_precision,
            machin_terms=machin_terms,
        )
        residual = heldout.rational_certificate
        heldout_delta = None if residual is None else residual.normalized_robust_delta_lower
        if heldout.validation_level is None or heldout_delta is None or heldout_delta <= 0:
            failures.append("CONTOUR_MENU_SELECTED_CANDIDATE_FAILED_HELDOUT_CONFIRMATION")
        else:
            rank_consistent = (
                heldout.certified_nominal_rank == selected.certificate.certified_nominal_rank
            )
            if not rank_consistent:
                failures.append("CONTOUR_MENU_SELECTED_CANDIDATE_HELDOUT_RANK_MISMATCH")
    status = (
        failures[0]
        if failures
        else "VERIFIED_PREDECLARED_RATIONAL_CONTOUR_SELECTION_AND_HELDOUT_CONFIRMATION"
    )
    return PredeclaredRationalContourSelectionCertificate(
        status=status,
        validation_level=None if failures else status,
        failure_codes=tuple(failures),
        robust_interior=not failures,
        menu=menu,
        minimum_development_advantage=minimum_advantage,
        development_results=tuple(development),
        selected_candidate_id=None if selected is None else selected.candidate.candidate_id,
        runner_up_candidate_id=None if runner is None else runner.candidate.candidate_id,
        development_advantage=advantage,
        development_selected_rank=(
            None if selected is None else selected.certificate.certified_nominal_rank
        ),
        heldout_certificate=heldout,
        heldout_selected_robust_delta_lower=heldout_delta,
        heldout_selected_rank=None if heldout is None else heldout.certified_nominal_rank,
        heldout_alternative_candidates_evaluated=False,
        selected_contour_confirmed_on_heldout=(
            heldout is not None
            and heldout.validation_level is not None
            and heldout_delta is not None
            and heldout_delta > 0
            and rank_consistent
        ),
        heldout_rank_consistency_verified=rank_consistent,
        empirical_matrix_provenance_verified=False,
        post_hoc_contour_search_admitted=False,
        continuous_contour_optimization_verified=False,
    )


__all__ = [
    "DevelopmentRationalContourCandidateResult",
    "PredeclaredRationalContourCandidate",
    "PredeclaredRationalContourMenu",
    "PredeclaredRationalContourSelectionCertificate",
    "predeclared_rational_contour_menu",
    "predeclared_rational_contour_selection",
]
