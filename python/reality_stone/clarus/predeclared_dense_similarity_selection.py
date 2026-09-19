"""Development-only selection from a frozen finite exact dense-similarity menu."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re

if __package__:
    from .verified_dense_similarity_interval_residual import (
        VerifiedDenseSimilarityResidualIntervalCircle,
        verified_dense_similarity_componentwise_residual_circle,
    )
    from .verified_rational_contour import QComplex, _fraction, _inverse, _matrix
else:
    from verified_dense_similarity_interval_residual import (  # type: ignore[no-redef]
        VerifiedDenseSimilarityResidualIntervalCircle,
        verified_dense_similarity_componentwise_residual_circle,
    )
    from verified_rational_contour import (  # type: ignore[no-redef]
        QComplex,
        _fraction,
        _inverse,
        _matrix,
    )


_CANDIDATE_ID = re.compile(r"[a-z][a-z0-9_-]*")


@dataclass(frozen=True)
class PredeclaredDenseSimilarityCandidate:
    candidate_id: str
    normalized_similarity: tuple[tuple[QComplex, ...], ...]


@dataclass(frozen=True)
class PredeclaredDenseSimilarityMenu:
    dimension: int
    candidates: tuple[PredeclaredDenseSimilarityCandidate, ...]
    canonical_menu_sha256: str


@dataclass(frozen=True)
class DevelopmentDenseSimilarityCandidateResult:
    candidate: PredeclaredDenseSimilarityCandidate
    certificate: VerifiedDenseSimilarityResidualIntervalCircle
    selection_score: Fraction | None


@dataclass(frozen=True)
class PredeclaredDenseSimilaritySelectionCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    robust_interior: bool
    menu: PredeclaredDenseSimilarityMenu
    minimum_development_advantage: Fraction
    development_results: tuple[DevelopmentDenseSimilarityCandidateResult, ...]
    selected_candidate_id: str | None
    selected_normalized_similarity: tuple[tuple[QComplex, ...], ...] | None
    runner_up_candidate_id: str | None
    development_advantage: Fraction | None
    heldout_certificate: VerifiedDenseSimilarityResidualIntervalCircle | None
    heldout_selected_dense_robust_delta_lower: Fraction | None
    heldout_alternative_candidates_evaluated: bool
    selected_dense_similarity_confirmed_on_heldout: bool
    empirical_matrix_provenance_verified: bool
    post_hoc_dense_similarity_search_admitted: bool


def _normalized_similarity(value: object, dimension: int):
    matrix = _matrix(value, "similarity")
    if len(matrix) != dimension:
        raise ValueError("every similarity dimension must match the menu dimension")
    pivot = next((entry for row in matrix for entry in row if entry != QComplex()), None)
    if pivot is None or _inverse(matrix) is None:
        raise ValueError("every similarity candidate must be exactly invertible")
    return tuple(tuple(entry / pivot for entry in row) for row in matrix)


def predeclared_dense_similarity_menu(
    *, dimension: int, candidate_similarities_by_id: object
) -> PredeclaredDenseSimilarityMenu:
    if type(dimension) is not int or dimension < 1:
        raise ValueError("dimension must be a positive built-in integer")
    if not isinstance(candidate_similarities_by_id, dict) or len(candidate_similarities_by_id) < 2:
        raise ValueError("candidate_similarities_by_id must contain at least two candidates")
    candidates = []
    seen = set()
    for candidate_id in sorted(candidate_similarities_by_id):
        if not isinstance(candidate_id, str) or _CANDIDATE_ID.fullmatch(candidate_id) is None:
            raise ValueError("candidate IDs must be canonical lowercase identifiers")
        similarity = _normalized_similarity(candidate_similarities_by_id[candidate_id], dimension)
        if similarity in seen:
            raise ValueError("candidate menu contains proportional duplicate similarities")
        seen.add(similarity)
        candidates.append(PredeclaredDenseSimilarityCandidate(candidate_id, similarity))
    canonical = tuple(candidates)
    payload = tuple(
        (
            candidate.candidate_id,
            tuple(
                tuple((str(entry.real), str(entry.imag)) for entry in row)
                for row in candidate.normalized_similarity
            ),
        )
        for candidate in canonical
    )
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return PredeclaredDenseSimilarityMenu(dimension, canonical, digest)


def predeclared_dense_similarity_selection(
    *,
    menu: PredeclaredDenseSimilarityMenu,
    nominal_transition: object,
    development_uncertainty_radii: object,
    development_approximate_inverses: object,
    heldout_uncertainty_radii: object,
    heldout_approximate_inverses: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    minimum_development_advantage: object,
    nodes: int = 4,
    sqrt_precision: int = 32,
) -> PredeclaredDenseSimilaritySelectionCertificate:
    if not isinstance(menu, PredeclaredDenseSimilarityMenu):
        raise ValueError("menu must be a predeclared dense-similarity menu")
    rebuilt = predeclared_dense_similarity_menu(
        dimension=menu.dimension,
        candidate_similarities_by_id={
            candidate.candidate_id: candidate.normalized_similarity
            for candidate in menu.candidates
        },
    )
    if rebuilt != menu:
        raise ValueError("menu must be the unchanged output of predeclared_dense_similarity_menu")
    matrix = _matrix(nominal_transition, "nominal_transition")
    if len(matrix) != menu.dimension:
        raise ValueError("menu dimension must match nominal_transition")
    minimum_advantage = _fraction(
        minimum_development_advantage, "minimum_development_advantage"
    )
    if minimum_advantage < 0:
        raise ValueError("minimum_development_advantage must be nonnegative")

    development = []
    eligible = []
    for candidate in menu.candidates:
        certificate = verified_dense_similarity_componentwise_residual_circle(
            nominal_transition,
            uncertainty_radii=development_uncertainty_radii,
            approximate_inverses=development_approximate_inverses,
            similarity=candidate.normalized_similarity,
            center=center,
            radius=radius,
            spectral_reference_scale=spectral_reference_scale,
            nodes=nodes,
            sqrt_precision=sqrt_precision,
        )
        dense_delta = certificate.dense_only_robust_delta_lower
        score = (
            dense_delta
            if certificate.validation_level is not None
            and dense_delta is not None
            and dense_delta > 0
            else None
        )
        item = DevelopmentDenseSimilarityCandidateResult(candidate, certificate, score)
        development.append(item)
        if score is not None:
            eligible.append(item)

    failures = []
    selected = None
    runner = None
    advantage = None
    if not eligible:
        failures.append("DENSE_MENU_NO_DEVELOPMENT_CANDIDATE_CERTIFIED")
    else:
        top_score = max(item.selection_score for item in eligible if item.selection_score is not None)
        winners = tuple(item for item in eligible if item.selection_score == top_score)
        if len(winners) != 1:
            failures.append("DENSE_MENU_DEVELOPMENT_WINNER_NOT_UNIQUE")
        else:
            selected = winners[0]
            others = tuple(item for item in eligible if item is not selected)
            if not others:
                failures.append("DENSE_MENU_DEVELOPMENT_RUNNER_UP_UNAVAILABLE")
            else:
                runner_score = max(item.selection_score for item in others if item.selection_score is not None)
                runners = tuple(item for item in others if item.selection_score == runner_score)
                if len(runners) != 1:
                    failures.append("DENSE_MENU_DEVELOPMENT_RUNNER_UP_NOT_UNIQUE")
                else:
                    runner = runners[0]
                    assert selected.selection_score is not None and runner.selection_score is not None
                    advantage = selected.selection_score - runner.selection_score
                    if advantage <= minimum_advantage:
                        failures.append("DENSE_MENU_DEVELOPMENT_ADVANTAGE_NOT_STRICT")

    heldout = None
    heldout_delta = None
    if not failures and selected is not None:
        heldout = verified_dense_similarity_componentwise_residual_circle(
            nominal_transition,
            uncertainty_radii=heldout_uncertainty_radii,
            approximate_inverses=heldout_approximate_inverses,
            similarity=selected.candidate.normalized_similarity,
            center=center,
            radius=radius,
            spectral_reference_scale=spectral_reference_scale,
            nodes=nodes,
            sqrt_precision=sqrt_precision,
        )
        heldout_delta = heldout.dense_only_robust_delta_lower
        if (
            heldout.validation_level is None
            or heldout_delta is None
            or heldout_delta <= 0
        ):
            failures.append("DENSE_MENU_SELECTED_CANDIDATE_FAILED_HELDOUT_CONFIRMATION")
    status = (
        failures[0]
        if failures
        else "VERIFIED_PREDECLARED_DENSE_SIMILARITY_SELECTION_AND_HELDOUT_CONFIRMATION"
    )
    return PredeclaredDenseSimilaritySelectionCertificate(
        status=status,
        validation_level=None if failures else status,
        failure_codes=tuple(failures),
        robust_interior=not failures,
        menu=menu,
        minimum_development_advantage=minimum_advantage,
        development_results=tuple(development),
        selected_candidate_id=None if selected is None else selected.candidate.candidate_id,
        selected_normalized_similarity=(
            None if selected is None else selected.candidate.normalized_similarity
        ),
        runner_up_candidate_id=None if runner is None else runner.candidate.candidate_id,
        development_advantage=advantage,
        heldout_certificate=heldout,
        heldout_selected_dense_robust_delta_lower=heldout_delta,
        heldout_alternative_candidates_evaluated=False,
        selected_dense_similarity_confirmed_on_heldout=(
            heldout is not None
            and heldout.validation_level is not None
            and heldout_delta is not None
            and heldout_delta > 0
        ),
        empirical_matrix_provenance_verified=False,
        post_hoc_dense_similarity_search_admitted=False,
    )


__all__ = [
    "DevelopmentDenseSimilarityCandidateResult",
    "PredeclaredDenseSimilarityCandidate",
    "PredeclaredDenseSimilarityMenu",
    "PredeclaredDenseSimilaritySelectionCertificate",
    "predeclared_dense_similarity_menu",
    "predeclared_dense_similarity_selection",
]
