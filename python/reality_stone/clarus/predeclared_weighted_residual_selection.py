"""Development-only selection from a frozen exact diagonal-weight menu."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re

if __package__:
    from .verified_rational_contour import _fraction, _matrix
    from .verified_weighted_interval_residual import (
        VerifiedWeightedResidualIntervalCircle,
        _weights,
        verified_weighted_componentwise_residual_circle,
    )
else:
    from verified_rational_contour import _fraction, _matrix  # type: ignore[no-redef]
    from verified_weighted_interval_residual import (  # type: ignore[no-redef]
        VerifiedWeightedResidualIntervalCircle,
        _weights,
        verified_weighted_componentwise_residual_circle,
    )


_CANDIDATE_ID = re.compile(r"[a-z][a-z0-9_-]*")


@dataclass(frozen=True)
class PredeclaredDiagonalWeightCandidate:
    candidate_id: str
    normalized_weights: tuple[Fraction, ...]


@dataclass(frozen=True)
class PredeclaredDiagonalWeightMenu:
    dimension: int
    candidates: tuple[PredeclaredDiagonalWeightCandidate, ...]
    canonical_menu_sha256: str


@dataclass(frozen=True)
class DevelopmentWeightCandidateResult:
    candidate: PredeclaredDiagonalWeightCandidate
    certificate: VerifiedWeightedResidualIntervalCircle
    weighted_only_robust_delta_lower: Fraction | None
    selection_score: Fraction | None


@dataclass(frozen=True)
class PredeclaredWeightedResidualSelectionCertificate:
    status: str
    validation_level: str | None
    claim_scope: str
    failure_codes: tuple[str, ...]
    robust_interior: bool
    menu: PredeclaredDiagonalWeightMenu
    minimum_development_advantage: Fraction
    development_results: tuple[DevelopmentWeightCandidateResult, ...]
    selected_candidate_id: str | None
    selected_normalized_weights: tuple[Fraction, ...] | None
    runner_up_candidate_id: str | None
    development_advantage: Fraction | None
    heldout_certificate: VerifiedWeightedResidualIntervalCircle | None
    heldout_selected_weight_robust_delta_lower: Fraction | None
    heldout_alternative_candidates_evaluated: bool
    selected_weight_confirmed_on_heldout: bool
    empirical_matrix_provenance_verified: bool
    post_hoc_weight_search_admitted: bool


def predeclared_diagonal_weight_menu(
    *, dimension: int, candidate_weights_by_id: object,
) -> PredeclaredDiagonalWeightMenu:
    if type(dimension) is not int or dimension < 1:
        raise ValueError("dimension must be a positive built-in integer")
    if not isinstance(candidate_weights_by_id, dict) or len(candidate_weights_by_id) < 2:
        raise ValueError("candidate_weights_by_id must contain at least two candidates")
    candidates = []
    seen = set()
    for candidate_id in sorted(candidate_weights_by_id):
        if not isinstance(candidate_id, str) or _CANDIDATE_ID.fullmatch(candidate_id) is None:
            raise ValueError("candidate IDs must be canonical lowercase identifiers")
        weights = _weights(candidate_weights_by_id[candidate_id], dimension)
        if weights in seen:
            raise ValueError("candidate menu contains duplicate normalized weights")
        seen.add(weights)
        candidates.append(PredeclaredDiagonalWeightCandidate(candidate_id, weights))
    canonical = tuple(candidates)
    payload = tuple(
        (candidate.candidate_id, tuple(str(value) for value in candidate.normalized_weights))
        for candidate in canonical
    )
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return PredeclaredDiagonalWeightMenu(dimension, canonical, digest)


def predeclared_weighted_residual_selection(
    *,
    menu: PredeclaredDiagonalWeightMenu,
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
) -> PredeclaredWeightedResidualSelectionCertificate:
    if not isinstance(menu, PredeclaredDiagonalWeightMenu):
        raise ValueError("menu must be a predeclared diagonal-weight menu")
    rebuilt_menu = predeclared_diagonal_weight_menu(
        dimension=menu.dimension,
        candidate_weights_by_id={
            candidate.candidate_id: candidate.normalized_weights
            for candidate in menu.candidates
        },
    )
    if rebuilt_menu != menu:
        raise ValueError("menu must be the unchanged output of predeclared_diagonal_weight_menu")
    matrix = _matrix(nominal_transition, "nominal_transition")
    if len(matrix) != menu.dimension:
        raise ValueError("menu dimension must match nominal_transition")
    minimum_advantage = _fraction(minimum_development_advantage, "minimum_development_advantage")
    if minimum_advantage < 0:
        raise ValueError("minimum_development_advantage must be nonnegative")

    development = []
    eligible = []
    for candidate in menu.candidates:
        result = verified_weighted_componentwise_residual_circle(
            nominal_transition,
            uncertainty_radii=development_uncertainty_radii,
            approximate_inverses=development_approximate_inverses,
            diagonal_weights=candidate.normalized_weights,
            center=center, radius=radius,
            spectral_reference_scale=spectral_reference_scale,
            nodes=nodes, sqrt_precision=sqrt_precision,
        )
        translated = tuple(
            node.translated_weighted_node_sigma_lower for node in result.nodes
        )
        weighted_delta = None
        if all(value is not None for value in translated):
            weighted_delta = (
                min(value for value in translated if value is not None)
                - result.base_circle.normalized_chord_upper
            )
        score = (
            weighted_delta
            if result.validation_level is not None
            and weighted_delta is not None
            and weighted_delta > 0
            else None
        )
        item = DevelopmentWeightCandidateResult(candidate, result, weighted_delta, score)
        development.append(item)
        if score is not None:
            eligible.append(item)

    failures = []
    selected = None
    runner = None
    advantage = None
    if not eligible:
        failures.append("WEIGHT_MENU_NO_DEVELOPMENT_CANDIDATE_CERTIFIED")
    else:
        top_score = max(item.selection_score for item in eligible if item.selection_score is not None)
        winners = tuple(item for item in eligible if item.selection_score == top_score)
        if len(winners) != 1:
            failures.append("WEIGHT_MENU_DEVELOPMENT_WINNER_NOT_UNIQUE")
        else:
            selected = winners[0]
            others = tuple(item for item in eligible if item is not selected)
            if others:
                runner_score = max(item.selection_score for item in others if item.selection_score is not None)
                runners = tuple(item for item in others if item.selection_score == runner_score)
                if len(runners) != 1:
                    failures.append("WEIGHT_MENU_DEVELOPMENT_RUNNER_UP_NOT_UNIQUE")
                else:
                    runner = runners[0]
                    assert selected.selection_score is not None and runner.selection_score is not None
                    advantage = selected.selection_score - runner.selection_score
                    if advantage <= minimum_advantage:
                        failures.append("WEIGHT_MENU_DEVELOPMENT_ADVANTAGE_NOT_STRICT")
            else:
                failures.append("WEIGHT_MENU_DEVELOPMENT_RUNNER_UP_UNAVAILABLE")

    heldout = None
    heldout_weighted_delta = None
    if not failures and selected is not None:
        heldout = verified_weighted_componentwise_residual_circle(
            nominal_transition,
            uncertainty_radii=heldout_uncertainty_radii,
            approximate_inverses=heldout_approximate_inverses,
            diagonal_weights=selected.candidate.normalized_weights,
            center=center, radius=radius,
            spectral_reference_scale=spectral_reference_scale,
            nodes=nodes, sqrt_precision=sqrt_precision,
        )
        heldout_translated = tuple(
            node.translated_weighted_node_sigma_lower for node in heldout.nodes
        )
        if all(value is not None for value in heldout_translated):
            heldout_weighted_delta = (
                min(value for value in heldout_translated if value is not None)
                - heldout.base_circle.normalized_chord_upper
            )
        if (
            heldout.validation_level is None
            or heldout_weighted_delta is None
            or heldout_weighted_delta <= 0
        ):
            failures.append("WEIGHT_MENU_SELECTED_CANDIDATE_FAILED_HELDOUT_CONFIRMATION")

    scope = (
        "PREDECLARED_FINITE_DIAGONAL_WEIGHT_MENU_SELECTED_ON_DEVELOPMENT_AND_"
        "SINGLE_SELECTED_CANDIDATE_CONFIRMED_ON_HELDOUT"
    )
    common = dict(
        claim_scope=scope, failure_codes=tuple(failures), robust_interior=not failures,
        menu=menu, minimum_development_advantage=minimum_advantage,
        development_results=tuple(development),
        selected_candidate_id=(None if selected is None else selected.candidate.candidate_id),
        selected_normalized_weights=(None if selected is None else selected.candidate.normalized_weights),
        runner_up_candidate_id=(None if runner is None else runner.candidate.candidate_id),
        development_advantage=advantage,
        heldout_certificate=heldout,
        heldout_selected_weight_robust_delta_lower=heldout_weighted_delta,
        heldout_alternative_candidates_evaluated=False,
        selected_weight_confirmed_on_heldout=(
            heldout is not None
            and heldout.validation_level is not None
            and heldout_weighted_delta is not None
            and heldout_weighted_delta > 0
        ),
        empirical_matrix_provenance_verified=False,
        post_hoc_weight_search_admitted=False,
    )
    if failures:
        return PredeclaredWeightedResidualSelectionCertificate(
            status=failures[0], validation_level=None, **common
        )
    status = "VERIFIED_PREDECLARED_WEIGHT_SELECTION_AND_HELDOUT_RESIDUAL_CONFIRMATION"
    return PredeclaredWeightedResidualSelectionCertificate(
        status=status, validation_level=status, **common
    )


__all__ = [
    "DevelopmentWeightCandidateResult",
    "PredeclaredDiagonalWeightCandidate",
    "PredeclaredDiagonalWeightMenu",
    "PredeclaredWeightedResidualSelectionCertificate",
    "predeclared_diagonal_weight_menu",
    "predeclared_weighted_residual_selection",
]
