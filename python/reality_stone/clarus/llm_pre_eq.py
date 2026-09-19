"""PreEq verifier for LLM candidate answers.

The verifier treats generated answers as finite pre-equality candidates.  A
defect energy scores evidence mismatch and instruction failures, then the
existing finite Gibbs machinery selects or abstains from the manifest answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .pre_eq import gibbs_reweight, normalize_weights


try:
    from . import _rust as _rust_mod

    _rust_llm_pre_eq_fwd = getattr(_rust_mod, "nn_llm_pre_eq_fwd", None)
    _rust_llm_claim_pre_eq_fwd = getattr(_rust_mod, "nn_llm_claim_pre_eq_fwd", None)
except ImportError:
    _rust_llm_pre_eq_fwd = None
    _rust_llm_claim_pre_eq_fwd = None


@dataclass(frozen=True)
class CandidateAnswer:
    """Single LLM candidate and its verifier-visible defect counts."""

    text: str
    prior_weight: float = 1.0
    supported_claims: int = 0
    unsupported_claims: int = 0
    contradicted_claims: int = 0
    instruction_violations: int = 0
    self_contradictions: int = 0
    uncertainty_flags: int = 0


@dataclass(frozen=True)
class EvidenceClaim:
    """Atomic claim-level verifier judgment."""

    text: str
    label: str


@dataclass(frozen=True)
class ClaimAudit:
    """Claim-level audit that can be folded into a candidate defect vector."""

    claims: tuple[EvidenceClaim, ...]
    instruction_violations: int = 0
    self_contradictions: int = 0
    uncertainty_flags: int = 0

    def to_candidate(self, text: str, prior_weight: float = 1.0) -> CandidateAnswer:
        supported = 0
        unsupported = 0
        contradicted = 0
        for claim in self.claims:
            label = claim.label.strip().lower()
            if label == "supported":
                supported += 1
            elif label == "unsupported":
                unsupported += 1
            elif label == "contradicted":
                contradicted += 1
            else:
                raise ValueError(f"unknown claim label: {claim.label}")
        return CandidateAnswer(
            text=text,
            prior_weight=prior_weight,
            supported_claims=supported,
            unsupported_claims=unsupported,
            contradicted_claims=contradicted,
            instruction_violations=self.instruction_violations,
            self_contradictions=self.self_contradictions,
            uncertainty_flags=self.uncertainty_flags,
        )


@dataclass(frozen=True)
class DefectWeights:
    """Weights for the answer defect energy."""

    contradicted: float = 3.0
    unsupported: float = 1.0
    no_evidence: float = 1.0
    coverage: float = 0.2
    instruction: float = 2.0
    self_contradiction: float = 2.0
    uncertainty: float = 0.25


@dataclass(frozen=True)
class PreEqVerifierConfig:
    """Selection policy for manifest answer readout."""

    beta: float = 2.0
    min_gap: float = 0.5
    min_posterior_log_gap: float = 0.0
    max_energy: float = 3.0
    min_manifest_posterior: float = 0.45
    require_defect_minimizer: bool = False
    weights: DefectWeights = DefectWeights()


@dataclass(frozen=True)
class ManifestDecision:
    """Readout from a finite candidate set."""

    selected_index: int | None
    selected_text: str | None
    posterior: np.ndarray
    energies: np.ndarray
    prior: np.ndarray
    defect_min_index: int
    energy_gap: float
    posterior_log_gap: float
    confidence: float
    abstained: bool
    reason: str
    backend: str = "numpy"


@dataclass(frozen=True)
class LabeledCandidateSet:
    """Small benchmark item for verifier regression tests."""

    candidates: tuple[CandidateAnswer, ...]
    correct_index: int | None


@dataclass(frozen=True)
class ClaimAxisEvidence:
    """One source judgment on one dimensionless claim verification axis."""

    axis: str
    value: float
    reference: float
    sigma: float = 1.0
    source_weight: float = 1.0
    source_reliability: float = 1.0
    source_family: str = "default"
    missing: bool = False


@dataclass(frozen=True)
class ResidualClaim:
    """A claim with structured evidence axes for CE residual verification."""

    text: str
    axes: tuple[ClaimAxisEvidence, ...]
    instruction_penalty: float = 0.0
    schema_penalty: float = 0.0


@dataclass(frozen=True)
class ClaimGraphEdge:
    """Signed relation between two claims in one candidate answer."""

    source: int
    target: int
    weight: float = 1.0
    relation: int = 1


@dataclass(frozen=True)
class ResidualAnswerCandidate:
    """Answer candidate as a bundle of claims plus CE tier penalties."""

    text: str
    claims: tuple[ResidualClaim, ...]
    prior_weight: float = 1.0
    graph_edges: tuple[ClaimGraphEdge, ...] = ()
    required_slots: int = 0
    covered_slots: int = 0
    tier_penalty: float = 0.0
    bridge_penalty: float = 0.0
    branch_penalty: float = 0.0
    transition_penalty: float = 0.0
    provenance_penalty: float = 0.0


@dataclass(frozen=True)
class ClaimActionWeights:
    """Weights from docs/4_공학적_활용/11_CE_claim_residual_verifier_formula.md."""

    residual: float = 0.10
    graph: float = 1.20
    tau: float = 0.10
    source: float = 1.50
    independence: float = 0.80
    missing: float = 1.0
    instruction: float = 1.0
    schema: float = 1.0
    coverage: float = 1.0
    unsupported: float = 1.0
    ce_penalty: float = 1.0


@dataclass(frozen=True)
class ClaimResidualVerifierConfig:
    """Selection policy for v2 claim residual verification."""

    beta: float = 4.0
    d_eff: float = 3.17775842
    eps_sigma: float = 1e-6
    n_star: float = 2.0
    min_effective_sources: float = 1.60
    accept_score: float = 0.35
    max_residual_norm: float = 0.5
    max_graph_norm: float = 2.5
    min_gap: float = 0.0
    max_action: float = 1.0
    min_manifest_posterior: float = 0.35
    min_accepted_fraction: float = 0.5
    weights: ClaimActionWeights = ClaimActionWeights()


@dataclass(frozen=True)
class ClaimResidualState:
    """Computed residual/action state for one claim."""

    text: str
    residual_by_axis: Mapping[str, float]
    residual_energy: float
    tau_squared: float
    source_reliability: float
    effective_sources: float
    independence_penalty: float
    missing_penalty: float
    graph_energy: float
    action: float
    accept_score: float
    accepted: bool


@dataclass(frozen=True)
class ResidualAnswerState:
    """Computed residual/action state for one answer candidate."""

    text: str
    claim_states: tuple[ClaimResidualState, ...]
    action: float
    accepted_fraction: float
    coverage_penalty: float
    unsupported_penalty: float
    ce_penalty: float


@dataclass(frozen=True)
class ResidualManifestDecision:
    """Readout for v2 claim residual answer selection."""

    selected_index: int | None
    selected_text: str | None
    accepted_claims: tuple[str, ...]
    posterior: np.ndarray
    actions: np.ndarray
    prior: np.ndarray
    action_min_index: int
    action_gap: float
    confidence: float
    abstained: bool
    reason: str
    backend: str = "numpy"


@dataclass(frozen=True)
class VerificationMetrics:
    """Aggregate numeric verifier metrics."""

    total: int
    answered: int
    correct: int
    abstained: int
    hallucinated: int
    baseline_correct: int
    baseline_hallucinated: int
    defect_baseline_correct: int
    defect_baseline_hallucinated: int

    @property
    def answer_rate(self) -> float:
        return self.answered / self.total if self.total else 0.0

    @property
    def accuracy_on_answered(self) -> float:
        return self.correct / self.answered if self.answered else 0.0

    @property
    def exact_accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def hallucination_rate_on_answered(self) -> float:
        return self.hallucinated / self.answered if self.answered else 0.0

    @property
    def baseline_accuracy(self) -> float:
        return self.baseline_correct / self.total if self.total else 0.0

    @property
    def baseline_hallucination_rate(self) -> float:
        return self.baseline_hallucinated / self.total if self.total else 0.0

    @property
    def defect_baseline_accuracy(self) -> float:
        return self.defect_baseline_correct / self.total if self.total else 0.0

    @property
    def defect_baseline_hallucination_rate(self) -> float:
        return self.defect_baseline_hallucinated / self.total if self.total else 0.0


class PreEqVerifier:
    """Finite PreEq selector for generated LLM answers."""

    def __init__(self, config: PreEqVerifierConfig | None = None) -> None:
        self.config = config or PreEqVerifierConfig()
        if self.config.beta < 0.0:
            raise ValueError("beta must be non-negative")
        if self.config.min_gap < 0.0:
            raise ValueError("min_gap must be non-negative")
        if self.config.min_posterior_log_gap < 0.0:
            raise ValueError("min_posterior_log_gap must be non-negative")
        if self.config.max_energy < 0.0:
            raise ValueError("max_energy must be non-negative")
        if not 0.0 <= self.config.min_manifest_posterior <= 1.0:
            raise ValueError("min_manifest_posterior must be in [0, 1]")
        weight_values = (
            self.config.weights.contradicted,
            self.config.weights.unsupported,
            self.config.weights.no_evidence,
            self.config.weights.coverage,
            self.config.weights.instruction,
            self.config.weights.self_contradiction,
            self.config.weights.uncertainty,
        )
        if any((not math.isfinite(value)) or value < 0.0 for value in weight_values):
            raise ValueError("defect weights must be finite and non-negative")

    def defect_components(self, candidate: CandidateAnswer) -> dict[str, float]:
        """Return the additive defect energy terms before final clipping."""
        if candidate.prior_weight < 0.0 or not math.isfinite(candidate.prior_weight):
            raise ValueError("candidate prior_weight must be finite and non-negative")
        counts = (
            candidate.supported_claims,
            candidate.unsupported_claims,
            candidate.contradicted_claims,
            candidate.instruction_violations,
            candidate.self_contradictions,
            candidate.uncertainty_flags,
        )
        if any(count < 0 for count in counts):
            raise ValueError("candidate defect counts must be non-negative")

        w = self.config.weights
        total_claims = (
            candidate.supported_claims
            + candidate.unsupported_claims
            + candidate.contradicted_claims
        )
        claim_denominator = max(1, total_claims)
        return {
            "contradicted": w.contradicted * candidate.contradicted_claims / claim_denominator,
            "unsupported": w.unsupported * candidate.unsupported_claims / claim_denominator,
            "no_evidence": w.no_evidence if total_claims == 0 else 0.0,
            "coverage": w.coverage / (1.0 + candidate.supported_claims),
            "instruction": w.instruction * candidate.instruction_violations,
            "self_contradiction": w.self_contradiction * candidate.self_contradictions,
            "uncertainty": w.uncertainty * candidate.uncertainty_flags,
        }

    def defect_energy(self, candidate: CandidateAnswer) -> float:
        """Compute the evidence/instruction defect energy for one answer."""
        raw = sum(self.defect_components(candidate).values())
        return max(0.0, float(raw))

    def energies(self, candidates: Sequence[CandidateAnswer]) -> np.ndarray:
        if not candidates:
            raise ValueError("candidates must be non-empty")
        return np.asarray([self.defect_energy(candidate) for candidate in candidates], dtype=float)

    def prior(self, candidates: Sequence[CandidateAnswer]) -> np.ndarray:
        if not candidates:
            raise ValueError("candidates must be non-empty")
        weights = [candidate.prior_weight for candidate in candidates]
        return normalize_weights(weights)

    def _counts(self, candidates: Sequence[CandidateAnswer]) -> tuple[np.ndarray, ...]:
        return (
            np.asarray([c.supported_claims for c in candidates], dtype=float),
            np.asarray([c.unsupported_claims for c in candidates], dtype=float),
            np.asarray([c.contradicted_claims for c in candidates], dtype=float),
            np.asarray([c.instruction_violations for c in candidates], dtype=float),
            np.asarray([c.self_contradictions for c in candidates], dtype=float),
            np.asarray([c.uncertainty_flags for c in candidates], dtype=float),
        )

    def posterior_state(
        self,
        candidates: Sequence[CandidateAnswer],
        *,
        backend: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        prior = self.prior(candidates)
        supported, unsupported, contradicted, instruction, self_contradiction, uncertainty = (
            self._counts(candidates)
        )
        w = self.config.weights
        if backend in {"auto", "rust"} and _rust_llm_pre_eq_fwd is not None:
            energies, posterior = _rust_llm_pre_eq_fwd(
                prior,
                supported,
                unsupported,
                contradicted,
                instruction,
                self_contradiction,
                uncertainty,
                float(self.config.beta),
                float(w.contradicted),
                float(w.unsupported),
                float(w.no_evidence),
                float(w.coverage),
                float(w.instruction),
                float(w.self_contradiction),
                float(w.uncertainty),
            )
            return np.asarray(prior), np.asarray(energies), np.asarray(posterior), "rust"
        if backend == "rust":
            raise RuntimeError("Rust LLM PreEq backend requested but unavailable")
        energies = self.energies(candidates)
        posterior = gibbs_reweight(prior, energies, self.config.beta)
        return prior, energies, posterior, "numpy"

    def select(self, candidates: Sequence[CandidateAnswer], *, backend: str = "auto") -> ManifestDecision:
        """Select a manifest candidate, or abstain if the defect geometry is weak."""
        if not candidates:
            raise ValueError("candidates must be non-empty")

        prior, energies, posterior, used_backend = self.posterior_state(candidates, backend=backend)
        selected_index = int(np.argmax(posterior))
        defect_min_index = int(np.argmin(energies))

        sorted_energies = np.sort(energies[np.isfinite(energies)])
        energy_gap = 0.0
        if sorted_energies.size >= 2:
            energy_gap = float(sorted_energies[1] - sorted_energies[0])
        elif sorted_energies.size == 1:
            energy_gap = math.inf

        sorted_posterior = np.sort(posterior)[::-1]
        posterior_log_gap = math.inf
        if sorted_posterior.size >= 2:
            posterior_log_gap = float(math.log(sorted_posterior[0]) - math.log(sorted_posterior[1]))

        confidence = float(posterior[selected_index])
        best_energy = float(energies[selected_index])
        reason = "selected"
        abstained = False
        if best_energy > self.config.max_energy:
            abstained = True
            reason = "energy_above_threshold"
        elif energy_gap < self.config.min_gap:
            abstained = True
            reason = "gap_below_threshold"
        elif confidence < self.config.min_manifest_posterior:
            abstained = True
            reason = "posterior_below_threshold"
        elif posterior_log_gap < self.config.min_posterior_log_gap:
            abstained = True
            reason = "posterior_gap_below_threshold"
        elif self.config.require_defect_minimizer and selected_index != defect_min_index:
            abstained = True
            reason = "posterior_not_defect_minimizer"

        if abstained:
            return ManifestDecision(
                selected_index=None,
                selected_text=None,
                posterior=posterior,
                energies=energies,
                prior=prior,
                defect_min_index=defect_min_index,
                energy_gap=energy_gap,
                posterior_log_gap=posterior_log_gap,
                confidence=confidence,
                abstained=True,
                reason=reason,
                backend=used_backend,
            )
        return ManifestDecision(
            selected_index=selected_index,
            selected_text=candidates[selected_index].text,
            posterior=posterior,
            energies=energies,
            prior=prior,
            defect_min_index=defect_min_index,
            energy_gap=energy_gap,
            posterior_log_gap=posterior_log_gap,
            confidence=confidence,
            abstained=False,
            reason=reason,
            backend=used_backend,
        )


class ClaimResidualVerifier:
    """CE v2 verifier: dimensionless claim residuals plus PreEq posterior."""

    def __init__(self, config: ClaimResidualVerifierConfig | None = None) -> None:
        self.config = config or ClaimResidualVerifierConfig()
        if self.config.beta < 0.0:
            raise ValueError("beta must be non-negative")
        if self.config.d_eff <= 0.0 or not math.isfinite(self.config.d_eff):
            raise ValueError("d_eff must be positive and finite")
        if self.config.eps_sigma <= 0.0 or not math.isfinite(self.config.eps_sigma):
            raise ValueError("eps_sigma must be positive and finite")
        if self.config.n_star <= 0.0 or not math.isfinite(self.config.n_star):
            raise ValueError("n_star must be positive and finite")
        if not 0.0 <= self.config.min_manifest_posterior <= 1.0:
            raise ValueError("min_manifest_posterior must be in [0, 1]")
        weights = self.config.weights
        weight_values = (
            weights.residual,
            weights.graph,
            weights.tau,
            weights.source,
            weights.independence,
            weights.missing,
            weights.instruction,
            weights.schema,
            weights.coverage,
            weights.unsupported,
            weights.ce_penalty,
        )
        if any((not math.isfinite(value)) or value < 0.0 for value in weight_values):
            raise ValueError("claim action weights must be finite and non-negative")

    def _axis_residuals(
        self,
        claim: ResidualClaim,
    ) -> tuple[dict[str, float], float, float, float, float, float]:
        if not claim.axes:
            return {}, 0.0, 0.0, 0.0, 0.0, 1.0

        grouped: dict[str, list[ClaimAxisEvidence]] = {}
        for axis in claim.axes:
            if axis.sigma < 0.0 or not math.isfinite(axis.sigma):
                raise ValueError("axis sigma must be finite and non-negative")
            if axis.source_weight < 0.0 or not math.isfinite(axis.source_weight):
                raise ValueError("axis source_weight must be finite and non-negative")
            if not 0.0 <= axis.source_reliability <= 1.0:
                raise ValueError("axis source_reliability must be in [0, 1]")
            if not math.isfinite(axis.value) or not math.isfinite(axis.reference):
                raise ValueError("axis value/reference must be finite")
            grouped.setdefault(axis.axis, []).append(axis)

        residuals: dict[str, float] = {}
        residual_energy = 0.0
        tau_squared = 0.0
        missing_axes = 0
        family_mass: dict[str, float] = {}
        reliability_mass = 0.0
        precision_mass_total = 0.0

        for axis, entries in grouped.items():
            valid = [entry for entry in entries if not entry.missing]
            if not valid:
                missing_axes += 1
                continue
            precision: list[float] = []
            for entry in valid:
                p = (
                    entry.source_weight
                    * entry.source_reliability
                    / (entry.sigma * entry.sigma + self.config.eps_sigma * self.config.eps_sigma)
                )
                precision.append(p)
                family_mass[entry.source_family] = family_mass.get(entry.source_family, 0.0) + p
                reliability_mass += p * entry.source_reliability
                precision_mass_total += p
            total_precision = sum(precision)
            if total_precision <= 0.0:
                missing_axes += 1
                continue
            baseline = sum(p * entry.reference for p, entry in zip(precision, valid)) / total_precision
            disagreement = (
                sum(p * (entry.reference - baseline) ** 2 for p, entry in zip(precision, valid))
                / total_precision
            )
            scale = math.sqrt(1.0 / total_precision + disagreement + self.config.eps_sigma**2)
            value = sum(p * entry.value for p, entry in zip(precision, valid)) / total_precision
            residual = (value - baseline) / scale
            residuals[axis] = residual
            residual_energy += residual * residual
            tau_squared += disagreement / (scale * scale)

        if precision_mass_total > 0.0:
            source_reliability = reliability_mass / precision_mass_total
            mass_sum = sum(family_mass.values())
            mass_sq_sum = sum(value * value for value in family_mass.values())
            effective_sources = (mass_sum * mass_sum / mass_sq_sum) if mass_sq_sum > 0.0 else 0.0
        else:
            source_reliability = 0.0
            effective_sources = 0.0

        independence_penalty = max(
            0.0,
            (self.config.n_star - effective_sources) / self.config.n_star,
        ) ** 2
        missing_penalty = missing_axes / max(1, len(grouped))
        return (
            residuals,
            residual_energy,
            tau_squared,
            source_reliability,
            effective_sources,
            missing_penalty,
        )

    @staticmethod
    def _relation_norm(
        left: Mapping[str, float],
        right: Mapping[str, float],
        relation: int,
    ) -> dict[str, float]:
        axes = set(left) | set(right)
        sign = 1.0 if relation >= 0 else -1.0
        return {axis: sign * right.get(axis, 0.0) - left.get(axis, 0.0) for axis in axes}

    def _graph_vectors(
        self,
        residuals: Sequence[Mapping[str, float]],
        reliabilities: Sequence[float],
        edges: Sequence[ClaimGraphEdge],
    ) -> tuple[dict[str, float], ...]:
        graph_vectors: list[dict[str, float]] = [dict() for _ in residuals]
        n_claims = len(residuals)
        for edge in edges:
            if edge.source < 0 or edge.source >= n_claims or edge.target < 0 or edge.target >= n_claims:
                raise ValueError("graph edge index out of range")
            if edge.weight < 0.0 or not math.isfinite(edge.weight):
                raise ValueError("graph edge weight must be finite and non-negative")
            diff = self._relation_norm(residuals[edge.source], residuals[edge.target], edge.relation)
            scale = edge.weight * reliabilities[edge.target]
            target = graph_vectors[edge.source]
            for axis, value in diff.items():
                target[axis] = target.get(axis, 0.0) + scale * value
        return tuple(graph_vectors)

    def answer_state(self, candidate: ResidualAnswerCandidate) -> ResidualAnswerState:
        if candidate.prior_weight < 0.0 or not math.isfinite(candidate.prior_weight):
            raise ValueError("candidate prior_weight must be finite and non-negative")
        if not candidate.claims:
            coverage_penalty = 1.0 if candidate.required_slots else 0.0
            unsupported_penalty = 1.0
            action = (
                self.config.weights.missing
                + self.config.weights.coverage * coverage_penalty
                + self.config.weights.unsupported * unsupported_penalty
            )
            return ResidualAnswerState(
                text=candidate.text,
                claim_states=(),
                action=action,
                accepted_fraction=0.0,
                coverage_penalty=coverage_penalty,
                unsupported_penalty=unsupported_penalty,
                ce_penalty=0.0,
            )

        base_rows = [self._axis_residuals(claim) for claim in candidate.claims]
        residual_maps = [row[0] for row in base_rows]
        reliabilities = [row[3] for row in base_rows]
        graph_vectors = self._graph_vectors(residual_maps, reliabilities, candidate.graph_edges)

        claim_states: list[ClaimResidualState] = []
        accepted = 0
        for claim, row, graph_vec in zip(candidate.claims, base_rows, graph_vectors):
            residuals, residual_energy, tau_squared, source_reliability, effective_sources, missing = row
            graph_energy = sum(value * value for value in graph_vec.values())
            source_unreliability = (1.0 - source_reliability) ** 2
            w = self.config.weights
            action = (
                w.residual * residual_energy
                + w.graph * graph_energy
                + w.tau * tau_squared
                + w.source * source_unreliability
                + w.independence
                * max(0.0, (self.config.n_star - effective_sources) / self.config.n_star) ** 2
                + w.missing * missing
                + w.instruction * claim.instruction_penalty
                + w.schema * claim.schema_penalty
            )
            accept_score = math.exp(-self.config.d_eff * action)
            graph_norm = math.sqrt(graph_energy)
            residual_norm = math.sqrt(residual_energy)
            is_accepted = (
                accept_score >= self.config.accept_score
                and residual_norm <= self.config.max_residual_norm
                and graph_norm <= self.config.max_graph_norm
                and effective_sources >= self.config.min_effective_sources
                and missing <= 0.0
            )
            accepted += int(is_accepted)
            claim_states.append(
                ClaimResidualState(
                    text=claim.text,
                    residual_by_axis=dict(residuals),
                    residual_energy=residual_energy,
                    tau_squared=tau_squared,
                    source_reliability=source_reliability,
                    effective_sources=effective_sources,
                    independence_penalty=max(
                        0.0,
                        (self.config.n_star - effective_sources) / self.config.n_star,
                    )
                    ** 2,
                    missing_penalty=missing,
                    graph_energy=graph_energy,
                    action=action,
                    accept_score=accept_score,
                    accepted=is_accepted,
                )
            )

        accepted_fraction = accepted / len(claim_states)
        coverage_penalty = 0.0
        if candidate.required_slots > 0:
            covered = min(candidate.covered_slots, candidate.required_slots)
            coverage_penalty = 1.0 - covered / candidate.required_slots
        unsupported_penalty = 1.0 - accepted_fraction
        ce_penalty = (
            candidate.tier_penalty
            + candidate.bridge_penalty
            + candidate.branch_penalty
            + candidate.transition_penalty
            + candidate.provenance_penalty
        )
        mean_claim_action = sum(state.action for state in claim_states) / len(claim_states)
        action = (
            mean_claim_action
            + self.config.weights.coverage * coverage_penalty
            + self.config.weights.unsupported * unsupported_penalty
            + self.config.weights.ce_penalty * ce_penalty
        )
        return ResidualAnswerState(
            text=candidate.text,
            claim_states=tuple(claim_states),
            action=action,
            accepted_fraction=accepted_fraction,
            coverage_penalty=coverage_penalty,
            unsupported_penalty=unsupported_penalty,
            ce_penalty=ce_penalty,
        )

    def answer_states(
        self,
        candidates: Sequence[ResidualAnswerCandidate],
    ) -> tuple[ResidualAnswerState, ...]:
        if not candidates:
            raise ValueError("candidates must be non-empty")
        return tuple(self.answer_state(candidate) for candidate in candidates)

    def _posterior_state(
        self,
        candidates: Sequence[ResidualAnswerCandidate],
        states: Sequence[ResidualAnswerState],
        *,
        backend: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        prior = normalize_weights([candidate.prior_weight for candidate in candidates])
        residual = np.asarray(
            [sum(claim.residual_energy for claim in state.claim_states) / max(1, len(state.claim_states)) for state in states],
            dtype=float,
        )
        graph = np.asarray(
            [sum(claim.graph_energy for claim in state.claim_states) / max(1, len(state.claim_states)) for state in states],
            dtype=float,
        )
        tau = np.asarray(
            [sum(claim.tau_squared for claim in state.claim_states) / max(1, len(state.claim_states)) for state in states],
            dtype=float,
        )
        source = np.asarray(
            [
                sum((1.0 - claim.source_reliability) ** 2 for claim in state.claim_states)
                / max(1, len(state.claim_states))
                for state in states
            ],
            dtype=float,
        )
        independence = np.asarray(
            [
                sum(claim.independence_penalty for claim in state.claim_states) / max(1, len(state.claim_states))
                for state in states
            ],
            dtype=float,
        )
        missing = np.asarray(
            [
                (
                    sum(claim.missing_penalty for claim in state.claim_states)
                    / len(state.claim_states)
                    if state.claim_states
                    else 1.0
                )
                for state in states
            ],
            dtype=float,
        )
        instruction = np.asarray(
            [
                sum(claim.instruction_penalty for claim in candidate.claims) / max(1, len(candidate.claims))
                for candidate in candidates
            ],
            dtype=float,
        )
        schema = np.asarray(
            [
                sum(claim.schema_penalty for claim in candidate.claims) / max(1, len(candidate.claims))
                for candidate in candidates
            ],
            dtype=float,
        )
        coverage = np.asarray([state.coverage_penalty for state in states], dtype=float)
        unsupported = np.asarray([state.unsupported_penalty for state in states], dtype=float)
        ce_penalty = np.asarray([state.ce_penalty for state in states], dtype=float)
        w = self.config.weights
        if backend in {"auto", "rust"} and _rust_llm_claim_pre_eq_fwd is not None:
            actions, posterior = _rust_llm_claim_pre_eq_fwd(
                prior,
                residual,
                graph,
                tau,
                source,
                independence,
                missing,
                instruction,
                schema,
                coverage,
                unsupported,
                ce_penalty,
                float(self.config.beta),
                float(w.residual),
                float(w.graph),
                float(w.tau),
                float(w.source),
                float(w.independence),
                float(w.missing),
                float(w.instruction),
                float(w.schema),
                float(w.coverage),
                float(w.unsupported),
                float(w.ce_penalty),
            )
            return np.asarray(prior), np.asarray(actions), np.asarray(posterior), "rust"
        if backend == "rust":
            raise RuntimeError("Rust LLM claim backend requested but unavailable")
        actions = np.asarray([state.action for state in states], dtype=float)
        posterior = gibbs_reweight(prior, actions, self.config.beta)
        return prior, actions, posterior, "numpy"

    def select(
        self,
        candidates: Sequence[ResidualAnswerCandidate],
        *,
        backend: str = "auto",
    ) -> ResidualManifestDecision:
        states = self.answer_states(candidates)
        prior, actions, posterior, used_backend = self._posterior_state(
            candidates,
            states,
            backend=backend,
        )
        selected_index = int(np.argmax(posterior))
        action_min_index = int(np.argmin(actions))
        sorted_actions = np.sort(actions[np.isfinite(actions)])
        action_gap = math.inf
        if sorted_actions.size >= 2:
            action_gap = float(sorted_actions[1] - sorted_actions[0])
        confidence = float(posterior[selected_index])
        reason = "selected"
        abstained = False
        if float(actions[selected_index]) > self.config.max_action:
            abstained = True
            reason = "action_above_threshold"
        elif action_gap < self.config.min_gap:
            abstained = True
            reason = "gap_below_threshold"
        elif confidence < self.config.min_manifest_posterior:
            abstained = True
            reason = "posterior_below_threshold"
        elif states[selected_index].accepted_fraction < self.config.min_accepted_fraction:
            abstained = True
            reason = "accepted_fraction_below_threshold"

        accepted_claims = tuple(
            claim.text for claim in states[selected_index].claim_states if claim.accepted
        )
        if abstained:
            return ResidualManifestDecision(
                selected_index=None,
                selected_text=None,
                accepted_claims=(),
                posterior=posterior,
                actions=actions,
                prior=prior,
                action_min_index=action_min_index,
                action_gap=action_gap,
                confidence=confidence,
                abstained=True,
                reason=reason,
                backend=used_backend,
            )
        return ResidualManifestDecision(
            selected_index=selected_index,
            selected_text=candidates[selected_index].text,
            accepted_claims=accepted_claims,
            posterior=posterior,
            actions=actions,
            prior=prior,
            action_min_index=action_min_index,
            action_gap=action_gap,
            confidence=confidence,
            abstained=False,
            reason=reason,
            backend=used_backend,
        )


def highest_prior_index(candidates: Sequence[CandidateAnswer]) -> int:
    """Baseline selector: choose the model-prior favorite."""
    if not candidates:
        raise ValueError("candidates must be non-empty")
    prior = normalize_weights([candidate.prior_weight for candidate in candidates])
    return int(np.argmax(prior))


def evaluate_labeled_sets(
    verifier: PreEqVerifier,
    cases: Iterable[LabeledCandidateSet],
) -> VerificationMetrics:
    """Evaluate verifier decisions against labeled finite candidate sets."""
    total = 0
    answered = 0
    correct = 0
    abstained = 0
    hallucinated = 0
    baseline_correct = 0
    baseline_hallucinated = 0
    defect_baseline_correct = 0
    defect_baseline_hallucinated = 0

    for case in cases:
        total += 1
        decision = verifier.select(case.candidates)
        baseline_idx = highest_prior_index(case.candidates)
        defect_baseline_idx = int(np.argmin(verifier.energies(case.candidates)))
        if case.correct_index is not None and baseline_idx == case.correct_index:
            baseline_correct += 1
        else:
            baseline_hallucinated += 1
        if case.correct_index is not None and defect_baseline_idx == case.correct_index:
            defect_baseline_correct += 1
        else:
            defect_baseline_hallucinated += 1

        if decision.abstained:
            abstained += 1
            continue
        answered += 1
        if case.correct_index is not None and decision.selected_index == case.correct_index:
            correct += 1
        else:
            hallucinated += 1

    return VerificationMetrics(
        total=total,
        answered=answered,
        correct=correct,
        abstained=abstained,
        hallucinated=hallucinated,
        baseline_correct=baseline_correct,
        baseline_hallucinated=baseline_hallucinated,
        defect_baseline_correct=defect_baseline_correct,
        defect_baseline_hallucinated=defect_baseline_hallucinated,
    )
