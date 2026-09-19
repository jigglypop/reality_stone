"""Fixed-point core registry for the CE formula chain.

This module owns the extinction fixed-point core only:

* ``CE_CORE_EXACT_V1`` evaluates the declared CE formula chain in binary64;
* ``LEGACY_DELTA_5DP_V1`` reproduces the historical rounded-``delta`` solver.

It is independent of any cosmology readout: no density partition, no
:math:`\\Omega` mapping, and no observational identification lives here.
Cosmology readouts were removed from this repo (2026-08-23 physics purge);
they live in the ce-cosmo repo and in git history.

``q_ext`` is an extinction probability.  Survival is always represented
explicitly as ``1 - q_ext``.

This module is intentionally standard-library-only so audits can load it by
file path without importing the torch-heavy :mod:`reality_stone.clarus`
package facade.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any


class RegistryRole(str, Enum):
    """How a registry entry may be used."""

    SCIENTIFIC_CORE = "scientific_core"
    CONDITIONAL_MODEL = "conditional_model"
    RESEARCH_CANDIDATE = "research_candidate"
    HISTORICAL_MODEL = "historical_model"
    COMPATIBILITY_BOUNDARY = "compatibility_boundary"
    RESEARCH_TARGET = "research_target"
    OBSERVATION_REFERENCE = "observation_reference"


class FormalStatus(str, Enum):
    """CE formal status; this is not an observational confidence label."""

    DEFINITION = "definition"
    THEOREM = "theorem"
    AXIOM = "axiom"
    DERIVATION = "derivation"
    EMPIRICAL = "empirical"
    INCOMPLETE = "incomplete"
    EXCLUDED = "excluded"


class ModelStatus(str, Enum):
    """Activation boundary for a model/configuration."""

    ACTIVE_MATHEMATICS = "active_mathematics"
    CONDITIONAL = "conditional"
    CANDIDATE = "candidate"
    HISTORICAL = "historical"
    COMPATIBILITY_ONLY = "compatibility_only"
    INCOMPLETE = "incomplete"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class Provenance:
    """Machine-readable provenance attached to a registry entry."""

    source_id: str
    source_kind: str
    source_path: str
    formula_version: str
    precision: str
    note: str = ""


@dataclass(frozen=True)
class FixedPointResult:
    """Certificate for the non-identity extinction fixed point."""

    model_id: str
    d_eff: float
    q_ext: float
    survival: float
    contraction: float
    signed_residual: float
    absolute_residual: float
    iterations: int
    precision: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bootstrap_residual(q_ext: float, d_eff: float) -> float:
    """Return ``q - exp(-D(1-q))`` for an extinction candidate."""

    if not math.isfinite(q_ext) or not math.isfinite(d_eff):
        raise ValueError("q_ext and d_eff must be finite")
    return q_ext - math.exp(-d_eff * (1.0 - q_ext))


def solve_low_extinction_root(
    d_eff: float,
    *,
    model_id: str,
    precision: str,
    max_iterations: int = 100,
) -> FixedPointResult:
    """Solve the stable non-identity extinction root for ``D > 1``.

    Newton steps are safeguarded inside ``(0, 1/D)``.  The upper endpoint is
    below the identity root, so the solver cannot accidentally certify
    ``q=1``.  The returned residual and precision label travel with the value.
    """

    if not math.isfinite(d_eff) or d_eff <= 1.0:
        raise ValueError("the stable non-identity extinction root requires finite D > 1")
    if not model_id:
        raise ValueError("model_id is required")

    lo = 0.0
    hi = 1.0 / d_eff
    q_ext = math.exp(-d_eff)
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        residual = bootstrap_residual(q_ext, d_eff)
        if residual > 0.0:
            hi = q_ext
        else:
            lo = q_ext

        derivative = 1.0 - d_eff * math.exp(-d_eff * (1.0 - q_ext))
        candidate = q_ext - residual / derivative
        if not (lo < candidate < hi) or not math.isfinite(candidate):
            candidate = 0.5 * (lo + hi)

        if candidate == q_ext:
            break
        q_ext = candidate
    else:  # pragma: no cover - defensive branch for altered numerical backends
        raise RuntimeError(f"low-root solve did not converge after {max_iterations} iterations")

    signed_residual = bootstrap_residual(q_ext, d_eff)
    absolute_residual = abs(signed_residual)
    residual_limit = 8.0 * math.ulp(q_ext)
    if not 0.0 < q_ext < 1.0 / d_eff or absolute_residual > residual_limit:
        raise RuntimeError(
            "low-root certificate failed: "
            f"q={q_ext!r}, residual={signed_residual!r}, limit={residual_limit!r}"
        )

    return FixedPointResult(
        model_id=model_id,
        d_eff=d_eff,
        q_ext=q_ext,
        survival=1.0 - q_ext,
        contraction=d_eff * q_ext,
        signed_residual=signed_residual,
        absolute_residual=absolute_residual,
        iterations=iterations,
        precision=precision,
    )


@dataclass(frozen=True)
class CoreChain:
    """Versioned CE formula chain and its fixed-point certificate."""

    model_id: str
    role: RegistryRole
    formal_status: FormalStatus
    status: ModelStatus
    alpha_s: float
    sin2_theta_w: float
    delta: float
    d_eff: float
    fixed_point: FixedPointResult
    provenance: Provenance

    @property
    def q_ext(self) -> float:
        return self.fixed_point.q_ext

    @property
    def survival(self) -> float:
        return self.fixed_point.survival

    @property
    def contraction(self) -> float:
        return self.fixed_point.contraction

    @property
    def residual(self) -> float:
        return self.fixed_point.absolute_residual

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_core_exact_v1() -> CoreChain:
    alpha_s = 0.11789
    sin2_theta_w = 4.0 * alpha_s ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = 3.0 + delta
    fixed_point = solve_low_extinction_root(
        d_eff,
        model_id="CE_CORE_EXACT_V1",
        precision="binary64 formula evaluation from alpha_s=0.11789",
    )
    return CoreChain(
        model_id="CE_CORE_EXACT_V1",
        role=RegistryRole.SCIENTIFIC_CORE,
        formal_status=FormalStatus.DERIVATION,
        status=ModelStatus.ACTIVE_MATHEMATICS,
        alpha_s=alpha_s,
        sin2_theta_w=sin2_theta_w,
        delta=delta,
        d_eff=d_eff,
        fixed_point=fixed_point,
        provenance=Provenance(
            source_id="R-U1-EXACT",
            source_kind="ce_formula_chain",
            source_path="docs/검증_원장/상수_우주론_원장.md",
            formula_version="alpha_s_to_sin2_to_delta_to_D_to_q/v1",
            precision="binary64; no intermediate decimal rounding",
            note="alpha_s scale/scheme and the sin2 mapping remain empirical model inputs",
        ),
    )


def _build_legacy_delta_5dp_v1() -> CoreChain:
    delta = 0.17776
    d_eff = 3.0 + delta
    fixed_point = solve_low_extinction_root(
        d_eff,
        model_id="LEGACY_DELTA_5DP_V1",
        precision="delta rounded to five decimal places before root solve",
    )
    return CoreChain(
        model_id="LEGACY_DELTA_5DP_V1",
        role=RegistryRole.COMPATIBILITY_BOUNDARY,
        formal_status=FormalStatus.AXIOM,
        status=ModelStatus.COMPATIBILITY_ONLY,
        alpha_s=0.11789,
        sin2_theta_w=4.0 * 0.11789 ** (4.0 / 3.0),
        delta=delta,
        d_eff=d_eff,
        fixed_point=fixed_point,
        provenance=Provenance(
            source_id="R-U1-LEGACY",
            source_kind="historical_rounded_input",
            source_path="reality_stone/python/reality_stone/clarus/bootstrap_solver.py",
            formula_version="legacy_delta_5dp/v1",
            precision="delta=0.17776 supplied at five decimal places",
            note="preserved for numerical regression; not the exact-chain value",
        ),
    )


CE_CORE_EXACT_V1 = _build_core_exact_v1()
LEGACY_DELTA_5DP_V1 = _build_legacy_delta_5dp_v1()


def q_ext_exact(core: CoreChain = CE_CORE_EXACT_V1) -> FixedPointResult:
    """Return an independently recomputed exact-chain extinction certificate."""

    if core.model_id != "CE_CORE_EXACT_V1":
        raise ValueError("q_ext_exact requires CE_CORE_EXACT_V1")
    return solve_low_extinction_root(
        core.d_eff,
        model_id=core.model_id,
        precision=core.fixed_point.precision,
    )


__all__ = [
    "CE_CORE_EXACT_V1",
    "LEGACY_DELTA_5DP_V1",
    "CoreChain",
    "FixedPointResult",
    "FormalStatus",
    "ModelStatus",
    "Provenance",
    "RegistryRole",
    "bootstrap_residual",
    "q_ext_exact",
    "solve_low_extinction_root",
]
