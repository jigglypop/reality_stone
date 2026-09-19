"""Fail-closed reconstruction ladder from minimal life toward human cognition.

Only completed, independently verified gates unlock the next engineering loop.
Passing a digital loop never upgrades an empirical biological claim by itself.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .origin_life_branching import build_branching_certificate
from .origin_life_branching_verifier import verify_branching_certificate
from .origin_life_coupled import build_coupled_certificate
from .origin_life_coupled_verifier import verify_coupled_certificate
from .origin_life_existence import build_existence_certificate
from .origin_life_existence_verifier import verify_existence_certificate
from .origin_life_finite_resource import build_finite_resource_certificate
from .origin_life_finite_resource_verifier import (
    verify_finite_resource_certificate,
)
from .origin_life_stoichiometric import build_stoichiometric_certificate
from .origin_life_stoichiometric_verifier import (
    verify_stoichiometric_certificate,
)


@dataclass(frozen=True)
class ReconstructionStage:
    stage_id: str
    target: str
    transition_to_validate: str


STAGES = (
    ReconstructionStage(
        "P0",
        "template-bearing protocell",
        "resource-coupled metabolism, copying, boundary growth, and recursive division",
    ),
    ReconstructionStage(
        "P1",
        "adaptive prokaryote (E. coli-like)",
        "internal state and chemotactic action improve viability under causal shifts",
    ),
    ReconstructionStage(
        "P2",
        "single-cell sensorimotor organism (Paramecium-like)",
        "history-dependent sensorimotor control and damage recovery",
    ),
    ReconstructionStage(
        "P3",
        "aneural multicellular coordinator (Trichoplax-like)",
        "distributed cell signalling creates organism-level action selection",
    ),
    ReconstructionStage(
        "P4",
        "nerve-net organism (Hydra-like)",
        "specialized excitable cells coordinate whole-body state and behavior",
    ),
    ReconstructionStage(
        "P5",
        "compact nervous system (C. elegans-like)",
        "weighted recurrent routing predicts held-out stimulus-action trials",
    ),
    ReconstructionStage(
        "P6",
        "centralized insect nervous system (Drosophila-like)",
        "reusable memory and action circuits compose across tasks",
    ),
    ReconstructionStage(
        "P7",
        "vertebrate learner (zebrafish-like)",
        "closed-loop neural dynamics predict continuous adaptive behavior",
    ),
    ReconstructionStage(
        "P8",
        "mammalian agent (mouse-like)",
        "causal world models, replay, and transfer survive perturbational tests",
    ),
    ReconstructionStage(
        "P9",
        "primate social learner",
        "compositional planning, imitation, and theory-of-agent generalize",
    ),
    ReconstructionStage(
        "P10",
        "human-level developmental agent",
        "language, cumulative culture, causal science, and guarded self-modification",
    ),
)


def _verification_row(
    *,
    name: str,
    builder_passed: bool,
    verifier_report: object,
) -> dict[str, object]:
    verified = bool(getattr(verifier_report, "verified"))
    checks = tuple(getattr(verifier_report, "checks"))
    errors = tuple(getattr(verifier_report, "errors"))
    return {
        "name": name,
        "passed": bool(builder_passed and verified),
        "builder_passed": bool(builder_passed),
        "independently_verified": verified,
        "independent_checks": len(checks),
        "errors": list(errors),
    }


def build_reconstruction_loop_status() -> dict[str, object]:
    """Recompute every completed P0 gate and return the next allowed action."""

    existence = build_existence_certificate()
    coupled = build_coupled_certificate()
    branching = build_branching_certificate()
    finite = build_finite_resource_certificate()
    stoichiometric = build_stoichiometric_certificate()

    exact_rows = [
        _verification_row(
            name="recurrent_state_existence",
            builder_passed=bool(existence["all_exact_model_theorems_passed"]),
            verifier_report=verify_existence_certificate(existence),
        ),
        _verification_row(
            name="coupled_heredity_selection",
            builder_passed=bool(coupled["all_exact_model_obligations_passed"]),
            verifier_report=verify_coupled_certificate(coupled),
        ),
        _verification_row(
            name="age_structured_partition_branching",
            builder_passed=bool(branching["all_exact_model_obligations_passed"]),
            verifier_report=verify_branching_certificate(branching),
        ),
    ]
    exact_loop_passed = all(bool(row["passed"]) for row in exact_rows)

    finite_report = verify_finite_resource_certificate(finite)
    finite_row = _verification_row(
        name="explicit_finite_resource_lineage",
        builder_passed=bool(finite["all_engineering_gates_passed"]),
        verifier_report=finite_report,
    )
    stoichiometric_report = verify_stoichiometric_certificate(stoichiometric)
    stoichiometric_row = _verification_row(
        name="stoichiometric_material_energy_geometry",
        builder_passed=bool(stoichiometric["all_stoichiometric_gates_passed"]),
        verifier_report=stoichiometric_report,
    )

    # A downstream gate may have locally passing evidence while an earlier gate
    # is failing.  Such evidence is still useful diagnostically, but it must not
    # advance the reconstruction ladder.  Compute the effective results as a
    # fail-closed prefix so no loop can pass or become ready out of sequence.
    p0_0_passed = exact_loop_passed
    p0_1_prerequisites_passed = p0_0_passed
    p0_1_gate_passed = bool(finite_row["passed"])
    p0_1_passed = p0_1_prerequisites_passed and p0_1_gate_passed
    p0_2_prerequisites_passed = p0_1_passed
    p0_2_gate_passed = bool(stoichiometric_row["passed"])
    p0_2_passed = p0_2_prerequisites_passed and p0_2_gate_passed
    p0_3_prerequisites_passed = p0_2_passed

    loop_results = [
        {
            "loop_id": "P0.0",
            "name": "conditional exact lineage mathematics",
            "status": "pass" if p0_0_passed else "fail",
            "passed": p0_0_passed,
            "evidence": exact_rows,
            "scope": "exact only inside the declared toy models",
        },
        {
            "loop_id": "P0.1",
            "name": "explicit copy bookkeeping under finite shared resources",
            "status": (
                "locked"
                if not p0_1_prerequisites_passed
                else "pass"
                if p0_1_gate_passed
                else "fail"
            ),
            "passed": p0_1_passed,
            "evidence": [finite_row],
            "scope": (
                "integer-token engineering model with finite-horizon recurrence; "
                "not autonomous chemistry"
            ),
        },
        {
            "loop_id": "P0.2",
            "name": "stoichiometric open-flow scaffold and membrane geometry",
            "status": (
                "locked"
                if not p0_2_prerequisites_passed
                else "pass"
                if p0_2_gate_passed
                else "fail"
            ),
            "passed": p0_2_passed,
            "evidence": [stoichiometric_row],
            "scope": (
                "declared coarse reaction network with exact material, carrier, "
                "standard-state free-energy, and membrane-geometry ledgers; "
                "not calibrated autonomous chemistry"
            ),
        },
        {
            "loop_id": "P0.3",
            "name": "mature-offspring supercriticality and robustness",
            "status": "ready" if p0_3_prerequisites_passed else "locked",
            "passed": False,
            "required_gates": [
                "mature daughter reaches its own first division",
                "full model lower confidence bound exceeds replacement",
                "paired ablations and term-specific rescues",
                "density regulation and parameter/environment holdouts",
            ],
        },
    ]

    last_completed_loop = None
    for result in loop_results:
        if not bool(result["passed"]):
            break
        last_completed_loop = result["loop_id"]

    protocell_promoted = all(
        bool(result["passed"]) for result in loop_results
    )

    stage_rows = []
    for index, stage in enumerate(STAGES):
        if index == 0:
            status = "engineering_pass" if protocell_promoted else "in_progress"
        elif index == 1 and protocell_promoted:
            status = "ready"
        else:
            status = "locked"
        stage_rows.append({**asdict(stage), "status": status})

    active_loop = next(
        result["loop_id"]
        for result in loop_results
        if result["status"] in {"ready", "fail"}
    )
    return {
        "artifact_type": "clarus_life_to_human_reconstruction_loop_status",
        "artifact_version": 2,
        "policy": {
            "advance_rule": "advance only after all registered gates pass",
            "failure_rule": "remain at the same loop, change one causal term, version, rerun all gates",
            "claim_rule": "digital engineering pass never implies empirical biological reconstruction",
        },
        "current_stage": "P0",
        "current_stage_target": STAGES[0].target,
        "current_stage_promoted": protocell_promoted,
        "last_completed_loop": last_completed_loop,
        "active_loop": active_loop,
        "next_organism_stage": "P1",
        "next_organism_unlocked": protocell_promoted,
        "loop_results": loop_results,
        "stage_ladder": stage_rows,
        "claim_scope": {
            "finite_resource_lineage_kernel_engineering_passed": bool(finite_row["passed"]),
            "stoichiometric_geometry_scaffold_engineering_passed": bool(
                stoichiometric_row["passed"]
            ),
            "template_bearing_protocell_engineering_promoted": protocell_promoted,
            "empirical_autonomous_protocell_proven": False,
            "human_reconstruction_complete": False,
        },
    }


def validate_reconstruction_loop_status(status: Mapping[str, object]) -> bool:
    return dict(status) == build_reconstruction_loop_status()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="artifacts/biology/origin_life_reconstruction_loop_status.json",
    )
    parser.add_argument(
        "--require-stage-promotion",
        action="store_true",
        help="return nonzero until every loop in the current stage has passed",
    )
    args = parser.parse_args(argv)
    status = build_reconstruction_loop_status()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return int(args.require_stage_promotion and not status["current_stage_promoted"])


if __name__ == "__main__":
    raise SystemExit(main())
