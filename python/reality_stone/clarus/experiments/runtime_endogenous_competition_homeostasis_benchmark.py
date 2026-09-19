"""CLI for the sealed BA-TR9 calibration and development stages."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Sequence

from .runtime_endogenous_competition_homeostasis import (
    CALIBRATION_SEED,
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    EndogenousCompetitionConfig,
    run_endogenous_competition_seed,
)


def _summary(stage: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pass_count = sum(row["status"] == "ENDOGENOUS_SOURCE_ALLOCATION_PASS" for row in rows)
    mean_collision = sum(float(row["collision_fraction"]) for row in rows) / len(rows)
    mean_control = sum(
        float(row["no_homeostasis_collision_fraction"]) for row in rows
    ) / len(rows)
    reduction = mean_control - mean_collision
    all_gates = all(all(bool(value) for value in row["gates"].values()) for row in rows)
    if stage == "calibration":
        decision = "CALIBRATION_PASS" if pass_count == 1 and all_gates else "CALIBRATION_STOP"
    else:
        decision = (
            "DEVELOPMENT_GO"
            if pass_count == len(rows) == len(DEVELOPMENT_SEEDS)
            and all_gates
            and reduction >= 0.20
            else "DEVELOPMENT_NO_GO"
        )
    return {
        "stage": stage,
        "decision": decision,
        "seed_count": len(rows),
        "pass_count": pass_count,
        "all_gates": all_gates,
        "mean_collision_fraction": mean_collision,
        "mean_no_homeostasis_collision_fraction": mean_control,
        "paired_mean_collision_reduction": reduction,
        "confirmation_opened": False,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "rows": list(rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("calibration", "development", "confirmation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage == "confirmation":
        raise SystemExit("confirmation is sealed by the BA-TR9 contract")
    seeds = (CALIBRATION_SEED,) if args.stage == "calibration" else DEVELOPMENT_SEEDS
    rows = [
        run_endogenous_competition_seed(
            seed,
            config=EndogenousCompetitionConfig(seed=seed),
        )
        for seed in seeds
    ]
    result = _summary(args.stage, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, sort_keys=True))
    return 0 if result["decision"].endswith(("PASS", "GO")) else 1


if __name__ == "__main__":
    raise SystemExit(main())

