"""CLI for the BA-TR10 calibration and development stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .runtime_local_stochastic_binding import (
    CALIBRATION_SEED,
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    LocalStochasticBindingConfig,
    run_local_stochastic_binding_seed,
)


def _summary(stage: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pass_count = sum(row["status"] == "LOCAL_STOCHASTIC_WEIGHT_CODE_PASS" for row in rows)
    all_gates = all(all(bool(value) for value in row["gates"].values()) for row in rows)
    mappings = [tuple(row["learned_evaluation"]["allocation"]["winner_by_source"]) for row in rows]
    source_zero_winners = sorted({mapping[0] for mapping in mappings if mapping[0] >= 0})
    if stage == "calibration":
        decision = "CALIBRATION_PASS" if pass_count == 1 and all_gates else "CALIBRATION_STOP"
    else:
        decision = (
            "DEVELOPMENT_GO"
            if pass_count == len(rows) == len(DEVELOPMENT_SEEDS)
            and all_gates
            and len(set(mappings)) >= 4
            and len(source_zero_winners) == 4
            else "DEVELOPMENT_NO_GO"
        )
    return {
        "stage": stage,
        "decision": decision,
        "seed_count": len(rows),
        "pass_count": pass_count,
        "all_gates": all_gates,
        "unique_learned_mappings": len(set(mappings)),
        "source_zero_winner_coordinates": source_zero_winners,
        "mean_minimum_column_distance": sum(
            float(row["learned"]["minimum_normalized_column_distance"]) for row in rows
        ) / len(rows),
        "no_homeostasis_bijection_count": sum(
            bool(row["no_homeostasis"]["evaluation"]["allocation"]["is_bijection"])
            for row in rows
        ),
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
        raise SystemExit("confirmation is sealed by the BA-TR10 contract")
    seeds = (CALIBRATION_SEED,) if args.stage == "calibration" else DEVELOPMENT_SEEDS
    rows = [
        run_local_stochastic_binding_seed(
            seed,
            config=LocalStochasticBindingConfig(seed=seed),
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

