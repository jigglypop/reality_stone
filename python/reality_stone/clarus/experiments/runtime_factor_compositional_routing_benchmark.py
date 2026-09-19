"""Frozen development runner for BA-TR5 factor composition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .runtime_factor_compositional_routing import (
    EXACT_24_NONORACLE_CONTROLS,
    ROUTES,
    run_factor_composition_seed,
)


DEVELOPMENT_SEEDS = tuple(range(97701, 97717))
CONFIRMATION_SEEDS = tuple(range(99701, 99733))  # Deliberately sealed.


def run_development() -> dict[str, Any]:
    torch.set_num_threads(1)
    rows = [run_factor_composition_seed(seed) for seed in DEVELOPMENT_SEEDS]
    apparatus_all = all(row["status"] != "APPARATUS_INVALID" for row in rows)
    pass_count = sum(row["status"] == "FACTOR_COMPOSITION_PASS" for row in rows)
    route_accuracy = {
        route: sum(float(row["routes"][route]["joint_accuracy"]) for row in rows) / len(rows)
        for route in ROUTES
    } if apparatus_all else {}
    route_A_accuracy = {
        route: sum(float(row["routes"][route]["A_accuracy"]) for row in rows) / len(rows)
        for route in ROUTES
    } if apparatus_all else {}
    route_B_accuracy = {
        route: sum(float(row["routes"][route]["B_accuracy"]) for row in rows) / len(rows)
        for route in ROUTES
    } if apparatus_all else {}
    mapping_balance = {
        f"{a}{b}": sum(tuple(row["task"]["parity_pair"]) == (a, b) for row in rows)
        for a in (0, 1) for b in (0, 1)
    } if apparatus_all else {}
    strongest_control = max(
        (route_accuracy[name] for name in EXACT_24_NONORACLE_CONTROLS),
        default=1.0,
    )
    learned_advantage = (
        route_accuracy["FACTORWISE_LEARNED"] - strongest_control if apparatus_all else -1.0
    )
    batch_gates = {
        "apparatus_all": apparatus_all,
        "seed_pass_count": pass_count >= 15,
        "mapping_balance": mapping_balance == {"00": 4, "01": 4, "10": 4, "11": 4},
        "learned_accuracy": bool(apparatus_all and route_accuracy["FACTORWISE_LEARNED"] >= 0.95),
        "oracle_accuracy": bool(apparatus_all and route_accuracy["ORACLE"] >= 0.95),
        "A_shuffle_joint": bool(apparatus_all and route_accuracy["A_FACTOR_SHUFFLE_TRAIN"] <= 0.05),
        "B_shuffle_joint": bool(apparatus_all and route_accuracy["B_FACTOR_SHUFFLE_TRAIN"] <= 0.05),
        "A_lesion": bool(apparatus_all and route_accuracy["A_LESION_STATIC_0"] <= 0.55),
        "B_lesion": bool(apparatus_all and route_accuracy["B_LESION_STATIC_0"] <= 0.55),
        "static_pairs": bool(apparatus_all and all(
            route_accuracy[name] <= 0.30 for name in ("STATIC_00", "STATIC_01", "STATIC_10", "STATIC_11")
        )),
        "random_matched": bool(apparatus_all and route_accuracy["RANDOM_MATCHED_24"] <= 0.55),
        "full_interference": bool(apparatus_all and route_accuracy["FULL_32"] <= 0.55),
        "learned_advantage": learned_advantage >= 0.40,
    }
    go = all(batch_gates.values())
    return {
        "status": "GO" if go else ("APPARATUS_INVALID" if not apparatus_all else "STOP"),
        "claim_status": (
            "SYNTHETIC_HELDOUT_FACTOR_COMPOSITION"
            if go else "FACTOR_COMPOSITION_NOT_IDENTIFIED"
        ),
        "development_seed_count": len(DEVELOPMENT_SEEDS),
        "seed_pass_count": pass_count,
        "batch_gates": batch_gates,
        "mapping_balance": mapping_balance,
        "route_mean_joint_accuracy": route_accuracy,
        "route_mean_A_accuracy": route_A_accuracy,
        "route_mean_B_accuracy": route_B_accuracy,
        "strongest_exact_24_nonoracle_control_accuracy": strongest_control,
        "learned_control_advantage": learned_advantage,
        "confirmation_opened": False,
        "confirmation_seed_count": len(CONFIRMATION_SEEDS),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development",), default="development")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_development()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
