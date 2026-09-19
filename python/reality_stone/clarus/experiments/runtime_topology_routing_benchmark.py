"""Development runner for topology routing after the event-time delay repair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .runtime_topology_routing import (
    ApparatusInvalid,
    run_binding_route,
    run_topology_circuit,
)


DEVELOPMENT_BINDING_SEEDS = tuple(range(97201, 97217))
DEVELOPMENT_FACTOR_SEEDS = tuple(range(97301, 97317))
ROUTES = (
    "FULL",
    "WEIGHT",
    "CLUSTER",
    "PATH_ONLY",
    "TOPOLOGY",
    "RETURN_SHUFFLED",
    "RANDOM_MATCHED",
    "WRONG_CONTEXT",
)


def _binding_pass(row: dict[str, Any]) -> bool:
    return bool(
        row["clean_accuracy"] >= 0.80
        and row["corrupt_accuracy"] >= 0.65
        and row["snapshot_immutable"]
        and row["finite"]
        and row["temporal_rows_after"] == 0
        and row["hippocampal_rows_after_rollout"] == 0
    )


def _integrity(row: dict[str, Any]) -> bool:
    return bool(
        row["budget_exact"]
        and row["snapshot_immutable"]
        and row["finite"]
        and row["temporal_rows_after"] == 0
        and row["hippocampal_rows_after_rollout"] == 0
    )


def run_development() -> dict[str, Any]:
    torch.set_num_threads(1)
    full_binding: list[dict[str, Any]] = []
    topology_binding: list[dict[str, Any]] = []
    factor: list[dict[str, Any]] = []
    try:
        for seed in DEVELOPMENT_BINDING_SEEDS:
            full_binding.append(run_binding_route(seed, route="FULL"))
        for seed in DEVELOPMENT_BINDING_SEEDS:
            topology_binding.append(run_binding_route(seed, route="TOPOLOGY"))
        for seed in DEVELOPMENT_FACTOR_SEEDS:
            factor.append(run_topology_circuit(seed))
    except ApparatusInvalid as error:
        failed_stage = (
            "FULL_BINDING"
            if len(full_binding) < len(DEVELOPMENT_BINDING_SEEDS)
            else "TOPOLOGY_BINDING"
            if len(topology_binding) < len(DEVELOPMENT_BINDING_SEEDS)
            else "FACTOR"
        )
        seeds = (
            DEVELOPMENT_BINDING_SEEDS
            if failed_stage != "FACTOR"
            else DEVELOPMENT_FACTOR_SEEDS
        )
        completed = {
            "FULL_BINDING": len(full_binding),
            "TOPOLOGY_BINDING": len(topology_binding),
            "FACTOR": len(factor),
        }[failed_stage]
        return {
            "status": "APPARATUS_INVALID",
            "reason": str(error),
            "failed_stage": failed_stage,
            "failed_seed": seeds[completed],
            "confirmation_opened": False,
            "binding_full_pass_count": sum(_binding_pass(row) for row in full_binding),
            "binding_full_rows": full_binding,
            "binding_topology_rows": topology_binding,
            "factor_rows": factor,
        }

    route_rows = {
        route: [circuit["routes"][route] for circuit in factor]
        for route in ROUTES
    }
    success = {
        route: sum(int(row["held_out_accuracy"] >= 1.0) for row in rows)
        for route, rows in route_rows.items()
    }
    mean_separation = {
        route: sum(float(row["separation"]) for row in rows) / len(rows)
        for route, rows in route_rows.items()
    }
    full_binding_passes = sum(_binding_pass(row) for row in full_binding)
    topology_binding_passes = sum(_binding_pass(row) for row in topology_binding)
    integrity_all = all(_integrity(row) for rows in route_rows.values() for row in rows)
    binding_integrity_all = all(
        row["snapshot_immutable"]
        and row["finite"]
        and row["temporal_rows_after"] == 0
        and row["hippocampal_rows_after_rollout"] == 0
        for row in (*full_binding, *topology_binding)
    )
    topology_path_difference_count = sum(
        row["topology_path_hamming"] > 0.0 for row in route_rows["TOPOLOGY"]
    )
    same_budget_controls = ("WEIGHT", "CLUSTER", "RANDOM_MATCHED", "WRONG_CONTEXT")
    path_go = bool(
        full_binding_passes >= 15
        and success["TOPOLOGY"] >= 13
        and success["TOPOLOGY"] >= success["FULL"] + 2
        and all(success["TOPOLOGY"] > success[name] for name in same_budget_controls)
        and all(
            mean_separation["TOPOLOGY"] > mean_separation[name]
            for name in same_budget_controls
        )
        and topology_binding_passes >= 15
        and topology_binding_passes >= full_binding_passes - 1
        and integrity_all
        and binding_integrity_all
    )
    topology_specific_go = bool(
        path_go
        and topology_path_difference_count >= 8
        and success["TOPOLOGY"] > success["PATH_ONLY"]
        and success["TOPOLOGY"] > success["RETURN_SHUFFLED"]
        and mean_separation["TOPOLOGY"] > mean_separation["PATH_ONLY"]
        and mean_separation["TOPOLOGY"] > mean_separation["RETURN_SHUFFLED"]
    )
    return {
        "status": "GO" if path_go else "STOP",
        "topology_specific_status": "GO" if topology_specific_go else "STOP",
        "confirmation_opened": False,
        "binding_full_pass_count": full_binding_passes,
        "binding_topology_pass_count": topology_binding_passes,
        "factor_success_count": success,
        "factor_mean_separation": mean_separation,
        "topology_path_difference_count": topology_path_difference_count,
        "integrity_all": integrity_all,
        "binding_integrity_all": binding_integrity_all,
        "binding_full_rows": full_binding,
        "binding_topology_rows": topology_binding,
        "factor_rows": factor,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development",), default="development")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_development()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("_rows")}, sort_keys=True))


if __name__ == "__main__":
    main()
