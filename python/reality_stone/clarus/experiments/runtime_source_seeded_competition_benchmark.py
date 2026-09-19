"""Frozen development runner for BA-TR8 seeded source allocation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .runtime_source_seeded_competition import run_seeded_source_competition_seed


DEVELOPMENT_SEEDS = tuple(range(98001, 98017))
CONFIRMATION_SEEDS = tuple(range(100801, 100833))  # Deliberately sealed.


def run_development() -> dict[str, Any]:
    torch.set_num_threads(1)
    rows = [run_seeded_source_competition_seed(seed) for seed in DEVELOPMENT_SEEDS]
    pass_count = sum(row["status"] == "SEEDED_SOURCE_ALLOCATION_PASS" for row in rows)
    bijection_count = sum(row["gates"]["seeded_capacity_bijection"] for row in rows)
    uniform_abstain_count = sum(
        row["gates"]["uniform_no_capacity_abstains"]
        and row["gates"]["competition_only_uniform_abstains"]
        for row in rows
    )
    raw_collision_mean = sum(float(row["raw_collision_fraction"]) for row in rows) / len(rows)
    order_change_count = sum(bool(row["order_changed_binding"]) for row in rows)
    endpoint_closed = all(not row["endpoint_opened"] for row in rows)
    all_gates = all(all(row["gates"].values()) for row in rows)
    development_go = bool(
        pass_count == len(rows)
        and bijection_count == len(rows)
        and uniform_abstain_count == len(rows)
        and raw_collision_mean > 0.0
        and endpoint_closed
        and all_gates
    )
    return {
        "status": "DEVELOPMENT_GO" if development_go else "DEVELOPMENT_STOP",
        "claim_status": "SEEDED_SOURCE_ALLOCATION_WITH_ENDPOINT_CLOSED",
        "development_seed_count": len(rows),
        "pass_count": pass_count,
        "capacity_bijection_count": bijection_count,
        "uniform_abstain_count": uniform_abstain_count,
        "mean_raw_collision_fraction": raw_collision_mean,
        "mean_capacity_collision_fraction": sum(
            float(row["capacity_collision_fraction"]) for row in rows
        ) / len(rows),
        "order_changed_binding_count": order_change_count,
        "all_gates": all_gates,
        "endpoint_opened": False,
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
