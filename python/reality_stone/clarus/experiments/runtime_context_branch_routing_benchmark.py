"""Frozen development runner for context-only shared-trunk branch routing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .runtime_context_branch_routing import ROUTES, run_context_branch_seed


DEVELOPMENT_SEEDS = tuple(range(97501, 97517))
CONFIRMATION_SEEDS = tuple(range(99501, 99533))  # Deliberately never opened here.


def run_development() -> dict[str, Any]:
    torch.set_num_threads(1)
    rows = [run_context_branch_seed(seed) for seed in DEVELOPMENT_SEEDS]
    apparatus_all = all(row["status"] != "APPARATUS_INVALID" for row in rows)
    pass_count = sum(row["status"] == "CONTEXT_BRANCH_PASS" for row in rows)
    route_accuracy = {
        route: sum(float(row["routes"][route]["accuracy"]) for row in rows) / len(rows)
        for route in ROUTES
    } if apparatus_all else {}
    route_opposite = {
        route: sum(float(row["routes"][route]["opposite_delivery"]) for row in rows) / len(rows)
        for route in ROUTES
    } if apparatus_all else {}
    result = {
        "status": "GO" if apparatus_all and pass_count >= 15 else (
            "APPARATUS_INVALID" if not apparatus_all else "STOP"
        ),
        "claim_status": (
            "SYNTHETIC_CONTEXT_ENTRY_BRANCH_IDENTIFIED"
            if apparatus_all and pass_count >= 15
            else "CONTEXT_BRANCH_NOT_IDENTIFIED"
        ),
        "development_seed_count": len(DEVELOPMENT_SEEDS),
        "seed_pass_count": pass_count,
        "apparatus_all": apparatus_all,
        "route_mean_accuracy": route_accuracy,
        "route_mean_opposite_delivery": route_opposite,
        "confirmation_opened": False,
        "confirmation_seed_count": len(CONFIRMATION_SEEDS),
        "rows": rows,
    }
    return result


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

