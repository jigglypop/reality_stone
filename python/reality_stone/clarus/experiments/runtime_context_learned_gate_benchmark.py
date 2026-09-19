"""Frozen development runner for BA-TR4 learned context gating."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .runtime_context_learned_gate import ROUTES, run_learned_context_gate_seed


DEVELOPMENT_SEEDS = tuple(range(97601, 97617))
CONFIRMATION_SEEDS = tuple(range(99601, 99633))  # Deliberately sealed.


def run_development() -> dict[str, Any]:
    torch.set_num_threads(1)
    rows = [run_learned_context_gate_seed(seed) for seed in DEVELOPMENT_SEEDS]
    apparatus_all = all(row["status"] != "APPARATUS_INVALID" for row in rows)
    pass_count = sum(row["status"] == "LEARNED_CONTEXT_GATE_PASS" for row in rows)
    route_accuracy = {
        route: sum(float(row["routes"][route]["accuracy"]) for row in rows) / len(rows)
        for route in ROUTES
    } if apparatus_all else {}
    route_opposite = {
        route: sum(float(row["routes"][route]["opposite_delivery"]) for row in rows) / len(rows)
        for route in ROUTES
    } if apparatus_all else {}
    mappings = [tuple(row["preflight"]["task_mapping"]) for row in rows] if apparatus_all else []
    mapping_balance = {
        "identity": sum(mapping == (0, 1) for mapping in mappings),
        "swap": sum(mapping == (1, 0) for mapping in mappings),
    }
    canonical_ok = bool(
        apparatus_all and route_accuracy["CANONICAL_CUE_MAP"] <= 0.55
        and mapping_balance == {"identity": 8, "swap": 8}
    )
    go = apparatus_all and pass_count >= 15 and canonical_ok
    return {
        "status": "GO" if go else ("APPARATUS_INVALID" if not apparatus_all else "STOP"),
        "claim_status": "SYNTHETIC_LEARNED_CONTEXT_SELECTOR" if go else "LEARNED_CONTEXT_GATE_NOT_IDENTIFIED",
        "development_seed_count": len(DEVELOPMENT_SEEDS),
        "seed_pass_count": pass_count,
        "apparatus_all": apparatus_all,
        "canonical_control_gate": canonical_ok,
        "mapping_balance": mapping_balance,
        "route_mean_accuracy": route_accuracy,
        "route_mean_opposite_delivery": route_opposite,
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
