"""Frozen development runner for BA-TR7 source-only symmetry no-go."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .runtime_source_only_symmetry_nogo import run_source_only_symmetry_seed


DEVELOPMENT_SEEDS = tuple(range(97901, 97917))
CONFIRMATION_SEEDS = tuple(range(99901, 99933))  # Deliberately sealed.


def run_development() -> dict[str, Any]:
    torch.set_num_threads(1)
    rows = [run_source_only_symmetry_seed(seed) for seed in DEVELOPMENT_SEEDS]
    no_go_count = sum(row["status"] == "SOURCE_ONLY_SYMMETRY_NO_GO" for row in rows)
    endpoint_closed = all(not row["endpoint_opened"] for row in rows)
    all_gates = all(all(row["gates"].values()) for row in rows)
    mapping_balance = {
        f"{a}{b}": sum(tuple(row["task"]["parity_pair"]) == (a, b) for row in rows)
        for a in (0, 1) for b in (0, 1)
    }
    confirmed = bool(
        no_go_count == len(rows)
        and endpoint_closed
        and all_gates
        and mapping_balance == {"00": 4, "01": 4, "10": 4, "11": 4}
    )
    return {
        "status": "NO_GO_CONFIRMED" if confirmed else "APPARATUS_INVALID",
        "claim_status": "UNIFORM_SOURCE_ONLY_EDGE_SELECTION_NONIDENTIFIABLE",
        "development_seed_count": len(DEVELOPMENT_SEEDS),
        "source_only_nogo_count": no_go_count,
        "all_gates": all_gates,
        "endpoint_opened": False,
        "mapping_balance": mapping_balance,
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
