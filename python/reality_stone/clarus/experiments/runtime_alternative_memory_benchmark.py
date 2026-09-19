"""CLI for the frozen M0/M1 alternative-memory development campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .runtime_alternative_memory import (
    AlternativeMemoryConfig,
    m0_capacity_rank_sweep,
    m1_delayed_three_factor,
)


DEVELOPMENT_SEEDS = range(97201, 97217)
CONFIRMATION_SEEDS = range(99201, 99233)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_freeze(path: Path, source_path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN":
        raise SystemExit("confirmation requires a manifest with status FROZEN")
    if payload.get("confirmation_seed_range") != [99201, 99232]:
        raise SystemExit("confirmation seed range does not match the contract")
    if payload.get("source_sha256") != _sha256(source_path):
        raise SystemExit("source hash differs from the frozen manifest")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=("all", "m0", "m1"), default="all")
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    source_path = Path(__file__).with_name("runtime_alternative_memory.py")
    freeze = None
    if args.confirmation:
        if args.freeze_manifest is None:
            raise SystemExit("confirmation is sealed until --freeze-manifest is supplied")
        freeze = _verify_freeze(args.freeze_manifest, source_path)

    seeds = CONFIRMATION_SEEDS if args.confirmation else DEVELOPMENT_SEEDS
    results: list[dict[str, Any]] = []
    for seed in seeds:
        config = AlternativeMemoryConfig(seed=seed)
        row: dict[str, Any] = {"seed": seed}
        if args.route in {"all", "m0"}:
            row["m0"] = m0_capacity_rank_sweep(seed, config)
        if args.route in {"all", "m1"}:
            row["m1"] = m1_delayed_three_factor(seed, config)
        results.append(row)

    result_bytes = json.dumps(results, sort_keys=True).encode("utf-8")
    report = {
        "mode": "confirmation" if args.confirmation else "development",
        "route": args.route,
        "seed_range": [min(seeds), max(seeds)],
        "source_sha256": _sha256(source_path),
        "freeze_manifest": freeze,
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "results": results,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    if not args.quiet:
        print(serialized, end="")


if __name__ == "__main__":
    main()
