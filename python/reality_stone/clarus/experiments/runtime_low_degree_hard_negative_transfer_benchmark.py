"""CLI for BA-TR29 low-degree hard-negative transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .runtime_low_degree_hard_negative_transfer import (
    CALIBRATION_SEEDS,
    DEVELOPMENT_SEEDS,
    analyze_low_degree_hard_negative_artifact,
    generate_low_degree_inputs,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "generate-calibration",
            "generate-development",
            "calibration",
            "development",
        ),
        required=True,
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage.startswith("generate-"):
        seeds = CALIBRATION_SEEDS if args.stage.endswith("calibration") else DEVELOPMENT_SEEDS
        result = generate_low_degree_inputs(seeds)
    else:
        if args.input is None:
            parser.error("--input is required for hard-negative analysis")
        result = analyze_low_degree_hard_negative_artifact(args.input, stage=args.stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "rows"},
            sort_keys=True,
        )
    )
    return 0 if result["status"].endswith(("READY", "PASS", "GO")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
