"""CLI for BA-TR12 curvature-cost falsification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .runtime_curvature_cost_prediction import analyze_development_artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze_development_artifact(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, sort_keys=True))
    return 0 if result["status"] in {
        "K_INSUFFICIENT_METRIC_STRAIN_REQUIRED",
        "CURVATURE_ASSOCIATION_NOT_SUFFICIENT",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
