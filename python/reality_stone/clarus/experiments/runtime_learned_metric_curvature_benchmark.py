"""CLI for the BA-TR11 diagnostic over frozen BA-TR10 development weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .runtime_learned_metric_curvature import analyze_development_artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze_development_artifact(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, sort_keys=True))
    return 0 if result["status"] == "CURVATURE_MEMORY_IDENTITY_REJECTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

