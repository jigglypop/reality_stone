"""CLI for the frozen C1 predictor-to-policy intervention."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .runtime_prediction_guided_metacontrol import (
    ARTIFACT_SCHEMA,
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    _current_environment,
    c1_source_hashes,
    run_c1_stage,
    summarize_c1,
    verify_c1_confirmation_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path: Path | None = None
    manifest = None
    if args.confirmation:
        if args.freeze_manifest is None:
            raise RuntimeError("C1 confirmation requires a frozen manifest")
        manifest_path = args.freeze_manifest.resolve()
        manifest = verify_c1_confirmation_manifest(manifest_path)
    elif args.freeze_manifest is not None:
        raise RuntimeError("C1 development does not accept a freeze manifest")

    stage = "confirmation" if args.confirmation else "development"
    seeds = CONFIRMATION_SEEDS if args.confirmation else DEVELOPMENT_SEEDS
    results = run_c1_stage(stage, confirmation_manifest=manifest_path)
    summary = summarize_c1(
        results,
        stage=stage,
        confirmation_manifest=manifest_path,
    )
    canonical = json.dumps(
        results, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    report = {
        "schema": ARTIFACT_SCHEMA,
        "mode": stage,
        "seed_start": min(seeds),
        "seed_stop_inclusive": max(seeds),
        "result_count": len(results),
        "summary": summary,
        "results_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_sha256": c1_source_hashes(),
        "environment": _current_environment(),
        "confirmation_manifest": manifest,
        "results": results,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "mode": stage,
                "output": str(output),
                "results_sha256": report["results_sha256"],
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
