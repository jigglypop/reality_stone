"""CLI for the frozen G3-D response/recall diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import torch

from .runtime_metric_memory_diagnostic import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    g3_source_hashes,
    run_g3_stage,
    summarize_g3,
    verify_g3_confirmation_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = None
    manifest = None
    if args.confirmation:
        if args.freeze_manifest is None:
            raise RuntimeError("confirmation requires a frozen manifest")
        manifest_path = args.freeze_manifest.resolve()
        manifest = verify_g3_confirmation_manifest(manifest_path)
    elif args.freeze_manifest is not None:
        raise RuntimeError("development does not accept a freeze manifest")

    stage = "confirmation" if args.confirmation else "development"
    seeds = CONFIRMATION_SEEDS if args.confirmation else DEVELOPMENT_SEEDS
    results = run_g3_stage(stage, confirmation_manifest=manifest_path)
    summary = summarize_g3(
        results, stage=stage, confirmation_manifest=manifest_path,
    )
    canonical = json.dumps(
        results, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    report = {
        "schema": "clarus.runtime_metric_memory_diagnostic.g3d.v1",
        "mode": "confirmation" if args.confirmation else "development",
        "seed_start": min(seeds),
        "seed_stop_inclusive": max(seeds),
        "result_count": len(results),
        "summary": summary,
        "results_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_sha256": g3_source_hashes(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "confirmation_manifest": manifest,
        "results": results,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "mode": report["mode"],
        "output": str(output),
        "results_sha256": report["results_sha256"],
        "summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
