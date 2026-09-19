"""CLI for the frozen G2 compressed metric-feature utility benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import torch

from .runtime_metric_sufficiency import (
    G2_CONFIRMATION_SEEDS,
    G2_DEVELOPMENT_SEEDS,
    run_g2_seed_range,
    summarize_g2,
)


_REPOSITORY = Path(__file__).resolve().parents[5]
_FREEZE_FILES = (
    "reality_stone/python/reality_stone/clarus/runtime.py",
    "reality_stone/python/reality_stone/clarus/experiments/runtime_metric_sufficiency.py",
    "reality_stone/python/reality_stone/clarus/experiments/runtime_metric_sufficiency_benchmark.py",
    "tests/test_runtime_metric_sufficiency.py",
    "_workspace/ce/brainruntime-weight-metric-dynamics-intervention-20260819/01-g2-contract.md",
    "_workspace/ce/brainruntime-weight-metric-dynamics-intervention-20260819/21-g2-audit.md",
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {name: _file_sha256(_REPOSITORY / name) for name in _FREEZE_FILES}


def _verify_confirmation_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN":
        raise RuntimeError("G2 confirmation manifest status must be FROZEN")
    if manifest.get("development_route_verdict") != "GO":
        raise RuntimeError("G2 confirmation is sealed because development did not GO")
    if manifest.get("files") != _source_hashes():
        raise RuntimeError("G2 confirmation source hash mismatch")
    artifact_name = manifest.get("development_artifact")
    artifact_hash = manifest.get("development_artifact_sha256")
    if not isinstance(artifact_name, str) or not isinstance(artifact_hash, str):
        raise RuntimeError("G2 development provenance is incomplete")
    artifact = _REPOSITORY / artifact_name
    if not artifact.is_file() or _file_sha256(artifact) != artifact_hash:
        raise RuntimeError("G2 development artifact hash mismatch")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = None
    if args.confirmation:
        if args.freeze_manifest is None:
            raise RuntimeError("G2 confirmation requires a frozen manifest")
        manifest = _verify_confirmation_manifest(args.freeze_manifest.resolve())
    elif args.freeze_manifest is not None:
        raise RuntimeError("G2 development does not accept a freeze manifest")

    seeds = G2_CONFIRMATION_SEEDS if args.confirmation else G2_DEVELOPMENT_SEEDS
    results = run_g2_seed_range(seeds)
    summary = summarize_g2(results)
    canonical = json.dumps(
        results, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    report = {
        "schema": "clarus.runtime_metric_sufficiency.g2.v1",
        "mode": "confirmation" if args.confirmation else "development",
        "seed_start": min(seeds),
        "seed_stop_inclusive": max(seeds),
        "result_count": len(results),
        "summary": summary,
        "results_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_sha256": _source_hashes(),
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
