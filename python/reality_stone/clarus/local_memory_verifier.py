"""Independent verifier for the preregistered AML32 local-memory result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


CANONICAL_TEXT_SHA256_SCHEME = "sha256_utf8_lf_normalized_v1"


def canonical_text_sha256(path: str | Path) -> str:
    """Hash UTF-8 text after platform-independent newline normalization.

    Git may materialize the same text with LF or CRLF depending on checkout
    settings.  The confirmatory lock commits every code character while
    treating those two transport encodings as the same implementation.
    """

    payload = Path(path).read_bytes()
    text = payload.decode("utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _portable_path(path: Path) -> str:
    """Use a reproducible workspace-relative path when one is available."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def verify_confirmation(
    preregistration: Mapping[str, object],
    results_by_horizon: Mapping[int, Mapping[str, object]],
    *,
    implementation_sha256: str,
) -> dict[str, object]:
    """Recompute every declared record and panel decision from numeric fields."""

    errors: list[str] = []
    expected_horizons = {
        int(value) for value in preregistration["horizons_required"]  # type: ignore[index]
    }
    if set(results_by_horizon) != expected_horizons:
        errors.append("result horizons do not exactly match preregistration")

    implementation = preregistration["implementation"]  # type: ignore[index]
    locked_implementation_hash = str(implementation["sha256"])  # type: ignore[index]
    if implementation_sha256.lower() != locked_implementation_hash.lower():
        errors.append("implementation hash differs from preregistration")

    panel_spec = preregistration["panel_gate"]  # type: ignore[index]
    recording_spec = preregistration["recording_gate"]  # type: ignore[index]
    confirmatory_panel = preregistration["confirmatory_panel"]  # type: ignore[index]
    expected_ids = set(confirmatory_panel["recording_ids"])  # type: ignore[index]
    expected_archive_hash = str(confirmatory_panel["sha256"])  # type: ignore[index]
    expected_count = int(panel_spec["recording_count"])  # type: ignore[index]
    min_recordings = int(panel_spec["min_recordings_passed_per_horizon"])  # type: ignore[index]
    min_targets = int(recording_spec["min_targets"])  # type: ignore[index]
    min_delta = float(recording_spec["min_median_delta_r2"])  # type: ignore[index]
    min_positive = float(recording_spec["min_positive_target_fraction"])  # type: ignore[index]
    max_p = float(recording_spec["max_null_p_value"])  # type: ignore[index]

    horizon_checks: dict[str, object] = {}
    for horizon in sorted(expected_horizons):
        artifact = results_by_horizon.get(horizon)
        if artifact is None:
            continue
        if artifact.get("phase") != "confirmatory":
            errors.append(f"h={horizon}: phase is not confirmatory")
        provenance = artifact["provenance"]  # type: ignore[index]
        if not provenance["sha256_verified"]:  # type: ignore[index]
            errors.append(f"h={horizon}: source archive hash was not verified")
        if str(provenance["sha256"]).lower() != expected_archive_hash.lower():  # type: ignore[index]
            errors.append(f"h={horizon}: source archive hash differs from preregistration")

        result = artifact["result"]  # type: ignore[index]
        recordings = result["recordings"]  # type: ignore[index]
        observed_ids = {recording["recording_id"] for recording in recordings}
        if len(recordings) != expected_count or observed_ids != expected_ids:
            errors.append(f"h={horizon}: confirmatory recording set changed")
        if int(result["min_recordings_passed"]) != min_recordings:  # type: ignore[index]
            errors.append(f"h={horizon}: panel threshold changed")

        recomputed_passes = 0
        record_checks: list[dict[str, object]] = []
        for recording in recordings:
            criteria = recording["criteria"]
            locked_criteria = (
                int(criteria["min_targets"]) == min_targets
                and float(criteria["min_memory_delta"]) == min_delta
                and float(criteria["min_positive_fraction"]) == min_positive
                and float(criteria["max_null_p"]) == max_p
                and int(criteria["n_null_shifts"]) == 19
            )
            recomputed = (
                locked_criteria
                and int(recording["n_targets_evaluated"]) >= min_targets
                and float(recording["median_delta_memory"]) > min_delta
                and float(recording["positive_fraction_memory"]) >= min_positive
                and float(recording["null_p_value"]) <= max_p
            )
            if bool(recording["passed"]) != recomputed:
                errors.append(
                    f"h={horizon} {recording['recording_id']}: "
                    "reported recording decision is inconsistent"
                )
            if not locked_criteria:
                errors.append(
                    f"h={horizon} {recording['recording_id']}: criteria changed"
                )
            recomputed_passes += int(recomputed)
            record_checks.append(
                {
                    "recording_id": recording["recording_id"],
                    "recomputed_passed": recomputed,
                    "median_delta_memory": recording["median_delta_memory"],
                    "positive_fraction_memory": recording[
                        "positive_fraction_memory"
                    ],
                    "null_p_value": recording["null_p_value"],
                }
            )

        recomputed_panel = recomputed_passes >= min_recordings
        if int(result["recordings_passed"]) != recomputed_passes:  # type: ignore[index]
            errors.append(f"h={horizon}: reported pass count is inconsistent")
        if bool(result["passed"]) != recomputed_panel:  # type: ignore[index]
            errors.append(f"h={horizon}: reported panel decision is inconsistent")
        if bool(artifact["gate_passed"]) != recomputed_panel:  # type: ignore[index]
            errors.append(f"h={horizon}: top-level gate decision is inconsistent")
        horizon_checks[str(horizon)] = {
            "recordings_passed": recomputed_passes,
            "recording_count": len(recordings),
            "required": min_recordings,
            "recomputed_passed": recomputed_panel,
            "recordings": record_checks,
        }

    all_horizons_passed = (
        set(results_by_horizon) == expected_horizons
        and all(
            bool(check["recomputed_passed"])  # type: ignore[index]
            for check in horizon_checks.values()
        )
    )
    return {
        "proof_type": "preregistered_computational_gate_verification",
        "proof_passed": not errors and all_horizons_passed,
        "errors": errors,
        "scope": (
            "held-out predictive information in measured same-unit history, "
            "conditional on the fixed nonlinear current-only baseline"
        ),
        "non_scope": (
            "biological causation, anatomical graph, CloudCell, monad, "
            "consciousness, or AGI"
        ),
        "horizons": horizon_checks,
    }


def build_verification_artifact(
    preregistration_path: str | Path,
    h1_result_path: str | Path,
    h6_result_path: str | Path,
    implementation_path: str | Path,
) -> dict[str, object]:
    preregistration_path = Path(preregistration_path)
    result_paths = {1: Path(h1_result_path), 6: Path(h6_result_path)}
    implementation_path = Path(implementation_path)
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    results = {
        horizon: json.loads(path.read_text(encoding="utf-8"))
        for horizon, path in result_paths.items()
    }
    verification = verify_confirmation(
        preregistration,
        results,
        implementation_sha256=canonical_text_sha256(implementation_path),
    )
    verification["input_hash_scheme"] = CANONICAL_TEXT_SHA256_SCHEME
    verification["inputs"] = {
        "preregistration": {
            "path": _portable_path(preregistration_path),
            "sha256": canonical_text_sha256(preregistration_path),
        },
        "implementation": {
            "path": _portable_path(implementation_path),
            "sha256": canonical_text_sha256(implementation_path),
        },
        "results": {
            str(horizon): {
                "path": _portable_path(path),
                "sha256": canonical_text_sha256(path),
            }
            for horizon, path in result_paths.items()
        },
    }
    return verification


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--h1-result", required=True)
    parser.add_argument("--h6-result", required=True)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    artifact = build_verification_artifact(
        args.preregistration,
        args.h1_result,
        args.h6_result,
        args.implementation,
    )
    payload = json.dumps(artifact, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return int(not artifact["proof_passed"])


__all__ = [
    "CANONICAL_TEXT_SHA256_SCHEME",
    "build_verification_artifact",
    "canonical_text_sha256",
    "verify_confirmation",
]


if __name__ == "__main__":
    raise SystemExit(main())
