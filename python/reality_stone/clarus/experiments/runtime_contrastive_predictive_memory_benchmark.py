"""CLI for the T1/M2/M3 contrastive-predictive memory campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .runtime_contrastive_predictive_memory import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    m2_lagged_contrastive_binding,
    m2_lagged_contrastive_factor_transfer,
    m3_predictor_audit,
    m3_replay_residual_binding,
    m3_replay_residual_factor_transfer,
    t1_m1_factor_transfer,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _t1_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    go_count = sum(row["status"] == "GO" for row in rows)
    integrity = (
        all(row["heldout_absence_audit"] for row in rows)
        and all(row["schedule_parity"] and row["schedule_contract"] for row in rows)
        and all(row["codebook_parity"] and row["snapshot_restore_parity"] for row in rows)
        and all(row["frozen_protocol"] for row in rows)
        and all(
            row["hippocampal_rows_after_rollout"] == 0
            and row["cutoff_audit"]["temporal_rows_after"] == 0
            and row["cutoff_audit"]["hippocampal_rows_after"] == 0
            for row in rows
        )
    )
    passes = go_count / count >= 0.80 and integrity
    return {
        "circuit_count": count,
        "go_count": go_count,
        "go_fraction": go_count / count,
        "passes_80_percent_gate": go_count / count >= 0.80,
        "route_verdict": "GO" if passes else "STOP",
        "held_out_accuracy_mean": sum(row["held_out_accuracy"] for row in rows) / count,
        "control_advantage_min": min(row["control_advantage"] for row in rows),
        "heldout_absence_all": all(row["heldout_absence_audit"] for row in rows),
        "schedule_parity_all": all(row["schedule_parity"] for row in rows),
        "schedule_contract_all": all(row["schedule_contract"] for row in rows),
        "snapshot_restore_parity_all": all(row["snapshot_restore_parity"] for row in rows),
        "frozen_protocol_all": all(row["frozen_protocol"] for row in rows),
        "codebook_parity_all": all(row["codebook_parity"] for row in rows),
        "zero_store_all": all(
            row["hippocampal_rows_after_rollout"] == 0
            and row["cutoff_audit"]["temporal_rows_after"] == 0
            and row["cutoff_audit"]["hippocampal_rows_after"] == 0
            for row in rows
        ),
        "factor_frequency_ratio": {
            "min": min(row["factor_frequency_sensitivity"]["max_to_min_abs_ratio"] for row in rows),
            "mean": sum(row["factor_frequency_sensitivity"]["max_to_min_abs_ratio"] for row in rows) / count,
            "max": max(row["factor_frequency_sensitivity"]["max_to_min_abs_ratio"] for row in rows),
        },
        "decoder_only_baseline_accuracy_mean": sum(
            row["decoder_only_baseline_accuracy"] for row in rows
        ) / count,
    }


def _verify_freeze(path: Path, source_path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN":
        raise SystemExit("confirmation requires a manifest with status FROZEN")
    if payload.get("confirmation_seed_range") != [99301, 99332]:
        raise SystemExit("confirmation seed range does not match the contract")
    if payload.get("source_sha256") != _sha256(source_path):
        raise SystemExit("source hash differs from the frozen manifest")
    return payload


def _m2_summary(rows: list[dict[str, Any]], *, task: str) -> dict[str, Any]:
    count = len(rows)
    go_count = sum(row["status"] == "GO" for row in rows)
    metric = "clean_accuracy" if task == "binding" else "held_out_accuracy"
    integrity = all(
        row["schedule_parity"]
        and row["identical_phase_zero_update"]
        and row["snapshot_restore_parity"]
        and row["dense_sparse_parity"]
        and row["finite"]
        and row["automatic_stdp_updates"] == 0
        and row["hippocampal_rows_after_rollout"] == 0
        for row in rows
    )
    passes = go_count / count >= 0.80 and integrity
    summary: dict[str, Any] = {
        "circuit_count": count,
        "go_count": go_count,
        "go_fraction": go_count / count,
        "passes_80_percent_gate": go_count / count >= 0.80,
        "route_verdict": "GO" if passes else "STOP",
        f"{metric}_mean": sum(row[metric] for row in rows) / count,
        "control_advantage_min": min(row["control_advantage"] for row in rows),
        "negative_correlation_nonzero_count": sum(row["contrastive_negative_nonzero"] for row in rows),
        "positive_only_same_count": sum(row["positive_only_applied_delta_same"] for row in rows),
        "identical_phase_zero_all": all(row["identical_phase_zero_update"] for row in rows),
        "schedule_parity_all": all(row["schedule_parity"] for row in rows),
        "snapshot_restore_parity_all": all(row["snapshot_restore_parity"] for row in rows),
        "zero_store_all": all(
            row["hippocampal_rows_after_rollout"] == 0
            and row["cutoff_audit"]["temporal_rows_after"] == 0
            and row["cutoff_audit"]["hippocampal_rows_after"] == 0
            for row in rows
        ),
        "negative_correlation_norm_max": max(row["negative_correlation_norm"] for row in rows),
    }
    if task == "binding":
        summary["task_gate_without_controls_count"] = sum(row["task_gate_without_controls"] for row in rows)
    else:
        summary["heldout_absence_all"] = all(row["heldout_absence_audit"] for row in rows)
    return summary


def _m3_predictor_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    go_count = sum(row["status"] == "GO" for row in rows)
    integrity = all(
        row["theta_frozen_during_score"]
        and row["fit_score_row_disjoint"]
        and row["automatic_stdp_updates"] == 0
        and row["weight_unchanged"]
        and row["finite"]
        for row in rows
    )
    return {
        "circuit_count": count,
        "go_count": go_count,
        "go_fraction": go_count / count,
        "passes_80_percent_gate": go_count / count >= 0.80,
        "route_verdict": "GO" if go_count / count >= 0.80 and integrity else "STOP",
        "model_mse_mean": sum(row["model_mse"] for row in rows) / count,
        "persistence_mse_mean": sum(row["persistence_mse"] for row in rows) / count,
        "mse_ratio_mean": sum(row["mse_ratio"] for row in rows) / count,
        "mse_ratio_min": min(row["mse_ratio"] for row in rows),
        "mse_ratio_max": max(row["mse_ratio"] for row in rows),
        "theta_frozen_all": all(row["theta_frozen_during_score"] for row in rows),
        "fit_score_disjoint_all": all(row["fit_score_row_disjoint"] for row in rows),
        "zero_weight_updates_all": all(row["weight_unchanged"] for row in rows),
        "effective_replay_vector_audit_max": max(
            row["effective_replay_vector_residual_max"] for row in rows
        ),
    }


def _m3_memory_summary(rows: list[dict[str, Any]], *, task: str) -> dict[str, Any]:
    count = len(rows)
    go_count = sum(row["status"] == "GO" for row in rows)
    metric = "clean_accuracy" if task == "binding" else "held_out_accuracy"
    controls = tuple(rows[0]["controls"])
    integrity = all(
        row["schedule_parity"]
        and row["predictor_frozen"]
        and row["automatic_stdp_updates"] == 0
        and row["snapshot_restore_parity"]
        and row["hippocampal_rows_after_rollout"] == 0
        for row in rows
    )
    summary: dict[str, Any] = {
        "circuit_count": count,
        "go_count": go_count,
        "go_fraction": go_count / count,
        "passes_80_percent_gate": go_count / count >= 0.80,
        "route_verdict": "GO" if go_count / count >= 0.80 and integrity else "STOP",
        f"{metric}_mean": sum(row[metric] for row in rows) / count,
        "control_advantage_mean": sum(row["control_advantage"] for row in rows) / count,
        "control_advantage_min": min(row["control_advantage"] for row in rows),
        "predictor_gate_pass_count": sum(row["prediction_gate_passed"] for row in rows),
        "schedule_parity_all": all(row["schedule_parity"] for row in rows),
        "zero_store_all": all(
            row["hippocampal_rows_after_rollout"] == 0
            and row["cutoff_audit"]["temporal_rows_after"] == 0
            and row["cutoff_audit"]["hippocampal_rows_after"] == 0
            for row in rows
        ),
        "control_accuracy_means": {
            name: sum(row["controls"][name][metric] for row in rows) / count
            for name in controls
        },
    }
    if task == "binding":
        summary["task_gate_without_controls_count"] = sum(row["task_gate_without_controls"] for row in rows)
    else:
        summary["heldout_absence_all"] = all(row["heldout_absence_audit"] for row in rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route",
        choices=(
            "t1", "m2-binding", "m2-factor", "m2-all",
            "m3-predictor", "m3-binding", "m3-factor", "m3-all",
        ),
        default="t1",
    )
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    source_path = Path(__file__).with_name("runtime_contrastive_predictive_memory.py")
    freeze = None
    if args.confirmation:
        if args.freeze_manifest is None:
            raise SystemExit("confirmation is sealed until --freeze-manifest is supplied")
        freeze = _verify_freeze(args.freeze_manifest, source_path)

    seeds = list(CONFIRMATION_SEEDS if args.confirmation else DEVELOPMENT_SEEDS)
    if args.route == "t1":
        route_rows: list[dict[str, Any]] = [t1_m1_factor_transfer(seed) for seed in seeds]
        summary: dict[str, Any] = _t1_summary(route_rows)
    elif args.route == "m2-binding":
        route_rows = [m2_lagged_contrastive_binding(seed) for seed in seeds]
        summary = _m2_summary(route_rows, task="binding")
    elif args.route == "m2-factor":
        route_rows = [m2_lagged_contrastive_factor_transfer(seed) for seed in seeds]
        summary = _m2_summary(route_rows, task="factor_transfer")
    elif args.route == "m2-all":
        binding_rows = [m2_lagged_contrastive_binding(seed) for seed in seeds]
        factor_rows = [m2_lagged_contrastive_factor_transfer(seed) for seed in seeds]
        route_rows = [
            {"seed": seed, "binding": binding, "factor_transfer": factor}
            for seed, binding, factor in zip(seeds, binding_rows, factor_rows)
        ]
        summary = {
            "binding": _m2_summary(binding_rows, task="binding"),
            "factor_transfer": _m2_summary(factor_rows, task="factor_transfer"),
        }
    elif args.route == "m3-predictor":
        route_rows = [m3_predictor_audit(seed) for seed in seeds]
        summary = _m3_predictor_summary(route_rows)
    elif args.route == "m3-binding":
        route_rows = [m3_replay_residual_binding(seed) for seed in seeds]
        summary = _m3_memory_summary(route_rows, task="binding")
    elif args.route == "m3-factor":
        route_rows = [m3_replay_residual_factor_transfer(seed) for seed in seeds]
        summary = _m3_memory_summary(route_rows, task="factor_transfer")
    else:
        binding_rows = [m3_replay_residual_binding(seed) for seed in seeds]
        factor_rows = [m3_replay_residual_factor_transfer(seed) for seed in seeds]
        route_rows = [
            {"seed": seed, "binding": binding, "factor_transfer": factor}
            for seed, binding, factor in zip(seeds, binding_rows, factor_rows)
        ]
        summary = {
            "binding": _m3_memory_summary(binding_rows, task="binding"),
            "factor_transfer": _m3_memory_summary(factor_rows, task="factor_transfer"),
        }
    result_bytes = json.dumps(route_rows, sort_keys=True).encode("utf-8")
    report = {
        "mode": "confirmation" if args.confirmation else "development",
        "route": args.route,
        "seed_range": [min(seeds), max(seeds)],
        "source_sha256": _sha256(source_path),
        "freeze_manifest": freeze,
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "summary": summary,
        "results": route_rows,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    if not args.quiet:
        print(serialized, end="")


if __name__ == "__main__":
    main()
