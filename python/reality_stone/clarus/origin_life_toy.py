"""Paired-ablation construction check for the minimum-life equation.

This module answers a deliberately narrow question: can one construct an open
two-template dynamical system in which autocatalysis, retention, and copying
are jointly sufficient for persistent heritable distinction?  Parameters are
paired across ablations so that a removed term is the only within-draw change.
The result is a model construction check, not an empirical necessity proof.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LifeCondition:
    autocatalysis: bool
    boundary: bool
    copying: bool


CONDITIONS = {
    "full": LifeCondition(True, True, True),
    "no_autocatalysis": LifeCondition(False, True, True),
    "no_boundary": LifeCondition(True, False, True),
    "no_copying": LifeCondition(True, True, False),
}


def draw_parameters(rng: np.random.Generator) -> dict[str, float]:
    return {
        "auto_rate": float(rng.uniform(0.35, 0.65)),
        "copy_rate": float(rng.uniform(0.16, 0.32)),
        "mutation_rate": float(rng.uniform(0.02, 0.045)),
        "feed_rate": float(rng.uniform(0.001, 0.003)),
        "leak_with_boundary": float(rng.uniform(0.015, 0.04)),
        "leak_without_boundary": float(rng.uniform(0.55, 0.90)),
        "capacity": float(rng.uniform(0.8, 1.2)),
    }


def simulate_lineage(
    parameters: Mapping[str, float],
    condition: LifeCondition,
    initial_a_fraction: float,
    *,
    cycles: int = 6,
    steps_per_cycle: int = 100,
    dt: float = 0.05,
    dilution: float = 0.55,
) -> np.ndarray:
    state = np.asarray([initial_a_fraction, 1.0 - initial_a_fraction]) * 0.25
    for _ in range(cycles):
        for _ in range(steps_per_cycle):
            mass = float(state.sum())
            frequency = state / max(mass, 1e-12)
            autocatalysis = (
                parameters["auto_rate"]
                * state
                * (1.0 - mass / parameters["capacity"])
                if condition.autocatalysis
                else np.zeros_like(state)
            )
            copying = (
                parameters["copy_rate"] * mass * (frequency - 0.5)
                if condition.copying
                else np.zeros_like(state)
            )
            mutation = parameters["mutation_rate"] * mass * (0.5 - frequency)
            leak_rate = (
                parameters["leak_with_boundary"]
                if condition.boundary
                else parameters["leak_without_boundary"]
            )
            feed = parameters["feed_rate"] * np.asarray([0.5, 0.5])
            derivative = autocatalysis + copying + mutation + feed - leak_rate * state
            state = np.clip(state + dt * derivative, 0.0, None)
        state *= dilution
    return state


def evaluate_condition(
    parameters: Mapping[str, float],
    condition: LifeCondition,
    *,
    mass_threshold: float = 0.25,
    heredity_threshold: float = 0.5,
) -> dict[str, float | bool]:
    high_a = simulate_lineage(parameters, condition, 0.62)
    low_a = simulate_lineage(parameters, condition, 0.38)
    masses = np.asarray([high_a.sum(), low_a.sum()])
    high_frequency = float(high_a[0] / max(high_a.sum(), 1e-12))
    low_frequency = float(low_a[0] / max(low_a.sum(), 1e-12))
    heredity = min(1.0, abs(high_frequency - low_frequency) / 0.24)
    viable = bool(masses.min() >= mass_threshold)
    heritable = bool(heredity >= heredity_threshold)
    return {
        "min_mass": float(masses.min()),
        "mean_mass": float(masses.mean()),
        "heredity": heredity,
        "viable": viable,
        "heritable": heritable,
        "passed": viable and heritable,
    }


def _summarize(rows: Sequence[Mapping[str, float | bool]]) -> dict[str, float | int]:
    return {
        "draws": len(rows),
        "pass_rate": float(np.mean([row["passed"] for row in rows])),
        "viable_rate": float(np.mean([row["viable"] for row in rows])),
        "heritable_rate": float(np.mean([row["heritable"] for row in rows])),
        "mean_min_mass": float(np.mean([row["min_mass"] for row in rows])),
        "mean_heredity": float(np.mean([row["heredity"] for row in rows])),
    }


def _operational_counterexamples() -> dict[str, object]:
    """Show that the toy metrics do not establish universal term necessity."""

    baseline = {
        "auto_rate": 0.5,
        "copy_rate": 0.24,
        "mutation_rate": 0.03,
        "feed_rate": 0.002,
        "leak_with_boundary": 0.025,
        "leak_without_boundary": 0.70,
        "capacity": 1.0,
    }
    cases = {
        "no_autocatalysis_with_strong_external_feed": (
            {**baseline, "feed_rate": 0.05},
            LifeCondition(False, True, True),
        ),
        "no_boundary_in_retentive_environment": (
            {**baseline, "leak_without_boundary": 0.025},
            LifeCondition(True, False, True),
        ),
        "no_explicit_copy_term_without_mixing": (
            {**baseline, "mutation_rate": 0.0, "feed_rate": 0.0},
            LifeCondition(True, True, False),
        ),
    }
    results = {
        name: evaluate_condition(parameters, condition)
        for name, (parameters, condition) in cases.items()
    }
    return {
        "all_three_ablations_can_pass_after_assumption_changes": all(
            bool(result["passed"]) for result in results.values()
        ),
        "cases": results,
        "interpretation": (
            "the original ablation result is conditional on its leak, feed, and "
            "mixing ranges; moreover proportional autocatalysis already preserves "
            "template composition, so the explicit copy term is not identifiable"
        ),
    }


def run_toy_gate(
    *,
    draws: int = 200,
    seed: int = 7,
    full_min_pass_rate: float = 0.8,
    ablation_max_pass_rate: float = 0.35,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    rows = {name: [] for name in CONDITIONS}
    for _ in range(draws):
        parameters = draw_parameters(rng)
        for name, condition in CONDITIONS.items():
            rows[name].append(evaluate_condition(parameters, condition))
    summaries = {name: _summarize(values) for name, values in rows.items()}
    full_pass = summaries["full"]["pass_rate"] >= full_min_pass_rate
    ablations_fail = all(
        summaries[name]["pass_rate"] <= ablation_max_pass_rate
        for name in ("no_autocatalysis", "no_boundary", "no_copying")
    )
    counterexamples = _operational_counterexamples()
    return {
        "artifact_type": "clarus_minimum_life_paired_ablation_toy",
        "artifact_version": 1,
        "passed": bool(full_pass and ablations_fail),
        "paired_parameters_across_conditions": True,
        "seed": seed,
        "criteria": {
            "draws": draws,
            "mass_threshold": 0.25,
            "heredity_threshold": 0.5,
            "full_min_pass_rate": full_min_pass_rate,
            "ablation_max_pass_rate": ablation_max_pass_rate,
        },
        "summaries": summaries,
        "operational_necessity_counterexamples": counterexamples,
        "universal_necessity_proven": False,
        "claim_supported": (
            "the three-term equation admits a parameter family with persistent "
            "mass and template distinction, while paired single-term ablations fail"
        ),
        "claim_not_supported": (
            "the parameter ranges encode strong leak and weak feed; therefore this "
            "construction does not prove empirical or universal necessity, and each "
            "operational ablation can pass under explicit alternative assumptions"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(argv)
    artifact = run_toy_gate(draws=args.draws, seed=args.seed)
    payload = json.dumps(artifact, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return int(args.require_pass and not artifact["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
