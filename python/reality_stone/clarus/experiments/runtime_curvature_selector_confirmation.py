"""BA-TR13: fresh-geometry confirmation of the frozen curvature selector."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch

from .runtime_curvature_cost_prediction import ray_costs
from .runtime_learned_metric_curvature import top_right_singular_plane


CONFIRMATION_AMPLITUDE = 1.25
CONFIRMATION_DIRECTIONS = tuple(
    (
        math.cos((index + 0.5) * math.pi / 8.0),
        math.sin((index + 0.5) * math.pi / 8.0),
    )
    for index in range(8)
)


def _rotation(left: int, right: int, angle: float) -> torch.Tensor:
    value = torch.eye(4, dtype=torch.float64)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    value[left, left] = cosine
    value[left, right] = -sine
    value[right, left] = sine
    value[right, right] = cosine
    return value


def confirmation_rotations() -> tuple[torch.Tensor, ...]:
    return (
        _rotation(0, 3, 0.29),
        _rotation(1, 2, 0.53),
        _rotation(0, 2, 0.77),
        _rotation(1, 3, 1.01),
        _rotation(0, 1, 0.91) @ _rotation(2, 3, 0.33),
        _rotation(0, 3, 0.47) @ _rotation(1, 2, 0.69),
    )


def _signed_permutation() -> torch.Tensor:
    value = torch.zeros(4, 4, dtype=torch.float64)
    for old, sign in enumerate((1.0, -1.0, 1.0, -1.0)):
        value[(old + 1) % 4, old] = sign
    return value


def confirm_matrix(B: torch.Tensor) -> dict[str, Any]:
    packed = torch.as_tensor(B, dtype=torch.float64)
    if packed.shape != (4, 4) or int(torch.linalg.matrix_rank(packed).item()) != 4:
        raise ValueError("B must be full-rank 4x4")
    plane = top_right_singular_plane(packed)
    base = packed @ plane
    rotations = confirmation_rotations()
    origin = base.T @ base
    maximum_origin_error = 0.0
    curvature_hits = 0
    strain_hits = 0
    static_hits = 0
    curvature_regrets = []
    strain_regrets = []
    static_regrets = []
    rows = []
    for direction_index, direction in enumerate(CONFIRMATION_DIRECTIONS):
        routes = []
        for route_index, rotation in enumerate(rotations):
            actuator = rotation @ base
            maximum_origin_error = max(
                maximum_origin_error,
                float((actuator.T @ actuator - origin).norm().item()),
            )
            routes.append({
                "route": route_index,
                **ray_costs(
                    actuator,
                    direction,
                    heldout_amplitude=CONFIRMATION_AMPLITUDE,
                ),
            })
        actual = min(routes, key=lambda value: (value["heldout_distortion"], value["route"]))
        curvature = min(routes, key=lambda value: (value["curvature_cost"], value["route"]))
        strain = min(routes, key=lambda value: (value["metric_strain_cost"], value["route"]))
        static = routes[0]
        curvature_hits += int(curvature["route"] == actual["route"])
        strain_hits += int(strain["route"] == actual["route"])
        static_hits += int(static["route"] == actual["route"])
        curvature_regrets.append(curvature["heldout_distortion"] - actual["heldout_distortion"])
        strain_regrets.append(strain["heldout_distortion"] - actual["heldout_distortion"])
        static_regrets.append(static["heldout_distortion"] - actual["heldout_distortion"])
        rows.append({
            "direction_index": direction_index,
            "direction": [float(value) for value in direction],
            "routes": routes,
            "actual": actual["route"],
            "curvature": curvature["route"],
            "strain": strain["route"],
            "static": static["route"],
        })
    equality = _signed_permutation()
    equality_residual = 0.0
    for direction in CONFIRMATION_DIRECTIONS:
        left = ray_costs(base, direction, heldout_amplitude=CONFIRMATION_AMPLITUDE)
        right = ray_costs(equality @ base, direction, heldout_amplitude=CONFIRMATION_AMPLITUDE)
        equality_residual = max(
            equality_residual,
            *(abs(left[key] - right[key]) for key in (
                "curvature_cost", "metric_strain_cost", "heldout_distortion"
            )),
        )
    count = len(CONFIRMATION_DIRECTIONS)
    return {
        "curvature_hit_rate": curvature_hits / count,
        "strain_hit_rate": strain_hits / count,
        "static_hit_rate": static_hits / count,
        "curvature_regret": sum(curvature_regrets) / count,
        "strain_regret": sum(strain_regrets) / count,
        "static_regret": sum(static_regrets) / count,
        "maximum_origin_metric_error": maximum_origin_error,
        "signed_permutation_equality_residual": equality_residual,
        "directions": rows,
    }


def confirm_development_artifact(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [
        {
            "seed": int(row["seed"]),
            **confirm_matrix(torch.tensor(row["learned"]["candidate_weights"])),
        }
        for row in payload["rows"]
    ]
    mean = lambda key: sum(float(row[key]) for row in rows) / len(rows)
    curvature_hit = mean("curvature_hit_rate")
    strain_hit = mean("strain_hit_rate")
    static_hit = mean("static_hit_rate")
    curvature_regret = mean("curvature_regret")
    strain_regret = mean("strain_regret")
    static_regret = mean("static_regret")
    gates = {
        "curvature_hit_at_least_point70": curvature_hit >= 0.70,
        "curvature_regret_at_most_point01": curvature_regret <= 0.01,
        "curvature_beats_strain_hit": curvature_hit > strain_hit,
        "curvature_beats_strain_regret": curvature_regret < strain_regret,
        "curvature_beats_static_hit": curvature_hit > static_hit,
        "curvature_beats_static_regret": curvature_regret < static_regret,
        "equal_origin_metric": max(row["maximum_origin_metric_error"] for row in rows) <= 1e-12,
        "signed_permutation_equality": max(
            row["signed_permutation_equality_residual"] for row in rows
        ) <= 1e-12,
    }
    return {
        "status": "FRESH_GEOMETRY_CURVATURE_SELECTOR_PASS" if all(gates.values()) else "CURVATURE_SELECTOR_STOP",
        "seed_count": len(rows),
        "gates": gates,
        "heldout_amplitude": CONFIRMATION_AMPLITUDE,
        "mean_curvature_hit_rate": curvature_hit,
        "mean_strain_hit_rate": strain_hit,
        "mean_static_hit_rate": static_hit,
        "mean_curvature_regret": curvature_regret,
        "mean_strain_regret": strain_regret,
        "mean_static_regret": static_regret,
        "maximum_origin_metric_error": max(row["maximum_origin_metric_error"] for row in rows),
        "maximum_equality_residual": max(
            row["signed_permutation_equality_residual"] for row in rows
        ),
        "endpoint_opened": False,
        "rows": rows,
    }

