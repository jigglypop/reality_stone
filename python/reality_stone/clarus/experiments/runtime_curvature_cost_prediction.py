"""BA-TR12: test curvature-only route cost against finite tanh distortion."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch

from .runtime_learned_metric_curvature import top_right_singular_plane


CALIBRATION_RADII = tuple(index / 10.0 for index in range(6))
HELDOUT_AMPLITUDE = 1.0
DIRECTIONS = tuple(
    (math.cos(index * math.pi / 8.0), math.sin(index * math.pi / 8.0))
    for index in range(8)
)


def _actuator(value: torch.Tensor | Sequence[Sequence[float]]) -> torch.Tensor:
    packed = torch.as_tensor(value, dtype=torch.float64)
    if packed.ndim != 2 or packed.shape[1] != 2 or packed.shape[0] < 2:
        raise ValueError("actuator must have shape m x 2 with m >= 2")
    if not torch.isfinite(packed).all():
        raise ValueError("actuator must be finite")
    if int(torch.linalg.matrix_rank(packed).item()) != 2:
        raise ValueError("actuator must have column rank two")
    return packed


def response(A: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    actuator = _actuator(A)
    point = torch.as_tensor(u, dtype=torch.float64).view(2)
    return torch.tanh(actuator @ point)


def geometry(A: torch.Tensor, u: torch.Tensor, *, tolerance: float = 1e-12) -> dict[str, Any]:
    actuator = _actuator(A)
    point = torch.as_tensor(u, dtype=torch.float64).view(2)
    z = actuator @ point
    tanh_z = torch.tanh(z)
    first = 1.0 - tanh_z.square()
    J = first[:, None] * actuator
    metric = 0.5 * (J.T @ J + (J.T @ J).T)
    eigenvalues = torch.linalg.eigvalsh(metric)
    if float(eigenvalues[0].item()) <= tolerance:
        raise ValueError("metric is rank deficient")
    second = -2.0 * tanh_z * first
    h00 = second * actuator[:, 0].square()
    h01 = second * actuator[:, 0] * actuator[:, 1]
    h11 = second * actuator[:, 1].square()
    normal = torch.eye(actuator.shape[0], dtype=torch.float64) - J @ torch.linalg.solve(
        metric, J.T,
    )
    ii00 = normal @ h00
    ii01 = normal @ h01
    ii11 = normal @ h11
    curvature = (
        torch.dot(ii00, ii11) - torch.dot(ii01, ii01)
    ) / torch.linalg.det(metric)
    return {
        "metric": metric,
        "curvature": float(curvature.item()),
        "jacobian": J,
    }


def _inverse_sqrt(metric: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(metric)
    if float(eigenvalues[0].item()) <= 1e-12:
        raise ValueError("origin metric is rank deficient")
    return eigenvectors @ torch.diag(eigenvalues.rsqrt()) @ eigenvectors.T


def ray_costs(
    A: torch.Tensor,
    direction: Sequence[float],
    *,
    radii: Sequence[float] = CALIBRATION_RADII,
    heldout_amplitude: float = HELDOUT_AMPLITUDE,
) -> dict[str, float]:
    actuator = _actuator(A)
    unit = torch.as_tensor(direction, dtype=torch.float64).view(2)
    unit = unit / unit.norm()
    grid = tuple(float(value) for value in radii)
    if len(grid) < 2 or grid[0] != 0.0 or any(right <= left for left, right in zip(grid, grid[1:])):
        raise ValueError("radii must be strictly increasing from zero")
    origin_metric = actuator.T @ actuator
    whitener = _inverse_sqrt(origin_metric)
    identity = torch.eye(2, dtype=torch.float64)
    length = 0.0
    curvature_integral = 0.0
    strain_integral = 0.0
    for left, right in zip(grid, grid[1:]):
        midpoint = 0.5 * (left + right)
        ds = right - left
        state = geometry(actuator, midpoint * unit)
        metric = state["metric"]
        speed = math.sqrt(float(unit @ metric @ unit))
        dell = speed * ds
        relative_metric = whitener @ metric @ whitener
        strain = float((relative_metric - identity).norm().item())
        length += dell
        curvature_integral += abs(float(state["curvature"])) * dell
        strain_integral += strain * dell
    curvature_cost = length * curvature_integral
    strain_cost = strain_integral / length
    point = float(heldout_amplitude) * unit
    linear = actuator @ point
    nonlinear = response(actuator, point)
    distortion = float((nonlinear - linear).norm().item()) / float(linear.norm().item())
    return {
        "length": length,
        "curvature_cost": curvature_cost,
        "metric_strain_cost": strain_cost,
        "heldout_distortion": distortion,
    }


def _rotation(dim: int, left: int, right: int, angle: float) -> torch.Tensor:
    value = torch.eye(dim, dtype=torch.float64)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    value[left, left] = cosine
    value[left, right] = -sine
    value[right, left] = sine
    value[right, right] = cosine
    return value


def route_rotations() -> tuple[torch.Tensor, ...]:
    identity = torch.eye(4, dtype=torch.float64)
    return (
        identity,
        _rotation(4, 0, 1, 0.37),
        _rotation(4, 0, 2, 0.61),
        _rotation(4, 1, 3, 0.83),
        _rotation(4, 2, 3, 1.07),
        _rotation(4, 0, 1, 0.43) @ _rotation(4, 2, 3, 0.71),
    )


def flat_nonlinear_counterexample(amplitude: float = HELDOUT_AMPLITUDE) -> dict[str, Any]:
    identity = torch.eye(2, dtype=torch.float64)
    rotation = _rotation(2, 0, 1, math.pi / 4.0)
    direction = (1.0, 0.0)
    first = ray_costs(identity, direction, heldout_amplitude=amplitude)
    second = ray_costs(rotation, direction, heldout_amplitude=amplitude)
    analytic_first = abs(math.tanh(amplitude) - amplitude) / amplitude
    analytic_second = abs(math.sqrt(2.0) * math.tanh(amplitude / math.sqrt(2.0)) - amplitude) / amplitude
    return {
        "same_origin_metric": bool(
            float((identity.T @ identity - rotation.T @ rotation).norm().item()) <= 1e-12
        ),
        "route_1": first,
        "route_2": second,
        "analytic_distortion_1": analytic_first,
        "analytic_distortion_2": analytic_second,
        "both_curvature_cost_zero": bool(
            abs(first["curvature_cost"]) <= 1e-12
            and abs(second["curvature_cost"]) <= 1e-12
        ),
        "distortion_separated": abs(first["heldout_distortion"] - second["heldout_distortion"]) > 1e-3,
        "metric_strain_selects_lower_distortion": bool(
            (first["metric_strain_cost"] < second["metric_strain_cost"])
            == (first["heldout_distortion"] < second["heldout_distortion"])
        ),
    }


def _signed_permutation() -> torch.Tensor:
    value = torch.zeros(4, 4, dtype=torch.float64)
    signs = (1.0, -1.0, 1.0, -1.0)
    for old in range(4):
        value[(old + 1) % 4, old] = signs[old]
    return value


def analyze_learned_matrix(B: torch.Tensor) -> dict[str, Any]:
    packed = torch.as_tensor(B, dtype=torch.float64)
    if packed.shape != (4, 4) or int(torch.linalg.matrix_rank(packed).item()) != 4:
        raise ValueError("learned B must be full-rank 4x4")
    plane = top_right_singular_plane(packed)
    base_actuator = packed @ plane
    rotations = route_rotations()
    origin_metric = base_actuator.T @ base_actuator
    maximum_origin_error = 0.0
    records = []
    curvature_regret = []
    strain_regret = []
    curvature_hits = 0
    strain_hits = 0
    for direction_index, direction in enumerate(DIRECTIONS):
        routes = []
        for route_index, rotation in enumerate(rotations):
            actuator = rotation @ base_actuator
            maximum_origin_error = max(
                maximum_origin_error,
                float((actuator.T @ actuator - origin_metric).norm().item()),
            )
            routes.append({
                "route": route_index,
                **ray_costs(actuator, direction),
            })
        best_actual = min(routes, key=lambda item: (item["heldout_distortion"], item["route"]))
        best_curvature = min(routes, key=lambda item: (item["curvature_cost"], item["route"]))
        best_strain = min(routes, key=lambda item: (item["metric_strain_cost"], item["route"]))
        curvature_hits += int(best_curvature["route"] == best_actual["route"])
        strain_hits += int(best_strain["route"] == best_actual["route"])
        curvature_regret.append(best_curvature["heldout_distortion"] - best_actual["heldout_distortion"])
        strain_regret.append(best_strain["heldout_distortion"] - best_actual["heldout_distortion"])
        records.append({
            "direction_index": direction_index,
            "direction": [float(value) for value in direction],
            "routes": routes,
            "best_actual_route": best_actual["route"],
            "best_curvature_route": best_curvature["route"],
            "best_strain_route": best_strain["route"],
        })

    equality = _signed_permutation()
    equality_residual = 0.0
    for direction in DIRECTIONS:
        base = ray_costs(base_actuator, direction)
        permuted = ray_costs(equality @ base_actuator, direction)
        equality_residual = max(
            equality_residual,
            *(abs(base[key] - permuted[key]) for key in (
                "curvature_cost", "metric_strain_cost", "heldout_distortion"
            )),
        )
    mean_curvature_regret = sum(curvature_regret) / len(curvature_regret)
    mean_strain_regret = sum(strain_regret) / len(strain_regret)
    return {
        "maximum_origin_metric_error": maximum_origin_error,
        "signed_permutation_equality_residual": equality_residual,
        "curvature_exact_hit_rate": curvature_hits / len(DIRECTIONS),
        "strain_exact_hit_rate": strain_hits / len(DIRECTIONS),
        "mean_curvature_regret": mean_curvature_regret,
        "mean_strain_regret": mean_strain_regret,
        "strain_regret_no_worse": mean_strain_regret <= mean_curvature_regret + 1e-15,
        "directions": records,
    }


def analyze_development_artifact(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [
        {
            "seed": int(row["seed"]),
            **analyze_learned_matrix(torch.tensor(row["learned"]["candidate_weights"])),
        }
        for row in payload["rows"]
    ]
    flat = flat_nonlinear_counterexample()
    mean_curvature_regret = sum(row["mean_curvature_regret"] for row in rows) / len(rows)
    mean_strain_regret = sum(row["mean_strain_regret"] for row in rows) / len(rows)
    apparatus_pass = (
        flat["same_origin_metric"]
        and flat["both_curvature_cost_zero"]
        and flat["distortion_separated"]
        and flat["metric_strain_selects_lower_distortion"]
        and all(row["maximum_origin_metric_error"] <= 1e-12 for row in rows)
        and all(row["signed_permutation_equality_residual"] <= 1e-12 for row in rows)
    )
    if not apparatus_pass:
        status = "CURVATURE_COST_DIAGNOSTIC_FAIL"
    elif mean_strain_regret <= mean_curvature_regret:
        status = "K_INSUFFICIENT_METRIC_STRAIN_REQUIRED"
    else:
        status = "CURVATURE_ASSOCIATION_NOT_SUFFICIENT"
    return {
        "status": status,
        "seed_count": len(rows),
        "flat_counterexample": flat,
        "mean_curvature_regret": mean_curvature_regret,
        "mean_metric_strain_regret": mean_strain_regret,
        "mean_curvature_hit_rate": sum(row["curvature_exact_hit_rate"] for row in rows) / len(rows),
        "mean_metric_strain_hit_rate": sum(row["strain_exact_hit_rate"] for row in rows) / len(rows),
        "maximum_origin_metric_error": max(row["maximum_origin_metric_error"] for row in rows),
        "maximum_equality_residual": max(row["signed_permutation_equality_residual"] for row in rows),
        "endpoint_opened": False,
        "rows": rows,
    }
