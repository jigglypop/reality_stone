"""BA-TR11: separate learned weights, pullback metric, and intrinsic curvature."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch


DEFAULT_POINTS = (
    (0.0, 0.0),
    (0.25, -0.25),
    (0.50, 0.25),
    (-0.50, 0.25),
    (0.25, 0.50),
)


def _matrix(value: torch.Tensor | Sequence[Sequence[float]]) -> torch.Tensor:
    packed = torch.as_tensor(value, dtype=torch.float64)
    if packed.shape != (4, 4) or not torch.isfinite(packed).all():
        raise ValueError("B must be a finite 4x4 matrix")
    return packed


def top_right_singular_plane(B: torch.Tensor) -> torch.Tensor:
    packed = _matrix(B)
    _u, _s, vh = torch.linalg.svd(packed, full_matrices=False)
    return vh[:2].T.contiguous()


def nonlinear_response(
    B: torch.Tensor,
    plane: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    packed = _matrix(B)
    chart = torch.as_tensor(plane, dtype=torch.float64)
    point = torch.as_tensor(u, dtype=torch.float64).view(2)
    if chart.shape != (4, 2) or not torch.isfinite(chart).all():
        raise ValueError("plane must be finite with shape 4x2")
    return torch.tanh(packed @ chart @ point)


def nonlinear_geometry(
    B: torch.Tensor,
    plane: torch.Tensor,
    u: torch.Tensor,
    *,
    rank_tolerance: float = 1e-12,
    condition_limit: float = 1e10,
) -> dict[str, Any]:
    packed = _matrix(B)
    chart = torch.as_tensor(plane, dtype=torch.float64)
    point = torch.as_tensor(u, dtype=torch.float64).view(2)
    if chart.shape != (4, 2) or not torch.isfinite(chart).all():
        raise ValueError("plane must be finite with shape 4x2")
    A = packed @ chart
    z = A @ point
    tanh_z = torch.tanh(z)
    first = 1.0 - tanh_z.square()
    J = first[:, None] * A
    metric = 0.5 * (J.T @ J + (J.T @ J).T)
    eigenvalues = torch.linalg.eigvalsh(metric)
    minimum = float(eigenvalues[0].item())
    maximum = float(eigenvalues[-1].item())
    condition = math.inf if minimum <= 0.0 else maximum / minimum
    if minimum <= rank_tolerance or condition > condition_limit:
        return {
            "status": "CURVATURE_UNDEFINED_DEGENERATE",
            "point": [float(value) for value in point.tolist()],
            "metric": [[float(value) for value in row] for row in metric.tolist()],
            "metric_eigenvalues": [float(value) for value in eigenvalues.tolist()],
            "condition": condition,
            "curvature": None,
        }
    second_factor = -2.0 * tanh_z * first
    h00 = second_factor * A[:, 0] * A[:, 0]
    h01 = second_factor * A[:, 0] * A[:, 1]
    h11 = second_factor * A[:, 1] * A[:, 1]
    normal_projection = torch.eye(4, dtype=torch.float64) - J @ torch.linalg.solve(
        metric, J.T,
    )
    ii00 = normal_projection @ h00
    ii01 = normal_projection @ h01
    ii11 = normal_projection @ h11
    determinant = torch.linalg.det(metric)
    curvature = (
        torch.dot(ii00, ii11) - torch.dot(ii01, ii01)
    ) / determinant
    return {
        "status": "CURVATURE_DEFINED",
        "point": [float(value) for value in point.tolist()],
        "metric": [[float(value) for value in row] for row in metric.tolist()],
        "metric_eigenvalues": [float(value) for value in eigenvalues.tolist()],
        "condition": condition,
        "determinant": float(determinant.item()),
        "curvature": float(curvature.item()),
        "jacobian": [[float(value) for value in row] for row in J.tolist()],
    }


def finite_difference_jacobian(
    B: torch.Tensor,
    plane: torch.Tensor,
    u: torch.Tensor,
    *,
    step: float = 2.0 ** -17,
) -> torch.Tensor:
    point = torch.as_tensor(u, dtype=torch.float64).view(2)
    columns = []
    for axis in range(2):
        direction = torch.zeros(2, dtype=torch.float64)
        direction[axis] = step
        columns.append(
            (nonlinear_response(B, plane, point + direction)
             - nonlinear_response(B, plane, point - direction)) / (2.0 * step)
        )
    return torch.stack(columns, dim=1)


def linear_flatness_certificate(B: torch.Tensor, *, tolerance: float = 1e-10) -> dict[str, Any]:
    packed = _matrix(B)
    singular_values = torch.linalg.svdvals(packed)
    cutoff = tolerance * max(1.0, float(singular_values[0].item()))
    rank = int(torch.count_nonzero(singular_values > cutoff).item())
    metric = packed.T @ packed
    return {
        "rank": rank,
        "singular_values": [float(value) for value in singular_values.tolist()],
        "determinant_B": float(torch.linalg.det(packed).item()),
        "metric": [[float(value) for value in row] for row in metric.tolist()],
        "curvature_status": (
            "LINEAR_PULLBACK_FLAT" if rank == 4 else "CURVATURE_UNDEFINED_DEGENERATE"
        ),
        "intrinsic_curvature": 0.0 if rank == 4 else None,
    }


def _relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).norm().item()) / max(1.0, float(right.norm().item()))


def _row_permutation() -> torch.Tensor:
    permutation = torch.zeros(4, 4, dtype=torch.float64)
    for old in range(4):
        permutation[(old + 1) % 4, old] = 1.0
    return permutation


def _source_permutation() -> torch.Tensor:
    permutation = torch.zeros(4, 4, dtype=torch.float64)
    for old in range(4):
        permutation[(old + 2) % 4, old] = 1.0
    return permutation


def _hidden_rotation() -> torch.Tensor:
    angle = 0.37
    rotation = torch.eye(4, dtype=torch.float64)
    rotation[0, 0] = math.cos(angle)
    rotation[0, 1] = -math.sin(angle)
    rotation[1, 0] = math.sin(angle)
    rotation[1, 1] = math.cos(angle)
    return rotation


def analyze_weight_code(B: torch.Tensor) -> dict[str, Any]:
    packed = _matrix(B)
    plane = top_right_singular_plane(packed)
    linear = linear_flatness_certificate(packed)
    nonlinear = [
        nonlinear_geometry(packed, plane, torch.tensor(point, dtype=torch.float64))
        for point in DEFAULT_POINTS
    ]
    if any(item["status"] != "CURVATURE_DEFINED" for item in nonlinear):
        raise ValueError("learned nonlinear probe became rank deficient")

    fd_errors = []
    for point, geometry in zip(DEFAULT_POINTS, nonlinear):
        analytic = torch.tensor(geometry["jacobian"], dtype=torch.float64)
        finite = finite_difference_jacobian(
            packed, plane, torch.tensor(point, dtype=torch.float64),
        )
        fd_errors.append(_relative_error(analytic, finite))

    row_map = _row_permutation()
    row_permuted = row_map @ packed
    row_geometries = [
        nonlinear_geometry(row_permuted, plane, torch.tensor(point, dtype=torch.float64))
        for point in DEFAULT_POINTS
    ]
    row_metric_error = max(
        _relative_error(
            torch.tensor(left["metric"]), torch.tensor(right["metric"]),
        )
        for left, right in zip(row_geometries, nonlinear)
    )
    row_curvature_error = max(
        abs(float(left["curvature"]) - float(right["curvature"]))
        for left, right in zip(row_geometries, nonlinear)
    )
    original_winners = torch.argmax(packed, dim=0)
    permuted_winners = torch.argmax(row_permuted, dim=0)

    source_map = _source_permutation()
    source_permuted = packed @ source_map
    source_plane = source_map.T @ plane
    source_geometries = [
        nonlinear_geometry(source_permuted, source_plane, torch.tensor(point, dtype=torch.float64))
        for point in DEFAULT_POINTS
    ]
    source_metric_error = max(
        _relative_error(
            torch.tensor(left["metric"]), torch.tensor(right["metric"]),
        )
        for left, right in zip(source_geometries, nonlinear)
    )
    source_curvature_error = max(
        abs(float(left["curvature"]) - float(right["curvature"]))
        for left, right in zip(source_geometries, nonlinear)
    )

    rotation = _hidden_rotation()
    rotated = rotation @ packed
    origin_metric_error = _relative_error(rotated.T @ rotated, packed.T @ packed)
    rotated_geometries = [
        nonlinear_geometry(rotated, plane, torch.tensor(point, dtype=torch.float64))
        for point in DEFAULT_POINTS
    ]
    rotation_curvature_difference = max(
        abs(float(left["curvature"]) - float(right["curvature"]))
        for left, right in zip(rotated_geometries[1:], nonlinear[1:])
    )

    uniform = torch.ones(4, 4, dtype=torch.float64)
    uniform_linear = linear_flatness_certificate(uniform)
    nonlinear_values = [float(item["curvature"]) for item in nonlinear]
    gates = {
        "learned_weight_full_rank": linear["rank"] == 4,
        "linear_code_is_flat": linear["curvature_status"] == "LINEAR_PULLBACK_FLAT",
        "uniform_metric_is_degenerate_not_zero_curvature": (
            uniform_linear["curvature_status"] == "CURVATURE_UNDEFINED_DEGENERATE"
        ),
        "analytic_jacobian_matches_fd": max(fd_errors) <= 5e-9,
        "nonlinear_curvature_is_state_dependent": max(abs(value) for value in nonlinear_values) > 1e-10,
        "hidden_relabel_changes_binding": bool(not torch.equal(original_winners, permuted_winners)),
        "hidden_relabel_preserves_metric": row_metric_error <= 1e-12,
        "hidden_relabel_preserves_curvature": row_curvature_error <= 1e-12,
        "source_rechart_preserves_metric": source_metric_error <= 1e-12,
        "source_rechart_preserves_curvature": source_curvature_error <= 1e-12,
        "same_origin_metric_can_have_different_nonlinear_curvature": bool(
            origin_metric_error <= 1e-12 and rotation_curvature_difference > 1e-10
        ),
    }
    return {
        "status": "CURVATURE_IS_DERIVED_NOT_MEMORY" if all(gates.values()) else "GEOMETRY_PROBE_FAIL",
        "gates": gates,
        "linear": linear,
        "uniform_linear": uniform_linear,
        "probe_plane": [[float(value) for value in row] for row in plane.tolist()],
        "nonlinear": nonlinear,
        "max_jacobian_fd_error": max(fd_errors),
        "max_abs_nonlinear_curvature": max(abs(value) for value in nonlinear_values),
        "row_permutation": {
            "original_winners": [int(value) for value in original_winners.tolist()],
            "permuted_winners": [int(value) for value in permuted_winners.tolist()],
            "metric_error": row_metric_error,
            "curvature_error": row_curvature_error,
        },
        "source_rechart": {
            "metric_error": source_metric_error,
            "curvature_error": source_curvature_error,
        },
        "hidden_rotation": {
            "origin_metric_error": origin_metric_error,
            "max_curvature_difference": rotation_curvature_difference,
        },
        "output_read_count": 0,
        "decoder_read_count": 0,
        "target_read_count": 0,
        "reward_read_count": 0,
    }


def analyze_development_artifact(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("development artifact must contain nonempty rows")
    analyzed = []
    for row in rows:
        weights = row["learned"]["candidate_weights"]
        result = analyze_weight_code(torch.tensor(weights, dtype=torch.float64))
        analyzed.append({"seed": int(row["seed"]), **result})
    return {
        "status": (
            "CURVATURE_MEMORY_IDENTITY_REJECTED"
            if all(row["status"] == "CURVATURE_IS_DERIVED_NOT_MEMORY" for row in analyzed)
            else "GEOMETRY_PROBE_FAIL"
        ),
        "seed_count": len(analyzed),
        "pass_count": sum(row["status"] == "CURVATURE_IS_DERIVED_NOT_MEMORY" for row in analyzed),
        "max_jacobian_fd_error": max(row["max_jacobian_fd_error"] for row in analyzed),
        "minimum_abs_nonzero_curvature_max": min(
            row["max_abs_nonlinear_curvature"] for row in analyzed
        ),
        "maximum_row_permutation_curvature_error": max(
            row["row_permutation"]["curvature_error"] for row in analyzed
        ),
        "minimum_hidden_rotation_curvature_difference": min(
            row["hidden_rotation"]["max_curvature_difference"] for row in analyzed
        ),
        "endpoint_opened": False,
        "rows": analyzed,
    }

