from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

import numpy as np
import torch


SOURCE = Path(__file__).resolve().parents[1] / "python"
if SOURCE.exists():
    source_s = str(SOURCE)
    if source_s not in sys.path:
        sys.path.insert(0, source_s)

import reality_stone as rs  # noqa: E402
from reality_stone.clarus import reality  # noqa: E402
from reality_stone.clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode  # noqa: E402


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Sparse .*",
        category=UserWarning,
    )

    torch.manual_seed(7)
    dim = 8

    weight = torch.randn(dim, dim) * 0.05
    weight = 0.5 * (weight + weight.t())
    weight.fill_diagonal_(0.0)

    runtime = BrainRuntime(
        weight,
        config=BrainRuntimeConfig(dim=dim, active_ratio=0.25, noise_sigma=0.0),
        backend="torch",
        device="cpu",
    )
    runtime.set_goal(torch.linspace(-0.2, 0.2, dim))
    step = runtime.step(
        external_input=torch.linspace(0.0, 0.5, dim),
        force_mode=RuntimeMode.WAKE,
    )

    attn = rs.MetricAttention(hidden_size=dim, rank=2)
    q = torch.randn(1, 1, 4, dim)
    k = torch.randn(1, 1, 4, dim)
    v = torch.randn(1, 1, 4, dim)
    attended = attn(q, k, v, causal=True)

    layer = reality.unified_riemannian_layer(
        metric_type="euclidean",
        curvature=0.0,
        input_dim=2,
        enable_bellman=True,
    )
    geodesic, energy = layer.forward(
        np.array([[0.0, 1.0]], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    summary = {
        "reality_stone": {
            "version": rs.__version__,
            "rust": bool(getattr(rs, "_has_rust_ext", False)),
            "cuda": bool(getattr(rs, "_has_cuda", False)),
        },
        "clarus_runtime": {
            "mode": step.mode.name,
            "active_modules": int(step.active_modules),
            "energy": float(step.energy),
        },
        "metric_attention": {
            "shape": list(attended.shape),
            "finite": bool(torch.isfinite(attended).all().item()),
        },
        "geodesic": {
            "point": geodesic.reshape(-1).round(6).tolist(),
            "bellman_residual": energy["bellman_residual"].reshape(-1).round(6).tolist(),
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
