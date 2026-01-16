from __future__ import annotations
import numpy as np
from reality_stone import build_causal_laplacian, bellman_geodesic_forward


def get_laplacian(n: int, window: int = 3) -> np.ndarray:
    return build_causal_laplacian(n, window)


def bellman_flow(x: np.ndarray, dt: float = 0.1) -> np.ndarray:
    return bellman_geodesic_forward(x, dt)


def solve_resolvent(L: np.ndarray, r: np.ndarray, rho: float = 1.0, nu: float = 1.0) -> np.ndarray:
    n = L.shape[0]
    A = rho * np.eye(n, dtype=np.float64) + nu * L.astype(np.float64)
    return np.linalg.solve(A, r.astype(np.float64)).astype(np.float32)


def smooth_winrate(L: np.ndarray, p: np.ndarray, rho: float = 0.5, nu: float = 1.0) -> np.ndarray:
    return solve_resolvent(L, p, rho=rho, nu=nu)
