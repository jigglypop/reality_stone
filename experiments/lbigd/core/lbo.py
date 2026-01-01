from __future__ import annotations

import numpy as np


def laplacian_mul(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Graph Laplacian multiply: (D - W) x

    - w: (n, n) non-negative weights (prefer symmetric)
    - x: (n,) or (n, k)
    """
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("w must be square (n, n)")

    if x.ndim == 1:
        if x.shape[0] != w.shape[0]:
            raise ValueError("x must have shape (n,)")
        deg = w.sum(axis=1)
        return deg * x - (w @ x)

    if x.ndim == 2:
        if x.shape[0] != w.shape[0]:
            raise ValueError("x must have shape (n, k)")
        deg = w.sum(axis=1)
        return deg[:, None] * x - (w @ x)

    raise ValueError("x must be 1D or 2D")


def dirichlet_energy(w: np.ndarray, u: np.ndarray) -> float:
    """
    Dirichlet energy: u^T (D - W) u  (>= 0 if w is symmetric and non-negative)
    """
    lu = laplacian_mul(w, u)
    return float(np.dot(u, lu))


