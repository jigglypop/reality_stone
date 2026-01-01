from __future__ import annotations

import numpy as np


def ring(n: int, weight: float = 1.0) -> np.ndarray:
    """
    Simple ring graph with degree 2.
    """
    if n <= 2:
        raise ValueError("n must be > 2")
    if weight < 0:
        raise ValueError("weight must be non-negative")

    w = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        w[i, (i - 1) % n] = weight
        w[i, (i + 1) % n] = weight
    return _symmetrize(_zero_diag(w))


def affinity(u: np.ndarray, tau: float) -> np.ndarray:
    """
    Dense affinity from current activity u:
      a_ij = exp(-(u_i - u_j)^2 / (2*tau^2))
    """
    if u.ndim != 1:
        raise ValueError("u must be 1D")
    if tau <= 0:
        raise ValueError("tau must be > 0")

    du = u[:, None] - u[None, :]
    a = np.exp(-(du * du) / (2.0 * float(tau * tau))).astype(np.float32)
    return _symmetrize(_zero_diag(a))


def update(
    w: np.ndarray,
    u: np.ndarray,
    *,
    lr: float,
    tau: float,
    topk: int,
    decay: float,
    w_max: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Hebbian-like metric update (activity-similarity graph learning).

    - builds an affinity graph from u
    - sparsifies with top-k
    - (optional) applies a structural mask to restrict possible edges
    - blends into current w with lr
    - applies decay + clamp
    """
    if lr <= 0 or lr > 1:
        raise ValueError("lr must be in (0, 1]")
    if topk <= 0:
        raise ValueError("topk must be > 0")
    if decay < 0 or decay >= 1:
        raise ValueError("decay must be in [0, 1)")
    if w_max <= 0:
        raise ValueError("w_max must be > 0")

    a = affinity(u.astype(np.float32), tau=float(tau))
    if mask is not None:
        m = np.asarray(mask, dtype=np.float32)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("mask must be square (n, n)")
        if m.shape != w.shape:
            raise ValueError("mask shape must match w shape")
        if float(m.min()) < 0.0:
            raise ValueError("mask must be non-negative")
        m = _symmetrize(_zero_diag(m))
        a = (a * m).astype(np.float32, copy=False)
    a = topk_mask(a, k=topk)
    a = _symmetrize(_zero_diag(a))

    w = w.astype(np.float32, copy=False)
    w = (1.0 - float(decay)) * w
    w = (1.0 - float(lr)) * w + float(lr) * a
    w = np.clip(w, 0.0, float(w_max)).astype(np.float32)
    return _symmetrize(_zero_diag(w))


def topk_mask(w: np.ndarray, k: int) -> np.ndarray:
    """
    Keep only top-k weights per row (excluding diagonal).
    """
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("w must be square (n, n)")
    n = w.shape[0]
    if k >= n:
        return w

    out = np.zeros_like(w, dtype=np.float32)
    for i in range(n):
        row = w[i].copy()
        row[i] = 0.0
        idx = np.argpartition(row, -k)[-k:]
        out[i, idx] = row[idx]
    return out


def _zero_diag(w: np.ndarray) -> np.ndarray:
    out = w.copy()
    np.fill_diagonal(out, 0.0)
    return out


def _symmetrize(w: np.ndarray) -> np.ndarray:
    return np.maximum(w, w.T).astype(np.float32)


