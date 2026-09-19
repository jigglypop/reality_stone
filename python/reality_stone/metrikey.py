from __future__ import annotations

import numpy as np
import torch

from ._fallback import deterministic_spd


def spd_metric_from_key_weighted(
    key: str,
    dim: int,
    min_lambda: float = 0.8,
    max_lambda: float = 1.2,
    mass: float = 1.0,
) -> np.ndarray:
    return deterministic_spd(key, dim, min_lambda, max_lambda, mass)


def spd_metric_from_key(
    key: str,
    dim: int,
    min_lambda: float = 0.8,
    max_lambda: float = 1.2,
) -> np.ndarray:
    return spd_metric_from_key_weighted(key, dim, min_lambda, max_lambda, 1.0)


def metric_factor_cholesky(g) -> np.ndarray:
    g_arr = np.asarray(g, dtype=np.float32)
    g_arr = 0.5 * (g_arr + g_arr.T)
    jitter = np.eye(g_arr.shape[0], dtype=np.float32) * 1e-5
    return np.linalg.cholesky(g_arr + jitter).astype(np.float32)


def metric_from_keys(
    keys,
    dim: int,
    min_lambda: float = 0.8,
    max_lambda: float = 1.2,
    masses=None,
):
    if masses is None:
        masses = [1.0] * len(keys)
    acc = np.zeros((dim, dim), dtype=np.float32)
    total = 0.0
    for key, mass in zip(keys, masses):
        m = float(mass)
        acc += spd_metric_from_key_weighted(str(key), dim, min_lambda, max_lambda, max(m, 1e-6))
        total += max(m, 1e-6)
    if total <= 0.0:
        total = 1.0
    return torch.from_numpy((acc / total).astype(np.float32))


def mahalanobis_distance_sq_g(x, y, g) -> float:
    dx = np.asarray(x, dtype=np.float32) - np.asarray(y, dtype=np.float32)
    g_arr = np.asarray(g, dtype=np.float32)
    return float(dx.T @ g_arr @ dx)


def mahalanobis_distance_sq_l(x, y, l_factor) -> float:
    dx = np.asarray(x, dtype=np.float32) - np.asarray(y, dtype=np.float32)
    l_arr = np.asarray(l_factor, dtype=np.float32)
    z = l_arr @ dx
    return float(z.T @ z)
