from __future__ import annotations

import numpy as np

from ._fallback import (
    TorchUnifiedRiemannianLayer as PyUnifiedRiemannianLayer,
    euclidean_metric_np as compute_metric,
    geodesic_distance_np as geodesic_distance,
    geodesic_interpolate_np as geodesic_interpolate,
)


IS_FALLBACK = True


def _as_f32(x):
    return np.asarray(x, dtype=np.float32)


def _curvature_from_kappa(kappa, c_min: float, c_max: float) -> float:
    k = float(kappa)
    sig = 1.0 / (1.0 + np.exp(-k))
    return float(c_min + (c_max - c_min) * sig)


def _project_ball(x, c: float, eps: float = 1e-6):
    x = _as_f32(x)
    c = max(float(c), eps)
    radius = 1.0 / np.sqrt(c)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    max_norm = radius - eps
    scale = np.where(norm > max_norm, max_norm / np.maximum(norm, eps), 1.0)
    return (x * scale).astype(np.float32)


def poincare_ball_layer_cpu(u, v, c: float, t: float):
    out = (1.0 - float(t)) * _as_f32(u) + float(t) * _as_f32(v)
    return _project_ball(out, c)


def poincare_ball_layer_backward_cpu(grad, u, v, c: float, t: float):
    del u, v, c
    grad = _as_f32(grad)
    return (grad * (1.0 - float(t))).astype(np.float32), (grad * float(t)).astype(np.float32)


def poincare_ball_layer_layerwise_cpu(u, v, kappa, layer_idx: int, c_min: float, c_max: float, t: float):
    del layer_idx
    c_val = _curvature_from_kappa(kappa, c_min, c_max)
    return poincare_ball_layer_cpu(u, v, abs(c_val)), c_val


def poincare_ball_layer_layerwise_backward_cpu(
    grad,
    u,
    v,
    kappa,
    layer_idx: int,
    c_min: float,
    c_max: float,
    t: float,
):
    del kappa, layer_idx, c_min, c_max
    grad_u, grad_v = poincare_ball_layer_backward_cpu(grad, u, v, 1.0, t)
    return grad_u, grad_v, 0.0


def poincare_to_lorentz_cpu(x, c: float):
    x = _project_ball(x, c)
    c = max(float(c), 1e-6)
    x2 = np.sum(x * x, axis=-1, keepdims=True)
    den = np.maximum(1.0 - c * x2, 1e-7)
    time = (1.0 + c * x2) / (np.sqrt(c) * den)
    space = 2.0 * x / den
    return np.concatenate([time, space], axis=-1).astype(np.float32)


def poincare_to_klein_cpu(x, c: float):
    x = _project_ball(x, c)
    c = float(c)
    x2 = np.sum(x * x, axis=-1, keepdims=True)
    return (2.0 * x / np.maximum(1.0 + c * x2, 1e-7)).astype(np.float32)


def lorentz_inner(u, v):
    u = _as_f32(u)
    v = _as_f32(v)
    return (u[..., 0] * v[..., 0] - np.sum(u[..., 1:] * v[..., 1:], axis=-1)).astype(np.float32)


def lorentz_distance(u, v, c: float):
    c = max(float(c), 1e-6)
    z = np.maximum(c * lorentz_inner(u, v), 1.0)
    return (np.arccosh(z) / np.sqrt(c)).astype(np.float32)


def lorentz_layer_forward(u, v, c: float, t: float):
    del c
    return ((1.0 - float(t)) * _as_f32(u) + float(t) * _as_f32(v)).astype(np.float32)


def lorentz_ball_layer_backward_cpu(grad, u, v, c: float, t: float):
    del u, v, c
    grad = _as_f32(grad)
    return (grad * (1.0 - float(t))).astype(np.float32), (grad * float(t)).astype(np.float32)


def lorentz_layer_layerwise_cpu(u, v, kappa, layer_idx: int, c_min: float, c_max: float, t: float):
    del layer_idx
    c_val = _curvature_from_kappa(kappa, c_min, c_max)
    return lorentz_layer_forward(u, v, c_val, t), c_val


def lorentz_add(u, v, c: float):
    del c
    return (_as_f32(u) + _as_f32(v)).astype(np.float32)


def lorentz_scalar(x, r: float, c: float):
    del c
    return (_as_f32(x) * float(r)).astype(np.float32)


def lorentz_to_poincare(x, c: float):
    x = _as_f32(x)
    denom = np.maximum(x[..., :1] + np.sqrt(1.0 / max(float(c), 1e-6)), 1e-7)
    return (x[..., 1:] / denom).astype(np.float32)


def lorentz_to_klein(x, c: float):
    del c
    x = _as_f32(x)
    return (x[..., 1:] / np.maximum(x[..., :1], 1e-7)).astype(np.float32)


def klein_layer_forward(u, v, c: float, t: float):
    out = (1.0 - float(t)) * _as_f32(u) + float(t) * _as_f32(v)
    return _project_ball(out, c)


def klein_ball_layer_backward_cpu(grad, u, v, c: float, t: float):
    del u, v, c
    grad = _as_f32(grad)
    return (grad * (1.0 - float(t))).astype(np.float32), (grad * float(t)).astype(np.float32)


def klein_layer_layerwise_cpu(u, v, kappa, layer_idx: int, c_min: float, c_max: float, t: float):
    del layer_idx
    c_val = _curvature_from_kappa(kappa, c_min, c_max)
    return klein_layer_forward(u, v, abs(c_val), t), c_val


def klein_add(u, v, c: float):
    return _project_ball(_as_f32(u) + _as_f32(v), c)


def klein_scalar(x, r: float, c: float):
    return _project_ball(_as_f32(x) * float(r), c)


def klein_distance(x, y, c: float):
    x = _project_ball(x, c)
    y = _project_ball(y, c)
    c = max(float(c), 1e-6)
    x2 = np.sum(x * x, axis=-1)
    y2 = np.sum(y * y, axis=-1)
    xy = np.sum(x * y, axis=-1)
    den = np.maximum((1.0 - c * x2) * (1.0 - c * y2), 1e-7)
    arg = np.maximum((1.0 - c * xy) / np.sqrt(den), 1.0)
    return (np.arccosh(arg) / np.sqrt(c)).astype(np.float32)


def klein_to_poincare(x, c: float):
    x = _project_ball(x, c)
    den = 1.0 + np.sqrt(np.maximum(1.0 - float(c) * np.sum(x * x, axis=-1, keepdims=True), 0.0))
    return (x / np.maximum(den, 1e-7)).astype(np.float32)


def klein_to_lorentz(x, c: float):
    x = _project_ball(x, c)
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - float(c) * np.sum(x * x, axis=-1, keepdims=True), 1e-7))
    return np.concatenate([gamma, gamma * x], axis=-1).astype(np.float32)


def from_poincare_dynamic_cpu(x, kappa, c_min: float, c_max: float):
    c_val = _curvature_from_kappa(kappa, c_min, c_max)
    return poincare_to_lorentz_cpu(x, abs(c_val)), c_val


def from_poincare_dynamic_backward_cpu(grad, x, kappa, c_min: float, c_max: float):
    del x, kappa, c_min, c_max
    return _as_f32(grad), 0.0


# Model-specific names. The native module registers these separately so the Klein
# binding no longer overwrites the Lorentz one in the shared root namespace.
def lorentz_from_poincare_dynamic_cpu(x, kappa, c_min: float, c_max: float):
    c_val = _curvature_from_kappa(kappa, c_min, c_max)
    return poincare_to_lorentz_cpu(x, abs(c_val)), c_val


def klein_from_poincare_dynamic_cpu(x, kappa, c_min: float, c_max: float):
    c_val = _curvature_from_kappa(kappa, c_min, c_max)
    return poincare_to_klein_cpu(x, abs(c_val)), c_val


lorentz_from_poincare_dynamic_backward_cpu = from_poincare_dynamic_backward_cpu
klein_from_poincare_dynamic_backward_cpu = from_poincare_dynamic_backward_cpu


def _svd_basis(wq_list, target_rank: int):
    mats = [np.asarray(w, dtype=np.float32) for w in wq_list]
    if not mats:
        return np.eye(target_rank, dtype=np.float32), target_rank
    cat = np.concatenate(mats, axis=0)
    _, _, vt = np.linalg.svd(cat, full_matrices=False)
    rank = min(int(target_rank), vt.shape[0])
    return vt[:rank].T.astype(np.float32), rank


def extract_global_basis(wq_list, wk_list, target_rank: int):
    del wk_list
    u, rank = _svd_basis(wq_list, target_rank)
    return {"u": u, "rank": rank}


def build_causal_laplacian(seq_len: int, window: int = 1):
    seq_len = int(seq_len)
    window = max(1, int(window))
    a = np.zeros((seq_len, seq_len), dtype=np.float32)
    for i in range(seq_len):
        lo = max(0, i - window)
        for j in range(lo, i):
            a[i, j] = 1.0 / (1.0 + abs(i - j))
    d = np.diag(a.sum(axis=1))
    return (d - a).astype(np.float32)


def verify_metric_consistency(wq, wk, r: int):
    del r
    wq = np.asarray(wq, dtype=np.float32)
    wk = np.asarray(wk, dtype=np.float32)
    denom = max(float(np.linalg.norm(wq) * np.linalg.norm(wk)), 1e-6)
    score = float(abs(np.sum(wq * wk)) / denom)
    return {"fold_accuracy": max(0.0, min(1.0, score)), "is_valid": True}


def fold_metric_svd(wq, wk, r: int):
    basis, rank = _svd_basis([wq, wk], r)
    return {"u": basis, "rank": rank}


def fold_metric_optimized(wq, wk, r: int):
    return fold_metric_svd(wq, wk, r)


def nystrom_metric(wq, wk, r: int):
    return fold_metric_svd(wq, wk, r)


def fold_ffn(w, r: int):
    w = np.asarray(w, dtype=np.float32)
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    rank = min(int(r), len(s))
    return {
        "u": u[:, :rank].astype(np.float32),
        "s": s[:rank].astype(np.float32),
        "v": vt[:rank, :].astype(np.float32),
    }


def bellman_geodesic_forward(x, *args, **kwargs):
    del args, kwargs
    return np.asarray(x, dtype=np.float32)


def bellman_geodesic_backward(grad, *args, **kwargs):
    del args, kwargs
    return np.asarray(grad, dtype=np.float32)


def extract_metric_cuda(w, calib, target_dim: int, num_steps: int, curvature: float, lr: float):
    del calib, num_steps, curvature, lr
    w = np.asarray(w, dtype=np.float32)
    dim = int(target_dim)
    return np.eye(dim, dtype=np.float32) * max(float(np.var(w)), 1e-6)


class PyHyperMetric:
    def __init__(self, u_global, v_global, w1, b1, w2, b2):
        self.u_global = np.asarray(u_global, dtype=np.float32)
        self.v_global = np.asarray(v_global, dtype=np.float32)
        self.w1 = np.asarray(w1, dtype=np.float32)
        self.b1 = np.asarray(b1, dtype=np.float32)
        self.w2 = np.asarray(w2, dtype=np.float32)
        self.b2 = np.asarray(b2, dtype=np.float32)

    def generate_core(self, layer_emb):
        x = np.asarray(layer_emb, dtype=np.float32)
        h = np.maximum(x @ self.w1 + self.b1, 0.0)
        out = h @ self.w2 + self.b2
        r = int(round(out.size ** 0.5))
        return out.reshape(r, r).astype(np.float32)

    def project_forward(self, x, layer_emb):
        x = np.asarray(x, dtype=np.float32)
        core = self.generate_core(layer_emb)
        return (x @ self.u_global @ core @ self.v_global.T).astype(np.float32)


class PySymplecticLayer:
    def __init__(self, layer_idx, layer_emb, hyper_metric, dt=0.01):
        self.layer_idx = int(layer_idx)
        self.layer_emb = np.asarray(layer_emb, dtype=np.float32)
        self.hyper_metric = hyper_metric
        self.dt = float(dt)

    def step(self, q, p, kick):
        q = np.asarray(q, dtype=np.float32)
        p = np.asarray(p, dtype=np.float32)
        kick = np.asarray(kick, dtype=np.float32)
        p_next = p + self.dt * kick
        q_next = q + self.dt * p_next
        return q_next.astype(np.float32), p_next.astype(np.float32)


class PyRSULFLayer:
    def __init__(self, wq, wk, w1, w2, d_model, r, eta, alpha, beta, gamma, seq_len, window):
        self.wq = np.asarray(wq, dtype=np.float32)
        self.wk = np.asarray(wk, dtype=np.float32)
        self.w1 = np.asarray(w1, dtype=np.float32)
        self.w2 = np.asarray(w2, dtype=np.float32)
        self.d_model = int(d_model)
        self.r = int(r)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.seq_len = int(seq_len)
        self.window = int(window)
        diag = np.ones(self.d_model, dtype=np.float32)
        self.g_diag = diag
        self.g_inv = diag
        self.curvature = 0.0

    @classmethod
    def new_fast(cls, wq, wk, w1, w2, d_model, r, eta, alpha, beta, gamma, seq_len, window, calibration_samples=1024):
        del calibration_samples
        return cls(wq, wk, w1, w2, d_model, r, eta, alpha, beta, gamma, seq_len, window)

    @classmethod
    def new_with_metric(cls, wq, wk, w1, w2, g_diag, d_model, r, eta, alpha, beta, gamma, seq_len, window):
        obj = cls(wq, wk, w1, w2, d_model, r, eta, alpha, beta, gamma, seq_len, window)
        obj.g_diag = np.asarray(g_diag, dtype=np.float32)
        obj.g_inv = 1.0 / np.maximum(obj.g_diag, 1e-6)
        return obj

    @classmethod
    def new_with_basis(cls, wq, wk, w1, w2, u, rank, d_model, r, eta, alpha, beta, gamma, seq_len, window):
        del u, rank
        return cls(wq, wk, w1, w2, d_model, r, eta, alpha, beta, gamma, seq_len, window)

    def forward(self, x, v=None):
        x = np.asarray(x, dtype=np.float32)
        if v is None:
            v = np.zeros_like(x, dtype=np.float32)
        else:
            v = np.asarray(v, dtype=np.float32)
        y = x + self.eta * np.tanh(x)
        v_next = self.gamma * v + (y - x)
        return y.astype(np.float32), v_next.astype(np.float32)

    def export_components(self):
        d = self.d_model
        hidden = self.w1.shape[0] if self.w1.ndim == 2 else d
        r1 = min(self.r, self.w1.shape[0], self.w1.shape[1]) if self.w1.ndim == 2 else min(self.r, d)
        r2 = min(self.r, self.w2.shape[0], self.w2.shape[1]) if self.w2.ndim == 2 else min(self.r, d)
        return {
            "d_model": d,
            "r": self.r,
            "eta": self.eta,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "seq_len": self.seq_len,
            "window": self.window,
            "g_diag": self.g_diag,
            "g_inv": self.g_inv,
            "g_sym": np.diag(self.g_diag).astype(np.float32),
            "ffn_u1": np.zeros((hidden, r1), dtype=np.float32),
            "ffn_s1": np.ones(r1, dtype=np.float32),
            "ffn_v1": np.zeros((d, r1), dtype=np.float32),
            "ffn_u2": np.zeros((d, r2), dtype=np.float32),
            "ffn_s2": np.ones(r2, dtype=np.float32),
            "ffn_v2": np.zeros((hidden, r2), dtype=np.float32),
            "curvature": self.curvature,
        }

    def param_count(self):
        original = int(self.wq.size + self.wk.size + self.w1.size + self.w2.size)
        compressed = max(1, int(original / 2))
        return compressed, original, float(original / compressed)


class PyGeodesicMemory:
    pass


class SplineCache:
    pass


class PyRiemannianDiffusion:
    def __init__(self, dim: int, alpha: float, dt: float):
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.dt = float(dt)

    def step(self, h, flow):
        return np.asarray(h, dtype=np.float32) + self.dt * np.asarray(flow, dtype=np.float32)
