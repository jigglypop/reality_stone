from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np
import torch


EPS = 1e-7


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def dynamic_curvature(kappa: float, c_min: float, c_max: float) -> float:
    return float(c_min) + (float(c_max) - float(c_min)) * sigmoid(float(kappa))


def _curvature_tensor(c, like: torch.Tensor) -> torch.Tensor:
    """Curvature as a tensor on ``like``'s device/dtype; keeps autograd through tensor ``c``."""
    if isinstance(c, torch.Tensor):
        return c.to(dtype=like.dtype, device=like.device)
    return torch.as_tensor(float(c), dtype=like.dtype, device=like.device)


def mobius_add_torch(x: torch.Tensor, y: torch.Tensor, c) -> torch.Tensor:
    c_t = _curvature_tensor(c, x)
    xy = (x * y).sum(dim=-1, keepdim=True)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c_t * xy + c_t * y2) * x + (1.0 - c_t * x2) * y
    den = 1.0 + 2.0 * c_t * xy + c_t * c_t * x2 * y2
    return num / den.clamp_min(EPS)


def mobius_scalar_torch(x: torch.Tensor, r: float, c) -> torch.Tensor:
    r = float(r)
    c_t = _curvature_tensor(c, x)
    c_value = float(c_t.detach())
    if abs(r) < EPS:
        return torch.zeros_like(x)
    if abs(c_value) < EPS:
        return x * r
    norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
    if c_value > 0.0:
        sqrt_c = torch.sqrt(c_t)
        arg = (sqrt_c * norm).clamp(max=1.0 - EPS)
        scale = torch.tanh(r * torch.atanh(arg)) / (sqrt_c * norm)
        return scale * x
    # For signed/experimental curvature schedules, keep a stable Euclidean limit.
    return x * r


def lorentz_inner_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x[..., 0] * y[..., 0] - (x[..., 1:] * y[..., 1:]).sum(dim=-1)


def lorentz_distance_torch(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    c = float(c)
    if abs(c) < EPS:
        return torch.linalg.norm(x - y, dim=-1)
    z = c * lorentz_inner_torch(x, y)
    valid = z >= 1.0
    hyper = torch.acosh(z.clamp_min(1.0)) / math.sqrt(abs(c))
    euclid = torch.linalg.norm(x - y, dim=-1)
    return torch.where(valid, hyper, euclid)


def klein_distance_torch(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    c = float(c)
    if abs(c) < EPS:
        return torch.linalg.norm(x - y, dim=-1)
    x2 = (x * x).sum(dim=-1)
    y2 = (y * y).sum(dim=-1)
    xy = (x * y).sum(dim=-1)
    den = ((1.0 - c * x2) * (1.0 - c * y2)).clamp_min(EPS).sqrt()
    arg = ((1.0 - c * xy) / den).clamp_min(1.0)
    return torch.acosh(arg) / math.sqrt(abs(c))


def euclidean_metric_np(x: np.ndarray, metric_type: str, curvature: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if metric_type == "poincare" and abs(curvature) > EPS:
        x2 = np.sum(x * x, axis=-1, keepdims=True)
        lam = 4.0 / np.maximum((1.0 - curvature * x2) ** 2, EPS)
        return np.repeat(lam, x.shape[-1], axis=-1).astype(np.float32)
    return np.ones_like(x, dtype=np.float32)


def geodesic_distance_np(
    x: np.ndarray,
    y: np.ndarray,
    metric_type: str = "diagonal",
    curvature: float = 0.0,
) -> np.ndarray:
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float32))
    if metric_type == "lorentz":
        return lorentz_distance_torch(x_t, y_t, curvature).numpy().astype(np.float32)
    if metric_type == "klein":
        return klein_distance_torch(x_t, y_t, curvature).numpy().astype(np.float32)
    if metric_type == "poincare" and abs(float(curvature)) > EPS:
        x2 = (x_t * x_t).sum(dim=-1)
        y2 = (y_t * y_t).sum(dim=-1)
        diff2 = ((x_t - y_t) * (x_t - y_t)).sum(dim=-1).clamp_min(0.0)
        den = ((1.0 - curvature * x2) * (1.0 - curvature * y2)).clamp_min(EPS)
        arg = 1.0 + 2.0 * curvature * diff2 / den
        return (torch.acosh(arg.clamp_min(1.0)) / math.sqrt(abs(float(curvature)))).numpy().astype(np.float32)
    return np.linalg.norm(np.asarray(x, dtype=np.float32) - np.asarray(y, dtype=np.float32), axis=-1).astype(np.float32)


def geodesic_interpolate_np(
    x: np.ndarray,
    y: np.ndarray,
    metric_type: str = "diagonal",
    curvature: float = 0.0,
    t: float = 0.5,
) -> np.ndarray:
    del metric_type, curvature
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    t = float(t)
    return ((1.0 - t) * x + t * y).astype(np.float32)


class TorchUnifiedRiemannianLayer:
    def __init__(
        self,
        metric_type: str = "diagonal",
        curvature: float = 0.0,
        input_dim: int = 1,
        enable_bellman: bool = False,
    ) -> None:
        self.metric_type = str(metric_type)
        self.curvature = float(curvature)
        self.input_dim = int(input_dim)
        self.enable_bellman = bool(enable_bellman)

    def forward(self, x, target=None):
        x_arr = np.asarray(x, dtype=np.float32)
        if target is None:
            out = x_arr.copy()
            velocity = np.zeros_like(x_arr)
        else:
            target_arr = np.asarray(target, dtype=np.float32)
            out = geodesic_interpolate_np(x_arr, target_arr, self.metric_type, self.curvature, 0.5)
            velocity = out - x_arr
        energy = self.compute_energy(x_arr, velocity, out, np.zeros(x_arr.shape[0], dtype=np.float32)) if self.enable_bellman else None
        return out, energy

    def backward(self, grad_output, x):
        del x
        return np.asarray(grad_output, dtype=np.float32)

    def geodesic_path(self, x, y, num_steps: int = 10):
        steps = max(2, int(num_steps))
        return [
            geodesic_interpolate_np(x, y, self.metric_type, self.curvature, i / float(steps - 1))
            for i in range(steps)
        ]

    def compute_energy(self, x, v, x_next, reward):
        x = np.asarray(x, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        x_next = np.asarray(x_next, dtype=np.float32)
        reward = np.asarray(reward, dtype=np.float32).reshape(-1)
        kinetic = 0.5 * np.sum(v * v, axis=-1)
        potential = 0.5 * np.sum(x_next * x_next, axis=-1) - reward
        lagrangian = kinetic - potential
        bellman_residual = np.linalg.norm(x_next - x, axis=-1)
        return {
            "kinetic": kinetic.astype(np.float32),
            "potential": potential.astype(np.float32),
            "lagrangian": lagrangian.astype(np.float32),
            "bellman_residual": bellman_residual.astype(np.float32),
        }

    def flow_step(self, x, num_steps: int = 1, learning_rate: float = 0.01):
        out = np.asarray(x, dtype=np.float32).copy()
        lr = float(learning_rate)
        for _ in range(max(1, int(num_steps))):
            out = out - lr * out
        return out.astype(np.float32)


def _hash_seed(parts: Iterable[object]) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8", "replace")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def deterministic_spd(key: str, dim: int, min_lambda: float, max_lambda: float, mass: float = 1.0) -> np.ndarray:
    dim = int(dim)
    rng = np.random.default_rng(_hash_seed((key, dim, min_lambda, max_lambda, mass)))
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)).astype(np.float32))
    vals = rng.uniform(float(min_lambda), float(max_lambda), size=dim).astype(np.float32)
    vals = vals * max(float(mass), EPS)
    return (q @ np.diag(vals) @ q.T).astype(np.float32)


# ---------------------------------------------------------------------------
# Differentiable torch formulas shared by the layer fallbacks.
#
# These are the mathematical definitions the compiled kernels implement; the
# layer modules call them whenever the native ``_rust`` extension is absent so
# that forward values and autograd gradients stay correct without Rust.
# ---------------------------------------------------------------------------


def dynamic_curvature_torch(kappa: torch.Tensor, c_min: float, c_max: float) -> torch.Tensor:
    return float(c_min) + (float(c_max) - float(c_min)) * torch.sigmoid(kappa)


def autograd_vjp(fn, grad_output: torch.Tensor, *tensors: torch.Tensor):
    """Vector-Jacobian product of ``fn`` at ``tensors`` computed with autograd.

    Returns one gradient per input tensor (zeros where the output does not depend
    on the input). Used inside ``autograd.Function.backward`` fallbacks so the
    custom Functions keep exact gradients when no native VJP kernel exists.
    """
    with torch.enable_grad():
        leaves = [t.detach().clone().requires_grad_(True) for t in tensors]
        out = fn(*leaves)
        grads = torch.autograd.grad(out, leaves, grad_output, allow_unused=True)
    return tuple(
        g if g is not None else torch.zeros_like(t)
        for g, t in zip(grads, tensors)
    )


def poincare_ball_layer_torch(u: torch.Tensor, v: torch.Tensor, c: float, t: float) -> torch.Tensor:
    """Geodesic blend on the Poincare ball, ``((1-t) (x) u) (+) (t (x) v)``."""
    return mobius_add_torch(
        mobius_scalar_torch(u, 1.0 - float(t), c),
        mobius_scalar_torch(v, float(t), c),
        c,
    )


def poincare_to_lorentz_torch(x: torch.Tensor, c) -> torch.Tensor:
    c_t = _curvature_tensor(c, x)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    den = (1.0 - c_t * x2).clamp_min(EPS)
    time = (1.0 + c_t * x2) / (torch.sqrt(c_t) * den)
    space = 2.0 * x / den
    return torch.cat([time, space], dim=-1)


def lorentz_to_poincare_torch(x: torch.Tensor, c) -> torch.Tensor:
    denom = x[..., :1] + torch.rsqrt(_curvature_tensor(c, x))
    return x[..., 1:] / denom.clamp_min(EPS)


def poincare_to_klein_torch(x: torch.Tensor, c) -> torch.Tensor:
    c_t = _curvature_tensor(c, x)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    return (2.0 * x) / (1.0 + c_t * x2).clamp_min(EPS)


def lorentz_geodesic_torch(u: torch.Tensor, v: torch.Tensor, c: float, t: float) -> torch.Tensor:
    """Hyperboloid geodesic ``sinh((1-t)a)/sinh(a) u + sinh(t a)/sinh(a) v``.

    ``a = acosh(c <u,v>_L)`` with the ``(+,-,...,-)`` Minkowski inner product used
    throughout this package. Falls back to the linear blend when ``a`` vanishes.
    """
    c_t = _curvature_tensor(c, u)
    t = float(t)
    inner = lorentz_inner_torch(u, v).unsqueeze(-1)
    z = (c_t * inner).clamp_min(1.0 + EPS)
    alpha = torch.acosh(z)
    sinh_a = torch.sinh(alpha).clamp_min(EPS)
    small = alpha.abs() < 1e-6
    w1 = torch.where(small, torch.full_like(alpha, 1.0 - t), torch.sinh((1.0 - t) * alpha) / sinh_a)
    w2 = torch.where(small, torch.full_like(alpha, t), torch.sinh(t * alpha) / sinh_a)
    return w1 * u + w2 * v


def lorentz_add_torch(u: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
    """Gyro-addition on the hyperboloid via the Poincare ball (Mobius) chart."""
    pu = lorentz_to_poincare_torch(u, c)
    pv = lorentz_to_poincare_torch(v, c)
    return poincare_to_lorentz_torch(mobius_add_torch(pu, pv, c), c)


def lorentz_scalar_torch(x: torch.Tensor, r: float, c: float) -> torch.Tensor:
    px = lorentz_to_poincare_torch(x, c)
    return poincare_to_lorentz_torch(mobius_scalar_torch(px, r, c), c)


def einstein_add_torch(u: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
    """Einstein addition in the Beltrami-Klein model.

    ``u (+)_E v = [u + v / g_u + (c g_u / (1 + g_u)) <u,v> u] / (1 + c <u,v>)`` with
    the Lorentz factor ``g_u = 1 / sqrt(1 - c |u|^2)``.
    """
    c = float(c)
    if abs(c) < EPS:
        return u + v
    uv = (u * v).sum(dim=-1, keepdim=True)
    u2 = (u * u).sum(dim=-1, keepdim=True)
    gamma_u = torch.rsqrt((1.0 - c * u2).clamp_min(EPS))
    num = u + v / gamma_u + (c * gamma_u / (1.0 + gamma_u)) * uv * u
    return num / (1.0 + c * uv).clamp_min(EPS)
