"""Riemannian Adam optimizer for the Poincare ball.

The step runs on the compiled ``reality_stone._rust.poincare`` kernel when it is
built. Otherwise a pure-torch implementation of the same update is used:

* Riemannian gradient ``g_R = g / lambda_x^2`` with ``lambda_x = 2 / (1 - c|x|^2)``
* Adam moments on ``g_R`` (``v`` tracks the Riemannian squared norm)
* retraction with the exponential map at ``x`` and projection back into the ball
"""

import math
from typing import Callable, Optional

import numpy as np
import torch
from torch.optim import Optimizer

try:
    from reality_stone import _rust as _rust_mod
except ImportError:  # pragma: no cover - package layout guarantees the stub
    _rust_mod = None

_HAS_RUST = (
    _rust_mod is not None
    and not bool(getattr(_rust_mod, "IS_FALLBACK", False))
    and hasattr(getattr(_rust_mod, "poincare", None), "poincare_riemannian_adam_step_cpu")
)


def _mobius_add(x: torch.Tensor, y: torch.Tensor, c: float, eps: float) -> torch.Tensor:
    xy = (x * y).sum(dim=-1, keepdim=True)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
    den = 1.0 + 2.0 * c * xy + c * c * x2 * y2
    return num / den.clamp_min(eps)


def _exp_map(x: torch.Tensor, v: torch.Tensor, c: float, eps: float) -> torch.Tensor:
    sqrt_c = math.sqrt(c)
    v_norm = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)
    lambda_x = 2.0 / (1.0 - c * (x * x).sum(dim=-1, keepdim=True)).clamp_min(eps)
    second = torch.tanh(sqrt_c * lambda_x * v_norm / 2.0) * v / (sqrt_c * v_norm)
    return _mobius_add(x, second, c, eps)


def _project(x: torch.Tensor, c: float, max_norm_eps: float) -> torch.Tensor:
    max_norm = (1.0 - max_norm_eps) / math.sqrt(c)
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    scale = torch.where(norm > max_norm, max_norm / norm.clamp_min(1e-12), torch.ones_like(norm))
    return x * scale


def _torch_step(
    x: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    step: int,
    c: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    max_norm_eps: float,
):
    lambda_x = 2.0 / (1.0 - c * (x * x).sum(dim=-1, keepdim=True)).clamp_min(eps)
    rgrad = grad / (lambda_x * lambda_x)
    m_new = beta1 * m + (1.0 - beta1) * rgrad
    riem_sq = ((rgrad * rgrad).sum(dim=-1, keepdim=True) * lambda_x * lambda_x).expand_as(rgrad)
    v_new = beta2 * v + (1.0 - beta2) * riem_sq
    m_hat = m_new / (1.0 - beta1 ** step)
    v_hat = v_new / (1.0 - beta2 ** step)
    direction = -lr * m_hat / (v_hat.sqrt() + eps)
    x_new = _project(_exp_map(x, direction, c, eps), c, max_norm_eps)
    return x_new, m_new, v_new


class PoincareRiemannianAdam(Optimizer):
    def __init__(
        self,
        params,
        c: float,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        max_norm_eps: float = 1e-7,  # Relaxed boundary constraint for f32
    ):
        if c <= 0:
            raise ValueError(f"Curvature c must be positive, got {c}")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if max_norm_eps < 0.0:
            raise ValueError(f"Invalid max_norm_eps value: {max_norm_eps}")
        defaults = dict(lr=lr, betas=betas, eps=eps, c=c, max_norm_eps=max_norm_eps)
        super().__init__(params, defaults)
        self._step = 0

    @property
    def uses_native_kernel(self) -> bool:
        return _HAS_RUST

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._step += 1
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            c = group["c"]
            max_norm_eps = group.get("max_norm_eps", 1e-7)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p, device="cpu", dtype=torch.float32)
                    state["v"] = torch.zeros_like(p, device="cpu", dtype=torch.float32)
                m = state["m"]
                v = state["v"]

                if not _HAS_RUST:
                    x = p.detach().to(device="cpu", dtype=torch.float32)
                    g = grad.to(device="cpu", dtype=torch.float32)
                    is_1d = x.dim() == 1
                    if is_1d:
                        x, g, m, v = x.unsqueeze(0), g.unsqueeze(0), m.unsqueeze(0), v.unsqueeze(0)
                    x_new, m_new, v_new = _torch_step(
                        x, g, m, v, self._step, float(c), float(lr),
                        float(beta1), float(beta2), float(eps), float(max_norm_eps),
                    )
                    if is_1d:
                        x_new, m_new, v_new = x_new.squeeze(0), m_new.squeeze(0), v_new.squeeze(0)
                    p.copy_(x_new.to(device=p.device, dtype=p.dtype))
                    state["m"] = m_new
                    state["v"] = v_new
                    continue

                x_np = p.detach().cpu().numpy().astype(np.float32)
                g_np = grad.cpu().numpy().astype(np.float32)
                m_np = m.cpu().numpy().astype(np.float32)
                v_np = v.cpu().numpy().astype(np.float32)
                is_1d = x_np.ndim == 1
                if is_1d:
                    x_np = x_np.reshape(1, -1)
                    g_np = g_np.reshape(1, -1)
                    m_np = m_np.reshape(1, -1)
                    v_np = v_np.reshape(1, -1)
                x_new_np, m_new_np, v_new_np = _rust_mod.poincare.poincare_riemannian_adam_step_cpu(  # type: ignore[attr-defined]
                    x_np, g_np, m_np, v_np, self._step,
                    float(c), float(lr), float(beta1), float(beta2), float(eps), float(max_norm_eps),
                )
                if is_1d:
                    x_new_np = x_new_np.reshape(-1)
                    m_new_np = m_new_np.reshape(-1)
                    v_new_np = v_new_np.reshape(-1)
                p.copy_(torch.from_numpy(x_new_np).to(p.device))
                state["m"] = torch.from_numpy(m_new_np)
                state["v"] = torch.from_numpy(v_new_np)

        return loss

    def zero_grad(self, set_to_none: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()
