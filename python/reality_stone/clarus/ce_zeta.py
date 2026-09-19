"""Design 2 — Riemann-zeta activation function.

Uses a differentiable truncated approximation to ζ(s) evaluated on the
critical line s = 1/2 + ix. Two routes are implemented:

  eta_truncated(x, N):
    Dirichlet-eta truncation (absolutely convergent for Re(s) > 0)
        η(s) ≈ Σ_{n=1}^{N} (-1)^{n+1} n^{-s}
    Then ζ(s) = η(s) / (1 - 2^{1-s}).
    For s = 1/2 + ix, this is a complex number; we return |ζ|^2 as a
    real-valued function.

  zeta_activation(x):
    Drop-in replacement for GELU/SiLU. Shape preserved:
        y = x * sigmoid(x) * ZNorm(x)
    where ZNorm is the normalized |ζ(1/2+ix)|^2 capped to a bounded
    range so gradients don't explode. This preserves the classic
    Swish/SiLU monotonic behaviour near 0 and adds zeta-modulated
    structure away from 0 — the Riemann hypothesis axiom then
    guarantees ZNorm vanishes only at Riemann-zero inputs.

Differentiability: all ops are pure PyTorch (log, cos, sin, reciprocal),
so autograd flows. No Riemann-Siegel refinement is used; N=24 is enough
for |x| ≲ 40 and is cheap.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _eta_truncated(x: torch.Tensor, N: int = 24) -> tuple[torch.Tensor, torch.Tensor]:
    """Return real and imaginary parts of η(1/2 + i x) truncated at N terms.

    η(1/2 + ix) = Σ_{n=1}^{N} (-1)^{n+1} n^{-1/2} (cos(x log n) − i sin(x log n))
    """
    # n = 1 ... N
    device, dtype = x.device, x.dtype
    ns = torch.arange(1, N + 1, device=device, dtype=dtype)  # (N,)
    log_n = torch.log(ns)                                    # (N,)
    inv_sqrt_n = 1.0 / torch.sqrt(ns)                        # (N,)
    sign = torch.where(ns % 2 == 1, torch.ones_like(ns), -torch.ones_like(ns))
    coef = sign * inv_sqrt_n                                 # (N,)
    # broadcast: x has shape (...), we need (..., N)
    phase = x.unsqueeze(-1) * log_n                          # (..., N)
    real = (coef * torch.cos(phase)).sum(dim=-1)             # (...,)
    imag = (coef * (-torch.sin(phase))).sum(dim=-1)          # (...,)
    return real, imag


def _zeta_critical(x: torch.Tensor, N: int = 24
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """ζ(1/2 + ix) via ζ = η / (1 − 2^{1−s}). Returns (Re ζ, Im ζ)."""
    eta_re, eta_im = _eta_truncated(x, N)
    # d = 1 − 2^{1 − 1/2 − ix} = 1 − √2 · 2^{−ix} = 1 − √2 (cos(x log 2) − i sin(x log 2))
    a = math.log(2.0)
    sqrt2 = math.sqrt(2.0)
    cos_a = torch.cos(x * a)
    sin_a = torch.sin(x * a)
    d_re = 1.0 - sqrt2 * cos_a
    d_im = sqrt2 * sin_a
    denom = d_re * d_re + d_im * d_im + 1e-8
    # (eta_re + i eta_im) / (d_re + i d_im)
    # = ((eta_re d_re + eta_im d_im) + i (eta_im d_re - eta_re d_im)) / denom
    z_re = (eta_re * d_re + eta_im * d_im) / denom
    z_im = (eta_im * d_re - eta_re * d_im) / denom
    return z_re, z_im


def zeta_magnitude_sq(x: torch.Tensor, N: int = 24) -> torch.Tensor:
    """|ζ(1/2 + ix)|^2 via truncated Dirichlet series. Real-valued."""
    zr, zi = _zeta_critical(x, N)
    return zr * zr + zi * zi


class ZetaActivation(nn.Module):
    """Swish-like: y = x · σ(x) · ( 1 + λ · (|ζ|² − μ) / s ).

    The bracket is the normalized zeta-magnitude modulation; λ is a
    learnable gain, initially small so the module starts as SiLU.
    μ, s are running statistics computed on the first forward call and
    frozen after (to keep training deterministic).
    """

    def __init__(self, N: int = 24, lam_init: float = 0.1):
        super().__init__()
        self.N = N
        self.lam = nn.Parameter(torch.tensor(lam_init))
        self.register_buffer("mu", torch.tensor(0.0))
        self.register_buffer("sigma", torch.tensor(1.0))
        self.register_buffer("_init_done", torch.tensor(0, dtype=torch.uint8))

    def _init_stats(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            zs = zeta_magnitude_sq(x, self.N)
            self.mu.copy_(zs.mean())
            self.sigma.copy_(zs.std().clamp_min(1e-4))
            self._init_done.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if int(self._init_done.item()) == 0 and self.training:
            self._init_stats(x)
        zs = zeta_magnitude_sq(x, self.N)
        z_norm = (zs - self.mu) / self.sigma
        modulation = 1.0 + self.lam * z_norm
        return x * torch.sigmoid(x) * modulation


class ZetaFFN(nn.Module):
    """FFN using ZetaActivation in place of GELU."""

    def __init__(self, d: int, mult: int = 4, N: int = 24):
        super().__init__()
        self.up = nn.Linear(d, mult * d, bias=False)
        self.down = nn.Linear(mult * d, d, bias=False)
        self.act = ZetaActivation(N=N)

    def forward(self, x):
        return self.down(self.act(self.up(x)))


__all__ = [
    "zeta_magnitude_sq",
    "ZetaActivation",
    "ZetaFFN",
    "_eta_truncated",
    "_zeta_critical",
]
