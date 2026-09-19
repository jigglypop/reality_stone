"""CE-theory FFN variants.

Baseline:      Linear(d, 4d) -> GELU -> Linear(4d, d)
SwiGLU:        Linear(d, 4d) + Linear(d, 4d) gated by SiLU (known-good)
EulerDecayFFN: GELU * e-decay gate (e role from CE theory)
EulerPhaseFFN: GELU * sin(π · h) periodic modulation (π role)
EulerFullFFN:  combines π-phase and e-decay simultaneously

Each module exposes the same forward signature (x -> x) and can
replace the standard FFN in any transformer block.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StdFFN(nn.Module):
    """Baseline: Linear -> GELU -> Linear."""

    def __init__(self, d: int, mult: int = 4):
        super().__init__()
        self.up = nn.Linear(d, mult * d, bias=False)
        self.down = nn.Linear(mult * d, d, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class SwiGLU_FFN(nn.Module):
    """SwiGLU — Llama / PaLM standard. Known to beat GELU-FFN."""

    def __init__(self, d: int, mult: int = 4):
        super().__init__()
        # keep param count comparable to StdFFN with mult=4:
        # SwiGLU has 2 up-projections so we use mult' = mult*2/3 rounded
        hidden = int(mult * 2 / 3 * d)
        # round to multiple of 8
        hidden = ((hidden + 7) // 8) * 8
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class EulerDecayFFN(nn.Module):
    """GELU modulated by e-decay gate: survival S = e^{-|h|/xi}.

    h   = W_up x
    s   = exp(-|h| / xi)       # e-decay, learnable log_xi
    a   = GELU(h) * s
    y   = W_down a

    Rationale: CE's survival function S(D) = e^{-D} used as an adaptive
    saturation. Large activations get softly clipped by e-decay rather
    than growing without bound (as in GELU).
    """

    def __init__(self, d: int, mult: int = 4, xi_init: float = 3.0):
        super().__init__()
        self.up = nn.Linear(d, mult * d, bias=False)
        self.down = nn.Linear(mult * d, d, bias=False)
        self.log_xi = nn.Parameter(torch.tensor(math.log(xi_init)))

    def forward(self, x):
        h = self.up(x)
        xi = torch.exp(self.log_xi)
        survival = torch.exp(-h.abs() / xi)
        return self.down(F.gelu(h) * survival)


class EulerPhaseFFN(nn.Module):
    """GELU with π-periodic modulation: a = GELU(h) * (1 + eta · cos(π h / tau)).

    Provides a periodic (phase) dimension to the pointwise nonlinearity
    without spoiling the GELU monotonic region near 0.
    """

    def __init__(self, d: int, mult: int = 4, tau_init: float = 2.0,
                 eta_init: float = 0.1):
        super().__init__()
        self.up = nn.Linear(d, mult * d, bias=False)
        self.down = nn.Linear(mult * d, d, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        self.eta = nn.Parameter(torch.tensor(eta_init))

    def forward(self, x):
        h = self.up(x)
        tau = torch.exp(self.log_tau)
        phase = torch.cos(math.pi * h / tau)
        return self.down(F.gelu(h) * (1.0 + self.eta * phase))


class EulerFullFFN(nn.Module):
    """Full Euler FFN: π-phase × e-decay on top of GELU.

    a = GELU(h) * (1 + eta · cos(π h / tau)) * exp(-|h|/xi)
    """

    def __init__(self, d: int, mult: int = 4,
                 xi_init: float = 3.0,
                 tau_init: float = 2.0,
                 eta_init: float = 0.1):
        super().__init__()
        self.up = nn.Linear(d, mult * d, bias=False)
        self.down = nn.Linear(mult * d, d, bias=False)
        self.log_xi = nn.Parameter(torch.tensor(math.log(xi_init)))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        self.eta = nn.Parameter(torch.tensor(eta_init))

    def forward(self, x):
        h = self.up(x)
        xi = torch.exp(self.log_xi)
        tau = torch.exp(self.log_tau)
        survival = torch.exp(-h.abs() / xi)
        phase = torch.cos(math.pi * h / tau)
        return self.down(F.gelu(h) * (1.0 + self.eta * phase) * survival)


def make_ffn(kind: str, d: int, mult: int = 4) -> nn.Module:
    if kind == "std":
        return StdFFN(d, mult)
    if kind == "swiglu":
        return SwiGLU_FFN(d, mult)
    if kind == "euler_decay":
        return EulerDecayFFN(d, mult)
    if kind == "euler_phase":
        return EulerPhaseFFN(d, mult)
    if kind == "euler_full":
        return EulerFullFFN(d, mult)
    if kind == "zeta":
        from .ce_zeta import ZetaFFN
        return ZetaFFN(d, mult)
    raise ValueError(f"unknown ffn kind: {kind!r}")


__all__ = [
    "StdFFN", "SwiGLU_FFN", "EulerDecayFFN", "EulerPhaseFFN", "EulerFullFFN",
    "make_ffn",
]
