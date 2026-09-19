"""Mellin-Riemann Attention Block (MRA) — refactored with ablation modes.

Spec: docs/8_리만/mra_block_spec.md

Design surface area (all optional, default == lean MRA):

    MellinRiemannAttention(
        freq_mode       = "rope"  | "zeta_log",     # position frequencies
        amp_weight      = True    | False,          # ζ amplitude weighting w_k
        decay_mode      = "none"  | "bias" | "mult",# critical-line decay form
        sparse_eps2     = 0.0     | 0.0487          # bootstrap top-k retention
        hermitian       = False   | True            # bidirectional only
        spectral_norm_o = False   | True            # σ₁(W_o) ≤ 1
    )

The only component that is **genuinely novel** in MRA is the ζ amplitude
weighting `w_k = 1/(1/2 + iγ_k)`. The other knobs exist for ablation and
to reach parity with the preceding `riemann_rope` formulation for
regression testing.

Notes on the earlier (now opt-in) components:

* `freq_mode="zeta_log"` uses `θ(p,k) = γ_k log(1+p)` (Mellin-kernel).
  γ_k/γ_1 spans only ~5×, which collapses RoPE's multi-scale resolution
  (10000^{-2k/d} spans ~3000×). Kept for ablation; not recommended.
* `decay_mode="mult"` multiplies scores by √((1+j)/(1+i)). This is a
  per-key temperature, not an additive log-bias; usually under-performs
  `"bias"` which applies ½(log(1+j)-log(1+i)) as an ALiBi-style add.
* `hermitian=True` symmetrises scores before the causal mask, which
  leaks future information into past scores on a causal LM. For
  bidirectional encoders it realises Hilbert–Pólya self-adjointness
  directly; for causal LM it is **unsafe**.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ce_riemann_attn import riemann_zeros


# Bootstrap fixed point from docs/6_뇌/sleep.md.
BOOTSTRAP_EPS2 = 0.0487


# ---------------------------------------------------------------------------
# Bootstrap sparsity
# ---------------------------------------------------------------------------


def bootstrap_sparse(
    attn: torch.Tensor,
    eps2: float = BOOTSTRAP_EPS2,
    min_keep: int = 1,
) -> torch.Tensor:
    """Top-k retention per row to enforce activation ratio ≈ ε² · N."""
    n = attn.shape[-1]
    k = max(min_keep, math.ceil(eps2 * n))
    if k >= n:
        return attn
    _, idx = attn.topk(k, dim=-1)
    mask = torch.zeros_like(attn).scatter_(-1, idx, 1.0)
    masked = attn * mask
    return masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-9)


# ---------------------------------------------------------------------------
# Mellin-Riemann Attention
# ---------------------------------------------------------------------------


class MellinRiemannAttention(nn.Module):
    """Self-attention with ζ-amplitude weighting.

    The score is computed as

        Re(S_{ij}) = Σ_k Re(w_k · q̃_i^{(k)} · conj(k̃_j^{(k)}))

    where ``w_k = 1/(1/2 + iγ_k)`` is the ζ explicit-formula amplitude
    (buffer, axiomatic from the Riemann hypothesis), and ``q̃, k̃`` are
    RoPE-rotated in complex form. See module docstring for ablation
    knobs.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block: int,
        freq_mode: str = "rope",
        amp_weight: bool = True,
        decay_mode: str = "none",
        sparse_eps2: float = 0.0,
        hermitian: bool = False,
        spectral_norm_o: bool = False,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        if freq_mode not in ("rope", "zeta_log"):
            raise ValueError(f"freq_mode must be 'rope' or 'zeta_log', got {freq_mode!r}")
        if decay_mode not in ("none", "bias", "mult"):
            raise ValueError(
                f"decay_mode must be 'none', 'bias', or 'mult', got {decay_mode!r}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        if self.d_head % 2 != 0:
            raise ValueError(f"d_head must be even, got {self.d_head}")
        self.K = self.d_head // 2
        self.block = block
        self.freq_mode = freq_mode
        self.amp_weight = amp_weight
        self.decay_mode = decay_mode
        self.hermitian = hermitian
        self.sparse_eps2 = float(sparse_eps2)

        # --- Frequencies ------------------------------------------------
        gamma = riemann_zeros(self.K)              # raw γ_k (buffer)
        gamma = gamma / gamma[0]                   # normalise: γ_0 -> 1
        self.register_buffer("gamma", gamma)

        if amp_weight:
            # w_k = 1/(1/2 + iγ_k) = (1/2 - iγ_k)/(1/4 + γ_k²)
            denom = 0.25 + gamma * gamma
            self.register_buffer("w_re", 0.5 / denom)
            self.register_buffer("w_im", -gamma / denom)
        else:
            self.register_buffer("w_re", torch.ones(self.K))
            self.register_buffer("w_im", torch.zeros(self.K))

        # --- Position phases --------------------------------------------
        pos = torch.arange(block, dtype=torch.float32)
        if freq_mode == "rope":
            # RoPE geometric frequencies (~3000× span) for multi-scale
            # resolution. γ_k is used only for amplitude, not frequency.
            k_idx = torch.arange(self.K, dtype=torch.float32)
            inv_freq = rope_base ** (-k_idx / self.K)     # (K,)
            phase = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
        else:  # "zeta_log"
            # Mellin-kernel phase θ(p,k) = γ_k log(1+p). Narrow (~5×) span.
            tau = torch.log1p(pos)
            phase = tau.unsqueeze(1) * gamma.unsqueeze(0)
        self.register_buffer("cos_p", phase.cos())    # (N, K)
        self.register_buffer("sin_p", phase.sin())

        # --- Decay factors ---------------------------------------------
        if decay_mode in ("bias", "mult"):
            log1p = torch.log1p(pos)
            # 0.5 * (log(1+j) - log(1+i))  is a pre-built (N, N) matrix.
            decay_mat = 0.5 * (log1p.view(1, -1) - log1p.view(-1, 1))
            self.register_buffer("log_decay", decay_mat)   # (N, N)
        else:
            self.register_buffer("log_decay", torch.zeros(block, block))

        # --- Causal mask -------------------------------------------------
        self.register_buffer(
            "tril", torch.tril(torch.ones(block, block, dtype=torch.bool)))

        # --- Projections -------------------------------------------------
        if hermitian:
            self.qk = nn.Linear(d_model, d_model, bias=False)
            self.q = self.qk
            self.k = self.qk
        else:
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)

        o_lin = nn.Linear(d_model, d_model, bias=False)
        if spectral_norm_o:
            self.o = nn.utils.parametrizations.spectral_norm(
                o_lin, n_power_iterations=5)
        else:
            self.o = o_lin

        # Cache constructor args for extend_to (recompute deterministic buffers).
        self._rope_base = float(rope_base)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extend_to(self, new_block: int) -> None:
        """Grow positional buffers for length-extrapolation eval.
        γ_k, w_k stay at training values; cos_p / sin_p / log_decay / tril
        are recomputed for `new_block` using the same rule as __init__."""
        cur = self.cos_p.shape[0]
        if new_block <= cur:
            return
        dev = self.cos_p.device
        pos = torch.arange(new_block, dtype=torch.float32, device=dev)
        if self.freq_mode == "rope":
            k_idx = torch.arange(self.K, dtype=torch.float32, device=dev)
            inv_freq = self._rope_base ** (-k_idx / self.K)
            phase = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
        else:
            tau = torch.log1p(pos)
            phase = tau.unsqueeze(1) * self.gamma.to(dev).unsqueeze(0)
        self.cos_p = phase.cos()
        self.sin_p = phase.sin()
        self.tril = torch.tril(
            torch.ones(new_block, new_block, dtype=torch.bool, device=dev))
        if self.decay_mode in ("bias", "mult"):
            log1p = torch.log1p(pos)
            self.log_decay = 0.5 * (log1p.view(1, -1) - log1p.view(-1, 1))
        else:
            self.log_decay = torch.zeros(new_block, new_block, device=dev)
        self.block = new_block

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, dh, K = self.n_heads, self.d_head, self.K

        q = self.q(x).view(B, N, H, dh)
        k = self.k(x).view(B, N, H, dh)
        v = self.v(x).view(B, N, H, dh)

        q_re = q[..., 0::2]
        q_im = q[..., 1::2]
        k_re = k[..., 0::2]
        k_im = k[..., 1::2]

        cos_b = self.cos_p[:N].view(1, N, 1, K)
        sin_b = self.sin_p[:N].view(1, N, 1, K)

        # RoPE-style rotation: (cos - i sin)(q_re + i q_im)
        qt_re = cos_b * q_re + sin_b * q_im
        qt_im = cos_b * q_im - sin_b * q_re
        kt_re = cos_b * k_re + sin_b * k_im
        kt_im = cos_b * k_im - sin_b * k_re

        # Apply ζ amplitude w_k (or identity if amp_weight=False).
        w_re = self.w_re.view(1, 1, 1, K)
        w_im = self.w_im.view(1, 1, 1, K)
        qhat_re = w_re * qt_re - w_im * qt_im
        qhat_im = w_re * qt_im + w_im * qt_re

        # Two real matmuls → Re(Σ_k w_k q̃_i · conj(k̃_j)).
        qhat_re = qhat_re.transpose(1, 2)
        qhat_im = qhat_im.transpose(1, 2)
        kt_re_t = kt_re.transpose(1, 2)
        kt_im_t = kt_im.transpose(1, 2)

        scores = (
            qhat_re @ kt_re_t.transpose(-1, -2)
            + qhat_im @ kt_im_t.transpose(-1, -2)
        )
        scores = scores / math.sqrt(dh)

        # Critical-line decay.
        if self.decay_mode == "bias":
            scores = scores + self.log_decay[:N, :N].view(1, 1, N, N)
        elif self.decay_mode == "mult":
            scores = scores * torch.exp(self.log_decay[:N, :N]).view(1, 1, N, N)

        # Self-adjoint projection (bidirectional only — will contaminate
        # causal scores with future-direction info).
        if self.hermitian:
            scores = 0.5 * (scores + scores.transpose(-1, -2))

        scores = scores.masked_fill(~self.tril[:N, :N], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        if self.sparse_eps2 > 0.0:
            attn = bootstrap_sparse(attn, self.sparse_eps2)

        v = v.transpose(1, 2)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.o(out)


# ---------------------------------------------------------------------------
# MRABlock
# ---------------------------------------------------------------------------


class MRABlock(nn.Module):
    """Pre-LN block wrapping `MellinRiemannAttention` + pluggable FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block: int,
        freq_mode: str = "rope",
        amp_weight: bool = True,
        decay_mode: str = "none",
        sparse_eps2: float = 0.0,
        hermitian: bool = False,
        spectral_norm_o: bool = False,
        ffn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MellinRiemannAttention(
            d_model, n_heads, block,
            freq_mode=freq_mode,
            amp_weight=amp_weight,
            decay_mode=decay_mode,
            sparse_eps2=sparse_eps2,
            hermitian=hermitian,
            spectral_norm_o=spectral_norm_o,
        )
        if ffn is None:
            ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model), nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
        self.ffn = ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


__all__ = [
    "BOOTSTRAP_EPS2",
    "bootstrap_sparse",
    "MellinRiemannAttention",
    "MRABlock",
]
