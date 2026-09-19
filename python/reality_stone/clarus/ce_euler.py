"""Euler-bitfield attention — {e, π, i, 1, 0} as minimal dimensionless grammar.

CE's docs/상수.md treats the Euler-identity constants {e, π, i, 1, 0}
as the minimum vocabulary that generates dimensionless cores. We apply
the same principle to positional/rotational attention encoding:

  - 5 "Euler basis" frequencies: B = {1, π, e, π·e, π/e}
  - Each attention head picks a BITFIELD b ∈ {0,1}^5 selecting the
    subset of frequencies it uses. The head's positional phase is

        theta_head(pos, k) = pos · sum_{j: b_j=1} B_j · 2^{-k / d_head}

    i.e. a sum of Euler-basis log-spaced frequencies.
  - Q and K are rotated by theta_head before scoring (RoPE-style).

The bitfield b is a learnable real-valued vector passed through a
sigmoid (soft bitfield) so gradients flow. At inference it thresholds
to {0, 1} for pure Euler-combinations.

Why this could help:
  1. Multi-base (e-based rotations don't align with π-based ones on any
     finite period) — unique long-range attention signatures.
  2. Compression: 5 bits per head vs O(d) free parameters.
  3. Theory grounding: CE's minimal grammar axiom.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


EULER_BASIS = (1.0, math.pi, math.e, math.pi * math.e, math.pi / math.e)
EULER_BASIS_NAMES = ("1", "pi", "e", "pi*e", "pi/e")


# ---------------------------------------------------------------------------
# CE first-principle constants for rotary base.
# ---------------------------------------------------------------------------
# Φ-relaxation coupling from CE physics layer (alpha_s at M_Z).
ALPHA_S = 0.11789
# CE effective dimension D_eff = 3 + δ where δ = sin²θ_W·(1-sin²θ_W) is the
# electroweak residual induced by Φ-relaxation. Numerically D_eff ≈ 3.178.
_SIN2 = 4.0 * ALPHA_S ** (4.0 / 3.0)
D_EFF = 3.0 + _SIN2 * (1.0 - _SIN2)


def ce_rotary_base(block: int, layer_idx: int = 0, n_layers: int = 1,
                   depth_aware: bool = False) -> float:
    """CE-faithful rotary base for π-phase encoding.

    base = π^(D_eff · depth_factor) · block

    Two first-principle factors:
      * π^D_eff : CE dimensional volume (replaces RoPE's empirical 10⁴).
      * × block : causal-cone scaling — keeps the slowest rotary mode
                  near-DC inside the context window for any block size.
                  This is the "step커지면 그만큼" correction: if the
                  sequence length grows N×, the base also grows N×, so
                  the longest period stays a fixed fraction of the window.
      * depth_factor (when depth_aware=True): per-layer RG-running of the
        effective dimension. depth_factor_ℓ = 1 + ℓ/(L-1) ∈ [1, 2].
        Compensates the cumulative phase added by stacking L layers.
    """
    if depth_aware and n_layers > 1:
        depth_factor = 1.0 + layer_idx / (n_layers - 1)
    else:
        depth_factor = 1.0
    return (math.pi ** (D_EFF * depth_factor)) * float(block)


def _rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply 2D rotation to adjacent dim pairs: [x0, x1] -> [x0·c - x1·s, x0·s + x1·c]."""
    # x: (..., n, d_head) with d_head even
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rx1
    out[..., 1::2] = rx2
    return out


# ---------------------------------------------------------------------------
# Shared SDPA helpers — used by EulerRotaryAttention, EulerCEAttention,
# and EulerCEMinimal so the FlashAttention-style Q-tiling + cached causal
# bias live in a single place.
# ---------------------------------------------------------------------------
Q_CHUNK_DEFAULT: int = 256
Q_CHUNK_THRESHOLD: int = 1024


def _causal_softmax_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Plain causal SDPA (no additive bias). FlashAttention on CUDA, tiled
    math path on CPU — either way the `(B, H, N, N)` scores tensor never
    materializes."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def _chunked_decay_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    xi_per_head: torch.Tensor,
    pos: torch.Tensor,
    n: int,
    *,
    e_gate: Optional[torch.Tensor] = None,
    q_chunk: int = Q_CHUNK_DEFAULT,
    q_chunk_threshold: int = Q_CHUNK_THRESHOLD,
) -> torch.Tensor:
    """SDPA with per-head ALiBi-style decay + causal mask, Q-tiled.

    Mask:  `m[h, i, j] = -|i-j| · e_gate[h] / xi[h]`  for `j ≤ i`,
                         `-inf`                       otherwise.

    Tiling cuts peak mask memory from `O(H·N·N)` to `O(H·Q_CHUNK·N)` —
    at N=4096, H=16, that is 32 MB instead of 1 GB. The per-row softmax
    is independent over the K axis, so the tiling is exact.
    """
    h = q.shape[1]
    rate_dev = q.device
    rate_dtype = q.dtype
    if e_gate is not None:
        bias_rate = (e_gate.to(rate_dtype) / xi_per_head.to(rate_dtype)).view(h, 1, 1)
    else:
        bias_rate = (1.0 / xi_per_head.to(rate_dtype)).view(h, 1, 1)
    cols = pos[:n].to(rate_dtype).view(1, 1, n)
    zero_s = torch.zeros((), dtype=rate_dtype, device=rate_dev)
    neg_inf_s = torch.full((), float("-inf"), dtype=rate_dtype, device=rate_dev)

    if n < q_chunk_threshold:
        rows = pos[:n].to(rate_dtype).view(1, n, 1)
        d_full = (rows - cols).abs()
        causal = torch.where(rows >= cols, zero_s, neg_inf_s)
        mask = (-d_full * bias_rate + causal).unsqueeze(0)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    out = torch.empty_like(q)
    for qs in range(0, n, q_chunk):
        qe = min(qs + q_chunk, n)
        rows = pos[qs:qe].to(rate_dtype).view(1, qe - qs, 1)
        d_chunk = (rows - cols).abs()
        causal_chunk = torch.where(rows >= cols, zero_s, neg_inf_s)
        mask_chunk = (-d_chunk * bias_rate + causal_chunk).unsqueeze(0)
        out[:, :, qs:qe] = F.scaled_dot_product_attention(
            q[:, :, qs:qe], k, v, attn_mask=mask_chunk
        )
    return out


class EulerRotaryAttention(nn.Module):
    """Multi-head attention with Euler-bitfield rotary positional encoding.

    Args:
        d_model, n_heads, block: as usual
        softmax_bitfield: if True, soft bitfield via sigmoid(logits).
            Otherwise hard 0/1 bitfield from init template.
        init_bits: initial bitfield per head. Shape (n_heads, 5).
            Default: each head uses all 5 bases.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block: int,
        softmax_bitfield: bool = True,
        init_bits: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert self.d_head % 2 == 0, "d_head must be even for rotary"

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("tril",
                             torch.tril(torch.ones(block, block, dtype=torch.bool)))

        basis = torch.tensor(EULER_BASIS, dtype=torch.float32)  # (5,)
        self.register_buffer("euler_basis", basis)

        if init_bits is None:
            init_bits = torch.zeros(n_heads, 5)  # start at 0 -> sigmoid 0.5
        if softmax_bitfield:
            self.bit_logits = nn.Parameter(init_bits)
        else:
            self.register_buffer("bit_logits", init_bits)

        # log-space-frequency exponents: k = 0, 2, 4, ..., d_head-2  (per pair)
        k = torch.arange(0, self.d_head, 2, dtype=torch.float32) / self.d_head
        self.register_buffer("inv_freq", 2.0 ** (-k))  # (d_head/2,)

        # positions
        self.register_buffer("pos", torch.arange(block, dtype=torch.float32))

    def bitfield(self) -> torch.Tensor:
        """(n_heads, 5) soft bitfield in [0, 1]."""
        return torch.sigmoid(self.bit_logits)

    def head_freq_scalars(self) -> torch.Tensor:
        """Per-head frequency scalar (weighted sum over Euler basis)."""
        b = self.bitfield()  # (n_heads, 5)
        return torch.matmul(b, self.euler_basis)  # (n_heads,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # (b, n, h, d_head)
        q = q.transpose(1, 2)  # (b, h, n, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Build head-specific rotation angles: theta[h, n, k/2] = pos[n] * freq[h] * inv_freq[k]
        freqs = self.head_freq_scalars()  # (h,)
        # theta = pos[n] * freq[h] * inv_freq[k]
        theta = self.pos[:n].view(1, 1, n, 1) * freqs.view(1, self.n_heads, 1, 1) \
                * self.inv_freq.view(1, 1, 1, -1)  # (1, h, n, d_head/2)
        cos = theta.cos()
        sin = theta.sin()

        q = _rotate_pairs(q, cos, sin)
        k = _rotate_pairs(k, cos, sin)

        # SDPA with causal mask — no `(B, H, N, N)` scores tensor, no
        # explicit softmax. Output is bit-identical to the prior manual
        # path within float precision.
        out = _causal_softmax_sdpa(q, k, v)
        out = out.transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.o(out)


class EulerAttnBlock(nn.Module):
    """Transformer block using Euler rotary attention."""

    def __init__(self, d_model: int, n_heads: int, block: int,
                 softmax_bitfield: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = EulerRotaryAttention(d_model, n_heads, block,
                                         softmax_bitfield=softmax_bitfield)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


__all__ = [
    "EULER_BASIS",
    "EULER_BASIS_NAMES",
    "EulerRotaryAttention",
    "EulerAttnBlock",
    "EulerCEAttention",
    "EulerCEBlock",
    "EulerCEMinimal",
    "EulerCEMinimalBlock",
    "RecursiveEulerCEBlock",
    "fixed_point_loss",
    "head_types_from_spec",
]


# ---------------------------------------------------------------------------
# EulerCEAttention — theory-correct assignment of {e, π, i, 1, 0}
# ---------------------------------------------------------------------------
#
# Per docs/경로적분.md (lines 51-67):
#   e   -> survival / decay:        S(D) = e^{-D}
#   π   -> periodic normalization:  α_total = 1 / (2π)
#   i   -> path-integral phase:     Z = ∫ Dφ e^{iS/ℏ}
#   1   -> normalized complete state
#   0   -> zero / branch selection
#
# The attention kernel is therefore
#
#   A_ij  =  softmax_j ( Q_i · R_π(i-j) · K_j / √d )  ·  e^{-|i-j|/ξ_e}
#                       └─── π-phase rotary ───┘        └── e decay ──┘
#
#     R_π(Δ): RoPE-style rotation with fundamental period π (not 10^4).
#     ξ_e:    learnable correlation length (decay base e).
#
# The "bitfield" selects which of {π-phase, e-decay, 1-bypass} are active
# per head. A head with bit=0 for π uses identity rotation (no RoPE);
# bit=0 for e means no distance decay. This exposes the 5-constant
# minimum vocabulary as an interpretable head-type switch.


class EulerCEAttention(nn.Module):
    """Theory-correct Euler attention: π-phase + e-decay + {1,0} gates.

    Args:
        d_model, n_heads, block: standard
        xi_init: initial correlation length for the e-decay term (in
            positions). Larger = weaker decay. Default block/2.
        learnable_gates: if True, the two gates (pi_gate, e_gate) are
            per-head learnable sigmoids; if False they are frozen at 1.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block: int,
        xi_init: Optional[float] = None,
        learnable_gates: bool = True,
        layer_idx: int = 0,
        n_layers: int = 1,
        depth_aware_freq: bool = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert self.d_head % 2 == 0, "d_head must be even"

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("tril",
                             torch.tril(torch.ones(block, block, dtype=torch.bool)))

        # π-phase rotary base — see ce_rotary_base() for derivation.
        base = ce_rotary_base(block, layer_idx=layer_idx,
                              n_layers=n_layers,
                              depth_aware=depth_aware_freq)
        k = torch.arange(0, self.d_head, 2, dtype=torch.float32) / self.d_head
        self.register_buffer("pi_inv_freq", base ** (-k))
        self.register_buffer("pos", torch.arange(block, dtype=torch.float32))

        # e-decay: log xi so xi > 0 strictly. Initialize xi = block/8 so the
        # decay actually bites in [0, block]: e^{-block/xi} = e^{-8} ≈ 3e-4.
        # block/2 made the decay numerically negligible.
        if xi_init is None:
            xi_init = block / 8.0
        self.log_xi = nn.Parameter(torch.full((n_heads,),
                                              math.log(xi_init), dtype=torch.float32))

        # Per-head gates. sigmoid(1.0) ≈ 0.73 starts mild but learnable.
        if learnable_gates:
            self.pi_gate_logit = nn.Parameter(torch.full((n_heads,), 1.0))
            self.e_gate_logit = nn.Parameter(torch.full((n_heads,), 1.0))
        else:
            self.register_buffer("pi_gate_logit", torch.full((n_heads,), 1e4))
            self.register_buffer("e_gate_logit", torch.full((n_heads,), 1e4))

        # Precompute |i-j| distance matrix (non-negative, upper-tri set by mask)
        d_mat = (torch.arange(block).unsqueeze(1) - torch.arange(block).unsqueeze(0)).abs().float()
        self.register_buffer("d_mat", d_mat)

    @torch.no_grad()
    def extend_to(self, new_block: int) -> None:
        """Grow positional / distance buffers for length-extrapolation eval.
        Learnable parameters (log_xi, gates, qkv, o) are unchanged. The
        rotary base (pi_inv_freq) is intentionally kept at its training-time
        value — that is the *block-aware* design point of EulerCE."""
        cur = self.pos.shape[0]
        if new_block <= cur:
            return
        dev = self.pos.device
        self.pos = torch.arange(new_block, dtype=torch.float32, device=dev)
        self.tril = torch.tril(
            torch.ones(new_block, new_block, dtype=torch.bool, device=dev))
        d = (torch.arange(new_block).unsqueeze(1)
             - torch.arange(new_block).unsqueeze(0)).abs().float().to(dev)
        self.d_mat = d

    def _rotate(self, x, cos, sin):
        x1 = x[..., 0::2]; x2 = x[..., 1::2]
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        out = torch.empty_like(x)
        out[..., 0::2] = rx1
        out[..., 1::2] = rx2
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)

        pi_g = torch.sigmoid(self.pi_gate_logit)      # (h,)
        e_g = torch.sigmoid(self.e_gate_logit)        # (h,)

        # π-phase rotation (per-head amplitude scaled by pi_g).
        theta = self.pos[:n].view(1, 1, n, 1) * self.pi_inv_freq.view(1, 1, 1, -1)
        theta = theta * pi_g.view(1, self.n_heads, 1, 1)
        cos = theta.cos(); sin = theta.sin()
        q_rot = self._rotate(q, cos, sin)
        k_rot = self._rotate(k, cos, sin)

        # SDPA with per-head ALiBi decay, Q-tiled (FlashAttention-style).
        # Algebraically identical to `softmax(QK/√d - e_g·|i-j|/ξ + causal)·V`
        # but never materializes the `(B, H, N, N)` scores tensor.
        xi = torch.exp(self.log_xi)                   # (h,)
        out = _chunked_decay_sdpa(
            q_rot, k_rot, v, xi, self.pos, n, e_gate=e_g
        )
        out = out.transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.o(out)


class EulerCEBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block: int,
                 learnable_gates: bool = True,
                 layer_idx: int = 0, n_layers: int = 1,
                 depth_aware_freq: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = EulerCEAttention(d_model, n_heads, block,
                                     learnable_gates=learnable_gates,
                                     layer_idx=layer_idx, n_layers=n_layers,
                                     depth_aware_freq=depth_aware_freq)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# RecursiveEulerCEBlock — self-referential fixed-point iteration (ClarusCell)
# ---------------------------------------------------------------------------
#
# CE bootstrap equation:   epsilon^2 = exp[-(1 - epsilon^2) * D_eff]
# This is a fixed-point equation x* = F(x*). A CE-faithful transformer
# block should therefore be allowed to apply itself repeatedly to its
# own output until convergence, rather than being a one-shot function.
#
# Two semantics offered:
#
# 1. FIXED DEPTH RECURSION (``max_iters=k``, ``tol=None``):
#    h_0 = x;  h_{t+1} = F(h_t);  out = h_k
#    "Universal Transformer" style, weights shared across depth.
#
# 2. WHILE-LOOP RECURSION (``tol>0``):
#    halt when ||h_{t+1} - h_t|| / ||h_t|| < tol  OR  t == max_iters.
#    The halting depth is recorded in ``.last_depths`` for analysis.
#    Non-differentiable halt; backprop flows through the final path.
#
# The optional self-consistency loss
#
#    L_fp = || F(F(h*)) - F(h*) ||^2
#
# pulls the output h* = F(x) toward being a true fixed point.


class RecursiveEulerCEBlock(nn.Module):
    """Self-referential transformer block — ClarusCell as while-loop.

    Args:
        d_model, n_heads, block: standard
        max_iters: maximum number of self-applications (>=1)
        tol: if not None, halt when relative change is below this
             threshold. If None, always run ``max_iters`` iterations.
        learnable_gates: forwarded to EulerCEAttention
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block: int,
        max_iters: int = 1,
        tol: Optional[float] = None,
        learnable_gates: bool = True,
        layer_idx: int = 0,
        n_layers: int = 1,
        depth_aware_freq: bool = False,
        depth_aware_iters: bool = False,
    ) -> None:
        super().__init__()
        self.core = EulerCEBlock(d_model, n_heads, block,
                                 learnable_gates=learnable_gates,
                                 layer_idx=layer_idx, n_layers=n_layers,
                                 depth_aware_freq=depth_aware_freq)
        # depth_aware_iters: deeper layers get more self-iterations to
        # compensate accumulated representational complexity. Schedule:
        #   iters_ℓ = max_iters + ℓ
        # so a 4-layer stack with max_iters=1 yields {1,2,3,4} effective.
        if depth_aware_iters:
            self.max_iters = max_iters + layer_idx
        else:
            self.max_iters = max_iters
        self.tol = tol
        self.last_depths: Optional[torch.Tensor] = None

    def _step(self, h: torch.Tensor) -> torch.Tensor:
        return self.core(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        depths = torch.full((x.shape[0],), self.max_iters,
                            dtype=torch.long, device=x.device)

        if self.tol is None:
            # Fixed-depth recursion (differentiable through all steps).
            for _ in range(self.max_iters):
                h = self._step(h)
            self.last_depths = depths
            return h

        # While-loop recursion with halting on tolerance.
        for t in range(self.max_iters):
            h_new = self._step(h)
            # per-example relative change (detached — halt decision is not diff)
            with torch.no_grad():
                num = (h_new - h).flatten(1).norm(dim=-1)
                den = h.flatten(1).norm(dim=-1).clamp_min(1e-8)
                rel = num / den
                # mark examples that have newly halted
                halted = rel < self.tol
                not_yet = depths == self.max_iters
                just_halted = halted & not_yet
                depths = torch.where(just_halted,
                                     torch.full_like(depths, t + 1),
                                     depths)
                if halted.all():
                    h = h_new
                    break
            h = h_new
        self.last_depths = depths
        return h


def fixed_point_loss(block: RecursiveEulerCEBlock, h: torch.Tensor,
                     scale: float = 1.0) -> torch.Tensor:
    """||F(F(h)) - F(h)||^2 averaged over batch and positions.

    Pulls h toward being a fixed point of ``block.core``. Use as a
    regularizer added to cross-entropy.
    """
    with torch.no_grad():
        fh = block.core(h)
    ffh = block.core(fh)
    diff = (ffh - fh).flatten(1)
    return scale * diff.pow(2).mean()


# ---------------------------------------------------------------------------
# EulerCEMinimal — 2-bit head-type taxonomy
# ---------------------------------------------------------------------------
#
# Operational reduction of {e, π, i, 1, 0}:
#   {π, i}  → rotation generator   (always paired as e^{iπt} = (cos, sin))
#   {e}     → exponential decay
#   {0, 1}  → on/off gate values   (1 bit each, by definition)
#
# → 2 functionally distinct axes (rotation, decay) × 2 gate values
#   = 2² = 4 head-types, each encoded by a 2-bit string (pi_bit, e_bit):
#
#       (pi, e)   head-type       canonical literature analogue
#       --------  --------------  -------------------------------
#       (0, 0)    identity        NoPE      (Kazemnejad 2023)
#       (0, 1)    decay only      ALiBi     (Press 2022)
#       (1, 0)    rotation only   RoPE      (Su 2021)
#       (1, 1)    rotation+decay  xPos      (Sun 2023) / EulerCE
#
# Per-head continuous parameters (only meaningful when bit is on):
#       pi_base : rotary base (RoPE-style geometric, defaults to 10000)
#       xi_h    : decay length (per-head learnable, default block/8)
#
# Empirical finding (`docs/8_리만/mra_paper.md` § 7.7, length extrap. ablation):
#   Head-type (1, 0) — pure rotation — is the only Tier 2 (catastrophic
#   length-OOD). The other three head-types are Tier 1 (extrapolate).
#   So among the 2² = 4 types, 3 are operationally useful → log₂ 3 ≈ 1.58
#   bits is the effective head-type capacity.


_HEAD_TYPE_NAMES = ("nope", "alibi", "rope", "xpos")  # indexed by 2*pi + e


def head_types_from_spec(spec, n_heads: int) -> torch.Tensor:
    """Convert a head-type spec into a (n_heads,) int tensor in {0,1,2,3}.

    Acceptable spec forms:
      * int in [0, 3]           — uniform (all heads same type)
      * list/tuple of length n  — per-head type values
      * str  in {"nope", "alibi", "rope", "xpos"} — uniform name
      * str  "mix" — alternating alibi / xpos
      * str  "all" — round-robin {nope, alibi, rope, xpos}
    """
    name_to_idx = {n: i for i, n in enumerate(_HEAD_TYPE_NAMES)}
    if isinstance(spec, str):
        if spec == "mix":
            ts = [(1 if h % 2 == 0 else 3) for h in range(n_heads)]
            return torch.tensor(ts, dtype=torch.long)
        if spec == "all":
            ts = [h % 4 for h in range(n_heads)]
            return torch.tensor(ts, dtype=torch.long)
        if spec in name_to_idx:
            return torch.full((n_heads,), name_to_idx[spec], dtype=torch.long)
        raise ValueError(f"unknown head-type spec: {spec!r}")
    if isinstance(spec, int):
        if not 0 <= spec <= 3:
            raise ValueError(f"head-type int must be in [0, 3], got {spec}")
        return torch.full((n_heads,), spec, dtype=torch.long)
    spec = torch.as_tensor(spec, dtype=torch.long)
    if spec.shape != (n_heads,):
        raise ValueError(f"head-type tensor must have shape ({n_heads},), got {tuple(spec.shape)}")
    if (spec < 0).any() or (spec > 3).any():
        raise ValueError("head-type values must be in [0, 3]")
    return spec


class EulerCEMinimal(nn.Module):
    """2-bit minimal Euler-CE attention.

    Each head commits to one of four operational types via a 2-bit
    spec (pi_bit, e_bit). The continuous parameters `xi_h` (decay
    length, per-head) and the rotary base are the only learnable
    positional state — head-type itself is an axiomatic design choice,
    not learned.

    Args:
        d_model, n_heads, block: standard.
        head_types: spec for per-head types. See `head_types_from_spec`.
            Default "alibi" (all decay-only, the strongest single tier-1
            choice from the length-extrap ablation).
        rope_base: base for the RoPE-style geometric frequencies used by
            heads with pi_bit = 1. Defaults to 10000 (RoFormer).
        xi_init: initial decay length for heads with e_bit = 1.
            Defaults to block/8 (the EulerCE original).
        learnable_xi: if False, freeze xi at its init value.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block: int,
        head_types: object = "alibi",
        rope_base: float = 10000.0,
        xi_init: Optional[float] = None,
        learnable_xi: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        if self.d_head % 2 != 0:
            raise ValueError(f"d_head must be even (got {self.d_head})")
        self.block = block
        self.rope_base = float(rope_base)

        # 2-bit head-type assignment (axiom — buffer, not learned).
        # High bit = rotation (π), low bit = decay (e). Extracted with
        # bitwise shift/mask — semantically the 2-bit field the name
        # implies, and cheaper than the old `// 2` / `% 2` form.
        types = head_types_from_spec(head_types, n_heads)            # (h,)
        pi_bits = ((types >> 1) & 1).float()
        e_bits = (types & 1).float()
        self.register_buffer("head_types", types)
        self.register_buffer("pi_bits", pi_bits)
        self.register_buffer("e_bits", e_bits)
        # Fast-path detector: when all heads share the same type we can
        # bypass the per-head gating and dispatch to PyTorch SDPA
        # (FlashAttention / Memory-Efficient backend).
        self._uniform_type = int(types[0].item()) if (types == types[0]).all() else -1

        # Packed bitmask form of the head-type assignment. For n_heads
        # ≤ 64 this collapses the (H,) float buffers (pi_bits, e_bits =
        # 8·H bytes) to two Python ints (≤ 16 bytes total). Python's
        # arbitrary-precision int transparently handles larger widths.
        pi_mask_int, e_mask_int = 0, 0
        for h, t in enumerate(types.tolist()):
            pi_mask_int |= ((t >> 1) & 1) << h
            e_mask_int |= (t & 1) << h
        self._pi_mask = pi_mask_int
        self._e_mask = e_mask_int

        # Pre-bucket heads by 2-bit type so the mixed path dispatches
        # SDPA once per present bucket. This eliminates:
        #   * `cos`/`sin` materialization on rotation-off heads,
        #   * `decay_bias` materialization on decay-off heads,
        #   * the `(b, H, n, n)` scores tensor (SDPA tiles internally),
        #   * explicit `softmax` (FlashAttention / mem-efficient path).
        present: list[int] = []
        bucket_heads: list[torch.Tensor] = []
        for t in range(4):
            idx = (types == t).nonzero(as_tuple=True)[0].long().contiguous()
            if idx.numel() > 0:
                present.append(t)
                self.register_buffer(f"_bucket_{t}_idx", idx, persistent=False)
                bucket_heads.append(idx)
        self._present_buckets: tuple[int, ...] = tuple(present)
        concat_idx = (
            torch.cat(bucket_heads, dim=0)
            if bucket_heads
            else torch.arange(n_heads, dtype=torch.long)
        )
        inv_perm = torch.empty(n_heads, dtype=torch.long)
        inv_perm[concat_idx] = torch.arange(n_heads, dtype=torch.long)
        self.register_buffer("_bucket_inv_perm", inv_perm.contiguous(), persistent=False)
        self._bucket_is_identity: bool = bool(
            (concat_idx == torch.arange(n_heads, dtype=torch.long)).all().item()
        )

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        # Position / distance buffers.
        self.register_buffer(
            "tril", torch.tril(torch.ones(block, block, dtype=torch.bool)))
        self.register_buffer(
            "pos", torch.arange(block, dtype=torch.float32))
        d_mat = (torch.arange(block).unsqueeze(1)
                 - torch.arange(block).unsqueeze(0)).abs().float()
        self.register_buffer("d_mat", d_mat)

        # RoPE-style frequencies for heads with rotation bit on.
        k = torch.arange(0, self.d_head, 2, dtype=torch.float32) / self.d_head
        self.register_buffer("inv_freq", self.rope_base ** (-k))   # (d_head/2,)

        # Per-head decay length.
        if xi_init is None:
            xi_init = block / 8.0
        if xi_init <= 0.0:
            raise ValueError("xi_init must be positive")
        log_xi = torch.full((n_heads,), math.log(xi_init), dtype=torch.float32)
        if learnable_xi:
            self.log_xi = nn.Parameter(log_xi)
        else:
            self.register_buffer("log_xi", log_xi)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extend_to(self, new_block: int) -> None:
        """Grow positional / distance buffers for length-extrap eval.
        Learnable parameters (qkv, o, log_xi) are unchanged."""
        cur = self.pos.shape[0]
        if new_block <= cur:
            return
        dev = self.pos.device
        self.pos = torch.arange(new_block, dtype=torch.float32, device=dev)
        self.tril = torch.tril(
            torch.ones(new_block, new_block, dtype=torch.bool, device=dev))
        self.d_mat = (torch.arange(new_block).unsqueeze(1)
                      - torch.arange(new_block).unsqueeze(0)).abs().float().to(dev)

    # ------------------------------------------------------------------
    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        out = torch.empty_like(x)
        out[..., 0::2] = rx1
        out[..., 1::2] = rx2
        return out

    # ------------------------------------------------------------------
    # Forward dispatch:
    #   uniform head-type → SDPA fast path (Flash / Memory-Efficient)
    #   mixed head-type   → per-head-gated reference path (slower but generic)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        H = self.n_heads
        qkv = self.qkv(x).view(b, n, 3, H, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)             # (b, H, n, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self._uniform_type >= 0:
            out = self._forward_uniform(q, k, v, n, H)
        else:
            out = self._forward_mixed(q, k, v, n, H)
        out = out.transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.o(out)

    # ------------------------------------------------------------------
    def _forward_uniform(self, q, k, v, n, H):
        """Fast path when all heads share the same 2-bit type.

        Dispatches to torch.nn.functional.scaled_dot_product_attention,
        which selects FlashAttention (no attn_mask) or Memory-Efficient
        Attention (with attn_mask) automatically.
        """
        ht = self._uniform_type
        rotate = (ht & 0b10) != 0
        decay = (ht & 0b01) != 0

        if rotate:
            theta = self.pos[:n].view(1, 1, n, 1) * self.inv_freq.view(1, 1, 1, -1)
            cos = theta.cos()
            sin = theta.sin()
            q = self._rotate(q, cos, sin)
            k = self._rotate(k, cos, sin)

        if not decay:
            return _causal_softmax_sdpa(q, k, v)

        # Decay path: Q-tiled to keep attn_mask peak at O(H·Q_CHUNK·N).
        xi = torch.exp(self.log_xi)                                  # (H,)
        return _chunked_decay_sdpa(q, k, v, xi, self.pos, n)

    # ------------------------------------------------------------------
    def _forward_mixed(self, q, k, v, n, H):
        """Bucketed zero-waste path for mixed head-types.

        Heads are grouped by 2-bit type at init. Each present bucket
        dispatches to `scaled_dot_product_attention` once with exactly
        the PE it needs (rotation for `pi_bit=1`, additive distance
        mask for `e_bit=1`). Compared to the prior scalar-gated path
        this drops `cos`/`sin` on rotation-off heads, the `(1, H, n, n)`
        decay bias on decay-off heads, the full `(b, H, n, n)` scores
        tensor, and the explicit softmax — SDPA tiles internally and
        picks FlashAttention when the mask allows it. Output is placed
        directly into the final head slot via `index_copy_`, avoiding
        a `cat` + permutation round-trip.
        """
        b = q.shape[0]
        d_head = q.shape[-1]
        out = torch.empty(b, H, n, d_head, dtype=q.dtype, device=q.device)

        theta_cos: Optional[torch.Tensor] = None
        theta_sin: Optional[torch.Tensor] = None
        causal_view: Optional[torch.Tensor] = None

        for t in self._present_buckets:
            idx = getattr(self, f"_bucket_{t}_idx")
            h_t = idx.numel()
            q_t = q.index_select(1, idx)
            k_t = k.index_select(1, idx)
            v_t = v.index_select(1, idx)

            if (t >> 1) & 1:  # rotation bit
                if theta_cos is None:
                    theta = self.pos[:n].view(1, 1, n, 1) * self.inv_freq.view(1, 1, 1, -1)
                    theta_cos = theta.cos()
                    theta_sin = theta.sin()
                q_t = self._rotate(q_t, theta_cos, theta_sin)
                k_t = self._rotate(k_t, theta_cos, theta_sin)

            if t & 1:  # decay bit
                xi = torch.exp(self.log_xi.index_select(0, idx))        # (h_t,)
                out_t = _chunked_decay_sdpa(q_t, k_t, v_t, xi, self.pos, n)
            else:
                out_t = _causal_softmax_sdpa(q_t, k_t, v_t)

            out.index_copy_(1, idx, out_t)

        return out



class EulerCEMinimalBlock(nn.Module):
    """Pre-LN block wrapping `EulerCEMinimal` + standard 4× FFN."""

    def __init__(self, d_model: int, n_heads: int, block: int,
                 head_types: object = "alibi",
                 rope_base: float = 10000.0,
                 xi_init: Optional[float] = None,
                 learnable_xi: bool = True) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = EulerCEMinimal(
            d_model, n_heads, block,
            head_types=head_types,
            rope_base=rope_base,
            xi_init=xi_init,
            learnable_xi=learnable_xi,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
