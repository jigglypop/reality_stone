"""Riemann-surface positional encoding for attention.

Engineering axiom: the Riemann Hypothesis is true. The non-trivial
zeros lie on the critical line s = 1/2 + i γ_n, and {γ_n} is GUE-
distributed (Montgomery-Dyson). The first 100 γ_n are hardcoded;
n > 100 uses the Riemann–von Mangoldt asymptotic γ_n ≈ 2π n / log n.

`RiemannRotaryAttention` implements the multi-sheet positional
encoding described in `docs/8_리만/riemann_pe_spec.md`:

    τ_p           = log(1 + p)                       # log-time lift
    θ(p, k)       = γ_k · τ_p                        # phase
    σ(p, k)       = floor(θ(p, k) / 2π)              # Riemann sheet
    rotation      = ((cos θ, -sin θ), (sin θ, cos θ))   on (q_{2k}, q_{2k+1})
    sheet_bias_ij = -λ_σ · mean_k |σ(i, k) - σ(j, k)|

Backend dispatch:
    backend="auto" picks cuda / rust / torch from the input device.

`riemann_zero_init` provides Design (4) — FFN key spacing seeded by
the Riemann-zero gap pattern.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# First 100 imaginary parts of the non-trivial Riemann zeta zeros.
# Source: Titchmarsh "The Theory of the Riemann Zeta-Function" Appendix,
# cross-checked with Odlyzko's published tables. 9 significant figures
# (sufficient for float32 attention).
RIEMANN_ZEROS_IM: tuple[float, ...] = (
    14.134725142,  21.022039639,  25.010857580,  30.424876126,  32.935061588,
    37.586178159,  40.918719012,  43.327073281,  48.005150881,  49.773832478,
    52.970321478,  56.446247697,  59.347044003,  60.831778525,  65.112544048,
    67.079810529,  69.546401711,  72.067157674,  75.704690699,  77.144840069,
    79.337375020,  82.910380854,  84.735492981,  87.425274613,  88.809111208,
    92.491899271,  94.651344041,  95.870634228,  98.831194218, 101.317851006,
    103.725538040, 105.446623052, 107.168611184, 111.029535543, 111.874659177,
    114.320220915, 116.226680321, 118.790782866, 121.370125002, 122.946829294,
    124.256818554, 127.516683880, 129.578704200, 131.087688531, 133.497737203,
    134.756509753, 138.116042055, 139.736208952, 141.123707404, 143.111845808,
    146.000982487, 147.422765343, 150.053520421, 150.925257612, 153.024693811,
    156.112909294, 157.597591818, 158.849988171, 161.188964138, 163.030709687,
    165.537069188, 167.184439978, 169.094515416, 169.911976479, 173.411536520,
    174.754191523, 176.441434298, 178.377407776, 179.916484020, 182.207078484,
    184.874467848, 185.598783678, 187.228922584, 189.416158656, 192.026656361,
    193.079726604, 195.265396680, 196.876481841, 198.015309676, 201.264751944,
    202.493594514, 204.189671803, 205.394697202, 207.906258888, 209.576509717,
    211.690862595, 213.347919360, 214.547044783, 216.169538508, 219.067596349,
    220.714918839, 221.430705555, 224.007000255, 224.983324670, 227.421444280,
    229.337413306, 231.250188700, 231.987235253, 233.693404179, 236.524229666,
)


_TAU = 2.0 * math.pi


def riemann_zeros(n: int) -> torch.Tensor:
    """Return the first n imaginary parts of non-trivial ζ zeros.

    For n > 100 (beyond the hardcoded table) uses the local-density
    extrapolation:  γ_{k+1} ≈ γ_k + 2π / log(γ_k / 2π).
    This follows from the Riemann–von Mangoldt counting formula
    N(T) ~ (T/2π)·(log(T/2π) - 1), differentiated to give the average
    spacing.  Guarantees monotonicity and joins smoothly to γ_100.
    """
    if n <= len(RIEMANN_ZEROS_IM):
        return torch.tensor(RIEMANN_ZEROS_IM[:n], dtype=torch.float32)
    vals = list(RIEMANN_ZEROS_IM)
    last = vals[-1]
    for _ in range(len(RIEMANN_ZEROS_IM), n):
        gap = _TAU / math.log(max(last / _TAU, math.e))
        last = last + gap
        vals.append(last)
    return torch.tensor(vals, dtype=torch.float32)


# --- backend hooks (filled at import time if available) ---------------------

try:
    from . import _rust as _rust_mod  # type: ignore[attr-defined]

    _rust_riemann_fwd = getattr(_rust_mod, "nn_ce_riemann_fwd", None)
    _rust_riemann_fwd_cuda = getattr(_rust_mod, "nn_ce_riemann_fwd_cuda", None)
    _rust_riemann_fwd_cuda_devptr = getattr(
        _rust_mod, "nn_ce_riemann_fwd_cuda_devptr", None
    )
except ImportError:
    _rust_mod = None
    _rust_riemann_fwd = None
    _rust_riemann_fwd_cuda = None
    _rust_riemann_fwd_cuda_devptr = None

_HAS_RUST = _rust_riemann_fwd is not None
_HAS_CUDA = _rust_riemann_fwd_cuda_devptr is not None


def has_rust_riemann() -> bool:
    return bool(_HAS_RUST)


def has_cuda_riemann() -> bool:
    return bool(_HAS_CUDA)


# --- core PyTorch reference impl --------------------------------------------


def _build_phase_and_sheet(
    pos: torch.Tensor,
    gamma: torch.Tensor,
    log_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute θ(p,k) and σ(p,k) for all (head, pos, pair).

    pos:        (n,)            positions 0..n-1
    gamma:      (n_pairs,)      Riemann-zero frequencies (already normalized)
    log_scale:  (h,)            per-head log "speed of light"

    Returns:
        theta: (1, h, n, n_pairs)
        sheet: (1, h, n, n_pairs)  int32
    """
    tau = torch.log1p(pos)                        # (n,)
    scale = torch.exp(log_scale)                  # (h,)
    # theta[h, n, k] = scale[h] * gamma[k] * tau[n]
    theta = (
        tau.view(1, 1, -1, 1)
        * gamma.view(1, 1, 1, -1)
        * scale.view(1, -1, 1, 1)
    )
    sheet = torch.floor(theta / _TAU).to(torch.int32)
    return theta, sheet


def _rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply 2D rotation to adjacent dim pairs of x (..., d_head)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rx1
    out[..., 1::2] = rx2
    return out


def _sheet_bias(sheet: torch.Tensor, lambda_sigma: torch.Tensor) -> torch.Tensor:
    """Mean cross-pair |σ_i - σ_j| × (-λ_σ).

    sheet:        (1, h, n, n_pairs)  int32
    lambda_sigma: (h,)
    Returns:      (1, h, n, n)
    """
    s = sheet.float()                             # (1, h, n, K)
    diff = s.unsqueeze(-2) - s.unsqueeze(-3)      # (1, h, n, n, K)
    mean_abs = diff.abs().mean(dim=-1)            # (1, h, n, n)
    return -lambda_sigma.view(1, -1, 1, 1) * mean_abs


def _attention_torch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    sheet_bias: torch.Tensor,
    causal_mask: torch.Tensor,
    d_head: int,
) -> torch.Tensor:
    q_rot = _rotate_pairs(q, cos, sin)
    k_rot = _rotate_pairs(k, cos, sin)
    scores = (q_rot @ k_rot.transpose(-1, -2)) / math.sqrt(d_head)
    scores = scores + sheet_bias
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return attn @ v


class RiemannRotaryAttention(nn.Module):
    """Riemann-surface positional encoding for multi-head attention.

    Args:
        d_model, n_heads, block: standard
        normalize_gamma: divide γ_n by γ_1 so the slowest mode has
            unit angular speed at log-time = 1.
        learnable_scale: per-head log "speed of light" multiplier.
        sheet_init: initial value for log λ_σ (log-space). Default
            log(0.0) → λ_σ = 0 (sheet bias inert at init), so the
            module gracefully starts equivalent to plain RoPE-on-log-time.
        backend: "auto" | "torch" | "rust" | "cuda".
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block: int,
        normalize_gamma: bool = True,
        learnable_scale: bool = True,
        sheet_init: float = -6.0,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        if self.d_head % 2 != 0:
            raise ValueError(f"d_head must be even, got {self.d_head}")
        n_pairs = self.d_head // 2
        self.n_pairs = n_pairs
        self.block = block
        self.backend_pref = backend

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block, block, dtype=torch.bool)),
        )

        gamma = riemann_zeros(n_pairs)
        if normalize_gamma:
            gamma = gamma / gamma[0]
        self.register_buffer("gamma", gamma)
        self.register_buffer("pos", torch.arange(block, dtype=torch.float32))

        if learnable_scale:
            self.log_scale = nn.Parameter(torch.zeros(n_heads))
        else:
            self.register_buffer("log_scale", torch.zeros(n_heads))

        # λ_σ = exp(log_lambda_sigma); init at sheet_init so that
        # exp(-6) ≈ 2.5e-3 → near-zero but learnable upward.
        self.log_lambda_sigma = nn.Parameter(
            torch.full((n_heads,), float(sheet_init))
        )

    # ------------------------------------------------------------------
    # backend selection
    # ------------------------------------------------------------------
    def _resolve_backend(self, x: torch.Tensor) -> str:
        pref = self.backend_pref
        if pref == "torch":
            return "torch"
        if pref == "rust":
            if not _HAS_RUST or x.is_cuda:
                return "torch"
            return "rust"
        if pref == "cuda":
            if not _HAS_CUDA or not x.is_cuda:
                return "torch"
            return "cuda"
        # auto
        if x.is_cuda and _HAS_CUDA:
            return "cuda"
        if not x.is_cuda and _HAS_RUST and not self.training:
            # rust path is forward-only; only use when no autograd.
            return "rust"
        return "torch"

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)                 # (b, h, n, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        backend = self._resolve_backend(x)

        theta, sheet = _build_phase_and_sheet(
            self.pos[:n], self.gamma, self.log_scale
        )                                       # (1, h, n, K)
        cos = theta.cos()
        sin = theta.sin()

        lambda_sigma = torch.exp(self.log_lambda_sigma)   # (h,)
        sheet_bias = _sheet_bias(sheet, lambda_sigma)     # (1, h, n, n)

        if backend == "torch":
            out = _attention_torch(
                q, k, v, cos, sin, sheet_bias,
                self.tril[:n, :n], self.d_head,
            )
        else:
            out = self._forward_native(
                q, k, v, cos, sin, sheet_bias, n, backend,
            )

        out = out.transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.o(out)

    # ------------------------------------------------------------------
    # native dispatch (Rust / CUDA)
    # ------------------------------------------------------------------
    def _forward_native(
        self,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        cos: torch.Tensor, sin: torch.Tensor, sheet_bias: torch.Tensor,
        n: int, backend: str,
    ) -> torch.Tensor:
        b, h, _, d = q.shape
        bh = b * h
        half = d // 2

        # Flatten to (bh, n, *) and broadcast cos/sin/sheet_bias to bh.
        q_f = q.contiguous().view(bh, n, d)
        k_f = k.contiguous().view(bh, n, d)
        v_f = v.contiguous().view(bh, n, d)
        cos_b = cos.expand(b, h, n, half).contiguous().view(bh, n, half)
        sin_b = sin.expand(b, h, n, half).contiguous().view(bh, n, half)
        sb_b  = sheet_bias.expand(b, h, n, n).contiguous().view(bh, n, n)

        if backend == "cuda":
            return self._forward_cuda_devptr(q_f, k_f, v_f, cos_b, sin_b, sb_b,
                                             b, h, n, d)
        # rust CPU path: flat numpy, single dispatch
        q_np = q_f.detach().cpu().numpy().reshape(-1)
        k_np = k_f.detach().cpu().numpy().reshape(-1)
        v_np = v_f.detach().cpu().numpy().reshape(-1)
        c_np = cos_b.detach().cpu().numpy().reshape(-1)
        s_np = sin_b.detach().cpu().numpy().reshape(-1)
        sb_np = sb_b.detach().cpu().numpy().reshape(-1)
        out_np = _rust_riemann_fwd(q_np, k_np, v_np, c_np, s_np, sb_np,
                                   bh, n, d, True)
        out = torch.from_numpy(out_np).view(b, h, n, d).to(q.device, q.dtype)
        return out

    def _forward_cuda_devptr(
        self,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        cos: torch.Tensor, sin: torch.Tensor, sb: torch.Tensor,
        b: int, h: int, n: int, d: int,
    ) -> torch.Tensor:
        bh = b * h
        # Allocate output on the same CUDA device.
        out = torch.empty((bh, n, d), device=q.device, dtype=torch.float32)
        # Sync the producing PyTorch stream so our default CUDA stream
        # sees consistent data; the Rust side syncs its own stream after
        # launch, after which results are visible to PyTorch.
        torch.cuda.synchronize(q.device)
        # PyTorch holds these as f32; if upstream is f16/bf16, cast here.
        def _f32(t: torch.Tensor) -> torch.Tensor:
            return t if t.dtype == torch.float32 else t.float().contiguous()
        qf = _f32(q); kf = _f32(k); vf = _f32(v)
        cf = _f32(cos); sf = _f32(sin); sbf = _f32(sb)
        _rust_riemann_fwd_cuda_devptr(
            int(qf.data_ptr()), int(kf.data_ptr()), int(vf.data_ptr()),
            int(cf.data_ptr()), int(sf.data_ptr()), int(sbf.data_ptr()),
            int(out.data_ptr()),
            bh, n, d, True,
        )
        return out.view(b, h, n, d).to(q.dtype)


class RiemannAttnBlock(nn.Module):
    """Pre-LN block using `RiemannRotaryAttention` + pluggable FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block: int,
        ffn_kind: Optional[str] = None,
        backend: str = "auto",
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = RiemannRotaryAttention(d_model, n_heads, block, backend=backend)
        if ffn_kind is None or ffn_kind == "std":
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model), nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
        else:
            from .ce_ffn import make_ffn
            self.ffn = make_ffn(ffn_kind, d_model, mult=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Design (4) — Riemann-zero spaced FFN init
# ---------------------------------------------------------------------------


@torch.no_grad()
def riemann_zero_init(linear: nn.Linear, axis: str = "in") -> None:
    """Modulate `linear.weight` columns/rows by Riemann-gap spacing.

    Kaiming-normal first; then multiply the chosen axis by a vector
    derived from cumulative γ-gaps (centered, unit-std). The orthogonal
    axis stays Kaiming. Hypothesis: keys spaced by GUE statistics give
    better memory coverage than iid-gaussian.
    """
    W = linear.weight                              # (out, in)
    if axis == "in":
        n = W.shape[1]
    elif axis == "out":
        n = W.shape[0]
    else:
        raise ValueError(f"axis must be 'in' or 'out', got {axis!r}")

    gamma = riemann_zeros(n)
    spacings = torch.cat([gamma[:1], gamma[1:] - gamma[:-1]])
    spacings = spacings / spacings.mean()
    positions = torch.cumsum(spacings, dim=0)
    positions = (positions - positions.mean()) / positions.std().clamp_min(1e-8)

    nn.init.kaiming_normal_(W, nonlinearity="linear")
    if axis == "in":
        W.mul_(positions.view(1, n))
    else:
        W.mul_(positions.view(n, 1))


__all__ = [
    "RIEMANN_ZEROS_IM",
    "riemann_zeros",
    "RiemannRotaryAttention",
    "RiemannAttnBlock",
    "riemann_zero_init",
    "has_rust_riemann",
    "has_cuda_riemann",
]
