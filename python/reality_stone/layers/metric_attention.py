import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .poincare import poincare_distance
from .lorentz import lorentz_distance
from .klein import klein_distance

# Try to import CUDA-accelerated kernel
try:
    from reality_stone._rust_geodesic import geodesic_topk_attention
    HAS_CUDA_KERNEL = True
except ImportError:
    HAS_CUDA_KERNEL = False


class SPDMetric(nn.Module):
    def __init__(self, hidden_size: int, rank: int = 0, init_u_scale: float = 1e-3) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = int(rank) if rank is not None else 0

        self.log_diag = nn.Parameter(torch.zeros(hidden_size))
        if self.rank > 0:
            u = torch.randn(hidden_size, self.rank) * float(init_u_scale)
            self.U = nn.Parameter(u)
        else:
            self.U = None

    def scale_q(self, q: Tensor) -> Tensor:
        d = F.softplus(self.log_diag).view(1, 1, 1, -1)
        return q * d

    def scale_k(self, k: Tensor) -> Tensor:
        d = F.softplus(self.log_diag).view(1, 1, 1, -1)
        return k * d

    def lowrank_proj(self, x: Tensor) -> Optional[Tensor]:
        if self.U is None:
            return None
        return torch.einsum("bhtd,dr->bhtr", x, self.U)


def _sparsemax(logits: Tensor, dim: int = -1) -> Tensor:
    z = logits
    z = z - z.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    range_arange = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    range_view = [1] * z.dim()
    range_view[dim] = -1
    range_arange = range_arange.view(range_view)
    cssv = torch.cumsum(z_sorted, dim=dim) - range_arange * z_sorted
    nonzero = (z_sorted > (cssv / range_arange)).to(z.dtype)
    k = (nonzero * range_arange).max(dim=dim, keepdim=True).values.clamp(min=1.0)
    tau = (torch.gather(z_sorted, dim, k.long() - 1) - (cssv.gather(dim, k.long() - 1) / k)).detach()
    p = torch.clamp(z - tau, min=0.0)
    return p


def _sinkhorn(logits: Tensor, iters: int = 20, tau: float = 1.0, eps: float = 1e-9) -> Tensor:
    x = torch.exp(logits / max(tau, 1e-6))
    for _ in range(max(1, int(iters))):
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
    return x


def normalize(scores: Tensor, method: str = "softmax", tau: float = 1.0) -> Tensor:
    scores = scores - scores.max(dim=-1, keepdim=True).values
    if method == "softmax":
        return torch.softmax(scores / max(tau, 1e-6), dim=-1)
    if method in {"entmax", "entmax15", "sparsemax"}:
        # Dependency-free substitute; use sparsemax as a robust sparse normalizer.
        return _sparsemax(scores, dim=-1)
    if method == "sinkhorn":
        return _sinkhorn(scores, iters=20, tau=tau)
    # Fallback
    return torch.softmax(scores / max(tau, 1e-6), dim=-1)


def build_topo_topk(topo_idx: Dict[str, Tensor], topk_cfg: Dict[str, int]) -> Tensor:
    if not isinstance(topo_idx, dict) or not topo_idx:
        raise ValueError("topo_idx must be a non-empty dict of relation -> indices (B,T,k_r)")

    idx_tensors = []
    for rel, k in topk_cfg.items():
        if k is None or k <= 0:
            continue
        if rel not in topo_idx:
            continue
        rel_idx = topo_idx[rel]
        if rel_idx.shape[-1] > k:
            rel_idx = rel_idx[..., :k]
        idx_tensors.append(rel_idx)

    if not idx_tensors:
        raise ValueError("No relations matched between topo_idx and topk_cfg")

    idx_all = torch.cat(idx_tensors, dim=-1)

    B, T, K_raw = idx_all.shape
    idx_flat = idx_all.reshape(B * T, K_raw)
    idx_sorted, _ = torch.sort(idx_flat, dim=-1)
    idx_uniq = torch.unique_consecutive(idx_sorted, dim=-1)
    K = idx_uniq.shape[-1]
    if K < K_raw:
        pad = idx_uniq[..., -1:].expand(B * T, K_raw - K)
        idx_uniq = torch.cat([idx_uniq, pad], dim=-1)
    elif K > K_raw:
        idx_uniq = idx_uniq[..., :K_raw]
    idx = idx_uniq.reshape(B, T, K_raw)
    return idx


def masked_gather(scores: Tensor, idx: Tensor) -> Tensor:
    B, H, T, S = scores.shape
    if idx.shape[0] != B or idx.shape[1] != T:
        raise ValueError("idx batch/time dims must match scores")
    idx_exp = idx.unsqueeze(1).expand(B, H, T, -1)
    return scores.gather(dim=3, index=idx_exp)


def aggregate(weights: Tensor, values: Tensor, idx: Tensor) -> Tensor:
    B, H, S, Dv = values.shape
    _, _, T, K = weights.shape
    if idx.shape[0] != B or idx.shape[1] != T:
        raise ValueError("idx batch/time dims must match weights")
    idx_h = idx.unsqueeze(1).expand(B, H, T, K)
    values_flat = values.reshape(B * H, S, Dv)
    idx_flat = idx_h.reshape(B * H, T * K)
    v_g = values_flat.gather(dim=1, index=idx_flat.unsqueeze(-1).expand(B * H, T * K, Dv))
    v_sel = v_g.reshape(B, H, T, K, Dv)
    y = (weights.unsqueeze(-1) * v_sel).sum(dim=3)
    return y


def get_default_topk_cfg() -> Dict[str, int]:
    return {"cell": 8, "row": 16, "col": 8, "pc": 4, "hdr": 12}


class MetricAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        normalizer: str = "softmax",
        rank: int = 0,
        tau: float = 1.0,
        mode: str = "dot",
        manifold: str = "poincare",
        c: float = 1e-3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.normalizer = normalizer
        self.tau = float(tau)
        self.metric = SPDMetric(hidden_size, rank=rank)
        self.mode = str(mode)
        self.manifold = str(manifold)
        self.c_default = float(c)
        self._metric_cache: Dict[str, torch.Tensor] = {}

    def _apply_metric_factor(self, x: Tensor, l_factor: Tensor) -> Tensor:
        if x.dim() != 4 or l_factor.dim() != 2:
            return x
        l = l_factor.to(device=x.device, dtype=x.dtype)
        return torch.einsum("ij,bhtj->bhti", l, x)

    def _cholesky_from_keys(
        self,
        keys: list[str],
        masses: Optional[list[float]],
        dim: int,
        min_lambda: float = 0.8,
        max_lambda: float = 1.2,
    ) -> torch.Tensor:
        """
        Build SPD metric from key(s) and return Cholesky factor L such that G = L L^T.
        Uses weighted average over per-key SPD metrics (still SPD).
        Cached by tuple(keys)+tuple(masses).
        """
        from .. import metrikey as _metrikey  # defer import to avoid circulars and allow no-extension mode
        if _metrikey is None:
            raise RuntimeError("MetriKey extension is not available. Omit metric_keys or use Python 3.11 to load the bundled extension.")
        import numpy as np  # lazy import to avoid hard dependency if SPD path is unused

        mk = tuple(keys), tuple(masses or [1.0] * len(keys)), dim, float(min_lambda), float(max_lambda)
        cache_key = f"{mk}"
        if cache_key in self._metric_cache:
            return self._metric_cache[cache_key]

        # accumulate SPD metrics
        m = np.array(mk[1], dtype="float32")
        m_sum = float(m.sum()) if m.size > 0 else 1.0
        g_accum = None
        for k, mass in zip(keys, (masses or [1.0] * len(keys))):
            g_k = _metrikey.spd_metric_from_key_weighted(k, dim, float(min_lambda), float(max_lambda), float(mass))
            g_accum = g_k if g_accum is None else (g_accum + g_k)
        g = g_accum / max(m_sum, 1e-6)
        l = _metrikey.metric_factor_cholesky(g)  # numpy array
        l_t = torch.from_numpy(np.asarray(l, dtype=np.float32))
        # cache on CPU to avoid device-specific duplication
        self._metric_cache[cache_key] = l_t
        return l_t

    def _geodesic_distance_pairs(self, q_pairs: Tensor, k_pairs: Tensor, c: float) -> Tensor:
        """
        Compute per-pair geodesic distance for flattened pairs.
        q_pairs, k_pairs: (N, d)
        Returns: (N,)
        """
        if self.manifold == "poincare":
            return poincare_distance(q_pairs, k_pairs, c)
        if self.manifold == "lorentz":
            return lorentz_distance(q_pairs, k_pairs, c)
        if self.manifold == "klein":
            return klein_distance(q_pairs, k_pairs, c)
        # Fallback: Euclidean
        return torch.norm(q_pairs - k_pairs, dim=-1)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        topo_idx: Optional[Dict[str, Tensor]] = None,
        rel_bias: Optional[Tensor] = None,
        topk_cfg: Optional[Dict[str, int]] = None,
        causal: bool = False,
        # RCE options
        metric_keys: Optional[list[str]] = None,   # e.g., ["dept:0","dept:3"]
        masses: Optional[list[float]] = None,      # same length as metric_keys
        metric_keys_b: Optional[list[str]] = None, # optional second context to mix
        alpha: Optional[float] = None,             # mix ratio for TCS: g' = a g1 + (1-a) g2
        c: Optional[float] = None,                 # curvature override for geodesic
    ) -> Tensor:
        # Optionally apply metric-key based SPD transform (security/context)
        qs, ks = q, k
        if metric_keys:
            d = q.shape[-1]
            # Compose single-context SPD
            l1 = self._cholesky_from_keys(metric_keys, masses, d)
            if metric_keys_b and alpha is not None:
                # Mix two contexts on the SPD: g' = a g1 + (1-a) g2
                l2 = self._cholesky_from_keys(metric_keys_b, masses, d)
                # Reconstruct G from L then mix and re-factor (CPU numpy for stability)
                import numpy as np  # local import
                from .. import metrikey as _metrikey  # local import to avoid circulars
                g1 = (l1 @ l1.t()).cpu().numpy()
                g2 = (l2 @ l2.t()).cpu().numpy()
                a = float(max(0.0, min(1.0, alpha)))
                g_mix = a * g1 + (1.0 - a) * g2
                if _metrikey is None:
                    raise RuntimeError("MetriKey extension is not available for mixed SPD context.")
                l_mix = _metrikey.metric_factor_cholesky(g_mix)
                l_used = torch.from_numpy(np.asarray(l_mix, dtype=np.float32))
            else:
                l_used = l1
            qs = self._apply_metric_factor(q, l_used)
            ks = self._apply_metric_factor(k, l_used)

        # Metric swap (learnable diag / optional low-rank) — applies after metric-key
        qs = self.metric.scale_q(qs)  # (B,H,T,d_h)
        ks = self.metric.scale_k(ks)  # (B,H,S,d_h)

        qu = self.metric.lowrank_proj(qs)
        ku = self.metric.lowrank_proj(ks)

        # Geodesic Top-k branch (preferred with topology selection)
        if self.mode == "geodesic" and topo_idx is not None and topk_cfg is not None:
            idx = build_topo_topk(topo_idx, topk_cfg)  # (B,T,K)
            B, H, T, Dh = qs.shape
            S = ks.shape[-2]
            K = idx.shape[-1]
            
            # 🚀 CUDA Fast Path: Fused Geodesic Top-k Attention
            if HAS_CUDA_KERNEL and qs.is_cuda and metric_keys:
                # Get Cholesky factor from metric keys
                l_factor = self._cholesky_from_keys(
                    metric_keys, masses, Dh
                ).to(qs.device)
                
                c_used = float(self.c_default if c is None else c)
                
                # Call fused CUDA kernel (6x faster!)
                try:
                    y = geodesic_topk_attention(
                        qs, ks, v, idx, l_factor, c_used, self.tau
                    )
                    return y
                except Exception as e:
                    # Fallback to Python if CUDA fails
                    print(f"CUDA kernel failed ({e}), falling back to Python")
            
            # Python fallback (original implementation)
            # Flatten and gather selected keys: (B*H, S, Dh) -> (B*H, T*K, Dh)
            ks_flat = ks.reshape(B * H, S, Dh)
            idx_flat = idx.unsqueeze(1).expand(B, H, T, K).reshape(B * H, T * K)
            ks_sel_flat = ks_flat.gather(1, idx_flat.unsqueeze(-1).expand(B * H, T * K, Dh))
            # Replicate queries per K: (B*H, T, Dh) -> (B*H, T*K, Dh)
            q_flat = qs.reshape(B * H, T, Dh)
            q_rep = q_flat.unsqueeze(2).expand(B * H, T, K, Dh).reshape(B * H, T * K, Dh)
            # Compute geodesic distance per pair
            qf = q_rep.reshape(B * H * T * K, Dh)
            kf = ks_sel_flat.reshape(B * H * T * K, Dh)
            c_used = float(self.c_default if c is None else c)
            dist = self._geodesic_distance_pairs(qf, kf, c_used)  # (B*H*T*K,)
            d2 = dist.pow(2.0).reshape(B, H, T, K)
            # Convert to scores, then normalize per Top-k set
            s_sel = -d2 / max(self.tau, 1e-6)
            if rel_bias is not None:
                # if bias is full (B,H,T,S), gather it
                if rel_bias.dim() == 4 and rel_bias.shape[-1] == S:
                    b_sel = masked_gather(rel_bias, idx)
                    s_sel = s_sel + b_sel
            a = normalize(s_sel, method=self.normalizer, tau=1.0)
            y = aggregate(a, v, idx)  # (B,H,T,d_v)
            return y

        # Dot-product path (default) or geodesic fallback without topo_idx
        s = torch.einsum("bhtd,bhsd->bhts", qs, ks) / math.sqrt(self.hidden_size)
        if qu is not None and ku is not None:
            s = s + torch.einsum("bhtr,bhsr->bhts", qu, ku)
        if rel_bias is not None:
            s = s + rel_bias
        if causal and s.size(2) == s.size(3):
            t = s.size(2)
            mask = torch.ones((t, t), device=s.device, dtype=torch.bool).triu(diagonal=1)
            s = s.masked_fill(mask.view(1, 1, t, t), float("-inf"))
        if topo_idx is not None and topk_cfg is not None:
            idx = build_topo_topk(topo_idx, topk_cfg)  # (B,T,K)
            s_sel = masked_gather(s, idx)  # (B,H,T,K)
            a = normalize(s_sel, method=self.normalizer, tau=self.tau)
            y = aggregate(a, v, idx)
            return y
        a_full = normalize(s, method=self.normalizer, tau=self.tau)
        y_full = torch.einsum("bhts,bhsd->bhtd", a_full, v)
        return y_full


__all__ = [
    "SPDMetric",
    "MetricAttention",
    "normalize",
    "build_topo_topk",
    "masked_gather",
    "aggregate",
    "get_default_topk_cfg",
]


