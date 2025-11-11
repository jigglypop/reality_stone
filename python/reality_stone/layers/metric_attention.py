import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SPDMetric(nn.Module):
    """
    SPD metric parameterization G = diag(softplus(d)) + U U^T.

    - Diagonal swap (token-wise precision gating) via softplus for SPD safety.
    - Low-rank swap provides a small auxiliary term for attention scores.

    Args:
        hidden_size: head hidden dimension (d_h)
        rank: low-rank dimension r (<= 8 recommended)
        init_u_scale: std for U init (kept very small for stability)
    """

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
        """Apply diagonal swap to queries: q' = q * softplus(d)."""
        d = F.softplus(self.log_diag).view(1, 1, 1, -1)
        return q * d

    def scale_k(self, k: Tensor) -> Tensor:
        """Apply diagonal swap to keys: k' = k * softplus(d)."""
        d = F.softplus(self.log_diag).view(1, 1, 1, -1)
        return k * d

    def lowrank_proj(self, x: Tensor) -> Optional[Tensor]:
        """
        Project onto low-rank factors: U^T x

        Shapes:
            x: (B, H, T, d_h) -> (B, H, T, r)
        """
        if self.U is None:
            return None
        return torch.einsum("bhtd,dr->bhtr", x, self.U)


def _sparsemax(logits: Tensor, dim: int = -1) -> Tensor:
    """
    Sparsemax (Martins & Astudillo, 2016) – simple, dependency-free sparse normalizer.
    """
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
    # Row-wise normalization is guaranteed
    return p


def _sinkhorn(logits: Tensor, iters: int = 20, tau: float = 1.0, eps: float = 1e-9) -> Tensor:
    """
    Minimal Sinkhorn-like row normalization for stability (non-square case).
    Note: For rectangular (B,H,T,k), we repeatedly renormalize rows across k.
    """
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
    """
    Build union Top-k neighbor indices per query token.

    Args:
        topo_idx: mapping relation -> indices tensor of shape (B, T, k_r)
                  e.g., keys: {'cell','row','col','pc','hdr'}; values are LongTensor
        topk_cfg: mapping relation -> k (use only if relation exists in topo_idx)

    Returns:
        idx: LongTensor (B, T, K) where K = sum_k over available relations
    """
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

    idx_all = torch.cat(idx_tensors, dim=-1)  # (B,T,K_raw)

    # Deduplicate while roughly preserving order (unique_consecutive on sorted per-row)
    B, T, K_raw = idx_all.shape
    idx_flat = idx_all.reshape(B * T, K_raw)
    idx_sorted, _ = torch.sort(idx_flat, dim=-1)
    idx_uniq = torch.unique_consecutive(idx_sorted, dim=-1)
    # If uniqueness shrinks length, right-pad with last element to keep shape consistent
    K = idx_uniq.shape[-1]
    if K < K_raw:
        pad = idx_uniq[..., -1:].expand(B * T, K_raw - K)
        idx_uniq = torch.cat([idx_uniq, pad], dim=-1)
    elif K > K_raw:
        idx_uniq = idx_uniq[..., :K_raw]
    idx = idx_uniq.reshape(B, T, K_raw)
    return idx


def masked_gather(scores: Tensor, idx: Tensor) -> Tensor:
    """
    Gather selected scores by idx along the key dimension.

    Args:
        scores: (B, H, T, S)
        idx:    (B, T, K)

    Returns:
        S_sel: (B, H, T, K)
    """
    B, H, T, S = scores.shape
    if idx.shape[0] != B or idx.shape[1] != T:
        raise ValueError("idx batch/time dims must match scores")
    idx_exp = idx.unsqueeze(1).expand(B, H, T, -1)
    return scores.gather(dim=3, index=idx_exp)


def aggregate(weights: Tensor, values: Tensor, idx: Tensor) -> Tensor:
    """
    Weighted aggregation of values with gathered indices.

    Args:
        weights: (B, H, T, K)
        values:  (B, H, S, D_v)
        idx:     (B, T, K)

    Returns:
        Y:       (B, H, T, D_v)
    """
    B, H, S, Dv = values.shape
    _, _, T, K = weights.shape
    if idx.shape[0] != B or idx.shape[1] != T:
        raise ValueError("idx batch/time dims must match weights")
    idx_h = idx.unsqueeze(1).expand(B, H, T, K)  # (B,H,T,K)
    # Gather along sequence dim with flattened (B*H) batches to satisfy dim rules
    values_flat = values.reshape(B * H, S, Dv)  # (BH, S, Dv)
    idx_flat = idx_h.reshape(B * H, T * K)  # (BH, T*K)
    v_g = values_flat.gather(dim=1, index=idx_flat.unsqueeze(-1).expand(B * H, T * K, Dv))  # (BH, T*K, Dv)
    v_sel = v_g.reshape(B, H, T, K, Dv)  # (B,H,T,K,Dv)
    y = (weights.unsqueeze(-1) * v_sel).sum(dim=3)  # (B,H,T,Dv)
    return y


def get_default_topk_cfg() -> Dict[str, int]:
    """Recommended default k per relation for tables."""
    return {"cell": 8, "row": 16, "col": 8, "pc": 4, "hdr": 12}


class MetricAttention(nn.Module):
    """
    Geodesic Top-k attention with SPD metric swap (diag/low-rank).

    Inputs expect head-split tensors:
        Q: (B, H, T, d_h)
        K: (B, H, S, d_h)
        V: (B, H, S, d_v)

    If topo_idx is provided, performs union Top-k masked attention in O(Tk).
    Otherwise falls back to full attention in O(TS).

    Normalizers: 'softmax' | 'entmax' (sparsemax substitute) | 'sinkhorn'

    A/B suggestions:
        - A0: normalizer='softmax', rank=0, no topo_idx (baseline)
        - A1: same, but apply static swap offline (outside this module)
        - A2: rank=0 with diag swap active (this module), no topo_idx
        - A3: rank in {4,8}, no topo_idx
        - B1+: provide topo_idx + topk_cfg, try normalizer='softmax'/'entmax'/'sinkhorn'
    """

    def __init__(
        self,
        hidden_size: int,
        normalizer: str = "softmax",
        rank: int = 0,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.normalizer = normalizer
        self.tau = float(tau)
        self.metric = SPDMetric(hidden_size, rank=rank)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        topo_idx: Optional[Dict[str, Tensor]] = None,
        rel_bias: Optional[Tensor] = None,
        topk_cfg: Optional[Dict[str, int]] = None,
        causal: bool = False,
    ) -> Tensor:
        # Metric swap (diagonal) and optional low-rank auxiliary term
        qs = self.metric.scale_q(q)  # (B,H,T,d_h)
        ks = self.metric.scale_k(k)  # (B,H,S,d_h)

        qu = self.metric.lowrank_proj(qs)
        ku = self.metric.lowrank_proj(ks)

        # Base scores
        s = torch.einsum("bhtd,bhsd->bhts", qs, ks) / math.sqrt(self.hidden_size)
        if qu is not None and ku is not None:
            s = s + torch.einsum("bhtr,bhsr->bhts", qu, ku)
        if rel_bias is not None:
            s = s + rel_bias

        # Optional causal mask (only valid when S==T)
        if causal and s.size(2) == s.size(3):
            t = s.size(2)
            mask = torch.ones((t, t), device=s.device, dtype=torch.bool).triu(diagonal=1)
            s = s.masked_fill(mask.view(1, 1, t, t), float("-inf"))

        if topo_idx is not None and topk_cfg is not None:
            idx = build_topo_topk(topo_idx, topk_cfg)  # (B,T,K)
            s_sel = masked_gather(s, idx)  # (B,H,T,K)
            a = normalize(s_sel, method=self.normalizer, tau=self.tau)
            y = aggregate(a, v, idx)  # (B,H,T,d_v)
            return y

        # Full attention fallback
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


