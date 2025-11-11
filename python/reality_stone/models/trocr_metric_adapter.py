from typing import Optional, Dict

import torch
import torch.nn as nn

from ..layers.metric_attention import MetricAttention


class BartLikeMetricSelfAttention(nn.Module):
    """
    Replace a BART-like self-attention using MetricAttention.
    Expects existing q_proj, k_proj, v_proj, out_proj modules.
    """

    def __init__(self, q_proj: nn.Module, k_proj: nn.Module, v_proj: nn.Module, out_proj: nn.Module,
                 num_heads: int, head_dim: int, normalizer: str = "softmax", rank: int = 0) -> None:
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.out_proj = out_proj
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.metric_attn = MetricAttention(head_dim, normalizer=normalizer, rank=rank)

    def _shape(self, x: torch.Tensor, bsz: int, tgt_len: int) -> torch.Tensor:
        return x.view(bsz, tgt_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

    def forward(self, hidden_states: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
                layer_state: Optional[Dict[str, torch.Tensor]] = None, attn_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False, **kwargs):
        bsz, tgt_len, embed_dim = hidden_states.size()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = self._shape(q, bsz, tgt_len)
        k = self._shape(k, bsz, tgt_len)
        v = self._shape(v, bsz, tgt_len)

        y = self.metric_attn(q, k, v, topo_idx=None, topk_cfg=None, causal=True)
        y = y.permute(0, 2, 1, 3).contiguous().view(bsz, tgt_len, self.num_heads * self.head_dim)
        y = self.out_proj(y)
        if output_attentions:
            return (y, None)
        return (y,)


def patch_trocr_decoder_with_metric(model: nn.Module, rank: int = 0, normalizer: str = "softmax") -> int:
    """
    Traverse decoder modules and replace self-attn blocks that look like BART with BartLikeMetricSelfAttention.
    Returns number of layers patched.
    """
    n_patched = 0
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return 0

    for name, module in decoder.named_modules():
        # Heuristic: module has q_proj, k_proj, v_proj, out_proj and num_heads/head_dim attributes
        has_qkv = all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj"))
        has_out = hasattr(module, "out_proj")
        has_dims = hasattr(module, "num_heads") and hasattr(module, "head_dim")
        if has_qkv and has_out and has_dims:
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            # Replace module in its parent
            parent = decoder
            if parent_name:
                for p in parent_name.split("."):
                    parent = getattr(parent, p)
            try:
                new_mod = BartLikeMetricSelfAttention(
                    module.q_proj, module.k_proj, module.v_proj, module.out_proj,
                    int(module.num_heads), int(module.head_dim), normalizer=normalizer, rank=rank
                )
                setattr(parent, name.split(".")[-1], new_mod)
                n_patched += 1
            except Exception:
                continue
    return n_patched


__all__ = [
    "patch_trocr_decoder_with_metric",
]




