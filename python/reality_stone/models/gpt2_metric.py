import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from ..layers.metric_attention import MetricAttention


class MetricGPT2Attention(nn.Module):
    """
    GPT-2 attention using MetricAttention (diag/low-rank/Top-k capable).

    This module expects to receive the original GPT-2 block's `c_attn` and `c_proj`
    via attribute assignment when used in a conversion utility.
    """

    def __init__(self, config: GPT2Config, is_cross_attention: bool = False, layer_idx: Optional[int] = None,
                 rank: int = 0, normalizer: str = "softmax") -> None:
        super().__init__()
        self.config = config
        self.is_cross_attention = is_cross_attention
        self.layer_idx = layer_idx
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.split_size = self.embed_dim
        self.scale_attn_weights = config.scale_attn_weights

        # Placeholders - to be set from original attn
        self.c_attn: nn.Module = None  # type: ignore
        self.c_proj: nn.Module = None  # type: ignore

        # Metric attention core
        self.metric_attn = MetricAttention(self.head_dim, normalizer=normalizer, rank=rank)

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        new_shape = tensor.size()[:-1] + (self.num_heads, self.head_dim)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (B,H,T,dh)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (self.num_heads * self.head_dim,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        topo_idx: Optional[dict] = None,
        topk_cfg: Optional[dict] = None,
        **kwargs,
    ):
        if encoder_hidden_states is not None:
            raise NotImplementedError("Cross-attention not implemented for MetricGPT2Attention")

        # HF compatibility: accept past_key_value alias
        if (layer_past is None) and ("past_key_value" in kwargs):
            pkv = kwargs.get("past_key_value", None)
            # Accept empty tuple or None gracefully
            if isinstance(pkv, (tuple, list)) and len(pkv) == 2:
                layer_past = (pkv[0], pkv[1])
            else:
                layer_past = None

        # qkv projection
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.split_size, dim=2)

        # Split heads
        q = self._split_heads(query)
        k = self._split_heads(key)
        v = self._split_heads(value)

        # Append past
        present = None
        if layer_past is not None and isinstance(layer_past, (tuple, list)) and len(layer_past) == 2:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        if use_cache:
            present = (k, v)

        # Causal mask support
        B, H, Tq, _ = q.shape
        Ts = k.shape[-2]
        causal = True

        # Attention via MetricAttention
        y = self.metric_attn(q, k, v, topo_idx=topo_idx, topk_cfg=topk_cfg, causal=causal)

        # Merge heads and output proj
        y = self._merge_heads(y)
        y = self.c_proj(y)
        y = self.resid_dropout(y)

        outputs = (y, present)
        if output_attentions:
            outputs += (None,)
        return outputs


def convert_gpt2_to_metric(model: nn.Module, rank: int = 0, normalizer: str = "softmax") -> nn.Module:
    """Replace GPT-2 attention with MetricGPT2Attention in-place, reusing projection weights."""
    config = model.config  # type: ignore
    for i, block in enumerate(model.transformer.h):  # type: ignore
        old_attn = block.attn
        new_attn = MetricGPT2Attention(config, layer_idx=i, rank=rank, normalizer=normalizer)
        # Reuse original projections
        new_attn.c_attn = old_attn.c_attn
        new_attn.c_proj = old_attn.c_proj
        block.attn = new_attn
    return model


__all__ = [
    "MetricGPT2Attention",
    "convert_gpt2_to_metric",
]


