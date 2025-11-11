"""
Riemann metric-based intent classifier (lightweight vs BERT).

Architecture:
  embed → pos_enc → 2~3 × RiemannEncoderBlock → pooler → classifier

Target:
  - Params ~5M (vs BERT 110M)
  - Accuracy ≥95% on CLINC150
  - Speed ≥10× BERT
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.metric_attention import MetricAttention
from ..layers.riemann_lowrank import RiemannLowRankLinear


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class RiemannEncoderBlock(nn.Module):
    """
    Single encoder block with metric attention + feedforward.

    Args:
        d_model: model hidden size
        n_heads: number of attention heads
        d_ff: feedforward inner dimension (default 4*d_model)
        rank: low-rank for metric attention (0 = diagonal only)
        normalizer: softmax | entmax | sinkhorn
        dropout: dropout rate
        use_riemann_ffn: if True, use RiemannLowRankLinear for FFN; else standard MLP
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 2,
        d_ff: Optional[int] = None,
        rank: int = 0,
        normalizer: str = "softmax",
        dropout: float = 0.1,
        use_riemann_ffn: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.use_riemann_ffn = use_riemann_ffn

        # Multi-head metric attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn = MetricAttention(self.d_head, normalizer=normalizer, rank=rank, tau=1.0)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feedforward
        if use_riemann_ffn:
            # Riemann low-rank MLP (2 layers)
            self.ffn_1 = RiemannLowRankLinear(d_model, self.d_ff, r=64, c=1e-3, bias=True)
            self.ffn_2 = RiemannLowRankLinear(self.d_ff, d_model, r=64, c=1e-3, bias=True)
        else:
            self.ffn_1 = nn.Linear(d_model, self.d_ff)
            self.ffn_2 = nn.Linear(self.d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, d_model = x.shape

        # Multi-head attention
        res = x
        x = self.norm1(x)
        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Reshape to (B, H, T, d_h)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # Metric attention
        y = self.attn(q, k, v, causal=causal)  # (B, H, T, d_h)
        y = y.transpose(1, 2).contiguous().view(B, T, d_model)
        y = self.out_proj(y)
        x = res + self.dropout(y)

        # Feedforward
        res = x
        x = self.norm2(x)
        if self.use_riemann_ffn:
            # Riemann path
            h = self.ffn_1(x)
            h = F.gelu(h)
            h = self.ffn_2(h)
        else:
            h = self.ffn_1(x)
            h = F.gelu(h)
            h = self.ffn_2(h)
        x = res + self.dropout(h)
        return x


class MeanMaxPooler(nn.Module):
    """Pooler combining mean & max pooling."""

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, d)
        if mask is not None:
            # mask: (B, T), 1=valid
            mask = mask.unsqueeze(-1).float()  # (B, T, 1)
            x_masked = x * mask
            x_mean = x_masked.sum(dim=1) / (mask.sum(dim=1) + 1e-9)
            x_max = x_masked.max(dim=1).values
        else:
            x_mean = x.mean(dim=1)
            x_max = x.max(dim=1).values
        return torch.cat([x_mean, x_max], dim=-1)  # (B, 2*d)


class IntentClassifier(nn.Module):
    """
    Lightweight Riemann metric-based intent classifier.

    Args:
        vocab_size: vocabulary size
        num_intents: number of intent classes
        d_model: embedding/hidden dimension (default 128)
        n_layers: number of encoder blocks (default 2)
        n_heads: number of attention heads per block (default 2)
        d_ff: feedforward inner dimension (default 4*d_model)
        rank: low-rank for metric attention (default 4)
        normalizer: softmax | entmax | sinkhorn
        pooling: 'meanmax' | 'cls' | 'mean'
        dropout: dropout rate
        use_riemann_ffn: if True, use RiemannLowRankLinear for FFN
        max_len: max sequence length
    """

    def __init__(
        self,
        vocab_size: int,
        num_intents: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 2,
        d_ff: Optional[int] = None,
        rank: int = 4,
        normalizer: str = "softmax",
        pooling: str = "meanmax",
        dropout: float = 0.1,
        use_riemann_ffn: bool = False,
        max_len: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.pooling = pooling

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        self.encoder_blocks = nn.ModuleList(
            [
                RiemannEncoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    rank=rank,
                    normalizer=normalizer,
                    dropout=dropout,
                    use_riemann_ffn=use_riemann_ffn,
                )
                for _ in range(n_layers)
            ]
        )

        if pooling == "meanmax":
            self.pooler = MeanMaxPooler()
            classifier_in = d_model * 2
        elif pooling == "cls":
            self.pooler = None
            classifier_in = d_model
        else:
            self.pooler = None
            classifier_in = d_model

        self.classifier = nn.Linear(classifier_in, num_intents)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # input_ids: (B, T) LongTensor
        # attention_mask: (B, T) 1=valid, 0=pad (optional)
        x = self.embed(input_ids)  # (B, T, d_model)
        x = self.pos_enc(x)

        for block in self.encoder_blocks:
            x = block(x, causal=False)

        # Pooling
        if self.pooling == "meanmax":
            x = self.pooler(x, mask=attention_mask)  # (B, 2*d_model)
        elif self.pooling == "cls":
            x = x[:, 0, :]  # (B, d_model)
        else:
            # mean
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                x = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
            else:
                x = x.mean(dim=1)

        logits = self.classifier(x)  # (B, num_intents)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = [
    "SinusoidalPositionalEncoding",
    "RiemannEncoderBlock",
    "MeanMaxPooler",
    "IntentClassifier",
]

