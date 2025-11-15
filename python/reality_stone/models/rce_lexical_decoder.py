"""Lexical constraint 기반 디코더 - Phase 4

docs/sentence_topic_architecture.md의 6장 L3: RCE-LexicalDecoder 명세 준수
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

try:
    from reality_stone.models.gpt2_metric import GPT2MetricModel
    from reality_stone.layers.metric_attention import MetricAttention
    HAS_REALITY_STONE = True
except ImportError:
    HAS_REALITY_STONE = False
    print("Warning: reality_stone modules not available, using fallback")


class RCELexicalDecoder(nn.Module):
    """
    Lexical constraint 기반 디코더
    
    docs 명세:
    - geodesic attention 기반 토큰 생성
    - 후보 집합 내에서만 토큰 선택
    - replacement_mask로 고정 토큰 보호
    - Lorentz manifold attention
    """
    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 768,
        n_layer: int = 6,
        n_head: int = 8,
        manifold: str = "lorentz",
        c_lorentz: float = -1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.manifold = manifold
        self.c_lorentz = c_lorentz
        self.vocab_size = vocab_size
        
        # Embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        
        # Transformer blocks with geodesic attention
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, manifold, c_lorentz)
            for _ in range(n_layer)
        ])
        
        # LM head
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        metric_ctx: torch.Tensor,
        replacement_mask: torch.Tensor,
        topo_idx: torch.Tensor,
        candidates: Dict[int, List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [B, T] 입력 토큰 ID
            metric_ctx: [B, T, d_h, d_h] SPD Cholesky factor
            replacement_mask: [B, T] 교체 가능 여부
            topo_idx: [B, T, K] topology index
            candidates: {token_id: [cand1, cand2, ...]}
        
        Returns:
            output_ids: [B, T] 출력 토큰 ID
            logits: [B, T, V] 제약된 logits
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        # Embedding
        token_emb = self.token_embed(input_ids)  # [B, T, d_model]
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos_ids)
        hidden = token_emb + pos_emb  # [B, T, d_model]
        
        # Transformer blocks
        for block in self.blocks:
            hidden = block(hidden, metric_ctx, topo_idx)
        
        # LM head
        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)  # [B, T, V]
        
        # Lexical constraint 적용
        constrained_logits = self._apply_lexical_constraint(
            logits, input_ids, replacement_mask, candidates
        )
        
        # Sampling or argmax
        output_ids = torch.argmax(constrained_logits, dim=-1)
        
        # replacement_mask=0 위치는 원본 유지
        output_ids = torch.where(
            replacement_mask.bool(),
            output_ids,
            input_ids
        )
        
        return output_ids, constrained_logits
    
    def _apply_lexical_constraint(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
        candidates: Dict[int, List[int]]
    ) -> torch.Tensor:
        """
        후보 집합 외 토큰은 -inf로 마스킹
        
        docs 명세:
        - 후보 집합 C_t에 대해서만 확률 계산
        - replacement_mask=0 위치는 제약 없음
        """
        B, T, V = logits.shape
        constrained = logits.clone()
        
        for b in range(B):
            for t in range(T):
                if mask[b, t].item() == 0:
                    continue  # 고정 토큰은 제약 없음
                
                token_id = input_ids[b, t].item()
                if token_id in candidates and len(candidates[token_id]) > 0:
                    # 후보 외 토큰 마스킹
                    valid_ids = candidates[token_id]
                    mask_tensor = torch.ones(V, dtype=torch.bool, device=logits.device)
                    mask_tensor[valid_ids] = False
                    constrained[b, t, mask_tensor] = float('-inf')
        
        return constrained


class TransformerBlock(nn.Module):
    """
    Transformer block with geodesic attention
    
    docs 명세:
    - MetricAttention 사용
    - Lorentz manifold
    """
    def __init__(self, d_model, n_head, manifold, c):
        super().__init__()

        # 현재 구현에서는 MetricAttention 대신 표준 MultiheadAttention만 사용한다.
        # (MetricAttention 경로는 향후 안정화 후 다시 연결 예정)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x, metric_ctx, topo_idx):
        """
        Args:
            x: [B, T, d_model]
            metric_ctx: [B, T, d_h, d_h] (현재 미사용, 향후 확장용 placeholder)
            topo_idx: [B, T, K] (현재 MultiheadAttention에는 직접 사용되지 않음)
        """
        # 현재는 표준 Self-Attention만 사용
        attn_out, _ = self.attn(x, x, x)
        
        x = x + attn_out
        x = self.ln1(x)
        
        # FFN
        x = x + self.mlp(x)
        x = self.ln2(x)
        
        return x

