"""문장 주제 분류 모듈 - Phase 2

docs/sentence_topic_architecture.md의 4장 L1: SentenceTopicHead 명세 준수
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

try:
    from reality_stone.layers.poincare_embedding import PoincareEmbedding
    from reality_stone.layers.metric_attention import MetricAttention
    HAS_REALITY_STONE = True
except ImportError:
    HAS_REALITY_STONE = False
    print("Warning: reality_stone modules not available, using fallback implementations")


class SentenceTopicHead(nn.Module):
    """
    문장 주제 분류 헤드
    
    docs 명세:
    - Poincaré embedding으로 주제 계층 구조 표현
    - geodesic 거리 기반 attention으로 문장 간 관계 파악
    - 주제 분류, 우선순위 점수, metric key seed 출력
    """
    def __init__(
        self,
        d_model: int = 768,
        d_head: int = 64,
        num_topics: int = 8,
        num_heads: int = 4,
        c_poincare: float = 1e-3,
        temperature: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.num_topics = num_topics
        
        # Poincaré embedding
        # 현재 SentenceTopicHead는 "문장 임베딩 x: [B, T, d_model]" 을 입력으로 받는다.
        # PoincareEmbedding 은 정수 ID를 입력으로 받는 임베딩 레이어이므로,
        # 여기서는 d_model → d_head 선형 변환으로 매핑만 수행한다.
        # (향후 토큰 ID 기반 Poincaré 임베딩을 쓰는 버전은 별도로 분리하는 것이 안전하다.)
        self.poincare_embed = nn.Linear(d_model, d_head)
        self.use_poincare = False
        
        # Head projection for MetricAttention
        self.num_heads = num_heads
        self.d_head_per_head = d_head // num_heads
        
        # Geodesic attention with SPD Metric Learning
        if HAS_REALITY_STONE:
            try:
                # SPDMetric은 per-head dimension을 사용
                self.metric_attn = MetricAttention(
                    hidden_size=self.d_head_per_head,  # d_h (per-head dimension)
                    normalizer="softmax",
                    rank=2,  # Low-rank SPD component (작게 유지)
                    mode="geodesic",
                    manifold="poincare",
                    c=c_poincare
                )
                self.use_metric_attn = True
                self.q_proj = nn.Linear(d_head, d_head)
                self.k_proj = nn.Linear(d_head, d_head)
                self.v_proj = nn.Linear(d_head, d_head)
                self.out_proj = nn.Linear(d_head, d_head)
                print(f"✓ Using MetricAttention with SPD Metric Learning (d_h={self.d_head_per_head}, rank=2)")
            except Exception as e:
                print(f"Warning: MetricAttention init failed ({e}), using MultiheadAttention fallback")
                self.metric_attn = nn.MultiheadAttention(d_head, num_heads, batch_first=True)
                self.use_metric_attn = False
        else:
            # Fallback: 표준 attention
            self.metric_attn = nn.MultiheadAttention(d_head, num_heads, batch_first=True)
            self.use_metric_attn = False
        
        # 주제 분류기
        self.topic_classifier = nn.Linear(d_head, num_topics)
        
        # 주제별 앵커 (학습 가능)
        self.topic_anchors = nn.Parameter(torch.randn(num_topics, d_head) * 0.1)
        
        # 주제 이름 매핑 (docs 명세)
        self.topic_names = [
            "chief_complaint",  # 주호소
            "history",          # 병력
            "physical_exam",    # 신체 검사
            "diagnosis",        # 진단
            "treatment_plan",   # 치료 계획
            "prognosis",        # 예후
            "follow_up",        # 추적 관찰
            "general"           # 기타
        ]
    
    def forward(
        self,
        x: torch.Tensor,
        topo_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Args:
            x: [B, T, d_model] 문장 임베딩
            topo_idx: [B, T, K] topology index
        
        Returns:
            P_topic: [B, T, num_topics] 주제 확률
            scores: [B, T] 우선순위 점수
            metric_keys: List[str] 문장별 metric key seed
        """
        B, T, _ = x.shape
        
        # 1. Poincaré embedding
        z = self.poincare_embed(x)  # [B, T, d_head]
        
        # 2. Geodesic attention with SPD Metric Learning
        if hasattr(self, 'use_metric_attn') and self.use_metric_attn:
            # MetricAttention: head-split 필요 (B, H, T, d_h)
            q = self.q_proj(z)  # [B, T, d_head]
            k = self.k_proj(z)
            v = self.v_proj(z)
            
            # Head split: [B, T, d_head] -> [B, H, T, d_h]
            B, T, _ = q.shape
            H = self.num_heads
            d_h = self.d_head_per_head
            
            q = q.view(B, T, H, d_h).transpose(1, 2)  # [B, H, T, d_h]
            k = k.view(B, T, H, d_h).transpose(1, 2)
            v = v.view(B, T, H, d_h).transpose(1, 2)
            
            # Topology index를 Dict 형태로 변환 (MetricAttention 기대 형식)
            topo_dict = None
            if topo_idx is not None:
                # topo_idx: [B, T, K] -> {"neighbor": [B, T, K]}
                topo_dict = {"neighbor": topo_idx}
            
            # MetricAttention forward (SPD Metric Learning!)
            attn_out = self.metric_attn(
                q, k, v,
                topo_idx=topo_dict,
                topk_cfg={"neighbor": topo_idx.shape[-1]} if topo_dict else None
            )  # [B, H, T, d_h]
            
            # Head merge: [B, H, T, d_h] -> [B, T, d_head]
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
            attn_out = self.out_proj(attn_out)
            
            # Attention weights 추정 (간단히 uniform)
            attn_weights = torch.ones(B, T, device=z.device) / T
        elif hasattr(self, 'use_metric_attn') and not self.use_metric_attn:
            # Fallback: 표준 attention
            attn_out, attn_weights = self.metric_attn(z, z, z)
            # attn_weights는 [B, T, T]이므로 평균을 내서 [B, T]로 변환
            if attn_weights.dim() == 3:
                attn_weights = attn_weights.mean(dim=-1)  # [B, T, T] -> [B, T] (마지막 차원 평균)
        else:
            # Legacy fallback
            attn_out, attn_weights = self.metric_attn(z, z, z)
            if attn_weights.dim() == 3:
                attn_weights = attn_weights.mean(dim=-1)
        
        # 3. 우선순위 점수 (attention weight 합)
        if attn_weights.dim() == 3:
            scores = attn_weights.sum(dim=-1)  # [B, T, K] -> [B, T]
        elif attn_weights.dim() == 2:
            scores = attn_weights  # 이미 [B, T]
        else:
            scores = attn_weights.sum(dim=-1)  # fallback
        
        # 4. 주제 분류
        logits = self.topic_classifier(attn_out)  # [B, T, num_topics]
        P_topic = F.softmax(logits, dim=-1)
        
        # 5. Metric key seed 생성
        metric_keys = self._generate_metric_keys(P_topic, scores)
        
        return P_topic, scores, metric_keys
    
    def _generate_metric_keys(
        self,
        P_topic: torch.Tensor,
        scores: torch.Tensor
    ) -> List[str]:
        """
        주제 확률과 score로부터 metric key 생성
        
        docs 명세:
        - 형식: "topic:{topic_name}|priority:{high/medium/low}"
        """
        B, T, _ = P_topic.shape
        keys = []
        
        # scores 차원 확인
        if scores.dim() == 1:
            # [T] -> [1, T]로 reshape
            scores = scores.unsqueeze(0)
        
        for b in range(B):
            for t in range(T):
                # 최고 확률 주제
                top_topic = P_topic[b, t].argmax().item()
                topic_name = self.topic_names[top_topic]
                
                # 우선순위 레벨
                score_val = scores[b, t].item() if b < scores.shape[0] else scores[0, t].item()
                if score_val > 0.7:
                    priority = "high"
                elif score_val > 0.4:
                    priority = "medium"
                else:
                    priority = "low"
                
                key = f"topic:{topic_name}|priority:{priority}"
                keys.append(key)
        
        return keys

