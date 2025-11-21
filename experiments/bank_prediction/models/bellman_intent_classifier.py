import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

try:
    from reality_stone.layers.metric_attention import SPDMetric
    HAS_SPD = True
except ImportError:
    HAS_SPD = False
    print("Warning: SPDMetric not available, using simplified version")


class BellmanIntentClassifier(nn.Module):
    """
    Bellman-Lagrangian 기반 의도 분류 모델
    
    핵심 수학:
    1. 벨만 방정식: V(s) = max_a[R(s,a) + γE[V(s')]]
    2. 잠재 에너지: V(x) = -Q*(s,a) (가치가 높으면 에너지가 낮음)
    3. 라그랑지안: L = T - V = (1/2)g_ij dx^i dx^j - V(x)
    4. 측지선: 최소 작용 원리에 따라 에너지를 최소화하는 경로
    
    아키텍처:
    - Input: DP features (75d) + Account digit sequence (14 digits)
    - Stage 1: Manifold Embedding (Euclidean -> Hyperbolic)
    - Stage 2: Geodesic Attention with SPD Metric
    - Stage 3: Value-driven Geodesic Flow
    - Stage 4: Prototype-based Classification
    """
    
    def __init__(
        self,
        dp_dim: int = 75,
        embed_dim: int = 64,
        hyp_dim: int = 32,
        num_classes: int = 54,
        num_layers: int = 2,
        curvature: float = 1.0,
        gamma: float = 0.99,
        use_spd: bool = True
    ):
        super().__init__()
        
        self.dp_dim = dp_dim
        self.embed_dim = embed_dim
        self.hyp_dim = hyp_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.c = abs(curvature)
        self.gamma = gamma
        self.use_spd = use_spd and HAS_SPD
        
        self.dp_to_euclidean = nn.Sequential(
            nn.Linear(dp_dim, 128),
            nn.LayerNorm(128),  # DP 쪽은 비교적 안정적인 정규화 사용
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.digit_embed = nn.Embedding(12, embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(14, embed_dim) * 0.02)

        # 하이퍼볼릭 베이스라인과 유사한 시퀀스 인코더 (경량 LSTM)
        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.embed_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.len_embed = nn.Embedding(16, self.embed_dim)
        
        self.seq_projector = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        if self.use_spd:
            self.spd_metric = SPDMetric(
                d=hyp_dim,
                rank=hyp_dim // 4,
                init_scale=1.0
            )
        
        self.to_hyperbolic = nn.Linear(embed_dim, hyp_dim)
        
        self.geodesic_layers = nn.ModuleList([
            GeodesicValueLayer(
                dim=hyp_dim,
                num_heads=4,
                curvature=self.c,
                dropout=0.1
            ) for _ in range(num_layers)
        ])
        
        self.value_network = ValueFunction(
            input_dim=hyp_dim,
            hidden_dim=hyp_dim * 2,
            curvature=self.c
        )
        
        self.bank_prototypes = nn.Parameter(
            torch.randn(num_classes, hyp_dim) * 0.1
        )
        
        self.temperature = nn.Parameter(torch.tensor(10.0))
        
        self.aux_classifier = nn.Linear(hyp_dim, num_classes)
        
    def euclidean_to_poincare(self, x: torch.Tensor) -> torch.Tensor:
        """유클리드 -> 푸앵카레 볼 매핑 (지수 맵 근사)"""
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = x / norm
        scale = torch.tanh(norm)
        return normalized * scale * 0.95
        
    def poincare_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """푸앵카레 볼 거리 (Vectorized)
        d(x,y) = arcosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))
        
        Args:
            x: (B, D) or (B, 1, D)
            y: (B, D) or (1, C, D) or (B, C, D)
        """
        # Squared Euclidean distance
        diff_norm_sq = (x - y).pow(2).sum(dim=-1)
        
        # Norm squared
        x_norm_sq = x.pow(2).sum(dim=-1).clamp(max=0.99)
        y_norm_sq = y.pow(2).sum(dim=-1).clamp(max=0.99)
        
        # Denominator
        denom = (1 - self.c * x_norm_sq) * (1 - self.c * y_norm_sq)
        denom = denom.clamp(min=1e-8)
        
        # Argument for arccosh
        arg = 1.0 + 2.0 * self.c * diff_norm_sq / denom
        arg = arg.clamp(min=1.0 + 1e-8)
        
        return torch.acosh(arg) / math.sqrt(self.c)
    
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """뫼비우스 덧셈 (Poincare ball)"""
        x_norm_sq = x.pow(2).sum(dim=-1, keepdim=True).clamp(max=0.99)
        y_norm_sq = y.pow(2).sum(dim=-1, keepdim=True).clamp(max=0.99)
        xy_inner = (x * y).sum(dim=-1, keepdim=True)
        
        numerator = (1 + 2 * self.c * xy_inner + self.c * y_norm_sq) * x + (1 - self.c * x_norm_sq) * y
        denominator = 1 + 2 * self.c * xy_inner + self.c * self.c * x_norm_sq * y_norm_sq
        denominator = denominator.clamp(min=1e-8)
        
        result = numerator / denominator
        return result * 0.95
        
    def forward(
        self,
        dp_features: torch.Tensor,
        account_digits: torch.Tensor,
        account_length: torch.Tensor,
        return_values: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            dp_features: (B, 75)
            account_digits: (B, 14) - padded digit sequence
            account_length: (B,)
            return_values: 가치 함수 값도 반환할지 여부
            
        Returns:
            logits: (B, num_classes)
            values: (B,) if return_values else None
        """
        batch_size = dp_features.size(0)
        
        # DP 특성 인코딩
        dp_feat = self.dp_to_euclidean(dp_features)  # (B, D)
        
        # 계좌번호 시퀀스 인코딩 (LSTM + 길이 임베딩)
        digit_emb = self.digit_embed(account_digits)               # (B, T, D)
        pos = self.pos_embed.unsqueeze(0).expand(batch_size, -1, -1)
        seq_in = digit_emb + pos

        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(seq_in)
        seq_feat = h_n[-1]                                        # (B, D)

        len_idx = account_length.squeeze().clamp(0, 15)
        len_feat = self.len_embed(len_idx)                        # (B, D)
        seq_final = seq_feat + len_feat
        
        combined = torch.cat([dp_feat, seq_final], dim=-1)
        euclidean_feat = self.seq_projector(combined)
        
        h_tan = self.to_hyperbolic(euclidean_feat)
        
        h = self.euclidean_to_poincare(h_tan)
        
        if self.use_spd:
            metric = self.spd_metric.get_metric()
        else:
            metric = None
        
        for layer in self.geodesic_layers:
            h = layer(h, metric=metric)
        
        values = self.value_network(h)
        
        h_final = self.geodesic_flow_with_value(h, values)
        
        # Prototypes: (num_classes, D)
        prototypes = self.euclidean_to_poincare(self.bank_prototypes)
        
        # Vectorized Distance Calculation
        # h_final: (B, D) -> (B, 1, D)
        # prototypes: (C, D) -> (1, C, D)
        h_exp = h_final.unsqueeze(1)
        proto_exp = prototypes.unsqueeze(0)
        
        # distances: (B, C)
        distances = self.poincare_distance(h_exp, proto_exp)
        
        logits_hyp = -distances * self.temperature
        
        logits_aux = self.aux_classifier(h_final)
        
        logits = logits_hyp + 0.3 * logits_aux
        
        if return_values:
            return logits, values
        return logits, None
    
    def geodesic_flow_with_value(
        self,
        h: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """
        가치 함수 기반 측지 흐름
        
        V(x) = -Q*(x) (잠재 에너지)
        표현은 에너지 그라디언트의 반대 방향(높은 가치 방향)으로 이동
        """
        h_norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        h_normalized = h / h_norm
        
        value_magnitude = torch.sigmoid(values).unsqueeze(-1)
        
        direction = h_normalized * value_magnitude * 0.1
        
        h_updated = self.mobius_add(h, direction)
        
        return h_updated


class GeodesicValueLayer(nn.Module):
    """
    측지 어텐션 + 가치 함수 레이어
    
    Self-attention을 측지선 거리 기반으로 수행하고,
    가치 함수로 정보 흐름 방향을 조정
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        curvature: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.c = abs(curvature)
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        metric: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, D) - 각 샘플은 하나의 벡터
            metric: (D, D) - SPD 메트릭 (선택)
        """
        residual = x
        x = self.norm1(x)
        
        if metric is not None:
            L = torch.linalg.cholesky(metric + torch.eye(metric.size(0), device=metric.device) * 1e-6)
            x_transformed = x @ L.t()
        else:
            x_transformed = x
        
        x_out = x_transformed
        
        x = self.to_out(x_out)
        x = self.dropout(x)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual
        
        return x


class ValueFunction(nn.Module):
    """
    벨만 가치 함수
    
    V(h) = 현재 상태 h에서 기대할 수 있는 미래 보상의 총합
    
    의도 분류 맥락에서:
    - 정확한 클래스에 가까울수록 높은 가치
    - 프로토타입과의 거리가 가까울수록 높은 가치
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        curvature: float = 1.0
    ):
        super().__init__()
        
        self.c = abs(curvature)
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, D) - 하이퍼볼릭 공간의 표현
            
        Returns:
            values: (B,) - 각 샘플의 가치 함수 값
        """
        h_norm = h.norm(dim=-1, keepdim=True)
        
        curvature_factor = 1.0 / (1.0 - self.c * h_norm.pow(2).clamp(max=0.99))
        h_scaled = h * curvature_factor
        
        values = self.network(h_scaled).squeeze(-1)
        
        return values

