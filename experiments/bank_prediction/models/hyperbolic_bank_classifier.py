import torch
import torch.nn as nn
import torch.nn.functional as F
from reality_stone import (
    KleinLayer,
    poincare_add,
    klein_distance
)

# -----------------------------------------------------------------------------
# Legacy Classes for Compatibility (Do not remove to avoid ImportErrors)
# -----------------------------------------------------------------------------

class HyperbolicDPEncoder(nn.Module):
    """(Legacy) DP 특성(75차원)을 쌍곡 공간으로 인코딩"""
    def __init__(self, dp_dim=75, hyp_dim=32, curvature=1.0):
        super().__init__()
        self.dp_to_hyp = nn.Linear(dp_dim, hyp_dim)
        self.c = abs(curvature)
        
    def forward(self, dp_features):
        h = torch.tanh(self.dp_to_hyp(dp_features)) * 0.9
        h = KleinLayer.apply(h, h, self.c, 0.5)
        return h


class HyperbolicPositionalEncoding(nn.Module):
    """(Legacy) 계좌번호 각 자리를 쌍곡 공간에서 인코딩"""
    def __init__(self, max_len=14, d_model=32, curvature=1.0):
        super().__init__()
        self.position_embeddings = nn.Parameter(
            torch.randn(max_len, d_model) * 0.3
        )
        self.digit_embeddings = nn.Parameter(
            torch.randn(11, d_model) * 0.3
        )
        self.c = abs(curvature)
        self.d_model = d_model
    
    def forward(self, account_digits):
        batch_size = account_digits.size(0)
        seq_len = account_digits.size(1)
        pos_emb = self.position_embeddings[:seq_len]
        digit_emb = self.digit_embeddings[account_digits]
        pos = pos_emb.unsqueeze(0).expand(batch_size, seq_len, self.d_model)
        pos_flat = pos.reshape(-1, self.d_model)
        digit_flat = digit_emb.reshape(-1, self.d_model)
        combined_flat = poincare_add(pos_flat, digit_flat, c=self.c)
        combined = combined_flat.view(batch_size, seq_len, self.d_model)
        return combined


# -----------------------------------------------------------------------------
# New Improved Model (Lorentz Version)
# -----------------------------------------------------------------------------

class HyperbolicBankClassifier(nn.Module):
    """
    Reality Stone 기반 은행 분류 모델 (Klein Model Version)
    
    사용자 요청: "푸앵카레 말고 로렌츠로 바꾸자"
    - Poincare Ball 모델은 시각화에 좋지만, 경계면 근처에서 수치적 불안정성이 있음
    - Lorentz(Hyperboloid) 모델은 수치적으로 훨씬 안정적이며 그라디언트 소실이 적음
    """
    
    def __init__(
        self,
        dp_dim=75,
        hyp_dim=32,
        num_classes=54,
        curvature=1.0,
        num_heads=4 # 호환성을 위해 인자는 유지하되 사용하지 않음
    ):
        super().__init__()
        
        self.c = abs(curvature)
        self.hyp_dim = hyp_dim
        
        # ---------------------------------------------------------
        # 1. 숫자 시퀀스 처리 (Text/NLP 방식)
        # ---------------------------------------------------------
        self.vocab_size = 12
        self.embedding_dim = 64
        
        self.digit_embed = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.len_embed = nn.Embedding(16, 128)
        
        # ---------------------------------------------------------
        # 2. DP(Design Pattern) 특성 처리
        # ---------------------------------------------------------
        self.dp_encoder = nn.Sequential(
            nn.Linear(dp_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # ---------------------------------------------------------
        # 3. 통합 및 잠재 벡터 생성
        # ---------------------------------------------------------
        self.combiner = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, hyp_dim)
        )
        
        # ---------------------------------------------------------
        # 4. 클라인 분류기 (Klein Classifier)
        # ---------------------------------------------------------
        # Tangent Space에 파라미터를 두고 Forward 시 로렌츠 공간으로 사영
        self.bank_prototypes_tan = nn.Parameter(
            torch.randn(num_classes, hyp_dim) * 0.1
        )
        
        self.logit_scale = nn.Parameter(torch.tensor(5.0))
        self.dropout = nn.Dropout(0.3)
        
        # ---------------------------------------------------------
        # 5. Auxiliary Euclidean Classifier
        # ---------------------------------------------------------
        self.euclidean_head = nn.Linear(hyp_dim, num_classes)
        
    def to_klein(self, x):
        """유클리드 벡터 -> 클라인(벨트라미) 공간 매핑 (KleinLayer 사용)"""
        x_tanh = torch.tanh(x)
        return KleinLayer.apply(x_tanh, x_tanh, self.c, 0.5)

    def forward(self, dp_features, account_digits, account_length):
        batch_size = dp_features.size(0)
        
        # 1. Feature Extraction (Euclidean)
        emb = self.digit_embed(account_digits)
        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(emb)
        seq_feat = h_n[-1]
        
        len_idx = account_length.squeeze().clamp(0, 15)
        len_feat = self.len_embed(len_idx)
        seq_final = seq_feat + len_feat
        
        dp_feat = self.dp_encoder(dp_features)
        combined = torch.cat([seq_final, dp_feat], dim=1)
        combined = self.dropout(combined)
        
        # 2. Latent Vector (Tangent Space)
        latent_euclid = self.combiner(combined)
        latent_euclid = self.dropout(latent_euclid)
        
        # --- Path A: Euclidean Classification ---
        logits_aux = self.euclidean_head(latent_euclid)
        
        # --- Path B: Klein Classification ---
        # 3. Mapping to Klein Space (B^n)
        z = self.to_klein(latent_euclid)           # [Batch, HypDim]
        prototypes = self.to_klein(self.bank_prototypes_tan)  # [Classes, HypDim]
        
        # 4. Klein Distance
        z_expanded = z.unsqueeze(1).expand(batch_size, prototypes.size(0), -1)
        proto_expanded = prototypes.unsqueeze(0).expand(batch_size, prototypes.size(0), -1)
        
        z_flat = z_expanded.reshape(-1, self.hyp_dim)
        proto_flat = proto_expanded.reshape(-1, self.hyp_dim)
        
        dists = klein_distance(z_flat, proto_flat, c=self.c)
        dists = dists.reshape(batch_size, -1)
 
        logits_hyp = -dists * self.logit_scale
 
        return logits_hyp + logits_aux
