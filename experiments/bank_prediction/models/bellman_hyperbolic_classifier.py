import torch
import torch.nn as nn
import torch.nn.functional as F
from reality_stone import KleinLayer, klein_distance


class BellmanHyperbolicClassifier(nn.Module):
    """
    벨만-하이퍼볼릭 분류기
    
    전략: 기존 HyperbolicBankClassifier의 강력한 인코더는 그대로 쓰되,
    분류 헤드만 "가치 함수(Value Function) 기반"으로 교체
    
    핵심 아이디어:
    - 각 은행(클래스)에 대한 "가치 V(s, bank)"를 학습
    - 거리가 아닌 "기대 보상(Expected Reward)"으로 분류
    - 벨만 일관성: V(s) ≈ R + γV(s')
    """
    
    def __init__(
        self,
        dp_dim=75,
        hyp_dim=32,
        num_classes=54,
        curvature=1.0,
        gamma=0.99,
        use_value_head=True
    ):
        super().__init__()
        
        self.c = abs(curvature)
        self.hyp_dim = hyp_dim
        self.num_classes = num_classes
        self.gamma = gamma
        self.use_value_head = use_value_head
        
        # ===== 인코더: 기존 HyperbolicBankClassifier와 동일 =====
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
        
        self.dp_encoder = nn.Sequential(
            nn.Linear(dp_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.combiner = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, hyp_dim)
        )
        
        self.dropout = nn.Dropout(0.3)
        
        # ===== 분류 헤드: 벨만 방식 =====
        if use_value_head:
            # 가치 함수 기반 분류
            self.value_network = nn.Sequential(
                nn.Linear(hyp_dim, hyp_dim * 2),
                nn.LayerNorm(hyp_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hyp_dim * 2, num_classes)
            )
        else:
            # 기존 거리 기반 (비교용)
            self.bank_prototypes_tan = nn.Parameter(
                torch.randn(num_classes, hyp_dim) * 0.1
            )
            self.logit_scale = nn.Parameter(torch.tensor(5.0))
        
        # Auxiliary classifier
        self.euclidean_head = nn.Linear(hyp_dim, num_classes)
        
    def to_klein(self, x):
        """유클리드 벡터 -> 클라인 공간"""
        x_tanh = torch.tanh(x)
        return KleinLayer.apply(x_tanh, x_tanh, self.c, 0.5)
    
    def forward(self, dp_features, account_digits, account_length, return_embedding=False):
        batch_size = dp_features.size(0)
        
        # ===== 인코딩 (기존과 동일) =====
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
        
        latent_euclid = self.combiner(combined)
        latent_euclid = self.dropout(latent_euclid)
        
        # ===== Klein 공간으로 매핑 =====
        z_klein = self.to_klein(latent_euclid)
        
        # ===== 분류 헤드 선택 =====
        if self.use_value_head:
            # **벨만 방식**: 가치 함수로 직접 로짓 출력
            # V(s, a) = "상태 s에서 은행 a를 선택했을 때 기대되는 장기 보상"
            logits_value = self.value_network(z_klein)
            
            # Auxiliary Euclidean 분류기
            logits_aux = self.euclidean_head(latent_euclid)
            
            # 결합: 가치 기반(70%) + 유클리드(30%)
            logits = 0.7 * logits_value + 0.3 * logits_aux
        else:
            # **기존 방식**: 거리 기반
            prototypes = self.to_klein(self.bank_prototypes_tan)
            
            z_expanded = z_klein.unsqueeze(1).expand(batch_size, prototypes.size(0), -1)
            proto_expanded = prototypes.unsqueeze(0).expand(batch_size, prototypes.size(0), -1)
            
            z_flat = z_expanded.reshape(-1, self.hyp_dim)
            proto_flat = proto_expanded.reshape(-1, self.hyp_dim)
            
            dists = klein_distance(z_flat, proto_flat, c=self.c)
            dists = dists.reshape(batch_size, -1)
            
            logits_hyp = -dists * self.logit_scale
            logits_aux = self.euclidean_head(latent_euclid)
            
            logits = logits_hyp + logits_aux
        
        if return_embedding:
            return logits, z_klein
        return logits


class BellmanConsistencyLoss(nn.Module):
    """
    벨만 일관성 손실 (선택적)
    
    L = L_cls + λ * L_bellman
    
    where L_bellman = (V(s) - (R + γ max_a' V(s')))²
    
    이 버전에서는 단순화:
    - R = 1 if correct, 0 otherwise (즉각 보상)
    - V(s') = detach된 다음 배치의 평균 (근사)
    """
    
    def __init__(self, lambda_bellman=0.1, gamma=0.99):
        super().__init__()
        self.lambda_bellman = lambda_bellman
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels, apply_bellman=True):
        # 분류 손실
        loss_cls = self.ce_loss(logits, labels)
        
        if not apply_bellman or self.lambda_bellman == 0:
            return {
                'total': loss_cls,
                'classification': loss_cls,
                'bellman': torch.tensor(0.0, device=logits.device)
            }
        
        # 벨만 일관성 손실
        # V(s, a) = logits[range(B), labels] (선택한 액션의 가치)
        batch_size = logits.size(0)
        current_values = logits[range(batch_size), labels]
        
        # 즉각 보상: 정답이면 1, 아니면 0
        preds = logits.argmax(dim=1)
        rewards = (preds == labels).float()
        
        # 다음 상태의 가치: max_a V(s', a) (greedy)
        # 학습 안정성을 위해 detach
        next_values = logits.max(dim=1)[0].detach()
        
        # 벨만 에러: V(s,a) - (R + γ max V(s'))
        bellman_error = (current_values - (rewards + self.gamma * next_values)).pow(2).mean()
        
        total_loss = loss_cls + self.lambda_bellman * bellman_error
        
        return {
            'total': total_loss,
            'classification': loss_cls,
            'bellman': bellman_error
        }

