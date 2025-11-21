import torch
import torch.nn as nn
from reality_stone import PoincareBallLayer, poincare_distance, poincare_add
from .hyperbolic_bank_classifier import HyperbolicPositionalEncoding


class AdaptiveCurvatureBankClassifier(nn.Module):
    """
    동적 곡률 버전 - 각 레이어가 다른 계층 깊이 학습
    """
    
    def __init__(
        self,
        dp_dim=75,
        hyp_dim=32,
        num_classes=54,
        num_layers=3,
        c_min=-2.0,
        c_max=-0.1,
        num_heads=4
    ):
        super().__init__()
        
        self.dp_to_hyp = nn.Linear(dp_dim, hyp_dim)
        
        self.pos_encoder = HyperbolicPositionalEncoding(
            max_len=14,
            d_model=hyp_dim,
            curvature=-1.0
        )
        
        self.length_embed = nn.Embedding(15, hyp_dim)
        
        self.kappas = nn.Parameter(torch.zeros(num_layers))
        
        self.hyp_attention = nn.MultiheadAttention(
            embed_dim=hyp_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.bank_prototypes = nn.Parameter(
            torch.randn(num_classes, hyp_dim) * 0.01
        )
        
        self.c_min = c_min
        self.c_max = c_max
        self.num_layers = num_layers
        self.hyp_dim = hyp_dim
    
    def forward(self, dp_features, account_digits, account_length):
        batch_size = dp_features.size(0)
        
        h = torch.tanh(self.dp_to_hyp(dp_features)) * 0.9
        
        pos_hyp = self.pos_encoder(account_digits)
        attn_out, _ = self.hyp_attention(pos_hyp, pos_hyp, pos_hyp)
        seq_repr = attn_out.mean(dim=1)
        
        len_hyp = self.length_embed(account_length)
        
        combined = h
        for component in [seq_repr, len_hyp]:
            combined = poincare_add(combined, component, c=1.0)
        
        for i in range(self.num_layers):
            combined = PoincareBallLayer.apply(
                combined, combined,
                None,
                0.5,
                self.kappas,
                i,
                self.c_min,
                self.c_max
            )
        
        combined_expanded = combined.unsqueeze(1).expand(
            batch_size, self.bank_prototypes.size(0), self.hyp_dim
        )
        prototypes_expanded = self.bank_prototypes.unsqueeze(0).expand(
            batch_size, self.bank_prototypes.size(0), self.hyp_dim
        )
        
        distances = poincare_distance(
            combined_expanded.reshape(-1, self.hyp_dim),
            prototypes_expanded.reshape(-1, self.hyp_dim),
            c=1.0
        ).reshape(batch_size, -1)
        
        logits = -distances
        
        return logits

