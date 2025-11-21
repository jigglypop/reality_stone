import torch
import torch.nn as nn
import torch.nn.functional as F
from reality_stone import (
    poincare_distance,
    poincare_add
)
from experiments.bank_prediction.models.riemannian_bank_encoder import RiemannianEncoderBlock

class ProductManifoldBankClassifier(nn.Module):
    """
    [SH-PMN] SPD-Hyperbolic Product Manifold Network
    
    User State is represented as a point on M = H^n x S_++^k
    - H (Hyperbolic): Represents hierarchical product preference (Tree structure)
    - S (SPD): Represents financial risk profile/covariance (Ellipsoid structure)
    
    Mathematical Optimization:
    Instead of expensive Matrix Logarithm for Log-Euclidean Distance on SPD,
    we parameterize the SPD matrix P as P = exp(S), where S is a Symmetric Matrix.
    The Log-Euclidean distance between P1 and P2 is then simply ||S1 - S2||_F.
    """
    def __init__(
        self,
        dp_dim=75,
        spd_dim=8,
        hyp_dim=32,
        num_classes=56,
        curvature=-1.0,
        top_k=5
    ):
        super().__init__()
        
        self.c = curvature
        self.hyp_dim = hyp_dim
        self.spd_dim = spd_dim
        self.num_classes = num_classes
        
        # --- 1. Hyperbolic Stream (Product Hierarchy) ---
        # Account Sequence -> Hyperbolic Embedding
        self.digit_embed = nn.Embedding(11, hyp_dim)  # 0-9 + padding
        self.pos_embed = nn.Parameter(torch.randn(14, hyp_dim) * 0.01)
        self.length_embed = nn.Embedding(15, hyp_dim)
        
        # Reuse the Riemannian Encoder Block from previous model
        self.seq_encoder = RiemannianEncoderBlock(
            hyp_dim=hyp_dim,
            spd_dim=spd_dim,
            top_k=top_k,
            c=curvature
        )

        # --- 2. SPD Stream (Financial Status) ---
        # DP Features -> Symmetric Matrix S (Tangent Space of SPD at Identity)
        # We map dp_dim -> spd_dim * spd_dim, then symmetrize
        self.dp_to_sym = nn.Linear(dp_dim, spd_dim * spd_dim)
        self.sym_norm = nn.LayerNorm(spd_dim * spd_dim) # Stabilization

        # --- 3. Product Manifold Prototypes ---
        # Class Prototypes exist on BOTH manifolds (Product Space)
        
        # A. Hyperbolic Prototypes
        self.proto_hyp = nn.Parameter(torch.randn(num_classes, hyp_dim) * 0.01)
        
        # B. SPD Prototypes (Parameterized as Symmetric Matrices S)
        self.proto_sym = nn.Parameter(torch.randn(num_classes, spd_dim * spd_dim) * 0.01)

        # --- 4. Metric Fusion ---
        # Learnable weights for distance fusion
        # d_total^2 = alpha * d_hyp^2 + beta * d_spd^2
        self.alpha = nn.Parameter(torch.tensor(1.0)) # Weight for Hyperbolic
        self.beta = nn.Parameter(torch.tensor(0.5))  # Weight for SPD (start smaller)
        
        # Optional: Mixing Linear Head for residuals
        self.linear_head = nn.Sequential(
            nn.Linear(hyp_dim + spd_dim * spd_dim, hyp_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hyp_dim, num_classes)
        )
        self.logit_mix = nn.Parameter(torch.tensor(0.0)) # Start with geometric dominance

    def get_sequence_embeddings(self, account_digits):
        """Embed account digits into Hyperbolic Sequence"""
        batch_size, seq_len = account_digits.size()
        
        # Embeddings
        digit_emb = self.digit_embed(account_digits) # (B, L, D)
        
        # Position Embeddings (broadcast)
        pos_emb = self.pos_embed.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Mobius Addition for Position Injection
        d_flat = digit_emb.reshape(-1, self.hyp_dim)
        p_flat = pos_emb.reshape(-1, self.hyp_dim)
        
        combined_flat = poincare_add(d_flat, p_flat, c=abs(self.c))
        
        return combined_flat.reshape(batch_size, seq_len, self.hyp_dim)

    def get_spd_symmetry(self, dp_features):
        """
        Map DP features to Symmetric Matrices S
        Note: We work in the Tangent Space (Symmetric Matrices) directly
        which corresponds to Log-Euclidean metric on SPD manifold.
        """
        batch_size = dp_features.size(0)
        raw = self.dp_to_sym(dp_features)
        raw = self.sym_norm(raw)
        
        # Reshape to Matrix
        mat = raw.view(batch_size, self.spd_dim, self.spd_dim)
        
        # Symmetrize: S = (A + A^T) / 2
        sym = (mat + mat.transpose(1, 2)) / 2.0
        
        # Flatten back for efficient distance computation
        return sym.view(batch_size, -1)

    def forward(self, dp_features, account_digits, account_length):
        batch_size = dp_features.size(0)

        # --- Stream 1: Hyperbolic (History & Intent) ---
        seq_hyp = self.get_sequence_embeddings(account_digits)
        h_seq = self.seq_encoder(seq_hyp) # (B, hyp_dim)
        
        # Length Embedding
        len_hyp = torch.tanh(self.length_embed(account_length)) * 0.95
        if len(len_hyp.shape) == 3:
             len_hyp = len_hyp.squeeze(1)
             
        # Combine Sequence + Length in Hyperbolic Space
        h_final = poincare_add(h_seq, len_hyp, c=abs(self.c)) # (B, hyp_dim)

        # --- Stream 2: SPD (Financial Status) ---
        # We use the property: d_SPD(exp(S1), exp(S2)) = ||S1 - S2||_F (Log-Euclidean)
        # So we just predict S and compute Euclidean distance
        s_user = self.get_spd_symmetry(dp_features) # (B, spd_dim^2)

        # --- Distance Computation on Product Manifold ---
        
        # 1. Hyperbolic Distance
        # Expand (B, 1, D) and (1, C, D)
        h_exp = h_final.unsqueeze(1).expand(batch_size, self.num_classes, self.hyp_dim)
        proto_h_exp = self.proto_hyp.unsqueeze(0).expand(batch_size, self.num_classes, self.hyp_dim)
        
        dist_hyp_sq = poincare_distance(
            h_exp.reshape(-1, self.hyp_dim),
            proto_h_exp.reshape(-1, self.hyp_dim),
            c=abs(self.c)
        ).pow(2).view(batch_size, self.num_classes)

        # 2. SPD Distance (Log-Euclidean)
        # User S: (B, D^2)
        # Proto S: (C, D^2) - We ensure proto_sym is symmetric in loss or construction?
        # For simplicity, we let the network learn symmetric-like features or just enforce symmetry here
        # Enforce symmetry on prototypes for correctness
        proto_mat = self.proto_sym.view(self.num_classes, self.spd_dim, self.spd_dim)
        proto_sym_mat = (proto_mat + proto_mat.transpose(1, 2)) / 2.0
        proto_sym_flat = proto_sym_mat.view(self.num_classes, -1)
        
        # Euclidean distance between symmetric matrices
        # (B, 1, D^2) - (1, C, D^2)
        s_diff = s_user.unsqueeze(1) - proto_sym_flat.unsqueeze(0)
        dist_spd_sq = s_diff.pow(2).sum(dim=-1) # Squared Frobenious Norm
        
        # --- Fusion ---
        # Weighted Product Metric
        dist_total_sq = (
            F.softplus(self.alpha) * dist_hyp_sq + 
            F.softplus(self.beta) * dist_spd_sq
        )
        
        logits_geo = -dist_total_sq
        
        # --- Linear Residual (Optional) ---
        # Concatenate features for linear head
        # We map h_final to tangent space for concatenation
        h_tan = torch.atanh(torch.clamp(h_final, -0.95, 0.95))
        features = torch.cat([h_tan, s_user], dim=1)
        logits_linear = self.linear_head(features)
        
        # Mix
        mix = torch.sigmoid(self.logit_mix)
        logits = mix * logits_geo + (1.0 - mix) * logits_linear
        
        return logits

