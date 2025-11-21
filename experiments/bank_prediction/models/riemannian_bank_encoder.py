import torch
import torch.nn as nn
import torch.nn.functional as F
from reality_stone import (
    geodesic_topk_attention,
    PoincareBallLayer,
    poincare_distance,
    poincare_add,
    poincare_scalar_mul
)

class RiemannianEncoderBlock(nn.Module):
    """
    Riemannian-Bellman Encoder Block
    
    Performs:
    1. Geodesic Top-K Attention (Sequence Aggregation)
    2. Hyperbolic Feed-Forward (Feature Transformation)
    """
    def __init__(self, hyp_dim, spd_dim=8, top_k=5, c=-1.0):
        super().__init__()
        self.hyp_dim = hyp_dim
        self.c = c
        self.top_k = top_k
        
        # Metric learning for Geodesic Attention
        # We learn a diagonal metric for efficiency
        self.metric_diag = nn.Parameter(torch.ones(hyp_dim))
        
        # Hyperbolic Feed-Forward Network
        # Operates in tangent space essentially, then projects back
        self.linear1 = nn.Linear(hyp_dim, hyp_dim * 2)
        self.linear2 = nn.Linear(hyp_dim * 2, hyp_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Layer Norm equivalent (optional, simplified scaling here)
        self.norm_scale = nn.Parameter(torch.ones(1))

    def forward(self, x_seq):
        """
        Args:
            x_seq: (batch, seq_len, hyp_dim) - Hyperbolic embeddings sequence
        Returns:
            h_pooled: (batch, hyp_dim) - Aggregated hyperbolic vector
        """
        batch_size, seq_len, _ = x_seq.size()
        
        # 1. Geodesic Top-K Attention (Pooling)
        # Construct metric tensor (batch, dim, dim)
        metric = torch.diag(F.softplus(self.metric_diag) + 1e-5)
        metric = metric.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Aggregate sequence into single vector
        # This finds the geometric "median" of top-k relevant items
        # idx: dummy top-k indices (using all sequence for now since seq_len is short)
        # We need to create a dummy index tensor since the kernel expects it.
        # Let's assume we attend to all items (or top-k if sequence is long)
        # For now, simple full attention simulation
        k_val = min(self.top_k, seq_len)
        
        # Create dummy indices: [batch, seq_len, k] 
        # Here we just select first k items for each query position. 
        # Since we are doing pooling, we treat "query" as each position attending to others?
        # Wait, geodesic_topk_attention in Rust binding expects:
        # q: [B, H, T, d_h]
        # k: [B, H, S, d_h]
        # v: [B, H, S, d_v]
        # idx: [B, T, K]
        # l_factor: [d_h, d_h]
        
        # In our pooling case:
        # Q = x_seq (as single head, or we split heads?)
        # K = x_seq
        # V = x_seq
        
        # Let's simplify: Single head attention for pooling
        # Reshape to add Head dim: (B, 1, L, D)
        x_expanded = x_seq.unsqueeze(1) 
        
        # Indices: We want each token to attend to 'k' other tokens. 
        # For global pooling, we might just want a learnable query? 
        # Or simply, we use the kernel to do self-attention and then mean pool.
        
        # Let's construct a simple "all-to-top-k" index. 
        # Since we don't have pre-computed top-k, we'll just take the first k elements 
        # as a naive approximation or random sampling if we can't compute distances first.
        # Ideally, we should use a kernel that computes top-k on the fly, 
        # but our kernel takes indices as input.
        
        # Fallback: For this task, standard Geodesic Attention might be overkill if we just want pooling.
        # But to use the kernel, we provide indices [0, 1, ..., k-1]
        
        # If k_val is larger than sequence length, we must clamp it
        # But indices should be valid indices in range [0, seq_len-1]
        # Since we use torch.arange(k_val), we must ensure k_val <= seq_len
        # We already did k_val = min(self.top_k, seq_len)
        
        indices = torch.arange(k_val, device=x_seq.device).expand(batch_size, seq_len, k_val)
        
        # Call Rust binding
        # Note: The metric is diagonal, so Cholesky is just sqrt(diag)
        # metric_diag is parameter, we need dense matrix L such that L*L^T = diag
        # So L is just diag(sqrt(metric_diag))
        
        L = torch.diag(torch.sqrt(F.softplus(self.metric_diag) + 1e-5))
        
        # Ensure contiguity
        x_expanded = x_expanded.contiguous()
        indices = indices.contiguous()
        L = L.contiguous()
        
        # Python binding signature: (q, k, v, idx, l_factor, c, tau)
        # IMPORTANT: We MUST detach and move to CPU/Numpy because PyO3 binding expects numpy array
        # UNLESS the binding is written to accept PyTorch tensors via DLPack or similar (which is not yet standard in simple PyO3)
        # However, transferring to CPU -> Rust -> GPU (in binding) is inefficient.
        # IF the binding supports CUDA pointer passing via `__cuda_array_interface__` (numpy-like), it might work?
        # BUT our previous error said "argument 'q': 'Tensor' object cannot be converted to 'PyArray<T, D>'"
        # This implies the binding expects a NumPy array.
        # 
        # CRITICAL FIX: The Rust binding seems to be compiled with `numpy` crate which works with Python lists or NumPy arrays.
        # Since we are running on CUDA, we cannot easily pass CUDA tensor to NumPy without moving to CPU.
        # THIS IS A BOTTLENECK. 
        # Ideally, we should update binding to accept `dlpack` or raw pointers.
        # But given constraints, let's try to be robust.
        
        # The error "CUDA error: an illegal memory access was encountered" usually happens when 
        # we pass a CPU pointer to a CUDA kernel or vice versa, OR index out of bounds.
        # Since we moved data to CPU for the binding call, the Rust side likely received CPU data.
        # BUT if the Rust binding calls `geodesic_topk_attention_cuda` (CUDA Kernel), 
        # it expects DEVICE pointers!
        # 
        # If we pass CPU numpy array to Rust, `as_slice()?.as_ptr()` gives a HOST pointer.
        # If `geodesic_topk_attention_cuda` is a CUDA kernel launch, it needs DEVICE pointers.
        # 
        # CONCLUSION: The Rust binding is flawed for this usage pattern. 
        # It expects PyArray (which is usually CPU) but then calls CUDA kernel with that pointer?
        # UNLESS `numpy` crate handles CUDA arrays (it usually doesn't without cupy).
        #
        # Workaround: We must disable the custom CUDA kernel call for now and use a PyTorch-native implementation
        # OR fix the binding to allocate device memory and copy.
        # Since I cannot change Rust binding compilation easily/safely without risk,
        # I will implement a Pure PyTorch fallback for Geodesic Attention in this file.
        # This is safer and avoids the FFI/pointer hell.
        
        h_attended = self.geodesic_attention_torch(x_expanded, x_expanded, x_expanded, L, k_topk=k_val)
        
        # Squeeze head dim and Mean pool over sequence
        h_pooled = h_attended.squeeze(1).mean(dim=1)
        
        # 2. Hyperbolic Feed-Forward
        # Tangent space approximation: Log -> MLP -> Exp
        # Since inputs are in Poincare ball (-1, 1), atanh maps to R
        
        # Safe atanh
        h_tan = torch.atanh(torch.clamp(h_pooled, -0.95, 0.95))
        
        # MLP
        out = F.gelu(self.linear1(h_tan))
        out = self.dropout(out)
        out = self.linear2(out)
        
        # Residual connection in tangent space (if dims matched)
        out = out + h_tan
        
        # Exp map (Tanh)
        h_out = torch.tanh(out) * 0.95
        
        return h_out

    def geodesic_attention_torch(self, q, k, v, L, k_topk):
        """
        Pure PyTorch implementation of Geodesic Attention
        q: (B, 1, T, D)
        k: (B, 1, S, D)
        v: (B, 1, S, D)
        L: (D, D) - Cholesky factor of Metric
        """
        B, H, T, D = q.shape
        _, _, S, _ = k.shape
        
        # 1. Apply Metric L
        # q' = q @ L.T
        q_trans = torch.matmul(q, L.t()) 
        k_trans = torch.matmul(k, L.t())
        
        # 2. Compute Poincare Distance matrix
        # We need distance between every q_i and k_j
        # Expanded for broadcasting: (B, 1, T, S, D)
        q_exp = q_trans.unsqueeze(3) 
        k_exp = k_trans.unsqueeze(2)
        
        # Euclidean distance squared in transformed space (Tangent approx or actual Poincare?)
        # The kernel used Poincare distance. Let's use Mobius addition approximation for speed
        # or exact formula.
        # d(x,y) = acosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
        # Here x, y are already transformed by L? No, L defines the metric at origin.
        # If we assume L transforms to a space where Euclidean distance approximates Geodesic, 
        # we can use simple Euclidean attention.
        # BUT the paper says Geodesic Attention.
        
        # Let's implement the exact Poincare distance on the last dim
        diff = q_exp - k_exp
        dist_sq = diff.pow(2).sum(dim=-1) # (B, 1, T, S)
        
        q_norm_sq = q_trans.pow(2).sum(dim=-1, keepdim=True) # (B, 1, T, 1)
        k_norm_sq = k_trans.pow(2).sum(dim=-1, keepdim=True).transpose(-1, -2) # (B, 1, 1, S)
        
        denom = (1 - q_norm_sq) * (1 - k_norm_sq)
        denom = torch.clamp(denom, min=1e-7)
        
        arg = 1 + 2 * dist_sq / denom
        dist = torch.acosh(torch.clamp(arg, min=1.0 + 1e-7))
        
        # 3. Top-K Masking (optional, or just Softmax over all)
        # If k is small, we mask others.
        if k_topk < S:
            topk_vals, topk_idx = torch.topk(dist, k_topk, dim=-1, largest=False) # Smallest distance
            # We only attend to these. 
            # Simplified: just run softmax on neg distance of topk
            scores = -topk_vals.pow(2) # Score = -dist^2
            attn_weights = F.softmax(scores, dim=-1) # (B, 1, T, K)
            
            # Gather values
            # v: (B, 1, S, D) -> expand to (B, 1, T, S, D) ?? Too big.
            # Gather V corresponding to topk_idx
            # topk_idx: (B, 1, T, K)
            # We need to gather from V along dim 2 (S dimension)
            # V is (B, 1, S, D). We want (B, 1, T, K, D)
            
            # Indexing trick
            # topk_idx: (B, 1, T, K)
            # We need to expand to (B, 1, T, K, D)
            idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)
            v_expanded = v.unsqueeze(2).expand(-1, -1, T, -1, -1) # (B, 1, T, S, D)
            v_selected = torch.gather(v_expanded, 3, idx_expanded) # (B, 1, T, K, D)
            
            out = (attn_weights.unsqueeze(-1) * v_selected).sum(dim=3) # (B, 1, T, D)
        else:
            scores = -dist.pow(2)
            attn_weights = F.softmax(scores, dim=-1)
            out = torch.matmul(attn_weights, v) # (B, 1, T, S) @ (B, 1, S, D) -> (B, 1, T, D)
            
        return out

class RiemannianBankEncoder(nn.Module):
    """
    Bank Intent & Account Prediction Model using Riemannian-Bellman Encoder Architecture
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
        self.num_classes = num_classes
        self.spd_dim = spd_dim
        
        # --- 1. Embedding Layers ---
        
        # DP Feature Embedding: Vector -> SPD -> Hyperbolic
        self.dp_to_spd_params = nn.Linear(dp_dim, spd_dim * spd_dim)
        self.spd_to_hyp = nn.Linear(spd_dim * spd_dim, hyp_dim)
        
        # Sequence Embedding
        self.digit_embed = nn.Embedding(11, hyp_dim)  # 0-9 + padding
        self.pos_embed = nn.Parameter(torch.randn(14, hyp_dim) * 0.01)
        
        # Length Embedding
        self.length_embed = nn.Embedding(15, hyp_dim)
        
        # --- 2. Encoder Blocks ---
        
        # Main Encoder Block for Account Sequence
        self.seq_encoder = RiemannianEncoderBlock(
            hyp_dim=hyp_dim,
            spd_dim=spd_dim,
            top_k=top_k,
            c=curvature
        )
        
        # --- 3. Geometric Classification ---
        
        # Bank Prototypes in Hyperbolic Space
        self.bank_prototypes = nn.Parameter(
            torch.randn(num_classes, hyp_dim) * 0.01
        )
        
        # Learnable per-class scaling for distance
        self.scale = nn.Parameter(torch.ones(num_classes))
        # Mixture weights for product-manifold style distance
        self.mix_w = nn.Parameter(torch.tensor([0.5, 0.25, 0.25]))
        self.linear_head = nn.Sequential(
            nn.Linear(hyp_dim, hyp_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hyp_dim, num_classes)
        )
        self.logit_mix = nn.Parameter(torch.tensor(0.5))

    def dp_to_spd(self, dp_features):
        """Map DP features to SPD matrices"""
        batch_size = dp_features.size(0)
        params = self.dp_to_spd_params(dp_features)
        A = params.view(batch_size, self.spd_dim, self.spd_dim)
        spd = torch.bmm(A.transpose(1, 2), A)
        eye = torch.eye(self.spd_dim, device=spd.device).unsqueeze(0)
        return spd + 1e-5 * eye # Ensure positive definite

    def get_sequence_embeddings(self, account_digits):
        """Embed account digits into Hyperbolic Sequence"""
        batch_size, seq_len = account_digits.size()
        
        # Embeddings
        digit_emb = self.digit_embed(account_digits) # (B, L, D)
        
        # Position Embeddings (broadcast)
        pos_emb = self.pos_embed.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Mobius Addition for Position Injection
        # x +_c p
        # We process this in a loop or batched operation if poincare_add supports broadcasting
        # poincare_add typically expects same shape.
        
        # Flatten for batch addition
        d_flat = digit_emb.reshape(-1, self.hyp_dim)
        p_flat = pos_emb.reshape(-1, self.hyp_dim)
        
        combined_flat = poincare_add(d_flat, p_flat, c=abs(self.c))
        
        return combined_flat.reshape(batch_size, seq_len, self.hyp_dim)

    def forward(self, dp_features, account_digits, account_length):
        batch_size = dp_features.size(0)
        
        # 1. DP Features path
        spd_matrices = self.dp_to_spd(dp_features)
        spd_flat = spd_matrices.view(batch_size, -1)
        
        # Project SPD to Hyperbolic (Tangent -> Exp)
        dp_hyp = torch.tanh(self.spd_to_hyp(spd_flat)) * 0.95
        
        # 2. Account Sequence path
        seq_hyp = self.get_sequence_embeddings(account_digits)
        
        # Apply Riemannian Encoder Block
        # (B, L, D) -> (B, D)
        seq_encoded = self.seq_encoder(seq_hyp)
        
        # 3. Length Embedding
        len_hyp = torch.tanh(self.length_embed(account_length)) * 0.95
        if len(len_hyp.shape) == 3: # Sometimes embedding returns (B, 1, D)
             len_hyp = len_hyp.squeeze(1)

        # 4. Fusion (Möbius Addition)
        # Combine: DP + Sequence + Length
        # h_final = ((dp + seq) + len)
        
        combined = poincare_add(dp_hyp, seq_encoded, c=abs(self.c))
        combined = poincare_add(combined, len_hyp, c=abs(self.c))
        
        # 5. Classification (Distance to Prototypes)
        # Expand for pairwise distance
        combined_expanded = combined.unsqueeze(1).expand(
            batch_size, self.num_classes, self.hyp_dim
        )
        prototypes_expanded = self.bank_prototypes.unsqueeze(0).expand(
            batch_size, self.num_classes, self.hyp_dim
        )
        
        # Hyperbolic Distance (combined)
        d_comb = poincare_distance(
            combined_expanded.reshape(-1, self.hyp_dim),
            prototypes_expanded.reshape(-1, self.hyp_dim),
            c=abs(self.c)
        ).reshape(batch_size, self.num_classes)
        # Additional distances for product-style mixing
        dp_exp = dp_hyp.unsqueeze(1).expand(batch_size, self.num_classes, self.hyp_dim)
        seq_exp = seq_encoded.unsqueeze(1).expand(batch_size, self.num_classes, self.hyp_dim)
        d_dp = poincare_distance(
            dp_exp.reshape(-1, self.hyp_dim),
            prototypes_expanded.reshape(-1, self.hyp_dim),
            c=abs(self.c)
        ).reshape(batch_size, self.num_classes)
        d_seq = poincare_distance(
            seq_exp.reshape(-1, self.hyp_dim),
            prototypes_expanded.reshape(-1, self.hyp_dim),
            c=abs(self.c)
        ).reshape(batch_size, self.num_classes)
        w = torch.softmax(self.mix_w, dim=0)
        dists = w[0] * d_comb + w[1] * d_dp + w[2] * d_seq
        
        logits_proto = -dists * F.softplus(self.scale).unsqueeze(0)
        logits_linear = self.linear_head(combined)
        mix = torch.sigmoid(self.logit_mix)
        logits = mix * logits_proto + (1.0 - mix) * logits_linear
        
        return logits
    
    def get_embeddings(self, dp_features, account_digits, account_length):
        """Extract internal embeddings for visualization"""
        with torch.no_grad():
            batch_size = dp_features.size(0)
            
            spd_matrices = self.dp_to_spd(dp_features)
            spd_flat = spd_matrices.view(batch_size, -1)
            dp_hyp = torch.tanh(self.spd_to_hyp(spd_flat)) * 0.95
            
            seq_hyp = self.get_sequence_embeddings(account_digits)
            seq_encoded = self.seq_encoder(seq_hyp)
            len_hyp = torch.tanh(self.length_embed(account_length)) * 0.95
            if len(len_hyp.shape) == 3:
                 len_hyp = len_hyp.squeeze(1)
            
            combined = poincare_add(dp_hyp, seq_encoded, c=abs(self.c))
            combined = poincare_add(combined, len_hyp, c=abs(self.c))
            
            return combined
