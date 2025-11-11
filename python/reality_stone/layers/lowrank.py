import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class SharedBasis:
    P: torch.Tensor  # [in_features, r]
    Q: torch.Tensor  # [out_features, r]

    def to(self, device: torch.device, dtype: torch.dtype) -> 'SharedBasis':
        return SharedBasis(P=self.P.to(device=device, dtype=dtype), Q=self.Q.to(device=device, dtype=dtype))


class LowRankLinear(nn.Module):
    """
    Linear layer parameterized as W^T = P Σ^T Q^T, with P:[in,r], Q:[out,r], Σ:[r,r].
    Forward uses y = ((x @ P) @ Σ^T) @ Q^T + b.
    Storage can be shared for P,Q across layers; Σ is per-layer small matrix.
    """

    def __init__(self, in_features: int, out_features: int, r: int, shared: Optional[SharedBasis] = None, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.shared = shared

        if shared is None:
            # own bases
            self.P = nn.Parameter(torch.empty(in_features, r))
            self.Q = nn.Parameter(torch.empty(out_features, r))
        else:
            self.register_buffer('P_buf', shared.P.clone(), persistent=False)
            self.register_buffer('Q_buf', shared.Q.clone(), persistent=False)
            self.P = None
            self.Q = None

        # per-layer small matrix Σ
        self.Sigma = nn.Parameter(torch.empty(r, r))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.P is not None:
                nn.init.orthogonal_(self.P)
            if self.Q is not None:
                nn.init.orthogonal_(self.Q)
            # small diag init
            self.Sigma.copy_(torch.eye(self.r) * 0.02)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def bases(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.shared is None:
            return self.P, self.Q
        else:
            return self.P_buf, self.Q_buf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        P, Q = self.bases()
        z = x.matmul(P)              # [B, r]
        z = z.matmul(self.Sigma.t()) # [B, r]
        y = z.matmul(Q.t())          # [B, out]
        if self.bias is not None:
            y = y + self.bias
        return y

    @classmethod
    def from_linear(cls, linear: nn.Module, r: int, shared: Optional[SharedBasis] = None) -> 'LowRankLinear':
        # Handle Conv1D (weight shape [in, out]) or Linear
        if 'Conv1D' in str(type(linear)):
            in_features = linear.weight.shape[0]
            out_features = linear.weight.shape[1]
            W = linear.weight.detach().t()  # transpose to [out, in]
            has_bias = linear.bias is not None
            b = linear.bias.detach() if has_bias else None
        else:
            in_features = linear.in_features
            out_features = linear.out_features
            W = linear.weight.detach()  # already [out, in]
            has_bias = linear.bias is not None
            b = linear.bias.detach() if has_bias else None
        
        device = linear.weight.device
        dtype = linear.weight.dtype

        layer = cls(in_features, out_features, r, shared=shared, bias=has_bias).to(device=device, dtype=dtype)

        # Compute low-rank factors
        with torch.no_grad():
            if shared is None:
                # full SVD then truncate
                try:
                    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                except RuntimeError:
                    # fallback to CPU
                    U, S, Vh = torch.linalg.svd(W.to('cpu'), full_matrices=False)
                    U = U.to(device=device, dtype=dtype)
                    S = S.to(device=device, dtype=dtype)
                    Vh = Vh.to(device=device, dtype=dtype)
                U_r = U[:, :r]
                V_r = Vh[:r, :].t()
                S_r = torch.diag(S[:r])
                
                # Store singular values in Sigma
                layer.P.copy_(V_r)
                layer.Q.copy_(U_r)
                layer.Sigma.copy_(S_r)
            else:
                # Use shared P,Q and fit Sigma = Q^T W P  (r x r)
                P = shared.P.to(device=device, dtype=dtype)
                Q = shared.Q.to(device=device, dtype=dtype)
                # Compute small Sigma via projections
                Sigma = Q.t().matmul(W).matmul(P)
                layer.Sigma.copy_(Sigma)

            if has_bias:
                layer.bias.copy_(b)

        return layer

    def compressed_num_params(self) -> int:
        own_PQ = 0 if self.shared is not None else (self.in_features * self.r + self.out_features * self.r)
        sigma = self.r * self.r
        bias = self.out_features if self.bias is not None else 0
        return own_PQ + sigma + bias


