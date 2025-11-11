import torch
import torch.nn as nn
from typing import Optional

from .poincare import project_to_ball, log_map_zero, exp_map_zero


class RiemannLowRankLinearFixed(nn.Module):
    """
    Fixed Poincaré-tangent low-rank linear with proper scaling:
      y = scale * Exp_0( ( (Log_0(x/scale) @ P) @ Sigma^T ) @ Q^T, c ) + bias
    
    The scale factor compensates for exp/log compression.
    Bias is added AFTER exp map to preserve magnitudes.
    """
    def __init__(self, in_features: int, out_features: int, r: int = 64, c: float = 1e-3, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.c = c
        
        self.P = nn.Parameter(torch.empty(in_features, r))
        self.Q = nn.Parameter(torch.empty(out_features, r))
        self.Sigma = nn.Parameter(torch.empty(r, r))
        
        # Scale factor to compensate exp/log compression
        self.scale = nn.Parameter(torch.ones(1))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.orthogonal_(self.P)
            nn.init.orthogonal_(self.Q)
            if not hasattr(self, '_from_linear_init'):
                self.Sigma.copy_(torch.eye(self.r) * 0.1)
            nn.init.ones_(self.scale)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale down input
        x_scaled = x / self.scale
        
        # Project to ball and map to tangent at 0
        x_proj = project_to_ball(x_scaled, epsilon=1e-5)
        v = log_map_zero(x_proj, c=self.c)
        
        # Low-rank linear in tangent
        z = v.matmul(self.P)              # [B, r]
        z = z.matmul(self.Sigma.t())      # [B, r]
        y_tan = z.matmul(self.Q.t())      # [B, out]
        
        # Map back to manifold
        y = exp_map_zero(y_tan, c=self.c)
        
        # Scale up output
        y = y * self.scale
        
        # Add bias AFTER scaling
        if self.bias is not None:
            y = y + self.bias
            
        return y
    
    @classmethod
    def from_linear(cls, linear: nn.Module, r: int = 64, c: float = 1e-3) -> 'RiemannLowRankLinearFixed':
        # Handle Conv1D or Linear
        if 'Conv1D' in str(type(linear)):
            in_features = linear.weight.shape[0]
            out_features = linear.weight.shape[1]
            W = linear.weight.detach().t()  # to [out,in]
            has_bias = linear.bias is not None
            b = linear.bias.detach() if has_bias else None
        else:
            in_features = linear.in_features
            out_features = linear.out_features
            W = linear.weight.detach()  # [out,in]
            has_bias = linear.bias is not None
            b = linear.bias.detach() if has_bias else None
        
        device = W.device
        dtype = W.dtype
        layer = cls(in_features, out_features, r=r, c=c, bias=has_bias).to(device=device, dtype=dtype)
        layer._from_linear_init = True
        
        # SVD init
        with torch.no_grad():
            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            except RuntimeError:
                U, S, Vh = torch.linalg.svd(W.to('cpu'), full_matrices=False)
                U = U.to(device=device, dtype=dtype)
                S = S.to(device=device, dtype=dtype)
                Vh = Vh.to(device=device, dtype=dtype)
            
            U_r = U[:, :r]
            V_r = Vh[:r, :].t()
            S_r = torch.diag(S[:r])
            
            layer.P.copy_(V_r)
            layer.Q.copy_(U_r)
            layer.Sigma.copy_(S_r)
            
            # Initialize scale based on singular values
            # Approximate compensation for exp/log compression
            avg_s = S[:r].mean()
            layer.scale.copy_(torch.tensor([10.0]))  # Empirical value
            
            if has_bias:
                layer.bias.copy_(b)
                
        return layer

