import torch
import torch.nn as nn
from typing import Optional

from .poincare import project_to_ball, log_map_zero, exp_map_zero
from .. import _rust as _rust_ext


class RiemannLowRankLinear(nn.Module):
    """
    Poincaré-tangent low-rank linear:
      y = Exp_0( ( (Log_0(Proj(x)) @ P) @ Sigma^T ) @ Q^T + b_tan, c )
    Where P:[in,r], Q:[out,r], Sigma:[r,r] (per-layer small), b_tan:[out]
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
        self.bt = nn.Parameter(torch.zeros(out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.orthogonal_(self.P)
            nn.init.orthogonal_(self.Q)
            # Don't overwrite Sigma if it was set by from_linear
            if not hasattr(self, '_from_linear_init'):
                self.Sigma.copy_(torch.eye(self.r) * 0.02)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project to ball and map to tangent at 0
        x_proj = project_to_ball(x, epsilon=1e-5)
        v = log_map_zero(x_proj, c=self.c)
        # Low-rank linear in tangent
        z = v.matmul(self.P)              # [B, r]
        z = z.matmul(self.Sigma.t())      # [B, r]
        y_tan = z.matmul(self.Q.t()) + self.bt  # [B, out]
        # Map back to manifold
        y = exp_map_zero(y_tan, c=self.c)
        if self.bias is not None:
            y = y + self.bias
        return y

    @classmethod
    def from_linear(cls, linear: nn.Module, r: int = 64, c: float = 1e-3) -> 'RiemannLowRankLinear':
        # Support nn.Linear and Conv1D-like (weight shape [in,out] transposed)
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
        layer._from_linear_init = True  # Mark to skip reset_parameters overwriting Sigma
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
            # Keep original singular values (not tiny 0.02)
            S_r = torch.diag(S[:r])
            layer.P.copy_(V_r)
            layer.Q.copy_(U_r)
            layer.Sigma.copy_(S_r)
            if has_bias:
                layer.bias.copy_(b)
        return layer


class RiemannLowRankLinearRust(nn.Module):
    """
    Rust-backed Riemann low-rank linear (Poincaré, tangent at origin):
      y = Exp_0( ((Log_0(Proj(x)) @ P) @ Σ^T) @ Q^T + b_tan, c )
    Parameters are learned in PyTorch; forward calls optimized Rust kernel.
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
        self.bt = nn.Parameter(torch.zeros(out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.orthogonal_(self.P)
            nn.init.orthogonal_(self.Q)
            # Don't overwrite Sigma if it was set by from_linear
            if not hasattr(self, '_from_linear_init'):
                self.Sigma.copy_(torch.eye(self.r) * 0.02)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect cpu float32 for Rust path
        x_np = x.detach().to(dtype=torch.float32, device='cpu').numpy()
        P_np = self.P.detach().to(dtype=torch.float32, device='cpu').numpy()
        Q_np = self.Q.detach().to(dtype=torch.float32, device='cpu').numpy()
        S_np = self.Sigma.detach().to(dtype=torch.float32, device='cpu').numpy()
        bt_np = self.bt.detach().to(dtype=torch.float32, device='cpu').numpy()
        y_np = _rust_ext.riemann_lowrank_forward_cpu(x_np, P_np, S_np, Q_np, bt_np, float(self.c), 1e-5)
        y = torch.from_numpy(y_np).to(device=x.device, dtype=x.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y

    @classmethod
    def from_linear(cls, linear: nn.Module, r: int = 64, c: float = 1e-3) -> 'RiemannLowRankLinearRust':
        if 'Conv1D' in str(type(linear)):
            in_features = linear.weight.shape[0]
            out_features = linear.weight.shape[1]
            W = linear.weight.detach().t()
            has_bias = linear.bias is not None
            b = linear.bias.detach() if has_bias else None
        else:
            in_features = linear.in_features
            out_features = linear.out_features
            W = linear.weight.detach()
            has_bias = linear.bias is not None
            b = linear.bias.detach() if has_bias else None
        device = W.device
        dtype = W.dtype
        layer = cls(in_features, out_features, r=r, c=c, bias=has_bias).to(device=device, dtype=dtype)
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
            # Keep original singular values (not tiny 0.02)
            S_r = torch.diag(S[:r])
            layer.P.copy_(V_r)
            layer.Q.copy_(U_r)
            layer.Sigma.copy_(S_r)
            if has_bias:
                layer.bias.copy_(b)
        return layer


