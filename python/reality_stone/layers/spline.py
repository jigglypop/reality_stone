import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .. import _rust

class SplineLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, k: int = 8, bias: bool = True, use_residual: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.use_residual = use_residual
        self._cached_weight = None  # cached materialized weight for eval
        self.register_buffer('blend_matrix', None)  # [out_features, k+1]
        
        self.control_points = nn.Parameter(torch.randn(k + 1, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        if use_residual:
            self.residual = nn.Parameter(torch.zeros(out_features, in_features))
        else:
            self.register_parameter('residual', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Fast factored forward: y = (x @ C^T) @ B^T [+ x @ R^T] + b
        if self.blend_matrix is None or self.blend_matrix.shape[0] != self.out_features or self.blend_matrix.shape[1] != (self.k + 1):
            self._refresh_blend_matrix()

        # (batch, k+1)
        proj_k = input.matmul(self.control_points.t())
        # (batch, out)
        y = proj_k.matmul(self.blend_matrix.t())
        if self.use_residual and self.residual is not None:
            y = y + input.matmul(self.residual.t())
        if self.bias is not None:
            y = y + self.bias
        return y

    @staticmethod
    def interpolate_weights_static(control_points, k, out_features):
        weights = []
        for i in range(out_features):
            t = i / (out_features - 1)
            t_scaled = t * k
            j = int(np.floor(t_scaled))
            j = max(1, min(j, k - 2))
            t_local = t_scaled - j
            
            t2, t3 = t_local * t_local, t_local * t_local * t_local
            c0 = -0.5 * t3 + t2 - 0.5 * t_local
            c1 = 1.5 * t3 - 2.5 * t2 + 1.0
            c2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t_local
            c3 = 0.5 * t3 - 0.5 * t2
            
            weight_row = (c0 * control_points[j-1] + 
                         c1 * control_points[j] + 
                         c2 * control_points[j+1] + 
                         c3 * control_points[j+2])
            weights.append(weight_row)
        return torch.stack(weights)

    def interpolate_weights_torch(self) -> torch.Tensor:
        return self.interpolate_weights_static(self.control_points, self.k, self.out_features)

    def precompute_weight(self) -> None:
        """Precompute and cache the materialized weight for faster inference."""
        with torch.no_grad():
            self._cached_weight = self.interpolate_weights_torch().to(device=self.control_points.device, dtype=self.control_points.dtype)

    def _refresh_blend_matrix(self) -> None:
        # Build dense blend matrix [out_features, k+1] with at most 4 non-zeros per row
        import numpy as np
        k = self.k
        out = self.out_features
        B = np.zeros((out, k + 1), dtype='float32')
        for i in range(out):
            t = i / max(1, (out - 1))
            t_scaled = t * k
            j = int(np.floor(t_scaled))
            j = max(1, min(j, k - 2))
            t_local = t_scaled - j
            t2, t3 = t_local * t_local, t_local * t_local * t_local
            c0 = -0.5 * t3 + t2 - 0.5 * t_local
            c1 = 1.5 * t3 - 2.5 * t2 + 1.0
            c2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t_local
            c3 = 0.5 * t3 - 0.5 * t2
            idx0 = max(0, j - 1)
            idx1 = j
            idx2 = min(k, j + 1)
            idx3 = min(k, j + 2)
            B[i, idx0] += c0
            B[i, idx1] += c1
            B[i, idx2] += c2
            B[i, idx3] += c3
        bm = torch.from_numpy(B).to(device=self.control_points.device, dtype=self.control_points.dtype)
        self.blend_matrix = bm

    @classmethod
    def from_linear(cls, linear: nn.Linear, k: int = 8, 
                   learning_rate: float = 0.01, steps: int = 100, use_residual: bool = True) -> 'SplineLinear':
        spline_layer = cls(linear.in_features, linear.out_features, k, 
                          bias=(linear.bias is not None), use_residual=use_residual)
        
        weight_np = linear.weight.detach().cpu().numpy()
        
        rust_spline_instance = _rust.spline.SplineLayer.from_weight_py(
            weight_np, k, learning_rate, steps
        )
        
        optimized_control_points = torch.from_numpy(
            rust_spline_instance.control_points
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        
        spline_layer.control_points.data.copy_(optimized_control_points)
        
        if use_residual:
            interpolated_weight = spline_layer.interpolate_weights_torch().detach()
            spline_layer.residual.data.copy_(linear.weight.data - interpolated_weight)
        
        if linear.bias is not None:
            spline_layer.bias.data.copy_(linear.bias.data)

        # Cache materialized weight for fast inference
        spline_layer.precompute_weight()
        
        return spline_layer

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, k={self.k}, '
                f'use_residual={self.use_residual}, compression_ratio={self.get_compression_ratio():.1f}x')

    def get_compression_ratio(self) -> float:
        original_params = self.in_features * self.out_features
        compressed_params = self.control_points.numel()
        if self.use_residual and self.residual is not None:
            compressed_params += self.residual.numel()
        return original_params / compressed_params if compressed_params > 0 else float('inf') 