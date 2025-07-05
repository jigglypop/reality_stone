import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import math

from ..advanced import (
    AdvancedConfig, 
    predict_dynamic_curvature, dynamic_mobius_add,
    hyperbolic_regularization, geodesic_activation, einstein_midpoint,
    hyperbolic_linear_fused, transform_regularize_fused, fix_mnist_nan
)

class DynamicCurvatureLayer(nn.Module):
    """동적 곡률 예측 레이어"""
    
    def __init__(self, input_dim: int, base_curvature: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.base_curvature = base_curvature
        
        # 곡률 예측을 위한 파라미터
        self.curvature_weight = nn.Parameter(torch.randn(1, input_dim) * 0.1)
        self.curvature_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 [B, D]
        Returns:
            torch.Tensor: 예측된 곡률값들 [B]
        """
        return predict_dynamic_curvature(x, self.curvature_weight, self.curvature_bias, self.base_curvature)

class GeodesicActivationLayer(nn.Module):
    """측지선 기반 활성화 레이어"""
    
    def __init__(self, input_dim: int, num_anchors: int = 4, curvature: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_anchors = num_anchors
        self.curvature = curvature
        
        # 앵커 포인트들 (포인카레 디스크 내부에 배치)
        self.anchors = nn.Parameter(torch.randn(num_anchors, input_dim) * 0.3)
        
        # 측지선 파라미터들
        self.t_values = nn.Parameter(torch.full((num_anchors,), 0.5))
        
        # 앵커별 가중치
        self.anchor_weights = nn.Parameter(torch.ones(num_anchors) / num_anchors)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 [B, D]
        Returns:
            torch.Tensor: 활성화된 텐서 [B, D]
        """
        return geodesic_activation(x, self.anchors, self.t_values, self.anchor_weights, self.curvature)

class RegularizedHyperbolicLayer(nn.Module):
    """정규화가 적용된 하이퍼볼릭 레이어"""
    
    def __init__(self, 
                 input_dim: int,
                 curvature: float = 1.0,
                 reg_lambda: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.curvature = curvature
        self.reg_lambda = reg_lambda
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 입력 텐서 [B, D]
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (정규화된 텐서, 정규화 손실)
        """
        return transform_regularize_fused(x, self.curvature, self.reg_lambda)

class FusedHyperbolicLayer(nn.Module):
    """Fused 연산만을 사용하는 고성능 레이어"""
    
    def __init__(self, input_dim: int, output_dim: int, curvature: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.curvature = curvature
        
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        self.bias = nn.Parameter(torch.empty(output_dim))
        
        # 최적화된 초기화
        scale = math.sqrt(2.0 / (input_dim + output_dim)) * 0.1
        nn.init.normal_(self.weight, 0, scale)
        nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 [B, input_dim]
        Returns:
            torch.Tensor: 출력 텐서 [B, output_dim]
        """
        return hyperbolic_linear_fused(x, self.weight, self.bias, self.curvature) 