"""
Reality Stone Advanced Layers - PyTorch nn.Module Interface
고급 기능들을 활용한 PyTorch 레이어들
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
import math

from .advanced import (
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

class HyperbolicLinearAdvanced(nn.Module):
    """고급 하이퍼볼릭 선형 레이어"""
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 config: Optional[AdvancedConfig] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config or AdvancedConfig()
        
        # 가중치와 바이어스
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        self.bias = nn.Parameter(torch.empty(output_dim))
        
        # 동적 곡률 (선택적)
        if self.config.enable_dynamic_curvature:
            self.dynamic_curvature = DynamicCurvatureLayer(input_dim, self.config.base_curvature)
        else:
            self.register_buffer('curvature', torch.tensor(self.config.base_curvature))
            
        # 측지선 활성화 (선택적)
        if self.config.enable_geodesic_activation:
            self.geodesic_activation = GeodesicActivationLayer(
                input_dim=output_dim, 
                num_anchors=self.config.num_anchors,
                curvature=self.config.base_curvature
            )
        
        self._init_parameters()
        
    def _init_parameters(self):
        """하이퍼볼릭 공간에 맞는 파라미터 초기화"""
        # Xavier 초기화 + 하이퍼볼릭 스케일링
        scale = math.sqrt(2.0 / (self.input_dim + self.output_dim)) * 0.1
        nn.init.normal_(self.weight, 0, scale)
        nn.init.normal_(self.bias, 0, 0.01)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 [B, D_in]
        Returns:
            torch.Tensor: 출력 텐서 [B, D_out]
        """
        if self.config.enable_dynamic_curvature:
            curvatures = self.dynamic_curvature(x)
            # 배치별 동적 곡률 사용 (간단화: 평균 사용)
            curvature = curvatures.mean().item()
        else:
            curvature = self.curvature.item()
            
        # Fused 연산 사용
        if self.config.enable_fused_ops:
            output = hyperbolic_linear_fused(x, self.weight, self.bias, curvature)
        else:
            # Fallback: 표준 선형 변환
            output = F.linear(x, self.weight, self.bias)
            
        # 측지선 활성화 (선택적)
        if self.config.enable_geodesic_activation:
            output = self.geodesic_activation(output)
            
        return output
        
    def compute_regularization_loss(self, x: torch.Tensor) -> torch.Tensor:
        """정규화 손실 계산"""
        if not self.config.enable_regularization:
            return torch.tensor(0.0, device=x.device)
            
        curvature = self.curvature.item() if hasattr(self, 'curvature') else self.config.base_curvature
        
        return hyperbolic_regularization(
            x, self.weight, curvature,
            self.config.lambda_boundary,
            self.config.lambda_curvature, 
            self.config.lambda_geodesic
        )

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

class AdvancedHyperbolicMLP(nn.Module):
    """모든 고급 기능이 포함된 하이퍼볼릭 MLP"""
    
    def __init__(self,
                 input_dim: int = 784,
                 hidden_dims: List[int] = [128, 64],
                 output_dim: int = 10,
                 config: Optional[AdvancedConfig] = None):
        super().__init__()
        self.config = config or AdvancedConfig()
        
        # 레이어 구성
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            layer = HyperbolicLinearAdvanced(
                dims[i], dims[i+1], 
                config=self.config
            )
            self.layers.append(layer)
            
        # 정규화 레이어 (선택적)
        if self.config.enable_regularization:
            self.regularization_layers = nn.ModuleList([
                RegularizedHyperbolicLayer(
                    dim, self.config.base_curvature, 
                    self.config.lambda_boundary
                ) for dim in hidden_dims
            ])
        
    def forward(self, x: torch.Tensor, return_reg_loss: bool = False):
        """
        Args:
            x: 입력 텐서 [B, input_dim]
            return_reg_loss: 정규화 손실 반환 여부
        Returns:
            torch.Tensor 또는 Tuple: 출력 (+ 정규화 손실)
        """
        x = x.view(x.size(0), -1)  # Flatten
        reg_losses = []
        
        # 히든 레이어들
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            
            # 정규화 적용 (선택적)
            if self.config.enable_regularization and i < len(self.regularization_layers):
                x, reg_loss = self.regularization_layers[i](x)
                reg_losses.append(reg_loss)
                
        # 출력 레이어
        output = self.layers[-1](x)
        
        # NaN 문제 해결
        output = fix_mnist_nan(output, self.config.base_curvature)
        
        if return_reg_loss:
            total_reg_loss = sum(reg_losses) if reg_losses else torch.tensor(0.0, device=x.device)
            return output, total_reg_loss
        else:
            return output

class DynamicCurvatureMLP(nn.Module):
    """동적 곡률을 사용하는 MLP"""
    
    def __init__(self,
                 input_dim: int = 784, 
                 hidden_dim: int = 128,
                 output_dim: int = 10,
                 base_curvature: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim
        
        # 동적 곡률 예측기
        self.curvature_predictor = DynamicCurvatureLayer(input_dim, base_curvature)
        
        # 일반 가중치들
        self.weight1 = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.01)
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim))
        self.weight2 = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.01)
        self.bias2 = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 [B, input_dim]
        Returns:
            torch.Tensor: 출력 텐서 [B, output_dim]
        """
        x = x.view(x.size(0), -1)
        
        # 동적 곡률 예측
        curvatures = self.curvature_predictor(x)
        
        # 첫 번째 레이어 (동적 곡률 사용)
        h1 = hyperbolic_linear_fused(x, self.weight1, self.bias1, curvatures.mean().item())
        
        # 두 번째 레이어
        output = hyperbolic_linear_fused(h1, self.weight2, self.bias2, curvatures.mean().item())
        
        return output

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

# ===============================
# Convenience Factory Functions
# ===============================

def create_mnist_model(config: Optional[AdvancedConfig] = None) -> nn.Module:
    """MNIST용 최적화 모델 생성
    
    Args:
        config: 고급 기능 설정
        
    Returns:
        nn.Module: MNIST 분류용 모델
    """
    if config is None:
        # MNIST NaN 문제 해결에 특화된 설정
        config = AdvancedConfig(
            enable_regularization=True,
            lambda_boundary=1.0,
            lambda_curvature=0.1,
            lambda_geodesic=0.01,
            enable_dynamic_curvature=False,
            base_curvature=1.0,
            enable_fused_ops=True,
            enable_geodesic_activation=False
        )
    
    return AdvancedHyperbolicMLP(
        input_dim=784,
        hidden_dims=[128, 64],
        output_dim=10,
        config=config
    )

def create_performance_model(input_dim: int, 
                           output_dim: int,
                           hidden_dims: List[int] = [256, 128]) -> nn.Module:
    config = AdvancedConfig(
        enable_regularization=False,  # 성능을 위해 비활성화
        enable_dynamic_curvature=False,
        enable_fused_ops=True,        # 핵심!
        enable_geodesic_activation=False
    )
    
    return AdvancedHyperbolicMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims, 
        output_dim=output_dim,
        config=config
    )

def create_research_model(input_dim: int,
                         output_dim: int,
                         hidden_dims: List[int] = [256, 128]) -> nn.Module:
    config = AdvancedConfig(
        enable_regularization=True,
        enable_dynamic_curvature=True,
        enable_fused_ops=True,
        enable_geodesic_activation=True,
        num_anchors=8
    )
    
    return AdvancedHyperbolicMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim, 
        config=config
    ) 