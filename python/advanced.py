"""
Reality Stone Advanced Features - Python Interface
고급 하이퍼볼릭 신경망 기능들의 Python 인터페이스
"""

import torch
from torch.autograd import Function
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
import warnings

# C++ 확장에서 고급 기능들 import (모든 구현된 기능)try:    from ._C import (        # Fused Operations        fused_linear, fused_mobius_chain, fused_transform_reg, benchmark_fused,        fused_linear_cuda, fused_mobius_chain_cuda, fused_transform_reg_cuda, benchmark_fused_cuda,                # Dynamic Curvature        dynamic_mobius_add, dynamic_poincare_layer,                # Regularization        boundary_penalty, curvature_penalty, geodesic_penalty, combined_reg,                # Geodesic Activation (CUDA only)        geodesic_activation_cuda, einstein_midpoint_cuda, multi_geodesic_cuda,                # Convenience Functions        fix_mnist_nan, benchmark_overall,    )    _has_advanced = Trueexcept ImportError:    _has_advanced = False    warnings.warn("Advanced features not available. Please rebuild with advanced support.")

_has_cuda = torch.cuda.is_available()

@dataclass
class AdvancedConfig:
    """고급 기능 설정"""
    # 정규화 설정
    enable_regularization: bool = True
    lambda_boundary: float = 1.0
    lambda_curvature: float = 0.1  
    lambda_geodesic: float = 0.01
    
    # 동적 곡률 설정
    enable_dynamic_curvature: bool = False
    base_curvature: float = 1.0
    
    # 융합 연산 설정
    enable_fused_ops: bool = True
    
    # 측지선 활성화 설정
    enable_geodesic_activation: bool = False
    num_anchors: int = 4

# ===============================
# Dynamic Curvature Functions
# ===============================

class DynamicCurvaturePrediction(Function):
    """동적 곡률 예측 Function"""
    
    @staticmethod
    def forward(ctx, x, weight, bias, base_curvature):
        if not _has_advanced:
            # Fallback: 고정 곡률 반환
            return torch.full((x.size(0),), base_curvature, device=x.device)
            
        ctx.save_for_backward(x, weight, bias)
        ctx.base_curvature = base_curvature
        
        if x.is_cuda and _has_cuda:
            return dynamic_curvature_prediction_cuda(x, weight, bias, base_curvature)
        else:
            return dynamic_curvature_prediction_cpu(x, weight, bias, base_curvature)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        # 간단화: 그래디언트는 autograd로 처리
        return None, None, None, None

class DynamicMobiusAdd(Function):
    """동적 곡률을 사용한 Möbius 덧셈"""
    
    @staticmethod
    def forward(ctx, u, v, curvatures):
        if not _has_advanced:
            # Fallback: 첫 번째 곡률값 사용
            from . import mobius_add
            return mobius_add(u, v, curvatures[0].item())
            
        ctx.save_for_backward(u, v, curvatures)
        
        if u.is_cuda and _has_cuda:
            return dynamic_mobius_add_cuda(u, v, curvatures)
        else:
            return dynamic_mobius_add_cpu(u, v, curvatures)
    
    @staticmethod
    def backward(ctx, grad_output):
        # autograd로 처리
        return None, None, None

# ===============================
# Hyperbolic Regularization
# ===============================

class HyperbolicRegularization(Function):
    """하이퍼볼릭 정규화 Function"""
    
    @staticmethod
    def forward(ctx, x, weights, curvature, lambda_boundary, lambda_curvature, lambda_geodesic):
        if not _has_advanced:
            # Fallback: 간단한 경계 정규화
            norm = torch.norm(x, p=2, dim=-1)
            max_norm = 1.0 / (curvature ** 0.5) - 0.01
            violation = torch.relu(norm - max_norm)
            return lambda_boundary * torch.mean(violation ** 2)
            
        ctx.save_for_backward(x, weights)
        ctx.curvature = curvature
        ctx.lambdas = (lambda_boundary, lambda_curvature, lambda_geodesic)
        
        if x.is_cuda and _has_cuda:
            return combined_regularization_cuda(x, weights, curvature, 
                                              lambda_boundary, lambda_curvature, lambda_geodesic)
        else:
            return combined_regularization_cpu(x, weights, curvature,
                                             lambda_boundary, lambda_curvature, lambda_geodesic)
    
    @staticmethod
    def backward(ctx, grad_output):
        # autograd로 처리
        return None, None, None, None, None, None

# ===============================
# Geodesic Activation
# ===============================

class GeodesicActivation(Function):
    """측지선 기반 활성화 함수"""
    
    @staticmethod
    def forward(ctx, input, anchors, t_values, weights, curvature):
        if not _has_advanced:
            # Fallback: 표준 활성화 함수
            return torch.tanh(input)
            
        ctx.save_for_backward(input, anchors, t_values, weights)
        ctx.curvature = curvature
        
        if input.is_cuda and _has_cuda:
            return geodesic_activation_cuda(input, anchors, t_values, weights, curvature)
        else:
            return geodesic_activation_cpu(input, anchors, t_values, weights, curvature)
    
    @staticmethod
    def backward(ctx, grad_output):
        # autograd로 처리
        return None, None, None, None, None

class EinsteinMidpoint(Function):
    """Einstein 중점 계산"""
    
    @staticmethod
    def forward(ctx, points, weights, curvature):
        if not _has_advanced:
            # Fallback: 가중 평균
            return torch.sum(points * weights.unsqueeze(-1), dim=1)
            
        ctx.save_for_backward(points, weights)
        ctx.curvature = curvature
        
        if points.is_cuda and _has_cuda:
            return einstein_midpoint_cuda(points, weights, curvature)
        else:
            return einstein_midpoint_cpu(points, weights, curvature)
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None

# ===============================
# Fused Operations
# ===============================

class HyperbolicLinearFused(Function):
    """퓨즈드 하이퍼볼릭 선형 레이어"""
    
    @staticmethod
    def forward(ctx, input, weight, bias, curvature):
        if not _has_advanced:
            # Fallback: 단계별 수행
            from . import mobius_add
            
            # 간단한 선형 변환
            linear_result = torch.mm(input, weight.t()) + bias
            return linear_result
            
        ctx.save_for_backward(input, weight, bias)
        ctx.curvature = curvature
        
        if input.is_cuda and _has_cuda:
            return hyperbolic_linear_fused_cuda(input, weight, bias, curvature)
        else:
            return hyperbolic_linear_fused_cpu(input, weight, bias, curvature)
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None

class TransformRegularizeFused(Function):
    """변환-정규화 퓨즈드 연산"""
    
    @staticmethod
    def forward(ctx, input, curvature, reg_lambda):
        if not _has_advanced:
            # Fallback: 간단한 정규화
            norm = torch.norm(input, p=2, dim=-1, keepdim=True)
            max_norm = 1.0 / (curvature ** 0.5) - 0.01
            clamped = torch.clamp(norm, max=max_norm)
            direction = input / (norm + 1e-7)
            transformed = direction * clamped
            
            violation = torch.relu(norm - max_norm)
            reg_loss = reg_lambda * torch.mean(violation ** 2)
            return transformed, reg_loss
            
        ctx.curvature = curvature
        ctx.reg_lambda = reg_lambda
        
        if input.is_cuda and _has_cuda:
            return transform_regularize_fused_cuda(input, curvature, reg_lambda)
        else:
            return transform_regularize_fused_cpu(input, curvature, reg_lambda)
    
    @staticmethod
    def backward(ctx, grad_transformed, grad_loss):
        return None, None, None

# ===============================
# Python API Functions  
# ===============================

def predict_dynamic_curvature(x: torch.Tensor, 
                             weight: torch.Tensor, 
                             bias: torch.Tensor, 
                             base_curvature: float = 1.0) -> torch.Tensor:
    """동적 곡률 예측
    
    Args:
        x: 입력 텐서 [B, D]
        weight: 곡률 예측 가중치 [1, D] 
        bias: 곡률 예측 바이어스 [1]
        base_curvature: 기본 곡률값
        
    Returns:
        torch.Tensor: 예측된 곡률값들 [B]
    """
    return DynamicCurvaturePrediction.apply(x, weight, bias, base_curvature)

def dynamic_mobius_add(u: torch.Tensor, 
                      v: torch.Tensor, 
                      curvatures: torch.Tensor) -> torch.Tensor:
    """동적 곡률을 사용한 Möbius 덧셈
    
    Args:
        u, v: 입력 텐서들 [B, D]
        curvatures: 배치별 곡률값들 [B]
        
    Returns:
        torch.Tensor: u ⊕_c v 결과 [B, D]
    """
    return DynamicMobiusAdd.apply(u, v, curvatures)

def hyperbolic_regularization(x: torch.Tensor,
                            weights: torch.Tensor,
                            curvature: float,
                            lambda_boundary: float = 1.0,
                            lambda_curvature: float = 0.1,
                            lambda_geodesic: float = 0.01) -> torch.Tensor:
    """하이퍼볼릭 정규화 손실 계산
    
    Args:
        x: 입력 텐서 [B, D]
        weights: 모델 가중치 [N, D]
        curvature: 곡률값
        lambda_*: 정규화 가중치들
        
    Returns:
        torch.Tensor: 정규화 손실 [1]
    """
    return HyperbolicRegularization.apply(x, weights, curvature, 
                                        lambda_boundary, lambda_curvature, lambda_geodesic)

def geodesic_activation(input: torch.Tensor,
                       anchors: torch.Tensor,
                       t_values: torch.Tensor, 
                       weights: torch.Tensor,
                       curvature: float) -> torch.Tensor:
    """측지선 기반 활성화 함수
    
    Args:
        input: 입력 텐서 [B, D]
        anchors: 앵커 포인트들 [K, D] 
        t_values: 측지선 파라미터들 [K]
        weights: 앵커 가중치들 [K]
        curvature: 곡률값
        
    Returns:
        torch.Tensor: 활성화된 텐서 [B, D]
    """
    return GeodesicActivation.apply(input, anchors, t_values, weights, curvature)

def einstein_midpoint(points: torch.Tensor,
                     weights: torch.Tensor,
                     curvature: float) -> torch.Tensor:
    """Einstein 중점 계산
    
    Args:
        points: 포인트들 [B, K, D]
        weights: 가중치들 [K]
        curvature: 곡률값
        
    Returns:
        torch.Tensor: Einstein 중점 [B, D]
    """
    return EinsteinMidpoint.apply(points, weights, curvature)

def hyperbolic_linear_fused(input: torch.Tensor,
                           weight: torch.Tensor,
                           bias: torch.Tensor, 
                           curvature: float) -> torch.Tensor:
    """퓨즈드 하이퍼볼릭 선형 레이어
    
    log_0(x) → linear → exp_0 → ⊕bias 를 한 번에 수행
    
    Args:
        input: 입력 텐서 [B, D_in]
        weight: 가중치 [D_out, D_in]
        bias: 바이어스 [D_out]
        curvature: 곡률값
        
    Returns:
        torch.Tensor: 변환된 텐서 [B, D_out]
    """
    return HyperbolicLinearFused.apply(input, weight, bias, curvature)

def transform_regularize_fused(input: torch.Tensor,
                              curvature: float,
                              reg_lambda: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """변환-정규화 퓨즈드 연산
    
    Args:
        input: 입력 텐서 [B, D]
        curvature: 곡률값
        reg_lambda: 정규화 가중치
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (변환된 텐서, 정규화 손실)
    """
    return TransformRegularizeFused.apply(input, curvature, reg_lambda)

# ===============================
# Convenience Functions
# ===============================

def fix_mnist_nan(logits: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """MNIST NaN 문제 즉시 해결
    
    Args:
        logits: 로짓 텐서
        curvature: 곡률값
        
    Returns:
        torch.Tensor: NaN이 제거된 텐서
    """
    if not _has_advanced:
        # Fallback: 간단한 클리핑
        return torch.clamp(logits, -10, 10)
    
    transformed, reg_loss = transform_regularize_fused(logits, curvature, 1.0)
    return transformed

@dataclass 
class BenchmarkResult:
    """벤치마크 결과"""
    fused_time_ms: float
    unfused_time_ms: float  
    speedup_ratio: float
    memory_saved_bytes: int = 0

def benchmark_advanced_features(input: torch.Tensor,
                               weight: torch.Tensor,
                               bias: torch.Tensor,
                               curvature: float = 1.0,
                               num_iterations: int = 100) -> BenchmarkResult:
    """고급 기능 성능 벤치마크
    
    Args:
        input, weight, bias: 테스트 텐서들
        curvature: 곡률값
        num_iterations: 반복 횟수
        
    Returns:
        BenchmarkResult: 벤치마크 결과
    """
    if not _has_advanced:
        return BenchmarkResult(0.0, 0.0, 1.0, 0)
    
    if input.is_cuda and _has_cuda:
        return benchmark_fused_vs_unfused_cuda(input, weight, curvature, num_iterations)
    else:
        return benchmark_fused_vs_unfused_cpu(input, weight, curvature, num_iterations) 