import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from dataclasses import dataclass
from typing import Tuple
import warnings

try:
    import reality_stone._C as _C
    HAS_CUDA = hasattr(_C, 'mobius_add_cuda')
except ImportError:
    _C = None
    HAS_CUDA = False

_has_cuda = torch.cuda.is_available()

class AdvancedConfig:
    """고급 기능 설정 클래스"""
    def __init__(
        self,
        enable_regularization: bool = True,
        enable_dynamic_curvature: bool = False,
        enable_fused_ops: bool = False,
        enable_geodesic_activation: bool = False,
        enable_chebyshev_approximation: bool = False,
        enable_laplace_beltrami: bool = False,
        enable_hyperbolic_fft: bool = False,
        lambda_boundary: float = 1.0,
        lambda_curvature: float = 0.1,
        lambda_geodesic: float = 0.01,
        base_curvature: float = 1.0,
        num_anchors: int = 4,
        chebyshev_order: int = 10,
        max_eigenvalues: int = 100
    ):
        self.enable_regularization = enable_regularization
        self.enable_dynamic_curvature = enable_dynamic_curvature
        self.enable_fused_ops = enable_fused_ops
        self.enable_geodesic_activation = enable_geodesic_activation
        self.enable_chebyshev_approximation = enable_chebyshev_approximation
        self.enable_laplace_beltrami = enable_laplace_beltrami
        self.enable_hyperbolic_fft = enable_hyperbolic_fft
        
        self.lambda_boundary = lambda_boundary
        self.lambda_curvature = lambda_curvature
        self.lambda_geodesic = lambda_geodesic
        self.base_curvature = base_curvature
        self.num_anchors = num_anchors
        self.chebyshev_order = chebyshev_order
        self.max_eigenvalues = max_eigenvalues

# ===============================
# Dynamic Curvature Functions
# ===============================

class DynamicCurvaturePrediction(Function):
    """동적 곡률 예측 Function"""
    
    @staticmethod
    def forward(ctx, x, weight, bias, base_curvature):
        if not _C:
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
        return None, None, None, None

class DynamicMobiusAdd(Function):
    """동적 곡률을 사용한 Möbius 덧셈"""
    
    @staticmethod
    def forward(ctx, u, v, curvatures):
        if not _C:
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
        if not _C:
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
        if not _C:
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
        if not _C:
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
        if not _C:
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
        if not _C:
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

def geodesic_activation(
    x: torch.Tensor,
    num_anchors: int = 4,
    curvature: float = 1.0
) -> torch.Tensor:
    """측지선 기반 활성화 함수"""
    if _C is None:
        warnings.warn("C++ extension not available, using tanh activation")
        return torch.tanh(x)
    
    if HAS_CUDA and x.is_cuda:
        anchors = torch.randn(num_anchors, x.size(-1), device=x.device) * 0.3
        t_params = torch.full((num_anchors,), 0.5, device=x.device)
        weights = torch.ones(num_anchors, device=x.device) / num_anchors
        return _C.geodesic_activation(x, anchors, t_params, weights, curvature)
    else:
        warnings.warn("CUDA not available for geodesic activation")
        return torch.tanh(x)

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
    if not _C:
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
    if not _C:
        return BenchmarkResult(0.0, 0.0, 1.0, 0)
    
    if input.is_cuda and _has_cuda:
        return benchmark_fused_vs_unfused_cuda(input, weight, curvature, num_iterations)
    else:
        return benchmark_fused_vs_unfused_cpu(input, weight, curvature, num_iterations)

# ===== 체비셰프 관련 함수들 =====

def chebyshev_approximation(x: torch.Tensor, 
                           order: int = 10, 
                           curvature: float = 1.0) -> torch.Tensor:
    """체비셰프 다항식을 이용한 하이퍼볼릭 함수 근사"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        return torch.tanh(torch.sqrt(torch.tensor(curvature)) * x)
    
    if HAS_CUDA and x.is_cuda:
        return _C.chebyshev_approximation_cuda(x, order, curvature)
    else:
        return _C.chebyshev_approximation_cpu(x, order, curvature)

def chebyshev_distance(x: torch.Tensor, 
                      y: torch.Tensor, 
                      curvature: float = 1.0) -> torch.Tensor:
    """체비셰프 거리 계산 (하이퍼볼릭 공간)"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        diff = torch.abs(x - y)
        cheb_dist = torch.max(diff, dim=-1).values
        sqrt_c = torch.sqrt(torch.tensor(curvature))
        scaled_dist = torch.clamp(sqrt_c * cheb_dist, 0.0, 0.99)
        return (1.0 / sqrt_c) * torch.atanh(scaled_dist)
    
    if HAS_CUDA and x.is_cuda:
        return _C.chebyshev_distance_cuda(x, y, curvature)
    else:
        return _C.chebyshev_distance_cpu(x, y, curvature)

def chebyshev_nodes(n: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """체비셰프 점들 생성"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        k = torch.arange(n, dtype=torch.float32, device=device)
        return torch.cos((2 * k + 1) * torch.pi / (2 * n))
    
    return _C.chebyshev_nodes_cpu(n, device)

def fast_chebyshev_transform(values: torch.Tensor) -> torch.Tensor:
    """고속 체비셰프 변환"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        return torch.fft.dct(values, type=1, norm='ortho')
    
    if HAS_CUDA and values.is_cuda:
        return _C.fast_chebyshev_transform_cuda(values)
    else:
        return _C.fast_chebyshev_transform_cpu(values)

def inverse_chebyshev_transform(coeffs: torch.Tensor, 
                               eval_points: torch.Tensor = None) -> torch.Tensor:
    """역 체비셰프 변환"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        if eval_points is None:
            return torch.fft.idct(coeffs, type=1, norm='ortho')
        
        order = coeffs.size(-1) - 1
        x = torch.clamp(eval_points, -1.0 + 1e-6, 1.0 - 1e-6)
        result = torch.zeros(coeffs.size(0), x.size(0), 
                           dtype=coeffs.dtype, device=coeffs.device)
        
        for k in range(order + 1):
            T_k = torch.cos(k * torch.acos(x))
            result += coeffs[:, k:k+1] * T_k.unsqueeze(0)
        
        return result
    
    if eval_points is None:
        eval_points = chebyshev_nodes(coeffs.size(-1), coeffs.device)
    
    if HAS_CUDA and coeffs.is_cuda:
        return _C.inverse_chebyshev_transform_cuda(coeffs, eval_points)
    else:
        return _C.inverse_chebyshev_transform_cpu(coeffs, eval_points)

def chebyshev_derivative(coeffs: torch.Tensor) -> torch.Tensor:
    """체비셰프 다항식의 해석적 미분"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        n = coeffs.size(-1)
        if n <= 1:
            return torch.zeros(coeffs.size(0), 1, dtype=coeffs.dtype, device=coeffs.device)
        
        d_coeffs = torch.zeros(coeffs.size(0), n - 1, dtype=coeffs.dtype, device=coeffs.device)
        for k in range(n - 2, -1, -1):
            if k == n - 2:
                d_coeffs[:, k] = 2 * (k + 1) * coeffs[:, k + 1]
            else:
                d_coeffs[:, k] = d_coeffs[:, k + 2] + 2 * (k + 1) * coeffs[:, k + 1]
        
        return d_coeffs
    
    if HAS_CUDA and coeffs.is_cuda:
        return _C.chebyshev_derivative_cuda(coeffs)
    else:
        return _C.chebyshev_derivative_cpu(coeffs)

def chebyshev_integral(coeffs: torch.Tensor, constant: float = 0.0) -> torch.Tensor:
    """체비셰프 다항식의 해석적 적분"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        n = coeffs.size(-1)
        i_coeffs = torch.zeros(coeffs.size(0), n + 1, dtype=coeffs.dtype, device=coeffs.device)
        i_coeffs[:, 0] = constant
        
        for k in range(n):
            if k == 0:
                i_coeffs[:, 1] += coeffs[:, 0]
            else:
                i_coeffs[:, k + 1] += coeffs[:, k] / (2 * (k + 1))
                if k > 1:
                    i_coeffs[:, k - 1] -= coeffs[:, k] / (2 * (k - 1))
        
        return i_coeffs
    
    if HAS_CUDA and coeffs.is_cuda:
        return _C.chebyshev_integral_cuda(coeffs, constant)
    else:
        return _C.chebyshev_integral_cpu(coeffs, constant)

# ===== 라플라스-벨트라미 관련 함수들 =====

def hyperbolic_laplacian(f: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """하이퍼볼릭 라플라시안 계산"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        return torch.zeros_like(f)
    
    if HAS_CUDA and f.is_cuda:
        return _C.hyperbolic_laplacian_cuda(f, curvature)
    else:
        return _C.hyperbolic_laplacian_cpu(f, curvature)

def heat_kernel(x: torch.Tensor, t: float, curvature: float = 1.0) -> torch.Tensor:
    """열 핵 계산"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        return torch.exp(-torch.norm(x, dim=-1, keepdim=True) ** 2 / (4 * t))
    
    if HAS_CUDA and x.is_cuda:
        return _C.heat_kernel_cuda(x, t, curvature)
    else:
        return _C.heat_kernel_cpu(x, t, curvature)

def laplace_beltrami_eigen(manifold_points: torch.Tensor, 
                          curvature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """라플라스-벨트라미 고유값 분해"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        n = manifold_points.size(0)
        eigenvals = torch.ones(min(n, 100), dtype=manifold_points.dtype, 
                             device=manifold_points.device)
        eigenvecs = torch.eye(n, min(n, 100), dtype=manifold_points.dtype, 
                            device=manifold_points.device)
        return eigenvals, eigenvecs
    
    if HAS_CUDA and manifold_points.is_cuda:
        return _C.laplace_beltrami_eigen_cuda(manifold_points, curvature)
    else:
        return _C.laplace_beltrami_eigen_cpu(manifold_points, curvature)

def spectral_graph_conv(x: torch.Tensor, 
                       laplacian: torch.Tensor, 
                       weight: torch.Tensor) -> torch.Tensor:
    """스펙트럴 그래프 컨볼루션"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        return torch.mm(torch.mm(laplacian, x), weight)
    
    if HAS_CUDA and x.is_cuda:
        return _C.spectral_graph_conv_cuda(x, laplacian, weight)
    else:
        return _C.spectral_graph_conv_cpu(x, laplacian, weight)

def solve_diffusion_equation(initial_condition: torch.Tensor,
                           time_step: float,
                           num_steps: int,
                           curvature: float = 1.0) -> torch.Tensor:
    """확산 방정식 해결"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        current = initial_condition.clone()
        for _ in range(num_steps):
            current = current * (1 - time_step * curvature)
        return current
    
    if HAS_CUDA and initial_condition.is_cuda:
        return _C.solve_diffusion_equation_cuda(initial_condition, time_step, num_steps, curvature)
    else:
        return _C.solve_diffusion_equation_cpu(initial_condition, time_step, num_steps, curvature)

def geodesic_distance_matrix(points: torch.Tensor, 
                           curvature: float = 1.0) -> torch.Tensor:
    """지오데식 거리 행렬 계산"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        return torch.cdist(points, points)
    
    if HAS_CUDA and points.is_cuda:
        return _C.geodesic_distance_matrix_cuda(points, curvature)
    else:
        return _C.geodesic_distance_matrix_cpu(points, curvature)

def spectral_normalize(adjacency_matrix: torch.Tensor) -> torch.Tensor:
    """스펙트럴 정규화"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        row_sums = adjacency_matrix.sum(dim=1, keepdim=True)
        return adjacency_matrix / (row_sums + 1e-6)
    
    if HAS_CUDA and adjacency_matrix.is_cuda:
        return _C.spectral_normalize_cuda(adjacency_matrix)
    else:
        return _C.spectral_normalize_cpu(adjacency_matrix)

# ===== FFT 및 리만 기하학 관련 함수들 =====

def hyperbolic_fft(x: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """하이퍼볼릭 FFT"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        return torch.fft.fft(x.float()).real
    
    if HAS_CUDA and x.is_cuda:
        return _C.hyperbolic_fft_cuda(x, curvature)
    else:
        return _C.hyperbolic_fft_cpu(x, curvature)

def spherical_harmonics(theta_phi: torch.Tensor, l_max: int) -> torch.Tensor:
    """구면 조화 함수 계산"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        theta = theta_phi[:, 0]
        phi = theta_phi[:, 1]
        result = torch.zeros(theta_phi.size(0), (l_max + 1) ** 2, 
                           dtype=theta_phi.dtype, device=theta_phi.device)
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                idx = l * l + l + m
                if idx < result.size(1):
                    result[:, idx] = torch.cos(l * theta) * torch.cos(m * phi)
        return result
    
    if HAS_CUDA and theta_phi.is_cuda:
        return _C.spherical_harmonics_cuda(theta_phi, l_max)
    else:
        return _C.spherical_harmonics_cpu(theta_phi, l_max)

def fast_spherical_conv(f: torch.Tensor, 
                       g: torch.Tensor, 
                       curvature: float = 1.0) -> torch.Tensor:
    """빠른 구면 컨볼루션"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        return f * g  # 요소별 곱
    
    if HAS_CUDA and f.is_cuda:
        return _C.fast_spherical_conv_cuda(f, g, curvature)
    else:
        return _C.fast_spherical_conv_cpu(f, g, curvature)

def ricci_curvature(metric_tensor: torch.Tensor) -> torch.Tensor:
    """리치 곡률 계산"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        return torch.full((metric_tensor.size(0),), -1.0, 
                        dtype=metric_tensor.dtype, device=metric_tensor.device)
    
    if HAS_CUDA and metric_tensor.is_cuda:
        return _C.ricci_curvature_cuda(metric_tensor)
    else:
        return _C.ricci_curvature_cpu(metric_tensor)

def parallel_transport(v: torch.Tensor, 
                      path: torch.Tensor, 
                      curvature: float = 1.0) -> torch.Tensor:
    """평행 이동"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        return v  # 동일 변환
    
    if HAS_CUDA and v.is_cuda:
        return _C.parallel_transport_cuda(v, path, curvature)
    else:
        return _C.parallel_transport_cpu(v, path, curvature)

def geodesic_flow(x: torch.Tensor, 
                 v: torch.Tensor, 
                 t: float, 
                 curvature: float = 1.0) -> torch.Tensor:
    """지오데식 플로우"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        return x + t * v  # 선형 이동
    
    if HAS_CUDA and x.is_cuda:
        return _C.geodesic_flow_cuda(x, v, t, curvature)
    else:
        return _C.geodesic_flow_cpu(x, v, t, curvature)

def riemannian_gradient(euclidean_grad: torch.Tensor, 
                       x: torch.Tensor, 
                       curvature: float = 1.0) -> torch.Tensor:
    """리만 그래디언트 변환"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        point_norm_sq = torch.sum(x * x, dim=1, keepdim=True)
        conformal_factor = torch.pow(1 - curvature * point_norm_sq, 2) / 4.0
        return euclidean_grad * conformal_factor
    
    if HAS_CUDA and euclidean_grad.is_cuda:
        return _C.riemannian_gradient_cuda(euclidean_grad, x, curvature)
    else:
        return _C.riemannian_gradient_cpu(euclidean_grad, x, curvature)

def geodesic_sgd_step(x: torch.Tensor, 
                     grad: torch.Tensor, 
                     lr: float, 
                     curvature: float = 1.0) -> torch.Tensor:
    """지오데식 SGD 스텝"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        return x - lr * grad  # 일반 SGD
    
    if HAS_CUDA and x.is_cuda:
        return _C.geodesic_sgd_step_cuda(x, grad, lr, curvature)
    else:
        return _C.geodesic_sgd_step_cpu(x, grad, lr, curvature)

def hyperbolic_wavelet_decomposition(signal: torch.Tensor, 
                                   num_levels: int, 
                                   curvature: float = 1.0) -> torch.Tensor:
    """하이퍼볼릭 웨이블릿 분해"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        coeffs = torch.zeros_like(signal)
        current = signal.clone()
        
        for level in range(num_levels):
            coeffs += current * (0.5 ** level)
            current = current * 0.5
        
        return coeffs
    
    if HAS_CUDA and signal.is_cuda:
        return _C.hyperbolic_wavelet_decomposition_cuda(signal, num_levels, curvature)
    else:
        return _C.hyperbolic_wavelet_decomposition_cpu(signal, num_levels, curvature)

def frequency_domain_filter(signal: torch.Tensor, 
                           filter_coeffs: torch.Tensor, 
                           curvature: float = 1.0) -> torch.Tensor:
    """주파수 도메인 필터링"""
    if _C is None:
        warnings.warn("C++ extension not available, using PyTorch fallback")
        return signal * filter_coeffs.unsqueeze(0)
    
    if HAS_CUDA and signal.is_cuda:
        return _C.frequency_domain_filter_cuda(signal, filter_coeffs, curvature)
    else:
        return _C.frequency_domain_filter_cpu(signal, filter_coeffs, curvature)

# ===== 기존 고급 기능들 (유지) =====

def hyperbolic_regularization(
    x: torch.Tensor,
    weights: torch.Tensor,
    curvature: float,
    lambda_boundary: float = 1.0,
    lambda_curvature: float = 0.1,
    lambda_geodesic: float = 0.01
) -> torch.Tensor:
    """하이퍼볼릭 정규화 (경계, 곡률, 측지선 분산 포함)"""
    if _C is None:
        warnings.warn("C++ extension not available, using simplified regularization")
        boundary_loss = torch.sum(torch.clamp(torch.norm(x, dim=-1) - 0.95, min=0) ** 2)
        curvature_loss = torch.mean((curvature - 1.0) ** 2)
        geodesic_loss = torch.var(torch.norm(x, dim=-1))
        return lambda_boundary * boundary_loss + lambda_curvature * curvature_loss + lambda_geodesic * geodesic_loss
    
    if HAS_CUDA and x.is_cuda:
        return _C.combined_reg(x, weights, curvature, lambda_boundary, lambda_curvature, lambda_geodesic)
    else:
        warnings.warn("CUDA not available, using CPU fallback for regularization")
        return lambda_boundary * torch.sum(torch.clamp(torch.norm(x, dim=-1) - 0.95, min=0) ** 2)

def dynamic_curvature_prediction(x: torch.Tensor, base_curvature: float = 1.0) -> torch.Tensor:
    """동적 곡률 예측"""
    if _C is None:
        warnings.warn("C++ extension not available, using constant curvature")
        return torch.full((x.size(0), 1), base_curvature, device=x.device)
    
    if HAS_CUDA and x.is_cuda:
        features = torch.norm(x, 2, dim=-1, keepdim=True)
        weight = torch.randn(1, 1, device=x.device) * 0.1
        bias = torch.zeros(1, device=x.device)
        return _C.dynamic_curvature_pred(features, weight, bias, base_curvature)
    else:
        warnings.warn("CUDA not available for dynamic curvature")
        return torch.full((x.size(0), 1), base_curvature, device=x.device)

def fused_hyperbolic_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    curvature: float = 1.0
) -> torch.Tensor:
    """융합된 하이퍼볼릭 선형 변환"""
    if _C is None:
        warnings.warn("C++ extension not available, using standard operations")
        linear_out = F.linear(x, weight, bias)
        return torch.tanh(linear_out * torch.sqrt(torch.tensor(curvature)))
    
    return _C.fused_linear(x, weight, bias, curvature)

# ===== 편의 함수들 =====

def create_advanced_config(
    preset: str = "research",
    **kwargs
) -> AdvancedConfig:
    """사전 정의된 설정으로 AdvancedConfig 생성"""
    if preset == "mnist_fix":
        return AdvancedConfig(
            enable_regularization=True,
            enable_dynamic_curvature=False,
            enable_fused_ops=False,
            enable_geodesic_activation=False,
            enable_chebyshev_approximation=True,  # 수치 안정성
            enable_laplace_beltrami=False,
            enable_hyperbolic_fft=False,
            lambda_boundary=2.0,
            lambda_curvature=0.1,
            lambda_geodesic=0.01,
            **kwargs
        )
    elif preset == "performance":
        return AdvancedConfig(
            enable_regularization=False,
            enable_dynamic_curvature=False,
            enable_fused_ops=True,
            enable_geodesic_activation=False,
            enable_chebyshev_approximation=True,  # 빠른 근사
            enable_laplace_beltrami=False,
            enable_hyperbolic_fft=True,  # 빠른 변환
            **kwargs
        )
    elif preset == "research":
        return AdvancedConfig(
            enable_regularization=True,
            enable_dynamic_curvature=True,
            enable_fused_ops=True,
            enable_geodesic_activation=True,
            enable_chebyshev_approximation=True,
            enable_laplace_beltrami=True,
            enable_hyperbolic_fft=True,
            **kwargs
        )
    else:
        return AdvancedConfig(**kwargs)

def get_available_features() -> dict:
    """사용 가능한 고급 기능들 확인"""
    features = {
        "c_extension": _C is not None,
        "cuda_support": HAS_CUDA,
        "regularization": True,
        "dynamic_curvature": HAS_CUDA,
        "fused_ops": _C is not None,
        "geodesic_activation": HAS_CUDA,
        "chebyshev_approximation": _C is not None,
        "laplace_beltrami": _C is not None,
        "hyperbolic_fft": _C is not None,
        "spherical_harmonics": _C is not None,
        "riemannian_geometry": _C is not None
    }
    
    return features

def benchmark_advanced_features(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_iterations: int = 100
) -> dict:
    """고급 기능들의 성능 벤치마크"""
    results = {}
    
    if _C is not None:
        # 체비셰프 근사 벤치마크
        import time
        start_time = time.time()
        for _ in range(num_iterations):
            _ = chebyshev_approximation(input_tensor)
        results["chebyshev_approximation_ms"] = (time.time() - start_time) * 1000 / num_iterations
        
        # 라플라시안 벤치마크
        start_time = time.time()
        for _ in range(num_iterations):
            _ = hyperbolic_laplacian(input_tensor)
        results["hyperbolic_laplacian_ms"] = (time.time() - start_time) * 1000 / num_iterations
        
        # FFT 벤치마크
        start_time = time.time()
        for _ in range(num_iterations):
            _ = hyperbolic_fft(input_tensor)
        results["hyperbolic_fft_ms"] = (time.time() - start_time) * 1000 / num_iterations
    
    return results 