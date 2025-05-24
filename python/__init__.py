import torch
from torch.autograd import Function

from ._C import (
    poincare_ball_forward_cpu, poincare_ball_backward_cpu,
    lorentz_forward_cpu,     lorentz_backward_cpu,
    klein_forward_cpu,       klein_backward_cpu,
    mobius_add_cpu,          mobius_scalar_cpu,
)

_has_cuda = False
if torch.cuda.is_available():
    try:
        from ._C import (
            poincare_ball_forward_cuda, poincare_ball_backward_cuda,
            lorentz_forward_cuda,     lorentz_backward_cuda,
            klein_forward_cuda,       klein_backward_cuda,
            mobius_add_cuda,          mobius_scalar_cuda,
        )
        _has_cuda = True
        print("CUDA is available")
    except ImportError:
        _has_cuda = False


class PoincareBall(Function):
    @staticmethod
    def forward(ctx, u, v, c, t):
        ctx.save_for_backward(u, v)
        ctx.c, ctx.t = c, t
        if u.is_cuda and _has_cuda:
            return poincare_ball_forward_cuda(u, v, c, t)
        else:
            return poincare_ball_forward_cpu(u, v, c, t)

    @staticmethod
    def backward(ctx, grad_output):
        u, v = ctx.saved_tensors
        c, t = ctx.c, ctx.t
        if u.is_cuda and _has_cuda:
            grad_u, grad_v = poincare_ball_backward_cuda(grad_output, u, v, c, t)
        else:
            grad_u, grad_v = poincare_ball_backward_cpu(grad_output, u, v, c, t)
        return grad_u, grad_v, None, None


class LorentzModel(Function):
    @staticmethod
    def forward(ctx, u, v, c, t):
        ctx.save_for_backward(u, v)
        ctx.c, ctx.t = c, t
        if u.is_cuda and _has_cuda:
            return lorentz_forward_cuda(u, v, c, t)
        else:
            return lorentz_forward_cpu(u, v, c, t)

    @staticmethod
    def backward(ctx, grad_output):
        u, v = ctx.saved_tensors
        c, t = ctx.c, ctx.t
        if u.is_cuda and _has_cuda:
            grad_u, grad_v = lorentz_backward_cuda(grad_output, u, v, c, t)
        else:
            grad_u, grad_v = lorentz_backward_cpu(grad_output, u, v, c, t)
        return grad_u, grad_v, None, None


class KleinModel(Function):
    @staticmethod
    def forward(ctx, u, v, c, t):
        ctx.save_for_backward(u, v)
        ctx.c, ctx.t = c, t
        if u.is_cuda and _has_cuda:
            return klein_forward_cuda(u, v, c, t)
        else:
            return klein_forward_cpu(u, v, c, t)

    @staticmethod
    def backward(ctx, grad_output):
        u, v = ctx.saved_tensors
        c, t = ctx.c, ctx.t
        if u.is_cuda and _has_cuda:
            grad_u, grad_v = klein_backward_cuda(grad_output, u, v, c, t)
        else:
            grad_u, grad_v = klein_backward_cpu(grad_output, u, v, c, t)
        return grad_u, grad_v, None, None


# Python API
def poincare_ball_layer(u, v, c, t):
    return PoincareBall.apply(u, v, c, t)

def lorentz_layer(u, v, c, t):
    return LorentzModel.apply(u, v, c, t)

def klein_layer(u, v, c, t):
    return KleinModel.apply(u, v, c, t)


# conversions
from ._C import (
    poincare_to_lorentz_cpu, lorentz_to_poincare_cpu,
    poincare_to_klein_cpu,   klein_to_poincare_cpu,
    lorentz_to_klein_cpu,    klein_to_lorentz_cpu,
)
if _has_cuda:
    from ._C import (
        poincare_to_lorentz_cuda, lorentz_to_poincare_cuda,
        poincare_to_klein_cuda,   klein_to_poincare_cuda,
        lorentz_to_klein_cuda,    klein_to_lorentz_cuda,
    )

def poincare_to_lorentz(x, c):
    fn = poincare_to_lorentz_cuda if (x.is_cuda and _has_cuda) else poincare_to_lorentz_cpu
    return fn(x, c)

def lorentz_to_poincare(x, c):
    fn = lorentz_to_poincare_cuda if (x.is_cuda and _has_cuda) else lorentz_to_poincare_cpu
    return fn(x, c)

def poincare_to_klein(x, c):
    fn = poincare_to_klein_cuda if (x.is_cuda and _has_cuda) else poincare_to_klein_cpu
    return fn(x, c)

def klein_to_poincare(x, c):
    fn = klein_to_poincare_cuda if (x.is_cuda and _has_cuda) else klein_to_poincare_cpu
    return fn(x, c)

def lorentz_to_klein(x, c):
    fn = lorentz_to_klein_cuda if (x.is_cuda and _has_cuda) else lorentz_to_klein_cpu
    return fn(x, c)

def klein_to_lorentz(x, c):
    fn = klein_to_lorentz_cuda if (x.is_cuda and _has_cuda) else klein_to_lorentz_cpu
    return fn(x, c)


def mobius_add(x, y, c):
    fn = mobius_add_cuda if (x.is_cuda and _has_cuda) else mobius_add_cpu
    return fn(x, y, c)

def mobius_scalar(x, r, c):
    fn = mobius_scalar_cuda if (x.is_cuda and _has_cuda) else mobius_scalar_cpu
    return fn(x, r, c)

# ===============================
# 🔥 ADVANCED FEATURES 🔥
# ===============================

# Advanced 기능들 import
from .advanced import (
    # 설정 클래스
    AdvancedConfig, BenchmarkResult,
    
    # 동적 곡률
    predict_dynamic_curvature, dynamic_mobius_add,
    
    # 하이퍼볼릭 정규화
    hyperbolic_regularization,
    
    # 측지선 활성화
    geodesic_activation, einstein_midpoint,
    
    # Fused 연산들
    hyperbolic_linear_fused, transform_regularize_fused,
    
    # 🆕 새로 추가된 체비셰프 기능들 🆕
    chebyshev_approximation, chebyshev_distance, chebyshev_nodes,
    fast_chebyshev_transform, inverse_chebyshev_transform,
    chebyshev_derivative, chebyshev_integral,
    
    # 🆕 새로 추가된 라플라스-벨트라미 기능들 🆕
    hyperbolic_laplacian, heat_kernel, laplace_beltrami_eigen,
    spectral_graph_conv, solve_diffusion_equation,
    geodesic_distance_matrix, spectral_normalize,
    
    # 🆕 새로 추가된 FFT 및 리만 기하학 기능들 🆕
    hyperbolic_fft, spherical_harmonics, fast_spherical_conv,
    ricci_curvature, parallel_transport, geodesic_flow,
    riemannian_gradient, geodesic_sgd_step,
    hyperbolic_wavelet_decomposition, frequency_domain_filter,
    
    # 편의 함수들
    fix_mnist_nan, benchmark_advanced_features
)

# 고급 레이어들 import
from .layers import (
    # 레이어 클래스들
    DynamicCurvatureLayer, HyperbolicLinearAdvanced, 
    GeodesicActivationLayer, RegularizedHyperbolicLayer,
    AdvancedHyperbolicMLP, DynamicCurvatureMLP, FusedHyperbolicLayer,
    
    # 팩토리 함수들
    create_mnist_model, create_performance_model, create_research_model
)

# 성능 최적화 import
from .optimizations import (
    # 설정 클래스들
    OptimizationConfig,
    
    # 최적화 도구들
    OptimizedModel, AdaptiveBatchSize, MemoryOptimizer,
    
    # 벤치마크 도구들
    benchmark_model_performance, optimize_for_inference,
    
    # 프로파일링
    enable_profiling, disable_profiling, print_performance_summary,
    
    # 빠른 설정 함수들
    quick_setup_for_mnist, quick_setup_for_production, quick_setup_for_research,
    
    # 작업별 설정
    create_optimized_config_for_task, setup_optimizations
)

# ===============================
# Quick Start Functions
# ===============================

def create_advanced_mnist_model(enable_all_features=False):
    """MNIST용 고급 모델 생성 (빠른 시작)
    
    Args:
        enable_all_features: 모든 고급 기능 활성화 여부
        
    Returns:
        nn.Module: MNIST 분류용 고급 모델
    """
    if enable_all_features:
        # 연구용: 모든 기능 활성화
        return create_research_model(784, 10, [128, 64])
    else:
        # 실용적: NaN 문제 해결 + 성능 최적화
        return create_mnist_model()

def setup_reality_stone_for_training():
    """훈련용 Reality Stone 설정"""
    quick_setup_for_mnist()
    return create_advanced_mnist_model()

def setup_reality_stone_for_inference():
    """추론용 Reality Stone 설정"""
    quick_setup_for_production()
    return create_performance_model(784, 10)

def setup_reality_stone_for_research():
    """연구용 Reality Stone 설정"""
    quick_setup_for_research()
    return create_research_model(784, 10, [256, 128, 64])

# ===============================
# Compatibility & Convenience
# ===============================

# 기존 models.py의 클래스들 import (하위 호환성)
from .models import LorentzMLP, KleinMLP

# 하위 호환성을 위한 별칭들
HyperbolicLinear = HyperbolicLinearAdvanced
HyperbolicMLP = AdvancedHyperbolicMLP

# 편의 함수들
def quick_fix_nan(tensor, curvature=1.0):
    """NaN 문제 빠른 해결 (별칭)"""
    return fix_mnist_nan(tensor, curvature)

def benchmark_performance(model, input_shape, device="cuda"):
    """성능 벤치마크 (간단 버전)"""
    return benchmark_model_performance(model, input_shape, device)

# ===============================
# Version & Feature Detection
# ===============================

__version__ = "2.0.0-advanced"

def get_available_features():
    """사용 가능한 기능들 반환"""
    features = {
        "basic_operations": True,
        "cuda_support": _has_cuda,
        "advanced_features": True,  # 새로 추가된 고급 기능들
        "dynamic_curvature": True,
        "hyperbolic_regularization": True,
        "geodesic_activation": True,
        "fused_operations": True,
        "performance_optimization": True
    }
    return features

def print_feature_status():
    """기능 상태 출력"""
    features = get_available_features()
    print("\n" + "="*50)
    print("Reality Stone Feature Status")
    print("="*50)
    for feature, available in features.items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"{feature:25}: {status}")
    print("="*50)

# ===============================
# Advanced API Shortcuts
# ===============================

# 자주 사용되는 고급 기능들의 단축 경로
class advanced:
    """고급 기능 네임스페이스"""
    
    # 설정
    Config = AdvancedConfig
    OptimConfig = OptimizationConfig
    
    # 레이어들
    Linear = HyperbolicLinearAdvanced
    MLP = AdvancedHyperbolicMLP
    DynamicMLP = DynamicCurvatureMLP
    GeodesicActivation = GeodesicActivationLayer
    
    # 함수들
    predict_curvature = predict_dynamic_curvature
    regularize = hyperbolic_regularization
    fused_linear = hyperbolic_linear_fused
    fix_nan = fix_mnist_nan
    
    # 🆕 체비셰프 관련 🆕
    chebyshev_approx = chebyshev_approximation
    chebyshev_dist = chebyshev_distance
    chebyshev_transform = fast_chebyshev_transform
    chebyshev_inverse = inverse_chebyshev_transform
    
    # 🆕 라플라스-벨트라미 관련 🆕
    laplacian = hyperbolic_laplacian
    heat_kernel = heat_kernel
    spectral_conv = spectral_graph_conv
    distance_matrix = geodesic_distance_matrix
    
    # 🆕 FFT 및 리만 기하학 관련 🆕
    fft = hyperbolic_fft
    spherical_harm = spherical_harmonics
    ricci = ricci_curvature
    transport = parallel_transport
    flow = geodesic_flow
    riem_grad = riemannian_gradient
    geo_sgd = geodesic_sgd_step
    wavelet = hyperbolic_wavelet_decomposition
    filter_freq = frequency_domain_filter
    
    # 팩토리
    create_mnist = create_mnist_model
    create_performance = create_performance_model
    create_research = create_research_model

class optim:
    """최적화 네임스페이스"""
    
    Config = OptimizationConfig
    OptimizedModel = OptimizedModel
    AdaptiveBatch = AdaptiveBatchSize
    MemoryOpt = MemoryOptimizer
    
    # 빠른 설정
    setup_mnist = quick_setup_for_mnist
    setup_production = quick_setup_for_production
    setup_research = quick_setup_for_research
    
    # 벤치마크
    benchmark = benchmark_model_performance
    profile_enable = enable_profiling
    profile_disable = disable_profiling
    profile_summary = print_performance_summary

# ===============================
# Examples & Tutorials
# ===============================

def show_example_usage():
    """사용 예제 출력"""
    print("""
    Reality Stone Advanced Features - Usage Examples
    ================================================
    
    # 1. 빠른 시작 (MNIST NaN 문제 해결)
    import reality_stone as rs
    model = rs.setup_reality_stone_for_training()
    
    # 2. 고급 기능 사용
    config = rs.AdvancedConfig(
        enable_dynamic_curvature=True,
        enable_fused_ops=True,
        enable_regularization=True
    )
    model = rs.create_research_model(784, 10, config=config)
    
    # 3. 성능 최적화
    rs.quick_setup_for_production()
    model = rs.OptimizedModel(model)
    
    # 4. 단축 경로 사용
    model = rs.advanced.create_mnist()
    rs.optim.setup_production()
    
    # 5. 기능 상태 확인
    rs.print_feature_status()
    
    # 6. 성능 벤치마크
    results = rs.benchmark_performance(model, (32, 784))
    
    ================================================
    """)

# ===============================
# 🆕 새로 추가된 고급 API 별칭 🆕
# ===============================

# advanced_api 별칭으로 새로운 함수들에 접근 가능
class advanced_api:
    """새로운 고급 기능들을 위한 API (테스트에서 사용)"""
    
    # 체비셰프 함수들
    chebyshev_approximation = chebyshev_approximation
    chebyshev_distance = chebyshev_distance
    chebyshev_nodes = chebyshev_nodes
    fast_chebyshev_transform = fast_chebyshev_transform
    inverse_chebyshev_transform = inverse_chebyshev_transform
    chebyshev_derivative = chebyshev_derivative
    chebyshev_integral = chebyshev_integral
    
    # 라플라스-벨트라미 함수들
    hyperbolic_laplacian = hyperbolic_laplacian
    heat_kernel = heat_kernel
    laplace_beltrami_eigen = laplace_beltrami_eigen
    spectral_graph_conv = spectral_graph_conv
    solve_diffusion_equation = solve_diffusion_equation
    geodesic_distance_matrix = geodesic_distance_matrix
    spectral_normalize = spectral_normalize
    
    # FFT 및 리만 기하학 함수들
    hyperbolic_fft = hyperbolic_fft
    spherical_harmonics = spherical_harmonics
    fast_spherical_conv = fast_spherical_conv
    ricci_curvature = ricci_curvature
    parallel_transport = parallel_transport
    geodesic_flow = geodesic_flow
    riemannian_gradient = riemannian_gradient
    geodesic_sgd_step = geodesic_sgd_step
    hyperbolic_wavelet_decomposition = hyperbolic_wavelet_decomposition
    frequency_domain_filter = frequency_domain_filter

# 자동으로 기능 상태 표시 (옵션)
import os
if os.getenv('REALITY_STONE_VERBOSE', '0') == '1':
    print_feature_status()