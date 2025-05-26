#pragma once

#include <torch/extension.h>
#include <config/constant.h>

namespace reality_stone::advanced {

// CUDA 함수 선언들
torch::Tensor chebyshev_approximation_cuda(
    const torch::Tensor& x,
    int order,
    float curvature
);

torch::Tensor chebyshev_distance_cuda(
    const torch::Tensor& x,
    const torch::Tensor& y,
    float curvature
);

torch::Tensor fast_chebyshev_transform_cuda(
    const torch::Tensor& values
);

torch::Tensor inverse_chebyshev_transform_cuda(
    const torch::Tensor& coeffs,
    const torch::Tensor& eval_points
);

torch::Tensor chebyshev_derivative_cuda(
    const torch::Tensor& coeffs
);

torch::Tensor chebyshev_integral_cuda(
    const torch::Tensor& coeffs,
    float constant = 0.0f
);

/**
 * 체비셰프 다항식 기반 하이퍼볼릭 함수 근사기
 * 수치적 안정성을 위해 체비셰프 급수 전개 사용
 */
class ChebyshevApproximator {
public:
    ChebyshevApproximator(int order = 10, float base_curvature = 1.0f);
    
    // 하이퍼볼릭 함수 근사
    torch::Tensor approximate_tanh(const torch::Tensor& x, float curvature);
    
    // 체비셰프 거리 계산 (Klein 모델)
    torch::Tensor compute_distance(
        const torch::Tensor& x,
        const torch::Tensor& y,
        float curvature
    );
    
    // 체비셰프 계수 계산
    torch::Tensor compute_coefficients(const torch::Tensor& func_values);
    
    // 고속 변환 (FFT 기반)
    torch::Tensor fast_transform(const torch::Tensor& values);
    
    // 역변환 (계수 -> 함수값)
    torch::Tensor inverse_transform(
        const torch::Tensor& coeffs,
        const torch::Tensor& eval_points
    );
    
private:
    int order;
    float base_curvature;
    torch::Tensor cheb_nodes;  // 체비셰프 점들
    torch::Tensor coeffs_cache; // 캐시된 계수들
    
    // 내부 도구 함수들
    torch::Tensor generate_nodes(int n, torch::Device device);
    torch::Tensor evaluate_polynomial(
        const torch::Tensor& coeffs,
        const torch::Tensor& x
    );
};

/**
 * 편의 함수들 (절차적 인터페이스)
 */

// 체비셰프 근사 (메인 함수)
torch::Tensor chebyshev_approximation_cpu(
    const torch::Tensor& x,
    int order = 10,
    float curvature = 1.0f
);

// 체비셰프 거리
torch::Tensor chebyshev_distance_cpu(
    const torch::Tensor& x,
    const torch::Tensor& y,
    float curvature = 1.0f
);

// 체비셰프 점들 생성
torch::Tensor chebyshev_nodes_cpu(
    int n,
    torch::Device device = torch::kCPU
);

// 체비셰프 계수 계산
torch::Tensor chebyshev_coefficients_cpu(
    const torch::Tensor& func_values,
    int order
);

// 고속 체비셰프 변환
torch::Tensor fast_chebyshev_transform_cpu(
    const torch::Tensor& values
);

torch::Tensor inverse_chebyshev_transform_cpu(
    const torch::Tensor& coeffs,
    const torch::Tensor& eval_points
);

torch::Tensor chebyshev_derivative_cpu(
    const torch::Tensor& coeffs
);

torch::Tensor chebyshev_integral_cpu(
    const torch::Tensor& coeffs,
    float constant = 0.0f
);
} 