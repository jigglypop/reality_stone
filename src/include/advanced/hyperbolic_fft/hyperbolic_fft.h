#pragma once

#include <torch/extension.h>
#include <config/constant.h>

namespace reality_stone::advanced {

// CUDA 함수 선언들
torch::Tensor hyperbolic_fft_cuda(
    const torch::Tensor& x,
    float curvature
);

torch::Tensor spherical_harmonics_cuda(
    const torch::Tensor& theta_phi,
    int l_max
);

torch::Tensor fast_spherical_conv_cuda(
    const torch::Tensor& f,
    const torch::Tensor& g,
    float curvature
);

torch::Tensor ricci_curvature_cuda(
    const torch::Tensor& metric_tensor
);

torch::Tensor parallel_transport_cuda(
    const torch::Tensor& v,
    const torch::Tensor& path,
    float curvature
);

torch::Tensor geodesic_flow_cuda(
    const torch::Tensor& x,
    const torch::Tensor& v,
    float t,
    float curvature
);

torch::Tensor riemannian_gradient_cuda(
    const torch::Tensor& euclidean_grad,
    const torch::Tensor& x,
    float curvature
);

torch::Tensor geodesic_sgd_step_cuda(
    const torch::Tensor& x,
    const torch::Tensor& grad,
    float lr,
    float curvature
);

torch::Tensor hyperbolic_wavelet_decomposition_cuda(
    const torch::Tensor& signal,
    int num_levels,
    float curvature
);

torch::Tensor frequency_domain_filter_cuda(
    const torch::Tensor& signal,
    const torch::Tensor& filter,
    float curvature
);

/**
 * 하이퍼볼릭 푸리에 변환 클래스
 * O(n log n) 복잡도로 하이퍼볼릭 공간에서의 주파수 분석
 */
class HyperbolicFFT {
public:
    HyperbolicFFT(float curvature = 1.0f, int max_l = 20);
    
    // 하이퍼볼릭 푸리에 변환
    torch::Tensor forward_transform(const torch::Tensor& x);
    
    // 역 하이퍼볼릭 푸리에 변환
    torch::Tensor inverse_transform(const torch::Tensor& coeffs);
    
    // 구면 조화 함수 계산
    torch::Tensor compute_spherical_harmonics(
        const torch::Tensor& theta_phi,
        int l,
        int m
    );
    
    // 고속 구면 컨볼루션
    torch::Tensor fast_spherical_convolution(
        const torch::Tensor& f,
        const torch::Tensor& g
    );
    
    // 주파수 도메인 필터링
    torch::Tensor frequency_filter(
        const torch::Tensor& x,
        const torch::Tensor& filter_coeffs
    );
    
    // 하이퍼볼릭 웨이블릿 변환
    torch::Tensor wavelet_transform(
        const torch::Tensor& x,
        int num_scales
    );
    
private:
    float curvature;
    int max_l;
    torch::Tensor legendre_cache;
    torch::Tensor spherical_cache;
    bool cache_valid;
    
    // 내부 도구 함수들
    torch::Tensor compute_legendre_polynomials(
        const torch::Tensor& x,
        int max_degree
    );
    
    torch::Tensor compute_associated_legendre(
        const torch::Tensor& x,
        int l,
        int m
    );
    
    std::complex<float> spherical_harmonic_coeff(int l, int m);
};

/**
 * 리만 기하학 연산자 클래스
 * 곡률, 평행 이동, 지오데식 플로우 등
 */
class RiemannianOperator {
public:
    RiemannianOperator(float curvature = 1.0f);
    
    // 리치 곡률 계산
    torch::Tensor compute_ricci_curvature(
        const torch::Tensor& metric_tensor
    );
    
    // 평행 이동
    torch::Tensor parallel_transport(
        const torch::Tensor& vector,
        const torch::Tensor& path,
        float path_parameter
    );
    
    // 지오데식 플로우
    torch::Tensor geodesic_flow(
        const torch::Tensor& initial_point,
        const torch::Tensor& initial_velocity,
        float time
    );
    
    // 리만 그래디언트
    torch::Tensor riemannian_gradient(
        const torch::Tensor& euclidean_grad,
        const torch::Tensor& point
    );
    
    // 지오데식 최적화 스텝
    torch::Tensor geodesic_sgd_step(
        const torch::Tensor& point,
        const torch::Tensor& grad,
        float learning_rate
    );
    
    // 크리스토펠 기호 계산
    torch::Tensor christoffel_symbols(
        const torch::Tensor& metric_tensor
    );
    
    // 곡률 텐서
    torch::Tensor riemann_curvature_tensor(
        const torch::Tensor& metric_tensor
    );
    
private:
    float curvature;
    
    // 내부 도구 함수들
    torch::Tensor metric_tensor_poincare(const torch::Tensor& x);
    torch::Tensor metric_inverse_poincare(const torch::Tensor& x);
    torch::Tensor connection_coefficients(
        const torch::Tensor& x,
        int i,
        int j,
        int k
    );
};

/**
 * 편의 함수들 (절차적 인터페이스)
 */

// 하이퍼볼릭 푸리에 변환
torch::Tensor hyperbolic_fft_cpu(
    const torch::Tensor& x,
    float curvature = 1.0f
);

// 구면 조화 함수
torch::Tensor spherical_harmonics_cpu(
    const torch::Tensor& theta_phi,
    int l_max
);

// 빠른 구면 컨볼루션
torch::Tensor fast_spherical_conv_cpu(
    const torch::Tensor& f,
    const torch::Tensor& g,
    float curvature = 1.0f
);

// 리치 곡률
torch::Tensor ricci_curvature_cpu(
    const torch::Tensor& metric_tensor
);

// 평행 이동
torch::Tensor parallel_transport_cpu(
    const torch::Tensor& v,
    const torch::Tensor& path,
    float curvature = 1.0f
);

// 지오데식 플로우
torch::Tensor geodesic_flow_cpu(
    const torch::Tensor& x,
    const torch::Tensor& v,
    float t,
    float curvature = 1.0f
);

// 리만 최적화
torch::Tensor riemannian_gradient_cpu(
    const torch::Tensor& euclidean_grad,
    const torch::Tensor& x,
    float curvature = 1.0f
);

// 지오데식 최적화
torch::Tensor geodesic_sgd_step_cpu(
    const torch::Tensor& x,
    const torch::Tensor& grad,
    float lr,
    float curvature = 1.0f
);

// 하이퍼볼릭 웨이블릿 분해
torch::Tensor hyperbolic_wavelet_decomposition_cpu(
    const torch::Tensor& signal,
    int num_levels,
    float curvature = 1.0f
);

// 주파수 도메인 필터링
torch::Tensor frequency_domain_filter_cpu(
    const torch::Tensor& signal,
    const torch::Tensor& filter,
    float curvature = 1.0f
);

} // namespace reality_stone::advanced 