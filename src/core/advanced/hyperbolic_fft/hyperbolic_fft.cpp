#include <torch/extension.h>
#include <advanced/hyperbolic_fft/hyperbolic_fft.h>
#include <cmath>
#include <complex>

namespace reality_stone::advanced {

// HyperbolicFFT 클래스 구현
HyperbolicFFT::HyperbolicFFT(float curvature, int max_l) 
    : curvature(curvature), max_l(max_l), cache_valid(false) {
}

torch::Tensor HyperbolicFFT::forward_transform(const torch::Tensor& x) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    
    TORCH_CHECK(dim >= 2, "Input dimension must be at least 2 for spherical coordinates");
    
    // 하이퍼볼릭 공간에서 구면 좌표로 변환
    auto r = torch::norm(x, 2, 1, true); // [B, 1]
    
    // Poincaré 디스크에서 각도 계산
    torch::Tensor theta, phi;
    if (dim == 2) {
        theta = torch::atan2(x.select(1, 1), x.select(1, 0)).unsqueeze(1); // [B, 1]
        phi = torch::zeros_like(theta);
    } else { // dim >= 3
        auto xy_norm = torch::norm(x.narrow(1, 0, 2), 2, 1, true);
        theta = torch::atan2(xy_norm.squeeze(1), x.select(1, 2)).unsqueeze(1); // [B, 1]
        phi = torch::atan2(x.select(1, 1), x.select(1, 0)).unsqueeze(1); // [B, 1]
    }
    
    auto theta_phi = torch::cat({theta, phi}, 1); // [B, 2]
    
    // 구면 조화 함수 계산
    auto total_coeffs = (max_l + 1) * (max_l + 1);
    auto coeffs = torch::zeros({batch_size, total_coeffs}, x.options());
    
    for (int l = 0; l <= max_l; ++l) {
        for (int m = -l; m <= l; ++m) {
            auto harmonic = compute_spherical_harmonics(theta_phi, l, m); // [B, 1]
            int idx = l * l + l + m;
            
            // 하이퍼볼릭 가중치 적용
            auto sqrt_c = std::sqrt(curvature);
            auto hyperbolic_weight = torch::tanh(sqrt_c * r).squeeze(1); // [B]
            
            coeffs.select(1, idx) = harmonic.squeeze(1) * hyperbolic_weight;
        }
    }
    
    return coeffs;
}

torch::Tensor HyperbolicFFT::inverse_transform(const torch::Tensor& coeffs) {
    auto batch_size = coeffs.size(0);
    auto total_coeffs = coeffs.size(1);
    
    // 출력 차원을 3D로 설정 (x, y, z)
    auto result = torch::zeros({batch_size, 3}, coeffs.options());
    
    for (int l = 0; l <= max_l; ++l) {
        for (int m = -l; m <= l; ++m) {
            int idx = l * l + l + m;
            if (idx >= total_coeffs) break;
            
            auto coeff_val = coeffs.select(1, idx); // [B]
            
            // 구면 좌표에서 직교 좌표로 변환
            float theta_val = M_PI * l / (2.0f * max_l);  // 균등 분포
            float phi_val = 2.0f * M_PI * m / (2.0f * l + 1.0f);
            
            auto sin_theta = std::sin(theta_val);
            auto cos_theta = std::cos(theta_val);
            auto sin_phi = std::sin(phi_val);
            auto cos_phi = std::cos(phi_val);
            
            // 하이퍼볼릭 반지름 계산
            auto sqrt_c = std::sqrt(curvature);
            auto hyperbolic_r = torch::atanh(torch::abs(coeff_val).clamp_max(0.99f)) / sqrt_c;
            
            // 직교 좌표 계산
            result.select(1, 0) += hyperbolic_r * sin_theta * cos_phi; // x
            result.select(1, 1) += hyperbolic_r * sin_theta * sin_phi; // y
            result.select(1, 2) += hyperbolic_r * cos_theta;           // z
        }
    }
    
    return result;
}

torch::Tensor HyperbolicFFT::compute_spherical_harmonics(
    const torch::Tensor& theta_phi,
    int l,
    int m
) {
    auto batch_size = theta_phi.size(0);
    auto theta = theta_phi.select(1, 0); // [B]
    auto phi = theta_phi.select(1, 1);   // [B]
    
    // Associated Legendre 다항식 계산
    auto cos_theta = torch::cos(theta);
    auto legendre = compute_associated_legendre(cos_theta, l, std::abs(m));
    
    // 정규화 인수
    auto factorial_ratio = 1.0f;
    for (int k = l - std::abs(m) + 1; k <= l + std::abs(m); ++k) {
        factorial_ratio *= k;
    }
    auto normalization = std::sqrt((2 * l + 1) * factorial_ratio / (4 * M_PI));
    
    // 구면 조화 함수
    torch::Tensor result;
    if (m >= 0) {
        result = normalization * legendre * torch::cos(m * phi);
    } else {
        result = normalization * legendre * torch::sin(std::abs(m) * phi);
    }
    
    return result.unsqueeze(1); // [B, 1]
}

torch::Tensor HyperbolicFFT::compute_associated_legendre(
    const torch::Tensor& x,
    int l,
    int m
) {
    TORCH_CHECK(m >= 0, "m must be non-negative for Associated Legendre polynomials");
    TORCH_CHECK(m <= l, "m must be <= l for Associated Legendre polynomials");
    
    if (l == 0 && m == 0) {
        return torch::ones_like(x);
    }
    
    auto sin_theta = torch::sqrt(1.0f - x * x);
    
    // P_m^m 계산
    auto pmm = torch::ones_like(x);
    if (m > 0) {
        auto somx2 = sin_theta;
        auto fact = 1.0f;
        for (int i = 1; i <= m; ++i) {
            pmm *= -fact * somx2;
            fact += 2.0f;
        }
    }
    
    if (l == m) {
        return pmm;
    }
    
    // P_{m+1}^m 계산
    auto pmmp1 = x * (2 * m + 1) * pmm;
    
    if (l == m + 1) {
        return pmmp1;
    }
    
    // 재귀 관계식으로 P_l^m 계산
    auto pll = pmm;
    auto plp1 = pmmp1;
    
    for (int ll = m + 2; ll <= l; ++ll) {
        auto pnew = ((2 * ll - 1) * x * plp1 - (ll + m - 1) * pll) / (ll - m);
        pll = plp1;
        plp1 = pnew;
    }
    
    return plp1;
}

torch::Tensor HyperbolicFFT::fast_spherical_convolution(
    const torch::Tensor& f,
    const torch::Tensor& g
) {
    // FFT 기반 컨볼루션 (O(n log n))
    auto f_fft = forward_transform(f);
    auto g_fft = forward_transform(g);
    
    // 주파수 도메인에서 요소별 곱셈
    auto result_fft = f_fft * g_fft;
    
    // 역변환
    return inverse_transform(result_fft);
}

// RiemannianOperator 클래스 구현
RiemannianOperator::RiemannianOperator(float curvature) 
    : curvature(curvature) {
}

torch::Tensor RiemannianOperator::compute_ricci_curvature(
    const torch::Tensor& metric_tensor
) {
    auto batch_size = metric_tensor.size(0);
    auto dim = metric_tensor.size(1);
    
    // 하이퍼볼릭 공간에서 리치 곡률은 상수
    // Ric = -(d-1) * c * g (여기서 g는 메트릭 텐서)
    auto ricci_scalar = -(dim - 1) * curvature;
    return torch::full({batch_size}, ricci_scalar, metric_tensor.options());
}

torch::Tensor RiemannianOperator::parallel_transport(
    const torch::Tensor& vector,
    const torch::Tensor& path,
    float path_parameter
) {
    auto batch_size = vector.size(0);
    auto dim = vector.size(1);
    
    TORCH_CHECK(path.size(1) >= 2 * dim, "Path must contain start and end points");
    
    // Poincaré 모델에서의 평행 이동
    auto start_point = path.narrow(1, 0, dim);      // [B, D]
    auto end_point = path.narrow(1, dim, dim);      // [B, D]
    
    // 평행 이동 인수 계산
    auto start_norm_sq = torch::sum(start_point * start_point, 1, true);
    auto vector_norm_sq = torch::sum(vector * vector, 1, true);
    auto dot_product = torch::sum(start_point * vector, 1, true);
    
    auto denominator = 1.0f - curvature * start_norm_sq;
    auto transport_factor = (1.0f - path_parameter * curvature * dot_product) / 
                           denominator.clamp_min(1e-6f);
    
    return vector * transport_factor.expand_as(vector);
}

torch::Tensor RiemannianOperator::geodesic_flow(
    const torch::Tensor& initial_point,
    const torch::Tensor& initial_velocity,
    float time
) {
    // 지오데식 플로우: exp_p(t * v) in Poincaré model
    auto velocity_norm = torch::norm(initial_velocity, 2, 1, true);
    auto sqrt_c = std::sqrt(curvature);
    
    // 하이퍼볼릭 삼각함수 사용
    auto arg = sqrt_c * velocity_norm * time;
    auto sinh_term = torch::sinh(arg) / (sqrt_c * velocity_norm.clamp_min(1e-6f));
    auto cosh_term = torch::cosh(arg);
    
    auto unit_velocity = initial_velocity / velocity_norm.clamp_min(1e-6f);
    
    return initial_point * cosh_term.expand_as(initial_point) + 
           unit_velocity * sinh_term.expand_as(unit_velocity);
}

torch::Tensor RiemannianOperator::riemannian_gradient(
    const torch::Tensor& euclidean_grad,
    const torch::Tensor& point
) {
    // Poincaré 모델에서의 리만 그래디언트 변환
    auto point_norm_sq = torch::sum(point * point, 1, true);
    auto conformal_factor = torch::pow(1 - curvature * point_norm_sq, 2) / 4.0f;
    
    return euclidean_grad * conformal_factor.expand_as(euclidean_grad);
}

torch::Tensor RiemannianOperator::geodesic_sgd_step(
    const torch::Tensor& point,
    const torch::Tensor& grad,
    float learning_rate
) {
    // 리만 그래디언트 계산
    auto riemannian_grad = riemannian_gradient(grad, point);
    
    // 지오데식 스텝 (Exponential map 사용)
    auto scaled_grad = -learning_rate * riemannian_grad;
    return geodesic_flow(point, scaled_grad, 1.0f);
}

// 편의 함수들
torch::Tensor hyperbolic_fft_cpu(const torch::Tensor& x, float curvature) {
    HyperbolicFFT fft(curvature, 20); // 더 높은 해상도
    return fft.forward_transform(x);
}
torch::Tensor inverse_hyperbolic_fft_cpu(const torch::Tensor& coeffs, float curvature) {
    int64_t N = coeffs.size(1);
    int64_t max_l = static_cast<int64_t>(std::floor(std::sqrt(static_cast<double>(N)))) - 1;
    HyperbolicFFT fft(curvature, static_cast<int>(max_l));
    return fft.inverse_transform(coeffs);
}

torch::Tensor spherical_harmonics_cpu(const torch::Tensor& theta_phi, int l_max) {
    HyperbolicFFT fft(1.0f, l_max);
    auto result = torch::zeros({theta_phi.size(0), (l_max + 1) * (l_max + 1)}, theta_phi.options());
    
    for (int l = 0; l <= l_max; ++l) {
        for (int m = -l; m <= l; ++m) {
            auto harmonic = fft.compute_spherical_harmonics(theta_phi, l, m);
            int idx = l * l + l + m;
            result.select(1, idx) = harmonic.squeeze(1);
        }
    }
    
    return result;
}

torch::Tensor fast_spherical_conv_cpu(
    const torch::Tensor& f,
    const torch::Tensor& g,
    float curvature
) {
    HyperbolicFFT fft(curvature, 20);
    return fft.fast_spherical_convolution(f, g);
}

torch::Tensor ricci_curvature_cpu(const torch::Tensor& metric_tensor) {
    RiemannianOperator op(1.0f);
    return op.compute_ricci_curvature(metric_tensor);
}

torch::Tensor parallel_transport_cpu(
    const torch::Tensor& v,
    const torch::Tensor& path,
    float curvature
) {
    RiemannianOperator op(curvature);
    return op.parallel_transport(v, path, 1.0f);
}

torch::Tensor geodesic_flow_cpu(
    const torch::Tensor& x,
    const torch::Tensor& v,
    float t,
    float curvature
) {
    RiemannianOperator op(curvature);
    return op.geodesic_flow(x, v, t);
}

torch::Tensor riemannian_gradient_cpu(
    const torch::Tensor& euclidean_grad,
    const torch::Tensor& x,
    float curvature
) {
    RiemannianOperator op(curvature);
    return op.riemannian_gradient(euclidean_grad, x);
}

torch::Tensor geodesic_sgd_step_cpu(
    const torch::Tensor& x,
    const torch::Tensor& grad,
    float lr,
    float curvature
) {
    RiemannianOperator op(curvature);
    return op.geodesic_sgd_step(x, grad, lr);
}

torch::Tensor hyperbolic_wavelet_decomposition_cpu(
    const torch::Tensor& signal,
    int num_levels,
    float curvature
) {
    // 하이퍼볼릭 웨이블릿 변환 (멀티레벨)
    auto coeffs = torch::zeros_like(signal);
    auto current = signal.clone();
    
    HyperbolicFFT fft(curvature, 20);
    
    for (int level = 0; level < num_levels; ++level) {
        auto fft_result = fft.forward_transform(current);
        
        // 주파수 대역 필터링 (웨이블릿 기저)
        auto freq_filter = torch::exp(-torch::arange(fft_result.size(1), 
                                     torch::TensorOptions().dtype(fft_result.dtype()).device(fft_result.device())) * 0.1f * (level + 1));
        auto filtered = fft_result * freq_filter.unsqueeze(0);
        
        auto level_coeffs = fft.inverse_transform(filtered);
        coeffs += level_coeffs;
        
        // 다음 레벨을 위한 다운샘플링
        current = fft.inverse_transform(fft_result * 0.5f);
    }
    
    return coeffs;
}

torch::Tensor frequency_domain_filter_cpu(
    const torch::Tensor& signal,
    const torch::Tensor& filter,
    float curvature
) {
    HyperbolicFFT fft(curvature, 20);
    
    auto signal_fft = fft.forward_transform(signal);
    auto filtered_fft = signal_fft * filter;
    
    return fft.inverse_transform(filtered_fft);
}

} // namespace reality_stone::advanced 