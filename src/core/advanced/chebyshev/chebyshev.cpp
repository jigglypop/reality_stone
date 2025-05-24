#include <torch/extension.h>
#include <ATen/ATen.h>
#include <advanced/chebyshev/chebyshev.h>
#include <vector>
#include <cmath>

namespace reality_stone::advanced {

// CUDA 함수 선언
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

// ChebyshevApproximator 클래스 구현
ChebyshevApproximator::ChebyshevApproximator(int order, float base_curvature) 
    : order(order), base_curvature(base_curvature) {
    
    // 체비셰프 점들 미리 계산
    cheb_nodes = generate_nodes(order + 1, torch::kCPU);
    
    // 캐시 초기화
    coeffs_cache = torch::zeros({order + 1}, torch::kFloat32);
}

torch::Tensor ChebyshevApproximator::approximate_tanh(
    const torch::Tensor& x, 
    float curvature
) {
    if (x.is_cuda()) {
        return chebyshev_approximation_cuda(x, order, curvature);
    }
    
    // CPU 구현: tanh(√c * x)를 체비셰프 급수로 근사
    auto x_clamped = torch::clamp(x, -0.999f, 0.999f);
    auto sqrt_c = std::sqrt(curvature);
    auto scaled_x = sqrt_c * x_clamped;
    
    auto result = torch::zeros_like(x);
    
    // 체비셰프 다항식 T_n(x) = cos(n * arccos(x))
    for (int n = 1; n <= order; n += 2) {  // 홀수 항만 (tanh는 홀함수)
        auto T_n = torch::cos(n * torch::acos(x_clamped));
        
        // tanh(x)의 체비셰프 계수 근사
        float coeff = std::pow(-1.0f, (n-1)/2) * 4.0f / (M_PI * (n*n - 0.25f));
        result += coeff * T_n;
    }
    
    return torch::clamp(result, -50.0f, 50.0f);  // 그래디언트 클리핑
}

torch::Tensor ChebyshevApproximator::compute_distance(
    const torch::Tensor& x,
    const torch::Tensor& y,
    float curvature
) {
    if (x.is_cuda() && y.is_cuda()) {
        return chebyshev_distance_cuda(x, y, curvature);
    }
    
    // CPU 구현: 체비셰프 거리 (max norm)
    auto diff = torch::abs(x - y);
    auto max_result = torch::max(diff, /*dim=*/-1);
    auto cheb_dist = std::get<0>(max_result);  // values
    
    // 하이퍼볼릭 공간 변환: d_H = (1/√c) * atanh(√c * d_cheb)
    auto sqrt_c = std::sqrt(curvature);
    auto scaled_dist = torch::clamp(sqrt_c * cheb_dist, 0.0f, 0.99f);
    
    return (1.0f / sqrt_c) * torch::atanh(scaled_dist);
}

torch::Tensor ChebyshevApproximator::generate_nodes(int n, torch::Device device) {
    // 체비셰프 점들: x_k = cos((2k+1)π/(2n))
    auto k = torch::arange(n, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    return torch::cos((2 * k + 1) * M_PI / (2 * n));
}

torch::Tensor ChebyshevApproximator::fast_transform(const torch::Tensor& values) {
    if (values.is_cuda()) {
        return fast_chebyshev_transform_cuda(values);
    }
    
    auto n = values.size(-1);
    if ((n & (n - 1)) != 0) {
        // 2의 거듭제곱이 아니면 일반 변환 사용
        return compute_coefficients(values);
    }
    
    // FFT 기반 고속 변환
    auto extended = torch::cat({values, values.flip(-1).narrow(-1, 1, n-2)}, -1);
    auto fft_result = torch::fft_fft(extended.to(torch::kComplexFloat));
    
    auto coeffs = torch::real(fft_result).narrow(-1, 0, n);
    coeffs.index({0}) *= 0.5f;
    if (n > 1) coeffs.index({-1}) *= 0.5f;
    
    return coeffs / (n - 1);
}

torch::Tensor ChebyshevApproximator::compute_coefficients(const torch::Tensor& func_values) {
    auto n = func_values.size(-1);
    auto device = func_values.device();
    auto coeffs = torch::zeros({order + 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto nodes = generate_nodes(n, device);
    
    for (int k = 0; k <= order; ++k) {
        auto T_k = torch::cos(k * torch::acos(nodes));
        auto coeff_sum = torch::sum(func_values * T_k);
        
        if (k == 0) {
            coeffs.index({k}) = coeff_sum / n;
        } else {
            coeffs.index({k}) = 2.0f * coeff_sum / n;
        }
    }
    
    return coeffs;
}

torch::Tensor ChebyshevApproximator::evaluate_polynomial(
    const torch::Tensor& coeffs,
    const torch::Tensor& x
) {
    auto order = coeffs.size(-1) - 1;
    auto x_clamped = torch::clamp(x, -1.0f + 1e-6f, 1.0f - 1e-6f);
    auto result = torch::zeros_like(x);
    
    for (int k = 0; k <= order; ++k) {
        auto T_k = torch::cos(k * torch::acos(x_clamped));
        result += coeffs.index({k}) * T_k;
    }
    
    return result;
}

// 편의 함수들 구현
torch::Tensor chebyshev_approximation_cpu(
    const torch::Tensor& x,
    int order,
    float curvature
) {
    static ChebyshevApproximator approximator(order, curvature);
    return approximator.approximate_tanh(x, curvature);
}

torch::Tensor chebyshev_distance_cpu(
    const torch::Tensor& x,
    const torch::Tensor& y,
    float curvature
) {
    static ChebyshevApproximator approximator;
    return approximator.compute_distance(x, y, curvature);
}

torch::Tensor chebyshev_nodes_cpu(int n, torch::Device device) {
    auto k = torch::arange(n, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    return torch::cos((2 * k + 1) * M_PI / (2 * n));
}

torch::Tensor chebyshev_coefficients_cpu(
    const torch::Tensor& func_values,
    int order
) {
    auto n = func_values.size(-1);
    auto device = func_values.device();
    auto coeffs = torch::zeros({order + 1}, device);
    auto nodes = chebyshev_nodes_cpu(n, device);
    
    for (int k = 0; k <= order; ++k) {
        auto T_k = torch::cos(k * torch::acos(nodes));
        auto coeff_sum = torch::sum(func_values * T_k);
        
        if (k == 0) {
            coeffs.index({k}) = coeff_sum / n;
        } else {
            coeffs.index({k}) = 2.0f * coeff_sum / n;
        }
    }
    
    return coeffs;
}

torch::Tensor fast_chebyshev_transform_cpu(const torch::Tensor& values) {
    static ChebyshevApproximator approximator;
    return approximator.fast_transform(values);
}

torch::Tensor inverse_chebyshev_transform_cpu(
    const torch::Tensor& coeffs,
    const torch::Tensor& eval_points
) {
    auto order = coeffs.size(-1) - 1;
    auto x = torch::clamp(eval_points, -1.0f + 1e-6f, 1.0f - 1e-6f);
    auto result = torch::zeros_like(x);
    
    for (int k = 0; k <= order; ++k) {
        auto T_k = torch::cos(k * torch::acos(x));
        result += coeffs.index({k}) * T_k;
    }
    
    return result;
}

torch::Tensor chebyshev_derivative_cpu(const torch::Tensor& coeffs) {
    auto n = coeffs.size(-1);
    if (n <= 1) return torch::zeros({1}, coeffs.options());
    
    auto d_coeffs = torch::zeros({n - 1}, coeffs.options());
    
    // 체비셰프 미분 점화식
    for (int k = n - 2; k >= 0; --k) {
        if (k == n - 2) {
            d_coeffs.index({k}) = 2 * (k + 1) * coeffs.index({k + 1});
        } else {
            d_coeffs.index({k}) = d_coeffs.index({k + 2}) + 2 * (k + 1) * coeffs.index({k + 1});
        }
    }
    
    return d_coeffs;
}

torch::Tensor chebyshev_integral_cpu(const torch::Tensor& coeffs, float constant) {
    auto n = coeffs.size(-1);
    auto int_coeffs = torch::zeros({n + 1}, coeffs.options());
    
    int_coeffs.index({0}) = constant;
    
    // 체비셰프 적분 공식
    for (int k = 1; k < n; ++k) {
        if (k == 1) {
            if (k + 1 < n) {
                int_coeffs.index({1}) = coeffs.index({0}) - coeffs.index({2}) / 4.0f;
            } else {
                int_coeffs.index({1}) = coeffs.index({0});
            }
        } else if (k == n - 1) {
            int_coeffs.index({k}) = coeffs.index({k - 1}) / (2 * k);
        } else {
            int_coeffs.index({k}) = (coeffs.index({k - 1}) - coeffs.index({k + 1})) / (2 * k);
        }
    }
    
    if (n > 0) {
        int_coeffs.index({n}) = coeffs.index({n - 1}) / (2 * n);
    }
    
    return int_coeffs;
}

} // namespace reality_stone::advanced 