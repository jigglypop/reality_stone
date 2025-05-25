#include <torch/extension.h>
#include <ATen/ATen.h>
#include <advanced/regularization/hyperbolic_regularization.h>
#include <config/constant.h>
#include <cmath>

namespace config = reality_stone::config;

namespace reality_stone::advanced {

// HyperbolicRegularizer 클래스 구현
torch::Tensor HyperbolicRegularizer::boundary_penalty(
    const torch::Tensor& x,
    float curvature,
    float epsilon
) {
    // CPU 구현: 경계 근접 페널티
    auto norm = torch::norm(x, 2, /*dim=*/-1);  // [B]
    
    // 포인카레 모델에서 최대 노름: 1 - epsilon
    // Klein 모델에서 최대 노름: 1/√c - epsilon
    auto max_norm = 1.0f / std::sqrt(curvature) - epsilon;
    
    // max(0, ||x|| - max_norm)²
    auto violation = torch::relu(norm - max_norm);
    auto penalty = violation * violation;
    
    return torch::mean(penalty);
}

torch::Tensor HyperbolicRegularizer::curvature_adaptive_penalty(
    const torch::Tensor& x,
    float curvature
) {
    // CPU 구현: 곡률 적응 정규화
    auto norm = torch::norm(x, 2, /*dim=*/-1);  // [B]
    auto sqrt_c = std::sqrt(curvature);
    
    // 안전한 atanh 계산을 위한 클리핑
    auto atanh_arg = torch::clamp(sqrt_c * norm, -0.99f, 0.99f);
    auto log_norm = torch::atanh(atanh_arg) / sqrt_c;
    
    // R_curvature(x) = ||log_0(x)||² · c
    auto penalty = curvature * log_norm * log_norm;
    
    return torch::mean(penalty);
}

torch::Tensor HyperbolicRegularizer::geodesic_variance_penalty(
    const torch::Tensor& weights,
    float curvature
) {
    // CPU 구현: 측지선 분산 정규화
    auto n = weights.size(0);
    auto total_dist_sq = torch::zeros({1}, weights.options());
    
    // 모든 쌍에 대해 하이퍼볼릭 거리 계산
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            auto wi = weights[i];  // [D]
            auto wj = weights[j];  // [D]
            
            // 안전한 하이퍼볼릭 거리 계산
            auto dist_sq = safe_hyperbolic_distance(wi.unsqueeze(0), wj.unsqueeze(0), curvature);
            total_dist_sq += dist_sq * dist_sq;
        }
    }
    
    // 평균 분산
    return total_dist_sq / (n * n);
}

torch::Tensor HyperbolicRegularizer::combined_regularization(
    const torch::Tensor& x,
    const torch::Tensor& weights,
    float curvature,
    float lambda_boundary,
    float lambda_curvature,
    float lambda_geodesic
) {
    // 각 정규화 항목 계산
    auto boundary_loss = boundary_penalty(x, curvature);
    auto curvature_loss = curvature_adaptive_penalty(x, curvature);
    auto geodesic_loss = geodesic_variance_penalty(weights, curvature);
    
    // 가중 합계
    return lambda_boundary * boundary_loss + 
           lambda_curvature * curvature_loss + 
           lambda_geodesic * geodesic_loss;
}

torch::Tensor HyperbolicRegularizer::safe_hyperbolic_distance(
    const torch::Tensor& x,
    const torch::Tensor& y,
    float curvature
) {
    // 포인카레 모델에서의 안전한 하이퍼볼릭 거리
    auto x_norm_sq = torch::sum(x * x, /*dim=*/-1, /*keepdim=*/true);
    auto y_norm_sq = torch::sum(y * y, /*dim=*/-1, /*keepdim=*/true);
    auto xy_dot = torch::sum(x * y, /*dim=*/-1, /*keepdim=*/true);
    
    // 수치적 안정성을 위한 클리핑
    x_norm_sq = torch::clamp(x_norm_sq, 0.0f, 1.0f - config::Constants::BOUNDARY_EPS);
    y_norm_sq = torch::clamp(y_norm_sq, 0.0f, 1.0f - config::Constants::BOUNDARY_EPS);
    
    // 포인카레 거리 공식
    auto numerator = torch::pow(x - y, 2).sum(-1, true);
    auto denominator = (1.0f - curvature * x_norm_sq) * (1.0f - curvature * y_norm_sq);
    denominator = torch::clamp(denominator, config::Constants::EPS, 1e6f);
    
    auto ratio = 1.0f + 2.0f * numerator / denominator;
    ratio = torch::clamp(ratio, 1.0f + config::Constants::EPS, 1e6f);
    
    auto dist = torch::acosh(ratio) / std::sqrt(curvature);
    return torch::clamp(dist, 0.0f, 50.0f);  // 거리 클리핑
}

torch::Tensor HyperbolicRegularizer::safe_log_map(
    const torch::Tensor& x,
    float curvature
) {
    // 안전한 로그 맵 계산
    auto norm = torch::norm(x, 2, /*dim=*/-1, /*keepdim=*/true);
    auto sqrt_c = std::sqrt(curvature);
    
    // atanh 발산 방지
    auto atanh_arg = torch::clamp(sqrt_c * norm, -0.99f, 0.99f);
    auto coeff = torch::atanh(atanh_arg) / (sqrt_c * norm + config::Constants::EPS);
    
    return coeff * x;
}

// 편의 함수들 (CPU 버전)
torch::Tensor boundary_penalty_cpu(
    const torch::Tensor& x,
    float curvature,
    float epsilon
) {
    return HyperbolicRegularizer::boundary_penalty(x, curvature, epsilon);
}

torch::Tensor curvature_adaptive_penalty_cpu(
    const torch::Tensor& x,
    float curvature
) {
    return HyperbolicRegularizer::curvature_adaptive_penalty(x, curvature);
}

torch::Tensor geodesic_variance_penalty_cpu(
    const torch::Tensor& weights,
    float curvature
) {
    return HyperbolicRegularizer::geodesic_variance_penalty(weights, curvature);
}

torch::Tensor combined_regularization_cpu(
    const torch::Tensor& x,
    const torch::Tensor& weights,
    float curvature,
    float lambda_boundary,
    float lambda_curvature,
    float lambda_geodesic
) {
    return HyperbolicRegularizer::combined_regularization(
        x, weights, curvature, 
        lambda_boundary, lambda_curvature, lambda_geodesic
    );
}

} // namespace reality_stone::advanced