#include <torch/extension.h>
#include <ATen/ATen.h>
#include <advanced/geodesic_activation/geodesic_activation.h>
#include <vector>
#include <cmath>

namespace reality_stone::advanced {

// CUDA 함수 선언
torch::Tensor geodesic_activation_cuda(
    const torch::Tensor& input,
    const torch::Tensor& anchors,
    const torch::Tensor& t_values,
    const torch::Tensor& weights,
    float curvature
);

torch::Tensor einstein_midpoint_cuda(
    const torch::Tensor& points,
    const torch::Tensor& weights,
    float curvature
);

torch::Tensor multi_geodesic_mixing_cuda(
    const torch::Tensor& input,
    const torch::Tensor& anchors,
    const torch::Tensor& t_values,
    const torch::Tensor& weights,
    float curvature
);

GeodesicActivation::GeodesicActivation(int input_dim, int num_anchors) 
    : input_dim(input_dim), num_anchors(num_anchors) {
    
    // 앵커 포인트 초기화 (포인카레 디스크 내부에 랜덤 배치)
    anchors = torch::randn({num_anchors, input_dim}, torch::kFloat32) * 0.3f;
    
    // 측지선 파라미터 초기화 (0.5 주변에서 시작)
    t_params = torch::full({num_anchors}, 0.5f, torch::kFloat32);
    
    // 앵커별 가중치 초기화
    weights = torch::ones({num_anchors}, torch::kFloat32) / num_anchors;
}

torch::Tensor GeodesicActivation::forward(const torch::Tensor& x, float curvature) {
    if (x.is_cuda()) {
        return geodesic_activation_cuda(x, anchors, t_params, weights, curvature);
    }
    
    // CPU 구현
    return multi_geodesic_mixing(x, curvature);
}

torch::Tensor GeodesicActivation::einstein_midpoint(
    const torch::Tensor& points,
    const torch::Tensor& weights,
    float curvature
) {
    if (points.is_cuda()) {
        return einstein_midpoint_cuda(points, weights, curvature);
    }
    
    // CPU 구현: M_E = exp_0(∑ᵢ wᵢ·log_0(xᵢ))
    auto batch_size = points.size(0);
    auto num_points = points.size(1);
    auto dim = points.size(2);
    
    auto result = torch::zeros({batch_size, dim}, points.options());
    
    for (int b = 0; b < batch_size; ++b) {
        auto weighted_log_sum = torch::zeros({dim}, points.options());
        
        for (int k = 0; k < num_points; ++k) {
            auto point = points[b][k];
            auto weight = weights[k];
            
            // 로그 맵 계산: log_0(x)
            auto norm = torch::norm(point, 2);
            auto sqrt_c = std::sqrt(curvature);
            
            // 안전한 atanh 계산
            auto atanh_arg = torch::clamp(sqrt_c * norm, -0.99f, 0.99f);
            auto coeff = torch::atanh(atanh_arg) / (sqrt_c * norm + 1e-7f);
            
            auto log_point = coeff * point;
            weighted_log_sum += weight * log_point;
        }
        
        // 지수 맵 계산: exp_0(weighted_log_sum)
        auto log_norm = torch::norm(weighted_log_sum, 2);
        auto sqrt_c = std::sqrt(curvature);
        
        auto tanh_arg = torch::clamp(sqrt_c * log_norm, -88.0f, 88.0f);
        auto exp_coeff = torch::tanh(tanh_arg) / (sqrt_c * log_norm + 1e-7f);
        
        result[b] = exp_coeff * weighted_log_sum;
    }
    
    return result;
}

torch::Tensor GeodesicActivation::multi_geodesic_mixing(
    const torch::Tensor& input,
    float curvature
) {
    if (input.is_cuda()) {
        return multi_geodesic_mixing_cuda(input, anchors, t_params, weights, curvature);
    }
    
    // CPU 구현
    auto batch_size = input.size(0);
    auto dim = input.size(1);
    auto result = torch::zeros_like(input);
    
    for (int b = 0; b < batch_size; ++b) {
        auto input_point = input[b];
        auto mixed_result = torch::zeros({dim}, input.options());
        float total_weight = 0.0f;
        
        for (int k = 0; k < num_anchors; ++k) {
            auto anchor = anchors[k];
            auto t = torch::sigmoid(t_params[k]).item<float>(); // [0,1] 범위로 정규화
            auto w = torch::softmax(weights, 0)[k].item<float>(); // 가중치 정규화
            
            // 측지선 보간: γ(t) = (1-t)·0 ⊕_c t·input
            // 원점에서 input까지의 측지선에서 t 지점
            auto scaled_input = t * input_point;
            
            // 간단한 Möbius 덧셈 (원점과 scaled_input)
            auto norm_sq = torch::sum(scaled_input * scaled_input);
            auto denom = 1.0f + curvature * norm_sq;
            auto geodesic_point = scaled_input / denom;
            
            // 앵커와의 거리 기반 가중치
            auto dist = torch::norm(input_point - anchor, 2);
            auto distance_weight = torch::exp(-dist);
            
            mixed_result += (w * distance_weight) * geodesic_point;
            total_weight += (w * distance_weight).item<float>();
        }
        
        result[b] = mixed_result / (total_weight + 1e-7f);
    }
    
    return result;
}

} // namespace reality_stone::advanced 