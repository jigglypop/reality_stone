#include <torch/extension.h>
#include <ATen/ATen.h>
#include <advanced/dynamic_curvature/dynamic_curvature.h>
#include <vector>
#include <cmath>

namespace reality_stone::advanced {

// CUDA 함수 선언
torch::Tensor dynamic_curvature_prediction_cuda(
    const torch::Tensor& features,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float c_base
);

torch::Tensor dynamic_mobius_add_cuda(
    const torch::Tensor& u,
    const torch::Tensor& v,
    const torch::Tensor& curvatures
);

torch::Tensor dynamic_poincare_layer_cuda(
    const torch::Tensor& u,
    const torch::Tensor& v,
    const torch::Tensor& curvatures,
    float t
);

DynamicCurvature::DynamicCurvature(int input_dim, float base_curvature) 
    : c_base(base_curvature) {
    // 곡률 예측을 위한 가중치 초기화
    weight_c = torch::randn({1, input_dim}, torch::kFloat32) * 0.1f;
    bias_c = torch::zeros({1}, torch::kFloat32);
}

torch::Tensor DynamicCurvature::extract_features(const torch::Tensor& x) {
    // L2 norm 기반 특징 추출
    // shape: [B, D] -> [B, 1]
    auto norm = torch::norm(x, 2, /*dim=*/-1, /*keepdim=*/true);
    return norm;
}

torch::Tensor DynamicCurvature::normalize_curvature(const torch::Tensor& logits) {
    // 시그모이드를 통한 곡률 정규화 [0, c_base]
    auto sigmoid_output = torch::sigmoid(logits);
    return c_base * sigmoid_output;
}

torch::Tensor DynamicCurvature::predict_curvature(const torch::Tensor& x) {
    // 특징 추출
    auto features = extract_features(x);
    
    if (x.is_cuda()) {
        return dynamic_curvature_prediction_cuda(features, weight_c, bias_c, c_base);
    }
    
    // CPU 구현
    auto logits = torch::mm(features, weight_c.t()) + bias_c;
    return normalize_curvature(logits);
}

torch::Tensor dynamic_mobius_add(
    const torch::Tensor& u,
    const torch::Tensor& v,
    const torch::Tensor& curvatures
) {
    if (u.is_cuda() && v.is_cuda() && curvatures.is_cuda()) {
        return dynamic_mobius_add_cuda(u, v, curvatures);
    }
    
    // CPU 구현
    auto batch_size = u.size(0);
    auto dim = u.size(1);
    auto result = torch::zeros_like(u);
    
    for (int b = 0; b < batch_size; ++b) {
        auto u_b = u[b];
        auto v_b = v[b];
        auto c = curvatures[b].item<float>();
        
        // ||u||², ||v||², <u,v>
        auto u2 = torch::sum(u_b * u_b);
        auto v2 = torch::sum(v_b * v_b);
        auto uv = torch::sum(u_b * v_b);
        
        // Möbius 덧셈 공식
        auto c2 = c * c;
        auto denom = 1.0f + 2.0f * c * uv + c2 * u2 * v2;
        
        auto num_u = (1.0f + 2.0f * c * uv + c * v2) * u_b;
        auto num_v = (1.0f - c * u2) * v_b;
        
        result[b] = (num_u + num_v) / denom;
    }
    
    return result;
}

torch::Tensor dynamic_poincare_layer(
    const torch::Tensor& u,
    const torch::Tensor& v,
    const torch::Tensor& curvatures,
    float t
) {
    if (u.is_cuda() && v.is_cuda() && curvatures.is_cuda()) {
        return dynamic_poincare_layer_cuda(u, v, curvatures, t);
    }
    
    // CPU 측지선 보간: γ(t) = (1-t)u ⊕_c tv
    auto tv = t * v;
    auto one_minus_t_u = (1.0f - t) * u;
    
    return dynamic_mobius_add(one_minus_t_u, tv, curvatures);
}

} // namespace reality_stone::advanced 