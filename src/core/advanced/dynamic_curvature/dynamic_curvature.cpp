#include <torch/extension.h>
#include <ATen/ATen.h>
#include <advanced/dynamic_curvature/dynamic_curvature.h>
#include <ops/mobius.h>
#include <vector>
#include <cmath>

namespace ops = reality_stone::ops;

namespace reality_stone::advanced {

DynamicCurvature::DynamicCurvature(int input_dim, float base_curvature, 
                                   float min_curvature, float max_curvature) 
    : c_base(base_curvature), min_curvature(min_curvature), max_curvature(max_curvature) {
    weight_c = torch::randn({1, input_dim}, torch::kFloat32) * 0.1f;
    bias_c = torch::zeros({1}, torch::kFloat32);
}

torch::Tensor DynamicCurvature::extract_features(const torch::Tensor& x) {
    auto norm = torch::norm(x, 2, -1, true);
    return norm;
}

torch::Tensor DynamicCurvature::normalize_curvature(const torch::Tensor& logits) {
    auto clamped_logits = torch::clamp(logits, -20.0f, 20.0f);
    auto exp_output = torch::exp(clamped_logits);
    auto curvatures = c_base * exp_output;
    return torch::clamp(curvatures, min_curvature, max_curvature);
}

torch::Tensor DynamicCurvature::predict_curvature(const torch::Tensor& x) {
    auto features = extract_features(x);
    auto logits = torch::mm(features, weight_c.t()) + bias_c;
    return normalize_curvature(logits);
}

torch::Tensor predict_dynamic_curvature_cpu(
    const torch::Tensor& features,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float c_base,
    float min_curvature,
    float max_curvature
) {
    auto logits = torch::mm(features, weight.t()) + bias;
    auto clamped_logits = torch::clamp(logits, -20.0f, 20.0f);
    auto exp_output = torch::exp(clamped_logits);
    auto curvatures = c_base * exp_output;
    return torch::clamp(curvatures, min_curvature, max_curvature).squeeze();
}

torch::Tensor dynamic_mobius_add_cpu(
    const torch::Tensor& u,
    const torch::Tensor& v,
    const torch::Tensor& curvatures
) {
    auto batch_size = u.size(0);
    auto result = torch::zeros_like(u);
    
    for (int64_t b = 0; b < batch_size; ++b) {
        auto u_b = u[b].unsqueeze(0);
        auto v_b = v[b].unsqueeze(0);
        float c_b = std::max(1e-6f, std::min(curvatures[b].item<float>(), 1e6f));
        
        auto result_b = ops::mobius_add_cpu(u_b, v_b, c_b);
        result[b] = result_b.squeeze(0);
    }
    
    return result;
}

torch::Tensor dynamic_poincare_layer_cpu(
    const torch::Tensor& u,
    const torch::Tensor& v,
    const torch::Tensor& curvatures,
    float t
) {
    auto tv = t * v;
    auto one_minus_t_u = (1.0f - t) * u;
    return dynamic_mobius_add_cpu(one_minus_t_u, tv, curvatures);
}

}