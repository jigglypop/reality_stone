#pragma once
#include <torch/extension.h>
#include <config/constant.h>

namespace reality_stone::advanced {

class DynamicCurvature {
private:
    torch::Tensor weight_c;
    torch::Tensor bias_c;
    float c_base;
    float min_curvature;
    float max_curvature;
    
public:
    DynamicCurvature(int input_dim, float base_curvature = 1.0f,
                     float min_curvature = 1e-6f, float max_curvature = 1e6f);
    
    torch::Tensor predict_curvature(const torch::Tensor& x);
    torch::Tensor extract_features(const torch::Tensor& x);
    torch::Tensor normalize_curvature(const torch::Tensor& logits);
    
    torch::Tensor& get_weight() { return weight_c; }
    torch::Tensor& get_bias() { return bias_c; }
    float get_base_curvature() const { return c_base; }
    float get_min_curvature() const { return min_curvature; }
    float get_max_curvature() const { return max_curvature; }
    
    void set_curvature_range(float min_c, float max_c) {
        min_curvature = min_c;
        max_curvature = max_c;
    }
};

torch::Tensor predict_dynamic_curvature_cpu(
    const torch::Tensor& features,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float c_base,
    float min_curvature = 1e-6f,
    float max_curvature = 1e6f
);

torch::Tensor dynamic_mobius_add_cpu(
    const torch::Tensor& u,
    const torch::Tensor& v,
    const torch::Tensor& curvatures
);

torch::Tensor dynamic_poincare_layer_cpu(
    const torch::Tensor& u,
    const torch::Tensor& v,
    const torch::Tensor& curvatures,
    float t
);

torch::Tensor dynamic_curvature_prediction_cuda(
    const torch::Tensor& features,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float c_base,
    float min_curvature = 1e-6f,
    float max_curvature = 1e6f
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

}