#include <torch/extension.h>
#include <ops/mobius.h>
#include <utils/numeric.h>
#include <config/constant.h>
#include <layers/poincare_ball.h>

namespace ops = reality_stone::ops;
namespace utils = reality_stone::utils;
namespace config = reality_stone::config;

namespace reality_stone::layers {

    std::tuple<torch::Tensor, torch::Tensor> poincare_ball_backward_cpu(
        torch::Tensor grad_output,
        torch::Tensor u,
        torch::Tensor v,
        float c,
        float t
    ) {
        int B = u.size(0), D = u.size(1);
        auto grad_u = torch::zeros_like(u);
        auto grad_v = torch::zeros_like(v);

        auto grad_out_accessor = grad_output.accessor<float, 2>();
        auto u_accessor = u.accessor<float, 2>();
        auto v_accessor = v.accessor<float, 2>();
        auto grad_u_accessor = grad_u.accessor<float, 2>();
        auto grad_v_accessor = grad_v.accessor<float, 2>();

        for (int b = 0; b < B; b++) {
            double u_norm_sq = 0.0, v_norm_sq = 0.0, uv_dot = 0.0;
            for (int d = 0; d < D; d++) {
                float u_val = u_accessor[b][d];
                float v_val = v_accessor[b][d];
                u_norm_sq += u_val * u_val;
                v_norm_sq += v_val * v_val;
                uv_dot += u_val * v_val;
            }

            float u_norm = std::sqrt(u_norm_sq);
            float v_norm = std::sqrt(v_norm_sq);
            for (int d = 0; d < D; d++) {
                float u_val = u_accessor[b][d];
                float v_val = v_accessor[b][d];
                float grad_out_val = grad_out_accessor[b][d];
                float denom = 1.0f + 2.0f * c * uv_dot + c * c * u_norm_sq * v_norm_sq;
                denom = std::max(denom, config::Constants::MIN_DENOMINATOR);
                float jacob_u = (1.0f + c * v_norm_sq - c * u_norm_sq) / denom;
                float jacob_v = ((1.0f - c * u_norm_sq) / denom) * t;
                grad_u_accessor[b][d] = grad_out_val * jacob_u;
                grad_v_accessor[b][d] = grad_out_val * jacob_v;
            }
        }
        return std::make_tuple(grad_u, grad_v);
    }
}