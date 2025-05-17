#include <torch/extension.h>
#include <utils/numeric.h>
#include <config/constant.h>
#include <layers/klein.h>

namespace utils = reality_stone::utils;
namespace config = reality_stone::config;

namespace reality_stone::layers {

    std::tuple<torch::Tensor, torch::Tensor> klein_backward_cpu(
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
            // Calculate squared norms for Klein model
            double u_norm_sq = 0.0, v_norm_sq = 0.0, uv_dot = 0.0;
            for (int d = 0; d < D; d++) {
                float u_val = u_accessor[b][d];
                float v_val = v_accessor[b][d];
                u_norm_sq += u_val * u_val;
                v_norm_sq += v_val * v_val;
                uv_dot += u_val * v_val;
            }

            // Ensure we're inside the Klein disk
            u_norm_sq = std::min(u_norm_sq, 1.0/c - config::Constants::BOUNDARY_EPS);
            v_norm_sq = std::min(v_norm_sq, 1.0/c - config::Constants::BOUNDARY_EPS);

            // Klein model metric factors
            float cu2 = c * u_norm_sq;
            float cv2 = c * v_norm_sq;
            float cuv = c * uv_dot;

            for (int d = 0; d < D; d++) {
                float u_val = u_accessor[b][d];
                float v_val = v_accessor[b][d];
                float grad_out_val = grad_out_accessor[b][d];

                // Klein geodesic interpolation jacobians
                // The Klein model uses projective geometry
                float denom_u = (1.0f - cu2) * (1.0f - cv2) - cuv * cuv;
                denom_u = std::max(denom_u, config::Constants::MIN_DENOMINATOR);

                // Jacobian with respect to u
                float jacob_u_numerator = (1.0f - cv2) * (1.0f - t) + 
                                         t * c * v_val * (cuv - u_val * (1.0f - cv2));
                float jacob_u = jacob_u_numerator / sqrt(denom_u);

                float jacob_v_numerator = (1.0f - cu2) * t - 
                                         t * c * u_val * (cuv - v_val * (1.0f - cu2));
                float jacob_v = jacob_v_numerator / sqrt(denom_u);

                grad_u_accessor[b][d] = grad_out_val * jacob_u;
                grad_v_accessor[b][d] = grad_out_val * jacob_v;
            }
        }
        return std::make_tuple(grad_u, grad_v);
    }

}