#include <torch/extension.h>
#include <utils/numeric.h>
#include <config/constant.h>
#include <layers/lorentz.h>

namespace utils = reality_stone::utils;
namespace config = reality_stone::config;

namespace reality_stone::layers {

    std::tuple<torch::Tensor, torch::Tensor> lorentz_backward_cpu(
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
            // Lorentz inner product: ⟨u,v⟩_L = -u₀v₀ + u₁v₁ + ... + u_dv_d
            double lorentz_inner = -u_accessor[b][0] * v_accessor[b][0];
            for (int d = 1; d < D; d++) {
                lorentz_inner += u_accessor[b][d] * v_accessor[b][d];
            }

            // Lorentz norms
            double u_lorentz_norm = -u_accessor[b][0] * u_accessor[b][0];
            double v_lorentz_norm = -v_accessor[b][0] * v_accessor[b][0];
            for (int d = 1; d < D; d++) {
                u_lorentz_norm += u_accessor[b][d] * u_accessor[b][d];
                v_lorentz_norm += v_accessor[b][d] * v_accessor[b][d];
            }
            u_lorentz_norm = std::max(u_lorentz_norm, -1.0/c + config::Constants::EPS);
            v_lorentz_norm = std::max(v_lorentz_norm, -1.0/c + config::Constants::EPS);

            for (int d = 0; d < D; d++) {
                float u_val = u_accessor[b][d];
                float v_val = v_accessor[b][d];
                float grad_out_val = grad_out_accessor[b][d];
                float jacob_u, jacob_v;
                if (d == 0) {
                    jacob_u = (1.0f - t) + t * c * lorentz_inner;
                    jacob_v = t * (1.0f + c * u_lorentz_norm);
                } else {
                    jacob_u = (1.0f - t) - t * c * lorentz_inner;
                    jacob_v = t * (1.0f - c * u_lorentz_norm);
                }

                grad_u_accessor[b][d] = grad_out_val * jacob_u;
                grad_v_accessor[b][d] = grad_out_val * jacob_v;
            }
        }
        return std::make_tuple(grad_u, grad_v);
    }

}