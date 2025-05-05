#include <torch/extension.h>
#include <hyper_butterfly/ops/mobius.h>
#include <hyper_butterfly/utils/numeric.h>
#include <hyper_butterfly/config/constant.h>

namespace ops = hyper_butterfly::ops;
namespace utils = hyper_butterfly::utils;
namespace config = hyper_butterfly::config;

namespace hyper_butterfly {
namespace layers {

std::tuple<torch::Tensor, torch::Tensor> geodesic_backward_cpu(
    torch::Tensor grad_output,
    torch::Tensor u,
    torch::Tensor v,
    float c,
    float t
) {
    int B = u.size(0), D = u.size(1);
    auto grad_u = torch::zeros_like(u);
    auto grad_v = torch::zeros_like(v);

    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "geodesic_backward_cpu", [&] {
        const scalar_t* grad_out_ptr = grad_output.data_ptr<scalar_t>();
        const scalar_t* u_ptr = u.data_ptr<scalar_t>();
        const scalar_t* v_ptr = v.data_ptr<scalar_t>();
        scalar_t* grad_u_ptr = grad_u.data_ptr<scalar_t>();
        scalar_t* grad_v_ptr = grad_v.data_ptr<scalar_t>();

        for (int b = 0; b < B; b++) {
            // 각 배치별 포인터
            const scalar_t* u_b = u_ptr + b * D;
            const scalar_t* v_b = v_ptr + b * D;
            scalar_t* grad_u_b = grad_u_ptr + b * D;
            scalar_t* grad_v_b = grad_v_ptr + b * D;
            const scalar_t* grad_out_b = grad_out_ptr + b * D;

            // 중간 계산: minus_u = -u
            auto minus_u_row = u.select(0, b) * (-1.0f);
            auto minus_u = minus_u_row.unsqueeze(0);

            // delta = (-u) ⊕ v
            auto delta_row = ops::mobius_add_cpu(minus_u, v.select(0, b).unsqueeze(0), c);
            auto delta = delta_row.squeeze(0);

            // delta_t = t * delta
            auto delta_t_row = ops::mobius_scalar_cpu(delta.unsqueeze(0), c, t);
            auto delta_t = delta_t_row.squeeze(0);

            // 야코비안 계산을 위한 노름
            float u_norm = 0.0f, v_norm = 0.0f, delta_norm = 0.0f;
            float u_dot_delta = 0.0f, u_dot_v = 0.0f;

            for (int d = 0; d < D; d++) {
                float u_val = u_b[d];
                float v_val = v_b[d];
                float delta_val = delta[d].item<float>();

                u_norm += u_val * u_val;
                v_norm += v_val * v_val;
                delta_norm += delta_val * delta_val;
                u_dot_delta += u_val * delta_val;
                u_dot_v += u_val * v_val;
            }

            u_norm = std::sqrt(std::max(u_norm, config::Constants::EPS));
            v_norm = std::sqrt(std::max(v_norm, config::Constants::EPS));
            delta_norm = std::sqrt(std::max(delta_norm, config::Constants::EPS));

            // 모비우스 연산의 야코비안 계산
            float c2 = c * c;
            float u2 = u_norm * u_norm;
            float v2 = v_norm * v_norm;
            float delta2 = delta_norm * delta_norm;

            // d(u ⊕ delta_t) / d(u)
            float denom1 = 1 + 2 * c * u_dot_delta + c2 * u2 * delta2;
            denom1 = std::max(denom1, config::Constants::MIN_DENOMINATOR);

            for (int d = 0; d < D; d++) {
                float u_val = u_b[d];
                float delta_t_val = delta_t[d].item<float>();
                float grad_out_val = grad_out_b[d];

                // 모비우스 덧셈 야코비안: ∂(u ⊕ v)/∂u
                float jacob_u = (1 + 2 * c * u_dot_delta + c * delta2) / denom1;
                jacob_u -= 4 * c2 * u_val * v[d].item<float>() * u_dot_delta / (denom1 * denom1);

                grad_u_b[d] = grad_out_val * jacob_u;

                // 체인룰로 v에 대한 그래디언트도 계산
                float jacob_v = (1 - c * u2) / denom1;
                jacob_v -= 2 * c2 * u_norm * v[d].item<float>() * u_dot_v / (denom1 * denom1);
                jacob_v *= t; // 모비우스 스칼라 곱의 t 파라미터

                grad_v_b[d] = grad_out_val * jacob_v;
            }
        }
        });

    return std::make_tuple(grad_u, grad_v);
}

}
}