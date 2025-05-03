#include <torch/extension.h>
#include <cmath>
#include <hyper_butterfly/ops/mobius.h>

namespace hyper_butterfly {
namespace ops {

torch::Tensor mobius_add_cpu(
    torch::Tensor u,
    torch::Tensor v,
    float c
) {
    auto u2 = torch::sum(u * u, /*dim=*/1, /*keepdim=*/true);
    auto v2 = torch::sum(v * v, /*dim=*/1, /*keepdim=*/true);
    auto uv = torch::sum(u * v, /*dim=*/1, /*keepdim=*/true);
    float c2 = c * c;

    auto num_u = (1 + 2 * c * uv + c * v2) * u;
    auto num_v = (1 - c * u2) * v;
    auto denom = (1 + 2 * c * uv + c2 * u2 * v2).clamp_min(1e-6f);

    return (num_u + num_v) / denom;
}

torch::Tensor mobius_scalar_cpu(
    torch::Tensor u,
    float c,
    float r
) {
    // norms: (B,1)
    auto norm = torch::norm(u, 2, /*dim=*/1, /*keepdim=*/true).clamp_min(1e-6f);
    float sqrtc = std::sqrt(c);
    // α = arctanh(√c ||u||)
    auto alpha = torch::atanh(sqrtc * norm);
    // β = tanh(r * α)
    auto beta = torch::tanh(r * alpha);
    // result = (1/√c) * β * (u / ||u||)
    return (beta / (sqrtc * norm)) * u;
}

} // namespace ops
} // namespace hyper_butterfly
