#include <torch/extension.h>
#include <cmath>
#include <ops/mobius.h>
#include <config/constant.h>

namespace config = reality_stone::config;

namespace reality_stone::ops {
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
        auto denom = (1 + 2 * c * uv + c2 * u2 * v2).clamp_min(config::Constants::MIN_DENOMINATOR);
        return (num_u + num_v) / denom;
    }

    torch::Tensor mobius_scalar_cpu(
        torch::Tensor u,
        float c,
        float r
    ) {
        // norms: (B,1)
        auto norm = torch::norm(u, 2, /*dim=*/1, /*keepdim=*/true).clamp_min(config::Constants::EPS);
        float sqrtc = std::sqrt(c);
        auto scn = (sqrtc * norm).clamp_min(config::Constants::EPS)
            .clamp_max(1.0f - config::Constants::BOUNDARY_EPS);
        auto alpha = torch::atanh(scn);
        auto beta = torch::tanh(r * alpha);
        return (beta / (sqrtc * norm)) * u;
    }

}