#include <torch/extension.h>
#include <cmath>
#include <ops/lorentz.h>
#include <config/constant.h>
#include <utils/numeric.h>

namespace config = reality_stone::config;
namespace utils = reality_stone::utils;

namespace reality_stone::ops {
    torch::Tensor lorentz_inner_cpu(
        torch::Tensor u,
        torch::Tensor v
    ) {
        auto time_part = u.select(1, 0) * v.select(1, 0);
        auto space_part = (u.narrow(1, 1, u.size(1) - 1) * v.narrow(1, 1, v.size(1) - 1)).sum(1);
        return (time_part - space_part).unsqueeze(1);
    }

    torch::Tensor lorentz_distance_cpu(
        torch::Tensor u,
        torch::Tensor v,
        float c
    ) {
        auto inner = -lorentz_inner_cpu(u, v);
        inner = inner.clamp_min(1.0f + config::Constants::EPS);
        return torch::acosh(inner) / std::sqrt(c);
    }

    torch::Tensor lorentz_add_cpu(
        torch::Tensor u,
        torch::Tensor v,
        float c
    ) {
        float sqrtc = std::sqrt(c);
        auto uv_inner = lorentz_inner_cpu(u, v);
        auto v_perp = v + uv_inner * u;
        auto norm_v_perp = torch::sqrt(-lorentz_inner_cpu(v_perp, v_perp));
        norm_v_perp = norm_v_perp.clamp_min(config::Constants::EPS);
        auto cos_theta = torch::cosh(norm_v_perp);
        auto sin_theta = torch::sinh(norm_v_perp);
        return cos_theta * u + sin_theta * v_perp / norm_v_perp;
    }

    torch::Tensor lorentz_scalar_cpu(
        torch::Tensor u,
        float c,
        float r
    ) {
        auto time_comp = u.select(1, 0);
        auto space_comp = u.narrow(1, 1, u.size(1) - 1);
        auto norm = torch::sqrt(
            torch::pow(space_comp, 2).sum(1) /
            (torch::pow(time_comp, 2) - config::Constants::EPS)
        ).unsqueeze(1);
        auto theta = torch::atanh(norm.clamp_max(1.0f - config::Constants::BOUNDARY_EPS)) * r;
        auto scale = torch::tanh(theta) / norm.clamp_min(config::Constants::EPS);
        auto scaled_space = space_comp * scale;
        auto scaled_time = torch::sqrt(1.0f + torch::pow(scaled_space, 2).sum(1)).unsqueeze(1);
        return torch::cat({ scaled_time, scaled_space }, 1);
    }

    torch::Tensor poincare_to_lorentz_cpu(
        torch::Tensor x,
        float c
    ) {
        float sqrtc = std::sqrt(c);
        auto x_norm_sq = torch::sum(x * x, /*dim=*/1, /*keepdim=*/true);
        auto denom = (1.0f - c * x_norm_sq).clamp_min(config::Constants::EPS);
        auto x0 = (1.0f + c * x_norm_sq) / denom;
        auto xi = (2.0f * x) / denom;
        return torch::cat({ x0, xi }, /*dim=*/1) / sqrtc;
    }

    torch::Tensor lorentz_to_poincare_cpu(
        torch::Tensor x,
        float c
    ) {

        float sqrtc = std::sqrt(c);
        auto x_scaled = x * sqrtc;
        auto x0 = x_scaled.select(1, 0);
        auto xi = x_scaled.narrow(1, 1, x_scaled.size(1) - 1);
        auto denom = (x0 + 1.0f).unsqueeze(1).clamp_min(config::Constants::EPS);
        return xi / denom;
    }
}