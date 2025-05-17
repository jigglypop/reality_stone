#include <torch/extension.h>
#include <ops/lorentz.h>
#include <layers/lorentz.h>

namespace ops = reality_stone::ops;

namespace reality_stone::layers {
    torch::Tensor lorentz_forward_cpu(torch::Tensor u, torch::Tensor v, float c, float t) {
        auto inner = ops::lorentz_inner_cpu(u, v);
        auto dist = ops::lorentz_distance_cpu(u, v, c);
        auto v_perp = v + inner * u;
        auto v_perp_norm = torch::sqrt(-ops::lorentz_inner_cpu(v_perp, v_perp))
            .clamp_min(1e-8);
        v_perp = v_perp / v_perp_norm;
        return torch::cosh(dist * t) * u + torch::sinh(dist * t) * v_perp;
    }
}