#include <torch/extension.h>
#include <ops/mobius.h>
#include <layers/poincare_ball.h>

namespace ops = reality_stone::ops;
namespace reality_stone::layers {
    torch::Tensor poincare_ball_forward_cpu(torch::Tensor u, torch::Tensor v, float c, float t) {
        auto minus_u = ops::mobius_scalar_cpu(u, c, -1.0f);
        auto delta = ops::mobius_add_cpu(minus_u, v, c);
        auto delta_t = ops::mobius_scalar_cpu(delta, c, t);
        return ops::mobius_add_cpu(u, delta_t, c);
    }
}