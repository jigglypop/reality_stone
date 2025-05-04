#include <torch/extension.h>
#include <hyper_butterfly/ops/mobius.h>
#include <hyper_butterfly/manifolds/geodesic.h>

namespace ops = hyper_butterfly::ops;

namespace hyper_butterfly {
namespace manifolds {



torch::Tensor geodesic_cpu(torch::Tensor u, torch::Tensor v, float c, float t) {
    auto minus_u = ops::mobius_scalar_cpu(u, c, -1.0f);
    auto delta = ops::mobius_add_cpu(minus_u, v, c);
    auto delta_t = ops::mobius_scalar_cpu(delta, c, t);
    return ops::mobius_add_cpu(u, delta_t, c);
}

}
}
