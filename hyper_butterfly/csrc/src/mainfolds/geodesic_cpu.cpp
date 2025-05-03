#include <torch/extension.h>
#include <hyper_butterfly/ops/mobius.h>
#include <hyper_butterfly/manifolds/geodesic.h>

namespace ops = hyper_butterfly::ops;

namespace hyper_butterfly {
namespace manifolds {

torch::Tensor mobius_sub_cpu(torch::Tensor u, torch::Tensor v, float c) {
    // (−1)⊗₍c₎ v
    auto nv = ops::mobius_scalar_cpu(v, c, -1.0f);
    return ops::mobius_add_cpu(u, nv, c);
}

torch::Tensor geodesic_cpu(torch::Tensor u, torch::Tensor v, float c, float t) {
    // δ = (⊖₍c₎u) ⊕₍c₎ v
    auto minus_u = ops::mobius_scalar_cpu(u, c, -1.0f);
    auto delta = ops::mobius_add_cpu(minus_u, v, c);
    // δ_t = t ⊗₍c₎ δ
    auto delta_t = ops::mobius_scalar_cpu(delta, c, t);
    // γ(t) = u ⊕₍c₎ δ_t
    return ops::mobius_add_cpu(u, delta_t, c);
}

} // namespace ops
} // namespace hyper_butterfly
