#include <torch/extension.h>
#include <hyper_butterfly/ops/mobius.h>
#include <hyper_butterfly/manifold/geodesic.h>

namespace hyper_butterfly {
namespace ops {

torch::Tensor mobius_sub_cpu(torch::Tensor u, torch::Tensor v, float c) {
    // (−1)⊗₍c₎ v
    auto nv = mobius_scalar_cpu(v, c, -1.0f);
    return mobius_add_cpu(u, nv, c);
}

torch::Tensor geodesic_cpu(torch::Tensor u, torch::Tensor v, float c, float t) {
    // δ = (⊖₍c₎u) ⊕₍c₎ v
    auto minus_u = mobius_scalar_cpu(u, c, -1.0f);
    auto delta = mobius_add_cpu(minus_u, v, c);
    // δ_t = t ⊗₍c₎ δ
    auto delta_t = mobius_scalar_cpu(delta, c, t);
    // γ(t) = u ⊕₍c₎ δ_t
    return mobius_add_cpu(u, delta_t, c);
}

} // namespace ops
} // namespace hyper_butterfly
