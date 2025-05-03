#include <torch/extension.h>
#include <hyper_butterfly/ops/mobius.h>
#include <hyper_butterfly/mainfolds/geodesic.h>
#include <hyper_butterfly/utils/cuda_utils.h>

namespace hyper_butterfly {
namespace ops {

torch::Tensor mobius_sub_cuda(torch::Tensor u, torch::Tensor v, float c) {
    auto nv = mobius_scalar_cuda(v, c, -1.0f);
    return mobius_add_cuda(u, nv, c);
}

torch::Tensor geodesic_cuda(torch::Tensor u, torch::Tensor v, float c, float t) {
    auto minus_u = mobius_scalar_cuda(u, c, -1.0f);
    auto delta = mobius_add_cuda(minus_u, v, c);
    auto delta_t = mobius_scalar_cuda(delta, c, t);
    return mobius_add_cuda(u, delta_t, c);
}

} // namespace ops
} // namespace hyper_butterfly
