#pragma once
#include <torch/extension.h>

namespace hyper_butterfly {
namespace ops {

/**
 * Möbius subtraction: u ⊖₍c₎ v = u ⊕₍c₎ (−1 ⊗₍c₎ v)
 */
torch::Tensor mobius_sub_cpu(
    torch::Tensor u,
    torch::Tensor v,
    float c
);
/**
 * Geodesic interpolation in the Poincaré ball:
 * γ(t) = u ⊕₍c₎ ( (⊖₍c₎u ⊕₍c₎ v) ⊗₍c₎ t )
 *  - u, v : (B,D) tensors
 *  - c     : curvature > 0
 *  - t     : in [0,1], the interpolation factor (scalar)
 */

torch::Tensor geodesic_cpu(
    torch::Tensor u,
    torch::Tensor v,
    float c,
    float t
);
#ifdef WITH_CUDA
torch::Tensor mobius_sub_cuda(
    torch::Tensor u,
    torch::Tensor v,
    float c
);
torch::Tensor geodesic_cuda(
    torch::Tensor u,
    torch::Tensor v,
    float c,
    float t
);
#endif
} // namespace ops
} // namespace hyper_butterfly
