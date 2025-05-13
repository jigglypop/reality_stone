#pragma once
#include <torch/extension.h>

namespace reality_stone::ops {
    torch::Tensor mobius_add_cpu(
        torch::Tensor u,
        torch::Tensor v,
        float c
    );
    torch::Tensor mobius_scalar_cpu(
        torch::Tensor u,
        float c,
        float r
    );
#ifdef WITH_CUDA
// u ⊕_c v
    torch::Tensor mobius_add_cuda(
        torch::Tensor u,
        torch::Tensor v,
        float c
    );
    // r ⊗_c u
    torch::Tensor mobius_scalar_cuda(
        torch::Tensor u,
        float c,
        float r
    );
#endif
}
