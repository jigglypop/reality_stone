#pragma once
#include <torch/extension.h>

namespace reality_stone::manifolds {
    torch::Tensor geodesic_cpu(
        torch::Tensor u,
        torch::Tensor v,
        float c,
        float t
    );
#ifdef WITH_CUDA
    torch::Tensor geodesic_cuda(
        torch::Tensor u,
        torch::Tensor v,
        float c,
        float t
    );
#endif
}
