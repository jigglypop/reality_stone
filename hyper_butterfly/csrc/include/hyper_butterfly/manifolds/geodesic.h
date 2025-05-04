#pragma once
#include <torch/extension.h>

namespace hyper_butterfly {
namespace manifolds {

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
}
