#pragma once
#include <torch/extension.h>
#include <hyper_butterfly/utils/common_defs.h>

namespace hyper_butterfly {
namespace manifolds {
std::vector<torch::Tensor> poincare_forward_cpu(
    torch::Tensor x,
    torch::Tensor params,
    torch::Tensor unused,
    float c,
    int L
);
#ifdef WITH_CUDA
std::vector<torch::Tensor> poincare_forward_cuda(
    torch::Tensor x,
    torch::Tensor params,
    torch::Tensor unused,
    float c,
    int L
);
std::vector<torch::Tensor> poincare_backward_cuda(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor params,
    float c,
    int L
);
#endif
}
}