#pragma once
#include <torch/extension.h>
#include <utils/common_defs.h>

namespace reality_stone::manifolds {
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
