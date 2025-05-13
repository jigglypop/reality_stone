#pragma once
#include <torch/extension.h>

namespace reality_stone::layers {
    torch::Tensor poincare_ball_forward_cpu(
        torch::Tensor u,
        torch::Tensor v,
        float c,
        float t
    );
    std::tuple<torch::Tensor, torch::Tensor> poincare_ball_backward_cpu(
        torch::Tensor grad_output,
        torch::Tensor u,
        torch::Tensor v,
        float c,
        float t
    );
#ifdef WITH_CUDA
    torch::Tensor poincare_ball_forward_cuda(
        torch::Tensor u,
        torch::Tensor v,
        float c,
        float t
    );
    std::tuple<torch::Tensor, torch::Tensor> poincare_ball_backward_cuda(
        torch::Tensor grad_output,
        torch::Tensor u,
        torch::Tensor v,
        float c,
        float t
    );
#endif
}
