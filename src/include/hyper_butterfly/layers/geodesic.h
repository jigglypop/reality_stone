#pragma once
#include <torch/extension.h>

namespace hyper_butterfly::layers {
    torch::Tensor geodesic_forward_cpu(
        torch::Tensor u,
        torch::Tensor v,
        float c,
        float t
    );
    std::tuple<torch::Tensor, torch::Tensor> geodesic_backward_cpu(
        torch::Tensor grad_output,
        torch::Tensor u,
        torch::Tensor v,
        float c,
        float t
    );
#ifdef WITH_CUDA
    torch::Tensor geodesic_forward_cuda(
        torch::Tensor u,
        torch::Tensor v,
        float c,
        float t
    );
    std::tuple<torch::Tensor, torch::Tensor> geodesic_backward_cuda(
        torch::Tensor grad_output,
        torch::Tensor u,
        torch::Tensor v,
        float c,
        float t
    );
#endif
}
