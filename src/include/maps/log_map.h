#pragma once
#ifdef WITH_CUDA
#include <cuda.h>
#endif
#include <torch/extension.h>

namespace reality_stone::maps {
    torch::Tensor log_map_cpu(torch::Tensor x, float c);
#if defined(WITH_CUDA)
    torch::Tensor log_map_forward_cuda(torch::Tensor x, float c);
    torch::Tensor log_map_backward_cuda(torch::Tensor grad_y, torch::Tensor x, float c);
#endif
}
