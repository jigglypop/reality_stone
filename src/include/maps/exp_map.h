#pragma once
#ifdef WITH_CUDA
#include <cuda.h>
#endif
#include <torch/extension.h>

namespace reality_stone::maps {
    torch::Tensor exp_map_cpu(torch::Tensor v, float c);
#if defined(WITH_CUDA) 
    torch::Tensor exp_map_forward_cuda(torch::Tensor v, float c);
    torch::Tensor exp_map_backward_cuda(torch::Tensor v, torch::Tensor grad_y, float c);
#endif
}

