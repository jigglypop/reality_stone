#pragma once
#include <torch/extension.h>

#ifdef __CUDACC__
template<typename scalar_t>
__global__ void log_map_forward_kernel(
    const scalar_t* x,
    scalar_t* out,
    float           c, int B, int D);

template<typename scalar_t>
__global__ void log_map_backward_kernel(
    const scalar_t* x,
    const scalar_t* grad_u,
    scalar_t* grad_x,
    float           c, int B, int D);
#endif

namespace hyper_butterfly {
namespace maps {
torch::Tensor log_map_cpu(torch::Tensor x, float c);
torch::Tensor log_map_cuda(torch::Tensor x, float c);
// #ifdef WITH_CUDA
// torch::Tensor log_map_cuda(torch::Tensor x, float c);
// #endif
}
}
