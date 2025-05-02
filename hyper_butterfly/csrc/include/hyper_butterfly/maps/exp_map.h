#pragma once
#include <torch/extension.h>

#ifdef __CUDACC__  
template<typename scalar_t>
__global__ void exp_map_forward_kernel(
    const scalar_t* v,
    scalar_t* out,
    float c, int B, int D);

template<typename scalar_t>
__global__ void exp_map_backward_kernel(
    const scalar_t* v,
    const scalar_t* grad_y,
    scalar_t* grad_v,
    float           c, int B, int D);
#endif

namespace hyper_butterfly {
namespace maps {
torch::Tensor exp_map_cpu(torch::Tensor v, float c);
torch::Tensor exp_map_cuda(torch::Tensor v, float c);
// #ifdef WITH_CUDA
// torch::Tensor exp_map_cuda(torch::Tensor v, float c);
// #endif
}
}
