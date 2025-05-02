#pragma once
#include <cuda_runtime.h>

namespace hyper_butterfly {
namespace maps {
#ifdef __CUDACC__  
template <typename scalar_t>
__global__ void exp_map_forward_kernel(
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    float c, int B, int D);

template <typename scalar_t>
__global__ void exp_map_backward_kernel(
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ grad_y,
    scalar_t* __restrict__ grad_v,
    float c, int B, int D);
#endif 
}
}
