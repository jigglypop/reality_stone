#pragma once
#include <cuda_runtime.h>

namespace hyper_butterfly {
namespace maps {
#if defined(__CUDACC__)
template<typename scalar_t>
__global__ void log_map_forward_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    float c, int B, int D);
template<typename scalar_t>
__global__ void log_map_backward_kernel(
    const scalar_t* __restrict__  x,
    const scalar_t* __restrict__  grad_u,
    scalar_t* __restrict__  grad_x,
    float c, int B, int D);
#endif
}
}
