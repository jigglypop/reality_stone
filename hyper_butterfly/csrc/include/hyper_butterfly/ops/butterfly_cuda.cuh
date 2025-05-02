// hyper_butterfly/ops/butterfly_cuda.cuh
#pragma once

#include <cuda_runtime.h>

namespace hyper_butterfly {
namespace ops {

#ifdef __CUDACC__

// forward kernel 선언
template <typename scalar_t>
__global__ void butterfly_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ params,
    int B, int D, int layer_idx);

// backward kernel 선언
template <typename scalar_t>
__global__ void butterfly_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ params,
    scalar_t* __restrict__ grad_input,
    scalar_t* __restrict__ grad_params,
    int B, int D, int layer_idx);

#endif  // __CUDACC__

}  // namespace ops
}  // namespace hyper_butterfly
