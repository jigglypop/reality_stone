#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <hyper_butterfly/utils/common_defs.h>
#include "butterfly.h"

// ─────────────────────────────────────────────────────────────────────────────
// 버터플라이 레이어 (forward)
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t>
__global__ void butterfly_layer_kernel(
    const scalar_t* __restrict__ input,
    scalar_t*       __restrict__ output,
    const scalar_t* __restrict__ params,
    int B, int D, int layer_idx) {

  int idx    = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x*gridDim.x;
  int bs     = 1 << layer_idx;
  int nb     = D / (2*bs);

  while(idx < B*D) {
    int b = idx / D, f = idx % D;
    int blk = (f/(2*bs)) % nb,
        loc = f % (2*bs),
        off = loc % bs;
    bool high = loc >= bs;
    int pi = blk*2;
    float a  = params[pi+0],
          bb = params[pi+1];
    int base = b*D + blk*2*bs;
    float x1 = input[base + off],
          x2 = input[base + off + bs];
    output[idx] = high
      ? (-bb*x1 + a*x2)
      : ( a*x1 + bb*x2 );
    idx += stride;
  }
}

// CUDA 버터플라이 레이어 forward 구현
torch::Tensor butterfly_layer_cuda(
    torch::Tensor input,
    torch::Tensor params,
    int layer_idx,
    int batch_size,
    int dim) {
  
  CHECK_CUDA_CONTIGUOUS(input);
  CHECK_CUDA_CONTIGUOUS(params);

  auto output = torch::empty_like(input);
  int threads = 256;
  int blocks = (batch_size * dim + threads - 1) / threads;
  
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_layer_cuda", [&]{
    butterfly_layer_kernel<scalar_t><<<blocks, threads>>>(
      input.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      params.data_ptr<scalar_t>(),
      batch_size, dim, layer_idx);
  });
  
  CUDA_CHECK(cudaGetLastError());
  return output;
} 