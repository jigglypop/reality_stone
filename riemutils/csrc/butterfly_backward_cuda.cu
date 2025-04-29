#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common_defs.h"
#include "butterfly.h"

// 버터플라이 레이어 역전파 커널
template <typename scalar_t>
__global__ void butterfly_layer_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ params,
    scalar_t*       __restrict__ grad_input,
    scalar_t*       __restrict__ grad_params,
    int B, int D, int layer_idx)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x*gridDim.x;
  int bs = 1<<layer_idx, nb = D/(2*bs);

  while(idx < B*D) {
    int b = idx/D, f = idx%D;
    int blk = (f/(2*bs))%nb, loc = f%(2*bs), off = loc%bs;
    bool high = loc>=bs;
    int pi = blk*2;
    float a = params[pi+0], bb = params[pi+1];
    int base = b*D + blk*2*bs;
    float x1 = input[base+off], x2 = input[base+off+bs];
    float gout = grad_out[idx];

    if(!high) {
      // y = a*x1 + b*x2
      atomicAdd(&grad_input[base+off  ],  a*gout);
      atomicAdd(&grad_input[base+off+bs],  bb*gout);
      atomicAdd(&grad_params[pi+0], x1*gout);
      atomicAdd(&grad_params[pi+1], x2*gout);
    } else {
      // y = -b*x1 + a*x2
      atomicAdd(&grad_input[base+off  ], -bb*gout);
      atomicAdd(&grad_input[base+off+bs],  a*gout);
      atomicAdd(&grad_params[pi+0],  x2*gout);
      atomicAdd(&grad_params[pi+1], -x1*gout);
    }
    idx += stride;
  }
}

// CUDA 버터플라이 레이어 backward 구현
std::vector<torch::Tensor> butterfly_layer_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor input,
    torch::Tensor params,
    int layer_idx)
{
  CHECK_CUDA_CONTIGUOUS(grad_out);
  CHECK_CUDA_CONTIGUOUS(input);
  CHECK_CUDA_CONTIGUOUS(params);

  int B = grad_out.size(0), D = grad_out.size(1);
  auto grad_input = torch::zeros_like(input);
  auto grad_params = torch::zeros_like(params);

  int threads = 256;
  int blocks = (B*D + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "butterfly_layer_backward_cuda", ([&] {
    butterfly_layer_backward_kernel<scalar_t><<<blocks, threads>>>(
      grad_out.data_ptr<scalar_t>(),
      input.data_ptr<scalar_t>(),
      params.data_ptr<scalar_t>(),
      grad_input.data_ptr<scalar_t>(),
      grad_params.data_ptr<scalar_t>(),
      B, D, layer_idx
    );
  }));
  cudaDeviceSynchronize();

  return { grad_input, grad_params };
} 