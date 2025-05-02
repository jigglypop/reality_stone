#include <torch/extension.h>
#include <cuda.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <hyper_butterfly/utils/common_defs.h>
#include <hyper_butterfly/utils/cuda_utils.h>
#include <hyper_butterfly/maps/exp_map.h>
#include <hyper_butterfly/maps/log_map.h>
#include <hyper_butterfly/ops/butterfly.h>

namespace utils = hyper_butterfly::utils;
namespace maps = hyper_butterfly::maps;

template <typename scalar_t>
__global__ void butterfly_backward_kernel(
  const scalar_t* __restrict__ grad_out,
  const scalar_t* __restrict__ input,
  const scalar_t* __restrict__ params,
  scalar_t* __restrict__ grad_input,
  scalar_t* __restrict__ grad_params,
  int B, int D, int layer_idx) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int bs = 1 << layer_idx, nb = D / (2 * bs);

  while (idx < B * D) {
    int b = idx / D, f = idx % D;
    int blk = (f / (2 * bs)) % nb, loc = f % (2 * bs), off = loc % bs;
    bool high = loc >= bs;
    int pi = blk * 2;
    float a = params[pi + 0], bb = params[pi + 1];
    int base = b * D + blk * 2 * bs;
    float x1 = input[base + off], x2 = input[base + off + bs];
    float gout = grad_out[idx];

    if (!high) {
      // y = a*x1 + b*x2
      atomicAdd(&grad_input[base + off], a * gout);
      atomicAdd(&grad_input[base + off + bs], bb * gout);
      atomicAdd(&grad_params[pi + 0], x1 * gout);
      atomicAdd(&grad_params[pi + 1], x2 * gout);
    }
    else {
      // y = -b*x1 + a*x2
      atomicAdd(&grad_input[base + off], -bb * gout);
      atomicAdd(&grad_input[base + off + bs], a * gout);
      atomicAdd(&grad_params[pi + 0], x2 * gout);
      atomicAdd(&grad_params[pi + 1], -x1 * gout);
    }
    idx += stride;
  }
}

template <typename scalar_t>
__global__ void butterfly_forward_kernel(
  const scalar_t* __restrict__ input,
  scalar_t* __restrict__ output,
  const scalar_t* __restrict__ params,
  int B, int D, int layer_idx) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int bs = 1 << layer_idx;
  int nb = D / (2 * bs);

  while (idx < B * D) {
    int b = idx / D, f = idx % D;
    int blk = (f / (2 * bs)) % nb,
      loc = f % (2 * bs),
      off = loc % bs;
    bool high = loc >= bs;
    int pi = blk * 2;
    float a = params[pi + 0],
      bb = params[pi + 1];
    int base = b * D + blk * 2 * bs;
    float x1 = input[base + off],
      x2 = input[base + off + bs];
    output[idx] = high
      ? (-bb * x1 + a * x2)
      : (a * x1 + bb * x2);
    idx += stride;
  }
}

namespace hyper_butterfly {
namespace ops {
torch::Tensor butterfly_forward_cuda(
  torch::Tensor input,
  torch::Tensor params,
  int layer_idx,
  int batch_size,
  int dim) {

  auto output = torch::empty_like(input);
  dim3 grid(std::min((batch_size * dim + 511) / 512, 1024));
  dim3 block(512);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_forward_cuda", ([&] {
    butterfly_forward_kernel<scalar_t> << <grid, block >> > (
      input.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      params.data_ptr<scalar_t>(),
      batch_size, dim, layer_idx);
    }));
  utils::check_cuda_error();
  return output;
}

std::vector<torch::Tensor> butterfly_backward_cuda(
  torch::Tensor grad_out,
  torch::Tensor input,
  torch::Tensor params,
  int layer_idx) {
  auto grad_input = torch::zeros_like(input);
  auto grad_params = torch::zeros_like(params);
  int batch_size = input.size(0);
  int dim = input.size(1);
  dim3 grid(std::min((batch_size * dim + 511) / 512, 1024));
  dim3 block(512);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_backward_cuda", ([&] {
    butterfly_backward_kernel<scalar_t> << <grid, block >> > (
      grad_out.data_ptr<scalar_t>(),
      input.data_ptr<scalar_t>(),
      grad_input.data_ptr<scalar_t>(),
      params.data_ptr<scalar_t>(),
      grad_params.data_ptr<scalar_t>(),
      batch_size, dim, layer_idx);
    }));
  utils::check_cuda_error();
  return { grad_input, grad_params };
}
}
}