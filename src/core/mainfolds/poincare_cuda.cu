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
#include <hyper_butterfly/manifolds/poincare.h>

namespace utils = hyper_butterfly::utils;
namespace maps = hyper_butterfly::maps;
namespace manifolds = hyper_butterfly::manifolds;

namespace hyper_butterfly::manifolds {
  std::vector<torch::Tensor> poincare_forward_cuda(
    torch::Tensor x,
    torch::Tensor params,
    torch::Tensor unused,
    float c,
    int L) {
    utils::check_cuda_tensor(x);
    utils::check_cuda_tensor(params);
    int B = x.size(0), D = x.size(1);
    int D_padded = utils::next_pow2(D);
    torch::Tensor x_padded;
    if (D_padded > D) {
      x_padded = torch::zeros({ B, D_padded }, x.options());
      x_padded.narrow(1, 0, D).copy_(x);
    }
    else {
      x_padded = x;
    }
    torch::Tensor u = maps::log_map_forward_cuda(x_padded, c);
    torch::Tensor v = u.clone();
    for (int l = 0; l < L; l++) {
      int layer_idx = l % int(std::log2(D_padded));
      v = ops::butterfly_forward_cpu(v, params, layer_idx, B, D_padded);
    }
    // 4: Exp map
    torch::Tensor y_padded = maps::exp_map_forward_cuda(v, c);
    // 5: Slice to original dimension if needed
    torch::Tensor y = (D_padded > D) ? y_padded.narrow(1, 0, D) : y_padded;
    return { y, u, v };
  }

  std::vector<torch::Tensor> poincare_backward_cuda(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor params,
    float c,
    int L) {
    utils::check_cuda_tensor(grad_y);
    utils::check_cuda_tensor(x);
    utils::check_cuda_tensor(params);
    int B = x.size(0), D = x.size(1);
    int D_padded = utils::next_pow2(D);
    torch::Tensor x_padded, grad_y_padded;
    if (D_padded > D) {
      x_padded = torch::zeros({ B, D_padded }, x.options());
      x_padded.narrow(1, 0, D).copy_(x);
      grad_y_padded = torch::zeros({ B, D_padded }, grad_y.options());
      grad_y_padded.narrow(1, 0, D).copy_(grad_y);
    }
    else {
      x_padded = x;
      grad_y_padded = grad_y;
    }
    torch::Tensor u = maps::log_map_forward_cuda(x_padded, c);
    std::vector<torch::Tensor> intermediates;
    intermediates.push_back(u);
    torch::Tensor v = u.clone();
    for (int l = 0; l < L; l++) {
      int layer_idx = l % int(std::log2(D_padded));
      v = ops::butterfly_forward_cuda(v, params, layer_idx, B, D_padded);
      intermediates.push_back(v);
    }
    torch::Tensor y_padded = maps::exp_map_forward_cuda(v, c);
    int threads = std::min(D_padded, 1024);
    int shbytes = 2 * sizeof(float);
    auto grad_v = maps::exp_map_backward_cuda(v, grad_y_padded, c);
    auto grad_params = torch::zeros_like(params);
    auto grad_u = torch::zeros_like(u);
    torch::Tensor grad_curr = grad_v;
    for (int l = L - 1; l >= 0; l--) {
      int layer_idx = l % int(std::log2(D_padded));
      torch::Tensor input = intermediates[l];
      auto result = ops::butterfly_backward_cuda(
        grad_curr, input, params, layer_idx);
      torch::Tensor grad_input = result[0];
      torch::Tensor layer_grad_params = result[1];
      int p_offset = 0;
      for (int i = 0; i < layer_idx; i++) {
        int block_size = 1 << i;
        p_offset += 2 * (D_padded / (2 * block_size));
      }
      int p_size = 2 * (D_padded / (2 * (1 << layer_idx)));
      grad_params.narrow(0, p_offset, p_size).add_(layer_grad_params.narrow(0, p_offset, p_size));
      grad_curr = grad_input;
    }
    grad_u = grad_curr;
    torch::Tensor grad_x_padded = torch::zeros_like(x_padded);
    grad_x_padded = maps::log_map_backward_cuda(x_padded, grad_u, c);
    torch::Tensor grad_x = (D_padded > D) ? grad_x_padded.narrow(1, 0, D) : grad_x_padded;
    return { grad_x, grad_params };
  }
}
