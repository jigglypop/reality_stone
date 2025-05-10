#pragma once
#include <torch/extension.h>
#include <utils/common_defs.h>

namespace hyper_butterfly::ops {
    torch::Tensor butterfly_forward_cpu(
        torch::Tensor input,
        torch::Tensor params,
        int layer_idx,
        int batch_size,
        int dim);

#ifdef WITH_CUDA
    torch::Tensor butterfly_forward_cuda(
        torch::Tensor input,
        torch::Tensor params,
        int layer_idx,
        int batch_size,
        int dim);

    std::vector<torch::Tensor> butterfly_backward_cuda(
        torch::Tensor grad_out,
        torch::Tensor input,
        torch::Tensor params,
        int layer_idx);
#endif
}