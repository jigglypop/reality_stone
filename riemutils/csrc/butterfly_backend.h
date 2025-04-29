#pragma once
#include <torch/extension.h>

// CUDA-only backward for one butterfly layer
std::vector<torch::Tensor> butterfly_layer_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor input,
    torch::Tensor params,
    int layer_idx
);
