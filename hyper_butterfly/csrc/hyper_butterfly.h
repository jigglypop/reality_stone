#pragma once
#include <torch/extension.h>
#include "common_defs.h"
#include "maps.h"
#include "butterfly.h"

// CPU exports
torch::Tensor log_map_origin_cpu_export(torch::Tensor x, float c);
torch::Tensor exp_map_origin_cpu_export(torch::Tensor v, float c);
std::vector<torch::Tensor> hyper_butterfly_cpu_export(
    torch::Tensor x, 
    torch::Tensor params, 
    torch::Tensor unused, 
    float c,
    int L
);

#ifdef WITH_CUDA
// CUDA exports
torch::Tensor log_map_origin_cuda(torch::Tensor x, float c);
torch::Tensor exp_map_origin_cuda(torch::Tensor v, float c);
std::vector<torch::Tensor> hyper_butterfly_cuda(
    torch::Tensor x,
    torch::Tensor params,
    torch::Tensor unused,
    float c,
    int L
);
std::vector<torch::Tensor> hyper_butterfly_backward_cuda(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor params,
    float c,
    int L
);
#endif