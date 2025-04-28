#pragma once
#include <torch/extension.h>

// CPU implementations
torch::Tensor log_map_origin_cpu_export(torch::Tensor x, float c);
torch::Tensor exp_map_origin_cpu_export(torch::Tensor v, float c);
std::vector<torch::Tensor> hyper_butterfly_cpu_export(
    torch::Tensor x, torch::Tensor params, torch::Tensor args, float c, int L);

// CUDA implementations
torch::Tensor log_map_origin_cuda(torch::Tensor x, float c);
torch::Tensor exp_map_origin_cuda(torch::Tensor v, float c);
std::vector<torch::Tensor> hyper_butterfly_cuda(
    torch::Tensor x, torch::Tensor params, torch::Tensor args, float c, int L);
