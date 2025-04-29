#pragma once
#include <torch/extension.h>

torch::Tensor log_map_cpu(torch::Tensor x, float c);

torch::Tensor log_map_cuda(torch::Tensor x, float c);