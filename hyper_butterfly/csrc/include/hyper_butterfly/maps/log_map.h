#pragma once
#include <torch/extension.h>
#include <cuda.h>

namespace hyper_butterfly {
namespace maps {
torch::Tensor log_map_cpu(torch::Tensor x, float c);
#if defined(WITH_CUDA)
torch::Tensor log_map_forward_cuda(torch::Tensor x, float c);
torch::Tensor log_map_backward_cuda(torch::Tensor grad_y, torch::Tensor x, float c);
#endif
}
}
