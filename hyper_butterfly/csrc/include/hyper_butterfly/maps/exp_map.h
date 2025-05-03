#pragma once
#include <torch/extension.h>
#include <cuda.h>

namespace hyper_butterfly {
namespace maps {

torch::Tensor exp_map_cpu(torch::Tensor v, float c);
#if defined(WITH_CUDA) 
torch::Tensor exp_map_forward_cuda(torch::Tensor v, float c);
torch::Tensor exp_map_backward_cuda(torch::Tensor v, torch::Tensor grad_y, float c);
#endif
}

}
