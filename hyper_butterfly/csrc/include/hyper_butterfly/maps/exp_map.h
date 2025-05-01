#pragma once
#include <torch/extension.h>

namespace hyper_butterfly {
namespace maps {
torch::Tensor exp_map_cpu(torch::Tensor v, float c);
#ifdef WITH_CUDA
torch::Tensor exp_map_cuda(torch::Tensor v, float c);
#endif
}
}
