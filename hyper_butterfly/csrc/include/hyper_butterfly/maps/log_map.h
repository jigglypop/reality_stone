#pragma once
#include <torch/extension.h>

namespace hyper_butterfly {
namespace maps {
torch::Tensor log_map_cpu(torch::Tensor x, float c);
#ifdef WITH_CUDA
torch::Tensor log_map_cuda(torch::Tensor x, float c);
#endif
}
}
