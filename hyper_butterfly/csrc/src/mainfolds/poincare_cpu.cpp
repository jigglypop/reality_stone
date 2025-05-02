#include <torch/extension.h>
#include <cmath>
#include <vector>
#include <hyper_butterfly/utils/common_defs.h>
#include <hyper_butterfly/utils/cuda_utils.h>
#include <hyper_butterfly/maps/log_map.h>
#include <hyper_butterfly/maps/exp_map.h>
#include <hyper_butterfly/ops/butterfly.h>

namespace utils = hyper_butterfly::utils;
namespace maps = hyper_butterfly::maps;
namespace ops = hyper_butterfly::ops;

namespace hyper_butterfly {
namespace manifolds {
std::vector<torch::Tensor> poincare_forward_cpu(
    torch::Tensor x,
    torch::Tensor params,
    torch::Tensor /*args*/,
    float c,
    int L) {
    auto u = maps::log_map_cpu(x, c);
    auto v = u;
    int batch_size = x.size(0);
    int dim = x.size(1);
    for (int l = 0; l < L; ++l) {
        int layer_idx = l % int(std::log2(dim));
        v = ops::butterfly_forward_cpu(v, params, layer_idx, batch_size, dim);
    }
    auto y = maps::exp_map_cpu(v, c);
    return { y, u, v };
}
}
}