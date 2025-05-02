#include <torch/extension.h>
#include <cmath>
#include <hyper_butterfly/utils/common_defs.h>
#include <hyper_butterfly/maps/exp_map.h>
#include <hyper_butterfly/utils/numeric.h>

namespace utils = hyper_butterfly::utils;
namespace hyper_butterfly {
namespace maps {
torch::Tensor log_map_cpu(torch::Tensor x, float c) {
    auto norm = torch::norm(x, 2, 1, true).clamp(utils::Constants::EPS);
    float sqrt_c = std::sqrt(c);
    auto scn = (sqrt_c * norm).clamp(utils::Constants::EPS, 1.0f - 1e-6f);
    auto denom = scn + utils::Constants::EPS;
    auto numer = torch::atanh(scn);
    auto factor = numer / denom;
    return factor * x;
}
}
}