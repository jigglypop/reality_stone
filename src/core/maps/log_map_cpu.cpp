#include <torch/extension.h>
#include <cmath>
#include <utils/common_defs.h>
#include <maps/exp_map.h>
#include <utils/numeric.h>
#include <config/constant.h>

namespace config = reality_stone::config;

namespace reality_stone::maps {

    torch::Tensor log_map_cpu(torch::Tensor x, float c) {
        auto norm = torch::norm(x, 2, 1, true).clamp(config::Constants::EPS);
        float sqrt_c = std::sqrt(c);
        auto scn = (sqrt_c * norm).clamp(config::Constants::EPS, 1.0f - 1e-6f);
        auto denom = scn + config::Constants::EPS;
        auto numer = torch::atanh(scn);
        auto factor = numer / denom;
        return factor * x;
    }


}