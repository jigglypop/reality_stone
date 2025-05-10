#include <torch/extension.h>
#include <cmath>
#include <tensor.h>
#include <tensor.h>

namespace config = hyper_butterfly::config;

namespace hyper_butterfly::maps {
    torch::Tensor exp_map_cpu(torch::Tensor v, float c) {
        auto norm = torch::norm(v, 2, 1, true).clamp(config::Constants::EPS);
        float sqrt_c = std::sqrt(c);
        auto scn = (sqrt_c * norm).clamp(config::Constants::EPS, 10.0f);
        auto denom = scn + 1e-3f;
        auto numer = torch::tanh(scn);
        auto factor = numer / denom;
        return factor * v;
    }
}