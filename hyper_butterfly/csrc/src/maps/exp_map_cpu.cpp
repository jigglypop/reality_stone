#include <torch/extension.h>
#include <cmath>
#include <hyper_butterfly/utils/common_defs.h>
#include <hyper_butterfly/maps/exp_map.h>

namespace hyper_butterfly {

namespace maps {
torch::Tensor exp_map_cpu(torch::Tensor v, float c) {
    // compute norm: clamp to avoid division by zero
    auto norm = torch::norm(v, 2, 1, true).clamp(hyper_butterfly::utils::EPS);
    // sqrt of curvature
    float sqrt_c = std::sqrt(c);
    // scaled norm clamped for numerical stability
    auto scn = (sqrt_c * norm).clamp(hyper_butterfly::utils::EPS, 10.0f);
    // denominator and numerator for the tanh factor
    auto denom = scn + 1e-3f;
    auto numer = torch::tanh(scn);
    auto factor = numer / denom;
    // return mapped tensor
    return factor * v;
}
} // namespace maps
} // namespace hyper_butterfly