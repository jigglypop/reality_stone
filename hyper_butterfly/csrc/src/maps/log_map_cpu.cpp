#include <torch/extension.h>
#include <cmath>
#include <hyper_butterfly/utils/common_defs.h>
#include <hyper_butterfly/maps/exp_map.h>

namespace hyper_butterfly {
namespace maps {
torch::Tensor log_map_cpu(torch::Tensor x, float c) {
    // compute norm: clamp to avoid division by zero
    auto norm = torch::norm(x, 2, 1, true).clamp(hyper_butterfly::utils::EPS);
    // sqrt of curvature
    float sqrt_c = std::sqrt(c);
    // scaled norm clamped for numerical stability
    auto scn = (sqrt_c * norm).clamp(hyper_butterfly::utils::EPS, 1.0f - 1e-6f);
    // denominator and numerator for the atanh factor
    auto denom = scn + hyper_butterfly::utils::EPS;
    auto numer = torch::atanh(scn);
    auto factor = numer / denom;
    // return mapped tensor
    return factor * x;
}

} // namespace maps
} // namespace hyper_butterfly