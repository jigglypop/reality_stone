#pragma once
#include <cmath>
#include <algorithm>

namespace hyper_butterfly {
namespace config {
// constants for numerical stability in hyperbolic geometry
struct Constants {
    // normal epslion (for numerical stability)
    static constexpr float EPS = 1e-6f;
    // tanh epslion (for numerical stability)
    static constexpr float BOUNDARY_EPS = 1e-6f;
    // tanh maximum value (for numerical stability)
    static constexpr float MAX_TANH_ARG = 15.0f;
    // tanh minimum value (for numerical stability)
    static constexpr float MIN_DENOMINATOR = 1e-8f;
    // log/exp map (for numerical stability)
    static constexpr float SAFE_LOGEXP_BOUNDARY = 1.0f - 1e-6f;
    // NaN value (for numerical stability)
    static constexpr float INF = std::numeric_limits<float>::infinity();
};

}
}