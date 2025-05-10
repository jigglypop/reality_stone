#pragma once
#include <cmath>
#include <algorithm>

namespace reality_stone::config {
    struct Constants {
        static constexpr float EPS = 1e-6f;
        static constexpr float BOUNDARY_EPS = 1e-6f;
        static constexpr float MAX_TANH_ARG = 15.0f;
        static constexpr float MIN_DENOMINATOR = 1e-8f;
        static constexpr float SAFE_LOGEXP_BOUNDARY = 1.0f - 1e-6f;
        static constexpr float INF = std::numeric_limits<float>::infinity();
    };
}