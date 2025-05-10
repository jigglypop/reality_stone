#pragma once
#include <cmath>
#include <algorithm>
#include <config/constant.h>

namespace config = reality_stone::config;

namespace reality_stone::utils {
inline float atanh(float x) {
    x = std::max(-1.0f + config::Constants::BOUNDARY_EPS,
        std::min(x, 1.0f - config::Constants::BOUNDARY_EPS));
    return 0.5f * std::log((1.0f + x) / (1.0f - x));
}
#ifdef __CUDACC__
__device__ __forceinline__ float atanh_device(float x) {
    x = fmaxf(-1.0f + config::Constants::BOUNDARY_EPS,
        fminf(x, 1.0f - config::Constants::BOUNDARY_EPS));
    return 0.5f * logf((1.0f + x) / (1.0f - x));
}
#endif
struct CPUContext {};
struct CUDAContext {};
} 