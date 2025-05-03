#pragma once
#include <cmath>
#include <algorithm>
#include <hyper_butterfly/config/constant.h>

namespace Constants = hyper_butterfly::config::Constants;

namespace hyper_butterfly {
namespace utils {
// CPU atanh 
inline float tanh(float x) {
    x = std::clamp(x, -Constants::MAX_TANH_ARG, Constants::MAX_TANH_ARG);
    return std::tanh(x);
}
inline float atanh(float x) {
    x = std::clamp(x,
        -1.0f + Constants::BOUNDARY_EPS,
        1.0f - Constants::BOUNDARY_EPS);
    return 0.5f * std::log((1.0f + x) / (1.0f - x));
}
// CUDA atanh
#ifdef __CUDACC__
__device__ __forceinline__ float atanh_device(float x) {
    // 경계 클램핑
    x = fmaxf(-1.0f + Constants::BOUNDARY_EPS,
        fminf(x, 1.0f - Constants::BOUNDARY_EPS));
    return 0.5f * logf((1.0f + x) / (1.0f - x));
}
// CUDA tanh 
__device__ __forceinline__ float tanh_device(float x) {
    // 큰 값 클램핑
    x = fmaxf(-Constants::MAX_TANH_ARG,
        fminf(x, Constants::MAX_TANH_ARG));
    return tanhf(x);
}
// ------------------------------------------------------------------
// 2. NaN, Inf check utility
// ------------------------------------------------------------------
// NaN check utility
template <typename T>
inline bool is_nan(T value) {
    return std::isnan(value);
}
// inf check utility
template <typename T>
inline bool is_inf(T value) {
    return std::isinf(value);
}
// NaN/Inf check utility
template <typename T>
inline bool is_valid(T value) {
    return !is_nan(value) && !is_inf(value);
}

// 컨텍스트 태그 타입
struct CPUContext {};
struct CUDAContext {};
}

} 