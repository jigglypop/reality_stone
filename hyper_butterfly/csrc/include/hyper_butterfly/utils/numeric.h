#pragma once
#include <cmath>
#include <algorithm>
#include <hyper_butterfly/config/constant.h>

namespace config = hyper_butterfly::config;

namespace hyper_butterfly {
namespace utils {
// CPU용 atanh 구현
inline float atanh(float x) {
    // 경계 클램핑
    x = std::max(-1.0f + config::Constants::BOUNDARY_EPS,
        std::min(x, 1.0f - config::Constants::BOUNDARY_EPS));
    return 0.5f * std::log((1.0f + x) / (1.0f - x));
}
// CUDA용 atanh 구현
#ifdef __CUDACC__
__device__ __forceinline__ float atanh_device(float x) {
    // 경계 클램핑
    x = fmaxf(-1.0f + config::Constants::BOUNDARY_EPS,
        fminf(x, 1.0f - config::Constants::BOUNDARY_EPS));
    return 0.5f * logf((1.0f + x) / (1.0f - x));
}
#endif
// 컨텍스트 태그 타입
struct CPUContext {};
struct CUDAContext {};
}

} 