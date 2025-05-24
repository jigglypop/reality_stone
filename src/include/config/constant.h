#pragma once
#include <cmath>
#include <algorithm>
#include <limits>

namespace reality_stone::config {
    struct Constants {
        // 기본 정밀도 상수
        static constexpr float EPS = 1e-6f;
        static constexpr float BOUNDARY_EPS = 1e-5f;  // 10배 증가
        static constexpr float MAX_TANH_ARG = 15.0f;
        static constexpr float MIN_DENOMINATOR = 1e-6f;  // 100배 증가! (1e-8f → 1e-6f)
        static constexpr float SAFE_LOGEXP_BOUNDARY = 1.0f - 1e-6f;
        static constexpr float INF = std::numeric_limits<float>::infinity();
        
        // 새로 추가된 안전 상수들
        static constexpr float GRADIENT_CLIP = 50.0f;        // 그래디언트 클리핑
        static constexpr float MAX_NORM_RATIO = 0.999f;      // 최대 norm 비율
        static constexpr float SAFE_BOUNDARY = 1e-4f;        // 안전 경계값
        static constexpr float MAX_SAFE_VALUE = 1e6f;        // 최대 안전값
        static constexpr float MIN_SAFE_VALUE = -1e6f;       // 최소 안전값
    };
}