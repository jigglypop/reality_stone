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
        static constexpr float MIN_DENOMINATOR = 1e-5f;      // 1e-6f → 1e-5f로 증가
        static constexpr float SAFE_LOGEXP_BOUNDARY = 1.0f - 1e-6f;
        static constexpr float INF = std::numeric_limits<float>::infinity();
        // 새로 추가된 안전 상수들
        static constexpr float GRADIENT_CLIP = 50.0f;        // 그래디언트 클리핑
        static constexpr float MAX_NORM_RATIO = 0.99f;       // 최대 노름 비율
        static constexpr float SAFE_BOUNDARY = 1e-4f;        // 안전 경계값
        static constexpr float MAX_SAFE_VALUE = 1e6f;        // 최대 안전값
        static constexpr float MIN_SAFE_VALUE = -1e6f;       // 최소 안전값
        // 수치 안정성 상수들
        static constexpr float BOUNDARY_EPSILON = 1e-4f;     // 경계 근처 안전 마진
        static constexpr float SAFE_ATANH_BOUND = 0.999f;    // atanh 안전 경계
        static constexpr float LOG_SUM_EXP_THRESHOLD = 50.0f; // log-sum-exp 임계값
        // 하이퍼볼릭 기하학 상수들
        static constexpr float DEFAULT_CURVATURE = 1.0f;
        static constexpr float MIN_CURVATURE = 1e-3f;
        static constexpr float MAX_CURVATURE = 100.0f;
        // 메모리 제한
        static constexpr int MAX_LORENTZ_DIM = 1024;          // Lorentz 모델 최대 차원
        static constexpr int CUDA_BLOCK_SIZE = 256;           // CUDA 블록 크기
    };
}