#pragma once
#include <torch/extension.h>
#include <limits>
#include <cmath>

namespace reality_stone::utils {

template<typename T>
class NumericSafety {
public:
    static constexpr T EPS = std::numeric_limits<T>::epsilon() * 100;
    static constexpr T SAFE_MIN = std::numeric_limits<T>::min() * 1000;
    static constexpr T SAFE_MAX = std::numeric_limits<T>::max() / 1000;
    static constexpr T ATANH_BOUND = T(0.9999);
    static constexpr T BOUNDARY_MARGIN = T(0.001);
    
    // CPU 버전
    static inline T safe_atanh(T x) {
        x = std::clamp(x, -ATANH_BOUND, ATANH_BOUND);
        return std::atanh(x);
    }
    
    static inline T safe_tanh(T x) {
        if (std::abs(x) > T(20.0)) {
            return x > 0 ? T(1.0) : T(-1.0);
        }
        return std::tanh(x);
    }
    
    static inline T safe_acosh(T x) {
        x = std::max(x, T(1.0) + EPS);
        return std::acosh(x);
    }
    
    static inline T safe_sqrt(T x) {
        return std::sqrt(std::max(x, T(0.0)));
    }
    
    static inline T safe_div(T a, T b) {
        return a / (b + (b >= 0 ? EPS : -EPS));
    }
    
    static inline bool is_finite(T x) {
        return std::isfinite(x) && std::abs(x) < SAFE_MAX;
    }
    
    // CUDA 버전
#ifdef __CUDACC__
    __device__ __forceinline__ static T safe_atanh_cuda(T x) {
        if constexpr (std::is_same_v<T, float>) {
            x = fmaxf(-ATANH_BOUND, fminf(x, ATANH_BOUND));
            return atanhf(x);
        } else {
            x = fmax(-ATANH_BOUND, fmin(x, ATANH_BOUND));
            return atanh(x);
        }
    }
    
    __device__ __forceinline__ static T safe_tanh_cuda(T x) {
        if constexpr (std::is_same_v<T, float>) {
            if (fabsf(x) > T(20.0)) {
                return x > 0 ? T(1.0) : T(-1.0);
            }
            return tanhf(x);
        } else {
            if (fabs(x) > T(20.0)) {
                return x > 0 ? T(1.0) : T(-1.0);
            }
            return tanh(x);
        }
    }
    
    __device__ __forceinline__ static T safe_acosh_cuda(T x) {
        if constexpr (std::is_same_v<T, float>) {
            x = fmaxf(x, T(1.0) + EPS);
            return acoshf(x);
        } else {
            x = fmax(x, T(1.0) + EPS);
            return acosh(x);
        }
    }
    
    __device__ __forceinline__ static T safe_sqrt_cuda(T x) {
        if constexpr (std::is_same_v<T, float>) {
            return sqrtf(fmaxf(x, T(0.0)));
        } else {
            return sqrt(fmax(x, T(0.0)));
        }
    }
    
    __device__ __forceinline__ static T safe_div_cuda(T a, T b) {
        return a / (b + (b >= 0 ? EPS : -EPS));
    }
    
    __device__ __forceinline__ static bool is_finite_cuda(T x) {
        if constexpr (std::is_same_v<T, float>) {
            return isfinite(x) && fabsf(x) < SAFE_MAX;
        } else {
            return isfinite(x) && fabs(x) < SAFE_MAX;
        }
    }
#endif
};

using SafeFloat = NumericSafety<float>;
using SafeDouble = NumericSafety<double>;

} 