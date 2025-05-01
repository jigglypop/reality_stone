// hyper_butterfly/csrc/include/hyper_butterfly/utils/cuda_utils.h
#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>                 // cudaError_t, cudaGetErrorString
#include <hyper_butterfly/utils/common_defs.h>

// — 매크로 중복 정의 방지
#ifndef CHECK_CUDA_CONTIGUOUS
#define CHECK_CUDA_CONTIGUOUS(x)                                  \
  TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor"); \
  TORCH_CHECK((x).is_contiguous(),     #x " must be contiguous")
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(err)                                           \
  do {                                                            \
    auto e = (err);                                               \
    TORCH_CHECK(e == cudaSuccess, "CUDA error: ",                 \
                cudaGetErrorString(e));                           \
  } while (0)
#endif

// — CUDA 컴파일러(vs host compiler) 감지하여 inline 정의
#if defined(__CUDACC__)
#define DEVICE_INLINE __device__ __forceinline__
#else
#define DEVICE_INLINE inline
#endif

namespace hyper_butterfly {
namespace utils {

/// atanh 헬퍼 (clamp 포함)
DEVICE_INLINE float atanh_device(float v) {
    v = fminf(fmaxf(v, -1.0f + EPS), 1.0f - EPS);
    return 0.5f * logf((1.0f + v) / (1.0f - v));
}

} // namespace utils
} // namespace hyper_butterfly
