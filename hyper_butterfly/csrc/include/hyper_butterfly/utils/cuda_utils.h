// #pragma once
// 
// #include <torch/extension.h>
// #include <cuda_runtime.h>                
// #include <hyper_butterfly/utils/common_defs.h>
// 
// // — 매크로 중복 정의 방지
// #ifndef CHECK_CUDA_CONTIGUOUS
// #define CHECK_CUDA_CONTIGUOUS(x)                                  \
//   TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor"); \
//   TORCH_CHECK((x).is_contiguous(),     #x " must be contiguous")
// #endif
// 
// #ifndef CUDA_CHECK
// #define CUDA_CHECK(err)                                           \
//   do {                                                            \
//     auto e = (err);                                               \
//     TORCH_CHECK(e == cudaSuccess, "CUDA error: ",                 \
//                 cudaGetErrorString(e));                           \
//   } while (0)
// #endif
// 
// // — CUDA 컴파일러(vs host compiler) 감지하여 inline 정의
// #if defined(__CUDACC__)
// #define DEVICE_INLINE __device__ __forceinline__
// #else
// #define DEVICE_INLINE inline
// #endif
// 
// namespace hyper_butterfly {
// namespace utils {
// 
// /// atanh 헬퍼 (clamp 포함)
// DEVICE_INLINE float atanh_device(float v) {
//   v = fminf(fmaxf(v, -1.0f + EPS), 1.0f - EPS);
//   return 0.5f * logf((1.0f + v) / (1.0f - v));
// }
// 
// }
// }

// hyper_butterfly/utils/cuda_utils.h

#pragma once
#include <torch/extension.h>
#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace hyper_butterfly {
namespace utils {

// CUDA error check function
inline void check_cuda_error() {
#ifdef WITH_CUDA
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
    "CUDA error: ", cudaGetErrorString(err));
#endif
}
// CUDA tensor check function
inline void check_cuda_tensor(const torch::Tensor& x) {
  TORCH_CHECK(x.device().is_cuda(), "tensor must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "tensor must be contiguous");
}
// device memory allocation
template <typename T>
inline T* cuda_malloc(size_t n) {
  T* ptr = nullptr;
#ifdef WITH_CUDA
  cudaMalloc(&ptr, n * sizeof(T));
#endif
  return ptr;
}
// device memory deallocation
template <typename T>
inline void cuda_free(T* ptr) {
#ifdef WITH_CUDA
  if (ptr) cudaFree(ptr);
#endif
}
}
}