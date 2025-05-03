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