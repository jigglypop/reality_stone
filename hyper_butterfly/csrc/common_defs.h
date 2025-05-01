// #pragma once
// 
// #include <torch/extension.h>
// #include <cmath>
// #include <vector>
// 
// // 공통 상수 정의
// static constexpr float EPS = 1e-6f;
// 
// // CUDA용 체크 매크로
// #ifdef WITH_CUDA
// #define CHECK_CUDA_CONTIGUOUS(x)                                    \
//   TORCH_CHECK((x).device().is_cuda(), #x " must be CUDA tensor");   \
//   TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
// 
// #define CUDA_CHECK(err)                                             \
//   do {                                                              \
//     auto e = (err);                                                 \
//     TORCH_CHECK(e == cudaSuccess, "CUDA error: ",                   \
//                 cudaGetErrorString(e));                             \
//   } while (0)
// #endif
// 
// // 다음 2의 거듭제곱 계산 유틸리티 함수
// inline int next_pow2(int v) {
//   v--;
//   v |= v >> 1;
//   v |= v >> 2;
//   v |= v >> 4;
//   v |= v >> 8;
//   v |= v >> 16;
//   return v + 1;
// } 