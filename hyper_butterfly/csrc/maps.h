// #pragma once
// #include <torch/extension.h>
// #include <hyper_butterfly/utils/common_defs.h>
// 
// namespace hyper_butterfly {
// 
// // CPU 함수 선언
// torch::Tensor log_map_origin_cpu(torch::Tensor x, float c);
// torch::Tensor exp_map_origin_cpu(torch::Tensor v, float c);
// 
// #ifdef WITH_CUDA
// // CUDA 함수 선언
// torch::Tensor log_map_origin_cuda(torch::Tensor x, float c);
// torch::Tensor exp_map_origin_cuda(torch::Tensor v, float c);
// #endif
// 
// // CPU 구현 (인라인)
// inline torch::Tensor log_map_origin_cpu(torch::Tensor x, float c) {
//     // 로그 맵 구현
//     auto norm = torch::norm(x, 2, 1, true).clamp(EPS);
//     float sqrt_c = std::sqrt(c);
//     auto scn = (sqrt_c * norm).clamp(EPS, 1.0f - 1e-6f);
//     auto denom = scn + EPS;
//     auto numer = torch::atanh(scn);
//     auto factor = numer / denom;
//     return factor * x;
// }
// 
// inline torch::Tensor exp_map_origin_cpu(torch::Tensor v, float c) {
//     // 지수 맵 구현
//     auto norm = torch::norm(v, 2, 1, true).clamp(EPS);
//     float sqrt_c = std::sqrt(c);
//     auto scn = (sqrt_c * norm).clamp(EPS, 10.0f);
//     auto denom = scn + 1e-3f;
//     auto numer = torch::tanh(scn);
//     auto factor = numer / denom;
//     return factor * v;
// }
// 
// } // namespace hyper_butterfly 