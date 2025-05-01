// #pragma once
// #include <torch/extension.h>
// 
// namespace hyper_butterfly {
// namespace manifold {
// 
// struct Manifold {
//     virtual torch::Tensor log_map(const torch::Tensor& x, float c) = 0;
//     virtual torch::Tensor exp_map(const torch::Tensor& v, float c) = 0;
//     virtual ~Manifold() = default;
// };
// 
// Manifold* create_manifold(const std::string& name);
// 
// } // namespace manifold
// } // namespace hyper_butterfly