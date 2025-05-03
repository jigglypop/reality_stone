#pragma once
#include <torch/extension.h>
namespace hyper_butterfly {
namespace manifolds {

struct Manifold {
    // Möbius add:  x ⊕ₚ y
    virtual torch::Tensor mobius_add(const torch::Tensor& x,
        const torch::Tensor& y,
        float c) const = 0;
    // Möbius scalar mul:  t ⊗ₚ v
    virtual torch::Tensor mobius_scalar(const torch::Tensor& v,
        float t,
        float c) const = 0;
    // exp map at origin
    virtual torch::Tensor exp_map0(const torch::Tensor& v,
        float c) const = 0;
    // log map at origin
    virtual torch::Tensor log_map0(const torch::Tensor& x,
        float c) const = 0;
    // closed-form geodesic from x along tangent v:  x ⊕ₚ (t⊗ v)
    torch::Tensor geodesic(const torch::Tensor& x,
        const torch::Tensor& v,
        float t,
        float c) const {
        auto tv = mobius_scalar(v, t, c);
        return mobius_add(x, tv, c);
    }
    virtual ~Manifold() = default;
};

} // manifolds
} // hyper_butterfly
