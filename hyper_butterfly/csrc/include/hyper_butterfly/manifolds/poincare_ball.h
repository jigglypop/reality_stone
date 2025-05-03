#pragma once
#include "manifold.h"
namespace hyper_butterfly {
namespace manifolds {

struct PoincareBall : Manifold {
    // x,y: (B×D), c>0
    torch::Tensor mobius_add(const torch::Tensor& x,
        const torch::Tensor& y,
        float c) const override {
        auto x2 = x.pow(2).sum(-1, true);
        auto y2 = y.pow(2).sum(-1, true);
        auto xy = (x * y).sum(-1, true);
        auto denom = 1 + 2 * c * xy + c * c * x2 * y2;
        auto num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y;
        return num / denom.clamp_min(1e-5f);
    }

    torch::Tensor mobius_scalar(const torch::Tensor& v,
        float t,
        float c) const override {
        // t ⊗ₚ v = (1/√c) tanh( t arctanh(√c‖v‖) ) v/‖v‖
        auto v_norm = v.norm(2, -1, true).clamp_min(1e-5f);
        auto sqrtc = std::sqrt(c);
        auto at = atanh(sqrtc * v_norm);
        auto t_at = (t * at).clamp(-15.0f, +15.0f);
        auto factor = (1.0f / sqrtc) * torch::tanh(t_at) / v_norm;
        return v * factor;
    }

    torch::Tensor exp_map0(const torch::Tensor& v,
        float c) const override {
        // same as mobius_scalar(v,1,c)
        return mobius_scalar(v, 1.0f, c);
    }

    torch::Tensor log_map0(const torch::Tensor& x,
        float c) const override {
        // log₀(x) = (1/√c) arctanh(√c‖x‖) x/‖x‖
        auto x_norm = x.norm(2, -1, true).clamp_min(1e-5f);
        auto sqrtc = std::sqrt(c);
        auto at = atanh(sqrtc * x_norm);
        auto factor = at / (sqrtc * x_norm);
        return x * factor;
    }

private:
    // reuse torch::atanh, but clamp inside
    torch::Tensor atanh(const torch::Tensor& z) const {
        // torch doesn’t have atanh on CPU at all versions
        return 0.5 * ((1 + z).log() - (1 - z).log());
    }
};

} // manifolds
} // hyper_butterfly
