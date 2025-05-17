#pragma once
#include <torch/extension.h>

namespace reality_stone::ops {
    torch::Tensor lorentz_add_cpu(
        torch::Tensor u,
        torch::Tensor v,
        float c
    );
    torch::Tensor lorentz_scalar_cpu(
        torch::Tensor u,
        float c,
        float r
    );
    torch::Tensor lorentz_inner_cpu(
        torch::Tensor u,
        torch::Tensor v
    );
    torch::Tensor lorentz_distance_cpu(
        torch::Tensor u,
        torch::Tensor v,
        float c
    );
    torch::Tensor poincare_to_lorentz_cpu(
        torch::Tensor x,
        float c
    );
    torch::Tensor lorentz_to_poincare_cpu(
        torch::Tensor x,
        float c
    );

#ifdef WITH_CUDA
    torch::Tensor lorentz_add_cuda(
        torch::Tensor u,
        torch::Tensor v,
        float c
    );
    torch::Tensor lorentz_scalar_cuda(
        torch::Tensor u,
        float c,
        float r
    );
    torch::Tensor lorentz_inner_cuda(
        torch::Tensor u,
        torch::Tensor v
    );
    torch::Tensor lorentz_distance_cuda(
        torch::Tensor u,
        torch::Tensor v,
        float c
    );
    torch::Tensor poincare_to_lorentz_cuda(
        torch::Tensor x,
        float c
    );
    torch::Tensor lorentz_to_poincare_cuda(
        torch::Tensor x,
        float c
    );
#endif
}