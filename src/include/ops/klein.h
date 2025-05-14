#pragma once
#include <torch/extension.h>

namespace reality_stone::ops {
    torch::Tensor klein_add_cpu(
        torch::Tensor u,
        torch::Tensor v,
        float c
    );
    torch::Tensor klein_scalar_cpu(
        torch::Tensor u,
        float c,
        float r
    );
    torch::Tensor klein_distance_cpu(
        torch::Tensor u,
        torch::Tensor v,
        float c
    );
    torch::Tensor poincare_to_klein_cpu(
        torch::Tensor x,
        float c
    );
    torch::Tensor klein_to_poincare_cpu(
        torch::Tensor x,
        float c
    );
    torch::Tensor lorentz_to_klein_cpu(
        torch::Tensor x,
        float c
    );
    torch::Tensor klein_to_lorentz_cpu(
        torch::Tensor x,
        float c
    );

#ifdef WITH_CUDA
    // CUDA 버전 선언
    torch::Tensor klein_add_cuda(
        torch::Tensor u,
        torch::Tensor v,
        float c
    );

    torch::Tensor klein_scalar_cuda(
        torch::Tensor u,
        float c,
        float r
    );

    torch::Tensor klein_distance_cuda(
        torch::Tensor u,
        torch::Tensor v,
        float c
    );

    torch::Tensor poincare_to_klein_cuda(
        torch::Tensor x,
        float c
    );

    torch::Tensor klein_to_poincare_cuda(
        torch::Tensor x,
        float c
    );

    torch::Tensor lorentz_to_klein_cuda(
        torch::Tensor x,
        float c
    );

    torch::Tensor klein_to_lorentz_cuda(
        torch::Tensor x,
        float c
    );
#endif
}