#pragma once
#include <torch/extension.h>
#include <config/constant.h>
#include <vector>

namespace reality_stone::advanced {
    
    /**
     * 융합 하이퍼볼릭 선형 레이어
     * 로그맵 → 선형변환 → 지수맵을 하나의 커널로 융합
     */
    torch::Tensor hyperbolic_linear_fused(
        const torch::Tensor& input,
        const torch::Tensor& weight,
        const torch::Tensor& bias,
        float curvature
    );
    
    /**
     * 융합 Möbius 덧셈 체인
     */
    torch::Tensor mobius_chain_fused(
        const std::vector<torch::Tensor>& inputs,
        const std::vector<float>& curvatures
    );
    
    /**
     * 융합 변환-정규화 연산
     */
    std::tuple<torch::Tensor, torch::Tensor> transform_regularize_fused(
        const torch::Tensor& input,
        float curvature,
        float reg_lambda = 0.1f
    );
    
} // namespace reality_stone::advanced 