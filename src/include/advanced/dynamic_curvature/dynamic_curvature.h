#pragma once
#include <torch/extension.h>
#include <config/constant.h>

namespace reality_stone::advanced {
    
    /**
     * 동적 곡률 예측 클래스
     * 입력에 따라 적응적으로 곡률을 조정
     */
    class DynamicCurvature {
    private:
        torch::Tensor weight_c;  // 곡률 예측 가중치
        torch::Tensor bias_c;    // 곡률 예측 바이어스
        float c_base;            // 기본 곡률 상수
        
    public:
        DynamicCurvature(int input_dim, float base_curvature = 1.0f);
        
        /**
         * 입력에 따른 곡률 예측
         * @param x: 입력 텐서 [B, D]
         * @return: 곡률 값들 [B]
         */
        torch::Tensor predict_curvature(const torch::Tensor& x);
        
        /**
         * 특징 추출 함수 (L2 norm 기반)
         */
        torch::Tensor extract_features(const torch::Tensor& x);
        
        /**
         * 시그모이드 곡률 정규화
         */
        torch::Tensor normalize_curvature(const torch::Tensor& logits);
        
        // 파라미터 접근자
        torch::Tensor& get_weight() { return weight_c; }
        torch::Tensor& get_bias() { return bias_c; }
    };
    
    /**
     * 동적 곡률을 사용하는 Möbius 덧셈
     */
    torch::Tensor dynamic_mobius_add(
        const torch::Tensor& u,
        const torch::Tensor& v, 
        const torch::Tensor& curvatures  // 배치별 곡률 [B]
    );
    
    /**
     * 동적 곡률을 사용하는 포인카레 레이어
     */
    torch::Tensor dynamic_poincare_layer(
        const torch::Tensor& u,
        const torch::Tensor& v,
        const torch::Tensor& curvatures,
        float t
    );
    
    // CUDA 함수 선언들
    torch::Tensor dynamic_curvature_prediction_cuda(
        const torch::Tensor& features,
        const torch::Tensor& weight,
        const torch::Tensor& bias,
        float c_base
    );
    
    torch::Tensor dynamic_mobius_add_cuda(
        const torch::Tensor& u,
        const torch::Tensor& v, 
        const torch::Tensor& curvatures
    );
    
    torch::Tensor dynamic_poincare_layer_cuda(
        const torch::Tensor& u,
        const torch::Tensor& v,
        const torch::Tensor& curvatures,
        float t
    );
    
} // namespace reality_stone::advanced