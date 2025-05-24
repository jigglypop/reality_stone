#pragma once
#include <torch/extension.h>
#include <config/constant.h>
#include <vector>

namespace reality_stone::advanced {
    
    /**
     * 측지선 기반 활성화 함수
     * 기존 비선형 활성화 함수 대신 측지선 변환 사용
     */
    class GeodesicActivation {
    private:
        torch::Tensor anchors;     // 앵커 포인트들 [K, D]
        torch::Tensor t_params;    // 측지선 파라미터 [K]
        torch::Tensor weights;     // 앵커별 가중치 [K]
        int num_anchors;
        int input_dim;
        
    public:
        GeodesicActivation(int input_dim, int num_anchors = 4);
        
        /**
         * 측지선 기반 비선형 변환
         * h = ∑ᵢ wᵢ · γ(tᵢ; 0, xᵢ)
         */
        torch::Tensor forward(const torch::Tensor& x, float curvature);
        
        /**
         * Einstein 중점 계산
         * M_E(x₁, ..., xₙ; w₁, ..., wₙ) = exp_0(∑ᵢ wᵢ·log_0(xᵢ))
         */
        torch::Tensor einstein_midpoint(
            const torch::Tensor& points,   // [K, D]
            const torch::Tensor& weights,  // [K]
            float curvature
        );
        
        /**
         * 다중 측지선 혼합
         */
        torch::Tensor multi_geodesic_mixing(
            const torch::Tensor& input,
            float curvature
        );
        
        // 파라미터 접근자
        torch::Tensor& get_anchors() { return anchors; }
        torch::Tensor& get_t_params() { return t_params; }
        torch::Tensor& get_weights() { return weights; }
    };
    
    // CUDA 함수 선언
    torch::Tensor geodesic_activation_cuda(
        const torch::Tensor& input,
        const torch::Tensor& anchors,
        const torch::Tensor& t_values,
        const torch::Tensor& weights,
        float curvature
    );
    
    torch::Tensor einstein_midpoint_cuda(
        const torch::Tensor& points,
        const torch::Tensor& weights,
        float curvature
    );
    
    torch::Tensor multi_geodesic_mixing_cuda(
        const torch::Tensor& input,
        const torch::Tensor& anchors,
        const torch::Tensor& t_values,
        const torch::Tensor& weights,
        float curvature
    );
    
} // namespace reality_stone::advanced 