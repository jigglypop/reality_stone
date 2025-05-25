#pragma once
#include <torch/extension.h>
#include <config/constant.h>

namespace reality_stone::advanced {
    
    enum RegularizationType {
        BOUNDARY_REG = 0,      // 경계 근접 페널티
        CURVATURE_REG = 1,     // 곡률 적응 정규화
        GEODESIC_VAR_REG = 2   // 측지선 분산 정규화
    };
    
    /**
     * 하이퍼볼릭 정규화 손실 계산
     * 
     * 이 함수는 이전 NaN 문제를 해결하는 핵심 기능입니다!
     * 포인카레 디스크 경계(||x|| → 1)에서의 수치적 불안정성을 방지합니다.
     */
    class HyperbolicRegularizer {
    public:
        /**
         * 경계 근접 페널티
         * R_boundary(x) = max(0, ||x|| - (1/√c - ε))²
         * 
         * 핵심: 포인카레 디스크 경계 접근을 강력하게 억제
         */
        static torch::Tensor boundary_penalty(
            const torch::Tensor& x,
            float curvature,
            float epsilon = 0.01f
        );
        
        /**
         * 곡률 적응 정규화
         * R_curvature(x) = ||log_0(x)||² · c
         * 
         * 원점으로부터의 하이퍼볼릭 거리에 비례한 페널티
         */
        static torch::Tensor curvature_adaptive_penalty(
            const torch::Tensor& x,
            float curvature
        );
        
        /**
         * 측지선 분산 정규화
         * R_geodesic(W) = ∑ᵢⱼ d_H²(wᵢ, wⱼ) / n²
         * 
         * 가중치들 간의 하이퍼볼릭 거리 분산 최소화
         */
        static torch::Tensor geodesic_variance_penalty(
            const torch::Tensor& weights,
            float curvature
        );
        
        /**
         * 통합 정규화 손실
         * λ₁·R_boundary + λ₂·R_curvature + λ₃·R_geodesic
         */
        static torch::Tensor combined_regularization(
            const torch::Tensor& x,
            const torch::Tensor& weights,
            float curvature,
            float lambda_boundary = 1.0f,
            float lambda_curvature = 0.1f,
            float lambda_geodesic = 0.01f
        );
        
    private:
        /**
         * 안전한 하이퍼볼릭 거리 계산
         * 수치적 안정성을 위한 클리핑 포함
         */
        static torch::Tensor safe_hyperbolic_distance(
            const torch::Tensor& x,
            const torch::Tensor& y,
            float curvature
        );
        
        /**
         * 안전한 로그 맵 계산
         * atanh 발산 방지
         */
        static torch::Tensor safe_log_map(
            const torch::Tensor& x,
            float curvature
        );
    };
    
    // CUDA 함수 선언들
    torch::Tensor boundary_penalty_cuda(
        const torch::Tensor& x,
        float curvature,
        float epsilon = 0.01f
    );
    
    torch::Tensor curvature_adaptive_penalty_cuda(
        const torch::Tensor& x,
        float curvature
    );
    
    torch::Tensor geodesic_variance_penalty_cuda(
        const torch::Tensor& weights,
        float curvature
    );
    
    torch::Tensor combined_regularization_cuda(
        const torch::Tensor& x,
        const torch::Tensor& weights,
        float curvature,
        float lambda_boundary = 1.0f,
        float lambda_curvature = 0.1f,
        float lambda_geodesic = 0.01f
    );
    
} // namespace reality_stone::advanced