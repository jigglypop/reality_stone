#pragma once

/**
 * Reality Stone Advanced Features
 * 고급 하이퍼볼릭 신경망 기능들의 통합 헤더
 */

#include <torch/extension.h>
#include <config/constant.h>

// Advanced 기능들 (올바른 경로)
#include <advanced/regularization/hyperbolic_regularization.h>
#include <advanced/dynamic_curvature/dynamic_curvature.h>
#include <advanced/fused_ops/fused_ops.h>
#include <advanced/geodesic_activation/geodesic_activation.h>

namespace reality_stone::advanced {
    
    /**
     * Advanced 기능 매니저
     */
    class AdvancedManager {
    public:
        struct Config {
            // 정규화 설정
            bool enable_regularization = true;
            float lambda_boundary = 1.0f;
            float lambda_curvature = 0.1f;
            float lambda_geodesic = 0.01f;
            
            // 동적 곡률 설정
            bool enable_dynamic_curvature = false;
            float base_curvature = 1.0f;
            
            // 융합 연산 설정
            bool enable_fused_ops = false;
            
            // 측지선 활성화 설정
            bool enable_geodesic_activation = false;
            int num_anchors = 4;
        };
        
        AdvancedManager(const Config& config = Config{});
        
        /**
         * MNIST NaN 문제 해결용 빠른 설정
         */
        static AdvancedManager create_for_mnist_fix();
        
        /**
         * 성능 최적화용 설정
         */
        static AdvancedManager create_for_performance();
        
        /**
         * 연구용 전체 기능 활성화
         */
        static AdvancedManager create_for_research();
        
        /**
         * 정규화 적용
         */
        torch::Tensor apply_regularization(
            const torch::Tensor& x,
            const torch::Tensor& weights,
            float curvature
        );
        
        /**
         * 동적 곡률 예측
         */
        torch::Tensor predict_curvature(const torch::Tensor& x);
        
        /**
         * 향상된 순전파 (모든 기능 통합)
         */
        torch::Tensor enhanced_forward(
            const torch::Tensor& input,
            const torch::Tensor& weight,
            const torch::Tensor& bias
        );
        
    private:
        Config config_;
    };
    
    /**
     * 편의 함수들
     */
    
    /**
     * 즉시 MNIST NaN 문제 해결
     */
    torch::Tensor fix_mnist_nan(
        const torch::Tensor& logits,
        float curvature = 1.0f
    );
    
    /**
     * 성능 벤치마크
     */
    struct OverallBenchmark {
        double baseline_time_ms;
        double advanced_time_ms;
        double speedup_ratio;
        double accuracy_improvement;
        size_t memory_overhead_mb;
    };
    
    OverallBenchmark benchmark_advanced_features(
        const torch::Tensor& input,
        const torch::Tensor& weight,
        const torch::Tensor& bias,
        int num_iterations = 100
    );
    
} // namespace reality_stone::advanced

/**
 * 편의 매크로들
 */
#define RS_ADVANCED_REGULARIZE(x, w, c) \
    reality_stone::advanced::fix_mnist_nan(x, c)

#define RS_ADVANCED_BENCHMARK(input, weight, bias) \
    reality_stone::advanced::benchmark_advanced_features(input, weight, bias) 