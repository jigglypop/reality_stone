#pragma once

#include <torch/extension.h>
#include <vector>

namespace reality_stone::advanced::compression {

class HybridCompression {
public:
    struct CompressionConfig {
        float svd_ratio;
        float fft_quality;
        bool use_phase_correction;
        bool adaptive_rank;
    };

    // SVD + FFT 하이브리드 압축
    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
    compress_weight(const torch::Tensor& weight, const CompressionConfig& config);
    
    // 압축된 가중치 복원
    static torch::Tensor reconstruct_weight(
        const torch::Tensor& U,
        const torch::Tensor& S, 
        const torch::Tensor& V
    );
    
    // 레이어 융합
    static torch::Tensor fuse_layers_fft(
        const std::vector<torch::Tensor>& weights,
        const torch::Tensor& layer_importance,
        float fft_quality
    );
    
    // 적응적 랭크 결정
    static int determine_optimal_rank(
        const torch::Tensor& singular_values,
        float target_energy,
        float min_rank_ratio = 0.05,
        float max_rank_ratio = 0.9
    );
    
    // 주파수 도메인 필터링
    static torch::Tensor frequency_filter(
        const torch::Tensor& fft_data,
        float quality_threshold
    );
};

// CUDA 구현
torch::Tensor hybrid_compress_cuda(
    const torch::Tensor& weight,
    float svd_ratio,
    float fft_quality
);

torch::Tensor batch_svd_compress_cuda(
    const torch::Tensor& weights,
    const torch::Tensor& compression_ratios
);

torch::Tensor adaptive_fft_fusion_cuda(
    const torch::Tensor& weight_stack,
    const torch::Tensor& importance_scores,
    float base_quality
);

} // namespace reality_stone::advanced::compression 