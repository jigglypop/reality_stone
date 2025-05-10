#pragma once
#include <torch/torch.h>

namespace reality_stone::manifolds {
    class ManifoldInterface {
        public:
        virtual ~ManifoldInterface() = default;
        // 핵심 연산들
        virtual torch::Tensor add(torch::Tensor u, torch::Tensor v, float c) = 0;
        virtual torch::Tensor scalar(torch::Tensor u, float c, float r) = 0;
        virtual torch::Tensor geodesic(torch::Tensor u, torch::Tensor v, float c, float t) = 0;
        // 각 매니폴드별 헬퍼 함수들  
        virtual torch::Tensor dist(torch::Tensor u, torch::Tensor v, float c) = 0;
        virtual torch::Tensor exp_map(torch::Tensor u, float c) = 0;
        virtual torch::Tensor log_map(torch::Tensor x, float c) = 0;
    };
    // Factory 패턴
    std::unique_ptr<ManifoldInterface> create_manifold(const std::string& type);

}
