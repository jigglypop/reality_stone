#include <torch/extension.h>
#include <ops/klein.h>
#include <layers/klein.h>

namespace ops = reality_stone::ops;

namespace reality_stone::layers {
    torch::Tensor klein_forward_cpu(torch::Tensor u, torch::Tensor v, float c, float t) {
        // 클라인 모델에서는 측지선이 직선 세그먼트로 표현되므로 직선 보간이 가능합니다.
        // 그러나 이는 비유클리드 거리를 보존하지 않으므로, 거리를 고려한 보정이 필요합니다.

        // 1. u와 v 사이의 거리 계산 
        auto dist = ops::klein_distance_cpu(u, v, c);

        // 2. 직선 세그먼트 계산 ((1-t)*u + t*v)
        auto direct_interpolation = (1.0f - t) * u + t * v;

        // 3. 정규화 (클라인 모델 경계 내부로 유지)
        auto norm_sq = torch::sum(direct_interpolation * direct_interpolation, /*dim=*/1, /*keepdim=*/true);
        auto max_norm_sq = 1.0f / c - 1e-6f;  // 경계에서 약간 안쪽으로

        // 벡터 방향은 유지하고 노름만 조정
        auto scale = torch::sqrt(torch::min(
            torch::ones_like(norm_sq),
            max_norm_sq / norm_sq.clamp_min(1e-8f)
        ));

        return direct_interpolation * scale;
    }

    std::tuple<torch::Tensor, torch::Tensor> klein_backward_cpu(
        torch::Tensor grad_output,
        torch::Tensor u,
        torch::Tensor v,
        float c,
        float t
    ) {
        auto grad_u = torch::zeros_like(u);
        auto grad_v = torch::zeros_like(v);

        // 단순화된 역전파 구현 (선형 보간 기준)
        grad_u = grad_output * (1.0f - t);
        grad_v = grad_output * t;

        return std::make_tuple(grad_u, grad_v);
    }
}