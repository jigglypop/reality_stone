#include <torch/extension.h>
#include <ops/klein.h>
#include <layers/klein.h>

namespace ops = reality_stone::ops;

namespace reality_stone::layers {
    torch::Tensor klein_forward_cpu(torch::Tensor u, torch::Tensor v, float c, float t) {
        auto dist = ops::klein_distance_cpu(u, v, c);
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

}