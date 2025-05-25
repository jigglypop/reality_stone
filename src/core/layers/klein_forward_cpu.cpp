#include <torch/extension.h>
#include <ops/klein.h>
#include <ops/mobius.h>
#include <layers/klein.h>

namespace ops = reality_stone::ops;

namespace reality_stone::layers {
    torch::Tensor klein_forward_cpu(torch::Tensor u, torch::Tensor v, float c, float t) {
        // Klein 모델에서 올바른 측지선 계산
        // 방법: Klein -> Poincaré -> 측지선 -> Klein
        
        // 1. Klein 점들을 Poincaré 모델로 변환
        auto u_poincare = ops::klein_to_poincare_cpu(u, c);
        auto v_poincare = ops::klein_to_poincare_cpu(v, c);
        
        // 2. Poincaré 모델에서 올바른 측지선 계산
        // Poincaré에서 측지선: u ⊕_c (t ⊗_c ((-u) ⊕_c v))
        auto minus_u = ops::mobius_scalar_cpu(u_poincare, c, -1.0f);
        auto delta = ops::mobius_add_cpu(minus_u, v_poincare, c);
        auto delta_t = ops::mobius_scalar_cpu(delta, c, t);
        auto result_poincare = ops::mobius_add_cpu(u_poincare, delta_t, c);
        
        // 3. 결과를 Klein 모델로 다시 변환
        auto result_klein = ops::poincare_to_klein_cpu(result_poincare, c);
        
        // 4. Klein 모델 경계 안전성 확보
        auto norm_sq = torch::sum(result_klein * result_klein, /*dim=*/1, /*keepdim=*/true);
        auto max_norm_sq = 1.0f / c - 1e-6f;  // 경계에서 약간 안쪽으로
        
        auto scale = torch::sqrt(torch::min(
            torch::ones_like(norm_sq),
            max_norm_sq / norm_sq.clamp_min(1e-8f)
        ));
        
        return result_klein * scale;
    }
}