#include <torch/extension.h>
#include <ops/mobius.h>
#include <layers/poincare_ball.h>
#include <utils/safety.h>

namespace ops = reality_stone::ops;
namespace utils = reality_stone::utils;

namespace reality_stone::layers {
    
    torch::Tensor poincare_ball_forward_cpu(torch::Tensor u, torch::Tensor v, float c, float t) {
        // 입력 안전성 체크
        u = utils::safe_clamp_tensor(u);
        v = utils::safe_clamp_tensor(v);
        
        // 파라미터 안전성 체크
        auto [safe_c, safe_t] = utils::safe_clamp_params(c, t);
        
        // 하이퍼볼릭 연산 수행
        auto minus_u = ops::mobius_scalar_cpu(u, safe_c, -1.0f);
        minus_u = utils::safe_clamp_tensor(minus_u);
        
        auto delta = ops::mobius_add_cpu(minus_u, v, safe_c);
        delta = utils::safe_clamp_tensor(delta);
        
        auto delta_t = ops::mobius_scalar_cpu(delta, safe_c, safe_t);
        delta_t = utils::safe_clamp_tensor(delta_t);
        
        auto result = ops::mobius_add_cpu(u, delta_t, safe_c);
        result = utils::poincare_boundary_safe(result);  // 최종 경계 안전성
        
        return result;
    }
}
