#pragma once
#include <torch/extension.h>
#include <config/constant.h>

namespace reality_stone::utils {
    
    /**
     * 텐서 안전성 체크 및 클리핑 (inline 구현)
     */
    inline torch::Tensor safe_clamp_tensor(torch::Tensor x) {
        namespace config = reality_stone::config;
        
        // 1. NaN/Inf 체크 및 제거
        auto mask_finite = torch::isfinite(x);
        if (!torch::all(mask_finite).item<bool>()) {
            x = torch::where(mask_finite, x, torch::zeros_like(x));
        }
        
        // 2. 극한값 클리핑
        x = torch::clamp(x, config::Constants::MIN_SAFE_VALUE, config::Constants::MAX_SAFE_VALUE);
        
        // 3. Norm 클리핑 (포인카레 디스크 내부 유지)
        auto norms = torch::norm(x, 2, /*dim=*/-1, /*keepdim=*/true);
        auto max_norm = config::Constants::MAX_NORM_RATIO;
        auto clipped_norms = torch::clamp_max(norms, max_norm);
        auto scale_factor = torch::where(norms > max_norm, clipped_norms / norms, torch::ones_like(norms));
        
        return x * scale_factor;
    }
    
    /**
     * 하이퍼볼릭 파라미터 안전성 체크 (inline 구현)
     */
    inline std::pair<float, float> safe_clamp_params(float c, float t) {
        namespace config = reality_stone::config;
        
        // c 값 안전성
        c = std::max(c, config::Constants::EPS);
        c = std::min(c, 100.0f);
        
        // t 값 범위 제한
        t = std::clamp(t, -1.0f, 1.0f);
        
        return std::make_pair(c, t);
    }
    
    /**
     * 그라디언트 폭발 방지 (inline 구현)
     */
    inline torch::Tensor gradient_safe_clamp(torch::Tensor x, float max_grad_norm = 50.0f) {
        auto grad_norm = torch::norm(x, 2);
        
        if (grad_norm.item<float>() > max_grad_norm) {
            x = x * (max_grad_norm / grad_norm);
        }
        
        return x;
    }
    
    /**
     * 포인카레 디스크 경계 안전성 (inline 구현)
     */
    inline torch::Tensor poincare_boundary_safe(torch::Tensor x) {
        namespace config = reality_stone::config;
        
        auto norms = torch::norm(x, 2, /*dim=*/-1, /*keepdim=*/true);
        auto safe_threshold = 1.0f - config::Constants::SAFE_BOUNDARY;
        
        auto scale_factor = torch::where(
            norms >= safe_threshold,
            safe_threshold / norms,
            torch::ones_like(norms)
        );
        
        return x * scale_factor;
    }
    
} 