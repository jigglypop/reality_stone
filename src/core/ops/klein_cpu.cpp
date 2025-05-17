#include <torch/extension.h>
#include <cmath>
#include <ops/klein.h>
#include <config/constant.h>
#include <utils/numeric.h>

namespace config = reality_stone::config;
namespace utils = reality_stone::utils;

namespace reality_stone::ops {

    torch::Tensor klein_distance_cpu(
        torch::Tensor u,
        torch::Tensor v,
        float c
    ) {
        float sqrtc = std::sqrt(c);
        auto u_norm_sq = torch::sum(u * u, /*dim=*/1, /*keepdim=*/true);
        auto v_norm_sq = torch::sum(v * v, /*dim=*/1, /*keepdim=*/true);
        auto uv = torch::sum(u * v, /*dim=*/1, /*keepdim=*/true);
        auto numerator = 2 * (u_norm_sq * v_norm_sq - uv * uv);
        auto denominator = ((1 - c * u_norm_sq) * (1 - c * v_norm_sq))
            .clamp_min(config::Constants::EPS);
        auto lambda = torch::sqrt(numerator / denominator);
        auto two_minus_lambda_sq = (2.0f - lambda).clamp_min(config::Constants::EPS);
        auto dist = torch::acosh((2.0f + lambda) / two_minus_lambda_sq) / sqrtc;
        return dist;
    }

    torch::Tensor klein_add_cpu(
        torch::Tensor u,
        torch::Tensor v,
        float c
    ) {
        auto u_norm_sq = torch::sum(u * u, /*dim=*/1, /*keepdim=*/true);
        auto v_norm_sq = torch::sum(v * v, /*dim=*/1, /*keepdim=*/true);
        auto u_denom = (1.0f - c * u_norm_sq).clamp_min(config::Constants::EPS);
        auto v_denom = (1.0f - c * v_norm_sq).clamp_min(config::Constants::EPS);

        // 결과 계산
        auto result = u / u_denom.sqrt() + v / v_denom.sqrt();

        // 정규화
        auto result_norm_sq = torch::sum(result * result, /*dim=*/1, /*keepdim=*/true);
        auto result_denom = (1.0f + torch::sqrt(1.0f + c * result_norm_sq))
            .clamp_min(config::Constants::EPS);

        return result / result_denom;
    }

    torch::Tensor klein_scalar_cpu(
        torch::Tensor u,
        float c,
        float r
    ) {
        // 클라인 모델에서 스칼라 곱셈 구현
        auto norm = torch::norm(u, 2, /*dim=*/1, /*keepdim=*/true)
            .clamp_min(config::Constants::EPS);

// 단위 벡터 방향
        auto u_dir = u / norm;

        // 스케일링된 노름 계산
        auto scaled_norm = norm * r;

        // 최대 유효 범위 제한 (클라인 모델의 경계는 1/sqrt(c))
        auto max_norm = 1.0f / std::sqrt(c) - config::Constants::BOUNDARY_EPS;
        scaled_norm = scaled_norm.clamp_max(max_norm);

        return u_dir * scaled_norm;
    }

    torch::Tensor poincare_to_klein_cpu(
        torch::Tensor x,
        float c
    ) {
        // 푸앵카레 볼에서 클라인 모델로 변환
        float sqrtc = std::sqrt(c);
        auto x_norm_sq = torch::sum(x * x, /*dim=*/1, /*keepdim=*/true);
        auto denom = (1.0f + c * x_norm_sq).clamp_min(config::Constants::EPS);

        return 2.0f * x / denom;
    }

    torch::Tensor klein_to_poincare_cpu(
        torch::Tensor x,
        float c
    ) {
        // 클라인 모델에서 푸앵카레 볼로 변환
        float sqrtc = std::sqrt(c);
        auto x_norm_sq = torch::sum(x * x, /*dim=*/1, /*keepdim=*/true);
        auto denom = (1.0f + torch::sqrt(1.0f - c * x_norm_sq))
            .clamp_min(config::Constants::EPS);

        return x / denom;
    }

    torch::Tensor lorentz_to_klein_cpu(
        torch::Tensor x,
        float c
    ) {
        // 로렌츠 모델에서 클라인 모델로 변환
        float sqrtc = std::sqrt(c);

        // 로렌츠 좌표에서 시간 성분과 공간 성분 분리
        auto x0 = x.select(1, 0);
        auto xi = x.narrow(1, 1, x.size(1) - 1);

        // 클라인 좌표 계산: xi / x0
        return xi / x0.unsqueeze(1).clamp_min(config::Constants::EPS);
    }

    torch::Tensor klein_to_lorentz_cpu(
        torch::Tensor x,
        float c
    ) {
        float sqrtc = std::sqrt(c);
        auto x_norm_sq = torch::sum(x * x, /*dim=*/1, /*keepdim=*/true);
        auto x0 = 1.0f / torch::sqrt(1.0f - c * x_norm_sq)
            .clamp_min(config::Constants::EPS);
        return torch::cat({ x0, x0 * x }, /*dim=*/1);
    }

}