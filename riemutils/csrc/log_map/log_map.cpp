#include <torch/extension.h>
#include <cmath>
#include <vector>
#include "hyper_butterfly.h"

// CPU에서의 로그 맵 구현 (클램핑 적용)
torch::Tensor log_map_cpu(torch::Tensor x, float c)
{
    // 최소 norm과 분모 안전화를 위한 EPS
    static constexpr float EPS = 1e-6f;
    // 1) 각 배치별 L2 norm 계산 및 최소값 클램핑
    auto norm = torch::norm(x, 2, 1, true).clamp(EPS);
    float sqrt_c = std::sqrt(c);
    // 2) atanh 인자 클램핑: [EPS, 1 - 1e-6]
    auto scn = (sqrt_c * norm).clamp(EPS, 1.0f - 1e-6f);
    // 3) 분모 안전화
    auto denom = scn + EPS;
    // 4) atanh 계산
    auto numer = torch::atanh(scn);
    auto factor = numer / denom;
    // 5) 최종 스케일 적용
    return factor * x;
}
