// core/csrc/geometry/base.h
#pragma once
#include <torch/extension.h>

using namespace torch;

namespace hyper_butterfly {
// 공통 상수 정의
namespace geometry {
static constexpr float EPS = 1e-6f;                 // 작은 값
static constexpr float MAX_NORM_TANH = 1.0f - EPS;  // tanh의 최대 입력값
static constexpr float MAX_TANH_INPUT = 1.0f - EPS; // tanh의 최대 입력값
// 리만 기하학에 대한 기본 연산 인터페이스
class RiemannianGeometry
{
public:
    // 로그 맵 (접공간으로 매핑)
    virtual torch::Tensor log_map(const torch::Tensor& x, float c) = 0;
    // 지수 맵 (다양체로 매핑)
    virtual torch::Tensor exp_map(const torch::Tensor& v, float c) = 0;
    // 가상 소멸자
    virtual ~RiemannianGeometry() {}
};
} // namespace geometry
} // namespace hyper_butterfly