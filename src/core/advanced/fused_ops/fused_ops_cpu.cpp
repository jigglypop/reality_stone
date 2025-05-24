#include <torch/extension.h>
#include <ATen/ATen.h>
#include <advanced/fused_ops/fused_ops.h>
#include <ops/mobius.h>
#include <vector>
#include <cmath>
#include <chrono>

namespace ops = reality_stone::ops;

namespace reality_stone::advanced {

// 함수들은 CUDA 파일에서 구현됨 - CPU/CUDA 공통 구현
// 중복 정의 에러 방지를 위해 이 파일은 비움

} // namespace reality_stone::advanced 