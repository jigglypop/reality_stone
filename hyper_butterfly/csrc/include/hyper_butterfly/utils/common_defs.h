#pragma once
#include <torch/extension.h>
#include <cmath>
#include <vector>
namespace hyper_butterfly {
namespace utils {
inline int next_pow2(int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}
}
}