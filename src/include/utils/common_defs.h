#pragma once
#include <torch/extension.h>
#include <cmath>
#include <vector>
namespace reality_stone::utils {
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