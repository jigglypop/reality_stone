### `__init__.py`
```cpp
import torch
from torch.autograd import Function
import math

from ._C import (
    log_map_cpu,
    exp_map_cpu,
    poincare_forward_cpu,
)
_has_cuda = False
if torch.cuda.is_available():
    try:
        from ._C import (
            log_map_cuda,
            exp_map_cuda,
            log_map_forward_cuda,
            exp_map_forward_cuda,
            poincare_forward_cuda,
            poincare_backward_cuda
        )
        _has_cuda = True
    except ImportError:
        _has_cuda = False

from .python.maps import log_map, exp_map
from .python.layers import HyperButterflyFunction

def hyper_butterfly(x: torch.Tensor, params: torch.Tensor, c: float, L: int):
    return HyperButterflyFunction.apply(x, params, c, L)

```

### `csrc\extension.cpp`
```cpp
#include <torch/extension.h>
#include <hyper_butterfly/utils/common_defs.h>
#include <hyper_butterfly/utils/cuda_utils.h>
#include <hyper_butterfly/maps/log_map.h>
#include <hyper_butterfly/maps/exp_map.h>
#include <hyper_butterfly/ops/butterfly.h>
#include <hyper_butterfly/manifolds/poincare.h>

namespace utils = hyper_butterfly::utils;
namespace maps = hyper_butterfly::maps;
namespace ops = hyper_butterfly::ops;
namespace manifolds = hyper_butterfly::manifolds;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // CPU exports - 포인터 형식으로 함수 참조
    m.def("log_map_cpu", &maps::log_map_cpu, "Log map origin (CPU)");
    m.def("exp_map_cpu", &maps::exp_map_cpu, "Exp map origin (CPU)");
    // 순전파
    m.def("poincare_forward_cpu", &manifolds::poincare_forward_cpu, "poincare_forward_cpu (CPU)");
#ifdef WITH_CUDA
    // CUDA exports - 포인터 형식으로 함수 참조
    m.def("log_map_forward_cuda", &maps::log_map_forward_cuda, "Log map origin (CUDA)");
    m.def("log_map_backward_cuda", &maps::log_map_backward_cuda, "Log map origin (CUDA)");
    m.def("exp_map_forward_cuda", &maps::exp_map_forward_cuda, "Exp map origin (CUDA)");
    m.def("exp_map_backward_cuda", &maps::exp_map_backward_cuda, "Exp map origin (CUDA)");
    // CUDA forward
    m.def("poincare_forward_cuda", &manifolds::poincare_forward_cuda, "poincare_forward_cuda (CUDA)");
    m.def("poincare_backward_cuda", &manifolds::poincare_backward_cuda, "poincare_backward_cuda (CUDA)");
#endif
}
```

### `csrc\include\hyper_butterfly\config\constant.h`
```cpp
#pragma once
#include <cmath>
#include <algorithm>

namespace hyper_butterfly {
namespace config {
// constants for numerical stability in hyperbolic geometry
struct Constants {
    // normal epslion (for numerical stability)
    static constexpr float EPS = 1e-6f;
    // tanh epslion (for numerical stability)
    static constexpr float BOUNDARY_EPS = 1e-6f;
    // tanh maximum value (for numerical stability)
    static constexpr float MAX_TANH_ARG = 15.0f;
    // tanh minimum value (for numerical stability)
    static constexpr float MIN_DENOMINATOR = 1e-8f;
    // log/exp map (for numerical stability)
    static constexpr float SAFE_LOGEXP_BOUNDARY = 1.0f - 1e-6f;
    // NaN value (for numerical stability)
    static constexpr float INF = std::numeric_limits<float>::infinity();
};

}
}
```

### `csrc\include\hyper_butterfly\manifolds\poincare.h`
```cpp
#pragma once
#include <torch/extension.h>
#include <hyper_butterfly/utils/common_defs.h>

namespace hyper_butterfly {
namespace manifolds {
std::vector<torch::Tensor> poincare_forward_cpu(
    torch::Tensor x,
    torch::Tensor params,
    torch::Tensor unused,
    float c,
    int L
);
#ifdef WITH_CUDA
std::vector<torch::Tensor> poincare_forward_cuda(
    torch::Tensor x,
    torch::Tensor params,
    torch::Tensor unused,
    float c,
    int L
);
std::vector<torch::Tensor> poincare_backward_cuda(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor params,
    float c,
    int L
);
#endif
}
}
```

### `csrc\include\hyper_butterfly\maps\exp_map.h`
```cpp
#pragma once
#include <torch/extension.h>
#include <cuda.h>

namespace hyper_butterfly {
namespace maps {

torch::Tensor exp_map_cpu(torch::Tensor v, float c);
#if defined(WITH_CUDA) 
torch::Tensor exp_map_forward_cuda(torch::Tensor v, float c);
torch::Tensor exp_map_backward_cuda(torch::Tensor v, torch::Tensor grad_y, float c);
#endif
}

}

```

### `csrc\include\hyper_butterfly\maps\log_map.h`
```cpp
#pragma once
#include <torch/extension.h>
#include <cuda.h>

namespace hyper_butterfly {
namespace maps {

torch::Tensor log_map_cpu(torch::Tensor x, float c);
#if defined(WITH_CUDA)
torch::Tensor log_map_forward_cuda(torch::Tensor x, float c);
torch::Tensor log_map_backward_cuda(torch::Tensor grad_y, torch::Tensor x, float c);
#endif
}

}

```

### `csrc\include\hyper_butterfly\ops\butterfly.h`
```cpp
#pragma once
#include <torch/extension.h>
#include <hyper_butterfly/utils/common_defs.h>

namespace hyper_butterfly {
namespace ops {
torch::Tensor butterfly_forward_cpu(
    torch::Tensor input,
    torch::Tensor params,
    int layer_idx,
    int batch_size,
    int dim);

#ifdef WITH_CUDA
torch::Tensor butterfly_forward_cuda(
    torch::Tensor input,
    torch::Tensor params,
    int layer_idx,
    int batch_size,
    int dim);

std::vector<torch::Tensor> butterfly_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor input,
    torch::Tensor params,
    int layer_idx);
#endif
}
}
```

### `csrc\include\hyper_butterfly\utils\common_defs.h`
```cpp
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
```

### `csrc\include\hyper_butterfly\utils\cuda_utils.h`
```cpp
#pragma once
#include <torch/extension.h>
#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace hyper_butterfly {
namespace utils {

// CUDA error check function
inline void check_cuda_error() {
#ifdef WITH_CUDA
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
    "CUDA error: ", cudaGetErrorString(err));
#endif
}
// CUDA tensor check function
inline void check_cuda_tensor(const torch::Tensor& x) {
  TORCH_CHECK(x.device().is_cuda(), "tensor must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "tensor must be contiguous");
}
// device memory allocation
template <typename T>
inline T* cuda_malloc(size_t n) {
  T* ptr = nullptr;
#ifdef WITH_CUDA
  cudaMalloc(&ptr, n * sizeof(T));
#endif
  return ptr;
}
// device memory deallocation
template <typename T>
inline void cuda_free(T* ptr) {
#ifdef WITH_CUDA
  if (ptr) cudaFree(ptr);
#endif
}

}
}
```

### `csrc\include\hyper_butterfly\utils\numeric.h`
```cpp
#pragma once
#include <cmath>
#include <algorithm>
#include <hyper_butterfly/config/constant.h>

namespace config = hyper_butterfly::config;

namespace hyper_butterfly {
namespace utils {
// CPU용 atanh 구현
inline float atanh(float x) {
    // 경계 클램핑
    x = std::max(-1.0f + config::Constants::BOUNDARY_EPS,
        std::min(x, 1.0f - config::Constants::BOUNDARY_EPS));
    return 0.5f * std::log((1.0f + x) / (1.0f - x));
}
// CUDA용 atanh 구현
#ifdef __CUDACC__
__device__ __forceinline__ float atanh_device(float x) {
    // 경계 클램핑
    x = fmaxf(-1.0f + config::Constants::BOUNDARY_EPS,
        fminf(x, 1.0f - config::Constants::BOUNDARY_EPS));
    return 0.5f * logf((1.0f + x) / (1.0f - x));
}
#endif
// 컨텍스트 태그 타입
struct CPUContext {};
struct CUDAContext {};
}

} 
```

### `csrc\src\mainfolds\poincare_cpu.cpp`
```cpp
#include <torch/extension.h>
#include <cmath>
#include <vector>
#include <hyper_butterfly/utils/common_defs.h>
#include <hyper_butterfly/utils/cuda_utils.h>
#include <hyper_butterfly/maps/log_map.h>
#include <hyper_butterfly/maps/exp_map.h>
#include <hyper_butterfly/ops/butterfly.h>

namespace utils = hyper_butterfly::utils;
namespace maps = hyper_butterfly::maps;
namespace ops = hyper_butterfly::ops;

namespace hyper_butterfly {
namespace manifolds {
std::vector<torch::Tensor> poincare_forward_cpu(
    torch::Tensor x,
    torch::Tensor params,
    torch::Tensor /*args*/,
    float c,
    int L) {
    auto u = maps::log_map_cpu(x, c);
    auto v = u;
    int batch_size = x.size(0);
    int dim = x.size(1);
    for (int l = 0; l < L; ++l) {
        int layer_idx = l % int(std::log2(dim));
        v = ops::butterfly_forward_cpu(v, params, layer_idx, batch_size, dim);
    }
    auto y = maps::exp_map_cpu(v, c);
    return { y, u, v };
}
}
}
```

### `csrc\src\mainfolds\poincare_cuda.cu`
```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <hyper_butterfly/utils/common_defs.h>
#include <hyper_butterfly/utils/cuda_utils.h>
#include <hyper_butterfly/maps/exp_map.h>
#include <hyper_butterfly/maps/log_map.h>
#include <hyper_butterfly/ops/butterfly.h>
#include <hyper_butterfly/manifolds/poincare.h>

namespace utils = hyper_butterfly::utils;
namespace maps = hyper_butterfly::maps;
namespace manifolds = hyper_butterfly::manifolds;

namespace hyper_butterfly {
namespace manifolds {
std::vector<torch::Tensor> poincare_forward_cuda(
  torch::Tensor x,
  torch::Tensor params,
  torch::Tensor unused,
  float c,
  int L) {
  utils::check_cuda_tensor(x);
  utils::check_cuda_tensor(params);
  int B = x.size(0), D = x.size(1);
  int D_padded = utils::next_pow2(D);
  // 1: Pad input if needed
  torch::Tensor x_padded;
  if (D_padded > D) {
    x_padded = torch::zeros({ B, D_padded }, x.options());
    x_padded.narrow(1, 0, D).copy_(x);
  }
  else {
    x_padded = x;
  }
  // 2: Log map
  torch::Tensor u = maps::log_map_forward_cuda(x_padded, c);
  // 3: Apply butterfly transforms
  torch::Tensor v = u.clone();
  for (int l = 0; l < L; l++) {
    int layer_idx = l % int(std::log2(D_padded));
    v = ops::butterfly_forward_cpu(v, params, layer_idx, B, D_padded);
  }
  // 4: Exp map
  torch::Tensor y_padded = maps::exp_map_forward_cuda(v, c);
  // 5: Slice to original dimension if needed
  torch::Tensor y = (D_padded > D) ? y_padded.narrow(1, 0, D) : y_padded;
  return { y, u, v };
}

std::vector<torch::Tensor> poincare_backward_cuda(
  torch::Tensor grad_y,
  torch::Tensor x,
  torch::Tensor params,
  float c,
  int L) {
  utils::check_cuda_tensor(grad_y);
  utils::check_cuda_tensor(x);
  utils::check_cuda_tensor(params);

  int B = x.size(0), D = x.size(1);
  int D_padded = utils::next_pow2(D);
  // 1: Pad input if needed
  torch::Tensor x_padded, grad_y_padded;
  if (D_padded > D) {
    x_padded = torch::zeros({ B, D_padded }, x.options());
    x_padded.narrow(1, 0, D).copy_(x);
    grad_y_padded = torch::zeros({ B, D_padded }, grad_y.options());
    grad_y_padded.narrow(1, 0, D).copy_(grad_y);
  }
  else {
    x_padded = x;
    grad_y_padded = grad_y;
  }
  // 2: Forward pass to get intermediate results
  torch::Tensor u = maps::log_map_forward_cuda(x_padded, c);
  // Apply butterfly transforms (forward)
  std::vector<torch::Tensor> intermediates;
  intermediates.push_back(u);
  torch::Tensor v = u.clone();
  for (int l = 0; l < L; l++) {
    int layer_idx = l % int(std::log2(D_padded));
    v = ops::butterfly_forward_cuda(v, params, layer_idx, B, D_padded);
    intermediates.push_back(v);
  }
  // Final forward result
  torch::Tensor y_padded = maps::exp_map_forward_cuda(v, c);
  // 3: Backward pass
  // torch::Tensor grad_v = torch::zeros_like(v);
  int threads = std::min(D_padded, 1024);
  int shbytes = 2 * sizeof(float);
  auto grad_v = maps::exp_map_backward_cuda(v, grad_y_padded, c);
  // Backward through butterfly layers
  auto grad_params = torch::zeros_like(params);
  auto grad_u = torch::zeros_like(u);
  // Final layer's grad_out is grad_v
  torch::Tensor grad_curr = grad_v;
  // Backward through butterfly layers (in reverse order)
  for (int l = L - 1; l >= 0; l--) {
    int layer_idx = l % int(std::log2(D_padded));
    torch::Tensor input = intermediates[l];
    // Butterfly backward
    auto result = ops::butterfly_backward_cuda(
      grad_curr, input, params, layer_idx);
    torch::Tensor grad_input = result[0];
    torch::Tensor layer_grad_params = result[1];
    // Accumulate parameter gradients
    int p_offset = 0;
    for (int i = 0; i < layer_idx; i++) {
      int block_size = 1 << i;
      p_offset += 2 * (D_padded / (2 * block_size));
    }
    int p_size = 2 * (D_padded / (2 * (1 << layer_idx)));
    grad_params.narrow(0, p_offset, p_size).add_(layer_grad_params.narrow(0, p_offset, p_size));
    // Update grad for next layer
    grad_curr = grad_input;
  }
  grad_u = grad_curr;
  torch::Tensor grad_x_padded = torch::zeros_like(x_padded);
  grad_x_padded = maps::log_map_backward_cuda(x_padded, grad_u, c);
  torch::Tensor grad_x = (D_padded > D) ? grad_x_padded.narrow(1, 0, D) : grad_x_padded;
  return { grad_x, grad_params };
}
}
}
```

### `csrc\src\maps\exp_map_cpu.cpp`
```cpp
#include <torch/extension.h>
#include <cmath>
#include <hyper_butterfly/utils/common_defs.h>
#include <hyper_butterfly/maps/exp_map.h>
#include <hyper_butterfly/utils/numeric.h>
#include <hyper_butterfly/config/constant.h>

namespace config = hyper_butterfly::config;

namespace hyper_butterfly {
namespace maps {
torch::Tensor exp_map_cpu(torch::Tensor v, float c) {
    auto norm = torch::norm(v, 2, 1, true).clamp(config::Constants::EPS);
    float sqrt_c = std::sqrt(c);
    auto scn = (sqrt_c * norm).clamp(config::Constants::EPS, 10.0f);
    auto denom = scn + 1e-3f;
    auto numer = torch::tanh(scn);
    auto factor = numer / denom;
    return factor * v;
}
}
}
```

### `csrc\src\maps\exp_map_cuda.cu`
```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <hyper_butterfly/utils/common_defs.h>
#include <hyper_butterfly/utils/cuda_utils.h>
#include <hyper_butterfly/utils/numeric.h>
#include <hyper_butterfly/maps/exp_map.h>

namespace utils = hyper_butterfly::utils;
namespace config = hyper_butterfly::config;

namespace hyper_butterfly {
namespace maps {
// exp  y = tanh(√c‖v‖)/(√c‖v‖) * v
template <typename scalar_t>
__global__ void exp_map_forward_kernel(
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    float c, int B, int D) {
    extern __shared__ float sdata[];
    float* s_norm2 = sdata;  // shared[0]
    if (threadIdx.x == 0) {
        s_norm2[0] = 0.f;
    }
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    __syncthreads();
    const scalar_t* vb = v + bid * D;
    scalar_t* yb = out + bid * D;
    // 1) ||v||^2 reduction
    float local = 0.f;
    for (int i = tid; i < D; i += stride) {
        float w = vb[i];
        local += w * w;
    }
    for (int off = warpSize / 2; off > 0; off >>= 1) {
        local += __shfl_down_sync(0xffffffff, local, off);
    }
    if ((tid & (warpSize - 1)) == 0) {
        atomicAdd(s_norm2, local);
    }
    __syncthreads();
    if (tid == 0) {
        s_norm2[0] = fmaxf(s_norm2[0], config::Constants::EPS);
    }
    __syncthreads();
    float u = fminf(fmaxf(sqrtf(c) * sqrtf(s_norm2[0]), 1e-6f), 10.0f);
    float factor = tanhf(u) / (u + 1e-3f);
    // 2) output
    for (int i = tid; i < D; i += stride) {
        yb[i] = factor * vb[i];
    }
}

// exp_map backward
template <typename scalar_t>
__global__ void exp_map_backward_kernel(
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ grad_y,
    scalar_t* __restrict__ grad_v,
    float c, int B, int D) {
    extern __shared__ float sdata[];
    float* s_v2 = sdata;      // [0]
    float* s_vg = sdata + 1;  // [1]
    if (threadIdx.x == 0) {
        s_v2[0] = 0.f;
        s_vg[0] = 0.f;
    }
    __syncthreads();
    int bid = blockIdx.x, tid = threadIdx.x, stride = blockDim.x;
    const scalar_t* vb = v + bid * D;
    const scalar_t* gy = grad_y + bid * D;
    // 1) ||v||^2, v·grad_y reduction
    float local_v2 = 0.f, local_vg = 0.f;
    for (int i = tid; i < D; i += stride) {
        float vv = vb[i], gyv = gy[i];
        local_v2 += vv * vv;
        local_vg += vv * gyv;
    }
    for (int off = warpSize / 2; off > 0; off >>= 1) {
        local_v2 += __shfl_down_sync(0xffffffff, local_v2, off);
        local_vg += __shfl_down_sync(0xffffffff, local_vg, off);
    }
    if ((tid & (warpSize - 1)) == 0) {
        atomicAdd(s_v2, local_v2);
        atomicAdd(s_vg, local_vg);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        s_v2[0] = fmaxf(s_v2[0], config::Constants::EPS);
    }
    __syncthreads();
    float norm = sqrtf(s_v2[0]);
    float u = sqrtf(c) * norm;
    u = fminf(fmaxf(u, 1e-6f), 10.0f);
    float tanhu = tanhf(u);
    float sech2 = 1.0f - tanhu * tanhu;
    float factor = tanhu / (u + 1e-3f);
    // d factor / d norm
    float df_du = (u * sech2 - tanhu) / (u * u);
    float df_dn = df_du * sqrtf(c);
    float vdotgy = s_vg[0];
    // 2) per-dim gradient
    for (int i = tid; i < D; i += stride) {
        float vi = vb[i], gyi = gy[i];
        grad_v[bid * D + i] = factor * gyi + (vi / norm) * (df_dn * vdotgy);
    }
}

torch::Tensor exp_map_forward_cuda(torch::Tensor v, float c) {
    utils::check_cuda_tensor(v);
    int B = v.size(0), D = v.size(1);
    auto out = torch::empty_like(v);
    int threads = std::min(D, 1024);
    int shbytes = sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "exp_map_forward_cuda", [&] {
        exp_map_forward_kernel<scalar_t> << <B, threads, shbytes >> > (
            v.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            c, B, D);
        });
    utils::check_cuda_error();
    return out;
}
// exp_map_backward_cuda: (v, grad_y, c) → grad_v 반환
torch::Tensor exp_map_backward_cuda(
    torch::Tensor v,
    torch::Tensor grad_y,
    float c) {
    // 1) 입력 검사
    utils::check_cuda_tensor(v);
    utils::check_cuda_tensor(grad_y);
    // 2) shape 뽑기
    int B = v.size(0), D = v.size(1);
    // 3) 출력 tensor 할당
    auto grad_v = torch::zeros_like(v);
    // 4) 커널 런칭 파라미터
    int threads = std::min(D, 1024);
    int shbytes = 2 * sizeof(float);  // s_v2 + s_vg
    // 5) 타입별 디스패치
    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "exp_map_backward_cuda", [&] {
        exp_map_backward_kernel<scalar_t> << <B, threads, shbytes >> > (
            v.data_ptr<scalar_t>(),
            grad_y.data_ptr<scalar_t>(),
            grad_v.data_ptr<scalar_t>(),
            c, B, D);
        });
    utils::check_cuda_error();
    return grad_v;
}

}
}

```

### `csrc\src\maps\log_map_cpu.cpp`
```cpp
#include <torch/extension.h>
#include <cmath>
#include <hyper_butterfly/utils/common_defs.h>
#include <hyper_butterfly/maps/exp_map.h>
#include <hyper_butterfly/utils/numeric.h>
#include <hyper_butterfly/config/constant.h>

namespace config = hyper_butterfly::config;

namespace hyper_butterfly {
namespace maps {

torch::Tensor log_map_cpu(torch::Tensor x, float c) {
    auto norm = torch::norm(x, 2, 1, true).clamp(config::Constants::EPS);
    float sqrt_c = std::sqrt(c);
    auto scn = (sqrt_c * norm).clamp(config::Constants::EPS, 1.0f - 1e-6f);
    auto denom = scn + config::Constants::EPS;
    auto numer = torch::atanh(scn);
    auto factor = numer / denom;
    return factor * x;
}

}
}
```

### `csrc\src\maps\log_map_cuda.cu`
```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <hyper_butterfly/utils/common_defs.h>
#include <hyper_butterfly/utils/cuda_utils.h>
#include <hyper_butterfly/utils/numeric.h>
#include <hyper_butterfly/maps/log_map.h>
#include <hyper_butterfly/config/constant.h>

namespace config = hyper_butterfly::config;
namespace utils = hyper_butterfly::utils;

namespace hyper_butterfly {
namespace maps {
// 로그 맵 forward 커널 y = atanh(√c‖x‖)/(√c‖x‖) * x
template <typename scalar_t>
__global__ void log_map_forward_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    float c, int B, int D) {
    extern __shared__ float sdata[];
    float* s_norm2 = sdata;
    if (threadIdx.x == 0)  s_norm2[0] = 0.f;
    __syncthreads();
    int bid = blockIdx.x, tid = threadIdx.x, stride = blockDim.x;
    const scalar_t* xb = x + bid * D;
    scalar_t* yb = out + bid * D;
    float local = 0.f;
    for (int i = tid; i < D; i += stride) {
        float v = xb[i];
        local += v * v;
    }
    // warp‐reduce
    for (int off = warpSize / 2; off > 0; off >>= 1) {
        local += __shfl_down_sync(0xffffffff, local, off);
    }
    if ((tid & (warpSize - 1)) == 0) {
        atomicAdd(s_norm2, local);
    }
    __syncthreads();
    // 2) clamp & factor
    if (tid == 0) {
        s_norm2[0] = fmaxf(s_norm2[0], config::Constants::EPS);
    }
    __syncthreads();
    float norm = sqrtf(s_norm2[0]);
    float u = sqrtf(c) * norm;
    u = fminf(fmaxf(u, 1e-6f), 0.999999f);
    float factor = utils::atanh_device(u) / (u + 1e-6f);
    // 3) output
    for (int i = tid; i < D; i += stride) {
        yb[i] = factor * xb[i];
    }
}

// log_map backward 커널
template <typename scalar_t>
__global__ void log_map_backward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ grad_u,
    scalar_t* __restrict__ grad_x,
    float c, int B, int D) {
    extern __shared__ float sdata[];
    float* s_x2 = sdata;
    float* s_xu = sdata + 1;
    if (threadIdx.x == 0) {
        s_x2[0] = 0.f;
        s_xu[0] = 0.f;
    }
    __syncthreads();
    int bid = blockIdx.x, tid = threadIdx.x, stride = blockDim.x;
    const scalar_t* xb = x + bid * D;
    const scalar_t* gu = grad_u + bid * D;
    // 1) ||x||^2, x·grad_u reduction
    float local_x2 = 0.f, local_xu = 0.f;
    for (int i = tid; i < D; i += stride) {
        float xi = xb[i], gui = gu[i];
        local_x2 += xi * xi;
        local_xu += xi * gui;
    }
    for (int off = warpSize / 2; off > 0; off >>= 1) {
        local_x2 += __shfl_down_sync(0xffffffff, local_x2, off);
        local_xu += __shfl_down_sync(0xffffffff, local_xu, off);
    }
    if ((tid & (warpSize - 1)) == 0) {
        atomicAdd(s_x2, local_x2);
        atomicAdd(s_xu, local_xu);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        s_x2[0] = fmaxf(s_x2[0], config::Constants::EPS);
    }
    __syncthreads();
    float norm = sqrtf(s_x2[0]);
    float u = sqrtf(c) * norm;
    u = fminf(fmaxf(u, 1e-6f), 0.999999f);
    // atanh(u) 계산
    float atanhu = 0.5f * logf((1.0f + u) / (1.0f - u));
    float factor = atanhu / (u + 1e-6f);
    float sech2 = 1.0f - u * u;
    float df_du = (u * sech2 - atanhu) / (u * u);
    float df_dn = df_du * sqrtf(c);
    float xdotg = s_xu[0];
    // 2) per-dim gradient
    for (int i = tid; i < D; i += stride) {
        float xi = xb[i], guv = gu[i];
        grad_x[bid * D + i] = factor * guv + (xi / norm) * (df_dn * xdotg);
    }
}

torch::Tensor log_map_forward_cuda(torch::Tensor x, float c) {
    utils::check_cuda_tensor(x);
    int B = x.size(0), D = x.size(1);
    auto out = torch::empty_like(x);
    int threads = std::min(D, 1024);
    int shbytes = sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "log_map_forward_cuda", [&] {
        log_map_forward_kernel<scalar_t> << <B, threads, shbytes >> > (
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            c, B, D);
        });
    utils::check_cuda_error();
    return out;
}

torch::Tensor log_map_backward_cuda(
    torch::Tensor x,
    torch::Tensor grad_u,
    float c) {
    utils::check_cuda_tensor(x);
    utils::check_cuda_tensor(grad_u);
    int B = x.size(0), D = x.size(1);
    auto grad_x = torch::zeros_like(x);
    int threads = std::min(D, 1024);
    int shbytes = 2 * sizeof(float);  // s_x2 + s_xu
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "log_map_backward_cuda", [&] {
        log_map_backward_kernel<scalar_t> << <B, threads, shbytes >> > (
            x.data_ptr<scalar_t>(),
            grad_u.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            c, B, D);
        });
    utils::check_cuda_error();
    return grad_x;
}
}
}
```

### `csrc\src\ops\butterfly_cpu.cpp`
```cpp
#include <torch/extension.h>
#include <cmath>
#include <vector>
#include <hyper_butterfly/utils/common_defs.h>
#include <hyper_butterfly/utils/cuda_utils.h>
#include <hyper_butterfly/maps/exp_map.h>
#include <hyper_butterfly/maps/log_map.h>
#include <hyper_butterfly/ops/butterfly.h>

namespace utils = hyper_butterfly::utils;
namespace maps = hyper_butterfly::maps;

namespace hyper_butterfly {
namespace ops {
torch::Tensor butterfly_forward_cpu(
    torch::Tensor input,
    torch::Tensor params,
    int layer_idx,
    int batch_size,
    int dim) {
    auto output = torch::empty_like(input);
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_forward_cpu", ([&] {
        const scalar_t* x_ptr = input.data_ptr<scalar_t>();
        scalar_t* y_ptr = output.data_ptr<scalar_t>();
        const scalar_t* p_ptr = params.data_ptr<scalar_t>();
        int block_size = 1 << layer_idx;
        int num_blocks = dim / (2 * block_size);
        for (int b = 0; b < batch_size; b++) {
            for (int f = 0; f < dim; f++) {
                int blk = (f / (2 * block_size)) % num_blocks;
                int loc = f % (2 * block_size);
                bool hi = loc >= block_size;
                int off = loc % block_size;
                int pidx = blk * 2;
                scalar_t a = p_ptr[pidx];
                scalar_t bb = p_ptr[pidx + 1];
                int base = b * dim + blk * 2 * block_size;
                scalar_t x1 = x_ptr[base + off];
                scalar_t x2 = x_ptr[base + off + block_size];
                y_ptr[b * dim + f] = hi ? (-bb * x1 + a * x2) : (a * x1 + bb * x2);
            }
        } }));
        return output;
}
}
}
```

### `csrc\src\ops\butterfly_cuda.cu`
```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <hyper_butterfly/utils/common_defs.h>
#include <hyper_butterfly/utils/cuda_utils.h>
#include <hyper_butterfly/maps/exp_map.h>
#include <hyper_butterfly/maps/log_map.h>
#include <hyper_butterfly/ops/butterfly.h>

namespace utils = hyper_butterfly::utils;
namespace maps = hyper_butterfly::maps;

namespace hyper_butterfly {
namespace ops {
template <typename scalar_t>
__global__ void butterfly_backward_kernel(
  const scalar_t* __restrict__ grad_out,
  const scalar_t* __restrict__ input,
  const scalar_t* __restrict__ params,
  scalar_t* __restrict__ grad_input,
  scalar_t* __restrict__ grad_params,
  int B, int D, int layer_idx) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int bs = 1 << layer_idx, nb = D / (2 * bs);

  while (idx < B * D) {
    int b = idx / D, f = idx % D;
    int blk = (f / (2 * bs)) % nb, loc = f % (2 * bs), off = loc % bs;
    bool high = loc >= bs;
    int pi = blk * 2;
    float a = params[pi + 0], bb = params[pi + 1];
    int base = b * D + blk * 2 * bs;
    float x1 = input[base + off], x2 = input[base + off + bs];
    float gout = grad_out[idx];

    if (!high) {
      // y = a*x1 + b*x2
      atomicAdd(&grad_input[base + off], a * gout);
      atomicAdd(&grad_input[base + off + bs], bb * gout);
      atomicAdd(&grad_params[pi + 0], x1 * gout);
      atomicAdd(&grad_params[pi + 1], x2 * gout);
    }
    else {
      // y = -b*x1 + a*x2
      atomicAdd(&grad_input[base + off], -bb * gout);
      atomicAdd(&grad_input[base + off + bs], a * gout);
      atomicAdd(&grad_params[pi + 0], x2 * gout);
      atomicAdd(&grad_params[pi + 1], -x1 * gout);
    }
    idx += stride;
  }
}

template <typename scalar_t>
__global__ void butterfly_forward_kernel(
  const scalar_t* __restrict__ input,
  scalar_t* __restrict__ output,
  const scalar_t* __restrict__ params,
  int B, int D, int layer_idx) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int bs = 1 << layer_idx;
  int nb = D / (2 * bs);

  while (idx < B * D) {
    int b = idx / D, f = idx % D;
    int blk = (f / (2 * bs)) % nb,
      loc = f % (2 * bs),
      off = loc % bs;
    bool high = loc >= bs;
    int pi = blk * 2;
    float a = params[pi + 0],
      bb = params[pi + 1];
    int base = b * D + blk * 2 * bs;
    float x1 = input[base + off],
      x2 = input[base + off + bs];
    output[idx] = high
      ? (-bb * x1 + a * x2)
      : (a * x1 + bb * x2);
    idx += stride;
  }
}

torch::Tensor butterfly_forward_cuda(
  torch::Tensor input,
  torch::Tensor params,
  int layer_idx,
  int batch_size,
  int dim) {

  auto output = torch::empty_like(input);
  dim3 grid(std::min((batch_size * dim + 511) / 512, 1024));
  dim3 block(512);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_forward_cuda", ([&] {
    ops::butterfly_forward_kernel<scalar_t> << <grid, block >> > (
      input.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      params.data_ptr<scalar_t>(),
      batch_size, dim, layer_idx);
    }));
  utils::check_cuda_error();
  return output;
}

std::vector<torch::Tensor> butterfly_backward_cuda(
  torch::Tensor grad_out,
  torch::Tensor input,
  torch::Tensor params,
  int layer_idx) {
  auto grad_input = torch::zeros_like(input);
  auto grad_params = torch::zeros_like(params);
  int batch_size = input.size(0);
  int dim = input.size(1);
  dim3 grid(std::min((batch_size * dim + 511) / 512, 1024));
  dim3 block(512);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_backward_cuda", ([&] {
    ops::butterfly_backward_kernel<scalar_t> << <grid, block >> > (
      grad_out.data_ptr<scalar_t>(),
      input.data_ptr<scalar_t>(),
      grad_input.data_ptr<scalar_t>(),
      params.data_ptr<scalar_t>(),
      grad_params.data_ptr<scalar_t>(),
      batch_size, dim, layer_idx);
    }));
  utils::check_cuda_error();
  return { grad_input, grad_params };
}
}
}
```

### `python\__init__.py`
```cpp


```

### `python\layers.py`
```cpp
# layers.py

import math
import torch
from torch.autograd import Function
from .._C import (
    poincare_forward_cpu,
    poincare_forward_cuda,
    poincare_backward_cuda,
)
from .. import _has_cuda
from .maps import log_map, exp_map

def butterfly_transform(x: torch.Tensor, params: torch.Tensor, L: int) -> torch.Tensor:
    batch, dim = x.shape
    log2_dim = int(math.log2(dim))
    out = x
    offset = 0
    for l in range(L):
        layer = l % log2_dim
        bs = 1 << layer
        nb = dim // (2 * bs)
        p = params[offset:offset + nb * 2].view(nb, 2)
        offset += nb * 2
        out = out.view(batch, nb, 2, bs)
        a = p[:, 0].view(1, nb, 1)
        b = p[:, 1].view(1, nb, 1)
        x1 = out[:, :, 0, :]
        x2 = out[:, :, 1, :]
        y1 = a * x1 + b * x2
        y2 = -b * x1 + a * x2
        out = torch.stack([y1, y2], dim=2).reshape(batch, dim)
    return out

def hyper_butterfly_py(x: torch.Tensor, params: torch.Tensor, c: float, L: int) -> torch.Tensor:
    u = log_map(x, c)
    v = butterfly_transform(u, params, L)
    y = exp_map(v, c)
    return y

class HyperButterflyFunction(Function):
    @staticmethod
    def forward(ctx, x, params, c, L):
        ctx.save_for_backward(x, params)
        ctx.c, ctx.L = c, L
        if x.is_cuda and _has_cuda:
            y, _, __ = poincare_forward_cuda(x, params, torch.empty(0,device=x.device), c, L)
        else:
            if not x.is_cuda and 'poincare_forward_cpu,' in globals():
                y, _, __ = poincare_forward_cpu(x, params, c, L)
            else:
                y = hyper_butterfly_py(x, params, c, L)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, params = ctx.saved_tensors
        c, L = ctx.c, ctx.L
        if x.is_cuda and _has_cuda:
            grad_x, grad_p = poincare_backward_cuda(
                grad_out.contiguous(), x, params, c, L
            )
            return grad_x, grad_p, None, None
        with torch.enable_grad():
            x_req = x.detach().requires_grad_()
            p_req = params.detach().requires_grad_()
            y = hyper_butterfly_py(x_req, p_req, c, L)
            gx, gp = torch.autograd.grad(y, (x_req, p_req), grad_out)
        return gx, gp, None, None
```

### `python\maps.py`
```cpp
import math
import torch
from .._C import (
    log_map_forward_cuda,
    exp_map_forward_cuda,
)
from .. import _has_cuda

def log_map(x: torch.Tensor, c: float) -> torch.Tensor:
    if x.is_cuda and _has_cuda:
        return log_map_forward_cuda(x, c)
    norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-6)
    scn = (math.sqrt(c) * norm).clamp(min=1e-6, max=1.0 - 1e-6)
    factor = torch.atanh(scn) / (scn + 1e-6)
    return factor * x

def exp_map(x: torch.Tensor, c: float) -> torch.Tensor:
    if x.is_cuda and _has_cuda:
        return exp_map_forward_cuda(x, c)
    norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-6)
    scn = (math.sqrt(c) * norm).clamp(min=1e-6, max=10.0)
    factor = torch.tanh(scn) / (scn + 1e-3)
    return factor * x
```
