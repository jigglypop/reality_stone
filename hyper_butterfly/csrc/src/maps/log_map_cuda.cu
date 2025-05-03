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