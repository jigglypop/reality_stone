#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <utils/common_defs.h>
#include <utils/cuda_utils.h>
#include <utils/numeric.h>
#include <maps/exp_map.h>
#include <config/constant.h>

namespace utils = hyper_butterfly::utils;
namespace config = hyper_butterfly::config;

namespace hyper_butterfly::maps {
    template <typename scalar_t>
    __global__ void exp_map_forward_kernel(
        const scalar_t* __restrict__ v,
        scalar_t* __restrict__ out,
        float c, int B, int D) {
        extern __shared__ float sdata[];
        float* s_norm2 = sdata;
        if (threadIdx.x == 0) {
            s_norm2[0] = 0.f;
        }
        const int bid = blockIdx.x;
        const int tid = threadIdx.x;
        const int stride = blockDim.x;
        __syncthreads();
        const scalar_t* vb = v + bid * D;
        scalar_t* yb = out + bid * D;
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
        for (int i = tid; i < D; i += stride) {
            yb[i] = factor * vb[i];
        }
    }
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
        float df_du = (u * sech2 - tanhu) / (u * u);
        float df_dn = df_du * sqrtf(c);
        float vdotgy = s_vg[0];
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
    torch::Tensor exp_map_backward_cuda(
        torch::Tensor v,
        torch::Tensor grad_y,
        float c) {
        utils::check_cuda_tensor(v);
        utils::check_cuda_tensor(grad_y);
        int B = v.size(0), D = v.size(1);
        auto grad_v = torch::zeros_like(v);
        int threads = std::min(D, 1024);
        int shbytes = 2 * sizeof(float);
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
