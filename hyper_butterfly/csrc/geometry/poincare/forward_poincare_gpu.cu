#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "forward_poincare.h"
#include "../../utils/common_defs.h"

use namespace std;
use namespace torch;

namespace hyper_butterfly
{
    namespace geometry
    {
        namespace poincare
        {
            // atanh 헬퍼 (clamp 포함)
            __device__ __forceinline__ float atanh_device(float x)
            {
                x = fminf(fmaxf(x, -1.0f + 1e-6f), 1.0f - 1e-6f);
                return 0.5f * logf((1.0f + x) / (1.0f - x));
            }

            // 로그 맵 커널
            template <typename scalar_t>
            __global__ void log_map_kernel(
                const scalar_t *__restrict__ x,
                scalar_t *__restrict__ out,
                float c, int B, int D)
            {

                // 블록 당 0으로 초기화
                extern __shared__ float sdata[];
                float *s_norm2 = sdata; // shared[0]
                if (threadIdx.x == 0)
                    s_norm2[0] = 0.f;
                __syncthreads();

                int bid = blockIdx.x, tid = threadIdx.x, stride = blockDim.x;
                const scalar_t *xb = x + bid * D;
                scalar_t *yb = out + bid * D;
                // 1) ||x||^2 reduction
                float local = 0.f;
                for (int i = tid; i < D; i += stride)
                {
                    float v = xb[i];
                    local += v * v;
                }
                // warp‐reduce
                for (int off = warpSize / 2; off > 0; off >>= 1)
                {
                    local += __shfl_down_sync(0xffffffff, local, off);
                }
                if ((tid & (warpSize - 1)) == 0)
                {
                    atomicAdd(s_norm2, local);
                }
                __syncthreads();

                // 2) clamp & factor
                if (tid == 0)
                {
                    s_norm2[0] = fmaxf(s_norm2[0], EPS);
                }
                __syncthreads();
                float norm = sqrtf(s_norm2[0]);
                float u = sqrtf(c) * norm;
                u = fminf(fmaxf(u, 1e-6f), 0.999999f);
                float factor = atanh_device(u) / (u + 1e-6f);
                // 3) output
                for (int i = tid; i < D; i += stride)
                {
                    yb[i] = factor * xb[i];
                }
            }

            // 지수 맵 커널
            template <typename scalar_t>
            __global__ void exp_map_kernel(
                const scalar_t *__restrict__ v,
                scalar_t *__restrict__ out,
                float c, int B, int D)
            {

                extern __shared__ float sdata[];
                float *s_norm2 = sdata; // shared[0]
                // 블록 당 0으로 초기화
                if (threadIdx.x == 0)
                {
                    s_norm2[0] = 0.f;
                }
                __syncthreads();
                int bid = blockIdx.x, tid = threadIdx.x, stride = blockDim.x;
                const scalar_t *vb = v + bid * D;
                scalar_t *yb = out + bid * D;
                // 1) ||v||^2 reduction
                float local = 0.f;
                for (int i = tid; i < D; i += stride)
                {
                    float w = vb[i];
                    local += w * w;
                }
                for (int off = warpSize / 2; off > 0; off >>= 1)
                {
                    local += __shfl_down_sync(0xffffffff, local, off);
                }
                if ((tid & (warpSize - 1)) == 0)
                {
                    atomicAdd(s_norm2, local);
                }
                __syncthreads();

                if (tid == 0)
                {
                    s_norm2[0] = fmaxf(s_norm2[0], EPS);
                }
                __syncthreads();

                float norm = sqrtf(s_norm2[0]);
                float u = sqrtf(c) * norm;
                u = fminf(fmaxf(u, 1e-6f), 10.0f);
                float tanhu = tanhf(u);
                float factor = tanhu / (u + 1e-3f);
                // 2) output
                for (int i = tid; i < D; i += stride)
                {
                    yb[i] = factor * vb[i];
                }
            }

            // CUDA 로그맵 함수 구현
            Tensor log_map_cuda(Tensor x, float c)
            {
                CHECK_CUDA_CONTIGUOUS(x);
                int B = x.size(0), D = x.size(1);
                auto out = empty_like(x);
                int threads = min(D, 1024);
                int shbytes = sizeof(float);

                AT_DISPATCH_FLOATING_TYPES(
                    x.scalar_type(), "log_map_cuda", [&]
                    { log_map_kernel<scalar_t><<<B, threads, shbytes>>>(
                          x.data_ptr<scalar_t>(),
                          out.data_ptr<scalar_t>(),
                          c, B, D); });
                CUDA_CHECK(cudaGetLastError());
                return out;
            }

            // CUDA 지수맵 함수 구현
            Tensor exp_map_cuda(Tensor v, float c)
            {
                CHECK_CUDA_CONTIGUOUS(v);
                int B = v.size(0), D = v.size(1);
                auto out = empty_like(v);
                int threads = min(D, 1024);
                int shbytes = sizeof(float);

                AT_DISPATCH_FLOATING_TYPES(
                    v.scalar_type(), "exp_map_cuda", [&]
                    { exp_map_kernel<scalar_t><<<B, threads, shbytes>>>(
                          v.data_ptr<scalar_t>(),
                          out.data_ptr<scalar_t>(),
                          c, B, D); });

                CUDA_CHECK(cudaGetLastError());
                return out;
            }

        } // namespace poincare
    } // namespace geometry
} // namespace hyper_butterfly

// Python 바인딩을 위한 함수들
Tensor log_map_cuda_export(Tensor x, float c)
{
    return hyper_butterfly::geometry::poincare::log_map_cuda(x, c);
}

Tensor exp_map_cuda_export(Tensor v, float c)
{
    return hyper_butterfly::geometry::poincare::exp_map_cuda(v, c);
}