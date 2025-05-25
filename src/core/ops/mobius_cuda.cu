#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <torch/extension.h>
#include <cmath>
#include <utils/cuda_utils.h>
#include <ops/mobius.h>
#include <config/constant.h>
#include <c10/util/Half.h> 

namespace config = reality_stone::config;
namespace utils = reality_stone::utils;

namespace reality_stone::ops {
    template <typename scalar_t>
    __global__ void mobius_add_kernel(
        const scalar_t* __restrict__ u,
        const scalar_t* __restrict__ v,
        scalar_t* __restrict__ out,
        int B, int D, float c
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= B) return;

        const scalar_t* up = u + idx * D;
        const scalar_t* vp = v + idx * D;
        scalar_t* yp = out + idx * D;

        float u2 = 0, v2 = 0, uv = 0;
        for (int j = 0; j < D; ++j) {
            float uu = up[j], vv = vp[j];
            u2 += uu * uu;
            v2 += vv * vv;
            uv += uu * vv;
        }
        float c2 = c * c;
        float denom = 1 + 2 * c * uv + c2 * u2 * v2;
        denom = fmaxf(denom, config::Constants::MIN_DENOMINATOR);
        for (int j = 0; j < D; ++j) {
            float uu = up[j], vv = vp[j];
            float nu = (1 + 2 * c * uv + c * v2) * uu;
            float nv = (1 - c * u2) * vv;
            yp[j] = (nu + nv) / denom;
        }
    }

    template <typename scalar_t>
    __global__ void mobius_scalar_kernel(
        const scalar_t* __restrict__ up,
        scalar_t* __restrict__ out,
        int B, int D, float c, float r
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= B) return;
        const scalar_t* u0 = up + idx * D;
        float norm2 = 0;
        for (int j = 0; j < D; ++j) {
            float uu = u0[j];
            norm2 += uu * uu;
        }
        float norm = sqrtf(fmaxf(norm2, config::Constants::EPS));
        float sqrtc = sqrtf(c);
        float scn = fminf(fmaxf(sqrtc * norm, config::Constants::EPS),
            1.0f - config::Constants::BOUNDARY_EPS);
        float alpha = atanhf(sqrtc * norm);
        float beta = tanhf(r * alpha);
        float scale = beta / (sqrtc * norm);
        scalar_t* y0 = out + idx * D;
        for (int j = 0; j < D; ++j) {
            y0[j] = scale * u0[j];
        }
    }


    torch::Tensor mobius_add_cuda(
        torch::Tensor u,
        torch::Tensor v,
        float c
    ) {
        utils::check_cuda_tensor(u);
        utils::check_cuda_tensor(v);
        int B = u.size(0), D = u.size(1);
        auto out = torch::empty_like(u);
        int threads = 256;
        int blocks = (B + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "mobius_add_cuda", [&] {
            mobius_add_kernel<scalar_t><<<blocks, threads>>> (
                u.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                B, D, c
                );
            });
        utils::check_cuda_error();
        return out;
    }

    torch::Tensor mobius_scalar_cuda(
        torch::Tensor u,
        float c,
        float r
    ) {
        utils::check_cuda_tensor(u);
        int B = u.size(0), D = u.size(1);
        auto out = torch::empty_like(u);
        int threads = 256;
        int blocks = (B + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "mobius_scalar_cuda", [&] {
            mobius_scalar_kernel<scalar_t><<<blocks, threads>>> (
                u.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                B, D, c, r
                );
            });
        utils::check_cuda_error();
        return out;
    }

}
