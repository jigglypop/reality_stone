#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <torch/extension.h>
#include <ops/klein.h>
#include <layers/klein.h>
#include <utils/cuda_utils.h>
#include <config/constant.h>

namespace ops = reality_stone::ops;
namespace utils = reality_stone::utils;
namespace config = reality_stone::config;

namespace reality_stone::layers {

    template <typename scalar_t>
    __global__ void klein_forward_kernel(
        const scalar_t* __restrict__ u,
        const scalar_t* __restrict__ v,
        scalar_t* __restrict__ result,
        int B, int D, float c, float t
    ) {
        int bid = blockIdx.x;
        if (bid >= B) return;

        const scalar_t* u_bid = u + bid * D;
        const scalar_t* v_bid = v + bid * D;
        scalar_t* result_bid = result + bid * D;

        // 1. 직선 세그먼트 계산 ((1-t)*u + t*v)
        scalar_t direct_interpolation[32];  // 스택에 임시 저장
        scalar_t norm_sq = 0.0f;

        for (int d = 0; d < D; ++d) {
            direct_interpolation[d] = (1.0f - t) * u_bid[d] + t * v_bid[d];
            norm_sq += direct_interpolation[d] * direct_interpolation[d];
        }

        // 2. 정규화 (클라인 모델 경계 내부로 유지)
        scalar_t max_norm_sq = 1.0f / c - 1e-6f;  // 경계에서 약간 안쪽으로
        scalar_t scale = 1.0f;

        if (norm_sq > max_norm_sq && norm_sq > config::Constants::EPS) {
            scale = sqrtf(max_norm_sq / norm_sq);
        }

        // 3. 결과 계산
        for (int d = 0; d < D; ++d) {
            result_bid[d] = direct_interpolation[d] * scale;
        }
    }

    torch::Tensor klein_forward_cuda(torch::Tensor u, torch::Tensor v, float c, float t) {
        utils::check_cuda_tensor(u);
        utils::check_cuda_tensor(v);

        int B = u.size(0), D = u.size(1);
        auto result = torch::empty_like(u);

        int threads = 256;
        int blocks = (B + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "klein_forward_cuda", [&] {
            klein_forward_kernel<scalar_t> << <blocks, 1 >> > (
                u.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(),
                B, D, c, t
                );
            });

        utils::check_cuda_error();
        return result;
    }
}