#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <torch/extension.h>
#include <ops/lorentz.h>
#include <layers/lorentz.h>
#include <utils/cuda_utils.h>
#include <config/constant.h>

namespace ops = reality_stone::ops;
namespace utils = reality_stone::utils;
namespace config = reality_stone::config;

namespace reality_stone::layers {

    template <typename scalar_t>
    __global__ void lorentz_forward_kernel(
        const scalar_t* __restrict__ u,
        const scalar_t* __restrict__ v,
        scalar_t* __restrict__ result,
        int B, int D, float c, float t
    ) {
        int bid = blockIdx.x;
        if (bid >= B) return;
        
        // 차원 안전성 체크
        if (D > config::Constants::MAX_LORENTZ_DIM) {
            // 차원이 너무 크면 단순 선형 보간으로 fallback
            const scalar_t* u_bid = u + bid * D;
            const scalar_t* v_bid = v + bid * D;
            scalar_t* result_bid = result + bid * D;
            
            for (int d = 0; d < D; ++d) {
                result_bid[d] = (1.0f - t) * u_bid[d] + t * v_bid[d];
            }
            return;
        }

        const scalar_t* u_bid = u + bid * D;
        const scalar_t* v_bid = v + bid * D;
        scalar_t* result_bid = result + bid * D;
        
        // Lorentz 내적 계산
        scalar_t inner = u_bid[0] * v_bid[0];
        for (int d = 1; d < D; ++d) {
            inner -= u_bid[d] * v_bid[d];
        }
        inner = fmaxf(-inner, 1.0f + config::Constants::EPS);
        
        // 하이퍼볼릭 거리
        scalar_t dist = acoshf(inner) / sqrtf(c);
        
        // v의 수직 성분 계산 (안전한 크기 배열 사용)
        scalar_t v_perp[config::Constants::MAX_LORENTZ_DIM];
        for (int d = 0; d < D; ++d) {
            v_perp[d] = v_bid[d] + inner * u_bid[d];
        }
        
        // v_perp의 Lorentz 노름 계산
        scalar_t v_perp_norm_sq = v_perp[0] * v_perp[0];
        for (int d = 1; d < D; ++d) {
            v_perp_norm_sq -= v_perp[d] * v_perp[d];
        }
        scalar_t v_perp_norm = sqrtf(fmaxf(-v_perp_norm_sq, 1e-8f));
        
        // 하이퍼볼릭 함수들
        scalar_t cosh_dist_t = coshf(dist * t);
        scalar_t sinh_dist_t = sinhf(dist * t);
        
        // 결과 계산
        for (int d = 0; d < D; ++d) {
            result_bid[d] = cosh_dist_t * u_bid[d] + sinh_dist_t * v_perp[d] / v_perp_norm;
        }
    }

    torch::Tensor lorentz_forward_cuda(torch::Tensor u, torch::Tensor v, float c, float t) {
        utils::check_cuda_tensor(u);
        utils::check_cuda_tensor(v);
        
        int B = u.size(0), D = u.size(1);
        
        // 차원 제한 체크
        TORCH_CHECK(D <= config::Constants::MAX_LORENTZ_DIM, 
                   "Dimension ", D, " exceeds maximum supported dimension ", 
                   config::Constants::MAX_LORENTZ_DIM, " for Lorentz model");
        
        auto result = torch::empty_like(u);
        int threads = 256;
        int blocks = (B + threads - 1) / threads;
        
        AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "lorentz_forward_cuda", [&] {
            lorentz_forward_kernel<scalar_t><<<blocks, 1>>>(
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