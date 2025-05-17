#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <torch/extension.h>
#include <ops/klein.h>
#include <utils/cuda_utils.h>
#include <config/constant.h>

namespace config = reality_stone::config;
namespace utils = reality_stone::utils;

namespace reality_stone::ops {
    template <typename scalar_t>
    __global__ void klein_distance_kernel(
        const scalar_t* __restrict__ u,
        const scalar_t* __restrict__ v,
        scalar_t* __restrict__ result,
        int B, int D, float c
    ) {
        int bid = blockIdx.x;
        if (bid >= B) return;
        
        const scalar_t* u_bid = u + bid * D;
        const scalar_t* v_bid = v + bid * D;
        
        // 노름 계산
        scalar_t u_norm_sq = 0.0f, v_norm_sq = 0.0f, uv = 0.0f;
        for (int d = 0; d < D; ++d) {
            u_norm_sq += u_bid[d] * u_bid[d];
            v_norm_sq += v_bid[d] * v_bid[d];
            uv += u_bid[d] * v_bid[d];
        }
        
        // 거리 공식 계산
        scalar_t numerator = 2.0f * (u_norm_sq * v_norm_sq - uv * uv);
        scalar_t denominator = ((1.0f - c * u_norm_sq) * (1.0f - c * v_norm_sq));
        denominator = fmaxf(denominator, config::Constants::EPS);
        
        scalar_t lambda = sqrtf(numerator / denominator);
        scalar_t two_minus_lambda_sq = fmaxf(2.0f - lambda, config::Constants::EPS);
        
        result[bid] = acoshf((2.0f + lambda) / two_minus_lambda_sq) / sqrtf(c);
    }
    
    template <typename scalar_t>
    __global__ void klein_add_kernel(
        const scalar_t* __restrict__ u,
        const scalar_t* __restrict__ v,
        scalar_t* __restrict__ result,
        int B, int D, float c
    ) {
        int bid = blockIdx.x;
        if (bid >= B) return;
        
        const scalar_t* u_bid = u + bid * D;
        const scalar_t* v_bid = v + bid * D;
        scalar_t* result_bid = result + bid * D;
        
        // 노름 계산
        scalar_t u_norm_sq = 0.0f, v_norm_sq = 0.0f;
        for (int d = 0; d < D; ++d) {
            u_norm_sq += u_bid[d] * u_bid[d];
            v_norm_sq += v_bid[d] * v_bid[d];
        }
        
        scalar_t u_denom = fmaxf(1.0f - c * u_norm_sq, config::Constants::EPS);
        scalar_t v_denom = fmaxf(1.0f - c * v_norm_sq, config::Constants::EPS);
        
        // 결과 계산
        scalar_t temp_result[32];  // 임시 결과 (스택에 저장)
        scalar_t result_norm_sq = 0.0f;
        
        for (int d = 0; d < D; ++d) {
            temp_result[d] = u_bid[d] / sqrtf(u_denom) + v_bid[d] / sqrtf(v_denom);
            result_norm_sq += temp_result[d] * temp_result[d];
        }
        
        // 정규화
        scalar_t result_denom = 1.0f + sqrtf(1.0f + c * result_norm_sq);
        result_denom = fmaxf(result_denom, config::Constants::EPS);
        
        for (int d = 0; d < D; ++d) {
            result_bid[d] = temp_result[d] / result_denom;
        }
    }
    
    template <typename scalar_t>
    __global__ void klein_scalar_kernel(
        const scalar_t* __restrict__ u,
        scalar_t* __restrict__ result,
        int B, int D, float c, float r
    ) {
        int bid = blockIdx.x;
        if (bid >= B) return;
        
        const scalar_t* u_bid = u + bid * D;
        scalar_t* result_bid = result + bid * D;
        
        // 노름 계산
        scalar_t norm_sq = 0.0f;
        for (int d = 0; d < D; ++d) {
            norm_sq += u_bid[d] * u_bid[d];
        }
        
        scalar_t norm = sqrtf(fmaxf(norm_sq, config::Constants::EPS));
        
        // 스케일링
        scalar_t scaled_norm = norm * r;
        
        // 최대 유효 범위 제한
        scalar_t max_norm = 1.0f / sqrtf(c) - config::Constants::BOUNDARY_EPS;
        if (scaled_norm > max_norm) {
            scaled_norm = max_norm;
        }
        
        // 방향 벡터에 스케일링 적용
        scalar_t scale = scaled_norm / norm;
        for (int d = 0; d < D; ++d) {
            result_bid[d] = u_bid[d] * scale;
        }
    }
    
    template <typename scalar_t>
    __global__ void poincare_to_klein_kernel(
        const scalar_t* __restrict__ x,
        scalar_t* __restrict__ result,
        int B, int D, float c
    ) {
        int bid = blockIdx.x;
        if (bid >= B) return;
        
        const scalar_t* x_bid = x + bid * D;
        scalar_t* result_bid = result + bid * D;
        
        // 노름 계산
        scalar_t x_norm_sq = 0.0f;
        for (int d = 0; d < D; ++d) {
            x_norm_sq += x_bid[d] * x_bid[d];
        }
        
        // 변환 공식 적용
        scalar_t denom = fmaxf(1.0f + c * x_norm_sq, config::Constants::EPS);
        
        for (int d = 0; d < D; ++d) {
            result_bid[d] = 2.0f * x_bid[d] / denom;
        }
    }
    
    template <typename scalar_t>
    __global__ void klein_to_poincare_kernel(
        const scalar_t* __restrict__ x,
        scalar_t* __restrict__ result,
        int B, int D, float c
    ) {
        int bid = blockIdx.x;
        if (bid >= B) return;
        
        const scalar_t* x_bid = x + bid * D;
        scalar_t* result_bid = result + bid * D;
        
        // 노름 계산
        scalar_t x_norm_sq = 0.0f;
        for (int d = 0; d < D; ++d) {
            x_norm_sq += x_bid[d] * x_bid[d];
        }
        
        // 변환 공식 적용
        scalar_t denom = 1.0f + sqrtf(fmaxf(1.0f - c * x_norm_sq, config::Constants::EPS));
        denom = fmaxf(denom, config::Constants::EPS);
        
        for (int d = 0; d < D; ++d) {
            result_bid[d] = x_bid[d] / denom;
        }
    }
    
    template <typename scalar_t>
    __global__ void lorentz_to_klein_kernel(
        const scalar_t* __restrict__ x,
        scalar_t* __restrict__ result,
        int B, int D_in, float c
    ) {
        int bid = blockIdx.x;
        if (bid >= B) return;
        
        const scalar_t* x_bid = x + bid * D_in;
        scalar_t* result_bid = result + bid * (D_in - 1);
        
        // 시간 성분
        scalar_t x0 = fmaxf(x_bid[0], config::Constants::EPS);
        
        // 공간 성분 / 시간 성분
        for (int d = 0; d < D_in - 1; ++d) {
            result_bid[d] = x_bid[d + 1] / x0;
        }
    }
    
    template <typename scalar_t>
    __global__ void klein_to_lorentz_kernel(
        const scalar_t* __restrict__ x,
        scalar_t* __restrict__ result,
        int B, int D, float c
    ) {
        int bid = blockIdx.x;
        if (bid >= B) return;
        
        const scalar_t* x_bid = x + bid * D;
        scalar_t* result_bid = result + bid * (D + 1);
        
        // 노름 계산
        scalar_t x_norm_sq = 0.0f;
        for (int d = 0; d < D; ++d) {
            x_norm_sq += x_bid[d] * x_bid[d];
        }
        
        // 시간 성분 계산
        scalar_t x0 = 1.0f / sqrtf(fmaxf(1.0f - c * x_norm_sq, config::Constants::EPS));
        
        // 로렌츠 좌표 계산
        result_bid[0] = x0;
        for (int d = 0; d < D; ++d) {
            result_bid[d + 1] = x0 * x_bid[d];
        }
    }

    torch::Tensor klein_distance_cuda(
        torch::Tensor u,
        torch::Tensor v,
        float c
    ) {
        utils::check_cuda_tensor(u);
        utils::check_cuda_tensor(v);
        
        int B = u.size(0), D = u.size(1);
        auto result = torch::empty({B}, u.options());
        
        int threads = 256;
        int blocks = (B + threads - 1) / threads;
        
        AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "klein_distance_cuda", [&] {
            klein_distance_kernel<scalar_t><<<blocks, 1>>>(
                u.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(),
                B, D, c
            );
        });
        
        utils::check_cuda_error();
        return result.unsqueeze(1);
    }
    
    torch::Tensor klein_add_cuda(
        torch::Tensor u,
        torch::Tensor v,
        float c
    ) {
        utils::check_cuda_tensor(u);
        utils::check_cuda_tensor(v);
        
        int B = u.size(0), D = u.size(1);
        auto result = torch::empty_like(u);
        
        int threads = 256;
        int blocks = (B + threads - 1) / threads;
        
        AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "klein_add_cuda", [&] {
            klein_add_kernel<scalar_t><<<blocks, 1>>>(
                u.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(),
                B, D, c
            );
        });
        
        utils::check_cuda_error();
        return result;
    }
    
    torch::Tensor klein_scalar_cuda(
        torch::Tensor u,
        float c,
        float r
    ) {
        utils::check_cuda_tensor(u);
        
        int B = u.size(0), D = u.size(1);
        auto result = torch::empty_like(u);
        
        int threads = 256;
        int blocks = (B + threads - 1) / threads;
        
        AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "klein_scalar_cuda", [&] {
            klein_scalar_kernel<scalar_t><<<blocks, 1>>>(
                u.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(),
                B, D, c, r
            );
        });
        
        utils::check_cuda_error();
        return result;
    }
    
    torch::Tensor poincare_to_klein_cuda(
        torch::Tensor x, 
        float c
    ) {
        utils::check_cuda_tensor(x);
        
        int B = x.size(0), D = x.size(1);
        auto result = torch::empty_like(x);
        
        int threads = 256;
        int blocks = (B + threads - 1) / threads;
        
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "poincare_to_klein_cuda", [&] {
            poincare_to_klein_kernel<scalar_t><<<blocks, 1>>>(
                x.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(),
                B, D, c
            );
        });
        
        utils::check_cuda_error();
        return result;
    }
    
    torch::Tensor klein_to_poincare_cuda(
        torch::Tensor x, 
        float c
    ) {
        utils::check_cuda_tensor(x);
        
        int B = x.size(0), D = x.size(1);
        auto result = torch::empty_like(x);
        
        int threads = 256;
        int blocks = (B + threads - 1) / threads;
        
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "klein_to_poincare_cuda", [&] {
            klein_to_poincare_kernel<scalar_t><<<blocks, 1>>>(
                x.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(),
                B, D, c
            );
        });
        
        utils::check_cuda_error();
        return result;
    }
    
    torch::Tensor lorentz_to_klein_cuda(
        torch::Tensor x, 
        float c
    ) {
        utils::check_cuda_tensor(x);
        
        int B = x.size(0), D_in = x.size(1);
        auto result = torch::empty({B, D_in - 1}, x.options());
        
        int threads = 256;
        int blocks = (B + threads - 1) / threads;
        
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "lorentz_to_klein_cuda", [&] {
            lorentz_to_klein_kernel<scalar_t><<<blocks, 1>>>(
                x.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(),
                B, D_in, c
            );
        });
        
        utils::check_cuda_error();
        return result;
    }
    
    torch::Tensor klein_to_lorentz_cuda(
        torch::Tensor x, 
        float c
    ) {
        utils::check_cuda_tensor(x);
        
        int B = x.size(0), D = x.size(1);
        auto result = torch::empty({B, D + 1}, x.options());
        
        int threads = 256;
        int blocks = (B + threads - 1) / threads;
        
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "klein_to_lorentz_cuda", [&] {
            klein_to_lorentz_kernel<scalar_t><<<blocks, 1>>>(
                x.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(),
                B, D, c
            );
        });
        
        utils::check_cuda_error();
        return result;
    }
    
} 