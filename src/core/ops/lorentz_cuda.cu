#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <torch/extension.h>
#include <ops/lorentz.h>
#include <utils/cuda_utils.h>
#include <config/constant.h>

namespace config = reality_stone::config;
namespace utils = reality_stone::utils;

namespace reality_stone::ops {

    template <typename scalar_t>
    __global__ void lorentz_inner_kernel(
        const scalar_t* __restrict__ u,
        const scalar_t* __restrict__ v,
        scalar_t* __restrict__ result,
        int B, int D
    ) {
        int bid = blockIdx.x;
        if (bid >= B) return;

        const scalar_t* u_bid = u + bid * D;
        const scalar_t* v_bid = v + bid * D;
        scalar_t inner = u_bid[0] * v_bid[0];  // 시간 성분

        // 공간 성분
        for (int d = 1; d < D; ++d) {
            inner -= u_bid[d] * v_bid[d];
        }

        result[bid] = inner;
    }

    template <typename scalar_t>
    __global__ void lorentz_distance_kernel(
        const scalar_t* __restrict__ u,
        const scalar_t* __restrict__ v,
        scalar_t* __restrict__ result,
        int B, int D, float c
    ) {
        int bid = blockIdx.x;
        if (bid >= B) return;

        const scalar_t* u_bid = u + bid * D;
        const scalar_t* v_bid = v + bid * D;

        // 민코프스키 내적
        scalar_t inner = u_bid[0] * v_bid[0];
        for (int d = 1; d < D; ++d) {
            inner -= u_bid[d] * v_bid[d];
        }

        // inner product는 -1보다 작아야 함 (로렌츠 공간에서)
        inner = fmaxf(-inner, 1.0f + config::Constants::EPS);

        // 거리 계산: acosh(-inner) / sqrt(c)
        result[bid] = acoshf(inner) / sqrtf(c);
    }

    template <typename scalar_t>
    __global__ void lorentz_add_kernel(
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

        // 민코프스키 내적 계산
        scalar_t inner = u_bid[0] * v_bid[0];
        for (int d = 1; d < D; ++d) {
            inner -= u_bid[d] * v_bid[d];
        }

        // v_perp 계산 (v + inner * u)
        scalar_t v_perp[32];  // 스택에 임시 저장 (제한된 차원 수)
        for (int d = 0; d < D; ++d) {
            v_perp[d] = v_bid[d] + inner * u_bid[d];
        }

        // v_perp의 노름 계산
        scalar_t v_perp_norm_sq = v_perp[0] * v_perp[0];
        for (int d = 1; d < D; ++d) {
            v_perp_norm_sq -= v_perp[d] * v_perp[d];
        }

        // 노름은 음수여야 함 (로렌츠 공간에서)
        scalar_t v_perp_norm = sqrtf(fmaxf(-v_perp_norm_sq, config::Constants::EPS));

        // 측지선 계산
        scalar_t cosh_theta = coshf(v_perp_norm);
        scalar_t sinh_theta = sinhf(v_perp_norm);

        for (int d = 0; d < D; ++d) {
            result_bid[d] = cosh_theta * u_bid[d] + sinh_theta * v_perp[d] / v_perp_norm;
        }
    }

    template <typename scalar_t>
    __global__ void lorentz_scalar_kernel(
        const scalar_t* __restrict__ u,
        scalar_t* __restrict__ result,
        int B, int D, float c, float r
    ) {
        int bid = blockIdx.x;
        if (bid >= B) return;

        const scalar_t* u_bid = u + bid * D;
        scalar_t* result_bid = result + bid * D;

        // 시간/공간 성분 분리
        scalar_t time_comp = u_bid[0];

        // 로렌츠 노름 계산
        scalar_t space_norm_sq = 0.0f;
        for (int d = 1; d < D; ++d) {
            space_norm_sq += u_bid[d] * u_bid[d];
        }

        scalar_t norm = sqrtf(space_norm_sq /
            fmaxf(time_comp * time_comp - 1.0f, config::Constants::EPS));

// 스케일링 계산
        scalar_t theta = atanhf(fminf(norm, 1.0f - config::Constants::BOUNDARY_EPS)) * r;
        scalar_t scale = tanhf(theta) / fmaxf(norm, config::Constants::EPS);

        // 결과 계산
        for (int d = 1; d < D; ++d) {
            result_bid[d] = u_bid[d] * scale;
        }

        // 시간 성분 재계산
        scalar_t scaled_space_norm_sq = 0.0f;
        for (int d = 1; d < D; ++d) {
            scaled_space_norm_sq += result_bid[d] * result_bid[d];
        }

        result_bid[0] = sqrtf(1.0f + scaled_space_norm_sq);
    }

    template <typename scalar_t>
    __global__ void poincare_to_lorentz_kernel(
        const scalar_t* __restrict__ x,
        scalar_t* __restrict__ result,
        int B, int D_in, float c
    ) {
        int bid = blockIdx.x;
        if (bid >= B) return;

        const scalar_t* x_bid = x + bid * D_in;
        scalar_t* result_bid = result + bid * (D_in + 1);

        // x의 노름 제곱 계산
        scalar_t x_norm_sq = 0.0f;
        for (int d = 0; d < D_in; ++d) {
            x_norm_sq += x_bid[d] * x_bid[d];
        }

        // 분모 계산
        scalar_t denom = fmaxf(1.0f - c * x_norm_sq, config::Constants::EPS);
        scalar_t sqrtc = sqrtf(c);

        // 시간 성분 계산
        result_bid[0] = (1.0f + c * x_norm_sq) / (denom * sqrtc);

        // 공간 성분 계산
        for (int d = 0; d < D_in; ++d) {
            result_bid[d + 1] = (2.0f * x_bid[d]) / (denom * sqrtc);
        }
    }

    template <typename scalar_t>
    __global__ void lorentz_to_poincare_kernel(
        const scalar_t* __restrict__ x,
        scalar_t* __restrict__ result,
        int B, int D_in, float c
    ) {
        int bid = blockIdx.x;
        if (bid >= B) return;

        const scalar_t* x_bid = x + bid * D_in;
        scalar_t* result_bid = result + bid * (D_in - 1);

        scalar_t sqrtc = sqrtf(c);
        scalar_t x0 = x_bid[0] * sqrtc;

        // 분모 계산
        scalar_t denom = fmaxf(x0 + 1.0f, config::Constants::EPS);

        // 푸앵카레 좌표 계산
        for (int d = 0; d < D_in - 1; ++d) {
            result_bid[d] = (x_bid[d + 1] * sqrtc) / denom;
        }
    }

    torch::Tensor lorentz_inner_cuda(
        torch::Tensor u,
        torch::Tensor v
    ) {
        utils::check_cuda_tensor(u);
        utils::check_cuda_tensor(v);

        int B = u.size(0), D = u.size(1);
        auto result = torch::empty({ B }, u.options());

        int threads = 256;
        int blocks = (B + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "lorentz_inner_cuda", [&] {
            lorentz_inner_kernel<scalar_t> << <blocks, 1 >> > (
                u.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(),
                B, D
                );
            });

        utils::check_cuda_error();
        return result.unsqueeze(1);
    }

    torch::Tensor lorentz_distance_cuda(
        torch::Tensor u,
        torch::Tensor v,
        float c
    ) {
        utils::check_cuda_tensor(u);
        utils::check_cuda_tensor(v);

        int B = u.size(0), D = u.size(1);
        auto result = torch::empty({ B }, u.options());

        int threads = 256;
        int blocks = (B + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "lorentz_distance_cuda", [&] {
            lorentz_distance_kernel<scalar_t> << <blocks, 1 >> > (
                u.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(),
                B, D, c
                );
            });

        utils::check_cuda_error();
        return result.unsqueeze(1);
    }

    torch::Tensor lorentz_add_cuda(
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

        AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "lorentz_add_cuda", [&] {
            lorentz_add_kernel<scalar_t> << <blocks, 1 >> > (
                u.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(),
                B, D, c
                );
            });

        utils::check_cuda_error();
        return result;
    }

    torch::Tensor lorentz_scalar_cuda(
        torch::Tensor u,
        float c,
        float r
    ) {
        utils::check_cuda_tensor(u);

        int B = u.size(0), D = u.size(1);
        auto result = torch::empty_like(u);

        int threads = 256;
        int blocks = (B + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "lorentz_scalar_cuda", [&] {
            lorentz_scalar_kernel<scalar_t> << <blocks, 1 >> > (
                u.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(),
                B, D, c, r
                );
            });

        utils::check_cuda_error();
        return result;
    }

    torch::Tensor poincare_to_lorentz_cuda(
        torch::Tensor x,
        float c
    ) {
        utils::check_cuda_tensor(x);

        int B = x.size(0), D_in = x.size(1);
        auto options = x.options();
        auto result = torch::empty({ B, D_in + 1 }, options);

        int threads = 256;
        int blocks = (B + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "poincare_to_lorentz_cuda", [&] {
            poincare_to_lorentz_kernel<scalar_t> << <blocks, 1 >> > (
                x.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(),
                B, D_in, c
                );
            });

        utils::check_cuda_error();
        return result;
    }

    torch::Tensor lorentz_to_poincare_cuda(
        torch::Tensor x,
        float c
    ) {
        utils::check_cuda_tensor(x);

        int B = x.size(0), D_in = x.size(1);
        auto options = x.options();
        auto result = torch::empty({ B, D_in - 1 }, options);

        int threads = 256;
        int blocks = (B + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "lorentz_to_poincare_cuda", [&] {
            lorentz_to_poincare_kernel<scalar_t> << <blocks, 1 >> > (
                x.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(),
                B, D_in, c
                );
            });
        utils::check_cuda_error();
        return result;
    }

} // namespace reality_stone::ops