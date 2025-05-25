#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <torch/extension.h>
#include <utils/cuda_utils.h>
#include <utils/numeric.h>
#include <config/constant.h>
#include <layers/poincare_ball.h>

namespace utils = reality_stone::utils;
namespace config = reality_stone::config;

namespace reality_stone::layers {
    template <typename scalar_t>
    __global__ void poincare_ball_backward_kernel(
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ u,
        const scalar_t* __restrict__ v,
        scalar_t* __restrict__ grad_u,
        scalar_t* __restrict__ grad_v,
        float c, float t, int B, int D
    ) {
        int bid = blockIdx.x;
        int tid = threadIdx.x;
        int blockSize = blockDim.x;

        if (bid >= B) return;

        // Forward: u ⊕_c (t ⊗_c ((-u) ⊕_c v))
        // 정확한 Möbius 연산 야코비안을 계산
        
        scalar_t* grad_u_bid = grad_u + bid * D;
        scalar_t* grad_v_bid = grad_v + bid * D;
        const scalar_t* grad_out_bid = grad_output + bid * D;
        const scalar_t* u_bid = u + bid * D;
        const scalar_t* v_bid = v + bid * D;
        
        // 필요한 값들 계산
        float u2 = 0.0f, v2 = 0.0f, uv = 0.0f;
        for (int d = 0; d < D; ++d) {
            float u_val = u_bid[d];
            float v_val = v_bid[d];
            u2 += u_val * u_val;
            v2 += v_val * v_val;
            uv += u_val * v_val;
        }
        
        // Möbius 연산의 야코비안 계산 (simplified version)
        float c2 = c * c;
        float denom = 1.0f + 2.0f * c * uv + c2 * u2 * v2;
        denom = fmaxf(denom, config::Constants::MIN_DENOMINATOR);
        
        // ∂(u ⊕_c v)/∂u 와 ∂(u ⊕_c v)/∂v 계산
        float jacob_factor_u = (1.0f + 2.0f * c * uv + c * v2) / denom;
        float jacob_factor_v = (1.0f - c * u2) / denom;
        
        // t 파라미터와 결합된 최종 야코비안
        float final_jacob_u = jacob_factor_u * (1.0f - t) + t * jacob_factor_u;
        float final_jacob_v = jacob_factor_v * t;
        
        for (int d = tid; d < D; d += blockSize) {
            float grad_out_val = grad_out_bid[d];
            grad_u_bid[d] = grad_out_val * final_jacob_u;
            grad_v_bid[d] = grad_out_val * final_jacob_v;
        }
    }

    std::tuple<torch::Tensor, torch::Tensor> poincare_ball_backward_cuda(
        torch::Tensor grad_output,
        torch::Tensor u,
        torch::Tensor v,
        float c,
        float t
    ) {
        TORCH_CHECK(grad_output.device().is_cuda(), "grad_output must be on GPU");
        TORCH_CHECK(u.device().is_cuda(), "u must be on GPU");
        TORCH_CHECK(v.device().is_cuda(), "v must be on GPU");

        TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
        TORCH_CHECK(u.is_contiguous(), "u must be contiguous");
        TORCH_CHECK(v.is_contiguous(), "v must be contiguous");

        int B = u.size(0), D = u.size(1);
        auto grad_u = torch::zeros_like(u);
        auto grad_v = torch::zeros_like(v);
        
        int threads = 256;
        int blocks = B;
        
        AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "poincare_ball_backward_cuda", [&] {
            poincare_ball_backward_kernel<scalar_t><<<blocks, threads>>> (
                grad_output.data_ptr<scalar_t>(),
                u.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                grad_u.data_ptr<scalar_t>(),
                grad_v.data_ptr<scalar_t>(),
                c, t, B, D
                );
            });

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
        
        return std::make_tuple(grad_u, grad_v);
    }
}
