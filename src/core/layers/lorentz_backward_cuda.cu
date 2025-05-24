#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <utils/cuda_utils.h>
#include <utils/numeric.h>
#include <config/constant.h>
#include <layers/lorentz.h>

namespace utils = reality_stone::utils;
namespace config = reality_stone::config;

namespace reality_stone::layers {
    template <typename scalar_t>
    __global__ void lorentz_backward_kernel(
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

        const scalar_t* u_bid = u + bid * D;
        const scalar_t* v_bid = v + bid * D;
        scalar_t* grad_u_bid = grad_u + bid * D;
        scalar_t* grad_v_bid = grad_v + bid * D;
        const scalar_t* grad_out_bid = grad_output + bid * D;

        // Shared memory for reduction
        extern __shared__ float shared_data[];
        float* u_norm_shared = shared_data;
        float* v_norm_shared = shared_data + blockSize;
        float* uv_inner_shared = shared_data + 2 * blockSize;

        // Calculate Lorentz inner products and norms
        float local_u_norm = 0.0f;
        float local_v_norm = 0.0f;
        float local_uv_inner = 0.0f;

        // Time component (negative signature)
        if (tid == 0) {
            local_u_norm -= u_bid[0] * u_bid[0];
            local_v_norm -= v_bid[0] * v_bid[0];
            local_uv_inner -= u_bid[0] * v_bid[0];
        }

        // Space components (positive signature)
        for (int d = max(1, tid); d < D; d += blockSize) {
            float u_val = u_bid[d];
            float v_val = v_bid[d];
            local_u_norm += u_val * u_val;
            local_v_norm += v_val * v_val;
            local_uv_inner += u_val * v_val;
        }

        // Store in shared memory
        u_norm_shared[tid] = local_u_norm;
        v_norm_shared[tid] = local_v_norm;
        uv_inner_shared[tid] = local_uv_inner;
        __syncthreads();

        // Reduction
        for (int s = blockSize / 2; s > 0; s >>= 1) {
            if (tid < s) {
                u_norm_shared[tid] += u_norm_shared[tid + s];
                v_norm_shared[tid] += v_norm_shared[tid + s];
                uv_inner_shared[tid] += uv_inner_shared[tid + s];
            }
            __syncthreads();
        }

        float u_lorentz_norm = u_norm_shared[0];
        float v_lorentz_norm = v_norm_shared[0];
        float lorentz_inner = uv_inner_shared[0];

        // Ensure we're on the hyperboloid
        u_lorentz_norm = fmaxf(u_lorentz_norm, -1.0f/c + config::Constants::EPS);
        v_lorentz_norm = fmaxf(v_lorentz_norm, -1.0f/c + config::Constants::EPS);

        // Calculate gradients
        for (int d = tid; d < D; d += blockSize) {
            float grad_out_val = grad_out_bid[d];

            float jacob_u, jacob_v;
            
            if (d == 0) {
                // Time component
                jacob_u = (1.0f - t) + t * c * lorentz_inner;
                jacob_v = t * (1.0f + c * u_lorentz_norm);
            } else {
                // Space components
                jacob_u = (1.0f - t) - t * c * lorentz_inner;
                jacob_v = t * (1.0f - c * u_lorentz_norm);
            }

            grad_u_bid[d] = grad_out_val * jacob_u;
            grad_v_bid[d] = grad_out_val * jacob_v;
        }
    }

    std::tuple<torch::Tensor, torch::Tensor> lorentz_backward_cuda(
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
        int shared_mem_size = 3 * threads * sizeof(float);

        AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "lorentz_backward_cuda", [&] {
            lorentz_backward_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
#endif // WITH_CUDA