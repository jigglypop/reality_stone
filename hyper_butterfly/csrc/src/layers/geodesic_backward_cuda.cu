#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <hyper_butterfly/utils/cuda_utils.h>
#include <hyper_butterfly/utils/numeric.h>
#include <hyper_butterfly/config/constant.h>
#include <hyper_butterfly/layers/geodesic.h>

namespace utils = hyper_butterfly::utils;
namespace config = hyper_butterfly::config;

namespace hyper_butterfly {
namespace layers {

template <typename scalar_t>
__global__ void geodesic_backward_kernel(
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

    // 노름 계산 (없이 단순화)
    float c2 = c * c;

    // 각 스레드가 여러 element 처리
    for (int d = tid; d < D; d += blockSize) {
        float u_val = u_bid[d];
        float v_val = v_bid[d];
        float grad_out_val = grad_out_bid[d];

        // 단순화된 야코비안 근사값 사용
        float jacob_u = 1.0f - t;
        float jacob_v = t;

        grad_u_bid[d] = grad_out_val * jacob_u;
        grad_v_bid[d] = grad_out_val * jacob_v;
    }
}

std::tuple<torch::Tensor, torch::Tensor> geodesic_backward_cuda(
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

    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "geodesic_backward_cuda", [&] {
        geodesic_backward_kernel<scalar_t> << <blocks, threads >> > (
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
}

// #include <torch/extension.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <hyper_butterfly/utils/cuda_utils.h>
// #include <hyper_butterfly/utils/numeric.h>
// #include <hyper_butterfly/config/constant.h>
// #include <hyper_butterfly/layers/geodesic.h>
// 
// namespace utils = hyper_butterfly::utils;
// namespace config = hyper_butterfly::config;
// 
// namespace hyper_butterfly {
// namespace layers {
// 
// __device__ void warp_reduce(volatile float* sdata, int tid) {
//     if (tid + 16 < 32) sdata[tid] += sdata[tid + 16];
//     if (tid + 8 < 32) sdata[tid] += sdata[tid + 8];
//     if (tid + 4 < 32) sdata[tid] += sdata[tid + 4];
//     if (tid + 2 < 32) sdata[tid] += sdata[tid + 2];
//     if (tid + 1 < 32) sdata[tid] += sdata[tid + 1];
// }
// 
// template <typename scalar_t>
// __global__ void geodesic_backward_kernel(
//     const scalar_t* __restrict__ grad_output,
//     const scalar_t* __restrict__ u,
//     const scalar_t* __restrict__ v,
//     scalar_t* __restrict__ grad_u,
//     scalar_t* __restrict__ grad_v,
//     float c, float t, int B, int D
// ) {
//     extern __shared__ float sdata[];
//     int bid = blockIdx.x;
//     int tid = threadIdx.x;
//     int blockSize = blockDim.x;
// 
//     // 0) 쓰레드 범위 검사
//     if (bid >= B || tid >= blockSize) return;
// 
//     // 1) shared memory layout
//     float* s_u_norm = sdata;
//     float* s_v_norm = sdata + blockSize;
//     float* s_u_dot_v = sdata + 2 * blockSize;
// 
//     const scalar_t* u_bid = u + bid * D;
//     const scalar_t* v_bid = v + bid * D;
//     scalar_t* grad_u_bid = grad_u + bid * D;
//     scalar_t* grad_v_bid = grad_v + bid * D;
//     const scalar_t* grad_out_bid = grad_output + bid * D;
// 
//     // 2) 초기화
//     s_u_norm[tid] = 0.0f;
//     s_v_norm[tid] = 0.0f;
//     s_u_dot_v[tid] = 0.0f;
//     __syncthreads();
// 
//     // 3) 노름 및 내적 합산
//     for (int d = tid; d < D; d += blockSize) {
//         float u_val = u_bid[d];
//         float v_val = v_bid[d];
//         s_u_norm[tid] += u_val * u_val;
//         s_v_norm[tid] += v_val * v_val;
//         s_u_dot_v[tid] += u_val * v_val;
//     }
//     __syncthreads();
// 
//     // 4) block-level reduction (범위 안전 검사)
//     if (blockSize > 256 && tid < blockSize - 256) {
//         s_u_norm[tid] += s_u_norm[tid + 256];
//         s_v_norm[tid] += s_v_norm[tid + 256];
//         s_u_dot_v[tid] += s_u_dot_v[tid + 256];
//     }
//     __syncthreads();
//     if (blockSize > 128 && tid < blockSize - 128) {
//         s_u_norm[tid] += s_u_norm[tid + 128];
//         s_v_norm[tid] += s_v_norm[tid + 128];
//         s_u_dot_v[tid] += s_u_dot_v[tid + 128];
//     }
//     __syncthreads();
//     if (blockSize > 64 && tid < blockSize - 64) {
//         s_u_norm[tid] += s_u_norm[tid + 64];
//         s_v_norm[tid] += s_v_norm[tid + 64];
//         s_u_dot_v[tid] += s_u_dot_v[tid + 64];
//     }
//     __syncthreads();
// 
//     // 5) warp-level reduction
//     if (tid < 32) {
//         warp_reduce(s_u_norm, tid);
//         warp_reduce(s_v_norm, tid);
//         warp_reduce(s_u_dot_v, tid);
//     }
//     __syncthreads();
// 
//     // 6) 최종 노름/내적 계산
//     float u_norm = sqrtf(fmaxf(s_u_norm[0], config::Constants::EPS));
//     float v_norm = sqrtf(fmaxf(s_v_norm[0], config::Constants::EPS));
//     float u_dot_v = s_u_dot_v[0];
//     float c2 = c * c;
//     float u2 = u_norm * u_norm;
//     float v2 = v_norm * v_norm;
// 
//     // 7) gradient back-propagation (d 경계 검사 추가)
//     for (int d = tid; d < D; d += blockSize) {
//         if (d >= D) continue;
//         float u_val = u_bid[d];
//         float v_val = v_bid[d];
//         float grad_out_v = grad_out_bid[d];
// 
//         float denom = 1.0f + 2.0f * c * u_dot_v + c2 * u2 * v2;
//         denom = fmaxf(denom, config::Constants::MIN_DENOMINATOR);
// 
//         float jacob_u = (1.0f + c * v2 - c * u2) / denom;
//         float jacob_v = (1.0f - c * u2) / denom * t;
// 
//         grad_u_bid[d] = grad_out_v * jacob_u;
//         grad_v_bid[d] = grad_out_v * jacob_v;
//     }
// }
// 
// std::tuple<torch::Tensor, torch::Tensor> geodesic_backward_cuda(
//     torch::Tensor grad_output,
//     torch::Tensor u,
//     torch::Tensor v,
//     float c,
//     float t
// ) {
//     utils::check_cuda_tensor(grad_output);
//     utils::check_cuda_tensor(u);
//     utils::check_cuda_tensor(v);
// 
//     int B = u.size(0), D = u.size(1);
//     auto grad_u = torch::zeros_like(u);
//     auto grad_v = torch::zeros_like(v);
// 
//     int threads = 256;
//     int shmem = 3 * threads * sizeof(float);
// 
//     AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "geodesic_backward_cuda", [&] {
//         geodesic_backward_kernel<scalar_t> << <B, threads, shmem >> > (
//             grad_output.data_ptr<scalar_t>(),
//             u.data_ptr<scalar_t>(),
//             v.data_ptr<scalar_t>(),
//             grad_u.data_ptr<scalar_t>(),
//             grad_v.data_ptr<scalar_t>(),
//             c, t, B, D
//             );
//         });
// 
//     utils::check_cuda_error();
//     return std::make_tuple(grad_u, grad_v);
// }
// 
// } // namespace layers
// } // namespace hyper_butterfly
