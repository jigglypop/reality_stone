#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <hyper_butterfly/utils/cuda_utils.h>
#include <hyper_butterfly/utils/numeric.h>
#include <hyper_butterfly/config/constant.h>

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
    extern __shared__ float sdata[];

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    if (bid >= B) return;

    const scalar_t* u_bid = u + bid * D;
    const scalar_t* v_bid = v + bid * D;
    scalar_t* grad_u_bid = grad_u + bid * D;
    scalar_t* grad_v_bid = grad_v + bid * D;
    const scalar_t* grad_out_bid = grad_output + bid * D;

    // 노름 계산
    float u_norm = 0.0f, v_norm = 0.0f;
    float u_dot_v = 0.0f;

    for (int d = tid; d < D; d += blockDim.x) {
        float u_val = u_bid[d];
        float v_val = v_bid[d];
        u_norm += u_val * u_val;
        v_norm += v_val * v_val;
        u_dot_v += u_val * v_val;
    }

    // 워프 리덕션
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        u_norm += __shfl_down_sync(0xffffffff, u_norm, offset);
        v_norm += __shfl_down_sync(0xffffffff, v_norm, offset);
        u_dot_v += __shfl_down_sync(0xffffffff, u_dot_v, offset);
    }

    __shared__ float s_u_norm, s_v_norm, s_u_dot_v;
    if (tid == 0) {
        s_u_norm = sqrtf(fmaxf(u_norm, config::Constants::EPS));
        s_v_norm = sqrtf(fmaxf(v_norm, config::Constants::EPS));
        s_u_dot_v = u_dot_v / (s_u_norm * s_v_norm);
    }
    __syncthreads();
    // 모비우스 덧셈 야코비안 계산
    float c2 = c * c;
    float u2 = s_u_norm * s_u_norm;
    float v2 = s_v_norm * s_v_norm;

    for (int d = tid; d < D; d += blockDim.x) {
        float u_val = u_bid[d];
        float v_val = v_bid[d];
        float grad_out_val = grad_out_bid[d];
        // delta = (-u) ⊕ v의 방향 계산
        float delta_dir = v_val / s_v_norm - u_val / s_u_norm;
        // 모비우스 덧셈 야코비안 
        float denom = 1 + 2 * c * s_u_dot_v * s_u_norm * s_v_norm + c2 * u2 * v2;
        denom = fmaxf(denom, config::Constants::MIN_DENOMINATOR);
        float jacob_u = (1 + c * v2 - c * u2) / denom;
        float jacob_v = (1 - c * u2) / denom;
        // 모비우스 스칼라 곱 효과
        jacob_v *= t;
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
    utils::check_cuda_tensor(grad_output);
    utils::check_cuda_tensor(u);
    utils::check_cuda_tensor(v);

    int B = u.size(0), D = u.size(1);
    auto grad_u = torch::zeros_like(u);
    auto grad_v = torch::zeros_like(v);

    int threads = std::min(D, 1024);
    int shmem = 3 * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "geodesic_backward_cuda", [&] {
        geodesic_backward_kernel<scalar_t> << <B, threads, shmem >> > (
            grad_output.data_ptr<scalar_t>(),
            u.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            grad_u.data_ptr<scalar_t>(),
            grad_v.data_ptr<scalar_t>(),
            c, t, B, D
            );
        });

    utils::check_cuda_error();
    return std::make_tuple(grad_u, grad_v);
}

}
}