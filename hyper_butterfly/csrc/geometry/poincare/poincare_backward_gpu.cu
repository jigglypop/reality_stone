#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "poincare_backward.h"
#include "../../utils/common_defs.h"

use namespace std;
use namespace torch;

namespace hyper_butterfly
{
    namespace geometry
    {
        namespace poincare
        {
            // atanh 미분 헬퍼 함수
            __device__ __forceinline__ float atanh_deriv_device(float x)
            {
                float x_sq = x * x;
                x_sq = fminf(fmaxf(x_sq, 0.0f), 0.999999f); // clamp for stability
                return 1.0f / (1.0f - x_sq);
            }
            // 로그 맵 역전파 커널
            template <typename scalar_t>
            __global__ void log_map_backward_kernel(
                const scalar_t *__restrict__ grad_output,
                const scalar_t *__restrict__ x,
                scalar_t *__restrict__ grad_x,
                float c, int B, int D)
            {
                extern __shared__ float sdata[];
                float *s_norm2 = sdata;      // 노름 제곱값 저장
                float *s_factor = &sdata[1]; // factor 값 저장
                float *s_proj = &sdata[2];   // 투영 계산값 저장

                int bid = blockIdx.x, tid = threadIdx.x, stride = blockDim.x;
                const scalar_t *x_b = x + bid * D;
                const scalar_t *grad_output_b = grad_output + bid * D;
                scalar_t *grad_x_b = grad_x + bid * D;

                // 초기화
                if (tid == 0)
                {
                    s_norm2[0] = 0.0f;
                    s_proj[0] = 0.0f;
                }
                __syncthreads();

                // 1. ||x||^2 계산
                float local_norm2 = 0.0f;
                for (int i = tid; i < D; i += stride)
                {
                    float xi = x_b[i];
                    local_norm2 += xi * xi;
                }
                // 워프 리덕션
                for (int offset = warpSize / 2; offset > 0; offset >>= 1)
                {
                    local_norm2 += __shfl_down_sync(0xffffffff, local_norm2, offset);
                }
                if ((tid & (warpSize - 1)) == 0)
                {
                    atomicAdd(s_norm2, local_norm2);
                }
                __syncthreads();

                // norm 계산 및 클램핑
                float norm = sqrtf(fmaxf(s_norm2[0], EPS));
                float sqrt_c = sqrtf(c);
                float scn = sqrt_c * norm;
                scn = fminf(fmaxf(scn, EPS), MAX_NORM_TANH);

                // factor 계산
                if (tid == 0)
                {
                    float atanh_val = atanhf(scn);
                    s_factor[0] = atanh_val / scn;

                    // factor 미분 계산
                    float atanh_derivative = atanh_deriv_device(scn);
                    float dfactor_dscn = (atanh_derivative - atanh_val / scn) / scn;
                    float dscn_dnorm = sqrt_c;
                    s_factor[1] = dfactor_dscn * dscn_dnorm; // dfactor_dnorm
                }
                __syncthreads();

                // 2. x 및 grad_output의 내적 계산 (proj 용)
                float local_proj = 0.0f;
                for (int i = tid; i < D; i += stride)
                {
                    local_proj += x_b[i] * grad_output_b[i];
                }
                // 워프 리덕션
                for (int offset = warpSize / 2; offset > 0; offset >>= 1)
                {
                    local_proj += __shfl_down_sync(0xffffffff, local_proj, offset);
                }
                if ((tid & (warpSize - 1)) == 0)
                {
                    atomicAdd(s_proj, local_proj);
                }
                __syncthreads();

                // 3. proj 계산 완료
                if (tid == 0)
                {
                    s_proj[0] /= (norm * norm);
                }
                __syncthreads();

                // 4. 각 차원별 그래디언트 계산
                float factor = s_factor[0];
                float dfactor_dnorm = s_factor[1];
                float proj = s_proj[0];

                for (int i = tid; i < D; i += stride)
                {
                    float xi = x_b[i];
                    float go = grad_output_b[i];

                    // 직접 항 (direct term)
                    float direct = factor * go;

                    // 간접 항 (indirect term)
                    float indirect = dfactor_dnorm * proj * xi;

                    // 최종 그래디언트
                    grad_x_b[i] = direct + indirect;
                }
            }

            // 지수 맵 역전파 커널
            template <typename scalar_t>
            __global__ void exp_map_backward_kernel(
                const scalar_t *__restrict__ grad_output,
                const scalar_t *__restrict__ v,
                scalar_t *__restrict__ grad_v,
                float c, int B, int D)
            {

                extern __shared__ float sdata[];
                float *s_norm2 = sdata;      // 노름 제곱값 저장
                float *s_factor = &sdata[1]; // factor 값 저장
                float *s_proj = &sdata[2];   // 투영 계산값 저장

                int bid = blockIdx.x, tid = threadIdx.x, stride = blockDim.x;
                const scalar_t *v_b = v + bid * D;
                const scalar_t *grad_output_b = grad_output + bid * D;
                scalar_t *grad_v_b = grad_v + bid * D;

                // 초기화
                if (tid == 0)
                {
                    s_norm2[0] = 0.0f;
                    s_proj[0] = 0.0f;
                }
                __syncthreads();

                // 1. ||v||^2 계산
                float local_norm2 = 0.0f;
                for (int i = tid; i < D; i += stride)
                {
                    float vi = v_b[i];
                    local_norm2 += vi * vi;
                }
                // 워프 리덕션
                for (int offset = warpSize / 2; offset > 0; offset >>= 1)
                {
                    local_norm2 += __shfl_down_sync(0xffffffff, local_norm2, offset);
                }
                if ((tid & (warpSize - 1)) == 0)
                {
                    atomicAdd(s_norm2, local_norm2);
                }
                __syncthreads();

                // norm 계산 및 클램핑
                float norm = sqrtf(fmaxf(s_norm2[0], EPS));
                float sqrt_c = sqrtf(c);
                float scn = sqrt_c * norm;
                scn = fminf(fmaxf(scn, EPS), MAX_TANH_INPUT);

                // factor 계산
                if (tid == 0)
                {
                    float tanh_val = tanhf(scn);
                    s_factor[0] = tanh_val / scn;

                    // factor 미분 계산
                    float tanh_sq = tanh_val * tanh_val;
                    float tanh_derivative = 1.0f - tanh_sq;
                    float dfactor_dscn = (tanh_derivative - tanh_val / scn) / scn;
                    float dscn_dnorm = sqrt_c;
                    s_factor[1] = dfactor_dscn * dscn_dnorm; // dfactor_dnorm
                }
                __syncthreads();

                // 2. v 및 grad_output의 내적 계산 (proj 용)
                float local_proj = 0.0f;
                for (int i = tid; i < D; i += stride)
                {
                    local_proj += v_b[i] * grad_output_b[i];
                }
                // 워프 리덕션
                for (int offset = warpSize / 2; offset > 0; offset >>= 1)
                {
                    local_proj += __shfl_down_sync(0xffffffff, local_proj, offset);
                }
                if ((tid & (warpSize - 1)) == 0)
                {
                    atomicAdd(s_proj, local_proj);
                }
                __syncthreads();

                // 3. proj 계산 완료
                if (tid == 0)
                {
                    s_proj[0] /= (norm * norm);
                }
                __syncthreads();

                // 4. 각 차원별 그래디언트 계산
                float factor = s_factor[0];
                float dfactor_dnorm = s_factor[1];
                float proj = s_proj[0];

                for (int i = tid; i < D; i += stride)
                {
                    float vi = v_b[i];
                    float go = grad_output_b[i];

                    // 직접 항 (direct term)
                    float direct = factor * go;

                    // 간접 항 (indirect term)
                    float indirect = dfactor_dnorm * proj * vi;

                    // 최종 그래디언트
                    grad_v_b[i] = direct + indirect;
                }
            }

            // CUDA 로그 맵 역전파 구현
            vector<Tensor> log_map_backward_cuda(
                Tensor grad_output,
                Tensor x,
                float c)
            {

                CHECK_CUDA_CONTIGUOUS(grad_output);
                CHECK_CUDA_CONTIGUOUS(x);

                int B = x.size(0), D = x.size(1);
                auto grad_x = zeros_like(x);

                int threads = std::min(D, 1024);
                int shbytes = 3 * sizeof(float);

                AT_DISPATCH_FLOATING_TYPES(
                    x.scalar_type(),
                    "log_map_backward_cuda",
                    ([&]
                     { log_map_backward_kernel<scalar_t><<<B, threads, shbytes>>>(
                           grad_output.data_ptr<scalar_t>(),
                           x.data_ptr<scalar_t>(),
                           grad_x.data_ptr<scalar_t>(),
                           c, B, D); }));

                CUDA_CHECK(cudaGetLastError());
                return {grad_x};
            }

            // CUDA 지수 맵 역전파 구현
            vector<Tensor> exp_map_backward_cuda(
                Tensor grad_output,
                Tensor v,
                float c)
            {

                CHECK_CUDA_CONTIGUOUS(grad_output);
                CHECK_CUDA_CONTIGUOUS(v);

                int B = v.size(0), D = v.size(1);
                auto grad_v = zeros_like(v);

                int threads = std::min(D, 1024);
                int shbytes = 3 * sizeof(float);

                AT_DISPATCH_FLOATING_TYPES(
                    v.scalar_type(),
                    "exp_map_backward_cuda",
                    ([&]
                     { exp_map_backward_kernel<scalar_t><<<B, threads, shbytes>>>(
                           grad_output.data_ptr<scalar_t>(),
                           v.data_ptr<scalar_t>(),
                           grad_v.data_ptr<scalar_t>(),
                           c, B, D); }));

                CUDA_CHECK(cudaGetLastError());
                return {grad_v};
            }
        }
    }
} // namespace hyper_butterfly::geometry::poincare

// Python 바인딩 함수
vector<Tensor> log_map_backward_cuda_export(
    Tensor grad_output,
    Tensor x,
    float c)
{
    return hyper_butterfly::geometry::poincare::log_map_backward_cuda(grad_output, x, c);
}
vector<Tensor> exp_map_backward_cuda_export(
    Tensor grad_output,
    Tensor v,
    float c)
{
    return hyper_butterfly::geometry::poincare::exp_map_backward_cuda(grad_output, v, c);
}