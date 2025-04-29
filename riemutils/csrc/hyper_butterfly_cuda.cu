// riemutils/csrc/hyper_butterfly_cuda.cu

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include "hyper_butterfly.h"

// -----------------------------------------------------------------------------
// 오류 체크 매크로
// -----------------------------------------------------------------------------
#define CHECK_CUDA_CONTIGUOUS(x)                                    \
    TORCH_CHECK((x).device().is_cuda(), #x " must be CUDA tensor"); \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

#define CUDA_CHECK(err)                               \
    do                                                \
    {                                                 \
        auto e = (err);                               \
        TORCH_CHECK(e == cudaSuccess, "CUDA error: ", \
                    cudaGetErrorString(e));           \
    } while (0)

static constexpr float EPS = 1e-7f;

// -----------------------------------------------------------------------------
// atanh 헬퍼 (인자 클램핑 적용)
// -----------------------------------------------------------------------------
__device__ __forceinline__ float atanh_device(float x)
{
    x = fminf(fmaxf(x, -1.0f + 1e-6f), 1.0f - 1e-6f);
    return 0.5f * logf((1.0f + x) / (1.0f - x));
}

// -----------------------------------------------------------------------------
// 다음 2의 거듭제곱 계산 (MSVC/Clang 호환)
// -----------------------------------------------------------------------------
static inline int next_pow2(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

// -----------------------------------------------------------------------------
// 1) 로그 맵 커널 (클램핑 + 안전 분모)
// -----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void log_map_origin_kernel(
    const scalar_t *__restrict__ x,
    scalar_t *__restrict__ out,
    float c,
    int batch,
    int dim)
{
    extern __shared__ float shared_norm[];
    int tid = threadIdx.x, bid = blockIdx.x;

    // 1) shared 초기화
    if (tid == 0)
        shared_norm[0] = 0.f;
    __syncthreads();

    // 2) partial sum of squares
    float sum = 0.f;
    const scalar_t *xb = x + bid * dim;
    for (int i = tid; i < dim; i += blockDim.x)
        sum += xb[i] * xb[i];
    // warp‐reduce
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if ((tid & (warpSize - 1)) == 0)
        atomicAdd(&shared_norm[0], sum);
    __syncthreads();

    // 3) 안정화
    if (tid == 0)
        shared_norm[0] = fmaxf(shared_norm[0], EPS);
    __syncthreads();

    // 4) clamp √c·‖x‖, 안전 분모
    float norm = sqrtf(shared_norm[0]);
    float scn = sqrtf(c) * norm;
    scn = fminf(fmaxf(scn, 1e-6f), 0.999999f); // 하한/상한 클램핑
    float denom = scn + 1e-6f;                 // 분모 안전화
    float numer = atanh_device(scn);
    float factor = numer / denom;

    // 5) output
    scalar_t *yb = out + bid * dim;
    for (int i = tid; i < dim; i += blockDim.x)
        yb[i] = factor * xb[i];
}

// -----------------------------------------------------------------------------
// 2) 지수 맵 커널 (클램핑 + 안전 분모)
// -----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void exp_map_origin_kernel(
    const scalar_t *__restrict__ v,
    scalar_t *__restrict__ out,
    float c,
    int batch,
    int dim)
{
    extern __shared__ float shared_norm[];
    int tid = threadIdx.x, bid = blockIdx.x;

    if (tid == 0)
        shared_norm[0] = 0.f;
    __syncthreads();

    float sum = 0.f;
    const scalar_t *vb = v + bid * dim;
    for (int i = tid; i < dim; i += blockDim.x)
        sum += vb[i] * vb[i];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if ((tid & (warpSize - 1)) == 0)
        atomicAdd(&shared_norm[0], sum);
    __syncthreads();

    if (tid == 0)
        shared_norm[0] = fmaxf(shared_norm[0], EPS);
    __syncthreads();

    float norm = sqrtf(shared_norm[0]);
    float scn = sqrtf(c) * norm;
    scn = fminf(fmaxf(scn, 1e-6f), 10.0f); // 상한 클램핑 (max=10)
    float denom = scn + 1e-3f;             // 분모 여유
    float numer = tanhf(scn);
    float factor = numer / denom;

    scalar_t *yb = out + bid * dim;
    for (int i = tid; i < dim; i += blockDim.x)
        yb[i] = factor * vb[i];
}

// -----------------------------------------------------------------------------
// 3) Butterfly 레이어 커널 (변경 없음)
// -----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void butterfly_layer_kernel(
    const scalar_t *__restrict__ input,
    scalar_t *__restrict__ output,
    const scalar_t *__restrict__ params,
    int batch,
    int dim,
    int layer_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int bs = 1 << layer_idx;
    int nb = dim / (2 * bs);

    if (idx >= batch * dim)
        return;
    while (idx < batch * dim)
    {
        int b = idx / dim, f = idx % dim;
        int blk = (f / (2 * bs)) % nb;
        int loc = f % (2 * bs), off = loc % bs;
        int pidx = blk * 2;

        float a = params[pidx], bb = params[pidx + 1];
        int base = b * dim + blk * 2 * bs;
        float x1 = input[base + off], x2 = input[base + off + bs];
        bool hi = loc >= bs;
        output[idx] = hi
                          ? (-bb * x1 + a * x2)
                          : (a * x1 + bb * x2);
        idx += stride;
    }
}

// -----------------------------------------------------------------------------
// 4) CUDA 래퍼 함수들
// -----------------------------------------------------------------------------
torch::Tensor log_map_origin_cuda(torch::Tensor x, float c)
{
    CHECK_CUDA_CONTIGUOUS(x);
    int batch = x.size(0), dim = x.size(1);
    auto out = torch::empty_like(x);
    int th = std::min(dim, 1024), sh = sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "log_map_origin_cuda", ([&]
                                                                        { log_map_origin_kernel<scalar_t><<<batch, th, sh>>>(
                                                                              x.data_ptr<scalar_t>(),
                                                                              out.data_ptr<scalar_t>(),
                                                                              c, batch, dim); }));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return out;
}

torch::Tensor exp_map_origin_cuda(torch::Tensor v, float c)
{
    CHECK_CUDA_CONTIGUOUS(v);
    int batch = v.size(0), dim = v.size(1);
    auto out = torch::empty_like(v);
    int th = std::min(dim, 1024), sh = sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "exp_map_origin_cuda", ([&]
                                                                        { exp_map_origin_kernel<scalar_t><<<batch, th, sh>>>(
                                                                              v.data_ptr<scalar_t>(),
                                                                              out.data_ptr<scalar_t>(),
                                                                              c, batch, dim); }));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return out;
}

std::vector<torch::Tensor> hyper_butterfly_cuda(
    torch::Tensor x,
    torch::Tensor params,
    torch::Tensor /*args*/,
    float c,
    int L)
{
    CHECK_CUDA_CONTIGUOUS(x);
    CHECK_CUDA_CONTIGUOUS(params);

    int batch = x.size(0), dim = x.size(1), orig = dim;
    // 1) 2^n 패딩
    int pd = next_pow2(dim);
    if (pd != dim)
    {
        auto xp = torch::zeros({batch, pd}, x.options());
        xp.narrow(1, 0, dim).copy_(x);
        x = xp;
        dim = pd;
    }
    // 2) params 길이 확인
    int log2d = int(log2f((float)dim)), need = 0;
    for (int i = 0; i < L; ++i)
    {
        int li = i % log2d;
        need += (dim / (2 * (1 << li))) * 2;
    }
    TORCH_CHECK(params.size(0) >= need, "not enough params");

    // 3) 버퍼 준비
    auto u = torch::empty_like(x), v = torch::empty_like(x), y = torch::empty_like(x);

    // 4) 로그 맵
    {
        int th = std::min(dim, 1024), sh = sizeof(float);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "logmap", ([&]
                                                               { log_map_origin_kernel<scalar_t><<<batch, th, sh>>>(
                                                                     x.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(),
                                                                     c, batch, dim); }));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    // 5) Butterfly 레이어
    {
        int th = 256, bl = std::min(1024, (batch * dim + th - 1) / th);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "butterfly", ([&]
                                                                  {
            int ofs=0;
            for(int i=0; i<L; ++i) {
                int li = i % log2d;
                butterfly_layer_kernel<scalar_t><<<bl,th>>>(
                    (i%2==0 ? u.data_ptr<scalar_t>() : v.data_ptr<scalar_t>()),
                    (i%2==0 ? v.data_ptr<scalar_t>() : u.data_ptr<scalar_t>()),
                    params.data_ptr<scalar_t>() + ofs,
                    batch, dim, li
                );
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
                ofs += (dim/(2*(1<<li))) * 2;
            }
            if (L%2==1) u.copy_(v); }));
    }
    // 6) 지수 맵
    {
        int th = std::min(dim, 1024), sh = sizeof(float);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "expmap", ([&]
                                                               { exp_map_origin_kernel<scalar_t><<<batch, th, sh>>>(
                                                                     u.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
                                                                     c, batch, dim); }));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    // 7) 원본 차원으로 슬라이스
    if (dim != orig)
    {
        auto yo = y.narrow(1, 0, orig).contiguous();
        return {yo, u, v};
    }
    return {y, u, v};
}

// // riemutils/csrc/hyper_butterfly_cuda.cu
//
// #include <torch/extension.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <vector>
// #include <cmath>
// #include "hyper_butterfly.h"
//
// // -----------------------------------------------------------------------------
// // 오류 체크 매크로
// // -----------------------------------------------------------------------------
// #define CHECK_CUDA_CONTIGUOUS(x)                                    \
//     TORCH_CHECK((x).device().is_cuda(), #x " must be CUDA tensor"); \
//     TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
//
// #define CUDA_CHECK(err)                                                       \
//     do                                                                        \
//     {                                                                         \
//         auto e = (err);                                                       \
//         TORCH_CHECK(e == cudaSuccess, "CUDA error: ", cudaGetErrorString(e)); \
//     } while (0)
//
// static constexpr float EPS = 1e-7f;
//
// // -----------------------------------------------------------------------------
// // atanh 헬퍼
// // -----------------------------------------------------------------------------
// __device__ __forceinline__ float atanh_device(float x)
// {
//     // atanh 유효 범위 클램핑
//     x = fminf(fmaxf(x, -0.9999f), 0.9999f);
//     return 0.5f * logf((1.0f + x) / (1.0f - x));
// }
//
// // -----------------------------------------------------------------------------
// // 다음 2의 거듭제곱 계산 (MSVC/Clang 호환)
// // -----------------------------------------------------------------------------
// static inline int next_pow2(int v)
// {
//     v--;
//     v |= v >> 1;
//     v |= v >> 2;
//     v |= v >> 4;
//     v |= v >> 8;
//     v |= v >> 16;
//     return v + 1;
// }
//
// // -----------------------------------------------------------------------------
// // 1) 로그 맵 커널 (with debug prints)
// // -----------------------------------------------------------------------------
// template <typename scalar_t>
// __global__ void log_map_origin_kernel(
//     const scalar_t *__restrict__ x,
//     scalar_t *__restrict__ out,
//     float c,
//     int batch,
//     int dim)
// {
//     extern __shared__ float shared_norm[]; // length = 1
//     int tid = threadIdx.x;
//     int bid = blockIdx.x;
//     if (tid == 0)
//         printf("[LOGMAP] block %d start (dim=%d)\n", bid, dim);
//
//     // 초기화
//     if (tid == 0)
//         shared_norm[0] = 0.f;
//     __syncthreads();
//
//     // partial sum
//     float sum = 0.f;
//     const scalar_t *xb = x + bid * dim;
//     for (int i = tid; i < dim; i += blockDim.x)
//     {
//         float v = xb[i];
//         sum += v * v;
//     }
//     // warp-level reduction
//     for (int offset = warpSize / 2; offset > 0; offset >>= 1)
//     {
//         sum += __shfl_down_sync(0xffffffff, sum, offset);
//     }
//     // accumulate
//     if ((tid & (warpSize - 1)) == 0)
//         atomicAdd(&shared_norm[0], sum);
//     __syncthreads();
//
//     // 안정화
//     if (tid == 0)
//     {
//         if (shared_norm[0] < EPS)
//             printf("[LOGMAP] small shared_norm: %f\n", shared_norm[0]);
//         shared_norm[0] = fmaxf(shared_norm[0], EPS);
//     }
//     __syncthreads();
//
//     float norm = sqrtf(shared_norm[0]);
//     float factor = atanh_device(sqrtf(c) * norm) / (sqrtf(c) * norm);
//
//     // write back
//     scalar_t *yb = out + bid * dim;
//     for (int i = tid; i < dim; i += blockDim.x)
//     {
//         yb[i] = factor * xb[i];
//     }
//     if (tid == 0)
//         printf("[LOGMAP] block %d done\n", bid);
// }
//
// // -----------------------------------------------------------------------------
// // 2) 지수 맵 커널 (with debug prints)
// // -----------------------------------------------------------------------------
// template <typename scalar_t>
// __global__ void exp_map_origin_kernel(
//     const scalar_t *__restrict__ v,
//     scalar_t *__restrict__ out,
//     float c,
//     int batch,
//     int dim)
// {
//     extern __shared__ float shared_norm[]; // length = 1
//     int tid = threadIdx.x;
//     int bid = blockIdx.x;
//     if (tid == 0)
//         printf("[EXPMAP] block %d start (dim=%d)\n", bid, dim);
//
//     // 초기화
//     if (tid == 0)
//         shared_norm[0] = 0.f;
//     __syncthreads();
//
//     // partial sum
//     float sum = 0.f;
//     const scalar_t *vb = v + bid * dim;
//     for (int i = tid; i < dim; i += blockDim.x)
//     {
//         float val = vb[i];
//         sum += val * val;
//     }
//     for (int offset = warpSize / 2; offset > 0; offset >>= 1)
//     {
//         sum += __shfl_down_sync(0xffffffff, sum, offset);
//     }
//     if ((tid & (warpSize - 1)) == 0)
//         atomicAdd(&shared_norm[0], sum);
//     __syncthreads();
//
//     if (tid == 0)
//     {
//         if (shared_norm[0] < EPS)
//             printf("[EXPMAP] small shared_norm: %f\n", shared_norm[0]);
//         shared_norm[0] = fmaxf(shared_norm[0], EPS);
//     }
//     __syncthreads();
//
//     float norm = sqrtf(shared_norm[0]);
//     float factor = tanhf(sqrtf(c) * norm) / (sqrtf(c) * norm);
//
//     // write back
//     scalar_t *yb = out + bid * dim;
//     for (int i = tid; i < dim; i += blockDim.x)
//     {
//         yb[i] = factor * vb[i];
//     }
//     if (tid == 0)
//         printf("[EXPMAP] block %d done\n", bid);
// }
//
// // -----------------------------------------------------------------------------
// // 3) Butterfly 레이어 커널 (with debug prints)
// // -----------------------------------------------------------------------------
// template <typename scalar_t>
// __global__ void butterfly_layer_kernel(
//     const scalar_t *__restrict__ input,
//     scalar_t *__restrict__ output,
//     const scalar_t *__restrict__ params,
//     int batch,
//     int dim,
//     int layer_idx)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     int bs = 1 << layer_idx;
//     int nb = dim / (2 * bs);
//
//     // OOB check
//     if (idx >= batch * dim)
//     {
//         if (idx < batch * dim + 5 && threadIdx.x < 2)
//             printf("[BF] OOB idx=%d batch*dim=%d\n", idx, batch * dim);
//         return;
//     }
//     // debug entry
//     if (idx == 0 && threadIdx.x == 0 && blockIdx.x == 0)
//         printf("[BF] layer %d start bs=%d nb=%d dim=%d\n", layer_idx, bs, nb, dim);
//
//     while (idx < batch * dim)
//     {
//         int b = idx / dim;
//         int f = idx % dim;
//         int blk = (f / (2 * bs)) % nb;
//         int loc = f % (2 * bs);
//         int pidx = blk * 2;
//
//         // params index check
//         if (pidx + 1 >= nb * 2)
//         {
//             printf("[BF] layer=%d blk=%d pidx OOB %d >= %d\n",
//                    layer_idx, blk, pidx + 1, nb * 2);
//             return;
//         }
//
//         int base = b * dim + blk * 2 * bs;
//         int off = loc % bs;
//         int i1 = base + off;
//         int i2 = base + off + bs;
//
//         // memory access check
//         if (i1 >= batch * dim || i2 >= batch * dim)
//         {
//             printf("[BF] i1/i2 OOB b=%d blk=%d i1=%d i2=%d max=%d\n",
//                    b, blk, i1, i2, batch * dim);
//             return;
//         }
//
//         // print some params
//         if (b == 0 && idx < 4 && threadIdx.x == 0)
//             printf("[BF] idx=%d params[%d]=%f params[%d]=%f\n",
//                    idx, pidx, params[pidx], pidx + 1, params[pidx + 1]);
//
//         // actual butterfly op
//         float x1 = input[i1];
//         float x2 = input[i2];
//         bool hi = loc >= bs;
//         output[idx] = hi
//                           ? (-params[pidx + 1] * x1 + params[pidx] * x2)
//                           : (params[pidx] * x1 + params[pidx + 1] * x2);
//
//         idx += stride;
//     }
//
//     if (threadIdx.x == 0 && blockIdx.x == 0)
//         printf("[BF] layer %d done\n", layer_idx);
// }
//
// // -----------------------------------------------------------------------------
// // 4) Hyper-Butterfly CUDA 래퍼 (with debug prints)
// // -----------------------------------------------------------------------------
// std::vector<torch::Tensor> hyper_butterfly_cuda(
//     torch::Tensor x,
//     torch::Tensor params,
//     torch::Tensor /*args*/,
//     float c,
//     int L)
// {
//     // << 이 한 줄을 추가 >>
//     printf("[WRAP-ENTRY] hyper_butterfly_cuda called: x=%p params=%p c=%f L=%d\n",
//            x.data_ptr(), params.data_ptr(), c, L);
//     CHECK_CUDA_CONTIGUOUS(x);
//     CHECK_CUDA_CONTIGUOUS(params);
//
//     TORCH_CHECK(x.dim() == 2, "x must be [batch, dim]");
//     TORCH_CHECK(params.dim() == 1, "params must be 1D");
//
//     int batch = x.size(0);
//     int dim = x.size(1);
//     int orig_dim = dim;
//
//     printf("[WRAP] start batch=%d orig_dim=%d L=%d\n", batch, orig_dim, L);
//
//     // 1) pad to next power of two
//     int pd = next_pow2(dim);
//     if (pd != dim)
//     {
//         printf("[WRAP] padding dim %d -> %d\n", dim, pd);
//         auto x_pad = torch::zeros({batch, pd}, x.options());
//         x_pad.narrow(1, 0, dim).copy_(x);
//         x = x_pad;
//         dim = pd;
//     }
//
//     // verify params length
//     int log2_dim = int(log2f((float)dim));
//     int total_p = 0;
//     for (int i = 0; i < L; ++i)
//     {
//         int li = i % log2_dim;
//         total_p += (dim / (2 * (1 << li))) * 2;
//     }
//     printf("[WRAP] using dim=%d total_params_required=%d provided=%lld\n",
//            dim, total_p, (long long)params.size(0));
//     TORCH_CHECK(params.size(0) >= total_p,
//                 "params length must be >= ", total_p);
//
//     // prepare buffers
//     auto u = torch::empty_like(x);
//     auto v = torch::empty_like(x);
//     auto y = torch::empty_like(x);
//
//     // 1) 로그 맵
//     {
//         printf("[WRAP] launching log_map_origin_kernel\n");
//         int threads = std::min(dim, 1024);
//         int shared = sizeof(float);
//         AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "log_map_cuda", ([&]
//                                                                      { log_map_origin_kernel<scalar_t><<<batch, threads, shared>>>(
//                                                                            x.data_ptr<scalar_t>(),
//                                                                            u.data_ptr<scalar_t>(),
//                                                                            c, batch, dim); }));
//         CUDA_CHECK(cudaGetLastError());
//         CUDA_CHECK(cudaDeviceSynchronize());
//     }
//
//     // 2) Butterfly layers
//     {
//         printf("[WRAP] launching butterfly_layer_kernel (%d layers)\n", L);
//         int threads = 256;
//         int blocks = std::min(1024, (batch * dim + threads - 1) / threads);
//         AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "butterfly_cuda", ([&]
//                                                                        {
//             int ofs = 0;
//             for (int i = 0; i < L; ++i) {
//                 printf("[WRAP] layer %d ofs=%d\n", i, ofs);
//                 int li = i % log2_dim;
//                 const scalar_t* inp = (i % 2 == 0 ? u.data_ptr<scalar_t>() : v.data_ptr<scalar_t>());
//                 scalar_t*       out = (i % 2 == 0 ? v.data_ptr<scalar_t>() : u.data_ptr<scalar_t>());
//                 butterfly_layer_kernel<scalar_t><<<blocks,threads>>>(
//                     inp, out,
//                     params.data_ptr<scalar_t>() + ofs,
//                     batch, dim, li);
//                 CUDA_CHECK(cudaGetLastError());
//                 CUDA_CHECK(cudaDeviceSynchronize());
//                 ofs += (dim/(2*(1<<li))) * 2;
//             }
//             if (L % 2 == 1) {
//                 u.copy_(v);
//                 printf("[WRAP] copied v->u for odd L\n");
//             } }));
//     }
//
//     // 3) 지수 맵
//     {
//         printf("[WRAP] launching exp_map_origin_kernel\n");
//         int threads = std::min(dim, 1024);
//         int shared = sizeof(float);
//         AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "exp_map_cuda", ([&]
//                                                                      { exp_map_origin_kernel<scalar_t><<<batch, threads, shared>>>(
//                                                                            u.data_ptr<scalar_t>(),
//                                                                            y.data_ptr<scalar_t>(),
//                                                                            c, batch, dim); }));
//         CUDA_CHECK(cudaGetLastError());
//         CUDA_CHECK(cudaDeviceSynchronize());
//     }
//
//     // slice back to orig_dim
//     if (dim != orig_dim)
//     {
//         printf("[WRAP] slicing back to orig_dim=%d from dim=%d\n", orig_dim, dim);
//         auto y_out = y.narrow(1, 0, orig_dim).contiguous();
//         return {y_out, u, v};
//     }
//     printf("[WRAP] done without slicing\n");
//     return {y, u, v};
// }

// #include <torch/extension.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <vector>
// #include <cmath>
// #include <algorithm>
// #include "hyper_butterfly.h"
//
// // 최적화된 오류 체크 통합 매크로
// #define CHECK_CUDA_CONTIGUOUS(x)                                      \
//     TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor"); \
//     TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
//
// // CUDA 오류 검사 매크로
// #define CUDA_CHECK(x)                                                             \
//     do                                                                            \
//     {                                                                             \
//         cudaError_t err = (x);                                                    \
//         TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
//     } while (0)
//
// static constexpr float EPS = 1e-7f;
//
// // atanh CUDA 헬퍼
// __device__ __forceinline__ float atanh_device(float x)
// {
//     return 0.5f * logf((1.0f + x) / (1.0f - x));
// }
//
// //-----------------------------------------------------------------------------
// // 0) Möbius 덧셈 커널 + 래퍼
// //-----------------------------------------------------------------------------
// template <typename scalar_t>
// __global__ void mobius_add_kernel(
//     const scalar_t *__restrict__ x,
//     const scalar_t *__restrict__ y,
//     scalar_t *__restrict__ out,
//     float c,
//     int dim)
// {
//     int batch_idx = blockIdx.x;
//     int tid = threadIdx.x;
//     int stride = blockDim.x;
//     const scalar_t *xp = x + batch_idx * dim;
//     const scalar_t *yp = y + batch_idx * dim;
//     scalar_t *op = out + batch_idx * dim;
//
//     float local_x2 = 0.0f, local_y2 = 0.0f, local_xy = 0.0f;
//     for (int i = tid; i < dim; i += stride)
//     {
//         float xv = xp[i], yv = yp[i];
//         local_x2 += xv * xv;
//         local_y2 += yv * yv;
//         local_xy += xv * yv;
//     }
//     for (int offset = warpSize / 2; offset > 0; offset >>= 1)
//     {
//         local_x2 += __shfl_down_sync(0xFFFFFFFF, local_x2, offset);
//         local_y2 += __shfl_down_sync(0xFFFFFFFF, local_y2, offset);
//         local_xy += __shfl_down_sync(0xFFFFFFFF, local_xy, offset);
//     }
//     __shared__ float sx2, sy2, sxy;
//     if ((tid & (warpSize - 1)) == 0)
//     {
//         atomicAdd(&sx2, local_x2);
//         atomicAdd(&sy2, local_y2);
//         atomicAdd(&sxy, local_xy);
//     }
//     __syncthreads();
//
//     float A = 1.0f + 2.0f * c * sxy + c * sy2;
//     float B = 1.0f - c * sx2;
//     float D = fmaxf(1.0f + 2.0f * c * sxy + c * c * sx2 * sy2, EPS);
//     for (int i = tid; i < dim; i += stride)
//     {
//         op[i] = (A * xp[i] + B * yp[i]) / D;
//     }
// }
//
// torch::Tensor mobius_add_cuda(
//     torch::Tensor x,
//     torch::Tensor y,
//     double c)
// {
//     CHECK_CUDA_CONTIGUOUS(x);
//     CHECK_CUDA_CONTIGUOUS(y);
//     int batch = x.size(0), dim = x.size(1);
//     auto out = torch::empty_like(x);
//     int threads = std::min(1024, ((dim + 31) / 32) * 32);
//     mobius_add_kernel<float><<<batch, threads>>>(
//         x.data_ptr<float>(),
//         y.data_ptr<float>(),
//         out.data_ptr<float>(),
//         (float)c,
//         dim);
//     CUDA_CHECK(cudaGetLastError());
//     return out;
// }
//
// //-----------------------------------------------------------------------------
// // 1) 로그 맵 커널 (최적화됨)
// //-----------------------------------------------------------------------------
// template <typename scalar_t>
// __global__ void log_map_origin_kernel(
//     const scalar_t *__restrict__ x,
//     scalar_t *__restrict__ output,
//     float c,
//     int batch_size,
//     int dim)
// {
//     int64_t N = int64_t(batch_size) * dim;
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     float sqrt_c = sqrtf(c);
//     for (int64_t idx = tid; idx < N; idx += stride)
//     {
//         int b = idx / dim;
//         // compute norm per-batch only once
//         if (idx % dim == 0)
//         {
//             float norm_sq = 0.0f;
//             for (int i = 0; i < dim; i++)
//             {
//                 float xv = x[b * dim + i];
//                 norm_sq += xv * xv;
//             }
//             float norm = sqrtf(fmaxf(norm_sq, EPS));
//             float factor = atanh_device(sqrt_c * norm) / (sqrt_c * norm);
//             for (int i = 0; i < dim; i++)
//             {
//                 output[b * dim + i] = factor * x[b * dim + i];
//             }
//         }
//     }
// }
//
// //-----------------------------------------------------------------------------
// // 2) 지수 맵 커널 (최적화됨)
// //-----------------------------------------------------------------------------
// template <typename scalar_t>
// __global__ void exp_map_origin_kernel(
//     const scalar_t *__restrict__ v,
//     scalar_t *__restrict__ output,
//     float c,
//     int batch_size,
//     int dim)
// {
//     int64_t N = int64_t(batch_size) * dim;
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     float sqrt_c = sqrtf(c);
//     for (int64_t idx = tid; idx < N; idx += stride)
//     {
//         int b = idx / dim;
//         if (idx % dim == 0)
//         {
//             float norm_sq = 0.0f;
//             for (int i = 0; i < dim; i++)
//             {
//                 float vv = v[b * dim + i];
//                 norm_sq += vv * vv;
//             }
//             float norm = sqrtf(fmaxf(norm_sq, EPS));
//             float factor = tanhf(sqrt_c * norm) / (sqrtf(c) * norm);
//             for (int i = 0; i < dim; i++)
//             {
//                 output[b * dim + i] = factor * v[b * dim + i];
//             }
//         }
//     }
// }
//
// //-----------------------------------------------------------------------------
// // 3) 버터플라이 레이어 커널 (완전히 최적화됨)
// //-----------------------------------------------------------------------------
// template <typename scalar_t>
// __global__ void butterfly_layer_kernel(
//     const scalar_t *__restrict__ input,
//     scalar_t *__restrict__ output,
//     const scalar_t *__restrict__ params,
//     int batch_size,
//     int dim,
//     int layer_idx)
// {
//     int64_t N = int64_t(batch_size) * dim;
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     int bs = 1 << layer_idx;
//     int nb = dim / (2 * bs);
//     for (int64_t idx = tid; idx < N; idx += stride)
//     {
//         int b = idx / dim;
//         int f = idx % dim;
//         int block = (f / (2 * bs)) % nb;
//         int loc = f % (2 * bs);
//         bool hi = loc >= bs;
//         int offd = loc % bs;
//         int pidx = block * 2;
//         float a = params[pidx + 0];
//         float bb = params[pidx + 1];
//         int base = b * dim + block * (2 * bs);
//         float x1 = input[base + offd];
//         float x2 = input[base + offd + bs];
//         output[idx] = hi ? (-bb * x1 + a * x2) : (a * x1 + bb * x2);
//     }
// }
//
// // 전체 버터플라이 변환 적용
// torch::Tensor butterfly_transform_cuda(
//     torch::Tensor x,
//     torch::Tensor params,
//     int L)
// {
//     CHECK_CUDA_CONTIGUOUS(x);
//     CHECK_CUDA_CONTIGUOUS(params);
//     int batch_size = x.size(0), dim = x.size(1);
//     auto out = x.clone();
//     auto tmp = torch::empty_like(x);
//     const int threads = 256;
//     int blocks = std::min(1024, int((batch_size * dim + threads - 1) / threads));
//     AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "butterfly_transform_cuda", ([&]
//                                                                              {
//         for (int l = 0; l < L; ++l) {
//             scalar_t* inp = (l%2==0 ? out.data_ptr<scalar_t>() : tmp.data_ptr<scalar_t>());
//             scalar_t* outp= (l%2==0 ? tmp.data_ptr<scalar_t>() : out.data_ptr<scalar_t>());
//             butterfly_layer_kernel<scalar_t><<<blocks,threads>>>(
//                 inp, outp, params.data_ptr<scalar_t>(), batch_size, dim, l%(int)log2f(dim)
//             );
//         }
//         if (L%2==1) out.copy_(tmp); }));
//     CUDA_CHECK(cudaGetLastError());
//     return out;
// }
//
// //-----------------------------------------------------------------------------
// // 4) 로그/지수 맵 CUDA 래퍼
// //-----------------------------------------------------------------------------
// torch::Tensor log_map_origin_cuda(torch::Tensor x, float c)
// {
//     CHECK_CUDA_CONTIGUOUS(x);
//     int batch = x.size(0), dim = x.size(1);
//     auto out = torch::empty_like(x);
//     const int threads = 256;
//     int blocks = std::min(1024, int((batch + threads - 1) / threads));
//     AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "log_map_origin_cuda", ([&]
//                                                                         { log_map_origin_kernel<scalar_t><<<blocks, threads>>>(
//                                                                               x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), c, batch, dim); }));
//     CUDA_CHECK(cudaGetLastError());
//     return out;
// }
//
// torch::Tensor exp_map_origin_cuda(torch::Tensor v, float c)
// {
//     CHECK_CUDA_CONTIGUOUS(v);
//     int batch = v.size(0), dim = v.size(1);
//     auto out = torch::empty_like(v);
//     const int threads = 256;
//     int blocks = std::min(1024, int((batch + threads - 1) / threads));
//     AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "exp_map_origin_cuda", ([&]
//                                                                         { exp_map_origin_kernel<scalar_t><<<blocks, threads>>>(
//                                                                               v.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), c, batch, dim); }));
//     CUDA_CHECK(cudaGetLastError());
//     return out;
// }
//
// //-----------------------------------------------------------------------------
// // 5) Hyper-Butterfly 함수 (CUDA)
// //-----------------------------------------------------------------------------
// std::vector<torch::Tensor> hyper_butterfly_cuda(
//     torch::Tensor x,
//     torch::Tensor params,
//     torch::Tensor args,
//     float c,
//     int L)
// {
//     CHECK_CUDA_CONTIGUOUS(x);
//     CHECK_CUDA_CONTIGUOUS(params);
//     auto u = log_map_origin_cuda(x, c);
//     auto v = butterfly_transform_cuda(u, params, L);
//     auto y = exp_map_origin_cuda(v, c);
//     return {y, u, v};
// }
