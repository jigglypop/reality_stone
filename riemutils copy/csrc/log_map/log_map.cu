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
__global__ void log_map_kernel(
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
// 4) CUDA 래퍼 함수들
// -----------------------------------------------------------------------------
torch::Tensor log_map_cuda(torch::Tensor x, float c)
{
    CHECK_CUDA_CONTIGUOUS(x);
    int batch = x.size(0), dim = x.size(1);
    auto out = torch::empty_like(x);
    int th = std::min(dim, 1024), sh = sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "log_map_cuda", ([&]
                                                                        { log_map_kernel<scalar_t><<<batch, th, sh>>>(
                                                                              x.data_ptr<scalar_t>(),
                                                                              out.data_ptr<scalar_t>(),
                                                                              c, batch, dim); }));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return out;
}
