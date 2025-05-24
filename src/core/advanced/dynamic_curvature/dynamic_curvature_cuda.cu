#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <algorithm>

#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32
#define MAX_DIM 512

namespace reality_stone::advanced {

// 디바이스 함수들
__device__ __forceinline__ float safe_sigmoid(float x) {
    // 오버플로우 방지
    if (x > 88.0f) return 1.0f;
    if (x < -88.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float safe_div(float a, float b, float eps = 1e-7f) {
    return a / (b + eps);
}

/**
 * 동적 곡률 예측 커널
 * 입력 특징으로부터 곡률을 예측
 */
__global__ void dynamic_curvature_prediction_kernel(
    const float* __restrict__ features,  // [B, 1]
    const float* __restrict__ weight,    // [1, 1] (simplified)
    const float* __restrict__ bias,      // [1]
    float* __restrict__ output,          // [B]
    float c_base,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    // 선형 변환: logit = w * feature + b
    float logit = weight[0] * features[idx] + bias[0];
    
    // 시그모이드 정규화: c = c_base * σ(logit)
    output[idx] = c_base * safe_sigmoid(logit);
}

/**
 * 동적 Möbius 덧셈 커널
 * 각 배치에 대해 서로 다른 곡률 사용
 */
__global__ void dynamic_mobius_add_kernel(
    const float* __restrict__ u,           // [B, D]
    const float* __restrict__ v,           // [B, D]
    const float* __restrict__ curvatures,  // [B]
    float* __restrict__ result,            // [B, D]
    int batch_size,
    int dim
) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    
    if (b >= batch_size || d >= dim) return;
    
    // 현재 배치의 곡률
    float c = curvatures[b];
    
    // 공유 메모리에 u, v 로드
    __shared__ float u_shared[MAX_DIM];
    __shared__ float v_shared[MAX_DIM];
    __shared__ float u2, v2, uv;
    
    if (d < dim) {
        u_shared[d] = u[b * dim + d];
        v_shared[d] = v[b * dim + d];
    }
    __syncthreads();
    
    // 워프 레벨에서 내적 계산
    float u_val = (d < dim) ? u_shared[d] : 0.0f;
    float v_val = (d < dim) ? v_shared[d] : 0.0f;
    
    float u2_local = u_val * u_val;
    float v2_local = v_val * v_val;
    float uv_local = u_val * v_val;
    
    // 워프 내 리덕션
    u2_local = warp_reduce_sum(u2_local);
    v2_local = warp_reduce_sum(v2_local);
    uv_local = warp_reduce_sum(uv_local);
    
    // 첫 번째 스레드가 공유 메모리에 저장
    if (threadIdx.x == 0) {
        u2 = u2_local;
        v2 = v2_local;
        uv = uv_local;
    }
    __syncthreads();
    
    // Möbius 덧셈 계산
    if (d < dim) {
        float c2 = c * c;
        float denom = 1.0f + 2.0f * c * uv + c2 * u2 * v2;
        
        float num_u = (1.0f + 2.0f * c * uv + c * v2) * u_val;
        float num_v = (1.0f - c * u2) * v_val;
        
        result[b * dim + d] = safe_div(num_u + num_v, denom);
    }
}

/**
 * 측지선 보간 커널
 * γ(t) = (1-t)u ⊕_c tv
 */
__global__ void dynamic_poincare_layer_kernel(
    const float* __restrict__ u,           // [B, D]
    const float* __restrict__ v,           // [B, D]
    const float* __restrict__ curvatures,  // [B]
    float* __restrict__ result,            // [B, D]
    float t,
    int batch_size,
    int dim
) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    
    if (b >= batch_size || d >= dim) return;
    
    float c = curvatures[b];
    
    // 공유 메모리
    __shared__ float u_shared[MAX_DIM];
    __shared__ float v_shared[MAX_DIM];
    __shared__ float scaled_u[MAX_DIM];
    __shared__ float scaled_v[MAX_DIM];
    __shared__ float u2, v2, uv;
    
    if (d < dim) {
        u_shared[d] = u[b * dim + d];
        v_shared[d] = v[b * dim + d];
        scaled_u[d] = (1.0f - t) * u_shared[d];
        scaled_v[d] = t * v_shared[d];
    }
    __syncthreads();
    
    // 스케일된 벡터들의 내적 계산
    float su_val = (d < dim) ? scaled_u[d] : 0.0f;
    float sv_val = (d < dim) ? scaled_v[d] : 0.0f;
    
    float u2_local = su_val * su_val;
    float v2_local = sv_val * sv_val;
    float uv_local = su_val * sv_val;
    
    // 워프 리덕션
    u2_local = warp_reduce_sum(u2_local);
    v2_local = warp_reduce_sum(v2_local);
    uv_local = warp_reduce_sum(uv_local);
    
    if (threadIdx.x == 0) {
        u2 = u2_local;
        v2 = v2_local;
        uv = uv_local;
    }
    __syncthreads();
    
    // Möbius 덧셈: (1-t)u ⊕_c tv
    if (d < dim) {
        float c2 = c * c;
        float denom = 1.0f + 2.0f * c * uv + c2 * u2 * v2;
        
        float num_u = (1.0f + 2.0f * c * uv + c * v2) * su_val;
        float num_v = (1.0f - c * u2) * sv_val;
        
        result[b * dim + d] = safe_div(num_u + num_v, denom);
    }
}

/**
 * 고성능 배치 처리를 위한 퓨즈드 커널
 * 특징 추출 + 곡률 예측을 하나의 커널에서 처리
 */
__global__ void fused_feature_extraction_curvature_prediction_kernel(
    const float* __restrict__ input,      // [B, D]
    const float* __restrict__ weight,     // [1, 1]
    const float* __restrict__ bias,       // [1]
    float* __restrict__ curvatures,       // [B]
    float c_base,
    int batch_size,
    int dim
) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    
    if (b >= batch_size) return;
    
    // 공유 메모리에서 L2 norm 계산
    __shared__ float norm_squared;
    
    float local_sum = 0.0f;
    for (int i = d; i < dim; i += blockDim.x) {
        float val = input[b * dim + i];
        local_sum += val * val;
    }
    
    // 블록 내 리덕션
    local_sum = warp_reduce_sum(local_sum);
    
    if (threadIdx.x == 0) {
        norm_squared = local_sum;
    }
    __syncthreads();
    
    // 첫 번째 스레드가 곡률 계산
    if (threadIdx.x == 0) {
        float feature = sqrtf(norm_squared);
        float logit = weight[0] * feature + bias[0];
        curvatures[b] = c_base * safe_sigmoid(logit);
    }
}

// 호스트 함수들
torch::Tensor dynamic_curvature_prediction_cuda(
    const torch::Tensor& features,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float c_base
) {
    auto batch_size = features.size(0);
    auto output = torch::zeros({batch_size}, features.options());
    
    const int threads = 256;
        const int blocks = (static_cast<int>(batch_size) + threads - 1) / threads;        dynamic_curvature_prediction_kernel<<<blocks, threads>>>(        features.data_ptr<float>(),        weight.data_ptr<float>(),        bias.data_ptr<float>(),        output.data_ptr<float>(),        c_base,        static_cast<int>(batch_size)    );        cudaDeviceSynchronize();    return output;}

torch::Tensor dynamic_mobius_add_cuda(
    const torch::Tensor& u,
    const torch::Tensor& v,
    const torch::Tensor& curvatures
) {
    auto batch_size = u.size(0);
    auto dim = u.size(1);
    auto result = torch::zeros_like(u);
    
    // 타입 충돌 방지를 위한 명시적 캐스팅
    dim3 blocks(static_cast<unsigned int>(batch_size));
    dim3 threads(static_cast<unsigned int>(std::min(static_cast<int64_t>(MAX_THREADS_PER_BLOCK), dim)));
    
    dynamic_mobius_add_kernel<<<blocks, threads>>>(
        u.data_ptr<float>(),
        v.data_ptr<float>(),
        curvatures.data_ptr<float>(),
        result.data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(dim)
    );
    
    cudaDeviceSynchronize();
    return result;
}

torch::Tensor dynamic_poincare_layer_cuda(
    const torch::Tensor& u,
    const torch::Tensor& v,
    const torch::Tensor& curvatures,
    float t
) {
    auto batch_size = u.size(0);
    auto dim = u.size(1);
    auto result = torch::zeros_like(u);
    
    // 타입 충돌 방지를 위한 명시적 캐스팅
    dim3 blocks(static_cast<unsigned int>(batch_size));
    dim3 threads(static_cast<unsigned int>(std::min(static_cast<int64_t>(MAX_THREADS_PER_BLOCK), dim)));
    
    dynamic_poincare_layer_kernel<<<blocks, threads>>>(
        u.data_ptr<float>(),
        v.data_ptr<float>(),
        curvatures.data_ptr<float>(),
        result.data_ptr<float>(),
        t,
        static_cast<int>(batch_size),
        static_cast<int>(dim)
    );
    
    cudaDeviceSynchronize();
    return result;
}

/** * 퓨즈드 연산 - 특징 추출 + 곡률 예측 */torch::Tensor fused_feature_curvature_prediction_cuda(    const torch::Tensor& input,    const torch::Tensor& weight,    const torch::Tensor& bias,    float c_base) {    auto batch_size = input.size(0);    auto dim = input.size(1);    auto curvatures = torch::zeros({batch_size}, input.options());        dim3 blocks(static_cast<unsigned int>(batch_size));    dim3 threads(static_cast<unsigned int>(std::min(static_cast<int64_t>(MAX_THREADS_PER_BLOCK), dim)));        fused_feature_extraction_curvature_prediction_kernel<<<blocks, threads>>>(        input.data_ptr<float>(),        weight.data_ptr<float>(),        bias.data_ptr<float>(),        curvatures.data_ptr<float>(),        c_base,        static_cast<int>(batch_size),        static_cast<int>(dim)    );        cudaDeviceSynchronize();    return curvatures;}

} // namespace reality_stone::advanced 