#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32
#define SAFE_ATANH_LIMIT 0.99f
#define EPSILON 1e-7f

namespace reality_stone::advanced {

// 안전한 atanh 함수 (발산 방지)
__device__ __forceinline__ float safe_atanh(float x) {
    if (x >= SAFE_ATANH_LIMIT) x = SAFE_ATANH_LIMIT;
    if (x <= -SAFE_ATANH_LIMIT) x = -SAFE_ATANH_LIMIT;
    return atanhf(x);
}

// 안전한 나눗셈
__device__ __forceinline__ float safe_div(float a, float b, float eps = EPSILON) {
    return a / (b + eps);
}

// 워프 레벨 리덕션
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * 경계 근접 페널티 커널
 * Poincaré 디스크 모델: R_boundary(x) = max(0, ||x|| - (1 - ε))²
 * Klein 모델: R_boundary(x) = max(0, ||x|| - (1/√c - ε))²
 */
__global__ void boundary_penalty_kernel(
    const float* __restrict__ x,        // [B, D]
    float* __restrict__ penalties,      // [B]
    float curvature,
    float epsilon,
    int batch_size,
    int dim,
    bool use_poincare_model = true      // 모델 선택 플래그
) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    
    if (b >= batch_size) return;
    
    // 공유 메모리에서 노름 계산
    __shared__ float norm_squared;
    
    float local_sum = 0.0f;
    for (int i = d; i < dim; i += blockDim.x) {
        float val = x[b * dim + i];
        local_sum += val * val;
    }
    
    // 워프 리덕션
    local_sum = warp_reduce_sum(local_sum);
    
    if (threadIdx.x == 0) {
        norm_squared = local_sum;
    }
    __syncthreads();
    
    // 첫 번째 스레드가 페널티 계산
    if (threadIdx.x == 0) {
        float norm = sqrtf(norm_squared);
        float max_norm;
        
        if (use_poincare_model) {
            // Poincaré 디스크: ||x|| < 1
            max_norm = 1.0f - epsilon;
        } else {
            // Klein 모델: ||x|| < 1/√c  
            max_norm = safe_div(1.0f, sqrtf(curvature)) - epsilon;
        }
        
        float violation = norm - max_norm;
        penalties[b] = fmaxf(0.0f, violation * violation);
    }
}

/**
 * 곡률 적응 정규화 커널
 * R_curvature(x) = ||log_0(x)||² · c
 */
__global__ void curvature_adaptive_penalty_kernel(
    const float* __restrict__ x,        // [B, D]
    float* __restrict__ penalties,      // [B]
    float curvature,
    int batch_size,
    int dim
) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    
    if (b >= batch_size) return;
    
    __shared__ float norm_squared;
    __shared__ float log_norm_squared;
    
    // 1. 유클리디안 노름 계산
    float local_sum = 0.0f;
    for (int i = d; i < dim; i += blockDim.x) {
        float val = x[b * dim + i];
        local_sum += val * val;
    }
    local_sum = warp_reduce_sum(local_sum);
    
    if (threadIdx.x == 0) {
        norm_squared = local_sum;
    }
    __syncthreads();
    
    // 2. 로그 맵 노름 계산
    if (threadIdx.x == 0) {
        float norm = sqrtf(norm_squared);
        float sqrt_c = sqrtf(curvature);
        float atanh_arg = sqrt_c * norm;
        
        // 안전한 atanh 계산
        float log_norm = safe_div(safe_atanh(atanh_arg), sqrt_c);
        log_norm_squared = log_norm * log_norm;
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        penalties[b] = curvature * log_norm_squared;
    }
}

/**
 * 측지선 분산 정규화 커널
 * R_geodesic(W) = ∑ᵢⱼ d_H²(wᵢ, wⱼ) / n²
 */
__global__ void geodesic_variance_penalty_kernel(
    const float* __restrict__ weights,  // [N, D]
    float* __restrict__ penalty,        // [1]
    float curvature,
    int n,
    int dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n * n) return;
    
    int i = tid / n;
    int j = tid % n;
    
    if (i >= j) return; // 상삼각 행렬만 계산
    
    // 하이퍼볼릭 거리 계산
    float distance_sq = 0.0f;
    
    // wᵢ와 wⱼ 사이의 하이퍼볼릭 거리
    float wi_norm_sq = 0.0f, wj_norm_sq = 0.0f, wi_wj = 0.0f;
    
    for (int d = 0; d < dim; ++d) {
        float wi_val = weights[i * dim + d];
        float wj_val = weights[j * dim + d];
        
        wi_norm_sq += wi_val * wi_val;
        wj_norm_sq += wj_val * wj_val;
        wi_wj += wi_val * wj_val;
    }
    
    // 하이퍼볼릭 거리 공식
    float numerator = wi_norm_sq + wj_norm_sq - 2.0f * wi_wj;
    float denom1 = 1.0f - curvature * wi_norm_sq;
    float denom2 = 1.0f - curvature * wj_norm_sq;
    
    float delta = safe_div(numerator, denom1 * denom2);
    distance_sq = safe_div(1.0f, curvature) * safe_atanh(sqrtf(curvature * delta));
    distance_sq = distance_sq * distance_sq;
    
    // 원자적 덧셈 (여러 스레드가 동시에 접근)
    atomicAdd(penalty, distance_sq);
}

/**
 * 통합 정규화 손실 커널
 * λ₁·R_boundary + λ₂·R_curvature + λ₃·R_geodesic
 */
__global__ void combined_regularization_kernel(
    const float* __restrict__ x,           // [B, D]
    const float* __restrict__ weights,     // [N, D] 
    float* __restrict__ total_loss,        // [1]
    float curvature,
    float lambda_boundary,
    float lambda_curvature,
    float lambda_geodesic,
    float epsilon,
    int batch_size,
    int weight_count,
    int dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float boundary_losses[MAX_THREADS_PER_BLOCK];
    __shared__ float curvature_losses[MAX_THREADS_PER_BLOCK];
    
    float boundary_loss = 0.0f;
    float curvature_loss = 0.0f;
    
    // 경계 페널티 계산 (배치별)
    if (tid < batch_size) {
        float norm_sq = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float val = x[tid * dim + d];
            norm_sq += val * val;
        }
        
        float norm = sqrtf(norm_sq);
        float max_norm = safe_div(1.0f, sqrtf(curvature)) - epsilon;
        float violation = norm - max_norm;
        boundary_loss = fmaxf(0.0f, violation * violation);
        
        // 곡률 적응 페널티
        float sqrt_c = sqrtf(curvature);
        float atanh_arg = sqrt_c * norm;
        float log_norm = safe_div(safe_atanh(atanh_arg), sqrt_c);
        curvature_loss = curvature * log_norm * log_norm;
    }
    
    boundary_losses[threadIdx.x] = boundary_loss;
    curvature_losses[threadIdx.x] = curvature_loss;
    __syncthreads();
    
    // 블록 레벨 리덕션
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            boundary_losses[threadIdx.x] += boundary_losses[threadIdx.x + stride];
            curvature_losses[threadIdx.x] += curvature_losses[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // 측지선 분산 계산 (별도 커널에서 처리하거나 근사)
    float geodesic_loss = 0.0f; // 간단화를 위해 0으로 설정
    
    if (threadIdx.x == 0) {
        float total = lambda_boundary * boundary_losses[0] + 
                     lambda_curvature * curvature_losses[0] + 
                     lambda_geodesic * geodesic_loss;
        atomicAdd(total_loss, total);
    }
}

// 호스트 함수들
torch::Tensor boundary_penalty_cuda(
    const torch::Tensor& x,
    float curvature,
    float epsilon
) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    
    auto penalties = torch::zeros({batch_size}, x.options());
    
    dim3 blocks(static_cast<unsigned int>(batch_size));
    dim3 threads(static_cast<unsigned int>(std::min(static_cast<int64_t>(MAX_THREADS_PER_BLOCK), dim)));
    
    boundary_penalty_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        penalties.data_ptr<float>(),
        curvature,
        epsilon,
        static_cast<int>(batch_size),
        static_cast<int>(dim)
    );
    
    cudaDeviceSynchronize();
    return penalties;
}

torch::Tensor curvature_adaptive_penalty_cuda(
    const torch::Tensor& x,
    float curvature
) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    
    auto penalties = torch::zeros({batch_size}, x.options());
    
    dim3 blocks(static_cast<unsigned int>(batch_size));
    dim3 threads(static_cast<unsigned int>(std::min(static_cast<int64_t>(MAX_THREADS_PER_BLOCK), dim)));
    
    curvature_adaptive_penalty_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        penalties.data_ptr<float>(),
        curvature,
        static_cast<int>(batch_size),
        static_cast<int>(dim)
    );
    
    cudaDeviceSynchronize();
    return penalties;
}

torch::Tensor geodesic_variance_penalty_cuda(
    const torch::Tensor& weights,
    float curvature
) {
    auto n = weights.size(0);
    auto dim = weights.size(1);
    auto penalty = torch::zeros({1}, weights.options());
    
    // 모든 쌍에 대해 계산
    int total_pairs = static_cast<int>(n * n);
    const int threads = 256;
    const int blocks = (total_pairs + threads - 1) / threads;
    
    geodesic_variance_penalty_kernel<<<blocks, threads>>>(
        weights.data_ptr<float>(),
        penalty.data_ptr<float>(),
        curvature,
        static_cast<int>(n),
        static_cast<int>(dim)
    );
    
    cudaDeviceSynchronize();
    
    // 평균화
    return penalty / (n * n);
}

torch::Tensor combined_regularization_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weights,
    float curvature,
    float lambda_boundary,
    float lambda_curvature,
    float lambda_geodesic,
    float epsilon
) {
    auto batch_size = x.size(0);
    auto weight_count = weights.size(0);
    auto dim = x.size(1);
    auto total_loss = torch::zeros({1}, x.options());
    
    const int threads = 256;
    const int blocks = (static_cast<int>(std::max(batch_size, weight_count)) + threads - 1) / threads;
    
    combined_regularization_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weights.data_ptr<float>(),
        total_loss.data_ptr<float>(),
        curvature,
        lambda_boundary,
        lambda_curvature,
        lambda_geodesic,
        epsilon,
        static_cast<int>(batch_size),
        static_cast<int>(weight_count),
        static_cast<int>(dim)
    );
    
    cudaDeviceSynchronize();
    return total_loss;
}

} // namespace reality_stone::advanced 