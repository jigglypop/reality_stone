#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32
#define MAX_DIM 512
#define MAX_ANCHORS 16

namespace reality_stone::advanced {

// 안전한 수학 함수들
__device__ __forceinline__ float safe_atanh(float x) {
    if (x >= 0.99f) x = 0.99f;
    if (x <= -0.99f) x = -0.99f;
    return atanhf(x);
}

__device__ __forceinline__ float safe_tanh(float x) {
    if (x > 88.0f) return 1.0f;
    if (x < -88.0f) return -1.0f;
    return tanhf(x);
}

__device__ __forceinline__ float safe_div(float a, float b, float eps = 1e-7f) {
    return a / (b + eps);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Einstein 중점 계산 커널
 * M_E = exp_0(∑ᵢ wᵢ·log_0(xᵢ))
 */
__global__ void einstein_midpoint_kernel(
    const float* __restrict__ points,    // [B, K, D]
    const float* __restrict__ weights,   // [K]
    float* __restrict__ result,          // [B, D]
    float curvature,
    int batch_size,
    int num_points,
    int dim
) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    
    if (b >= batch_size) return;
    
    // 공유 메모리
    __shared__ float weighted_log_sum[MAX_DIM];
    __shared__ float log_norm_sq;
    
    // 가중 로그 맵 합계 계산
    float local_sum = 0.0f;
    
    if (d < dim) {
        for (int k = 0; k < num_points; ++k) {
            float point_val = points[b * num_points * dim + k * dim + d];
            float weight = weights[k];
            
            // 포인트의 노름 계산 (모든 차원에 대해)
            float point_norm_sq = 0.0f;
            for (int dd = 0; dd < dim; ++dd) {
                float val = points[b * num_points * dim + k * dim + dd];
                point_norm_sq += val * val;
            }
            float point_norm = sqrtf(point_norm_sq);
            
            // 로그 맵: log_0(x) = atanh(√c||x||) / (√c||x||) * x
            float sqrt_c = sqrtf(curvature);
            float atanh_arg = fminf(sqrt_c * point_norm, 0.99f);
            float coeff = safe_div(safe_atanh(atanh_arg), sqrt_c * point_norm);
            
            local_sum += weight * coeff * point_val;
        }
        weighted_log_sum[d] = local_sum;
    } else {
        weighted_log_sum[d] = 0.0f;
    }
    __syncthreads();
    
    // 가중 로그 합의 노름 계산
    if (threadIdx.x == 0) {
        float norm_sq = 0.0f;
        for (int dd = 0; dd < dim; ++dd) {
            norm_sq += weighted_log_sum[dd] * weighted_log_sum[dd];
        }
        log_norm_sq = norm_sq;
    }
    __syncthreads();
    
    // 지수 맵: exp_0(y) = tanh(√c||y||) / (√c||y||) * y
    if (d < dim) {
        float log_norm = sqrtf(log_norm_sq);
        float sqrt_c = sqrtf(curvature);
        float tanh_arg = sqrt_c * log_norm;
        float exp_coeff = safe_div(safe_tanh(tanh_arg), sqrt_c * log_norm);
        
        result[b * dim + d] = exp_coeff * weighted_log_sum[d];
    }
}

/**
 * 다중 측지선 혼합 커널
 * 각 앵커에 대해 측지선 보간 후 가중 평균
 */
__global__ void multi_geodesic_mixing_kernel(
    const float* __restrict__ input,     // [B, D]
    const float* __restrict__ anchors,   // [K, D]
    const float* __restrict__ t_values,  // [K]
    const float* __restrict__ weights,   // [K]
    float* __restrict__ result,          // [B, D]
    float curvature,
    int batch_size,
    int num_anchors,
    int dim
) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    
    if (b >= batch_size || d >= dim) return;
    
    __shared__ float input_shared[MAX_DIM];
    __shared__ float mixed_result[MAX_DIM];
    __shared__ float total_weight;
    
    // 입력 로드
    if (d < dim) {
        input_shared[d] = input[b * dim + d];
        mixed_result[d] = 0.0f;
    }
    
    if (threadIdx.x == 0) {
        total_weight = 0.0f;
    }
    __syncthreads();
    
    // 각 앵커에 대한 처리
    for (int k = 0; k < num_anchors; ++k) {
        __shared__ float anchor_k[MAX_DIM];
        __shared__ float t_k, w_k;
        __shared__ float distance_weight;
        
        // 앵커와 파라미터 로드
        if (d < dim) {
            anchor_k[d] = anchors[k * dim + d];
        }
        
        if (threadIdx.x == 0) {
            // 시그모이드로 t 정규화
            t_k = 1.0f / (1.0f + expf(-t_values[k]));
            w_k = weights[k];
            
            // 거리 기반 가중치 계산
            float dist_sq = 0.0f;
            for (int dd = 0; dd < dim; ++dd) {
                float diff = input_shared[dd] - anchor_k[dd];
                dist_sq += diff * diff;
            }
            distance_weight = expf(-sqrtf(dist_sq));
        }
        __syncthreads();
        
        // 측지선 보간: γ(t) = (1-t)·0 ⊕_c t·input
        if (d < dim) {
            float scaled_input = t_k * input_shared[d];
            
            // 간단한 Möbius 덧셈 (원점과 scaled_input)
            float norm_sq = 0.0f;
            for (int dd = 0; dd < dim; ++dd) {
                float val = t_k * input_shared[dd];
                norm_sq += val * val;
            }
            
            float denom = 1.0f + curvature * norm_sq;
            float geodesic_point = scaled_input / denom;
            
            // 가중 누적
            float combined_weight = w_k * distance_weight;
            mixed_result[d] += combined_weight * geodesic_point;
            
            if (d == 0) {
                atomicAdd(&total_weight, combined_weight);
            }
        }
        __syncthreads();
    }
    
    // 정규화
    if (d < dim) {
        result[b * dim + d] = safe_div(mixed_result[d], total_weight);
    }
}

/**
 * 통합 측지선 활성화 커널
 * multi_geodesic_mixing_kernel과 동일한 구현
 */
__global__ void geodesic_activation_kernel(
    const float* __restrict__ input,     // [B, D]
    const float* __restrict__ anchors,   // [K, D]
    const float* __restrict__ t_values,  // [K]
    const float* __restrict__ weights,   // [K]
    float* __restrict__ result,          // [B, D]
    float curvature,
    int batch_size,
    int num_anchors,
    int dim
) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    
    if (b >= batch_size || d >= dim) return;
    
    __shared__ float input_shared[MAX_DIM];
    __shared__ float mixed_result[MAX_DIM];
    __shared__ float total_weight;
    
    // 입력 로드
    if (d < dim) {
        input_shared[d] = input[b * dim + d];
        mixed_result[d] = 0.0f;
    }
    
    if (threadIdx.x == 0) {
        total_weight = 0.0f;
    }
    __syncthreads();
    
    // 각 앵커에 대한 처리
    for (int k = 0; k < num_anchors; ++k) {
        __shared__ float anchor_k[MAX_DIM];
        __shared__ float t_k, w_k;
        __shared__ float distance_weight;
        
        // 앵커와 파라미터 로드
        if (d < dim) {
            anchor_k[d] = anchors[k * dim + d];
        }
        
        if (threadIdx.x == 0) {
            // 시그모이드로 t 정규화
            t_k = 1.0f / (1.0f + expf(-t_values[k]));
            w_k = weights[k];
            
            // 거리 기반 가중치 계산
            float dist_sq = 0.0f;
            for (int dd = 0; dd < dim; ++dd) {
                float diff = input_shared[dd] - anchor_k[dd];
                dist_sq += diff * diff;
            }
            distance_weight = expf(-sqrtf(dist_sq));
        }
        __syncthreads();
        
        // 측지선 보간: γ(t) = (1-t)·0 ⊕_c t·input
        if (d < dim) {
            float scaled_input = t_k * input_shared[d];
            
            // 간단한 Möbius 덧셈 (원점과 scaled_input)
            float norm_sq = 0.0f;
            for (int dd = 0; dd < dim; ++dd) {
                float val = t_k * input_shared[dd];
                norm_sq += val * val;
            }
            
            float denom = 1.0f + curvature * norm_sq;
            float geodesic_point = scaled_input / denom;
            
            // 가중 누적
            float combined_weight = w_k * distance_weight;
            mixed_result[d] += combined_weight * geodesic_point;
            
            if (d == 0) {
                atomicAdd(&total_weight, combined_weight);
            }
        }
        __syncthreads();
    }
    
    // 정규화
    if (d < dim) {
        result[b * dim + d] = safe_div(mixed_result[d], total_weight);
    }
}

// 호스트 함수들
torch::Tensor einstein_midpoint_cuda(
    const torch::Tensor& points,
    const torch::Tensor& weights,
    float curvature
) {
    auto batch_size = points.size(0);
    auto num_points = points.size(1);
    auto dim = points.size(2);
    
    auto result = torch::zeros({batch_size, dim}, points.options());
    
    dim3 blocks(static_cast<unsigned int>(batch_size));
    dim3 threads(static_cast<unsigned int>(std::min(static_cast<int64_t>(MAX_THREADS_PER_BLOCK), dim)));
    
    einstein_midpoint_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(),
        weights.data_ptr<float>(),
        result.data_ptr<float>(),
        curvature,
        static_cast<int>(batch_size),
        static_cast<int>(num_points),
        static_cast<int>(dim)
    );
    
    cudaDeviceSynchronize();
    return result;
}

torch::Tensor multi_geodesic_mixing_cuda(
    const torch::Tensor& input,
    const torch::Tensor& anchors,
    const torch::Tensor& t_values,
    const torch::Tensor& weights,
    float curvature
) {
    auto batch_size = input.size(0);
    auto dim = input.size(1);
    auto num_anchors = anchors.size(0);
    auto result = torch::zeros_like(input);
    
    dim3 blocks(static_cast<unsigned int>(batch_size));
    dim3 threads(static_cast<unsigned int>(std::min(static_cast<int64_t>(MAX_THREADS_PER_BLOCK), dim)));
    
    multi_geodesic_mixing_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        anchors.data_ptr<float>(),
        t_values.data_ptr<float>(),
        weights.data_ptr<float>(),
        result.data_ptr<float>(),
        curvature,
        static_cast<int>(batch_size),
        static_cast<int>(num_anchors),
        static_cast<int>(dim)
    );
    
    cudaDeviceSynchronize();
    return result;
}

torch::Tensor geodesic_activation_cuda(
    const torch::Tensor& input,
    const torch::Tensor& anchors,
    const torch::Tensor& t_values,
    const torch::Tensor& weights,
    float curvature
) {
    return multi_geodesic_mixing_cuda(input, anchors, t_values, weights, curvature);
}

} // namespace reality_stone::advanced 