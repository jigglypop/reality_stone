#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace {
    const float EPS = 1e-7f;
    const int MAX_THREADS = 256;

    // Warp-level reduction
    __device__ inline float warpReduceSum(float val) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        return val;
    }

    __device__ inline float warpReduceMax(float val) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        return val;
    }

    // Block-level reduction
    __device__ float blockReduceSum(float val) {
        __shared__ float shared[32];
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;
        
        val = warpReduceSum(val);
        if (lane == 0) shared[wid] = val;
        __syncthreads();
        
        val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
        if (wid == 0) val = warpReduceSum(val);
        return val;
    }

    __device__ float blockReduceMax(float val) {
        __shared__ float shared[32];
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;
        
        val = warpReduceMax(val);
        if (lane == 0) shared[wid] = val;
        __syncthreads();
        
        val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -1e9f;
        if (wid == 0) val = warpReduceMax(val);
        return val;
    }

    // Poincaré geodesic distance
    __device__ inline float poincare_distance(
        const float* q, const float* k, int d_h, float c
    ) {
        float dist_sq = 0.0f;
        float q_norm_sq = 0.0f;
        float k_norm_sq = 0.0f;
        
        for (int i = 0; i < d_h; ++i) {
            float diff = q[i] - k[i];
            dist_sq += diff * diff;
            q_norm_sq += q[i] * q[i];
            k_norm_sq += k[i] * k[i];
        }
        
        // d = arccosh(1 + 2c||x-y||² / ((1-c||x||²)(1-c||y||²)))
        float denom = fmaxf((1.0f - c*q_norm_sq) * (1.0f - c*k_norm_sq), EPS);
        float arg = 1.0f + 2.0f * c * dist_sq / denom;
        return acoshf(fmaxf(arg, 1.0f + EPS));
    }
}

/**
 * Fused Geodesic Top-k Attention Kernel
 * 
 * 한 커널에서 다음을 모두 수행:
 * 1. SPD metric 적용 (L @ Q, L @ K)
 * 2. Top-k key gathering
 * 3. Geodesic distance 계산
 * 4. Softmax
 * 5. Weighted sum with values
 * 
 * 1 thread block = 1 query token
 * Shared memory를 활용하여 global memory I/O 최소화
 */
__global__ void geodesic_topk_attention_fused_kernel(
    // Inputs
    const float* __restrict__ Q,      // [B, H, T, d_h]
    const float* __restrict__ K,      // [B, H, S, d_h]
    const float* __restrict__ V,      // [B, H, S, d_v]
    const int64_t* __restrict__ idx,  // [B, T, K] top-k indices
    const float* __restrict__ L,      // [d_h, d_h] SPD Cholesky factor
    // Parameters
    const float c,                    // curvature
    const float tau,                  // temperature
    const int B, const int H, 
    const int T, const int K, 
    const int d_h, const int d_v,
    // Output
    float* __restrict__ out           // [B, H, T, d_v]
) {
    // Block indices: (b, h, t)
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    
    if (b >= B || h >= H || t >= T) return;
    
    // Shared memory layout
    extern __shared__ float smem[];
    float* q_local = smem;                         // [d_h]
    float* k_local = smem + d_h;                   // [K * d_h]
    float* v_local = smem + d_h + K*d_h;           // [K * d_v]
    float* scores = smem + d_h + K*(d_h + d_v);    // [K]
    
    // ============================================================
    // Step 1: Load Q and apply SPD metric L
    // ============================================================
    // q' = L @ q
    for (int i = threadIdx.x; i < d_h; i += blockDim.x) {
        float q_transformed = 0.0f;
        const int q_offset = ((b*H + h)*T + t)*d_h;
        
        // Matrix-vector product: q'[i] = sum_j L[i,j] * q[j]
        for (int j = 0; j < d_h; ++j) {
            q_transformed += L[i*d_h + j] * Q[q_offset + j];
        }
        q_local[i] = q_transformed;
    }
    __syncthreads();
    
    // ============================================================
    // Step 2: Load top-K keys and apply SPD metric
    // ============================================================
    for (int k_idx = 0; k_idx < K; ++k_idx) {
        const int64_t s = idx[(b*T + t)*K + k_idx];
        
        for (int i = threadIdx.x; i < d_h; i += blockDim.x) {
            float k_transformed = 0.0f;
            const int k_offset = ((b*H + h)*s)*d_h;
            
            // k'[i] = sum_j L[i,j] * k[j]
            for (int j = 0; j < d_h; ++j) {
                k_transformed += L[i*d_h + j] * K[k_offset + j];
            }
            k_local[k_idx*d_h + i] = k_transformed;
        }
    }
    __syncthreads();
    
    // ============================================================
    // Step 3: Compute geodesic distances (parallel over K)
    // ============================================================
    for (int k_idx = threadIdx.x; k_idx < K; k_idx += blockDim.x) {
        float dist = poincare_distance(
            q_local, 
            k_local + k_idx*d_h, 
            d_h, 
            c
        );
        
        // Score = -dist² / τ
        scores[k_idx] = -(dist * dist) / tau;
    }
    __syncthreads();
    
    // ============================================================
    // Step 4: Softmax (numerically stable)
    // ============================================================
    // 4a. Find max score
    float max_score = -1e9f;
    for (int k_idx = threadIdx.x; k_idx < K; k_idx += blockDim.x) {
        max_score = fmaxf(max_score, scores[k_idx]);
    }
    max_score = blockReduceMax(max_score);
    if (threadIdx.x == 0) {
        scores[K] = max_score;  // Store in extra slot
    }
    __syncthreads();
    max_score = scores[K];
    
    // 4b. Compute exp and sum
    float sum_exp = 0.0f;
    for (int k_idx = threadIdx.x; k_idx < K; k_idx += blockDim.x) {
        float exp_val = expf(scores[k_idx] - max_score);
        scores[k_idx] = exp_val;
        sum_exp += exp_val;
    }
    sum_exp = blockReduceSum(sum_exp);
    if (threadIdx.x == 0) {
        scores[K] = sum_exp;  // Store in extra slot
    }
    __syncthreads();
    sum_exp = scores[K];
    
    // 4c. Normalize
    for (int k_idx = threadIdx.x; k_idx < K; k_idx += blockDim.x) {
        scores[k_idx] /= fmaxf(sum_exp, EPS);
    }
    __syncthreads();
    
    // ============================================================
    // Step 5: Load values
    // ============================================================
    for (int k_idx = 0; k_idx < K; ++k_idx) {
        const int64_t s = idx[(b*T + t)*K + k_idx];
        const int v_offset = ((b*H + h)*s)*d_v;
        
        for (int i = threadIdx.x; i < d_v; i += blockDim.x) {
            v_local[k_idx*d_v + i] = V[v_offset + i];
        }
    }
    __syncthreads();
    
    // ============================================================
    // Step 6: Weighted sum (parallel over d_v)
    // ============================================================
    const int out_offset = ((b*H + h)*T + t)*d_v;
    for (int i = threadIdx.x; i < d_v; i += blockDim.x) {
        float sum = 0.0f;
        for (int k_idx = 0; k_idx < K; ++k_idx) {
            sum += scores[k_idx] * v_local[k_idx*d_v + i];
        }
        out[out_offset + i] = sum;
    }
}

/**
 * Host function to launch the kernel
 */
extern "C" void geodesic_topk_attention_cuda(
    const float* Q,
    const float* K,
    const float* V,
    const int64_t* idx,
    const float* L,
    float c,
    float tau,
    int B, int H, int T, int S, int K_topk,
    int d_h, int d_v,
    float* out
) {
    // Shared memory size
    // q_local[d_h] + k_local[K*d_h] + v_local[K*d_v] + scores[K+1]
    size_t smem_size = (d_h + K_topk*d_h + K_topk*d_v + K_topk + 1) * sizeof(float);
    
    // Grid: (T, H, B)
    dim3 grid(T, H, B);
    
    // Block: up to 256 threads
    int block_size = min(MAX_THREADS, max(d_h, max(d_v, K_topk)));
    block_size = (block_size + 31) / 32 * 32;  // Round up to warp size
    
    geodesic_topk_attention_fused_kernel<<<grid, block_size, smem_size>>>(
        Q, K, V, idx, L,
        c, tau,
        B, H, T, K_topk, d_h, d_v,
        out
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

/**
 * FP16 version for Tensor Cores (future optimization)
 */
#ifdef ENABLE_FP16
#include <cuda_fp16.h>

__global__ void geodesic_topk_attention_fp16_kernel(
    const half* Q,
    const half* K,
    const half* V,
    const int64_t* idx,
    const half* L,
    float c, float tau,
    int B, int H, int T, int K, int d_h, int d_v,
    half* out
) {
    // TODO: Implement FP16 version with Tensor Cores
    // Use wmma API for matrix operations
}
#endif

