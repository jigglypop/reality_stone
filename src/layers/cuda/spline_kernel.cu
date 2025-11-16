// src/layers/cuda/spline_kernel.cu

#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cstdio>

// Warp-level helper to sum all thread values in a warp.
// Must be defined before kernels that use it.
__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Catmull-Rom spline interpolation kernel
__global__ void spline_interpolate_kernel(
    const float* __restrict__ control_points,  // (k+1) x in_features
    float* __restrict__ weights,               // out_features x in_features
    const int k,
    const int in_features,
    const int out_features
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = out_features * in_features;
    
    if (tid >= total_elements) return;
    
    const int out_idx = tid / in_features;
    const int in_idx = tid % in_features;
    
    // Normalize output index to [0, 1]
    const float t = (float)out_idx / (out_features - 1);
    const float t_scaled = t * k;
    
    // Compute control point index (with clamping)
    int j = (int)floorf(t_scaled);
    j = max(1, min(j, k - 2));
    
    const float t_local = t_scaled - j;
    
    // Load 4 control points
    const float p0 = control_points[(j - 1) * in_features + in_idx];
    const float p1 = control_points[j * in_features + in_idx];
    const float p2 = control_points[(j + 1) * in_features + in_idx];
    const float p3 = control_points[(j + 2) * in_features + in_idx];
    
    // Catmull-Rom coefficients
    const float t2 = t_local * t_local;
    const float t3 = t2 * t_local;
    
    const float c0 = -0.5f * t3 + t2 - 0.5f * t_local;
    const float c1 = 1.5f * t3 - 2.5f * t2 + 1.0f;
    const float c2 = -1.5f * t3 + 2.0f * t2 + 0.5f * t_local;
    const float c3 = 0.5f * t3 - 0.5f * t2;
    
    // Compute interpolated value
    weights[out_idx * in_features + in_idx] = c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3;
}

// Spline-based GEMM kernel (matrix multiply of input and interpolated weights)
__global__ void spline_gemm_kernel(
    const float* __restrict__ input,           // batch_size x in_features
    const float* __restrict__ control_points,  // (k+1) x in_features
    float* __restrict__ output,                // batch_size x out_features
    const int batch_size,
    const int k,
    const int in_features,
    const int out_features
) {
    // Each block computes one output element
    const int batch_idx = blockIdx.y;
    const int out_idx = blockIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= out_features) return;
    
    // Load input into shared memory (tiling)
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    
    // Compute spline parameters
    const float t = (float)out_idx / (out_features - 1);
    const float t_scaled = t * k;
    int j = (int)floorf(t_scaled);
    j = max(1, min(j, k - 2));
    const float t_local = t_scaled - j;
    
    // Catmull-Rom 계수
    const float t2 = t_local * t_local;
    const float t3 = t2 * t_local;
    
    const float c0 = -0.5f * t3 + t2 - 0.5f * t_local;
    const float c1 = 1.5f * t3 - 2.5f * t2 + 1.0f;
    const float c2 = -1.5f * t3 + 2.0f * t2 + 0.5f * t_local;
    const float c3 = 0.5f * t3 - 0.5f * t2;
    
    // Load input into shared memory
    for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
        shared_input[i] = input[batch_idx * in_features + i];
    }
    __syncthreads();
    
    // Compute inner product
    float sum = 0.0f;
    for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
        // 4개의 제어점에서 보간
        const float p0 = control_points[(j - 1) * in_features + i];
        const float p1 = control_points[j * in_features + i];
        const float p2 = control_points[(j + 1) * in_features + i];
        const float p3 = control_points[(j + 2) * in_features + i];
        
        const float weight = c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3;
        sum += shared_input[i] * weight;
    }
    
    // Warp reduction
    sum = warpReduceSum(sum);
    
    // First thread stores the result
    if (threadIdx.x == 0) {
        output[batch_idx * out_features + out_idx] = sum;
    }
}

// FP16 version (placeholder for future Tensor Core optimization)
__global__ void spline_gemm_fp16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ control_points,
    half* __restrict__ output,
    const int batch_size,
    const int k,
    const int in_features,
    const int out_features
) {
    // TODO: FP16/Tensor Core 구현
}

// Backward kernel: compute gradients for control points
__global__ void spline_backward_kernel(
    const float* __restrict__ grad_output,     // (batch_size, out_features)
    const float* __restrict__ input,           // (batch_size, in_features)
    float* __restrict__ grad_control_points, // (k+1, in_features)
    const int batch_size,
    const int k,
    const int in_features,
    const int out_features
) {
    const int cp_idx = blockIdx.x; // (k+1) 제어점 중 하나
    const int in_f_idx = threadIdx.x; // in_features 중 하나

    if (cp_idx > k || in_f_idx >= in_features) return;

    float grad_sum = 0.0f;

    // Accumulate this control point's contribution over all (batch, out_feature)
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < out_features; ++j) {
            
            // 1. Check whether this control point (cp_idx) is used for output j
            const float t = (float)j / (out_features - 1);
            const float t_scaled = t * k;
            const int p1_idx = max(1, min((int)floorf(t_scaled), k - 2));

            // Catmull-Rom uses 4 control points (p0, p1, p2, p3)
            // p1 index is j, so p0=j-1, p2=j+1, p3=j+2
            if (cp_idx < p1_idx - 1 || cp_idx > p1_idx + 2) {
                continue; // this control point is not used
            }

            // 2. If used, compute Catmull-Rom coefficient for this control point
            const float t_local = t_scaled - p1_idx;
            const float t2 = t_local * t_local;
            const float t3 = t2 * t_local;
            float c = 0.0f;
            
            if (cp_idx == p1_idx - 1) {       // p0
                c = -0.5f * t3 + t2 - 0.5f * t_local;
            } else if (cp_idx == p1_idx) {    // p1
                c = 1.5f * t3 - 2.5f * t2 + 1.0f;
            } else if (cp_idx == p1_idx + 1) {  // p2
                c = -1.5f * t3 + 2.0f * t2 + 0.5f * t_local;
            } else {                          // p3
                c = 0.5f * t3 - 0.5f * t2;
            }
            
            // 3. Chain rule: dL/dCp = dL/dOut * dOut/dW * dW/dCp
            // dOut/dW = input,  dW/dCp = c
            grad_sum += grad_output[i * out_features + j] * input[i * in_features + in_f_idx] * c;
        }
    }
    
    // Atomically accumulate the computed gradient
    atomicAdd(&grad_control_points[cp_idx * in_features + in_f_idx], grad_sum);
}


// C++ interface
extern "C" {

void spline_interpolate_cuda(
    const float* control_points,
    float* weights,
    int k,
    int in_features,
    int out_features
) {
    const int total_elements = out_features * in_features;
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    spline_interpolate_kernel<<<blocks, threads_per_block>>>(
        control_points, weights, k, in_features, out_features
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in spline_interpolate: %s\n", cudaGetErrorString(error));
    }
}

void spline_forward_cuda(
    const float* input,
    const float* control_points,
    float* output,
    int batch_size,
    int k,
    int in_features,
    int out_features
) {
    // 블록 구성: (out_features, batch_size)
    dim3 blocks(out_features, batch_size);
    const int threads_per_block = 128;
    const int shared_mem_size = in_features * sizeof(float);
    
    spline_gemm_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        input, control_points, output, batch_size, k, in_features, out_features
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in spline_forward: %s\n", cudaGetErrorString(error));
    }
}

void spline_backward_cuda(
    const float* grad_output,
    const float* input,
    float* grad_control_points,
    int batch_size,
    int k,
    int in_features,
    int out_features
) {
    // Initialize control point gradients
    cudaMemset(grad_control_points, 0, (k + 1) * in_features * sizeof(float));

    // Blocks: number of control points (k+1)
    // Threads: in_features
    dim3 blocks(k + 1);
    dim3 threads(in_features);
    
    spline_backward_kernel<<<blocks, threads>>>(
        grad_output, input, grad_control_points,
        batch_size, k, in_features, out_features
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in spline_backward: %s\n", cudaGetErrorString(error));
    }
}

} // extern "C" 