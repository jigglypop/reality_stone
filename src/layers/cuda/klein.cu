#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace {
    const float EPS = 1e-7f;
    const float BOUNDARY_EPS = 1e-5f;

    __device__ inline float dot(const float* x, const float* y, int dim) {
        float result = 0.0f;
        for (int i = 0; i < dim; ++i) {
            result += x[i] * y[i];
        }
        return result;
    }

    __device__ inline float norm_sq(const float* x, int dim) {
        return dot(x, x, dim);
    }
}

// Klein Distance CUDA Kernel
__global__ void klein_distance_kernel(float* out, const float* u, const float* v, float c, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* u_row = u + idx * dim;
    const float* v_row = v + idx * dim;
    
    float u_norm_sq = norm_sq(u_row, dim);
    float v_norm_sq = norm_sq(v_row, dim);
    float uv_dot = dot(u_row, v_row, dim);

    float num = 2.0f * (u_norm_sq * v_norm_sq - uv_dot * uv_dot);
    float den = fmaxf((1.0f - c * u_norm_sq) * (1.0f - c * v_norm_sq), EPS);
    float lambda_sq = num / den;
    float lambda = sqrtf(lambda_sq);
    
    float val = (2.0f + lambda) / fmaxf(2.0f - lambda, EPS);
    out[idx] = acoshf(val) / sqrtf(c);
}

// Klein Layer Forward CUDA Kernel
__global__ void klein_layer_forward_kernel(float* out, const float* u, const float* v, float c, float t, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float u_prime[1024]; // Assuming max dim 1024
    float v_prime[1024];
    
    const float* u_row = u + idx * dim;
    const float* v_row = v + idx * dim;
    
    // Scalar Mul for u
    float u_norm = sqrtf(norm_sq(u_row, dim));
    float u_norm_clamped = fmaxf(u_norm, EPS);
    float u_scaled_norm = fminf(u_norm_clamped * (1.0f - t), 1.0f/sqrtf(c) - BOUNDARY_EPS);
    float u_scale = u_scaled_norm / u_norm_clamped;
    for(int i=0; i<dim; ++i) u_prime[i] = u_row[i] * u_scale;

    // Scalar Mul for v
    float v_norm = sqrtf(norm_sq(v_row, dim));
    float v_norm_clamped = fmaxf(v_norm, EPS);
    float v_scaled_norm = fminf(v_norm_clamped * t, 1.0f/sqrtf(c) - BOUNDARY_EPS);
    float v_scale = v_scaled_norm / v_norm_clamped;
    for(int i=0; i<dim; ++i) v_prime[i] = v_row[i] * v_scale;
    
    // Klein Add
    float u_prime_norm_sq = norm_sq(u_prime, dim);
    float v_prime_norm_sq = norm_sq(v_prime, dim);
    float u_denom = sqrtf(fmaxf(1.0f - c * u_prime_norm_sq, EPS));
    float v_denom = sqrtf(fmaxf(1.0f - c * v_prime_norm_sq, EPS));

    float temp[1024];
    for(int i=0; i<dim; ++i) temp[i] = u_prime[i] / u_denom + v_prime[i] / v_denom;

    float temp_norm_sq = norm_sq(temp, dim);
    float res_denom = 1.0f + sqrtf(1.0f + c * temp_norm_sq);
    
    float* out_row = out + idx * dim;
    for(int i=0; i<dim; ++i) out_row[i] = temp[i] / fmaxf(res_denom, EPS);
}

// Klein Layer Backward CUDA Kernel
__global__ void klein_layer_backward_kernel(
    const float* grad_output, const float* u, const float* v,
    float* grad_u, float* grad_v,
    float c, float t, int batch_size, int dim
) {
    // Note: This is a placeholder as the VJP is complex.
    // A proper CUDA implementation requires porting the complex VJP logic from Rust.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    const float* grad_row = grad_output + idx * dim;
    float* grad_u_row = grad_u + idx * dim;
    float* grad_v_row = grad_v + idx * dim;

    for(int i=0; i<dim; ++i) {
        grad_u_row[i] = grad_row[i] * (1.0f - t);
        grad_v_row[i] = grad_row[i] * t;
    }
}

extern "C" {
    void klein_distance_cuda(float* out, const float* u, const float* v, float c, int batch_size, int dim) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        klein_distance_kernel<<<blocks, threads>>>(out, u, v, c, batch_size, dim);
    }

    void klein_layer_forward_cuda(float* out, const float* u, const float* v, float c, float t, int batch_size, int dim) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        klein_layer_forward_kernel<<<blocks, threads>>>(out, u, v, c, t, batch_size, dim);
    }

    void klein_layer_backward_cuda(const float* grad_output, const float* u, const float* v, float* grad_u, float* grad_v, float c, float t, int batch_size, int dim) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        klein_layer_backward_kernel<<<blocks, threads>>>(grad_output, u, v, grad_u, grad_v, c, t, batch_size, dim);
    }
} 