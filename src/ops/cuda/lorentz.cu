#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace {
    const float EPS = 1e-7f;

    __device__ inline float lorentz_inner_product(const float* u, const float* v, int dim) {
        float result = u[0] * v[0];
        for (int i = 1; i < dim; ++i) {
            result -= u[i] * v[i];
        }
        return result;
    }
}

// Lorentz Distance CUDA Kernel
__global__ void lorentz_distance_kernel(float* out, const float* u, const float* v, float c, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* u_row = u + idx * dim;
    const float* v_row = v + idx * dim;
    
    float inner = lorentz_inner_product(u_row, v_row, dim);
    out[idx] = acoshf(fmaxf(-inner, 1.0f + EPS)) / sqrtf(c);
}

// Lorentz Layer Forward CUDA Kernel
__global__ void lorentz_layer_forward_kernel(float* out, const float* u, const float* v, float c, float t, int batch_size, int dim) {
    // Note: This is a placeholder/simplified version of the forward layer.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    const float* u_row = u + idx * dim;
    const float* v_row = v + idx * dim;
    float* out_row = out + idx * dim;

    for (int i = 0; i < dim; ++i) {
        out_row[i] = (1.0f - t) * u_row[i] + t * v_row[i];
    }
}

// Lorentz Layer Backward CUDA Kernel
__global__ void lorentz_layer_backward_kernel(
    const float* grad_output, const float* u, const float* v,
    float* grad_u, float* grad_v,
    float c, float t, int batch_size, int dim
) {
    // Note: This is a placeholder due to VJP complexity.
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
    void lorentz_distance_cuda(float* out, const float* u, const float* v, float c, int batch_size, int dim) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        lorentz_distance_kernel<<<blocks, threads>>>(out, u, v, c, batch_size, dim);
    }

    void lorentz_layer_forward_cuda(float* out, const float* u, const float* v, float c, float t, int batch_size, int dim) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        lorentz_layer_forward_kernel<<<blocks, threads>>>(out, u, v, c, t, batch_size, dim);
    }

    void lorentz_layer_backward_cuda(const float* grad_output, const float* u, const float* v, float* grad_u, float* grad_v, float c, float t, int batch_size, int dim) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        lorentz_layer_backward_kernel<<<blocks, threads>>>(grad_output, u, v, grad_u, grad_v, c, t, batch_size, dim);
    }
} 