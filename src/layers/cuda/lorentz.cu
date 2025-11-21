#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define LORENTZ_EPS 1e-6f

__device__ float lorentz_inner_product(const float* u, const float* v, int dim) {
    float inner = u[0] * v[0];
    for (int i = 1; i < dim; ++i) {
        inner -= u[i] * v[i];
    }
    return inner;
}

__device__ float safe_acosh(float x) {
    return acoshf(fmaxf(x, 1.0f + LORENTZ_EPS));
}

__device__ float safe_sqrt(float x) {
    return sqrtf(fmaxf(x, LORENTZ_EPS));
}

__global__ void lorentz_distance_kernel(
    float* out, const float* u, const float* v, 
    float c, long long batch_size, long long dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* u_row = u + idx * dim;
    const float* v_row = v + idx * dim;
    
    float inner = lorentz_inner_product(u_row, v_row, dim);
    float sqrt_c = safe_sqrt(c);
    out[idx] = safe_acosh(fmaxf(c * inner, 1.0f + LORENTZ_EPS)) / sqrt_c;
}

__global__ void lorentz_layer_forward_kernel(
    float* out, const float* u, const float* v, 
    float c, float t, long long batch_size, long long dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* p = u + idx * dim;
    const float* q = v + idx * dim;
    float* result = out + idx * dim;
    
    float inner = lorentz_inner_product(p, q, dim);
    float theta = safe_acosh(fmaxf(c * inner, 1.0f + LORENTZ_EPS));
    float sinh_theta = fmaxf(sinhf(theta), LORENTZ_EPS);
    
    float w1, w2;
    if (fabsf(theta) < 1e-6f) {
        w1 = 1.0f - t;
        w2 = t;
    } else {
        w1 = sinhf((1.0f - t) * theta) / sinh_theta;
        w2 = sinhf(t * theta) / sinh_theta;
    }
    
    for (int j = 0; j < dim; ++j) {
        result[j] = w1 * p[j] + w2 * q[j];
    }
}

__global__ void lorentz_layer_backward_kernel(
    const float* grad_output, const float* u, const float* v,
    float* grad_u, float* grad_v,
    float c, float t, long long batch_size, long long dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* p = u + idx * dim;
    const float* q = v + idx * dim;
    const float* g = grad_output + idx * dim;
    float* gu = grad_u + idx * dim;
    float* gv = grad_v + idx * dim;
    
    float inner = lorentz_inner_product(p, q, dim);
    float alpha_arg = fmaxf(c * inner, 1.0f + LORENTZ_EPS);
    float alpha = acoshf(alpha_arg);
    float sinh_alpha = fmaxf(sinhf(alpha), LORENTZ_EPS);
    float cosh_alpha = coshf(alpha);
    
    float w1, w2, dw1_dalpha, dw2_dalpha;
    
    if (fabsf(alpha) < 1e-6f) {
        w1 = 1.0f - t;
        w2 = t;
        dw1_dalpha = 0.0f;
        dw2_dalpha = 0.0f;
    } else {
        w1 = sinhf((1.0f - t) * alpha) / sinh_alpha;
        w2 = sinhf(t * alpha) / sinh_alpha;
        
        float num1 = (1.0f - t) * coshf((1.0f - t) * alpha) * sinh_alpha 
                   - sinhf((1.0f - t) * alpha) * cosh_alpha;
        float num2 = t * coshf(t * alpha) * sinh_alpha 
                   - sinhf(t * alpha) * cosh_alpha;
        float denom = fmaxf(sinh_alpha * sinh_alpha, LORENTZ_EPS);
        
        dw1_dalpha = num1 / denom;
        dw2_dalpha = num2 / denom;
    }
    
    float scale = c / sinh_alpha;
    
    float g_dot_p = 0.0f;
    float g_dot_q = 0.0f;
    for (int j = 0; j < dim; ++j) {
        g_dot_p += g[j] * p[j];
        g_dot_q += g[j] * q[j];
    }
    
    float grad_term = g_dot_p * dw1_dalpha + g_dot_q * dw2_dalpha;
    
    for (int j = 0; j < dim; ++j) {
        float dalpha_dp_j = scale * ((j == 0) ? q[j] : -q[j]);
        float dalpha_dq_j = scale * ((j == 0) ? p[j] : -p[j]);
        
        gu[j] = w1 * g[j] + grad_term * dalpha_dp_j;
        gv[j] = w2 * g[j] + grad_term * dalpha_dq_j;
    }
}

extern "C" {
    void lorentz_distance_cuda(
        float* out, const float* u, const float* v, 
        float c, long long batch_size, long long dim
    ) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        lorentz_distance_kernel<<<blocks, threads>>>(out, u, v, c, batch_size, dim);
        cudaDeviceSynchronize();
    }
    
    void lorentz_layer_forward_cuda(
        float* out, const float* u, const float* v, 
        float c, float t, long long batch_size, long long dim
    ) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        lorentz_layer_forward_kernel<<<blocks, threads>>>(out, u, v, c, t, batch_size, dim);
        cudaDeviceSynchronize();
    }
    
    void lorentz_layer_backward_cuda(
        const float* grad_output, const float* u, const float* v,
        float* grad_u, float* grad_v,
        float c, float t, long long batch_size, long long dim
    ) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        lorentz_layer_backward_kernel<<<blocks, threads>>>(
            grad_output, u, v, grad_u, grad_v, c, t, batch_size, dim
        );
        cudaDeviceSynchronize();
    }
}

