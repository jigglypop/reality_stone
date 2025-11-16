#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define LORENTZ_EPS 1e-7f

namespace {

    __device__ inline float lorentz_inner_product(const float* u, const float* v, int dim) {
        float result = u[0] * v[0];
        for (int i = 1; i < dim; ++i) {
            result -= u[i] * v[i];
        }
        return result;
    }
}

// Lorentz Distance CUDA Kernel (matches CPU: d = acosh(max(-c <u,v>, 1+EPS)) / sqrt(c))
__global__ void lorentz_distance_kernel(float* out, const float* u, const float* v, float c, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* u_row = u + idx * dim;
    const float* v_row = v + idx * dim;
    
    float inner = lorentz_inner_product(u_row, v_row, dim);
    out[idx] = acoshf(fmaxf(-c * inner, 1.0f + LORENTZ_EPS)) / sqrtf(c);
}

// Lorentz Layer Forward CUDA Kernel (geodesic interpolation on hyperboloid)
__global__ void lorentz_layer_forward_kernel(float* out, const float* u, const float* v, float c, float t, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    const float* u_row = u + idx * dim;
    const float* v_row = v + idx * dim;
    float* out_row = out + idx * dim;
    // Minkowski inner product
    float inner = u_row[0] * v_row[0];
    for (int j = 1; j < dim; ++j) {
        inner -= u_row[j] * v_row[j];
    }
    // alpha = acosh(max(c * <u,v>, 1+EPS))
    float z = fmaxf(c * inner, 1.0f + LORENTZ_EPS);
    float alpha = acoshf(z);
    float sinh_alpha = fmaxf(sinhf(alpha), LORENTZ_EPS);
    // weights
    float w1, w2;
    if (fabsf(alpha) < 1e-6f) {
        w1 = 1.0f - t;
        w2 = t;
    } else {
        w1 = sinhf((1.0f - t) * alpha) / sinh_alpha;
        w2 = sinhf(t * alpha) / sinh_alpha;
    }
    // ambient combination
    for (int j = 0; j < dim; ++j) {
        out_row[j] = w1 * u_row[j] + w2 * v_row[j];
    }
}

// Lorentz Layer Backward CUDA Kernel
__global__ void lorentz_layer_backward_kernel(
    const float* grad_output, const float* u, const float* v,
    float* grad_u, float* grad_v,
    float c, float t, int batch_size, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    const float* p = u + idx * dim;
    const float* q = v + idx * dim;
    const float* g = grad_output + idx * dim;
    float* gu = grad_u + idx * dim;
    float* gv = grad_v + idx * dim;

    // Minkowski inner and alpha
    float inner = p[0] * q[0];
    for (int j = 1; j < dim; ++j) inner -= p[j] * q[j];
    float z = fmaxf(c * inner, 1.0f + LORENTZ_EPS);
    float alpha = acoshf(z);
    float sinh_alpha = fmaxf(sinhf(alpha), LORENTZ_EPS);
    float cosh_alpha = coshf(alpha);

    // weights
    float w1, w2;
    if (fabsf(alpha) < 1e-6f) {
        w1 = 1.0f - t;
        w2 = t;
    } else {
        w1 = sinhf((1.0f - t) * alpha) / sinh_alpha;
        w2 = sinhf(t * alpha) / sinh_alpha;
    }

    // derivatives dw/dalpha
    float num1 = (1.0f - t) * coshf((1.0f - t) * alpha) * sinh_alpha - sinhf((1.0f - t) * alpha) * cosh_alpha;
    float num2 = t * coshf(t * alpha) * sinh_alpha - sinhf(t * alpha) * cosh_alpha;
    float denom = fmaxf(sinh_alpha * sinh_alpha, LORENTZ_EPS);
    float dw1_dalpha = (fabsf(alpha) < 1e-6f) ? 0.0f : (num1 / denom);
    float dw2_dalpha = (fabsf(alpha) < 1e-6f) ? 0.0f : (num2 / denom);

    // d alpha / d p and d alpha / d q (match CPU convention)
    // scale = -c / sinh(alpha)
    float scale = -c / sinh_alpha;
    // dalpha/dp
    // time component
    float dalpha_dp0 = scale * q[0];
    float dalpha_dq0 = scale * p[0];
    // accumulate g·p and g·q (Euclidean)
    float g_dot_p = 0.0f, g_dot_q = 0.0f;
    for (int j = 0; j < dim; ++j) {
        g_dot_p += g[j] * p[j];
        g_dot_q += g[j] * q[j];
    }

    // per-dimension grads
    for (int j = 0; j < dim; ++j) {
        float dalpha_dp_j = (j == 0) ? dalpha_dp0 : scale * (-q[j]);
        float dalpha_dq_j = (j == 0) ? dalpha_dq0 : scale * (-p[j]);
        float chain = g_dot_p * dw1_dalpha + g_dot_q * dw2_dalpha;
        gu[j] = w1 * g[j] + chain * dalpha_dp_j;
        gv[j] = w2 * g[j] + chain * dalpha_dq_j;
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