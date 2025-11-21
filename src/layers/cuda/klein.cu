#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define KLEIN_EPS 1e-6f
#define BOUNDARY_EPS 1e-5f

__device__ float norm_sq(const float* x, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum += x[i] * x[i];
    }
    return sum;
}

__device__ float dot_product(const float* x, const float* y, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

__device__ float safe_acosh(float x) {
    return acoshf(fmaxf(x, 1.0f + KLEIN_EPS));
}

__device__ float safe_sqrt(float x) {
    return sqrtf(fmaxf(x, KLEIN_EPS));
}

__global__ void klein_distance_kernel(
    float* out, const float* u, const float* v, 
    float c, long long batch_size, long long dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* u_row = u + idx * dim;
    const float* v_row = v + idx * dim;
    
    float u2 = norm_sq(u_row, dim);
    float v2 = norm_sq(v_row, dim);
    float uv = dot_product(u_row, v_row, dim);
    
    float sqrt_c = safe_sqrt(c);
    float numerator = 1.0f - c * uv;
    float denominator = safe_sqrt((1.0f - c * u2) * (1.0f - c * v2));
    float arg = fmaxf(numerator / denominator, 1.0f + KLEIN_EPS);
    
    out[idx] = safe_acosh(arg) / sqrt_c;
}

__device__ void klein_scalar_impl(
    const float* x, float* out, int dim, float c, float r
) {
    float norm_sq_val = norm_sq(x, dim);
    float norm_val = fmaxf(safe_sqrt(norm_sq_val), KLEIN_EPS);
    float scaled_norm = fminf(norm_val * r, 1.0f / safe_sqrt(c) - BOUNDARY_EPS);
    float scale = scaled_norm / norm_val;
    
    for (int i = 0; i < dim; ++i) {
        out[i] = scale * x[i];
    }
}

__device__ void klein_add_impl(
    const float* u, const float* v, float* out, int dim, float c
) {
    float u_norm_sq = norm_sq(u, dim);
    float uv_dot = dot_product(u, v, dim);
    
    float gamma_u = 1.0f / safe_sqrt(1.0f - c * u_norm_sq);
    float denom = fmaxf(1.0f + c * uv_dot, KLEIN_EPS);
    float denom_inv = 1.0f / denom;
    
    float inv_gamma_u = 1.0f / gamma_u;
    float coeff_u_part = (c * gamma_u * uv_dot) / (1.0f + gamma_u);
    float coeff_u = 1.0f + coeff_u_part;
    
    for (int i = 0; i < dim; ++i) {
        out[i] = denom_inv * (coeff_u * u[i] + inv_gamma_u * v[i]);
    }
}

__global__ void klein_layer_forward_kernel(
    float* out, const float* u, const float* v, 
    float c, float t, long long batch_size, long long dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* u_row = u + idx * dim;
    const float* v_row = v + idx * dim;
    float* out_row = out + idx * dim;
    
    float u_prime[256];
    float v_prime[256];
    
    if (dim > 256) return;
    
    klein_scalar_impl(u_row, u_prime, dim, c, 1.0f - t);
    klein_scalar_impl(v_row, v_prime, dim, c, t);
    klein_add_impl(u_prime, v_prime, out_row, dim, c);
}

__device__ void klein_scalar_vjp_impl(
    const float* grad_output_prime, const float* x, 
    float c, float r, float* grad_x, int dim
) {
    float x_norm_sq = norm_sq(x, dim);
    float x_norm = safe_sqrt(x_norm_sq);
    float x_norm_clamped = fmaxf(x_norm, KLEIN_EPS);
    
    float boundary = 1.0f / safe_sqrt(c) - BOUNDARY_EPS;
    float scaled_norm = fminf(r * x_norm_clamped, boundary);
    float scale = scaled_norm / x_norm_clamped;
    
    float rn = r * x_norm_clamped;
    float d_scale_d_norm = (rn < boundary) ? 0.0f : -1.0f / fmaxf(x_norm_clamped * x_norm_clamped, KLEIN_EPS);
    
    float grad_norm_component = 0.0f;
    for (int i = 0; i < dim; ++i) {
        grad_norm_component += grad_output_prime[i] * x[i];
    }
    
    for (int i = 0; i < dim; ++i) {
        grad_x[i] = grad_output_prime[i] * scale 
                  + (grad_norm_component * d_scale_d_norm / x_norm_clamped) * x[i];
    }
}

__device__ void klein_add_vjp_impl(
    const float* grad_output, const float* u, const float* v,
    float c, float* grad_u, float* grad_v, int dim
) {
    float u_norm_sq = norm_sq(u, dim);
    float v_norm_sq = norm_sq(v, dim);
    float uv = dot_product(u, v, dim);
    
    float gamma_u = 1.0f / safe_sqrt(1.0f - c * u_norm_sq);
    float denom = fmaxf(1.0f + c * uv, KLEIN_EPS);
    float denom_inv = 1.0f / denom;
    
    float inv_gamma_u = 1.0f / gamma_u;
    float coeff_u_part = (c * gamma_u * uv) / (1.0f + gamma_u);
    float coeff_u = 1.0f + coeff_u_part;
    
    float output_dot_grad = 0.0f;
    for (int j = 0; j < dim; ++j) {
        float out_j = denom_inv * (coeff_u * u[j] + inv_gamma_u * v[j]);
        output_dot_grad += out_j * grad_output[j];
    }
    
    float grad_denom = -output_dot_grad * denom_inv;
    
    for (int j = 0; j < dim; ++j) {
        float grad_num_u = coeff_u * grad_output[j] * denom_inv;
        float grad_num_v = inv_gamma_u * grad_output[j] * denom_inv;
        
        grad_u[j] = grad_num_u + c * grad_denom * v[j];
        grad_v[j] = grad_num_v + c * grad_denom * u[j];
    }
    
    float grad_coeff_u = 0.0f;
    float grad_inv_gamma_u = 0.0f;
    for (int j = 0; j < dim; ++j) {
        grad_coeff_u += (grad_output[j] * denom_inv) * u[j];
        grad_inv_gamma_u += (grad_output[j] * denom_inv) * v[j];
    }
    
    for (int j = 0; j < dim; ++j) {
        grad_u[j] -= u[j] * (grad_inv_gamma_u * c * gamma_u);
    }
    
    float d_coeff_u_d_uv = c * gamma_u / (1.0f + gamma_u);
    float d_coeff_u_d_gamma_u = (c * uv) / ((1.0f + gamma_u) * (1.0f + gamma_u));
    
    float grad_uv = grad_coeff_u * d_coeff_u_d_uv;
    float grad_gamma_u = grad_coeff_u * d_coeff_u_d_gamma_u;
    
    for (int j = 0; j < dim; ++j) {
        grad_u[j] += grad_uv * v[j] + grad_gamma_u * c * gamma_u * gamma_u * gamma_u * u[j];
        grad_v[j] += grad_uv * u[j];
    }
}

__global__ void klein_layer_backward_kernel(
    const float* grad_output, const float* u, const float* v,
    float* grad_u, float* grad_v,
    float c, float t, long long batch_size, long long dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    if (dim > 256) return;
    
    const float* u_row = u + idx * dim;
    const float* v_row = v + idx * dim;
    const float* grad_out = grad_output + idx * dim;
    float* gu = grad_u + idx * dim;
    float* gv = grad_v + idx * dim;
    
    float u_prime[256];
    float v_prime[256];
    float grad_u_prime[256];
    float grad_v_prime[256];
    
    klein_scalar_impl(u_row, u_prime, dim, c, 1.0f - t);
    klein_scalar_impl(v_row, v_prime, dim, c, t);
    
    klein_add_vjp_impl(grad_out, u_prime, v_prime, c, grad_u_prime, grad_v_prime, dim);
    
    klein_scalar_vjp_impl(grad_u_prime, u_row, c, 1.0f - t, gu, dim);
    klein_scalar_vjp_impl(grad_v_prime, v_row, c, t, gv, dim);
}

extern "C" {
    void klein_distance_cuda(
        float* out, const float* u, const float* v, 
        float c, long long batch_size, long long dim
    ) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        klein_distance_kernel<<<blocks, threads>>>(out, u, v, c, batch_size, dim);
        cudaDeviceSynchronize();
    }
    
    void klein_layer_forward_cuda(
        float* out, const float* u, const float* v, 
        float c, float t, long long batch_size, long long dim
    ) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        klein_layer_forward_kernel<<<blocks, threads>>>(out, u, v, c, t, batch_size, dim);
        cudaDeviceSynchronize();
    }
    
    void klein_layer_backward_cuda(
        const float* grad_output, const float* u, const float* v,
        float* grad_u, float* grad_v,
        float c, float t, long long batch_size, long long dim
    ) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        klein_layer_backward_kernel<<<blocks, threads>>>(
            grad_output, u, v, grad_u, grad_v, c, t, batch_size, dim
        );
        cudaDeviceSynchronize();
    }
}

