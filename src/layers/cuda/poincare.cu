#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#include "mobius_common.cuh"

#define POINCARE_EPS 1e-6f
#define POINCARE_BOUNDARY_EPS 1e-5f
#define POINCARE_MIN_DENOM 1e-6f
#define POINCARE_ATANH_CLAMP 1e-3f

__global__ void poincare_ball_layer_forward_kernel(const float* u, const float* v, float* out, float c, float t, long long batch_size, long long dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;
    
    const float* u_i = u + i * dim;
    const float* v_i = v + i * dim;
    float* out_i = out + i * dim;

    float u_prime[256]; // Max dim 256
    float v_prime[256];
    
    mobius_scalar_point(u_i, u_prime, dim, c, 1.0f - t, POINCARE_EPS, POINCARE_BOUNDARY_EPS);
    mobius_scalar_point(v_i, v_prime, dim, c, t, POINCARE_EPS, POINCARE_BOUNDARY_EPS);
    mobius_add_point(u_prime, v_prime, out_i, dim, c, POINCARE_EPS);
}

// Helper device function for mobius_scalar_vjp
__device__ void mobius_scalar_vjp(
    const float* grad_output_prime, const float* x, float c, float r,
    float* grad_x, int dim, float eps) {

    float x_norm_sq = 0;
    for (int i = 0; i < dim; ++i) {
        x_norm_sq += x[i] * x[i];
    }
    float x_norm = fmaxf(sqrtf(x_norm_sq), eps);
    
    if (fabsf(c) < eps) {
        // c = 0: Euclidean case
        for (int i = 0; i < dim; ++i) {
            grad_x[i] = r * grad_output_prime[i];
        }
        return;
    }
    
    float scale;
    float grad_scale_factor;
    
    if (c > 0.0f) {
        // Positive curvature
        float sqrt_c = sqrtf(c);
        float scn = fminf(sqrt_c * x_norm, 1.0f - eps);
        float alpha = atanhf(scn);
        float beta = tanhf(r * alpha);
        scale = beta / (sqrt_c * x_norm);
        
        float inner_deriv_atanh = r * (1.0f - beta * beta);
        float inner_deriv_norm = (1.0f / fmaxf(1.0f - scn * scn, eps)) * (sqrt_c / x_norm);
        grad_scale_factor = inner_deriv_atanh * inner_deriv_norm / (sqrt_c * x_norm) - scale / x_norm;
    } else {
        // Negative curvature
        float sqrt_abs_c = sqrtf(-c);
        float scn = sqrt_abs_c * x_norm;
        float alpha = atanf(scn);
        float beta = tanf(r * alpha);
        scale = beta / (sqrt_abs_c * x_norm);
        
        float inner_deriv_atan = r * (1.0f + beta * beta);
        float inner_deriv_norm = (1.0f / (1.0f + scn * scn)) * (sqrt_abs_c / x_norm);
        grad_scale_factor = inner_deriv_atan * inner_deriv_norm / (sqrt_abs_c * x_norm) - scale / x_norm;
    }

    float grad_scale = 0;
    for (int i = 0; i < dim; ++i) {
        grad_scale += grad_output_prime[i] * x[i];
    }

    for (int i = 0; i < dim; ++i) {
        grad_x[i] = scale * grad_output_prime[i] + grad_scale_factor * grad_scale * x[i];
    }
}

// Helper device function for mobius_add_vjp
__device__ void mobius_add_vjp(
    const float* grad_output, const float* x, const float* y, float c,
    float* grad_x, float* grad_y, int dim, float eps) {

    float x2 = 0, y2 = 0, xy = 0;
    for(int i=0; i<dim; ++i) {
        x2 += x[i] * x[i];
        y2 += y[i] * y[i];
        xy += x[i] * y[i];
    }

    float den = 1.0f + 2.0f * c * xy + c * c * x2 * y2;
    den = fmaxf(den, eps);

    float u_calc[256]; // Assuming max dim 256
    for(int i=0; i<dim; ++i) {
        u_calc[i] = (1.0f + 2.0f * c * xy + c * y2) * x[i] + (1.0f - c * x2) * y[i];
    }

    float output[256];
    for(int i=0; i<dim; ++i) {
        output[i] = u_calc[i] / den;
    }

    float grad_u[256];
    for(int i=0; i<dim; ++i) {
        grad_u[i] = grad_output[i] / den;
    }

    float grad_den_sum = 0;
    for(int i=0; i<dim; ++i) {
        grad_den_sum -= grad_output[i] * output[i] / den;
    }
    
    float grad_x_from_u[256], grad_y_from_u[256];
    float factor_x = 1.0f + 2.0f * c * xy + c * y2;
    float factor_y = 1.0f - c * x2;
    for(int i=0; i<dim; ++i) {
        grad_x_from_u[i] = grad_u[i] * factor_x;
        grad_y_from_u[i] = grad_u[i] * factor_y;
    }
    
    float grad_xy_from_u = 0, grad_x2_from_u = 0;
    for(int i=0; i<dim; ++i) {
        grad_xy_from_u += 2.0f * c * grad_u[i] * x[i];
        grad_x2_from_u -= c * grad_u[i] * y[i];
    }

    float grad_xy_from_den = 2.0f * c * grad_den_sum;
    float grad_x2_from_den = c * c * y2 * grad_den_sum;
    float grad_y2_from_den = c * c * x2 * grad_den_sum;

    float grad_xy_val = grad_xy_from_u + grad_xy_from_den;
    float grad_x2_val = grad_x2_from_u + grad_x2_from_den;
    float grad_y2_val = grad_y2_from_den;

    for(int i=0; i<dim; ++i) {
        grad_x[i] = grad_x_from_u[i] + 2.0f * grad_x2_val * x[i] + grad_xy_val * y[i];
        grad_y[i] = grad_y_from_u[i] + 2.0f * grad_y2_val * y[i] + grad_xy_val * x[i];
    }
}

__device__ float poincare_distance_impl(const float* x, const float* y, int dim, float c, float eps, float boundary_eps) {
    // Poincare distance: d = (2/sqrt(c)) * atanh(sqrt(c * ||x-y||^2 / ((1-c||x||^2)(1-c||y||^2))))
    float norm_sq_diff = 0.0f;  // ||x-y||²
    float x2 = 0.0f;            // ||x||²
    float y2 = 0.0f;            // ||y||²
    
    for (int i = 0; i < dim; ++i) {
        float diff = x[i] - y[i];
        norm_sq_diff += diff * diff;
        x2 += x[i] * x[i];
        y2 += y[i] * y[i];
    }
    
    // frac = c * ||x-y||^2 / ((1-c||x||^2)(1-c||y||^2))
    float den = (1.0f - c * x2) * (1.0f - c * y2);
    // Increased denominator clamp for numerical stability near boundary
    den = fmaxf(den, boundary_eps);
    float frac = (c * norm_sq_diff) / den;
    frac = fmaxf(frac, 0.0f);
    
    // d = (2/sqrt(c)) * atanh(sqrt(frac / (1 + frac)))
    float sqrtc = sqrtf(c);
    float arg = sqrtf(frac / (1.0f + frac));
    // More conservative atanh domain restriction
    arg = fminf(arg, 1.0f - boundary_eps);
    
    return (2.0f / sqrtc) * atanhf(arg);
}

    __global__ void poincare_distance_kernel(const float* x, const float* y, float* out, int batch_size, int dim, float c, float boundary_eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    const float* x_i = x + i * dim;
    const float* y_i = y + i * dim;
    
    out[i] = poincare_distance_impl(x_i, y_i, dim, c, POINCARE_EPS, boundary_eps);
}


// Backward Kernel for Poincare Ball Layer
__global__ void poincare_ball_layer_backward_kernel(
    const float* grad_output, const float* u, const float* v,
    float* grad_u, float* grad_v, float c, float t, long long batch_size, long long dim) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;
    
    const float* u_i = u + i * dim;
    const float* v_i = v + i * dim;
    const float* grad_output_i = grad_output + i * dim;
    float* grad_u_i = grad_u + i * dim;
    float* grad_v_i = grad_v + i * dim;

    float u_prime[256], v_prime[256];
    mobius_scalar_point(u_i, u_prime, dim, c, 1.0f - t, POINCARE_EPS, POINCARE_BOUNDARY_EPS);
    mobius_scalar_point(v_i, v_prime, dim, c, t, POINCARE_EPS, POINCARE_BOUNDARY_EPS);

    float grad_u_prime[256], grad_v_prime[256];
    mobius_add_vjp(grad_output_i, u_prime, v_prime, c, grad_u_prime, grad_v_prime, dim, POINCARE_EPS);
    
    mobius_scalar_vjp(grad_u_prime, u_i, c, 1.0f - t, grad_u_i, dim, POINCARE_EPS);
    mobius_scalar_vjp(grad_v_prime, v_i, c, t, grad_v_i, dim, POINCARE_EPS);
}

__device__ void exp_map_poincare_point(
    const float* x,
    const float* v,
    float* out,
    int dim,
    float c,
    float eps) {
    float x_norm_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float xi = x[i];
        x_norm_sq += xi * xi;
    }
    float v_norm_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float vi = v[i];
        v_norm_sq += vi * vi;
    }
    float v_norm = sqrtf(fmaxf(v_norm_sq, eps));
    if (fabsf(c) < eps) {
        for (int i = 0; i < dim; ++i) {
            out[i] = x[i] + v[i];
        }
        return;
    }
    float one_minus_cx2 = fmaxf(1.0f - c * x_norm_sq, eps);
    float lambda = 2.0f / one_minus_cx2;
    float sqrt_c = sqrtf(fabsf(c));
    float arg = 0.5f * lambda * sqrt_c * v_norm;
    float beta;
    if (c > 0.0f) {
        // Correct formula: beta = tanh(arg)
        // No atanh/clamp needed here as arg is in real domain
        beta = tanhf(arg);
    } else {
        // Correct formula: beta = tan(arg)
        beta = tanf(arg);
    }
    float scale = beta / (sqrt_c * v_norm);
    float u_temp[256];
    for (int i = 0; i < dim; ++i) {
        u_temp[i] = scale * v[i];
    }
    mobius_add_point(x, u_temp, out, dim, c, eps);
}

__global__ void poincare_riemannian_adam_kernel(
    float* x,
    const float* grad,
    float* m,
    float* v,
    float c,
    float lr,
    float beta1,
    float beta2,
    float eps,
    long long batch_size,
    long long dim,
    long long step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;
    float* x_i = x + i * dim;
    const float* g_i = grad + i * dim;
    float* m_i = m + i * dim;
    float* v_i = v + i * dim;
    float g_r[256];
    if (fabsf(c) < eps) {
        for (int j = 0; j < dim; ++j) {
            g_r[j] = g_i[j];
        }
    } else {
        float x_norm_sq = 0.0f;
        for (int j = 0; j < dim; ++j) {
            float xi = x_i[j];
            x_norm_sq += xi * xi;
        }
        float one_minus_cx2 = fmaxf(1.0f - c * x_norm_sq, eps);
        float lambda = 2.0f / one_minus_cx2;
        float inv_lambda_sq = 1.0f / (lambda * lambda);
        for (int j = 0; j < dim; ++j) {
            g_r[j] = inv_lambda_sq * g_i[j];
        }
    }
    float one_minus_b1 = 1.0f - beta1;
    float one_minus_b2 = 1.0f - beta2;
    for (int j = 0; j < dim; ++j) {
        float mj = m_i[j];
        float vj = v_i[j];
        float gr = g_r[j];
        mj = beta1 * mj + one_minus_b1 * gr;
        vj = beta2 * vj + one_minus_b2 * gr * gr;
        m_i[j] = mj;
        v_i[j] = vj;
    }
    float t = (float)step;
    float bias_c1 = 1.0f - powf(beta1, t);
    float bias_c2 = 1.0f - powf(beta2, t);
    float u[256];
    for (int j = 0; j < dim; ++j) {
        float m_hat = m_i[j] / bias_c1;
        float v_hat = v_i[j] / bias_c2;
        u[j] = -lr * m_hat / (sqrtf(v_hat) + eps);
    }
    float x_new[256];
    exp_map_poincare_point(x_i, u, x_new, dim, c, eps);
    float radius = c > 0.0f ? 1.0f / sqrtf(c) : 1.0f;
    float max_norm = radius - POINCARE_BOUNDARY_EPS;
    float norm_sq = 0.0f;
    for (int j = 0; j < dim; ++j) {
        norm_sq += x_new[j] * x_new[j];
    }
    float norm = sqrtf(fmaxf(norm_sq, eps));
    float scale = 1.0f;
    if (norm > max_norm) {
        scale = max_norm / norm;
    }
    for (int j = 0; j < dim; ++j) {
        x_i[j] = x_new[j] * scale;
    }
}



extern "C" {
    void poincare_ball_layer_cuda(float* out, const float* u, const float* v, float c, float t, long long batch_size, long long dim) {
        dim3 threads_per_block(256);
        dim3 num_blocks((batch_size + threads_per_block.x - 1) / threads_per_block.x);
        poincare_ball_layer_forward_kernel<<<num_blocks, threads_per_block>>>(u, v, out, c, t, batch_size, dim);
    }
    
    void poincare_ball_layer_backward_cuda(
        const float* grad_output, const float* u, const float* v,
        float* grad_u, float* grad_v, float c, float t, long long batch_size, long long dim) {
        
        dim3 threads_per_block(256);
        dim3 num_blocks((batch_size + threads_per_block.x - 1) / threads_per_block.x);
        poincare_ball_layer_backward_kernel<<<num_blocks, threads_per_block>>>(
            grad_output, u, v, grad_u, grad_v, c, t, batch_size, dim);
    }

    void poincare_distance_cuda(float* out, const float* x, const float* y, float c, float boundary_eps, long long batch_size, long long dim) {
        dim3 threads_per_block(256);
        dim3 num_blocks((batch_size + threads_per_block.x - 1) / threads_per_block.x);
        poincare_distance_kernel<<<num_blocks, threads_per_block>>>(x, y, out, batch_size, dim, c, boundary_eps);
    }

    void poincare_riemannian_adam_cuda(
        float* x,
        const float* grad,
        float* m,
        float* v,
        float c,
        float lr,
        float beta1,
        float beta2,
        float eps,
        long long batch_size,
        long long dim,
        long long step) {
        dim3 threads_per_block(256);
        dim3 num_blocks((batch_size + threads_per_block.x - 1) / threads_per_block.x);
        poincare_riemannian_adam_kernel<<<num_blocks, threads_per_block>>>(
            x,
            grad,
            m,
            v,
            c,
            lr,
            beta1,
            beta2,
            eps,
            batch_size,
            dim,
            step);
    }
} 