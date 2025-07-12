#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

__device__ void mobius_scalar_kernel_impl(const float* x, float* out, int dim, float c, float r, float eps) {
    float x_norm_sq = 0;
    for (int i = 0; i < dim; ++i) {
        x_norm_sq += x[i] * x[i];
    }
    float x_norm = fmaxf(sqrtf(x_norm_sq), eps);
    
    if (fabsf(c) < eps) {
        // c = 0: 유클리드 경우
        for (int i = 0; i < dim; ++i) {
            out[i] = r * x[i];
        }
        return;
    }
    
    float scale;
    if (c > 0.0f) {
        // 양수 곡률
        float sqrt_c = sqrtf(c);
        float scn = fminf(sqrt_c * x_norm, 1.0f - eps);
        float alpha = atanhf(scn);
        float beta = tanhf(r * alpha);
        scale = beta / (sqrt_c * x_norm);
    } else {
        // 음수 곡률
        float sqrt_abs_c = sqrtf(-c);
        float scn = sqrt_abs_c * x_norm;
        float alpha = atanf(scn);
        float beta = tanf(r * alpha);
        scale = beta / (sqrt_abs_c * x_norm);
    }

    for (int i = 0; i < dim; ++i) {
        out[i] = scale * x[i];
    }
}

__device__ void mobius_add_kernel_impl(const float* x, const float* y, float* out, int dim, float c, float eps) {
    float x2 = 0, y2 = 0, xy = 0;
    for (int i = 0; i < dim; ++i) {
        x2 += x[i] * x[i];
        y2 += y[i] * y[i];
        xy += x[i] * y[i];
    }
    float den = 1.0f + 2.0f * c * xy + c * c * x2 * y2;
    den = fmaxf(den, eps);
    float factor_x = (1.0f + 2.0f * c * xy + c * y2);
    float factor_y = (1.0f - c * x2);
    for (int i = 0; i < dim; ++i) {
        out[i] = (factor_x * x[i] + factor_y * y[i]) / den;
    }
}

__global__ void poincare_ball_layer_forward_kernel(const float* u, const float* v, float* out, float c, float t, long long batch_size, long long dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;
    
    const float* u_i = u + i * dim;
    const float* v_i = v + i * dim;
    float* out_i = out + i * dim;

    float u_prime[256]; // Max dim 256
    float v_prime[256];
    
    mobius_scalar_kernel_impl(u_i, u_prime, dim, c, 1.0f - t, 1e-7f);
    mobius_scalar_kernel_impl(v_i, v_prime, dim, c, t, 1e-7f);
    mobius_add_kernel_impl(u_prime, v_prime, out_i, dim, c, 1e-7f);
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
        // c = 0: 유클리드 경우
        for (int i = 0; i < dim; ++i) {
            grad_x[i] = r * grad_output_prime[i];
        }
        return;
    }
    
    float scale;
    float grad_scale_factor;
    
    if (c > 0.0f) {
        // 양수 곡률
        float sqrt_c = sqrtf(c);
        float scn = fminf(sqrt_c * x_norm, 1.0f - eps);
        float alpha = atanhf(scn);
        float beta = tanhf(r * alpha);
        scale = beta / (sqrt_c * x_norm);
        
        float inner_deriv_atanh = r * (1.0f - beta * beta);
        float inner_deriv_norm = (1.0f / fmaxf(1.0f - scn * scn, eps)) * (sqrt_c / x_norm);
        grad_scale_factor = inner_deriv_atanh * inner_deriv_norm / (sqrt_c * x_norm) - scale / x_norm;
    } else {
        // 음수 곡률
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

__device__ float poincare_dist_sq(const float* x, const float* y, int dim, float c, float eps) {
    float xy = 0, x2 = 0, y2 = 0;
    for (int i = 0; i < dim; ++i) {
        xy += (x[i] - y[i]) * (x[i] - y[i]);
        x2 += x[i] * x[i];
        y2 += y[i] * y[i];
    }
    float num = 2 * c * xy;
    float den = (1 - c * x2) * (1 - c * y2);
    den = fmaxf(den, eps);
    return acoshf(1.0f + num / den);
}

__global__ void poincare_distance_kernel(const float* x, const float* y, float* out, int batch_size, int dim, float c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    const float* x_i = x + i * dim;
    const float* y_i = y + i * dim;
    
    float dist_sq = poincare_dist_sq(x_i, y_i, dim, c, 1e-7f);
    out[i] = dist_sq / sqrtf(c);
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
    mobius_scalar_kernel_impl(u_i, u_prime, dim, c, 1.0f - t, 1e-7f);
    mobius_scalar_kernel_impl(v_i, v_prime, dim, c, t, 1e-7f);

    float grad_u_prime[256], grad_v_prime[256];
    mobius_add_vjp(grad_output_i, u_prime, v_prime, c, grad_u_prime, grad_v_prime, dim, 1e-7f);
    
    mobius_scalar_vjp(grad_u_prime, u_i, c, 1.0f - t, grad_u_i, dim, 1e-7f);
    mobius_scalar_vjp(grad_v_prime, v_i, c, t, grad_v_i, dim, 1e-7f);
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

    void poincare_distance_cuda(float* out, const float* x, const float* y, float c, long long batch_size, long long dim) {
        dim3 threads_per_block(256);
        dim3 num_blocks((batch_size + threads_per_block.x - 1) / threads_per_block.x);
        poincare_distance_kernel<<<num_blocks, threads_per_block>>>(x, y, out, batch_size, dim, c);
    }
} 