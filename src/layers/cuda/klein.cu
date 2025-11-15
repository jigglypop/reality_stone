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
// Klein distance: d_K(u,v) = (1/√c) * acosh((1 - c⟨u,v⟩) / √((1-c||u||²)(1-c||v||²)))
__global__ void klein_distance_kernel(float* out, const float* u, const float* v, float c, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* u_row = u + idx * dim;
    const float* v_row = v + idx * dim;
    
    float u_norm_sq = norm_sq(u_row, dim);
    float v_norm_sq = norm_sq(v_row, dim);
    float uv_dot = dot(u_row, v_row, dim);

    // 표준 Klein distance 공식
    float numerator = 1.0f - c * uv_dot;
    float denominator = sqrtf(fmaxf((1.0f - c * u_norm_sq) * (1.0f - c * v_norm_sq), EPS));
    float arg = fmaxf(numerator / denominator, 1.0f + EPS);
    
    out[idx] = acoshf(arg) / sqrtf(c);
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

// Klein Layer Backward CUDA Kernel (matches CPU klein_add_vjp + klein_scalar_vjp)
__global__ void klein_layer_backward_kernel(
    const float* grad_output, const float* u, const float* v,
    float* grad_u, float* grad_v,
    float c, float t, int batch_size, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* u_row = u + idx * dim;
    const float* v_row = v + idx * dim;
    const float* g_row = grad_output + idx * dim;
    float* gu_row = grad_u + idx * dim;
    float* gv_row = grad_v + idx * dim;

    // 1) u' = scale_u * u, v' = scale_v * v with scale via Klein scalar
    auto norm = [&](const float* x) {
        float s = 0.0f; for (int j=0;j<dim;++j) s += x[j]*x[j]; return sqrtf(fmaxf(s, EPS));
    };
    float norm_u = norm(u_row);
    float norm_v = norm(v_row);
    float inv_norm_u = (norm_u < EPS) ? 0.0f : (1.0f / norm_u);
    float inv_norm_v = (norm_v < EPS) ? 0.0f : (1.0f / norm_v);
    float radius = 1.0f / sqrtf(c) - BOUNDARY_EPS;
    float scaled_u = fminf((1.0f - t) * norm_u, radius);
    float scaled_v = fminf(t * norm_v, radius);
    float scale_u = (norm_u < EPS) ? (1.0f - t) : (scaled_u * inv_norm_u);
    float scale_v = (norm_v < EPS) ? t : (scaled_v * inv_norm_v);

    // u' and v' (stack arrays)
    const int MAX_DIM = 1024;
    float uprime[MAX_DIM];
    float vprime[MAX_DIM];
    for (int j=0;j<dim && j<MAX_DIM;++j) { uprime[j] = u_row[j] * scale_u; vprime[j] = v_row[j] * scale_v; }

    // 2) y = klein_add(u', v') = temp / denom, with temp = u'/den_u + v'/den_v
    auto sq = [&](const float* x){ float s=0.0f; for(int j=0;j<dim;++j) s+=x[j]*x[j]; return s; };
    float u2 = sq(uprime), v2 = sq(vprime);
    float den_u = sqrtf(fmaxf(1.0f - c * u2, EPS));
    float den_v = sqrtf(fmaxf(1.0f - c * v2, EPS));
    // temp
    float temp[MAX_DIM];
    for (int j=0;j<dim && j<MAX_DIM;++j) temp[j] = uprime[j] / den_u + vprime[j] / den_v;
    float temp2 = sq(temp);
    float result_denom_inner_sqrt = sqrtf(fmaxf(1.0f + c * temp2, EPS));
    float result_denom = fmaxf(1.0f + result_denom_inner_sqrt, EPS);

    // 3) grad wrt temp and denoms (from CPU klein_add_vjp)
    float gtemp[MAX_DIM];
    for (int j=0;j<dim && j<MAX_DIM;++j) gtemp[j] = g_row[j] / result_denom;
    float dot_g_y_over_den = 0.0f; // -(g · y) / result_denom
    for (int j=0;j<dim && j<MAX_DIM;++j) dot_g_y_over_den -= g_row[j] * (temp[j] / result_denom);
    float grad_temp_norm_sq = dot_g_y_over_den * (c / (2.0f * result_denom_inner_sqrt));
    for (int j=0;j<dim && j<MAX_DIM;++j) gtemp[j] += 2.0f * grad_temp_norm_sq * temp[j];

    // grad wrt u' and v' from temp
    float gu_prime[MAX_DIM];
    float gv_prime[MAX_DIM];
    // dot(gtemp, u') and dot(gtemp, v')
    float s_u = 0.0f, s_v = 0.0f;
    for (int j=0;j<dim && j<MAX_DIM;++j) { s_u += gtemp[j] * uprime[j]; s_v += gtemp[j] * vprime[j]; }
    float add_u = (c / (den_u * den_u * den_u)) * s_u;
    float add_v = (c / (den_v * den_v * den_v)) * s_v;
    for (int j=0;j<dim && j<MAX_DIM;++j) {
        gu_prime[j] = gtemp[j] / den_u + add_u * uprime[j];
        gv_prime[j] = gtemp[j] / den_v + add_v * vprime[j];
    }

    // 4) back through scalar: x' = scale * x, scale depends on ||x|| and clamp
    auto back_scalar = [&](const float* x, const float* gxp, float norm_x, float scale, float inv_norm, float rparam, float* gx){
        // d scale / d norm: piecewise due to clamp at radius
        float boundary = 1.0f / sqrtf(c) - BOUNDARY_EPS;
        float rn = rparam * norm_x;
        float dscale_dnorm = (rn < boundary) ? 0.0f : (-1.0f / fmaxf(norm_x * norm_x, EPS));
        // grad wrt x: g = gxp * scale + ( (gxp·x) * dscale_dnorm / norm ) * x
        float gdotx = 0.0f; for (int j=0;j<dim;++j) gdotx += gxp[j] * x[j];
        for (int j=0;j<dim;++j) gx[j] = gxp[j] * scale + (gdotx * dscale_dnorm * inv_norm) * x[j];
    };

    back_scalar(u_row, gu_prime, norm_u, scale_u, inv_norm_u, (1.0f - t), gu_row);
    back_scalar(v_row, gv_prime, norm_v, scale_v, inv_norm_v, t, gv_row);
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