#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define KLEIN_EPS 1e-7f
#define KLEIN_BOUNDARY_EPS 1e-5f

namespace {

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
    float denominator = sqrtf(fmaxf((1.0f - c * u_norm_sq) * (1.0f - c * v_norm_sq), KLEIN_EPS));
    float arg = fmaxf(numerator / denominator, 1.0f + KLEIN_EPS);
    
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
    float u_norm_clamped = fmaxf(u_norm, KLEIN_EPS);
    float u_scaled_norm = fminf(u_norm_clamped * (1.0f - t), 1.0f/sqrtf(c) - KLEIN_BOUNDARY_EPS);
    float u_scale = u_scaled_norm / u_norm_clamped;
    for(int i=0; i<dim; ++i) u_prime[i] = u_row[i] * u_scale;

    // Scalar Mul for v
    float v_norm = sqrtf(norm_sq(v_row, dim));
    float v_norm_clamped = fmaxf(v_norm, KLEIN_EPS);
    float v_scaled_norm = fminf(v_norm_clamped * t, 1.0f/sqrtf(c) - KLEIN_BOUNDARY_EPS);
    float v_scale = v_scaled_norm / v_norm_clamped;
    for(int i=0; i<dim; ++i) v_prime[i] = v_row[i] * v_scale;
    
    // Klein Add
    float u_prime_norm_sq = norm_sq(u_prime, dim);
    float v_prime_norm_sq = norm_sq(v_prime, dim);
    float u_denom = sqrtf(fmaxf(1.0f - c * u_prime_norm_sq, KLEIN_EPS));
    float v_denom = sqrtf(fmaxf(1.0f - c * v_prime_norm_sq, KLEIN_EPS));

    float temp[1024];
    for(int i=0; i<dim; ++i) temp[i] = u_prime[i] / u_denom + v_prime[i] / v_denom;

    float temp_norm_sq = norm_sq(temp, dim);
    float res_denom = 1.0f + sqrtf(1.0f + c * temp_norm_sq);
    
    float* out_row = out + idx * dim;
    for(int i=0; i<dim; ++i) out_row[i] = temp[i] / fmaxf(res_denom, KLEIN_EPS);
}

// Klein Layer Backward CUDA Kernel (direct translation of CPU klein_layer_backward + klein_add_vjp + klein_scalar_vjp)
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

    const int MAX_DIM = 1024;

    // -------- 1) Forward Klein scalar for u and v (as in klein_scalar) --------
    // u'
    float u_norm_sq0 = 0.0f;
    for (int j = 0; j < dim && j < MAX_DIM; ++j) {
        u_norm_sq0 += u_row[j] * u_row[j];
    }
    float u_norm0 = sqrtf(fmaxf(u_norm_sq0, KLEIN_EPS));
    float u_norm0_clamped = fmaxf(u_norm0, KLEIN_EPS);

    // v'
    float v_norm_sq0 = 0.0f;
    for (int j = 0; j < dim && j < MAX_DIM; ++j) {
        v_norm_sq0 += v_row[j] * v_row[j];
    }
    float v_norm0 = sqrtf(fmaxf(v_norm_sq0, KLEIN_EPS));
    float v_norm0_clamped = fmaxf(v_norm0, KLEIN_EPS);

    float radius = 1.0f / sqrtf(c) - KLEIN_BOUNDARY_EPS;

    // For u: r = (1 - t)
    float scaled_norm_u = fminf(u_norm0_clamped * (1.0f - t), radius);
    float scale_u = scaled_norm_u / u_norm0_clamped;

    // For v: r = t
    float scaled_norm_v = fminf(v_norm0_clamped * t, radius);
    float scale_v = scaled_norm_v / v_norm0_clamped;

    float uprime[MAX_DIM];
    float vprime[MAX_DIM];
    for (int j = 0; j < dim && j < MAX_DIM; ++j) {
        uprime[j] = u_row[j] * scale_u;
        vprime[j] = v_row[j] * scale_v;
    }

    // -------- 2) klein_add_vjp on u', v' --------
    // Norms and denominators for u', v'
    float u2 = 0.0f, v2 = 0.0f;
    for (int j = 0; j < dim && j < MAX_DIM; ++j) {
        u2 += uprime[j] * uprime[j];
        v2 += vprime[j] * vprime[j];
    }
    float u_denom = sqrtf(fmaxf(1.0f - c * u2, KLEIN_EPS));
    float v_denom = sqrtf(fmaxf(1.0f - c * v2, KLEIN_EPS));

    // temp = u'/u_denom + v'/v_denom
    float temp[MAX_DIM];
    for (int j = 0; j < dim && j < MAX_DIM; ++j) {
        temp[j] = uprime[j] / u_denom + vprime[j] / v_denom;
    }

    // temp_norm_sq
    float temp_norm_sq = 0.0f;
    for (int j = 0; j < dim && j < MAX_DIM; ++j) {
        temp_norm_sq += temp[j] * temp[j];
    }

    float result_denom_inner_sqrt = sqrtf(fmaxf(1.0f + c * temp_norm_sq, KLEIN_EPS));
    float result_denom = 1.0f + result_denom_inner_sqrt;

    // grad_temp_part1 = grad_output / result_denom
    float gtemp_part1[MAX_DIM];
    for (int j = 0; j < dim && j < MAX_DIM; ++j) {
        gtemp_part1[j] = g_row[j] / result_denom;
    }

    // grad_result_denom = -sum(grad_output * temp / result_denom^2)
    float grad_result_denom = 0.0f;
    float denom_sq = result_denom * result_denom;
    for (int j = 0; j < dim && j < MAX_DIM; ++j) {
        grad_result_denom -= g_row[j] * temp[j] / denom_sq;
    }

    // grad_temp_norm_sq = grad_result_denom * c / (2 * result_denom_inner_sqrt)
    float grad_temp_norm_sq = grad_result_denom * c / (2.0f * result_denom_inner_sqrt);

    // grad_temp = grad_temp_part1 + 2 * grad_temp_norm_sq * temp
    float gtemp[MAX_DIM];
    for (int j = 0; j < dim && j < MAX_DIM; ++j) {
        gtemp[j] = gtemp_part1[j] + 2.0f * grad_temp_norm_sq * temp[j];
    }

    // grad_u_from_temp = grad_temp / u_denom, grad_v_from_temp = grad_temp / v_denom
    float grad_u_from_temp[MAX_DIM];
    float grad_v_from_temp[MAX_DIM];
    for (int j = 0; j < dim && j < MAX_DIM; ++j) {
        grad_u_from_temp[j] = gtemp[j] / u_denom;
        grad_v_from_temp[j] = gtemp[j] / v_denom;
    }

    // grad_u_denom = -(grad_temp * u' / u_denom^2).sum
    float grad_u_denom = 0.0f;
    float u_denom_sq = u_denom * u_denom;
    for (int j = 0; j < dim && j < MAX_DIM; ++j) {
        grad_u_denom -= gtemp[j] * uprime[j] / u_denom_sq;
    }

    // grad_v_denom = -(grad_temp * v' / v_denom^2).sum
    float grad_v_denom = 0.0f;
    float v_denom_sq = v_denom * v_denom;
    for (int j = 0; j < dim && j < MAX_DIM; ++j) {
        grad_v_denom -= gtemp[j] * vprime[j] / v_denom_sq;
    }

    // grad_u_norm_sq = grad_u_denom * (-c / (2*u_denom))
    float grad_u_norm_sq = grad_u_denom * (-c / (2.0f * u_denom));
    float grad_v_norm_sq = grad_v_denom * (-c / (2.0f * v_denom));

    // grad_u_prime = grad_u_from_temp + 2 * grad_u_norm_sq * u'
    float grad_u_prime[MAX_DIM];
    float grad_v_prime[MAX_DIM];
    for (int j = 0; j < dim && j < MAX_DIM; ++j) {
        grad_u_prime[j] = grad_u_from_temp[j] + 2.0f * grad_u_norm_sq * uprime[j];
        grad_v_prime[j] = grad_v_from_temp[j] + 2.0f * grad_v_norm_sq * vprime[j];
    }

    // -------- 3) klein_scalar_vjp for u and v --------
    float boundary = 1.0f / sqrtf(c) - KLEIN_BOUNDARY_EPS;

    // For u: r = (1 - t)
    float rn_u = (1.0f - t) * u_norm0_clamped;
    float dscale_dnorm_u = (rn_u < boundary) ? 0.0f : (-1.0f / fmaxf(u_norm0_clamped * u_norm0_clamped, KLEIN_EPS));

    float grad_norm_comp_u = 0.0f;
    for (int j = 0; j < dim && j < MAX_DIM; ++j) {
        grad_norm_comp_u += grad_u_prime[j] * u_row[j];
    }

    for (int j = 0; j < dim && j < MAX_DIM; ++j) {
        gu_row[j] = grad_u_prime[j] * scale_u
            + (grad_norm_comp_u * dscale_dnorm_u / u_norm0_clamped) * u_row[j];
    }

    // For v: r = t
    float rn_v = t * v_norm0_clamped;
    float dscale_dnorm_v = (rn_v < boundary) ? 0.0f : (-1.0f / fmaxf(v_norm0_clamped * v_norm0_clamped, KLEIN_EPS));

    float grad_norm_comp_v = 0.0f;
    for (int j = 0; j < dim && j < MAX_DIM; ++j) {
        grad_norm_comp_v += grad_v_prime[j] * v_row[j];
    }

    for (int j = 0; j < dim && j < MAX_DIM; ++j) {
        gv_row[j] = grad_v_prime[j] * scale_v
            + (grad_norm_comp_v * dscale_dnorm_v / v_norm0_clamped) * v_row[j];
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