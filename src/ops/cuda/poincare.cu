#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>

#define EPS 1e-7f

__global__ void poincare_distance_kernel(float* out, const float* u, const float* v, float c, int batch_size, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size) {
        const float* u_row = u + i * dim;
        const float* v_row = v + i * dim;

        float u2 = 0.0f;
        float v2 = 0.0f;
        float uv = 0.0f;
        float norm_sq_diff = 0.0f;

        for (int j = 0; j < dim; ++j) {
            float u_val = u_row[j];
            float v_val = v_row[j];
            u2 += u_val * u_val;
            v2 += v_val * v_val;
            uv += u_val * v_val;
        }
        
        norm_sq_diff = u2 + v2 - 2.0f * uv;
        if (norm_sq_diff < EPS) {
            norm_sq_diff = EPS;
        }
        
        float denom = (1.0f - c * u2) * (1.0f - c * v2);
        if (denom < EPS) {
            denom = EPS;
        }
        
        float sqrtc = sqrtf(c);
        float arg = sqrtf((c * norm_sq_diff) / denom);
        
        out[i] = (2.0f / sqrtc) * atanhf(arg);
    }
}

extern "C" {
    void poincare_distance_cuda_launcher(float* out, const float* u, const float* v, float c, int64_t batch_size, int64_t dim) {
        int threads_per_block = 256;
        int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
        poincare_distance_kernel<<<blocks_per_grid, threads_per_block>>>(out, u, v, c, batch_size, dim);
    }
}

__device__ void calculate_mobius_scalar(float* out_vec, const float* in_vec, float c, float r, int dim) {
    float norm_sq = 0.0f;
    for (int j = 0; j < dim; ++j) {
        norm_sq += in_vec[j] * in_vec[j];
    }

    if (norm_sq < EPS * EPS) {
        for (int j = 0; j < dim; ++j) {
            out_vec[j] = r * in_vec[j];
        }
        return;
    }

    float norm = sqrtf(norm_sq);
    float sqrtc = sqrtf(c);
    float scn = sqrtc * norm;
    if (scn > 1.0f - EPS) scn = 1.0f - EPS;
    
    float alpha = atanhf(scn);
    float beta = tanhf(r * alpha);
    float scale = beta / (sqrtc * norm);

    for (int j = 0; j < dim; ++j) {
        out_vec[j] = scale * in_vec[j];
    }
}

__device__ void calculate_mobius_add(float* out_vec, const float* u_vec, const float* v_vec, float c, int dim) {
    float u2 = 0.0f;
    float v2 = 0.0f;
    float uv = 0.0f;
    for (int j = 0; j < dim; ++j) {
        u2 += u_vec[j] * u_vec[j];
        v2 += v_vec[j] * v_vec[j];
        uv += u_vec[j] * v_vec[j];
    }

    float denom = 1.0f + 2.0f * c * uv + c * c * u2 * v2;
    if (denom < EPS) denom = EPS;
    
    float coeff_u = (1.0f + 2.0f * c * uv + c * v2) / denom;
    float coeff_v = (1.0f - c * u2) / denom;

    for (int j = 0; j < dim; ++j) {
        out_vec[j] = coeff_u * u_vec[j] + coeff_v * v_vec[j];
    }
}

__device__ void project_to_ball_device(float* x, float c, int dim) {
    float norm_sq = 0.0f;
    for (int j = 0; j < dim; ++j) {
        norm_sq += x[j] * x[j];
    }
    
    float max_norm_sq = (1.0f / c) - EPS;
    if (norm_sq >= max_norm_sq) {
        float scale = sqrtf(max_norm_sq / norm_sq) - EPS;
        for (int j = 0; j < dim; ++j) {
            x[j] *= scale;
        }
    }
}

__global__ void poincare_ball_layer_kernel(float* out, const float* u, const float* v, float c, float t, int batch_size, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    const float* u_row = u + i * dim;
    const float* v_row = v + i * dim;
    float* out_row = out + i * dim;

    float u_prime[256]; // Max dim 256
    float v_prime[256];

    calculate_mobius_scalar(u_prime, u_row, c, 1.0f - t, dim);
    calculate_mobius_scalar(v_prime, v_row, c, t, dim);
    calculate_mobius_add(out_row, u_prime, v_prime, c, dim);
}

extern "C" {
    void poincare_ball_layer_cuda_launcher(float* out, const float* u, const float* v, float c, float t, int64_t batch_size, int64_t dim) {
        int threads_per_block = 256;
        int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
        poincare_ball_layer_kernel<<<blocks_per_grid, threads_per_block>>>(out, u, v, c, t, batch_size, dim);
    }
} 