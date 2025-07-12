#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>

#define MIN_DENOMINATOR 1e-6f
#define EPS 1e-7f
#define BOUNDARY_EPS 1e-5f

__global__ void mobius_add_kernel(float* out, const float* u, const float* v, float c, int batch_size, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size) {
        const float* u_row = u + i * dim;
        const float* v_row = v + i * dim;
        float* out_row = out + i * dim;

        float u2 = 0.0f;
        float v2 = 0.0f;
        float uv = 0.0f;

        for (int j = 0; j < dim; ++j) {
            u2 += u_row[j] * u_row[j];
            v2 += v_row[j] * v_row[j];
            uv += u_row[j] * v_row[j];
        }

        float c2 = c * c;
        float denominator = 1.0f + 2.0f * c * uv + c2 * u2 * v2;
        if (denominator < MIN_DENOMINATOR) {
            denominator = MIN_DENOMINATOR;
        }

        float coeff_u = (1.0f + 2.0f * c * uv + c * v2) / denominator;
        float coeff_v = (1.0f - c * u2) / denominator;

        for (int j = 0; j < dim; ++j) {
            out_row[j] = coeff_u * u_row[j] + coeff_v * v_row[j];
        }
    }
}

extern "C" {
    void mobius_add_cuda(float* out, const float* u, const float* v, float c, int64_t batch_size, int64_t dim) {
        int threads_per_block = 256;
        int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
        mobius_add_kernel<<<blocks_per_grid, threads_per_block>>>(out, u, v, c, batch_size, dim);
    }
}

// --- Mobius Scalar Multiplication ---

__global__ void mobius_scalar_kernel(float* out, const float* u, float c, float r, int batch_size, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size) {
        const float* u_row = u + i * dim;
        float* out_row = out + i * dim;

        float norm_sq = 0.0f;
        for (int j = 0; j < dim; ++j) {
            norm_sq += u_row[j] * u_row[j];
        }
        
        if (norm_sq < EPS * EPS) {
            // 작은 벡터는 그대로 복사 (gradient flow 유지)
            for (int j = 0; j < dim; ++j) {
                out_row[j] = r * u_row[j];
            }
            return;
        }

        float norm = sqrtf(norm_sq);
        float sqrtc = sqrtf(c);
        float scn = sqrtc * norm;
        if (scn > 1.0f - BOUNDARY_EPS) {
            scn = 1.0f - BOUNDARY_EPS;
        }
        
        float alpha = atanhf(scn);
        float beta = tanhf(r * alpha);
        float scale = beta / (sqrtc * norm);

        for (int j = 0; j < dim; ++j) {
            out_row[j] = scale * u_row[j];
        }
    }
}

extern "C" {
    void mobius_scalar_cuda(float* out, const float* u, float c, float r, int64_t batch_size, int64_t dim) {
        int threads_per_block = 256;
        int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
        mobius_scalar_kernel<<<blocks_per_grid, threads_per_block>>>(out, u, c, r, batch_size, dim);
    }
} 