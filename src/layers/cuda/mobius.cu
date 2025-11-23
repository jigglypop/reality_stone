#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>

#include "mobius_common.cuh"

#define MIN_DENOMINATOR 1e-6f
#define EPS 1e-7f
#define BOUNDARY_EPS 1e-5f

__global__ void mobius_add_kernel(float* out, const float* u, const float* v, float c, int batch_size, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size) {
        const float* u_row = u + i * dim;
        const float* v_row = v + i * dim;
        float* out_row = out + i * dim;
        mobius_add_point(u_row, v_row, out_row, dim, c, MIN_DENOMINATOR);
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
        mobius_scalar_point(u_row, out_row, dim, c, r, EPS, BOUNDARY_EPS);
    }
}

extern "C" {
    void mobius_scalar_cuda(float* out, const float* u, float c, float r, int64_t batch_size, int64_t dim) {
        int threads_per_block = 256;
        int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
        mobius_scalar_kernel<<<blocks_per_grid, threads_per_block>>>(out, u, c, r, batch_size, dim);
    }
} 