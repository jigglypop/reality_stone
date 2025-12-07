#include <cuda_runtime.h>

extern "C" __global__ void laplace_beltrami_apply_kernel(
    const float* __restrict__ lap,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n,
    int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * d;
    if (idx >= total) return;
    int i = idx / d;
    int k = idx % d;
    float acc = 0.0f;
    int row_offset = i * n;
    for (int j = 0; j < n; ++j) {
        float lij = lap[row_offset + j];
        if (lij != 0.0f) {
            acc += lij * x[j * d + k];
        }
    }
    out[idx] = acc;
}

extern "C" void laplace_beltrami_apply_cuda(
    const float* lap,
    const float* x,
    float* out,
    int n,
    int d,
    cudaStream_t stream
) {
    int total = n * d;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    laplace_beltrami_apply_kernel<<<grid_size, block_size, 0, stream>>>(lap, x, out, n, d);
}
