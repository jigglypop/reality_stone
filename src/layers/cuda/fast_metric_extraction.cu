#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

__global__ void init_random_basis_kernel(
    float* V,
    float* U,
    int in_dim,
    int out_dim,
    int k,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    int total_v = in_dim * k;
    int total_u = out_dim * k;
    
    if (idx < total_v) {
        float val = curand_normal(&state) / sqrtf((float)k);
        V[idx] = val;
    }
    
    if (idx < total_u) {
        float val = curand_normal(&state) / sqrtf((float)k);
        U[idx] = val;
    }
}

__global__ void compute_metric_from_weight_kernel(
    const float* W,
    const float* V,
    const float* U,
    float* G,
    int out_dim,
    int in_dim,
    int k
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= k || col >= k) return;
    
    float sum = 0.0f;
    
    for (int i = 0; i < out_dim; i++) {
        for (int j = 0; j < in_dim; j++) {
            float w_ij = W[i * in_dim + j];
            float u_ik = U[i * k + row];
            float v_jl = V[j * k + col];
            sum += u_ik * w_ij * v_jl;
        }
    }
    
    G[row * k + col] = sum;
}

__global__ void orthogonalize_basis_kernel(
    float* V,
    int dim,
    int k,
    int col_idx
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dim) return;
    
    __shared__ float dot_product;
    __shared__ float norm_sq;
    
    if (threadIdx.x == 0) {
        dot_product = 0.0f;
        norm_sq = 0.0f;
    }
    __syncthreads();
    
    for (int prev = 0; prev < col_idx; prev++) {
        float local_dot = V[row * k + col_idx] * V[row * k + prev];
        atomicAdd(&dot_product, local_dot);
        __syncthreads();
        
        V[row * k + col_idx] -= (dot_product / (norm_sq + 1e-8f)) * V[row * k + prev];
        __syncthreads();
    }
    
    float local_norm = V[row * k + col_idx] * V[row * k + col_idx];
    atomicAdd(&norm_sq, local_norm);
    __syncthreads();
    
    V[row * k + col_idx] /= sqrtf(norm_sq + 1e-8f);
}

extern "C" void fast_extract_metric_cuda(
    const float* W,
    float* U,
    float* G, 
    float* V,
    int out_dim,
    int in_dim,
    int k
) {
    float *d_W, *d_U, *d_G, *d_V;
    
    size_t w_size = out_dim * in_dim * sizeof(float);
    size_t u_size = out_dim * k * sizeof(float);
    size_t g_size = k * k * sizeof(float);
    size_t v_size = in_dim * k * sizeof(float);
    
    cudaMalloc(&d_W, w_size);
    cudaMalloc(&d_U, u_size);
    cudaMalloc(&d_G, g_size);
    cudaMalloc(&d_V, v_size);
    
    cudaMemcpy(d_W, W, w_size, cudaMemcpyHostToDevice);
    
    int max_dim = (in_dim > out_dim) ? in_dim : out_dim;
    int threads = 256;
    int blocks = (max_dim * k + threads - 1) / threads;
    
    unsigned long long seed = 42;
    init_random_basis_kernel<<<blocks, threads>>>(d_V, d_U, in_dim, out_dim, k, seed);
    cudaDeviceSynchronize();
    
    dim3 block_2d(16, 16);
    dim3 grid_2d((k + 15) / 16, (k + 15) / 16);
    compute_metric_from_weight_kernel<<<grid_2d, block_2d>>>(d_W, d_V, d_U, d_G, out_dim, in_dim, k);
    cudaDeviceSynchronize();
    
    cudaMemcpy(U, d_U, u_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(G, d_G, g_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(V, d_V, v_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_W);
    cudaFree(d_U);
    cudaFree(d_G);
    cudaFree(d_V);
}

