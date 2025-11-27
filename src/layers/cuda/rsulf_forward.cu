#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>

#define RSULF_EPS 1e-6f
#define WARP_SIZE 32

namespace {

__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__device__ inline float leaky_relu(float x) {
    return x > 0.0f ? x : 0.017f * x;
}

}

__global__ void rsulf_forward_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ v1,
    const float* __restrict__ s1,
    const float* __restrict__ u1,
    const float* __restrict__ v2,
    const float* __restrict__ s2,
    const float* __restrict__ u2,
    const float* __restrict__ g_inv,
    const float* __restrict__ v_mem,
    const float eta,
    const float alpha,
    const float gamma_param,
    const int batch,
    const int d,
    const int r,
    const int ffn_dim,
    float* __restrict__ x_out,
    float* __restrict__ v_out
) {
    extern __shared__ float smem[];
    
    const int b = blockIdx.x;
    if (b >= batch) return;
    
    float* x_local = smem;
    float* h1 = smem + d;
    float* h2 = smem + d + r;
    float* phi_grad = smem + d + r + ffn_dim;
    
    const float* x_row = x + b * d;
    float* x_out_row = x_out + b * d;
    
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        x_local[i] = x_row[i];
    }
    __syncthreads();
    
    for (int j = threadIdx.x; j < r; j += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            sum += x_local[i] * v1[i * r + j];
        }
        h1[j] = sum * s1[j];
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < ffn_dim; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < r; ++j) {
            sum += h1[j] * u1[i * r + j];
        }
        h2[i] = leaky_relu(sum);
    }
    __syncthreads();
    
    for (int j = threadIdx.x; j < r; j += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < ffn_dim; ++i) {
            sum += h2[i] * v2[i * r + j];
        }
        h1[j] = sum * s2[j];
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < r; ++j) {
            sum += h1[j] * u2[i * r + j];
        }
        phi_grad[i] = sum;
    }
    __syncthreads();
    
    float local_phi_sq = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        local_phi_sq += phi_grad[i] * phi_grad[i];
    }
    float phi_val = blockReduceSum(local_phi_sq) * 0.5f;
    
    __shared__ float shared_mean[1];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        local_sum += x_local[i];
    }
    float mean_val = blockReduceSum(local_sum) / (float)d;
    if (threadIdx.x == 0) {
        shared_mean[0] = mean_val;
    }
    __syncthreads();
    mean_val = shared_mean[0];
    
    float v_prev = (v_mem != nullptr) ? v_mem[b] : 0.0f;
    float v_new = gamma_param * v_prev + (1.0f - gamma_param) * phi_val;
    
    if (threadIdx.x == 0) {
        v_out[b] = v_new;
    }
    
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float g_i = (i < r) ? g_inv[i] : 1.0f;
        
        float term1 = -eta * g_i * phi_grad[i];
        float term2 = alpha * (x_local[i] - mean_val);
        float term3 = gamma_param * v_new;
        
        float velocity = term1 + term2 + term3;
        velocity = fmaxf(-1.0f, fminf(1.0f, velocity));
        
        float x_next = x_local[i] + velocity;
        x_out_row[i] = fmaxf(-10.0f, fminf(10.0f, x_next));
    }
}

__global__ void rsulf_forward_vectorized_kernel(
    const float* __restrict__ x,
    const float* __restrict__ v1,
    const float* __restrict__ s1,
    const float* __restrict__ u1_t,
    const float* __restrict__ v2,
    const float* __restrict__ s2,
    const float* __restrict__ u2_t,
    const float* __restrict__ g_inv,
    const float* __restrict__ v_mem,
    const float eta,
    const float alpha,
    const float gamma_param,
    const int batch,
    const int d,
    const int r,
    const int ffn_dim,
    float* __restrict__ x_out,
    float* __restrict__ v_out
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    if (bid >= batch) return;
    
    extern __shared__ float smem[];
    float* s_x = smem;
    float* s_h1 = smem + d;
    float* s_h2 = smem + d + r;
    float* s_phi = smem + d + r + ffn_dim;
    float* s_reduce = smem + 2 * d + r + ffn_dim;
    
    const float* x_in = x + bid * d;
    float* x_o = x_out + bid * d;
    
    for (int i = tid; i < d; i += blockDim.x) {
        s_x[i] = x_in[i];
    }
    __syncthreads();
    
    for (int j = tid; j < r; j += blockDim.x) {
        float acc = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < d; i += 4) {
            if (i + 3 < d) {
                acc += s_x[i] * v1[i * r + j];
                acc += s_x[i+1] * v1[(i+1) * r + j];
                acc += s_x[i+2] * v1[(i+2) * r + j];
                acc += s_x[i+3] * v1[(i+3) * r + j];
            } else {
                for (int k = i; k < d; ++k) {
                    acc += s_x[k] * v1[k * r + j];
                }
            }
        }
        s_h1[j] = acc * s1[j];
    }
    __syncthreads();
    
    for (int i = tid; i < ffn_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < r; ++j) {
            acc += s_h1[j] * u1_t[j * ffn_dim + i];
        }
        s_h2[i] = leaky_relu(acc);
    }
    __syncthreads();
    
    for (int j = tid; j < r; j += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < ffn_dim; ++i) {
            acc += s_h2[i] * v2[i * r + j];
        }
        s_h1[j] = acc * s2[j];
    }
    __syncthreads();
    
    for (int i = tid; i < d; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < r; ++j) {
            acc += s_h1[j] * u2_t[j * d + i];
        }
        s_phi[i] = acc;
    }
    __syncthreads();
    
    float local_sum = 0.0f;
    float local_phi_sq = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        local_sum += s_x[i];
        local_phi_sq += s_phi[i] * s_phi[i];
    }
    
    s_reduce[tid] = local_sum;
    s_reduce[tid + blockDim.x] = local_phi_sq;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
            s_reduce[tid + blockDim.x] += s_reduce[tid + blockDim.x + stride];
        }
        __syncthreads();
    }
    
    float mean_val = s_reduce[0] / (float)d;
    float phi_val = s_reduce[blockDim.x] * 0.5f;
    
    float v_prev = (v_mem != nullptr) ? v_mem[bid] : 0.0f;
    float v_new = gamma_param * v_prev + (1.0f - gamma_param) * phi_val;
    
    if (tid == 0) {
        v_out[bid] = v_new;
    }
    
    for (int i = tid; i < d; i += blockDim.x) {
        float g_i = (i < r) ? g_inv[i] : 1.0f;
        float velocity = -eta * g_i * s_phi[i] + alpha * (s_x[i] - mean_val) + gamma_param * v_new;
        velocity = fmaxf(-1.0f, fminf(1.0f, velocity));
        x_o[i] = fmaxf(-10.0f, fminf(10.0f, s_x[i] + velocity));
    }
}

extern "C" void rsulf_forward_cuda(
    const float* x,
    const float* v1,
    const float* s1,
    const float* u1,
    const float* v2,
    const float* s2,
    const float* u2,
    const float* g_inv,
    const float* v_mem,
    float eta,
    float alpha,
    float gamma_param,
    int batch,
    int d,
    int r,
    int ffn_dim,
    float* x_out,
    float* v_out
) {
    size_t smem_size = (2 * d + r + ffn_dim + 512) * sizeof(float);
    
    int block_size = 256;
    dim3 grid(batch);
    dim3 block(block_size);
    
    if (d <= 1024 && r <= 256 && ffn_dim <= 4096) {
        rsulf_forward_vectorized_kernel<<<grid, block, smem_size>>>(
            x, v1, s1, u1, v2, s2, u2, g_inv, v_mem,
            eta, alpha, gamma_param,
            batch, d, r, ffn_dim,
            x_out, v_out
        );
    } else {
        rsulf_forward_fused_kernel<<<grid, block, smem_size>>>(
            x, v1, s1, u1, v2, s2, u2, g_inv, v_mem,
            eta, alpha, gamma_param,
            batch, d, r, ffn_dim,
            x_out, v_out
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in rsulf_forward: %s\n", cudaGetErrorString(err));
    }
}

__global__ void rsulf_batch_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ v1,
    const float* __restrict__ s1,
    const float* __restrict__ u1_t,
    const float* __restrict__ v2,
    const float* __restrict__ s2,
    const float* __restrict__ u2_t,
    const float* __restrict__ g_inv,
    const float* __restrict__ v_mem,
    const float eta,
    const float alpha,
    const float gamma_param,
    const int batch,
    const int seq_len,
    const int d,
    const int r,
    const int ffn_dim,
    float* __restrict__ x_out,
    float* __restrict__ v_out
) {
    const int tid = threadIdx.x;
    const int total_tokens = batch * seq_len;
    const int token_idx = blockIdx.x;
    
    if (token_idx >= total_tokens) return;
    
    extern __shared__ float smem[];
    float* s_x = smem;
    float* s_h1 = smem + d;
    float* s_h2 = smem + d + r;
    float* s_phi = smem + d + r + ffn_dim;
    float* s_reduce = smem + 2 * d + r + ffn_dim;
    
    const float* x_in = x + token_idx * d;
    float* x_o = x_out + token_idx * d;
    
    for (int i = tid; i < d; i += blockDim.x) {
        s_x[i] = x_in[i];
    }
    __syncthreads();
    
    for (int j = tid; j < r; j += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < d; ++i) {
            acc += s_x[i] * v1[i * r + j];
        }
        s_h1[j] = acc * s1[j];
    }
    __syncthreads();
    
    for (int i = tid; i < ffn_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < r; ++j) {
            acc += s_h1[j] * u1_t[j * ffn_dim + i];
        }
        s_h2[i] = leaky_relu(acc);
    }
    __syncthreads();
    
    for (int j = tid; j < r; j += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < ffn_dim; ++i) {
            acc += s_h2[i] * v2[i * r + j];
        }
        s_h1[j] = acc * s2[j];
    }
    __syncthreads();
    
    for (int i = tid; i < d; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < r; ++j) {
            acc += s_h1[j] * u2_t[j * d + i];
        }
        s_phi[i] = acc;
    }
    __syncthreads();
    
    float local_sum = 0.0f;
    float local_phi_sq = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        local_sum += s_x[i];
        local_phi_sq += s_phi[i] * s_phi[i];
    }
    
    s_reduce[tid] = local_sum;
    s_reduce[tid + blockDim.x] = local_phi_sq;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
            s_reduce[tid + blockDim.x] += s_reduce[tid + blockDim.x + stride];
        }
        __syncthreads();
    }
    
    float mean_val = s_reduce[0] / (float)d;
    float phi_val = s_reduce[blockDim.x] * 0.5f;
    
    float v_prev = (v_mem != nullptr) ? v_mem[token_idx] : 0.0f;
    float v_new = gamma_param * v_prev + (1.0f - gamma_param) * phi_val;
    
    if (tid == 0) {
        v_out[token_idx] = v_new;
    }
    
    for (int i = tid; i < d; i += blockDim.x) {
        float g_i = (i < r) ? g_inv[i] : 1.0f;
        float velocity = -eta * g_i * s_phi[i] + alpha * (s_x[i] - mean_val) + gamma_param * v_new;
        velocity = fmaxf(-1.0f, fminf(1.0f, velocity));
        x_o[i] = fmaxf(-10.0f, fminf(10.0f, s_x[i] + velocity));
    }
}

extern "C" void rsulf_batch_forward_cuda(
    const float* x,
    const float* v1,
    const float* s1,
    const float* u1,
    const float* v2,
    const float* s2,
    const float* u2,
    const float* g_inv,
    const float* v_mem,
    float eta,
    float alpha,
    float gamma_param,
    int batch,
    int seq_len,
    int d,
    int r,
    int ffn_dim,
    float* x_out,
    float* v_out
) {
    int total_tokens = batch * seq_len;
    size_t smem_size = (2 * d + r + ffn_dim + 512) * sizeof(float);
    
    int block_size = 256;
    dim3 grid(total_tokens);
    dim3 block(block_size);
    
    rsulf_batch_forward_kernel<<<grid, block, smem_size>>>(
        x, v1, s1, u1, v2, s2, u2, g_inv, v_mem,
        eta, alpha, gamma_param,
        batch, seq_len, d, r, ffn_dim,
        x_out, v_out
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in rsulf_batch_forward: %s\n", cudaGetErrorString(err));
    }
}

