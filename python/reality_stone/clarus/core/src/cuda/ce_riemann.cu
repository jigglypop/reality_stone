// Riemann-surface positional encoding attention — batched CUDA kernel.
//
// Mirrors `nn_ops::ce_riemann_fwd` (Rust) and
// `clarus.ce_riemann_attn.RiemannRotaryAttention` (PyTorch reference).
//
// Layout
//   Grid:  (BH, N, 1)                — one block per (head-batch, query row)
//   Block: (THREADS, 1, 1)           — threads cooperate on the row
//   Smem:  D + N floats              — rotated q row + score row
//
// All input tensors are pre-broadcast to shape (BH, N, *) row-major. cos/sin
// have last dim D/2; sheet_bias has last dim N.
//
// NVRTC has no <math.h>; INFINITY / isfinite are not visible, so we provide
// explicit sentinels and a lightweight finite check.

#define NEG_INF (-3.4028235e38f)

__device__ __forceinline__ int is_finite_f(float x) {
    return (x == x) && (x < 3.4028235e38f) && (x > -3.4028235e38f);
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(mask, v, off);
        if (other > v) v = other;
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(mask, v, off);
    }
    return v;
}

extern "C" __global__ void ce_riemann_fwd_kernel(
    const float* __restrict__ q,            // (BH, N, D)
    const float* __restrict__ k,            // (BH, N, D)
    const float* __restrict__ v,            // (BH, N, D)
    const float* __restrict__ cos_buf,      // (BH, N, D/2)
    const float* __restrict__ sin_buf,      // (BH, N, D/2)
    const float* __restrict__ sheet_bias,   // (BH, N, N)
    float* __restrict__ out,                // (BH, N, D)
    int N,
    int D,
    int causal
) {
    const int bh  = blockIdx.x;
    const int i   = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int half = D >> 1;
    const float scale = rsqrtf((float)D);

    extern __shared__ float smem[];
    float* qrot   = smem;            // [D]
    float* scores = smem + D;        // [N]

    const float* q_row   = q       + ((size_t)bh * N + i) * D;
    const float* cos_row = cos_buf + ((size_t)bh * N + i) * half;
    const float* sin_row = sin_buf + ((size_t)bh * N + i) * half;
    const float* sb_row  = sheet_bias + ((size_t)bh * N + i) * N;
    float*       out_row = out     + ((size_t)bh * N + i) * D;

    // --- 1. Rotate q[i,:] into shared memory, parallel over pairs ----------
    for (int p = tid; p < half; p += nthreads) {
        float c = cos_row[p];
        float s = sin_row[p];
        float a = q_row[2 * p];
        float b = q_row[2 * p + 1];
        qrot[2 * p]     = a * c - b * s;
        qrot[2 * p + 1] = a * s + b * c;
    }
    __syncthreads();

    // --- 2. Score row, parallel over j (each thread rotates its k_j on-the-fly)
    for (int j = tid; j < N; j += nthreads) {
        if (causal && j > i) {
            scores[j] = NEG_INF;
            continue;
        }
        const float* k_row = k       + ((size_t)bh * N + j) * D;
        const float* cos_j = cos_buf + ((size_t)bh * N + j) * half;
        const float* sin_j = sin_buf + ((size_t)bh * N + j) * half;

        float dot = 0.0f;
        #pragma unroll 4
        for (int p = 0; p < half; ++p) {
            float c = cos_j[p];
            float s = sin_j[p];
            float a = k_row[2 * p];
            float b = k_row[2 * p + 1];
            float kr0 = a * c - b * s;
            float kr1 = a * s + b * c;
            dot += qrot[2 * p] * kr0 + qrot[2 * p + 1] * kr1;
        }
        scores[j] = dot * scale + sb_row[j];
    }
    __syncthreads();

    // --- 3. Block-reduce max -------------------------------------------------
    __shared__ float warp_buf[32];

    float local_max = NEG_INF;
    for (int j = tid; j < N; j += nthreads) {
        float vv = scores[j];
        if (vv > local_max) local_max = vv;
    }
    local_max = warp_reduce_max(local_max);

    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int n_warps = (nthreads + 31) >> 5;

    if (lane == 0) warp_buf[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float vv = (lane < n_warps) ? warp_buf[lane] : NEG_INF;
        vv = warp_reduce_max(vv);
        if (lane == 0) warp_buf[0] = vv;
    }
    __syncthreads();
    const float row_max = warp_buf[0];

    // --- 4. Exp + block-reduce sum ------------------------------------------
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += nthreads) {
        float vv = scores[j];
        float e  = is_finite_f(vv) ? __expf(vv - row_max) : 0.0f;
        scores[j] = e;
        local_sum += e;
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) warp_buf[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float vv = (lane < n_warps) ? warp_buf[lane] : 0.0f;
        vv = warp_reduce_sum(vv);
        if (lane == 0) warp_buf[0] = vv;
    }
    __syncthreads();
    const float row_sum = warp_buf[0];
    const float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    // --- 5. Weighted sum -> out[i, :], parallel over d ----------------------
    for (int d = tid; d < D; d += nthreads) {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) {
            acc += scores[j] * v[((size_t)bh * N + j) * D + d];
        }
        out_row[d] = acc * inv_sum;
    }
}
