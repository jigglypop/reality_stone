// riemutils/csrc/hyper_butterfly_cuda.cu

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

#define CHECK_CUDA_CONTIGUOUS(x)                                    \
  TORCH_CHECK((x).device().is_cuda(), #x " must be CUDA tensor");   \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CUDA_CHECK(err)                                             \
  do {                                                              \
    auto e = (err);                                                 \
    TORCH_CHECK(e == cudaSuccess, "CUDA error: ",                   \
                cudaGetErrorString(e));                             \
  } while (0)

static constexpr float EPS = 1e-7f;

// atanh 헬퍼 (clamp 포함)
__device__ __forceinline__ float atanh_device(float x) {
  x = fminf(fmaxf(x, -1.0f + 1e-6f), 1.0f - 1e-6f);
  return 0.5f * logf((1.0f + x) / (1.0f - x));
}

// 다음 2의 거듭제곱
static inline int next_pow2(int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// 1) 로그 맵 forward 커널
//    y = atanh(√c‖x‖)/(√c‖x‖) * x
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t>
__global__ void log_map_origin_kernel(
    const scalar_t* __restrict__ x,
    scalar_t*       __restrict__ out,
    float c, int B, int D) {

  extern __shared__ float sdata[];
  float* s_norm2 = sdata;  // shared[0]

  // ─── 반드시 block 당 0으로 초기화 ─────────────────────
  if (threadIdx.x == 0) {
    s_norm2[0] = 0.f;
  }
  __syncthreads();
  // ────────────────────────────────────────────────────

  int bid = blockIdx.x, tid = threadIdx.x, stride = blockDim.x;
  const scalar_t* xb = x + bid*D;
  scalar_t*       yb = out + bid*D;

  // 1) ||x||^2 reduction
  float local = 0.f;
  for (int i = tid; i < D; i += stride) {
    float v = xb[i];
    local += v*v;
  }
  // warp‐reduce
  for (int off = warpSize/2; off > 0; off >>= 1) {
    local += __shfl_down_sync(0xffffffff, local, off);
  }
  if ((tid & (warpSize-1)) == 0) {
    atomicAdd(s_norm2, local);
  }
  __syncthreads();

  // 2) clamp & factor
  if (tid == 0) {
    s_norm2[0] = fmaxf(s_norm2[0], EPS);
  }
  __syncthreads();
  float norm = sqrtf(s_norm2[0]);
  float u    = sqrtf(c)*norm;
  u = fminf(fmaxf(u, 1e-6f), 0.999999f);
  float factor = atanh_device(u)/(u + 1e-6f);

  // 3) output
  for (int i = tid; i < D; i += stride) {
    yb[i] = factor * xb[i];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2) exp 맵 forward 커널
//    y = tanh(√c‖v‖)/(√c‖v‖) * v
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t>
__global__ void exp_map_origin_kernel(
    const scalar_t* __restrict__ v,
    scalar_t*       __restrict__ out,
    float c, int B, int D) {

  extern __shared__ float sdata[];
  float* s_norm2 = sdata;  // shared[0]

  // ─── 반드시 block 당 0으로 초기화 ─────────────────────
  if (threadIdx.x == 0) {
    s_norm2[0] = 0.f;
  }
  __syncthreads();
  // ────────────────────────────────────────────────────

  int bid = blockIdx.x, tid = threadIdx.x, stride = blockDim.x;
  const scalar_t* vb = v + bid*D;
  scalar_t*       yb = out + bid*D;

  // 1) ||v||^2 reduction
  float local = 0.f;
  for (int i = tid; i < D; i += stride) {
    float w = vb[i];
    local += w*w;
  }
  for (int off = warpSize/2; off > 0; off >>= 1) {
    local += __shfl_down_sync(0xffffffff, local, off);
  }
  if ((tid & (warpSize-1)) == 0) {
    atomicAdd(s_norm2, local);
  }
  __syncthreads();

  if (tid == 0) {
    s_norm2[0] = fmaxf(s_norm2[0], EPS);
  }
  __syncthreads();

  float norm = sqrtf(s_norm2[0]);
  float u    = sqrtf(c)*norm;
  u = fminf(fmaxf(u, 1e-6f), 10.0f);
  float tanhu = tanhf(u);
  float factor = tanhu/(u + 1e-3f);

  // 2) output
  for (int i = tid; i < D; i += stride) {
    yb[i] = factor * vb[i];
  }
}
// ─────────────────────────────────────────────────────────────────────────────
// 3) Butterfly 레이어 (forward)
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t>
__global__ void butterfly_layer_kernel(
    const scalar_t* __restrict__ input,
    scalar_t*       __restrict__ output,
    const scalar_t* __restrict__ params,
    int B, int D, int layer_idx) {

  int idx    = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x*gridDim.x;
  int bs     = 1 << layer_idx;
  int nb     = D / (2*bs);

  while(idx < B*D) {
    int b = idx / D, f = idx % D;
    int blk = (f/(2*bs)) % nb,
        loc = f % (2*bs),
        off = loc % bs;
    bool high = loc >= bs;
    int pi = blk*2;
    float a  = params[pi+0],
          bb = params[pi+1];
    int base = b*D + blk*2*bs;
    float x1 = input[base + off],
          x2 = input[base + off + bs];
    output[idx] = high
      ? (-bb*x1 + a*x2)
      : ( a*x1 + bb*x2 );
    idx += stride;
  }
}

template <typename scalar_t>
__global__ void butterfly_layer_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ input,
    scalar_t*       __restrict__ grad_in,
    const scalar_t* __restrict__ params,
    scalar_t*       __restrict__ grad_params,
    int B, int D, int layer_idx) 
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x*gridDim.x;
  int bs = 1<<layer_idx, nb = D/(2*bs);

  while(idx < B*D) {
    int b = idx/D, f = idx%D;
    int blk = (f/(2*bs))%nb, loc = f%(2*bs), off = loc%bs;
    bool high = loc>=bs;
    int pi = blk*2;
    float a = params[pi+0], bb = params[pi+1];
    int base = b*D + blk*2*bs;
    float x1 = input[base+off], x2 = input[base+off+bs];
    float gout = grad_out[idx];

    if(!high) {
      // y = a*x1 + b*x2
      atomicAdd(&grad_in[base+off  ],  a*gout);
      atomicAdd(&grad_in[base+off+bs],  bb*gout);
      atomicAdd(&grad_params[pi+0], x1*gout);
      atomicAdd(&grad_params[pi+1], x2*gout);
    } else {
      // y = -b*x1 + a*x2
      atomicAdd(&grad_in[base+off  ], -bb*gout);
      atomicAdd(&grad_in[base+off+bs],  a*gout);
      atomicAdd(&grad_params[pi+0],  x2*gout);
      atomicAdd(&grad_params[pi+1], -x1*gout);
    }
    idx += stride;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3) exp_map backward 커널
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t>
__global__ void exp_map_backward_kernel(
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ grad_y,
    scalar_t*       __restrict__ grad_v,
    float c, int B, int D) {

  extern __shared__ float sdata[];
  float* s_v2 = sdata;      // [0]
  float* s_vg = sdata + 1;  // [1]

  // ─── 반드시 block 당 0으로 초기화 ─────────────────────
  if (threadIdx.x == 0) {
    s_v2[0] = 0.f;
    s_vg[0] = 0.f;
  }
  __syncthreads();
  // ────────────────────────────────────────────────────

  int bid = blockIdx.x, tid = threadIdx.x, stride = blockDim.x;
  const scalar_t* vb = v + bid*D;
  const scalar_t* gy = grad_y + bid*D;

  // 1) ||v||^2, v·grad_y reduction
  float local_v2 = 0.f, local_vg = 0.f;
  for (int i = tid; i < D; i += stride) {
    float vv = vb[i], gyv = gy[i];
    local_v2 += vv*vv;
    local_vg += vv*gyv;
  }
  for (int off = warpSize/2; off > 0; off >>= 1) {
    local_v2 += __shfl_down_sync(0xffffffff, local_v2, off);
    local_vg += __shfl_down_sync(0xffffffff, local_vg, off);
  }
  if ((tid & (warpSize-1)) == 0) {
    atomicAdd(s_v2, local_v2);
    atomicAdd(s_vg, local_vg);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    s_v2[0] = fmaxf(s_v2[0], EPS);
  }
  __syncthreads();

  float norm = sqrtf(s_v2[0]);
  float u    = sqrtf(c)*norm;
  u = fminf(fmaxf(u, 1e-6f), 10.0f);
  float tanhu = tanhf(u);
  float sech2 = 1.0f - tanhu*tanhu;
  float factor = tanhu/(u + 1e-3f);

  // d factor / d norm
  float df_du = (u*sech2 - tanhu)/(u*u);
  float df_dn = df_du*sqrtf(c);
  float vdotgy= s_vg[0];

  // 2) per-dim gradient
  for (int i = tid; i < D; i += stride) {
    float vi  = vb[i], gyi = gy[i];
    grad_v[bid*D + i] = factor*gyi + (vi/norm)*(df_dn*vdotgy);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4) log_map backward 커널
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t>
__global__ void log_map_backward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ grad_u,
    scalar_t*       __restrict__ grad_x,
    float c, int B, int D) {

  extern __shared__ float sdata[];
  float* s_x2 = sdata;      // [0]
  float* s_xu = sdata + 1;  // [1]

  // ─── 반드시 block 당 0으로 초기화 ─────────────────────
  if (threadIdx.x == 0) {
    s_x2[0] = 0.f;
    s_xu[0] = 0.f;
  }
  __syncthreads();
  // ────────────────────────────────────────────────────

  int bid = blockIdx.x, tid = threadIdx.x, stride = blockDim.x;
  const scalar_t* xb = x     + bid*D;
  const scalar_t* gu = grad_u+ bid*D;

  // 1) ||x||^2, x·grad_u reduction
  float local_x2 = 0.f, local_xu = 0.f;
  for (int i = tid; i < D; i += stride) {
    float xi  = xb[i], gui = gu[i];
    local_x2 += xi*xi;
    local_xu += xi*gui;
  }
  for (int off = warpSize/2; off > 0; off >>= 1) {
    local_x2 += __shfl_down_sync(0xffffffff, local_x2, off);
    local_xu += __shfl_down_sync(0xffffffff, local_xu, off);
  }
  if ((tid & (warpSize-1)) == 0) {
    atomicAdd(s_x2, local_x2);
    atomicAdd(s_xu, local_xu);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    s_x2[0] = fmaxf(s_x2[0], EPS);
  }
  __syncthreads();

  float norm = sqrtf(s_x2[0]);
  float u    = sqrtf(c)*norm;
  u = fminf(fmaxf(u, 1e-6f), 0.999999f);
  float numer = atanh_device(u);
  float factor= numer/(u + 1e-6f);

  float sech2 = 1.0f - numer*numer;
  float df_du = (u*sech2 - numer)/(u*u);
  float df_dn = df_du*sqrtf(c);
  float xdotg= s_xu[0];

  // 2) per-dim gradient
  for (int i = tid; i < D; i += stride) {
    float xi   = xb[i], guv = gu[i];
    grad_x[bid*D + i] = factor*guv + (xi/norm)*(df_dn*xdotg);
  }
}


// 2. log_map_origin_cuda 및 exp_map_origin_cuda 구현
torch::Tensor log_map_origin_cuda(torch::Tensor x, float c) {
  CHECK_CUDA_CONTIGUOUS(x);
  int B = x.size(0), D = x.size(1);
  auto out = torch::empty_like(x);
  int threads = std::min(D, 1024);
  int shbytes = sizeof(float);
  
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "log_map_origin_cuda", [&]{
    log_map_origin_kernel<scalar_t><<<B, threads, shbytes>>>(
      x.data_ptr<scalar_t>(),
      out.data_ptr<scalar_t>(),
      c, B, D);
  });
  
  CUDA_CHECK(cudaGetLastError());
  return out;
}

torch::Tensor exp_map_origin_cuda(torch::Tensor v, float c) {
  CHECK_CUDA_CONTIGUOUS(v);
  int B = v.size(0), D = v.size(1);
  auto out = torch::empty_like(v);
  int threads = std::min(D, 1024);
  int shbytes = sizeof(float);
  
  AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "exp_map_origin_cuda", [&]{
    exp_map_origin_kernel<scalar_t><<<B, threads, shbytes>>>(
      v.data_ptr<scalar_t>(),
      out.data_ptr<scalar_t>(),
      c, B, D);
  });
  
  CUDA_CHECK(cudaGetLastError());
  return out;
}

std::vector<torch::Tensor> hyper_butterfly_cuda(
    torch::Tensor x,
    torch::Tensor params,
    torch::Tensor unused,
    float c,
    int L)
{
  CHECK_CUDA_CONTIGUOUS(x);
  CHECK_CUDA_CONTIGUOUS(params);

  int B = x.size(0), D = x.size(1);
  int D_padded = next_pow2(D);
  
  // Step 1: Pad input if needed
  torch::Tensor x_padded;
  if (D_padded > D) {
    x_padded = torch::zeros({B, D_padded}, x.options());
    x_padded.narrow(1, 0, D).copy_(x);
  } else {
    x_padded = x;
  }

  // Step 2: Log map
  torch::Tensor u = log_map_origin_cuda(x_padded, c);
  
  // Step 3: Apply butterfly transforms
  torch::Tensor v = u.clone();
  int threads = std::min(D_padded, 1024);
  int blocks = (B * D_padded + threads - 1) / threads;
  
  for (int l = 0; l < L; l++) {
    torch::Tensor next_v = torch::empty_like(v);
    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "butterfly_layer_cuda", [&]{
      butterfly_layer_kernel<scalar_t><<<blocks, threads>>>(
        v.data_ptr<scalar_t>(),
        next_v.data_ptr<scalar_t>(),
        params.data_ptr<scalar_t>(),
        B, D_padded, l);
    });
    v = next_v;
  }
  
  // Step 4: Exp map
  torch::Tensor y_padded = exp_map_origin_cuda(v, c);
  
  // Step 5: Slice to original dimension if needed
  torch::Tensor y = (D_padded > D) ? y_padded.narrow(1, 0, D) : y_padded;
  
  return {y, u, v};
}

std::vector<torch::Tensor> hyper_butterfly_backward_cuda(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor params,
    float c,
    int L)
{
  CHECK_CUDA_CONTIGUOUS(grad_y);
  CHECK_CUDA_CONTIGUOUS(x);
  CHECK_CUDA_CONTIGUOUS(params);

  int B = x.size(0), D = x.size(1);
  int D_padded = next_pow2(D);
  
  // Step 1: Pad input if needed
  torch::Tensor x_padded, grad_y_padded;
  if (D_padded > D) {
    x_padded = torch::zeros({B, D_padded}, x.options());
    x_padded.narrow(1, 0, D).copy_(x);
    
    grad_y_padded = torch::zeros({B, D_padded}, grad_y.options());
    grad_y_padded.narrow(1, 0, D).copy_(grad_y);
  } else {
    x_padded = x;
    grad_y_padded = grad_y;
  }

  // Step 2: Forward pass to get intermediate results
  torch::Tensor u = log_map_origin_cuda(x_padded, c);
  
  // Apply butterfly transforms (forward)
  std::vector<torch::Tensor> intermediates;
  intermediates.push_back(u);
  
  torch::Tensor v = u.clone();
  int threads = std::min(D_padded, 1024);
  int blocks = (B * D_padded + threads - 1) / threads;
  
  for (int l = 0; l < L; l++) {
    torch::Tensor next_v = torch::empty_like(v);
    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "butterfly_layer_cuda", [&]{
      butterfly_layer_kernel<scalar_t><<<blocks, threads>>>(
        v.data_ptr<scalar_t>(),
        next_v.data_ptr<scalar_t>(),
        params.data_ptr<scalar_t>(),
        B, D_padded, l);
    });
    v = next_v;
    intermediates.push_back(v);
  }
  
  // Final forward result
  torch::Tensor y_padded = exp_map_origin_cuda(v, c);
  
  // Step 3: Backward pass
  // Starting with grad_out at exp_map
  torch::Tensor grad_v = torch::zeros_like(v);
  int shbytes = sizeof(float);
  
  // Exp map backward
  AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "exp_map_backward_cuda", [&]{
    exp_map_backward_kernel<scalar_t><<<B, threads, shbytes>>>(
      v.data_ptr<scalar_t>(),
      grad_y_padded.data_ptr<scalar_t>(),
      grad_v.data_ptr<scalar_t>(),
      c, B, D_padded);
  });
  
  // Butterfly layers backward
  torch::Tensor grad_params = torch::zeros_like(params);
  
  for (int l = L-1; l >= 0; l--) {
    torch::Tensor grad_prev_v = torch::zeros_like(intermediates[l]);
    AT_DISPATCH_FLOATING_TYPES(grad_v.scalar_type(), "butterfly_backward_cuda", [&]{
      butterfly_layer_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_v.data_ptr<scalar_t>(),
        intermediates[l].data_ptr<scalar_t>(),
        grad_prev_v.data_ptr<scalar_t>(),
        params.data_ptr<scalar_t>(),
        grad_params.data_ptr<scalar_t>(),
        B, D_padded, l);
    });
    grad_v = grad_prev_v;
  }
  
  // Log map backward
  torch::Tensor grad_x_padded = torch::zeros_like(x_padded);
  AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "log_map_backward_cuda", [&]{
    log_map_backward_kernel<scalar_t><<<B, threads, shbytes>>>(
      x_padded.data_ptr<scalar_t>(),
      grad_v.data_ptr<scalar_t>(),
      grad_x_padded.data_ptr<scalar_t>(),
      c, B, D_padded);
  });
  
  // Step 6: Slice gradient to original dimension if needed
  torch::Tensor grad_x = (D_padded > D) ? grad_x_padded.narrow(1, 0, D) : grad_x_padded;
  
  return {grad_x, grad_params};
}

