#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <ops/mobius.h>
#include <cmath>
#include <algorithm>

#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32

namespace ops = reality_stone::ops;

namespace reality_stone::advanced {

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void dynamic_curvature_prediction_kernel(
    const float* __restrict__ features,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float c_base,
    float min_curvature,
    float max_curvature,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    float logit = weight[0] * features[idx] + bias[0];
    float exp_val = expf(fmaxf(fminf(logit, 20.0f), -20.0f));
    float curvature = c_base * exp_val;
    output[idx] = fmaxf(fminf(curvature, max_curvature), min_curvature);
}

torch::Tensor dynamic_curvature_prediction_cuda(
    const torch::Tensor& features,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float c_base,
    float min_curvature,
    float max_curvature
) {
    auto batch_size = features.size(0);
    auto output = torch::zeros({batch_size}, features.options());
    
    const int threads = 256;
    const int blocks = (static_cast<int>(batch_size) + threads - 1) / threads;
    
    dynamic_curvature_prediction_kernel<<<blocks, threads>>>(
        features.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        c_base,
        min_curvature,
        max_curvature,
        static_cast<int>(batch_size)
    );
    
    cudaDeviceSynchronize();
    return output;
}

torch::Tensor dynamic_mobius_add_cuda(
    const torch::Tensor& u,
    const torch::Tensor& v,
    const torch::Tensor& curvatures
) {
    auto batch_size = u.size(0);
    auto cpu_curvatures = curvatures.cpu();
    auto first_c = cpu_curvatures[0].item<float>();
    
    bool all_same = true;
    for (int i = 1; i < cpu_curvatures.size(0); ++i) {
        if (std::abs(cpu_curvatures[i].item<float>() - first_c) > 1e-6f) {
            all_same = false;
            break;
        }
    }
    
    if (all_same) {
        return ops::mobius_add_cuda(u, v, first_c);
    } else {
        auto result = torch::zeros_like(u);
        for (int b = 0; b < batch_size; ++b) {
            auto u_b = u[b].unsqueeze(0);
            auto v_b = v[b].unsqueeze(0);
            auto c_b = cpu_curvatures[b].item<float>();
            auto result_b = ops::mobius_add_cuda(u_b, v_b, c_b);
            result[b] = result_b.squeeze(0);
        }
        return result;
    }
}

torch::Tensor dynamic_poincare_layer_cuda(
    const torch::Tensor& u,
    const torch::Tensor& v,
    const torch::Tensor& curvatures,
    float t
) {
    auto tv = t * v;
    auto one_minus_t_u = (1.0f - t) * u;
    return dynamic_mobius_add_cuda(one_minus_t_u, tv, curvatures);
}

}