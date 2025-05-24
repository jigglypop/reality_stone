#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <advanced/fused_ops/fused_ops.h>
#include <ops/mobius.h>
#include <vector>
#include <chrono>

namespace ops = reality_stone::ops;

namespace reality_stone::advanced {

__global__ void transform_regularize_fused_kernel(
    const float* __restrict__ input,    // [B, D]
    float* __restrict__ output,         // [B, D]
    float* __restrict__ reg_loss,       // [1]
    float curvature,
    float reg_lambda,
    int B, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * D;
    
    if (idx >= total) return;
    
    int b = idx / D;
    
    // 해당 배치의 노름 계산
    float norm_sq = 0.0f;
    for (int dd = 0; dd < D; ++dd) {
        float val = input[b * D + dd];
        norm_sq += val * val;
    }
    float norm = sqrtf(norm_sq);
    
    // 최대 허용 노름
    float max_norm = 1.0f / sqrtf(curvature) - 0.01f;
    
    // 변환된 출력
    float clamped_norm = fminf(norm, max_norm);
    float direction = input[idx] / (norm + 1e-7f);
    output[idx] = direction * clamped_norm;
    
    // 정규화 손실 (첫 번째 스레드만 계산)
    if (idx == 0) {
        float total_violation = 0.0f;
        for (int bb = 0; bb < B; ++bb) {
            float batch_norm_sq = 0.0f;
            for (int dd = 0; dd < D; ++dd) {
                float val = input[bb * D + dd];
                batch_norm_sq += val * val;
            }
            float batch_norm = sqrtf(batch_norm_sq);
            float violation = fmaxf(0.0f, batch_norm - max_norm);
            total_violation += violation * violation;
        }
        reg_loss[0] = reg_lambda * total_violation / B;
    }
}

torch::Tensor hyperbolic_linear_fused(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float curvature
) {
    // 기본적인 하이퍼볼릭 선형 변환 수행
    // log_map -> linear -> exp_map 순서
    
    auto norm = torch::norm(input, 2, -1, true);
    auto sqrt_c = std::sqrt(curvature);
    auto atanh_arg = torch::clamp(sqrt_c * norm, -0.99f, 0.99f);
    auto coeff = torch::atanh(atanh_arg) / (sqrt_c * norm + 1e-7f);
    auto log_input = coeff * input;
    
    // 선형 변환
    auto linear_result = torch::mm(log_input, weight.t()) + bias;
    
    // exp_map
    auto result_norm = torch::norm(linear_result, 2, -1, true);
    auto tanh_arg = sqrt_c * result_norm;
    auto exp_coeff = torch::tanh(tanh_arg) / (sqrt_c * result_norm + 1e-7f);
    
    return exp_coeff * linear_result;
}

torch::Tensor mobius_chain_fused(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<float>& curvatures
) {
    if (inputs.empty()) {
        throw std::invalid_argument("Empty inputs vector");
    }
    
    if (inputs.size() != curvatures.size()) {
        throw std::invalid_argument("inputs and curvatures size mismatch");
    }
    
    auto result = inputs[0];
    
    // 연속적인 Möbius 덧셈 체인
    for (size_t i = 1; i < inputs.size(); ++i) {
        if (result.is_cuda()) {
            result = ops::mobius_add_cuda(result, inputs[i], curvatures[i]);
        } else {
            result = ops::mobius_add_cpu(result, inputs[i], curvatures[i]);
        }
    }
    
    return result;
}

std::tuple<torch::Tensor, torch::Tensor> transform_regularize_fused(
    const torch::Tensor& input,
    float curvature,
    float reg_lambda
) {
    if (!input.is_cuda()) {
        // CPU 버전으로 fallback
        auto norm = torch::norm(input, 2, /*dim=*/-1, /*keepdim=*/true);
        auto max_norm = 1.0f / std::sqrt(curvature) - 0.01f;
        auto clamped_norm = torch::clamp(norm, 0.0f, max_norm);
        auto direction = input / (norm + 1e-7f);
        auto transformed = direction * clamped_norm;
        
        auto boundary_violation = torch::relu(norm - max_norm);
        auto reg_loss = reg_lambda * torch::mean(boundary_violation * boundary_violation);
        
        return std::make_tuple(transformed, reg_loss);
    }
    
    // CUDA 구현
    auto B = input.size(0);
    auto D = input.size(1);
    auto output = torch::zeros_like(input);
    auto reg_loss = torch::zeros({1}, input.options());
    
    const int threads = 256;
    const int blocks = (static_cast<int>(B * D) + threads - 1) / threads;
    
    transform_regularize_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        reg_loss.data_ptr<float>(),
        curvature,
        reg_lambda,
        static_cast<int>(B), 
        static_cast<int>(D)
    );
    
    cudaDeviceSynchronize();
    return std::make_tuple(output, reg_loss);
}



} // namespace reality_stone::advanced 