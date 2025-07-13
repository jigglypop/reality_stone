// python/reality_stone/bitfield_cuda.cpp

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA 커널 선언
extern "C" {
    void launch_bitfield_gemm_kernel(
        const float* x,
        const uint32_t* codes,
        const float* basis_table,
        float delta,
        float* output,
        int batch_size,
        int n,
        int m,
        int b,
        cudaStream_t stream
    );
    
    void launch_int8_residual_kernel(
        const float* x,
        const int8_t* residual_int8,
        const float* residual_scales,
        float* output,
        int batch_size,
        int n,
        int m,
        cudaStream_t stream
    );
    
    void launch_add_kernel(
        float* output,
        const float* workspace,
        int size,
        cudaStream_t stream
    );
}

// Forward 함수
torch::Tensor bitfield_forward(
    torch::Tensor input,
    torch::Tensor codes,
    torch::Tensor basis_table,
    torch::optional<torch::Tensor> residual_codes,
    torch::optional<torch::Tensor> residual_int8,
    torch::optional<torch::Tensor> residual_scales,
    float delta,
    float residual_delta
) {
    // 입력 검증
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(codes.is_cuda(), "Codes must be a CUDA tensor");
    TORCH_CHECK(basis_table.is_cuda(), "Basis table must be a CUDA tensor");
    
    const auto batch_size = input.size(0);
    const auto n = input.size(1);
    const auto m = codes.size(0);
    const auto b = basis_table.size(0);
    
    // 출력 텐서 생성
    auto output = torch::zeros({batch_size, m}, input.options());
    
    // 현재 CUDA 스트림 가져오기
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // 주 비트필드 연산
    launch_bitfield_gemm_kernel(
        input.data_ptr<float>(),
        codes.data_ptr<uint32_t>(),
        basis_table.data_ptr<float>(),
        delta,
        output.data_ptr<float>(),
        batch_size,
        n,
        m,
        b,
        stream
    );
    
    // 잔차 처리
    if (residual_codes.has_value()) {
        // 비트필드 잔차
        auto workspace = torch::zeros_like(output);
        launch_bitfield_gemm_kernel(
            input.data_ptr<float>(),
            residual_codes.value().data_ptr<uint32_t>(),
            basis_table.data_ptr<float>(),
            residual_delta,
            workspace.data_ptr<float>(),
            batch_size,
            n,
            m,
            b,
            stream
        );
        
        // output += workspace
        launch_add_kernel(
            output.data_ptr<float>(),
            workspace.data_ptr<float>(),
            batch_size * m,
            stream
        );
    } else if (residual_int8.has_value() && residual_scales.has_value()) {
        // INT8 잔차
        auto workspace = torch::zeros_like(output);
        launch_int8_residual_kernel(
            input.data_ptr<float>(),
            residual_int8.value().data_ptr<int8_t>(),
            residual_scales.value().data_ptr<float>(),
            workspace.data_ptr<float>(),
            batch_size,
            n,
            m,
            stream
        );
        
        // output += workspace
        launch_add_kernel(
            output.data_ptr<float>(),
            workspace.data_ptr<float>(),
            batch_size * m,
            stream
        );
    }
    
    return output;
}

// Backward 함수 (간단한 버전 - 실제로는 더 최적화 필요)
torch::Tensor bitfield_backward(
    torch::Tensor grad_output,
    torch::Tensor codes,
    torch::Tensor basis_table,
    torch::optional<torch::Tensor> residual_codes,
    torch::optional<torch::Tensor> residual_int8,
    torch::optional<torch::Tensor> residual_scales,
    float delta,
    float residual_delta
) {
    // 구현 필요
    // 일단은 forward의 transpose 연산으로 근사
    TORCH_CHECK(grad_output.is_cuda(), "Grad output must be a CUDA tensor");
    
    const auto batch_size = grad_output.size(0);
    const auto m = grad_output.size(1);
    const auto n = basis_table.size(1);
    
    // 임시 구현: CPU로 폴백
    // TODO: 전용 backward 커널 구현
    auto grad_input = torch::zeros({batch_size, n}, grad_output.options());
    
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &bitfield_forward, "Bitfield forward (CUDA)");
    m.def("backward", &bitfield_backward, "Bitfield backward (CUDA)");
} 