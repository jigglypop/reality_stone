#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// CUDA 커널 매크로: 에러 체크
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 스레드 블록 크기 정의
const int BLOCK_SIZE = 256;

///////////////////////////////////////////////////////
// 버터플라이 변환 CUDA 커널
///////////////////////////////////////////////////////

// 버터플라이 레이어 적용 커널
template <typename scalar_t>
__global__ void butterfly_layer_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ a_params,
    const scalar_t* __restrict__ b_params,
    const int batch_size,
    const int dim,
    const int layer_idx) {
    
    // 현재 블록 크기
    const int block_size = 1 << layer_idx;
    const int num_blocks = dim / (2 * block_size);
    
    // 글로벌 인덱스
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 유효한 스레드만 처리
    if (idx < batch_size * dim) {
        const int batch_idx = idx / dim;
        const int feature_idx = idx % dim;
        
        // 블록 인덱스 계산
        const int block_idx = (feature_idx / (2 * block_size)) % num_blocks;
        const int local_idx = feature_idx % (2 * block_size);
        const int is_second_half = local_idx / block_size;
        const int offset_idx = local_idx % block_size;
        
        // a, b 파라미터
        const scalar_t a = a_params[block_idx];
        const scalar_t b = b_params[block_idx];
        
        // 입력 값 가져오기
        const int in_idx1 = batch_idx * dim + block_idx * (2 * block_size) + offset_idx;
        const int in_idx2 = in_idx1 + block_size;
        
        const scalar_t x1 = input[in_idx1];
        const scalar_t x2 = input[in_idx2];
        
        // 버터플라이 연산 (Givens 회전)
        if (is_second_half == 0) {
            // 첫 번째 절반
            output[idx] = a * x1 + b * x2;
        } else {
            // 두 번째 절반
            output[idx] = -b * x1 + a * x2;
        }
    }
}

// 버터플라이 레이어 역전파 커널
template <typename scalar_t>
__global__ void butterfly_layer_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_input,
    const scalar_t* __restrict__ a_params,
    const scalar_t* __restrict__ b_params,
    scalar_t* __restrict__ grad_a,
    scalar_t* __restrict__ grad_b,
    const int batch_size,
    const int dim,
    const int layer_idx) {
    
    // 블록 크기 계산
    const int block_size = 1 << layer_idx;
    const int num_blocks = dim / (2 * block_size);
    
    // 글로벌 인덱스
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 유효한 스레드만 처리
    if (idx < batch_size * dim) {
        const int batch_idx = idx / dim;
        const int feature_idx = idx % dim;
        
        // 블록 인덱스 계산
        const int block_idx = (feature_idx / (2 * block_size)) % num_blocks;
        const int local_idx = feature_idx % (2 * block_size);
        const int is_second_half = local_idx / block_size;
        const int offset_idx = local_idx % block_size;
        
        // a, b 파라미터
        const scalar_t a = a_params[block_idx];
        const scalar_t b = b_params[block_idx];
        
        // 그래디언트 인덱스 계산
        const int out_idx = idx;
        const int base_idx = batch_idx * dim + block_idx * (2 * block_size);
        const int in_idx1 = base_idx + offset_idx;
        const int in_idx2 = in_idx1 + block_size;
        
        const scalar_t grad = grad_output[out_idx];
        
        // 역전파: 입력에 대한 그래디언트
        if (is_second_half == 0) {
            // 첫 번째 절반
            atomicAdd(&grad_input[in_idx1], a * grad);
            atomicAdd(&grad_input[in_idx2], b * grad);
        } else {
            // 두 번째 절반
            atomicAdd(&grad_input[in_idx1], -b * grad);
            atomicAdd(&grad_input[in_idx2], a * grad);
        }
        
        // a, b 파라미터에 대한 그래디언트 (로컬 계산)
        if (offset_idx == 0 && is_second_half == 0) {
            scalar_t grad_a_local = 0.0;
            scalar_t grad_b_local = 0.0;
            
            // 모든 배치와 로컬 인덱스에 대해 그래디언트 계산
            for (int batch = 0; batch < batch_size; batch++) {
                for (int offset = 0; offset < block_size; offset++) {
                    const int base = batch * dim + block_idx * (2 * block_size);
                    const int idx1 = base + offset;
                    const int idx2 = idx1 + block_size;
                    
                    const scalar_t x1 = grad_input[idx1];
                    const scalar_t x2 = grad_input[idx2];
                    
                    const scalar_t dy1 = grad_output[base + offset];
                    const scalar_t dy2 = grad_output[base + offset + block_size];
                    
                    grad_a_local += x1 * dy1 + x2 * dy2;
                    grad_b_local += x2 * dy1 - x1 * dy2;
                }
            }
            
            atomicAdd(&grad_a[block_idx], grad_a_local);
            atomicAdd(&grad_b[block_idx], grad_b_local);
        }
    }
}

// 호스트 함수: 버터플라이 레이어 순전파
torch::Tensor butterfly_layer_forward_cuda(
    torch::Tensor input,
    torch::Tensor a_params,
    torch::Tensor b_params,
    int layer_idx) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(a_params);
    CHECK_INPUT(b_params);
    
    const int batch_size = input.size(0);
    const int dim = input.size(1);
    
    auto output = torch::zeros_like(input);
    
    const int threads = BLOCK_SIZE;
    const int blocks = (batch_size * dim + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "butterfly_layer_forward_cuda", ([&] {
        butterfly_layer_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            a_params.data_ptr<scalar_t>(),
            b_params.data_ptr<scalar_t>(),
            batch_size,
            dim,
            layer_idx
        );
    }));
    
    return output;
}

// 호스트 함수: 버터플라이 레이어 역전파
std::vector<torch::Tensor> butterfly_layer_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor a_params,
    torch::Tensor b_params,
    int layer_idx) {
    
    CHECK_INPUT(grad_output);
    CHECK_INPUT(a_params);
    CHECK_INPUT(b_params);
    
    const int batch_size = grad_output.size(0);
    const int dim = grad_output.size(1);
    const int block_size = 1 << layer_idx;
    const int num_blocks = dim / (2 * block_size);
    
    auto grad_input = torch::zeros_like(grad_output);
    auto grad_a = torch::zeros_like(a_params);
    auto grad_b = torch::zeros_like(b_params);
    
    const int threads = BLOCK_SIZE;
    const int blocks = (batch_size * dim + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "butterfly_layer_backward_cuda", ([&] {
        butterfly_layer_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            a_params.data_ptr<scalar_t>(),
            b_params.data_ptr<scalar_t>(),
            grad_a.data_ptr<scalar_t>(),
            grad_b.data_ptr<scalar_t>(),
            batch_size,
            dim,
            layer_idx
        );
    }));
    
    return {grad_input, grad_a, grad_b};
}

///////////////////////////////////////////////////////
// 전체 버터플라이 변환 CUDA 커널 (모든 레이어)
///////////////////////////////////////////////////////

// 전체 버터플라이 변환 순전파 CUDA 구현
torch::Tensor butterfly_transform_forward_cuda(
    torch::Tensor input,
    std::vector<torch::Tensor> a_params_list,
    std::vector<torch::Tensor> b_params_list) {
    
    CHECK_INPUT(input);
    
    const int batch_size = input.size(0);
    const int dim = input.size(1);
    const int log_dim = a_params_list.size();
    
    auto x = input.clone();
    
    // 각 버터플라이 레이어 적용
    for (int layer_idx = 0; layer_idx < log_dim; layer_idx++) {
        CHECK_INPUT(a_params_list[layer_idx]);
        CHECK_INPUT(b_params_list[layer_idx]);
        
        x = butterfly_layer_forward_cuda(
            x, 
            a_params_list[layer_idx], 
            b_params_list[layer_idx], 
            layer_idx
        );
    }
    
    return x;
}

// 전체 버터플라이 변환 역전파 CUDA 구현
std::vector<std::vector<torch::Tensor>> butterfly_transform_backward_cuda(
    torch::Tensor grad_output,
    std::vector<torch::Tensor> a_params_list,
    std::vector<torch::Tensor> b_params_list,
    torch::Tensor input) {
    
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    
    const int batch_size = grad_output.size(0);
    const int dim = grad_output.size(1);
    const int log_dim = a_params_list.size();
    
    std::vector<torch::Tensor> a_grads_list(log_dim);
    std::vector<torch::Tensor> b_grads_list(log_dim);
    std::vector<torch::Tensor> layer_outputs(log_dim + 1);
    
    // 순전파 값 저장
    layer_outputs[0] = input.clone();
    for (int layer_idx = 0; layer_idx < log_dim; layer_idx++) {
        layer_outputs[layer_idx + 1] = butterfly_layer_forward_cuda(
            layer_outputs[layer_idx],
            a_params_list[layer_idx],
            b_params_list[layer_idx],
            layer_idx
        );
    }
    
    // 역전파 수행
    auto grad = grad_output.clone();
    for (int layer_idx = log_dim - 1; layer_idx >= 0; layer_idx--) {
        auto grads = butterfly_layer_backward_cuda(
            grad,
            a_params_list[layer_idx],
            b_params_list[layer_idx],
            layer_idx
        );
        
        grad = grads[0];
        a_grads_list[layer_idx] = grads[1];
        b_grads_list[layer_idx] = grads[2];
    }
    
    std::vector<std::vector<torch::Tensor>> result = {
        a_grads_list,
        b_grads_list,
        {grad}  // 입력에 대한 그래디언트
    };
    
    return result;
}

///////////////////////////////////////////////////////
// Python 바인딩
///////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("butterfly_layer_forward", &butterfly_layer_forward_cuda, "Butterfly Layer Forward (CUDA)");
    m.def("butterfly_layer_backward", &butterfly_layer_backward_cuda, "Butterfly Layer Backward (CUDA)");
    m.def("butterfly_transform_forward", &butterfly_transform_forward_cuda, "Butterfly Transform Forward (CUDA)");
    m.def("butterfly_transform_backward", &butterfly_transform_backward_cuda, "Butterfly Transform Backward (CUDA)");
} 