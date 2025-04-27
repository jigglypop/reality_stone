#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// CUDA 커널 정의
namespace {

// CUDA 에러 체크 매크로
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while(0)

// CUDA 커널: 버터플라이 변환 구현
template <typename scalar_t>
__global__ void butterfly_factor_cuda_kernel(
    scalar_t* __restrict__ result,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ params,
    const int n, 
    const int block_size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n / 2) {
        // 블록 인덱스 계산
        const int b = idx / (block_size / 2);
        const int i = (idx % (block_size / 2)) * 2;
        const int pos = b * block_size + i;
        
        // 파라미터 인덱스
        const int param_idx = b * 2;
        
        // 회전 파라미터
        const scalar_t a = params[param_idx];
        const scalar_t b_val = params[param_idx + 1];
        
        // Givens 회전 적용
        const scalar_t temp1 = a * input[pos] + b_val * input[pos + 1];
        const scalar_t temp2 = -b_val * input[pos] + a * input[pos + 1];
        
        // 결과 업데이트
        result[pos] = temp1;
        result[pos + 1] = temp2;
    }
}

// CUDA 커널: 배치 포인카레 지수 사상 구현
template <typename scalar_t>
__global__ void poincare_exp_map_cuda_kernel(
    scalar_t* __restrict__ result,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ v,
    const scalar_t c,
    const int batch_size, 
    const int dim) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // 배치 및 차원을 반복
    for (int b = tid; b < batch_size; b += stride) {
        const scalar_t eps = 1e-8;
        
        // x의 노름 제곱 계산
        scalar_t x_norm_squared = 0.0;
        for (int d = 0; d < dim; d++) {
            const scalar_t x_val = x[b * dim + d];
            x_norm_squared += x_val * x_val;
        }
        
        // 공형적 인자 계산
        const scalar_t lambda_x = 2.0 / (1.0 - c * x_norm_squared + eps);
        
        // v의 노름 계산
        scalar_t v_norm = 0.0;
        for (int d = 0; d < dim; d++) {
            const scalar_t v_val = v[b * dim + d];
            v_norm += v_val * v_val;
        }
        v_norm = sqrt(v_norm + eps);
        
        // 스케일 계수 계산
        const scalar_t sqrt_c = sqrt(c);
        const scalar_t scale = tanh(sqrt_c * lambda_x * v_norm / 2.0) / (sqrt_c * v_norm);
        
        // 스케일된 v 계산 및 분자 계산
        scalar_t scaled_v[1024]; // 가정: 최대 차원이 1024 이하
        for (int d = 0; d < dim; d++) {
            scaled_v[d] = scale * v[b * dim + d];
        }
        
        // x와 scaled_v의 내적 계산
        scalar_t x_scaled_v_inner = 0.0;
        for (int d = 0; d < dim; d++) {
            x_scaled_v_inner += x[b * dim + d] * scaled_v[d];
        }
        
        // scaled_v의 노름 제곱 계산
        scalar_t scaled_v_norm_squared = 0.0;
        for (int d = 0; d < dim; d++) {
            scaled_v_norm_squared += scaled_v[d] * scaled_v[d];
        }
        
        // 분모 계산
        const scalar_t denominator = 1.0 - 2.0 * c * x_scaled_v_inner + c * c * x_norm_squared * scaled_v_norm_squared + eps;
        
        // 결과 계산
        for (int d = 0; d < dim; d++) {
            result[b * dim + d] = x[b * dim + d] + (1.0 - c * x_norm_squared) * scaled_v[d] / denominator;
        }
    }
}

// CUDA 커널: 배치 포인카레 로그 사상 구현
template <typename scalar_t>
__global__ void poincare_log_map_cuda_kernel(
    scalar_t* __restrict__ result,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    const scalar_t c,
    const int batch_size, 
    const int dim) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // 배치 및 차원을 반복
    for (int b = tid; b < batch_size; b += stride) {
        const scalar_t eps = 1e-8;
        
        // x의 노름 제곱 계산
        scalar_t x_norm_squared = 0.0;
        for (int d = 0; d < dim; d++) {
            const scalar_t x_val = x[b * dim + d];
            x_norm_squared += x_val * x_val;
        }
        
        // y의 노름 제곱 계산
        scalar_t y_norm_squared = 0.0;
        for (int d = 0; d < dim; d++) {
            const scalar_t y_val = y[b * dim + d];
            y_norm_squared += y_val * y_val;
        }
        
        // x와 y의 내적 계산
        scalar_t xy_inner_prod = 0.0;
        for (int d = 0; d < dim; d++) {
            xy_inner_prod += x[b * dim + d] * y[b * dim + d];
        }
        
        // 공형적 인자 계산
        const scalar_t lambda_x = 2.0 / (1.0 - c * x_norm_squared + eps);
        
        // 분자 계산을 위한 임시 배열
        scalar_t diff[1024]; // 가정: 최대 차원이 1024 이하
        for (int d = 0; d < dim; d++) {
            diff[d] = (1.0 - 2.0 * c * xy_inner_prod + c * y_norm_squared) * x[b * dim + d] - 
                     (1.0 - c * x_norm_squared) * y[b * dim + d];
        }
        
        // 분모 계산
        const scalar_t denominator = 1.0 - 2.0 * c * xy_inner_prod + c * c * x_norm_squared * y_norm_squared + eps;
        
        // 차이 벡터 정규화
        for (int d = 0; d < dim; d++) {
            diff[d] /= denominator;
        }
        
        // 차이 벡터의 노름 계산
        scalar_t diff_norm = 0.0;
        for (int d = 0; d < dim; d++) {
            diff_norm += diff[d] * diff[d];
        }
        diff_norm = sqrt(diff_norm + eps);
        
        // 최종 결과 계산
        const scalar_t sqrt_c = sqrt(c);
        const scalar_t atanhc = atanh(sqrt_c * diff_norm) / (sqrt_c * diff_norm);
        
        for (int d = 0; d < dim; d++) {
            result[b * dim + d] = 2.0 / (sqrt_c * lambda_x) * atanhc * diff[d];
        }
    }
}

} // namespace

// C++ 래퍼 함수: 버터플라이 변환
torch::Tensor butterfly_factor_cuda(
    torch::Tensor input,
    torch::Tensor params,
    int layer) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(params);
    
    const int n = input.size(0);
    const int block_size = 1 << layer;
    const int num_blocks = n / block_size;
    
    // 결과 텐서 초기화
    auto result = torch::zeros_like(input);
    
    // 이미 input에서 복사
    result.copy_(input);
    
    // 스레드 수 계산
    const int threads = 256;
    const int blocks = (n / 2 + threads - 1) / threads;
    
    // 데이터 타입에 따라 커널 실행
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_factor_cuda", ([&] {
        butterfly_factor_cuda_kernel<scalar_t><<<blocks, threads>>>(
            result.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            params.data_ptr<scalar_t>(),
            n,
            block_size
        );
    }));
    
    return result;
}

// C++ 래퍼 함수: 포인카레 지수 사상
torch::Tensor poincare_exp_map_cuda(
    torch::Tensor x,
    torch::Tensor v,
    double c) {
    
    CHECK_INPUT(x);
    CHECK_INPUT(v);
    
    const int batch_size = x.size(0);
    int dim;
    
    // 입력 차원에 따라 처리
    if (x.dim() == 1) {
        dim = x.size(0);
    } else {
        dim = x.size(1);
    }
    
    // 결과 텐서 초기화
    auto result = torch::zeros_like(x);
    
    // 스레드 수 계산
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    // 데이터 타입에 따라 커널 실행
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "poincare_exp_map_cuda", ([&] {
        poincare_exp_map_cuda_kernel<scalar_t><<<blocks, threads>>>(
            result.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            static_cast<scalar_t>(c),
            batch_size,
            dim
        );
    }));
    
    return result;
}

// C++ 래퍼 함수: 포인카레 로그 사상
torch::Tensor poincare_log_map_cuda(
    torch::Tensor x,
    torch::Tensor y,
    double c) {
    
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    
    const int batch_size = x.size(0);
    int dim;
    
    // 입력 차원에 따라 처리
    if (x.dim() == 1) {
        dim = x.size(0);
    } else {
        dim = x.size(1);
    }
    
    // 결과 텐서 초기화
    auto result = torch::zeros_like(x);
    
    // 스레드 수 계산
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    // 데이터 타입에 따라 커널 실행
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "poincare_log_map_cuda", ([&] {
        poincare_log_map_cuda_kernel<scalar_t><<<blocks, threads>>>(
            result.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            static_cast<scalar_t>(c),
            batch_size,
            dim
        );
    }));
    
    return result;
}

// 모듈 내보내기를 위한 선언
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("butterfly_factor", &butterfly_factor_cuda, "Butterfly transform (CUDA)");
    m.def("poincare_exp_map", &poincare_exp_map_cuda, "Poincare exponential map (CUDA)");
    m.def("poincare_log_map", &poincare_log_map_cuda, "Poincare logarithmic map (CUDA)");
} 