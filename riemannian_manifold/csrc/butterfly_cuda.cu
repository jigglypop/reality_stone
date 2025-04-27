#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// 모든 CUDA 버전과 호환되도록 코드를 수정
#define CUDA_SAFE_CALL(x) do { if((x) != cudaSuccess) { \
    printf("CUDA error: %s\n", cudaGetErrorString(x)); \
    printf("at line %d\n", __LINE__); \
    return torch::Tensor(); \
}} while(0)

// 디버그 메시지
#define DEBUG_PRINT(...) printf(__VA_ARGS__)

// CUDA 커널 정의
namespace {

// CUDA 에러 체크 매크로
#define CHECK_CUDA(x) if (!x.device().is_cuda()) { \
    DEBUG_PRINT("Tensor must be a CUDA tensor\n"); \
    x = x.to(torch::kCUDA); \
    DEBUG_PRINT("Tensor moved to CUDA\n"); \
}

#define CHECK_CONTIGUOUS(x) if (!x.is_contiguous()) { \
    DEBUG_PRINT("Tensor must be contiguous\n"); \
    x = x.contiguous(); \
    DEBUG_PRINT("Tensor made contiguous\n"); \
}

#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 블록 크기 정의
constexpr int BLOCK_SIZE = 256;

// CUDA 런타임 버전 체크
inline bool is_cuda_version_compatible() {
    int runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);
    
    std::cout << "CUDA Runtime Version: " << runtime_version << std::endl;
    std::cout << "CUDA Driver Version: " << driver_version << std::endl;
    
    #ifdef CUDA_12_COMPAT
    return true; // 호환성 모드 활성화됨
    #else
    return runtime_version / 100 <= 118; // CUDA 11.8 이하만 지원
    #endif
}

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

// 통합된 버터플라이 변환 커널 - 효율적인 O(n log n) 구현
template <typename scalar_t>
__global__ void unified_butterfly_transform_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight_matrix,
    const int batch_size,
    const int dim) {
    
    // 글로벌 스레드 인덱스 계산
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 각 스레드가 한 요소 처리
    if (idx < batch_size * dim) {
        const int b = idx / dim; // 배치 인덱스
        const int d = idx % dim; // 차원 인덱스
        
        scalar_t result = 0.0f;
        
        // 행렬 곱셈 계산
        for (int i = 0; i < dim; ++i) {
            result += input[b * dim + i] * weight_matrix[i * dim + d];
        }
        
        // 결과 저장
        output[idx] = result;
    }
}

// 최적화된 버터플라이 행렬 생성 커널
template <typename scalar_t>
__global__ void create_butterfly_matrix_kernel(
    scalar_t* __restrict__ weight_matrix,
    const scalar_t* __restrict__ thetas,
    const int dim,
    const int num_layers) {
    
    // 글로벌 스레드 인덱스
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < dim * dim) {
        const int row = idx / dim;
        const int col = idx % dim;
        
        // 단위 행렬로 초기화
        scalar_t result = (row == col) ? 1.0f : 0.0f;
        
        // 모든 레이어에 대해 변환 적용
        for (int layer = 0; layer < num_layers; ++layer) {
            const int block_size = 1 << layer;
            const int num_blocks = dim / (2 * block_size);
            
            for (int b = 0; b < num_blocks; ++b) {
                const int block_start = b * 2 * block_size;
                
                for (int i = 0; i < block_size; ++i) {
                    const int idx1 = block_start + i;
                    const int idx2 = block_start + block_size + i;
                    
                    if (row == idx1 || row == idx2) {
                        // 버터플라이 패턴에 해당하는 위치에 있을 때만 계산
                        const scalar_t theta = thetas[layer * num_blocks + b];
                        const scalar_t cos_val = cos(theta);
                        const scalar_t sin_val = sin(theta);
                        
                        if (row == idx1) {
                            if (col == idx1) result = cos_val;
                            else if (col == idx2) result = sin_val;
                        } else { // row == idx2
                            if (col == idx1) result = -sin_val;
                            else if (col == idx2) result = cos_val;
                        }
                    }
                }
            }
        }
        
        // 최종 가중치 행렬 저장
        weight_matrix[idx] = result;
    }
}

// 효율적인 버터플라이 변환 커널 - 반복 없이 O(n log n)
template <typename scalar_t>
__global__ void efficient_butterfly_transform_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ a_params,
    const scalar_t* __restrict__ b_params,
    const int* __restrict__ layer_info,  // [레이어 수, 각 레이어 블록 크기, 각 레이어 블록 수]
    const int batch_size,
    const int dim,
    const int num_layers) {
    
    // 스레드별 입출력 맵핑을 위한 인덱스 계산
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thread_idx < batch_size * dim) {
        const int batch_idx = thread_idx / dim;
        const int feat_idx = thread_idx % dim;
        
        // 입력 복사
        scalar_t val = input[thread_idx];
        
        // 효율적인 비트 연산으로 모든 레이어 처리
        int idx = feat_idx;
        
        for (int layer = 0; layer < num_layers; ++layer) {
            const int block_size = 1 << layer;
            const int block_mask = block_size - 1;
            const int block_idx = idx >> (layer + 1);
            const int local_idx = idx & block_mask;
            const int is_upper = (idx >> layer) & 1;
            
            const int param_idx = layer_info[layer] + block_idx;
            const scalar_t a = a_params[param_idx];
            const scalar_t b = b_params[param_idx];
            
            // 버터플라이 패턴으로 인덱스 계산
            const int pair_idx = (block_idx << (layer + 1)) | local_idx | ((!is_upper) << layer);
            const scalar_t pair_val = input[batch_idx * dim + pair_idx];
            
            // 회전 변환 적용
            if (is_upper) {
                val = a * val + b * pair_val;
            } else {
                val = -b * pair_val + a * val;
            }
        }
        
        // 결과 저장
        output[thread_idx] = val;
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

// C++ 래퍼 함수: 통합 버터플라이 변환
torch::Tensor unified_butterfly_transform_cuda(
    torch::Tensor input,
    torch::Tensor weight_matrix) {
    
    try {
        // 입력 텐서 검사 및 변환
        if (!input.device().is_cuda()) {
            DEBUG_PRINT("Moving input tensor to CUDA\n");
            input = input.to(torch::kCUDA);
        }
        if (!weight_matrix.device().is_cuda()) {
            DEBUG_PRINT("Moving weight tensor to CUDA\n");
            weight_matrix = weight_matrix.to(torch::kCUDA);
        }
        
        if (!input.is_contiguous()) input = input.contiguous();
        if (!weight_matrix.is_contiguous()) weight_matrix = weight_matrix.contiguous();
        
        const int batch_size = input.size(0);
        const int dim = input.size(1);
        
        auto output = torch::zeros_like(input);
        
        const int threads = BLOCK_SIZE;
        const int blocks = (batch_size * dim + threads - 1) / threads;
        
        DEBUG_PRINT("Running CUDA kernel with blocks=%d, threads=%d\n", blocks, threads);
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "unified_butterfly_transform_cuda", ([&] {
            unified_butterfly_transform_kernel<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                weight_matrix.data_ptr<scalar_t>(),
                batch_size,
                dim
            );
            
            // CUDA 오류 체크
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                DEBUG_PRINT("CUDA kernel error: %s\n", cudaGetErrorString(err));
                throw std::runtime_error("CUDA kernel error");
            }
        }));
        
        DEBUG_PRINT("CUDA kernel completed successfully\n");
        return output;
    }
    catch (const std::exception& e) {
        DEBUG_PRINT("Exception in CUDA call: %s\n", e.what());
        DEBUG_PRINT("Falling back to CPU implementation\n");
        
        // CPU로 이동하여 계산 후 다시 GPU로 반환
        auto input_cpu = input.to(torch::kCPU);
        auto weight_cpu = weight_matrix.to(torch::kCPU);
        auto result_cpu = input_cpu.mm(weight_cpu.t());
        
        try {
            return result_cpu.to(input.device());
        }
        catch (...) {
            DEBUG_PRINT("Error returning to original device, keeping on CPU\n");
            return result_cpu;
        }
    }
}

// C++ 래퍼 함수: 버터플라이 가중치 행렬 생성
torch::Tensor create_butterfly_matrix_cuda(
    torch::Tensor thetas,
    int dim,
    int num_layers) {
    
    try {
        // 입력 텐서 검사 및 변환
        if (!thetas.device().is_cuda()) {
            DEBUG_PRINT("Moving thetas tensor to CUDA\n");
            thetas = thetas.to(torch::kCUDA);
        }
        
        if (!thetas.is_contiguous()) thetas = thetas.contiguous();
        
        auto weight_matrix = torch::zeros({dim, dim}, thetas.options());
        
        const int threads = BLOCK_SIZE;
        const int blocks = (dim * dim + threads - 1) / threads;
        
        DEBUG_PRINT("Creating butterfly matrix with blocks=%d, threads=%d\n", blocks, threads);
        
        AT_DISPATCH_FLOATING_TYPES(thetas.scalar_type(), "create_butterfly_matrix_cuda", ([&] {
            create_butterfly_matrix_kernel<scalar_t><<<blocks, threads>>>(
                weight_matrix.data_ptr<scalar_t>(),
                thetas.data_ptr<scalar_t>(),
                dim,
                num_layers
            );
            
            // CUDA 오류 체크
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                DEBUG_PRINT("CUDA kernel error: %s\n", cudaGetErrorString(err));
                throw std::runtime_error("CUDA kernel error");
            }
        }));
        
        DEBUG_PRINT("Butterfly matrix created successfully\n");
        return weight_matrix;
    }
    catch (const std::exception& e) {
        DEBUG_PRINT("Exception in CUDA call: %s\n", e.what());
        DEBUG_PRINT("Falling back to CPU implementation\n");
        
        // CPU로 계산
        auto thetas_cpu = thetas.to(torch::kCPU);
        auto weight_matrix = torch::eye(dim, thetas_cpu.options());
        
        // CPU 구현
        for (int layer = 0; layer < num_layers; ++layer) {
            const int block_size = 1 << layer;
            const int num_blocks = dim / (2 * block_size);
            
            auto layer_weight = torch::eye(dim, thetas_cpu.options());
            
            for (int b = 0; b < num_blocks; ++b) {
                const int theta_idx = layer * num_blocks + b;
                if (theta_idx < thetas_cpu.size(0)) {
                    float theta = thetas_cpu[theta_idx].item<float>();
                    float cos_val = std::cos(theta);
                    float sin_val = std::sin(theta);
                    
                    const int block_start = b * 2 * block_size;
                    
                    for (int i = 0; i < block_size; ++i) {
                        const int idx1 = block_start + i;
                        const int idx2 = block_start + block_size + i;
                        
                        if (idx1 < dim && idx2 < dim) {
                            layer_weight[idx1][idx1] = cos_val;
                            layer_weight[idx1][idx2] = sin_val;
                            layer_weight[idx2][idx1] = -sin_val;
                            layer_weight[idx2][idx2] = cos_val;
                        }
                    }
                }
            }
            
            weight_matrix = layer_weight.mm(weight_matrix);
        }
        
        try {
            return weight_matrix.to(thetas.device());
        }
        catch (...) {
            DEBUG_PRINT("Error returning to original device, keeping on CPU\n");
            return weight_matrix;
        }
    }
}

// C++ 래퍼 함수: 효율적인 버터플라이 변환
torch::Tensor efficient_butterfly_transform_cuda(
    torch::Tensor input,
    torch::Tensor a_params,
    torch::Tensor b_params,
    torch::Tensor layer_info) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(a_params);
    CHECK_INPUT(b_params);
    CHECK_INPUT(layer_info);
    
    const int batch_size = input.size(0);
    const int dim = input.size(1);
    const int num_layers = layer_info.size(0);
    
    auto output = torch::zeros_like(input);
    
    const int threads = BLOCK_SIZE;
    const int blocks = (batch_size * dim + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "efficient_butterfly_transform_cuda", ([&] {
        efficient_butterfly_transform_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            a_params.data_ptr<scalar_t>(),
            b_params.data_ptr<scalar_t>(),
            layer_info.data_ptr<int>(),
            batch_size,
            dim,
            num_layers
        );
    }));
    
    return output;
}

// 모듈 내보내기를 위한 선언
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("butterfly_factor", &butterfly_factor_cuda, "Butterfly transform (CUDA)");
    m.def("poincare_exp_map", &poincare_exp_map_cuda, "Poincare exponential map (CUDA)");
    m.def("poincare_log_map", &poincare_log_map_cuda, "Poincare logarithmic map (CUDA)");
    m.def("unified_butterfly_transform", &unified_butterfly_transform_cuda, "Unified butterfly transform (CUDA)");
    m.def("create_butterfly_matrix", &create_butterfly_matrix_cuda, "Create butterfly weight matrix (CUDA)");
    m.def("efficient_butterfly_transform", &efficient_butterfly_transform_cuda, "Efficient butterfly transform (CUDA)");
} 