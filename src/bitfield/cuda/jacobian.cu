//! CUDA 비트 야코비안 구현
//! 
//! 비트 연산을 이용한 야코비안 계산을 GPU에서 수행합니다.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdint>
#include <cstdio>

// 비트필드 디코딩을 위한 마스크 상수들
#define PHASE_MASK    0xFF000000u  // bits 31-24
#define AMP_FINE_MASK 0x00C00000u  // bits 23-22
#define CAT_MASK      0x00300000u  // bits 21-20
#define SUB_MASK      0x000C0000u  // bits 19-18
#define IDX_MASK      0x0003FC00u  // bits 17-10
#define SIGN_MASK     0x00000200u  // bit 9
#define D_MASK        0x00000100u  // bit 8
#define AMP_MASK      0x000000FFu  // bits 7-0

// 디바이스 함수: 비트필드에서 함수 정보 추출
__device__ inline void decode_function_info(uint32_t code, uint8_t* func_idx, uint8_t* diff_order) {
    *func_idx = (code & SUB_MASK) >> 18;
    *diff_order = (code & D_MASK) >> 8;
}

// 디바이스 함수: 야코비안 대각 원소 계산
__device__ float compute_jacobian_element(uint32_t code, float x_value, uint8_t requested_diff_order) {
    // 함수 정보 추출
    uint8_t func_idx, base_diff;
    decode_function_info(code, &func_idx, &base_diff);
    
    // 전체 미분 차수 = 기본 차수 + 요청된 차수
    uint8_t total_diff = (base_diff + requested_diff_order) % 4;
    
    // 위상 정보
    float phase = float((code & PHASE_MASK) >> 24) / 255.0f * 2.0f * 3.14159265359f;
    float x_with_phase = x_value + phase;
    
    // func_idx와 total_diff에 따라 실제 함수값 계산
    float func_value = 0.0f;
    
    // func_idx: 0=sin, 1=cos, 2=-sin, 3=-cos
    switch (func_idx) {
        case 0: // sin 계열
            switch (total_diff) {
                case 0: func_value = sinf(x_with_phase); break;
                case 1: func_value = cosf(x_with_phase); break;
                case 2: func_value = -sinf(x_with_phase); break;
                case 3: func_value = -cosf(x_with_phase); break;
            }
            break;
            
        case 1: // cos 계열
            switch (total_diff) {
                case 0: func_value = cosf(x_with_phase); break;
                case 1: func_value = -sinf(x_with_phase); break;
                case 2: func_value = -cosf(x_with_phase); break;
                case 3: func_value = sinf(x_with_phase); break;
            }
            break;
            
        case 2: // -sin 계열
            switch (total_diff) {
                case 0: func_value = -sinf(x_with_phase); break;
                case 1: func_value = -cosf(x_with_phase); break;
                case 2: func_value = sinf(x_with_phase); break;
                case 3: func_value = cosf(x_with_phase); break;
            }
            break;
            
        case 3: // -cos 계열
            switch (total_diff) {
                case 0: func_value = -cosf(x_with_phase); break;
                case 1: func_value = sinf(x_with_phase); break;
                case 2: func_value = cosf(x_with_phase); break;
                case 3: func_value = -sinf(x_with_phase); break;
            }
            break;
    }
    
    // 진폭 적용
    float amp = float(code & AMP_MASK) / 255.0f;
    float amp_fine = float((code & AMP_FINE_MASK) >> 22) / 4.0f;
    float total_amp = amp + amp_fine / 4.0f;
    
    return func_value * total_amp;
}

// 비트 야코비안 계산 커널
__global__ void compute_bit_jacobian_kernel(
    const uint32_t* codes,
    const float* x_values,
    float* jacobian_diag,
    uint8_t diff_order,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    uint32_t code = codes[tid];
    float x_value = x_values[tid];
    
    jacobian_diag[tid] = compute_jacobian_element(code, x_value, diff_order);
}

// 비트 마스크를 대각 행렬로 변환하는 커널
__global__ void bit_mask_to_diagonal_kernel(
    uint32_t mask,
    float* diagonal,
    int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size || tid >= 32) return;
    
    diagonal[tid] = ((mask >> tid) & 1) ? 1.0f : 0.0f;
}

// 야코비안 전치 적용 커널 (대각 행렬이므로 단순 element-wise 곱)
__global__ void apply_jacobian_transpose_kernel(
    const float* grad_output,
    const float* jacobian_diag,
    float* grad_input,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    grad_input[tid] = grad_output[tid] * jacobian_diag[tid];
}

// 쌍곡 함수 야코비안 커널
__global__ void compute_hyperbolic_jacobian_kernel(
    const uint8_t* func_types,
    const float* x_values,
    float* jacobian_values,
    uint8_t diff_order,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    uint8_t func_type = func_types[tid];
    float x = x_values[tid];
    uint8_t cycle_idx = diff_order % 2;
    
    float func_value = 0.0f;
    
    // func_type: 0=sinh, 1=cosh
    if (func_type == 0) {
        func_value = (cycle_idx == 0) ? sinhf(x) : coshf(x);
    } else if (func_type == 1) {
        func_value = (cycle_idx == 0) ? coshf(x) : sinhf(x);
    }
    
    jacobian_values[tid] = func_value;
}

// C 인터페이스
extern "C" {
    // 비트 야코비안 계산
    void launch_bit_jacobian_kernel(
        const uint32_t* codes,
        const float* x_values,
        float* jacobian_diag,
        uint8_t diff_order,
        int n,
        cudaStream_t stream
    ) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        
        compute_bit_jacobian_kernel<<<grid_size, block_size, 0, stream>>>(
            codes, x_values, jacobian_diag, diff_order, n
        );
    }
    
    // 비트 마스크 대각 변환
    void launch_bit_mask_diagonal_kernel(
        uint32_t mask,
        float* diagonal,
        int size,
        cudaStream_t stream
    ) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        
        bit_mask_to_diagonal_kernel<<<grid_size, block_size, 0, stream>>>(
            mask, diagonal, size
        );
    }
    
    // 야코비안 전치 적용
    void launch_jacobian_transpose_kernel(
        const float* grad_output,
        const float* jacobian_diag,
        float* grad_input,
        int n,
        cudaStream_t stream
    ) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        
        apply_jacobian_transpose_kernel<<<grid_size, block_size, 0, stream>>>(
            grad_output, jacobian_diag, grad_input, n
        );
    }
    
    // 쌍곡 함수 야코비안
    void launch_hyperbolic_jacobian_kernel(
        const uint8_t* func_types,
        const float* x_values,
        float* jacobian_values,
        uint8_t diff_order,
        int n,
        cudaStream_t stream
    ) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        
        compute_hyperbolic_jacobian_kernel<<<grid_size, block_size, 0, stream>>>(
            func_types, x_values, jacobian_values, diff_order, n
        );
    }
} 