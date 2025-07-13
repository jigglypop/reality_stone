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

// 디바이스 함수: 32비트 코드를 8개 필드로 디코딩
__device__ inline void decode_bitfield(uint32_t code, 
                                      uint8_t* cat, uint8_t* sub, uint8_t* idx, 
                                      uint8_t* sign, uint8_t* d, uint8_t* amp, 
                                      uint8_t* amp_fine, uint8_t* phase) {
    *phase = (code & PHASE_MASK) >> 24;
    *amp_fine = (code & AMP_FINE_MASK) >> 22;
    *cat = (code & CAT_MASK) >> 20;
    *sub = (code & SUB_MASK) >> 18;
    *idx = (code & IDX_MASK) >> 10;
    *sign = (code & SIGN_MASK) >> 9;
    *d = (code & D_MASK) >> 8;
    *amp = code & AMP_MASK;
}

// 디바이스 함수: 리만 기저 함수 lookup_and_apply
__device__ inline float lookup_and_apply_gpu(uint8_t cat, uint8_t sub, uint8_t d, float r, uint8_t phase) {
    const float EPS = 1e-7f;
    const float phase_rad = (float(phase) / 255.0f) * 2.0f * 3.14159265359f;
    
    switch (cat) {
        case 0: // Poincaré 볼 기하학
            switch (sub) {
                case 0: // 기본 푸앵카레 스케일링
                    return (d == 0) ? tanhf(r * 0.5f) : 2.0f * tanhf(r * 0.25f);
                case 1: // 쌍곡 함수족
                    return (d == 0) ? sinhf(r) / fmaxf(1.0f + coshf(r), EPS) : tanhf(r);
                case 2: // 삼각 함수족 - 위상 사용
                    if (d == 0) {
                        return sinf(r) * cosf(phase_rad) / fmaxf(r, EPS);
                    } else {
                        return cosf(r) * sinf(phase_rad);
                    }
                default: // 지수/로그 함수족
                    return (d == 0) ? (expf(r) - 1.0f) / fmaxf(r, EPS) : logf(fmaxf(1.0f + r, EPS));
            }
        case 1: // Lorentz 기하학
            switch (sub) {
                case 0:
                    return (d == 0) ? sinhf(r) : coshf(r);
                case 1:
                    return (d == 0) ? r * sinhf(r) : r * coshf(r);
                default:
                    return (d == 0) ? tanhf(r * 0.5f) : 1.0f / coshf(r);
            }
        case 2: // Klein 기하학
            switch (sub) {
                case 0:
                    return (d == 0) ? 2.0f * r / (1.0f + r * r) : (1.0f - r * r) / (1.0f + r * r);
                default:
                    return (d == 0) ? sinf(r) : cosf(r);
            }
        default: // 특수 함수
            return (d == 0) ? expf(-r * r * 0.5f) : r * expf(-r * r * 0.5f); // Gaussian
    }
}

// 최적화된 비트필드 기반 직접 추론 커널
__global__ void gemm_hyper_bit_kernel(const float* x, const uint32_t* codes,
                                     const float* basis_table, float delta,
                                     float* output, int batch_size, int input_dim,
                                     int output_dim, int basis_size) {
    // 각 스레드가 하나의 출력 요소를 담당
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * output_dim;
    
    if (tid >= total_elements) return;
    
    int batch_idx = tid / output_dim;
    int out_idx = tid % output_dim;
    
    // 해당 출력에 대한 코드 디코딩
    uint32_t code = codes[out_idx];
    uint8_t cat, sub, idx, sign, d, amp;
    uint8_t amp_fine, phase;
    decode_bitfield(code, &cat, &sub, &idx, &sign, &d, &amp, &amp_fine, &phase);
    
    // 필요한 기저 벡터와의 내적만 계산
    float dot = 0.0f;
    const float* x_row = x + batch_idx * input_dim;
    const float* basis_row = basis_table + idx * input_dim;
    
    // 벡터화된 내적 계산
    for (int i = 0; i < input_dim; i++) {
        dot += x_row[i] * basis_row[i];
    }
    
    // 스케일 계산
    float r = (float(amp) * 4.0f + float(amp_fine)) * delta;
    if (sign == 1) r = -r;
    float scale = lookup_and_apply_gpu(cat, sub, d, r, phase);
    
    // 최종 출력
    output[tid] = scale * dot;
}

// 잔차 추론을 위한 저정밀도 GEMM 커널
__global__ void residual_gemm_kernel(const float* x, const float* residual_weights,
                                    float* output, int batch_size, int input_dim, 
                                    int output_dim) {
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || out_idx >= output_dim) return;
    
    float sum = 0.0f;
    for (int i = 0; i < input_dim; i++) {
        sum += x[batch_idx * input_dim + i] * residual_weights[out_idx * input_dim + i];
    }
    
    // 기존 출력에 잔차 결과 추가
    output[batch_idx * output_dim + out_idx] += sum;
}

// 잔차를 포함한 통합 CUDA 커널 (캐시된 메모리 사용)
__global__ void gemm_hyper_bit_cached_kernel(
    const float* x,              // [batch_size, input_dim]
    const uint32_t* codes,       // [output_dim] - GPU에 영구 저장
    const float* basis_table,    // [basis_size, input_dim] - GPU에 영구 저장
    const float* residual_weights, // [output_dim, input_dim] - GPU에 영구 저장 (이미 FP32로 변환됨)
    float delta,
    float* output,               // [batch_size, output_dim]
    int batch_size,
    int input_dim,
    int output_dim,
    int basis_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / output_dim;
    int out_idx = tid % output_dim;
    if (batch_idx >= batch_size || out_idx >= output_dim) return;
    // 1. 비트필드 디코딩
    uint32_t code = codes[out_idx];
    uint8_t cat, sub, idx, sign, d, amp, amp_fine, phase;
    decode_bitfield(code, &cat, &sub, &idx, &sign, &d, &amp, &amp_fine, &phase);
    // 2. r 값 계산 (10비트 정밀도)
    float r = float(amp * 4 + amp_fine) * delta;
    if (sign == 1) r = -r;
    // 3. 스케일 팩터 계산
    float scale = lookup_and_apply_gpu(cat, sub, d, r, phase);
    // 4. 기저 벡터와의 내적 계산
    float sum = 0.0f;
    const float* x_row = x + batch_idx * input_dim;
    const float* basis_row = basis_table + idx * input_dim;
    
    for (int i = 0; i < input_dim; i++) {
        sum += x_row[i] * basis_row[i];
    }
    // 5. 스케일 적용
    float result = sum * scale;
    // 6. 잔차 추가 (있는 경우)
    if (residual_weights != nullptr) {
        const float* residual_row = residual_weights + out_idx * input_dim;
        float residual_sum = 0.0f;
        for (int i = 0; i < input_dim; i++) {
            residual_sum += x_row[i] * residual_row[i];
        }
        result += residual_sum;
    }
    // 7. 결과 저장
    output[batch_idx * output_dim + out_idx] = result;
}

// 공유 메모리를 활용하는 최적화된 CUDA 커널
__global__ void gemm_hyper_bit_optimized_kernel(
    const float* x,              // [batch_size, input_dim]
    const uint32_t* codes,       // [output_dim] - GPU에 영구 저장
    const float* basis_table,    // [basis_size, input_dim] - GPU에 영구 저장
    const uint32_t* residual_codes, // [output_dim] - 잔차 코드 (있는 경우)
    float delta,
    float residual_delta,
    float* output,               // [batch_size, output_dim]
    int batch_size,
    int input_dim,
    int output_dim,
    int basis_size
) {
    // 공유 메모리 선언
    extern __shared__ float shared_mem[];
    float* shared_x = shared_mem; // 입력 데이터 캐싱
    float* shared_basis = &shared_mem[blockDim.x]; // 기저 벡터 캐싱
    
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = global_tid / output_dim;
    int out_idx = global_tid % output_dim;
    
    if (batch_idx >= batch_size || out_idx >= output_dim) return;
    
    // 1. 입력 데이터를 공유 메모리로 로드 (협력적 로딩)
    const float* x_row = x + batch_idx * input_dim;
    for (int i = tid; i < input_dim; i += blockDim.x) {
        if (i < input_dim) {
            shared_x[i] = x_row[i];
        }
    }
    __syncthreads();
    
    // 2. 주 코드 처리
    uint32_t code = codes[out_idx];
    uint8_t cat, sub, idx, sign, d, amp, amp_fine, phase;
    decode_bitfield(code, &cat, &sub, &idx, &sign, &d, &amp, &amp_fine, &phase);
    
    // 경계 체크
    if (idx >= basis_size) {
        output[batch_idx * output_dim + out_idx] = 0.0f;
        return;
    }
    
    // 3. 기저 벡터를 공유 메모리로 로드
    const float* basis_row = basis_table + idx * input_dim;
    for (int i = tid; i < input_dim; i += blockDim.x) {
        if (i < input_dim) {
            shared_basis[i] = basis_row[i];
        }
    }
    __syncthreads();
    
    // 4. 내적 계산 (공유 메모리 사용)
    float sum = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < input_dim; i++) {
        sum += shared_x[i] * shared_basis[i];
    }
    
    // 5. 스케일 팩터 적용
    float r = float(amp * 4 + amp_fine) * delta;
    if (sign == 1) r = -r;
    float scale = lookup_and_apply_gpu(cat, sub, d, r, phase);
    float result = sum * scale;
    
    // 6. 잔차 처리 (있는 경우)
    if (residual_codes != nullptr) {
        uint32_t res_code = residual_codes[out_idx];
        if (res_code != 0) {
            // 잔차 디코딩
            decode_bitfield(res_code, &cat, &sub, &idx, &sign, &d, &amp, &amp_fine, &phase);
            
            if (idx < basis_size) {
                // 잔차 기저 벡터 로드 및 계산
                const float* res_basis_row = basis_table + idx * input_dim;
                float res_sum = 0.0f;
                
                #pragma unroll 4
                for (int i = 0; i < input_dim; i++) {
                    res_sum += shared_x[i] * res_basis_row[i];
                }
                
                float res_r = float(amp * 4 + amp_fine) * residual_delta;
                if (sign == 1) res_r = -res_r;
                float res_scale = lookup_and_apply_gpu(cat, sub, d, res_r, phase);
                result += res_sum * res_scale;
            }
        }
    }
    
    // 7. 결과 저장
    output[batch_idx * output_dim + out_idx] = result;
}

// 간단한 잔차 디코딩 함수
__device__ inline float decode_residual_gpu(uint32_t res_code, float residual_delta) {
    // 간단한 예시: 코드를 부동소수점으로 변환
    return (float)(res_code & 0xFFFF) * residual_delta - 0.5f;
}

// 공유 메모리를 활용하는 최적화된 직접 포인터 커널
__global__ void gemm_hyper_bit_direct_kernel(
    const float* __restrict__ input,       // PyTorch 입력 텐서의 GPU 포인터
    const uint32_t* __restrict__ codes,
    const float* __restrict__ basis_table,
    const uint32_t* __restrict__ residual_codes,
    float delta,
    float residual_delta,
    float* __restrict__ output,           // PyTorch 출력 텐서의 GPU 포인터
    int batch_size,
    int input_dim,
    int output_dim,
    int basis_size
) {
    // 공유 메모리 선언 (타일 크기는 블록 크기에 맞춤)
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;  // [TILE_SIZE][input_dim]
    
    const int TILE_SIZE = blockDim.x;
    const int tid = threadIdx.x;
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_idx >= batch_size * output_dim) return;
    
    const int batch_idx = global_idx / output_dim;
    const int out_idx = global_idx % output_dim;
    
    // 1. 입력을 공유 메모리로 로드 (coalesced access)
    if (tid < TILE_SIZE && batch_idx < batch_size) {
        for (int i = 0; i < input_dim; i += TILE_SIZE) {
            if (i + tid < input_dim) {
                shared_input[tid * input_dim + i + tid] = input[batch_idx * input_dim + i + tid];
            }
        }
    }
    __syncthreads();
    
    // 2. 비트필드 디코딩 (레지스터에 저장)
    uint32_t code = codes[out_idx];
    uint8_t cat = (code >> 24) & 0xFF;
    uint8_t sub = (code >> 21) & 0x07;
    uint8_t idx = (code >> 13) & 0xFF;
    uint8_t sign = (code >> 12) & 0x01;
    uint8_t d = (code >> 9) & 0x07;
    uint8_t amp = (code >> 2) & 0x7F;
    uint8_t amp_fine = code & 0x03;
    uint8_t phase = (code >> 11) & 0x01;
    
    // 3. r 값 계산
    float r = float(amp * 4 + amp_fine) * delta;
    if (sign == 1) r = -r;
    
    // 4. 스케일 팩터 계산 (인라인 최적화)
    float scale = lookup_and_apply_gpu(cat, sub, d, r, phase);
    
    // 5. 기저 벡터와의 내적 (공유 메모리 활용)
    float sum = 0.0f;
    const float* basis_row = basis_table + idx * input_dim;
    
    // Vectorized load (float4 사용)
    const int vec_size = 4;
    const int vec_dim = input_dim / vec_size;
    
    if (batch_idx < batch_size) {
        // 벡터화된 연산
        for (int i = 0; i < vec_dim; i++) {
            float4 a = reinterpret_cast<const float4*>(&shared_input[tid * input_dim])[i];
            float4 b = reinterpret_cast<const float4*>(basis_row)[i];
            sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        
        // 남은 요소 처리
        for (int i = vec_dim * vec_size; i < input_dim; i++) {
            sum += shared_input[tid * input_dim + i] * basis_row[i];
        }
        
        // 6. 스케일 적용
        float result = sum * scale;
        
        // 7. 잔차 추가 (있는 경우) - Fused operation
        if (residual_codes != nullptr) {
            uint32_t res_code = residual_codes[out_idx];
            // 간단한 잔차 디코딩 및 적용
            float res_val = decode_residual_gpu(res_code, residual_delta);
            result += res_val * sum;  // 잔차도 입력과의 내적에 비례
        }
        
        // 8. 결과 저장 (coalesced write)
        output[batch_idx * output_dim + out_idx] = result;
    }
}

// Backward 커널 (최적화된 버전)
__global__ void gemm_hyper_bit_backward_direct_kernel(
    const float* __restrict__ grad_output,
    const uint32_t* __restrict__ codes,
    const float* __restrict__ basis_table,
    const uint32_t* __restrict__ residual_codes,
    float delta,
    float residual_delta,
    float* __restrict__ grad_input,
    int batch_size,
    int input_dim,
    int output_dim,
    int basis_size
) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_idx >= batch_size * input_dim) return;
    
    const int batch_idx = global_idx / input_dim;
    const int in_idx = global_idx % input_dim;
    
    float grad_sum = 0.0f;
    
    // 모든 출력 차원에 대해 역전파 계산
    for (int out_idx = 0; out_idx < output_dim; out_idx++) {
        // 비트필드 디코딩
        uint32_t code = codes[out_idx];
        uint8_t cat = (code >> 24) & 0xFF;
        uint8_t sub = (code >> 21) & 0x07;
        uint8_t idx = (code >> 13) & 0xFF;
        uint8_t sign = (code >> 12) & 0x01;
        uint8_t d = (code >> 9) & 0x07;
        uint8_t amp = (code >> 2) & 0x7F;
        uint8_t amp_fine = code & 0x03;
        uint8_t phase = (code >> 11) & 0x01;
        
        // r 값과 스케일 계산
        float r = float(amp * 4 + amp_fine) * delta;
        if (sign == 1) r = -r;
        float scale = lookup_and_apply_gpu(cat, sub, d, r, phase);
        
        // 기저 벡터 가중치
        float basis_weight = basis_table[idx * input_dim + in_idx];
        
        // 그래디언트 기여도 계산
        float grad_out = grad_output[batch_idx * output_dim + out_idx];
        grad_sum += grad_out * scale * basis_weight;
        
        // 잔차 항목의 기여도
        if (residual_codes != nullptr) {
            uint32_t res_code = residual_codes[out_idx];
            float res_val = decode_residual_gpu(res_code, residual_delta);
            grad_sum += grad_out * res_val * basis_weight;
        }
    }
    
    // 최종 그래디언트 저장
    grad_input[batch_idx * input_dim + in_idx] = grad_sum;
}

// INT8 최적화된 비트필드 추론 커널
__global__ void gemm_hyper_bit_int8_kernel(
    const float* __restrict__ x,
    const uint32_t* __restrict__ codes,
    const int8_t* __restrict__ basis_table_int8,
    const float* __restrict__ basis_scales,
    const float delta,
    float* __restrict__ output,
    const int batch_size,
    const int in_features,
    const int out_features,
    const int basis_size
) {
    extern __shared__ float shared_mem[];
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_features;
    
    if (tid >= total_elements) return;
    
    const int batch_idx = tid / out_features;
    const int out_idx = tid % out_features;
    
    // 코드 디코딩
    const uint32_t code = codes[out_idx];
    const uint8_t phase = (code >> 24) & 0xFF;
    const uint8_t amp_fine = (code >> 22) & 0x03;
    const uint8_t cat = (code >> 20) & 0x03;
    const uint8_t sub = (code >> 18) & 0x03;
    const uint16_t idx = (code >> 10) & 0xFF;
    const uint8_t sign = (code >> 9) & 0x01;
    const uint8_t d = (code >> 8) & 0x01;
    const uint8_t amp = code & 0xFF;
    
    // 스케일 팩터 계산
    const float r = (amp * 4.0f + amp_fine) * delta;
    const float signed_r = sign ? -r : r;
    const float scale_factor = lookup_and_apply_gpu(cat, sub, d, signed_r, phase);
    
    // 기저 벡터의 스케일
    const float basis_scale = basis_scales[idx];
    
    // INT8 내적 계산 (벡터화)
    float sum = 0.0f;
    const int8_t* basis_row = &basis_table_int8[idx * in_features];
    const float* input_row = &x[batch_idx * in_features];
    
    // 4개씩 벡터화 처리
    const int vec_size = in_features / 4;
    const int remainder = in_features % 4;
    
    #pragma unroll 4
    for (int i = 0; i < vec_size; i++) {
        int4 basis_vec = reinterpret_cast<const int4*>(basis_row)[i];
        float4 input_vec = reinterpret_cast<const float4*>(input_row)[i];
        
        // INT8을 float으로 변환하고 내적 계산
        sum += (float)basis_vec.x * input_vec.x;
        sum += (float)basis_vec.y * input_vec.y;
        sum += (float)basis_vec.z * input_vec.z;
        sum += (float)basis_vec.w * input_vec.w;
    }
    
    // 나머지 처리
    for (int i = vec_size * 4; i < in_features; i++) {
        sum += (float)basis_row[i] * input_row[i];
    }
    
    // 최종 스케일링 적용
    output[tid] = sum * scale_factor * basis_scale;
}

// C++ 인터페이스 함수들
extern "C" {
    void launch_gemm_hyper_bit_kernel(const float* x, const uint32_t* codes,
                                     const float* basis_table, float delta,
                                     float* output, int batch_size, int input_dim,
                                     int output_dim, int basis_size) {
        int total_elements = batch_size * output_dim;
        int block_size = 256; // 더 큰 블록 크기 사용
        int grid_size = (total_elements + block_size - 1) / block_size;
        
        gemm_hyper_bit_kernel<<<grid_size, block_size>>>(
            x, codes, basis_table, delta, output, 
            batch_size, input_dim, output_dim, basis_size
        );
        cudaDeviceSynchronize();
    }
    
    void launch_residual_gemm_kernel(const float* x, const float* residual_weights,
                                    float* output, int batch_size, int input_dim,
                                    int output_dim) {
        dim3 block(16, 16);
        dim3 grid(batch_size, (output_dim + block.y - 1) / block.y);
        residual_gemm_kernel<<<grid, block>>>(
            x, residual_weights, output, batch_size, input_dim, output_dim
        );
        cudaDeviceSynchronize();
    }
} 

// 캐시된 메모리를 사용하는 C 인터페이스
extern "C" void launch_gemm_hyper_bit_cached_kernel(
    const float* x,
    const uint32_t* codes_gpu,      // 이미 GPU에 있음
    const float* basis_table_gpu,   // 이미 GPU에 있음
    const float* residual_weights_gpu, // 이미 GPU에 있음 (null일 수 있음)
    float delta,
    float* output,
    int batch_size,
    int input_dim,
    int output_dim,
    int basis_size
) {
    int total_threads = batch_size * output_dim;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    gemm_hyper_bit_cached_kernel<<<blocks, threads_per_block>>>(
        x, codes_gpu, basis_table_gpu, residual_weights_gpu,
        delta, output, batch_size, input_dim, output_dim, basis_size
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA sync error: %s\n", cudaGetErrorString(err));
    }
}

// 최적화된 C 인터페이스 (스트림 지원)
extern "C" void launch_gemm_hyper_bit_optimized_kernel(
    const float* x,
    const uint32_t* codes_gpu,
    const float* basis_table_gpu,
    const uint32_t* residual_codes_gpu,
    float delta,
    float residual_delta,
    float* output,
    int batch_size,
    int input_dim,
    int output_dim,
    int basis_size,
    void* stream
) {
    // 최적의 블록 크기 계산
    int threads_per_block = 256;
    int shared_mem_size = (threads_per_block + input_dim) * sizeof(float);
    
    // 그리드 크기 계산
    int total_threads = batch_size * output_dim;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    // 커널 실행 (스트림 사용)
    gemm_hyper_bit_optimized_kernel<<<blocks, threads_per_block, shared_mem_size, (cudaStream_t)stream>>>(
        x, codes_gpu, basis_table_gpu, residual_codes_gpu,
        delta, residual_delta, output, batch_size, input_dim, output_dim, basis_size
    );
}

// C++ 인터페이스 함수
extern "C" void launch_gemm_hyper_bit_direct_kernel(
    const float* x_gpu,
    const uint32_t* codes_gpu,
    const float* basis_table_gpu,
    const uint32_t* residual_codes_gpu,
    float delta,
    float residual_delta,
    float* output_gpu,
    int batch_size,
    int input_dim,
    int output_dim,
    int basis_size,
    void* stream
) {
    // 블록과 그리드 크기 계산
    const int BLOCK_SIZE = 256;
    const int total_threads = batch_size * output_dim;
    const int grid_size = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // 공유 메모리 크기 계산
    const int shared_mem_size = BLOCK_SIZE * input_dim * sizeof(float);
    
    // 커널 실행
    gemm_hyper_bit_direct_kernel<<<grid_size, BLOCK_SIZE, shared_mem_size, (cudaStream_t)stream>>>(
        x_gpu, codes_gpu, basis_table_gpu, residual_codes_gpu,
        delta, residual_delta, output_gpu,
        batch_size, input_dim, output_dim, basis_size
    );
}

extern "C" void launch_gemm_hyper_bit_backward_direct_kernel(
    const float* grad_output_gpu,
    const uint32_t* codes_gpu,
    const float* basis_table_gpu,
    const uint32_t* residual_codes_gpu,
    float delta,
    float residual_delta,
    float* grad_input_gpu,
    int batch_size,
    int input_dim,
    int output_dim,
    int basis_size,
    void* stream
) {
    const int BLOCK_SIZE = 256;
    const int total_threads = batch_size * input_dim;
    const int grid_size = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    gemm_hyper_bit_backward_direct_kernel<<<grid_size, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(
        grad_output_gpu, codes_gpu, basis_table_gpu, residual_codes_gpu,
        delta, residual_delta, grad_input_gpu,
        batch_size, input_dim, output_dim, basis_size
    );
}

extern "C" void launch_gemm_hyper_bit_int8_kernel(
    const float* x,
    const uint32_t* codes_gpu,
    const int8_t* basis_table_int8_gpu,
    const float* basis_scales_gpu,
    const float delta,
    float* output,
    const int batch_size,
    const int input_dim,
    const int output_dim,
    const int basis_size,
    void* stream
) {
    const int total_elements = batch_size * output_dim;
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    // 공유 메모리 크기 계산
    const size_t shared_mem_size = threads_per_block * sizeof(float);
    
    gemm_hyper_bit_int8_kernel<<<num_blocks, threads_per_block, shared_mem_size, (cudaStream_t)stream>>>(
        x, codes_gpu, basis_table_int8_gpu, basis_scales_gpu, delta,
        output, batch_size, input_dim, output_dim, basis_size
    );
}


