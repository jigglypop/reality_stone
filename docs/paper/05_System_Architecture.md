## 5. 시스템 아키텍처: Reality Stone의 설계와 구현

본 장에서는 RBE를 실제로 구현한 `Reality Stone` 프레임워크의 시스템 아키텍처를 상세히 설명한다. 고성능 추론과 효율적인 메모리 사용을 위한 Rust/CUDA 기반의 설계 결정들과, PyTorch와의 완벽한 통합을 위한 바인딩 구조를 다룬다.

### 5.1. 전체 시스템 구조

#### 5.1.1. 계층적 아키텍처

`Reality Stone`은 다음과 같은 4계층 구조로 설계되었다:

```
┌─────────────────────────────────────────────────────────┐
│                   Python API Layer                      │
│  - PyTorch nn.Module 호환 인터페이스                     │
│  - 자동 미분 지원                                       │
│  - 사용자 친화적 API                                    │
├─────────────────────────────────────────────────────────┤
│                   PyO3 Binding Layer                    │
│  - Rust-Python 인터페이스                               │
│  - 텐서 변환 및 메모리 관리                              │
│  - 에러 처리 및 타입 변환                                │
├─────────────────────────────────────────────────────────┤
│                   Rust Core Layer                       │
│  - 비트필드 인코딩/디코딩 로직                           │
│  - 메모리 안전성 보장                                    │
│  - SIMD 최적화 (x86_64, ARM)                           │
├─────────────────────────────────────────────────────────┤
│                   CUDA Kernel Layer                     │
│  - GPU 병렬 처리                                        │
│  - Tensor Core 활용                                     │
│  - 공유 메모리 최적화                                    │
└─────────────────────────────────────────────────────────┘
```

#### 5.1.2. 모듈 구성

```rust
// src/lib.rs - 메인 모듈 구조
pub mod layers {
    pub mod bitfield;    // 비트필드 압축 레이어
    pub mod poincare;    // 푸앵카레 기하학 레이어
    pub mod lorentz;     // 로렌츠 모델 레이어
    pub mod klein;       // 클라인 모델 레이어
    pub mod spline;      // 스플라인 압축 (실험적)
}

pub mod ops {
    pub mod mobius;      // 뫼비우스 연산
    pub mod project;     // 투영 연산
    pub mod curvature;   // 곡률 계산
}

pub mod cuda {
    pub mod kernels;     // CUDA 커널 구현
    pub mod memory;      // GPU 메모리 관리
}
```

### 5.2. 비트필드 인코딩 구현

#### 5.2.1. 기본 비트필드 구조

RBE의 핵심은 32비트 정수로 가중치를 표현하는 것이다:

```rust
pub struct BitfieldWeight {
    data: u32,  // 32비트 압축 표현
}

// 비트 레이아웃
// [31:24] - 위상 (8비트)
// [23:20] - 크기 지수 (4비트)  
// [19:16] - 크기 가수 (4비트)
// [15:8]  - 주파수 (8비트)
// [7:0]   - 기저 인덱스 (8비트)
```

#### 5.2.2. 고급 위상 인코딩 기법

##### 고정소수점 표현 (Fixed-point representation)

8비트 인코딩:
```c
// 0 ~ 2π를 0 ~ 255로 매핑
uint8_t phase_8bit = (uint8_t)(phase_radians * 255.0 / (2.0 * PI));

// 복원
float phase_radians = (float)phase_8bit * 2.0 * PI / 255.0;
```

16비트 고정밀도:
```c
// 0 ~ 2π를 0 ~ 65535로 매핑
uint16_t phase_16bit = (uint16_t)(phase_radians * 65535.0 / (2.0 * PI));

// 복원
float phase_radians = (float)phase_16bit * 2.0 * PI / 65535.0;
```

##### 2의 보수 표현 (Two's complement)

```c
// -π ~ π를 -128 ~ 127로 매핑 (8비트)
int8_t phase_8bit = (int8_t)(phase_radians * 127.0 / PI);

// -π ~ π를 -32768 ~ 32767로 매핑 (16비트)
int16_t phase_16bit = (int16_t)(phase_radians * 32767.0 / PI);
```

##### 복소수 기반 표현 (16비트)

```c
// 16비트를 실부/허부로 분할
typedef struct {
    int8_t real;  // 8비트 실부
    int8_t imag;  // 8비트 허부
} ComplexInt16;

// 위상과 크기를 복소수로 인코딩
ComplexInt16 encode_phase_magnitude(float phase, float magnitude) {
    ComplexInt16 result;
    result.real = (int8_t)(magnitude * 127.0f * cosf(phase));
    result.imag = (int8_t)(magnitude * 127.0f * sinf(phase));
    return result;
}

// 디코딩
void decode_phase_magnitude(ComplexInt16 c, float* phase, float* magnitude) {
    float r = (float)c.real / 127.0f;
    float i = (float)c.imag / 127.0f;
    *magnitude = sqrtf(r*r + i*i);
    *phase = atan2f(i, r);
}
```

##### CORDIC 친화적 표현

```c
// CORDIC 알고리즘에 최적화된 각도 표현
const float CORDIC_ANGLES[16] = {
    45.0f, 26.565f, 14.036f, 7.125f, 3.576f, 1.790f, 0.895f, 0.448f,
    0.224f, 0.112f, 0.056f, 0.028f, 0.014f, 0.007f, 0.003f, 0.002f
};

uint16_t encode_cordic_angle(float angle_degrees) {
    uint16_t result = 0;
    float remaining = angle_degrees;
    
    for (int i = 0; i < 16; i++) {
        if (remaining >= CORDIC_ANGLES[i]) {
            result |= (1 << (15 - i));
            remaining -= CORDIC_ANGLES[i];
        }
    }
    return result;
}
```

##### 극좌표 비트필드 (16비트)

```c
// 12비트 위상 + 4비트 크기
#define PHASE_BITS 12
#define MAG_BITS 4
#define PHASE_MASK 0xFFF0
#define MAG_MASK 0x000F

uint16_t encode_polar(float phase, float magnitude) {
    // 위상: 0-2π를 0-4095로 매핑 (12비트)
    uint16_t phase_int = (uint16_t)((phase / (2.0f * M_PI)) * 4095.0f);
    // 크기: 0-1을 0-15로 매핑 (4비트)
    uint16_t mag_int = (uint16_t)(magnitude * 15.0f);
    
    return (phase_int << 4) | mag_int;
}
```

#### 5.2.3. CUDA 비트필드 연산 최적화

```cuda
__global__ void bitfield_decode_optimized(
    const uint32_t* __restrict__ bitfields,
    float* __restrict__ weights,
    const float* __restrict__ basis_functions,
    int num_weights,
    int basis_dim
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = threadIdx.x & 31;
    
    if (tid >= num_weights) return;
    
    // 워프 레벨 로드
    uint32_t bitfield = __ldg(&bitfields[tid]);
    
    // 비트 추출 (비트 시프트 최적화)
    uint8_t phase_bits = (bitfield >> 24) & 0xFF;
    uint8_t mag_exp = (bitfield >> 20) & 0x0F;
    uint8_t mag_mant = (bitfield >> 16) & 0x0F;
    uint8_t freq = (bitfield >> 8) & 0xFF;
    uint8_t basis_idx = bitfield & 0xFF;
    
    // 위상 디코딩 (룩업 테이블 사용)
    float phase = __ldg(&phase_lut[phase_bits]);
    
    // 크기 디코딩 (비트 연산)
    float magnitude = __ldg(&mag_lut[mag_exp]) * (1.0f + mag_mant * 0.0625f);
    
    // 주파수 스케일링
    float freq_scale = __ldg(&freq_lut[freq]);
    
    // 기저 함수 적용 (텍스처 메모리 활용)
    float basis_val = tex1Dfetch(basis_texture, basis_idx * basis_dim + threadIdx.y);
    
    // 최종 가중치 계산
    weights[tid] = magnitude * __cosf(phase + freq_scale * basis_val);
}
```

#### 5.2.4. SIMD 비트필드 연산

AVX2 구현:
```c
void decode_bitfields_avx2(
    const uint32_t* bitfields,
    float* weights,
    const float* basis,
    int count
) {
    const __m256i phase_shift = _mm256_set1_epi32(24);
    const __m256i phase_mask = _mm256_set1_epi32(0xFF);
    const __m256 phase_scale = _mm256_set1_ps(2.0f * M_PI / 255.0f);
    
    for (int i = 0; i < count; i += 8) {
        // 8개 비트필드 로드
        __m256i bf = _mm256_loadu_si256((__m256i*)&bitfields[i]);
        
        // 위상 추출
        __m256i phase_bits = _mm256_and_si256(
            _mm256_srl_epi32(bf, phase_shift),
            phase_mask
        );
        
        // 정수를 float로 변환
        __m256 phase_float = _mm256_cvtepi32_ps(phase_bits);
        
        // 스케일링
        __m256 phase_rad = _mm256_mul_ps(phase_float, phase_scale);
        
        // 코사인 계산 (AVX2 근사)
        __m256 cos_phase = _mm256_cos_ps(phase_rad);
        
        // 저장
        _mm256_storeu_ps(&weights[i], cos_phase);
    }
}
```

ARM NEON 구현:
```c
void decode_bitfields_neon(
    const uint32_t* bitfields,
    float* weights,
    int count
) {
    const uint32x4_t phase_mask = vdupq_n_u32(0xFF000000);
    const float32x4_t phase_scale = vdupq_n_f32(2.0f * M_PI / 255.0f);
    
    for (int i = 0; i < count; i += 4) {
        // 4개 비트필드 로드
        uint32x4_t bf = vld1q_u32(&bitfields[i]);
        
        // 위상 추출
        uint32x4_t phase_bits = vshrq_n_u32(
            vandq_u32(bf, phase_mask), 24
        );
        
        // float 변환 및 스케일링
        float32x4_t phase_rad = vmulq_f32(
            vcvtq_f32_u32(phase_bits),
            phase_scale
        );
        
        // 결과 저장
        vst1q_f32(&weights[i], phase_rad);
    }
}
```

### 5.3. CUDA 커널 구현

#### 5.3.1. 압축 도메인 추론 커널

압축된 상태에서 직접 추론하는 것이 RBE의 핵심이다. 다음은 최적화된 CUDA 커널이다:

```cuda
// src/layers/cuda/bitfield_kernel.cu
#include <cuda_fp16.h>
#include <mma.h>

// 공유 메모리 크기 정의
#define SHARED_MEM_SIZE 49152  // 48KB
#define WARP_SIZE 32
#define MAX_BASIS 256

using namespace nvcuda;

__global__ void bitfield_forward_kernel(
    const uint32_t* __restrict__ codes,        // [out_features]
    const float* __restrict__ input,           // [batch_size, in_features]
    const float* __restrict__ basis_table,     // [256, in_features]
    const int8_t* __restrict__ residual,       // [out_features, in_features]
    float* __restrict__ output,                // [batch_size, out_features]
    const int batch_size,
    const int in_features,
    const int out_features
) {
    // 블록당 공유 메모리 할당
    extern __shared__ float shared_mem[];
    float* s_basis_dots = shared_mem;  // [256] 기저 내적 결과
    
    // 스레드 인덱스
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int batch_idx = blockIdx.y;
    
    // 각 블록이 하나의 출력 타일 처리
    const int out_tile_start = bid * blockDim.x;
    const int out_tile_end = min(out_tile_start + blockDim.x, out_features);
    
    if (batch_idx >= batch_size) return;
    
    // Step 1: 입력과 모든 기저의 내적 계산 (협력적)
    const float* input_row = &input[batch_idx * in_features];
    
    // 각 워프가 다른 기저 처리
    for (int basis_idx = tid; basis_idx < MAX_BASIS; basis_idx += blockDim.x) {
        float dot = 0.0f;
        const float* basis_row = &basis_table[basis_idx * in_features];
        
        // 벡터화된 내적 (float4 사용)
        const int vec_size = in_features / 4;
        #pragma unroll 4
        for (int i = 0; i < vec_size; i++) {
            float4 inp = reinterpret_cast<const float4*>(input_row)[i];
            float4 bas = reinterpret_cast<const float4*>(basis_row)[i];
            dot += inp.x * bas.x + inp.y * bas.y + inp.z * bas.z + inp.w * bas.w;
        }
        
        // 나머지 원소 처리
        for (int i = vec_size * 4; i < in_features; i++) {
            dot += input_row[i] * basis_row[i];
        }
        
        s_basis_dots[basis_idx] = dot;
    }
    
    __syncthreads();
    
    // Step 2: 각 스레드가 하나의 출력 계산
    const int out_idx = out_tile_start + tid;
    if (out_idx < out_features) {
        // 비트필드 디코딩
        const uint32_t code = codes[out_idx];
        const uint8_t idx = (code >> 10) & 0xFF;
        const float scale = decode_scale(code);  // 인라인 함수
        
        // 청사진 기여분
        float blueprint_result = s_basis_dots[idx] * scale;
        
        // 잔차 기여분 (INT8 DP4A 활용)
        float residual_result = 0.0f;
        const int8_t* res_row = &residual[out_idx * in_features];
        
        // INT8 벡터 내적
        const int vec8_size = in_features / 8;
        #pragma unroll 2
        for (int i = 0; i < vec8_size; i++) {
            int32_t dp4a_result = 0;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                dp4a_result += res_row[i*8 + j] * 
                              __float2int_rn(input_row[i*8 + j] * 127.0f);
            }
            residual_result += dp4a_result / (127.0f * 127.0f);
        }
        
        // 최종 결과
        output[batch_idx * out_features + out_idx] = 
            blueprint_result + residual_result;
    }
}
```

#### 5.3.2. Tensor Core 활용 (Ampere 이상)

```cuda
// INT8 Tensor Core를 활용한 고속 추론
__global__ void bitfield_tensorcore_kernel(
    const uint32_t* codes,
    const int8_t* input_int8,      // 양자화된 입력
    const int8_t* basis_int8,      // 양자화된 기저
    const int8_t* residual,
    int8_t* output_int8,
    const float input_scale,
    const float output_scale,
    const int M, const int N, const int K
) {
    // Tensor Core 프래그먼트 선언
    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> c_frag;
    
    // 워프 좌표
    const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int warpN = blockIdx.y;
    
    // 누산기 초기화
    wmma::fill_fragment(c_frag, 0);
    
    // 타일 단위 행렬 곱셈
    for (int k = 0; k < K; k += 16) {
        // 입력 타일 로드
        const int8_t* a_ptr = &input_int8[warpM * 16 * K + k];
        wmma::load_matrix_sync(a_frag, a_ptr, K);
        
        // 기저 선택 및 로드
        const uint32_t code = codes[warpN * 16];
        const uint8_t basis_idx = (code >> 10) & 0xFF;
        const int8_t* b_ptr = &basis_int8[basis_idx * K + k];
        wmma::load_matrix_sync(b_frag, b_ptr, K);
        
        // Tensor Core 연산
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // 스케일 적용 및 저장
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] = __float2int_rn(
            c_frag.x[i] * input_scale * output_scale
        );
    }
    
    // 결과 저장
    int8_t* c_ptr = &output_int8[warpM * 16 * N + warpN * 16];
    wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
}
```

### 5.4. PyTorch 통합

#### 5.4.1. Python 바인딩

```python
# python/reality_stone/layers/bitfield.py
import torch
import torch.nn as nn
from .. import _rust  # Rust 바인딩

class BitfieldLinear(nn.Module):
    """RBE 압축을 적용한 선형 레이어"""
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        basis_size: int = 256,
        residual_bits: int = 8,
        trainable: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.basis_size = basis_size
        
        # Rust 구현체 생성
        self.rust_layer = _rust.BitfieldLinear(
            in_features, out_features, basis_size, residual_bits
        )
        
        # 학습 가능한 파라미터
        if trainable:
            # 기저 테이블 (공유 가능)
            self.register_parameter(
                'basis_table',
                nn.Parameter(torch.randn(basis_size, in_features))
            )
            # 연속적인 가중치 (QAT용)
            self.register_parameter(
                'continuous_weight',
                nn.Parameter(torch.randn(out_features, in_features))
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and hasattr(self, 'continuous_weight'):
            # QAT: 양자화 시뮬레이션
            codes, residual = self.rust_layer.encode_weights(
                self.continuous_weight, self.basis_table
            )
            return BitfieldFunction.apply(
                x, codes, residual, self.basis_table, self.rust_layer
            )
        else:
            # 추론: 압축된 상태에서 직접 계산
            return self.rust_layer.forward(x)
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs):
        """기존 Linear 레이어를 BitfieldLinear로 변환"""
        layer = cls(
            linear.in_features, 
            linear.out_features, 
            **kwargs
        )
        
        # 가중치 압축
        with torch.no_grad():
            layer.rust_layer.compress_from_weights(
                linear.weight.data,
                linear.bias.data if linear.bias is not None else None
            )
        
        return layer
```

#### 5.4.2. 자동 미분 지원

```python
class BitfieldFunction(torch.autograd.Function):
    """RBE 연산의 자동 미분 지원"""
    
    @staticmethod
    def forward(ctx, input, codes, residual, basis_table, rust_layer):
        # 순전파 계산
        output = rust_layer.forward_with_params(
            input, codes, residual, basis_table
        )
        
        # 역전파를 위한 정보 저장
        ctx.save_for_backward(input, codes, residual, basis_table)
        ctx.rust_layer = rust_layer
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, codes, residual, basis_table = ctx.saved_tensors
        rust_layer = ctx.rust_layer
        
        # Rust에서 그래디언트 계산
        grad_input, grad_codes, grad_residual, grad_basis = \
            rust_layer.backward(
                grad_output, input, codes, residual, basis_table
            )
        
        # STE 적용 (codes는 이산적이므로)
        if grad_codes is not None:
            # continuous_weight로 그래디언트 전파
            grad_weight = rust_layer.codes_to_weight_gradient(
                grad_codes, basis_table
            )
        else:
            grad_weight = None
        
        return grad_input, None, grad_residual, grad_basis, None
```

### 5.5. 최적화 기법

#### 5.5.1. 메모리 풀링

```rust
// src/cuda/memory.rs
use std::sync::Mutex;
use std::collections::HashMap;

pub struct CudaMemoryPool {
    pools: Mutex<HashMap<usize, Vec<CudaBuffer>>>,
    total_allocated: AtomicUsize,
    high_water_mark: AtomicUsize,
}

impl CudaMemoryPool {
    pub fn allocate(&self, size: usize) -> Result<CudaBuffer, CudaError> {
        // 크기를 2의 제곱으로 반올림
        let aligned_size = size.next_power_of_two();
        
        let mut pools = self.pools.lock().unwrap();
        
        // 재사용 가능한 버퍼 확인
        if let Some(pool) = pools.get_mut(&aligned_size) {
            if let Some(buffer) = pool.pop() {
                return Ok(buffer);
            }
        }
        
        // 새 버퍼 할당
        let buffer = CudaBuffer::new(aligned_size)?;
        self.total_allocated.fetch_add(aligned_size, Ordering::Relaxed);
        
        // 최고 수위 갱신
        self.high_water_mark.fetch_max(
            self.total_allocated.load(Ordering::Relaxed),
            Ordering::Relaxed
        );
        
        Ok(buffer)
    }
    
    pub fn deallocate(&self, buffer: CudaBuffer) {
        let size = buffer.size();
        let mut pools = self.pools.lock().unwrap();
        pools.entry(size).or_insert_with(Vec::new).push(buffer);
    }
}
```

#### 5.5.2. 커널 퓨전

```cuda
// 여러 연산을 하나의 커널로 통합
__global__ void fused_bitfield_layernorm_gelu_kernel(
    const uint32_t* codes,
    const float* input,
    const float* basis_table,
    const int8_t* residual,
    const float* ln_gamma,
    const float* ln_beta,
    float* output,
    const int batch_size,
    const int hidden_size
) {
    // 공유 메모리
    extern __shared__ float shared[];
    float* s_mean = shared;
    float* s_var = &shared[1];
    float* s_hidden = &shared[2];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    // Step 1: BitfieldLinear 계산
    // ... (이전 커널과 동일)
    
    // Step 2: LayerNorm (Welford's algorithm)
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = s_hidden[i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // 워프 리덕션
    local_sum = warpReduceSum(local_sum);
    local_sum_sq = warpReduceSum(local_sum_sq);
    
    if (tid % 32 == 0) {
        atomicAdd(s_mean, local_sum);
        atomicAdd(s_var, local_sum_sq);
    }
    
    __syncthreads();
    
    // 평균과 분산 계산
    if (tid == 0) {
        *s_mean /= hidden_size;
        *s_var = *s_var / hidden_size - (*s_mean) * (*s_mean);
        *s_var = rsqrtf(*s_var + 1e-5f);
    }
    
    __syncthreads();
    
    // Step 3: GELU 활성화와 함께 정규화 적용
    const float mean = *s_mean;
    const float inv_std = *s_var;
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = (s_hidden[i] - mean) * inv_std;
        float scaled = ln_gamma[i] * normalized + ln_beta[i];
        
        // GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        const float c0 = 0.7978845608f;  // sqrt(2/π)
        const float c1 = 0.044715f;
        float x3 = scaled * scaled * scaled;
        float tanh_arg = c0 * (scaled + c1 * x3);
        float gelu = 0.5f * scaled * (1.0f + tanhf(tanh_arg));
        
        output[bid * hidden_size + i] = gelu;
    }
}
```

### 5.6. 프로파일링과 디버깅

#### 5.6.1. 성능 프로파일링

```rust
// src/profiling.rs
use std::time::Instant;

pub struct Profiler {
    events: Vec<ProfileEvent>,
    cuda_events: Vec<CudaEvent>,
}

#[derive(Debug)]
pub struct ProfileEvent {
    name: String,
    start: Instant,
    duration: Duration,
    memory_used: usize,
}

impl Profiler {
    pub fn profile_kernel<F>(&mut self, name: &str, f: F) 
    where 
        F: FnOnce() 
    {
        // CUDA 이벤트 생성
        let start_event = CudaEvent::new();
        let end_event = CudaEvent::new();
        
        // 메모리 사용량 기록
        let mem_before = cuda_mem_info();
        
        // 커널 실행
        start_event.record();
        let start = Instant::now();
        
        f();
        
        end_event.record();
        end_event.synchronize();
        
        let duration = start.elapsed();
        let mem_after = cuda_mem_info();
        
        // 프로파일 정보 저장
        self.events.push(ProfileEvent {
            name: name.to_string(),
            start,
            duration,
            memory_used: mem_after.used - mem_before.used,
        });
        
        // CUDA 타이밍
        let cuda_time = start_event.elapsed_time(&end_event);
        println!("[{}] CPU: {:?}, GPU: {:.3}ms, Memory: {} MB", 
                 name, duration, cuda_time, 
                 (mem_after.used - mem_before.used) / 1024 / 1024);
    }
}
```

### 5.7. 배포와 최적화

#### 5.7.1. 플랫폼별 최적화

```rust
// src/layers/bitfield/simd.rs
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

pub fn optimized_decode(codes: &[u32], scales: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        decode_avx2(codes, scales);
    }
    
    #[cfg(target_arch = "aarch64")]
    unsafe {
        decode_neon(codes, scales);
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        decode_scalar(codes, scales);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_avx2(codes: &[u32], scales: &mut [f32]) {
    const SHUFFLE_MASK: __m256i = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    
    for chunk in codes.chunks_exact(8) {
        // 8개 코드를 한번에 로드
        let codes_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
        
        // 비트 추출 (SIMD)
        let amp = _mm256_and_si256(codes_vec, _mm256_set1_epi32(0xFF));
        let amp_fine = _mm256_srli_epi32(codes_vec, 22);
        
        // 정수를 부동소수점으로 변환
        let amp_f = _mm256_cvtepi32_ps(amp);
        let amp_fine_f = _mm256_cvtepi32_ps(amp_fine);
        
        // 스케일 계산
        let scale = _mm256_div_ps(
            _mm256_add_ps(
                amp_f,
                _mm256_div_ps(amp_fine_f, _mm256_set1_ps(1024.0))
            ),
            _mm256_set1_ps(128.0)
        );
        
        // tanh 근사 (빠른 버전)
        let tanh_approx = fast_tanh_avx2(scale);
        
        // 결과 저장
        _mm256_storeu_ps(scales.as_mut_ptr(), tanh_approx);
        scales = &mut scales[8..];
    }
}
```

### 5.8. 성능 벤치마크

#### 5.8.1. 마이크로벤치마크

```rust
#[cfg(test)]
mod benches {
    use criterion::{criterion_group, criterion_main, Criterion};
    
    fn bench_bitfield_forward(c: &mut Criterion) {
        let mut group = c.benchmark_group("bitfield_forward");
        
        for &size in &[512, 1024, 2048, 4096] {
            group.bench_function(
                format!("{}x{}", size, size),
                |b| {
                    let layer = BitfieldLinear::new(size, size);
                    let input = torch::randn(&[32, size], options);
                    
                    b.iter(|| {
                        let _ = layer.forward(&input);
                        torch::cuda::synchronize();
                    });
                }
            );
        }
        
        group.finish();
    }
}
```

### 5.9. 결론

`Reality Stone`의 시스템 아키텍처는 다음의 핵심 설계 원칙을 따른다:

1. **계층적 추상화**: Python API부터 CUDA 커널까지 명확한 책임 분리
2. **메모리 효율성**: 비트 단위 패킹과 메모리 풀링으로 최소 메모리 사용
3. **연산 최적화**: Tensor Core, SIMD, 커널 퓨전 등 모든 최적화 기법 활용
4. **확장성**: 새로운 기하학 모델과 압축 기법을 쉽게 추가 가능
5. **실용성**: PyTorch와의 완벽한 통합으로 기존 워크플로우에 즉시 적용 가능

이러한 설계를 통해 RBE의 이론적 장점을 실제 시스템에서 완전히 실현할 수 있었다. 