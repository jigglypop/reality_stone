# 시스템 아키텍처

Reality Stone의 내부 구조와 설계 원칙을 상세히 설명합니다.

## 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Python API Layer                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │   Poincaré      │  │    Lorentz      │  │     Klein       ││
│  │   Layers        │  │    Layers       │  │    Layers       ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Core Operations (Mobius, etc.)                ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                   PyO3 Bindings                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │            Rust-Python Interface                           ││
│  │         (Type conversion, Error handling)                  ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Rust Core                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │   CPU Kernels   │  │  CUDA Kernels   │  │   Memory Mgmt   ││
│  │                 │  │                 │  │                 ││
│  │ • Poincaré Ops  │  │ • GPU Poincaré  │  │ • Safe Alloc    ││
│  │ • Lorentz Ops   │  │ • GPU Lorentz   │  │ • Buffer Mgmt   ││
│  │ • Klein Ops     │  │ • GPU Klein     │  │ • Error Safety  ││
│  │ • Mobius Ops    │  │ • GPU Mobius    │  │                 ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 🦀 Rust 코어 구조

### 디렉토리 구조

```
src/
├── lib.rs                 # 메인 라이브러리 진입점
├── layers/                # 하이퍼볼릭 레이어 구현
│   ├── mod.rs            # 레이어 모듈 정의
│   ├── poincare.rs       # Poincaré Ball 레이어
│   ├── lorentz.rs        # Lorentz 레이어
│   ├── klein.rs          # Klein 레이어
│   ├── mobius.rs         # Mobius 변환
│   ├── utils.rs          # 공통 유틸리티
│   └── cuda/             # CUDA 구현
│       ├── poincare.cu   # Poincaré CUDA 커널
│       ├── lorentz.cu    # Lorentz CUDA 커널
│       ├── klein.cu      # Klein CUDA 커널
│       └── mobius.cu     # Mobius CUDA 커널
├── bindings/             # Python 바인딩
│   ├── mod.rs            # 바인딩 모듈 정의
│   ├── poincare.rs       # Poincaré 바인딩
│   ├── lorentz.rs        # Lorentz 바인딩
│   ├── klein.rs          # Klein 바인딩
│   └── mobius.rs         # Mobius 바인딩
└── ops/                  # 기본 연산 (미래 확장)
    └── mod.rs
```

### 핵심 설계 원칙

#### 1. 메모리 안전성
```rust
// 모든 메모리 접근은 Rust의 소유권 시스템으로 보호
pub fn poincare_add_cpu(
    x: &Array2<f64>,
    y: &Array2<f64>,
    c: f64,
) -> Result<Array2<f64>, Box<dyn Error>> {
    // 안전한 메모리 접근 보장
    let result = Array2::zeros(x.dim());
    // ... 구현
    Ok(result)
}
```

#### 2. 에러 처리
```rust
// 모든 함수는 Result 타입으로 에러 처리
pub type HyperbolicResult<T> = Result<T, HyperbolicError>;

#[derive(Debug, thiserror::Error)]
pub enum HyperbolicError {
    #[error("Invalid curvature parameter: {0}")]
    InvalidCurvature(f64),
    #[error("Tensor dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("CUDA error: {0}")]
    CudaError(String),
}
```

#### 3. 성능 최적화
```rust
// SIMD 최적화 사용
use simba::simd::*;

// 병렬 처리
use rayon::prelude::*;

pub fn parallel_poincare_add(
    x: &Array2<f64>,
    y: &Array2<f64>,
    c: f64,
) -> Array2<f64> {
    x.axis_iter(Axis(0))
        .into_par_iter()
        .zip(y.axis_iter(Axis(0)))
        .map(|(x_row, y_row)| {
            // 병렬 처리로 각 행 계산
            mobius_add_row(x_row, y_row, c)
        })
        .collect()
}
```

## 🐍 Python 바인딩 구조

### PyO3 바인딩 패턴

```rust
// src/bindings/poincare.rs
use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};

#[pyfunction]
pub fn poincare_ball_layer_cpu(
    u: PyReadonlyArray2<f64>,
    v: PyReadonlyArray2<f64>,
    c: f64,
    t: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    // NumPy 배열을 Rust 배열로 변환
    let u_array = u.as_array();
    let v_array = v.as_array();
    
    // Rust 함수 호출
    let result = crate::layers::poincare::poincare_ball_layer(
        &u_array, &v_array, c, t
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    // 결과를 NumPy 배열로 변환
    Ok(result.into_pyarray(py).to_owned())
}
```

### Python 래퍼 구조

```python
# python/reality_stone/layers/poincare.py
from torch.autograd import Function
from .. import _rust

class PoincareBallLayer(Function):
    @staticmethod
    def forward(ctx, u, v, c, t):
        # 컨텍스트 저장
        ctx.save_for_backward(u, v)
        ctx.c = c
        ctx.t = t
        
        # Rust 함수 호출
        if u.is_cuda:
            return _rust.poincare_ball_layer_cuda(u, v, c, t)
        else:
            return _rust.poincare_ball_layer_cpu(u, v, c, t)
    
    @staticmethod
    def backward(ctx, grad_output):
        u, v = ctx.saved_tensors
        c, t = ctx.c, ctx.t
        
        # 역전파 계산
        grad_u, grad_v = _rust.poincare_ball_layer_backward(
            grad_output, u, v, c, t
        )
        return grad_u, grad_v, None, None
```

## CUDA 구현 구조

### CUDA 커널 구조

```cuda
// src/layers/cuda/poincare.cu
__global__ void poincare_add_kernel(
    const float* x,
    const float* y,
    float* result,
    int batch_size,
    int dim,
    float c
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / dim;
    int feat_idx = idx % dim;
    
    if (batch_idx < batch_size && feat_idx < dim) {
        // Mobius 덧셈 계산
        float x_val = x[idx];
        float y_val = y[idx];
        result[idx] = mobius_add_element(x_val, y_val, c);
    }
}
```

### 메모리 관리

```rust
// CUDA 메모리 관리
pub struct CudaBuffer<T> {
    ptr: *mut T,
    size: usize,
    _phantom: PhantomData<T>,
}

impl<T> CudaBuffer<T> {
    pub fn new(size: usize) -> Result<Self, CudaError> {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            cuda_malloc(&mut ptr, size * std::mem::size_of::<T>())?;
        }
        Ok(CudaBuffer { ptr, size, _phantom: PhantomData })
    }
}

impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            cuda_free(self.ptr);
        }
    }
}
```

## 🔄 데이터 플로우

### 1. Forward Pass
```
Python Input (torch.Tensor)
    ↓
NumPy Array (via .numpy())
    ↓
Rust ndarray
    ↓
CUDA Memory (if GPU)
    ↓
CUDA Kernel Execution
    ↓
Result Back to CPU
    ↓
Rust ndarray
    ↓
NumPy Array
    ↓
PyTorch Tensor
```

### 2. Backward Pass
```
Gradient (torch.Tensor)
    ↓
Saved Context (u, v, c, t)
    ↓
Rust Backward Function
    ↓
Computed Gradients
    ↓
Return (grad_u, grad_v, None, None)
```

## 🧪 테스트 아키텍처

### 테스트 계층

```
tests/
├── unit/                 # 단위 테스트
│   ├── test_poincare.py  # Poincaré 레이어 테스트
│   ├── test_lorentz.py   # Lorentz 레이어 테스트
│   └── test_mobius.py    # Mobius 연산 테스트
├── integration/          # 통합 테스트
│   ├── test_gradients.py # 그래디언트 정확성 테스트
│   └── test_cuda.py      # CUDA 구현 테스트
└── benchmarks/           # 성능 테스트
    ├── memory_test.py    # 메모리 사용량 테스트
    └── speed_test.py     # 속도 벤치마크
```

## 빌드 시스템

### Cargo.toml 구조

```toml
[package]
name = "reality_stone"
version = "0.2.0"
edition = "2021"

[lib]
name = "_rust"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.19", features = ["extension-module"] }
numpy = "0.19"
ndarray = { version = "0.15", features = ["rayon"] }
rayon = "1.7"

[build-dependencies]
cc = "1.0"
glob = "0.3"

[features]
default = []
cuda = []
```

### 빌드 스크립트 (build.rs)

```rust
use std::env;
use cc::Build;

fn main() {
    #[cfg(feature = "cuda")]
    {
        let cuda_path = env::var("CUDA_HOME").expect("CUDA_HOME not set");
        
        // CUDA 파일 컴파일
        let cu_files = glob::glob("src/layers/cuda/*.cu")
            .expect("Failed to read CUDA files")
            .collect::<Result<Vec<_>, _>>()
            .expect("Failed to collect CUDA files");
        
        for file in cu_files {
            Build::new()
                .cuda(true)
                .flag("-arch=sm_70")
                .include(format!("{}/include", cuda_path))
                .file(file)
                .compile("cuda_kernels");
        }
    }
}
```

## 성능 고려사항

### 메모리 레이아웃

```rust
// 연속 메모리 레이아웃 사용
#[repr(C)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

// 캐시 친화적 데이터 구조
pub struct BatchedPoints {
    pub data: Vec<f64>,  // [x1, y1, x2, y2, ...]
    pub batch_size: usize,
    pub dim: usize,
}
```

### SIMD 최적화

```rust
use simba::simd::f64x4;

pub fn vectorized_mobius_add(
    x: &[f64],
    y: &[f64],
    c: f64,
) -> Vec<f64> {
    let chunks = x.chunks_exact(4)
        .zip(y.chunks_exact(4))
        .map(|(x_chunk, y_chunk)| {
            let x_vec = f64x4::from_slice_unaligned(x_chunk);
            let y_vec = f64x4::from_slice_unaligned(y_chunk);
            mobius_add_simd(x_vec, y_vec, c)
        })
        .collect()
}
```

## 🔍 디버깅 및 프로파일링

### 디버그 빌드

```bash
# 디버그 정보 포함 빌드
maturin develop --profile dev

# 메모리 디버깅
valgrind --tool=memcheck python test_script.py

# CUDA 디버깅
cuda-gdb python test_script.py
```

### 프로파일링

```bash
# Rust 프로파일링
cargo build --release --features cuda
perf record --call-graph=dwarf ./target/release/reality_stone

# Python 프로파일링
python -m cProfile -o profile.stats test_script.py
```

이 아키텍처는 성능, 안전성, 유지보수성을 모두 고려한 설계입니다. 각 레이어는 명확한 책임을 가지며, 타입 안전성과 에러 처리를 통해 안정적인 시스템을 구축했습니다. 