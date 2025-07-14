# Reality Stone 리팩토링 계획서

## 📋 목차
1. [개요](#개요)
2. [현재 문제점](#현재-문제점)
3. [리팩토링 목표](#리팩토링-목표)
4. [아키텍처 개선안](#아키텍처-개선안)
5. [구현 계획](#구현-계획)
6. [마이그레이션 전략](#마이그레이션-전략)
7. [성공 지표](#성공-지표)

## 개요

Reality Stone 프로젝트의 코드 구조를 개선하여 유지보수성, 성능, 일관성을 향상시키기 위한 종합 리팩토링 계획입니다.

## 현재 문제점

### 1. 바인딩 분산
```
src/bindings/
├── poincare.rs (292줄)
├── lorentz.rs (239줄)
├── klein.rs (227줄)
├── mobius.rs (159줄)
├── bitfield.rs (400+줄)
└── spline.rs (10줄)
```
- 6개 파일에 분산된 바인딩 코드
- 각 파일마다 유사한 패턴 반복
- 새 레이어 추가 시 여러 파일 수정 필요

### 2. 인터페이스 불일치
```python
# 함수형 API
result = poincare_add(x, y, c=1.0)

# 클래스형 API
layer = BitfieldLinear(...)
result = layer(x)

# 혼합형 API
layer = PoincareBallLayer.apply(u, v, c, t)
```

### 3. 코드 중복
- 동일한 배열 변환 코드가 수십 번 반복
- 에러 처리 로직 중복
- CUDA 포인터 관리 코드 중복

### 4. 불완전한 구현
- SplineLinear: Python 전용, Rust/CUDA 미구현
- 일부 레이어만 다차원 텐서 지원
- GPU 메모리 관리 비효율

## 리팩토링 목표

1. **통합 바인딩 시스템**: 단일 인터페이스로 모든 레이어 관리
2. **일관된 API**: 모든 레이어가 동일한 패턴 사용
3. **코드 재사용**: 공통 로직 추출 및 중복 제거
4. **완전한 구현**: 모든 레이어의 Rust/CUDA 구현 완성
5. **성능 최적화**: 메모리 효율성 및 병렬처리 개선

## 아키텍처 개선안

### 1. 통합 레이어 시스템

```rust
// src/core/layer.rs (새로 생성)
pub trait Layer: Send + Sync {
    type Config;
    
    fn new(config: Self::Config) -> Self;
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn backward(&self, grad_output: &Tensor) -> Result<Tensor>;
    
    // 선택적 메서드
    fn to_cuda(&mut self) -> Result<()> { Ok(()) }
    fn parameters(&self) -> Vec<&Tensor> { vec![] }
}

// src/core/registry.rs (새로 생성)
pub struct LayerRegistry {
    layers: HashMap<String, Box<dyn Layer>>,
}

impl LayerRegistry {
    pub fn register<L: Layer + 'static>(&mut self, name: &str, layer: L) {
        self.layers.insert(name.to_string(), Box::new(layer));
    }
}
```

### 2. 매크로 기반 바인딩 자동화

```rust
// src/bindings/macros.rs (새로 생성)
#[macro_export]
macro_rules! create_py_layer {
    ($name:ident, $rust_type:ty) => {
        #[pyclass(name = stringify!($name))]
        pub struct $name {
            inner: $rust_type,
        }
        
        #[pymethods]
        impl $name {
            #[new]
            fn new(config: PyObject) -> PyResult<Self> {
                let config = parse_config(config)?;
                Ok(Self {
                    inner: <$rust_type>::new(config)
                })
            }
            
            fn forward(&self, input: PyObject) -> PyResult<PyObject> {
                unified_forward(&self.inner, input)
            }
            
            fn backward(&self, grad: PyObject) -> PyResult<PyObject> {
                unified_backward(&self.inner, grad)
            }
        }
    };
}
```

### 3. 새로운 디렉토리 구조

```
src/
├── core/                    # 핵심 공통 모듈
│   ├── mod.rs
│   ├── layer.rs            # Layer 트레이트
│   ├── tensor.rs           # 텐서 추상화
│   ├── registry.rs         # 레이어 레지스트리
│   └── error.rs            # 통합 에러 타입
├── layers/                  # 레이어 구현
│   ├── mod.rs
│   ├── hyperbolic/         # 하이퍼볼릭 레이어
│   │   ├── poincare.rs
│   │   ├── lorentz.rs
│   │   └── klein.rs
│   ├── compressed/         # 압축 레이어
│   │   ├── bitfield.rs
│   │   └── spline.rs
│   └── cuda/               # CUDA 커널
├── ops/                     # 공통 연산
│   ├── mod.rs
│   ├── mobius.rs
│   ├── batch.rs
│   └── memory.rs
└── bindings/               # Python 바인딩
    ├── mod.rs
    ├── unified.rs          # 통합 바인딩
    └── macros.rs           # 바인딩 매크로

python/reality_stone/
├── __init__.py
├── core/                   # 핵심 기능
│   ├── __init__.py
│   ├── base.py            # 기본 클래스
│   └── registry.py        # 레이어 레지스트리
└── layers/                 # 레이어별 래퍼
    ├── __init__.py
    └── {layer}.py          # 각 레이어 (base 상속)
```

## 구현 계획

### Phase 1: 기반 구조 구축 (1주)

#### 1.1 핵심 모듈 생성
```rust
// src/core/layer.rs
pub trait Layer {
    // ... 트레이트 정의
}

// src/core/tensor.rs
pub enum Tensor {
    Cpu(Array<f32, IxDyn>),
    #[cfg(feature = "cuda")]
    Gpu(CudaTensor),
}

// src/core/error.rs
#[derive(Error, Debug)]
pub enum LayerError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    // ...
}
```

#### 1.2 레지스트리 시스템
```rust
// src/core/registry.rs
impl LayerRegistry {
    pub fn new() -> Self {
        let mut registry = Self::default();
        
        // 자동 등록
        registry.register("poincare", PoincareLayer::default());
        registry.register("lorentz", LorentzLayer::default());
        // ...
        
        registry
    }
}
```

### Phase 2: 바인딩 통합 (1주)

#### 2.1 통합 바인딩 구현
```rust
// src/bindings/unified.rs
#[pymodule]
fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // 버전 정보
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    // 레이어 자동 등록
    let registry = LayerRegistry::new();
    for (name, layer) in registry.layers() {
        m.add_class(create_py_wrapper(name, layer)?)?;
    }
    
    Ok(())
}
```

#### 2.2 Python 기본 클래스
```python
# python/reality_stone/core/base.py
class BaseLayer(nn.Module):
    """모든 레이어의 기본 클래스"""
    
    def __init__(self, layer_type: str, **config):
        super().__init__()
        self._rust_layer = _rust.create_layer(layer_type, config)
    
    def forward(self, *args, **kwargs):
        return LayerFunction.apply(args, kwargs, self._rust_layer)
```

### Phase 3: SplineLinear 완전 구현 (1주)

#### 3.1 Rust 구현
```rust
// src/layers/compressed/spline.rs
pub struct SplineLayer {
    control_points: Array2<f32>,
    k: usize,
    residual: Option<Array2<f32>>,
    #[cfg(feature = "cuda")]
    gpu_state: Option<SplineGpuState>,
}

impl SplineLayer {
    pub fn interpolate_weights(&self) -> Array2<f32> {
        // Catmull-Rom 스플라인 보간
        let m = self.out_features;
        let mut weights = Array2::zeros((m, self.in_features));
        
        for i in 0..m {
            let t = i as f32 / (m - 1) as f32;
            let weight_row = self.catmull_rom_interpolate(t);
            weights.row_mut(i).assign(&weight_row);
        }
        
        weights
    }
}
```

#### 3.2 CUDA 커널
```cuda
// src/layers/cuda/spline_kernel.cu
__global__ void spline_interpolation_kernel(
    const float* control_points,
    float* weights,
    int k, int m, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m) return;
    
    float t = float(tid) / float(m - 1);
    int j = int(t * k);
    j = max(1, min(j, k - 2));
    
    float t_local = t * k - j;
    
    // Catmull-Rom 계수
    float c0 = -0.5f * t_local * t_local * t_local + t_local * t_local - 0.5f * t_local;
    float c1 = 1.5f * t_local * t_local * t_local - 2.5f * t_local * t_local + 1.0f;
    float c2 = -1.5f * t_local * t_local * t_local + 2.0f * t_local * t_local + 0.5f * t_local;
    float c3 = 0.5f * t_local * t_local * t_local - 0.5f * t_local * t_local;
    
    // 보간
    for (int i = 0; i < n; i++) {
        weights[tid * n + i] = 
            c0 * control_points[(j-1) * n + i] +
            c1 * control_points[j * n + i] +
            c2 * control_points[(j+1) * n + i] +
            c3 * control_points[(j+2) * n + i];
    }
}
```

### Phase 4: 성능 최적화 (2주)

#### 4.1 메모리 풀
```rust
// src/core/memory.rs
pub struct MemoryPool {
    cpu_pool: HashMap<usize, Vec<Box<[f32]>>>,
    #[cfg(feature = "cuda")]
    gpu_pool: HashMap<usize, Vec<CudaBuffer>>,
}

impl MemoryPool {
    pub fn allocate(&mut self, size: usize) -> PooledBuffer {
        // 재사용 가능한 버퍼 찾기
        if let Some(buffers) = self.cpu_pool.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                return PooledBuffer::new(buffer, self);
            }
        }
        
        // 새로 할당
        PooledBuffer::new(vec![0.0; size].into_boxed_slice(), self)
    }
}
```

#### 4.2 배치 최적화
```rust
// src/ops/batch.rs
pub fn parallel_batch_gemm(
    a: &Array3<f32>,
    b: &Array3<f32>,
) -> Array3<f32> {
    let batch_size = a.shape()[0];
    let m = a.shape()[1];
    let k = a.shape()[2];
    let n = b.shape()[2];
    
    let mut result = Array3::zeros((batch_size, m, n));
    
    // Rayon을 사용한 병렬 처리
    result.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(a.axis_iter(Axis(0)).into_par_iter())
        .zip(b.axis_iter(Axis(0)).into_par_iter())
        .for_each(|((mut c, a), b)| {
            general_mat_mul(1.0, &a, &b, 0.0, &mut c);
        });
    
    result
}
```

### Phase 5: 테스트 및 검증 (1주)

#### 5.1 통합 테스트
```python
# tests/test_unified_layers.py
import pytest
from reality_stone import layers

@pytest.mark.parametrize("layer_type", [
    "poincare", "lorentz", "klein", "bitfield", "spline"
])
def test_layer_consistency(layer_type):
    """모든 레이어의 일관성 테스트"""
    layer = layers.create_layer(layer_type, in_features=10, out_features=5)
    
    # 공통 인터페이스 테스트
    assert hasattr(layer, 'forward')
    assert hasattr(layer, 'backward')
    
    # 기능 테스트
    x = torch.randn(32, 10)
    y = layer(x)
    assert y.shape == (32, 5)
```

## 마이그레이션 전략

### 1. 단계적 마이그레이션
1. 새 구조와 기존 구조 병행 운영
2. 레이어별로 순차적 마이그레이션
3. 테스트 통과 확인 후 기존 코드 제거

### 2. 하위 호환성 유지
```python
# python/reality_stone/layers/__init__.py
# 기존 API 유지
from .unified import create_layer

# 하위 호환성을 위한 별칭
PoincareBallLayer = lambda: create_layer("poincare")
LorentzLayer = lambda: create_layer("lorentz")
# ...
```

### 3. 문서화
- 마이그레이션 가이드 작성
- API 변경사항 문서화
- 예제 코드 업데이트

## 성공 지표

### 정량적 지표
1. **코드 감소**: 바인딩 코드 70% 감소
2. **성능 향상**: 추론 속도 20% 향상
3. **메모리 효율**: GPU 메모리 사용량 30% 감소
4. **테스트 커버리지**: 95% 이상

### 정성적 지표
1. **개발 속도**: 새 레이어 추가 시간 80% 단축
2. **유지보수성**: 버그 수정 시간 50% 단축
3. **일관성**: 모든 레이어가 동일한 API 사용
4. **문서화**: 100% API 문서화

## 일정

| 주차 | 작업 내용 | 산출물 |
|------|-----------|--------|
| 1주차 | 기반 구조 구축 | core 모듈, Layer 트레이트 |
| 2주차 | 바인딩 통합 | 통합 바인딩 시스템 |
| 3주차 | SplineLinear 구현 | 완전한 Rust/CUDA 구현 |
| 4-5주차 | 성능 최적화 | 메모리 풀, 배치 최적화 |
| 6주차 | 테스트 및 문서화 | 통합 테스트, 마이그레이션 가이드 |

## 리스크 관리

### 잠재적 리스크
1. **하위 호환성 문제**: 기존 사용자 코드 영향
2. **성능 저하**: 추상화로 인한 오버헤드
3. **복잡도 증가**: 과도한 일반화

### 대응 방안
1. **점진적 마이그레이션**: 기존 API 유지
2. **성능 모니터링**: 각 단계별 벤치마크
3. **단순성 우선**: 필요한 만큼만 추상화 