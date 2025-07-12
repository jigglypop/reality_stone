# 새 레이어 추가 가이드

이 가이드는 Reality Stone에 새로운 하이퍼볼릭 레이어를 추가하는 방법을 단계별로 설명합니다.

## 개요

새로운 하이퍼볼릭 레이어를 추가하려면 다음 단계를 따라야 합니다:

1. **수학적 기초 정의**: 새로운 하이퍼볼릭 모델의 수학적 정의
2. **Rust 구현**: 핵심 연산을 Rust로 구현
3. **CUDA 커널**: GPU 가속을 위한 CUDA 구현 (선택사항)
4. **Python 바인딩**: PyTorch와 연동하는 Python 래퍼
5. **테스트 작성**: 단위 테스트 및 통합 테스트
6. **문서 작성**: API 문서 및 사용 예제

## 1단계: 수학적 기초 정의

### 새로운 모델 정의

예시로 "Hyperboloid" 모델을 추가한다고 가정하겠습니다.

```rust
// src/layers/hyperboloid.rs
use torch::Tensor;

/// Hyperboloid 모델의 기본 연산들
pub struct HyperboloidOps {
    pub curvature: f64,
}

impl HyperboloidOps {
    pub fn new(curvature: f64) -> Self {
        Self { curvature }
    }
    
    /// 두 점의 거리 계산
    pub fn distance(&self, x: &Tensor, y: &Tensor) -> Tensor {
        // 수학적 정의에 따른 구현
        // d(x,y) = arccosh(-<x,y>_L)
        let inner = self.lorentz_inner(x, y);
        (-inner).arccosh()
    }
    
    /// 민코프스키 내적
    pub fn lorentz_inner(&self, x: &Tensor, y: &Tensor) -> Tensor {
        // <x,y>_L = -x_0*y_0 + sum(x_i*y_i)
        let x_time = x.select(1, 0);
        let y_time = y.select(1, 0);
        let x_space = x.narrow(1, 1, x.size()[1] - 1);
        let y_space = y.narrow(1, 1, y.size()[1] - 1);
        
        -x_time * y_time + (x_space * y_space).sum_dim_intlist(
            &[1], false, torch::Kind::Float
        )
    }
    
    /// 점을 hyperboloid 제약 조건에 맞게 정규화
    pub fn normalize(&self, x: &Tensor) -> Tensor {
        // 제약 조건: <x,x>_L = -1/c
        let norm_sq = self.lorentz_inner(x, x);
        let scale = (-1.0 / self.curvature / norm_sq).sqrt();
        x * scale
    }
}
```

## 2단계: Rust 구현

### 레이어 구조 정의

```rust
// src/layers/hyperboloid.rs (계속)
use crate::ops::HyperboloidOps;

/// Hyperboloid 레이어 구현
pub struct HyperboloidLayer {
    ops: HyperboloidOps,
}

impl HyperboloidLayer {
    pub fn new(curvature: f64) -> Self {
        Self {
            ops: HyperboloidOps::new(curvature),
        }
    }
    
    /// Forward pass 구현
    pub fn forward(
        &self,
        u: &Tensor,
        v: &Tensor,
        t: f64,
    ) -> Tensor {
        // 1. 스칼라 곱셈
        let u_scaled = self.ops.scalar_mul(u, 1.0 - t);
        let v_scaled = self.ops.scalar_mul(v, t);
        
        // 2. 덧셈 연산
        let result = self.ops.add(&u_scaled, &v_scaled);
        
        // 3. 정규화
        self.ops.normalize(&result)
    }
    
    /// Backward pass 구현
    pub fn backward(
        &self,
        grad_output: &Tensor,
        u: &Tensor,
        v: &Tensor,
        t: f64,
    ) -> (Tensor, Tensor) {
        // 그래디언트 계산 구현
        // 이는 수학적 미분에 기반해야 함
        
        // 간단한 예시 (실제로는 더 복잡함)
        let grad_u = grad_output * (1.0 - t);
        let grad_v = grad_output * t;
        
        (grad_u, grad_v)
    }
}
```

### 모듈에 추가

```rust
// src/layers/mod.rs
pub mod hyperboloid;
pub use hyperboloid::HyperboloidLayer;
```

## 3단계: CUDA 커널 (선택사항)

### CUDA 구현

```cuda
// src/layers/cuda/hyperboloid.cu
#include <cuda_runtime.h>
#include <math.h>

__device__ float lorentz_inner_cuda(
    const float* x, 
    const float* y, 
    int dim
) {
    float result = -x[0] * y[0];  // 시간 성분
    for (int i = 1; i < dim; i++) {
        result += x[i] * y[i];    // 공간 성분
    }
    return result;
}

__global__ void hyperboloid_distance_kernel(
    const float* x,
    const float* y,
    float* output,
    int batch_size,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float inner = lorentz_inner_cuda(
            &x[idx * dim], 
            &y[idx * dim], 
            dim
        );
        output[idx] = acoshf(-inner);
    }
}

extern "C" {
    void hyperboloid_distance_cuda(
        const float* x,
        const float* y,
        float* output,
        int batch_size,
        int dim
    ) {
        dim3 block(256);
        dim3 grid((batch_size + block.x - 1) / block.x);
        
        hyperboloid_distance_kernel<<<grid, block>>>(
            x, y, output, batch_size, dim
        );
        
        cudaDeviceSynchronize();
    }
}
```

### Rust에서 CUDA 연동

```rust
// src/layers/hyperboloid.rs (CUDA 부분 추가)
#[cfg(feature = "cuda")]
extern "C" {
    fn hyperboloid_distance_cuda(
        x: *const f32,
        y: *const f32,
        output: *mut f32,
        batch_size: i32,
        dim: i32,
    );
}

impl HyperboloidOps {
    #[cfg(feature = "cuda")]
    pub fn distance_cuda(&self, x: &Tensor, y: &Tensor) -> Tensor {
        let batch_size = x.size()[0] as i32;
        let dim = x.size()[1] as i32;
        
        let output = Tensor::zeros(&[batch_size as i64], x.kind());
        
        unsafe {
            hyperboloid_distance_cuda(
                x.data_ptr() as *const f32,
                y.data_ptr() as *const f32,
                output.data_ptr() as *mut f32,
                batch_size,
                dim,
            );
        }
        
        output
    }
}
```

## 4단계: Python 바인딩

### PyTorch 함수 정의

```python
# python/reality_stone/layers/hyperboloid.py
import torch
from torch.autograd import Function
from .. import _rust

class HyperboloidFunction(Function):
    @staticmethod
    def forward(ctx, u, v, c, t):
        """
        Forward pass for Hyperboloid layer
        
        Args:
            u: First input tensor [batch_size, dim+1]
            v: Second input tensor [batch_size, dim+1]
            c: Curvature parameter
            t: Interpolation parameter
            
        Returns:
            Combined tensor in hyperboloid space
        """
        # 컨텍스트에 저장 (backward에서 사용)
        ctx.save_for_backward(u, v)
        ctx.c = c
        ctx.t = t
        
        # Rust 함수 호출
        output = _rust.hyperboloid_forward(u, v, c, t)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for Hyperboloid layer
        """
        u, v = ctx.saved_tensors
        c = ctx.c
        t = ctx.t
        
        # Rust 함수 호출
        grad_u, grad_v = _rust.hyperboloid_backward(
            grad_output, u, v, c, t
        )
        
        # c와 t에 대한 그래디언트는 None 반환
        return grad_u, grad_v, None, None

def hyperboloid_layer(u, v, c=1.0, t=0.5):
    """
    Hyperboloid layer implementation
    
    Args:
        u: First input tensor
        v: Second input tensor  
        c: Curvature parameter (default: 1.0)
        t: Interpolation parameter (default: 0.5)
        
    Returns:
        Combined tensor in hyperboloid space
    """
    return HyperboloidFunction.apply(u, v, c, t)
```

### 모듈에 추가

```python
# python/reality_stone/layers/__init__.py
from .hyperboloid import hyperboloid_layer

__all__ = [
    'hyperboloid_layer',
    # ... 기존 레이어들
]
```

```python
# python/reality_stone/__init__.py
from .layers import hyperboloid_layer

__all__ = [
    'hyperboloid_layer',
    # ... 기존 함수들
]
```

## 5단계: 테스트 작성

### 단위 테스트

```python
# tests/test_hyperboloid.py
import torch
import pytest
import reality_stone as rs

class TestHyperboloidLayer:
    def test_forward_shape(self):
        """출력 형태가 올바른지 테스트"""
        u = torch.randn(32, 65)  # batch_size=32, dim+1=65
        v = torch.randn(32, 65)
        
        output = rs.hyperboloid_layer(u, v, c=1.0, t=0.5)
        
        assert output.shape == (32, 65)
        assert output.dtype == torch.float32
    
    def test_constraint_satisfaction(self):
        """Hyperboloid 제약 조건 만족 여부 테스트"""
        u = torch.randn(16, 33)
        v = torch.randn(16, 33)
        
        output = rs.hyperboloid_layer(u, v, c=1.0, t=0.5)
        
        # Lorentz 내적 계산: <x,x>_L = -1
        inner = rs.lorentz_inner(output, output)
        expected = torch.full_like(inner, -1.0)
        
        torch.testing.assert_close(inner, expected, atol=1e-6)
    
    def test_gradient_flow(self):
        """그래디언트 흐름 테스트"""
        u = torch.randn(8, 17, requires_grad=True)
        v = torch.randn(8, 17, requires_grad=True)
        
        output = rs.hyperboloid_layer(u, v, c=1.0, t=0.5)
        loss = output.sum()
        loss.backward()
        
        assert u.grad is not None
        assert v.grad is not None
        assert u.grad.shape == u.shape
        assert v.grad.shape == v.shape
    
    def test_numerical_stability(self):
        """수치적 안정성 테스트"""
        # 경계 근처 값들
        u = torch.randn(10, 9) * 0.99
        v = torch.randn(10, 9) * 0.99
        
        output = rs.hyperboloid_layer(u, v, c=1e-3, t=0.5)
        
        # NaN이나 Inf가 없어야 함
        assert torch.isfinite(output).all()
    
    @pytest.mark.parametrize("c", [1e-3, 1e-2, 1e-1, 1.0])
    def test_different_curvatures(self, c):
        """다양한 곡률 값 테스트"""
        u = torch.randn(5, 9)
        v = torch.randn(5, 9)
        
        output = rs.hyperboloid_layer(u, v, c=c, t=0.5)
        
        assert output.shape == (5, 9)
        assert torch.isfinite(output).all()
```

### 통합 테스트

```python
# tests/test_hyperboloid_integration.py
import torch
import torch.nn as nn
import reality_stone as rs

class HyperboloidMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder1 = nn.Linear(input_dim, hidden_dim)
        self.encoder2 = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        u = torch.tanh(self.encoder1(x))
        v = torch.tanh(self.encoder2(x))
        
        # Hyperboloid 공간으로 투영
        u_proj = rs.project_to_hyperboloid(u, c=1.0)
        v_proj = rs.project_to_hyperboloid(v, c=1.0)
        
        # Hyperboloid 레이어 적용
        h = rs.hyperboloid_layer(u_proj, v_proj, c=1.0, t=0.5)
        
        # 분류를 위해 공간 성분만 사용
        h_space = h[:, 1:]  # 시간 성분 제외
        
        return self.classifier(h_space)

def test_hyperboloid_training():
    """Hyperboloid 레이어를 사용한 모델 훈련 테스트"""
    model = HyperboloidMLP(64, 32, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 더미 데이터
    x = torch.randn(100, 64)
    y = torch.randint(0, 10, (100,))
    
    # 훈련 루프
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # 모델이 정상적으로 훈련되었는지 확인
    assert loss.item() < 10.0  # 합리적인 손실 값
```

## 6단계: 문서 작성

### API 문서

```markdown
# docs/user_manual/api_reference/layers/hyperboloid.md

# Hyperboloid 레이어

Hyperboloid 레이어는 hyperboloid 모델에서 동작하는 하이퍼볼릭 신경망 레이어입니다.

## 개요

Hyperboloid 모델은 민코프스키 공간에서 정의되는 하이퍼볼릭 공간의 모델로, 수치적 안정성이 뛰어납니다.

## 함수 API

### `hyperboloid_layer()`

```python
def hyperboloid_layer(
    u: torch.Tensor, 
    v: torch.Tensor, 
    c: float = 1.0, 
    t: float = 0.5
) -> torch.Tensor
```

두 텐서를 Hyperboloid 공간에서 결합합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `u` | `torch.Tensor` | 첫 번째 입력 텐서 | 필수 |
| `v` | `torch.Tensor` | 두 번째 입력 텐서 | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |
| `t` | `float` | 보간 비율 | `0.5` |

#### 예제

```python
import torch
import reality_stone as rs

u = torch.randn(32, 65)
v = torch.randn(32, 65)

result = rs.hyperboloid_layer(u, v, c=1.0, t=0.5)
```
```

## 7단계: 빌드 시스템 업데이트

### Cargo.toml 업데이트

```toml
# Cargo.toml에 새로운 기능 추가
[features]
default = ["hyperboloid"]
hyperboloid = []
```

### build.rs 업데이트

```rust
// build.rs
fn main() {
    // 기존 빌드 코드...
    
    // Hyperboloid CUDA 커널 빌드
    #[cfg(feature = "cuda")]
    {
        cc::Build::new()
            .cuda(true)
            .file("src/layers/cuda/hyperboloid.cu")
            .compile("hyperboloid_cuda");
    }
}
```

## 8단계: CI/CD 업데이트

### GitHub Actions 워크플로우

```yaml
# .github/workflows/test.yml
- name: Test new hyperboloid layer
  run: |
    python -m pytest tests/test_hyperboloid.py -v
    python -m pytest tests/test_hyperboloid_integration.py -v
```

## 모범 사례

### 1. 수치적 안정성
- 경계 조건 근처에서의 안정성 확보
- 적절한 클리핑 및 정규화 사용
- 수치적 오차 누적 방지

### 2. 성능 최적화
- 벡터화된 연산 사용
- 불필요한 메모리 할당 방지
- CUDA 커널 최적화

### 3. 테스트 커버리지
- 단위 테스트: 개별 함수 테스트
- 통합 테스트: 전체 워크플로우 테스트
- 성능 테스트: 벤치마크 및 회귀 테스트

### 4. 문서화
- 수학적 배경 설명
- 실용적인 사용 예제
- 성능 특성 및 제한사항

## 문제 해결

### 자주 발생하는 문제들

1. **컴파일 오류**: Rust 타입 불일치
2. **CUDA 오류**: 메모리 관리 문제
3. **Python 바인딩 오류**: 타입 변환 문제
4. **수치적 불안정성**: 경계 조건 처리

### 디버깅 팁

- 단계별 테스트 실행
- 작은 데이터셋으로 검증
- 수학적 정확성 확인
- 성능 프로파일링 수행

## 참고 자료

- **Hyperbolic Geometry**: 하이퍼볼릭 기하학 기초
- **PyTorch Extension**: PyTorch 확장 개발 가이드
- **CUDA Programming**: CUDA 프로그래밍 가이드
- **Rust FFI**: Rust Foreign Function Interface 