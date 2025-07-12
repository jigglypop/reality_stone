# 코딩 스타일 가이드

Reality Stone 프로젝트의 코딩 스타일과 베스트 프랙티스를 정의합니다. 일관된 코드 스타일은 가독성을 높이고 협업을 원활하게 만듭니다.

## 전체 원칙

### 1. 가독성 우선
- 복잡한 로직보다 읽기 쉬운 코드를 우선시
- 의미 있는 변수명과 함수명 사용
- 적절한 주석과 문서화

### 2. DRY (Don't Repeat Yourself)
- 중복 코드 제거
- 공통 로직은 함수나 모듈로 추출
- 상수는 한 곳에서 관리

### 3. KISS (Keep It Simple, Stupid)
- 과도한 추상화 지양
- 단순하고 직관적인 해결책 선호
- 필요 이상으로 복잡하게 만들지 않기

## Rust 코딩 스타일

### 네이밍 규칙

```rust
// 모듈명: snake_case
mod hyperbolic_layers;
mod tensor_utils;

// 구조체명: PascalCase
struct PoincareBallLayer;
struct HyperboloidOps;

// 함수명: snake_case
fn compute_distance() {}
fn normalize_tensor() {}

// 상수명: SCREAMING_SNAKE_CASE
const DEFAULT_CURVATURE: f64 = 1.0;
const MAX_ITERATIONS: usize = 1000;

// 변수명: snake_case
let batch_size = 32;
let learning_rate = 1e-3;
```

### 함수 정의

```rust
// 좋은 예시
impl PoincareBallOps {
    /// 두 점 사이의 하이퍼볼릭 거리를 계산합니다.
    /// 
    /// # Arguments
    /// * `x` - 첫 번째 점
    /// * `y` - 두 번째 점
    /// * `c` - 곡률 매개변수
    /// 
    /// # Returns
    /// 하이퍼볼릭 거리 텐서
    pub fn distance(
        &self, 
        x: &Tensor, 
        y: &Tensor, 
        c: f64
    ) -> Result<Tensor, Error> {
        self.validate_inputs(x, y)?;
        
        let diff = self.mobius_add(x, &self.mobius_neg(y), c)?;
        let norm = self.norm(&diff)?;
        
        Ok(2.0 / c.sqrt() * norm.atanh())
    }
    
    fn validate_inputs(&self, x: &Tensor, y: &Tensor) -> Result<(), Error> {
        if x.size() != y.size() {
            return Err(Error::DimensionMismatch);
        }
        Ok(())
    }
}

// 나쁜 예시
impl PoincareBallOps {
    // 문서화 없음, 긴 함수, 에러 처리 없음
    pub fn dist(x: &Tensor, y: &Tensor, c: f64) -> Tensor {
        let a = x.clone();
        let b = y.clone();
        let neg_b = -b;
        let diff = self.mobius_add(&a, &neg_b, c);
        let n = diff.norm();
        2.0 / c.sqrt() * n.atanh()
    }
}
```

### 에러 처리

```rust
// Result 타입 사용
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HyperbolicError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Invalid curvature: {curvature} (must be positive)")]
    InvalidCurvature { curvature: f64 },
    
    #[error("Numerical instability detected")]
    NumericalInstability,
}

// 사용 예시
fn compute_distance(x: &Tensor, y: &Tensor, c: f64) -> Result<Tensor, HyperbolicError> {
    if c <= 0.0 {
        return Err(HyperbolicError::InvalidCurvature { curvature: c });
    }
    
    if x.size()[1] != y.size()[1] {
        return Err(HyperbolicError::DimensionMismatch {
            expected: x.size()[1],
            actual: y.size()[1],
        });
    }
    
    // 실제 계산...
    Ok(result)
}
```

### 모듈 구조

```rust
// src/layers/mod.rs
pub mod poincare;
pub mod lorentz;
pub mod klein;

pub use poincare::PoincareBallLayer;
pub use lorentz::LorentzLayer;
pub use klein::KleinLayer;

// 공통 트레이트 정의
pub trait HyperbolicLayer {
    fn forward(&self, u: &Tensor, v: &Tensor, t: f64) -> Result<Tensor, HyperbolicError>;
    fn backward(&self, grad: &Tensor, ctx: &Context) -> Result<(Tensor, Tensor), HyperbolicError>;
}
```

## Python 코딩 스타일

### PEP 8 준수

```python
# 좋은 예시
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

class HyperbolicMLP(nn.Module):
    """하이퍼볼릭 공간에서 동작하는 다층 퍼셉트론.
    
    Args:
        input_dim: 입력 차원
        hidden_dim: 은닉층 차원
        output_dim: 출력 차원
        curvature: 곡률 매개변수 (기본값: 1.0)
        model_type: 하이퍼볼릭 모델 타입 ('poincare', 'lorentz', 'klein')
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        curvature: float = 1.0,
        model_type: str = 'poincare'
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.curvature = curvature
        self.model_type = model_type
        
        # 레이어 정의
        self.encoder1 = nn.Linear(input_dim, hidden_dim)
        self.encoder2 = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass 수행.
        
        Args:
            x: 입력 텐서 [batch_size, input_dim]
            
        Returns:
            출력 텐서 [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        # 두 개의 인코더로 다른 표현 학습
        u = torch.tanh(self.encoder1(x))
        v = torch.tanh(self.encoder2(x))
        
        # 하이퍼볼릭 공간에서 결합
        if self.model_type == 'poincare':
            features = poincare_ball_layer(u, v, c=self.curvature, t=0.5)
        elif self.model_type == 'lorentz':
            features = lorentz_layer(u, v, c=self.curvature, t=0.5)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return self.classifier(features)

# 나쁜 예시
class HyperbolicMLP(nn.Module):
    def __init__(self, i, h, o, c=1.0, t='poincare'):  # 의미 없는 변수명
        super().__init__()
        self.l1 = nn.Linear(i, h)  # 의미 없는 레이어명
        self.l2 = nn.Linear(i, h)
        self.l3 = nn.Linear(h, o)
        
    def forward(self, x):  # 타입 힌트 없음, 문서화 없음
        a = torch.tanh(self.l1(x))
        b = torch.tanh(self.l2(x))
        c = poincare_ball_layer(a, b, c=1.0, t=0.5)  # 하드코딩된 값
        return self.l3(c)
```

### 타입 힌트 사용

```python
from typing import Union, Optional, Tuple, List, Dict, Any
import torch

def hyperbolic_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    model: str = 'poincare',
    curvature: float = 1.0
) -> torch.Tensor:
    """하이퍼볼릭 공간에서 두 점 사이의 거리를 계산합니다."""
    pass

def batch_convert_coordinates(
    tensors: List[torch.Tensor],
    from_model: str,
    to_model: str,
    curvature: float = 1.0
) -> List[torch.Tensor]:
    """여러 텐서를 일괄 좌표 변환합니다."""
    pass

class ModelConfig:
    """모델 설정을 담는 데이터 클래스."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        curvatures: Optional[Dict[str, float]] = None
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.curvatures = curvatures or {'default': 1.0}
```

### 예외 처리

```python
class HyperbolicError(Exception):
    """하이퍼볼릭 연산 관련 예외의 기본 클래스."""
    pass

class DimensionMismatchError(HyperbolicError):
    """차원 불일치 예외."""
    
    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"Expected dimension {expected}, got {actual}")
        self.expected = expected
        self.actual = actual

class InvalidCurvatureError(HyperbolicError):
    """잘못된 곡률 값 예외."""
    
    def __init__(self, curvature: float) -> None:
        super().__init__(f"Invalid curvature: {curvature} (must be positive)")
        self.curvature = curvature

# 사용 예시
def validate_curvature(c: float) -> None:
    """곡률 값의 유효성을 검증합니다."""
    if c <= 0:
        raise InvalidCurvatureError(c)
    if c > 10:
        warnings.warn(f"Large curvature value {c} may cause numerical instability")

def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """Poincaré Ball에서 거리를 계산합니다."""
    validate_curvature(c)
    
    if x.shape != y.shape:
        raise DimensionMismatchError(x.shape[-1], y.shape[-1])
    
    # 실제 계산...
    return result
```

## CUDA 코딩 스타일

### 커널 명명 규칙

```cuda
// 좋은 예시
__global__ void poincare_distance_kernel(
    const float* x,
    const float* y,
    float* output,
    int batch_size,
    int dim,
    float curvature
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float dist = compute_poincare_distance_single(
            &x[idx * dim], 
            &y[idx * dim], 
            dim, 
            curvature
        );
        output[idx] = dist;
    }
}

__device__ float compute_poincare_distance_single(
    const float* x,
    const float* y,
    int dim,
    float curvature
) {
    // 단일 점 거리 계산
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = x[i] - y[i];
        norm_sq += diff * diff;
    }
    
    float x_norm_sq = 0.0f;
    float y_norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        x_norm_sq += x[i] * x[i];
        y_norm_sq += y[i] * y[i];
    }
    
    float numerator = 2.0f * norm_sq;
    float denominator = (1.0f - curvature * x_norm_sq) * (1.0f - curvature * y_norm_sq);
    
    return acoshf(1.0f + numerator / denominator) / sqrtf(curvature);
}

// 나쁜 예시
__global__ void kernel(float* a, float* b, float* c, int n, int d, float k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = 0;
        for (int j = 0; j < d; j++) {
            s += (a[i*d+j] - b[i*d+j]) * (a[i*d+j] - b[i*d+j]);
        }
        c[i] = acoshf(1 + 2*s/((1-k*dot(a+i*d,a+i*d,d))*(1-k*dot(b+i*d,b+i*d,d)))) / sqrtf(k);
    }
}
```

### 메모리 관리

```cuda
// 좋은 예시: 명확한 메모리 관리
extern "C" {
    void poincare_distance_cuda_wrapper(
        const float* h_x,
        const float* h_y,
        float* h_output,
        int batch_size,
        int dim,
        float curvature
    ) {
        // 디바이스 메모리 할당
        float* d_x = nullptr;
        float* d_y = nullptr;
        float* d_output = nullptr;
        
        size_t input_size = batch_size * dim * sizeof(float);
        size_t output_size = batch_size * sizeof(float);
        
        cudaMalloc(&d_x, input_size);
        cudaMalloc(&d_y, input_size);
        cudaMalloc(&d_output, output_size);
        
        // 호스트에서 디바이스로 복사
        cudaMemcpy(d_x, h_x, input_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y, input_size, cudaMemcpyHostToDevice);
        
        // 커널 실행
        dim3 block_size(256);
        dim3 grid_size((batch_size + block_size.x - 1) / block_size.x);
        
        poincare_distance_kernel<<<grid_size, block_size>>>(
            d_x, d_y, d_output, batch_size, dim, curvature
        );
        
        // 동기화 및 에러 체크
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        }
        
        // 결과 복사
        cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);
        
        // 메모리 해제
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_output);
    }
}
```

## 테스트 코딩 스타일

### 테스트 구조

```python
# tests/test_poincare_layer.py
import pytest
import torch
import reality_stone as rs
from reality_stone.errors import DimensionMismatchError, InvalidCurvatureError

class TestPoincareBallLayer:
    """Poincaré Ball 레이어 테스트 클래스."""
    
    @pytest.fixture
    def sample_tensors(self):
        """테스트용 샘플 텐서들을 생성합니다."""
        torch.manual_seed(42)  # 재현 가능한 결과
        u = torch.randn(32, 64) * 0.1
        v = torch.randn(32, 64) * 0.1
        return u, v
    
    def test_forward_shape_정상_입력(self, sample_tensors):
        """정상적인 입력에 대해 출력 형태가 올바른지 테스트합니다."""
        u, v = sample_tensors
        output = rs.poincare_ball_layer(u, v, c=1.0, t=0.5)
        
        assert output.shape == u.shape
        assert output.dtype == u.dtype
    
    def test_forward_차원_불일치_예외(self):
        """차원이 다른 입력에 대해 예외가 발생하는지 테스트합니다."""
        u = torch.randn(32, 64)
        v = torch.randn(32, 32)  # 다른 차원
        
        with pytest.raises(DimensionMismatchError):
            rs.poincare_ball_layer(u, v, c=1.0, t=0.5)
    
    def test_invalid_curvature_예외(self, sample_tensors):
        """잘못된 곡률 값에 대해 예외가 발생하는지 테스트합니다."""
        u, v = sample_tensors
        
        with pytest.raises(InvalidCurvatureError):
            rs.poincare_ball_layer(u, v, c=-1.0, t=0.5)  # 음수 곡률
        
        with pytest.raises(InvalidCurvatureError):
            rs.poincare_ball_layer(u, v, c=0.0, t=0.5)   # 0 곡률
    
    @pytest.mark.parametrize("curvature", [1e-3, 1e-2, 1e-1, 1.0, 10.0])
    def test_different_curvatures_수치적_안정성(self, sample_tensors, curvature):
        """다양한 곡률 값에서 수치적 안정성을 테스트합니다."""
        u, v = sample_tensors
        output = rs.poincare_ball_layer(u, v, c=curvature, t=0.5)
        
        # NaN이나 Inf가 없어야 함
        assert torch.isfinite(output).all(), f"Numerical instability at c={curvature}"
        
        # 출력이 단위 원 내부에 있어야 함
        norms = torch.norm(output, dim=-1)
        assert torch.all(norms < 1.0), f"Output outside unit ball at c={curvature}"
    
    def test_gradient_flow_정상_동작(self, sample_tensors):
        """그래디언트가 정상적으로 흐르는지 테스트합니다."""
        u, v = sample_tensors
        u.requires_grad_(True)
        v.requires_grad_(True)
        
        output = rs.poincare_ball_layer(u, v, c=1.0, t=0.5)
        loss = output.sum()
        loss.backward()
        
        assert u.grad is not None, "u에 대한 그래디언트가 None입니다"
        assert v.grad is not None, "v에 대한 그래디언트가 None입니다"
        assert torch.isfinite(u.grad).all(), "u 그래디언트에 NaN/Inf가 있습니다"
        assert torch.isfinite(v.grad).all(), "v 그래디언트에 NaN/Inf가 있습니다"
```

## 문서화 스타일

### Rust 문서화

```rust
/// Poincaré Ball 모델에서 두 점 사이의 거리를 계산합니다.
///
/// 이 함수는 하이퍼볼릭 공간에서 정의된 Poincaré Ball 메트릭을 사용하여
/// 두 점 사이의 거리를 계산합니다. 계산된 거리는 항상 양수입니다.
///
/// # Arguments
///
/// * `x` - 첫 번째 점을 나타내는 텐서 `[batch_size, dim]`
/// * `y` - 두 번째 점을 나타내는 텐서 `[batch_size, dim]`
/// * `c` - 곡률 매개변수 (양수여야 함)
///
/// # Returns
///
/// 각 배치 요소에 대한 거리를 담은 텐서 `[batch_size]`
///
/// # Errors
///
/// * `HyperbolicError::InvalidCurvature` - 곡률이 0 이하인 경우
/// * `HyperbolicError::DimensionMismatch` - 입력 텐서의 차원이 다른 경우
///
/// # Examples
///
/// ```rust
/// use reality_stone::poincare::distance;
/// use torch::Tensor;
///
/// let x = Tensor::randn(&[32, 64], torch::Kind::Float) * 0.1;
/// let y = Tensor::randn(&[32, 64], torch::Kind::Float) * 0.1;
/// 
/// let distances = distance(&x, &y, 1.0)?;
/// assert_eq!(distances.size(), [32]);
/// ```
///
/// # Mathematical Background
///
/// Poincaré Ball에서의 거리는 다음 공식으로 계산됩니다:
///
/// ```text
/// d(x,y) = (2/√c) * tanh⁻¹(√c * |(-x) ⊕_c y|)
/// ```
///
/// 여기서 ⊕_c는 Mobius 덧셈을 나타냅니다.
pub fn distance(x: &Tensor, y: &Tensor, c: f64) -> Result<Tensor, HyperbolicError> {
    // 구현...
}
```

### Python 문서화

```python
def poincare_ball_layer(
    u: torch.Tensor,
    v: torch.Tensor,
    c: float = 1.0,
    t: float = 0.5
) -> torch.Tensor:
    """Poincaré Ball 공간에서 두 텐서를 결합합니다.
    
    이 함수는 두 입력 텐서를 하이퍼볼릭 공간에서 보간하여 결합합니다.
    결과는 계층적 표현 학습에 유용한 특성을 가집니다.
    
    Args:
        u: 첫 번째 입력 텐서, 형태 [batch_size, dim]
        v: 두 번째 입력 텐서, 형태 [batch_size, dim]
        c: 곡률 매개변수. 양수여야 하며, 작을수록 평평한 공간을 의미합니다.
           일반적으로 1e-3에서 1e-1 사이의 값을 사용합니다.
        t: 보간 비율. 0.0이면 u만, 1.0이면 v만, 0.5면 균등 혼합을 의미합니다.
    
    Returns:
        하이퍼볼릭 공간에서 결합된 텐서, 형태 [batch_size, dim]
    
    Raises:
        DimensionMismatchError: u와 v의 차원이 다른 경우
        InvalidCurvatureError: c가 0 이하인 경우
        ValueError: t가 [0, 1] 범위를 벗어난 경우
    
    Examples:
        기본 사용법:
        
        >>> import torch
        >>> import reality_stone as rs
        >>> u = torch.randn(32, 64) * 0.1
        >>> v = torch.randn(32, 64) * 0.1
        >>> result = rs.poincare_ball_layer(u, v, c=1.0, t=0.5)
        >>> result.shape
        torch.Size([32, 64])
        
        다양한 곡률 값 사용:
        
        >>> # 거의 평평한 공간
        >>> result_flat = rs.poincare_ball_layer(u, v, c=1e-3, t=0.5)
        >>> # 높은 곡률 공간
        >>> result_curved = rs.poincare_ball_layer(u, v, c=1e-1, t=0.5)
    
    Note:
        - 입력 텐서의 노름이 1에 가까우면 수치적 불안정성이 발생할 수 있습니다.
        - 큰 곡률 값(c > 1)은 그래디언트 폭발을 야기할 수 있습니다.
        - 성능상 중요한 부분에서는 CUDA 가속 버전이 자동으로 사용됩니다.
    
    Mathematical Background:
        이 함수는 다음 연산을 수행합니다:
        
        1. 스칼라 곱셈: u' = (1-t) ⊗_c u, v' = t ⊗_c v
        2. Mobius 덧셈: result = u' ⊕_c v'
        
        여기서 ⊗_c와 ⊕_c는 각각 Mobius 스칼라 곱셈과 덧셈입니다.
    """
    # 구현...
```

## 리팩토링 체크리스트

### 코드 리뷰 전 확인사항

- [ ] 함수가 단일 책임을 가지고 있는가?
- [ ] 변수명과 함수명이 의도를 명확히 표현하는가?
- [ ] 중복 코드가 없는가?
- [ ] 매직 넘버가 상수로 정의되어 있는가?
- [ ] 에러 처리가 적절히 되어 있는가?
- [ ] 테스트 코드가 작성되어 있는가?

### 성능 관련

- [ ] 불필요한 반복문이 없는가?
- [ ] 캐싱을 활용할 수 있는 부분이 있는가?
- [ ] 비동기 처리가 효율적으로 되어 있는가?

### 가독성 관련

- [ ] 코드 중첩이 3단계를 넘지 않는가?
- [ ] 복잡한 조건문이 함수로 추출되어 있는가?

## 자동화 도구

### 코드 포맷팅

```bash
# Rust 코드 포맷팅
cargo fmt

# Python 코드 포맷팅
black .
isort .

# CUDA 코드 포맷팅 (clang-format 사용)
find . -name "*.cu" -o -name "*.cuh" | xargs clang-format -i
```

### 린팅

```bash
# Rust 린팅
cargo clippy -- -D warnings

# Python 린팅
flake8 .
mypy .

# 전체 검사
./scripts/check_code_style.sh
```

이러한 코딩 스타일을 일관되게 적용하면 코드의 품질과 가독성을 크게 향상시킬 수 있습니다. 