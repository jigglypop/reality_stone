# Reality Stone 곡률 연산 코딩 분석

## 1. 개요

Reality Stone에서 곡률(curvature) 파라미터 `c`는 쌍곡 공간의 기하학적 특성을 결정하는 핵심 요소입니다. 이 문서는 곡률이 코드에서 어떻게 계산되고 사용되는지 분석합니다.

## 2. 곡률 연산 구현 계층

### 2.1 Rust 코어 구현

**YES, Rust를 사용합니다!** 핵심 곡률 연산은 모두 Rust로 구현되어 있습니다:

```rust
// src/ops/mobius.rs
pub fn mobius_add(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let u2 = u_row.dot(&u_row);
    let v2 = v_row.dot(&v_row);
    let uv = u_row.dot(&v_row);
    let c2 = c * c;  // 곡률의 제곱
    
    // Möbius 덧셈 공식에서 곡률 사용
    let coeff_u = (1.0 + 2.0 * c * uv + c * v2) / 
                  (1.0 + 2.0 * c * uv + c2 * u2 * v2).max(MIN_DENOMINATOR);
    let coeff_v = (1.0 - c * u2) / 
                  (1.0 + 2.0 * c * uv + c2 * u2 * v2).max(MIN_DENOMINATOR);
}

pub fn mobius_scalar(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {
    let sqrtc = c.sqrt();  // 곡률의 제곱근
    let scn = (sqrtc * norm).min(1.0 - BOUNDARY_EPS).max(EPS);
    let alpha = scn.atanh();
    let beta = (r * alpha).tanh();
    let scale = beta / (sqrtc * norm);
}
```

### 2.2 Python 바인딩

Rust 함수들은 PyO3를 통해 Python으로 노출됩니다:

```rust
// src/bindings/mobius.rs
#[pyfunction]
pub fn mobius_add_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,  // 곡률 파라미터
) -> &'py PyArray2<f32> {
    let result = mobius::mobius_add(&u_arr, &v_arr, c);
    result.into_pyarray(py)
}
```

### 2.3 Python 래퍼

Python에서는 GPU/CPU와 gradient 지원을 위한 래퍼가 있습니다:

```python
# python/reality_stone/core/ops.py
def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    if x.device.type == 'cuda':
        # CUDA 버전 호출
        mobius_add_cuda(
            x.data_ptr(),
            y.data_ptr(),
            out.data_ptr(),
            batch_size,
            dim,
            c  # 곡률 전달
        )
    else:
        # CPU 버전 (Rust 구현) 호출
        result_np = mobius_add_cpu(x_np, y_np, c)
```

## 3. 곡률 사용 패턴

### 3.1 고정 곡률

가장 기본적인 사용 방법:

```python
# 고정된 곡률값 사용
c = 1e-3  # 매우 작은 곡률 (거의 유클리드)
z = rs.poincare_ball_layer(h, u, c, t)
```

### 3.2 동적 곡률

입력에 따라 곡률을 학습하는 고급 기능:

```python
class DynamicCurvatureLayer(nn.Module):
    def __init__(self, input_dim: int, base_curvature: float = 1.0):
        super().__init__()
        # 곡률 예측을 위한 학습 가능한 파라미터
        self.curvature_weight = nn.Parameter(torch.randn(1, input_dim) * 0.1)
        self.curvature_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 입력에 따라 곡률 예측
        return predict_dynamic_curvature(
            x, 
            self.curvature_weight, 
            self.curvature_bias, 
            self.base_curvature
        )
```

### 3.3 곡률 예측 구현

```python
# advanced.py의 DynamicCurvaturePrediction
class DynamicCurvaturePrediction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, base_curvature):
        if not _C:  # Rust 확장이 없을 때
            return torch.full((x.size(0),), base_curvature, device=x.device)
        
        if x.is_cuda and _has_cuda:
            # CUDA 구현 호출
            return dynamic_curvature_prediction_cuda(x, weight, bias, base_curvature)
        else:
            # CPU 구현 호출
            return dynamic_curvature_prediction_cpu(x, weight, bias, base_curvature)
```

## 4. 곡률의 수학적 역할

### 4.1 공간의 경계 결정

Poincaré ball의 경계는 곡률에 의해 결정됩니다:

```python
max_norm = 1.0 / sqrt(c)  # Ball의 반지름

# 경계 처리
if norm >= max_norm:
    x = x * (max_norm - eps) / norm
```

### 4.2 메트릭 스케일링

리만 메트릭이 곡률에 따라 변합니다:

```python
# Poincaré ball의 메트릭 계수
metric_factor = 4.0 / ((1 - c * norm_sq) ** 2)
```

### 4.3 거리 계산

```python
# Poincaré 거리
d = (2.0 / sqrt(c)) * arctanh(sqrt(c) * ||u - v|| / denominator)
```

## 5. CUDA 최적화

CUDA 커널에서도 곡률이 효율적으로 처리됩니다:

```cuda
// src/ops/cuda/mobius.cu
__global__ void mobius_add_kernel(
    float* out, const float* u, const float* v, 
    float c, int batch_size, int dim
) {
    float c2 = c * c;
    float coeff_u = (1.0f + 2.0f * c * uv + c * v2) / denominator;
    float coeff_v = (1.0f - c * u2) / denominator;
}
```

## 6. 곡률 정규화

곡률이 적절한 범위를 유지하도록 정규화:

```python
def hyperbolic_regularization(x, weights, curvature, ...):
    # 곡률 정규화 손실
    curvature_loss = (curvature - target_curvature) ** 2
    return lambda_curvature * curvature_loss
```

## 7. 성능 최적화 전략

### 7.1 곡률 계산 캐싱

```python
# 자주 사용되는 값들을 미리 계산
sqrtc = sqrt(c)
c2 = c * c
inv_sqrtc = 1.0 / sqrtc
```

### 7.2 Fused Operations

여러 곡률 연산을 하나의 커널로 통합:

```python
def hyperbolic_linear_fused(input, weight, bias, curvature):
    # log_map → linear → exp_map을 한 번에 수행
    return HyperbolicLinearFused.apply(input, weight, bias, curvature)
```

## 8. 실제 사용 예시

### 8.1 MNIST 실험

```python
class GeodesicMLP(nn.Module):
    def __init__(self, c=1e-3):  # 작은 곡률
        self.c = c
        
    def forward(self, x):
        z = rs.poincare_ball_layer(h, u, self.c, self.t)
```

### 8.2 곡률별 성능

| 곡률 (c) | 공간 특성 | Ball 반지름 | 성능 |
|---------|----------|------------|------|
| 1e-3 | 거의 유클리드 | 31.62 | 안정적 |
| 1.0 | 표준 쌍곡 | 1.0 | 균형 |
| 10.0 | 강한 쌍곡성 | 0.316 | 불안정 |

## 9. 구현 흐름도

```
Python 호출
    ↓
곡률 파라미터 c 전달
    ↓
Gradient 필요? ─Yes→ PyTorch 구현 (tensors.py)
    ↓ No
CUDA 디바이스? ─Yes→ CUDA 커널 (*.cu)
    ↓ No
Rust CPU 구현 (ops/*.rs)
    ↓
결과 반환
```

## 10. PyTorch Gradient 지원

Gradient가 필요한 경우 PyTorch 구현을 사용합니다:

```python
# core/tensors.py
def _mobius_add_torch(x: torch.Tensor, y: torch.Tensor, c: float, eps: float = 1e-7) -> torch.Tensor:
    """PyTorch implementation of Möbius addition (for gradient computation)"""
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True)
    xy = (x * y).sum(dim=1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return num / denom.clamp_min(eps)

def _mobius_scalar_torch(x: torch.Tensor, r: float, c: float, eps: float = 1e-7) -> torch.Tensor:
    """PyTorch implementation of Möbius scalar multiplication"""
    sqrtc = c ** 0.5
    x_norm = x.norm(dim=1, keepdim=True).clamp_min(eps)
    scale = torch.tanh(r * torch.atanh(sqrtc * x_norm)) / (sqrtc * x_norm)
    return scale * x
```

## 11. 동적 곡률의 실제 사용

```python
# layers/poincare.py
class HyperbolicLinearAdvanced(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.enable_dynamic_curvature:
            # 입력에 따라 곡률 예측
            curvatures = self.dynamic_curvature(x)
            curvature = curvatures.mean().item()
        else:
            curvature = self.curvature.item()
            
        # Fused 연산 사용
        if self.config.enable_fused_ops:
            output = hyperbolic_linear_fused(x, self.weight, self.bias, curvature)
```

## 12. C++ 확장과의 연동

```python
# advanced.py
try:
    import reality_stone._C as _C
    HAS_CUDA = hasattr(_C, 'mobius_add_cuda')
except ImportError:
    _C = None
    HAS_CUDA = False
```

C++ 확장이 있을 때는 더 효율적인 구현을 사용하고, 없을 때는 PyTorch fallback을 사용합니다.

## 13. 곡률 관련 설정

```python
class AdvancedConfig:
    def __init__(
        self,
        enable_dynamic_curvature: bool = False,  # 동적 곡률 사용 여부
        lambda_curvature: float = 0.1,          # 곡률 정규화 가중치
        base_curvature: float = 1.0,            # 기본 곡률값
        ...
    ):
```

## 14. 결론

Reality Stone의 곡률 연산은:

1. **Rust 기반**: 핵심 연산은 모두 Rust로 구현 ✓
2. **3단계 구조**: Rust 코어 → Python 바인딩 → PyTorch 래퍼
3. **유연성**: 고정/동적 곡률 모두 지원
4. **최적화**: CUDA 가속 및 Fused Operations
5. **수치 안정성**: 경계 처리 및 정규화
6. **Gradient 지원**: PyTorch 구현으로 자동 미분 가능

이러한 설계로 수학적 정확성과 계산 효율성을 동시에 달성합니다. 