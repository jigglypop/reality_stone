# Lorentz 레이어

Lorentz 레이어는 하이퍼볼로이드 모델에서 동작하는 하이퍼볼릭 신경망 레이어입니다.

## 개요

Lorentz 모델은 하이퍼볼로이드 $\mathbb{L}^n = \{x \in \mathbb{R}^{n+1} : \langle x,x \rangle_{\mathcal{L}} = -1, x_0 > 0\}$에서 정의되는 하이퍼볼릭 공간의 모델입니다. 민코프스키 공간의 기하학적 특성을 활용하여 안정적인 수치 계산을 제공합니다.

## 함수 API

### `lorentz_layer()`

```python
def lorentz_layer(
    u: torch.Tensor, 
    v: torch.Tensor, 
    c: float, 
    t: float
) -> torch.Tensor
```

두 텐서를 Lorentz 공간에서 결합합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `u` | `torch.Tensor` | 첫 번째 입력 텐서 `[batch_size, dim+1]` | 필수 |
| `v` | `torch.Tensor` | 두 번째 입력 텐서 `[batch_size, dim+1]` | 필수 |
| `c` | `float` | 곡률 매개변수 (양수) | 필수 |
| `t` | `float` | 보간 비율 (0~1) | 필수 |

#### 반환값

- **타입**: `torch.Tensor`
- **형태**: `[batch_size, dim+1]` (입력과 동일)
- **설명**: Lorentz 공간에서 결합된 결과

#### 예제

```python
import torch
import reality_stone as rs

# Lorentz 공간의 점들 (dim+1 차원)
u = torch.randn(32, 65)  # 64차원 → 65차원 (하이퍼볼로이드)
v = torch.randn(32, 65)

# Lorentz 제약 조건 만족하도록 정규화
u = rs.lorentz_normalize(u, c=1.0)
v = rs.lorentz_normalize(v, c=1.0)

result = rs.lorentz_layer(u, v, c=1.0, t=0.5)
```

## 클래스 API

### `LorentzLayer`

```python
class LorentzLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v, c, t)
    
    @staticmethod
    def backward(ctx, grad_output)
```

PyTorch의 `autograd.Function`을 상속한 클래스로, 자동 미분을 지원합니다.

## 관련 연산 함수들

### `lorentz_add()`

```python
def lorentz_add(u: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor
```

Lorentz 공간에서의 덧셈 연산

#### 예제

```python
u = torch.randn(32, 65)
v = torch.randn(32, 65)
result = rs.lorentz_add(u, v, c=1.0)
```

### `lorentz_scalar_mul()`

```python
def lorentz_scalar_mul(x: torch.Tensor, r: float, c: float) -> torch.Tensor
```

Lorentz 공간에서의 스칼라 곱셈

#### 예제

```python
x = torch.randn(32, 65)
result = rs.lorentz_scalar_mul(x, r=0.5, c=1.0)
```

### `lorentz_distance()`

```python
def lorentz_distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor
```

Lorentz 공간에서의 거리 계산

#### 예제

```python
x = torch.randn(32, 65)
y = torch.randn(32, 65)
distances = rs.lorentz_distance(x, y, c=1.0)
print(distances.shape)  # [32]
```

### `lorentz_inner()`

```python
def lorentz_inner(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor
```

민코프스키 내적 계산

#### 예제

```python
u = torch.randn(32, 65)
v = torch.randn(32, 65)
inner_products = rs.lorentz_inner(u, v)
print(inner_products.shape)  # [32]
```

## 좌표 변환 함수들

### `lorentz_to_poincare()`

```python
def lorentz_to_poincare(x: torch.Tensor, c: float) -> torch.Tensor
```

Lorentz 모델에서 Poincaré Ball로 변환

### `lorentz_to_klein()`

```python
def lorentz_to_klein(x: torch.Tensor, c: float) -> torch.Tensor
```

Lorentz 모델에서 Klein 모델로 변환

### `from_poincare()`

```python
def from_poincare(
    x: torch.Tensor, 
    c: float = None, 
    kappas: torch.Tensor = None, 
    c_min: float = -2.0, 
    c_max: float = -0.1
) -> torch.Tensor
```

Poincaré Ball에서 Lorentz 모델로 변환

## 수학적 배경

### Lorentz 모델 (하이퍼볼로이드)

**정의**: $\mathbb{L}^n = \{x \in \mathbb{R}^{n+1} : \langle x,x \rangle_{\mathcal{L}} = -1, x_0 > 0\}$

**민코프스키 내적**:
$$\langle x,y \rangle_{\mathcal{L}} = -x_0y_0 + \sum_{i=1}^n x_iy_i$$

**거리 함수**:
$$d_{\mathbb{L}}(x,y) = \text{arccosh}(-\langle x,y \rangle_{\mathcal{L}})$$

### 주요 연산

**Lorentz 덧셈**:
$$x \oplus_c y = x + y + \frac{c}{1-c\langle x,y \rangle_{\mathcal{L}}} \langle x,y \rangle_{\mathcal{L}} \cdot \frac{x+y}{\|x+y\|_{\mathcal{L}}}$$

**스칼라 곱셈**:
$$r \otimes_c x = \text{cosh}(r \cdot \text{arccosh}(-x_0)) \cdot e_0 + \text{sinh}(r \cdot \text{arccosh}(-x_0)) \cdot \frac{x_{1:n}}{\|x_{1:n}\|}$$

## 성능 특성

### 시간 복잡도
- **Forward**: $O(bd)$ (b: 배치 크기, d: 차원)
- **Backward**: $O(bd)$

### 수치적 안정성
- Poincaré Ball보다 경계 근처에서 더 안정적
- 민코프스키 내적의 특성으로 인한 안정성

### GPU 가속
- CUDA 구현으로 대규모 배치 처리 최적화
- 메모리 효율적인 구현

## 🔧 실제 사용 예제

### 1. Lorentz 기반 분류 모델

```python
import torch
import torch.nn as nn
import reality_stone as rs

class LorentzClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, curvature=1.0):
        super().__init__()
        self.curvature = curvature
        
        # 입력을 Lorentz 공간으로 매핑
        self.to_lorentz = nn.Linear(input_dim, hidden_dim + 1)
        
        # 분류 헤드
        self.classifier = nn.Linear(hidden_dim + 1, num_classes)
        
    def forward(self, x):
        # Lorentz 공간으로 변환
        lorentz_x = self.to_lorentz(x)
        
        # Lorentz 제약 조건 만족
        lorentz_x = self.project_to_lorentz(lorentz_x)
        
        # 자기 자신과 결합 (identity operation)
        lorentz_features = rs.lorentz_layer(
            lorentz_x, lorentz_x, 
            c=self.curvature, t=0.5
        )
        
        # 분류
        return self.classifier(lorentz_features)
    
    def project_to_lorentz(self, x):
        # Lorentz 제약 조건 만족: <x,x>_L = -1
        x_space = x[:, 1:]  # 공간 부분
        x_time = torch.sqrt(1 + torch.sum(x_space ** 2, dim=1, keepdim=True))
        return torch.cat([x_time, x_space], dim=1)
```

### 2. Poincaré-Lorentz 하이브리드 모델

```python
class HybridModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, curvature=1.0):
        super().__init__()
        self.curvature = curvature
        
        self.poincare_encoder = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim + 1, 1)
        
    def forward(self, x):
        # Poincaré 공간에서 인코딩
        poincare_features = torch.tanh(self.poincare_encoder(x)) * 0.1
        
        # Lorentz 공간으로 변환
        lorentz_features = rs.from_poincare(
            poincare_features, c=self.curvature
        )
        
        # Lorentz 레이어 적용
        enhanced_features = rs.lorentz_layer(
            lorentz_features, lorentz_features,
            c=self.curvature, t=0.3
        )
        
        return self.output_layer(enhanced_features)
```

## 🚨 주의사항

### Lorentz 제약 조건
- 모든 점은 $\langle x,x \rangle_{\mathcal{L}} = -1$ 조건을 만족해야 함
- 입력 데이터를 적절히 정규화하여 사용

### 차원 증가
- n차원 입력이 (n+1)차원 Lorentz 공간으로 매핑됨
- 메모리 사용량 증가 고려 필요

### 수치적 정밀도
- 민코프스키 내적 계산 시 정밀도 중요
- float64 사용 권장 (정밀도가 중요한 경우)

## 📚 참고 문헌

1. **Hyperbolic Neural Networks** - Ganea et al. (2018)
2. **Lorentzian Distance Learning for Hyperbolic Representations** - Law et al. (2019)
3. **Hyperbolic Graph Neural Networks** - Chami et al. (2019) 