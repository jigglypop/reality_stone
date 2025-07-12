# Lorentz 연산

Lorentz 모델에서 사용되는 핵심 연산들입니다. 이 문서는 Reality Stone에서 제공하는 Lorentz 연산 함수들을 상세히 설명합니다.

## 개요

Lorentz 모델은 하이퍼볼로이드에서 정의되는 하이퍼볼릭 공간의 모델로, 민코프스키 내적을 사용하여 수치적으로 안정적인 연산을 제공합니다.

## 핵심 연산 함수들

### `lorentz_add()`

```python
def lorentz_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor
```

두 점의 Lorentz 덧셈을 계산합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 첫 번째 점 `[batch_size, dim+1]` | 필수 |
| `y` | `torch.Tensor` | 두 번째 점 `[batch_size, dim+1]` | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 수학적 정의

$$x \oplus_c y = x + y + \frac{c}{1-c\langle x,y \rangle_{\mathcal{L}}} \langle x,y \rangle_{\mathcal{L}} \cdot \frac{x+y}{\|x+y\|_{\mathcal{L}}}$$

여기서 $\langle x,y \rangle_{\mathcal{L}} = -x_0y_0 + \sum_{i=1}^n x_iy_i$는 민코프스키 내적입니다.

#### 예제

```python
import torch
import reality_stone as rs

# Lorentz 제약 조건을 만족하는 점들
x = torch.tensor([[1.1, 0.1, 0.2], [1.2, 0.3, 0.4]])
y = torch.tensor([[1.05, 0.05, 0.1], [1.15, 0.15, 0.25]])

result = rs.lorentz_add(x, y, c=1.0)
print(result.shape)  # [2, 3]
```

### `lorentz_scalar_mul()`

```python
def lorentz_scalar_mul(x: torch.Tensor, r: float, c: float = 1.0) -> torch.Tensor
```

점의 Lorentz 스칼라 곱셈을 계산합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 입력 점 `[batch_size, dim+1]` | 필수 |
| `r` | `float` | 스칼라 값 | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 수학적 정의

$$r \otimes_c x = \text{cosh}(r \cdot \text{arccosh}(-x_0)) \cdot e_0 + \text{sinh}(r \cdot \text{arccosh}(-x_0)) \cdot \frac{x_{1:n}}{\|x_{1:n}\|}$$

#### 예제

```python
import torch
import reality_stone as rs

x = torch.tensor([[1.1, 0.1, 0.2], [1.2, 0.3, 0.4]])
r = 0.5

result = rs.lorentz_scalar_mul(x, r, c=1.0)
print(result.shape)  # [2, 3]
```

### `lorentz_distance()`

```python
def lorentz_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor
```

두 점 사이의 하이퍼볼릭 거리를 계산합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 첫 번째 점 `[batch_size, dim+1]` | 필수 |
| `y` | `torch.Tensor` | 두 번째 점 `[batch_size, dim+1]` | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 수학적 정의

$$d_{\mathbb{L}}(x,y) = \frac{1}{\sqrt{c}} \text{arccosh}(-\sqrt{c}\langle x,y \rangle_{\mathcal{L}})$$

#### 예제

```python
import torch
import reality_stone as rs

x = torch.tensor([[1.1, 0.1, 0.2], [1.2, 0.3, 0.4]])
y = torch.tensor([[1.05, 0.05, 0.1], [1.15, 0.15, 0.25]])

distances = rs.lorentz_distance(x, y, c=1.0)
print(distances.shape)  # [2]
```

### `lorentz_inner()`

```python
def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor
```

민코프스키 내적을 계산합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 첫 번째 점 `[batch_size, dim+1]` | 필수 |
| `y` | `torch.Tensor` | 두 번째 점 `[batch_size, dim+1]` | 필수 |

#### 수학적 정의

$$\langle x,y \rangle_{\mathcal{L}} = -x_0y_0 + \sum_{i=1}^n x_iy_i$$

#### 예제

```python
import torch
import reality_stone as rs

x = torch.tensor([[1.1, 0.1, 0.2], [1.2, 0.3, 0.4]])
y = torch.tensor([[1.05, 0.05, 0.1], [1.15, 0.15, 0.25]])

inner_products = rs.lorentz_inner(x, y)
print(inner_products.shape)  # [2]
```

## 유틸리티 함수들

### `lorentz_normalize()`

```python
def lorentz_normalize(x: torch.Tensor, c: float = 1.0) -> torch.Tensor
```

벡터를 Lorentz 제약 조건을 만족하도록 정규화합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 입력 벡터 `[batch_size, dim+1]` | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 수학적 정의

정규화된 벡터는 다음 조건을 만족합니다:
$$\langle x,x \rangle_{\mathcal{L}} = -\frac{1}{c}$$

#### 예제

```python
import torch
import reality_stone as rs

x = torch.randn(32, 65)  # 임의의 벡터
x_normalized = rs.lorentz_normalize(x, c=1.0)

# 제약 조건 확인
inner = rs.lorentz_inner(x_normalized, x_normalized)
print(torch.allclose(inner, torch.tensor(-1.0)))  # True
```

### `project_to_lorentz()`

```python
def project_to_lorentz(x: torch.Tensor, c: float = 1.0) -> torch.Tensor
```

유클리드 벡터를 Lorentz 다양체로 투영합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 입력 벡터 `[batch_size, dim]` | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 수학적 정의

$$\text{proj}(x) = \left(\sqrt{\frac{1}{c} + \|x\|^2}, x\right)$$

#### 예제

```python
import torch
import reality_stone as rs

x = torch.randn(32, 64)  # 유클리드 벡터
x_lorentz = rs.project_to_lorentz(x, c=1.0)
print(x_lorentz.shape)  # [32, 65]
```

## 좌표 변환 함수들

### `lorentz_to_poincare()`

```python
def lorentz_to_poincare(x: torch.Tensor, c: float = 1.0) -> torch.Tensor
```

Lorentz 좌표를 Poincaré Ball 좌표로 변환합니다.

#### 수학적 정의

$$\phi^{-1}(x) = \frac{x_{1:n}}{x_0 + \sqrt{c}}$$

### `lorentz_to_klein()`

```python
def lorentz_to_klein(x: torch.Tensor, c: float = 1.0) -> torch.Tensor
```

Lorentz 좌표를 Klein 좌표로 변환합니다.

#### 수학적 정의

$$\psi(x) = \frac{x_{1:n}}{x_0}$$

## 성능 특성

### 시간 복잡도
- 모든 연산: $O(bd)$ (b: 배치 크기, d: 차원)

### 수치적 안정성
- Poincaré Ball보다 경계 근처에서 더 안정적
- 민코프스키 내적의 특성으로 인한 안정성
- arccosh 함수의 안정적인 구현 사용

### GPU 최적화
- 모든 연산이 CUDA로 가속됨
- 벡터화된 배치 처리 지원
- 메모리 효율적인 구현

## 실제 사용 예제

```python
import torch
import torch.nn as nn
import reality_stone as rs

class LorentzMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, c=1.0):
        super().__init__()
        self.c = c
        
        # 유클리드 → Lorentz 투영
        self.input_proj = lambda x: rs.project_to_lorentz(x, c=self.c)
        
        # Lorentz 공간에서의 변환
        self.transform = nn.Linear(input_dim + 1, hidden_dim + 1)
        
        # 출력 레이어
        self.output = nn.Linear(hidden_dim + 1, output_dim)
        
    def forward(self, x):
        # 입력을 Lorentz 공간으로 투영
        x_lorentz = self.input_proj(x)
        
        # 선형 변환
        h = self.transform(x_lorentz)
        
        # Lorentz 제약 조건 만족하도록 정규화
        h_normalized = rs.lorentz_normalize(h, c=self.c)
        
        # 출력
        output = self.output(h_normalized)
        
        return output

# 사용 예제
model = LorentzMLP(64, 128, 10, c=1.0)
x = torch.randn(32, 64)
output = model(x)
print(output.shape)  # [32, 10]
```

## 주의사항

### Lorentz 제약 조건
- 모든 점은 하이퍼볼로이드에 있어야 함: $\langle x,x \rangle_{\mathcal{L}} = -1/c$
- 시간 좌표는 양수여야 함: $x_0 > 0$

### 수치적 안정성
- arccosh 함수의 정의역 주의: 인수는 1 이상이어야 함
- 민코프스키 내적이 음수가 되도록 보장

### 차원 관리
- Lorentz 모델은 n차원 하이퍼볼릭 공간을 (n+1)차원에서 표현
- 메모리 사용량과 계산 복잡도 증가

## 참고 문헌

1. **Hyperbolic Neural Networks** - Ganea et al. (2018)
2. **Lorentzian Distance Learning for Hyperbolic Representations** - Law et al. (2019)
3. **Hyperbolic Graph Neural Networks** - Chami et al. (2019) 