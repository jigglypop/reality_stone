# Klein 연산

Klein 모델에서 사용되는 핵심 연산들입니다. 이 문서는 Reality Stone에서 제공하는 Klein 연산 함수들을 상세히 설명합니다.

## 개요

Klein 모델은 단위 원판에서 직선 측지선을 갖는 하이퍼볼릭 공간의 모델입니다. 기하학적 직관이 뛰어나며 측지선이 유클리드 직선으로 표현되는 특징이 있습니다.

## 핵심 연산 함수들

### `klein_add()`

```python
def klein_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor
```

두 점의 Klein 덧셈을 계산합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 첫 번째 점 `[batch_size, dim]` | 필수 |
| `y` | `torch.Tensor` | 두 번째 점 `[batch_size, dim]` | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 수학적 정의

Klein 모델에서의 덧셈은 Poincaré Ball을 통한 변환으로 수행됩니다:

1. Klein → Poincaré: $\psi^{-1}(x) = \frac{x}{1+\sqrt{1-\|x\|^2}}$
2. Poincaré 덧셈: $u \oplus_c v$
3. Poincaré → Klein: $\psi(z) = \frac{2z}{1+\|z\|^2}$

#### 예제

```python
import torch
import reality_stone as rs

x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
y = torch.tensor([[0.2, 0.1], [0.1, 0.3]])

result = rs.klein_add(x, y, c=1.0)
print(result.shape)  # [2, 2]
```

### `klein_scalar_mul()`

```python
def klein_scalar_mul(x: torch.Tensor, r: float, c: float = 1.0) -> torch.Tensor
```

점의 Klein 스칼라 곱셈을 계산합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 입력 점 `[batch_size, dim]` | 필수 |
| `r` | `float` | 스칼라 값 | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 수학적 정의

Klein 모델에서의 스칼라 곱셈도 Poincaré Ball을 통해 수행됩니다:

1. Klein → Poincaré 변환
2. Poincaré 스칼라 곱셈
3. Poincaré → Klein 변환

#### 예제

```python
import torch
import reality_stone as rs

x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
r = 0.5

result = rs.klein_scalar_mul(x, r, c=1.0)
print(result.shape)  # [2, 2]
```

### `klein_distance()`

```python
def klein_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor
```

두 점 사이의 하이퍼볼릭 거리를 계산합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 첫 번째 점 `[batch_size, dim]` | 필수 |
| `y` | `torch.Tensor` | 두 번째 점 `[batch_size, dim]` | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 수학적 정의

$$d_K(x,y) = \frac{1}{\sqrt{c}} \text{arccosh}\left(\frac{1-\langle x,y\rangle}{\sqrt{(1-\|x\|^2)(1-\|y\|^2)}}\right)$$

#### 예제

```python
import torch
import reality_stone as rs

x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
y = torch.tensor([[0.2, 0.1], [0.1, 0.3]])

distances = rs.klein_distance(x, y, c=1.0)
print(distances.shape)  # [2]
```

### `klein_norm()`

```python
def klein_norm(x: torch.Tensor, c: float = 1.0) -> torch.Tensor
```

Klein 공간에서의 노름을 계산합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 입력 점 `[batch_size, dim]` | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 수학적 정의

$$\|x\|_K = \frac{1}{\sqrt{c}} \text{arccosh}\left(\frac{1}{\sqrt{1-\|x\|^2}}\right)$$

#### 예제

```python
import torch
import reality_stone as rs

x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

norms = rs.klein_norm(x, c=1.0)
print(norms.shape)  # [2]
```

## 좌표 변환 함수들

### `klein_to_poincare()`

```python
def klein_to_poincare(x: torch.Tensor, c: float = 1.0) -> torch.Tensor
```

Klein 좌표를 Poincaré Ball 좌표로 변환합니다.

#### 수학적 정의

$$\psi^{-1}(x) = \frac{x}{1+\sqrt{1-\|x\|^2}}$$

#### 예제

```python
import torch
import reality_stone as rs

x_klein = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
x_poincare = rs.klein_to_poincare(x_klein, c=1.0)
print(x_poincare.shape)  # [2, 2]
```

### `klein_to_lorentz()`

```python
def klein_to_lorentz(x: torch.Tensor, c: float = 1.0) -> torch.Tensor
```

Klein 좌표를 Lorentz 좌표로 변환합니다.

#### 수학적 정의

$$\phi(x) = \frac{1}{\sqrt{c}} \left(\frac{1}{\sqrt{1-\|x\|^2}}, \frac{x}{\sqrt{1-\|x\|^2}}\right)$$

#### 예제

```python
import torch
import reality_stone as rs

x_klein = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
x_lorentz = rs.klein_to_lorentz(x_klein, c=1.0)
print(x_lorentz.shape)  # [2, 3]
```

### `poincare_to_klein()`

```python
def poincare_to_klein(x: torch.Tensor, c: float = 1.0) -> torch.Tensor
```

Poincaré Ball 좌표를 Klein 좌표로 변환합니다.

#### 수학적 정의

$$\psi(x) = \frac{2x}{1+\|x\|^2}$$

### `lorentz_to_klein()`

```python
def lorentz_to_klein(x: torch.Tensor, c: float = 1.0) -> torch.Tensor
```

Lorentz 좌표를 Klein 좌표로 변환합니다.

#### 수학적 정의

$$\psi(x) = \frac{x_{1:n}}{x_0}$$

## 유틸리티 함수들

### `klein_project()`

```python
def klein_project(x: torch.Tensor, c: float = 1.0) -> torch.Tensor
```

벡터를 Klein 디스크 내부로 투영합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 입력 벡터 `[batch_size, dim]` | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 수학적 정의

$$\text{proj}(x) = \begin{cases}
x & \text{if } \|x\| < 1 \\
\frac{x}{\|x\|} \cdot (1 - \epsilon) & \text{if } \|x\| \geq 1
\end{cases}$$

여기서 $\epsilon$은 작은 양수입니다.

#### 예제

```python
import torch
import reality_stone as rs

x = torch.randn(32, 64)  # 임의의 벡터
x_projected = rs.klein_project(x, c=1.0)

# 모든 점이 단위 원 내부에 있는지 확인
norms = torch.norm(x_projected, dim=-1)
print(torch.all(norms < 1.0))  # True
```

### `klein_geodesic()`

```python
def klein_geodesic(x: torch.Tensor, y: torch.Tensor, t: float, c: float = 1.0) -> torch.Tensor
```

두 점 사이의 측지선 상의 점을 계산합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 시작점 `[batch_size, dim]` | 필수 |
| `y` | `torch.Tensor` | 끝점 `[batch_size, dim]` | 필수 |
| `t` | `float` | 매개변수 (0~1) | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 수학적 정의

Klein 모델에서 측지선은 유클리드 직선이므로:

$$\gamma(t) = (1-t)x + ty$$

단, 결과가 단위 원 내부에 있어야 합니다.

#### 예제

```python
import torch
import reality_stone as rs

x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
y = torch.tensor([[0.2, 0.1], [0.1, 0.3]])
t = 0.5

midpoint = rs.klein_geodesic(x, y, t, c=1.0)
print(midpoint.shape)  # [2, 2]
```

## 성능 특성

### 시간 복잡도
- 대부분의 연산: $O(bd)$ (b: 배치 크기, d: 차원)
- 좌표 변환: $O(bd)$

### 수치적 안정성
- 경계 근처 ($\|x\| \to 1$)에서 주의 필요
- 직선 측지선으로 인한 기하학적 직관성
- 각도 보존하지 않음

### GPU 최적화
- 모든 연산이 CUDA로 가속됨
- 벡터화된 배치 처리 지원
- 메모리 효율적인 구현

## 실제 사용 예제

```python
import torch
import torch.nn as nn
import reality_stone as rs

class KleinAttention(nn.Module):
    def __init__(self, dim, num_heads=8, c=1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.c = c
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, D = x.shape
        
        # QKV 계산
        q = self.query(x).reshape(B, N, self.num_heads, D // self.num_heads)
        k = self.key(x).reshape(B, N, self.num_heads, D // self.num_heads)
        v = self.value(x).reshape(B, N, self.num_heads, D // self.num_heads)
        
        # Klein 공간으로 투영
        q = rs.klein_project(q, c=self.c)
        k = rs.klein_project(k, c=self.c)
        v = rs.klein_project(v, c=self.c)
        
        # 하이퍼볼릭 거리 기반 어텐션
        attn_weights = []
        for i in range(N):
            # 각 쿼리와 모든 키 사이의 거리 계산
            distances = rs.klein_distance(
                q[:, i:i+1].expand(-1, N, -1, -1),
                k, 
                c=self.c
            )
            # 거리를 유사도로 변환 (거리가 가까울수록 높은 가중치)
            weights = torch.exp(-distances)
            attn_weights.append(weights)
        
        attn_weights = torch.stack(attn_weights, dim=1)
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 가중 합계 계산
        output = torch.einsum('bnhm,bmhd->bnhd', attn_weights, v)
        output = output.reshape(B, N, D)
        
        return output

# 사용 예제
model = KleinAttention(512, num_heads=8, c=1.0)
x = torch.randn(32, 100, 512)
output = model(x)
print(output.shape)  # [32, 100, 512]
```

## 주의사항

### 정의역 제약
- 모든 점은 단위 원 내부에 있어야 함: $\|x\| < 1$
- 경계 근처에서 수치적 불안정성 발생 가능

### 각도 보존
- Klein 모델은 등각 변환이 아님
- 각도가 보존되지 않으므로 주의 필요

### 측지선 특성
- 측지선이 유클리드 직선이므로 직관적
- 하지만 거리 계산은 비유클리드적

### 변환 정확도
- 다른 모델과의 변환 시 정밀도 손실 가능
- 중요한 계산에서는 정밀도 확인 필요

## 참고 문헌

1. **Hyperbolic Neural Networks** - Ganea et al. (2018)
2. **Hyperbolic Graph Neural Networks** - Chami et al. (2019)
3. **Hyperbolic Geometry and Poincaré Embeddings** - Nickel & Kiela (2017) 