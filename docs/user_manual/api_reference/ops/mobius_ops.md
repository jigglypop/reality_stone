# Mobius 연산

Mobius 변환은 Poincaré Ball 모델에서 사용되는 핵심 연산들입니다. 이 문서는 Reality Stone에서 제공하는 Mobius 연산 함수들을 상세히 설명합니다.

## 개요

Mobius 변환은 하이퍼볼릭 공간에서 거리와 각도를 보존하는 등거리 변환입니다. Poincaré Ball 모델에서 모든 기본 연산은 Mobius 변환을 통해 구현됩니다.

## 핵심 연산 함수들

### `mobius_add()`

```python
def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor
```

두 점의 Mobius 덧셈을 계산합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 첫 번째 점 `[batch_size, dim]` | 필수 |
| `y` | `torch.Tensor` | 두 번째 점 `[batch_size, dim]` | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 수학적 정의

$$x \oplus_c y = \frac{(1+2c\langle x,y\rangle + c\|y\|^2)x + (1-c\|x\|^2)y}{1+2c\langle x,y\rangle + c^2\|x\|^2\|y\|^2}$$

#### 예제

```python
import torch
import reality_stone as rs

x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
y = torch.tensor([[0.2, 0.1], [0.1, 0.3]])

result = rs.mobius_add(x, y, c=1.0)
print(result.shape)  # [2, 2]
```

### `mobius_scalar_mul()`

```python
def mobius_scalar_mul(x: torch.Tensor, r: float, c: float = 1.0) -> torch.Tensor
```

점의 Mobius 스칼라 곱셈을 계산합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 입력 점 `[batch_size, dim]` | 필수 |
| `r` | `float` | 스칼라 값 | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 수학적 정의

$$r \otimes_c x = \frac{1}{\sqrt{c}} \tanh\left(r \tanh^{-1}(\sqrt{c}\|x\|)\right) \frac{x}{\|x\|}$$

#### 예제

```python
import torch
import reality_stone as rs

x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
r = 0.5

result = rs.mobius_scalar_mul(x, r, c=1.0)
print(result.shape)  # [2, 2]
```

### `mobius_matvec()`

```python
def mobius_matvec(m: torch.Tensor, x: torch.Tensor, c: float = 1.0) -> torch.Tensor
```

행렬-벡터 Mobius 곱셈을 계산합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `m` | `torch.Tensor` | 변환 행렬 `[dim, dim]` | 필수 |
| `x` | `torch.Tensor` | 입력 점 `[batch_size, dim]` | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 예제

```python
import torch
import reality_stone as rs

m = torch.eye(2) * 0.5  # 스케일링 행렬
x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

result = rs.mobius_matvec(m, x, c=1.0)
print(result.shape)  # [2, 2]
```

### `mobius_distance()`

```python
def mobius_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor
```

두 점 사이의 하이퍼볼릭 거리를 계산합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 첫 번째 점 `[batch_size, dim]` | 필수 |
| `y` | `torch.Tensor` | 두 번째 점 `[batch_size, dim]` | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 수학적 정의

$$d(x,y) = \frac{2}{\sqrt{c}} \tanh^{-1}(\sqrt{c}\|(-x) \oplus_c y\|)$$

#### 예제

```python
import torch
import reality_stone as rs

x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
y = torch.tensor([[0.2, 0.1], [0.1, 0.3]])

distances = rs.mobius_distance(x, y, c=1.0)
print(distances.shape)  # [2]
```

## 고급 연산 함수들

### `expmap()`

```python
def expmap(x: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor
```

점 x에서 접선 벡터 v의 지수 매핑을 계산합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 기준점 `[batch_size, dim]` | 필수 |
| `v` | `torch.Tensor` | 접선 벡터 `[batch_size, dim]` | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 수학적 정의

$$\text{Exp}_x^c(v) = x \oplus_c \left(\tanh\left(\sqrt{c}\frac{\lambda_x^c \|v\|}{2}\right) \frac{v}{\sqrt{c}\|v\|}\right)$$

여기서 $\lambda_x^c = \frac{2}{1-c\|x\|^2}$는 공형 인수입니다.

### `logmap()`

```python
def logmap(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor
```

점 x에서 점 y로의 로그 매핑을 계산합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 기준점 `[batch_size, dim]` | 필수 |
| `y` | `torch.Tensor` | 대상점 `[batch_size, dim]` | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 수학적 정의

$$\text{Log}_x^c(y) = \frac{2}{\sqrt{c}\lambda_x^c} \tanh^{-1}(\sqrt{c}\|(-x) \oplus_c y\|) \frac{(-x) \oplus_c y}{\|(-x) \oplus_c y\|}$$

## 성능 특성

### 시간 복잡도
- 모든 연산: $O(bd)$ (b: 배치 크기, d: 차원)

### 수치적 안정성
- 경계 근처 ($\|x\| \to 1$)에서 주의 필요
- 안전한 클리핑 사용 권장

### GPU 최적화
- 모든 연산이 CUDA로 가속됨
- 벡터화된 배치 처리 지원

## 실제 사용 예제

```python
import torch
import reality_stone as rs

# 하이퍼볼릭 공간에서의 선형 변환
class MobiusLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, c=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))
        
    def forward(self, x):
        # 행렬-벡터 곱셈
        mv = rs.mobius_matvec(self.weight, x, c=self.c)
        
        # 편향 추가
        output = rs.mobius_add(mv, self.bias, c=self.c)
        
        return output

# 사용 예제
layer = MobiusLinear(64, 32, c=1.0)
x = torch.randn(16, 64) * 0.1
output = layer(x)
print(output.shape)  # [16, 32]
```

## 주의사항

### 정의역 제약
- 모든 점은 단위 원 내부에 있어야 함: $\|x\| < 1$
- 경계 근처에서 수치적 불안정성 발생 가능

### 그래디언트 폭발
- 하이퍼볼릭 공간에서 그래디언트가 폭발할 수 있음
- 그래디언트 클리핑 사용 권장

### 곡률 매개변수
- 너무 큰 곡률 값은 수치적 문제 야기
- 일반적으로 $c \in [1e-3, 1e-1]$ 범위 사용

## 참고 문헌

1. **Hyperbolic Neural Networks** - Ganea et al. (2018)
2. **Poincaré Embeddings for Learning Hierarchical Representations** - Nickel & Kiela (2017)
3. **Hyperbolic Geometry of Complex Networks** - Krioukov et al. (2010) 