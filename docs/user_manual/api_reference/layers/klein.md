# Klein 레이어

Klein 레이어는 Klein 디스크 모델에서 동작하는 하이퍼볼릭 신경망 레이어입니다.

## 개요

Klein 모델은 단위 원판에서 직선 측지선을 갖는 하이퍼볼릭 공간의 모델입니다. 기하학적 직관이 뛰어나며 측지선이 유클리드 직선으로 표현되는 특징이 있습니다.

## 함수 API

### `klein_layer()`

```python
def klein_layer(u: torch.Tensor, v: torch.Tensor, c: float, t: float) -> torch.Tensor
```

두 텐서를 Klein 공간에서 결합합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `u` | `torch.Tensor` | 첫 번째 입력 텐서 `[batch_size, dim]` | 필수 |
| `v` | `torch.Tensor` | 두 번째 입력 텐서 `[batch_size, dim]` | 필수 |
| `c` | `float` | 곡률 매개변수 (양수) | 필수 |
| `t` | `float` | 보간 비율 (0~1) | 필수 |

#### 예제

```python
import torch
import reality_stone as rs

u = torch.randn(32, 64) * 0.1
v = torch.randn(32, 64) * 0.1
result = rs.klein_layer(u, v, c=1.0, t=0.5)
```

## 관련 연산 함수들

### `klein_add()`

```python
def klein_add(u: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor
```

Klein 공간에서의 덧셈 연산

### `klein_scalar_mul()`

```python
def klein_scalar_mul(x: torch.Tensor, r: float, c: float) -> torch.Tensor
```

Klein 공간에서의 스칼라 곱셈

### `klein_distance()`

```python
def klein_distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor
```

Klein 공간에서의 거리 계산

## 좌표 변환 함수들

### `klein_to_poincare()`

```python
def klein_to_poincare(x: torch.Tensor, c: float) -> torch.Tensor
```

Klein 모델에서 Poincaré Ball로 변환

### `klein_to_lorentz()`

```python
def klein_to_lorentz(x: torch.Tensor, c: float) -> torch.Tensor
```

Klein 모델에서 Lorentz 모델로 변환

## 수학적 배경

### Klein 모델

**정의**: 단위 원판 $\mathbb{D}^n = \{x \in \mathbb{R}^n : \|x\| < 1\}$에서 직선 측지선을 갖는 모델

**거리 함수**:
$$d_K(x,y) = \frac{1}{\sqrt{c}} \text{arccosh}\left(\frac{1-\langle x,y\rangle}{\sqrt{(1-\|x\|^2)(1-\|y\|^2)}}\right)$$

**주요 특징**:
- 측지선이 유클리드 직선
- 각도 보존하지 않음 (등각 변환 아님)
- 경계에서의 수치적 안정성

## 실제 사용 예제

```python
import torch
import torch.nn as nn
import reality_stone as rs

class KleinClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, curvature=1.0):
        super().__init__()
        self.curvature = curvature
        
        self.encoder1 = nn.Linear(input_dim, hidden_dim)
        self.encoder2 = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        u = torch.tanh(self.encoder1(x)) * 0.1
        v = torch.tanh(self.encoder2(x)) * 0.1
        
        # Klein 공간에서 결합
        klein_features = rs.klein_layer(u, v, c=self.curvature, t=0.5)
        
        return self.classifier(klein_features)
```

## 주의사항

### 경계 조건
- 모든 점은 단위 원판 내부에 있어야 함: $\|x\| < 1$
- 경계 근처에서 수치적 불안정성 가능

### 변환 정확도
- 다른 모델과의 변환 시 정밀도 손실 가능
- 중요한 계산에서는 정밀도 확인 필요

## 참고 문헌

1. **Hyperbolic Neural Networks** - Ganea et al. (2018)
2. **Hyperbolic Graph Neural Networks** - Chami et al. (2019) 