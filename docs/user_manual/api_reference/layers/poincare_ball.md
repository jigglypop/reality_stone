# Poincaré Ball 레이어

Poincaré Ball 레이어는 포인카레 디스크 모델에서 동작하는 하이퍼볼릭 신경망의 핵심 구성요소입니다.

## 개요

Poincaré Ball 모델은 단위 원판 $\mathbb{D}^n = \{x \in \mathbb{R}^n : \|x\| < 1\}$에서 정의되는 하이퍼볼릭 공간의 등각 모델입니다. 이 레이어는 두 입력 벡터를 하이퍼볼릭 공간에서 결합하여 계층적 표현을 학습합니다.

## 함수 API

### `poincare_ball_layer()`

```python
def poincare_ball_layer(
    u: torch.Tensor, 
    v: torch.Tensor, 
    c: float = None, 
    t: float = 0.5, 
    kappas: torch.Tensor = None, 
    layer_idx: int = None, 
    c_min: float = -2.0, 
    c_max: float = -0.1
) -> torch.Tensor
```

두 텐서를 Poincaré Ball 공간에서 결합합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `u` | `torch.Tensor` | 첫 번째 입력 텐서 `[batch_size, dim]` | 필수 |
| `v` | `torch.Tensor` | 두 번째 입력 텐서 `[batch_size, dim]` | 필수 |
| `c` | `float` | 곡률 매개변수 (양수) | `None` |
| `t` | `float` | 보간 비율 (0~1) | `0.5` |
| `kappas` | `torch.Tensor` | 동적 곡률 매개변수 | `None` |
| `layer_idx` | `int` | 레이어 인덱스 (동적 곡률 사용 시) | `None` |
| `c_min` | `float` | 최소 곡률 값 | `-2.0` |
| `c_max` | `float` | 최대 곡률 값 | `-0.1` |

#### 반환값

- **타입**: `torch.Tensor`
- **형태**: `[batch_size, dim]` (입력과 동일)
- **설명**: 하이퍼볼릭 공간에서 결합된 결과

#### 예제

```python
import torch
import reality_stone as rs

# 기본 사용법
u = torch.randn(32, 64) * 0.1
v = torch.randn(32, 64) * 0.1
result = rs.poincare_ball_layer(u, v, c=1e-3, t=0.5)

# 동적 곡률 사용
kappas = torch.tensor([0.1, 0.2, 0.3])  # 각 레이어별 곡률
result = rs.poincare_ball_layer(u, v, kappas=kappas, layer_idx=0, t=0.5)
```

## 클래스 API

### `PoincareBallLayer`

```python
class PoincareBallLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v, c, t, kappas, layer_idx, c_min, c_max)
    
    @staticmethod
    def backward(ctx, grad_output)
```

PyTorch의 `autograd.Function`을 상속한 클래스로, 자동 미분을 지원합니다.

#### 사용법

```python
# 직접 호출 (권장하지 않음)
result = rs.PoincareBallLayer.apply(u, v, c, t, kappas, layer_idx, c_min, c_max)

# 래퍼 함수 사용 (권장)
result = rs.poincare_ball_layer(u, v, c=c, t=t)
```

## 관련 연산 함수들

### `poincare_add()`

```python
def poincare_add(x: torch.Tensor, y: torch.Tensor, c: float = None) -> torch.Tensor
```

Poincaré Ball에서의 덧셈 연산 (Mobius 덧셈)

#### 예제

```python
x = torch.randn(32, 64) * 0.1
y = torch.randn(32, 64) * 0.1
result = rs.poincare_add(x, y, c=1.0)
```

### `poincare_scalar_mul()`

```python
def poincare_scalar_mul(x: torch.Tensor, r: float, c: float) -> torch.Tensor
```

Poincaré Ball에서의 스칼라 곱셈

#### 예제

```python
x = torch.randn(32, 64) * 0.1
result = rs.poincare_scalar_mul(x, r=0.5, c=1.0)
```

### `poincare_distance()`

```python
def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor
```

Poincaré Ball에서의 거리 계산

#### 예제

```python
x = torch.randn(32, 64) * 0.1
y = torch.randn(32, 64) * 0.1
distances = rs.poincare_distance(x, y, c=1.0)
print(distances.shape)  # [32]
```

## 좌표 변환 함수들

### `poincare_to_lorentz()`

```python
def poincare_to_lorentz(x: torch.Tensor, c: float) -> torch.Tensor
```

Poincaré Ball에서 Lorentz 모델로 변환

### `poincare_to_klein()`

```python
def poincare_to_klein(x: torch.Tensor, c: float) -> torch.Tensor
```

Poincaré Ball에서 Klein 모델로 변환

## 수학적 배경

### Poincaré Ball 모델

**정의**: $\mathbb{D}^n = \{x \in \mathbb{R}^n : \|x\| < 1\}$

**Riemannian 계량**:
$$g_x = \frac{4}{(1-\|x\|^2)^2} \cdot I_n$$

**거리 함수**:
$$d_{\mathbb{D}}(x,y) = \text{arccosh}\left(1 + \frac{2\|x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)}\right)$$

### Mobius 변환

Poincaré Ball에서의 핵심 연산은 Mobius 변환을 통해 구현됩니다:

**Mobius 덧셈**:
$$x \oplus_c y = \frac{(1+2c\langle x,y\rangle + c\|y\|^2)x + (1-c\|x\|^2)y}{1+2c\langle x,y\rangle + c^2\|x\|^2\|y\|^2}$$

**Mobius 스칼라 곱셈**:
$$r \otimes_c x = \frac{1}{\sqrt{c}} \tanh\left(r \tanh^{-1}(\sqrt{c}\|x\|)\right) \frac{x}{\|x\|}$$

### 레이어 연산

Poincaré Ball 레이어는 다음 연산을 수행합니다:

1. **스칼라 곱셈**: $u' = (1-t) \otimes_c u$, $v' = t \otimes_c v$
2. **Mobius 덧셈**: $\text{output} = u' \oplus_c v'$

## 성능 특성

### 시간 복잡도
- **Forward**: $O(bd)$ (b: 배치 크기, d: 차원)
- **Backward**: $O(bd)$

### 공간 복잡도
- **중간 텐서**: $O(bd)$

### GPU 가속
- CUDA 구현으로 대규모 배치 처리 최적화
- 자동 CPU/GPU 선택

## 실제 사용 예제

### 1. 기본 분류 모델

```python
import torch
import torch.nn as nn
import reality_stone as rs

class HyperbolicClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, curvature=1e-3):
        super().__init__()
        self.curvature = curvature
        
        self.encoder1 = nn.Linear(input_dim, hidden_dim)
        self.encoder2 = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # 두 개의 다른 인코딩 경로
        u = torch.tanh(self.encoder1(x))
        v = torch.tanh(self.encoder2(x))
        
        # 하이퍼볼릭 공간에서 결합
        h = rs.poincare_ball_layer(u, v, c=self.curvature, t=0.5)
        
        # 분류
        return self.classifier(h)

# 사용 예제
model = HyperbolicClassifier(784, 128, 10)
x = torch.randn(32, 784)
output = model(x)
```

### 2. 동적 곡률 사용

```python
class AdaptiveCurvatureModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        # 각 레이어별 곡률 매개변수
        self.kappas = nn.Parameter(torch.zeros(num_layers))
        
        self.encoders = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        features = []
        
        # 각 레이어별 인코딩
        for i, encoder in enumerate(self.encoders):
            features.append(torch.tanh(encoder(x)))
        
        # 하이퍼볼릭 공간에서 순차적 결합
        result = features[0]
        for i in range(1, self.num_layers):
            result = rs.poincare_ball_layer(
                result, features[i], 
                kappas=self.kappas, 
                layer_idx=i-1,
                t=0.5
            )
        
        return result
```

## 🚨 주의사항

### 수치적 안정성
- 입력 텐서의 norm이 1에 가까우면 수치적 불안정성 발생
- 입력을 적절히 스케일링: `x = x * 0.1`

### 그래디언트 폭발
- 하이퍼볼릭 공간에서 그래디언트가 폭발할 수 있음
- 그래디언트 클리핑 사용: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`

### 곡률 매개변수 선택
- 너무 큰 곡률 값은 수치적 불안정성 야기
- 권장 범위: `1e-4 ~ 1e-1`

## 📚 참고 문헌

1. **Hyperbolic Neural Networks** - Ganea et al. (2018)
2. **Poincaré Embeddings for Learning Hierarchical Representations** - Nickel & Kiela (2017)
3. **Hyperbolic Graph Neural Networks** - Chami et al. (2019) 