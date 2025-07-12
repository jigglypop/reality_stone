# 유틸리티 함수들

Reality Stone에서 제공하는 공통 유틸리티 함수들입니다. 이 문서는 하이퍼볼릭 공간에서 자주 사용되는 보조 함수들을 설명합니다.

## 개요

유틸리티 함수들은 하이퍼볼릭 신경망 구현에서 자주 필요한 공통 기능들을 제공합니다. 수치적 안정성, 좌표 변환, 그래디언트 처리 등의 기능을 포함합니다.

## 수치적 안정성 함수들

### `safe_arccosh()`

```python
def safe_arccosh(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor
```

수치적으로 안정적인 arccosh 함수를 계산합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 입력 텐서 (x >= 1) | 필수 |
| `eps` | `float` | 수치적 안정성을 위한 작은 값 | `1e-8` |

#### 수학적 정의

$$\text{arccosh}(x) = \ln(x + \sqrt{x^2 - 1})$$

안정적인 구현:
$$\text{arccosh}(x) = \ln(x + \sqrt{\max(x^2 - 1, \epsilon)})$$

#### 예제

```python
import torch
import reality_stone as rs

x = torch.tensor([1.0, 1.5, 2.0, 10.0])
result = rs.safe_arccosh(x)
print(result)  # [0.0000, 0.9624, 1.3170, 2.9932]
```

### `safe_arctanh()`

```python
def safe_arctanh(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor
```

수치적으로 안정적인 arctanh 함수를 계산합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 입력 텐서 (-1 < x < 1) | 필수 |
| `eps` | `float` | 수치적 안정성을 위한 작은 값 | `1e-8` |

#### 수학적 정의

$$\text{arctanh}(x) = \frac{1}{2}\ln\left(\frac{1+x}{1-x}\right)$$

안정적인 구현:
$$\text{arctanh}(x) = \frac{1}{2}\ln\left(\frac{1+\text{clamp}(x, -1+\epsilon, 1-\epsilon)}{1-\text{clamp}(x, -1+\epsilon, 1-\epsilon)}\right)$$

#### 예제

```python
import torch
import reality_stone as rs

x = torch.tensor([-0.9, -0.5, 0.0, 0.5, 0.9])
result = rs.safe_arctanh(x)
print(result)  # [-1.4722, -0.5493, 0.0000, 0.5493, 1.4722]
```

### `clamp_norm()`

```python
def clamp_norm(x: torch.Tensor, max_norm: float = 0.99, dim: int = -1) -> torch.Tensor
```

벡터의 노름을 지정된 최대값으로 제한합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 입력 텐서 | 필수 |
| `max_norm` | `float` | 최대 노름 값 | `0.99` |
| `dim` | `int` | 노름을 계산할 차원 | `-1` |

#### 수학적 정의

$$\text{clamp\_norm}(x) = \begin{cases}
x & \text{if } \|x\| \leq \text{max\_norm} \\
\frac{x}{\|x\|} \cdot \text{max\_norm} & \text{if } \|x\| > \text{max\_norm}
\end{cases}$$

#### 예제

```python
import torch
import reality_stone as rs

x = torch.randn(32, 64)  # 임의의 벡터
x_clamped = rs.clamp_norm(x, max_norm=0.95)

# 모든 벡터의 노름이 0.95 이하인지 확인
norms = torch.norm(x_clamped, dim=-1)
print(torch.all(norms <= 0.95))  # True
```

## 좌표 변환 유틸리티

### `convert_coordinates()`

```python
def convert_coordinates(
    x: torch.Tensor, 
    from_model: str, 
    to_model: str, 
    c: float = 1.0
) -> torch.Tensor
```

하이퍼볼릭 모델 간 좌표 변환을 수행합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 입력 좌표 | 필수 |
| `from_model` | `str` | 원본 모델 ('poincare', 'lorentz', 'klein') | 필수 |
| `to_model` | `str` | 대상 모델 ('poincare', 'lorentz', 'klein') | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 지원하는 변환

- **poincare ↔ lorentz**
- **poincare ↔ klein**
- **lorentz ↔ klein**

#### 예제

```python
import torch
import reality_stone as rs

# Poincaré Ball 좌표
x_poincare = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

# Lorentz 좌표로 변환
x_lorentz = rs.convert_coordinates(x_poincare, 'poincare', 'lorentz', c=1.0)
print(x_lorentz.shape)  # [2, 3]

# Klein 좌표로 변환
x_klein = rs.convert_coordinates(x_poincare, 'poincare', 'klein', c=1.0)
print(x_klein.shape)  # [2, 2]
```

### `batch_convert()`

```python
def batch_convert(
    tensors: List[torch.Tensor], 
    from_model: str, 
    to_model: str, 
    c: float = 1.0
) -> List[torch.Tensor]
```

여러 텐서를 한 번에 좌표 변환합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `tensors` | `List[torch.Tensor]` | 입력 텐서들의 리스트 | 필수 |
| `from_model` | `str` | 원본 모델 | 필수 |
| `to_model` | `str` | 대상 모델 | 필수 |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 예제

```python
import torch
import reality_stone as rs

tensors = [
    torch.randn(32, 64) * 0.1,
    torch.randn(16, 64) * 0.1,
    torch.randn(8, 64) * 0.1
]

converted = rs.batch_convert(tensors, 'poincare', 'lorentz', c=1.0)
print(len(converted))  # 3
print(converted[0].shape)  # [32, 65]
```

## 그래디언트 처리 함수들

### `clip_hyperbolic_gradients()`

```python
def clip_hyperbolic_gradients(
    parameters: Iterable[torch.Tensor], 
    max_norm: float = 1.0, 
    model: str = 'poincare',
    c: float = 1.0
) -> float
```

하이퍼볼릭 공간에서 그래디언트를 클리핑합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `parameters` | `Iterable[torch.Tensor]` | 모델 매개변수들 | 필수 |
| `max_norm` | `float` | 최대 그래디언트 노름 | `1.0` |
| `model` | `str` | 하이퍼볼릭 모델 타입 | `'poincare'` |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 반환값

- **타입**: `float`
- **설명**: 클리핑 전 총 그래디언트 노름

#### 예제

```python
import torch
import torch.nn as nn
import reality_stone as rs

model = nn.Linear(64, 32)
optimizer = torch.optim.Adam(model.parameters())

# 훈련 루프에서
for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    # 하이퍼볼릭 그래디언트 클리핑
    total_norm = rs.clip_hyperbolic_gradients(
        model.parameters(), 
        max_norm=1.0, 
        model='poincare'
    )
    
    optimizer.step()
```

### `riemannian_gradient()`

```python
def riemannian_gradient(
    grad: torch.Tensor, 
    x: torch.Tensor, 
    model: str = 'poincare',
    c: float = 1.0
) -> torch.Tensor
```

유클리드 그래디언트를 리만 그래디언트로 변환합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `grad` | `torch.Tensor` | 유클리드 그래디언트 | 필수 |
| `x` | `torch.Tensor` | 현재 위치 | 필수 |
| `model` | `str` | 하이퍼볼릭 모델 타입 | `'poincare'` |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 수학적 정의

**Poincaré Ball**:
$$\nabla_{\mathcal{R}} f = \left(\frac{1-c\|x\|^2}{2}\right)^2 \nabla_{\mathcal{E}} f$$

**Lorentz**:
$$\nabla_{\mathcal{R}} f = \nabla_{\mathcal{E}} f + \langle \nabla_{\mathcal{E}} f, x \rangle_{\mathcal{L}} \cdot x$$

#### 예제

```python
import torch
import reality_stone as rs

x = torch.randn(32, 64, requires_grad=True)
loss = torch.sum(x**2)
loss.backward()

# 유클리드 그래디언트를 리만 그래디언트로 변환
riemannian_grad = rs.riemannian_gradient(
    x.grad, x, model='poincare', c=1.0
)
```

## 메트릭 및 거리 함수들

### `compute_distortion()`

```python
def compute_distortion(
    embeddings: torch.Tensor, 
    tree_distances: torch.Tensor, 
    model: str = 'poincare',
    c: float = 1.0
) -> Dict[str, float]
```

임베딩의 왜곡 정도를 계산합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `embeddings` | `torch.Tensor` | 임베딩 벡터들 | 필수 |
| `tree_distances` | `torch.Tensor` | 트리에서의 실제 거리들 | 필수 |
| `model` | `str` | 하이퍼볼릭 모델 타입 | `'poincare'` |
| `c` | `float` | 곡률 매개변수 | `1.0` |

#### 반환값

- **타입**: `Dict[str, float]`
- **내용**: 
  - `'mean_distortion'`: 평균 왜곡
  - `'worst_distortion'`: 최대 왜곡
  - `'map_score'`: Mean Average Precision 점수

#### 예제

```python
import torch
import reality_stone as rs

# 임베딩과 실제 거리
embeddings = torch.randn(100, 64) * 0.1
tree_distances = torch.randint(1, 10, (100, 100)).float()

distortion = rs.compute_distortion(
    embeddings, tree_distances, model='poincare', c=1.0
)
print(distortion)
# {'mean_distortion': 1.23, 'worst_distortion': 3.45, 'map_score': 0.78}
```

### `hyperbolic_centroid()`

```python
def hyperbolic_centroid(
    points: torch.Tensor, 
    weights: torch.Tensor = None,
    model: str = 'poincare',
    c: float = 1.0,
    max_iter: int = 100
) -> torch.Tensor
```

하이퍼볼릭 공간에서 점들의 중심을 계산합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `points` | `torch.Tensor` | 점들 `[num_points, dim]` | 필수 |
| `weights` | `torch.Tensor` | 가중치 `[num_points]` | `None` |
| `model` | `str` | 하이퍼볼릭 모델 타입 | `'poincare'` |
| `c` | `float` | 곡률 매개변수 | `1.0` |
| `max_iter` | `int` | 최대 반복 횟수 | `100` |

#### 예제

```python
import torch
import reality_stone as rs

points = torch.randn(10, 64) * 0.1
weights = torch.ones(10)

centroid = rs.hyperbolic_centroid(
    points, weights, model='poincare', c=1.0
)
print(centroid.shape)  # [64]
```

## 시각화 유틸리티

### `plot_poincare_disk()`

```python
def plot_poincare_disk(
    points: torch.Tensor, 
    labels: torch.Tensor = None,
    colors: List[str] = None,
    title: str = "Poincaré Disk",
    save_path: str = None
) -> None
```

2차원 Poincaré 디스크를 시각화합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `points` | `torch.Tensor` | 2차원 점들 `[num_points, 2]` | 필수 |
| `labels` | `torch.Tensor` | 점들의 라벨 | `None` |
| `colors` | `List[str]` | 색상 리스트 | `None` |
| `title` | `str` | 그래프 제목 | `"Poincaré Disk"` |
| `save_path` | `str` | 저장 경로 | `None` |

#### 예제

```python
import torch
import reality_stone as rs

# 2차원 점들 생성
points = torch.randn(100, 2) * 0.3
labels = torch.randint(0, 3, (100,))

# 시각화
rs.plot_poincare_disk(
    points, labels, 
    title="My Embeddings",
    save_path="poincare_plot.png"
)
```

## 성능 최적화 함수들

### `optimize_curvature()`

```python
def optimize_curvature(
    embeddings: torch.Tensor,
    tree_distances: torch.Tensor,
    model: str = 'poincare',
    c_init: float = 1.0,
    lr: float = 0.01,
    max_iter: int = 1000
) -> float
```

주어진 데이터에 대해 최적의 곡률을 찾습니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `embeddings` | `torch.Tensor` | 임베딩 벡터들 | 필수 |
| `tree_distances` | `torch.Tensor` | 실제 거리들 | 필수 |
| `model` | `str` | 하이퍼볼릭 모델 타입 | `'poincare'` |
| `c_init` | `float` | 초기 곡률 값 | `1.0` |
| `lr` | `float` | 학습률 | `0.01` |
| `max_iter` | `int` | 최대 반복 횟수 | `1000` |

#### 반환값

- **타입**: `float`
- **설명**: 최적화된 곡률 값

#### 예제

```python
import torch
import reality_stone as rs

embeddings = torch.randn(100, 64) * 0.1
tree_distances = torch.randint(1, 10, (100, 100)).float()

optimal_c = rs.optimize_curvature(
    embeddings, tree_distances, 
    model='poincare', c_init=1.0
)
print(f"Optimal curvature: {optimal_c}")
```

## 디버깅 및 검증 함수들

### `validate_hyperbolic_constraints()`

```python
def validate_hyperbolic_constraints(
    x: torch.Tensor,
    model: str = 'poincare',
    c: float = 1.0,
    tolerance: float = 1e-6
) -> Dict[str, bool]
```

하이퍼볼릭 제약 조건을 검증합니다.

#### 매개변수

| 매개변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `x` | `torch.Tensor` | 검증할 점들 | 필수 |
| `model` | `str` | 하이퍼볼릭 모델 타입 | `'poincare'` |
| `c` | `float` | 곡률 매개변수 | `1.0` |
| `tolerance` | `float` | 허용 오차 | `1e-6` |

#### 반환값

- **타입**: `Dict[str, bool]`
- **내용**: 각 제약 조건의 만족 여부

#### 예제

```python
import torch
import reality_stone as rs

x = torch.randn(32, 64) * 0.1
validation = rs.validate_hyperbolic_constraints(
    x, model='poincare', c=1.0
)
print(validation)
# {'norm_constraint': True, 'numerical_stability': True}
```

## 참고 문헌

1. **Hyperbolic Neural Networks** - Ganea et al. (2018)
2. **Poincaré Embeddings for Learning Hierarchical Representations** - Nickel & Kiela (2017)
3. **Hyperbolic Graph Neural Networks** - Chami et al. (2019)
4. **Optimization on Manifolds** - Absil et al. (2008) 