# 통합 리만 레이어 아키텍처 (Unified Riemannian Architecture)

## 1. 개요

`UnifiedRiemannianLayer`는 Reality Stone의 핵심 통합 레이어로, 다음 요소들을 하나의 일관된 프레임워크로 결합합니다:

- 4가지 리만 메트릭 모델 (푸앵카레, 로렌츠, 클라인, 대각)
- 벨만 가치 함수 (강화학습 기반 목표 지향성)
- 라그랑지안 에너지 최적화
- 측지선 흐름 (기하학적 최적 경로)

## 2. 아키텍처 구조

### 2.1 계층 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│          UnifiedRiemannianLayer (통합 레이어)               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Metric    │  │  Bellman     │  │  Lagrangian      │   │
│  │   Selector  │  │  Value Fn    │  │  Energy System   │   │
│  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘   │
│         │                │                    │              │
│         ▼                ▼                    ▼              │
│  ┌───────────────────────────────────────────────────┐      │
│  │         Geodesic Flow Engine                      │      │
│  │  - Exponential Map                                 │      │
│  │  - Logarithmic Map                                 │      │
│  │  - Christoffel Symbols                             │      │
│  │  - Parallel Transport                              │      │
│  └───────────────────────────────────────────────────┘      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 데이터 흐름

```
Input (x)
    │
    ├─> Metric Selection ──> g_ij(x)
    │                          │
    ├─> Value Function ────> V(x), ∇V(x)
    │                          │
    └──────────┬──────────────┘
               │
               ▼
        Geodesic Flow
     x' = Exp_x(-η g^{-1} ∇V)
               │
               ├─> Energy Computation
               │     - Kinetic: T = ½ g_ij v^i v^j
               │     - Potential: V_Bell = (V(x) - R - γV(x'))²
               │     - Lagrangian: L = T - V
               │
               ▼
          Output (x')
```

## 3. 핵심 구성 요소

### 3.1 메트릭 선택기 (Metric Selector)

**역할**: 입력 데이터와 작업 유형에 따라 적절한 리만 메트릭 선택

**지원 메트릭**:
```rust
pub enum MetricType {
    Diagonal(DiagonalMetric),     // 학습 가능
    Poincare(PoincareMetric),     // 계층 데이터
    Lorentz(LorentzMetric),       // 수치 안정성
    Klein(KleinMetric),           // 시각화
}
```

**선택 기준**:
- 데이터 구조 (계층적 vs 평탄)
- 계산 효율성
- 수치 안정성 요구사항

### 3.2 벨만 가치 함수 (Bellman Value Function)

**수학적 정의**:

$$
V(x) = \max_a \left[ R(x,a) + \gamma \mathbb{E}_{x'}\big[V(x')\big] \right]
$$

**구현**:
```rust
pub struct ValueFunction {
    pub weights: Array2<f32>,  // MLP weights
    pub bias: Array1<f32>,
}

impl ValueFunction {
    fn compute(&self, x: &ArrayView2<f32>) -> Array1<f32>;
    fn gradient(&self, x: &ArrayView2<f32>) -> Array2<f32>;
}
```

**역할**:
- 각 상태의 "바람직함" 평가
- 에너지 지형(energy landscape) 생성
- 흐름 방향 결정

### 3.3 라그랑지안 에너지 시스템 (Lagrangian Energy System)

**통합 라그랑지안**:

$$
L(x, \dot{x}) = T(x, \dot{x}) - V(x)
$$

여기서:
- 운동 에너지: $T = \frac{1}{2} g_{ij}(x) \dot{x}^i \dot{x}^j$
- 잠재 에너지: $V = V_{\text{Bell}}(x) + V_{\text{reg}}(x)$

**벨만 잠재 에너지**:

$$
V_{\text{Bell}}(x) = \left( V(x) - \big(R + \gamma V(x')\big) \right)^2
$$

**정규화**:
- Attractor 항: $\beta \sum_k d(x, m_k)^2$
- 곡률 복잡도: $\gamma \|\text{Ric}(g)\|^2$

### 3.4 측지선 흐름 엔진 (Geodesic Flow Engine)

**핵심 연산**:

1. **Exponential Map**: $\text{Exp}_x(v)$
   - 점 $x$에서 tangent vector $v$ 방향으로 이동
   - 측지선 방정식 수치 적분

2. **Logarithmic Map**: $\text{Log}_x(y)$
   - 두 점을 연결하는 tangent vector 계산
   - 역문제 (Newton 방법)

3. **Parallel Transport**
   - Tangent vector를 측지선을 따라 이동
   - 접속(connection) 보존

## 4. 순전파 프로세스 (Forward Pass)

### 4.1 전체 흐름

```python
def forward(x, target=None):
    # 1. 메트릭 계산
    g = metric.compute_metric(x)
    
    # 2. 출력 계산
    if target is not None:
        # 목표 지향: 측지선 보간
        output = geodesic_interpolation(x, target, t=0.5)
    elif enable_bellman:
        # 가치 함수 기반 흐름
        grad_V = value_fn.gradient(x)
        riemannian_grad = g_inv * grad_V
        output = exponential_map(x, -η * riemannian_grad)
    else:
        # 항등 (메트릭만 적용)
        output = x
    
    # 3. 에너지 계산
    if enable_bellman:
        v = (output - x) / dt
        energy = compute_energy(x, v, output, reward)
    
    return output, energy
```

### 4.2 상세 단계

**Step 1: 메트릭 계산**
```rust
let g = metric.compute_metric(&x);  // (batch, dim)
```

**Step 2: 가치 함수 평가 (벨만 모드)**
```rust
let V_x = value_fn.compute(&x);        // V(x)
let grad_V = value_fn.gradient(&x);    // ∇V(x)
```

**Step 3: 리만 그래디언트**
```rust
let g_inv = metric.compute_inverse_metric(&x);
let riemannian_grad = grad_V * g_inv;  // g^{-1} ∇V
```

**Step 4: 측지선 이동**
```rust
let direction = -learning_rate * riemannian_grad;
let output = exponential_map(&metric, &x, &direction, 1.0);
```

**Step 5: 에너지 계산**
```rust
let velocity = (output - x) / dt;
let kinetic = 0.5 * (g * velocity.mapv(|v| v*v)).sum();
let potential = bellman_potential(&value_fn, &x, &output, &reward, gamma);
let lagrangian = kinetic - potential;
```

## 5. 역전파 (Backward Pass)

### 5.1 그래디언트 체인

출력 $y = f(x; \theta)$에 대한 손실 $L$의 그래디언트:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x}
$$

$$
\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial \theta}
$$

### 5.2 파라미터 그래디언트

**메트릭 파라미터** ($w_i$ for Diagonal):

$$
\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial g_{ii}} \frac{\partial g_{ii}}{\partial w_i}
$$

**가치 함수 파라미터**:

$$
\frac{\partial L}{\partial \theta_V} = \frac{\partial L}{\partial V} \frac{\partial V}{\partial \theta_V}
$$

### 5.3 구현

```rust
pub fn backward(
    &self,
    grad_output: &ArrayView2<f32>,
    x: &ArrayView2<f32>,
    cache: &LayerCache,
) -> LayerGradients {
    // 입력 그래디언트
    let grad_input = compute_input_gradient(grad_output, x, cache);
    
    // 메트릭 그래디언트 (학습 가능한 경우)
    let grad_metric = if self.enable_metric_learning {
        Some(compute_metric_gradient(grad_output, x, cache))
    } else {
        None
    };
    
    // 가치 함수 그래디언트
    let grad_value_fn = if self.enable_bellman {
        Some(compute_value_fn_gradient(grad_output, x, cache))
    } else {
        None
    };
    
    LayerGradients {
        grad_input,
        grad_metric,
        grad_value_fn,
    }
}
```

## 6. 학습 알고리즘

### 6.1 표현 학습 (Representation Learning)

**목표**: 입력 공간에서 최적의 표현 학습

**알고리즘**:
```
for epoch in epochs:
    for batch in data:
        # 순전파
        output, energy = layer.forward(batch.x, batch.target)
        
        # 손실 계산
        loss = task_loss(output, batch.y) - λ * energy.lagrangian
        
        # 역전파
        grad = layer.backward(loss_grad, batch.x)
        
        # 업데이트
        optimizer.step(grad)
```

### 6.2 메트릭 학습 (Metric Learning)

**목표**: 데이터에 적응적인 메트릭 학습 (Diagonal만)

**업데이트 규칙**:

$$
w_i \leftarrow w_i + \eta \frac{\partial L}{\partial w_i}
$$

여기서:

$$
\frac{\partial L}{\partial w_i} \approx \frac{1}{2} v_i^2 \cdot L
$$

### 6.3 가치 함수 학습 (Value Function Learning)

**TD(0) 업데이트**:

$$
V(x) \leftarrow V(x) + \alpha \left[ R + \gamma V(x') - V(x) \right]
$$

**구현**:
```rust
pub fn update_value_function(
    &mut self,
    x: &ArrayView2<f32>,
    x_next: &ArrayView2<f32>,
    reward: &ArrayView1<f32>,
    learning_rate: f32,
) {
    bellman_update(
        self.value_fn.as_mut().unwrap(),
        x,
        x_next,
        reward,
        self.lagrangian_params.gamma,
        learning_rate,
    );
}
```

## 7. 사용 예시

### 7.1 계층적 분류

```python
import reality_stone as rs

# 푸앵카레 메트릭으로 계층 표현
layer = rs.UnifiedRiemannianLayer(
    metric_type="poincare",
    curvature=1.0,
    input_dim=768,
    enable_bellman=False
)

# 임베딩 학습
embeddings = layer.forward(text_features)
```

### 7.2 강화학습

```python
# 벨만 가치 함수 활성화
layer = rs.UnifiedRiemannianLayer(
    metric_type="diagonal",
    curvature=0.0,
    input_dim=128,
    enable_bellman=True,
    gamma=0.99
)

# 에피소드 학습
for state, action, reward, next_state in trajectory:
    # 순전파
    output, energy = layer.forward(state)
    
    # 가치 함수 업데이트
    layer.update_value_function(state, next_state, reward, lr=0.01)
```

### 7.3 메트릭 학습

```python
# 학습 가능한 메트릭
layer = rs.UnifiedRiemannianLayer(
    metric_type="diagonal",
    curvature=0.0,
    input_dim=256,
    enable_bellman=True
)

# 학습 루프
for x, y in dataloader:
    output, energy = layer.forward(x, target=y)
    
    # 라그랑지안 최대화
    loss = -energy['lagrangian'].mean()
    loss.backward()
    
    # 메트릭 업데이트
    layer.update_metric(x, velocity, lr=0.001)
```

## 8. 성능 최적화

### 8.1 복잡도 분석

| 연산 | 복잡도 | 최적화 |
|------|--------|--------|
| 메트릭 계산 | $O(d)$ | 대각 근사 |
| 크리스토펠 기호 | $O(d)$ | 대각 근사 |
| Exponential map | $O(d \cdot steps)$ | Verlet 적분 |
| 가치 함수 | $O(d \cdot h)$ | MLP |
| 전체 순전파 | $O(d \cdot h)$ | 병렬화 |

### 8.2 메모리 사용

- 메트릭: $O(batch \times d)$
- 가치 함수: $O(d \times h + h)$
- 캐시: $O(batch \times d)$
- 총: $O(batch \cdot d + d \cdot h)$

### 8.3 CUDA 가속 (향후)

현재 지원되는 CUDA 커널:
- `poincare_distance_cuda`
- `lorentz_layer_forward_cuda`
- `klein_distance_cuda`

향후 추가 예정:
- `unified_layer_forward_cuda`
- `exponential_map_cuda`
- `christoffel_symbols_cuda`

## 9. 결론

`UnifiedRiemannianLayer`는 리만 기하학, 강화학습, 물리학적 최적화를 통합한 범용 레이어입니다. 

**주요 장점**:
- 4가지 메트릭 모델 유연한 선택
- 벨만 기반 목표 지향적 학습
- 에너지 보존 법칙 활용
- 대각 근사로 실용적 속도

**적용 분야**:
- 계층적 데이터 임베딩
- 강화학습 가치 함수
- 메트릭 학습
- 기하학적 딥러닝

