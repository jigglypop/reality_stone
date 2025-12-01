# Unified Riemannian Layer API Reference

## Python API

### UnifiedRiemannianLayer

통합 리만 레이어 - 푸앵카레/로렌츠/클라인/대각 메트릭을 하나의 인터페이스로 제공

#### 생성자

```python
import reality_stone as rs

layer = rs.UnifiedRiemannianLayer(
    metric_type: str,
    curvature: float = 1.0,
    input_dim: int = 64,
    enable_bellman: bool = False,
    gamma: float = 0.99
)
```

**Parameters:**

- `metric_type` (str): 메트릭 유형
  - `"poincare"`: 푸앵카레 볼 메트릭 (계층 데이터)
  - `"lorentz"`: 로렌츠/초쌍곡면 메트릭 (수치 안정성)
  - `"klein"`: 클라인 사영 메트릭 (시각화)
  - `"diagonal"`: 학습 가능한 대각 메트릭

- `curvature` (float, default=1.0): 곡률 파라미터 (양수)
  - 큰 값: 강한 곡률, 계층이 깊은 데이터
  - 작은 값: 완만한 곡률, 계층이 얕은 데이터

- `input_dim` (int, default=64): 입력 차원

- `enable_bellman` (bool, default=False): 벨만 가치 함수 활성화
  - `True`: 강화학습 모드, 에너지 최적화
  - `False`: 단순 측지선 흐름

- `gamma` (float, default=0.99): 할인율 (벨만 활성화 시)

**Returns:** `UnifiedRiemannianLayer` 인스턴스

**Example:**

```python
# 푸앵카레 메트릭 (계층 데이터)
poincare_layer = rs.UnifiedRiemannianLayer(
    metric_type="poincare",
    curvature=1.0,
    input_dim=768
)

# 학습 가능한 메트릭 + 벨만
learnable_layer = rs.UnifiedRiemannianLayer(
    metric_type="diagonal",
    input_dim=256,
    enable_bellman=True,
    gamma=0.95
)
```

---

#### Methods

##### forward(x, target=None)

순전파 - 입력에 대한 출력 및 에너지 계산

```python
output, energy = layer.forward(x, target=None)
```

**Parameters:**

- `x` (np.ndarray): 입력 텐서, shape=(batch_size, input_dim)
- `target` (np.ndarray, optional): 목표점, shape=(batch_size, input_dim)
  - 제공 시: 측지선 보간 (x와 target 사이)
  - 미제공 시: 가치 함수 기반 흐름 (벨만 활성화 시) 또는 항등

**Returns:**

- `output` (np.ndarray): 출력, shape=(batch_size, input_dim)
- `energy` (dict or None): 에너지 정보 (벨만 활성화 시)
  - `"kinetic"`: 운동 에너지, shape=(batch_size,)
  - `"potential"`: 잠재 에너지, shape=(batch_size,)
  - `"lagrangian"`: 라그랑지안, shape=(batch_size,)
  - `"bellman_residual"`: 벨만 잔차, shape=(batch_size,)

**Example:**

```python
import numpy as np

x = np.random.randn(32, 768) * 0.1
y = np.random.randn(32, 768) * 0.1

# 측지선 보간
output, energy = layer.forward(x, target=y)

# 흐름 (벨만 활성화 시)
output, energy = layer.forward(x)
if energy is not None:
    print(f"Kinetic: {energy['kinetic'].mean()}")
    print(f"Lagrangian: {energy['lagrangian'].mean()}")
```

---

##### backward(grad_output, x)

역전파 - 그래디언트 계산

```python
grad_input = layer.backward(grad_output, x)
```

**Parameters:**

- `grad_output` (np.ndarray): 출력에 대한 그래디언트, shape=(batch_size, input_dim)
- `x` (np.ndarray): 입력, shape=(batch_size, input_dim)

**Returns:**

- `grad_input` (np.ndarray): 입력에 대한 그래디언트, shape=(batch_size, input_dim)

**Example:**

```python
grad_out = np.ones_like(output)
grad_in = layer.backward(grad_out, x)
```

---

##### geodesic_path(start, end, num_steps)

측지선 경로 생성 - start에서 end까지의 경로

```python
path = layer.geodesic_path(start, end, num_steps)
```

**Parameters:**

- `start` (np.ndarray): 시작점, shape=(batch_size, input_dim)
- `end` (np.ndarray): 끝점, shape=(batch_size, input_dim)
- `num_steps` (int): 경로 분할 개수

**Returns:**

- `path` (list of np.ndarray): 경로상의 점들, 각 shape=(batch_size, input_dim)

**Example:**

```python
start = np.random.randn(1, 64) * 0.1
end = np.random.randn(1, 64) * 0.1

path = layer.geodesic_path(start, end, num_steps=10)
print(f"Path length: {len(path)}")  # 10

# 시각화
import matplotlib.pyplot as plt
path_array = np.array([p[0, :2] for p in path])
plt.plot(path_array[:, 0], path_array[:, 1], 'o-')
plt.show()
```

---

##### compute_energy(x, v, x_next, reward)

에너지 계산 - 운동/잠재/라그랑지안 에너지

```python
energy = layer.compute_energy(x, v, x_next, reward)
```

**Parameters:**

- `x` (np.ndarray): 현재 상태, shape=(batch_size, input_dim)
- `v` (np.ndarray): 속도 (변화율), shape=(batch_size, input_dim)
- `x_next` (np.ndarray): 다음 상태, shape=(batch_size, input_dim)
- `reward` (np.ndarray): 보상, shape=(batch_size,)

**Returns:**

- `energy` (dict): 에너지 구성 요소
  - `"kinetic"`: $T = \frac{1}{2} g_{ij} v^i v^j$
  - `"potential"`: $V = (V(x) - R - \gamma V(x'))^2$
  - `"lagrangian"`: $L = T - V$
  - `"bellman_residual"`: $V(x) - (R + \gamma V(x'))$

**Example:**

```python
x = np.random.randn(16, 128) * 0.1
v = np.random.randn(16, 128) * 0.01
x_next = x + v * 0.1
reward = np.random.randn(16)

energy = layer.compute_energy(x, v, x_next, reward)
print(f"Kinetic: {energy['kinetic']}")
print(f"Potential: {energy['potential']}")
```

---

##### flow_step(x, num_steps, learning_rate)

표현 흐름 - 가치 함수 그래디언트를 따라 이동

```python
result = layer.flow_step(x, num_steps, learning_rate)
```

**Parameters:**

- `x` (np.ndarray): 현재 상태, shape=(batch_size, input_dim)
- `num_steps` (int): 흐름 반복 횟수
- `learning_rate` (float): 학습률 (스텝 크기)

**Returns:**

- `result` (np.ndarray): 흐름 후 상태, shape=(batch_size, input_dim)

**Example:**

```python
x_init = np.random.randn(8, 64) * 0.1

# 10 스텝 흐름
x_final = layer.flow_step(x_init, num_steps=10, learning_rate=0.01)

# 거리 계산
distance = np.linalg.norm(x_final - x_init, axis=1)
print(f"Flow distance: {distance.mean()}")
```

---

##### update_value_function(x, x_next, reward, learning_rate)

가치 함수 업데이트 - TD(0) 학습

```python
layer.update_value_function(x, x_next, reward, learning_rate)
```

**Parameters:**

- `x` (np.ndarray): 현재 상태, shape=(batch_size, input_dim)
- `x_next` (np.ndarray): 다음 상태, shape=(batch_size, input_dim)
- `reward` (np.ndarray): 보상, shape=(batch_size,)
- `learning_rate` (float): 학습률

**Returns:** None (in-place 업데이트)

**Example:**

```python
# 강화학습 루프
for episode in range(100):
    state = env.reset()
    for t in range(max_steps):
        action = policy(state)
        next_state, reward, done = env.step(action)
        
        # 가치 함수 업데이트
        layer.update_value_function(
            state[np.newaxis],
            next_state[np.newaxis],
            np.array([reward]),
            learning_rate=0.01
        )
        
        if done:
            break
        state = next_state
```

---

##### update_metric(x, v, learning_rate)

메트릭 업데이트 - 대각 메트릭 학습 (diagonal만 지원)

```python
layer.update_metric(x, v, learning_rate)
```

**Parameters:**

- `x` (np.ndarray): 현재 상태, shape=(batch_size, input_dim)
- `v` (np.ndarray): 속도, shape=(batch_size, input_dim)
- `learning_rate` (float): 학습률

**Returns:** None (in-place 업데이트)

**Example:**

```python
# 메트릭 학습 가능한 레이어
layer = rs.UnifiedRiemannianLayer(
    metric_type="diagonal",
    input_dim=256,
    enable_bellman=True
)

x = np.random.randn(32, 256) * 0.1
v = np.random.randn(32, 256) * 0.01

# 메트릭 업데이트
layer.update_metric(x, v, learning_rate=0.001)
```

---

## 독립 함수

### compute_metric(x, metric_type, curvature)

메트릭 텐서 계산

```python
metric = rs.compute_metric(x, metric_type, curvature)
```

**Parameters:**

- `x` (np.ndarray): 입력, shape=(batch_size, input_dim)
- `metric_type` (str): "poincare", "lorentz", "klein", "diagonal"
- `curvature` (float): 곡률 파라미터

**Returns:**

- `metric` (np.ndarray): 메트릭 텐서 (대각 원소), shape=(batch_size, input_dim)

---

### geodesic_distance(x, y, metric_type, curvature)

측지선 거리 계산

```python
distance = rs.geodesic_distance(x, y, metric_type, curvature)
```

**Parameters:**

- `x, y` (np.ndarray): 두 점, shape=(batch_size, input_dim)
- `metric_type` (str): 메트릭 유형
- `curvature` (float): 곡률

**Returns:**

- `distance` (np.ndarray): 거리, shape=(batch_size,)

**Example:**

```python
x = np.random.randn(100, 64) * 0.1
y = np.random.randn(100, 64) * 0.1

dist_poincare = rs.geodesic_distance(x, y, "poincare", 1.0)
dist_euclidean = rs.geodesic_distance(x, y, "diagonal", 0.0)

print(f"Poincaré dist: {dist_poincare.mean()}")
print(f"Euclidean dist: {dist_euclidean.mean()}")
```

---

### geodesic_interpolate(x, y, metric_type, curvature, t=0.5)

측지선 보간

```python
mid = rs.geodesic_interpolate(x, y, metric_type, curvature, t=0.5)
```

**Parameters:**

- `x, y` (np.ndarray): 시작/끝점, shape=(batch_size, input_dim)
- `metric_type` (str): 메트릭 유형
- `curvature` (float): 곡률
- `t` (float, default=0.5): 보간 파라미터 (0: x, 1: y)

**Returns:**

- `mid` (np.ndarray): 보간점, shape=(batch_size, input_dim)

**Example:**

```python
x = np.zeros((1, 64))
y = np.ones((1, 64))

# 중간점
mid = rs.geodesic_interpolate(x, y, "poincare", 1.0, t=0.5)

# 1/4 지점
quarter = rs.geodesic_interpolate(x, y, "poincare", 1.0, t=0.25)
```

---

## 사용 예제

### 예제 1: 계층적 임베딩

```python
import numpy as np
import reality_stone as rs

# 푸앵카레 레이어
layer = rs.UnifiedRiemannianLayer(
    metric_type="poincare",
    curvature=1.0,
    input_dim=768,
    enable_bellman=False
)

# 텍스트 임베딩 (예: BERT 출력)
text_embeddings = np.random.randn(128, 768) * 0.1

# 계층 공간으로 매핑
hierarchical_embeddings, _ = layer.forward(text_embeddings)
```

### 예제 2: 강화학습

```python
# 벨만 활성화
rl_layer = rs.UnifiedRiemannianLayer(
    metric_type="diagonal",
    input_dim=128,
    enable_bellman=True,
    gamma=0.99
)

# 학습 루프
for state, action, reward, next_state in experience_buffer:
    # 가치 함수 업데이트
    rl_layer.update_value_function(
        state[np.newaxis],
        next_state[np.newaxis],
        np.array([reward]),
        learning_rate=0.01
    )
    
    # 에너지 계산
    velocity = (next_state - state) / dt
    energy = rl_layer.compute_energy(
        state[np.newaxis],
        velocity[np.newaxis],
        next_state[np.newaxis],
        np.array([reward])
    )
    
    print(f"Lagrangian: {energy['lagrangian'][0]}")
```

### 예제 3: 측지선 시각화

```python
import matplotlib.pyplot as plt

layer = rs.UnifiedRiemannianLayer("poincare", 1.0, 2, False)

start = np.array([[0.0, 0.0]])
end = np.array([[0.8, 0.0]])

# 측지선 경로
path = layer.geodesic_path(start, end, num_steps=50)
path_array = np.array([p[0] for p in path])

# 푸앵카레 디스크 경계
theta = np.linspace(0, 2*np.pi, 100)
circle = np.column_stack([np.cos(theta), np.sin(theta)])

plt.figure(figsize=(8, 8))
plt.plot(circle[:, 0], circle[:, 1], 'k--', alpha=0.3)
plt.plot(path_array[:, 0], path_array[:, 1], 'b-o', markersize=3)
plt.plot(start[0, 0], start[0, 1], 'go', markersize=10, label='Start')
plt.plot(end[0, 0], end[0, 1], 'ro', markersize=10, label='End')
plt.axis('equal')
plt.legend()
plt.title('Geodesic in Poincaré Disk')
plt.show()
```

---

## 참고사항

### 수치 안정성

- 푸앵카레/클라인: 경계 ($\|x\| \to 1$) 근처에서 주의
- 로렌츠: 일반적으로 안정적
- 대각: 학습 초기에 불안정할 수 있음 (learning rate 조절)

### 성능 팁

- 배치 크기: 32-256 권장
- 흐름 스텝: 5-10 (너무 크면 느림)
- 학습률: 메트릭 0.001, 가치 함수 0.01

### CUDA 지원

현재는 CPU만 지원. CUDA 버전은 향후 업데이트 예정.

