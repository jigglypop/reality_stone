# Reality Stone: 핵심 수식 체계 (Core Equations)

> **목적:** 전체 이론의 “기본식”만 모아, 나머지 문서들이 모두 이 위에 올라가도록 하는 최소 수식 집합.

## 1. 벨만 계열 (목표 지향성)

강화학습의 기본 가치를 그대로 유지하되, 상태 공간만 리만 다양체로 바꿔 쓴다.

### 1.1 가치 함수 (Value Function)

$$
V(s) = \max_a \left[ R(s,a) + \gamma \, \mathbb{E}_{s'}\big[V(s')\big] \right]
$$

### 1.2 Q-함수와 잠재 에너지 연결

$$
Q^*(s, a) = \mathbb{E}\left[ R_{t+1} + \gamma \max_{a'} Q^*(s', a') \mid s, a \right]
$$

$$
V(x_s) = - \max_a Q^*(s, a)
$$

즉, **가치가 높은 상태일수록 잠재 에너지가 낮다.**

## 2. 기하 구조 (리만 메트릭)

상태 공간을 리만 다양체 $(M, g)$ 로 보고, 메트릭 텐서로 “거리”와 “에너지”를 정의한다.

### 2.1 일반 리만 메트릭

$$
g_{ij}(s) = \langle \partial_i, \partial_j \rangle, \quad
ds^2 = g_{ij} \, dx^i dx^j
$$

### 2.2 등각 메트릭 (Conformal Metric, 구현용 단순화)

복잡한 $d \times d$ 메트릭 대신, 스칼라 스케일 $\lambda(s)$ 만 학습하는 형태:

$$
g_{ij}(s) = \lambda(s)^2 \delta_{ij}, \quad
\lambda(s) = e^{\phi(s)}
$$

푸앵카레 볼에서는 다음과 같은 표준 메트릭을 쓸 수 있다:

$$
g_{ij}^P(x) = \left(\frac{2}{1-\lVert x \rVert^2}\right)^2 \delta_{ij}
$$

### 2.3 하이퍼볼릭 거리 (Poincaré 기준)

$$
d(x, y) = \operatorname{arcosh}\left( 1 + 2 \frac{\lVert x-y \rVert^2}{(1-\lVert x \rVert^2)(1-\lVert y \rVert^2)} \right)
$$

## 3. 에너지와 라그랑지안 (Energy & Lagrangian)

상태 변화의 효율성을 “운동 에너지 – 잠재 에너지”로 표현한다.

### 3.1 통합 라그랑지안

$$
L(x, \dot{x}, g)
= \frac{1}{2} g_{\mu\nu}(x) \, \dot{x}^\mu \dot{x}^\nu
  - \big( -Q^*(x) + V_{\text{reg}}(x, g) \big)
$$

여기서

- 운동 에너지: 
  \[
  T = \frac{1}{2} g_{\mu\nu}(x) \, \dot{x}^\mu \dot{x}^\nu
  \]
- 잠재 에너지:
  \[
  V(x, g) = -Q^*(x) + V_{\text{reg}}(x, g)
  \]

정규화 항 $V_{\text{reg}}$ 의 전형적인 구성은 다음과 같다:

- 기억 attractor $m_k$ 로 끌어당기는 항:  
  \[
  \beta \, d(x, m_k)^2
  \]
- 곡률이 과도하게 복잡해지는 것을 억제하는 항:  
  \[
  \gamma \, \lVert \mathrm{Ric}(g) \rVert^2
  \]

### 3.2 최소 작용 원리

궤적 전체에 대한 작용:

$$
S = \int_{t_1}^{t_2} L(x, \dot{x}, g) \, dt
$$

최소 작용 원리:

$$
\delta S = \delta \int L \, dt = 0
$$

## 4. 두 개의 흐름 (표현 흐름 / 메트릭 흐름)

위 라그랑지안에 최소 작용 원리를 각각 적용하면, 상태와 기하에 대한 두 개의 핵심 업데이트 식이 나온다.

### 4.1 표현 흐름 (Representation Flow)

> 주어진 기하 $g$ 위에서, 상태 $x$ 는 **가치를 높이고 에너지를 낮추는 방향의 측지선**을 따라 이동한다.

개념적으로는 다음과 같이 쓸 수 있다:

$$
x' = \exp_x\big(-\eta \, \nabla_g V(x)\big)
$$

여기서

- $\nabla_g V$: 리만 메트릭 $g$ 를 기준으로 한 잠재 에너지의 그라디언트
- $\exp_x(\cdot)$: 다양체 위에서 해당 방향으로 이동시키는 지수 사상(Exponential Map)

### 4.2 메트릭 흐름 (Metric Flow)

> 기하 $g$ 는 전체 라그랑지안 $L$ 을 더 잘 만족시키는 방향으로 서서히 변형된다.

$$
g' = g + \eta \, \frac{\partial L}{\partial g}
$$

$\partial L / \partial g$ 는 “현재 정보 처리에 가장 효율적인 기하 구조가 무엇인지”를 나타내는 방향이다.

## 5. 구현 관점 요약

실제 코드/모델 구현에서 필요한 최소 수식은 다음과 같이 정리할 수 있다.

1. **벨만 계열:**  
   - 가치 함수:  
     \[
     V(s) = \max_a [R(s,a) + \gamma \, \mathbb{E}_{s'} V(s')]
     \]
   - Q-함수와 잠재 에너지의 연결:  
     \[
     V(x_s) = - \max_a Q^*(s, a)
     \]
2. **기하:**  
   - 등각 메트릭 또는 푸앵카레 메트릭  
   - 하이퍼볼릭 거리 $d(x,y)$
3. **에너지:**  
   - 통합 라그랑지안 $L(x,\dot{x},g)$  
4. **흐름 식:**  
   - 표현 흐름: $x' = \exp_x(-\eta \nabla_g V)$  
   - 메트릭 흐름: $g' = g + \eta \, \partial L / \partial g$

이 문서는 위 네 블록만 유지하고, 보다 세부적인 정의·파생·추가 손실 항은 `EQUATION_REFERENCE.md` 에서 참조하는 것을 기준으로 삼는다.
