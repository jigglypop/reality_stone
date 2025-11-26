# 09‑03. Graph Diffusion, Riemannian Laplacian, DP Memory

> 이 문서는 RS‑ULF의 “context 흐름”을 담당하는  
> **방향 그래프 디퓨전, Riemannian Laplacian, Bellman/DP 메모리**를 정리한다.

---

## 1. 방향 그래프 디퓨전

### 1.1 그래프 정의

- 노드 집합: $V$ (자산, 지표, 토큰 그룹 등)
- 방향 엣지 집합: $E \subset V \times V$
- 인접행렬: $A \in \mathbb R^{|V|\times|V|}$
  - $A_{ij} > 0$ 이면 $j \to i$ 방향 영향이 존재
- 차수행렬: $D = \text{diag}(A \mathbf 1)$
- 라플라시안: $L = D - A$

### 1.2 디퓨전 동역학

기본 diffusion PDE:

$$
\partial_t x = - L x
$$

- one‑step 이산화:
  $$
  x' = x - \tau L x
  $$
  - $\tau > 0$: time step (diffusion 강도)

의미:

- 그래프를 따라 에너지가 흘러가며 smoothing
- 금융/퀀트에서는 “변수 간 영향 전파”를 자연스럽게 모델링

### 1.3 RS‑ULF에서의 사용

RS‑ULF 레이어의 업데이트 항 중:

$$
\beta L x
$$

- $\beta$: diffusion 강도
- $Lx$: 각 변수/토큰이 연결된 이웃들로부터 받는 순 변화량

---

## 2. Riemannian Laplacian $\Delta_g$

### 2.1 정의

리만 다양체 위에서의 Laplacian:

$$
\Delta_g f = \frac{1}{\sqrt{|g|}} \partial_i\big(\sqrt{|g|} g^{ij} \partial_j f\big)
$$

여기서:

- $g^{ij}$: metric 역행렬
- $|g|$: metric 행렬식

### 2.2 RS‑ULF에서의 단순 근사

실전에서는 full tensor 대신 아래와 같은 단순 형태를 사용한다.

1. 가장 단순한 형태:
   $$
   \Delta_g x \approx x - \bar x
   $$
   - $\bar x$: batch 또는 시퀀스 평균
2. 조금 더 구조적인 형태:
   $$
   \Delta_g x \approx g^{-1}(x - \bar x)
   $$

RS‑ULF 업데이트 항에서는:

$$
\alpha \Delta_g x
$$

로 들어가며, metric 정보를 반영한 smoothing 역할을 한다.

---

## 3. DP/Bellman Memory $V_t$

### 3.1 정의

RS‑ULF에서 long‑range dependency는 DP/Bellman 메모리로 처리한다.

$$
V_t = \gamma V_{t-1} + \Phi(x_t), \quad 0 < \gamma < 1
$$

- $\Phi(x_t)$: 현재 state에서의 potential
- $\gamma$: discount factor
- $V_t$: 시간 축을 따라 누적된 “가치(value)/컨텍스트”

### 3.2 역할

- 과거의 중요한 상태들이 스칼라(또는 저차원 벡터)로 응축되어 저장
- Attention matrix 없이도 long‑range 정보를 유지
- RS‑ULF 업데이트 항에서는:
  $$
  \gamma V_t
  $$
  형태로 들어가 전체 흐름을 bias 한다.

---

## 4. RS‑ULF 업데이트에서의 조합

RS‑ULF 레이어의 전체 업데이트 항:

$$
x_{t+1}
 = \exp_{x_t} \Big[
 - \eta \nabla_g \Phi(x_t)
 + \alpha \Delta_g x_t
 + \beta L x_t
 + \gamma V_t
 \Big]
$$

각 항의 역할 요약:

- $- \eta \nabla_g \Phi(x_t)$: **local non‑linear dynamics (FFN 역할)**
- $\alpha \Delta_g x_t$: **metric 기반 smoothing (geometry 안정화)**
- $\beta L x_t$: **그래프 기반 multivariate 상호작용 (변수 간 영향 전파)**
- $\gamma V_t$: **시간축 long‑range memory**

이 조합을 통해:

- Transformer의 attention이 담당하던
  - long‑range dependency
  - variable interaction
  - context weighting  
  을 **곡률+디퓨전+DP 구조**가 나눠서 담당하게 된다.

---

## 5. 퀀트/Multivariate 시나리오에서의 해석

### 5.1 노드/엣지 해석

- 노드:
  - 개별 종목, 섹터, 지표, factor, macro 변수 등
- 엣지:
  - 실제/학습된 상관관계, causal edge, 거래 네트워크, factor loading 등

### 5.2 모델 관점

- $\beta L x$:  
  - 시장 구조/자금 흐름/종목 간 영향 같은 “공간적 구조”
- $\alpha \Delta_g x$:  
  - metric이 encode한 방향별 중요도 기반의 regularization
- $\gamma V_t$:  
  - 레짐 전환, 누적 리스크, drawdown 패턴을 포착하는 시간축 메모리

이렇게 보면 RS‑ULF는:

- **기하학(곡률)** + **그래프 구조(네트워크)** + **동학(DP/Bellman)** 을 동시에 사용하는  
  multivariate dynamics 엔진으로 해석된다.

---

## 6. 단계별 체크포인트와 테스트

### 6.1 그래프/라플라시안 체크포인트

- 인접행렬 $A$가 의도한 방향/가중치를 잘 반영하는가?
- $L = D - A$ 계산 후, $L \mathbf 1 = 0$ 이 성립하는가?
- 여러 step 동안 $x \leftarrow x - \tau Lx$ 를 반복했을 때
  - 에너지가 안정적으로 감소하는가?
  - 수치 폭발이 없는가?

테스트 아이디어:

- 간단한 장난감 그래프에서 analytic 해와 수치해 비교
- diffusion만 켠 상태에서 수렴 속도/패턴 시각화

### 6.2 Laplacian 체크포인트

- $\Delta_g x \approx x - \bar x$ 를 사용했을 때,
  - 분산이 지나치게 줄거나 늘지 않는지
- metric을 곱한 형태를 쓸 경우,
  - $g^{-1}(x - \bar x)$ 가 특정 축에서 과도하게 커지지 않는지

### 6.3 DP/Bellman Memory 체크포인트

- $\gamma$ 값에 따라
  - 과거 정보의 “half‑life”가 적절한지
- 긴 시퀀스에서 $V_t$가 overflow/underflow 없이 안정적으로 유지되는지
- $\Phi(x_t)$의 변화가 $V_t$에 “부드럽게” 반영되는지

테스트 아이디어:

- 정해진 synthetic 시나리오에서
  - sudden shock, regime change 등을 넣어보고
  - $V_t$의 반응이 기대한 패턴(충격 후 완만한 감소 등)을 보이는지 확인

---

## 7. 이 문서의 역할

- RS‑ULF에서 **“attention의 역할을 대체하는 세 가지 축”**  
  (그래프 디퓨전, Riemannian Laplacian, DP memory)를 명확하게 분리해 설명한다.
- `02_METRIC_AND_POTENTIAL.md`가 “local dynamics”에 해당한다면,  
  이 문서는 “context 흐름”과 “multivariate 구조”를 담당하는 부분을 정리한다.
- 다음 문서인 `04_TRANSFORMER_MAPPING_AND_TESTS.md`에서는  
  이 구조들이 실제 Transformer attention과 어떤 수식적 관계를 가지는지,  
  그리고 어떻게 정합성 테스트를 설계할지 다룬다.


