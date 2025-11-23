# Banking77을 위한 리만-라그랑지안 의도 분류 (Riemannian-Lagrangian Intent Classification)

## 1. 개요 (Overview)

Banking77 데이터셋은 77개의 세분화된(fine-grained) 뱅킹 의도를 분류하는 과제입니다. 클래스 간의 의미적 경계가 모호하고 겹치는 영역이 많아, 기존의 유클리드 기하학 기반 선형 분류기로는 한계가 있습니다.

본 문서는 **리만 기하학(Riemannian Geometry)** 과 **라그랑주 역학(Lagrangian Mechanics)** 의 원리를 적용하여, 의도 분류 문제를 **"곡선 공간에서의 에너지 최소화 경로 탐색"** 문제로 재정의하고 해결하는 방법을 기술합니다.

---

## 2. 이론적 배경 (Theoretical Background)

### 2.1. 왜 리만 기하학인가? (Why Riemannian?)

언어의 의미 공간은 평탄하지 않습니다. "카드 분실"과 "카드 도난"은 매우 가깝지만, "계좌 개설"과는 멉니다. 이러한 의미적 거리는 중심부(일반적 의도)에서 가장자리(구체적 의도)로 갈수록 기하급수적으로 팽창하는 **계층적 구조(Hierarchy)** 를 가집니다.

- **유클리드 공간 ($\mathbb{R}^n$)**: 공간의 크기가 반경에 정비례 ($r^n$). 계층 구조를 담기에는 공간이 부족하여 차원을 억지로 늘려야 함.
- **쌍곡 공간 ($\mathbb{H}^n$)**: 공간의 크기가 반경에 대해 지수적으로 증가 ($e^r$). 적은 차원으로도 복잡한 계층적 의도를 효과적으로 임베딩 가능.

### 2.2. 라그랑주 역학적 해석 (Lagrangian Interpretation)

분류 모델의 추론 과정을 **입력 상태($x$)에서 정답 상태($y$)로 이동하는 입자의 운동**으로 봅니다. 자연계의 모든 운동은 **작용(Action, $S$)** 을 최소화하는 경로를 따릅니다.

$$
S = \int L(q, \dot{q}, t) dt
$$

여기서 라그랑지안 $L = T - V$ (운동 에너지 - 위치 에너지)입니다.
우리의 분류기에서 이를 다음과 같이 해석합니다:

- **운동 에너지 ($T$)**: 임베딩 벡터가 공간상에서 이동하려는 성질. (Regularization)
- **위치 에너지 ($V$)**: 정답 클래스(프로토타입)와의 거리. (Classification Loss)

따라서 학습은 **총 에너지를 최소화하는 기하학적 구조(Metric)를 찾는 과정**이 됩니다.

---

## 3. 수식 설계 (Mathematical Formulation)

### 3.1. 상태 초기화 (State Initialization)

강력한 문맥 이해를 위해 Pre-trained Language Model (PLM)인 `RoBERTa-Large`를 사용하여 초기 상태 벡터를 얻습니다.

$$
z = \text{RoBERTa}(x) \in \mathbb{R}^{1024}
$$

이 $z$는 아직 유클리드 공간(접공간)에 존재합니다.

### 3.2. 접공간 투영 (Tangent Space Projection)

Transformer의 `Linear` 레이어 대신, 리만 다양체의 접공간($T_0\mathbb{D}$)에서의 변환으로 정의된 `EquivalentHyperbolicLinear`를 사용합니다. 이는 차원을 축소하면서 의미적 특징을 추출합니다.

$$
h_{\text{tan}} = W z + b \quad \in \mathbb{R}^{d} \quad (d=128)
$$

### 3.3. 지수 맵 (Exponential Map)

접공간의 벡터를 실제 리만 다양체(푸앵카레 볼, $\mathbb{D}^d_c$) 위로 투영합니다. 곡률 $c$는 공간이 얼마나 휘어있는지를 결정합니다.

$$
h_{\text{hyp}} = \exp_0^c(h_{\text{tan}}) = \tanh(\sqrt{c} \|h_{\text{tan}}\|) \frac{h_{\text{tan}}}{\sqrt{c}\|h_{\text{tan}}\|}
$$

이 연산을 통해 데이터는 **유클리드 감옥**을 탈출하여, 무한한 깊이를 가진 **쌍곡 공간**으로 진입합니다.

### 3.4. 등각 변환 (Conformal Scaling)

데이터의 밀도에 따라 공간의 척도(Metric)를 동적으로 조절합니다. 푸앵카레 볼 모델의 등각 계수(Conformal Factor) $\lambda_x$는 다음과 같습니다:

$$
\lambda_x^c = \frac{2}{1 - c\|x\|^2}
$$

경계($\|x\| \to 1/\sqrt{c}$)로 갈수록 $\lambda_x \to \infty$가 됩니다. 이는 구체적인 의도가 몰려있는 가장자리 영역에서의 거리를 확대하여 변별력을 높입니다.

### 3.5. 확률적 분류 (Probabilistic Classification)

각 의도 클래스 $k$는 다양체 상의 프로토타입 점 $p_k \in \mathbb{D}^d_c$로 표현됩니다.
입력 $x$가 클래스 $k$일 확률은 **리만 거리의 제곱**에 반비례하는 볼츠만 분포를 따릅니다.

$$
P(y=k|x) = \frac{\exp\left(-\tau \cdot d_c(h_{\text{hyp}}, p_k)^2\right)}{\sum_{j} \exp\left(-\tau \cdot d_c(h_{\text{hyp}}, p_j)^2\right)}
$$

여기서 온도 $\tau$는 등각 계수에 의해 조절됩니다: $\tau = \alpha (\lambda_{h_{\text{hyp}}})^\gamma$.

---

## 4. 구현 아키텍처 (Implementation Architecture)

### 4.1. 모델 구조: `RiemannianIntentClassifier`

1.  **Backbone**: `roberta-large` (Frozen or Finetuned)
2.  **Projection Head**:
    - `EquivalentHyperbolicLinear` (1024 $\to$ 1024) $\to$ GELU $\to$ Dropout
    - `EquivalentHyperbolicLinear` (1024 $\to$ 128) $\to$ LayerNorm
    - **목적**: 고차원 유클리드 정보를 저차원 리만 정보로 압축.
3.  **Manifold Mapping**:
    - `exp_map_zero`: Tangent $\to$ Poincare.
    - `project_to_ball`: 수치적 안정성을 위해 경계(radius 1.0) 내부로 강제 ($1-\epsilon$).
4.  **Distance Head**:
    - `poincare_dist_sq`: 모든 클래스 프로토타입과의 쌍곡 거리 계산.
    - `logsumexp`: 수치적으로 안정적인 Log-Softmax 계산.

### 4.2. 학습 전략: `train_banking77`

- **Loss Function**:
  $$ \mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{proto}} \mathcal{L}_{\text{reg}} + \lambda_{\text{lap}} \mathcal{L}_{\text{lap}} $$
  - $\mathcal{L}_{\text{CE}}$: 분류 정확도 (Cross Entropy).
  - $\mathcal{L}_{\text{reg}}$: 프로토타입이 원점에서 너무 멀어지지 않도록 제어.
  - $\mathcal{L}_{\text{lap}}$: 같은 클래스의 데이터끼리 뭉치게 하는 Laplacian Loss.

---

## 5. 기대 효과 (Expected Benefits)

1.  **소수 클래스 성능 향상**: 데이터가 적은 희귀 의도들도 쌍곡 공간의 가장자리에서 충분한 공간을 확보하여 분류됨.
2.  **강건성(Robustness)**: 입력의 미세한 노이즈가 등각 변환에 의해 적절히 스케일링되어, 의미적 거리가 보존됨.
3.  **해석 가능성**: 임베딩 공간을 시각화했을 때, 중심부(일반적) $\leftrightarrow$ 가장자리(구체적)의 계층 구조가 명확히 드러남.

