# 2장. 압축 메커니즘 (Compression Mechanism)

## 1. 개요: 함수형 압축 (Functional Compression)

Reality Stone v2의 압축 메커니즘은 "데이터 압축(Data Compression)"이 아닌 **"함수형 근사(Functional Approximation)"**이다. 우리는 $L$개의 거대한 가중치 행렬들을 각각 저장하는 대신, 이 행렬들을 생성해낼 수 있는 **하나의 함수(Function)**를 학습하여 저장한다.

$$ f_{gen}(l) \to W^{(l)} $$

이 함수 $f_{gen}$은 **전역 기저(Global Basis)**와 **하이퍼네트워크(Hypernetwork)**의 결합으로 구성된다.

---

## 2. 전역 기저 학습 (Global Basis Learning)

모델의 모든 레이어는 서로 다른 역할을 수행하지만, 수학적으로는 동일한 고차원 공간 내에서 작동한다. 우리는 이 공간을 정의하는 **공통의 축(Axis)**을 찾아낸다.

### 2.1 알고리즘
1.  **텐서 스택킹**: 모든 레이어의 $W_Q, W_K, W_V, W_{FFN}$ 행렬을 모아 거대한 3차원 텐서 $\mathcal{W} \in \mathbb{R}^{L \times d_{model} \times d_{model}}$ (또는 $4L \times \dots$)를 구성한다.
2.  **주성분 분석 (PCA/SVD)**: 텐서의 $L$축을 제외한 나머지 차원에 대해 SVD를 수행하여, 데이터 분산(Variance)을 가장 잘 설명하는 상위 $r$개의 기저 벡터 $U, V$를 추출한다.
3.  **공유 구조**: 이 $U, V$는 모델 전체의 "어휘(Vocabulary)"가 되며, 모든 레이어에서 공유된다. 이는 메모리 사용량을 획기적으로 줄이는 핵심 요인이다.

---

## 3. 하이퍼네트워크 (Hypernetwork)

공유 기저 $U, V$ 위에서 각 레이어가 구체적으로 어떤 형태를 띠는지 결정하는 **좌표(Coefficient)**를 생성한다.

### 3.1 구조 (Architecture)
*   **입력**: 레이어 인덱스 $l$ (정규화된 값 또는 임베딩 벡터).
*   **Hidden Layers**: 2~3층의 소형 MLP. (예: $32 \to 128 \to 128 \to r^2$)
*   **출력**: $r \times r$ 크기의 **코어 텐서(Core Tensor)** $C^{(l)}$.
    *   이 $C^{(l)}$은 기저 $U$와 $V$ 사이의 상호작용(Interaction) 강도를 정의한다.

### 3.2 가중치 생성 (Weight Generation)
실제 가중치 행렬 $W^{(l)}$은 다음과 같이 복원된다.
$$ W^{(l)} \approx U \cdot C^{(l)} \cdot V^\top $$
이 연산은 런타임에 수행되며(Lazy Evaluation), 전체 행렬을 복원하지 않고 입력 $x$와 결합하여 최적화된 순서로 계산된다.

---

## 4. 심플렉틱 동역학 통합 (Symplectic Integration)

압축된 가중치를 사용하여 추론을 수행할 때, 단순한 행렬 곱이 아닌 **물리적 보존 법칙**을 따르는 동역학 시스템으로 해석한다.

### 4.1 위상 공간 매핑
*   입력 토큰 $x$를 위치 $q$로, 잠재 변수(Latent)를 운동량 $p$로 매핑한다.
*   $W_Q, W_K$는 위치 $q$의 기하학적 구조(Metric)를 결정한다.
*   $W_{FFN}$은 운동량 $p$에 가해지는 힘(Force)을 결정한다.

### 4.2 에너지 보존
이 구조는 학습된 가중치가 다소 부정확하더라도(압축 오차), 시스템의 총 에너지(Hamiltonian)가 발산하거나 소멸하지 않도록 강제한다. 이는 특히 긴 시퀀스 생성 시 모델의 안정성을 크게 향상시킨다.
