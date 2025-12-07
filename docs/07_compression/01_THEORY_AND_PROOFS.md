# 1장. 이론적 토대와 수학적 증명 (Theory & Proofs)

## 1. 서론: 보편 압축의 불가능성과 해법

### 1.1 불가능성 정리 (Impossibility Theorem)
사드 정리(Sard's Theorem)와 위상수학적 차원 이론에 따르면, 낮은 차원 공간 $\mathbb{R}^M$에서 높은 차원 공간 $\mathbb{R}^P$ ($M < P$)로 가는 매끄러운(Smooth) 함수 $F$가 전사(Surjective)가 되는 것은 불가능하다. 즉, 임의의 트랜스포머 가중치를 더 적은 파라미터로 무손실 압축하는 "보편 압축기"는 수학적으로 존재할 수 없다.

### 1.2 내재 차원 가설 (Intrinsic Dimension Hypothesis)
그러나 딥러닝 모델의 가중치는 무작위로 분포하지 않는다. 데이터 $\mathcal{D}$로 학습된 파라미터 $W^*$는 전체 파라미터 공간 $\mathbb{R}^P$ 내의 매우 낮은 차원을 갖는 부분 다양체(Submanifold) $\mathcal{M}_{valid}$ 위에 존재한다. Reality Stone v2는 이 $\mathcal{M}_{valid}$를 매개변수화(Parameterization)함으로써 압축을 달성한다.

---

## 2. 함수형 다양체 압축 (Functional Manifold Compression)

### 2.1 가중치 텐서의 함수적 표현
$L$개의 레이어를 갖는 트랜스포머 모델을 고려하자. 각 레이어 $l \in \{1, \dots, L\}$의 가중치 행렬 $W^{(l)} \in \mathbb{R}^{d \times d}$는 서로 독립적인 이산적 점들이 아니라, 매끄러운 곡선(Smooth Curve) $\gamma: [0, 1] \to \mathbb{R}^{d \times d}$ 상의 샘플링 포인트로 간주할 수 있다.

$$ 
W^{(l)} = \gamma(t_l), \quad t_l = \frac{l}{L} 
$$

### 2.2 구조적 분해 (Structural Decomposition)
이 곡선 $\gamma(t)$는 전역적인 기하학적 구조(Global Geometry)와 국소적인 변동(Local Variation)으로 분해된다. 우리는 이를 다음과 같은 함수형 폼(Functional Form)으로 정의한다.

$$ 
W(t) \approx U_{global} \cdot \phi_\theta(t) \cdot V_{global}^\top 
$$

여기서:
*   **$U_{global}, V_{global} \in \mathbb{R}^{d \times r}$ (Global Basis)**:
    모든 레이어가 공유하는 고차원 기저 공간이다. 이는 모델이 학습한 지식의 "형태(Shape)"를 정의하는 시불변(Time-invariant) 구조체다. ($r \ll d$)
*   **$\phi_\theta(t): [0, 1] \to \mathbb{R}^{r \times r}$ (Local Coordinate Function)**:
    레이어 인덱스(시간) $t$에 따라 기저들의 결합 계수를 생성하는 함수다. 이는 모델 깊이에 따른 지식의 "흐름(Flow)"을 정의한다. 이 함수 $\phi$는 작은 신경망(Hypernetwork)으로 근사된다.

이 표현 방식은 행렬의 **랭크(Rank)**를 제한하는 것이 아니라, 행렬을 생성하는 **자유도(Degrees of Freedom)**를 제한하는 것이다. 따라서 양자화(Quantization) 없이도 부동소수점 정밀도를 유지하며 극단적인 압축이 가능하다.

---

## 3. 심플렉틱 동역학 (Symplectic Dynamics)

### 3.1 문제 제기: FFN의 비보존성
기존의 지오데식 접근법은 트랜스포머의 잔차 업데이트(Residual Update)를 리만 지오데식으로 해석하려 했으나, FFN(Feed-Forward Network) 벡터장 $F(x)$는 일반적으로 $\nabla \times F \neq 0$인 비보존장(Non-conservative field)이므로 스칼라 포텐셜 $-\nabla \Phi$로 표현할 수 없다. 이는 수학적 모순을 야기한다.

### 3.2 해밀토니안 리프팅 (Hamiltonian Lifting)
이 문제를 해결하기 위해, 우리는 시스템을 **위상 공간(Phase Space) $T^*\mathcal{M} \cong \mathbb{R}^{2d}$**로 확장(Lifting)한다. 상태 변수를 위치 $q$와 운동량 $p$로 분리하고, 트랜스포머의 업데이트를 해밀토니안 시스템의 **심플렉틱 맵(Symplectic Map)**으로 재정의한다.

해밀토니안 $H(q, p)$를 다음과 같이 정의한다.

$$ 
H(q, p) = \frac{1}{2} p^\top M^{-1} p + V(q) 
$$

여기에 FFN을 운동량에 작용하는 **비보존적 킥(Non-conservative Kick)**으로 모델링한다.

### 3.3 심플렉틱 오일러 적분 (Symplectic Euler Integration)
트랜스포머의 한 레이어 업데이트는 위상 공간에서의 **1차 심플렉틱 오일러 적분 단계**와 수학적으로 동치이다.

$$
\begin{aligned}
p_{t+1} &= p_t - \epsilon \nabla V(q_t) + \epsilon \mathcal{F}_{FFN}(q_t) \\
q_{t+1} &= q_t + \epsilon M^{-1} p_{t+1}
\end{aligned}
$$

*   **Attention**: 위치 $q$에 따른 메트릭/포텐셜 효과로 해석 ($\nabla V$).
*   **FFN**: 운동량 $p$를 직접 변화시키는 외력($\mathcal{F}_{FFN}$)으로 해석.
*   **Residual**: 운동량에 의한 위치의 갱신 ($M^{-1} p$).

이 구조는 리우빌 정리(Liouville's Theorem)에 의해 위상 공간의 부피(정보량)를 보존하며, 장기간의 추론에서도 수치적 안정성을 보장한다.

---

## 4. 증명: 압축 오차와 복원 (Error Bounds)

### 4.1 매끄러움에 의한 오차 상계
함수 $\phi(t)$가 $L$-Lipschitz 연속이라 가정하면, 하이퍼네트워크 근사 오차 $\delta$는 다음과 같이 상계된다.

$$ 
\| W^{(l)} - \tilde{W}^{(l)} \|_F \le C \cdot \frac{1}{N_{params}^\alpha} 
$$

여기서 $N_{params}$는 하이퍼네트워크의 파라미터 수이며, $\alpha$는 근사 차수다. 이는 파라미터 수를 늘릴수록 오차가 다항식적으로(Polynomially) 감소함을 의미한다.

### 4.2 동역학적 안정성
심플렉틱 통합기를 사용하므로, 국소 오차(Local Error)가 $O(\epsilon^2)$일 때 전역 오차(Global Error)는 $O(\epsilon)$으로 선형적으로만 증가한다. 이는 비-심플렉틱 방법(예: Runge-Kutta)이 지수적(Exponential)으로 오차를 누적시키는 것과 대비되어, 깊은 레이어(Deep Layers)를 가진 모델 압축에 유리하다.
