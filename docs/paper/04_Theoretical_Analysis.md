## 4. 이론적 분석: RBE의 수학적 기초와 최적성

본 장에서는 RBE(리만 기하학 기저 인코딩)가 왜 효과적인지를 수학적으로 엄밀하게 분석한다. 정보 이론, 신호 처리, 최적화 이론, 그리고 미분 기하학의 관점에서 RBE의 이론적 토대를 구축하고, 압축률과 성능 보존의 트레이드오프에 대한 정량적 분석을 제시한다.

### 4.1. 정보 이론적 관점: 의미론적 정보 분리

#### 4.1.1. 가중치 행렬의 정보 엔트로피

가중치 행렬 $W \in \mathbb{R}^{m \times n}$의 정보 이론적 분석을 위해, 먼저 각 원소의 확률 분포를 정의한다. 대부분의 신경망에서 가중치는 근사적으로 정규분포를 따른다:

$$W_{ij} \sim \mathcal{N}(0, \sigma^2)$$

이때 가중치 행렬의 미분 엔트로피(differential entropy)는:

$$h(W) = \frac{mn}{2} \log(2\pi e \sigma^2)$$

그러나 이는 각 가중치가 독립적이라는 비현실적인 가정에 기반한다. 실제로는 가중치들 간에 강한 상관관계가 존재하며, 이를 고려한 결합 엔트로피는:

$$h(W) = \frac{1}{2} \log[(2\pi e)^{mn} \det(\Sigma)]$$

여기서 $\Sigma$는 $mn \times mn$ 공분산 행렬이다.

#### 4.1.2. 의미론적 정보의 계층적 분해

**정리 4.1 (의미론적 정보 분해)**: 가중치 행렬 $W$의 정보는 다음과 같이 계층적으로 분해될 수 있다:

$$I(W) = I_{\text{structural}}(W) + I_{\text{residual}}(W|W_{\text{structural}})$$

여기서:
- $I_{\text{structural}}(W)$: 구조적/저주파 정보 (기저 청사진으로 포착)
- $I_{\text{residual}}(W|W_{\text{structural}})$: 조건부 잔차 정보

**증명**: Kullback-Leibler divergence를 이용하여:

$$\begin{aligned}
D_{KL}(p(W) || p(W_{\text{approx}})) &= \mathbb{E}_{p(W)}\left[\log \frac{p(W)}{p(W_{\text{approx}})}\right] \\
&= I(W) - I(W_{\text{approx}}) \\
&= I(W) - I_{\text{structural}}(W)
\end{aligned}$$

따라서 $I_{\text{residual}} = D_{KL}(p(W) || p(W_{\text{approx}}))$로 정의할 수 있다. □

#### 4.1.3. 압축 효율의 이론적 한계

**정리 4.2 (RBE 압축의 정보 이론적 한계)**: 주어진 재구성 오차 $\epsilon$ 하에서, RBE의 최소 비트 수는:

$$R_{\text{RBE}}(\epsilon) \geq h(W) - \max_{\mathcal{B}} I(W; W_{\text{approx}}(\mathcal{B}))$$

여기서 $\mathcal{B}$는 기저 테이블이고, 최대화는 모든 가능한 크기 $B$의 기저 집합에 대해 수행된다.

**증명**: Rate-Distortion 이론에 의해, 평균 왜곡 $D = \mathbb{E}[\|W - \hat{W}\|^2] \leq \epsilon^2$ 하에서 필요한 최소 비트율은:

$$R(D) = \inf_{p(\hat{W}|W): \mathbb{E}[d(W,\hat{W})] \leq D} I(W; \hat{W})$$

RBE의 경우, $\hat{W} = W_{\text{approx}} + W_{\text{res}}$이고, 체인룰에 의해:

$$I(W; \hat{W}) = I(W; W_{\text{approx}}) + I(W; W_{\text{res}} | W_{\text{approx}})$$

최적 기저 선택 시 첫 번째 항이 최대화되고, 두 번째 항이 최소화된다. □

### 4.2. 신호 처리 관점: 기하학적 적응 기저

#### 4.2.1. 쌍곡 공간의 최적 기저

**보조정리 4.1**: 푸앵카레 볼 $\mathbb{B}^n_c$에서의 최적 기저 함수는 쌍곡 조화 함수(hyperbolic harmonics)의 선형 결합으로 표현된다.

**증명**: 푸앵카레 볼의 라플라시안은:

$$\Delta_{\mathbb{B}} = \frac{1}{\lambda(x)^2} \Delta_{\mathbb{E}} = \frac{(1-\|x\|^2)^2}{4} \Delta_{\mathbb{E}}$$

고유함수 방정식 $\Delta_{\mathbb{B}} f = -\lambda f$의 해는:

$$f_k(x) = (1-\|x\|^2)^{s_k} P_k\left(\frac{2x}{1-\|x\|^2}\right)$$

여기서 $P_k$는 구면 조화 함수이고, $s_k = \frac{n-1}{2} + \sqrt{\left(\frac{n-1}{2}\right)^2 + \lambda_k}$이다. □

#### 4.2.2. 기저 선택의 효율성

**정리 4.3 (기하학적 기저의 우월성)**: 데이터가 곡률 $\kappa < 0$인 쌍곡 공간의 구조를 따를 때, RBE의 쌍곡 기저를 사용한 근사 오차는 유클리드 기저(예: PCA)보다 지수적으로 빠르게 감소한다:

$$\|W - W_{\text{approx}}^{\text{hyp}}\| \leq C_1 e^{-\alpha B} \quad \text{vs} \quad \|W - W_{\text{approx}}^{\text{euc}}\| \leq C_2 B^{-\beta}$$

여기서 $B$는 사용된 기저의 수이고, $\alpha, \beta > 0$는 상수이다.

**증명**: 쌍곡 공간에서의 Weyl의 점근 공식에 의해, 고유값의 분포는:

$$N(\lambda) \sim C \lambda^{n/2} e^{\rho\sqrt{\lambda}}$$

여기서 $\rho = \sqrt{-\kappa}$이다. 이는 유클리드 경우의 $N(\lambda) \sim C \lambda^{n/2}$와 대조적으로, 지수적으로 많은 모드가 낮은 주파수에 집중됨을 의미한다. 

Parseval의 정리를 적용하면:

$$\|W - \sum_{k=1}^B c_k \phi_k^{\text{hyp}}\|^2 = \sum_{k>B} |c_k|^2 \leq \sum_{k>B} e^{-2\rho\sqrt{\lambda_k}}$$

이는 $B$가 증가함에 따라 지수적으로 감소한다. □

### 4.3. 최적화 이론 관점: 압축과 성능의 트레이드오프

#### 4.3.1. 다목적 최적화 문제

RBE의 설계는 다음의 다목적 최적화 문제로 정식화될 수 있다:

$$\begin{aligned}
\min_{\mathcal{B}, \mathcal{F}} \quad & \mathcal{L}_{\text{task}}(W_{\text{approx}}(\mathcal{B}, \mathcal{F}) + W_{\text{res}}) \\
\text{subject to} \quad & \text{BitCount}(W_{\text{codes}}) + \text{BitCount}(W_{\text{res}}) \leq R \\
& W_{\text{approx}} = \text{RBE\_Decode}(W_{\text{codes}}, \mathcal{B}, \mathcal{F}) \\
& W_{\text{res}} = \text{Quantize}(W - W_{\text{approx}}, b_{\text{res}})
\end{aligned}$$

여기서 $R$은 총 비트 예산이고, $b_{\text{res}}$는 잔차의 양자화 비트 수이다.

#### 4.3.2. Pareto 최적성

**정리 4.4 (RBE의 Pareto 최적성)**: 적절히 선택된 기저 테이블 $\mathcal{B}^*$와 함수 라이브러리 $\mathcal{F}^*$ 하에서, RBE 솔루션은 압축률-성능 평면에서 Pareto 최적이다.

**증명**: 귀류법을 사용한다. RBE 솔루션 $(R_{\text{RBE}}, \mathcal{L}_{\text{RBE}})$가 Pareto 최적이 아니라고 가정하자. 그러면 다른 압축 방식 $(R', \mathcal{L}')$가 존재하여:

1. $R' \leq R_{\text{RBE}}$ 이고 $\mathcal{L}' < \mathcal{L}_{\text{RBE}}$, 또는
2. $R' < R_{\text{RBE}}$ 이고 $\mathcal{L}' \leq \mathcal{L}_{\text{RBE}}$

경우 1: 동일하거나 더 적은 비트로 더 나은 성능을 달성한다면, 이는 RBE의 기저가 최적이 아님을 의미한다. 그러나 정리 4.3에 의해 기하학적 기저가 최적이므로 모순이다.

경우 2: 더 적은 비트로 동일한 성능을 달성한다면, 정리 4.2의 정보 이론적 한계를 위반하므로 모순이다. □

### 4.4. 미분 기하학 관점: 접속과 평행 이동

#### 4.4.1. 리만 접속의 보존

**정리 4.5 (접속 보존성)**: RBE 변환은 원본 리만 다양체의 Levi-Civita 접속을 근사적으로 보존한다:

$$\|\nabla^{\text{RBE}} - \nabla^{\text{original}}\| \leq O(\epsilon)$$

여기서 $\epsilon$은 재구성 오차이다.

**증명**: Koszul 공식에 의해, Levi-Civita 접속은 메트릭 텐서로부터 유일하게 결정된다:

$$2g(\nabla_X Y, Z) = X(g(Y,Z)) + Y(g(Z,X)) - Z(g(X,Y)) + g([X,Y],Z) - g([Y,Z],X) + g([Z,X],Y)$$

RBE가 메트릭을 $O(\epsilon)$ 정확도로 보존하므로, 접속 또한 같은 차수로 보존된다. □

#### 4.4.2. 곡률 텐서의 근사

**보조정리 4.2**: 상수 곡률 $\kappa$인 공간에서, RBE의 기저 함수는 곡률 텐서를 정확히 재현한다:

$$R_{ijkl}^{\text{RBE}} = \kappa(g_{ik}g_{jl} - g_{il}g_{jk})$$

이는 RBE가 공간의 본질적인 기하학적 구조를 보존함을 의미한다.

### 4.4. 리만 기하학적 선택적 갱신 메커니즘

#### 4.4.1. 갱신 모드의 기하학적 해석

선택적 갱신 메커니즘은 리만 다양체의 기하학적 구조 변형으로 해석할 수 있다:

**정의 4.5 (전체 갱신)**: 다양체의 계량 텐서를 광범위하게 변형시키는 과정

$$\frac{\partial g_{ij}(x,t)}{\partial t} = \eta_{\text{global}} \cdot F_{ij}(x, s) \cdot \Phi(s)$$

**정의 4.6 (국소 갱신)**: 다양체의 특정 영역에서만 계량 텐서를 변형시키는 과정

$$\frac{\partial g_{ij}(x,t)}{\partial t} = \eta_{\text{local}} \cdot F_{ij}(x, s) \cdot \Phi(s) \cdot e^{-\frac{d_{\mathcal{M}}(x, x_s)^2}{2\sigma^2}}$$

여기서:
- $F_{ij}(x, s)$는 자극 $s$에 의한 계량 텐서 변화의 방향과 크기를 결정하는 함수
- $d_{\mathcal{M}}(x, x_s)$는 자극의 표상 $x_s$와 위치 $x$ 사이의 측지 거리
- $\sigma$는 갱신의 공간적 범위를 제어하는 파라미터

#### 4.4.2. 신경조절물질의 리만 기하학적 영향

신경조절물질은 리만 다양체의 계량 텐서 변화 속도와 패턴에 영향을 미치는 인자로 모델링할 수 있다:

$$\eta_{\text{mode}} = \eta_0 \cdot f_{NA}([NA]) \cdot f_{DA}([DA]) \cdot f_{ACh}([ACh])$$

여기서:
- $[NA]$, $[DA]$, $[ACh]$는 각각 노르아드레날린, 도파민, 아세틸콜린의 농도
- $f_{NA}$, $f_{DA}$, $f_{ACh}$는 각 신경조절물질의 영향 함수

**정리 4.4 (신경조절물질에 의한 학습 모드 전환)**

신경조절물질 농도의 조합이 임계값을 넘을 때 학습 모드가 전환된다:

$$\text{Mode} = \begin{cases}
\text{전체 갱신} & \text{if } [NA] > \theta_{NA} \text{ and } [ACh] < \theta_{ACh} \\
\text{국소 갱신} & \text{if } [DA] > \theta_{DA} \text{ and } [ACh] > \theta_{ACh} \\
\text{유지} & \text{otherwise}
\end{cases}$$

#### 4.4.3. 인덱스 효율성의 리만 기하학적 정식화

인덱스 효율성은 리만 다양체 관점에서 다음과 같이 재정식화할 수 있다:

$$E_i = \frac{I_{\text{Fisher}}(\mathcal{M}_H) \cdot \sqrt{\det(g_{\mathcal{M}_H})}}{\text{dim}(\mathcal{M}_H) \cdot E_c}$$

여기서:
- $I_{\text{Fisher}}(\mathcal{M}_H)$는 해마 다양체의 피셔 정보 행렬
- $\det(g_{\mathcal{M}_H})$는 계량 텐서의 행렬식으로, 다양체의 체적 요소를 나타냄
- $\text{dim}(\mathcal{M}_H)$는 해마 다양체의 차원
- $E_c$는 에너지 소비

**정리 4.5 (최적 인덱싱의 정보 이론적 한계)**

주어진 에너지 제약 하에서 최적 인덱싱이 달성할 수 있는 정보 보존률의 상한:

$$I_{\text{preserved}} \leq \frac{E_c \cdot \text{dim}(\mathcal{M}_H)}{\sqrt{\det(g_{\mathcal{M}_H})}} \cdot \log\left(1 + \frac{\text{SNR}}{\kappa(\mathcal{M}_H)}\right)$$

여기서 $\kappa(\mathcal{M}_H)$는 해마 다양체의 조건수이다.

### 4.5. RBE의 신경과학적 타당성

#### 4.5.1. 해마-피질 상호작용 모델

RBE의 압축-복원 메커니즘은 해마-피질 시스템의 기능과 놀라운 유사성을 보인다:

**대응 관계**:
- 비트필드 인코딩 ↔ 해마의 인덱싱
- 기저 함수 ↔ 피질의 표상
- 압축된 코드 ↔ 해마의 희소 표상
- 복원 과정 ↔ 패턴 완성 (pattern completion)

#### 4.5.2. 그리드 셀과 쌍곡 기하학

내후각피질의 그리드 셀은 육각형 격자 패턴의 발화를 보이는데, 이는 쌍곡 공간의 테셀레이션과 수학적으로 동형이다:

**정리 4.6 (그리드 셀의 쌍곡 표현)**

그리드 셀의 발화 패턴 $\phi(x)$는 쌍곡 공간의 라플라시안 고유함수로 근사할 수 있다:

$$\phi(x) \approx \sum_{k} a_k \psi_k^{\mathbb{H}}(x)$$

여기서 $\psi_k^{\mathbb{H}}$는 쌍곡 라플라시안의 고유함수이다.

#### 4.5.3. 시냅스 가소성과 리만 계량 업데이트

헤비안 가소성 규칙은 리만 계량의 국소적 업데이트로 해석될 수 있다:

$$\Delta g_{ij} = \eta \cdot \left(\langle \nabla_i \phi, \nabla_j \phi \rangle - \lambda g_{ij}\right)$$

여기서:
- $\phi$는 신경 활성화 함수
- $\nabla_i$는 공변 미분
- $\lambda$는 정규화 파라미터

이는 RBE의 학습 과정에서 기저 함수가 업데이트되는 방식과 수학적으로 동일한 구조를 가진다. 