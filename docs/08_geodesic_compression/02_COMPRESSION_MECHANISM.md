# 2장. 압축 메커니즘 및 아키텍처 (Compression Mechanism)

## 1. 개요

이 장에서는 1장의 수학적 이론을 바탕으로 실제 압축을 수행하는 **3단계 메커니즘(Q/K 분해, FFN 분해, 지오데식 스플라인)**의 구조와 알고리즘을 상세히 정의한다.

## 2. 메트릭-게이지 분해 (Metric-Gauge Decomposition)

트랜스포머의 어텐션 가중치 행렬을 물리적 실체로 변환하는 핵심 과정이다.

### 2.1 알고리즘 상세
입력: 레이어 $l$의 가중치 $W_Q, W_K \in \mathbb{R}^{d \times d}$.

1.  **쌍선형 형상 구성**: $B = W_Q^\top W_K$.
2.  **대칭/반대칭 분리**:
    $$ g_{raw} = \frac{1}{2}(B + B^\top), \quad A_{raw} = \frac{1}{2}(B - B^\top) $$
3.  **글로벌 베이시스 투영 (Folding)**:
    전체 레이어 공통 기저 $U_{global} \in \mathbb{R}^{d \times r}$에 대해,
    $$ g_{core}^{(l)} = U_{global}^\top g_{raw}^{(l)} U_{global} $$
    $$ A_{core}^{(l)} = U_{global}^\top A_{raw}^{(l)} U_{global} $$
4.  **잔차 에너지의 곡률화**:
    $$ E_{resid} = \| g_{raw} - U g_{core} U^\top \|_F^2 $$
    $$ \kappa^{(l)} = \alpha \cdot E_{resid} $$
    (여기서 $\kappa^{(l)}$은 해당 레이어 공간의 추가 곡률 파라미터가 된다.)

### 2.2 데이터 구조
*   `GlobalBasis`: $U \in \mathbb{R}^{d \times r}$ (공유됨, 1개)
*   `LayerParams`:
    *   `metric_core`: $r \times r$ 대칭 행렬.
    *   `gauge_core`: $r \times r$ 반대칭 행렬.
    *   `curvature`: 스칼라 값.

## 3. FFN 헬름홀츠 압축 (Helmholtz Compression of FFN)

FFN의 비선형성을 에너지 보존계로 근사한다.

### 3.1 포텐셜 증류 (Potential Distillation)
입력: FFN 가중치 $W_1, W_2$.

1.  **타겟 생성**: 샘플 입력 $X$에 대해 $Y = W_2 \sigma(W_1 X)$ 계산.
2.  **포텐셜 함수 근사**:
    $$ \Phi(x) = \frac{1}{2} x^\top P x + \text{NeuralPotential}(x; \theta_{small}) $$
    여기서 $P$는 2차 근사, $\text{NeuralPotential}$은 작은 MLP다.
3.  **최적화 목적 함수**:
    $$ \mathcal{L} = \| -\nabla \Phi(x) - Y \|^2 + \lambda \| \Phi(x) - \int Y \|^2 $$
    힘(Force)과 에너지(Energy)를 동시에 맞춘다.

## 4. 지오데식 스플라인 (Geodesic Spline for KV Cache)

시퀀스 길이에 비례하는 메모리를 상수 배로 줄이는 시공간 압축 기술이다.

### 4.1 제어점 선정 (Control Point Selection)
모든 토큰 $x_t$를 저장하는 대신, 궤적의 곡률 $\kappa(t)$가 임계값을 넘는 지점 $t_k$만 선정한다.

$$ t_{k+1} = \min \{ t > t_k : \| \nabla_{\dot{x}} \dot{x} \| > \theta \} $$

### 4.2 복원 메커니즘
제어점 $(x_{t_k}, v_{t_k})$ 쌍으로부터 구간 $[t_k, t_{k+1}]$ 사이의 상태를 복원한다.

$$ x(t) \approx \text{Exp}_{x_{t_k}} \left( (t - t_k) v_{t_k} + \frac{1}{2}(t - t_k)^2 \text{Force}(x_{t_k}) \right) $$

이 방식은 선형 보간(Linear Interpolation)과 달리 물리 법칙을 따르므로, 적은 점으로도 원래의 궤적(Context Flow)을 매우 정확하게 복원한다.

