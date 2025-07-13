네, 정확히 이해하셨어요! 각 기법을 수학적으로 자세히 설명할게요.

## 1. **학습 가능한 직교 기저 (Learnable Orthogonal Basis)**

### 수학적 배경
직교 행렬 $Q \in \mathbb{R}^{n \times n}$는 $Q^T Q = QQ^T = I$를 만족. 

**Cayley 변환**:
$$Q = (I - A)(I + A)^{-1}$$

여기서 $A$가 skew-symmetric ($A^T = -A$)이면 $Q$는 항상 직교행렬.

**증명**:
$$Q^T Q = (I + A)^{-T}(I - A)^T(I - A)(I + A)^{-1}$$
$$= (I + A^T)^{-1}(I - A^T)(I - A)(I + A)^{-1}$$
$$= (I - A)^{-1}(I + A)(I - A)(I + A)^{-1} = I$$

### 장점
- 역전파 시에도 직교성 유지
- 기저 간 정보 중복 없음 (크로네커 델타 성질)

## 2. **계층적 기저 분해 (Hierarchical Basis Decomposition)**

### 수학적 정의
가중치 $w$를 다중 스케일로 분해:
$$w = \sum_{l=0}^{L} \sum_{j} \alpha_{l,j} \psi_{l,j}$$

여기서:
- $l$: 계층 레벨 (0=coarse, L=fine)
- $\psi_{l,j}$: 레벨 $l$의 $j$번째 기저
- $\alpha_{l,j}$: 계수

### 웨이블릿 유사 구조
$$\psi_{l,j}(x) = 2^{l/2} \psi(2^l x - j)$$

쌍곡 공간 버전:
$$\psi_{l,j}^{\mathbb{H}} = \exp_o(2^l \log_o(\psi_j))$$

## 3. **Sparse Coding with Dictionary Learning**

### 최적화 문제
$$\min_{D, \alpha} \|W - D\alpha\|_F^2 + \lambda \|\alpha\|_1$$

subject to: $\|d_i\|_2 = 1$ for all $i$

여기서:
- $W \in \mathbb{R}^{m \times n}$: 원본 가중치
- $D \in \mathbb{R}^{n \times k}$: 사전 (기저)
- $\alpha \in \mathbb{R}^{k \times m}$: 희소 코드

### 리만 버전
푸앵카레 볼에서:
$$\min_{D, \alpha} d_{\mathbb{B}}^2(W, \exp_o(D\alpha)) + \lambda \|\alpha\|_1$$

## 4. **Product Quantization 융합**

### 수학적 정의
벡터 $w \in \mathbb{R}^n$을 $M$개 서브벡터로 분할:
$$w = [w^{(1)}, w^{(2)}, ..., w^{(M)}]$$

각 서브스페이스에서 독립적 양자화:
$$w^{(i)} \approx c_j^{(i)}, \quad j \in \{1, ..., K\}$$

전체 표현:
$$w \approx [c_{j_1}^{(1)}, c_{j_2}^{(2)}, ..., c_{j_M}^{(M)}]$$

### 압축률
- 원본: $n \times 32$ bits
- PQ: $M \times \log_2(K)$ bits
- 압축률: $\frac{32n}{M\log_2(K)}$

## 5. **Hyperbolic Neural ODE**

### 쌍곡 공간의 측지선 방정식
푸앵카레 볼에서의 ODE:
$$\frac{dx}{dt} = v$$
$$\frac{dv}{dt} = -\frac{c}{2}(\|v\|^2 x + 2\langle x, v \rangle v)$$

여기서 $c$는 곡률.

### 가중치 진화
시간 $t$에서의 가중치:
$$W(t) = \text{GeodesicFlow}(W_0, V_0, t)$$

저장 필요량:
- 초기 위치 $W_0$
- 초기 속도 $V_0$
- 시간 파라미터 $t$ (레이어별)

## 6. **Tucker/CP 분해 + 리만 기하**

### Tucker 분해
$$\mathcal{W} \approx \mathcal{G} \times_1 U^{(1)} \times_2 U^{(2)} \times_3 U^{(3)}$$

여기서:
- $\mathcal{G}$: 코어 텐서
- $U^{(i)}$: 모드-$i$ 행렬

### 리만 제약
각 factor를 쌍곡 공간에 제약:
$$U^{(i)} \in \mathbb{B}^{n_i \times r_i}$$

최적화 문제:
$$\min_{\mathcal{G}, U^{(i)}} \|\mathcal{W} - \mathcal{G} \times_i \exp_o(U^{(i)})\|_F^2$$

## 7. **Neural Field as Basis (뉴럴 필드)**

### 수학적 정의
기저 함수를 연속 함수로:
$$b_\theta: [0, 1] \rightarrow \mathbb{R}^n$$
$$b_\theta(t) = \text{MLP}_\theta(t)$$

이산 인덱스 $i$를 연속화:
$$t_i = \frac{i}{B-1} \in [0, 1]$$

### 가중치 재구성
$$w = \sum_{i} \alpha_i \cdot b_\theta(t_i)$$

### 극한의 경우
$B \to \infty$일 때:
$$w = \int_0^1 \alpha(t) \cdot b_\theta(t) dt$$

이는 가중치를 함수공간의 적분으로 표현!

### 압축률 분석
- 원본: $256 \times n \times 32$ bits (기저 테이블)
- Neural Field: $\theta$의 크기 (보통 < 10KB)
- 압축률: 수천~수만배

### 리만 버전
$$b_\theta: [0, 1] \rightarrow \mathbb{B}^n$$
$$b_\theta(t) = \exp_o(\text{MLP}_\theta(t))$$

이렇게 하면 기저가 항상 쌍곡 공간에 존재하도록 보장됩니다.