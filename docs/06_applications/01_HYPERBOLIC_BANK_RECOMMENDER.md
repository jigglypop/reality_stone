# 하이퍼볼릭 은행 계좌 추천 시스템
## 리만-벨만 인코더(Riemannian-Bellman Encoder)를 활용한 고효율 의도 및 계좌 예측

이 문서는 리만-벨만 LLM 아키텍처의 **인코더(Encoder)** 부분만을 추출하여 활용하는 은행 계좌 추천 시스템의 설계를 다룹니다. 거대 생성 모델 전체를 사용하는 대신, 인코더만 사용하여 분류 작업에서 비약적인 성능 향상과 계산 효율성을 달성하는 것을 목표로 합니다.

## 1. 동기 (Motivation)

기존의 유클리드 공간 기반 모델(예: 일반 Transformer, LightGBM)은 금융 데이터의 내재된 계층 구조를 포착하는 데 한계가 있습니다:
- **은행 코드**: 지역 은행, 시중 은행, 인터넷 은행 등 암묵적인 계층 구조를 가집니다.
- **사용자 행동(DP 특성)**: 단순한 평면 벡터 공간이 아닌 복잡한 다양체(Manifold) 위에 분포합니다.
- **계좌 번호 패턴**: 순차적인 숫자의 나열이 기하학적인 경로(Geodesic path)를 형성합니다.

우리의 목표는 다음 두 가지를 수행하는 단순하면서도 강력한 모델을 구축하는 것입니다:
1. **의도 분류 (Intent Classification)**: 거래 로그/특성을 기반으로 사용자의 의도 파악.
2. **계좌 추천 (Account Recommendation)**: 다음 거래에 사용할 가능성이 가장 높은 계좌 예측.

## 2. 핵심 전략: 리만-벨만 인코더 (Riemannian-Bellman Encoder)

텍스트 생성을 위한 거대 LLM 전체를 사용하는 대신, **리만 인코더(Riemannian Encoder)** 블록만을 추출하여 사용합니다. 이는 Reality Stone의 기하학적 딥러닝 능력과 판별 모델(Discriminator)의 효율성을 결합한 접근 방식입니다.

### 주요 구성 요소
1. **다양체 매핑 (Manifold Mapping)**: 원시 특성(75차원 DP 벡터)과 시퀀스를 하이퍼볼릭 공간($\mathbb{H}^n$)으로 투영.
2. **측지 어텐션 (Geodesic Attention)**: `geodesic_topk_attention`을 사용하여 리만 메트릭 기반으로 시퀀스의 중요한 부분에 집중.
3. **프로토타입 학습 (Prototype Learning)**: 타겟 클래스(은행)를 푸앵카레 볼(Poincaré ball) 내의 학습 가능한 프로토타입으로 표현.

## 3. 아키텍처 설계

모델은 크게 **임베딩(Embedding)**, **리만 인코딩(Riemannian Encoding)**, **기하학적 분류(Geometric Classification)** 세 단계로 구성됩니다.

### 3.1 입력 및 임베딩 레이어

#### 특성 임베딩 (DP Features)
- **입력**: 75차원 DP 특성 벡터 $x \in \mathbb{R}^{75}$.
- **처리 과정**:
  1. **SPD 변환**: 공분산 구조를 포착하기 위해 $x$를 SPD(Symmetric Positive Definite) 행렬로 매핑.
  2. **촐레스키 투영(Cholesky Projection)**: SPD 행렬을 분해하고 접공간(Tangent space)으로 투영.
  3. **지수 맵(Exponential Map)**: $\exp_x$를 통해 접공간에서 하이퍼볼릭 공간(Poincaré Ball)으로 매핑.
  
  ```python
  # 개념 코드
  spd = dp_to_spd(x)  # 학습 가능한 SPD 매핑
  h_feat = spd_to_hyperbolic(spd)  # H^n으로 투영
  ```

#### 시퀀스 임베딩 (계좌 번호 숫자)
- **입력**: 계좌 번호 숫자 시퀀스 $S = \{s_1, s_2, ..., s_L\}$.
- **처리 과정**:
  1. **하이퍼볼릭 임베딩**: 각 숫자를 $\mathbb{H}^d$ 벡터로 매핑.
  2. **하이퍼볼릭 위치 인코딩**: 뫼비우스 덧셈($\oplus_c$)을 사용하여 위치 정보 추가.
  
  ```python
  # 개념 코드
  emb = embedding(s) # 하이퍼볼릭 임베딩
  pos = position_encoding(indices)
  h_seq = emb ⊕_c pos
  ```

### 3.2 리만 인코더 블록 (Riemannian Encoder Block)

이 부분이 "리만-벨만" 아키텍처의 핵심입니다. 표준 Self-Attention을 **Geodesic Attention**으로 대체합니다.

- **측지 어텐션 메커니즘 (Geodesic Attention)**:
  내적($Q \cdot K^T$) 대신 음의 거리 제곱 또는 로렌츠 내적을 사용합니다.
  
  $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{-d_{\mathcal{M}}(Q, K)^2}{\sqrt{d_k}}\right) \otimes_{\mathcal{M}} V $$
  
  *참고: 구현 시 효율성을 위해 `geodesic_topk_attention` 커널을 사용합니다.*

- **하이퍼볼릭 피드포워드 (Hyperbolic Feed-Forward)**:
  표준 MLP 연산을 뫼비우스 선형 레이어로 대체합니다:
  
  $$ f(x) = W \otimes x \oplus b $$
  
  여기서 $\otimes$는 뫼비우스 행렬-벡터 곱셈입니다.

### 3.3 기하학적 분류 레이어

평면적인 Softmax 레이어 대신 **거리 기반 분류(Distance-based Classification)**를 사용합니다.

- **프로토타입 (Prototypes)**: 각 은행 클래스 $C_k$는 하이퍼볼릭 공간 내의 학습 가능한 프로토타입 벡터 $p_k \in \mathbb{H}^d$로 표현됩니다.
- **로짓 계산 (Logits Calculation)**: 클래스 $k$에 속할 확률은 푸앵카레 거리와 반비례합니다.
  
  $$ P(y=k|x) = \frac{\exp(-d_{\mathbb{H}}(h_{final}, p_k))}{\sum_j \exp(-d_{\mathbb{H}}(h_{final}, p_j))} $$

## 4. 구현 로드맵

### 1단계: 데이터 파이프라인
- `BankAccountDataset`을 재사용하되, 다양체 매핑을 위해 입력 데이터의 전처리(정규화 등)를 강화합니다.
- DP 특성을 통계적 다양체(Statistical Manifold) 상의 좌표로 취급합니다.

### 2단계: 인코더 구현
- `GeodesicAttention`과 `MöbiusLinear`를 결합한 `RiemannianEncoderLayer`를 구현합니다.
- $N$개의 레이어를 쌓습니다 (이 작업에는 $N=2$ 또는 $3$이면 충분할 것으로 예상).

### 3단계: 학습 전략
- **손실 함수**: 리만 교차 엔트로피 (사실상 음의 거리에 대한 Log-Softmax).
- **최적화**: 리만 Adam (RADAM)이 이상적이나, `reality_stone` 라이브러리의 특성상 일반 Adam에 적절한 그라디언트 리스케일링(또는 `retr` 연산)을 적용하여 학습합니다.

## 5. 기대 효과

1. **데이터 효율성**: 하이퍼볼릭 기하학은 계층 구조를 표현하는 데 더 적은 차원을 필요로 하므로, 적은 데이터로도 더 나은 일반화 성능을 보입니다.
2. **계층적 정확도**: 모델이 실수를 하더라도, 기하학적으로 가까운(유사한 성격의) 오답을 낼 확률이 높아져(예: 엉뚱한 은행 대신 유사한 지역 은행 예측), 실질적인 사용자 경험을 해치지 않습니다.
3. **추론 속도**: 생성 모델(Auto-regressive decoding)이 아닌 인코더 기반 분류 모델이므로, 출력 길이에 관계없이 $O(1)$의 추론 속도를 가지며 CUDA 병렬화에 최적화되어 있습니다.

---

**다음 단계**:
1. `geodesic_topk_attention` CUDA 커널 컴파일 확인.
2. `src/models/`에 `RiemannianEncoderBlock` 구현.
3. LightGBM 및 표준 Transformer 베이스라인과 성능 비교 벤치마크 실행.
