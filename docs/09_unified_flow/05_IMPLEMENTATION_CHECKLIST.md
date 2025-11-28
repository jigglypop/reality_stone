# 09‑05. RS‑ULF 구현 체크리스트

> 이 문서는 RS‑ULF를 실제 코드로 구현할 때  
> “무엇을, 어떤 순서로, 어떤 테스트와 함께” 진행할지 정리한 체크리스트다.

---

## 1. 1단계 — 코어 기하 모듈

### 1.1 해야 할 일

- Metric 추출/안정화 함수
  - $g_{\text{raw}} = W_Q^\top W_K$ 
  - diagonal/low‑rank/conformal 옵션
- Potential $\Phi(x)$ , gradient $\nabla \Phi(x)$  함수
  - $\Phi(x) = \frac12 \| W_2 \sigma(W_1 x) \|^2$ 
- Riemannian gradient $\nabla_g \Phi = g^{-1} \nabla \Phi$ 
- Geodesic step (exp map 1차 근사)

### 1.2 체크포인트

- $g$ 가 수치적으로 안정 (PD 또는 충분히 well‑conditioned)
- $\Phi(x)$ 는 항상 스칼라, autograd로 $\nabla \Phi(x)$  계산 가능
- $\Phi(x_{t+1}) < \Phi(x_t)$  비율이 충분히 높음

---

## 2. 2단계 — RS‑Lagrangian 레이어 (기본 형태)

### 2.1 해야 할 일

- 그래프, DP 메모리 없이:
  $$
  x_{t+1} = \exp_x[-\eta \nabla_g \Phi(x)]
  $$
  
  형태의 단일 레이어 구현

### 2.2 체크포인트

- 단일 레이어에서 FFN+residual과 비슷한 업데이트를 보이는지
- 작은 벡터 공간/장난감 모델에서 수렴 패턴 확인

---

## 3. 3단계 — 그래프 디퓨전 및 Laplacian 추가

### 3.1 해야 할 일

- 방향 그래프 $G=(V,E)$  정의, 인접행렬 $A$ 와 라플라시안 $L=D-A$  생성
- 단순 Laplacian $\Delta_g x \approx x - \bar x$  추가
- RS‑Lagrangian 업데이트에 $\alpha \Delta_g x + \beta Lx$  항 추가

### 3.2 체크포인트

- diffusion만 켠 상태에서
  - 에너지가 안정적으로 감소하는지
  - 과도한 oversmoothing이 없는지
- 장난감 그래프에서 analytic 해와 수치해 비교

---

## 4. 4단계 — Bellman/DP Memory 통합

### 4.1 해야 할 일

- DP memory:
  $$
  V_t = \gamma V_{t-1} + \Phi(x_t)
  $$
  
- RS‑Lagrangian + diffusion 업데이트에 $\gamma V_t$  항 추가

### 4.2 체크포인트

- 긴 시퀀스에서 $V_t$ 가 overflow/underflow 없이 안정
- synthetic long‑range benchmark(needle‑in‑haystack 등)에서 Transformer와 유사한 성능

---

## 5. 5단계 — RS‑ULF Unified 레이어 완성

### 5.1 해야 할 일

- 전체 업데이트:
  $$
  x_{t+1}
  = \exp_x \big[
    -\eta \nabla_g \Phi(x)
    + \alpha \Delta_g x
    + \beta Lx
    + \gamma V_t
  \big]
  $$
  
- 모듈들을 하나의 레이어 클래스로 묶기

### 5.2 체크포인트

- 각 항의 scale이 균형
- 수치 안정성 (NaN/inf 없음)
- 작은 모델에서 학습이 실제로 수렴하는지

---

## 6. 6단계 — Transformer 가중치 변환기

### 6.1 해야 할 일

- Mistral/Qwen 등에서 레이어별 가중치 추출:
  - $W_Q, W_K, W_1, W_2$  (필수)
- 각 레이어에 대해:
  - metric $g = W_Q^\top W_K$  추출/안정화
  - potential $\Phi$  정의
  - RS‑ULF 레이어 인스턴스 생성

### 6.2 체크포인트

- 레이어별 정합성 테스트:
  - Inner‑product test
  - Gradient test
  - Attention vs geometric update test
- 전체 모델 기준으로:
  - 소규모 데이터셋에서 perplexity/로스 비교

---

## 7. 7단계 — 차원 Folding 및 Metric Upgrade

### 7.1 해야 할 일

- SVD 기반 차원 축소:
  - $W_Q \approx U\Sigma V^\top$ , 상위 $k$  singular value만 유지
- anchor metric $g_*$  정의, 레이어별 metric을 anchor 기준으로 정렬

### 7.2 체크포인트

- folding 전후:
  - 레이어 출력 차이가 허용 범위 내
  - 정합성 테스트 지표(특히 cosine similarity)가 큰 변화 없음
- 복잡도/메모리 감소 효과 수치 확인

---

## 8. 8단계 — 통합 정합성 및 벤치마크

### 8.1 해야 할 일

- End‑to‑end:
  - 동일 입력에 대해 Transformer vs RS‑ULF 전체 모델 출력 비교
  - 다양한 벤치마크(MMLU, GSM8K, long‑range 등)에서 성능 측정
- 퀀트/시계열:
  - multivariate forecasting, volatility 예측, regime detection 등 테스트

### 8.2 체크포인트

- 주요 벤치마크에서:
  - Transformer 대비 정확도 손실이 허용 범위 이내
  - 시간/메모리 사용량이 목표 비율만큼 감소

---

## 9. Fold Consistency Verification (추가)

### 9.1 Metric Consistency

| 검증 항목 | 수식 | 통과 기준 |
|----------|------|-----------|
| 폴드 정확도 | $\sum_{i=1}^{r} \sigma_i^2 / \|G\|_F^2$ | $\geq 0.90$ |
| 조건수 | $\sigma_1 / \sigma_r$ | $< 10^6$ |
| 대각 양정치 | $g_{ii} > 0$ | 모든 $i$ |

### 9.2 Gradient Consistency

- FFN 출력과 Potential gradient 방향 일치:
  $$
  \cos(f(x), \nabla\Phi(x)) > 0.9
  $$

### 9.3 Curvature Correction

- 잔차 곡률: $\kappa = \sqrt{\sum_{i>r} \sigma_i^2}$
- 보정항 적용 조건: $|\kappa| > 10^{-6}$

자세한 수학적 기초는 `07_FOLD_CONSISTENCY.md` 참조.

---

## 10. 문서 간 역할 분리 요약

- `01_RS_UNIFIED_FLOW_SPEC.md`  
  - 전체 수식/스펙/모듈 정의
- `02_METRIC_AND_POTENTIAL.md`  
  - metric, potential, geodesic step 세부 스펙
- `03_GRAPH_DIFFUSION_AND_DP_MEMORY.md`  
  - 그래프 디퓨전, Laplacian, DP 메모리 스펙
- `04_TRANSFORMER_MAPPING_AND_TESTS.md`  
  - Transformer → RS‑ULF 매핑, 정합성 테스트 정의
- `05_IMPLEMENTATION_CHECKLIST.md` (이 문서)  
  - 위 모든 것을 **실제 구현 단계**로 재정렬한 체크리스트
- `07_FOLD_CONSISTENCY.md` (신규)  
  - 폴드 정합성, 메트릭 정확성, 곡률 해석의 수학적 기초

이 체크리스트를 따라가면:

1. 코어 수학 모듈
2. RS‑ULF 레이어
3. Transformer 가중치 변환기
4. folding/metric upgrade
5. 전체 벤치마크  

까지 순서대로 구현할 수 있다.


