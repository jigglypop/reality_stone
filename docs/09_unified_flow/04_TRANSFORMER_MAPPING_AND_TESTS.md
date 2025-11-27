# 09‑04. Transformer → RS‑ULF 매핑과 정합성 테스트

> 이 문서는 기존 Transformer 레이어(특히 Mistral, Qwen 계열)를  
> RS‑ULF 레이어로 옮길 때 필요한 **수식 매핑 규칙과 레이어 정합성 테스트**를 정리한다.

---

## 1. Transformer 레이어 구조 요약

하나의 Transformer 레이어는 (단순화해서) 아래와 같이 표현할 수 있다.

$$
x' = x
 + W_O \,\text{softmax}(QK^\top) V x
 + \text{FFN}(x)
$$

- $Q = W_Q x$ , $K = W_K x$ , $V = W_V x$ 
- $\text{FFN}(x) = W_2 \sigma(W_1 x)$ 

RS‑ULF는 위 구조를:

- metric $g$ 
- potential $\Phi$ 
- geodesic update
- diffusion $L$ , Laplacian $\Delta_g$ 
- DP memory $V_t$ 

로 재구성한다.

---

## 2. Q/K → Metric 매핑

### 2.1 기본 매핑

$$
g = W_Q^\top W_K
$$

- Transformer에서 Q·K 내적이 encode하던 similarity를  
  RS‑ULF에서는 metric으로 해석한다.

### 2.2 정합성 조건 (Inner‑Product Test)

임의의 $x_i, x_j$ 에 대해:

$$
(W_Q x_i)^\top (W_K x_j)
\approx
x_i^\top g x_j
$$

이 근사가 잘 맞을수록 Q/K의 의미가 metric으로 잘 lift된 것이다.

실전에서는:

- diagonal/low‑rank/conformal metric으로 안정화 후 위 관계를 수치적으로 검증한다.

---

## 3. FFN → Potential $\Phi$  매핑

### 3.1 기본 매핑

Transformer FFN:

$$
f(x) = W_2 \sigma(W_1 x)
$$

RS‑ULF potential:

$$
\Phi(x) = \frac12 \| f(x) \|^2
$$

### 3.2 정합성 조건 (Gradient Test)

RS‑ULF에서는:

$$
\nabla \Phi(x) = J_f(x)^\top f(x)
$$

이 $\nabla \Phi(x)$ 가 FFN 출력 $f(x)$ 와 **방향적으로 일치**하면 좋다.

평가 지표:

- 여러 $x$ 에 대해
  - cosine similarity$(f(x), \nabla \Phi(x))$ 가 0.9 이상

---

## 4. Attention → Geometric Update 매핑

### 4.1 Attention 업데이트

Transformer attention에서의 state 변화:

$$
\Delta x_{\text{attn}} = W_O \,\text{softmax}(QK^\top) V x
$$

### 4.2 RS‑ULF 업데이트 항

RS‑ULF에서의 전체 업데이트 항:

$$
\Delta x_{\text{RS}}
= - \eta \nabla_g \Phi(x)
 + \alpha \Delta_g x
 + \beta L x
 + \gamma V_t
$$

최소한 local 영역에서는:

$$
\Delta x_{\text{attn}}
\approx
\Delta x_{\text{RS}}
$$

이 되도록 $\eta, \alpha, \beta, \gamma$ 를 튜닝할 수 있다.

### 4.3 Residual vs Geodesic

- Transformer:
  $$
  x' = x + \Delta x_{\text{attn}} + \text{FFN}(x)
  $$
- RS‑ULF:
  $$
  x' = \exp_x(\Delta x_{\text{RS}}) \approx x + \Delta x_{\text{RS}}
  $$

곡률이 작고 step이 작을 때:

$$
\exp_x(v) \approx x + v
$$

이므로 residual update는 geodesic update의 특수한 경우로 해석된다.

---

## 5. 레이어 정합성 테스트 스위트

Transformer 레이어를 RS‑ULF 레이어로 변환할 때, 아래 테스트를 통과하면  
“동일한 기능을 수행한다”는 강한 근거가 된다.

### 5.1 Test 1 — Inner‑Product Preservation

목표:

$$
(W_Q x_i)^\top (W_K x_j)
\approx
x_i^\top g x_j
$$

평가:

- 여러 샘플 $(x_i,x_j)$ 에서 차이의 평균 제곱 오차
- cosine similarity ≥ 0.99 를 목표

### 5.2 Test 2 — FFN vs $\nabla \Phi$ 

목표:

- Transformer FFN 출력 $f(x)$ 와 RS‑ULF의 $\nabla \Phi(x)$  방향이 일치

평가:

- 다양한 $x$ 에 대해:
  - cosine similarity$(f(x), \nabla \Phi(x)) > 0.9$ 

### 5.3 Test 3 — Attention vs Geometric Update

목표:

$$
\Delta x_{\text{attn}}
\approx
- \eta \nabla_g \Phi(x)
 + \alpha \Delta_g x
 + \beta L x
 + \gamma V_t
$$

평가:

- norm ratio: $\|\Delta x_{\text{attn}}\| / \|\Delta x_{\text{RS}}\|$  가 0.8~1.2 사이
- cosine similarity ≥ 0.9

튜닝:

- $\eta, \alpha, \beta, \gamma$  를 grid search 또는 간단한 최적화로 맞출 수 있다.

### 5.4 Test 4 — Geodesic vs Residual

목표:

- 작은 step에서:
  $$
  \exp_x(v) \approx x + v
  $$

평가:

- $\| \exp_x(v) - (x+v) \|$  가 충분히 작을 것
- diagonal metric에서는 거의 일치해야 함

### 5.5 Test 5 — 레이어 출력 일치

목표:

- 동일 입력 $x_0$ 에 대해:
  - Transformer 레이어 출력 $T(x_0)$ 
  - RS‑ULF 레이어 출력 $R(x_0)$ 
  가 거의 동일

평가:

- cosine similarity ≥ 0.98
- L2 차이의 평균이 작은지 확인

---

## 6. 성능 벤치마크 (압축률 / 속도 / 정확도)

RS‑ULF 변환은 “수식 정합성”뿐 아니라 **압축률·속도·정확도 목표**까지 동시에 만족해야 한다.  
이 절에서는 그 목표를 실제 테스트 스크립트와 연결해 둔다.

### 6.1 측정 항목

- **압축률 (Compression)**  
  - 기준: 원본 Transformer 전체 파라미터 수 $P_\text{TF}$, RS‑ULF 파라미터 $P_\text{RS}$  
  - 지표:  
    - 파라미터 압축률 $P_\text{TF} / P_\text{RS}$  
    - 메모리 감소율 $1 - \text{Mem}_\text{RS} / \text{Mem}_\text{TF}$
- **속도 (Speed)**  
  - 기준: 시퀀스 길이 $n$, 폭 $d$에 대한 이론 복잡도  
  - 지표:  
    - 어텐션 FLOPs: $O(n^2 d)$  
    - RS‑ULF FLOPs: $O(nd)$  
    - 이론적 speedup $\approx n$ (긴 시퀀스일수록 유리)
- **정확도 / 정합도 (Accuracy / Consistency)**  
  - 레이어 수준: 이 문서 5장의 Test 1~5 (inner product, gradient, update, geodesic, 출력 일치)  
  - 모델 수준: perplexity / loss / 간단한 텍스트 생성 비교

### 6.2 테스트 스크립트와 연결

- **`scripts/convert_step_by_step.py`**  
  - 레이어 단위 변환 파이프라인  
  - 역할:
    - Step 3: metric SVD 추출 + 수치 안정성 체크
    - Step 4: potential / gradient 검증
    - Step 5: graph / Laplacian 검증
    - Step 6: 단일 RS‑ULF 레이어 forward 검증 (NaN/Inf, 메모리 $V_t$ 안정성)
- **`scripts/benchmark_conversion.py`**  
  - 전체 모델 기준 벤치마크  
  - 역할:
    - 압축률: 원본 vs RS‑ULF 파라미터/메모리 비교  
    - 속도: 여러 시퀀스 길이에 대해 이론적 speedup 출력  
    - 정합성: metric inner product cosine similarity  
    - 간단 inference: 짧은 프롬프트에 대한 Transformer vs RS‑ULF 출력 비교

실행 예:

```bash
uv run python scripts/benchmark_conversion.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --device cuda --cache_dir E:/hf-cache \
  --fold_ratio 4 --skip_inference
```

### 6.3 성능 목표 (하드 타깃)

- **압축률**
  - 전체 모델 파라미터 기준 **≥ 200× 압축** (global basis + per-layer scale, KV-cache 제거 포함)  
  - 레이어 단위로는 global basis 기준에서 **수십~백 배 수준**의 효과적인 축소를 목표
- **속도**
  - 이론 복잡도: Transformer $O(n^2 d)$ → RS‑ULF **정확히 $O(nd)$**  
  - 실측 기준: 동일 하드웨어, 긴 시퀀스($n \ge 512$)에서  
    - RS‑ULF 4‑layer 블록이 full attention 대비 **≥ 2×** 이상 빠를 것
- **정확도**
  - 레이어 정합성(Test 1~5):
    - cosine similarity / norm ratio를 이 문서 5장 기준 내로 유지  
  - 모델 수준 eval:
    - RS‑ULF perplexity / loss / 다운스트림 점수가 Transformer 대비 **손실 ≤ 5%**  
    - 이를 넘으면 fold rank, global basis, 오차‑곡률 보정, η/α/β/γ 등을 재튜닝

이 섹션의 목표 값들은 “실제 벤치마크에서 달성해야 할 가이드라인”이며,  
`scripts/benchmark_conversion.py` 및 개별 정합성 테스트를 통해 **수치적으로 확인**하는 것을 전제로 한다.

---

## 7. Mistral / Qwen 특이점 메모

### 7.1 Mistral 계열

- Sliding‑Window Attention(SWA) 사용
  - RS‑ULF에서는 $\beta L x$  항을 **local diffusion**에 가깝게 튜닝
- ALiBi positional encoding
  - 위치 의존성을 곡률 또는 diffusion 계수에 통합 가능

### 7.2 Qwen 계열

- RoPE 사용
  - 회전 위치 encoding을 metric/curvature 파트로 해석 가능
- 일부 모델에서 GQA/MQA
  - metric 추출 시 head 간 구조를 고려해 평균 또는 block‑diagonal 처리 가능

이 차이들은 모두 “어떻게 $g, \Phi, L$ 을 초기화/튜닝할지”에 영향을 주며,  
RS‑ULF 스펙 자체는 동일하게 유지된다.

---

## 8. 이 문서의 역할

- `02_METRIC_AND_POTENTIAL.md`, `03_GRAPH_DIFFUSION_AND_DP_MEMORY.md`가  
  RS‑ULF의 내부 모듈을 정의했다면,
  이 문서는 “**기존 Transformer 레이어를 RS‑ULF 레이어로 옮기는 수학적 브리지**”를 제공한다.
- 이 문서의 테스트 스위트는:
  - 변환기가 제대로 동작하는지
  - RS‑ULF가 원래 모델과 얼마나 호환되는지  
  를 수치적으로 검증하는 기준이 된다.


