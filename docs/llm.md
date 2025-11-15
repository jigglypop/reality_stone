## 리만 기하학 기반 LLM 설계 개요

### 목표

- **문제의식**: 기존 LLM은 주로 유클리드 공간과 통계적 최적화에 의존하여, 뇌/생물학이 보여주는 **기하학적·위계적 처리**와는 거리가 있다.
- **지향점**: 하이퍼볼릭/리만 기하를 사용해
  - **(1) 위계 구조(기관 → 조직 → 세포 / 개념 트리)** 를 자연스럽게 표현하고,
  - **(2) 컨텍스트·보안 키에 따라 메트릭을 스위칭하며,
  - **(3) 제어·의학 데이터에 대해 “예측 + 제어 가능성”을 동시에 보는 LLM**
  을 설계한다.

---

### 1. 수학적 기반 (요약 수식)

- **리만 다양체**
  - 입력/표현 공간을 리만 다양체 \((\mathcal{M}, g)\) 로 본다.
  - 여기서 \(g_x\) 는 점 \(x \in \mathcal{M}\) 에서의 내적(메트릭)으로, SPD 행렬로 표현 가능:
    $$
    g_x(u, v) = u^\top G_x v, \quad G_x \succ 0.
    $$

- **SPDMetric (지금 코드 기준)**
  - 헤드 차원 \(d_h\) 에 대해
    $$
    G = \text{diag}(\text{softplus}(d)) + U U^\top
    $$
    로 파라미터화된 SPD 메트릭을 사용한다.
  - 이는 attention 전에 쿼리/키에 적용되는 선형 변환 \(L\) (Cholesky factor)로 구현:
    $$
    q' = L q,\quad k' = L k.
    $$

- **지오데식 거리 기반 어텐션**
  - Poincaré/Lorentz/Klein 모델에서 두 점 \(x, y \in \mathcal{M}\) 사이의 거리 \(d_{\mathcal{M}}(x,y)\) 를 사용:
    $$
    s_{ij} = -\frac{1}{\tau} \, d_{\mathcal{M}}(q'_i, k'_j)^2,
    \quad
    \alpha_{ij} = \text{Normalizer}(s_{ij}),
    $$
    여기서 \(\tau\) 는 온도(temperature), Normalizer는 softmax/sparsemax/sinkhorn 등.

- **Top-k / 토폴로지 인덱스**
  - 전체 \(T \times T\) 모든 쌍이 아니라,
    $$
    \text{idx} \in \mathbb{N}^{B \times T \times K}
    $$
    로 정의된 토폴로지(예: 시간/행/열/그래프 이웃) 상의 상위 \(K\) 이웃만 attention에 사용:
    $$
    \alpha_{i, j} = 0 \quad \text{if } j \notin \text{Top-}K(i).
    $$

---

### 2. RCE-Transformer 아키텍처 (설계 초안)

- **기본 블록**
  - **입력 임베딩**: 토큰/신호 \(x_t\) 를 \(\mathbb{R}^{d}\) 로 매핑 (embedding).
  - **머리 분할**: \(d = H \cdot d_h\), 각 head마다 별도 메트릭/지오데식 attention.
  - **MetricAttention**:
    - `mode="geodesic"`, `manifold ∈ {poincare, lorentz, klein}`,
    - `SPDMetric`으로 \((Q, K)\)를 재스케일/회전 후, 거리 기반 Top‑k attention 수행.
  - **FFN/잔차**: 표준 Transformer FFN + residual + normalization (필요 시 리만 정규화 포함).

- **계층 구조**
  - 여러 RCE 블록을 쌓아 **RCE-Transformer** 를 구성:
    - 하위 블록: 시계열/로우 레벨 신호 (예: EEG, 바이탈).
    - 상위 블록: 위키/EMR 텍스트, 진단 코드, 지식 그래프 등.
  - 각 층마다 manifold/곡률/메트릭을 다르게 두어,
    - 하위 층은 시공간/동역학(Lorentz),
    - 상위 층은 개념/위계(Poincaré)를 담당하도록 설계 가능.

---

### 3. Metric-key 컨텍스트 스위칭 / 보안

- **아이디어**
  - 메트릭을 “컨텍스트 키”로 사용하여, 같은 입력이라도 **다른 메트릭 키**를 주면 **다른 geometry, 다른 attention, 다른 출력**이 나오게 한다.
  - 예:
    - `metric_keys = ["patient:123", "mode:sleep", "drug:X"]`
    - `metric_keys_b = ["patient:123", "mode:awake"]`, `alpha` 로 혼합.

- **수식 스케치**
  - 메트릭 키 집합 \(\{k_m\}\) 에 대해 SPD 메트릭 \(G_m\)를 구성:
    $$
    G_m = \text{MetriKey}(k_m) \in \mathbb{R}^{d_h \times d_h},\quad G_m \succ 0.
    $$
  - 두 컨텍스트를 가중합:
    $$
    G' = \alpha G_1 + (1-\alpha) G_2,\quad \alpha \in [0,1],
    $$
  - 그리고 \(G' = L L^\top\) 로 factorization 한 뒤, attention에 사용:
    $$
    q' = L q,\quad k' = L k.
    $$

- **보안/제어 관점**
  - 올바른 키 조합에서만 \(G'\) 이 “의미 있는 geometry”를 형성하도록 만들면,
    - 잘못된 키에서는 attention이 붕괴 → 출력이 의미 없음.
  - 이는 **암호키/접근 권한을 메트릭 공간의 구조로 인코딩** 하는 방식으로 확장 가능.

---

### 4. 뇌/의학 데이터와의 대응 (개념 레벨)

- **뇌의 기하 연산 가설**
  - 동물/인간의 뇌는 야생 환경에서 **위치·방향·위협·먹이** 등을 빠르게 파악해야 했고,
    - grid cell, head direction cell 등이 보여주듯 **저차원 매니폴드 상의 기하 연산**을 수행하는 증거가 축적되고 있다.
  - 따라서 고차원 유클리드 벡터보다, **하이퍼볼릭/리만 공간 상의 representation**이 뇌 메커니즘에 더 가깝다는 가설은 정합성이 있다.

- **의학/제어 데이터 매핑**
  - 예:
    - **시계열 생체 신호** (EEG, HRV, 호흡 등): Lorentz manifold 기반 동역학 레이어.
    - **진단 코드/질병 위계**: Poincaré embedding으로 트리/그래프 구조 표현.
    - **약물/치료 프로토콜**: 메트릭 키로 인코딩하여, 같은 입력이라도 치료 모드에 따라 다른 geometry 사용.
  - 이 위에 RCE-Transformer를 올리면, **“데이터 → manifold → 제어 가능한 representation”** 흐름을 만들 수 있다.

---

### 5. 학습 목표와 로스 설계 (제어 관점 포함)

- **표준 LLM 로스**
  - Next-token cross-entropy:
    $$
    \mathcal{L}_{\text{LM}} = -\sum_{t} \log p_\theta(x_t \mid x_{<t}).
    $$

- **기하 정규화 로스**
  - SPD 메트릭, 곡률, geodesic 길이에 대한 제약:
    - **메트릭 안정성**: \(G\) 의 eigenvalue가 \([\lambda_{\min}, \lambda_{\max}]\) 안에 머물도록:
      $$
      \mathcal{L}_{\text{metric}} = \sum \text{penalty}(\lambda_i(G)).
      $$
    - **곡률 범위**: 레이어별 \(c\) 가 물리적으로 의미 있는 범위 내에 있게:
      $$
      \mathcal{L}_{c} = \sum \text{penalty}(c_\ell; c_{\min}, c_{\max}).
      $$

- **제어/의학 로스 (스케치)**
  - 특정 타겟 지표(예: 바이탈 안정 구간, 발작 가능성, 수면 단계 전환 등)를 예측/제어하는 경우,
    - **예측 오차**: \(\mathcal{L}_{\text{pred}} = \| \hat{y} - y \|^2\)
    - **제어 안정성**: 입력/노이즈에 대한 출력 민감도를 제한하는 regularizer
      (Lipschitz 상수 upper bound 근사 등).
  - 최종 로스:
    $$
    \mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda_{\text{metric}}\mathcal{L}_{\text{metric}}
                 + \lambda_{c}\mathcal{L}_c + \lambda_{\text{pred}}\mathcal{L}_{\text{pred}} + \cdots
    $$

---

### 6. 이 리포지토리 기준 구현 로드맵 (요약)

- **1단계: TinyRCE → RCE-Block 일반화**
  - `tests/rce_ko_rce.py` 의 `TinyRCE` 를 일반화하여,
    - 임의 길이 시퀀스,
    - multi-layer,
    - residual/FFN 포함한 **RCE 블록** 클래스로 분리.

- **2단계: RCE-Transformer 스켈레톤**
  - `python/reality_stone/models/` 에
    - `rce_transformer.py` (가칭) 를 추가하고,
    - 토큰 임베딩 + N개의 RCE 블록 + 출력 projection 을 갖는 기본 LLM 구조 구현.
  - 기존 GPT2 변형(`gpt2_metric.py`)과의 비교/호환 고려.

- **3단계: Metric-key / 메트릭 스위칭 통합**
  - `MetricAttention`의 `metric_keys`, `masses`, `alpha` 를
    - 모델 외부에서 제어 가능한 인터페이스로 노출 (예: `forward(..., metric_ctx=...)`).
  - simple 예제:
    - 동일 문장을 `mode="baseline"`, `mode="sleep"`, `mode="alert"` 등으로 주고 geometry 차이를 시각화.

- **4단계: 의료/제어 데이터에 특화된 실험**
  - 시계열 + 텍스트 혼합 입력에 대해
    - baseline Transformer vs RCE-Transformer 비교.
  - 특정 제어타겟(예: event 발생 확률 억제)을 둔 로스/평가 설계.

- **5단계: 이론/안정성 분석**
  - 작은 차원/단순 구조에서
    - geodesic attention의 Lipschitz 상한,
    - SPD 파라미터/곡률에 따른 안정성 조건
    를 계산/증명하여, “제어 가능한 LLM”의 기초를 다진다.

---

이 문서는 **“리만 메트릭 컨텍스트 스위칭 + 물리 위계 저장법 + 메트릭 기반 보안 키”** 라는 아이디어를, 현재 `reality_stone` 코드베이스 위에 어떻게 LLM 구조로 구체화할지에 대한 1차 설계 초안이다.  

---

### 7. Rust 코어 설계 정교화 (LLM용)

여기서는 Python 레이어가 아닌 **Rust 코어 관점**에서, RCE-Transformer를 뒷받침할 구조를 정리한다. (실제 구현은 별도 단계에서 진행)

- **7.1 모듈 분리 계획**
  - `src/layers/riemann.rs`:
    - 현재 존재하는 Riemann low-rank 연산을 정리하고,
    - **공용 Riemann 연산 모듈**로 확장: `exp/log`, `distance`, `parallel transport` 등.
  - `src/ops/metrikey.rs`:
    - SPD 메트릭 합성/Cholesky factor 계산을 LLM에서 재사용할 수 있도록,
    - `metric_from_keys(keys, masses, dim, min_lambda, max_lambda)` 와 같은 함수군을 정교화.
  - **신규 모듈 제안**:
    - `src/ops/geodesic_attention.rs` (또는 `layers/geodesic_attn.rs`):
      - Q,K,V와 topo 인덱스를 받아 **지오데식 Top‑k attention**을 계산하는 순수 Rust 커널.
    - `src/bindings/rce.rs`:
      - 위 커널들을 PyO3로 노출하는 thin binding (Python에서 `MetricAttention`이 직접 호출).

- **7.2 공통 Trait 설계 (개념 레벨)**
  - `Manifold` trait (Poincaré/Lorentz/Klein 공통 인터페이스):
    ```rust
    pub trait Manifold {
        fn exp(&self, x: &Array2<f32>, v: &Array2<f32>, c: f32) -> Array2<f32>;
        fn log(&self, x: &Array2<f32>, y: &Array2<f32>, c: f32) -> Array2<f32>;
        fn distance(&self, x: &Array2<f32>, y: &Array2<f32>, c: f32) -> Array2<f32>;
    }
    ```
    - 구현체: `PoincareManifold`, `LorentzManifold`, `KleinManifold` 가 각각 `bindings::poincare/lorentz/klein` 와 연결.
  - `GeodesicAttentionKernel` (RCE 전용):
    ```rust
    pub trait GeodesicAttentionKernel<M: Manifold> {
        fn forward_topk(
            &self,
            q: &Array4<f32>,   // [B,H,T,d_h]
            k: &Array4<f32>,   // [B,H,S,d_h]
            v: &Array4<f32>,   // [B,H,S,d_v]
            topo_idx: &Array3<i64>, // [B,T,K]
            spd_metric: &Array2<f32>, // [d_h,d_h], optional (I if none)
            c: f32,
        ) -> Array4<f32>;     // [B,H,T,d_v]
    }
    ```
    - 내부에서:
      - SPD 메트릭 factor \(L\) 적용 (필요 시),
      - \((q',k')\)에 대해 거리를 계산하고,
      - Top‑k score/softmax/집계까지 수행.

- **7.3 FFI/바인딩 레벨 스펙**
  - PyO3로 노출할 함수 예시 (`src/bindings/rce.rs`):
    ```rust
    #[pyfunction]
    pub fn geodesic_topk_attention_poincare(
        q: PyReadonlyArrayDyn<f32>,   // [B,H,T,d_h]
        k: PyReadonlyArrayDyn<f32>,   // [B,H,S,d_h]
        v: PyReadonlyArrayDyn<f32>,   // [B,H,S,d_v]
        topo_idx: PyReadonlyArrayDyn<i64>, // [B,T,K]
        l_factor: Option<PyReadonlyArray2<f32>>, // SPD Cholesky, optional
        c: f32,
    ) -> Py<PyArray4<f32>> { ... }
    ```
    - Python 쪽 `MetricAttention`에서는, 현재 pure Python 구현을 유지하되 **옵션으로 Rust 커널을 호출**:
      - small model / 디버깅: Python 경로.
      - large LLM / production: Rust 경로 (`if _has_rust_ext: ...`).

- **7.4 성능/제어 관점의 Rust 책임 분리**
  - **성능**:
    - 거리 계산/Top‑k/집계는 Rust(+CUDA)에서 처리하여, Python 오버헤드를 제거.
    - 특히 LLM 스케일에서는 \(B,H,T,K\) 루프를 모두 Rust에서 처리하는 것이 필수.
  - **제어/안정성**:
    - Lipschitz 상한, SPD eigenvalue 범위, 곡률 범위 등을 계산/체크하는 유틸 함수를
      `src/ops/curvature.rs` 및 `src/ops/metrikey.rs` 에 추가하여,
      - Python 쪽에서 “이 레이어 설정이 제어 관점에서 안전한가?” 를 검사할 수 있게 한다.

- **7.5 구현 단계 요약 (Rust 한정)**
  1. `Manifold` trait 도입 및 기존 poincare/lorentz/klein 코드 정리 (interface 통일).
  2. `GeodesicAttentionKernel` trait 및 `geodesic_attention.rs` 구현 (CPU 기준).
  3. PyO3 바인딩 (`bindings/rce.rs`) 추가 및 Python `MetricAttention`에서 선택적으로 사용.
  4. 필요 시 CUDA 커널로 확장 (기존 `layers/cuda/*.cu` 구조 재사용).
  5. 제어/안정성 분석용 helper 함수들을 Rust 쪽 ops 모듈에 추가.

이 섹션은 “Rust 코어를 어떻게 정교하게 쪼개고, RCE‑Transformer/LLM을 뒷받침할지”에 대한 설계 초안이다.  
구현 단계에서는 위 trait/함수 시그니처를 기준으로, 작은 단위부터 점진적으로 Rust 코드를 옮겨가는 전략을 취한다.

---

## 8. 문장·주제 지시형 LLM (Sentence-Topic Guided LLM)

### 8.1 요구사항 정리

- **동작 순서**: 문단 입력 → 문장 단위 분해 → 각 문장 주제/역할 판정 → 주제/역할 따라 문장 재배열 또는 선택 → 단어 교체만 허용하는 재생성.
- **제약**:
  - reality_stone API (`MetricAttention`, `metrikey`, `PoincareEmbedding`, `gpt2_metric`)을 직접 활용.
  - 모든 attention은 geodesic metric 기반 Top‑k 제약을 따른다.
  - 단어 교체(lexical substitution) 단계에서 원문 토큰 집합을 강하게 참조해 의미 변형을 최소화한다.
- **평가 기준**: Postman으로 호출 가능한 API(`POST /sentence_topic_rewrite`)를 제공하고, 응답에는 주제별 문장 score, 교체 후보, 최종 문장을 포함한다.

### 8.2 계층적 구조 개요

| 층위 | 명칭 | 책임 | manifold/metric | reality_stone 모듈 |
|------|------|------|-----------------|--------------------|
| L0 | Pre-Segmenter | 문단→문장 토큰화, 길이/위치 메타 생성 | Euclidean | 사용자 파서 + `torchtext` |
| L1 | SentenceTopicHead | 문장 임베딩, 주제 확률, 우선순위 산출 | Poincaré (곡률 \(c_p < 0\)) | `layers.poincare_embedding`, `layers.metric_attention` |
| L2 | Metric Context Router | metric_keys를 주제/보안 태그로부터 생성, SPD 합성 | SPD manifold | `reality_stone.metrikey`, `layers.lowrank.SPDMetric` |
| L3 | RCE-LexicalDecoder | 문장별 토큰 시퀀스를 geodesic attention으로 복원, 단어 교체만 허용 | Lorentz (역동성), Klein (정규화) 혼합 | `models.gpt2_metric`, `layers.metric_attention`, `_rust.geodesic_topk_attention_*` |
| L4 | Post-Controller | 원문/출력 비교, 교체 로그, API 응답 구성 | Euclidean | Python glue + reality_stone utils |

---

## 9. SentenceTopicHead 상세

### 9.1 입력/출력 정의

- 입력: 문장 임베딩 \(X \in \mathbb{R}^{B \times T \times d}\), 문장 위치(토폴로지) \( \text{topo} \in \mathbb{N}^{B \times T \times K} \).
- 출력:
  - 주제 확률 \(P_{\text{topic}} \in \Delta^{B \times T \times C}\).
  - 문장 우선순위 score \(s \in \mathbb{R}^{B \times T}\).
  - metric context seed \(k_m\) (문장별 키 문자열).

### 9.2 연산

1. `PoincareEmbedding`:  
   \( z_t = \text{Exp}_0(x_t; c_p) \in \mathbb{D}^{d_h} \).
2. 지오데식 거리 기반 score:
   $$
   d_{ij} = d_{\text{poincare}}(z_i, z_j; c_p),
   \quad s_i = -\frac{1}{\tau_1} \sum_{j \in \text{Top-}K(i)} d_{ij}
   $$
3. 주제 분류:
   $$
   P_{\text{topic}}(i,c) = \text{softmax}_c\left( w_c^\top \log_{a_c}(z_i) \right)
   $$
   여기서 \(a_c\) 는 주제별 볼록 앵커.
4. metric key 생성:  
   `metric_keys[i] = f_topic(P_topic[i], meta_i)` (예: `"topic:diagnosis|style:formal"`).

### 9.3 구현 포인트

- `python/reality_stone/models/sentence_topic_head.py` 신설 (사용자 승인 후).  
- 내부에서 `MetricAttention(mode="geodesic", manifold="poincare")`를 호출해 score/Top‑k을 계산.  
- SPD 안정성을 위해 `layers.lowrank.SPDMetric`을 identity 초기화 후 fine-tune.

---

## 10. Metric Context Router

### 10.1 SPD 합성

- metric key 집합 \(\mathcal{K}_i = \{k_{i1}, \ldots, k_{im}\}\) 에 대해:
  $$
  G_i = \sum_{k \in \mathcal{K}_i} \alpha_{ik} \cdot \text{MetriKey}(k, d_h, \lambda_{\min}, \lambda_{\max})
  $$
  - \(\alpha_{ik} = \text{softmax}(\beta \cdot s_{ik})\) 는 SentenceTopicHead score에서 유도.
  - `MetriKey`는 reality_stone Python 바인딩에서 SPD 행렬과 Cholesky factor \(L_i\) 을 반환.

### 10.2 Router API

```python
from reality_stone import metrikey

class MetricContextRouter:
    def __call__(self, metric_keys, scores):
        spd = metrikey.metric_from_keys(metric_keys, dim=d_h,
                                        min_lambda=0.1, max_lambda=5.0,
                                        masses=scores.softmax(-1))
        return spd.cholesky()  # L_i
```

- 반환된 \(L_i\) 는 L3의 RCE-LexicalDecoder에 주입되어 \(q' = L_i q, k' = L_i k\) 로 활용.

---

## 11. RCE-LexicalDecoder

### 11.1 구조

- 베이스: `python/reality_stone/models/gpt2_metric.GPT2MetricModel`.
- 수정 사항:
  - `forward(input_ids, metric_ctx, replacement_mask, topo_idx, ...)`.
  - Attention 블록을 `MetricAttention(mode="geodesic", manifold="lorentz")` 로 교체.
  - `_rust.geodesic_topk_attention_*` 커널을 우선 사용하고, 없으면 Python 구현 fallback.

### 11.2 단어 교체 제약

1. 원문 토큰 \(x_t\) 에 대해 후보 집합 \(C_t\) 를 사전에 정의 (동의어, 사용자 사전, cosine 근접 등).
2. geodesic score로 후보 재정렬:
   $$
   \tilde{s}_{tc} = -\frac{1}{\tau_2} d_{\text{lorentz}}\left( q'_t, k'_c; c_l \right)^2
   $$
3. softmax를 후보에만 적용:
   $$
   p(y_t = c \mid x_{<t}) =
   \begin{cases}
   \frac{\exp(\tilde{s}_{tc})}{\sum_{c' \in C_t} \exp(\tilde{s}_{tc'})} & c \in C_t \\
   0 & \text{otherwise}
   \end{cases}
   $$
4. 최종 토큰 선택은 `torch.multinomial` 혹은 argmax로 결정하되, `replacement_mask` 가 0인 위치는 원문 토큰을 유지.

### 11.3 잔차/정규화

- Lorentz attention 결과를 Klein chart로 투영하여 FFN에 공급:
  \( h^{(\text{klein})}_t = \text{project\_to\_klein}(h^{(\text{lorentz})}_t) \).
- 표준 `RMSNorm` 대신 `riemann_lowrank` 모듈의 norm을 사용해 곡률 일관성을 유지.

---

## 12. 파이프라인 & API 설계

### 12.1 데이터 플로우

1. `POST /sentence_topic_rewrite` 요청 수신.
2. Pre-Segmenter가 문단을 문장 리스트와 토큰 텐서로 변환.
3. `SentenceTopicHead` 호출 → \(P_{\text{topic}}, s, metric\_keys\).
4. `MetricContextRouter(metric_keys, s)` → SPD \(L_i\).
5. `RCE-LexicalDecoder` 를 문장별로 호출, `topo_idx`는 문장 순서를 반영.
6. Post-Controller가 원문 대비 교체 비율, 주제 유지 여부를 계산하고 응답 JSON 구성.

### 12.2 reality_stone 연동 요약

- `reality_stone.layers.metric_attention.MetricAttention` : 모든 attention 블록.
- `reality_stone.layers.poincare_embedding.PoincareEmbedding` : 문장 임베딩.
- `reality_stone.metrikey.metric_from_keys` : metric routing.
- `reality_stone.layers.lowrank.SPDMetric` : 안정적 SPD 파라미터화.
- `reality_stone._rust.geodesic_topk_attention_*` : 고속 Top‑k attention (선택).
- `reality_stone.layers.klein.project_to_klein` : manifold간 변환.

---

## 13. 학습 및 평가 계획

### 13.1 손실 함수

- 문장 주제 정합:
  \( \mathcal{L}_{\text{topic}} = \text{CE}(P_{\text{topic}}, y_{\text{topic}}) \).
- 문장 우선순위:
  \( \mathcal{L}_{\text{order}} = \| \sigma(s) - y_{\text{order}} \|_2^2 \).
- 단어 교체 제약:
  \( \mathcal{L}_{\text{lex}} = \sum_t \text{KL}(p_t \parallel \hat{p}_t) + \lambda_{\text{copy}} \|y_t - x_t\|_1 \cdot (1 - m_t) \).
- 전체:
  $$
  \mathcal{L} =
  \mathcal{L}_{\text{LM}} +
  \lambda_{\text{topic}}\mathcal{L}_{\text{topic}} +
  \lambda_{\text{order}}\mathcal{L}_{\text{order}} +
  \lambda_{\text{lex}}\mathcal{L}_{\text{lex}} +
  \lambda_{\text{metric}}\mathcal{L}_{\text{metric}} +
  \lambda_{c}\mathcal{L}_{c}.
  $$

### 13.2 데이터셋/전처리

- 뉴스/의료 보고서 등의 문단 데이터를 선택하여 문장 주제 라벨을 수동/모델 기반으로 생성.
- 동의어/대체 후보 사전은 WordNet + 도메인 사전으로 구성하고, reality_stone의 geodesic score로 검증.

### 13.3 평가 지표

- 주제 유지율: SentenceTopicHead가 재생성된 문장에 대해 원문 주제를 재추정했을 때 일치율.
- BLEU/ROUGE vs 원문 (높을수록).
- 교체 비율: \( \frac{\text{변경된 토큰 수}}{\text{전체 토큰 수}} \le \rho_{\max} \).
- 현실 API 검증: Postman 시나리오 (정상 입력, 잘못된 metric key, 제한된 후보) 등을 자동화.

---

## 14. 구현 로드맵 (통합)

1. **문단 파이프라인**: Pre-Segmenter + 데이터 라벨링 스크립트 (`python/scripts/segment_topics.py`).
2. **SentenceTopicHead 모듈**: reality_stone layers로 구현, 단위테스트 추가.
3. **MetricContextRouter**: `metrikey` 기반 SPD 합성 클래스 구현 및 CLI 예제.
4. **RCE-LexicalDecoder 확장**: `gpt2_metric` 파생 클래스로 단어 교체 제약 적용, Rust 커널 연동.
5. **API 서버**: FastAPI/Flask 등에서 reality_stone 모델 래핑, Postman collection 제공.
6. **평가 스크립트**: 교체 비율, 주제 유지율 검증 스위트.

각 단계는 reality_stone 모듈 재사용을 최우선으로 하며, 새로운 파일/클래스는 사용자 승인 후 생성한다.


---

## 15. Sentence-Topic Guided LLM 구현 계획서

### 15.1 개요
- **목표**: 문단 입력을 문장·주제 단위로 해석하고, 동일 주제를 유지한 상태에서 단어 교체만 허용하는 Sentence-Topic Guided LLM을 reality_stone 스택으로 구현한다.
- **성공 기준**: `POST /sentence_topic_rewrite` API가 Postman에서 주제 score, metric key 로그, 교체 내역, 최종 문장을 정확히 반환.
- **사용 모듈**: `reality_stone.layers.metric_attention`, `reality_stone.metrikey`, `reality_stone.models.gpt2_metric`, `_rust.geodesic_topk_attention_*`, `layers.klein`, `layers.lowrank`.

### 15.2 요구사항 및 설계 원칙
- 문장 → 주제 판단 → 재배열 → 단어 교체 순서를 강제하며 토큰 수를 보존한다.
- 단어 교체는 사전 정의 후보 집합 내에서만 수행하고 삽입/삭제나 임의 신조어 생성을 금지한다.
- metric key가 geometry를 결정하며 잘못된 키 조합에서는 출력 품질을 의도적으로 저하해 보안성을 확보한다.
- DRY·KISS·SRP 준수, reality_stone API 재사용, 불필요한 파일/함수 생성 금지.

### 15.3 전체 아키텍처 개요
1. **Pre-Segmenter (L0, Euclidean)**: 문단→문장 분해, `topo_idx` 생성, `replacement_mask` 출력.
2. **SentenceTopicHead (L1, Poincaré)**: `MetricAttention(mode="geodesic")`로 주제 분포/우선순위/metric key seed 산출.
3. **Metric Context Router (L2, SPD)**: `metrikey.metric_from_keys`로 SPD 합성 후 Cholesky factor 제공.
4. **RCE-LexicalDecoder (L3, Lorentz/Klein)**: `gpt2_metric` 확장, geodesic Top‑k attention으로 후보 내 단어 교체.
5. **Post-Controller/API (L4, Euclidean)**: 비교/로그/응답 구성.

### 15.4 데이터 파이프라인
- 문장 토큰화: 위치·접속사 기반으로 `B×T×K`형 topology index 생성.
- 후보 사전: WordNet + 도메인 동의어 + cosine 근접 필터로 토큰별 후보 리스트를 캐싱.
- 주제 라벨링: 뉴스/의료 문단에 대한 semi-automatic topic annotation, metric key seed (`"topic:diagnosis|style:formal"`) 생성.

### 15.5 모듈별 상세 설계
#### 15.5.1 Pre-Segmenter
- 출력: 문장 리스트, token tensor, `replacement_mask`, `topo_idx`.
- 검증: 토큰 수 보존, 문장 경계 재현성, topology determinism 테스트.

#### 15.5.2 SentenceTopicHead
- 위치: `python/reality_stone/models/sentence_topic_head.py` (사용자 승인 후).
- 구성: `PoincareEmbedding` → geodesic Top‑k score → topic classifier → metric key seed 생성.
- 안정화: `layers.lowrank.SPDMetric` identity 초기화, eigenvalue 범위 모니터링.

#### 15.5.3 Metric Context Router
- 기능: `__call__(metric_keys, scores)`에서 `metrikey.metric_from_keys(...).cholesky()` 반환.
- 특징: 배치/캐시 지원, eigenvalue 클램프(0.1~5.0), 키 미존재 시 graceful fallback.

#### 15.5.4 RCE-LexicalDecoder
- 기반: `gpt2_metric.GPT2MetricModel` 파생, `forward(input_ids, metric_ctx, replacement_mask, topo_idx, candidates)`.
- Attention: `_rust.geodesic_topk_attention_lorentz` 우선, 미지원 시 Python fallback.
- Lexical guard: 후보 내 softmax, `replacement_mask=0` 위치는 원문 고정, Lorentz 결과를 Klein chart로 투영 후 FFN 처리.

#### 15.5.5 Post-Controller
- 기능: 원본 대비 교체 비율, 주제 재평가, metric key 로그 생성.
- 응답: 문장별 topic score, metric key, 교체 로그, 최종 문장 문자열.

### 15.6 API 설계
- **Endpoint**: `POST /sentence_topic_rewrite`
- **Request**:
  ```json
  {
    "paragraph": "string",
    "lexical_overrides": {"token": ["cand1", "cand2"]},
    "metric_hint": "optional"
  }
  ```
- **Response**: `sentences`, `topics`, `metric_keys`, `replacements`, `final_text`, `stats`.
- 오류 처리: 잘못된 metric key → 경고 및 안전한 출력, 후보 부재 → 원문 유지 로그.

### 15.7 학습 및 평가 전략
- Loss:  \( \mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda_{\text{topic}}\mathcal{L}_{\text{topic}} + \lambda_{\text{order}}\mathcal{L}_{\text{order}} + \lambda_{\text{lex}}\mathcal{L}_{\text{lex}} + \lambda_{\text{metric}}\mathcal{L}_{\text{metric}} + \lambda_{c}\mathcal{L}_{c} \)
- 데이터: 뉴스/의료 문단, topic label, lexical candidate pairs, metric key metadata.
- 지표: 주제 유지율, BLEU/ROUGE, 교체 비율 ≤ ρ_max, 잘못된 metric key 시 출력 붕괴 여부, API latency.

### 15.8 구현 로드맵
1. Pre-Segmenter & 라벨링 스크립트 작성, 단위 테스트.
2. SentenceTopicHead 모듈 구현, topic loss 학습 루프 구성.
3. Metric Context Router 통합, SPD 안정성·캐시 검증.
4. RCE-LexicalDecoder 확장, geodesic 커널 검증, lexical guard 확정.
5. FastAPI/Flask API 작성, reality_stone 모델 래핑, Postman collection 배포.
6. 학습 파이프라인/평가 스위트 실행, 결과 리포트화.

### 15.9 검증 및 운영 가이드
- 테스트: 모듈 단위, 통합 inference, API e2e, 잘못된 metric key 시나리오.
- 모니터링: stage별 latency, replacement ratio, topic mismatch율, 실패 케이스 분류.
- 배포 전 체크리스트: reality_stone API 버전, metric key 목록, Postman 시나리오 통과, eigenvalue 범위 로그.
