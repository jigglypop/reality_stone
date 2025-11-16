## 계층적 Sentence-Topic LLM 설계 v2

### 1. 목표 및 배경

- **기존 한계**
  - 1세대 설계는 사실상 **문단 → 문장 → (문장 대표 토큰 1개 교체)** 에 가까운 3레벨 구조였다.
  - metric-key / SPD 메트릭도 **문장 레벨**에만 붙어 있고, 상·하위 레벨 간 기하학적 연결이 약했다.
  - “단어는 교체만”이라는 제약이 문서에 고정되어 있어, 삽입/삭제/재배열 같은 고급 편집 전략을 설계 차원에서 다루기 어려웠다.

- **v2 설계의 지향점**
  - **(1) 무한 확장 가능한 계층 구조**: 문단, 소문단, 문장, 구, 단어, 서브워드 … 를 모두 하나의 **문서 트리**로 표현.
  - **(2) 리만 메트릭 컨텍스트 스위칭**: 각 노드/레벨에 대해 다른 manifold/메트릭을 선택하고, metric-key 로 geometry 를 바꿀 수 있게 설계.
  - **(3) 최소한의 학습**: 거대한 가중치 대신 **SPD 메트릭(=리만 메모리 슬롯)** 위주로 학습해서 학습/추론 속도를 모두 확보.
  - **(4) 상향식 학습 / 하향식 추론**: 학습 때는 단어→문장→문단으로 올라가고, 추론 때는 문단→문장→단어 방향으로 내려가는 구조.

이 문서는 기존 `docs/llm.md`와 Sentence-Topic 문서들을 **“계층적 리만 LLM” 관점에서 재정리한 v2 설계 초안**이다.

---

### 2. 문서 표현: 트리 기반 계층 구조

#### 2.1 문서 트리 정의

- 문서를 유한 루트 트리 \(T = (V, E)\) 로 본다.
  - 각 노드 \(v \in V\) 는 텍스트 span 과 타입을 가진다.
    \[
    \mathrm{type}(v) \in \{\mathrm{document}, \mathrm{section}, \mathrm{subsection}, \mathrm{paragraph}, \mathrm{sentence}, \mathrm{phrase}, \mathrm{token}, \mathrm{subword}, \dots\}
    \]
  - 자식 집합:
    \[
    \mathrm{children}(v) = \{u \mid (v, u) \in E\}
    \]
- **핵심**: “문단/문장/단어 3단계”에 고정되지 않고, `type(v)` 만 정의되면 **레벨을 마음대로 추가/삭제**할 수 있다.

#### 2.2 계층적 세그멘테이션 함수

- 전체 세그멘테이션:
  \[
  \mathrm{Seg} : \text{Text} \rightarrow T
  \]
- 레벨별 플러그인 형태:
  - `Seg_level(document)`, `Seg_level(section)`, `Seg_level(sentence)` …
  - 현재 구현된 `PreSegmenter` 는 “문단 → 문장 → 토큰” 까지만 처리하지만,
  - 설계상은 동일 패턴으로 **소문단/구/단어/서브워드** 레벨을 재귀적으로 추가 가능하도록 구조화한다.

이 트리 표현 위에, 각 노드별로 manifold / metric / representation / 편집 연산을 올려서 LLM을 정의한다.

---

### 3. 레벨별 리만 다양체 및 메트릭

#### 3.1 레벨별 manifold 할당

- 각 노드 타입(레벨)에 대해 적합한 manifold 를 매핑한다.
  - **상위 구조 (document / section / paragraph)**  
    - 트리/위계 구조 → **Poincaré / Klein** 모델이 유리.
  - **시계열/동역학 (시간순 보고, 바이탈/EEG 등)**  
    - 시간 방향 + 하이퍼볼로이드 구조 → **Lorentz** 모델 사용.
  - **로컬 의미/특징 공간 (단어/문장 임베딩)**  
    - **SPD manifold / Euclidean**: SPD 메트릭이나 일반 LLM 임베딩 공간.

- 노드 \(v\)의 manifold:
  \[
  \mathcal{M}_v \in \{\mathbb{D}^{d}_{c_p},\ \mathbb{H}^{d+1}_{c_l},\ \mathrm{SPD}(d_h),\ \mathbb{R}^d, \dots\}
  \]

#### 3.2 노드 임베딩과 리만 메트릭

- 각 노드 \(v\)마다 표현 \(h_v \in \mathcal{M}_v\) 를 갖는다.
- 리만 메트릭 \(g_v\) 는 SPD 행렬 \(G_v \succ 0\) 로 표현한다.
  \[
  g_v(u, w) = u^\top G_v w,\quad G_v \in \mathrm{SPD}(d_v)
  \]
- `reality_stone` 에서 이미 사용 중인 SPD 파라미터화:
  \[
  G_v = \mathrm{diag}(\mathrm{softplus}(d_v)) + U_v U_v^\top
  \]
  를 **attention head 차원 \(d_h\)** 기준으로 재사용한다.

#### 3.3 Product manifold (다중 레벨 동시 사용)

- 여러 레벨의 표현을 동시에 고려할 때 **product manifold** 를 정의한다.
  \[
  \mathcal{M}_{\text{total}} = \prod_{\ell = 0}^{L} \mathcal{M}^{(\ell)}
  \]
- 예:
  - \(\mathcal{M}^{(0)}\): paragraph Poincaré
  - \(\mathcal{M}^{(1)}\): sentence Poincaré
  - \(\mathcal{M}^{(2)}\): token Euclidean
- product manifold 거리:
  \[
  d^2_{\text{total}}(x, y)
  = \sum_{\ell} \lambda_\ell\, d^2_{\mathcal{M}^{(\ell)}}\big(x^{(\ell)}, y^{(\ell)}\big)
  \]
  로 정의해, 상위 구조/하위 lexical 정보가 동시에 들어가도록 한다.

---

### 4. 리만 메트릭 컨텍스트 스위칭 (metric-key v2)

#### 4.1 노드별 metric-key 집합

- 1세대 설계에서는 문장별 키:
  \[
  \text{key}^{\text{sent}}_i = \text{"topic:diagnosis|priority:high"}
  \]
  을 사용했다.
- v2에서는 모든 노드에 대해 metric-key 집합을 정의한다.
  \[
  \mathcal{K}_v = \{k_{v,1}, \dots, k_{v,m}\}
  \]
  - 문단 노드 예: `"section:intro|domain:cardiology"`
  - 문장 노드 예: `"topic:treatment|priority:high"`
  - 토큰 노드 예: `"lexical:medical_term|sensitivity:high"`

#### 4.2 키 → SPD 메트릭 매핑

- 각 노드 \(v\)에서:
  \[
  G_v = \sum_{k \in \mathcal{K}_v} \alpha_{v,k}\, \mathrm{MetriKey}(k)
  \]
  - \(\mathrm{MetriKey}(k) \in \mathrm{SPD}(d_h)\) 는 `reality_stone.metrikey.metric_from_keys` 에 대응.
  - \(\alpha_{v,k} = \mathrm{softmax}(\beta s_{v,k})\) 는 SentenceTopicHead 류에서 나온 score/우선순위를 사용.

- **상·하위 레벨 메트릭 혼합**:
  \[
  G_v^{\mathrm{eff}}
  = \gamma_\uparrow G_{\mathrm{parent}(v)}
  + \gamma_0 G_v
  + \gamma_\downarrow \,\overline{G_{\mathrm{children}(v)}}
  \]
  - \(\overline{G_{\mathrm{children}(v)}}\): 자식 메트릭의 평균.
  - \(\gamma_\uparrow, \gamma_0, \gamma_\downarrow \in [0,1],\ \gamma_\uparrow + \gamma_0 + \gamma_\downarrow = 1\).
  - → 문단 geometry 와 문장/토큰 geometry 를 자연스럽게 섞는 **cross-level 메트릭**.

#### 4.3 컨텍스트 스위칭과 보안

- 컨텍스트 변경이란, 키 집합(혹은 그 조합 가중치)을 바꾸는 것:
  \[
  \mathcal{K}_v \mapsto \mathcal{K}_v'
  \quad\Rightarrow\quad
  G_v^{\mathrm{eff}} \mapsto (G_v^{\mathrm{eff}})'
  \]
- 같은 입력 트리 \(T\) 에 대해서도:
  - `"mode:clinical|topic:diagnosis"` vs `"mode:summary|topic:general"`  
    → 서로 다른 geometry → 다른 attention 패턴/편집 정책.

- **보안 관점**:
  - 허용된 키 조합에 대해서만 \(G_v^{\mathrm{eff}}\) 가 잘-conditioned SPD 가 되도록 학습.
  - 학습되지 않은 키 조합에서는 eigenvalue 가 극단적이거나 identity 에 가까워지도록 만들어,
    - geodesic attention 이 사실상 무의미해지고,
    - 출력 품질이 의도적으로 붕괴되게 하여 키 기반 보안성을 확보.

---

### 5. 계층적 지오데식 어텐션 (트리 구조용)

#### 5.1 동일 레벨에서의 geodesic score

- 같은 레벨 \(\ell\) 의 노드 \(u, v\) 에 대해:
  \[
  s^{(\ell)}_{uv}
  = -\frac{1}{\tau_\ell}\,
    d^2_{\mathcal{M}^{(\ell)}}\!\big(f_\ell(h_u), f_\ell(h_v)\big)
  \]
  - \(f_\ell\): 필요 시 manifold 간 좌표 변환 (예: Poincaré ↔ Lorentz ↔ Klein).
  - \(\tau_\ell\): 레벨별 temperature.

- **트리 기반 Top‑k 이웃 선택**
  - 후보 집합:
    - 동일 부모 아래의 형제 노드,
    - 부모/조상 노드,
    - 시간/순서 이웃.
  - 후보 중 geodesic 거리 상위 \(K\) 값만 사용:
    \[
    \alpha_{u,v} = 0 \quad \text{if } v \notin \mathrm{Top}\text{-}K(u)
    \]

#### 5.2 레벨 간 (up/down) 어텐션

- **자식 → 부모 업데이트**:
  \[
  h_{\mathrm{parent}}'
  = \mathrm{RiemannAgg}\Big(\{h_{\mathrm{parent}}\} \cup \{h_c : c \in \mathrm{children(parent)}\}\Big)
  \]
  - RiemannAgg 는 geodesic attention 또는 Riemannian 평균으로 구현.

- **부모 → 자식 컨텍스트 주입**:
  - 부모 표현 \(h_{\mathrm{parent}}\) 를 자식 manifold 로 투영한 뒤,
  - 자식 레벨 attention 의 query/key context 로 사용.

- **product manifold 기반 score**:
  \[
  s_{uv}
  = -\sum_{\ell}
    \frac{\lambda_\ell}{\tau_\ell}\,
    d^2_{\mathcal{M}^{(\ell)}}\big(h_u^{(\ell)}, h_v^{(\ell)}\big)
  \]
  - 상위 구조와 하위 lexical 정보가 동시에 점수에 반영된다.

---

### 6. 편집 연산 공간: 교체만이 아닌 삽입/삭제/재배열까지

#### 6.1 편집 연산 집합 \(\mathcal{E}_v\)

- 노드 \(v\) 에 대해 정의되는 편집 연산:
  \[
  \mathcal{E}_v \in
  \{\text{keep},\ \text{replace},\ \text{insert\_before},\ \text{insert\_after},\ \text{delete},\ \text{reorder\_siblings}\}
  \]
- LLM 은 각 노드에 대해:
  \[
  p_\theta(e \mid T, v, G^{\mathrm{eff}}, \text{context}),\quad e \in \mathcal{E}_v
  \]
  의 분포를 내놓는다.

- **기본 안전 모드**:
  - 허용 연산을 \(\{\text{keep}, \text{replace}\}\) 로 제한 → 현재 구현과 거의 동일.
- **확장 모드**:
  - 삽입/삭제/재배열까지 허용하되,
    - 레벨별 편집 budget \(\rho_\ell\) (예: 문장 레벨: reorder만, 토큰 레벨: insert ≤ 10%),
    - semantic consistency loss 로 의미 보존을 강하게 제약.

#### 6.2 lexical 공간에서의 연산

- 토큰 \(t\) 에 대한 후보 집합 \(C_t\) (동의어/도메인 사전/코사인 근접/lexical_overrides 등):
  - **replace**: \(t \rightarrow c,\ c \in C_t\)
  - **insert\_before/after**: 새 토큰 \(c \in C_t\) 를 주변에 삽입
  - **delete**: \(t\) 제거

- geodesic 기반 score:
  \[
  \tilde{s}_{t,c}
  = -\frac{1}{\tau_{\text{lex}}}\,
    d^2_{\mathcal{M}^{\text{token}}}\big(q'_t, k'_c\big)
  \]
  - 기존 RCE-LexicalDecoder 의 “토큰 후보 위 geodesic score + softmax” 설계를 그대로 확장.

#### 6.3 편집 제약 로스 (업그레이드)

- 편집 비용:
  \[
  \mathcal{L}_{\mathrm{edit}}
  =
    \lambda_{\mathrm{rep}}\cdot \#\mathrm{replace}
  + \lambda_{\mathrm{ins}}\cdot \#\mathrm{insert}
  + \lambda_{\mathrm{del}}\cdot \#\mathrm{delete}
  + \lambda_{\mathrm{ord}}\cdot \#\mathrm{reorder}
  \]

- 의미 보존:
  \[
  \mathcal{L}_{\mathrm{semantic}}
  =
  \sum_{v}
    d^2_{\mathcal{M}^{\text{sentence}}}\big(h_v^{\mathrm{orig}}, h_v^{\mathrm{edited}}\big)
  \]
  - 원문/편집 문장의 sentence-level representation 이 크게 변하지 않도록 제한.

- 전체 손실 (기존 항 + 편집 항):
  \[
  \mathcal{L}
  =
  \mathcal{L}_{\mathrm{LM}}
  + \lambda_{\mathrm{topic}}\mathcal{L}_{\mathrm{topic}}
  + \lambda_{\mathrm{metric}}\mathcal{L}_{\mathrm{metric}}
  + \lambda_{c}\mathcal{L}_{c}
  + \lambda_{\mathrm{edit}}\mathcal{L}_{\mathrm{edit}}
  + \lambda_{\mathrm{sem}}\mathcal{L}_{\mathrm{semantic}}
  \]

---

### 7. 상향식 학습 vs 하향식 추론, 그리고 “최소한의 학습”

#### 7.1 학습: 하위 → 상위 (bottom‑up encode)

- **방향**: \(L_{\text{token}} \rightarrow L_{\text{phrase}} \rightarrow L_{\text{sentence}} \rightarrow L_{\text{paragraph}} \rightarrow L_{\text{document}}\)
  - 토큰/구 레벨:
    - 일반 LLM / RCE 블록을 사용해 로컬 표현을 만들고, 이 부분은 **pretrain 후 거의 고정**.
  - 문장/소문단/문단 레벨:
    - 하위 레벨 표현들을 **Riemannian pooling(평균) / geodesic attention** 으로 올려보낸 뒤,
    - 그 위에 **주제/역할/구조/metric-key** 를 예측하는 얇은 헤드를 얹고,  
      **SPD 메트릭 \(G_v\)** 와 소수의 앵커/곡률만 집중적으로 학습.

- **학습 포인트**
  - 손실은 가능한 한 **상위 레벨 (문장/문단 주제, 구조, 메트릭 안정성)** 에 집중.
  - 백본 파라미터는 동결 또는 작은 LR,  
    메트릭/앵커/곡률 파라미터는 상대적으로 큰 LR → 학습 속도와 안정성 확보.

#### 7.2 추론: 상위 → 하위 (top‑down decode)

- **방향**: 문단/문서 조건 + metric-key → 문단/소문단 구조 → 문장 → 구/단어 편집
  1. 상위 레벨(document/paragraph)의 metric-key 로부터 \(G_v^{\mathrm{eff}}\) 를 결정.
  2. 해당 geometry 안에서 **문단/소문단/문장 단위의 구조(선택/재배열/생성)** 를 먼저 결정.
  3. 마지막에 토큰/서브워드 레벨에서 **lexical 연산(교체 + 선택적 삽입/삭제)** 을 수행.

- 이렇게 하면:
  - **구조(위계/순서)** 를 먼저 고정하고,
  - **lexical 디테일** 은 그 geometry 안에서만 탐색하게 되어,
  - 추론 시 탐색 공간이 줄어들고, 문단 차원의 일관성이 유지된다.

#### 7.3 “최소한의 학습”: 메트릭 슬롯 기반 메모리

- **가중치 대신 메트릭을 메모리 슬롯으로**
  - 거대한 W\_Q/W\_K/W\_V 를 계속 수정하는 대신,
  - 도메인/스타일/환자/보안 모드별로 **metric-key → SPD 메트릭 \(G_k\)** 와 몇 개의 앵커만 저장한다.

- 저장 구조:
  - `base_model.pt`:
    - 거의 고정된 RCE-Transformer / Sentence-TopicHead 백본.
  - `metric_slots.pt`:
    - `"topic:diagnosis|priority:high"` 등 key → SPD 파라미터/앵커/곡률 테이블.

- 효과:
  - **학습 속도**: 새 도메인은 새 metric 슬롯만 학습 → 파라미터 수와 업데이트 범위가 작다.
  - **추론 속도**: 키 → \(G \to L\)(Cholesky) 는 키당 한 번만 계산하고 캐시.
  - **운영**: 도메인/고객/모드 추가/삭제가 “metric 슬롯 추가/제거” 수준으로 단순해진다.

---

### 8. 현실 코드베이스와의 매핑

- **Pre-Segmenter / TreeBuilder**
  - 현재: `python/reality_stone/utils/pre_segmenter.py` 가 “문단 → 문장 → 토큰” 까지 담당.
  - v2: 동일 패턴으로 소문단/구/단어 레벨을 확장할 수 있도록,  
    내부 표현을 “문서 트리 \(T\)” 형태로 일반화.

- **SentenceTopicHead (L1, Poincaré 기반 헤드)**
  - 현재: `python/reality_stone/models/hierarchical_sentence_topic_llm.py` 의 `SentenceTopicHead` 가 문장 레벨에서 주제/metric-key seed 생성.
  - v2: 문장 뿐 아니라 section/paragraph 레벨에도 유사한 헤드를 확장하거나 공유할 수 있도록 설계.

- **MetricContextRouter (L2, SPD 합성)**
  - 현재: `python/reality_stone/models/hierarchical_sentence_topic_llm.py` 의 `MetricContextRouter` 가 문장별 key → SPD \(G\) → Cholesky \(L\) 을 계산.
  - v2: 노드 타입/레벨 정보를 입력으로 받아, 상·하위 메트릭 혼합 및 product manifold 설정을 지원.

- **Lexical / LM Decoder (L3)**
  - 현재: `python/reality_stone/models/hierarchical_sentence_topic_llm.py` 안에 `RCELexicalDecoder` (lexical constraint) 와 `HierarchicalLMDecoder` (순수 LM) 가 함께 정의되어 있다.
  - v2: 편집 연산 집합 \(\mathcal{E}_v\) 를 도입해,
    - 기본 모드에서는 replace-only,
    - 확장 모드에서는 insert/delete/reorder 를 포함하는 구조로 확장.

- **통합 계층 LLM**
  - 현재: 같은 파일의 `HierarchicalSentenceTopicLLM` 이 L0(Pre-Segmenter 출력) → L1(SentenceTopicHead) → L2(MetricContextRouter/SPDMetricMixer) → L3(HierarchicalLMDecoder)를 하나의 모델로 묶고,
    - `train_hierarchical_llm_from_text` 로 텍스트 파일 기반 joint 학습,
    - `infer_hierarchical_llm_on_text`, `answer_question_from_corpus` 로 추론/QA 유틸을 제공한다.

- **Rust/CUDA 커널 (core)**
  - geodesic distance, Möbius 연산, SPD 메트릭 합성, MetriKey 등은 모두 `docs/core/README.md` 에서 링크하는 커널 문서와 대응.
  - v2 설계에서도 이 커널들을 그대로 사용하고, 필요 시 `geodesic_attention` 바인딩을 추가해 LLM 규모에 맞는 성능을 확보한다.

이 문서는 기존 Sentence-Topic 설계를  
**트리 기반 계층 구조 + 리만 메트릭 컨텍스트 스위칭 + 메트릭 슬롯 메모리** 관점에서 한 단계 업그레이드한 설계 기준으로 삼는다.

---

### 9. 수학적 정식화: 확률 모형과 정보 기하 관점

#### 9.1 문서 트리 상의 확률 모형 스케치

- 입력 문서 \(x\) 에 대해, 세그멘테이션과 트리 구성까지 포함한 **관측 변수**를
  \[
  T = (V, E),\quad \{x_v\}_{v\in V}
  \]
  로 둔다. (각 노드 \(v\) 의 텍스트 span/메타데이터 포함)
- LLM 이 학습/추론하는 **잠재 변수**들은 대략 다음과 같이 볼 수 있다.
  \[
  \{h_v\}_{v\in V},\ \{G_v\}_{v\in V},\ \{\mathcal{E}_v\}_{v\in V},\ \{\mathcal{K}_v\}_{v\in V}
  \]
  - \(h_v\): manifold \(\mathcal{M}_v\) 상의 표현.
  - \(G_v\): SPD 메트릭 (또는 \(G_v^{\mathrm{eff}}\)).
  - \(\mathcal{E}_v\): 편집 연산 \(e_v \in \mathcal{E}_v\) (keep/replace/insert/delete/reorder).
  - \(\mathcal{K}_v\): metric-key 집합 및 mixing 계수.
- 단순화된 조건부 생성 모형(“원문 \(x\) 를 조건으로 편집된 문서 \(y\) 생성”):
  \[
  p_\theta(y \mid x)
  =
  \sum_{T, h, G, \mathcal{E}, \mathcal{K}}
  p_\theta(T, h, G, \mathcal{E}, \mathcal{K} \mid x)\,
  \delta\big(y = \mathrm{ApplyEdits}(x, T, \mathcal{E})\big).
  \]
  - 실제 구현에서는 \(T\) 는 deterministic 세그멘테이션이고,
  - \(h, G, \mathcal{E}, \mathcal{K}\) 에 대해 **최대우도 혹은 1-step MAP 근사**를 사용:
    \[
    \hat{y} \approx \mathrm{ApplyEdits}\Big(x, T, \arg\max_{\mathcal{E}} p_\theta(\mathcal{E} \mid x, T)\Big).
    \]

#### 9.2 상향식 인코딩 = 트리 메시지 전달 (Riemannian message passing)

- 트리 방향성을 명시하면, 하향식/상향식 연산을 **message passing** 으로 해석할 수 있다.
- **상향식 인코딩**(학습 시):
  - 자식 → 부모로 올라가는 리만 평균 / geodesic attention:
    \[
    h_v
    =
    \mathrm{RiemannAgg}\Big(\{h_c : c\in\mathrm{children}(v)\};\ \mathcal{M}_v, G_v\Big)
    \]
    \[
    \approx
    \mathrm{Exp}_{\mu}\Big(
      \sum_c \alpha_{v,c}\, \log_{\mu}(h_c)
    \Big),
    \]
    - \(\mu\): 초기 기준점(예: 부모의 이전 표현).
    - \(\log_\mu, \mathrm{Exp}_\mu\): 해당 manifold 의 로그/지수 맵.
    - \(\alpha_{v,c}\): geodesic score 기반 attention 가중치.
- 이렇게 보면, 각 레벨의 업데이트는 **리만 다양체 상의 Graph Neural Network (GNN)** 와 유사하며,  
  학습은 전체 파라미터 \(\theta\) (백본 + 메트릭 슬롯)에 대한 Riemannian SGD 로 볼 수 있다.

#### 9.3 하향식 디코딩 = 조건부 편집 정책

- **하향식 추론**은 “부모 상태 + 메트릭 + 로컬 컨텍스트”를 조건으로 한 **편집 정책**:
  \[
  \pi_\theta(e_v \mid h_{\mathrm{parent}(v)}, h_v, G_v^{\mathrm{eff}}, \text{local context}).
  \]
  - 여기서 \(\pi_\theta\) 는 discrete policy (edit 연산 선택) + continuous policy (lexical 후보 선택)로 구성.
- 토큰/서브워드 레벨에서,
  \[
  p_\theta(c \mid t, h_v, G_v^{\mathrm{eff}})
  \propto
  \exp\Big(
    -\tfrac{1}{\tau_{\text{lex}}}\, d^2_{\mathcal{M}^{\text{token}}}(q_t', k_c')
  \Big)
  \]
  와 같이 **geodesic softmax 정책**으로 볼 수 있으며,
  - 이는 정보 기하 관점에서 “메트릭에 의해 정의된 자연 거리”를 활용한 Gibbs 분포에 해당.

#### 9.4 SPD 메트릭 슬롯 = 정보 기하 메모리

- MetriKey 로부터 생성되는 SPD 메트릭 \(G_k\) 들은, SPD 다양체 \(\mathrm{SPD}(d_h)\) 상의 점:
  \[
  G_k \in \mathrm{SPD}(d_h),\quad
  d_{\mathrm{SPD}}(G_1, G_2)
  =
  \big\|\log(G_1^{-1/2} G_2 G_1^{-1/2})\big\|_F
  \]
  (Affine-invariant Riemannian metric 예시)

- **키 혼합을 SPD 바리센터로 해석**  
  - 노드 \(v\)에 대해 metric-key 슬롯 \(G_k\) 와 가중치 \(\alpha_{v,k}\) 가 있을 때,
    단순 선형 결합 대신 **SPD barycenter** 를 표준으로 삼는다:
    \[
    G_v
    =
    \mathop{\arg\min}_{G\succ 0}
      \sum_{k\in\mathcal{K}_v} \alpha_{v,k}\, d_{\mathrm{SPD}}(G, G_k)^2.
    \]
  - 상·하위 메트릭 혼합도 같은 방식으로,
    \[
    G_v^{\mathrm{eff}}
    =
    \mathop{\arg\min}_{G\succ 0}
      \Big(
        \gamma_\uparrow d_{\mathrm{SPD}}(G, G_{\mathrm{parent}(v)})^2
        + \gamma_0 d_{\mathrm{SPD}}(G, G_v)^2
        + \gamma_\downarrow d_{\mathrm{SPD}}(G, \overline{G_{\mathrm{children}(v)}})^2
      \Big),
    \]
    로 해석할 수 있다.
  - 실제 구현에서는 log-Euclidean 근사(행렬 로그/지수 후 유클리드 평균)를 사용해 효율적으로 계산할 수 있다.

- “최소한의 학습”을 정보 기하 관점에서 보면,
  - 백본 파라미터는 거의 고정하고,
  - 각 도메인/모드/환자 키에 대해 **slot 메트릭 \(\{G_k\}\)** 만 업데이트:
    \[
    G_k^{(t+1)}
    =
    \mathrm{Exp}_{G_k^{(t)}}\big(
      -\eta \, \mathrm{grad}_{G_k}\, \mathcal{L}
    \big),
    \]
  - 여기서 \(\mathrm{grad}_{G_k}\) 는 SPD 다양체 상의 Riemannian gradient,  
    \(\mathrm{Exp}_{G_k}\) 는 SPD manifold 상의 지수 맵.
- 실제 구현에서는 `diag(softplus(d)) + UU^\top` 로 파라미터화된 **유클리드 좌표**에서 최적화를 하지만,  
  설계 관점에서는 위와 같은 **Riemannian SGD / 자연 경사도(natural gradient) 근사**를 목표로 한다고 보는 것이 자연스럽다.

#### 9.5 Product manifold 로서의 multi-level geometry

- 여러 레벨 \(\ell\) 의 manifold \((\mathcal{M}^{(\ell)}, g^{(\ell)})\) 를 동시에 사용할 때,
  product manifold \(\mathcal{M}_{\text{total}} = \prod_{\ell=0}^{L} \mathcal{M}^{(\ell)}\) 를
  block-diagonal 메트릭으로 정의할 수 있다:
  \[
  g_{(x^{(0)},\dots,x^{(L)})}
  =
  \bigoplus_{\ell=0}^{L} \lambda_\ell\, g^{(\ell)}_{x^{(\ell)}}.
  \]
- 이때 geodesic 길이를 제곱 거리의 가중합으로 근사하면,
  \[
  d^2_{\text{total}}(x, y)
  \approx
  \sum_{\ell} \lambda_\ell\, d^2_{\mathcal{M}^{(\ell)}}\big(x^{(\ell)}, y^{(\ell)}\big),
  \]
  가 되고, 앞서 정의한 multi-level score
  \[
  s_{uv}
  = -\sum_{\ell}
    \frac{\lambda_\ell}{\tau_\ell}\,
    d^2_{\mathcal{M}^{(\ell)}}\big(h_u^{(\ell)}, h_v^{(\ell)}\big)
  \]
  은 \(\mathcal{M}_{\text{total}}\) 상 하나의 geodesic 기반 Gibbs score 로 해석된다.

#### 9.6 조건부 구조의 factorization 스케치

- 트리 \(T=(V,E)\) 가 주어졌다고 할 때, 잠재 변수들의 조건부 구조를 한 번에 정리하면:
  \[
  p_\theta(h, G, \mathcal{E}, \mathcal{K} \mid x, T)
  =
  \prod_{v\in V}
  p_\theta\big(h_v \mid \mathrm{children}(v), G_v\big)\,
  p_\theta\big(G_v \mid \mathcal{K}_v\big)\,
  p_\theta\big(\mathcal{K}_v \mid x_v\big)\,
  p_\theta\big(e_v \mid \mathrm{pa}(v), h, G\big).
  \]
  - \(p(h_v\mid\mathrm{children}, G_v)\): 9.2의 RiemannAgg 업데이트.
  - \(p(G_v\mid\mathcal{K}_v)\): 9.4의 SPD 바리센터.
  - \(p(\mathcal{K}_v\mid x_v)\): SentenceTopicHead / MetricRouter 가 담당.
  - \(p(e_v\mid\mathrm{pa}(v),h,G)\): 9.3의 편집 정책 \(\pi_\theta\).
- 이 factorization 을 기준으로 보면,
  - 상향식은 \(p(h_v\mid\mathrm{children},G_v)\),
  - 하향식은 \(p(e_v\mid\mathrm{pa},h,G)\),
  - 메트릭/키는 \(p(G_v\mid\mathcal{K}_v), p(\mathcal{K}_v\mid x_v)\)
  로 역할이 분리되어, 모듈별 책임을 수학적으로 명확히 설명할 수 있다.

#### 9.7 Lipschitz / 안정성 관점의 정리 후보

- RCE-Transformer 블록 하나를,
  \[
  h^{\ell+1} = \mathcal{F}_\ell(h^\ell; G_\ell, c_\ell)
  \]
  로 보면,
  - 지오데식 attention + SPD scaling 이 만드는 Lipschitz 상한을
    \(\mathrm{Lip}(\mathcal{F}_\ell)\) 로 근사/추정할 수 있다.
- 이상적인 목표:
  - **곡률 \(c_\ell\)**, SPD eigenvalue 범위 \([\lambda_{\min}, \lambda_{\max}]\),  
    geodesic dropout/top‑k sparsity 정도를 바탕으로,
  - 각 블록에 대해
    \[
    \mathrm{Lip}(\mathcal{F}_\ell)
    \leq L_\ell(c_\ell, \lambda_{\max}, K, \tau_\ell)
    \]
    형태의 상계를 주고, 전체 네트워크의 안정성(gradient 폭발/소실, 제어가능성)을 분석하는 것이다.
- 현재 리포지토리에서는 안정성 상계를 **실험적·휴리스틱 제약(regularizer)** 로 대체하고 있으나,  
  위와 같은 수학적 정리를 향해 설계를 정렬해 두면 이후 이론/논문 단계로 확장하기 용이하다.

---

### 10. 이론상 무한 트리 레벨을 위한 추상화

위 설계는 문단/문장/토큰 같은 유한한 레벨에 맞춰 예시를 들었지만,  
실제 목표는 **“레벨 개수를 사전에 정하지 않아도 되는 일반 트리 연산자”** 이다.  
이를 위해 타입/연산을 레벨 번호가 아니라 **지역 규칙(local rule)** 로 정의한다.

#### 10.1 타입 체계와 레벨-불변 트리 정의

- **노드 타입 집합** \(\mathcal{T}\) 를 도입한다.
  - 예: \(\mathcal{T} = \{\text{document}, \text{section}, \dots, \text{token}, \text{subword}, \dots\}\).
  - 새 타입을 추가하는 것은 단지 \(\mathcal{T}\) 에 원소를 추가하고, 아래 정의되는 로컬 규칙 몇 개를 더하는 것에 해당한다.
- 문서/상태는 여전히 유한 트리 \(T=(V,E)\) 로 표현하되, 각 노드에 타입이 붙는다:
  \[
  \mathrm{type}: V \to \mathcal{T}.
  \]
- **깊이(depth)** 는 더 이상 고정 레벨 인덱스가 아니고, 단순히
  \[
  \mathrm{depth}(v) = 
  \begin{cases}
    0 & v \text{ is root} \\
    \mathrm{depth}(\mathrm{parent}(v)) + 1 & \text{otherwise}
  \end{cases}
  \]
  로 정의된다.  
  - 트리가 유한인 한, \(\max_{v\in V} \mathrm{depth}(v)\) 는 있지만, 설계상 어떤 상한을 두지 않는다.

#### 10.2 타입별 manifold/메트릭/연산자 사전

각 타입 \(\tau \in \mathcal{T}\) 에 대해 다음을 정의한다.

- **Manifold 사상**:
  \[
  \mathcal{M}_\tau: \tau \mapsto (\mathcal{M}_\tau, g_\tau, \Theta_\tau),
  \]
  - \(\mathcal{M}_\tau\): Poincaré/Lorentz/Klein/SPD/Euclidean 등 중 하나 (또는 product).
  - \(g_\tau\): 해당 manifold 의 Riemannian metric.
  - \(\Theta_\tau\): 이 manifold 위에서 사용할 연산/파라미터 집합 (예: 곡률 범위, 슬롯 인덱스 등).
- **메트릭 슬롯 규칙**:
  - 타입별로 slot 키 공간 \(\mathcal{K}_\tau\) 와 MetriKey 매핑을 둔다:
    \[
    \mathrm{MetriKey}_\tau: \mathcal{K}_\tau \to \mathrm{SPD}(d_\tau).
    \]
  - 실제 노드 \(v\) 의 메트릭은 이전과 같이 SPD barycenter 로 구성:
    \[
    G_v = \operatorname*{argmin}_{G\succ 0} \sum_{k\in\mathcal{K}_v} \alpha_{v,k}\, d_{\mathrm{SPD}}(G, G_{k})^2,
    \]
    단 \(\mathcal{K}_v \subseteq \mathcal{K}_{\mathrm{type}(v)}\).
- **로컬 업데이트 연산자**:
  - 타입별 상향식 연산자:
    \[
    \mathrm{UP}_\tau: 
      \big(\{h_c\}_{c\in\mathrm{children}(v)}, \{G_c\}, G_v, \text{topology/local features}\big)
      \mapsto h_v^{\uparrow}.
    \]
  - 타입별 하향식 연산자:
    \[
    \mathrm{DOWN}_\tau:
      \big(h_v^{\uparrow}, \{h_p^{\uparrow}\}_{p\in\mathrm{anc}(v)}, G_v, \text{topology/local features}\big)
      \mapsto h_v^{\downarrow}.
    \]
- 이 사전만 정의되면, **트리 깊이와 상관없이** 노드 타입에 따라 지역 규칙이 적용된다.

#### 10.3 레벨-불변 Riemannian message passing

전체 encode/decode 연산자는 이제 “레벨 수를 모르는” 형태로 재정의할 수 있다.

- **상향식 패스**:
  1. 리프 노드 집합 \(L = \{v \mid \mathrm{children}(v) = \emptyset\}\) 부터 시작.
  2. 위로 올라가며, 각 노드 \(v\) 에 대해:
     \[
     h_v^{\uparrow} = \mathrm{UP}_{\mathrm{type}(v)}(\cdot),
     \]
     이때 자식 상태들이 모두 업데이트 되었을 때만 \(\mathrm{UP}\) 적용.
  3. 이는 단순히 **트리의 topological order** 를 따라가며 타입별 연산자를 적용하는 것이고, 깊이가 얼마든 상관 없다.

- **하향식 패스**:
  1. 루트에서 시작하여, 각 노드 \(v\) 에 대해:
     \[
     h_v^{\downarrow} = \mathrm{DOWN}_{\mathrm{type}(v)}(\cdot).
     \]
  2. 부모/조상들의 \(h^{\uparrow}\) (또는 이전 iteration 의 \(h^{\downarrow}\)) 을 입력으로 사용.

- 전체 코어 연산자는
  \[
  \Phi_\theta(T, h, G) = (T, h', G'),
  \]
  로 정의되며,  
  여기서 \(\theta\) 는 모든 타입별 연산자/슬롯 파라미터 \(\{\Theta_\tau\}_{\tau\in\mathcal{T}}\) 를 포함한다.

이 정의는 **트리의 최대 깊이에 완전히 무관**하며,  
새 타입을 추가해도 \(\mathcal{T}\) 와 \(\mathrm{UP}_\tau, \mathrm{DOWN}_\tau, \mathrm{MetriKey}_\tau\) 만 덧붙이면 되므로  
이론상 임의 레벨/임의 타입의 확장(예: 더 깊은 의미 단위, 프로그램 AST, 멀티모달 노드)까지 자연스럽게 수용한다.

