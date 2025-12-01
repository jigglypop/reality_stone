# 기존 LLM 위에 Reality Stone 이론 복사 가이드

> **Note**: 본 문서는 "Sidecar" 또는 "Plugin" 방식의 통합을 다룹니다. 모델을 완전히 리만 구조로 변환/병합하는 **"Native Transformation"** 및 **"Folding"** 기법은 **[Seed Pack/01_MATH_FOLDING.md](../00_seed_pack/01_MATH_FOLDING.md)** 및 **[Seed Pack/02_HETEROGENEOUS_MERGE.md](../00_seed_pack/02_HETEROGENEOUS_MERGE.md)**를 참조하세요.

## 1. 목표와 범위

- 이 문서는 **이미 학습된 LLM(phi, LLaMA, Mistral 등)을 그대로 두고**, Reality Stone 이론(리만 기하 + 메트릭 슬롯 + 계층 트리)을 **최소 침습적으로 복사/적용하는 방법**을 정리한다.
- 코드 레벨 기준점:
  - Reality Stone LLM 구현: `python/reality_stone/models/hierarchical_sentence_topic_llm.py`
  - 세그멘테이션/트리: `reality_stone.utils.pre_segmenter.PreSegmenter`
  - 메트릭/지오데식 어텐션: `reality_stone.layers.metric_attention.MetricAttention`, `RiemannianAggregation`
- 여기서 말하는 “복사”는 두 가지를 포함한다.
  1. **행태 복사**: 기존 LLM이 만드는 분포/스타일을 유지하면서, 리만 구조를 덧씌우는 것.
  2. **아키텍처 복사**: 기존 LLM의 self-attention/decoding을 Reality Stone식 지오데식 어텐션으로 대체하는 것.

아래 세 가지 패턴을 기본 전략으로 사용한다.

- 패턴 A: **사이드카 리라이팅** – 기존 LLM은 유지, Reality Stone은 “토큰 선택 필터/편집기”로 동작
- 패턴 B: **지식 증류(KD)** – 기존 LLM을 교사로 두고, Reality Stone LLM이 그 행태를 학습
- 패턴 C: **내부 어텐션 교체** – 기존 LLM의 어텐션을 지오데식 어텐션으로 치환

---

## 2. Reality Stone LLM 구조 요약 (복사 대상)

Reality Stone 쪽 “이론 구현체”는 대략 다음 네 레이어로 나뉜다.

1. **L0: 트리/세그멘테이션**
   - `PreSegmenter` 가 문단 → 문장 → 토큰 트리(`DocumentTree`)를 만든다.
   - `LevelInvariantTreeProcessor` 가 타입별 연산(`TreeNodeOperator`)으로 상향식/하향식 메시지 전달을 수행한다.
2. **L1: SentenceTopicHead**
   - 문장 임베딩을 푸앵카레 볼로 사상하고, `MetricAttention` 기반 geodesic attention 으로 **토픽 분포 + metric-key** 를 추출한다.
   - 결과: `P_topic`, `scores`, `metric_keys`.
3. **L2: MetricContextRouter + SPDMetricMixer**
   - `metric_keys` 와 점수로 **SPD 메트릭 슬롯**을 만들고(라우팅),
   - 문단/상하위 문장의 메트릭을 SPD barycenter(또는 가중합)로 섞어 **유효 메트릭 컨텍스트 `metric_ctx`** 를 만든다.
4. **L3: HierarchicalLMDecoder**
   - 토큰 시퀀스를 받아, Poincaré-Lorentz product manifold 거리로 계산된 **지오데식 어텐션**을 통해 logits 를 만든다.

기존 LLM 위에 “복사”한다는 것은, 이 네 계층의 역할을 유지한 채로 **입출력(히든/로짓)만 기존 LLM에 맞게 연결해 주는 것**이다.

---

## 3. 패턴 A: 사이드카 리라이팅 (기존 LLM 그대로 유지)

### 3.1 개념: 기존 LLM + 리만 필터

- 기존 LLM은 그대로 두고, Reality Stone을 **“후단 필터/편집기”** 로 붙인다.
- 흐름:
  1. 입력 텍스트 → Reality Stone `PreSegmenter` 로 문단/문장/토큰 트리 생성
  2. 같은 텍스트를 기존 LLM 토크나이저로 토큰화하고, 기존 LLM을 한 번 forward
  3. 기존 LLM의 토큰 히든을 Reality Stone의 L0~L2(트리/토픽/메트릭)로 올려 리만 구조를 계산
  4. 기존 LLM이 낸 logits 위에 **지오데식 거리 기반 재점수/편집** 을 적용해 최종 토큰을 선택

이 방식은 **가중치와 아키텍처를 전혀 건드리지 않고**도 Reality Stone 이론의 대부분(트리, metric-key, SPD 슬롯, 지오데식 softmax, 편집 정책)을 덧씌울 수 있다.

### 3.2 단계별 플로우

1. **세그멘테이션과 트리 생성**
   - Reality Stone 내부:
     - `PreSegmenter(text) -> {"tokens": [T, L], "topo_idx": [T, K], "tree": DocumentTree, ...}`
   - 기존 LLM:
     - `teacher_tokenizer(text, return_tensors="pt")` 로 토큰화.
   - 두 토크나이저의 vocab 은 다를 수 있으므로, **트리 구조(T, K)** 만 공유하고 토큰 ID 는 각자 독립적으로 관리한다.

2. **기존 LLM 히든을 Reality Stone 쪽으로 전달**
   - 기존 LLM에서 마지막 히든 `H_teacher` 를 얻는다.
   - Reality Stone 쪽에서는 `HierarchicalSentenceTopicLLM.encode_tokens_to_sentences` 가 자체 임베딩을 사용하도록 설계되어 있으므로,
     - 가장 간단한 복사 전략은:
       - Reality Stone 쪽에서는 평소처럼 자체 토큰 임베딩/세그멘테이션으로 문장 임베딩을 만든다.
       - 기존 LLM의 CLS/문장 임베딩을 **보조 특징**으로 사용해서, 손실이나 후단 정책에만 반영한다.
   - 더 강하게 복사하고 싶으면, 외부 프로젝트에서 별도 래퍼를 만들어
     - 토큰 단위가 아니라 **문장 단위 representation** 레벨에서 `H_teacher_sentence` 와 Reality Stone 문장 임베딩을 맞추는 식으로 쓴다.

3. **SentenceTopicHead 를 통한 metric-key 추출**
   - 문장 임베딩 `sentence_embeddings_raw` 를 `SentenceTopicHead` 에 넣어:
     - 토픽 분포 `P_topic`
     - 점수 `scores`
     - 문자열 metric-key 리스트 `metric_keys` 를 얻는다.
   - 이 key 는 예를 들어 `"topic:diagnosis|priority:high"` 와 같이, **도메인/역할/우선순위** 등을 부호화하는 역할을 한다.

4. **MetricContextRouter + SPDMetricMixer 로 메트릭 컨텍스트 계산**
   - `MetricContextRouter(metric_keys, scores) -> metric_ctx_sentence`:
     - 각 문장에 대해 SPD 메트릭의 Cholesky 분해 `L` 을 만든다.
   - `SPDMetricMixer.mix_hierarchy(...) -> metric_ctx`:
     - 문단 메트릭, 자기 메트릭, 이웃 문장 메트릭을 SPD barycenter (또는 가중합) 로 섞어,
     - 토큰/문장 레벨에서 사용할 `metric_ctx` 를 만든다.

5. **기존 LLM logits 재점수**
   - 기존 LLM의 logits 를 `logits_teacher` 라 두면,
   - Reality Stone의 manifold/메트릭을 활용해, 현재 히든 `h_t` 와 후보 토큰 임베딩 `e_i` 간 리만 거리 \(d_{\mathcal{M}}(h_t, e_i)^2\) 를 계산하고,
   - 최종 logits 를 다음과 같이 조정한다.
     - \[
       \tilde{\ell}_i
       =
       \ell^{\text{teacher}}_i
       -
       \lambda \, d_{\mathcal{M}}(h_t, e_i)^2
       \]
   - 여기서:
     - \( \lambda \) 는 메트릭 강도를 조정하는 하이퍼파라미터다.
     - \(d_{\mathcal{M}}\) 은 Reality Stone 가 이미 구현해 둔 Poincaré/Lorentz product manifold 거리 함수를 재사용한다.
   - 이 단계는 **외부 프로젝트 쪽에서 “디코더 래퍼”로 구현하는 것을 권장**한다.
     - Reality Stone 리포 안에서는 코어 커널/레이어만 유지하고,
     - “교사 LLM + Reality Stone 메트릭” 을 결합하는 로직은 별도 서비스 코드에서 작성하는 것이 안전하다.

6. **편집 모드 (선택)**
   - 토큰 생성이 아니라 “리라이팅/편집” 용도라면:
     - Reality Stone 쪽의 `EditOperationHead` 와 `infer_hierarchical_llm_on_text` 가 이미 **keep/replace/insert/delete/reorder** 정책을 구현해 두고 있다.
     - 이때도 “후보 토큰” 만 기존 LLM 에서 가져오고, **어떤 후보를 선택/삽입/삭제할지** 는 Reality Stone 메트릭과 편집 정책으로 결정하는 식으로 결합할 수 있다.

요약하면, 패턴 A 는 **기존 LLM을 “언어/지식 엔진”으로, Reality Stone을 “기하학적 트리/정책 엔진”으로 나누어** 파이프라인으로 붙이는 방식이다.

---

## 4. 패턴 B: 지식 증류로 Reality Stone LLM 학습

### 4.1 개념

- 여기서는 Reality Stone 내부 LLM(`HierarchicalSentenceTopicLLM`) 을 **교사 LLM과 최대한 비슷하게 만들기** 위해 학습한다.
- Reality Stone 리포에는 이미 이를 위한 훅이 있다.
  - 함수: `train_hierarchical_llm_from_text`
  - 인자: `teacher_model`, `teacher_tokenizer`, `kd_proj`, `kd_weight`
  - 데이터셋: `SentenceTopicDataset`, `teacher_ko_phi2.jsonl` 예제

### 4.2 데이터 포맷과 파이프라인

1. **JSONL 데이터 준비**
   - 각 라인은 “문단 단위 텍스트”를 가진 JSON 객체로 구성:
     - Reality Stone 리포의 `teacher_ko_phi2.jsonl` 을 참고해서 동일 포맷으로 확장한다.
   - 한 줄 = 1 개 문단 → `SentenceTopicDataset` 이 내부에서 `PreSegmenter` 로 문장/토큰을 나누고, `topo_idx`, `tree`, `tokens` 를 만든다.

2. **교사 LLM/토크나이저 로드**
   - 예시:
     - `teacher_model = AutoModel.from_pretrained("microsoft/phi-2")`
     - `teacher_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")`
   - `kd_proj` 는 교사 LLM 히든 차원 → Reality Stone 문단 임베딩 차원(`config.d_model`) 으로 사상하는 단순 선형층이면 충분하다.

3. **훈련 함수 호출**

```python
model, info = train_hierarchical_llm_from_text(
    data_path="teacher_ko_phi2.jsonl",
    config=HierarchicalLLMConfig(),
    epochs=E,
    batch_size=B,
    teacher_model=teacher_model,
    teacher_tokenizer=teacher_tokenizer,
    kd_proj=kd_proj,
    kd_weight=kd_lambda,
)
```

- 내부에서 수행되는 일:
  - LM loss: Reality Stone 자체 `HierarchicalLMDecoder` 에 대한 next-token cross entropy
  - 토픽/메트릭 loss: 문단 내 토픽 일관성, 배치 간 다양성, SPD 메트릭 정규화, 의미 보존 손실
  - KD loss(선택): `paragraph_embedding` 과 `kd_proj(teacher_hidden)` 간 MSE
    - 여기서 `teacher_hidden` 은 교사 LLM 의 CLS 또는 평균 풀링 히든.

### 4.3 결과 해석

- 학습이 끝난 Reality Stone LLM 은:
  - 토큰 분포/스타일은 **교사 LLM을 흉내 내면서**,
  - 토픽 분해, metric-key, SPD 메트릭 슬롯, 지오데식 어텐션은 **Reality Stone 구조를 그대로 사용**한다.
- 이 상태에서 패턴 A 의 “로짓 재점수” 를 끄고 Reality Stone LLM만 단독으로 사용해도,
  - 교사 LLM과 유사한 답변 품질을 기대할 수 있다.

---

## 5. 패턴 C: 기존 LLM 내부에 지오데식 어텐션 삽입

### 5.1 개념

- 여기서는 기존 LLM의 self-attention 을 Reality Stone 의 **지오데식 어텐션**으로 직접 교체한다.
- Reality Stone 리포에서 참고해야 할 구현:
  - `HierarchicalLMDecoder._DecoderBlock`
  - `MetricAttention` (dot / geodesic 모드)
  - Poincaré/Lorentz 거리 함수: `poincare_distance`, `lorentz_distance`, `from_poincare`

### 5.2 최소 교체 단위

1. **Q/K/V 프로젝션은 기존 LLM 그대로 사용**
   - MultiheadAttention 블록에서 `q_proj`, `k_proj`, `v_proj` 까지는 그대로 둔다.
2. **점수 계산 부분만 교체**
   - 기존:
     - \( \text{scores} = \frac{Q K^\top}{\sqrt{d}} \)
   - Reality Stone 식:
     - Poincaré/Lorentz product manifold 거리:
       - \( d_p = d_{\text{Poincaré}}(q, k) \)
       - \( d_l = d_{\text{Lorentz}}(q, k) \)
       - \( d_{\text{total}}^2 = \lambda_p d_p^2 + \lambda_l d_l^2 \)
     - 점수:
       - \( \text{scores} = - d_{\text{total}}^2 / \tau + \varepsilon \cdot s_{\text{lowrank}} \)
         - \(s_{\text{lowrank}}\) 는 SPDMetric 의 low-rank 항(gradient 경로 확보용) 참고.
3. **metric_ctx 주입**
   - `HierarchicalLMDecoder._DecoderBlock` 처럼,
     - `metric_ctx[b, s]` 가 SPD Cholesky `L` 이고,
     - `q, k` 를 `L` 로 스케일:
       - `q' = L q`, `k' = L k`
   - 기존 LLM 쪽에서는 `metric_ctx` 를 추가 인자로 받는 새 attention 클래스를 정의하고,
     - Reality Stone 트리/메트릭 모듈이 만들어 준 `metric_ctx` 를 연결해 준다.

### 5.3 적용 순서

1. 실험용으로 **1–2개 레이어에만** 지오데식 어텐션을 삽입해 본다.
2. teacher LLM 을 그대로 두고, 교체된 레이어의 출력이 원래 LLM 과 크게 다르지 않도록 **local KD** 로 미세 조정한다.
3. 안정성이 확보되면, 점차 더 많은 레이어를 Reality Stone 버전으로 교체해 간다.

이 패턴은 아키텍처를 거의 완전히 Reality Stone 식으로 바꾸는 작업이라, 패턴 A/B를 충분히 검증한 뒤 진행하는 것이 안전하다.

---

## 6. 권장 워크플로우 요약

1. **1단계 – 패턴 A (사이드카 리라이팅)**
   - 기존 LLM 위에 Reality Stone 트리/메트릭/편집기를 붙여서,
   - “리만 필터를 통과한 LLM” 을 먼저 확보한다.
2. **2단계 – 패턴 B (KD 기반 Reality Stone LLM 학습)**
   - Reality Stone LLM 을 교사 LLM에 맞춰 학습시켜,
   - 서비스 환경에서 교사 없이도 Reality Stone LLM 단독 운용이 가능하도록 만든다.
3. **3단계 – 패턴 C (내부 어텐션 교체, 선택 사항)**
   - 특정 LLM 한 종에 대해 아주 깊게 통합이 필요할 때만,
   - MultiheadAttention 을 Reality Stone 지오데식 어텐션으로 치환하고 짧은 파인튜닝으로 안정화한다.

이 문서는 설계/통합 관점의 “지도” 역할을 하며, 실제 구현 코드는 **가능한 한 Reality Stone 리포 바깥(서비스 레이어)** 에서 진행하고,
리포 내부에서는 커널/메트릭/트리/LLM 핵심 모듈만 유지하는 것을 기본 원칙으로 한다.



