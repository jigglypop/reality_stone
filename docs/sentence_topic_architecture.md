# Sentence-Topic Guided LLM 아키텍처 상세 설계서

## 1. 시스템 개요

### 1.1 목적 및 배경
- **핵심 아이디어**: 기존 LLM은 토큰 단위로 생성하지만, 본 시스템은 **문장 단위 주제 판정 → 주제 기반 재배열 → 단어 교체만 허용**하는 계층적 생성 방식을 채택한다.
- **차별점**: 
  - 문장 구조와 주제를 먼저 결정하므로 일관성 있는 문단 생성이 가능하다.
  - 단어 교체만 허용하여 원문의 의미와 길이를 보존한다.
  - reality_stone의 리만 기하 기반 attention으로 주제 공간을 명시적으로 모델링한다.

### 1.2 설계 제약
- **필수 사용 모듈**: `reality_stone.layers.metric_attention`, `reality_stone.metrikey`, `reality_stone.models.gpt2_metric`
- **금지 사항**: 
  - 토큰 삽입/삭제 금지 (교체만 허용)
  - 임의 신조어 생성 금지 (사전 정의 후보 집합 내에서만 선택)
  - 불필요한 파일/함수 생성 금지 (DRY, KISS, SRP 준수)
- **성공 기준**: Postman으로 `POST /sentence_topic_rewrite` 호출 시 주제·교체 로그·최종 문장 반환

---

## 2. 전체 아키텍처

### 2.1 5계층 구조

```
입력 문단
    ↓
[L0] Pre-Segmenter (Euclidean)
    ↓ 문장 리스트, token tensor, topo_idx, replacement_mask
[L1] SentenceTopicHead (Poincaré)
    ↓ topic 확률, 우선순위 score, metric_key seed
[L2] Metric Context Router (SPD manifold)
    ↓ SPD 메트릭 L_i (Cholesky factor)
[L3] RCE-LexicalDecoder (Lorentz/Klein)
    ↓ 교체된 토큰 시퀀스
[L4] Post-Controller (Euclidean)
    ↓
출력 JSON (주제, metric_keys, 교체 로그, 최종 문장)
```

### 2.2 계층별 책임

| 계층 | 명칭                | 입력                          | 출력                                | 사용 Manifold  | reality_stone 모듈                                      |
| ---- | ------------------- | ----------------------------- | ----------------------------------- | -------------- | ------------------------------------------------------- |
| L0   | Pre-Segmenter       | 문단 텍스트                   | 문장 리스트, tokens, topo_idx, mask | Euclidean      | torchtext + 사용자 파서                                 |
| L1   | SentenceTopicHead   | 문장 임베딩, topo_idx         | P_topic, score, metric_keys         | Poincaré (c<0) | `layers.poincare_embedding`, `layers.metric_attention`  |
| L2   | MetricContextRouter | metric_keys, scores           | SPD Cholesky L_i                    | SPD manifold   | `metrikey.metric_from_keys`, `layers.lowrank.SPDMetric` |
| L3   | RCE-LexicalDecoder  | tokens, L_i, mask, candidates | 교체 토큰 시퀀스                    | Lorentz/Klein  | `models.gpt2_metric`, `_rust.geodesic_topk_attention_*` |
| L4   | Post-Controller     | 원본/출력 비교                | API 응답 JSON                       | Euclidean      | Python glue                                             |

### 2.3 데이터 흐름 예시

```python
# 입력
paragraph = "환자는 고혈압 진단을 받았다. 약물 치료를 시작했다."

# L0: Pre-Segmenter
sentences = ["환자는 고혈압 진단을 받았다.", "약물 치료를 시작했다."]
tokens = [[101, 1234, ...], [101, 5678, ...]]  # [B, T, d]
topo_idx = [[[0,1], [0,1]], ...]  # [B, T, K] - 문장 순서 기반 이웃
replacement_mask = [[1,1,1,0,1], [1,1,0,1,1]]  # 1=교체 가능, 0=고정

# L1: SentenceTopicHead
P_topic = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]  # [B, T, C] - 주제 확률
scores = [0.9, 0.7]  # 문장 우선순위
metric_keys = ["topic:diagnosis|style:formal", "topic:treatment|style:formal"]

# L2: MetricContextRouter
L_i = metrikey.metric_from_keys(metric_keys, dim=64, ...).cholesky()
# L_i: [B, T, d_h, d_h] SPD Cholesky factor

# L3: RCE-LexicalDecoder
output_tokens = [[101, 1234', ...], [101, 5678', ...]]  # 일부 토큰 교체됨
# 예: "환자는" → "환자는" (고정), "고혈압" → "고혈압" (고정), "진단" → "판정" (교체)

# L4: Post-Controller
response = {
    "sentences": ["환자는 고혈압 판정을 받았다.", "약물 치료를 개시했다."],
    "topics": [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]],
    "metric_keys": ["topic:diagnosis|style:formal", "topic:treatment|style:formal"],
    "replacements": [{"pos": 3, "old": "진단", "new": "판정"}, ...],
    "final_text": "환자는 고혈압 판정을 받았다. 약물 치료를 개시했다."
}
```

---

## 3. L0: Pre-Segmenter 상세

### 3.1 기능
- 문단을 문장 단위로 분해하고 각 문장을 토큰화한다.
- 문장 간 위치 관계를 topology index로 표현한다.
- 교체 가능/불가능 토큰을 `replacement_mask`로 표시한다.

### 3.2 입출력 스펙
```python
def pre_segment(paragraph: str) -> Dict:
    """
    Args:
        paragraph: 입력 문단 텍스트
    Returns:
        {
            "sentences": List[str],           # 문장 리스트
            "tokens": torch.Tensor,           # [B, T, d] 토큰 임베딩
            "topo_idx": torch.Tensor,         # [B, T, K] 이웃 인덱스
            "replacement_mask": torch.Tensor, # [B, T] 0/1 마스크
            "metadata": Dict                  # 문장 길이, 위치 등
        }
    """
```

### 3.3 Topology Index 생성 규칙
- **시간 순서**: 각 문장은 이전/다음 문장을 이웃으로 갖는다.
- **접속사 기반**: "그러나", "따라서" 등의 접속사가 있으면 논리적 연결 추가.
- **주제 유사도**: 사전 학습된 임베딩으로 cosine 유사도 상위 K개 선택.

```python
# 예시: 3개 문장, K=2
# 문장 0: 이웃 [1, 2] (다음 문장 + 유사 문장)
# 문장 1: 이웃 [0, 2] (이전 문장 + 다음 문장)
# 문장 2: 이웃 [1, 0] (이전 문장 + 유사 문장)
topo_idx = torch.tensor([[[1, 2]], [[0, 2]], [[1, 0]]])  # [B=1, T=3, K=2]
```

### 3.4 Replacement Mask 생성 규칙
- **고정 토큰**: 고유명사, 숫자, 특수 기호, [CLS]/[SEP] 등
- **교체 가능**: 일반 명사, 동사, 형용사 등
- 사용자 지정 `lexical_overrides`가 있으면 우선 적용

```python
# 예시
sentence = "환자는 고혈압 진단을 받았다."
tokens = ["환자", "는", "고혈압", "진단", "을", "받았다", "."]
mask =    [1,      0,    1,       1,      0,    1,        0]
# "환자" 교체 가능, "는" 조사 고정, "고혈압" 교체 가능, ...
```

### 3.5 구현 포인트
- `nltk.sent_tokenize` 또는 `kss` (한국어) 사용
- 토큰화는 `transformers.AutoTokenizer` 활용
- topology 계산은 캐시하여 동일 문단 재처리 시 재사용

---

## 4. L1: SentenceTopicHead 상세

### 4.1 기능
- 문장 임베딩을 Poincaré 공간으로 매핑하여 주제 계층 구조를 표현한다.
- geodesic 거리 기반 attention으로 문장 간 관계를 파악한다.
- 주제 분류, 우선순위 점수, metric key seed를 출력한다.

### 4.2 수학적 정의

#### 4.2.1 Poincaré Embedding
문장 임베딩 \( x_t \in \mathbb{R}^d \) 를 Poincaré ball \( \mathbb{D}^{d_h} \) 로 매핑:
$$
z_t = \text{Exp}_0(x_t; c_p) = \tanh\left(\sqrt{|c_p|} \|x_t\| \right) \frac{x_t}{\|x_t\|}
$$
여기서 \( c_p < 0 \) 는 음의 곡률.

#### 4.2.2 Geodesic Distance
Poincaré ball 상의 두 점 \( z_i, z_j \) 사이의 거리:
$$
d_{\text{poincare}}(z_i, z_j; c_p) = \frac{2}{\sqrt{|c_p|}} \tanh^{-1}\left( \sqrt{|c_p|} \left\| \frac{z_i - z_j}{1 - \langle z_i, z_j \rangle} \right\| \right)
$$

#### 4.2.3 Attention Score
Top-K 이웃에 대한 geodesic 거리 기반 score:
$$
s_i = -\frac{1}{\tau_1} \sum_{j \in \text{Top-}K(i)} d_{\text{poincare}}(z_i, z_j; c_p)^2
$$

#### 4.2.4 Topic Classification
주제별 앵커 \( a_c \in \mathbb{D}^{d_h} \) 에 대한 log map 후 분류:
$$
P_{\text{topic}}(i, c) = \text{softmax}_c\left( w_c^\top \log_{a_c}(z_i) \right)
$$
여기서 \( \log_{a_c}(z_i) \) 는 \( a_c \) 에서 \( z_i \) 로의 tangent vector.

### 4.3 구현 스펙

```python
class SentenceTopicHead(nn.Module):
    def __init__(self, d_model, d_head, num_topics, num_heads, c_poincare=-1.0):
        super().__init__()
        self.poincare_embed = PoincareEmbedding(d_model, d_head, c=c_poincare)
        self.metric_attn = MetricAttention(
            d_head, num_heads, mode="geodesic", manifold="poincare",
            temperature=0.1, top_k=None  # topology로 제한
        )
        self.topic_anchors = nn.Parameter(torch.randn(num_topics, d_head) * 0.1)
        self.topic_classifier = nn.Linear(d_head, num_topics)
        
    def forward(self, x, topo_idx):
        """
        Args:
            x: [B, T, d_model] 문장 임베딩
            topo_idx: [B, T, K] topology index
        Returns:
            P_topic: [B, T, num_topics] 주제 확률
            scores: [B, T] 우선순위 점수
            metric_keys: List[str] 문장별 metric key seed
        """
        # Poincaré embedding
        z = self.poincare_embed(x)  # [B, T, d_head]
        
        # Geodesic attention
        attn_out, attn_weights = self.metric_attn(
            z, z, z, topo_idx=topo_idx
        )  # [B, T, d_head], [B, T, K]
        
        # Priority score (attention weight 합)
        scores = attn_weights.sum(dim=-1)  # [B, T]
        
        # Topic classification
        logits = self.topic_classifier(attn_out)  # [B, T, num_topics]
        P_topic = F.softmax(logits, dim=-1)
        
        # Metric key seed 생성
        metric_keys = self._generate_metric_keys(P_topic, scores)
        
        return P_topic, scores, metric_keys
    
    def _generate_metric_keys(self, P_topic, scores):
        """주제 확률과 score로부터 metric key 문자열 생성"""
        keys = []
        topic_names = ["diagnosis", "treatment", "prognosis", "general"]
        for b in range(P_topic.shape[0]):
            for t in range(P_topic.shape[1]):
                top_topic = P_topic[b, t].argmax().item()
                score_level = "high" if scores[b, t] > 0.7 else "low"
                key = f"topic:{topic_names[top_topic]}|priority:{score_level}"
                keys.append(key)
        return keys
```

### 4.4 학습 목표
- **주제 분류 손실**: 
  $$
  \mathcal{L}_{\text{topic}} = \text{CE}(P_{\text{topic}}, y_{\text{topic}})
  $$
- **우선순위 손실** (순서 보존):
  $$
  \mathcal{L}_{\text{order}} = \sum_{i<j} \max(0, s_j - s_i + \text{margin})
  $$
  여기서 \( i < j \) 는 원본 문장 순서.
- **메트릭 안정성**:
  $$
  \mathcal{L}_{\text{metric}} = \text{penalty}(\lambda_{\min}(G), \lambda_{\max}(G))
  $$

---

## 5. L2: Metric Context Router 상세

### 5.1 기능
- 문장별 metric key와 우선순위 score를 받아 SPD 메트릭을 합성한다.
- `reality_stone.metrikey.metric_from_keys`를 사용하여 키 기반 SPD 생성.
- Cholesky factorization으로 \( L_i \) 를 계산하여 attention에 주입한다.

### 5.2 SPD 합성 수식

#### 5.2.1 단일 키 메트릭
metric key \( k \) 에 대해 SPD 메트릭 \( G_k \) 생성:
$$
G_k = \text{MetriKey}(k, d_h, \lambda_{\min}, \lambda_{\max})
$$
내부적으로:
$$
G_k = \text{diag}(\text{softplus}(d_k)) + U_k U_k^\top
$$
여기서 \( d_k, U_k \) 는 키 \( k \) 에 대응하는 학습 가능 파라미터.

#### 5.2.2 다중 키 혼합
문장 \( i \) 가 여러 metric key \( \{k_{i1}, \ldots, k_{im}\} \) 를 가질 때:
$$
G_i = \sum_{j=1}^{m} \alpha_{ij} G_{k_{ij}}
$$
여기서 \( \alpha_{ij} = \text{softmax}(\beta \cdot s_{ij}) \), \( s_{ij} \) 는 우선순위 score.

#### 5.2.3 Eigenvalue 클램핑
SPD 조건 유지 및 수치 안정성을 위해:
$$
G'_i = Q \text{diag}(\text{clamp}(\lambda, \lambda_{\min}, \lambda_{\max})) Q^\top
$$
여기서 \( G_i = Q \Lambda Q^\top \) 는 eigenvalue decomposition.

#### 5.2.4 Cholesky Factorization
$$
G'_i = L_i L_i^\top
$$
\( L_i \) 는 하삼각 행렬로, attention에서 \( q' = L_i q, k' = L_i k \) 로 사용.

### 5.3 구현 스펙

```python
class MetricContextRouter:
    def __init__(self, d_head, lambda_min=0.1, lambda_max=5.0, cache_size=1000):
        self.d_head = d_head
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.cache = {}  # 키 조합 캐싱
        self.cache_size = cache_size
        
    def __call__(self, metric_keys: List[str], scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            metric_keys: [B*T] 문장별 metric key 리스트
            scores: [B, T] 우선순위 점수
        Returns:
            L: [B, T, d_head, d_head] Cholesky factor
        """
        B, T = scores.shape
        L_list = []
        
        for i, key in enumerate(metric_keys):
            # 캐시 확인
            cache_key = (key, scores.flatten()[i].item())
            if cache_key in self.cache:
                L_list.append(self.cache[cache_key])
                continue
            
            # SPD 메트릭 생성
            G = metrikey.metric_from_keys(
                [key], dim=self.d_head,
                min_lambda=self.lambda_min,
                max_lambda=self.lambda_max,
                masses=[scores.flatten()[i].item()]
            )
            
            # Eigenvalue 클램핑
            eigvals, eigvecs = torch.linalg.eigh(G)
            eigvals = torch.clamp(eigvals, self.lambda_min, self.lambda_max)
            G_clamped = eigvecs @ torch.diag(eigvals) @ eigvecs.T
            
            # Cholesky factorization
            L = torch.linalg.cholesky(G_clamped)
            
            # 캐시 저장
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = L
            
            L_list.append(L)
        
        return torch.stack(L_list).view(B, T, self.d_head, self.d_head)
```

### 5.4 보안 메커니즘
- **잘못된 키 조합**: 학습되지 않은 키 조합에서는 eigenvalue가 불안정하거나 identity에 가까워져 attention이 무의미해진다.
- **키 검증**: 허용된 키 목록과 비교하여 불일치 시 경고 로그 + fallback metric 사용.
- **Gradient masking**: 특정 키에 대해 gradient를 차단하여 역공학 방지.

---

## 6. L3: RCE-LexicalDecoder 상세

### 6.1 기능
- `gpt2_metric.GPT2MetricModel`을 확장하여 geodesic attention 기반 토큰 생성.
- 후보 집합 내에서만 토큰 선택 (lexical constraint).
- `replacement_mask`로 고정 토큰 보호.
- Lorentz manifold에서 attention 수행 후 Klein chart로 투영하여 FFN 처리.

### 6.2 Lorentz Manifold 기반 Attention

#### 6.2.1 Lorentz Distance
Lorentz model에서 두 점 \( x, y \in \mathbb{H}^{d+1} \) 사이의 거리:
$$
d_{\text{lorentz}}(x, y; c_l) = \frac{1}{\sqrt{|c_l|}} \cosh^{-1}\left( -\langle x, y \rangle_L \right)
$$
여기서 \( \langle x, y \rangle_L = -x_0 y_0 + \sum_{i=1}^{d} x_i y_i \) 는 Minkowski inner product.

#### 6.2.2 Geodesic Attention Score
$$
s_{ij} = -\frac{1}{\tau_2} d_{\text{lorentz}}(q'_i, k'_j; c_l)^2
$$
여기서 \( q'_i = L_i q_i, k'_j = L_j k_j \) (metric context 적용).

#### 6.2.3 Lexical Constraint Softmax
후보 집합 \( C_t \) 에 대해서만 확률 계산:
$$
p(y_t = c \mid x_{<t}) =
\begin{cases}
\frac{\exp(s_{tc})}{\sum_{c' \in C_t} \exp(s_{tc'})} & c \in C_t \\
0 & \text{otherwise}
\end{cases}
$$

### 6.3 구현 스펙

```python
class RCELexicalDecoder(GPT2MetricModel):
    def __init__(self, config, manifold="lorentz", c_lorentz=-1.0):
        super().__init__(config)
        self.manifold = manifold
        self.c_lorentz = c_lorentz
        
        # Attention 블록을 geodesic으로 교체
        for block in self.transformer.h:
            block.attn = MetricAttention(
                config.n_embd, config.n_head,
                mode="geodesic", manifold=self.manifold,
                temperature=0.1
            )
    
    def forward(
        self,
        input_ids,
        metric_ctx,           # [B, T, d_h, d_h] Cholesky L_i
        replacement_mask,     # [B, T] 0/1
        topo_idx,             # [B, T, K]
        candidates,           # Dict[int, List[int]] 토큰별 후보
        **kwargs
    ):
        """
        Args:
            input_ids: [B, T] 입력 토큰 ID
            metric_ctx: SPD Cholesky factor
            replacement_mask: 교체 가능 여부
            topo_idx: topology index
            candidates: {token_id: [cand1, cand2, ...]}
        Returns:
            output_ids: [B, T] 출력 토큰 ID
            logits: [B, T, V] 제약된 logits
        """
        B, T = input_ids.shape
        
        # Embedding
        hidden = self.transformer.wte(input_ids)  # [B, T, d_model]
        
        # Transformer blocks with metric context
        for i, block in enumerate(self.transformer.h):
            # Geodesic attention with metric
            hidden = block(
                hidden,
                metric_ctx=metric_ctx,
                topo_idx=topo_idx
            )
        
        # LM head
        logits = self.lm_head(hidden)  # [B, T, V]
        
        # Lexical constraint 적용
        constrained_logits = self._apply_lexical_constraint(
            logits, replacement_mask, candidates
        )
        
        # Sampling or argmax
        output_ids = torch.argmax(constrained_logits, dim=-1)
        
        # replacement_mask=0 위치는 원본 유지
        output_ids = torch.where(
            replacement_mask.bool(),
            output_ids,
            input_ids
        )
        
        return output_ids, constrained_logits
    
    def _apply_lexical_constraint(self, logits, mask, candidates):
        """후보 집합 외 토큰은 -inf로 마스킹"""
        B, T, V = logits.shape
        constrained = logits.clone()
        
        for b in range(B):
            for t in range(T):
                if mask[b, t] == 0:
                    continue  # 고정 토큰은 제약 없음
                
                token_id = input_ids[b, t].item()
                if token_id in candidates:
                    # 후보 외 토큰 마스킹
                    valid_ids = candidates[token_id]
                    mask_tensor = torch.ones(V, dtype=torch.bool)
                    mask_tensor[valid_ids] = False
                    constrained[b, t, mask_tensor] = float('-inf')
        
        return constrained
```

### 6.4 Klein Projection
Lorentz 출력을 Klein model로 투영하여 FFN에 공급:
$$
h^{(\text{klein})}_t = \frac{(h^{(\text{lorentz})}_t)_{1:d}}{(h^{(\text{lorentz})}_t)_0 + 1}
$$
이는 곡률 일관성을 유지하면서 Euclidean FFN과 호환된다.

---

## 7. L4: Post-Controller 및 API

### 7.1 Post-Controller 기능
- 원본 토큰과 출력 토큰 비교하여 교체 로그 생성.
- 주제 유지율 계산 (SentenceTopicHead로 재평가).
- metric key 로그, 통계 정보 구성.

### 7.2 API 설계

#### 7.2.1 Endpoint
```
POST /sentence_topic_rewrite
Content-Type: application/json
```

#### 7.2.2 Request Schema
```json
{
  "paragraph": "환자는 고혈압 진단을 받았다. 약물 치료를 시작했다.",
  "lexical_overrides": {
    "진단": ["판정", "확인"],
    "시작": ["개시", "착수"]
  },
  "metric_hint": "topic:diagnosis",
  "options": {
    "max_replacement_ratio": 0.3,
    "temperature": 0.1
  }
}
```

#### 7.2.3 Response Schema
```json
{
  "sentences": [
    "환자는 고혈압 판정을 받았다.",
    "약물 치료를 개시했다."
  ],
  "topics": [
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1]
  ],
  "metric_keys": [
    "topic:diagnosis|priority:high",
    "topic:treatment|priority:high"
  ],
  "replacements": [
    {"sentence": 0, "pos": 3, "old": "진단", "new": "판정"},
    {"sentence": 1, "pos": 2, "old": "시작", "new": "개시"}
  ],
  "final_text": "환자는 고혈압 판정을 받았다. 약물 치료를 개시했다.",
  "stats": {
    "total_tokens": 10,
    "replaced_tokens": 2,
    "replacement_ratio": 0.2,
    "topic_retention": 0.95
  }
}
```

### 7.3 구현 스펙 (FastAPI)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

app = FastAPI()

class RewriteRequest(BaseModel):
    paragraph: str
    lexical_overrides: Optional[Dict[str, List[str]]] = {}
    metric_hint: Optional[str] = None
    options: Optional[Dict] = {}

class RewriteResponse(BaseModel):
    sentences: List[str]
    topics: List[List[float]]
    metric_keys: List[str]
    replacements: List[Dict]
    final_text: str
    stats: Dict

@app.post("/sentence_topic_rewrite", response_model=RewriteResponse)
async def rewrite(request: RewriteRequest):
    try:
        # L0: Pre-Segmenter
        seg_output = pre_segmenter(request.paragraph)
        
        # L1: SentenceTopicHead
        P_topic, scores, metric_keys = sentence_topic_head(
            seg_output["tokens"], seg_output["topo_idx"]
        )
        
        # metric_hint 적용
        if request.metric_hint:
            metric_keys = [request.metric_hint + "|" + k for k in metric_keys]
        
        # L2: MetricContextRouter
        L_i = metric_router(metric_keys, scores)
        
        # L3: RCE-LexicalDecoder
        output_ids, _ = rce_decoder(
            seg_output["tokens"],
            L_i,
            seg_output["replacement_mask"],
            seg_output["topo_idx"],
            candidates=build_candidates(request.lexical_overrides)
        )
        
        # L4: Post-Controller
        response = post_controller(
            seg_output, output_ids, P_topic, metric_keys
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 8. 통합 및 검증

### 8.1 단위 테스트
- **L0**: 문장 분해 재현성, topology 일관성
- **L1**: 주제 분류 정확도, geodesic score 범위
- **L2**: SPD eigenvalue 범위, Cholesky 수치 안정성
- **L3**: lexical constraint 준수, replacement_mask 보존
- **L4**: API 응답 스키마, 오류 처리

### 8.2 통합 테스트
- **E2E 파이프라인**: 입력 문단 → API 응답까지 전체 흐름
- **Metric key 보안**: 잘못된 키 조합 시 출력 품질 저하 확인
- **교체 비율 제한**: `max_replacement_ratio` 준수 여부

### 8.3 Postman Collection
```json
{
  "info": { "name": "Sentence-Topic LLM API" },
  "item": [
    {
      "name": "정상 입력",
      "request": {
        "method": "POST",
        "url": "http://localhost:8000/sentence_topic_rewrite",
        "body": {
          "paragraph": "환자는 고혈압 진단을 받았다. 약물 치료를 시작했다."
        }
      }
    },
    {
      "name": "잘못된 metric key",
      "request": {
        "body": {
          "paragraph": "...",
          "metric_hint": "invalid_key"
        }
      }
    }
  ]
}
```

---

이 문서는 Sentence-Topic Guided LLM의 5계층 아키텍처를 수식·코드·예시를 통해 상세히 설명한다.  
각 계층의 입출력 스펙, reality_stone 모듈 연동 방법, API 설계를 포함하여 구현 단계에서 직접 참조 가능하다.

