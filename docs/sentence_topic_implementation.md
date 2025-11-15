# Sentence-Topic LLM 구현 가이드

## 1. 구현 로드맵

### 1.1 전체 일정

| 단계 | 작업 | 예상 기간 | 의존성 | 산출물 |
|------|------|-----------|--------|--------|
| Phase 0 | 환경 설정 및 reality_stone API 검증 | 1일 | - | 테스트 스크립트 |
| Phase 1 | Pre-Segmenter 구현 | 2일 | Phase 0 | `pre_segmenter.py` |
| Phase 2 | SentenceTopicHead 모듈 | 3일 | Phase 1 | `sentence_topic_head.py` |
| Phase 3 | MetricContextRouter | 2일 | Phase 2 | `metric_router.py` |
| Phase 4 | RCE-LexicalDecoder | 4일 | Phase 3 | `rce_lexical_decoder.py` |
| Phase 5 | API 서버 | 2일 | Phase 4 | `api_server.py`, Postman collection |
| Phase 6 | 학습 파이프라인 | 3일 | Phase 2-4 | `train.py`, `evaluate.py` |
| Phase 7 | 평가 및 튜닝 | 5일 | Phase 6 | 평가 리포트 |

**총 예상 기간**: 22일 (약 4주)

### 1.2 마일스톤

- **M1 (1주차)**: Pre-Segmenter + SentenceTopicHead 완료, 단위 테스트 통과
- **M2 (2주차)**: MetricContextRouter + RCE-LexicalDecoder 완료, 통합 테스트 통과
- **M3 (3주차)**: API 서버 + 학습 파이프라인 완료, Postman 검증 통과
- **M4 (4주차)**: 평가 완료, 문서화, 배포 준비

---

## 2. Phase 0: 환경 설정

### 2.1 의존성 설치

```bash
# reality_stone 설치 (이미 설치되어 있음)
cd /e/reality_stone
pip install -e .

# 추가 의존성
pip install fastapi uvicorn pydantic
pip install kss nltk konlpy gensim
pip install transformers torch
pip install h5py pandas pyarrow
pip install pytest pytest-cov

# NLTK 데이터
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

### 2.2 reality_stone API 검증

```python
# tests/test_reality_stone_api.py
import torch
from reality_stone.layers.metric_attention import MetricAttention
from reality_stone.layers.poincare_embedding import PoincareEmbedding
from reality_stone import metrikey

def test_metric_attention():
    """MetricAttention 기본 동작 검증"""
    B, T, d_model, num_heads = 2, 10, 64, 4
    
    attn = MetricAttention(
        d_model, num_heads,
        mode="geodesic",
        manifold="poincare",
        temperature=0.1
    )
    
    x = torch.randn(B, T, d_model)
    topo_idx = torch.randint(0, T, (B, T, 5))
    
    out, weights = attn(x, x, x, topo_idx=topo_idx)
    
    assert out.shape == (B, T, d_model)
    assert weights.shape == (B, T, 5)
    print("✓ MetricAttention 검증 완료")

def test_poincare_embedding():
    """PoincareEmbedding 검증"""
    B, T, d_model, d_head = 2, 10, 64, 32
    
    embed = PoincareEmbedding(d_model, d_head, c=-1.0)
    x = torch.randn(B, T, d_model)
    z = embed(x)
    
    assert z.shape == (B, T, d_head)
    # Poincaré ball 내부 확인 (norm < 1)
    norms = torch.norm(z, dim=-1)
    assert (norms < 1.0).all()
    print("✓ PoincareEmbedding 검증 완료")

def test_metrikey():
    """metrikey.metric_from_keys 검증"""
    keys = ["topic:diagnosis", "topic:treatment"]
    dim = 32
    
    G = metrikey.metric_from_keys(
        keys, dim=dim,
        min_lambda=0.1, max_lambda=5.0,
        masses=[0.8, 0.2]
    )
    
    assert G.shape == (dim, dim)
    # SPD 확인
    eigvals = torch.linalg.eigvalsh(G)
    assert (eigvals > 0).all()
    print("✓ metrikey 검증 완료")

if __name__ == "__main__":
    test_metric_attention()
    test_poincare_embedding()
    test_metrikey()
    print("\n✓ 모든 reality_stone API 검증 완료")
```

### 2.3 디렉토리 구조

```
reality_stone/
├── python/
│   └── reality_stone/
│       ├── models/
│       │   ├── sentence_topic_head.py      # Phase 2
│       │   ├── metric_router.py            # Phase 3
│       │   └── rce_lexical_decoder.py      # Phase 4
│       └── utils/
│           ├── pre_segmenter.py            # Phase 1
│           ├── lexical_candidates.py       # Phase 1
│           └── topology_builder.py         # Phase 1
├── scripts/
│   ├── prepare_data.py                     # 데이터 전처리
│   ├── train.py                            # Phase 6
│   └── evaluate.py                         # Phase 6
├── api/
│   ├── server.py                           # Phase 5
│   └── schemas.py                          # Phase 5
├── tests/
│   ├── test_reality_stone_api.py           # Phase 0
│   ├── test_pre_segmenter.py               # Phase 1
│   ├── test_sentence_topic_head.py         # Phase 2
│   ├── test_metric_router.py               # Phase 3
│   ├── test_rce_decoder.py                 # Phase 4
│   └── test_api.py                         # Phase 5
└── docs/
    ├── llm.md                              # 기존
    ├── sentence_topic_architecture.md      # 아키텍처
    ├── sentence_topic_data_pipeline.md     # 데이터
    └── sentence_topic_implementation.md    # 본 문서
```

---

## 3. Phase 1: Pre-Segmenter 구현

### 3.1 파일: `python/reality_stone/utils/pre_segmenter.py`

```python
"""문단 분해 및 전처리 모듈"""
import torch
import kss
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
from konlpy.tag import Mecab

class PreSegmenter:
    def __init__(
        self,
        tokenizer_name: str = "klue/bert-base",
        max_length: int = 128,
        use_mecab: bool = True
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.mecab = Mecab() if use_mecab else None
    
    def __call__(self, paragraph: str) -> Dict:
        """
        문단을 문장 단위로 분해하고 전처리
        
        Args:
            paragraph: 입력 문단
        
        Returns:
            {
                "sentences": List[str],
                "tokens": torch.Tensor [num_sents, seq_len],
                "attention_mask": torch.Tensor [num_sents, seq_len],
                "replacement_mask": torch.Tensor [num_sents, seq_len],
                "topo_idx": torch.Tensor [num_sents, k],
                "metadata": Dict
            }
        """
        # 1. 문장 분해
        sentences = self._segment_sentences(paragraph)
        
        # 2. 토큰화
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 3. Replacement mask 생성
        replacement_mask = self._generate_replacement_mask(
            encoded["input_ids"],
            sentences
        )
        
        # 4. Topology index 생성
        topo_idx = self._build_topology(len(sentences), k=3)
        
        # 5. 메타데이터
        metadata = {
            "num_sentences": len(sentences),
            "sentence_lengths": [len(s.split()) for s in sentences],
            "total_tokens": encoded["input_ids"].shape[1]
        }
        
        return {
            "sentences": sentences,
            "tokens": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "replacement_mask": replacement_mask,
            "topo_idx": topo_idx,
            "metadata": metadata
        }
    
    def _segment_sentences(self, paragraph: str) -> List[str]:
        """문장 분해"""
        sentences = kss.split_sentences(paragraph)
        # 후처리: 너무 짧은 문장 병합
        merged = []
        buffer = ""
        for sent in sentences:
            if len(sent) < 10 and buffer:
                buffer += " " + sent
            else:
                if buffer:
                    merged.append(buffer)
                buffer = sent
        if buffer:
            merged.append(buffer)
        return merged
    
    def _generate_replacement_mask(
        self,
        input_ids: torch.Tensor,
        sentences: List[str]
    ) -> torch.Tensor:
        """교체 가능 토큰 마스크 생성"""
        mask = torch.zeros_like(input_ids)
        
        for i, sent in enumerate(sentences):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
            
            # 형태소 분석
            if self.mecab:
                pos_tags = self._align_pos_tags(tokens, sent)
            else:
                pos_tags = ["X"] * len(tokens)
            
            # 마스크 생성
            for j, (token, pos) in enumerate(zip(tokens, pos_tags)):
                if self._is_replaceable(token, pos):
                    mask[i, j] = 1
        
        return mask
    
    def _is_replaceable(self, token: str, pos: str) -> bool:
        """토큰 교체 가능 여부 판정"""
        # 특수 토큰 제외
        if token in ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]:
            return False
        # 조사, 어미 제외
        if pos in ["JX", "JC", "EP", "EF", "EC"]:
            return False
        # 숫자, 기호 제외
        if token.isdigit() or token in ".,!?;:":
            return False
        # 일반 명사, 동사, 형용사 허용
        if pos in ["NNG", "NNB", "VV", "VA", "MAG"]:
            return True
        return False
    
    def _align_pos_tags(self, tokens: List[str], sentence: str) -> List[str]:
        """토큰과 형태소 태그 정렬"""
        morphs = self.mecab.pos(sentence)
        pos_tags = []
        morph_idx = 0
        
        for token in tokens:
            if token.startswith("##"):  # subword
                pos_tags.append(pos_tags[-1] if pos_tags else "X")
            elif token in ["[CLS]", "[SEP]", "[PAD]"]:
                pos_tags.append("X")
            else:
                if morph_idx < len(morphs):
                    pos_tags.append(morphs[morph_idx][1])
                    morph_idx += 1
                else:
                    pos_tags.append("X")
        
        return pos_tags
    
    def _build_topology(self, num_sentences: int, k: int = 3) -> torch.Tensor:
        """시간 순서 기반 topology 생성"""
        topo = []
        for i in range(num_sentences):
            neighbors = []
            # 이전 문장
            if i > 0:
                neighbors.append(i - 1)
            # 다음 문장
            if i < num_sentences - 1:
                neighbors.append(i + 1)
            # k개 채우기
            while len(neighbors) < k:
                neighbors.append(i)
            topo.append(neighbors[:k])
        
        return torch.tensor(topo)
```

### 3.2 테스트: `tests/test_pre_segmenter.py`

```python
import pytest
from reality_stone.utils.pre_segmenter import PreSegmenter

def test_basic_segmentation():
    """기본 문장 분해 테스트"""
    segmenter = PreSegmenter()
    paragraph = "환자는 고혈압 진단을 받았다. 약물 치료를 시작했다. 경과가 양호하다."
    
    result = segmenter(paragraph)
    
    assert len(result["sentences"]) == 3
    assert result["tokens"].shape[0] == 3
    assert result["replacement_mask"].shape == result["tokens"].shape
    assert result["topo_idx"].shape == (3, 3)

def test_replacement_mask():
    """Replacement mask 정확성 테스트"""
    segmenter = PreSegmenter()
    paragraph = "환자는 고혈압 진단을 받았다."
    
    result = segmenter(paragraph)
    mask = result["replacement_mask"][0]
    tokens = segmenter.tokenizer.convert_ids_to_tokens(result["tokens"][0])
    
    # [CLS], [SEP]는 0이어야 함
    assert mask[0] == 0  # [CLS]
    assert mask[-1] == 0 or mask[-2] == 0  # [SEP]
    
    # 조사 "는", "을"은 0이어야 함
    for i, token in enumerate(tokens):
        if token in ["는", "을"]:
            assert mask[i] == 0

def test_topology_consistency():
    """Topology 일관성 테스트"""
    segmenter = PreSegmenter()
    paragraph = "문장1. 문장2. 문장3."
    
    result = segmenter(paragraph)
    topo = result["topo_idx"]
    
    # 첫 문장은 다음 문장(1)을 이웃으로 가져야 함
    assert 1 in topo[0]
    # 중간 문장은 이전(0)과 다음(2)을 이웃으로
    assert 0 in topo[1] and 2 in topo[1]
    # 마지막 문장은 이전(1)을 이웃으로
    assert 1 in topo[2]
```

---

## 4. Phase 2: SentenceTopicHead 구현

### 4.1 파일: `python/reality_stone/models/sentence_topic_head.py`

```python
"""문장 주제 분류 모듈"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from reality_stone.layers.poincare_embedding import PoincareEmbedding
from reality_stone.layers.metric_attention import MetricAttention

class SentenceTopicHead(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        d_head: int = 64,
        num_topics: int = 8,
        num_heads: int = 4,
        c_poincare: float = -1.0,
        temperature: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.num_topics = num_topics
        
        # Poincaré embedding
        self.poincare_embed = PoincareEmbedding(d_model, d_head, c=c_poincare)
        
        # Geodesic attention
        self.metric_attn = MetricAttention(
            d_head, num_heads,
            mode="geodesic",
            manifold="poincare",
            temperature=temperature
        )
        
        # 주제 분류기
        self.topic_classifier = nn.Linear(d_head, num_topics)
        
        # 주제별 앵커 (학습 가능)
        self.topic_anchors = nn.Parameter(torch.randn(num_topics, d_head) * 0.1)
        
        # 주제 이름 매핑
        self.topic_names = [
            "chief_complaint", "history", "physical_exam", "diagnosis",
            "treatment_plan", "prognosis", "follow_up", "general"
        ]
    
    def forward(
        self,
        x: torch.Tensor,
        topo_idx: torch.Tensor
    ) -> tuple:
        """
        Args:
            x: [B, T, d_model] 문장 임베딩
            topo_idx: [B, T, K] topology index
        
        Returns:
            P_topic: [B, T, num_topics] 주제 확률
            scores: [B, T] 우선순위 점수
            metric_keys: List[str] 문장별 metric key
        """
        B, T, _ = x.shape
        
        # 1. Poincaré embedding
        z = self.poincare_embed(x)  # [B, T, d_head]
        
        # 2. Geodesic attention
        attn_out, attn_weights = self.metric_attn(
            z, z, z, topo_idx=topo_idx
        )  # [B, T, d_head], [B, T, K]
        
        # 3. 우선순위 점수 (attention weight 합)
        scores = attn_weights.sum(dim=-1)  # [B, T]
        
        # 4. 주제 분류
        logits = self.topic_classifier(attn_out)  # [B, T, num_topics]
        P_topic = F.softmax(logits, dim=-1)
        
        # 5. Metric key 생성
        metric_keys = self._generate_metric_keys(P_topic, scores)
        
        return P_topic, scores, metric_keys
    
    def _generate_metric_keys(
        self,
        P_topic: torch.Tensor,
        scores: torch.Tensor
    ) -> List[str]:
        """주제 확률과 score로부터 metric key 생성"""
        B, T, _ = P_topic.shape
        keys = []
        
        for b in range(B):
            for t in range(T):
                # 최고 확률 주제
                top_topic = P_topic[b, t].argmax().item()
                topic_name = self.topic_names[top_topic]
                
                # 우선순위 레벨
                score_val = scores[b, t].item()
                if score_val > 0.7:
                    priority = "high"
                elif score_val > 0.4:
                    priority = "medium"
                else:
                    priority = "low"
                
                key = f"topic:{topic_name}|priority:{priority}"
                keys.append(key)
        
        return keys
```

### 4.2 테스트: `tests/test_sentence_topic_head.py`

```python
import pytest
import torch
from reality_stone.models.sentence_topic_head import SentenceTopicHead

def test_forward_pass():
    """Forward pass 기본 동작"""
    model = SentenceTopicHead(d_model=768, d_head=64, num_topics=8)
    
    B, T, d_model = 2, 5, 768
    x = torch.randn(B, T, d_model)
    topo_idx = torch.randint(0, T, (B, T, 3))
    
    P_topic, scores, metric_keys = model(x, topo_idx)
    
    assert P_topic.shape == (B, T, 8)
    assert scores.shape == (B, T)
    assert len(metric_keys) == B * T
    
    # 확률 합 = 1
    assert torch.allclose(P_topic.sum(dim=-1), torch.ones(B, T))

def test_metric_key_format():
    """Metric key 형식 검증"""
    model = SentenceTopicHead()
    
    x = torch.randn(1, 3, 768)
    topo_idx = torch.randint(0, 3, (1, 3, 3))
    
    _, _, metric_keys = model(x, topo_idx)
    
    for key in metric_keys:
        assert "topic:" in key
        assert "priority:" in key
        assert key.split("|")[0].startswith("topic:")
        assert key.split("|")[1].startswith("priority:")

def test_gradient_flow():
    """Gradient 흐름 확인"""
    model = SentenceTopicHead()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    x = torch.randn(1, 3, 768)
    topo_idx = torch.randint(0, 3, (1, 3, 3))
    target = torch.randint(0, 8, (1, 3))
    
    P_topic, _, _ = model(x, topo_idx)
    loss = F.cross_entropy(P_topic.view(-1, 8), target.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 파라미터가 업데이트되었는지 확인
    assert any(p.grad is not None for p in model.parameters())
```

---

## 5. Phase 3: MetricContextRouter 구현

### 5.1 파일: `python/reality_stone/models/metric_router.py`

```python
"""Metric context routing 모듈"""
import torch
from typing import List, Dict
from reality_stone import metrikey

class MetricContextRouter:
    def __init__(
        self,
        d_head: int = 64,
        lambda_min: float = 0.1,
        lambda_max: float = 5.0,
        cache_size: int = 1000
    ):
        self.d_head = d_head
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.cache = {}
        self.cache_size = cache_size
    
    def __call__(
        self,
        metric_keys: List[str],
        scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Metric key와 score로부터 SPD Cholesky factor 생성
        
        Args:
            metric_keys: [B*T] 문장별 metric key
            scores: [B, T] 우선순위 점수
        
        Returns:
            L: [B, T, d_head, d_head] Cholesky factor
        """
        B, T = scores.shape
        L_list = []
        
        scores_flat = scores.flatten()
        
        for i, key in enumerate(metric_keys):
            score_val = scores_flat[i].item()
            
            # 캐시 확인
            cache_key = (key, round(score_val, 2))
            if cache_key in self.cache:
                L_list.append(self.cache[cache_key])
                continue
            
            # SPD 메트릭 생성
            try:
                G = metrikey.metric_from_keys(
                    [key],
                    dim=self.d_head,
                    min_lambda=self.lambda_min,
                    max_lambda=self.lambda_max,
                    masses=[score_val]
                )
            except Exception as e:
                # 키가 없으면 identity 사용
                print(f"Warning: metric key '{key}' not found, using identity")
                G = torch.eye(self.d_head)
            
            # Eigenvalue 클램핑
            G = self._clamp_eigenvalues(G)
            
            # Cholesky factorization
            try:
                L = torch.linalg.cholesky(G)
            except RuntimeError:
                # 수치 불안정 시 regularization
                G = G + torch.eye(self.d_head) * 1e-6
                L = torch.linalg.cholesky(G)
            
            # 캐시 저장
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = L.clone()
            
            L_list.append(L)
        
        return torch.stack(L_list).view(B, T, self.d_head, self.d_head)
    
    def _clamp_eigenvalues(self, G: torch.Tensor) -> torch.Tensor:
        """Eigenvalue 범위 제한"""
        eigvals, eigvecs = torch.linalg.eigh(G)
        eigvals = torch.clamp(eigvals, self.lambda_min, self.lambda_max)
        return eigvecs @ torch.diag(eigvals) @ eigvecs.T
```

### 5.2 테스트: `tests/test_metric_router.py`

```python
import pytest
import torch
from reality_stone.models.metric_router import MetricContextRouter

def test_basic_routing():
    """기본 routing 동작"""
    router = MetricContextRouter(d_head=32)
    
    keys = ["topic:diagnosis|priority:high", "topic:treatment|priority:low"]
    scores = torch.tensor([[0.8, 0.3]])
    
    L = router(keys, scores)
    
    assert L.shape == (1, 2, 32, 32)
    
    # 하삼각 행렬 확인
    for i in range(2):
        assert torch.allclose(L[0, i], torch.tril(L[0, i]))

def test_cache_hit():
    """캐시 동작 확인"""
    router = MetricContextRouter(d_head=32)
    
    keys = ["topic:diagnosis|priority:high"] * 2
    scores = torch.tensor([[0.8, 0.8]])
    
    L1 = router(keys, scores)
    
    # 캐시 크기 확인
    assert len(router.cache) == 1
    
    # 동일 키로 재호출
    L2 = router(keys, scores)
    
    # 결과 동일
    assert torch.allclose(L1, L2)

def test_eigenvalue_clamping():
    """Eigenvalue 클램핑 확인"""
    router = MetricContextRouter(d_head=32, lambda_min=0.5, lambda_max=2.0)
    
    keys = ["topic:diagnosis|priority:high"]
    scores = torch.tensor([[0.9]])
    
    L = router(keys, scores)
    G = L[0, 0] @ L[0, 0].T
    
    eigvals = torch.linalg.eigvalsh(G)
    assert (eigvals >= 0.5).all()
    assert (eigvals <= 2.0).all()
```

---

## 6. Phase 4: RCE-LexicalDecoder 구현

### 6.1 파일: `python/reality_stone/models/rce_lexical_decoder.py`

```python
"""Lexical constraint 기반 디코더"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from reality_stone.models.gpt2_metric import GPT2MetricModel
from reality_stone.layers.metric_attention import MetricAttention

class RCELexicalDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 768,
        n_layer: int = 6,
        n_head: int = 8,
        manifold: str = "lorentz",
        c_lorentz: float = -1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.manifold = manifold
        self.c_lorentz = c_lorentz
        
        # Embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        
        # Transformer blocks with geodesic attention
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, manifold, c_lorentz)
            for _ in range(n_layer)
        ])
        
        # LM head
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        metric_ctx: torch.Tensor,
        replacement_mask: torch.Tensor,
        topo_idx: torch.Tensor,
        candidates: Dict[int, List[int]]
    ) -> tuple:
        """
        Args:
            input_ids: [B, T] 입력 토큰 ID
            metric_ctx: [B, T, d_h, d_h] SPD Cholesky factor
            replacement_mask: [B, T] 교체 가능 여부
            topo_idx: [B, T, K] topology index
            candidates: {token_id: [cand1, cand2, ...]}
        
        Returns:
            output_ids: [B, T] 출력 토큰 ID
            logits: [B, T, V] 제약된 logits
        """
        B, T = input_ids.shape
        
        # Embedding
        token_emb = self.token_embed(input_ids)
        pos_emb = self.pos_embed(torch.arange(T, device=input_ids.device))
        hidden = token_emb + pos_emb  # [B, T, d_model]
        
        # Transformer blocks
        for block in self.blocks:
            hidden = block(hidden, metric_ctx, topo_idx)
        
        # LM head
        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)  # [B, T, V]
        
        # Lexical constraint 적용
        constrained_logits = self._apply_lexical_constraint(
            logits, input_ids, replacement_mask, candidates
        )
        
        # Sampling
        output_ids = torch.argmax(constrained_logits, dim=-1)
        
        # replacement_mask=0 위치는 원본 유지
        output_ids = torch.where(
            replacement_mask.bool(),
            output_ids,
            input_ids
        )
        
        return output_ids, constrained_logits
    
    def _apply_lexical_constraint(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
        candidates: Dict[int, List[int]]
    ) -> torch.Tensor:
        """후보 집합 외 토큰 마스킹"""
        B, T, V = logits.shape
        constrained = logits.clone()
        
        for b in range(B):
            for t in range(T):
                if mask[b, t] == 0:
                    continue
                
                token_id = input_ids[b, t].item()
                if token_id in candidates and len(candidates[token_id]) > 0:
                    # 후보 외 토큰 -inf
                    valid_ids = candidates[token_id]
                    mask_tensor = torch.ones(V, dtype=torch.bool, device=logits.device)
                    mask_tensor[valid_ids] = False
                    constrained[b, t, mask_tensor] = float('-inf')
        
        return constrained


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, manifold, c):
        super().__init__()
        self.attn = MetricAttention(
            d_model, n_head,
            mode="geodesic",
            manifold=manifold,
            temperature=0.1
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x, metric_ctx, topo_idx):
        # Attention
        attn_out, _ = self.attn(x, x, x, topo_idx=topo_idx)
        x = x + attn_out
        x = self.ln1(x)
        
        # FFN
        x = x + self.mlp(x)
        x = self.ln2(x)
        
        return x
```

### 6.2 테스트: `tests/test_rce_decoder.py`

```python
import pytest
import torch
from reality_stone.models.rce_lexical_decoder import RCELexicalDecoder

def test_forward_pass():
    """Forward pass 기본 동작"""
    model = RCELexicalDecoder(vocab_size=1000, d_model=128, n_layer=2, n_head=4)
    
    B, T = 2, 10
    input_ids = torch.randint(0, 1000, (B, T))
    metric_ctx = torch.randn(B, T, 16, 16)
    replacement_mask = torch.randint(0, 2, (B, T))
    topo_idx = torch.randint(0, T, (B, T, 3))
    candidates = {i: [i, i+1, i+2] for i in range(1000)}
    
    output_ids, logits = model(
        input_ids, metric_ctx, replacement_mask, topo_idx, candidates
    )
    
    assert output_ids.shape == (B, T)
    assert logits.shape == (B, T, 1000)

def test_replacement_mask_preservation():
    """Replacement mask 보존 확인"""
    model = RCELexicalDecoder(vocab_size=100, d_model=64, n_layer=1, n_head=2)
    
    B, T = 1, 5
    input_ids = torch.tensor([[10, 20, 30, 40, 50]])
    metric_ctx = torch.randn(B, T, 8, 8)
    replacement_mask = torch.tensor([[1, 0, 1, 0, 1]])  # 0 위치는 고정
    topo_idx = torch.randint(0, T, (B, T, 2))
    candidates = {i: [i, i+1] for i in range(100)}
    
    output_ids, _ = model(
        input_ids, metric_ctx, replacement_mask, topo_idx, candidates
    )
    
    # mask=0 위치는 원본 유지
    assert output_ids[0, 1] == 20
    assert output_ids[0, 3] == 40

def test_lexical_constraint():
    """Lexical constraint 동작 확인"""
    model = RCELexicalDecoder(vocab_size=100, d_model=64, n_layer=1, n_head=2)
    
    B, T = 1, 3
    input_ids = torch.tensor([[10, 20, 30]])
    metric_ctx = torch.randn(B, T, 8, 8)
    replacement_mask = torch.ones(B, T)
    topo_idx = torch.randint(0, T, (B, T, 2))
    
    # 후보 제한: 10 → [10, 11], 20 → [20, 21, 22], 30 → [30]
    candidates = {
        10: [10, 11],
        20: [20, 21, 22],
        30: [30]
    }
    
    output_ids, _ = model(
        input_ids, metric_ctx, replacement_mask, topo_idx, candidates
    )
    
    # 출력이 후보 내에 있는지 확인
    assert output_ids[0, 0].item() in [10, 11]
    assert output_ids[0, 1].item() in [20, 21, 22]
    assert output_ids[0, 2].item() == 30
```

---

## 7. Phase 5: API 서버 구현

### 7.1 파일: `api/server.py`

```python
"""FastAPI 서버"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch

from reality_stone.utils.pre_segmenter import PreSegmenter
from reality_stone.models.sentence_topic_head import SentenceTopicHead
from reality_stone.models.metric_router import MetricContextRouter
from reality_stone.models.rce_lexical_decoder import RCELexicalDecoder

app = FastAPI(title="Sentence-Topic LLM API")

# 모델 로딩 (전역)
pre_segmenter = PreSegmenter()
sentence_topic_head = SentenceTopicHead()
metric_router = MetricContextRouter()
rce_decoder = RCELexicalDecoder()

# 체크포인트 로드 (학습 후)
# sentence_topic_head.load_state_dict(torch.load("checkpoints/topic_head.pt"))
# rce_decoder.load_state_dict(torch.load("checkpoints/decoder.pt"))

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
        with torch.no_grad():
            P_topic, scores, metric_keys = sentence_topic_head(
                seg_output["tokens"].unsqueeze(0).float(),  # [1, T, d]
                seg_output["topo_idx"].unsqueeze(0)
            )
        
        # metric_hint 적용
        if request.metric_hint:
            metric_keys = [request.metric_hint + "|" + k for k in metric_keys]
        
        # L2: MetricContextRouter
        L_i = metric_router(metric_keys, scores)
        
        # L3: RCE-LexicalDecoder
        candidates = _build_candidates(request.lexical_overrides)
        with torch.no_grad():
            output_ids, _ = rce_decoder(
                seg_output["tokens"].unsqueeze(0),
                L_i,
                seg_output["replacement_mask"].unsqueeze(0),
                seg_output["topo_idx"].unsqueeze(0),
                candidates
            )
        
        # L4: Post-Controller
        response = _build_response(
            seg_output, output_ids[0], P_topic[0], metric_keys
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _build_candidates(overrides: Dict[str, List[str]]) -> Dict[int, List[int]]:
    """후보 사전 구축 (간단 버전)"""
    # 실제로는 lexical_candidates.py 사용
    return {}

def _build_response(seg_output, output_ids, P_topic, metric_keys) -> RewriteResponse:
    """응답 구성"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    
    output_sentences = []
    replacements = []
    
    for i, sent_ids in enumerate(output_ids):
        decoded = tokenizer.decode(sent_ids, skip_special_tokens=True)
        output_sentences.append(decoded)
        
        # 교체 로그
        input_ids = seg_output["tokens"][i]
        for j in range(len(sent_ids)):
            if input_ids[j] != sent_ids[j]:
                replacements.append({
                    "sentence": i,
                    "pos": j,
                    "old": tokenizer.decode([input_ids[j]]),
                    "new": tokenizer.decode([sent_ids[j]])
                })
    
    # 통계
    total_tokens = seg_output["tokens"].numel()
    replaced_tokens = len(replacements)
    
    return RewriteResponse(
        sentences=output_sentences,
        topics=P_topic.tolist(),
        metric_keys=metric_keys,
        replacements=replacements,
        final_text=" ".join(output_sentences),
        stats={
            "total_tokens": total_tokens,
            "replaced_tokens": replaced_tokens,
            "replacement_ratio": replaced_tokens / total_tokens
        }
    )

@app.get("/health")
async def health():
    return {"status": "ok"}
```

### 7.2 실행

```bash
cd /e/reality_stone
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

### 7.3 Postman Collection

```json
{
  "info": {
    "name": "Sentence-Topic LLM API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "url": "http://localhost:8000/health"
      }
    },
    {
      "name": "정상 입력",
      "request": {
        "method": "POST",
        "url": "http://localhost:8000/sentence_topic_rewrite",
        "header": [{"key": "Content-Type", "value": "application/json"}],
        "body": {
          "mode": "raw",
          "raw": "{\"paragraph\": \"환자는 고혈압 진단을 받았다. 약물 치료를 시작했다.\"}"
        }
      }
    },
    {
      "name": "Lexical Override",
      "request": {
        "method": "POST",
        "url": "http://localhost:8000/sentence_topic_rewrite",
        "body": {
          "mode": "raw",
          "raw": "{\"paragraph\": \"환자는 고혈압 진단을 받았다.\", \"lexical_overrides\": {\"진단\": [\"판정\", \"확인\"]}}"
        }
      }
    },
    {
      "name": "잘못된 Metric Key",
      "request": {
        "method": "POST",
        "url": "http://localhost:8000/sentence_topic_rewrite",
        "body": {
          "mode": "raw",
          "raw": "{\"paragraph\": \"환자는 고혈압 진단을 받았다.\", \"metric_hint\": \"invalid_key_12345\"}"
        }
      }
    }
  ]
}
```

---

## 8. Phase 6: 학습 파이프라인

### 8.1 파일: `scripts/train.py`

```python
"""학습 스크립트"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from reality_stone.models.sentence_topic_head import SentenceTopicHead
from reality_stone.models.rce_lexical_decoder import RCELexicalDecoder
from reality_stone.utils.pre_segmenter import PreSegmenter

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader):
        tokens = batch["tokens"].to(device)
        topo_idx = batch["topo_idx"].to(device)
        topic_labels = batch["topic_labels"].to(device)
        
        # Forward
        P_topic, scores, _ = model(tokens, topo_idx)
        
        # Loss
        loss = nn.CrossEntropyLoss()(
            P_topic.view(-1, P_topic.size(-1)),
            topic_labels.view(-1)
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델
    model = SentenceTopicHead(
        d_model=args.d_model,
        d_head=args.d_head,
        num_topics=args.num_topics
    ).to(device)
    
    # 데이터
    # dataset = SentenceTopicDataset(args.data_path)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 학습
    for epoch in range(args.epochs):
        loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")
        
        # 체크포인트 저장
        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), f"checkpoints/topic_head_epoch{epoch+1}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_head", type=int, default=64)
    parser.add_argument("--num_topics", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=2)
    args = parser.parse_args()
    
    main(args)
```

---

이 문서는 Sentence-Topic LLM의 구현 로드맵, 환경 설정, Phase별 상세 구현 가이드, 테스트 코드, API 서버, 학습 스크립트를 포함한다.  
각 Phase는 독립적으로 테스트 가능하며, reality_stone API와의 연동 방법을 명시하여 실제 구현 시 직접 참조 가능하다.

