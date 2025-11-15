# Sentence-Topic LLM 데이터 파이프라인 및 학습 전략

## 1. 데이터 파이프라인 개요

### 1.1 목적
- 문단 단위 입력을 문장·주제·토큰·후보 집합으로 변환하여 모델 학습/추론에 공급한다.
- 주제 라벨, metric key seed, lexical candidate를 자동/반자동으로 생성한다.
- reality_stone의 geodesic attention과 호환되는 topology index를 구성한다.

### 1.2 파이프라인 구조

```
원본 문단 코퍼스
    ↓
[단계 1] 문장 분해 및 토큰화
    ↓
[단계 2] 주제 라벨링 (자동/수동)
    ↓
[단계 3] Topology Index 생성
    ↓
[단계 4] Lexical Candidate 구축
    ↓
[단계 5] Metric Key Seed 생성
    ↓
학습/평가 데이터셋 (HDF5/Parquet)
```

---

## 2. 단계 1: 문장 분해 및 토큰화

### 2.1 문장 분해 (Sentence Segmentation)

#### 2.1.1 한국어 처리
```python
import kss

def segment_korean(paragraph: str) -> List[str]:
    """한국어 문장 분해"""
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
```

#### 2.1.2 영어 처리
```python
import nltk

def segment_english(paragraph: str) -> List[str]:
    """영어 문장 분해"""
    return nltk.sent_tokenize(paragraph)
```

#### 2.1.3 의료 도메인 특화
```python
import re

def segment_medical(paragraph: str) -> List[str]:
    """의료 보고서 문장 분해 (약어 처리)"""
    # 약어 보호: Dr., Mr., vs. 등
    protected = paragraph
    protected = re.sub(r'Dr\.', 'Dr<DOT>', protected)
    protected = re.sub(r'vs\.', 'vs<DOT>', protected)
    
    sentences = kss.split_sentences(protected)
    
    # 복원
    sentences = [s.replace('<DOT>', '.') for s in sentences]
    return sentences
```

### 2.2 토큰화 (Tokenization)

#### 2.2.1 Transformer 기반 토큰화
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

def tokenize_sentences(sentences: List[str]) -> Dict:
    """
    Returns:
        {
            "input_ids": List[List[int]],      # [num_sents, seq_len]
            "attention_mask": List[List[int]],
            "token_type_ids": List[List[int]],
            "tokens": List[List[str]]          # 디버깅용
        }
    """
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in encoded["input_ids"]]
    
    return {
        "input_ids": encoded["input_ids"].tolist(),
        "attention_mask": encoded["attention_mask"].tolist(),
        "token_type_ids": encoded.get("token_type_ids", [[0]*len(ids) for ids in encoded["input_ids"]]).tolist(),
        "tokens": tokens
    }
```

#### 2.2.2 Replacement Mask 생성
```python
def generate_replacement_mask(tokens: List[str], pos_tags: List[str]) -> List[int]:
    """
    토큰별 교체 가능 여부 판정
    Args:
        tokens: ["[CLS]", "환자", "는", "고혈압", ...]
        pos_tags: ["X", "NNG", "JX", "NNG", ...]  # 형태소 태그
    Returns:
        [0, 1, 0, 1, ...]  # 0=고정, 1=교체 가능
    """
    mask = []
    for token, pos in zip(tokens, pos_tags):
        # 특수 토큰 고정
        if token in ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]:
            mask.append(0)
        # 조사, 어미 고정
        elif pos in ["JX", "JC", "EP", "EF", "EC"]:
            mask.append(0)
        # 숫자, 기호 고정
        elif token.isdigit() or token in ".,!?;:":
            mask.append(0)
        # 고유명사 고정 (선택적)
        elif pos == "NNP":
            mask.append(0)
        # 일반 명사, 동사, 형용사 교체 가능
        elif pos in ["NNG", "NNB", "VV", "VA", "MAG"]:
            mask.append(1)
        else:
            mask.append(0)  # 기본값: 고정
    
    return mask
```

#### 2.2.3 형태소 분석 연동
```python
from konlpy.tag import Mecab

mecab = Mecab()

def get_pos_tags(sentence: str) -> List[Tuple[str, str]]:
    """형태소 분석"""
    return mecab.pos(sentence)

def align_pos_to_tokens(tokens: List[str], sentence: str) -> List[str]:
    """토큰과 형태소 태그 정렬"""
    morphs = mecab.pos(sentence)
    pos_tags = []
    
    morph_idx = 0
    for token in tokens:
        if token.startswith("##"):  # subword
            pos_tags.append(pos_tags[-1] if pos_tags else "X")
        elif token in ["[CLS]", "[SEP]", "[PAD]"]:
            pos_tags.append("X")
        else:
            # 형태소와 매칭
            if morph_idx < len(morphs):
                pos_tags.append(morphs[morph_idx][1])
                morph_idx += 1
            else:
                pos_tags.append("X")
    
    return pos_tags
```

---

## 3. 단계 2: 주제 라벨링

### 3.1 자동 라벨링 (Zero-shot Classification)

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def auto_label_topics(sentences: List[str], candidate_labels: List[str]) -> List[int]:
    """
    Args:
        sentences: 문장 리스트
        candidate_labels: ["diagnosis", "treatment", "prognosis", "general"]
    Returns:
        topic_ids: [0, 1, 2, ...] 문장별 주제 ID
    """
    topic_ids = []
    for sent in sentences:
        result = classifier(sent, candidate_labels)
        top_label = result["labels"][0]
        topic_ids.append(candidate_labels.index(top_label))
    
    return topic_ids
```

### 3.2 반자동 라벨링 (Active Learning)

```python
def active_label_topics(
    sentences: List[str],
    model_predictions: List[int],
    confidence_threshold: float = 0.8
) -> List[int]:
    """
    모델 예측 신뢰도가 낮은 샘플만 수동 라벨링
    Args:
        sentences: 문장 리스트
        model_predictions: 모델 예측 주제 ID
        confidence_threshold: 신뢰도 임계값
    Returns:
        final_labels: 수동 보정된 주제 ID
    """
    final_labels = []
    for i, (sent, pred) in enumerate(zip(sentences, model_predictions)):
        confidence = model_predictions_probs[i].max()
        
        if confidence < confidence_threshold:
            # 수동 라벨링 요청
            print(f"[{i}] {sent}")
            print(f"Model prediction: {pred} (confidence: {confidence:.2f})")
            manual_label = int(input("Enter correct label (0-3): "))
            final_labels.append(manual_label)
        else:
            final_labels.append(pred)
    
    return final_labels
```

### 3.3 의료 도메인 주제 분류

```python
MEDICAL_TOPICS = {
    0: "chief_complaint",      # 주호소
    1: "history",              # 병력
    2: "physical_exam",        # 신체 검사
    3: "diagnosis",            # 진단
    4: "treatment_plan",       # 치료 계획
    5: "prognosis",            # 예후
    6: "follow_up",            # 추적 관찰
    7: "general"               # 기타
}

def classify_medical_topic(sentence: str) -> int:
    """의료 문장 주제 분류 (규칙 기반)"""
    keywords = {
        0: ["호소", "증상", "불편", "통증"],
        1: ["병력", "과거력", "가족력", "이전"],
        2: ["검사", "청진", "촉진", "활력징후"],
        3: ["진단", "판정", "소견", "질환"],
        4: ["치료", "처방", "투약", "수술"],
        5: ["예후", "경과", "회복", "예상"],
        6: ["추적", "재진", "관찰", "모니터링"],
    }
    
    for topic_id, kws in keywords.items():
        if any(kw in sentence for kw in kws):
            return topic_id
    
    return 7  # general
```

---

## 4. 단계 3: Topology Index 생성

### 4.1 시간 순서 기반 Topology

```python
def build_temporal_topology(num_sentences: int, k: int = 2) -> torch.Tensor:
    """
    시간 순서 기반 이웃 구성
    Args:
        num_sentences: 문장 개수
        k: 이웃 개수
    Returns:
        topo_idx: [num_sentences, k] 이웃 인덱스
    """
    topo = []
    for i in range(num_sentences):
        neighbors = []
        # 이전 문장
        if i > 0:
            neighbors.append(i - 1)
        # 다음 문장
        if i < num_sentences - 1:
            neighbors.append(i + 1)
        
        # k개 채우기 (부족하면 자기 자신 추가)
        while len(neighbors) < k:
            neighbors.append(i)
        
        topo.append(neighbors[:k])
    
    return torch.tensor(topo)
```

### 4.2 주제 유사도 기반 Topology

```python
from sklearn.metrics.pairwise import cosine_similarity

def build_topic_topology(
    sentence_embeddings: np.ndarray,
    k: int = 3
) -> torch.Tensor:
    """
    주제 유사도 기반 이웃 구성
    Args:
        sentence_embeddings: [num_sentences, d_model] 문장 임베딩
        k: 이웃 개수
    Returns:
        topo_idx: [num_sentences, k]
    """
    # Cosine 유사도 계산
    sim_matrix = cosine_similarity(sentence_embeddings)
    
    # 자기 자신 제외하고 상위 k개 선택
    topo = []
    for i in range(len(sim_matrix)):
        sim_matrix[i, i] = -1  # 자기 자신 제외
        top_k_indices = np.argsort(sim_matrix[i])[-k:][::-1]
        topo.append(top_k_indices.tolist())
    
    return torch.tensor(topo)
```

### 4.3 하이브리드 Topology

```python
def build_hybrid_topology(
    num_sentences: int,
    sentence_embeddings: np.ndarray,
    k: int = 4,
    temporal_weight: float = 0.5
) -> torch.Tensor:
    """
    시간 순서 + 주제 유사도 혼합
    Args:
        temporal_weight: 시간 순서 가중치 (0~1)
    """
    # 시간 순서 이웃
    temporal_topo = build_temporal_topology(num_sentences, k)
    
    # 주제 유사도 이웃
    topic_topo = build_topic_topology(sentence_embeddings, k)
    
    # 혼합: 각각에서 k//2개씩 선택
    k_temporal = int(k * temporal_weight)
    k_topic = k - k_temporal
    
    hybrid_topo = []
    for i in range(num_sentences):
        neighbors = set()
        neighbors.update(temporal_topo[i, :k_temporal].tolist())
        neighbors.update(topic_topo[i, :k_topic].tolist())
        
        # k개로 맞추기
        neighbors = list(neighbors)[:k]
        while len(neighbors) < k:
            neighbors.append(i)
        
        hybrid_topo.append(neighbors[:k])
    
    return torch.tensor(hybrid_topo)
```

### 4.4 그래프 기반 Topology (의료 지식 그래프)

```python
import networkx as nx

def build_knowledge_graph_topology(
    sentences: List[str],
    entities: List[List[str]],  # 문장별 엔티티 리스트
    kg: nx.Graph,                # 의료 지식 그래프
    k: int = 3
) -> torch.Tensor:
    """
    지식 그래프 기반 이웃 구성
    Args:
        entities: [["고혈압", "진단"], ["약물", "치료"], ...]
        kg: 의료 지식 그래프 (엔티티 간 관계)
    """
    topo = []
    for i, ents_i in enumerate(entities):
        # 다른 문장과의 엔티티 연결 강도 계산
        scores = []
        for j, ents_j in enumerate(entities):
            if i == j:
                scores.append(-1)
                continue
            
            # 지식 그래프 상 최단 경로 길이
            min_path_len = float('inf')
            for e_i in ents_i:
                for e_j in ents_j:
                    if kg.has_node(e_i) and kg.has_node(e_j):
                        try:
                            path_len = nx.shortest_path_length(kg, e_i, e_j)
                            min_path_len = min(min_path_len, path_len)
                        except nx.NetworkXNoPath:
                            pass
            
            # 거리 역수를 score로
            score = 1.0 / (min_path_len + 1) if min_path_len < float('inf') else 0
            scores.append(score)
        
        # 상위 k개 선택
        top_k_indices = np.argsort(scores)[-k:][::-1]
        topo.append(top_k_indices.tolist())
    
    return torch.tensor(topo)
```

---

## 5. 단계 4: Lexical Candidate 구축

### 5.1 WordNet 기반 동의어

```python
from nltk.corpus import wordnet

def get_wordnet_synonyms(word: str, pos: str = None) -> List[str]:
    """
    WordNet 동의어 추출
    Args:
        word: 단어
        pos: 품사 ('n', 'v', 'a', 'r')
    Returns:
        synonyms: 동의어 리스트
    """
    synonyms = set()
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.add(lemma.name().replace('_', ' '))
    
    return list(synonyms)
```

### 5.2 한국어 동의어 사전

```python
# 의료 도메인 동의어 사전 예시
KOREAN_MEDICAL_SYNONYMS = {
    "진단": ["판정", "확인", "소견"],
    "치료": ["요법", "처치", "처방"],
    "환자": ["피험자", "대상자", "환자분"],
    "증상": ["징후", "소견", "양상"],
    "약물": ["약제", "의약품", "처방약"],
    "검사": ["진찰", "평가", "측정"],
    "수술": ["시술", "오퍼레이션", "외과적 처치"],
}

def get_korean_synonyms(word: str) -> List[str]:
    """한국어 동의어 조회"""
    return KOREAN_MEDICAL_SYNONYMS.get(word, [])
```

### 5.3 임베딩 기반 유사 단어

```python
from gensim.models import Word2Vec

# 사전 학습된 Word2Vec 모델
w2v_model = Word2Vec.load("medical_w2v.model")

def get_embedding_similar_words(
    word: str,
    topn: int = 5,
    threshold: float = 0.7
) -> List[str]:
    """
    임베딩 유사도 기반 후보 추출
    Args:
        word: 대상 단어
        topn: 상위 n개
        threshold: 유사도 임계값
    Returns:
        candidates: 후보 단어 리스트
    """
    try:
        similar = w2v_model.wv.most_similar(word, topn=topn)
        candidates = [w for w, score in similar if score >= threshold]
        return candidates
    except KeyError:
        return []
```

### 5.4 통합 후보 구축

```python
def build_lexical_candidates(
    tokens: List[str],
    pos_tags: List[str],
    use_wordnet: bool = True,
    use_embedding: bool = True,
    max_candidates: int = 10
) -> Dict[str, List[str]]:
    """
    토큰별 교체 후보 통합 구축
    Args:
        tokens: 토큰 리스트
        pos_tags: 품사 태그
        max_candidates: 후보 최대 개수
    Returns:
        candidates: {token: [cand1, cand2, ...]}
    """
    candidates = {}
    
    for token, pos in zip(tokens, pos_tags):
        # 특수 토큰 제외
        if token.startswith("[") or token in ".,!?":
            continue
        
        cands = set()
        
        # 1. 사전 기반 동의어
        if token in KOREAN_MEDICAL_SYNONYMS:
            cands.update(KOREAN_MEDICAL_SYNONYMS[token])
        
        # 2. WordNet (영어)
        if use_wordnet and pos in ["NNG", "VV", "VA"]:
            wordnet_pos = {"NNG": "n", "VV": "v", "VA": "a"}.get(pos)
            cands.update(get_wordnet_synonyms(token, wordnet_pos))
        
        # 3. 임베딩 유사도
        if use_embedding:
            cands.update(get_embedding_similar_words(token))
        
        # 4. 원본 토큰 포함 (교체 안 할 수도 있음)
        cands.add(token)
        
        # 최대 개수 제한
        candidates[token] = list(cands)[:max_candidates]
    
    return candidates
```

### 5.5 Geodesic Score 기반 필터링

```python
def filter_candidates_by_geodesic(
    token: str,
    candidates: List[str],
    embeddings: Dict[str, np.ndarray],
    manifold: str = "poincare",
    c: float = -1.0,
    threshold: float = 2.0
) -> List[str]:
    """
    reality_stone geodesic 거리로 후보 필터링
    Args:
        token: 원본 토큰
        candidates: 후보 리스트
        embeddings: 토큰별 임베딩 {token: embedding}
        threshold: 거리 임계값 (이하만 유지)
    """
    from reality_stone.layers.poincare import poincare_distance
    
    if token not in embeddings:
        return candidates
    
    token_emb = torch.tensor(embeddings[token])
    filtered = []
    
    for cand in candidates:
        if cand not in embeddings:
            continue
        
        cand_emb = torch.tensor(embeddings[cand])
        dist = poincare_distance(token_emb, cand_emb, c=c)
        
        if dist <= threshold:
            filtered.append(cand)
    
    return filtered if filtered else [token]  # 최소한 원본은 유지
```

---

## 6. 단계 5: Metric Key Seed 생성

### 6.1 주제 기반 Key

```python
def generate_topic_metric_key(topic_id: int, priority_score: float) -> str:
    """
    주제 ID와 우선순위로 metric key 생성
    Args:
        topic_id: 0~7 (의료 주제)
        priority_score: 0~1 (문장 중요도)
    Returns:
        key: "topic:diagnosis|priority:high"
    """
    topic_names = [
        "chief_complaint", "history", "physical_exam", "diagnosis",
        "treatment_plan", "prognosis", "follow_up", "general"
    ]
    
    topic_name = topic_names[topic_id]
    priority = "high" if priority_score > 0.7 else "medium" if priority_score > 0.4 else "low"
    
    return f"topic:{topic_name}|priority:{priority}"
```

### 6.2 스타일 기반 Key

```python
def detect_style(sentence: str) -> str:
    """문장 스타일 감지"""
    formal_keywords = ["진단", "처방", "소견", "판정"]
    informal_keywords = ["아프다", "괜찮다", "좀", "막"]
    
    if any(kw in sentence for kw in formal_keywords):
        return "formal"
    elif any(kw in sentence for kw in informal_keywords):
        return "informal"
    else:
        return "neutral"

def generate_style_metric_key(sentence: str) -> str:
    """스타일 기반 metric key"""
    style = detect_style(sentence)
    return f"style:{style}"
```

### 6.3 통합 Metric Key

```python
def generate_full_metric_key(
    topic_id: int,
    priority_score: float,
    sentence: str,
    security_level: str = "standard"
) -> str:
    """
    통합 metric key 생성
    Args:
        security_level: "public", "standard", "confidential"
    Returns:
        key: "topic:diagnosis|priority:high|style:formal|security:standard"
    """
    topic_key = generate_topic_metric_key(topic_id, priority_score)
    style_key = generate_style_metric_key(sentence)
    
    return f"{topic_key}|{style_key}|security:{security_level}"
```

---

## 7. 데이터셋 저장 및 로딩

### 7.1 HDF5 형식 저장

```python
import h5py

def save_dataset_hdf5(
    output_path: str,
    paragraphs: List[str],
    sentences_list: List[List[str]],
    tokens_list: List[torch.Tensor],
    topo_idx_list: List[torch.Tensor],
    replacement_masks: List[torch.Tensor],
    topic_labels: List[List[int]],
    metric_keys: List[List[str]],
    candidates: List[Dict[str, List[str]]]
):
    """학습 데이터셋을 HDF5로 저장"""
    with h5py.File(output_path, 'w') as f:
        # 메타데이터
        f.attrs['num_paragraphs'] = len(paragraphs)
        f.attrs['created_at'] = str(datetime.now())
        
        # 각 문단별 그룹
        for i, para in enumerate(paragraphs):
            grp = f.create_group(f'paragraph_{i}')
            grp.attrs['text'] = para
            grp.attrs['num_sentences'] = len(sentences_list[i])
            
            # 문장
            grp.create_dataset('sentences', data=np.array(sentences_list[i], dtype=h5py.string_dtype()))
            
            # 토큰
            grp.create_dataset('tokens', data=tokens_list[i].numpy())
            
            # Topology
            grp.create_dataset('topo_idx', data=topo_idx_list[i].numpy())
            
            # Replacement mask
            grp.create_dataset('replacement_mask', data=replacement_masks[i].numpy())
            
            # 주제 라벨
            grp.create_dataset('topic_labels', data=np.array(topic_labels[i]))
            
            # Metric keys
            grp.create_dataset('metric_keys', data=np.array(metric_keys[i], dtype=h5py.string_dtype()))
            
            # Candidates (JSON 문자열로 저장)
            import json
            grp.attrs['candidates'] = json.dumps(candidates[i])
```

### 7.2 PyTorch Dataset 클래스

```python
from torch.utils.data import Dataset

class SentenceTopicDataset(Dataset):
    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        with h5py.File(hdf5_path, 'r') as f:
            self.num_paragraphs = f.attrs['num_paragraphs']
    
    def __len__(self):
        return self.num_paragraphs
    
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as f:
            grp = f[f'paragraph_{idx}']
            
            return {
                'paragraph': grp.attrs['text'],
                'sentences': grp['sentences'][:].tolist(),
                'tokens': torch.tensor(grp['tokens'][:]),
                'topo_idx': torch.tensor(grp['topo_idx'][:]),
                'replacement_mask': torch.tensor(grp['replacement_mask'][:]),
                'topic_labels': torch.tensor(grp['topic_labels'][:]),
                'metric_keys': grp['metric_keys'][:].tolist(),
                'candidates': json.loads(grp.attrs['candidates'])
            }
```

---

## 8. 학습 전략

### 8.1 손실 함수 구성

```python
def compute_total_loss(
    outputs: Dict,
    targets: Dict,
    lambda_topic: float = 1.0,
    lambda_order: float = 0.5,
    lambda_lex: float = 2.0,
    lambda_metric: float = 0.1,
    lambda_c: float = 0.1
) -> torch.Tensor:
    """
    통합 손실 함수
    Args:
        outputs: 모델 출력 {P_topic, scores, output_ids, logits, G_metrics}
        targets: 정답 {topic_labels, order_labels, input_ids, replacement_mask}
    """
    # 1. 주제 분류 손실
    loss_topic = F.cross_entropy(
        outputs['P_topic'].view(-1, outputs['P_topic'].size(-1)),
        targets['topic_labels'].view(-1)
    )
    
    # 2. 우선순위 순서 손실 (pairwise ranking)
    scores = outputs['scores']
    order_labels = targets['order_labels']
    loss_order = 0
    for i in range(len(scores) - 1):
        for j in range(i + 1, len(scores)):
            if order_labels[i] < order_labels[j]:  # i가 j보다 우선
                loss_order += F.relu(scores[j] - scores[i] + 0.1)
    loss_order /= (len(scores) * (len(scores) - 1) / 2)
    
    # 3. Lexical 제약 손실
    output_ids = outputs['output_ids']
    input_ids = targets['input_ids']
    replacement_mask = targets['replacement_mask']
    
    # 교체 가능 위치에서만 KL divergence
    logits = outputs['logits']
    target_dist = F.one_hot(input_ids, num_classes=logits.size(-1)).float()
    loss_lex = F.kl_div(
        F.log_softmax(logits, dim=-1),
        target_dist,
        reduction='none'
    ).sum(dim=-1)
    loss_lex = (loss_lex * replacement_mask).sum() / replacement_mask.sum()
    
    # 4. 메트릭 안정성 손실
    G_metrics = outputs['G_metrics']  # [B, T, d_h, d_h]
    eigvals = torch.linalg.eigvalsh(G_metrics)
    loss_metric = F.relu(0.1 - eigvals.min()) + F.relu(eigvals.max() - 5.0)
    
    # 5. 곡률 범위 손실
    c_values = outputs['curvatures']
    loss_c = F.relu(-10.0 - c_values).mean() + F.relu(c_values - (-0.1)).mean()
    
    # 통합
    total_loss = (
        lambda_topic * loss_topic +
        lambda_order * loss_order +
        lambda_lex * loss_lex +
        lambda_metric * loss_metric +
        lambda_c * loss_c
    )
    
    return total_loss, {
        'loss_topic': loss_topic.item(),
        'loss_order': loss_order.item(),
        'loss_lex': loss_lex.item(),
        'loss_metric': loss_metric.item(),
        'loss_c': loss_c.item()
    }
```

### 8.2 학습 루프

```python
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = 'cuda'
) -> Dict[str, float]:
    """1 epoch 학습"""
    model.train()
    total_losses = defaultdict(float)
    
    for batch in tqdm(dataloader):
        # 데이터 이동
        tokens = batch['tokens'].to(device)
        topo_idx = batch['topo_idx'].to(device)
        replacement_mask = batch['replacement_mask'].to(device)
        topic_labels = batch['topic_labels'].to(device)
        
        # Forward
        outputs = model(
            tokens,
            topo_idx=topo_idx,
            replacement_mask=replacement_mask,
            candidates=batch['candidates']
        )
        
        # Loss
        loss, loss_dict = compute_total_loss(
            outputs,
            {'topic_labels': topic_labels, 'input_ids': tokens, 'replacement_mask': replacement_mask}
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 로깅
        for k, v in loss_dict.items():
            total_losses[k] += v
    
    # 평균
    for k in total_losses:
        total_losses[k] /= len(dataloader)
    
    return total_losses
```

### 8.3 평가 지표

```python
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """모델 평가"""
    model.eval()
    
    total_topic_acc = 0
    total_replacement_ratio = 0
    total_bleu = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            tokens = batch['tokens'].to(device)
            topo_idx = batch['topo_idx'].to(device)
            replacement_mask = batch['replacement_mask'].to(device)
            topic_labels = batch['topic_labels'].to(device)
            
            # Forward
            outputs = model(tokens, topo_idx, replacement_mask, batch['candidates'])
            
            # 주제 정확도
            P_topic = outputs['P_topic']
            topic_pred = P_topic.argmax(dim=-1)
            topic_acc = (topic_pred == topic_labels).float().mean()
            total_topic_acc += topic_acc.item()
            
            # 교체 비율
            output_ids = outputs['output_ids']
            changed = (output_ids != tokens).float()
            replacement_ratio = (changed * replacement_mask).sum() / replacement_mask.sum()
            total_replacement_ratio += replacement_ratio.item()
            
            # BLEU (문장 단위)
            from nltk.translate.bleu_score import sentence_bleu
            for i in range(len(batch['sentences'])):
                ref = batch['sentences'][i]
                hyp = tokenizer.decode(output_ids[i])
                bleu = sentence_bleu([ref.split()], hyp.split())
                total_bleu += bleu
            
            num_samples += len(batch['sentences'])
    
    return {
        'topic_accuracy': total_topic_acc / len(dataloader),
        'replacement_ratio': total_replacement_ratio / len(dataloader),
        'bleu': total_bleu / num_samples
    }
```

---

## 9. 데이터 증강 (Augmentation)

### 9.1 Back-translation

```python
def augment_by_backtranslation(sentence: str) -> str:
    """역번역 증강"""
    # 한국어 → 영어 → 한국어
    from transformers import MarianMTModel, MarianTokenizer
    
    # Ko → En
    ko_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
    ko_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
    en_text = ko_en_model.generate(**ko_en_tokenizer(sentence, return_tensors="pt"))
    en_text = ko_en_tokenizer.decode(en_text[0], skip_special_tokens=True)
    
    # En → Ko
    en_ko_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ko")
    en_ko_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ko")
    ko_text = en_ko_model.generate(**en_ko_tokenizer(en_text, return_tensors="pt"))
    ko_text = en_ko_tokenizer.decode(ko_text[0], skip_special_tokens=True)
    
    return ko_text
```

### 9.2 동의어 교체

```python
def augment_by_synonym_replacement(
    sentence: str,
    candidates: Dict[str, List[str]],
    num_replacements: int = 2
) -> str:
    """동의어 교체 증강"""
    tokens = sentence.split()
    replaceable = [i for i, token in enumerate(tokens) if token in candidates]
    
    if len(replaceable) == 0:
        return sentence
    
    # 랜덤하게 num_replacements개 교체
    import random
    replace_indices = random.sample(replaceable, min(num_replacements, len(replaceable)))
    
    for idx in replace_indices:
        token = tokens[idx]
        new_token = random.choice(candidates[token])
        tokens[idx] = new_token
    
    return " ".join(tokens)
```

---

이 문서는 Sentence-Topic LLM의 데이터 파이프라인, 주제 라벨링, topology 생성, lexical candidate 구축, metric key 생성, 학습 전략을 상세히 설명한다.  
각 단계별 코드 예시와 reality_stone 연동 방법을 포함하여 실제 구현 시 직접 활용 가능하다.

