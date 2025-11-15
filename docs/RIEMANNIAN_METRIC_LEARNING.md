# 🔥 리만 메트릭 학습 (Riemannian Metric Learning)

## 핵심 구현 확인

**Sentence-Topic LLM**의 핵심은 **SPD Metric Learning**입니다.

### ✅ 구현 완료

```
✓ Using MetricAttention with SPD Metric Learning (d_h=16, rank=2)
```

## 1. SPD Metric Parameterization

**`python/reality_stone/layers/metric_attention.py`의 `SPDMetric` 클래스**

```python
class SPDMetric(nn.Module):
    """
    SPD metric: G = diag(softplus(d)) + U U^T
    
    - Diagonal: softplus(log_diag) → SPD 보장
    - Low-rank: U ∈ R^{d×r}, r=2
    """
    def __init__(self, hidden_size: int, rank: int = 2):
        self.log_diag = nn.Parameter(torch.zeros(hidden_size))
        self.U = nn.Parameter(torch.randn(hidden_size, rank) * 1e-3)
    
    def scale_q(self, q):
        """q' = q * softplus(d)"""
        d = F.softplus(self.log_diag)
        return q * d
```

### 학습 가능한 파라미터

1. **`log_diag`**: [d_h] - 대각 성분 (SPD 보장)
2. **`U`**: [d_h, r] - Low-rank 성분 (r=2)

→ **총 `d_h * (1 + r)` = 16 * 3 = 48개 파라미터**

## 2. Geodesic Attention

**Poincaré Manifold 기반**

```python
# sentence_topic_head.py
self.metric_attn = MetricAttention(
    hidden_size=16,        # per-head dimension
    normalizer="softmax",
    rank=2,                # Low-rank SPD
    mode="geodesic",       # 🔥 Geodesic distance
    manifold="poincare",   # 🔥 Poincaré manifold
    c=1e-3                 # curvature
)
```

### Geodesic Distance (Poincaré)

```python
def poincare_distance(x, y, c):
    """
    d(x, y) = arccosh(1 + 2c * ||x - y||² / ((1 - c||x||²)(1 - c||y||²)))
    """
    norm_x = torch.sum(x * x, dim=-1, keepdim=True)
    norm_y = torch.sum(y * y, dim=-1, keepdim=True)
    norm_diff = torch.sum((x - y) ** 2, dim=-1)
    
    denom = (1 - c * norm_x) * (1 - c * norm_y)
    arg = 1 + 2 * c * norm_diff / (denom + 1e-8)
    return torch.acosh(torch.clamp(arg, min=1.0 + 1e-7))
```

### Attention Score

```python
# Geodesic distance → Attention score
dist = poincare_distance(q, k, c)  # [B, H, T, K]
scores = -dist² / τ                # 거리 → 유사도
attn_weights = softmax(scores)     # 정규화
```

## 3. Metric Key Context Switching

**`python/reality_stone/models/metric_router.py`**

```python
class MetricContextRouter:
    """
    Metric key → SPD 메트릭 합성
    
    예: "topic:diagnosis|priority:high" → G ∈ SPD(d)
    """
    def forward(self, metric_keys, scores):
        L_list = []
        for key, score in zip(metric_keys, scores):
            # metrikey.metric_from_keys() 호출
            G = self._get_metric_from_key(key, score)
            
            # Eigenvalue 클램핑 [0.8, 1.2]
            G = self._clamp_eigenvalues(G)
            
            # Cholesky: G = L L^T
            L = torch.linalg.cholesky(G)
            L_list.append(L)
        
        return torch.stack(L_list)  # [B, T, d, d]
```

### Metric Key 형식

```
"topic:{topic_name}|priority:{high/medium/low}"

예시:
- "topic:diagnosis|priority:high"
- "topic:treatment_plan|priority:medium"
- "topic:follow_up|priority:low"
```

## 4. 전체 파이프라인

```
입력 문장 임베딩 [B, T, d_model]
    ↓
[L1] SentenceTopicHead
    ├─ Poincaré Embedding (Fallback: Linear)
    ├─ 🔥 MetricAttention (SPD Metric Learning)
    │   ├─ Q, K, V projection
    │   ├─ Head split: [B, T, d] → [B, H, T, d_h]
    │   ├─ SPD scale: q' = q * softplus(d)
    │   ├─ Geodesic distance (Poincaré)
    │   └─ Attention: softmax(-dist²/τ)
    ├─ Topic classification (8 topics)
    └─ Metric key generation
    ↓
[L2] MetricContextRouter
    ├─ Metric key → SPD G
    ├─ Eigenvalue clamp [0.8, 1.2]
    └─ Cholesky: L = chol(G)
    ↓
[L3] RCE-LexicalDecoder
    └─ Lexical constraint decoding
```

## 5. 학습 목표

### Loss Functions

```python
# 1. Next-token prediction
ℒ_LM = CrossEntropy(logits, targets)

# 2. Topic classification
ℒ_topic = CrossEntropy(P_topic, topic_labels)

# 3. Metric regularization
ℒ_metric = ||G - I||_F²  # SPD 안정성

# 4. Curvature regularization
ℒ_curv = (c - c_target)²  # 곡률 제약

# Total
ℒ = ℒ_LM + λ₁ℒ_topic + λ₂ℒ_metric + λ₃ℒ_curv
```

### 학습 가능한 파라미터

1. **SPD Metric** (L1)
   - `log_diag`: [d_h] = [16]
   - `U`: [d_h, r] = [16, 2]
   - **총 48개 파라미터**

2. **Topic Classifier** (L1)
   - `topic_classifier`: [d_head, num_topics] = [64, 8]
   - **512개 파라미터**

3. **Metric Router** (L2)
   - Metric key cache (non-parametric)

4. **Decoder** (L3)
   - Transformer blocks

## 6. 실행 결과

```bash
$ .venv/Scripts/python.exe demo.py

✓ Using MetricAttention with SPD Metric Learning (d_h=16, rank=2)

테스트 1: ✓✓ 성공!
  - L0: 2개 문장 분해
  - L1: 주제 분류 (SPD Metric Learning)
  - L2: SPD 메트릭 생성
  - L3: 디코더 실행 (50.0% 토큰 변경)

테스트 2: ✓✓ 성공!
테스트 3: ✓✓ 성공!
```

## 7. 핵심 차별점

### 기존 Transformer vs. Reality Stone

| 항목 | 기존 Transformer | Reality Stone (SPD Metric) |
|------|-----------------|---------------------------|
| Attention | Dot-product | **Geodesic distance** |
| Manifold | Euclidean | **Poincaré/Lorentz** |
| Metric | Fixed (I) | **Learnable SPD (G)** |
| Context | Single | **Metric key switching** |
| Hierarchy | Flat | **Hyperbolic (계층)** |

### SPD Metric의 장점

1. **Geometry Learning**: 데이터에 맞는 기하 학습
2. **Context Switching**: 주제별 다른 geometry
3. **Hierarchical**: 쌍곡 기하로 계층 구조 표현
4. **Stable**: SPD 보장으로 수치 안정성

## 8. 다음 단계

### 학습 강화

```bash
# 데이터 준비
python scripts/prepare_data.py

# SPD Metric Learning 학습
python scripts/train.py \
  --epochs 100 \
  --lr 1e-4 \
  --lambda_metric 0.01 \
  --lambda_curv 0.001
```

### Metric Key 확장

```python
# 더 세밀한 metric key
"topic:diagnosis|priority:high|domain:cardiology|security:level3"

# 다중 context 혼합
metric_keys_a = ["topic:diagnosis|..."]
metric_keys_b = ["topic:treatment|..."]
alpha = 0.7  # 70% diagnosis, 30% treatment
```

## 📚 참고 문헌

1. **Poincaré Embeddings** (Nickel & Kiela, 2017)
2. **Hyperbolic Neural Networks** (Ganea et al., 2018)
3. **SPD Matrix Learning** (Huang & Van Gool, 2017)
4. **Riemannian Optimization** (Bonnabel, 2013)

---

**핵심: 리만 메트릭 학습이 활성화되어 작동 중!** 🔥

