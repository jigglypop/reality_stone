# LLM 레이어별 압축 전략

## 1. 현재 상황 분석

### 압축된 부분
- **Linear 레이어**: RBE로 64비트 시드로 압축 (압축률 2,535,842:1)
- 실제 가중치 자체는 극도로 압축됨

### 압축되지 않은 부분 (용량을 차지하는 원인)
1. **임베딩 레이어** (wte, wpe)
2. **정규화 레이어** (LayerNorm)
3. **편향(bias) 파라미터**
4. **모델 구조 정보** (config.json)
5. **토크나이저** (vocab, merges)

## 2. 임베딩 레이어 압축

### 2.1 Token Embedding (wte)
```
현재: vocab_size × hidden_size (50257 × 768 = 38.5M 파라미터)
```

**압축 방안:**
1. **SVD 기반 분해**
   - U × Σ × V^T로 분해
   - 상위 k개 특이값만 유지
   - 압축률: ~10:1

2. **양자화 + 코드북**
   - 임베딩을 k-means 클러스터링
   - 각 토큰을 클러스터 인덱스로 표현
   - 압축률: ~100:1

3. **해시 트릭**
   - 다중 해시 함수로 임베딩 생성
   - 저장 공간: O(1)
   - 품질 손실 있음

### 2.2 Position Embedding (wpe)
```
현재: max_position × hidden_size (1024 × 768 = 0.8M 파라미터)
```

**압축 방안:**
1. **수식 기반 생성**
   - Sinusoidal encoding처럼 수식으로 생성
   - 저장 공간: 0
   
2. **학습 가능한 Fourier Features**
   - 적은 수의 파라미터로 위치 정보 인코딩
   - 압축률: ~100:1

## 3. LayerNorm 압축

### 현재 상황
- 각 레이어마다 weight, bias 저장
- GPT-2: 25개 × 768 × 2 = 38,400 파라미터

**압축 방안:**
1. **파라미터 공유**
   - 모든 레이어가 동일한 LayerNorm 파라미터 사용
   - 압축률: 25:1

2. **저차원 분해**
   - weight = base_weight + layer_specific_delta
   - delta를 저차원으로 표현

3. **제거 후 재학습**
   - LayerNorm 없이 작동하도록 fine-tuning
   - 압축률: ∞

## 4. Attention 메커니즘 압축

### Multi-Head Attention 최적화
1. **Head 수 축소**
   - 중요도가 낮은 head 제거
   - 압축률: 2-4:1

2. **Sparse Attention**
   - 전체 시퀀스 대신 일부만 attend
   - 메모리 및 계산량 감소

3. **Low-Rank Attention**
   - QKV 행렬을 low-rank로 분해
   - 압축률: 4-8:1

## 5. 편향(Bias) 압축

### 현재 상황
- 각 Linear/Conv 레이어마다 별도 저장
- 전체 모델에서 ~1-2% 차지

**압축 방안:**
1. **Bias 제거**
   - 많은 경우 bias 없이도 성능 유지
   - 압축률: ∞

2. **양자화**
   - FP32 → INT8/INT4
   - 압축률: 4-8:1

3. **클러스터링**
   - 유사한 bias 값들을 그룹화
   - 인덱스만 저장

## 6. 전체 모델 압축 파이프라인

### Phase 1: 구조적 압축
1. Head pruning
2. Layer pruning (중복성 높은 레이어 제거)
3. Width pruning (hidden dimension 축소)

### Phase 2: 파라미터 압축
1. Linear layers → RBE (완료)
2. Embeddings → SVD/Quantization
3. LayerNorm → Parameter sharing
4. Bias → Removal/Quantization

### Phase 3: 극한 압축
1. **모델 DNA 방식**
   - 전체 모델을 단일 시드로 표현
   - 생성 함수로 모델 구조와 가중치 복원
   
2. **Neural Architecture Search 역방향**
   - 모델 구조 자체를 압축된 형태로 표현
   - 메타 학습으로 복원

## 7. 예상 압축률

### 현재 (RBE만 적용)
- 원본: 489.46 MB
- 압축: 158.17 MB
- 압축률: 3.1:1

### 전체 파이프라인 적용 시
- Phase 1: 489 MB → 250 MB
- Phase 2: 250 MB → 50 MB  
- Phase 3: 50 MB → 5-10 MB

**최종 목표: 50-100:1 압축률**

## 8. 구현 우선순위

1. **임베딩 압축** (가장 큰 효과)
2. **Bias 제거/압축**
3. **LayerNorm 공유**
4. **Attention 최적화**
5. **극한 압축 연구**

## 9. 품질 보존 전략

### 압축 시 고려사항
1. **단계별 압축**
   - 한 번에 하나씩 적용
   - 각 단계마다 성능 검증

2. **Fine-tuning**
   - 압축 후 소량의 데이터로 재학습
   - 품질 손실 최소화

3. **앙상블 방식**
   - 여러 압축 모델의 앙상블
   - 개별 모델은 작지만 함께 사용 시 고품질

## 10. 하드웨어 고려사항

### 디코딩 속도 최적화
1. **GPU 친화적 구조**
   - 병렬 처리 가능한 형태로 설계
   - 메모리 접근 패턴 최적화

2. **캐싱 전략**
   - 자주 사용되는 가중치 캐싱
   - 계층적 캐시 구조

3. **양자화 하드웨어 활용**
   - INT8/INT4 연산 가속
   - 특수 하드웨어 활용 